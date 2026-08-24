"""Fit helpers and worker thread for the analysis pipeline.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds satlas2 models from saved config dicts, loads and bins CLS data,
and runs least-squares / emcee fits, either one file per subprocess
(parallel) or all files together in a single simultaneous fitter. Also
handles voltage-merged spectra, virtual splits, per-file GP centroid
corrections, scan filtering, burn-in/thinning, and diagnostic outputs
(walk plots, correlation plots, confidence bands, chi-square maps).
``_build_models_on_source`` and ``_fit_single_run`` are module-level
functions (not class methods) so they remain picklable for
``ProcessPoolExecutor``.

Depends on: gui.analysis.naming, gui.analysis.expr_validation,
gui.analysis.binning, gui.analysis.merge, gui.scan_filter; uses the
satlas2 fitting library and the clstools data loader.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtCore import QThread, Signal

from gui.analysis.naming import (
    NON_FIT_PARAMS, is_fit_param_name, safe_model_name,
    source_name_for_path, source_name_for_merged,
    source_name_for_descriptor, full_param_name,
)
from gui.analysis.expr_validation import lower_model_expression

# clstools drags in pandas + scipy (~2 s of app startup); probe for it
# on first use (this module is on the startup import path).
_CLSTOOLS_OK = None


def _has_clstools():
    global _CLSTOOLS_OK
    if _CLSTOOLS_OK is None:
        try:
            import clstools  # noqa: F401
            _CLSTOOLS_OK = True
        except ImportError:
            _CLSTOOLS_OK = False
    return _CLSTOOLS_OK


def _compute_tof_hist(data, pmt_gate):
    """ToF histogram (1 µs bins) of the PMT-gated events in a CLSDataFrame.

    Returns ``{"centers": [...], "counts": [...]}`` (bin centres + counts)
    or ``None`` when no TOF/TDC data is available. Uses fixed 1 µs bins
    spanning the observed TOF range so the output ToF plot shows the
    beam's time structure.
    """
    try:
        import pandas as _pd
        df = getattr(data, "Sorted", None)
        if not isinstance(df, _pd.DataFrame):
            df = data.Run.compute()
        if "TOF" not in df.columns:
            return None
        tof = df["TOF"].values.astype(float)
        if pmt_gate is not None and "TDC" in df.columns:
            tof = tof[df["TDC"].isin(list(pmt_gate)).values]
        if len(tof) == 0:
            return None
        binsize = 1.0
        bins = np.arange(tof.min() - 0.5 * binsize,
                         tof.max() + 0.5 * binsize, binsize)
        if len(bins) < 2:
            return None
        counts, edges = np.histogram(tof, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {"centers": centers.tolist(),
                "counts": counts.astype(float).tolist()}
    except Exception:
        return None


def _apply_emcee_burnin(fitter, chain_file, burn, thin):
    """Recompute emcee parameter values / uncertainties from the saved
    chain AFTER discarding the first ``burn`` steps and keeping every
    ``thin``-th step.

    satlas2.Fitter.fit() doesn't forward burn/thin to lmfit's emcee, so
    its result reflects the FULL chain (initial transient included).
    This slices the chain on our side and rebuilds each varying
    parameter's value + stderr using satlas2's own convention (median,
    half the 15.87/84.13 percentile spread), so the fit report, CSV and
    fit-quality summary reflect the burned-in posterior.

    Returns the integrated autocorrelation time per varying parameter
    (or None), useful for choosing burn-in ~ 2-3x tau. No-op (returns
    None) when burn<=0 and thin<=1, or if the chain can't be read.
    """
    burn = int(burn or 0)
    thin = int(thin or 1)
    if burn <= 0 and thin <= 1:
        return None
    if not chain_file or not os.path.isfile(chain_file):
        return None
    try:
        import h5py
        with h5py.File(chain_file, "r") as hf:
            g = hf["mcmc"]
            chain = g["chain"][:]               # (steps, walkers, ndim)
            labels = [l.decode() if isinstance(l, bytes) else str(l)
                      for l in g.attrs["labels"]]
    except Exception as exc:
        print(f"[FitWorker] burn-in chain read failed: {exc}", flush=True)
        return None

    nsteps = chain.shape[0]
    if burn >= nsteps:
        # Asking to discard everything: clamp to leave a few samples so
        # the percentiles are still defined.
        burn = max(0, nsteps - 2)
    sliced = chain[burn::max(1, thin), :, :]    # (steps', walkers, ndim)
    if sliced.shape[0] < 2:
        return None

    params = fitter.result.params
    # Only the columns whose label is a VARYING param map to the chain's
    # ndim. satlas2 stores the chain over varying params in label order.
    var_idx = [i for i, lab in enumerate(labels)
               if lab in params and params[lab].vary]
    if not var_idx:
        return None
    var_labels = [labels[i] for i in var_idx]
    flat = sliced[:, :, var_idx].reshape((-1, len(var_idx)))
    q = np.percentile(flat, [15.87, 50, 84.13], axis=0)
    for j, lab in enumerate(var_labels):
        std_l, median, std_u = q[:, j]
        params[lab].value = float(median)
        params[lab].stderr = float(0.5 * (std_u - std_l))
    try:
        params.update_constraints()
    except Exception:
        pass

    # Autocorrelation time (per varying param) for the fit report.
    acor = None
    try:
        import emcee as _emcee
        acor = _emcee.autocorr.integrated_time(
            sliced[:, :, var_idx], tol=0)
        acor = {var_labels[j].split("___")[-1]: float(acor[j])
                for j in range(len(var_labels))}
    except Exception:
        acor = None
    print(f"[FitWorker] burn-in applied: discarded {burn} steps, "
          f"thin {thin} -> {sliced.shape[0]} steps", flush=True)
    return acor


def _smooth_grid_with_gaps(x_data, n_points=500, gap_factor=5.0):
    """Build a dense ``x_smooth`` grid that omits ranges the scan
    didn't cover, so the plotted model curve doesn't draw a smooth
    line across voltage regions where there's no data.

    Returns ``(x_smooth, gap_mask)`` where ``gap_mask`` is a boolean
    array marking grid points that fall inside a detected gap. Callers
    set ``y_fit_smooth[gap_mask] = np.nan`` (and the same for any
    component-curve arrays) so matplotlib breaks the line at the gap.

    A "gap" is a span between consecutive sorted data points wider than
    ``gap_factor`` × the median scan step. The grid itself is uniform
    so all stored arrays remain rectangular for serialization.
    """
    x_arr = np.asarray(x_data, dtype=float)
    x_smooth = np.linspace(x_arr.min(), x_arr.max(), n_points)
    gap_mask = np.zeros_like(x_smooth, dtype=bool)
    if x_arr.size < 2:
        return x_smooth, gap_mask
    x_sorted = np.sort(x_arr)
    dx = np.diff(x_sorted)
    pos = dx[dx > 0]
    if pos.size == 0:
        return x_smooth, gap_mask
    step = float(np.median(pos))
    threshold = gap_factor * step
    half = 0.5 * step
    for i in np.where(dx > threshold)[0]:
        lo = x_sorted[i] + half
        hi = x_sorted[i + 1] - half
        if hi > lo:
            gap_mask |= (x_smooth > lo) & (x_smooth < hi)
    return x_smooth, gap_mask


def _shift_binning_info_x(info, corr_mhz):
    """Update x-axis summary fields in a binning_info dict so they
    describe the post-correction axis. ``compute_binned`` populates
    ``x_min`` / ``x_max`` / ``x_mean`` etc. before the GP correction
    is applied; without this they'd report the pre-shift range and
    mislead anyone reading the binning summary or the run_summary
    CSV. Counts and bin widths are translation-invariant so they
    don't need updating.
    """
    if not info or not corr_mhz:
        return
    for k in ("x_min", "x_max", "x_mean"):
        if k in info and info[k] is not None:
            try:
                info[k] = float(info[k]) - float(corr_mhz)
            except (TypeError, ValueError):
                pass


def _resolve_fit_yerr(yerr, y, use_callable_yerr):
    """Resolve the per-bin uncertainty array passed to ``satlas2.Source``.

    yerr_mode "None" makes binning return ``yerr=None`` (a deliberate
    no-error-bar choice for the Pre-Analysis display). satlas2 divides by
    yerr internally, so handing it ``None`` raises a TypeError and crashes
    the fit. The "None" mode is documented as a plain, unit-weight least
    squares, so substitute ``ones_like(y)`` to deliver exactly that.
    Model-based yerr (a callable, ``use_callable_yerr=True``) is left as-is.
    """
    if yerr is None and not use_callable_yerr:
        return np.ones_like(y)
    return yerr


def _genuine_chisq(y, y_fit, yerr, n_data, n_vary):
    """Return ``(chisqr, redchi)`` from the fit residuals.

    A real goodness-of-fit even when the fit MINIMISED a likelihood objective:
    under ``llh`` satlas2 reports the optimized -logL surrogate in
    ``chisqr``/``redchi`` (not a chi-square), which would mislead anyone reading
    "Reduced Chi-sq". This recomputes the chi-square from
    ``((y - f)/yerr)**2`` -- matching the value the default chi-square fit
    already reports -- so the displayed goodness-of-fit is meaningful for both
    objectives (code review 2026-06-02, llh-redchi-mislabeled).
    """
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    r = (y - y_fit) / yerr
    chisqr = float(np.sum(r * r))
    dof = max(1, int(n_data or len(y)) - int(n_vary or 0))
    return chisqr, chisqr / dof


def _extract_correction(centroid_correction):
    """Decode the optional per-file centroid-correction dict into
    ``(correction_mhz, sigma_mhz)`` floats. Returns ``(0.0, 0.0)`` for
    None / empty / "Off"-mode entries so callers can apply the result
    unconditionally.
    """
    if not centroid_correction:
        return 0.0, 0.0
    mode = centroid_correction.get("mode", "Auto")
    if mode == "Off":
        return 0.0, 0.0
    try:
        mhz = float(centroid_correction.get("correction_mhz", 0.0))
        sig = float(centroid_correction.get("sigma_mhz", 0.0))
    except (TypeError, ValueError):
        return 0.0, 0.0
    return mhz, sig


def _merged_run_metadata(merged_data):
    """Build a run_metadata dict from a merged spectrum's per-run table.

    ``ts_start`` is the earliest start, ``ts_stop`` the latest stop;
    ``date`` is taken from the first per-run entry. The other fields
    (cooler_v, laser_set, ...) are zeroed because a merged spectrum
    has no single representative value -- the SourceBlock's per-source
    overrides are still honored separately during fitting.
    """
    pr = (merged_data.get("per_run") or []) if merged_data else []
    starts = [float(p.get("ts_start", 0) or 0) for p in pr]
    stops = [float(p.get("ts_stop", 0) or 0) for p in pr]
    starts_pos = [s for s in starts if s > 0]
    stops_pos = [s for s in stops if s > 0]
    # Aggregate the per-file GP corrections that were applied at
    # merge time. compute_merged_spectrum stores the (μ, σ) it
    # actually subtracted on each per_run entry. We surface a
    # mean(μ) + RMS(σ) audit line AND, where present, a
    # `correction_constituents` list of per-file (t, σ, mode) so
    # Phase 5 propagation can rebuild the proper variance instead
    # of treating the merged result as one pseudo-run.
    applied_mhz = [float(p.get("centroid_correction_mhz", 0.0) or 0.0)
                   for p in pr]
    applied_sig = [
        float(p.get("centroid_correction_sigma_mhz", 0.0) or 0.0)
        for p in pr]
    constituents = []
    for p, mhz, sig in zip(pr, applied_mhz, applied_sig):
        ts = float(p.get("ts_start", 0) or 0)
        mode = p.get("centroid_correction_mode")
        # Constituent is "applied" iff mode is Auto/Manual. Gating
        # on sig > 0 would silently drop a Manual entry typed with
        # zero uncertainty (a confident user value) and mismatch
        # the single-run path which gates on the mode/applied flag.
        if mode not in ("Auto", "Manual") or ts <= 0:
            continue
        constituents.append({
            "ts_start": ts,
            "centroid_correction_mhz": mhz,
            "centroid_correction_sigma_mhz": sig,
            "centroid_correction_mode": mode,
            "n_events": float(p.get("n_events", 0) or 0),
        })
    has_corrections = bool(constituents)
    if has_corrections:
        # Weight the audit aggregates by the same shares the
        # downstream IS-tab propagation uses (n_events fractions,
        # equal fallback) so the displayed mean μ / RMS σ match
        # the effective correction the merged-fit centroid
        # actually carries. Without weighting, an uneven merge can
        # show a mean μ that disagrees with the propagated value.
        c_mhz = np.array(
            [c["centroid_correction_mhz"] for c in constituents],
            dtype=float)
        c_sig = np.array(
            [c["centroid_correction_sigma_mhz"] for c in constituents],
            dtype=float)
        c_n = np.array(
            [c.get("n_events", 0.0) for c in constituents],
            dtype=float)
        if float(np.sum(c_n)) > 0:
            shares = c_n / float(np.sum(c_n))
        else:
            shares = np.full(len(constituents),
                              1.0 / len(constituents))
        merged_corr_mhz = float(np.sum(shares * c_mhz))
        # The audit σ is the propagated standard deviation of
        # μ_merged = Σ shares · μ_i under the assumption of
        # uncorrelated constituents (a conservative report; the
        # IS-tab propagation can subtract the GP cross-term where
        # it knows the constituents are correlated). Σ shares² σ_i²
        # in quadrature.
        merged_corr_sigma = float(np.sqrt(
            np.sum((shares * c_sig) ** 2)))
    else:
        merged_corr_mhz = 0.0
        merged_corr_sigma = 0.0
    out = {
        "cooler_v": 0,
        "laser_set": 0,
        "daq_time": 0,
        "n_events": 0,
        "ts_start": float(min(starts_pos)) if starts_pos else 0.0,
        "ts_stop": float(max(stops_pos)) if stops_pos else 0.0,
        "date": str(pr[0].get("date", "")) if pr else "",
        "centroid_correction_mhz": merged_corr_mhz,
        "centroid_correction_sigma_mhz": merged_corr_sigma,
        "centroid_correction_applied": has_corrections,
        # Mode is "merged" so downstream readers know to expand the
        # constituents list rather than treat this as a single Auto
        # or Manual pseudo-run.
        "centroid_correction_mode": "merged" if has_corrections else None,
    }
    if has_corrections:
        out["correction_constituents"] = constituents
    return out


def _patch_satlas2_emcee_numpy2():
    """Make satlas2's emcee path work under numpy 2.

    lmfit's _lnprob returns a 1-element ndarray when the user function
    returns a scalar with ``float_behavior='posterior'``. satlas2's
    ``SATLASSampler.compute_log_prob`` then does ``float(l)`` on each
    walker's result, which raised only a deprecation warning under
    numpy 1.x but is a hard ``TypeError`` under numpy 2.x (only 0-d
    arrays convert).

    Replace that conversion with a shape-safe one. Idempotent; called
    after each ``import satlas2`` so subprocess workers pick it up too.
    """
    import numpy as _np
    import satlas2.overwrite as _ov

    if getattr(_ov.SATLASSampler.compute_log_prob,
               "_numpy2_patched", False):
        return

    _ndarray_to_list = _ov.ndarray_to_list_of_dicts

    def compute_log_prob(self, coords):
        p = coords
        if _np.any(_np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if _np.any(_np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")
        if self.params_are_named:
            p = _ndarray_to_list(p, self.parameter_names)
        if self.vectorize:
            results = self.log_prob_fn(p)
        else:
            if self.pool is not None:
                map_func = self.pool.map
            else:
                map_func = map
            results = list(map_func(self.log_prob_fn, p))
        log_prob = _np.array(
            [_np.asarray(r, dtype=float).ravel()[0] for r in results],
            dtype=float,
        )
        if _np.any(_np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")
        return log_prob, None

    compute_log_prob._numpy2_patched = True
    _ov.SATLASSampler.compute_log_prob = compute_log_prob


# ══════════════════════════════════════════════════════════════════
#  Shared model-building helper (top-level for picklability)
# ══════════════════════════════════════════════════════════════════

def _build_models_on_source(source, model_configs, sat_module):
    """Build all models from *model_configs* on a satlas2 Source.

    Parameters
    ----------
    source : satlas2.Source
        The source object to add models to.
    model_configs : list[dict]
        Each dict carries ``type``, ``name``, ``params``, and optional keys
        such as ``racah``, ``peak_shape``, ``sidepeak_n``, etc.
    sat_module : module
        The ``satlas2`` module reference (passed explicitly so the caller
        can use a locally-imported copy, keeping the function picklable).

    Returns
    -------
    dict[str, object]
        Mapping of model name -> model object for every model that supports
        FWHM / position calculation (HFS, Voigt, Skewed Voigt, Exponential
        Decay).  Polynomial models are excluded.
    """
    model_names_map = {}

    for mc in model_configs:
        mt = mc["type"]
        params = mc["params"]
        mc_safe_name = safe_model_name(mc["name"])

        if mt == "HFS":
            hfs_kwargs = dict(
                I=params["I"]["value"],
                J=[params["Jl"]["value"], params["Ju"]["value"]],
                A=[params["Al"]["value"], params["Au"]["value"]],
                B=[params["Bl"]["value"], params["Bu"]["value"]],
                C=[params.get("Cl", {}).get("value", 0),
                   params.get("Cu", {}).get("value", 0)],
                df=params["centroid"]["value"],
                scale=params["scale"]["value"],
                fwhmg=params["FWHMG"]["value"],
                fwhml=params["FWHML"]["value"],
                racah=mc.get("racah", True),
                name=mc_safe_name,
                prefunc=None,
            )
            # satlas2 0.1.10 supports Voigt only for HFS; the
            # peak_shape config field is preserved on disk but not
            # forwarded to the constructor.
            if mc.get("sidepeak_n", 0) > 0:
                hfs_kwargs["N"] = mc["sidepeak_n"]
                hfs_kwargs["offset"] = mc["sidepeak_offset"]
                hfs_kwargs["poisson"] = mc.get("sidepeak_poisson", 0.0)
            hfs = sat_module.HFS(**hfs_kwargs)
            # Apply vary/min/max
            for pname in ["Al", "Au", "Bl", "Bu", "Cl", "Cu",
                          "centroid", "scale", "FWHMG", "FWHML"]:
                if pname in params:
                    p = params[pname]
                    hfs.params[pname].vary = p.get("vary", True)
                    if "min" in p and p["min"] is not None:
                        hfs.params[pname].min = p["min"]
                    if "max" in p and p["max"] is not None:
                        hfs.params[pname].max = p["max"]

            # Apply peak amplitude overrides
            if not mc.get("racah", True):
                peak_amps = mc.get("peak_amplitudes", {})
                for line in hfs.lines:
                    amp_key = f"Amp{line}"
                    if line in peak_amps:
                        hfs.params[amp_key].value = peak_amps[line]["value"]
                        hfs.params[amp_key].vary = peak_amps[line].get(
                            "vary", True)
                    else:
                        hfs.params[amp_key].vary = False

            source.addModel(hfs)
            model_names_map[mc_safe_name] = hfs

            # Background polynomial
            bkg_val = params.get("Bkg_p0", {}).get("value", 0)
            bkg_name = f"{mc_safe_name}_bkg"
            bkg = sat_module.Polynomial([bkg_val], name=bkg_name)
            bkg.params["p0"].vary = params.get("Bkg_p0", {}).get(
                "vary", True)
            source.addModel(bkg)
            model_names_map[bkg_name] = bkg

        elif mt == "Voigt":
            voigt = sat_module.Voigt(
                params["A"]["value"], params["mu"]["value"],
                params["FWHMG"]["value"], params["FWHML"]["value"],
                name=mc_safe_name, prefunc=None,
            )
            for pname in ["A", "mu", "FWHMG", "FWHML"]:
                if pname in params:
                    p = params[pname]
                    voigt.params[pname].vary = p.get("vary", True)
                    if "min" in p and p["min"] is not None:
                        voigt.params[pname].min = p["min"]
                    if "max" in p and p["max"] is not None:
                        voigt.params[pname].max = p["max"]
            source.addModel(voigt)
            model_names_map[mc_safe_name] = voigt

            bkg_val = params.get("Bkg_p0", {}).get("value", 0)
            bkg = sat_module.Polynomial(
                [bkg_val], name=f"{mc_safe_name}_bkg")
            bkg.params["p0"].vary = params.get("Bkg_p0", {}).get("vary", True)
            source.addModel(bkg)

        elif mt == "Skewed Voigt":
            sv = sat_module.SkewedVoigt(
                params["A"]["value"], params["mu"]["value"],
                params["FWHMG"]["value"], params["FWHML"]["value"],
                params.get("Skew", {}).get("value", 0.0),
                name=mc_safe_name, prefunc=None,
            )
            for pname in ["A", "mu", "FWHMG", "FWHML", "Skew"]:
                if pname in params:
                    p = params[pname]
                    sv.params[pname].vary = p.get("vary", True)
                    if "min" in p and p["min"] is not None:
                        sv.params[pname].min = p["min"]
                    if "max" in p and p["max"] is not None:
                        sv.params[pname].max = p["max"]
            source.addModel(sv)
            model_names_map[mc_safe_name] = sv

            bkg_val = params.get("Bkg_p0", {}).get("value", 0)
            bkg = sat_module.Polynomial(
                [bkg_val], name=f"{mc_safe_name}_bkg")
            bkg.params["p0"].vary = params.get("Bkg_p0", {}).get("vary", True)
            source.addModel(bkg)

        elif mt == "Exponential Decay":
            ed = sat_module.ExponentialDecay(
                params["amplitude"]["value"],
                params["halflife"]["value"],
                name=mc_safe_name, prefunc=None,
            )
            for pname in ["amplitude", "halflife"]:
                if pname in params:
                    p = params[pname]
                    ed.params[pname].vary = p.get("vary", True)
                    if "min" in p and p["min"] is not None:
                        ed.params[pname].min = p["min"]
                    if "max" in p and p["max"] is not None:
                        ed.params[pname].max = p["max"]
            source.addModel(ed)
            model_names_map[mc_safe_name] = ed

        elif mt == "Piecewise Constant":
            values = []
            bounds = []
            for k in sorted(params.keys()):
                if k.startswith("value"):
                    values.append(params[k]["value"])
                elif k.startswith("bound"):
                    bounds.append(params[k]["value"])
            pc = sat_module.PiecewiseConstant(values, bounds, name=mc_safe_name)
            for i, k in enumerate(sorted(
                    k2 for k2 in params if k2.startswith("value"))):
                pc.params[f"value{i}"].vary = params[k].get("vary", True)
            source.addModel(pc)

        elif mt == "Polynomial":
            # Config keys: p0 (constant), p1 (linear), p2 (quadratic), ...
            # satlas2 Polynomial([c0, c1, ...]) reverses: p(N-1)=c0, ...
            # So pass coefficients in REVERSE sorted order so that
            # poly.params["pN"] gets config["pN"]'s value.
            sorted_keys = sorted(params.keys())  # [p0, p1, ...]
            coeffs = [params[k]["value"] for k in reversed(sorted_keys)]
            poly = sat_module.Polynomial(coeffs, name=mc_safe_name)
            for k in sorted_keys:
                if k in poly.params:
                    poly.params[k].vary = params[k].get("vary", True)
            source.addModel(poly)

    return model_names_map


# ══════════════════════════════════════════════════════════════════
#  Fit Worker (runs in subprocess for parallel fitting)
# ══════════════════════════════════════════════════════════════════

def _fit_single_run(run_file_path, source_config, model_configs, fitter_config,
                     output_config=None, merged_data=None,
                     split_descriptor=None,
                     binning_override=None, centroid_correction=None,
                     scan_filter=None, calibrations=None,
                     progress_queue=None):
    """Pure function for fitting a single run in a subprocess.
    All inputs/outputs must be picklable (plain dicts/lists/arrays).

    ``binning_override`` is a sparse dict of binning-only overrides
    (see binning.BINNING_OVERRIDE_KEYS); empty / None means inherit.

    ``calibrations`` is a snapshot of the whole calibration registry
    (``path -> spec``), passed in the same way and for the same reason as
    ``scan_filter``: this runs in a subprocess that cannot reach the
    singleton. The whole map travels, not just this run's entry, because
    a ``borrow`` spec has to resolve its donor. ``None`` means every run
    uses its file's own calibration.

    ``split_descriptor`` (virtual-split feature) is the per-file
    descriptor for a ``.vasdf`` entry: ``{parent_path, source_id,
    split_lo, split_hi, metadata_override}``. When non-None, the
    worker loads ``parent_path`` instead of ``run_file_path``, names
    the source ``Run_<source_id>`` so two splits of the same parent
    don't collide, composes the V-gate via
    ``intersect_v_gate(source_v_gate, [split_lo, split_hi])``, and
    applies the metadata overrides (cooler_v / laser_sp) with
    precedence ``SourceBlock global override > per-split override >
    ASDF metadata``.

    ``centroid_correction`` (Phase D) is the GP-derived per-file
    frequency-axis correction in MHz: a dict
    ``{correction_mhz, sigma_mhz, mode}`` or None. When non-None and
    ``correction_mhz != 0``, an additional ``Shift_Ref(ref=...)`` is
    applied after the per-source ``ref_shift`` so the spectrum is
    centered on the GP-corrected reference. ``sigma_mhz`` is recorded
    on ``run_metadata`` for downstream uncertainty propagation
    (Phase 5) but does NOT change the fit's residuals here.
    """
    import sys, os
    # Subprocesses spawned from a windowed Qt app (pythonw.exe) inherit
    # sys.stdout/stderr = None, which crashes tqdm's progress bar inside
    # satlas2's emcee. Redirect to devnull so tqdm has a writable file.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    import satlas2 as sat
    _patch_satlas2_emcee_numpy2()
    import numpy as np
    import traceback
    from gui.analysis.binning import (
        compute_binned, summarize_x, effective_binning_config,
        intersect_v_gate)
    from gui.analysis.pipeline import prepare_run_data
    if output_config is None:
        output_config = {}
    binning_override = binning_override or {}

    binning_info = None
    tof_hist = None   # set in the standard ASDF branch; None for merged

    try:
        # 1. Load & bin data (or use pre-merged spectrum)
        if merged_data is not None:
            # Phase 4: voltage-merged spectrum fitted in frequency
            # mode -- project bin centers onto rest-frame MHz using
            # the user-picked merge_metadata. Counts/yerr unchanged.
            extra_warnings = []
            if (merged_data.get("x_unit") == "V"
                    and source_config.get("bin_mode") == "Frequency"):
                from gui.analysis.merge import (
                    project_voltage_merge_to_frequency)
                x_proj, y_proj, yerr_proj, w_proj = (
                    project_voltage_merge_to_frequency(
                        merged_data, source_config))
                x = np.array(x_proj)
                y = np.array(y_proj)
                yerr = np.array(yerr_proj)
                extra_warnings = w_proj
                effective_x_unit = "MHz"
            else:
                # Use pre-merged spectrum directly
                x = np.array(merged_data["x"])
                y = np.array(merged_data["y"])
                yerr = np.array(merged_data["yerr"])
                effective_x_unit = merged_data.get("x_unit", "MHz")
            # Phase 5 follow-up: compute_merged_spectrum may have
            # written voltage-mode "MANUAL_OFFSET_IGNORED_…" warnings
            # to ``merge_warnings`` on the merged_data. Forward them
            # through the same channel as the V→F projection
            # warnings so the fit-report header surfaces both.
            for w in merged_data.get("merge_warnings", []) or []:
                extra_warnings.append(dict(w))
            source_name = source_name_for_merged(merged_data)
            run_num = source_name[len("Run_"):]
            use_callable_yerr = False
            xerr = None
            _msummary = summarize_x(x)
            binning_info = {
                "source": "merged",
                "merged_name": merged_data.get("merged_name"),
                "bin_mode": ("Frequency"
                             if effective_x_unit == "MHz"
                             else "Raw Voltage"),
                "bin_definition": "Pre-merged",
                "fallback_used": False,
                "requested_bin_count": None,
                "requested_bin_width_mhz": None,
                "effective_n_bins": _msummary["n_bins"],
                "effective_bin_width_mhz": (
                    _msummary["dx_median"]
                    if effective_x_unit == "MHz" else None),
                "effective_bin_width_v": (
                    _msummary["dx_median"]
                    if effective_x_unit != "MHz" else None),
                "voltage_to_frequency_warnings": extra_warnings,
                "override_active": False,
                "override_keys": [],
                **_msummary,
            }
        else:
            # Standard ASDF loading pipeline. For a virtual-split entry
            # (.vasdf), the actual ASDF events live in `parent_path`;
            # `run_file_path` is the descriptor and remains the entry's
            # identity in result tables and keyed maps.
            # Standard ASDF loading pipeline. For a virtual-split entry
            # (.vasdf), the actual ASDF events live in `parent_path`;
            # `run_file_path` is the descriptor and remains the entry's
            # identity in result tables and keyed maps.
            is_split = split_descriptor is not None
            data_path = (split_descriptor["parent_path"]
                         if is_split else run_file_path)

            # Load + interpret + Doppler-prepare via the shared helper so this
            # path, the simultaneous fit, the auto-fitter, merge, preview, and
            # diagnostics cannot drift (ref_shift composition, the cooler-
            # override Vcool fix, split V-gate intersection). run_metadata is
            # captured before overrides; the centroid-correction fields are
            # added below, once we know the bin_mode allowed the post-binning
            # shift to land (a Raw Voltage fit must not be reported as
            # corrected -- its x-axis is in volts, the GP works in MHz).
            data, eff_cfg, run_metadata = prepare_run_data(
                run_file_path, source_config,
                binning_override=binning_override,
                split_descriptor=split_descriptor,
                calibrations=calibrations)
            bin_mode = eff_cfg.get("bin_mode", "Frequency")

            # Phase D: apply the per-file GP correction post-binning
            # in MHz directly. Why not Shift_Ref like ref_shift above?
            # clstools.Shift_Ref does ``self.Reference = ref``
            # (overwrite, not compose) -- chaining it on top of an
            # earlier Shift_Ref would silently wipe ref_shift's effect,
            # and Shift_Ref takes Hz while the panel produces MHz. The
            # binned x-axis is already in MHz, so a direct subtraction
            # is unambiguous and avoids touching clstools' state.
            _corr_mhz, _corr_sigma = _extract_correction(
                centroid_correction)
            _corr_mode = (centroid_correction.get("mode")
                           if isinstance(centroid_correction, dict)
                           else None)
            # Per-file scan filter (drops user-excluded scans before
            # binning). Wrapped in a context manager so data.Run is
            # restored to its full state on exit; no other downstream
            # code in this function reads data after compute_binned,
            # but the restoration keeps the function safe to refactor.
            from gui.scan_filter import filter_data_for_binning
            with filter_data_for_binning(
                    data, scan_filter, asdf_path=data_path):
                res = compute_binned(data, eff_cfg)
                # ToF histogram of the PMT-gated events (1 µs bins),
                # captured here so the output ToF plot can show the
                # beam's time structure + the gate that was applied.
                tof_hist = _compute_tof_hist(
                    data, eff_cfg.get("pmt_gate"))
            # The correction lands on the spectrum only in Frequency
            # mode AND when the panel produced a non-Off entry for
            # this file. Anything else -- Raw Voltage, mode="Off",
            # or no entry at all -- means the fit's centroid is in
            # the un-corrected frame and downstream propagation
            # must NOT include this run.
            _correction_applied = (
                bin_mode != "Raw Voltage"
                and _corr_mode in ("Auto", "Manual"))
            if _correction_applied and _corr_mhz != 0.0:
                res["x"] = res["x"] - _corr_mhz
                _shift_binning_info_x(res.get("info"), _corr_mhz)
            run_metadata["centroid_correction_applied"] = (
                bool(_correction_applied))
            run_metadata["centroid_correction_mode"] = (
                _corr_mode if _correction_applied else None)
            run_metadata["centroid_correction_mhz"] = (
                float(_corr_mhz) if _correction_applied else 0.0)
            run_metadata["centroid_correction_sigma_mhz"] = (
                float(_corr_sigma) if _correction_applied else 0.0)
            x, y = res["x"], res["y"]
            yerr = res["yerr"]
            xerr = res["xerr"]
            use_callable_yerr = res["use_callable_yerr"]
            binning_info = res["info"]
            binning_info["override_active"] = bool(binning_override)
            binning_info["override_keys"] = sorted(binning_override.keys())
            # Which voltage calibration produced this spectrum's x-axis.
            # Recorded unconditionally -- a result whose calibration is
            # unstated is a result that can't be defended later.
            binning_info["calibration"] = run_metadata.get("calibration")
            if binning_info.get("fallback_used"):
                base_fb = os.path.basename(run_file_path)
                print(f"[FitWorker] Binning fallback used for {base_fb}: "
                      f"{binning_info.get('n_bins')} bins "
                      f"(clstools default failed)", flush=True)

            if is_split:
                source_name = source_name_for_descriptor({
                    "kind": "virtual_split",
                    "source_id": split_descriptor["source_id"],
                })
            else:
                source_name = source_name_for_path(run_file_path)
            run_num = source_name[len("Run_"):]

        # yerr_mode "None" -> yerr is None; satlas2 divides by yerr, so
        # substitute unit weights (documented plain-least-squares case).
        yerr = _resolve_fit_yerr(yerr, y, use_callable_yerr)
        fitter = sat.Fitter()
        source = sat.Source(x, y, yerr=yerr, xerr=xerr,
                            name=source_name)
        model_names_map = _build_models_on_source(source, model_configs, sat)
        fitter.addSource(source)

        # 3. Apply expressions from model configs.
        # Skip non-fit params (HFS metadata I/Jl/Ju, Piecewise bound*).
        # Model expressions reference bare param names (Al, Au, ...)
        # and are lowered to full lmfit names per source before being
        # handed to lmfit.
        for mc in model_configs:
            mt = mc.get("type", "")
            local_params = {p for p in mc["params"]
                            if is_fit_param_name(mt, p)}
            for pname, pdata in mc["params"].items():
                if not is_fit_param_name(mt, pname):
                    continue
                expr = pdata.get("expr", "")
                if not expr.strip():
                    continue
                lowered = lower_model_expression(
                    expr, source_name, mc["name"], local_params)
                fitter.setExpr(
                    full_param_name(source_name, mc["name"], pname),
                    lowered)

        # 3b. Apply Gaussian priors that target THIS run. Priors are full
        # names (Run_<id>___Model___param); in separate mode each run is fit on
        # its own, so only priors whose run-part matches this source apply.
        # Previously priors were applied ONLY in the simultaneous path, so a
        # penalised fit ran unpenalised with no warning in Separate mode (the
        # default) -- code review 2026-06-02, priors-separate-mode-ignored.
        for prior in fitter_config.get("priors", []):
            parts = str(prior.get("param", "")).split("___")
            if len(parts) == 3 and parts[0] == source_name:
                fitter.setParamPrior(parts[0], parts[1], parts[2],
                                     prior["value"], prior["uncertainty"])

        # 4. Fit
        fit_kwargs = {"method": fitter_config["method"],
                      "scale_covar": fitter_config.get("scale_covar", True)}
        if fitter_config.get("llh"):
            fit_kwargs["llh"] = True
            fit_kwargs["llh_method"] = fitter_config["llh_method"]
            fit_kwargs["scale_covar"] = False
        if fitter_config["method"] == "emcee":
            fit_kwargs["nwalkers"] = fitter_config.get("nwalkers", 50)
            fit_kwargs["steps"] = fitter_config.get("steps", 1000)
            fit_kwargs["convergence"] = fitter_config.get("convergence", False)
            # emcee requires a filename for the HDF5 chain backend
            import tempfile
            chain_file = os.path.join(
                tempfile.gettempdir(),
                f"emcee_run_{run_num}_{os.getpid()}.h5")
            fit_kwargs["filename"] = chain_file

        # satlas2's emcee builds bounds as np.array([]) when no params vary,
        # which crashes lmfit's _lnprob with a cryptic IndexError on
        # bounds[:, 1]. Surface a clear message instead.
        fitter._prepareFit()
        n_vary = sum(1 for p in fitter.lmpars.values()
                     if p.vary and p.expr is None)
        if n_vary == 0:
            raise ValueError(
                "No varying parameters: every parameter is fixed "
                "(vary=False) or constrained by an expression. Enable "
                "at least one parameter to fit.")

        # Per-iteration progress reporting. lmfit's `iter_cb` fires once
        # per minimizer step (or per emcee sample); we throttle and push
        # a (filepath, fraction) tuple onto the cross-process queue so
        # the FitWorker can blend it into the aggregate progress bar.
        # For least-squares we estimate progress vs the default
        # max_nfev=200*(N_vary+1) cap and clamp to 0.95 so the bar
        # doesn't claim 100% before the fit finishes; for emcee we know
        # the exact total via `steps`.
        if progress_queue is not None:
            import time as _time
            method = fitter_config["method"]
            if method == "emcee":
                _max_iter = max(1, int(fitter_config.get("steps", 1000)))
                _cap = 1.0
            else:
                _max_iter = max(50, 200 * (n_vary + 1))
                _cap = 0.95
            _last_emit = [0.0]

            def _iter_cb(_params, iteration, _resid, *_a, **_kw):
                now = _time.monotonic()
                if now - _last_emit[0] < 0.2:
                    return
                _last_emit[0] = now
                frac = min(_cap, float(iteration) / _max_iter)
                try:
                    progress_queue.put_nowait((run_file_path, frac))
                except Exception:
                    pass
            fit_kwargs["iter_cb"] = _iter_cb

        fitter.fit(**fit_kwargs)
        # Mark this run as done so the poller doesn't sit at 95%.
        if progress_queue is not None:
            try:
                progress_queue.put_nowait((run_file_path, 1.0))
            except Exception:
                pass

        # 5. Collect results
        # emcee MinimizerResult may lack 'message' -- set a fallback
        if not hasattr(fitter.result, 'message'):
            fitter.result.message = "Fit completed"

        # Burn-in / thinning: recompute params from the trimmed chain
        # BEFORE the report and result dataframe are built, so they
        # reflect the burned-in posterior.
        emcee_acor = None
        if fitter_config["method"] == "emcee":
            emcee_acor = _apply_emcee_burnin(
                fitter, fit_kwargs.get("filename"),
                fitter_config.get("burn", 0),
                fitter_config.get("thin", 1))

        report = fitter.reportFit(
            show_correl=fitter_config.get("show_correl", True),
            min_correl=fitter_config.get("min_correl", 0.1))
        if emcee_acor:
            _acor_lines = "\n".join(
                f"    {k}: {v:.1f}" for k, v in emcee_acor.items())
            report = (
                report
                + "\n\n[[Burn-in]]\n"
                + f"    Discarded first {int(fitter_config.get('burn', 0))} "
                + f"steps; thin = {int(fitter_config.get('thin', 1))}.\n"
                + "    Integrated autocorrelation time (steps):\n"
                + _acor_lines)
        params_df = fitter.createResultDataframe()
        metadata_df = fitter.createMetadataDataframe()

        # Fit quality summary: chi-square, reduced chi-square, ndata,
        # nvarys for the fit-report quality block.
        fit_quality = {
            "chisqr": float(fitter.chisqr) if hasattr(fitter, 'chisqr') else None,
            "redchi": float(fitter.redchi) if hasattr(fitter, 'redchi') else None,
            "ndata": int(fitter.ndata) if hasattr(fitter, 'ndata') else None,
            "nvarys": int(fitter.nvarys) if hasattr(fitter, 'nvarys') else None,
        }
        # Under an LLH objective satlas2's chisqr/redchi are the optimized
        # -logL surrogate, not a chi-square. Replace the reported goodness-of-
        # fit (dict + metadata_df columns) with a genuine reduced chi-square
        # from the residuals so "Reduced Chi-sq" stays meaningful; the default
        # chi-square fit is unaffected (code review 2026-06-02,
        # llh-redchi-mislabeled).
        if fitter_config.get("llh"):
            _yq = source.evaluate(x)
            _eq = (np.sqrt(np.maximum(_yq, 1.0))
                   if use_callable_yerr else yerr)
            _cx, _rc = _genuine_chisq(y, _yq, _eq,
                                      fit_quality["ndata"],
                                      fit_quality["nvarys"])
            fit_quality["chisqr"] = _cx
            fit_quality["redchi"] = _rc
            fit_quality["gof_from_residuals"] = True
            for _col, _val in (("Chisquare", _cx), ("Redchi", _rc)):
                if _col in metadata_df.columns:
                    metadata_df[_col] = _val

        x_smooth, _smooth_gap_mask = _smooth_grid_with_gaps(x)
        y_fit = source.evaluate(x)
        y_fit_smooth = source.evaluate(x_smooth)
        if _smooth_gap_mask.any():
            y_fit_smooth = y_fit_smooth.astype(float, copy=True)
            y_fit_smooth[_smooth_gap_mask] = np.nan

        # Per-model component curves for overlay plotting
        component_curves = {}
        for mname, model in model_names_map.items():
            try:
                cv = np.asarray(model.f(x_smooth), dtype=float)
                if _smooth_gap_mask.any():
                    cv = cv.copy()
                    cv[_smooth_gap_mask] = np.nan
                component_curves[mname] = cv.tolist()
            except Exception:
                pass
        if use_callable_yerr:
            yerr_arr = np.sqrt(np.maximum(y_fit, 1.0))
            residuals = (y - y_fit) / yerr_arr
            yerr = yerr_arr  # store array for serialization
        else:
            residuals = (y - y_fit) / yerr

        # 5b. Calculate FWHM and peak positions for applicable models
        model_fwhm = {}
        model_positions = {}
        for mname, model in model_names_map.items():
            try:
                fwhm_val, fwhm_unc = model.calculateFWHM()
                model_fwhm[mname] = {"value": float(fwhm_val),
                                      "uncertainty": float(fwhm_unc)}
            except Exception:
                pass
            try:
                positions = model.pos()
                model_positions[mname] = [float(p) for p in positions]
            except Exception:
                pass

        # 6. Collect diagnostic raw data (plot generation in _save_results)
        #    Keep data as numpy arrays – they pickle much more efficiently
        #    than nested Python lists for cross-process transport.
        diagnostics = {}
        is_emcee = fitter_config["method"] == "emcee"
        chain_file = fit_kwargs.get("filename")

        if is_emcee and chain_file and os.path.isfile(chain_file):
            print(f"[FitWorker] Reading chain file for diagnostics: "
                  f"{os.path.basename(chain_file)}", flush=True)
            try:
                import h5py
                with h5py.File(chain_file, "r") as hf:
                    g = hf["mcmc"]
                    chain = g["chain"][:]       # (steps, walkers, ndim)
                    labels = list(g.attrs["labels"])
                labels = [l.decode() if isinstance(l, bytes) else str(l)
                          for l in labels]
                var_mask = [i for i, l in enumerate(labels)
                            if fitter.result.params[l].vary]
                var_labels = [labels[i].split("___")[-1] for i in var_mask]
                var_chain = chain[:, :, var_mask]

                # Filter by user-selected diagnostic params
                diag_sel = output_config.get("diag_params", [])
                if diag_sel:
                    sel = [j for j, vl in enumerate(var_labels)
                           if vl in diag_sel]
                    if sel:
                        var_labels = [var_labels[j] for j in sel]
                        var_chain = var_chain[:, :, sel]

                if output_config.get("walk_plot"):
                    diagnostics["walk_data"] = {
                        "labels": var_labels,
                        "chain": var_chain,
                    }
                if output_config.get("correl_plot"):
                    flat = var_chain.reshape((-1, len(var_labels)))
                    diagnostics["correl_data"] = {
                        "labels": var_labels,
                        "flatchain": flat,
                    }
                print(f"[FitWorker] Chain diagnostics collected OK "
                      f"({var_chain.shape})", flush=True)
            except Exception as exc:
                print(f"[FitWorker] Chain read error: {exc}", flush=True)

        # MCMC confidence bands via thinned evaluateOverWalk
        if (is_emcee and chain_file and os.path.isfile(chain_file)
                and output_config.get("confidence_bands")):
            print("[FitWorker] Computing confidence bands (thinned)...",
                  flush=True)
            try:
                # Thin the chain so evaluateOverWalk evaluates ~200 samples
                # instead of walkers×steps (e.g. 50,000).
                import h5py, tempfile
                with h5py.File(chain_file, "r") as hf:
                    g = hf["mcmc"]
                    full_chain = g["chain"][:]     # (steps, walkers, ndim)
                    labels_raw = list(g.attrs["labels"])
                nsteps, nwalkers, ndim = full_chain.shape
                target_samples = output_config.get("band_samples", 200)
                thin = max(1, (nsteps * nwalkers) // target_samples)
                thinned = full_chain[::thin, :, :]  # thin along steps
                # Write thinned chain to a temp file for evaluateOverWalk
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".h5", delete=False)
                tmp_path = tmp.name
                tmp.close()
                try:
                    with h5py.File(tmp_path, "w") as hf:
                        g = hf.create_group("mcmc")
                        g.create_dataset("chain", data=thinned)
                        g.attrs["labels"] = labels_raw
                        # emcee HDFBackend also needs these attrs
                        g.attrs["has_blobs"] = False
                        g.attrs["nwalkers"] = nwalkers
                        g.attrs["ndim"] = ndim
                        g.attrs["iteration"] = thinned.shape[0]
                        g.create_dataset(
                            "log_prob",
                            data=np.zeros((thinned.shape[0], nwalkers)))
                    x_bands, bands_list = fitter.evaluateOverWalk(
                        tmp_path, burnin=0, x=x_smooth)
                    if bands_list:
                        bands = bands_list[0]
                        lo_b = np.asarray(bands[0], dtype=float)
                        md_b = np.asarray(bands[1], dtype=float)
                        hi_b = np.asarray(bands[2], dtype=float)
                        if _smooth_gap_mask.any():
                            lo_b = lo_b.copy(); lo_b[_smooth_gap_mask] = np.nan
                            md_b = md_b.copy(); md_b[_smooth_gap_mask] = np.nan
                            hi_b = hi_b.copy(); hi_b[_smooth_gap_mask] = np.nan
                        diagnostics["confidence_bands"] = {
                            "x": x_smooth,
                            "lo_1sigma": lo_b,
                            "median": md_b,
                            "hi_1sigma": hi_b,
                        }
                    print(f"[FitWorker] Confidence bands OK "
                          f"(thinned {nsteps}→{thinned.shape[0]} steps)",
                          flush=True)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            except Exception as exc:
                print(f"[FitWorker] Confidence bands error: {exc}",
                      flush=True)

        # Chi-square map scan (needs fitter object, so must run here)
        # Note: satlas2 deletes fitter.temp_y after fit(), so we must
        # re-set it before calling resid().
        if output_config.get("chisq_map"):
            try:
                fitter.temp_y = fitter.y()
                fitter.setParameters(fitter.result.params)

                vary_params = [(name, p) for name, p in
                               fitter.result.params.items() if p.vary]
                # Filter by user-selected diagnostic params
                diag_sel = output_config.get("diag_params", [])
                if diag_sel:
                    vary_params = [(n, p) for n, p in vary_params
                                   if n.split("___")[-1] in diag_sel]
                chisq_scans = []
                n_pts = 30
                for pname, par in vary_params:
                    best = par.value
                    stderr = par.stderr if par.stderr else (
                        abs(best) * 0.1 or 1.0)
                    scan = np.linspace(best - 3 * stderr,
                                       best + 3 * stderr, n_pts)
                    chisq_vals = []
                    orig_val = par.value
                    for sv in scan:
                        par.value = sv
                        fitter.setParameters(fitter.result.params)
                        r = fitter.resid()
                        chisq_vals.append(float(np.sum(r ** 2)))
                    par.value = orig_val
                    fitter.setParameters(fitter.result.params)
                    chisq_scans.append({
                        "label": pname.split("___")[-1],
                        "scan": scan.tolist(),
                        "chisq": chisq_vals,
                        "best": float(best),
                    })
                if chisq_scans:
                    diagnostics["chisq_data"] = chisq_scans
                del fitter.temp_y
            except Exception as exc:
                diagnostics["chisq_error"] = f"{type(exc).__name__}: {exc}"

        return {
            "success": True,
            "run_number": run_num,
            "run_file": run_file_path,
            "source_name": source_name,
            "report": report,
            "params_df": params_df.to_dict("list"),
            "metadata_df": metadata_df.to_dict("list"),
            "x": x.tolist(),
            "y": y.tolist(),
            "yerr": yerr.tolist(),
            "y_fit": y_fit.tolist(),
            "x_smooth": x_smooth.tolist(),
            "y_fit_smooth": y_fit_smooth.tolist(),
            "component_curves": component_curves,
            "residuals": residuals.tolist(),
            "diagnostics": diagnostics,
            "fwhm": model_fwhm,
            "peak_positions": model_positions,
            "chain_file": fit_kwargs.get("filename"),
            "fit_quality": fit_quality,
            "run_metadata": (run_metadata if not merged_data
                             else _merged_run_metadata(merged_data)),
            "harmonic": source_config.get("harmonic", 0),
            "binning_info": binning_info,
            "tof_hist": tof_hist,
            "tof_gate": list(eff_cfg.get("tof_gate"))
                        if eff_cfg.get("tof_gate") else None,
        }

    except Exception as e:
        return {
            "success": False,
            "run_number": run_file_path,
            "run_file": run_file_path,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ══════════════════════════════════════════════════════════════════
#  Fit Orchestrator Thread
# ══════════════════════════════════════════════════════════════════

class FitWorkerThread(QThread):
    """Orchestrates fitting: parallel for separate, single for simultaneous."""
    progress = Signal(int, int, str)   # (completed, total, current_run)
    fit_done = Signal(dict)            # single run result
    all_done = Signal(list)            # all results
    error = Signal(str)                # fatal error

    def __init__(self, run_files, source_config, model_configs, fitter_config,
                 output_config, n_cores=1, merged_data_map=None,
                 split_descriptors_map=None,
                 file_overrides_map=None, centroid_corrections_map=None,
                 scan_filters_map=None, calibrations_map=None,
                 parent=None):
        super().__init__(parent)
        self.run_files = run_files
        self.source_config = source_config
        self.model_configs = model_configs
        self.fitter_config = fitter_config
        self.output_config = output_config
        self.n_cores = max(1, n_cores)
        self.merged_data_map = merged_data_map or {}
        # Per-file virtual-split descriptor: {parent_path, source_id,
        # split_lo, split_hi, metadata_override}. Drives where the
        # worker reads ASDF events from and how it composes the V-gate.
        self.split_descriptors_map = split_descriptors_map or {}
        self.file_overrides_map = file_overrides_map or {}
        # GP-derived per-file frequency-axis corrections (Phase D).
        # Keyed by file path; value = {correction_mhz, sigma_mhz, mode}.
        # An empty dict means no correction is applied; spectra are
        # fitted in their un-shifted frame.
        self.centroid_corrections_map = centroid_corrections_map or {}
        # Per-file scan exclusion sets, keyed by ASDF path. Snapshot
        # in the main thread before submitting to subprocesses, since
        # the registry singleton lives in the parent process.
        self.scan_filters_map = scan_filters_map or {}
        # Per-file voltage-calibration specs, keyed by ASDF path.
        # Snapshotted in the main thread for the same reason as the scan
        # filters above. Passed to the worker whole rather than per-file:
        # a "borrow" spec has to resolve its donor run, which may not be
        # among the files being fitted.
        self.calibrations_map = calibrations_map or {}
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        # For simultaneous mode: forcefully terminate the thread if
        # it's stuck in a blocking fitter.fit() call.
        # We give it 2 seconds to finish gracefully, then terminate.
        if not self.fitter_config.get("separate", True):
            import threading
            def _force_terminate():
                import time
                time.sleep(2)
                if self.isRunning():
                    self.terminate()
                    self.wait(3000)
            t = threading.Thread(target=_force_terminate, daemon=True)
            t.start()

    def run(self):
        if self.fitter_config.get("separate", True):
            self._run_separate()
        else:
            self._run_simultaneous()

    def _run_separate(self):
        """Fit each file independently, in parallel.

        Each worker process writes per-iteration progress into a shared
        Manager.Queue; a poller thread in this worker blends those
        fractions with the per-run completion count to form a real
        aggregate-percent progress bar.
        """
        import threading
        from multiprocessing import Manager
        results = []
        total = len(self.run_files)

        # Per-run timeout: generous default (10 min) to avoid infinite hangs
        timeout_per_run = 600

        # Cross-process progress queue. Workers push (filepath, fraction)
        # tuples; the poller below maintains `inner_progress[fp]` and
        # emits the combined percent at most every ~150ms.
        manager = Manager()
        progress_queue = manager.Queue()
        inner_progress = {fp: 0.0 for fp in self.run_files}
        progress_lock = threading.Lock()
        stop_poll = threading.Event()
        last_status = ["Starting..."]

        def _emit_combined():
            with progress_lock:
                pct = (sum(inner_progress.values()) / total
                       if total else 0.0)
            self.progress.emit(
                int(round(pct * 100)), 100,
                f"{int(round(pct * 100))}%  {last_status[0]}")

        def _poller():
            import time as _t
            last_emit = 0.0
            while not stop_poll.is_set():
                try:
                    msg = progress_queue.get(timeout=0.2)
                except Exception:
                    msg = None
                if msg is not None:
                    fp, frac = msg
                    with progress_lock:
                        # Monotonic: a worker may push 0.5 just before
                        # we've seen its "done" 1.0; never go backwards.
                        prev = inner_progress.get(fp, 0.0)
                        inner_progress[fp] = max(prev, float(frac))
                now = _t.monotonic()
                if now - last_emit >= 0.15:
                    _emit_combined()
                    last_emit = now
            # Final emit on shutdown so the bar lands on the true value
            _emit_combined()

        poll_thread = threading.Thread(target=_poller, daemon=True)
        poll_thread.start()

        try:
            with ProcessPoolExecutor(max_workers=self.n_cores) as pool:
                futures = {}
                for filepath in self.run_files:
                    if self._cancelled:
                        break
                    merged = self.merged_data_map.get(filepath)
                    split = self.split_descriptors_map.get(filepath)
                    override = self.file_overrides_map.get(filepath)
                    corr = self.centroid_corrections_map.get(filepath)
                    # Scan filter is keyed by ASDF path. For a virtual
                    # split, the data path is the parent ASDF, so use
                    # that key; merged spectra carry no ASDF and are
                    # skipped (the filter has nowhere to apply).
                    if split is not None:
                        sf_key = split.get("parent_path")
                    elif merged is not None:
                        sf_key = None
                    else:
                        sf_key = filepath
                    sf = (self.scan_filters_map.get(sf_key)
                          if sf_key else None)
                    future = pool.submit(
                        _fit_single_run, filepath,
                        self.source_config, self.model_configs,
                        self.fitter_config, self.output_config,
                        merged_data=merged,
                        split_descriptor=split,
                        binning_override=override,
                        centroid_correction=corr,
                        scan_filter=sf,
                        calibrations=self.calibrations_map,
                        progress_queue=progress_queue)
                    futures[future] = filepath

                last_status[0] = f"Fitting {total} run(s)..."

                completed = 0
                for future in as_completed(futures):
                    if self._cancelled:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    fname = os.path.basename(futures[future])
                    try:
                        result = future.result(timeout=timeout_per_run)
                        results.append(result)
                        self.fit_done.emit(result)
                    except TimeoutError:
                        print(f"[FitWorker] Timeout for {fname}")
                        results.append({
                            "success": False,
                            "run_file": futures[future],
                            "run_number": "?",
                            "error": f"Fit timed out after "
                                     f"{timeout_per_run}s",
                        })
                    except Exception as e:
                        import traceback
                        print(f"[FitWorker] Future error for "
                              f"{futures[future]}: {e}")
                        traceback.print_exc()
                        results.append({
                            "success": False,
                            "run_file": futures[future],
                            "run_number": "?",
                            "error": str(e),
                        })
                    # Whatever happened, this run is "done" — pin its
                    # inner progress to 1.0 so the bar reflects it even
                    # if the worker bailed before sending the final 1.0.
                    with progress_lock:
                        inner_progress[futures[future]] = 1.0
                    completed += 1
                    last_status[0] = f"{fname}  ({completed}/{total})"
                    _emit_combined()

        except Exception as e:
            import traceback
            print(f"[FitWorker] Pool-level error: {e}")
            traceback.print_exc()
            self.error.emit(str(e))
            return
        finally:
            # Always stop the poller and shut the Manager helper process down
            # deterministically (it was previously left to GC, leaking an idle
            # helper process per fit run -- code review 2026-06-02,
            # fitworker-manager-leak).
            stop_poll.set()
            poll_thread.join(timeout=1.0)
            try:
                manager.shutdown()
            except Exception:
                pass

        print(f"[FitWorker] All done: {len(results)} results, "
              f"cancelled={self._cancelled}")
        if self._cancelled:
            for r in results:
                r["cancelled"] = True
        self.all_done.emit(results)

    def _run_simultaneous(self):
        """Fit all files simultaneously in a single fitter."""
        if not _has_clstools():
            self.error.emit("clstools is required for simultaneous fitting "
                            "but is not installed.")
            return
        try:
            # A windowed Qt app (pythonw.exe) leaves sys.stdout/stderr
            # None, which crashes tqdm inside satlas2's emcee. Redirect
            # to devnull so tqdm has a writable file.
            import sys, os
            if sys.stdout is None:
                sys.stdout = open(os.devnull, "w")
            if sys.stderr is None:
                sys.stderr = open(os.devnull, "w")

            import satlas2 as sat
            _patch_satlas2_emcee_numpy2()
            from gui.analysis.binning import (
                compute_binned, summarize_x, effective_binning_config,
                intersect_v_gate, rebin_to_common_grid)
            from gui.analysis.pipeline import prepare_run_data

            fitter = sat.Fitter()
            source_objects = {}

            # Pass 1: load + bin every file, collect descriptors. Source
            # objects are NOT created yet so the optional common-grid step
            # can re-bin everything in a second pass.
            descriptors = []   # one per checked file, in order
            for filepath in self.run_files:
                merged = self.merged_data_map.get(filepath)
                if merged is not None:
                    # Phase 4: voltage-merged spectrum fitted in
                    # frequency mode -- project bin centers onto MHz
                    # using merge_metadata.
                    sim_extra_warnings = []
                    if (merged.get("x_unit") == "V"
                            and self.source_config.get("bin_mode")
                                == "Frequency"):
                        from gui.analysis.merge import (
                            project_voltage_merge_to_frequency)
                        x_proj, y_proj, yerr_proj, w_proj = (
                            project_voltage_merge_to_frequency(
                                merged, self.source_config))
                        x = np.array(x_proj)
                        y = np.array(y_proj)
                        yerr = np.array(yerr_proj)
                        sim_extra_warnings = w_proj
                        _x_unit = "MHz"
                    else:
                        x = np.array(merged["x"])
                        y = np.array(merged["y"])
                        yerr = np.sqrt(y + 1)
                        _x_unit = merged.get("x_unit", "MHz")
                    # Also forward any compute-time merge_warnings
                    # (e.g. Phase-5 MANUAL_OFFSET_IGNORED_…) so the
                    # fit report surfaces them on this run.
                    for w in merged.get("merge_warnings", []) or []:
                        sim_extra_warnings.append(dict(w))
                    source_name = source_name_for_merged(merged)
                    run_num = source_name[len("Run_"):]
                    _msum = summarize_x(x)
                    descriptors.append({
                        "name": source_name,
                        "x": x, "y": y, "yerr": yerr, "xerr": None,
                        "use_callable_yerr": False,
                        "yerr_mode": "Poisson sqrt(y+1)",
                        "run_num": run_num, "filepath": filepath,
                        "run_metadata": _merged_run_metadata(merged),
                        "binning_info": {
                            "source": "merged",
                            "merged_name": merged.get("merged_name"),
                            "bin_mode": ("Frequency" if _x_unit == "MHz"
                                         else "Raw Voltage"),
                            "bin_definition": "Pre-merged",
                            "fallback_used": False,
                            "requested_bin_count": None,
                            "requested_bin_width_mhz": None,
                            "effective_n_bins": _msum["n_bins"],
                            "effective_bin_width_mhz": (
                                _msum["dx_median"] if _x_unit == "MHz"
                                else None),
                            "effective_bin_width_v": (
                                _msum["dx_median"] if _x_unit != "MHz"
                                else None),
                            "voltage_to_frequency_warnings":
                                sim_extra_warnings,
                            "override_active": False,
                            "override_keys": [],
                            **_msum,
                        },
                    })
                    continue

                # Standard ASDF loading. Virtual-split entries point at a
                # parent ASDF and carry per-side metadata + a raw-voltage gate
                # that prepare_run_data composes with the SourceBlock V-gate.
                # Shared with the separate fit / auto-fit / merge / preview so
                # they cannot drift. Correction fields are filled below, AFTER
                # the bin_mode branch -- Raw Voltage must not be reported as
                # corrected.
                _split_desc = self.split_descriptors_map.get(filepath)
                _is_split = _split_desc is not None
                _data_path = (_split_desc["parent_path"]
                              if _is_split else filepath)
                _override = self.file_overrides_map.get(filepath) or {}

                data, _eff_cfg, _run_meta = prepare_run_data(
                    filepath, self.source_config,
                    binning_override=_override,
                    split_descriptor=_split_desc,
                    calibrations=self.calibrations_map)
                bin_mode = _eff_cfg.get("bin_mode", "Frequency")

                # Phase D: per-file GP correction is applied
                # post-binning in MHz directly. clstools.Shift_Ref
                # overwrites rather than composes (it would wipe the
                # earlier ref_shift) and takes Hz not MHz, so a direct
                # subtraction on the MHz binned axis is used instead.
                _corr_dict = self.centroid_corrections_map.get(filepath)
                _corr_mhz, _corr_sigma = _extract_correction(_corr_dict)
                _corr_mode = (_corr_dict.get("mode")
                               if isinstance(_corr_dict, dict)
                               else None)
                # Per-file scan filter: keyed by parent ASDF path for
                # virtual splits, otherwise by the file's own path.
                # The merged branch above already `continue`d, so we
                # don't need a None-merged guard here.
                _sf_key = (_split_desc.get("parent_path")
                           if _is_split else filepath)
                _sf = self.scan_filters_map.get(_sf_key)
                from gui.scan_filter import filter_data_for_binning
                with filter_data_for_binning(
                        data, _sf, asdf_path=_data_path):
                    res = compute_binned(data, _eff_cfg)
                _correction_applied = (
                    bin_mode != "Raw Voltage"
                    and _corr_mode in ("Auto", "Manual"))
                if _correction_applied and _corr_mhz != 0.0:
                    res["x"] = res["x"] - _corr_mhz
                    _shift_binning_info_x(res.get("info"), _corr_mhz)
                _run_meta["centroid_correction_applied"] = bool(
                    _correction_applied)
                _run_meta["centroid_correction_mode"] = (
                    _corr_mode if _correction_applied else None)
                _run_meta["centroid_correction_mhz"] = (
                    float(_corr_mhz) if _correction_applied else 0.0)
                _run_meta["centroid_correction_sigma_mhz"] = (
                    float(_corr_sigma) if _correction_applied else 0.0)
                _binning_info = res["info"]
                _binning_info["override_active"] = bool(_override)
                _binning_info["override_keys"] = sorted(_override.keys())
                if _binning_info.get("fallback_used"):
                    print(f"[FitWorker] Binning fallback used for "
                          f"{os.path.basename(filepath)}: "
                          f"{_binning_info.get('n_bins')} bins "
                          f"(clstools default failed)", flush=True)

                if _is_split:
                    source_name = source_name_for_descriptor({
                        "kind": "virtual_split",
                        "source_id": _split_desc["source_id"],
                    })
                else:
                    source_name = source_name_for_path(filepath)
                run_num = source_name[len("Run_"):]
                descriptors.append({
                    "name": source_name,
                    "x": res["x"], "y": res["y"],
                    "yerr": res["yerr"], "xerr": res["xerr"],
                    "use_callable_yerr": res["use_callable_yerr"],
                    "yerr_mode": _eff_cfg.get("yerr_mode",
                                              "Poisson sqrt(y+1)"),
                    "run_num": run_num, "filepath": filepath,
                    "run_metadata": _run_meta,
                    "binning_info": _binning_info,
                })

            # Optional common-grid rebinning before sources are created.
            if (self.fitter_config.get("common_grid")
                    and len(descriptors) >= 2):
                modes = {d["binning_info"].get("bin_mode") for d in descriptors}
                if len(modes) != 1:
                    print(f"[FitWorker] Common grid skipped: mixed bin "
                          f"modes {modes}. Falling back to independent "
                          f"grids.", flush=True)
                    for d in descriptors:
                        d["binning_info"]["common_grid_used"] = False
                        d["binning_info"]["common_grid_skipped_reason"] = (
                            "mixed bin modes")
                else:
                    bw_req = float(self.fitter_config.get(
                        "common_grid_width", 0.0)) or None
                    rebinned, centers, bw_used = rebin_to_common_grid(
                        [(d["x"], d["y"]) for d in descriptors],
                        bin_width=bw_req)
                    print(f"[FitWorker] Common grid: {len(centers)} bins "
                          f"@ {bw_used:.4f}", flush=True)
                    only_mode = next(iter(modes))
                    width_key = ("effective_bin_width_mhz"
                                 if only_mode == "Frequency"
                                 else "effective_bin_width_v")
                    for d, (cx, cy) in zip(descriptors, rebinned):
                        # Recompute yerr from rebinned counts using the
                        # source's original yerr_mode. Skip the low-count
                        # Poisson interval correction here — it's a per-bin
                        # statistical refinement that loses meaning once we
                        # have summed counts from multiple samplings.
                        ymode = d["yerr_mode"]
                        if ymode == "Poisson sqrt(y)":
                            yerr_new = np.sqrt(cy)
                            yerr_new[yerr_new == 0] = 1.0
                        elif ymode == "Model-based":
                            # callable; satlas2 will keep evaluating
                            # sqrt(model) per iteration. No change needed.
                            yerr_new = np.sqrt
                        else:   # default: Poisson sqrt(y+1)
                            yerr_new = np.sqrt(cy + 1)
                        d["x"] = cx
                        d["y"] = cy
                        d["yerr"] = yerr_new
                        d["xerr"] = None   # dropped by common-grid rebin
                        info = d["binning_info"]
                        info["common_grid_used"] = True
                        info["common_grid_width"] = bw_used
                        info["effective_n_bins"] = int(len(cx))
                        info[width_key] = bw_used
            else:
                for d in descriptors:
                    d["binning_info"].setdefault("common_grid_used", False)

            # Pass 2: build Source objects from the (possibly rebinned)
            # descriptors and add them to the simultaneous fitter.
            for d in descriptors:
                # yerr_mode "None" -> yerr is None; unit weights so the plain
                # least-squares case does not crash inside satlas2.
                d["yerr"] = _resolve_fit_yerr(d["yerr"], d["y"], False)
                source = sat.Source(d["x"], d["y"], yerr=d["yerr"],
                                    xerr=d["xerr"], name=d["name"])
                _build_models_on_source(source, self.model_configs, sat)
                fitter.addSource(source)
                source_objects[d["name"]] = {
                    "source": source, "x": d["x"], "y": d["y"],
                    "yerr": d["yerr"], "run_num": d["run_num"],
                    "filepath": d["filepath"],
                    "run_metadata": d["run_metadata"],
                    "binning_info": d["binning_info"],
                }

            # Apply parameter sharing
            for param_name in self.fitter_config.get("shared_params", []):
                fitter.shareModelParams(param_name)
            for param_name in self.fitter_config.get("shared_all_params", []):
                fitter.shareParams(param_name)

            # Apply model-level expressions across all sources. Each
            # expression is lowered from the model-local namespace
            # (bare names) into the full lmfit namespace per source.
            # Fitter-block expressions below take precedence: setExpr
            # is last-write-wins.
            for mc in self.model_configs:
                mt = mc.get("type", "")
                local_params = {p for p in mc["params"]
                                if is_fit_param_name(mt, p)}
                for pname, pdata in mc["params"].items():
                    if not is_fit_param_name(mt, pname):
                        continue
                    expr = (pdata.get("expr") or "").strip()
                    if not expr:
                        continue
                    for source_name in source_objects:
                        lowered = lower_model_expression(
                            expr, source_name, mc["name"], local_params)
                        fitter.setExpr(
                            full_param_name(
                                source_name, mc["name"], pname),
                            lowered,
                        )

            # Apply fitter-block expressions (full names already)
            for full_name, expr in self.fitter_config.get(
                    "expressions", {}).items():
                fitter.setExpr(full_name, expr)

            # Apply priors
            for prior in self.fitter_config.get("priors", []):
                parts = prior["param"].split("___")
                if len(parts) == 3:
                    fitter.setParamPrior(parts[0], parts[1], parts[2],
                                         prior["value"], prior["uncertainty"])

            self.progress.emit(0, 100, "Fitting all sources simultaneously...")

            # Fit
            fit_kwargs = {"method": self.fitter_config["method"],
                          "scale_covar": self.fitter_config.get(
                              "scale_covar", True)}
            if self.fitter_config.get("llh"):
                fit_kwargs["llh"] = True
                fit_kwargs["llh_method"] = self.fitter_config["llh_method"]
                fit_kwargs["scale_covar"] = False
            if self.fitter_config["method"] == "emcee":
                fit_kwargs["nwalkers"] = self.fitter_config.get("nwalkers", 50)
                fit_kwargs["steps"] = self.fitter_config.get("steps", 1000)
                # emcee requires a filename for the HDF5 chain backend
                import tempfile
                chain_file = os.path.join(
                    tempfile.gettempdir(),
                    f"emcee_simultaneous_{os.getpid()}.h5")
                fit_kwargs["filename"] = chain_file

            # satlas2's emcee builds empty bounds when no params vary,
            # which crashes lmfit's _lnprob with a cryptic IndexError.
            # Surface a clear message instead.
            fitter._prepareFit()
            n_vary = sum(1 for p in fitter.lmpars.values()
                         if p.vary and p.expr is None)
            if n_vary == 0:
                raise ValueError(
                    "No varying parameters: every parameter is fixed "
                    "(vary=False) or constrained by an expression. "
                    "Enable at least one parameter to fit.")

            # Real-percent progress: for emcee the total = `steps`, for
            # least-squares we estimate against lmfit's default cap of
            # max_nfev = 200 * (N_vary + 1) and clamp at 95 % so the bar
            # doesn't claim done before the fit returns. Throttled to
            # 200 ms.
            import time as _time_sim
            method = self.fitter_config["method"]
            if method == "emcee":
                _max_iter_sim = max(1, int(fit_kwargs.get("steps", 1000)))
                _cap_sim = 100
            else:
                _max_iter_sim = max(50, 200 * (n_vary + 1))
                _cap_sim = 95
            _last_emit_sim = [0.0]

            def _iter_cb_sim(_params, iteration, _resid, *_a, **_kw):
                # Cooperative cancellation: lmfit aborts the fit at the next
                # iteration boundary when iter_cb returns True, so a Cancel no
                # longer relies on QThread.terminate() killing a thread mid
                # C-call (which can corrupt the interpreter) -- terminate()
                # remains only as a backstop for a wedged fit (code review
                # 2026-06-02, qthread-terminate-on-c-fit).
                if self._cancelled:
                    return True
                now = _time_sim.monotonic()
                if now - _last_emit_sim[0] < 0.2:
                    return
                _last_emit_sim[0] = now
                pct = min(_cap_sim,
                          int(round(100.0 * iteration / _max_iter_sim)))
                self.progress.emit(
                    pct, 100,
                    f"{pct}%  iter {iteration}/~{_max_iter_sim}")
            fit_kwargs["iter_cb"] = _iter_cb_sim

            fitter.fit(**fit_kwargs)
            if self._cancelled:
                # Aborted via iter_cb above; don't emit a partial/aborted fit.
                return

            # emcee MinimizerResult may lack 'message' -- set a fallback
            if not hasattr(fitter.result, 'message'):
                fitter.result.message = "Fit completed"

            # Burn-in / thinning before the report + result dataframe.
            emcee_acor = None
            if self.fitter_config["method"] == "emcee":
                emcee_acor = _apply_emcee_burnin(
                    fitter, fit_kwargs.get("filename"),
                    self.fitter_config.get("burn", 0),
                    self.fitter_config.get("thin", 1))

            report = fitter.reportFit(
                show_correl=self.fitter_config.get("show_correl", True))
            if emcee_acor:
                _acor_lines = "\n".join(
                    f"    {k}: {v:.1f}" for k, v in emcee_acor.items())
                report = (
                    report
                    + "\n\n[[Burn-in]]\n"
                    + f"    Discarded first "
                    + f"{int(self.fitter_config.get('burn', 0))} steps; "
                    + f"thin = {int(self.fitter_config.get('thin', 1))}.\n"
                    + "    Integrated autocorrelation time (steps):\n"
                    + _acor_lines)
            params_df = fitter.createResultDataframe()
            metadata_df = fitter.createMetadataDataframe()

            # Collect diagnostic raw data for simultaneous fit
            diagnostics = {}
            is_emcee = self.fitter_config["method"] == "emcee"
            chain_file = fit_kwargs.get("filename")

            if is_emcee and chain_file and os.path.isfile(chain_file):
                print(f"[FitWorker] Reading chain file for diagnostics "
                      f"(simultaneous)", flush=True)
                try:
                    import h5py
                    with h5py.File(chain_file, "r") as hf:
                        g = hf["mcmc"]
                        chain = g["chain"][:]
                        labels = list(g.attrs["labels"])
                    labels = [l.decode() if isinstance(l, bytes) else str(l)
                              for l in labels]
                    var_mask = [i for i, l in enumerate(labels)
                                if fitter.result.params[l].vary]
                    var_labels = [labels[i].split("___")[-1]
                                  for i in var_mask]
                    var_chain = chain[:, :, var_mask]

                    # Filter by user-selected diagnostic params
                    diag_sel = self.output_config.get("diag_params", [])
                    if diag_sel:
                        sel = [j for j, vl in enumerate(var_labels)
                               if vl in diag_sel]
                        if sel:
                            var_labels = [var_labels[j] for j in sel]
                            var_chain = var_chain[:, :, sel]

                    if self.output_config.get("walk_plot"):
                        diagnostics["walk_data"] = {
                            "labels": var_labels,
                            "chain": var_chain,
                        }
                    if self.output_config.get("correl_plot"):
                        flat = var_chain.reshape((-1, len(var_labels)))
                        diagnostics["correl_data"] = {
                            "labels": var_labels,
                            "flatchain": flat,
                        }
                    print(f"[FitWorker] Chain diagnostics collected OK "
                          f"(simultaneous)", flush=True)
                except Exception as exc:
                    print(f"[FitWorker] Chain read error (simultaneous): "
                          f"{exc}", flush=True)

            if self.output_config.get("chisq_map"):
                try:
                    fitter.temp_y = fitter.y()
                    fitter.setParameters(fitter.result.params)

                    vary_params = [(name, p) for name, p in
                                   fitter.result.params.items() if p.vary]
                    # Filter by user-selected diagnostic params
                    diag_sel = self.output_config.get("diag_params", [])
                    if diag_sel:
                        vary_params = [(n, p) for n, p in vary_params
                                       if n.split("___")[-1] in diag_sel]
                    chisq_scans = []
                    n_pts = 30
                    for pname, par in vary_params:
                        best = par.value
                        stderr = par.stderr if par.stderr else (
                            abs(best) * 0.1 or 1.0)
                        scan = np.linspace(best - 3 * stderr,
                                           best + 3 * stderr, n_pts)
                        chisq_vals = []
                        orig_val = par.value
                        for sv in scan:
                            par.value = sv
                            fitter.setParameters(fitter.result.params)
                            r = fitter.resid()
                            chisq_vals.append(
                                float(np.sum(r ** 2)))
                        par.value = orig_val
                        fitter.setParameters(fitter.result.params)
                        chisq_scans.append({
                            "label": pname.split("___")[-1],
                            "scan": scan.tolist(),
                            "chisq": chisq_vals,
                            "best": float(best),
                        })
                    if chisq_scans:
                        diagnostics["chisq_data"] = chisq_scans
                    del fitter.temp_y
                except Exception as exc:
                    diagnostics["chisq_error"] = (
                        f"{type(exc).__name__}: {exc}")

            # Fit quality
            fit_quality = {
                "chisqr": float(fitter.chisqr)
                          if hasattr(fitter, 'chisqr') else None,
                "redchi": float(fitter.redchi)
                          if hasattr(fitter, 'redchi') else None,
                "ndata": int(fitter.ndata)
                         if hasattr(fitter, 'ndata') else None,
                "nvarys": int(fitter.nvarys)
                          if hasattr(fitter, 'nvarys') else None,
            }
            # Under LLH, satlas2's chisqr/redchi are the -logL surrogate, not a
            # chi-square. Replace with a genuine reduced chi-square summed over
            # all sources' residuals so "Reduced Chi-sq" stays meaningful; the
            # chi-square fit is unaffected (code review 2026-06-02,
            # llh-redchi-mislabeled).
            if self.fitter_config.get("llh"):
                _tot = 0.0
                for _sd in source_objects.values():
                    _xx = np.asarray(_sd["x"], dtype=float)
                    _yy = np.asarray(_sd["y"], dtype=float)
                    _yf = _sd["source"].evaluate(_xx)
                    _ye = _sd["yerr"]
                    if callable(_ye):
                        _ye = np.sqrt(np.maximum(_yf, 1.0))
                    _ye = np.asarray(_ye, dtype=float)
                    _r = (_yy - _yf) / _ye
                    _tot += float(np.sum(_r * _r))
                _dof = max(1, int(fit_quality["ndata"] or 0)
                           - int(fit_quality["nvarys"] or 0))
                fit_quality["chisqr"] = _tot
                fit_quality["redchi"] = _tot / _dof
                fit_quality["gof_from_residuals"] = True
                for _col, _val in (("Chisquare", _tot),
                                   ("Redchi", _tot / _dof)):
                    if _col in metadata_df.columns:
                        metadata_df[_col] = _val

            # Build per-source results
            results = []
            for i, (sname, sdata) in enumerate(source_objects.items()):
                src = sdata["source"]
                x = sdata["x"]
                y = sdata["y"]
                yerr_arr = sdata["yerr"]
                if callable(yerr_arr):
                    y_fit = src.evaluate(x)
                    yerr_arr = np.sqrt(np.maximum(y_fit, 1.0))
                else:
                    y_fit = src.evaluate(x)
                x_smooth, _smooth_gap_mask = _smooth_grid_with_gaps(x)
                _y_fit_smooth = src.evaluate(x_smooth)
                if _smooth_gap_mask.any():
                    _y_fit_smooth = np.asarray(
                        _y_fit_smooth, dtype=float).copy()
                    _y_fit_smooth[_smooth_gap_mask] = np.nan

                # FWHM and peak positions for this source's models
                model_fwhm = {}
                model_positions = {}
                for mname, model in src.models:
                    try:
                        fv, fu = model.calculateFWHM()
                        model_fwhm[mname] = {"value": float(fv),
                                              "uncertainty": float(fu)}
                    except Exception:
                        pass
                    try:
                        pos = model.pos()
                        model_positions[mname] = [float(p) for p in pos]
                    except Exception:
                        pass

                entry = {
                    "success": True,
                    "run_number": sdata["run_num"],
                    "run_file": sdata["filepath"],
                    "source_name": sname,
                    "report": report,
                    "params_df": params_df.to_dict("list"),
                    "metadata_df": metadata_df.to_dict("list"),
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "yerr": yerr_arr.tolist(),
                    "y_fit": y_fit.tolist(),
                    "x_smooth": x_smooth.tolist(),
                    "y_fit_smooth": np.asarray(_y_fit_smooth).tolist(),
                    "residuals": ((y - y_fit) / yerr_arr).tolist(),
                    "diagnostics": diagnostics if i == 0 else {},
                    "fwhm": model_fwhm,
                    "peak_positions": model_positions,
                    "fit_quality": fit_quality,
                    "chain_file": fit_kwargs.get("filename"),
                    "run_metadata": sdata.get("run_metadata", {}),
                    "harmonic": self.source_config.get("harmonic", 0),
                    "binning_info": sdata.get("binning_info"),
                }
                results.append(entry)

            self.progress.emit(1, 1, "Done")
            self.all_done.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
