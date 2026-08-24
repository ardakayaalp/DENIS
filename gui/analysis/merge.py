"""Multi-run spectrum merging: dialog, computation, view, and ASDF export.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Loads several clstools ASDF runs, Doppler-converts each to a common
rest-frame frequency (or raw-voltage) axis, histograms them onto a shared
bin grid, and returns a single merged spectrum. Provides the configuration
dialog, the multi-tab viewer, a voltage-merge-to-frequency projection used
at fit time, and read/write of DENIS-labelled merged-spectrum ASDF files.

Depends on: gui.analysis.binning, gui.analysis.helpers,
gui.analysis.shared_widgets, cls_estimations.doppler; uses clstools,
NumPy/pandas, matplotlib, and PySide6.
"""

import os
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QPushButton, QDialogButtonBox, QListWidget,
    QTabWidget, QWidget, QScrollArea, QMessageBox, QFileDialog,
)
from PySide6.QtCore import Qt, QLocale

try:
    import clstools as cls
    _HAS_CLSTOOLS = True
except ImportError:
    _HAS_CLSTOOLS = False


# ══════════════════════════════════════════════════════════════════
#  Merge computation
# ══════════════════════════════════════════════════════════════════

def compute_merged_spectrum(file_paths, source_config, merge_params,
                            file_overrides=None,
                            centroid_corrections_map=None,
                            manual_offsets_map=None):
    """Load multiple ASDF runs and merge into a single spectrum.

    Frequency mode correctness
    --------------------------
    Each file is independently loaded and Doppler-converted by clstools
    using that file's own ``Laser_set`` and cooler voltage (from ASDF
    metadata: ``VCoolDiv`` / ``VCoolOffset``).  This means files with
    different laser setpoints or acceleration voltages are correctly mapped
    to the *same* rest-frame frequency axis before binning, even though
    they may scan different voltage ranges in the lab frame.

    When ``cooler_override`` or ``laser_override`` is non-zero, all files
    are forced to the same value; when zero, per-file ASDF values are used.

    Parameters
    ----------
    file_paths : list[str]
        Paths to ASDF files.
    source_config : dict
        From ``SourceBlock.get_source_config()``.
    merge_params : dict
        ``bin_step_mhz`` (float or None for auto),
        ``cooler_correction`` (str),
        ``centroid_correction`` (bool) — when True AND
        ``centroid_corrections_map`` is supplied, the per-file GP
        correction is subtracted from each file's binned MHz axis
        before histogramming.
        ``domain`` (str, optional) — "voltage" or "frequency". When
        present, overrides the auto-decision based on
        ``bin_mode``/physics; "voltage" forces a raw-voltage merge
        regardless of whether ref_freq/mass are set, "frequency"
        requires both to be set and raises otherwise.
        ``merge_metadata`` (dict, optional) — user-chosen Doppler
        metadata for a voltage merge (cooler_v, laser_sp, mass_amu,
        harmonic). Passed through to the output so the voltage-merged
        fit path can apply a Doppler shift after the fact. Ignored for
        frequency merges.
    file_overrides : dict[str, dict] or None
        Optional sparse per-file binning overrides keyed by file path
        (same shape used elsewhere in the analysis pipeline). Each
        value is whitelisted by ``BINNING_OVERRIDE_KEYS`` so only
        binning-shape keys take effect.
    centroid_corrections_map : dict[str, dict] or None
        Optional per-file GP corrections keyed by file path. Same
        shape as the dict the fit pipeline consumes:
        ``{file_path: {correction_mhz, sigma_mhz, mode}}``. Mode
        ``"Off"`` is honored.

    Returns
    -------
    dict
        Keys: ``x``, ``y``, ``yerr``, ``per_run``, ``bin_step_mhz``,
        ``x_unit``. When ``merge_params["domain"] == "voltage"`` and
        ``merge_params["merge_metadata"]`` is supplied, the output
        also carries a ``merge_metadata`` key for downstream Doppler
        application.
    """
    import clstools as _cls
    file_overrides = file_overrides or {}
    corr_map = centroid_corrections_map or {}
    manual_map = manual_offsets_map or {}
    apply_correction = bool(merge_params.get("centroid_correction", False))

    per_run = []           # [{x, y, run_num, tof_index, tof_counts, ...}]
    freq_steps = []

    cal_order = source_config.get("cal_order", 1)
    cooler_corr = merge_params.get("cooler_correction",
                                    source_config.get("cooler_correction",
                                                       "pbp"))
    tof_gate = source_config.get("tof_gate")
    pmt_gate = source_config.get("pmt_gate", [3, 4])
    v_gate = source_config.get("v_gate")
    f_gate = source_config.get("f_gate")
    mass = source_config["mass"]
    ref_freq = source_config["ref_freq"]
    harmonic = source_config["harmonic"]

    # Explicit domain override. When the caller doesn't pass one we keep
    # the "decide from bin_mode + physics" behavior so existing callers
    # (Analysis SourceBlock) don't change shape. When the caller passes
    # one, it's authoritative -- "voltage"
    # forces a raw-voltage merge even if ref_freq/mass are set, which
    # is what the unified dialog uses to honor the user's domain
    # selector.
    explicit_domain = merge_params.get("domain")
    if explicit_domain not in (None, "voltage", "frequency"):
        raise ValueError(
            f"merge_params['domain'] must be 'voltage' or 'frequency', "
            f"got {explicit_domain!r}")
    # Pre-flight: a frequency-domain merge needs the physics. Catch
    # this BEFORE the file loop's first Load_Run so a misconfigured
    # caller doesn't waste I/O before erroring out. ``harmonic <= 0``
    # is checked too because clstools.Compute_WL multiplies the laser
    # by the harmonic; a non-positive value yields a degenerate axis
    # (F = 0 − ref everywhere) that the dialog's spinner range
    # ``[1, 32]`` already prevents, but a programmatic caller could
    # still pass garbage.
    if explicit_domain == "frequency" and (
            ref_freq == 0 or mass <= 0 or harmonic <= 0):
        raise ValueError(
            "Frequency-domain merge requires non-zero ref_freq, "
            f"positive mass, and positive harmonic in source_config; "
            f"got ref_freq={ref_freq}, mass={mass}, "
            f"harmonic={harmonic}.")

    # The calibration each contributing run is merged with. Resolved once up
    # front: a merged spectrum bakes its runs' voltage axes in permanently, so
    # the calibration in force at merge time is the one that survives into the
    # result -- and it gets recorded per run below.
    from gui.calibration import get_registry as _get_cal_registry
    from gui.calibration import load_run_calibrated as _load_calibrated
    cal_map = _get_cal_registry().to_dict()

    for fpath in file_paths:
        data = _cls.CLSDataFrame()
        cal_res = _load_calibrated(data, fpath, cal_map, cal_order=cal_order)

        cooler_override = source_config.get("cooler_override", 0)
        if cooler_override > 0:
            data.VCoolDiv = 0
            data.VCoolOffset = cooler_override
        laser_override = source_config.get("laser_override", 0)
        if laser_override > 0:
            data.Laser_set = laser_override

        data.Compute_Voltages(cooler_correction=cooler_corr)

        nf = source_config.get("noise_filter", 0)
        if nf > 0:
            data.apply_filter(filter_window=nf)

        # TOF for display. ``data.ToF_binned`` from clstools puts
        # every unique event timestamp on its own ~0.01 µs bin (8000+
        # bins across 0-100 µs), so a direct histogram on those is
        # visually flat noise even when the underlying bunch arrival
        # is a clear peak. Re-bin into 1 µs bins so the bunch peak is
        # visible.
        data.Compute_ToF(V_gate=v_gate, PMT_gate=pmt_gate)
        tof_idx_fine = np.array(data.ToF_binned.index.to_list(),
                                  dtype=float)
        tof_cts_fine = np.array(data.ToF_binned.values[:, 0],
                                  dtype=float)
        if len(tof_idx_fine) > 0:
            t_lo = float(np.floor(tof_idx_fine.min()))
            t_hi = float(np.ceil(tof_idx_fine.max())) + 1.0
            edges = np.arange(t_lo, t_hi + 1.0, 1.0)
            # Weighted histogram so the per-event-bin counts are
            # preserved: np.histogram with weights sums them per 1 µs
            # bin in one pass.
            tof_cts, _ = np.histogram(
                tof_idx_fine, bins=edges, weights=tof_cts_fine)
            tof_idx = 0.5 * (edges[:-1] + edges[1:])
        else:
            tof_idx = np.array([], dtype=float)
            tof_cts = np.array([], dtype=float)

        # Bin: use frequency mode if physics params are set,
        # otherwise auto-fall-back to raw voltage mode.
        # Per-file binning override (e.g. flipping bin_mode) is
        # applied here so the merged spectrum honors the same
        # per-file settings the rest of the pipeline does.
        from gui.analysis.binning import (
            compute_binned, effective_binning_config)
        eff_cfg = effective_binning_config(
            source_config, file_overrides.get(fpath, {}))
        bin_mode = eff_cfg.get("bin_mode", "Frequency")
        if explicit_domain == "voltage":
            use_voltage = True
        elif explicit_domain == "frequency":
            # ref_freq / mass were validated above the loop; this just
            # honors the caller's domain choice.
            use_voltage = False
        else:
            # Legacy auto-decision when no explicit domain was passed.
            use_voltage = (bin_mode == "Raw Voltage" or ref_freq == 0
                           or mass <= 0)
        if not use_voltage:
            # Fix Vcool_init so Compute_WL derives Frequency_stepsize
            # correctly under a cooler override (mirrors the fit/auto-fit
            # prep; previously missing here, so an overridden frequency merge
            # got a degenerate stepsize -- code review 2026-06-02,
            # merge-cooler-override-missing-vcool-fix).
            if cooler_override > 0:
                data.Vcool_init = cooler_override
                data.VCoolDiv = 1
            data.Compute_WL(Mass=mass, ref=ref_freq, harmonic=harmonic)
            if cooler_override > 0:
                data.VCoolDiv = 0
            ref_shift = source_config.get("ref_shift", 0.0)
            if ref_shift != 0.0:
                # Compose with ref_freq instead of overwriting --
                # clstools.Shift_Ref does `self.Reference = ref`
                # (assignment, not addition), so passing just
                # ref_shift would discard the rest-frame transition
                # calibration set by Compute_WL.
                data.Shift_Ref(ref=ref_freq + ref_shift)
            freq_steps.append(data.Frequency_stepsize)

        # eff_cfg["bin_mode"] must agree with use_voltage in both
        # directions -- otherwise compute_binned reads the wrong column
        # and we stamp the wrong x_unit. Force "Frequency" too (not just
        # "Raw Voltage"), so an explicit_domain="frequency" request that
        # overrides a Raw-Voltage source_config doesn't silently produce
        # voltage-axis data labelled x_unit="MHz".
        eff_cfg = dict(eff_cfg)
        eff_cfg["bin_mode"] = "Raw Voltage" if use_voltage else "Frequency"
        res = compute_binned(data, eff_cfg)
        run_x, run_y = res["x"], res["y"]
        x_unit = "V" if use_voltage else "MHz"

        # Apply the per-file GP centroid correction post-binning, in
        # MHz, only in frequency mode. Direct subtraction on the binned
        # axis sidesteps clstools' overwrite-not-compose Shift_Ref
        # behavior. Track the applied (μ, σ) per file so a downstream
        # merged-data fit can report what was subtracted rather than
        # silently zeroing out the audit trail.
        #
        # A manual per-file offset (set in PA via right-click
        # "Set centroid offset…") is additive with the GP correction
        # and applied unconditionally in frequency mode, regardless of
        # the apply_correction checkbox -- the checkbox gates the GP
        # path only. Both are recorded separately so the audit shows
        # what came from where.
        applied_mhz = 0.0
        applied_sigma = 0.0
        applied_mode: str | None = None
        manual_offset = 0.0
        if not use_voltage:
            if apply_correction:
                corr = corr_map.get(fpath)
                if corr and corr.get("mode") in ("Auto", "Manual"):
                    try:
                        applied_mhz = float(corr.get("correction_mhz", 0.0))
                        applied_sigma = float(corr.get("sigma_mhz", 0.0))
                    except (TypeError, ValueError):
                        applied_mhz = 0.0
                        applied_sigma = 0.0
                    applied_mode = corr.get("mode")
            try:
                manual_offset = float(manual_map.get(fpath, 0.0) or 0.0)
            except (TypeError, ValueError):
                manual_offset = 0.0
            total_shift = applied_mhz + manual_offset
            if total_shift != 0.0:
                run_x = run_x - total_shift

        base = os.path.basename(fpath)
        run_num = base.replace(".asdf", "").split("_")[-1]

        per_run.append({
            "run_num": run_num,
            "path": fpath,
            "x": run_x,
            "y": run_y,
            "tof_index": tof_idx,
            "tof_counts": tof_cts,
            "ts_start": getattr(data, "TSstart", 0),
            "ts_stop": getattr(data, "TSstop", 0),
            "n_events": getattr(data, "Size", 0),
            "laser_set": getattr(data, "Laser_set", 0),
            "cooler_v": getattr(data, "Vcool_init", 0),
            "dwell_time": getattr(data, "Dwell_Time", 0),
            "experiment": getattr(data, "Experiment", ""),
            "date": getattr(data, "Date", ""),
            "step_size": getattr(data, "Step_Size", 0),
            "scanning_ranges": getattr(data, "ScanningRanges", []),
            "cal_set": list(getattr(data, "Cal_df", pd.DataFrame()).get(
                "Set", [])),
            "cal_read": list(getattr(data, "Cal_df", pd.DataFrame()).get(
                "Read", [])),
            # Which voltage calibration this run's axis was built from.
            # A merged spectrum can never be re-calibrated -- its per-event
            # data is gone -- so the choice has to be recorded here or it is
            # lost for good.
            "calibration": cal_res.to_report_dict(),
            "vcool_div": getattr(data, "VCoolDiv", 10000),
            "vacc_div": getattr(data, "VAccDiv", 1000),
            # Audit: record what (if anything) was actually subtracted
            # from this file's MHz axis pre-merge. Zero / zero / None
            # when no correction ran.
            "centroid_correction_mhz": float(applied_mhz),
            "centroid_correction_sigma_mhz": float(applied_sigma),
            "centroid_correction_mode": applied_mode,
            # Separate audit for the PA-side manual offset. Tracked
            # apart from the GP correction so a downstream consumer can
            # attribute drift to one source or the other.
            "manual_offset_mhz": float(manual_offset),
        })

    # Determine bin step
    bin_step = merge_params.get("bin_step_mhz")
    if bin_step is None or bin_step <= 0:
        # Auto: compute median bin spacing from per-run data
        all_spacings = []
        for rd in per_run:
            rx = np.sort(rd["x"])
            if len(rx) > 1:
                diffs = np.diff(rx)
                good = diffs[diffs > 0]
                if len(good) > 0:
                    all_spacings.append(float(np.median(good)))
        if all_spacings:
            bin_step = float(np.median(all_spacings))
        else:
            bin_step = 1.0

    # Build a common grid that covers the union of all per-run x ranges
    all_x_cat = np.concatenate([rd["x"] for rd in per_run])
    if len(all_x_cat) == 0:
        raise ValueError("No data after gating -- nothing to merge.")

    x_lo = all_x_cat.min() - 0.5 * bin_step
    x_hi = all_x_cat.max() + 0.5 * bin_step
    edges = np.arange(x_lo, x_hi + bin_step, bin_step)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Sum counts: for each per-run bin center, find the nearest
    # common-grid bin and add the counts there
    y_merged = np.zeros(len(bin_centers))
    for rd in per_run:
        rx, ry = rd["x"], rd["y"]
        for xi, yi in zip(rx, ry):
            # Find nearest bin center
            idx = int(np.argmin(np.abs(bin_centers - xi)))
            y_merged[idx] += yi

    # Remove empty bins at the edges
    nonzero = np.where(y_merged > 0)[0]
    if len(nonzero) > 0:
        lo_idx = max(0, nonzero[0] - 2)
        hi_idx = min(len(y_merged), nonzero[-1] + 3)
        bin_centers = bin_centers[lo_idx:hi_idx]
        y_merged = y_merged[lo_idx:hi_idx]

    x_merged = bin_centers
    # Poisson sqrt(y+1): matches the default yerr the simultaneous fit and the
    # auto-fitter use for merged spectra, so all three fit paths weight a
    # merged spectrum identically (code review 2026-06-02, merged-yerr-
    # divergence). The old sqrt(y) here vs sqrt(y+1) elsewhere gave a ~41%
    # weight difference at one count.
    yerr_merged = np.sqrt(y_merged + 1.0)

    out = {
        "x": x_merged,
        "y": y_merged.astype(float),
        "yerr": yerr_merged,
        "per_run": per_run,
        "bin_step_mhz": bin_step,
        "x_unit": x_unit,
    }

    # Surfaced via the fit-report channel: re-bin approximation note, the
    # voltage-merge setpoint-spread warning, and the manual-offset note.
    merge_warnings = _build_merge_warnings(per_run, use_voltage, manual_map)
    if merge_warnings:
        out["merge_warnings"] = merge_warnings
    # The fit-time path reads merge_metadata to Doppler-shift a
    # voltage-merged spectrum. Only meaningful for x_unit=="V"; passing
    # it through for "MHz" merges is harmless audit info and lets
    # downstream code uniformly check "did the user pick a cooler
    # explicitly?" without caring about domain.
    mm = merge_params.get("merge_metadata")
    if mm:
        out["merge_metadata"] = dict(mm)
    return out


# ══════════════════════════════════════════════════════════════════
#  Voltage merge → frequency-fit projection
# ══════════════════════════════════════════════════════════════════

# Spread tolerances above which the user gets a warning that a single
# Doppler shift is a poor approximation for their source files. Picked
# generously enough that "near-identical" runs pass without noise;
# tightened later if the audit log shows too many false negatives.
SPREAD_COOLER_V = 0.5      # V
SPREAD_LASER_CM = 0.0001   # cm^-1


def _build_merge_warnings(per_run, use_voltage, manual_map):
    """Build the ``merge_warnings`` list for a merged spectrum.

    Pure function over the per-run summaries (plain dicts with ``laser_set`` /
    ``cooler_v``) so it is testable without clstools. Covers:

    * MERGE_REBINNED_APPROXIMATE (info, any multi-file merge): the merge
      re-bins per-file histograms rather than pooling raw events, so the
      merged centroid can be biased by up to half a bin;
    * VOLTAGE_MERGE_SETPOINT_SPREAD (warning): a voltage merge of files whose
      laser setpoint / beam energy differ beyond the spread tolerances -- a
      raw-voltage overlay assumes they are identical;
    * MANUAL_OFFSET_IGNORED_VOLTAGE_MERGE (warning): MHz offsets can't apply
      to a voltage axis.
    """
    warnings = []
    if len(per_run) >= 2:
        warnings.append({
            "code": "MERGE_REBINNED_APPROXIMATE",
            "level": "info",
            "run": None,
            "message": (
                "Merged spectrum is re-binned from per-file histograms (raw "
                "events are not pooled before binning). For files scanned at "
                "different steps the merged peak position can be biased by up "
                "to about half a bin."),
        })
    if use_voltage and len(per_run) >= 2:
        lasers = [float(rd.get("laser_set", 0) or 0) for rd in per_run]
        coolers = [float(rd.get("cooler_v", 0) or 0) for rd in per_run]
        laser_spread = max(lasers) - min(lasers)
        cooler_spread = max(coolers) - min(coolers)
        if laser_spread > SPREAD_LASER_CM or cooler_spread > SPREAD_COOLER_V:
            warnings.append({
                "code": "VOLTAGE_MERGE_SETPOINT_SPREAD",
                "level": "warning",
                "run": None,
                "message": (
                    f"Voltage merge of files with differing setpoints (laser "
                    f"spread {laser_spread:.5f} cm^-1, cooler spread "
                    f"{cooler_spread:.3f} V). A raw-voltage overlay assumes an "
                    f"identical laser setpoint and beam energy; the merged "
                    f"spectrum and any Raw-Voltage fit of it may be "
                    f"distorted."),
            })
    if use_voltage and manual_map and any(
            float(v or 0.0) != 0.0 for v in manual_map.values()):
        warnings.append({
            "code": "MANUAL_OFFSET_IGNORED_VOLTAGE_MERGE",
            "level": "warning",
            "run": None,
            "message": (
                f"{sum(1 for v in manual_map.values() if v)} per-file "
                "manual centroid offset(s) were ignored because the merge is "
                "voltage-domain. Offsets are MHz; subtracting MHz from a "
                "voltage axis is dimensionally meaningless. Switch the merge "
                "domain to 'frequency' to apply them."),
        })
    return warnings


def project_voltage_merge_to_frequency(merged_data, source_config):
    """Return (x_mhz, y, yerr, warnings) for a voltage-merged spectrum
    fitted in frequency mode.

    Lets a voltage merge act like a synthetic ASDF: the user picked one
    (cooler, laser, mass, harmonic) at merge time, we apply
    ``voltage_to_frequency_array`` to the binned voltage bin centers,
    and the resulting MHz values feed the fit pipeline as if they had
    come from a normal frequency-binned spectrum.

    The caller checks ``merged_data["x_unit"] == "V"`` and
    ``source_config["bin_mode"] == "Frequency"`` before invoking this
    helper -- a frequency merge already has x in MHz, and a voltage-
    mode fit doesn't need any projection.

    Parameters
    ----------
    merged_data : dict
        From a ``MergedFileEntry`` (or the Analysis SourceBlock's
        merged-entry dict). Must carry ``x``, ``y``, ``yerr``, and
        ``merge_metadata`` (the user-picked cooler/laser/mass/harmonic).
        ``per_run`` is consulted for the spread-warning check; when
        empty the check is skipped silently.
    source_config : dict
        SourceBlock-shaped config. ``ref_freq`` (Hz) is used to shift
        the projected MHz onto the rest-frame transition axis, matching
        the convention ``Compute_WL(ref=ref_freq)`` produces for
        non-merged entries.

    Returns
    -------
    x_mhz : 1-D ndarray
        Rest-frame MHz, one per input bin. Counts/yerr are unchanged.
    y : 1-D ndarray
    yerr : 1-D ndarray
    warnings : list[dict]
        Same shape as ``binning.build_binning_warnings``: each warning
        is ``{"code", "level", "run", "message"}``. Includes a
        per-merge "VOLTAGE_TO_FREQUENCY_PROJECTION" info note and
        any cooler/laser/mass spread warnings.
    """
    from cls_estimations.doppler import voltage_to_frequency_array

    x_v = np.asarray(merged_data["x"], dtype=float)
    y = np.asarray(merged_data["y"], dtype=float)
    yerr = np.asarray(
        merged_data.get("yerr", np.sqrt(y + 1.0)),  # sqrt(y+1): see merge yerr
        dtype=float)

    mm = merged_data.get("merge_metadata") or {}
    # Fall back to per_run means when the user left a merge-level
    # field unset. The MergeDialog converts 0.0 -> None for the three
    # float fields (cooler/laser/mass) before emitting merge_metadata,
    # so `mm["cooler_v"]` is either None or a non-zero float -- the
    # `mm.get(...) or _mean_from_per_run(...)` chain below treats
    # missing/None the same as 0 (both fall through to per_run).
    per_run = merged_data.get("per_run") or []

    def _mean_from_per_run(key):
        vals = [rd.get(key) for rd in per_run
                if isinstance(rd.get(key), (int, float))
                and rd.get(key)]
        return float(np.mean(vals)) if vals else None

    cooler_v = mm.get("cooler_v") or _mean_from_per_run("cooler_v")
    laser_cm = mm.get("laser_sp") or _mean_from_per_run("laser_set") \
        or _mean_from_per_run("laser_sp")
    mass_amu = mm.get("mass_amu") or _mean_from_per_run("mass_amu")
    harmonic = mm.get("harmonic") or _mean_from_per_run("harmonic")
    if not (cooler_v and laser_cm and mass_amu and harmonic):
        raise ValueError(
            "Voltage-merged spectrum needs a complete merge_metadata "
            "(cooler_v, laser_sp, mass_amu, harmonic) to be projected "
            "into frequency space for a fit. Open the merge dialog and "
            "set them, or switch Source 'Bin mode' to 'Raw Voltage'.")

    ref_freq_hz = float(source_config.get("ref_freq", 0.0))
    x_mhz = voltage_to_frequency_array(
        x_v, cooler_v=cooler_v, laser_cm1=laser_cm,
        mass_amu=mass_amu, harmonic=int(harmonic),
        ref_freq_hz=ref_freq_hz)

    name = merged_data.get("merged_name", "merged")
    warnings = [{
        "code": "VOLTAGE_TO_FREQUENCY_PROJECTION",
        "level": "info",
        "run": name,
        "message": (
            f"Voltage-merged spectrum projected into frequency at fit "
            f"time using merge metadata: cooler={cooler_v:.3f} V, "
            f"laser={laser_cm:.6f} cm⁻¹, mass={mass_amu:.4f} amu, "
            f"harmonic={int(harmonic)}."),
    }]

    # Spread checks: source files that disagree on cooler/laser/mass
    # break the single-Doppler-shift assumption. Surface but don't
    # block. Filter out zeros too -- a per_run entry from an ASDF
    # missing VCoolDiv/MassAMU reports 0, which would otherwise
    # explode the spread to the full cooler magnitude and trip a
    # false positive.
    coolers = [rd.get("cooler_v") for rd in per_run
               if isinstance(rd.get("cooler_v"), (int, float))
                  and rd.get("cooler_v")]
    raw_lasers = [rd.get("laser_set") or rd.get("laser_sp")
                  for rd in per_run]
    lasers = [v for v in raw_lasers
              if isinstance(v, (int, float)) and v]
    if len(coolers) >= 2:
        spread = max(coolers) - min(coolers)
        if spread > SPREAD_COOLER_V:
            warnings.append({
                "code": "MERGE_COOLER_SPREAD",
                "level": "warning",
                "run": name,
                "message": (
                    f"Cooler voltage spread across source files is "
                    f"{spread:.3f} V (> {SPREAD_COOLER_V} V); a single "
                    f"Doppler shift may bias the merged centroid."),
            })
    if len(lasers) >= 2:
        spread = max(lasers) - min(lasers)
        if spread > SPREAD_LASER_CM:
            warnings.append({
                "code": "MERGE_LASER_SPREAD",
                "level": "warning",
                "run": name,
                "message": (
                    f"Laser setpoint spread across source files is "
                    f"{spread:.6f} cm⁻¹ (> {SPREAD_LASER_CM} cm⁻¹); a "
                    f"single Doppler shift may bias the merged "
                    f"centroid."),
            })
    return x_mhz, y, yerr, warnings


# ══════════════════════════════════════════════════════════════════
#  Metadata helpers
# ══════════════════════════════════════════════════════════════════

_META_FIELDS = [
    ("run_name",   "Run name",        "str",   "Auto"),
    ("cooler_v",   "Cooler V (raw)",  "float", "Auto"),
    ("laser_set",  "Laser set [cm-1]","float", "Auto"),
    ("dwell_time", "Dwell time",      "float", "Auto"),
    ("experiment", "Experiment",      "str",   "Auto"),
    ("date",       "Date",            "str",   "Auto"),
    ("step_size",  "Step size",       "float", "Auto"),
]


def _auto_metadata(per_run, merged_name):
    """Build default metadata from per-run info."""
    meta = {"run_name": merged_name}
    for key in ["cooler_v", "laser_set", "dwell_time", "step_size"]:
        vals = [r.get(key, 0) for r in per_run if r.get(key)]
        meta[key] = float(np.mean(vals)) if vals else 0.0
    meta["experiment"] = per_run[0].get("experiment", "") if per_run else ""
    meta["date"] = per_run[0].get("date", "") if per_run else ""
    # Union of scanning ranges
    all_ranges = []
    for r in per_run:
        all_ranges.extend(r.get("scanning_ranges", []))
    meta["scanning_ranges"] = all_ranges
    # Cal from first run
    meta["cal_set"] = per_run[0].get("cal_set", []) if per_run else []
    meta["cal_read"] = per_run[0].get("cal_read", []) if per_run else []
    return meta


# ══════════════════════════════════════════════════════════════════
#  Merge Dialog
# ══════════════════════════════════════════════════════════════════

class MergeDialog(QDialog):
    """Dialog for configuring and previewing a multi-run merge.

    Used by both the Analysis Source block (where ``source_config`` is
    the real block's config) and the Pre-Analysis tab (which builds a
    synthetic ``source_config`` from its physics spinners). The dialog
    is responsible for:

    * Picking a merge domain (voltage / frequency) explicitly --
      overrides the ``compute_merged_spectrum`` auto-decision.
    * Collecting merge-level Doppler metadata (cooler/laser/mass/
      harmonic) when the domain is voltage. The fit-time path reads
      these to Doppler-shift the merged voltage axis, treating the
      merged spectrum as a synthetic ASDF.
    * Per-file GP centroid corrections (when supplied).
    """

    def __init__(self, file_entries, source_config, parent=None,
                 existing_result=None,
                 centroid_corrections_map=None,
                 default_domain=None,
                 default_merge_metadata=None,
                 manual_offsets_map=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Runs")
        self.setMinimumWidth(600)
        self.setMinimumHeight(620)

        self._file_entries = file_entries
        self._source_config = source_config
        self._result = existing_result
        # Per-file GP corrections for the merge. None / empty means
        # the dialog should keep the "Align centroids" checkbox
        # disabled (no useful action available); a non-empty dict
        # enables the checkbox.
        self._corrections_map = centroid_corrections_map or {}
        # Manual per-file centroid offsets keyed by path. Always passed
        # through to compute_merged_spectrum -- the offsets are
        # user-driven and additive with the GP path.
        self._manual_offsets_map = manual_offsets_map or {}
        # Pre-filled merge-level Doppler metadata. The caller passes
        # a {cooler_v, laser_sp, mass_amu, harmonic} dict (any subset)
        # so the user lands on sensible numbers rather than zeros.
        self._default_merge_metadata = default_merge_metadata or {}

        layout = QVBoxLayout(self)

        # ── File list ──
        # Annotate each row with any pending manual centroid offset
        # or GP correction coverage so the user can spot per-file
        # adjustments before clicking OK. The annotations are read
        # from the same maps the compute path consumes, so what the
        # user sees is what gets applied.
        files_grp = QGroupBox(
            f"Files to Merge ({len(file_entries)} runs)")
        fl = QVBoxLayout(files_grp)
        self._file_list = QListWidget()
        self._run_nums = []
        for e in file_entries:
            rn = e.get("run_number", "?")
            self._run_nums.append(rn)
            extras = []
            mo = float(
                self._manual_offsets_map.get(e["path"], 0.0) or 0.0)
            if mo:
                extras.append(f"offset {mo:+.2f} MHz")
            gp = self._corrections_map.get(e["path"])
            if gp and gp.get("mode") in ("Auto", "Manual"):
                extras.append(f"GP corr. ({gp.get('mode')})")
            suffix = f"  [{', '.join(extras)}]" if extras else ""
            self._file_list.addItem(f"Run {rn}  —  {e['path']}{suffix}")
        self._file_list.setMaximumHeight(100)
        fl.addWidget(self._file_list)
        layout.addWidget(files_grp)

        # ── Merge parameters ──
        params_grp = QGroupBox("Merge Parameters")
        pf = QFormLayout(params_grp)

        # Domain selector. Default picks frequency when the physics
        # params are set (and the caller didn't force otherwise) so the
        # dialog falls through to the same behavior the legacy
        # auto-decision in compute_merged_spectrum produced.
        self._domain_combo = QComboBox()
        self._domain_combo.addItems(["voltage", "frequency"])
        self._domain_combo.setToolTip(
            "Voltage: merge in raw DAC voltage. The merged spectrum "
            "needs explicit cooler/laser/mass/harmonic to be Doppler-"
            "shifted at fit time -- those defaults are filled in "
            "below.\n"
            "Frequency: each file is Doppler-shifted with its own "
            "cooler/laser before binning, so the merged axis is "
            "rest-frame MHz directly.")
        # Pick the default domain. Explicit caller wins; otherwise
        # auto-decide from the physics params (mass / ref_freq / bin_mode).
        if default_domain in ("voltage", "frequency"):
            idx = self._domain_combo.findText(default_domain)
        else:
            mass = source_config.get("mass", 0)
            ref_freq = source_config.get("ref_freq", 0)
            bin_mode = source_config.get("bin_mode", "Frequency")
            auto_voltage = (
                bin_mode == "Raw Voltage" or ref_freq == 0 or mass <= 0)
            idx = self._domain_combo.findText(
                "voltage" if auto_voltage else "frequency")
        if idx >= 0:
            self._domain_combo.setCurrentIndex(idx)
        self._domain_combo.currentTextChanged.connect(
            self._on_domain_changed)
        pf.addRow("Merge domain:", self._domain_combo)

        step_row = QHBoxLayout()
        self._step_mode = QComboBox()
        self._step_mode.addItems(["Auto", "Manual"])
        self._step_mode.currentTextChanged.connect(self._on_step_mode)
        step_row.addWidget(self._step_mode)
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setLocale(QLocale(QLocale.Language.English,
                                          QLocale.Country.UnitedStates))
        self._step_spin.setRange(0.1, 1000.0)
        self._step_spin.setSingleStep(1.0)
        self._step_spin.setValue(20.0)
        self._step_spin.setEnabled(False)
        step_row.addWidget(self._step_spin)
        # Sibling label that explains what Auto picks (the median Δx
        # across the source files, computed at merge time). Updated
        # by _on_step_mode so a disabled spin isn't the only feedback.
        self._step_hint = QLabel(
            "<i>median Δx</i>")
        self._step_hint.setStyleSheet("color:#888; font-size: 9pt;")
        step_row.addWidget(self._step_hint, 1)
        pf.addRow("Bin step:", step_row)

        self._cooler_corr = QComboBox()
        self._cooler_corr.addItems(["pbp", "mean"])
        idx = self._cooler_corr.findText(
            source_config.get("cooler_correction", "pbp"))
        if idx >= 0:
            self._cooler_corr.setCurrentIndex(idx)
        pf.addRow("Cooler correction:", self._cooler_corr)

        self._centroid_corr = QCheckBox("Align centroids before merging")
        # Enable only when the SourceBlock supplied per-file
        # corrections from the IS-tab Reference Correction panel. A
        # disabled-but-clearly-explained checkbox is less misleading
        # than an enabled one that silently no-ops.
        n_avail = sum(
            1 for fp in (e.get("path") for e in file_entries)
            if (self._corrections_map.get(fp) or {}).get("mode")
                in ("Auto", "Manual"))
        n_total = len(file_entries)
        if n_avail > 0 and n_avail == n_total:
            # Full coverage: pre-check and explain.
            self._centroid_corr.setEnabled(True)
            self._centroid_corr.setChecked(True)
            self._centroid_corr.setToolTip(
                "Subtract the GP-derived per-file centroid "
                "correction from each file's MHz axis before "
                "histogramming.")
            pf.addRow(self._centroid_corr)
        elif n_avail > 0:
            # Partial coverage: enable but DON'T pre-check, warn the
            # user. Mixing corrected and uncorrected files in a single
            # merge can bias the merged centroid.
            self._centroid_corr.setEnabled(True)
            self._centroid_corr.setChecked(False)
            self._centroid_corr.setText(
                "Align centroids before merging  (partial coverage)")
            self._centroid_corr.setToolTip(
                f"Corrections available for {n_avail} of {n_total} "
                f"files. Enabling will apply the GP correction only "
                f"to those files; the remaining "
                f"{n_total - n_avail} pass through unshifted, which "
                f"can bias the merged centroid. Recommended: refit "
                f"the GP for all files first, then re-merge.")
            pf.addRow(self._centroid_corr)
            uncorrected = [
                os.path.basename(e["path"]) for e in file_entries
                if (self._corrections_map.get(e["path"]) or {}
                    ).get("mode") not in ("Auto", "Manual")]
            warn_lbl = QLabel(
                "<span style='color:#c97800'>"
                f"⚠ {n_total - n_avail} file(s) without correction: "
                + ", ".join(uncorrected[:5])
                + (f", +{len(uncorrected) - 5} more"
                   if len(uncorrected) > 5 else "")
                + "</span>")
            warn_lbl.setWordWrap(True)
            pf.addRow("", warn_lbl)
        else:
            self._centroid_corr.setEnabled(False)
            self._centroid_corr.setToolTip(
                "No per-file corrections available. Fit the GP in "
                "the Isotope Shifts tab and click 'Compute per-file "
                "corrections' for this project's runs first.")
            pf.addRow(self._centroid_corr)
        layout.addWidget(params_grp)

        # ── Merge-level Doppler metadata (voltage merges only) ──
        # These Doppler-shift the merged voltage axis at fit time. For
        # a frequency merge each file is already shifted by its own
        # (cooler, laser) before binning, so the merge-level values are
        # not used downstream -- the panel is hidden in that case.
        # Defaults come from the caller (e.g. PA passes the mean of
        # source files; SourceBlock can pass its current override or 0).
        self._merge_meta_grp = QGroupBox(
            "Merge-level Doppler metadata (voltage merges)")
        self._merge_meta_grp.setToolTip(
            "Used to Doppler-shift the merged voltage axis at fit time. "
            "Treat the merged spectrum as a synthetic ASDF with these "
            "physics parameters. Source files with significantly "
            "different values raise a spread warning at fit time but "
            "don't block the merge.")
        mmf = QFormLayout(self._merge_meta_grp)
        d = self._default_merge_metadata
        self._merge_cooler = QDoubleSpinBox()
        self._merge_cooler.setLocale(QLocale(QLocale.Language.English,
                                              QLocale.Country.UnitedStates))
        self._merge_cooler.setRange(0.0, 1e7)
        self._merge_cooler.setDecimals(3)
        self._merge_cooler.setValue(float(d.get("cooler_v") or 0.0))
        mmf.addRow("Cooler V:", self._merge_cooler)
        self._merge_laser = QDoubleSpinBox()
        self._merge_laser.setLocale(QLocale(QLocale.Language.English,
                                             QLocale.Country.UnitedStates))
        self._merge_laser.setRange(0.0, 1e6)
        self._merge_laser.setDecimals(6)
        self._merge_laser.setValue(float(d.get("laser_sp") or 0.0))
        mmf.addRow("Laser setpoint (cm⁻¹):", self._merge_laser)
        self._merge_mass = QDoubleSpinBox()
        self._merge_mass.setLocale(QLocale(QLocale.Language.English,
                                            QLocale.Country.UnitedStates))
        self._merge_mass.setRange(0.0, 1e4)
        self._merge_mass.setDecimals(6)
        self._merge_mass.setValue(float(d.get("mass_amu") or 0.0))
        mmf.addRow("Mass (amu):", self._merge_mass)
        self._merge_harmonic = QSpinBox()
        self._merge_harmonic.setRange(1, 32)
        self._merge_harmonic.setValue(int(d.get("harmonic") or 2))
        mmf.addRow("Harmonic:", self._merge_harmonic)
        layout.addWidget(self._merge_meta_grp)
        # Initial visibility follows the domain default.
        self._on_domain_changed(self._domain_combo.currentText())

        # ── Metadata ──
        meta_grp = QGroupBox("Metadata (for ASDF export)")
        meta_scroll = QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_scroll.setMaximumHeight(200)
        meta_w = QWidget()
        mf = QFormLayout(meta_w)
        mf.setVerticalSpacing(6)

        self._meta_widgets = {}  # {field_key: (combo, edit)}
        run_labels = [f"Run {rn}" for rn in self._run_nums]
        for key, label, ftype, default in _META_FIELDS:
            row = QHBoxLayout()
            combo = QComboBox()
            combo.addItems(["Auto"] + run_labels + ["Custom"])
            combo.setFixedWidth(120)
            edit = QLineEdit()
            edit.setReadOnly(True)
            combo.currentTextChanged.connect(
                lambda txt, k=key, e=edit: self._on_meta_source_changed(
                    k, txt, e))
            row.addWidget(combo)
            row.addWidget(edit, 1)
            mf.addRow(f"{label}:", row)
            self._meta_widgets[key] = (combo, edit)

        meta_scroll.setWidget(meta_w)
        ml = QVBoxLayout(meta_grp)
        ml.addWidget(meta_scroll)
        layout.addWidget(meta_grp)

        # ── Preview ──
        preview_btn = QPushButton("Preview Merge")
        preview_btn.clicked.connect(self._preview)
        layout.addWidget(preview_btn)

        # ── Buttons ──
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Populate default metadata
        if existing_result:
            self._populate_from_existing(existing_result)

    def _on_step_mode(self, text):
        self._step_spin.setEnabled(text == "Manual")
        # Echo the unit on both the spin (tooltip) and the sibling
        # hint label so the user sees it whether the spin is enabled
        # or grayed-out.
        unit = "V" if self._domain_combo.currentText() == "voltage" \
            else "MHz"
        if text == "Manual":
            self._step_spin.setToolTip(f"Bin step in {unit}")
            self._step_hint.setText(f"<i>{unit}</i>")
        else:
            self._step_spin.setToolTip("Bin step (Auto)")
            self._step_hint.setText(
                f"<i>median Δx across sources ({unit})</i>")

    def _on_domain_changed(self, text):
        """Show the merge-level Doppler metadata only for voltage
        merges -- frequency merges per-file-shift before binning."""
        self._merge_meta_grp.setVisible(text == "voltage")
        # Propagate the unit hint to the step spin tooltip.
        self._on_step_mode(self._step_mode.currentText())

    def _on_meta_source_changed(self, key, source_text, edit):
        if source_text == "Custom":
            edit.setReadOnly(False)
            return
        edit.setReadOnly(True)
        if source_text == "Auto":
            edit.setText("(computed on merge)")
        else:
            # "Run XXXX" → find the matching per_run entry
            for rn, fe in zip(self._run_nums, self._file_entries):
                if f"Run {rn}" == source_text:
                    # We'd need to load the file to get the value;
                    # for now show the path hint
                    edit.setText(f"from {fe['path']}")
                    break

    def _get_merge_params(self):
        step = None
        if self._step_mode.currentText() == "Manual":
            step = self._step_spin.value()
        params = {
            "bin_step_mhz": step,
            "cooler_correction": self._cooler_corr.currentText(),
            "centroid_correction": self._centroid_corr.isChecked(),
            "domain": self._domain_combo.currentText(),
        }
        # The fit-time path only consults merge_metadata for voltage
        # merges. Always emit the dict so downstream code can detect
        # "user explicitly set these" vs "left at default" by checking
        # the values rather than the key's presence.
        params["merge_metadata"] = {
            "cooler_v": float(self._merge_cooler.value()) or None,
            "laser_sp": float(self._merge_laser.value()) or None,
            "mass_amu": float(self._merge_mass.value()) or None,
            "harmonic": int(self._merge_harmonic.value()),
        }
        return params

    def _compute(self):
        paths = [e["path"] for e in self._file_entries]
        # Pass per-file binning overrides so the merge honors per-file
        # bin_mode flips that the rest of the pipeline already applies.
        file_overrides = {
            e["path"]: e["binning_override"]
            for e in self._file_entries
            if e.get("binning_override")
        }
        return compute_merged_spectrum(
            paths, self._source_config, self._get_merge_params(),
            file_overrides=file_overrides,
            centroid_corrections_map=self._corrections_map,
            manual_offsets_map=self._manual_offsets_map)

    def _preview(self):
        try:
            result = self._compute()
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", str(e))
            return

        from gui.shared_widgets import get_plot_type_settings
        ps = get_plot_type_settings("preview")
        fig = Figure(figsize=(ps["figsize_w"], ps["figsize_h"]))
        ax = fig.add_subplot(111)

        # Individual runs faded (label includes laser/cooler for verification)
        for rd in result["per_run"]:
            laser = rd.get("laser_set", "?")
            cooler = rd.get("cooler_v", "?")
            ax.plot(rd["x"], rd["y"], '.', alpha=0.3, ms=3,
                    label=f"Run {rd['run_num']} "
                          f"(laser={laser}, cooler={cooler})")

        # Merged in solid. Lower error bar clipped at y=0 (counts).
        from gui.shared_widgets import count_errorbar_pairs
        ax.errorbar(result["x"], result["y"],
                     yerr=count_errorbar_pairs(
                         result["y"], result["yerr"]),
                     fmt='.', color="black", ms=4, capsize=2,
                     lw=1.5, label="Merged")

        x_unit = result.get("x_unit", "MHz")
        x_label = "Frequency [MHz]" if x_unit == "MHz" else "Voltage [V]"
        ax.set_xlabel(x_label, fontsize=ps["label_size"])
        ax.set_ylabel("Counts", fontsize=ps["label_size"])
        ax.set_title("Merge Preview", fontsize=ps["title_size"])
        ax.legend(fontsize=8)

        from gui.analysis.helpers import PopupPlotWindow
        win = PopupPlotWindow(fig, title="Merge Preview", parent=self)
        win.show()
        self._result = result

    def _build_metadata(self, result):
        """Build metadata dict from user selections."""
        per_run = result["per_run"]
        merged_name = "merged_" + "_".join(
            str(r["run_num"]) for r in per_run)
        auto = _auto_metadata(per_run, merged_name)

        meta = {}
        for key, label, ftype, default in _META_FIELDS:
            combo, edit = self._meta_widgets[key]
            src = combo.currentText()
            if src == "Custom":
                val = edit.text()
                if ftype == "float":
                    try:
                        val = float(val)
                    except ValueError:
                        val = 0.0
                meta[key] = val
            elif src == "Auto":
                meta[key] = auto.get(key, "")
            else:
                # "Run XXXX" → use that run's metadata
                for rd in per_run:
                    if f"Run {rd['run_num']}" == src:
                        meta[key] = rd.get(key, auto.get(key, ""))
                        break

        meta["scanning_ranges"] = auto["scanning_ranges"]
        meta["cal_set"] = auto["cal_set"]
        meta["cal_read"] = auto["cal_read"]
        return meta

    def _on_accept(self):
        try:
            if self._result is None:
                self._result = self._compute()
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", str(e))
            return

        meta = self._build_metadata(self._result)
        self._result["metadata"] = meta
        self._result["merged_name"] = meta.get(
            "run_name",
            "merged_" + "_".join(
                str(r["run_num"]) for r in self._result["per_run"]))
        self._result["source_files"] = [e["path"]
                                         for e in self._file_entries]
        self._result["source_runs"] = list(self._run_nums)
        self._result["merge_params"] = self._get_merge_params()
        self.accept()

    def get_result(self):
        return self._result

    def _populate_from_existing(self, existing):
        mp = existing.get("merge_params", {})
        step = mp.get("bin_step_mhz")
        if step and step > 0:
            self._step_mode.setCurrentText("Manual")
            self._step_spin.setValue(step)
        cc = mp.get("cooler_correction", "pbp")
        idx = self._cooler_corr.findText(cc)
        if idx >= 0:
            self._cooler_corr.setCurrentIndex(idx)
        self._centroid_corr.setChecked(mp.get("centroid_correction", False))
        # Restore domain + merge-level Doppler metadata.
        # ``merge_params["domain"]`` was written by _get_merge_params on
        # the original Accept; falling back to the x_unit-based guess
        # keeps editing older merged_data (saved without a domain) working.
        domain = mp.get("domain")
        if domain not in ("voltage", "frequency"):
            domain = (
                "voltage" if existing.get("x_unit") == "V"
                else "frequency")
        idx = self._domain_combo.findText(domain)
        if idx >= 0:
            self._domain_combo.setCurrentIndex(idx)
        mm = mp.get("merge_metadata") or existing.get("merge_metadata") or {}
        if mm.get("cooler_v") is not None:
            self._merge_cooler.setValue(float(mm["cooler_v"]))
        if mm.get("laser_sp") is not None:
            self._merge_laser.setValue(float(mm["laser_sp"]))
        if mm.get("mass_amu") is not None:
            self._merge_mass.setValue(float(mm["mass_amu"]))
        if mm.get("harmonic") is not None:
            self._merge_harmonic.setValue(int(mm["harmonic"]))
        # Restore per-file manual centroid offsets from the existing
        # per_run audit. Without this, Edit-Merge silently drops
        # user-set offsets on the second run.
        per_run = existing.get("per_run") or []
        for rd in per_run:
            mo = rd.get("manual_offset_mhz")
            if mo:
                path = rd.get("path")
                if path:
                    self._manual_offsets_map[path] = float(mo)
        # Apply visibility based on restored domain.
        self._on_domain_changed(self._domain_combo.currentText())


# ══════════════════════════════════════════════════════════════════
#  Merge View Dialog
# ══════════════════════════════════════════════════════════════════

class MergeViewDialog(QDialog):
    """Multi-tab viewer for a merged run.

    ``source_config``: when the caller passes the Source-block's config
    and ``bin_mode == "Frequency"`` while the merged data is in voltage,
    the Spectrum tab Doppler-projects the x-axis using
    ``project_voltage_merge_to_frequency`` so the preview matches what
    the fit will see. Without this the user flips bin_mode to Frequency,
    opens the preview, and still sees a voltage axis even though the fit
    is going to project.
    """

    def __init__(self, merged_data, parent=None, source_config=None):
        super().__init__(parent)
        self.setWindowTitle(f"View: {merged_data.get('merged_name', '?')}")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        from gui.shared_widgets import get_plot_type_settings
        ps = get_plot_type_settings("preview")

        # ── Spectrum tab ──
        # If the Source block is asking for a frequency-domain fit on
        # this voltage merge, project the x-axis here too so what the
        # user previews matches what the fitter sees. Counts/yerr
        # unchanged. Falls back to the stored x_unit when no
        # source_config is supplied (menu-driven View… stays in
        # whatever the merge was stored in).
        merged_x = merged_data["x"]
        merged_y = merged_data["y"]
        merged_yerr = merged_data.get(
            "yerr", np.sqrt(merged_data["y"] + 1))
        display_x_unit = merged_data.get("x_unit", "MHz")
        if (source_config is not None
                and merged_data.get("x_unit") == "V"
                and source_config.get("bin_mode") == "Frequency"):
            try:
                proj_x, proj_y, proj_yerr, _w = (
                    project_voltage_merge_to_frequency(
                        merged_data, source_config))
                merged_x = proj_x
                merged_y = proj_y
                merged_yerr = proj_yerr
                display_x_unit = "MHz"
            except ValueError:
                # Missing merge_metadata. Fall through to the stored
                # voltage view; the badge tooltip on the entry row
                # tells the user why.
                pass

        spec_fig = Figure(figsize=(ps["figsize_w"], ps["figsize_h"]))
        ax = spec_fig.add_subplot(111)
        # Per-source x/y arrays only exist for merges produced by the
        # Analysis-side ``compute_merged_spectrum`` (which retains
        # per-file binned spectra in per_run). PA-imported merges
        # carry per-source metadata but no x/y -- skip the per-run
        # overlay there so the dialog still shows the merged spectrum
        # cleanly (the TOF tab guards on ``rd.get(...)`` the same way).
        for rd in merged_data.get("per_run", []):
            rx = rd.get("x")
            ry = rd.get("y")
            if rx is None or ry is None or len(rx) == 0:
                continue
            ax.plot(rx, ry, '.', alpha=0.25, ms=3,
                    label=f"Run {rd.get('run_num', '?')}")
        # Clip the lower error bar at y=0 -- counts are non-negative.
        from gui.shared_widgets import count_errorbar_pairs
        ax.errorbar(merged_x, merged_y,
                     yerr=count_errorbar_pairs(merged_y, merged_yerr),
                     fmt='.', color="black", ms=4, capsize=2,
                     lw=1.5, label="Merged")
        x_label = ("Frequency [MHz]" if display_x_unit == "MHz"
                   else "Voltage [V]")
        ax.set_xlabel(x_label, fontsize=ps["label_size"])
        ax.set_ylabel("Counts", fontsize=ps["label_size"])
        title = "Merged Spectrum"
        if (source_config is not None
                and merged_data.get("x_unit") == "V"
                and display_x_unit == "MHz"):
            title += "  (V→F projected for Frequency-mode fit)"
        ax.set_title(title, fontsize=ps["title_size"])
        ax.legend(fontsize=8)
        spec_fig.tight_layout()
        spec_canvas = FigureCanvasQTAgg(spec_fig)
        tabs.addTab(spec_canvas, "Spectrum")

        # ── TOF tab ──
        tof_fig = Figure(figsize=(ps["figsize_w"], ps["figsize_h"]))
        ax_tof = tof_fig.add_subplot(111)
        for rd in merged_data.get("per_run", []):
            ti = rd.get("tof_index", np.array([]))
            tc = rd.get("tof_counts", np.array([]))
            if len(ti) > 0:
                ax_tof.step(ti, tc, alpha=0.3, lw=1,
                            label=f"Run {rd['run_num']}")
        ax_tof.set_xlabel("Time of Flight", fontsize=ps["label_size"])
        ax_tof.set_ylabel("Counts", fontsize=ps["label_size"])
        ax_tof.set_title("TOF Histograms", fontsize=ps["title_size"])
        ax_tof.legend(fontsize=8)
        tof_fig.tight_layout()
        tof_canvas = FigureCanvasQTAgg(tof_fig)
        tabs.addTab(tof_canvas, "TOF")

        # ── Info tab ──
        info_lines = [f"<b>Merged Name:</b> {merged_data.get('merged_name')}",
                      f"<b>Runs:</b> {merged_data.get('source_runs')}",
                      f"<b>Bin step:</b> {merged_data.get('bin_step_mhz') or 0:.2f} MHz",
                      f"<b>Total bins:</b> {len(merged_data.get('x', []))}",
                      f"<b>Total counts:</b> {int(np.sum(merged_data.get('y', [])))}",
                      ""]
        for rd in merged_data.get("per_run", []):
            info_lines.append(
                f"<b>Run {rd['run_num']}:</b> "
                f"{rd.get('n_events', '?')} events, "
                f"laser={rd.get('laser_set', '?')}, "
                f"date={rd.get('date', '?')}")
        meta = merged_data.get("metadata", {})
        if meta:
            info_lines.append("<hr><b>Export Metadata:</b>")
            for k, v in meta.items():
                info_lines.append(f"  {k}: {v}")

        from gui.analysis.helpers import _show_scrollable_info
        info_w = QWidget()
        info_l = QVBoxLayout(info_w)
        info_label = QLabel("<br>".join(info_lines))
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(info_label)
        info_l.addWidget(scroll)
        tabs.addTab(info_w, "Info")

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


# ══════════════════════════════════════════════════════════════════
#  ASDF Export
# ══════════════════════════════════════════════════════════════════

#: Schema marker for DENIS merged-spectrum ASDFs. Bumped on any
#: incompatible change to the on-disk layout. Loaders should check
#: ``tree.get("denis_schema")`` before trusting any other field --
#: an ASDF without this key is a regular clstools run, not a merge.
DENIS_MERGED_SCHEMA = "denis_merged_v1"


def export_merged_asdf(merged_data, output_path):
    """Write a merged spectrum as a DENIS-labelled ASDF.

    Stores the binned histogram directly (``x``, ``y``, ``yerr``)
    plus the merge-level metadata and per-source audit, NOT
    synthesised raw events. Reasons:

    * Clstools' ``Load_Run`` runs ``Compute_Voltages`` which does a
      cal-polynomial fit; synthesising events with a 2-point CalSet
      yielded "number of data points must exceed order to scale the
      covariance matrix" errors on re-load.
    * Synthesised one-row-per-count files were ~1000× larger than
      the histogram they represented.
    * All synthesised events had ``TS=TOF=0`` which silently broke
      anything that gates on TOF or computes rates from timestamps.

    Files written by this function are recognised by
    :func:`load_merged_asdf`. The PA "Open…" path and the Analysis
    Source-block "+ Add Files" path both peek for the
    ``denis_schema`` marker and route a merged ASDF through the
    merged-entry flow instead of ``cls.CLSDataFrame.Load_Run``.
    """
    import asdf

    meta = merged_data.get("metadata", {}) or {}
    per_run = merged_data.get("per_run", []) or []
    merge_metadata = merged_data.get("merge_metadata", {}) or {}

    # Strip the GUI back-reference (a QCheckBox + QWidget) that
    # _add_merged_entry stashes on merged_data; ASDF can't serialize
    # Qt objects. Same problem the fit-submit path solved at
    # blocks.py for the multiprocessing pickle.
    safe_meta = {k: v for k, v in meta.items() if k != "_entry"}

    # Drop x/y arrays from per_run -- they're per-source histograms
    # we no longer need for round-trip (the merged x/y is what gets
    # fit), and they bloat the file. Keep the audit fields.
    audit_per_run = []
    for rd in per_run:
        clean = {k: v for k, v in rd.items()
                 if k not in ("x", "y") and not callable(v)}
        audit_per_run.append(clean)

    # Consistency guard: catch the historical bin_mode/domain
    # mismatch bug (voltage-axis data stamped x_unit="MHz") before
    # we serialize it to disk. A genuine frequency-domain merge of
    # a CLS scan spans thousands of MHz; voltage-axis values stamped
    # as MHz would land within ±1000 with very high probability.
    # Refuse to write such files so the user re-merges with current
    # code instead of carrying stale state to disk.
    x_arr = np.asarray(merged_data["x"], dtype=float)
    declared_unit = merged_data.get("x_unit", "MHz")
    if declared_unit == "MHz" and len(x_arr) > 0:
        x_span = float(np.max(np.abs(x_arr)))
        if x_span < 1000.0:
            raise ValueError(
                "Refusing to export merged ASDF: merged_data["
                "'x_unit'] is 'MHz' but |x| stays under 1000 -- "
                "this looks like the legacy bin_mode/domain "
                "mismatch bug (voltage-axis data labelled as "
                "frequency). Re-merge with current code, then "
                "Export. If this is a real low-detuning frequency "
                "merge, set the unit explicitly downstream.")

    tree = {
        "denis_schema":     DENIS_MERGED_SCHEMA,
        "merged_name":      merged_data.get("merged_name", "merged"),
        "domain":           ("voltage"
                             if merged_data.get("x_unit") == "V"
                             else "frequency"),
        "x":                x_arr,
        "y":                np.asarray(merged_data["y"], dtype=float),
        "yerr":             np.asarray(merged_data["yerr"], dtype=float),
        "x_unit":           declared_unit,
        "bin_step":         float(merged_data.get("bin_step_mhz", 0)),
        "merge_metadata":   {
            "cooler_v": merge_metadata.get("cooler_v"),
            "laser_sp": merge_metadata.get("laser_sp"),
            "mass_amu": merge_metadata.get("mass_amu"),
            "harmonic": merge_metadata.get("harmonic"),
        },
        "source_runs":      list(merged_data.get("source_runs", []) or []),
        "source_files":     list(merged_data.get("source_files", []) or []),
        "per_run_audit":    audit_per_run,
        "merge_params":     merged_data.get("merge_params", {}) or {},
        "metadata":         safe_meta,
    }
    af = asdf.AsdfFile(tree)
    af.write_to(output_path)


def _materialise_per_run(rd):
    """Deep-copy a per-run dict, eagerly converting any lazy ASDF
    ndarray values into in-memory numpy arrays so they survive the
    ``with asdf.open(...)`` block closing. Non-array values pass
    through unchanged.
    """
    out = {}
    for k, v in (rd or {}).items():
        # asdf's ndarray proxy fails the bare ``isinstance(v, np.ndarray)``
        # check, so probe for the .__array__ interface (which both
        # numpy and asdf's ndarray support) and force evaluation.
        if hasattr(v, "__array__") and not isinstance(
                v, (str, bytes, int, float, bool, list, tuple, dict)):
            try:
                out[k] = np.array(v)
                continue
            except Exception:
                pass
        out[k] = v
    return out


def load_merged_asdf(path):
    """Return ``merged_data`` for a DENIS-labelled merged ASDF, or
    ``None`` if the file isn't one (regular clstools run, malformed,
    or missing).

    The dict shape matches what ``compute_merged_spectrum`` produces
    and what ``_add_merged_entry`` (Analysis) / ``MergedFileEntry``
    (PA) consume, so callers can route directly into the existing
    merged-entry flow without further translation.
    """
    import asdf

    try:
        # ASDF lazy-loads arrays -- accessing them after the file is
        # closed raises "ASDF file has already been closed". Copy
        # every array we care about *inside* the with block to
        # detach from the file handle.
        with asdf.open(path) as af:
            if af.tree.get("denis_schema") != DENIS_MERGED_SCHEMA:
                return None
            mm = dict(af.tree.get("merge_metadata") or {})
            out = {
                "merged_name":    af.tree.get("merged_name", "merged"),
                "x":              np.array(af.tree.get("x", []),
                                            dtype=float),
                "y":              np.array(af.tree.get("y", []),
                                            dtype=float),
                "yerr":           np.array(af.tree.get("yerr", []),
                                            dtype=float),
                "x_unit":         af.tree.get("x_unit", "MHz"),
                "bin_step_mhz":   float(af.tree.get("bin_step", 0)),
                "merge_metadata": {
                    "cooler_v": mm.get("cooler_v"),
                    "laser_sp": mm.get("laser_sp"),
                    "mass_amu": mm.get("mass_amu"),
                    "harmonic": mm.get("harmonic"),
                },
                "source_runs":    list(af.tree.get("source_runs", [])
                                       or []),
                "source_files":   list(af.tree.get("source_files", [])
                                       or []),
                # per_run_audit may contain lazy ASDF arrays
                # (tof_index, tof_counts, etc.). Shallow-copying the
                # dict keeps those arrays bound to the file handle;
                # accessing them after the ``with`` block exits
                # raises "Attempt to read block data from missing
                # block". Materialise every array-shaped value here.
                "per_run":        [_materialise_per_run(rd) for rd in
                                   (af.tree.get("per_run_audit") or [])],
                "merge_params":   dict(af.tree.get("merge_params", {})
                                       or {}),
                "metadata":       dict(af.tree.get("metadata", {})
                                       or {}),
            }
    except Exception:
        return None
    # Consistency guard for legacy buggy exports: x_unit="MHz" with
    # voltage-axis values. A real frequency-domain CLS merge spans
    # thousands of MHz; |x|<1000 means voltage values got mislabelled
    # by the pre-fix bin_mode/domain mismatch bug. Raise so the
    # caller surfaces a clear "re-merge required" message instead of
    # rendering wrong physics.
    if (out.get("x_unit") == "MHz" and len(out["x"]) > 0
            and float(np.max(np.abs(out["x"]))) < 1000.0):
        raise ValueError(
            f"DENIS merged ASDF at {path} has x_unit='MHz' but "
            f"|x| < 1000, which is the legacy bin_mode/domain "
            f"mismatch bug (voltage-axis data labelled as "
            f"frequency). Re-merge with current code and re-export "
            f"to produce a valid file.")
    return out


def is_merged_asdf(path):
    """Cheap check: does ``path`` look like a DENIS merged ASDF?

    Used by the file-open paths to decide whether to route to the
    merged-entry flow vs the standard ``Load_Run``. Returns False
    for any error -- callers fall through to clstools on False.
    """
    import asdf

    try:
        with asdf.open(path) as af:
            return af.tree.get("denis_schema") == DENIS_MERGED_SCHEMA
    except Exception:
        return False
