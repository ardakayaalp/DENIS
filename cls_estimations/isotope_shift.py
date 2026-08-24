"""Isotope shift calculation, error propagation, and systematic estimation.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Extracts fitted centroids, forms inverse-variance (and Birge-inflated)
weighted averages, propagates the GP reference-correction systematic
through single- and cross-isotope covariances, estimates the
cooler/laser voltage systematic, and assembles the full isotope-shift
table relative to a chosen reference. Implements Phase-5 of the
analysis pipeline's uncertainty budget.

Depends on: cls_estimations.doppler (Doppler-shift formulae for the
voltage/laser systematic).
"""

import warnings

import numpy as np
from .doppler import beta_from_voltage, nu_seen_by_ion


def extract_centroid(params_df_dict, model_name=None, source_name=None):
    """Extract centroid value and uncertainty from a params_df dict.

    Parameters
    ----------
    params_df_dict : dict
        From ``result["params_df"]`` (DataFrame.to_dict("list")).
        Expected columns: Parameter, Value, Stderr (or Error), Model,
        Source.
    model_name : str or None
        If given, filter to rows matching this model name.
    source_name : str or None
        If given, filter to rows matching this source name. Required to
        disambiguate simultaneous fits where every per-run result shares
        one global ``params_df`` containing centroids for all sources.

    Returns
    -------
    (centroid_value, centroid_uncertainty) : tuple of float
        Both in MHz.  Returns ``(None, None)`` if centroid not found.
    """
    params = params_df_dict.get("Parameter", [])
    values = params_df_dict.get("Value", [])
    # satlas2 uses "Stderr"; project.py renames to "Error" in CSV export
    errors = params_df_dict.get("Stderr", params_df_dict.get("Error", []))
    models = params_df_dict.get("Model", [None] * len(params))
    sources = params_df_dict.get("Source", [None] * len(params))

    for i, p in enumerate(params):
        if p != "centroid":
            continue
        if model_name is not None and i < len(models):
            if models[i] is not None and models[i] != model_name:
                continue
        if source_name is not None and i < len(sources):
            if sources[i] is not None and sources[i] != source_name:
                continue
        val = values[i] if i < len(values) else None
        err = errors[i] if i < len(errors) else None
        if val is None:
            continue
        return float(val), float(err) if err is not None else 0.0

    return None, None


def weighted_average(values, uncertainties):
    """Inverse-variance weighted average.

    Parameters
    ----------
    values : array-like of float
    uncertainties : array-like of float (must be > 0)

    Returns
    -------
    (weighted_mean, weighted_uncertainty) : tuple of float
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(uncertainties, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), float(s[0])
    # Replace zero uncertainties with a small but stable floor
    s = np.where(s > 0, s, np.median(s[s > 0]) * 0.01
                 if np.any(s > 0) else 1.0)
    w = 1.0 / s**2
    w_sum = np.sum(w)
    mean = np.sum(w * v) / w_sum
    sigma = 1.0 / np.sqrt(w_sum)
    return float(mean), float(sigma)


def weighted_average_with_birge(values, uncertainties):
    """Inverse-variance weighted average plus Birge-ratio inflation.

    When the per-input ``σᵢ`` understate the actual scatter (reduced
    χ² of the inputs about the weighted mean is > 1), inflate the
    statistical uncertainty by the Birge ratio = √χ²_red. This
    matches SATLAS2's ``weightedAverage`` and the standard PDG
    treatment of inconsistent measurements.

    Parameters
    ----------
    values : array-like of float
    uncertainties : array-like of float (must be > 0)

    Returns
    -------
    mean : float
        Inverse-variance weighted mean.
    sigma_fit : float
        ``√(1/Σwᵢ)`` -- pure inverse-variance prediction (the
        "statistical" uncertainty before any inflation).
    sigma_scatter : float
        Excess uncertainty contributed by Birge inflation, i.e.
        ``√(sigma_total² - sigma_fit²)`` where
        ``sigma_total = sigma_fit · max(1, √χ²_red)``. Zero when the
        inputs are mutually consistent (χ²_red ≤ 1).
    birge : float
        ``max(1, √χ²_red)`` -- the inflation factor itself.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(uncertainties, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0.0, 1.0
    if len(v) == 1:
        return float(v[0]), float(s[0]), 0.0, 1.0
    s = np.where(s > 0, s, np.median(s[s > 0]) * 0.01
                 if np.any(s > 0) else 1.0)
    w = 1.0 / s ** 2
    w_sum = float(np.sum(w))
    mean = float(np.sum(w * v) / w_sum)
    sigma_fit = float(1.0 / np.sqrt(w_sum))
    chi2_red = float(np.sum(w * (v - mean) ** 2) / (len(v) - 1))
    birge = max(1.0, float(np.sqrt(max(chi2_red, 0.0))))
    sigma_total = sigma_fit * birge
    sigma_scatter = float(np.sqrt(max(sigma_total ** 2 - sigma_fit ** 2,
                                       0.0)))
    return mean, sigma_fit, sigma_scatter, birge


def gp_propagated_sigma(corrector, t_hours, weights, *,
                         total_weight=None):
    """Single-isotope σ contribution from a GP correction propagated
    through a weighted average of runs.

    For a weighted mean ``c̄ = Σ_full wᵢ cᵢ / W_total`` where the
    GP correction was applied to a (possibly proper) subset of the
    runs, the correction term contributes
    ``Var(c̄_corr) = (1/W_total²) · w_subsetᵀ K_GP w_subset``.
    Runs without timestamps were not corrected at fit time, so
    they enter ``c̄`` un-shifted and contribute zero variance from
    the GP -- but they DO contribute to the normalizing W_total.

    Parameters
    ----------
    corrector : ReferenceCorrector
        Must already be fit. Only ``corrector.cov(t1, t2)`` is used.
    t_hours : array-like of float
        Per-run timestamps for the SUBSET of runs that were
        corrected (in the same units the corrector was trained).
    weights : array-like of float
        Per-run weights for the SAME subset (``wᵢ = 1/σᵢ²``).
    total_weight : float, optional
        ``Σ wⱼ`` over ALL runs that contributed to the centroid
        (corrected and uncorrected). Defaults to ``sum(weights)``,
        which is correct when every run was corrected. Pass an
        explicit larger value when some runs lack timestamps and
        were not corrected -- otherwise σ_correction is biased
        upward by ``(W_total / W_subset)²``.

    Returns
    -------
    float
        ``σ_correction`` for the weighted-average centroid (MHz).
    """
    t = np.asarray(t_hours, dtype=float)
    w = np.asarray(weights, dtype=float)
    W_subset = float(np.sum(w))
    if len(t) == 0 or W_subset == 0.0:
        return 0.0
    W_norm = (float(total_weight) if total_weight is not None
              else W_subset)
    if W_norm <= 0.0:
        return 0.0
    K = np.asarray(corrector.cov(t, t), dtype=float)
    var = float(w @ K @ w) / (W_norm * W_norm)
    psd_tol = max(1e-9 * float(np.max(np.abs(K)) if K.size else 1.0),
                  1e-12)
    if not np.isfinite(var):
        return 0.0
    if var < -psd_tol:
        warnings.warn(
            f"gp_propagated_sigma: posterior variance below PSD "
            f"tolerance ({var:.3e}); clamping to 0. Kernel matrix "
            f"may be ill-conditioned (near-duplicate timestamps?).",
            stacklevel=2,
        )
    return float(np.sqrt(max(var, 0.0)))


def mixed_mode_propagated_sigma(corrector=None, *,
                                 auto_t=(), auto_w=(),
                                 manual_sigmas=(), manual_w=(),
                                 total_weight):
    """σ_correction for an isotope's weighted-mean centroid when
    some runs were Auto-corrected (GP) and others Manual (user
    sigma).

    Auto rows propagate through the GP posterior covariance::

        Var_auto = (1/W²) · w_autoᵀ K_GP w_auto

    Manual rows are treated as independent of each other and of
    the GP -- the user typed in a value, with no claim on
    correlation::

        Var_manual = (1/W²) · Σ wᵢ² σᵢ²

    The two pieces add::

        σ_correction = √(Var_auto + Var_manual)

    Cross-isotope cross-covariance (used by the shift formula in
    ``compute_all_shifts``) only involves Auto rows -- Manual
    sigmas are uncorrelated across isotopes too. Use
    ``gp_cross_covariance`` with the auto subsets directly.

    ``corrector`` is optional. It's only consulted when ``auto_t``
    is non-empty -- so a Manual-only call (for legacy/intermediate
    saves where the GP corrector wasn't restored, but Manual
    metadata is still on disk) returns the correct diagonal
    propagation without needing one. Passing ``corrector=None``
    while ``auto_t`` is non-empty raises ``ValueError`` -- there's
    no honest fallback in that case.

    Returns 0 when both Auto and Manual lists are empty or
    ``total_weight <= 0``.
    """
    W = float(total_weight)
    if W <= 0.0:
        return 0.0
    var = 0.0
    auto_t = np.asarray(auto_t, dtype=float)
    auto_w = np.asarray(auto_w, dtype=float)
    if len(auto_t) > 0:
        if corrector is None:
            raise ValueError(
                "mixed_mode_propagated_sigma: auto_t is non-empty "
                "but no corrector was supplied; the GP posterior "
                "covariance cannot be evaluated.")
        K = np.asarray(corrector.cov(auto_t, auto_t), dtype=float)
        var += float(auto_w @ K @ auto_w) / (W * W)
    manual_sigmas = np.asarray(manual_sigmas, dtype=float)
    manual_w = np.asarray(manual_w, dtype=float)
    if len(manual_sigmas) > 0:
        var += float(np.sum((manual_w * manual_sigmas) ** 2)) / (W * W)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0
    return float(np.sqrt(var))


def gp_cross_covariance(corrector, t_a, w_a, t_b, w_b, *,
                          total_weight_a=None, total_weight_b=None):
    """Cross-covariance between two isotopes' weighted-mean
    centroids when both share the same GP corrector.

    ``Cov(c̄_a, c̄_b) = (1/(W_a_total · W_b_total))
                       · w_a_subsetᵀ K_GP(t_a, t_b) w_b_subset``

    Subset/total semantics mirror :func:`gp_propagated_sigma`: the
    weights and timestamps describe the runs that were actually
    GP-corrected, while ``total_weight_*`` describes the full
    centroid normalization.

    Used to subtract the shared-systematic term from
    ``Var(δν) = Var(c_a) + Var(c_b) - 2·Cov``: similar timestamps
    tighten the shift; well-separated timestamps return ~0.

    Parameters
    ----------
    corrector : ReferenceCorrector
    t_a, t_b : array-like of float
    w_a, w_b : array-like of float
    total_weight_a, total_weight_b : float, optional
        Defaults to ``sum(w_a)`` / ``sum(w_b)``. Pass explicitly
        when some runs lacked timestamps.

    Returns
    -------
    float
        Cov in MHz².
    """
    ta = np.asarray(t_a, dtype=float)
    tb = np.asarray(t_b, dtype=float)
    wa = np.asarray(w_a, dtype=float)
    wb = np.asarray(w_b, dtype=float)
    if len(ta) == 0 or len(tb) == 0:
        return 0.0
    Wa_sub = float(np.sum(wa))
    Wb_sub = float(np.sum(wb))
    if Wa_sub == 0.0 or Wb_sub == 0.0:
        return 0.0
    Wa = (float(total_weight_a) if total_weight_a is not None
          else Wa_sub)
    Wb = (float(total_weight_b) if total_weight_b is not None
          else Wb_sub)
    if Wa <= 0.0 or Wb <= 0.0:
        return 0.0
    K = np.asarray(corrector.cov(ta, tb), dtype=float)
    cov = float(wa @ K @ wb) / (Wa * Wb)
    return cov if np.isfinite(cov) else 0.0


def isotope_shift(centroid_A, sigma_A, centroid_ref, sigma_ref):
    """Compute isotope shift and propagated error.

    Parameters
    ----------
    centroid_A, sigma_A : float
        Centroid and uncertainty for isotope A [MHz].
    centroid_ref, sigma_ref : float
        Centroid and uncertainty for reference isotope [MHz].

    Returns
    -------
    (delta_nu, sigma_delta_nu) : tuple of float
        delta_nu = centroid_A - centroid_ref [MHz]
    """
    delta = centroid_A - centroid_ref
    sigma = np.sqrt(sigma_A**2 + sigma_ref**2)
    return float(delta), float(sigma)


def cooler_voltage_systematic(run_metadata_by_project, mass_amu_by_project,
                               charge, geometry, nu_laser_MHz, harmonic=1):
    """Estimate systematic frequency error from experimental jitter.

    For each project independently:
    - Compute mean and std of cooler voltage and laser setpoint across runs
    - Use std/sqrt(N) (standard error of the mean) as the voltage/laser
      uncertainty when runs are averaged
    - Convert to frequency uncertainty via numerical differentiation of
      the Doppler formula
    - Combine cooler and laser contributions in quadrature

    Projects with only 1 run get std=0 and are flagged.

    Parameters
    ----------
    run_metadata_by_project : dict[str, list[dict]]
        ``{project_name: [run_metadata_dict, ...]}`` where each dict
        has keys ``"cooler_v"`` and optionally ``"laser_set"``.
    mass_amu_by_project : dict[str, float]
    charge : int
    geometry : str
    nu_laser_MHz : float
        Laser/reference frequency (MHz) for the Doppler sensitivity. When 0
        (E-levels unset) the cooler-voltage term cannot be computed and is
        flagged as omitted.
    harmonic : int
        Laser harmonic. The setpoint is the fundamental wavenumber, so a
        setpoint jitter shifts the ion frequency by ``harmonic`` times as much.

    Returns
    -------
    dict with keys:
        ``"systematic_errors"``  : dict[str, float]  (project -> sigma_sys MHz)
        ``"project_stats"``      : dict[str, dict]  (per-project statistics)
        ``"details"``            : str  (human-readable log)
    """
    log_lines = ["=== Systematic Error Estimation ===\n"]

    # Collect per-project statistics for cooler voltage and laser setpoint
    stats = {}
    for pname, metadata_list in run_metadata_by_project.items():
        voltages = [m.get("cooler_v") for m in metadata_list
                    if m.get("cooler_v") is not None]
        lasers = [m.get("laser_set") for m in metadata_list
                  if m.get("laser_set") is not None]

        n_v = len(voltages)
        v_arr = np.array(voltages) if voltages else np.array([0.0])
        n_l = len(lasers)
        l_arr = np.array(lasers) if lasers else np.array([0.0])

        v_mean = float(np.mean(v_arr))
        v_std = float(np.std(v_arr, ddof=1)) if n_v > 1 else 0.0
        v_sem = v_std / np.sqrt(n_v) if n_v > 1 else 0.0

        l_mean = float(np.mean(l_arr))
        l_std = float(np.std(l_arr, ddof=1)) if n_l > 1 else 0.0
        l_sem = l_std / np.sqrt(n_l) if n_l > 1 else 0.0

        stats[pname] = {
            "n_runs": n_v,
            "v_mean": v_mean, "v_std": v_std, "v_sem": v_sem,
            "l_mean": l_mean, "l_std": l_std, "l_sem": l_sem,
        }

        log_lines.append(f"Project '{pname}' ({n_v} runs):")
        log_lines.append(
            f"  Cooler V:  mean = {v_mean:.4f} V, "
            f"std = {v_std:.6f} V, "
            f"SEM = {v_sem:.6f} V")
        log_lines.append(
            f"  Laser set: mean = {l_mean:.6f} cm-1, "
            f"std = {l_std:.8f} cm-1, "
            f"SEM = {l_sem:.8f} cm-1")
        if n_v <= 1:
            log_lines.append(
                "  WARNING: only 1 run — SEM = 0, systematic error may be "
                "underestimated. Consider adding runs or manual systematic.")

    if not stats:
        return {
            "systematic_errors": {},
            "project_stats": {},
            "details": "No run metadata found in any project.",
        }

    # Compute frequency sensitivity and systematic error per project
    step_V = 0.01   # 10 mV step for d(nu)/d(V)
    systematic_errors = {}

    log_lines.append(
        f"\nDoppler sensitivity (charge={charge}, "
        f"geometry='{geometry}', nu_laser={nu_laser_MHz:.3f} MHz, "
        f"harmonic={harmonic}):\n")
    if nu_laser_MHz <= 0:
        # With no laser/reference frequency, nu_seen_by_ion is 0 and the
        # finite-difference d(nu)/d(V) collapses to 0, so the (usually
        # dominant) cooler-voltage term silently vanishes. Surface it instead
        # (code review 2026-06-02, sys-voltage-silently-zero-when-no-reffreq).
        log_lines.append(
            "  WARNING: laser/reference frequency is 0 -- set the atomic "
            "energy levels (E lower / E upper), or 'Pull from Pre-Analysis'. "
            "The cooler-voltage sensitivity d(nu)/d(V) is therefore 0, so the "
            "usually-dominant cooler term is OMITTED and the systematic below "
            "is UNDERESTIMATED.\n")

    for pname, mass in mass_amu_by_project.items():
        if pname not in stats:
            continue
        st = stats[pname]
        V0 = st["v_mean"]

        # d(nu)/d(V) — sensitivity to cooler voltage
        beta_lo = beta_from_voltage(V0 - step_V / 2, mass, charge)
        beta_hi = beta_from_voltage(V0 + step_V / 2, mass, charge)
        nu_lo = nu_seen_by_ion(nu_laser_MHz, beta_lo, geometry)
        nu_hi = nu_seen_by_ion(nu_laser_MHz, beta_hi, geometry)
        dnu_dV = float(abs(nu_hi - nu_lo) / step_V)  # MHz/V

        # Cooler voltage contribution: use SEM (std/sqrt(N))
        sigma_V = dnu_dV * st["v_sem"]

        # Laser setpoint contribution: delta(nu_laser) in MHz
        # Laser setpoint is in cm-1; convert std to MHz
        # 1 cm-1 = c * 100 Hz = 29979.2458 MHz
        CM1_TO_MHZ = 29979.2458
        # The setpoint is the FUNDAMENTAL wavenumber; the ion sees the
        # harmonic-multiplied frequency, so a setpoint jitter scales by the
        # harmonic (code review 2026-06-02, sys-laser-missing-harmonic).
        sigma_L = st["l_sem"] * CM1_TO_MHZ * harmonic  # lab-frame shift
        # In ion frame: scaled by Doppler factor ~= nu_ion / nu_laser
        beta0 = beta_from_voltage(V0, mass, charge)
        nu_ion = nu_seen_by_ion(nu_laser_MHz, beta0, geometry)
        doppler_factor = nu_ion / nu_laser_MHz if nu_laser_MHz > 0 else 1.0
        sigma_L_ion = sigma_L * doppler_factor

        sigma_total = float(np.sqrt(sigma_V**2 + sigma_L_ion**2))
        systematic_errors[pname] = sigma_total

        log_lines.append(f"  {pname} (A={mass:.1f} amu):")
        log_lines.append(
            f"    d(nu)/d(V) = {dnu_dV:.4f} MHz/V")
        log_lines.append(
            f"    Cooler: SEM = {st['v_sem']:.6f} V "
            f"-> sigma_V = {sigma_V:.4f} MHz")
        log_lines.append(
            f"    Laser:  SEM = {st['l_sem']:.8f} cm-1 "
            f"-> sigma_L = {sigma_L_ion:.4f} MHz")
        log_lines.append(
            f"    Combined: sigma_sys = sqrt(sigma_V^2 + sigma_L^2) "
            f"= {sigma_total:.4f} MHz")
        if st["n_runs"] <= 1:
            log_lines.append(
                f"    (!) Single run: SEM=0, systematic may be "
                f"underestimated")

    log_lines.append(
        "\nMethod: sigma_sys = sqrt( (|d(nu)/d(V)| * SEM_V)^2 "
        "+ (SEM_laser * harmonic * Doppler_factor)^2 )")
    log_lines.append(
        "SEM = std / sqrt(N) — standard error of the mean across runs. "
        "NOTE: SEM treats run-to-run jitter as random and shrinks with N; a "
        "correlated drift would be underestimated (use the std column above "
        "as a conservative alternative).")

    return {
        "systematic_errors": systematic_errors,
        "project_stats": stats,
        "details": "\n".join(log_lines),
    }


def compute_all_shifts(isotope_entries, reference_key, *,
                       gp_cross_cov=None):
    """Compute isotope shifts for all entries relative to a reference.

    Parameters
    ----------
    isotope_entries : dict[str, dict]
        ``{label: {"centroid": float, "A": int, ...}}`` where the
        per-isotope sigma keys may be either:

        Legacy form (still accepted for backwards compatibility):
          ``"sigma_stat"`` (statistical, including any inflation),
          ``"sigma_sys"`` (single bucket of systematics).

        Phase-5 form (preferred when available):
          ``"sigma_fit"``        — inverse-variance from per-run fits
          ``"sigma_scatter"``    — Birge-ratio inflation excess
          ``"sigma_correction"`` — GP centroid-correction propagation
          ``"sigma_voltage"``    — cooler/laser systematic
        These are summed in quadrature into ``sigma_total`` per
        isotope. The legacy ``sigma_stat`` is mapped onto
        ``sigma_fit`` (and ``sigma_sys`` onto ``sigma_voltage``)
        when the new keys are not present, so the call site can
        migrate gradually.

    reference_key : str
        Key in *isotope_entries* to use as the reference (shift = 0).

    gp_cross_cov : dict[(str, str), float], optional
        Cross-covariance between paired isotopes' weighted-mean
        centroids under a shared GP corrector, in MHz². When given
        and a non-zero entry is present for ``(label, reference_key)``
        (or its swap), the shift variance becomes
        ``Var(c_a) + Var(c_ref) − 2·Cov``. When absent or zero, all
        sigma contributions add in quadrature -- matching the
        pre-Phase-5 behavior.

    Returns
    -------
    list[dict], sorted by A. Each row carries the legacy keys
    (``sigma_stat``, ``sigma_sys``, ``sigma_total``,
    ``sigma_delta_nu_stat``, ``sigma_delta_nu_sys``,
    ``sigma_delta_nu``) AND the new Phase-5 keys
    (``sigma_fit``, ``sigma_scatter``, ``sigma_correction``,
    ``sigma_voltage``, ``sigma_delta_nu_scatter``,
    ``sigma_delta_nu_correction``, ``cov_with_reference_mhz2``).

    The legacy ``sigma_stat`` keeps its pre-Phase-5 meaning of "pure
    inverse-variance σ" (= ``sigma_fit``) so old CSV / log readers
    don't silently get a different number; the Birge-inflation
    contribution is exposed separately as ``sigma_scatter``.
    Likewise ``sigma_delta_nu_stat`` is the legacy quadrature sum of
    per-isotope ``sigma_fit`` only; ``sigma_delta_nu_scatter``
    carries the Birge contribution. ``sigma_total`` and
    ``sigma_delta_nu`` are full-budget (everything in quadrature
    minus 2·Cov for the correction term).
    """
    def _split_sigmas(entry):
        """Return (sigma_fit, sigma_scatter, sigma_correction,
        sigma_voltage) preferring the new keys; fall back to legacy
        sigma_stat/sigma_sys."""
        if "sigma_fit" in entry:
            return (
                float(entry.get("sigma_fit", 0.0)),
                float(entry.get("sigma_scatter", 0.0)),
                float(entry.get("sigma_correction", 0.0)),
                float(entry.get("sigma_voltage", 0.0)),
            )
        return (
            float(entry.get("sigma_stat", 0.0)),
            0.0,
            0.0,
            float(entry.get("sigma_sys", 0.0)),
        )

    cross = gp_cross_cov or {}

    def _cov_with_ref(label):
        if label == reference_key:
            return 0.0
        v = cross.get((label, reference_key))
        if v is None:
            v = cross.get((reference_key, label))
        return float(v or 0.0)

    ref_entry = isotope_entries[reference_key]
    ref_c = ref_entry["centroid"]
    ref_fit, ref_scat, ref_corr, ref_volt = _split_sigmas(ref_entry)
    var_ref = (ref_fit ** 2 + ref_scat ** 2 + ref_corr ** 2
               + ref_volt ** 2)

    rows = []
    for label, entry in isotope_entries.items():
        c = entry["centroid"]
        s_fit, s_scat, s_corr, s_volt = _split_sigmas(entry)
        # Total per-isotope σ aggregates everything in quadrature.
        s_total = float(np.sqrt(s_fit ** 2 + s_scat ** 2
                                  + s_corr ** 2 + s_volt ** 2))
        # Legacy aggregations: sigma_stat keeps its pre-Phase-5
        # meaning of pure inverse-variance σ (no Birge); sigma_sys
        # rolls correction + voltage so old code paths that read
        # `sigma_sys` keep working.
        s_stat = s_fit
        s_sys = float(np.sqrt(s_corr ** 2 + s_volt ** 2))
        is_ref = (label == reference_key)

        cov_ab = _cov_with_ref(label)
        if is_ref:
            dnu = 0.0
            s_dnu_stat = 0.0
            s_dnu_scatter = 0.0
            s_dnu_corr = 0.0
            s_dnu_sys = 0.0
            s_dnu = 0.0
        else:
            dnu = float(c - ref_c)
            # Pure stat: inverse-variance σ from each isotope, in
            # quadrature. Legacy semantics.
            s_dnu_stat = float(np.sqrt(
                s_fit ** 2 + ref_fit ** 2))
            # Birge-inflation contribution: separate column so
            # callers can choose how to report it.
            s_dnu_scatter = float(np.sqrt(
                s_scat ** 2 + ref_scat ** 2))
            # Voltage systematic: independent per isotope.
            var_dnu_volt = s_volt ** 2 + ref_volt ** 2
            # GP correction: corrected centroids share the same GP,
            # so subtract 2·Cov from the naive quadrature. With a
            # PSD posterior the result is non-negative; clamp +
            # warn to defend against numerical roundoff.
            raw_var_corr = (s_corr ** 2 + ref_corr ** 2
                             - 2.0 * cov_ab)
            tol = 1e-9 * max(s_corr ** 2, ref_corr ** 2, 1.0)
            if raw_var_corr < -tol:
                warnings.warn(
                    f"compute_all_shifts: shift-correction variance "
                    f"for {label!r} below PSD tolerance "
                    f"({raw_var_corr:.3e}); clamping to 0.",
                    stacklevel=2,
                )
            var_dnu_corr = max(raw_var_corr, 0.0)
            s_dnu_corr = float(np.sqrt(var_dnu_corr))
            s_dnu_sys = float(np.sqrt(var_dnu_volt + var_dnu_corr))
            s_dnu = float(np.sqrt(s_dnu_stat ** 2
                                    + s_dnu_scatter ** 2
                                    + s_dnu_sys ** 2))

        rows.append({
            "label": label,
            "A": entry.get("A", 0),
            "centroid": float(c),
            # Phase-5 per-isotope sigma columns
            "sigma_fit": s_fit,
            "sigma_scatter": s_scat,
            "sigma_correction": s_corr,
            "sigma_voltage": s_volt,
            # Legacy aggregated columns (compatible meanings)
            "sigma_stat": s_stat,
            "sigma_sys": s_sys,
            "sigma_total": s_total,
            # Shift columns
            "delta_nu": float(dnu),
            "sigma_delta_nu_stat": float(s_dnu_stat),
            "sigma_delta_nu_scatter": float(s_dnu_scatter),
            "sigma_delta_nu_correction": float(s_dnu_corr),
            "sigma_delta_nu_sys": float(s_dnu_sys),
            "sigma_delta_nu": float(s_dnu),
            "cov_with_reference_mhz2": float(cov_ab),
            "is_reference": is_ref,
        })

    rows.sort(key=lambda r: r["A"])
    return rows
