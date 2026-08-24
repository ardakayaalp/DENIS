#!/usr/bin/env python3
"""DENIS collinear laser spectroscopy run-time estimation script.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Command-line driver for the DENIS estimation toolkit. Reads a YAML
configuration describing an element, transition, beam, and isotopes, then
computes hyperfine structure, Doppler-shifted resonance voltages, and the
measurement time required to reach a target statistical significance. Emits
a logged report (and optional plots) summarizing peaks and timing.

Usage:
    python cls_estimate.py <config.yaml> [--output-dir DIR] [--no-plot]

Depends on: cls_estimations.constants, cls_estimations.mass_lookup,
cls_estimations.hfs_model, cls_estimations.doppler, cls_estimations.statistics,
cls_estimations.config_parser, cls_estimations.plotting, cls_estimations.schmidt
"""
import argparse
import logging
import math
import os
import sys
from datetime import datetime

import numpy as np
from tabulate import tabulate

from cls_estimations.constants import CM_TO_HZ
from cls_estimations.mass_lookup import load_mass_table, get_mass
from cls_estimations.hfs_model import compute_hfs_lines, scale_A, scale_B
from cls_estimations.doppler import (
    transition_frequency_MHz,
    resonance_voltage,
    generate_voltage_spectrum,
    laser_frequency_for_resonance,
)
from cls_estimations.statistics import (
    signal_rate_for_peak,
    time_for_sigma,
    scanning_time,
    estimate_isotope_timing,
)
from cls_estimations.config_parser import load_config
from cls_estimations.plotting import (
    plot_all_cases, plot_combined_overview, set_palette, PALETTES,
)
from cls_estimations.schmidt import calculate_schmidt_moment


def _fmt_time(seconds):
    """Format seconds as a human-readable string."""
    h = seconds / 3600.0
    if seconds < 120:
        return f"{seconds:.1f}s"
    if h < 1:
        return f"{seconds / 60:.1f}min"
    return f"{h:.2f}h"


def _fmt_J(j):
    """Format a J quantum number: integer if whole, else half-integer (e.g. 5/2)."""
    if j == int(j):
        return f"{int(j)}"
    twice = round(2 * j)
    if abs(2 * j - twice) < 1e-6 and twice % 2 == 1:
        return f"{twice}/2"
    return f"{j:.1f}"


def _fmt_efficiency(eff):
    """Format efficiency as '1 photon per N ions'."""
    if eff <= 0:
        return "0"
    ratio = 1.0 / eff
    if ratio == int(ratio):
        return f"1 photon per {int(ratio)} ions"
    return f"1 photon per {ratio:.1f} ions"


def _spin_production_ratio(J_gs, J_isomer, sigma):
    """Estimate production ratio from spin distribution statistics.

    P(J_gs)/P(J_m) = (2J_gs+1)/(2J_m+1) * exp(-(J_gs(J_gs+1) - J_m(J_m+1)) / (2*sigma^2))

    Returns (production_ratio, gs_to_iso_ratio) where production_ratio is the
    fraction going to the ground state.
    """
    exponent = -(J_gs * (J_gs + 1) - J_isomer * (J_isomer + 1)) / (2.0 * sigma**2)
    ratio = (2.0 * J_gs + 1) / (2.0 * J_isomer + 1) * math.exp(exponent)
    production_ratio = ratio / (ratio + 1.0)
    return production_ratio, ratio


def _get_hfs_params(iso, ref, is_isomer=False, isomer_cfg=None, g_s_factor=1.0):
    """Determine HFS A/B parameters: use overrides if provided, else scale from reference.

    Returns (A_l, B_l, A_u, B_u, mu_used, mu_source) where:
    - mu_used is the magnetic moment used for scaling
    - mu_source is a string describing where mu came from
    """
    if is_isomer and isomer_cfg is not None:
        cfg = isomer_cfg
        I = cfg.I
        mu_config = cfg.mu
        Q = cfg.Q
        schmidt_spec = cfg.schmidt_valence
    else:
        cfg = iso
        I = iso.I
        mu_config = iso.mu
        Q = iso.Q
        schmidt_spec = iso.schmidt_valence

    # Determine mu: Schmidt > config value
    mu_source = ""
    if schmidt_spec is not None:
        # Build dict for schmidt calculation
        spec_dict = {}
        if schmidt_spec.type is not None and schmidt_spec.orbital is not None:
            spec_dict = {"type": schmidt_spec.type, "orbital": schmidt_spec.orbital}
        elif all([schmidt_spec.type1, schmidt_spec.orbital1,
                  schmidt_spec.type2, schmidt_spec.orbital2, schmidt_spec.I is not None]):
            spec_dict = {
                "type1": schmidt_spec.type1, "orbital1": schmidt_spec.orbital1,
                "type2": schmidt_spec.type2, "orbital2": schmidt_spec.orbital2,
                "I": schmidt_spec.I
            }

        if spec_dict:
            mu, mu_desc = calculate_schmidt_moment(spec_dict, g_s_factor=g_s_factor)
            mu_source = f"Schmidt: {mu_desc}"
        else:
            mu = mu_config
            mu_source = "config"
    else:
        mu = mu_config
        mu_source = "config"

    if cfg.A_lower_MHz is not None:
        A_l = cfg.A_lower_MHz
    else:
        if I == 0 or mu == 0:
            A_l = 0.0
        else:
            A_l = scale_A(ref.A_lower_MHz, ref.mu, ref.I, mu, I)

    if cfg.A_upper_MHz is not None:
        A_u = cfg.A_upper_MHz
    else:
        if I == 0 or mu == 0:
            A_u = 0.0
        else:
            A_u = scale_A(ref.A_upper_MHz, ref.mu, ref.I, mu, I)

    if cfg.B_lower_MHz is not None:
        B_l = cfg.B_lower_MHz
    else:
        B_l = scale_B(ref.B_lower_MHz, ref.Q, Q)

    if cfg.B_upper_MHz is not None:
        B_u = cfg.B_upper_MHz
    else:
        B_u = scale_B(ref.B_upper_MHz, ref.Q, Q)

    return A_l, B_l, A_u, B_u, mu, mu_source


def _compute_peak_voltages(offsets_MHz, centroid_MHz, nu_laser_MHz, mass_amu,
                           charge, V0, geometry):
    """Compute resonance dV for each HFS peak."""
    dVs = []
    for off in offsets_MHz:
        nu_rest = centroid_MHz + off
        V_res = resonance_voltage(nu_rest, nu_laser_MHz, mass_amu, charge, geometry)
        if V_res is not None:
            dVs.append(V0 - V_res)
        else:
            dVs.append(None)
    return dVs


def _print_nuclear_params_table(out_func, reference, nuclear_params, g_s_factor):
    """Print summary table of nuclear parameters and HFS constants used."""
    out_func()
    out_func("=" * 64)
    out_func("  NUCLEAR PARAMETERS & HFS CONSTANTS")
    out_func("=" * 64)

    # Reference isotope
    out_func()
    out_func(f"Reference: {reference.A} (I={reference.I})")
    out_func(f"  mu = {reference.mu:.4f} mu_N")
    out_func(f"  Q = {reference.Q:.4f} b")
    out_func(f"  A_lower = {reference.A_lower_MHz:.2f} MHz")
    out_func(f"  A_upper = {reference.A_upper_MHz:.2f} MHz")
    out_func(f"  B_lower = {reference.B_lower_MHz:.2f} MHz")
    out_func(f"  B_upper = {reference.B_upper_MHz:.2f} MHz")

    if g_s_factor != 1.0:
        out_func()
        out_func(f"Schmidt g_s quenching factor: {g_s_factor:.3f}")

    # Nuclear parameters table
    nuc_rows = []
    for params in nuclear_params:
        mu_src = params["mu_source"]
        nuc_rows.append([
            params["label"],
            params["state"] or "-",
            f"{params['mass_amu']:.9f}",
            f"{params['I']:.1f}",
            f"{params['mu']:.4f}",
            mu_src,
        ])

    out_func()
    out_func(tabulate(
        nuc_rows,
        headers=["Isotope", "State", "Mass (amu)", "I", "mu (mu_N)", "mu source"],
        tablefmt="pretty",
        colalign=("left", "left", "right", "right", "right", "left"),
        disable_numparse=[2],
    ))

    # HFS constants table
    hfs_rows = []
    for params in nuclear_params:
        hfs_rows.append([
            params["label"],
            params["state"] or "-",
            f"{params['A_l']:.2f}",
            f"{params['A_u']:.2f}",
            f"{params['B_l']:.2f}",
            f"{params['B_u']:.2f}",
        ])

    out_func()
    out_func("HFS Constants (MHz):")
    out_func(tabulate(
        hfs_rows,
        headers=["Isotope", "State", "A_lower", "A_upper", "B_lower", "B_upper"],
        tablefmt="pretty",
        colalign=("left", "left", "right", "right", "right", "right"),
    ))
    out_func("=" * 64)


def run(config_path, output_dir, no_plot, palette="default", peak_list=None):
    # peak_list is accepted but unused: the peak list is always emitted in
    # both intensity and voltage orderings.
    del peak_list
    # --- Derive config name and create subfolder ---
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join(output_dir, config_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load config ---
    cfg = load_config(config_path)

    # --- Build isotope file tag (e.g. "_59Co_54Co") ---
    seen = []
    for iso in cfg.isotopes:
        if iso.label not in seen:
            seen.append(iso.label)
    file_tag = "_" + "_".join(seen)

    # --- Logging: dual console + file ---
    log_path = os.path.join(output_dir, f"{config_name}_cls_estimate{file_tag}.log")
    logger = logging.getLogger("cls_estimate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    def out(msg=""):
        logger.info(msg)

    # --- Load mass table ---
    mass_table = load_mass_table()

    # --- Derived quantities ---
    nu0_MHz = transition_frequency_MHz(cfg.transition.lower_level_cm,
                                       cfg.transition.upper_level_cm)
    V0 = cfg.beam.voltage_kV * 1e3

    # --- Resolve anchor (auto-calculate setpoint_cm) ---
    anchor_info = None  # set if anchor is used, for header output
    if cfg.laser.setpoint_cm is None:
        # Find the anchor isotope in the isotopes list
        anchor_label = cfg.laser.anchor_isotope
        anchor_iso = None
        for iso in cfg.isotopes:
            if iso.label == anchor_label:
                anchor_iso = iso
                break
        if anchor_iso is None:
            raise ValueError(
                f"Anchor isotope '{anchor_label}' not found in isotopes list."
            )

        anchor_mass = get_mass(mass_table, cfg.Z, anchor_iso.A)

        # Determine centroid frequency based on state
        anchor_state = cfg.laser.anchor_state or "gs"
        if anchor_state == "isomer":
            if anchor_iso.isomer is None:
                raise ValueError(
                    f"Anchor state 'isomer' requested but '{anchor_label}' has no isomer."
                )
            anchor_shift = anchor_iso.isomer.isotope_shift_MHz
            state_desc = f"isomer (I={anchor_iso.isomer.I})"
        else:
            anchor_shift = anchor_iso.isotope_shift_MHz
            state_desc = "g.s."
        centroid_anchor_MHz = nu0_MHz + anchor_shift

        # Compute resonance voltage and inverse Doppler
        V_res = V0 - cfg.laser.anchor_dV
        nu_laser_anchor_MHz = float(laser_frequency_for_resonance(
            centroid_anchor_MHz, V_res, anchor_mass,
            cfg.beam.charge_state, cfg.beam.geometry,
        ))

        # Convert to setpoint_cm (fundamental wavenumber)
        setpoint_cm = nu_laser_anchor_MHz * 1e6 / (cfg.laser.harmonic * CM_TO_HZ)
        cfg.laser.setpoint_cm = setpoint_cm

        anchor_info = {
            "label": anchor_label,
            "state_desc": state_desc,
            "dV": cfg.laser.anchor_dV,
            "setpoint_cm": setpoint_cm,
        }

    nu_laser_MHz = cfg.laser.setpoint_cm * cfg.laser.harmonic * CM_TO_HZ / 1e6

    if not cfg.spectrum_only:
        scan_params = dict(
            voltage_range_V=cfg.scan.voltage_range_V,
            step_size_V=cfg.scan.step_size_V,
            dwell_time_ms=cfg.scan.dwell_time_ms,
        )

    # --- Header ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out("=" * 64)
    if cfg.spectrum_only:
        out(f"  DENIS - Spectrum Mode - {now}")
    else:
        out(f"  DENIS - Run Time Estimation - {now}")
    out("=" * 64)
    out(f"Element: {cfg.element} (Z={cfg.Z}) | "
        f"Transition: {cfg.transition.lower_level_cm} -> {cfg.transition.upper_level_cm} cm-1 "
        f"(J={_fmt_J(cfg.transition.J_lower)}->{_fmt_J(cfg.transition.J_upper)})")
    fund_cm = cfg.laser.setpoint_cm
    harm_cm = fund_cm * cfg.laser.harmonic
    fund_nm = 1e7 / fund_cm
    harm_nm = 1e7 / harm_cm
    if anchor_info is not None:
        out(f"Laser: auto-calculated from anchor "
            f"({anchor_info['label']} {anchor_info['state_desc']} centroid "
            f"at dV={anchor_info['dV']:+.1f}V)")
        out(f"  -> {fund_cm:.6f} cm-1 ({fund_nm:.3f} nm) x {cfg.laser.harmonic} = "
            f"{harm_cm:.6f} cm-1 ({harm_nm:.3f} nm)")
    else:
        out(f"Laser: {fund_cm:.6f} cm-1 ({fund_nm:.3f} nm) x {cfg.laser.harmonic} = "
            f"{harm_cm:.6f} cm-1 ({harm_nm:.3f} nm)")
    out(f"Beam: {cfg.beam.voltage_kV:.3f} kV, q={cfg.beam.charge_state}, "
        f"{cfg.beam.geometry} | FWHM: {cfg.linewidth_fwhm_MHz} MHz")
    half_range = cfg.scan.voltage_range_V / 2
    if cfg.spectrum_only:
        out(f"Mode: spectrum only | Scan range: +/-{half_range:.0f}V")
    else:
        uniform_note = " | Uniform timing: ON" if cfg.uniform_timing else ""
        out(f"Sigma required: {cfg.required_sigma} | "
            f"Scan: +/-{half_range:.0f}V, {cfg.scan.step_size_V}V steps, "
            f"{cfg.scan.dwell_time_ms:.0f}ms dwell{uniform_note}")
    out("-" * 64)

    grand_total_s = 0.0
    all_plot_results = []
    all_peaks = []          # collected for optional peak-list log
    nuclear_params = []     # collected for nuclear parameters summary table

    for iso in cfg.isotopes:
        mass_amu = get_mass(mass_table, cfg.Z, iso.A)

        # Split yield if isomer exists
        has_isomer = iso.isomer is not None
        gs_yield = None
        iso_yield = None
        spin_ratio_info = None  # populated when using spin distribution
        if not cfg.spectrum_only:
            if has_isomer:
                isom = iso.isomer
                if isom.yield_ions_per_sec is not None:
                    # Independent yields: g.s. and isomer specified separately
                    gs_yield = iso.yield_ions_per_sec
                    iso_yield = isom.yield_ions_per_sec
                elif isom.spin_distribution_sigma is not None:
                    # Auto-calculate from spin statistics
                    prod_ratio, gs_iso_ratio = _spin_production_ratio(
                        iso.I, isom.I, isom.spin_distribution_sigma
                    )
                    gs_yield = iso.yield_ions_per_sec * prod_ratio
                    iso_yield = iso.yield_ions_per_sec * (1.0 - prod_ratio)
                    spin_ratio_info = {
                        "sigma": isom.spin_distribution_sigma,
                        "J_gs": iso.I,
                        "J_isomer": isom.I,
                        "ratio": gs_iso_ratio,
                        "prod_ratio_gs": prod_ratio,
                        "prod_ratio_iso": 1.0 - prod_ratio,
                        "total_yield": iso.yield_ions_per_sec,
                    }
                else:
                    # Split from total yield using production_ratio
                    gs_yield = iso.yield_ions_per_sec * isom.production_ratio
                    iso_yield = iso.yield_ions_per_sec * (1.0 - isom.production_ratio)
            else:
                gs_yield = iso.yield_ions_per_sec

        # Determine uniform timing: per-isotope overrides global default
        use_uniform = iso.uniform_timing if iso.uniform_timing is not None else cfg.uniform_timing

        # ============ Ground state ============
        A_l, B_l, A_u, B_u, mu_used, mu_source = _get_hfs_params(
            iso, cfg.reference, g_s_factor=cfg.g_s_factor
        )
        iso_shift = iso.isotope_shift_MHz
        centroid_MHz = nu0_MHz + iso_shift

        # Collect nuclear params for summary table
        nuclear_params.append({
            "label": iso.label,
            "state": "g.s." if has_isomer else "",
            "mass_amu": mass_amu,
            "I": iso.I,
            "mu": mu_used,
            "mu_source": mu_source,
            "Q": iso.Q,
            "A_l": A_l,
            "A_u": A_u,
            "B_l": B_l,
            "B_u": B_u,
        })

        out()
        out("-" * 64)
        state_label = "g.s." if has_isomer else ""
        if iso.I == 0 and not has_isomer:
            state_label = "I=0"
        elif iso.I == 0:
            state_label = "g.s., I=0"
        else:
            state_label = f"g.s., I={iso.I}" if has_isomer else f"I={iso.I}"

        if cfg.spectrum_only:
            out(f"{iso.label} ({state_label})")
        else:
            yield_note = ""
            if has_isomer and iso.isomer.yield_ions_per_sec is not None:
                yield_note = " [independent]"
            elif spin_ratio_info is not None:
                yield_note = " [spin dist.]"
            out(f"{iso.label} ({state_label}) | Yield: {gs_yield:.2e} cps{yield_note} | "
                f"Eff: {iso.spectroscopic_efficiency} ({_fmt_efficiency(iso.spectroscopic_efficiency)})")
            if spin_ratio_info is not None:
                sri = spin_ratio_info
                out(f"  Spin distribution (sigma={sri['sigma']:.1f}): "
                    f"P(g.s.)/P(iso) = {sri['ratio']:.2f}:1 "
                    f"-> g.s. {sri['prod_ratio_gs']:.1%} ({gs_yield:.2e} cps), "
                    f"isomer {sri['prod_ratio_iso']:.1%} ({iso_yield:.2e} cps) "
                    f"of {sri['total_yield']:.2e} total")
            if iso.background_continuous_Hz is not None:
                out(f"  Background: {iso.background_continuous_Hz:.0f} Hz cont. "
                    f"\u00d7 {iso.background_tof_gate_us:.1f} \u00b5s TOF gate / "
                    f"{cfg.scan.dwell_time_ms:.0f} ms dwell = "
                    f"{iso.background_rate_Hz:.3f} Hz effective")
            else:
                out(f"  Background: {iso.background_rate_Hz} Hz")
            if use_uniform:
                out("  ** Uniform timing: full yield assumed per peak, same time for all peaks **")

        out(f"  Isotope shift: {iso_shift:+.2f} MHz")

        # Voltage-frequency sensitivity at centroid
        _V_c = resonance_voltage(centroid_MHz, nu_laser_MHz, mass_amu,
                                 cfg.beam.charge_state, cfg.beam.geometry)
        _V_c1 = resonance_voltage(centroid_MHz + 0.5, nu_laser_MHz, mass_amu,
                                  cfg.beam.charge_state, cfg.beam.geometry)
        _V_c2 = resonance_voltage(centroid_MHz - 0.5, nu_laser_MHz, mass_amu,
                                  cfg.beam.charge_state, cfg.beam.geometry)
        if _V_c is not None and _V_c1 is not None and _V_c2 is not None:
            _v_per_mhz = abs(_V_c1 - _V_c2)
            _mhz_per_v = 1.0 / _v_per_mhz if _v_per_mhz > 0 else float('inf')
            out(f"  1 V = {_mhz_per_v:.4f} MHz  |  1 MHz = {_v_per_mhz:.4f} V")

        if iso.I == 0:
            out("  Single peak (no HFS)")
        else:
            out(f"  A_l={A_l:.2f} A_u={A_u:.2f} B_l={B_l:.2f} B_u={B_u:.2f} MHz")

        offsets, intens = compute_hfs_lines(
            iso.I, cfg.transition.J_lower, cfg.transition.J_upper,
            A_l, B_l, A_u, B_u,
        )

        dVs = _compute_peak_voltages(offsets, centroid_MHz, nu_laser_MHz,
                                      mass_amu, cfg.beam.charge_state,
                                      V0, cfg.beam.geometry)

        # Collect all peaks for peak-list log
        gs_state = "g.s." if has_isomer else ""
        for off, amp, dv in zip(offsets, intens, dVs):
            all_peaks.append(dict(
                label=iso.label, state=gs_state,
                offset_MHz=float(off), intensity=float(amp),
                dV=dv, V_acc=(V0 - dv) if dv is not None else None,
                iso_shift_MHz=iso_shift,
            ))

        # Timing & peak table
        if cfg.spectrum_only:
            # Spectrum-only: show all peaks, no timing
            measured_dVs = [dv for dv in dVs]
            peak_rows = []
            for i, (off, amp, dv) in enumerate(zip(offsets, intens, dVs), 1):
                dv_str = f"{dv:+.3f}" if dv is not None else "N/A"
                peak_rows.append([i, f"{off:+.1f}", dv_str, f"{amp:.4f}"])
            out(tabulate(
                peak_rows,
                headers=["#", "Offset(MHz)", "dV(V)", "Intensity"],
                tablefmt="pretty",
                colalign=("right", "right", "right", "right"),
            ))
        else:
            timing_results = estimate_isotope_timing(
                offsets, intens, iso.peaks_to_measure,
                gs_yield, iso.spectroscopic_efficiency,
                iso.background_rate_Hz, cfg.required_sigma, scan_params,
                uniform_timing=use_uniform,
            )

            measured_dVs = []
            subtotal_s = 0.0
            peak_rows = []
            for i, tr in enumerate(timing_results, 1):
                idx = np.argmin(np.abs(offsets - tr["offset_MHz"]))
                dv = dVs[idx]
                dv_str = f"{dv:+.3f}" if dv is not None else "N/A"
                measured_dVs.append(dv)

                sc = tr["scan_result"]
                peak_rows.append([
                    i,
                    f"{tr['offset_MHz']:+.1f}",
                    dv_str,
                    f"{tr['intensity']:.4f}",
                    f"{tr['signal_rate_Hz']:.2f}",
                    _fmt_time(tr['t_on_peak_s']),
                    sc['n_sweeps'],
                    _fmt_time(sc['total_time_s']),
                ])
                subtotal_s += sc["total_time_s"]

            out(tabulate(
                peak_rows,
                headers=["#", "Offset(MHz)", "dV(V)", "Intensity",
                         "Signal(Hz)", "t_peak", "Sweeps", "Total"],
                tablefmt="pretty",
                colalign=("right", "right", "right", "right",
                          "right", "right", "right", "right"),
            ))
            out(f"  Subtotal: {_fmt_time(subtotal_s)} "
                f"({subtotal_s / 3600:.2f}h, {subtotal_s / 3600 / 8:.3f} shifts)")
            grand_total_s += subtotal_s

        # Generate spectrum for plotting
        valid_dVs = [dv for dv in dVs if dv is not None]
        if valid_dVs:
            dv_min = min(valid_dVs) - 20
            dv_max = max(valid_dVs) + 20
            # Expand to at least scan range
            dv_min = min(dv_min, -half_range)
            dv_max = max(dv_max, half_range)
        else:
            # No valid voltages - use scan range
            dv_min, dv_max = -half_range, half_range

        dV_arr, int_arr = generate_voltage_spectrum(
            offsets, intens, centroid_MHz, nu_laser_MHz,
            mass_amu, cfg.beam.charge_state, V0, cfg.beam.geometry,
            cfg.linewidth_fwhm_MHz, (dv_min, dv_max), v_steps=2000,
        )

        plot_entry = dict(
            label=f"{iso.label} (g.s.)" if has_isomer else iso.label,
            dV_array=dV_arr,
            intensity_array=int_arr,
            measured_peak_dVs=[v for v in measured_dVs if v is not None],
        )

        # ============ Isomer ============
        if has_isomer:
            isom = iso.isomer
            A_l_m, B_l_m, A_u_m, B_u_m, mu_used_m, mu_source_m = _get_hfs_params(
                iso, cfg.reference, is_isomer=True, isomer_cfg=isom,
                g_s_factor=cfg.g_s_factor
            )
            iso_shift_m = isom.isotope_shift_MHz
            centroid_m = nu0_MHz + iso_shift_m

            # Collect nuclear params for isomer
            nuclear_params.append({
                "label": iso.label,
                "state": f"isomer (I={isom.I})",
                "mass_amu": mass_amu,
                "I": isom.I,
                "mu": mu_used_m,
                "mu_source": mu_source_m,
                "Q": isom.Q,
                "A_l": A_l_m,
                "A_u": A_u_m,
                "B_l": B_l_m,
                "B_u": B_u_m,
            })

            out()
            if cfg.spectrum_only:
                out(f"{iso.label} (isomer, I={isom.I})")
            else:
                iso_yield_note = ""
                if isom.yield_ions_per_sec is not None:
                    iso_yield_note = " [independent]"
                elif spin_ratio_info is not None:
                    iso_yield_note = " [spin dist.]"
                out(f"{iso.label} (isomer, I={isom.I}) | Yield: {iso_yield:.2e} cps{iso_yield_note} | "
                    f"Eff: {iso.spectroscopic_efficiency} ({_fmt_efficiency(iso.spectroscopic_efficiency)})")
                if iso.background_continuous_Hz is not None:
                    out(f"  Background: {iso.background_continuous_Hz:.0f} Hz cont. "
                        f"\u00d7 {iso.background_tof_gate_us:.1f} \u00b5s TOF gate / "
                        f"{cfg.scan.dwell_time_ms:.0f} ms dwell = "
                        f"{iso.background_rate_Hz:.3f} Hz effective")
                else:
                    out(f"  Background: {iso.background_rate_Hz} Hz")
                if use_uniform:
                    out("  ** Uniform timing: full yield assumed per peak, same time for all peaks **")
            out(f"  Isomer shift: {iso_shift_m:+.2f} MHz")

            # Voltage-frequency sensitivity at isomer centroid
            _V_cm = resonance_voltage(centroid_m, nu_laser_MHz, mass_amu,
                                      cfg.beam.charge_state, cfg.beam.geometry)
            _V_cm1 = resonance_voltage(centroid_m + 0.5, nu_laser_MHz, mass_amu,
                                       cfg.beam.charge_state, cfg.beam.geometry)
            _V_cm2 = resonance_voltage(centroid_m - 0.5, nu_laser_MHz, mass_amu,
                                       cfg.beam.charge_state, cfg.beam.geometry)
            if _V_cm is not None and _V_cm1 is not None and _V_cm2 is not None:
                _v_per_mhz_m = abs(_V_cm1 - _V_cm2)
                _mhz_per_v_m = 1.0 / _v_per_mhz_m if _v_per_mhz_m > 0 else float('inf')
                out(f"  1 V = {_mhz_per_v_m:.4f} MHz  |  1 MHz = {_v_per_mhz_m:.4f} V")

            if isom.I == 0:
                out("  Single peak (no HFS)")
            else:
                scaled_note = "Scaled" if isom.A_lower_MHz is None else "Override"
                out(f"  {scaled_note}: A_l={A_l_m:.2f} A_u={A_u_m:.2f} "
                    f"B_l={B_l_m:.2f} B_u={B_u_m:.2f} MHz")

            offsets_m, intens_m = compute_hfs_lines(
                isom.I, cfg.transition.J_lower, cfg.transition.J_upper,
                A_l_m, B_l_m, A_u_m, B_u_m,
            )

            dVs_m = _compute_peak_voltages(offsets_m, centroid_m, nu_laser_MHz,
                                            mass_amu, cfg.beam.charge_state,
                                            V0, cfg.beam.geometry)

            for off, amp, dv in zip(offsets_m, intens_m, dVs_m):
                all_peaks.append(dict(
                    label=iso.label, state=f"isomer (I={isom.I})",
                    offset_MHz=float(off), intensity=float(amp),
                    dV=dv, V_acc=(V0 - dv) if dv is not None else None,
                    iso_shift_MHz=iso_shift_m,
                ))

            if cfg.spectrum_only:
                measured_dVs_m = [dv for dv in dVs_m]
                peak_rows_m = []
                for i, (off, amp, dv) in enumerate(zip(offsets_m, intens_m, dVs_m), 1):
                    dv_str = f"{dv:+.3f}" if dv is not None else "N/A"
                    peak_rows_m.append([i, f"{off:+.1f}", dv_str, f"{amp:.4f}"])
                out(tabulate(
                    peak_rows_m,
                    headers=["#", "Offset(MHz)", "dV(V)", "Intensity"],
                    tablefmt="pretty",
                    colalign=("right", "right", "right", "right"),
                ))
            else:
                timing_m = estimate_isotope_timing(
                    offsets_m, intens_m, isom.peaks_to_measure,
                    iso_yield, iso.spectroscopic_efficiency,
                    iso.background_rate_Hz, cfg.required_sigma, scan_params,
                    uniform_timing=use_uniform,
                )

                measured_dVs_m = []
                subtotal_m = 0.0
                peak_rows_m = []
                for i, tr in enumerate(timing_m, 1):
                    idx = np.argmin(np.abs(offsets_m - tr["offset_MHz"]))
                    dv = dVs_m[idx]
                    dv_str = f"{dv:+.3f}" if dv is not None else "N/A"
                    measured_dVs_m.append(dv)

                    sc = tr["scan_result"]
                    peak_rows_m.append([
                        i,
                        f"{tr['offset_MHz']:+.1f}",
                        dv_str,
                        f"{tr['intensity']:.4f}",
                        f"{tr['signal_rate_Hz']:.2f}",
                        _fmt_time(tr['t_on_peak_s']),
                        sc['n_sweeps'],
                        _fmt_time(sc['total_time_s']),
                    ])
                    subtotal_m += sc["total_time_s"]

                out(tabulate(
                    peak_rows_m,
                    headers=["#", "Offset(MHz)", "dV(V)", "Intensity",
                             "Signal(Hz)", "t_peak", "Sweeps", "Total"],
                    tablefmt="pretty",
                    colalign=("right", "right", "right", "right",
                              "right", "right", "right", "right"),
                ))
                out(f"  Subtotal: {_fmt_time(subtotal_m)} "
                    f"({subtotal_m / 3600:.2f}h, {subtotal_m / 3600 / 8:.3f} shifts)")
                grand_total_s += subtotal_m

            # Isomer spectrum
            valid_dVs_m = [v for v in dVs_m if v is not None]
            if valid_dVs_m:
                dv_min_m = min(valid_dVs_m) - 20
                dv_max_m = max(valid_dVs_m) + 20
            else:
                dv_min_m, dv_max_m = -half_range, half_range
            # Use combined range for overlay
            combined_min = min(dv_min, dv_min_m)
            combined_max = max(dv_max, dv_max_m)

            dV_arr_m, int_arr_m = generate_voltage_spectrum(
                offsets_m, intens_m, centroid_m, nu_laser_MHz,
                mass_amu, cfg.beam.charge_state, V0, cfg.beam.geometry,
                cfg.linewidth_fwhm_MHz, (combined_min, combined_max), v_steps=2000,
            )

            # Re-generate gs spectrum on combined range for overlay
            dV_arr_gs, int_arr_gs = generate_voltage_spectrum(
                offsets, intens, centroid_MHz, nu_laser_MHz,
                mass_amu, cfg.beam.charge_state, V0, cfg.beam.geometry,
                cfg.linewidth_fwhm_MHz, (combined_min, combined_max), v_steps=2000,
            )
            plot_entry["dV_array"] = dV_arr_gs
            plot_entry["intensity_array"] = int_arr_gs

            plot_entry["isomer_label"] = f"{iso.label} (isomer, I={isom.I})"
            plot_entry["isomer_dV"] = dV_arr_m
            plot_entry["isomer_intensity"] = int_arr_m
            plot_entry["isomer_measured_dVs"] = [v for v in measured_dVs_m if v is not None]
            plot_entry["plot_with_gs"] = isom.plot_with_gs

        all_plot_results.append(plot_entry)

    # --- Nuclear parameters summary table ---
    _print_nuclear_params_table(out, cfg.reference, nuclear_params, cfg.g_s_factor)

    # --- Grand total ---
    out()
    out("=" * 64)
    if cfg.spectrum_only:
        out("  Spectrum-only mode: no timing estimates produced.")
    else:
        total_h = grand_total_s / 3600.0
        total_min = grand_total_s / 60.0
        shifts = total_h / 8.0
        if total_h >= 1:
            out(f"GRAND TOTAL: {total_h:.1f}h {(grand_total_s % 3600) / 60:.0f}min "
                f"({shifts:.1f} shifts of 8h)")
        else:
            out(f"GRAND TOTAL: {total_min:.1f}min ({shifts:.2f} shifts of 8h)")
    out("=" * 64)

    # --- Plots ---
    if not no_plot and all_plot_results:
        set_palette(palette)
        p1 = plot_all_cases(all_plot_results, output_dir, file_tag, config_name)
        p2 = plot_combined_overview(all_plot_results, output_dir, file_tag, config_name)
        out(f"\nPlots saved: {p1}")
        out(f"             {p2}")

    # --- Peak list log (both orderings always emitted) ---
    if all_peaks:
        def _format_peak_table(peaks):
            rows = []
            for i, pk in enumerate(peaks, 1):
                dv_s = f"{pk['dV']:+.3f}" if pk["dV"] is not None else "N/A"
                va_s = f"{pk['V_acc']:.3f}" if pk["V_acc"] is not None else "N/A"
                rows.append([
                    i, pk["label"], pk["state"] or "-",
                    f"{pk['iso_shift_MHz']:+.2f}",
                    f"{pk['offset_MHz']:+.2f}", dv_s, va_s,
                    f"{pk['intensity']:.6f}",
                ])
            return tabulate(
                rows,
                headers=["#", "Isotope", "State", "Shift(MHz)",
                         "Offset(MHz)", "dV(V)", "V_acc(V)", "Intensity"],
                tablefmt="pretty",
                colalign=("right", "left", "left", "right",
                          "right", "right", "right", "right"),
            )

        by_intensity = sorted(all_peaks, key=lambda p: p["intensity"],
                              reverse=True)
        out("\n── Peak List (sorted by intensity) ──")
        out(_format_peak_table(by_intensity))

        by_voltage = sorted(all_peaks,
                            key=lambda p: (p["dV"] is None, p["dV"] or 0))
        out("\n── Peak List (sorted by voltage) ──")
        out(_format_peak_table(by_voltage))

        out(f"Total peaks: {len(all_peaks)}")

    out(f"Log saved:   {log_path}")

    return all_plot_results, all_peaks


def main():
    parser = argparse.ArgumentParser(description="CLS Estimation Toolkit")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--palette", default="default",
                        choices=list(PALETTES.keys()),
                        help="Color palette for plots (default: default)")
    parser.add_argument("--peak-list", default=None,
                        choices=["voltage", "intensity"],
                        help="Generate peak_list.log sorted by voltage or intensity")
    args = parser.parse_args()
    run(args.config, args.output_dir, args.no_plot, args.palette, args.peak_list)


if __name__ == "__main__":
    main()
