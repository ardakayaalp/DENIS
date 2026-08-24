"""Single source of truth for loading and physically interpreting a run.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

`prepare_run_data` performs the load -> interpret -> Doppler-prepare sequence
that every data path in the Analysis tab needs before binning: load the ASDF
(or a virtual-split parent), **apply the run's voltage calibration**, capture
per-run metadata, apply cooler/laser overrides, run ``Compute_Voltages``,
compose the effective binning config (intersecting a virtual split's V-gate),
and — for Frequency/Wavenumber modes — run ``Compute_WL`` followed by the
*composed* ``Shift_Ref(ref_freq + ref_shift)``.

The calibration step must land between ``Load_Run`` and ``Compute_Voltages``:
the latter turns each event's DAC value into a voltage using ``Cal`` /
``Cal_order`` / ``VAccDiv`` and nothing else, so overwriting ``Cal`` first is
what makes a per-run calibration override take effect at all (and what makes
the app's numbers independent of which clstools is installed — upstream's
newer build silently drops calibration points at load time).

This used to be copy-pasted inline into the fit worker (separate and
simultaneous), the auto-fitter loader, the merge backend, and the Source-block
preview and binning-diagnostics dialogs. The copies drifted: the auto-fitter
omitted ``ref_shift`` entirely, the preview/diagnostics applied a *bare*
``ref_shift`` (wiping the rest-frame calibration), and the merge backend
skipped the ``Vcool_init``/``VCoolDiv`` fix under a cooler override. Routing
every caller through this one function makes those agree by construction; the
``compute_binned`` call, scan-filter context, centroid correction, and ToF
histogram remain the caller's responsibility.

Depends on: gui.analysis.binning (effective_binning_config, intersect_v_gate)
and clstools' CLSDataFrame at call time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from gui.analysis.binning import effective_binning_config, intersect_v_gate


def prepare_run_data(
    run_file_path: str,
    source_config: Dict[str, Any],
    *,
    binning_override: Optional[Dict[str, Any]] = None,
    split_descriptor: Optional[Dict[str, Any]] = None,
    apply_ref_shift: bool = True,
    calibrations: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Load and interpret a run; return ``(data, eff_cfg, run_metadata)``.

    ``run_file_path`` identifies the run (for a virtual split the ASDF events
    live in ``split_descriptor['parent_path']``). ``binning_override`` is the
    per-file override dict merged into the effective config. ``apply_ref_shift``
    can be set False by callers (e.g. some diagnostics) that deliberately want
    the un-shifted rest-frame axis; the default applies the composed shift so
    the prepared axis matches the actual fit.

    ``calibrations`` is the whole ``path -> spec`` map from the calibration
    registry, not just this run's entry -- a ``borrow`` spec has to look its
    donor up. It is passed explicitly rather than read from the registry
    singleton because this function also runs inside the fit subprocess, where
    that singleton is empty; falling back to it there would silently fit with
    the *uncorrected* calibration, which is precisely the failure this feature
    exists to prevent. In-process callers pass
    ``gui.calibration.get_registry().to_dict()``.

    The returned ``data`` is a clstools ``CLSDataFrame`` with Voltages and —
    for non-Raw-Voltage modes — the rest-frame frequency column computed and
    referenced. ``eff_cfg`` is the effective binning config (with any virtual
    split V-gate intersected in). ``run_metadata`` holds the pre-override
    cooler/laser/timing fields the fit worker records.
    """
    is_split = split_descriptor is not None
    data_path = (split_descriptor["parent_path"]
                 if is_split else run_file_path)
    split_md = (split_descriptor.get("metadata_override", {})
                if is_split else {})

    import clstools as cls_module
    from gui.calibration import load_run_calibrated

    data = cls_module.CLSDataFrame()
    # Load the run and immediately overwrite data.Cal with the calibration the
    # user actually chose -- strictly before Compute_Voltages below, which
    # reads Cal/Cal_order/VAccDiv and nothing else. That ordering IS the
    # feature. A virtual split keys off the parent ASDF, since the calibration
    # belongs to the file the events came from.
    cal_result = load_run_calibrated(
        data, data_path, calibrations,
        cal_order=source_config.get("cal_order", 1))

    # Per-run metadata captured BEFORE overrides (overrides change the cooler
    # value; the recorded value should be what the file actually held).
    run_metadata = {
        # What calibration produced this run's voltages, for the audit trail.
        "calibration": cal_result.to_report_dict(),
        "cooler_v": float(getattr(data, "Vcool_init", 0)
                          * getattr(data, "VCoolDiv", 10000)),
        "laser_set": float(getattr(data, "Laser_set", 0)),
        "daq_time": float(getattr(data, "DAQTStime", 0)),
        "n_events": int(getattr(data, "Size", 0)),
        "ts_start": float(getattr(data, "TSstart", 0) or 0),
        "ts_stop": float(getattr(data, "TSstop", 0) or 0),
        "date": str(getattr(data, "Date", "") or ""),
    }

    # Cooler/laser override precedence:
    # SourceBlock global override > per-split metadata override > ASDF value.
    src_cooler = (source_config.get("cooler_override", 0)
                  if source_config.get("override_enabled") else 0)
    cooler_override = src_cooler if src_cooler > 0 else float(
        split_md.get("cooler_v", 0) or 0)
    if cooler_override > 0:
        data.VCoolDiv = 0
        data.VCoolOffset = cooler_override
    data.Compute_Voltages(
        cooler_correction=source_config.get("cooler_correction", "pbp"))

    src_laser = (source_config.get("laser_override", 0)
                 if source_config.get("override_enabled") else 0)
    laser_override = src_laser if src_laser > 0 else float(
        split_md.get("laser_sp", 0) or 0)
    if laser_override > 0:
        data.Laser_set = laser_override

    noise_filter = source_config.get("noise_filter", 0)
    if noise_filter > 0:
        data.apply_filter(filter_window=noise_filter)

    # Effective config first so per-file overrides (notably bin_mode) decide
    # the Frequency vs Raw Voltage branch below.
    eff_cfg = effective_binning_config(source_config, binning_override)
    if is_split:
        src_vg = eff_cfg.get("v_gate")
        composed = intersect_v_gate(
            tuple(src_vg) if src_vg else None,
            (split_descriptor["split_lo"], split_descriptor["split_hi"]))
        if composed is None:
            raise ValueError(
                f"Split V-gate "
                f"[{split_descriptor['split_lo']:.3f}, "
                f"{split_descriptor['split_hi']:.3f}] does not overlap the "
                f"Source-block V-gate {src_vg!r}; no events would survive.")
        eff_cfg = dict(eff_cfg)
        eff_cfg["v_gate"] = list(composed)

    bin_mode = eff_cfg.get("bin_mode", "Frequency")
    if bin_mode != "Raw Voltage":
        # Fix Vcool_init so Compute_WL derives Frequency_stepsize correctly
        # when a cooler override is active, then restore VCoolDiv.
        if cooler_override > 0:
            data.Vcool_init = cooler_override
            data.VCoolDiv = 1
        data.Compute_WL(
            Mass=source_config["mass"],
            ref=source_config["ref_freq"],
            harmonic=source_config["harmonic"],
        )
        if cooler_override > 0:
            data.VCoolDiv = 0
        if apply_ref_shift:
            ref_shift = source_config.get("ref_shift", 0.0)
            if ref_shift != 0.0:
                # Compose with ref_freq -- clstools.Shift_Ref does
                # ``self.Reference = ref`` (assignment), so passing just
                # ref_shift would discard the rest-frame transition
                # calibration set by Compute_WL.
                data.Shift_Ref(ref=source_config["ref_freq"] + ref_shift)

    return data, eff_cfg, run_metadata
