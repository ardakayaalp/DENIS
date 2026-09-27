"""Put a fit result into the GP's drift-free frame.

The GP drift model says where the reference line sat at every moment.
A centroid measured at time t is in the drift-free frame once that
level -- averaged over the run's own acquisition window, G-bar -- has
been subtracted.

That subtraction can happen in one of two places:

* **at fit time**, when the per-file corrections were computed and the
  project re-fitted. The binned x-axis is shifted by G-bar before the
  fit, and ``run_metadata`` records ``centroid_correction_applied``.
* **afterwards**, on the fitted centroid, here.

For a single run the two are identical, not approximately equal. The
fit-time shift is applied *after* binning (``res["x"] -= corr``), so
the bin contents the fitter sees are the same either way. The model
depends on frequency only through (x - centroid), because the HFS
components sit at fixed offsets from it and a polynomial background
merely re-expresses its coefficients under a shift. So the minimum is
the same point, and the fitted centroid is exactly c_raw - G-bar with
the same statistical error.

For a merged spectrum the fit-time route aligns each constituent
before summing; afterwards we can only subtract the count-weighted
G-bar of the constituents. The centroid agrees to first order, the
merged lineshape does not.

Every reader of "the corrected centroid" -- the Isotope Shifts table
and plot, the GP panel's lower plot -- goes through this module. That
way a fit that predates the correction is corrected rather than
silently shown raw, and a fit that was already corrected is never
subtracted twice.
"""
from __future__ import annotations

import math


def source_block(project):
    """First SourceBlock of *project*, or None."""
    if project is None:
        return None
    from gui.analysis.blocks import SourceBlock
    for b in getattr(project, "_blocks", []) or []:
        if isinstance(b, SourceBlock):
            return b
    return None


def result_runs(project, result, src=None):
    """``[(run_num, ts_start, ts_stop, n_events)]`` behind one fit
    result: the constituents for a merged spectrum, otherwise the
    single run itself.

    Read from the Source block's merged table rather than from the
    fit, so it works for a project whose fits predate the correction.
    *src* overrides the Source-block lookup, for callers that already
    have their own (the Isotope Shifts tab resolves it through
    ``_get_source_block``).
    """
    rm = result.get("run_metadata") or {}
    run_file = str(result.get("run_file") or "")
    # A result reloaded from disk has run_file='' -- the merged
    # identity survives only in run_number, which for a merge is the
    # merged name. Try both (2026-09-21).
    name = (run_file[len("merged://"):]
            if run_file.startswith("merged://")
            else str(result.get("run_number") or ""))
    if name:
        if src is None:
            src = source_block(project)
        for entry in (getattr(src, "_file_entries", []) or []):
            md = entry.get("merged_data") or {}
            if not entry.get("is_merged"):
                continue
            if str(md.get("merged_name", "")) != name:
                continue
            out = []
            for pr in md.get("per_run", []) or []:
                out.append((
                    str(pr.get("run_num", "") or "?"),
                    float(pr.get("ts_start", 0) or 0),
                    float(pr.get("ts_stop", 0) or 0),
                    float(pr.get("n_events", 0) or 0)))
            if out:
                return out
            # Matched a merge whose per-run table is gone (legacy
            # save): fall back to its aggregate window rather than
            # pretending it is a single run.
            return [(name, float(rm.get("ts_start", 0) or 0),
                     float(rm.get("ts_stop", 0) or 0), 0.0)]
    return [(str(result.get("run_number", "") or "?"),
             float(rm.get("ts_start", 0) or 0),
             float(rm.get("ts_stop", 0) or 0),
             float(rm.get("n_events", 0) or 0))]


def fit_time_correction(result):
    """The correction the fit already subtracted, in MHz, or None.

    None means the centroid is still in the raw frame -- the thing
    that makes this module necessary.
    """
    rm = result.get("run_metadata") or {}
    if not rm.get("centroid_correction_applied"):
        return None
    try:
        return float(rm.get("centroid_correction_mhz", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def drift_level(rc, project, result, src=None):
    """G-bar for *result*: the GP averaged over its acquisition window.

    Returns ``{"g", "g_sd", "parts", "merged"}`` or None when there is
    no fitted GP or no usable timestamp. ``parts`` is
    ``[(run, ts0, ts1, n_events, g_i, sd_i)]``; for a merge ``g`` is
    the count-weighted mean of the parts (equal weights when the
    counts are unknown), because the merged centroid is itself a
    count-weighted blend of the constituent lines.

    ``g_sd`` is a display figure: for a merge it assumes the parts are
    fully correlated (they are close in time on one smooth curve), so
    it errs large. The Isotope Shifts propagation does not use it; it
    goes through the GP posterior covariance.
    """
    if rc is None or not getattr(rc, "is_fit", False):
        return None
    parts = []
    for run, ts0, ts1, n_ev in result_runs(project, result, src):
        if ts0 <= 0:
            continue
        t1 = ts1 if ts1 > ts0 else ts0
        try:
            g_i, sd_i = rc.predict_interval(ts0 / 3600.0, t1 / 3600.0)
        except Exception:  # noqa: BLE001
            continue
        g_i, sd_i = float(g_i), float(sd_i)
        if not math.isfinite(g_i):
            continue
        parts.append((run, ts0, ts1, n_ev, g_i, sd_i))
    if not parts:
        return None
    counts = [p[3] for p in parts]
    total = sum(counts)
    w = ([c / total for c in counts] if total > 0
         else [1.0 / len(parts)] * len(parts))
    g = sum(wi * p[4] for wi, p in zip(w, parts))
    g_sd = sum(wi * p[5] for wi, p in zip(w, parts))
    return {"g": g, "g_sd": g_sd, "parts": parts,
            "merged": len(parts) > 1}


def frame_centroid(rc, project, result, c_fit, src=None):
    """Where *c_fit* sits in the drift-free frame.

    Returns a dict:

    ``raw``        the centroid before any GP correction
    ``corrected``  the centroid in the drift-free frame, or None when
                   there is no GP to correct it with
    ``shift``      what still has to be subtracted now -- 0 when the
                   fit already did it, G-bar when it did not. The same
                   number shifts the spectrum's x-axis for plotting.
    ``g``          the drift level this result is corrected by
    ``g_sd``       its display uncertainty (see drift_level)
    ``on_the_fly`` True when the correction is being applied here
                   rather than having been applied at fit time
    ``level``      the drift_level() dict, or None
    """
    applied = fit_time_correction(result)
    if applied is not None:
        # Already in the drift-free frame. Honoured as-is: re-deriving
        # it from the current GP would quietly change a number the
        # user may already have published, and for a merged fit the
        # un-correction is not even exact.
        return {"raw": c_fit + applied, "corrected": c_fit,
                "shift": 0.0, "g": applied,
                "g_sd": float((result.get("run_metadata") or {}).get(
                    "centroid_correction_sigma_mhz", 0.0) or 0.0),
                "on_the_fly": False, "level": None}
    lvl = drift_level(rc, project, result, src)
    if lvl is None:
        return {"raw": c_fit, "corrected": None, "shift": 0.0,
                "g": None, "g_sd": None, "on_the_fly": False,
                "level": None}
    return {"raw": c_fit, "corrected": c_fit - lvl["g"],
            "shift": lvl["g"], "g": lvl["g"], "g_sd": lvl["g_sd"],
            "on_the_fly": True, "level": lvl}


def effective_run_metadata(result, frame):
    """``run_metadata`` as it would read had the fit applied the GP.

    The Isotope Shifts propagation (sigma_correction and the GP
    cross-covariance) only takes a run into account when it carries
    ``centroid_correction_applied``. A run corrected on the fly has
    to join those same pools, or its shift would carry the GP's
    central value without the GP's uncertainty. Mirrors what
    fitting.py writes, including the per-constituent list for a
    merge.
    """
    rm = dict(result.get("run_metadata") or {})
    if not frame.get("on_the_fly"):
        return rm
    lvl = frame["level"]
    rm["centroid_correction_applied"] = True
    rm["centroid_correction_mhz"] = frame["g"]
    rm["centroid_correction_sigma_mhz"] = frame["g_sd"]
    rm["centroid_correction_on_the_fly"] = True
    if lvl["merged"]:
        rm["centroid_correction_mode"] = "merged"
        rm["correction_constituents"] = [
            {"run_num": run, "ts_start": ts0, "ts_stop": ts1,
             "n_events": n_ev, "centroid_correction_mode": "Auto",
             "centroid_correction_mhz": g_i,
             "centroid_correction_sigma_mhz": sd_i}
            for run, ts0, ts1, n_ev, g_i, sd_i in lvl["parts"]]
    else:
        rm["centroid_correction_mode"] = "Auto"
        run, ts0, ts1, *_ = lvl["parts"][0]
        rm.setdefault("ts_start", ts0)
        if not rm.get("ts_start"):
            rm["ts_start"] = ts0
    return rm


def observation_label(result):
    """The label a reference observation carries for *result*.

    The run NUMBER when there is one, else the run file's basename.
    Not the other way round: a run fitted in this session has
    ``run_file`` set, the same run reloaded from disk has
    ``run_file = ''``, so a file-based label changed across a reload
    and every exclusion keyed on it silently stopped matching
    (2026-09-22). The GP panel keys exclusions on
    ``"<project>/<label>"``, so every reader asking "is this run
    excluded?" builds it here -- AnalysisProject included.
    """
    import os
    run_num = str(result.get("run_number", "") or "")
    if run_num:
        return run_num
    run_file = str(result.get("run_file") or "")
    return os.path.basename(run_file) if run_file else "run"


def canonical_obs_key(key):
    """Normalise an exclusion key to ``"<project>/<run number>"``.

    Keys saved before 2026-09-22 end in a file name
    (``74Ge_T02/run_7961.asdf``); test and legacy labels may read
    ``run_7961``. All three name the same run, so all three map to
    ``74Ge_T02/7961``. Anything that is not a run file or a
    ``run_<N>`` label -- a merged name, say -- is left as it is.
    """
    import os
    import re
    key = str(key)
    proj, sep, tail = key.rpartition("/")
    stem = os.path.splitext(os.path.basename(tail))[0] \
        if tail.lower().endswith((".asdf", ".h5", ".hdf5")) else tail
    m = re.fullmatch(r"run[_-]?(\d+)", stem, flags=re.IGNORECASE)
    if m:
        stem = m.group(1)
    return f"{proj}{sep}{stem}"


def observation_key(project, result):
    """``"<project>/<run number>"`` -- the GP panel's exclusion key."""
    name = getattr(project, "project_name", None) or getattr(
        project, "_project_name", "")
    return canonical_obs_key(f"{name}/{observation_label(result)}")
