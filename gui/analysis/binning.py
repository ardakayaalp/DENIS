"""Single source of truth for binning a CLSDataFrame in the Analysis tab.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

`compute_binned(data, cfg)` is the only place that understands the full
binning configuration: bin_definition (Auto / Fixed bin count / Fixed bin
width), bin_count, bin_width_mhz, the silent clstools fallback
(``max(100, sqrt(N))``), x_column, yerr_mode (incl. Poisson low-count
correction), xerr_mode, and all gates (TOF / PMT / V / F). It also builds
the structured binning warnings and re-bins sources onto a common grid for
simultaneous fits.

The cfg argument is the *effective* binning config for one file: the
source-level config with any per-file binning overrides already merged in
by ``effective_binning_config``.

Depends on: standard library and third-party packages only (numpy, pandas,
satlas2, plus clstools' CLSDataFrame at call time).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Child of the "denis" session logger (configured in gui.session_log); using
# the stdlib logger directly keeps this module a gui-dependency-free leaf.
_log = logging.getLogger("denis.binning")

# ── Public defaults (also used by the SourceBlock UI) ──────────────────
BIN_DEFINITIONS: Tuple[str, ...] = ("Auto", "Fixed bin count",
                                    "Fixed bin width", "Per scan step")
YERR_MODES: Tuple[str, ...] = ("None", "Poisson sqrt(y+1)",
                               "Poisson sqrt(y)", "Model-based")
# "Per scan step" is the physically faithful default: one bin per raw
# scanning-voltage step at that step's own Doppler-shifted frequency —
# a uniform ("Auto") grid at ~step width sits exactly at the aliasing
# point (doubled-count spikes; see _per_step_frequency_bins). Loaded
# projects keep whatever their save recorded.
DEFAULT_BIN_DEFINITION = "Per scan step"
DEFAULT_BIN_COUNT = 100
DEFAULT_BIN_WIDTH_MHZ = 10.0

# Minimum bin count for the sqrt(N) fallback used when clstools' default
# binning fails: the fallback uses max(_FALLBACK_FLOOR, sqrt(N)) bins.
_FALLBACK_FLOOR = 100

# Per-file overrides may only touch binning. Physics, gates, mass, refs,
# and laser/cooler overrides stay Source-level — otherwise the per-file
# dialog becomes a second Source block hidden behind every run.
BINNING_OVERRIDE_KEYS = frozenset({
    "bin_mode",
    "x_column",
    "yerr_mode",
    "xerr_mode",
    "bin_definition",
    "bin_count",
    "bin_width_mhz",
    "step_multiple",
})


def intersect_v_gate(
    src: Optional[Tuple[float, float]],
    split: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Intersect two ``(lo, hi)`` voltage gates.

    Either argument may be ``None`` (meaning "no gate on this side"),
    in which case the other is returned. If both are present, returns
    the overlap ``(max(src.lo, split.lo), min(src.hi, split.hi))``.
    Returns ``None`` if the ranges are disjoint or empty — callers
    should treat that as "no events would survive" and abort the fit
    with a clean error rather than computing on zero rows.

    Used by both the pre-analysis preview (``_bin_spectrum``) and the
    fit worker so the displayed spectrum and the fitted spectrum stay
    in lock-step for a virtual-split entry.
    """
    if src is None and split is None:
        return None
    if src is None:
        return tuple(split) if split is not None else None
    if split is None:
        return tuple(src)
    lo = max(float(src[0]), float(split[0]))
    hi = min(float(src[1]), float(split[1]))
    if hi <= lo:
        return None
    return (lo, hi)


def effective_binning_config(source_cfg: Dict[str, Any],
                             override: Optional[Dict[str, Any]]
                             ) -> Dict[str, Any]:
    """Return source_cfg overlaid with the whitelisted override keys.

    Override is treated as a sparse patch: missing keys inherit. Any keys
    outside BINNING_OVERRIDE_KEYS in the override are silently ignored,
    which keeps callers from accidentally smuggling physics changes
    through the per-file path.
    """
    cfg = dict(source_cfg)
    if not override:
        return cfg
    for key, value in override.items():
        if key in BINNING_OVERRIDE_KEYS:
            cfg[key] = value
    return cfg


# ── Internal helpers ───────────────────────────────────────────────────

def _do_compute_bins(data, gates, bins) -> bool:
    """Call clstools Compute_Bins; return True if the sqrt(N) fallback fired."""
    tof_gate, pmt_gate, v_gate, f_gate = gates
    try:
        data.Compute_Bins(TOF_gate=tof_gate, PMT_gate=pmt_gate,
                          V_gate=v_gate, F_gate=f_gate, bins=bins)
        return False
    except (ValueError, ZeroDivisionError):
        fb = max(_FALLBACK_FLOOR, int(getattr(data, "Size", 0) ** 0.5))
        data.Compute_Bins(TOF_gate=tof_gate, PMT_gate=pmt_gate,
                          V_gate=v_gate, F_gate=f_gate, bins=fb)
        return True


def _make_yerr(y: np.ndarray, yerr_mode: str):
    """Return (yerr, use_callable_yerr) for the requested mode.

    "None" returns (None, False) and signals callers to render the
    spectrum without error bars (used by Pre-Analysis defaults).
    """
    if yerr_mode == "None":
        return None, False
    if yerr_mode == "Poisson sqrt(y+1)":
        return np.sqrt(y + 1), False
    if yerr_mode == "Poisson sqrt(y)":
        yerr = np.sqrt(y)
        yerr[yerr == 0] = 1.0
        return yerr, False
    if yerr_mode == "Model-based":
        return np.sqrt, True   # callable; lmfit uses sqrt(model)
    return np.sqrt(y + 1), False  # safe fallback


def _apply_low_count_correction(y: np.ndarray, yerr: np.ndarray) -> np.ndarray:
    """Replace yerr in low-count bins with Poisson-interval half-widths."""
    try:
        import satlas2 as sat
    except ImportError:
        return yerr
    try:
        low_mask = y < 10
        if np.any(low_mask):
            lo, hi = sat.poissonInterval(y[low_mask], sigma=1)
            yerr[low_mask] = (np.array(hi) - np.array(lo)) / 2.0
            yerr[yerr == 0] = 1.0
    except Exception:
        _log.warning(
            "Low-count Poisson interval failed for %d bin(s); keeping "
            "sqrt-based yerr", int(np.count_nonzero(y < 10)), exc_info=True)
    return yerr


def summarize_x(x: np.ndarray) -> Dict[str, Any]:
    """Spacing / range stats for a 1-D x array (in whatever units x has).

    Returned keys: n_bins, dx_median, dx_min, dx_max, x_min, x_max.
    Used by compute_binned and by callers that produce x outside clstools
    (e.g. pre-merged spectra) and want the same shape of summary.
    """
    if len(x) == 0:
        return {"n_bins": 0, "dx_median": 0.0, "dx_min": 0.0, "dx_max": 0.0,
                "x_min": 0.0, "x_max": 0.0}
    sx = np.sort(np.asarray(x, dtype=float))
    if len(sx) < 2:
        return {"n_bins": int(len(sx)), "dx_median": 0.0, "dx_min": 0.0,
                "dx_max": 0.0, "x_min": float(sx[0]), "x_max": float(sx[0])}
    diffs = np.diff(sx)
    good = diffs[diffs > 0]
    return {
        "n_bins": int(len(sx)),
        "dx_median": float(np.median(good)) if len(good) else 0.0,
        "dx_min": float(good.min()) if len(good) else 0.0,
        "dx_max": float(good.max()) if len(good) else 0.0,
        "x_min": float(sx[0]),
        "x_max": float(sx[-1]),
    }


# ── Public API ─────────────────────────────────────────────────────────

def _diagnostics_from_clstools_intervals(data) -> Dict[str, Any]:
    """Pull authoritative bin edges from data.Binned['bins'] (pd.Interval).

    clstools' Compute_Bins uses pd.cut with consecutive non-overlapping
    intervals, so for N bins we read N+1 edges directly. Returns values in
    Hz; caller converts to MHz at the same boundary as ``x``.
    """
    intervals = data.Binned["bins"].dropna().values
    if len(intervals) == 0:
        return {"bin_edges": np.array([]), "bin_centers": np.array([]),
                "bin_widths": np.array([]),
                "edges_source": "clstools_intervals_empty"}
    edges = [float(intervals[0].left)]
    edges.extend(float(iv.right) for iv in intervals)
    edges = np.asarray(edges, dtype=float)
    centers = data.Binned["bins_center"].values.astype(float)
    widths = data.Binned["bins_width"].values.astype(float)
    return {"bin_edges": edges, "bin_centers": centers,
            "bin_widths": widths,
            "edges_source": "clstools_intervals"}


def _capture_raw_events(data, bin_mode, tof_gate, pmt_gate, v_gate,
                         f_gate, n):
    """Sample up to ``n`` gated raw event x-positions from a CLSDataFrame.

    Returns the raw values in their native units (Hz for Frequency mode,
    V for Raw Voltage). Caller is responsible for unit conversion. Tries
    the pandas materialised view (``data.Sorted``) first; falls back to
    materialising ``data.Run`` only if the needed column isn't there.
    Same gating logic as clstools' Compute_Bins.
    """
    import pandas as _pd
    col = "F" if bin_mode == "Frequency" else "DV"

    df = None
    sorted_df = getattr(data, "Sorted", None)
    if isinstance(sorted_df, _pd.DataFrame) and col in sorted_df.columns:
        df = sorted_df
    elif hasattr(data, "Run"):
        try:
            df = data.Run.compute()
        except Exception:
            return np.array([], dtype=float)
    if df is None or col not in df.columns:
        return np.array([], dtype=float)

    if tof_gate is not None and "TOF" in df.columns:
        df = df[(df["TOF"] < max(tof_gate)) & (df["TOF"] > min(tof_gate))]
    if v_gate is not None and "DV" in df.columns:
        df = df[(df["DV"] < max(v_gate)) & (df["DV"] > min(v_gate))]
    if (f_gate is not None and bin_mode == "Frequency"
            and "F" in df.columns):
        df = df[(df["F"] < max(f_gate)) & (df["F"] > min(f_gate))]
    if pmt_gate is not None and "TDC" in df.columns:
        df = df[df["TDC"].isin(list(pmt_gate))]
    if len(df) == 0:
        return np.array([], dtype=float)
    if len(df) > n:
        df = df.sample(n=n, random_state=0)
    return df[col].values.astype(float)


def _per_step_frequency_bins(data, tof_gate, pmt_gate, v_gate, f_gate,
                             want_xerr=False):
    """One frequency bin per scanning-voltage step (no uniform re-histogram).

    Groups the gated per-event frame by ``DV`` -- the raw DAC scanning-voltage
    step -- and places each step at the mean frequency of its own events.
    This is the frequency-domain twin of ``Compute_Raw_Bins``: it keeps the
    scan's native, one-bin-per-step sampling instead of laying an equal-width
    grid over it.

    Why it exists: clstools' Auto frequency binning does ``pd.cut`` into
    ``(maxF-minF)/Frequency_stepsize`` *equal-width* bins. The voltage->frequency
    map is Doppler-nonlinear, so a uniform grid at ~one-bin-per-step width sits
    exactly at the aliasing point -- where the step spacing dips below the bin
    width two adjacent steps land in one bin (a doubled-count spike) and
    elsewhere bins fall empty. Binning per step cannot alias: every step is its
    own bin, none is shared, none is empty.

    Requires ``Compute_WL`` to have populated the per-event ``F`` column (the
    same precondition the clstools frequency path has). Returns
    ``(x_mhz, counts, xerr_mhz|None)`` sorted by frequency; ``xerr`` is the
    per-step spread of event frequencies when ``want_xerr`` is set, else None.
    """
    import pandas as pd

    df = None
    sorted_df = getattr(data, "Sorted", None)
    if (isinstance(sorted_df, pd.DataFrame)
            and {"F", "DV"} <= set(sorted_df.columns)):
        df = sorted_df
    elif hasattr(data, "Run"):
        try:
            df = data.Run.compute()
        except Exception:
            return np.array([]), np.array([]), None
    if df is None or "F" not in df.columns or "DV" not in df.columns:
        return np.array([]), np.array([]), None

    # Same gating as clstools' Compute_Bins / _capture_raw_events.
    if tof_gate is not None and "TOF" in df.columns:
        df = df[(df["TOF"] < max(tof_gate)) & (df["TOF"] > min(tof_gate))]
    if v_gate is not None and "DV" in df.columns:
        df = df[(df["DV"] < max(v_gate)) & (df["DV"] > min(v_gate))]
    if f_gate is not None and "F" in df.columns:
        df = df[(df["F"] < max(f_gate)) & (df["F"] > min(f_gate))]
    if pmt_gate is not None and "TDC" in df.columns:
        df = df[df["TDC"].isin(list(pmt_gate))]
    if len(df) == 0:
        return np.array([]), np.array([]), None

    grp = df.groupby("DV")["F"]
    centers_hz = grp.mean().values
    counts = grp.size().values.astype(float)
    order = np.argsort(centers_hz)
    x = centers_hz[order] / 1e6
    y = counts[order]
    xerr = None
    if want_xerr:
        std_hz = grp.std().values[order]
        xerr = np.nan_to_num(np.asarray(std_hz, dtype=float)) / 1e6
    return x, y, xerr


def _voltage_frequency_scale(data, gates):
    """Empirical scan-voltage ↔ rest-frame-frequency conversion.

    The Doppler map ν(DV) is monotonic and, over a typical CLS scan,
    very nearly linear, so the local slope dν/dV (MHz per scanning
    volt) is what turns a raw-voltage bin size into a rest-frame
    frequency bin size and back. We take it empirically from the run's
    own gated events — group by DAC step ``DV``, average ``F`` (Hz)
    within each step, then use the median of the adjacent-step slopes
    (robust to the odd sparse step). The min/max slopes are reported
    too, so any Doppler nonlinearity across the scan is visible rather
    than hidden behind a single number.

    Returns a dict (empty on any failure — this is diagnostics only):
      mhz_per_volt              : median |dν/dV|              (MHz/V)
      mhz_per_volt_min/max      : slope range across the scan (MHz/V)
      v_step_median             : median raw DV step           (V)
      v_span                    : DV_max − DV_min              (V)
      f_span_mhz                : F_max − F_min                (MHz)
    """
    tof_gate, pmt_gate, v_gate, f_gate = gates
    try:
        import pandas as pd
        df = getattr(data, "Sorted", None)
        if not (isinstance(df, pd.DataFrame)
                and {"F", "DV"} <= set(df.columns)):
            df = data.Run.compute() if hasattr(data, "Run") else None
        if df is None or "F" not in df.columns or "DV" not in df.columns:
            return {}
        if tof_gate is not None and "TOF" in df.columns:
            df = df[(df["TOF"] < max(tof_gate)) & (df["TOF"] > min(tof_gate))]
        if v_gate is not None and "DV" in df.columns:
            df = df[(df["DV"] < max(v_gate)) & (df["DV"] > min(v_gate))]
        if f_gate is not None and "F" in df.columns:
            df = df[(df["F"] < max(f_gate)) & (df["F"] > min(f_gate))]
        if pmt_gate is not None and "TDC" in df.columns:
            df = df[df["TDC"].isin(list(pmt_gate))]
        if len(df) == 0:
            return {}
        grp = df.groupby("DV")["F"].mean()
        dv = np.asarray(grp.index, dtype=float)
        f_mhz = np.asarray(grp.values, dtype=float) / 1e6
        order = np.argsort(dv)
        dv, f_mhz = dv[order], f_mhz[order]
        if len(dv) < 2:
            return {}
        ddv = np.diff(dv)
        good = ddv > 0
        if not np.any(good):
            return {}
        slopes = np.abs(np.diff(f_mhz)[good] / ddv[good])
        slopes = slopes[np.isfinite(slopes)]
        if slopes.size == 0:
            return {}
        return {
            "mhz_per_volt": float(np.median(slopes)),
            "mhz_per_volt_min": float(slopes.min()),
            "mhz_per_volt_max": float(slopes.max()),
            "v_step_median": float(np.median(ddv[good])),
            "v_span": float(dv[-1] - dv[0]),
            "f_span_mhz": float(abs(f_mhz[-1] - f_mhz[0])),
        }
    except Exception:
        return {}


def _uniform_grid_bins(data, cfg, info, bin_def, requested_n,
                       requested_w_mhz, gates):
    """The clstools uniform-grid frequency path (Auto / Fixed count /
    Fixed width). Mutates ``info`` (fallback_used) and returns
    ``(x_mhz, y, xerr_mhz|None, x_label)``. Shared by compute_binned's
    grid branch and the per-step structural fallback."""
    if bin_def == "Fixed bin count":
        info["fallback_used"] = _do_compute_bins(
            data, gates, max(1, requested_n))
    elif bin_def == "Fixed bin width":
        # 1st pass: auto, to learn the gated F range. The raw 'F'
        # column is consumed by clstools' groupby+aggregate inside
        # Compute_Bins, so the post-binning DataFrame only carries
        # `bins_center` / `Fmean` / etc. — read the F range from
        # `bins_center` (always defined, in Hz, regardless of which
        # bins ended up populated).
        fb1 = _do_compute_bins(data, gates, None)
        info["fallback_used"] = fb1
        if not fb1:
            try:
                centers = data.Binned["bins_center"].astype(float)
                fmax = float(centers.max())
                fmin = float(centers.min())
            except Exception:
                # Defensive: fall back to whatever the auto pass
                # produced if bins_center is missing.
                fmax = fmin = 0.0
            width_hz = requested_w_mhz * 1e6
            if width_hz > 0 and fmax > fmin:
                target_n = max(1, int(round((fmax - fmin) / width_hz)))
                fb2 = _do_compute_bins(data, gates, target_n)
                if fb2:
                    info["fallback_used"] = True
    else:  # Auto
        info["fallback_used"] = _do_compute_bins(data, gates, None)

    x_col = cfg.get("x_column", "bins_center")
    x = data.Binned[x_col].values / 1e6  # Hz -> MHz
    y = data.Binned["Fcount"].values.astype(float)

    xerr = None
    if cfg.get("xerr_mode", "None") == "From voltage std":
        try:
            vstd = data.Binned.get("Vstd")
            if vstd is not None:
                xerr = vstd.values.astype(float) / 1e6
        except Exception:
            xerr = None
    return x, y, xerr, "Frequency [MHz]"


def _per_step_possible(data) -> bool:
    """True when the data carries the per-event frame per-step needs.

    Distinguishes "structurally impossible" (no Sorted frame with F/DV —
    fall back to the uniform grid) from "gates removed every event"
    (legitimately empty — no fallback, the spectrum IS empty).
    """
    try:
        import pandas as pd
        df = getattr(data, "Sorted", None)
        return (isinstance(df, pd.DataFrame)
                and {"F", "DV"} <= set(df.columns))
    except Exception:
        return False


def _group_adjacent_steps(x, y, xerr, n):
    """Merge every ``n`` adjacent (sorted) step bins into one.

    Used for the "integer multiple of the raw scan step" bin size: the
    per-step (or unique-DV) bins keep the scan's native sampling, and
    grouping N neighbours coarsens it without ever re-histogramming onto
    a uniform grid (so it cannot alias either). Center = count-weighted
    mean of the member centers; counts sum (yerr is computed from the
    summed counts downstream); xerr combines the within-step spread and
    the between-step spread in quadrature.
    """
    n = int(n or 1)
    m = len(x)
    if n <= 1 or m == 0:
        return x, y, xerr
    gx, gy, gxe = [], [], []
    for i in range(0, m, n):
        xs = np.asarray(x[i:i + n], dtype=float)
        w = np.asarray(y[i:i + n], dtype=float)
        tot = float(w.sum())
        if tot > 0:
            center = float(np.average(xs, weights=w))
            between = float(np.average((xs - center) ** 2, weights=w))
        else:
            center = float(xs.mean())
            between = float(np.var(xs))
        gx.append(center)
        gy.append(tot)
        if xerr is not None:
            xe = np.nan_to_num(np.asarray(xerr[i:i + n], dtype=float))
            if tot > 0:
                within = float(np.average(xe ** 2, weights=w))
            elif len(xe):
                within = float(np.mean(xe ** 2))
            else:
                within = 0.0
            gxe.append(float(np.sqrt(within + between)))
    return (np.asarray(gx, dtype=float), np.asarray(gy, dtype=float),
            np.asarray(gxe, dtype=float) if xerr is not None else None)


def _diagnostics_from_unique_voltages(centers: np.ndarray) -> Dict[str, Any]:
    """For Raw Voltage mode: each unique DV is its own group.

    There are no real bin edges; we synthesise midpoints between
    consecutive centers (padding the outer edges by half the adjacent
    width) and flag this in edges_source so the dialog can label them
    as estimated.
    """
    centers = np.sort(np.asarray(centers, dtype=float))
    if len(centers) == 0:
        return {"bin_edges": np.array([]), "bin_centers": centers,
                "bin_widths": np.array([]),
                "edges_source": "midpoints_between_dv_values_empty"}
    if len(centers) == 1:
        # No neighbour to derive a width from; show a hairline edge.
        c = centers[0]
        return {"bin_edges": np.array([c, c]), "bin_centers": centers,
                "bin_widths": np.array([0.0]),
                "edges_source": "midpoints_between_dv_values_singleton"}
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[1:-1] = midpoints
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    widths = np.diff(edges)
    return {"bin_edges": edges, "bin_centers": centers,
            "bin_widths": widths,
            "edges_source": "midpoints_between_dv_values"}


def compute_binned(data, cfg: Dict[str, Any],
                    include_diagnostics: bool = False,
                    raw_event_sample_size: int = 0) -> Dict[str, Any]:
    """
    Bin a clstools CLSDataFrame into (x, y, yerr, xerr) plus an info dict.

    Frequency mode requires Compute_WL/Shift_Ref to have been called already
    on `data`. Voltage mode requires Compute_Voltages.

    Recognized cfg keys (all optional unless noted):
      bin_mode         : "Frequency" | "Raw Voltage"   (default Frequency)
      bin_definition   : "Auto" | "Fixed bin count" | "Fixed bin width"
      bin_count        : int        (used for Fixed bin count)
      bin_width_mhz    : float      (used for Fixed bin width, freq mode)
      tof_gate, pmt_gate, v_gate, f_gate : standard clstools gates
      x_column         : "bins_center" | "Fmean"       (frequency mode)
      yerr_mode        : "Poisson sqrt(y+1)" | "Poisson sqrt(y)" | "Model-based"
      xerr_mode        : "None" | "From voltage std"   (frequency mode)
      step_multiple    : int >= 1 — group N adjacent native step bins into
                         one (Per-scan-step and Raw-Voltage modes only;
                         uniform-grid definitions use bin_count/width)

    Returns dict with keys:
      x, y                 : ndarray
      yerr                 : ndarray, OR np.sqrt callable when use_callable_yerr
      xerr                 : ndarray | None
      use_callable_yerr    : bool
      x_label              : str  ("Frequency [MHz]" / "Voltage [V]")
      info                 : dict, see below

    info dict contains:
      source                           : "clstools" (binned here)
      bin_mode, bin_definition         : echoed (forced "Auto" for Raw Voltage)
      requested_bin_count              : int | None  (None unless Fixed-count)
      requested_bin_width_mhz          : float | None (None unless Fixed-width)
      n_bins, dx_median, dx_min, dx_max, x_min, x_max  : raw binning summary
      effective_n_bins                 : alias of n_bins (what to show in UI)
      effective_bin_width_mhz          : median spacing in MHz (freq only)
      effective_bin_width_v            : median spacing in V   (voltage only)
      fallback_used                    : True if clstools default failed and
                                         the sqrt(N) fallback was triggered

    Fixed bin width is approximate: clstools only accepts an integer bin
    count, so the requested width drives n = round(range/width). The
    ``effective_*`` keys reflect what was actually produced.
    """
    bin_mode = cfg.get("bin_mode", "Frequency")
    bin_def = cfg.get("bin_definition", DEFAULT_BIN_DEFINITION)
    requested_n = int(cfg.get("bin_count", DEFAULT_BIN_COUNT))
    requested_w_mhz = float(cfg.get("bin_width_mhz", DEFAULT_BIN_WIDTH_MHZ))
    try:
        step_multiple = max(1, int(cfg.get("step_multiple", 1) or 1))
    except (TypeError, ValueError):
        step_multiple = 1
    tof_gate = cfg.get("tof_gate")
    pmt_gate = cfg.get("pmt_gate", [3, 4])
    v_gate = cfg.get("v_gate")
    f_gate = cfg.get("f_gate")

    info: Dict[str, Any] = {
        "source": "clstools",
        "bin_mode": bin_mode,
        "bin_definition": bin_def,
        "fallback_used": False,
        "requested_bin_count": (requested_n
                                if bin_def == "Fixed bin count" else None),
        "requested_bin_width_mhz": (requested_w_mhz
                                    if bin_def == "Fixed bin width" else None),
    }

    if bin_mode == "Raw Voltage":
        # Voltage mode is Auto only: clstools groups unique DV values, so
        # bin_count / bin_width are not natively supported here.
        data.Compute_Raw_Bins(TOF_gate=tof_gate, V_gate=v_gate,
                              PMT_gate=pmt_gate)
        raw = data.Raw_binned
        # Compute_Raw_Bins does groupby('DV').sum(), which DROPS voltage
        # steps that have zero surviving events after TOF/PMT gating.
        # That makes the spectrum step-line jump across the missing steps
        # (overlapping diagonals) and hides the gate's effect (no visible
        # zero bins). Reindex onto the COMPLETE set of scanning-voltage
        # steps so emptied steps reappear as explicit 0 bins on a fixed
        # voltage grid. Counts are preserved; only zero-fill rows are added.
        try:
            import pandas as _pd  # noqa: F401  (local, keeps module light)
            src_df = getattr(data, "Sorted", None)
            if src_df is None or "DV" not in getattr(src_df, "columns", []):
                src_df = data.Run.compute()
            full_dv = np.unique(np.asarray(src_df["DV"], dtype=float))
            if v_gate is not None:
                full_dv = full_dv[(full_dv >= min(v_gate))
                                  & (full_dv <= max(v_gate))]
            if len(full_dv) >= len(raw):
                raw = raw.reindex(full_dv, fill_value=0)
        except Exception:
            # Any mismatch (float index drift, missing column) → keep the
            # un-reindexed result rather than risk dropping real counts.
            pass
        x = raw.index.values.astype(float)
        y = raw["counts"].values.astype(float)
        x_label = "Voltage [V]"
        info["voltage_forced_auto"] = (bin_def not in ("Auto", "Per scan step"))
        info["bin_definition"] = "Auto"
        xerr = None
        if step_multiple > 1:
            x, y, xerr = _group_adjacent_steps(x, y, xerr, step_multiple)
            info["step_multiple"] = step_multiple
    elif bin_def == "Per scan step":
        # One frequency bin per scanning-voltage step -- see
        # _per_step_frequency_bins. Sidesteps the uniform-grid aliasing that
        # makes an Auto frequency view show a spurious doubled-count spike.
        x, y, xerr = _per_step_frequency_bins(
            data, tof_gate, pmt_gate, v_gate, f_gate,
            want_xerr=(cfg.get("xerr_mode", "None") == "From voltage std"))
        if len(x) == 0 and not _per_step_possible(data):
            # Structural absence (no per-event F/DV frame) — not a gate
            # emptying the spectrum. Fall back to the uniform grid so the
            # per-step DEFAULT never blanks data the Auto path can bin.
            _log.warning(
                "Per-scan-step binning unavailable (no per-event F/DV "
                "frame); falling back to the Auto grid.")
            info["per_step_fallback_auto"] = True
            info["bin_definition"] = "Auto"
            x, y, xerr, x_label = _uniform_grid_bins(
                data, cfg, info, "Auto", requested_n, requested_w_mhz,
                (tof_gate, pmt_gate, v_gate, f_gate))
        else:
            x_label = "Frequency [MHz]"
            info["per_step"] = True
            if step_multiple > 1:
                x, y, xerr = _group_adjacent_steps(x, y, xerr,
                                                   step_multiple)
                info["step_multiple"] = step_multiple
    else:
        x, y, xerr, x_label = _uniform_grid_bins(
            data, cfg, info, bin_def, requested_n, requested_w_mhz,
            (tof_gate, pmt_gate, v_gate, f_gate))

    yerr_mode = cfg.get("yerr_mode", "Poisson sqrt(y+1)")
    yerr, use_callable_yerr = _make_yerr(y, yerr_mode)
    if yerr is not None and not use_callable_yerr:
        yerr = _apply_low_count_correction(y, yerr)

    info.update(summarize_x(x))
    info["effective_n_bins"] = info["n_bins"]
    if bin_mode == "Frequency" and info.get("per_step"):
        # Per-step bins are non-uniform (one per DAC step); there is no
        # single bin width, so report the median frequency step spacing.
        info["effective_bin_width_mhz"] = info["dx_median"]
    elif bin_mode == "Frequency":
        # For x_column="bins_center" the dx_median IS the bin width
        # (centers are evenly spaced). For x_column="Fmean" it's the
        # spacing of event-mean frequencies, NOT the bin width -- bins
        # with clustered events on one side give a misleading number.
        # Use the authoritative bin widths from clstools.
        try:
            widths_hz = data.Binned["bins_width"].dropna().values
            widths_mhz = np.asarray(widths_hz, dtype=float) / 1e6
            info["effective_bin_width_mhz"] = float(np.median(widths_mhz))
        except Exception:
            info["effective_bin_width_mhz"] = info["dx_median"]
    else:
        info["effective_bin_width_v"] = info["dx_median"]

    out = {
        "x": x, "y": y, "yerr": yerr, "xerr": xerr,
        "use_callable_yerr": use_callable_yerr,
        "x_label": x_label, "info": info,
    }
    if include_diagnostics:
        # Scan-voltage ↔ rest-frame-frequency scale (dν/dV) plus the
        # bin size expressed in BOTH domains, so the dialog can show
        # the raw-voltage bin size, the 1 V → MHz factor, and the
        # Doppler-shifted rest-frame bin size side by side.
        scale = _voltage_frequency_scale(
            data, (tof_gate, pmt_gate, v_gate, f_gate))
        info.update(scale)
        mpv = scale.get("mhz_per_volt")
        eff_v = info.get("effective_bin_width_v")
        eff_mhz = info.get("effective_bin_width_mhz")
        if bin_mode == "Raw Voltage":
            # Bin size is native volts; project to rest-frame MHz.
            info["bin_width_v"] = eff_v
            if eff_v is not None and mpv:
                info["bin_width_rest_mhz"] = eff_v * mpv
        else:
            # Frequency axis IS the rest-frame; back-project to volts.
            info["bin_width_rest_mhz"] = eff_mhz
            if eff_mhz is not None and mpv:
                info["bin_width_v"] = eff_mhz / mpv
        if bin_mode == "Raw Voltage" or info.get("per_step"):
            # Per-step frequency bins have no clstools intervals; synthesise
            # edges from the (non-uniform) centers, same as voltage mode.
            out["diagnostics"] = _diagnostics_from_unique_voltages(
                np.asarray(x, dtype=float))
            if info.get("per_step"):
                out["diagnostics"]["edges_source"] = "per_scan_step_frequency"
        else:
            try:
                d = _diagnostics_from_clstools_intervals(data)
                # x is in MHz; clstools intervals are in Hz. Match units.
                d["bin_edges"] = d["bin_edges"] / 1e6
                d["bin_centers"] = d["bin_centers"] / 1e6
                d["bin_widths"] = d["bin_widths"] / 1e6
            except Exception:
                # Fallback: synthesise edges from the returned centers so
                # the dialog still has something to draw.
                centers = np.asarray(x, dtype=float)
                d = _diagnostics_from_unique_voltages(centers)
                d["edges_source"] = "reconstructed_from_centers"
            out["diagnostics"] = d
        if raw_event_sample_size > 0:
            try:
                sample = _capture_raw_events(
                    data, bin_mode, tof_gate, pmt_gate, v_gate, f_gate,
                    n=int(raw_event_sample_size))
                if bin_mode == "Frequency":
                    out["diagnostics"]["raw_x_sample"] = sample / 1e6
                    out["diagnostics"]["raw_x_unit"] = "MHz"
                else:
                    out["diagnostics"]["raw_x_sample"] = sample
                    out["diagnostics"]["raw_x_unit"] = "V"
                out["diagnostics"]["raw_x_sample_size"] = int(len(sample))
            except Exception:
                # Rug is optional; never block the main result.
                pass
    return out


# ── Warnings ────────────────────────────────────────────────────────────
#
# Warning shape: {"code": str, "level": "warning"|"info",
#                 "run": str|None, "message": str}.
# `run = None` means a cross-run warning (e.g. dx spread).

# Effective vs. requested bin width is allowed to drift this much before
# we surface a warning. clstools converts width -> integer bin count, so a
# few percent of slop is normal; >5% means range/width didn't divide evenly.
_WIDTH_MISMATCH_TOL = 0.05
# Simultaneous-fit dx_median spread that triggers a heads-up.
_DX_SPREAD_THRESHOLD = 2.0


def build_binning_warnings(per_run_infos):
    """Return a list of structured binning warnings.

    per_run_infos: iterable of (run_label, info_dict). info_dict is the
    ``info`` block returned by compute_binned (or the merged-data
    equivalent built by the fit worker).
    """
    warnings = []
    dx_values = []   # for cross-run spread check (frequency only)

    for label, info in per_run_infos:
        if not info:
            continue

        # Forward voltage-to-frequency projection notes / spread
        # warnings the merge module produced at fit time, so the fit
        # report header and the binning-dialog warnings tab both surface
        # them through the same channel as ordinary binning warnings.
        for w in (info.get("voltage_to_frequency_warnings") or []):
            warnings.append(dict(w))

        if info.get("fallback_used"):
            warnings.append({
                "code": "BINNING_FALLBACK",
                "level": "warning",
                "run": label,
                "message": (f"clstools default binning failed; sqrt(N) "
                            f"fallback used ({info.get('effective_n_bins')} "
                            f"bins)."),
            })

        if info.get("voltage_forced_auto"):
            warnings.append({
                "code": "RAW_VOLTAGE_FORCED_AUTO",
                "level": "info",
                "run": label,
                "message": ("Raw Voltage mode groups unique DV values; "
                            "bin_definition forced to Auto."),
            })

        req_w = info.get("requested_bin_width_mhz")
        eff_w = info.get("effective_bin_width_mhz")
        if (req_w and eff_w
                and abs(eff_w - req_w) / req_w > _WIDTH_MISMATCH_TOL):
            warnings.append({
                "code": "EFFECTIVE_WIDTH_MISMATCH",
                "level": "info",
                "run": label,
                "message": (f"Requested {req_w:.3f} MHz bin width; "
                            f"effective {eff_w:.3f} MHz "
                            f"({100 * (eff_w - req_w) / req_w:+.1f}%)."),
            })

        if (info.get("bin_mode") == "Frequency"
                and info.get("effective_bin_width_mhz", 0) > 0):
            dx_values.append((label, info["effective_bin_width_mhz"]))

    if len(dx_values) >= 2:
        widths = [w for _, w in dx_values]
        spread = max(widths) / min(widths)
        if spread > _DX_SPREAD_THRESHOLD:
            lo_label = min(dx_values, key=lambda lw: lw[1])[0]
            hi_label = max(dx_values, key=lambda lw: lw[1])[0]
            warnings.append({
                "code": "DX_SPREAD_LARGE",
                "level": "warning",
                "run": None,
                "message": (f"Effective bin width spread is {spread:.2f}x "
                            f"(narrowest {min(widths):.3f} MHz "
                            f"@ {lo_label}, widest {max(widths):.3f} MHz "
                            f"@ {hi_label}). In a simultaneous fit this "
                            f"changes statistical weights between runs."),
            })

    # Common-grid (simultaneous) was applied → one informational line.
    cg_widths = [info.get("common_grid_width")
                 for _, info in per_run_infos
                 if info and info.get("common_grid_used")]
    if cg_widths:
        warnings.append({
            "code": "COMMON_GRID_USED",
            "level": "info",
            "run": None,
            "message": (f"Common-grid simultaneous fit: every source "
                        f"re-binned onto a shared grid at width "
                        f"{cg_widths[0]:.4f}. xerr (if any) was dropped."),
        })

    return warnings


def rebin_to_common_grid(per_source_xy, bin_width=None):
    """Re-bin each source's (x, y) onto a single shared grid.

    Used by the simultaneous fit's "Common grid" option. Each source keeps
    its own y values (statistics are not merged across runs), but every
    source ends up on the same x edges so the chi-square weights one bin
    equally across all runs.

    Parameters
    ----------
    per_source_xy : list of (x, y) tuples
        The per-source binned data, in matching units (all MHz or all V).
        Caller is responsible for ensuring units match.
    bin_width : float, optional
        Width of the common grid (same units as x). If None, uses the
        median dx_median across sources.

    Returns
    -------
    rebinned : list of (x_centers, y_summed) per source, in the same order.
    common_centers : ndarray of bin centers used.
    bin_width_used : float
    """
    if not per_source_xy:
        return [], np.array([]), 0.0

    # Pick bin width from median spacing across sources if unspecified.
    if bin_width is None or bin_width <= 0:
        spacings = []
        for x, _ in per_source_xy:
            if len(x) > 1:
                d = np.diff(np.sort(np.asarray(x, dtype=float)))
                d = d[d > 0]
                if len(d):
                    spacings.append(float(np.median(d)))
        bin_width = float(np.median(spacings)) if spacings else 1.0

    # Common edges over the union of all source ranges, padded half a bin
    # so the extreme samples land in real bins, not on edges.
    x_lo = min(float(np.min(x)) for x, _ in per_source_xy) - 0.5 * bin_width
    x_hi = max(float(np.max(x)) for x, _ in per_source_xy) + 0.5 * bin_width
    edges = np.arange(x_lo, x_hi + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rebinned = []
    for x, y in per_source_xy:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        # Assign each sample to its bin via np.searchsorted, an
        # O(N log M) lookup over the M bin edges.
        idx = np.clip(np.searchsorted(edges, x_arr, side="right") - 1,
                      0, len(centers) - 1)
        y_new = np.zeros(len(centers), dtype=float)
        np.add.at(y_new, idx, y_arr)
        rebinned.append((centers.copy(), y_new))

    return rebinned, centers, bin_width


def format_binning_warnings_text(warnings):
    """Render warnings for the fit_report.txt header. Empty string if none."""
    if not warnings:
        return ""
    lines = ["", "=" * 20 + " Binning Warnings " + "=" * 20]
    for w in warnings:
        tag = "WARN" if w["level"] == "warning" else "INFO"
        run_part = f" [{w['run']}]" if w.get("run") else ""
        lines.append(f"  [{tag}] {w['code']}{run_part}: {w['message']}")
    lines.append("")
    return "\n".join(lines)
