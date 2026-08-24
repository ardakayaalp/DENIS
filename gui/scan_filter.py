"""Per-file scan filtering: identify, summarize, and exclude scans.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Identifies the individual scans within an ASDF run, summarizes them for
the UI, and keeps a process-wide registry of which scans the user has
excluded from each file. Also provides context managers that temporarily
drop excluded-scan or out-of-time-window events from a clstools data
object so a downstream binning step only sees the kept events.

Depends on: standard library and third-party packages only (numpy,
PySide6; dask/asdf imported lazily inside the relevant helpers).

Scan model
----------
A "scan" is one full sweep of the voltage range. clstools' raw events
carry a monotonically-increasing ``Bunch`` index across the whole run;
the number of bunches per scan is::

    bunches_per_scan = sum(steps_per_range) * BunchesPerChannel
    steps_per_range  = round((V_hi - V_lo) / StepSize) + 1

so the per-event scan index is ``bunch // bunches_per_scan``. This
recovers exact integer scan numbers for files like the example
``run_7740.asdf`` (500 scans of 371 steps), and generalizes to
multi-range experiments (sum the per-range step counts).

Filter sync
-----------
Both Pre-Analysis and Analysis projects can hold the same physical
file. The filter dictionary lives in a single ``ScanFilterRegistry``
keyed by canonical file path, with a ``filters_changed`` signal so any
open dialogs / plots refresh whenever a filter is mutated. The
registry serializes under the YAML's top-level ``scan_filters`` key
on save.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable

import numpy as np

from PySide6.QtCore import QObject, Signal


# ── Path normalization ───────────────────────────────────────────────

def canonical_path(path: str) -> str:
    """Normalize a file path to a single canonical key.

    The same file referenced as ``C:/foo/x.asdf`` and ``C:\\foo\\x.asdf``
    must hit the same registry entry, so we apply the OS's case-folding
    plus ``abspath``. Returns the original string unchanged on TypeError
    (defensive against stray None values from hand-edited YAML).
    """
    if not isinstance(path, str) or not path:
        return path
    try:
        return os.path.normcase(os.path.abspath(path))
    except (TypeError, ValueError):
        return path


# ── Scan index derivation ────────────────────────────────────────────

def bunches_per_scan(scanning_ranges, step_size, bunches_per_channel):
    """Total bunches per full scan across all voltage ranges.

    A scan visits every voltage step in every range; ``BunchesPerChannel``
    multiplies because each step records that many bunches before
    advancing. Returns a positive int, or 0 if the inputs are degenerate
    (e.g. step_size <= 0 or empty ranges) -- callers should treat 0 as
    "scan derivation not possible" and skip filtering.
    """
    if step_size is None or step_size <= 0:
        return 0
    if not scanning_ranges:
        return 0
    total_steps = 0
    for rng in scanning_ranges:
        try:
            lo, hi = float(min(rng)), float(max(rng))
        except (TypeError, ValueError):
            return 0
        steps = int(round((hi - lo) / float(step_size))) + 1
        if steps <= 0:
            return 0
        total_steps += steps
    bpc = int(bunches_per_channel) if bunches_per_channel else 1
    return total_steps * max(1, bpc)


def derive_scan_indices(bunch, scanning_ranges, step_size,
                        bunches_per_channel):
    """Per-event scan index from the raw bunch column.

    ``bunch`` is the per-event bunch number from the ASDF ``raw`` table
    (column 2 in the standard schema). Returns an integer ndarray the
    same length as ``bunch``. If ``bunches_per_scan`` can't be
    determined, returns an all-zeros array (one scan covering
    everything) so downstream code degrades to "no filtering possible".
    """
    bps = bunches_per_scan(scanning_ranges, step_size,
                            bunches_per_channel)
    bunch_arr = np.asarray(bunch, dtype=np.int64)
    if bps <= 0:
        return np.zeros_like(bunch_arr)
    return bunch_arr // bps


def summarize_scans(bunch, timestamp, channel, *, pmt_gate,
                     scanning_ranges, step_size, bunches_per_channel):
    """Build a per-scan summary table from per-event arrays.

    Parameters
    ----------
    bunch, timestamp, channel : ndarray
        Per-event columns from the raw ASDF table.
    pmt_gate : iterable of int
        TDC channels treated as signal (e.g. ``[3, 4]``). DC bins
        stored as 0 inside the gate are honored.
    scanning_ranges, step_size, bunches_per_channel : as for
        ``bunches_per_scan``.

    Returns
    -------
    list of dict, one per scan that has at least one event::

        {
            "scan": int,
            "n_events": int,        # total events in the scan
            "n_pmt": int,           # events passing pmt_gate
            "ts_start": float,      # earliest event timestamp (Unix s)
            "ts_stop": float,       # latest event timestamp
            "duration": float,      # ts_stop - ts_start
            "rate_hz": float,       # n_pmt / duration; 0 if duration==0
        }

    Scans with zero events in the file are omitted; the dialog's
    "scan #" column shows the original index (with gaps) so the user
    can correlate with the raw data.
    """
    scan_idx = derive_scan_indices(
        bunch, scanning_ranges, step_size, bunches_per_channel)
    ts_arr = np.asarray(timestamp, dtype=float)
    ch_arr = np.asarray(channel, dtype=int) if channel is not None \
        else np.zeros_like(scan_idx)
    pmt_set = set(int(c) for c in (pmt_gate or []))

    pmt_mask = np.isin(ch_arr, list(pmt_set)) if pmt_set else np.zeros(
        len(ch_arr), dtype=bool)

    out = []
    if len(scan_idx) == 0:
        return out

    # Group by scan via sorted indices -- much faster than per-scan
    # boolean masks when there are hundreds of scans (e.g. 500 in the
    # example file). Each contiguous run of equal scan_idx values is
    # one group.
    order = np.argsort(scan_idx, kind="stable")
    s_sorted = scan_idx[order]
    ts_sorted = ts_arr[order]
    pmt_sorted = pmt_mask[order]

    edges = np.flatnonzero(np.diff(s_sorted)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(s_sorted)]))

    for lo, hi in zip(starts, stops):
        s = int(s_sorted[lo])
        ts_chunk = ts_sorted[lo:hi]
        ts_start = float(ts_chunk.min())
        ts_stop = float(ts_chunk.max())
        duration = ts_stop - ts_start
        n_events = int(hi - lo)
        n_pmt = int(pmt_sorted[lo:hi].sum())
        rate = (n_pmt / duration) if duration > 0 else 0.0
        out.append({
            "scan": s,
            "n_events": n_events,
            "n_pmt": n_pmt,
            "ts_start": ts_start,
            "ts_stop": ts_stop,
            "duration": duration,
            "rate_hz": rate,
        })
    return out


def make_event_mask(bunch, excluded_scans, scanning_ranges, step_size,
                     bunches_per_channel):
    """Boolean keep-mask: True for events whose scan is NOT in
    ``excluded_scans``.

    Returns an all-True array when ``excluded_scans`` is empty or when
    scan derivation isn't possible (e.g. missing scanning_ranges).
    """
    bunch_arr = np.asarray(bunch, dtype=np.int64)
    if not excluded_scans:
        return np.ones(len(bunch_arr), dtype=bool)
    scan_idx = derive_scan_indices(
        bunch, scanning_ranges, step_size, bunches_per_channel)
    excl = np.fromiter((int(s) for s in excluded_scans),
                       dtype=np.int64,
                       count=len(set(excluded_scans)))
    return ~np.isin(scan_idx, excl)


def read_bunches_per_channel(asdf_path: str) -> int:
    """Cheap metadata-only read of ``BunchesPerChannel`` from an ASDF.

    clstools' ``CLSDataFrame`` doesn't surface this key, but it lives
    at the top of the ASDF tree. Returns ``1`` if the file is missing,
    unreadable, or doesn't carry the key -- 1 is the sane default and
    matches the legacy single-bunch DAQ.
    """
    try:
        import asdf
        with asdf.open(asdf_path) as af:
            v = af.tree.get("BunchesPerChannel", 1)
            return int(v) if v else 1
    except Exception:
        return 1


@contextmanager
def filter_data_for_binning(data, excluded_scans, asdf_path=None):
    """Context manager: temporarily drop excluded-scan events from
    ``data.Run`` so a downstream ``Compute_Bins`` only sees the kept
    events. The original ``data.Run`` / ``data.Sorted`` / ``data.Size``
    are restored on exit, so the same data object can be reused in
    follow-up calls (e.g. PA replots or per-file diagnostics) with no
    side effects.

    Parameters
    ----------
    data : clstools.CLSDataFrame
        Loaded run; must have ``Run`` populated (i.e. ``Load_Run``
        already called).
    excluded_scans : Iterable[int]
        Scans to drop. Empty -> no-op (yields immediately, restoring
        nothing).
    asdf_path : str, optional
        Source ASDF path. Used to read ``BunchesPerChannel`` if it
        isn't already stashed on ``data`` (clstools doesn't load it).
        When omitted and the field is missing, defaults to 1.

    Yields nothing -- callers run their binning inside the ``with``.
    """
    excluded = set(excluded_scans or [])
    run = getattr(data, "Run", None)
    if not excluded or run is None:
        yield
        return

    sr = getattr(data, "ScanningRanges", None)
    step = getattr(data, "Step_Size", None)
    bpc = getattr(data, "BunchesPerChannel", None)
    if bpc is None and asdf_path:
        bpc = read_bunches_per_channel(asdf_path)
    if bpc is None:
        bpc = 1

    bps = bunches_per_scan(sr, step, bpc)
    if bps <= 0:
        # Can't derive scans for this file; transparently fall through
        # so callers don't have to special-case missing metadata.
        yield
        return

    # Materialize once. data.Run may be a Dask DataFrame; .compute()
    # returns a pandas frame regardless. For an already-pandas Run we
    # use it directly.
    try:
        import dask.dataframe as dd
        is_dask = isinstance(run, dd.DataFrame)
    except ImportError:
        dd = None
        is_dask = False

    try:
        df = run.compute() if is_dask else run
    except Exception:
        # If the materialization fails for some reason, prefer "no
        # filtering" over crashing the fit. The user will see all
        # scans included, which is the same behavior they had before
        # this feature.
        yield
        return

    bunch_col = df["Bunch"].values if "Bunch" in df.columns else None
    if bunch_col is None:
        yield
        return
    mask = make_event_mask(bunch_col, excluded, sr, step, bpc)
    if mask.all():
        yield
        return

    df_filt = df[mask].reset_index(drop=True)

    # Save originals so we can restore even if the caller raises.
    orig_run = data.Run
    orig_size = getattr(data, "Size", None)
    orig_sorted = getattr(data, "Sorted", None)
    orig_ts_start = getattr(data, "TSstart", None)
    orig_ts_stop = getattr(data, "TSstop", None)
    orig_daq = getattr(data, "DAQTStime", None)

    try:
        if dd is not None:
            # Match clstools' default partitioning style; one partition
            # per ~1M events keeps memory bounded for very long runs.
            n_part = max(1, len(df_filt) // 1_000_000 + 1)
            data.Run = dd.from_pandas(df_filt, npartitions=n_part)
        else:
            data.Run = df_filt
        data.Size = int(len(df_filt))
        data.Sorted = None  # invalidate; clstools rebuilds on next compute
        if "TS" in df_filt.columns and len(df_filt) > 0:
            data.TSstart = float(df_filt["TS"].min())
            data.TSstop = float(df_filt["TS"].max())
            data.DAQTStime = data.TSstop - data.TSstart
        yield
    finally:
        data.Run = orig_run
        data.Size = orig_size
        data.Sorted = orig_sorted
        data.TSstart = orig_ts_start
        data.TSstop = orig_ts_stop
        data.DAQTStime = orig_daq


@contextmanager
def gate_data_by_timestamp(data, ts_gate):
    """Context manager: temporarily keep only events whose absolute
    timestamp falls inside ``ts_gate = (lo, hi)``, so a downstream
    ``Compute_Bins`` builds the spectrum from just that time window.

    This mirrors the old collinear Data Viewer, where a timestamp gate
    on the bottom plot filters ONLY the spectrum (the ToF and timestamp
    plots themselves still show all events). ``ts_gate`` is in the SAME
    units as ``data.Run["TS"]`` (seconds, absolute). ``None`` / a zero-
    width range is a transparent no-op. Originals are restored on exit.
    """
    run = getattr(data, "Run", None)
    if not ts_gate or run is None:
        yield
        return
    lo, hi = float(min(ts_gate)), float(max(ts_gate))
    if hi <= lo:
        yield
        return

    try:
        import dask.dataframe as dd
        is_dask = isinstance(run, dd.DataFrame)
    except ImportError:
        dd = None
        is_dask = False
    try:
        df = run.compute() if is_dask else run
    except Exception:
        yield
        return
    if "TS" not in df.columns:
        yield
        return

    ts = df["TS"].values
    mask = (ts >= lo) & (ts <= hi)
    if mask.all():
        yield
        return
    df_filt = df[mask].reset_index(drop=True)

    orig_run = data.Run
    orig_size = getattr(data, "Size", None)
    orig_sorted = getattr(data, "Sorted", None)
    try:
        if dd is not None:
            n_part = max(1, len(df_filt) // 1_000_000 + 1)
            data.Run = dd.from_pandas(df_filt, npartitions=n_part)
        else:
            data.Run = df_filt
        data.Size = int(len(df_filt))
        data.Sorted = None
        yield
    finally:
        data.Run = orig_run
        data.Size = orig_size
        data.Sorted = orig_sorted


# ── Registry ─────────────────────────────────────────────────────────

class ScanFilterRegistry(QObject):
    """Process-wide map of file path -> excluded-scan set.

    Owned by MainWindow; accessed via ``get_registry()`` from any tab.
    Mutations emit ``filters_changed`` so open dialogs and per-entry
    badges refresh in lock-step. Serializes under the YAML's top-level
    ``scan_filters`` key.
    """

    filters_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Keys are canonical paths (case-folded + absolute) so that
        # /foo/x.asdf, \foo\x.asdf, ./x.asdf etc. dedupe.
        self._filters: dict[str, set[int]] = {}

    # ── Lookups ────────────────────────────────────────────

    def get(self, path: str) -> set[int]:
        """Return a copy of the excluded-scan set for ``path``.

        Always returns a new set, so callers can mutate it without
        accidentally aliasing the registry's storage. Empty when no
        filter is set.
        """
        key = canonical_path(path)
        return set(self._filters.get(key, set()))

    def has(self, path: str) -> bool:
        key = canonical_path(path)
        return bool(self._filters.get(key))

    def all_paths(self):
        """Iterate over the canonical paths that currently carry a
        non-empty filter. Used by ``to_dict``."""
        return list(self._filters.keys())

    # ── Mutations ──────────────────────────────────────────

    def set(self, path: str, excluded: Iterable[int]):
        """Replace the excluded set for ``path``.

        An empty / falsy ``excluded`` clears the entry entirely so
        ``has(path)`` reads False afterwards. Always emits
        ``filters_changed`` -- callers don't need to debounce.
        """
        key = canonical_path(path)
        new_set = {int(s) for s in (excluded or [])}
        if not new_set:
            self._filters.pop(key, None)
        else:
            self._filters[key] = new_set
        self.filters_changed.emit()

    def clear(self, path: str):
        self.set(path, ())

    def clear_all(self):
        self._filters.clear()
        self.filters_changed.emit()

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize as ``{path: {excluded: [sorted ints]}}``.

        Sorted lists keep the YAML diff-friendly. Empty filters are
        skipped (``set`` with empty input already evicted them).
        """
        out = {}
        for key, scans in self._filters.items():
            if not scans:
                continue
            out[key] = {"excluded": sorted(int(s) for s in scans)}
        return out

    def from_dict(self, d: dict | None):
        """Replace the entire map from a saved dict.

        Tolerates legacy / malformed entries: any value not shaped
        ``{"excluded": [...]}`` is silently dropped, since the YAML
        could have been hand-edited. Always re-emits
        ``filters_changed`` so subscribers refresh after load.
        """
        self._filters.clear()
        if isinstance(d, dict):
            for path, payload in d.items():
                if not isinstance(payload, dict):
                    continue
                excluded = payload.get("excluded")
                if not isinstance(excluded, (list, tuple, set)):
                    continue
                key = canonical_path(path)
                clean = {int(s) for s in excluded
                         if isinstance(s, (int, float))
                         and float(s).is_integer()}
                if clean:
                    self._filters[key] = clean
        self.filters_changed.emit()


# ── Module-level singleton plumbing ──────────────────────────────────
#
# MainWindow installs the registry on first construction; tabs reach
# it via ``get_registry()``. We keep the reference module-global so
# headless tests / Pre-Analysis dialogs that don't see the MainWindow
# directly can still subscribe. Tests that need isolation should call
# ``set_registry(ScanFilterRegistry())`` in setUp.

_REGISTRY: ScanFilterRegistry | None = None


def get_registry() -> ScanFilterRegistry:
    """Return the process-wide registry, creating one on first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ScanFilterRegistry()
    return _REGISTRY


def set_registry(registry: ScanFilterRegistry | None) -> None:
    """Install (or reset) the singleton. Mainly used by tests."""
    global _REGISTRY
    _REGISTRY = registry
