"""The contents of an ASDF run file, read for inspection -- no Qt.

Used by Tools ▸ ASDF Viewer (gui/asdf_viewer.py). Kept free of Qt and of
the ``gui`` package so the viewer's worker processes, which read many
files in parallel, start by importing only numpy and asdf.

FIELD_DOCS and COLUMN_DOCS say what each header entry and event-table
column is, in which unit, and how clstools / DENIS use it. They were
written from the code that reads these files (clstools.CLSDataFrame and
gui/scan_filter.py), not from memory: an entry that DENIS never uses is
described as such rather than given a guessed unit.
"""
import datetime as _dt
import os

import numpy as np

META_KEYS = ("asdf_library", "history")
#: Distinct values are listed when a column has at most this many.
MAX_LISTED_VALUES = 12
#: clstools' scalings (CLSDataFrame defaults): stored value x factor = V.
COOLER_SCALE = 10000        # VCoolDiv: CoolerVoltage, raw 'cooler'
READBACK_SCALE = 1000       # VAccDiv: CalReadback and the fitted scan voltage

FIELD_DOCS = {
    "Run": "Run number.",
    "Date": "Start of the run, as written by the acquisition.",
    "Experiment": "Experiment label written by the acquisition.",
    "CoolerVoltage": (
        "Cooler (RFQ) platform voltage at the start of the run, divided "
        f"by {COOLER_SCALE:,}: x {COOLER_SCALE:,} = V. The per-event "
        "value is the 'cooler' column of raw."),
    "LaserSetpoint": (
        "Laser setpoint in cm⁻¹ (the fundamental; the Source "
        "block's harmonic multiplies it)."),
    "MassAMU": (
        "Mass in u recorded by the acquisition. DENIS uses the Source "
        "block's isotope mass for the Doppler shift, not this value."),
    "MassSetPoint": (
        "Recorded by the acquisition. DENIS does not use it, and the file "
        "does not state its unit."),
    "BunchesPerChannel": (
        "Ion bunches recorded at each voltage step. With ScanningRanges "
        "and StepSize it gives the bunches per scan, which is how DENIS's "
        "scan filter splits a run into scans."),
    "DwellTime": (
        "Recorded by the acquisition. DENIS only displays it; the file "
        "does not state its unit."),
    "StepSize": "Scan-voltage step in V (setpoint units, as 'voltage').",
    "ScanningRanges": (
        "The scanned setpoint ranges [start, stop] in V, in the order "
        "scanned: the values the 'voltage' column steps through."),
    "SortingUtilities": "Software that sorted the raw data into this file.",
    "CalSet": (
        "Voltage-calibration points: the scan-voltage setpoints in V."),
    "CalReadback": (
        "Voltage-calibration points: the voltage read back at each "
        f"setpoint, divided by {READBACK_SCALE:,} (x {READBACK_SCALE:,} = "
        "V). The fit of CalReadback against CalSet turns a setpoint into "
        "the applied scan voltage."),
}

COLUMN_DOCS = {
    "timestamp": (
        "Event time as a Unix timestamp: seconds since 1970-01-01 00:00 "
        "UTC; the fraction is sub-second. Hover a cell for the date and "
        "time. clstools: TS."),
    "voltage": (
        "Scan-voltage setpoint of the step the event was recorded at, in "
        "V (the same values as CalSet). The voltage actually applied is "
        f"the calibration fit at this setpoint x {READBACK_SCALE:,}; the "
        "ions' voltage is then cooler x "
        f"{COOLER_SCALE:,} minus that. clstools: DV."),
    "bunch_number": (
        "Index of the ion bunch the event came in, counted from the start "
        "of the run and not reset between scans. Scan number = "
        "bunch_number // (steps per scan x BunchesPerChannel), as DENIS's "
        "scan filter derives it. clstools: Bunch."),
    "channel": (
        "PMT channel (TDC input) that detected the photon. The Source "
        "block's PMT gate selects which ones count. clstools: TDC."),
    "time": (
        "Time of flight in µs, from the bunch release to the photon "
        "detection. The ToF gate keeps the window in which the ion bunch "
        "is in front of the PMTs. clstools: TOF."),
    "cooler": (
        "Cooler (RFQ) platform voltage monitor at the time of the event, "
        f"divided by {COOLER_SCALE:,}: x {COOLER_SCALE:,} = V. The Source "
        "block's cooler correction 'pbp' uses it event by event, 'mean' "
        "the run average. clstools: Vrfq."),
}


def _plain(value):
    """asdf tagged objects -> plain Python / numpy, arrays loaded."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.array(value)
    return value


def value_text(value):
    """How a header value is shown: as stored, dates in ISO form."""
    if isinstance(value, _dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, np.ndarray):
        return f"array {value.shape} {value.dtype}"
    return str(value)


def type_text(value):
    if isinstance(value, np.ndarray):
        return f"ndarray {value.dtype}"
    if isinstance(value, (list, tuple)):
        return f"list[{len(value)}]"
    return type(value).__name__


def number_text(v):
    """15 significant figures: enough for a Unix timestamp's fraction of
    a second (1778236192.93), without float noise (0.1, not
    0.10000000000000001)."""
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.15g}"
    return str(v)


def cell_note(column, value):
    """A converted reading of one cell, for its tooltip ("" if none)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if column == "timestamp":
        t = _dt.datetime.fromtimestamp(v, tz=_dt.timezone.utc)
        return (f"{t:%Y-%m-%d %H:%M:%S.%f}"[:-3] + " UTC  ("
                f"{t.astimezone():%H:%M:%S} local)")
    if column in ("cooler", "CoolerVoltage"):
        return f"{v * COOLER_SCALE:,.2f} V"
    if column == "CalReadback":
        return f"{v * READBACK_SCALE:,.3f} V"
    return ""


def _tables(tree):
    tables = []
    by_length = {}
    for name, value in tree.items():
        if not isinstance(value, np.ndarray) or value.size == 0:
            continue
        if value.ndim == 2:
            names = tree.get(f"{name}_header")
            if not (isinstance(names, list)
                    and len(names) == value.shape[1]):
                names = [f"{name}[{i}]" for i in range(value.shape[1])]
            tables.append({"name": name, "columns": [str(n) for n in names],
                           "data": value})
        elif value.ndim == 1:
            by_length.setdefault(len(value), []).append((name, value))
    for _length, arrays in sorted(by_length.items()):
        numeric = all(np.issubdtype(a.dtype, np.number) for _, a in arrays)
        tables.append({
            "name": " + ".join(n for n, _ in arrays),
            "columns": [n for n, _ in arrays],
            "data": (np.column_stack([a for _, a in arrays]) if numeric
                     else np.array([list(r) for r in
                                    zip(*[a for _, a in arrays])],
                                   dtype=object)),
        })
    # The event table first: it is what the viewer is opened for.
    tables.sort(key=lambda t: (t["data"].ndim != 2 or t["name"] != "raw",
                               -t["data"].shape[0]))
    return tables


def read_asdf_file(source):
    """Header, tables and tree of one plain .asdf file.

    Module-level and Qt-free so it can run in a worker process.
    """
    import asdf
    with asdf.open(source, lazy_load=False, memmap=False) as af:
        tree = {str(k): _plain(v) for k, v in af.tree.items()}
    header = [(k, v) for k, v in tree.items()
              if k not in META_KEYS and not isinstance(v, np.ndarray)]
    return {"parent_path": source, "header": header,
            "tables": _tables(tree), "tree": tree,
            "size_bytes": os.path.getsize(source)}


def column_stats(data):
    """[(min, max, mean, distinct)] per column; distinct is the sorted
    list of values when there are at most MAX_LISTED_VALUES, else the
    count."""
    out = []
    for j in range(data.shape[1]):
        col = data[:, j]
        if not np.issubdtype(np.asarray(col).dtype, np.number):
            out.append((None, None, None, len(set(map(str, col)))))
            continue
        col = np.asarray(col, dtype=float)
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out.append((None, None, None, 0))
            continue
        uniq = np.unique(finite)
        distinct = ([float(u) for u in uniq]
                    if uniq.size <= MAX_LISTED_VALUES else int(uniq.size))
        out.append((float(finite.min()), float(finite.max()),
                    float(finite.mean()), distinct))
    return out
