"""Walk a calibration project's fit across a grid of cooler offsets.

The Yb calibration needs the same runs fitted many times over, each
time pretending the cooler sat at a slightly different voltage. Rather
than a second fitting path, this drives the project's OWN fit -- the
one the Run Fit button uses -- with ``_config_overrides["source"]``
set to the offset under test. Whatever the project is configured to do
(binning, gates, model, fitter, splits, merges, scan filters, voltage
calibrations) is what the scan does, and it cannot drift from the real
analysis the way a second copy of the config gathering would.

One offset at a time, sequentially: each fit already runs its own runs
in parallel across cores, and the results have to be attributed to the
offset that produced them.

File output is suppressed for the duration -- a 15-point scan over 5
runs would otherwise write 75 reports, 75 plot sets and 75 CSVs that
nobody asked for. The fitted values live in memory; only the
calibration's own summary is ever written.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from cls_estimations.cooler_calibration import ScanPoint

#: Output-block switches turned off while scanning.
_QUIET_OUTPUT = {
    "report": False, "params_csv": False, "metadata_csv": False,
    "fit_plots": False, "tof_plots": False,
}


def extract_parameter(params_df_dict, name, source_name=None):
    """``(value, sigma)`` of parameter *name*, or ``(None, None)``.

    Mirrors cls_estimations.isotope_shift.extract_centroid, which is
    the same lookup for one particular parameter; hyperfine constants
    need the general form.
    """
    params = params_df_dict.get("Parameter", []) or []
    values = params_df_dict.get("Value", []) or []
    errors = (params_df_dict.get("Stderr")
              or params_df_dict.get("Error") or [])
    sources = params_df_dict.get("Source", [None] * len(params))
    for i, p in enumerate(params):
        if str(p) != name:
            continue
        if source_name is not None and i < len(sources):
            if sources[i] is not None and str(sources[i]) != source_name:
                continue
        try:
            val = float(values[i])
        except (TypeError, ValueError, IndexError):
            return None, None
        try:
            err = float(errors[i])
        except (TypeError, ValueError, IndexError):
            err = float("nan")
        return val, err
    return None, None


class CoolerScan(QObject):
    """Runs (project, offset) jobs one after another.

    ``start`` returns immediately; the work happens on the projects'
    own fit workers. Listen to :attr:`progress`, :attr:`finished` and
    :attr:`failed`.
    """

    progress = Signal(int, int, str)     # done, total, what is running
    finished = Signal(list)              # [ScanPoint]
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: list[tuple] = []
        self._points: list[ScanPoint] = []
        self._done = 0
        self._total = 0
        self._current = None
        self._parameter = "Al"
        self._stop = False
        self._pending = False
        self._connected: set = set()

    # ── control ──
    def start(self, projects, offsets, *, parameter="Al"):
        """*projects* is ``[(project, isotope)]``, *offsets* a list of
        cooler offsets in volts."""
        if self.busy:
            raise RuntimeError("a scan is already running")
        offsets = [float(v) for v in offsets]
        if len(offsets) < 2:
            raise ValueError("a scan needs at least two offsets")
        if not projects:
            raise ValueError("no calibration projects selected")
        self._jobs = [(p, iso, dv) for p, iso in projects
                      for dv in offsets]
        self._points, self._done, self._stop = [], 0, False
        self._total = len(self._jobs)
        self._parameter = parameter
        for project, _iso in projects:
            if project not in self._connected:
                project.results_ready.connect(self._on_results)
                self._connected.add(project)
        # Never finish -- or fail -- inside start(): a caller that
        # connects to the signals on the line after would miss them,
        # and a project that cannot fit does fail immediately.
        self._pending = True
        QTimer.singleShot(0, self._begin)

    def _begin(self):
        self._pending = False
        self._next()

    def stop(self):
        """Finish the fit in flight, then stop."""
        self._stop = True

    @property
    def busy(self):
        return self._current is not None or self._pending

    # ── internals ──
    def _next(self):
        if self._stop or not self._jobs:
            self._finish()
            return
        project, isotope, dv = self._jobs.pop(0)
        self._current = (project, isotope, dv)
        self.progress.emit(
            self._done, self._total,
            f"{project.project_name} @ {dv:+.2f} V")
        project._config_overrides = {
            "source": {"cooler_offset_v": dv},
            "output": dict(_QUIET_OUTPUT),
        }
        try:
            project._on_fit_requested()
        except Exception as exc:                     # noqa: BLE001
            self._abort(f"{project.project_name}: {exc}")
            return
        # _on_fit_requested returns early -- without starting a worker
        # and without emitting anything -- when the project is not
        # fit-ready (no Source block, bad mass, a validation error it
        # already reported). Without this the scan would wait forever
        # for a signal that is never coming.
        worker = getattr(project, "_fit_worker", None)
        if worker is None or not worker.isRunning():
            self._abort(
                f"{project.project_name} did not start a fit at "
                f"{dv:+.2f} V. Check the project runs on its own "
                f"first.")

    def _on_results(self, project_name, results, _output_config):
        if not self.busy:
            return
        project, isotope, dv = self._current
        if project_name != project.project_name:
            return
        self._collect(project, isotope, dv, results)
        self._done += 1
        self._current = None
        project._config_overrides = {}
        self._next()

    def _collect(self, project, isotope, dv, results):
        for r in results or []:
            if not r.get("success"):
                continue
            val, sig = extract_parameter(
                r.get("params_df", {}) or {}, self._parameter,
                source_name=r.get("source_name"))
            if val is None:
                continue
            self._points.append(ScanPoint(
                run=str(r.get("run_number", "") or "?"),
                isotope=isotope, dv=float(dv), value=float(val),
                sigma=float(sig) if sig == sig else float("nan")))

    def _abort(self, message):
        self._cleanup()
        self.failed.emit(message)

    def _finish(self):
        points = list(self._points)
        self._cleanup()
        self.progress.emit(self._done, self._total, "done")
        self.finished.emit(points)

    def _cleanup(self):
        self._pending = False
        if self._current is not None:
            self._current[0]._config_overrides = {}
        for project in self._connected:
            try:
                project.results_ready.disconnect(self._on_results)
            except (RuntimeError, TypeError):
                pass
        self._connected.clear()
        self._current = None
        self._jobs = []
