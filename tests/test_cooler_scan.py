"""The cooler-offset scan drives a project's own fit, once per offset.

The scan must not become a second fitting path: whatever the project
is configured to do is what the calibration measures. So it sets
``_config_overrides["source"]["cooler_offset_v"]`` and calls the very
method the Run Fit button calls.

Two properties matter beyond "it collects numbers":

* the override is REMOVED afterwards -- a leftover cooler offset
  would silently move every later fit of that project;
* file output is suppressed -- a 15-point scan over 5 runs would
  otherwise write 75 reports and 75 plot sets.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_cooler_scan.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui.analysis.cooler_scan import (  # noqa: E402
    CoolerScan, extract_parameter,
)


class _Worker:
    def __init__(self, running=True):
        self._r = running

    def isRunning(self):
        return self._r


class _StubProject(QObject):
    """A project that 'fits' by evaluating a straight line in dV."""
    results_ready = Signal(str, list, dict)

    def __init__(self, name, runs, slope, base, *, async_=True,
                 worker=True):
        super().__init__()
        self.project_name = name
        self._runs = runs
        self._slope = slope
        self._base = base
        self._async = async_
        self._fit_worker = _Worker() if worker else None
        self._config_overrides = {}
        self.seen = []                 # (dv, output_overrides)

    def _on_fit_requested(self):
        src = self._config_overrides.get("source", {})
        dv = src.get("cooler_offset_v")
        self.seen.append((dv, dict(self._config_overrides.get("output", {}))))
        results = [{
            "success": True, "run_number": run, "source_name": f"Run_{run}",
            "params_df": {"Source": [f"Run_{run}"], "Model": ["m"],
                          "Parameter": ["Al"],
                          "Value": [self._base + self._slope * dv],
                          "Stderr": [0.5]},
        } for run in self._runs]
        emit = lambda: self.results_ready.emit(  # noqa: E731
            self.project_name, results, {})
        if self._async:
            QTimer.singleShot(0, emit)
        else:
            emit()


def _spin(scan, timeout_ms=4000):
    """Run the event loop until the scan ends."""
    loop = QEventLoop()
    out = {}
    scan.finished.connect(lambda pts: (out.setdefault("points", pts),
                                       loop.quit()))
    scan.failed.connect(lambda msg: (out.setdefault("error", msg),
                                     loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return out


class ExtractParameterTests(unittest.TestCase):

    DF = {"Source": ["Run_1", "Run_1", "Run_2"],
          "Model": ["m", "m", "m"],
          "Parameter": ["Al", "Au", "Al"],
          "Value": [12636.37, 2111.02, -3496.42],
          "Stderr": [2.86, 3.55, 0.17]}

    def test_it_finds_a_named_parameter(self):
        self.assertEqual(extract_parameter(self.DF, "Au"),
                         (2111.02, 3.55))

    def test_a_simultaneous_fit_needs_the_source_to_disambiguate(self):
        """One params_df can hold every run's parameters; without the
        source name the first Al wins, whichever run it belongs to."""
        self.assertEqual(
            extract_parameter(self.DF, "Al", source_name="Run_2")[0],
            -3496.42)

    def test_a_missing_parameter_is_not_an_error(self):
        self.assertEqual(extract_parameter(self.DF, "Bl"), (None, None))
        self.assertEqual(extract_parameter({}, "Al"), (None, None))

    def test_the_error_column_may_be_named_either_way(self):
        df = dict(self.DF)
        df["Error"] = df.pop("Stderr")
        self.assertEqual(extract_parameter(df, "Au")[1], 3.55)


class ScanTests(unittest.TestCase):

    def _scan(self, *projects_with_iso, offsets=(-50.0, 0.0, 50.0)):
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        scan.start(list(projects_with_iso), list(offsets))
        return scan, _spin(scan)

    def test_it_visits_every_project_at_every_offset(self):
        p = _StubProject("171Yb_cal", ["7547", "7507"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        _scan, out = self._scan((p, "171Yb"))
        self.assertEqual([dv for dv, _o in p.seen], [-50.0, 0.0, 50.0])
        # two runs x three offsets
        self.assertEqual(len(out["points"]), 6)

    def test_the_points_carry_the_offset_that_produced_them(self):
        p = _StubProject("171Yb_cal", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        _scan, out = self._scan((p, "171Yb"))
        by_dv = {pt.dv: pt for pt in out["points"]}
        self.assertAlmostEqual(by_dv[-50.0].value, 12636.0 + 10.5)
        self.assertAlmostEqual(by_dv[50.0].value, 12636.0 - 10.5)
        self.assertEqual(by_dv[0.0].isotope, "171Yb")
        self.assertEqual(by_dv[0.0].run, "7547")
        self.assertEqual(by_dv[0.0].sigma, 0.5)

    def test_two_isotopes_in_one_scan(self):
        a = _StubProject("171", ["7547"], -0.21, 12636.0)
        b = _StubProject("173", ["7502"], +0.059, -3496.4)
        for p in (a, b):
            self.addCleanup(p.deleteLater)
        _scan, out = self._scan((a, "171Yb"), (b, "173Yb"))
        self.assertEqual({pt.isotope for pt in out["points"]},
                         {"171Yb", "173Yb"})

    def test_the_override_is_removed_when_the_scan_ends(self):
        """A leftover offset would silently move every later fit of
        that project."""
        p = _StubProject("171Yb_cal", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        self._scan((p, "171Yb"))
        self.assertEqual(p._config_overrides, {})

    def test_file_output_is_suppressed_while_scanning(self):
        p = _StubProject("171Yb_cal", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        self._scan((p, "171Yb"))
        for _dv, out in p.seen:
            self.assertFalse(out["report"])
            self.assertFalse(out["fit_plots"])
            self.assertFalse(out["params_csv"])

    def test_a_project_that_cannot_fit_stops_the_scan_with_a_reason(self):
        """_on_fit_requested returns early and emits nothing when the
        project is not fit-ready; without noticing that, the scan
        would wait forever."""
        p = _StubProject("broken", ["1"], 0.0, 1.0, worker=False)
        self.addCleanup(p.deleteLater)
        _scan, out = self._scan((p, "171Yb"))
        self.assertNotIn("points", out)
        self.assertIn("did not start a fit", out["error"])
        self.assertEqual(p._config_overrides, {})

    def test_a_synchronous_fit_works_too(self):
        """Nothing may depend on the worker being asynchronous."""
        p = _StubProject("171", ["7547"], -0.21, 12636.0, async_=False)
        self.addCleanup(p.deleteLater)
        _scan, out = self._scan((p, "171Yb"))
        self.assertEqual(len(out["points"]), 3)

    def test_progress_is_reported(self):
        p = _StubProject("171", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        seen = []
        scan.progress.connect(lambda d, t, s: seen.append((d, t, s)))
        scan.start([(p, "171Yb")], [-10.0, 10.0])
        _spin(scan)
        self.assertEqual(seen[0][1], 2)                  # total
        self.assertIn("171", seen[0][2])
        self.assertEqual(seen[-1][0], 2)                 # all done

    def test_stop_ends_it_early(self):
        p = _StubProject("171", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        scan.progress.connect(
            lambda d, _t, _s: scan.stop() if d >= 1 else None)
        scan.start([(p, "171Yb")], [-50.0, -25.0, 0.0, 25.0, 50.0])
        out = _spin(scan)
        self.assertLess(len(p.seen), 5)
        self.assertIn("points", out)
        self.assertEqual(p._config_overrides, {})

    def test_one_offset_is_not_a_scan(self):
        p = _StubProject("171", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        with self.assertRaises(ValueError):
            scan.start([(p, "171Yb")], [0.0])

    def test_no_projects_is_refused(self):
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        with self.assertRaises(ValueError):
            scan.start([], [-10.0, 10.0])


class ScanFeedsTheCalibrationTests(unittest.TestCase):
    """End to end on stubs: a scan of two isotopes whose lines cross
    where they were built to cross."""

    def test_the_scan_result_calibrates(self):
        from cls_estimations.cooler_calibration import (
            LITERATURE_A, calibrate)
        a171, a173 = LITERATURE_A["171Yb"], LITERATURE_A["173Yb"]
        # Lines crossing literature at -33.46 V and -14.00 V.
        a = _StubProject("171", ["7547", "7507"], -0.21,
                         a171 + 0.21 * -33.46)
        b = _StubProject("173", ["7502"], 0.058,
                         a173 - 0.058 * -14.00)
        for p in (a, b):
            self.addCleanup(p.deleteLater)
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        scan.start([(a, "171Yb"), (b, "173Yb")],
                   [-50.0, -25.0, 0.0, 25.0, 50.0])
        out = _spin(scan)
        res = calibrate(out["points"])
        self.assertAlmostEqual(res.dv, -29.25, delta=0.1)
        self.assertAlmostEqual(res.scan_lo, -33.46, delta=0.05)
        self.assertAlmostEqual(res.scan_hi, -14.00, delta=0.05)


if __name__ == "__main__":
    unittest.main()
