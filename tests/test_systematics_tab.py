"""The Systematics tab: setup, step list, band, and what it exports.

The physics is tested in test_systematics.py and the chain in
test_systematic_scan.py. What matters here is that the tab shows the
band that was actually measured, that a step can be taken out of it or
fitted again with different starting values, and that a reopened
session does not mean re-running 250 fits.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_systematics_tab.py -q
"""
import math
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from cls_estimations.systematics import (  # noqa: E402
    FULL_WIDTH, HALF_WIDTH, STD_DEV,
)
from gui.analysis.systematic_scan import Baseline, value_key  # noqa: E402
from gui.analysis.systematics_tab import (  # noqa: E402
    SeedDialog, SystematicsTab, draw_systematics,
)

PROJECT = "70Ge_T02"
MODEL = "HFS_1"
SOURCES = ("Run_7947", "Run_7948")


class _StubProject:
    def __init__(self, name, reference=False, calibration=False):
        self.project_name = name
        self.is_reference = reference
        self.is_calibration = calibration

    def to_dict(self):
        return {"project_name": self.project_name, "blocks": []}


class _StubAnalysisTab:
    def __init__(self, projects=()):
        self._projects = list(projects)
        self._cal_tab = None
        self._gp_tab = None
        self._is_tab = None

    projects_changed = None


def _baseline(value=-100.0, sigma=0.4):
    params = {}
    run_of_source = {}
    for i, src in enumerate(SOURCES):
        params[src] = {(MODEL, "centroid"): {
            "value": value + i, "sigma": sigma, "vary": True,
            "min": float("-inf"), "max": float("inf")}}
        run_of_source[src] = f"794{7 + i}"
    return Baseline(params=params, run_of_source=run_of_source)


def _records(slope=-13.0, base=-100.0, offsets=(-30.0, -20.0, -10.0),
             sigma=0.5):
    """A scan in which the centroid depends linearly on the offset."""
    out = {}
    for dv in offsets:
        values = {}
        for i, src in enumerate(SOURCES):
            values[value_key(PROJECT, src, MODEL, "centroid")] = {
                "value": base + i + slope * dv, "sigma": sigma}
        out[float(dv)] = {
            "dv": float(dv), "status": "ok", "message": "",
            "values": values, "redchi": {}, "shifts": [],
            "labels": {f"{PROJECT}|{src}": f"794{7 + i}"
                       for i, src in enumerate(SOURCES)},
            "iterations": {PROJECT: f"sys_001_{dv:+g}V_CO"}}
    return out


class TabSetupTests(unittest.TestCase):

    def _tab(self, projects):
        tab = SystematicsTab(_StubAnalysisTab(projects))
        self.addCleanup(tab.deleteLater)
        return tab

    def test_it_lists_the_projects(self):
        tab = self._tab([_StubProject("74Ge", reference=True),
                         _StubProject("70Ge")])
        self.assertEqual(tab._proj_table.rowCount(), 2)
        self.assertEqual(tab._proj_table.item(0, 2).text(), "reference")
        self.assertEqual(tab._proj_table.item(1, 2).text(), "sample")

    def test_calibration_projects_are_not_scanned(self):
        """They measure the offset; they are not corrected by it."""
        tab = self._tab([_StubProject("Yb171", calibration=True),
                         _StubProject("70Ge")])
        self.assertEqual(
            [tab._proj_table.item(r, 1).text()
             for r in range(tab._proj_table.rowCount())], ["70Ge"])

    def test_the_offsets_come_from_the_range(self):
        tab = self._tab([_StubProject("70Ge")])
        tab._dv_min.setValue(-30.0)
        tab._dv_max.setValue(-10.0)
        tab._steps.setValue(5)
        self.assertEqual(tab.offsets(), [-30.0, -25.0, -20.0, -15.0, -10.0])

    def test_the_range_is_shown_before_it_is_run(self):
        tab = self._tab([_StubProject("70Ge")])
        tab._dv_min.setValue(-30.0)
        tab._dv_max.setValue(-10.0)
        tab._steps.setValue(3)
        self.assertIn("-30.00", tab._offsets_label.text())

    def test_only_ticked_projects_are_targets(self):
        tab = self._tab([_StubProject("74Ge", reference=True),
                         _StubProject("70Ge")])
        tab._proj_table.cellWidget(0, 0).setChecked(False)
        self.assertEqual([t.name for t in tab._selected_targets()],
                         ["70Ge"])

    def test_the_reference_keeps_its_role(self):
        tab = self._tab([_StubProject("74Ge", reference=True)])
        self.assertTrue(tab._selected_targets()[0].is_reference)

    def test_refreshing_does_not_pile_up_cell_widgets(self):
        """Replacing a cell widget leaves the old one parented to the
        viewport, painting at its old geometry."""
        tab = self._tab([_StubProject("70Ge"), _StubProject("72Ge")])
        for _ in range(4):
            tab.refresh_projects()
        from PySide6.QtWidgets import QCheckBox, QComboBox
        live = [w for w in tab._proj_table.viewport().children()
                if isinstance(w, (QCheckBox, QComboBox))]
        self.assertEqual(len(live), 4)      # 2 rows x (tick + combo)


class IterationListingTests(unittest.TestCase):
    """The baseline is chosen from what is on disk."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        import gui.shared_widgets as sw
        real = sw.get_analysis_dir
        sw.get_analysis_dir = lambda: self.dir
        self.addCleanup(setattr, sw, "get_analysis_dir", real)
        for name in ("iter_001", "iter_002", "sys_001_-30V_CO"):
            os.makedirs(os.path.join(self.dir, PROJECT, name))

    def _tab(self):
        tab = SystematicsTab(_StubAnalysisTab([_StubProject(PROJECT)]))
        self.addCleanup(tab.deleteLater)
        return tab

    def test_the_iterations_are_offered(self):
        combo = self._tab()._proj_table.cellWidget(0, 3)
        self.assertEqual([combo.itemText(i) for i in range(combo.count())],
                         ["iter_001", "iter_002"])

    def test_scan_folders_are_not_offered_as_a_baseline(self):
        """sys_001_-30V_CO is a step of a scan, not a fit to publish."""
        combo = self._tab()._proj_table.cellWidget(0, 3)
        self.assertNotIn(
            "sys_001_-30V_CO",
            [combo.itemText(i) for i in range(combo.count())])

    def test_the_newest_iteration_is_picked_by_default(self):
        self.assertEqual(
            self._tab()._proj_table.cellWidget(0, 3).currentText(),
            "iter_002")

    def test_a_project_with_no_output_says_so(self):
        tab = SystematicsTab(_StubAnalysisTab([_StubProject("nothing")]))
        self.addCleanup(tab.deleteLater)
        self.assertIn("no iteration",
                      tab._proj_table.cellWidget(0, 3).currentText())


class BandDisplayTests(unittest.TestCase):
    """What the tab makes of a finished scan."""

    def setUp(self):
        self.tab = SystematicsTab(_StubAnalysisTab([_StubProject(PROJECT)]))
        self.addCleanup(self.tab.deleteLater)
        self.tab._baselines[PROJECT] = _baseline()
        self.tab._records = _records()
        self.tab.compute()

    def test_the_parameter_is_offered(self):
        self.assertEqual(self.tab._plot_param.currentText(),
                         f"{MODEL} / centroid")

    def test_a_band_is_built_per_run_plus_the_mean(self):
        runs, mean = self.tab.bands()
        self.assertEqual(len(runs), 2)
        self.assertEqual(mean.n, 3)

    def test_the_band_spans_what_the_scan_found(self):
        """13 MHz per volt over 20 V is a 260 MHz band."""
        _runs, mean = self.tab.bands()
        self.assertAlmostEqual(mean.width, 260.0)
        self.assertAlmostEqual(mean.systematic(HALF_WIDTH), 130.0)

    def test_the_step_table_has_a_row_per_offset(self):
        self.assertEqual(self.tab._step_table.rowCount(), 3)
        self.assertEqual(self.tab._step_table.item(0, 1).text(), "-30")

    def test_the_summary_quotes_the_chosen_definition(self):
        self.tab._definition.setCurrentText(FULL_WIDTH)
        row = [r for r in self.tab.summary_rows()
               if r["run"] == "<weighted mean>"][0]
        self.assertAlmostEqual(row["systematic"], 260.0)
        self.tab._definition.setCurrentText(HALF_WIDTH)
        row = [r for r in self.tab.summary_rows()
               if r["run"] == "<weighted mean>"][0]
        self.assertAlmostEqual(row["systematic"], 130.0)

    def test_every_definition_is_in_the_row(self):
        row = self.tab.summary_rows()[0]
        for key in ("half_width", "band_width", "std", "max_dev",
                    "rms_dev"):
            self.assertIn(key, row)

    def test_per_run_rows_can_be_shown(self):
        self.assertEqual(len(self.tab.summary_rows()), 1)
        self.tab._per_run_cb.setChecked(True)
        self.assertEqual(len(self.tab.summary_rows()), 3)

    def test_the_headline_names_the_two_errors(self):
        self.tab._definition.setCurrentText(HALF_WIDTH)
        self.tab.compute()
        text = self.tab._headline.text()
        self.assertIn("stat", text)
        self.assertIn("cooler V", text)

    def test_the_sensitivity_is_reported(self):
        _runs, mean = self.tab.bands()
        self.assertAlmostEqual(mean.slope, -13.0, places=6)

    def test_the_grid_is_runs_by_offsets(self):
        self.assertEqual(self.tab._grid_table.columnCount(), 4)
        self.assertEqual(self.tab._grid_table.rowCount(), 3)
        self.assertEqual(self.tab._grid_table.item(2, 0).text(),
                         "weighted mean")

    def test_a_failed_step_shows_as_failed(self):
        self.tab._records[-20.0]["status"] = "failed"
        self.tab._records[-20.0]["values"] = {}
        self.tab.compute()
        self.assertEqual(self.tab._step_table.item(1, 2).text(), "failed")
        self.assertEqual(self.tab.bands()[1].n, 2)


class ExclusionTests(unittest.TestCase):
    """A step that went somewhere else must be removable from the band
    without being deleted -- it is evidence."""

    def setUp(self):
        self.tab = SystematicsTab(_StubAnalysisTab([_StubProject(PROJECT)]))
        self.addCleanup(self.tab.deleteLater)
        self.tab._baselines[PROJECT] = _baseline()
        self.tab._records = _records()
        self.tab.compute()

    def test_excluding_a_step_narrows_the_band(self):
        self.tab._toggle_excluded(0, 1)          # the -30 V row
        self.assertAlmostEqual(self.tab.bands()[1].width, 130.0)

    def test_the_step_is_still_listed(self):
        self.tab._toggle_excluded(0, 1)
        self.assertEqual(self.tab._step_table.rowCount(), 3)
        self.assertIn("excluded", self.tab._step_table.item(0, 2).text())

    def test_it_toggles_back(self):
        self.tab._toggle_excluded(0, 1)
        self.tab._toggle_excluded(0, 1)
        self.assertAlmostEqual(self.tab.bands()[1].width, 260.0)

    def test_an_excluded_step_is_marked_on_the_plot(self):
        self.tab._toggle_excluded(0, 1)
        data = self.tab.plot_data()
        self.assertEqual(data["runs"][0]["excluded_dv"], [-30.0])


class PlotTests(unittest.TestCase):

    def setUp(self):
        self.tab = SystematicsTab(_StubAnalysisTab([_StubProject(PROJECT)]))
        self.addCleanup(self.tab.deleteLater)
        self.tab._baselines[PROJECT] = _baseline()
        self.tab._records = _records()
        self.tab.compute()

    def test_the_payload_carries_every_run_and_the_mean(self):
        d = self.tab.plot_data()
        self.assertEqual(len(d["runs"]), 2)
        self.assertEqual(len(d["mean"]["dv"]), 3)

    def test_the_baseline_is_on_the_plot(self):
        d = self.tab.plot_data()
        self.assertTrue(math.isfinite(d["baseline"]))
        self.assertTrue(math.isfinite(d["baseline_sigma"]))

    def test_the_change_view_is_offered(self):
        self.tab._delta_cb.setChecked(True)
        self.assertTrue(self.tab.plot_data()["delta"])

    def test_it_draws(self):
        fig = Figure()
        draw_systematics(fig, self.tab.plot_data())
        self.assertEqual(len(fig.axes), 1)

    def test_it_draws_the_change_view(self):
        self.tab._delta_cb.setChecked(True)
        fig = Figure()
        draw_systematics(fig, self.tab.plot_data())
        self.assertIn("delta", fig.axes[0].get_ylabel())

    def test_an_empty_scan_draws_a_message_not_a_traceback(self):
        fig = Figure()
        draw_systematics(fig, {})
        self.assertEqual(len(fig.axes), 1)


class SeedDialogTests(unittest.TestCase):
    """Where a step that would not converge gets nudged."""

    def _dlg(self, overrides=None):
        dlg = SeedDialog(-30.0, PROJECT, list(SOURCES), _baseline(),
                         overrides or {})
        self.addCleanup(dlg.deleteLater)
        return dlg

    def test_it_lists_the_parameters_of_the_baseline(self):
        dlg = self._dlg()
        self.assertEqual(dlg._table.rowCount(), 1)
        self.assertEqual(dlg._table.item(0, 1).text(), "centroid")

    def test_a_typed_value_becomes_an_override(self):
        dlg = self._dlg()
        dlg._table.item(0, 3).setText("390")
        dlg._harvest()
        key = value_key(PROJECT, "", MODEL, "centroid")
        self.assertAlmostEqual(dlg.overrides[key]["value"], 390.0)

    def test_it_can_target_one_run(self):
        """Usually only one run misbehaves at the end of the range."""
        dlg = self._dlg()
        dlg._src.setCurrentText("Run_7948")
        dlg._table.item(0, 3).setText("390")
        dlg._harvest()
        self.assertIn(value_key(PROJECT, "Run_7948", MODEL, "centroid"),
                      dlg.overrides)

    def test_bounds_can_be_widened_for_one_step(self):
        dlg = self._dlg()
        dlg._table.item(0, 4).setText("-1000")
        dlg._table.item(0, 5).setText("1000")
        dlg._harvest()
        ov = dlg.overrides[value_key(PROJECT, "", MODEL, "centroid")]
        self.assertEqual((ov["min"], ov["max"]), (-1000.0, 1000.0))

    def test_an_empty_cell_means_leave_it_to_the_scan(self):
        key = value_key(PROJECT, "", MODEL, "centroid")
        dlg = self._dlg({key: {"value": 390.0}})
        self.assertEqual(dlg._table.item(0, 3).text(), "390")
        dlg._table.item(0, 3).setText("")
        dlg._harvest()
        self.assertNotIn(key, dlg.overrides)

    def test_it_shows_what_the_baseline_fitted(self):
        """So the user can see what they are overriding."""
        dlg = self._dlg()
        dlg._src.setCurrentText("Run_7947")
        self.assertEqual(dlg._table.item(0, 2).text(), "-100")

    def test_runs_that_disagree_say_so(self):
        dlg = self._dlg()
        self.assertEqual(dlg._table.item(0, 2).text(), "varies")

    def test_a_different_minimiser_can_be_chosen_for_a_step(self):
        dlg = self._dlg()
        dlg._method.setCurrentText("nelder")
        self.assertEqual(dlg.method, "nelder")

    def test_the_project_default_means_no_override(self):
        self.assertEqual(self._dlg().method, "")

    def test_nonsense_is_not_an_override(self):
        dlg = self._dlg()
        dlg._table.item(0, 3).setText("about 390")
        dlg._harvest()
        self.assertEqual(dlg.overrides, {})


class PersistenceTests(unittest.TestCase):
    """250 fits is not something to repeat because a session was
    reopened."""

    def _tab(self):
        tab = SystematicsTab(_StubAnalysisTab([_StubProject(PROJECT)]))
        self.addCleanup(tab.deleteLater)
        return tab

    def _loaded(self):
        a = self._tab()
        a._baselines[PROJECT] = _baseline()
        a._records = _records()
        a._excluded = {-30.0}
        a._seed_overrides = {-30.0: {
            value_key(PROJECT, "", MODEL, "centroid"): {"value": 390.0}}}
        a._step_methods = {-30.0: "nelder"}
        a._dv_min.setValue(-33.46)
        a._definition.setCurrentText(STD_DEV)
        b = self._tab()
        b._baselines[PROJECT] = _baseline()
        b.from_dict(a.to_dict())
        return a, b

    def test_the_measured_values_come_back(self):
        _a, b = self._loaded()
        self.assertEqual(len(b._records), 3)
        self.assertAlmostEqual(b.bands()[1].width, 130.0)   # -30 dropped

    def test_the_exclusions_come_back(self):
        _a, b = self._loaded()
        self.assertEqual(b._excluded, {-30.0})

    def test_the_hand_set_seeds_come_back(self):
        _a, b = self._loaded()
        key = value_key(PROJECT, "", MODEL, "centroid")
        self.assertAlmostEqual(b._seed_overrides[-30.0][key]["value"],
                               390.0)

    def test_a_per_step_minimiser_comes_back(self):
        _a, b = self._loaded()
        self.assertEqual(b._step_methods[-30.0], "nelder")

    def test_the_settings_come_back(self):
        _a, b = self._loaded()
        self.assertAlmostEqual(b._dv_min.value(), -33.46)
        self.assertEqual(b._definition.currentText(), STD_DEV)

    def test_an_empty_save_changes_nothing(self):
        tab = self._tab()
        tab.from_dict({})
        self.assertEqual(tab._records, {})


class ExportTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        import gui.shared_widgets as sw
        real = sw.get_analysis_dir
        sw.get_analysis_dir = lambda: self.dir
        self.addCleanup(setattr, sw, "get_analysis_dir", real)
        self.tab = SystematicsTab(_StubAnalysisTab([_StubProject(PROJECT)]))
        self.addCleanup(self.tab.deleteLater)
        self.tab._baselines[PROJECT] = _baseline()
        self.tab._records = _records()
        self.tab.compute()
        self.sent = []
        self.tab.results_ready.connect(
            lambda *a: self.sent.append(a))
        from PySide6.QtWidgets import QMessageBox
        self._real_info = QMessageBox.information
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        self.addCleanup(setattr, QMessageBox, "information",
                        self._real_info)

    def test_it_writes_a_report_and_the_tables(self):
        self.tab._send_to_results()
        idir = os.path.join(self.dir, "Systematics", "iter_001")
        for name in ("fit_report.txt", "systematics_summary.csv",
                     "scan_points.csv"):
            self.assertTrue(os.path.isfile(os.path.join(idir, name)),
                            name)
        self.assertTrue(os.path.isfile(
            os.path.join(idir, "plots", "systematics.png")))

    def test_it_reaches_the_results_tab(self):
        self.tab._send_to_results()
        self.assertEqual(self.sent[0][0], "Systematics")

    def test_the_report_says_which_systematic_this_is(self):
        """It is not the run-to-run voltage jitter the IS tab already
        reports; quoting both as one number would double count."""
        self.tab._send_to_results()
        with open(os.path.join(self.dir, "Systematics", "iter_001",
                               "fit_report.txt"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("CALIBRATION OFFSET", text)
        self.assertIn("jitter", text)

    def test_every_scan_point_is_written_out(self):
        self.tab._send_to_results()
        with open(os.path.join(self.dir, "Systematics", "iter_001",
                               "scan_points.csv"), encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        self.assertEqual(len(rows), 1 + 3 * 2)     # header + 3 dv x 2 runs

    def test_a_second_export_does_not_overwrite_the_first(self):
        self.tab._send_to_results()
        self.tab._send_to_results()
        self.assertTrue(os.path.isdir(
            os.path.join(self.dir, "Systematics", "iter_002")))


class ChainGuardTests(unittest.TestCase):
    """A ticked box that quietly does nothing is worse than an
    unticked one: a scan whose GP leg never ran reports a systematic
    on uncorrected centroids and looks exactly like one that worked."""

    def setUp(self):
        self.tab = SystematicsTab(_StubAnalysisTab([_StubProject(PROJECT)]))
        self.addCleanup(self.tab.deleteLater)
        from PySide6.QtWidgets import QMessageBox
        self.asked = []
        real = QMessageBox.question
        QMessageBox.question = staticmethod(
            lambda *a, **k: (self.asked.append(a[2]),
                             QMessageBox.StandardButton.No)[1])
        self.addCleanup(setattr, QMessageBox, "question", real)

    def test_it_says_the_gp_leg_will_not_run(self):
        self.tab._use_gp.setChecked(True)
        self.assertFalse(self.tab._confirm_chain())
        self.assertIn("No GP is fitted", self.asked[0])

    def test_it_says_the_shifts_will_not_be_recorded(self):
        self.tab._use_gp.setChecked(False)
        self.tab._use_is.setChecked(True)
        self.assertFalse(self.tab._confirm_chain())
        self.assertIn("shifts", self.asked[0])

    def test_nothing_to_warn_about(self):
        self.tab._use_gp.setChecked(False)
        self.tab._use_is.setChecked(False)
        self.assertTrue(self.tab._confirm_chain())
        self.assertEqual(self.asked, [])


class _LiveProject(QObject):
    """A project whose 'fit' puts the centroid on a known straight
    line in the assumed offset."""
    results_ready = Signal(str, list, dict)

    def __init__(self, name, slope=-13.0, base=100.0, runs=SOURCES):
        super().__init__()
        self.project_name = name
        self.is_reference = False
        self.is_calibration = False
        self._slope = slope
        self._base = base
        self._runs = list(runs)
        self._config_overrides = {}
        self._last_iter_dir = ""
        self._fit_worker = type("W", (), {"isRunning": lambda s: True})()

    def to_dict(self):
        return {"project_name": self.project_name, "blocks": []}

    def _on_fit_requested(self):
        dv = float((self._config_overrides.get("source") or {})
                   .get("cooler_offset_v", 0.0))
        results = [{
            "success": True, "run_number": src.split("_")[-1],
            "source_name": src, "fit_quality": {"redchi": 1.0},
            "params_df": {"Source": [src], "Model": [MODEL],
                          "Parameter": ["centroid"],
                          "Value": [self._base + i + self._slope * dv],
                          "Stderr": [0.5]},
        } for i, src in enumerate(self._runs)]
        QTimer.singleShot(
            0, lambda: self.results_ready.emit(self.project_name,
                                               results, {}))


class EndToEndTests(unittest.TestCase):
    """Scan a project whose answer is known exactly, and check the
    number the tab reports against the arithmetic.

    -13 MHz/V over a 20 V range is a 260 MHz band whatever the code
    does in between.
    """

    def setUp(self):
        self.p = _LiveProject("70Ge", slope=-13.0, base=100.0)
        self.addCleanup(self.p.deleteLater)
        self.tab = SystematicsTab(_StubAnalysisTab([self.p]))
        self.addCleanup(self.tab.deleteLater)
        self.tab._baselines["70Ge"] = _baseline(value=100.0 + 13.0 * 20.0)
        self.tab._plot_project.addItem("70Ge")
        self.tab._plot_project.setCurrentText("70Ge")

    def _run(self, offsets=(-30.0, -25.0, -20.0, -15.0, -10.0)):
        from gui.analysis.systematic_scan import ScanTarget
        targets = [ScanTarget(project=self.p, role="sample",
                              baseline=self.tab._baselines["70Ge"])]
        loop = QEventLoop()
        self.tab._scan.finished.connect(lambda *_a: loop.quit())
        self.tab._scan.failed.connect(lambda *_a: loop.quit())
        self.tab._scan.start(targets, list(offsets), baseline_dv=-20.0)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()

    def test_every_offset_produced_a_step(self):
        self._run()
        self.assertEqual(len(self.tab._records), 5)
        self.assertTrue(all(r["status"] == "ok"
                            for r in self.tab._records.values()))

    def test_the_band_is_the_analytic_one(self):
        self._run()
        _runs, mean = self.tab.bands()
        self.assertAlmostEqual(mean.width, 260.0, places=6)

    def test_every_definition_comes_out_right(self):
        self._run()
        _runs, mean = self.tab.bands()
        self.assertAlmostEqual(mean.systematic(HALF_WIDTH), 130.0)
        self.assertAlmostEqual(mean.systematic(FULL_WIDTH), 260.0)
        values = sorted(p.value for p in mean.used)
        import statistics
        self.assertAlmostEqual(mean.systematic(STD_DEV),
                               statistics.stdev(values), places=6)

    def test_the_sensitivity_is_recovered(self):
        self._run()
        _runs, mean = self.tab.bands()
        self.assertAlmostEqual(mean.slope, -13.0, places=6)

    def test_excluding_the_end_of_the_range_narrows_it_exactly(self):
        self._run()
        self.tab._excluded = {-30.0}
        self.tab.compute()
        _runs, mean = self.tab.bands()
        self.assertAlmostEqual(mean.width, 13.0 * 15.0, places=6)

    def test_the_steps_are_walked_outward_from_the_baseline(self):
        """So each one starts from a neighbour already fitted."""
        seen = []
        self.tab._scan.progress.connect(
            lambda d, t, what: seen.append(what))
        self._run()
        offsets = [w for w in seen if w.startswith("70Ge @")]
        self.assertEqual(offsets[0], "70Ge @ -20.00 V")


if __name__ == "__main__":
    unittest.main()
