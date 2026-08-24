"""Phase 2 (UI overhaul): Save/Save As, dirty tracking, session outputs.

Date:    2026-07-24
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the session/persistence model:

1. Dirty tracking — a fresh window is clean, edits make it dirty,
   Save / Save As / Load mark it clean again (so closeEvent only prompts
   when something actually changed).
2. Save writes to the current file without a dialog once a file is
   loaded; Save As behavior (dialog) is untouched.
3. Analysis projects record their session iterations; the app save file
   carries them (``include_iterations=True``) while the per-iteration
   config snapshot stays clean; Results restores exactly those
   iterations (missing folders reported, no whole-dir scan).
4. Estimate runs persist their arrays (``estimate_results.npz`` +
   ``peaks.csv``) into the run folder and restore into the plots panel,
   peak table and log without re-running.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_phase2_session_model -v

Depends on: gui.main_window (MainWindow), gui.analysis.project
(AnalysisProject), gui.results_tab (ResultsTab); PySide6, numpy.
"""

import os
import tempfile
import types
import unittest

import numpy as np

from PySide6.QtWidgets import QApplication


_APP = None
_WIN = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _shared_window():
    """One MainWindow for the whole module — construction is heavy."""
    global _WIN
    if _WIN is None:
        from gui.main_window import MainWindow
        _WIN = MainWindow()
    return _WIN


class DirtyTrackingTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        self.win = _shared_window()
        self.win._mark_saved()

    def test_fresh_state_is_clean(self):
        self.assertFalse(self.win._is_dirty())

    def test_edit_makes_dirty_and_mark_saved_cleans(self):
        edit = self.win.estimate_tab.params_tab.global_params.element_edit
        old = edit.text()
        edit.setText(old + "X")
        try:
            self.assertTrue(self.win._is_dirty())
        finally:
            edit.setText(old)
        self.assertFalse(self.win._is_dirty())

    def test_save_writes_current_path_without_dialog(self):
        edit = self.win.estimate_tab.params_tab.global_params.element_edit
        old = edit.text()
        edit.setText("Si")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "session.yaml")
            self.win._write_save_file(path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(self.win._config_path, path)
            self.assertFalse(self.win._is_dirty())
            self.assertIn("session.yaml", self.win.windowTitle())

            # Edit again; _save() must write to the same path, no dialog.
            edit.setText("Ge")
            self.assertTrue(self.win._is_dirty())
            self.assertTrue(self.win._save())
            self.assertFalse(self.win._is_dirty())
            import yaml
            with open(path) as f:
                raw = yaml.safe_load(f)
            self.assertEqual(raw["estimate"]["element"], "Ge")
            # Leave no dangling _config_path into other tests.
            self.win._config_path = None
        edit.setText(old)
        self.win._mark_saved()

    def test_fingerprint_stable_across_calls(self):
        a = self.win._state_fingerprint()
        b = self.win._state_fingerprint()
        self.assertIsNotNone(a)
        self.assertEqual(a, b)


class SessionIterationsTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_to_dict_iterations_opt_in(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("P")
        try:
            p._session_iterations = ["iter_001", "iter_002"]
            self.assertNotIn("iterations", p.to_dict())
            d = p.to_dict(include_iterations=True)
            self.assertEqual(d["iterations"], ["iter_001", "iter_002"])
            p2 = AnalysisProject("Q")
            try:
                p2.from_dict(d)
                self.assertEqual(p2._session_iterations,
                                 ["iter_001", "iter_002"])
            finally:
                p2.deleteLater()
        finally:
            p.deleteLater()

    def test_results_restore_iterations(self):
        from gui.results_tab import ResultsTab
        import gui.shared_widgets as sw
        tab = ResultsTab()
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "analysis")
            iter_dir = os.path.join(base, "projA", "iter_001")
            os.makedirs(iter_dir)
            with open(os.path.join(iter_dir, "fit_report.txt"), "w") as f:
                f.write("report")
            orig = sw.get_analysis_dir
            sw.get_analysis_dir = lambda: base
            try:
                missing = tab.restore_iterations(
                    {"projA": ["iter_001", "iter_gone"]})
            finally:
                sw.get_analysis_dir = orig
            self.assertEqual(missing, ["projA/iter_gone"])
            self.assertIn("projA", tab._results_data)
            self.assertEqual(
                list(tab._results_data["projA"].keys()), ["iter_001"])
            self.assertEqual(
                tab._results_data["projA"]["iter_001"]["directory"],
                iter_dir)
            self.assertEqual(tab._tree.topLevelItemCount(), 1)
        tab.deleteLater()


class EstimateRunPersistenceTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        self.win = _shared_window()

    def _fake_worker(self):
        plot_results = [{
            "label": "28Si",
            "dV_array": np.linspace(-50.0, 50.0, 101),
            "intensity_array": np.exp(
                -0.5 * (np.linspace(-50.0, 50.0, 101) / 5.0) ** 2),
            "measured_peak_dVs": [0.0],
        }]
        all_peaks = [{
            "label": "28Si", "state": "gs", "iso_shift_MHz": 0.0,
            "offset_MHz": 0.0, "dV": 0.0, "V_acc": 29907.0,
            "intensity": 1.0,
        }]
        return types.SimpleNamespace(plot_results=plot_results,
                                     all_peaks=all_peaks)

    def test_persist_and_restore_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "cls_2026-07-24_00-00-00")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "cls_test.log"), "w") as f:
                f.write("estimation log line\n")

            self.win._worker = self._fake_worker()
            self.win._persist_estimate_outputs(run_dir)
            self.assertTrue(os.path.isfile(
                os.path.join(run_dir, "estimate_results.npz")))
            self.assertTrue(os.path.isfile(
                os.path.join(run_dir, "peaks.csv")))

            # Wipe the UI state, then restore from the folder alone.
            et = self.win.estimate_tab
            et.plots_tab.set_plot_data([])
            et.plots_tab.set_peak_data([])
            et.run_tab.log_text.setPlainText("")
            self.win._worker = None
            self.win._last_estimate_run = None

            self.win._restore_estimate_outputs({
                "run_dir": run_dir,
                "config_name": "cls_test",
                "file_tag": "_28Si",
                "palette": "default",
            })
            self.assertEqual(len(et.plots_tab._plot_results), 1)
            self.assertEqual(et.plots_tab._plot_results[0]["label"], "28Si")
            self.assertEqual(len(et.plots_tab._all_peaks), 1)
            self.assertIn("estimation log line",
                          et.run_tab.log_text.toPlainText())
            self.assertEqual(self.win._last_estimate_run["run_dir"], run_dir)

    def test_restore_missing_dir_is_quiet(self):
        self.win._last_estimate_run = None
        self.win._restore_estimate_outputs({
            "run_dir": r"C:\definitely\not\here",
            "config_name": "x", "file_tag": "", "palette": "default",
        })
        self.assertIsNone(self.win._last_estimate_run)

    def test_last_run_lands_in_estimate_dict(self):
        self.win._last_estimate_run = {
            "run_dir": r"C:\somewhere\cls_x", "config_name": "cls_x",
            "file_tag": "_28Si", "palette": "default",
        }
        d = self.win._build_estimate_dict()
        self.assertEqual(d["last_run"]["config_name"], "cls_x")
        self.win._last_estimate_run = None
        self.assertNotIn("last_run", self.win._build_estimate_dict())


if __name__ == "__main__":
    unittest.main()
