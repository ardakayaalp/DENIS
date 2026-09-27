"""Loading a save over an open session must not destroy the workspace.

_restore_analysis empties the Analysis tab bar before rebuilding it,
and it used to decide what to spare with a hardcoded pair --
(Isotope Shifts, GP). Every other permanent tab was deleted with the
projects, and re-adding a deleted C++ object raises inside a slot,
which PySide6 turns into an abort: DENIS died on File > Load All
whenever a session was already open (2026-09-24).

The tabs are permanent, so the test is about identity: the same
objects have to survive, not equivalent ones.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_session_load_tabs.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def _analysis(projects):
    return {"projects": [{"project_name": n, "blocks": []}
                         for n in projects]}


def _settle():
    """Let deleteLater() actually delete.

    Without this the test proves nothing: the widgets the load
    scheduled for deletion are still alive when it re-adds them, and
    the crash only happens once the event loop has run -- which in the
    app it always has.
    """
    QCoreApplication.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


class SessionLoadTabTests(unittest.TestCase):

    def setUp(self):
        for name in ("information", "warning", "critical"):
            real = getattr(QMessageBox, name)
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
            self.addCleanup(setattr, QMessageBox, name, real)
        real_q = QMessageBox.question
        QMessageBox.question = staticmethod(
            lambda *a, **k: QMessageBox.StandardButton.Yes)
        self.addCleanup(setattr, QMessageBox, "question", real_q)

    @classmethod
    def setUpClass(cls):
        """One window for the whole class.

        Building a MainWindow is not cheap and destroying one is not
        safe to force: MainWindow._apply_zoom walks
        QApplication.allWidgets(), so a window torn down outside the
        event loop and another built immediately after is a
        use-after-free (heap corruption, seen 2026-09-24). Every test
        here starts by loading, and loading is exactly what clears the
        previous test's projects.
        """
        from gui.main_window import MainWindow
        cls.window = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.window.deleteLater()

    def _window(self):
        return self.window

    @staticmethod
    def _titles(at):
        return [at._project_tabs.tabText(i)
                for i in range(at._project_tabs.count())]

    def test_loading_over_an_open_session_does_not_crash(self):
        w = self._window()
        w._restore_analysis(_analysis(["74Ge", "70Ge"]))
        _settle()
        w._restore_analysis(_analysis(["74Ge", "70Ge"]))
        self.assertEqual(self._titles(w.analysis_tab)[-3:],
                         ["Isotope Shifts", "Reference Correction (GP)",
                          "Systematics"])

    def test_the_permanent_tabs_are_the_same_objects_afterwards(self):
        """Not re-created: their state and every connection live on
        them, and the Analysis tab keeps its own references."""
        w = self._window()
        at = w.analysis_tab
        before = (at._is_tab, at._gp_tab, at._cal_tab, at._sys_tab)
        w._restore_analysis(_analysis(["74Ge"]))
        _settle()
        w._restore_analysis(_analysis(["70Ge"]))
        _settle()
        self.assertEqual((at._is_tab, at._gp_tab, at._cal_tab,
                          at._sys_tab), before)

    def test_the_systematics_tab_still_works_after_a_reload(self):
        """The crash was re-adding a deleted C++ object, so touching
        the widget is the real check."""
        w = self._window()
        w._restore_analysis(_analysis(["74Ge"]))
        _settle()
        w._restore_analysis(_analysis(["74Ge", "70Ge"]))
        _settle()
        st = w.analysis_tab._sys_tab
        st.refresh_projects()
        self.assertEqual(st._proj_table.rowCount(), 2)

    def test_three_loads_in_a_row(self):
        w = self._window()
        for _ in range(3):
            w._restore_analysis(_analysis(["74Ge", "70Ge"]))
            _settle()
        self.assertEqual(len(w.analysis_tab._projects), 2)

    def test_the_cooler_tab_comes_back_for_a_calibration_session(self):
        """Its visibility flag describes the tab BAR, which the load
        empties -- left stale, the tab never returns."""
        w = self._window()
        at = w.analysis_tab
        at._add_project("Yb171", is_calibration=True)
        self.assertTrue(at._cal_tab_visible)
        _settle()
        w._restore_analysis({"projects": [
            {"project_name": "Yb171", "blocks": [],
             "is_calibration": True}]})
        self.assertIn("Cooler Calibration", self._titles(at))

    def test_the_cooler_tab_stays_away_for_an_ordinary_session(self):
        w = self._window()
        at = w.analysis_tab
        at._add_project("Yb171", is_calibration=True)
        _settle()
        w._restore_analysis(_analysis(["70Ge"]))
        _settle()
        self.assertFalse(at._cal_tab_visible)
        self.assertNotIn("Cooler Calibration", self._titles(at))

    def test_the_projects_themselves_are_replaced(self):
        w = self._window()
        w._restore_analysis(_analysis(["74Ge", "70Ge"]))
        _settle()
        w._restore_analysis(_analysis(["72Ge"]))
        _settle()
        self.assertEqual(
            [p.project_name for p in w.analysis_tab._projects], ["72Ge"])
        self.assertEqual(self._titles(w.analysis_tab)[0], "72Ge")


if __name__ == "__main__":
    unittest.main()
