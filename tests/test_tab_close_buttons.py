"""The ✕ on a project tab closes the project.

style_project_tab_bar (2026-07-25) wraps each tab's native close
button in a container, to give it a right margin. But QTabBar decides
which tab a close click belongs to by comparing the clicked button
with the button registered for each tab -- and after wrapping, the
registered one is the container. The native button matched no tab,
tabCloseRequested never fired, and from then until 2026-09-22 no
project could be closed from its tab, in Pre-Analysis or Analysis.

These tests click the actual button, rather than calling
_close_project directly as the old tests did -- which is how two months
of a dead button went unnoticed.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_tab_close_buttons.py -q
"""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMessageBox, QTabBar, QTabWidget, QWidget,
)

_APP = QApplication.instance() or QApplication([])


def _close_button(tabs, index):
    """The ✕ the user sees on tab *index* -- inside the wrapper."""
    w = tabs.tabBar().tabButton(index, QTabBar.ButtonPosition.RightSide)
    if w is None:
        return None
    if w.objectName() == "closeWrap":
        inner = [c for c in w.children() if hasattr(c, "click")]
        return inner[0] if inner else None
    return w


def _yes():
    return mock.patch.object(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))


class WrappedButtonTests(unittest.TestCase):

    def _tabs(self, n=3):
        from gui.theme import style_project_tab_bar
        tabs = QTabWidget()
        self.addCleanup(tabs.deleteLater)
        tabs.setTabsClosable(True)
        for i in range(n):
            tabs.addTab(QWidget(), f"P{i}")
        style_project_tab_bar(tabs)
        got = []
        tabs.tabCloseRequested.connect(got.append)
        return tabs, got

    def test_the_button_is_wrapped(self):
        """The wrapper stays -- it is what gives the ✕ its margin."""
        tabs, _ = self._tabs()
        w = tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.RightSide)
        self.assertEqual(w.objectName(), "closeWrap")

    def test_clicking_it_asks_to_close_that_tab(self):
        tabs, got = self._tabs()
        _close_button(tabs, 1).click()
        self.assertEqual(got, [1])

    def test_the_index_is_looked_up_at_click_time(self):
        """Tabs close and move; a button must still name its own tab,
        not the index it was created at."""
        tabs, got = self._tabs()
        btn_of_p2 = _close_button(tabs, 2)
        tabs.removeTab(0)                   # P2 is now index 1
        btn_of_p2.click()
        self.assertEqual(got, [1])
        tabs.tabBar().moveTab(1, 0)         # P2 is now index 0
        btn_of_p2.click()
        self.assertEqual(got, [1, 0])

    def test_one_click_is_one_request(self):
        """Qt's own handler is still connected; it must stay silent
        rather than add a second request."""
        tabs, got = self._tabs()
        _close_button(tabs, 0).click()
        self.assertEqual(got, [0])

    def test_restyling_does_not_double_wire(self):
        """style_project_tab_bar runs again on every add and rename."""
        from gui.theme import style_project_tab_bar
        tabs, got = self._tabs()
        style_project_tab_bar(tabs)
        style_project_tab_bar(tabs)
        _close_button(tabs, 0).click()
        self.assertEqual(got, [0])


class PreAnalysisCloseTests(unittest.TestCase):

    def test_the_x_closes_a_pre_analysis_project(self):
        from gui.preanalysis_container import PreAnalysisContainer
        c = PreAnalysisContainer()
        self.addCleanup(c.deleteLater)
        c._add_project("73Ge_T04")
        c._add_project("76Ge_T04")
        before = c._project_tabs.count()
        with _yes():
            _close_button(c._project_tabs, before - 1).click()
        self.assertEqual(c._project_tabs.count(), before - 1)
        self.assertNotIn("76Ge_T04", [c._project_tabs.tabText(i)
                                      for i in range(c._project_tabs.count())])

    def test_answering_no_keeps_it(self):
        from gui.preanalysis_container import PreAnalysisContainer
        c = PreAnalysisContainer()
        self.addCleanup(c.deleteLater)
        c._add_project("73Ge_T04")
        before = c._project_tabs.count()
        with mock.patch.object(
                QMessageBox, "question",
                staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)):
            _close_button(c._project_tabs, before - 1).click()
        self.assertEqual(c._project_tabs.count(), before)


class AnalysisCloseTests(unittest.TestCase):

    def test_the_x_closes_an_analysis_project(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        at._add_project("70Ge_T04")
        at._add_project("72Ge_T04")
        tabs = at._project_tabs
        idx = next(i for i in range(tabs.count())
                   if tabs.tabText(i).startswith("72Ge_T04"))
        with _yes():
            _close_button(tabs, idx).click()
        names = [p.project_name for p in at._projects]
        self.assertEqual(names, ["70Ge_T04"])

    def test_the_permanent_tabs_still_have_no_x(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        at._add_project("70Ge_T04")
        tabs = at._project_tabs
        for w in at._special_tabs():
            self.assertIsNone(_close_button(tabs, tabs.indexOf(w)))


if __name__ == "__main__":
    unittest.main()
