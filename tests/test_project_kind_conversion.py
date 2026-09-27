"""Converting a project between sample, reference and calibration.

A calibration project used to be a one-way street: created from the
menu, and the right-click menu only ever offered "reference". The runs
are the same files whichever role they play, and which role that is
changes -- a Yb set used for the cooler calibration is an ordinary
sample project the moment you want its centroids (Arda, 2026-09-25).

Run from the project root:
    .venv/Scripts/python.exe -m pytest \\
        tests/test_project_kind_conversion.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

KINDS = ("sample", "reference", "calibration")


class ConversionTests(unittest.TestCase):

    def setUp(self):
        from gui.analysis.tab import AnalysisTab
        self.tab = AnalysisTab()
        self.addCleanup(self.tab.deleteLater)

    def _project(self, name="70Ge", kind="sample"):
        self.tab._add_project(
            name, is_reference=(kind == "reference"),
            is_calibration=(kind == "calibration"))
        return self.tab._projects[-1]

    def _index(self, project):
        return self.tab._project_tabs.indexOf(project)

    # -- every direction ---------------------------------------------
    def test_every_kind_converts_to_every_other(self):
        for start in KINDS:
            for target in KINDS:
                with self.subTest(start=start, target=target):
                    p = self._project(f"{start}_{target}", start)
                    self.tab._set_project_kind(self._index(p), target)
                    self.assertEqual(self.tab.project_kind(p), target)

    def test_a_calibration_project_can_become_a_sample(self):
        """The one the menu never offered."""
        p = self._project("Yb171_cal", "calibration")
        self.tab._set_project_kind(self._index(p), "sample")
        self.assertFalse(p.is_calibration)
        self.assertFalse(p.is_reference)

    def test_the_flags_are_mutually_exclusive(self):
        p = self._project("Yb171_cal", "calibration")
        self.tab._set_project_kind(self._index(p), "reference")
        self.assertTrue(p.is_reference)
        self.assertFalse(p.is_calibration)

    def test_the_tab_title_follows(self):
        p = self._project("Yb171", "calibration")
        i = self._index(p)
        self.assertIn("Cooler Cal.", self.tab._project_tabs.tabText(i))
        self.tab._set_project_kind(i, "reference")
        self.assertIn("Reference", self.tab._project_tabs.tabText(i))
        self.tab._set_project_kind(i, "sample")
        self.assertEqual(self.tab._project_tabs.tabText(i), "Yb171")

    def test_the_blocks_are_kept(self):
        """Conversion is a label, not a rebuild."""
        p = self._project("70Ge")
        blocks = list(p._blocks)
        self.tab._set_project_kind(self._index(p), "calibration")
        self.assertEqual(p._blocks, blocks)

    def test_it_tells_the_rest_of_the_app(self):
        p = self._project("70Ge")
        seen = []
        self.tab.projects_changed.connect(lambda: seen.append(True))
        self.tab._set_project_kind(self._index(p), "reference")
        self.assertTrue(seen)

    def test_an_unknown_kind_changes_nothing(self):
        p = self._project("70Ge")
        self.tab._set_project_kind(self._index(p), "something else")
        self.assertEqual(self.tab.project_kind(p), "sample")

    def test_a_permanent_tab_is_not_a_project(self):
        self._project("70Ge")          # makes the permanent tabs appear
        idx = self.tab._project_tabs.indexOf(self.tab._is_tab)
        self.tab._set_project_kind(idx, "reference")
        self.assertFalse(getattr(self.tab._is_tab, "_is_reference", False))

    # -- the calibration tab follows ---------------------------------
    def test_converting_to_calibration_brings_the_tab_up(self):
        p = self._project("Yb171")
        self.assertFalse(self.tab._cal_tab_visible)
        self.tab._set_project_kind(self._index(p), "calibration")
        self.assertTrue(self.tab._cal_tab_visible)

    def test_converting_the_last_one_away_takes_it_down(self):
        p = self._project("Yb171", "calibration")
        self.assertTrue(self.tab._cal_tab_visible)
        self.tab._set_project_kind(self._index(p), "sample")
        self.assertFalse(self.tab._cal_tab_visible)

    def test_the_calibration_tab_lists_the_new_project(self):
        p = self._project("Yb173")
        self.tab._set_project_kind(self._index(p), "calibration")
        names = [self.tab._cal_tab._proj_table.item(r, 1).text()
                 for r in range(self.tab._cal_tab._proj_table.rowCount())]
        self.assertEqual(names, ["Yb173"])

    # -- what the menu offers ----------------------------------------
    def test_the_menu_offers_the_other_two_kinds(self):
        labels = dict((k, lbl) for k, lbl, _t in self.tab.PROJECT_KINDS)
        self.assertEqual(set(labels), set(KINDS))
        for kind in KINDS:
            with self.subTest(kind=kind):
                offered = [lbl for k, lbl, _t in self.tab.PROJECT_KINDS
                           if k != kind]
                self.assertEqual(len(offered), 2)

    def test_the_old_toggle_still_works(self):
        p = self._project("70Ge")
        self.tab._toggle_project_reference(self._index(p))
        self.assertEqual(self.tab.project_kind(p), "reference")
        self.tab._toggle_project_reference(self._index(p))
        self.assertEqual(self.tab.project_kind(p), "sample")

    def test_the_kind_survives_a_save_and_load(self):
        p = self._project("Yb171")
        self.tab._set_project_kind(self._index(p), "calibration")
        d = p.to_dict()
        from gui.analysis.project import AnalysisProject
        q = AnalysisProject("x")
        self.addCleanup(q.deleteLater)
        q.from_dict(d)
        self.assertTrue(q.is_calibration)
        self.assertFalse(q.is_reference)


if __name__ == "__main__":
    unittest.main()
