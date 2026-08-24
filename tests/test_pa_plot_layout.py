"""Pre-Analysis plot layout switcher (Stacked vs Classic).

Date:    2026-07-26
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the layout toggle: default stacked (ToF / Spectrum / Timestamp
rows), classic = Spectrum | ToF side by side over a Timestamp strip
(the original Data Viewer arrangement); panels survive reparenting
(same widget objects, all visible), the combo stays in sync, and the
mode round-trips through the save dict.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_pa_plot_layout.py -q

Depends on: gui.preanalysis_tab; PySide6.
"""

import unittest

from PySide6.QtWidgets import QApplication, QSplitter


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class PlotLayoutTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.preanalysis_tab import PreAnalysisTab
        self.pa = PreAnalysisTab()

    def test_default_stacked(self):
        sp = self.pa._plot_splitter
        self.assertEqual(self.pa._plot_layout_mode, "stacked")
        self.assertEqual(sp.count(), 3)
        self.assertIs(sp.widget(0), self.pa._tof_widget)
        self.assertIs(sp.widget(1), self.pa._spec_widget)
        self.assertIs(sp.widget(2), self.pa._ts_widget)
        self.assertEqual(self.pa._layout_combo.currentIndex(), 0)

    def test_switcher_lives_in_subtab_corner_with_row_names(self):
        """The combo sits top-right on the sub-tab row (corner widget
        of Spectrum/Calibrations/Cooler tabs) and uses the
        '3 row stacked' / '2 row stacked' wording."""
        from PySide6.QtCore import Qt
        corner = self.pa._center_tabs.cornerWidget(
            Qt.Corner.TopRightCorner)
        self.assertIsNotNone(corner)
        self.assertTrue(corner.isAncestorOf(self.pa._layout_combo))
        items = [self.pa._layout_combo.itemText(i)
                 for i in range(self.pa._layout_combo.count())]
        self.assertEqual(items, ["3 row stacked", "2 row stacked"])

    def test_switch_to_classic_and_back(self):
        pa = self.pa
        pa._set_plot_layout("classic", replot=False)
        sp = pa._plot_splitter
        self.assertEqual(pa._plot_layout_mode, "classic")
        self.assertEqual(sp.count(), 2)
        top = sp.widget(0)
        self.assertIsInstance(top, QSplitter)
        self.assertIs(top.widget(0), pa._spec_widget)
        self.assertIs(top.widget(1), pa._tof_widget)
        self.assertIs(sp.widget(1), pa._ts_widget)
        self.assertEqual(pa._layout_combo.currentIndex(), 1)
        # The SAME canvases live on (no rebuild) and none is hidden.
        for w in (pa._tof_widget, pa._spec_widget, pa._ts_widget):
            self.assertFalse(w.isHidden())
        # Back to stacked: the intermediate splitter is dissolved.
        pa._set_plot_layout("stacked", replot=False)
        self.assertEqual(sp.count(), 3)
        self.assertIs(sp.widget(0), pa._tof_widget)
        self.assertIsNone(pa._top_hsplit)
        self.assertEqual(pa._layout_combo.currentIndex(), 0)

    def test_combo_drives_layout(self):
        self.pa._layout_combo.setCurrentIndex(1)
        self.assertEqual(self.pa._plot_layout_mode, "classic")
        self.pa._layout_combo.setCurrentIndex(0)
        self.assertEqual(self.pa._plot_layout_mode, "stacked")

    def test_session_global_across_projects(self):
        """Changing the switcher in ONE project applies to every open
        Pre-Analysis project, and new projects adopt the mode."""
        from gui.preanalysis_container import PreAnalysisContainer
        c = PreAnalysisContainer()
        p1 = c._add_project("PA_1")
        p2 = c._add_project("PA_2")
        p1._layout_combo.setCurrentIndex(1)   # user picks 2-row
        self.assertEqual(p1._plot_layout_mode, "classic")
        self.assertEqual(p2._plot_layout_mode, "classic")
        self.assertEqual(p2._layout_combo.currentIndex(), 1)
        p3 = c._add_project("PA_3")           # new project adopts it
        self.assertEqual(p3._plot_layout_mode, "classic")
        p3._layout_combo.setCurrentIndex(0)   # switch back from p3
        self.assertEqual(p1._plot_layout_mode, "stacked")
        self.assertEqual(p2._plot_layout_mode, "stacked")

    def test_layout_round_trips_through_save_dict(self):
        pa = self.pa
        pa._set_plot_layout("classic", replot=False)
        d = pa._build_config_dict()
        self.assertEqual(
            d["preanalysis"]["plot_options"]["plot_layout"], "classic")
        from gui.preanalysis_tab import PreAnalysisTab
        pa2 = PreAnalysisTab()
        pa2._restore_from_dict(d["preanalysis"])
        self.assertEqual(pa2._plot_layout_mode, "classic")
        self.assertEqual(pa2._plot_splitter.count(), 2)
        # Legacy saves without the key default to stacked.
        del d["preanalysis"]["plot_options"]["plot_layout"]
        pa3 = PreAnalysisTab()
        pa3._restore_from_dict(d["preanalysis"])
        self.assertEqual(pa3._plot_layout_mode, "stacked")


if __name__ == "__main__":
    unittest.main()
