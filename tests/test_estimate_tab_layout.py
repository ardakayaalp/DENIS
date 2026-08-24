"""Estimate tab layout: one surface, three columns, embedded plots.

Date:    2026-06-02 (rewritten 2026-07-24 for the UI overhaul Phase 3)
Version: 2.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the single-surface Estimate tab: no sub-tabs; a horizontal
three-column splitter (Element/Transition | Isotope list | Run column);
the run column stacks Run Options, the plots panel and a shortened log
in a vertical splitter; the ⛶ toggle expands the plots over the whole
tab; isotope panels carry per-position color stripes that survive
reference toggling; and the historical ``run_tab`` / ``plots_tab``
handles keep working (main_window relies on them).

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_estimate_tab_layout -v

Depends on: gui.estimate_tab (EstimateTab, GlobalParamsWidget,
IsotopeListWidget, RunTab, PlotsTab, ISOTOPE_COLORS); PySide6.
"""

import unittest

from PySide6.QtWidgets import QApplication, QSplitter, QTabWidget


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class EstimateTabStructureTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_no_subtabs_single_surface(self):
        """The Plots sub-tab is gone: the tab hosts no QTabWidget."""
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        self.assertFalse(hasattr(tab, "tabs"))
        self.assertIsNone(tab.findChild(QTabWidget))

    def test_three_column_splitter(self):
        from gui.estimate_tab import (
            EstimateTab, GlobalParamsWidget, IsotopeListWidget, RunTab)
        tab = EstimateTab()
        sp = tab.params_tab._main_splitter
        self.assertIsInstance(sp, QSplitter)
        self.assertEqual(sp.count(), 3)
        self.assertIsInstance(sp.widget(0), GlobalParamsWidget)
        self.assertIsInstance(sp.widget(1), IsotopeListWidget)
        self.assertIsInstance(sp.widget(2), RunTab)

    def test_column_width_caps(self):
        """Columns 1-2 are capped so the plots get the freed width.

        Column 1's band is measured from the content's layout minimum
        (font/DPI dependent), so assert the invariants rather than
        pixels: wide enough that the group borders can't clip, and no
        more than a small slack above that.
        """
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        gp = tab.params_tab.global_params
        need = gp.preferred_width()
        self.assertGreaterEqual(gp.minimumWidth(), need)
        self.assertLessEqual(gp.maximumWidth(), need + 60)
        self.assertLessEqual(tab.params_tab.isotope_list.maximumWidth(),
                             560)

    def test_run_column_hosts_options_plots_log(self):
        """Column 3: options at natural height, then plots|log splitter
        (options in a splitter pane left dead space under Run)."""
        from gui.estimate_tab import EstimateTab, PlotsTab
        tab = EstimateTab()
        vs = tab.run_tab._vsplit
        self.assertEqual(vs.count(), 2)
        self.assertIsInstance(vs.widget(0), PlotsTab)
        self.assertIs(vs.widget(1), tab.run_tab._log_host)
        # Options host sits OUTSIDE the splitter, above it.
        self.assertNotIn(tab.run_tab._top_host,
                         [vs.widget(i) for i in range(vs.count())])
        self.assertTrue(
            tab.run_tab.isAncestorOf(tab.run_tab._top_host))
        self.assertTrue(hasattr(tab.run_tab, "load_requested"))
        self.assertTrue(
            tab.run_tab._log_host.isAncestorOf(tab.run_tab.log_text))

    def test_handles_still_forward(self):
        """main_window contracts: run_tab / plots_tab / run_requested."""
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        self.assertIs(tab.run_tab, tab.params_tab.run_tab)
        self.assertIs(tab.plots_tab, tab.params_tab.plots_panel)
        self.assertTrue(hasattr(tab, "run_requested"))
        for name in ("append_log", "get_options", "log_text", "run_btn"):
            self.assertTrue(hasattr(tab.run_tab, name))
        for name in ("set_plot_data", "set_peak_data", "display_results",
                     "refresh_current", "expand_btn"):
            self.assertTrue(hasattr(tab.plots_tab, name))

    def test_expand_toggle_hides_everything_but_plots(self):
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        tab.show()
        QApplication.processEvents()
        tab.plots_tab.expand_btn.setChecked(True)
        QApplication.processEvents()
        self.assertFalse(tab.params_tab.global_params.isVisible())
        self.assertFalse(tab.params_tab.isotope_list.isVisible())
        self.assertFalse(tab.run_tab._top_host.isVisible())
        self.assertFalse(tab.run_tab._log_host.isVisible())
        self.assertTrue(tab.plots_tab.isVisible())
        tab.plots_tab.expand_btn.setChecked(False)
        QApplication.processEvents()
        self.assertTrue(tab.params_tab.global_params.isVisible())
        self.assertTrue(tab.run_tab._log_host.isVisible())
        tab.hide()


class RunLoadRowTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_run_and_load_regular_buttons_in_options_row(self):
        """Run Estimation and Load Estimation are regular-sized buttons
        sharing one row INSIDE the Run Options group (the oversized
        full-width pair below the group dominated the column)."""
        from PySide6.QtWidgets import QGroupBox
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        rt = tab.run_tab
        parent = rt.run_btn.parentWidget()
        self.assertIs(parent, rt.load_btn.parentWidget())
        self.assertIsInstance(parent, QGroupBox)
        self.assertEqual(parent.title(), "Run Options")
        # Regular size: no oversized minimum height, no enlarged font.
        self.assertLess(rt.run_btn.minimumHeight(), 40)
        self.assertFalse(rt.run_btn.font().bold())
        tab.show()
        QApplication.processEvents()
        self.assertLessEqual(
            abs(rt.run_btn.geometry().y() - rt.load_btn.geometry().y()), 4)
        tab.hide()


class PdfFallbackTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_pdf_fallback_renders_onto_canvas(self):
        """A loaded run without estimate_results.npz renders the saved
        PDF pages onto the matplotlib canvas (standard toolbar
        pan/zoom/save keep working), not a text hint."""
        import os
        import tempfile
        from matplotlib.figure import Figure
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        pt = tab.plots_tab
        with tempfile.TemporaryDirectory() as d:
            fig = Figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            ax.plot([0, 1], [0, 1])
            fig.savefig(os.path.join(d, "cfg_hfs_spectra_60Co.pdf"))
            pt.display_results(d, "cfg", "_60Co", "default")
            self.assertEqual(pt._stack.currentIndex(), 0)
            self.assertGreaterEqual(len(pt.figure.axes), 1)
            self.assertTrue(pt.figure.axes[0].get_images())
            # No handle may outlive the render: the temp dir removal
            # below fails on Windows if the PDF were still open.

    def test_missing_pdf_still_shows_hint(self):
        """No arrays AND no PDF on disk: the canvas hint remains."""
        import tempfile
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        pt = tab.plots_tab
        with tempfile.TemporaryDirectory() as d:
            pt.display_results(d, "cfg", "_60Co", "default")
            self.assertEqual(pt._stack.currentIndex(), 0)
            self.assertFalse(pt.figure.axes[0].get_images())


class IsotopeColorCodingTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_panels_get_position_colors(self):
        from gui.estimate_tab import EstimateTab, ISOTOPE_COLORS
        tab = EstimateTab()
        lst = tab.params_tab.isotope_list
        panels = [lst.add_isotope() for _ in range(3)]
        for i, p in enumerate(panels):
            self.assertEqual(p._color_index, i)
            self.assertIn(ISOTOPE_COLORS[i], p.styleSheet())

    def test_reference_border_composes_with_color(self):
        from gui.estimate_tab import EstimateTab, ISOTOPE_COLORS
        tab = EstimateTab()
        lst = tab.params_tab.isotope_list
        p0 = lst.add_isotope()
        p1 = lst.add_isotope()
        p1.ref_check.setChecked(True)
        self.assertIn("#42a5f5", p1.styleSheet())      # reference border
        self.assertIn(ISOTOPE_COLORS[1], p1.styleSheet())  # color stripe
        p1.ref_check.setChecked(False)
        self.assertNotIn("#42a5f5", p1.styleSheet())
        self.assertIn(ISOTOPE_COLORS[1], p1.styleSheet())
        # Style stays objectName-scoped so nested groups are untouched.
        self.assertIn("QGroupBox#isotopePanel", p0.styleSheet())

    def test_colors_follow_removal_renumbering(self):
        from gui.estimate_tab import EstimateTab, ISOTOPE_COLORS
        tab = EstimateTab()
        lst = tab.params_tab.isotope_list
        panels = [lst.add_isotope() for _ in range(3)]
        lst._on_remove(panels[0])
        self.assertEqual(panels[1]._color_index, 0)
        self.assertIn(ISOTOPE_COLORS[0], panels[1].styleSheet())
        self.assertEqual(panels[2]._color_index, 1)


if __name__ == "__main__":
    unittest.main()
