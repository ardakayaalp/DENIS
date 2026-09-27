"""Adjustable window geometry survives a save/load round-trip.

Date:    2026-09-20
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Splitter / column positions are tuned for a particular screen, so they
are stored under a top-level ``ui_layout`` key of the save file (window
state, not analysis state). Pinned here:

* every tab contributes its own sub-dict and can re-apply it;
* a layout whose pane count no longer matches is IGNORED, so an older
  save can never collapse a panel to zero width;
* ``ui_layout`` is excluded from the dirty-state fingerprint -- a plain
  window resize shifts every splitter and must not raise the
  "save your changes?" prompt;
* zooming SCALES analysis block widths instead of recomputing them from
  the class default, which used to discard a width the user had dragged
  or just restored from a save file.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_ui_layout_persistence.py -q

Depends on: gui.ui_layout, gui.main_window, gui.analysis.project; PySide6.
"""

import unittest

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QSplitter, QWidget


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _splitter(n=2):
    sp = QSplitter()
    for _ in range(n):
        sp.addWidget(QWidget())
    sp.resize(600, 400)
    return sp


class SplitterHelperTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_sizes_round_trip(self):
        """On a splitter that is on screen the restore is immediate."""
        from gui.ui_layout import apply_splitter_sizes, splitter_sizes
        sp = _splitter(2)
        sp.show()
        self.addCleanup(sp.close)
        QCoreApplication.processEvents()
        sp.setSizes([200, 400])
        sizes = splitter_sizes(sp)
        self.assertEqual(sum(sizes), sum(sp.sizes()))
        sp.setSizes([500, 100])
        self.assertTrue(apply_splitter_sizes(sp, sizes))
        self.assertEqual(sp.sizes(), sizes)

    def test_a_hidden_splitter_is_restored_when_first_shown(self):
        """The Isotope Shifts plot vanished because its layout was
        restored while the tab was a hidden page: the splitter still
        had its construction-time width, the stored split was scaled
        down to it, the table's minimum won, and the plot pane got 0.
        A restore onto a hidden splitter now waits for it."""
        from gui.ui_layout import apply_splitter_sizes
        sp = _splitter(2)
        sp.resize(606, 400)
        self.assertFalse(sp.isVisible())
        self.assertTrue(apply_splitter_sizes(sp, [200, 400]))
        self.assertTrue(sp.property("ui_layout_restored"))
        sp.show()
        self.addCleanup(sp.close)
        for _ in range(3):
            QCoreApplication.processEvents()
        s = sp.sizes()
        self.assertAlmostEqual(s[0] / sum(s), 200 / 600, delta=0.02)

    def test_a_zero_width_pane_is_never_restored(self):
        """Zero is a collapsed panel, not a layout anyone chose."""
        from gui.ui_layout import apply_splitter_sizes
        sp = _splitter(2)
        sp.show()
        self.addCleanup(sp.close)
        QCoreApplication.processEvents()
        sp.setSizes([300, 300])
        self.assertFalse(apply_splitter_sizes(sp, [600, 0]))
        self.assertFalse(apply_splitter_sizes(sp, [590, 10]))
        self.assertGreater(min(sp.sizes()), 0)

    def test_mismatched_pane_count_is_ignored(self):
        """An older save must not be able to size a splitter that has
        since gained or lost a pane."""
        from gui.ui_layout import apply_splitter_sizes
        sp = _splitter(2)
        sp.setSizes([300, 300])
        before = sp.sizes()
        self.assertFalse(apply_splitter_sizes(sp, [100, 100, 100]))
        self.assertEqual(sp.sizes(), before)

    def test_degenerate_inputs_are_ignored(self):
        from gui.ui_layout import apply_splitter_sizes, splitter_sizes
        sp = _splitter(2)
        self.assertIsNone(splitter_sizes(None))
        self.assertFalse(apply_splitter_sizes(sp, None))
        self.assertFalse(apply_splitter_sizes(sp, []))
        self.assertFalse(apply_splitter_sizes(sp, [0, 0]))
        self.assertFalse(apply_splitter_sizes(None, [1, 2]))

    def test_collect_drops_empty_entries(self):
        from gui.ui_layout import collect
        sp = _splitter(2)
        sp.setSizes([200, 400])
        out = collect(present=sp, missing=None)
        self.assertIn("present", out)
        self.assertNotIn("missing", out)


class TabsExposeTheirLayoutTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_every_tab_implements_the_protocol(self):
        from gui.estimate_tab import EstimateTab
        from gui.preanalysis_container import PreAnalysisContainer
        from gui.preanalysis_tab import PreAnalysisTab
        from gui.results_tab import ResultsTab
        for cls in (EstimateTab, PreAnalysisTab, PreAnalysisContainer,
                    ResultsTab):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(hasattr(cls, "ui_layout"))
                self.assertTrue(hasattr(cls, "apply_ui_layout"))

    def test_estimate_columns_round_trip(self):
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        tab.resize(1600, 900)
        tab.show()
        QCoreApplication.processEvents()
        saved = tab.ui_layout()
        self.assertIn("columns", saved)
        tab.params_tab._main_splitter.setSizes([120, 120, 120])
        QCoreApplication.processEvents()
        tab.apply_ui_layout(saved)
        QCoreApplication.processEvents()
        self.assertEqual(tab.ui_layout()["columns"], saved["columns"])
        tab.close()

    def test_apply_tolerates_junk(self):
        from gui.results_tab import ResultsTab
        tab = ResultsTab()
        for junk in (None, {}, {"tree_viewer": "nope"},
                     {"tree_viewer": [1, 2, 3, 4, 5]}):
            with self.subTest(junk=junk):
                tab.apply_ui_layout(junk)      # must not raise


class PreAnalysisLeftColumnTests(unittest.TestCase):
    """A dragged left column has to survive a resize and a reload.

    ``_fit_main_splitter`` keeps the two panes inside the tab, but it
    used to re-derive the left one as ``min(max(sizes[0], 380), 540)``
    -- a hard ceiling -- on every resizeEvent AND on showEvent. So a
    restored session went: layout applied, user opens the
    Pre-Analysis tab, showEvent fires against the not-yet-final
    geometry, left pane clamped to 540. Once clamped the panes fit,
    so no later resize ever put it back.
    """

    def setUp(self):
        _ensure_app()

    def _tab(self, width=1900):
        from gui.preanalysis_tab import PreAnalysisTab
        t = PreAnalysisTab()
        t.resize(width, 1000)
        t.show()
        QCoreApplication.processEvents()
        self.addCleanup(t.deleteLater)
        self.addCleanup(t.close)
        return t

    def test_a_dragged_width_survives_refitting(self):
        """Whatever width Qt actually grants the drag, re-fitting must
        not take any of it back. (No pixel threshold: the ambient
        font decides what Qt grants, and the full suite runs with a
        different one than this file alone.)"""
        t = self._tab()
        target = 620
        t._main_splitter.setSizes([target, t.width() - 8 - target])
        t._on_main_splitter_moved()
        QCoreApplication.processEvents()
        dragged = t._main_splitter.sizes()[0]
        t._fit_main_splitter()
        QCoreApplication.processEvents()
        self.assertEqual(t._main_splitter.sizes()[0], dragged)

    def test_no_hard_ceiling_on_the_left_pane(self):
        """The specific regression: the pane used to be re-derived as
        min(max(sizes[0], 380), 540), so a wider column could never
        survive. Asserted on the source because the ceiling only
        showed up at widths the headless geometry may not grant."""
        import inspect
        from gui.preanalysis_tab import PreAnalysisTab
        src = inspect.getsource(PreAnalysisTab._fit_main_splitter)
        # Comments still MENTION the old ceiling (that is the point of
        # the comment), so strip them before looking for it in code.
        code = "\n".join(line.split("#")[0]
                         for line in src.splitlines())
        self.assertNotIn("540", code)
        self.assertIn("_desired_left_width", code)

    def test_the_column_yields_only_as_far_as_the_plots_need(self):
        t = self._tab()
        t._desired_left_width = 620
        t.resize(900, 1000)
        QCoreApplication.processEvents()
        t._fit_main_splitter()
        QCoreApplication.processEvents()
        left = t._main_splitter.sizes()[0]
        self.assertGreaterEqual(left, 380)
        self.assertLess(left, 620)

    def test_the_width_comes_back_when_there_is_room_again(self):
        """The half that was missing: after shrinking, widening the
        window must restore the intended column, not leave it
        clamped forever."""
        t = self._tab()
        target = 620
        t._main_splitter.setSizes([target, t.width() - 8 - target])
        t._on_main_splitter_moved()
        QCoreApplication.processEvents()
        want = t._main_splitter.sizes()[0]
        t.resize(900, 1000)
        QCoreApplication.processEvents()
        t._fit_main_splitter()
        QCoreApplication.processEvents()
        t.resize(1900, 1000)
        QCoreApplication.processEvents()
        t._fit_main_splitter()
        QCoreApplication.processEvents()
        self.assertEqual(t._main_splitter.sizes()[0], want)

    def test_a_restored_layout_states_the_width_like_a_drag(self):
        t = self._tab()
        t.apply_ui_layout({"main": [640, 1252]})
        QCoreApplication.processEvents()
        self.assertEqual(t._desired_left_width, 640)
        applied = t._main_splitter.sizes()[0]
        t._fit_main_splitter()
        QCoreApplication.processEvents()
        # Re-fitting keeps what the restore achieved rather than
        # treating it as an accident to clamp away.
        self.assertEqual(t._main_splitter.sizes()[0], applied)

    def test_every_parameter_label_fits_its_column(self):
        """"Background" rendered as "ckground": the column was a
        hardcoded 65 px whatever the font. Renaming it was not enough
        -- at a zoomed-in font even "Bkg model" needs 76 px -- so the
        column is measured from the widest label instead. Checked for
        EVERY row, at whatever font this process happens to carry."""
        from PySide6.QtGui import QFontMetrics
        from gui.preanalysis_tab import HFSModelPanel
        p = HFSModelPanel("Model_1")
        self.addCleanup(p.deleteLater)
        labels = [d["widgets"][0] for d in p._slider_params.values()]
        self.assertTrue(labels)
        for lbl in labels:
            with self.subTest(text=lbl.text()):
                fm = QFontMetrics(lbl.font())
                self.assertLessEqual(
                    fm.horizontalAdvance(lbl.text()), lbl.width(),
                    f"{lbl.text()!r} does not fit its label column")



class AnalysisPermanentTabsTests(unittest.TestCase):
    """Isotope Shifts, Reference Correction (GP) and Systematics are
    siblings of the isotope projects, not sub-tabs, and none of them
    can be closed."""

    def setUp(self):
        _ensure_app()

    def _tab(self):
        from gui.analysis.tab import AnalysisTab
        t = AnalysisTab()
        self.addCleanup(t.deleteLater)
        return t

    @staticmethod
    def _titles(t):
        return [t._project_tabs.tabText(i)
                for i in range(t._project_tabs.count())]

    def test_both_appear_once_a_project_exists(self):
        t = self._tab()
        self.assertEqual(self._titles(t), [])
        t._add_project("74Ge")
        self.assertEqual(self._titles(t)[-3:],
                         ["Isotope Shifts", "Reference Correction (GP)",
                          "Systematics"])

    def test_they_stay_to_the_right_of_new_projects(self):
        t = self._tab()
        t._add_project("74Ge")
        t._add_project("70Ge")
        t._add_project("72Ge")
        titles = self._titles(t)
        self.assertEqual(titles[:3], ["74Ge", "70Ge", "72Ge"])
        self.assertEqual(titles[-3:],
                         ["Isotope Shifts", "Reference Correction (GP)",
                          "Systematics"])

    def test_neither_is_treated_as_a_project(self):
        t = self._tab()
        proj = t._add_project("74Ge")
        self.assertTrue(t._is_special_tab(t._is_tab))
        self.assertTrue(t._is_special_tab(t._gp_tab))
        self.assertFalse(t._is_special_tab(proj))

    def test_the_gp_panel_is_still_owned_by_the_is_tab(self):
        """Save/load and every signal relay go through the IS tab, so
        the YAML key is unchanged -- only the widget's parent moved."""
        t = self._tab()
        self.assertIs(t._gp_tab, t._is_tab._ref_corr_panel)
        self.assertIn("reference_correction", t._is_tab.to_dict())

    def test_the_gp_tab_puts_controls_left_and_plots_right(self):
        t = self._tab()
        gp = t._gp_tab
        self.assertEqual(gp._main_splitter.count(), 2)
        left = gp._main_splitter.widget(0)
        right = gp._main_splitter.widget(1)
        self.assertIs(gp._corr_table.parentWidget(), left)
        self.assertIs(gp._ref_list.parentWidget(), left)
        self.assertIs(gp._canvas.parentWidget(), right)

    def test_both_disappear_with_the_last_project(self):
        """_close_project asks for confirmation, and a real modal
        blocks a headless run forever -- answer it in the stub."""
        import gui.analysis.tab as tabmod
        from PySide6.QtWidgets import QMessageBox

        t = self._tab()
        t._add_project("74Ge")
        self.assertEqual(len(self._titles(t)), 4)

        original = tabmod.QMessageBox.question
        tabmod.QMessageBox.question = staticmethod(
            lambda *a, **k: QMessageBox.StandardButton.Yes)
        self.addCleanup(setattr, tabmod.QMessageBox, "question", original)

        t._close_project(0)
        self.assertEqual(self._titles(t), [])


class MainWindowLayoutTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_layout_saved_but_excluded_from_dirty_state(self):
        """Resizing a window must not trigger the unsaved-changes
        prompt, but the geometry must still be written to the file."""
        from gui.main_window import MainWindow
        w = MainWindow()
        w.resize(1700, 950)
        w.show()
        QCoreApplication.processEvents()
        w._mark_saved()
        self.assertFalse(w._is_dirty())

        w.estimate_tab.params_tab._main_splitter.setSizes([300, 420, 900])
        w.results_tab._main_splitter.setSizes([420, 800])
        QCoreApplication.processEvents()

        self.assertIn("ui_layout", w._build_save_dict())
        self.assertFalse(
            w._is_dirty(),
            "a splitter drag must not mark the session dirty")
        w.close()

    def test_zoom_scales_block_widths_instead_of_resetting_them(self):
        """Regression: _apply_zoom recomputed every block width from the
        class default, discarding whatever the user had dragged (or had
        just restored from a save file). It must scale what is actually
        there by how far the font moved.

        Asserted on the code rather than by driving a real MainWindow:
        _apply_zoom re-fonts every widget in the application, which
        takes seconds per call once a few windows exist in the test
        process -- and it persists zoom_level to the user's settings.
        """
        import inspect
        from gui.main_window import MainWindow
        src = inspect.getsource(MainWindow._apply_zoom)
        self.assertIn("_block_zoom_scale", src)
        self.assertIn("block._current_width * ratio", src)
        self.assertNotIn("int(block.BLOCK_WIDTH * scale)", src)

    def test_zoom_ratio_preserves_relative_widths(self):
        """The arithmetic _apply_zoom uses: scaling by a ratio keeps the
        proportions the user dragged, and round-trips back."""
        custom = [599, 522, 444, 430]
        mins = [220, 220, 220, 220]
        ratio = 12 / 10
        scaled = [max(int(m * ratio), int(round(w * ratio)))
                  for w, m in zip(custom, mins)]
        self.assertEqual(scaled, [719, 626, 533, 516])
        # Source stays the widest: proportions survive.
        self.assertGreater(scaled[0], scaled[1])
        back = [max(m, int(round(w / ratio))) for w, m in zip(scaled, mins)]
        self.assertEqual(back, custom)



if __name__ == "__main__":
    unittest.main()
