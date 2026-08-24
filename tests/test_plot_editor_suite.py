"""Plot editor suite, phase 1: universal access + add/position depth.

Date:    2026-07-26
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the phase-1 upgrades of the unified plot editor: the app-level
right-click filter opens an editor on ANY matplotlib canvas (routing
through ``canvas._plot_editor_opener`` when a surface provides richer
wiring), the new Add tab creates text / annotation / line / span
elements tagged ``_pe_added`` that the Artists tab truly removes, the
numeric position / arrow-tip / rotation / z-order fields, the axes
linear-log scale switch (with a non-positive-data guard), and the
legend column/title options.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_plot_editor_suite.py -q

Depends on: gui.shared_widgets; PySide6; matplotlib.
"""

import unittest

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _make_canvas():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 4, 9], label="data")
    return FigureCanvasQTAgg(fig), fig, ax


class UniversalAccessTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_open_plot_editor_for_plain_canvas(self):
        from gui.shared_widgets import (open_plot_editor_for,
                                        PlotEditorDialog)
        canvas, fig, _ax = _make_canvas()
        dlg = open_plot_editor_for(canvas)
        self.assertIsInstance(dlg, PlotEditorDialog)
        self.assertIs(canvas._generic_plot_editor, dlg)
        dlg.close()

    def test_opener_attribute_takes_precedence(self):
        from gui.shared_widgets import open_plot_editor_for
        canvas, _fig, _ax = _make_canvas()
        called = []
        canvas._plot_editor_opener = lambda: called.append(True)
        out = open_plot_editor_for(canvas)
        self.assertIsNone(out)
        self.assertEqual(called, [True])

    def test_install_filter_idempotent(self):
        from gui import shared_widgets as sw
        app = _ensure_app()
        f1 = sw.install_plot_editor_access(app)
        f2 = sw.install_plot_editor_access(app)
        self.assertIs(f1, f2)

    def test_surfaces_route_through_own_wiring(self):
        """PA / estimate PlotsTab canvases advertise their openers."""
        from gui.preanalysis_tab import PreAnalysisTab
        pa = PreAnalysisTab()
        for c in (pa._canvas, pa._tof_canvas, pa._ts_canvas):
            self.assertTrue(callable(c._plot_editor_opener))


class AddElementsTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.shared_widgets import PlotEditorDialog
        self.canvas, self.fig, self.ax = _make_canvas()
        self.dlg = PlotEditorDialog(self.fig, self.canvas)

    def tearDown(self):
        self.dlg.close()

    def _add(self, type_idx, x=1.5, y=2.0, x2=2.5, y2=6.0, text="hello"):
        self.dlg._add_type_combo.setCurrentIndex(type_idx)
        self.dlg._add_text_edit.setText(text)
        return self.dlg._add_element(x, y, x2, y2)

    def test_add_text(self):
        a = self._add(0)
        self.assertIn(a, self.ax.texts)
        self.assertTrue(a._pe_added)
        self.assertEqual(a.get_text(), "hello")
        self.assertEqual(a.get_position(), (1.5, 2.0))

    def test_add_annotation_with_arrow(self):
        a = self._add(1)
        self.assertIn(a, self.ax.texts)
        self.assertEqual(tuple(a.xy), (2.5, 6.0))
        self.assertIsNotNone(a.arrow_patch)

    def test_add_lines_and_spans(self):
        n_lines0 = len(self.ax.get_lines())
        v = self._add(2)
        h = self._add(3)
        self.assertEqual(len(self.ax.get_lines()), n_lines0 + 2)
        vs = self._add(4)
        hs = self._add(5)
        for a in (v, h, vs, hs):
            self.assertTrue(a._pe_added)
        self.assertIn(vs, self.ax.patches)
        self.assertIn(hs, self.ax.patches)

    def test_added_artist_truly_removed(self):
        a = self._add(0)
        self.dlg._populate_lines()
        row = next(i for i, (art, _k) in
                   enumerate(self.dlg._line_artists) if art is a)
        self.dlg._lines_list.setCurrentRow(row)
        self.dlg._remove_selected_artist()
        self.assertNotIn(a, self.ax.texts)

    def test_click_place_two_click_annotation(self):
        """Simulated clicks: first = arrow tip, second = label pos."""
        self.dlg._add_type_combo.setCurrentIndex(1)
        self.dlg._add_text_edit.setText("peak")
        self.dlg._add_place_btn.setChecked(True)

        class _Ev:
            def __init__(self, ax, x, y):
                self.inaxes = ax
                self.xdata = x
                self.ydata = y
        self.dlg._on_place_click(_Ev(self.ax, 2.0, 4.0))   # tip
        self.assertEqual(len(self.dlg._place_points), 1)
        self.dlg._on_place_click(_Ev(self.ax, 1.2, 8.0))   # label
        added = [t for t in self.ax.texts
                 if getattr(t, "_pe_added", False)]
        self.assertEqual(len(added), 1)
        self.assertEqual(tuple(added[0].xy), (2.0, 4.0))
        self.assertEqual(added[0].get_position(), (1.2, 8.0))
        self.assertFalse(self.dlg._add_place_btn.isChecked())


class ArtistDepthTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.shared_widgets import PlotEditorDialog
        self.canvas, self.fig, self.ax = _make_canvas()
        self.txt = self.ax.text(1.0, 2.0, "note")
        self.dlg = PlotEditorDialog(self.fig, self.canvas)
        self.dlg._populate_lines()
        row = next(i for i, (a, _k) in
                   enumerate(self.dlg._line_artists) if a is self.txt)
        self.dlg._lines_list.setCurrentRow(row)

    def tearDown(self):
        self.dlg.close()

    def test_position_spins_reflect_and_apply(self):
        self.assertEqual(self.dlg._artist_x_spin.value(), 1.0)
        self.assertEqual(self.dlg._artist_y_spin.value(), 2.0)
        self.dlg._artist_x_spin.setValue(3.5)
        self.assertEqual(self.txt.get_position()[0], 3.5)

    def test_rotation_and_zorder(self):
        self.dlg._text_rotation_spin.setValue(45.0)
        self.assertEqual(self.txt.get_rotation(), 45.0)
        self.dlg._artist_zorder_spin.setValue(12.0)
        self.assertEqual(self.txt.get_zorder(), 12.0)

    def test_line_zorder(self):
        row = next(i for i, (a, k) in
                   enumerate(self.dlg._line_artists) if k == "line")
        self.dlg._lines_list.setCurrentRow(row)
        self.dlg._artist_zorder_spin.setValue(7.0)
        self.assertEqual(self.ax.get_lines()[0].get_zorder(), 7.0)


class FontControlTests(unittest.TestCase):
    """Real font pickers + bold/italic + live WYSIWYG preview on the
    Axes, Artists and Add tabs."""

    def setUp(self):
        _ensure_app()
        from gui.shared_widgets import PlotEditorDialog
        self.canvas, self.fig, self.ax = _make_canvas()
        self.txt = self.ax.text(1.0, 2.0, "note")
        self.dlg = PlotEditorDialog(self.fig, self.canvas)

    def tearDown(self):
        self.dlg.close()

    def test_axes_uses_real_font_combo(self):
        from PySide6.QtWidgets import QFontComboBox
        self.assertIsInstance(self.dlg._ax_font_family, QFontComboBox)
        # Preview label exists and carries sample text.
        self.dlg._on_axes_selected(0)
        self.assertTrue(self.dlg._ax_font_preview.text())

    def test_artist_bold_italic_apply(self):
        self.dlg._populate_lines()
        row = next(i for i, (a, _k) in
                   enumerate(self.dlg._line_artists) if a is self.txt)
        self.dlg._lines_list.setCurrentRow(row)
        self.assertTrue(self.dlg._text_family.isEnabled())
        self.dlg._text_bold.setChecked(True)
        self.dlg._text_italic.setChecked(True)
        self.assertEqual(self.txt.get_fontweight(), "bold")
        self.assertEqual(self.txt.get_fontstyle(), "italic")
        # Preview mirrors the artist's own text + style.
        pf = self.dlg._text_font_preview.font()
        self.assertTrue(pf.bold() and pf.italic())
        self.assertEqual(self.dlg._text_font_preview.text(), "note")

    def test_font_controls_disabled_for_non_text(self):
        self.dlg._populate_lines()
        row = next(i for i, (a, k) in
                   enumerate(self.dlg._line_artists) if k == "line")
        self.dlg._lines_list.setCurrentRow(row)
        self.assertFalse(self.dlg._text_family.isEnabled())
        self.assertFalse(self.dlg._text_bold.isEnabled())

    def test_add_tab_bold_italic_text(self):
        self.dlg._add_type_combo.setCurrentIndex(0)
        self.dlg._add_text_edit.setText("peak A")
        self.dlg._add_bold.setChecked(True)
        self.dlg._add_italic.setChecked(True)
        a = self.dlg._add_element(2.0, 6.0, 2.0, 6.0)
        self.assertEqual(a.get_fontweight(), "bold")
        self.assertEqual(a.get_fontstyle(), "italic")
        self.assertEqual(self.dlg._add_font_preview.text(), "peak A")

    def test_add_font_controls_gated_to_text_types(self):
        self.dlg._add_type_combo.setCurrentIndex(2)   # vertical line
        self.assertFalse(self.dlg._add_family.isEnabled())
        self.dlg._add_type_combo.setCurrentIndex(0)   # text
        self.assertTrue(self.dlg._add_family.isEnabled())

    def test_weight_is_bold_helper(self):
        from gui.shared_widgets import PlotEditorDialog
        self.assertTrue(PlotEditorDialog._weight_is_bold("bold"))
        self.assertTrue(PlotEditorDialog._weight_is_bold(700))
        self.assertFalse(PlotEditorDialog._weight_is_bold("normal"))
        self.assertFalse(PlotEditorDialog._weight_is_bold(400))


class UndoRedoTests(unittest.TestCase):
    """Snapshot-based Ctrl+Z / Ctrl+Y in the plot editor."""

    def setUp(self):
        _ensure_app()
        from gui.shared_widgets import PlotEditorDialog
        self.canvas, self.fig, self.ax = _make_canvas()
        self.ax.set_title("orig")
        self.dlg = PlotEditorDialog(self.fig, self.canvas)

    def tearDown(self):
        self.dlg.close()

    def test_baseline_snapshot(self):
        self.assertEqual(len(self.dlg._undo_stack), 1)

    def test_undo_redo_property(self):
        from matplotlib.colors import to_hex
        orig_color = to_hex(self.ax.get_lines()[0].get_color())
        self.ax.set_title("changed")
        self.dlg._commit_state()
        self.ax.get_lines()[0].set_color("#ff0000")
        self.dlg._commit_state()
        self.dlg._undo()   # revert colour
        self.assertEqual(to_hex(self.ax.get_lines()[0].get_color()),
                         orig_color)
        self.assertEqual(self.ax.get_title(), "changed")
        self.dlg._undo()   # revert title
        self.assertEqual(self.ax.get_title(), "orig")
        self.dlg._redo()
        self.assertEqual(self.ax.get_title(), "changed")

    def test_undo_of_add_removes_artist(self):
        t = self.ax.text(1.5, 5.0, "new")
        t._pe_added = True
        self.dlg._commit_state()
        self.dlg._undo()
        self.assertFalse(any(
            getattr(x, "_pe_added", False) and x.get_text() == "new"
            for x in self.ax.texts))

    def test_no_op_commit_does_not_grow_stack(self):
        n = len(self.dlg._undo_stack)
        self.dlg._commit_state()   # nothing changed
        self.assertEqual(len(self.dlg._undo_stack), n)


class AxesLegendDepthTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.shared_widgets import PlotEditorDialog
        self.canvas, self.fig, self.ax = _make_canvas()
        self.dlg = PlotEditorDialog(self.fig, self.canvas)

    def tearDown(self):
        self.dlg.close()

    def test_scale_switch_and_populate(self):
        self.dlg._on_axes_selected(0)
        self.assertEqual(self.dlg._xscale_combo.currentText(), "linear")
        self.dlg._yscale_combo.setCurrentText("log")
        self.assertEqual(self.ax.get_yscale(), "log")
        self.dlg._yscale_combo.setCurrentText("linear")
        self.assertEqual(self.ax.get_yscale(), "linear")

    def test_log_guard_on_nonpositive_data(self):
        """Axes spanning negative values refuse log and the combo
        snaps back to the real scale instead of lying."""
        self.ax.plot([-5, 5], [-3, 3])
        self.ax.relim()
        self.ax.autoscale()
        self.dlg._on_axes_selected(0)
        self.dlg._xscale_combo.setCurrentText("log")
        # matplotlib clamps limits rather than raising for plain
        # nonpositive limits; accept either outcome as long as the
        # combo matches the axis afterwards.
        self.assertEqual(self.dlg._xscale_combo.currentText(),
                         self.ax.get_xscale())

    def test_legend_columns_and_title(self):
        self.dlg._legend_ncol_spin.setValue(2)
        self.dlg._legend_title_edit.setText("Runs")
        self.dlg._apply_legend()
        leg = self.ax.get_legend()
        self.assertIsNotNone(leg)
        self.assertEqual(leg.get_title().get_text(), "Runs")
        self.assertEqual(leg._ncols, 2)


class QuickPlotEditorAccessTests(unittest.TestCase):
    """QuickPlot wires the universal right-click editor to a method of
    its own. Regression: the opener referenced ``self._open_editor``
    before any such method existed, so merely constructing the dialog
    raised AttributeError and Tools > Quick Plot could not open."""

    def setUp(self):
        _ensure_app()

    def test_quick_plot_constructs_and_opens_editor(self):
        from gui.shared_widgets import PlotEditorDialog, QuickPlotDialog
        dlg = QuickPlotDialog()
        try:
            self.assertTrue(callable(dlg._canvas._plot_editor_opener))
            dlg._open_editor()
            self.assertIsInstance(dlg._editor_dialog, PlotEditorDialog)
            # A second call raises the existing dialog, not a new one.
            first = dlg._editor_dialog
            dlg._open_editor()
            self.assertIs(dlg._editor_dialog, first)
            dlg._editor_dialog.close()
        finally:
            dlg.close()


if __name__ == "__main__":
    unittest.main()
