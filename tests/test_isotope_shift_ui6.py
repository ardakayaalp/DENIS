"""Phase 6 (UI overhaul): isotope-shift labels, arrows, plot editor.

Date:    2026-07-24
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the Phase-6 contracts:

1. ``format_value_error`` — concise value(err) unit notation for the
   δν labels ("-136.5(14) MHz").
2. gid-based style round-trip — FancyArrowPatch δν arrows are captured
   by ``_extract_figure_style`` and re-applied by gid; saved entries of
   the wrong artist KIND are skipped (the old index-matching smeared δν
   texts onto annotation arrows — the doubled-arrow bug); a style's
   ``removed_gids`` deletes regenerated artists on apply.
3. Plot editor — arrows appear in the artist list; text artists get the
   dedicated font-size spin (range beyond the old 10-pt cap); Remove
   deletes gid-tagged artists and records them in ``removed_gids``.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_isotope_shift_ui6 -v

Depends on: gui.analysis.helpers, gui.results_tab, gui.shared_widgets;
PySide6, matplotlib.
"""

import unittest

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _shift_figure():
    """A minimal isotope-shift-like figure: one line, one gid'd arrow,
    one gid'd δν label, one plain label."""
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 2], [1, 3, 2], label="run")
    arrow = FancyArrowPatch((0.2, 2.5), (1.8, 2.5), arrowstyle="<->",
                            mutation_scale=12, color="#333333", lw=1.5)
    arrow.set_gid("isarrow:30Si")
    ax.add_patch(arrow)
    txt = ax.text(1.0, 2.6, "δν = -251.2(21) MHz", ha="center",
                  fontsize=8, color="#333333")
    txt.set_gid("istext:30Si")
    ax.text(0.02, 0.9, "30Si", transform=ax.transAxes)
    return fig, ax, arrow, txt


class FormatValueErrorTests(unittest.TestCase):
    def _f(self, *a, **k):
        from gui.analysis.helpers import format_value_error
        return format_value_error(*a, **k)

    def test_concise_notation(self):
        self.assertEqual(self._f(-136.5, 1.4, "MHz"), "-136.5(14) MHz")
        self.assertEqual(self._f(-251.2, 2.1, "MHz"), "-251.2(21) MHz")

    def test_no_error_falls_back_to_bare_value(self):
        self.assertEqual(self._f(10.0, None, "MHz"), "10.0 MHz")
        self.assertEqual(self._f(10.0, 0.0, "MHz"), "10.0 MHz")
        self.assertEqual(self._f(10.0, float("nan"), "MHz"), "10.0 MHz")

    def test_tiny_error_floors_at_one_unit(self):
        self.assertEqual(self._f(5.0, 0.01, "MHz"), "5.0(1) MHz")

    def test_no_unit(self):
        self.assertEqual(self._f(1.25, 0.5, digits=2), "1.25(50)")


class GidStyleRoundTripTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_arrows_captured_and_reapplied_by_gid(self):
        from gui.results_tab import (_extract_figure_style,
                                     _apply_figure_style)
        fig, ax, arrow, txt = _shift_figure()
        style = _extract_figure_style(fig)
        arrows = style["axes"][0]["arrows"]
        self.assertEqual(len(arrows), 1)
        self.assertEqual(arrows[0]["gid"], "isarrow:30Si")
        self.assertIn("posA", arrows[0])

        # Mutate the saved style, apply onto a FRESH default render.
        arrows[0]["color"] = "#ff0000"
        arrows[0]["lw"] = 3.0
        arrows[0]["posA"] = [0.1, 2.0]
        arrows[0]["posB"] = [1.9, 2.0]
        for st in style["axes"][0]["texts"]:
            if st.get("gid") == "istext:30Si":
                st["fontsize"] = 14.0
        fig2, ax2, arrow2, txt2 = _shift_figure()
        _apply_figure_style(fig2, style)
        self.assertEqual(arrow2.get_linewidth(), 3.0)
        pos = getattr(arrow2, "_posA_posB")
        self.assertAlmostEqual(pos[0][1], 2.0)
        self.assertEqual(txt2.get_fontsize(), 14.0)

    def test_kind_mismatch_entries_are_skipped(self):
        """A saved annotation entry must never style a plain text."""
        from gui.results_tab import _apply_figure_style
        fig, ax, arrow, txt = _shift_figure()
        before = txt.get_text()
        style = {"axes": [{
            "texts": [{"is_annotation": True, "text": "SMEARED",
                       "fontsize": 30.0}],
        }]}
        _apply_figure_style(fig, style)
        self.assertEqual(txt.get_text(), before)

    def test_removed_gids_delete_on_apply(self):
        from gui.results_tab import (_extract_figure_style,
                                     _apply_figure_style)
        fig, ax, arrow, txt = _shift_figure()
        style = _extract_figure_style(fig)
        style["removed_gids"] = ["isarrow:30Si", "istext:30Si"]
        _apply_figure_style(fig, style)
        self.assertNotIn(arrow, ax.patches)
        self.assertNotIn(txt, ax.texts)


class PlotEditorPhase6Tests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def _editor(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from gui.shared_widgets import PlotEditorDialog
        fig, ax, arrow, txt = _shift_figure()
        canvas = FigureCanvasQTAgg(fig)
        dlg = PlotEditorDialog(fig, canvas)
        return dlg, ax, arrow, txt

    def test_arrows_listed_and_selectable(self):
        dlg, ax, arrow, txt = self._editor()
        kinds = [k for _a, k in dlg._line_artists]
        self.assertIn("arrow", kinds)
        row = kinds.index("arrow")
        self.assertIs(dlg._line_artists[row][0], arrow)
        self.assertIn("isarrow:30Si",
                      dlg._lines_list.item(row).text())
        dlg.deleteLater()

    def test_text_gets_font_spin_beyond_old_cap(self):
        dlg, ax, arrow, txt = self._editor()
        self.assertGreaterEqual(dlg._text_size_spin.maximum(), 36)
        kinds = [k for _a, k in dlg._line_artists]
        row = next(i for i, (a, k) in enumerate(dlg._line_artists)
                   if a is txt)
        dlg._lines_list.setCurrentRow(row)
        dlg._on_line_selected(row)
        self.assertTrue(dlg._text_size_spin.isEnabled())
        dlg._text_size_spin.setValue(24.0)
        self.assertEqual(txt.get_fontsize(), 24.0)
        # Width spin no longer doubles as font size for text.
        self.assertFalse(dlg._line_width_spin.isEnabled())
        dlg.deleteLater()

    def test_remove_records_gid_and_deletes(self):
        dlg, ax, arrow, txt = self._editor()
        row = next(i for i, (a, k) in enumerate(dlg._line_artists)
                   if a is arrow)
        dlg._lines_list.setCurrentRow(row)
        dlg._on_line_selected(row)
        dlg._remove_selected_artist()
        self.assertIn("isarrow:30Si", dlg.removed_gids)
        self.assertNotIn(arrow, ax.patches)
        dlg.deleteLater()

    def test_remove_untagged_hides_instead(self):
        dlg, ax, arrow, txt = self._editor()
        line = ax.lines[0]
        row = next(i for i, (a, k) in enumerate(dlg._line_artists)
                   if a is line)
        dlg._lines_list.setCurrentRow(row)
        dlg._on_line_selected(row)
        dlg._remove_selected_artist()
        self.assertIn(line, ax.lines)          # still there...
        self.assertFalse(line.get_visible())   # ...but hidden
        self.assertFalse(getattr(dlg, "removed_gids", set()))
        dlg.deleteLater()


if __name__ == "__main__":
    unittest.main()
