"""Results tab: copy/paste a plot's saved style between plots.

Date:    2026-07-26
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the right-click copy/paste of a plot's .style.json: copy strips
per-plot deletions (removed_gids) and remembers the source; paste
writes the copied look onto the target while preserving the target's
OWN deletions; only .npz-backed editable plots are styleable.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_results_style_clipboard.py -q

Depends on: gui.results_tab; PySide6.
"""

import json
import os
import tempfile
import unittest

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class StyleClipboardTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.results_tab import ResultsTab
        import gui.results_tab as R
        # Silence the modal confirmation in paste.
        self._orig_info = R.QMessageBox.information
        R.QMessageBox.information = staticmethod(lambda *a, **k: None)
        # Auto-accept the property-selection dialog with all categories
        # ticked (its default), so copy/paste run head-less.
        self._orig_exec = R.StylePropertyDialog.exec
        R.StylePropertyDialog.exec = lambda _self: 1
        self.R = R
        self.rt = ResultsTab()
        self.d = tempfile.mkdtemp()
        self.src = os.path.join(self.d, "fit_run_2.npz")
        self.tgt = os.path.join(self.d, "fit_run_5.npz")
        open(self.src, "w").close()
        open(self.tgt, "w").close()

    def tearDown(self):
        self.R.QMessageBox.information = self._orig_info
        self.R.StylePropertyDialog.exec = self._orig_exec

    def _style_path(self, npz):
        from gui.results_tab import _style_path
        return _style_path(npz)

    def test_styleable_detection(self):
        self.assertTrue(self.rt._is_styleable_plot(
            {"type": "editable_plot", "path": self.src}))
        self.assertFalse(self.rt._is_styleable_plot(
            {"type": "image", "path": "x.png"}))
        self.assertFalse(self.rt._is_styleable_plot(
            {"type": "editable_plot", "path": "x.png"}))

    def test_copy_strips_deletions_and_records_source(self):
        style = {"figsize": [6, 4],
                 "axes": [{"title": "custom", "title_size": 18.0}],
                 "removed_gids": ["arrow_1"]}
        with open(self._style_path(self.src), "w") as f:
            json.dump(style, f)
        self.rt._copy_plot_style(self.src)
        self.assertIsNotNone(self.rt._style_clipboard)
        self.assertEqual(self.rt._style_clipboard["source"],
                         "fit_run_2.npz")
        self.assertNotIn("removed_gids",
                         self.rt._style_clipboard["style"])

    def test_copy_without_saved_style_is_noop(self):
        self.rt._copy_plot_style(self.src)   # no .style.json exists
        self.assertIsNone(self.rt._style_clipboard)

    def test_paste_writes_style_preserving_target_deletions(self):
        with open(self._style_path(self.src), "w") as f:
            json.dump({"axes": [{"title": "custom", "title_size": 18.0}]},
                      f)
        with open(self._style_path(self.tgt), "w") as f:
            json.dump({"removed_gids": ["arrow_9"]}, f)
        self.rt._copy_plot_style(self.src)
        self.rt._paste_plot_style([self.tgt])
        with open(self._style_path(self.tgt)) as f:
            merged = json.load(f)
        self.assertEqual(merged["axes"][0]["title"], "custom")
        self.assertEqual(merged["axes"][0]["title_size"], 18.0)
        # Target keeps its own deletion, not the source's (none here).
        self.assertEqual(merged["removed_gids"], ["arrow_9"])

    def test_paste_without_clipboard_is_noop(self):
        self.rt._paste_plot_style([self.tgt])
        self.assertFalse(os.path.isfile(self._style_path(self.tgt)))

    def test_filter_keeps_only_selected_categories(self):
        from gui.results_tab import _filter_style
        full = {"figsize": [6, 4],
                "axes": [{"title": "T", "title_size": 18.0,
                          "xlim": [0, 10], "grid": True,
                          "artists": [{"kind": "line"}]}]}
        f = _filter_style(full, ["font_sizes", "limits"])
        self.assertNotIn("figsize", f)
        ax = f["axes"][0]
        self.assertEqual(set(ax), {"title_size", "xlim", "ylim"} & set(ax))
        self.assertIn("title_size", ax)
        self.assertIn("xlim", ax)
        self.assertNotIn("title", ax)
        self.assertNotIn("artists", ax)

    def test_categories_present_reflects_content(self):
        from gui.results_tab import _style_categories_present
        self.assertEqual(
            _style_categories_present({"axes": [{"grid": True}]}),
            ["grid"])
        self.assertIn(
            "figure",
            _style_categories_present({"figsize": [6, 4], "axes": []}))

    def test_partial_paste_does_not_reset_other_props(self):
        """A style carrying only font sizes must not blank the title."""
        from gui.results_tab import _apply_figure_style
        from matplotlib.figure import Figure
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title("Keep me", fontsize=10)
        _apply_figure_style(fig, {"axes": [{"title_size": 20.0}]})
        self.assertEqual(ax.get_title(), "Keep me")
        self.assertEqual(ax.title.get_fontsize(), 20.0)


if __name__ == "__main__":
    unittest.main()
