"""Phase 8 round 4: plot options dialog, values box, styling defaults.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the round-4 contracts:

1. ``PlotTypeOptionsDialog`` — Global tab (incl. minor-tick controls)
   plus one tab per plot type (incl. Isotope Shifts); Apply writes the
   ``plot_defaults`` / ``plot_type_defaults`` settings keys with x→y
   tick mirroring and re-applies matplotlib defaults.
2. ``format_fit_value_lines`` — "Name = value ± err" lines for the
   on-plot values box (2 significant error digits, capitalized names,
   model prefix only on duplicate names).
3. OutputBlock — "Plot Options…" button, "Fit values on plot" checkbox
   + parameter tree (populated by update_tracker_params, persisted via
   get_output_config / from_dict pending state).
4. ``param_axis_label`` — capitalization + (MHz) for HFS parameters.
5. Styling defaults — minor ticks visible by default; shade-under-fit
   keys on fit + isotope-shift plot types; academic tracker defaults.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_plot_options_round4.py -q

Depends on: gui.shared_widgets, gui.analysis.blocks,
gui.analysis.helpers; PySide6.
"""

import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QCheckBox


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class PlotTypeOptionsDialogTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_dialog_builds_all_tabs_and_widgets(self):
        from gui.shared_widgets import (
            PlotTypeOptionsDialog, _PLOT_TYPE_TAB_LABELS)
        with patch("gui.shared_widgets._load_settings", return_value={}):
            dlg = PlotTypeOptionsDialog()
        self.assertIn("isotope_shift_plot", _PLOT_TYPE_TAB_LABELS)
        self.assertEqual(set(dlg._pt_widgets), set(_PLOT_TYPE_TAB_LABELS))
        # Global tab covers minor ticks, as a checkbox for visibility.
        self.assertIn("xtick.minor.visible", dlg._global_widgets)
        self.assertIsInstance(dlg._global_widgets["xtick.minor.visible"],
                              QCheckBox)
        # Per-type tabs picked up the new shade keys.
        self.assertIn("shade_under_fit", dlg._pt_widgets["fit_plot"])
        self.assertIn("boxed_labels",
                      dlg._pt_widgets["isotope_shift_plot"])

    def test_apply_writes_mirrored_settings(self):
        from gui.shared_widgets import PlotTypeOptionsDialog
        saved = {}
        with patch("gui.shared_widgets._load_settings", return_value={}):
            dlg = PlotTypeOptionsDialog()
        dlg._global_widgets["xtick.major.width"].setValue(1.5)
        dlg._global_widgets["xtick.minor.visible"].setChecked(False)
        dlg._pt_widgets["fit_plot"]["shade_alpha"].setValue(0.3)
        with patch("gui.shared_widgets._load_settings", return_value={}), \
                patch("gui.shared_widgets._save_settings",
                      side_effect=lambda s: saved.update(s)), \
                patch("gui.shared_widgets.apply_plot_settings") as ap:
            dlg.apply_settings()
        self.assertEqual(saved["plot_defaults"]["xtick.major.width"], 1.5)
        self.assertEqual(saved["plot_defaults"]["ytick.major.width"], 1.5)
        self.assertIs(saved["plot_defaults"]["xtick.minor.visible"], False)
        self.assertIs(saved["plot_defaults"]["ytick.minor.visible"], False)
        self.assertAlmostEqual(
            saved["plot_type_defaults"]["fit_plot"]["shade_alpha"], 0.3)
        ap.assert_called_once()


class FitValueLinesTests(unittest.TestCase):
    def test_lines_formatting_and_selection(self):
        from gui.shared_widgets import format_fit_value_lines
        records = [
            {"Model": "Model_1", "Parameter": "centroid",
             "Value": 75.04, "Stderr": 11.23},
            {"Model": "Model_1", "Parameter": "Al",
             "Value": -298.61, "Stderr": 10.5},
            {"Model": "Model_1", "Parameter": "bkg",
             "Value": 25.0, "Stderr": 0.0},
        ]
        lines = format_fit_value_lines(
            records, ["Model_1:centroid", "Model_1:Al"])
        self.assertEqual(len(lines), 2)
        self.assertIn("Centroid = 75 ± 11", lines[0])
        self.assertIn("Al = ", lines[1])
        # Nothing selected → nothing rendered.
        self.assertEqual(format_fit_value_lines(records, []), [])

    def test_duplicate_names_get_model_prefix(self):
        from gui.shared_widgets import format_fit_value_lines
        records = [
            {"Model": "A", "Parameter": "centroid",
             "Value": 1.0, "Stderr": 0.1},
            {"Model": "B", "Parameter": "centroid",
             "Value": 2.0, "Stderr": 0.1},
        ]
        lines = format_fit_value_lines(
            records, ["A:centroid", "B:centroid"])
        self.assertTrue(lines[0].startswith("A: Centroid"))
        self.assertTrue(lines[1].startswith("B: Centroid"))

    def test_zero_error_falls_back_to_bare_value(self):
        from gui.shared_widgets import format_fit_value_lines
        records = [{"Model": "M", "Parameter": "scale",
                    "Value": 45.0, "Stderr": 0.0}]
        (line,) = format_fit_value_lines(records, ["M:scale"])
        self.assertNotIn("±", line)


class OutputBlockValuesTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def _block(self):
        from gui.analysis.blocks import OutputBlock
        return OutputBlock("Output_1")

    def test_has_plot_options_button_and_values_ui(self):
        blk = self._block()
        self.assertTrue(hasattr(blk, "_plot_options_btn"))
        self.assertTrue(hasattr(blk, "_values_on_plot"))
        self.assertTrue(hasattr(blk, "_values_param_tree"))
        cfg = blk.get_output_config()
        self.assertIn("values_on_plot", cfg)
        self.assertIn("values_params", cfg)

    def test_values_tree_populated_and_persisted(self):
        from PySide6.QtCore import Qt
        blk = self._block()
        blk.update_tracker_params({"Model_1": ["centroid", "Al"]})
        root = blk._values_param_tree.invisibleRootItem()
        self.assertEqual(root.childCount(), 1)
        model_item = root.child(0)
        self.assertEqual(model_item.childCount(), 2)
        # Default: none checked.
        self.assertEqual(blk.get_values_params(), [])
        model_item.child(0).setCheckState(0, Qt.CheckState.Checked)
        self.assertEqual(blk.get_values_params(), ["Model_1:centroid"])
        cfg = blk.get_output_config()
        self.assertEqual(cfg["values_params"], ["Model_1:centroid"])

    def test_from_dict_pending_applies_on_populate(self):
        blk = self._block()
        blk.from_dict({"values_on_plot": True,
                       "values_params": ["Model_1:Al"]})
        self.assertTrue(blk._values_on_plot.isChecked())
        blk.update_tracker_params({"Model_1": ["centroid", "Al"]})
        self.assertEqual(blk.get_values_params(), ["Model_1:Al"])


class StyleDefaultsTests(unittest.TestCase):
    def test_minor_ticks_default_on(self):
        from gui.shared_widgets import _DEFAULT_PLOT_SETTINGS
        self.assertIs(_DEFAULT_PLOT_SETTINGS["xtick.minor.visible"], True)
        self.assertIs(_DEFAULT_PLOT_SETTINGS["ytick.minor.visible"], True)

    def test_shade_and_values_defaults(self):
        from gui.shared_widgets import _DEFAULT_PLOT_TYPE_SETTINGS
        fit = _DEFAULT_PLOT_TYPE_SETTINGS["fit_plot"]
        self.assertIn("shade_under_fit", fit)
        self.assertIn("shade_color", fit)
        self.assertIn("shade_alpha", fit)
        self.assertIn("values_box_fontsize", fit)
        iso = _DEFAULT_PLOT_TYPE_SETTINGS["isotope_shift_plot"]
        self.assertIn("boxed_labels", iso)
        self.assertIn("shade_under_fit", iso)

    def test_tracker_academic_defaults(self):
        from gui.shared_widgets import _DEFAULT_PLOT_TYPE_SETTINGS
        tr = _DEFAULT_PLOT_TYPE_SETTINGS["tracker_plot"]
        self.assertEqual(tr["marker_fmt"], "o")   # points, no line
        self.assertIn("mean_line", tr)
        self.assertIn("connect_runs", tr)

    def test_param_axis_label(self):
        from gui.analysis.helpers import param_axis_label
        self.assertEqual(param_axis_label("centroid"), "Centroid (MHz)")
        self.assertEqual(param_axis_label("Al"), "Al (MHz)")
        self.assertEqual(param_axis_label("scale"), "Scale")
        self.assertEqual(param_axis_label("bkg"), "Bkg")


if __name__ == "__main__":
    unittest.main()
