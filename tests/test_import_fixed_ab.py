"""Carry fixed hyperfine A/B constants into the model block on import.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

The Pre-Analysis HFS panel models the *ratio* fix (Fix Al/Au, Fix
Bl/Bu); HFSModelPanel.hyperfine_state() expands that to a per-constant
{name: (value, fixed)} map whose names match the Analysis ModelBlock
rows exactly. ModelBlock.apply_fixed_param() then sets the value BEFORE
marking the row Fixed (Mode "Fixed" == vary off). These tests verify
both the per-row apply step and the end-to-end ratio-to-fixed expansion.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_import_fixed_ab -v

Depends on: gui.analysis.blocks (ModelBlock), gui.preanalysis_tab
(HFSModelPanel); PySide6 for QApplication.
"""

import unittest

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _hfs_model_block():
    """A ModelBlock switched to the HFS type so Al/Au/Bl/Bu rows exist."""
    from gui.analysis.blocks import ModelBlock
    mb = ModelBlock("HFS_1")
    mb._type_combo.setCurrentText("HFS")
    return mb


class ApplyFixedParamTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_fixed_value_and_mode_carried(self):
        mb = _hfs_model_block()
        rows = {r["name"]: r for r in mb._param_rows}
        self.assertIn("Au", rows)

        ok = mb.apply_fixed_param("Au", 345.6, True)
        self.assertTrue(ok)
        self.assertAlmostEqual(rows["Au"]["value"].value(), 345.6, places=3)
        # Fixed -> Mode "Fixed" -> not varying.
        self.assertEqual(rows["Au"]["mode"].currentText(), "Fixed")
        self.assertFalse(mb._get_param_vary("Au"))

    def test_unfixed_value_carried_mode_free(self):
        mb = _hfs_model_block()
        rows = {r["name"]: r for r in mb._param_rows}

        ok = mb.apply_fixed_param("Bl", 12.0, False)
        self.assertTrue(ok)
        self.assertAlmostEqual(rows["Bl"]["value"].value(), 12.0, places=3)
        self.assertEqual(rows["Bl"]["mode"].currentText(), "Free")
        self.assertTrue(mb._get_param_vary("Bl"))

    def test_missing_param_is_noop(self):
        mb = _hfs_model_block()
        self.assertFalse(mb.apply_fixed_param("NoSuchParam", 1.0, True))


class HyperfineStateTests(unittest.TestCase):
    """HFSModelPanel.hyperfine_state() maps the A/B ratio locks to a
    per-constant {name: (value, fixed)} dict for import."""

    def setUp(self):
        _ensure_app()

    def test_fix_a_ratio_marks_Al_Au_fixed(self):
        from gui.preanalysis_tab import HFSModelPanel

        panel = HFSModelPanel("M1")
        panel._slider_params["Al"]["value"].setValue(10.0)
        panel._slider_params["Au"]["value"].setValue(20.0)
        panel._slider_params["Bl"]["value"].setValue(3.0)
        panel._slider_params["Bu"]["value"].setValue(4.0)
        panel._fix_a_ratio.setChecked(True)
        panel._fix_b_ratio.setChecked(False)

        state = panel.hyperfine_state()
        self.assertEqual(state["Al"], (10.0, True))
        self.assertEqual(state["Au"], (20.0, True))
        self.assertEqual(state["Bl"], (3.0, False))
        self.assertEqual(state["Bu"], (4.0, False))

    def test_state_drives_apply_fixed_param(self):
        """End-to-end: a FIXED A from Pre-Analysis arrives in the model
        block with that value AND the Fixed mode (vary False)."""
        from gui.preanalysis_tab import HFSModelPanel

        panel = HFSModelPanel("M1")
        panel._slider_params["Au"]["value"].setValue(100.0)
        panel._fix_a_ratio.setChecked(True)

        mb = _hfs_model_block()
        rows = {r["name"]: r for r in mb._param_rows}

        for name, (value, fixed) in panel.hyperfine_state().items():
            mb.apply_fixed_param(name, value, fixed)

        self.assertAlmostEqual(rows["Au"]["value"].value(), 100.0, places=3)
        self.assertEqual(rows["Au"]["mode"].currentText(), "Fixed")
        self.assertFalse(mb._get_param_vary("Au"))


if __name__ == "__main__":
    unittest.main()
