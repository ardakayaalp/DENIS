"""A model's background can be a polynomial, not just a constant.

Date:    2026-09-20
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Every model that owns a background owned exactly one coefficient,
``Bkg_p0`` -- a flat line. A baseline that slopes across the scan
(laser power or transmission drifting) had to be modelled by adding a
second Model block of type Polynomial and typing its coefficients by
hand, with nothing in Pre-Analysis to tune them against the data.

satlas2 already builds the background as its own summed Polynomial,
so higher orders cost nothing in the model. Pinned here:

* ``Bkg_p0`` alone still means a flat background -- old saves, which
  carry no ``bkg_order``, reload byte-identical;
* the coefficient list is REVERSED on the way into satlas2, whose
  ``Polynomial`` is highest-power-first (``np.polyval``). Getting
  this backwards swaps the offset for the slope, silently;
* ``Bkg_p1`` composes the full lmfit name ``..._bkg___p1``, so
  expressions and Fitter constraints can reach it;
* changing the background shape keeps the parameters the user
  already typed;
* the shape and its coefficients survive Pre-Analysis -> Analysis
  with full precision (a slope is ~0.02 counts/MHz, which the
  panel's default 2 decimals destroyed).

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_polynomial_background.py -q

Depends on: gui.analysis.naming, gui.analysis.fitting,
gui.analysis.blocks, gui.preanalysis_tab; satlas2, PySide6, NumPy.
"""

import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

import satlas2  # noqa: E402

from gui.analysis.fitting import _add_background_model  # noqa: E402
from gui.analysis.naming import (  # noqa: E402
    background_coefficients, bkg_param_key, full_param_name,
)


class _Source:
    """The only thing _add_background_model asks of a satlas2 Source."""

    def __init__(self):
        self.models = []

    def addModel(self, m):
        self.models.append(m)


def _bkg(params):
    return _add_background_model(satlas2, _Source(), params, "m_bkg")


class NamingTests(unittest.TestCase):
    def test_key_translation_keeps_the_p(self):
        """Regression: slicing the whole ``Bkg_p`` prefix yields a
        bare "1", which matches no satlas2 parameter -- vary/min/max
        were dropped and the full name came out as ``..._bkg___1``."""
        self.assertEqual(bkg_param_key("Bkg_p0"), "p0")
        self.assertEqual(bkg_param_key("Bkg_p1"), "p1")
        self.assertEqual(bkg_param_key("Bkg_p12"), "p12")

    def test_full_name_reaches_the_summed_polynomial(self):
        self.assertEqual(
            full_param_name("Run_7904", "Model.1", "Bkg_p1"),
            "Run_7904___Model_1_bkg___p1")

    def test_legacy_constant_name_is_unchanged(self):
        self.assertEqual(
            full_param_name("Run_7904", "Model.1", "Bkg_p0"),
            "Run_7904___Model_1_bkg___p0")

    def test_coefficients_come_back_lowest_first(self):
        found = background_coefficients(
            {"Bkg_p2": 1, "centroid": 0, "Bkg_p0": 1, "Bkg_p1": 1})
        self.assertEqual(found, ["Bkg_p0", "Bkg_p1", "Bkg_p2"])

    def test_non_coefficients_are_not_swept_up(self):
        self.assertEqual(background_coefficients({"Bkg_px": 1}), [])
        self.assertEqual(background_coefficients({"Bkgp0": 1}), [])
        self.assertEqual(background_coefficients({}), [])


class BackgroundModelTests(unittest.TestCase):
    X = np.array([-100.0, 0.0, 100.0])

    def test_constant_is_unchanged(self):
        b = _bkg({"Bkg_p0": {"value": 5.0}})
        np.testing.assert_allclose(b.f(self.X), [5.0, 5.0, 5.0])

    def test_linear_is_p0_plus_p1_x(self):
        """The reversal check. Fed straight through, satlas2 would
        read 5.0 as the SLOPE and 0.02 as the offset."""
        b = _bkg({"Bkg_p0": {"value": 5.0}, "Bkg_p1": {"value": 0.02}})
        np.testing.assert_allclose(b.f(self.X), [3.0, 5.0, 7.0])

    def test_quadratic(self):
        b = _bkg({"Bkg_p0": {"value": 5.0}, "Bkg_p1": {"value": 0.0},
                  "Bkg_p2": {"value": 1e-3}})
        np.testing.assert_allclose(b.f(self.X), [15.0, 5.0, 15.0])

    def test_vary_and_bounds_reach_every_coefficient(self):
        b = _bkg({
            "Bkg_p0": {"value": 5.0, "vary": True},
            "Bkg_p1": {"value": 0.02, "vary": False,
                       "min": -1.0, "max": 1.0}})
        self.assertTrue(b.params["p0"].vary)
        self.assertFalse(b.params["p1"].vary)
        self.assertEqual(b.params["p1"].min, -1.0)
        self.assertEqual(b.params["p1"].max, 1.0)

    def test_a_model_with_no_background_row_still_gets_a_flat_one(self):
        """Legacy configs whose params dict predates the Bkg rows."""
        b = _bkg({"centroid": {"value": 0.0}})
        np.testing.assert_allclose(b.f(self.X), [0.0, 0.0, 0.0])


class ModelBlockTests(unittest.TestCase):
    def _block(self):
        from gui.analysis.blocks import ModelBlock
        b = ModelBlock("Model_1")
        self.addCleanup(b.deleteLater)
        return b

    @staticmethod
    def _names(block):
        return [r["name"] for r in block._param_rows]

    def test_default_is_a_single_constant(self):
        b = self._block()
        self.assertEqual(b._bkg_order(), 0)
        self.assertEqual(self._names(b)[-1], "Bkg_p0")
        self.assertNotIn("Bkg_p1", self._names(b))

    def test_linear_adds_one_row(self):
        b = self._block()
        b._bkg_combo.setCurrentIndex(1)
        self.assertEqual(self._names(b)[-2:], ["Bkg_p0", "Bkg_p1"])

    def test_changing_shape_keeps_what_the_user_typed(self):
        """Unlike a model-type change, the parameter set only grows or
        shrinks at the tail -- nothing else should be reset."""
        b = self._block()
        row = {r["name"]: r for r in b._param_rows}["centroid"]
        row["value"].setValue(-38.44)
        b._bkg_combo.setCurrentIndex(2)
        again = {r["name"]: r for r in b._param_rows}["centroid"]
        self.assertAlmostEqual(again["value"].value(), -38.44)

    def test_shape_and_values_round_trip(self):
        b = self._block()
        b._bkg_combo.setCurrentIndex(1)
        b._param_rows[-1]["value"].setValue(0.0215)
        d = b.to_dict()
        self.assertEqual(d["bkg_order"], 1)

        b2 = self._block()
        b2.from_dict(d)
        self.assertEqual(b2._bkg_order(), 1)
        self.assertEqual(self._names(b2)[-1], "Bkg_p1")
        self.assertAlmostEqual(b2._param_rows[-1]["value"].value(), 0.0215)

    def test_a_save_without_bkg_order_reloads_as_constant(self):
        b = self._block()
        d = b.to_dict()
        d.pop("bkg_order")
        b2 = self._block()
        b2.from_dict(d)
        self.assertEqual(b2._bkg_order(), 0)
        self.assertEqual(self._names(b2)[-1], "Bkg_p0")

    def test_models_that_ARE_backgrounds_hide_the_selector(self):
        b = self._block()
        for t in ("Polynomial", "Piecewise Constant", "Exponential Decay"):
            with self.subTest(model_type=t):
                b._type_combo.setCurrentText(t)
                self.assertFalse(b._bkg_combo.isVisibleTo(b))
        for t in ("HFS", "Voigt", "Skewed Voigt"):
            with self.subTest(model_type=t):
                b._type_combo.setCurrentText(t)
                self.assertTrue(b._bkg_combo.isVisibleTo(b))

    def test_slope_row_has_enough_decimals_to_be_editable(self):
        b = self._block()
        b._bkg_combo.setCurrentIndex(2)
        rows = {r["name"]: r for r in b._param_rows}
        self.assertGreater(rows["Bkg_p1"]["value"].decimals(),
                           rows["Bkg_p0"]["value"].decimals())
        self.assertGreater(rows["Bkg_p2"]["value"].decimals(),
                           rows["Bkg_p1"]["value"].decimals())


class PreAnalysisHandoffTests(unittest.TestCase):
    def _panel(self):
        from gui.preanalysis_tab import HFSModelPanel
        p = HFSModelPanel("Model_1")
        p.show()
        self.addCleanup(p.deleteLater)
        return p

    @staticmethod
    def _row_visible(panel, key):
        return panel._slider_params[key]["widgets"][0].isVisibleTo(panel)

    def test_unused_coefficient_rows_are_hidden(self):
        p = self._panel()
        self.assertTrue(self._row_visible(p, "bkg"))
        self.assertFalse(self._row_visible(p, "bkg_p1"))
        p._bkg_combo.setCurrentIndex(1)
        self.assertTrue(self._row_visible(p, "bkg_p1"))
        self.assertFalse(self._row_visible(p, "bkg_p2"))

    def test_slope_keeps_its_precision(self):
        """The panel's spinboxes carry 2 decimals, which rounds a
        0.0215 counts/MHz slope to 0.02 -- a 7% error on the tilt
        before the value ever leaves Pre-Analysis."""
        p = self._panel()
        p._bkg_combo.setCurrentIndex(1)
        p.set_value("bkg_p1", 0.0215)
        self.assertAlmostEqual(p.get_value("bkg_p1"), 0.0215)

    def test_coefficients_stop_at_the_chosen_order(self):
        p = self._panel()
        p.set_value("bkg", 5.0)
        p.set_value("bkg_p1", 0.02)
        self.assertEqual(p.bkg_coefficients(), [5.0])
        p._bkg_combo.setCurrentIndex(1)
        self.assertEqual(p.bkg_coefficients(), [5.0, 0.02])

    def test_shape_round_trips_through_the_save_file(self):
        p = self._panel()
        p._bkg_combo.setCurrentIndex(1)
        p.set_value("bkg", 5.02)
        p.set_value("bkg_p1", 0.0215)
        d = p.to_dict()
        p2 = self._panel()
        p2.from_dict(d)
        self.assertEqual(p2.bkg_order(), 1)
        self.assertEqual(p2.bkg_coefficients(), [5.02, 0.0215])

    def test_import_carries_the_shape_into_the_model_block(self):
        """The whole point: no hand-built second model block."""
        import gui.analysis.blocks as blocks
        from gui.analysis.blocks import ModelBlock

        p = self._panel()
        p._bkg_combo.setCurrentIndex(1)
        p.set_value("bkg", 5.02)
        p.set_value("bkg_p1", 0.0215)

        class _Tab:
            _model_panels = [p]

        original = blocks._choose_pa_project
        blocks._choose_pa_project = lambda *a, **k: _Tab()
        self.addCleanup(setattr, blocks, "_choose_pa_project", original)

        b = ModelBlock("Model_1")
        self.addCleanup(b.deleteLater)
        b._import_from_preanalysis()

        self.assertEqual(b._bkg_order(), 1)
        rows = {r["name"]: r for r in b._param_rows}
        self.assertAlmostEqual(rows["Bkg_p0"]["value"].value(), 5.02)
        self.assertAlmostEqual(rows["Bkg_p1"]["value"].value(), 0.0215)


if __name__ == "__main__":
    unittest.main()
