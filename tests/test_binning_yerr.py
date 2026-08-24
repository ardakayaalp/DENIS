"""Headless tests for gui.analysis.binning yerr handling.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Verifies the y-error modes exposed by the binning module: that "None" is
listed first and yields no error array, that the Poisson sqrt(y+1) mode
returns the expected per-bin uncertainties, and that compute_binned with
yerr_mode "None" produces yerr=None while still populating x and y. Uses
a minimal fake CLSDataFrame so the suite needs no real .asdf file.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_binning_yerr -v

Depends on: gui.analysis.binning (YERR_MODES, _make_yerr,
_apply_low_count_correction, compute_binned).
"""

import unittest

import numpy as np

from gui.analysis.binning import (
    YERR_MODES,
    _make_yerr,
    _apply_low_count_correction,
)


class YerrNoneTests(unittest.TestCase):

    def test_none_listed_first_in_yerr_modes(self):
        self.assertEqual(YERR_MODES[0], "None")
        # Existing modes must still be present and in order.
        self.assertIn("Poisson sqrt(y+1)", YERR_MODES)
        self.assertIn("Poisson sqrt(y)", YERR_MODES)
        self.assertIn("Model-based", YERR_MODES)

    def test_make_yerr_none_returns_none(self):
        y = np.array([1.0, 2.0, 3.0])
        yerr, use_callable = _make_yerr(y, "None")
        self.assertIsNone(yerr)
        self.assertFalse(use_callable)

    def test_make_yerr_poisson_sqrt_y_plus_one_unchanged(self):
        y = np.array([0.0, 1.0, 4.0])
        yerr, use_callable = _make_yerr(y, "Poisson sqrt(y+1)")
        np.testing.assert_allclose(yerr, np.sqrt(y + 1))
        self.assertFalse(use_callable)


class ComputeBinnedYerrNoneTests(unittest.TestCase):
    """compute_binned with yerr_mode='None' must return yerr=None.

    Builds a minimal fake CLSDataFrame so we don't depend on a real
    .asdf file. Frequency mode is exercised because Raw Voltage uses
    Compute_Raw_Bins which has the same yerr path.
    """

    def test_compute_binned_yerr_none(self):
        import pandas as pd
        from gui.analysis.binning import compute_binned

        class FakeData:
            def Compute_Bins(self, **kwargs):
                # Two non-empty bins at 100/200 MHz (1e8/2e8 Hz).
                self.Binned = pd.DataFrame({
                    "Fcount": [4.0, 9.0],
                    "bins_center": [1.0e8, 2.0e8],
                    "bins_width": [1.0e7, 1.0e7],
                })

        data = FakeData()
        out = compute_binned(data, {"yerr_mode": "None"})
        self.assertIsNone(out["yerr"])
        # Sanity: x/y are still populated.
        self.assertEqual(len(out["x"]), 2)
        self.assertEqual(out["y"].tolist(), [4.0, 9.0])


if __name__ == "__main__":
    unittest.main()
