"""Tests for the count-data error-bar lower-clip helper.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Verifies count_errorbar_pairs(y, yerr): it returns a [lower, upper] pair
where the lower bar is clipped so that y - lower >= 0 everywhere (the
rendered error bar never reaches into negative counts), leaves the upper
bar untouched, passes high counts through symmetrically, and broadcasts a
scalar yerr across the y array.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_count_errorbar_pairs -v

Depends on: gui.shared_widgets (count_errorbar_pairs).
"""

import unittest

import numpy as np


class CountErrorbarPairsTests(unittest.TestCase):
    """``count_errorbar_pairs(y, yerr)`` returns a ``[lower, upper]``
    pair where ``y − lower ≥ 0`` everywhere, so the rendered error
    bar never reaches into negative counts."""

    def test_zero_count_bar_runs_zero_to_yerr(self):
        from gui.shared_widgets import count_errorbar_pairs

        y = np.array([0.0, 0.0])
        yerr = np.array([1.0, 2.0])
        pair = count_errorbar_pairs(y, yerr)
        np.testing.assert_array_equal(pair[0], [0.0, 0.0])
        np.testing.assert_array_equal(pair[1], [1.0, 2.0])

    def test_high_count_bar_symmetric(self):
        from gui.shared_widgets import count_errorbar_pairs

        y = np.array([10.0, 100.0])
        yerr = np.array([3.0, 10.0])
        pair = count_errorbar_pairs(y, yerr)
        # Lower bar isn't clipped (y − 3 = 7 ≥ 0, y − 10 = 90 ≥ 0).
        np.testing.assert_array_equal(pair[0], [3.0, 10.0])
        np.testing.assert_array_equal(pair[1], [3.0, 10.0])

    def test_partial_clip_for_low_counts(self):
        """y=2, yerr=3 → lower=2 (so y − lower = 0), upper unchanged."""
        from gui.shared_widgets import count_errorbar_pairs

        y = np.array([2.0])
        yerr = np.array([3.0])
        pair = count_errorbar_pairs(y, yerr)
        np.testing.assert_array_equal(pair[0], [2.0])
        np.testing.assert_array_equal(pair[1], [3.0])

    def test_scalar_yerr_broadcasts(self):
        from gui.shared_widgets import count_errorbar_pairs

        y = np.array([0.0, 5.0, 0.0])
        pair = count_errorbar_pairs(y, 1.0)
        np.testing.assert_array_equal(pair[0], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(pair[1], [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
