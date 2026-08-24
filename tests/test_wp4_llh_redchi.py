"""Tests for WP4: genuine reduced chi-square under an LLH objective.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Under a Poisson/Gaussian LLH fit, satlas2 reports the optimized -logL surrogate
in chisqr/redchi, so the displayed "Reduced Chi-sq" is not a chi-square.
fitting._genuine_chisq recomputes a real chi-square from the residuals; the fit
worker uses it to overwrite the reported goodness-of-fit when llh is on. These
tests pin the formula and confirm it reproduces the chi-square a default
least-squares fit reports (so it is a valid goodness-of-fit).

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_wp4_llh_redchi -v

Depends on: gui.analysis.fitting._genuine_chisq; the satlas2 cross-check needs
satlas2.
"""

import unittest

import numpy as np

from gui.analysis.fitting import _genuine_chisq


class GenuineChisqFormulaTests(unittest.TestCase):
    def test_matches_hand_computation(self):
        y = np.array([10.0, 12.0, 9.0, 11.0])
        f = np.array([10.5, 11.0, 9.5, 10.0])
        e = np.array([1.0, 2.0, 1.5, 1.0])
        r = (y - f) / e
        exp_chisq = float(np.sum(r * r))
        chisq, redchi = _genuine_chisq(y, f, e, n_data=4, n_vary=2)
        self.assertAlmostEqual(chisq, exp_chisq)
        self.assertAlmostEqual(redchi, exp_chisq / 2)  # dof = 4 - 2

    def test_dof_floor_avoids_zero_division(self):
        y = np.array([1.0, 2.0])
        chisq, redchi = _genuine_chisq(y, y, np.array([1.0, 1.0]),
                                       n_data=2, n_vary=5)
        # n_data - n_vary <= 0 -> dof floored to 1, no ZeroDivisionError.
        self.assertEqual(chisq, 0.0)
        self.assertEqual(redchi, 0.0)


class MatchesSatlas2ChiSquareTests(unittest.TestCase):
    def test_genuine_redchi_matches_satlas2_chi2_fit(self):
        import pytest
        satlas2 = pytest.importorskip("satlas2")

        x = np.linspace(-200.0, 200.0, 120)
        truth = satlas2.Voigt(8.0, 0.0, 40.0, 40.0, name="M1")
        y = truth.f(x) + 5.0
        yerr = np.sqrt(y + 1.0)

        src = satlas2.Source(x, y, yerr=yerr, name="R")
        src.addModel(satlas2.Voigt(6.0, 5.0, 30.0, 30.0, name="M1"))
        fitter = satlas2.Fitter()
        fitter.addSource(src)
        fitter.fit(method="leastsq", scale_covar=False)

        y_fit = src.evaluate(x)
        _cx, rc = _genuine_chisq(y, y_fit, yerr, fitter.ndata, fitter.nvarys)
        # Our recomputed reduced chi-square reproduces what the chi-square fit
        # itself reports -> it is a valid goodness-of-fit to show under LLH.
        self.assertAlmostEqual(rc, float(fitter.redchi), places=4)


if __name__ == "__main__":
    unittest.main()
