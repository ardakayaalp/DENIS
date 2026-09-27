"""lmfit's x2 on likelihood error bars: flagged, and corrected on request.

For every minimiser except leastsq, least_squares and emcee, lmfit
estimates the covariance as 2 x inverse(Hessian) of the minimised
value -- exact for a chi-square (chi2 = -2 ln L). Under a likelihood
satlas2 minimises -ln L, half a chi-square, so the error bars come out
sqrt(2) too large. Arda's decision (2026-09-26): warn, and let the user
correct it knowingly -- off by default, the report says which was done.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_llh_error_correction.py -q
"""
import math
import os
import unittest
import warnings

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

satlas2 = __import__("pytest").importorskip("satlas2")

from gui.analysis.fitting import (  # noqa: E402
    LLH_ERROR_FACTOR, _correct_llh_errors, _fit_single_run,
    llh_errors_inflated)


def _cfg(**kw):
    cfg = {"separate": True, "method": "nelder", "statistics": "Poisson LLH",
           "llh": True, "llh_method": "poisson", "scale_covar": False}
    cfg.update(kw)
    return cfg


class DetectionTests(unittest.TestCase):

    def test_scalar_minimisers_under_a_likelihood(self):
        for m in ("nelder", "powell", "cobyla", "slsqp"):
            with self.subTest(method=m):
                self.assertTrue(llh_errors_inflated(_cfg(method=m)))

    def test_leastsq_counts_because_it_becomes_nelder(self):
        self.assertTrue(llh_errors_inflated(_cfg(method="leastsq")))
        self.assertTrue(llh_errors_inflated(_cfg(method="least_squares")))

    def test_emcee_is_not_affected(self):
        self.assertFalse(llh_errors_inflated(_cfg(method="emcee")))

    def test_a_chi_square_is_not_affected(self):
        for m in ("leastsq", "nelder", "powell", "emcee"):
            with self.subTest(method=m):
                self.assertFalse(llh_errors_inflated(
                    _cfg(method=m, llh=False, llh_method="")))

    def test_the_factor(self):
        self.assertAlmostEqual(LLH_ERROR_FACTOR, 1 / math.sqrt(2))


class CorrectionTests(unittest.TestCase):

    def _fitter(self):
        x = np.linspace(-300, 300, 121)
        y = np.random.default_rng(5).poisson(
            satlas2.Voigt(40.0, 0.0, 60.0, 40.0, name="v").f(x) + 3.0
        ).astype(float)
        src = satlas2.Source(x, y, yerr=np.sqrt(y + 1), name="s")
        src.addModel(satlas2.Voigt(35.0, 5.0, 60.0, 40.0, name="v"))
        src.addModel(satlas2.Polynomial([2.0], name="b"))
        f = satlas2.Fitter()
        f.addSource(src)
        f.fit(llh=True, llh_method="poisson", method="nelder")
        return f

    def test_every_error_is_divided_by_sqrt2(self):
        f = self._fitter()
        before = {n: p.stderr for n, p in f.result.params.items()}
        _correct_llh_errors(f)
        for n, p in f.result.params.items():
            with self.subTest(param=n):
                if before[n] is None:
                    self.assertIsNone(p.stderr)
                else:
                    self.assertAlmostEqual(p.stderr,
                                           before[n] / math.sqrt(2))

    def test_the_covariance_is_halved(self):
        f = self._fitter()
        before = np.array(f.result.covar)
        _correct_llh_errors(f)
        np.testing.assert_allclose(f.result.covar, before / 2)

    def test_the_values_do_not_move(self):
        f = self._fitter()
        before = {n: p.value for n, p in f.result.params.items()}
        _correct_llh_errors(f)
        for n, p in f.result.params.items():
            self.assertEqual(p.value, before[n])


# ── end to end, through DENIS's own worker ──────────────────────────
def _source_config():
    return {
        "Z": 32, "A": 73, "mass": 72.923459, "harmonic": 2,
        "e_lower": 0, "e_upper": 0, "tof_gate": [38.0, 44.0],
        "pmt_gate": [3, 4], "v_gate": None, "f_gate": None,
        "noise_filter": 0, "cooler_correction": "pbp",
        "ref_freq": 0, "ref_shift": 0,
        "cooler_override": 29977.0, "laser_override": 10920.0,
        "cal_order": 1, "override_enabled": False,
        "bin_mode": "Frequency", "yerr_mode": "Poisson sqrt(y+1)",
        "x_column": "bins_center",
    }


def _voigt_config():
    def p(value):
        return {"value": value, "vary": True, "min": None, "max": None,
                "expr": ""}
    return [{"type": "Voigt", "name": "M1",
             "params": {"A": p(38.0), "mu": p(3.0), "FWHMG": p(55.0),
                        "FWHML": p(45.0), "Bkg_p0": p(2.5)}}]


def _merged():
    x = np.linspace(-300.0, 300.0, 121)
    truth = satlas2.Voigt(40.0, 0.0, 60.0, 40.0, name="v").f(x) + 3.0
    y = np.random.default_rng(9).poisson(truth).astype(float)
    return {"merged_name": "synthetic", "x": x.tolist(), "y": y.tolist(),
            "yerr": np.sqrt(y + 1).tolist(), "x_unit": "MHz",
            "source_runs": ["1"], "per_run": [], "source_files": []}


def _fit(**over):
    res = _fit_single_run("merged://synthetic", _source_config(),
                          _voigt_config(), _cfg(**over), output_config={},
                          merged_data=_merged())
    assert res.get("success"), res.get("error")
    p = res["params_df"]
    return res, dict(zip(p["Parameter"], p["Value"])), \
        dict(zip(p["Parameter"], p["Stderr"]))


class EndToEndTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.off, cls.v_off, cls.e_off = _fit(llh_error_correction=False)
        cls.on, cls.v_on, cls.e_on = _fit(llh_error_correction=True)

    def test_off_by_default_nothing_changes_but_it_says_so(self):
        self.assertIn("[Errors]", self.off["report"])
        self.assertIn("NOT corrected", self.off["report"])

    def test_on_divides_by_sqrt2_and_says_so(self):
        for n in ("A", "mu", "FWHMG", "FWHML"):
            with self.subTest(param=n):
                self.assertAlmostEqual(self.e_on[n],
                                       self.e_off[n] / math.sqrt(2),
                                       places=9)
        self.assertIn("divided by sqrt(2)", self.on["report"])

    def test_the_values_are_the_same(self):
        for n in self.v_off:
            self.assertEqual(self.v_on[n], self.v_off[n])

    def test_the_corrected_error_is_the_curvature_of_minus_ln_l(self):
        """The whole point: corrected = inverse Hessian of -ln L."""
        import numdifftools as ndt
        x = np.asarray(self.on["x"])
        y = np.asarray(self.on["y"])
        names = ["A", "mu", "FWHMG", "FWHML", "p0"]
        best = np.array([self.v_on[n] for n in names])

        def nll(t):
            f = satlas2.Voigt(t[0], t[1], t[2], t[3], name="c").f(x) + t[4]
            return float(np.sum(f - y * np.log(f)))
        cov = np.linalg.inv(ndt.Hessian(nll, step=1e-4)(best))
        truth = dict(zip(names, np.sqrt(np.diag(cov))))
        for n in ("A", "mu", "FWHML"):
            with self.subTest(param=n):
                self.assertAlmostEqual(self.e_on[n] / truth[n], 1.0,
                                       delta=0.02)
                self.assertAlmostEqual(self.e_off[n] / truth[n],
                                       math.sqrt(2), delta=0.03)

    def test_a_chi_square_fit_is_never_touched(self):
        res, _v, e = _fit(llh=False, llh_method="", statistics="Chi-square",
                          llh_error_correction=True)
        res0, _v0, e0 = _fit(llh=False, llh_method="",
                             statistics="Chi-square",
                             llh_error_correction=False)
        self.assertNotIn("[Errors]", res["report"])
        self.assertEqual(e, e0)

    def test_the_order_in_the_report(self):
        res, _v, _e = _fit(method="leastsq", llh_error_correction=True)
        rep = res["report"]
        self.assertLess(rep.index("[Minimiser]"), rep.index("[Errors]"))
        self.assertLess(rep.index("[Errors]"), rep.index("[[Fit Statistics]]"))


class FitterBlockTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _block(self, method, statistics):
        from gui.analysis.blocks import FitterBlock
        b = FitterBlock()
        self.addCleanup(b.deleteLater)
        b._method_combo.setCurrentText(method)
        b._stats_combo.setCurrentText(statistics)
        return b

    def test_the_option_shows_only_where_it_applies(self):
        cases = {("nelder", "Poisson LLH"): True,
                 ("leastsq", "Poisson LLH"): True,
                 ("powell", "Gaussian LLH"): True,
                 ("emcee", "Poisson LLH"): False,
                 ("nelder", "Chi-square"): False,
                 ("leastsq", "Chi-square"): False}
        for (m, st), shown in cases.items():
            with self.subTest(method=m, statistics=st):
                b = self._block(m, st)
                self.assertEqual(b._llh_err_fix.isVisibleTo(b), shown)

    def test_it_is_off_by_default(self):
        b = self._block("nelder", "Poisson LLH")
        self.assertFalse(b._llh_err_fix.isChecked())
        self.assertFalse(b.get_fitter_config()["llh_error_correction"])

    def test_the_warning_follows_the_choice(self):
        b = self._block("nelder", "Poisson LLH")
        self.assertIn("too large", b._llh_method_hint.text())
        b._llh_err_fix.setChecked(True)
        self.assertIn("divided", b._llh_method_hint.text())
        self.assertNotIn("too large", b._llh_method_hint.text())

    def test_it_is_saved_and_restored(self):
        b = self._block("nelder", "Poisson LLH")
        b._llh_err_fix.setChecked(True)
        d = b.to_dict()
        self.assertTrue(d["llh_error_correction"])
        c = self._block("leastsq", "Chi-square")
        c.from_dict(d)
        self.assertTrue(c._llh_err_fix.isChecked())
        self.assertTrue(c.get_fitter_config()["llh_error_correction"])

    def test_an_old_session_keeps_the_numbers_as_they_were(self):
        b = self._block("leastsq", "Chi-square")
        b.from_dict({"method": "nelder", "statistics": "Poisson LLH"})
        self.assertFalse(b._llh_err_fix.isChecked())


if __name__ == "__main__":
    unittest.main()
