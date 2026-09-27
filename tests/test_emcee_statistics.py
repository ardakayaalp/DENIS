"""What an emcee (or likelihood) fit reports about itself.

Found on run 7507 (2026-09-25): a Chi-square + emcee fit whose curve
had a reduced chi-square of 1.72 reported 181134, with "# data points
= 1". lmfit evaluates an emcee objective once at the medians, gets the
summed log-likelihood as ONE number, and computes chisqr = (ln L)^2
from it. A Poisson + emcee fit was recomputed from the residuals but
divided by a dof taken from that "1 data point", so its reduced
chi-square was the total chi-square.

Also found: after the burn-in recompute the table held the burned-in
values while the models -- which every plotted curve, residual and
chi-square is evaluated from -- still held satlas2's full-chain ones.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_emcee_statistics.py -q
"""
import os
import tempfile
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

satlas2 = __import__("pytest").importorskip("satlas2")

from gui.analysis.fitting import (  # noqa: E402
    RESIDUAL_STATS_NOTE, _apply_emcee_burnin, _fit_single_run,
    _residual_statistics, _rewrite_fit_statistics)

REPORT = """[[Fit Statistics]]
    # fitting method   = emcee
    # function evals   = 50000
    # data points      = 1
    # variables        = 8
    chi-square         = 181134.581
    reduced chi-square = 181134.581
    Akaike info crit   = 28.1069956
    Bayesian info crit = 12.1069956
[[Variables]]
    s___v___mu: 628.437 +/- 0.819
"""


class ResidualStatisticsTests(unittest.TestCase):

    def test_hand_computed(self):
        y = np.array([10.0, 12.0, 9.0, 11.0, 30.0])
        f = np.array([10.5, 11.0, 9.5, 10.0, 29.0])
        e = np.array([1.0, 2.0, 1.5, 1.0, 5.0])
        st = _residual_statistics(y, f, e, n_vary=2)
        chi = float(np.sum(((y - f) / e) ** 2))
        self.assertEqual(st["ndata"], 5)
        self.assertAlmostEqual(st["chisqr"], chi)
        self.assertAlmostEqual(st["redchi"], chi / 3)
        self.assertAlmostEqual(st["aic"], 5 * np.log(chi / 5) + 2 * 2)
        self.assertAlmostEqual(st["bic"], 5 * np.log(chi / 5) + np.log(5) * 2)

    def test_it_matches_lmfit_on_a_least_squares_fit(self):
        """The same definitions lmfit uses, so objectives compare."""
        x = np.linspace(-300, 300, 121)
        rng = np.random.default_rng(3)
        y = rng.poisson(satlas2.Voigt(40.0, 0.0, 60.0, 40.0, name="v").f(x)
                        + 3.0).astype(float)
        src = satlas2.Source(x, y, yerr=np.sqrt(y + 1), name="s")
        src.addModel(satlas2.Voigt(35.0, 5.0, 60.0, 40.0, name="v"))
        src.addModel(satlas2.Polynomial([2.0], name="b"))
        f = satlas2.Fitter()
        f.addSource(src)
        f.fit(method="leastsq", scale_covar=False)
        st = _residual_statistics(y, src.evaluate(x), np.sqrt(y + 1),
                                  f.nvarys)
        self.assertEqual(st["ndata"], f.ndata)
        self.assertAlmostEqual(st["chisqr"], f.chisqr, places=6)
        self.assertAlmostEqual(st["redchi"], f.redchi, places=6)
        self.assertAlmostEqual(st["aic"], f.result.aic, places=6)
        self.assertAlmostEqual(st["bic"], f.result.bic, places=6)

    def test_unusable_bins_are_left_out(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        e = np.array([1.0, 0.0, 1.0, 1.0])
        st = _residual_statistics(y, y, e, n_vary=1)
        self.assertEqual(st["ndata"], 2)
        self.assertEqual(st["chisqr"], 0.0)


class ReportRewriteTests(unittest.TestCase):

    STATS = {"ndata": 502, "nvarys": 8, "nfree": 494, "chisqr": 851.2,
             "redchi": 1.7231, "aic": 281.1, "bic": 314.8}

    def test_the_numbers_are_replaced(self):
        out = _rewrite_fit_statistics(REPORT, self.STATS)
        self.assertIn("# data points      = 502", out)
        self.assertIn("reduced chi-square = 1.7231", out)
        self.assertNotIn("181134", out)

    def test_everything_else_is_kept(self):
        out = _rewrite_fit_statistics(REPORT, self.STATS)
        self.assertIn("# fitting method   = emcee", out)
        self.assertIn("# variables        = 8", out)
        self.assertIn("s___v___mu: 628.437 +/- 0.819", out)
        self.assertLess(out.index("[[Fit Statistics]]"),
                        out.index("[[Variables]]"))

    def test_it_says_where_the_numbers_came_from_once(self):
        once = _rewrite_fit_statistics(REPORT, self.STATS)
        twice = _rewrite_fit_statistics(once, self.STATS)
        self.assertEqual(twice.count(RESIDUAL_STATS_NOTE), 1)
        lines = once.splitlines()
        i = next(k for k, ln in enumerate(lines) if "Bayesian" in ln)
        self.assertIn(RESIDUAL_STATS_NOTE, lines[i + 1])

    def test_a_report_without_statistics_is_untouched(self):
        self.assertEqual(_rewrite_fit_statistics("no stats", self.STATS),
                         "no stats")


class BurnInModelTests(unittest.TestCase):
    """After the burn-in recompute the models hold the table's values."""

    def _fitter(self):
        x = np.linspace(-300, 300, 121)
        rng = np.random.default_rng(5)
        y = rng.poisson(satlas2.Voigt(40.0, 0.0, 60.0, 40.0, name="v").f(x)
                        + 3.0).astype(float)
        src = satlas2.Source(x, y, yerr=np.sqrt(y + 1), name="s")
        src.addModel(satlas2.Voigt(35.0, 5.0, 60.0, 40.0, name="v"))
        # With the background in the model no width collapses onto its
        # bound, where lmfit would clip a fake chain's median.
        src.addModel(satlas2.Polynomial([2.0], name="b"))
        f = satlas2.Fitter()
        f.addSource(src)
        f.fit(method="leastsq", scale_covar=False)
        return f

    def _chain_file(self, fitter, shift):
        import h5py
        names = [n for n, p in fitter.result.params.items() if p.vary]
        best = np.array([fitter.result.params[n].value for n in names])
        rng = np.random.default_rng(1)
        chain = best + rng.normal(scale=0.1, size=(60, 8, len(names)))
        chain[:20] += shift                  # a transient to discard
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        with h5py.File(path, "w") as hf:
            g = hf.create_group("mcmc")
            g.create_dataset("chain", data=chain)
            g.attrs["labels"] = names
        self.addCleanup(os.remove, path)
        return path, names, chain

    def _model_value(self, fitter, full_name):
        s, m, p = full_name.split("___")
        for sname, src in fitter.sources:
            for mname, model in src.models:
                if sname == s and mname == m:
                    return model.params[p].value
        raise KeyError(full_name)

    def test_values_come_from_the_kept_steps(self):
        f = self._fitter()
        path, names, chain = self._chain_file(f, shift=50.0)
        _apply_emcee_burnin(f, path, burn=20, thin=1)
        for j, n in enumerate(names):
            with self.subTest(param=n):
                self.assertAlmostEqual(f.result.params[n].value,
                                       float(np.median(chain[20:, :, j])))

    def test_the_models_hold_them_too(self):
        f = self._fitter()
        path, names, _chain = self._chain_file(f, shift=50.0)
        _apply_emcee_burnin(f, path, burn=20, thin=1)
        for n in names:
            with self.subTest(param=n):
                self.assertAlmostEqual(self._model_value(f, n),
                                       f.result.params[n].value)

    def test_correlations_come_from_the_kept_steps(self):
        f = self._fitter()
        path, names, chain = self._chain_file(f, shift=50.0)
        _apply_emcee_burnin(f, path, burn=20, thin=1)
        flat = chain[20:].reshape(-1, len(names))
        want = np.corrcoef(flat.T)[0, 1]
        self.assertAlmostEqual(
            f.result.params[names[0]].correl[names[1]], want)


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


class EmceeFitEndToEndTests(unittest.TestCase):
    """A short emcee fit through DENIS's own worker."""

    @classmethod
    def setUpClass(cls):
        cls.fitter = {"separate": True, "method": "emcee",
                      "statistics": "Chi-square", "llh": False,
                      "llh_method": "", "scale_covar": False,
                      "nwalkers": 16, "steps": 200, "burn": 50, "thin": 1,
                      "convergence": False}
        cls.res = _fit_single_run(
            "merged://synthetic", _source_config(), _voigt_config(),
            cls.fitter, output_config={"walk_plot": True,
                                       "correl_plot": True},
            merged_data=_merged())

    def test_it_fits(self):
        self.assertTrue(self.res.get("success"), self.res.get("error"))

    def test_the_data_points_are_the_bins(self):
        self.assertEqual(self.res["fit_quality"]["ndata"], 121)
        self.assertEqual(self.res["metadata_df"]["Data points"], [121])
        self.assertIn("# data points      = 121", self.res["report"])

    def test_the_reduced_chi_square_is_a_chi_square(self):
        rc = self.res["fit_quality"]["redchi"]
        self.assertTrue(0.3 < rc < 3.0, rc)
        self.assertIn(RESIDUAL_STATS_NOTE, self.res["report"])

    def test_it_is_the_chi_square_of_the_drawn_curve(self):
        y = np.asarray(self.res["y"])
        yf = np.asarray(self.res["y_fit"])
        ye = np.asarray(self.res["yerr"])
        chi = float(np.sum(((y - yf) / ye) ** 2))
        self.assertAlmostEqual(self.res["fit_quality"]["chisqr"], chi,
                               places=6)

    def test_the_drawn_curve_is_the_fit_in_the_table(self):
        p = self.res["params_df"]
        v = dict(zip(p["Parameter"], p["Value"]))
        x = np.asarray(self.res["x"])
        curve = (satlas2.Voigt(v["A"], v["mu"], v["FWHMG"], v["FWHML"],
                               name="c").f(x) + v["p0"])
        np.testing.assert_allclose(self.res["y_fit"], curve, rtol=1e-6)

    def test_the_walk_plot_knows_the_cut(self):
        wd = self.res["diagnostics"]["walk_data"]
        self.assertEqual(wd["burn"], 50)
        self.assertEqual(np.asarray(wd["chain"]).shape[:2], (200, 16))

    def test_the_corner_plot_uses_the_kept_steps(self):
        flat = np.asarray(self.res["diagnostics"]["correl_data"]["flatchain"])
        self.assertEqual(flat.shape[0], (200 - 50) * 16)

    def test_the_cut_travels_with_the_result(self):
        self.assertEqual(self.res["mcmc_burn"], 50)
        self.assertEqual(self.res["mcmc_thin"], 1)


class LeastSquaresUntouchedTests(unittest.TestCase):

    def test_a_chi_square_leastsq_report_is_lmfits_own(self):
        cfg = {"separate": True, "method": "leastsq",
               "statistics": "Chi-square", "llh": False, "llh_method": "",
               "scale_covar": False}
        res = _fit_single_run("merged://synthetic", _source_config(),
                              _voigt_config(), cfg, output_config={},
                              merged_data=_merged())
        self.assertTrue(res.get("success"), res.get("error"))
        self.assertNotIn(RESIDUAL_STATS_NOTE, res["report"])
        self.assertEqual(res["fit_quality"]["ndata"], 121)


if __name__ == "__main__":
    unittest.main()
