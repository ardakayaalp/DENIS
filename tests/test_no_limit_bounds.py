"""An unticked Min/Max reaches lmfit as +-inf, not as +-1e12.

The Model block stores "no limit" as -1e12 / +1e12 (a limit only counts
as set inside +-1e11). Passed on as numbers, lmfit fitted those
parameters in its bounded internal coordinate, where over a 2e12 range
its finite-difference steps are useless (2026-09-26, run 7507):
  * nelder / powell: the numerical Hessian failed -> no error bars;
  * leastsq: the background error came out 0.111 against 0.247 from
    the exact chi-square curvature, scale 6.66 against 8.37.
T02's explicit finite bounds (1e6-1e8) were not affected (<= 0.5 %).

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_no_limit_bounds.py -q
"""
import copy
import math
import os
import unittest
import warnings

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

satlas2 = __import__("pytest").importorskip("satlas2")

import gui.analysis.fitting as F  # noqa: E402
from gui.analysis.fitting import _fit_single_run, _unset_sentinel_bounds  # noqa: E402


def _p(value, lo, hi):
    return {"value": value, "vary": True, "min": lo, "max": hi, "expr": ""}


def _configs():
    """A Voigt as the Model block sends it with limits unticked."""
    return [{"type": "Voigt", "name": "M1", "params": {
        "A": _p(38.0, 0.0, 1e12),           # min ticked, max unticked
        "mu": _p(3.0, -1e12, 1e12),         # both unticked
        "FWHMG": _p(55.0, 1.0, 600.0),      # both ticked
        "FWHML": _p(45.0, 1.0, 600.0),
        "Bkg_p0": _p(2.5, 0.0, 1e12)}}]


class UnitTests(unittest.TestCase):

    def test_no_limit_becomes_infinite(self):
        p = _unset_sentinel_bounds(_configs())[0]["params"]
        self.assertEqual(p["mu"]["min"], -math.inf)
        self.assertEqual(p["mu"]["max"], math.inf)
        self.assertEqual(p["A"]["max"], math.inf)
        self.assertEqual(p["Bkg_p0"]["max"], math.inf)

    def test_real_limits_are_kept(self):
        p = _unset_sentinel_bounds(_configs())[0]["params"]
        self.assertEqual(p["A"]["min"], 0.0)
        self.assertEqual((p["FWHMG"]["min"], p["FWHMG"]["max"]), (1.0, 600.0))

    def test_large_but_real_limits_are_kept(self):
        """T02 carries 1e6-1e8 bounds on purpose; they stay."""
        cfg = [{"type": "Voigt", "name": "M", "params": {
            "mu": _p(0.0, -1e8, 1e8), "Bkg_p0": _p(1.0, -1e6, 1e6)}}]
        p = _unset_sentinel_bounds(cfg)[0]["params"]
        self.assertEqual((p["mu"]["min"], p["mu"]["max"]), (-1e8, 1e8))
        self.assertEqual(p["Bkg_p0"]["max"], 1e6)

    def test_none_stays_none(self):
        cfg = [{"type": "Voigt", "name": "M",
                "params": {"mu": _p(0.0, None, None)}}]
        p = _unset_sentinel_bounds(cfg)[0]["params"]["mu"]
        self.assertIsNone(p["min"])
        self.assertIsNone(p["max"])

    def test_the_callers_configs_are_not_touched(self):
        cfg = _configs()
        before = copy.deepcopy(cfg)
        _unset_sentinel_bounds(cfg)
        self.assertEqual(cfg, before)

    def test_every_model_builder_goes_through_it(self):
        import inspect
        src = inspect.getsource(F._build_models_on_source)
        self.assertIn("_unset_sentinel_bounds(model_configs)", src)


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


def _merged():
    x = np.linspace(-300.0, 300.0, 121)
    truth = satlas2.Voigt(40.0, 0.0, 60.0, 40.0, name="v").f(x) + 3.0
    y = np.random.default_rng(9).poisson(truth).astype(float)
    return {"merged_name": "synthetic", "x": x.tolist(), "y": y.tolist(),
            "yerr": np.sqrt(y + 1).tolist(), "x_unit": "MHz",
            "source_runs": ["1"], "per_run": [], "source_files": []}


def _fit(**over):
    cfg = {"separate": True, "method": "leastsq", "statistics": "Chi-square",
           "llh": False, "llh_method": "", "scale_covar": False}
    cfg.update(over)
    res = _fit_single_run("merged://synthetic", _source_config(), _configs(),
                          cfg, output_config={}, merged_data=_merged())
    assert res.get("success"), res.get("error")
    p = res["params_df"]
    return (res, dict(zip(p["Parameter"], p["Value"])),
            dict(zip(p["Parameter"], p["Stderr"])))


class EndToEndTests(unittest.TestCase):

    def test_nelder_under_a_likelihood_has_error_bars(self):
        _r, _v, e = _fit(method="nelder", llh=True, llh_method="poisson",
                         statistics="Poisson LLH")
        for n in ("A", "mu", "p0"):
            with self.subTest(param=n):
                self.assertTrue(e[n] is not None and e[n] > 0, e[n])

    def test_leastsq_errors_match_the_exact_curvature(self):
        import numdifftools as ndt
        res, v, e = _fit()
        x = np.asarray(res["x"])
        y = np.asarray(res["y"])
        ye = np.asarray(res["yerr"])
        names = ["A", "mu", "FWHMG", "FWHML", "p0"]

        def chi2(t):
            f = satlas2.Voigt(t[0], t[1], t[2], t[3], name="c").f(x) + t[4]
            return float(np.sum(((y - f) / ye) ** 2))
        best = np.array([v[n] for n in names])
        exact = dict(zip(names, np.sqrt(np.diag(
            2 * np.linalg.inv(ndt.Hessian(chi2, step=1e-4)(best))))))
        for n in ("mu", "p0"):
            with self.subTest(param=n):
                self.assertAlmostEqual(e[n] / exact[n], 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
