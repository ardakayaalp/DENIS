"""Headless tests for gui.calibration: the fit, the rules, the resolution.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

The load-bearing test here is ``ClstoolsParityTests``: with no override,
our fit must reproduce clstools' ``Cal`` bit-for-bit, because that is the
promise the whole feature rests on -- routing every load path through
``apply_calibration`` changes nothing until the user asks for a change.

The rest covers the volts/raw convention (a p0 in volts and a p1 gain of
~1, vs clstools' mixed-unit slope of ~1e-3), each rejection rule, and
spec resolution including ``borrow`` and hand-entered coefficients.
Synthetic ASDF files carrying only CalSet/CalReadback stand in for real
runs, so the suite needs no experimental data.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_calibration_fit -v

Depends on: gui.calibration, numpy, asdf.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from gui.calibration import (
    CalibrationError,
    DEFAULT_VACC_DIV,
    FLAG_MIN_IMPACT_V,
    apply_calibration,
    auto_outliers,
    calibration_shift_v,
    clear_points_cache,
    describe_spec,
    diagnose,
    fit_calibration,
    normalize_spec,
    read_cal_points,
    resolve_calibration,
    spec_fingerprint,
    suspect_outliers,
)

VACC = DEFAULT_VACC_DIV  # 1000.0


# ── Synthetic calibration tables ─────────────────────────────────────

def make_table(n=21, gain=1.0007, offset=0.35, noise=0.03, seed=7,
               bad=((0, -4.0), (1, -2.5), (2, -1.4))):
    """A DAC->HV calibration table as the ASDF stores it.

    ``CalSet`` is in volts (a 0..400 V deceleration sweep); ``CalReadback``
    is in monitor units, i.e. volts / VAccDiv. ``bad`` injects the failure
    this feature exists for: the HV supply still settling over the first
    few points of the sweep.
    """
    rng = np.random.default_rng(seed)
    set_v = np.linspace(0.0, 400.0, n)
    read_v = gain * set_v + offset + rng.normal(0.0, noise, n)
    for idx, err in bad:
        read_v[idx] += err
    return set_v, read_v / VACC


def clstools_cal(set_v, read_raw, order=1):
    """clstools' own algorithm, verbatim (DataFrame.py:491-500)."""
    values, cov = np.polyfit(set_v, read_raw, order, cov=True)
    cal, cal_err = [], []
    for i, v in enumerate(values):
        cal.append(v)
        cal_err.append(cov[i, i])
    cal.reverse()
    cal_err.reverse()
    return cal, cal_err


class _FakeData:
    """Stand-in for clstools' CLSDataFrame (only the calibration attrs)."""

    def __init__(self, vacc_div=VACC):
        self.VAccDiv = vacc_div
        self.Cal = [0.0, 1.0]
        self.Cal_err = [0.0, 0.0]
        self.Cal_order = 1
        self.Cal_df = None


# ── Parity: the no-override default must change nothing ──────────────

class ClstoolsParityTests(unittest.TestCase):

    def test_fit_reproduces_clstools_cal_exactly(self):
        set_v, read_raw = make_table()
        expected, _ = clstools_cal(set_v, read_raw, 1)

        fit = fit_calibration(set_v, read_raw, order=1, excluded=(),
                              vacc_div=VACC)
        ours = [c / VACC for c in fit.coeffs_v]

        np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=0.0)

    def test_parity_holds_for_orders_2_and_3(self):
        set_v, read_raw = make_table(n=25)
        for order in (2, 3):
            with self.subTest(order=order):
                expected, _ = clstools_cal(set_v, read_raw, order)
                fit = fit_calibration(set_v, read_raw, order, (), VACC)
                ours = [c / VACC for c in fit.coeffs_v]
                np.testing.assert_allclose(ours, expected, rtol=1e-10)

    def test_cal_is_ascending_p0_first(self):
        """clstools reverses polyfit's output, so Cal[0] is the constant
        term and Compute_Voltages reads Cal[1] as the linear one."""
        set_v, read_raw = make_table(bad=())
        fit = fit_calibration(set_v, read_raw, 1, (), VACC)
        # p0 ~ the offset in volts, p1 ~ a gain near 1.
        self.assertLess(abs(fit.coeffs_v[1] - 1.0), 0.01)
        self.assertLess(abs(fit.coeffs_v[0]), 5.0)

    def test_predict_matches_compute_voltages_formula(self):
        """DV_cal[V] = poly_raw(DV) * VAccDiv must equal poly_v(DV)."""
        set_v, read_raw = make_table()
        fit = fit_calibration(set_v, read_raw, 1, (), VACC)
        cal_raw = np.array([c / VACC for c in fit.coeffs_v])

        dv = np.array([0.0, 137.5, 400.0])
        clstools_dv_cal = (dv * cal_raw[1] + cal_raw[0]) * VACC  # DataFrame.py:217
        np.testing.assert_allclose(fit.predict_v(dv), clstools_dv_cal,
                                   rtol=1e-12)


# ── The volts convention ─────────────────────────────────────────────

class UnitConventionTests(unittest.TestCase):

    def test_volts_raw_roundtrip(self):
        set_v, read_raw = make_table()
        fit = fit_calibration(set_v, read_raw, 1, (), VACC)
        back = [(c / VACC) * VACC for c in fit.coeffs_v]
        np.testing.assert_allclose(back, fit.coeffs_v, rtol=1e-15)

    def test_residuals_are_in_volts(self):
        """A readback 2 V above the calibration shows a 2 V residual.

        Excluded from the fit, so the line stays the clean identity and the
        residual is the injected error exactly. (Left *in* the fit, least
        squares absorbs part of it -- the residual would read 1.82 V, not 2 --
        which is correct behaviour and not what this test is about.)
        """
        set_v = np.linspace(0.0, 400.0, 11)
        read_v = 1.0 * set_v          # a perfect identity calibration
        read_v[5] += 2.0              # ...with one point 2 V high
        fit = fit_calibration(set_v, read_v / VACC, 1, excluded=(5,),
                              vacc_div=VACC)
        self.assertAlmostEqual(fit.residuals_v[5], 2.0, places=6)

    def test_errors_are_std_not_variance(self):
        """clstools stores cov[i,i] (a variance); we store sqrt of it."""
        set_v, read_raw = make_table()
        _, cls_errs = clstools_cal(set_v, read_raw, 1)   # variances, ascending
        fit = fit_calibration(set_v, read_raw, 1, (), VACC)
        ours_raw = [e / VACC for e in fit.errs_v]
        np.testing.assert_allclose(ours_raw, np.sqrt(np.abs(cls_errs)),
                                   rtol=1e-10)


# ── Rejection rules ──────────────────────────────────────────────────

class RejectionRuleTests(unittest.TestCase):

    def setUp(self):
        self.set_v, self.read_raw = make_table()   # first 3 points bad
        self.truth = (0.35, 1.0007)

    def _fit(self, excluded):
        return fit_calibration(self.set_v, self.read_raw, 1, excluded, VACC)

    def test_none_drops_nothing(self):
        self.assertEqual(
            auto_outliers(self.set_v, self.read_raw, 1, rule="none"), ())

    def test_first_n_drops_the_leading_points(self):
        out = auto_outliers(self.set_v, self.read_raw, 1,
                            rule="first_n", drop_first=3)
        self.assertEqual(out, (0, 1, 2))

    def test_first_n_clamps_to_table_length(self):
        out = auto_outliers(self.set_v, self.read_raw, 1,
                            rule="first_n", drop_first=999)
        self.assertEqual(len(out), len(self.set_v))

    def test_single_pass_sigma_matches_clstools_filter(self):
        """Bit-for-bit with the 2-sigma cut clstools c0334c0 applies."""
        import pandas as pd
        cal, _ = clstools_cal(self.set_v, self.read_raw, 1)
        df = pd.DataFrame({"Set": self.set_v, "Read": self.read_raw})
        df["Fit"] = df["Set"].apply(lambda x: np.polyval(cal[::-1], x))
        df["Residual"] = df["Read"] - df["Fit"]
        std = df["Residual"].std()                    # pandas ddof=1
        expected = set(df.index[np.abs(df["Residual"]) > 2 * std])

        ours = set(auto_outliers(self.set_v, self.read_raw, 1, rule="sigma",
                                 sigma=2.0, iterative=False, vacc_div=VACC))
        self.assertEqual(ours, expected)

    def test_single_pass_sigma_misses_clustered_outliers(self):
        """The weakness that motivates the iterative rule: three bad points
        inflate sigma enough that the 2-sigma cut only catches the worst."""
        out = auto_outliers(self.set_v, self.read_raw, 1, rule="sigma",
                            sigma=2.0, iterative=False, vacc_div=VACC)
        self.assertEqual(out, (0,))                 # 1 and 2 survive
        gain = self._fit(out).coeffs_v[1]
        self.assertGreater(abs(gain - self.truth[1]), 1e-3)   # still wrong

    def test_iterative_sigma_catches_the_whole_cluster(self):
        out = auto_outliers(self.set_v, self.read_raw, 1, rule="sigma",
                            sigma=3.0, iterative=True, vacc_div=VACC)
        self.assertEqual(out, (0, 1, 2))
        fit = self._fit(out)
        self.assertAlmostEqual(fit.coeffs_v[0], self.truth[0], delta=0.05)
        self.assertAlmostEqual(fit.coeffs_v[1], self.truth[1], delta=1e-4)

    def test_iterative_sigma_keeps_a_clean_table_intact(self):
        set_v, read_raw = make_table(bad=())
        self.assertEqual(
            auto_outliers(set_v, read_raw, 1, rule="sigma", sigma=3.0,
                          iterative=True, vacc_div=VACC), ())

    def test_iterative_never_makes_the_fit_degenerate(self):
        """Even against pure noise it must leave enough points to fit."""
        rng = np.random.default_rng(1)
        set_v = np.linspace(0, 400, 8)
        read_raw = rng.normal(0, 1, 8) / VACC
        out = auto_outliers(set_v, read_raw, 1, rule="sigma", sigma=0.01,
                            iterative=True, vacc_div=VACC)
        self.assertGreaterEqual(len(set_v) - len(out), 3)

    def test_unknown_rule_raises(self):
        with self.assertRaises(CalibrationError):
            auto_outliers(self.set_v, self.read_raw, 1, rule="wishful")


# ── Zero intercept (clstools' ignore_intercept, done in our own fit) ──

class ZeroInterceptTests(unittest.TestCase):
    """Forcing p0 = 0 must constrain the offset *out* of the fit, not fit
    it and then blank it -- otherwise the slope keeps the offset the data
    wanted and the calibration is simply wrong."""

    def test_through_origin_data_is_recovered_exactly(self):
        set_v = np.linspace(0.0, 400.0, 21)
        read_v = 1.0007 * set_v                 # no offset, no noise
        fit = fit_calibration(set_v, read_v / VACC, 1, (), VACC,
                              ignore_intercept=True)
        self.assertEqual(fit.coeffs_v[0], 0.0)
        self.assertAlmostEqual(fit.coeffs_v[1], 1.0007, places=9)

    def test_offset_is_forced_to_zero_and_slope_absorbs_it(self):
        """With a genuine +offset, the unconstrained fit finds it; the
        constrained fit pins p0 to 0 and the through-origin slope is
        pulled up to compensate -- provably NOT the unconstrained slope."""
        set_v = np.linspace(10.0, 400.0, 21)
        read_v = 1.0007 * set_v + 5.0           # 5 V offset
        y = read_v / VACC
        free = fit_calibration(set_v, y, 1, (), VACC)
        forced = fit_calibration(set_v, y, 1, (), VACC, ignore_intercept=True)

        self.assertAlmostEqual(free.coeffs_v[0], 5.0, delta=1e-6)   # found it
        self.assertEqual(forced.coeffs_v[0], 0.0)                   # forced out
        # A fit-then-zero would leave forced.p1 == free.p1; a real
        # constrained fit inflates the slope to carry the lost offset.
        self.assertGreater(forced.coeffs_v[1], free.coeffs_v[1])
        # Closed form for a through-origin line: c1 = <x·y> / <x²>.
        expected = float(np.sum(set_v * read_v) / np.sum(set_v ** 2))
        self.assertAlmostEqual(forced.coeffs_v[1], expected, places=9)

    def test_intercept_error_is_exactly_zero_slope_error_finite(self):
        set_v, read_raw = make_table(bad=())
        fit = fit_calibration(set_v, read_raw, 1, (), VACC,
                              ignore_intercept=True)
        self.assertEqual(fit.errs_v[0], 0.0)
        self.assertTrue(np.isfinite(fit.errs_v[1]) and fit.errs_v[1] > 0)

    def test_order_two_keeps_the_quadratic_but_kills_the_offset(self):
        set_v = np.linspace(0.0, 400.0, 25)
        read_v = 2e-4 * set_v ** 2 + 1.0 * set_v      # curved, through origin
        fit = fit_calibration(set_v, read_v / VACC, 2, (), VACC,
                              ignore_intercept=True)
        self.assertEqual(fit.coeffs_v[0], 0.0)
        self.assertAlmostEqual(fit.coeffs_v[1], 1.0, places=6)
        self.assertAlmostEqual(fit.coeffs_v[2], 2e-4, places=10)

    def test_resolve_and_apply_honour_ignore_intercept(self):
        import asdf
        tmp = tempfile.mkdtemp(prefix="cal_zi_")
        try:
            set_v = np.linspace(10.0, 400.0, 21)
            read_v = 1.0007 * set_v + 5.0
            path = os.path.join(tmp, "run_3003.asdf")
            asdf.AsdfFile({"CalSet": set_v,
                           "CalReadback": read_v / VACC}).write_to(path)
            clear_points_cache()

            cal = {path: {"mode": "fit", "ignore_intercept": True}}
            res = resolve_calibration(path, cal)
            self.assertTrue(res.ignore_intercept)
            self.assertEqual(res.coeffs_v[0], 0.0)
            self.assertTrue(res.to_report_dict()["ignore_intercept"])

            data = _FakeData()
            apply_calibration(data, path, cal)
            self.assertEqual(data.Cal[0], 0.0)      # raw convention, offset 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── Fit edge cases ───────────────────────────────────────────────────

class FitEdgeCaseTests(unittest.TestCase):

    def test_too_few_points_raises(self):
        with self.assertRaises(CalibrationError) as ctx:
            fit_calibration([1.0, 2.0], [0.001, 0.002], order=3)
        self.assertIn("order-3", str(ctx.exception))

    def test_exclusions_can_starve_the_fit(self):
        set_v, read_raw = make_table(n=5, bad=())
        with self.assertRaises(CalibrationError):
            fit_calibration(set_v, read_raw, 1, excluded=(0, 1, 2, 3, 4))

    def test_covariance_unavailable_yields_nan_errors(self):
        """np.polyfit(cov=True) needs n > order + 2; the fit itself does not."""
        set_v = np.array([0.0, 100.0, 200.0])
        read_raw = set_v / VACC
        fit = fit_calibration(set_v, read_raw, 1, (), VACC)
        self.assertTrue(all(np.isnan(e) for e in fit.errs_v))
        np.testing.assert_allclose(fit.coeffs_v, (0.0, 1.0), atol=1e-9)

    def test_out_of_range_exclusions_are_ignored(self):
        set_v, read_raw = make_table(n=10, bad=())
        fit = fit_calibration(set_v, read_raw, 1, excluded=(99, -5),
                              vacc_div=VACC)
        self.assertEqual(fit.excluded, ())
        self.assertEqual(fit.n_used, 10)

    def test_used_mask_and_counts(self):
        set_v, read_raw = make_table(n=10, bad=())
        fit = fit_calibration(set_v, read_raw, 1, excluded=(0, 3),
                              vacc_div=VACC)
        self.assertEqual(fit.n_points, 10)
        self.assertEqual(fit.n_used, 8)
        self.assertEqual(len(fit.residuals_v), 10)   # residuals for ALL points
        self.assertFalse(fit.used_mask[0])
        self.assertFalse(fit.used_mask[3])
        self.assertTrue(fit.used_mask[1])


# ── Flagging ─────────────────────────────────────────────────────────

class FlaggingTests(unittest.TestCase):
    """The default policy is 'change nothing, but say something'. That only
    works if the warning is trustworthy, which means flagging on physical
    impact rather than on statistical oddity -- Gaussian noise alone throws
    a 3-sigma point often enough to cry wolf on perfectly good runs."""

    def test_bad_run_nominates_the_settling_points(self):
        set_v, read_raw = make_table()
        self.assertEqual(suspect_outliers(set_v, read_raw, 1, VACC), (0, 1, 2))

    def test_clean_run_can_still_nominate_a_noise_point(self):
        """Documents *why* nomination alone must not drive the warning."""
        set_v, read_raw = make_table(bad=(), seed=1)
        self.assertEqual(suspect_outliers(set_v, read_raw, 1, VACC), (3,))

    def test_but_that_noise_point_has_no_physical_impact(self):
        set_v, read_raw = make_table(bad=(), seed=1)
        full = fit_calibration(set_v, read_raw, 1, (), VACC)
        trimmed = fit_calibration(set_v, read_raw, 1, (3,), VACC)
        shift = calibration_shift_v(full.coeffs_v, trimmed.coeffs_v,
                                    set_v.min(), set_v.max())
        self.assertLess(shift, FLAG_MIN_IMPACT_V)      # ~0.004 V

    def test_while_the_settling_glitch_has_a_large_one(self):
        set_v, read_raw = make_table()
        full = fit_calibration(set_v, read_raw, 1, (), VACC)
        trimmed = fit_calibration(set_v, read_raw, 1, (0, 1, 2), VACC)
        shift = calibration_shift_v(full.coeffs_v, trimmed.coeffs_v,
                                    set_v.min(), set_v.max())
        self.assertGreater(shift, 1.0)                 # volts -> tens of MHz

    def test_shift_of_a_calibration_against_itself_is_zero(self):
        self.assertEqual(
            calibration_shift_v((0.35, 1.0007), (0.35, 1.0007), 0.0, 400.0),
            0.0)


# ── Specs ────────────────────────────────────────────────────────────

class SpecNormalizationTests(unittest.TestCase):

    def test_non_dict_and_bad_mode_become_none(self):
        for junk in (None, "nope", 42, [], {}, {"mode": "wishful"}):
            self.assertIsNone(normalize_spec(junk))

    def test_inert_fit_spec_becomes_none(self):
        """A 'fit' that changes nothing IS the file default -- it must not
        light up an override badge."""
        self.assertIsNone(normalize_spec(
            {"mode": "fit", "order": 1, "reject": "none", "excluded": []}))

    def test_fit_spec_with_exclusions_survives(self):
        spec = normalize_spec({"mode": "fit", "excluded": [2, 0, 0]})
        self.assertEqual(spec["excluded"], [0, 2])   # deduped + sorted

    def test_ignore_intercept_alone_is_a_real_override(self):
        """Forcing p0 = 0 changes the calibration, so a fit spec that sets
        nothing else must still survive normalization (not collapse to the
        file default)."""
        spec = normalize_spec({"mode": "fit", "ignore_intercept": True})
        self.assertIsNotNone(spec)
        self.assertTrue(spec["ignore_intercept"])
        # ...and the explicit-False, otherwise-inert case is still None.
        self.assertIsNone(normalize_spec(
            {"mode": "fit", "ignore_intercept": False}))

    def test_ignore_intercept_changes_the_fingerprint(self):
        base = {"mode": "fit", "excluded": [1]}
        zi = {"mode": "fit", "excluded": [1], "ignore_intercept": True}
        self.assertNotEqual(spec_fingerprint(base), spec_fingerprint(zi))

    def test_order_is_clamped(self):
        self.assertEqual(normalize_spec(
            {"mode": "fit", "order": 99, "excluded": [1]})["order"], 3)
        self.assertEqual(normalize_spec(
            {"mode": "fit", "order": 0, "excluded": [1]})["order"], 1)

    def test_borrow_needs_a_donor(self):
        self.assertIsNone(normalize_spec({"mode": "borrow"}))
        self.assertEqual(
            normalize_spec({"mode": "borrow", "donor": "x.asdf"})["donor"],
            "x.asdf")

    def test_coeffs_are_validated(self):
        self.assertIsNone(normalize_spec({"mode": "coeffs"}))
        self.assertIsNone(normalize_spec({"mode": "coeffs",
                                          "coeffs_v": [1.0]}))     # order 0
        self.assertIsNone(normalize_spec({"mode": "coeffs",
                                          "coeffs_v": [0.0, float("nan")]}))
        spec = normalize_spec({"mode": "coeffs", "coeffs_v": [0.35, 1.0007]})
        self.assertEqual(spec["order"], 1)

    def test_fingerprint_is_stable_and_order_insensitive(self):
        a = {"mode": "fit", "excluded": [1, 0], "order": 1}
        b = {"order": 1, "excluded": [0, 1], "mode": "fit"}
        self.assertEqual(spec_fingerprint(a), spec_fingerprint(b))
        self.assertEqual(spec_fingerprint(None), "")
        self.assertNotEqual(spec_fingerprint(a),
                            spec_fingerprint({"mode": "fit",
                                              "excluded": [0, 1, 2]}))

    def test_describe_spec_reads_like_a_badge(self):
        self.assertEqual(describe_spec(None), "file default")
        self.assertIn("2 points excluded",
                      describe_spec({"mode": "fit", "excluded": [0, 1]}))
        self.assertIn("run_1001",
                      describe_spec({"mode": "borrow",
                                     "donor": "C:/d/run_1001.asdf"}))
        self.assertIn("p₀=0",
                      describe_spec({"mode": "fit", "ignore_intercept": True}))


# ── Resolution against real ASDF files ───────────────────────────────

class ResolutionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import asdf
        cls.tmp = tempfile.mkdtemp(prefix="cal_test_")

        def write(name, set_v, read_raw):
            path = os.path.join(cls.tmp, name)
            asdf.AsdfFile({"CalSet": np.asarray(set_v),
                           "CalReadback": np.asarray(read_raw)}).write_to(path)
            return path

        # 1001 is clean; 1002 has the settling pathology; 1003 has no table.
        cls.good_set, cls.good_read = make_table(bad=(), seed=1)
        cls.bad_set, cls.bad_read = make_table(seed=2)
        cls.p_good = write("run_1001.asdf", cls.good_set, cls.good_read)
        cls.p_bad = write("run_1002.asdf", cls.bad_set, cls.bad_read)
        cls.p_empty = write("run_1003.asdf", [], [])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        clear_points_cache()

    def test_read_cal_points_returns_set_volts_and_raw_readback(self):
        set_v, read_raw = read_cal_points(self.p_bad)
        np.testing.assert_allclose(set_v, self.bad_set)
        np.testing.assert_allclose(read_raw, self.bad_read)
        # Readback is ~1/1000 of set: it is in monitor units, not volts.
        self.assertLess(read_raw.max(), 1.0)
        self.assertGreater(set_v.max(), 100.0)

    def test_no_spec_gives_the_plain_clstools_fit(self):
        res = resolve_calibration(self.p_bad, {})
        expected, _ = clstools_cal(self.bad_set, self.bad_read, 1)
        np.testing.assert_allclose(res.coeffs_raw(VACC), expected, rtol=1e-12)
        self.assertEqual(res.mode, "file")
        self.assertEqual(res.excluded, ())

    def test_missing_table_falls_back_rather_than_failing(self):
        """No override was asked for, so a run with no table must still load."""
        res = resolve_calibration(self.p_empty, {})
        self.assertTrue(res.fallback)
        self.assertEqual(res.mode, "file")

    def test_missing_table_with_an_explicit_override_raises(self):
        """An override the user *did* ask for must never silently degrade."""
        with self.assertRaises(CalibrationError):
            resolve_calibration(
                self.p_empty, {self.p_empty: {"mode": "fit",
                                              "excluded": [0]}})

    def test_fit_spec_excludes_and_refits(self):
        cal = {self.p_bad: {"mode": "fit", "reject": "first_n",
                            "drop_first": 3}}
        res = resolve_calibration(self.p_bad, cal)
        self.assertEqual(res.mode, "fit")
        self.assertEqual(res.excluded, (0, 1, 2))
        self.assertAlmostEqual(res.coeffs_v[1], 1.0007, delta=1e-4)

    def test_rule_and_manual_exclusions_union(self):
        cal = {self.p_bad: {"mode": "fit", "reject": "first_n",
                            "drop_first": 2, "excluded": [7]}}
        res = resolve_calibration(self.p_bad, cal)
        self.assertEqual(res.excluded, (0, 1, 7))

    def test_borrow_takes_the_donors_coefficients(self):
        cal = {self.p_bad: {"mode": "borrow", "donor": self.p_good}}
        borrowed = resolve_calibration(self.p_bad, cal)
        donor = resolve_calibration(self.p_good, cal)
        self.assertEqual(borrowed.mode, "borrow")
        self.assertEqual(borrowed.donor, self.p_good)
        np.testing.assert_allclose(borrowed.coeffs_v, donor.coeffs_v,
                                   rtol=1e-15)

    def test_borrow_honours_the_donors_own_override(self):
        """1002 borrows 1001, and 1001 itself drops its first points."""
        cal = {
            self.p_good: {"mode": "fit", "reject": "first_n",
                          "drop_first": 4},
            self.p_bad: {"mode": "borrow", "donor": self.p_good},
        }
        borrowed = resolve_calibration(self.p_bad, cal)
        donor = resolve_calibration(self.p_good, cal)
        np.testing.assert_allclose(borrowed.coeffs_v, donor.coeffs_v,
                                   rtol=1e-15)
        self.assertEqual(donor.excluded, (0, 1, 2, 3))

    def test_borrow_shows_residuals_against_the_recipients_own_points(self):
        """The diagnostic must be able to reveal a donor that doesn't
        actually describe this run."""
        cal = {self.p_bad: {"mode": "borrow", "donor": self.p_good}}
        res = resolve_calibration(self.p_bad, cal)
        self.assertIsNotNone(res.fit)
        self.assertEqual(res.fit.n_points, len(self.bad_set))
        # 1002's three settling points still stick out under 1001's polynomial.
        self.assertGreater(abs(res.fit.residuals_v[0]), 1.0)

    def test_borrow_does_not_cross_index_the_donors_excluded_points(self):
        """The donor's excluded indices address the DONOR's point table.

        Regression: they used to be copied onto the borrower's result, whose
        ``fit`` is over the *borrower's* points. With a donor that has more
        calibration points than the borrower, ``apply_calibration`` then
        indexed a 12-element array with index 19 and blew up on every load
        path for that run; when the index happened to be in range it silently
        wrote a "dropped points" audit record for points the run never
        dropped.
        """
        import asdf
        short = os.path.join(self.tmp, "run_1010.asdf")   # only 6 points
        s_set, s_read = make_table(n=6, bad=(), seed=5)
        asdf.AsdfFile({"CalSet": s_set,
                       "CalReadback": s_read}).write_to(short)

        cal = {
            # the donor (21 points) drops its last one
            self.p_good: {"mode": "fit", "excluded": [20]},
            short: {"mode": "borrow", "donor": self.p_good},
        }
        res = resolve_calibration(short, cal)

        self.assertEqual(res.excluded, ())            # this run dropped none
        self.assertEqual(res.donor_excluded, (20,))   # recorded, but separate
        self.assertEqual(res.n_points, 6)

        # And applying it must not index a 6-point array with index 20.
        data = _FakeData()
        rep = apply_calibration(data, short, cal).to_report_dict()
        self.assertIsNone(data.Dropped_calibration_points)
        self.assertEqual(rep["excluded"], [])
        self.assertEqual(rep["donor_excluded"], [20])
        self.assertFalse(data.Cal_df["Excluded"].any())

    def test_borrow_from_a_donor_with_no_table_raises(self):
        """An explicit override that cannot be honoured must fail loudly, not
        fall back to this run's own uncorrected calibration."""
        cal = {self.p_bad: {"mode": "borrow", "donor": self.p_empty}}
        with self.assertRaises(CalibrationError) as ctx:
            resolve_calibration(self.p_bad, cal)
        self.assertIn("no readable calibration table", str(ctx.exception))

    def test_chained_borrow_is_rejected(self):
        cal = {
            self.p_bad: {"mode": "borrow", "donor": self.p_good},
            self.p_good: {"mode": "borrow", "donor": self.p_empty},
        }
        with self.assertRaises(CalibrationError) as ctx:
            resolve_calibration(self.p_bad, cal)
        self.assertIn("Chained borrows", str(ctx.exception))

    def test_borrow_from_a_missing_donor_raises(self):
        cal = {self.p_bad: {"mode": "borrow",
                            "donor": os.path.join(self.tmp, "gone.asdf")}}
        with self.assertRaises(CalibrationError) as ctx:
            resolve_calibration(self.p_bad, cal)
        self.assertIn("does not exist", str(ctx.exception))

    def test_manual_coefficients_are_used_verbatim(self):
        cal = {self.p_bad: {"mode": "coeffs", "coeffs_v": [0.35, 1.0007]}}
        res = resolve_calibration(self.p_bad, cal)
        self.assertEqual(res.mode, "coeffs")
        np.testing.assert_allclose(res.coeffs_v, (0.35, 1.0007))
        np.testing.assert_allclose(res.coeffs_raw(VACC),
                                   (0.00035, 0.0010007))

    def test_diagnose_flags_the_bad_run_only(self):
        d_bad = diagnose(self.p_bad, {}, run_number="1002")
        d_good = diagnose(self.p_good, {}, run_number="1001")

        self.assertTrue(d_bad.flagged)
        self.assertEqual(d_bad.suspect, (0, 1, 2))
        self.assertGreater(d_bad.impact_v, 1.0)     # volts -> tens of MHz

        # The clean run nominates a noise point, but dropping it would move
        # the voltage by ~0.004 V (~0.07 MHz), so it must NOT be flagged.
        self.assertFalse(d_good.flagged)
        self.assertLess(d_good.impact_v, FLAG_MIN_IMPACT_V)

    def test_diagnose_stops_flagging_once_an_override_exists(self):
        cal = {self.p_bad: {"mode": "fit", "reject": "first_n",
                            "drop_first": 3}}
        d = diagnose(self.p_bad, cal, run_number="1002")
        self.assertTrue(d.has_override)
        self.assertFalse(d.flagged)          # handled: stop nagging

    def test_diagnose_never_raises_on_a_broken_spec(self):
        cal = {self.p_bad: {"mode": "borrow", "donor": "/nope/gone.asdf"}}
        d = diagnose(self.p_bad, cal)
        self.assertTrue(d.error)

    def test_an_acknowledged_run_stops_being_flagged(self):
        """The user read the warning and chose to leave the run alone. Keeping
        it blinking after that just teaches them to ignore it."""
        d = diagnose(self.p_bad, {}, run_number="1002", acknowledged=True)
        self.assertTrue(d.suspect)          # the outliers are still there
        self.assertGreater(d.impact_v, 1.0)  # and still cost real MHz
        self.assertTrue(d.acknowledged)
        self.assertFalse(d.flagged)          # ...but we have said our piece

    def test_acknowledging_does_not_change_the_calibration(self):
        """Dismissing a warning is not the same as accepting a fix -- the run
        must still fit with every point included."""
        plain = resolve_calibration(self.p_bad, {})
        d = diagnose(self.p_bad, {}, acknowledged=True)
        self.assertFalse(d.has_override)
        self.assertEqual(plain.excluded, ())


# ── Applying to a data object ────────────────────────────────────────

class ApplyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import asdf
        cls.tmp = tempfile.mkdtemp(prefix="cal_apply_")
        cls.set_v, cls.read_raw = make_table(seed=3)
        cls.path = os.path.join(cls.tmp, "run_2002.asdf")
        asdf.AsdfFile({"CalSet": cls.set_v,
                       "CalReadback": cls.read_raw}).write_to(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        clear_points_cache()

    def test_apply_writes_cal_in_the_raw_convention(self):
        data = _FakeData()
        res = apply_calibration(data, self.path, {})
        expected, _ = clstools_cal(self.set_v, self.read_raw, 1)
        np.testing.assert_allclose(data.Cal, expected, rtol=1e-12)
        self.assertEqual(data.Cal_order, 1)
        self.assertIs(data.CalibrationInfo, res)

    def test_apply_writes_std_errors_not_variances(self):
        data = _FakeData()
        apply_calibration(data, self.path, {})
        _, variances = clstools_cal(self.set_v, self.read_raw, 1)
        np.testing.assert_allclose(data.Cal_err, np.sqrt(np.abs(variances)),
                                   rtol=1e-10)

    def test_apply_records_the_dropped_points_in_volts(self):
        cal = {self.path: {"mode": "fit", "reject": "first_n",
                           "drop_first": 3}}
        data = _FakeData()
        apply_calibration(data, self.path, cal)
        np.testing.assert_allclose(data.Dropped_calibration_points,
                                   self.set_v[:3])

    def test_apply_keeps_cal_df_column_convention(self):
        """merge.py reads Cal_df['Set'] / ['Read'] -- Set stays in volts and
        Read stays in monitor units, or the merge provenance silently
        changes units."""
        data = _FakeData()
        apply_calibration(data, self.path, {})
        df = data.Cal_df
        np.testing.assert_allclose(df["Set"].to_numpy(), self.set_v)
        np.testing.assert_allclose(df["Read"].to_numpy(), self.read_raw,
                                   rtol=1e-12)
        self.assertIn("Residual_V", df.columns)
        self.assertIn("Excluded", df.columns)
        self.assertFalse(df["Excluded"].any())

    def test_apply_marks_excluded_rows(self):
        cal = {self.path: {"mode": "fit", "excluded": [0, 1]}}
        data = _FakeData()
        apply_calibration(data, self.path, cal)
        excl = data.Cal_df["Excluded"].to_numpy()
        self.assertTrue(excl[0] and excl[1])
        self.assertFalse(excl[2:].any())

    def test_apply_honours_a_non_default_vaccdiv(self):
        data = _FakeData(vacc_div=500.0)
        res = apply_calibration(data, self.path, {})
        np.testing.assert_allclose(
            np.array(data.Cal) * 500.0, res.coeffs_v, rtol=1e-12)

    def test_report_dict_is_picklable_and_complete(self):
        import pickle
        cal = {self.path: {"mode": "fit", "reject": "first_n",
                           "drop_first": 3, "note": "HV settling"}}
        data = _FakeData()
        rep = apply_calibration(data, self.path, cal).to_report_dict()
        self.assertEqual(rep["mode"], "fit")
        self.assertEqual(rep["excluded"], [0, 1, 2])
        self.assertEqual(rep["n_excluded"], 3)
        self.assertEqual(rep["note"], "HV settling")
        self.assertIn("sigma_v", rep)
        pickle.loads(pickle.dumps(rep))     # crosses the subprocess boundary


if __name__ == "__main__":
    unittest.main(verbosity=2)
