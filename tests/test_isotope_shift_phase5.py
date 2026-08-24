"""Unit tests for Phase 5 isotope-shift uncertainty propagation.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Exercises Birge inflation of weighted averages, GP-propagated drift
sigma, GP cross-covariance between isotopes, the mixed Auto+Manual
correction sigma, and the extended compute_all_shifts that combines
them into per-isotope shift uncertainties.

Depends on: cls_estimations.isotope_shift (weighted_average_with_birge,
gp_propagated_sigma, gp_cross_covariance, mixed_mode_propagated_sigma,
compute_all_shifts).
"""

import math
import unittest

import numpy as np

from cls_estimations.isotope_shift import (
    weighted_average, weighted_average_with_birge,
    gp_propagated_sigma, gp_cross_covariance,
    compute_all_shifts,
)


class WeightedAverageWithBirgeTests(unittest.TestCase):

    def test_consistent_inputs_birge_one(self):
        # Inputs with a true mean of 100 and σ=1 each, no extra scatter
        rng = np.random.default_rng(0)
        truth = 100.0
        n = 50
        sigmas = np.full(n, 1.0)
        vals = truth + rng.normal(0, 1.0, size=n)
        mean, sf, ss, birge = weighted_average_with_birge(vals, sigmas)
        # Mean recovered to within ~σ/√N
        self.assertAlmostEqual(mean, truth, delta=3 * 1.0 / math.sqrt(n))
        # σ_fit = 1/√N for unit weights
        self.assertAlmostEqual(sf, 1.0 / math.sqrt(n), delta=1e-9)
        # χ²_red ≈ 1, so birge ≈ 1, σ_scatter ≈ 0 (allow ~10% Monte
        # Carlo wobble around 1.0 -> σ_scatter < 0.5 σ_fit)
        self.assertLess(ss, 0.5 * sf)
        self.assertLess(birge, 1.5)

    def test_inflated_inputs_produce_scatter(self):
        # Inputs that disagree more than their stated σ implies → χ²_red > 1
        # → birge > 1 → σ_scatter > 0
        vals = [100.0, 102.0, 98.0, 103.0, 97.0]   # spread ~5
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1]         # claims spread 0.1
        mean, sf, ss, birge = weighted_average_with_birge(vals, sigmas)
        # Inverse-variance mean
        self.assertAlmostEqual(mean, 100.0, delta=1e-9)
        # σ_fit before inflation
        self.assertAlmostEqual(sf, 0.1 / math.sqrt(5), delta=1e-9)
        # birge much larger than 1 because of disagreement
        self.assertGreater(birge, 5.0)
        # σ_scatter accounts for the inflation: total = sf * birge
        total_inflated = sf * birge
        np.testing.assert_allclose(
            ss ** 2 + sf ** 2, total_inflated ** 2, atol=1e-12)

    def test_single_input_no_inflation(self):
        mean, sf, ss, birge = weighted_average_with_birge([5.0], [0.5])
        self.assertEqual(mean, 5.0)
        self.assertEqual(sf, 0.5)
        self.assertEqual(ss, 0.0)
        self.assertEqual(birge, 1.0)

    def test_empty_input(self):
        mean, sf, ss, birge = weighted_average_with_birge([], [])
        self.assertTrue(math.isnan(mean))


class _FakeCorrector:
    """Minimal stand-in: a Gaussian-like posterior covariance that
    decays with |Δt| and saturates at a known amplitude, so the math
    in the propagation helpers is exercised without spinning up PyMC.
    """

    def __init__(self, amplitude=1.0, length=10.0):
        self.amplitude = amplitude
        self.length = length

    def cov(self, t1, t2):
        t1 = np.atleast_1d(np.asarray(t1, dtype=float))
        t2 = np.atleast_1d(np.asarray(t2, dtype=float))
        d = t1[:, None] - t2[None, :]
        return self.amplitude ** 2 * np.exp(-0.5 * (d / self.length) ** 2)


class GPPropagatedSigmaTests(unittest.TestCase):

    def test_single_run_matches_diagonal(self):
        rc = _FakeCorrector(amplitude=2.0, length=5.0)
        s = gp_propagated_sigma(rc, [10.0], [1.0])
        # σ² = (1/W²) · w² · K(t,t) = (1/1) · 1 · 4 = 4 → σ = 2
        self.assertAlmostEqual(s, 2.0, delta=1e-9)

    def test_correlated_runs_reduce_to_amplitude(self):
        # Many runs at the same timestamp -> K is rank-1 with
        # k(t,t) = amplitude² everywhere; σ → amplitude regardless of
        # how many runs you take. Averaging correlated noise doesn't
        # help.
        rc = _FakeCorrector(amplitude=3.0, length=5.0)
        t = [10.0] * 10
        w = [1.0] * 10
        s = gp_propagated_sigma(rc, t, w)
        self.assertAlmostEqual(s, 3.0, delta=1e-9)

    def test_well_separated_runs_average_down(self):
        # Many runs spaced FAR apart compared to the kernel length
        # behave as independent samples and σ scales with 1/√N.
        rc = _FakeCorrector(amplitude=2.0, length=1.0)  # short range
        n = 25
        t = list(range(0, n * 100, 100))     # every 100 hours, ℓ=1
        w = [1.0] * n
        s = gp_propagated_sigma(rc, t, w)
        # σ → amplitude/√N for uncorrelated samples
        self.assertAlmostEqual(s, 2.0 / math.sqrt(n), delta=1e-3)

    def test_zero_weights_returns_zero(self):
        rc = _FakeCorrector()
        s = gp_propagated_sigma(rc, [10.0, 20.0], [0.0, 0.0])
        self.assertEqual(s, 0.0)

    def test_total_weight_normalization(self):
        """Regression: when some runs lack timestamps and aren't
        GP-corrected, σ_correction must normalize by W_total of the
        full centroid average -- not W_subset of the corrected runs.
        Otherwise the result is overestimated by (W_total/W_subset)².
        """
        rc = _FakeCorrector(amplitude=2.0, length=5.0)
        # Two corrected runs and two un-corrected runs, all unit weight.
        # W_subset = 2; W_total = 4. The correct σ should be HALF of
        # what the subset-only call gives (since 2/4 ratio in W means
        # 1/4 in W² in the variance).
        sigma_subset = gp_propagated_sigma(rc, [10.0, 20.0], [1.0, 1.0])
        sigma_total = gp_propagated_sigma(
            rc, [10.0, 20.0], [1.0, 1.0], total_weight=4.0)
        np.testing.assert_allclose(sigma_total, sigma_subset / 2.0)


class GPCrossCovarianceTests(unittest.TestCase):

    def test_same_timestamps_cov_equals_amplitude_squared(self):
        # Both isotopes measured at the SAME single time. Cov should
        # equal the diagonal amplitude² (full correlation).
        rc = _FakeCorrector(amplitude=2.5, length=5.0)
        cov = gp_cross_covariance(rc, [10.0], [1.0], [10.0], [1.0])
        self.assertAlmostEqual(cov, 6.25, delta=1e-9)

    def test_well_separated_cov_approaches_zero(self):
        rc = _FakeCorrector(amplitude=2.0, length=1.0)
        cov = gp_cross_covariance(
            rc, [0.0], [1.0], [10000.0], [1.0])
        self.assertAlmostEqual(cov, 0.0, delta=1e-9)

    def test_empty_inputs(self):
        rc = _FakeCorrector()
        self.assertEqual(
            gp_cross_covariance(rc, [], [], [10.0], [1.0]), 0.0)
        self.assertEqual(
            gp_cross_covariance(rc, [10.0], [1.0], [], []), 0.0)


class ComputeAllShiftsPhase5Tests(unittest.TestCase):

    def test_legacy_keys_still_work(self):
        """Pre-Phase-5 callers pass sigma_stat/sigma_sys; the function
        must continue to produce the legacy columns and treat the
        systematic as quadrature-only."""
        entries = {
            "ref": {"centroid": 0.0, "sigma_stat": 1.0, "sigma_sys": 2.0,
                    "A": 100},
            "iso": {"centroid": 100.0, "sigma_stat": 1.5, "sigma_sys": 2.5,
                    "A": 102},
        }
        rows = compute_all_shifts(entries, "ref")
        ref_row = next(r for r in rows if r["is_reference"])
        iso_row = next(r for r in rows if not r["is_reference"])
        # Legacy keys mapped: sigma_fit = sigma_stat, sigma_voltage =
        # sigma_sys, sigma_scatter = sigma_correction = 0
        self.assertAlmostEqual(iso_row["sigma_fit"], 1.5)
        self.assertAlmostEqual(iso_row["sigma_voltage"], 2.5)
        self.assertEqual(iso_row["sigma_scatter"], 0.0)
        self.assertEqual(iso_row["sigma_correction"], 0.0)
        # Legacy sigma_stat retains its pre-Phase-5 meaning of pure
        # inverse-variance (= sigma_fit), so old log/CSV consumers
        # see the same number they used to.
        self.assertAlmostEqual(iso_row["sigma_stat"], 1.5)
        # σ_dnu_stat is pure stat from each isotope (no scatter).
        self.assertAlmostEqual(
            iso_row["sigma_delta_nu_stat"],
            math.sqrt(1.0 ** 2 + 1.5 ** 2))
        # σ_dnu = √(σ_a²+σ_ref²) for both stat and sys without cov
        np.testing.assert_allclose(
            iso_row["sigma_delta_nu"],
            math.sqrt(1.0 ** 2 + 1.5 ** 2 + 2.0 ** 2 + 2.5 ** 2),
            atol=1e-9)

    def test_birge_scatter_separated_from_dnu_stat(self):
        """σ_δν(stat) is pure inverse-variance; Birge scatter lives
        in its own column σ_δν(scatter)."""
        entries = {
            "ref": {"centroid": 0.0,
                    "sigma_fit": 0.1, "sigma_scatter": 1.0,
                    "sigma_correction": 0.0, "sigma_voltage": 0.0,
                    "A": 100},
            "iso": {"centroid": 50.0,
                    "sigma_fit": 0.2, "sigma_scatter": 0.5,
                    "sigma_correction": 0.0, "sigma_voltage": 0.0,
                    "A": 102},
        }
        rows = compute_all_shifts(entries, "ref")
        iso_row = next(r for r in rows if not r["is_reference"])
        # Pure stat: only sigma_fit values
        self.assertAlmostEqual(
            iso_row["sigma_delta_nu_stat"],
            math.sqrt(0.1 ** 2 + 0.2 ** 2))
        # Birge contribution in its own column
        self.assertAlmostEqual(
            iso_row["sigma_delta_nu_scatter"],
            math.sqrt(1.0 ** 2 + 0.5 ** 2))
        # Total still sums everything in quadrature
        self.assertAlmostEqual(
            iso_row["sigma_delta_nu"],
            math.sqrt(0.1 ** 2 + 0.2 ** 2 + 1.0 ** 2 + 0.5 ** 2))

    def test_correlated_correction_reduces_shift_uncertainty(self):
        """Two isotopes with the SAME σ_correction but high correlation
        should produce a tighter shift than the no-cov quadrature sum."""
        entries = {
            "ref": {"centroid": 0.0, "sigma_fit": 0.0, "sigma_scatter": 0.0,
                    "sigma_correction": 1.0, "sigma_voltage": 0.0,
                    "A": 100},
            "iso": {"centroid": 50.0, "sigma_fit": 0.0,
                    "sigma_scatter": 0.0, "sigma_correction": 1.0,
                    "sigma_voltage": 0.0, "A": 102},
        }
        # No cross-cov -> σ_dnu_corr = √(1+1) = √2
        no_cov = compute_all_shifts(entries, "ref")
        no_cov_row = next(r for r in no_cov if not r["is_reference"])
        np.testing.assert_allclose(
            no_cov_row["sigma_delta_nu_correction"],
            math.sqrt(2.0), atol=1e-9)
        # Strong positive cov ~ amplitude² -> systematic cancels
        full_cov = compute_all_shifts(
            entries, "ref",
            gp_cross_cov={("iso", "ref"): 1.0})
        full_row = next(r for r in full_cov if not r["is_reference"])
        # Var = 1 + 1 - 2 = 0
        self.assertLess(full_row["sigma_delta_nu_correction"], 1e-9)

    def test_uncorrelated_correction_quadrature(self):
        entries = {
            "ref": {"centroid": 0.0, "sigma_fit": 0.0, "sigma_scatter": 0.0,
                    "sigma_correction": 1.0, "sigma_voltage": 0.0,
                    "A": 100},
            "iso": {"centroid": 50.0, "sigma_fit": 0.0,
                    "sigma_scatter": 0.0, "sigma_correction": 1.0,
                    "sigma_voltage": 0.0, "A": 102},
        }
        # Cov = 0 (well-separated timestamps)
        rows = compute_all_shifts(
            entries, "ref", gp_cross_cov={("iso", "ref"): 0.0})
        iso_row = next(r for r in rows if not r["is_reference"])
        np.testing.assert_allclose(
            iso_row["sigma_delta_nu_correction"],
            math.sqrt(2.0), atol=1e-9)

    def test_voltage_systematic_independent_per_isotope(self):
        # σ_voltage adds in quadrature regardless of cov on σ_correction
        entries = {
            "ref": {"centroid": 0.0, "sigma_fit": 0.0, "sigma_scatter": 0.0,
                    "sigma_correction": 0.0, "sigma_voltage": 1.0,
                    "A": 100},
            "iso": {"centroid": 50.0, "sigma_fit": 0.0,
                    "sigma_scatter": 0.0, "sigma_correction": 0.0,
                    "sigma_voltage": 1.0, "A": 102},
        }
        rows = compute_all_shifts(
            entries, "ref", gp_cross_cov={("iso", "ref"): 100.0})
        iso_row = next(r for r in rows if not r["is_reference"])
        # σ_dnu_sys = √(σ_volt_a² + σ_volt_ref²) (cov doesn't affect it)
        np.testing.assert_allclose(
            iso_row["sigma_delta_nu_sys"],
            math.sqrt(2.0), atol=1e-9)


class MixedModePropagationTests(unittest.TestCase):
    """When an isotope has BOTH Auto and Manual rows, σ_correction
    must combine the GP posterior cov over the Auto rows with the
    diagonal user σ over the Manual rows. Manual rows must not pull
    from corrector.cov."""

    def test_auto_only_matches_existing_helper(self):
        """A pure-Auto call should reproduce the existing
        gp_propagated_sigma result exactly."""
        rc = _FakeCorrector(amplitude=2.0, length=5.0)
        from cls_estimations.isotope_shift import (
            mixed_mode_propagated_sigma,
        )
        s_existing = gp_propagated_sigma(
            rc, [10.0, 20.0], [1.0, 1.0], total_weight=2.0)
        s_mixed = mixed_mode_propagated_sigma(
            rc, auto_t=[10.0, 20.0], auto_w=[1.0, 1.0],
            manual_sigmas=(), manual_w=(), total_weight=2.0)
        self.assertAlmostEqual(s_existing, s_mixed, places=12)

    def test_manual_only_uses_diagonal_sigma(self):
        """A pure-Manual call ignores the corrector entirely;
        σ_correction = √(Σ wᵢ² σᵢ²) / W."""
        from cls_estimations.isotope_shift import (
            mixed_mode_propagated_sigma,
        )
        # Two manual rows at σ=1.0 each, equal weight, W_total=2:
        # Var = (1²·1² + 1²·1²) / 4 = 0.5; σ = √0.5
        rc = _FakeCorrector(amplitude=999.0, length=1.0)  # unused
        s = mixed_mode_propagated_sigma(
            rc, auto_t=(), auto_w=(),
            manual_sigmas=[1.0, 1.0], manual_w=[1.0, 1.0],
            total_weight=2.0)
        self.assertAlmostEqual(s, math.sqrt(0.5), places=12)

    def test_mixed_combines_auto_gp_and_manual_diagonal(self):
        """Mixed Auto+Manual: variances add."""
        from cls_estimations.isotope_shift import (
            mixed_mode_propagated_sigma,
        )
        rc = _FakeCorrector(amplitude=2.0, length=5.0)
        # 1 Auto row at t=10 with weight 1; 1 Manual row σ=1 weight 1;
        # W_total = 2.
        s_auto_only = mixed_mode_propagated_sigma(
            rc, auto_t=[10.0], auto_w=[1.0],
            manual_sigmas=(), manual_w=(),
            total_weight=2.0)
        s_manual_only = mixed_mode_propagated_sigma(
            rc, auto_t=(), auto_w=(),
            manual_sigmas=[1.0], manual_w=[1.0],
            total_weight=2.0)
        s_mixed = mixed_mode_propagated_sigma(
            rc, auto_t=[10.0], auto_w=[1.0],
            manual_sigmas=[1.0], manual_w=[1.0],
            total_weight=2.0)
        # Variances add
        self.assertAlmostEqual(
            s_mixed,
            math.sqrt(s_auto_only ** 2 + s_manual_only ** 2),
            places=12)

    def test_empty_returns_zero(self):
        from cls_estimations.isotope_shift import (
            mixed_mode_propagated_sigma,
        )
        rc = _FakeCorrector()
        self.assertEqual(
            mixed_mode_propagated_sigma(
                rc, auto_t=(), auto_w=(),
                manual_sigmas=(), manual_w=(),
                total_weight=2.0),
            0.0)

    def test_zero_total_weight_returns_zero(self):
        from cls_estimations.isotope_shift import (
            mixed_mode_propagated_sigma,
        )
        rc = _FakeCorrector()
        self.assertEqual(
            mixed_mode_propagated_sigma(
                rc, auto_t=[10.0], auto_w=[1.0],
                manual_sigmas=(), manual_w=(),
                total_weight=0.0),
            0.0)

    def test_manual_only_works_without_corrector(self):
        """A Manual-only call doesn't need the corrector at all --
        σ_correction is just diagonal user σ². This is what the IS
        tab needs when reopening a project with Manual correction
        metadata but no GP corrector loaded."""
        from cls_estimations.isotope_shift import (
            mixed_mode_propagated_sigma,
        )
        s = mixed_mode_propagated_sigma(
            None,
            auto_t=(), auto_w=(),
            manual_sigmas=[1.0, 1.0], manual_w=[1.0, 1.0],
            total_weight=2.0)
        # Var = (1²·1² + 1²·1²)/4 = 0.5; σ = √0.5
        self.assertAlmostEqual(s, math.sqrt(0.5), places=12)

    def test_auto_without_corrector_raises(self):
        """If Auto rows are present, a corrector is required. There
        is no honest fallback for evaluating the GP posterior cov
        without the GP itself."""
        from cls_estimations.isotope_shift import (
            mixed_mode_propagated_sigma,
        )
        with self.assertRaises(ValueError):
            mixed_mode_propagated_sigma(
                None,
                auto_t=[10.0], auto_w=[1.0],
                manual_sigmas=(), manual_w=(),
                total_weight=1.0)


if __name__ == "__main__":
    unittest.main()
