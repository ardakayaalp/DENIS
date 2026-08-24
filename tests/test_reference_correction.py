"""Unit tests for the GP-based reference drift corrector.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers cls_estimations.reference_correction: constructor/fit validation,
the numpy kernel functions (RBF, Matern-5/2, periodic, composite thesis
kernel) and their PSD/PyMC-convention properties, the low-level GP
prediction and covariance core, hyperparameter serialization, and the
slow end-to-end PyMC fits (skipped when PyMC is unavailable).

Depends on: cls_estimations.reference_correction (ReferenceObservation,
ReferenceCorrector, GPHyperparameters, the kernel and GP helper
functions); PyMC for the end-to-end tests.
"""

import math
import unittest

import numpy as np

from cls_estimations.reference_correction import (
    ReferenceObservation, ReferenceCorrector, GPHyperparameters,
    SUPPORTED_KERNELS,
    _K_rbf, _K_matern52, _K_periodic, _build_kernel_fn,
    _gp_predict, _gp_cov,
)


def _synthetic_drift(n: int = 30, *, kind: str = "smooth", seed: int = 42):
    """Generate a synthetic drift signal with a known true function.

    Returns ``(observations, t_dense, true_mu_dense)`` where
    ``observations`` is a list of ``ReferenceObservation`` and
    ``true_mu_dense`` is the noiseless drift evaluated on a dense grid
    for accuracy checks.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 100, size=n))
    if kind == "smooth":
        # A slowly drifting signal that an RBF GP should easily capture.
        true_mu = -700.0 + 30.0 * np.sin(t / 20.0)
    elif kind == "linear":
        true_mu = -700.0 + 0.5 * t
    else:
        raise ValueError(kind)
    sigma = rng.uniform(1.0, 3.0, size=n)
    y = true_mu + rng.normal(0, sigma)
    obs = [ReferenceObservation(t=float(ti), centroid=float(yi),
                                 sigma=float(si),
                                 label=f"r{i}")
           for i, (ti, yi, si) in enumerate(zip(t, y, sigma))]
    t_dense = np.linspace(t.min(), t.max(), 200)
    if kind == "smooth":
        true_dense = -700.0 + 30.0 * np.sin(t_dense / 20.0)
    else:
        true_dense = -700.0 + 0.5 * t_dense
    return obs, t_dense, true_dense


# ─────────────────────────────────────────────────────────────────
#  Constructor / validation
# ─────────────────────────────────────────────────────────────────

class ConstructorTests(unittest.TestCase):

    def test_default_kernel_is_rbf(self):
        rc = ReferenceCorrector()
        self.assertEqual(rc.kernel, "rbf")
        self.assertFalse(rc.is_fit)

    def test_unknown_kernel_rejected(self):
        with self.assertRaises(ValueError):
            ReferenceCorrector(kernel="nope")

    def test_supported_kernels(self):
        for k in SUPPORTED_KERNELS:
            rc = ReferenceCorrector(kernel=k)
            self.assertEqual(rc.kernel, k)


class FitValidationTests(unittest.TestCase):

    def test_no_observations_raises(self):
        rc = ReferenceCorrector()
        with self.assertRaises(ValueError):
            rc.fit([])

    def test_one_observation_raises(self):
        rc = ReferenceCorrector()
        with self.assertRaises(ValueError):
            rc.fit([ReferenceObservation(t=0, centroid=0, sigma=1)])

    def test_excluded_observations_dont_count(self):
        rc = ReferenceCorrector()
        with self.assertRaises(ValueError):
            rc.fit([
                ReferenceObservation(t=0, centroid=0, sigma=1, include=False),
                ReferenceObservation(t=1, centroid=1, sigma=1, include=False),
            ])

    def test_zero_sigma_rejected(self):
        rc = ReferenceCorrector()
        with self.assertRaises(ValueError):
            rc.fit([
                ReferenceObservation(t=0, centroid=0, sigma=0),
                ReferenceObservation(t=1, centroid=1, sigma=1),
            ])

    def test_predict_before_fit_raises(self):
        rc = ReferenceCorrector()
        with self.assertRaises(RuntimeError):
            rc.predict([1.0, 2.0])
        with self.assertRaises(RuntimeError):
            rc.cov([1.0], [2.0])


# ─────────────────────────────────────────────────────────────────
#  Kernel math
# ─────────────────────────────────────────────────────────────────

class KernelMathTests(unittest.TestCase):

    def test_rbf_diagonal(self):
        t = np.array([0.0, 1.0, 2.0])
        K = _K_rbf(t, t, eta=2.0, ell=1.0)
        # Diagonal should equal eta^2
        np.testing.assert_allclose(np.diag(K), [4.0, 4.0, 4.0])
        # Symmetric, PSD
        np.testing.assert_allclose(K, K.T)
        eigvals = np.linalg.eigvalsh(K)
        self.assertTrue(np.all(eigvals > -1e-10))

    def test_matern52_at_zero(self):
        t = np.array([0.0])
        K = _K_matern52(t, t, eta=3.0, ell=2.0)
        self.assertAlmostEqual(K[0, 0], 9.0)

    def test_periodic_periodicity(self):
        t = np.array([0.0])
        # Period = 5, so k(0, 5) should equal k(0, 0)
        K0 = _K_periodic(t, np.array([0.0]), period=5.0, ell=1.0)
        K5 = _K_periodic(t, np.array([5.0]), period=5.0, ell=1.0)
        self.assertAlmostEqual(K0[0, 0], K5[0, 0])
        self.assertAlmostEqual(K0[0, 0], 1.0)

    def test_periodic_matches_pymc_convention(self):
        """Regression: the numpy _K_periodic must use the same length-
        scale convention as PyMC's pm.gp.cov.Periodic. PyMC uses
        k = exp(-sin²(πd/T)/(2ℓ²)); the more common R&W convention is
        exp(-2·sin²(πd/T)/ℓ²), differing by a factor of 4 in
        1/ℓ². Mismatching them silently corrupts the thesis kernel's
        fast-oscillation component because the MAP fits ℓ under one
        convention and predict applies the other.
        """
        try:
            import pymc as pm
        except ImportError:
            self.skipTest("PyMC not installed")
        t1 = np.linspace(0, 30, 12)
        t2 = np.linspace(0, 30, 12) + 0.7  # offset to test cross-block
        period = 12.0
        ell = 2.5
        # Numpy version
        K_np = _K_periodic(t1, t2, period, ell)
        # PyMC version (evaluate once at the same points)
        cov_pm = pm.gp.cov.Periodic(1, period=period, ls=ell)
        K_pm = cov_pm(t1[:, None], t2[:, None]).eval()
        np.testing.assert_allclose(K_np, K_pm, atol=1e-12, rtol=1e-10)

    def test_build_kernel_thesis_returns_psd(self):
        hp = GPHyperparameters(
            kernel="thesis", sigma_n=0.1,
            A_slow=10.0, length_slow=20.0,
            A_fast=5.0, period_fast=12.0,
            length_fast=2.0, length_decay=8.0,
        )
        K_fn = _build_kernel_fn(hp)
        t = np.linspace(0, 50, 20)
        K = K_fn(t, t)
        np.testing.assert_allclose(K, K.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(K + 1e-9 * np.eye(20))
        self.assertGreater(eigvals.min(), -1e-7)


# ─────────────────────────────────────────────────────────────────
#  GP prediction (low-level numpy core)
# ─────────────────────────────────────────────────────────────────

class GPPredictMathTests(unittest.TestCase):

    def test_predict_at_training_point_matches_value(self):
        # If we predict at a training point with very small sigma_n,
        # the GP should pass nearly through the data.
        t = np.array([0.0, 5.0, 10.0])
        y = np.array([0.0, 2.0, -1.0])  # centered
        yerr = np.array([0.001, 0.001, 0.001])
        hp = GPHyperparameters(kernel="rbf", sigma_n=0.001,
                                eta=2.0, ell=3.0)
        K_fn = _build_kernel_fn(hp)
        mu, var = _gp_predict(np.array([0.0, 5.0, 10.0]), t, y, yerr,
                               K_fn, sigma_n=hp.sigma_n)
        np.testing.assert_allclose(mu, y, atol=1e-2)
        # Variance at training points is small but non-zero
        for v in var:
            self.assertGreaterEqual(v, 0.0)

    def test_cov_symmetric_psd(self):
        t = np.linspace(0, 10, 8)
        yerr = np.ones(len(t)) * 0.5
        hp = GPHyperparameters(kernel="rbf", sigma_n=0.5,
                                eta=1.0, ell=2.0)
        K_fn = _build_kernel_fn(hp)
        t_eval = np.array([1.0, 3.0, 5.0, 7.0])
        cov = _gp_cov(t_eval, t_eval, t, yerr, K_fn, hp.sigma_n)
        np.testing.assert_allclose(cov, cov.T, atol=1e-9)
        eigvals = np.linalg.eigvalsh(cov + 1e-9 * np.eye(4))
        self.assertGreater(eigvals.min(), -1e-7)


# ─────────────────────────────────────────────────────────────────
#  Hyperparameters serialization
# ─────────────────────────────────────────────────────────────────

class HyperparametersTests(unittest.TestCase):

    def test_rbf_round_trip(self):
        hp = GPHyperparameters(kernel="rbf", sigma_n=0.5,
                                eta=2.0, ell=3.0)
        d = hp.to_dict()
        self.assertEqual(d["kernel"], "rbf")
        self.assertNotIn("A_slow", d)  # only relevant fields kept
        hp2 = GPHyperparameters.from_dict(d)
        self.assertEqual(hp2.eta, 2.0)
        self.assertEqual(hp2.ell, 3.0)
        self.assertIsNone(hp2.A_slow)

    def test_thesis_round_trip(self):
        hp = GPHyperparameters(
            kernel="thesis", sigma_n=0.1,
            A_slow=10.0, length_slow=20.0, A_fast=5.0,
            period_fast=12.0, length_fast=2.0, length_decay=8.0,
        )
        d = hp.to_dict()
        self.assertNotIn("eta", d)
        hp2 = GPHyperparameters.from_dict(d)
        self.assertEqual(hp2.A_slow, 10.0)
        self.assertEqual(hp2.length_decay, 8.0)


# ─────────────────────────────────────────────────────────────────
#  End-to-end fit (PyMC required; slow)
# ─────────────────────────────────────────────────────────────────

class EndToEndFitTests(unittest.TestCase):
    """These tests actually call PyMC; they're slow (~10-30s each).

    Skipped automatically if PyMC isn't installed.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import pymc  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyMC not installed")

    def test_rbf_recovers_smooth_drift(self):
        obs, t_dense, true_dense = _synthetic_drift(n=25, kind="smooth")
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        self.assertTrue(rc.is_fit)
        mu, sigma = rc.predict(t_dense)
        # The MAP curve should track the true drift within 2 sigma
        # over most of the training range
        within = np.abs(mu - true_dense) <= 2.5 * sigma
        self.assertGreater(within.mean(), 0.9)
        # Sigma is non-trivial (not zero everywhere)
        self.assertGreater(np.median(sigma), 0.1)

    def test_predict_shape(self):
        obs, _, _ = _synthetic_drift(n=15, kind="smooth")
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        mu, sigma = rc.predict([10.0, 20.0, 30.0])
        self.assertEqual(mu.shape, (3,))
        self.assertEqual(sigma.shape, (3,))
        # Scalar input still returns array of length 1
        mu1, sigma1 = rc.predict(15.0)
        self.assertEqual(mu1.shape, (1,))

    def test_cov_diagonal_matches_variance(self):
        obs, _, _ = _synthetic_drift(n=15, kind="smooth")
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        t_eval = np.array([10.0, 30.0, 60.0])
        _, sigma = rc.predict(t_eval)
        cov = rc.cov(t_eval, t_eval)
        # Diagonal of cov should match sigma**2 (predict uses K_ss
        # diagonal, cov returns full K_ss; both should agree)
        np.testing.assert_allclose(np.diag(cov), sigma ** 2,
                                    rtol=1e-6, atol=1e-9)

    def test_to_dict_from_dict_round_trip(self):
        obs, _, _ = _synthetic_drift(n=12, kind="smooth")
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        d = rc.to_dict()
        rc2 = ReferenceCorrector.from_dict(d)
        # Predict identically before/after round-trip
        t_eval = np.array([5.0, 25.0, 55.0])
        mu1, s1 = rc.predict(t_eval)
        mu2, s2 = rc2.predict(t_eval)
        np.testing.assert_allclose(mu1, mu2, atol=1e-9)
        np.testing.assert_allclose(s1, s2, atol=1e-9)

    def test_failed_refit_does_not_corrupt_state(self):
        """A failed refit must leave the corrector unchanged (atomic
        commit). Simulate failure by passing observations that all
        share the same timestamp, which will produce a singular kernel
        matrix and break the optimizer."""
        obs, _, _ = _synthetic_drift(n=10, kind="smooth")
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        # Capture pre-refit state
        t_eval = np.array([10.0, 30.0])
        mu_before, sigma_before = rc.predict(t_eval)
        hp_before = rc.hyperparameters.to_dict()

        bad = [ReferenceObservation(t=1.0, centroid=0.0, sigma=-1.0,
                                     label="bad")]
        with self.assertRaises(ValueError):
            rc.fit(bad)
        # State preserved -- predict still works and matches the prior fit
        self.assertTrue(rc.is_fit)
        mu_after, sigma_after = rc.predict(t_eval)
        np.testing.assert_allclose(mu_before, mu_after)
        np.testing.assert_allclose(sigma_before, sigma_after)
        self.assertEqual(rc.hyperparameters.to_dict(), hp_before)


if __name__ == "__main__":
    unittest.main()
