"""Reference centroid drift correction via Gaussian Process regression.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Implements the GP-based drift correction from van den Borne (KU Leuven
PhD, 2025), section 3.3.1 + Appendix B.2. The corrector is trained on
``(t_i, c_i, σ_i)`` triples extracted from reference-isotope scans and
predicts the drift mean μ(t) and uncertainty σ(t) at arbitrary sample
timestamps, supplying the centroid correction and its systematic
uncertainty to the analysis pipeline.

Three kernels are exposed:

- ``rbf``: ``η² ExpQuad(ℓ) + WhiteNoise``. Smooth slow drift.
- ``matern``: ``η² Matérn(5/2, ℓ) + WhiteNoise``. Slightly less smooth.
- ``thesis``: ``A_slow² ExpQuad(ℓ_slow)
                + A_fast² Periodic(P, ℓ_fast) · Matérn52(ℓ_decay)
                + WhiteNoise(σ_n)`` — the composite kernel from
  Eq. 3.5 of the thesis.

The fitting backend is PyMC v5 (the thesis used the deprecated PyMC3;
v5 has the same API for our use). PyMC is imported lazily inside
``fit`` so projects load on hosts without it; ``predict``/``cov`` do
not require PyMC because they evaluate the kernel directly in numpy
using the MAP-fitted hyperparameters.

Depends on: standard library and third-party packages only (numpy
always, plus PyMC/arviz/matplotlib imported lazily where needed).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


SUPPORTED_KERNELS: tuple[str, ...] = ("rbf", "matern", "thesis")

# Match PyMC's stabilize() default so the matrix used at predict time
# matches what marginal_likelihood saw during MAP optimization. Smaller
# values (e.g. 1e-9) caused the predictions to disagree with the fit at
# small per-observation sigma.
JITTER = 1e-6


# ──────────────────────────────────────────────────────────────────
#  Data types
# ──────────────────────────────────────────────────────────────────

@dataclass
class ReferenceObservation:
    """One time-stamped reference centroid measurement.

    ``t`` is whatever time unit the user works in (typically hours
    since some reference); the corrector preserves it. ``sigma``
    must be strictly positive — the GP marginal likelihood needs
    finite per-observation noise.
    """
    t: float
    centroid: float
    sigma: float
    label: str = ""
    include: bool = True


@dataclass
class GPHyperparameters:
    """MAP-fitted hyperparameters for one of the supported kernels.

    Only the fields relevant to the chosen kernel are populated; the
    others stay ``None``. ``sigma_n`` is the homoscedastic
    white-noise term added to the diagonal on top of the
    per-observation ``yerr``.
    """
    kernel: str
    sigma_n: float
    eta: float | None = None              # rbf/matern signal stddev
    ell: float | None = None              # rbf/matern length scale
    A_slow: float | None = None           # thesis kernel
    length_slow: float | None = None
    A_fast: float | None = None
    period_fast: float | None = None
    length_fast: float | None = None
    length_decay: float | None = None

    def to_dict(self) -> dict:
        d = {"kernel": self.kernel, "sigma_n": float(self.sigma_n)}
        for k in ("eta", "ell", "A_slow", "length_slow", "A_fast",
                  "period_fast", "length_fast", "length_decay"):
            v = getattr(self, k)
            if v is not None:
                d[k] = float(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GPHyperparameters":
        return cls(
            kernel=d["kernel"],
            sigma_n=float(d["sigma_n"]),
            eta=d.get("eta"),
            ell=d.get("ell"),
            A_slow=d.get("A_slow"),
            length_slow=d.get("length_slow"),
            A_fast=d.get("A_fast"),
            period_fast=d.get("period_fast"),
            length_fast=d.get("length_fast"),
            length_decay=d.get("length_decay"),
        )


# ──────────────────────────────────────────────────────────────────
#  Kernel functions (numpy)
# ──────────────────────────────────────────────────────────────────

def _K_rbf(t1: np.ndarray, t2: np.ndarray, eta: float, ell: float
           ) -> np.ndarray:
    d2 = (t1[:, None] - t2[None, :]) ** 2
    return (eta ** 2) * np.exp(-0.5 * d2 / (ell ** 2))


def _K_matern52(t1: np.ndarray, t2: np.ndarray, eta: float, ell: float
                ) -> np.ndarray:
    d = np.abs(t1[:, None] - t2[None, :])
    r = np.sqrt(5.0) * d / ell
    return (eta ** 2) * (1.0 + r + (r ** 2) / 3.0) * np.exp(-r)


def _K_periodic(t1: np.ndarray, t2: np.ndarray, period: float, ell: float
                ) -> np.ndarray:
    """PyMC convention: ``k = exp(-sin²(π d / T) / (2 ℓ²))``.

    PyMC's ``gp.cov.Periodic`` uses this form (see pymc/gp/cov.py:809);
    the more common Rasmussen-Williams convention uses ``-2·sin²/ℓ²``.
    The two differ by a factor of 4 in the inverse length-scale, so the
    ℓ that PyMC fits at MAP time must be applied with PyMC's formula
    here at predict time.
    """
    d = np.abs(t1[:, None] - t2[None, :])
    s = np.sin(np.pi * d / period)
    return np.exp(-0.5 * (s / ell) ** 2)


def _build_kernel_fn(hp: GPHyperparameters):
    """Return a callable ``K(t1, t2) -> np.ndarray`` for the kernel
    described by ``hp`` (excluding the diagonal noise σ_n²I, which
    callers add themselves)."""
    if hp.kernel == "rbf":
        return lambda t1, t2: _K_rbf(t1, t2, hp.eta, hp.ell)
    if hp.kernel == "matern":
        return lambda t1, t2: _K_matern52(t1, t2, hp.eta, hp.ell)
    if hp.kernel == "thesis":
        def K(t1, t2):
            slow = _K_rbf(t1, t2, hp.A_slow, hp.length_slow)
            fast = (
                (hp.A_fast ** 2)
                * _K_periodic(t1, t2, hp.period_fast, hp.length_fast)
                * _K_matern52(t1, t2, eta=1.0, ell=hp.length_decay)
            )
            return slow + fast
        return K
    raise ValueError(f"Unknown kernel '{hp.kernel}'.")


# ──────────────────────────────────────────────────────────────────
#  GP prediction (pure numpy)
# ──────────────────────────────────────────────────────────────────

def _gp_decompose(t_train: np.ndarray, yerr_train: np.ndarray,
                  K_fn, sigma_n: float):
    """Cholesky decompose the training-noise-augmented kernel matrix
    once; reused for predict and cov."""
    n = len(t_train)
    K = K_fn(t_train, t_train)
    K = K + np.diag(yerr_train ** 2 + sigma_n ** 2) + JITTER * np.eye(n)
    L = np.linalg.cholesky(K)
    return L


def _kernel_diag(t: np.ndarray, K_fn) -> np.ndarray:
    """K(t, t) on the diagonal only -- one row per element. Faster
    than calling ``K_fn`` element-wise; for stationary kernels this
    is a constant we extract from a single 1×1 evaluation."""
    if len(t) == 0:
        return np.empty(0)
    # Stationary kernels (rbf/matern/thesis composite) all yield a
    # constant on the diagonal: η², or A_slow² + A_fast². Evaluate it
    # once and broadcast.
    diag_const = K_fn(t[:1], t[:1])[0, 0]
    return np.full(len(t), float(diag_const))


def _gp_predict(t_new: np.ndarray, t_train: np.ndarray,
                y_train_centered: np.ndarray, yerr_train: np.ndarray,
                K_fn, sigma_n: float):
    L = _gp_decompose(t_train, yerr_train, K_fn, sigma_n)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train_centered))
    K_s = K_fn(t_new, t_train)
    K_ss_diag = _kernel_diag(t_new, K_fn)
    mu = K_s @ alpha
    v = np.linalg.solve(L, K_s.T)
    var = K_ss_diag - np.sum(v ** 2, axis=0)
    # Numerical loss can drive var slightly negative for points very
    # close to a training observation. Tolerate up to ~PSD jitter; warn
    # on anything larger (likely indicates ill-conditioning).
    psd_tol = max(1e-6 * float(K_ss_diag.max() if len(K_ss_diag) else 1.0),
                  10 * JITTER)
    if var.size and var.min() < -psd_tol:
        warnings.warn(
            f"GP posterior variance went below tolerance "
            f"({var.min():.3e}); clipping to 0. The kernel matrix may "
            f"be ill-conditioned -- consider a smaller kernel "
            f"length-scale prior or fewer near-duplicate timestamps.",
            stacklevel=2,
        )
    var = np.clip(var, 0.0, None)
    return mu, var


def _gp_cov(t1: np.ndarray, t2: np.ndarray, t_train: np.ndarray,
            yerr_train: np.ndarray, K_fn, sigma_n: float):
    L = _gp_decompose(t_train, yerr_train, K_fn, sigma_n)
    K_t1_train = K_fn(t1, t_train)
    K_t2_train = K_fn(t2, t_train)
    K_t1_t2 = K_fn(t1, t2)
    v1 = np.linalg.solve(L, K_t1_train.T)
    v2 = np.linalg.solve(L, K_t2_train.T)
    return K_t1_t2 - v1.T @ v2


# ──────────────────────────────────────────────────────────────────
#  ReferenceCorrector
# ──────────────────────────────────────────────────────────────────

class ReferenceCorrector:
    """Trains a Gaussian Process on reference-scan centroids and
    predicts the drift correction at arbitrary sample timestamps.

    Workflow::

        rc = ReferenceCorrector(kernel="thesis")
        rc.fit(observations)
        mu, sigma = rc.predict(sample_t_array)
        # mu is the per-sample correction in MHz; sigma propagates to
        # the centroid systematic.

    Parameters
    ----------
    kernel : str
        One of ``rbf``, ``matern``, ``thesis``. Default ``rbf``.
    run_mcmc : bool
        If True, in addition to MAP also run an MCMC sample of the
        hyperparameter posterior — used purely for diagnostics; the
        MAP point is still what drives ``predict``/``cov``. Default
        False because MCMC is slow.
    n_tune, n_samples, n_chains : int
        MCMC settings (only used when ``run_mcmc=True``). Defaults
        match a fast-but-meaningful diagnostic run; the thesis used
        ``tune=500, draws=5000, chains=4``.
    random_seed : int or None
    """

    SUPPORTED_KERNELS = SUPPORTED_KERNELS

    def __init__(self, kernel: str = "rbf", *, run_mcmc: bool = False,
                 n_tune: int = 500, n_samples: int = 1000,
                 n_chains: int = 2, random_seed: int | None = None):
        if kernel not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Choose from "
                f"{self.SUPPORTED_KERNELS}.")
        self.kernel = kernel
        self.run_mcmc = bool(run_mcmc)
        self.n_tune = int(n_tune)
        self.n_samples = int(n_samples)
        self.n_chains = int(n_chains)
        self.random_seed = random_seed

        self._observations: list[ReferenceObservation] = []
        self._t0: float = 0.0
        self._y_mean: float = 0.0
        self._train_t: np.ndarray | None = None
        self._train_y_centered: np.ndarray | None = None
        self._train_yerr: np.ndarray | None = None
        self._hp: GPHyperparameters | None = None
        self._mcmc_summary: dict | None = None

    # ── State ────────────────────────────────────────────────

    @property
    def is_fit(self) -> bool:
        return self._hp is not None

    @property
    def observations(self) -> list[ReferenceObservation]:
        return list(self._observations)

    @property
    def hyperparameters(self) -> GPHyperparameters | None:
        return self._hp

    # ── Fit ─────────────────────────────────────────────────

    def fit(self, observations: Iterable[ReferenceObservation]) -> None:
        """Train the GP on the included observations. Updates the
        corrector's internal state atomically: state is committed only
        after the optimization succeeds, so a failed refit leaves the
        previous corrector untouched.

        Raises ``RuntimeError`` if PyMC is not installed and
        ``ValueError`` for input problems (no observations, invalid
        sigma, etc.).

        All centroids and sigmas must be in MHz. The thesis-kernel
        priors (e.g. ``HalfCauchy(A_slow, beta=1000)``) implicitly
        assume MHz-scale drift; reusing this corrector for inputs in
        a different unit will silently mis-set the priors.

        Notes
        -----
        Not thread-safe: PyTensor's compile cache is process-global.
        Call from a single worker (``QThreadPool(maxThreadCount=1)``
        or a subprocess) -- not concurrently from multiple Qt threads.

        Without a C compiler PyTensor falls back to a pure-Python
        execution mode that is 10-100× slower; on Windows install
        ``conda install m2w64-toolchain`` (or use MSVC) for a
        production-speed fit.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("No reference observations provided.")
        included = [o for o in obs_list if o.include]
        if len(included) < 2:
            raise ValueError(
                "Need at least 2 included reference observations "
                f"to fit the GP (got {len(included)}).")
        for o in included:
            if not (o.sigma > 0 and math.isfinite(o.sigma)):
                raise ValueError(
                    f"Observation '{o.label}' has invalid sigma "
                    f"{o.sigma}.")
            if not math.isfinite(o.t) or not math.isfinite(o.centroid):
                raise ValueError(
                    f"Observation '{o.label}' has non-finite t or "
                    f"centroid.")

        try:
            import pymc as pm  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "PyMC is required for ReferenceCorrector.fit() but is "
                "not installed. Run `uv sync` to install project "
                "dependencies."
            ) from e

        t_arr = np.array([o.t for o in included], dtype=float)
        y_arr = np.array([o.centroid for o in included], dtype=float)
        yerr = np.array([o.sigma for o in included], dtype=float)

        # Center for numerical stability -- the priors implicitly
        # assume a zero-mean GP. The mean is re-added in predict.
        new_t0 = float(t_arr.min())
        new_y_mean = float(np.mean(y_arr))
        t_rel = t_arr - new_t0
        y_centered = y_arr - new_y_mean

        # Run the optimizer FIRST. Only commit state on success so a
        # failed refit doesn't leave the corrector in a hybrid state
        # (new training arrays / old MAP).
        new_hp, new_mcmc_summary = self._fit_pymc(t_rel, y_centered, yerr)

        self._observations = obs_list
        self._t0 = new_t0
        self._y_mean = new_y_mean
        self._train_t = t_rel
        self._train_y_centered = y_centered
        self._train_yerr = yerr
        self._hp = new_hp
        self._mcmc_summary = new_mcmc_summary

    def _fit_pymc(self, t: np.ndarray, y: np.ndarray, yerr: np.ndarray
                  ) -> tuple[GPHyperparameters, dict | None]:
        """Build the PyMC model, run ``find_MAP`` (and optional
        ``sample``), and return ``(MAP hyperparameters,
        mcmc_summary_or_None)``. The caller commits these atomically."""
        import pymc as pm

        # Heuristic priors based on data scale; the thesis priors are
        # used verbatim for the 'thesis' kernel.
        t_range = float(t.max() - t.min()) or 1.0
        y_std = float(np.std(y)) or 1.0
        yerr_med = float(np.median(yerr)) or 1.0

        with pm.Model() as model:
            if self.kernel == "rbf":
                eta = pm.HalfNormal("eta", sigma=max(y_std * 3, 1.0))
                ell = pm.Gamma(
                    "ell", alpha=2.0, beta=2.0 / max(t_range / 4, 1e-3))
                cov_main = (eta ** 2) * pm.gp.cov.ExpQuad(1, ell)
            elif self.kernel == "matern":
                eta = pm.HalfNormal("eta", sigma=max(y_std * 3, 1.0))
                ell = pm.Gamma(
                    "ell", alpha=2.0, beta=2.0 / max(t_range / 4, 1e-3))
                cov_main = (eta ** 2) * pm.gp.cov.Matern52(1, ell)
            else:  # "thesis"
                A_slow = pm.HalfCauchy("A_slow", beta=1000.0)
                length_slow = pm.Gamma("length_slow", alpha=160.0, beta=1.0)
                cov_slow = (A_slow ** 2) * pm.gp.cov.ExpQuad(1, length_slow)

                period_fast = pm.Gamma("period_fast", alpha=12.0, beta=1.0)
                length_fast = pm.Gamma("length_fast", alpha=10.0, beta=1.0)
                A_fast = pm.Gamma("A_fast", alpha=100.0, beta=1.0)
                length_decay = pm.Gamma(
                    "length_decay", alpha=10.0, beta=0.5)
                cov_fast = (
                    (A_fast ** 2)
                    * pm.gp.cov.Periodic(1, period_fast, length_fast)
                    * pm.gp.cov.Matern52(1, length_decay)
                )
                cov_main = cov_slow + cov_fast

            sigma_n = pm.HalfNormal("sigma_n", sigma=max(yerr_med, 1.0))
            cov_total = cov_main + pm.gp.cov.WhiteNoise(sigma_n)
            gp = pm.gp.Marginal(cov_func=cov_total)
            gp.marginal_likelihood(
                "y_obs", X=t[:, None], y=y, sigma=yerr)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                map_kwargs = {"progressbar": False}
                if self.random_seed is not None:
                    # Forward the seed to find_MAP so MAP-only fits are
                    # reproducible (PyMC ≥ 5.10 supports seed=).
                    map_kwargs["seed"] = self.random_seed
                map_pt = pm.find_MAP(**map_kwargs)

            mcmc_summary: dict | None = None
            if self.run_mcmc:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trace = pm.sample(
                        draws=self.n_samples, tune=self.n_tune,
                        chains=self.n_chains,
                        # cores=1: avoid PyMC's spawn-based
                        # multiprocessing on Windows, which races on
                        # PyTensor's process-global compile cache and
                        # adds 30s/worker startup overhead under the
                        # pure-Python fallback.
                        cores=1,
                        random_seed=self.random_seed,
                        progressbar=False, return_inferencedata=True)
                mcmc_summary = self._summarize_trace(trace)

        # find_MAP returns numpy 0-d arrays; coerce to floats.
        def _flt(name: str) -> float:
            return float(np.asarray(map_pt[name]).item())

        if self.kernel in ("rbf", "matern"):
            hp = GPHyperparameters(
                kernel=self.kernel,
                sigma_n=_flt("sigma_n"),
                eta=_flt("eta"),
                ell=_flt("ell"),
            )
        else:
            hp = GPHyperparameters(
                kernel="thesis",
                sigma_n=_flt("sigma_n"),
                A_slow=_flt("A_slow"),
                length_slow=_flt("length_slow"),
                A_fast=_flt("A_fast"),
                period_fast=_flt("period_fast"),
                length_fast=_flt("length_fast"),
                length_decay=_flt("length_decay"),
            )
        return hp, mcmc_summary

    @staticmethod
    def _summarize_trace(trace) -> dict:
        """Return per-parameter mean / std / 2.5-97.5 percentiles for
        the diagnostic display. Lightweight enough to serialize."""
        import arviz as az
        df = az.summary(trace, hdi_prob=0.95, round_to=6)
        out = {}
        for name, row in df.iterrows():
            out[str(name)] = {
                "mean": float(row.get("mean", float("nan"))),
                "sd": float(row.get("sd", float("nan"))),
                "hdi_2.5%": float(row.get("hdi_2.5%", float("nan"))),
                "hdi_97.5%": float(row.get("hdi_97.5%", float("nan"))),
            }
        return out

    # ── Predict / cov ────────────────────────────────────────

    def predict(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean μ(t) and 1σ at the given timestamps.

        ``t`` is in the same units as the training observations'
        ``t``. Returns two arrays of the same length: ``mu`` (MHz,
        the predicted reference centroid drift, NOT centered) and
        ``sigma`` (MHz).
        """
        self._require_fit()
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t_rel = t - self._t0
        K_fn = _build_kernel_fn(self._hp)
        mu, var = _gp_predict(
            t_rel, self._train_t, self._train_y_centered,
            self._train_yerr, K_fn, self._hp.sigma_n)
        return mu + self._y_mean, np.sqrt(var)

    def cov(self, t1, t2) -> np.ndarray:
        """Posterior cross-covariance ``Cov(μ(t1_i), μ(t2_j))``.

        Used by Phase 5 to propagate the GP-correction systematic
        through isotope-shift differences when two isotopes share
        the same corrector.
        """
        self._require_fit()
        t1 = np.atleast_1d(np.asarray(t1, dtype=float)) - self._t0
        t2 = np.atleast_1d(np.asarray(t2, dtype=float)) - self._t0
        K_fn = _build_kernel_fn(self._hp)
        return _gp_cov(t1, t2, self._train_t, self._train_yerr, K_fn,
                       self._hp.sigma_n)

    def _require_fit(self):
        if not self.is_fit:
            raise RuntimeError(
                "ReferenceCorrector has not been fit yet. Call fit() "
                "with at least two reference observations first.")

    # ── Diagnostic plot ─────────────────────────────────────

    def diagnostic_plot(self, ax=None, *, n_grid: int = 400, t_unit: str = "h"):
        """Reproduce thesis Fig. B.2: scatter of observations with
        errorbars, MAP curve, and 1σ / 2σ bands.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            If None, a new figure+axes is created and returned.
        n_grid : int
            Number of evaluation points across the training range.
        t_unit : str
            Label for the x-axis ("h" by default).

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._require_fit()
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        # Plot in time relative to the first training point so the
        # x-axis matches thesis Fig. B.2 ("Timestamp [h] (since first
        # measurement)") rather than displaying ~10⁶ epoch hours.
        t_train_rel = self._train_t  # already (absolute - _t0)
        y_train_abs = self._train_y_centered + self._y_mean
        ax.errorbar(
            t_train_rel, y_train_abs, yerr=self._train_yerr,
            fmt="k.", capsize=2, label="Experimental centroid values")

        pad = 0.02 * (float(t_train_rel.max() - t_train_rel.min()) + 1.0)
        t_grid_rel = np.linspace(
            float(t_train_rel.min()) - pad,
            float(t_train_rel.max()) + pad,
            n_grid,
        )
        # predict() expects absolute time; re-add t0 for the call.
        mu, sigma = self.predict(t_grid_rel + self._t0)

        ax.fill_between(
            t_grid_rel, mu - 2 * sigma, mu + 2 * sigma,
            color="#aac6e0", alpha=0.6, label=r"2-$\sigma$ interval")
        ax.fill_between(
            t_grid_rel, mu - sigma, mu + sigma,
            color="#f3c98a", alpha=0.85, label=r"1-$\sigma$ interval")
        ax.plot(t_grid_rel, mu, color="#1f5d8c", label="MAP")

        ax.set_xlabel(f"Timestamp [{t_unit}] (since first measurement)")
        ax.set_ylabel("Centroid")
        ax.legend(loc="best")
        return ax

    # ── Save / load ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the corrector state (config + observations + MAP
        hyperparameters + training arrays) to a plain dict suitable
        for YAML / JSON dump.

        ``predict``/``cov`` work after ``from_dict`` without re-fit.
        ``run_mcmc`` settings are preserved but the trace itself is
        not (refit if you want a fresh trace).
        """
        d = {
            "kernel": self.kernel,
            "run_mcmc": self.run_mcmc,
            "n_tune": self.n_tune,
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
            "random_seed": self.random_seed,
            "observations": [
                {
                    "t": float(o.t),
                    "centroid": float(o.centroid),
                    "sigma": float(o.sigma),
                    "label": str(o.label),
                    "include": bool(o.include),
                }
                for o in self._observations
            ],
        }
        if self.is_fit:
            d["fit"] = {
                "t0": float(self._t0),
                "y_mean": float(self._y_mean),
                "train_t": [float(x) for x in self._train_t],
                "train_y_centered": [float(x)
                                     for x in self._train_y_centered],
                "train_yerr": [float(x) for x in self._train_yerr],
                "hyperparameters": self._hp.to_dict(),
                "mcmc_summary": self._mcmc_summary,
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceCorrector":
        rc = cls(
            kernel=d.get("kernel", "rbf"),
            run_mcmc=bool(d.get("run_mcmc", False)),
            n_tune=int(d.get("n_tune", 500)),
            n_samples=int(d.get("n_samples", 1000)),
            n_chains=int(d.get("n_chains", 2)),
            random_seed=d.get("random_seed"),
        )
        rc._observations = [
            ReferenceObservation(
                t=float(o["t"]),
                centroid=float(o["centroid"]),
                sigma=float(o["sigma"]),
                label=str(o.get("label", "")),
                include=bool(o.get("include", True)),
            )
            for o in d.get("observations", [])
        ]
        fit = d.get("fit")
        if fit:
            rc._t0 = float(fit["t0"])
            rc._y_mean = float(fit["y_mean"])
            rc._train_t = np.asarray(fit["train_t"], dtype=float)
            rc._train_y_centered = np.asarray(
                fit["train_y_centered"], dtype=float)
            rc._train_yerr = np.asarray(fit["train_yerr"], dtype=float)
            rc._hp = GPHyperparameters.from_dict(fit["hyperparameters"])
            rc._mcmc_summary = fit.get("mcmc_summary")
        return rc
