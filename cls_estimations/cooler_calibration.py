"""Cooler-voltage offset calibration from Yb hyperfine constants.

The beam energy, and so the Doppler-corrected frequency axis, depends
on the cooler voltage. If the logged cooler voltage is wrong by a
constant offset, every extracted frequency scales slightly, and a
hyperfine A constant -- a frequency *difference* within one spectrum --
moves with it. Two isotopes with precisely known A constants therefore
pin the offset down.

The method, following the Yb II 369.42 nm calibration report:

1. For each calibration run, fit the spectrum at a grid of assumed
   cooler offsets dV and record the extracted A (with its error).
2. Per run, fit a straight line A(dV) = a + b dV. Solve A(dV*) = A_lit
   for the offset dV* at which the run reproduces the literature
   value.
3. Per isotope, pool every point of every run of that isotope into one
   line of (A - A_lit) against dV. Its zero crossing is where that
   isotope agrees with literature.
4. The two isotopes' lines cross at one offset: the point where both
   isotopes are equally wrong, which is the calibration. When the
   calibration is perfect the crossing sits on zero and both zero
   crossings coincide; in practice they do not, and the interval
   BETWEEN the two zero crossings is the range over which the analysis
   is repeated to get the systematic uncertainty from the cooler
   voltage.

Everything here is pure numerics on (dV, value, sigma) triples -- no
Qt, no fitting -- so it can be checked against the published report
(see tests/test_cooler_calibration.py, which reproduces its Table 1
and Table 2 to the last printed digit).

Sign convention: A constants are used as reported, INCLUDING sign.
A(173Yb+) is negative; feeding its magnitude would flip the slope and
put the crossing in the wrong place.

Depends on: numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Ground-state (6s S_1/2) magnetic-dipole constants of the Yb ions,
#: de Groote, Atoms 2024, 12, 60, Table 2 -- the values the IGISOL Yb
#: calibration runs are compared against. MHz.
#: 171: 12,642,812,118.4682(4) Hz; 173: 3,497,240,079.85(3) Hz, taken
#: negative by the usual convention (173Yb has a negative moment).
LITERATURE_A = {
    "171Yb": 12642.8121184682,
    "173Yb": -3497.24007985,
}
LITERATURE_A_SIGMA = {
    "171Yb": 4e-10,
    "173Yb": 3e-8,
}
#: Where those numbers come from, for reports and tooltips.
LITERATURE_SOURCE = ("de Groote, Atoms 2024, 12, 60, Table 2 "
                     "(171Yb+ [75], 173Yb+ [76])")


@dataclass(frozen=True)
class ScanPoint:
    """One fit of one run at one assumed cooler offset."""
    run: str
    isotope: str
    dv: float            # assumed cooler-voltage offset, V
    value: float         # extracted A, MHz
    sigma: float         # its fit error, MHz


@dataclass
class LineFit:
    """Weighted straight-line fit ``y = intercept + slope * x``."""
    intercept: float
    slope: float
    cov: np.ndarray      # 2x2, order (intercept, slope)
    n: int
    chi2: float
    ndf: int

    @property
    def intercept_sigma(self):
        return math.sqrt(max(self.cov[0, 0], 0.0))

    @property
    def slope_sigma(self):
        return math.sqrt(max(self.cov[1, 1], 0.0))

    @property
    def reduced_chi2(self):
        return self.chi2 / self.ndf if self.ndf > 0 else float("nan")

    def value(self, x):
        return self.intercept + self.slope * np.asarray(x, dtype=float)

    def sigma_at(self, x):
        """1-sigma band of the fitted line at *x*."""
        x = np.asarray(x, dtype=float)
        var = (self.cov[0, 0] + 2.0 * x * self.cov[0, 1]
               + x ** 2 * self.cov[1, 1])
        return np.sqrt(np.maximum(var, 0.0))

    def root(self):
        """``(x0, sigma)`` where the line crosses zero, or None.

        The error propagates the full covariance, including the
        intercept/slope correlation -- ignoring it noticeably
        understates the uncertainty when the crossing is far from the
        data.
        """
        if not self.slope:
            return None
        x0 = -self.intercept / self.slope
        # d x0/d intercept = -1/b ; d x0/d slope = a/b^2
        j = np.array([-1.0 / self.slope,
                      self.intercept / self.slope ** 2])
        var = float(j @ self.cov @ j)
        return x0, math.sqrt(max(var, 0.0))


def fit_line(x, y, sigma=None):
    """Weighted least-squares line through *(x, y)* with errors.

    Points with a non-finite or non-positive error are given unit
    weight rather than dropped: a fit result with a zero error bar is
    usually a parameter that hit a bound, and silently discarding it
    would change which runs the calibration rests on.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        raise ValueError("a line needs at least two points "
                         f"(got {x.size})")
    if sigma is None:
        w = np.ones_like(x)
    else:
        s = np.asarray(sigma, dtype=float)
        good = np.isfinite(s) & (s > 0)
        w = np.where(good, 1.0 / np.where(good, s, 1.0) ** 2, 1.0)

    design = np.vstack([np.ones_like(x), x]).T
    wd = design * w[:, None]
    normal = design.T @ wd
    try:
        cov = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:      # every dV identical
        raise ValueError(
            "cannot fit a line: the offsets do not vary") from exc
    beta = cov @ (wd.T @ y)
    resid = y - design @ beta
    chi2 = float(np.sum(w * resid ** 2))
    return LineFit(intercept=float(beta[0]), slope=float(beta[1]),
                   cov=cov, n=int(x.size), chi2=chi2,
                   ndf=int(x.size) - 2)


def weighted_mean(values, sigmas):
    """``(mean, sigma)`` by inverse variance; equal weights if no
    usable errors."""
    v = np.asarray(values, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    good = np.isfinite(s) & (s > 0)
    if not good.any():
        return float(np.mean(v)), float("nan")
    w = np.zeros_like(v)
    w[good] = 1.0 / s[good] ** 2
    tot = float(w.sum())
    return float((w * v).sum() / tot), math.sqrt(1.0 / tot)


#: Two slopes closer than this (relative to their own size) are
#: parallel. Exact equality is not enough: two lines built from the
#: same slope differ by ~1e-17 in floating point, which would report a
#: crossing at 1e16 V instead of none.
PARALLEL_TOL = 1e-9


def intersect(line_a, line_b):
    """``(dv, sigma)`` where two lines cross, or None if parallel.

    The two isotopes are fitted from different data, so their
    covariances add independently. Near-parallel lines are not
    rejected -- the crossing is a long lever and the returned sigma
    grows accordingly, which is the honest answer.
    """
    db = line_a.slope - line_b.slope
    scale = max(abs(line_a.slope), abs(line_b.slope), 1.0)
    if abs(db) <= PARALLEL_TOL * scale:
        return None
    da = line_b.intercept - line_a.intercept
    x = da / db
    # x = (a2 - a1) / (b1 - b2)
    ja = np.array([-1.0 / db, -da / db ** 2])     # d/d(a1, b1)
    jb = np.array([1.0 / db, da / db ** 2])       # d/d(a2, b2)
    var = float(ja @ line_a.cov @ ja) + float(jb @ line_b.cov @ jb)
    return x, math.sqrt(max(var, 0.0))


@dataclass
class RunSolution:
    """One run's line and the offset at which it matches literature."""
    run: str
    isotope: str
    line: LineFit                 # of (A - A_lit) against dV
    dv: float | None = None       # offset where A = A_lit
    dv_sigma: float | None = None

    @property
    def slope(self):
        return self.line.slope

    @property
    def slope_sigma(self):
        return self.line.slope_sigma

    @property
    def offset_deviation(self):
        """``a - A_lit``: how far off the run is at dV = 0."""
        return self.line.intercept

    @property
    def offset_deviation_sigma(self):
        return self.line.intercept_sigma


@dataclass
class IsotopeSolution:
    """Every run of one isotope, pooled."""
    isotope: str
    literature: float
    line: LineFit                 # pooled (A - A_lit) against dV
    runs: list[RunSolution] = field(default_factory=list)
    zero: float | None = None     # pooled zero crossing
    zero_sigma: float | None = None
    mean_dv: float | None = None      # weighted mean of the per-run dv
    mean_dv_sigma: float | None = None


@dataclass
class CalibrationResult:
    """The calibration: where the isotope lines cross, and the range
    the systematic scan should cover."""
    isotopes: list[IsotopeSolution]
    dv: float | None = None            # the calibrated offset
    dv_sigma: float | None = None
    scan_lo: float | None = None       # systematic range, from the
    scan_hi: float | None = None       # two zero crossings
    note: str = ""

    @property
    def scan_range(self):
        if self.scan_lo is None or self.scan_hi is None:
            return None
        return (self.scan_lo, self.scan_hi)

    @property
    def scan_width(self):
        r = self.scan_range
        return None if r is None else abs(r[1] - r[0])


def calibrate(points, literature=None):
    """Work the calibration out from the scan points.

    *points* is any iterable of :class:`ScanPoint`. *literature* maps
    isotope -> A_lit in MHz, defaulting to :data:`LITERATURE_A`.

    Each isotope contributes its POOLED line to the crossing, which
    is the report's Fig. 1. The per-run lines and their individual
    solutions are computed too -- they are its Table 1 -- and the
    weighted mean of those per-run solutions is its Table 2.
    """
    lit = dict(LITERATURE_A if literature is None else literature)
    pts = list(points)
    if not pts:
        return CalibrationResult(isotopes=[], note="no scan points")

    by_iso: dict[str, list[ScanPoint]] = {}
    for p in pts:
        by_iso.setdefault(p.isotope, []).append(p)

    solutions = []
    for iso in sorted(by_iso):
        a_lit = lit.get(iso)
        if a_lit is None:
            continue
        iso_pts = by_iso[iso]
        by_run: dict[str, list[ScanPoint]] = {}
        for p in iso_pts:
            by_run.setdefault(p.run, []).append(p)

        runs = []
        for run in sorted(by_run):
            rp = sorted(by_run[run], key=lambda q: q.dv)
            if len(rp) < 2:
                continue
            try:
                line = fit_line([q.dv for q in rp],
                                [q.value - a_lit for q in rp],
                                [q.sigma for q in rp])
            except ValueError:
                continue
            sol = RunSolution(run=run, isotope=iso, line=line)
            root = line.root()
            if root is not None:
                sol.dv, sol.dv_sigma = root
            runs.append(sol)

        try:
            pooled = fit_line([q.dv for q in iso_pts],
                              [q.value - a_lit for q in iso_pts],
                              [q.sigma for q in iso_pts])
        except ValueError:
            continue
        iso_sol = IsotopeSolution(isotope=iso, literature=a_lit,
                                  line=pooled, runs=runs)
        root = pooled.root()
        if root is not None:
            iso_sol.zero, iso_sol.zero_sigma = root
        solved = [r for r in runs if r.dv is not None]
        if solved:
            iso_sol.mean_dv, iso_sol.mean_dv_sigma = weighted_mean(
                [r.dv for r in solved],
                [r.dv_sigma if r.dv_sigma else float("nan")
                 for r in solved])
        solutions.append(iso_sol)

    result = CalibrationResult(isotopes=solutions)
    if len(solutions) < 2:
        result.note = ("need two isotopes to cross: "
                       f"have {[s.isotope for s in solutions]}")
        return result
    if len(solutions) > 2:
        result.note = (f"{len(solutions)} isotopes present; crossing "
                       f"taken from {solutions[0].isotope} and "
                       f"{solutions[1].isotope}")

    a, b = solutions[0], solutions[1]
    cross = intersect(a.line, b.line)
    if cross is not None:
        result.dv, result.dv_sigma = cross
    zeros = [s.zero for s in (a, b) if s.zero is not None]
    if len(zeros) == 2:
        result.scan_lo, result.scan_hi = min(zeros), max(zeros)
    return result



#: Marks a cooler-offset in an output folder name.
OFFSET_TAG_SUFFIX = "V_CO"


def offset_tag(offset):
    """Folder-name tag for a cooler offset: ``-30.0`` -> ``_-30V_CO``.

    Empty for no offset, so an ordinary analysis keeps the plain
    ``iter_003`` it always had. The sign is always written, since
    "+12V" and "-12V" are different analyses and a bare "12V" would
    not say which. Trailing zeros are dropped (``-29.250`` ->
    ``-29.25``) to keep the name readable.
    """
    try:
        v = float(offset or 0.0)
    except (TypeError, ValueError):
        return ""
    if not v:
        return ""
    return f"_{v:+g}{OFFSET_TAG_SUFFIX}"
