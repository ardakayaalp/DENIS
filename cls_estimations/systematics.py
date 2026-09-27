"""Turning a scan over an assumed parameter into a systematic error.

The Yb calibration (cooler_calibration.py) ends with a range rather
than a single number: the two isotopes' zero crossings bracket the
cooler offset, and anywhere in that interval is defensible. The
systematic uncertainty on an extracted parameter is how much that
parameter moves when the analysis is repeated across the interval.

This module is the arithmetic of that: collect (offset, value, sigma)
triples for one parameter of one run, describe the band they span, and
quote one number from it. It is deliberately ignorant of how the
values were obtained -- the fits are driven elsewhere
(gui/analysis/systematic_scan.py) -- so the definitions can be checked
against hand arithmetic.

Two things here are not obvious:

* **There is no one right definition.** A uniform scan across an
  interval is not a probability distribution, so the half band width,
  the full width and the standard deviation all have defenders. All of
  them are computed; the caller picks which one is quoted (DEFINITIONS).

* **Seeding matters more than the arithmetic.** Moving the assumed
  cooler voltage moves the whole Doppler-corrected frequency axis, so
  a centroid travels of order 10 MHz per volt -- at the end of a 20 V
  scan it is hundreds of MHz from where the baseline fit left it, far
  outside a line width, and the fit fails to converge. ``order_outward``
  and ``seed_from_history`` walk the scan outward from the baseline so
  every step starts from a neighbour that has already been fitted.

Depends on: numpy, and fit_line/LineFit from cooler_calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cls_estimations.cooler_calibration import fit_line

#: How a band is turned into one quoted number. The caller chooses;
#: every one of them is reported in the table either way.
HALF_WIDTH = "Half band width"
FULL_WIDTH = "Full band width"
STD_DEV = "Std deviation"
MAX_DEV = "Max deviation from baseline"
RMS_DEV = "RMS deviation from baseline"

DEFINITIONS = (HALF_WIDTH, FULL_WIDTH, STD_DEV, MAX_DEV, RMS_DEV)
DEFAULT_DEFINITION = HALF_WIDTH

#: Points this far from the fitted trend (in combined sigma) are
#: flagged. A scan of a well-behaved parameter is close to a straight
#: line in the assumed offset; a point far off it is a fit that went
#: somewhere else, not physics.
OUTLIER_SIGMA = 5.0

#: A trend cannot be tested with fewer points than this: three points
#: and a two-parameter line leave one degree of freedom, which no
#: outlier can ever fail.
MIN_TREND_POINTS = 4


@dataclass(frozen=True)
class ScanValue:
    """One fitted number at one assumed offset."""
    dv: float                   # the assumed offset, V
    value: float
    sigma: float = float("nan")
    include: bool = True        # unticked steps stay visible but
    #                             take no part in the band
    note: str = ""              # why it is flagged, if it is

    @property
    def usable(self):
        return (self.include and math.isfinite(self.value))


@dataclass
class ParameterBand:
    """Every value one parameter took across the scan.

    ``run`` is empty for a band that is itself an average over runs
    (see :func:`mean_over_runs`).
    """
    project: str
    run: str
    parameter: str
    points: list = field(default_factory=list)   # [ScanValue]
    baseline: float = float("nan")               # the value being
    #                                              corrected, from the
    #                                              baseline iteration
    baseline_sigma: float = float("nan")         # its fit error
    unit: str = "MHz"

    # -- the points that count --------------------------------------
    @property
    def used(self):
        return [p for p in self.points if p.usable]

    @property
    def n(self):
        return len(self.used)

    @property
    def dvs(self):
        return np.array([p.dv for p in self.used], dtype=float)

    @property
    def values(self):
        return np.array([p.value for p in self.used], dtype=float)

    @property
    def sigmas(self):
        return np.array([p.sigma for p in self.used], dtype=float)

    # -- the band ---------------------------------------------------
    @property
    def lo(self):
        return float(self.values.min()) if self.n else float("nan")

    @property
    def hi(self):
        return float(self.values.max()) if self.n else float("nan")

    @property
    def width(self):
        """Full spread of the values across the scanned offsets."""
        return self.hi - self.lo if self.n else float("nan")

    @property
    def half_width(self):
        return self.width / 2.0 if self.n else float("nan")

    @property
    def std(self):
        """Sample standard deviation. Zero for a single point -- a
        one-point scan has no spread, which is different from an
        unknown one."""
        if self.n == 0:
            return float("nan")
        if self.n == 1:
            return 0.0
        return float(np.std(self.values, ddof=1))

    @property
    def max_dev(self):
        """Largest excursion from the baseline fit. Needs a baseline:
        without one there is nothing to deviate from."""
        if not self.n or not math.isfinite(self.baseline):
            return float("nan")
        return float(np.max(np.abs(self.values - self.baseline)))

    @property
    def rms_dev(self):
        if not self.n or not math.isfinite(self.baseline):
            return float("nan")
        d = self.values - self.baseline
        return float(math.sqrt(float(np.mean(d * d))))

    # -- sensitivity ------------------------------------------------
    @property
    def trend(self):
        """Weighted straight line through (offset, value), or None.

        Its slope is the sensitivity of the parameter to the assumed
        offset -- the number to compare against the analytic Doppler
        derivative, and the one that says whether a scan is measuring
        a real dependence or noise.
        """
        if self.n < 2:
            return None
        try:
            return fit_line(self.dvs, self.values, self.sigmas)
        except ValueError:
            return None

    @property
    def slope(self):
        t = self.trend
        return float("nan") if t is None else t.slope

    @property
    def slope_sigma(self):
        t = self.trend
        return float("nan") if t is None else t.slope_sigma

    # -- the quoted number ------------------------------------------
    def systematic(self, definition=DEFAULT_DEFINITION):
        """The band as one number, by the caller's definition."""
        return {
            HALF_WIDTH: self.half_width,
            FULL_WIDTH: self.width,
            STD_DEV: self.std,
            MAX_DEV: self.max_dev,
            RMS_DEV: self.rms_dev,
        }.get(definition, self.half_width)

    def total(self, definition=DEFAULT_DEFINITION):
        """Statistical and systematic in quadrature."""
        sys_ = self.systematic(definition)
        stat = self.baseline_sigma
        parts = [x for x in (stat, sys_) if math.isfinite(x)]
        if not parts:
            return float("nan")
        return math.sqrt(sum(x * x for x in parts))

    @property
    def key(self):
        return f"{self.project}/{self.run or '<mean>'}/{self.parameter}"


def trend_outliers(band, n_sigma=OUTLIER_SIGMA):
    """Indices (into ``band.points``) that sit off the trend line.

    The comparison is against the line's own uncertainty as well as
    the point's, so a scan whose line is poorly determined does not
    flag everything.
    """
    if band.n < MIN_TREND_POINTS:
        return []
    line = band.trend
    if line is None:
        return []
    out = []
    for i, p in enumerate(band.points):
        if not p.usable:
            continue
        resid = p.value - float(line.value(p.dv))
        s_point = p.sigma if (math.isfinite(p.sigma) and p.sigma > 0) else 0.0
        s_line = float(line.sigma_at(p.dv))
        denom = math.hypot(s_point, s_line)
        if denom <= 0:
            continue
        if abs(resid) > n_sigma * denom:
            out.append(i)
    return out


def quality_flag(*, success=True, sigma=float("nan"),
                 baseline_sigma=float("nan"), redchi=float("nan"),
                 baseline_redchi=float("nan"), error_factor=10.0,
                 chi_factor=3.0):
    """A short reason this step's fit looks wrong, or "".

    Everything here is a symptom of a fit that went somewhere the
    baseline did not: it fell over, it came back without an error bar
    (a parameter pinned to a bound), its error blew up, or its
    chi-square moved by more than the given factor. The user decides
    what to do; this only says where to look.
    """
    if not success:
        return "fit failed"
    if not math.isfinite(sigma) or sigma <= 0:
        return "no error"
    if (math.isfinite(baseline_sigma) and baseline_sigma > 0
            and sigma > error_factor * baseline_sigma):
        return f"error x{sigma / baseline_sigma:.0f}"
    if (math.isfinite(redchi) and math.isfinite(baseline_redchi)
            and baseline_redchi > 0 and redchi > 0):
        ratio = redchi / baseline_redchi
        if ratio > chi_factor or ratio < 1.0 / chi_factor:
            return f"chi2 x{ratio:.1f}"
    return ""


def mean_over_runs(bands, project="", parameter=""):
    """Collapse per-run bands into the band of their weighted mean.

    The number that gets published is usually the average over runs,
    not one run, and its band is NOT the average of the per-run bands:
    the runs move together when the offset changes, so averaging first
    and banding second is the right order.

    Offsets that no run reached are dropped; a band whose points are
    all excluded contributes nothing.
    """
    from cls_estimations.isotope_shift import weighted_average

    bands = [b for b in bands if b is not None]
    if not bands:
        return None
    if not project:
        project = bands[0].project
    if not parameter:
        parameter = bands[0].parameter

    by_dv = {}
    for b in bands:
        for p in b.points:
            if p.usable:
                by_dv.setdefault(p.dv, []).append(p)

    points = []
    for dv in sorted(by_dv):
        group = by_dv[dv]
        mean, sig = weighted_average([g.value for g in group],
                                     [g.sigma for g in group])
        points.append(ScanValue(dv=float(dv), value=float(mean),
                                sigma=float(sig)))

    base_vals = [b.baseline for b in bands if math.isfinite(b.baseline)]
    base_sigs = [b.baseline_sigma for b in bands
                 if math.isfinite(b.baseline)]
    if base_vals:
        bmean, bsig = weighted_average(base_vals, base_sigs)
    else:
        bmean = bsig = float("nan")

    return ParameterBand(
        project=project, run="", parameter=parameter, points=points,
        baseline=float(bmean), baseline_sigma=float(bsig),
        unit=bands[0].unit)


def summarise(bands, definition=DEFAULT_DEFINITION):
    """One row per band, for the summary table, the CSV and the report."""
    rows = []
    for b in bands:
        if b is None:
            continue
        rows.append({
            "project": b.project,
            "run": b.run or "<weighted mean>",
            "parameter": b.parameter,
            "unit": b.unit,
            "n": b.n,
            "baseline": b.baseline,
            "stat": b.baseline_sigma,
            "min": b.lo,
            "max": b.hi,
            "band_width": b.width,
            "half_width": b.half_width,
            "std": b.std,
            "max_dev": b.max_dev,
            "rms_dev": b.rms_dev,
            "systematic": b.systematic(definition),
            "total": b.total(definition),
            "slope": b.slope,
            "slope_sigma": b.slope_sigma,
        })
    return rows


# -- walking the scan ------------------------------------------------

def scan_offsets(lo, hi, steps):
    """The offsets a scan visits: *steps* points from *lo* to *hi*."""
    steps = int(steps)
    if steps < 2:
        raise ValueError("a scan needs at least two offsets "
                         f"(got {steps})")
    lo, hi = float(lo), float(hi)
    if lo == hi:
        raise ValueError("the scan range has zero width")
    if lo > hi:
        lo, hi = hi, lo
    return [float(v) for v in np.linspace(lo, hi, steps)]


def order_outward(offsets, start):
    """*offsets* ordered by distance from *start*.

    This is the order a continuation scan walks them in: by the time
    any offset is fitted, the offset next to it on the way back to the
    baseline has already been fitted, so there is always a nearby
    answer to start from. Ties (two offsets equally far out, one each
    side) keep their natural order, lowest first, so the walk is
    reproducible.
    """
    start = float(start)
    return sorted((float(v) for v in offsets),
                  key=lambda v: (abs(v - start), v))


def seed_from_history(dv, history, *, extrapolate=True):
    """Where to start a fit at *dv*, given what earlier steps found.

    ``history`` is ``[(dv, value, sigma)]`` for ONE parameter of ONE
    run, in any order, holding only steps that actually converged.

    With two or more neighbours the value is carried along the line
    through the two nearest ones -- the dependence on the assumed
    offset is very nearly linear, so this lands close even at the far
    end of the scan. The line is only used when those two neighbours
    actually differ by more than their errors; otherwise the parameter
    is not measurably moving and extrapolating it would just amplify
    noise.

    Returns None when there is nothing to go on, and the caller falls
    back to the baseline fit.
    """
    pts = [(float(d), float(v), float(s))
           for d, v, s in (history or [])
           if math.isfinite(d) and math.isfinite(v)]
    if not pts:
        return None
    pts.sort(key=lambda t: abs(t[0] - float(dv)))
    if len(pts) == 1 or not extrapolate:
        return pts[0][1]

    (d0, v0, s0), (d1, v1, s1) = pts[0], pts[1]
    if d0 == d1:
        return v0
    noise = math.hypot(s0 if math.isfinite(s0) else 0.0,
                       s1 if math.isfinite(s1) else 0.0)
    if abs(v1 - v0) <= noise:
        return v0
    slope = (v1 - v0) / (d1 - d0)
    return v0 + slope * (float(dv) - d0)


def shift_bounds(value, lo, hi, new_value):
    """Keep a parameter's window around its seed.

    A seed outside its own bounds is refused by the fitter, and
    clipping it silently would pin the parameter at a bound and
    quietly produce a fit with no error bar. So when the seed would
    fall outside, the window travels with it, keeping its width.

    A window that still contains the seed is left exactly where it is:
    those bounds are usually a physical limit (a width that may not go
    negative, an amplitude that may not), not a leash on the value.
    """
    if not math.isfinite(new_value):
        return lo, hi

    def _finite(b):
        return b is not None and math.isfinite(b)

    if ((not _finite(lo) or new_value >= lo)
            and (not _finite(hi) or new_value <= hi)):
        return lo, hi
    shift = float(new_value) - float(value) if math.isfinite(value) else 0.0
    return (lo + shift if _finite(lo) else lo,
            hi + shift if _finite(hi) else hi)
