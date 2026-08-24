"""Per-run voltage calibration: fit, diagnose, override, share across tabs.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Every ASDF run carries a DAC->HV calibration table (``CalSet`` ->
``CalReadback``). clstools fits a polynomial through it inside
``Load_Run`` and ``Compute_Voltages`` uses the coefficients to turn each
event's DAC value into a real voltage::

    DV_cal = poly(DV) * VAccDiv
    V      = Vrfq*VCoolDiv + VCoolOffset - DV_cal

When the HV supply is still settling at the start of the calibration
sweep the first readbacks are wrong, the polynomial is pulled toward
them, and *every* voltage in the run is biased -- which propagates
straight into the Doppler-shifted frequency axis and shifts the fitted
centroid. At 30 kV / mass 51 the sensitivity is ~18 MHz per volt, so a
bad calibration is the same size as the isotope shifts being measured.

The one invariant
-----------------
``Compute_Voltages`` reads *only* ``Cal``, ``Cal_order`` and ``VAccDiv``.
So the whole feature reduces to::

    Load_Run  ->  overwrite data.Cal  ->  Compute_Voltages

``apply_calibration`` does the overwrite, which means the app owns the
fit and the physics no longer depends on which clstools happens to be
installed (upstream added an on-by-default 2-sigma cut in ``c0334c0``
that silently drops points and rewrites ``Cal_df``). For the same
reason the calibration points are always re-read from the ASDF header
rather than taken from ``data.Cal_df`` -- a filtering clstools will have
already discarded points from that frame.

Note what this *does* change: against the newer, filtering clstools, the
app's default is no longer the same. That build was dropping every point
beyond 2 sigma of the residuals on every load; we switch its filter off
and keep all points unless the user says otherwise. Runs with a
calibration outlier can therefore fit to a different centroid than they
did before -- deliberately, because rejection becomes an explicit and
recorded choice instead of a silent one, and because it is the *pinned*
dependency's behaviour. ``reject="sigma", sigma=2.0, iterative=False``
reproduces the old filter exactly for anyone who needs to.

Zero intercept
--------------
clstools ``f6b9d7`` added an ``ignore_intercept`` load option that forces
the calibration offset ``p0`` to zero -- a gain-only, through-the-origin
calibration for a chain known to have no DC offset. Because the app owns
the fit, that clstools flag would never reach our numbers, so the control
is reimplemented here instead: a ``fit`` spec may set
``ignore_intercept=True`` and ``fit_calibration`` then constrains ``p0``
out of the least-squares problem entirely (not a fit-then-zero, which
would leave the slope biased by the offset it just discarded). Same
reasoning as the rejection rules: an upstream calibration knob becomes an
explicit, recorded, per-run choice the UI exposes.

Units
-----
``CalSet`` is in volts (Pre-Analysis subtracts it from the readback
directly, ``preanalysis_tab.py:4919``). ``CalReadback`` is in monitor
units and ``VAccDiv`` (=1000) converts it to volts. clstools therefore
fits in mixed units, with a slope of ~1e-3.

We work in the *volts convention*, defined exactly by what
``Compute_Voltages`` produces::

    DV_cal[V] = sum_i coeffs_v[i] * DV**i         coeffs_v[i] = Cal[i] * VAccDiv

which makes ``p0`` an offset in volts and ``p1`` a gain of ~1 -- the
numbers a physicist can actually judge. Specs store ``coeffs_v``;
``apply_calibration`` divides by ``VAccDiv`` on the way into ``data.Cal``.

We also record ``Cal_err`` as ``sqrt(diag(cov))``. clstools stores
``cov[i, i]``, i.e. the *variance*, not the standard error. Nothing in
the app reads it today, so correcting it here is safe.

Registry
--------
A calibration override is a property of the *file*, not of an analysis
block, so it lives in a process-wide ``CalibrationRegistry`` keyed by
canonical path -- a deliberate mirror of ``gui.scan_filter``'s
``ScanFilterRegistry``, which already solves this exact problem. It
serializes under the project YAML's top-level ``calibrations`` key.

Subprocess note
---------------
Fits run in a worker that cannot reach the singleton. Every entry point
here therefore takes an explicit ``calibrations`` **map** (path -> spec)
rather than reading the registry, and the map is passed whole rather
than per-file so that a ``borrow`` spec can still resolve its donor.
In-process callers pass ``get_registry().to_dict()``.

Depends on: numpy, PySide6; asdf and pandas imported lazily.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QObject, Signal


# ── Constants ────────────────────────────────────────────────────────

DEFAULT_VACC_DIV = 1000.0
DEFAULT_ORDER = 1
MAX_ORDER = 3

MODES = ("fit", "borrow", "coeffs")
REJECT_RULES = ("none", "sigma", "first_n", "manual")

#: Robust-sigma threshold used to *nominate* outlier candidates when
#: flagging a run. Flagging never changes a number -- it only puts a
#: warning in front of the user, who then decides.
FLAG_SIGMA = 3.0

#: ...but statistical oddity alone is not enough to flag. On a perfectly
#: clean 20-point table, Gaussian noise routinely throws one point past
#: 3 robust sigmas, and a warning that cries wolf on good runs is worse
#: than no warning at all -- the user learns to click through it and then
#: misses the real one.
#:
#: So a run is flagged only if dropping its suspect points would actually
#: *move the voltage*: the two calibrations must disagree by more than
#: this, somewhere in the sweep. At ~18 MHz/V (30 kV, A=51) 0.05 V is
#: about 1 MHz -- an order of magnitude below a typical statistical
#: centroid error, and three orders below the multi-volt tilt a real
#: settling glitch produces. The separation is so wide that the exact
#: value hardly matters.
FLAG_MIN_IMPACT_V = 0.05

#: Iterations for the iterative/MAD rejection rule.
MAX_REJECT_ITER = 10

#: numpy 2 moved RankWarning to np.exceptions.
_RankWarning = getattr(getattr(np, "exceptions", np), "RankWarning", Warning)


class CalibrationError(RuntimeError):
    """Raised when a calibration spec cannot be resolved.

    Deliberately fatal: an *explicit* override that cannot be honoured
    must never silently degrade to the file default, because the whole
    point of the feature is that the calibration in force is the one the
    user chose.
    """


# ── Path normalization ───────────────────────────────────────────────

def canonical_path(path: str) -> str:
    """Normalize a file path to a single registry key.

    Same rule as ``gui.scan_filter.canonical_path`` so that the two
    registries agree on what "the same file" means.
    """
    if not isinstance(path, str) or not path:
        return path
    try:
        return os.path.normcase(os.path.abspath(path))
    except (TypeError, ValueError):
        return path


# ── Reading the calibration table ────────────────────────────────────

# (canonical_path, mtime_ns) -> (set_v, read_raw). The header read is
# cheap but it happens on every load of every run, and the ASDF open
# dominates it.
_POINTS_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}


def read_cal_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``(CalSet, CalReadback)`` straight from the ASDF header.

    Returns ``(set_v, read_raw)`` -- ``set_v`` already in volts,
    ``read_raw`` in raw monitor units (multiply by ``VAccDiv`` for
    volts). Both are float ndarrays of equal length; a run with no
    calibration table yields two empty arrays.

    Read from the file rather than from ``data.Cal_df`` on purpose: a
    filtering clstools rewrites that frame with the *surviving* points
    only, so it is not a trustworthy source for the full table.
    """
    import asdf

    key = None
    try:
        st = os.stat(path)
        key = (canonical_path(path), st.st_mtime_ns)
        cached = _POINTS_CACHE.get(key)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()
    except OSError:
        pass

    with asdf.open(path) as af:
        set_v = np.asarray(af.tree.get("CalSet", []), dtype=float)
        read_raw = np.asarray(af.tree.get("CalReadback", []), dtype=float)

    n = min(len(set_v), len(read_raw))
    set_v, read_raw = set_v[:n], read_raw[:n]

    if key is not None:
        _POINTS_CACHE[key] = (set_v.copy(), read_raw.copy())
    return set_v, read_raw


_BEAM_CACHE: Dict[Tuple[str, int], Tuple[float, float]] = {}


def read_beam_header(path: str) -> Tuple[float, float]:
    """``(cooler_v, laser_cm)`` from the ASDF header, for the MHz readout.

    The Doppler shift a calibration error produces depends on the beam energy
    and the laser setpoint, and **both are per-run**: quoting one run's cost in
    MHz using a neighbour's cooler voltage would be worse than quoting none.
    Mass and harmonic, by contrast, belong to the analysis, so callers supply
    those.

    Cached like the calibration points: the overview calls this once per row
    and refreshes on every registry change, so an uncached ASDF open here would
    make a hundred-run campaign re-read a hundred headers on every click.

    Returns ``(0.0, 0.0)`` when the file cannot be read; callers treat a zero
    as "physics unknown" and fall back to volts.
    """
    import asdf

    key = None
    try:
        st = os.stat(path)
        key = (canonical_path(path), st.st_mtime_ns)
        hit = _BEAM_CACHE.get(key)
        if hit is not None:
            return hit
    except OSError:
        return 0.0, 0.0

    try:
        with asdf.open(path) as af:
            cooler = float(af.tree.get("CoolerVoltage", 0) or 0)
            laser = float(af.tree.get("LaserSetpoint", 0) or 0)
    except (OSError, KeyError, ValueError, TypeError):
        return 0.0, 0.0
    # CoolerVoltage is the monitor reading; VCoolDiv (10000) makes it volts.
    out = (cooler * 10000.0, laser)
    if key is not None:
        _BEAM_CACHE[key] = out
    return out


def clear_points_cache() -> None:
    """Drop the ASDF header caches (tests; also after a file is rewritten)."""
    _POINTS_CACHE.clear()
    _BEAM_CACHE.clear()


# ── The fit ──────────────────────────────────────────────────────────

@dataclass(eq=False)
class CalibrationFit:
    """A polynomial fit of readback[V] against set[V].

    ``coeffs_v`` / ``errs_v`` are **ascending** (``p0`` first), in the
    volts convention: ``DV_cal[V] = sum_i coeffs_v[i] * DV**i``. So
    ``p0`` is an offset in volts and ``p1`` a gain of ~1.

    ``residuals_v`` covers **every** point, excluded ones included, so
    the diagnostic can draw the points it dropped. The scalar summaries
    (``sigma_v``, ``rms_v``, ``max_abs_v``) are over the *used* points
    only.

    There is deliberately no chi2/ndf: the readback carries no quoted
    uncertainty, so any chi-square would just be the residual scatter
    divided by itself. sigma / RMS / max|residual| say the same thing
    without pretending to a goodness-of-fit.
    """

    coeffs_v: Tuple[float, ...]
    errs_v: Tuple[float, ...]
    order: int
    excluded: Tuple[int, ...]
    residuals_v: np.ndarray
    set_v: np.ndarray
    read_v: np.ndarray
    sigma_v: float
    rms_v: float
    max_abs_v: float

    @property
    def n_points(self) -> int:
        return int(len(self.set_v))

    @property
    def n_used(self) -> int:
        return int(self.n_points - len(self.excluded))

    @property
    def used_mask(self) -> np.ndarray:
        m = np.ones(self.n_points, dtype=bool)
        if self.excluded:
            m[list(self.excluded)] = False
        return m

    def predict_v(self, x) -> np.ndarray:
        """Calibrated voltage for one or more raw DAC values."""
        return np.polyval(np.asarray(self.coeffs_v, float)[::-1],
                          np.asarray(x, dtype=float))


def _polyfit_descending(xu, yu, order: int, ignore_intercept: bool):
    """Least-squares coefficients (descending) and their standard errors.

    Returns ``(values, errs)`` shaped exactly like ``np.polyfit(cov=True)``:
    both length ``order + 1``, highest power first, so the caller's
    ``[::-1]`` yields the ascending volts convention. Errors are ``NaN``
    when too few points remain to estimate a covariance.

    With ``ignore_intercept`` the constant term is dropped from the design
    matrix rather than fitted and zeroed afterwards -- forcing ``p0 = 0``
    after an unconstrained fit would leave the slope carrying whatever
    offset the data wanted, which is not the same calibration at all. The
    intercept then has an error of exactly 0 (it was not estimated).
    """
    n_used = int(len(xu))
    if not ignore_intercept:
        # np.polyfit(cov=True) additionally demands n > order + 2 for its
        # covariance estimate. Below that the fit itself is still perfectly
        # well defined -- we just can't quote coefficient errors.
        if n_used > order + 2:
            values, cov = np.polyfit(xu, yu, order, cov=True)
            errs = np.sqrt(np.abs(np.diag(cov)))
        else:
            values = np.polyfit(xu, yu, order)
            errs = np.full(order + 1, np.nan)
        return values, errs

    # Forced zero intercept: fit y ~ c1*x + c2*x**2 + ... (no constant
    # column), then splice p0 = 0 back in as the last (lowest-power)
    # descending coefficient so np.polyval sees an offset of zero.
    powers = np.arange(1, order + 1)
    design = np.asarray(xu, float)[:, None] ** powers[None, :]   # (n, order)
    coef, *_ = np.linalg.lstsq(design, yu, rcond=None)           # c1..c_order
    values = np.concatenate([coef[::-1], [0.0]])
    errs = np.full(order + 1, np.nan)
    errs[-1] = 0.0   # the intercept is fixed, not estimated
    dof = n_used - order   # order free parameters
    if dof >= 2:
        resid = yu - design @ coef
        try:
            cov_free = (float(resid @ resid) / dof) * np.linalg.inv(
                design.T @ design)
            e_free = np.sqrt(np.abs(np.diag(cov_free)))          # c1..c_order
            errs = np.concatenate([e_free[::-1], [0.0]])
        except np.linalg.LinAlgError:
            pass
    return values, errs


def fit_calibration(set_v, read_raw, order: int = DEFAULT_ORDER,
                    excluded: Iterable[int] = (),
                    vacc_div: float = DEFAULT_VACC_DIV, *,
                    ignore_intercept: bool = False) -> CalibrationFit:
    """Least-squares polynomial fit of readback[V] vs set[V].

    ``read_raw`` is in monitor units as stored in the ASDF; it is scaled
    by ``vacc_div`` here so that the returned coefficients and residuals
    are in the volts convention.

    ``ignore_intercept`` forces ``p0 = 0`` (a through-the-origin,
    gain-only calibration) by constraining it out of the least-squares
    problem -- the app-owned equivalent of clstools' ``ignore_intercept``
    load option.

    Raises ``CalibrationError`` when too few points survive the
    exclusions (a degree-n polynomial needs n+1 of them).
    """
    x = np.asarray(set_v, dtype=float)
    y = np.asarray(read_raw, dtype=float) * float(vacc_div)
    n = int(len(x))
    order = int(order)

    excl = tuple(sorted({int(i) for i in (excluded or ()) if 0 <= int(i) < n}))
    mask = np.ones(n, dtype=bool)
    if excl:
        mask[list(excl)] = False

    n_used = int(mask.sum())
    if n_used < order + 1:
        raise CalibrationError(
            f"Cannot fit an order-{order} calibration to {n_used} point"
            f"{'s' if n_used != 1 else ''} "
            f"({n} in the file, {len(excl)} excluded); "
            f"an order-{order} polynomial needs at least {order + 1}.")

    xu, yu = x[mask], y[mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _RankWarning)
        values, errs = _polyfit_descending(xu, yu, order, ignore_intercept)

    # np.polyfit returns descending; the app (and clstools' Cal) is
    # ascending, p0 first.
    coeffs_v = tuple(float(c) for c in values[::-1])
    errs_v = tuple(float(e) for e in errs[::-1])

    resid = y - np.polyval(values, x)
    ru = resid[mask]
    sigma = float(np.std(ru, ddof=1)) if n_used > 1 else 0.0

    return CalibrationFit(
        coeffs_v=coeffs_v,
        errs_v=errs_v,
        order=order,
        excluded=excl,
        residuals_v=resid,
        set_v=x,
        read_v=y,
        sigma_v=sigma,
        rms_v=float(np.sqrt(np.mean(ru ** 2))) if n_used else 0.0,
        max_abs_v=float(np.max(np.abs(ru))) if n_used else 0.0,
    )


def _robust_sigma(resid: np.ndarray) -> float:
    """MAD-based scatter estimate, normalized to a Gaussian sigma.

    The plain standard deviation is the wrong tool for finding outliers:
    one bad point inflates sigma enough to hide itself inside the cut.
    That is exactly the failure mode of clstools' single-pass 2-sigma
    filter, and exactly the case this feature exists to catch.
    """
    if len(resid) == 0:
        return 0.0
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return 1.4826 * mad


def auto_outliers(set_v, read_raw, order: int = DEFAULT_ORDER, *,
                  rule: str = "none", sigma: float = 2.0,
                  iterative: bool = False, drop_first: int = 0,
                  vacc_div: float = DEFAULT_VACC_DIV,
                  ignore_intercept: bool = False) -> Tuple[int, ...]:
    """Indices the given rejection rule would drop. Never mutates state.

    Rules:

    ``none``
        Drops nothing. The app default -- no number changes unless the
        user asks for it.
    ``first_n``
        Drops the first ``drop_first`` points. The most physically
        motivated rule for this failure mode: the HV supply is settling,
        and settling happens at the start.
    ``sigma``
        Drops ``|residual| > sigma * scatter``. With ``iterative=False``
        this is a single pass using the plain standard deviation, i.e.
        bit-for-bit what clstools ``c0334c0`` does at ``sigma=2``. With
        ``iterative=True`` it re-fits after each pass and uses the robust
        MAD scatter, which actually catches a cluster of bad points.
    ``manual``
        Drops nothing here; the caller supplies the indices.
    """
    x = np.asarray(set_v, dtype=float)
    n = int(len(x))
    if rule == "first_n":
        return tuple(range(min(max(int(drop_first), 0), n)))
    if rule in ("none", "manual") or n == 0:
        return ()
    if rule != "sigma":
        raise CalibrationError(
            f"Unknown rejection rule {rule!r}; "
            f"expected one of {', '.join(REJECT_RULES)}.")

    if not iterative:
        # clstools-compatible: one pass, plain std, computed over all
        # points including the outliers.
        fit = fit_calibration(x, read_raw, order, (), vacc_div,
                              ignore_intercept=ignore_intercept)
        s = fit.sigma_v
        if s <= 0:
            return ()
        bad = np.abs(fit.residuals_v) > sigma * s
        return tuple(int(i) for i in np.flatnonzero(bad))

    excluded: set[int] = set()
    for _ in range(MAX_REJECT_ITER):
        if n - len(excluded) < order + 2:
            break
        fit = fit_calibration(x, read_raw, order, tuple(excluded), vacc_div,
                              ignore_intercept=ignore_intercept)
        keep = fit.used_mask
        s = _robust_sigma(fit.residuals_v[keep])
        if s <= 0:
            break
        bad = np.flatnonzero(np.abs(fit.residuals_v) > sigma * s)
        new = {int(i) for i in bad} - excluded
        if not new:
            break
        # Never strip so many points that the fit becomes degenerate.
        room = (n - order - 2) - len(excluded)
        if room <= 0:
            break
        if len(new) > room:
            worst = sorted(new, key=lambda i: -abs(fit.residuals_v[i]))
            new = set(worst[:room])
        excluded |= new
    return tuple(sorted(excluded))


# ── Specs ────────────────────────────────────────────────────────────

def normalize_spec(spec: Optional[dict]) -> Optional[dict]:
    """Validate and clean a spec dict; ``None`` means "file default".

    Tolerant by design -- the project YAML can be hand-edited, so a
    malformed entry degrades to ``None`` (file default) rather than
    taking the whole project down. An entry that *is* well-formed but
    impossible (e.g. a borrow whose donor is gone) is a different case:
    that fails loudly later, in ``resolve_calibration``.
    """
    if not isinstance(spec, dict):
        return None
    mode = spec.get("mode")
    if mode not in MODES:
        return None

    out: Dict[str, Any] = {"mode": mode}

    order = spec.get("order", DEFAULT_ORDER)
    try:
        order = int(order)
    except (TypeError, ValueError):
        order = DEFAULT_ORDER
    out["order"] = min(max(order, 1), MAX_ORDER)

    note = spec.get("note")
    if isinstance(note, str) and note.strip():
        out["note"] = note.strip()

    if mode == "fit":
        rule = spec.get("reject", "none")
        out["reject"] = rule if rule in REJECT_RULES else "none"
        try:
            out["sigma"] = float(spec.get("sigma", 2.0))
        except (TypeError, ValueError):
            out["sigma"] = 2.0
        out["iterative"] = bool(spec.get("iterative", False))
        try:
            out["drop_first"] = max(int(spec.get("drop_first", 0)), 0)
        except (TypeError, ValueError):
            out["drop_first"] = 0
        out["ignore_intercept"] = bool(spec.get("ignore_intercept", False))
        excl = spec.get("excluded") or ()
        if isinstance(excl, (list, tuple, set)):
            out["excluded"] = sorted(
                {int(i) for i in excl
                 if isinstance(i, (int, float)) and float(i).is_integer()
                 and int(i) >= 0})
        else:
            out["excluded"] = []
        # A "fit" that changes nothing is the file default; drop it so an
        # inert entry never shows an override badge. Forcing p0 = 0 is a
        # real change even when nothing else is set.
        if (out["reject"] == "none" and not out["excluded"]
                and out["order"] == DEFAULT_ORDER
                and not out["ignore_intercept"] and "note" not in out):
            return None

    elif mode == "borrow":
        donor = spec.get("donor")
        if not isinstance(donor, str) or not donor:
            return None
        out["donor"] = donor

    elif mode == "coeffs":
        raw = spec.get("coeffs_v")
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return None
        try:
            coeffs = [float(c) for c in raw]
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(coeffs)):
            return None
        coeffs = coeffs[:MAX_ORDER + 1]
        out["coeffs_v"] = coeffs
        out["order"] = len(coeffs) - 1

    return out


def _canonical_map(calibrations: Optional[Dict[str, dict]]) -> Dict[str, dict]:
    """Re-key a path -> spec map canonically.

    The registry already stores canonical keys, but a map can also reach
    us straight from a hand-edited YAML or a caller that built it with
    ``os.path.join``. A lookup miss here would silently fall back to the
    file default -- the one failure mode this module exists to prevent --
    so normalize both sides rather than trusting the caller.
    """
    if not calibrations:
        return {}
    return {canonical_path(k): v for k, v in calibrations.items()}


def spec_fingerprint(spec: Optional[dict]) -> str:
    """Stable string for a spec, for use inside cache keys.

    Pre-Analysis keys its binned-data cache on every knob that can change
    the result (``preanalysis_tab.py:4369``); the calibration is now one
    of them.
    """
    norm = normalize_spec(spec)
    if norm is None:
        return ""
    return json.dumps(norm, sort_keys=True, separators=(",", ":"))


def describe_spec(spec: Optional[dict]) -> str:
    """One-line human summary, for badges and tooltips."""
    norm = normalize_spec(spec)
    if norm is None:
        return "file default"
    mode = norm["mode"]
    if mode == "borrow":
        return f"borrowed from {os.path.basename(norm['donor'])}"
    if mode == "coeffs":
        return "manual coefficients (" + ", ".join(
            f"{c:g}" for c in norm["coeffs_v"]) + ")"
    bits = [f"order {norm['order']}"]
    if norm.get("ignore_intercept"):
        bits.append("p₀=0")
    rule = norm.get("reject", "none")
    if rule == "sigma":
        bits.append(f"{norm['sigma']:g}σ"
                    + (" iterative" if norm.get("iterative") else ""))
    elif rule == "first_n":
        bits.append(f"drop first {norm['drop_first']}")
    n_excl = len(norm.get("excluded") or ())
    if n_excl:
        bits.append(f"{n_excl} point{'s' if n_excl != 1 else ''} excluded")
    return ", ".join(bits)


# ── Resolution ───────────────────────────────────────────────────────

@dataclass(eq=False)
class CalibrationResult:
    """The calibration actually in force for one run.

    ``mode`` is ``file`` when no override applied. ``fit`` is present
    whenever the run's own points were fitted (so the diagnostic and the
    overview can show residuals); it is ``None`` only when the
    coefficients came from somewhere else entirely and the file has no
    points of its own to compare against.
    """

    coeffs_v: Tuple[float, ...]
    errs_v: Tuple[float, ...]
    order: int
    mode: str
    #: Indices excluded from **this run's own** calibration table. For a
    #: borrow or hand-entered coefficients it is always empty: no point of
    #: this run was dropped, the polynomial simply came from somewhere else.
    #: The donor's own exclusions live in ``donor_excluded`` -- conflating the
    #: two would index this run's point array with another run's indices.
    excluded: Tuple[int, ...] = ()
    donor: Optional[str] = None
    donor_excluded: Tuple[int, ...] = ()
    note: str = ""
    fit: Optional[CalibrationFit] = None
    n_points: int = 0
    fallback: bool = False
    #: The polynomial was fitted with its offset constrained to zero.
    #: Only ever true for a ``fit``-mode result; ``coeffs`` and ``borrow``
    #: take their p0 from wherever the coefficients came from.
    ignore_intercept: bool = False

    @property
    def sigma_v(self) -> float:
        return self.fit.sigma_v if self.fit is not None else float("nan")

    def coeffs_raw(self, vacc_div: float = DEFAULT_VACC_DIV):
        """Coefficients in clstools' convention (what goes into ``Cal``)."""
        return [c / float(vacc_div) for c in self.coeffs_v]

    def predict_v(self, x) -> np.ndarray:
        return np.polyval(np.asarray(self.coeffs_v, float)[::-1],
                          np.asarray(x, dtype=float))

    def to_report_dict(self) -> dict:
        """Compact, picklable audit record for the fit report / merge."""
        d = {
            "mode": self.mode,
            "order": int(self.order),
            "coeffs_v": [float(c) for c in self.coeffs_v],
            "n_points": int(self.n_points),
            "n_excluded": int(len(self.excluded)),
            "excluded": [int(i) for i in self.excluded],
        }
        if self.donor:
            d["donor"] = self.donor
            # The donor's own dropped points, kept separate from this run's
            # ``excluded`` (which is empty for a borrow) so the report can
            # never be read as "this run dropped points 0-2".
            d["donor_excluded"] = [int(i) for i in self.donor_excluded]
        if self.note:
            d["note"] = self.note
        if self.ignore_intercept:
            d["ignore_intercept"] = True
        if self.fit is not None:
            d["sigma_v"] = float(self.fit.sigma_v)
            d["max_abs_resid_v"] = float(self.fit.max_abs_v)
        if self.fallback:
            d["fallback"] = True
        return d


def resolve_calibration(path: str,
                        calibrations: Optional[Dict[str, dict]] = None,
                        *,
                        cal_order: int = DEFAULT_ORDER,
                        vacc_div: float = DEFAULT_VACC_DIV,
                        _depth: int = 0) -> CalibrationResult:
    """Work out the coefficients in force for ``path``.

    ``calibrations`` is the whole path -> spec map, not just this file's
    entry, because a ``borrow`` has to look its donor up. Pass
    ``get_registry().to_dict()`` in-process; the fit worker is handed a
    snapshot of the same map.

    ``cal_order`` is the Source block's order, used when the spec does
    not pin one of its own.
    """
    calibrations = _canonical_map(calibrations)
    spec = normalize_spec(calibrations.get(canonical_path(path)))

    if spec is None:
        # File default: our own plain polyfit over every point. If the
        # file has too few points to fit, fall back to whatever clstools
        # computed rather than failing a run that works today -- "no
        # override requested" must never make things worse.
        try:
            set_v, read_raw = read_cal_points(path)
            fit = fit_calibration(set_v, read_raw, cal_order, (), vacc_div)
        except (CalibrationError, OSError, KeyError, ValueError):
            return CalibrationResult(
                coeffs_v=(), errs_v=(), order=cal_order, mode="file",
                fit=None, n_points=0, fallback=True)
        return CalibrationResult(
            coeffs_v=fit.coeffs_v, errs_v=fit.errs_v, order=fit.order,
            mode="file", excluded=(), fit=fit, n_points=fit.n_points)

    order = spec.get("order", cal_order)
    note = spec.get("note", "")

    if spec["mode"] == "coeffs":
        coeffs_v = tuple(spec["coeffs_v"])
        # Still fit the file's own points, purely so the diagnostic can
        # show how the hand-entered polynomial sits against them.
        ref_fit = None
        n_points = 0
        try:
            set_v, read_raw = read_cal_points(path)
            n_points = int(len(set_v))
            if n_points:
                ref_fit = _fit_against(set_v, read_raw, coeffs_v, order,
                                       vacc_div)
        except (OSError, KeyError, ValueError):
            pass
        return CalibrationResult(
            coeffs_v=coeffs_v,
            errs_v=tuple(float("nan") for _ in coeffs_v),
            order=len(coeffs_v) - 1, mode="coeffs", note=note,
            fit=ref_fit, n_points=n_points)

    if spec["mode"] == "borrow":
        if _depth:
            raise CalibrationError(
                f"Calibration for {os.path.basename(path)} borrows from "
                f"{os.path.basename(spec['donor'])}, which itself borrows "
                f"from another run. Chained borrows are not allowed -- "
                f"point it at the run that owns the calibration.")
        donor = spec["donor"]
        if not os.path.exists(donor):
            raise CalibrationError(
                f"Calibration for {os.path.basename(path)} borrows from "
                f"{donor}, which does not exist.")
        donor_res = resolve_calibration(
            donor, calibrations, cal_order=order, vacc_div=vacc_div,
            _depth=_depth + 1)
        if donor_res.fallback or not donor_res.coeffs_v:
            # The donor file exists but its calibration table could not be
            # read (corrupt, or a run still being written). Fail loudly: an
            # override the user explicitly asked for must never quietly
            # degrade to this run's own uncorrected calibration.
            raise CalibrationError(
                f"Calibration for {os.path.basename(path)} borrows from "
                f"{os.path.basename(donor)}, but that run carries no readable "
                f"calibration table.")

        # Show the borrowed polynomial against *this* run's points, so the
        # diagnostic can reveal a donor that does not actually describe
        # this run.
        ref_fit = None
        n_points = 0
        try:
            set_v, read_raw = read_cal_points(path)
            n_points = int(len(set_v))
            if n_points:
                ref_fit = _fit_against(set_v, read_raw, donor_res.coeffs_v,
                                       donor_res.order, vacc_div)
        except (OSError, KeyError, ValueError):
            pass
        return CalibrationResult(
            coeffs_v=donor_res.coeffs_v, errs_v=donor_res.errs_v,
            order=donor_res.order, mode="borrow", donor=donor, note=note,
            fit=ref_fit, n_points=n_points,
            # NOT donor_res.excluded: those index the *donor's* point table,
            # which need not even be the same length as this run's. This run
            # dropped none of its own points -- it took someone else's
            # polynomial whole.
            excluded=(),
            donor_excluded=donor_res.excluded)

    # mode == "fit"
    set_v, read_raw = read_cal_points(path)
    if len(set_v) == 0:
        raise CalibrationError(
            f"{os.path.basename(path)} carries no calibration table, so "
            f"the requested calibration override cannot be applied.")
    ignore_intercept = bool(spec.get("ignore_intercept", False))
    auto = auto_outliers(
        set_v, read_raw, order,
        rule=spec.get("reject", "none"),
        sigma=spec.get("sigma", 2.0),
        iterative=spec.get("iterative", False),
        drop_first=spec.get("drop_first", 0),
        vacc_div=vacc_div,
        ignore_intercept=ignore_intercept)
    excluded = tuple(sorted(set(auto) | set(spec.get("excluded") or ())))
    fit = fit_calibration(set_v, read_raw, order, excluded, vacc_div,
                          ignore_intercept=ignore_intercept)
    return CalibrationResult(
        coeffs_v=fit.coeffs_v, errs_v=fit.errs_v, order=fit.order,
        mode="fit", excluded=fit.excluded, note=note, fit=fit,
        n_points=fit.n_points, ignore_intercept=ignore_intercept)


def _fit_against(set_v, read_raw, coeffs_v, order: int,
                 vacc_div: float) -> CalibrationFit:
    """Wrap a *given* polynomial as a CalibrationFit over these points.

    Used for the ``coeffs`` and ``borrow`` modes: the coefficients did
    not come from these points, but the residuals against them are
    exactly what tells the user whether the choice is sane.
    """
    x = np.asarray(set_v, dtype=float)
    y = np.asarray(read_raw, dtype=float) * float(vacc_div)
    resid = y - np.polyval(np.asarray(coeffs_v, float)[::-1], x)
    n = len(x)
    return CalibrationFit(
        coeffs_v=tuple(float(c) for c in coeffs_v),
        errs_v=tuple(float("nan") for _ in coeffs_v),
        order=int(order), excluded=(), residuals_v=resid,
        set_v=x, read_v=y,
        sigma_v=float(np.std(resid, ddof=1)) if n > 1 else 0.0,
        rms_v=float(np.sqrt(np.mean(resid ** 2))) if n else 0.0,
        max_abs_v=float(np.max(np.abs(resid))) if n else 0.0,
    )


# ── Flagging (the default policy) ────────────────────────────────────

@dataclass(eq=False)
class CalibrationDiagnosis:
    """What the overview needs to triage one run at a glance.

    ``impact_v`` is the headline number: the largest voltage disagreement,
    anywhere in the sweep, between the calibration currently in force and
    the one you would get by dropping the suspect points. That -- not the
    residual scatter -- is what actually moves a centroid, and it is what
    ``flagged`` keys off.
    """

    path: str
    run_number: str
    n_points: int
    sigma_v: float
    max_abs_v: float
    suspect: Tuple[int, ...]
    impact_v: float
    has_override: bool
    spec_summary: str
    error: str = ""
    #: The user has seen this warning and chosen to leave the run alone.
    #: A warning that keeps nagging after it has been read is one people
    #: learn to click past, and then it is worth nothing when it matters.
    acknowledged: bool = False

    @property
    def flagged(self) -> bool:
        return (bool(self.suspect)
                and self.impact_v > FLAG_MIN_IMPACT_V
                and not self.has_override
                and not self.acknowledged)


def calibration_shift_v(coeffs_a: Sequence[float], coeffs_b: Sequence[float],
                        x_lo: float, x_hi: float, samples: int = 64) -> float:
    """Largest |DV_cal| disagreement between two calibrations over a sweep.

    Both coefficient sets are ascending, volts convention, so the result
    is directly the voltage error the choice of calibration costs you --
    and multiplying by the run's dnu/dV (~18 MHz/V at 30 kV, A=51) turns
    it into the centroid shift, which is the only form a physicist can
    actually act on.
    """
    if not len(coeffs_a) or not len(coeffs_b):
        return 0.0
    xs = np.linspace(float(min(x_lo, x_hi)), float(max(x_lo, x_hi)),
                     max(int(samples), 2))
    a = np.polyval(np.asarray(coeffs_a, float)[::-1], xs)
    b = np.polyval(np.asarray(coeffs_b, float)[::-1], xs)
    return float(np.max(np.abs(a - b)))


def shift_mhz_over_scan(coeffs_a: Sequence[float], coeffs_b: Sequence[float],
                        dv_lo: float, dv_hi: float, *,
                        cooler_v: float, laser_cm: float, mass_amu: float,
                        harmonic: int = 2, samples: int = 64):
    """Centroid shift, in MHz, of calibration A relative to B across a sweep.

    Returns ``(dv, dnu_mhz)``: the sampled DAC axis and the frequency
    difference at each point. This is the number that decides whether a
    calibration problem is worth acting on -- volts of residual mean
    nothing to a spectroscopist, and MHz mean everything.

    Note the difference is generally *not* constant across the sweep: an
    offset error and a gain error partially cancel somewhere in the middle
    and add at the edges, so a bad calibration tilts the frequency axis
    rather than translating it, moving peaks by different amounts
    depending on where they sit in the scan.

    Reuses the same Doppler projection the merge path uses
    (``cls_estimations.doppler.voltage_to_frequency_array``), so the number
    quoted here is the one the analysis would actually produce.
    """
    from cls_estimations.doppler import voltage_to_frequency_array

    dv = np.linspace(float(min(dv_lo, dv_hi)), float(max(dv_lo, dv_hi)),
                     max(int(samples), 2))
    dv_cal_a = np.polyval(np.asarray(coeffs_a, float)[::-1], dv)
    dv_cal_b = np.polyval(np.asarray(coeffs_b, float)[::-1], dv)
    kw = dict(cooler_v=float(cooler_v), laser_cm1=float(laser_cm),
              mass_amu=float(mass_amu), harmonic=int(harmonic))
    nu_a = voltage_to_frequency_array(dv_cal_a, **kw)
    nu_b = voltage_to_frequency_array(dv_cal_b, **kw)
    return dv, np.asarray(nu_a, float) - np.asarray(nu_b, float)


def suspect_outliers(set_v, read_raw, order: int = DEFAULT_ORDER,
                     vacc_div: float = DEFAULT_VACC_DIV) -> Tuple[int, ...]:
    """Points a run *would* drop under the robust rule, without dropping them.

    Statistical nomination only -- see ``FLAG_MIN_IMPACT_V`` for why a
    caller must also check that dropping them would move the voltage
    before putting a warning in front of the user. Uses the
    iterative/MAD rule because the single-pass 2-sigma test misses
    exactly the clustered start-of-sweep outliers we are hunting.
    """
    try:
        return auto_outliers(set_v, read_raw, order, rule="sigma",
                             sigma=FLAG_SIGMA, iterative=True,
                             vacc_div=vacc_div)
    except CalibrationError:
        return ()


def diagnose(path: str, calibrations: Optional[Dict[str, dict]] = None, *,
             run_number: str = "", cal_order: int = DEFAULT_ORDER,
             vacc_div: float = DEFAULT_VACC_DIV,
             acknowledged: bool = False) -> CalibrationDiagnosis:
    """Summarize one run for the overview table. Never raises.

    ``acknowledged`` is passed in rather than read from the registry so this
    stays a pure function -- the fit subprocess imports this module, and a
    singleton lookup there would find an empty registry.
    """
    calibrations = _canonical_map(calibrations)
    spec = normalize_spec(calibrations.get(canonical_path(path)))

    def _broken(n, exc):
        return CalibrationDiagnosis(
            path=path, run_number=run_number, n_points=n,
            sigma_v=float("nan"), max_abs_v=float("nan"), suspect=(),
            impact_v=0.0, has_override=spec is not None,
            spec_summary=describe_spec(spec), error=str(exc),
            acknowledged=acknowledged)

    try:
        set_v, read_raw = read_cal_points(path)
    except (OSError, KeyError, ValueError) as exc:
        return _broken(0, exc)

    try:
        res = resolve_calibration(path, calibrations, cal_order=cal_order,
                                  vacc_div=vacc_div)
    except CalibrationError as exc:
        return _broken(int(len(set_v)), exc)

    sigma = res.fit.sigma_v if res.fit is not None else float("nan")
    max_abs = res.fit.max_abs_v if res.fit is not None else float("nan")

    # What would dropping the suspects actually buy? Compare the
    # calibration in force against the one without them, across the sweep.
    suspect = suspect_outliers(set_v, read_raw, cal_order, vacc_div)
    impact = 0.0
    if suspect and len(set_v):
        try:
            trimmed = fit_calibration(set_v, read_raw, cal_order, suspect,
                                      vacc_div)
            impact = calibration_shift_v(res.coeffs_v, trimmed.coeffs_v,
                                         float(np.min(set_v)),
                                         float(np.max(set_v)))
        except CalibrationError:
            impact = 0.0

    return CalibrationDiagnosis(
        path=path, run_number=run_number, n_points=int(len(set_v)),
        sigma_v=sigma, max_abs_v=max_abs, suspect=suspect, impact_v=impact,
        has_override=spec is not None, spec_summary=describe_spec(spec),
        acknowledged=acknowledged)


# ── Applying to a clstools data object ───────────────────────────────

def apply_calibration(data, path: str,
                      calibrations: Optional[Dict[str, dict]] = None, *,
                      cal_order: int = DEFAULT_ORDER) -> CalibrationResult:
    """Overwrite ``data``'s calibration in place. Call *before*
    ``Compute_Voltages`` -- that is the entire contract.

    Writes ``Cal`` (raw convention, ascending), ``Cal_err``
    (``sqrt(diag(cov))``, not clstools' variance), ``Cal_order``,
    ``Cal_df`` and ``Dropped_calibration_points``, and stashes the
    result on ``data.CalibrationInfo`` for the audit trail.

    ``Cal_df`` keeps clstools' mixed-unit column convention (``Set`` in
    volts, ``Read`` in monitor units) so that existing readers -- notably
    the merge provenance at ``merge.py:307`` -- keep working, and adds
    ``Residual_V`` and ``Excluded`` columns.
    """
    vacc_div = float(getattr(data, "VAccDiv", DEFAULT_VACC_DIV)
                     or DEFAULT_VACC_DIV)
    res = resolve_calibration(path, calibrations, cal_order=cal_order,
                              vacc_div=vacc_div)

    if res.fallback:
        # No usable points: leave clstools' own values alone.
        # Still define Dropped_calibration_points -- upstream only assigns it
        # inside its `if filter_calibration:` branch, so with the filter off
        # (which is how we load) the attribute would otherwise not exist and
        # clstools' own Info() would raise AttributeError.
        if not hasattr(data, "Dropped_calibration_points"):
            data.Dropped_calibration_points = None
        data.CalibrationInfo = res
        return res

    data.Cal = list(res.coeffs_raw(vacc_div))
    data.Cal_err = [e / vacc_div for e in res.errs_v]
    data.Cal_order = int(res.order)

    fit = res.fit
    if fit is not None and fit.n_points:
        import pandas as pd

        coeffs_raw = np.asarray(res.coeffs_raw(vacc_div), dtype=float)
        df = pd.DataFrame({
            "Set": fit.set_v,
            "Read": fit.read_v / vacc_div,
        })
        df["Fit"] = np.polyval(coeffs_raw[::-1], fit.set_v)
        df["Residual"] = df["Read"] - df["Fit"]
        df["Residual_V"] = fit.residuals_v
        df["Excluded"] = ~fit.used_mask
        data.Cal_df = df
        data.Dropped_calibration_points = (
            [float(v) for v in fit.set_v[list(res.excluded)]]
            if res.excluded else None)
    else:
        data.Dropped_calibration_points = None

    data.CalibrationInfo = res
    return res


def _load_run_supports_filter(data) -> bool:
    """Does this clstools' ``Load_Run`` take ``filter_calibration``?

    Upstream ``c0334c0`` added an on-by-default 2-sigma cut; the commit
    pinned in ``uv.lock`` predates it and rejects the kwarg outright. We
    have to run against both.
    """
    try:
        import inspect
        return "filter_calibration" in inspect.signature(
            data.Load_Run).parameters
    except (TypeError, ValueError):
        return False


def load_run_calibrated(data, path: str,
                        calibrations: Optional[Dict[str, dict]] = None, *,
                        cal_order: int = DEFAULT_ORDER,
                        **load_kwargs) -> CalibrationResult:
    """``Load_Run`` followed immediately by ``apply_calibration``.

    The single entry point every load path in the app goes through, so
    that no code can accidentally compute voltages from the calibration
    clstools happened to fit. Extra kwargs are forwarded to ``Load_Run``.

    Where the installed clstools supports it we switch its own outlier
    filter *off*. Not for the physics -- ``apply_calibration`` overwrites
    ``Cal`` either way, which is what makes the app's numbers independent
    of the installed version -- but because a filtering ``Load_Run``
    prints "3 points removed" to stdout, and with no override in force the
    app has in fact removed nothing. A warning that contradicts what the
    app actually did is worse than no warning at all. It also skips a
    polyfit whose result we immediately discard.
    """
    if _load_run_supports_filter(data):
        load_kwargs.setdefault("filter_calibration", False)
    data.Load_Run(path, cal_order=cal_order, **load_kwargs)
    return apply_calibration(data, path, calibrations, cal_order=cal_order)


# ── Registry ─────────────────────────────────────────────────────────

class CalibrationRegistry(QObject):
    """Process-wide map of file path -> calibration spec.

    Accessed via ``get_registry()`` from any tab. Mutations emit
    ``calibrations_changed`` so open dialogs, per-entry badges and
    cached spectra refresh in lock-step. Serializes under the project
    YAML's top-level ``calibrations`` key.

    A calibration override is a property of the *file*, which is why it
    lives here rather than on an analysis block's file entry (unlike a
    binning override, which is a property of the analysis).
    """

    calibrations_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._specs: Dict[str, dict] = {}
        # Runs whose calibration warning the user has read and dismissed.
        # Deliberately NOT part of ``to_dict``: that map is the calibration
        # itself, and it is what gets handed to the fit subprocess. Whether a
        # human has looked at a warning has no business influencing a fit.
        self._acked: set[str] = set()

    # ── Lookups ────────────────────────────────────────────

    def get(self, path: str) -> Optional[dict]:
        """The spec for ``path``, or ``None`` for the file default.

        Returns a copy: callers may mutate it freely.
        """
        spec = self._specs.get(canonical_path(path))
        return dict(spec) if spec else None

    def has(self, path: str) -> bool:
        return canonical_path(path) in self._specs

    def all_paths(self):
        return list(self._specs.keys())

    # ── Mutations ──────────────────────────────────────────

    def set(self, path: str, spec: Optional[dict]):
        """Install (or, with a ``None``/inert spec, remove) an override."""
        key = canonical_path(path)
        norm = normalize_spec(spec)
        if norm is None:
            self._specs.pop(key, None)
        else:
            self._specs[key] = norm
        self.calibrations_changed.emit()

    def set_many(self, mapping: Dict[str, Optional[dict]]):
        """Apply several overrides, emitting ``calibrations_changed`` once.

        Subscribers react to the signal by *reloading* the affected runs --
        a calibration rewrites every event's voltage, so unlike a scan
        filter it cannot be absorbed by invalidating a binned cache. Firing
        per-file while applying one calibration to fifty checked runs would
        therefore trigger fifty full reload passes; hence the batch.
        """
        if not mapping:
            return
        for path, spec in mapping.items():
            key = canonical_path(path)
            norm = normalize_spec(spec)
            if norm is None:
                self._specs.pop(key, None)
            else:
                self._specs[key] = norm
        self.calibrations_changed.emit()

    def clear(self, path: str):
        self.set(path, None)

    def clear_all(self):
        self._specs.clear()
        self._acked.clear()
        self.calibrations_changed.emit()

    # ── Acknowledgements ───────────────────────────────────

    def acknowledge(self, path: str):
        """The user has read this run's calibration warning; stop nagging."""
        key = canonical_path(path)
        if key not in self._acked:
            self._acked.add(key)
            self.calibrations_changed.emit()

    def unacknowledge(self, path: str):
        key = canonical_path(path)
        if key in self._acked:
            self._acked.discard(key)
            self.calibrations_changed.emit()

    def is_acknowledged(self, path: str) -> bool:
        return canonical_path(path) in self._acked

    def acks_to_list(self) -> list:
        """Serialize under the YAML's ``calibration_acks`` key."""
        return sorted(self._acked)

    def acks_from_list(self, paths) -> None:
        self._acked.clear()
        if isinstance(paths, (list, tuple, set)):
            for p in paths:
                if isinstance(p, str) and p:
                    self._acked.add(canonical_path(p))
        self.calibrations_changed.emit()

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize as ``{path: spec}``.

        Doubles as the snapshot handed to the fit worker, which cannot
        reach this singleton.
        """
        return {k: dict(v) for k, v in sorted(self._specs.items())}

    def from_dict(self, d: Optional[dict]):
        """Replace the whole map. Malformed entries are dropped, since the
        YAML may have been hand-edited."""
        self._specs.clear()
        if isinstance(d, dict):
            for path, spec in d.items():
                norm = normalize_spec(spec)
                if norm is not None:
                    self._specs[canonical_path(path)] = norm
        self.calibrations_changed.emit()


# ── Module-level singleton plumbing ──────────────────────────────────
#
# Same shape as gui.scan_filter: tabs reach the registry through
# ``get_registry()`` rather than a MainWindow attribute, so Pre-Analysis
# dialogs and headless tests can subscribe without seeing the window.
# Tests that need isolation call ``set_registry(CalibrationRegistry())``.

_REGISTRY: Optional[CalibrationRegistry] = None


def get_registry() -> CalibrationRegistry:
    """Return the process-wide registry, creating one on first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = CalibrationRegistry()
    return _REGISTRY


def set_registry(registry: Optional[CalibrationRegistry]) -> None:
    """Install (or reset) the singleton. Mainly used by tests."""
    global _REGISTRY
    _REGISTRY = registry
