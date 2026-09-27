"""Cooler-voltage offset calibration from Yb hyperfine constants.

The numbers here are Arda's own published calibration report ("Cooler-
voltage calibration via A_l(171Yb+) and A_l(173Yb+)", May 2026): its
Table 1 (per-run lines and the offset where each run reproduces the
literature A), Table 2 (per-isotope weighted means) and its
recommended dV* = -29.27 +/- 1.70 V. Reproducing a result that was
produced independently, outside DENIS, is the only real check that
this module implements the intended method.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_cooler_calibration.py -q
"""
import math
import unittest

import numpy as np

from cls_estimations.cooler_calibration import (
    LITERATURE_A, LineFit, ScanPoint, calibrate, fit_line, intersect,
    weighted_mean,
)

#: Report Table 1: run, isotope, b (MHz/V), sigma_b, a - A_lit (MHz),
#: sigma, and the dV* / sigma it printed.
REPORT_TABLE_1 = [
    ("7547", "171Yb", -0.2097, 0.0268, -6.466, 0.741, -30.83, 5.30),
    ("7507", "171Yb", -0.2083, 0.0170, -6.827, 0.470, -32.78, 3.50),
    ("7509", "171Yb", -0.2098, 0.0179, -7.487, 0.497, -35.69, 3.87),
    ("7502", "173Yb", +0.0591, 0.0016, +0.822, 0.044, -13.92, 0.84),
    ("7548", "173Yb", +0.0572, 0.0112, +1.078, 0.311, -18.85, 6.58),
]


def _published_line(b, sb, a, sa):
    """A LineFit standing in for one of the report's rows. The offset
    grid was symmetric about zero, so intercept and slope are
    essentially uncorrelated and a diagonal covariance is right."""
    return LineFit(intercept=a, slope=b, cov=np.diag([sa ** 2, sb ** 2]),
                   n=15, chi2=0.0, ndf=13)


def _points_on(line_zero, slope, *, run, isotope, a_lit, dvs=None,
               sigma=0.05):
    """Scan points lying exactly on ``(A - A_lit) = slope (dV - zero)``."""
    dvs = np.linspace(-50, 50, 15) if dvs is None else np.asarray(dvs)
    return [ScanPoint(run=run, isotope=isotope, dv=float(dv),
                      value=a_lit + slope * (dv - line_zero),
                      sigma=sigma) for dv in dvs]


class ReportTable1Tests(unittest.TestCase):
    """Per run: where the line crosses the literature value."""

    def test_every_run_reproduces_its_published_offset(self):
        for run, _iso, b, sb, a, sa, dv_p, sig_p in REPORT_TABLE_1:
            with self.subTest(run=run):
                dv, sig = _published_line(b, sb, a, sa).root()
                self.assertAlmostEqual(dv, dv_p, delta=0.02)
                self.assertAlmostEqual(sig, sig_p, delta=0.02)

    def test_the_error_is_not_just_the_intercept_term(self):
        """Both the intercept and the slope are uncertain, and for
        run 7547 they contribute about equally (3.5 and 3.9 V). Taking
        only the first would quote 3.5 instead of 5.3."""
        _run, _iso, b, sb, a, sa, _dv, sig_p = REPORT_TABLE_1[0]
        _dv, sig = _published_line(b, sb, a, sa).root()
        self.assertAlmostEqual(sig, sig_p, delta=0.02)
        self.assertGreater(sig, abs(sa / b) * 1.3)

    def test_a_flat_line_has_no_solution(self):
        """A parameter that does not respond to the cooler voltage
        cannot calibrate it."""
        self.assertIsNone(
            LineFit(0.5, 0.0, np.eye(2), 5, 0.0, 3).root())


class ReportTable2Tests(unittest.TestCase):

    def test_the_per_isotope_weighted_means(self):
        for iso, want, want_sig in (("171Yb", -33.46, 2.33),
                                    ("173Yb", -14.00, 0.83)):
            rows = [r for r in REPORT_TABLE_1 if r[1] == iso]
            dv, sig = zip(*[_published_line(b, sb, a, sa).root()
                            for _r, _i, b, sb, a, sa, _d, _s in rows])
            m, s = weighted_mean(dv, sig)
            with self.subTest(isotope=iso):
                self.assertAlmostEqual(m, want, delta=0.02)
                self.assertAlmostEqual(s, want_sig, delta=0.02)

    def test_the_precise_run_dominates(self):
        """7502 (sigma 0.84) against 7548 (6.58): the mean must sit
        next to 7502, not midway."""
        m, _s = weighted_mean([-13.92, -18.85], [0.84, 6.58])
        self.assertLess(abs(m - (-13.92)), abs(m - (-18.85)))


class IntersectionTests(unittest.TestCase):

    def test_two_lines_cross_where_the_report_says(self):
        """With the report's isotope slopes and zero crossings, the
        lines meet at -29.25 V -- its recommended -29.27."""
        a = fit_line([-50, 50], [-0.21 * (-50 + 33.46),
                                 -0.21 * (50 + 33.46)])
        b = fit_line([-50, 50], [0.058 * (-50 + 14.0),
                                 0.058 * (50 + 14.0)])
        dv, _sig = intersect(a, b)
        self.assertAlmostEqual(dv, -29.25, delta=0.05)

    def test_parallel_lines_never_cross(self):
        a = fit_line([0, 10], [0.0, 1.0])
        b = fit_line([0, 10], [5.0, 6.0])
        self.assertIsNone(intersect(a, b))

    def test_nearly_parallel_lines_are_badly_determined(self):
        """The crossing of two similar slopes is a long lever; its
        error must blow up rather than look precise."""
        pts_a = _points_on(-30.0, -0.20, run="a", isotope="X",
                           a_lit=0.0, sigma=0.5)
        steep = fit_line([p.dv for p in pts_a], [p.value for p in pts_a],
                         [p.sigma for p in pts_a])
        close = fit_line([p.dv for p in pts_a],
                         [-0.19 * (p.dv + 12.0) for p in pts_a],
                         [p.sigma for p in pts_a])
        far = fit_line([p.dv for p in pts_a],
                       [0.20 * (p.dv + 12.0) for p in pts_a],
                       [p.sigma for p in pts_a])
        self.assertGreater(intersect(steep, close)[1],
                           intersect(steep, far)[1] * 5)


class LineFitTests(unittest.TestCase):

    def test_it_recovers_a_known_line(self):
        line = fit_line([-50, -25, 0, 25, 50],
                        [-6.0, -3.5, -1.0, 1.5, 4.0])
        self.assertAlmostEqual(line.slope, 0.1, places=9)
        self.assertAlmostEqual(line.intercept, -1.0, places=9)

    def test_a_loose_point_does_not_drag_the_fit(self):
        x = [-50, -25, 0, 25, 50]
        y = [-6.0, -3.5, -1.0, 1.5, 40.0]           # last one is wild
        tight = fit_line(x, y, [0.1, 0.1, 0.1, 0.1, 100.0])
        self.assertAlmostEqual(tight.slope, 0.1, delta=0.005)

    def test_a_zero_error_is_weighted_not_dropped(self):
        """A parameter pinned at a bound comes back with sigma = 0.
        Dropping it would quietly change which runs the calibration
        rests on."""
        line = fit_line([-50, 0, 50], [-6.0, -1.0, 4.0],
                        [0.0, 0.1, 0.1])
        self.assertEqual(line.n, 3)
        self.assertAlmostEqual(line.slope, 0.1, delta=1e-6)

    def test_identical_offsets_cannot_make_a_line(self):
        with self.assertRaises(ValueError):
            fit_line([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])

    def test_the_band_is_narrowest_in_the_middle_of_the_data(self):
        pts = _points_on(-30.0, -0.2, run="r", isotope="X", a_lit=0.0)
        line = fit_line([p.dv for p in pts], [p.value for p in pts],
                        [p.sigma for p in pts])
        self.assertLess(float(line.sigma_at(0.0)),
                        float(line.sigma_at(200.0)))


class CalibrateTests(unittest.TestCase):

    def _points(self):
        a171, a173 = LITERATURE_A["171Yb"], LITERATURE_A["173Yb"]
        pts = []
        for run in ("7547", "7507", "7509"):
            pts += _points_on(-33.46, -0.21, run=run, isotope="171Yb",
                              a_lit=a171)
        for run in ("7502", "7548"):
            pts += _points_on(-14.00, 0.058, run=run, isotope="173Yb",
                              a_lit=a173)
        return pts

    def test_end_to_end(self):
        res = calibrate(self._points())
        self.assertEqual([s.isotope for s in res.isotopes],
                         ["171Yb", "173Yb"])
        self.assertAlmostEqual(res.dv, -29.25, delta=0.05)
        # The systematic scan runs between the two zero crossings.
        self.assertAlmostEqual(res.scan_lo, -33.46, delta=0.01)
        self.assertAlmostEqual(res.scan_hi, -14.00, delta=0.01)
        self.assertAlmostEqual(res.scan_width, 19.46, delta=0.02)
        # ...and the calibration sits inside it.
        self.assertLess(res.scan_lo, res.dv)
        self.assertLess(res.dv, res.scan_hi)

    def test_every_run_is_solved_too(self):
        res = calibrate(self._points())
        by_iso = {s.isotope: s for s in res.isotopes}
        self.assertEqual(len(by_iso["171Yb"].runs), 3)
        for r in by_iso["171Yb"].runs:
            self.assertAlmostEqual(r.dv, -33.46, delta=0.01)
        self.assertAlmostEqual(by_iso["171Yb"].mean_dv, -33.46,
                               delta=0.01)

    def test_the_sign_of_the_173_constant_matters(self):
        """A(173Yb+) is negative. Handing over its magnitude flips the
        slope of that isotope's line and moves the crossing."""
        good = calibrate(self._points())
        bad = calibrate(self._points(),
                        literature={"171Yb": LITERATURE_A["171Yb"],
                                    "173Yb": -LITERATURE_A["173Yb"]})
        self.assertNotAlmostEqual(good.dv, bad.dv, delta=1.0)

    def test_one_isotope_alone_cannot_calibrate(self):
        pts = [p for p in self._points() if p.isotope == "171Yb"]
        res = calibrate(pts)
        self.assertIsNone(res.dv)
        self.assertIn("two isotopes", res.note)
        # ...but its own zero crossing is still reported.
        self.assertAlmostEqual(res.isotopes[0].zero, -33.46, delta=0.01)

    def test_a_run_with_a_single_offset_is_skipped(self):
        pts = self._points()
        pts.append(ScanPoint("9999", "171Yb", 0.0, 1.0, 0.1))
        res = calibrate(pts)
        runs = [r.run for s in res.isotopes for r in s.runs]
        self.assertNotIn("9999", runs)

    def test_no_points_is_not_a_crash(self):
        res = calibrate([])
        self.assertIsNone(res.dv)
        self.assertEqual(res.isotopes, [])

    def test_literature_defaults_match_the_paper(self):
        """de Groote, Atoms 2024, 12, 60, Table 2."""
        self.assertAlmostEqual(LITERATURE_A["171Yb"],
                               12642.8121184682, places=9)
        self.assertAlmostEqual(LITERATURE_A["173Yb"],
                               -3497.24007985, places=7)
        self.assertLess(LITERATURE_A["173Yb"], 0)


if __name__ == "__main__":
    unittest.main()
