"""The arithmetic that turns an offset scan into a systematic error.

Pure numbers, no Qt and no fitting: every expected value here is one
that can be worked out by hand, which is the point of keeping this
module separate from the scan that drives the fits.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_systematics.py -q
"""
import math
import unittest

from cls_estimations.systematics import (
    DEFINITIONS, FULL_WIDTH, HALF_WIDTH, MAX_DEV, RMS_DEV, STD_DEV,
    ParameterBand, ScanValue, mean_over_runs, order_outward,
    quality_flag, scan_offsets, seed_from_history, shift_bounds,
    summarise, trend_outliers,
)


def line_band(slope=2.0, intercept=100.0, sigma=0.5,
              dvs=(-30, -25, -20, -15, -10), **kw):
    """A band whose values sit exactly on ``intercept + slope * dv``."""
    pts = [ScanValue(float(d), intercept + slope * d, sigma) for d in dvs]
    kw.setdefault("baseline", intercept)
    kw.setdefault("baseline_sigma", 0.4)
    return ParameterBand("70Ge", "7947", "centroid", pts, **kw)


class BandStatisticsTests(unittest.TestCase):
    """The band the values span across the scanned offsets."""

    def test_the_spread_of_a_straight_line(self):
        b = line_band()
        self.assertAlmostEqual(b.lo, 40.0)      # 100 + 2*(-30)
        self.assertAlmostEqual(b.hi, 80.0)      # 100 + 2*(-10)
        self.assertAlmostEqual(b.width, 40.0)
        self.assertAlmostEqual(b.half_width, 20.0)

    def test_the_standard_deviation(self):
        b = line_band(dvs=(-30, -20, -10))      # values 40, 60, 80
        self.assertAlmostEqual(b.std, 20.0)

    def test_one_point_has_no_spread(self):
        """Zero, not NaN: a single fit genuinely spans nothing, which
        is different from not knowing."""
        b = line_band(dvs=(-20,))
        self.assertEqual(b.n, 1)
        self.assertEqual(b.std, 0.0)
        self.assertEqual(b.width, 0.0)

    def test_an_empty_band(self):
        b = ParameterBand("70Ge", "7947", "centroid", [])
        for v in (b.lo, b.hi, b.width, b.half_width, b.std):
            self.assertTrue(math.isnan(v))

    def test_deviation_from_the_baseline(self):
        b = line_band(dvs=(-30, -20, -10))      # 40, 60, 80; base 100
        self.assertAlmostEqual(b.max_dev, 60.0)
        self.assertAlmostEqual(
            b.rms_dev, math.sqrt((60 ** 2 + 40 ** 2 + 20 ** 2) / 3))

    def test_deviation_needs_a_baseline(self):
        b = line_band(baseline=float("nan"))
        self.assertTrue(math.isnan(b.max_dev))
        self.assertTrue(math.isnan(b.rms_dev))

    def test_an_excluded_step_takes_no_part(self):
        """The user unticks a step whose fit went somewhere else; the
        band must forget it, not just grey it out."""
        b = line_band()
        b.points.append(ScanValue(-5.0, 1e4, 0.5, include=False))
        self.assertEqual(b.n, 5)
        self.assertAlmostEqual(b.hi, 80.0)

    def test_a_non_finite_value_takes_no_part(self):
        b = line_band()
        b.points.append(ScanValue(-5.0, float("nan"), 0.5))
        self.assertEqual(b.n, 5)
        self.assertAlmostEqual(b.width, 40.0)

    def test_the_key_names_the_band(self):
        self.assertEqual(line_band().key, "70Ge/7947/centroid")
        self.assertEqual(
            ParameterBand("70Ge", "", "centroid").key,
            "70Ge/<mean>/centroid")


class DefinitionTests(unittest.TestCase):
    """Which number gets quoted is the user's choice (Arda, 2026-09-24),
    so every definition has to be available from the same band."""

    def setUp(self):
        self.b = line_band(dvs=(-30, -20, -10))   # 40, 60, 80

    def test_every_definition_is_offered(self):
        self.assertEqual(len(DEFINITIONS), 5)
        for d in DEFINITIONS:
            self.assertTrue(math.isfinite(self.b.systematic(d)), d)

    def test_half_width(self):
        self.assertAlmostEqual(self.b.systematic(HALF_WIDTH), 20.0)

    def test_full_width(self):
        self.assertAlmostEqual(self.b.systematic(FULL_WIDTH), 40.0)

    def test_std(self):
        self.assertAlmostEqual(self.b.systematic(STD_DEV), 20.0)

    def test_max_deviation(self):
        self.assertAlmostEqual(self.b.systematic(MAX_DEV), 60.0)

    def test_rms_deviation(self):
        self.assertAlmostEqual(self.b.systematic(RMS_DEV),
                               math.sqrt((3600 + 1600 + 400) / 3))

    def test_an_unknown_definition_falls_back(self):
        """A save file from a later version must not make the number
        disappear."""
        self.assertAlmostEqual(self.b.systematic("something else"), 20.0)

    def test_total_is_the_quadrature_sum(self):
        b = line_band(dvs=(-30, -20, -10), baseline_sigma=15.0)
        self.assertAlmostEqual(b.total(HALF_WIDTH),
                               math.hypot(15.0, 20.0))

    def test_total_without_a_statistical_error(self):
        b = line_band(dvs=(-30, -20, -10), baseline_sigma=float("nan"))
        self.assertAlmostEqual(b.total(HALF_WIDTH), 20.0)


class SensitivityTests(unittest.TestCase):
    """How fast the parameter moves with the assumed offset -- the
    number to hold against the analytic Doppler derivative."""

    def test_the_slope_of_a_straight_line(self):
        self.assertAlmostEqual(line_band(slope=2.0).slope, 2.0)
        self.assertAlmostEqual(line_band(slope=-13.0).slope, -13.0)

    def test_the_slope_has_an_error(self):
        b = line_band(sigma=0.5)
        self.assertGreater(b.slope_sigma, 0.0)
        self.assertLess(b.slope_sigma, 0.1)

    def test_a_single_point_has_no_trend(self):
        self.assertIsNone(line_band(dvs=(-20,)).trend)
        self.assertTrue(math.isnan(line_band(dvs=(-20,)).slope))


class OutlierTests(unittest.TestCase):
    """A step that did not converge usually lands off the line."""

    def test_a_clean_scan_flags_nothing(self):
        self.assertEqual(trend_outliers(line_band(dvs=range(-30, -5, 5))),
                         [])

    def test_a_step_that_went_elsewhere_is_found(self):
        b = line_band(dvs=(-30, -25, -20, -15, -10))
        b.points[2] = ScanValue(-20.0, 500.0, 0.5)
        self.assertIn(2, trend_outliers(b))

    def test_three_points_are_not_enough_to_judge(self):
        """Three points and a two-parameter line leave one degree of
        freedom: anything can be made to look consistent."""
        b = line_band(dvs=(-30, -20, -10))
        b.points[1] = ScanValue(-20.0, 500.0, 0.5)
        self.assertEqual(trend_outliers(b), [])

    def test_an_excluded_point_is_not_flagged_again(self):
        b = line_band()
        b.points[2] = ScanValue(-20.0, 500.0, 0.5, include=False)
        self.assertEqual(trend_outliers(b), [])


class QualityFlagTests(unittest.TestCase):
    """Saying where to look, not deciding for the user."""

    def test_a_good_step_is_not_flagged(self):
        self.assertEqual(
            quality_flag(sigma=0.5, baseline_sigma=0.4, redchi=1.1,
                         baseline_redchi=1.0), "")

    def test_a_failed_fit(self):
        self.assertEqual(quality_flag(success=False), "fit failed")

    def test_a_parameter_that_came_back_without_an_error(self):
        """Usually a parameter pinned to a bound."""
        self.assertEqual(quality_flag(sigma=float("nan")), "no error")
        self.assertEqual(quality_flag(sigma=0.0), "no error")

    def test_an_error_that_blew_up(self):
        self.assertTrue(
            quality_flag(sigma=9.0, baseline_sigma=0.4).startswith("error"))

    def test_chi_square_that_moved(self):
        self.assertTrue(
            quality_flag(sigma=0.5, baseline_sigma=0.4, redchi=8.0,
                         baseline_redchi=1.0).startswith("chi2"))
        self.assertTrue(
            quality_flag(sigma=0.5, baseline_sigma=0.4, redchi=0.1,
                         baseline_redchi=1.0).startswith("chi2"))

    def test_a_missing_baseline_chi_square_is_not_a_flag(self):
        self.assertEqual(
            quality_flag(sigma=0.5, baseline_sigma=0.4, redchi=8.0), "")


class MeanOverRunsTests(unittest.TestCase):
    """What gets published is the average over runs, and its band is
    not the average of the per-run bands: the runs move together."""

    def _bands(self):
        a = line_band(slope=2.0, intercept=100.0, sigma=1.0,
                      dvs=(-30, -20, -10), baseline_sigma=1.0)
        b = ParameterBand(
            "70Ge", "7948", "centroid",
            [ScanValue(float(d), 104.0 + 2.0 * d, 1.0)
             for d in (-30, -20, -10)],
            baseline=104.0, baseline_sigma=1.0)
        return [a, b]

    def test_it_averages_at_each_offset(self):
        m = mean_over_runs(self._bands())
        self.assertEqual(m.run, "")
        self.assertEqual(m.n, 3)
        self.assertAlmostEqual(m.values[0], 42.0)     # (40 + 44)/2
        self.assertAlmostEqual(m.values[-1], 82.0)

    def test_the_band_of_the_mean_is_the_common_movement(self):
        """Both runs move 40 MHz across the scan, so their mean does
        too -- averaging does not shrink a systematic."""
        self.assertAlmostEqual(mean_over_runs(self._bands()).width, 40.0)

    def test_the_baseline_is_averaged_too(self):
        self.assertAlmostEqual(mean_over_runs(self._bands()).baseline,
                               102.0)

    def test_an_offset_only_one_run_reached(self):
        bands = self._bands()
        bands[1].points.append(ScanValue(-5.0, 94.0, 1.0))
        m = mean_over_runs(bands)
        self.assertEqual(m.n, 4)
        self.assertAlmostEqual(m.values[-1], 94.0)

    def test_excluded_points_stay_out_of_the_mean(self):
        bands = self._bands()
        bands[0].points[0] = ScanValue(-30.0, 1e4, 1.0, include=False)
        m = mean_over_runs(bands)
        self.assertAlmostEqual(m.values[0], 44.0)

    def test_nothing_to_average(self):
        self.assertIsNone(mean_over_runs([]))


class SummaryTests(unittest.TestCase):
    def test_one_row_per_band(self):
        rows = summarise([line_band(), None, line_band()])
        self.assertEqual(len(rows), 2)

    def test_a_row_carries_what_the_report_quotes(self):
        row = summarise([line_band(dvs=(-30, -20, -10))], HALF_WIDTH)[0]
        self.assertEqual(row["project"], "70Ge")
        self.assertEqual(row["run"], "7947")
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["systematic"], 20.0)
        self.assertAlmostEqual(row["band_width"], 40.0)
        self.assertAlmostEqual(row["slope"], 2.0)
        self.assertAlmostEqual(row["total"], math.hypot(0.4, 20.0))

    def test_a_mean_row_says_so(self):
        rows = summarise([mean_over_runs([line_band()])])
        self.assertEqual(rows[0]["run"], "<weighted mean>")


class ScanWalkTests(unittest.TestCase):
    """Which offsets, and in what order they are fitted."""

    def test_the_offsets_span_the_range(self):
        offs = scan_offsets(-33.46, -14.0, 5)
        self.assertEqual(len(offs), 5)
        self.assertAlmostEqual(offs[0], -33.46)
        self.assertAlmostEqual(offs[-1], -14.0)

    def test_a_range_given_backwards_still_works(self):
        self.assertEqual(scan_offsets(-14.0, -33.46, 5),
                         scan_offsets(-33.46, -14.0, 5))

    def test_a_range_needs_two_points_and_some_width(self):
        with self.assertRaises(ValueError):
            scan_offsets(-30, -10, 1)
        with self.assertRaises(ValueError):
            scan_offsets(-30, -30, 5)

    def test_the_walk_starts_at_the_baseline_and_works_outward(self):
        """Every step then has a neighbour, closer to the baseline,
        that has already been fitted."""
        order = order_outward([-33.46, -29.25, -25, -20, -14], -29.25)
        self.assertEqual(order[0], -29.25)
        self.assertEqual(order[-1], -14.0)

    def test_the_walk_alternates_sides(self):
        order = order_outward([-4, -2, 0, 2, 4], 0)
        self.assertEqual(order, [0.0, -2.0, 2.0, -4.0, 4.0])

    def test_the_walk_is_reproducible(self):
        a = order_outward([1, 2, 3], 2)
        b = order_outward([3, 1, 2], 2)
        self.assertEqual(a, b)


class SeedingTests(unittest.TestCase):
    """Where a step's fit starts.

    The centroid moves of order 10 MHz per volt, so at the end of a
    20 V scan the baseline value is hundreds of MHz away -- far outside
    a line width. Carrying the trend along is what makes these fits
    converge.
    """

    def test_nothing_to_go_on(self):
        self.assertIsNone(seed_from_history(-30, []))
        self.assertIsNone(seed_from_history(-30, None))

    def test_one_neighbour_is_used_as_is(self):
        self.assertAlmostEqual(
            seed_from_history(-30, [(-25, 50.0, 0.1)]), 50.0)

    def test_two_neighbours_carry_the_trend(self):
        """(-25, 50) and (-20, 100) is 10 per volt; at -30 that is 0."""
        self.assertAlmostEqual(
            seed_from_history(-30, [(-25, 50.0, 0.1), (-20, 100.0, 0.1)]),
            0.0)

    def test_the_two_NEAREST_neighbours_are_used(self):
        hist = [(-20, 100.0, 0.1), (-25, 50.0, 0.1), (0, 1e4, 0.1)]
        self.assertAlmostEqual(seed_from_history(-30, hist), 0.0)

    def test_a_parameter_that_is_not_moving_is_not_extrapolated(self):
        """Two points a fraction of their errors apart carry no trend;
        extrapolating them would amplify noise into the seed."""
        self.assertAlmostEqual(
            seed_from_history(-30, [(-25, 50.0, 5.0), (-20, 51.0, 5.0)]),
            50.0)

    def test_extrapolation_can_be_turned_off(self):
        self.assertAlmostEqual(
            seed_from_history(-30, [(-25, 50.0, 0.1), (-20, 100.0, 0.1)],
                              extrapolate=False),
            50.0)

    def test_history_need_not_be_sorted(self):
        hist = [(-20, 100.0, 0.1), (-25, 50.0, 0.1)]
        self.assertAlmostEqual(seed_from_history(-30, hist), 0.0)

    def test_a_broken_entry_is_ignored(self):
        hist = [(-25, float("nan"), 0.1), (-20, 100.0, 0.1)]
        self.assertAlmostEqual(seed_from_history(-30, hist), 100.0)

    def test_two_neighbours_at_the_same_offset(self):
        self.assertAlmostEqual(
            seed_from_history(-30, [(-25, 50.0, 0.1), (-25, 52.0, 0.1)]),
            50.0)


class BoundTests(unittest.TestCase):
    """A seed outside its own bounds is refused by the fitter, and
    clipping it silently pins the parameter and loses its error bar."""

    def test_a_window_that_still_holds_the_seed_does_not_move(self):
        """Those bounds are usually a physical limit -- a width that
        may not go negative -- not a leash on the value."""
        self.assertEqual(shift_bounds(10.0, 0.0, 20.0, 15.0), (0.0, 20.0))

    def test_a_window_travels_with_a_seed_that_left_it(self):
        self.assertEqual(shift_bounds(10.0, 0.0, 20.0, 25.0), (15.0, 35.0))

    def test_it_travels_downward_too(self):
        self.assertEqual(shift_bounds(10.0, 0.0, 20.0, -5.0),
                         (-15.0, 5.0))

    def test_the_window_keeps_its_width(self):
        lo, hi = shift_bounds(10.0, 0.0, 20.0, 25.0)
        self.assertAlmostEqual(hi - lo, 20.0)
        self.assertGreaterEqual(25.0, lo)
        self.assertLessEqual(25.0, hi)

    def test_an_open_bound_stays_open(self):
        self.assertEqual(shift_bounds(10.0, None, float("inf"), 1e6),
                         (None, float("inf")))

    def test_only_the_closed_side_moves(self):
        lo, hi = shift_bounds(10.0, 0.0, None, -5.0)
        self.assertEqual((lo, hi), (-15.0, None))

    def test_a_seed_that_is_not_a_number_changes_nothing(self):
        self.assertEqual(shift_bounds(10.0, 0.0, 20.0, float("nan")),
                         (0.0, 20.0))


if __name__ == "__main__":
    unittest.main()
