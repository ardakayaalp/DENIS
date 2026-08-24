"""Tests for the WP2 merge fixes (2026-06-02 code review).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers gui.analysis.merge._build_merge_warnings -- the pure helper that
surfaces:

* MERGE_REBINNED_APPROXIMATE: any multi-file merge re-bins per-file histograms
  rather than pooling raw events, so the centroid can be biased ~half a bin
  (previously silent; #1 merge-after-binning-double-rebin);
* VOLTAGE_MERGE_SETPOINT_SPREAD: a voltage merge of files with differing laser
  setpoint / beam energy (#6 voltage-merge-no-laser-guard);
* MANUAL_OFFSET_IGNORED_VOLTAGE_MERGE: MHz offsets on a voltage axis.

The yerr unification (#2 merged-yerr-divergence: sqrt(y+1) everywhere) is a
one-line formula swap exercised by the existing merge/projection suite; here
we assert the projection's yerr fallback uses sqrt(y+1).

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_wp2_merge -v

Depends on: gui.analysis.merge (_build_merge_warnings, SPREAD_*,
project_voltage_merge_to_frequency); cls_estimations.doppler.
"""

import unittest

import numpy as np

from gui.analysis.merge import (
    _build_merge_warnings, SPREAD_LASER_CM, SPREAD_COOLER_V,
)


def _runs(n, laser=12000.0, cooler=30000.0):
    return [{"laser_set": laser, "cooler_v": cooler} for _ in range(n)]


def _codes(warnings):
    return {w["code"] for w in warnings}


class MergeWarningTests(unittest.TestCase):
    def test_single_file_no_warnings(self):
        self.assertEqual(_build_merge_warnings(_runs(1), False, None), [])

    def test_multifile_emits_rebin_note(self):
        w = _build_merge_warnings(_runs(2), False, None)
        self.assertIn("MERGE_REBINNED_APPROXIMATE", _codes(w))
        # Info level, not a scary warning.
        note = next(x for x in w if x["code"] == "MERGE_REBINNED_APPROXIMATE")
        self.assertEqual(note["level"], "info")

    def test_voltage_merge_setpoint_spread_warns(self):
        runs = [{"laser_set": 12000.0, "cooler_v": 30000.0},
                {"laser_set": 12000.0 + 10 * SPREAD_LASER_CM,
                 "cooler_v": 30000.0}]
        w = _build_merge_warnings(runs, True, None)
        self.assertIn("VOLTAGE_MERGE_SETPOINT_SPREAD", _codes(w))

    def test_voltage_merge_matched_setpoints_no_spread_warning(self):
        w = _build_merge_warnings(_runs(2), True, None)
        self.assertNotIn("VOLTAGE_MERGE_SETPOINT_SPREAD", _codes(w))
        # ...but the re-bin note still appears for a multi-file merge.
        self.assertIn("MERGE_REBINNED_APPROXIMATE", _codes(w))

    def test_frequency_merge_ignores_setpoint_spread(self):
        # In frequency mode each file is Doppler-converted with its own
        # setpoint, so a spread is not a problem -> no spread warning.
        runs = [{"laser_set": 12000.0, "cooler_v": 30000.0},
                {"laser_set": 12000.0 + 100 * SPREAD_LASER_CM,
                 "cooler_v": 30000.0 + 100 * SPREAD_COOLER_V}]
        w = _build_merge_warnings(runs, False, None)
        self.assertNotIn("VOLTAGE_MERGE_SETPOINT_SPREAD", _codes(w))

    def test_manual_offset_on_voltage_merge_warns(self):
        w = _build_merge_warnings(_runs(2), True,
                                  {"a.asdf": 5.0, "b.asdf": 0.0})
        self.assertIn("MANUAL_OFFSET_IGNORED_VOLTAGE_MERGE", _codes(w))

    def test_manual_offset_all_zero_no_warning(self):
        w = _build_merge_warnings(_runs(2), True,
                                  {"a.asdf": 0.0, "b.asdf": 0.0})
        self.assertNotIn("MANUAL_OFFSET_IGNORED_VOLTAGE_MERGE", _codes(w))


class ProjectionYerrFallbackTests(unittest.TestCase):
    def test_projection_yerr_fallback_is_sqrt_y_plus_one(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency
        y = np.array([0.0, 1.0, 4.0, 9.0])
        merged = {
            "x": np.array([100.0, 101.0, 102.0, 103.0]),  # volts
            "y": y,
            "x_unit": "V",
            "per_run": [],
            "merge_metadata": {"cooler_v": 30000.0, "laser_sp": 12000.0,
                               "mass_amu": 40.0, "harmonic": 2},
        }
        source_config = {"ref_freq": 5.0e14}
        _x, _y, yerr, _w = project_voltage_merge_to_frequency(
            merged, source_config)
        np.testing.assert_allclose(yerr, np.sqrt(y + 1.0))


if __name__ == "__main__":
    unittest.main()
