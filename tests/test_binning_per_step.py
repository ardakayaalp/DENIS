"""Headless tests for per-scan-step frequency binning.

Date:    2026-07-15
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

The "Per scan step" bin definition puts one frequency bin per raw DAC
scanning-voltage step, at the mean event frequency of that step, instead
of laying clstools' equal-width grid over the Doppler-nonlinear axis. The
load-bearing test here is ``AliasingTests``: on data engineered so a
uniform grid merges two adjacent steps into one bin (the spurious
doubled-count spike a user sees when switching an Auto view to Frequency),
per-step binning keeps every step separate and shows no spike.

Uses a minimal fake CLSDataFrame carrying only a ``Sorted`` event frame,
so the suite needs no real .asdf file and never calls clstools.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_binning_per_step -v

Depends on: gui.analysis.binning, numpy, pandas.
"""

import unittest

import numpy as np

from gui.analysis.binning import (
    BIN_DEFINITIONS,
    _per_step_frequency_bins,
    compute_binned,
)


def make_fake(step_centers_mhz, events_per_step=30, tdc=3, tof=41.0):
    """A fake CLSDataFrame whose per-event frame has the given step centers.

    Each unique DV step gets ``events_per_step`` events at exactly its
    center frequency (zero jitter, so the per-step mean is exact). All
    events share one PMT channel and TOF so gating is a no-op by default.
    """
    import pandas as pd

    dv, f_hz, tof_col, tdc_col = [], [], [], []
    for i, c_mhz in enumerate(step_centers_mhz):
        dv.extend([float(i)] * events_per_step)
        f_hz.extend([c_mhz * 1e6] * events_per_step)
        tof_col.extend([tof] * events_per_step)
        tdc_col.extend([tdc] * events_per_step)
    df = pd.DataFrame({"DV": dv, "F": f_hz, "TOF": tof_col, "TDC": tdc_col})

    class FakeData:
        pass

    data = FakeData()
    data.Sorted = df
    return data


class PerStepBasicsTests(unittest.TestCase):

    def test_per_scan_step_is_a_known_definition(self):
        self.assertIn("Per scan step", BIN_DEFINITIONS)

    def test_pre_analysis_default_is_per_step(self):
        from gui.preanalysis_tab import DEFAULT_PA_BIN_DEFINITION
        self.assertEqual(DEFAULT_PA_BIN_DEFINITION, "Per scan step")

    def test_one_bin_per_step_at_the_mean_frequency(self):
        centers = [0.0, 10.0, 20.0, 30.0, 40.0]
        data = make_fake(centers, events_per_step=25)
        x, y, xerr = _per_step_frequency_bins(
            data, None, [3, 4], None, None)
        np.testing.assert_allclose(x, centers)      # ascending, in MHz
        self.assertEqual(list(y), [25.0] * 5)       # every event counted once
        self.assertIsNone(xerr)                     # not requested

    def test_pmt_gate_excludes_other_channels(self):
        data = make_fake([0.0, 10.0], events_per_step=12, tdc=1)
        # All events are on channel 1; gating to [3,4] leaves nothing.
        x, y, _ = _per_step_frequency_bins(data, None, [3, 4], None, None)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(y), 0)

    def test_compute_binned_routes_per_step(self):
        centers = [0.0, 12.0, 24.0]
        data = make_fake(centers, events_per_step=8)
        out = compute_binned(data, {"bin_mode": "Frequency",
                                    "bin_definition": "Per scan step",
                                    "yerr_mode": "None"})
        self.assertTrue(out["info"].get("per_step"))
        self.assertEqual(out["x_label"], "Frequency [MHz]")
        np.testing.assert_allclose(out["x"], centers)
        self.assertEqual(list(out["y"]), [8.0, 8.0, 8.0])


class AliasingTests(unittest.TestCase):
    """The whole point: a uniform grid doubles a bin where two steps sit
    closer than the bin width; per-step binning cannot."""

    # Steps ~10 MHz apart, EXCEPT 20 and 24 which are only 4 MHz apart.
    STEP_CENTERS = [0.0, 10.0, 20.0, 24.0, 34.0, 44.0]
    PER_STEP_COUNT = 30

    def test_uniform_grid_produces_a_doubled_spike(self):
        """Documents the artifact this feature removes."""
        all_f = np.repeat(self.STEP_CENTERS, self.PER_STEP_COUNT)
        # A 10 MHz uniform grid (~one bin per step, as clstools' Auto lays
        # down) merges the 20 and 24 MHz steps into the [20, 30) bin.
        edges = np.arange(0.0, 50.0 + 10.0, 10.0)
        counts, _ = np.histogram(all_f, bins=edges)
        self.assertEqual(counts.max(), 2 * self.PER_STEP_COUNT)   # the spike

    def test_per_step_shows_no_spike(self):
        data = make_fake(self.STEP_CENTERS, events_per_step=self.PER_STEP_COUNT)
        out = compute_binned(data, {"bin_mode": "Frequency",
                                    "bin_definition": "Per scan step",
                                    "yerr_mode": "None"})
        # One bin per step, every bin exactly PER_STEP_COUNT -- no bin ever
        # holds two steps, so the max is the single-step count, not double.
        self.assertEqual(len(out["x"]), len(self.STEP_CENTERS))
        self.assertEqual(out["y"].max(), float(self.PER_STEP_COUNT))
        self.assertEqual(out["y"].sum(),
                         float(self.PER_STEP_COUNT * len(self.STEP_CENTERS)))


class StepMultipleTests(unittest.TestCase):
    """'Bin: N × step' — grouping N adjacent native step bins (UI overhaul
    Phase 4). Grouping keeps the scan's own sampling, so it cannot alias."""

    def test_identity_when_n_is_one(self):
        from gui.analysis.binning import _group_adjacent_steps
        x = np.array([0.0, 10.0])
        y = np.array([3.0, 4.0])
        gx, gy, gxe = _group_adjacent_steps(x, y, None, 1)
        np.testing.assert_array_equal(gx, x)
        np.testing.assert_array_equal(gy, y)
        self.assertIsNone(gxe)

    def test_pairs_sum_counts_and_weight_centers(self):
        from gui.analysis.binning import _group_adjacent_steps
        x = np.array([0.0, 10.0, 20.0, 30.0])
        y = np.array([10.0, 30.0, 5.0, 5.0])
        gx, gy, gxe = _group_adjacent_steps(x, y, None, 2)
        np.testing.assert_allclose(gy, [40.0, 10.0])
        np.testing.assert_allclose(gx, [7.5, 25.0])  # count-weighted means
        self.assertIsNone(gxe)

    def test_partial_tail_group_survives(self):
        from gui.analysis.binning import _group_adjacent_steps
        x = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        y = np.ones(5)
        gx, gy, _ = _group_adjacent_steps(x, y, None, 2)
        self.assertEqual(len(gx), 3)
        np.testing.assert_allclose(gy, [2.0, 2.0, 1.0])
        self.assertEqual(gx[-1], 40.0)

    def test_xerr_combines_within_and_between_spread(self):
        from gui.analysis.binning import _group_adjacent_steps
        x = np.array([0.0, 10.0])
        y = np.array([1.0, 1.0])
        xerr = np.array([2.0, 2.0])
        _gx, _gy, gxe = _group_adjacent_steps(x, y, xerr, 2)
        # between-variance of centers 0/10 with equal weights = 25;
        # mean within-variance = 4 -> sqrt(29).
        np.testing.assert_allclose(gxe, [np.sqrt(29.0)])

    def test_compute_binned_applies_step_multiple(self):
        centers = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        data = make_fake(centers, events_per_step=7)
        out = compute_binned(data, {"bin_mode": "Frequency",
                                    "bin_definition": "Per scan step",
                                    "step_multiple": 2,
                                    "yerr_mode": "None"})
        self.assertEqual(len(out["x"]), 3)
        self.assertEqual(list(out["y"]), [14.0, 14.0, 14.0])
        self.assertEqual(out["info"].get("step_multiple"), 2)

    def test_step_multiple_is_a_per_file_override_key(self):
        from gui.analysis.binning import (
            BINNING_OVERRIDE_KEYS, effective_binning_config)
        self.assertIn("step_multiple", BINNING_OVERRIDE_KEYS)
        cfg = effective_binning_config(
            {"bin_definition": "Per scan step"}, {"step_multiple": 3})
        self.assertEqual(cfg["step_multiple"], 3)


class DefaultFlipTests(unittest.TestCase):
    """Phase 5.1: the Analysis default is now the aliasing-safe per-step
    binning, matching Pre-Analysis."""

    def test_module_default_is_per_step(self):
        from gui.analysis.binning import DEFAULT_BIN_DEFINITION
        self.assertEqual(DEFAULT_BIN_DEFINITION, "Per scan step")

    def test_compute_binned_defaults_to_per_step(self):
        data = make_fake([0.0, 10.0, 20.0], events_per_step=4)
        out = compute_binned(data, {"bin_mode": "Frequency",
                                    "yerr_mode": "None"})
        self.assertTrue(out["info"].get("per_step"))
        self.assertEqual(len(out["x"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
