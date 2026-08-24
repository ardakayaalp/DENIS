"""Unit tests for scan derivation, filtering, and the filter registry.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers gui.scan_filter: bunches-per-scan computation, per-event scan
index derivation and masking, scan summaries (counts/rate), path
canonicalization, the ScanFilterRegistry save/load round-trip, and the
filter_data_for_binning context manager that drops excluded scans and
restores state on exit.

Depends on: gui.scan_filter (bunches_per_scan, derive_scan_indices,
make_event_mask, summarize_scans, canonical_path, ScanFilterRegistry,
filter_data_for_binning); PySide6 for the QApplication fixture.
"""

import os
import unittest

import numpy as np

from PySide6.QtWidgets import QApplication

# Qt singleton needed because ScanFilterRegistry derives from QObject.
_APP = QApplication.instance() or QApplication([])

from gui.scan_filter import (
    bunches_per_scan,
    canonical_path,
    derive_scan_indices,
    get_registry,
    make_event_mask,
    set_registry,
    summarize_scans,
    ScanFilterRegistry,
)


class BunchesPerScanTests(unittest.TestCase):

    def test_single_range_one_bpc(self):
        # The example file run_7740: V=-70..300, step=1 -> 371 steps,
        # BunchesPerChannel=1 -> 371 bunches per scan. Anchored on the
        # actual file so accidental refactors that break this formula
        # surface immediately.
        self.assertEqual(
            bunches_per_scan([[-70, 300]], 1.0, 1), 371)

    def test_step_size_two(self):
        # 0..10 V at 2 V step -> 6 steps (0, 2, 4, 6, 8, 10).
        self.assertEqual(bunches_per_scan([[0, 10]], 2.0, 1), 6)

    def test_bunches_per_channel_multiplies(self):
        self.assertEqual(bunches_per_scan([[0, 10]], 1.0, 4), 11 * 4)

    def test_multi_range_sums(self):
        self.assertEqual(
            bunches_per_scan([[-10, 0], [50, 60]], 1.0, 1),
            22)

    def test_degenerate_inputs_return_zero(self):
        # 0 means "no scan derivation possible" -> filtering disabled.
        self.assertEqual(bunches_per_scan([], 1.0, 1), 0)
        self.assertEqual(bunches_per_scan(None, 1.0, 1), 0)
        self.assertEqual(bunches_per_scan([[0, 10]], 0, 1), 0)
        self.assertEqual(bunches_per_scan([[0, 10]], -1, 1), 0)


class DeriveScanIndicesTests(unittest.TestCase):

    def test_split_by_bunches_per_scan(self):
        # 11 steps per scan, 1 bpc -> bunches_per_scan = 11.
        # Bunches 0..10 -> scan 0; 11..21 -> scan 1; 22 -> scan 2.
        bunch = np.array([0, 5, 10, 11, 21, 22, 33])
        idx = derive_scan_indices(bunch, [[0, 10]], 1.0, 1)
        np.testing.assert_array_equal(idx, [0, 0, 0, 1, 1, 2, 3])

    def test_zeros_when_derivation_impossible(self):
        bunch = np.array([0, 100, 200])
        idx = derive_scan_indices(bunch, None, 1.0, 1)
        np.testing.assert_array_equal(idx, [0, 0, 0])


class MakeEventMaskTests(unittest.TestCase):

    def test_all_true_when_no_excluded(self):
        bunch = np.arange(40)
        mask = make_event_mask(bunch, set(), [[0, 10]], 1.0, 1)
        self.assertTrue(mask.all())

    def test_drops_only_excluded_scans(self):
        # 11 bunches per scan -> bunches 0..10 are scan 0, 11..21 scan 1,
        # 22..32 scan 2, 33..43 scan 3.
        bunch = np.array([0, 5, 11, 17, 22, 25, 33, 40])
        mask = make_event_mask(bunch, {1, 3}, [[0, 10]], 1.0, 1)
        # Scan 0 + 2 kept, scan 1 + 3 dropped.
        np.testing.assert_array_equal(
            mask, [True, True, False, False, True, True, False, False])


class SummarizeScansTests(unittest.TestCase):

    def test_basic_counts_and_rate(self):
        # Two scans, channels 3 and 4 are PMT.
        # Scan 0: events at t=0..10s, mix of PMT and non-PMT channels.
        # Scan 1: events at t=15..25s, all PMT.
        bunch = np.array([0, 5, 10,        # scan 0
                          11, 16, 21])     # scan 1
        ts = np.array(
            [1000.0, 1005.0, 1010.0,        # scan 0 spans 10s
             1015.0, 1020.0, 1025.0])       # scan 1 spans 10s
        ch = np.array([3, 2, 4,             # 2 PMT in scan 0
                       3, 4, 4])            # 3 PMT in scan 1
        rows = summarize_scans(
            bunch, ts, ch,
            pmt_gate=[3, 4],
            scanning_ranges=[[0, 10]], step_size=1.0,
            bunches_per_channel=1)
        self.assertEqual(len(rows), 2)
        scan0, scan1 = rows
        self.assertEqual(scan0["scan"], 0)
        self.assertEqual(scan0["n_events"], 3)
        self.assertEqual(scan0["n_pmt"], 2)
        self.assertEqual(scan0["duration"], 10.0)
        self.assertAlmostEqual(scan0["rate_hz"], 0.2)

        self.assertEqual(scan1["scan"], 1)
        self.assertEqual(scan1["n_events"], 3)
        self.assertEqual(scan1["n_pmt"], 3)
        self.assertAlmostEqual(scan1["rate_hz"], 0.3)

    def test_omits_empty_scans(self):
        # Scans with zero events shouldn't appear; the dialog only
        # shows scans actually present in the file.
        bunch = np.array([0, 33])  # scan 0 and scan 3 (with 11/scan)
        ts = np.array([1000.0, 2000.0])
        ch = np.array([3, 3])
        rows = summarize_scans(
            bunch, ts, ch, pmt_gate=[3, 4],
            scanning_ranges=[[0, 10]], step_size=1.0,
            bunches_per_channel=1)
        self.assertEqual([r["scan"] for r in rows], [0, 3])

    def test_handles_empty_input(self):
        rows = summarize_scans(
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            pmt_gate=[3, 4],
            scanning_ranges=[[0, 10]], step_size=1.0,
            bunches_per_channel=1)
        self.assertEqual(rows, [])


class CanonicalPathTests(unittest.TestCase):

    def test_normcase_and_abspath(self):
        # canonical_path applies normcase + abspath; we just check that
        # two different-looking spellings of the same path collapse.
        a = canonical_path("./foo.asdf")
        b = canonical_path(os.path.abspath("./foo.asdf"))
        self.assertEqual(a, b)

    def test_non_string_passthrough(self):
        self.assertIsNone(canonical_path(None))
        self.assertEqual(canonical_path(""), "")


class ScanFilterRegistryTests(unittest.TestCase):

    def setUp(self):
        # Each test gets a fresh registry so prior state doesn't bleed.
        set_registry(ScanFilterRegistry())
        self.reg = get_registry()
        self.signal_count = 0
        self.reg.filters_changed.connect(self._inc)

    def _inc(self):
        self.signal_count += 1

    def test_get_returns_empty_set_for_unknown_path(self):
        self.assertEqual(self.reg.get("/no/such.asdf"), set())
        self.assertFalse(self.reg.has("/no/such.asdf"))

    def test_set_and_get(self):
        self.reg.set("/x/a.asdf", [1, 5, 7])
        self.assertEqual(self.reg.get("/x/a.asdf"), {1, 5, 7})
        self.assertTrue(self.reg.has("/x/a.asdf"))
        self.assertEqual(self.signal_count, 1)

    def test_get_returns_copy(self):
        # Mutating the returned set must not change registry storage.
        self.reg.set("/x/a.asdf", [1, 2])
        s = self.reg.get("/x/a.asdf")
        s.add(99)
        self.assertEqual(self.reg.get("/x/a.asdf"), {1, 2})

    def test_empty_set_clears_entry(self):
        self.reg.set("/x/a.asdf", [1])
        self.reg.set("/x/a.asdf", [])
        self.assertFalse(self.reg.has("/x/a.asdf"))
        # Both mutations emitted the signal.
        self.assertEqual(self.signal_count, 2)

    def test_path_canonicalization_dedupes(self):
        self.reg.set("./foo.asdf", [1, 2])
        # Same file via abspath should hit the same entry.
        self.assertEqual(
            self.reg.get(os.path.abspath("./foo.asdf")), {1, 2})

    def test_to_dict_round_trip(self):
        self.reg.set("/x/a.asdf", [3, 1, 9])
        self.reg.set("/x/b.asdf", [7])
        d = self.reg.to_dict()
        # Excluded lists are sorted for diff-friendly YAML.
        a_key = canonical_path("/x/a.asdf")
        self.assertEqual(d[a_key]["excluded"], [1, 3, 9])

        # Round trip through from_dict on a fresh registry.
        reg2 = ScanFilterRegistry()
        reg2.from_dict(d)
        self.assertEqual(reg2.get("/x/a.asdf"), {1, 3, 9})
        self.assertEqual(reg2.get("/x/b.asdf"), {7})

    def test_from_dict_tolerates_garbage(self):
        # Hand-edited YAML may contain weird shapes; loader shouldn't
        # crash, just skip the bad entries.
        reg = ScanFilterRegistry()
        reg.from_dict({
            "/x/good.asdf": {"excluded": [1, 2]},
            "/x/bad1.asdf": "not a dict",
            "/x/bad2.asdf": {"excluded": "not a list"},
            "/x/bad3.asdf": {"excluded": [1, "two", 3.5, 4]},  # mixed
        })
        self.assertEqual(reg.get("/x/good.asdf"), {1, 2})
        self.assertFalse(reg.has("/x/bad1.asdf"))
        self.assertFalse(reg.has("/x/bad2.asdf"))
        # Only ints (or float-integers) survive the cleaning.
        self.assertEqual(reg.get("/x/bad3.asdf"), {1, 4})

    def test_clear_all(self):
        self.reg.set("/x/a.asdf", [1])
        self.reg.set("/x/b.asdf", [2])
        self.reg.clear_all()
        self.assertEqual(self.reg.to_dict(), {})


class FakeData:
    """Minimal stand-in for clstools' CLSDataFrame.

    The context manager only touches ``Run`` / ``Size`` / ``Sorted``
    plus the timestamp triplet; everything else lives untouched. We
    use a plain pandas DataFrame for ``Run`` to bypass Dask entirely,
    which keeps the test fast and dependency-light.
    """

    def __init__(self, bunch, ts, scanning_ranges, step_size, bpc):
        import pandas as pd
        self.Run = pd.DataFrame({"Bunch": bunch, "TS": ts})
        self.Size = len(bunch)
        self.Sorted = None
        self.ScanningRanges = scanning_ranges
        self.Step_Size = step_size
        self.BunchesPerChannel = bpc
        self.TSstart = float(min(ts))
        self.TSstop = float(max(ts))
        self.DAQTStime = self.TSstop - self.TSstart


class FilterDataForBinningTests(unittest.TestCase):

    def setUp(self):
        import numpy as np
        # 22 events spread across 4 scans (11 bunches/scan, BPC=1):
        # scan 0: bunches 0,5,10  -> 3 events
        # scan 1: bunches 11,16,21 -> 3 events
        # scan 2: bunches 22,27,32 -> 3 events
        # scan 3: bunches 33,38,43 -> 3 events
        self.bunch = np.array(
            [0, 5, 10, 11, 16, 21, 22, 27, 32, 33, 38, 43])
        self.ts = np.array(
            [1000.0, 1001, 1002, 1100, 1101, 1102,
             1200, 1201, 1202, 1300, 1301, 1302])
        self.data = FakeData(
            self.bunch, self.ts,
            scanning_ranges=[[0, 10]], step_size=1.0, bpc=1)

    def _run_len(self):
        run = self.data.Run
        try:
            return len(run.compute())
        except AttributeError:
            return len(run)

    def test_empty_excluded_is_noop(self):
        from gui.scan_filter import filter_data_for_binning
        with filter_data_for_binning(self.data, set()):
            self.assertEqual(self._run_len(), 12)
        self.assertEqual(self.data.Size, 12)

    def test_drops_excluded_scans(self):
        from gui.scan_filter import filter_data_for_binning
        with filter_data_for_binning(self.data, {1, 3}):
            # 6 events kept (scans 0 + 2)
            self.assertEqual(self._run_len(), 6)
            self.assertEqual(self.data.Size, 6)

    def test_restores_after_exit(self):
        from gui.scan_filter import filter_data_for_binning
        with filter_data_for_binning(self.data, {1, 3}):
            pass
        self.assertEqual(self._run_len(), 12)
        self.assertEqual(self.data.Size, 12)
        # Timestamps restored too.
        self.assertEqual(self.data.TSstart, 1000.0)
        self.assertEqual(self.data.TSstop, 1302.0)

    def test_restores_after_exception(self):
        from gui.scan_filter import filter_data_for_binning

        class BoomError(RuntimeError):
            pass

        with self.assertRaises(BoomError):
            with filter_data_for_binning(self.data, {1, 2, 3}):
                raise BoomError("simulated downstream failure")
        # Even when the caller raises, the filter must roll back.
        self.assertEqual(self._run_len(), 12)
        self.assertEqual(self.data.Size, 12)

    def test_missing_scan_metadata_falls_through(self):
        # If we can't derive scans (no ScanningRanges), the context
        # manager yields without altering state. This degrades to "no
        # filtering" rather than crashing the fit.
        from gui.scan_filter import filter_data_for_binning
        self.data.ScanningRanges = None
        with filter_data_for_binning(self.data, {1, 3}):
            self.assertEqual(self._run_len(), 12)


if __name__ == "__main__":
    unittest.main()
