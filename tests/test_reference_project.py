"""Unit tests for reference projects and correction extraction.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers the AnalysisProject is_reference flag and its YAML round-trip,
the get_reference_observations centroid export (separate vs simultaneous
fits, merged/virtual-split source-name stamping, and skip conditions),
the _extract_correction pipeline gate, and _build_corrections_map.

Depends on: gui.analysis.project (AnalysisProject), gui.analysis.fitting
(_extract_correction); PySide6 for the QApplication fixture.
"""

import math
import unittest

from PySide6.QtWidgets import QApplication

# Single QApplication shared by the test process; PySide6 widgets need it.
_APP = QApplication.instance() or QApplication([])

from gui.analysis.project import AnalysisProject
from gui.analysis.fitting import _extract_correction


def _make_params_df(centroids_per_source):
    """Build a satlas2-style params_df dict with one centroid per source.

    centroids_per_source: list of ``(source, model, value, stderr)`` tuples
    plus optional non-centroid rows the caller may want.
    """
    rows = []
    for src, model, val, err in centroids_per_source:
        rows.append((src, model, "centroid", val, err))
    return {
        "Source": [r[0] for r in rows],
        "Model": [r[1] for r in rows],
        "Parameter": [r[2] for r in rows],
        "Value": [r[3] for r in rows],
        "Stderr": [r[4] for r in rows],
    }


def _make_result(run_num, run_file, params_df, ts_start, source_name=None):
    out = {
        "success": True,
        "run_number": run_num,
        "run_file": run_file,
        "params_df": params_df,
        "run_metadata": {"ts_start": ts_start},
    }
    if source_name is not None:
        out["source_name"] = source_name
    return out


class IsReferenceFlagTests(unittest.TestCase):

    def test_default_is_sample(self):
        p = AnalysisProject("P")
        self.assertFalse(p.is_reference)

    def test_explicit_reference(self):
        p = AnalysisProject("P", is_reference=True)
        self.assertTrue(p.is_reference)

    def test_to_dict_round_trip_reference(self):
        p = AnalysisProject("P", is_reference=True)
        d = p.to_dict()
        self.assertTrue(d["is_reference"])
        p2 = AnalysisProject("P2")
        p2.from_dict(d)
        self.assertTrue(p2.is_reference)

    def test_to_dict_round_trip_sample(self):
        p = AnalysisProject("P")
        d = p.to_dict()
        self.assertFalse(d["is_reference"])
        p2 = AnalysisProject("P2", is_reference=True)
        p2.from_dict(d)
        self.assertFalse(p2.is_reference)

    def test_legacy_yaml_without_flag_loads_as_sample(self):
        # Older YAML saved before this commit has no is_reference key
        legacy = {"project_name": "Legacy", "blocks": []}
        p = AnalysisProject("Tmp")
        p.from_dict(legacy)
        self.assertFalse(p.is_reference)


class GetReferenceObservationsTests(unittest.TestCase):

    def test_sample_project_returns_empty(self):
        p = AnalysisProject("P")
        p._last_results = [_make_result(
            "1", "/x/run_1.asdf",
            _make_params_df([("Run_1", "m", 100.0, 1.0)]),
            ts_start=1700000000.0,
        )]
        self.assertEqual(p.get_reference_observations(), [])

    def test_no_results_returns_empty(self):
        p = AnalysisProject("P", is_reference=True)
        self.assertEqual(p.get_reference_observations(), [])

    def test_basic_extraction_separate_fit(self):
        """Each result has only its own source's params (separate fit)."""
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [
            _make_result("1", "/x/run_1.asdf",
                         _make_params_df([("Run_1", "m", 100.0, 1.0)]),
                         ts_start=1700000000.0),
            _make_result("2", "/x/run_2.asdf",
                         _make_params_df([("Run_2", "m", 101.5, 0.5)]),
                         ts_start=1700003600.0),
        ]
        obs = p.get_reference_observations()
        self.assertEqual(len(obs), 2)
        self.assertAlmostEqual(obs[0]["centroid_mhz"], 100.0)
        self.assertAlmostEqual(obs[0]["sigma_mhz"], 1.0)
        self.assertEqual(obs[0]["label"], "run_1.asdf")
        self.assertEqual(obs[0]["run_file"], "/x/run_1.asdf")
        self.assertTrue(obs[0]["include"])
        self.assertAlmostEqual(obs[1]["centroid_mhz"], 101.5)

    def test_simultaneous_fit_disambiguates_by_source(self):
        """In simultaneous fits every result shares one global params_df
        with all sources' centroids; we must filter on Source."""
        global_params = _make_params_df([
            ("Run_111", "m", 100.0, 1.0),
            ("Run_222", "m", 105.0, 1.5),
            ("Run_333", "m", 110.0, 2.0),
        ])
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [
            _make_result("111", "/x/run_111.asdf", global_params,
                         ts_start=1700000000.0),
            _make_result("222", "/x/run_222.asdf", global_params,
                         ts_start=1700003600.0),
            _make_result("333", "/x/run_333.asdf", global_params,
                         ts_start=1700007200.0),
        ]
        obs = p.get_reference_observations()
        self.assertEqual(len(obs), 3)
        # Each row's centroid must match its OWN source, not the
        # first-listed one.
        self.assertAlmostEqual(obs[0]["centroid_mhz"], 100.0)
        self.assertAlmostEqual(obs[1]["centroid_mhz"], 105.0)
        self.assertAlmostEqual(obs[2]["centroid_mhz"], 110.0)

    def test_merged_spectrum_uses_stamped_source_name(self):
        """A merged result's `run_file` is the synthetic key
        `merged://<name>`. Deriving the source name from that path
        would mismatch what satlas2 actually used
        (`source_name_for_merged` prefixes another `Run_`), so the
        fitter stamps the real name onto the result. Without that
        stamp the centroid is silently skipped."""
        merged_name = "run_7164_7165"
        fit_src = f"Run_{merged_name}"  # source_name_for_merged result
        global_params = _make_params_df([
            ("Run_111", "m", 100.0, 1.0),
            (fit_src, "m", 222.0, 2.0),
        ])
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [
            _make_result("111", "/x/run_111.asdf", global_params,
                         ts_start=1700000000.0,
                         source_name="Run_111"),
            _make_result(merged_name, f"merged://{merged_name}",
                         global_params, ts_start=1700003600.0,
                         source_name=fit_src),
        ]
        obs = p.get_reference_observations()
        self.assertEqual(len(obs), 2)
        self.assertAlmostEqual(obs[0]["centroid_mhz"], 100.0)
        self.assertAlmostEqual(obs[1]["centroid_mhz"], 222.0)
        # Label comes from the basename of the synthetic path -- still
        # readable for the GP UI.
        self.assertEqual(obs[1]["label"], merged_name)

    def test_virtual_split_uses_stamped_source_name(self):
        """A virtual-split's source name is `Run_<source_id>`
        (e.g. `Run_7509_A`). Path-basename heuristics would yield the
        wrong token; the stamp is required."""
        fit_src = "Run_7509_A"
        params = _make_params_df([(fit_src, "m", 333.0, 0.5)])
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [
            _make_result("7509_A", "/x/run_7509.vasdf", params,
                         ts_start=1700000000.0,
                         source_name=fit_src),
        ]
        obs = p.get_reference_observations()
        self.assertEqual(len(obs), 1)
        self.assertAlmostEqual(obs[0]["centroid_mhz"], 333.0)

    def test_legacy_result_without_source_name_falls_back(self):
        """Results stamped before this fix won't have `source_name`.
        For standard ASDF runs the path-basename heuristic still
        recovers the correct source, so old in-memory results keep
        working after a project upgrade."""
        params = _make_params_df([("Run_1", "m", 100.0, 1.0)])
        p = AnalysisProject("P", is_reference=True)
        # Explicitly no source_name key:
        p._last_results = [_make_result(
            "1", "/x/run_1.asdf", params, ts_start=1700000000.0)]
        obs = p.get_reference_observations()
        self.assertEqual(len(obs), 1)
        self.assertAlmostEqual(obs[0]["centroid_mhz"], 100.0)

    def test_skip_failed_fits(self):
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [
            _make_result("1", "/x/run_1.asdf",
                         _make_params_df([("Run_1", "m", 100.0, 1.0)]),
                         ts_start=1700000000.0),
            {"success": False, "run_number": "2"},
        ]
        self.assertEqual(len(p.get_reference_observations()), 1)

    def test_skip_zero_or_missing_sigma(self):
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [
            _make_result("1", "/x/run_1.asdf",
                         _make_params_df([("Run_1", "m", 100.0, 0.0)]),
                         ts_start=1700000000.0),
            _make_result("2", "/x/run_2.asdf",
                         _make_params_df([("Run_2", "m", 101.0, None)]),
                         ts_start=1700003600.0),
        ]
        # Both have unusable sigma; skipped.
        self.assertEqual(p.get_reference_observations(), [])

    def test_skip_zero_or_missing_ts_start(self):
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [
            _make_result("1", "/x/run_1.asdf",
                         _make_params_df([("Run_1", "m", 100.0, 1.0)]),
                         ts_start=0.0),
            {
                "success": True, "run_number": "2",
                "run_file": "/x/run_2.asdf",
                "params_df": _make_params_df(
                    [("Run_2", "m", 101.0, 1.0)]),
                "run_metadata": {},  # no ts_start at all
            },
        ]
        self.assertEqual(p.get_reference_observations(), [])

    def test_skip_nan_centroid(self):
        p = AnalysisProject("P", is_reference=True)
        p._last_results = [_make_result(
            "1", "/x/run_1.asdf",
            _make_params_df([("Run_1", "m", float("nan"), 1.0)]),
            ts_start=1700000000.0,
        )]
        self.assertEqual(p.get_reference_observations(), [])


class ExtractCorrectionTests(unittest.TestCase):
    """The Phase D pipeline gate that translates a panel correction
    dict into the (mhz, sigma) tuple the fitter applies. Must return
    zeros for any "skip" condition so the downstream
    ``if _corr_mhz != 0`` branch evaluates correctly."""

    def test_none(self):
        self.assertEqual(_extract_correction(None), (0.0, 0.0))

    def test_empty_dict(self):
        self.assertEqual(_extract_correction({}), (0.0, 0.0))

    def test_off_mode(self):
        self.assertEqual(
            _extract_correction({"mode": "Off",
                                  "correction_mhz": 5.0,
                                  "sigma_mhz": 0.3}),
            (0.0, 0.0))

    def test_auto_mode(self):
        self.assertEqual(
            _extract_correction({"mode": "Auto",
                                  "correction_mhz": 5.0,
                                  "sigma_mhz": 0.3}),
            (5.0, 0.3))

    def test_manual_mode(self):
        self.assertEqual(
            _extract_correction({"mode": "Manual",
                                  "correction_mhz": -2.5,
                                  "sigma_mhz": 1.0}),
            (-2.5, 1.0))

    def test_bad_value_returns_zeros(self):
        # A typo / non-numeric should be treated as "no correction"
        # rather than crashing the fit.
        self.assertEqual(
            _extract_correction({"mode": "Auto",
                                  "correction_mhz": "bogus",
                                  "sigma_mhz": 1.0}),
            (0.0, 0.0))


class BuildCorrectionsMapTests(unittest.TestCase):
    """AnalysisProject._build_corrections_map needs apply_globally,
    a fitted corrector, and Auto/Manual mode for each file."""

    def _make_panel(self, *, apply_globally=True, has_corrector=True):
        class FakeCorrector:
            is_fit = True

        class FakePanel:
            def __init__(self):
                self.apply_globally = apply_globally
                self.corrector = FakeCorrector() if has_corrector else None
                self._table = {}

            def get_correction(self, project, fpath):
                return self._table.get((project, fpath))

        return FakePanel()

    def test_no_panel_returns_empty(self):
        p = AnalysisProject("P")
        self.assertEqual(
            p._build_corrections_map(["/x/a.asdf"]), {})

    def test_global_off_returns_empty(self):
        p = AnalysisProject("P")
        # Inject a fake panel via monkey-patch
        panel = self._make_panel(apply_globally=False)
        p._find_reference_correction_panel = lambda: panel
        panel._table[("P", "/x/a.asdf")] = {
            "correction_mhz": 5.0, "sigma_mhz": 0.5, "mode": "Auto"}
        self.assertEqual(
            p._build_corrections_map(["/x/a.asdf"]), {})

    def test_no_corrector_returns_empty(self):
        p = AnalysisProject("P")
        panel = self._make_panel(has_corrector=False)
        p._find_reference_correction_panel = lambda: panel
        panel._table[("P", "/x/a.asdf")] = {
            "correction_mhz": 5.0, "sigma_mhz": 0.5, "mode": "Auto"}
        self.assertEqual(
            p._build_corrections_map(["/x/a.asdf"]), {})

    def test_off_files_skipped(self):
        p = AnalysisProject("P")
        panel = self._make_panel()
        p._find_reference_correction_panel = lambda: panel
        panel._table[("P", "/x/a.asdf")] = {
            "correction_mhz": 5.0, "sigma_mhz": 0.5, "mode": "Auto"}
        panel._table[("P", "/x/b.asdf")] = {
            "correction_mhz": 6.0, "sigma_mhz": 0.6, "mode": "Off"}
        out = p._build_corrections_map(["/x/a.asdf", "/x/b.asdf"])
        self.assertIn("/x/a.asdf", out)
        self.assertNotIn("/x/b.asdf", out)
        self.assertAlmostEqual(out["/x/a.asdf"]["correction_mhz"], 5.0)
        self.assertAlmostEqual(out["/x/a.asdf"]["sigma_mhz"], 0.5)


if __name__ == "__main__":
    unittest.main()
