"""Unit tests for the unified merge dialog and domain plumbing.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers the shared MergeDialog used by both the Pre-Analysis tab and
the Analysis SourceBlock: domain defaulting, the merge-metadata panel,
restoring domain/metadata when editing an existing merge, and the
``domain`` / ``merge_metadata`` plumbing through compute_merged_spectrum.
Also pins that the legacy _PreAnalysisMergeDialog has been removed.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_phase3_unified_merge_dialog -v

Depends on: gui.analysis.merge (MergeDialog, compute_merged_spectrum),
gui.preanalysis_tab; PySide6 for the QApplication fixture.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _stub_source_config(**overrides):
    cfg = {
        "cal_order": 1,
        "cooler_correction": "pbp",
        "tof_gate": None,
        "pmt_gate": [3, 4],
        "v_gate": None,
        "f_gate": None,
        "mass": 47.95,
        "ref_freq": 7.514e14,
        "harmonic": 2,
        "ref_shift": 0.0,
        "noise_filter": 0,
        "cooler_override": 0,
        "laser_override": 0,
        "bin_mode": "Frequency",
        "bin_definition": "Auto",
        "bin_count": 100,
        "bin_width_mhz": 10.0,
    }
    cfg.update(overrides)
    return cfg


def _stub_file_entries(n=2):
    return [
        {"path": f"/data/run_{7735 + i}.asdf",
         "run_number": f"{7735 + i}",
         "binning_override": {}}
        for i in range(n)
    ]


class MergeDialogDomainTests(unittest.TestCase):
    """The domain combo defaults from ``default_domain`` (Phase 3) or
    from the legacy auto-decision over source_config. The merge-level
    metadata panel is hidden for frequency merges."""

    def setUp(self):
        _ensure_app()

    def test_default_domain_voltage_shows_metadata_panel(self):
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            default_domain="voltage")
        self.assertEqual(dlg._domain_combo.currentText(), "voltage")
        # Voltage merges need the merge-level Doppler panel; the
        # _on_domain_changed call at __init__ end leaves the panel
        # set-visible (would appear if the dialog were show()n).
        self.assertTrue(dlg._merge_meta_grp.isVisibleTo(dlg))

    def test_default_domain_frequency_hides_metadata_panel(self):
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            default_domain="frequency")
        self.assertEqual(dlg._domain_combo.currentText(), "frequency")
        # Frequency merges don't need merge-level Doppler -- panel hidden.
        self.assertFalse(dlg._merge_meta_grp.isVisibleTo(dlg))

    def test_default_domain_auto_picks_frequency_when_physics_set(self):
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(),
            _stub_source_config(bin_mode="Frequency",
                                 mass=47.95, ref_freq=7.5e14))
        self.assertEqual(dlg._domain_combo.currentText(), "frequency")

    def test_default_domain_auto_picks_voltage_when_physics_zero(self):
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(),
            _stub_source_config(bin_mode="Raw Voltage",
                                 mass=0, ref_freq=0))
        self.assertEqual(dlg._domain_combo.currentText(), "voltage")

    def test_domain_change_toggles_metadata_panel(self):
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            default_domain="voltage")
        # Voltage → panel visible.
        self.assertTrue(dlg._merge_meta_grp.isVisibleTo(dlg))
        dlg._domain_combo.setCurrentText("frequency")
        self.assertFalse(dlg._merge_meta_grp.isVisibleTo(dlg))
        dlg._domain_combo.setCurrentText("voltage")
        self.assertTrue(dlg._merge_meta_grp.isVisibleTo(dlg))


class MergeDialogMetadataTests(unittest.TestCase):
    """default_merge_metadata pre-fills the per-field spins; the user
    can override; _get_merge_params returns the live values."""

    def setUp(self):
        _ensure_app()

    def test_default_merge_metadata_prefills_spins(self):
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            default_domain="voltage",
            default_merge_metadata={
                "cooler_v": 29895.4,
                "laser_sp": 15975.4150,
                "mass_amu": 47.95,
                "harmonic": 2,
            })
        self.assertAlmostEqual(dlg._merge_cooler.value(), 29895.4)
        self.assertAlmostEqual(dlg._merge_laser.value(), 15975.4150)
        self.assertAlmostEqual(dlg._merge_mass.value(), 47.95)
        self.assertEqual(dlg._merge_harmonic.value(), 2)

    def test_get_merge_params_returns_domain_and_metadata(self):
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            default_domain="voltage",
            default_merge_metadata={"cooler_v": 29895.4,
                                      "laser_sp": 15975.4150,
                                      "mass_amu": 47.95,
                                      "harmonic": 2})
        params = dlg._get_merge_params()
        self.assertEqual(params["domain"], "voltage")
        mm = params["merge_metadata"]
        self.assertAlmostEqual(mm["cooler_v"], 29895.4)
        self.assertAlmostEqual(mm["laser_sp"], 15975.4150)
        self.assertEqual(mm["harmonic"], 2)

    def test_zero_spin_becomes_none_in_merge_metadata(self):
        """The user leaving a field at 0 means "not set" -- downstream
        Phase-4 code expects None, not 0, so it can fall back to per_run
        means."""
        from gui.analysis.merge import MergeDialog

        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            default_domain="voltage",
            default_merge_metadata={})
        mm = dlg._get_merge_params()["merge_metadata"]
        self.assertIsNone(mm["cooler_v"])
        self.assertIsNone(mm["laser_sp"])
        self.assertIsNone(mm["mass_amu"])
        # Harmonic stays as the default 2 (zero would be physically
        # nonsensical for a harmonic number).
        self.assertEqual(mm["harmonic"], 2)


class PopulateFromExistingTests(unittest.TestCase):
    """The Edit-existing-merge code path must restore the Phase 3
    domain + merge_metadata so re-merging doesn't silently revert."""

    def setUp(self):
        _ensure_app()

    def test_existing_voltage_merge_restores_domain_and_metadata(self):
        from gui.analysis.merge import MergeDialog

        existing = {
            "x": np.array([0.0, 1.0]), "y": np.array([1.0, 2.0]),
            "yerr": np.array([1.0, 1.4]),
            "x_unit": "V",
            "per_run": [],
            "merge_params": {
                "bin_step_mhz": None,
                "cooler_correction": "pbp",
                "centroid_correction": False,
                "domain": "voltage",
                "merge_metadata": {"cooler_v": 29895.4,
                                    "laser_sp": 15975.4150,
                                    "mass_amu": 47.95,
                                    "harmonic": 2},
            },
        }
        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            existing_result=existing)
        self.assertEqual(dlg._domain_combo.currentText(), "voltage")
        self.assertAlmostEqual(dlg._merge_cooler.value(), 29895.4)
        self.assertEqual(dlg._merge_harmonic.value(), 2)

    def test_pre_phase3_existing_merge_falls_back_to_x_unit(self):
        """An ``existing_result`` from before Phase 3 has no
        merge_params['domain']. Fall back to x_unit-based guess."""
        from gui.analysis.merge import MergeDialog

        existing = {
            "x": np.array([0.0, 1.0]), "y": np.array([1.0, 2.0]),
            "yerr": np.array([1.0, 1.4]),
            "x_unit": "V",
            "per_run": [],
            "merge_params": {  # no "domain" key
                "bin_step_mhz": None,
                "cooler_correction": "pbp",
                "centroid_correction": False,
            },
        }
        dlg = MergeDialog(
            _stub_file_entries(), _stub_source_config(),
            existing_result=existing)
        self.assertEqual(dlg._domain_combo.currentText(), "voltage")


class ComputeMergedSpectrumDomainTests(unittest.TestCase):
    """The explicit ``domain`` in merge_params overrides the legacy
    auto-decision and is validated."""

    def setUp(self):
        _ensure_app()

    def test_invalid_domain_raises(self):
        from gui.analysis.merge import compute_merged_spectrum

        with self.assertRaises(ValueError) as ctx:
            compute_merged_spectrum(
                ["/data/run_x.asdf"], _stub_source_config(),
                {"domain": "garbage"})
        self.assertIn("domain", str(ctx.exception))

    def test_frequency_domain_requires_physics(self):
        from gui.analysis.merge import compute_merged_spectrum

        with patch("clstools.CLSDataFrame") as MockCLS:
            data = MagicMock()
            data.ToF_binned.index.to_list.return_value = [0]
            data.ToF_binned.values = np.array([[0]])
            MockCLS.return_value = data
            with self.assertRaises(ValueError) as ctx:
                compute_merged_spectrum(
                    ["/data/run_x.asdf"],
                    _stub_source_config(mass=0, ref_freq=0),
                    {"domain": "frequency"})
            self.assertIn("Frequency", str(ctx.exception))

    def test_frequency_domain_overrides_raw_voltage_bin_mode(self):
        """When source_config says Raw Voltage but the user picks
        ``domain="frequency"``, the merge must Compute_WL AND bin in
        frequency mode -- not silently bin in voltage and stamp
        x_unit="MHz" on a voltage axis."""
        from gui.analysis.merge import compute_merged_spectrum

        captured_cfgs = []

        def fake_compute_binned(data, cfg, **_kw):
            captured_cfgs.append(dict(cfg))
            return {"x": np.array([0.0, 1.0]),
                    "y": np.array([1.0, 2.0]),
                    "yerr": None, "xerr": None,
                    "use_callable_yerr": False,
                    "x_label": "Frequency [MHz]",
                    "info": {"n_bins": 2, "dx_median": 1.0}}

        with patch("clstools.CLSDataFrame") as MockCLS, \
                patch("gui.analysis.binning.compute_binned",
                       side_effect=fake_compute_binned):
            data = MagicMock()
            data.ToF_binned.index.to_list.return_value = [0]
            data.ToF_binned.values = np.array([[0]])
            data.Frequency_stepsize = 1.0
            data.Run = ""
            data.Size = 100
            data.Laser_set = 15975.4150
            data.Vcool_init = 29895.3
            MockCLS.return_value = data

            result = compute_merged_spectrum(
                ["/data/run_a.asdf"],
                _stub_source_config(bin_mode="Raw Voltage",
                                     mass=47.95, ref_freq=7.5e14),
                {"domain": "frequency"})
            # compute_binned must have been called with bin_mode
            # forced to "Frequency" -- otherwise we'd bin against
            # voltage events but later stamp x_unit="MHz".
            self.assertEqual(captured_cfgs[0]["bin_mode"], "Frequency")
            self.assertEqual(result["x_unit"], "MHz")

    def test_merge_metadata_passed_through_to_output(self):
        """compute_merged_spectrum writes merge_params['merge_metadata']
        to the output dict so downstream code (the bridge, the fit
        pipeline) can read it without re-deriving."""
        from gui.analysis.merge import compute_merged_spectrum

        # Stub clstools to avoid needing real ASDFs.
        with patch("clstools.CLSDataFrame") as MockCLS, \
                patch("gui.analysis.binning.compute_binned",
                       return_value={
                           "x": np.array([0.0, 50.0]),
                           "y": np.array([1.0, 2.0]),
                           "yerr": None, "xerr": None,
                           "use_callable_yerr": False,
                           "x_label": "Voltage [V]",
                           "info": {"n_bins": 2,
                                    "dx_median": 50.0}}):
            data = MagicMock()
            data.ToF_binned.index.to_list.return_value = [0]
            data.ToF_binned.values = np.array([[0]])
            data.Run = ""
            data.Size = 100
            data.Laser_set = 15975.4150
            data.Vcool_init = 29895.3
            MockCLS.return_value = data

            result = compute_merged_spectrum(
                ["/data/run_a.asdf"],
                _stub_source_config(bin_mode="Raw Voltage"),
                {"domain": "voltage",
                 "merge_metadata": {"cooler_v": 29895.4,
                                     "laser_sp": 15975.4150,
                                     "mass_amu": 47.95,
                                     "harmonic": 2}})
            self.assertIn("merge_metadata", result)
            self.assertAlmostEqual(
                result["merge_metadata"]["cooler_v"], 29895.4)
            self.assertEqual(
                result["merge_metadata"]["harmonic"], 2)


class LegacyDialogClassRemovedTests(unittest.TestCase):
    """Phase 3 deletes _PreAnalysisMergeDialog -- pin that down so a
    future revert doesn't sneak the symbol back in."""

    def test_pre_analysis_merge_dialog_class_is_gone(self):
        import gui.preanalysis_tab as pa_mod
        self.assertFalse(hasattr(pa_mod, "_PreAnalysisMergeDialog"),
                         "_PreAnalysisMergeDialog should be deleted; "
                         "PA's _merge_checked now uses "
                         "gui.analysis.merge.MergeDialog.")

    def test_compute_merge_method_is_gone(self):
        from gui.preanalysis_tab import PreAnalysisTab
        self.assertFalse(hasattr(PreAnalysisTab, "_compute_merge"),
                         "_compute_merge should be deleted; PA now "
                         "uses compute_merged_spectrum.")


if __name__ == "__main__":
    unittest.main()
