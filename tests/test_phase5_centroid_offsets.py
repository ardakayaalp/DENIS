"""Unit tests for per-file manual centroid offsets in Pre-Analysis.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers the PA-side manual centroid offset stored per FileEntry: its
default, the compute-time application that subtracts the offset from the
frequency axis additively with any GP-derived centroid correction (and
its no-op-with-warning behavior in voltage merges), and the YAML
save/load round-trip including the pre-Phase-5 default-to-zero path.

Depends on: gui.preanalysis_tab (FileEntry, PreAnalysisTab),
gui.analysis.merge (compute_merged_spectrum); PySide6 for the
QApplication fixture.
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


class FileEntryOffsetTests(unittest.TestCase):
    """FileEntry stores centroid_offset_mhz (default 0). The right-
    click menu adds a "Set centroid offset…" action."""

    def setUp(self):
        _ensure_app()

    def test_file_entry_default_offset_is_zero(self):
        from gui.preanalysis_tab import FileEntry

        fe = FileEntry(
            filepath="/data/run_7735.asdf",
            run_number="7735", cooler_v=29895.3, date="2026-05-13",
            laser_sp=15975.4150, mass_amu=47.95)
        self.assertEqual(fe.centroid_offset_mhz, 0.0)

    def test_set_offset_persists_on_entry(self):
        from gui.preanalysis_tab import FileEntry

        fe = FileEntry(
            filepath="/data/run_7735.asdf",
            run_number="7735", cooler_v=29895.3, date="2026-05-13",
            laser_sp=15975.4150, mass_amu=47.95)
        fe.centroid_offset_mhz = 12.5
        self.assertEqual(fe.centroid_offset_mhz, 12.5)


class ManualOffsetThreadedThroughComputeTests(unittest.TestCase):
    """``compute_merged_spectrum`` applies the manual offset additively
    with the GP correction in frequency mode, and records both
    separately in per_run audit."""

    def setUp(self):
        _ensure_app()

    def test_manual_offset_subtracted_from_frequency_axis(self):
        from gui.analysis.merge import compute_merged_spectrum

        captured_xs = []

        def fake_compute_binned(data, cfg, **_kw):
            return {"x": np.array([100.0, 200.0, 300.0]),
                    "y": np.array([1.0, 2.0, 3.0]),
                    "yerr": None, "xerr": None,
                    "use_callable_yerr": False,
                    "x_label": "Frequency [MHz]",
                    "info": {"n_bins": 3, "dx_median": 100.0}}

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

            src_cfg = {
                "mass": 47.95, "ref_freq": 7.5e14, "harmonic": 2,
                "bin_mode": "Frequency", "cooler_correction": "pbp",
                "pmt_gate": [3, 4]}
            result = compute_merged_spectrum(
                ["/data/run_a.asdf"], src_cfg,
                {"domain": "frequency"},
                manual_offsets_map={"/data/run_a.asdf": 50.0})
            # Per-run x was shifted by -50; the global grid follows.
            shifted = np.array([50.0, 150.0, 250.0])
            np.testing.assert_array_equal(result["per_run"][0]["x"],
                                            shifted)
            self.assertAlmostEqual(
                result["per_run"][0]["manual_offset_mhz"], 50.0)
            # No GP correction was supplied, so the GP audit is zero.
            self.assertAlmostEqual(
                result["per_run"][0]["centroid_correction_mhz"], 0.0)

    def test_manual_offset_additive_with_gp_correction(self):
        """When both manual and GP corrections are present, the
        per-file x axis is shifted by the SUM, and each is recorded
        separately in the audit."""
        from gui.analysis.merge import compute_merged_spectrum

        def fake_compute_binned(data, cfg, **_kw):
            return {"x": np.array([100.0]),
                    "y": np.array([1.0]),
                    "yerr": None, "xerr": None,
                    "use_callable_yerr": False,
                    "x_label": "Frequency [MHz]",
                    "info": {"n_bins": 1, "dx_median": 1.0}}

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

            src_cfg = {
                "mass": 47.95, "ref_freq": 7.5e14, "harmonic": 2,
                "bin_mode": "Frequency", "cooler_correction": "pbp",
                "pmt_gate": [3, 4]}
            result = compute_merged_spectrum(
                ["/data/run_a.asdf"], src_cfg,
                {"domain": "frequency",
                 "centroid_correction": True},
                centroid_corrections_map={
                    "/data/run_a.asdf": {
                        "correction_mhz": 30.0, "sigma_mhz": 1.0,
                        "mode": "Auto"}},
                manual_offsets_map={"/data/run_a.asdf": 50.0})
            rd = result["per_run"][0]
            # 100 - (30 + 50) = 20
            np.testing.assert_array_equal(rd["x"], np.array([20.0]))
            self.assertAlmostEqual(rd["centroid_correction_mhz"], 30.0)
            self.assertAlmostEqual(rd["manual_offset_mhz"], 50.0)

    def test_voltage_merge_with_manual_offsets_emits_warning(self):
        """When manual_offsets_map is non-empty but the merge is
        voltage-domain, the offset isn't applied -- compute_merged_spectrum
        must surface a warning so the user notices (instead of silently
        producing an unshifted merge)."""
        from gui.analysis.merge import compute_merged_spectrum

        def fake_compute_binned(data, cfg, **_kw):
            return {"x": np.array([100.0]),
                    "y": np.array([1.0]),
                    "yerr": None, "xerr": None,
                    "use_callable_yerr": False,
                    "x_label": "Voltage [V]",
                    "info": {"n_bins": 1, "dx_median": 1.0}}

        with patch("clstools.CLSDataFrame") as MockCLS, \
                patch("gui.analysis.binning.compute_binned",
                       side_effect=fake_compute_binned):
            data = MagicMock()
            data.ToF_binned.index.to_list.return_value = [0]
            data.ToF_binned.values = np.array([[0]])
            data.Run = ""
            data.Size = 100
            data.Laser_set = 15975.4150
            data.Vcool_init = 29895.3
            MockCLS.return_value = data

            src_cfg = {
                "mass": 0, "ref_freq": 0, "harmonic": 2,
                "bin_mode": "Raw Voltage", "cooler_correction": "pbp",
                "pmt_gate": [3, 4]}
            result = compute_merged_spectrum(
                ["/data/run_a.asdf"], src_cfg,
                {"domain": "voltage"},
                manual_offsets_map={"/data/run_a.asdf": 50.0})
            warnings = result.get("merge_warnings", [])
            codes = [w["code"] for w in warnings]
            self.assertIn("MANUAL_OFFSET_IGNORED_VOLTAGE_MERGE", codes)

    def test_manual_offset_not_applied_in_voltage_mode(self):
        """Voltage merges don't have a frequency axis to shift on;
        the manual offset is silently ignored (no error)."""
        from gui.analysis.merge import compute_merged_spectrum

        def fake_compute_binned(data, cfg, **_kw):
            return {"x": np.array([100.0]),
                    "y": np.array([1.0]),
                    "yerr": None, "xerr": None,
                    "use_callable_yerr": False,
                    "x_label": "Voltage [V]",
                    "info": {"n_bins": 1, "dx_median": 1.0}}

        with patch("clstools.CLSDataFrame") as MockCLS, \
                patch("gui.analysis.binning.compute_binned",
                       side_effect=fake_compute_binned):
            data = MagicMock()
            data.ToF_binned.index.to_list.return_value = [0]
            data.ToF_binned.values = np.array([[0]])
            data.Run = ""
            data.Size = 100
            data.Laser_set = 15975.4150
            data.Vcool_init = 29895.3
            MockCLS.return_value = data

            src_cfg = {
                "mass": 0, "ref_freq": 0, "harmonic": 2,
                "bin_mode": "Raw Voltage", "cooler_correction": "pbp",
                "pmt_gate": [3, 4]}
            result = compute_merged_spectrum(
                ["/data/run_a.asdf"], src_cfg,
                {"domain": "voltage"},
                manual_offsets_map={"/data/run_a.asdf": 50.0})
            rd = result["per_run"][0]
            # Voltage axis untouched.
            np.testing.assert_array_equal(rd["x"], np.array([100.0]))
            # Audit shows 0 manual offset applied.
            self.assertAlmostEqual(rd["manual_offset_mhz"], 0.0)


class YamlRoundTripTests(unittest.TestCase):
    """centroid_offset_mhz survives PA YAML save/load round-trip."""

    def setUp(self):
        _ensure_app()

    def test_offset_persists_in_yaml(self):
        import yaml
        from gui.preanalysis_tab import PreAnalysisTab, FileEntry

        pa = PreAnalysisTab()
        # Inject a FileEntry directly (bypass _load_file's filesystem
        # read).
        fe = FileEntry(
            filepath="/data/run_7735.asdf",
            run_number="7735", cooler_v=29895.3, date="2026-05-13",
            laser_sp=15975.4150, mass_amu=47.95)
        fe.centroid_offset_mhz = 12.5
        pa._file_entries.append(fe)

        d = pa._build_config_dict()
        files = d["preanalysis"]["files"]
        self.assertEqual(len(files), 1)
        self.assertAlmostEqual(files[0]["centroid_offset_mhz"], 12.5)

        # Round trip through YAML.
        text = yaml.safe_dump(d)
        d2 = yaml.safe_load(text)
        self.assertAlmostEqual(
            d2["preanalysis"]["files"][0]["centroid_offset_mhz"], 12.5)

    def test_load_pre_phase5_yaml_defaults_offset_to_zero(self):
        """A pre-Phase-5 config dict has no ``centroid_offset_mhz``
        key; the restore path must default to 0.0 on each loaded
        FileEntry. We patch ``_load_file`` to append a stub so the
        restore exercises the per-entry assignment without needing a
        real ASDF on disk."""
        from gui.preanalysis_tab import PreAnalysisTab, FileEntry

        cfg = {
            "files": [{"path": "/data/run_7735.asdf",
                       "color": "#1f77b4", "alpha": 1.0,
                       "linestyle": "-", "checked": True}],
            "merged_entries": [],
            "models": [],
            "plot_options": {},
            "tof_gate": {},
            "binning": {},
        }
        pa = PreAnalysisTab()

        def _fake_load_file(filepath):
            fe = FileEntry(
                filepath=filepath, run_number="7735",
                cooler_v=29895.3, date="2026-05-13",
                laser_sp=15975.4150, mass_amu=47.95)
            pa._file_entries.append(fe)

        with patch("os.path.isfile", return_value=True), \
                patch.object(pa, "_load_file",
                              side_effect=_fake_load_file):
            pa._restore_from_dict(cfg)

        self.assertEqual(len(pa._file_entries), 1)
        # Default 0 is applied even though the YAML didn't carry
        # the key. (FileEntry.__init__ also defaults to 0; the
        # restore code's .get default covers the load-with-key path.)
        self.assertEqual(pa._file_entries[0].centroid_offset_mhz, 0.0)


if __name__ == "__main__":
    unittest.main()
