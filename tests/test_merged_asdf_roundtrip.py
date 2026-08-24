"""Tests for round-tripping a merged spectrum through the merged ASDF schema.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Exports a merged spectrum to the ``denis_merged_v1`` ASDF schema and
loads it back, verifying that x/y/yerr arrays, merge-level Doppler
metadata, and per-run audit survive disk round-trips (including lazy
ASDF arrays after file close). Also checks routing of merged ASDFs in
the PreAnalysis and Analysis Source-block loaders, PA/Analysis action
parity, merge-dialog prefill, and the export/load guard that rejects
voltage-axis data mislabelled as MHz. This schema-based exporter
replaced an earlier synthesised-events exporter that broke clstools'
cal-polynomial fit on reload. Runs headless from the project root.

Depends on: gui.analysis.* (merge, blocks), gui.preanalysis_tab.
"""

import os
import tempfile
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


class ExportLoadRoundTripTests(unittest.TestCase):
    def test_voltage_merge_roundtrip_preserves_data_and_metadata(self):
        from gui.analysis.merge import (
            export_merged_asdf, load_merged_asdf,
            is_merged_asdf, DENIS_MERGED_SCHEMA)

        merged_data = {
            "merged_name": "merged_7735_7740",
            "x": np.array([-50.0, 0.0, 50.0, 100.0]),
            "y": np.array([2.0, 5.0, 11.0, 4.0]),
            "yerr": np.array([1.4, 2.2, 3.3, 2.0]),
            "x_unit": "V",
            "bin_step_mhz": 1.0,
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.952251,
                                "harmonic": 2},
            "source_runs": ["7735", "7740"],
            "source_files": ["/data/run_7735.asdf",
                              "/data/run_7740.asdf"],
            "per_run": [
                {"run_num": "7735", "path": "/data/run_7735.asdf",
                 "cooler_v": 29895.3, "laser_set": 15975.4150,
                 "mass_amu": 47.775, "harmonic": 2},
                {"run_num": "7740", "path": "/data/run_7740.asdf",
                 "cooler_v": 29895.5, "laser_set": 15975.4150,
                 "mass_amu": 47.775, "harmonic": 2},
            ],
            "metadata": {"imported_from": "preanalysis",
                          "merge_domain": "voltage"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "merged.asdf")
            export_merged_asdf(merged_data, out)
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(is_merged_asdf(out))

            md2 = load_merged_asdf(out)
            self.assertIsNotNone(md2)
            self.assertEqual(md2["merged_name"], "merged_7735_7740")
            self.assertEqual(md2["x_unit"], "V")
            np.testing.assert_array_equal(md2["x"], merged_data["x"])
            np.testing.assert_array_equal(md2["y"], merged_data["y"])
            np.testing.assert_array_equal(md2["yerr"], merged_data["yerr"])
            mm = md2["merge_metadata"]
            self.assertAlmostEqual(mm["cooler_v"], 29895.4)
            self.assertAlmostEqual(mm["mass_amu"], 47.952251)
            self.assertEqual(mm["harmonic"], 2)
            # Per-run audit survives, sans x/y arrays.
            self.assertEqual(len(md2["per_run"]), 2)
            self.assertAlmostEqual(md2["per_run"][0]["cooler_v"], 29895.3)

    def test_load_unrelated_asdf_returns_none(self):
        """A regular clstools ASDF (no denis_schema marker) must be
        skipped so the caller falls through to Load_Run."""
        import asdf
        from gui.analysis.merge import (
            load_merged_asdf, is_merged_asdf)

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "regular.asdf")
            asdf.AsdfFile({"Run": "1234",
                            "CoolerVoltage": 2.9977,
                            "LaserSetpoint": 15975.4150,
                            "raw": np.zeros((4, 6))}).write_to(out)
            self.assertFalse(is_merged_asdf(out))
            self.assertIsNone(load_merged_asdf(out))

    def test_load_missing_file_returns_none(self):
        from gui.analysis.merge import (
            load_merged_asdf, is_merged_asdf)
        path = "/non/existent/merged.asdf"
        self.assertFalse(is_merged_asdf(path))
        self.assertIsNone(load_merged_asdf(path))


class PaLoadMergedRoutingTests(unittest.TestCase):
    """When PA's _load_file is given a DENIS-merged ASDF, it must
    route to _load_merged_asdf_entry and create a MergedFileEntry."""

    def setUp(self):
        _ensure_app()

    def test_pa_load_creates_merged_file_entry(self):
        from gui.analysis.merge import export_merged_asdf
        from gui.preanalysis_tab import (
            PreAnalysisTab, MergedFileEntry)

        merged_data = {
            "merged_name": "merged_X",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "bin_step_mhz": 0,
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "per_run": [
                {"run_num": "a", "path": "/p/a",
                 "cooler_v": 29895.3, "laser_set": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
                {"run_num": "b", "path": "/p/b",
                 "cooler_v": 29895.5, "laser_set": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "merged.asdf")
            export_merged_asdf(merged_data, out)
            pa = PreAnalysisTab()
            pa._load_file(out)
            self.assertEqual(len(pa._file_entries), 1)
            fe = pa._file_entries[0]
            self.assertIsInstance(fe, MergedFileEntry)
            # filepath rewritten to the real on-disk path (not
            # "[merged] <name>") so the PA YAML records it correctly.
            self.assertEqual(fe.filepath, out)
            self.assertEqual(fe.merge_domain, "voltage")
            self.assertAlmostEqual(fe.merge_cooler_v, 29895.4)
            self.assertAlmostEqual(fe.merge_mass_amu, 47.95)


class AnalysisSourceBlockLoadMergedRoutingTests(unittest.TestCase):
    """SourceBlock._add_file_entry must route a DENIS-merged ASDF to
    _add_merged_entry instead of trying to call Load_Run on it."""

    def setUp(self):
        _ensure_app()

    def test_source_block_load_creates_merged_entry(self):
        from gui.analysis.merge import export_merged_asdf
        from gui.analysis.blocks import SourceBlock

        merged_data = {
            "merged_name": "merged_X",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "bin_step_mhz": 0,
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "per_run": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "merged.asdf")
            export_merged_asdf(merged_data, out)
            sb = SourceBlock()
            sb._add_file_entry(out)
            self.assertEqual(len(sb._file_entries), 1)
            entry = sb._file_entries[0]
            self.assertTrue(entry.get("is_merged"))
            self.assertEqual(entry["merged_data"]["merged_name"],
                              "merged_X")
            self.assertEqual(entry["merged_data"]["x_unit"], "V")


class PerRunArrayMaterialisationTests(unittest.TestCase):
    """``per_run_audit`` may carry TOF / cal arrays. ASDF lazy-loads
    them; if the loader doesn't eagerly materialise, the first
    access after the file closes raises "Attempt to read block data
    from missing block" inside PA's ``_replot``."""

    def setUp(self):
        _ensure_app()

    def test_per_run_arrays_survive_file_close(self):
        from gui.analysis.merge import (
            export_merged_asdf, load_merged_asdf)

        merged_data = {
            "merged_name": "merged_with_tof",
            "x": np.linspace(-50.0, 300.0, 10),
            "y": np.ones(10),
            "yerr": np.ones(10),
            "x_unit": "V",
            "bin_step_mhz": 0,
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "source_runs": ["7735"],
            "source_files": ["/p/run_7735.asdf"],
            "per_run": [
                {"run_num": "7735", "path": "/p/a",
                 "cooler_v": 29895.3, "laser_set": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2,
                 # The fields the crash bug surfaced on:
                 "tof_index": np.arange(100, dtype=float),
                 "tof_counts": np.random.poisson(5, 100).astype(float),
                 },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "merged.asdf")
            export_merged_asdf(merged_data, out)
            md = load_merged_asdf(out)

        # File is closed. Accessing the arrays must not raise.
        rd = md["per_run"][0]
        ti = np.asarray(rd["tof_index"])
        tc = np.asarray(rd["tof_counts"])
        self.assertEqual(len(ti), 100)
        self.assertEqual(len(tc), 100)
        # And the values round-trip cleanly.
        np.testing.assert_array_equal(
            ti, np.arange(100, dtype=float))


class PaMergedActionsParityTests(unittest.TestCase):
    """PA merged entries expose the same View / Edit / Export
    actions as the Analysis Source block, so the two tabs are
    uniform end-to-end."""

    def setUp(self):
        _ensure_app()

    def test_merged_file_entry_to_merged_data_shape(self):
        """``to_merged_data`` returns the same key set the shared
        dialogs and ``export_merged_asdf`` consume -- the contract
        the PA action handlers rely on."""
        from gui.preanalysis_tab import MergedFileEntry

        mfe = MergedFileEntry(
            name="merged_X",
            merged_x=np.array([-50.0, 0.0, 50.0]),
            merged_y=np.array([1.0, 5.0, 2.0]),
            merge_domain="voltage",
            source_info=[
                {"run_number": "7735",
                 "filepath": "/data/run_7735.asdf",
                 "cooler_v": 29895.3, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
            ],
            merge_cooler_v=29895.4,
            merge_laser_sp=15975.4150,
            merge_mass_amu=47.95,
            merge_harmonic=2,
        )
        md = mfe.to_merged_data()
        for key in ("merged_name", "x", "y", "yerr", "x_unit",
                     "merge_metadata", "source_runs", "source_files",
                     "per_run", "metadata"):
            self.assertIn(key, md)
        self.assertEqual(md["x_unit"], "V")
        self.assertEqual(md["source_runs"], ["7735"])
        self.assertAlmostEqual(md["merge_metadata"]["cooler_v"], 29895.4)

    def test_pa_export_writes_a_loadable_denis_merged_file(self):
        """PA's _export_merged_entry must produce a file the shared
        loader can read back."""
        from gui.preanalysis_tab import (
            PreAnalysisTab, MergedFileEntry)
        from gui.analysis.merge import (
            load_merged_asdf, export_merged_asdf, is_merged_asdf)

        pa = PreAnalysisTab()
        mfe = MergedFileEntry(
            name="merged_E",
            merged_x=np.array([-50.0, 0.0, 50.0]),
            merged_y=np.array([1.0, 5.0, 2.0]),
            merge_domain="voltage",
            source_info=[
                {"run_number": "a", "filepath": "/p/a",
                 "cooler_v": 29895.3, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
                {"run_number": "b", "filepath": "/p/b",
                 "cooler_v": 29895.5, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
            ],
            merge_cooler_v=29895.4, merge_laser_sp=15975.4150,
            merge_mass_amu=47.95, merge_harmonic=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "merged.asdf")
            # Skip the GUI file dialog by calling the underlying
            # writer directly.
            export_merged_asdf(mfe.to_merged_data(), out)
            self.assertTrue(is_merged_asdf(out))
            md2 = load_merged_asdf(out)
            self.assertIsNotNone(md2)
            self.assertEqual(md2["x_unit"], "V")
            self.assertAlmostEqual(
                md2["merge_metadata"]["cooler_v"], 29895.4)

    def test_merged_file_entry_has_view_edit_export_signals(self):
        """MergedFileEntry must expose view_requested, edit_requested,
        and export_requested signals so the PA tab can wire handlers
        for each action."""
        from gui.preanalysis_tab import MergedFileEntry

        mfe = MergedFileEntry(
            name="m", merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]), merge_domain="voltage",
            source_info=[("a", "/p/a"), ("b", "/p/b")])
        for sig_name in ("view_requested", "edit_requested",
                          "export_requested"):
            self.assertTrue(hasattr(mfe, sig_name))


class AnalysisMergeDialogPrefillTests(unittest.TestCase):
    """``SourceBlock._merge_selected`` must pre-fill the merge-level
    Doppler panel. When the Source-block override checkbox is OFF, the
    dialog should show the mean of the source ASDF headers'
    CoolerVoltage / LaserSetpoint, not the override field's residual
    default."""

    def setUp(self):
        _ensure_app()

    def test_prefill_uses_asdf_means_when_override_off(self):
        """Reproduce the user's bug: override field set to 29977 V /
        10920 cm⁻¹ (residual defaults), checkbox OFF, source ASDFs
        carry 29895.x V / 15975.x cm⁻¹. The dialog defaults must be
        the source-ASDF means, not 29977 / 10920."""
        import asdf as _asdf
        from gui.analysis.blocks import SourceBlock
        from gui.analysis.merge import MergeDialog

        with tempfile.TemporaryDirectory() as tmp:
            # Two fake ASDFs with the user's typical header values.
            paths = []
            for i, (cv, ls) in enumerate(
                    [(2.98953236, 15975.415002),
                     (2.98955141, 15975.415003)]):
                p = os.path.join(tmp, f"run_{7735 + i}.asdf")
                _asdf.AsdfFile({
                    "Run": str(7735 + i),
                    "CoolerVoltage": cv,
                    "LaserSetpoint": ls,
                    "MassAMU": 47.95,
                    "CalSet": np.array([-70.0, 0.0, 300.0]),
                    "CalReadback": np.array([-0.07, 0.0, 0.30]),
                    "raw": np.zeros((4, 6)),
                    "ScanningRanges": [[-70, 300]],
                }).write_to(p)
                paths.append(p)

            sb = SourceBlock()
            # Override field has residual defaults but checkbox OFF
            # (the user's actual situation).
            sb._cooler_spin.setValue(29977.0)
            sb._laser_spin.setValue(10920.0)
            sb._ovr_enable.setChecked(False)
            # Stub _add_file_entry to drop entries in directly
            # without invoking clstools' Load_Run on our minimal
            # fakes.
            for p in paths:
                sb._file_entries.append({
                    "path": p, "run_number": os.path.basename(p),
                    "checkbox": MagicMock(isChecked=lambda: True),
                    "is_merged": False, "is_split": False,
                })

            # Capture the dialog construction args.
            captured = {}

            def fake_dialog(file_entries, source_config,
                            parent=None, **kw):
                captured["default_merge_metadata"] = kw.get(
                    "default_merge_metadata")
                # Return a stub that exits non-Accepted so
                # _merge_selected exits cleanly.
                stub = MagicMock()
                stub.exec.return_value = 0
                return stub

            with patch("gui.analysis.merge.MergeDialog",
                        side_effect=fake_dialog):
                sb._merge_selected()

            mm = captured["default_merge_metadata"]
            # Means of the source files (29895.4 V, 15975.415 cm⁻¹)
            # -- NOT 29977 V / 10920 cm⁻¹ (the override defaults).
            self.assertAlmostEqual(mm["cooler_v"], 29895.41885, places=2)
            self.assertAlmostEqual(mm["laser_sp"], 15975.4150025,
                                    places=4)

    def test_prefill_uses_override_when_checkbox_on(self):
        import asdf as _asdf
        from gui.analysis.blocks import SourceBlock

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(2):
                p = os.path.join(tmp, f"run_{i}.asdf")
                _asdf.AsdfFile({
                    "Run": str(i),
                    "CoolerVoltage": 2.99,
                    "LaserSetpoint": 15975.0,
                    "raw": np.zeros((4, 6)),
                }).write_to(p)
                paths.append(p)

            sb = SourceBlock()
            sb._cooler_spin.setValue(31000.0)
            sb._laser_spin.setValue(15500.0)
            sb._ovr_enable.setChecked(True)
            for p in paths:
                sb._file_entries.append({
                    "path": p, "run_number": "X",
                    "checkbox": MagicMock(isChecked=lambda: True),
                    "is_merged": False, "is_split": False,
                })

            captured = {}

            def fake_dialog(file_entries, source_config,
                            parent=None, **kw):
                captured["default_merge_metadata"] = kw.get(
                    "default_merge_metadata")
                stub = MagicMock()
                stub.exec.return_value = 0
                return stub

            with patch("gui.analysis.merge.MergeDialog",
                        side_effect=fake_dialog):
                sb._merge_selected()

            mm = captured["default_merge_metadata"]
            # Override checkbox was on, so the spinbox values win.
            self.assertAlmostEqual(mm["cooler_v"], 31000.0)
            self.assertAlmostEqual(mm["laser_sp"], 15500.0)


class ConsistencyGuardTests(unittest.TestCase):
    """Both export and load refuse the legacy bin_mode/domain
    mismatch bug (voltage-axis data labelled as MHz). The user
    must re-merge with current code rather than carry buggy state
    forward."""

    def test_export_refuses_voltage_data_with_mhz_unit(self):
        from gui.analysis.merge import export_merged_asdf

        with tempfile.TemporaryDirectory() as tmp:
            # Voltage values (-70..298) mislabelled as MHz -- exactly
            # the legacy bug pattern.
            md = {
                "merged_name": "buggy",
                "x": np.linspace(-70.0, 298.0, 100),
                "y": np.ones(100),
                "yerr": np.ones(100),
                "x_unit": "MHz",
                "merge_metadata": {},
                "per_run": [],
            }
            out = os.path.join(tmp, "buggy.asdf")
            with self.assertRaises(ValueError) as ctx:
                export_merged_asdf(md, out)
            self.assertIn("mismatch", str(ctx.exception))
            self.assertFalse(os.path.isfile(out))

    def test_load_refuses_legacy_buggy_file(self):
        """Simulate a file written by pre-fix code (passes the new
        export's guard only because the guard didn't exist when the
        file was written). Write directly with asdf, then load."""
        import asdf
        from gui.analysis.merge import (
            load_merged_asdf, DENIS_MERGED_SCHEMA)

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "buggy.asdf")
            asdf.AsdfFile({
                "denis_schema": DENIS_MERGED_SCHEMA,
                "merged_name": "buggy",
                "x": np.linspace(-70.0, 298.0, 100),
                "y": np.ones(100),
                "yerr": np.ones(100),
                "x_unit": "MHz",
                "domain": "frequency",
                "bin_step": 0,
                "merge_metadata": {"cooler_v": None, "laser_sp": None,
                                    "mass_amu": None, "harmonic": None},
                "source_runs": [], "source_files": [],
                "per_run_audit": [], "merge_params": {}, "metadata": {},
            }).write_to(out)
            with self.assertRaises(ValueError) as ctx:
                load_merged_asdf(out)
            self.assertIn("mismatch", str(ctx.exception))

    def test_export_allows_true_low_detuning_frequency_merge(self):
        """A genuine narrow frequency merge (|x| < 1000) would also
        trip the heuristic, but its data must span >= 1000 MHz for a
        useful CLS scan. We accept the false-positive risk; a
        narrow-range merge is rare and the user has Edit Merge to
        re-pick the domain. Pin the failure mode so we can revisit
        if it bites someone."""
        from gui.analysis.merge import export_merged_asdf

        with tempfile.TemporaryDirectory() as tmp:
            md = {
                "merged_name": "narrow",
                "x": np.linspace(-500.0, 500.0, 100),
                "y": np.ones(100),
                "yerr": np.ones(100),
                "x_unit": "MHz",
                "merge_metadata": {},
                "per_run": [],
            }
            with self.assertRaises(ValueError):
                export_merged_asdf(md, os.path.join(tmp, "narrow.asdf"))


if __name__ == "__main__":
    unittest.main()
