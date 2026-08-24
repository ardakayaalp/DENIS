"""Tests for the merged-entry source_info schema and merge-level metadata.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Locks in four behaviors of the merge-unification schema:
 1. ``_normalize_source_info`` wraps old 2-tuple / 2-list pairs into
    dicts with explicit ``None`` for the new fields.
 2. ``MergedFileEntry`` accepts the new merge_cooler_v / merge_laser_sp
    / merge_mass_amu / merge_harmonic keyword args and stores them.
 3. The Analysis bridge populates a non-empty ``per_run`` list and a
    ``merge_metadata`` dict on the Source-side merged_data entry.
 4. YAML round-trip preserves the new fields and still loads old YAMLs
    (no merge_* keys, 2-list source_info).
Runs headless from the project root with the project interpreter.

Depends on: gui.preanalysis_tab, gui.analysis.blocks,
cls_estimations.constants.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class NormalizeSourceInfoTests(unittest.TestCase):
    def test_wraps_two_list_into_dict(self):
        from gui.preanalysis_tab import _normalize_source_info

        norm = _normalize_source_info([
            ["7735", "/data/run_7735.asdf"],
            ["7740", "/data/run_7740.asdf"],
        ])

        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0]["run_number"], "7735")
        self.assertEqual(norm[0]["filepath"], "/data/run_7735.asdf")
        for k in ("cooler_v", "laser_sp", "mass_amu", "harmonic"):
            self.assertIsNone(norm[0][k])
            self.assertIsNone(norm[1][k])

    def test_wraps_two_tuple_into_dict(self):
        from gui.preanalysis_tab import _normalize_source_info

        norm = _normalize_source_info([
            ("7735", "/data/run_7735.asdf"),
        ])
        self.assertEqual(norm[0]["run_number"], "7735")
        self.assertEqual(norm[0]["filepath"], "/data/run_7735.asdf")

    def test_preserves_full_dict_fields(self):
        from gui.preanalysis_tab import _normalize_source_info

        norm = _normalize_source_info([
            {"run_number": "7735", "filepath": "/data/run_7735.asdf",
             "cooler_v": 29895.3, "laser_sp": 15975.4150,
             "mass_amu": 47.95, "harmonic": 2},
        ])
        self.assertEqual(norm[0]["cooler_v"], 29895.3)
        self.assertAlmostEqual(norm[0]["laser_sp"], 15975.4150)
        self.assertEqual(norm[0]["mass_amu"], 47.95)
        self.assertEqual(norm[0]["harmonic"], 2)

    def test_partial_dict_missing_keys_become_none(self):
        from gui.preanalysis_tab import _normalize_source_info

        norm = _normalize_source_info([
            {"run_number": "X", "filepath": "/p"},
        ])
        self.assertEqual(norm[0]["run_number"], "X")
        self.assertIsNone(norm[0]["cooler_v"])
        self.assertIsNone(norm[0]["mass_amu"])

    def test_empty_or_none_returns_empty_list(self):
        from gui.preanalysis_tab import _normalize_source_info

        self.assertEqual(_normalize_source_info(None), [])
        self.assertEqual(_normalize_source_info([]), [])


class MergedFileEntrySchemaTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_init_stores_merge_metadata(self):
        from gui.preanalysis_tab import MergedFileEntry

        mfe = MergedFileEntry(
            name="merged_X",
            merged_x=np.array([0.0, 1.0, 2.0]),
            merged_y=np.array([1.0, 2.0, 3.0]),
            merge_domain="voltage",
            source_info=[("a", "/p/a"), ("b", "/p/b")],
            merge_cooler_v=29895.4,
            merge_laser_sp=15975.4150,
            merge_mass_amu=47.95,
            merge_harmonic=2,
        )
        self.assertAlmostEqual(mfe.merge_cooler_v, 29895.4)
        self.assertAlmostEqual(mfe.merge_laser_sp, 15975.4150)
        self.assertAlmostEqual(mfe.merge_mass_amu, 47.95)
        self.assertEqual(mfe.merge_harmonic, 2)

    def test_init_defaults_merge_metadata_to_none(self):
        from gui.preanalysis_tab import MergedFileEntry

        mfe = MergedFileEntry(
            name="merged_X",
            merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]),
            merge_domain="voltage",
            source_info=[("a", "/p/a"), ("b", "/p/b")],
        )
        self.assertIsNone(mfe.merge_cooler_v)
        self.assertIsNone(mfe.merge_laser_sp)
        self.assertIsNone(mfe.merge_mass_amu)
        self.assertIsNone(mfe.merge_harmonic)

    def test_init_normalizes_old_source_info(self):
        from gui.preanalysis_tab import MergedFileEntry

        mfe = MergedFileEntry(
            name="merged_X",
            merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]),
            merge_domain="voltage",
            source_info=[("a", "/p/a")],  # old shape
        )
        # source_info should now be a list of dicts on the instance.
        self.assertIsInstance(mfe.source_info[0], dict)
        self.assertEqual(mfe.source_info[0]["run_number"], "a")
        self.assertEqual(mfe.source_info[0]["filepath"], "/p/a")
        self.assertIsNone(mfe.source_info[0]["cooler_v"])


class AnalysisBridgePopulatesPerRunTests(unittest.TestCase):
    """The Analysis-side bridge must thread per-source metadata into
    per_run and the merge-level Doppler metadata into merge_metadata.
    Phase 4's fit path keys off both."""

    def setUp(self):
        _ensure_app()

    def test_per_run_carries_cooler_laser_when_dicts_present(self):
        from gui.preanalysis_tab import MergedFileEntry
        from gui.analysis.blocks import SourceBlock

        mfe = MergedFileEntry(
            name="merged_X",
            merged_x=np.array([0.0, 50.0]),
            merged_y=np.array([1.0, 2.0]),
            merge_domain="voltage",
            source_info=[
                {"run_number": "7735",
                 "filepath": "/data/run_7735.asdf",
                 "cooler_v": 29895.3, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
                {"run_number": "7740",
                 "filepath": "/data/run_7740.asdf",
                 "cooler_v": 29895.5, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
            ],
            merge_cooler_v=29895.4,
            merge_laser_sp=15975.4150,
            merge_mass_amu=47.95,
            merge_harmonic=2,
        )
        pa = MagicMock()
        pa._file_entries = [mfe]
        pa._e_lower = MagicMock()
        pa._e_lower.value = MagicMock(return_value=0.0)
        pa._e_upper = MagicMock()
        pa._e_upper.value = MagicMock(return_value=25068.222)
        pa._harmonic = MagicMock()
        pa._harmonic.value = MagicMock(return_value=2)

        sb = SourceBlock()
        sb._import_files_from_preanalysis(pa)

        md = sb._file_entries[0]["merged_data"]
        self.assertEqual(len(md["per_run"]), 2)
        self.assertEqual(md["per_run"][0]["run_num"], "7735")
        self.assertAlmostEqual(md["per_run"][0]["cooler_v"], 29895.3)
        self.assertAlmostEqual(md["per_run"][1]["cooler_v"], 29895.5)
        self.assertAlmostEqual(md["per_run"][0]["laser_set"], 15975.4150)

        mm = md["merge_metadata"]
        self.assertAlmostEqual(mm["cooler_v"], 29895.4)
        self.assertAlmostEqual(mm["laser_sp"], 15975.4150)
        self.assertEqual(mm["harmonic"], 2)

    def test_frequency_merge_preserves_per_run_and_metadata(self):
        """Frequency merges store rest-frame MHz; per_run + merge_metadata
        must still be populated. Phase 4 reads them for the audit trail
        even though no V→F shift happens at fit time."""
        from gui.preanalysis_tab import MergedFileEntry
        from gui.analysis.blocks import SourceBlock
        from cls_estimations.constants import C_LIGHT

        # Build a frequency-domain merge whose x was stored as
        # absolute lab-frame MHz (PA convention).
        e_lower_cm = 0.0
        e_upper_cm = 25068.222
        offset_mhz = (e_upper_cm - e_lower_cm) * C_LIGHT * 100.0 / 1e6
        mfe = MergedFileEntry(
            name="merged_freq",
            merged_x=np.array(
                [offset_mhz - 100.0, offset_mhz, offset_mhz + 100.0]),
            merged_y=np.array([1.0, 7.0, 2.0]),
            merge_domain="frequency",
            source_info=[
                {"run_number": "7735",
                 "filepath": "/data/run_7735.asdf",
                 "cooler_v": 29895.3, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
                {"run_number": "7740",
                 "filepath": "/data/run_7740.asdf",
                 "cooler_v": 29895.5, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
            ],
            merge_cooler_v=29895.4,
            merge_laser_sp=15975.4150,
            merge_mass_amu=47.95,
            merge_harmonic=2,
        )
        pa = MagicMock()
        pa._file_entries = [mfe]
        pa._e_lower = MagicMock()
        pa._e_lower.value = MagicMock(return_value=e_lower_cm)
        pa._e_upper = MagicMock()
        pa._e_upper.value = MagicMock(return_value=e_upper_cm)
        pa._harmonic = MagicMock()
        pa._harmonic.value = MagicMock(return_value=2)

        sb = SourceBlock()
        sb._import_files_from_preanalysis(pa)

        md = sb._file_entries[0]["merged_data"]
        # x_unit is MHz and x is ref-shifted to ~0.
        self.assertEqual(md["x_unit"], "MHz")
        np.testing.assert_allclose(md["x"], [-100.0, 0.0, 100.0],
                                    atol=1e-6)
        # per_run + merge_metadata propagate for frequency merges too.
        self.assertEqual(len(md["per_run"]), 2)
        self.assertAlmostEqual(md["per_run"][0]["cooler_v"], 29895.3)
        self.assertAlmostEqual(
            md["merge_metadata"]["cooler_v"], 29895.4)

    def test_old_two_list_source_info_still_imports(self):
        """Regression check for existing PA YAMLs."""
        from gui.preanalysis_tab import MergedFileEntry
        from gui.analysis.blocks import SourceBlock

        mfe = MergedFileEntry(
            name="merged_old",
            merged_x=np.array([-50.0, 0.0, 50.0]),
            merged_y=np.array([1.0, 5.0, 1.0]),
            merge_domain="voltage",
            source_info=[("7735", "/data/run_7735.asdf"),
                         ("7740", "/data/run_7740.asdf")],
        )
        pa = MagicMock()
        pa._file_entries = [mfe]
        pa._e_lower = MagicMock()
        pa._e_lower.value = MagicMock(return_value=0.0)
        pa._e_upper = MagicMock()
        pa._e_upper.value = MagicMock(return_value=25068.222)
        pa._harmonic = MagicMock()
        pa._harmonic.value = MagicMock(return_value=2)

        sb = SourceBlock()
        sb._import_files_from_preanalysis(pa)

        md = sb._file_entries[0]["merged_data"]
        self.assertEqual(md["source_runs"], ["7735", "7740"])
        # per_run is populated with None cooler/laser since the old
        # 2-list shape didn't carry them.
        self.assertEqual(len(md["per_run"]), 2)
        self.assertIsNone(md["per_run"][0]["cooler_v"])
        # merge_metadata propagates None too -- merge_metadata
        # consumers must handle this case (Phase 4: fall back to
        # source-mean computed at use time).
        self.assertIsNone(md["merge_metadata"]["cooler_v"])


class YamlRoundTripTests(unittest.TestCase):
    """Saving + loading a PA config containing a Phase-2 merged entry
    preserves the metadata. Old (Phase-1) configs continue to load."""

    def setUp(self):
        _ensure_app()

    def test_phase2_save_then_load_preserves_merge_metadata(self):
        from gui.preanalysis_tab import (
            PreAnalysisTab, MergedFileEntry)

        pa = PreAnalysisTab()
        # Inject a merged entry with Phase-2 metadata.
        pa._add_merged_entry(
            name="merged_X",
            merged_x=np.array([-50.0, 0.0, 50.0]),
            merged_y=np.array([1.0, 5.0, 1.0]),
            domain="voltage",
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

        d = pa._build_config_dict()
        merged = d["preanalysis"]["merged_entries"][0]
        self.assertEqual(merged["merge_cooler_v"], 29895.4)
        self.assertEqual(merged["merge_harmonic"], 2)
        # source_info is dict-shaped on save.
        self.assertEqual(merged["source_info"][0]["cooler_v"], 29895.3)

        # Reload into a fresh PA tab.
        pa2 = PreAnalysisTab()
        pa2._restore_from_dict(d["preanalysis"])

        mfes = [fe for fe in pa2._file_entries
                if isinstance(fe, MergedFileEntry)]
        self.assertEqual(len(mfes), 1)
        m = mfes[0]
        self.assertAlmostEqual(m.merge_cooler_v, 29895.4)
        self.assertEqual(m.merge_harmonic, 2)
        self.assertEqual(m.source_info[0]["run_number"], "7735")
        self.assertAlmostEqual(m.source_info[0]["cooler_v"], 29895.3)

    def test_yaml_text_roundtrip_preserves_merge_metadata(self):
        """Full save→YAML-text→load cycle. PyYAML round-trips ``None``
        through ``safe_dump``/``safe_load`` as ``null`` -- this test
        pins that down at the schema layer so Phase 4 doesn't have to
        guess what shape an absent-on-disk merge_* field takes."""
        import yaml
        from gui.preanalysis_tab import (
            PreAnalysisTab, MergedFileEntry)

        pa = PreAnalysisTab()
        # Mix one merged entry with full Phase-2 metadata and one
        # with all-None merge_* (the "not yet configured" path).
        pa._add_merged_entry(
            name="merged_full",
            merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]),
            domain="voltage",
            source_info=[
                {"run_number": "7735",
                 "filepath": "/data/run_7735.asdf",
                 "cooler_v": 29895.3, "laser_sp": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
            ],
            merge_cooler_v=29895.3, merge_laser_sp=15975.4150,
            merge_mass_amu=47.95, merge_harmonic=2,
        )
        pa._add_merged_entry(
            name="merged_unset",
            merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]),
            domain="voltage",
            source_info=[("a", "/p/a")],
            # merge_* all None (default)
        )

        d = pa._build_config_dict()
        text = yaml.safe_dump(d)
        d2 = yaml.safe_load(text)

        pa2 = PreAnalysisTab()
        pa2._restore_from_dict(d2["preanalysis"])

        mfes = [fe for fe in pa2._file_entries
                if isinstance(fe, MergedFileEntry)]
        self.assertEqual(len(mfes), 2)
        full = next(m for m in mfes if m.run_number == "merged_full")
        unset = next(m for m in mfes if m.run_number == "merged_unset")
        self.assertAlmostEqual(full.merge_cooler_v, 29895.3)
        self.assertEqual(full.merge_harmonic, 2)
        # None survives the text round-trip (PyYAML null ↔ Python None).
        self.assertIsNone(unset.merge_cooler_v)
        self.assertIsNone(unset.merge_harmonic)

    def test_merged_entries_null_in_yaml_does_not_blow_up(self):
        """A hand-edited config with ``merged_entries: null`` should be
        treated as an empty list, not iterated as None."""
        from gui.preanalysis_tab import PreAnalysisTab

        cfg = {
            "merged_entries": None,   # the pathological case
            "files": [],
            "models": [],
            "plot_options": {},
            "tof_gate": {},
            "binning": {},
        }
        pa = PreAnalysisTab()
        # Should not raise.
        pa._restore_from_dict(cfg)

    def test_phase1_yaml_still_loads(self):
        """A pre-Phase-2 YAML (no merge_* keys, 2-list source_info)
        must load without error and produce a usable MergedFileEntry."""
        from gui.preanalysis_tab import (
            PreAnalysisTab, MergedFileEntry)

        cfg = {
            "merged_entries": [{
                "name": "merged_old",
                "domain": "voltage",
                "x": [-50.0, 0.0, 50.0],
                "y": [1.0, 5.0, 1.0],
                "source_info": [["7735", "/data/run_7735.asdf"],
                                ["7740", "/data/run_7740.asdf"]],
                "color": "#333333",
                "alpha": 1.0,
                "linestyle": "-",
                "checked": True,
            }],
            "files": [],
            "models": [],
            "plot_options": {},
            "tof_gate": {},
            "binning": {},
        }
        pa = PreAnalysisTab()
        pa._restore_from_dict(cfg)

        mfes = [fe for fe in pa._file_entries
                if isinstance(fe, MergedFileEntry)]
        self.assertEqual(len(mfes), 1)
        m = mfes[0]
        # Old format → all merge_* are None.
        self.assertIsNone(m.merge_cooler_v)
        # source_info auto-normalized to dict shape.
        self.assertEqual(m.source_info[0]["run_number"], "7735")
        self.assertEqual(
            m.source_info[1]["filepath"], "/data/run_7740.asdf")
        self.assertIsNone(m.source_info[0]["cooler_v"])


if __name__ == "__main__":
    unittest.main()
