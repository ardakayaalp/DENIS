"""Tests for importing Pre-Analysis merged entries into the Analysis tab.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Verifies that SourceBlock._import_files_from_preanalysis translates a
Pre-Analysis MergedFileEntry into an Analysis-side merged_data dict,
including the frequency-domain x-axis shift to the rest-frame
transition, that the merged_data sent to the fit pool is strippable to a
picklable form, and that legacy YAMLs encoding source_info as
!!python/tuple still load under yaml.safe_load.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_import_preanalysis_merged -v

Depends on: gui.preanalysis_tab (MergedFileEntry, SafeLoader registration),
gui.analysis.blocks (SourceBlock), cls_estimations.constants (C_LIGHT);
PySide6 for the headless QApplication.
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


class ImportPreAnalysisMergedTests(unittest.TestCase):
    """The Pull-from-Pre-Analysis button must translate a MergedFileEntry
    into an Analysis-side merged_data dict (so the Preview button doesn't
    later try to load '[merged] <name>' as an ASDF path)."""

    def setUp(self):
        _ensure_app()

    def test_voltage_domain_merge_becomes_analysis_merged_entry(self):
        from gui.preanalysis_tab import MergedFileEntry
        from gui.analysis.blocks import SourceBlock

        # Build a fake pre-analysis tab with one MergedFileEntry
        # (voltage-domain merge of two runs).
        mfe = MergedFileEntry(
            name="merged_7735_7740",
            merged_x=np.array([-50.0, 0.0, 50.0, 100.0]),
            merged_y=np.array([2.0, 5.0, 11.0, 4.0]),
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

        # Should have ONE merged entry, not a fake-path regular entry.
        self.assertEqual(len(sb._file_entries), 1)
        e = sb._file_entries[0]
        self.assertTrue(e.get("is_merged"))
        self.assertEqual(e["path"], "merged://merged_7735_7740")

        md = e["merged_data"]
        self.assertEqual(md["merged_name"], "merged_7735_7740")
        self.assertEqual(md["x_unit"], "V")
        np.testing.assert_array_equal(md["x"], [-50.0, 0.0, 50.0, 100.0])
        np.testing.assert_array_equal(md["y"], [2.0, 5.0, 11.0, 4.0])
        self.assertEqual(md["source_runs"], ["7735", "7740"])
        self.assertEqual(md["source_files"],
                         ["/data/run_7735.asdf", "/data/run_7740.asdf"])

    def test_frequency_domain_merge_x_is_ref_shifted(self):
        from gui.preanalysis_tab import MergedFileEntry
        from gui.analysis.blocks import SourceBlock
        from cls_estimations.constants import C_LIGHT

        # Pre-Analysis stores ABSOLUTE MHz for frequency-domain merges;
        # Analysis expects MHz relative to the rest-frame transition.
        # The import must subtract transition_hz / 1e6 from the x array.
        e_lower_cm = 0.0
        e_upper_cm = 25068.222
        harmonic = 2
        transition_cm = e_upper_cm - e_lower_cm
        # Pre-Analysis applies (transition / harmonic) * harmonic = transition
        # in the display-axis subtraction, so the equivalent shift is:
        offset_mhz = transition_cm * C_LIGHT * 100.0 / 1e6
        absolute_mhz = np.array(
            [offset_mhz - 100.0, offset_mhz, offset_mhz + 100.0])
        mfe = MergedFileEntry(
            name="merged_freq",
            merged_x=absolute_mhz,
            merged_y=np.array([1.0, 7.0, 2.0]),
            merge_domain="frequency",
            source_info=[("7735", "/data/run_7735.asdf"),
                         ("7740", "/data/run_7740.asdf")],
        )
        pa = MagicMock()
        pa._file_entries = [mfe]
        pa._e_lower = MagicMock()
        pa._e_lower.value = MagicMock(return_value=e_lower_cm)
        pa._e_upper = MagicMock()
        pa._e_upper.value = MagicMock(return_value=e_upper_cm)
        pa._harmonic = MagicMock()
        pa._harmonic.value = MagicMock(return_value=harmonic)

        sb = SourceBlock()
        sb._import_files_from_preanalysis(pa)

        md = sb._file_entries[0]["merged_data"]
        self.assertEqual(md["x_unit"], "MHz")
        # After ref-shift, the centred element should be ~0.
        np.testing.assert_allclose(md["x"], [-100.0, 0.0, 100.0], atol=1e-6)

    def test_regular_file_entries_still_imported_as_files(self):
        """Mixing regular and merged entries: regulars go through
        _add_file_entry(path); merged go through _add_merged_entry."""
        from gui.preanalysis_tab import MergedFileEntry
        from gui.analysis.blocks import SourceBlock

        regular = MagicMock()
        regular.filepath = "/data/run_7735.asdf"
        regular._is_merged = False
        # MergedFileEntry has _is_merged=True attribute.
        merged = MergedFileEntry(
            name="merged_X",
            merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([0.0, 1.0]),
            merge_domain="voltage",
            source_info=[("a", "/p/a"), ("b", "/p/b")],
        )
        pa = MagicMock()
        pa._file_entries = [regular, merged]
        pa._e_lower = MagicMock()
        pa._e_lower.value = MagicMock(return_value=0.0)
        pa._e_upper = MagicMock()
        pa._e_upper.value = MagicMock(return_value=25068.222)
        pa._harmonic = MagicMock()
        pa._harmonic.value = MagicMock(return_value=2)

        sb = SourceBlock()
        # Stub _add_file_entry to avoid actual filesystem access
        sb._add_file_entry = MagicMock(side_effect=lambda p: sb._file_entries.append({"path": p, "is_merged": False}))
        sb._import_files_from_preanalysis(pa)

        self.assertEqual(len(sb._file_entries), 2)
        self.assertEqual(sb._file_entries[0]["path"], "/data/run_7735.asdf")
        self.assertFalse(sb._file_entries[0]["is_merged"])
        self.assertTrue(sb._file_entries[1]["is_merged"])
        self.assertEqual(sb._file_entries[1]["path"], "merged://merged_X")


class MergedDataPicklabilityTests(unittest.TestCase):
    """merged_data sent to the fit pool must not carry Qt widgets.
    _add_merged_entry stores a back-reference `_entry` (containing a
    QCheckBox + QWidget) for GUI removal; the fit submit path strips
    that key so multiprocessing.pickle doesn't choke on it.
    """

    def setUp(self):
        _ensure_app()

    def test_merged_data_map_strips_entry_backref(self):
        import pickle
        from gui.analysis.blocks import SourceBlock

        sb = SourceBlock()
        merged_data = {
            "merged_name": "merged_X",
            "x": np.array([0.0, 1.0]),
            "y": np.array([1.0, 2.0]),
            "yerr": np.array([1.0, 1.4]),
            "x_unit": "MHz",
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "per_run": [],
            "bin_step_mhz": 0,
        }
        sb._add_merged_entry(merged_data)
        # _add_merged_entry mutates merged_data with a back-reference.
        self.assertIn("_entry", merged_data)
        # Pickling the raw merged_data would fail (QCheckBox inside).
        with self.assertRaises(TypeError):
            pickle.dumps(merged_data)
        # The fit-submit path strips _entry; the stripped copy must pickle.
        sent = {k: v for k, v in merged_data.items() if k != "_entry"}
        pickle.dumps(sent)


class LegacyTuplesInYamlLoadTests(unittest.TestCase):
    """Older Pre-Analysis YAMLs encode merged source_info as
    !!python/tuple. yaml.safe_load must not error on those — see the
    SafeLoader constructor registered at the top of preanalysis_tab."""

    def setUp(self):
        _ensure_app()

    def test_safe_load_accepts_python_tuple_tag(self):
        import yaml

        # Import preanalysis_tab to ensure the SafeLoader constructor
        # registration runs.
        import gui.preanalysis_tab  # noqa: F401

        yaml_text = (
            "preanalysis:\n"
            "  merged_entries:\n"
            "    - name: merged_7735_7740\n"
            "      domain: voltage\n"
            "      x: [0.0, 1.0]\n"
            "      y: [0.0, 2.0]\n"
            "      source_info:\n"
            "        - !!python/tuple\n"
            "          - '7735'\n"
            "          - /data/run_7735.asdf\n"
            "        - !!python/tuple\n"
            "          - '7740'\n"
            "          - /data/run_7740.asdf\n"
        )
        cfg = yaml.safe_load(yaml_text)
        merged = cfg["preanalysis"]["merged_entries"][0]
        # Tuples are loaded as lists, so the structure is iterable
        # the same way. s[0] / s[1] still work.
        self.assertEqual(merged["source_info"][0][0], "7735")
        self.assertEqual(
            merged["source_info"][0][1], "/data/run_7735.asdf")
        self.assertEqual(merged["source_info"][1][0], "7740")


if __name__ == "__main__":
    unittest.main()
