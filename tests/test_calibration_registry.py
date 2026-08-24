"""Headless tests for gui.calibration's cross-tab registry.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

A calibration override is a property of the *file*, so it lives in a
process-wide registry keyed by canonical path -- the same shape as
gui.scan_filter's ScanFilterRegistry, and for the same reason: both
Pre-Analysis and Analysis can hold the same physical run, and they must
not disagree about how its voltages were computed.

Covers path canonicalization (the same file reached through different
spellings must hit one entry), the change signal that keeps badges and
cached spectra in step, and YAML round-tripping -- including tolerance
for a hand-edited project file, which is the realistic way this dict
gets corrupted.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_calibration_registry -v

Depends on: gui.calibration (CalibrationRegistry, get_registry,
set_registry), PySide6.
"""

import unittest

from gui.calibration import (
    CalibrationRegistry,
    canonical_path,
    get_registry,
    set_registry,
)

FIT = {"mode": "fit", "reject": "first_n", "drop_first": 3}
BORROW = {"mode": "borrow", "donor": "C:/data/run_1001.asdf"}


class RegistryBasicsTests(unittest.TestCase):

    def setUp(self):
        self.reg = CalibrationRegistry()

    def test_unknown_path_is_the_file_default(self):
        self.assertIsNone(self.reg.get("C:/data/run_1002.asdf"))
        self.assertFalse(self.reg.has("C:/data/run_1002.asdf"))

    def test_set_then_get(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        spec = self.reg.get("C:/data/run_1002.asdf")
        self.assertEqual(spec["mode"], "fit")
        self.assertEqual(spec["drop_first"], 3)
        self.assertTrue(self.reg.has("C:/data/run_1002.asdf"))

    def test_get_returns_a_copy(self):
        """Callers must not be able to mutate the registry by accident."""
        self.reg.set("C:/data/run_1002.asdf", FIT)
        got = self.reg.get("C:/data/run_1002.asdf")
        got["drop_first"] = 99
        self.assertEqual(self.reg.get("C:/data/run_1002.asdf")["drop_first"], 3)

    def test_clear_restores_the_file_default(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.clear("C:/data/run_1002.asdf")
        self.assertIsNone(self.reg.get("C:/data/run_1002.asdf"))

    def test_setting_an_inert_spec_removes_the_entry(self):
        """A 'fit' that excludes nothing IS the file default; it must not
        linger as an override and light up a badge."""
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.set("C:/data/run_1002.asdf",
                     {"mode": "fit", "reject": "none", "excluded": []})
        self.assertFalse(self.reg.has("C:/data/run_1002.asdf"))

    def test_setting_none_removes_the_entry(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.set("C:/data/run_1002.asdf", None)
        self.assertFalse(self.reg.has("C:/data/run_1002.asdf"))

    def test_clear_all(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.set("C:/data/run_1003.asdf", BORROW)
        self.reg.clear_all()
        self.assertEqual(self.reg.all_paths(), [])

    def test_malformed_spec_is_rejected_at_the_door(self):
        self.reg.set("C:/data/run_1002.asdf", {"mode": "wishful"})
        self.assertFalse(self.reg.has("C:/data/run_1002.asdf"))


class PathCanonicalizationTests(unittest.TestCase):
    """The same run reached through different spellings is one entry --
    otherwise Pre-Analysis and Analysis could hold contradictory
    calibrations for the same physical file."""

    def setUp(self):
        self.reg = CalibrationRegistry()

    def test_slash_direction_does_not_split_the_entry(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.assertTrue(self.reg.has("C:\\data\\run_1002.asdf"))

    def test_case_does_not_split_the_entry(self):
        self.reg.set("C:/Data/Run_1002.asdf", FIT)
        self.assertTrue(self.reg.has("c:/data/run_1002.asdf"))

    def test_stored_keys_are_canonical(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.assertEqual(self.reg.all_paths(),
                         [canonical_path("C:/data/run_1002.asdf")])


class SignalTests(unittest.TestCase):
    """Badges, open dialogs and Pre-Analysis' cached spectra all refresh off
    this signal; a mutation that stays silent leaves a stale spectrum on
    screen."""

    def setUp(self):
        self.reg = CalibrationRegistry()
        self.hits = []
        self.reg.calibrations_changed.connect(lambda: self.hits.append(1))

    def test_set_emits(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.assertEqual(len(self.hits), 1)

    def test_clear_emits(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.clear("C:/data/run_1002.asdf")
        self.assertEqual(len(self.hits), 2)

    def test_clear_all_emits(self):
        self.reg.clear_all()
        self.assertEqual(len(self.hits), 1)

    def test_from_dict_emits_so_a_project_load_refreshes_the_ui(self):
        self.reg.from_dict({"C:/data/run_1002.asdf": FIT})
        self.assertEqual(len(self.hits), 1)


class AcknowledgementTests(unittest.TestCase):
    """The blinking warning must stop once it has been read.

    A warning that keeps nagging after the user has seen it and decided is one
    they learn to click past -- and then it is worth nothing on the run where
    it actually matters.
    """

    def setUp(self):
        self.reg = CalibrationRegistry()

    def test_unacknowledged_by_default(self):
        self.assertFalse(self.reg.is_acknowledged("C:/data/run_1002.asdf"))

    def test_acknowledge_then_unacknowledge(self):
        self.reg.acknowledge("C:/data/run_1002.asdf")
        self.assertTrue(self.reg.is_acknowledged("C:/data/run_1002.asdf"))
        self.reg.unacknowledge("C:/data/run_1002.asdf")
        self.assertFalse(self.reg.is_acknowledged("C:/data/run_1002.asdf"))

    def test_acknowledge_is_path_canonical(self):
        self.reg.acknowledge("C:/Data/Run_1002.asdf")
        self.assertTrue(self.reg.is_acknowledged("c:\\data\\run_1002.asdf"))

    def test_acknowledge_emits_so_the_badge_hides(self):
        hits = []
        self.reg.calibrations_changed.connect(lambda: hits.append(1))
        self.reg.acknowledge("C:/data/run_1002.asdf")
        self.assertEqual(len(hits), 1)
        self.reg.acknowledge("C:/data/run_1002.asdf")   # already acked
        self.assertEqual(len(hits), 1)                  # no redundant emit

    def test_acks_round_trip(self):
        self.reg.acknowledge("C:/data/run_1002.asdf")
        self.reg.acknowledge("C:/data/run_1003.asdf")
        dumped = self.reg.acks_to_list()

        other = CalibrationRegistry()
        other.acks_from_list(dumped)
        self.assertTrue(other.is_acknowledged("C:/data/run_1002.asdf"))
        self.assertTrue(other.is_acknowledged("C:/data/run_1003.asdf"))

    def test_acks_are_not_in_the_spec_map(self):
        """to_dict() is what the fit subprocess is handed. Whether a human has
        read a warning must never influence a fit."""
        self.reg.acknowledge("C:/data/run_1002.asdf")
        self.assertEqual(self.reg.to_dict(), {})

    def test_acks_from_list_tolerates_junk(self):
        self.reg.acks_from_list(["C:/data/run_1002.asdf", None, 42, ""])
        self.assertEqual(len(self.reg.acks_to_list()), 1)

    def test_clear_all_clears_acks_too(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.acknowledge("C:/data/run_1003.asdf")
        self.reg.clear_all()
        self.assertEqual(self.reg.acks_to_list(), [])


class SerializationTests(unittest.TestCase):

    def setUp(self):
        self.reg = CalibrationRegistry()

    def test_round_trip(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.set("C:/data/run_1003.asdf", BORROW)
        dumped = self.reg.to_dict()

        other = CalibrationRegistry()
        other.from_dict(dumped)
        self.assertEqual(other.to_dict(), dumped)
        self.assertEqual(other.get("C:/data/run_1003.asdf")["donor"],
                         "C:/data/run_1001.asdf")

    def test_to_dict_is_plain_types_for_yaml(self):
        self.reg.set("C:/data/run_1002.asdf",
                     {"mode": "coeffs", "coeffs_v": [0.35, 1.0007]})
        d = self.reg.to_dict()
        spec = next(iter(d.values()))
        self.assertIsInstance(spec, dict)
        self.assertIsInstance(spec["coeffs_v"], list)
        self.assertTrue(all(isinstance(c, float) for c in spec["coeffs_v"]))

    def test_to_dict_is_a_snapshot_not_a_view(self):
        """It doubles as the map handed to the fit subprocess."""
        self.reg.set("C:/data/run_1002.asdf", FIT)
        snap = self.reg.to_dict()
        self.reg.clear_all()
        self.assertEqual(len(snap), 1)

    def test_from_dict_replaces_rather_than_merges(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.from_dict({"C:/data/run_1003.asdf": BORROW})
        self.assertFalse(self.reg.has("C:/data/run_1002.asdf"))
        self.assertTrue(self.reg.has("C:/data/run_1003.asdf"))

    def test_from_dict_survives_a_hand_edited_yaml(self):
        self.reg.from_dict({
            "C:/data/run_1002.asdf": FIT,             # good
            "C:/data/run_1003.asdf": "not a dict",    # junk
            "C:/data/run_1004.asdf": {"mode": "borrow"},   # no donor
            "C:/data/run_1005.asdf": {"mode": "coeffs", "coeffs_v": ["x"]},
        })
        self.assertEqual(self.reg.all_paths(),
                         [canonical_path("C:/data/run_1002.asdf")])

    def test_from_dict_of_none_clears(self):
        self.reg.set("C:/data/run_1002.asdf", FIT)
        self.reg.from_dict(None)
        self.assertEqual(self.reg.all_paths(), [])


class SingletonTests(unittest.TestCase):
    """Tabs reach the registry through get_registry() rather than a
    MainWindow attribute, so a Pre-Analysis dialog and the Analysis tab
    provably see the same object."""

    def tearDown(self):
        set_registry(None)

    def test_get_registry_is_stable(self):
        set_registry(None)
        self.assertIs(get_registry(), get_registry())

    def test_set_registry_installs_an_isolated_instance(self):
        fresh = CalibrationRegistry()
        set_registry(fresh)
        self.assertIs(get_registry(), fresh)

    def test_a_write_from_one_caller_is_visible_to_another(self):
        set_registry(CalibrationRegistry())
        get_registry().set("C:/data/run_1002.asdf", FIT)
        self.assertTrue(get_registry().has("C:/data/run_1002.asdf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
