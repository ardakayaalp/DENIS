"""Headless tests: calibrations survive a save, a load, and a moved project.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

A calibration override that quietly fails to persist is worse than no
feature at all: the run still loads, the fit still succeeds, and the
number is silently wrong again. So the YAML round-trip is tested, and so
is the case that actually bites -- a project whose data has moved.

``gui.missing_files`` walks the tab sections to relocate file paths, but
the registries are keyed *by* path, so their keys were invisible to it.
That is the pre-existing bug ``_remap_registry_keys_inplace`` fixes, for
both ``calibrations`` and ``scan_filters``; a ``borrow`` spec's ``donor``
needs relocating too, or the calibration resolves to a dead path.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_calibration_persistence -v

Depends on: gui.calibration, gui.missing_files, PyYAML.
"""

import unittest

import yaml

from gui.calibration import (
    CalibrationRegistry,
    canonical_path,
    set_registry,
)
from gui.missing_files import remap_paths_inplace

OLD_DIR = "C:/old/data"
NEW_DIR = "D:/new/data"


def _p(d, run):
    return f"{d}/run_{run}.asdf"


class YamlRoundTripTests(unittest.TestCase):

    def setUp(self):
        self.reg = CalibrationRegistry()
        set_registry(self.reg)

    def tearDown(self):
        set_registry(None)

    def test_specs_survive_a_yaml_round_trip(self):
        self.reg.set(_p(OLD_DIR, 1002),
                     {"mode": "fit", "reject": "first_n", "drop_first": 3,
                      "note": "HV settling"})
        self.reg.set(_p(OLD_DIR, 1003),
                     {"mode": "borrow", "donor": _p(OLD_DIR, 1001)})
        self.reg.set(_p(OLD_DIR, 1004),
                     {"mode": "coeffs", "coeffs_v": [0.35, 1.0007]})

        text = yaml.dump({"calibrations": self.reg.to_dict()},
                         default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(text)

        other = CalibrationRegistry()
        other.from_dict(raw["calibrations"])

        self.assertEqual(other.get(_p(OLD_DIR, 1002))["drop_first"], 3)
        self.assertEqual(other.get(_p(OLD_DIR, 1002))["note"], "HV settling")
        self.assertEqual(other.get(_p(OLD_DIR, 1003))["donor"],
                         _p(OLD_DIR, 1001))
        self.assertEqual(other.get(_p(OLD_DIR, 1004))["coeffs_v"],
                         [0.35, 1.0007])

    def test_yaml_is_plain_scalars_not_python_objects(self):
        """`yaml.safe_load` must be able to read back what we wrote."""
        self.reg.set(_p(OLD_DIR, 1002),
                     {"mode": "coeffs", "coeffs_v": [0.35, 1.0007]})
        text = yaml.dump({"calibrations": self.reg.to_dict()})
        self.assertNotIn("!!python", text)
        yaml.safe_load(text)     # would raise on a bare numpy float

    def test_an_empty_registry_writes_nothing(self):
        """Older projects that never had the key must stay clean."""
        self.assertEqual(self.reg.to_dict(), {})


class RelocationTests(unittest.TestCase):
    """The failure this guards against: a moved project that loads fine and
    silently reverts every run to its uncorrected calibration.

    Critically, the registry sections are keyed the way the app *actually*
    writes them -- through ``canonical_path``, i.e. ``os.path.normcase`` +
    ``abspath``, which on Windows lowercases and flips the separators -- while
    the file-entry slots (and hence the remap the locate dialog produces) hold
    the raw strings. An earlier version of this test hand-wrote the registry
    keys in the same raw spelling as the file slots, so it passed against a
    ``remap.get(path)`` that could never match in production. Build the keys
    the way the registry does, or the test proves nothing.
    """

    def _project(self):
        reg = CalibrationRegistry()
        reg.set(_p(OLD_DIR, 1002), {"mode": "fit", "reject": "first_n",
                                    "drop_first": 3})
        reg.set(_p(OLD_DIR, 1003), {"mode": "borrow",
                                    "donor": _p(OLD_DIR, 1001)})
        return {
            "analysis": {"projects": [{"blocks": [{
                "type": "Source",
                "files": [{"path": _p(OLD_DIR, 1002), "run_number": "1002"}],
            }]}]},
            "calibrations": reg.to_dict(),          # canonical keys
            "scan_filters": {
                canonical_path(_p(OLD_DIR, 1002)): {"excluded": [4, 7]},
            },
        }

    def test_the_registry_keys_really_are_canonical(self):
        """Guard the guard: if this ever stops being true the tests below
        would go back to proving nothing."""
        proj = self._project()
        self.assertNotIn(_p(OLD_DIR, 1002), proj["calibrations"])
        self.assertIn(canonical_path(_p(OLD_DIR, 1002)), proj["calibrations"])

    def test_calibration_keys_follow_the_files(self):
        proj = self._project()
        remap = {_p(OLD_DIR, 1002): _p(NEW_DIR, 1002)}
        remap_paths_inplace(proj, remap)

        self.assertIn(_p(NEW_DIR, 1002), proj["calibrations"])
        self.assertNotIn(canonical_path(_p(OLD_DIR, 1002)),
                         proj["calibrations"])
        self.assertEqual(
            proj["calibrations"][_p(NEW_DIR, 1002)]["drop_first"], 3)
        # ...and the Source block's own path was rewritten as before.
        block = proj["analysis"]["projects"][0]["blocks"][0]
        self.assertEqual(block["files"][0]["path"], _p(NEW_DIR, 1002))

    def test_a_relocated_calibration_still_resolves_after_a_reload(self):
        """The end of the chain: rewritten key -> registry -> lookup hits."""
        proj = self._project()
        remap_paths_inplace(proj, {_p(OLD_DIR, 1002): _p(NEW_DIR, 1002)})
        reg = CalibrationRegistry()
        reg.from_dict(proj["calibrations"])
        self.assertIsNotNone(reg.get(_p(NEW_DIR, 1002)))
        self.assertEqual(reg.get(_p(NEW_DIR, 1002))["drop_first"], 3)

    def test_a_borrow_donor_is_relocated_too(self):
        """Miss this and the spec points at a dead file, failing the fit."""
        proj = self._project()
        remap_paths_inplace(proj, {_p(OLD_DIR, 1001): _p(NEW_DIR, 1001)})
        self.assertEqual(
            proj["calibrations"][canonical_path(_p(OLD_DIR, 1003))]["donor"],
            _p(NEW_DIR, 1001))

    def test_scan_filter_keys_follow_the_files_as_well(self):
        proj = self._project()
        remap_paths_inplace(proj, {_p(OLD_DIR, 1002): _p(NEW_DIR, 1002)})
        self.assertIn(_p(NEW_DIR, 1002), proj["scan_filters"])
        self.assertEqual(
            proj["scan_filters"][_p(NEW_DIR, 1002)]["excluded"], [4, 7])

    def test_unrelocated_entries_are_left_alone(self):
        proj = self._project()
        remap_paths_inplace(proj, {_p(OLD_DIR, 1002): _p(NEW_DIR, 1002)})
        self.assertIn(canonical_path(_p(OLD_DIR, 1003)),
                      proj["calibrations"])

    def test_acknowledgements_follow_the_files(self):
        """Leave an ack behind on a moved project and the run starts blinking
        its warning again on every load -- which trains the user to ignore it."""
        proj = self._project()
        proj["calibration_acks"] = [canonical_path(_p(OLD_DIR, 1002)),
                                    canonical_path(_p(OLD_DIR, 1009))]
        remap_paths_inplace(proj, {_p(OLD_DIR, 1002): _p(NEW_DIR, 1002)})

        reg = CalibrationRegistry()
        reg.acks_from_list(proj["calibration_acks"])
        self.assertTrue(reg.is_acknowledged(_p(NEW_DIR, 1002)))
        self.assertFalse(reg.is_acknowledged(_p(OLD_DIR, 1002)))
        # the unrelocated one is left alone
        self.assertTrue(reg.is_acknowledged(_p(OLD_DIR, 1009)))

    def test_an_empty_remap_changes_nothing(self):
        proj = self._project()
        before = yaml.dump(proj, sort_keys=True)
        remap_paths_inplace(proj, {})
        self.assertEqual(yaml.dump(proj, sort_keys=True), before)

    def test_a_project_with_no_registries_still_remaps(self):
        proj = {"analysis": {"projects": [{"blocks": [{
            "type": "Source",
            "files": [{"path": _p(OLD_DIR, 1002)}]}]}]}}
        n = remap_paths_inplace(proj, {_p(OLD_DIR, 1002): _p(NEW_DIR, 1002)})
        self.assertEqual(n, 1)


class MainWindowWiringTests(unittest.TestCase):
    """The five sites where scan_filters is handled must handle calibrations
    too -- a save that forgets one of them loses the override on reload."""

    def test_main_window_reads_and_writes_the_calibrations_key(self):
        import inspect
        from gui import main_window
        src = inspect.getsource(main_window)
        # save-all, save-tab (set + pop), load-tab merge, load-all replace
        self.assertGreaterEqual(src.count('"calibrations"'), 5)
        self.assertIn('raw.get("calibrations")', src)

    def test_preanalysis_persists_the_registries_on_its_own_save(self):
        import inspect
        from gui import preanalysis_tab
        src = inspect.getsource(preanalysis_tab)
        self.assertIn('d["calibrations"] = cal', src)
        self.assertIn('raw.get("calibrations")', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
