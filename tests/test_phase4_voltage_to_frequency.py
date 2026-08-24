"""Tests for projecting a voltage-merged spectrum onto rest-frame MHz.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

A voltage-merged spectrum fitted in frequency mode is re-projected
onto rest-frame MHz at fit time using the user-picked merge metadata,
treating the merged spectrum as a synthetic ASDF. These tests check
the Doppler array projection against the per-event path, the
ref-frequency detuning shift, the cooler/laser spread warnings, the
auto-fit short-circuit, and the merge-view preview. Runs headless
from the project root with the project interpreter.

Depends on: cls_estimations.* (doppler, constants), gui.analysis.*
(merge, binning, project).
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


class VoltageToFrequencyArrayTests(unittest.TestCase):
    def test_matches_pa_per_file_doppler(self):
        """``voltage_to_frequency_array(v, ...)`` must agree with the
        per-event PA path ``_voltage_to_frequency``: same Doppler
        formula, just applied to binned voltage centers."""
        from cls_estimations.doppler import (
            voltage_to_frequency_array,
            beta_from_voltage, nu_seen_by_ion)
        from cls_estimations.constants import C_LIGHT

        v = np.array([-50.0, 0.0, 50.0])
        cooler = 29895.4
        laser = 15975.4150
        mass = 47.95
        harmonic = 2
        beam_v = np.clip(cooler - v, 1.0, None)
        nu_laser_mhz = laser * harmonic * C_LIGHT * 100.0 / 1e6
        beta = beta_from_voltage(beam_v, mass, 1)
        expected = nu_seen_by_ion(nu_laser_mhz, beta, 'anti-collinear')

        actual = voltage_to_frequency_array(
            v, cooler, laser, mass, harmonic, ref_freq_hz=0.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_ref_freq_shifts_to_detuning_axis(self):
        """Non-zero ref_freq_hz subtracts the rest-frame reference so
        the result is detuning MHz rather than absolute frequency."""
        from cls_estimations.doppler import voltage_to_frequency_array

        v = np.array([0.0])
        no_shift = voltage_to_frequency_array(
            v, 29895.4, 15975.4150, 47.95, 2, ref_freq_hz=0.0)
        ref_hz = 9.583e14
        with_shift = voltage_to_frequency_array(
            v, 29895.4, 15975.4150, 47.95, 2, ref_freq_hz=ref_hz)
        np.testing.assert_allclose(
            with_shift, no_shift - ref_hz / 1e6, rtol=1e-12)


class ProjectVoltageMergeToFrequencyTests(unittest.TestCase):
    def test_uses_merge_metadata_when_provided(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency

        merged_data = {
            "merged_name": "merged_v",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [],
        }
        src_cfg = {"ref_freq": 0.0}
        x, y, yerr, warnings = project_voltage_merge_to_frequency(
            merged_data, src_cfg)
        # Same length, monotone in voltage (Doppler is monotone given
        # one cooler).
        self.assertEqual(len(x), 3)
        self.assertTrue(np.all(np.diff(x) != 0))
        # Counts/yerr preserved.
        np.testing.assert_array_equal(y, merged_data["y"])
        np.testing.assert_array_equal(yerr, merged_data["yerr"])
        codes = [w["code"] for w in warnings]
        self.assertIn("VOLTAGE_TO_FREQUENCY_PROJECTION", codes)
        # No spread warnings (per_run is empty).
        self.assertNotIn("MERGE_COOLER_SPREAD", codes)
        self.assertNotIn("MERGE_LASER_SPREAD", codes)

    def test_falls_back_to_per_run_means_when_metadata_unset(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency

        merged_data = {
            "merged_name": "merged_v",
            "x": np.array([0.0]),
            "y": np.array([5.0]),
            "yerr": np.array([2.2]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": None, "laser_sp": None,
                                "mass_amu": None, "harmonic": None},
            "per_run": [
                {"run_num": "a", "cooler_v": 29895.3,
                 "laser_set": 15975.4150, "mass_amu": 47.95,
                 "harmonic": 2},
                {"run_num": "b", "cooler_v": 29895.5,
                 "laser_set": 15975.4150, "mass_amu": 47.95,
                 "harmonic": 2},
            ],
        }
        x, y, yerr, _w = project_voltage_merge_to_frequency(
            merged_data, {"ref_freq": 0.0})
        # Computed without error -- means came from per_run.
        self.assertEqual(len(x), 1)

    def test_raises_when_no_metadata_available(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency

        merged_data = {
            "merged_name": "merged_v",
            "x": np.array([0.0]),
            "y": np.array([5.0]),
            "yerr": np.array([2.2]),
            "x_unit": "V",
            "merge_metadata": {},
            "per_run": [],
        }
        with self.assertRaises(ValueError) as ctx:
            project_voltage_merge_to_frequency(
                merged_data, {"ref_freq": 0.0})
        self.assertIn("merge_metadata", str(ctx.exception))

    def test_cooler_spread_above_tolerance_emits_warning(self):
        """When source files differ in cooler voltage by more than
        SPREAD_COOLER_V, the projection function emits a warning so
        the user knows the single-Doppler-shift is approximate."""
        from gui.analysis.merge import (
            project_voltage_merge_to_frequency, SPREAD_COOLER_V)

        merged_data = {
            "merged_name": "merged_wide",
            "x": np.array([0.0]),
            "y": np.array([1.0]),
            "yerr": np.array([1.0]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [
                {"run_num": "a", "cooler_v": 29893.0,
                 "laser_set": 15975.4150, "mass_amu": 47.95,
                 "harmonic": 2},
                {"run_num": "b", "cooler_v": 29897.0,
                 "laser_set": 15975.4150, "mass_amu": 47.95,
                 "harmonic": 2},
            ],
        }
        _, _, _, warnings = project_voltage_merge_to_frequency(
            merged_data, {"ref_freq": 0.0})
        codes = [w["code"] for w in warnings]
        self.assertIn("MERGE_COOLER_SPREAD", codes)
        # Spread is 4 V, well above the 0.5 V tolerance.
        self.assertGreater(4.0, SPREAD_COOLER_V)

    def test_laser_spread_above_tolerance_emits_warning(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency

        merged_data = {
            "merged_name": "merged_laser",
            "x": np.array([0.0]),
            "y": np.array([1.0]),
            "yerr": np.array([1.0]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [
                {"run_num": "a", "cooler_v": 29895.4,
                 "laser_set": 15975.4140, "mass_amu": 47.95,
                 "harmonic": 2},
                {"run_num": "b", "cooler_v": 29895.4,
                 "laser_set": 15975.4160, "mass_amu": 47.95,
                 "harmonic": 2},
            ],
        }
        _, _, _, warnings = project_voltage_merge_to_frequency(
            merged_data, {"ref_freq": 0.0})
        codes = [w["code"] for w in warnings]
        self.assertIn("MERGE_LASER_SPREAD", codes)

    def test_close_sources_dont_warn(self):
        """Reproduces the user's actual case: cooler 29895.3 vs 29895.5
        (0.2 V spread, below the 0.5 V tolerance) must NOT trigger a
        warning -- otherwise every normal merge gets noise."""
        from gui.analysis.merge import project_voltage_merge_to_frequency

        merged_data = {
            "merged_name": "merged_close",
            "x": np.array([0.0]),
            "y": np.array([1.0]),
            "yerr": np.array([1.0]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [
                {"run_num": "7735", "cooler_v": 29895.3,
                 "laser_set": 15975.4150, "mass_amu": 47.95,
                 "harmonic": 2},
                {"run_num": "7740", "cooler_v": 29895.5,
                 "laser_set": 15975.4150, "mass_amu": 47.95,
                 "harmonic": 2},
            ],
        }
        _, _, _, warnings = project_voltage_merge_to_frequency(
            merged_data, {"ref_freq": 0.0})
        codes = [w["code"] for w in warnings]
        self.assertNotIn("MERGE_COOLER_SPREAD", codes)
        self.assertNotIn("MERGE_LASER_SPREAD", codes)


class AbsoluteDopplerReferenceTests(unittest.TestCase):
    """Non-circular check: compares the output against an independently
    hand-computed expected MHz. This catches a common drift in both
    ``voltage_to_frequency_array`` and the per-event PA path that a
    mutual-agreement check between them would miss."""

    def test_zero_scan_voltage_matches_textbook_doppler(self):
        from cls_estimations.doppler import voltage_to_frequency_array
        from cls_estimations.constants import (
            C_LIGHT, AMU_TO_KG, E_CHARGE)
        import math

        # Hand-computed: V=0, cooler=29895.4 V, laser=15975.4150 cm^-1,
        # mass=47.95 amu, harmonic=2, anti-collinear.
        cooler = 29895.4
        laser_cm = 15975.4150
        mass = 47.95
        harmonic = 2

        # nu_laser_seen_at_ion = harmonic * laser * c (in MHz)
        nu_laser_mhz = laser_cm * harmonic * C_LIGHT * 100.0 / 1e6
        # beta from V=cooler-0=29895.4 V acceleration
        beam_v = cooler  # because scan V = 0
        m = mass * AMU_TO_KG
        E0 = m * C_LIGHT ** 2
        Etot = 1 * E_CHARGE * beam_v + E0
        beta = math.sqrt(1.0 - (E0 ** 2) / (Etot ** 2))
        # anti-collinear: nu_seen = nu_laser * sqrt((1+beta)/(1-beta))
        expected = nu_laser_mhz * math.sqrt(
            (1.0 + beta) / (1.0 - beta))

        actual = voltage_to_frequency_array(
            np.array([0.0]), cooler, laser_cm, mass, harmonic,
            ref_freq_hz=0.0)
        np.testing.assert_allclose(
            actual[0], expected, rtol=1e-12,
            err_msg=(f"Hand-computed Doppler MHz: {expected}; "
                     f"got {actual[0]}. Off by "
                     f"{actual[0] - expected} MHz."))


class WarningsSurfaceTests(unittest.TestCase):
    """The V→F projection warnings must flow through
    ``build_binning_warnings`` into the fit report channel, not get
    stashed in binning_info and forgotten."""

    def test_voltage_to_frequency_warnings_pulled_into_warnings_list(self):
        from gui.analysis.binning import build_binning_warnings

        info = {
            "bin_mode": "Frequency",
            "voltage_to_frequency_warnings": [
                {"code": "VOLTAGE_TO_FREQUENCY_PROJECTION",
                 "level": "info", "run": "merged_X",
                 "message": "Projected."},
                {"code": "MERGE_COOLER_SPREAD",
                 "level": "warning", "run": "merged_X",
                 "message": "Spread = 2.0 V."},
            ],
        }
        warnings = build_binning_warnings(
            [("merged_X", info)])
        codes = [w["code"] for w in warnings]
        self.assertIn("VOLTAGE_TO_FREQUENCY_PROJECTION", codes)
        self.assertIn("MERGE_COOLER_SPREAD", codes)

    def test_no_v2f_warnings_when_absent(self):
        """build_binning_warnings shouldn't synthesize anything when
        voltage_to_frequency_warnings is missing -- it should pass
        through normally for non-merged sources."""
        from gui.analysis.binning import build_binning_warnings

        info = {"bin_mode": "Frequency", "fallback_used": False,
                "effective_bin_width_mhz": 1.0}
        warnings = build_binning_warnings([("run_X", info)])
        codes = [w["code"] for w in warnings]
        self.assertNotIn("VOLTAGE_TO_FREQUENCY_PROJECTION", codes)


class ZeroPerRunValuesIgnoredTests(unittest.TestCase):
    """A per_run entry whose ASDF missed VCoolDiv reports cooler_v=0;
    that must not throw the spread-warning logic by making
    max-min equal to the full cooler magnitude (~30 kV)."""

    def test_per_run_zero_cooler_does_not_trigger_spread(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency

        merged_data = {
            "merged_name": "merged_z",
            "x": np.array([0.0]),
            "y": np.array([1.0]),
            "yerr": np.array([1.0]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [
                {"run_num": "good", "cooler_v": 29895.3,
                 "laser_set": 15975.4150, "mass_amu": 47.95,
                 "harmonic": 2},
                # Pathological entry from a missing-MassAMU ASDF:
                {"run_num": "missing", "cooler_v": 0,
                 "laser_set": 0, "mass_amu": 0,
                 "harmonic": 2},
            ],
        }
        _, _, _, warnings = project_voltage_merge_to_frequency(
            merged_data, {"ref_freq": 0.0})
        codes = [w["code"] for w in warnings]
        # The zero entry was filtered out; only the good one remains,
        # so n_coolers < 2 → no spread check fires.
        self.assertNotIn("MERGE_COOLER_SPREAD", codes)
        self.assertNotIn("MERGE_LASER_SPREAD", codes)


class AutoFitProjectionTests(unittest.TestCase):
    """``_load_data_for_autofit`` short-circuits when ``merged_data``
    is passed. Phase 4: that short-circuit must apply the V→F
    projection when the source bin_mode is Frequency."""

    def setUp(self):
        _ensure_app()

    def test_voltage_merge_with_frequency_bin_mode_gets_projected(self):
        from gui.analysis.project import AnalysisProject

        merged_data = {
            "merged_name": "merged_v",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [],
        }
        source_config = {
            "bin_mode": "Frequency",
            "ref_freq": 0.0,
            "cal_order": 1,
        }
        proj = AnalysisProject()
        x, y, yerr = proj._load_data_for_autofit(
            run_file=None,
            source_config=source_config,
            merged_data=merged_data)
        # Doppler-projected x is in MHz, not V.
        self.assertEqual(len(x), 3)
        self.assertNotAlmostEqual(x[0], -50.0, places=1)
        np.testing.assert_array_equal(y, merged_data["y"])

    def test_voltage_merge_with_voltage_bin_mode_passes_through(self):
        from gui.analysis.project import AnalysisProject

        merged_data = {
            "merged_name": "merged_v",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "merge_metadata": {},
            "per_run": [],
        }
        source_config = {
            "bin_mode": "Raw Voltage",
            "ref_freq": 0.0,
            "cal_order": 1,
        }
        proj = AnalysisProject()
        x, y, yerr = proj._load_data_for_autofit(
            run_file=None,
            source_config=source_config,
            merged_data=merged_data)
        # Untouched.
        np.testing.assert_array_equal(x, merged_data["x"])
        np.testing.assert_array_equal(y, merged_data["y"])

    def test_frequency_merge_passes_through(self):
        from gui.analysis.project import AnalysisProject

        merged_data = {
            "merged_name": "merged_f",
            "x": np.array([-100.0, 0.0, 100.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "MHz",
            "merge_metadata": {},
            "per_run": [],
        }
        source_config = {
            "bin_mode": "Frequency",
            "ref_freq": 0.0,
            "cal_order": 1,
        }
        proj = AnalysisProject()
        x, y, _yerr = proj._load_data_for_autofit(
            run_file=None,
            source_config=source_config,
            merged_data=merged_data)
        # No projection -- frequency merge is already in MHz.
        np.testing.assert_array_equal(x, merged_data["x"])
        np.testing.assert_array_equal(y, merged_data["y"])


class MergeViewDialogPaImportedTests(unittest.TestCase):
    """The MergeViewDialog must not crash when per_run lacks x/y --
    PA-imported merges have per-source physics metadata but no
    per-source binned spectra (PA's merge collapses them at compute
    time)."""

    def setUp(self):
        _ensure_app()

    def test_pa_imported_per_run_without_x_y_does_not_crash_view(self):
        from gui.analysis.merge import MergeViewDialog

        # Shape that _add_merged_from_preanalysis produces: per_run
        # has cooler_v/laser_set/etc., but no x or y arrays.
        merged_data = {
            "merged_name": "merged_X",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "source_runs": ["7735", "7740"],
            "source_files": ["/data/run_7735.asdf",
                              "/data/run_7740.asdf"],
            "per_run": [
                {"run_num": "7735", "path": "/data/run_7735.asdf",
                 "cooler_v": 29895.3, "laser_set": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
                {"run_num": "7740", "path": "/data/run_7740.asdf",
                 "cooler_v": 29895.5, "laser_set": 15975.4150,
                 "mass_amu": 47.95, "harmonic": 2},
            ],
            "bin_step_mhz": 0,
            "metadata": {"imported_from": "preanalysis",
                          "merge_domain": "voltage"},
        }
        # Should NOT raise KeyError 'x'.
        dlg = MergeViewDialog(merged_data, parent=None)
        # And it should have built three tabs.
        from PySide6.QtWidgets import QTabWidget
        tabs = dlg.findChild(QTabWidget)
        self.assertIsNotNone(tabs)
        self.assertGreaterEqual(tabs.count(), 1)

    def _find_spectrum_axes(self, dlg):
        """Find the Spectrum-tab axes (xlabel contains "Frequency"
        or "Voltage", not "Time of Flight")."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        for c in dlg.findChildren(FigureCanvasQTAgg):
            ax = c.figure.axes[0]
            xl = ax.get_xlabel()
            if "Frequency" in xl or "Voltage" in xl:
                return ax
        return None

    def test_preview_projects_voltage_merge_when_source_is_frequency(self):
        """Phase-4 follow-up: the Source-block preview view must
        Doppler-project a voltage merge when bin_mode=Frequency, so
        the preview matches what the fit pipeline produces."""
        from gui.analysis.merge import MergeViewDialog

        merged_data = {
            "merged_name": "merged_V",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [],
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "bin_step_mhz": 0,
        }
        source_config = {"bin_mode": "Frequency", "ref_freq": 0.0}

        dlg = MergeViewDialog(merged_data, parent=None,
                               source_config=source_config)
        ax = self._find_spectrum_axes(dlg)
        self.assertIsNotNone(ax)
        self.assertIn("Frequency", ax.get_xlabel())
        self.assertIn("V→F projected", ax.get_title())

    def test_preview_keeps_voltage_when_source_is_raw_voltage(self):
        from gui.analysis.merge import MergeViewDialog

        merged_data = {
            "merged_name": "merged_V",
            "x": np.array([-50.0, 0.0, 50.0]),
            "y": np.array([1.0, 5.0, 2.0]),
            "yerr": np.array([1.0, 2.2, 1.4]),
            "x_unit": "V",
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
            "per_run": [],
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "bin_step_mhz": 0,
        }
        source_config = {"bin_mode": "Raw Voltage", "ref_freq": 0.0}

        dlg = MergeViewDialog(merged_data, parent=None,
                               source_config=source_config)
        ax = self._find_spectrum_axes(dlg)
        self.assertIsNotNone(ax)
        self.assertIn("Voltage", ax.get_xlabel())
        self.assertNotIn("projected", ax.get_title())


if __name__ == "__main__":
    unittest.main()
