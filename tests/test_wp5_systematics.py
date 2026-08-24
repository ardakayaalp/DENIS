"""Tests for WP5 systematic-uncertainty fixes (2026-06-02 code review).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

cls_estimations.isotope_shift.cooler_voltage_systematic:

* #33 sys-laser-missing-harmonic: the laser-setpoint term must scale with the
  laser harmonic (the setpoint is the fundamental wavenumber). With the cooler
  term isolated to zero, the systematic scales linearly with harmonic.
* #34 sys-voltage-silently-zero-when-no-reffreq: with no laser/reference
  frequency the cooler-voltage sensitivity collapses to 0 and the term is
  omitted -- this must be surfaced in the details log, not silent.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_wp5_systematics -v

Depends on: cls_estimations.isotope_shift.cooler_voltage_systematic
(and cls_estimations.doppler).
"""

import unittest

from cls_estimations.isotope_shift import cooler_voltage_systematic

# Two runs: identical cooler voltage (so the cooler term's SEM is 0 and the
# total is the laser term alone), differing laser setpoint (so its SEM > 0).
_MD = {"P": [{"cooler_v": 30000.0, "laser_set": 12000.00},
             {"cooler_v": 30000.0, "laser_set": 12000.01}]}
_MASS = {"P": 40.0}
_NU = 5.0e8  # MHz (~500 THz)


class HarmonicScalingTests(unittest.TestCase):
    def test_laser_systematic_scales_with_harmonic(self):
        r1 = cooler_voltage_systematic(_MD, _MASS, 1, "anti-collinear",
                                       _NU, harmonic=1)
        r2 = cooler_voltage_systematic(_MD, _MASS, 1, "anti-collinear",
                                       _NU, harmonic=2)
        s1 = r1["systematic_errors"]["P"]
        s2 = r2["systematic_errors"]["P"]
        self.assertGreater(s1, 0.0)
        # Cooler SEM is 0 here, so the total is the laser term, which scales
        # linearly with harmonic.
        self.assertAlmostEqual(s2 / s1, 2.0, delta=0.02)

    def test_default_harmonic_is_one(self):
        r_default = cooler_voltage_systematic(_MD, _MASS, 1, "anti-collinear",
                                              _NU)
        r_h1 = cooler_voltage_systematic(_MD, _MASS, 1, "anti-collinear",
                                         _NU, harmonic=1)
        self.assertAlmostEqual(r_default["systematic_errors"]["P"],
                               r_h1["systematic_errors"]["P"])


class ZeroFrequencyWarningTests(unittest.TestCase):
    def test_zero_laser_frequency_surfaces_omission(self):
        r = cooler_voltage_systematic(_MD, _MASS, 1, "anti-collinear", 0.0)
        details = r["details"]
        self.assertIn("WARNING", details)
        self.assertIn("OMITTED", details)

    def test_nonzero_frequency_has_no_omission_warning(self):
        r = cooler_voltage_systematic(_MD, _MASS, 1, "anti-collinear", _NU)
        self.assertNotIn("OMITTED", r["details"])


class ShiftCsvParityTests(unittest.TestCase):
    """#17 is-csv-column-mismatch: the canonical isotope_shifts.csv and the
    manual export must share the full Phase-5 column set (one helper)."""

    def test_helper_emits_full_phase5_columns(self):
        from gui.analysis.isotope_shift_tab import IsotopeShiftTab

        class _Stub:
            _last_shift_data = [{
                "label": "Ag-107", "A": 107, "centroid": 1.0,
                "sigma_fit": 0.1, "sigma_scatter": 0.2,
                "sigma_correction": 0.05, "sigma_voltage": 0.3,
                "sigma_stat": 0.1, "sigma_sys": 0.3, "sigma_total": 0.4,
                "delta_nu": 5.0, "sigma_delta_nu_stat": 0.1,
                "sigma_delta_nu_scatter": 0.2,
                "sigma_delta_nu_correction": 0.05,
                "sigma_delta_nu_sys": 0.3, "sigma_delta_nu": 0.4,
                "cov_with_reference_mhz2": 0.01, "is_reference": False,
            }]

        header, rows = IsotopeShiftTab._shift_csv_header_and_rows(_Stub())
        # The Phase-5 budget columns that the legacy auto-saved CSV dropped:
        for col in ("Sigma_scatter", "Sigma_correction",
                    "Sigma_dnu_scatter", "Sigma_dnu_correction",
                    "Cov_with_reference_MHz2"):
            self.assertIn(col, header)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(header))  # every column populated


if __name__ == "__main__":
    unittest.main()
