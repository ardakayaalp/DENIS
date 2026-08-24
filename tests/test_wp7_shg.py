"""Regression tests for the WP7 SHG-calculator fixes (2026-06-02 review).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Two fixes:

1. KDP used the Zernike 5-term dispersion shoehorned into a 4-term form,
   dropping the dominant IR-pole term and mis-predicting the Type-I cut angle
   by ~8 deg. With the correct Zernike form, KDP 1064->532 nm must phase-match
   near the literature value 41.2 deg.
2. LBO is biaxial and cannot be handled by the uniaxial Type-I model; it is
   now flagged unsupported (so the UI says so) instead of silently returning
   "No phase matching" for every input.

The BBO control (correct before and after) anchors the solver: ~22.8 deg.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_wp7_shg -v

Depends on: gui.shared_widgets (_shg_angle_type1, _shg_wavelength_from_angle_type1,
_crystal_index, _crystal_supported, _SELLMEIER).
"""

import unittest

from gui.shared_widgets import (
    _shg_angle_type1,
    _shg_wavelength_from_angle_type1,
    _crystal_index,
    _crystal_supported,
)


class KDPDispersionTests(unittest.TestCase):
    """KDP must reproduce literature indices and the 41.2 deg cut angle."""

    def test_kdp_indices_at_1064(self):
        # Zernike KDP: no(1.064 um) ~ 1.4938, ne(1.064 um) ~ 1.4600.
        self.assertAlmostEqual(_crystal_index("KDP", "no", 1.064), 1.4938,
                               places=3)
        self.assertAlmostEqual(_crystal_index("KDP", "ne", 1.064), 1.4600,
                               places=3)

    def test_kdp_type1_1064_to_532_angle(self):
        angle = _shg_angle_type1("KDP", 1064.0)
        self.assertIsNotNone(angle)
        # Literature KDP Type-I 1064->532 phase-matching angle ~ 41.2 deg.
        self.assertAlmostEqual(angle, 41.2, delta=0.4)

    def test_kdp_inverse_is_self_consistent(self):
        # KDP dispersion is weak, so a given angle is satisfied across a broad
        # wavelength band (the inverse is ambiguous). Assert the inverse finds
        # a wavelength whose forward angle matches, rather than a single nm.
        a = _shg_angle_type1("KDP", 1064.0)
        lam = _shg_wavelength_from_angle_type1("KDP", a)
        self.assertIsNotNone(lam)
        self.assertAlmostEqual(_shg_angle_type1("KDP", lam), a, delta=0.05)


class BBOControlTests(unittest.TestCase):
    """BBO solver was correct; confirm it stayed correct after the refactor."""

    def test_bbo_type1_1064_angle(self):
        angle = _shg_angle_type1("BBO", 1064.0)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 22.8, delta=0.3)

    def test_bbo_angle_to_wavelength_roundtrip(self):
        # BBO dispersion is strong enough that the inverse is well-conditioned.
        a = _shg_angle_type1("BBO", 1064.0)
        lam = _shg_wavelength_from_angle_type1("BBO", a)
        self.assertIsNotNone(lam)
        self.assertAlmostEqual(lam, 1064.0, delta=10.0)


class LBOUnsupportedTests(unittest.TestCase):
    """LBO is biaxial — must be reported unsupported, not silently empty."""

    def test_lbo_marked_unsupported(self):
        self.assertFalse(_crystal_supported("LBO"))
        self.assertTrue(_crystal_supported("BBO"))
        self.assertTrue(_crystal_supported("KDP"))

    def test_lbo_angle_returns_none(self):
        self.assertIsNone(_shg_angle_type1("LBO", 1064.0))
        self.assertIsNone(_shg_wavelength_from_angle_type1("LBO", 11.0))


if __name__ == "__main__":
    unittest.main()
