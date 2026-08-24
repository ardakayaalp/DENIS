"""End-to-end: a calibration fix moves the real spectrum by the promised MHz.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Every other test in this feature checks a part. This one checks the
promise: the diagnostic dialog tells the user "correcting this run's
calibration moves your centroid by N MHz", and the number has to be the
one the *fitter* actually sees, not a plausible-looking figure computed
down a parallel path. So this builds a complete synthetic run -- a real
ASDF with events and a resonance, whose calibration sweep has the first
three points wrong because the HV supply was still settling -- pushes it
through the same ``prepare_run_data`` -> ``compute_binned`` chain the fit
worker uses, and compares the centroid that comes out against what
``shift_mhz_over_scan`` predicted.

It also pins the thing that makes this a systematic rather than a
nuisance: the error is *not* a rigid translation. An offset error and a
gain error cancel somewhere mid-sweep and add at the edges, so a bad
calibration tilts the frequency axis, moving peaks by different amounts
depending on where in the scan they sit.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_calibration_e2e -v

Depends on: gui.calibration, gui.analysis.pipeline, gui.analysis.binning,
clstools (the real one -- this test deliberately uses no fake), asdf.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from gui.calibration import (
    CalibrationRegistry,
    clear_points_cache,
    fit_calibration,
    read_cal_points,
    set_registry,
    shift_mhz_over_scan,
)

MASS = 51.0           # 51-V
LASER = 15975.02      # cm^-1, from configs/V_analysis.yaml
HARMONIC = 2
COOLER_KV = 3.0       # Vrfq; * VCoolDiv(10000) = 30 kV
PEAK_V = 200.0        # the resonance sits at DV = 200 V
VACC = 1000.0

SOURCE_CFG = dict(
    mass=MASS, ref_freq=0.0, harmonic=HARMONIC, bin_mode="Frequency",
    cooler_correction="pbp", cal_order=1, x_column="Fmean",
    yerr_mode="sqrt(N)", xerr_mode="None", bin_definition="Auto",
    bin_count=0, bin_width_mhz=0.0, pmt_gate=[3, 4], tof_gate=None,
    v_gate=None, f_gate=None, noise_filter=0, ref_shift=0.0,
)


class CalibrationEndToEndTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import asdf
        cls.tmp = tempfile.mkdtemp(prefix="cal_e2e_")
        rng = np.random.default_rng(11)

        # Calibration sweep, with the first three readbacks wrong.
        cls.cal_set = np.linspace(0.0, 400.0, 21)
        cal_rb_v = (1.0007 * cls.cal_set + 0.35
                    + rng.normal(0.0, 0.03, cls.cal_set.size))
        cal_rb_v[:3] += [-4.0, -2.5, -1.4]        # HV still settling
        cls.cal_read = cal_rb_v / VACC            # ASDF stores monitor units

        # Events: a Gaussian resonance at DV = PEAK_V on a flat background.
        steps = np.linspace(0.0, 400.0, 81)
        dv, tof, tdc, ts, bunch = [], [], [], [], []
        for i, s in enumerate(steps):
            rate = 20 + 260 * np.exp(-0.5 * ((s - PEAK_V) / 12.0) ** 2)
            k = int(rng.poisson(rate))
            dv += [s] * k
            tof += list(rng.normal(50.0, 3.0, k))
            tdc += list(rng.choice([3, 4], k))
            ts += list(np.full(k, i * 0.1))
            bunch += [i] * k
        raw = np.column_stack([ts, dv, bunch, tdc, tof,
                               np.full(len(dv), COOLER_KV)])

        cls.path = os.path.join(cls.tmp, "run_9001.asdf")
        asdf.AsdfFile({
            "Run": 9001, "CoolerVoltage": COOLER_KV, "LaserSetpoint": LASER,
            "DwellTime": 0.1, "Experiment": "V", "Date": "2026-07-14",
            "StepSize": 5.0, "ScanningRanges": [[0.0, 400.0]],
            "CalSet": cls.cal_set, "CalReadback": cls.cal_read, "raw": raw,
        }).write_to(cls.path)

        cls.FIX = {cls.path: {"mode": "fit", "reject": "first_n",
                              "drop_first": 3}}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        clear_points_cache()
        set_registry(CalibrationRegistry())

    def tearDown(self):
        set_registry(None)

    # ── helpers ──────────────────────────────────────────────

    def _centroid_mhz(self, calibrations):
        """Centroid of the spectrum the fit worker would actually be handed."""
        from gui.analysis.binning import compute_binned
        from gui.analysis.pipeline import prepare_run_data

        data, eff_cfg, meta = prepare_run_data(
            self.path, dict(SOURCE_CFG), calibrations=calibrations)
        res = compute_binned(data, eff_cfg)
        x = np.asarray(res["x"], dtype=float)      # MHz
        y = np.asarray(res["y"], dtype=float)
        w = np.clip(y - np.median(y), 0.0, None)   # background-subtracted
        return float((x * w).sum() / w.sum()), meta["calibration"]

    def _predicted_shift_at_peak(self):
        set_v, read_raw = read_cal_points(self.path)
        base = fit_calibration(set_v, read_raw, 1, (), VACC)
        fixed = fit_calibration(set_v, read_raw, 1, (0, 1, 2), VACC)
        dv, dnu = shift_mhz_over_scan(
            base.coeffs_v, fixed.coeffs_v, 0.0, 400.0,
            cooler_v=COOLER_KV * 10000.0, laser_cm=LASER,
            mass_amu=MASS, harmonic=HARMONIC)
        return float(np.interp(PEAK_V, dv, dnu)), dnu

    # ── the promise ──────────────────────────────────────────

    def test_the_predicted_mhz_is_the_shift_the_fitter_actually_sees(self):
        """The whole feature, end to end.

        If these two ever diverge, the dialog is lying to the user about the
        size of their systematic -- which is worse than showing nothing.
        """
        c_uncorrected, _ = self._centroid_mhz({})
        c_corrected, _ = self._centroid_mhz(self.FIX)
        measured = c_uncorrected - c_corrected

        predicted, _dnu = self._predicted_shift_at_peak()

        self.assertGreater(abs(measured), 5.0)      # a real systematic
        # Agreement to well under a tenth of a MHz; the remainder is centroid
        # estimator noise from Poisson counts and finite bins, not a bias.
        self.assertAlmostEqual(measured, predicted, delta=0.3)

    def test_the_bad_calibration_tilts_the_axis_rather_than_shifting_it(self):
        """Why this is a systematic and not a nuisance: the offset and gain
        errors cancel mid-sweep and add at the edges, so peaks at different
        scan positions move by different amounts. A single global offset
        correction could not fix this."""
        _predicted, dnu = self._predicted_shift_at_peak()
        tilt = float(dnu.max() - dnu.min())
        self.assertGreater(tilt, 10.0)
        # And the tilt is bigger than the shift at the peak, so "just recentre
        # it" is not an available escape.
        self.assertGreater(tilt, abs(float(np.interp(
            PEAK_V, np.linspace(0, 400, len(dnu)), dnu))))

    def test_the_override_is_recorded_in_the_run_metadata(self):
        _c, meta = self._centroid_mhz(self.FIX)
        self.assertEqual(meta["mode"], "fit")
        self.assertEqual(meta["excluded"], [0, 1, 2])
        self.assertEqual(meta["n_points"], 21)

    def test_with_no_override_the_metadata_says_so_explicitly(self):
        _c, meta = self._centroid_mhz({})
        self.assertEqual(meta["mode"], "file")
        self.assertEqual(meta["n_excluded"], 0)

    def test_borrowing_a_good_run_recovers_the_same_centroid(self):
        """1002 is broken, 1001 is fine, and they share a supply: borrowing
        1001's calibration must land the centroid where fixing 1002's own
        points does."""
        import asdf
        rng = np.random.default_rng(4)
        d_set = np.linspace(0.0, 400.0, 21)
        d_read = (1.0007 * d_set + 0.35
                  + rng.normal(0.0, 0.03, d_set.size)) / VACC
        donor = os.path.join(self.tmp, "run_9000.asdf")
        asdf.AsdfFile({"CalSet": d_set,
                       "CalReadback": d_read}).write_to(donor)

        c_fixed, _ = self._centroid_mhz(self.FIX)
        c_borrowed, meta = self._centroid_mhz(
            {self.path: {"mode": "borrow", "donor": donor}})

        self.assertEqual(meta["mode"], "borrow")
        # Both land on the true axis; they differ only by the two runs'
        # independent calibration noise.
        self.assertAlmostEqual(c_borrowed, c_fixed, delta=1.0)

    def test_a_clean_run_is_unaffected_by_routing_through_the_calibration(self):
        """With no override the app must reproduce the *unfiltered* clstools
        fit exactly -- the guarantee that made it safe to reroute every load
        path."""
        from gui.analysis.pipeline import prepare_run_data

        data, _eff, _meta = prepare_run_data(self.path, dict(SOURCE_CFG))
        values = np.polyfit(self.cal_set, self.cal_read, 1)
        expected = list(values)[::-1]          # clstools reverses to ascending
        np.testing.assert_allclose(data.Cal, expected, rtol=1e-12)

    def test_raw_voltage_display_axis_uses_the_calibration_polynomial(self):
        """Pre-Analysis' Raw Voltage mode used to build its "Calibrated
        voltage" axis by snapping each DAC step to the nearest *raw*
        CalReadback sample. That read straight off the settling glitch and
        ignored an override entirely -- so excluding bad points visibly moved
        the Frequency-mode spectrum while leaving this axis exactly where it
        was. It now applies the polynomial, which is what Compute_Voltages
        applies to every event.
        """
        from gui.calibration import (
            apply_calibration, fit_calibration, resolve_calibration)

        dv = np.linspace(0.0, 400.0, 9)

        base = resolve_calibration(self.path, {})
        fixed = resolve_calibration(self.path, self.FIX)

        x_base = base.predict_v(dv)
        x_fixed = fixed.predict_v(dv)
        moved = x_fixed - x_base

        # The axis must respond to the override at all...
        self.assertGreater(np.abs(moved).max(), 0.5)
        # ...and it must *rotate*, not translate: an offset error and a gain
        # error cancel mid-sweep and add at the edges, so the correction is
        # larger at one end of the scan than the other and changes sign.
        self.assertGreater(moved[0], 0.0)
        self.assertLess(moved[-1], 0.0)

        # And it is exactly what clstools computes for the events themselves.
        cal_raw = np.asarray(fixed.coeffs_raw(VACC), dtype=float)
        clstools_dv_cal = (dv * cal_raw[1] + cal_raw[0]) * VACC
        np.testing.assert_allclose(x_fixed, clstools_dv_cal, rtol=1e-12)

    def test_a_virtual_split_is_gated_and_calibrated_like_the_fit(self):
        """The Analysis binning-diagnostics loader used to hand clstools the
        ``.vasdf`` sidecar (which it cannot open) and never intersected the
        split's V-gate. Routing it through ``prepare_run_data`` fixes both --
        so a split's diagnostics now show *the split*, not the parent run's
        whole spectrum, and with the parent's calibration applied.
        """
        from gui.analysis.binning import compute_binned
        from gui.analysis.pipeline import prepare_run_data
        from gui.analysis.vasdf import write_vasdf

        vpath = os.path.join(self.tmp, "9001_A.vasdf")
        write_vasdf(vpath, parent_path=self.path, source_id="9001_A",
                    label="9001_A", lo=0.0, hi=PEAK_V)
        desc = {"parent_path": self.path, "source_id": "9001_A",
                "split_lo": 0.0, "split_hi": PEAK_V,
                "metadata_override": {}}

        cfg = dict(SOURCE_CFG, bin_mode="Raw Voltage")
        full, eff_f, _m = prepare_run_data(self.path, dict(cfg))
        x_full = np.asarray(compute_binned(full, eff_f)["x"], dtype=float)

        # The .vasdf path is what the UI holds; the events come from the parent.
        split, eff_s, meta = prepare_run_data(
            vpath, dict(cfg), split_descriptor=desc,
            calibrations=self.FIX)
        x_split = np.asarray(compute_binned(split, eff_s)["x"], dtype=float)

        self.assertEqual(list(eff_s["v_gate"]), [0.0, PEAK_V])
        self.assertLessEqual(x_split.max(), PEAK_V + 5.0)
        self.assertLess(len(x_split), len(x_full))
        # ...and the parent's calibration override reached it.
        self.assertEqual(meta["calibration"]["excluded"], [0, 1, 2])

    def test_the_documented_escape_hatch_reproduces_the_old_filter_exactly(self):
        """The one back-compat promise the README makes.

        Switching clstools' own ``filter_calibration`` off *did* change the
        default on any machine running the newer, filtering build: points it
        used to drop are now kept. The README tells anyone who needs the old
        numbers back to set ``Reject: n·σ, σ = 2, non-iterative`` -- so that
        had better be bit-for-bit the same polynomial, against the real library
        rather than a reimplementation of it.
        """
        import clstools

        from gui.calibration import load_run_calibrated

        old = clstools.CLSDataFrame()
        old.Load_Run(self.path)          # filter_calibration defaults to True
        if not getattr(old, "Dropped_calibration_points", None):
            self.skipTest("installed clstools does not filter calibration "
                          "points; nothing to be bit-compatible with")

        new = clstools.CLSDataFrame()
        load_run_calibrated(new, self.path, {
            self.path: {"mode": "fit", "reject": "sigma", "sigma": 2.0,
                        "iterative": False}})

        np.testing.assert_allclose(new.Cal, old.Cal, rtol=1e-12, atol=0.0)
        self.assertEqual(new.Dropped_calibration_points,
                         old.Dropped_calibration_points)


if __name__ == "__main__":
    unittest.main(verbosity=2)
