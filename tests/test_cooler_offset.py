"""A cooler-voltage offset, added on top of each run's own value.

The Source block could only ever REPLACE the cooler voltage for every
file. The Yb calibration needs the other thing: a constant added to
whatever each file recorded, so runs taken at different cooler
voltages keep their differences while all moving together. The
calibration scan then walks that one number across a grid.

Built on a real synthetic ASDF driven through the real pipeline --
the point is what clstools does with the number, which a mock could
not show.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_cooler_offset.py -q
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from gui.analysis.binning import compute_binned
from gui.analysis.pipeline import prepare_run_data

MASS = 51.0
LASER = 15975.02
HARMONIC = 2
COOLER_KV = 3.0          # * VCoolDiv (10000) = 30 kV
VCOOL_DIV = 10000.0

SOURCE_CFG = dict(
    mass=MASS, ref_freq=0.0, harmonic=HARMONIC, bin_mode="Frequency",
    cooler_correction="pbp", cal_order=1, x_column="Fmean",
    yerr_mode="sqrt(N)", xerr_mode="None", bin_definition="Auto",
    bin_count=0, bin_width_mhz=0.0, pmt_gate=[3, 4], tof_gate=None,
    v_gate=None, f_gate=None, noise_filter=0, ref_shift=0.0,
)


class CoolerOffsetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import asdf
        cls.tmp = tempfile.mkdtemp(prefix="cooler_offset_")
        rng = np.random.default_rng(5)
        steps = np.linspace(0.0, 400.0, 81)
        dv, tof, tdc, ts, bunch = [], [], [], [], []
        for i, s in enumerate(steps):
            rate = 20 + 260 * np.exp(-0.5 * ((s - 200.0) / 12.0) ** 2)
            k = int(rng.poisson(rate))
            dv += [s] * k
            tof += list(rng.normal(50.0, 3.0, k))
            tdc += list(rng.choice([3, 4], k))
            ts += list(np.full(k, i * 0.1))
            bunch += [i] * k
        raw = np.column_stack([ts, dv, bunch, tdc, tof,
                               np.full(len(dv), COOLER_KV)])
        # An identity scan-voltage calibration: this test is about the
        # cooler voltage, so the divider chain must add nothing of its
        # own. (Readbacks are stored in monitor units, set/1000.)
        cal_set = np.linspace(0.0, 400.0, 21)
        cls.path = os.path.join(cls.tmp, "run_9100.asdf")
        asdf.AsdfFile({
            "Run": 9100, "CoolerVoltage": COOLER_KV,
            "LaserSetpoint": LASER, "DwellTime": 0.1, "Experiment": "Yb",
            "Date": "2026-09-24", "StepSize": 5.0,
            "ScanningRanges": [[0.0, 400.0]],
            "CalSet": cal_set, "CalReadback": cal_set / 1000.0,
            "raw": raw,
        }).write_to(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, **extra):
        cfg = dict(SOURCE_CFG)
        cfg.update(extra)
        return prepare_run_data(self.path, cfg)

    def _spectrum(self, **extra):
        """The binned frequency axis the fitter would see."""
        data, eff, _meta = self._run(**extra)
        return np.asarray(compute_binned(data, eff)["x"], dtype=float)

    # ── the recorded value ──

    def test_no_offset_leaves_the_run_as_recorded(self):
        _d, _cfg, meta = self._run()
        self.assertEqual(meta["cooler_offset_v"], 0.0)
        self.assertAlmostEqual(meta["cooler_v"], COOLER_KV * VCOOL_DIV,
                               places=6)
        self.assertAlmostEqual(meta["cooler_v_effective"],
                               meta["cooler_v"], places=6)

    def test_the_offset_is_added_to_the_run_s_own_voltage(self):
        _d, _cfg, meta = self._run(cooler_offset_v=-29.27)
        self.assertAlmostEqual(meta["cooler_v"], 30000.0, places=6)
        self.assertAlmostEqual(meta["cooler_v_effective"],
                               30000.0 - 29.27, places=6)
        self.assertEqual(meta["cooler_offset_v"], -29.27)

    def test_the_run_s_own_value_is_still_recorded_unchanged(self):
        """The audit trail has to say what the FILE held, not what the
        analysis assumed -- that is how a wrong offset is ever
        noticed."""
        _d, _cfg, meta = self._run(cooler_offset_v=+50.0)
        self.assertAlmostEqual(meta["cooler_v"], 30000.0, places=6)

    def test_it_stacks_on_an_absolute_override(self):
        """Override says 'use 29 000 V for every file'; the offset
        then moves that."""
        _d, _cfg, meta = self._run(override_enabled=True,
                                   cooler_override=29000.0,
                                   cooler_offset_v=-30.0)
        self.assertAlmostEqual(meta["cooler_v_effective"], 28970.0,
                               places=6)

    def test_an_offset_that_cancels_the_beam_is_refused(self):
        """Better a message naming the number than a silent fit of a
        zero-energy beam."""
        with self.assertRaises(ValueError) as ctx:
            self._run(cooler_offset_v=-40000.0)
        self.assertIn("beam energy", str(ctx.exception))

    # ── what it does to the physics ──

    def test_the_frequency_axis_moves_with_the_offset(self):
        """The whole point: the cooler voltage sets the beam energy,
        so the Doppler-corrected frequency axis moves with it. That
        is what lets a hyperfine A constant measure the offset."""
        x0 = self._spectrum()
        x1 = self._spectrum(cooler_offset_v=-100.0)
        self.assertEqual(x0.size, x1.size)
        self.assertGreater(abs(float(np.mean(x1 - x0))), 1.0)   # MHz

    def test_the_frequency_SPAN_stretches_too(self):
        """Not just a shift: the scale changes, which is why a
        frequency DIFFERENCE inside one spectrum (an A constant) can
        calibrate an absolute voltage."""
        d0, _c, _m = self._run()
        d1, _c1, _m1 = self._run(cooler_offset_v=-100.0)
        self.assertNotAlmostEqual(float(d0.Frequency_stepsize),
                                  float(d1.Frequency_stepsize),
                                  delta=1.0)

    def test_a_larger_offset_moves_it_further(self):
        base = self._spectrum()
        half = self._spectrum(cooler_offset_v=-50.0)
        full = self._spectrum(cooler_offset_v=-100.0)
        self.assertGreater(abs(float(np.mean(full - base))),
                           abs(float(np.mean(half - base))))

    def test_zero_offset_is_bit_for_bit_the_old_behaviour(self):
        """The offset is new; every existing analysis must be
        untouched by its presence."""
        np.testing.assert_array_equal(self._spectrum(),
                                      self._spectrum(cooler_offset_v=0.0))


class SourceBlockFieldTests(unittest.TestCase):
    """The offset is a Source-block setting: it has to survive a
    save/load, or an analysis would silently revert to uncorrected."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _block(self):
        from gui.analysis.blocks import SourceBlock
        b = SourceBlock()
        self.addCleanup(b.deleteLater)
        return b

    def test_it_defaults_to_zero(self):
        self.assertEqual(
            self._block().get_source_config()["cooler_offset_v"], 0.0)

    def test_it_reaches_the_source_config(self):
        b = self._block()
        b._cooler_offset.setValue(-29.27)
        self.assertAlmostEqual(
            b.get_source_config()["cooler_offset_v"], -29.27)

    def test_it_round_trips_through_a_save(self):
        a = self._block()
        a._cooler_offset.setValue(-29.27)
        b = self._block()
        b.from_dict(a.to_dict())
        self.assertAlmostEqual(b._cooler_offset.value(), -29.27)

    def test_an_older_save_loads_as_zero(self):
        b = self._block()
        b._cooler_offset.setValue(-12.0)
        d = b.to_dict()
        d.pop("cooler_offset_v")
        b.from_dict(d)
        self.assertEqual(b._cooler_offset.value(), 0.0)

    def test_it_is_not_the_override(self):
        """Separate controls: the override REPLACES the voltage for
        every file, the offset MOVES each file's own."""
        b = self._block()
        b._cooler_offset.setValue(-29.27)
        cfg = b.get_source_config()
        self.assertFalse(cfg["override_enabled"])
        self.assertEqual(cfg["cooler_override"], 0.0)


class OffsetTagTests(unittest.TestCase):
    """The tag that marks a Results iteration as belonging to one
    assumed cooler offset. A systematics scan writes one iteration per
    offset into the same project, so the offset has to be legible from
    the folder name alone -- iter_003 tells you nothing about which
    scan point it is."""

    def tag(self, v):
        from cls_estimations.cooler_calibration import offset_tag
        return offset_tag(v)

    def test_no_offset_leaves_the_name_alone(self):
        for v in (0, 0.0, None, "", "0"):
            self.assertEqual(self.tag(v), "", repr(v))

    def test_a_negative_offset(self):
        self.assertEqual(self.tag(-30.0), "_-30V_CO")

    def test_a_positive_offset_keeps_its_sign(self):
        """Without the sign, +30 and -30 would produce the same folder
        and the second scan point would look like a duplicate."""
        self.assertEqual(self.tag(12.5), "_+12.5V_CO")

    def test_trailing_zeros_are_dropped(self):
        self.assertEqual(self.tag(-30.000), "_-30V_CO")

    def test_a_fractional_offset_survives(self):
        """The calibration lands on -29.25, not a round number."""
        self.assertEqual(self.tag(-29.25), "_-29.25V_CO")

    def test_a_numeric_string_is_accepted(self):
        self.assertEqual(self.tag("-30"), "_-30V_CO")

    def test_junk_is_not_fatal(self):
        """An old save with a bad value must not stop results being
        written."""
        self.assertEqual(self.tag("none"), "")
        self.assertEqual(self.tag(object()), "")

    def test_it_is_a_legal_folder_name(self):
        bad = set(chr(c) for c in range(32)) | set('<>:"/|?*' + chr(92))
        self.assertFalse(bad & set(self.tag(-29.25)))


class ResultsTagFromRunsTests(unittest.TestCase):
    """The tag is read from what the fits ACTUALLY used, not from the
    Source block: the widget can be edited between a fit and its save,
    which would mislabel the results."""

    def tag(self, results):
        from gui.analysis.project import cooler_offset_tag
        return cooler_offset_tag(results)

    def _res(self, offset):
        return {"success": True, "run_metadata": {"cooler_offset_v": offset}}

    def test_nothing_to_go_on(self):
        self.assertEqual(self.tag([]), "")
        self.assertEqual(self.tag(None), "")

    def test_runs_without_the_key(self):
        self.assertEqual(self.tag([{"success": True}, {}]), "")

    def test_an_uncorrected_run_set(self):
        self.assertEqual(self.tag([self._res(0.0)]), "")

    def test_it_finds_the_offset(self):
        self.assertEqual(self.tag([self._res(-30.0)]), "_-30V_CO")

    def test_it_looks_past_runs_that_do_not_carry_it(self):
        self.assertEqual(
            self.tag([{"success": True}, self._res(-30.0)]), "_-30V_CO")


class IterationLabelTests(unittest.TestCase):
    """The folder the Results tab shows."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        import gui.shared_widgets as sw
        real = sw.get_analysis_dir
        sw.get_analysis_dir = lambda: self.dir
        self.addCleanup(setattr, sw, "get_analysis_dir", real)

    def _project(self, name="70Ge"):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject(name)
        self.addCleanup(p.deleteLater)
        return p

    def _save(self, project, offset, **out):
        cfg = dict(iter_mode="Auto", report=False, params_csv=False,
                   metadata_csv=False, fit_plots=False, tof_plots=False)
        cfg.update(out)
        project._save_results(
            [{"success": True, "run_number": "7947", "run_file": "",
              "params_df": {}, "metadata_df": {}, "report": "r",
              "run_metadata": {"cooler_offset_v": offset}}], cfg)
        return sorted(os.listdir(os.path.join(self.dir, project._project_name)))

    def test_an_offset_fit_is_labelled(self):
        self.assertEqual(self._save(self._project(), -30.0),
                         ["iter_001_-30V_CO"])

    def test_an_ordinary_fit_is_not(self):
        """Every existing project must keep the names it already has."""
        self.assertEqual(self._save(self._project(), 0.0), ["iter_001"])

    def test_numbering_counts_past_a_tagged_folder(self):
        """A scan point must not overwrite the one before it."""
        p = self._project()
        self._save(p, -30.0)
        self.assertEqual(self._save(p, -25.0),
                         ["iter_001_-30V_CO", "iter_002_-25V_CO"])

    def test_a_manual_label_is_tagged_too(self):
        self.assertEqual(
            self._save(self._project(), -30.0, iter_mode="Manual",
                       iter_label="scan"),
            ["scan_-30V_CO"])

    def test_the_latest_iteration_is_still_the_newest(self):
        """load_results_from_disk picks sorted()[-1]; a suffix must not
        make iteration 1 look newer than iteration 2."""
        p = self._project()
        for dv in (-30.0, -25.0, 0.0):
            self._save(p, dv)
        iters = sorted(os.listdir(os.path.join(self.dir, "70Ge")))
        self.assertEqual(iters[-1], "iter_003")


class IterationAttributionTests(unittest.TestCase):
    """A fit has to be filed under the folder it was written to.

    The Results tab used to work it out by taking the last directory
    in sorted order. That is right for iter_NNN, where the zero-padded
    number dominates, and wrong for labels -- and a systematic scan
    writes sys_001_+12.5V_CO next to sys_001_-30V_CO, where '+' sorts
    before '-'. Every step would have been filed under whichever
    folder happened to sort last.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        import gui.shared_widgets as sw
        real = sw.get_analysis_dir
        sw.get_analysis_dir = lambda: self.dir
        self.addCleanup(setattr, sw, "get_analysis_dir", real)

    def _results_tab(self):
        from gui.results_tab import ResultsTab
        t = ResultsTab()
        self.addCleanup(t.deleteLater)
        return t

    def _make(self, *names):
        for n in names:
            os.makedirs(os.path.join(self.dir, "70Ge", n), exist_ok=True)

    def test_the_writer_names_the_iteration(self):
        self._make("sys_001_+12.5V_CO", "sys_001_-30V_CO")
        tab = self._results_tab()
        _p, name = tab.add_results(
            "70Ge", [], {"iter_name": "sys_001_+12.5V_CO"})
        self.assertEqual(name, "sys_001_+12.5V_CO")

    def test_the_sign_trap(self):
        """'+' sorts before '-', so the directory scan would have
        filed this under the -30 V folder."""
        self._make("sys_001_+12.5V_CO", "sys_001_-30V_CO")
        tab = self._results_tab()
        _p, guessed = tab.add_results("70Ge", [], {})
        _p, told = tab.add_results(
            "70Ge", [], {"iter_name": "sys_001_+12.5V_CO"})
        self.assertEqual(guessed, "sys_001_-30V_CO")
        self.assertEqual(told, "sys_001_+12.5V_CO")

    def test_a_caller_that_says_nothing_still_works(self):
        self._make("iter_001", "iter_002")
        tab = self._results_tab()
        _p, name = tab.add_results("70Ge", [], {})
        self.assertEqual(name, "iter_002")

    def test_the_directory_is_the_named_one(self):
        self._make("sys_001_+12.5V_CO", "sys_001_-30V_CO")
        tab = self._results_tab()
        tab.add_results("70Ge", [], {"iter_name": "sys_001_+12.5V_CO"})
        entry = tab._results_data["70Ge"]["sys_001_+12.5V_CO"]
        self.assertTrue(entry["directory"].endswith("sys_001_+12.5V_CO"))

    def test_a_saved_fit_stamps_its_own_folder(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("70Ge")
        self.addCleanup(p.deleteLater)
        out = dict(iter_mode="Manual", iter_label="sys_001",
                   report=False, params_csv=False, metadata_csv=False,
                   fit_plots=False, tof_plots=False)
        p._save_results(
            [{"success": True, "run_number": "7947", "run_file": "",
              "params_df": {}, "metadata_df": {}, "report": "r",
              "run_metadata": {"cooler_offset_v": 12.5}}], out)
        self.assertEqual(out["iter_name"], "sys_001_+12.5V_CO")


if __name__ == "__main__":
    unittest.main()
