"""Headless tests: the calibration reaches every load path, in the right order.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Two things can silently break this feature, and both are tested here.

**Ordering.** ``Compute_Voltages`` turns each event's DAC value into a
voltage from ``Cal`` / ``Cal_order`` / ``VAccDiv`` and nothing else. If a
calibration override is applied *after* it rather than before, every
number downstream is computed from the old polynomial and nothing
complains. So the fake CLSDataFrame here snapshots ``Cal`` at the moment
``Compute_Voltages`` is called, and the tests assert on that snapshot --
not on the final state, which would pass either way.

**The subprocess boundary.** Fits run in a worker that cannot reach the
registry singleton. If the calibration map isn't handed across
explicitly, ``get_registry()`` there returns an *empty* registry, every
run silently falls back to its file default, and the fit quietly ignores
the correction the user asked for. The map is therefore threaded through
``prepare_run_data`` -> ``_fit_single_run`` -> ``FitWorkerThread``, and
that wiring is asserted rather than assumed.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_calibration_pipeline -v

Depends on: gui.calibration, gui.analysis.pipeline, gui.analysis.fitting.
"""

import inspect
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

from gui.calibration import (
    CalibrationRegistry,
    clear_points_cache,
    fit_calibration,
    set_registry,
)

VACC = 1000.0


def make_table(n=21, gain=1.0007, offset=0.35, seed=5,
               bad=((0, -4.0), (1, -2.5), (2, -1.4))):
    rng = np.random.default_rng(seed)
    set_v = np.linspace(0.0, 400.0, n)
    read_v = gain * set_v + offset + rng.normal(0.0, 0.03, n)
    for idx, err in bad:
        read_v[idx] += err
    return set_v, read_v / VACC


class _FakeCLS:
    """Fake CLSDataFrame that records *when* things happened.

    ``Compute_Voltages`` snapshots ``Cal``, because "was the override in
    place by then?" is the only question that matters.
    """

    instances = []

    def __init__(self):
        self.calls = []
        self.VAccDiv = VACC
        self.VCoolDiv = 10000
        self.VCoolOffset = 0
        self.Vcool_init = 3.0
        self.Laser_set = 12000.0
        self.Size = 100
        self.DAQTStime = 1.0
        self.TSstart = 0.0
        self.TSstop = 5.0
        self.Date = "2026-07-14"
        self.Cal = None
        self.Cal_err = None
        self.Cal_order = None
        self.Cal_df = None
        self.cal_at_compute_voltages = None
        _FakeCLS.instances.append(self)

    def Load_Run(self, path, cal_order=1, **kw):
        self.calls.append("Load_Run")
        self.loaded_path = path
        # What clstools itself would have left behind. If our override never
        # lands, this is what Compute_Voltages would use -- so make it
        # obviously wrong, and any test that passes by accident will fail.
        self.Cal = [-999.0, -999.0]
        self.Cal_order = cal_order

    def Compute_Voltages(self, cooler_correction="pbp"):
        self.calls.append("Compute_Voltages")
        self.cal_at_compute_voltages = list(self.Cal)

    def Compute_WL(self, Mass=None, ref=None, harmonic=None):
        self.calls.append("Compute_WL")

    def Shift_Ref(self, ref=None):
        self.calls.append("Shift_Ref")

    def apply_filter(self, filter_window=0):
        self.calls.append("apply_filter")


def _install_fake_clstools():
    mod = types.ModuleType("clstools")
    mod.CLSDataFrame = _FakeCLS
    sys.modules["clstools"] = mod


CFG = {
    "mass": 51.0,
    "ref_freq": 5.0e14,
    "harmonic": 2,
    "bin_mode": "Frequency",
    "cooler_correction": "pbp",
    "cal_order": 1,
}


class PipelineOrderingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import asdf
        cls.tmp = tempfile.mkdtemp(prefix="cal_pipe_")
        cls.set_v, cls.read_raw = make_table()
        cls.path = os.path.join(cls.tmp, "run_1002.asdf")
        asdf.AsdfFile({"CalSet": cls.set_v,
                       "CalReadback": cls.read_raw}).write_to(cls.path)
        # A virtual split's events live in the parent ASDF.
        cls.split = {
            "parent_path": cls.path,
            "source_id": "1002_A",
            "split_lo": 0.0,
            "split_hi": 400.0,
            "metadata_override": {},
        }
        _install_fake_clstools()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        sys.modules.pop("clstools", None)

    def setUp(self):
        clear_points_cache()
        _FakeCLS.instances.clear()
        set_registry(CalibrationRegistry())

    def tearDown(self):
        set_registry(None)

    def _prepare(self, calibrations=None, **kw):
        from gui.analysis.pipeline import prepare_run_data
        return prepare_run_data(self.path, dict(CFG),
                                calibrations=calibrations, **kw)

    # ── ordering ─────────────────────────────────────────────

    def test_calibration_lands_between_load_and_compute_voltages(self):
        data, _cfg, _meta = self._prepare()
        self.assertEqual(data.calls[:2], ["Load_Run", "Compute_Voltages"])
        # Cal was replaced before Compute_Voltages read it: the -999 sentinel
        # clstools left behind never reached the voltage computation.
        self.assertNotIn(-999.0, data.cal_at_compute_voltages)

    def test_override_is_in_force_at_compute_voltages(self):
        """The whole feature in one assertion."""
        cal = {self.path: {"mode": "fit", "reject": "first_n",
                           "drop_first": 3}}
        data, _cfg, _meta = self._prepare(calibrations=cal)

        expected = fit_calibration(self.set_v, self.read_raw, 1, (0, 1, 2),
                                   VACC)
        expected_raw = [c / VACC for c in expected.coeffs_v]
        np.testing.assert_allclose(data.cal_at_compute_voltages, expected_raw,
                                   rtol=1e-12)

    def test_no_override_reproduces_the_plain_clstools_fit(self):
        """Routing every load through apply_calibration must, on its own,
        change nothing -- otherwise the refactor silently moved everyone's
        published centroids."""
        data, _cfg, _meta = self._prepare()
        values = np.polyfit(self.set_v, self.read_raw, 1)
        expected = list(values)[::-1]        # clstools reverses to ascending
        np.testing.assert_allclose(data.cal_at_compute_voltages, expected,
                                   rtol=1e-12)

    def test_borrowed_calibration_reaches_compute_voltages(self):
        import asdf
        donor = os.path.join(self.tmp, "run_1001.asdf")
        d_set, d_read = make_table(bad=(), seed=9)
        asdf.AsdfFile({"CalSet": d_set,
                       "CalReadback": d_read}).write_to(donor)

        cal = {self.path: {"mode": "borrow", "donor": donor}}
        data, _cfg, _meta = self._prepare(calibrations=cal)

        expected = fit_calibration(d_set, d_read, 1, (), VACC)
        np.testing.assert_allclose(
            data.cal_at_compute_voltages,
            [c / VACC for c in expected.coeffs_v], rtol=1e-12)

    def test_manual_coefficients_reach_compute_voltages(self):
        cal = {self.path: {"mode": "coeffs", "coeffs_v": [0.35, 1.0007]}}
        data, _cfg, _meta = self._prepare(calibrations=cal)
        np.testing.assert_allclose(data.cal_at_compute_voltages,
                                   [0.00035, 0.0010007], rtol=1e-12)

    # ── splits ───────────────────────────────────────────────

    def test_a_split_keys_off_its_parent_asdf(self):
        """The calibration belongs to the file the events came from, so a
        .vasdf child must pick up the parent's override -- the same rule the
        scan filter uses."""
        cal = {self.path: {"mode": "fit", "reject": "first_n",
                           "drop_first": 3}}
        from gui.analysis.pipeline import prepare_run_data
        data, _cfg, _meta = prepare_run_data(
            "C:/nowhere/1002_A.vasdf", dict(CFG),
            split_descriptor=self.split, calibrations=cal)

        expected = fit_calibration(self.set_v, self.read_raw, 1, (0, 1, 2),
                                   VACC)
        np.testing.assert_allclose(
            data.cal_at_compute_voltages,
            [c / VACC for c in expected.coeffs_v], rtol=1e-12)

    # ── audit trail ──────────────────────────────────────────

    def test_run_metadata_records_the_calibration(self):
        cal = {self.path: {"mode": "fit", "reject": "first_n",
                           "drop_first": 3, "note": "HV settling"}}
        _data, _cfg, meta = self._prepare(calibrations=cal)
        rec = meta["calibration"]
        self.assertEqual(rec["mode"], "fit")
        self.assertEqual(rec["excluded"], [0, 1, 2])
        self.assertEqual(rec["note"], "HV settling")

    def test_metadata_records_even_the_default_calibration(self):
        """A result whose calibration is unstated can't be defended later."""
        _data, _cfg, meta = self._prepare()
        self.assertEqual(meta["calibration"]["mode"], "file")
        self.assertEqual(meta["calibration"]["n_excluded"], 0)

    # ── a missing/unreadable table must not break a working run ──

    def test_a_run_with_no_cal_table_still_loads(self):
        import asdf
        empty = os.path.join(self.tmp, "run_1009.asdf")
        asdf.AsdfFile({"CalSet": [], "CalReadback": []}).write_to(empty)
        from gui.analysis.pipeline import prepare_run_data
        data, _cfg, _meta = prepare_run_data(empty, dict(CFG))
        # Falls back to whatever clstools computed rather than failing: "no
        # override requested" must never make things worse than today.
        self.assertEqual(data.calls[:2], ["Load_Run", "Compute_Voltages"])
        self.assertEqual(data.cal_at_compute_voltages, [-999.0, -999.0])


class SubprocessHandoffTests(unittest.TestCase):
    """The map has to cross the process boundary explicitly; a subprocess
    that reads the singleton finds it empty and silently uses the file
    default."""

    def test_fit_single_run_accepts_a_calibrations_map(self):
        from gui.analysis.fitting import _fit_single_run
        params = inspect.signature(_fit_single_run).parameters
        self.assertIn("calibrations", params)
        self.assertIsNone(params["calibrations"].default)

    def test_prepare_run_data_accepts_a_calibrations_map(self):
        from gui.analysis.pipeline import prepare_run_data
        params = inspect.signature(prepare_run_data).parameters
        self.assertIn("calibrations", params)
        self.assertEqual(params["calibrations"].kind,
                         inspect.Parameter.KEYWORD_ONLY)

    def test_fit_worker_stores_and_defaults_the_map(self):
        from gui.analysis.fitting import FitWorkerThread
        params = inspect.signature(FitWorkerThread.__init__).parameters
        self.assertIn("calibrations_map", params)

        worker = FitWorkerThread(
            run_files=[], source_config={}, model_configs=[],
            fitter_config={}, output_config={},
            calibrations_map={"C:/d/run_1002.asdf": {"mode": "fit",
                                                     "excluded": [0]}})
        self.assertEqual(worker.calibrations_map,
                         {"C:/d/run_1002.asdf": {"mode": "fit",
                                                 "excluded": [0]}})

        bare = FitWorkerThread(
            run_files=[], source_config={}, model_configs=[],
            fitter_config={}, output_config={})
        self.assertEqual(bare.calibrations_map, {})

    def test_the_map_is_picklable(self):
        """It is handed to a ProcessPoolExecutor."""
        import pickle
        spec = {"C:/d/run_1002.asdf": {"mode": "borrow",
                                       "donor": "C:/d/run_1001.asdf"}}
        self.assertEqual(pickle.loads(pickle.dumps(spec)), spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
