"""The systematic scan: one chained step per assumed cooler offset.

The scan must never become a second analysis path. It sets
``_config_overrides`` and calls the same method the Run Fit button
calls, so whatever the project is configured to do is what gets
banded. What is tested here is the chain around that:

* the whole chain runs per offset, in the right order -- reference,
  GP, corrections, samples, shifts -- because the cooler offset moves
  the lab-frame voltages and therefore the rest frame the GP lives in;
* a step that fails does NOT stop the scan (the point is to find out
  which offsets are hard), but a reference that fails skips that
  step's samples rather than fitting them against a stale frame;
* every override is removed afterwards;
* each step starts from its neighbour's answer.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_systematic_scan.py -q
"""
import math
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui.analysis.systematic_scan import (  # noqa: E402
    Baseline, LIGHT_OUTPUT, SEED_BASELINE, SEED_BLOCKS, SEED_CONTINUATION,
    ScanTarget, SystematicScan, checked_entries_of, config_differences,
    model_configs_of, read_baseline, seed_model_configs, split_key,
    value_key,
)


class _Worker:
    def __init__(self, running=True):
        self._r = running

    def isRunning(self):
        return self._r


class _StubProject(QObject):
    """A project whose 'fit' evaluates a straight line in the offset."""
    results_ready = Signal(str, list, dict)

    def __init__(self, name, runs=("7947",), slope=-13.0, base=100.0,
                 *, worker=True, fail_at=(), sigma=0.5):
        super().__init__()
        self.project_name = name
        self._runs = list(runs)
        self._slope = slope
        self._base = base
        self._sigma = sigma
        self._fail_at = set(fail_at)
        self._fit_worker = _Worker() if worker else None
        self._config_overrides = {}
        self._last_iter_dir = ""
        self.seen = []            # (dv, overrides) in call order

    def _on_fit_requested(self):
        ov = self._config_overrides
        dv = float((ov.get("source") or {}).get("cooler_offset_v", 0.0))
        self.seen.append((dv, {k: v for k, v in ov.items()}))
        label = (ov.get("output") or {}).get("iter_label", "")
        self._last_iter_dir = os.path.join("analysis", self.project_name,
                                           f"{label}_{dv:+g}V_CO")
        ok = dv not in self._fail_at
        results = []
        for run in self._runs:
            if not ok:
                results.append({"success": False, "run_number": run,
                                "source_name": f"Run_{run}",
                                "error": "did not converge"})
                continue
            results.append({
                "success": True, "run_number": run,
                "source_name": f"Run_{run}",
                "fit_quality": {"redchi": 1.0},
                "params_df": {
                    "Source": [f"Run_{run}"] * 2,
                    "Model": ["HFS_1"] * 2,
                    "Parameter": ["centroid", "Al"],
                    "Value": [self._base + self._slope * dv, 200.0],
                    "Stderr": [self._sigma, 1.0],
                },
            })
        QTimer.singleShot(
            0, lambda: self.results_ready.emit(self.project_name,
                                               results, {}))


class _StubGP(QObject):
    gp_fit_done = Signal()
    gp_fit_failed = Signal(str)

    def __init__(self, *, fail=False, start=True):
        super().__init__()
        self._fail = fail
        self._start = start
        self.fits = 0
        self.recomputes = 0
        self.order = []

    def start_gp_fit_for_scan(self):
        self.fits += 1
        self.order.append("gp")
        if not self._start:
            return False
        QTimer.singleShot(
            0, lambda: (self.gp_fit_failed.emit("singular")
                        if self._fail else self.gp_fit_done.emit()))
        return True

    def recompute_corrections(self):
        self.recomputes += 1
        self.order.append("corr")


class _StubIS:
    def __init__(self, rows=None):
        # `rows or default` would swallow a deliberate empty list.
        self.rows = (list(rows) if rows is not None
                     else [{"label": "70Ge", "delta_nu": 1.0}])
        self.calls = 0

    def shifts_for_scan(self):
        self.calls += 1
        return list(self.rows)


def _spin(scan, timeout_ms=5000):
    loop = QEventLoop()
    out = {}
    scan.finished.connect(lambda recs: (out.setdefault("records", recs),
                                        loop.quit()))
    scan.failed.connect(lambda msg: (out.setdefault("error", msg),
                                     loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return out


def _target(project, role="sample", baseline=None):
    return ScanTarget(project=project, role=role,
                      baseline=baseline or Baseline())


class KeyTests(unittest.TestCase):
    """Flat keys, because these get saved with the session."""

    def test_a_key_round_trips(self):
        k = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        self.assertEqual(split_key(k),
                         ("70Ge", "Run_7947", "HFS_1", "centroid"))

    def test_a_short_key_does_not_crash(self):
        self.assertEqual(split_key("70Ge"), ("70Ge", "", "", ""))


class ReadBaselineTests(unittest.TestCase):
    """What the chosen iteration hands the scan."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _write(self, name, text):
        with open(os.path.join(self.dir, name), "w",
                  encoding="utf-8") as fh:
            fh.write(text)

    def _params(self):
        self._write("parameters.csv",
                    "run_number,Parameter,Value,Error,Vary,Min,Max,"
                    "Source,Model\n"
                    "7947,centroid,-123.5,0.4,True,-inf,inf,"
                    "Run_7947,HFS_1\n"
                    "7947,FWHMG,80.0,2.0,True,0.0,500.0,"
                    "Run_7947,HFS_1\n"
                    "7948,centroid,-125.0,0.5,False,-inf,inf,"
                    "Run_7948,HFS_1\n")

    def test_an_empty_directory_is_not_an_error(self):
        b = read_baseline(os.path.join(self.dir, "nope"))
        self.assertEqual(b.params, {})
        self.assertEqual(b.sources, [])

    def test_it_reads_the_fitted_values(self):
        self._params()
        b = read_baseline(self.dir)
        self.assertEqual(b.sources, ["Run_7947", "Run_7948"])
        self.assertAlmostEqual(b.value("Run_7947", "HFS_1", "centroid"),
                               -123.5)
        self.assertAlmostEqual(b.sigma("Run_7947", "HFS_1", "centroid"),
                               0.4)

    def test_it_is_keyed_by_source_not_run(self):
        """A merged spectrum and a virtual split have no run number of
        their own, but they always have a source name."""
        self._params()
        b = read_baseline(self.dir)
        self.assertIn("Run_7947", b.params)
        self.assertEqual(b.run_of_source["Run_7947"], "7947")

    def test_it_keeps_the_bounds_and_the_vary_flag(self):
        self._params()
        row = read_baseline(self.dir).params["Run_7947"][("HFS_1", "FWHMG")]
        self.assertEqual((row["min"], row["max"]), (0.0, 500.0))
        self.assertTrue(row["vary"])
        self.assertFalse(
            read_baseline(self.dir).params["Run_7948"][
                ("HFS_1", "centroid")]["vary"])

    def test_open_bounds_read_as_infinite(self):
        self._params()
        row = read_baseline(self.dir).params["Run_7947"][
            ("HFS_1", "centroid")]
        self.assertEqual(row["min"], float("-inf"))
        self.assertEqual(row["max"], float("inf"))

    def test_it_lists_the_parameters_that_were_fitted(self):
        self._params()
        self.assertEqual(read_baseline(self.dir).parameters(),
                         [("HFS_1", "FWHMG"), ("HFS_1", "centroid")])

    def test_it_reads_the_config_snapshot(self):
        self._write("config_snapshot.yaml",
                    "project_name: 70Ge\nblocks:\n  - type: Source\n")
        b = read_baseline(self.dir)
        self.assertEqual(b.config["project_name"], "70Ge")

    def test_it_reads_the_per_run_chi_square(self):
        self._write("metadata.csv",
                    "run_number,Reduced Chi-sq\n7947,1.23\n")
        self.assertAlmostEqual(read_baseline(self.dir).redchi["7947"], 1.23)

    def test_a_corrupt_csv_does_not_bring_the_tab_down(self):
        self._write("parameters.csv", "not,a,parameters,file\n1,2\n")
        self.assertEqual(read_baseline(self.dir).sources, [])


class ConfigDifferenceTests(unittest.TestCase):
    """The scan fits with the project as it stands now; rebuilding the
    blocks under the user would be worse than telling them."""

    def test_identical_configs_differ_in_nothing(self):
        cfg = {"blocks": [{"type": "Source", "bin_count": 40}]}
        self.assertEqual(config_differences(cfg, dict(cfg)), [])

    def test_a_changed_setting_is_named(self):
        now = {"blocks": [{"type": "Source", "bin_count": 50}]}
        then = {"blocks": [{"type": "Source", "bin_count": 40}]}
        diff = config_differences(now, then)
        self.assertEqual(len(diff), 1)
        self.assertIn("bin_count", diff[0])
        self.assertIn("40", diff[0])
        self.assertIn("50", diff[0])

    def test_a_block_added_or_removed(self):
        now = {"blocks": [{"type": "Source"}, {"type": "Model"}]}
        then = {"blocks": [{"type": "Source"}]}
        self.assertIn("entries", config_differences(now, then)[0])

    def test_cosmetics_are_not_differences(self):
        now = {"blocks": [{"type": "Source", "order": 1}]}
        then = {"blocks": [{"type": "Source", "order": 0}]}
        self.assertEqual(config_differences(now, then), [])

    def test_a_deeply_nested_config_terminates(self):
        deep = cur = {}
        for _ in range(30):
            cur["k"] = {}
            cur = cur["k"]
        self.assertIsInstance(config_differences(deep, {}), list)


class SeedModelConfigTests(unittest.TestCase):
    """Seeds say where to start, not what is free."""

    def _cfgs(self):
        return [{
            "type": "HFS", "name": "HFS 1",
            "params": {
                "centroid": {"value": -100.0, "vary": True,
                             "min": -500.0, "max": 500.0, "expr": ""},
                "FWHMG": {"value": 80.0, "vary": False,
                          "min": 0.0, "max": None, "expr": ""},
            },
            "peak_amplitudes": {"1_1": {"value": 5.0, "vary": True}},
        }]

    def test_a_seed_replaces_the_starting_value(self):
        out = seed_model_configs(
            self._cfgs(), {("HFS_1", "centroid"): -350.0})
        self.assertAlmostEqual(out[0]["params"]["centroid"]["value"],
                               -350.0)

    def test_it_matches_the_SANITIZED_model_name(self):
        """A fit result names the model 'HFS_1'; the block is called
        'HFS 1'. Matching on the block name would seed nothing."""
        out = seed_model_configs(
            self._cfgs(), {("HFS 1", "centroid"): -350.0})
        self.assertAlmostEqual(out[0]["params"]["centroid"]["value"],
                               -100.0)

    def test_the_vary_flag_is_left_alone(self):
        out = seed_model_configs(
            self._cfgs(), {("HFS_1", "FWHMG"): 90.0})
        self.assertFalse(out[0]["params"]["FWHMG"]["vary"])

    def test_a_window_that_still_holds_the_seed_does_not_move(self):
        out = seed_model_configs(
            self._cfgs(), {("HFS_1", "centroid"): -300.0})
        self.assertEqual(out[0]["params"]["centroid"]["min"], -500.0)

    def test_a_window_travels_with_a_seed_that_left_it(self):
        """Otherwise the fitter refuses the seed, or pins the
        parameter to the bound and returns it without an error."""
        out = seed_model_configs(
            self._cfgs(), {("HFS_1", "centroid"): -700.0})
        p = out[0]["params"]["centroid"]
        self.assertLessEqual(p["min"], -700.0)
        self.assertAlmostEqual(p["max"] - p["min"], 1000.0)

    def test_peak_amplitudes_are_seeded_too(self):
        out = seed_model_configs(
            self._cfgs(), {("HFS_1", "Amp1_1"): 7.5})
        self.assertAlmostEqual(
            out[0]["peak_amplitudes"]["1_1"]["value"], 7.5)

    def test_a_seed_for_something_else_is_ignored(self):
        out = seed_model_configs(
            self._cfgs(), {("HFS_9", "centroid"): 1e6,
                           ("HFS_1", "nonesuch"): 1e6})
        self.assertAlmostEqual(out[0]["params"]["centroid"]["value"],
                               -100.0)

    def test_a_non_finite_seed_is_ignored(self):
        out = seed_model_configs(
            self._cfgs(), {("HFS_1", "centroid"): float("nan")})
        self.assertAlmostEqual(out[0]["params"]["centroid"]["value"],
                               -100.0)

    def test_the_project_blocks_are_not_touched(self):
        cfgs = self._cfgs()
        seed_model_configs(cfgs, {("HFS_1", "centroid"): -350.0})
        self.assertAlmostEqual(cfgs[0]["params"]["centroid"]["value"],
                               -100.0)

    def test_nothing_to_seed(self):
        cfgs = self._cfgs()
        self.assertEqual(seed_model_configs(cfgs, {}), cfgs)


class ProjectReadingTests(unittest.TestCase):
    """The helpers degrade quietly on anything that is not a real
    project, so a stub in a test harness cannot crash a scan."""

    def test_a_stub_has_no_blocks(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        self.assertEqual(model_configs_of(p), [])
        self.assertEqual(checked_entries_of(p), [])


class ScanWalkTests(unittest.TestCase):

    def _run(self, targets, offsets, **kw):
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan.start(targets, offsets, **kw)
        return scan, _spin(scan)

    def test_it_fits_every_offset(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        _s, out = self._run([_target(p)], [-30.0, -20.0, -10.0])
        self.assertEqual(sorted(dv for dv, _ov in p.seen),
                         [-30.0, -20.0, -10.0])
        self.assertEqual(len(out["records"]), 3)

    def test_it_walks_outward_from_the_baseline(self):
        """So every step has a neighbour, closer in, already fitted."""
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        self._run([_target(p)], [-30.0, -20.0, -10.0], baseline_dv=-20.0)
        self.assertEqual([dv for dv, _ov in p.seen],
                         [-20.0, -30.0, -10.0])

    def test_the_offset_reaches_the_fit_as_a_source_override(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        self._run([_target(p)], [-30.0, -10.0])
        _dv, ov = p.seen[0]
        self.assertEqual(ov["source"], {"cooler_offset_v": -30.0}
                         if _dv == -30.0 else ov["source"])
        self.assertIn("cooler_offset_v", ov["source"])

    def test_each_step_writes_its_own_labelled_folder(self):
        """Manual mode plus the offset tag gives sys_001_-30V_CO, so a
        re-run overwrites its own folder instead of piling up."""
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        self._run([_target(p)], [-30.0, -10.0], label="sys_007")
        _dv, ov = p.seen[0]
        self.assertEqual(ov["output"]["iter_mode"], "Manual")
        self.assertEqual(ov["output"]["iter_label"], "sys_007")

    def test_plot_output_is_off_by_default(self):
        """Ten offsets over five projects would otherwise write fifty
        plot sets nobody asked for."""
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        self._run([_target(p)], [-30.0, -10.0])
        _dv, ov = p.seen[0]
        for key in LIGHT_OUTPUT:
            self.assertFalse(ov["output"][key], key)

    def test_full_output_can_be_asked_for(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        self._run([_target(p)], [-30.0, -10.0], light_output=False)
        _dv, ov = p.seen[0]
        self.assertNotIn("fit_plots", ov["output"])

    def test_the_override_is_removed_afterwards(self):
        """A leftover offset would silently move every later fit."""
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        self._run([_target(p)], [-30.0, -10.0])
        self.assertEqual(p._config_overrides, {})

    def test_it_collects_the_fitted_values(self):
        p = _StubProject("70Ge", runs=("7947",), slope=-13.0, base=100.0)
        self.addCleanup(p.deleteLater)
        _s, out = self._run([_target(p)], [-30.0, -10.0])
        rec = {r["dv"]: r for r in out["records"]}[-30.0]
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        self.assertAlmostEqual(rec["values"][key]["value"],
                               100.0 + 390.0)
        self.assertAlmostEqual(rec["values"][key]["sigma"], 0.5)

    def test_it_records_which_run_each_source_is(self):
        p = _StubProject("70Ge", runs=("7947",))
        self.addCleanup(p.deleteLater)
        _s, out = self._run([_target(p)], [-30.0, -10.0])
        self.assertEqual(out["records"][0]["labels"]["70Ge|Run_7947"],
                         "7947")

    def test_it_records_the_folder_each_step_wrote(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        _s, out = self._run([_target(p)], [-30.0, -10.0], label="sys_001")
        self.assertIn("sys_001", out["records"][0]["iterations"]["70Ge"])

    def test_stop_ends_the_scan_after_the_step_in_flight(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan.step_done.connect(lambda *_a: scan.stop())
        scan.start([_target(p)], [-30.0, -20.0, -10.0])
        out = _spin(scan)
        self.assertEqual(len(out["records"]), 1)

    def test_a_project_that_cannot_fit_aborts_the_scan(self):
        """Rather than waiting for a signal that is never coming."""
        p = _StubProject("70Ge", worker=False)
        self.addCleanup(p.deleteLater)
        _s, out = self._run([_target(p)], [-30.0, -10.0])
        self.assertIn("did not start a fit", out["error"])

    def test_the_scan_refuses_an_empty_setup(self):
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        with self.assertRaises(ValueError):
            scan.start([], [-30.0])
        with self.assertRaises(ValueError):
            scan.start([_target(_StubProject("70Ge"))], [])


class StepFailureTests(unittest.TestCase):
    """A hard offset is a finding, not a crash."""

    def _run(self, targets, offsets, **kw):
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        self.failures = []
        scan.step_failed.connect(
            lambda dv, msg: self.failures.append((dv, msg)))
        scan.start(targets, offsets, **kw)
        return scan, _spin(scan)

    def test_a_failed_step_does_not_stop_the_scan(self):
        p = _StubProject("70Ge", fail_at=(-30.0,))
        self.addCleanup(p.deleteLater)
        _s, out = self._run([_target(p)], [-30.0, -20.0, -10.0])
        self.assertEqual(len(out["records"]), 3)
        self.assertEqual([r["status"] for r in out["records"]],
                         ["failed", "ok", "ok"])

    def test_the_failure_is_reported_with_its_offset(self):
        p = _StubProject("70Ge", fail_at=(-30.0,))
        self.addCleanup(p.deleteLater)
        self._run([_target(p)], [-30.0, -10.0])
        self.assertEqual(self.failures[0][0], -30.0)

    def test_a_failed_step_contributes_no_numbers(self):
        p = _StubProject("70Ge", fail_at=(-30.0,))
        self.addCleanup(p.deleteLater)
        _s, out = self._run([_target(p)], [-30.0, -10.0])
        self.assertEqual(out["records"][0]["values"], {})

    def test_a_failed_reference_skips_that_step_s_samples(self):
        """Fitting them against a GP trained on a frame that never
        arrived would report a number that means nothing."""
        ref = _StubProject("74Ge", fail_at=(-30.0,))
        sample = _StubProject("70Ge")
        self.addCleanup(ref.deleteLater)
        self.addCleanup(sample.deleteLater)
        self._run([_target(ref, "reference"), _target(sample)],
                  [-30.0, -10.0])
        self.assertEqual([dv for dv, _ov in sample.seen], [-10.0])
        self.assertEqual(sorted(dv for dv, _ov in ref.seen),
                         [-30.0, -10.0])


class ChainTests(unittest.TestCase):
    """The cooler offset moves the lab-frame voltages, so the rest
    frame the GP lives in moves too: every step retrains it."""

    def _run(self, targets, offsets, **kw):
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan.start(targets, offsets, **kw)
        return scan, _spin(scan)

    def _setup(self, **kw):
        ref = _StubProject("74Ge")
        sample = _StubProject("70Ge")
        self.addCleanup(ref.deleteLater)
        self.addCleanup(sample.deleteLater)
        gp = _StubGP(**kw)
        self.addCleanup(gp.deleteLater)
        return ref, sample, gp

    def test_the_gp_is_refitted_at_every_offset(self):
        ref, sample, gp = self._setup()
        self._run([_target(ref, "reference"), _target(sample)],
                  [-30.0, -20.0, -10.0], gp=gp)
        self.assertEqual(gp.fits, 3)

    def test_the_corrections_are_recomputed_after_each_refit(self):
        ref, sample, gp = self._setup()
        self._run([_target(ref, "reference"), _target(sample)],
                  [-30.0, -10.0], gp=gp)
        self.assertEqual(gp.recomputes, 2)
        self.assertEqual(gp.order, ["gp", "corr", "gp", "corr"])

    def test_the_reference_is_fitted_before_the_gp_and_the_sample_after(self):
        order = []
        ref, sample, gp = self._setup()
        ref.results_ready.connect(lambda *_a: order.append("ref"))
        sample.results_ready.connect(lambda *_a: order.append("sample"))
        gp.gp_fit_done.connect(lambda: order.append("gp"))
        self._run([_target(ref, "reference"), _target(sample)],
                  [-30.0], gp=gp)
        self.assertEqual(order, ["ref", "gp", "sample"])

    def test_a_gp_that_will_not_refit_fails_the_step_not_the_scan(self):
        ref, sample, gp = self._setup(start=False)
        _s, out = self._run([_target(ref, "reference"), _target(sample)],
                            [-30.0, -10.0], gp=gp)
        self.assertNotIn("error", out)
        self.assertEqual([r["status"] for r in out["records"]],
                         ["failed", "failed"])

    def test_a_gp_fit_that_fails_fails_the_step(self):
        ref, sample, gp = self._setup(fail=True)
        _s, out = self._run([_target(ref, "reference"), _target(sample)],
                            [-30.0], gp=gp)
        self.assertEqual(out["records"][0]["status"], "failed")
        self.assertIn("GP", out["records"][0]["message"])

    def test_the_isotope_shifts_are_recomputed_per_offset(self):
        ref, sample, gp = self._setup()
        is_tab = _StubIS()
        self._run([_target(ref, "reference"), _target(sample)],
                  [-30.0, -10.0], gp=gp, is_tab=is_tab)
        self.assertEqual(is_tab.calls, 2)

    def test_the_shifts_are_stored_with_the_step(self):
        ref, sample, gp = self._setup()
        is_tab = _StubIS([{"label": "70Ge", "delta_nu": 42.0}])
        _s, out = self._run([_target(sample)], [-30.0, -10.0],
                            is_tab=is_tab)
        self.assertEqual(out["records"][0]["shifts"][0]["delta_nu"], 42.0)

    def test_without_a_gp_the_chain_is_just_the_fits(self):
        ref, sample, gp = self._setup()
        self._run([_target(sample)], [-30.0, -10.0])
        self.assertEqual(gp.fits, 0)


class SeedingTests(unittest.TestCase):
    """Each step starts from what the neighbouring offset found."""

    def setUp(self):
        self.scan = SystematicScan()
        self.addCleanup(self.scan.deleteLater)
        self.p = _StubProject("70Ge")
        self.addCleanup(self.p.deleteLater)
        base = Baseline(params={"Run_7947": {
            ("HFS_1", "centroid"): {"value": -100.0, "sigma": 0.4,
                                    "vary": True, "min": float("-inf"),
                                    "max": float("inf")}}})
        self.t = _target(self.p, baseline=base)

    def _seed(self, dv):
        return self.scan._seeds_for(self.t, "Run_7947", dv)

    def test_the_first_step_starts_from_the_baseline(self):
        self.scan._seed_mode = SEED_CONTINUATION
        self.assertAlmostEqual(self._seed(-30.0)[("HFS_1", "centroid")],
                               -100.0)

    def test_a_later_step_starts_from_its_neighbour(self):
        self.scan._seed_mode = SEED_CONTINUATION
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        self.scan._remember({"dv": -20.0, "status": "ok", "values": {
            key: {"value": 160.0, "sigma": 0.4}}})
        self.assertAlmostEqual(self._seed(-30.0)[("HFS_1", "centroid")],
                               160.0)

    def test_two_neighbours_carry_the_trend(self):
        """The centroid moves about 13 MHz per volt; at the far end of
        the scan the baseline value is hundreds of MHz away."""
        self.scan._seed_mode = SEED_CONTINUATION
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        for dv, v in ((-20.0, 160.0), (-15.0, 95.0)):
            self.scan._remember({"dv": dv, "status": "ok", "values": {
                key: {"value": v, "sigma": 0.4}}})
        self.assertAlmostEqual(self._seed(-30.0)[("HFS_1", "centroid")],
                               290.0)

    def test_a_failed_step_teaches_nothing(self):
        self.scan._seed_mode = SEED_CONTINUATION
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        self.scan._remember({"dv": -20.0, "status": "failed", "values": {
            key: {"value": 1e9, "sigma": 0.4}}})
        self.assertAlmostEqual(self._seed(-30.0)[("HFS_1", "centroid")],
                               -100.0)

    def test_re_running_a_step_forgets_what_it_had_found(self):
        """Otherwise the fit being thrown away seeds its own re-run."""
        self.scan._seed_mode = SEED_CONTINUATION
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        self.scan._remember({"dv": -30.0, "status": "ok", "values": {
            key: {"value": 1e9, "sigma": 0.4}}})
        self.scan._forget(-30.0)
        self.assertAlmostEqual(self._seed(-30.0)[("HFS_1", "centroid")],
                               -100.0)

    def test_baseline_mode_ignores_the_neighbours(self):
        self.scan._seed_mode = SEED_BASELINE
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        self.scan._remember({"dv": -20.0, "status": "ok", "values": {
            key: {"value": 160.0, "sigma": 0.4}}})
        self.assertAlmostEqual(self._seed(-30.0)[("HFS_1", "centroid")],
                               -100.0)

    def test_blocks_mode_seeds_nothing_at_all(self):
        self.scan._seed_mode = SEED_BLOCKS
        self.assertEqual(self.scan._seed_map(self.t, -30.0), {})

    def test_a_run_the_baseline_never_fitted_gets_no_seed(self):
        self.scan._seed_mode = SEED_CONTINUATION
        self.assertEqual(self.scan._seeds_for(self.t, "Run_9999", -30.0),
                         {})


class ResumeTests(unittest.TestCase):
    """Re-running one bad step must not re-run the whole scan."""

    def test_records_handed_in_are_kept(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        old = {-10.0: {"dv": -10.0, "status": "ok", "message": "",
                       "values": {key: {"value": 7.0, "sigma": 0.4}},
                       "redchi": {}, "shifts": [], "labels": {},
                       "iterations": {}}}
        scan.start([_target(p)], [-30.0], records=old)
        out = _spin(scan)
        got = {r["dv"]: r for r in out["records"]}
        self.assertEqual(sorted(got), [-30.0, -10.0])
        self.assertAlmostEqual(got[-10.0]["values"][key]["value"], 7.0)

    def test_only_the_named_offsets_are_fitted_again(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan.start([_target(p)], [-30.0], records={
            -10.0: {"dv": -10.0, "status": "ok", "values": {},
                    "message": "", "redchi": {}, "shifts": [],
                    "labels": {}, "iterations": {}}})
        _spin(scan)
        self.assertEqual([dv for dv, _ov in p.seen], [-30.0])

    def test_a_kept_step_still_seeds_the_one_being_re_run(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        key = value_key("70Ge", "Run_7947", "HFS_1", "centroid")
        scan.start([_target(p)], [-30.0], records={
            -20.0: {"dv": -20.0, "status": "ok", "values": {
                key: {"value": 160.0, "sigma": 0.4}}, "message": "",
                "redchi": {}, "shifts": [], "labels": {},
                "iterations": {}}})
        _spin(scan)
        by_dv = {dv: v for dv, v, _s in scan._history_for(key)}
        self.assertAlmostEqual(by_dv[-20.0], 160.0)


class _FakeFuture:
    def __init__(self, value):
        self._v = value

    def result(self, timeout=None):
        return self._v


class _FakePool:
    """Runs the 'subprocess' inline and records what it was given."""
    calls = []

    def __init__(self, max_workers=1):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, filepath, source_config, model_configs, *a, **kw):
        _FakePool.calls.append((filepath, model_configs))
        return _FakeFuture({"success": True, "run_number": "x",
                            "run_file": filepath})

    def shutdown(self, **kw):
        pass


class _FakeManager:
    def Queue(self):
        import queue
        return queue.Queue()

    def shutdown(self):
        pass


class PerRunSeedPlumbingTests(unittest.TestCase):
    """The seeds have to reach the worker, per file.

    Every fit before this shared one set of starting values across all
    runs. A scan seeds each run from its own previous fit, which is the
    difference between converging at the end of the range and not.
    """

    MODELS = [{"type": "HFS", "name": "HFS_1",
               "params": {"centroid": {"value": -100.0}}}]

    def _run(self, model_configs_map):
        from unittest import mock
        from gui.analysis.fitting import FitWorkerThread
        _FakePool.calls = []
        w = FitWorkerThread(
            run_files=["a.asdf", "b.asdf"], source_config={},
            model_configs=self.MODELS, fitter_config={"separate": True},
            output_config={}, n_cores=1,
            model_configs_map=model_configs_map)
        self.addCleanup(w.deleteLater)
        with mock.patch("multiprocessing.Manager", _FakeManager), \
             mock.patch("gui.analysis.fitting.ProcessPoolExecutor",
                        _FakePool), \
             mock.patch("gui.analysis.fitting.as_completed",
                        lambda fs: list(fs)):
            w._run_separate()
        return dict(_FakePool.calls)

    def test_without_a_map_every_run_shares_the_blocks(self):
        """The path every other caller takes must be untouched."""
        got = self._run(None)
        self.assertEqual(got["a.asdf"], self.MODELS)
        self.assertIs(got["b.asdf"], self.MODELS)

    def test_a_seeded_run_gets_its_own_starting_values(self):
        seeded = [{"type": "HFS", "name": "HFS_1",
                   "params": {"centroid": {"value": 390.0}}}]
        got = self._run({"a.asdf": seeded})
        self.assertAlmostEqual(
            got["a.asdf"][0]["params"]["centroid"]["value"], 390.0)

    def test_the_other_runs_keep_the_shared_values(self):
        seeded = [{"type": "HFS", "name": "HFS_1",
                   "params": {"centroid": {"value": 390.0}}}]
        got = self._run({"a.asdf": seeded})
        self.assertAlmostEqual(
            got["b.asdf"][0]["params"]["centroid"]["value"], -100.0)


class HeadlessChainTests(unittest.TestCase):
    """The scan drives the GP panel and the isotope-shift tab with no
    user in front of them. Nothing in that path may open a dialog: a
    modal box in the middle of a 250-fit scan stops it dead."""

    def setUp(self):
        from gui.analysis.tab import AnalysisTab
        self.at = AnalysisTab()
        self.addCleanup(self.at.deleteLater)
        self.panel = self.at._gp_tab
        self.is_tab = self.at._is_tab
        from PySide6.QtWidgets import QMessageBox
        self._boxes = []
        for name in ("warning", "information", "critical", "question"):
            real = getattr(QMessageBox, name)
            setattr(QMessageBox, name,
                    staticmethod(lambda *a, _n=name, **k:
                                 self._boxes.append(_n)))
            self.addCleanup(setattr, QMessageBox, name, real)

    def test_the_panel_offers_what_the_scan_needs(self):
        for name in ("start_gp_fit_for_scan", "recompute_corrections",
                     "gp_fit_done", "gp_fit_failed"):
            self.assertTrue(hasattr(self.panel, name), name)

    def test_a_gp_refit_with_nothing_to_fit_says_no(self):
        """Rather than starting a fit the scan would wait forever for."""
        self.assertFalse(self.panel.start_gp_fit_for_scan())
        self.assertEqual(self._boxes, [])

    def test_recomputing_without_a_gp_is_harmless(self):
        self.assertEqual(self.panel.recompute_corrections(), 0)
        self.assertEqual(self._boxes, [])

    def test_a_failed_gp_fit_reaches_the_scan_silently(self):
        seen = []
        self.panel.gp_fit_failed.connect(seen.append)
        self.panel._quiet = True
        self.panel._fit_failed("singular matrix")
        self.assertEqual(seen, ["singular matrix"])
        self.assertEqual(self._boxes, [])

    def test_a_failed_gp_fit_still_tells_a_user_who_is_there(self):
        self.panel._quiet = False
        self.panel._fit_failed("singular matrix")
        self.assertEqual(self._boxes, ["critical"])

    def test_the_quiet_flag_does_not_stick(self):
        """A scan that ends in a failure must not silence the panel
        for the rest of the session."""
        self.panel._quiet = True
        self.panel._fit_failed("singular matrix")
        self.assertFalse(self.panel._quiet)

    def test_the_is_tab_offers_what_the_scan_needs(self):
        self.assertTrue(hasattr(self.is_tab, "shifts_for_scan"))

    def test_shifts_from_an_unconfigured_tab_are_empty_and_quiet(self):
        self.assertEqual(self.is_tab.shifts_for_scan(), [])
        self.assertEqual(self._boxes, [])

    def test_it_says_why_it_had_nothing_to_give(self):
        self.is_tab.shifts_for_scan()
        self.assertTrue(self.is_tab._quiet_messages)
        self.assertIn("isotopes", self.is_tab._quiet_messages[0])

    def test_it_never_hands_back_the_previous_offset_s_shifts(self):
        """A stale row silently attributed to this offset would be
        worse than a gap."""
        self.is_tab._last_shift_data = [{"label": "stale"}]
        self.assertEqual(self.is_tab.shifts_for_scan(), [])

    def test_a_user_facing_compute_still_warns(self):
        self.is_tab._compute_shifts()
        self.assertEqual(self._boxes, ["warning"])


class _FitterBlockStub:
    def __init__(self):
        self.running = None

    def set_running(self, on):
        self.running = on

    def set_progress(self, *a):
        pass


class QuietProjectTests(unittest.TestCase):
    """A project driven by the scan has no user in front of it.

    Every dialog in the fit path is a place the scan can hang for
    ever: a modal box blocks the event loop the fits are reported on.
    """

    def setUp(self):
        from PySide6.QtWidgets import QMessageBox
        self.boxes = []
        for name in ("warning", "information", "critical", "question"):
            real = getattr(QMessageBox, name)
            setattr(QMessageBox, name,
                    staticmethod(lambda *a, _n=name, **k:
                                 self.boxes.append(_n)))
            self.addCleanup(setattr, QMessageBox, name, real)

    def _project(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("70Ge")
        self.addCleanup(p.deleteLater)
        return p

    def test_a_project_that_cannot_fit_stays_quiet(self):
        """No files ticked: the driver notices no worker started."""
        p = self._project()
        p._scan_driven = True
        p._on_fit_requested()
        self.assertEqual(self.boxes, [])
        self.assertIsNone(getattr(p, "_fit_worker", None))

    def test_a_user_pressing_Run_Fit_is_still_told(self):
        p = self._project()
        p._scan_driven = False
        p._on_fit_requested()
        self.assertEqual(self.boxes, ["warning"])

    def test_a_worker_failure_reaches_the_driver(self):
        """A pool-level error emits `error` and never `all_done`, so
        without this the scan waits for a signal that is not coming."""
        p = self._project()
        p._scan_driven = True
        seen = []
        p.results_ready.connect(lambda n, r, o: seen.append((n, r)))
        p._on_fit_error("pool blew up", _FitterBlockStub())
        self.assertEqual(seen, [("70Ge", [])])
        self.assertEqual(self.boxes, [])

    def test_a_worker_failure_still_reaches_a_user(self):
        p = self._project()
        p._scan_driven = False
        p._on_fit_error("pool blew up", _FitterBlockStub())
        self.assertEqual(self.boxes, ["critical"])


class PartialStepTests(unittest.TestCase):
    """One project failing at one offset is not the end of the scan."""

    def _run(self, targets, offsets, **kw):
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan.start(targets, offsets, **kw)
        return scan, _spin(scan)

    def _pair(self, fail_at=()):
        a = _StubProject("70Ge", base=100.0, fail_at=fail_at)
        b = _StubProject("72Ge", base=200.0)
        self.addCleanup(a.deleteLater)
        self.addCleanup(b.deleteLater)
        return a, b

    def test_a_sample_that_failed_makes_the_step_partial(self):
        a, b = self._pair(fail_at=(-30.0,))
        _s, out = self._run([_target(a), _target(b)], [-30.0, -10.0])
        got = {r["dv"]: r for r in out["records"]}
        self.assertEqual(got[-30.0]["status"], "partial")
        self.assertEqual(got[-10.0]["status"], "ok")

    def test_the_step_names_the_project_that_is_missing(self):
        a, b = self._pair(fail_at=(-30.0,))
        _s, out = self._run([_target(a), _target(b)], [-30.0, -10.0])
        got = {r["dv"]: r for r in out["records"]}
        self.assertEqual(got[-30.0]["failed_projects"], ["70Ge"])

    def test_what_did_fit_is_kept(self):
        a, b = self._pair(fail_at=(-30.0,))
        _s, out = self._run([_target(a), _target(b)], [-30.0, -10.0])
        got = {r["dv"]: r for r in out["records"]}
        key = value_key("72Ge", "Run_7947", "HFS_1", "centroid")
        self.assertIn(key, got[-30.0]["values"])

    def test_a_partial_step_still_seeds_the_next_one(self):
        a, b = self._pair(fail_at=(-30.0,))
        scan, _out = self._run([_target(a), _target(b)], [-30.0, -10.0])
        key = value_key("72Ge", "Run_7947", "HFS_1", "centroid")
        self.assertEqual(len(scan._history_for(key)), 2)

    def test_the_project_is_released_when_the_scan_ends(self):
        """A project left in scan mode would swallow the warnings its
        own user needs to see."""
        a, b = self._pair()
        self._run([_target(a), _target(b)], [-30.0, -10.0])
        self.assertFalse(getattr(a, "_scan_driven", False))
        self.assertFalse(getattr(b, "_scan_driven", False))


class ShiftBandingTests(unittest.TestCase):
    """The isotope shift is the number that gets published.

    Its dependence on the assumed cooler offset is much weaker than
    either centroid's, because the reference and the sample move
    together -- and measuring that cancellation rather than assuming
    it is the whole point of running the chain per offset.
    """

    def _rows(self, dv):
        return [{"label": "70Ge", "A": 70,
                 "centroid": 1500.0 - 13.0 * dv,
                 "sigma_fit": 0.4,
                 "delta_nu": 400.0 - 0.02 * dv,
                 "sigma_delta_nu_stat": 0.6}]

    def _run(self, offsets=(-30.0, -10.0)):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        is_tab = _StubIS()
        is_tab.shifts_for_scan = lambda: self._rows(
            scan._current_dv if scan._current_dv is not None else 0.0)
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan.start([_target(p)], list(offsets), is_tab=is_tab)
        return scan, _spin(scan)

    def test_the_shift_is_stored_as_a_banded_value(self):
        _s, out = self._run()
        key = value_key("Isotope shifts", "70Ge", "IS", "delta_nu")
        rec = {r["dv"]: r for r in out["records"]}[-30.0]
        self.assertAlmostEqual(rec["values"][key]["value"], 400.6)
        self.assertAlmostEqual(rec["values"][key]["sigma"], 0.6)

    def test_the_drift_free_centroid_is_stored_too(self):
        """It is not the same number as the fitted centroid: this one
        has the GP correction in it."""
        _s, out = self._run()
        key = value_key("Isotope shifts", "70Ge", "IS",
                        "centroid_corrected")
        rec = {r["dv"]: r for r in out["records"]}[-30.0]
        self.assertAlmostEqual(rec["values"][key]["value"], 1890.0)

    def test_the_rows_are_kept_whole_as_well(self):
        _s, out = self._run()
        self.assertEqual(out["records"][0]["shifts"][0]["label"], "70Ge")

    def test_the_isotope_is_labelled(self):
        _s, out = self._run()
        self.assertEqual(
            out["records"][0]["labels"]["Isotope shifts|70Ge"], "70Ge")

    def test_a_tab_that_gives_nothing_back_adds_nothing(self):
        p = _StubProject("70Ge")
        self.addCleanup(p.deleteLater)
        is_tab = _StubIS([])
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan.start([_target(p)], [-30.0], is_tab=is_tab)
        out = _spin(scan)
        self.assertFalse([k for k in out["records"][0]["values"]
                          if k.startswith("Isotope shifts")])


if __name__ == "__main__":
    unittest.main()
