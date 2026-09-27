"""GP corrections cover every loaded run, not only the fitted ones.

Date:    2026-09-20
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

"Compute per-file corrections" used to key its table off the project's
last fit results. That left the merge dialog's "Align centroids before
merging" permanently greyed out in the one situation where the
correction matters most: a project whose fit runs on a MERGED entry.
The fit result is keyed ``merged://<name>``, so the constituent ASDFs
-- the paths the merge dialog looks up -- never got a row.

A correction needs nothing but a timestamp, so the panel now collects
one for every run it can date, from the cheapest source available:
last fit results, then a merged entry's per-run audit table, then one
lazily-read cell of the ASDF itself. Pinned here:

* merged constituents get rows even when only the merge was fitted;
* an unfitted project still yields rows (correct, merge, then fit --
  the natural order);
* ``merged://`` pseudo-paths never get a row of their own, because a
  merged spectrum inherits its constituents' corrections at merge
  time (``_merged_run_metadata``) and a second one would double-count;
* the ASDF timestamp reader agrees with the ``ts_start`` merge.py
  records, and degrades to 0.0 rather than raising.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_gp_correction_coverage.py -q

Depends on: gui.analysis.reference_correction_panel; PySide6, asdf,
NumPy.
"""

import os
import tempfile
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from gui.analysis.reference_correction_panel import (  # noqa: E402
    ReferenceCorrectionPanel, _asdf_ts_start, _kernel_name, _wrap_tip,
)


# ── Fakes ────────────────────────────────────────────────────────
# _run_timestamps is a staticmethod that only reads _last_results and
# the Source blocks' _file_entries, so a duck-typed project is enough
# and keeps the test off the real widget tree (fast, no fit pipeline).

class _FakeSourceBlock:
    def __init__(self, entries):
        self._file_entries = entries


class _FakeProject:
    def __init__(self, entries=(), results=None):
        self._blocks = [_FakeSourceBlock(list(entries))]
        self._last_results = list(results or [])

    def _get_blocks_by_type(self, cls):
        # The real one filters by isinstance + is_enabled; the panel
        # only ever asks for SourceBlock, so hand back what we have.
        return self._blocks


def _plain(path, run_number, **extra):
    e = {"path": path, "run_number": run_number}
    e.update(extra)
    return e


def _merged(name, constituents):
    """A merged file entry, shaped like SourceBlock._add_merged_entry."""
    return {
        "path": f"merged://{name}",
        "run_number": name,
        "is_merged": True,
        "merged_data": {
            "merged_name": name,
            "per_run": [{"path": p, "run_num": r, "ts_start": ts}
                        for p, r, ts in constituents],
        },
    }


def _result(run_file, run_number, ts_start, success=True):
    return {"success": success, "run_file": run_file,
            "run_number": run_number,
            "run_metadata": {"ts_start": ts_start}}


_TS = ReferenceCorrectionPanel._run_timestamps


class MergedConstituentsTests(unittest.TestCase):
    """The regression the user hit: fitted on the merge, no rows."""

    def test_constituents_get_rows_when_only_the_merge_was_fitted(self):
        entries = [
            _plain("C:/data/run_7895.asdf", "7895"),
            _plain("C:/data/run_7899.asdf", "7899"),
            _merged("merged_7895_7899",
                    [("C:/data/run_7895.asdf", "7895", 1783957209.22),
                     ("C:/data/run_7899.asdf", "7899", 1783960239.04)]),
        ]
        proj = _FakeProject(
            entries,
            [_result("merged://merged_7895_7899", "merged_7895_7899", 0)])
        stamps, no_ts = _TS(proj)
        self.assertEqual(
            sorted(stamps), ["C:/data/run_7895.asdf",
                             "C:/data/run_7899.asdf"])
        self.assertEqual(stamps["C:/data/run_7895.asdf"],
                         (1783957209.22, 0.0, "7895"))
        self.assertEqual(no_ts, [])

    def test_merged_pseudo_path_never_gets_its_own_row(self):
        """A merged spectrum inherits its constituents' corrections at
        merge time; a row of its own would apply a second shift."""
        entries = [_merged("m", [("C:/data/run_1.asdf", "1", 100.0)])]
        proj = _FakeProject(entries, [
            {"success": True, "run_file": "merged://m",
             "run_number": "m", "run_metadata": {"ts_start": 12345.0}},
        ])
        stamps, _ = _TS(proj)
        self.assertNotIn("merged://m", stamps)
        self.assertIn("C:/data/run_1.asdf", stamps)

    def test_fit_result_wins_over_the_merge_audit_table(self):
        """Both sources carry the same run; the fit result is the
        authoritative one and must not be overwritten."""
        entries = [
            _plain("C:/data/run_1.asdf", "1"),
            _merged("m", [("C:/data/run_1.asdf", "1", 222.0)]),
        ]
        proj = _FakeProject(
            entries, [_result("C:/data/run_1.asdf", "1", 111.0)])
        stamps, _ = _TS(proj)
        self.assertEqual(stamps["C:/data/run_1.asdf"], (111.0, 0.0, "1"))


class UnfittedProjectTests(unittest.TestCase):
    def test_timestamps_read_from_the_asdf_when_nothing_was_fitted(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_asdf(d, "run_42.asdf", 1700000000.5)
            proj = _FakeProject([_plain(path, "42")])
            stamps, no_ts = _TS(proj)
            self.assertEqual(no_ts, [])
            self.assertAlmostEqual(stamps[path][0], 1700000000.5, places=2)
            self.assertEqual(stamps[path][2], "42")
            # The acquisition END also comes back, so the correction
            # can be averaged over the run rather than taken at its
            # first instant.
            self.assertGreater(stamps[path][1], stamps[path][0])

    def test_unreadable_file_is_reported_not_raised(self):
        proj = _FakeProject([_plain("C:/nope/run_9.asdf", "9")])
        stamps, no_ts = _TS(proj)
        self.assertEqual(stamps, {})
        self.assertEqual(no_ts, ["9"])

    def test_split_entry_dates_from_its_parent_asdf(self):
        """A .vasdf is a descriptor with no events of its own."""
        with tempfile.TemporaryDirectory() as d:
            parent = _write_asdf(d, "run_7.asdf", 1650000000.0)
            split = os.path.join(d, "run_7_lo.vasdf")
            proj = _FakeProject([
                _plain(split, "7lo", is_split=True, parent_path=parent)])
            stamps, no_ts = _TS(proj)
            self.assertEqual(no_ts, [])
            self.assertAlmostEqual(stamps[split][0], 1650000000.0, places=2)

    def test_failed_fits_do_not_block_the_asdf_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_asdf(d, "run_3.asdf", 1600000000.0)
            proj = _FakeProject(
                [_plain(path, "3")],
                [_result(path, "3", 1600000000.0, success=False)])
            stamps, no_ts = _TS(proj)
            self.assertEqual(no_ts, [])
            self.assertIn(path, stamps)


class AsdfTimestampReaderTests(unittest.TestCase):
    def test_reads_the_first_event_timestamp(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_asdf(d, "run_1.asdf", 1783957209.22)
            self.assertAlmostEqual(_asdf_ts_start(path), 1783957209.22,
                                   places=2)

    def test_degenerate_paths_return_zero(self):
        for bad in ("", None, "merged://m", "C:/definitely/not/here.asdf"):
            with self.subTest(path=bad):
                self.assertEqual(_asdf_ts_start(bad), 0.0)

    def test_non_asdf_file_returns_zero_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "notes.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("not an asdf")
            self.assertEqual(_asdf_ts_start(p), 0.0)


class IntervalAveragedCorrectionTests(unittest.TestCase):
    """A run's fitted centroid records the drift AVERAGED over its
    acquisition, so that is what has to be subtracted -- not the
    drift at the instant the run began, which is the one moment the
    measurement is least representative of.

    On the Ge set this is worth ~2 MHz on a 5-minute run and 4.4 MHz
    on the 33-minute run 7899, against 4-6 MHz statistical errors.
    """

    def _corrector(self):
        """A GP on a steep, smooth ramp so the window average and the
        start value are cleanly distinguishable."""
        import numpy as np
        from cls_estimations.reference_correction import (
            ReferenceCorrector, ReferenceObservation)
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 6.0, 14)
        y = -100.0 - 14.0 * t + rng.normal(0, 0.4, t.size)
        obs = [ReferenceObservation(t=ti, centroid=yi, sigma=3.0)
               for ti, yi in zip(t, y)]
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        return rc

    def test_average_sits_between_the_endpoints(self):
        rc = self._corrector()
        a = float(rc.predict(2.0)[0][0])
        b = float(rc.predict(3.0)[0][0])
        mid, _ = rc.predict_interval(2.0, 3.0)
        self.assertGreater(mid, min(a, b))
        self.assertLess(mid, max(a, b))

    def test_it_differs_from_the_start_value_on_a_long_run(self):
        rc = self._corrector()
        start = float(rc.predict(2.0)[0][0])
        avg, _ = rc.predict_interval(2.0, 3.0)
        # A 1 h window on a ~14 MHz/h ramp: about half the ramp.
        self.assertGreater(abs(avg - start), 4.0)

    def test_sigma_uses_the_full_covariance_not_iid(self):
        """Values minutes apart are almost perfectly correlated.
        Averaging them must NOT shrink the error like sigma/sqrt(n) --
        that claimed a 5.4x improvement on real data that does not
        exist."""
        import math
        rc = self._corrector()
        _, sd_point = rc.predict(2.5)
        sd_point = float(sd_point[0])
        _, sd_avg = rc.predict_interval(2.4, 2.6, n=32)
        self.assertGreater(sd_avg, sd_point / 2.0)
        self.assertLess(sd_avg, sd_point * 1.5)
        self.assertGreater(sd_avg, sd_point / math.sqrt(32) * 3)

    def test_quadrature_has_converged_by_the_default_n(self):
        rc = self._corrector()
        coarse, _ = rc.predict_interval(1.0, 2.0, n=32)
        fine, _ = rc.predict_interval(1.0, 2.0, n=256)
        self.assertAlmostEqual(coarse, fine, places=2)

    def test_degenerate_window_falls_back_to_a_point(self):
        rc = self._corrector()
        mu, sd = rc.predict(2.0)
        for t2 in (2.0, 1.0, float("nan")):
            with self.subTest(t2=t2):
                self.assertEqual(rc.predict_interval(2.0, t2),
                                 (float(mu[0]), float(sd[0])))

    def test_the_window_survives_the_save_file(self):
        from gui.analysis.reference_correction_panel import _FileCorrection
        fc = _FileCorrection(
            project="70Ge", run_number="7891", file_path="C:/d/a.asdf",
            ts_start=100.0, t_hours=0.0, ts_stop=1500.0)
        self.assertEqual(fc.ts_stop, 1500.0)
        # ts_stop is LAST in the field list: positional construction
        # (which the older tests use) must keep meaning what it says.
        pos = _FileCorrection("70Ge", "7891", "C:/d/a.asdf", 100.0, 0.0,
                              "Auto", 1.0, 0.5)
        self.assertEqual(pos.mode, "Auto")
        self.assertEqual(pos.auto_value_mhz, 1.0)
        self.assertEqual(pos.ts_stop, 0.0)


class ResultsRecoveredFromDiskTests(unittest.TestCase):
    """A reopened session must not demand the fits be re-run.

    Fit results live on disk in the project's iteration directory --
    the Results tab reads them straight from there. But the GP panel
    asks the PROJECT for its reference observations, and those came
    only from ``_last_results``, which a fresh process has not
    populated. So the panel reported "0 obs -- fit first" about fits
    that had already succeeded and were on screen in another tab,
    and "Refresh project list" could not help because it re-scanned
    the same empty list.

    The loader that reads them back never needed the Isotope Shifts
    tab it lived on: it takes a project, reads that project's newest
    iteration, and fills its ``_last_results``. It is now a method on
    AnalysisProject, and the readers reach for it before giving up.
    """

    def _project_with_results_on_disk(self):
        """A project whose results exist only on disk."""
        import csv
        import os
        import tempfile
        from unittest import mock
        from gui.analysis.project import AnalysisProject

        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmp,
                        ignore_errors=True)
        iter_dir = os.path.join(tmp, "74Ge", "iter_001")
        os.makedirs(iter_dir)
        with open(os.path.join(iter_dir, "parameters.csv"), "w",
                  newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["run_number", "Parameter", "Value", "Error"])
            for run, c in (("7935", -152.5), ("7937", -140.9)):
                wr.writerow([run, "centroid", c, 3.0])
        with open(os.path.join(iter_dir, "run_summary.csv"), "w",
                  newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["run_number", "ts_start", "ts_stop"])
            wr.writerow(["7935", 1784053502.92, 1784053720.60])
            wr.writerow(["7937", 1784054143.27, 1784054271.36])

        p = AnalysisProject("74Ge", is_reference=True)
        self.addCleanup(p.deleteLater)
        patcher = mock.patch(
            "gui.shared_widgets.get_analysis_dir", return_value=tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        return p

    def test_the_loader_is_on_the_project(self):
        """Not on the Isotope Shifts tab, which is why the GP panel
        could not reach it."""
        from gui.analysis.project import AnalysisProject
        self.assertTrue(hasattr(AnalysisProject, "load_results_from_disk"))

    def test_reference_observations_recover_from_disk(self):
        p = self._project_with_results_on_disk()
        self.assertFalse(p._last_results)           # fresh process
        obs = p.get_reference_observations()
        self.assertEqual(len(obs), 2)
        self.assertEqual({o["run_number"] for o in obs}, {"7935", "7937"})
        # And the load is remembered, not repeated per call.
        self.assertEqual(len(p._last_results), 2)

    def test_the_acquisition_window_survives_the_round_trip(self):
        """ts_start/ts_stop come back too -- the correction averages
        the GP over that window, so losing it would silently fall
        back to a point prediction."""
        p = self._project_with_results_on_disk()
        obs = p.get_reference_observations()
        by_run = {o["run_number"]: o for o in obs}
        self.assertAlmostEqual(by_run["7935"]["ts_start"], 1784053502.92)

    def test_a_sample_project_still_returns_nothing(self):
        """Only reference projects supply observations; the disk read
        must not change that."""
        p = self._project_with_results_on_disk()
        p._is_reference = False
        self.assertEqual(p.get_reference_observations(), [])

    def test_a_missing_directory_is_not_an_error(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("NoSuchProject", is_reference=True)
        self.addCleanup(p.deleteLater)
        p.load_results_from_disk()            # must not raise
        self.assertEqual(p.get_reference_observations(), [])


class ExcludeReferenceRunTests(unittest.TestCase):
    """Dropping an untrustworthy reference run from the GP.

    ReferenceObservation.include and the `included` filter in fit()
    both existed from the start, but no widget ever set the flag, so
    a bad centroid could only be removed by deleting the run from the
    project entirely. The case that forced it: in Arda's Ge T02 set,
    run_7961's calibration stream was invalid and its centroid lands
    at +85 MHz among neighbours near -130. Fitted, it pulls the slow
    amplitude from 11.9 to 144.9 MHz -- a 12x inflation -- because a
    GP has no notion of an outlier and will bend to reach any point
    it is given. His own pipeline flags the run "excluded from GP
    training"; DENIS had no way to agree with it.
    """

    class _RefProject:
        is_reference = True

        def __init__(self, name, obs):
            self.project_name = name
            self._obs = obs

        def get_reference_observations(self):
            return list(self._obs)

    @staticmethod
    def _obs(run, ts_h, centroid, sigma=3.0):
        return {"label": f"run_{run}", "run_number": str(run),
                "ts_start": ts_h * 3600.0,
                "ts_stop": ts_h * 3600.0 + 120.0,
                "centroid_mhz": centroid, "sigma_mhz": sigma}

    def _panel(self):
        """A panel over one reference project holding the T02 shape:
        two sane runs around -130 and one bad one at +85."""
        from PySide6.QtWidgets import QWidget
        stub = QWidget()
        stub._projects = [self._RefProject("74Ge", [
            self._obs(7958, 100.0, -137.9),
            self._obs(7961, 100.8, 85.0),
            self._obs(7964, 101.1, -125.9),
        ])]
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        panel.refresh_projects()
        return panel

    def _row(self, panel, run):
        from PySide6.QtCore import Qt
        for i in range(panel._obs_list.count()):
            it = panel._obs_list.item(i)
            if str(it.data(Qt.ItemDataRole.UserRole)).endswith(run):
                return it
        self.fail(f"{run} not in the reference runs list")

    def test_every_run_is_listed_and_ticked_by_default(self):
        """Opting out has to be explicit -- a run silently missing
        from the fit is the failure mode this replaces."""
        panel = self._panel()
        self.assertEqual(panel._obs_list.count(), 3)
        obs = panel._gather_observations()
        self.assertTrue(all(o.include for o in obs))

    def test_runs_are_listed_in_time_order(self):
        """The list sits beside a time-axis plot; any other order
        makes matching a dot to a row guesswork."""
        panel = self._panel()
        from PySide6.QtCore import Qt
        labels = [str(panel._obs_list.item(i).data(
            Qt.ItemDataRole.UserRole))
            for i in range(panel._obs_list.count())]
        self.assertEqual(labels, ["74Ge/7958", "74Ge/7961",
                                  "74Ge/7964"])

    def test_the_row_shows_the_centroid_so_the_outlier_is_visible(self):
        """+85 among -130s is the whole reason the user is here; it
        should not require hovering each row to find it."""
        panel = self._panel()
        self.assertIn("+85.0", self._row(panel, "7961").text())

    def test_unticking_a_run_excludes_it_from_the_fit(self):
        from PySide6.QtCore import Qt
        panel = self._panel()
        self._row(panel, "7961").setCheckState(Qt.CheckState.Unchecked)
        self.assertEqual(panel._excluded_obs, {"74Ge/7961"})
        by_label = {o.label: o for o in panel._gather_observations()}
        self.assertFalse(by_label["74Ge/7961"].include)
        self.assertTrue(by_label["74Ge/7958"].include)
        # ...and re-ticking puts it back.
        self._row(panel, "7961").setCheckState(Qt.CheckState.Checked)
        self.assertEqual(panel._excluded_obs, set())

    def test_the_label_reports_how_many_are_excluded(self):
        from PySide6.QtCore import Qt
        panel = self._panel()
        self.assertIn("3 total", panel._obs_lbl.text())
        self._row(panel, "7961").setCheckState(Qt.CheckState.Unchecked)
        self.assertIn("1 excluded", panel._obs_lbl.text())

    def test_a_project_refresh_does_not_silently_untick_everything(self):
        """Rebuilding the list emits itemChanged per row. Without the
        guard, each one reads as the user unticking that run and the
        whole set would exclude itself on every refresh."""
        from PySide6.QtCore import Qt
        panel = self._panel()
        self._row(panel, "7961").setCheckState(Qt.CheckState.Unchecked)
        panel.refresh_projects()
        self.assertEqual(panel._excluded_obs, {"74Ge/7961"})
        self.assertEqual(
            self._row(panel, "7961").checkState(),
            Qt.CheckState.Unchecked)
        self.assertEqual(
            self._row(panel, "7958").checkState(), Qt.CheckState.Checked)

    def test_exclusions_round_trip_through_the_save_file(self):
        from PySide6.QtCore import Qt
        panel = self._panel()
        self._row(panel, "7961").setCheckState(Qt.CheckState.Unchecked)
        d = panel.to_dict()
        self.assertEqual(d["excluded_observations"], ["74Ge/7961"])

        other = self._panel()
        other.from_dict(d)
        other.refresh_projects()
        self.assertEqual(other._excluded_obs, {"74Ge/7961"})
        self.assertEqual(
            self._row(other, "7961").checkState(),
            Qt.CheckState.Unchecked)

    def test_exclusions_are_stored_by_label_not_index(self):
        """Indices point at whatever run happens to sit there after a
        re-scan; labels survive reordering and reloading."""
        panel = self._panel()
        from PySide6.QtCore import Qt
        self._row(panel, "7961").setCheckState(Qt.CheckState.Unchecked)
        self.assertEqual(panel.to_dict()["excluded_observations"],
                         ["74Ge/7961"])

    def test_fit_refuses_when_too_few_remain_included(self):
        """The guard fires before PyMC is touched, so the user gets a
        sentence rather than a linear-algebra failure."""
        from cls_estimations.reference_correction import (
            ReferenceCorrector, ReferenceObservation)
        obs = [ReferenceObservation(t=0.0, centroid=-1.0, sigma=1.0),
               ReferenceObservation(t=1.0, centroid=-2.0, sigma=1.0,
                                    include=False),
               ReferenceObservation(t=2.0, centroid=-3.0, sigma=1.0,
                                    include=False)]
        with self.assertRaises(ValueError) as ctx:
            ReferenceCorrector(kernel="rbf").fit(obs)
        self.assertIn("included", str(ctx.exception))


class ExcludedPointsOnThePlotTests(unittest.TestCase):
    """An excluded run stays visible, drawn but not fitted.

    A point that simply disappears from the figure is indistinguish-
    able from a run that was never taken, and the reader cannot tell
    whether the curve ignored the outlier or never saw it.
    """

    _cached = None

    @classmethod
    def _corrector(cls):
        if cls._cached is None:
            from cls_estimations.reference_correction import (
                ReferenceCorrector, ReferenceObservation)
            obs = [ReferenceObservation(
                t=float(i), centroid=-130.0 + 2.0 * i, sigma=3.0,
                label=f"74Ge/run_{7900 + i}") for i in range(8)]
            # The T02 shape: one wild point, excluded.
            obs.insert(4, ReferenceObservation(
                t=3.5, centroid=85.0, sigma=3.5,
                label="74Ge/run_7961", include=False))
            rc = ReferenceCorrector(kernel="rbf", random_seed=0)
            rc.fit(obs)
            cls._cached = rc
        return cls._cached

    def test_excluded_points_are_carried_into_the_arrays(self):
        d = self._corrector().diagnostic_arrays(n_grid=32)
        self.assertEqual(len(d["t_excluded"]), 1)
        self.assertAlmostEqual(float(d["y_excluded"][0]), 85.0)
        self.assertAlmostEqual(float(d["yerr_excluded"][0]), 3.5)

    def test_excluded_points_are_not_in_the_training_set(self):
        d = self._corrector().diagnostic_arrays(n_grid=32)
        self.assertEqual(len(d["t_train"]), 8)
        self.assertNotIn(85.0, list(d["y_train"]))

    def test_excluded_time_is_on_the_same_axis_as_the_training_times(self):
        """Both are rebased on t0; an absolute epoch hour here would
        put the grey cross off the edge of the plot."""
        d = self._corrector().diagnostic_arrays(n_grid=32)
        t_ex = float(d["t_excluded"][0])
        self.assertGreater(t_ex, float(d["t_train"].min()) - 1.0)
        self.assertLess(t_ex, float(d["t_train"].max()) + 1.0)

    def test_the_curve_does_not_chase_the_excluded_point(self):
        """The whole point of excluding it. Near t=3.5 the MAP should
        stay on the -130..-116 trend, not detour toward +85."""
        rc = self._corrector()
        mu, _ = rc.predict(3.5 + rc._t0)
        self.assertLess(float(mu[0]), -100.0)

    def test_the_plot_draws_them(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cls_estimations.reference_correction import (
            MAP_LEGEND_EXCLUDED)
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        self._corrector().diagnostic_plot(ax=ax, n_grid=32)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn(MAP_LEGEND_EXCLUDED, labels)


class HoverTipsWithExclusionsTests(unittest.TestCase):
    """The tips zip the training arrays against the observations.

    Those were the same list until a run could be excluded. Now
    t_train holds only the included points, so zipping against every
    observation pairs each point with the wrong run from the first
    exclusion onward -- and zip() truncating means the last run
    loses its tip with no error at all. Arda uses these tips to find
    which dot is which; silently mislabelling them is worse than not
    having them.
    """

    def _panel_with_excluded_fit(self):
        from PySide6.QtWidgets import QWidget
        stub = QWidget()
        stub._projects = []
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        panel._corrector = ExcludedPointsOnThePlotTests._corrector()
        panel._show_corrected.setChecked(False)
        panel._render_diagnostic_plot()
        return panel

    def test_every_included_run_gets_a_tip(self):
        panel = self._panel_with_excluded_fit()
        tips = [t for *_, t in panel._hover_points]
        self.assertEqual(len(tips), 9)          # 8 fitted + 1 excluded

    def test_tips_name_the_run_that_is_actually_at_that_point(self):
        """The regression proper: check the label against the y value
        it is attached to, not merely that some label exists."""
        panel = self._panel_with_excluded_fit()
        for _ax, _x, y, text in panel._hover_points:
            head = text.split("\n")[0]
            label = head.replace("  (excluded)", "").replace(
                "  (off scale)", "")
            run = int(label.rsplit("_", 1)[1])
            if "off scale" in head:
                # Pinned to the frame edge: the tip carries the true
                # value, the marker does not.
                y = float(text.split("\n")[1].split(" MHz")[0])
            if run == 7961:
                self.assertAlmostEqual(y, 85.0, places=3)
            else:
                # y = -130 + 2*i for run 7900+i
                self.assertAlmostEqual(
                    y, -130.0 + 2.0 * (run - 7900), places=3)

    def test_the_excluded_point_says_so(self):
        panel = self._panel_with_excluded_fit()
        tips = [t for *_, t in panel._hover_points if "7961" in t]
        self.assertEqual(len(tips), 1)
        self.assertIn("excluded", tips[0])


class PlotViewControlTests(unittest.TestCase):
    """Time unit, hiding the excluded crosses, and the residual panel.

    All three are display choices over one fit -- none of them may
    touch the GP, so switching must never require a re-fit.
    """

    def _panel(self):
        from PySide6.QtWidgets import QWidget
        stub = QWidget()
        stub._projects = []
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        panel._corrector = ExcludedPointsOnThePlotTests._corrector()
        panel._show_corrected.setChecked(False)
        return panel

    # ── Time unit ──

    def test_hours_is_the_default(self):
        panel = self._panel()
        self.assertEqual(panel._t_unit.currentData(), 1.0)

    def test_minutes_scales_the_axis_and_relabels_it(self):
        panel = self._panel()
        panel._t_unit.setCurrentIndex(1)          # minutes
        ax = panel._figure.axes[0]
        self.assertIn("min", ax.get_xlabel())
        # The fit spans 0..7 h, so in minutes the data must reach
        # well past 7 -- a relabel without a rescale would not.
        self.assertGreater(max(x for _a, x, _y, _t
                               in panel._hover_points), 300.0)

    def test_switching_units_does_not_disturb_the_fit(self):
        """It is a display choice. If it re-fitted, or invalidated
        the corrector, the user would lose minutes for a label."""
        panel = self._panel()
        before = panel._corrector.hyperparameters.ell
        panel._t_unit.setCurrentIndex(1)
        panel._t_unit.setCurrentIndex(0)
        self.assertEqual(panel._corrector.hyperparameters.ell, before)

    def test_hover_tips_follow_the_axis_unit(self):
        """A tip reading '210.0 h' beside a minutes axis is worse
        than no tip."""
        panel = self._panel()
        panel._t_unit.setCurrentIndex(1)
        tips = [t for *_, t in panel._hover_points]
        self.assertTrue(any("min" in t for t in tips))
        self.assertFalse(any(t.endswith(" h") for t in tips))

    # ── Show excluded ──

    def test_excluded_crosses_show_by_default(self):
        panel = self._panel()
        panel._render_diagnostic_plot()
        from cls_estimations.reference_correction import (
            MAP_LEGEND_EXCLUDED)
        labels = [t.get_text()
                  for t in panel._figure.axes[0].get_legend().get_texts()]
        self.assertIn(MAP_LEGEND_EXCLUDED, labels)

    def test_hiding_them_drops_the_marker_and_its_legend_entry(self):
        from cls_estimations.reference_correction import (
            MAP_LEGEND_EXCLUDED)
        panel = self._panel()
        panel._show_excluded.setChecked(False)
        labels = [t.get_text()
                  for t in panel._figure.axes[0].get_legend().get_texts()]
        self.assertNotIn(MAP_LEGEND_EXCLUDED, labels)

    def test_hiding_them_drops_their_hover_tips_too(self):
        """A tip for an invisible point is a tip that fires over
        empty space."""
        panel = self._panel()
        with_them = len(panel._hover_points)
        panel._show_excluded.setChecked(False)
        self.assertEqual(len(panel._hover_points), with_them - 1)
        self.assertFalse(any("7961" in t
                             for *_, t in panel._hover_points))

    def test_hiding_them_does_not_change_the_fit(self):
        panel = self._panel()
        before = panel._corrector.hyperparameters.ell
        panel._show_excluded.setChecked(False)
        self.assertEqual(panel._corrector.hyperparameters.ell, before)

    # ── Residuals ──

    def test_residual_panel_is_off_by_default(self):
        panel = self._panel()
        panel._render_diagnostic_plot()
        self.assertEqual(len(panel._figure.axes), 1)

    def test_residual_panel_adds_an_axis_labelled_in_sigma(self):
        panel = self._panel()
        panel._show_residuals.setChecked(True)
        self.assertEqual(len(panel._figure.axes), 2)
        self.assertIn("σ", panel._figure.axes[1].get_ylabel())

    def test_residuals_use_measurement_error_plus_the_noise_floor(self):
        """(y-mu)/sqrt(sigma_obs^2 + sigma_n^2).

        The GP's PREDICTIVE sd must not appear. Including it shrinks
        every residual by crediting the point with the mean
        function's uncertainty, which is not what a residual asks --
        and it is not what the reference pipeline this panel is read
        against does. Its residual file states the definition
        outright: "observed minus fitted GP / reference measurement
        and extra-noise sigma". With the predictive sd in, DENIS read
        +0.68 where that pipeline read +0.93 on the same run.
        """
        import numpy as np
        rc = ExcludedPointsOnThePlotTests._corrector()
        t, r = rc.residuals()
        y = rc._train_y_centered + rc._y_mean
        mu, sd = rc.predict(t + rc._t0)
        sigma_n = rc.hyperparameters.sigma_n
        expect = (y - mu) / np.sqrt(rc._train_yerr ** 2 + sigma_n ** 2)
        np.testing.assert_allclose(r, expect, rtol=1e-9)
        # The wrong denominator has to be genuinely different, or
        # this test would pass under either definition.
        with_pred_sd = (y - mu) / np.sqrt(rc._train_yerr ** 2 + sd ** 2)
        self.assertFalse(np.allclose(r, with_pred_sd, rtol=1e-6))

    def test_residuals_cover_only_the_fitted_points(self):
        rc = ExcludedPointsOnThePlotTests._corrector()
        t, r = rc.residuals()
        self.assertEqual(len(r), 8)               # 9 obs, 1 excluded

    def test_every_panel_shares_one_x_label_at_the_bottom(self):
        """Three stacked panels each captioned 'Timestamp' wastes the
        vertical space the plots need."""
        panel = self._panel()
        panel._show_residuals.setChecked(True)
        axes = panel._figure.axes
        self.assertEqual(axes[0].get_xlabel(), "")
        self.assertIn("Timestamp", axes[-1].get_xlabel())


class RefitNagTests(unittest.TestCase):
    """The "re-fit the GP" hint has to mean stale, not 'something is
    excluded'. Keyed off the latter it stayed up after a perfectly
    good re-fit, which teaches the reader to ignore it."""

    def _panel(self):
        from PySide6.QtWidgets import QWidget
        stub = QWidget()
        stub._projects = [ExcludeReferenceRunTests._RefProject("74Ge", [
            ExcludeReferenceRunTests._obs(7958, 100.0, -137.9),
            ExcludeReferenceRunTests._obs(7961, 100.8, 85.0),
        ])]
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        panel.refresh_projects()
        return panel

    def test_no_nag_before_any_fit(self):
        from PySide6.QtCore import Qt
        panel = self._panel()
        for i in range(panel._obs_list.count()):
            panel._obs_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.assertNotIn("re-fit", panel._obs_lbl.text())

    def test_nag_appears_when_ticks_diverge_from_the_fit(self):
        from PySide6.QtCore import Qt
        panel = self._panel()
        panel._corrector = ExcludedPointsOnThePlotTests._corrector()
        panel._fitted_excluded = set()            # fitted with all in
        panel._obs_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        self.assertIn("re-fit", panel._obs_lbl.text())

    def test_nag_clears_once_the_fit_matches_the_ticks(self):
        from PySide6.QtCore import Qt
        panel = self._panel()
        panel._corrector = ExcludedPointsOnThePlotTests._corrector()
        # Untick through the widget, as the user does: the label
        # counts what is on screen, so poking _excluded_obs alone
        # would test a state the panel never actually reaches.
        for i in range(panel._obs_list.count()):
            it = panel._obs_list.item(i)
            if str(it.data(Qt.ItemDataRole.UserRole)).endswith("7961"):
                it.setCheckState(Qt.CheckState.Unchecked)
        panel._fitted_excluded = set(panel._excluded_obs)
        panel._update_obs_label()
        self.assertIn("1 excluded", panel._obs_lbl.text())
        self.assertNotIn("re-fit", panel._obs_lbl.text())


class GPReferenceOptionTests(unittest.TestCase):
    """The Runs selector offers the GP curve as the reference.

    With the correction on, every sample centroid is ALREADY a shift
    against the drift curve -- it has had the curve subtracted over
    its own acquisition window. So the reference is that curve, and
    it sits at zero. Offering only "a specific run" or "weighted
    average" forced the reference to be a measurement, which takes
    the reference level off a second time and makes every shift
    depend on which run was picked (58.6 MHz of spread across the 13
    74Ge runs on the Ge set).
    """

    def _tab(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        at._add_project("74Ge")
        ist = at._is_tab
        ist._refresh_projects()
        if not ist._entries:          # no fit results to scan
            ist._add_row(project_name="74Ge", label="74Ge", A=74)
        return ist

    def test_the_option_is_offered(self):
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        ist = self._tab()
        self.assertTrue(ist._entries)
        combo = ist._entries[0].run_combo
        opts = [combo.itemText(i) for i in range(combo.count())]
        self.assertIn(GP_REFERENCE, opts)
        self.assertIn("Weighted Average", opts)

    def test_it_survives_a_project_change(self):
        """Switching project repopulates the run list; a choice that
        is still on offer must not be silently reset."""
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        ist = self._tab()
        e = ist._entries[0]
        e.run_combo.setCurrentText(GP_REFERENCE)
        ist._on_project_changed(e)
        self.assertEqual(e.run_combo.currentText(), GP_REFERENCE)

    def test_the_row_reads_zero(self):
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        ist = self._tab()
        e = ist._entries[0]
        e.run_combo.setCurrentText(GP_REFERENCE)
        ist._update_centroids()
        self.assertEqual(e.centroid_item.text(), "0.0000")
        self.assertEqual(e.error_item.text(), "0.0000")

    def test_it_round_trips_through_the_save_file(self):
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        ist = self._tab()
        ist._entries[0].run_combo.setCurrentText(GP_REFERENCE)
        d = ist.to_dict()
        found = [ed.get("run_selection") for ed in d.get("entries", [])]
        self.assertIn(GP_REFERENCE, found)


class CentroidDiagnosticTests(unittest.TestCase):
    """The Centroids tab: how each isotope shift was assembled.

    The numbers are computed from the CORRECTOR, over each run's own
    acquisition window -- not read back from the fit. That is what
    lets the tab answer "what does the drift model say about this
    measurement?" for a project whose fits predate the correction,
    and for a reference row set to GP drift model, which has no runs
    of its own.

    A single-run isotope gets one line; a merge spanning hours gets
    one line per constituent, each with its own reference estimate,
    plus the count-weighted combination -- because two runs taken
    hours apart cannot share one reference number.
    """

    T0_H = 1783955109.0 / 3600.0        # first reference measurement

    def _gp(self):
        """A GP on a steep ramp, anchored at the Ge campaign's t0."""
        import numpy as np
        from cls_estimations.reference_correction import (
            ReferenceCorrector, ReferenceObservation)
        t = self.T0_H + np.linspace(0.0, 6.0, 14)
        y = -104.0 - 13.5 * (t - self.T0_H)
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit([ReferenceObservation(t=ti, centroid=yi, sigma=3.0)
                for ti, yi in zip(t, y)])
        return rc

    def _tab(self, *, with_gp=True):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        ist = at._is_tab
        if with_gp:
            ist._ref_corr_panel._corrector = self._gp()
        return ist

    @staticmethod
    def _entry(label, project, run_sel="Weighted Average"):
        class _Txt:
            def __init__(self, v):
                self.v = v

            def text(self):
                return self.v

            def currentText(self):
                return self.v

        class _Spin:
            def value(self):
                return 0

        e = type("E", (), {})()
        e.label_edit = _Txt(label)
        e.project_combo = _Txt(project)
        e.run_combo = _Txt(run_sel)
        e.a_spin = _Spin()
        return e

    _SINGLE = {
        "success": True, "run_number": "7897", "run_file": "C:/d/7897.asdf",
        "run_metadata": {"ts_start": 1783958919.0, "ts_stop": 1783959823.0}}

    _MERGED = {
        "success": True, "run_number": "merged_7891_7901",
        "run_file": "merged://merged_7891_7901",
        "run_metadata": {"ts_start": 1783955946.0,
                         "ts_stop": 1783963976.0}}

    _MERGED_ENTRY = {
        "is_merged": True,
        "merged_data": {
            "merged_name": "merged_7891_7901",
            "per_run": [
                {"run_num": "7891", "ts_start": 1783955946.0,
                 "ts_stop": 1783956247.0, "n_events": 981},
                {"run_num": "7901", "ts_start": 1783963214.0,
                 "ts_stop": 1783963976.0, "n_events": 2439}]}}

    def _build(self, ist, label, results, centroid,
               merged_entries=(), run_sel="Weighted Average"):
        project = type("P", (), {})()
        project.project_name = label
        project._last_results = results
        src = type("S", (), {})()
        src._file_entries = list(merged_entries)
        ist._find_project = lambda n, _p=project: _p
        ist._get_source_block = lambda p, _s=src: _s
        ist._build_centroid_diagnostic(
            [self._entry(label, label, run_sel)],
            {label: {"centroid": centroid}})
        return ist._centroid_rows

    # -- single run ------------------------------------------------
    def test_single_run_shows_its_window_and_reference(self):
        ist = self._tab()
        rows = self._build(ist, "76Ge", [self._SINGLE], -99.045)
        self.assertEqual(len(rows), 2)              # the run + combined
        label, run, kind, t0, dur, w, mu, sd, applied, corrected = rows[0]
        self.assertEqual((run, kind), ("7897", "single"))
        self.assertAlmostEqual(dur, 904.0)          # its acquisition
        self.assertIsNotNone(mu)
        self.assertGreater(sd, 0.0)
        self.assertIsNone(corrected)                # on the combined row
        self.assertEqual(rows[1][1], "combined")
        self.assertAlmostEqual(rows[1][9], -99.045)

    def test_times_are_relative_to_the_first_reference_measurement(self):
        """Not to the isotope's own first run: every row has to share
        the origin the drift plot uses, or they cannot be compared
        across isotopes."""
        ist = self._tab()
        rows = self._build(ist, "76Ge", [self._SINGLE], -99.045)
        expected = (1783958919.0 - 1783955109.0) / 3600.0
        self.assertAlmostEqual(rows[0][3], expected, places=4)

    # -- merged ----------------------------------------------------
    def test_merged_runs_get_one_reference_estimate_each(self):
        ist = self._tab()
        rows = self._build(ist, "70Ge", [self._MERGED], -62.674,
                           merged_entries=[self._MERGED_ENTRY])
        self.assertEqual([r[1] for r in rows],
                         ["7891", "7901", "combined"])
        self.assertEqual(rows[0][2], "merged")
        self.assertAlmostEqual(rows[0][4], 301.0)   # own windows
        self.assertAlmostEqual(rows[1][4], 762.0)
        # Two hours apart on a 13.5 MHz/h ramp: the two references
        # must differ substantially.
        self.assertGreater(abs(rows[1][6] - rows[0][6]), 20.0)

    def test_a_reloaded_merge_is_still_split(self):
        """Results loaded from disk carry run_file='' -- the merged
        identity survives only in run_number. Keying the lookup on
        run_file alone reported every merge in a REOPENED project as
        one "single" run spanning the whole merge window, with a
        single reference for two runs taken hours apart."""
        ist = self._tab()
        from_disk = dict(self._MERGED)
        from_disk["run_file"] = ""          # what a reload looks like
        rows = self._build(ist, "70Ge", [from_disk], -62.674,
                           merged_entries=[self._MERGED_ENTRY])
        self.assertEqual([r[1] for r in rows],
                         ["7891", "7901", "combined"])
        self.assertEqual(rows[0][2], "merged")
        # Not the 8030 s aggregate window of the whole merge.
        self.assertAlmostEqual(rows[0][4], 301.0)

    def test_constituents_are_weighted_by_counts(self):
        ist = self._tab()
        rows = self._build(ist, "70Ge", [self._MERGED], -62.674,
                           merged_entries=[self._MERGED_ENTRY])
        self.assertAlmostEqual(rows[0][5], 981 / (981 + 2439), places=4)
        self.assertAlmostEqual(rows[1][5], 2439 / (981 + 2439), places=4)

    def test_combined_matches_the_weighted_arithmetic(self):
        ist = self._tab()
        rows = self._build(ist, "70Ge", [self._MERGED], -62.674,
                           merged_entries=[self._MERGED_ENTRY])
        w0, w1 = rows[0][5], rows[1][5]
        mu = w0 * rows[0][6] + w1 * rows[1][6]
        sd = ((rows[0][7] * w0) ** 2 + (rows[1][7] * w1) ** 2) ** 0.5
        self.assertAlmostEqual(rows[2][6], mu, places=6)
        self.assertAlmostEqual(rows[2][7], sd, places=6)
        self.assertEqual(rows[2][2], "weighted")

    def test_a_merged_reference_lies_between_its_constituents(self):
        ist = self._tab()
        rows = self._build(ist, "70Ge", [self._MERGED], -62.674,
                           merged_entries=[self._MERGED_ENTRY])
        lo, hi = sorted((rows[0][6], rows[1][6]))
        self.assertGreater(rows[2][6], lo)
        self.assertLess(rows[2][6], hi)

    # -- the cases that had nothing to show before ------------------
    def test_gp_reference_row_says_so(self):
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        ist = self._tab()
        rows = self._build(ist, "74Ge", [], 0.0, run_sel=GP_REFERENCE)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], GP_REFERENCE)
        self.assertEqual(rows[0][2], "reference")
        self.assertAlmostEqual(rows[0][9], 0.0)

    def test_without_a_gp_it_says_so_instead_of_showing_nothing(self):
        ist = self._tab(with_gp=False)
        rows = self._build(ist, "76Ge", [self._SINGLE], -99.045)
        self.assertEqual(rows[0][1], "(no GP fitted)")

    def test_a_stale_fit_is_flagged(self):
        """A fit that subtracted something else is shown in Applied
        so the mismatch is visible rather than believed."""
        ist = self._tab()
        stale = dict(self._SINGLE)
        stale["run_metadata"] = dict(self._SINGLE["run_metadata"])
        stale["run_metadata"].update({
            "centroid_correction_applied": True,
            "centroid_correction_mhz": -999.0})
        rows = self._build(ist, "76Ge", [stale], -99.045)
        self.assertAlmostEqual(rows[0][8], -999.0)
        self.assertNotAlmostEqual(rows[0][6], -999.0, places=1)

    def test_the_table_renders_every_row(self):
        ist = self._tab()
        self._build(ist, "70Ge", [self._MERGED], -62.674,
                    merged_entries=[self._MERGED_ENTRY])
        t = ist._centroid_table
        self.assertEqual(t.rowCount(), 3)
        self.assertEqual(t.item(0, 1).text(), "7891")
        self.assertEqual(t.item(2, 1).text(), "combined")
        self.assertTrue(t.item(2, 1).font().bold())
        self.assertFalse(t.item(0, 1).font().bold())

    def test_the_empty_tab_explains_itself(self):
        ist = self._tab()
        ist._centroid_rows = []
        ist._populate_centroid_table()
        self.assertEqual(ist._centroid_table.rowCount(), 1)
        self.assertIn("Compute Shifts",
                      ist._centroid_table.item(0, 0).text())


class PresentationTests(unittest.TestCase):
    def test_kernel_key_never_shown_raw(self):
        """'thesis' is the on-disk key for the composite kernel; it
        means nothing to a reader of the status line."""
        self.assertEqual(_kernel_name("thesis"), "Composite")
        self.assertEqual(_kernel_name("rbf"), "RBF")
        # An unknown key still renders something rather than crashing.
        self.assertEqual(_kernel_name("zzz"), "zzz")

    def test_long_tooltips_are_wrapped_into_a_block(self):
        tip = ("How large the correction is compared with its own "
               "uncertainty. Below 2 the drift is barely resolved; the "
               "row turns amber above 2 and red above 3.")
        out = _wrap_tip(tip, width=54)
        self.assertIn("\n", out)
        self.assertTrue(all(len(line) <= 54 for line in out.split("\n")))
        # No words lost or reordered by the wrap.
        self.assertEqual(out.split(), tip.split())

    def test_wrap_preserves_existing_paragraph_breaks(self):
        out = _wrap_tip("first para\nsecond para", width=54)
        self.assertEqual(out, "first para\nsecond para")


class ReferenceFrameTests(unittest.TestCase):
    """The reference isotope has to sit in the corrected frame too.

    ``predict()`` returns the ABSOLUTE reference centroid, and the fit
    subtracts it, so a corrected sample centroid is already
    nu_A - nu_ref(t_A) -- the shift against the reference at that
    sample's own moment. Leaving the reference project on its raw
    scale made ``compute_all_shifts``' ``dnu = c - ref_c`` take the
    reference level off a second time, and made every isotope shift
    depend on WHICH reference run was ticked as Ref (~45 MHz across
    the Ge campaign). It also starved the cross-covariance term,
    which only fires when the reference has Auto-corrected runs, so
    every sigma_dnu was overestimated.
    """

    def test_gp_trains_on_raw_centroids_after_the_reference_is_corrected(self):
        """The loop has to be idempotent: correcting the reference
        makes its next fit report residuals, and re-training on those
        would collapse the GP onto a flat zero and silently switch
        the correction off."""
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("74Ge", is_reference=True)
        p._last_results = [_ref_result(-152.3, corrected_by=-152.803)]
        obs = p.get_reference_observations()
        self.assertEqual(len(obs), 1)
        # Exactly what the fit subtracted, added back.
        self.assertAlmostEqual(obs[0]["centroid_mhz"], -305.103, places=3)

    def test_uncorrected_reference_run_passes_through_untouched(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("74Ge", is_reference=True)
        p._last_results = [_ref_result(-152.3, corrected_by=None)]
        self.assertAlmostEqual(
            p.get_reference_observations()[0]["centroid_mhz"], -152.3)

    def test_a_zero_correction_flag_is_still_honoured(self):
        """applied=True with 0.0 MHz is a real state (the GP predicted
        no drift at that instant); it must not be confused with
        'never corrected'."""
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("74Ge", is_reference=True)
        p._last_results = [_ref_result(-152.3, corrected_by=0.0)]
        self.assertAlmostEqual(
            p.get_reference_observations()[0]["centroid_mhz"], -152.3)

    def test_compute_covers_checked_reference_projects(self):
        """Source-level check: _collect_corrections is handed the
        union of ticked sample AND reference projects, and no longer
        skips a project for being the reference."""
        import inspect
        src = inspect.getsource(
            ReferenceCorrectionPanel._compute_corrections_clicked)
        self.assertIn("self._checked_names(self._ref_list)", src)
        self.assertIn("_collect_corrections(wanted)", src)
        collect = inspect.getsource(
            ReferenceCorrectionPanel._collect_corrections)
        self.assertNotIn("p.is_reference", collect)


class CorrectedCentroidPanelTests(unittest.TestCase):
    """The lower panel of the two-panel figure: every isotope in the
    drift-free frame. The reference should scatter about zero; a
    reference still tracking the drift curve is the visible symptom
    of the bug above."""

    def test_no_corrector_means_no_lower_panel(self):
        from PySide6.QtWidgets import QWidget
        stub = QWidget()
        stub._projects = []
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        self.assertEqual(panel._corrected_centroids(), {})

    class _Corr:
        """A GP whose drift is known exactly: G(t) = -150 + 10 (t-100),
        t in hours. Runs here have no stop time, so the window average
        is the value at the start."""
        is_fit = True
        _t0 = 100.0

        @staticmethod
        def predict_interval(t1, t2, n=32):
            return -150.0 + 10.0 * (0.5 * (t1 + t2) - 100.0), 0.5

    def _panel(self, projects):
        from PySide6.QtWidgets import QWidget
        stub = QWidget()
        stub._projects = projects
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        panel._corrector = self._Corr()
        return panel

    class _P:
        def __init__(self, name, results):
            self.project_name = name
            self._last_results = results

    def test_series_are_grouped_per_project_and_rebased_on_the_gp(self):
        # Raw centroids that ride on the drift: 74Ge sits 1 above and
        # 2 below the curve, 70Ge 85 below it.
        s = self._panel([
            self._P("74Ge", [_centroid_result(-139.0, 3600.0 * 101.0),
                             _centroid_result(-132.0, 3600.0 * 102.0)]),
            self._P("70Ge", [_centroid_result(-220.0, 3600.0 * 101.5)]),
        ])._corrected_centroids()
        self.assertEqual(sorted(s), ["70Ge", "74Ge"])
        # Times are rebased on the GP's own t0, so every panel shares
        # one axis.
        self.assertEqual(s["74Ge"]["t"], [1.0, 2.0])
        self.assertEqual(s["70Ge"]["t"], [1.5])
        self.assertEqual(s["74Ge"]["y"], [1.0, -2.0])
        self.assertEqual(s["70Ge"]["y"], [-85.0])

    def test_an_uncorrected_fit_is_corrected_not_shown_raw(self):
        """The regression. The panel used to read params_df and call it
        corrected; on T02 none of the 29 runs had been corrected at
        fit time, so the 'Corrected centroid' panel plotted raw
        centroids -- 74Ge at -150 instead of about 0."""
        s = self._panel([self._P("74Ge", [
            _centroid_result(-139.0, 3600.0 * 101.0)])])._corrected_centroids()
        d = s["74Ge"]
        self.assertEqual(d["raw"], [-139.0])
        self.assertEqual(d["g"], [-140.0])
        self.assertEqual(d["y"], [1.0])
        self.assertEqual(d["fly"], [True])

    def test_a_fit_time_correction_is_honoured_not_repeated(self):
        """A run the fit already corrected keeps its centroid. The
        stub GP would say -140 here; subtracting it again would put
        the point 140 MHz off."""
        r = _centroid_result(1.0, 3600.0 * 101.0)
        r["run_metadata"].update({"centroid_correction_applied": True,
                                  "centroid_correction_mhz": -140.0})
        d = self._panel([self._P("74Ge", [r])])._corrected_centroids()["74Ge"]
        self.assertEqual(d["y"], [1.0])
        self.assertEqual(d["raw"], [-139.0])
        self.assertEqual(d["fly"], [False])

    def test_runs_without_a_timestamp_are_dropped(self):
        s = self._panel([self._P("74Ge", [
            _centroid_result(1.0, 0.0),
            _centroid_result(-128.0, 3600.0 * 102.0)])])._corrected_centroids()
        self.assertEqual(s["74Ge"]["y"], [2.0])


def _ref_result(centroid, *, corrected_by):
    meta = {"ts_start": 1783965428.62}
    if corrected_by is not None:
        meta["centroid_correction_applied"] = True
        meta["centroid_correction_mhz"] = corrected_by
    return {
        "success": True, "run_number": "1",
        "run_file": "C:/d/run_1.asdf", "source_name": "Run_1",
        "params_df": {"Source": ["Run_1"], "Model": ["m"],
                      "Parameter": ["centroid"],
                      "Value": [centroid], "Stderr": [4.0]},
        "run_metadata": meta,
    }


def _centroid_result(centroid, ts_start):
    return {
        "success": True, "run_number": "1", "run_file": "",
        "source_name": "Run_1",
        "params_df": {"Source": ["Run_1"], "Model": ["m"],
                      "Parameter": ["centroid"],
                      "Value": [centroid], "Stderr": [3.0]},
        "run_metadata": {"ts_start": ts_start},
    }


class FitBusyIndicatorTests(unittest.TestCase):
    """A GP fit runs off the GUI thread; without a moving indicator
    the only sign of life was a static status line."""

    def _panel(self):
        from PySide6.QtWidgets import QWidget
        stub = QWidget()
        stub._projects = []
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        return panel

    def test_bar_and_clock_follow_busy_state(self):
        panel = self._panel()
        panel._status.setText("Fitting GP (RBF kernel, 13 obs)...")
        panel._set_busy(True)
        self.assertTrue(panel._fit_progress.isVisibleTo(panel))
        self.assertTrue(panel._fit_tick.isActive())
        # Indeterminate: find_MAP reports no iteration count, so a
        # percentage would be a fiction.
        self.assertEqual(
            (panel._fit_progress.minimum(), panel._fit_progress.maximum()),
            (0, 0))
        panel._set_busy(False)
        self.assertFalse(panel._fit_progress.isVisibleTo(panel))
        self.assertFalse(panel._fit_tick.isActive())

    def test_elapsed_counter_appends_to_the_fit_status(self):
        panel = self._panel()
        base = "Fitting GP (RBF kernel, 13 obs)..."
        panel._status.setText(base)
        panel._set_busy(True)
        panel._update_fit_elapsed()
        self.assertTrue(panel._status.text().startswith(base))
        self.assertRegex(panel._status.text(), r"\[\d+:\d\d\]$")
        # Repeated ticks must not stack counters on each other.
        panel._update_fit_elapsed()
        panel._update_fit_elapsed()
        self.assertEqual(panel._status.text().count("["), 1)
        panel._set_busy(False)


def _write_asdf(directory, name, ts_start):
    """Minimal ASDF with the raw event table the reader looks at.

    Column 0 of ``raw`` is the event timestamp; clstools' ``TSstart``
    is its minimum, and events are written in acquisition order.
    """
    import asdf
    path = os.path.join(directory, name)
    raw = np.array(
        [[ts_start + i, -185.0, 15.0, 3.0, 65.0, 2.99] for i in range(4)],
        dtype=float)
    tree = {
        "Run": name.split("_")[-1].split(".")[0],
        "raw": raw,
        "raw_header": ["timestamp", "voltage", "bunch_number",
                       "channel", "time", "cooler"],
    }
    asdf.AsdfFile(tree).write_to(path)
    return path


if __name__ == "__main__":
    unittest.main()
