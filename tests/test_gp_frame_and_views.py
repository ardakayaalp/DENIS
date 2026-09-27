"""Drift-free frame, GP-mode isotope shifts, GP plot views, IS layout.

2026-09-22. Arda's T02 fits were never corrected at fit time -- all 29
runs carry centroid_correction_applied = False -- and three readers
assumed they had been:

* the GP panel's "Corrected centroid" plot, which showed raw values;
* Isotope Shifts in GP mode, which set the reference to 0 and took
  raw sample centroids, putting every shift off by the drift level;
* the Isotope Shifts plot, which drew raw spectra and left the GP
  reference panel empty.

gui/analysis/gp_frame.py is now the one place a result is put into the
drift-free frame. It corrects a run the fit did not, and never
corrects one twice.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_gp_frame_and_views.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_APP = QApplication.instance() or QApplication([])


# ── Fixtures ──────────────────────────────────────────────────

class _LinearGP:
    """A 'GP' with a known drift, G(t) = -150 + 10 (t - 100) MHz, t in
    hours, so every expected value can be worked out by hand. The
    window average of a linear function is its value at the midpoint."""
    is_fit = True
    _t0 = 100.0

    @staticmethod
    def g(t):
        return -150.0 + 10.0 * (t - 100.0)

    def predict_interval(self, t1, t2, n=32):
        return self.g(0.5 * (t1 + t2)), 0.5


def _result(centroid, ts_start, *, run="1", ts_stop=0.0, err=3.0,
            applied=None, run_file=""):
    meta = {"ts_start": ts_start, "ts_stop": ts_stop}
    if applied is not None:
        meta.update({"centroid_correction_applied": True,
                     "centroid_correction_mhz": applied,
                     "centroid_correction_mode": "Auto"})
    return {
        "success": True, "run_number": run, "run_file": run_file,
        "source_name": f"Run_{run}",
        "params_df": {"Source": [f"Run_{run}"], "Model": ["m"],
                      "Parameter": ["centroid"],
                      "Value": [centroid], "Stderr": [err]},
        "run_metadata": meta,
    }


_REAL_GP = None


def _real_gp():
    """A real RBF fit on eight points with one exclusion, cached --
    for the paths that need a genuine posterior (propagation)."""
    global _REAL_GP
    if _REAL_GP is None:
        from cls_estimations.reference_correction import (
            ReferenceCorrector, ReferenceObservation)
        obs = [ReferenceObservation(
            t=float(i), centroid=-150.0 + 2.0 * i, sigma=3.0,
            label=f"74Ge/{7900 + i}") for i in range(8)]
        obs.insert(4, ReferenceObservation(
            t=3.5, centroid=85.0, sigma=3.5, label="74Ge/7961",
            include=False))
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        _REAL_GP = rc
    return _REAL_GP


# ── gp_frame ──────────────────────────────────────────────────

class FrameCentroidTests(unittest.TestCase):

    def test_an_uncorrected_run_is_corrected_by_its_window_average(self):
        from gui.analysis.gp_frame import frame_centroid
        r = _result(-139.0, 3600.0 * 101.0, ts_stop=3600.0 * 103.0)
        fr = frame_centroid(_LinearGP(), None, r, -139.0)
        # window 101..103 h -> midpoint 102 h -> G = -130
        self.assertAlmostEqual(fr["g"], -130.0)
        self.assertAlmostEqual(fr["corrected"], -9.0)
        self.assertAlmostEqual(fr["shift"], -130.0)
        self.assertEqual(fr["raw"], -139.0)
        self.assertTrue(fr["on_the_fly"])

    def test_a_fit_time_correction_is_never_applied_twice(self):
        from gui.analysis.gp_frame import frame_centroid
        r = _result(1.0, 3600.0 * 101.0, applied=-140.0)
        fr = frame_centroid(_LinearGP(), None, r, 1.0)
        self.assertEqual(fr["corrected"], 1.0)
        self.assertEqual(fr["shift"], 0.0)
        self.assertEqual(fr["raw"], -139.0)
        self.assertFalse(fr["on_the_fly"])

    def test_no_gp_means_no_corrected_value(self):
        from gui.analysis.gp_frame import frame_centroid
        fr = frame_centroid(None, None, _result(1.0, 3600.0), 1.0)
        self.assertIsNone(fr["corrected"])

    def test_a_merge_is_corrected_by_the_count_weighted_level(self):
        """The merged centroid is a count-weighted blend of its
        constituents' lines, so it is corrected by the same blend."""
        from gui.analysis import gp_frame

        class _Src:
            _file_entries = [{
                "is_merged": True,
                "merged_data": {"merged_name": "m1", "per_run": [
                    {"run_num": "a", "ts_start": 3600.0 * 100,
                     "ts_stop": 0, "n_events": 1000},
                    {"run_num": "b", "ts_start": 3600.0 * 104,
                     "ts_stop": 0, "n_events": 3000}]}}]

        r = _result(-100.0, 3600.0 * 100, run="m1")
        lvl = gp_frame.drift_level(_LinearGP(), None, r, _Src())
        # G(100) = -150, G(104) = -110; weights 1:3 -> -120
        self.assertAlmostEqual(lvl["g"], -120.0)
        self.assertTrue(lvl["merged"])

    def test_effective_metadata_puts_the_run_in_the_propagation(self):
        """Isotope Shifts only propagates the GP's uncertainty for runs
        flagged as corrected. A run corrected on the fly must carry
        that flag, or its shift gets the GP's value without its
        error."""
        from gui.analysis.gp_frame import (
            effective_run_metadata, frame_centroid)
        r = _result(-139.0, 3600.0 * 101.0)
        rm = effective_run_metadata(r, frame_centroid(
            _LinearGP(), None, r, -139.0))
        self.assertTrue(rm["centroid_correction_applied"])
        self.assertEqual(rm["centroid_correction_mode"], "Auto")
        self.assertEqual(rm["ts_start"], 3600.0 * 101.0)
        self.assertTrue(rm["centroid_correction_on_the_fly"])
        # ...and the original is not mutated.
        self.assertNotIn("centroid_correction_applied", r["run_metadata"])

    def test_a_fit_time_corrected_run_keeps_its_own_metadata(self):
        from gui.analysis.gp_frame import (
            effective_run_metadata, frame_centroid)
        r = _result(1.0, 3600.0 * 101.0, applied=-140.0)
        rm = effective_run_metadata(r, frame_centroid(
            _LinearGP(), None, r, 1.0))
        self.assertEqual(rm, r["run_metadata"])

    def test_a_saved_exclusion_survives_reopening_the_file(self):
        """The T02 case: saved as '74Ge_T02/run_7961.asdf' from an
        in-session fit, matched against '74Ge_T02/7961' after the
        results came back from disk."""
        from gui.analysis.gp_frame import canonical_obs_key
        for old in ("74Ge_T02/run_7961.asdf", "74Ge_T02/run_7961",
                    "74Ge_T02/7961", "74Ge_T02/RUN-7961.ASDF"):
            with self.subTest(old=old):
                self.assertEqual(canonical_obs_key(old), "74Ge_T02/7961")
        # A merged name is not a run file and is left alone.
        self.assertEqual(canonical_obs_key("74Ge_T02/merged_1_2"),
                         "74Ge_T02/merged_1_2")

    def test_the_exclusion_key_is_built_one_way_everywhere(self):
        """The GP panel keys exclusions on '<project>/<label>'. The
        IS plot guessed 'run_<N>' and would never have matched."""
        import inspect
        from gui.analysis import project
        from gui.analysis.gp_frame import observation_key, observation_label
        # The run NUMBER, whether or not the result has a file: a run
        # fitted in session has one, the same run reloaded from disk
        # does not, and a file-based label changed across the reload.
        self.assertEqual(observation_label(
            {"run_file": "C:/d/run_7961.asdf", "run_number": "7961"}),
            "7961")
        self.assertEqual(observation_label(
            {"run_file": "", "run_number": "7961"}), "7961")

        class _P:
            project_name = "74Ge_T02"
        self.assertEqual(observation_key(_P(), {"run_number": "7961"}),
                         "74Ge_T02/7961")
        self.assertIn("observation_label", inspect.getsource(
            project.AnalysisProject.get_reference_observations))


class PostBinningShiftTests(unittest.TestCase):
    """Why correcting afterwards is exact for a single run.

    DENIS applies the fit-time correction after binning, as a constant
    subtraction from the x-axis. The bin contents are unchanged, and a
    line model depends on x only through (x - centroid), so the fit
    lands on the same minimum with the centroid moved by exactly the
    shift and the same uncertainty. Checked on a real least-squares
    fit rather than asserted."""

    def test_centroid_moves_by_exactly_the_shift(self):
        from scipy.optimize import curve_fit

        def model(x, c, a, w, b):
            return a * w ** 2 / ((x - c) ** 2 + w ** 2) + b

        rng = np.random.default_rng(3)
        x = np.linspace(-600, 200, 161)
        y = model(x, -181.1, 140.0, 60.0, 8.0)
        y = y + rng.normal(0, np.sqrt(y))
        err = np.sqrt(np.maximum(y, 1.0))
        g = -150.0
        # Starting values given in each frame, as a user would.
        p_raw, c_raw = curve_fit(model, x, y, sigma=err,
                                 p0=[-180, 130, 50, 5])
        p_cor, c_cor = curve_fit(model, x - g, y, sigma=err,
                                 p0=[-180 - g, 130, 50, 5])
        self.assertAlmostEqual(p_cor[0], p_raw[0] - g, places=5)
        self.assertAlmostEqual(np.sqrt(c_cor[0, 0]),
                               np.sqrt(c_raw[0, 0]), places=5)


# ── Isotope Shifts in GP mode ─────────────────────────────────

class _ISFixture(unittest.TestCase):
    """An AnalysisTab with a reference and one sample project holding
    injected fit results, and the real GP loaded in the panel."""

    def _tab(self, sample_results, ref_results=None):
        from gui.analysis.tab import AnalysisTab
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        ref = at._add_project("74Ge", is_reference=True)
        smp = at._add_project("70Ge")
        ref._last_results = ref_results or [
            _result(-150.0 + 2.0 * i, 3600.0 * i, run=str(7900 + i),
                    err=2.0 + 0.1 * i) for i in range(8)]
        smp._last_results = sample_results
        ist = at._is_tab
        ist._ref_corr_panel._corrector = _real_gp()
        ist._refresh_projects()
        rows = {e.project_combo.currentText(): e for e in ist._entries}
        for name, a in (("74Ge", 74), ("70Ge", 70)):
            if name not in rows:
                ist._add_row(project_name=name, label=name, A=a)
        rows = {e.project_combo.currentText(): e for e in ist._entries}
        rows["74Ge"].ref_radio.setChecked(True)
        rows["74Ge"].run_combo.setCurrentText(GP_REFERENCE)
        for e in ist._entries:
            e.select_cb.setChecked(
                e.project_combo.currentText() in ("74Ge", "70Ge"))
        return ist, rows


class GPModeTests(_ISFixture):

    def _level(self, t_h):
        g, _ = _real_gp().predict_interval(t_h, t_h)
        return float(g)

    def test_the_sample_cell_shows_the_corrected_centroid(self):
        """A raw centroid beside a reference of 0 is how a 150 MHz
        error stayed invisible."""
        g = self._level(2.0)
        ist, rows = self._tab([_result(g - 85.0, 3600.0 * 2.0, run="7947")])
        ist._update_centroids()
        self.assertAlmostEqual(
            float(rows["70Ge"].centroid_item.text()), -85.0, places=3)
        self.assertEqual(rows["74Ge"].centroid_item.text(), "0.0000")

    def test_the_shift_is_taken_in_the_drift_free_frame(self):
        g = self._level(2.0)
        ist, rows = self._tab([_result(g - 85.0, 3600.0 * 2.0, run="7947")])
        from unittest import mock
        with mock.patch("gui.analysis.isotope_shift_tab.QMessageBox"):
            ist._compute_shifts()
        by = {row["label"]: row for row in ist._last_shift_data}
        self.assertAlmostEqual(by["70Ge"]["delta_nu"], -85.0, places=3)

    def test_the_shift_carries_the_gp_uncertainty(self):
        """On-the-fly runs join the propagation pools, so the shift's
        error is more than the fit error alone."""
        g = self._level(2.0)
        ist, rows = self._tab([_result(g - 85.0, 3600.0 * 2.0, run="7947",
                                       err=3.0)])
        from unittest import mock
        with mock.patch("gui.analysis.isotope_shift_tab.QMessageBox"):
            ist._compute_shifts()
        row = {r["label"]: r for r in ist._last_shift_data}["70Ge"]
        # The GP term is kept out of the pure-statistics column by
        # design; it lands in sigma_correction and the total.
        self.assertGreater(row["sigma_correction"], 0.0)
        self.assertGreater(row["sigma_delta_nu"], 3.0)
        self.assertEqual(row["sigma_delta_nu_stat"], 3.0)
        self.assertIn("GP applied here", ist._last_log)

    def test_a_run_corrected_at_fit_time_is_left_alone(self):
        ist, rows = self._tab([_result(-85.0, 3600.0 * 2.0, run="7947",
                                       applied=self._level(2.0))])
        ist._update_centroids()
        self.assertAlmostEqual(
            float(rows["70Ge"].centroid_item.text()), -85.0, places=3)

    def test_without_a_fitted_gp_gp_mode_refuses(self):
        ist, rows = self._tab([_result(-230.0, 3600.0 * 2.0)])
        ist._ref_corr_panel._corrector = None
        ist._update_centroids()
        self.assertEqual(rows["70Ge"].centroid_item.text(), "fit GP")
        from unittest import mock
        with mock.patch(
                "gui.analysis.isotope_shift_tab.QMessageBox") as mb:
            ist._compute_shifts()
        self.assertTrue(mb.warning.called)

    def test_the_weighted_average_is_over_corrected_centroids(self):
        """Averaging raw centroids folds the drift into the scatter and
        inflates the Birge ratio by exactly what the correction is
        there to remove."""
        runs = [_result(self._level(t) - 85.0 + d, 3600.0 * t,
                        run=str(8000 + i), err=2.0)
                for i, (t, d) in enumerate(((1.0, 0.5), (5.0, -0.5)))]
        ist, rows = self._tab(runs)
        ist._update_centroids()
        self.assertAlmostEqual(
            float(rows["70Ge"].centroid_item.text()), -85.0, places=2)

    def test_changing_the_run_updates_the_cells_at_once(self):
        """Stepping through runs with the arrow keys showed each run's
        preview next to the previous run's numbers until Compute."""
        runs = [_result(self._level(1.0) - 80.0, 3600.0 * 1.0, run="8001"),
                _result(self._level(5.0) - 90.0, 3600.0 * 5.0, run="8002")]
        ist, rows = self._tab(runs)
        combo = rows["70Ge"].run_combo
        for run in ("8001", "8002"):
            if combo.findText(run) < 0:
                combo.addItem(run)
        combo.setCurrentText("8001")
        a = rows["70Ge"].centroid_item.text()
        combo.setCurrentText("8002")
        b = rows["70Ge"].centroid_item.text()
        self.assertAlmostEqual(float(a), -80.0, places=2)
        self.assertAlmostEqual(float(b), -90.0, places=2)


class GPModePlotTests(_ISFixture):

    def _plot(self, ist):
        from unittest import mock
        with mock.patch("gui.analysis.isotope_shift_tab.QMessageBox"):
            ist._compute_shifts()
        return ist._figure

    def _with_spectra(self, results, offset=0.0):
        x = np.linspace(-600, 200, 81)
        for r in results:
            c = r["params_df"]["Value"][0]
            r.update({"x": list(x), "y": list(100 / (1 + ((x - c) / 50) ** 2)),
                      "yerr": list(np.ones_like(x)),
                      "x_smooth": list(x), "y_fit_smooth": list(x * 0)})
        return results

    def test_the_gp_reference_panel_is_not_empty(self):
        """It used to look for a run named 'GP drift model', find none
        and skip the axis -- an empty panel with 0..1 ticks."""
        g = float(_real_gp().predict_interval(2.0, 2.0)[0])
        ref = self._with_spectra([
            _result(-150.0 + 2.0 * i, 3600.0 * i, run=str(7900 + i),
                    err=2.0 + 0.1 * i) for i in range(8)])
        smp = self._with_spectra([_result(g - 85.0, 3600.0 * 2.0,
                                          run="7947")])
        ist, rows = self._tab(smp, ref_results=ref)
        fig = self._plot(ist)
        drawn = [ax for ax in fig.axes if ax.lines or ax.collections]
        self.assertGreaterEqual(len(drawn), 2)
        notes = [t.get_text() for ax in fig.axes for t in ax.texts]
        self.assertTrue(any("drift-corrected" in n for n in notes))

    def test_the_most_precise_reference_run_is_shown(self):
        from gui.analysis.isotope_shift_tab import IsotopeShiftTab
        ref = [_result(-150.0, 3600.0 * i, run=str(7900 + i),
                       err=e) for i, e in enumerate((3.0, 1.5, 2.0))]
        ist, rows = self._tab([_result(-230.0, 3600.0)], ref_results=ref)
        proj = ist._find_project("74Ge")
        best = IsotopeShiftTab._best_reference_run(ist, proj, ref)
        self.assertEqual(best["run_number"], "7901")

    def test_an_excluded_run_is_never_the_one_shown(self):
        from gui.analysis.isotope_shift_tab import IsotopeShiftTab
        ref = [_result(-150.0, 3600.0 * i, run=str(7900 + i), err=e)
               for i, e in enumerate((3.0, 1.5))]
        ist, rows = self._tab([_result(-230.0, 3600.0)], ref_results=ref)
        ist._ref_corr_panel._excluded_obs = {"74Ge/7901"}
        best = IsotopeShiftTab._best_reference_run(
            ist, ist._find_project("74Ge"), ref)
        self.assertEqual(best["run_number"], "7900")


# ── GP panel views ────────────────────────────────────────────

class _PanelFixture(unittest.TestCase):

    class _P:
        def __init__(self, name, results, is_ref=False):
            self.project_name = name
            self._last_results = results
            self.is_reference = is_ref

    def _panel(self, *, excluded=()):
        from gui.analysis.reference_correction_panel import (
            ReferenceCorrectionPanel)
        rc = _real_gp()
        ref = [_result(-150.0 + 2.0 * i, 3600.0 * i, run=str(7900 + i))
               for i in range(8)]
        ref.append(_result(85.0, 3600.0 * 3.5, run="7961", err=3.5))
        g = float(rc.predict_interval(2.0, 2.0)[0])
        stub = QWidget()
        stub._projects = [self._P("74Ge", ref, is_ref=True),
                          self._P("70Ge", [_result(g - 85, 3600.0 * 2.0,
                                                   run="7947")])]
        panel = ReferenceCorrectionPanel(stub)
        self.addCleanup(panel.deleteLater)
        self.addCleanup(stub.deleteLater)
        panel._excluded_obs = set(excluded)
        panel._corrector = rc
        return panel


class CorrectedPanelViewTests(_PanelFixture):

    def test_the_zero_line_is_gone(self):
        panel = self._panel()
        panel._render_diagnostic_plot()
        ax = panel._figure.axes[-1]
        flat_zero = [ln for ln in ax.lines
                     if np.allclose(ln.get_ydata(), 0.0)]
        self.assertEqual(flat_zero, [])

    def test_the_reference_average_line(self):
        panel = self._panel(excluded={"74Ge/7961"})
        panel._show_ref_avg.setChecked(True)
        ax = panel._figure.axes[-1]
        labels = ax.get_legend_handles_labels()[1]
        avg = [lbl for lbl in labels if "weighted avg" in lbl]
        self.assertEqual(len(avg), 1)
        self.assertIn("74Ge", avg[0])

    def test_an_excluded_run_stays_out_of_the_average(self):
        """run 7961 sits ~230 MHz off. In the average it would drag the
        line far from the reference's real level."""
        d_in = self._panel()._plot_data()
        d_out = self._panel(excluded={"74Ge/7961"})._plot_data()
        self.assertGreater(abs(float(d_in["ref_avg"][0])
                               - float(d_out["ref_avg"][0])), 5.0)

    @staticmethod
    def _n_excluded_marks(ax):
        """Grey excluded markers: 'x' in range, a triangle pinned to
        the frame edge when off scale."""
        from gui.analysis.gp_figure import EXCLUDED_COLOR
        from matplotlib.colors import to_hex
        n = 0
        for ln in ax.lines:
            if (ln.get_marker() in ("x", "^", "v")
                    and len(ln.get_xdata())
                    and to_hex(ln.get_color()) == EXCLUDED_COLOR):
                n += len(ln.get_xdata())
        return n

    def test_excluded_points_follow_the_toggle_in_the_corrected_panel(self):
        panel = self._panel(excluded={"74Ge/7961"})
        panel._render_diagnostic_plot()
        self.assertEqual(self._n_excluded_marks(panel._figure.axes[-1]), 1)
        panel._show_excluded.setChecked(False)
        self.assertEqual(self._n_excluded_marks(panel._figure.axes[-1]), 0)
        tips = [t for *_, t in panel._hover_points]
        self.assertFalse(any("7961" in t for t in tips))

    def test_excluded_points_follow_the_toggle_in_the_residual_panel(self):
        panel = self._panel(excluded={"74Ge/7961"})
        panel._show_corrected.setChecked(False)
        panel._show_residuals.setChecked(True)
        self.assertEqual(self._n_excluded_marks(panel._figure.axes[1]), 1)
        panel._show_excluded.setChecked(False)
        self.assertEqual(self._n_excluded_marks(panel._figure.axes[1]), 0)

    def test_an_excluded_outlier_does_not_set_the_scale(self):
        """run_7961 is ~75 sigma off. Drawn as-is it put the residual
        axis at +/-50 sigma and flattened every fitted point onto 0."""
        panel = self._panel(excluded={"74Ge/7961"})
        panel._show_corrected.setChecked(False)
        panel._show_residuals.setChecked(True)
        lo, hi = panel._figure.axes[1].get_ylim()
        self.assertLess(hi, 10.0)
        tips = [t for *_, t in panel._hover_points if "7961" in t
                and "residual" in t]
        self.assertTrue(tips and "off scale" in tips[0])

    def test_corrected_hover_gives_original_gp_and_corrected(self):
        panel = self._panel()
        panel._render_diagnostic_plot()
        corr_ax = panel._figure.axes[-1]
        tips = [t for ax, _x, _y, t in panel._hover_points
                if ax is corr_ax and "7947" in t]
        self.assertEqual(len(tips), 1)
        for word in ("original", "GP", "corrected"):
            self.assertIn(word, tips[0])
        self.assertIn("-85.00", tips[0])

    def test_residual_hover_gives_original_and_gp(self):
        panel = self._panel()
        panel._show_corrected.setChecked(False)
        panel._show_residuals.setChecked(True)
        res_ax = panel._figure.axes[1]
        tips = [t for ax, *_r, t in panel._hover_points if ax is res_ax]
        self.assertTrue(tips)
        for word in ("residual", "original", "GP"):
            self.assertIn(word, tips[0])

    def test_send_to_results_sits_on_the_plot_row(self):
        panel = self._panel()
        right = panel._canvas.parentWidget()
        self.assertIs(panel._send_btn.parentWidget(), right)


class SentFigureMatchesScreenTests(_PanelFixture):
    """Send to Results saves the dict the panel drew, and the Results
    tab draws it with the same function. So the figure that comes back
    has the panels, unit and exclusions that were on screen."""

    def test_the_npz_redraws_the_same_view(self):
        import tempfile
        from matplotlib.figure import Figure
        from gui.analysis import gp_figure
        panel = self._panel(excluded={"74Ge/7961"})
        panel._t_unit.setCurrentIndex(1)            # minutes
        panel._show_residuals.setChecked(True)
        d = panel._plot_data()
        path = os.path.join(tempfile.mkdtemp(), "g.npz")
        np.savez(path, plot_type="gp_reference", **d)
        data = np.load(path, allow_pickle=True)
        loaded = {k: data[k] for k in data.files}
        self.assertEqual(gp_figure.layout(loaded), gp_figure.layout(d))
        fig = Figure()
        axes = gp_figure.draw(fig, loaded)
        self.assertEqual(list(axes), ["drift", "res", "corr"])
        self.assertIn("min", fig.axes[-1].get_xlabel())

    def test_an_old_npz_still_renders(self):
        """Saves from before the shared renderer carry only the drift
        arrays and the lower panel."""
        from matplotlib.figure import Figure
        from gui.analysis import gp_figure
        old = {"t_train": np.arange(3.0), "y_train": np.zeros(3),
               "yerr_train": np.ones(3), "t_grid": np.linspace(0, 2, 5),
               "mu": np.zeros(5), "sigma": np.ones(5),
               "corr_labels": np.array(["74Ge"], dtype=object),
               "corr_index": np.zeros(3, int), "corr_t": np.arange(3.0),
               "corr_y": np.zeros(3), "corr_yerr": np.ones(3),
               "xlabel": np.array("Timestamp [h] (since first measurement)")}
        axes = gp_figure.draw(Figure(), old)
        self.assertEqual(list(axes), ["drift", "corr"])


# ── Isotope Shifts layout ─────────────────────────────────────

class ISLayoutTests(unittest.TestCase):

    def _tab(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        return at._is_tab

    def test_neither_pane_can_collapse(self):
        """A collapsed QSplitter pane stays collapsed through every
        later resize -- which is how the plot vanished."""
        ist = self._tab()
        self.assertFalse(ist._main_splitter.childrenCollapsible())
        self.assertGreaterEqual(
            ist._main_splitter.widget(1).minimumWidth(), 360)

    def test_the_default_split_puts_the_plot_at_the_table_edge(self):
        ist = self._tab()
        ist.resize(2400, 1000)
        ist.show()
        self.addCleanup(ist.close)
        for _ in range(4):
            QCoreApplication.processEvents()
        left = ist._main_splitter.sizes()[0]
        self.assertAlmostEqual(left, ist._entries_min_width, delta=4)
        self.assertGreater(ist._main_splitter.sizes()[1], 1000)

    def test_a_layout_restored_while_hidden_keeps_the_plot(self):
        """The T02 case: [858, 1674] restored before the tab was ever
        shown."""
        ist = self._tab()
        ist.apply_ui_layout({"main": [858, 1674]})
        ist.resize(2540, 1000)
        ist.show()
        self.addCleanup(ist.close)
        for _ in range(4):
            QCoreApplication.processEvents()
        s = ist._main_splitter.sizes()
        self.assertGreater(s[1], 1000)
        # The saved 858 px, unless the table now needs more -- the
        # left pane's minimum is the table's measured width and wins.
        want = max(858 * sum(s) / 2532, ist._entries_min_width)
        self.assertAlmostEqual(s[0], want, delta=0.02 * sum(s))

    def test_a_font_change_re_measures_the_table(self):
        """Measured once at construction, the columns missed the zoom
        and the Win98 font applied after it.

        Asserted on the mechanism, not on pixel widths: the offscreen
        test platform has no font families at all, so its metrics do
        not grow with point size and a width comparison would prove
        nothing either way."""
        from unittest import mock
        from PySide6.QtGui import QFont
        ist = self._tab()
        f = QFont(ist.font())
        f.setPointSize(f.pointSize() + 4)
        with mock.patch.object(type(ist), "_measure_entry_columns",
                               autospec=True) as m:
            ist.setFont(f)
            QCoreApplication.processEvents()
        self.assertTrue(m.called)

    def test_the_re_measure_waits_for_every_widget_to_have_its_font(self):
        """Zoom sets fonts widget by widget in arbitrary order. The tab
        can hear its FontChange before the table has the new font, so
        the measurement is deferred a tick rather than taken at once."""
        import inspect
        from gui.analysis.isotope_shift_tab import IsotopeShiftTab
        src = inspect.getsource(IsotopeShiftTab.changeEvent)
        self.assertIn("singleShot", src)


if __name__ == "__main__":
    unittest.main()
