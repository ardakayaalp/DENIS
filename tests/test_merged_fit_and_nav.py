"""Merged-run fitting, merge override gating, and Results navigation.

Date:    2026-09-20
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the 2026-09-20 fixes:

* ``_fit_single_run`` on a MERGED entry no longer raises
  ``UnboundLocalError: eff_cfg`` -- that variable is only bound in the
  standard ASDF branch, but the result dict read it unconditionally to
  report the ToF gate, so every merged-run fit died.
* A merged frequency axis sitting absurdly far from the reference
  raises ``MERGED_AXIS_FAR_FROM_REFERENCE`` instead of "succeeding"
  with a flat model no parameter could reach.
* ``compute_merged_spectrum`` honors the SourceBlock's cooler/laser
  override TICK (``override_enabled``) rather than reading the
  spinboxes unconditionally -- they keep non-zero defaults while
  unticked, which silently Doppler-shifted every merge with a laser
  setpoint the run never used.
* The Results tree drives the viewer from the CURRENT item, so arrow
  keys change the displayed output and not just the highlight.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_merged_fit_and_nav.py -q

Depends on: gui.analysis.fitting, gui.analysis.merge, gui.results_tab;
PySide6; satlas2 (via the Polynomial model used as a cheap fit).
"""

import inspect
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _source_config(**over):
    cfg = {
        "Z": 32, "A": 73, "mass": 72.923459, "harmonic": 2,
        "e_lower": 0, "e_upper": 0, "tof_gate": [38.0, 44.0],
        "pmt_gate": [3, 4], "v_gate": None, "f_gate": None,
        "noise_filter": 0, "cooler_correction": "pbp",
        "ref_freq": 0, "ref_shift": 0,
        "cooler_override": 29977.0, "laser_override": 10920.0,
        "cal_order": 1, "override_enabled": False,
        "bin_mode": "Frequency", "yerr_mode": "Poisson sqrt(y+1)",
        "x_column": "bins_center",
    }
    cfg.update(over)
    return cfg


def _merged_data(x_offset=0.0):
    """A synthetic pre-merged spectrum: one Gaussian peak on a pedestal."""
    x = np.linspace(-500.0, 500.0, 120) + x_offset
    peak = 20.0 + 60.0 * np.exp(-0.5 * ((x - x_offset - 50.0) / 60.0) ** 2)
    y = np.random.default_rng(1).poisson(peak).astype(float)
    return {
        "merged_name": "merged_1_2",
        "x": x.tolist(), "y": y.tolist(),
        "yerr": np.sqrt(y + 1).tolist(),
        "x_unit": "MHz", "source_runs": ["1", "2"],
        "per_run": [], "source_files": [],
    }


def _poly_model():
    return [{
        "name": "Poly_1", "type": "Polynomial",
        "model_type": "Polynomial", "enabled": True,
        "source_assign": "Source_1",
        "params": {"p0": {"value": 20.0, "vary": True,
                          "min": -1e6, "max": 1e6, "expr": ""}},
    }]


_FITTER = {"separate": True, "method": "leastsq",
           "statistics": "Chi-square", "scale_covar": False}


def _axis_warning_codes(result):
    info = result.get("binning_info") or {}
    return [w.get("code")
            for w in (info.get("voltage_to_frequency_warnings") or [])]


class MergedRunFitTests(unittest.TestCase):
    def test_merged_fit_does_not_raise_unbound_eff_cfg(self):
        """Regression: the merged branch never binds eff_cfg, but the
        result dict read it to report the ToF gate."""
        from gui.analysis.fitting import _fit_single_run
        res = _fit_single_run(
            "merged://merged_1_2", _source_config(), _poly_model(),
            _FITTER, output_config={}, merged_data=_merged_data())
        self.assertTrue(
            res.get("success"),
            "merged fit failed: " + str(res.get("error"))[:400])
        # A merged spectrum was gated at merge time; there is no
        # per-run gate to report even though the Source block has one.
        self.assertIsNone(res.get("tof_gate"))

    def test_sane_merged_axis_raises_no_axis_warning(self):
        from gui.analysis.fitting import _fit_single_run
        res = _fit_single_run(
            "merged://merged_1_2", _source_config(), _poly_model(),
            _FITTER, output_config={}, merged_data=_merged_data())
        self.assertNotIn("MERGED_AXIS_FAR_FROM_REFERENCE",
                         _axis_warning_codes(res))

    def test_far_merged_axis_is_flagged(self):
        """A merge built in a different Doppler frame lands hundreds of
        THz from the reference; the fit must say so rather than return
        a flat model that silently 'converged'."""
        from gui.analysis.fitting import _fit_single_run
        res = _fit_single_run(
            "merged://merged_1_2", _source_config(), _poly_model(),
            _FITTER, output_config={},
            merged_data=_merged_data(x_offset=-4.5822e8))
        self.assertTrue(res.get("success"))
        self.assertIn("MERGED_AXIS_FAR_FROM_REFERENCE",
                      _axis_warning_codes(res))


class MergeOverrideGatingTests(unittest.TestCase):
    """compute_merged_spectrum must read the override spinboxes only
    when the SourceBlock tick is on -- they hold non-zero defaults
    (29977 V / 10920 cm^-1) while unticked, and applying those silently
    rebuilt every merged spectrum in the wrong Doppler frame."""

    def test_source_reads_override_only_when_enabled(self):
        import gui.analysis.merge as merge_mod
        src = inspect.getsource(merge_mod.compute_merged_spectrum)
        self.assertIn("override_enabled", src,
                      "merge ignores the override tick")
        # The gate must sit ahead of the statements that read the values.
        idx_gate = src.index("override_enabled")
        idx_cool = src.index('source_config.get("cooler_override"')
        idx_laser = src.index('source_config.get("laser_override"')
        self.assertLess(idx_gate, idx_cool)
        self.assertLess(idx_gate, idx_laser)

    def test_pipeline_and_merge_agree_on_the_gate(self):
        """The fit path and the merge path must not diverge again."""
        import gui.analysis.merge as merge_mod
        import gui.analysis.pipeline as pipe_mod
        pairs = ((merge_mod, merge_mod.compute_merged_spectrum),
                 (pipe_mod, pipe_mod.prepare_run_data))
        for mod, fn in pairs:
            with self.subTest(module=mod.__name__):
                self.assertIn("override_enabled", inspect.getsource(fn))


class ResultsTreeNavigationTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_viewer_follows_current_item_not_just_clicks(self):
        """Arrow keys move the current item without emitting
        itemClicked; the viewer has to follow currentItemChanged."""
        from PySide6.QtWidgets import QTreeWidgetItem
        from gui.results_tab import ResultsTab
        tab = ResultsTab()
        self.assertTrue(hasattr(tab, "_on_current_item_changed"))
        seen = []
        tab._on_item_clicked = lambda item, col: seen.append(item)

        tab._on_current_item_changed(None, None)      # tree cleared
        self.assertEqual(seen, [])

        node = QTreeWidgetItem(["x"])
        tab._updating_checks = True                   # rebuild in flight
        tab._on_current_item_changed(node, None)
        self.assertEqual(seen, [])

        tab._updating_checks = False
        tab._on_current_item_changed(node, None)
        self.assertEqual(seen, [node])

    def test_delete_paths_drop_the_stale_item_handle(self):
        """_rebuild_tree destroys every item; a kept handle would be a
        deleted C++ object the next re-render touches."""
        import gui.results_tab as rt
        for name in ("_delete_iteration", "_delete_project"):
            with self.subTest(method=name):
                src = inspect.getsource(getattr(rt.ResultsTab, name))
                self.assertIn("_current_tree_item = None", src)


class GpPanelTests(unittest.TestCase):
    """GP diagnostic styling and the project-agnostic corrections
    lookup the Pre-Analysis merge path needs."""

    def setUp(self):
        _ensure_app()

    def test_map_curve_is_burgundy_and_legend_names_reference(self):
        from cls_estimations import reference_correction as rc
        self.assertEqual(rc.MAP_LINE_COLOR, "#800020")
        self.assertEqual(rc.MAP_LINE_ALPHA, 0.8)
        self.assertEqual(rc.MAP_LEGEND_OBS, "Reference centroids")
        src = inspect.getsource(rc.ReferenceCorrector.diagnostic_plot)
        self.assertIn("MAP_LINE_COLOR", src)
        self.assertIn("MAP_LEGEND_OBS", src)
        # The Results renderer must reuse the same constants, not
        # copies. It now draws through gp_figure.draw -- the function
        # the live panel uses -- so check both ends of that.
        import gui.results_tab as rt
        from gui.analysis import gp_figure
        self.assertIn(
            "gp_figure",
            inspect.getsource(rt.ResultsTab._render_gp_reference_plot))
        self.assertIn("MAP_LINE_COLOR", inspect.getsource(gp_figure.draw))
        self.assertIn("MAP_LEGEND_OBS", inspect.getsource(gp_figure.draw))

    def test_results_dispatcher_knows_the_gp_plot_type(self):
        import gui.results_tab as rt
        src = inspect.getsource(rt.ResultsTab._render_from_npz)
        self.assertIn("gp_reference", src)

    def test_corrections_for_paths_skips_ambiguous_and_honors_gate(self):
        import types
        from gui.analysis.reference_correction_panel import (
            ReferenceCorrectionPanel)

        panel = ReferenceCorrectionPanel(None)
        panel._corrector = object()          # "a GP has been fitted"

        def fc(value):
            return types.SimpleNamespace(
                value_mhz=value, sigma_mhz=1.0, mode="Auto")

        panel._corrections = {
            "P1": {"/x/a.asdf": fc(-10.0), "/x/b.asdf": fc(-20.0)},
            "P2": {"/x/b.asdf": fc(-99.0)},
        }
        out = panel.corrections_for_paths(
            ["/x/a.asdf", "/x/b.asdf", "/x/c.asdf"])
        self.assertEqual(out["/x/a.asdf"]["correction_mhz"], -10.0)
        self.assertNotIn("/x/b.asdf", out)   # same run in two projects
        self.assertNotIn("/x/c.asdf", out)   # no correction computed

        panel._apply_global_cb.setChecked(False)
        self.assertEqual(panel.corrections_for_paths(["/x/a.asdf"]), {})

    def test_pa_merge_declares_override_enabled(self):
        """PA gates its override spinboxes itself, so it must tell the
        merge those values are authoritative -- otherwise the new
        override_enabled gate silently drops them."""
        import gui.preanalysis_tab as pa
        for name in ("_merge_checked", "_pa_synthetic_source_config"):
            with self.subTest(method=name):
                src = inspect.getsource(getattr(pa.PreAnalysisTab, name))
                self.assertIn("override_enabled", src)


class PreAnalysisCalibrationBadgeTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_calibration_plots_use_the_calibration_in_force(self):
        """The Calibrations sub-tab plotted only the raw header arrays,
        which never change, so an edited calibration redrew
        pixel-identical."""
        import gui.preanalysis_tab as pa
        src = inspect.getsource(pa.PreAnalysisTab._replot_calibrations)
        self.assertIn("CalibrationInfo", src)
        self.assertIn("predict_v", src)

    def test_file_rows_expose_an_edited_marker(self):
        import gui.preanalysis_tab as pa
        self.assertTrue(
            hasattr(pa.FileEntry, "refresh_cal_edited_badge"))


if __name__ == "__main__":
    unittest.main()
