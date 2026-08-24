"""Binning dialog: 3-tab layout and gap-aware occupancy stats.

Date:    2026-07-26
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the redesigned BinningDialog: Summary / Run detail / Compare tabs
(the old Diagnostics + Occupancy + Widths trio collapsed into one
spectrum-only detail view — occupancy and width health surface as
title/warning text, not extra panels), the fit-relevant Summary
columns (Empty, Median counts, Total counts — no Verdict/Why/Fallback
noise), the scan-gap heuristic in ``_occupancy_stats``, and the
aliasing narrow-bin flag firing only on uniform grids.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_binning_dialog.py -q

Depends on: gui.analysis.blocks; PySide6; numpy; matplotlib.
"""

import unittest

import numpy as np
from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _make_run(n=120, gap=True, narrow=False, per_step=False, seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(-500.0, 500.0, n)
    y = rng.poisson(40, n).astype(float)
    y += 300 * np.exp(-0.5 * ((x - 120) / 30) ** 2)
    if gap:
        y[30:80] = 0.0          # 50 contiguous zeros -> scan gap
        y[5] = 0.0              # lone in-scan empty
        y[90] = 3.0             # low-count bin
    edges = np.concatenate(
        [[x[0] - 4.0], (x[:-1] + x[1:]) / 2, [x[-1] + 4.0]])
    widths = np.diff(edges)
    if narrow:
        widths[10] = widths[10] * 0.2   # aliasing-style narrow bin
    return {
        "x": x, "y": y, "yerr": np.sqrt(np.maximum(y, 1.0)),
        "x_label": "Doppler-shifted frequency [MHz]",
        "info": {
            "source": "run_0001.h5", "bin_mode": "Frequency",
            "bin_definition": ("Per scan step" if per_step else "Auto"),
            "effective_n_bins": n,
            "effective_bin_width_mhz": float(np.median(widths)),
            "x_min": float(x[0]), "x_max": float(x[-1]),
            "per_step": per_step,
        },
        "diagnostics": {
            "bin_centers": x, "bin_edges": edges, "bin_widths": widths,
            "edges_source": "clstools_intervals",
            "raw_x_sample": rng.uniform(-500, 500, 2000),
        },
    }


class OccupancyStatsTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_gap_vs_true_empty(self):
        from gui.analysis.blocks import BinningDialog
        y = np.ones(120)
        y[30:80] = 0.0   # 50-run  -> gap (>= max(30, 6))
        y[5] = 0.0       # lone    -> true empty
        s = BinningDialog._occupancy_stats(y)
        self.assertEqual(s["n_gap"], 50)
        self.assertEqual(s["n_empty"], 1)
        self.assertEqual(s["n_in_scan"], 70)
        self.assertTrue(s["gap_mask"][30] and s["gap_mask"][79])
        self.assertTrue(s["true_empty_mask"][5])

    def test_short_zero_runs_stay_empty(self):
        from gui.analysis.blocks import BinningDialog
        y = np.ones(100)
        y[10:30] = 0.0   # 20 < threshold max(30, 5) -> NOT a gap
        s = BinningDialog._occupancy_stats(y)
        self.assertEqual(s["n_gap"], 0)
        self.assertEqual(s["n_empty"], 20)

    def test_low_median_total(self):
        from gui.analysis.blocks import BinningDialog
        y = np.array([0.0, 3.0, 5.0, 100.0, 40.0])
        s = BinningDialog._occupancy_stats(y)
        self.assertEqual(s["n_low"], 2)          # 3 and 5
        self.assertEqual(s["median_nonzero"], 22.5)
        self.assertEqual(s["total"], 148.0)
        self.assertEqual(s["max"], 100.0)

    def test_empty_array(self):
        from gui.analysis.blocks import BinningDialog
        s = BinningDialog._occupancy_stats(np.array([]))
        self.assertEqual(s["n_in_scan"], 0)
        self.assertEqual(s["total"], 0.0)


class VoltageFrequencyScaleTests(unittest.TestCase):
    """dν/dV measured from a run's own events + the two-domain bin
    sizes surfaced in the Run-detail conversion strip."""

    def setUp(self):
        _ensure_app()

    @staticmethod
    def _fake(dv_vals, f_mhz, events=10, tdc=3):
        import pandas as pd
        rows = []
        for v, f in zip(dv_vals, f_mhz):
            rows += [{"DV": float(v), "F": f * 1e6,
                      "TOF": 41.0, "TDC": tdc}] * events
        d = type("D", (), {})()
        d.Sorted = pd.DataFrame(rows)
        return d

    def test_linear_scale_exact(self):
        from gui.analysis.binning import _voltage_frequency_scale
        # F = 25 MHz/V, DV steps of 2 V.
        dv = [0, 2, 4, 6, 8]
        s = _voltage_frequency_scale(
            self._fake(dv, [25.0 * v for v in dv]),
            (None, [3, 4], None, None))
        self.assertAlmostEqual(s["mhz_per_volt"], 25.0)
        self.assertAlmostEqual(s["mhz_per_volt_min"], 25.0)
        self.assertAlmostEqual(s["mhz_per_volt_max"], 25.0)
        self.assertAlmostEqual(s["v_step_median"], 2.0)

    def test_nonlinear_spread_reported(self):
        from gui.analysis.binning import _voltage_frequency_scale
        s = _voltage_frequency_scale(
            self._fake([0, 2, 4, 6, 8], [0, 40, 90, 150, 220]),
            (None, [3, 4], None, None))
        self.assertAlmostEqual(s["mhz_per_volt"], 27.5)   # median 20/25/30/35
        self.assertAlmostEqual(s["mhz_per_volt_min"], 20.0)
        self.assertAlmostEqual(s["mhz_per_volt_max"], 35.0)

    def test_compute_binned_populates_scale_and_bin_sizes(self):
        from gui.analysis.binning import compute_binned
        dv = [0, 2, 4, 6, 8, 10]
        out = compute_binned(
            self._fake(dv, [25.0 * v for v in dv]),
            {"bin_mode": "Frequency", "bin_definition": "Per scan step",
             "yerr_mode": "None"},
            include_diagnostics=True)
        info = out["info"]
        self.assertAlmostEqual(info["mhz_per_volt"], 25.0)
        # 2 V steps × 25 MHz/V = 50 MHz between steps = rest-frame bin.
        self.assertAlmostEqual(info["bin_width_rest_mhz"], 50.0)
        self.assertAlmostEqual(info["bin_width_v"], 2.0)

    def test_scale_strip_html(self):
        from gui.analysis.blocks import BinningDialog
        html = BinningDialog._scale_strip_html(
            {"mhz_per_volt": 25.0, "bin_width_v": 2.0,
             "bin_width_rest_mhz": 50.0})
        self.assertIn("raw-voltage bin", html)
        self.assertIn("25", html)
        self.assertIn("rest-frame bin", html)
        # No data → explicit unavailable note, never a crash.
        self.assertIn("unavailable",
                      BinningDialog._scale_strip_html({}))

    def test_detail_tab_shows_scale_strip(self):
        from gui.analysis.blocks import BinningDialog
        res = _make_run()
        res["info"].update({
            "mhz_per_volt": 25.0, "mhz_per_volt_min": 24.0,
            "mhz_per_volt_max": 26.0, "bin_width_v": 2.0,
            "bin_width_rest_mhz": 50.0})
        dlg = BinningDialog([("run_0001", res, None)], [])
        dlg._tabs.setCurrentIndex(1)
        self.assertIn("Scale:", dlg._diag_scale.text())
        self.assertIn("MHz", dlg._diag_scale.text())


class BinningDialogTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.analysis.blocks import BinningDialog
        self._cls = BinningDialog
        self.infos = [
            ("run_0001", _make_run(gap=True, narrow=True), None),
            ("run_0002", _make_run(gap=False, seed=11), None),
            ("run_bad", None, "file unreadable"),
        ]

    def test_three_tabs(self):
        dlg = self._cls(self.infos, [])
        labels = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
        self.assertEqual(labels, ["Summary", "Run detail", "Compare"])

    def test_summary_columns_no_verdict(self):
        dlg = self._cls(self.infos, [])
        t = dlg._summary_table
        hdr = [t.horizontalHeaderItem(c).text()
               for c in range(t.columnCount())]
        self.assertEqual(hdr[-3:],
                         ["Empty", "Median counts", "Total counts"])
        for gone in ("Verdict", "Why", "Fallback"):
            self.assertNotIn(gone, hdr)
        # Gap-aware Empty cell: 1 true empty over 70 in-scan bins.
        self.assertTrue(t.item(0, 8).text().startswith("1/70"))
        self.assertEqual(t.item(0, 10).text(),
                         f"{self.infos[0][1]['y'].sum():.0f}")
        self.assertTrue(t.item(2, 1).text().startswith("ERROR"))

    def test_detail_is_single_spectrum_panel(self):
        """Per Arda: only the spectrum plot — occupancy and width
        health live in the title/warning strips, not extra panels."""
        dlg = self._cls(self.infos, [])
        dlg._tabs.setCurrentIndex(1)
        self.assertEqual(len(dlg._diag_fig.axes), 1)
        dlg._rug_cb.setChecked(True)     # optional rug strip only
        axes = dlg._diag_fig.axes
        self.assertEqual(len(axes), 2)
        self.assertTrue(axes[1].get_shared_x_axes().joined(*axes))
        # Occupancy still reported as text on the title strip.
        self.assertIn("empty", dlg._diag_title.text())

    def test_narrow_bin_warning_only_on_uniform_grids(self):
        dlg = self._cls(self.infos, [])
        dlg._tabs.setCurrentIndex(1)
        self.assertIn("narrower than half", dlg._diag_warn.text())
        dlg._run_combo.setCurrentText("run_0002")
        self.assertNotIn("narrower than half", dlg._diag_warn.text())
        # Same narrow widths, but per-step binning -> no aliasing flag.
        infos = [("run_ps", _make_run(narrow=True, per_step=True), None)]
        dlg2 = self._cls(infos, [], default_tab="diagnostics")
        self.assertNotIn("narrower than half", dlg2._diag_warn.text())

    def test_diagnostics_default_tab_maps_to_detail(self):
        dlg = self._cls(self.infos, [], default_tab="diagnostics",
                        default_run="run_0002")
        self.assertEqual(dlg._tabs.currentIndex(), 1)
        self.assertEqual(dlg._run_combo.currentText(), "run_0002")

    def test_picker_hidden_on_summary_and_compare(self):
        dlg = self._cls(self.infos, [])
        self.assertFalse(dlg._picker_row.isVisibleTo(dlg))
        dlg._tabs.setCurrentIndex(1)
        self.assertTrue(dlg._picker_row.isVisibleTo(dlg))
        dlg._tabs.setCurrentIndex(2)
        self.assertFalse(dlg._picker_row.isVisibleTo(dlg))


if __name__ == "__main__":
    unittest.main()
