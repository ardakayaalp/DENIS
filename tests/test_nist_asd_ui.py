"""NIST ASD browser UI: window, plotting, persistence round-trips.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the M3–M5 contracts:

- plotting renderers draw the expected artists (levels bars, stick
  spectrum with allowed/forbidden split, scheme arrows) and survive
  empty inputs;
- NistAsdWindow builds; set_data fills tables; filters filter;
- collect_search_config ⇄ from_dict/to_dict round-trips, including
  lasers with roles, isobars and orbital filters;
- set_results fills the ranked table and drives the diagram + details;
- window to_dict → fresh window from_dict restores config AND results
  offline (no network);
- MainWindow save-dict integration: pending state passes through
  _build_save_dict without the tool ever opening.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_nist_asd_ui.py -q

Depends on: gui.nist_asd.*, PySide6, matplotlib; the synthetic system
from test_nist_asd_search.
"""

import unittest

import pandas as pd

from PySide6.QtWidgets import QApplication

from matplotlib.figure import Figure

from gui.nist_asd import data as nd
from gui.nist_asd import plotting as nplot
from gui.nist_asd.models import Laser
from gui.nist_asd.search import SchemeSearchConfig, SchemeSearcher

from tests.test_nist_asd_search import _system, _two_step_cfg


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _ranked():
    lines, levels = _system()
    return SchemeSearcher(_two_step_cfg(), lines, levels).run()


class PlottingTests(unittest.TestCase):
    def test_plot_levels_draws_bars(self):
        lines, levels = _system()
        fig = Figure()
        ax = fig.add_subplot(111)
        n = nplot.plot_levels(ax, levels, lines_df=lines)
        self.assertEqual(n, len(levels))
        # Individual hover-capable lines, one per level.
        info_lines = [l for l in ax.lines
                      if getattr(l, "_nist_info", None)]
        self.assertEqual(len(info_lines), n)
        self.assertIn("cm⁻¹", info_lines[0]._nist_info)
        self.assertIn("τ", info_lines[0]._nist_info)
        # Lifetime → opacity: the long-lived 100 cm-1 level (slow M1
        # only, τ=100 s) is denser than the short-lived 45000 cm-1.
        by_e = {round(l.get_ydata()[0], 1): l.get_alpha() or 1.0
                for l in info_lines}
        self.assertGreater(by_e[100.0], by_e[45000.0])
        # Ground state (no decays → τ unknown/∞) is fully dense.
        self.assertEqual(by_e[0.0], 1.0)
        # Adaptive minor ticks installed on the energy axis.
        self.assertGreater(len(ax.yaxis.get_minorticklocs()), 0)
        # Zooming into a narrow band thickens the visible bars.
        lw_before = info_lines[0].get_linewidth()
        ax.set_ylim(0, 500)
        self.assertGreater(info_lines[0].get_linewidth(), lw_before)
        # Empty input: placeholder, no crash.
        self.assertEqual(nplot.plot_levels(ax, pd.DataFrame()), 0)

    def test_plot_lines_stick_splits_allowed_forbidden(self):
        lines, _levels = _system()
        fig = Figure()
        ax = fig.add_subplot(111)
        n = nplot.plot_lines_stick(ax, lines, "vacuum")
        self.assertEqual(n, 4)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("Allowed (E1)", labels)
        self.assertIn("Forbidden", labels)

    def test_plot_scheme_draws_steps_and_detection(self):
        ranked = _ranked()
        fig = Figure()
        ax = fig.add_subplot(111)
        nplot.plot_scheme(ax, ranked[0], "vacuum")
        self.assertGreaterEqual(len(ax.collections), 1)  # level bars
        # No score in the title (the ranked table carries it) and the
        # exact energies sit AT the bars in data coordinates.
        self.assertNotIn("Score", ax.get_title(loc="left"))
        energy_texts = [t for t in ax.texts
                        if "cm$^{-1}$" in t.get_text()]
        self.assertGreaterEqual(len(energy_texts), 3)
        for t in energy_texts:
            x, _y = t.get_position()
            self.assertLessEqual(x, 1.0)   # inside the bar span

    def test_diagnostics_renderers(self):
        lines, _levels = _system()
        fig = Figure()
        ax = fig.add_subplot(111)
        # Decay channels of the 45000 cm-1 level: two channels, drawn
        # as a level diagram (bars + arrows), not giant barh blocks.
        n = nplot.plot_decay_channels(ax, lines, 45000.0, "vacuum")
        self.assertEqual(n, 2)
        self.assertIn("45,000", ax.get_title(loc="left"))
        self.assertGreaterEqual(len(ax.collections), 3)  # level bars
        # y-axis starts at zero.
        self.assertEqual(ax.get_ylim()[0], 0.0)
        # BR labels present on the arrows.
        br_texts = [t for t in ax.texts if "BR" in t.get_text()]
        self.assertEqual(len(br_texts), 2)
        # Config (mathtext) on one side, energy on the other, per
        # destination bar.
        conf_texts = [t for t in ax.texts
                      if t.get_text().startswith("$3s")]
        self.assertGreaterEqual(len(conf_texts), 3)  # upper + dests
        # Destination + λ/BR labels are vertical (orthogonal).
        rotated = [t for t in ax.texts if t.get_rotation() == 90.0]
        self.assertGreaterEqual(len(rotated), 4)
        # Lifetimes: three distinct upper levels (100/20000/45000).
        self.assertEqual(nplot.plot_lifetimes(ax, lines), 3)
        # Line density bins every transition — and a pathologically
        # small bin over a huge span must not explode (bin cap).
        self.assertEqual(
            nplot.plot_line_density(ax, lines, "vacuum", 50.0), 4)
        self.assertEqual(
            nplot.plot_line_density(ax, lines, "vacuum", 0.001), 4)
        # All tolerate empty input.
        self.assertEqual(
            nplot.plot_decay_channels(ax, pd.DataFrame(), 0.0), 0)
        self.assertEqual(nplot.plot_lifetimes(ax, pd.DataFrame()), 0)
        self.assertEqual(
            nplot.plot_line_density(ax, pd.DataFrame()), 0)

    def test_decay_channels_crop_when_all_destinations_high(self):
        """A level whose channels all end high (e.g. 4p → 4s only)
        must not waste 80% of the plot on an empty 0..min region —
        the y floor crops below the lowest destination."""
        lines = nd.clean_lines_df(pd.DataFrame([
            {"Ei(cm-1)": 40000.0, "Ek(cm-1)": 45000.0,
             "Aki(s^-1)": 5.0e7, "conf_i": "3s2.3p.4s",
             "conf_k": "3s2.3p.4p", "term_i": "3P*", "term_k": "3D",
             "J_i": "1", "J_k": "2", "Type": ""},
            {"Ei(cm-1)": 41000.0, "Ek(cm-1)": 45000.0,
             "Aki(s^-1)": 1.0e7, "conf_i": "3s2.3p.4s",
             "conf_k": "3s2.3p.4p", "term_i": "1P*", "term_k": "3D",
             "J_i": "1", "J_k": "2", "Type": ""},
        ]))
        fig = Figure()
        ax = fig.add_subplot(111)
        n = nplot.plot_decay_channels(ax, lines, 45000.0, "vacuum")
        self.assertEqual(n, 2)
        self.assertGreater(ax.get_ylim()[0], 30000.0)

    def test_textbook_label_formatting(self):
        self.assertEqual(nplot.format_config_label("3s2.3p2"),
                         "$3s^{2}\\,3p^{2}$")
        # Odd parity renders as the textbook superscript "o".
        self.assertEqual(nplot.format_term_label("3P*"),
                         "$^{3}P^{o}$")
        self.assertEqual(nplot.format_term_label("2[3/2]"),
                         "$^{2}[3/2]$")
        self.assertEqual(
            nplot.format_level_label("3s2.3p2", "3P", "2"),
            "$3s^{2}\\,3p^{2}\\;^{3}P_{2}$")
        # jj-coupling parent gets superscript + J subscript.
        lbl = nplot.format_config_label("3s2.3p.(2P*<3/2>).4f")
        self.assertIn("^{2}P^{o}_{3/2}", lbl)
        self.assertEqual(nplot.format_config_label(""), "?")
        self.assertEqual(nplot.format_term_label(""), "")

    def test_level_diagram_uses_config_columns_with_mathtext(self):
        _lines, levels = _system()
        fig = Figure()
        ax = fig.add_subplot(111)
        n = nplot.plot_levels(ax, levels)
        self.assertEqual(n, len(levels))
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(len(ticklabels), 3)   # three configurations
        self.assertTrue(all(t.startswith("$") for t in ticklabels))

    def test_display_wavelength_falls_back_to_observed(self):
        row = {"ritz_wl_vac(nm)": float("nan"),
               "Ei(cm-1)": float("nan"), "Ek(cm-1)": float("nan"),
               "obs_wl_vac(nm)": 500.0}
        self.assertAlmostEqual(
            nd.display_wavelength_nm(row, "vacuum"), 500.0)
        self.assertLess(nd.display_wavelength_nm(row, "air"), 500.0)


class WindowTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def _window(self):
        from gui.nist_asd.tab import NistAsdWindow
        w = NistAsdWindow()
        lines, levels = _system()
        w.set_data(lines, levels)
        return w

    def test_set_data_fills_tables(self):
        w = self._window()
        self.assertEqual(w.levels_table.rowCount(), 4)
        self.assertEqual(w.lines_table.rowCount(), 4)
        self.assertIn("lines", w.cache_label.text())
        # Diagnostics view combos exist with the new entries.
        self.assertEqual(w.lv_plot_combo.count(), 3)
        self.assertEqual(w.ln_plot_combo.count(), 2)
        # Switching to decay channels with a selection draws bars.
        w.levels_table.selectRow(3)          # 45000 cm-1 (sorted asc)
        w.lv_plot_combo.setCurrentIndex(1)
        self.assertTrue(w.levels_fig.axes)

    def test_levels_filters(self):
        w = self._window()
        w.lv_emin.setValue(10000.0)
        w._fill_levels_table()
        self.assertEqual(w.levels_table.rowCount(), 2)
        w.lv_emin.setValue(0.0)
        w.lv_meta_check.setChecked(True)
        w._fill_levels_table()
        self.assertEqual(w.levels_table.rowCount(), 1)   # the 100 cm-1

    def test_lines_filters(self):
        w = self._window()
        w.ln_aki_edit.setText("1e7")
        w._fill_lines_table()
        self.assertEqual(w.lines_table.rowCount(), 3)
        w.ln_type_combo.setCurrentText("Forbidden")
        w._fill_lines_table()
        self.assertEqual(w.lines_table.rowCount(), 0)
        w.ln_aki_edit.setText("")
        w._fill_lines_table()
        self.assertEqual(w.lines_table.rowCount(), 1)    # the M1 line

    def test_manual_axis_limits(self):
        w = self._window()
        w.ln_xmin.setValue(200.0)
        w.ln_xmax.setValue(600.0)
        w._apply_lines_limits()
        ax = w.lines_fig.axes[0]
        self.assertEqual(ax.get_xlim(), (200.0, 600.0))
        # Limits survive a redraw (filters/plot switch)...
        w._draw_lines_plot()
        self.assertEqual(w.lines_fig.axes[0].get_xlim(),
                         (200.0, 600.0))
        # ...and Auto clears them.
        w._auto_lines_limits()
        self.assertEqual(w.ln_xmax.value(), 0.0)
        self.assertNotEqual(w.lines_fig.axes[0].get_xlim(),
                            (200.0, 600.0))

    def test_collect_search_config_round_trip(self):
        w = self._window()
        w._add_start_row(6298.85, 0.5)
        w._add_laser_row("UV", 250.0, 50.0, "PUMP")
        w._add_laser_row("Dye", 420.0, 30.0, "PROBE")
        w._add_iso_row("P I", 2.0)
        w.steps_spin.setValue(2)
        w.aki_probe_edit.setText("1e6")
        w.orb_probe_edit.setText("s->p, p->d")
        cfg = w.collect_search_config()
        self.assertEqual(len(cfg.lasers), 2)
        self.assertEqual(cfg.laser_objects()[0].role, "PUMP")
        self.assertEqual(cfg.aki_min_probe, 1e6)
        self.assertEqual(cfg.orbital_filter_probe, ["s->p", "p->d"])
        self.assertEqual(cfg.isobars,
                         [{"spectrum": "P I", "proximity_nm": 2.0}])
        self.assertEqual(cfg.starting_levels[0]["level"], 6298.85)

    def test_results_table_and_details(self):
        w = self._window()
        ranked = _ranked()
        w.set_results(ranked)
        self.assertEqual(w.results_table.rowCount(), 2)
        self.assertEqual(w.results_table.currentRow(), 0)
        self.assertIn("Score", w.detail_text.toPlainText())
        self.assertIn("BR", w.detail_text.toPlainText())

    def test_window_state_round_trip_offline(self):
        from gui.nist_asd.tab import NistAsdWindow
        w = self._window()
        w._add_laser_row("UV", 250.0, 50.0, "PUMP")
        w.set_results(_ranked())
        state = w.to_dict()
        self.assertEqual(len(state["results"]), 2)

        w2 = NistAsdWindow()
        w2.from_dict(state)   # no network: cache miss is tolerated
        self.assertEqual(w2.results_table.rowCount(), 2)
        self.assertEqual(w2.laser_table.rowCount(), 1)
        self.assertEqual(
            w2.collect_search_config().laser_objects()[0].name, "UV")
        # Round-trips to the same dict (modulo live-table absence).
        state2 = w2.to_dict()
        self.assertEqual(state2["search_config"],
                         state["search_config"])
        self.assertEqual(state2["results"], state["results"])


class MainWindowIntegrationTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_pending_state_passes_through_save_dict(self):
        """A loaded save file's nist_asd block must survive a
        load→save cycle even when the tool window is never opened."""
        from gui.main_window import MainWindow
        mw = MainWindow()
        try:
            payload = {"spectrum": "Nb I", "medium": "air",
                       "search_config":
                           SchemeSearchConfig(spectrum="Nb I").to_dict(),
                       "results": []}
            mw._pending_nist_state = payload
            d = mw._build_save_dict()
            self.assertEqual(d.get("nist_asd"), payload)
        finally:
            mw.deleteLater()

    def test_open_applies_pending_state(self):
        from gui.main_window import MainWindow
        mw = MainWindow()
        try:
            payload = {"spectrum": "Nb I", "medium": "vacuum",
                       "search_config":
                           SchemeSearchConfig(
                               spectrum="Nb I",
                               lasers=[Laser("X", 300, 5,
                                             "ANY").to_dict()],
                           ).to_dict(),
                       "results": []}
            mw._pending_nist_state = payload
            mw._open_nist_browser()
            w = mw._nist_browser
            self.assertIsNotNone(w)
            self.assertEqual(w.spectrum_edit.text(), "Nb I")
            self.assertEqual(w.laser_table.rowCount(), 1)
            self.assertIsNone(mw._pending_nist_state)
            # And save now reads the live window.
            d = mw._build_save_dict()
            self.assertEqual(d["nist_asd"]["spectrum"], "Nb I")
            w.close()
        finally:
            mw.deleteLater()


if __name__ == "__main__":
    unittest.main()
