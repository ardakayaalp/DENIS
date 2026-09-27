"""Walk and correlation (corner) plots of an emcee chain.

Two requests from Arda (2026-09-25): mark the burn-in cut on the walk
plot as a vertical line, and draw the correlation plot like a
publication corner plot -- filled 2-D credible regions at 0.5, 1, 1.5
and 2 sigma, step histograms with the 16/50/84 % lines, value titles --
with a drop-down for the colour scale. Also pinned here: the corner
plot and the band use the burned-in chain, the same samples the
reported values come from.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_mcmc_plots.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.contour import QuadContourSet  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from gui.analysis.mcmc_plots import (  # noqa: E402
    CONTOUR_CMAPS, CORNER_STYLES, CREDIBLE_SIGMAS, credible_mass,
    density_thresholds, draw_corner, draw_walk, posterior_slice,
    region_colours, value_title)

WALK_STYLE = {"trace_alpha": 0.3, "trace_lw": 0.6, "label_size": 11,
              "tick_size": 9, "title_size": 12, "show_burn_line": True,
              "burn_line_color": "#000000", "burn_line_width": 1.5}
CORNER_STYLE = {"hist_bins": 30, "hist_color": "steelblue",
                "hist_alpha": 0.7, "scatter_s": 0.5, "scatter_alpha": 0.25,
                "scatter_color": "black", "label_size": 9, "tick_size": 8,
                "title_size": 12, "cell_size": 2.0,
                "style": "Credible regions", "contour_cmap": "Blues",
                "font_family": "serif", "contour_bins": 25, "smooth": 1.0,
                "hist_lw": 1.5, "quantile_lw": 1.0, "contour_lw": 1.0,
                "frame_lw": 2.0, "max_ticks": 3}


def _chain(steps=400, walkers=12, ndim=3, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.6, 0.0], [0.6, 2.0, -0.3], [0.0, -0.3, 0.5]])
    cov = cov[:ndim, :ndim]
    return rng.multivariate_normal(np.arange(ndim) * 10.0, cov,
                                   size=(steps, walkers))


def _lines_with_gid(fig, gid):
    return [ln for ax in fig.axes for ln in ax.get_lines()
            if ln.get_gid() == gid]


class CredibleLevelTests(unittest.TestCase):

    def test_the_masses_are_the_2d_gaussian_ones(self):
        got = [round(100 * credible_mass(s), 1) for s in CREDIBLE_SIGMAS]
        self.assertEqual(got, [11.8, 39.3, 67.5, 86.5])

    def test_a_threshold_encloses_its_mass(self):
        """The 1-sigma region of a 2-D Gaussian holds 39.3 % of it."""
        rng = np.random.default_rng(3)
        xy = rng.normal(size=(400_000, 2))
        H, _xe, _ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=120,
                                     range=[[-5, 5], [-5, 5]])
        masses = [credible_mass(s) for s in CREDIBLE_SIGMAS]
        levels = density_thresholds(H, masses)
        for m, v in zip(sorted(masses, reverse=True), levels):
            with self.subTest(mass=m):
                inside = H[H >= v].sum() / H.sum()
                self.assertAlmostEqual(inside, m, delta=0.02)

    def test_levels_rise_strictly(self):
        H = np.zeros((10, 10))
        H[5, 5] = 4.0          # every mass lands on the same bin
        levels = density_thresholds(H, [0.1, 0.4, 0.7, 0.9])
        self.assertTrue(np.all(np.diff(levels) > 0))

    def test_an_empty_histogram_has_no_levels(self):
        self.assertIsNone(density_thresholds(np.zeros((5, 5)), [0.5]))


class PosteriorSliceTests(unittest.TestCase):

    def test_no_burn_is_the_whole_chain(self):
        c = _chain(steps=50)
        sliced, used = posterior_slice(c, 0, 1)
        self.assertIs(sliced, c)
        self.assertEqual(used, 0)

    def test_burn_and_thin(self):
        c = _chain(steps=50)
        sliced, used = posterior_slice(c, 10, 2)
        self.assertEqual(used, 10)
        np.testing.assert_array_equal(sliced, c[10::2])

    def test_burning_everything_keeps_two_steps(self):
        c = _chain(steps=50)
        sliced, used = posterior_slice(c, 500, 1)
        self.assertEqual(sliced.shape[0], 2)
        self.assertEqual(used, 48)

    def test_it_matches_the_fit_worker(self):
        """The plots and the reported numbers use one rule."""
        import inspect
        from gui.analysis import fitting
        self.assertIn("posterior_slice(chain, burn, thin)",
                      inspect.getsource(fitting._apply_emcee_burnin))


class WalkPlotTests(unittest.TestCase):

    def _draw(self, burn, **style):
        ws = dict(WALK_STYLE, **style)
        fig = Figure()
        draw_walk(fig, ["centroid", "Al", "FWHML"], _chain(), ws,
                  run_num="7507", burn=burn)
        return fig

    def test_the_cut_is_marked_on_every_panel(self):
        fig = self._draw(100)
        lines = _lines_with_gid(fig, "burn_in_line")
        self.assertEqual(len(lines), 3)
        for ln in lines:
            self.assertEqual(list(ln.get_xdata()), [100, 100])

    def test_it_is_labelled(self):
        fig = self._draw(100)
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        self.assertTrue(any("burn-in: 100" in t for t in texts))

    def test_no_burn_no_line(self):
        self.assertEqual(_lines_with_gid(self._draw(0), "burn_in_line"), [])

    def test_it_can_be_switched_off(self):
        fig = self._draw(100, show_burn_line=False)
        self.assertEqual(_lines_with_gid(fig, "burn_in_line"), [])

    def test_a_cut_beyond_the_chain_is_not_drawn(self):
        self.assertEqual(_lines_with_gid(self._draw(10_000),
                                         "burn_in_line"), [])

    def test_the_colour_setting_is_used(self):
        fig = self._draw(100, burn_line_color="#ff00ff")
        ln = _lines_with_gid(fig, "burn_in_line")[0]
        self.assertEqual(matplotlib.colors.to_hex(ln.get_color()), "#ff00ff")


class CornerPlotTests(unittest.TestCase):

    LABELS = ["centroid", "Al", "FWHML"]

    def _draw(self, **style):
        cs = dict(CORNER_STYLE, **style)
        flat = _chain().reshape(-1, 3)
        fig = Figure(figsize=(6, 6))
        axes = draw_corner(fig, self.LABELS, flat, cs, run_num="7507")
        return fig, axes, flat

    def test_the_upper_triangle_is_empty(self):
        _fig, axes, _ = self._draw()
        for i in range(3):
            for j in range(3):
                with self.subTest(i=i, j=j):
                    self.assertEqual(axes[i, j].get_visible(), j <= i)

    def test_every_pair_has_filled_regions(self):
        _fig, axes, _ = self._draw()
        for i, j in ((1, 0), (2, 0), (2, 1)):
            with self.subTest(i=i, j=j):
                sets = [c for c in axes[i, j].collections
                        if isinstance(c, QuadContourSet) and c.filled]
                self.assertEqual(len(sets), 1)
                self.assertEqual(len(sets[0].levels),
                                 len(CREDIBLE_SIGMAS) + 1)

    def test_the_legend_names_the_regions(self):
        fig, _axes, _ = self._draw()
        leg = next(lg for lg in fig.legends
                   if lg.get_gid() == "credible_regions_legend")
        texts = [t.get_text() for t in leg.get_texts()]
        self.assertEqual(len(texts), 4)
        for pct in ("11.8 %", "39.3 %", "67.5 %", "86.5 %"):
            self.assertTrue(any(pct in t for t in texts), pct)
        self.assertEqual(leg.get_title().get_text(), "2-D credible regions")

    def test_the_diagonal_titles_carry_the_median(self):
        _fig, axes, flat = self._draw()
        for k in range(3):
            with self.subTest(k=k):
                title = axes[k, k].get_title()
                med = np.percentile(flat[:, k], 50)
                self.assertIn(f"{med:.2f}", title)
                self.assertIn("^{+", title)

    def test_the_diagonal_marks_16_50_84(self):
        _fig, axes, flat = self._draw()
        xs = sorted(ln.get_xdata()[0] for ln in axes[0, 0].get_lines())
        want = np.percentile(flat[:, 0], [15.87, 50, 84.13])
        np.testing.assert_allclose(xs, want)

    def test_the_colour_scale_changes_the_fills(self):
        _f1, a1, _ = self._draw(contour_cmap="Blues")
        _f2, a2, _ = self._draw(contour_cmap="Reds")

        def first_fill(axes):
            cs = next(c for c in axes[1, 0].collections
                      if isinstance(c, QuadContourSet) and c.filled)
            return tuple(np.round(cs.get_facecolor()[0], 3))
        self.assertNotEqual(first_fill(a1), first_fill(a2))

    def test_an_unknown_colour_scale_falls_back(self):
        self._draw(contour_cmap="not-a-colormap")   # must not raise

    def test_the_inner_region_is_the_darkest(self):
        for name in CONTOUR_CMAPS:
            with self.subTest(cmap=name):
                fills, _line = region_colours(name)
                lum = [0.2126 * r + 0.7152 * g + 0.0722 * b
                       for r, g, b, _a in fills]
                self.assertEqual(lum, sorted(lum, reverse=True))

    def test_the_scatter_style_is_still_there(self):
        fig, axes, _ = self._draw(style="Scatter")
        self.assertTrue(any(isinstance(c, PathCollection)
                            for c in axes[1, 0].collections))
        self.assertFalse(any(lg.get_gid() == "credible_regions_legend"
                             for lg in fig.legends))

    def test_both_styles_are_offered(self):
        self.assertEqual(CORNER_STYLES[0], "Credible regions")
        self.assertIn("Scatter", CORNER_STYLES)

    def test_a_single_parameter_draws(self):
        fig = Figure()
        draw_corner(fig, ["centroid"], _chain(ndim=1).reshape(-1, 1),
                    CORNER_STYLE, run_num="1")
        self.assertEqual(fig.legends, [])

    def test_a_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            draw_corner(Figure(), ["a", "b"], np.zeros((10, 3)),
                        CORNER_STYLE)


class TitleFormatTests(unittest.TestCase):

    def test_two_significant_figures_of_the_error(self):
        self.assertEqual(value_title("c", 628.594, 0.581, 0.577),
                         "c = $628.59^{+0.58}_{-0.58}$")
        self.assertEqual(value_title("c", 1.23456, 0.0045, 0.0047),
                         "c = $1.2346^{+0.0045}_{-0.0047}$")


class SettingsTests(unittest.TestCase):
    """The drop-downs in both settings dialogs."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_the_choices_come_from_the_renderer(self):
        from gui.shared_widgets import get_plot_type_choices
        self.assertEqual(get_plot_type_choices("correlation_plot",
                                               "contour_cmap"),
                         CONTOUR_CMAPS)
        self.assertEqual(get_plot_type_choices("correlation_plot", "style"),
                         CORNER_STYLES)
        self.assertIsNone(get_plot_type_choices("fit_plot", "style"))

    def test_the_defaults_are_valid_choices(self):
        from gui.shared_widgets import (_DEFAULT_PLOT_TYPE_SETTINGS,
                                        get_plot_type_choices)
        d = _DEFAULT_PLOT_TYPE_SETTINGS["correlation_plot"]
        for key in ("style", "contour_cmap", "font_family"):
            with self.subTest(key=key):
                self.assertIn(d[key],
                              get_plot_type_choices("correlation_plot", key))

    def test_the_widget_is_a_drop_down(self):
        from PySide6.QtWidgets import QComboBox
        from gui.shared_widgets import _make_plot_setting_widget
        w = _make_plot_setting_widget("Blues", "Reds", "", key="contour_cmap",
                                      choices=CONTOUR_CMAPS)
        self.addCleanup(w.deleteLater)
        self.assertIsInstance(w, QComboBox)
        self.assertEqual(w.currentText(), "Reds")

    def test_an_unknown_saved_value_falls_back_to_the_default(self):
        from gui.shared_widgets import _make_plot_setting_widget
        w = _make_plot_setting_widget("Blues", "jet", "", key="contour_cmap",
                                      choices=CONTOUR_CMAPS)
        self.addCleanup(w.deleteLater)
        self.assertEqual(w.currentText(), "Blues")

    def test_both_dialogs_show_drop_downs(self):
        from PySide6.QtWidgets import QComboBox
        from gui.main_window import SettingsDialog
        from gui.shared_widgets import PlotTypeOptionsDialog
        for cls in (SettingsDialog, PlotTypeOptionsDialog):
            with self.subTest(dialog=cls.__name__):
                dlg = cls()
                self.addCleanup(dlg.deleteLater)
                w = dlg._pt_widgets["correlation_plot"]
                for key in ("style", "contour_cmap", "font_family"):
                    self.assertIsInstance(w[key], QComboBox, key)


if __name__ == "__main__":
    unittest.main()
