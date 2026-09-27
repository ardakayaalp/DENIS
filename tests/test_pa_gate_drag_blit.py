"""The ToF-gate drag must never leave the previous spectrum on screen.

Arda (2026-09-27): "when I move the ToF window around, the spectrum plot
doesn't update properly ... it shows the previous spectrum". The drag
redraws only the spectrum line over a cached background (blitting). When
the new counts left the y-view, the fast path rescaled and re-captured
that background with the line still VISIBLE, so the old spectrum was
baked into it and every later frame drew the new one on top.

These tests drive PreAnalysisTab._replot_spectrum_fast_impl on a plain
Agg canvas with a stand-in for the tab, so no data file is needed.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_pa_gate_drag_blit.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from gui.preanalysis_tab import PreAnalysisTab  # noqa: E402

X = np.arange(40, dtype=float)


class _Timer:
    def stop(self):
        pass


class _Tab:
    """Just what the fast path touches."""

    _replot_spectrum_fast_impl = PreAnalysisTab._replot_spectrum_fast_impl

    def __init__(self, y0):
        self.fig = Figure(figsize=(4, 3), dpi=60)
        self._canvas = FigureCanvasAgg(self.fig)
        self._ax = self.fig.add_subplot(111)
        (line,) = self._ax.step(X, y0, where="mid", color="red", lw=2)
        self._ax.set_ylim(0, y0.max() * 1.08)
        self._canvas.draw()
        self._fast = {"entry": None, "line": line, "normalize": False,
                      "xdisplay": X}
        self._fast_bg = None
        self._replot_timer = _Timer()
        self.next_y = y0
        self.full_replots = 0

    def _current_gate_args(self, entry):
        return None, None, None

    def _fast_histogram(self, cache, pmt, tofg, tsg):
        return self.next_y

    def _replot(self, spectrum_only=False):
        self.full_replots += 1

    def move_gate(self, y):
        self.next_y = np.asarray(y, dtype=float)
        self._replot_spectrum_fast_impl()


def _red_pixels(canvas):
    buf = np.asarray(canvas.buffer_rgba())
    return int(np.sum((buf[..., 0] > 200) & (buf[..., 1] < 80)
                      & (buf[..., 2] < 80)))


class GateDragBlitTests(unittest.TestCase):

    def _background_red_pixels(self, tab):
        tab._canvas.restore_region(tab._fast_bg)
        return _red_pixels(tab._canvas)

    def test_a_rescale_does_not_bake_the_line_into_the_background(self):
        tab = _Tab(np.full(40, 40.0))
        tab.move_gate(np.full(40, 38.0))            # first capture
        tab.move_gate(np.full(40, 2.0))             # counts collapse: rescale
        self.assertEqual(tab.full_replots, 0, "fell back to a full replot")
        self.assertEqual(self._background_red_pixels(tab), 0)

    def test_growing_counts_rescale_cleanly_too(self):
        tab = _Tab(np.full(40, 5.0))
        tab.move_gate(np.full(40, 5.0))
        tab.move_gate(np.full(40, 50.0))            # above the view: rescale
        self.assertEqual(self._background_red_pixels(tab), 0)

    def test_after_a_rescale_only_the_new_spectrum_is_shown(self):
        tab = _Tab(np.full(40, 40.0))
        tab.move_gate(np.full(40, 38.0))
        tab.move_gate(np.full(40, 2.0))
        tab.move_gate(np.full(40, 1.5))             # a frame after the rescale
        shown = np.asarray(tab._canvas.buffer_rgba()).copy()
        tab._canvas.draw()                          # what it should be
        truth = np.asarray(tab._canvas.buffer_rgba())
        differing = np.any(shown != truth, axis=-1)
        # Only the line's own anti-aliased edge along the x-axis may
        # differ (blit layers the line over the spine); never a second
        # spectrum -- which would be a whole row of red pixels.
        self.assertLess(int(differing.sum()), 60)

    def test_the_line_ends_with_the_new_counts(self):
        tab = _Tab(np.full(40, 40.0))
        tab.move_gate(np.full(40, 2.0))
        np.testing.assert_array_equal(tab._fast["line"].get_ydata(),
                                      np.full(40, 2.0))


if __name__ == "__main__":
    unittest.main()
