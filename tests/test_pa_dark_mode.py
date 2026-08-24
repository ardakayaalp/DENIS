"""Pre-Analysis dark-mode plot toggle + neon palette.

Date:    2026-07-26
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the sun/moon dark-mode toggle: state flips, the plot chrome goes
black-with-white in dark mode and back, line/model colours switch to
the neon palette, the choice is session-global across every open
Pre-Analysis project, and it round-trips through the save dict.
``_set_dark_mode`` (not the button's own toggle) is used wherever a
test would otherwise write the real settings file.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_pa_dark_mode.py -q

Depends on: gui.preanalysis_tab, gui.preanalysis_container; PySide6.
"""

import unittest

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP




class DarkModeTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.preanalysis_tab import PreAnalysisTab
        self.pa = PreAnalysisTab()

    def test_button_present_and_consistent(self):
        # The default is settings-derived, so assert the button state
        # matches _dark_mode rather than a hardcoded value.
        self.assertTrue(self.pa._dark_btn.isCheckable())
        self.assertEqual(self.pa._dark_btn.isChecked(), self.pa._dark_mode)

    def test_set_dark_mode_paints_chrome(self):
        self.pa._set_dark_mode(True, replot=False)
        self.assertTrue(self.pa._dark_mode)
        self.assertTrue(self.pa._dark_btn.isChecked())
        # Black canvas, white spines.
        self.assertEqual(self.pa._ax.get_facecolor()[:3], (0.0, 0.0, 0.0))
        self.assertEqual(
            self.pa._ax.spines["bottom"].get_edgecolor()[:3],
            (1.0, 1.0, 1.0))
        # Back to light: white canvas, black spines.
        self.pa._set_dark_mode(False, replot=False)
        self.assertEqual(self.pa._ax.get_facecolor()[:3], (1.0, 1.0, 1.0))
        self.assertEqual(
            self.pa._ax.spines["bottom"].get_edgecolor()[:3],
            (0.0, 0.0, 0.0))

    def test_entry_dual_color_follows_mode(self):
        from gui.preanalysis_tab import FileEntry
        e = FileEntry("f.h5", "run_1", 30.0, "", 0.0, 50,
                      color="#000080", dark_color="#00E5FF")
        self.assertEqual(e.color, "#000080")       # light active
        e.set_dark_active(True)
        self.assertEqual(e.color, "#00E5FF")       # dark active
        # Editing the dark colour is honoured in dark mode (the bug).
        e.dark_color = "#FF2D95"
        self.assertEqual(e.color, "#FF2D95")
        e.set_dark_active(False)
        self.assertEqual(e.color, "#000080")       # light preserved

    def test_toggle_syncs_entry_and_model_swatches(self):
        from gui.preanalysis_tab import FileEntry, HFSModelPanel
        e = FileEntry("f.h5", "run_1", 30.0, "", 0.0, 50,
                      color="#000080", dark_color="#00E5FF")
        self.pa._file_entries.append(e)
        m = HFSModelPanel(color="#000000", dark_color="#FFE500")
        self.pa._model_panels.append(m)
        self.pa._set_dark_mode(True, replot=False)
        self.assertEqual(e.color, "#00E5FF")       # entry active = dark
        self.assertEqual(m.color, "#FFE500")       # model active = dark
        self.pa._set_dark_mode(False, replot=False)
        self.assertEqual(e.color, "#000080")
        self.assertEqual(m.color, "#000000")

    def test_display_color_is_entry_active_color(self):
        from gui.preanalysis_tab import FileEntry
        e = FileEntry("f.h5", "run_1", 30.0, "", 0.0, 50,
                      color="#000080", dark_color="#00E5FF")
        self.assertEqual(self.pa._display_color(e), "#000080")
        e.set_dark_active(True)
        self.assertEqual(self.pa._display_color(e), "#00E5FF")

    def test_model_dual_color_round_trips(self):
        from gui.preanalysis_tab import HFSModelPanel
        m = HFSModelPanel(color="#111111", dark_color="#FFE500")
        d = m.to_dict()
        self.assertEqual(d["color"], "#111111")
        self.assertEqual(d["dark_color"], "#FFE500")
        m2 = HFSModelPanel()
        m2.from_dict(d)
        self.assertEqual(m2.light_color, "#111111")
        self.assertEqual(m2.dark_color, "#FFE500")

    def test_toggle_auto_assigns_neon_when_dark_equals_light(self):
        """Models/files that never got a distinct dark colour (old
        saves) get a neon one automatically on the dark toggle."""
        from gui.preanalysis_tab import (FileEntry, HFSModelPanel,
                                         NEON_COLORS, NEON_MODEL_COLORS)
        e = FileEntry("f.h5", "run_1", 30.0, "", 0.0, 50,
                      color="#654321")   # dark defaults == light
        e.color_index = 2
        self.assertEqual(e.light_color, e.dark_color)
        self.pa._file_entries.append(e)
        m = HFSModelPanel(color="#123456")   # dark == light
        self.assertEqual(m.light_color, m.dark_color)
        self.pa._model_panels.append(m)
        self.pa._set_dark_mode(True, replot=False)
        self.assertEqual(e.dark_color, NEON_COLORS[2])
        self.assertEqual(e.color, NEON_COLORS[2])
        self.assertEqual(m.dark_color, NEON_MODEL_COLORS[0])
        self.assertEqual(m.color, NEON_MODEL_COLORS[0])
        self.pa._set_dark_mode(False, replot=False)
        self.assertEqual(e.color, "#654321")
        self.assertEqual(m.color, "#123456")

    def test_restored_model_is_neon_on_the_first_draw(self):
        """A save with no per-model ``dark_color`` (pre-feature, which is
        every existing session file) restored in dark mode must be drawn
        in its neon colour by the FIRST replot.

        The bug: ``_restore_from_dict`` built the panel with a bare
        ``HFSModelPanel()``, so dark fell back to light (black); the
        replot at the end of the restore drew a black curve on the black
        canvas, and only the container's later ``_set_dark_mode`` upgraded
        the swatch to neon -- leaving a yellow swatch over a black line
        until the user unticked and re-ticked the model.
        """
        from gui.preanalysis_tab import (PreAnalysisTab, HFSModelPanel,
                                         NEON_MODEL_COLORS)
        legacy = HFSModelPanel(color="#000000").to_dict()
        del legacy["dark_color"]          # pre-feature save
        cfg = {"plot_options": {"dark_mode": True}, "models": [legacy]}

        # Record the colour every draw during the load would have used.
        drawn = []
        orig = PreAnalysisTab._update_models

        def spy(tab):
            drawn.extend(p.color for p in tab._model_panels)
            return orig(tab)

        PreAnalysisTab._update_models = spy
        try:
            pa = PreAnalysisTab()
            pa._restore_from_dict(cfg)
        finally:
            PreAnalysisTab._update_models = orig

        panel = pa._model_panels[0]
        self.assertEqual(panel.light_color, "#000000")   # save preserved
        self.assertEqual(panel.dark_color, NEON_MODEL_COLORS[0])
        self.assertEqual(panel.color, NEON_MODEL_COLORS[0])
        self.assertTrue(drawn, "the model was never drawn during the load")
        self.assertEqual(set(drawn), {NEON_MODEL_COLORS[0]})

    def test_neon_upgrade_without_replot_still_recolours_canvas(self):
        """``_set_dark_mode(replot=False)`` may hand a model its neon
        colour; the artists already on the canvas have to follow, or the
        swatch and the curve disagree (the session-load symptom)."""
        from gui.preanalysis_tab import (PreAnalysisTab, HFSModelPanel,
                                         NEON_MODEL_COLORS)
        calls = []
        orig = PreAnalysisTab._update_models

        def spy(tab):
            calls.append(True)
            return orig(tab)

        m = HFSModelPanel(color="#123456")     # dark == light: upgradeable
        self.pa._model_panels.append(m)
        PreAnalysisTab._update_models = spy
        try:
            self.pa._set_dark_mode(True, replot=False)
        finally:
            PreAnalysisTab._update_models = orig
        self.assertEqual(m.color, NEON_MODEL_COLORS[0])
        self.assertTrue(calls, "upgraded colour never reached the canvas")

        # No upgrade left to make -> no redraw work on a repeat call.
        calls.clear()
        PreAnalysisTab._update_models = spy
        try:
            self.pa._set_dark_mode(True, replot=False)
        finally:
            PreAnalysisTab._update_models = orig
        self.assertFalse(calls)

    def test_grid_toggles_and_dark_colour(self):
        # Buttons exist between layout and dark button; toggling sets
        # state and the grid colour tracks the theme.
        self.assertFalse(self.pa._grid_x or self.pa._grid_y)
        self.pa._set_grids(True, False)
        self.assertTrue(self.pa._grid_x)
        self.assertFalse(self.pa._grid_y)
        self.assertTrue(self.pa._grid_x_btn.isChecked())
        gl = self.pa._ax.get_xgridlines()
        self.assertTrue(gl and gl[0].get_visible())

    def test_win98_light_palette_default(self):
        from gui.preanalysis_tab import DEFAULT_COLORS
        # Navy first — the Windows 98 system palette, not tab10.
        self.assertEqual(DEFAULT_COLORS[0], "#000080")

    def test_round_trips_through_save_dict(self):
        from gui.preanalysis_tab import PreAnalysisTab
        self.pa._set_dark_mode(True, replot=False)
        d = self.pa._build_config_dict()
        self.assertTrue(
            d["preanalysis"]["plot_options"]["dark_mode"])
        pa2 = PreAnalysisTab()
        pa2._restore_from_dict(d["preanalysis"])
        self.assertTrue(pa2._dark_mode)
        # Legacy save without the key keeps the tab's current state.
        del d["preanalysis"]["plot_options"]["dark_mode"]
        pa3 = PreAnalysisTab()
        pa3._set_dark_mode(False, replot=False)
        pa3._restore_from_dict(d["preanalysis"])
        self.assertFalse(pa3._dark_mode)

    def test_session_global_across_projects(self):
        from gui.preanalysis_container import PreAnalysisContainer
        c = PreAnalysisContainer()
        p1 = c._add_project("PA_1")
        p2 = c._add_project("PA_2")
        # Drive via the container's sync (the button path would write
        # the real settings file).
        c._sync_dark_mode(True)
        self.assertTrue(p1._dark_mode and p2._dark_mode)
        p3 = c._add_project("PA_3")          # new project adopts it
        self.assertTrue(p3._dark_mode)
        c._sync_dark_mode(False)
        self.assertFalse(p1._dark_mode or p2._dark_mode or p3._dark_mode)

    def test_grids_session_global_and_round_trip(self):
        from gui.preanalysis_container import PreAnalysisContainer
        from gui.preanalysis_tab import PreAnalysisTab
        c = PreAnalysisContainer()
        p1 = c._add_project("PA_1")
        p2 = c._add_project("PA_2")
        c._sync_grids(True, True)
        self.assertTrue(p1._grid_x and p1._grid_y)
        self.assertTrue(p2._grid_x and p2._grid_y)
        p3 = c._add_project("PA_3")           # new project adopts it
        self.assertTrue(p3._grid_x and p3._grid_y)
        # Save round-trip.
        self.pa._set_grids(True, False)
        d = self.pa._build_config_dict()
        self.assertTrue(d["preanalysis"]["plot_options"]["grid_x"])
        self.assertFalse(d["preanalysis"]["plot_options"]["grid_y"])
        pa2 = PreAnalysisTab()
        pa2._restore_from_dict(d["preanalysis"])
        self.assertTrue(pa2._grid_x)
        self.assertFalse(pa2._grid_y)


if __name__ == "__main__":
    unittest.main()
