"""Phase 1 (UI overhaul): theme application, wheel-focus guard, tooltips.

Date:    2026-07-24
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the three Phase-1 contracts:

1. ``gui.theme.apply_theme`` styles the whole app (palette + a stylesheet
   that reaches tooltips, scrollbars, tables...) and installs the wheel
   guard exactly once.
2. The wheel guard: scrolling over an UNFOCUSED spin box / combo / slider
   / tab bar must not change its value; once focused (clicked), the wheel
   works again; WheelFocus policies are downgraded to StrongFocus so the
   wheel can never *give* focus.
3. ``reflow_tooltip`` joins hand-wrapped tooltip lines into flowing rich
   text while keeping structure (equations, "x = ..." glossary lines,
   new sentences after punctuation, paragraph breaks) — and leaves rich
   or short tooltips alone.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_theme_wheel_tooltip -v

Depends on: gui.theme, gui.shared_widgets, gui.dialog_style; PySide6.
"""

import unittest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QSlider, QTabBar,
)


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _wheel(widget, delta=120):
    """Send a synthetic vertical wheel event to ``widget``."""
    pos = QPointF(5.0, 5.0)
    gpos = QPointF(widget.mapToGlobal(QPoint(5, 5)))
    ev = QWheelEvent(pos, gpos, QPoint(0, 0), QPoint(0, delta),
                     Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


class ThemeApplyTests(unittest.TestCase):
    def setUp(self):
        self.app = _ensure_app()

    def test_apply_theme_sets_stylesheet_and_palette(self):
        from gui.theme import apply_theme
        apply_theme(self.app)
        qss = self.app.styleSheet()
        for token in ("QToolTip", "QScrollBar", "QTableWidget",
                      "QGroupBox", "QComboBox", "QAbstractSpinBox",
                      "QFrame#plotCard"):
            self.assertIn(token, qss)
        self.assertEqual(
            self.app.palette().color(self.app.palette().ColorRole.Highlight)
                .name(), "#42a5f5")

    def test_wheel_guard_installed_once(self):
        from gui.theme import apply_theme, install_wheel_guard
        apply_theme(self.app)
        first = self.app._denis_wheel_guard
        self.assertIs(install_wheel_guard(self.app), first)

    def test_dialog_style_compat_surface(self):
        import gui.dialog_style as ds
        from gui import theme
        self.assertIsInstance(ds.DIALOG_QSS, str)
        self.assertEqual(ds.ACCENT, theme.ACCENT_TITLE)
        ds.style_dialog(None)  # must not raise


class Win98ThemeTests(unittest.TestCase):
    """The second theme: Classic 98 (dark), switchable at runtime.

    IMPORTANT: these tests must NOT call ``apply_theme`` on the shared
    QApplication. An app-wide restyle re-polishes every widget the
    whole test session has accumulated (hundreds, including
    deleteLater'd zombies that never flush without a running event
    loop) and hard-crashed the full suite (exit 255, no traceback).
    Content is asserted on the builder outputs; the real switch
    round-trip runs in a subprocess with a fresh QApplication — which
    is also what a real launch looks like.
    """

    def setUp(self):
        _ensure_app()

    def test_theme_registry(self):
        from gui.theme import THEMES
        self.assertIn("dark", THEMES)
        self.assertIn("win98", THEMES)

    def test_win98_qss_and_palette_content(self):
        from gui.theme import build_win98_palette, build_win98_qss
        qss = build_win98_qss()
        self.assertIn("#ffffe1", qss)           # buttermilk tooltip
        self.assertIn("border-top-color", qss)  # two-tone 3D bevels
        self.assertIn("#000080", qss)           # navy selection
        self.assertNotIn("#c0c0c0", qss)        # dark, not silver
        self.assertNotIn("border-radius: 3px", qss)  # sharp corners
        self.assertNotIn("font-family", qss)    # font set via QFont,
                                                # not a QSS rule
        # Same widget coverage as the dark sheet.
        for token in ("QToolTip", "QScrollBar", "QTableWidget",
                      "QGroupBox", "QComboBox", "QAbstractSpinBox"):
            self.assertIn(token, qss)
        pal = build_win98_palette()
        self.assertEqual(
            pal.color(pal.ColorRole.Highlight).name(), "#000080")
        self.assertEqual(
            pal.color(pal.ColorRole.Window).name(), "#2e2e2e")

    def test_retro_font_is_pixelated_monospace(self):
        from PySide6.QtGui import QFont
        from gui.theme import retro_font
        f = retro_font()
        self.assertEqual(f.styleStrategy(),
                         QFont.StyleStrategy.NoAntialias)
        self.assertEqual(f.family(), "Lucida Console")

    def test_main_tabs_qss_follows_theme(self):
        from gui.theme import main_tabs_qss
        dark = main_tabs_qss("dark")
        retro = main_tabs_qss("win98")
        self.assertIn("QTabWidget#MainTabs", dark)
        self.assertIn("QTabWidget#MainTabs", retro)
        self.assertIn("#232323", dark)
        self.assertIn("border-top-color", retro)   # beveled retro tabs
        self.assertIn("#3a3a3a", retro)            # dark faces
        self.assertNotEqual(dark, retro)

    def test_project_tab_close_button_wrapped_once(self):
        """The close button is wrapped in a right-margin container so
        it doesn't hug the tab border — and re-styling must not nest
        wrappers."""
        from PySide6.QtWidgets import QTabBar, QTabWidget, QWidget
        from gui.theme import style_project_tab_bar
        tabs = QTabWidget()
        tabs.setTabsClosable(True)
        tabs.addTab(QWidget(), "ProjA")
        tabs.addTab(QWidget(), "ProjB")
        style_project_tab_bar(tabs)
        style_project_tab_bar(tabs)  # idempotent
        bar = tabs.tabBar()
        for i in range(tabs.count()):
            wrap = bar.tabButton(i, QTabBar.ButtonPosition.RightSide)
            self.assertIsNotNone(wrap)
            self.assertEqual(wrap.objectName(), "closeWrap")
            lay = wrap.layout()
            self.assertEqual(lay.count(), 1)          # no nesting
            self.assertEqual(lay.contentsMargins().right(), 5)
            inner = lay.itemAt(0).widget()
            self.assertNotEqual(inner.objectName(), "closeWrap")
            # Regression: setTabButton hides the replaced widget (the
            # real close button inside the wrapper) — it must be
            # re-shown or the ✕ disappears from every project tab.
            self.assertFalse(inner.isHidden())

    def test_switch_roundtrip_in_fresh_app(self):
        """apply_theme win98 → dark on a FRESH QApplication: palette,
        sheet, pixel font applied and fully restored. Run in a
        subprocess so the shared test app is never restyled."""
        import os
        import subprocess
        import sys
        code = (
            "from PySide6.QtWidgets import QApplication\n"
            "from PySide6.QtGui import QFont\n"
            "app = QApplication([])\n"
            "from gui.theme import apply_theme, active_theme, "
            "retro_font\n"
            "default_family = app.font().family()\n"
            "apply_theme(app, 'win98')\n"
            "assert active_theme() == 'win98'\n"
            "assert app.palette().color("
            "app.palette().ColorRole.Highlight).name() == '#000080'\n"
            "assert '#ffffe1' in app.styleSheet()\n"
            "assert app.font().family() == retro_font().family()\n"
            "assert app.font().styleStrategy() == "
            "QFont.StyleStrategy.NoAntialias\n"
            "apply_theme(app, 'vaporwave')\n"
            "assert active_theme() == 'dark'\n"
            "assert app.palette().color("
            "app.palette().ColorRole.Highlight).name() == '#42a5f5'\n"
            "assert '#ffffe1' not in app.styleSheet()\n"
            "assert app.font().family() == default_family\n"
            "print('OK')\n"
        )
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True,
            text=True, cwd=root, env=env, timeout=120)
        self.assertEqual(res.returncode, 0,
                         f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
        self.assertIn("OK", res.stdout)


class WheelGuardTests(unittest.TestCase):
    def setUp(self):
        self.app = _ensure_app()
        from gui.theme import install_wheel_guard
        install_wheel_guard(self.app)

    def _arm(self, widget):
        """Polish delivers the guard; ensurePolished triggers it headless."""
        widget.ensurePolished()
        self.assertTrue(widget.property("_denisWheelGuarded"))

    def test_unfocused_spinbox_ignores_wheel(self):
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setValue(50.0)
        self._arm(spin)
        self.assertFalse(spin.hasFocus())
        _wheel(spin)
        self.assertEqual(spin.value(), 50.0)

    def test_focused_spinbox_accepts_wheel(self):
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setValue(50.0)
        spin.setSingleStep(1.0)
        self._arm(spin)
        spin.show()
        spin.setFocus()
        QApplication.processEvents()
        if not spin.hasFocus():
            spin.hide()
            self.skipTest("window activation unavailable in this session")
        _wheel(spin)
        spin.hide()
        self.assertEqual(spin.value(), 51.0)

    def test_unfocused_combo_keeps_selection(self):
        combo = QComboBox()
        combo.addItems(["a", "b", "c"])
        combo.setCurrentIndex(1)
        self._arm(combo)
        _wheel(combo)
        _wheel(combo, -120)
        self.assertEqual(combo.currentIndex(), 1)

    def test_unfocused_slider_keeps_value(self):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(40)
        self._arm(slider)
        _wheel(slider)
        self.assertEqual(slider.value(), 40)

    def test_tabbar_wheel_never_switches(self):
        bar = QTabBar()
        for name in ("one", "two", "three"):
            bar.addTab(name)
        bar.setCurrentIndex(1)
        self._arm(bar)
        _wheel(bar)
        _wheel(bar, -120)
        self.assertEqual(bar.currentIndex(), 1)

    def test_wheelfocus_downgraded_to_strongfocus(self):
        spin = QDoubleSpinBox()
        self.assertEqual(spin.focusPolicy(), Qt.FocusPolicy.WheelFocus)
        self._arm(spin)
        self.assertEqual(spin.focusPolicy(), Qt.FocusPolicy.StrongFocus)

    def test_scrollbar_not_guarded(self):
        from PySide6.QtWidgets import QScrollBar
        bar = QScrollBar(Qt.Orientation.Vertical)
        bar.ensurePolished()
        self.assertFalse(bool(bar.property("_denisWheelGuarded")))


class ReflowTooltipTests(unittest.TestCase):
    def _reflow(self, text):
        from gui.shared_widgets import reflow_tooltip
        return reflow_tooltip(text)

    def test_short_and_rich_pass_through(self):
        self.assertIsNone(self._reflow("Drag to reorder"))
        self.assertIsNone(self._reflow("<b>already rich</b>"))
        self.assertIsNone(self._reflow(""))
        self.assertIsNone(self._reflow(None))

    def test_hard_wrapped_lines_join(self):
        tip = ("How the cooler-voltage drift is subtracted from each "
               "event's\npost-deceleration voltage in "
               "clstools.Compute_Voltages:\n"
               "    V = Vcooler · VCoolDiv + VCoolOffset − DV_cal")
        out = self._reflow(tip)
        self.assertTrue(out.startswith("<qt>"))
        self.assertIn("post-deceleration voltage in clstools", out)
        self.assertIn("<br>V = Vcooler", out)

    def test_definition_lines_keep_rows_and_join_continuations(self):
        tip = ("Modes:\n\n"
               "pbp = point-by-point: use the cooler voltage measured at "
               "each\nevent's timestamp. Tracks drift through the run.\n"
               "Pick this when the cooler isn't perfectly stable.")
        out = self._reflow(tip)
        self.assertIn("timestamp. Tracks drift through the run.", out)
        # New sentence after '.' starting uppercase keeps its own row.
        self.assertIn("<br>Pick this when", out)
        # Paragraph break preserved.
        self.assertIn("<br><br>", out)

    def test_escapes_html(self):
        tip = ("Compare a & b when the range x -> y is larger than the "
               "step\nand the scan wraps around the edge of the window.")
        out = self._reflow(tip)
        self.assertIn("a &amp; b", out)

    def test_numbered_glossary_rows_survive(self):
        tip = ("Polynomial order for the calibration.\n"
               "1 = linear (slope + offset) — the default.\n"
               "2/3 = parabolic / cubic — only for non-linear readback\n"
               "across the scan range.")
        out = self._reflow(tip)
        self.assertIn("<br>1 = linear", out)
        self.assertIn("<br>2/3 = parabolic", out)
        self.assertIn("readback across the scan range.", out)


if __name__ == "__main__":
    unittest.main()
