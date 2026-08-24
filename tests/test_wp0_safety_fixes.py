"""Regression tests for the WP0 safety fixes (2026-06-02 code review).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers two fixes:

1. yerr_mode "None" must not crash the fit. binning returns yerr=None for
   that mode; satlas2 divides by yerr, so the fit construction substitutes
   unit weights via fitting._resolve_fit_yerr (the documented plain
   least-squares semantics). A model-based callable yerr is left untouched.

2. The close-confirmation must not discard unsaved state when the user
   cancels the Save dialog. _save_all / _save_tab now return False on cancel,
   and closeEvent only accepts the close when the save returned True.

These run headless without a real QApplication or .asdf file by stubbing the
window object and patching the Qt dialogs.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_wp0_safety_fixes -v

Depends on: gui.analysis.fitting._resolve_fit_yerr; gui.main_window.MainWindow
(_save_all/_save_tab/closeEvent), patched QFileDialog/QMessageBox.
"""

import unittest
from unittest import mock

import numpy as np

from gui.analysis.fitting import _resolve_fit_yerr


class ResolveFitYerrTests(unittest.TestCase):
    """fitting._resolve_fit_yerr — the yerr_mode 'None' crash fix."""

    def test_none_not_callable_becomes_unit_weights(self):
        y = np.array([4.0, 9.0, 16.0])
        out = _resolve_fit_yerr(None, y, use_callable_yerr=False)
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(out, np.ones_like(y))
        self.assertEqual(out.shape, y.shape)

    def test_none_with_callable_flag_left_as_none(self):
        # Model-based yerr is signalled by use_callable_yerr=True; satlas2
        # evaluates it per iteration, so we must not clobber it.
        y = np.array([1.0, 2.0])
        self.assertIsNone(_resolve_fit_yerr(None, y, use_callable_yerr=True))

    def test_real_array_passes_through_unchanged(self):
        y = np.array([1.0, 2.0, 3.0])
        yerr = np.array([1.0, 1.4142, 1.7320])
        out = _resolve_fit_yerr(yerr, y, use_callable_yerr=False)
        self.assertIs(out, yerr)

    def test_callable_passes_through_unchanged(self):
        y = np.array([1.0, 2.0])
        out = _resolve_fit_yerr(np.sqrt, y, use_callable_yerr=True)
        self.assertIs(out, np.sqrt)


# ── Stubs for the save/close tests (no QApplication needed) ──

class _FakeEvent:
    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class _Role:
    AcceptRole = 1
    DestructiveRole = 2


class _Icon:
    Question = 1


class _StdButton:
    Cancel = 99


class _FakeMsgBox:
    """Minimal stand-in for QMessageBox used by MainWindow.closeEvent.

    Buttons are added in the order save_all, save_tab, discard, cancel;
    CLICK_INDEX selects which one clickedButton() reports.
    """

    ButtonRole = _Role
    Icon = _Icon
    StandardButton = _StdButton
    CLICK_INDEX = 0

    def __init__(self, parent=None):
        self._buttons = []

    def setWindowTitle(self, *a):
        pass

    def setText(self, *a):
        pass

    def setIcon(self, *a):
        pass

    def addButton(self, *a):
        btn = object()
        self._buttons.append(btn)
        return btn

    def setDefaultButton(self, *a):
        pass

    def exec(self):
        pass

    def clickedButton(self):
        return self._buttons[self.CLICK_INDEX]


class _StubTabs:
    def currentWidget(self):
        return "widget"

    def tabText(self, _i):
        return "Estimate"

    def currentIndex(self):
        return 0


class _StubWindow:
    """Carries only what _save_all/_save_tab/closeEvent touch."""

    def __init__(self, save_result=True, dirty=True):
        self.tabs = _StubTabs()
        self._save_result = save_result
        self._dirty = dirty
        self._config_path = None

    def _config_dialog_start_dir(self, _action):
        return "."

    def _tab_key(self, _widget):
        return "estimate"

    # closeEvent consults the dirty flag first (Phase 2: clean sessions
    # close without a prompt), then calls _save() on the Save button.
    def _is_dirty(self):
        return self._dirty

    def _save(self):
        return self._save_result

    def _save_all(self):
        return self._save_result

    def _save_tab(self):
        return self._save_result


class SaveCancelReturnsFalseTests(unittest.TestCase):
    """_save_all / _save_tab return False when the dialog is cancelled."""

    def test_save_all_returns_false_on_cancel(self):
        from gui.main_window import MainWindow
        with mock.patch("gui.main_window.QFileDialog.getSaveFileName",
                        return_value=("", "")):
            self.assertFalse(MainWindow._save_all(_StubWindow()))

    def test_save_tab_returns_false_on_cancel(self):
        from gui.main_window import MainWindow
        with mock.patch("gui.main_window.QFileDialog.getSaveFileName",
                        return_value=("", "")):
            self.assertFalse(MainWindow._save_tab(_StubWindow()))


class CloseEventRespectsSaveResultTests(unittest.TestCase):
    """closeEvent must ignore the close when the save was cancelled."""

    def _run_close(self, save_result, dirty=True):
        from gui.main_window import MainWindow
        _FakeMsgBox.CLICK_INDEX = 0  # "Save" button
        ev = _FakeEvent()
        with mock.patch("gui.main_window.QMessageBox", _FakeMsgBox):
            MainWindow.closeEvent(
                _StubWindow(save_result=save_result, dirty=dirty), ev)
        return ev

    def test_cancelled_save_keeps_window_open(self):
        ev = self._run_close(save_result=False)
        self.assertIs(ev.accepted, False)  # event.ignore() -> stay open

    def test_successful_save_closes(self):
        ev = self._run_close(save_result=True)
        self.assertIs(ev.accepted, True)  # event.accept() -> close

    def test_clean_session_closes_without_prompt(self):
        """Phase 2: nothing changed -> no save prompt at all."""
        from gui.main_window import MainWindow
        ev = _FakeEvent()

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError(
                    "closeEvent built a QMessageBox for a clean session")
        with mock.patch("gui.main_window.QMessageBox", _Boom):
            MainWindow.closeEvent(_StubWindow(dirty=False), ev)
        self.assertIs(ev.accepted, True)


if __name__ == "__main__":
    unittest.main()
