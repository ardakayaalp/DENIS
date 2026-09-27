"""A progress window for loading a save file.

Restoring a session is slow for a good reason: every run named in the
file is opened and read. On a five-project campaign that is half a
minute during which DENIS looks hung -- no window updates, no idea
whether it is working or stuck.

This shows what it is doing, and it keeps the app ALIVE while it
does. The second part is not decoration: an application that does not
pump its message queue for half a minute is marked "Not Responding" by
Windows, which then offers to close it for you -- and doing that kills
DENIS mid-restore. From the outside that is a crash on the loading
screen, with nothing in the session log but the banner. The first
version of this window painted itself by hand to avoid re-entrancy and
walked straight into it (2026-09-25).

So updates pump the event loop, under two restrictions that keep that
safe from the middle of a synchronous restore:

* **User input is excluded** (``ExcludeUserInputEvents``), so no click
  or keystroke reaches a half-built UI. The window is modal as well.
* **Deferred deletes stay queued** -- ``processEvents`` does not
  process them -- so widgets a restore has scheduled for deletion live
  until it has finished. Destroying them mid-load is a use-after-free
  through ``MainWindow._apply_zoom``, which touches every widget in the
  process.

Timers do fire. The one that matters is the Pre-Analysis replot, which
stands down for the duration (``is_loading``); that restore ends with a
replot of its own.

**Reporting is a no-op when no window is up.** The loaders call
:func:`report` unconditionally; tests and headless runs pay nothing.

Depends on: PySide6.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QCoreApplication, QEventLoop, Qt
from PySide6.QtWidgets import (
    QDialog, QLabel, QProgressBar, QVBoxLayout,
)

#: Resolution of the bar. Finer than percent so a long phase still
#: creeps rather than sitting still.
_SCALE = 1000

#: Don't repaint more often than this. A repaint is a synchronous
#: draw; doing one per run file on a fast disk would cost more than
#: the work it is reporting.
_MIN_REPAINT_S = 0.05

#: Settings key. True (the default) shows the window; False keeps
#: the pumping -- which is what stops Windows declaring the app
#: dead -- with no window at all. A switch, because the window is
#: the one part of this that interacts with the rest of the
#: desktop, and when a load is crashing that is the first thing to
#: take out of the picture.
SETTING = "show_load_progress"

_active = None


def report(text="", done=None, total=None):
    """Say what is being loaded now, from wherever it is happening.

    ``done``/``total`` move the bar inside the current phase; without
    them only the text changes. Safe to call when nothing is showing.
    """
    if _active is not None:
        _active.report(text, done, total)


def active():
    """The open progress window, or None."""
    return _active


def is_loading():
    """True while a load is in progress.

    Work a restore will redo at the end anyway -- the Pre-Analysis
    replot -- checks this and stands down, so pumping the event loop
    during a load cannot turn one debounced replot into thirty.
    """
    return _active is not None


class SilentProgress:
    """Progress with no window: it only keeps the event loop pumped.

    Same interface as the dialog, so the loaders cannot tell which one
    they were given.
    """

    def __init__(self, phases=()):
        self._last_paint = 0.0

    @classmethod
    def begin(cls, parent, filename, phases):
        global _active
        dlg = cls(phases)
        _active = dlg
        return dlg

    def finish(self):
        global _active
        if _active is self:
            _active = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.finish()
        return False

    def phase(self, key, detail=""):
        self._pump(force=True)

    def report(self, text="", done=None, total=None):
        self._pump()

    def _pump(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_paint < _MIN_REPAINT_S:
            return
        self._last_paint = now
        QCoreApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


def begin(parent, filename, phases, show=None):
    """Start reporting: the window, or the silent stand-in.

    ``show`` defaults to the ``show_load_progress`` setting.
    """
    if show is None:
        try:
            from gui.shared_widgets import _load_settings
            show = bool(_load_settings().get(SETTING, True))
        except Exception:                                # noqa: BLE001
            show = True
    if not show:
        return SilentProgress.begin(parent, filename, phases)
    return LoadProgressDialog.begin(parent, filename, phases)


class LoadProgressDialog(QDialog):
    """Modal, un-closable progress window for one load.

    Un-closable on purpose: a half-applied restore is a worse state
    than a slow one, and there is nothing to cancel back to.
    """

    def __init__(self, parent, filename, phases):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint,
                           False)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        self._title = QLabel(f"Loading {filename}")
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        lay.addWidget(self._title)

        self._phase_label = QLabel("Reading the file...")
        lay.addWidget(self._phase_label)

        self._bar = QProgressBar()
        self._bar.setRange(0, _SCALE)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        lay.addWidget(self._bar)

        self._detail = QLabel("")
        # Dim, but not invisible on the dark theme.
        self._detail.setStyleSheet("color: palette(placeholder-text);")
        self._detail.setMinimumWidth(380)
        # Fixed height, and the text elided by hand below: this window
        # is painted without ever returning to the event loop, so a
        # label that wants to grow would simply be clipped -- the
        # layout it asks for never runs.
        self._detail.setFixedHeight(
            self._detail.fontMetrics().height() + 4)
        lay.addWidget(self._detail)

        # [(key, label, weight)] -> the fraction of the bar each phase
        # owns, so a file with no Pre-Analysis section does not leave a
        # gap in the middle.
        total = float(sum(max(0.0, w) for _k, _l, w in phases)) or 1.0
        self._spans = {}
        at = 0.0
        for key, label, weight in phases:
            share = max(0.0, weight) / total
            self._spans[key] = (at, share, label)
            at += share
        self._base = 0.0
        self._span = 0.0
        self._last_paint = 0.0

    # -- lifecycle --------------------------------------------------
    @classmethod
    def begin(cls, parent, filename, phases):
        """Show the window and make it the reporting target."""
        global _active
        dlg = cls(parent, filename, phases)
        dlg.show()
        dlg.raise_()
        # The one and only event-loop re-entry: the window has to be
        # mapped before it can be painted, and nothing has been
        # rebuilt yet, so a stray timer here is harmless.
        QCoreApplication.processEvents()
        _active = dlg
        return dlg

    def finish(self):
        global _active
        if _active is self:
            _active = None
        self.close()
        self.deleteLater()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.finish()
        return False

    # -- reporting --------------------------------------------------
    def phase(self, key, detail=""):
        """Move to a named phase (one of the keys given at creation)."""
        base, span, label = self._spans.get(key, (self._base, 0.0, key))
        self._base, self._span = base, span
        self._phase_label.setText(label)
        self._detail.setText(self._elide(detail))
        self._paint(force=True)

    def report(self, text="", done=None, total=None):
        if text:
            self._detail.setText(self._elide(str(text)))
        if done is not None and total:
            frac = min(max(float(done) / float(total), 0.0), 1.0)
            self._bar.setValue(
                int((self._base + self._span * frac) * _SCALE))
        self._paint()

    def _elide(self, text):
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(self._detail.font())
        return fm.elidedText(text, Qt.TextElideMode.ElideMiddle,
                             max(120, self._detail.width() - 4))

    def _paint(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_paint < _MIN_REPAINT_S:
            return
        self._last_paint = now
        # Pump, so Windows keeps seeing a live application. User input
        # is excluded, and processEvents leaves DeferredDelete events
        # alone, so nothing a restore queued for deletion is destroyed
        # under it.
        QCoreApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
