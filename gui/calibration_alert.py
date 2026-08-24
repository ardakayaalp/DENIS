"""The blinking calibration warning on a file row.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

DENIS's calibration policy is "change nothing, but say something": a run
whose calibration outliers would actually move its centroid is flagged
rather than silently fixed, and the user decides. That only works if the
warning finds them -- an overview dialog nobody opens is not a warning.

So a flagged run gets a blinking ``!`` right on its row, in both
Pre-Analysis and Analysis. Clicking it explains what is wrong and what it
costs in MHz, offers to open the diagnostic, and stops the blinking for
good: a warning that keeps nagging after it has been read is one the user
learns to ignore, and then it is worth nothing when it matters.

"Stops for good" means the acknowledgement persists with the project (it
serializes under the YAML's ``calibration_acks`` key). Fixing the run's
calibration also clears the flag, since a run with an override is no
longer un-triaged.

One QTimer drives every badge -- with a few hundred files open, one timer
per row would be a few hundred timers.

Depends on: gui.calibration, gui.calibration_dialog, PySide6.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QToolButton,
    QVBoxLayout,
)

from gui.dialog_style import style_dialog
from gui.calibration import (
    DEFAULT_ORDER,
    canonical_path,
    diagnose,
    get_registry,
    read_beam_header,
    shift_mhz_over_scan,
    fit_calibration,
    read_cal_points,
    resolve_calibration,
    DEFAULT_VACC_DIV,
)

_ON_STYLE = ("color: #ffffff; background: #e65100; border: 1px solid #bf360c;"
             "border-radius: 3px; font-weight: bold;")
_OFF_STYLE = ("color: #e65100; background: transparent;"
              "border: 1px solid #e65100; border-radius: 3px;"
              "font-weight: bold;")

BLINK_MS = 650


class _BlinkClock(QObject):
    """A single heartbeat every badge listens to."""

    tick = Signal(bool)

    def __init__(self):
        super().__init__()
        self._phase = True
        self._timer = QTimer(self)
        self._timer.setInterval(BLINK_MS)
        self._timer.timeout.connect(self._beat)

    def ensure_running(self):
        if not self._timer.isActive():
            self._timer.start()

    def _beat(self):
        self._phase = not self._phase
        self.tick.emit(self._phase)


_CLOCK: Optional[_BlinkClock] = None


def blink_clock() -> _BlinkClock:
    global _CLOCK
    if _CLOCK is None:
        _CLOCK = _BlinkClock()
    return _CLOCK


def calibration_cost_mhz(path: str, cal_map: dict, suspect,
                         physics: dict, cal_order: int = DEFAULT_ORDER):
    """``(centroid_mhz, tilt_mhz)`` this run's outliers are costing, or None.

    Beam voltage and laser come from the run's own header -- they are per-run
    values, and quoting one run's cost with a neighbour's beam energy would be
    confidently wrong.
    """
    if not suspect or not physics:
        return None
    if not all(physics.get(k) for k in ("mass_amu", "harmonic")):
        return None
    cooler_v, laser_cm = read_beam_header(path)
    if not (cooler_v and laser_cm):
        return None
    try:
        set_v, read_raw = read_cal_points(path)
        cur = resolve_calibration(path, cal_map, cal_order=cal_order)
        fixed = fit_calibration(set_v, read_raw, cal_order, suspect,
                                DEFAULT_VACC_DIV)
        _dv, dnu = shift_mhz_over_scan(
            cur.coeffs_v, fixed.coeffs_v,
            float(set_v.min()), float(set_v.max()),
            cooler_v=cooler_v, laser_cm=laser_cm,
            mass_amu=float(physics["mass_amu"]),
            harmonic=int(physics["harmonic"]))
        return float(dnu[len(dnu) // 2]), float(dnu.max() - dnu.min())
    except Exception:
        return None


class CalibrationAlertDialog(QDialog):
    """What is wrong with this run's calibration, and what it costs."""

    OPEN = 2      # user wants the full diagnostic

    def __init__(self, diag, cost, run_label, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Calibration warning — {run_label}")
        self.setMinimumWidth(540)
        style_dialog(self)
        lay = QVBoxLayout(self)

        n = len(diag.suspect)
        head = QLabel(
            f"<b>{run_label}</b>'s voltage calibration has "
            f"<b>{n} outlying point{'s' if n != 1 else ''}</b> "
            f"(index{'es' if n != 1 else ''} "
            f"{', '.join(str(i) for i in diag.suspect)}"
            f"{' — the start of the sweep' if set(diag.suspect) == set(range(n)) else ''}).")
        head.setWordWrap(True)
        lay.addWidget(head)

        body = ["They are still <b>included</b> — DENIS has not changed "
                "anything."]
        if cost is not None:
            mid, tilt = cost
            body.append(
                f"Leaving them in shifts this run's centroid by "
                f"<b>{mid:+.1f} MHz</b>")
            if abs(tilt) > 1.0:
                body.append(
                    f"and <b>tilts</b> its frequency axis by {tilt:.1f} MHz "
                    f"across the scan, so peaks move by different amounts "
                    f"depending where they sit.")
        else:
            body.append(
                f"Dropping them would move the calibration by up to "
                f"<b>{diag.impact_v:.3f} V</b> across the sweep.")
        txt = QLabel(" ".join(body))
        txt.setWordWrap(True)
        txt.setStyleSheet("padding: 6px 0;")
        lay.addWidget(txt)

        note = QLabel(
            "<span style='color:#888'>This warning will not appear again for "
            "this run.</span>")
        note.setWordWrap(True)
        lay.addWidget(note)

        btns = QDialogButtonBox()
        open_btn = btns.addButton("Open calibration…",
                                  QDialogButtonBox.ButtonRole.AcceptRole)
        btns.addButton("Keep as-is",
                       QDialogButtonBox.ButtonRole.RejectRole)
        open_btn.clicked.connect(lambda: self.done(self.OPEN))
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)


class CalibrationAlertBadge(QToolButton):
    """A blinking ``!`` shown only while a run's calibration needs triage.

    ``physics_fn`` and ``peers_fn`` are callables rather than values because
    the mass / harmonic / loaded-run list all change while the badge is alive.
    """

    def __init__(self, path_fn: Callable[[], Optional[str]], run_label: str,
                 physics_fn: Callable[[], dict],
                 peers_fn: Optional[Callable[[], list]] = None,
                 cal_order_fn: Optional[Callable[[], int]] = None,
                 parent=None):
        super().__init__(parent)
        # The path is resolved lazily, not captured. A file row is built before
        # its owner has finished constructing -- a virtual split does not know
        # its parent ASDF yet, and a merged entry has not yet been marked as
        # one -- so a path captured at construction time would key a split off
        # its .vasdf sidecar and put a calibration warning on a merged
        # spectrum that has no calibration at all. Returning None means
        # "nothing to warn about".
        self._path_fn = path_fn
        self._run_label = run_label
        self._physics_fn = physics_fn
        self._peers_fn = peers_fn or (lambda: [])
        self._cal_order_fn = cal_order_fn or (lambda: DEFAULT_ORDER)
        self._alerting = False
        self._diag = None
        self._path = None

        self.setText("!")
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(_ON_STYLE)
        self.clicked.connect(self._on_click)
        self.hide()

        get_registry().calibrations_changed.connect(self.refresh)
        blink_clock().tick.connect(self._on_tick)
        self.refresh()

    # ── state ────────────────────────────────────────────────

    def refresh(self):
        """Recompute whether this run still needs triage."""
        self._alerting = False
        try:
            self._path = self._path_fn()
            if self._path:
                reg = get_registry()
                self._diag = diagnose(
                    self._path, reg.to_dict(), run_number=self._run_label,
                    cal_order=self._cal_order_fn(),
                    acknowledged=reg.is_acknowledged(self._path))
                self._alerting = bool(self._diag.flagged)
        except Exception:
            self._alerting = False

        self.setVisible(self._alerting)
        if not self._alerting:
            return

        blink_clock().ensure_running()
        n = len(self._diag.suspect)
        self.setToolTip(
            f"{self._run_label}: {n} outlying calibration point"
            f"{'s' if n != 1 else ''} would move this run's centroid.\n"
            f"Nothing has been changed — click to see what it costs.")

    def _on_tick(self, on: bool):
        if not self._alerting:
            return
        self.setStyleSheet(_ON_STYLE if on else _OFF_STYLE)

    # ── click ────────────────────────────────────────────────

    def _on_click(self):
        if self._diag is None or not self._path:
            return
        reg = get_registry()
        cost = calibration_cost_mhz(
            self._path, reg.to_dict(), self._diag.suspect,
            self._physics_fn() or {}, self._cal_order_fn())

        dlg = CalibrationAlertDialog(self._diag, cost, self._run_label,
                                     parent=self.window())
        result = dlg.exec()

        # Informed is informed: stop blinking either way. A warning that keeps
        # nagging after it has been read is one the user learns to click past.
        reg.acknowledge(self._path)     # emits -> refresh() -> hides

        if result == CalibrationAlertDialog.OPEN:
            from gui.calibration_dialog import CalibrationDialog
            CalibrationDialog(
                self._path, run_label=self._run_label,
                cal_order=self._cal_order_fn(),
                physics=self._physics_fn(),
                peers=self._peers_fn(),
                parent=self.window()).exec()
