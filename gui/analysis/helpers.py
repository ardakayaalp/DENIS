"""Analysis-tab helper widgets and small UI utilities.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Reusable Qt building blocks for the Analysis tab: wheel-safe spin boxes,
a cursor-position-aware analysis spin box, a compact bounds-editing
button, factory functions for configured spin boxes, a scrollable
rich-text info dialog, a colored-swatch icon helper, and a non-modal
matplotlib popup window.

Depends on: standard library and third-party packages only (PySide6,
matplotlib Qt backend).
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QDoubleSpinBox, QSpinBox, QPushButton, QCheckBox,
    QHBoxLayout, QVBoxLayout, QDialog, QDialogButtonBox,
    QScrollArea, QLabel, QWidget, QApplication,
)
from PySide6.QtCore import Qt, Signal, QLocale, QTimer, QEvent
from PySide6.QtGui import QColor, QPixmap, QIcon


# ── Helpers ────────────────────────────────────────────────────────

def _color_icon(color_hex, size=16):
    pm = QPixmap(size, size)
    pm.fill(QColor(color_hex))
    return QIcon(pm)


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores wheel events unless focused."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores wheel events unless focused."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


def _make_double(value=0.0, min_val=-1e12, max_val=1e12, decimals=4,
                 step=1.0, suffix="", tooltip=""):
    sb = _NoScrollDoubleSpinBox()
    sb.setLocale(QLocale(QLocale.Language.English,
                         QLocale.Country.UnitedStates))
    sb.setRange(min_val, max_val)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(value)
    sb.setKeyboardTracking(False)
    sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def _make_int(value=0, min_val=0, max_val=99999, tooltip=""):
    sb = _NoScrollSpinBox()
    sb.setRange(min_val, max_val)
    sb.setValue(value)
    sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def format_value_error(value, err, unit="", digits=1):
    """Compact value(error) notation: ``-136.5(14) MHz``.

    The parenthesised error is in units of the last shown digit (the
    standard concise physics notation, so ``(14)`` on one decimal means
    ±1.4). ``err`` that is None / zero / non-finite falls back to the
    bare value; errors smaller than one last-digit unit floor at (1).
    """
    import math
    txt = f"{float(value):.{digits}f}"
    try:
        e = abs(float(err))
    except (TypeError, ValueError):
        e = 0.0
    if math.isfinite(e) and e > 0:
        txt += f"({max(int(round(e * 10 ** digits)), 1)})"
    return f"{txt} {unit}".strip() if unit else txt


# HFS parameters whose values are frequencies — their tracker /
# diagnostic axes get a "(MHz)" unit suffix.
_MHZ_PARAMS = {"centroid", "al", "au", "bl", "bu",
               "fwhm", "fwhm_g", "fwhm_l", "fwhmg", "fwhml"}


def param_axis_label(param):
    """Axis label for a fitted-parameter name: first letter
    capitalized, plus a ``(MHz)`` unit for known frequency-valued HFS
    parameters (centroid, A/B constants, FWHMs)."""
    p = str(param)
    label = p[:1].upper() + p[1:] if p else p
    if p.lower() in _MHZ_PARAMS:
        label += " (MHz)"
    return label


def _show_scrollable_info(parent, title, html_text):
    """Show an info dialog with scrollable rich-text content."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(520, 420)
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(8, 8, 8, 8)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    # Wrap with compact CSS to reduce spacing between sections
    styled = (
        "<style>"
        "h3 { margin-top: 0; margin-bottom: 4px; }"
        "h4 { margin-top: 8px; margin-bottom: 2px; }"
        "p { margin-top: 2px; margin-bottom: 2px; }"
        "</style>"
    ) + html_text.replace("<br><br>", "<br>")
    label = QLabel(styled)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setContentsMargins(10, 10, 10, 10)
    scroll.setWidget(label)
    lay.addWidget(scroll, 1)

    btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    btn_box.accepted.connect(dlg.accept)
    lay.addWidget(btn_box)
    dlg.exec()


# ── Custom spinbox with arrow-key support ──

class _AnalysisSpinBox(QDoubleSpinBox):
    """SpinBox with cursor-position-aware stepping and digit highlighting.

    - Enter: confirms value, stays in current field
    - Tab: default Qt focus navigation
    - Up/Down: step the digit at cursor position
    - Left/Right: move cursor between digits with highlight
    - Ctrl+C/V: explicit clipboard copy/paste
    """
    enter_pressed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.lineEdit().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.lineEdit() and event.type() == QEvent.Type.MouseButtonRelease:
            QTimer.singleShot(0, self._highlight_digit)
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        QTimer.singleShot(0, self._highlight_digit)

    def _find_digit_pos(self, pos=None):
        le = self.lineEdit()
        text = le.text()
        if pos is None:
            if le.hasSelectedText():
                pos = le.selectionStart()
            else:
                pos = le.cursorPosition()
        if not text:
            return 0
        if pos >= len(text):
            pos = len(text) - 1
        if 0 <= pos < len(text) and text[pos].isdigit():
            return pos
        for i in range(pos - 1, -1, -1):
            if text[i].isdigit():
                return i
        for i in range(pos + 1, len(text)):
            if text[i].isdigit():
                return i
        return max(0, min(pos, len(text) - 1))

    def _highlight_digit(self):
        le = self.lineEdit()
        text = le.text()
        if not text:
            return
        pos = self._find_digit_pos()
        if 0 <= pos < len(text) and text[pos].isdigit():
            le.setSelection(pos, 1)

    def _cursor_power(self):
        text = self.lineEdit().text()
        if not text:
            return 0
        pos = self._find_digit_pos()
        dot = text.find('.')
        if dot < 0:
            dot = len(text)
        if pos < dot:
            return dot - pos - 1
        else:
            return dot - pos

    def stepBy(self, steps):
        power = self._cursor_power()
        step_size = 10.0 ** power
        new_val = self.value() + steps * step_size
        new_val = round(new_val, self.decimals())
        new_val = max(self.minimum(), min(new_val, self.maximum()))
        self.setValue(new_val)
        new_text = self.lineEdit().text()
        new_dot = new_text.find('.')
        if new_dot < 0:
            new_dot = len(new_text)
        if power >= 0:
            new_pos = new_dot - power - 1
        else:
            new_pos = new_dot - power
        sign_off = 1 if new_text.startswith('-') else 0
        new_pos = max(sign_off, min(new_pos, len(new_text) - 1))
        if 0 <= new_pos < len(new_text) and new_text[new_pos].isdigit():
            self.lineEdit().setSelection(new_pos, 1)
        else:
            self.lineEdit().setCursorPosition(new_pos)

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.interpretText()
            return

        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_C:
            QApplication.clipboard().setText(str(self.value()))
            return
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_V:
            text = QApplication.clipboard().text().strip()
            try:
                self.setValue(float(text))
            except (ValueError, OverflowError):
                pass
            return
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_A:
            self.lineEdit().selectAll()
            return

        if key == Qt.Key.Key_Up:
            self.stepBy(1)
            return
        if key == Qt.Key.Key_Down:
            self.stepBy(-1)
            return

        if not mods and key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            le = self.lineEdit()
            text = le.text()
            pos = self._find_digit_pos()
            if key == Qt.Key.Key_Left:
                new_pos = pos - 1
                while new_pos >= 0 and not text[new_pos].isdigit():
                    new_pos -= 1
            else:
                new_pos = pos + 1
                while new_pos < len(text) and not text[new_pos].isdigit():
                    new_pos += 1
            if 0 <= new_pos < len(text) and text[new_pos].isdigit():
                le.setSelection(new_pos, 1)
            return

        super().keyPressEvent(event)


def _make_analysis_spin(value=0.0, lo=-1e12, hi=1e12, decimals=3,
                        step=1.0, tooltip=""):
    s = _AnalysisSpinBox()
    s.setLocale(QLocale(QLocale.Language.English,
                        QLocale.Country.UnitedStates))
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if tooltip:
        s.setToolTip(tooltip)
    return s


class _BoundsButton(QPushButton):
    """Compact button that stores min/max bounds and opens a popup to edit."""
    bounds_changed = Signal()

    def __init__(self, pmin=-1e12, pmax=1e12, parent=None):
        super().__init__(parent)
        self._min = pmin
        self._max = pmax
        self._min_enabled = (pmin > -1e11)
        self._max_enabled = (pmax < 1e11)
        self._update_display()
        self.clicked.connect(self._open_dialog)

    def _update_display(self):
        if self._min_enabled or self._max_enabled:
            lo = f"{self._min:.4g}" if self._min_enabled else "-\u221e"
            hi = f"{self._max:.4g}" if self._max_enabled else "\u221e"
            self.setText(f"[{lo}, {hi}]")
            self.setToolTip(f"Bounds: [{lo}, {hi}]")
        else:
            self.setText("None")
            self.setToolTip("No bounds (click to set)")

    def _open_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Parameter Bounds")
        dlg.setMinimumWidth(340)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        # Min row
        min_row = QHBoxLayout()
        min_row.setSpacing(8)
        self._dlg_min_cb = QCheckBox("Min:")
        self._dlg_min_cb.setChecked(self._min_enabled)
        self._dlg_min_cb.setMinimumHeight(28)
        min_row.addWidget(self._dlg_min_cb)
        self._dlg_min_spin = _make_analysis_spin(
            self._min if self._min_enabled else 0.0,
            -1e12, 1e12, 4, 1.0)
        self._dlg_min_spin.setMinimumHeight(28)
        self._dlg_min_spin.setEnabled(self._min_enabled)
        self._dlg_min_cb.toggled.connect(self._dlg_min_spin.setEnabled)
        min_row.addWidget(self._dlg_min_spin, 1)
        lay.addLayout(min_row)

        # Max row
        max_row = QHBoxLayout()
        max_row.setSpacing(8)
        self._dlg_max_cb = QCheckBox("Max:")
        self._dlg_max_cb.setChecked(self._max_enabled)
        self._dlg_max_cb.setMinimumHeight(28)
        max_row.addWidget(self._dlg_max_cb)
        self._dlg_max_spin = _make_analysis_spin(
            self._max if self._max_enabled else 0.0,
            -1e12, 1e12, 4, 1.0)
        self._dlg_max_spin.setMinimumHeight(28)
        self._dlg_max_spin.setEnabled(self._max_enabled)
        self._dlg_max_cb.toggled.connect(self._dlg_max_spin.setEnabled)
        max_row.addWidget(self._dlg_max_spin, 1)
        lay.addLayout(max_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._min_enabled = self._dlg_min_cb.isChecked()
            self._max_enabled = self._dlg_max_cb.isChecked()
            if self._min_enabled:
                self._min = self._dlg_min_spin.value()
            else:
                self._min = -1e12
            if self._max_enabled:
                self._max = self._dlg_max_spin.value()
            else:
                self._max = 1e12
            self._update_display()
            self.bounds_changed.emit()

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def is_min_enabled(self):
        return self._min_enabled

    def is_max_enabled(self):
        return self._max_enabled

    def set_bounds(self, pmin, pmax, min_enabled=None, max_enabled=None):
        self._min = pmin
        self._max = pmax
        if min_enabled is not None:
            self._min_enabled = min_enabled
        else:
            self._min_enabled = (pmin > -1e11)
        if max_enabled is not None:
            self._max_enabled = max_enabled
        else:
            self._max_enabled = (pmax < 1e11)
        self._update_display()


# ── Pop-up Plot Window ────────────────────────────────────────────

class PopupPlotWindow(QDialog):
    """Non-modal dialog that shows a matplotlib Figure inside a Qt canvas.

    Use this instead of ``plt.show()`` to avoid backend conflicts (TkAgg vs Qt).

    Usage::

        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        win = PopupPlotWindow(fig, title="Preview", parent=self)
        win.show()
    """

    def __init__(self, figure, title="Plot", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(int(figure.get_figwidth() * 100),
                     int(figure.get_figheight() * 100))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas, 1)
        figure.tight_layout()
