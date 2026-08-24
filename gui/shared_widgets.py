"""Shared GUI widgets, dialogs, and helpers for the DENIS app.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Reusable PySide6 building blocks used across the Estimate, Pre-Analysis,
Analysis, and Results tabs: scientific spin boxes, an application-wide
clipboard/undo event filter, undo-command bases with cross-tab focus and
highlight, a matplotlib plot editor dialog, a background estimation worker
thread, standalone tool dialogs (Schmidt moment, shell configuration, unit
converter, SHG / SFG-DFG calculators, quick plot), persistent settings and
last-used-directory helpers, plot-style defaults, and the status-bar manager.

Depends on: cls_estimations.schmidt, cls_estimations.plotting; cls_estimate
and gui.results_tab are imported lazily to keep startup light. Built on
PySide6, matplotlib, and periodictable.
"""

import sys
import os
import re
import tempfile
import yaml

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, NullLocator
from matplotlib.colors import to_hex
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QAbstractSpinBox,
    QComboBox, QCheckBox, QRadioButton, QButtonGroup, QPushButton,
    QPlainTextEdit, QScrollArea, QFileDialog, QMessageBox,
    QSizePolicy, QGridLayout,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QSlider, QColorDialog, QTabWidget, QToolButton, QToolTip,
    QTableWidget, QTableWidgetItem, QHeaderView, QApplication,
    QFontComboBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QLocale, QObject, QEvent
from PySide6.QtGui import (
    QFont, QValidator,
    QSyntaxHighlighter, QTextCharFormat, QColor,
    QPixmap, QIcon, QKeySequence, QUndoCommand,
)

from cls_estimations.schmidt import (
    ORBITALS, calculate_schmidt_moment, g_factor_from_mu_j, get_orbital,
)
from cls_estimations.plotting import PALETTES
# cls_estimate (run) is imported lazily inside WorkerThread.run so importing the
# GUI package doesn't pull the estimation backend (satlas2 chain) at startup.
import periodictable


# ── Icon helper ─────────────────────────────────────────────────
_ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "icons", "lucide")


def lucide_icon(name):
    """Load a Lucide SVG icon by name. Returns empty QIcon if missing."""
    path = os.path.join(_ICON_DIR, f"{name}.svg")
    return QIcon(path) if os.path.exists(path) else QIcon()


# ── Orbital names for combo boxes ────────────────────────────────
ORBITAL_NAMES = [orb["name"] for orb in ORBITALS]


# ── Nuclear shell-model helpers ──────────────────────────────────
def _get_element_symbol(Z):
    try:
        elem = periodictable.elements[Z]
        if elem is not None:
            return elem.symbol
    except (KeyError, IndexError):
        pass
    return "X"


def _fill_orbitals(num):
    remaining = num
    filled = []
    for orb in ORBITALS:
        occ = min(orb["capacity"], remaining)
        filled.append((orb, occ))
        remaining -= occ
    return filled


def _get_unpaired(filled):
    for orb, occ in reversed(filled):
        if occ % 2 == 1:
            j, l = orb["j"], orb["l"]
            parity = "+" if l % 2 == 0 else "-"
            jpi = (f"{int(j)}{parity}" if j.is_integer()
                   else f"{int(2*j)}/2{parity}")
            return j, l, parity, jpi
    return None


def _nordheim_rule(jp, lp, jn, ln):
    NN = (jp - lp) + (jn - ln)
    return abs(jp - jn) if abs(NN) < 1e-6 else jp + jn


def _format_configuration(filled):
    idx_partial = next(
        (i for i, (o, occ) in enumerate(filled)
         if 0 < occ < o["capacity"]), None)
    if idx_partial is None:
        inside = " ".join(
            f"({o['name']}){occ}" for o, occ in filled if occ > 0)
        return f"[ {inside} ]"
    inside = " ".join(
        f"({o['name']}){occ}" for o, occ in filled[:idx_partial]
        if occ > 0)
    outside = " ".join(
        f"({o['name']}){occ}" for o, occ in filled[idx_partial:]
        if occ > 0)
    return f"[ {inside} ] {outside}"


def _nuclear_shell_model(Z, N):
    p_filled = _fill_orbitals(Z)
    n_filled = _fill_orbitals(N)
    p_info = _get_unpaired(p_filled)
    n_info = _get_unpaired(n_filled)
    coupled = None
    if p_info is None and n_info is None:
        total_jpi = "0+"
    elif p_info is None or n_info is None:
        info = p_info or n_info
        total_jpi = info[3]
    else:
        jp, lp, pip, jpi_p = p_info
        jn, ln, pin, jpi_n = n_info
        J = _nordheim_rule(jp, lp, jn, ln)
        parity = "+" if pip == pin else "-"
        Jstr = f"{int(J)}" if J.is_integer() else f"{int(2*J)}/2"
        total_jpi = f"{Jstr}{parity}"
        coupled = f"coupled({jpi_p},{jpi_n})"
    return {
        "proton_filled": p_filled,
        "neutron_filled": n_filled,
        "proton_config_str": _format_configuration(p_filled),
        "neutron_config_str": _format_configuration(n_filled),
        "spin_parity": total_jpi,
        "coupled": coupled,
    }


def _name_to_latex(name, display_n_quantum=True):
    m = re.match(r"(\d+)([spdfghijk])(\d+)/2", name)
    if not m:
        return f"${name}$"
    n_str, l_str, j_num = m.groups()
    if display_n_quantum:
        return rf"${n_str}{l_str}_{{\frac{{{j_num}}}{{2}}}}$"
    return rf"${l_str}_{{\frac{{{j_num}}}{{2}}}}$"


# ── ScientificSpinBox ────────────────────────────────────────────
class ScientificSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that displays and accepts scientific notation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDecimals(10)
        self.setRange(-1e15, 1e15)
        self.setLocale(QLocale(QLocale.Language.English,
                               QLocale.Country.UnitedStates))

    def textFromValue(self, value):
        if value == 0.0:
            return "0.0"
        s = f"{value:.6g}"
        return s

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return 0.0

    def validate(self, text, pos):
        # Accept valid floats (including scientific notation)
        try:
            float(text)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            pass
        # Allow partial input like "1e", "1e+", "-", ""
        if text == "" or re.match(r'^[+-]?\d*\.?\d*([eE][+-]?\d*)?$', text):
            return (QValidator.State.Intermediate, text, pos)
        return (QValidator.State.Invalid, text, pos)


# ── Helper: create a labelled spinbox row ────────────────────────
def _make_double(value=0.0, min_val=-1e12, max_val=1e12, decimals=6,
                 step=1.0, suffix="", tooltip=""):
    sb = QDoubleSpinBox()
    sb.setLocale(QLocale(QLocale.Language.English,
                         QLocale.Country.UnitedStates))
    sb.setRange(min_val, max_val)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(value)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def _make_int(value=0, min_val=0, max_val=999, tooltip=""):
    sb = QSpinBox()
    sb.setRange(min_val, max_val)
    sb.setValue(value)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def _make_sci(value=0.0, min_val=0.0, max_val=1e15, tooltip=""):
    sb = ScientificSpinBox()
    sb.setRange(min_val, max_val)
    sb.setValue(value)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def count_errorbar_pairs(y, yerr):
    """Return a ``(lower, upper)`` pair for matplotlib ``errorbar``
    that never draws into negative counts.

    Histogram data is non-negative by construction: a bin can have 0,
    1, 2, … events but not −1. The symmetric Poisson error bar
    ``sqrt(y+1)`` (or ``sqrt(y)`` with the y=0 → 1 floor) reaches
    below zero for low-count bins, which is visually misleading.

    Clip the lower bar so ``y − lower ≥ 0``; the upper bar is
    unchanged so the *positive* uncertainty (which is physically
    meaningful) is preserved. For y=0 with the y=0 → 1 yerr floor,
    the bar runs from 0 to +1 instead of −1 to +1.

    Accepts scalar or array ``yerr``; broadcasts to the shape of ``y``.
    Returns a 2-row array suitable for ``ax.errorbar(..., yerr=...)``.
    """
    import numpy as _np
    y_arr = _np.asarray(y, dtype=float)
    yerr_arr = _np.asarray(yerr, dtype=float)
    if yerr_arr.ndim == 0:
        yerr_arr = _np.full_like(y_arr, float(yerr_arr))
    lower = _np.minimum(yerr_arr, _np.clip(y_arr, 0.0, None))
    return _np.vstack([lower, yerr_arr])


# ══════════════════════════════════════════════════════════════════
#  Application-level clipboard / undo filter
# ══════════════════════════════════════════════════════════════════
class _AppShortcutFilter(QObject):
    """Application-wide event filter for Ctrl+C/V/Z/Y and undo tracking.

    Plain QSpinBox / QDoubleSpinBox accept Copy/Paste only on selected
    text inside their inner line edit, which surprises users who expect
    Ctrl+C to copy the whole numeric value. This filter also tracks
    spinbox value changes to populate a global undo stack — newly added
    spinboxes are picked up automatically the first time they are
    focused, so dynamically created widgets (added isotopes, new
    analysis blocks, etc.) Just Work.

    Behaviour:
    - Ctrl+C on a spinbox with no selection -> copy str(value()).
    - Ctrl+V on a spinbox -> parse clipboard text as a number, setValue.
    - Spinbox FocusIn  -> snapshot current value.
    - Spinbox FocusOut -> if value changed, push an undo command.
    - Ctrl+Z / Ctrl+Y anywhere -> if the global stack has work, run it.
      We commit any pending change on the focused spinbox first so the
      "press Ctrl+Z right after editing" path always works.
    """

    def __init__(self, undo_stack=None, parent=None):
        super().__init__(parent)
        self._undo_stack = undo_stack

    def _spinbox_for(self, obj):
        """Resolve obj (or its parent) to a QAbstractSpinBox if any."""
        if isinstance(obj, QAbstractSpinBox):
            return obj
        if isinstance(obj, QLineEdit):
            parent = obj.parent()
            if isinstance(parent, QAbstractSpinBox):
                return parent
        return None

    def _commit_pending(self, sb):
        """Commit any change on ``sb`` since its last focus-in."""
        if sb is None or self._undo_stack is None:
            return
        old = getattr(sb, "_undo_prev", None)
        if old is None:
            return
        try:
            # Force any uncommitted text in the line edit into the value.
            sb.interpretText()
            new = sb.value()
        except RuntimeError:
            return  # widget already destroyed
        if new != old:
            self._undo_stack.push(_SpinBoxCommand(sb, old, new))
            sb._undo_prev = new

    def eventFilter(self, obj, event):
        et = event.type()

        # ── Track focus on spinboxes for undo snapshots ──
        if et == QEvent.Type.FocusIn:
            sb = self._spinbox_for(obj)
            if sb is not None:
                try:
                    sb._undo_prev = sb.value()
                except RuntimeError:
                    pass
            return False

        if et == QEvent.Type.FocusOut:
            sb = self._spinbox_for(obj)
            if sb is not None:
                self._commit_pending(sb)
            return False

        if et != QEvent.Type.KeyPress:
            return False

        is_undo = event.matches(QKeySequence.StandardKey.Undo)
        is_redo = event.matches(QKeySequence.StandardKey.Redo)
        if (is_undo or is_redo) and self._undo_stack is not None:
            # Commit any in-flight spinbox edit so it is undoable.
            sb = self._spinbox_for(obj)
            if sb is not None:
                self._commit_pending(sb)
            if is_undo and self._undo_stack.canUndo():
                self._undo_stack.undo()
                # Re-sync the focused spinbox snapshot so the next edit
                # measures the diff against the post-undo value.
                if sb is not None:
                    try:
                        sb._undo_prev = sb.value()
                    except RuntimeError:
                        pass
                return True
            if is_redo and self._undo_stack.canRedo():
                self._undo_stack.redo()
                if sb is not None:
                    try:
                        sb._undo_prev = sb.value()
                    except RuntimeError:
                        pass
                return True

        sb = self._spinbox_for(obj)
        if sb is None:
            return False

        if event.matches(QKeySequence.StandardKey.Copy):
            le = sb.lineEdit() if hasattr(sb, "lineEdit") else None
            if le is not None and le.hasSelectedText():
                return False  # default copy of the highlighted text
            QApplication.clipboard().setText(sb.text())
            return True

        if event.matches(QKeySequence.StandardKey.Paste):
            text = QApplication.clipboard().text().strip()
            if not text:
                return True
            try:
                num = float(text)
            except (ValueError, OverflowError):
                return True  # swallow, but don't crash
            if isinstance(sb, QSpinBox):
                sb.setValue(int(round(num)))
            else:
                sb.setValue(num)
            return True

        return False


# ══════════════════════════════════════════════════════════════════
#  Undo command for spinbox value changes
# ══════════════════════════════════════════════════════════════════
def _SpinBoxCommand(widget, old_val, new_val):
    """Build a QUndoCommand that flips a spinbox between two values."""
    from PySide6.QtGui import QUndoCommand

    label = ""
    if widget.accessibleName():
        label = widget.accessibleName()
    elif widget.toolTip():
        label = widget.toolTip()[:40]

    class _Cmd(QUndoCommand):
        def __init__(self):
            super().__init__(f"Change {label}" if label else "Change value")
            self._widget = widget
            self._old = old_val
            self._new = new_val
            self._first_redo = True

        def redo(self):
            # Skip the very first redo — Qt invokes redo() when the
            # command is pushed, but the spinbox already holds _new.
            if self._first_redo:
                self._first_redo = False
                return
            self._widget.blockSignals(True)
            self._widget.setValue(self._new)
            self._widget.blockSignals(False)

        def undo(self):
            self._widget.blockSignals(True)
            self._widget.setValue(self._old)
            self._widget.blockSignals(False)

    return _Cmd()


# ══════════════════════════════════════════════════════════════════
#  UI-wide undo/redo: cross-tab focus + highlight
# ══════════════════════════════════════════════════════════════════

def _widget_alive(w):
    """True if the C++ side of a Qt widget still exists."""
    if w is None:
        return False
    try:
        w.isVisible()      # touches the C++ object
        return True
    except RuntimeError:
        return False


def flash_widget(widget, pulses=3, interval=170):
    """Non-destructive attention pulse on ``widget``.

    Toggles a QGraphicsColorizeEffect on/off a few times then removes it.
    Doesn't touch the widget's stylesheet, so it can't clobber existing
    styling (FileEntry selection style, block borders, …)."""
    from PySide6.QtWidgets import QGraphicsColorizeEffect
    from PySide6.QtCore import QTimer
    if not _widget_alive(widget):
        return
    try:
        eff = QGraphicsColorizeEffect(widget)
        eff.setColor(QColor("#FFC83D"))
        eff.setStrength(0.85)
        widget.setGraphicsEffect(eff)
    except RuntimeError:
        return
    state = {"i": 0}
    timer = QTimer(widget)

    def _tick():
        if not _widget_alive(widget):
            timer.stop()
            return
        state["i"] += 1
        if state["i"] >= pulses * 2:
            timer.stop()
            try:
                widget.setGraphicsEffect(None)
            except RuntimeError:
                pass
            return
        eff.setStrength(0.0 if state["i"] % 2 else 0.85)

    timer.timeout.connect(_tick)
    timer.start(interval)


def focus_and_highlight(widget):
    """Surface ``widget`` and pulse a highlight on it.

    For every QTabWidget ancestor, switches to the page that contains the
    widget -- so an undo/redo targeting a different top-level tab (Estimate
    / Pre-Analysis / Analysis / Results) AND a different sub-tab (a specific
    project) is revealed automatically. Then scrolls it into view inside
    any QScrollArea and flashes it so the user sees what changed."""
    if not _widget_alive(widget):
        return
    chain = []
    anc = widget
    while anc is not None:
        chain.append(anc)
        anc = anc.parentWidget()
    for node in chain:
        if isinstance(node, QTabWidget):
            for idx in range(node.count()):
                page = node.widget(idx)
                if page is widget or (page is not None
                                      and page.isAncestorOf(widget)):
                    node.setCurrentIndex(idx)
                    break
    anc = widget.parentWidget()
    while anc is not None:
        if isinstance(anc, QScrollArea):
            try:
                anc.ensureWidgetVisible(widget, 50, 50)
            except Exception:
                pass
            break
        anc = anc.parentWidget()
    flash_widget(widget)


# Lines that must keep their own row when a hand-wrapped tooltip is
# reflowed: bullet/numbered items and "name = ..." definition lines
# (equations, mode glossaries like "pbp = point-by-point: ...").
_TIP_STRUCT_RE = None


def reflow_tooltip(tip):
    """Reflow a hand-wrapped PLAIN-text tooltip into rich text, or None.

    Source tooltips are authored with hard ``\\n`` breaks at arbitrary
    widths, which Qt renders literally (a tall ragged column). This joins
    continuation lines back into flowing paragraphs and returns Qt rich
    text, whose tooltip label word-wraps at a sensible width natively.

    Kept as rows rather than joined: blank-line paragraph breaks,
    bullet/numbered lines, ``name = ...`` definition lines, and a line
    that starts a new sentence after terminal punctuation (so authored
    "Pick this for: ..." conventions survive). Rich-text tooltips and
    short one-liners return None (leave Qt's default handling alone).
    """
    global _TIP_STRUCT_RE
    if not tip or "<" in tip:
        return None
    if "\n" not in tip and len(tip) <= 88:
        return None
    import html as _html
    import re as _re
    if _TIP_STRUCT_RE is None:
        _TIP_STRUCT_RE = _re.compile(
            r"^\s*(?:[-•*·]\s|\d+[.)]\s|[A-Za-z0-9_()\[\]/+±·]+\s*=)")
    blocks = []
    for para in tip.split("\n\n"):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        rows = []
        for ln in lines:
            new_sentence = (rows
                            and rows[-1][-1:] in ".:;?!"
                            and ln[:1].isupper())
            if rows and not _TIP_STRUCT_RE.match(ln) and not new_sentence:
                rows[-1] += " " + ln
            else:
                rows.append(ln)
        blocks.append("<br>".join(_html.escape(r) for r in rows))
    if not blocks:
        return None
    return "<qt>" + "<br><br>".join(blocks) + "</qt>"


class _TooltipWrapFilter(QObject):
    """App-wide filter that renders every tooltip as a wrapped rich-text box.

    Two paths:
    - Widget tooltips (``setToolTip``): reflowed via :func:`reflow_tooltip`.
    - Item tooltips (``ToolTipRole`` on combo popups, tables, trees): those
      are delivered to the view's *viewport* and never consult the widget
      tooltip, so they are looked up via ``indexAt`` and reflowed the same
      way — previously they bypassed wrapping entirely and rendered the
      authored hard line breaks verbatim.
    """

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.ToolTip:
            return False
        # Item-view path: obj is the viewport of a QAbstractItemView.
        try:
            from PySide6.QtWidgets import QAbstractItemView
            view = obj.parent() if isinstance(obj, QWidget) else None
            if (isinstance(view, QAbstractItemView)
                    and obj is view.viewport()):
                index = view.indexAt(event.pos())
                tip = index.data(Qt.ItemDataRole.ToolTipRole) if (
                    index.isValid()) else None
                if isinstance(tip, str):
                    wrapped = reflow_tooltip(tip)
                    if wrapped is not None:
                        QToolTip.showText(event.globalPos(), wrapped, obj)
                        return True
                return False
        except Exception:
            pass
        # Widget path.
        try:
            tip = obj.toolTip()
        except Exception:
            return False
        wrapped = reflow_tooltip(tip)
        if wrapped is None:
            return False
        widget = obj if isinstance(obj, QWidget) else None
        try:
            QToolTip.showText(event.globalPos(), wrapped, widget)
        except Exception:
            return False
        return True


class AppUndoCommand(QUndoCommand):
    """Base for app-wide undoable actions with cross-tab focus + highlight.

    Subclasses implement ``_apply_redo(initial)`` and ``_apply_undo()`` and
    call ``self.reveal(widget)`` to switch tabs + flash the affected
    widget. The FIRST redo (when Qt pushes the command) applies WITHOUT
    revealing, because the user just performed the action on the current
    tab; later redo()s (after an undo) reveal like undo does."""

    def __init__(self, text):
        super().__init__(text)
        self._first_redo = True

    def redo(self):
        first = self._first_redo
        self._first_redo = False
        self._apply_redo(first)

    def undo(self):
        self._apply_undo()

    @staticmethod
    def reveal(widget):
        focus_and_highlight(widget)

    def _apply_redo(self, initial):
        raise NotImplementedError

    def _apply_undo(self):
        raise NotImplementedError


# ── Log syntax highlighter ────────────────────────────────────────

def _fmt(color, bold=False, italic=False):
    """Create a QTextCharFormat with the given style."""
    f = QTextCharFormat()
    f.setForeground(QColor(color))
    if bold:
        f.setFontWeight(QFont.Weight.Bold)
    if italic:
        f.setFontItalic(True)
    return f


class LogHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for CLS run-time estimation log output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Build rules as (compiled_regex, format) pairs.
        # Order matters: later rules can override earlier ones on overlap,
        # but we apply them first-match-wins per character position.
        self._rules = []

        # ── Separator bars ────────────────────────────────
        self._rules.append(
            (re.compile(r'^[=]{4,}$'), _fmt("#90a4ae")))           # silver
        self._rules.append(
            (re.compile(r'^[-]{4,}$'), _fmt("#b0bec5")))           # light silver

        # ── Table border lines (+---+---+) ────────────────
        self._rules.append(
            (re.compile(r'^\+[-+]+\+$'), _fmt("#b0bec5")))         # light silver

        # ── Section titles (all-caps between === bars) ────
        self._rules.append(
            (re.compile(r'^\s+(NUCLEAR PARAMETERS.*|GRAND TOTAL.*)'),
             _fmt("#42a5f5", bold=True)))                            # bright blue

        # ── Main title line ───────────────────────────────
        self._rules.append(
            (re.compile(r'^\s*CLS Estimation Toolkit.*'),
             _fmt("#42a5f5", bold=True)))                            # bright blue

        # ── Isotope header lines (e.g. "59Co (I=3.5) | Yield: ...") ──
        self._rules.append(
            (re.compile(r'^\d+\w+\s+\(.*\)\s*\|'),
             _fmt("#66bb6a", bold=True)))                            # bright green

        # ── Global parameter keywords ─────────────────────
        kw_pattern = r'^(Element:|Laser:|Beam:|Sigma required:|FWHM:)'
        self._rules.append(
            (re.compile(kw_pattern), _fmt("#26c6da", bold=True)))   # bright cyan

        # ── Isotope detail keywords ───────────────────────
        detail_kw = (r'^\s*(Background:|Override:|Single peak|'
                     r'A_l=|A_u=|B_l=|B_u=|Reference:|'
                     r'mu =|Q =|A_lower|A_upper|B_lower|B_upper)')
        self._rules.append(
            (re.compile(detail_kw), _fmt("#ffa726", bold=True)))    # bright orange

        # ── Table header rows (pretty format with |) ──────
        self._rules.append(
            (re.compile(r'^\|\s*#\s*\|.*Offset.*\|'),
             _fmt("#78909c", bold=True)))                            # blue-gray
        self._rules.append(
            (re.compile(r'^\|\s*Isotope\s*\|'),
             _fmt("#78909c", bold=True)))                            # blue-gray

        # ── Subtotal lines ────────────────────────────────
        self._rules.append(
            (re.compile(r'^\s*Subtotal:.*'),
             _fmt("#ce93d8", bold=True)))                            # bright purple

        # ── Table section headers ─────────────────────────
        self._rules.append(
            (re.compile(r'^HFS Constants.*'),
             _fmt("#78909c", bold=True)))                            # blue-gray

        # ── File output paths ─────────────────────────────
        self._rules.append(
            (re.compile(r'^(Log saved:|Plots saved:|Spectra saved:|Peak list:).*'),
             _fmt("#a1b2c3", italic=True)))                          # soft blue-gray

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            m = pattern.search(text)
            if m:
                self.setFormat(0, len(text), fmt)
                return  # first full-line match wins


# ══════════════════════════════════════════════════════════════════
#  Plot Editor Dialog
# ══════════════════════════════════════════════════════════════════

LEGEND_LOCATIONS = [
    "best", "upper right", "upper left", "lower left",
    "lower right", "right", "center left", "center right",
    "lower center", "upper center", "center",
]

LINESTYLE_MAP = [
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dash-dot", "-."),
    ("Dotted", ":"),
    ("None", "None"),
]

MARKER_MAP = [
    ("None", "None"),
    ("Point", "."),
    ("Circle", "o"),
    ("Square", "s"),
    ("Diamond", "D"),
    ("Triangle up", "^"),
    ("Triangle down", "v"),
    ("Star", "*"),
    ("Plus", "+"),
    ("Cross", "x"),
]


class PlotEditorDialog(QDialog):
    """Modeless dialog for editing matplotlib plot properties in real-time."""

    def __init__(self, figure, canvas, parent=None, extra_figures=None,
                 source_path="", arrow_registry=None,
                 on_position_changed=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Editor")
        self.resize(520, 700)
        self.setModal(False)

        # Support multiple figures: list of (name, figure, canvas)
        self._figures = [("Spectrum", figure, canvas)]
        if extra_figures:
            self._figures.extend(extra_figures)
        self._figure = figure
        self._canvas = canvas
        self._source_path = source_path
        self._pick_cid = None
        self._line_artists = []
        self._updating = False
        # Drag-to-move state for text / annotation artists
        self._drag_press_cid = None
        self._drag_motion_cid = None
        self._drag_release_cid = None
        self._dragging = None  # dict with artist + start positions
        # Optional linkage for the isotope-shift δν artists so that
        # FancyArrowPatch endpoints (which live in ax.patches, not
        # ax.texts) are independently grabbable, a label drag also
        # moves its linked arrow tail (matched by a shared ``_is_id``),
        # and drags are persisted by the owning tab.
        registry = arrow_registry or {}
        self._labels_by_id = registry.get("labels", {}) or {}
        self._arrows_by_id = registry.get("arrows", {}) or {}
        self._on_position_changed = on_position_changed
        # When an arrow endpoint is grabbed: the artist + which end.
        self._drag_target = None
        self._drag_kind = None  # "tail" | "head" | "text"

        main_layout = QVBoxLayout(self)

        # Figure selector (only shown if multiple figures)
        if len(self._figures) > 1:
            fig_sel_row = QHBoxLayout()
            fig_sel_row.addWidget(QLabel("<b>Plot:</b>"))
            self._fig_selector = QComboBox()
            for name, _, _ in self._figures:
                self._fig_selector.addItem(name)
            self._fig_selector.currentIndexChanged.connect(
                self._on_figure_selected)
            fig_sel_row.addWidget(self._fig_selector, 1)
            main_layout.addLayout(fig_sel_row)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        tabs.addTab(self._build_figure_tab(), "Figure")
        tabs.addTab(self._build_axes_tab(), "Axes")
        tabs.addTab(self._build_lines_tab(), "Artists")
        tabs.addTab(self._build_add_tab(), "Add")
        tabs.addTab(self._build_legend_tab(), "Legend")
        tabs.addTab(self._build_export_tab(), "Export")

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self._on_close)
        main_layout.addWidget(btn_box)

        self._populate_axes_selector()
        for i in range(len(self._figure.axes)):
            self._lines_axes_selector.addItem(f"Axes {i + 1}")
            self._legend_axes_selector.addItem(f"Axes {i + 1}")
            self._add_axes_selector.addItem(f"Axes {i + 1}")
        self._populate_lines()
        self._init_undo()

    # ── Undo / redo (Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z) ──────────────

    def _init_undo(self):
        """Snapshot-based undo. Every canvas redraw (each edit ends in
        one) schedules a debounced state capture; Ctrl+Z restores the
        previous snapshot. Covers property, position, colour, font and
        add/remove edits (add/remove via the _pe_added tag)."""
        from PySide6.QtGui import QShortcut, QKeySequence
        from PySide6.QtCore import QTimer
        self._undo_stack = []
        self._redo_stack = []
        self._restoring = False
        self._undo_timer = QTimer(self)
        self._undo_timer.setSingleShot(True)
        self._undo_timer.setInterval(250)
        self._undo_timer.timeout.connect(self._commit_state)
        # Hook every canvas's draw_idle so any edit schedules a commit.
        for _name, _fig, canvas in self._figures:
            orig = canvas.draw_idle

            def _wrapped(*a, _orig=orig, **k):
                if not self._restoring:
                    self._undo_timer.start()
                return _orig(*a, **k)
            canvas.draw_idle = _wrapped
        # Baseline snapshot of the initial figure.
        self._undo_stack.append(self._capture_state())
        QShortcut(QKeySequence.StandardKey.Undo, self,
                  activated=self._undo)
        QShortcut(QKeySequence.StandardKey.Redo, self,
                  activated=self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self,
                  activated=self._redo)

    def _capture_state(self):
        """Compact, self-contained snapshot of the active figure's
        editable state (no dependency on results_tab)."""
        from matplotlib.text import Annotation
        fig = self._figure
        snap = {"figsize": [fig.get_figwidth(), fig.get_figheight()],
                "axes": []}
        for ax in fig.axes:
            a = {
                "title": ax.get_title(),
                "title_size": float(ax.title.get_fontsize()),
                "title_pos": list(ax.title.get_position()),
                "xlabel": ax.get_xlabel(),
                "xlabel_size": float(ax.xaxis.label.get_fontsize()),
                "ylabel": ax.get_ylabel(),
                "ylabel_size": float(ax.yaxis.label.get_fontsize()),
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
                "xscale": ax.get_xscale(),
                "yscale": ax.get_yscale(),
                "lines": [], "texts": [],
            }
            for ln in ax.get_lines():
                a["lines"].append({
                    "color": to_hex(ln.get_color()),
                    "lw": float(ln.get_linewidth()),
                    "ls": ln.get_linestyle(),
                    "marker": str(ln.get_marker()),
                    "ms": float(ln.get_markersize()),
                    "alpha": ln.get_alpha(),
                    "visible": ln.get_visible(),
                    "zorder": float(ln.get_zorder()),
                    "pe_added": bool(getattr(ln, "_pe_added", False)),
                })
            for t in ax.texts:
                fam = t.get_fontfamily()
                a["texts"].append({
                    "text": t.get_text(),
                    "color": to_hex(t.get_color()),
                    "fontsize": float(t.get_fontsize()),
                    "position": list(t.get_position()),
                    "rotation": float(t.get_rotation()),
                    "family": fam[0] if fam else "",
                    "weight": t.get_fontweight(),
                    "style": t.get_fontstyle(),
                    "visible": t.get_visible(),
                    "is_annotation": isinstance(t, Annotation),
                    "xy": (list(t.xy) if isinstance(t, Annotation)
                           else None),
                    "pe_added": bool(getattr(t, "_pe_added", False)),
                })
            snap["axes"].append(a)
        return snap

    def _apply_state(self, snap):
        """Restore a snapshot captured by _capture_state."""
        fig = self._figure
        try:
            fig.set_size_inches(*snap.get("figsize",
                                          fig.get_size_inches()))
        except Exception:
            pass
        for ax, a in zip(fig.axes, snap.get("axes", [])):
            ax.set_title(a["title"], fontsize=a["title_size"])
            ax.title.set_position(a["title_pos"])
            ax.set_xlabel(a["xlabel"], fontsize=a["xlabel_size"])
            ax.set_ylabel(a["ylabel"], fontsize=a["ylabel_size"])
            try:
                ax.set_xscale(a["xscale"])
                ax.set_yscale(a["yscale"])
            except Exception:
                pass
            ax.set_xlim(a["xlim"])
            ax.set_ylim(a["ylim"])
            # Lines: drop editor-added extras beyond the snapshot, then
            # restore properties positionally.
            lines = ax.get_lines()
            saved = a["lines"]
            for extra in lines[len(saved):]:
                if getattr(extra, "_pe_added", False):
                    try:
                        extra.remove()
                    except Exception:
                        pass
            for ln, sl in zip(ax.get_lines(), saved):
                ln.set_color(sl["color"])
                ln.set_linewidth(sl["lw"])
                ln.set_linestyle(sl["ls"])
                ln.set_marker(sl["marker"])
                ln.set_markersize(sl["ms"])
                ln.set_alpha(sl["alpha"])
                ln.set_visible(sl["visible"])
                ln.set_zorder(sl["zorder"])
            # Texts: same drop-extras-then-restore approach.
            texts = list(ax.texts)
            saved_t = a["texts"]
            for extra in texts[len(saved_t):]:
                if getattr(extra, "_pe_added", False):
                    try:
                        extra.remove()
                    except Exception:
                        pass
            for t, st in zip(list(ax.texts), saved_t):
                try:
                    t.set_text(st["text"])
                    t.set_color(st["color"])
                    t.set_fontsize(st["fontsize"])
                    t.set_position(st["position"])
                    t.set_rotation(st["rotation"])
                    if st["family"]:
                        t.set_fontfamily(st["family"])
                    t.set_fontweight(st["weight"])
                    t.set_fontstyle(st["style"])
                    t.set_visible(st["visible"])
                    if st["is_annotation"] and st["xy"] is not None:
                        t.xy = tuple(st["xy"])
                except Exception:
                    pass

    def _commit_state(self):
        snap = self._capture_state()
        if self._undo_stack and snap == self._undo_stack[-1]:
            return   # nothing actually changed
        self._undo_stack.append(snap)
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self):
        # Flush any pending capture so the stack top is current.
        if self._undo_timer.isActive():
            self._undo_timer.stop()
            self._commit_state()
        if len(self._undo_stack) < 2:
            return
        self._redo_stack.append(self._undo_stack.pop())
        self._restoring = True
        try:
            self._apply_state(self._undo_stack[-1])
            self._canvas.draw()
        finally:
            self._restoring = False
        self._populate_lines()

    def _redo(self):
        if not self._redo_stack:
            return
        snap = self._redo_stack.pop()
        self._undo_stack.append(snap)
        self._restoring = True
        try:
            self._apply_state(snap)
            self._canvas.draw()
        finally:
            self._restoring = False
        self._populate_lines()

    def _on_figure_selected(self, index):
        """Switch the active figure/canvas when the user picks a different plot."""
        if index < 0 or index >= len(self._figures):
            return
        _, fig, canvas = self._figures[index]
        # Disconnect pick from old canvas
        if self._pick_cid is not None:
            try:
                self._canvas.mpl_disconnect(self._pick_cid)
            except Exception:
                pass
            self._pick_cid = None
        self._figure = fig
        self._canvas = canvas
        # Refresh all selectors
        self._updating = True
        self._fig_width_spin.setValue(fig.get_figwidth())
        self._fig_height_spin.setValue(fig.get_figheight())
        self._fig_dpi_spin.setValue(int(fig.get_dpi()))
        self._updating = False
        self._populate_axes_selector()
        self._lines_axes_selector.blockSignals(True)
        self._lines_axes_selector.clear()
        for i in range(len(fig.axes)):
            self._lines_axes_selector.addItem(f"Axes {i + 1}")
        self._lines_axes_selector.blockSignals(False)
        self._legend_axes_selector.blockSignals(True)
        self._legend_axes_selector.clear()
        for i in range(len(fig.axes)):
            self._legend_axes_selector.addItem(f"Axes {i + 1}")
        self._legend_axes_selector.blockSignals(False)
        self._add_axes_selector.blockSignals(True)
        self._add_axes_selector.clear()
        for i in range(len(fig.axes)):
            self._add_axes_selector.addItem(f"Axes {i + 1}")
        self._add_axes_selector.blockSignals(False)
        self._cancel_click_place()
        self._populate_lines()
        # Undo history is per-figure; reset the baseline on a switch.
        if hasattr(self, "_undo_stack"):
            self._undo_stack = [self._capture_state()]
            self._redo_stack.clear()

    # ── Font preview (shared by Axes / Artists / Add tabs) ────────

    @staticmethod
    def _new_font_preview():
        lbl = QLabel()
        lbl.setObjectName("fontPreview")
        lbl.setMinimumHeight(30)
        lbl.setStyleSheet(
            "QLabel#fontPreview { border: 1px solid #555; "
            "border-radius: 3px; padding: 2px 6px; "
            "background: rgba(255,255,255,0.04); }")
        return lbl

    @staticmethod
    def _set_font_preview(label, family, size_pt, bold, italic, text):
        """Render ``text`` in the given typeface so the user sees the
        actual look before committing. Preview point size is clamped so
        a 72 pt label can't blow the row height."""
        f = QFont(family)
        f.setPointSizeF(min(max(float(size_pt), 6.0), 22.0))
        f.setBold(bool(bold))
        f.setItalic(bool(italic))
        label.setFont(f)
        label.setText(text or "AaBbGg 0123")

    @staticmethod
    def _weight_is_bold(weight):
        """matplotlib weight may be a name or a 0–1000 number."""
        if isinstance(weight, (int, float)):
            return weight >= 600
        return str(weight).lower() in (
            "bold", "semibold", "demibold", "heavy", "black", "extra bold")

    # ── Figure tab ────────────────────────────────────────────────

    def _build_figure_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # Canvas size group
        size_grp = QGroupBox("Canvas Size")
        size_layout = QVBoxLayout(size_grp)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Unit:"))
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["in", "mm", "px"])
        unit_row.addWidget(self._unit_combo)
        unit_row.addStretch()
        size_layout.addLayout(unit_row)

        form = QFormLayout()
        self._fig_width_spin = QDoubleSpinBox()
        self._fig_width_spin.setRange(0.5, 30)
        self._fig_width_spin.setSingleStep(0.5)
        self._fig_width_spin.setDecimals(1)
        self._fig_width_spin.setValue(self._figure.get_figwidth())
        form.addRow("Width:", self._fig_width_spin)

        self._fig_height_spin = QDoubleSpinBox()
        self._fig_height_spin.setRange(0.5, 30)
        self._fig_height_spin.setSingleStep(0.5)
        self._fig_height_spin.setDecimals(1)
        self._fig_height_spin.setValue(self._figure.get_figheight())
        form.addRow("Height:", self._fig_height_spin)

        self._fig_dpi_spin = QSpinBox()
        self._fig_dpi_spin.setRange(72, 600)
        self._fig_dpi_spin.setSingleStep(10)
        self._fig_dpi_spin.setValue(int(self._figure.get_dpi()))
        form.addRow("Resolution (DPI):", self._fig_dpi_spin)

        size_layout.addLayout(form)
        layout.addWidget(size_grp)

        # Subplot margins group
        pad_grp = QGroupBox("Subplot Margins (fraction of figure)")
        pad_layout = QVBoxLayout(pad_grp)
        pad_form = QFormLayout()

        self._pad_left = QDoubleSpinBox()
        self._pad_left.setRange(0, 0.49)
        self._pad_left.setSingleStep(0.01)
        self._pad_left.setDecimals(2)
        pad_form.addRow("Left:", self._pad_left)

        self._pad_right = QDoubleSpinBox()
        self._pad_right.setRange(0.51, 1.0)
        self._pad_right.setSingleStep(0.01)
        self._pad_right.setDecimals(2)
        pad_form.addRow("Right:", self._pad_right)

        self._pad_bottom = QDoubleSpinBox()
        self._pad_bottom.setRange(0, 0.49)
        self._pad_bottom.setSingleStep(0.01)
        self._pad_bottom.setDecimals(2)
        pad_form.addRow("Bottom:", self._pad_bottom)

        self._pad_top = QDoubleSpinBox()
        self._pad_top.setRange(0.51, 1.0)
        self._pad_top.setSingleStep(0.01)
        self._pad_top.setDecimals(2)
        pad_form.addRow("Top:", self._pad_top)

        pad_layout.addLayout(pad_form)
        tight_btn = QPushButton("Auto (Tight Layout)")
        tight_btn.clicked.connect(self._apply_tight_layout)
        pad_layout.addWidget(tight_btn)
        layout.addWidget(pad_grp)

        layout.addStretch()

        self._sync_padding_from_figure()

        # Connect signals after initial values are set
        self._fig_width_spin.valueChanged.connect(
            self._on_figure_size_changed)
        self._fig_height_spin.valueChanged.connect(
            self._on_figure_size_changed)
        self._fig_dpi_spin.valueChanged.connect(self._on_dpi_changed)
        self._unit_combo.currentTextChanged.connect(self._on_unit_changed)
        self._pad_left.valueChanged.connect(self._on_padding_changed)
        self._pad_right.valueChanged.connect(self._on_padding_changed)
        self._pad_bottom.valueChanged.connect(self._on_padding_changed)
        self._pad_top.valueChanged.connect(self._on_padding_changed)
        return w

    def _to_display(self, inches):
        """Convert inches to current display unit."""
        unit = self._unit_combo.currentText()
        if unit == "mm":
            return inches * 25.4
        elif unit == "px":
            return inches * self._fig_dpi_spin.value()
        return inches

    def _from_display(self, value):
        """Convert current display unit to inches."""
        unit = self._unit_combo.currentText()
        if unit == "mm":
            return value / 25.4
        elif unit == "px":
            return value / max(self._fig_dpi_spin.value(), 1)
        return value

    def _on_unit_changed(self, _text=None):
        self._updating = True
        w_in = self._figure.get_figwidth()
        h_in = self._figure.get_figheight()
        unit = self._unit_combo.currentText()
        if unit == "in":
            self._fig_width_spin.setDecimals(1)
            self._fig_width_spin.setSingleStep(0.5)
            self._fig_width_spin.setRange(0.5, 30)
            self._fig_height_spin.setDecimals(1)
            self._fig_height_spin.setSingleStep(0.5)
            self._fig_height_spin.setRange(0.5, 30)
        elif unit == "mm":
            self._fig_width_spin.setDecimals(0)
            self._fig_width_spin.setSingleStep(5)
            self._fig_width_spin.setRange(10, 800)
            self._fig_height_spin.setDecimals(0)
            self._fig_height_spin.setSingleStep(5)
            self._fig_height_spin.setRange(10, 800)
        elif unit == "px":
            self._fig_width_spin.setDecimals(0)
            self._fig_width_spin.setSingleStep(10)
            self._fig_width_spin.setRange(50, 10000)
            self._fig_height_spin.setDecimals(0)
            self._fig_height_spin.setSingleStep(10)
            self._fig_height_spin.setRange(50, 10000)
        self._fig_width_spin.setValue(self._to_display(w_in))
        self._fig_height_spin.setValue(self._to_display(h_in))
        self._updating = False

    def _on_figure_size_changed(self):
        if self._updating:
            return
        w_in = self._from_display(self._fig_width_spin.value())
        h_in = self._from_display(self._fig_height_spin.value())
        self._figure.set_size_inches(w_in, h_in)
        dpi = self._figure.get_dpi()
        self._canvas.setFixedSize(int(w_in * dpi), int(h_in * dpi))
        self._canvas.draw_idle()

    def _on_dpi_changed(self, val):
        if self._updating:
            return
        self._figure.set_dpi(val)
        w_in = self._figure.get_figwidth()
        h_in = self._figure.get_figheight()
        self._canvas.setFixedSize(int(w_in * val), int(h_in * val))
        if self._unit_combo.currentText() == "px":
            self._updating = True
            self._fig_width_spin.setValue(w_in * val)
            self._fig_height_spin.setValue(h_in * val)
            self._updating = False
        self._canvas.draw_idle()

    def _sync_padding_from_figure(self):
        self._updating = True
        sp = self._figure.subplotpars
        self._pad_left.setValue(sp.left)
        self._pad_right.setValue(sp.right)
        self._pad_bottom.setValue(sp.bottom)
        self._pad_top.setValue(sp.top)
        self._updating = False

    def _on_padding_changed(self):
        if self._updating:
            return
        self._figure.subplots_adjust(
            left=self._pad_left.value(),
            right=self._pad_right.value(),
            bottom=self._pad_bottom.value(),
            top=self._pad_top.value(),
        )
        self._canvas.draw_idle()

    def _apply_tight_layout(self):
        self._figure.tight_layout()
        self._sync_padding_from_figure()
        self._canvas.draw_idle()

    # ── Axes tab ──────────────────────────────────────────────────

    def _build_axes_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Axes:"))
        self._axes_selector = QComboBox()
        self._axes_selector.currentIndexChanged.connect(self._on_axes_selected)
        sel_row.addWidget(self._axes_selector)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        form = QFormLayout()
        self._ax_title_edit = QPlainTextEdit()
        self._ax_title_edit.setFixedHeight(50)
        self._ax_title_edit.setTabChangesFocus(True)
        form.addRow("Title:", self._ax_title_edit)

        # Title position (x, y in axes coordinates)
        title_pos_row = QHBoxLayout()
        title_pos_row.addWidget(QLabel("x:"))
        self._title_x = QDoubleSpinBox()
        self._title_x.setRange(-0.5, 1.5)
        self._title_x.setDecimals(2)
        self._title_x.setSingleStep(0.05)
        self._title_x.setValue(0.5)
        title_pos_row.addWidget(self._title_x)
        title_pos_row.addWidget(QLabel("y:"))
        self._title_y = QDoubleSpinBox()
        self._title_y.setRange(-0.5, 2.0)
        self._title_y.setDecimals(2)
        self._title_y.setSingleStep(0.02)
        self._title_y.setValue(1.0)
        title_pos_row.addWidget(self._title_y)
        form.addRow("Title pos:", title_pos_row)

        self._ax_xlabel_edit = QLineEdit()
        form.addRow("X label:", self._ax_xlabel_edit)

        # X label position (x, y in axes coordinates)
        xlabel_pos_row = QHBoxLayout()
        xlabel_pos_row.addWidget(QLabel("x:"))
        self._xlabel_x = QDoubleSpinBox()
        self._xlabel_x.setRange(-0.5, 1.5)
        self._xlabel_x.setDecimals(2)
        self._xlabel_x.setSingleStep(0.05)
        self._xlabel_x.setValue(0.5)
        xlabel_pos_row.addWidget(self._xlabel_x)
        xlabel_pos_row.addWidget(QLabel("y:"))
        self._xlabel_y = QDoubleSpinBox()
        self._xlabel_y.setRange(-1.0, 1.0)
        self._xlabel_y.setDecimals(2)
        self._xlabel_y.setSingleStep(0.02)
        self._xlabel_y.setValue(-0.1)
        xlabel_pos_row.addWidget(self._xlabel_y)
        form.addRow("X label pos:", xlabel_pos_row)

        self._ax_ylabel_edit = QLineEdit()
        form.addRow("Y label:", self._ax_ylabel_edit)

        # Y label position (x, y in axes coordinates)
        ylabel_pos_row = QHBoxLayout()
        ylabel_pos_row.addWidget(QLabel("x:"))
        self._ylabel_x = QDoubleSpinBox()
        self._ylabel_x.setRange(-1.0, 1.0)
        self._ylabel_x.setDecimals(2)
        self._ylabel_x.setSingleStep(0.02)
        self._ylabel_x.setValue(-0.1)
        ylabel_pos_row.addWidget(self._ylabel_x)
        ylabel_pos_row.addWidget(QLabel("y:"))
        self._ylabel_y = QDoubleSpinBox()
        self._ylabel_y.setRange(-0.5, 1.5)
        self._ylabel_y.setDecimals(2)
        self._ylabel_y.setSingleStep(0.05)
        self._ylabel_y.setValue(0.5)
        ylabel_pos_row.addWidget(self._ylabel_y)
        form.addRow("Y label pos:", ylabel_pos_row)

        self._ax_font_family = QFontComboBox()
        self._ax_font_family.setToolTip(
            "Typeface for the title and axis labels. Lists the fonts "
            "installed on this machine (each shown in its own face).")
        form.addRow("Font family:", self._ax_font_family)
        self._ax_font_preview = self._new_font_preview()
        form.addRow("Preview:", self._ax_font_preview)

        self._ax_title_size = QSpinBox()
        self._ax_title_size.setRange(6, 36)
        self._ax_title_size.setValue(13)
        form.addRow("Title size:", self._ax_title_size)

        self._ax_label_size = QSpinBox()
        self._ax_label_size.setRange(6, 36)
        self._ax_label_size.setValue(12)
        form.addRow("Label size:", self._ax_label_size)

        self._ax_tick_size = QSpinBox()
        self._ax_tick_size.setRange(6, 36)
        self._ax_tick_size.setValue(10)
        form.addRow("Tick label size:", self._ax_tick_size)
        layout.addLayout(form)

        # X axis
        xgrp = QGroupBox("X Axis")
        xf = QFormLayout(xgrp)
        self._xlim_auto = QCheckBox("Auto")
        self._xlim_auto.setChecked(True)
        xf.addRow("Limits:", self._xlim_auto)
        xlim_row = QHBoxLayout()
        self._xlim_min = QDoubleSpinBox()
        self._xlim_min.setRange(-1e6, 1e6)
        self._xlim_min.setDecimals(2)
        self._xlim_min.setSingleStep(1)
        self._xlim_min.setEnabled(False)
        xlim_row.addWidget(QLabel("Min:"))
        xlim_row.addWidget(self._xlim_min)
        self._xlim_max = QDoubleSpinBox()
        self._xlim_max.setRange(-1e6, 1e6)
        self._xlim_max.setDecimals(2)
        self._xlim_max.setSingleStep(1)
        self._xlim_max.setEnabled(False)
        xlim_row.addWidget(QLabel("Max:"))
        xlim_row.addWidget(self._xlim_max)
        xf.addRow("", xlim_row)
        self._xtick_major_spin = QDoubleSpinBox()
        self._xtick_major_spin.setRange(0, 10000)
        self._xtick_major_spin.setDecimals(1)
        self._xtick_major_spin.setSpecialValueText("Auto")
        xf.addRow("Major spacing:", self._xtick_major_spin)
        self._xtick_minor_check = QCheckBox("Enable")
        xf.addRow("Minor ticks:", self._xtick_minor_check)
        self._xtick_minor_count = QSpinBox()
        self._xtick_minor_count.setRange(2, 10)
        self._xtick_minor_count.setValue(5)
        xf.addRow("Minor count:", self._xtick_minor_count)
        self._xscale_combo = QComboBox()
        self._xscale_combo.addItems(["linear", "log"])
        self._xscale_combo.setToolTip(
            "Axis scale. Switching to log with non-positive data keeps "
            "the axis linear.")
        xf.addRow("Scale:", self._xscale_combo)
        self._xtick_dir_combo = QComboBox()
        self._xtick_dir_combo.addItems(["in", "out", "inout"])
        xf.addRow("Direction:", self._xtick_dir_combo)
        self._xgrid_check = QCheckBox("Show")
        xf.addRow("Grid:", self._xgrid_check)
        layout.addWidget(xgrp)

        # Y axis
        ygrp = QGroupBox("Y Axis")
        yf = QFormLayout(ygrp)
        self._ylim_auto = QCheckBox("Auto")
        self._ylim_auto.setChecked(True)
        yf.addRow("Limits:", self._ylim_auto)
        ylim_row = QHBoxLayout()
        self._ylim_min = QDoubleSpinBox()
        self._ylim_min.setRange(-1e6, 1e6)
        self._ylim_min.setDecimals(4)
        self._ylim_min.setSingleStep(0.01)
        self._ylim_min.setEnabled(False)
        ylim_row.addWidget(QLabel("Min:"))
        ylim_row.addWidget(self._ylim_min)
        self._ylim_max = QDoubleSpinBox()
        self._ylim_max.setRange(-1e6, 1e6)
        self._ylim_max.setDecimals(4)
        self._ylim_max.setSingleStep(0.01)
        self._ylim_max.setEnabled(False)
        ylim_row.addWidget(QLabel("Max:"))
        ylim_row.addWidget(self._ylim_max)
        yf.addRow("", ylim_row)
        self._ytick_major_spin = QDoubleSpinBox()
        self._ytick_major_spin.setRange(0, 10000)
        self._ytick_major_spin.setDecimals(3)
        self._ytick_major_spin.setSpecialValueText("Auto")
        yf.addRow("Major spacing:", self._ytick_major_spin)
        self._ytick_minor_check = QCheckBox("Enable")
        yf.addRow("Minor ticks:", self._ytick_minor_check)
        self._ytick_minor_count = QSpinBox()
        self._ytick_minor_count.setRange(2, 10)
        self._ytick_minor_count.setValue(5)
        yf.addRow("Minor count:", self._ytick_minor_count)
        self._yscale_combo = QComboBox()
        self._yscale_combo.addItems(["linear", "log"])
        self._yscale_combo.setToolTip(
            "Axis scale. Switching to log with non-positive data keeps "
            "the axis linear.")
        yf.addRow("Scale:", self._yscale_combo)
        self._ytick_dir_combo = QComboBox()
        self._ytick_dir_combo.addItems(["in", "out", "inout"])
        yf.addRow("Direction:", self._ytick_dir_combo)
        self._ygrid_check = QCheckBox("Show")
        yf.addRow("Grid:", self._ygrid_check)
        layout.addWidget(ygrp)

        layout.addStretch()

        # Connect signals
        self._ax_title_edit.textChanged.connect(self._apply_ax_text)
        self._ax_xlabel_edit.textChanged.connect(self._apply_ax_text)
        self._ax_ylabel_edit.textChanged.connect(self._apply_ax_text)
        self._title_x.valueChanged.connect(self._apply_text_positions)
        self._title_y.valueChanged.connect(self._apply_text_positions)
        self._xlabel_x.valueChanged.connect(self._apply_text_positions)
        self._xlabel_y.valueChanged.connect(self._apply_text_positions)
        self._ylabel_x.valueChanged.connect(self._apply_text_positions)
        self._ylabel_y.valueChanged.connect(self._apply_text_positions)
        self._ax_font_family.currentFontChanged.connect(self._apply_ax_font)
        self._ax_title_size.valueChanged.connect(self._apply_ax_font)
        self._ax_label_size.valueChanged.connect(self._apply_ax_font)
        self._ax_tick_size.valueChanged.connect(self._apply_ax_font)
        self._xlim_auto.toggled.connect(self._on_xlim_auto_toggled)
        self._xlim_min.valueChanged.connect(self._apply_limits)
        self._xlim_max.valueChanged.connect(self._apply_limits)
        self._ylim_auto.toggled.connect(self._on_ylim_auto_toggled)
        self._ylim_min.valueChanged.connect(self._apply_limits)
        self._ylim_max.valueChanged.connect(self._apply_limits)
        self._xtick_major_spin.valueChanged.connect(self._apply_ticks)
        self._xtick_minor_check.toggled.connect(self._apply_ticks)
        self._xtick_minor_count.valueChanged.connect(self._apply_ticks)
        self._xtick_dir_combo.currentTextChanged.connect(self._apply_ticks)
        self._ytick_major_spin.valueChanged.connect(self._apply_ticks)
        self._ytick_minor_check.toggled.connect(self._apply_ticks)
        self._ytick_minor_count.valueChanged.connect(self._apply_ticks)
        self._ytick_dir_combo.currentTextChanged.connect(self._apply_ticks)
        self._xscale_combo.currentTextChanged.connect(self._apply_scales)
        self._yscale_combo.currentTextChanged.connect(self._apply_scales)
        self._xgrid_check.toggled.connect(self._apply_grid)
        self._ygrid_check.toggled.connect(self._apply_grid)
        return w

    def _populate_axes_selector(self):
        self._updating = True
        self._axes_selector.clear()
        for i, ax in enumerate(self._figure.axes):
            self._axes_selector.addItem(f"Axes {i + 1}")
        self._updating = False
        if self._figure.axes:
            self._on_axes_selected(0)

    def _current_ax(self):
        idx = self._axes_selector.currentIndex()
        if 0 <= idx < len(self._figure.axes):
            return self._figure.axes[idx]
        return None

    def _on_axes_selected(self, index):
        ax = self._current_ax()
        if ax is None:
            return
        self._updating = True
        self._ax_title_edit.setPlainText(ax.get_title())
        self._ax_xlabel_edit.setText(ax.get_xlabel())
        self._ax_ylabel_edit.setText(ax.get_ylabel())
        # Title position
        tx, ty = ax.title.get_position()
        self._title_x.setValue(tx)
        self._title_y.setValue(ty)
        # X label position
        lx, ly = ax.xaxis.label.get_position()
        self._xlabel_x.setValue(lx)
        self._xlabel_y.setValue(ly)
        # Y label position
        yx, yy = ax.yaxis.label.get_position()
        self._ylabel_x.setValue(yx)
        self._ylabel_y.setValue(yy)
        self._ax_title_size.setValue(int(ax.title.get_fontsize()))
        self._ax_label_size.setValue(int(ax.xaxis.label.get_fontsize()))
        self._ax_tick_size.setValue(
            int(ax.xaxis.get_ticklabels()[0].get_fontsize())
            if ax.xaxis.get_ticklabels() else 10)
        fam = ax.title.get_fontfamily()
        fam = fam[0] if fam else "sans-serif"
        self._ax_font_family.setCurrentFont(QFont(fam))
        self._set_font_preview(
            self._ax_font_preview, self._ax_font_family.currentText(),
            self._ax_title_size.value(), False, False,
            text=f"{self._ax_font_family.currentText()} — AaBbGg 0123")
        # Axis limits
        xlim = ax.get_xlim()
        self._xlim_min.setValue(xlim[0])
        self._xlim_max.setValue(xlim[1])
        self._xlim_auto.setChecked(True)
        self._xlim_min.setEnabled(False)
        self._xlim_max.setEnabled(False)
        ylim = ax.get_ylim()
        self._ylim_min.setValue(ylim[0])
        self._ylim_max.setValue(ylim[1])
        self._ylim_auto.setChecked(True)
        self._ylim_min.setEnabled(False)
        self._ylim_max.setEnabled(False)
        # Grid: reflect current per-axis gridline visibility
        def _grid_on(axis):
            gl = axis.get_gridlines()
            return bool(gl) and gl[0].get_visible()
        self._xgrid_check.setChecked(_grid_on(ax.xaxis))
        self._ygrid_check.setChecked(_grid_on(ax.yaxis))
        # Tick major spacing
        x_loc = ax.xaxis.get_major_locator()
        if isinstance(x_loc, MultipleLocator):
            self._xtick_major_spin.setValue(x_loc._edge.step)
        else:
            self._xtick_major_spin.setValue(0)
        y_loc = ax.yaxis.get_major_locator()
        if isinstance(y_loc, MultipleLocator):
            self._ytick_major_spin.setValue(y_loc._edge.step)
        else:
            self._ytick_major_spin.setValue(0)
        # Tick direction
        xdir = ax.xaxis.get_tick_params().get("direction", "in")
        idx_d = self._xtick_dir_combo.findText(str(xdir))
        if idx_d >= 0:
            self._xtick_dir_combo.setCurrentIndex(idx_d)
        ydir = ax.yaxis.get_tick_params().get("direction", "in")
        idx_d = self._ytick_dir_combo.findText(str(ydir))
        if idx_d >= 0:
            self._ytick_dir_combo.setCurrentIndex(idx_d)
        # Axis scales
        for combo, scale in ((self._xscale_combo, ax.get_xscale()),
                             (self._yscale_combo, ax.get_yscale())):
            i = combo.findText(scale)
            combo.setCurrentIndex(i if i >= 0 else 0)
        self._updating = False

    def _apply_scales(self, *_):
        if self._updating:
            return
        ax = self._current_ax()
        if ax is None:
            return
        for combo, setter, getter in (
                (self._xscale_combo, ax.set_xscale, ax.get_xscale),
                (self._yscale_combo, ax.set_yscale, ax.get_yscale)):
            want = combo.currentText()
            if want == getter():
                continue
            try:
                setter(want)
            except (ValueError, RuntimeError):
                # e.g. log with non-positive limits — reflect reality.
                self._updating = True
                i = combo.findText(getter())
                combo.setCurrentIndex(i if i >= 0 else 0)
                self._updating = False
        self._canvas.draw_idle()

    def _on_xlim_auto_toggled(self, checked):
        self._xlim_min.setEnabled(not checked)
        self._xlim_max.setEnabled(not checked)
        if not checked:
            ax = self._current_ax()
            if ax is not None:
                self._updating = True
                xlim = ax.get_xlim()
                self._xlim_min.setValue(xlim[0])
                self._xlim_max.setValue(xlim[1])
                self._updating = False
        self._apply_limits()

    def _on_ylim_auto_toggled(self, checked):
        self._ylim_min.setEnabled(not checked)
        self._ylim_max.setEnabled(not checked)
        if not checked:
            ax = self._current_ax()
            if ax is not None:
                self._updating = True
                ylim = ax.get_ylim()
                self._ylim_min.setValue(ylim[0])
                self._ylim_max.setValue(ylim[1])
                self._updating = False
        self._apply_limits()

    def _apply_limits(self):
        if self._updating:
            return
        ax = self._current_ax()
        if ax is None:
            return
        if self._xlim_auto.isChecked():
            ax.autoscale(axis="x")
        else:
            ax.set_xlim(self._xlim_min.value(), self._xlim_max.value())
        if self._ylim_auto.isChecked():
            ax.autoscale(axis="y")
        else:
            ax.set_ylim(self._ylim_min.value(), self._ylim_max.value())
        self._canvas.draw_idle()

    def _apply_ax_text(self):
        if self._updating:
            return
        ax = self._current_ax()
        if ax is None:
            return
        ax.set_title(self._ax_title_edit.toPlainText(),
                     fontsize=self._ax_title_size.value())
        ax.set_xlabel(self._ax_xlabel_edit.text(),
                      fontsize=self._ax_label_size.value())
        ax.set_ylabel(self._ax_ylabel_edit.text(),
                      fontsize=self._ax_label_size.value())
        self._canvas.draw_idle()

    def _apply_text_positions(self):
        if self._updating:
            return
        ax = self._current_ax()
        if ax is None:
            return
        ax.title.set_position((self._title_x.value(), self._title_y.value()))
        ax.xaxis.label.set_position(
            (self._xlabel_x.value(), self._xlabel_y.value()))
        ax.yaxis.label.set_position(
            (self._ylabel_x.value(), self._ylabel_y.value()))
        self._canvas.draw_idle()

    def _apply_ax_font(self):
        if self._updating:
            return
        ax = self._current_ax()
        if ax is None:
            return
        family = self._ax_font_family.currentText()
        if family:   # empty only on a degenerate/headless font DB
            ax.title.set_fontfamily(family)
            ax.xaxis.label.set_fontfamily(family)
            ax.yaxis.label.set_fontfamily(family)
        ax.title.set_fontsize(self._ax_title_size.value())
        ax.xaxis.label.set_fontsize(self._ax_label_size.value())
        ax.yaxis.label.set_fontsize(self._ax_label_size.value())
        ax.tick_params(labelsize=self._ax_tick_size.value())
        self._set_font_preview(
            self._ax_font_preview, family, self._ax_title_size.value(),
            False, False, text=f"{family} — AaBbGg 0123")
        self._canvas.draw_idle()

    def _apply_ticks(self):
        if self._updating:
            return
        ax = self._current_ax()
        if ax is None:
            return
        # X ticks
        x_major = self._xtick_major_spin.value()
        if x_major > 0:
            ax.xaxis.set_major_locator(MultipleLocator(x_major))
        else:
            ax.xaxis.set_major_locator(plt.AutoLocator())
        if self._xtick_minor_check.isChecked():
            ax.xaxis.set_minor_locator(
                AutoMinorLocator(self._xtick_minor_count.value()))
        else:
            ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x",
                       direction=self._xtick_dir_combo.currentText(),
                       which="both")
        # Y ticks
        y_major = self._ytick_major_spin.value()
        if y_major > 0:
            ax.yaxis.set_major_locator(MultipleLocator(y_major))
        else:
            ax.yaxis.set_major_locator(plt.AutoLocator())
        if self._ytick_minor_check.isChecked():
            ax.yaxis.set_minor_locator(
                AutoMinorLocator(self._ytick_minor_count.value()))
        else:
            ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="y",
                       direction=self._ytick_dir_combo.currentText(),
                       which="both")
        self._canvas.draw_idle()

    def _apply_grid(self, *_):
        """Show/hide the x and y grid lines independently."""
        if self._updating:
            return
        ax = self._current_ax()
        if ax is None:
            return
        ax.grid(self._xgrid_check.isChecked(), axis='x')
        ax.grid(self._ygrid_check.isChecked(), axis='y')
        self._canvas.draw_idle()

    # ── Lines tab ─────────────────────────────────────────────────

    def _build_lines_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Axes:"))
        self._lines_axes_selector = QComboBox()
        self._lines_axes_selector.currentIndexChanged.connect(
            lambda _: self._populate_lines())
        sel_row.addWidget(self._lines_axes_selector)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        layout.addWidget(QLabel("Artists:"))
        self._lines_list = QListWidget()
        self._lines_list.currentRowChanged.connect(self._on_line_selected)
        layout.addWidget(self._lines_list)

        grp = QGroupBox("Properties")
        form = QFormLayout(grp)
        self._line_label_edit = QLineEdit()
        form.addRow("Label:", self._line_label_edit)

        self._line_color_btn = QPushButton()
        self._line_color_btn.setFixedSize(60, 24)
        self._line_color_btn.clicked.connect(self._on_pick_color)
        form.addRow("Color:", self._line_color_btn)

        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0, 10.0)
        self._line_width_spin.setSingleStep(0.1)
        self._line_width_spin.setDecimals(1)
        form.addRow("Width:", self._line_width_spin)

        # Dedicated font-size field for text/annotation artists. The
        # Width spin used to double as "font size" for text — its 0-10
        # range capped labels at 10 pt, which is the bug this fixes.
        self._text_size_spin = QDoubleSpinBox()
        self._text_size_spin.setRange(4, 72)
        self._text_size_spin.setSingleStep(1.0)
        self._text_size_spin.setDecimals(1)
        self._text_size_spin.setValue(10.0)
        self._text_size_spin.setEnabled(False)
        self._text_size_spin.setToolTip(
            "Font size (points) of the selected text or annotation "
            "label. Enabled only for text artists.")
        self._text_size_spin.valueChanged.connect(self._apply_text_size)
        form.addRow("Font size:", self._text_size_spin)

        # Typeface / weight / style for the selected text or annotation.
        self._text_family = QFontComboBox()
        self._text_family.setEnabled(False)
        self._text_family.setToolTip(
            "Typeface of the selected text/annotation label. Lists the "
            "fonts installed on this machine.")
        form.addRow("Font family:", self._text_family)

        text_style_row = QHBoxLayout()
        self._text_bold = QCheckBox("Bold")
        self._text_bold.setEnabled(False)
        self._text_italic = QCheckBox("Italic")
        self._text_italic.setEnabled(False)
        text_style_row.addWidget(self._text_bold)
        text_style_row.addWidget(self._text_italic)
        text_style_row.addStretch()
        form.addRow("Font style:", text_style_row)

        self._text_font_preview = self._new_font_preview()
        form.addRow("Preview:", self._text_font_preview)

        self._line_style_combo = QComboBox()
        for label, _ in LINESTYLE_MAP:
            self._line_style_combo.addItem(label)
        form.addRow("Style:", self._line_style_combo)

        self._line_marker_combo = QComboBox()
        for label, _ in MARKER_MAP:
            self._line_marker_combo.addItem(label)
        form.addRow("Marker:", self._line_marker_combo)

        self._line_marker_size = QDoubleSpinBox()
        self._line_marker_size.setRange(0, 30)
        self._line_marker_size.setSingleStep(0.5)
        self._line_marker_size.setDecimals(1)
        self._line_marker_size.setValue(6)
        form.addRow("Marker size:", self._line_marker_size)

        alpha_row = QHBoxLayout()
        self._line_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._line_alpha_slider.setRange(0, 100)
        self._line_alpha_slider.setValue(100)
        self._line_alpha_label = QLabel("1.00")
        alpha_row.addWidget(self._line_alpha_slider)
        alpha_row.addWidget(self._line_alpha_label)
        form.addRow("Alpha:", alpha_row)

        self._artist_visible_check = QCheckBox("Visible")
        self._artist_visible_check.setChecked(True)
        form.addRow("", self._artist_visible_check)

        def _coord_spin():
            s = QDoubleSpinBox()
            s.setRange(-1e12, 1e12)
            s.setDecimals(4)
            s.setSingleStep(1.0)
            s.setEnabled(False)
            return s

        # Exact placement for text/annotation artists (drag is quick,
        # numbers are reproducible). For annotations the second pair is
        # the arrow tip in data coordinates.
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("x:"))
        self._artist_x_spin = _coord_spin()
        pos_row.addWidget(self._artist_x_spin)
        pos_row.addWidget(QLabel("y:"))
        self._artist_y_spin = _coord_spin()
        pos_row.addWidget(self._artist_y_spin)
        form.addRow("Position:", pos_row)

        tip_row = QHBoxLayout()
        tip_row.addWidget(QLabel("x:"))
        self._tip_x_spin = _coord_spin()
        tip_row.addWidget(self._tip_x_spin)
        tip_row.addWidget(QLabel("y:"))
        self._tip_y_spin = _coord_spin()
        tip_row.addWidget(self._tip_y_spin)
        form.addRow("Arrow tip:", tip_row)

        self._text_rotation_spin = QDoubleSpinBox()
        self._text_rotation_spin.setRange(-180.0, 180.0)
        self._text_rotation_spin.setDecimals(1)
        self._text_rotation_spin.setSingleStep(5.0)
        self._text_rotation_spin.setEnabled(False)
        form.addRow("Rotation (°):", self._text_rotation_spin)

        self._artist_zorder_spin = QDoubleSpinBox()
        self._artist_zorder_spin.setRange(-100.0, 1000.0)
        self._artist_zorder_spin.setDecimals(1)
        self._artist_zorder_spin.setToolTip(
            "Draw order: higher values render on top of lower ones.")
        form.addRow("Z-order:", self._artist_zorder_spin)

        self._remove_artist_btn = QPushButton("Remove artist")
        self._remove_artist_btn.setToolTip(
            "Delete the selected artist from the plot. Tagged artists "
            "(the δν arrows and labels) stay removed when the plot is "
            "re-rendered or sent to Results; untagged artists are "
            "hidden instead (same as unticking Visible).")
        self._remove_artist_btn.clicked.connect(
            self._remove_selected_artist)
        form.addRow("", self._remove_artist_btn)

        layout.addWidget(grp)

        self._pick_check = QCheckBox("Click to select / drag on plot")
        self._pick_check.setToolTip(
            "Enable interactive editing on the plot:\n"
            "  • Click a line, scatter, or error-bar to select it in "
            "the Artists list above.\n"
            "  • Click and drag any text label or annotation arrow to "
            "reposition it. For arrows the text and the tip move "
            "together so the arrow stays attached to the label.")
        self._pick_check.toggled.connect(self._toggle_interact_mode)
        layout.addWidget(self._pick_check)

        layout.addStretch()

        # Connect editing signals
        self._line_label_edit.textChanged.connect(self._apply_line_label)
        self._line_width_spin.valueChanged.connect(self._apply_line_width)
        self._line_style_combo.currentIndexChanged.connect(
            self._apply_line_style)
        self._line_marker_combo.currentIndexChanged.connect(
            self._apply_line_marker)
        self._line_marker_size.valueChanged.connect(
            self._apply_line_marker_size)
        self._line_alpha_slider.valueChanged.connect(self._apply_line_alpha)
        self._artist_visible_check.toggled.connect(
            self._apply_artist_visible)
        self._artist_x_spin.valueChanged.connect(self._apply_artist_position)
        self._artist_y_spin.valueChanged.connect(self._apply_artist_position)
        self._tip_x_spin.valueChanged.connect(self._apply_annotation_tip)
        self._tip_y_spin.valueChanged.connect(self._apply_annotation_tip)
        self._text_rotation_spin.valueChanged.connect(
            self._apply_text_rotation)
        self._artist_zorder_spin.valueChanged.connect(
            self._apply_artist_zorder)
        self._text_family.currentFontChanged.connect(self._apply_text_font)
        self._text_bold.toggled.connect(self._apply_text_font)
        self._text_italic.toggled.connect(self._apply_text_font)
        return w

    def _refresh_text_font_preview(self):
        artist, kind = self._current_artist()
        if artist is not None and kind in ("text", "annotation"):
            sample = artist.get_text() or "AaBbGg 0123"
        else:
            sample = "AaBbGg 0123"
        self._set_font_preview(
            self._text_font_preview, self._text_family.currentText(),
            self._text_size_spin.value(), self._text_bold.isChecked(),
            self._text_italic.isChecked(), text=sample)

    def _apply_text_font(self, *_):
        """Typeface / bold / italic for the selected text or annotation."""
        if self._updating:
            self._refresh_text_font_preview()
            return
        artist, kind = self._current_artist()
        if artist is None or kind not in ("text", "annotation"):
            return
        fam = self._text_family.currentText()
        if fam:   # empty only on a degenerate/headless font DB
            artist.set_fontfamily(fam)
        artist.set_fontweight(
            "bold" if self._text_bold.isChecked() else "normal")
        artist.set_fontstyle(
            "italic" if self._text_italic.isChecked() else "normal")
        self._refresh_text_font_preview()
        self._canvas.draw_idle()

    # ── Add tab — create new elements on any axes ─────────────────

    _ADD_TYPES = ("Text", "Annotation (text + arrow)", "Vertical line",
                  "Horizontal line", "Vertical span", "Horizontal span")

    def _build_add_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Axes:"))
        self._add_axes_selector = QComboBox()
        sel_row.addWidget(self._add_axes_selector)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        form = QFormLayout()
        self._add_type_combo = QComboBox()
        self._add_type_combo.addItems(list(self._ADD_TYPES))
        self._add_type_combo.currentIndexChanged.connect(
            self._on_add_type_changed)
        form.addRow("Element:", self._add_type_combo)

        self._add_text_edit = QLineEdit()
        self._add_text_edit.setPlaceholderText("Label text…")
        form.addRow("Text:", self._add_text_edit)

        self._add_color = "#d62728"
        self._add_color_btn = QPushButton()
        self._add_color_btn.setFixedSize(60, 24)
        self._add_color_btn.setStyleSheet(
            f"background-color: {self._add_color}; border: 1px solid #888;")
        self._add_color_btn.clicked.connect(self._on_add_pick_color)
        form.addRow("Color:", self._add_color_btn)

        def _coord():
            s = QDoubleSpinBox()
            s.setRange(-1e12, 1e12)
            s.setDecimals(4)
            s.setSingleStep(1.0)
            return s

        c1_row = QHBoxLayout()
        c1_row.addWidget(QLabel("x:"))
        self._add_x_spin = _coord()
        c1_row.addWidget(self._add_x_spin)
        c1_row.addWidget(QLabel("y:"))
        self._add_y_spin = _coord()
        c1_row.addWidget(self._add_y_spin)
        form.addRow("Position:", c1_row)

        c2_row = QHBoxLayout()
        c2_row.addWidget(QLabel("x:"))
        self._add_x2_spin = _coord()
        c2_row.addWidget(self._add_x2_spin)
        c2_row.addWidget(QLabel("y:"))
        self._add_y2_spin = _coord()
        c2_row.addWidget(self._add_y2_spin)
        self._add_second_label = QLabel("2nd point:")
        form.addRow(self._add_second_label, c2_row)

        self._add_size_spin = QDoubleSpinBox()
        self._add_size_spin.setRange(4, 72)
        self._add_size_spin.setValue(11.0)
        self._add_size_spin.setDecimals(1)
        form.addRow("Font size:", self._add_size_spin)

        self._add_family = QFontComboBox()
        self._add_family.setToolTip(
            "Typeface for the new text/annotation label.")
        form.addRow("Font family:", self._add_family)

        add_style_row = QHBoxLayout()
        self._add_bold = QCheckBox("Bold")
        self._add_italic = QCheckBox("Italic")
        add_style_row.addWidget(self._add_bold)
        add_style_row.addWidget(self._add_italic)
        add_style_row.addStretch()
        form.addRow("Font style:", add_style_row)

        self._add_font_preview = self._new_font_preview()
        form.addRow("Preview:", self._add_font_preview)

        self._add_lw_spin = QDoubleSpinBox()
        self._add_lw_spin.setRange(0.1, 10.0)
        self._add_lw_spin.setValue(1.2)
        self._add_lw_spin.setSingleStep(0.1)
        self._add_lw_spin.setDecimals(1)
        form.addRow("Line width:", self._add_lw_spin)

        self._add_alpha_spin = QDoubleSpinBox()
        self._add_alpha_spin.setRange(0.05, 1.0)
        self._add_alpha_spin.setValue(0.25)
        self._add_alpha_spin.setSingleStep(0.05)
        self._add_alpha_spin.setDecimals(2)
        form.addRow("Span alpha:", self._add_alpha_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self._add_place_btn = QPushButton("Place by clicking")
        self._add_place_btn.setToolTip(
            "Arm click placement, then click the plot where the element "
            "should go. Annotations and spans take two clicks (arrow "
            "tip first, then the label; span edge, then the other "
            "edge). Esc or pressing the button again cancels.")
        self._add_place_btn.setCheckable(True)
        self._add_place_btn.toggled.connect(self._toggle_click_place)
        btn_row.addWidget(self._add_place_btn)
        add_btn = QPushButton("Add at coordinates")
        add_btn.clicked.connect(self._add_element_from_fields)
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._add_status = QLabel("")
        self._add_status.setWordWrap(True)
        layout.addWidget(self._add_status)
        layout.addStretch()

        self._add_text_edit.textChanged.connect(self._refresh_add_preview)
        self._add_size_spin.valueChanged.connect(self._refresh_add_preview)
        self._add_family.currentFontChanged.connect(self._refresh_add_preview)
        self._add_bold.toggled.connect(self._refresh_add_preview)
        self._add_italic.toggled.connect(self._refresh_add_preview)

        self._place_cid = None
        self._place_points = []
        self._on_add_type_changed(0)
        self._refresh_add_preview()
        return w

    def _refresh_add_preview(self, *_):
        self._set_font_preview(
            self._add_font_preview, self._add_family.currentText(),
            self._add_size_spin.value(), self._add_bold.isChecked(),
            self._add_italic.isChecked(),
            text=self._add_text_edit.text() or "AaBbGg 0123")

    def _on_add_type_changed(self, idx):
        """Enable only the fields the selected element type uses."""
        is_text = idx in (0, 1)
        needs_two = idx in (1, 4, 5)
        is_span = idx in (4, 5)
        self._add_text_edit.setEnabled(is_text)
        self._add_size_spin.setEnabled(is_text)
        self._add_family.setEnabled(is_text)
        self._add_bold.setEnabled(is_text)
        self._add_italic.setEnabled(is_text)
        self._add_font_preview.setEnabled(is_text)
        self._add_lw_spin.setEnabled(idx in (1, 2, 3))
        self._add_alpha_spin.setEnabled(is_span)
        self._add_second_label.setVisible(needs_two)
        for s in (self._add_x2_spin, self._add_y2_spin):
            s.setVisible(needs_two)
            s.setEnabled(needs_two)
        # vline uses only x, hline only y — keep both spins enabled
        # anyway; the unused one is simply ignored.
        self._add_second_label.setText(
            "Arrow tip:" if idx == 1 else "2nd point:")

    def _on_add_pick_color(self):
        color = QColorDialog.getColor(QColor(self._add_color), self,
                                      "Pick color")
        if color.isValid():
            self._add_color = color.name()
            self._add_color_btn.setStyleSheet(
                f"background-color: {self._add_color}; "
                f"border: 1px solid #888;")

    def _add_current_ax(self):
        idx = self._add_axes_selector.currentIndex()
        if idx < 0 or idx >= len(self._figure.axes):
            return None
        return self._figure.axes[idx]

    def _add_element_from_fields(self):
        self._add_element(self._add_x_spin.value(),
                          self._add_y_spin.value(),
                          self._add_x2_spin.value(),
                          self._add_y2_spin.value())

    def _add_element(self, x, y, x2, y2):
        """Create the selected element; tag it so Remove really removes."""
        ax = self._add_current_ax()
        if ax is None:
            return None
        t = self._add_type_combo.currentIndex()
        text = self._add_text_edit.text() or "text"
        color = self._add_color
        lw = self._add_lw_spin.value()
        font_kw = dict(
            fontweight="bold" if self._add_bold.isChecked() else "normal",
            fontstyle="italic" if self._add_italic.isChecked() else "normal",
        )
        _fam = self._add_family.currentText()
        if _fam:   # empty only on a degenerate/headless font DB
            font_kw["family"] = _fam
        try:
            if t == 0:
                a = ax.text(x, y, text, color=color,
                            fontsize=self._add_size_spin.value(),
                            zorder=10, **font_kw)
            elif t == 1:
                a = ax.annotate(
                    text, xy=(x2, y2), xytext=(x, y),
                    color=color, fontsize=self._add_size_spin.value(),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                    zorder=10, annotation_clip=False, **font_kw)
            elif t == 2:
                a = ax.axvline(x, color=color, linewidth=lw, zorder=5)
            elif t == 3:
                a = ax.axhline(y, color=color, linewidth=lw, zorder=5)
            elif t == 4:
                a = ax.axvspan(min(x, x2), max(x, x2), color=color,
                               alpha=self._add_alpha_spin.value(),
                               zorder=1)
            else:
                a = ax.axhspan(min(y, y2), max(y, y2), color=color,
                               alpha=self._add_alpha_spin.value(),
                               zorder=1)
        except Exception as exc:                      # noqa: BLE001
            self._add_status.setText(f"Could not add element: {exc}")
            return None
        a._pe_added = True
        self._add_status.setText(
            f"Added {self._ADD_TYPES[t]} — it is now editable and "
            f"removable in the Artists tab.")
        # Refresh the Artists list if it shows the same axes.
        if (self._lines_axes_selector.currentIndex()
                == self._add_axes_selector.currentIndex()):
            self._populate_lines()
        self._canvas.draw_idle()
        return a

    # ── Click-to-place ────────────────────────────────────────────

    def _toggle_click_place(self, on):
        if on:
            self._place_points = []
            self._place_cid = self._canvas.mpl_connect(
                "button_press_event", self._on_place_click)
            t = self._add_type_combo.currentIndex()
            first = ("arrow tip" if t == 1
                     else "first edge" if t in (4, 5)
                     else "target point")
            self._add_status.setText(
                f"Click the plot: {first}."
                + (" Two clicks total." if t in (1, 4, 5) else ""))
        else:
            self._cancel_click_place()

    def _cancel_click_place(self):
        if getattr(self, "_place_cid", None) is not None:
            try:
                self._canvas.mpl_disconnect(self._place_cid)
            except Exception:
                pass
        self._place_cid = None
        self._place_points = []
        if hasattr(self, "_add_place_btn"):
            self._add_place_btn.blockSignals(True)
            self._add_place_btn.setChecked(False)
            self._add_place_btn.blockSignals(False)

    def _on_place_click(self, event):
        ax = self._add_current_ax()
        if event.inaxes is not ax or event.xdata is None:
            return
        t = self._add_type_combo.currentIndex()
        self._place_points.append((event.xdata, event.ydata))
        needs_two = t in (1, 4, 5)
        if needs_two and len(self._place_points) < 2:
            second = "label position" if t == 1 else "second edge"
            self._add_status.setText(f"Now click the {second}.")
            return
        pts = self._place_points
        if t == 1:
            # First click = arrow tip, second = label position.
            (x2, y2), (x, y) = pts[0], pts[1]
        elif needs_two:
            (x, y), (x2, y2) = pts[0], pts[1]
        else:
            (x, y) = pts[0]
            x2, y2 = x, y
        # Mirror into the spins so "Add at coordinates" repeats it.
        for spin, val in ((self._add_x_spin, x), (self._add_y_spin, y),
                          (self._add_x2_spin, x2),
                          (self._add_y2_spin, y2)):
            spin.setValue(float(val))
        self._cancel_click_place()
        self._add_element(x, y, x2, y2)

    def _make_color_icon(self, color):
        hex_color = to_hex(color)
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(hex_color))
        return QIcon(pixmap)

    def _populate_lines(self):
        from matplotlib.container import ErrorbarContainer
        from matplotlib.collections import (
            LineCollection, PathCollection, PolyCollection)
        ax_idx = self._lines_axes_selector.currentIndex()
        if ax_idx < 0 or ax_idx >= len(self._figure.axes):
            return
        ax = self._figure.axes[ax_idx]
        self._updating = True
        self._line_artists = []  # list of (artist, kind) tuples
        self._lines_list.clear()

        # 1. ErrorbarContainers first (before lines, to group data+caps+bars)
        eb_lines = set()  # track lines owned by errorbars
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                data_line, cap_lines, bar_lines = container
                if data_line is not None:
                    eb_lines.add(id(data_line))
                for c in (cap_lines or []):
                    eb_lines.add(id(c))
                for b in (bar_lines or []):
                    eb_lines.add(id(b))
                label = container.get_label()
                if label.startswith("_"):
                    label = "Errorbar (unlabeled)"
                color = data_line.get_color() if data_line else "black"
                self._line_artists.append((container, "errorbar"))
                item = QListWidgetItem()
                item.setText(f"[EB] {label}")
                item.setIcon(self._make_color_icon(color))
                self._lines_list.addItem(item)

        # 2. Regular lines (skip those belonging to errorbars)
        for line in ax.get_lines():
            if id(line) in eb_lines:
                continue
            label = line.get_label()
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if label.startswith("_"):
                if len(xdata) == 2 and xdata[0] == xdata[1]:
                    display_label = f"vline @ {xdata[0]:.4g}"
                elif len(ydata) == 2 and ydata[0] == ydata[1]:
                    display_label = f"hline @ {ydata[0]:.4g}"
                else:
                    display_label = f"Line (unlabeled, {len(xdata)} pts)"
            else:
                display_label = label
            self._line_artists.append((line, "line"))
            item = QListWidgetItem()
            item.setText(
                f"[L] {display_label}  (lw={line.get_linewidth():.1f})")
            item.setIcon(self._make_color_icon(line.get_color()))
            self._lines_list.addItem(item)

        # 3. Collections (skip those belonging to errorbars)
        for coll in ax.collections:
            if id(coll) in eb_lines:
                continue
            if isinstance(coll, PathCollection):
                label = coll.get_label()
                if label.startswith("_"):
                    label = f"Scatter ({len(coll.get_offsets())} pts)"
                fc = coll.get_facecolor()
                color = fc[0] if len(fc) > 0 else [0, 0, 0, 1]
                self._line_artists.append((coll, "scatter"))
                item = QListWidgetItem()
                item.setText(f"[S] {label}")
                item.setIcon(self._make_color_icon(color))
                self._lines_list.addItem(item)
            elif isinstance(coll, PolyCollection):
                label = coll.get_label()
                if label.startswith("_"):
                    label = "Fill region"
                fc = coll.get_facecolor()
                color = fc[0] if len(fc) > 0 else [0, 0, 0, 1]
                self._line_artists.append((coll, "fill"))
                item = QListWidgetItem()
                item.setText(f"[F] {label}")
                item.setIcon(self._make_color_icon(color))
                self._lines_list.addItem(item)
            elif isinstance(coll, LineCollection):
                label = coll.get_label()
                if label.startswith("_"):
                    label = "Line collection"
                colors = coll.get_colors()
                color = colors[0] if len(colors) > 0 else [0, 0, 0, 1]
                self._line_artists.append((coll, "linecoll"))
                item = QListWidgetItem()
                item.setText(f"[LC] {label}")
                item.setIcon(self._make_color_icon(color))
                self._lines_list.addItem(item)

        # 4. Bar patches (histograms)
        for container in ax.containers:
            if not isinstance(container, ErrorbarContainer):
                label = container.get_label()
                if label.startswith("_"):
                    label = f"Bars ({len(container)} patches)"
                fc = container[0].get_facecolor() if container else "blue"
                self._line_artists.append((container, "bars"))
                item = QListWidgetItem()
                item.setText(f"[B] {label}")
                item.setIcon(self._make_color_icon(fc))
                self._lines_list.addItem(item)

        # 5. Text labels and annotations (with arrows). The isotope-shift
        # plot uses ax.text() for δν labels and ax.annotate() for the
        # connecting arrows; both go into ax.texts.
        from matplotlib.text import Annotation, Text
        for txt in list(ax.texts):
            content = txt.get_text() or ""
            short = (content[:25] + "…") if len(content) > 26 else content
            color = txt.get_color()
            if isinstance(txt, Annotation):
                self._line_artists.append((txt, "annotation"))
                item = QListWidgetItem()
                item.setText(f"[A→] \"{short}\"")
                item.setIcon(self._make_color_icon(color))
                self._lines_list.addItem(item)
            elif isinstance(txt, Text):
                self._line_artists.append((txt, "text"))
                item = QListWidgetItem()
                item.setText(f"[T] \"{short}\"")
                item.setIcon(self._make_color_icon(color))
                self._lines_list.addItem(item)

        # 6. Standalone arrow patches (the isotope-shift δν arrows are
        # FancyArrowPatch artists in ax.patches — previously they never
        # appeared here at all, so they couldn't be selected or edited).
        from matplotlib.patches import FancyArrowPatch
        for p in ax.patches:
            if not isinstance(p, FancyArrowPatch):
                continue
            name = p.get_gid() or "arrow"
            self._line_artists.append((p, "arrow"))
            item = QListWidgetItem()
            item.setText(f"[↔] {name}")
            try:
                item.setIcon(self._make_color_icon(
                    to_hex(p.get_edgecolor())))
            except Exception:
                pass
            self._lines_list.addItem(item)

        self._updating = False

    def _get_artist_color(self, artist, kind):
        """Extract the primary color from an artist."""
        from matplotlib.container import ErrorbarContainer
        if kind == "line":
            return to_hex(artist.get_color())
        elif kind == "errorbar":
            dl = artist[0]
            return to_hex(dl.get_color()) if dl else "#000000"
        elif kind in ("scatter", "fill", "linecoll"):
            fc = artist.get_facecolor() if kind != "linecoll" \
                else artist.get_colors()
            return to_hex(fc[0]) if len(fc) > 0 else "#000000"
        elif kind == "bars":
            return to_hex(artist[0].get_facecolor()) if artist else "#000000"
        elif kind in ("text", "annotation"):
            return to_hex(artist.get_color())
        elif kind == "arrow":
            try:
                return to_hex(artist.get_edgecolor())
            except Exception:
                return "#000000"
        return "#000000"

    def _primary_child(self, artist, kind):
        """Get the primary child artist for container types."""
        if kind == "errorbar":
            return artist[0]  # data line
        elif kind == "bars" and artist:
            return artist[0]  # first patch
        return artist

    def _on_line_selected(self, row):
        if self._updating or row < 0 or row >= len(self._line_artists):
            return
        self._updating = True
        try:
            artist, kind = self._line_artists[row]
            child = self._primary_child(artist, kind)

            # Label / text content
            if kind in ("text", "annotation"):
                # For text artists the "Label" edit doubles as the text
                # content editor — that's the single most useful action.
                label = artist.get_text() or ""
            else:
                label = artist.get_label()
                if label.startswith("_"):
                    if kind == "line":
                        xdata = artist.get_xdata()
                        ydata = artist.get_ydata()
                        if len(xdata) == 2 and xdata[0] == xdata[1]:
                            label = f"vline @ {xdata[0]:.4g}"
                        elif len(ydata) == 2 and ydata[0] == ydata[1]:
                            label = f"hline @ {ydata[0]:.4g}"
                        else:
                            label = f"Line ({len(xdata)} pts)"
                    elif kind == "errorbar":
                        label = "Errorbar"
                    elif kind == "scatter":
                        label = "Scatter"
                    elif kind == "fill":
                        label = "Fill"
                    else:
                        label = kind
            self._line_label_edit.setText(label)

            # Color
            hex_c = self._get_artist_color(artist, kind)
            self._line_color_btn.setStyleSheet(
                f"background-color: {hex_c}; border: 1px solid #888;")

            # Line width
            if kind == "line":
                self._line_width_spin.setValue(artist.get_linewidth())
            elif kind == "errorbar" and child is not None:
                self._line_width_spin.setValue(child.get_linewidth())
            elif kind in ("scatter", "fill", "linecoll"):
                lws = artist.get_linewidths()
                self._line_width_spin.setValue(
                    float(lws[0]) if len(lws) > 0 else 1.0)
            elif kind == "bars" and child is not None:
                self._line_width_spin.setValue(child.get_linewidth())
            elif kind == "annotation":
                # Arrow-patch line width when the arrow exists; the
                # label's font size lives in the Font size spin now.
                ap = artist.arrow_patch
                self._line_width_spin.setValue(
                    float(ap.get_linewidth()) if ap is not None else 1.0)
            elif kind == "arrow":
                self._line_width_spin.setValue(
                    float(artist.get_linewidth()))
            else:
                self._line_width_spin.setValue(1.0)

            # Font size — its own field with a sane range (the Width
            # spin used to double as font size, capping labels at 10 pt).
            is_texty = kind in ("text", "annotation")
            self._text_size_spin.setEnabled(is_texty)
            if is_texty:
                self._text_size_spin.setValue(
                    float(artist.get_fontsize()))
            self._line_width_spin.setEnabled(kind != "text")

            # Line style
            if kind == "line":
                ls = artist.get_linestyle()
            elif kind == "errorbar" and child is not None:
                ls = child.get_linestyle()
            else:
                ls = None
            if ls is not None:
                style_idx = 0
                for i, (_, code) in enumerate(LINESTYLE_MAP):
                    if code == ls:
                        style_idx = i
                        break
                self._line_style_combo.setCurrentIndex(style_idx)
            else:
                self._line_style_combo.setCurrentIndex(0)

            # Marker
            if kind == "line":
                mk = artist.get_marker()
                mk_idx = 0
                for i, (_, code) in enumerate(MARKER_MAP):
                    if str(mk) == code:
                        mk_idx = i
                        break
                self._line_marker_combo.setCurrentIndex(mk_idx)
                self._line_marker_size.setValue(artist.get_markersize())
            elif kind == "errorbar" and child is not None:
                mk = child.get_marker()
                mk_idx = 0
                for i, (_, code) in enumerate(MARKER_MAP):
                    if str(mk) == code:
                        mk_idx = i
                        break
                self._line_marker_combo.setCurrentIndex(mk_idx)
                self._line_marker_size.setValue(child.get_markersize())
            elif kind == "scatter":
                sizes = artist.get_sizes()
                self._line_marker_size.setValue(
                    float(sizes[0]) ** 0.5 if len(sizes) > 0 else 6)
                self._line_marker_combo.setCurrentIndex(0)
            else:
                self._line_marker_combo.setCurrentIndex(0)
                self._line_marker_size.setValue(6)

            # Alpha
            if kind == "line":
                alpha = artist.get_alpha() or 1.0
            elif kind in ("errorbar", "bars"):
                alpha = child.get_alpha() or 1.0 if child else 1.0
            elif kind in ("scatter", "fill", "linecoll"):
                alpha = artist.get_alpha()
                alpha = float(alpha) if alpha is not None else 1.0
            elif kind in ("text", "annotation"):
                a = artist.get_alpha()
                alpha = float(a) if a is not None else 1.0
            else:
                alpha = 1.0
            self._line_alpha_slider.setValue(int(alpha * 100))
            self._line_alpha_label.setText(f"{alpha:.2f}")

            # Visibility (containers don't have get_visible)
            if kind in ("errorbar", "bars"):
                visible = child.get_visible() if child else True
            else:
                visible = artist.get_visible()
            self._artist_visible_check.setChecked(visible)

            # Position (texts/annotations), arrow tip (annotations)
            texty = kind in ("text", "annotation")
            self._artist_x_spin.setEnabled(texty)
            self._artist_y_spin.setEnabled(texty)
            if texty:
                px, py = artist.get_position()
                self._artist_x_spin.setValue(float(px))
                self._artist_y_spin.setValue(float(py))
            has_tip = kind == "annotation"
            self._tip_x_spin.setEnabled(has_tip)
            self._tip_y_spin.setEnabled(has_tip)
            if has_tip:
                tx, ty = artist.xy
                self._tip_x_spin.setValue(float(tx))
                self._tip_y_spin.setValue(float(ty))

            # Rotation (texts only) and z-order (any artist)
            self._text_rotation_spin.setEnabled(texty)
            if texty:
                self._text_rotation_spin.setValue(
                    float(artist.get_rotation()))
            z_target = child if kind in ("errorbar", "bars") else artist
            try:
                self._artist_zorder_spin.setValue(
                    float(z_target.get_zorder()))
            except (AttributeError, TypeError):
                self._artist_zorder_spin.setValue(2.0)

            # Typeface / weight / style (texts and annotations only)
            self._text_family.setEnabled(texty)
            self._text_bold.setEnabled(texty)
            self._text_italic.setEnabled(texty)
            if texty:
                fam = artist.get_fontfamily()
                fam = fam[0] if fam else "sans-serif"
                self._text_family.setCurrentFont(QFont(fam))
                self._text_bold.setChecked(
                    self._weight_is_bold(artist.get_fontweight()))
                self._text_italic.setChecked(
                    str(artist.get_fontstyle()).lower()
                    in ("italic", "oblique"))
            self._refresh_text_font_preview()
        finally:
            self._updating = False

    def _current_artist(self):
        row = self._lines_list.currentRow()
        if row < 0 or row >= len(self._line_artists):
            return None, None
        return self._line_artists[row]

    def _on_pick_color(self):
        artist, kind = self._current_artist()
        if artist is None:
            return
        current = QColor(self._get_artist_color(artist, kind))
        color = QColorDialog.getColor(current, self, "Pick color")
        if not color.isValid():
            return
        cn = color.name()
        if kind == "line":
            artist.set_color(cn)
        elif kind == "errorbar":
            if artist[0] is not None:
                artist[0].set_color(cn)
            for cap in (artist[1] or []):
                cap.set_color(cn)
            for bar in (artist[2] or []):
                bar.set_color(cn)
        elif kind == "scatter":
            artist.set_facecolors(cn)
            artist.set_edgecolors(cn)
        elif kind == "fill":
            artist.set_facecolor(cn)
        elif kind == "linecoll":
            artist.set_color(cn)
        elif kind == "bars":
            for p in artist:
                p.set_facecolor(cn)
        elif kind == "text":
            artist.set_color(cn)
        elif kind == "annotation":
            artist.set_color(cn)
            ap = artist.arrow_patch
            if ap is not None:
                ap.set_color(cn)
        elif kind == "arrow":
            artist.set_color(cn)
        self._line_color_btn.setStyleSheet(
            f"background-color: {cn}; border: 1px solid #888;")
        self._lines_list.currentItem().setIcon(
            self._make_color_icon(cn))
        self._canvas.draw_idle()

    def _apply_line_label(self, text):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        if kind in ("text", "annotation"):
            # The "Label" field doubles as the text-content editor for
            # text/annotation artists, since editing what the label says
            # is the most useful action.
            artist.set_text(text)
            row = self._lines_list.currentRow()
            if row >= 0:
                short = (text[:25] + "…") if len(text) > 26 else text
                prefix = "[A→]" if kind == "annotation" else "[T]"
                self._lines_list.item(row).setText(
                    f"{prefix} \"{short}\"")
            self._refresh_text_font_preview()
        else:
            artist.set_label(text)
        self._canvas.draw_idle()

    def _apply_line_width(self, val):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        if kind == "line":
            artist.set_linewidth(val)
        elif kind == "errorbar":
            if artist[0] is not None:
                artist[0].set_linewidth(val)
            for cap in (artist[1] or []):
                cap.set_linewidth(val)
            for bar in (artist[2] or []):
                bar.set_linewidth(val)
        elif kind in ("scatter", "fill", "linecoll"):
            artist.set_linewidths([val])
        elif kind == "bars":
            for p in artist:
                p.set_linewidth(val)
        elif kind == "annotation":
            ap = artist.arrow_patch
            if ap is not None:
                ap.set_linewidth(val)
        elif kind == "arrow":
            artist.set_linewidth(val)
        self._canvas.draw_idle()

    def _apply_text_size(self, val):
        """Font size for text/annotation artists (dedicated field)."""
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None or kind not in ("text", "annotation"):
            return
        artist.set_fontsize(val)
        self._refresh_text_font_preview()
        self._canvas.draw_idle()

    def _apply_artist_position(self, *_):
        """Exact x/y for text/annotation artists (data coordinates)."""
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None or kind not in ("text", "annotation"):
            return
        artist.set_position((self._artist_x_spin.value(),
                             self._artist_y_spin.value()))
        if self._on_position_changed is not None:
            try:
                self._on_position_changed(artist)
            except Exception:
                pass
        self._canvas.draw_idle()

    def _apply_annotation_tip(self, *_):
        """Arrow-tip x/y for annotation artists."""
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None or kind != "annotation":
            return
        artist.xy = (self._tip_x_spin.value(), self._tip_y_spin.value())
        self._canvas.draw_idle()

    def _apply_text_rotation(self, val):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None or kind not in ("text", "annotation"):
            return
        artist.set_rotation(val)
        self._canvas.draw_idle()

    def _apply_artist_zorder(self, val):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        try:
            if kind in ("errorbar", "bars"):
                for part in artist.get_children():
                    part.set_zorder(val)
            else:
                artist.set_zorder(val)
        except (AttributeError, TypeError):
            return
        self._canvas.draw_idle()

    def _remove_selected_artist(self):
        """Delete the selected artist. gid-tagged artists (δν arrows /
        labels) are recorded in ``self.removed_gids`` so the deletion
        survives re-renders (the renderers regenerate defaults, and the
        style sidecar re-removes them by gid); untagged artists are
        hidden instead, which the sidecar persists via its per-artist
        ``visible`` flag."""
        artist, kind = self._current_artist()
        if artist is None:
            return
        # Artists added from the editor's own Add tab are truly
        # deleted — nothing re-renders them, so hiding would just leave
        # invisible clutter behind.
        if getattr(artist, "_pe_added", False):
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                artist.set_visible(False)
            self._populate_lines()
            self._canvas.draw_idle()
            return
        if not hasattr(self, "removed_gids"):
            self.removed_gids = set()
        gid = None
        try:
            gid = artist.get_gid()
        except Exception:
            pass
        if gid:
            self.removed_gids.add(gid)
            try:
                artist.remove()
            except Exception:
                try:
                    artist.set_visible(False)
                except Exception:
                    pass
        else:
            try:
                if kind in ("errorbar", "bars"):
                    for child_artist in artist:
                        if hasattr(child_artist, "set_visible"):
                            child_artist.set_visible(False)
                else:
                    artist.set_visible(False)
            except Exception:
                pass
        self._populate_lines()
        self._canvas.draw_idle()

    def _apply_line_style(self, idx):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        _, code = LINESTYLE_MAP[idx]
        if kind == "line":
            artist.set_linestyle(code)
        elif kind == "errorbar" and artist[0] is not None:
            artist[0].set_linestyle(code)
        elif kind == "linecoll":
            artist.set_linestyle(code)
        self._canvas.draw_idle()

    def _apply_line_marker(self, idx):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        _, code = MARKER_MAP[idx]
        if kind == "line":
            artist.set_marker(code)
        elif kind == "errorbar" and artist[0] is not None:
            artist[0].set_marker(code)
        self._canvas.draw_idle()

    def _apply_line_marker_size(self, val):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        if kind == "line":
            artist.set_markersize(val)
        elif kind == "errorbar" and artist[0] is not None:
            artist[0].set_markersize(val)
        elif kind == "scatter":
            artist.set_sizes([val ** 2])
        self._canvas.draw_idle()

    def _apply_line_alpha(self, val):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        alpha = val / 100.0
        if kind == "line":
            artist.set_alpha(alpha)
        elif kind == "errorbar":
            if artist[0] is not None:
                artist[0].set_alpha(alpha)
            for cap in (artist[1] or []):
                cap.set_alpha(alpha)
            for bar in (artist[2] or []):
                bar.set_alpha(alpha)
        elif kind in ("scatter", "fill", "linecoll"):
            artist.set_alpha(alpha)
        elif kind == "bars":
            for p in artist:
                p.set_alpha(alpha)
        elif kind == "text":
            artist.set_alpha(alpha)
        elif kind == "annotation":
            artist.set_alpha(alpha)
            ap = artist.arrow_patch
            if ap is not None:
                ap.set_alpha(alpha)
        elif kind == "arrow":
            artist.set_alpha(alpha)
        self._line_alpha_label.setText(f"{alpha:.2f}")
        self._canvas.draw_idle()

    def _apply_artist_visible(self, checked):
        if self._updating:
            return
        artist, kind = self._current_artist()
        if artist is None:
            return
        if kind in ("errorbar", "bars"):
            # Containers don't have set_visible; toggle children
            for child_artist in artist:
                if hasattr(child_artist, 'set_visible'):
                    child_artist.set_visible(checked)
            # Also toggle errorbar caps and bars
            if kind == "errorbar":
                for cap in (artist[1] or []):
                    cap.set_visible(checked)
                for bar in (artist[2] or []):
                    bar.set_visible(checked)
        else:
            artist.set_visible(checked)
        self._canvas.draw_idle()

    # ── Pick mode ─────────────────────────────────────────────────

    def _toggle_interact_mode(self, enabled):
        """Single toggle for the merged 'click to select / drag'
        checkbox: drives both the pick selector (lines, scatter,
        errorbars) and the drag-to-reposition mode (text / arrows).
        """
        self._toggle_pick_mode(enabled)
        self._toggle_drag_mode(enabled)

    def _toggle_pick_mode(self, enabled):
        if enabled:
            for ax in self._figure.axes:
                for line in ax.get_lines():
                    line.set_picker(5)
                for coll in ax.collections:
                    coll.set_picker(5)
            self._pick_cid = self._canvas.mpl_connect(
                "pick_event", self._on_pick)
        else:
            if self._pick_cid is not None:
                self._canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None
            for ax in self._figure.axes:
                for line in ax.get_lines():
                    line.set_picker(False)
                for coll in ax.collections:
                    coll.set_picker(False)

    # ── Drag-to-move text / annotation artists ────────────────────

    def _toggle_drag_mode(self, enabled):
        """Hook the canvas mouse events so the user can grab text labels
        and annotation arrows and drag them to a new location.
        """
        if enabled:
            self._drag_press_cid = self._canvas.mpl_connect(
                "button_press_event", self._on_drag_press)
            self._drag_motion_cid = self._canvas.mpl_connect(
                "motion_notify_event", self._on_drag_motion)
            self._drag_release_cid = self._canvas.mpl_connect(
                "button_release_event", self._on_drag_release)
        else:
            for cid_attr in ("_drag_press_cid", "_drag_motion_cid",
                             "_drag_release_cid"):
                cid = getattr(self, cid_attr)
                if cid is not None:
                    self._canvas.mpl_disconnect(cid)
                    setattr(self, cid_attr, None)
            self._dragging = None

    def _on_drag_press(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        if event.x is None or event.y is None:
            return
        ax = event.inaxes
        # First, hit-test the endpoints of any δν FancyArrowPatch in
        # this axes (in display/pixel space). Grabbing an endpoint lets
        # the user reshape the arrow; each end moves independently.
        grabbed = self._hit_test_arrow_endpoint(ax, event)
        if grabbed is not None:
            arrow, which = grabbed
            self._drag_target = arrow
            self._drag_kind = which  # "tail" | "head"
            self._dragging = None
            return
        # Otherwise, hit-test text artists in the axes under the
        # cursor. Iterate in reverse so the topmost (last drawn) artist
        # wins overlaps.
        from matplotlib.text import Annotation
        for artist in reversed(list(ax.texts)):
            if not artist.get_visible():
                continue
            try:
                hit, _ = artist.contains(event)
            except Exception:
                hit = False
            if not hit:
                continue
            # Compute the click position in the *artist's own* coord
            # system. Many panel labels are placed with
            # transform=ax.transAxes (axes-fraction coords); using raw
            # data deltas there sends the text far off-screen.
            try:
                inv = artist.get_transform().inverted()
                start_xa, start_ya = inv.transform((event.x, event.y))
            except Exception:
                return
            state = {"artist": artist, "ax": ax,
                     "start_x": float(start_xa),
                     "start_y": float(start_ya)}
            if isinstance(artist, Annotation):
                # Only allow data-coord annotations — otherwise xy and
                # xytext live in different transforms and the simple
                # "shift both by dx/dy" scheme below doesn't apply.
                if (artist.xycoords != "data" or
                        artist.anncoords != "data"):
                    return
                state["orig_xy"] = tuple(artist.xy)
                state["orig_xytext"] = tuple(artist.get_position())
            else:
                state["orig_pos"] = tuple(artist.get_position())
            self._dragging = state
            # If this is a linked δν label, mark it so the motion
            # handler keeps its arrow's tail attached and the release
            # handler persists the new position.
            self._drag_target = artist
            self._drag_kind = "text"
            # Highlight the picked row in the artists list
            for row, (a, _kind) in enumerate(self._line_artists):
                if a is artist:
                    self._lines_list.setCurrentRow(row)
                    break
            return

    @staticmethod
    def _arrow_positions(arrow):
        """Return a FancyArrowPatch's (posA, posB) endpoints in data
        coords. matplotlib (≥3.x) has no public getter -- the endpoints
        live in the private ``_posA_posB`` attribute -- so read that."""
        return tuple(arrow._posA_posB)

    def _hit_test_arrow_endpoint(self, ax, event, pickradius=10.0):
        """Return ``(arrow, "tail"|"head")`` if the click is within
        ``pickradius`` pixels of an endpoint of one of the registered
        δν FancyArrowPatch artists in ``ax``, else None. Endpoints are
        compared in display (pixel) space so the radius is uniform
        regardless of the data scale."""
        best = None
        best_d2 = pickradius * pickradius
        for arrow in self._arrows_by_id.values():
            if arrow.axes is not ax or not arrow.get_visible():
                continue
            try:
                posA, posB = self._arrow_positions(arrow)
            except Exception:
                continue
            ax_disp = ax.transData.transform(posA)
            bx_disp = ax.transData.transform(posB)
            for which, (px, py) in (("tail", ax_disp), ("head", bx_disp)):
                d2 = (event.x - px) ** 2 + (event.y - py) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (arrow, which)
        return best

    def _on_drag_motion(self, event):
        if event.x is None or event.y is None:
            return
        # ── Arrow endpoint drag: move only the grabbed end. ──
        if self._drag_kind in ("tail", "head") \
                and self._drag_target is not None:
            arrow = self._drag_target
            ax = arrow.axes
            if ax is None:
                return
            try:
                data_xy = ax.transData.inverted().transform(
                    (event.x, event.y))
            except Exception:
                return
            posA, posB = self._arrow_positions(arrow)
            if self._drag_kind == "tail":
                arrow.set_positions(tuple(data_xy), posB)
            else:
                arrow.set_positions(posA, tuple(data_xy))
            self._canvas.draw_idle()
            return
        # ── Text / annotation drag. ──
        if self._dragging is None:
            return
        from matplotlib.text import Annotation
        artist = self._dragging["artist"]
        try:
            inv = artist.get_transform().inverted()
            cur_xa, cur_ya = inv.transform((event.x, event.y))
        except Exception:
            return
        dx = cur_xa - self._dragging["start_x"]
        dy = cur_ya - self._dragging["start_y"]
        if isinstance(artist, Annotation):
            ox, oy = self._dragging["orig_xy"]
            tx, ty = self._dragging["orig_xytext"]
            artist.xy = (ox + dx, oy + dy)
            artist.set_position((tx + dx, ty + dy))
        else:
            ox, oy = self._dragging["orig_pos"]
            artist.set_position((ox + dx, oy + dy))
        # If this is a linked δν label, keep its arrow's tail attached
        # so the label and its arrow move together.
        aid = getattr(artist, "_is_id", None)
        arrow = self._arrows_by_id.get(aid) if aid is not None else None
        if arrow is not None:
            try:
                _posA, posB = self._arrow_positions(arrow)
                arrow.set_positions(artist.get_position(), posB)
            except Exception:
                pass
        self._canvas.draw_idle()

    def _on_drag_release(self, event):
        # Persist the final position back to the owning tab (if wired)
        # so the drag survives replot, save and reload.
        cb = self._on_position_changed
        if cb is not None:
            if self._drag_kind == "text" and self._dragging is not None:
                artist = self._dragging["artist"]
                aid = getattr(artist, "_is_id", None)
                if aid is not None:
                    cb("label", aid, artist.get_position())
                    arrow = self._arrows_by_id.get(aid)
                    if arrow is not None:
                        cb("arrow", aid, self._arrow_positions(arrow))
            elif self._drag_kind in ("tail", "head") \
                    and self._drag_target is not None:
                aid = getattr(self._drag_target, "_is_id", None)
                if aid is not None:
                    cb("arrow", aid,
                       self._arrow_positions(self._drag_target))
        self._dragging = None
        self._drag_target = None
        self._drag_kind = None

    def _on_pick(self, event):
        picked = event.artist
        # Find which axes and which artist entry it belongs to
        for ax_idx, ax in enumerate(self._figure.axes):
            found = (picked in ax.get_lines() or
                     picked in ax.collections)
            if not found:
                # Check if it belongs to an errorbar container
                for c in ax.containers:
                    if picked in (c[0],) or picked in (c[1] or []) \
                            or picked in (c[2] or []):
                        found = True
                        break
            if found:
                self._lines_axes_selector.blockSignals(True)
                self._lines_axes_selector.setCurrentIndex(ax_idx)
                self._lines_axes_selector.blockSignals(False)
                self._populate_lines()
                # Find the matching entry
                for row, (art, kind) in enumerate(self._line_artists):
                    if art is picked:
                        self._lines_list.setCurrentRow(row)
                        return
                    # Check errorbar sub-artists
                    if kind == "errorbar":
                        if (picked is art[0] or
                                picked in (art[1] or []) or
                                picked in (art[2] or [])):
                            self._lines_list.setCurrentRow(row)
                            return
                break

    # ── Legend tab ────────────────────────────────────────────────

    def _build_legend_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Axes:"))
        self._legend_axes_selector = QComboBox()
        sel_row.addWidget(self._legend_axes_selector)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        form = QFormLayout()
        self._legend_loc_combo = QComboBox()
        self._legend_loc_combo.addItems(LEGEND_LOCATIONS)
        form.addRow("Position:", self._legend_loc_combo)

        self._legend_fontsize_spin = QSpinBox()
        self._legend_fontsize_spin.setRange(6, 24)
        self._legend_fontsize_spin.setValue(9)
        form.addRow("Font size:", self._legend_fontsize_spin)

        self._legend_frame_check = QCheckBox("Visible")
        self._legend_frame_check.setChecked(True)
        form.addRow("Frame:", self._legend_frame_check)

        self._legend_ncol_spin = QSpinBox()
        self._legend_ncol_spin.setRange(1, 6)
        self._legend_ncol_spin.setValue(1)
        form.addRow("Columns:", self._legend_ncol_spin)

        self._legend_title_edit = QLineEdit()
        self._legend_title_edit.setPlaceholderText("(no title)")
        form.addRow("Title:", self._legend_title_edit)

        self._legend_alpha_spin = QDoubleSpinBox()
        self._legend_alpha_spin.setRange(0, 1)
        self._legend_alpha_spin.setSingleStep(0.1)
        self._legend_alpha_spin.setDecimals(1)
        self._legend_alpha_spin.setValue(0.8)
        form.addRow("Frame alpha:", self._legend_alpha_spin)
        layout.addLayout(form)

        apply_btn = QPushButton("Apply Legend")
        apply_btn.clicked.connect(self._apply_legend)
        layout.addWidget(apply_btn)

        layout.addStretch()
        return w

    def _apply_legend(self):
        ax_idx = self._legend_axes_selector.currentIndex()
        if ax_idx < 0 or ax_idx >= len(self._figure.axes):
            return
        ax = self._figure.axes[ax_idx]
        title = self._legend_title_edit.text().strip()
        ax.legend(
            loc=self._legend_loc_combo.currentText(),
            fontsize=self._legend_fontsize_spin.value(),
            frameon=self._legend_frame_check.isChecked(),
            framealpha=self._legend_alpha_spin.value(),
            ncol=self._legend_ncol_spin.value(),
            title=title or None,
        )
        self._canvas.draw_idle()

    # ── Export tab ────────────────────────────────────────────────

    def _build_export_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()

        self._export_dpi = QSpinBox()
        self._export_dpi.setRange(72, 1200)
        self._export_dpi.setSingleStep(50)
        self._export_dpi.setValue(300)
        form.addRow("Export DPI:", self._export_dpi)
        layout.addLayout(form)

        # Overwrite original file
        self._overwrite_btn = QPushButton("Overwrite Original")
        self._overwrite_btn.setToolTip(
            "Save over the original image file on disk")
        self._overwrite_btn.clicked.connect(self._overwrite_original)
        # Only enabled if source_path is set
        self._overwrite_btn.setEnabled(bool(self._source_path))
        layout.addWidget(self._overwrite_btn)

        # Save As buttons
        btn_row = QHBoxLayout()
        for fmt in ("png", "pdf", "svg"):
            btn = QPushButton(f"Save as {fmt.upper()}")
            btn.clicked.connect(lambda checked, f=fmt: self._save_figure(f))
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        return w

    def _overwrite_original(self):
        """Save figure over the original image file on disk."""
        if not self._source_path:
            return
        # Determine the image path next to the .npz
        base = self._source_path
        if base.lower().endswith('.npz'):
            base = base[:-4]
        # Find existing image format or default to png
        img_path = None
        for ext in ('.png', '.pdf', '.svg'):
            candidate = base + ext
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            img_path = base + '.png'
        self._figure.savefig(img_path, dpi=self._export_dpi.value(),
                             bbox_inches="tight")
        # Also save style for live re-rendering persistence
        if self._source_path.lower().endswith('.npz'):
            try:
                from gui.results_tab import _extract_figure_style, _style_path
                style = _extract_figure_style(self._figure)
                sp = _style_path(self._source_path)
                import json
                with open(sp, 'w') as f:
                    json.dump(style, f, indent=1)
            except Exception:
                pass
        QMessageBox.information(
            self, "Saved", f"Overwritten:\n{img_path}")

    def _save_figure(self, fmt):
        # Suggest filename based on source path, prefixed with the
        # last-used data save dir so the dialog opens in a sensible place.
        default_name = ""
        if self._source_path:
            base = os.path.splitext(
                os.path.basename(self._source_path))[0]
            default_name = f"{base}.{fmt}"
        last_save = get_last_dir("data", "save")
        if last_save and default_name:
            default_name = os.path.join(last_save, default_name)
        elif last_save:
            default_name = last_save
        filters = {"png": "PNG files (*.png)",
                   "pdf": "PDF files (*.pdf)",
                   "svg": "SVG files (*.svg)"}
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save as {fmt.upper()}", default_name,
            filters.get(fmt, f"{fmt.upper()} files (*.{fmt})"))
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path += f".{fmt}"
        remember_last_dir("data", "save", path)
        self._figure.savefig(path, dpi=self._export_dpi.value(),
                             bbox_inches="tight")
        QMessageBox.information(self, "Saved", f"Figure saved to:\n{path}")

    # ── Lifecycle ─────────────────────────────────────────────────

    def refresh(self):
        """Re-read figure state after the plot was redrawn."""
        if self._pick_cid is not None:
            self._canvas.mpl_disconnect(self._pick_cid)
            self._pick_cid = None
            self._pick_check.setChecked(False)

        self._updating = True
        self._fig_width_spin.setValue(
            self._to_display(self._figure.get_figwidth()))
        self._fig_height_spin.setValue(
            self._to_display(self._figure.get_figheight()))
        self._fig_dpi_spin.setValue(int(self._figure.get_dpi()))
        self._updating = False

        self._sync_padding_from_figure()
        self._populate_axes_selector()
        # Sync the lines and legend axes selectors
        self._lines_axes_selector.blockSignals(True)
        self._lines_axes_selector.clear()
        for i in range(len(self._figure.axes)):
            self._lines_axes_selector.addItem(f"Axes {i + 1}")
        self._lines_axes_selector.blockSignals(False)

        self._legend_axes_selector.blockSignals(True)
        self._legend_axes_selector.clear()
        for i in range(len(self._figure.axes)):
            self._legend_axes_selector.addItem(f"Axes {i + 1}")
        self._legend_axes_selector.blockSignals(False)

        self._populate_lines()

    def _disconnect_canvas_handlers(self):
        """Detach pick + drag handlers and clear picker flags."""
        if self._pick_cid is not None:
            self._canvas.mpl_disconnect(self._pick_cid)
            self._pick_cid = None
        for cid_attr in ("_drag_press_cid", "_drag_motion_cid",
                         "_drag_release_cid"):
            cid = getattr(self, cid_attr, None)
            if cid is not None:
                self._canvas.mpl_disconnect(cid)
                setattr(self, cid_attr, None)
        self._dragging = None
        for ax in self._figure.axes:
            for line in ax.get_lines():
                line.set_picker(False)

    def _on_close(self):
        self._disconnect_canvas_handlers()
        self._cancel_click_place()
        self.close()

    def closeEvent(self, event):
        self._disconnect_canvas_handlers()
        self._cancel_click_place()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════
#  Universal plot-editor access (right-click on any canvas)
# ══════════════════════════════════════════════════════════════════

def open_plot_editor_for(canvas):
    """Open the plot editor for any matplotlib canvas in the app.

    Surfaces with richer wiring (Results style persistence, the
    isotope-shift arrow registry, PA/Estimate extra-figure lists) set
    ``canvas._plot_editor_opener`` to their own method; everything else
    gets a fresh generic ``PlotEditorDialog`` on the canvas's figure.
    """
    opener = getattr(canvas, "_plot_editor_opener", None)
    if callable(opener):
        opener()
        return None
    old = getattr(canvas, "_generic_plot_editor", None)
    if old is not None:
        # Always rebuild: the figure may have been cleared/redrawn since,
        # and the dialog caches artist references.
        try:
            old.close()
            old.deleteLater()
        except RuntimeError:
            pass
    dlg = PlotEditorDialog(canvas.figure, canvas, parent=canvas.window())
    canvas._generic_plot_editor = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


class _CanvasEditorFilter(QObject):
    """App-level filter: right-click on ANY matplotlib canvas offers
    "Edit plot…" — the universal entry point to the plot editor, with
    no per-surface wiring. Suppressed while a navigation-toolbar tool
    (pan/zoom) is engaged so it never steals the right-drag zoom."""

    def eventFilter(self, obj, ev):
        try:
            if (ev.type() == QEvent.Type.ContextMenu
                    and isinstance(obj, FigureCanvasQTAgg)
                    and getattr(obj, "figure", None) is not None):
                tb = getattr(obj, "toolbar", None)
                if tb is not None and getattr(tb, "mode", ""):
                    return False
                from PySide6.QtWidgets import QMenu
                menu = QMenu(obj)
                act = menu.addAction("Edit plot…")
                act.triggered.connect(
                    lambda *_, c=obj: open_plot_editor_for(c))
                menu.exec(ev.globalPos())
                return True
        except Exception:
            return False
        return False


_canvas_editor_filter = None


def install_plot_editor_access(app):
    """Install the right-click → "Edit plot…" filter once per app."""
    global _canvas_editor_filter
    if _canvas_editor_filter is None:
        _canvas_editor_filter = _CanvasEditorFilter(app)
        app.installEventFilter(_canvas_editor_filter)
    return _canvas_editor_filter


# ══════════════════════════════════════════════════════════════════
#  Worker Thread
# ══════════════════════════════════════════════════════════════════
class _StdoutCapture:
    """Redirects stdout writes to a Qt signal, line by line.

    ``original`` may be a stream that is None or non-functional when the
    app is launched via pythonw.exe without a console. Writes to the
    original are guarded so a broken upstream stream never blocks log
    delivery to the GUI.
    """

    def __init__(self, signal, original):
        self.signal = signal
        self.original = original
        self._buf = ""

    def write(self, text):
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.signal.emit(line)

    def flush(self):
        if self._buf:
            self.signal.emit(self._buf)
            self._buf = ""
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass

    def fileno(self):
        if self.original is None:
            raise OSError("no underlying stream")
        return self.original.fileno()

    def isatty(self):
        return False


class WorkerThread(QThread):
    """Runs cls_estimate.run() in a background thread."""
    log_line = Signal(str)
    finished_ok = Signal(str)   # output_dir
    finished_error = Signal(str)

    def __init__(self, config_path, output_dir, no_plot, palette):
        super().__init__()
        self.config_path = config_path
        self.output_dir = output_dir
        self.no_plot = no_plot
        self.palette = palette
        self.plot_results = None
        self.all_peaks = None

    def run(self):
        # Capture stdout: run() creates a StreamHandler(sys.stdout),
        # so redirecting stdout captures all log output for the GUI.
        original_stdout = sys.stdout
        capture = _StdoutCapture(self.log_line, original_stdout)
        sys.stdout = capture

        def _safe_flush():
            try:
                capture.flush()
            except Exception:
                pass

        try:
            from cls_estimate import run as run_estimation  # lazy import
            result = run_estimation(
                self.config_path,
                self.output_dir,
                self.no_plot,
                self.palette,
            )
            if isinstance(result, tuple) and len(result) == 2:
                self.plot_results, self.all_peaks = result
            else:
                self.plot_results = result
                self.all_peaks = None
            _safe_flush()
            self.finished_ok.emit(self.output_dir)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            _safe_flush()
            self.finished_error.emit(f"{e}\n\n{tb}")
        finally:
            sys.stdout = original_stdout


# ══════════════════════════════════════════════════════════════════
#  Schmidt Moment Calculator
# ══════════════════════════════════════════════════════════════════
class SchmidtCalculatorDialog(QDialog):
    """Standalone Schmidt moment calculator tool."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Schmidt Moment Calculator")
        self.setWindowIcon(lucide_icon("target"))
        self.resize(450, 500)
        self.setModal(False)

        layout = QVBoxLayout(self)

        # Mode selection
        mode_row = QHBoxLayout()
        self._single_radio = QRadioButton("Single particle")
        self._two_radio = QRadioButton("Two particles")
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._single_radio)
        mode_group.addButton(self._two_radio)
        self._single_radio.setChecked(True)
        mode_row.addWidget(self._single_radio)
        mode_row.addWidget(self._two_radio)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Single particle group
        self._single_grp = QGroupBox("Single Particle")
        sf = QFormLayout(self._single_grp)
        self._sp_type = QComboBox()
        self._sp_type.addItems(["proton", "neutron"])
        sf.addRow("Type:", self._sp_type)
        self._sp_orbital = QComboBox()
        self._sp_orbital.addItems(ORBITAL_NAMES)
        sf.addRow("Orbital:", self._sp_orbital)
        layout.addWidget(self._single_grp)

        # Two particle group
        self._two_grp = QGroupBox("Two Particles")
        tf = QFormLayout(self._two_grp)
        self._tp_type1 = QComboBox()
        self._tp_type1.addItems(["proton", "neutron"])
        tf.addRow("Type 1:", self._tp_type1)
        self._tp_orbital1 = QComboBox()
        self._tp_orbital1.addItems(ORBITAL_NAMES)
        tf.addRow("Orbital 1:", self._tp_orbital1)
        self._tp_type2 = QComboBox()
        self._tp_type2.addItems(["proton", "neutron"])
        tf.addRow("Type 2:", self._tp_type2)
        self._tp_orbital2 = QComboBox()
        self._tp_orbital2.addItems(ORBITAL_NAMES)
        tf.addRow("Orbital 2:", self._tp_orbital2)
        self._tp_I = QDoubleSpinBox()
        self._tp_I.setRange(0, 20)
        self._tp_I.setSingleStep(0.5)
        self._tp_I.setDecimals(1)
        self._tp_I.setValue(0.5)
        tf.addRow("Total I:", self._tp_I)
        layout.addWidget(self._two_grp)
        self._two_grp.setVisible(False)

        # Quenching factor
        quench_row = QHBoxLayout()
        quench_row.addWidget(QLabel("g_s quenching factor:"))
        self._gs_factor = QDoubleSpinBox()
        self._gs_factor.setRange(0.01, 2.0)
        self._gs_factor.setSingleStep(0.01)
        self._gs_factor.setDecimals(3)
        self._gs_factor.setValue(1.0)
        self._gs_factor.setToolTip(
            "1.0 = free nucleon g-factors\n"
            "~0.6-0.7 = effective (quenched) values\n"
            "Multiplies g_s before moment calculation")
        quench_row.addWidget(self._gs_factor)
        quench_row.addStretch()
        layout.addLayout(quench_row)

        # Results
        res_grp = QGroupBox("Results")
        rf = QFormLayout(res_grp)
        self._res_mu = QLineEdit()
        self._res_mu.setReadOnly(True)
        rf.addRow("\u03bc (\u03bc\u2099):", self._res_mu)
        self._res_g = QLineEdit()
        self._res_g.setReadOnly(True)
        rf.addRow("g-factor:", self._res_g)
        self._res_desc = QLineEdit()
        self._res_desc.setReadOnly(True)
        rf.addRow("Description:", self._res_desc)
        layout.addWidget(res_grp)

        layout.addStretch()

        # Close button
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.close)
        layout.addWidget(btn_box)

        # Connect all inputs to auto-calculate
        self._single_radio.toggled.connect(self._on_mode_changed)
        self._sp_type.currentIndexChanged.connect(self._calculate)
        self._sp_orbital.currentIndexChanged.connect(self._calculate)
        self._tp_type1.currentIndexChanged.connect(self._calculate)
        self._tp_orbital1.currentIndexChanged.connect(self._calculate)
        self._tp_type2.currentIndexChanged.connect(self._calculate)
        self._tp_orbital2.currentIndexChanged.connect(self._calculate)
        self._tp_I.valueChanged.connect(self._calculate)
        self._gs_factor.valueChanged.connect(self._calculate)

        self._calculate()

    def _on_mode_changed(self):
        single = self._single_radio.isChecked()
        self._single_grp.setVisible(single)
        self._two_grp.setVisible(not single)
        self._calculate()

    def _calculate(self):
        try:
            gs = self._gs_factor.value()
            if self._single_radio.isChecked():
                spec = {
                    "type": self._sp_type.currentText(),
                    "orbital": self._sp_orbital.currentText(),
                }
                mu, desc = calculate_schmidt_moment(spec, g_s_factor=gs)
                orb = get_orbital(spec["orbital"])
                g = g_factor_from_mu_j(mu, orb["j"])
            else:
                I_val = self._tp_I.value()
                if I_val <= 0:
                    self._res_mu.setText("")
                    self._res_g.setText("")
                    self._res_desc.setText("Set I > 0")
                    return
                spec = {
                    "type1": self._tp_type1.currentText(),
                    "orbital1": self._tp_orbital1.currentText(),
                    "type2": self._tp_type2.currentText(),
                    "orbital2": self._tp_orbital2.currentText(),
                    "I": I_val,
                }
                mu, desc = calculate_schmidt_moment(spec, g_s_factor=gs)
                g = mu / I_val

            self._res_mu.setText(f"{mu:.6f}")
            self._res_g.setText(f"{g:.6f}")
            self._res_desc.setText(desc)
        except Exception as e:
            self._res_mu.setText(f"Error: {e}")
            self._res_g.setText("")
            self._res_desc.setText("")


# ══════════════════════════════════════════════════════════════════
#  Shell Configuration Plotter
# ══════════════════════════════════════════════════════════════════
_MAGIC_SPACING = {
    2: 1.5, 8: 2.1, 20: 2.5, 28: 2.2, 50: 2.8,
    82: 3.0, 126: 3.2, 184: 3.5,
}
_MAGIC_STYLE = {
    m: {"fontsize": 11, "circle_pad": 0.4, "lw": 1.0}
    for m in (2, 8, 20, 28, 50, 82, 126, 184)
}


class ShellConfigDialog(QDialog):
    """Standalone shell configuration plotter tool."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shell Configuration Plotter")
        self.setWindowIcon(lucide_icon("atom"))
        self.resize(900, 700)
        self.setModal(False)
        self._editor_dialog = None

        main_layout = QHBoxLayout(self)

        # ── Left panel: inputs ──
        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Nucleon inputs
        nuc_grp = QGroupBox("Nucleus")
        nf = QFormLayout(nuc_grp)
        self._z_spin = QSpinBox()
        self._z_spin.setRange(0, 200)
        self._z_spin.setValue(32)
        nf.addRow("Z (protons):", self._z_spin)
        self._n_spin = QSpinBox()
        self._n_spin.setRange(0, 200)
        self._n_spin.setValue(40)
        nf.addRow("N (neutrons):", self._n_spin)
        self._a_label = QLabel("72")
        nf.addRow("A (mass):", self._a_label)
        self._elem_label = QLabel("Ge")
        nf.addRow("Element:", self._elem_label)
        left_layout.addWidget(nuc_grp)

        # Plot options
        opt_grp = QGroupBox("Plot Options")
        of = QFormLayout(opt_grp)
        self._shell_start = QComboBox()
        self._shell_start.addItems(ORBITAL_NAMES)
        of.addRow("From shell:", self._shell_start)
        self._shell_end = QComboBox()
        self._shell_end.addItems(ORBITAL_NAMES)
        # Default to 1g7/2 (index 11) -- covers most nuclei up to Z,N~50
        _default_end = ORBITAL_NAMES.index("1g7/2") \
            if "1g7/2" in ORBITAL_NAMES else len(ORBITAL_NAMES) - 1
        self._shell_end.setCurrentIndex(_default_end)
        of.addRow("To shell:", self._shell_end)
        self._show_title = QCheckBox()
        self._show_title.setChecked(True)
        of.addRow("Show title:", self._show_title)
        self._show_legend = QCheckBox()
        of.addRow("Show legend:", self._show_legend)
        self._show_border = QCheckBox()
        self._show_border.setChecked(True)
        of.addRow("Show border:", self._show_border)
        self._disp_vacancies = QCheckBox()
        self._disp_vacancies.setChecked(True)
        of.addRow("Show vacancies:", self._disp_vacancies)
        self._vac_only_occ = QCheckBox()
        self._vac_only_occ.setChecked(True)
        of.addRow("Vacancies if occupied:", self._vac_only_occ)
        self._show_n_quantum = QCheckBox()
        self._show_n_quantum.setChecked(True)
        of.addRow("Show n quantum #:", self._show_n_quantum)
        self._draw_magic = QCheckBox()
        self._draw_magic.setChecked(True)
        of.addRow("Draw magic numbers:", self._draw_magic)
        self._circle_size = QSpinBox()
        self._circle_size.setRange(2, 20)
        self._circle_size.setValue(8)
        of.addRow("Circle size:", self._circle_size)
        left_layout.addWidget(opt_grp)

        # Results
        res_grp = QGroupBox("Shell Model Results")
        rf = QFormLayout(res_grp)
        self._res_pconfig = QLineEdit()
        self._res_pconfig.setReadOnly(True)
        rf.addRow("Proton config:", self._res_pconfig)
        self._res_nconfig = QLineEdit()
        self._res_nconfig.setReadOnly(True)
        rf.addRow("Neutron config:", self._res_nconfig)
        self._res_jpi = QLineEdit()
        self._res_jpi.setReadOnly(True)
        rf.addRow("Predicted J\u03c0:", self._res_jpi)
        left_layout.addWidget(res_grp)

        # Buttons
        btn_row = QHBoxLayout()
        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self._plot)
        btn_row.addWidget(plot_btn)
        self._edit_btn = QPushButton("Edit Plot")
        self._edit_btn.clicked.connect(self._open_editor)
        self._edit_btn.setEnabled(False)
        btn_row.addWidget(self._edit_btn)
        left_layout.addLayout(btn_row)

        left_layout.addStretch()
        main_layout.addWidget(left)

        # ── Right panel: canvas ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(5, 10))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        right_layout.addWidget(self._toolbar)
        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(False)
        right_layout.addWidget(scroll)
        main_layout.addWidget(right, 1)

        # Connect Z/N to auto-update info
        self._z_spin.valueChanged.connect(self._update_info)
        self._n_spin.valueChanged.connect(self._update_info)
        self._update_info()

    def _update_info(self):
        Z = self._z_spin.value()
        N = self._n_spin.value()
        A = Z + N
        self._a_label.setText(str(A))
        self._elem_label.setText(_get_element_symbol(Z))
        res = _nuclear_shell_model(Z, N)
        self._res_pconfig.setText(res["proton_config_str"])
        self._res_nconfig.setText(res["neutron_config_str"])
        jpi = res["spin_parity"]
        if res["coupled"]:
            jpi += f"  ({res['coupled']})"
        self._res_jpi.setText(jpi)

    def _plot(self):
        Z = self._z_spin.value()
        N = self._n_spin.value()
        res = _nuclear_shell_model(Z, N)
        self._draw_shell_diagram(
            res["proton_filled"], res["neutron_filled"], Z, N)
        self._edit_btn.setEnabled(True)

    def _draw_shell_diagram(self, proton_filled, neutron_filled, Z, N):
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        normal_spacing = 1.0
        line_gap = 0.10
        circle_size = self._circle_size.value()
        shell_lw = 3
        label_fs = 20
        display_n = self._show_n_quantum.isChecked()
        display_vac = self._disp_vacancies.isChecked()
        vac_only_occ = self._vac_only_occ.isChecked()

        # Determine orbital subset
        s_idx = self._shell_start.currentIndex()
        e_idx = self._shell_end.currentIndex()
        if s_idx > e_idx:
            s_idx, e_idx = e_idx, s_idx
        sub_orbs = ORBITALS[s_idx:e_idx + 1]

        # y positions
        y_vals, y = [], 0.0
        for orb in ORBITALS[:s_idx]:
            if orb["magic_gap_after"] and "magic_number" in orb:
                y += _MAGIC_SPACING.get(orb["magic_number"], 1.5)
            elif orb["magic_gap_after"]:
                y += 1.5
            else:
                y += normal_spacing
        for orb in sub_orbs:
            y_vals.append(y)
            if orb["magic_gap_after"] and "magic_number" in orb:
                gap = _MAGIC_SPACING.get(orb["magic_number"], 1.5)
            elif orb["magic_gap_after"]:
                gap = 1.5
            else:
                gap = normal_spacing
            y += gap

        # occupancy lookup
        p_occ = {o["name"]: occ for o, occ in proton_filled}
        n_occ = {o["name"]: occ for o, occ in neutron_filled}

        p_line = (-0.5, -line_gap)
        n_line = (line_gap, 0.5)
        p_cx = sum(p_line) / 2
        n_cx = sum(n_line) / 2
        dx = 0.04

        for orb, yv in zip(sub_orbs, y_vals):
            ax.plot(p_line, [yv, yv], "k-", lw=shell_lw)
            ax.plot(n_line, [yv, yv], "k-", lw=shell_lw)
            ax.text(0, yv,
                    _name_to_latex(orb["name"], display_n_quantum=display_n),
                    ha="center", va="center", fontsize=label_fs)

            cap = orb["capacity"]
            for occ, cx, color in (
                (p_occ.get(orb["name"], 0), p_cx, "red"),
                (n_occ.get(orb["name"], 0), n_cx, "blue"),
            ):
                if display_vac and (not vac_only_occ or occ > 0):
                    total_w = (cap - 1) * dx
                    x0 = cx - total_w / 2
                    for i in range(cap):
                        x = x0 + i * dx
                        ax.plot(x, yv, "o", markersize=circle_size,
                                markerfacecolor=color if i < occ else "white",
                                markeredgecolor=color,
                                markeredgewidth=1.5, linestyle="None")
                elif not display_vac and occ > 0:
                    total_w = (occ - 1) * dx
                    x0 = cx - total_w / 2
                    for i in range(occ):
                        x = x0 + i * dx
                        ax.plot(x, yv, "o", markersize=circle_size,
                                color=color)

        # Magic numbers
        if self._draw_magic.isChecked():
            for i, orb in enumerate(sub_orbs):
                if orb.get("magic_gap_after") and "magic_number" in orb:
                    mm = orb["magic_number"]
                    style = _MAGIC_STYLE.get(mm, {})
                    if i + 1 < len(y_vals):
                        y_next = y_vals[i + 1]
                    else:
                        y_next = y_vals[i] + 2.0
                    y_mid = 0.5 * (y_vals[i] + y_next)
                    ax.text(
                        0, y_mid, f"{mm}",
                        ha="center", va="center",
                        fontsize=style.get("fontsize", 11),
                        bbox=dict(
                            boxstyle=f"circle,pad="
                                     f"{style.get('circle_pad', 0.4)}",
                            ec="black", fc="white",
                            lw=style.get("lw", 1.0)))

        # Title
        if self._show_title.isChecked():
            elem = _get_element_symbol(Z)
            A = Z + N
            ax.set_title(
                rf"Shell Diagram for $^{{{A}}}_{{{Z}}}{elem}$"
                f"  (Z={Z}, N={N})",
                fontsize=14, pad=15)

        # Legend
        if self._show_legend.isChecked():
            import matplotlib.lines as mlines
            ax.legend(
                handles=[
                    mlines.Line2D([], [], marker="o", color="red",
                                  linestyle="None", label="Protons"),
                    mlines.Line2D([], [], marker="o", color="blue",
                                  linestyle="None", label="Neutrons"),
                ],
                loc="upper right", fontsize=12, frameon=False)

        # Border
        if not self._show_border.isChecked():
            for s in ax.spines.values():
                s.set_visible(False)

        ax.set_xlim(-0.6, 0.6)
        if y_vals:
            ax.set_ylim(y_vals[0] - 1, y_vals[-1] + 2)
        ax.set_xticks([])
        ax.set_yticks([])
        self._figure.tight_layout()

        dpi = self._figure.get_dpi()
        w = self._figure.get_figwidth()
        h = self._figure.get_figheight()
        self._canvas.setFixedSize(int(w * dpi), int(h * dpi))
        self._canvas.draw()

    def _open_editor(self):
        if (self._editor_dialog is not None
                and self._editor_dialog.isVisible()):
            self._editor_dialog.raise_()
            self._editor_dialog.activateWindow()
            return
        self._editor_dialog = PlotEditorDialog(
            self._figure, self._canvas, parent=self)
        self._editor_dialog.show()


# ══════════════════════════════════════════════════════════════════
#  Settings helpers (shared across tabs)
# ══════════════════════════════════════════════════════════════════

_DEFAULT_SETTINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings")
_OLD_SETTINGS_PATHS = [
    os.path.expanduser("~/.cls_toolkit/settings.yaml"),
    os.path.join(os.path.expanduser("~"), "denis", "settings.yaml"),
]


def _get_settings_path():
    """Return the active settings file path, migrating from old locations."""
    new_path = os.path.join(_DEFAULT_SETTINGS_DIR, "settings.yaml")
    # Migrate: if an old path exists but new doesn't, copy it
    if not os.path.isfile(new_path):
        for old in _OLD_SETTINGS_PATHS:
            if os.path.isfile(old):
                os.makedirs(_DEFAULT_SETTINGS_DIR, exist_ok=True)
                import shutil
                shutil.copy2(old, new_path)
                break
    return new_path


def _load_settings():
    path = _get_settings_path()
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def _save_settings(settings):
    path = _get_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)


# ── Unified output directory layout ───────────────────────────────────
#
# A single output root holds ``estimates/`` and ``analysis/`` subfolders.
# These helpers are the single source of truth for output paths -- update
# them and every output path follows.

def get_output_root():
    """Root output directory (settings key ``output_directory``).

    The "Open output folder" menu in the main window opens this
    directory; estimates and analysis results live in subfolders
    inside it. Users who had ``./analysis_output`` set on an older
    settings file keep their existing root -- ``estimates/`` and
    ``analysis/`` get created inside it on first write.
    """
    return _load_settings().get("output_directory", "./output")


def get_estimates_dir():
    """``<root>/estimates`` -- where the Estimate tab writes."""
    return os.path.join(get_output_root(), "estimates")


def get_analysis_dir():
    """``<root>/analysis`` -- where the Analysis tab (project iters,
    IS-tab exports, etc.) writes."""
    return os.path.join(get_output_root(), "analysis")


# ── Last-used directory helpers (per kind / direction) ────────────────
#
# Four independent directories are tracked so save and load can diverge,
# and so config files don't drag the data-file picker into a config
# folder (or vice-versa). All file-dialog call sites should seed their
# initial directory with `get_last_dir(kind, action)` and call
# `remember_last_dir(kind, action, chosen_path)` after a successful pick.
# Falls back silently to '' (current working directory) when the saved
# path no longer exists on disk.

_DIR_KEYS = {
    ("config", "save"): "last_config_save_dir",
    ("config", "load"): "last_config_load_dir",
    ("data",   "save"): "last_data_save_dir",
    ("data",   "load"): "last_data_load_dir",
}


def get_last_dir(kind, action):
    """Return the remembered directory for (kind, action) or ''.

    Silently drops paths that no longer exist on disk. For data load,
    falls back to the legacy `last_data_dir` setting (used by older
    versions) so existing users don't lose their picker context.
    """
    if (kind, action) not in _DIR_KEYS:
        return ""
    settings = _load_settings()
    key = _DIR_KEYS[(kind, action)]
    p = settings.get(key, "") or ""
    if p and os.path.isdir(p):
        return p
    if kind == "data":
        legacy = settings.get("last_data_dir", "") or ""
        if legacy and os.path.isdir(legacy):
            return legacy
    return ""


def remember_last_dir(kind, action, path):
    """Persist the directory of `path` as the last-used dir for
    (kind, action). No-op when path is empty/None.
    """
    if not path or (kind, action) not in _DIR_KEYS:
        return
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if not d:
        return
    settings = _load_settings()
    settings[_DIR_KEYS[(kind, action)]] = d
    # Keep the legacy key in sync for backward compatibility with any
    # call site we haven't migrated yet.
    if kind == "data" and action == "load":
        settings["last_data_dir"] = d
    _save_settings(settings)


def convert_path_for_os(path_str):
    """Convert a file path between Windows and Linux/macOS conventions.

    If the current OS is Windows and the path looks like a POSIX path
    (starts with ``/``), convert it.  If the current OS is POSIX and the
    path looks like a Windows path (contains ``\\`` or a drive letter),
    convert it.  Nextcloud / network-drive mount-point differences are
    handled via simple prefix replacement heuristics.

    Returns the converted path, or the original if no conversion is needed.
    """
    import platform
    import re

    if not path_str or not isinstance(path_str, str):
        return path_str

    is_windows = platform.system() == "Windows"

    if is_windows:
        # POSIX → Windows
        if path_str.startswith("/"):
            # Try Nextcloud heuristic first (handles /home/user/Nextcloud/...)
            home = os.path.expanduser("~")
            if "/Nextcloud/" in path_str:
                rel = path_str.split("/Nextcloud/", 1)[1]
                candidate = os.path.join(home, "Nextcloud",
                                         rel.replace("/", "\\"))
                if os.path.exists(candidate) or os.path.exists(
                        os.path.dirname(candidate)):
                    return candidate
            # /c/Users/... → C:\Users\...  (Git Bash / MSYS convention)
            m = re.match(r"^/([a-zA-Z])/(.*)", path_str)
            if m:
                drive = m.group(1).upper()
                rest = m.group(2).replace("/", "\\")
                return f"{drive}:\\{rest}"
            # generic: just flip slashes
            return path_str.replace("/", "\\")
    else:
        # Windows → POSIX
        # Normalise backslashes for heuristic matching
        normalised = path_str.replace("\\", "/")
        # Try Nextcloud heuristic first (handles C:\Users\...\Nextcloud\...)
        if "Nextcloud/" in normalised:
            home = os.path.expanduser("~")
            rel = normalised.split("Nextcloud/", 1)[-1]
            candidate = os.path.join(home, "Nextcloud", rel)
            if os.path.exists(candidate) or os.path.exists(
                    os.path.dirname(candidate)):
                return candidate
        # C:\Users\... → /c/Users/...
        m = re.match(r"^([A-Za-z]):\\(.*)", path_str)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2).replace("\\", "/")
            return f"/{drive}/{rest}"
        if "\\" in path_str:
            return normalised

    return path_str


def maybe_convert_path(path_str):
    """Convert path if auto_path_conversion is enabled in settings."""
    settings = _load_settings()
    if not settings.get("auto_path_conversion", True):
        return path_str
    converted = convert_path_for_os(path_str)
    return converted


# ══════════════════════════════════════════════════════════════════
#  Default plot settings
# ══════════════════════════════════════════════════════════════════

# 2026-07-25 (Arda): bolder on-screen defaults — thicker lines/axes/
# ticks, larger axis labels — and minor ticks visible on EVERY plot
# (UI and saved outputs) by default.
_DEFAULT_PLOT_SETTINGS = {
    "lines.linewidth": 2.5,
    "lines.markersize": 6.0,
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "errorbar.capsize": 3.0,
}

# Per-plot-type defaults for analysis tab plots
_DEFAULT_PLOT_TYPE_SETTINGS = {
    "fit_plot": {
        # Base style + region-based auto-sizing. The plot picks one of
        # {low, med, high} sub-styles based on the highest-count bin in
        # the data, then falls back to the unprefixed keys for anything
        # that region doesn't override. Toggle off to disable scaling.
        "auto_size_by_counts": True,
        "low_count_max": 200,
        "med_count_max": 3000,
        # ── Low-count region (big markers, visible caps) ──
        "low_data_ms": 6.0,
        "low_data_alpha": 0.85,
        "low_data_capsize": 3.0,
        "low_data_elinewidth": 1.2,
        "low_fit_alpha": 0.9,
        # ── Medium-count region ──
        "med_data_ms": 3.5,
        "med_data_alpha": 0.7,
        "med_data_capsize": 2.0,
        "med_data_elinewidth": 1.0,
        "med_fit_alpha": 0.92,
        # ── High-count region (tiny markers, no caps, thin lines) ──
        "high_data_ms": 2.0,
        "high_data_alpha": 0.5,
        "high_data_capsize": 0.0,
        "high_data_elinewidth": 0.7,
        "high_fit_alpha": 0.95,
        # Base style (shared across regions) — used when auto-sizing is
        # off, or to fill in any missing per-region keys.
        "data_fmt": "k.",
        "data_ms": 4.0,
        "data_alpha": 0.7,
        "data_capsize": 2.0,
        "data_elinewidth": 1.0,
        "data_ecolor": "0.55",
        "fit_color": "red",
        "fit_lw": 3.0,
        "fit_alpha": 0.92,
        # Shaded area under the fit curve ("auto" = pale fit color).
        "shade_under_fit": True,
        "shade_color": "auto",
        "shade_alpha": 0.12,
        # On-plot fit-values box (which params: Output block picker).
        "values_box_fontsize": 11,
        "resid_fmt": "k.",
        "resid_ms": 4.0,
        "band_alpha": 0.15,
        "grid_alpha": 0.3,
        "figsize_w": 8.0,
        "figsize_h": 5.0,
        "title_size": 14,
        "label_size": 14,
    },
    "walk_plot": {
        "trace_alpha": 0.3,
        "trace_lw": 0.6,
        "label_size": 11,
        "tick_size": 9,
        "title_size": 12,
        "figsize_w": 9.0,
        "panel_h": 2.2,
    },
    "correlation_plot": {
        "hist_bins": 40,
        "hist_color": "steelblue",
        "hist_alpha": 0.7,
        "scatter_s": 0.5,
        "scatter_alpha": 0.25,
        "scatter_color": "black",
        "label_size": 9,
        "tick_size": 8,
        "title_size": 12,
        "cell_size": 2.8,
    },
    "chisq_map": {
        "line_fmt": "b.-",
        "line_ms": 4.0,
        "best_color": "red",
        "best_ls": "--",
        "label_size": 11,
        "tick_size": 9,
        "title_size": 12,
        "panel_w": 3.5,
        "panel_h": 3.5,
    },
    "tracker_plot": {
        # 2026-07-25 academic restyle: unconnected points with caps, a
        # weighted-mean reference line and a HORIZONTAL ±1σ band (the
        # old ".-" polygon band connecting error bars read as noise).
        "marker_fmt": "o",
        "marker_ms": 5.5,
        "capsize": 3.0,
        "elinewidth": 1.2,
        "connect_runs": False,
        "mean_line": True,
        "mean_color": "0.25",
        "mean_ls": "--",
        "band_alpha": 0.15,
        "grid_alpha": 0.3,
        "title_size": 14,
        "label_size": 14,
        "figsize_w": 9.0,
        "panel_h": 3.2,
    },
    "preview": {
        "data_fmt": "o",
        "data_ms": 4.0,
        "data_capsize": 3.0,
        "figsize_w": 8.0,
        "figsize_h": 4.5,
        "title_size": 14,
        "label_size": 14,
    },
    "isotope_shift_plot": {
        "data_fmt": ".",
        "data_ms": 5.0,
        "data_alpha": 0.7,
        "data_capsize": 3.0,
        "fit_lw": 2.5,
        "fit_alpha": 0.85,
        # Boxed per-isotope labels + shading under the fitted curves,
        # matching the reference stacked-spectra style.
        "boxed_labels": True,
        "shade_under_fit": True,
        "shade_color": "auto",
        "shade_alpha": 0.12,
        "centroid_ls": "--",
        "centroid_alpha": 0.5,
        "arrow_color": "#333333",
        "arrow_lw": 1.5,
        "grid_alpha": 0.3,
        "figsize_w": 10.0,
        "figsize_h": 8.0,
        "title_size": 14,
        "label_size": 14,
    },
}

# Static style settings (not user-configurable)
_STATIC_PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif",
                    "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
}


def apply_plot_settings():
    """Apply plot defaults from settings (or built-in defaults) to rcParams."""
    import matplotlib.pyplot as _plt
    settings = _load_settings()
    user_plot = settings.get("plot_defaults", {})
    merged = dict(_DEFAULT_PLOT_SETTINGS)
    merged.update(user_plot)
    _plt.rcParams.update(merged)
    # Static style applied LAST so it can't be overridden by user YAML edits
    _plt.rcParams.update(_STATIC_PLOT_STYLE)


def get_plot_type_settings(plot_type):
    """Get merged per-plot-type settings (user overrides + defaults).

    Parameters
    ----------
    plot_type : str
        One of: ``fit_plot``, ``walk_plot``, ``correlation_plot``,
        ``chisq_map``, ``tracker_plot``, ``preview``.

    Returns
    -------
    dict
        Settings dict with all keys populated.
    """
    defaults = _DEFAULT_PLOT_TYPE_SETTINGS.get(plot_type, {})
    settings = _load_settings()
    user = settings.get("plot_type_defaults", {}).get(plot_type, {})
    merged = dict(defaults)
    merged.update(user)
    return merged


def fit_plot_settings_for_counts(max_count):
    """Return the effective fit-plot style dict given the highest-count
    bin in the data. When `auto_size_by_counts` is on, picks one of
    low / medium / high regions based on `low_count_max` and
    `med_count_max`, and overlays its `<region>_*` keys on top of the
    base style. Returns the regular fit_plot settings unchanged when
    auto-sizing is disabled.

    The mapping is::

        max_count <= low_count_max          → "low"  region
        max_count <= med_count_max          → "med"  region
        max_count >  med_count_max          → "high" region
    """
    base = get_plot_type_settings("fit_plot")
    if not base.get("auto_size_by_counts", True):
        return base
    try:
        c = float(max_count) if max_count is not None else 0.0
    except (TypeError, ValueError):
        c = 0.0
    low_max = float(base.get("low_count_max", 200))
    med_max = float(base.get("med_count_max", 3000))
    if c <= low_max:
        prefix = "low_"
    elif c <= med_max:
        prefix = "med_"
    else:
        prefix = "high_"
    out = dict(base)
    for k, v in base.items():
        if k.startswith(prefix):
            tail = k[len(prefix):]   # strip "low_" / "med_" / "high_"
            out[tail] = v
    return out


# Per-key tooltips for the Plot Defaults UI. Keyed by plot type, then by
# settings key. Missing keys fall back to a generic label-style tooltip.
_PLOT_TYPE_TOOLTIPS = {
    "fit_plot": {
        "auto_size_by_counts": (
            "Pick marker size, alpha and error-bar style automatically "
            "based on the highest-count bin in the run.\n"
            "Off → use the base values below for every fit, regardless "
            "of statistics."),
        "low_count_max": (
            "Highest peak ≤ this count → 'Low' region (big markers,\n"
            "high alpha, visible error-bar caps). Sparse / low-stats "
            "scans."),
        "med_count_max": (
            "Highest peak ≤ this count → 'Medium' region.\n"
            "Above this threshold → 'High' region (tiny markers, "
            "transparent, no caps)."),
        "low_data_ms":          "Marker size (pt) for low-count fits.",
        "low_data_alpha":       "Marker / error-bar opacity for low-count fits.",
        "low_data_capsize":     "Error-bar cap length (pt) for low-count fits.",
        "low_data_elinewidth":  "Error-bar line width (pt) for low-count fits.",
        "low_fit_alpha":        "Fit-line opacity for low-count fits.",
        "med_data_ms":          "Marker size (pt) for medium-count fits.",
        "med_data_alpha":       "Marker / error-bar opacity for medium-count fits.",
        "med_data_capsize":     "Error-bar cap length (pt) for medium-count fits.",
        "med_data_elinewidth":  "Error-bar line width (pt) for medium-count fits.",
        "med_fit_alpha":        "Fit-line opacity for medium-count fits.",
        "high_data_ms":         "Marker size (pt) for high-count fits.",
        "high_data_alpha":      "Marker / error-bar opacity for high-count fits.",
        "high_data_capsize":    "Error-bar cap length (pt) for high-count fits.",
        "high_data_elinewidth": "Error-bar line width (pt) for high-count fits.",
        "high_fit_alpha":       "Fit-line opacity for high-count fits.",
        "data_fmt":      "Matplotlib fmt string for data points (e.g. 'k.').",
        "data_ms":       "Fallback marker size when auto-size is off.",
        "data_alpha":    "Fallback marker alpha when auto-size is off.",
        "data_capsize":  "Fallback error-bar cap size when auto-size is off.",
        "data_elinewidth": "Fallback error-bar line width when auto-size "
                           "is off.",
        "data_ecolor":   "Error-bar color (e.g. '0.55' = mid-gray).",
        "fit_color":     "Fit-line color.",
        "fit_lw":        "Fit-line width (pt).",
        "fit_alpha":     "Fallback fit-line alpha when auto-size is off.",
        "shade_under_fit": "Fill the area under the fit curve with a "
                           "pale shade (publication style).",
        "shade_color":   "Shade color under the fit curve; 'auto' = "
                         "the fit-line color.",
        "shade_alpha":   "Opacity of the shaded area under the fit "
                         "curve (0-1).",
        "values_box_fontsize": "Font size (pt) of the on-plot fit-values "
                               "box. Which parameters appear is picked "
                               "in the Output block ('Fit values on "
                               "plot').",
        "resid_fmt":     "Matplotlib fmt string for residual points.",
        "resid_ms":      "Marker size (pt) for residual points.",
        "band_alpha":    "Confidence-band fill opacity.",
        "grid_alpha":    "Grid-line opacity.",
        "figsize_w":     "Figure width (in).",
        "figsize_h":     "Figure height (in).",
        "title_size":    "Subplot title font size (pt).",
        "label_size":    "Axis label font size (pt).",
    },
}


def get_plot_type_tooltip(plot_type, key):
    return _PLOT_TYPE_TOOLTIPS.get(plot_type, {}).get(key, "")


# Optional grouping for the Plot Defaults dialog. When a plot type has
# an entry here, the settings dialog renders each {title, keys} group as
# a QGroupBox; types without an entry stay in the historical flat
# layout. The `note` field, if present, becomes a small info label at
# the top of the section. `keys` controls both inclusion and order.
_PLOT_TYPE_SECTIONS = {
    "fit_plot": [
        {
            "title": "Auto-sizing thresholds",
            "note": "Plot picks one of {Low, Medium, High} based on the "
                    "highest-count bin in each run.",
            "keys": ["auto_size_by_counts",
                     "low_count_max", "med_count_max"],
        },
        {
            "title": "Low-count region",
            "note": "Region applied when the highest-count bin ≤ "
                    "Low → Medium threshold.",
            "keys": ["low_data_ms", "low_data_alpha", "low_data_capsize",
                     "low_data_elinewidth", "low_fit_alpha"],
        },
        {
            "title": "Medium-count region",
            "note": "Region applied between Low → Medium and "
                    "Medium → High thresholds.",
            "keys": ["med_data_ms", "med_data_alpha", "med_data_capsize",
                     "med_data_elinewidth", "med_fit_alpha"],
        },
        {
            "title": "High-count region",
            "note": "Region applied when the highest-count bin > "
                    "Medium → High threshold (no upper limit).",
            "keys": ["high_data_ms", "high_data_alpha", "high_data_capsize",
                     "high_data_elinewidth", "high_fit_alpha"],
        },
        {
            "title": "Base style (used when auto-sizing is off, "
                     "or as fallback)",
            "keys": ["data_fmt", "data_ms", "data_alpha", "data_capsize",
                     "data_elinewidth", "data_ecolor", "fit_color",
                     "fit_lw", "fit_alpha", "resid_fmt", "resid_ms",
                     "band_alpha", "grid_alpha"],
        },
        {
            "title": "Shade under fit / values box",
            "keys": ["shade_under_fit", "shade_color", "shade_alpha",
                     "values_box_fontsize"],
        },
        {
            "title": "Layout",
            "keys": ["figsize_w", "figsize_h", "title_size", "label_size"],
        },
    ],
}


# Optional pretty labels for the Plot Defaults dialog. Replaces the
# auto-prettified "Snake_case → Title Case" rendering on the form rows
# so renamed thresholds and per-region marker sizes read naturally.
_PLOT_TYPE_LABELS = {
    "fit_plot": {
        "auto_size_by_counts":   "Auto-size by counts",
        "low_count_max":         "Low → Medium threshold",
        "med_count_max":         "Medium → High threshold",
        "low_data_ms":           "Marker size",
        "low_data_alpha":        "Marker / errorbar alpha",
        "low_data_capsize":      "Errorbar cap size",
        "low_data_elinewidth":   "Errorbar line width",
        "low_fit_alpha":         "Fit-line alpha",
        "med_data_ms":           "Marker size",
        "med_data_alpha":        "Marker / errorbar alpha",
        "med_data_capsize":      "Errorbar cap size",
        "med_data_elinewidth":   "Errorbar line width",
        "med_fit_alpha":         "Fit-line alpha",
        "high_data_ms":          "Marker size",
        "high_data_alpha":       "Marker / errorbar alpha",
        "high_data_capsize":     "Errorbar cap size",
        "high_data_elinewidth":  "Errorbar line width",
        "high_fit_alpha":        "Fit-line alpha",
        "data_fmt":              "Marker fmt string",
        "data_ms":               "Marker size",
        "data_alpha":            "Marker alpha",
        "data_capsize":          "Errorbar cap size",
        "data_elinewidth":       "Errorbar line width",
        "data_ecolor":           "Errorbar color",
        "fit_color":             "Fit-line color",
        "fit_lw":                "Fit-line width",
        "fit_alpha":             "Fit-line alpha",
        "shade_under_fit":       "Shade under fit",
        "shade_color":           "Shade color",
        "shade_alpha":           "Shade alpha",
        "values_box_fontsize":   "Values-box font size (pt)",
        "resid_fmt":             "Residual marker fmt",
        "resid_ms":              "Residual marker size",
        "band_alpha":            "Confidence-band alpha",
        "grid_alpha":            "Grid alpha",
        "figsize_w":             "Figure width (in)",
        "figsize_h":             "Figure height (in)",
        "title_size":            "Title size (pt)",
        "label_size":            "Axis label size (pt)",
    },
}


def get_plot_type_sections(plot_type):
    return _PLOT_TYPE_SECTIONS.get(plot_type)


def get_plot_type_label(plot_type, key, fallback=None):
    label = _PLOT_TYPE_LABELS.get(plot_type, {}).get(key)
    if label:
        return label
    return fallback or key.replace("_", " ").title()


# ══════════════════════════════════════════════════════════════════
#  Output Plot Options dialog (detailed per-plot-type appearance)
# ══════════════════════════════════════════════════════════════════

# Global rcParams spec for the dialog's Global tab: (key, label, min,
# max, step) — an int spinbox when step is None, a checkbox when step
# is "bool". The x/y tick pairs are edited via ONE control and
# mirrored on save, matching the Settings dialog's behavior.
_GLOBAL_PLOT_SPEC = [
    ("lines.linewidth", "Line width", 0.5, 10.0, 0.5),
    ("lines.markersize", "Marker size", 0.5, 20.0, 0.5),
    ("font.size", "Font size", 6, 30, None),
    ("axes.labelsize", "Axis label size", 6, 30, None),
    ("axes.titlesize", "Title size", 6, 30, None),
    ("axes.linewidth", "Axes line width", 0.2, 5.0, 0.1),
    ("xtick.labelsize", "Tick label size", 6, 24, None),
    ("xtick.major.size", "Major tick size", 1.0, 15.0, 0.5),
    ("xtick.minor.size", "Minor tick size", 0.5, 10.0, 0.5),
    ("xtick.major.width", "Major tick width", 0.2, 3.0, 0.1),
    ("xtick.minor.width", "Minor tick width", 0.2, 3.0, 0.1),
    ("xtick.minor.visible", "Minor ticks visible", None, None, "bool"),
    ("legend.fontsize", "Legend font size", 6, 24, None),
    ("errorbar.capsize", "Errorbar cap size", 0.0, 10.0, 0.5),
    ("figure.dpi", "Figure DPI", 72, 300, None),
    ("savefig.dpi", "Save DPI", 72, 600, None),
]

# One control edits both axes for these:
_TICK_MIRROR_KEYS = {
    "xtick.labelsize": "ytick.labelsize",
    "xtick.major.size": "ytick.major.size",
    "xtick.minor.size": "ytick.minor.size",
    "xtick.major.width": "ytick.major.width",
    "xtick.minor.width": "ytick.minor.width",
    "xtick.minor.visible": "ytick.minor.visible",
}

_PLOT_TYPE_TAB_LABELS = {
    "fit_plot": "Fit Plot",
    "walk_plot": "Walk Plot",
    "correlation_plot": "Correlation",
    "chisq_map": "Chi-sq Map",
    "tracker_plot": "Tracker",
    "preview": "Preview / TOF",
    "isotope_shift_plot": "Isotope Shifts",
}


def format_fit_value_lines(params_records, selected_keys):
    """Format 'name = value ± error' lines for the on-plot values box.

    ``params_records`` is a fit result's ``params_df`` payload (dict of
    columns or list of row dicts with Parameter / Value / Stderr /
    Model). ``selected_keys`` holds "Model:Parameter" strings from the
    Output block picker. Parameter names are shown with the first
    letter capitalized; the model prefix is added only when the same
    parameter name is selected for more than one model.
    """
    import pandas as pd
    if not selected_keys:
        return []
    try:
        df = pd.DataFrame(params_records)
    except Exception:
        return []
    if df.empty or "Parameter" not in df.columns:
        return []
    sel = list(selected_keys)
    rows = []
    for _, row in df.iterrows():
        model = str(row.get("Model", ""))
        pname = str(row.get("Parameter", ""))
        if f"{model}:{pname}" in sel:
            rows.append((model, pname, row.get("Value"),
                         row.get("Stderr")))
    name_counts = {}
    for _m, pname, _v, _e in rows:
        name_counts[pname] = name_counts.get(pname, 0) + 1

    def _fmt(v, e):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = float("nan")
        if not np.isfinite(e) or e <= 0:
            return f"{v:.1f}"
        # Two significant digits on the error; value matched to it.
        import math
        dec = max(0, 1 - int(math.floor(math.log10(e))))
        return f"{v:.{dec}f} ± {e:.{dec}f}"

    lines = []
    for model, pname, v, e in rows:
        s = _fmt(v, e)
        if s is None:
            continue
        disp = pname[:1].upper() + pname[1:] if pname else pname
        if name_counts.get(pname, 0) > 1:
            disp = f"{model}: {disp}"
        lines.append(f"{disp} = {s}")
    return lines

# Count-threshold keys carry raw photon counts (can exceed 1e6).
_PLOT_COUNT_THRESHOLD_KEYS = {"low_count_max", "med_count_max"}


def _make_plot_setting_widget(default_v, val, tip, key=None):
    """Widget for one plot-settings entry (bool checked BEFORE int —
    bool subclasses int and would otherwise render as a spinbox)."""
    if isinstance(default_v, bool):
        w = QCheckBox()
        w.setChecked(bool(val))
    elif isinstance(default_v, float):
        w = QDoubleSpinBox()
        w.setRange(0.0, 100.0)
        w.setSingleStep(0.5)
        w.setDecimals(2)
        w.setValue(float(val))
    elif isinstance(default_v, int):
        w = QSpinBox()
        w.setRange(0, 2_147_483_647
                   if key in _PLOT_COUNT_THRESHOLD_KEYS else 1_000_000)
        w.setValue(int(val))
    elif isinstance(default_v, str):
        w = QLineEdit(str(val))
    else:
        return None
    if tip:
        w.setToolTip(tip)
    return w


class PlotTypeOptionsDialog(QDialog):
    """Detailed appearance settings for every generated plot type.

    Opened from the Analysis Output block's "Plot Options…" button.
    Edits the SAME persisted keys as Settings ▸ Plot Defaults
    (``plot_defaults`` + ``plot_type_defaults`` in settings.yaml):
    global fonts / major & minor ticks / line widths on the Global
    tab, plus per-type marker styles, colors, alphas and panel sizes
    on one tab per plot type. Apply/OK saves app-wide and re-applies
    matplotlib defaults, so plots generated from then on use the new
    style. (The per-type editor intentionally mirrors SettingsDialog's
    generic builder — new keys added to ``_DEFAULT_PLOT_TYPE_SETTINGS``
    appear in both UIs automatically.)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Output Plot Options")
        self.setMinimumSize(560, 640)

        outer = QVBoxLayout(self)
        settings = _load_settings()

        tabs = QTabWidget()
        outer.addWidget(tabs, 1)

        # ── Global tab (rcParams incl. major/minor ticks) ──
        self._global_widgets = {}
        user_plot = settings.get("plot_defaults", {})
        glob_w = QWidget()
        form = QFormLayout(glob_w)
        form.setVerticalSpacing(8)
        form.setContentsMargins(8, 12, 8, 8)
        for key, label, lo, hi, step in _GLOBAL_PLOT_SPEC:
            default_v = _DEFAULT_PLOT_SETTINGS.get(key, 0)
            val = user_plot.get(key, default_v)
            if step == "bool":
                w = QCheckBox()
                w.setChecked(bool(val))
            elif step is None:
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setValue(int(val))
            else:
                w = QDoubleSpinBox()
                w.setRange(float(lo), float(hi))
                w.setSingleStep(float(step))
                w.setValue(float(val))
            if key in _TICK_MIRROR_KEYS:
                w.setToolTip("Applies to both x and y axes.")
            form.addRow(f"{label}:", w)
            self._global_widgets[key] = w
        glob_scroll = QScrollArea()
        glob_scroll.setWidgetResizable(True)
        glob_scroll.setWidget(glob_w)
        tabs.addTab(glob_scroll, "Global")

        # ── Per-plot-type tabs ──
        user_pt = settings.get("plot_type_defaults", {})
        self._pt_widgets = {}
        for pt_key, pt_label in _PLOT_TYPE_TAB_LABELS.items():
            defaults = _DEFAULT_PLOT_TYPE_SETTINGS.get(pt_key, {})
            user_vals = user_pt.get(pt_key, {})
            sections = get_plot_type_sections(pt_key)
            widgets = {}
            tab_w = QWidget()

            def _add_rows(form_, keys):
                for k in keys:
                    default_v = defaults[k]
                    val = user_vals.get(k, default_v)
                    tip = get_plot_type_tooltip(pt_key, k)
                    w = _make_plot_setting_widget(default_v, val, tip,
                                                  key=k)
                    if w is None:
                        continue
                    label_w = QLabel(f"{get_plot_type_label(pt_key, k)}:")
                    if tip:
                        label_w.setToolTip(tip)
                    form_.addRow(label_w, w)
                    widgets[k] = w

            if sections:
                sec_outer = QVBoxLayout(tab_w)
                sec_outer.setContentsMargins(8, 12, 8, 8)
                sec_outer.setSpacing(10)
                listed = set()
                for sec in sections:
                    grp = QGroupBox(sec.get("title", ""))
                    grp_lay = QVBoxLayout(grp)
                    grp_lay.setContentsMargins(8, 12, 8, 8)
                    grp_lay.setSpacing(6)
                    note = sec.get("note")
                    if note:
                        note_label = QLabel(note)
                        note_label.setWordWrap(True)
                        note_label.setObjectName("sectionNote")
                        grp_lay.addWidget(note_label)
                    sec_form = QFormLayout()
                    sec_form.setVerticalSpacing(6)
                    grp_lay.addLayout(sec_form)
                    keys = [k for k in sec.get("keys", [])
                            if k in defaults]
                    listed.update(keys)
                    _add_rows(sec_form, keys)
                    sec_outer.addWidget(grp)
                others = [k for k in defaults if k not in listed]
                if others:
                    grp = QGroupBox("Other")
                    of = QFormLayout(grp)
                    of.setVerticalSpacing(6)
                    of.setContentsMargins(8, 12, 8, 8)
                    _add_rows(of, others)
                    sec_outer.addWidget(grp)
                sec_outer.addStretch()
            else:
                flat = QFormLayout(tab_w)
                flat.setVerticalSpacing(8)
                flat.setContentsMargins(8, 12, 8, 8)
                _add_rows(flat, list(defaults))

            self._pt_widgets[pt_key] = widgets
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(tab_w)
            tabs.addTab(scroll, pt_label)

        # ── Buttons ──
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.Apply
                    ).clicked.connect(self.apply_settings)
        btns.button(QDialogButtonBox.StandardButton.RestoreDefaults
                    ).clicked.connect(self._restore_defaults)
        outer.addWidget(btns)

    def _widget_value(self, w):
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            return w.value()
        if isinstance(w, QLineEdit):
            return w.text()
        return None

    def _set_widget_value(self, w, val):
        if isinstance(w, QCheckBox):
            w.setChecked(bool(val))
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(val)
        elif isinstance(w, QLineEdit):
            w.setText(str(val))

    def apply_settings(self):
        """Persist the edited values app-wide and re-apply matplotlib
        defaults so subsequently generated plots use them."""
        settings = _load_settings()  # preserve unrelated keys
        pd = dict(settings.get("plot_defaults", {}))
        for key, w in self._global_widgets.items():
            val = self._widget_value(w)
            pd[key] = val
            mirror = _TICK_MIRROR_KEYS.get(key)
            if mirror:
                pd[mirror] = val
        settings["plot_defaults"] = pd

        pt_out = dict(settings.get("plot_type_defaults", {}))
        for pt_key, widgets in self._pt_widgets.items():
            d = dict(pt_out.get(pt_key, {}))
            for k, w in widgets.items():
                d[k] = self._widget_value(w)
            pt_out[pt_key] = d
        settings["plot_type_defaults"] = pt_out

        _save_settings(settings)
        apply_plot_settings()

    def _on_ok(self):
        self.apply_settings()
        self.accept()

    def _restore_defaults(self):
        """Reset every widget to the built-in defaults (not saved
        until Apply/OK)."""
        for key, w in self._global_widgets.items():
            self._set_widget_value(w, _DEFAULT_PLOT_SETTINGS.get(key, 0))
        for pt_key, widgets in self._pt_widgets.items():
            defaults = _DEFAULT_PLOT_TYPE_SETTINGS.get(pt_key, {})
            for k, w in widgets.items():
                if k in defaults:
                    self._set_widget_value(w, defaults[k])


# ══════════════════════════════════════════════════════════════════
#  Unit Converter Dialog
# ══════════════════════════════════════════════════════════════════

_H_PLANCK = 6.62607015e-34   # J s
_EV_TO_J = 1.602176634e-19   # J per eV
_C = 299792458.0              # m/s

_UNIT_NAMES = ["nm", "cm\u207b\u00b9", "MHz", "GHz", "THz", "eV"]


def _wavelength_m_to_all(lam_m):
    """Convert wavelength in metres to all supported units."""
    if lam_m <= 0:
        return {u: 0.0 for u in _UNIT_NAMES}
    return {
        "nm": lam_m * 1e9,
        "cm\u207b\u00b9": 1.0 / (lam_m * 100.0),
        "MHz": _C / lam_m / 1e6,
        "GHz": _C / lam_m / 1e9,
        "THz": _C / lam_m / 1e12,
        "eV": _H_PLANCK * _C / lam_m / _EV_TO_J,
    }


def _unit_to_wavelength_m(value, unit):
    """Convert a value in the given unit to wavelength in metres."""
    if value <= 0:
        return 0.0
    if unit == "nm":
        return value * 1e-9
    if unit == "cm\u207b\u00b9":
        return 1.0 / (value * 100.0)
    if unit == "MHz":
        return _C / (value * 1e6)
    if unit == "GHz":
        return _C / (value * 1e9)
    if unit == "THz":
        return _C / (value * 1e12)
    if unit == "eV":
        return _H_PLANCK * _C / (value * _EV_TO_J)
    return 0.0


class UnitConverterDialog(QDialog):
    """Wavelength / frequency / energy unit converter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unit Converter")
        self.setWindowIcon(lucide_icon("ruler"))
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        # Input
        in_grp = QGroupBox("Input")
        in_form = QFormLayout(in_grp)
        self._in_spin = ScientificSpinBox()
        self._in_spin.setRange(1e-15, 1e15)
        self._in_spin.setValue(532.0)
        self._in_spin.setToolTip(
            "Enter the numeric value to convert.\n"
            "Supports scientific notation (e.g. 3.2e14).")
        self._in_unit = QComboBox()
        self._in_unit.addItems(_UNIT_NAMES)
        self._in_unit.setToolTip(
            "Select the unit of the input value.\n"
            "All other units are computed automatically.")
        row = QHBoxLayout()
        row.addWidget(self._in_spin, 1)
        row.addWidget(self._in_unit)
        in_form.addRow("Value:", row)
        layout.addWidget(in_grp)

        # Output
        out_grp = QGroupBox("Converted Values")
        out_form = QFormLayout(out_grp)
        self._out_fields = {}
        _UNIT_TOOLTIPS = {
            "nm": "Wavelength in nanometres (10\u207b\u2079 m).\n"
                  "Commonly used for visible and UV light "
                  "(e.g. 532 nm = green Nd:YAG).",
            "cm\u207b\u00b9": "Wavenumber: reciprocal of wavelength in cm.\n"
                              "Standard unit in atomic/molecular spectroscopy.\n"
                              "\u03bd\u0303 = 1/\u03bb(cm).",
            "MHz": "Frequency in megahertz (10\u2076 Hz).\n"
                   "Used in CLS for isotope shift measurements\n"
                   "and hyperfine splittings.",
            "GHz": "Frequency in gigahertz (10\u2079 Hz).\n"
                   "Typical for microwave and fine-structure intervals.",
            "THz": "Frequency in terahertz (10\u00b9\u00b2 Hz).\n"
                   "Optical frequencies are hundreds of THz\n"
                   "(e.g. 532 nm \u2248 563 THz).",
            "eV": "Photon energy in electron-volts.\n"
                  "E = h\u00b7c/\u03bb.  Visible photons are ~1.5\u20133.3 eV.",
        }
        for unit in _UNIT_NAMES:
            le = QLineEdit()
            le.setReadOnly(True)
            le.setToolTip(_UNIT_TOOLTIPS.get(unit, ""))
            self._out_fields[unit] = le
            out_form.addRow(f"{unit}:", le)
        layout.addWidget(out_grp)

        self._in_spin.valueChanged.connect(self._update)
        self._in_unit.currentTextChanged.connect(self._update)
        self._update()

    def _update(self):
        val = self._in_spin.value()
        unit = self._in_unit.currentText()
        lam_m = _unit_to_wavelength_m(val, unit)
        results = _wavelength_m_to_all(lam_m)
        for u, le in self._out_fields.items():
            if u == unit:
                le.setText(f"{val:.6g}")
            else:
                le.setText(f"{results[u]:.6g}")


# ══════════════════════════════════════════════════════════════════
#  SHG Crystal Angle Calculator
# ══════════════════════════════════════════════════════════════════

import math as _math

# Sellmeier dispersion (lam in micrometres). Two functional forms are used:
#   "uv-ir-sub":  n^2 = A + B/(lam^2 - C) - D*lam^2          (BBO; Kato 1986)
#   "zernike":    n^2 = A + B_ir*lam^2/(lam^2 - P) + B_uv/(lam^2 - C_uv)
#                 (KDP; Zernike 1964 revised. The dominant IR-pole term
#                  B_ir*lam^2/(lam^2-P) is essential; folding KDP into the
#                  4-term "uv-ir-sub" form drops it and mis-predicts the cut
#                  angle by ~8 deg, so KDP gets its own 5-term form.)
# LBO is biaxial: Type-I phase matching in its XY plane cannot be represented
# by the negative-uniaxial model below, so it is flagged unsupported rather
# than silently returning a wrong/empty answer.
_SELLMEIER = {
    "BBO": {  # Kato 1986, Eimerl 1987
        "form": "uv-ir-sub",
        "no": (2.7359, 0.01878, 0.01822, 0.01354),
        "ne": (2.3753, 0.01224, 0.01667, 0.01516),
        "range": (190, 3500),  # nm transparency range
    },
    "KDP": {  # Zernike 1964 (revised), incl. IR pole at lam^2 = 400 um^2
        "form": "zernike",
        "no": (2.259276, 13.00522, 400.0, 0.01008956, 0.012942625),
        "ne": (2.132668, 3.2279924, 400.0, 0.008637494, 0.012281043),
        "range": (200, 1500),
    },
    "LBO": {  # Kato 1994 — biaxial; uniaxial Type-I model not applicable
        "form": "uv-ir-sub",
        "biaxial": True,
        "no": (2.4542, 0.01125, 0.01135, 0.01388),
        "ne": (2.5390, 0.01277, 0.01189, 0.01848),
        "range": (160, 2600),
    },
}


def _sellmeier_uv_ir_sub(coeffs, lam_um):
    """n^2 = A + B/(lam^2 - C) - D*lam^2 (lam in micrometres)."""
    A, B, C, D = coeffs
    n2 = A + B / (lam_um**2 - C) - D * lam_um**2
    return _math.sqrt(max(n2, 1.0))


def _sellmeier_zernike(coeffs, lam_um):
    """n^2 = A + B_ir*lam^2/(lam^2 - P) + B_uv/(lam^2 - C_uv) (lam in um)."""
    A, B_ir, P, B_uv, C_uv = coeffs
    l2 = lam_um**2
    n2 = A + B_ir * l2 / (l2 - P) + B_uv / (l2 - C_uv)
    return _math.sqrt(max(n2, 1.0))


_SELLMEIER_FORMS = {
    "uv-ir-sub": _sellmeier_uv_ir_sub,
    "zernike": _sellmeier_zernike,
}


def _crystal_index(crystal, which, lam_um):
    """Refractive index of ``crystal`` ('no'/'ne') at ``lam_um`` micrometres,
    dispatched to the crystal's Sellmeier form."""
    c = _SELLMEIER[crystal]
    return _SELLMEIER_FORMS[c.get("form", "uv-ir-sub")](c[which], lam_um)


def _crystal_supported(crystal):
    """Whether the uniaxial Type-I model below is valid for ``crystal``.
    Biaxial crystals (e.g. LBO) need an XY-plane biaxial treatment that this
    calculator does not implement, so they are reported as unsupported rather
    than producing a silently-wrong or empty result."""
    return not _SELLMEIER[crystal].get("biaxial", False)


def _shg_angle_type1(crystal, lam_fund_nm):
    """Phase-matching angle (degrees) for Type I SHG (o+o->e).

    Returns None if phase matching is not possible.
    """
    if not _crystal_supported(crystal):
        return None
    lam_f = lam_fund_nm / 1000.0   # um
    lam_s = lam_f / 2.0            # SHG wavelength

    no_f = _crystal_index(crystal, "no", lam_f)   # ordinary at fundamental
    no_s = _crystal_index(crystal, "no", lam_s)   # ordinary at SHG
    ne_s = _crystal_index(crystal, "ne", lam_s)   # extraordinary at SHG

    # Phase matching: ne(theta, lam_s) = no(lam_f)
    # 1/ne(theta)^2 = cos^2(theta)/no_s^2 + sin^2(theta)/ne_s^2
    # Solve for sin^2(theta):
    num = 1.0 / no_f**2 - 1.0 / no_s**2
    den = 1.0 / ne_s**2 - 1.0 / no_s**2
    if den == 0 or num / den < 0 or num / den > 1:
        return None
    sin2 = num / den
    return _math.degrees(_math.asin(_math.sqrt(sin2)))


def _shg_wavelength_from_angle_type1(crystal, angle_deg):
    """Find fundamental wavelength (nm) for a given Type I phase-matching angle.

    Sample the angle-vs-wavelength curve across the crystal's
    transparency range, find an adjacent pair of valid samples that
    brackets the requested angle, then bisect inside that bracket.
    Bracketing only over valid samples handles the case where one end
    of the range lies outside the phase-matching window (e.g. the BBO
    short-wavelength edge, where sin^2(theta) > 1 and the Type-I
    equation has no real solution).

    Returns None if the requested angle is not achievable anywhere in
    the crystal's transparency range.
    """
    lo, hi = _SELLMEIER[crystal]["range"]
    lo = max(lo * 2, 400)  # fundamental must be > 2x the SHG UV cutoff
    hi = min(hi, 3000)

    N = 400
    samples = []
    for i in range(N + 1):
        lam = lo + (hi - lo) * i / N
        a = _shg_angle_type1(crystal, lam)
        if a is not None:
            samples.append((lam, a))
    if len(samples) < 2:
        return None

    bracket = None
    for (l1, a1), (l2, a2) in zip(samples, samples[1:]):
        if (a1 - angle_deg) * (a2 - angle_deg) <= 0:
            bracket = (l1, a1, l2)
            break
    if bracket is None:
        return None

    l1, a1, l2 = bracket
    for _ in range(60):
        mid = 0.5 * (l1 + l2)
        am = _shg_angle_type1(crystal, mid)
        if am is None:
            return 0.5 * (l1 + l2)
        if abs(am - angle_deg) < 1e-4:
            return mid
        if (a1 - angle_deg) * (am - angle_deg) < 0:
            l2 = mid
        else:
            l1, a1 = mid, am
    return 0.5 * (l1 + l2)


class SHGCalculatorDialog(QDialog):
    """SHG crystal phase-matching angle calculator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SHG Crystal Angle Calculator")
        self.setWindowIcon(lucide_icon("flask-conical"))
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)

        # Crystal & type
        config_grp = QGroupBox("Configuration")
        cf = QFormLayout(config_grp)
        self._crystal = QComboBox()
        self._crystal.addItems(list(_SELLMEIER.keys()))
        self._crystal.setToolTip(
            "Nonlinear crystal material.\n\n"
            "BBO (\u03b2-Barium Borate): wide transparency 190\u20133500 nm,\n"
            "  high damage threshold. Most common for UV SHG.\n"
            "  Sellmeier: Kato 1986.\n\n"
            "KDP (Potassium Dihydrogen Phosphate): 200\u20131500 nm,\n"
            "  available in large apertures, lower damage threshold.\n"
            "  Sellmeier: Zernike & Midwinter 1973.\n\n"
            "LBO (Lithium Triborate): 160\u20132600 nm, high damage\n"
            "  threshold. Biaxial \u2014 XY-plane Type-I phase matching is\n"
            "  NOT supported by this uniaxial calculator. Sellmeier: Kato 1994.")
        cf.addRow("Crystal:", self._crystal)
        layout.addWidget(config_grp)

        # Direction
        dir_grp = QGroupBox("Calculation Direction")
        df = QFormLayout(dir_grp)
        self._direction = QComboBox()
        self._direction.addItems([
            "Wavelength \u2192 Angle",
            "Angle \u2192 Wavelength",
        ])
        self._direction.setToolTip(
            "Choose which quantity to calculate.\n\n"
            "Wavelength \u2192 Angle: given the laser wavelength,\n"
            "  find the crystal cut angle for phase matching.\n\n"
            "Angle \u2192 Wavelength: given a crystal cut angle,\n"
            "  find which fundamental wavelength is phase-matched.")
        self._direction.currentIndexChanged.connect(self._on_direction)
        df.addRow("Mode:", self._direction)
        layout.addWidget(dir_grp)

        # Input
        in_grp = QGroupBox("Input")
        in_form = QFormLayout(in_grp)
        self._wl_spin = QDoubleSpinBox()
        self._wl_spin.setRange(100, 5000)
        self._wl_spin.setDecimals(2)
        self._wl_spin.setValue(1064.0)
        self._wl_spin.setSuffix(" nm")
        self._wl_spin.setToolTip(
            "Fundamental laser wavelength before frequency doubling.\n"
            "The SHG output will be at half this wavelength.\n"
            "Example: 1064 nm Nd:YAG \u2192 532 nm green.")
        self._wl_label = QLabel("Fundamental wavelength:")
        in_form.addRow(self._wl_label, self._wl_spin)

        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(0.1, 89.9)
        self._angle_spin.setDecimals(3)
        self._angle_spin.setValue(22.9)
        self._angle_spin.setSuffix("\u00b0")
        self._angle_spin.setToolTip(
            "Crystal cut angle \u03b8 relative to the optic axis.\n"
            "The angle at which the phase-matching condition\n"
            "n_e(\u03b8, \u03bb_SHG) = n_o(\u03bb_fund) is satisfied.")
        self._angle_label = QLabel("Phase-matching angle:")
        in_form.addRow(self._angle_label, self._angle_spin)
        self._angle_label.setVisible(False)
        self._angle_spin.setVisible(False)
        layout.addWidget(in_grp)

        # Output
        out_grp = QGroupBox("Results")
        of = QFormLayout(out_grp)
        self._out_angle = QLineEdit()
        self._out_angle.setReadOnly(True)
        self._out_angle.setToolTip(
            "Phase-matching angle \u03b8_pm: the angle between the\n"
            "beam propagation direction and the crystal optic axis\n"
            "where \u0394k = 0 (momentum conservation).")
        self._out_angle_label = QLabel("Phase-matching angle:")
        of.addRow(self._out_angle_label, self._out_angle)
        self._out_fund = QLineEdit()
        self._out_fund.setReadOnly(True)
        self._out_fund.setToolTip(
            "Wavelength of the input (fundamental) laser beam.")
        self._out_fund_label = QLabel("Fundamental wavelength:")
        of.addRow(self._out_fund_label, self._out_fund)
        # Mode 1 (wl -> angle): hide the fundamental-wavelength echo
        # since it's just what the user typed in.
        self._out_fund_label.setVisible(False)
        self._out_fund.setVisible(False)
        self._out_shg = QLineEdit()
        self._out_shg.setReadOnly(True)
        self._out_shg.setToolTip(
            "Wavelength of the frequency-doubled output.\n"
            "\u03bb_SHG = \u03bb_fund / 2.")
        of.addRow("SHG wavelength:", self._out_shg)
        layout.addWidget(out_grp)

        # Connect signals
        self._wl_spin.valueChanged.connect(self._compute)
        self._angle_spin.valueChanged.connect(self._compute)
        self._crystal.currentTextChanged.connect(self._compute)
        self._compute()

    def _on_direction(self, idx):
        wl_mode = idx == 0
        # Inputs: show the spinbox for the chosen mode.
        self._wl_label.setVisible(wl_mode)
        self._wl_spin.setVisible(wl_mode)
        self._angle_label.setVisible(not wl_mode)
        self._angle_spin.setVisible(not wl_mode)
        # Results: hide the field that would just echo the input.
        self._out_angle_label.setVisible(wl_mode)
        self._out_angle.setVisible(wl_mode)
        self._out_fund_label.setVisible(not wl_mode)
        self._out_fund.setVisible(not wl_mode)
        self._compute()

    def _compute(self):
        crystal = self._crystal.currentText()
        if not _crystal_supported(crystal):
            # Biaxial crystal (e.g. LBO): the uniaxial Type-I model can't
            # represent XY-plane phase matching. Say so explicitly rather
            # than showing the generic "No phase matching" / "No solution".
            msg = "Not supported (biaxial crystal)"
            self._out_angle.setText(msg)
            self._out_fund.setText(msg)
            self._out_shg.setText("–")
            return
        if self._direction.currentIndex() == 0:
            # Wavelength -> Angle
            lam = self._wl_spin.value()
            angle = _shg_angle_type1(crystal, lam)
            self._out_fund.setText(f"{lam:.2f} nm")
            self._out_shg.setText(f"{lam / 2:.2f} nm")
            if angle is not None:
                self._out_angle.setText(f"{angle:.3f}\u00b0")
            else:
                self._out_angle.setText("No phase matching")
        else:
            # Angle -> Wavelength
            angle = self._angle_spin.value()
            lam = _shg_wavelength_from_angle_type1(crystal, angle)
            self._out_angle.setText(f"{angle:.3f}\u00b0")
            if lam is not None:
                self._out_fund.setText(f"{lam:.2f} nm")
                self._out_shg.setText(f"{lam / 2:.2f} nm")
            else:
                self._out_fund.setText("No solution")
                self._out_shg.setText("-")


# ══════════════════════════════════════════════════════════════════
#  SFG / DFG Calculator
# ══════════════════════════════════════════════════════════════════

class _HiResScientificSpinBox(ScientificSpinBox):
    """ScientificSpinBox that preserves enough significant figures to
    show 6+ decimal places for typical wavelength / wavenumber inputs.
    The base class formats with %.6g, which collapses to ~2 decimals
    once the integer part has 4 digits (e.g. 1064.123456 -> '1064.12')."""

    def textFromValue(self, value):
        if value == 0.0:
            return "0.0"
        return f"{value:.12g}"


class SFGDFGCalculatorDialog(QDialog):
    """Sum- / difference-frequency generation calculator.

    Given two input wavelengths (in any spectroscopic unit supported
    by the Unit Converter), computes the SFG output
        1/lambda_out = 1/lambda_1 + 1/lambda_2
    or the DFG output
        1/lambda_out = |1/lambda_1 - 1/lambda_2|
    and reports it in nm, cm^-1, MHz, GHz, THz, and eV.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SFG / DFG Calculator")
        self.setWindowIcon(lucide_icon("merge"))
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        # ── Inputs ──
        in_grp = QGroupBox("Inputs")
        in_form = QFormLayout(in_grp)

        def _make_row(default_value, default_unit):
            spin = _HiResScientificSpinBox()
            spin.setRange(1e-15, 1e15)
            spin.setValue(default_value)
            unit = QComboBox()
            unit.addItems(_UNIT_NAMES)
            unit.setCurrentText(default_unit)
            row = QHBoxLayout()
            row.addWidget(spin, 1)
            row.addWidget(unit)
            return spin, unit, row

        self._in1, self._unit1, row1 = _make_row(1064.0, "nm")
        in_form.addRow("Input 1:", row1)
        self._in2, self._unit2, row2 = _make_row(532.0, "nm")
        in_form.addRow("Input 2:", row2)
        layout.addWidget(in_grp)

        # ── Operation ──
        op_grp = QGroupBox("Operation")
        op_lay = QHBoxLayout(op_grp)
        self._op_sfg = QRadioButton("Sum (SFG)")
        self._op_sfg.setChecked(True)
        self._op_sfg.setToolTip(
            "Sum-frequency generation:\n"
            "  ω₃ = ω₁ + ω₂\n"
            "  1/λ₃ = 1/λ₁ + 1/λ₂\n"
            "(SHG is the special case λ₁ = λ₂.)")
        self._op_dfg = QRadioButton("Difference (DFG)")
        self._op_dfg.setToolTip(
            "Difference-frequency generation:\n"
            "  ω₃ = |ω₁ − ω₂|\n"
            "  1/λ₃ = |1/λ₁ − 1/λ₂|\n"
            "Order doesn't matter; magnitude is taken so the\n"
            "result wavelength is always positive.")
        op_btn_group = QButtonGroup(self)
        op_btn_group.addButton(self._op_sfg)
        op_btn_group.addButton(self._op_dfg)
        op_lay.addWidget(self._op_sfg)
        op_lay.addWidget(self._op_dfg)
        op_lay.addStretch()
        layout.addWidget(op_grp)

        # ── Output ──
        out_grp = QGroupBox("Generated wavelength")
        out_form = QFormLayout(out_grp)
        self._out_fields = {}
        for unit in _UNIT_NAMES:
            le = QLineEdit()
            le.setReadOnly(True)
            self._out_fields[unit] = le
            out_form.addRow(f"{unit}:", le)
        layout.addWidget(out_grp)

        # ── Wire up ──
        for w in (self._in1, self._in2):
            w.valueChanged.connect(self._compute)
        for w in (self._unit1, self._unit2):
            w.currentTextChanged.connect(self._compute)
        for r in (self._op_sfg, self._op_dfg):
            r.toggled.connect(self._compute)
        self._compute()

    def _compute(self):
        lam1_m = _unit_to_wavelength_m(
            self._in1.value(), self._unit1.currentText())
        lam2_m = _unit_to_wavelength_m(
            self._in2.value(), self._unit2.currentText())
        if lam1_m <= 0 or lam2_m <= 0:
            for w in self._out_fields.values():
                w.setText("-")
            return
        # Wavenumbers (cm^-1) are linearly additive (proportional to
        # photon energy). Adding/subtracting wavelengths directly is
        # physically meaningless.
        sigma1 = 1.0 / (lam1_m * 100.0)
        sigma2 = 1.0 / (lam2_m * 100.0)
        if self._op_sfg.isChecked():
            sigma_out = sigma1 + sigma2
        else:
            sigma_out = abs(sigma1 - sigma2)
        if sigma_out <= 0:
            for u, le in self._out_fields.items():
                le.setText("∞" if u == "nm" else "0")
            return
        lam_out_m = 1.0 / (sigma_out * 100.0)
        results = _wavelength_m_to_all(lam_out_m)
        for u, le in self._out_fields.items():
            le.setText(f"{results[u]:.6g}")


# ══════════════════════════════════════════════════════════════════
#  Quick Plot Dialog
# ══════════════════════════════════════════════════════════════════

import csv as _csv
import io as _io


class QuickPlotDialog(QDialog):
    """Quick Plot: load CSV/TSV or type data, pick columns, plot instantly."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Plot")
        self.setWindowIcon(lucide_icon("line-chart"))
        self.setMinimumSize(900, 620)

        self._headers = []
        self._data = []  # list of dicts

        root = QVBoxLayout(self)

        # ── Top: data input tabs ──
        self._input_tabs = QTabWidget()
        self._input_tabs.setMaximumHeight(260)
        root.addWidget(self._input_tabs)

        # Tab 1: File
        file_w = QWidget()
        fl = QVBoxLayout(file_w)
        fl.setContentsMargins(8, 8, 8, 8)
        file_row = QHBoxLayout()
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText(
            "Drop a .csv / .tsv file here or click Browse...")
        self._file_path.setReadOnly(True)
        file_row.addWidget(self._file_path, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(browse_btn)
        fl.addWidget(QLabel("Load CSV / TSV / TXT file:"))
        fl.addLayout(file_row)

        # Delimiter option
        delim_row = QHBoxLayout()
        delim_row.addWidget(QLabel("Delimiter:"))
        self._delim_combo = QComboBox()
        self._delim_combo.addItems(["Auto", "Comma (,)", "Tab (\\t)",
                                     "Semicolon (;)", "Space"])
        delim_row.addWidget(self._delim_combo)
        delim_row.addStretch()
        fl.addLayout(delim_row)
        fl.addStretch()
        self._input_tabs.addTab(file_w, "Load File")

        # Tab 2: Manual entry (spreadsheet)
        manual_w = QWidget()
        ml = QVBoxLayout(manual_w)
        ml.setContentsMargins(4, 4, 4, 4)
        btn_row = QHBoxLayout()
        for label, slot in [
            ("+ Row", self._add_row), ("+ Col", self._add_col),
            ("- Row", self._del_row), ("- Col", self._del_col),
            ("Clear", self._clear_table),
        ]:
            b = QPushButton(label)
            b.setFixedWidth(55)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        paste_btn = QPushButton("Paste from Clipboard")
        paste_btn.clicked.connect(self._paste_clipboard)
        btn_row.addWidget(paste_btn)
        ml.addLayout(btn_row)

        self._table = QTableWidget(10, 2)
        self._table.setHorizontalHeaderLabels(["X", "Y"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        ml.addWidget(self._table)
        self._input_tabs.addTab(manual_w, "Enter Data")

        # ── Plot controls row ──
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X:"))
        self._x_col = QComboBox()
        self._x_col.setMinimumWidth(200)
        self._x_col.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        ctrl.addWidget(self._x_col)
        ctrl.addWidget(QLabel("Y:"))
        self._y_col = QComboBox()
        self._y_col.setMinimumWidth(200)
        self._y_col.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        ctrl.addWidget(self._y_col)
        ctrl.addWidget(QLabel("Type:"))
        self._chart_type = QComboBox()
        self._chart_type.addItems(["Line", "Scatter", "Step", "Bar"])
        ctrl.addWidget(self._chart_type)

        ctrl.addWidget(QLabel("Color:"))
        self._color_btn = QToolButton()
        self._color_btn.setFixedSize(22, 22)
        self._plot_color = "#dc143c"
        self._color_btn.setStyleSheet(
            f"background-color: {self._plot_color}; border: 1px solid gray;")
        self._color_btn.clicked.connect(self._pick_plot_color)
        ctrl.addWidget(self._color_btn)

        ctrl.addStretch()
        save_btn = QPushButton("Save")
        save_btn.setToolTip(
            "Save the current data (headers + values) to a CSV file")
        save_btn.clicked.connect(self._save_data)
        ctrl.addWidget(save_btn)
        plot_btn = QPushButton("Plot")
        plot_btn.setStyleSheet("font-weight: bold;")
        plot_btn.clicked.connect(self._do_plot)
        ctrl.addWidget(plot_btn)
        root.addLayout(ctrl)

        # ── Settings row (collapsible) ──
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Title:"))
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("(none)")
        self._title_edit.setMaximumWidth(150)
        settings_row.addWidget(self._title_edit)
        settings_row.addWidget(QLabel("X label:"))
        self._xlabel_edit = QLineEdit()
        self._xlabel_edit.setPlaceholderText("auto")
        self._xlabel_edit.setMaximumWidth(100)
        settings_row.addWidget(self._xlabel_edit)
        settings_row.addWidget(QLabel("Y label:"))
        self._ylabel_edit = QLineEdit()
        self._ylabel_edit.setPlaceholderText("auto")
        self._ylabel_edit.setMaximumWidth(100)
        settings_row.addWidget(self._ylabel_edit)
        self._grid_check = QCheckBox("Grid")
        self._grid_check.setChecked(True)
        settings_row.addWidget(self._grid_check)
        self._legend_check = QCheckBox("Legend")
        self._legend_check.setChecked(True)
        settings_row.addWidget(self._legend_check)
        settings_row.addStretch()
        root.addLayout(settings_row)

        # ── Matplotlib canvas ──
        self._fig = Figure(figsize=(9, 4.5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas._plot_editor_opener = self._open_editor
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        root.addWidget(self._toolbar)
        root.addWidget(self._canvas, 1)

        # Auto-plot on column change
        self._x_col.currentIndexChanged.connect(self._do_plot)
        self._y_col.currentIndexChanged.connect(self._do_plot)

        # Show placeholder
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, "Load a file or enter data, then click Plot",
                ha='center', va='center', color='gray', fontsize=12,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self._canvas.draw()

    def _open_editor(self):
        # Right-click "Edit plot…" target (canvas._plot_editor_opener):
        # QuickPlot has no persistence, so a plain editor on the live
        # figure is all that is needed.
        dlg = getattr(self, "_editor_dialog", None)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        self._editor_dialog = PlotEditorDialog(
            self._fig, self._canvas, parent=self)
        self._editor_dialog.show()

    # ── File loading ──

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", get_last_dir("data", "load"),
            "Data files (*.csv *.tsv *.txt *.dat);;All files (*)")
        if not path:
            return
        remember_last_dir("data", "load", path)
        self._file_path.setText(path)
        self._load_file(path)

    def _get_delimiter(self):
        t = self._delim_combo.currentText()
        if "Comma" in t:
            return ','
        if "Tab" in t:
            return '\t'
        if "Semicolon" in t:
            return ';'
        if "Space" in t:
            return None  # split on whitespace
        return None  # Auto

    def _load_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot read file:\n{e}")
            return

        delim = self._get_delimiter()
        if delim is None:
            # Auto-detect from first few lines
            sample = '\n'.join(text.split('\n')[:5])
            for d in ['\t', ';', ',']:
                if d in sample:
                    delim = d
                    break

        if delim is not None:
            reader = _csv.reader(_io.StringIO(text), delimiter=delim)
        else:
            lines = text.strip().split('\n')
            reader = [line.split() for line in lines]

        rows = list(reader)
        if len(rows) < 2:
            QMessageBox.warning(self, "Warning", "File has too few rows.")
            return

        headers = [h.strip() for h in rows[0]]
        data = []
        for row in rows[1:]:
            if not any(c.strip() for c in row):
                continue
            d = {}
            for i, h in enumerate(headers):
                val = row[i].strip() if i < len(row) else ''
                try:
                    d[h] = float(val)
                except ValueError:
                    d[h] = val
            data.append(d)

        self._headers = headers
        self._data = data
        self._update_combos()
        self._do_plot()

    # ── Manual entry ──

    def _add_row(self):
        self._table.setRowCount(self._table.rowCount() + 5)

    def _add_col(self):
        c = self._table.columnCount()
        self._table.setColumnCount(c + 1)
        self._table.setHorizontalHeaderItem(
            c, QTableWidgetItem(f"C{c + 1}"))

    def _del_row(self):
        if self._table.rowCount() > 1:
            self._table.setRowCount(self._table.rowCount() - 1)

    def _del_col(self):
        if self._table.columnCount() > 1:
            self._table.setColumnCount(self._table.columnCount() - 1)

    def _clear_table(self):
        self._table.clearContents()
        self._table.setRowCount(10)
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["X", "Y"])

    def _paste_clipboard(self):
        """Paste tab/comma separated data from clipboard."""
        app = QApplication.instance()
        text = app.clipboard().text()
        if not text.strip():
            return
        # Detect delimiter
        first_line = text.strip().split('\n')[0]
        if '\t' in first_line:
            delim = '\t'
        elif ';' in first_line:
            delim = ';'
        elif ',' in first_line:
            delim = ','
        else:
            delim = None  # whitespace

        lines = text.strip().split('\n')
        if delim:
            rows = [line.split(delim) for line in lines]
        else:
            rows = [line.split() for line in lines]

        if not rows:
            return

        # Check if first row is headers (non-numeric)
        first_numeric = all(
            self._is_number(c.strip()) for c in rows[0] if c.strip())
        if not first_numeric and len(rows) > 1:
            headers = [c.strip() for c in rows[0]]
            data_rows = rows[1:]
        else:
            headers = [f"C{i+1}" for i in range(len(rows[0]))]
            data_rows = rows

        ncols = len(headers)
        self._table.setColumnCount(ncols)
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(data_rows) + 3)
        for r, row in enumerate(data_rows):
            for c, val in enumerate(row):
                if c < ncols:
                    self._table.setItem(r, c, QTableWidgetItem(val.strip()))

        # Also load into internal data for plotting
        self._read_table_data()
        self._update_combos()
        self._do_plot()

    @staticmethod
    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _read_table_data(self):
        """Read data from the manual table into _headers/_data."""
        ncols = self._table.columnCount()
        headers = []
        for c in range(ncols):
            h = self._table.horizontalHeaderItem(c)
            headers.append(h.text() if h else f"C{c+1}")
        self._headers = headers

        data = []
        for r in range(self._table.rowCount()):
            d = {}
            has_val = False
            for c in range(ncols):
                item = self._table.item(r, c)
                val = item.text().strip() if item else ''
                if val:
                    has_val = True
                    try:
                        d[headers[c]] = float(val)
                    except ValueError:
                        d[headers[c]] = val
                else:
                    d[headers[c]] = ''
            if has_val:
                data.append(d)
        self._data = data

    # ── Combos ──

    def _update_combos(self):
        self._x_col.blockSignals(True)
        self._y_col.blockSignals(True)
        self._x_col.clear()
        self._y_col.clear()
        for h in self._headers:
            self._x_col.addItem(h)
            self._y_col.addItem(h)
        if len(self._headers) > 1:
            self._y_col.setCurrentIndex(1)
        self._x_col.blockSignals(False)
        self._y_col.blockSignals(False)

    # ── Color ──

    def _pick_plot_color(self):
        c = QColorDialog.getColor(
            QColor(self._plot_color), self, "Plot color")
        if c.isValid():
            self._plot_color = c.name()
            self._color_btn.setStyleSheet(
                f"background-color: {self._plot_color}; "
                f"border: 1px solid gray;")

    # ── Save ──

    def _save_data(self):
        """Save current data (headers + values) to a CSV file."""
        # Make sure manual-tab data is up-to-date
        if self._input_tabs.currentIndex() == 1:
            self._read_table_data()

        if not self._data or not self._headers:
            QMessageBox.information(
                self, "Save", "No data to save. Load a file or "
                "enter data first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", get_last_dir("data", "save"),
            "CSV files (*.csv);;TSV files (*.tsv);;All files (*)")
        if not path:
            return
        remember_last_dir("data", "save", path)

        delim = '\t' if path.lower().endswith('.tsv') else ','
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = _csv.writer(f, delimiter=delim)
                writer.writerow(self._headers)
                for row in self._data:
                    writer.writerow(
                        [row.get(h, '') for h in self._headers])
            QMessageBox.information(
                self, "Saved",
                f"Data saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Cannot save file:\n{e}")

    # ── Plot ──

    def _do_plot(self):
        # If on manual tab, re-read table data
        if self._input_tabs.currentIndex() == 1:
            self._read_table_data()

        if not self._data or not self._headers:
            return

        x_col = self._x_col.currentText()
        y_col = self._y_col.currentText()
        if not x_col or not y_col:
            return

        xs, ys = [], []
        for row in self._data:
            xv = row.get(x_col, '')
            yv = row.get(y_col, '')
            try:
                xs.append(float(xv))
                ys.append(float(yv))
            except (ValueError, TypeError):
                continue

        if not xs:
            return

        xs = np.array(xs)
        ys = np.array(ys)

        self._fig.clear()
        ax = self._fig.add_subplot(111)

        chart = self._chart_type.currentText()
        color = self._plot_color
        label = y_col + " vs " + x_col

        if chart == "Line":
            ax.plot(xs, ys, '-o', color=color, markersize=4,
                    linewidth=1.5, label=label)
        elif chart == "Scatter":
            ax.scatter(xs, ys, color=color, s=20, label=label)
        elif chart == "Step":
            ax.step(xs, ys, where='mid', color=color,
                    linewidth=1.5, label=label)
        elif chart == "Bar":
            span = xs.max() - xs.min()
            bar_w = max(0.5, span / len(xs) * 0.7) if len(xs) > 1 else 1.0
            ax.bar(xs, ys, color=color, alpha=0.7,
                   width=bar_w, label=label)

        title = self._title_edit.text().strip()
        if title:
            ax.set_title(title)
        ax.set_xlabel(self._xlabel_edit.text().strip() or x_col)
        ax.set_ylabel(self._ylabel_edit.text().strip() or y_col)
        if self._grid_check.isChecked():
            ax.grid(True, alpha=0.3)
        if self._legend_check.isChecked():
            ax.legend(fontsize=8)
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════
#  Status bar manager
# ══════════════════════════════════════════════════════════════════

from PySide6.QtWidgets import QProgressBar
from PySide6.QtCore import QTimer
from datetime import datetime as _dt


class StatusBarManager:
    """Manages the main window status bar with timestamps and auto-clear."""

    def __init__(self, status_bar, timeout_ms=8000):
        self._bar = status_bar
        self._timeout = timeout_ms
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setMaximumHeight(16)
        self._progress.setVisible(False)
        self._bar.addPermanentWidget(self._progress)

    def show(self, message, timeout_ms=None):
        """Show a status message with timestamp. Auto-clears after timeout."""
        ts = _dt.now().strftime("%H:%M:%S")
        self._bar.showMessage(f"[{ts}]  {message}")
        self._timer.start(timeout_ms or self._timeout)

    def show_persistent(self, message):
        """Show a message that stays until replaced."""
        ts = _dt.now().strftime("%H:%M:%S")
        self._bar.showMessage(f"[{ts}]  {message}")
        self._timer.stop()

    def show_progress(self, current, total, message=""):
        """Show progress bar + message."""
        self._progress.setVisible(True)
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(current)
        if message:
            self.show_persistent(message)
        if current >= total:
            QTimer.singleShot(3000, self.hide_progress)

    def hide_progress(self):
        """Hide the progress bar (public - call on cancel/error too)."""
        self._progress.setVisible(False)

    def flash(self, message):
        """Pulse the status bar to draw attention to a finished action
        until ``stop_flash`` is called (e.g. analysis done -> nudge the
        user toward the Results tab without stealing focus). The message
        stays put; only the background colour pulses.
        """
        self._flash_msg = message
        ts = _dt.now().strftime("%H:%M:%S")
        self._bar.showMessage(f"[{ts}]  {message}")
        self._timer.stop()  # don't let the auto-clear wipe the flash
        if not hasattr(self, "_flash_timer"):
            self._flash_timer = QTimer()
            self._flash_timer.timeout.connect(self._flash_tick)
        self._flash_on = False
        self._flash_timer.start(600)
        self._flash_tick()

    def _flash_tick(self):
        self._flash_on = not getattr(self, "_flash_on", False)
        if self._flash_on:
            self._bar.setStyleSheet(
                "QStatusBar { background-color: #b8860b; color: white; "
                "font-weight: bold; }")
        else:
            self._bar.setStyleSheet("")

    def stop_flash(self):
        """Stop the attention pulse and restore the normal status bar.
        No-op when no flash is active so it can't clobber an unrelated
        status message.
        """
        ft = getattr(self, "_flash_timer", None)
        if ft is None or not ft.isActive():
            return
        ft.stop()
        self._bar.setStyleSheet("")
        self._bar.showMessage("Ready")

    def _on_timeout(self):
        self._bar.showMessage("Ready")
