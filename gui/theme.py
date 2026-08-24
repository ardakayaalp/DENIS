"""One place for the app's look: palette, stylesheet, and input guards.

Date:    2026-07-24
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Before this module the theme was spread over three layers that could not
see each other: a Fusion palette + a three-class stylesheet (QMenu /
QDialog / QPushButton) inline in ``main_window.main()``, an opt-in
``dialog_style.DIALOG_QSS`` that only four modules applied, and ~70
per-widget ``setStyleSheet`` calls. Every surface those layers missed
(tool dialogs, Settings, tables, trees, scrollbars, tooltips, nested
tabs, spin boxes...) fell back to bare Fusion, which is why pop-outs and
context menus never quite matched the tabs.

``apply_theme(app)`` is now the single entry point: Fusion + dark
palette + one application-wide stylesheet covering every widget class
the app uses, plus the :class:`WheelFocusGuard` that stops hover-scroll
from editing spin boxes / combos / sliders (they must be clicked into
focus first — wheel over an unfocused field scrolls the page instead).

Matplotlib figures stay **white** on purpose: they are publication
output (see ``shared_widgets._STATIC_PLOT_STYLE``); only the Qt chrome
around them is dark.

Two themes ship: ``"dark"`` (default) and ``"win98"`` — "Classic 98
(dark)": Windows-98 structure with dark colors (charcoal chrome, two-
tone 3D bevels, sharp corners, navy selection, buttermilk tooltips,
bitmap Fixedsys type). ``apply_theme(app, name)`` switches between
them; the Settings dialog persists the choice under the ``ui_theme``
key and re-applies it live. Widget-level accent sheets (isotope
stripes, the main tab bar via :func:`main_tabs_qss`) follow the theme.

Depends on: PySide6 only.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, Qt, QObject
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDial, QSlider, QTabBar, QWidget,
)

# ── Design tokens ────────────────────────────────────────────────────────
# Interactive accent (links, selection, focus rings, checked marks) —
# the palette Highlight/Link blue the app has always used.
ACCENT = "#42a5f5"
# Muted accent for panel/group titles (dialog_style's historical blue).
ACCENT_TITLE = "#6cb6ff"
# Selection background for item views / tables (calmer than pure ACCENT
# so selected rows stay readable).
SELECT_BG = "#2f5f8f"

_TOKENS = {
    "accent": ACCENT,
    "accent_title": ACCENT_TITLE,
    "select_bg": SELECT_BG,
    "window": "#2d2d30",       # QPalette.Window
    "field": "#26262a",        # line edits / spin boxes / combos
    "field_disabled": "#2b2b2d",
    "raised": "#3a3a3f",       # buttons
    "raised_hover": "#46464c",
    "raised_pressed": "#2a2a2e",
    "sunken": "#232326",       # tables / trees / lists background
    "sunken_alt": "#2a2a2f",   # alternate rows
    "border": "#4d4d54",       # field borders
    "border_soft": "#4a4a4f",  # group/table borders (dialog_style value)
    "border_hover": "#62626a",
    "text": "#e0e0e0",
    "text_dim": "#9a9aa0",
    "text_disabled": "#777777",
    "group_bg": "#333338",     # group-box card background
    "header_bg": "#3a3a3f",    # table headers
    "tooltip_bg": "#2a2a2e",
    "tooltip_border": "#5a5a60",
}


def _icon_url(name: str) -> str:
    """Absolute forward-slash url() for a lucide icon, quoted for QSS."""
    icons = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "icons", "lucide", f"{name}.svg")
    return '"' + icons.replace("\\", "/") + '"'


# ── Theme registry ───────────────────────────────────────────────────────
# name → label shown in Settings. "dark" is the default; "win98" is the
# classic-98-but-dark retro look (the key predates the dark rework and
# stays stable because it is persisted in settings.yaml).
THEMES = {
    "dark": "Dark (default)",
    "win98": "Classic 98 (dark)",
}

_ACTIVE_THEME = "dark"


def active_theme() -> str:
    """Name of the theme most recently applied via :func:`apply_theme`."""
    return _ACTIVE_THEME


def build_palette() -> QPalette:
    """The Fusion dark palette (moved verbatim from ``main_window.main``)."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    p.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base, QColor(30, 30, 32))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(42, 42, 46))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(224, 224, 224))
    p.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 64, 64))
    p.setColor(QPalette.ColorRole.Link, QColor(66, 165, 245))
    p.setColor(QPalette.ColorRole.Highlight, QColor(66, 165, 245))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role,
                   QColor(127, 127, 127))
    return p


def build_qss() -> str:
    """The application-wide stylesheet.

    Notes on scope:
    - The main tab bar keeps its own rolodex sheet set directly on the
      ``#MainTabs`` widget (widget-level sheets win over this one).
    - Semantic per-widget styles (Mode-cell green/red, color swatches,
      the reference-isotope border, log fonts) also win — they are set
      on the widgets themselves.
    - matplotlib canvases are untouched (white publication plots).
    """
    t = dict(_TOKENS)
    t["check_icon"] = _icon_url("check")
    t["minus_icon"] = _icon_url("minus")
    t["chevron_up"] = _icon_url("chevron-up")
    t["chevron_down"] = _icon_url("chevron-down")
    return """
/* ── Menus ─────────────────────────────────────────────────────────── */
QMenu {
    background-color: %(window)s;
    border: 1px solid #555;
    padding: 4px;
}
QMenu::item {
    padding: 5px 24px 5px 20px;
    color: #dddddd;
    border-radius: 2px;
}
QMenu::item:selected {
    background-color: #4a4a4a;
    color: #ffffff;
    font-weight: bold;
}
QMenu::item:disabled { color: %(text_disabled)s; }
QMenu::separator {
    height: 1px;
    background-color: #444;
    margin: 4px 8px;
}
QMenuBar { background: transparent; color: %(text)s; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: %(raised_hover)s; border-radius: 3px; }

/* ── Dialog surfaces ───────────────────────────────────────────────── */
QDialog { background-color: %(window)s; }

/* ── Tooltips ──────────────────────────────────────────────────────── */
QToolTip {
    background-color: %(tooltip_bg)s;
    color: %(text)s;
    border: 1px solid %(tooltip_border)s;
    padding: 5px 8px;
}

/* ── Buttons ───────────────────────────────────────────────────────── */
QPushButton {
    background-color: %(raised)s;
    border: 1px solid #5a5a60;
    padding: 4px 12px;
    border-radius: 3px;
    color: %(text)s;
}
QPushButton:hover {
    background-color: %(raised_hover)s;
    border-color: #6e6e76;
}
QPushButton:pressed { background-color: %(raised_pressed)s; }
QPushButton:focus { border: 1px solid %(accent)s; }
QPushButton:default { border: 1px solid %(accent)s; }
QPushButton:checked {
    background-color: %(select_bg)s;
    border-color: %(accent)s;
}
QPushButton:disabled {
    color: %(text_disabled)s;
    background-color: #333336;
    border-color: #44444a;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 2px;
    color: %(text)s;
}
QToolButton:hover {
    background: %(raised_hover)s;
    border-color: #5a5a60;
}
QToolButton:pressed { background: %(raised_pressed)s; }
QToolButton:checked {
    background: %(select_bg)s;
    border-color: %(accent)s;
}
QToolButton:disabled { color: #666; }

/* ── Text fields ───────────────────────────────────────────────────── */
QLineEdit, QPlainTextEdit, QTextEdit {
    background-color: %(field)s;
    border: 1px solid %(border)s;
    border-radius: 3px;
    padding: 2px 5px;
    color: %(text)s;
    selection-background-color: %(accent)s;
    selection-color: #000000;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover {
    border-color: %(border_hover)s;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid %(accent)s;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {
    color: %(text_disabled)s;
    background-color: %(field_disabled)s;
    border-color: #44444a;
}
QLineEdit[readOnly="true"] { background-color: %(sunken)s; }

/* ── Spin boxes ────────────────────────────────────────────────────── */
QAbstractSpinBox {
    background-color: %(field)s;
    border: 1px solid %(border)s;
    border-radius: 3px;
    padding: 1px 4px;
    color: %(text)s;
    selection-background-color: %(accent)s;
    selection-color: #000000;
}
QAbstractSpinBox:hover { border-color: %(border_hover)s; }
QAbstractSpinBox:focus { border: 1px solid %(accent)s; }
QAbstractSpinBox:disabled {
    color: %(text_disabled)s;
    background-color: %(field_disabled)s;
    border-color: #44444a;
}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    subcontrol-origin: border;
    width: 16px;
    background: #33333a;
    border-left: 1px solid %(border)s;
}
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover,
QAbstractSpinBox::down-button:hover { background: %(raised_hover)s; }
QAbstractSpinBox::up-button:pressed,
QAbstractSpinBox::down-button:pressed { background: %(raised_pressed)s; }
QAbstractSpinBox::up-arrow {
    image: url(%(chevron_up)s);
    width: 10px; height: 10px;
}
QAbstractSpinBox::down-arrow {
    image: url(%(chevron_down)s);
    width: 10px; height: 10px;
}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off,
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {
    image: none;
}

/* ── Combo boxes ───────────────────────────────────────────────────── */
QComboBox {
    background-color: %(field)s;
    border: 1px solid %(border)s;
    border-radius: 3px;
    padding: 2px 6px;
    color: %(text)s;
    min-height: 18px;
}
QComboBox:hover { border-color: %(border_hover)s; }
QComboBox:focus { border: 1px solid %(accent)s; }
QComboBox:disabled {
    color: %(text_disabled)s;
    background-color: %(field_disabled)s;
    border-color: #44444a;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border-left: 1px solid %(border)s;
    background: #33333a;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}
QComboBox::down-arrow {
    image: url(%(chevron_down)s);
    width: 11px; height: 11px;
}
QComboBox QAbstractItemView {
    background-color: %(tooltip_bg)s;
    color: %(text)s;
    border: 1px solid %(tooltip_border)s;
    selection-background-color: %(select_bg)s;
    selection-color: #ffffff;
    outline: 0;
}

/* ── Check boxes / radio buttons / item-view checks ────────────────── */
QCheckBox, QRadioButton { color: %(text)s; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: %(text_disabled)s; }
QCheckBox::indicator, QGroupBox::indicator,
QTreeView::indicator, QTableView::indicator, QListView::indicator {
    width: 15px; height: 15px;
    border: 1px solid #62626a;
    border-radius: 3px;
    background: %(field)s;
}
QCheckBox::indicator:hover, QGroupBox::indicator:hover,
QTreeView::indicator:hover, QTableView::indicator:hover,
QListView::indicator:hover { border-color: #8a8a92; }
QCheckBox::indicator:checked, QGroupBox::indicator:checked,
QTreeView::indicator:checked, QTableView::indicator:checked,
QListView::indicator:checked {
    background-color: %(accent)s;
    border-color: %(accent)s;
    image: url(%(check_icon)s);
}
QCheckBox::indicator:indeterminate, QGroupBox::indicator:indeterminate,
QTreeView::indicator:indeterminate, QTableView::indicator:indeterminate,
QListView::indicator:indeterminate {
    background-color: %(select_bg)s;
    border-color: %(select_bg)s;
    image: url(%(minus_icon)s);
}
QCheckBox::indicator:disabled, QGroupBox::indicator:disabled,
QTreeView::indicator:disabled, QTableView::indicator:disabled,
QListView::indicator:disabled {
    background: %(field_disabled)s;
    border-color: #44444a;
}
QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1px solid #62626a;
    border-radius: 8px;
    background: %(field)s;
}
QRadioButton::indicator:hover { border-color: #8a8a92; }
QRadioButton::indicator:checked {
    border: 1px solid %(accent)s;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.55,
                                fx:0.5, fy:0.5,
                                stop:0 %(accent)s, stop:0.55 %(accent)s,
                                stop:0.7 %(field)s, stop:1 %(field)s);
}
QRadioButton::indicator:disabled {
    background: %(field_disabled)s;
    border-color: #44444a;
}

/* ── Group boxes (dialog_style card language, now app-wide) ────────── */
QGroupBox {
    border: 1px solid %(border_soft)s;
    border-radius: 4px;
    margin-top: 10px;
    padding: 7px 6px 6px 6px;
    background-color: %(group_bg)s;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 9px;
    padding: 0 5px;
    color: %(accent_title)s;
    font-weight: bold;
}

/* ── Tabs (nested; #MainTabs keeps its own widget-level sheet) ─────── */
QTabWidget::pane {
    border: 1px solid %(border_soft)s;
    border-radius: 3px;
    top: -1px;
}
QTabBar::tab {
    background: #35353a;
    color: #cccccc;
    padding: 5px 14px;
    border: 1px solid %(border_soft)s;
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}
QTabBar::tab:selected {
    background: #45454c;
    color: #ffffff;
    font-weight: bold;
}
QTabBar::tab:!selected:hover { background: #3d3d44; color: #e6e6e6; }

/* ── Tables / headers ──────────────────────────────────────────────── */
QTableWidget, QTableView {
    background-color: %(sunken)s;
    alternate-background-color: %(sunken_alt)s;
    gridline-color: #3a3a3f;
    selection-background-color: %(select_bg)s;
    selection-color: #ffffff;
    border: 1px solid %(border_soft)s;
    border-radius: 3px;
}
QHeaderView::section {
    background-color: %(header_bg)s;
    color: %(text)s;
    padding: 4px 6px;
    border: none;
    border-right: 1px solid %(window)s;
    border-bottom: 1px solid %(window)s;
    font-weight: bold;
}
QTableCornerButton::section {
    background-color: %(header_bg)s;
    border: none;
}

/* ── Trees / lists ─────────────────────────────────────────────────── */
QTreeView, QTreeWidget, QListView, QListWidget {
    background-color: %(sunken)s;
    alternate-background-color: %(sunken_alt)s;
    border: 1px solid %(border_soft)s;
    border-radius: 3px;
    color: %(text)s;
}
QTreeView::item, QListView::item { padding: 1px 2px; }
QTreeView::item:hover, QListView::item:hover { background: #33333a; }
QTreeView::item:selected, QListView::item:selected {
    background: %(select_bg)s;
    color: #ffffff;
}

/* ── Scrollbars ────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #242428;
    width: 12px;
    margin: 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: #4a4a52;
    min-height: 28px;
    border-radius: 5px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover { background: #5f5f68; }
QScrollBar:horizontal {
    background: #242428;
    height: 12px;
    margin: 0;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #4a4a52;
    min-width: 28px;
    border-radius: 5px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: #5f5f68; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ── Sliders ───────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #3a3a40;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: %(accent)s;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #d0d0d5;
    border: 1px solid #666;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
QSlider::groove:vertical {
    width: 4px;
    background: #3a3a40;
    border-radius: 2px;
}
QSlider::handle:vertical {
    height: 14px;
    margin: 0 -5px;
    border-radius: 7px;
    background: #d0d0d5;
    border: 1px solid #666;
}
QSlider::handle:vertical:hover { background: #ffffff; }

/* ── Progress bars ─────────────────────────────────────────────────── */
QProgressBar {
    background-color: %(field)s;
    border: 1px solid %(border)s;
    border-radius: 3px;
    text-align: center;
    color: %(text)s;
}
QProgressBar::chunk {
    background-color: %(select_bg)s;
    border-radius: 2px;
}

/* ── Structure chrome ──────────────────────────────────────────────── */
QScrollArea { border: none; }
QSplitter::handle { background: #2c2c30; }
QSplitter::handle:hover { background: #3a5a7a; }
QStatusBar { border-top: 1px solid #3f3f45; }
QStatusBar::item { border: none; }

/* ── Plot cards (dialog_style language, now global) ────────────────── */
QFrame#plotCard {
    background-color: %(window)s;
    border: 1px solid %(border_soft)s;
    border-radius: 4px;
}
QLabel#plotCardTitle { color: %(accent_title)s; font-weight: bold; }
QLabel#sectionNote { color: %(text_dim)s; }
""" % t


# ── "Classic 98 (dark)" theme ────────────────────────────────────────────
# Windows-98 STRUCTURE, dark COLORS (per Arda: "classic dark look like
# 98 look but dark, sharp corners, pixelated fonts"). Charcoal chrome
# with explicit two-tone 3D bevels (light top-left / near-black
# bottom-right, inverted for sunken fields), navy #000080 selection,
# the iconic buttermilk #ffffe1 tooltip, no rounded corner anywhere,
# and a bitmap-font stack (Fixedsys → Terminal → Lucida Console) for
# the pixelated type. Fidelity is "inspired", not pixel-perfect.

def build_win98_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(46, 46, 46))
    p.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base, QColor(28, 28, 28))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(38, 38, 38))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 225))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    p.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button, QColor(58, 58, 58))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 85, 85))
    p.setColor(QPalette.ColorRole.Link, QColor(122, 180, 255))
    p.setColor(QPalette.ColorRole.Highlight, QColor(0, 0, 128))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role,
                   QColor(122, 122, 122))
    return p


def build_win98_qss() -> str:
    t = {
        "window": "#2e2e2e",       # dialog / window surfaces
        "face": "#3a3a3a",         # raised control face
        "face_hover": "#444444",
        "face_pressed": "#303030",
        "field": "#1c1c1c",        # sunken input wells
        "text": "#dcdcdc",
        "navy": "#000080",         # the classic selection navy
        "lt": "#6e6e6e",           # bevel light (top/left of raised)
        "dk": "#0e0e0e",           # bevel dark (bottom/right of raised)
        "disabled": "#7a7a7a",
        "check_icon": _icon_url("check"),
        "minus_icon": _icon_url("minus"),
        "chevron_up": _icon_url("chevron-up"),
        "chevron_down": _icon_url("chevron-down"),
    }
    return """
/* Pixel type note: the retro font is NOT set here. Qt 6's DirectWrite
   backend cannot load the classic raster fonts (Fixedsys/Terminal are
   .fon files — CreateFontFaceFromHDC fails), so apply_theme() instead
   sets an app-wide scalable monospace with the NoAntialias style
   strategy, which renders genuinely pixelated. */

/* ── Menus ─────────────────────────────────────────────────────────── */
QMenu {
    background-color: %(face)s;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
    padding: 2px;
    color: %(text)s;
}
QMenu::item { padding: 4px 24px 4px 20px; color: %(text)s; }
QMenu::item:selected { background-color: %(navy)s; color: #ffffff; }
QMenu::item:disabled { color: %(disabled)s; }
QMenu::separator {
    height: 2px;
    background-color: transparent;
    border-top: 1px solid %(dk)s;
    border-bottom: 1px solid %(lt)s;
    margin: 3px 2px;
}
QMenuBar { background: %(window)s; color: %(text)s; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: %(navy)s; color: #ffffff; }

/* ── Dialog surfaces ───────────────────────────────────────────────── */
QDialog { background-color: %(window)s; }

/* ── Tooltips (the iconic buttermilk survives the dark look) ───────── */
QToolTip {
    background-color: #ffffe1;
    color: #000000;
    border: 1px solid #000000;
    padding: 4px 6px;
}

/* ── Buttons: raised 3D bevel, sharp corners ───────────────────────── */
QPushButton {
    background-color: %(face)s;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
    padding: 3px 12px;
    color: %(text)s;
}
QPushButton:hover { background-color: %(face_hover)s; }
QPushButton:pressed, QPushButton:checked {
    background-color: %(face_pressed)s;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
}
QPushButton:default { border-top-color: #8a8a8a; border-left-color: #8a8a8a; }
QPushButton:disabled { color: %(disabled)s; }
QToolButton {
    background: %(face)s;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
    padding: 1px;
    color: %(text)s;
}
QToolButton:hover { background: %(face_hover)s; }
QToolButton:pressed, QToolButton:checked {
    background: %(face_pressed)s;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
}
QToolButton:disabled { color: %(disabled)s; }

/* ── Text fields: sunken dark wells ────────────────────────────────── */
QLineEdit, QPlainTextEdit, QTextEdit {
    background-color: %(field)s;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    padding: 2px 4px;
    color: %(text)s;
    selection-background-color: %(navy)s;
    selection-color: #ffffff;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {
    color: %(disabled)s;
    background-color: %(window)s;
}
QLineEdit[readOnly="true"] { background-color: #242424; }

/* ── Spin boxes ────────────────────────────────────────────────────── */
QAbstractSpinBox {
    background-color: %(field)s;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    padding: 1px 3px;
    color: %(text)s;
    selection-background-color: %(navy)s;
    selection-color: #ffffff;
}
QAbstractSpinBox:disabled {
    color: %(disabled)s;
    background-color: %(window)s;
}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    subcontrol-origin: border;
    width: 16px;
    background: %(face)s;
    border: 1px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
}
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:pressed,
QAbstractSpinBox::down-button:pressed {
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
}
QAbstractSpinBox::up-arrow {
    image: url(%(chevron_up)s);
    width: 9px; height: 9px;
}
QAbstractSpinBox::down-arrow {
    image: url(%(chevron_down)s);
    width: 9px; height: 9px;
}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off,
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {
    image: none;
}

/* ── Combo boxes ───────────────────────────────────────────────────── */
QComboBox {
    background-color: %(field)s;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    padding: 2px 5px;
    color: %(text)s;
    min-height: 18px;
    selection-background-color: %(navy)s;
    selection-color: #ffffff;
}
QComboBox:disabled { color: %(disabled)s; background-color: %(window)s; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 18px;
    background: %(face)s;
    border: 1px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
}
QComboBox::down-arrow {
    image: url(%(chevron_down)s);
    width: 10px; height: 10px;
}
QComboBox QAbstractItemView {
    background-color: %(field)s;
    color: %(text)s;
    border: 1px solid %(lt)s;
    selection-background-color: %(navy)s;
    selection-color: #ffffff;
    outline: 0;
}

/* ── Check boxes / radio buttons / item-view checks ────────────────── */
QCheckBox, QRadioButton { color: %(text)s; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: %(disabled)s; }
QCheckBox::indicator, QGroupBox::indicator,
QTreeView::indicator, QTableView::indicator, QListView::indicator {
    width: 13px; height: 13px;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    background: %(field)s;
}
QCheckBox::indicator:checked, QGroupBox::indicator:checked,
QTreeView::indicator:checked, QTableView::indicator:checked,
QListView::indicator:checked { image: url(%(check_icon)s); }
QCheckBox::indicator:indeterminate, QGroupBox::indicator:indeterminate,
QTreeView::indicator:indeterminate, QTableView::indicator:indeterminate,
QListView::indicator:indeterminate { image: url(%(minus_icon)s); }
QCheckBox::indicator:disabled, QGroupBox::indicator:disabled,
QTreeView::indicator:disabled, QTableView::indicator:disabled,
QListView::indicator:disabled { background: %(window)s; }
QRadioButton::indicator {
    width: 12px; height: 12px;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    border-radius: 8px;
    background: %(field)s;
}
QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                                fx:0.5, fy:0.5,
                                stop:0 %(text)s, stop:0.45 %(text)s,
                                stop:0.6 %(field)s, stop:1 %(field)s);
}
QRadioButton::indicator:disabled { background: %(window)s; }

/* ── Group boxes (etched frame, flat dark card) ────────────────────── */
QGroupBox {
    border: 2px groove #565656;
    margin-top: 10px;
    padding: 7px 6px 6px 6px;
    background-color: %(window)s;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 9px;
    padding: 0 4px;
    color: %(text)s;
    background-color: %(window)s;
}

/* ── Tabs ──────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
    top: -1px;
    background: %(window)s;
}
QTabBar::tab {
    background: %(face)s;
    color: %(text)s;
    padding: 4px 12px;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s;
    border-bottom: none;
}
QTabBar::tab:selected { background: #474747; }
QTabBar::tab:!selected { margin-top: 2px; }

/* ── Tables / headers ──────────────────────────────────────────────── */
QTableWidget, QTableView {
    background-color: %(field)s;
    alternate-background-color: #262626;
    gridline-color: %(face)s;
    selection-background-color: %(navy)s;
    selection-color: #ffffff;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    color: %(text)s;
}
QHeaderView::section {
    background-color: %(face)s;
    color: %(text)s;
    padding: 3px 6px;
    border: 1px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
}
QTableCornerButton::section {
    background-color: %(face)s;
    border: 1px solid %(dk)s;
}

/* ── Trees / lists ─────────────────────────────────────────────────── */
QTreeView, QTreeWidget, QListView, QListWidget {
    background-color: %(field)s;
    alternate-background-color: #262626;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    color: %(text)s;
}
QTreeView::item, QListView::item { padding: 1px 2px; }
QTreeView::item:selected, QListView::item:selected {
    background: %(navy)s;
    color: #ffffff;
}

/* ── Scrollbars: chunky beveled handles, square everything ─────────── */
QScrollBar:vertical {
    background: #252525;
    width: 16px;
    margin: 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: %(face)s;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
    min-height: 24px;
}
QScrollBar:horizontal {
    background: #252525;
    height: 16px;
    margin: 0;
    border: none;
}
QScrollBar::handle:horizontal {
    background: %(face)s;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
    min-width: 24px;
}
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ── Sliders ───────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: %(field)s;
    border: 1px solid %(dk)s;
}
QSlider::handle:horizontal {
    width: 11px;
    margin: -6px 0;
    background: %(face)s;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
}
QSlider::groove:vertical {
    width: 4px;
    background: %(field)s;
    border: 1px solid %(dk)s;
}
QSlider::handle:vertical {
    height: 11px;
    margin: 0 -6px;
    background: %(face)s;
    border: 2px solid;
    border-top-color: %(lt)s; border-left-color: %(lt)s;
    border-right-color: %(dk)s; border-bottom-color: %(dk)s;
}

/* ── Progress bars ─────────────────────────────────────────────────── */
QProgressBar {
    background-color: %(field)s;
    border: 2px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
    text-align: center;
    color: %(text)s;
}
QProgressBar::chunk { background-color: %(navy)s; }

/* ── Structure chrome ──────────────────────────────────────────────── */
QScrollArea { border: none; }
QSplitter::handle { background: %(window)s; }
QSplitter::handle:hover { background: %(face_hover)s; }
QStatusBar {
    background: %(window)s;
    color: %(text)s;
    border-top: 1px solid %(lt)s;
}
QStatusBar::item {
    border: 1px solid;
    border-top-color: %(dk)s; border-left-color: %(dk)s;
    border-right-color: %(lt)s; border-bottom-color: %(lt)s;
}

/* ── Plot cards ────────────────────────────────────────────────────── */
QFrame#plotCard {
    background-color: %(window)s;
    border: 2px groove #565656;
}
QLabel#plotCardTitle { color: %(text)s; font-weight: bold; }
QLabel#sectionNote { color: #9a9a9a; }
""" % t


def main_tabs_qss(theme: str | None = None) -> str:
    """Widget-level sheet for the top-level ``#MainTabs`` tab bar.

    Kept at widget level (wins over the app sheet) so nested tab
    widgets keep the plain theme look. The dark variant is the
    historical rolodex sheet from ``MainWindow.__init__``; the win98
    variant renders classic raised folder tabs.
    """
    if theme is None:
        theme = _ACTIVE_THEME
    if theme == "win98":
        return (
            # Classic-98-but-dark: square raised folder tabs with the
            # two-tone bevel; the selected tab pops lighter.
            "QTabWidget#MainTabs::pane {"
            "  border-top: 1px solid #6e6e6e;"
            "  background: transparent;"
            "  top: -1px;"
            "}"
            "QTabWidget#MainTabs > QTabBar { background: transparent; }"
            "QTabWidget#MainTabs > QTabBar::tab {"
            "  padding: 4px 18px;"
            "  margin: 3px 0 0 0;"
            "  color: #dcdcdc;"
            "  background: #3a3a3a;"
            "  border: 2px solid;"
            "  border-top-color: #6e6e6e;"
            "  border-left-color: #6e6e6e;"
            "  border-right-color: #0e0e0e;"
            "  border-bottom: none;"
            "}"
            "QTabWidget#MainTabs > QTabBar::tab:selected {"
            "  background: #474747;"
            "  margin-top: 0px;"
            "  margin-bottom: -1px;"
            "}"
        )
    return (
        # Rolodex / folder-tab feel: tabs touch each other with no
        # gaps; unselected tabs sit a few pixels lower with a darker
        # background so they visually recede behind the active tab;
        # the active tab fuses into the pane below via a negative
        # bottom margin.
        "QTabWidget#MainTabs::pane {"
        "  border-top: 1px solid #404040;"
        "  background: transparent;"
        "  top: -1px;"
        "}"
        "QTabWidget#MainTabs > QTabBar {"
        "  background: transparent;"
        "}"
        "QTabWidget#MainTabs > QTabBar::tab {"
        "  padding: 4px 18px;"
        "  margin: 3px 0 0 0;"
        "  color: #6e6e6e;"
        "  background: #232323;"
        "  border: 1px solid #303030;"
        "  border-bottom: none;"
        "  border-top-left-radius: 4px;"
        "  border-top-right-radius: 4px;"
        "}"
        "QTabWidget#MainTabs > QTabBar::tab:!selected:hover {"
        "  color: #b0b0b0;"
        "  background: #2c2c2c;"
        "  margin-top: 2px;"
        "}"
        "QTabWidget#MainTabs > QTabBar::tab:selected {"
        "  color: #ffffff;"
        "  background: #353535;"
        "  border: 1px solid #404040;"
        "  border-bottom: 2px solid #4287c2;"
        "  margin-top: 0px;"
        "  margin-bottom: -1px;"
        "}"
    )


# ── Wheel/focus guard ────────────────────────────────────────────────────
# Widget types whose value must not change from a hover-scroll. QScrollBar
# (also a QAbstractSlider) is deliberately NOT here — scrollbars exist to
# be wheeled. QTabBar is here so scrolling past a tab strip can't switch
# tabs; it keeps its NoFocus policy, so its wheel is effectively always
# blocked.
_GUARDED_TYPES = (QAbstractSpinBox, QComboBox, QSlider, QDial, QTabBar)

_GUARD_MARK = "_denisWheelGuarded"


class _WheelBlocker(QObject):
    """Per-widget filter: swallow wheel unless the widget has focus.

    Installed on the widget itself (not the app) so it runs inside Qt's
    wheel-propagation loop: returning True with the event ignored stops
    the *widget* from reacting but lets the loop continue to the parent
    chain, so an enclosing scroll area still scrolls the page.
    """

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Wheel
                and isinstance(obj, QWidget)
                and not obj.hasFocus()):
            event.ignore()
            return True
        return False


class WheelFocusGuard(QObject):
    """App-level filter that arms every spin box / combo / slider / tab bar.

    On Polish (fires once per widget when its style is first applied) it
    downgrades WheelFocus to StrongFocus — so scrolling can never *give*
    the widget focus — and installs the shared :class:`_WheelBlocker`.
    Editing therefore requires a deliberate click (or Tab) first, which
    is exactly the "field must be clicked before the wheel or keyboard
    changes it" rule. Once focused, wheel behaves normally.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blocker = _WheelBlocker(self)

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Polish
                and isinstance(obj, _GUARDED_TYPES)
                and not obj.property(_GUARD_MARK)):
            obj.setProperty(_GUARD_MARK, True)
            obj.installEventFilter(self._blocker)
            if (not isinstance(obj, QTabBar)
                    and obj.focusPolicy() == Qt.FocusPolicy.WheelFocus):
                obj.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        return False


def style_project_tab_bar(tabs) -> None:
    """Project tab bars (PA / Analysis): subtle per-name color dot +
    visually centered labels.

    - Each tab gets a small dot whose hue derives stably from the tab
      NAME (same project → same color across sessions) — subtle but
      instantly distinguishable. Done with icons because the app-wide
      ``QTabBar::tab { color: ... }`` rule overrides per-tab text
      colors.
    - The close button sits inside the tab's right edge, which makes
      symmetric padding read as left-shifted text; the icon on the left
      plus slightly tightened padding re-balances it.

    Idempotent — call again after adding or renaming tabs.
    """
    import hashlib
    from PySide6.QtGui import QIcon, QPainter, QPixmap
    from PySide6.QtWidgets import QHBoxLayout
    bar = tabs.tabBar()
    if bar.objectName() != "projectTabBar":
        bar.setObjectName("projectTabBar")
        # Scoped to this bar only (nested tab widgets keep the theme).
        tabs.setStyleSheet(
            "QTabBar#projectTabBar::tab { padding: 5px 6px; }")
    for i in range(tabs.count()):
        name = tabs.tabText(i).replace("&", "")
        if not name:
            continue
        hue = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8],
                  16) % 360
        pm = QPixmap(10, 10)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor.fromHsl(hue, 120, 150))
        p.drawEllipse(0, 0, 9, 9)
        p.end()
        tabs.setTabIcon(i, QIcon(pm))

        # The native close button hugs the tab's right border while the
        # color dot sits inset — wrap it in a container with a right
        # margin so both ends of the tab breathe equally. (The button's
        # position is laid out by QTabBar itself, so a wrapper is the
        # reliable way to indent it.)
        btn = bar.tabButton(i, QTabBar.ButtonPosition.RightSide)
        if btn is not None and btn.objectName() != "closeWrap":
            wrap = QWidget(bar)
            wrap.setObjectName("closeWrap")
            wl = QHBoxLayout(wrap)
            wl.setContentsMargins(0, 0, 5, 0)
            wl.setSpacing(0)
            wl.addWidget(btn)
            bar.setTabButton(i, QTabBar.ButtonPosition.RightSide, wrap)
            # setTabButton HIDES the widget it replaces — which is the
            # close button now living inside the wrapper. Un-hide it,
            # or every project tab shows an empty container instead of
            # its ✕.
            btn.show()


def install_wheel_guard(app) -> WheelFocusGuard:
    """Install (once) and return the app-wide wheel/focus guard."""
    existing = getattr(app, "_denis_wheel_guard", None)
    if existing is not None:
        return existing
    guard = WheelFocusGuard(app)
    app.installEventFilter(guard)
    app._denis_wheel_guard = guard  # keep a strong ref on the app
    return guard


def retro_font() -> QFont:
    """The Classic-98 pixel font: a scalable monospace drawn WITHOUT
    antialiasing. Qt 6's DirectWrite backend cannot load the real
    bitmap fonts (Fixedsys/Terminal are raster .fon files and fail
    with CreateFontFaceFromHDC), but NoAntialias on a crisp monospace
    gives the same jagged-pixel rendering safely."""
    f = QFont("Lucida Console", 9)
    f.setStyleStrategy(QFont.StyleStrategy.NoAntialias)
    return f


def apply_theme(app, theme: str = "dark") -> None:
    """Fusion + the named theme's palette, stylesheet, font + wheel
    guard.

    ``theme`` is a key of :data:`THEMES`; unknown names fall back to
    dark. Callable again at runtime to switch themes live (the Settings
    dialog does exactly that). The platform default font is captured on
    first call so leaving the retro theme restores it. (Caveat: the
    zoom feature stamps fonts directly onto existing widgets; widgets
    zoomed under one theme keep that family until the next zoom step
    re-stamps them.)
    """
    global _ACTIVE_THEME
    if theme not in THEMES:
        theme = "dark"
    if not hasattr(app, "_denis_default_font"):
        app._denis_default_font = QFont(app.font())
    app.setStyle("Fusion")
    if theme == "win98":
        app.setPalette(build_win98_palette())
        app.setStyleSheet(build_win98_qss())
        app.setFont(retro_font())
    else:
        app.setPalette(build_palette())
        app.setStyleSheet(build_qss())
        app.setFont(QFont(app._denis_default_font))
    _ACTIVE_THEME = theme
    install_wheel_guard(app)
