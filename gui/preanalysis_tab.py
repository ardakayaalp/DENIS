"""Pre-Analysis tab: live data viewer with HFS model overlay.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds the Pre-Analysis tab of the DENIS app: loads CLS event ASDF files,
bins them into a live spectrum across several x-axis domains (voltage,
calibrated voltage, beam energy, wavenumber, frequency), and overlays
satlas2 hyperfine-structure (HFS) models for quick visual estimation.
Also provides TOF / timestamp gating, per-scan filtering, calibration and
cooler-voltage diagnostic panels, file merging, and project save/load.

Depends on: cls_estimations.constants, cls_estimations.doppler,
gui.shared_widgets, gui.analysis.vasdf, gui.analysis.binning (and, at use
time, gui.analysis.merge / gui.split_editor); built on PySide6, matplotlib
and satlas2.
"""

import logging
import os
import numpy as np
import yaml
import asdf

# Child of the "denis" session logger (configured in gui.session_log).
_log = logging.getLogger("denis.preanalysis")


# Older Pre-Analysis YAMLs were written with yaml.dump's default representer,
# which emits `!!python/tuple` for the (run_number, path) pairs in merged
# entries' source_info. yaml.safe_load rejects that tag and the load aborts
# at the first merged entry. Register a SafeLoader constructor that treats
# the tag as a plain list so old configs still load. New saves use plain
# lists (see _merge_checked).
def _construct_python_tuple_as_list(loader, node):
    return loader.construct_sequence(node)


yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple_as_list)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.colors import to_hex
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# satlas2 is imported lazily where used (HFS overlay / preview) so app startup
# does not pay its (lmfit/scipy/emcee) import cost.

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QPushButton, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QColorDialog, QSizePolicy,
    QSlider, QToolButton, QGridLayout, QTabWidget, QDialog,
    QDialogButtonBox, QApplication,
)
from PySide6.QtCore import Qt, Signal, QTimer, QLocale, QEvent
from PySide6.QtGui import QColor, QPixmap, QIcon

from cls_estimations.constants import C_LIGHT, AMU_TO_KG, E_CHARGE
from cls_estimations.doppler import beta_from_voltage, nu_seen_by_ion
from gui.shared_widgets import (
    maybe_convert_path, lucide_icon, _load_settings, _save_settings,
    AppUndoCommand,
)
from gui.analysis.vasdf import is_vasdf_path, read_vasdf
from gui.analysis.binning import (
    BIN_DEFINITIONS,
    YERR_MODES,
    DEFAULT_BIN_COUNT,
    DEFAULT_BIN_WIDTH_MHZ,
    compute_binned,
)

#: Pre-Analysis defaults its frequency binning to per-scan-step rather than
#: clstools' Auto (equal-width) grid. Auto sits at the aliasing point for
#: Doppler-nonlinear axes -- one bin per step, but uniform width -- so two
#: adjacent steps can pile into one bin and show a spurious doubled-count
#: spike. Per-step bins one DAC step at a time, which cannot alias. The
#: Analysis/fit SourceBlock keeps "Auto" as its default (see BIN_DEFINITIONS).
DEFAULT_PA_BIN_DEFINITION = "Per scan step"

# clstools drags in pandas + scipy (~2 s of app startup); import it on
# first data load, not at module import (this module loads at startup).
_CLSTOOLS = None


def _get_clstools():
    """Import clstools on first use. Returns the module, or None when
    it is not installed (sentinel False caches the failed probe)."""
    global _CLSTOOLS
    if _CLSTOOLS is None:
        try:
            import clstools
            _CLSTOOLS = clstools
        except ImportError:
            _CLSTOOLS = False
    return _CLSTOOLS or None

try:
    import periodictable
    _HAS_PERIODICTABLE = True
except ImportError:
    _HAS_PERIODICTABLE = False


# ── Default colors for loaded files ──────────────────────────────
# Windows 98 system palette (classic 16-color VGA), keeping the dark
# halves that stay readable on the white plot canvas and ordering for
# maximum contrast between consecutively loaded runs. The bright trio
# (lime/cyan/yellow) is too faint on white and is left out.
DEFAULT_COLORS = [
    "#000080",  # navy
    "#800000",  # maroon
    "#008000",  # green
    "#800080",  # purple
    "#008080",  # teal
    "#FF0000",  # red
    "#0000FF",  # blue
    "#808000",  # olive
    "#FF00FF",  # fuchsia
    "#808080",  # gray
]

# Dark-mode line palette: high-contrast neon that pops on a black
# canvas (the Win98 dark halves above vanish on black). Index-aligned
# with DEFAULT_COLORS so a run keeps a stable slot when toggling.
NEON_COLORS = [
    "#00E5FF",  # cyan
    "#FF2D95",  # hot pink
    "#7CFF3F",  # lime
    "#FFE500",  # yellow
    "#FF9A2E",  # orange
    "#22B0FF",  # electric blue
    "#00FFA3",  # spring green
    "#C86BFF",  # violet
    "#FF5C5C",  # coral red
    "#F0F0F0",  # near-white
]

# Dark-mode HFS model curve palette (bright, distinct from the most
# common line colors above so a fit stands out over the data).
NEON_MODEL_COLORS = [
    "#FFE500",  # yellow
    "#00E5FF",  # cyan
    "#FF2D95",  # hot pink
    "#7CFF3F",  # lime
]

# ── X-axis mode options ──────────────────────────────────────────
XAXIS_MODES = [
    "Voltage",
    "Calibrated voltage",
    "Calibrated beam energy",
    "Wavenumber",
    "Frequency",
]
# When Bin mode == Frequency, only display modes with a clean inverse
# from a single-bin-centre frequency are allowed.
XAXIS_MODES_FREQ_BIN = ["Wavenumber", "Frequency"]


def _color_icon(color_hex, size=16):
    """Create a small square icon filled with the given color."""
    pm = QPixmap(size, size)
    pm.fill(QColor(color_hex))
    return QIcon(pm)


_MAX_CUSTOM_COLORS = 12


def _load_custom_colors():
    """User-saved custom line colours (shared across files & models),
    persisted in the global settings file."""
    try:
        vals = _load_settings().get("pa_custom_colors", []) or []
        out = []
        for v in vals:
            c = QColor(str(v))
            if c.isValid():
                out.append(c.name())
        return out[:_MAX_CUSTOM_COLORS]
    except Exception:
        return []


def _save_custom_colors(colors):
    try:
        s = _load_settings()
        s["pa_custom_colors"] = list(colors)[:_MAX_CUSTOM_COLORS]
        _save_settings(s)
    except Exception:
        pass


class LineColorDialog(QDialog):
    """Pick a plot line's LIGHT-mode and DARK-mode colour together.

    Each mode has a swatch preview, a hex / R,G,B text box, the preset
    palette (Win98 for light, neon for dark), the user's saved custom
    colours, a "＋" to save the current colour to a custom slot, and a
    full custom picker. The active colour follows the plot theme so a
    run's list swatch always matches what's drawn.
    """

    def __init__(self, light, dark, parent=None, title="Line colours"):
        super().__init__(parent)
        self.setWindowTitle(title)
        try:
            from gui.dialog_style import style_dialog
            style_dialog(self)
        except Exception:
            pass
        self._light = light
        self._dark = dark
        self._customs = _load_custom_colors()
        self._custom_strips = []   # (layout, is_dark) to rebuild on save

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 10)
        outer.setSpacing(8)

        outer.addWidget(self._build_row(
            "Light mode", DEFAULT_COLORS, is_dark=False))
        outer.addWidget(self._build_row(
            "Dark mode", NEON_COLORS, is_dark=True))

        from PySide6.QtWidgets import QDialogButtonBox
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _cur(self, is_dark):
        return self._dark if is_dark else self._light

    def _build_row(self, label, palette, is_dark):
        box = QGroupBox(label)
        row = QVBoxLayout(box)
        row.setSpacing(4)

        # Preview swatch + hex/RGB entry + custom picker.
        head = QHBoxLayout()
        swatch = QLabel()
        swatch.setFixedSize(30, 20)

        def _paint():
            swatch.setStyleSheet(
                f"background:{self._cur(is_dark)}; border:1px solid #888;")
        _paint()
        head.addWidget(swatch)

        hex_edit = QLineEdit(self._cur(is_dark))
        hex_edit.setFixedWidth(110)
        hex_edit.setToolTip("Enter a hex colour (#RRGGBB) or R,G,B values")

        def _set(color, sync_edit=True):
            c = QColor(color)
            if not c.isValid():
                return
            name = c.name()
            if is_dark:
                self._dark = name
            else:
                self._light = name
            _paint()
            if sync_edit:
                hex_edit.blockSignals(True)
                hex_edit.setText(name)
                hex_edit.blockSignals(False)

        def _from_text():
            txt = hex_edit.text().strip()
            c = QColor(txt)
            if not c.isValid() and "," in txt:
                try:
                    r, g, b = (int(p) for p in txt.split(",")[:3])
                    c = QColor(r, g, b)
                except Exception:
                    c = QColor()
            if c.isValid():
                _set(c.name(), sync_edit=False)
        hex_edit.editingFinished.connect(_from_text)
        head.addWidget(hex_edit)

        custom = QPushButton("Custom…")

        def _pick_custom():
            c = QColorDialog.getColor(QColor(self._cur(is_dark)), self,
                                      "Pick colour")
            if c.isValid():
                _set(c.name())
        custom.clicked.connect(_pick_custom)
        head.addWidget(custom)

        save_btn = QToolButton()
        save_btn.setText("＋")
        save_btn.setToolTip("Save this colour to a custom slot")

        def _save_current():
            cur = self._cur(is_dark)
            if cur not in self._customs:
                self._customs.insert(0, cur)
                self._customs = self._customs[:_MAX_CUSTOM_COLORS]
                _save_custom_colors(self._customs)
                self._rebuild_custom_strips()
        save_btn.clicked.connect(_save_current)
        head.addWidget(save_btn)
        head.addStretch()
        row.addLayout(head)

        # Preset palette.
        pal = QHBoxLayout()
        pal.setSpacing(3)
        pal.addWidget(QLabel("Presets:"))
        for hexc in palette:
            b = QToolButton()
            b.setFixedSize(20, 20)
            b.setToolTip(hexc)
            b.setStyleSheet(
                f"QToolButton {{ background:{hexc}; border:1px solid #666; }}")
            b.clicked.connect(lambda _=False, h=hexc: _set(h))
            pal.addWidget(b)
        pal.addStretch()
        row.addLayout(pal)

        # Custom saved colours + empty slots.
        cust = QHBoxLayout()
        cust.setSpacing(3)
        cust.addWidget(QLabel("Custom:"))
        self._custom_strips.append((cust, is_dark, _set))
        row.addLayout(cust)
        self._fill_custom_strip(cust, is_dark, _set)
        return box

    def _fill_custom_strip(self, layout, is_dark, setter):
        # Clear existing slot buttons (keep the leading "Custom:" label).
        while layout.count() > 1:
            item = layout.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for hexc in self._customs:
            b = QToolButton()
            b.setFixedSize(20, 20)
            b.setToolTip(hexc)
            b.setStyleSheet(
                f"QToolButton {{ background:{hexc}; border:1px solid #666; }}")
            b.clicked.connect(lambda _=False, h=hexc: setter(h))
            layout.addWidget(b)
        # A few empty slots so the row reads as "space to save more".
        for _ in range(max(0, 4 - len(self._customs))):
            e = QToolButton()
            e.setFixedSize(20, 20)
            e.setEnabled(False)
            e.setStyleSheet(
                "QToolButton { border:1px dashed #777; background:transparent; }")
            layout.addWidget(e)
        layout.addStretch()

    def _rebuild_custom_strips(self):
        for layout, is_dark, setter in self._custom_strips:
            self._fill_custom_strip(layout, is_dark, setter)

    def colors(self):
        """Return the chosen ``(light, dark)`` colours."""
        return self._light, self._dark


class _NoScrollDouble(QDoubleSpinBox):
    """Ignores wheel unless focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class _NoScrollInt(QSpinBox):
    """Ignores wheel unless focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


def _make_double(value=0.0, min_val=-1e12, max_val=1e12, decimals=6,
                 step=1.0, suffix="", tooltip=""):
    sb = _NoScrollDouble()
    sb.setLocale(QLocale(QLocale.Language.English,
                         QLocale.Country.UnitedStates))
    sb.setRange(min_val, max_val)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(value)
    sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def _make_int(value=0, min_val=0, max_val=999, tooltip=""):
    sb = _NoScrollInt()
    sb.setRange(min_val, max_val)
    sb.setValue(value)
    sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def _stabilize_toolbar(toolbar, label_width=180):
    # locLabel has Expanding sizePolicy and a Fixed-policy toolbar derives its
    # actual size from sizeHint, so the toolbar grew/shrank on every mouse move
    # and shifted the buttons in the header. Pinning the label width keeps the
    # toolbar's footprint constant; moving its action to the front yields the
    # order [coords][buttons] with the buttons right-anchored.
    label = getattr(toolbar, "locLabel", None)
    if label is None:
        return
    label.setFixedWidth(label_width)
    label.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    actions = toolbar.actions()
    if len(actions) < 2:
        return
    loc_action = actions[-1]  # matplotlib appends the locLabel last
    toolbar.removeAction(loc_action)
    toolbar.insertAction(actions[0], loc_action)


# ══════════════════════════════════════════════════════════════════
#  Centroid offset dialog (themed replacement for QInputDialog)
# ══════════════════════════════════════════════════════════════════

class CentroidOffsetDialog(QDialog):
    """Small themed dialog for entering a per-file manual centroid
    offset in MHz.

    Behaves like ``QInputDialog.getDouble`` semantically -- title,
    info label, single spinbox, OK/Cancel -- but lays out as a
    regular QDialog so the app's dark stylesheet (panel + buttons +
    spinbox) applies. The popped-up QInputDialog renders a paler
    panel that doesn't match the rest of the UI.
    """

    def __init__(self, run_number, current_value, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Centroid offset")
        self.setModal(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        info = QLabel(
            f"Manual centroid offset for run {run_number}\n"
            "(MHz, subtracted from the file's frequency axis before "
            "merging; additive with any GP-based correction):")
        info.setWordWrap(True)
        outer.addWidget(info)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(-1e9, 1e9)
        self._spin.setDecimals(3)
        self._spin.setSingleStep(0.5)
        self._spin.setValue(float(current_value))
        self._spin.setMinimumWidth(180)
        # Letting the user type freely is preferable to forcing them
        # through up/down arrows for a value that can span six orders
        # of magnitude. Selecting all text on focus makes the typical
        # "replace existing offset" flow take one keystroke.
        self._spin.selectAll()
        row.addWidget(QLabel("Offset:"))
        row.addWidget(self._spin, 1)
        row.addWidget(QLabel("MHz"))
        outer.addLayout(row)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

        # Focus the input + select-all so Enter immediately commits
        # the typed value without further navigation.
        self._spin.setFocus()

    def value(self) -> float:
        return float(self._spin.value())


# ══════════════════════════════════════════════════════════════════
#  File entry widget (one per loaded ASDF file)
# ══════════════════════════════════════════════════════════════════
class FileEntry(QWidget):
    """A single loaded data file with checkbox, metadata, and color picker."""
    toggled = Signal()
    color_changed = Signal()
    removed = Signal(object)
    clicked = Signal(object)
    # Re-read the underlying ASDF from disk. Useful for ongoing
    # scans that grow while the user has the file open. Hidden on
    # ``MergedFileEntry`` (its data is in-memory, not on disk).
    reload_requested = Signal(object)

    _STYLE_NORMAL = ""
    _STYLE_SELECTED = (
        "FileEntry, FileEntry * "
        "{ background-color: #3a3d42; border-radius: 3px; }"
    )

    def __init__(self, filepath, run_number, cooler_v, date, laser_sp,
                 mass_amu, color="#000080", dark_color=None, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.run_number = run_number
        self.cooler_v = cooler_v
        self.date = date
        self.laser_sp = laser_sp
        self.mass_amu = mass_amu
        # Two colours: one for the light plot theme, one for dark. The
        # active one follows the enclosing tab's mode (set via
        # set_dark_active) so the list swatch always matches the plot.
        self._color = color
        self._dark_color = dark_color or color
        self._dark_active = False
        self._alpha = 1.0
        self._linestyle = "-"
        self._selected = False
        # Manual per-file centroid offset (MHz). Subtracted from the
        # file's frequency axis before binning during a frequency-domain
        # merge, additive with any GP-based correction the IS-tab
        # supplies for the same file. Default 0.0 means "no manual
        # offset". Persists in the PA YAML via _build_config_dict /
        # _restore_from_dict.
        self.centroid_offset_mhz = 0.0

        # Cached numpy arrays (populated after load)
        self.cls_data = None   # clstools.CLSDataFrame (kept for compute_binned)
        self.np_v = None       # calibrated voltage per event
        self.np_dv = None      # raw DAC voltage per event (for grouping)
        self.np_bunch = None   # bunch index per event (drives scan derivation)
        # Per-file scan metadata (ScanningRanges / StepSize /
        # BunchesPerChannel). Populated at load time so the timestamp
        # plot's scan overlay and the right-click filter dialog don't
        # need to re-open the ASDF for cheap header reads.
        self.scan_meta = None  # dict | None
        self.np_tof = None     # TOF per event (µs)
        self.np_tdc = None     # channel per event
        self.np_ts = None      # timestamp per event
        self.np_cal_set = None       # calibration set voltages
        self.np_cal_readback = None  # calibration readback voltages
        self.np_vcool = None         # per-event cooler voltage (V)
        self.run_time_s = 0.0  # run duration in seconds

        outer = QHBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        self.check = QCheckBox()
        self.check.setChecked(True)
        self.check.toggled.connect(lambda _: self.toggled.emit())
        outer.addWidget(self.check)

        # Color button
        self.color_btn = QToolButton()
        self.color_btn.setFixedSize(20, 20)
        self._update_color_icon()
        self.color_btn.clicked.connect(self._pick_color)
        outer.addWidget(self.color_btn)

        # Reload button -- re-reads the ASDF for ongoing scans that
        # grow on disk while open. Hidden on MergedFileEntry below.
        self.reload_btn = QToolButton()
        self.reload_btn.setFixedSize(20, 20)
        self.reload_btn.setIcon(lucide_icon("refresh-cw"))
        self.reload_btn.setToolTip(
            "Reload this file from disk. Useful for ongoing scans "
            "that are still being written -- new events show up "
            "after the reload.")
        self.reload_btn.clicked.connect(
            lambda: self.reload_requested.emit(self))
        outer.addWidget(self.reload_btn)

        # Card layout: two lines
        card = QVBoxLayout()
        card.setContentsMargins(0, 0, 0, 0)
        card.setSpacing(0)

        # Line 1: run name
        self._name_label = QLabel(f"<b>run_{run_number}</b>")
        card.addWidget(self._name_label)

        # Line 2: voltage, laser, run time (updated after data load)
        date_str = str(date)[:16] if date else "?"
        self._detail_label = QLabel(
            f"V={cooler_v:.1f}  |  "
            f"\u03bb={laser_sp:.4f} cm\u207b\u00b9  |  "
            f"{date_str}")
        self._detail_label.setStyleSheet("color: gray; font-size: 11px;")
        card.addWidget(self._detail_label)

        outer.addLayout(card, 1)

        # Blinking "!" when this run's calibration outliers would actually
        # move its centroid. Hidden unless there is something to say; clicking
        # it explains the cost in MHz and stops the blinking for good. The
        # path is resolved lazily: a MergedFileEntry has not marked itself as
        # merged yet, and a SplitFileEntry does not know its parent ASDF yet.
        from gui.calibration_alert import CalibrationAlertBadge
        self.cal_alert = CalibrationAlertBadge(
            path_fn=lambda: (None if getattr(self, "_is_merged", False)
                             else self._cal_path()),
            run_label=f"run_{run_number}",
            physics_fn=self._cal_physics,
            peers_fn=lambda: (self._pa_tab()._cal_peers()
                              if self._pa_tab() is not None else []),
            parent=self)
        outer.addWidget(self.cal_alert)

        # code review 2026-06-02, file-mass-tooltip-misleading-zero: show
        # "(not in file)" when the ASDF had no MassAMU (mass_amu is None)
        # rather than a misleading "0.0000 amu" that looks like a value.
        mass_str = (f"{mass_amu:.4f} amu" if mass_amu is not None
                    else "(not in file)")
        self.setToolTip(
            f"File: {filepath}\n"
            f"Run: {run_number}\n"
            f"Cooler: {cooler_v:.2f} V\n"
            f"Laser: {laser_sp:.6f} cm\u207b\u00b9\n"
            f"Mass: {mass_str}\n"
            f"Date: {date}")

    def contextMenuEvent(self, event):
        """Right-click context menu for file entries.

        Source-file entries get "Set centroid offset…" and "Filter
        scans…" actions; merged entries instead get View / Edit /
        Export actions for the merged spectrum. Remove and Check/Uncheck
        are always present.
        """
        from PySide6.QtWidgets import QMenu, QInputDialog
        menu = QMenu(self)
        is_merged = getattr(self, "_is_merged", False)
        if is_merged:
            # Merged spectra expose View / Edit / Export of the merged
            # result instead of the source-file actions below.
            view_action = menu.addAction("View…")
            view_action.triggered.connect(
                lambda: self.view_requested.emit(self))
            edit_action = menu.addAction("Edit…")
            edit_action.triggered.connect(
                lambda: self.edit_requested.emit(self))
            export_action = menu.addAction("Export ASDF…")
            export_action.triggered.connect(
                lambda: self.export_requested.emit(self))
            menu.addSeparator()
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(lambda: self.removed.emit(self))
        toggle_action = menu.addAction(
            "Uncheck" if self.check.isChecked() else "Check")
        toggle_action.triggered.connect(
            lambda: self.check.setChecked(not self.check.isChecked()))
        # The manual centroid offset is a *source-file* property -- a
        # merged spectrum's x-axis was fixed at merge time and there's
        # no consumer for an offset on a MergedFileEntry.
        if not is_merged:
            menu.addSeparator()
            offset = getattr(self, "centroid_offset_mhz", 0.0)
            offset_label = (
                f"Set centroid offset…  ({offset:+.2f} MHz)"
                if offset else "Set centroid offset…")
            offset_action = menu.addAction(offset_label)

            def _edit_offset():
                # Use a custom themed dialog so the app's dark QDialog
                # stylesheet applies; the stock input-double popup renders
                # a paler panel whose field reads too low-contrast.
                dlg = CentroidOffsetDialog(
                    run_number=self.run_number,
                    current_value=float(offset),
                    parent=self)
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    self.centroid_offset_mhz = float(dlg.value())
                    # code review 2026-06-02, centroid-offset-no-feedback:
                    # acknowledge the stored value in the status bar so the
                    # user sees it took effect (applies at frequency merge).
                    win = self.window()
                    if hasattr(win, "statusBar"):
                        win.statusBar().showMessage(
                            f"Centroid offset for run {self.run_number} "
                            f"set to {self.centroid_offset_mhz:+.2f} MHz "
                            "(applied at frequency merge)", 5000)

            offset_action.triggered.connect(_edit_offset)

            # Per-file scan filter. Splits share their parent ASDF, so
            # the filter key is the parent's path; the dialog opens
            # against the file that actually carries bunch data.
            from gui.scan_filter import get_registry
            scan_path = (getattr(self, "parent_path", None)
                          or self.filepath)
            n_excluded = len(get_registry().get(scan_path))
            sf_label = (f"Filter scans…  ({n_excluded} excluded)"
                        if n_excluded else "Filter scans…")
            scan_action = menu.addAction(sf_label)
            scan_action.setToolTip(
                "Open the per-scan filter for this file. Each scan is "
                "one voltage sweep; excluded scans are dropped from "
                "all downstream binning and fitting.")

            def _open_scan_filter():
                from gui.scan_filter_dialog import ScanFilterDialog
                # PMT gate comes from the enclosing PA tab when
                # available so the per-scan rate matches the user's
                # current channel selection; defaults to (3, 4) when
                # not reachable (FileEntry detached, headless tests).
                pmt_gate = self._pmt_gate_from_parent()
                ScanFilterDialog(
                    scan_path, pmt_gate=pmt_gate, parent=self).exec()

            scan_action.triggered.connect(_open_scan_filter)

            # Per-file voltage calibration. Same key as the scan filter (a
            # split shares its parent's calibration), and the same registry
            # the Analysis tab reads, so a fix made here is the fix the fit
            # uses.
            from gui.calibration import get_registry as _get_cal_registry
            cal_spec = _get_cal_registry().get(scan_path)
            cal_action = menu.addAction(
                "Calibration…  (overridden)" if cal_spec else "Calibration…")
            cal_action.setToolTip(
                "Inspect this run's DAC→HV voltage calibration: the fit, its\n"
                "residuals, and what it costs in MHz. Exclude bad points,\n"
                "borrow another run's calibration, or enter coefficients.")
            cal_action.triggered.connect(self._open_calibration_dialog)

            overview_action = menu.addAction("Calibration overview…")
            overview_action.setToolTip(
                "Triage every loaded run's calibration at once, worst first.")
            overview_action.triggered.connect(self._open_calibration_overview)

        menu.exec(event.globalPos())

    # ── Voltage calibration ──

    def _cal_path(self):
        """A split shares its parent ASDF's calibration table."""
        return getattr(self, "parent_path", None) or self.filepath

    def _open_calibration_dialog(self):
        from gui.calibration_dialog import CalibrationDialog
        tab = self._pa_tab()
        CalibrationDialog(
            self._cal_path(),
            run_label=f"run_{self.run_number}",
            physics=self._cal_physics(),
            peers=(tab._cal_peers() if tab is not None else None),
            parent=self).exec()

    def _open_calibration_overview(self):
        from gui.calibration_dialog import CalibrationOverviewDialog
        tab = self._pa_tab()
        if tab is None:
            return
        CalibrationOverviewDialog(
            tab._cal_peers(), physics=self._cal_physics(),
            parent=self).exec()

    def _pa_tab(self):
        """Walk up to the enclosing PreAnalysisTab (None when detached)."""
        w = self.parentWidget()
        while w is not None:
            if hasattr(w, "_cal_peers"):
                return w
            w = w.parentWidget()
        return None

    def _cal_physics(self):
        """Doppler inputs so the dialog can quote MHz rather than volts.

        Mass and harmonic come from the tab, not from ``self.mass_amu``: the
        ASDF's ``MassAMU`` field is frequently absent (then it is None), and
        the tab's isotope selector is what the displayed spectrum is actually
        computed with. Quoting a shift against a different mass than the one
        on screen would be worse than quoting none.
        """
        tab = self._pa_tab()
        mass = self.mass_amu
        harmonic = 2
        if tab is not None:
            try:
                harmonic = int(tab._harmonic.value())
            except Exception:
                pass
            try:
                mass = tab._get_isotope_mass() or mass
            except Exception:
                pass
        return {
            "mass_amu": mass,
            "laser_cm": self.laser_sp,
            "cooler_v": self.cooler_v,
            "harmonic": harmonic,
        }

    def _pmt_gate_from_parent(self):
        """Return the enclosing PA tab's current PMT-channel selection.

        Used by the scan-filter dialog so its per-scan rate matches
        what the user sees in the main spectrum. Walks up the widget
        tree to find ``PreAnalysisTab``; if it can't (headless test,
        detached entry), returns the conventional ``(3, 4)`` default
        for signal channels.
        """
        w = self.parentWidget()
        while w is not None:
            if hasattr(w, "_channels"):
                return [i + 1 for i, cb in enumerate(w._channels)
                        if cb.isChecked()]
            w = w.parentWidget()
        return [3, 4]

    def _update_color_icon(self):
        self.color_btn.setIcon(_color_icon(self.color))

    def set_dark_active(self, on):
        """Tell the entry which plot theme is active so its list swatch
        and the drawn line use the matching colour."""
        self._dark_active = bool(on)
        self._update_color_icon()

    def _pick_color(self):
        """Open a small popup with color, alpha, and line style controls."""
        from PySide6.QtWidgets import QMenu, QWidgetAction
        menu = QMenu(self)

        # Color picker row — opens the dual light/dark chooser.
        color_action = menu.addAction("Colours (light + dark)…")
        color_action.triggered.connect(self._open_color_dialog)

        menu.addSeparator()

        # Alpha slider
        alpha_w = QWidget()
        al = QHBoxLayout(alpha_w)
        al.setContentsMargins(8, 4, 8, 4)
        al.addWidget(QLabel("Opacity:"))
        alpha_slider = QSlider(Qt.Orientation.Horizontal)
        alpha_slider.setRange(0, 100)
        alpha_slider.setValue(int(self._alpha * 100))
        alpha_slider.setFixedWidth(80)
        alpha_lbl = QLabel(f"{int(self._alpha * 100)}%")
        alpha_lbl.setFixedWidth(32)
        alpha_slider.valueChanged.connect(
            lambda v: (
                setattr(self, '_alpha', v / 100.0),
                alpha_lbl.setText(f"{v}%"),
                self.color_changed.emit(),
            ) and None)
        al.addWidget(alpha_slider)
        al.addWidget(alpha_lbl)
        alpha_act = QWidgetAction(menu)
        alpha_act.setDefaultWidget(alpha_w)
        menu.addAction(alpha_act)

        # Line style combo
        ls_w = QWidget()
        ll = QHBoxLayout(ls_w)
        ll.setContentsMargins(8, 4, 8, 4)
        ll.addWidget(QLabel("Style:"))
        ls_combo = QComboBox()
        _LS_OPTIONS = [
            ("Solid", "-"), ("Dashed", "--"),
            ("Dash-dot", "-."), ("Dotted", ":"),
        ]
        for label, _ in _LS_OPTIONS:
            ls_combo.addItem(label)
        # Set current
        for i, (_, code) in enumerate(_LS_OPTIONS):
            if code == self._linestyle:
                ls_combo.setCurrentIndex(i)
                break
        ls_combo.currentIndexChanged.connect(
            lambda idx: (
                setattr(self, '_linestyle', _LS_OPTIONS[idx][1]),
                self.color_changed.emit(),
            ) and None)
        ll.addWidget(ls_combo)
        ls_act = QWidgetAction(menu)
        ls_act.setDefaultWidget(ls_w)
        menu.addAction(ls_act)

        menu.exec(self.color_btn.mapToGlobal(
            self.color_btn.rect().bottomLeft()))

    def _open_color_dialog(self):
        dlg = LineColorDialog(self._color, self._dark_color, self,
                              title=f"Line colours — run {self.run_number}")
        if dlg.exec():
            self._color, self._dark_color = dlg.colors()
            self._update_color_icon()
            self.color_changed.emit()

    @property
    def color(self):
        """The colour for the ACTIVE plot theme (light or dark)."""
        return self._dark_color if self._dark_active else self._color

    @color.setter
    def color(self, val):
        # Sets the light-mode colour (the serialised "color" field).
        self._color = val
        self._update_color_icon()

    @property
    def light_color(self):
        return self._color

    @light_color.setter
    def light_color(self, val):
        self._color = val
        self._update_color_icon()

    @property
    def dark_color(self):
        return self._dark_color

    @dark_color.setter
    def dark_color(self, val):
        self._dark_color = val
        self._update_color_icon()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = float(val)

    @property
    def linestyle(self):
        return self._linestyle

    @linestyle.setter
    def linestyle(self, val):
        self._linestyle = val

    def update_detail(self):
        """Refresh the detail label after data is loaded."""
        if self.np_ts is not None and len(self.np_ts) > 1:
            self.run_time_s = self.np_ts.max() - self.np_ts.min()
        else:
            self.run_time_s = 0.0

        if self.run_time_s >= 3600:
            rt_str = f"{self.run_time_s / 3600:.1f} h"
        elif self.run_time_s >= 60:
            rt_str = f"{self.run_time_s / 60:.1f} min"
        else:
            rt_str = f"{self.run_time_s:.0f} s"

        self._detail_label.setText(
            f"V={self.cooler_v:.1f}  |  "
            f"\u03bb={self.laser_sp:.4f} cm\u207b\u00b9  |  "
            f"t={rt_str}")

    @property
    def is_loaded(self):
        return self.np_v is not None

    @property
    def selected(self):
        return getattr(self, "_selected", False)

    @selected.setter
    def selected(self, val):
        # No-op when unchanged: highlighting one entry calls this on EVERY
        # entry, so the guard avoids re-applying identical stylesheets N times.
        if getattr(self, "_selected", None) == val:
            return
        self._selected = val
        self.setStyleSheet(self._STYLE_SELECTED if val else self._STYLE_NORMAL)

    def mousePressEvent(self, event):
        self.clicked.emit(self)
        super().mousePressEvent(event)

def _normalize_source_info(source_info):
    """Coerce a heterogeneous ``source_info`` list into a list of dicts
    with a known shape.

    Older Pre-Analysis YAMLs (and any code path still passing 2-element
    pairs) used ``[(run_number, filepath), ...]``. Newer entries store
    per-source physics parameters too -- cooler_v, laser_sp, mass_amu,
    harmonic -- so the voltage-merged fit pipeline can pick a sensible
    Doppler shift after the fact.

    Each output entry has the keys ``run_number``, ``filepath``,
    ``cooler_v``, ``laser_sp``, ``mass_amu``, ``harmonic`` \u2014 values
    missing in the input become ``None`` so callers can distinguish
    "not stored" from "explicitly zero".
    """
    out = []
    for item in source_info or []:
        if isinstance(item, dict):
            out.append({
                "run_number": item.get("run_number"),
                "filepath":   item.get("filepath"),
                "cooler_v":   item.get("cooler_v"),
                "laser_sp":   item.get("laser_sp"),
                "mass_amu":   item.get("mass_amu"),
                "harmonic":   item.get("harmonic"),
            })
        else:
            # Old 2-list / 2-tuple form: (run_number, filepath).
            try:
                rn, fp = item[0], item[1]
            except (IndexError, TypeError):
                rn, fp = None, None
            out.append({
                "run_number": rn,
                "filepath":   fp,
                "cooler_v":   None,
                "laser_sp":   None,
                "mass_amu":   None,
                "harmonic":   None,
            })
    return out


class MergedFileEntry(FileEntry):
    """A synthetic entry representing a merged spectrum from multiple runs.

    Carries two flavors of metadata in addition to the merged (x, y):

    ``source_info``
        Normalized list of dicts (one per source file): ``run_number``,
        ``filepath``, ``cooler_v``, ``laser_sp``, ``mass_amu``,
        ``harmonic``. See :func:`_normalize_source_info` for the
        backward-compat path on old (run, path) pairs.

    Merge-level metadata (``merge_cooler_v``, ``merge_laser_sp``,
    ``merge_mass_amu``, ``merge_harmonic``)
        User-chosen Doppler parameters that apply to the *merged*
        spectrum as a whole \u2014 needed by downstream code that treats a
        voltage merge as a synthetic ASDF (one common cooler/laser/mass
        to Doppler-shift the binned voltage axis against). ``None``
        means "fall back to mean-of-sources at use time"; a numeric
        value means "use exactly this".
    """

    # Signals emitted from the right-click context menu so the PA tab
    # can handle View / Edit / Export. Routing through signals keeps the
    # entry widget free of references to its parent tab.
    view_requested = Signal(object)
    edit_requested = Signal(object)
    export_requested = Signal(object)

    def __init__(self, name, merged_x, merged_y, merge_domain,
                 source_info,
                 merge_cooler_v=None, merge_laser_sp=None,
                 merge_mass_amu=None, merge_harmonic=None,
                 per_run=None,
                 color="#333333", parent=None):
        # Pass dummy values to FileEntry.__init__
        super().__init__(
            filepath=f"[merged] {name}",
            run_number=name,
            cooler_v=0, date="", laser_sp=0, mass_amu=0,
            color=color, parent=parent,
        )
        self.merged_x = np.array(merged_x, dtype=float)
        self.merged_y = np.array(merged_y, dtype=float)
        self.merge_domain = merge_domain  # "voltage" or "frequency"
        self.source_info = _normalize_source_info(source_info)
        # Merge-level Doppler metadata, used when Doppler-shifting a
        # voltage-merged spectrum at fit time.
        self.merge_cooler_v = merge_cooler_v
        self.merge_laser_sp = merge_laser_sp
        self.merge_mass_amu = merge_mass_amu
        self.merge_harmonic = merge_harmonic
        # In-memory per_run from compute_merged_spectrum (TOF arrays,
        # timestamps, dwell times, etc.). NOT persisted to YAML --
        # large enough to bloat the config noticeably, and the
        # source_info dict already carries the essential physics fields
        # (cooler/laser/mass/harmonic). When the entry is round-tripped
        # via YAML this is None; the Analysis bridge then falls back
        # to source_info-derived reconstruction.
        self.per_run = per_run
        self._is_merged = True
        # Merged entries have no on-disk ASDF backing them; hide the
        # inherited reload button so the icon doesn't tease an
        # action that would crash.
        self.reload_btn.hide()

        # Override label
        n = len(self.source_info)
        domain_label = "freq" if merge_domain == "frequency" else "voltage"
        self._name_label.setText(f"<b>\u2726 {name}</b>")
        self._detail_label.setText(
            f"Merged ({n} runs, {domain_label})  |  "
            f"{len(self.merged_x)} bins")
        run_labels = ", ".join(
            str(s.get("run_number")) for s in self.source_info)
        self.setToolTip(
            f"Merged spectrum: {name}\n"
            f"Domain: {merge_domain}\n"
            f"Bins: {len(self.merged_x)}\n"
            f"Source runs: {run_labels}")

    @property
    def is_loaded(self):
        return self.merged_x is not None and len(self.merged_x) > 0

    def update_detail(self):
        pass  # No raw event data to update from

    def to_merged_data(self):
        """Return the ``merged_data`` dict shape that
        ``compute_merged_spectrum`` produces and that
        ``MergeViewDialog`` / ``export_merged_asdf`` consume.

        Source-of-truth for shuttling a PA-side ``MergedFileEntry``
        into the shared merge/view/export paths without each caller
        having to remember the per-field mapping.
        """
        y = np.asarray(self.merged_y, dtype=float)
        # sqrt(y+1) to match compute_merged_spectrum and the fit paths so a
        # PA-side merged entry weights identically (code review 2026-06-02).
        yerr = np.sqrt(y + 1.0)
        return {
            "merged_name":    str(self.run_number),
            "x":              np.asarray(self.merged_x, dtype=float),
            "y":              y,
            "yerr":           yerr,
            "x_unit":         ("MHz" if self.merge_domain == "frequency"
                                else "V"),
            "bin_step_mhz":   0.0,
            "merge_metadata": {
                "cooler_v": self.merge_cooler_v,
                "laser_sp": self.merge_laser_sp,
                "mass_amu": self.merge_mass_amu,
                "harmonic": self.merge_harmonic,
            },
            "source_runs":  [s.get("run_number")
                              for s in (self.source_info or [])],
            "source_files": [s.get("filepath")
                              for s in (self.source_info or [])],
            "per_run":      list(self.per_run) if self.per_run else [],
            "metadata":     {},
        }


class SplitFileEntry(FileEntry):
    """File-list entry for a loaded ``.vasdf`` virtual split.

    A ``.vasdf`` is a tiny YAML sidecar that points at a parent ASDF
    plus a raw-voltage range (``split_lo``, ``split_hi``) and optional
    metadata overrides. The original ASDF is never touched. On load,
    the file loader reads the descriptor, loads the parent ASDF's
    per-event arrays into this entry's own ``np_*`` fields, and
    applies the V-gate at every consumer (the pre-analysis preview's
    ``_bin_spectrum`` and the analysis-tab fitter).

    Identity is the ``.vasdf`` path, so the existing path-keyed maps
    (overrides, corrections, fit names) treat each split as an
    independent run \u2014 different splits of the same parent ASDF
    don't collide.
    """

    def __init__(self, vasdf_path, parent_path, source_id,
                 split_lo, split_hi,
                 cooler_v, date, laser_sp, mass_amu,
                 label=None, metadata_override=None,
                 color="#000080", dark_color=None, parent=None):
        super().__init__(
            filepath=vasdf_path,
            run_number=label or source_id,
            cooler_v=cooler_v, date=date,
            laser_sp=laser_sp, mass_amu=mass_amu,
            color=color, dark_color=dark_color, parent=parent,
        )
        self.parent_path = parent_path
        self.source_id = source_id
        self.split_lo = float(split_lo)
        self.split_hi = float(split_hi)
        self.label = label or source_id
        self.metadata_override = dict(metadata_override or {})
        self._is_split = True

        self._name_label.setText(f"<b>\u2702 {self.label}</b>")
        self._detail_label.setText(
            f"V\u2208[{self.split_lo:.1f}, {self.split_hi:.1f}]  |  "
            f"V_cool={self.cooler_v:.1f}  |  "
            f"\u03bb={self.laser_sp:.4f} cm\u207b\u00b9")
        comment = self.metadata_override.get("comment", "").strip()
        comment_line = f"\nComment: {comment}" if comment else ""
        self.setToolTip(
            f"Virtual split file: {os.path.basename(vasdf_path)}\n"
            f"Parent ASDF: {os.path.basename(parent_path)}\n"
            f"Source ID: {source_id}\n"
            f"V-gate: [{self.split_lo:.2f}, {self.split_hi:.2f}] V\n"
            f"Cooler: {self.cooler_v:.2f} V\n"
            f"Laser: {self.laser_sp:.6f} cm\u207b\u00b9"
            f"{comment_line}")

    def update_detail(self):
        # Splits are voltage-defined, not time-defined; keep the
        # split-specific detail label stable.
        pass


# ══════════════════════════════════════════════════════════════════
#  Slider-limits dialog (opened from the "..." button)
# ══════════════════════════════════════════════════════════════════
class _SliderLimitsDialog(QDialog):
    """Small dialog to set slider min/max for a parameter."""

    def __init__(self, key, lo_val, hi_val, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Slider Limits \u2013 {key}")
        self.setFixedWidth(220)
        form = QFormLayout(self)
        self._lo = QDoubleSpinBox()
        self._lo.setRange(-1e8, 1e8)
        self._lo.setDecimals(2)
        self._lo.setValue(lo_val)
        self._hi = QDoubleSpinBox()
        self._hi.setRange(-1e8, 1e8)
        self._hi.setDecimals(2)
        self._hi.setValue(hi_val)
        form.addRow("Min:", self._lo)
        form.addRow("Max:", self._hi)
        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        form.addRow(btn)

    def values(self):
        return self._lo.value(), self._hi.value()


# Virtual-split editing lives in the standalone
# gui.split_editor.SplitFileEditor window, invoked from
# Edit > Split File… in the main menu. Splits are first-class
# .vasdf files loaded through the regular Open dialog.


# ══════════════════════════════════════════════════════════════════
#  Custom QDoubleSpinBox with focus highlight and arrow-key support
# ══════════════════════════════════════════════════════════════════
class _HFSSpinBox(QDoubleSpinBox):
    """SpinBox with cursor-position-aware stepping and digit highlighting.

    - Enter: confirms value, stays in current field
    - Tab: moves to next parameter (via enter_pressed signal)
    - Up/Down: step the digit at cursor position
    - Left/Right: move cursor between digits with highlight
    - Ctrl+C/V: explicit clipboard copy/paste
    """

    enter_pressed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_ss = ""
        # Catch mouse clicks on the line edit to re-highlight digits
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
        self._default_ss = self.styleSheet()
        self.setStyleSheet("border: 2px solid #d4a017; border-radius: 3px;")
        super().focusInEvent(event)
        QTimer.singleShot(0, self._highlight_digit)

    def focusOutEvent(self, event):
        self.setStyleSheet(self._default_ss)
        super().focusOutEvent(event)

    # -- digit helpers --

    def _find_digit_pos(self, pos=None):
        """Return a valid digit position, skipping '.', '-', spaces.

        Uses selectionStart() when a digit is highlighted so that
        repeated Up/Down presses keep stepping the same digit.
        """
        le = self.lineEdit()
        text = le.text()
        if pos is None:
            # selectionStart() gives the LEFT edge of the highlight;
            # cursorPosition() gives the RIGHT edge, which is one past
            # the selected digit — that would pick the wrong digit.
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
        """Select the single digit at cursor position."""
        le = self.lineEdit()
        text = le.text()
        if not text:
            return
        pos = self._find_digit_pos()
        if 0 <= pos < len(text) and text[pos].isdigit():
            le.setSelection(pos, 1)

    def _cursor_power(self):
        """Return the power-of-10 of the digit at cursor."""
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

    # -- stepping --

    def stepBy(self, steps):
        """Step by the place value of the digit at cursor position."""
        power = self._cursor_power()
        step_size = 10.0 ** power
        new_val = self.value() + steps * step_size
        new_val = round(new_val, self.decimals())
        new_val = max(self.minimum(), min(new_val, self.maximum()))

        self.setValue(new_val)

        # Restore highlight to the same power-of-10 position
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

    # -- key handling --

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier

        # Enter: confirm value, stay in field
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.interpretText()
            return

        # Tab: move to next parameter
        if key == Qt.Key.Key_Tab:
            self.enter_pressed.emit()
            return

        # Ctrl+C: copy numeric value
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_C:
            QApplication.clipboard().setText(str(self.value()))
            return

        # Ctrl+V: paste numeric value
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_V:
            text = QApplication.clipboard().text().strip()
            try:
                self.setValue(float(text))
            except (ValueError, OverflowError):
                pass
            return

        # Ctrl+A: select all
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_A:
            self.lineEdit().selectAll()
            return

        # Up/Down: step based on cursor position
        if key == Qt.Key.Key_Up:
            self.stepBy(1)
            return
        if key == Qt.Key.Key_Down:
            self.stepBy(-1)
            return

        # Left/Right (no modifiers): move between digits with highlight
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


def _make_hfs_spin(default, lo, hi, decimals, step, tooltip=""):
    s = _HFSSpinBox()
    s.setLocale(QLocale(QLocale.Language.English,
                        QLocale.Country.UnitedStates))
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(default)
    s.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    if tooltip:
        s.setToolTip(tooltip)
    s.setKeyboardTracking(False)
    return s


# ══════════════════════════════════════════════════════════════════
#  HFS Model Panel (one per model)
# ══════════════════════════════════════════════════════════════════
class HFSModelPanel(QGroupBox):
    """Interactive HFS model parameter panel with sliders."""
    params_changed = Signal()

    def __init__(self, name="Model 1", color="#000000",
                 dark_color=None, parent=None):
        super().__init__(name, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(lambda _: self.params_changed.emit())
        # Light + dark model-curve colours; active one follows the tab.
        self._color = color
        self._dark_color = dark_color or color
        self._dark_active = False
        self._building = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 6, 3, 3)
        layout.setSpacing(1)

        # Top row: name + color + style + alpha
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        self._name_edit = QLineEdit(name)
        self._name_edit.setPlaceholderText("Model name")
        self._name_edit.setFixedWidth(80)
        self._name_edit.editingFinished.connect(self._on_name_changed)
        top_row.addWidget(self._name_edit)
        self.color_btn = QToolButton()
        self.color_btn.setFixedSize(20, 20)
        self._update_color_icon()
        self.color_btn.clicked.connect(self._pick_color)
        top_row.addWidget(self.color_btn)
        self._linestyle_combo = QComboBox()
        self._linestyle_combo.addItems(
            ["Solid", "Dashed", "Dotted", "Dash-dot"])
        self._linestyle_combo.setFixedWidth(70)
        self._linestyle_combo.currentIndexChanged.connect(
            lambda: self.params_changed.emit())
        top_row.addWidget(self._linestyle_combo)
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(100)
        self._alpha_slider.setToolTip("Transparency")
        self._alpha_slider.setFixedWidth(50)
        self._alpha_slider.valueChanged.connect(
            lambda: self.params_changed.emit())
        top_row.addWidget(self._alpha_slider)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Nuclear parameters — all three on ONE compact row (the old
        # one-per-row layout cost two extra rows and pushed the panel
        # into scrolling).
        spin_row = QHBoxLayout()
        spin_row.setContentsMargins(0, 0, 0, 0)
        spin_row.setSpacing(4)
        for i, (attr, label, default, tip) in enumerate([
            ("spin_I",  "I",  3.5, "Nuclear spin I"),
            ("spin_Jl", "Jl", 4.5, "J lower"),
            ("spin_Ju", "Ju", 5.5, "J upper"),
        ]):
            lbl = QLabel(label)
            if i == 0:
                lbl.setFixedWidth(65)   # align with the param labels
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                             | Qt.AlignmentFlag.AlignVCenter)
            spin_row.addWidget(lbl)
            sb = _make_double(default, 0, 20, 1, 0.5, tooltip=tip)
            sb.setFixedWidth(62)
            spin_row.addWidget(sb)
            setattr(self, attr, sb)
        spin_row.addStretch()
        layout.addLayout(spin_row)

        # A ratio (inserted after Au row below)
        self._fix_a_ratio = QCheckBox("Fix Al/Au")
        self._fix_a_ratio.setToolTip("Lock Al/Au ratio")
        self._a_ratio = _make_double(1.0, -1000, 1000, 4, 0.01,
                                     tooltip="Al / Au ratio")
        self._a_ratio.setFixedWidth(80)

        # B ratio (inserted after Bu row below)
        self._fix_b_ratio = QCheckBox("Fix Bl/Bu")
        self._fix_b_ratio.setToolTip("Lock Bl/Bu ratio")
        self._b_ratio = _make_double(1.0, -1000, 1000, 4, 0.01,
                                     tooltip="Bl / Bu ratio")
        self._b_ratio.setFixedWidth(80)

        # HFS parameters -- single-line: label | value | slider | "..." button
        self._slider_params = {}
        self._ordered_keys = []
        params_info = [
            ("Al", "Al", 0.0, -500.0, 500.0),
            ("Au", "Au", 0.0, -500.0, 500.0),
            ("Bl", "Bl", 0.0, -500.0, 500.0),
            ("Bu", "Bu", 0.0, -500.0, 500.0),
            ("centroid", "Centroid", 0.0, -50000.0, 50000.0),
            ("scale", "Scale", 100.0, 0.0, 10000.0),
            ("bkg", "Bkg", 0.0, 0.0, 100.0),
            ("fwhm_g", "FWHM_G", 50.0, 1.0, 500.0),
            ("fwhm_l", "FWHM_L", 50.0, 1.0, 500.0),
        ]

        for key, label, default, lo, hi in params_info:
            self._ordered_keys.append(key)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            lbl = QLabel(label)
            lbl.setFixedWidth(65)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                             | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)

            val_spin = _make_hfs_spin(default, -1e8, 1e8, 2, 1.0,
                                      tooltip=f"{key}")
            val_spin.setFixedWidth(95)
            row.addWidget(val_spin)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 10000)
            slider.setValue(5000)
            slider.setFixedHeight(18)
            row.addWidget(slider, 1)

            lim_btn = QToolButton()
            lim_btn.setText("\u2026")
            lim_btn.setFixedSize(20, 20)
            lim_btn.setToolTip("Adjust slider limits")
            row.addWidget(lim_btn)

            layout.addLayout(row)

            # Insert ratio row after Au / Bu
            if key == "Au":
                a_ratio_row = QHBoxLayout()
                a_ratio_row.setContentsMargins(0, 0, 0, 0)
                a_ratio_row.setSpacing(4)
                a_ratio_row.addSpacing(69)  # label width + spacing
                a_ratio_row.addWidget(self._fix_a_ratio)
                a_ratio_row.addWidget(self._a_ratio)
                a_ratio_row.addStretch()
                layout.addLayout(a_ratio_row)
            elif key == "Bu":
                b_ratio_row = QHBoxLayout()
                b_ratio_row.setContentsMargins(0, 0, 0, 0)
                b_ratio_row.setSpacing(4)
                b_ratio_row.addSpacing(69)
                b_ratio_row.addWidget(self._fix_b_ratio)
                b_ratio_row.addWidget(self._b_ratio)
                b_ratio_row.addStretch()
                layout.addLayout(b_ratio_row)

            # Store hidden lo/hi values (no visible spinbox)
            self._slider_params[key] = {
                "_lo": lo, "_hi": hi,
                "slider": slider, "value": val_spin,
            }

            slider.valueChanged.connect(
                lambda pos, k=key: self._on_slider_moved(k))
            val_spin.valueChanged.connect(
                lambda val, k=key: self._on_value_changed(k))
            lim_btn.clicked.connect(
                lambda checked=False, k=key: self._open_limits_dialog(k))

            # Enter -> focus next parameter
            val_spin.enter_pressed.connect(
                lambda k=key: self._focus_next(k))

        self.spin_I.valueChanged.connect(self._rebuild_peak_rows)
        self.spin_Jl.valueChanged.connect(self._rebuild_peak_rows)
        self.spin_Ju.valueChanged.connect(self._rebuild_peak_rows)

        # ── Peak Amplitudes section (collapsible) ──
        # The panel runs at layout spacing 1 for density; give this row
        # its own gap so it doesn't fuse with the last slider above.
        layout.addSpacing(6)
        peaks_row = QHBoxLayout()
        self._peaks_toggle = QCheckBox("Peak Amplitudes")
        self._peaks_toggle.setToolTip("Show per-peak amplitude controls")
        self._peaks_toggle.toggled.connect(self._toggle_peaks_section)
        peaks_row.addWidget(self._peaks_toggle)
        self._peak_labels_toggle = QCheckBox("Show Labels")
        self._peak_labels_toggle.setToolTip(
            "Annotate each peak on the spectrum plot with its "
            "transition label (e.g. 3\u21924)")
        self._peak_labels_toggle.toggled.connect(
            lambda _: self.params_changed.emit())
        peaks_row.addWidget(self._peak_labels_toggle)
        peaks_row.addStretch()
        layout.addLayout(peaks_row)

        self._peaks_container = QWidget()
        self._peaks_layout = QVBoxLayout(self._peaks_container)
        self._peaks_layout.setContentsMargins(0, 0, 0, 0)
        self._peaks_layout.setSpacing(1)
        self._peaks_container.setVisible(False)
        layout.addWidget(self._peaks_container)

        self._peak_rows = []       # list of dicts per peak
        self._racah_cache = {}     # label -> racah amplitude value

        # Compact heights so a whole model fits the left column without
        # scrolling (the theme's default field height is ~24 px).
        for _cls in (QDoubleSpinBox, QSpinBox):
            for _w in self.findChildren(_cls):
                _w.setFixedHeight(22)

        # Sync initial slider positions to defaults
        self._building = False
        for key in self._slider_params:
            self._on_value_changed(key)
        self._building = True
        self._building = False

    # -- Name / color --

    def _on_name_changed(self):
        name = self._name_edit.text().strip()
        if name:
            self.setTitle(name)
            self.params_changed.emit()

    @property
    def model_name(self):
        return self._name_edit.text().strip() or self.title()

    def _update_color_icon(self):
        self.color_btn.setIcon(_color_icon(self.color))

    def set_dark_active(self, on):
        self._dark_active = bool(on)
        self._update_color_icon()

    def _pick_color(self):
        dlg = LineColorDialog(self._color, self._dark_color, self,
                              title="Model colours — light + dark")
        if dlg.exec():
            self._color, self._dark_color = dlg.colors()
            self._update_color_icon()
            self.params_changed.emit()

    @property
    def color(self):
        """Colour for the ACTIVE plot theme."""
        return self._dark_color if self._dark_active else self._color

    @property
    def light_color(self):
        return self._color

    @property
    def dark_color(self):
        return self._dark_color

    @dark_color.setter
    def dark_color(self, val):
        self._dark_color = val
        self._update_color_icon()

    @property
    def linestyle(self):
        mapping = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
        return mapping.get(self._linestyle_combo.currentText(), "-")

    @property
    def alpha(self):
        return self._alpha_slider.value() / 100.0

    # -- Slider / value sync --

    def _on_slider_moved(self, key):
        p = self._slider_params[key]
        lo, hi = p["_lo"], p["_hi"]
        pos = p["slider"].value()
        value = lo + (hi - lo) * pos / 10000.0
        p["value"].blockSignals(True)
        p["value"].setValue(value)
        p["value"].blockSignals(False)
        if not self._building:
            self._apply_ratio(key, value)
            self.params_changed.emit()

    def _on_value_changed(self, key):
        p = self._slider_params[key]
        val = p["value"].value()
        lo, hi = p["_lo"], p["_hi"]
        if hi > lo:
            pos = int(10000 * (val - lo) / (hi - lo))
            pos = max(0, min(10000, pos))
        else:
            pos = 5000
        p["slider"].blockSignals(True)
        p["slider"].setValue(pos)
        p["slider"].blockSignals(False)
        if not self._building:
            self._apply_ratio(key, val)
            self.params_changed.emit()

    def _apply_ratio(self, key, val):
        """Enforce A or B ratio coupling."""
        if key == "Au" and self._fix_a_ratio.isChecked():
            r = self._a_ratio.value()
            self._set_linked("Al", val * r)
        elif key == "Al" and self._fix_a_ratio.isChecked():
            r = self._a_ratio.value()
            if r != 0:
                self._set_linked("Au", val / r)
        elif key == "Bu" and self._fix_b_ratio.isChecked():
            r = self._b_ratio.value()
            self._set_linked("Bl", val * r)
        elif key == "Bl" and self._fix_b_ratio.isChecked():
            r = self._b_ratio.value()
            if r != 0:
                self._set_linked("Bu", val / r)

    def _set_linked(self, key, val):
        """Set a linked parameter without re-triggering ratio logic."""
        p = self._slider_params[key]
        p["value"].blockSignals(True)
        p["value"].setValue(val)
        p["value"].blockSignals(False)
        lo, hi = p["_lo"], p["_hi"]
        if hi > lo:
            pos = int(10000 * (val - lo) / (hi - lo))
            pos = max(0, min(10000, pos))
        else:
            pos = 5000
        p["slider"].blockSignals(True)
        p["slider"].setValue(pos)
        p["slider"].blockSignals(False)

    # -- Limits dialog --

    def _open_limits_dialog(self, key):
        p = self._slider_params[key]
        dlg = _SliderLimitsDialog(key, p["_lo"], p["_hi"], parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_lo, new_hi = dlg.values()
            p["_lo"] = new_lo
            p["_hi"] = new_hi
            self._on_value_changed(key)

    # -- Focus navigation --

    def _focus_next(self, key):
        idx = self._ordered_keys.index(key)
        next_idx = (idx + 1) % len(self._ordered_keys)
        self._slider_params[self._ordered_keys[next_idx]]["value"].setFocus()
        self._slider_params[self._ordered_keys[next_idx]]["value"].selectAll()

    # -- Peak amplitudes section --

    def _toggle_peaks_section(self, on):
        self._peaks_container.setVisible(on)
        if on and not self._peak_rows:
            self._rebuild_peak_rows()
        else:
            self.params_changed.emit()

    def _rebuild_peak_rows(self):
        """Recreate peak rows based on current I, Jl, Ju."""
        # Clear old rows
        for row in self._peak_rows:
            row["widget"].deleteLater()
        self._peak_rows.clear()
        self._racah_cache.clear()

        I = self.spin_I.value()
        Jl = self.spin_Jl.value()
        Ju = self.spin_Ju.value()
        if I < 0 or Jl < 0 or Ju < 0:
            self.params_changed.emit()
            return
        if (2*I) != int(2*I) or (2*Jl) != int(2*Jl) or (2*Ju) != int(2*Ju):
            self.params_changed.emit()
            return

        try:
            import satlas2  # lazy import (avoids satlas2 cost at startup)
            hfs = satlas2.HFS(
                I=I, J=[Jl, Ju], A=[0, 0], B=[0, 0], C=[0, 0],
                df=0, scale=1, racah=True, fwhmg=50, fwhml=50)
        except Exception:
            self.params_changed.emit()
            return

        lines = hfs.lines
        # Show satlas2's Racah amplitudes verbatim (no normalization):
        # the absolute amplitude scale is degenerate with the global
        # Scale parameter in the HFS model, so rescaling the per-peak
        # amplitudes is cosmetic only and would obscure the satlas2
        # internal values when cross-checking against external sources.
        for label in lines:
            amp_key = f"Amp{label}"
            self._racah_cache[label] = float(hfs.params[amp_key].value)

        # Get all peak labels for linked-to combos
        all_labels = list(lines)

        for label in lines:
            racah_val = self._racah_cache[label]
            w = QWidget()
            outer = QVBoxLayout(w)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            # Main row: label | mode | value
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)

            lbl = QLabel(label)
            lbl.setMinimumWidth(30)
            lbl.setToolTip(f"Racah: {racah_val:.6f}")
            row_layout.addWidget(lbl)

            racah_lbl = QLabel(f"({racah_val:.2f})")
            racah_lbl.setStyleSheet("color: gray; font-size: 10px;")
            row_layout.addWidget(racah_lbl)

            mode = QComboBox()
            mode.addItems(["Racah", "Free", "Linked"])
            mode.setFixedWidth(74)
            row_layout.addWidget(mode)

            val_spin = _make_hfs_spin(racah_val, 0.0, 100.0, 4, 0.01)
            val_spin.setFixedWidth(95)
            val_spin.setEnabled(False)
            row_layout.addWidget(val_spin)
            row_layout.addStretch()
            outer.addLayout(row_layout)

            # Link sub-row (hidden by default): indent + "= ratio x" + combo
            link_row_w = QWidget()
            link_row = QHBoxLayout(link_row_w)
            link_row.setContentsMargins(20, 0, 0, 0)
            link_row.setSpacing(3)

            ratio_spin = _make_hfs_spin(1.0, -100.0, 100.0, 3, 0.1,
                                        tooltip="Ratio: this = ratio * linked")
            ratio_spin.setFixedWidth(58)
            link_row.addWidget(ratio_spin)

            link_row.addWidget(QLabel("\u00d7"))

            link_combo = QComboBox()
            link_combo.addItems([l for l in all_labels if l != label])
            link_combo.setMinimumWidth(50)
            link_row.addWidget(link_combo)

            link_row.addStretch()
            link_row_w.setVisible(False)
            outer.addWidget(link_row_w)

            row_data = {
                "widget": w,
                "label": label,
                "racah_val": racah_val,
                "mode": mode,
                "value": val_spin,
                "link_widget": link_row_w,
                "link_combo": link_combo,
                "ratio": ratio_spin,
            }
            self._peak_rows.append(row_data)
            self._peaks_layout.addWidget(w)

            mode.currentTextChanged.connect(
                lambda text, rd=row_data: self._on_peak_mode_changed(rd))
            val_spin.valueChanged.connect(
                lambda v: self.params_changed.emit())
            link_combo.currentTextChanged.connect(
                lambda t: self.params_changed.emit())
            ratio_spin.valueChanged.connect(
                lambda v: self.params_changed.emit())

        if not self._peaks_toggle.isChecked():
            return
        self.params_changed.emit()

    def _on_peak_mode_changed(self, row_data):
        mode = row_data["mode"].currentText()
        if mode == "Racah":
            row_data["value"].setEnabled(False)
            row_data["value"].setValue(row_data["racah_val"])
            row_data["link_widget"].setVisible(False)
        elif mode == "Free":
            row_data["value"].setEnabled(True)
            row_data["link_widget"].setVisible(False)
        else:  # Linked
            row_data["value"].setEnabled(False)
            row_data["link_widget"].setVisible(True)
        self.params_changed.emit()

    def get_peak_overrides(self):
        """Return dict of {label: amplitude} for non-Racah peaks.
        Linked peaks are resolved here."""
        if not self._peaks_toggle.isChecked() or not self._peak_rows:
            return {}
        overrides = {}
        # First pass: collect free values
        for row in self._peak_rows:
            mode = row["mode"].currentText()
            if mode == "Free":
                overrides[row["label"]] = row["value"].value()
        # Second pass: resolve linked values
        for row in self._peak_rows:
            if row["mode"].currentText() != "Linked":
                continue
            linked_label = row["link_combo"].currentText()
            ratio = row["ratio"].value()
            # Find the linked peak's effective amplitude
            linked_amp = self._racah_cache.get(linked_label, 1.0)
            for other in self._peak_rows:
                if other["label"] == linked_label:
                    other_mode = other["mode"].currentText()
                    if other_mode == "Free":
                        linked_amp = other["value"].value()
                    elif other_mode == "Linked":
                        # Already resolved or use racah
                        linked_amp = overrides.get(
                            linked_label,
                            self._racah_cache.get(linked_label, 1.0))
                    break
            overrides[row["label"]] = ratio * linked_amp
        return overrides

    # -- Public API --

    def get_value(self, key):
        return self._slider_params[key]["value"].value()

    def set_value(self, key, val):
        self._slider_params[key]["value"].setValue(val)

    def get_model_params(self):
        return {
            "I": self.spin_I.value(),
            "Jl": self.spin_Jl.value(),
            "Ju": self.spin_Ju.value(),
            "Al": self.get_value("Al"),
            "Au": self.get_value("Au"),
            "Bl": self.get_value("Bl"),
            "Bu": self.get_value("Bu"),
            "centroid": self.get_value("centroid"),
            "scale": self.get_value("scale"),
            "bkg": self.get_value("bkg"),
            "fwhm_g": self.get_value("fwhm_g"),
            "fwhm_l": self.get_value("fwhm_l"),
        }

    def hyperfine_state(self):
        """Per-constant A/B state for an Analysis-side import.

        Returns ``{name: (value, fixed)}`` for the hyperfine constants
        Al/Au/Bl/Bu. ``fixed`` comes from the ratio locks: the Al/Au
        pair is fixed when ``Fix Al/Au`` is ticked, Bl/Bu when
        ``Fix Bl/Bu`` is ticked. Names match the Analysis ModelBlock
        rows exactly so the import can map them 1:1.
        """
        fix_a = self._fix_a_ratio.isChecked()
        fix_b = self._fix_b_ratio.isChecked()
        return {
            "Al": (self.get_value("Al"), fix_a),
            "Au": (self.get_value("Au"), fix_a),
            "Bl": (self.get_value("Bl"), fix_b),
            "Bu": (self.get_value("Bu"), fix_b),
        }

    def to_dict(self):
        d = {
            "name": self.model_name,
            "enabled": self.isChecked(),
            "color": self._color,
            "dark_color": self._dark_color,
            "linestyle": self._linestyle_combo.currentText(),
            "alpha": self.alpha,
            "fix_a_ratio": self._fix_a_ratio.isChecked(),
            "a_ratio": self._a_ratio.value(),
            "fix_b_ratio": self._fix_b_ratio.isChecked(),
            "b_ratio": self._b_ratio.value(),
        }
        d.update(self.get_model_params())
        ranges = {}
        for key, p in self._slider_params.items():
            ranges[key] = [p["_lo"], p["_hi"]]
        d["slider_ranges"] = ranges
        # Peak amplitude overrides
        d["peaks_enabled"] = self._peaks_toggle.isChecked()
        d["peak_labels_enabled"] = self._peak_labels_toggle.isChecked()
        peak_cfg = []
        for row in self._peak_rows:
            peak_cfg.append({
                "label": row["label"],
                "mode": row["mode"].currentText(),
                "value": row["value"].value(),
                "linked_to": row["link_combo"].currentText(),
                "ratio": row["ratio"].value(),
            })
        d["peaks"] = peak_cfg
        return d

    def from_dict(self, d):
        self._building = True
        name = d.get("name", self.title())
        self.setTitle(name)
        self._name_edit.setText(name)
        self.setChecked(d.get("enabled", True))
        self._color = d.get("color", self._color)
        self._dark_color = d.get("dark_color", self._dark_color)
        self._update_color_icon()

        ls_text = d.get("linestyle", "Solid")
        idx = self._linestyle_combo.findText(ls_text)
        if idx >= 0:
            self._linestyle_combo.setCurrentIndex(idx)
        self._alpha_slider.setValue(int(d.get("alpha", 1.0) * 100))

        self._fix_a_ratio.setChecked(d.get("fix_a_ratio", False))
        self._a_ratio.setValue(d.get("a_ratio", 1.0))
        self._fix_b_ratio.setChecked(d.get("fix_b_ratio", False))
        self._b_ratio.setValue(d.get("b_ratio", 1.0))

        self.spin_I.setValue(float(d.get("I", 3.5)))
        self.spin_Jl.setValue(float(d.get("Jl", 4.5)))
        self.spin_Ju.setValue(float(d.get("Ju", 5.5)))

        ranges = d.get("slider_ranges", {})
        for key, p in self._slider_params.items():
            if key in ranges:
                p["_lo"] = ranges[key][0]
                p["_hi"] = ranges[key][1]

        for key in self._slider_params:
            if key in d:
                self.set_value(key, float(d[key]))

        self._building = False
        self._building = False

        # Restore peak amplitude settings
        self._peaks_toggle.setChecked(d.get("peaks_enabled", False))
        self._peak_labels_toggle.setChecked(
            d.get("peak_labels_enabled", False))
        if d.get("peaks_enabled", False):
            self._rebuild_peak_rows()
            peak_cfg = d.get("peaks", [])
            cfg_by_label = {p["label"]: p for p in peak_cfg}
            for row in self._peak_rows:
                cfg = cfg_by_label.get(row["label"])
                if cfg:
                    idx = row["mode"].findText(cfg.get("mode", "Racah"))
                    if idx >= 0:
                        row["mode"].setCurrentIndex(idx)
                    row["value"].setValue(cfg.get("value", row["racah_val"]))
                    link_idx = row["link_combo"].findText(
                        cfg.get("linked_to", ""))
                    if link_idx >= 0:
                        row["link_combo"].setCurrentIndex(link_idx)
                    row["ratio"].setValue(cfg.get("ratio", 1.0))


# ══════════════════════════════════════════════════════════════════
#  Undo commands
# ══════════════════════════════════════════════════════════════════
class _FileRemoveCommand(AppUndoCommand):
    """Undoable removal of a Pre-Analysis file entry. Undo restores the
    entry (kept alive, not deleted) at its original position and reveals
    it (switching to this Pre-Analysis sub-tab + flashing the row)."""

    def __init__(self, tab, entry):
        label = getattr(entry, "run_number", None) or "file"
        super().__init__(f"Remove {label}")
        self._tab = tab
        self._entry = entry
        self._index = tab._file_entries.index(entry)

    def _apply_redo(self, initial):
        if not initial:
            # Re-doing a removal: surface the tab first so the user sees
            # which file disappears.
            self.reveal(self._tab)
        self._tab._do_soft_remove_entry(self._entry)

    def _apply_undo(self):
        self._tab._do_reinsert_entry(self._entry, self._index)
        self.reveal(self._entry)


class _ModelRemoveCommand(AppUndoCommand):
    """Undoable removal of one or more checked HFS model panels."""

    def __init__(self, tab, items):
        n = len(items)
        super().__init__(f"Remove {n} model" + ("s" if n != 1 else ""))
        self._tab = tab
        self._items = list(items)          # [(panel, index), ...]
        self._panels = [p for p, _ in items]

    def _apply_redo(self, initial):
        if not initial:
            self.reveal(self._tab)
        self._tab._do_soft_remove_models(self._panels)

    def _apply_undo(self):
        self._tab._do_reinsert_models(self._items)
        if self._panels:
            self.reveal(self._panels[0])


# ══════════════════════════════════════════════════════════════════
#  Pre-Analysis Tab
# ══════════════════════════════════════════════════════════════════
class PreAnalysisTab(QWidget):
    """Pre-Analysis tab for data viewing and HFS model overlay."""

    # User picked a plot layout — the container mirrors it onto every
    # open Pre-Analysis project (session-global setting).
    layout_mode_changed = Signal(str)
    # User toggled dark plots — also session-global.
    dark_mode_changed = Signal(bool)
    # User toggled the x / y grid — session-global (x_on, y_on).
    grid_changed = Signal(bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_entries = []
        self._model_panels = []
        self._model_counter = 0
        self._plot_editor = None
        self._tof_span = None
        self._ts_span = None
        # Dark-mode plots: default from the global settings file so the
        # choice persists across sessions (a loaded save may override).
        # Set BEFORE the UI is built so the toggle button and first draw
        # reflect it.
        try:
            self._dark_mode = bool(
                _load_settings().get("pa_dark_plots", False))
        except Exception:
            self._dark_mode = False
        # Spectrum-tab grid toggles (x / y independently), session-global.
        self._grid_x = False
        self._grid_y = False

        # Debounce timer for replot
        self._replot_timer = QTimer()
        self._replot_timer.setSingleShot(True)
        self._replot_timer.setInterval(30)
        self._replot_timer.timeout.connect(self._on_replot_timer)
        # Whether the next (coalesced) debounced replot is a gate-drag
        # "spectrum-only" fast path. The full path redraws all three
        # panels (spectrum + TOF + timestamp) plus the six calibration /
        # cooler diagnostic axes -- a full multi-figure matplotlib draw
        # that costs hundreds of ms. While the user drags a TOF / time
        # gate only the binned spectrum changes, so those drags schedule a
        # spectrum-only replot (see _schedule_replot_light). A pending
        # full replot always wins over a light one in the same debounce
        # window. First draw is full.
        self._pending_spectrum_only = False

        # ── Gate-drag fast path ──────────────────────────────────────
        # To keep gate dragging instant, a full replot caches the
        # per-event native coordinate (raw DV for Voltage modes, lab-frame
        # MHz for Frequency) + the bin edges + the spectrum Line2D, after
        # proving (by a self-check) that an in-memory numpy histogram
        # reproduces clstools' compute_binned EXACTLY for the current gate.
        # Then each gate move just re-histograms (~0.1 ms) and blits the
        # single line (~1 ms) instead of re-binning through dask +
        # clearing/redrawing the whole figure (~185 ms). ``_fast`` is the
        # cache (or None when the current view isn't eligible -> safe
        # fallback to the dask spectrum-only path); ``_fast_bg`` is the
        # captured blit background (everything except the spectrum line).
        self._fast = None
        self._fast_bg = None

        # Any per-file scan exclusion change (here or in the Analysis
        # tab) should re-trigger the spectrum binning. Connecting at
        # tab construction is cheap and survives project saves/loads;
        # the signal fires from MainWindow's Load All flow too.
        from gui.scan_filter import get_registry as _get_sf_registry
        _get_sf_registry().filters_changed.connect(
            self._schedule_replot)

        # A calibration change (here, in the Analysis tab, or from a project
        # load) needs more than a replot: it rewrites every event's voltage,
        # so the cached CLSDataFrame and np_v have to be rebuilt, not just
        # re-binned.
        from gui.calibration import get_registry as _get_cal_registry
        _get_cal_registry().calibrations_changed.connect(
            self._on_calibrations_changed)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Main 3-panel horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter = splitter

        # ── Left panel: file list + plot options ─────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        left_layout.addWidget(QLabel("<b>Data Files</b>"))

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(4, 0, 4, 0)
        btn_row.setSpacing(4)
        open_btn = QPushButton("Open...")
        open_btn.setToolTip("Open ASDF data file(s)")
        open_btn.clicked.connect(self._open_files)
        btn_row.addWidget(open_btn)
        remove_btn = QPushButton("Remove checked")
        remove_btn.setToolTip("Remove the checked files from the list")
        remove_btn.clicked.connect(self._remove_checked)
        btn_row.addWidget(remove_btn)
        merge_btn = QPushButton("Merge Checked")
        merge_btn.setToolTip(
            "Merge checked files into a single spectrum entry.\n"
            "Merge in voltage (lab frame) or Doppler-shifted frequency "
            "(rest frame).")
        merge_btn.clicked.connect(self._merge_checked)
        btn_row.addWidget(merge_btn)
        left_layout.addLayout(btn_row)

        # Master "check all" tickbox: tri-state (Checked / Unchecked /
        # PartiallyChecked) reflecting the per-file checkboxes. Clicking
        # it forces every entry to checked-or-unchecked depending on the
        # current aggregate state, so a campaign with dozens of runs
        # doesn't need a click per row.
        master_row = QHBoxLayout()
        master_row.setContentsMargins(4, 0, 4, 0)
        master_row.setSpacing(4)
        self._master_check = QCheckBox("Check all")
        self._master_check.setTristate(True)
        self._master_check.setEnabled(False)  # no files yet
        self._master_check.setToolTip(
            "Check or uncheck every file at once.\n"
            "Mixed state (square) appears when some files are checked "
            "and others aren't.")
        self._master_check.clicked.connect(self._on_master_check_clicked)
        master_row.addWidget(self._master_check)
        master_row.addStretch()
        left_layout.addLayout(master_row)

        # File list scroll area
        self._file_scroll = QScrollArea()
        self._file_scroll.setWidgetResizable(True)
        self._file_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._file_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._file_container = QWidget()
        self._file_list_layout = QVBoxLayout(self._file_container)
        self._file_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._file_list_layout.setContentsMargins(0, 0, 0, 0)
        self._file_list_layout.setSpacing(2)
        self._file_scroll.setWidget(self._file_container)
        # Compact: ~5 entries visible, scroll for more. The list used to
        # take all the column's stretch, which is why it ran most of the
        # window height with a single file loaded.
        self._file_scroll.setMinimumHeight(96)
        self._file_scroll.setMaximumHeight(240)
        left_layout.addWidget(self._file_scroll)

        # -- Plot Options section --
        opts = QGroupBox("Plot Options")
        opts_form = QFormLayout(opts)
        opts_form.setContentsMargins(4, 8, 4, 4)
        opts_form.setHorizontalSpacing(8)
        opts_form.setVerticalSpacing(3)
        opts_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        opts_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._xaxis_combo = QComboBox()
        self._xaxis_combo.addItems(XAXIS_MODES)
        self._xaxis_combo.setCurrentText("Voltage")
        self._xaxis_combo.currentIndexChanged.connect(self._schedule_replot)
        opts_form.addRow("X-axis:", self._xaxis_combo)


        self._e_lower = _make_double(0, -1e6, 1e6, 4, 0.001,
                                      tooltip="Lower energy level [cm\u207b\u00b9]")
        self._e_lower.valueChanged.connect(self._update_transition_labels)
        self._e_lower.valueChanged.connect(self._schedule_replot)
        opts_form.addRow("E lower (cm\u207b\u00b9):", self._e_lower)

        self._e_upper = _make_double(0, -1e6, 1e6, 4, 0.001,
                                      tooltip="Upper energy level [cm\u207b\u00b9]")
        self._e_upper.valueChanged.connect(self._update_transition_labels)
        self._e_upper.valueChanged.connect(self._schedule_replot)
        opts_form.addRow("E upper (cm\u207b\u00b9):", self._e_upper)

        self._transition_label = QLabel("0.0000 cm\u207b\u00b9")
        self._transition_label.setStyleSheet(
            "padding: 2px 4px; color: #aaa;")
        self._transition_label.setToolTip(
            "Rest-frame transition = E_upper \u2212 E_lower")
        opts_form.addRow("Transition:", self._transition_label)

        self._harmonic = _make_int(2, 1, 10, tooltip="Laser harmonic")
        self._harmonic.valueChanged.connect(self._update_transition_labels)
        self._harmonic.valueChanged.connect(self._schedule_replot)
        opts_form.addRow("Harmonic:", self._harmonic)

        self._fundamental_label = QLabel("0.0000 cm\u207b\u00b9")
        self._fundamental_label.setStyleSheet(
            "padding: 2px 4px; color: #aaa;")
        self._fundamental_label.setToolTip(
            "Fundamental = Transition / harmonic\n"
            "(laser setpoint wavenumber)")
        opts_form.addRow("Fundamental:", self._fundamental_label)

        # Z and A
        za_widget = QWidget()
        za_layout = QHBoxLayout(za_widget)
        za_layout.setContentsMargins(0, 0, 0, 0)
        za_layout.setSpacing(6)
        self._z_spin = _make_int(1, 1, 118, tooltip="Atomic number")
        self._z_spin.setMinimumWidth(55)
        self._z_spin.valueChanged.connect(self._on_z_changed)
        za_layout.addWidget(self._z_spin)
        za_layout.addWidget(QLabel("A:"))
        self._a_spin = _make_int(1, 1, 300, tooltip="Mass number")
        self._a_spin.setMinimumWidth(55)
        self._a_spin.valueChanged.connect(self._on_a_changed)
        za_layout.addWidget(self._a_spin)
        self._isotope_label = QLabel("")
        za_layout.addWidget(self._isotope_label)
        za_layout.addStretch()
        opts_form.addRow("Z:", za_widget)
        self._update_isotope_label()

        # Mass row: looked-up value + override
        mass_widget = QWidget()
        mass_layout = QHBoxLayout(mass_widget)
        mass_layout.setContentsMargins(0, 0, 0, 0)
        mass_layout.setSpacing(6)
        self._mass_spin = _make_double(
            1.0, 0.001, 400.0, 6, 0.000001,
            tooltip="Atomic mass in amu used for the Doppler shift.\n"
                    "Auto-filled from the periodictable database when "
                    "Z and A change. Tick 'Override' to use a custom "
                    "value (e.g. AME2020 mass not in the database).")
        self._mass_spin.setMinimumWidth(110)
        self._mass_spin.setReadOnly(True)
        self._mass_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._mass_override = QCheckBox("Override")
        self._mass_override.setToolTip(
            "Use the value typed above instead of the looked-up mass.")
        self._mass_override.toggled.connect(self._on_mass_override_toggled)
        self._mass_spin.valueChanged.connect(self._schedule_replot)
        mass_layout.addWidget(self._mass_spin)
        mass_layout.addWidget(self._mass_override)
        mass_layout.addStretch()
        opts_form.addRow("Mass [amu]:", mass_widget)
        self._refresh_mass_display()

        # Channels
        ch_widget = QWidget()
        ch_layout = QHBoxLayout(ch_widget)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        ch_layout.setSpacing(4)
        self._channels = []
        for i in range(1, 6):
            label = "DC" if i == 5 else str(i)
            cb = QCheckBox(label)
            cb.setChecked(i in (3, 4))
            cb.toggled.connect(self._schedule_replot)
            ch_layout.addWidget(cb)
            self._channels.append(cb)
        ch_layout.addStretch()
        opts_form.addRow("Channels:", ch_widget)

        self._normalize = QCheckBox("Normalize")
        self._normalize.toggled.connect(self._schedule_replot)
        opts_form.addRow("", self._normalize)

        # Cooler voltage (with per-parameter override tick)
        cooler_widget = QWidget()
        cooler_layout = QHBoxLayout(cooler_widget)
        cooler_layout.setContentsMargins(0, 0, 0, 0)
        cooler_layout.setSpacing(4)
        self._cooler_override_enabled = QCheckBox()
        self._cooler_override_enabled.setToolTip(
            "Override cooler voltage for all files.\n"
            "When unticked, each file uses its own value.")
        cooler_layout.addWidget(self._cooler_override_enabled)
        self._cooler_override = _make_double(
            29977, 0, 100000, 2, 1.0,
            tooltip="Cooler voltage for Doppler conversion")
        self._cooler_override.setMinimumWidth(100)
        self._cooler_override.valueChanged.connect(self._schedule_replot)
        cooler_layout.addWidget(self._cooler_override)
        self._cooler_auto_btn = QPushButton("\u2193")
        self._cooler_auto_btn.setFixedWidth(24)
        self._cooler_auto_btn.setToolTip("Auto-fill from selected file")
        self._cooler_auto_btn.clicked.connect(self._auto_fill_cooler)
        cooler_layout.addWidget(self._cooler_auto_btn)
        opts_form.addRow("Cooler (V):", cooler_widget)

        # Laser setpoint (with per-parameter override tick)
        laser_widget = QWidget()
        laser_layout = QHBoxLayout(laser_widget)
        laser_layout.setContentsMargins(0, 0, 0, 0)
        laser_layout.setSpacing(4)
        self._laser_override_enabled = QCheckBox()
        self._laser_override_enabled.setToolTip(
            "Override laser setpoint for all files.\n"
            "When unticked, each file uses its own value.")
        laser_layout.addWidget(self._laser_override_enabled)
        self._laser_override = _make_double(
            10920, 0, 100000, 6, 0.1,
            tooltip="Laser setpoint for Doppler conversion")
        self._laser_override.setMinimumWidth(100)
        self._laser_override.valueChanged.connect(self._schedule_replot)
        laser_layout.addWidget(self._laser_override)
        self._laser_auto_btn = QPushButton("\u2193")
        self._laser_auto_btn.setFixedWidth(24)
        self._laser_auto_btn.setToolTip("Auto-fill from selected file")
        self._laser_auto_btn.clicked.connect(self._auto_fill_laser)
        laser_layout.addWidget(self._laser_auto_btn)
        opts_form.addRow("Laser (cm\u207b\u00b9):", laser_widget)

        # Wire per-parameter override toggles now that widgets exist
        self._cooler_override_enabled.toggled.connect(
            self._on_cooler_override_toggled)
        self._laser_override_enabled.toggled.connect(
            self._on_laser_override_toggled)
        self._on_cooler_override_toggled(False)
        self._on_laser_override_toggled(False)

        # The binning controls are intentionally NOT shown in the
        # Pre-Analysis sidebar: the live preview always uses the default
        # Raw-Voltage / Auto / no-yerr binning (one bin per raw DV step).
        # The widget objects are still built (kept hidden) so config
        # save/load and _binning_cfg keep working unchanged.
        self._binning_group = self._build_binning_group()
        self._binning_group.setVisible(False)
        opts_form.addRow(self._binning_group)

        edit_btn = QPushButton("Edit Plot")
        edit_btn.clicked.connect(self._open_plot_editor)
        opts_form.addRow("", edit_btn)

        left_layout.addWidget(opts)

        # Width bounds sized so the Data Files action buttons ("Open...",
        # "Remove checked", "Merge Checked") aren't clipped, with a slightly
        # wider default for the side panel.
        # Slightly wider band than before: the column now also hosts the
        # HFS Models panel (moved from the deleted right column).
        left.setMinimumWidth(380)
        left.setMaximumWidth(540)
        splitter.addWidget(left)

        # ── Center panel: tabbed plots ──────────────────────────────
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._center_tabs = QTabWidget()

        # Plot-layout switcher, top-right on the sub-tab row (same
        # height as Spectrum / Calibrations / Cooler Voltage). The
        # setting is SESSION-GLOBAL: the container mirrors a change
        # onto every open Pre-Analysis project.
        corner = QWidget()
        corner_lay = QHBoxLayout(corner)
        corner_lay.setContentsMargins(0, 0, 6, 2)
        corner_lay.setSpacing(4)
        corner_lay.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["3 row stacked", "2 row stacked"])
        self._layout_combo.setToolTip(
            "Arrangement of the three plots — applies to every open "
            "Pre-Analysis project:\n"
            "3 row stacked — ToF, Spectrum and Timestamp as rows.\n"
            "2 row stacked — Spectrum and ToF side by side with the "
            "Timestamp strip below (original Data Viewer layout).")
        self._layout_combo.currentIndexChanged.connect(
            self._on_layout_combo_changed)
        corner_lay.addWidget(self._layout_combo)

        # X / Y grid toggles for the spectrum-tab plots (spectrum, ToF,
        # timestamp). Grid colour follows the light/dark theme.
        corner_lay.addSpacing(6)
        corner_lay.addWidget(QLabel("Grid:"))
        self._grid_x_btn = QToolButton()
        self._grid_x_btn.setText("X")
        self._grid_x_btn.setCheckable(True)
        self._grid_x_btn.setAutoRaise(True)
        self._grid_x_btn.setFixedSize(22, 22)
        self._grid_x_btn.setToolTip(
            "Toggle the vertical (x) grid lines on every spectrum-tab "
            "plot")
        self._grid_x_btn.toggled.connect(
            lambda on: self._on_grid_toggle("x", on))
        corner_lay.addWidget(self._grid_x_btn)
        self._grid_y_btn = QToolButton()
        self._grid_y_btn.setText("Y")
        self._grid_y_btn.setCheckable(True)
        self._grid_y_btn.setAutoRaise(True)
        self._grid_y_btn.setFixedSize(22, 22)
        self._grid_y_btn.setToolTip(
            "Toggle the horizontal (y) grid lines on every "
            "spectrum-tab plot")
        self._grid_y_btn.toggled.connect(
            lambda on: self._on_grid_toggle("y", on))
        corner_lay.addWidget(self._grid_y_btn)
        corner_lay.addSpacing(6)

        # Dark-mode toggle: sun/crescent-moon glyph that flips the plot
        # canvases to a black background with white chrome and neon
        # line colors (and back). Session-global like the layout.
        self._dark_btn = QToolButton()
        self._dark_btn.setIcon(lucide_icon("sun-moon"))
        self._dark_btn.setCheckable(True)
        self._dark_btn.setAutoRaise(True)
        self._dark_btn.setFixedSize(24, 22)
        self._dark_btn.setToolTip(
            "Dark plots: OFF — click for a black canvas with neon lines")
        self._dark_btn.toggled.connect(self._on_dark_toggle)
        corner_lay.addWidget(self._dark_btn)
        # Reflect the settings-derived default without emitting toggled
        # (the theme applies on the first draw via _finalize_axes).
        if self._dark_mode:
            self._dark_btn.blockSignals(True)
            self._dark_btn.setChecked(True)
            self._dark_btn.blockSignals(False)
            self._update_dark_btn_tooltip()

        self._center_tabs.setCornerWidget(corner,
                                          Qt.Corner.TopRightCorner)

        # ── Spectrum tab ────────────────────────────────────────────
        spectrum_tab = QWidget()
        spectrum_tab_layout = QVBoxLayout(spectrum_tab)
        spectrum_tab_layout.setContentsMargins(0, 0, 0, 0)
        spectrum_tab_layout.setSpacing(0)

        # Info button row
        spec_header = QHBoxLayout()
        spec_header.setContentsMargins(4, 2, 4, 0)
        spec_info_btn = QPushButton(lucide_icon("circle-help"), "Info")
        spec_info_btn.setFixedWidth(75)
        spec_info_btn.clicked.connect(self._show_spectrum_info)
        spec_header.addWidget(spec_info_btn)
        spec_header.addStretch()
        spectrum_tab_layout.addLayout(spec_header)

        plot_splitter = QSplitter(Qt.Orientation.Vertical)

        # -- TOF plot (top, compact) --
        tof_widget = QWidget()
        tof_layout = QVBoxLayout(tof_widget)
        tof_layout.setContentsMargins(0, 0, 0, 0)
        tof_layout.setSpacing(0)

        self._tof_fig = Figure(dpi=100, constrained_layout=True)
        self._tof_ax = self._tof_fig.add_subplot(111)
        self._tof_ax.set_xlabel("Time (\u00b5s)", fontsize=8)
        self._tof_ax.set_ylabel("Counts", fontsize=8)
        self._tof_ax.tick_params(labelsize=7)
        self._tof_canvas = FigureCanvasQTAgg(self._tof_fig)
        self._tof_canvas._plot_editor_opener = self._open_plot_editor

        tof_header = QHBoxLayout()
        tof_header.setContentsMargins(4, 2, 4, 0)
        tof_header.addWidget(QLabel("<b>TOF</b>"))
        tof_header.addSpacing(8)
        self._tof_enable = QCheckBox("Gate")
        self._tof_enable.setToolTip("Enable TOF gating")
        self._tof_enable.toggled.connect(self._on_tof_enable_toggled)
        tof_header.addWidget(self._tof_enable)
        self._tof_lo = _make_double(35, 0, 10000, 1, 0.5,
                                    tooltip="TOF gate lower bound (\u00b5s)")
        self._tof_lo.setFixedWidth(80)
        tof_header.addWidget(self._tof_lo)
        tof_header.addWidget(QLabel("\u2013"))
        self._tof_hi = _make_double(61, 0, 10000, 1, 0.5,
                                    tooltip="TOF gate upper bound (\u00b5s)")
        self._tof_hi.setFixedWidth(80)
        tof_header.addWidget(self._tof_hi)
        tof_header.addSpacing(12)
        tof_header.addWidget(QLabel("Bin:"))
        # TOF histogram bin size \u2014 the old hard-coded 1 \u00b5s looks steppy
        # on narrow bunches; sub-\u00b5s values resolve the peak shape.
        self._tof_binsize = _make_double(
            1.0, 0.001, 1000, 3, 0.1,
            tooltip="TOF histogram bin size in \u00b5s (was fixed at 1 \u00b5s). "
                    "Smaller bins resolve the bunch shape; larger bins "
                    "smooth low statistics.")
        self._tof_binsize.setFixedWidth(80)
        self._tof_binsize.setSuffix(" \u00b5s")
        self._tof_binsize.valueChanged.connect(
            lambda *_: self._schedule_replot())
        tof_header.addWidget(self._tof_binsize)
        self._tof_lo.valueChanged.connect(self._on_tof_spinbox_changed)
        self._tof_hi.valueChanged.connect(self._on_tof_spinbox_changed)
        tof_header.addStretch()
        self._tof_toolbar = NavigationToolbar2QT(self._tof_canvas, tof_widget)
        self._tof_toolbar.setMaximumHeight(24)
        self._tof_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._tof_toolbar)
        tof_header.addWidget(self._tof_toolbar)
        tof_layout.addLayout(tof_header)
        tof_layout.addWidget(self._tof_canvas)
        plot_splitter.addWidget(tof_widget)

        # -- Spectrum plot (middle, dominant) --
        spec_widget = QWidget()
        spec_layout = QVBoxLayout(spec_widget)
        spec_layout.setContentsMargins(0, 0, 0, 0)
        spec_layout.setSpacing(0)

        self._fig = Figure(dpi=100, constrained_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("Frequency (MHz)")
        self._ax.set_ylabel("Counts")
        self._canvas = FigureCanvasQTAgg(self._fig)
        # Right-click editor shows Spectrum + TOF + Timestamp
        # together (the tab's own multi-figure wiring).
        self._canvas._plot_editor_opener = self._open_plot_editor
        self._toolbar = NavigationToolbar2QT(self._canvas, spec_widget)
        self._toolbar.setMaximumHeight(24)
        self._toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._toolbar)
        spec_plot_header = QHBoxLayout()
        spec_plot_header.setContentsMargins(4, 2, 4, 0)
        spec_plot_header.addWidget(QLabel("<b>Spectrum</b>"))
        spec_plot_header.addSpacing(12)
        spec_plot_header.addWidget(QLabel("Bin:"))
        # Bin size in integer multiples of the raw scanning-voltage step.
        # 1 = the file's native bins (default): raw DV steps in voltage
        # mode, and in the Doppler-shifted frequency view each step's own
        # mean rest-frame frequency. N groups N adjacent steps.
        self._spec_bin_mult = _NoScrollInt()
        self._spec_bin_mult.setRange(1, 1000)
        self._spec_bin_mult.setValue(1)
        self._spec_bin_mult.setSuffix(" × step")
        self._spec_bin_mult.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._spec_bin_mult.setToolTip(
            "Spectrum bin size as an integer multiple of the raw "
            "scanning-voltage step. 1 = the file's native voltage bins "
            "(or their Doppler-shifted frequency equivalents); larger "
            "values sum N adjacent steps into one bin.")
        self._spec_bin_mult.valueChanged.connect(
            lambda *_: self._schedule_replot())
        spec_plot_header.addWidget(self._spec_bin_mult)
        spec_plot_header.addStretch()
        spec_plot_header.addWidget(self._toolbar)
        spec_layout.addLayout(spec_plot_header)
        spec_layout.addWidget(self._canvas)
        plot_splitter.addWidget(spec_widget)

        # -- Timestamp plot (bottom, compact) --
        ts_widget = QWidget()
        ts_layout = QVBoxLayout(ts_widget)
        ts_layout.setContentsMargins(0, 0, 0, 0)
        ts_layout.setSpacing(0)

        self._ts_fig = Figure(dpi=100, constrained_layout=True)
        self._ts_ax = self._ts_fig.add_subplot(111)
        self._ts_ax.set_xlabel("Timestamp (s)", fontsize=8)
        self._ts_ax.set_ylabel("Events", fontsize=8)
        self._ts_ax.tick_params(labelsize=7)
        self._ts_canvas = FigureCanvasQTAgg(self._ts_fig)
        self._ts_canvas._plot_editor_opener = self._open_plot_editor

        ts_header = QHBoxLayout()
        # Symmetric vertical margins + explicit AlignVCenter on every
        # widget below so the row reads as a single horizontal line.
        # The previous (4, 2, 4, 0) shipped widgets at slightly
        # different vertical offsets because the row was taller than
        # the smallest control and the layout's default cross-axis
        # placement varied by widget size policy.
        ts_header.setContentsMargins(4, 4, 4, 4)
        ts_header.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        _vc = Qt.AlignmentFlag.AlignVCenter

        ts_header.addWidget(QLabel("<b>Timestamp</b>"), 0, _vc)
        ts_header.addSpacing(8)
        ts_header.addWidget(QLabel("Unit:"), 0, _vc)
        self._ts_unit = QComboBox()
        self._ts_unit.addItems(["Seconds", "Minutes", "Hours", "Days"])
        self._ts_unit.setFixedWidth(95)
        self._ts_prev_divisor = 1.0   # Seconds; tracks the unit before a change
        self._ts_unit.currentIndexChanged.connect(self._on_ts_unit_changed)
        ts_header.addWidget(self._ts_unit, 0, _vc)
        ts_header.addSpacing(8)
        ts_header.addWidget(QLabel("Bin:"), 0, _vc)
        self._ts_binsize = _make_double(1.0, 0.001, 100000, 3, 1.0,
                                        tooltip="Timestamp bin size (in selected unit)")
        self._ts_binsize.setFixedWidth(80)
        self._ts_binsize.valueChanged.connect(self._schedule_replot)
        ts_header.addWidget(self._ts_binsize, 0, _vc)
        ts_header.addSpacing(8)

        # Time gate: like the ToF gate but on the event timestamp. When
        # enabled it filters ONLY the spectrum (the ToF and timestamp
        # plots keep showing all events), so the user can watch how the
        # spectrum's counts evolve over a chosen time window. Bounds are
        # in the same (relative, selected-unit) coordinates the timestamp
        # x-axis shows; they're converted to absolute seconds at filter
        # time.
        self._ts_enable = QCheckBox("Gate")
        self._ts_enable.setToolTip(
            "Enable time gating: filter the spectrum to events whose\n"
            "timestamp falls in the window below. ToF/Timestamp plots\n"
            "still show all events.")
        self._ts_enable.toggled.connect(self._on_ts_enable_toggled)
        ts_header.addWidget(self._ts_enable, 0, _vc)
        self._ts_lo = _make_double(0.0, 0.0, 1e12, 3, 1.0,
                                   tooltip="Time gate lower bound (x-axis units)")
        self._ts_lo.setFixedWidth(90)
        ts_header.addWidget(self._ts_lo, 0, _vc)
        ts_header.addWidget(QLabel("–"), 0, _vc)
        self._ts_hi = _make_double(0.0, 0.0, 1e12, 3, 1.0,
                                   tooltip="Time gate upper bound (x-axis units)")
        self._ts_hi.setFixedWidth(90)
        ts_header.addWidget(self._ts_hi, 0, _vc)
        self._ts_lo.valueChanged.connect(self._on_ts_spinbox_changed)
        self._ts_hi.valueChanged.connect(self._on_ts_spinbox_changed)
        ts_header.addSpacing(8)

        # Per-scan overlay controls. Drawing scan boundaries on the
        # timestamp plot lets the user zoom in on a low-signal stretch
        # and exclude it in bulk without having to read off times by
        # hand and type them into the filter dialog.
        self._ts_show_scans = QCheckBox("Show scans")
        self._ts_show_scans.setToolTip(
            "Overlay scan boundaries on the timestamp plot. Each scan "
            "(one voltage sweep) gets a vertical tick; excluded scans "
            "are shaded red.\n"
            "Only applies when exactly one non-merged file is shown -- "
            "scan numbers are per-file and would collide otherwise.")
        self._ts_show_scans.toggled.connect(self._on_show_scans_toggled)
        ts_header.addWidget(self._ts_show_scans, 0, _vc)
        # Visible gap before the button — checkbox and button read as
        # one control when they touch.
        ts_header.addSpacing(10)

        self._ts_exclude_view_btn = QPushButton("Exclude scans in view")
        self._ts_exclude_view_btn.setToolTip(
            "Add every scan whose start time falls inside the current "
            "x-axis range to the file's exclusion set. Pan/zoom first, "
            "then click. Existing exclusions are preserved.")
        self._ts_exclude_view_btn.clicked.connect(
            self._exclude_scans_in_view)
        ts_header.addWidget(self._ts_exclude_view_btn, 0, _vc)

        ts_header.addStretch()
        self._ts_toolbar = NavigationToolbar2QT(self._ts_canvas, ts_widget)
        self._ts_toolbar.setMaximumHeight(24)
        self._ts_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._ts_toolbar)
        ts_header.addWidget(self._ts_toolbar, 0, _vc)
        ts_layout.addLayout(ts_header)
        ts_layout.addWidget(self._ts_canvas)
        plot_splitter.addWidget(ts_widget)

        # Default pane proportions: a tall TOF and dominant Spectrum,
        # with a compact Timestamp strip at the bottom (~ 3 : 4 : 1).
        plot_splitter.setStretchFactor(0, 3)   # TOF
        plot_splitter.setStretchFactor(1, 4)   # Spectrum
        plot_splitter.setStretchFactor(2, 3)   # Timestamp (taller default)

        # Handles for the layout switcher (Stacked vs Classic).
        self._plot_splitter = plot_splitter
        self._tof_widget = tof_widget
        self._spec_widget = spec_widget
        self._ts_widget = ts_widget
        self._top_hsplit = None
        self._plot_layout_mode = "stacked"

        # Scroll-wheel zoom on hover for every spectrum-tab canvas, so
        # the user can zoom without reaching for the toolbar buttons.
        for _c in (self._tof_canvas, self._canvas, self._ts_canvas):
            _c.mpl_connect("scroll_event", self._on_scroll_zoom)

        spectrum_tab_layout.addWidget(plot_splitter, 1)
        self._center_tabs.addTab(spectrum_tab, "Spectrum")

        # ── Calibrations tab ────────────────────────────────────────
        calib_tab = QWidget()
        calib_tab_layout = QVBoxLayout(calib_tab)
        calib_tab_layout.setContentsMargins(0, 0, 0, 0)
        calib_tab_layout.setSpacing(0)

        # Info button row
        calib_header = QHBoxLayout()
        calib_header.setContentsMargins(4, 2, 4, 0)
        calib_info_btn = QPushButton(lucide_icon("circle-help"), "Info")
        calib_info_btn.setFixedWidth(75)
        calib_info_btn.clicked.connect(self._show_calib_info)
        calib_header.addWidget(calib_info_btn)
        calib_header.addStretch()
        calib_tab_layout.addLayout(calib_header)

        calib_splitter = QSplitter(Qt.Orientation.Vertical)

        # -- Readback vs Set voltage --
        cal_rb_widget = QWidget()
        cal_rb_layout = QVBoxLayout(cal_rb_widget)
        cal_rb_layout.setContentsMargins(0, 0, 0, 0)
        cal_rb_layout.setSpacing(0)
        # constrained_layout instead of a fixed left margin: the old
        # left=0.08 band was mostly dead space next to short y-labels.
        self._cal_readback_fig = Figure(dpi=100, constrained_layout=True)
        self._cal_readback_ax = self._cal_readback_fig.add_subplot(111)
        self._cal_readback_ax.set_ylabel("Readback (V)", fontsize=8)
        self._cal_readback_ax.tick_params(labelsize=7)
        self._cal_readback_canvas = FigureCanvasQTAgg(self._cal_readback_fig)
        self._cal_readback_toolbar = NavigationToolbar2QT(
            self._cal_readback_canvas, cal_rb_widget)
        self._cal_readback_toolbar.setMaximumHeight(24)
        self._cal_readback_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._cal_readback_toolbar)
        cal_rb_hdr = QHBoxLayout()
        cal_rb_hdr.setContentsMargins(0, 0, 0, 0)
        cal_rb_hdr.addStretch()
        cal_rb_hdr.addWidget(self._cal_readback_toolbar)
        cal_rb_layout.addLayout(cal_rb_hdr)
        cal_rb_layout.addWidget(self._cal_readback_canvas)
        calib_splitter.addWidget(cal_rb_widget)

        # -- Voltage difference (Readback - Set) vs Set --
        cal_diff_widget = QWidget()
        cal_diff_layout = QVBoxLayout(cal_diff_widget)
        cal_diff_layout.setContentsMargins(0, 0, 0, 0)
        cal_diff_layout.setSpacing(0)
        self._cal_diff_fig = Figure(dpi=100, constrained_layout=True)
        self._cal_diff_ax = self._cal_diff_fig.add_subplot(111)
        self._cal_diff_ax.set_ylabel("Readback \u2212 Set (V)\n[offset/drift]", fontsize=8)
        self._cal_diff_ax.tick_params(labelsize=7)
        self._cal_diff_canvas = FigureCanvasQTAgg(self._cal_diff_fig)
        self._cal_diff_toolbar = NavigationToolbar2QT(
            self._cal_diff_canvas, cal_diff_widget)
        self._cal_diff_toolbar.setMaximumHeight(24)
        self._cal_diff_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._cal_diff_toolbar)
        cal_diff_hdr = QHBoxLayout()
        cal_diff_hdr.setContentsMargins(0, 0, 0, 0)
        cal_diff_hdr.addStretch()
        cal_diff_hdr.addWidget(self._cal_diff_toolbar)
        cal_diff_layout.addLayout(cal_diff_hdr)
        cal_diff_layout.addWidget(self._cal_diff_canvas)
        calib_splitter.addWidget(cal_diff_widget)

        # -- Step size: diff(Readback) vs Set --
        cal_step_widget = QWidget()
        cal_step_layout = QVBoxLayout(cal_step_widget)
        cal_step_layout.setContentsMargins(0, 0, 0, 0)
        cal_step_layout.setSpacing(0)
        self._cal_step_fig = Figure(dpi=100, constrained_layout=True)
        self._cal_step_ax = self._cal_step_fig.add_subplot(111)
        self._cal_step_ax.set_xlabel("Set voltage (V)", fontsize=8)
        self._cal_step_ax.set_ylabel("\u0394 Readback (V)\n[step uniformity]", fontsize=8)
        self._cal_step_ax.tick_params(labelsize=7)
        self._cal_step_canvas = FigureCanvasQTAgg(self._cal_step_fig)
        self._cal_step_toolbar = NavigationToolbar2QT(
            self._cal_step_canvas, cal_step_widget)
        self._cal_step_toolbar.setMaximumHeight(24)
        self._cal_step_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._cal_step_toolbar)
        cal_step_hdr = QHBoxLayout()
        cal_step_hdr.setContentsMargins(0, 0, 0, 0)
        cal_step_hdr.addStretch()
        cal_step_hdr.addWidget(self._cal_step_toolbar)
        cal_step_layout.addLayout(cal_step_hdr)
        cal_step_layout.addWidget(self._cal_step_canvas)
        calib_splitter.addWidget(cal_step_widget)

        calib_splitter.setStretchFactor(0, 1)
        calib_splitter.setStretchFactor(1, 1)
        calib_splitter.setStretchFactor(2, 1)

        calib_tab_layout.addWidget(calib_splitter, 1)
        self._center_tabs.addTab(calib_tab, "Calibrations")

        # ── Cooler Voltage tab ──────────────────────────────────────
        cooler_tab = QWidget()
        cooler_tab_layout = QVBoxLayout(cooler_tab)
        cooler_tab_layout.setContentsMargins(0, 0, 0, 0)
        cooler_tab_layout.setSpacing(0)

        # Header row: Info button + clip-y toggle + bin-count spinbox.
        cooler_header = QHBoxLayout()
        cooler_header.setContentsMargins(4, 2, 4, 0)
        cooler_info_btn = QPushButton(lucide_icon("circle-help"), "Info")
        cooler_info_btn.setFixedWidth(75)
        cooler_info_btn.clicked.connect(self._show_cooler_info)
        cooler_header.addWidget(cooler_info_btn)
        cooler_header.addStretch()
        self._cooler_clip_y = QCheckBox("Clip y to ±4σ")
        self._cooler_clip_y.setChecked(True)
        self._cooler_clip_y.setToolTip(
            "Clip the deviation-pane y-axis to ±4·σ (the robust sigma, "
            "1.4826·MAD — immune to spikes) so normal "
            "ripple stays readable when there are big spikes. The full "
            "extremes are still reported in the status strip and "
            "legend; uncheck to include them in the y-scale.")
        self._cooler_clip_y.toggled.connect(self._replot_calibrations)
        cooler_header.addWidget(self._cooler_clip_y)
        cooler_header.addSpacing(12)
        cooler_header.addWidget(QLabel("Bins:"))
        self._cooler_bins = QSpinBox()
        self._cooler_bins.setRange(10, 5000)
        self._cooler_bins.setValue(100)
        self._cooler_bins.setSingleStep(10)
        self._cooler_bins.setFixedWidth(80)
        self._cooler_bins.setToolTip(
            "Number of equal-width time bins used by the deviation and "
            "ripple-strength panes (per-bin median, RMS, and P95−P5).")
        self._cooler_bins.valueChanged.connect(self._replot_calibrations)
        cooler_header.addWidget(self._cooler_bins)
        cooler_tab_layout.addLayout(cooler_header)

        # Per-run summary table: V_ref, robust σ / p–p, spike count,
        # signed extremes, and σ expressed as its approximate Doppler
        # impact in MHz — sortable, so run health can be judged (and
        # ranked) at a glance. Replaces the old monospace status strip.
        from PySide6.QtWidgets import QTableWidget, QHeaderView
        self._cooler_table = QTableWidget(0, 7)
        self._cooler_table.setHorizontalHeaderLabels(
            ["Run", "V_ref [V]", "σ [V]", "p–p [V]", "spikes",
             "max −/+ [V]", "σ [MHz]"])
        self._cooler_table.verticalHeader().setVisible(False)
        self._cooler_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._cooler_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._cooler_table.setSortingEnabled(True)
        self._cooler_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._cooler_table.setMaximumHeight(122)
        self._cooler_table.setToolTip(
            "One row per checked run. σ and p–p are robust "
            "(1.4826·MAD and P95−P5); spikes = samples beyond 3σ from "
            "V_ref; σ [MHz] is the cooler ripple converted to its "
            "approximate Doppler-shift impact at the current beam "
            "energy — the physically meaningful scale.")
        self._cooler_table.setVisible(False)
        cooler_tab_layout.addWidget(self._cooler_table)

        # Three stacked panes: raw V (top), deviation from V_ref (middle),
        # ripple-strength metrics over time (bottom). V_ref is a robust
        # run-average (median), so spikes don't bias the baseline.
        cooler_splitter = QSplitter(Qt.Orientation.Vertical)

        # -- Top pane: raw cooler voltage --
        cool_line_widget = QWidget()
        cool_line_layout = QVBoxLayout(cool_line_widget)
        cool_line_layout.setContentsMargins(0, 0, 0, 0)
        cool_line_layout.setSpacing(0)
        self._cal_cooler_fig = Figure(dpi=100)
        self._cal_cooler_fig.subplots_adjust(
            left=0.055, right=0.995, top=0.985, bottom=0.10)
        self._cal_cooler_ax = self._cal_cooler_fig.add_subplot(111)
        self._cal_cooler_ax.set_xlabel(
            "Time since run start (s)", fontsize=8, labelpad=2)
        self._cal_cooler_ax.set_ylabel(
            "Cooler V (V)  [raw]", fontsize=8, labelpad=2)
        self._cal_cooler_ax.tick_params(labelsize=7)
        self._cal_cooler_ax.ticklabel_format(
            useOffset=False, style='plain', axis='y')
        self._cal_cooler_ax.minorticks_on()
        self._cal_cooler_ax.grid(
            True, which="major", linestyle="-",
            linewidth=0.5, color="#bbbbbb", alpha=0.7)
        self._cal_cooler_ax.grid(
            True, which="minor", linestyle=":",
            linewidth=0.4, color="#cccccc", alpha=0.5)
        self._cal_cooler_canvas = FigureCanvasQTAgg(self._cal_cooler_fig)
        self._cal_cooler_toolbar = NavigationToolbar2QT(
            self._cal_cooler_canvas, cool_line_widget)
        self._cal_cooler_toolbar.setMaximumHeight(24)
        self._cal_cooler_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._cal_cooler_toolbar)
        cool_line_tb = QHBoxLayout()
        cool_line_tb.setContentsMargins(0, 0, 0, 0)
        cool_line_tb.addStretch()
        cool_line_tb.addWidget(self._cal_cooler_toolbar)
        cool_line_layout.addLayout(cool_line_tb)
        cool_line_layout.addWidget(self._cal_cooler_canvas, 1)
        cooler_splitter.addWidget(cool_line_widget)

        # -- Middle pane: deviation from run-average V_ref. (Variable
        #    names start with `_cool_ohlc_` for historical reasons; this
        #    is the deviation/drift diagnostic pane now.) --
        cool_ohlc_widget = QWidget()
        cool_ohlc_layout = QVBoxLayout(cool_ohlc_widget)
        cool_ohlc_layout.setContentsMargins(0, 0, 0, 0)
        cool_ohlc_layout.setSpacing(0)
        self._cool_ohlc_fig = Figure(dpi=100)
        self._cool_ohlc_fig.subplots_adjust(
            left=0.055, right=0.995, top=0.985, bottom=0.10)
        self._cool_ohlc_ax = self._cool_ohlc_fig.add_subplot(111)
        self._cool_ohlc_ax.set_xlabel(
            "Time since run start (s)", fontsize=8, labelpad=2)
        self._cool_ohlc_ax.set_ylabel(
            "Deviation from run avg (V)", fontsize=8, labelpad=2)
        self._cool_ohlc_ax.tick_params(labelsize=7)
        self._cool_ohlc_ax.ticklabel_format(
            useOffset=False, style='plain', axis='y')
        self._cool_ohlc_ax.minorticks_on()
        self._cool_ohlc_ax.grid(
            True, which="major", linestyle="-",
            linewidth=0.5, color="#bbbbbb", alpha=0.7)
        self._cool_ohlc_ax.grid(
            True, which="minor", linestyle=":",
            linewidth=0.4, color="#cccccc", alpha=0.5)
        self._cool_ohlc_canvas = FigureCanvasQTAgg(self._cool_ohlc_fig)
        self._cool_ohlc_toolbar = NavigationToolbar2QT(
            self._cool_ohlc_canvas, cool_ohlc_widget)
        self._cool_ohlc_toolbar.setMaximumHeight(24)
        self._cool_ohlc_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._cool_ohlc_toolbar)
        cool_ohlc_tb = QHBoxLayout()
        cool_ohlc_tb.setContentsMargins(0, 0, 0, 0)
        cool_ohlc_tb.addStretch()
        cool_ohlc_tb.addWidget(self._cool_ohlc_toolbar)
        cool_ohlc_layout.addLayout(cool_ohlc_tb)
        cool_ohlc_layout.addWidget(self._cool_ohlc_canvas, 1)
        cooler_splitter.addWidget(cool_ohlc_widget)

        # -- Bottom pane: ripple strength over time
        #    (per-bin RMS and robust peak-to-peak P95−P5). --
        cool_ripple_widget = QWidget()
        cool_ripple_layout = QVBoxLayout(cool_ripple_widget)
        cool_ripple_layout.setContentsMargins(0, 0, 0, 0)
        cool_ripple_layout.setSpacing(0)
        self._cool_ripple_fig = Figure(dpi=100)
        self._cool_ripple_fig.subplots_adjust(
            left=0.055, right=0.995, top=0.985, bottom=0.10)
        self._cool_ripple_ax = self._cool_ripple_fig.add_subplot(111)
        self._cool_ripple_ax.set_xlabel(
            "Time since run start (s)", fontsize=8, labelpad=2)
        self._cool_ripple_ax.set_ylabel(
            "Ripple (V)", fontsize=8, labelpad=2)
        self._cool_ripple_ax.tick_params(labelsize=7)
        self._cool_ripple_ax.ticklabel_format(
            useOffset=False, style='plain', axis='y')
        self._cool_ripple_ax.minorticks_on()
        self._cool_ripple_ax.grid(
            True, which="major", linestyle="-",
            linewidth=0.5, color="#bbbbbb", alpha=0.7)
        self._cool_ripple_ax.grid(
            True, which="minor", linestyle=":",
            linewidth=0.4, color="#cccccc", alpha=0.5)
        self._cool_ripple_canvas = FigureCanvasQTAgg(self._cool_ripple_fig)
        self._cool_ripple_toolbar = NavigationToolbar2QT(
            self._cool_ripple_canvas, cool_ripple_widget)
        self._cool_ripple_toolbar.setMaximumHeight(24)
        self._cool_ripple_toolbar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _stabilize_toolbar(self._cool_ripple_toolbar)
        cool_ripple_tb = QHBoxLayout()
        cool_ripple_tb.setContentsMargins(0, 0, 0, 0)
        cool_ripple_tb.addStretch()
        cool_ripple_tb.addWidget(self._cool_ripple_toolbar)
        cool_ripple_layout.addLayout(cool_ripple_tb)
        cool_ripple_layout.addWidget(self._cool_ripple_canvas, 1)
        cooler_splitter.addWidget(cool_ripple_widget)

        cooler_splitter.setStretchFactor(0, 2)
        cooler_splitter.setStretchFactor(1, 2)
        cooler_splitter.setStretchFactor(2, 1)

        cooler_tab_layout.addWidget(cooler_splitter, 1)
        self._center_tabs.addTab(cooler_tab, "Cooler Voltage")

        center_layout.addWidget(self._center_tabs, 1)
        splitter.addWidget(center)

        # ── HFS estimator — stacked under Plot Options in the LEFT
        # column (the old dedicated right column is gone, freeing its
        # ~500 px for the plots) ──────────────────────────────────
        left_layout.addWidget(QLabel("<b>HFS Models</b>"))

        # Two rows so the longer "Duplicate checked" / "Remove checked"
        # labels never clip.
        add_model_btn = QPushButton("+ Add Model")
        add_model_btn.clicked.connect(self._add_model)
        left_layout.addWidget(add_model_btn)

        model_btn_row = QHBoxLayout()
        dup_model_btn = QPushButton("Duplicate checked")
        dup_model_btn.setToolTip(
            "Duplicate the checked (ticked) model(s) in this tab.\n"
            "Tick a model's checkbox to mark it, then duplicate it.")
        dup_model_btn.clicked.connect(self._duplicate_checked_models)
        model_btn_row.addWidget(dup_model_btn)
        remove_model_btn = QPushButton("Remove checked")
        remove_model_btn.setToolTip(
            "Remove the checked (ticked) model(s) in this tab.")
        remove_model_btn.clicked.connect(self._remove_checked_models)
        model_btn_row.addWidget(remove_model_btn)
        left_layout.addLayout(model_btn_row)

        # Model panels scroll area — takes the column's remaining
        # stretch (its own scrollbar absorbs overflow, so the left
        # column itself never needs an outer scroll).
        self._model_scroll = QScrollArea()
        self._model_scroll.setWidgetResizable(True)
        self._model_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._model_container = QWidget()
        self._model_list_layout = QVBoxLayout(self._model_container)
        self._model_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._model_scroll.setWidget(self._model_container)
        self._model_scroll.setMinimumHeight(120)
        left_layout.addWidget(self._model_scroll, 1)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self._setup_tof_span()
        self._setup_ts_span()

    def showEvent(self, event):
        super().showEvent(event)
        # Defer one tick so width() reflects the final shown geometry.
        QTimer.singleShot(0, self._fit_main_splitter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_main_splitter()

    def _fit_main_splitter(self):
        """Keep the 2-panel splitter within the tab width.

        Without this the splitter can be seeded against a stale/too-wide
        geometry (at startup, on maximize, or when tabs are restored from a
        saved config) and overflow the viewport -- the panels stay clipped
        until a manual window restore+maximize. Only re-fit when the panes
        actually overflow, so a user's manual pane drag (which doesn't
        change the tab width) is preserved. (The HFS Models panel now
        lives inside the left column; the plots own everything else.)
        """
        sp = getattr(self, "_main_splitter", None)
        if sp is None:
            return
        avail = self.width() - 8  # main_layout contents margins (4 + 4)
        if avail <= 100:
            return
        sizes = sp.sizes()
        if len(sizes) != 2:
            return
        if sum(sizes) <= avail + 2:
            return  # already fits within the viewport
        left = min(max(sizes[0] or 400, 380), 540)
        center = max(400, avail - left)
        sp.setSizes([left, center])

    # ── Debounce and TOF span ─────────────────────────────────

    def _get_transition_cm(self):
        """Return the rest-frame transition in cm⁻¹ (E_upper - E_lower)."""
        return self._e_upper.value() - self._e_lower.value()

    def _get_fundamental_cm(self):
        """Return the fundamental (transition / harmonic) in cm⁻¹."""
        h = self._harmonic.value()
        return self._get_transition_cm() / h if h > 0 else 0.0

    def _update_transition_labels(self):
        """Update the read-only transition and fundamental labels."""
        trans = self._get_transition_cm()
        fund = self._get_fundamental_cm()
        self._transition_label.setText(f"{trans:.4f} cm\u207b\u00b9")
        self._fundamental_label.setText(f"{fund:.4f} cm\u207b\u00b9")

    # ── Binning sidebar helpers ──────────────────────────────────

    def _build_binning_group(self):
        """Build the 'Binning' QGroupBox for the Plot Options sidebar.

        Mirrors gui.analysis.blocks SourceBlock's binning section but with
        Pre-Analysis defaults (Raw Voltage + yerr=None) that preserve the
        legacy spectrum look.
        """
        grp = QGroupBox("Binning")
        form = QFormLayout(grp)
        form.setContentsMargins(4, 8, 4, 4)

        self._bin_mode_combo = QComboBox()
        self._bin_mode_combo.addItems(["Frequency", "Raw Voltage"])
        self._bin_mode_combo.setCurrentText("Raw Voltage")
        self._bin_mode_combo.setToolTip(
            "Domain in which events are binned.\n"
            "  Raw Voltage: groups events by unique DAC voltage.\n"
            "  Frequency: bins in MHz via clstools.Compute_Bins.")
        form.addRow("Bin mode:", self._bin_mode_combo)

        self._x_col_combo = QComboBox()
        self._x_col_combo.addItems(["bins_center", "Fmean"])
        self._x_col_combo.setCurrentText("bins_center")
        self._x_col_combo.setToolTip(
            "Frequency mode only: bin midpoint vs. mean event frequency.")
        form.addRow("x values:", self._x_col_combo)

        self._yerr_combo = QComboBox()
        self._yerr_combo.addItems(list(YERR_MODES))
        self._yerr_combo.setCurrentText("None")
        self._yerr_combo.setToolTip(
            "Per-bin y-uncertainty. 'None' renders the spectrum without\n"
            "error bars (Pre-Analysis default — matches the legacy look).")
        form.addRow("yerr mode:", self._yerr_combo)

        self._xerr_combo = QComboBox()
        self._xerr_combo.addItems(["None", "From voltage std"])
        self._xerr_combo.setCurrentText("None")
        self._xerr_combo.setToolTip(
            "Optional per-bin x-error (Frequency mode only).")
        form.addRow("x-error:", self._xerr_combo)

        self._bin_def_combo = QComboBox()
        self._bin_def_combo.addItems(list(BIN_DEFINITIONS))
        self._bin_def_combo.setCurrentText(DEFAULT_PA_BIN_DEFINITION)
        self._bin_def_combo.setToolTip(
            "How frequency bins are decided (Frequency mode only).\n\n"
            "Per scan step — one bin per scanning-voltage step, at the mean\n"
            "    frequency of its events. The default: it never lays a uniform\n"
            "    grid over the Doppler-nonlinear axis, so it cannot alias two\n"
            "    steps into one bin (the spurious doubled-count spike Auto shows).\n"
            "Auto — clstools' equal-width grid (~one bin per step).\n"
            "Fixed bin count / width — an explicit uniform grid.")
        form.addRow("Bin definition:", self._bin_def_combo)

        self._bin_count_spin = _make_int(
            DEFAULT_BIN_COUNT, 1, 1000000,
            tooltip="Fixed bin count (Frequency mode only).")
        form.addRow("Bin count:", self._bin_count_spin)

        self._bin_width_spin = _make_double(
            DEFAULT_BIN_WIDTH_MHZ, 1e-4, 1e6, 4, 1.0,
            tooltip="Fixed bin width in MHz (Frequency mode only).")
        form.addRow("Bin width [MHz]:", self._bin_width_spin)

        # Enable/disable rules: count/width gated on definition + mode.
        self._bin_mode_combo.currentTextChanged.connect(
            self._update_binning_enabled)
        self._bin_def_combo.currentTextChanged.connect(
            self._update_binning_enabled)
        self._update_binning_enabled()

        # Hook every change to a replot.
        for w in (self._bin_mode_combo, self._x_col_combo, self._yerr_combo,
                  self._xerr_combo, self._bin_def_combo,
                  self._bin_count_spin, self._bin_width_spin):
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._schedule_replot)
            else:
                w.valueChanged.connect(self._schedule_replot)

        return grp

    def _update_binning_enabled(self):
        """Grey out bin_count / bin_width based on mode + definition;
        also restrict the X-axis combo entries to those compatible with
        the current Bin mode.
        """
        is_freq = (self._bin_mode_combo.currentText() == "Frequency")
        bin_def = self._bin_def_combo.currentText()
        self._bin_def_combo.setEnabled(is_freq)
        self._x_col_combo.setEnabled(is_freq)
        self._xerr_combo.setEnabled(is_freq)
        self._bin_count_spin.setEnabled(
            is_freq and bin_def == "Fixed bin count")
        self._bin_width_spin.setEnabled(
            is_freq and bin_def == "Fixed bin width")
        self._sync_xaxis_for_bin_mode()

    def _sync_xaxis_for_bin_mode(self):
        """Restrict X-axis combo entries based on current Bin mode.

        Frequency binning has no clean inverse to volts, so Voltage /
        Cal voltage / Cal beam energy are hidden when Bin mode ==
        Frequency. The combo is temporarily blocked so the rebuild
        doesn't fire an extra replot mid-update.
        """
        bin_mode = self._bin_mode_combo.currentText()
        allowed = (XAXIS_MODES_FREQ_BIN if bin_mode == "Frequency"
                   else XAXIS_MODES)
        current = self._xaxis_combo.currentText()
        self._xaxis_combo.blockSignals(True)
        self._xaxis_combo.clear()
        self._xaxis_combo.addItems(allowed)
        if current in allowed:
            self._xaxis_combo.setCurrentText(current)
        else:
            self._xaxis_combo.setCurrentText("Frequency")
        self._xaxis_combo.blockSignals(False)

    def _effective_bin_mode(self):
        """Choose the binning domain from the active display axis.

        A Frequency / Wavenumber display bins in the frequency domain:
        clstools Compute_WL gives a per-event rest-frame frequency, then
        Compute_Bins lays down uniform-width frequency bins (count sized
        to the Doppler-shifted voltage step) -- the same binning the fit
        pipeline uses, so initial guesses read off here match the fit.
        Voltage-family axes keep one bin per raw DV step.

        Users normally view in Raw Voltage; flipping to a Frequency /
        Wavenumber axis to read off initial guesses reproduces the exact
        binning the fit will use.
        """
        if self._xaxis_combo.currentText() in ("Frequency", "Wavenumber"):
            return "Frequency"
        return "Raw Voltage"

    def _binning_cfg(self, entry, pmt_gate, tof_gate):
        """Build a cfg dict for gui.analysis.binning.compute_binned.

        Pulls binning controls from the sidebar widgets and gates from the
        passed arguments + per-entry virtual-split state. ``bin_mode`` is
        derived from the display axis (see ``_effective_bin_mode``). The
        hidden binning widgets supply the sub-defaults x_column=bins_center
        and bin_definition=Auto; yerr stays "None" to preserve the clean
        step look used for quick on-the-fly viewing.
        """
        v_gate = None
        # SplitFileEntry carries split_lo/hi as a permanent raw-voltage gate.
        if isinstance(entry, SplitFileEntry):
            v_gate = (float(entry.split_lo), float(entry.split_hi))

        return {
            "bin_mode": self._effective_bin_mode(),
            "x_column": self._x_col_combo.currentText(),
            "yerr_mode": self._yerr_combo.currentText(),
            "xerr_mode": self._xerr_combo.currentText(),
            "bin_definition": self._bin_def_combo.currentText(),
            "bin_count": int(self._bin_count_spin.value()),
            "bin_width_mhz": float(self._bin_width_spin.value()),
            # Spectrum-header "Bin: N × step" — groups N adjacent native
            # step bins (raw voltage steps, or their Doppler-shifted
            # per-step frequency twins).
            "step_multiple": int(self._spec_bin_mult.value()),
            "tof_gate": tof_gate,
            "pmt_gate": list(pmt_gate),
            "v_gate": v_gate,
        }

    def _prepare_frequency_data(self, entry, cooler_v, laser_sp,
                                mass, harmonic):
        """Re-run Compute_Voltages and Compute_WL with the user's current
        cooler/laser overrides on entry.cls_data.

        Forces the cooler voltage via VCoolDiv=0, VCoolOffset=cooler_v so
        that V = cooler_v - DV_cal. Pre-Analysis has no Shift_Ref
        equivalent (no ref_shift control on this tab); ref stays at
        clstools' default (0 Hz).
        """
        if entry.cls_data is None:
            return
        # Skip the per-event Compute_Voltages + Compute_WL when the physics
        # inputs are unchanged since the last prep on this entry: the F column
        # already holds exactly what a re-run would produce. The scan-filter
        # context restores data.Run (with F) on exit, so F survives between
        # replots; this key is invalidated on data (re)load.
        prep_key = (round(float(cooler_v), 6), round(float(laser_sp), 9),
                    round(float(mass), 9), int(harmonic))
        if getattr(entry, "_freq_prep_key", None) == prep_key:
            return
        data = entry.cls_data
        # Force the cooler voltage: VCoolDiv=0 makes V = VCoolOffset.
        data.VCoolDiv = 0
        data.VCoolOffset = float(cooler_v)
        data.Laser_set = float(laser_sp)
        data.Compute_Voltages(cooler_correction='pbp')
        # For Frequency_stepsize in Compute_WL we need Vcool_init *
        # VCoolDiv to equal cooler_v.
        data.Vcool_init = float(cooler_v)
        data.VCoolDiv = 1
        data.Compute_WL(
            Mass=float(mass), ref=0, harmonic=int(harmonic))
        # Restore VCoolDiv=0 so Compute_Bins' V column stays well-defined.
        data.VCoolDiv = 0
        entry._freq_prep_key = prep_key

    def _display_x(self, x_bin, bin_mode, xaxis_mode, entry,
                   cooler_v, laser_sp, mass, harmonic, offset):
        """Convert binned x (volts or MHz) to the user's display axis.

        Returns (x_display, xlabel). Returns (None, "") if the (bin_mode,
        xaxis_mode) combination is unsupported — the caller skips that
        entry and Task 6 hides the option from the combo.
        """
        if bin_mode == "Raw Voltage":
            dv_bins = x_bin   # bin centers are raw DV values
            if xaxis_mode == "Voltage":
                return dv_bins, "Scanning voltage (V)"

            # For Cal voltage / Beam energy / Wavenumber / Frequency we
            # need the calibrated voltage corresponding to each DV step.
            #
            # This applies the run's calibration *polynomial* -- literally what
            # clstools' Compute_Voltages does to every event
            # (``DV_cal = poly(DV) * VAccDiv``) -- so the axis shown here is the
            # voltage the fit actually uses, and it honours a per-run
            # calibration override.
            #
            # It used to snap each DAC step to the nearest raw CalReadback
            # sample instead. That quantized the axis to the calibration
            # sweep's step size, read straight off any settling-glitch points
            # the table contained, and silently ignored an override -- so
            # excluding bad calibration points visibly moved the Frequency-mode
            # spectrum while leaving this one exactly where it was.
            cal_info = getattr(
                getattr(entry, "cls_data", None), "CalibrationInfo", None)
            cal_set = entry.np_cal_set
            cal_rb = entry.np_cal_readback
            if cal_info is not None and getattr(cal_info, "coeffs_v", None):
                v_cal = cal_info.predict_v(dv_bins)
            elif cal_set is not None and len(cal_set) > 0:
                # No calibration resolved (e.g. a run with an unreadable
                # table): fall back to the nearest readback sample rather than
                # showing a raw DAC value on an axis labelled "volts".
                idx = np.array(
                    [np.argmin(np.abs(cal_set - v)) for v in dv_bins])
                v_cal = cal_rb[idx]
            else:
                v_cal = dv_bins   # no cal table → fall back to raw DV

            if xaxis_mode == "Calibrated voltage":
                return v_cal, "Calibrated voltage (V)"
            if xaxis_mode == "Calibrated beam energy":
                return cooler_v - v_cal, "Beam energy (V)"

            freq_MHz = self._voltage_to_frequency(
                cooler_v - v_cal, mass, harmonic, laser_sp)
            if xaxis_mode == "Wavenumber":
                wn = (freq_MHz * 1e6 / (C_LIGHT * 100.0)
                      - offset * harmonic)
                return wn, "Wavenumber (cm$^{-1}$)"
            # Frequency
            x = freq_MHz - offset * harmonic * C_LIGHT * 100.0 / 1e6
            return x, "Frequency (MHz)"

        # bin_mode == "Frequency": x_bin is absolute lab-frame MHz
        # (Compute_WL was called with ref=0).
        if xaxis_mode == "Wavenumber":
            wn = (x_bin * 1e6 / (C_LIGHT * 100.0) - offset * harmonic)
            return wn, "Wavenumber (cm$^{-1}$)"
        if xaxis_mode == "Frequency":
            x = x_bin - offset * harmonic * C_LIGHT * 100.0 / 1e6
            return x, "Frequency (MHz)"
        # Voltage / Cal V / Beam energy: no clean inverse from a single
        # frequency-bin centre. Caller skips this entry; Task 6 hides
        # these from the X-axis combo when Bin mode is Frequency.
        return None, ""

    def _on_replot_timer(self):
        """Debounce-timer slot: dispatch a full or gate-drag fast replot."""
        if self._pending_spectrum_only:
            self._pending_spectrum_only = False
            # Fast path when the cache is valid; it self-falls-back to the
            # dask spectrum-only replot when the view isn't eligible.
            self._replot_spectrum_fast()
        else:
            self._replot(spectrum_only=False)

    def _schedule_replot(self):
        # Full replot: redraw all panels + the calibration/cooler axes.
        self._pending_spectrum_only = False
        self._replot_timer.start()

    # ── Plot layout (Stacked vs Classic) ────────────────────────────

    def _on_layout_combo_changed(self, index):
        mode = "classic" if index == 1 else "stacked"
        self._set_plot_layout(mode)
        # Let the container mirror this onto every open project.
        self.layout_mode_changed.emit(mode)

    def _set_plot_layout(self, mode, replot=True):
        """Arrange the three plot panels.

        "stacked": ToF / Spectrum / Timestamp as three rows (DENIS
        default). "classic": Spectrum and ToF side by side on top with
        the Timestamp strip below — the original Data Viewer layout.
        Panels are reparented between splitters; a full replot follows
        so constrained layout, span selectors and the blit cache are
        rebuilt against the new canvas geometries.
        """
        mode = "classic" if mode == "classic" else "stacked"
        if mode == getattr(self, "_plot_layout_mode", "stacked"):
            return
        sp = self._plot_splitter
        # Detach the three panels (and dissolve a previous top row).
        for w in (self._tof_widget, self._spec_widget,
                  self._ts_widget):
            w.setParent(None)
        if self._top_hsplit is not None:
            self._top_hsplit.setParent(None)
            self._top_hsplit.deleteLater()
            self._top_hsplit = None
        if mode == "classic":
            top = QSplitter(Qt.Orientation.Horizontal)
            top.addWidget(self._spec_widget)
            top.addWidget(self._tof_widget)
            top.setStretchFactor(0, 5)
            top.setStretchFactor(1, 3)
            top.setSizes([760, 430])
            self._top_hsplit = top
            sp.addWidget(top)
            sp.addWidget(self._ts_widget)
            sp.setStretchFactor(0, 4)
            sp.setStretchFactor(1, 1)
            sp.setSizes([680, 220])
        else:
            sp.addWidget(self._tof_widget)
            sp.addWidget(self._spec_widget)
            sp.addWidget(self._ts_widget)
            sp.setStretchFactor(0, 3)
            sp.setStretchFactor(1, 4)
            sp.setStretchFactor(2, 3)
        for w in (self._tof_widget, self._spec_widget,
                  self._ts_widget):
            w.show()
        self._plot_layout_mode = mode
        # Keep the combo in sync when called programmatically.
        want = 1 if mode == "classic" else 0
        if self._layout_combo.currentIndex() != want:
            self._layout_combo.blockSignals(True)
            self._layout_combo.setCurrentIndex(want)
            self._layout_combo.blockSignals(False)
        if replot:
            self._schedule_replot()

    # ── Dark-mode plots (session-global, like the layout) ─────────

    def _update_dark_btn_tooltip(self):
        if self._dark_mode:
            self._dark_btn.setToolTip(
                "Dark plots: ON — click to return to the light canvas")
        else:
            self._dark_btn.setToolTip(
                "Dark plots: OFF — click for a black canvas with "
                "neon lines")

    def _on_dark_toggle(self, checked):
        """User clicked the sun/moon button on THIS project."""
        self._set_dark_mode(checked, replot=True)
        # Remember globally so new sessions open the same way.
        try:
            s = _load_settings()
            s["pa_dark_plots"] = bool(checked)
            _save_settings(s)
        except Exception:
            pass
        self.dark_mode_changed.emit(bool(checked))

    def _set_dark_mode(self, on, replot=True):
        """Apply dark/light plot theme. Idempotent; safe before any
        data is loaded (empty axes just get re-themed)."""
        self._dark_mode = bool(on)
        if self._dark_btn.isChecked() != self._dark_mode:
            self._dark_btn.blockSignals(True)
            self._dark_btn.setChecked(self._dark_mode)
            self._dark_btn.blockSignals(False)
        self._update_dark_btn_tooltip()
        # Give any entry / model that never got a distinct dark colour a
        # neon one by index, so toggling ALWAYS visibly recolours it
        # (old saves / pre-feature models had dark == light and so never
        # changed).
        # Track whether that fallback actually *changed* a colour: it
        # mutates the swatch of an artist that may already be drawn, so a
        # replot=False caller would otherwise leave the canvas showing the
        # old colour until something else redrew it (session load did
        # exactly that -- black model line under a yellow swatch).
        upgraded = False
        for i, entry in enumerate(self._file_entries):
            if hasattr(entry, "set_dark_active"):
                if (getattr(entry, "light_color", None)
                        == getattr(entry, "dark_color", None)):
                    idx = getattr(entry, "color_index", None)
                    if idx is None:
                        idx = i
                    entry.dark_color = NEON_COLORS[idx % len(NEON_COLORS)]
                    upgraded = True
                entry.set_dark_active(self._dark_mode)
        for i, panel in enumerate(self._model_panels):
            if hasattr(panel, "set_dark_active"):
                if panel.light_color == panel.dark_color:
                    panel.dark_color = NEON_MODEL_COLORS[
                        i % len(NEON_MODEL_COLORS)]
                    upgraded = True
                panel.set_dark_active(self._dark_mode)
        if replot and any(getattr(e, "check", None)
                          and e.check.isChecked()
                          for e in self._file_entries):
            # Fast path: recolour the existing artists in place instead
            # of a full re-bin + re-render of all nine figures (that
            # double-draw is what made the toggle lag).
            self._restyle_plots_in_place()
        else:
            # No live data (or replot suppressed): just repaint chrome.
            self._apply_plot_theme()
            self._apply_grids()
            if replot:
                self._schedule_replot()
            elif upgraded and self._dark_mode:
                # A colour upgrade under already-drawn artists: recolour
                # them in place (no re-bin, no re-render) so the canvas
                # matches the swatches the fallback just changed. Only
                # the dark colour moved, so light mode needs nothing.
                self._restyle_plots_in_place()

    def _restyle_plots_in_place(self):
        """Recolour the already-drawn spectrum / ToF / timestamp artists
        to the active per-run colours and refresh the models, without
        re-binning or re-rendering. Much faster than a full replot for a
        pure colour/theme change (dark-mode toggle)."""
        cmap = {}
        for e in self._file_entries:
            c = e.color
            cmap[f"run_{e.run_number}"] = c
            cmap[str(e.run_number)] = c
        for ax in (self._ax, self._tof_ax, self._ts_ax):
            for ln in ax.get_lines():
                if getattr(ln, "_is_hfs_model", False):
                    continue
                c = cmap.get(ln.get_label())
                if c:
                    ln.set_color(c)
            for cont in ax.containers:
                c = cmap.get(cont.get_label())
                if not c:
                    continue
                try:
                    dl = cont[0] if len(cont) else None
                    if dl is not None:
                        dl.set_color(c)
                    for cap in (cont[1] if len(cont) > 1 else []) or []:
                        cap.set_color(c)
                    for bar in (cont[2] if len(cont) > 2 else []) or []:
                        bar.set_color(c)
                except Exception:
                    pass
            for p in ax.patches:   # ToF / timestamp stairs = StepPatch
                try:
                    c = cmap.get(p.get_label())
                except Exception:
                    c = None
                if c:
                    try:
                        p.set_edgecolor(c)
                    except Exception:
                        pass
        # Models: authoritative recolour (removes + redraws their lines).
        self._update_models()
        # Chrome + grids follow the theme; a single coalesced draw each.
        self._apply_plot_theme([
            (self._fig, self._canvas),
            (self._tof_fig, self._tof_canvas),
            (self._ts_fig, self._ts_canvas),
        ])
        self._apply_grids()

    def _display_color(self, entry):
        """Line colour for a file entry — the entry's own active colour
        (light or dark, per set_dark_active)."""
        return entry.color

    def _display_color(self, entry):
        """Line colour for a file entry — the entry's own active colour
        (light or dark, per set_dark_active)."""
        return entry.color

    def _themed_figures(self):
        """(figure, canvas) pairs this tab owns, for theme application."""
        pairs = [
            ("_fig", "_canvas"), ("_tof_fig", "_tof_canvas"),
            ("_ts_fig", "_ts_canvas"),
            ("_cal_readback_fig", "_cal_readback_canvas"),
            ("_cal_diff_fig", "_cal_diff_canvas"),
            ("_cal_step_fig", "_cal_step_canvas"),
            ("_cal_cooler_fig", "_cal_cooler_canvas"),
            ("_cool_ohlc_fig", "_cool_ohlc_canvas"),
            ("_cool_ripple_fig", "_cool_ripple_canvas"),
        ]
        out = []
        for fig_attr, canvas_attr in pairs:
            fig = getattr(self, fig_attr, None)
            canvas = getattr(self, canvas_attr, None)
            if fig is not None and canvas is not None:
                out.append((fig, canvas))
        return out

    def _apply_plot_theme(self, pairs=None):
        """Paint figure chrome for the current mode: black canvas +
        white labels/ticks/spines/legend in dark mode, the matplotlib
        defaults otherwise. ``pairs`` limits the work to specific
        (figure, canvas) tuples — the gate-drag spectrum-only path
        re-themes just the spectrum, since ``ax.clear()`` resets the
        axis facecolor/spines every replot."""
        dark = self._dark_mode
        fig_fc = "#0d0d0d" if dark else "white"
        ax_fc = "black" if dark else "white"
        fg = "white" if dark else "black"
        grid_c = "#3a3a3a" if dark else "#b0b0b0"
        for fig, canvas in (pairs if pairs is not None
                            else self._themed_figures()):
            try:
                fig.set_facecolor(fig_fc)
                for ax in fig.axes:
                    ax.set_facecolor(ax_fc)
                    for spine in ax.spines.values():
                        spine.set_color(fg)
                    ax.tick_params(colors=fg, which="both")
                    ax.xaxis.label.set_color(fg)
                    ax.yaxis.label.set_color(fg)
                    ax.title.set_color(fg)
                    ax.xaxis.get_offset_text().set_color(fg)
                    ax.yaxis.get_offset_text().set_color(fg)
                    for gl in (ax.get_xgridlines() + ax.get_ygridlines()):
                        gl.set_color(grid_c)
                    leg = ax.get_legend()
                    if leg is not None:
                        for t in leg.get_texts():
                            t.set_color(fg)
                        frame = leg.get_frame()
                        frame.set_facecolor(ax_fc)
                        frame.set_edgecolor(fg)
                canvas.draw_idle()
            except Exception:
                _log.debug("Plot theme could not be applied to a figure",
                           exc_info=True)

    # ── Spectrum-tab grid toggles (session-global) ───────────────

    def _on_grid_toggle(self, axis, on):
        if axis == "x":
            self._grid_x = bool(on)
        else:
            self._grid_y = bool(on)
        self._apply_grids()
        self.grid_changed.emit(self._grid_x, self._grid_y)

    def _set_grids(self, x_on, y_on, emit=False):
        """Set both grid toggles (used by the container to mirror the
        session-global state and by save-restore)."""
        self._grid_x = bool(x_on)
        self._grid_y = bool(y_on)
        for btn, state in ((self._grid_x_btn, self._grid_x),
                           (self._grid_y_btn, self._grid_y)):
            if btn.isChecked() != state:
                btn.blockSignals(True)
                btn.setChecked(state)
                btn.blockSignals(False)
        self._apply_grids()
        if emit:
            self.grid_changed.emit(self._grid_x, self._grid_y)

    def _apply_grids(self, pairs=None):
        """Show/hide the x and y grids on the spectrum-tab plots, with a
        colour matched to the active (light/dark) theme. ``pairs`` limits
        the work to specific (axis, canvas) tuples — the gate-drag path
        re-grids only the spectrum, since ax.clear() drops the grid."""
        grid_c = "#3a3a3a" if self._dark_mode else "#b0b0b0"
        if pairs is None:
            pairs = [(getattr(self, "_ax", None),
                      getattr(self, "_canvas", None)),
                     (getattr(self, "_tof_ax", None),
                      getattr(self, "_tof_canvas", None)),
                     (getattr(self, "_ts_ax", None),
                      getattr(self, "_ts_canvas", None))]
        for ax, canvas in pairs:
            if ax is None:
                continue
            # Pass line properties ONLY when enabling — matplotlib turns
            # the grid ON if any style kwarg is supplied, even with
            # visible=False.
            if self._grid_x:
                ax.grid(True, axis="x", color=grid_c,
                        linewidth=0.6, alpha=0.9)
            else:
                ax.grid(False, axis="x")
            if self._grid_y:
                ax.grid(True, axis="y", color=grid_c,
                        linewidth=0.6, alpha=0.9)
            else:
                ax.grid(False, axis="y")
            if canvas is not None:
                canvas.draw_idle()

    def _schedule_replot_light(self, *_):
        """Schedule a spectrum-only replot (gate-drag fast path).

        Used by the ToF / time gate interactions, which only change the
        binned spectrum — the TOF / timestamp histograms and calibration
        panels don't depend on the gate being dragged. A full replot
        already pending in this debounce window wins (the timer is already
        active and the flag stays False), so a stray light schedule can't
        downgrade a queued full redraw. Accepts and ignores a signal
        argument so it can be connected to toggled(bool)."""
        if not self._replot_timer.isActive():
            self._pending_spectrum_only = True
        self._replot_timer.start()

    # ── Gate-drag fast path ─────────────────────────────────────────

    def _invalidate_fast_bg(self, *_):
        """Drop the cached blit background so the next gate-drag update
        re-captures it. Wired to the spectrum axes' xlim/ylim_changed so
        scroll-zoom / pan can't blit onto a stale background."""
        self._fast_bg = None

    @staticmethod
    def _edges_from_centers(centers):
        """Bin edges as midpoints between centers, ends extended by half
        the end spacing. Reproduces the edges clstools binned into, so an
        in-memory np.histogram lands every event in the same bin."""
        c = np.asarray(centers, dtype=float)
        if len(c) < 2:
            return None
        mids = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[c[0] - (c[1] - c[0]) * 0.5],
                               mids,
                               [c[-1] + (c[-1] - c[-2]) * 0.5]])

    def _current_gate_args(self, entry):
        """Live (pmt_gate, tof_gate|None, ts_gate_seconds|None) from the
        gate widgets -- the inputs that change as the user drags."""
        pmt = [i + 1 for i, cb in enumerate(self._channels)
               if cb.isChecked()]
        tofg = ([self._tof_lo.value(), self._tof_hi.value()]
                if self._tof_enable.isChecked() else None)
        tsg = self._ts_gate_seconds(entry)
        return pmt, tofg, tsg

    @staticmethod
    def _fast_histogram(cache, pmt, tofg, tsg):
        """In-memory gated histogram (raw counts), matching clstools'
        gate conventions EXACTLY: PMT via isin, TOF strict (min<t<max,
        per clstools.Compute_*_Bins), timestamp inclusive ([lo,hi], per
        scan_filter.gate_data_by_timestamp)."""
        m = np.isin(cache["tdc"], pmt)
        if tofg is not None:
            m = m & (cache["tof"] > min(tofg)) & (cache["tof"] < max(tofg))
        if tsg is not None:
            m = m & (cache["ts"] >= tsg[0]) & (cache["ts"] <= tsg[1])
        y, _ = np.histogram(cache["coord"][m], cache["edges"])
        return y.astype(float)

    def _try_build_fast_cache(self, entry, cfg, bin_mode, line, normalize,
                              xaxis_mode, cooler_v, laser_sp, mass,
                              harmonic, offset):
        """Build + self-check the gate-drag fast-path cache for a single
        entry. Returns the cache dict, or None if the view isn't
        numpy-histogrammable or the self-check fails (-> dask fallback).

        CRITICAL: the cache is built on the CANONICAL UNGATED bin grid,
        not on the (possibly gated) spectrum currently displayed. A TOF or
        timestamp gate can empty the extreme bins, and clstools' Frequency
        binning (and the Raw-Voltage reindex, which keys off the gated
        Sorted) then drop those edge bins -- producing a NARROWER grid. If
        we cached that, removing/narrowing the gate later could never
        restore the missing bins (np.histogram drops events outside the
        cached edges) and the x-range would stay shrunk until a full
        replot. Re-binning ungated here makes every gate a strict subset.

        The self-check re-histograms the ungated grid in-memory and
        requires bit-exact equality with clstools' compute_binned. Gated
        correctness then follows from the exactly-matched gate conventions
        in _fast_histogram (verified across modes x gate combinations).
        Also resets ``line`` to the full grid so the displayed spectrum
        and the cache share one x-axis.
        """
        try:
            # Canonical full grid: same channels, but NO TOF/TS/scan gate.
            cfg_full = dict(cfg)
            cfg_full["tof_gate"] = None
            out_full = compute_binned(entry.cls_data, cfg_full)
            centers = np.asarray(out_full["x"], dtype=float)
            edges = self._edges_from_centers(centers)
            if edges is None:
                return None
            S = getattr(entry.cls_data, "Sorted", None)
            if S is None:
                return None
            if hasattr(S, "compute"):
                S = S.compute()
            cols = getattr(S, "columns", [])
            if not all(c in cols for c in ("TOF", "TS", "TDC", "DV")):
                return None
            if bin_mode == "Raw Voltage":
                coord = S["DV"].to_numpy().astype(float)
            elif bin_mode == "Frequency":
                if "F" not in cols:
                    return None
                # clstools stores F in Hz post-Compute_WL; compute_binned
                # centers (out['x']) are in MHz.
                coord = S["F"].to_numpy().astype(float) / 1e6
            else:
                return None
            # Display-axis x for the full grid (drives the line + xlim).
            x_full, _ = self._display_x(
                centers, bin_mode, xaxis_mode, entry,
                cooler_v, laser_sp, mass, harmonic, offset)
            if x_full is None:
                return None
            cache = {
                "entry": entry,
                "line": line,
                "coord": coord,
                "edges": edges,
                "tof": S["TOF"].to_numpy().astype(float),
                "ts": S["TS"].to_numpy().astype(float),
                "tdc": S["TDC"].to_numpy(),
                "xdisplay": np.asarray(x_full, dtype=float),
                "normalize": bool(normalize),
            }
            # Self-check the UNGATED grid against the dask result.
            pmt_full = cfg.get("pmt_gate") or [
                i + 1 for i, cb in enumerate(self._channels)
                if cb.isChecked()]
            y_full = self._fast_histogram(cache, pmt_full, None, None)
            y0 = np.asarray(out_full["y"], dtype=float)
            if len(y_full) != len(y0) or not np.array_equal(y_full, y0):
                return None
            # Repaint the displayed line on the full grid with the CURRENT
            # gate's counts, so the spectrum and cache share one x-axis
            # (the enclosing full replot's autoscale then hugs the grid).
            pmt, tofg, tsg = self._current_gate_args(entry)
            y_now = self._fast_histogram(cache, pmt, tofg, tsg)
            if normalize:
                ymax = y_now.max()
                if ymax > 0:
                    y_now = y_now / ymax
            line.set_data(cache["xdisplay"], y_now)
            return cache
        except Exception:
            return None

    def _replot_spectrum_fast(self):
        """Gate-drag fast path: re-histogram in-memory and blit the single
        spectrum line. Falls back to the dask spectrum-only replot when
        the cache is absent/stale. ~1 ms vs ~185 ms for a full re-bin +
        figure redraw, which is what makes dragging feel instant."""
        # Re-entrancy guard: this method calls canvas.draw()/blit(); if it
        # were re-entered (e.g. a debounce timer firing inside a Qt paint
        # during a modal loop) the Agg renderer could be re-entered and
        # crash. Skip nested calls.
        if getattr(self, "_in_fast_replot", False):
            return
        self._in_fast_replot = True
        try:
            self._replot_spectrum_fast_impl()
        finally:
            self._in_fast_replot = False

    def _replot_spectrum_fast_impl(self):
        self._replot_timer.stop()
        cache = self._fast
        if cache is None:
            self._replot(spectrum_only=True)
            return
        try:
            entry = cache["entry"]
            pmt, tofg, tsg = self._current_gate_args(entry)
            y = self._fast_histogram(cache, pmt, tofg, tsg)
            if len(y) != len(cache["xdisplay"]):
                raise ValueError("bin count drift")
            if cache["normalize"]:
                ymax0 = y.max()
                if ymax0 > 0:
                    y = y / ymax0
            ln = cache["line"]
            ln.set_ydata(y)
            ax = self._ax
            canvas = self._canvas

            # Capture the static background (everything except the
            # spectrum line) once. Re-captured after any rescale below.
            if self._fast_bg is None:
                ln.set_visible(False)
                canvas.draw()
                self._fast_bg = canvas.copy_from_bbox(ax.bbox)
                ln.set_visible(True)

            ymax = float(np.max(y)) if len(y) else 1.0
            top = ax.get_ylim()[1]
            if ymax > top * 0.999 or ymax < top * 0.4:
                # New counts leave the current y-view (e.g. a gate
                # widened/removed): rescale and re-capture the background,
                # then blit. One full draw here, blits afterwards.
                ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
                canvas.draw()
                self._fast_bg = canvas.copy_from_bbox(ax.bbox)
            canvas.restore_region(self._fast_bg)
            ax.draw_artist(ln)
            canvas.blit(ax.bbox)
        except Exception:
            # Any blit/cache problem -> drop to the safe dask redraw.
            self._fast = None
            self._fast_bg = None
            self._replot(spectrum_only=True)

    def _on_tof_enable_toggled(self, checked):
        """Enable -> full replot (creates the gate SpanSelector + arms the
        fast path). Disable -> hide the gate rectangle and fast-update the
        spectrum, so REMOVING a gate is instant instead of a full re-bin +
        figure redraw."""
        if checked:
            self._schedule_replot()
        else:
            self._hide_span(self._tof_span, self._tof_canvas)
            self._schedule_replot_light()

    def _on_ts_enable_toggled(self, checked):
        """Timestamp-gate twin of _on_tof_enable_toggled."""
        if checked:
            self._schedule_replot()
        else:
            self._hide_span(self._ts_span, self._ts_canvas)
            self._schedule_replot_light()

    @staticmethod
    def _hide_span(span, canvas):
        """Hide an interactive SpanSelector's shaded rectangle and repaint
        its single-axis canvas. Cheap next to a full replot; a later full
        replot (re-enable, or a fresh drag that re-checks the gate)
        recreates the selector."""
        if span is None:
            return
        try:
            span.set_visible(False)
        except Exception:
            pass
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def _setup_tof_span(self):
        # Save gate values from spinboxes (source of truth)
        lo = self._tof_lo.value()
        hi = self._tof_hi.value()

        # Always create a fresh SpanSelector on the replotted axes.
        # onmove_callback gives LIVE spectrum feedback as the gate is
        # dragged/resized (matplotlib fires it for region-move, edge-
        # resize and creation alike), so the gate updates continuously.
        #
        # useblit=False is DELIBERATE: with useblit a *visible* selection
        # makes the selector's draw_event handler (update_background) run a
        # NESTED synchronous canvas.draw() on every redraw. During a modal
        # context-menu loop a Qt paintEvent can re-enter that nested draw
        # and crash the Agg renderer (the "gate active -> right-click menu
        # laggy + UI crash" bug). Without blit the selector repaints via
        # coalesced draw_idle instead -- the TOF/timestamp figures are
        # cheap, and the SPECTRUM stays smooth through its own numpy
        # fast-path blit (a separate canvas with no SpanSelector).
        self._tof_span = SpanSelector(
            self._tof_ax, self._on_tof_span_select,
            'horizontal', useblit=False,
            props=dict(alpha=0.3, facecolor='orange'),
            interactive=True, drag_from_anywhere=True,
            onmove_callback=self._on_tof_span_move,
        )

        if self._tof_enable.isChecked() and hi > lo:
            # The interactive SpanSelector's own rectangle IS the gate
            # overlay (alpha 0.3 orange). We deliberately do NOT add a
            # separate static axvspan: during a spectrum-only gate drag we
            # don't redraw this axis, so a static span would freeze at the
            # pre-drag position and leave a stale second shaded region next
            # to the live selector. Setting extents shows the selector
            # rectangle at the current gate without that hazard.
            try:
                self._tof_span.extents = (lo, hi)
            except Exception:
                pass

    def _set_tof_spinboxes(self, xmin, xmax):
        """Write gate edges to the spinboxes without re-triggering their
        valueChanged handler (avoids fighting the active drag)."""
        self._tof_lo.blockSignals(True)
        self._tof_hi.blockSignals(True)
        self._tof_lo.setValue(round(xmin, 1))
        self._tof_hi.setValue(round(xmax, 1))
        self._tof_lo.blockSignals(False)
        self._tof_hi.blockSignals(False)

    def _on_tof_span_move(self, xmin, xmax):
        """Live drag: update the spectrum continuously as the gate moves.
        Calls the in-memory fast path directly (~6 ms) for instant
        feedback; falls back to the debounced dask replot only when the
        fast cache isn't armed (multi-file / merged views)."""
        if not self._tof_enable.isChecked():
            return  # creating the very first gate -> wait for release
        self._set_tof_spinboxes(xmin, xmax)
        if self._fast is not None:
            self._replot_spectrum_fast()
        else:
            self._schedule_replot_light()

    def _on_tof_span_select(self, xmin, xmax):
        self._set_tof_spinboxes(xmin, xmax)
        if not self._tof_enable.isChecked():
            self._tof_enable.setChecked(True)
        else:
            # Drag release: final spectrum update (rescales the y-view if
            # the live blit frames drifted out of range).
            self._schedule_replot_light()

    def _on_tof_spinbox_changed(self):
        if self._tof_span is not None and self._tof_enable.isChecked():
            lo = self._tof_lo.value()
            hi = self._tof_hi.value()
            if hi > lo:
                try:
                    self._tof_span.extents = (lo, hi)
                    self._tof_canvas.draw_idle()
                except Exception:
                    pass
        self._schedule_replot_light()

    def _setup_ts_span(self):
        """Interactive time-gate SpanSelector on the timestamp axis,
        mirroring _setup_tof_span. Press-drag on the timestamp plot to
        set the gate; the shaded region is redrawn from the spinboxes.
        """
        lo = self._ts_lo.value()
        hi = self._ts_hi.value()
        # useblit=False on purpose -- see _setup_tof_span for the nested
        # draw / modal-menu crash this avoids.
        self._ts_span = SpanSelector(
            self._ts_ax, self._on_ts_span_select,
            'horizontal', useblit=False,
            props=dict(alpha=0.3, facecolor='deepskyblue'),
            interactive=True, drag_from_anywhere=True,
            onmove_callback=self._on_ts_span_move,
        )
        if self._ts_enable.isChecked() and hi > lo:
            # Selector rectangle is the gate overlay; no static axvspan
            # (see _setup_tof_span for the stale-overlay rationale).
            try:
                self._ts_span.extents = (lo, hi)
            except Exception:
                pass

    def _set_ts_spinboxes(self, xmin, xmax):
        self._ts_lo.blockSignals(True)
        self._ts_hi.blockSignals(True)
        self._ts_lo.setValue(round(xmin, 3))
        self._ts_hi.setValue(round(xmax, 3))
        self._ts_lo.blockSignals(False)
        self._ts_hi.blockSignals(False)

    def _on_ts_span_move(self, xmin, xmax):
        """Live time-gate drag -> continuous spectrum update (see
        _on_tof_span_move)."""
        if not self._ts_enable.isChecked():
            return
        self._set_ts_spinboxes(xmin, xmax)
        if self._fast is not None:
            self._replot_spectrum_fast()
        else:
            self._schedule_replot_light()

    def _on_ts_span_select(self, xmin, xmax):
        self._set_ts_spinboxes(xmin, xmax)
        if not self._ts_enable.isChecked():
            self._ts_enable.setChecked(True)
        else:
            self._schedule_replot_light()

    def _on_ts_spinbox_changed(self):
        if self._ts_span is not None and self._ts_enable.isChecked():
            lo = self._ts_lo.value()
            hi = self._ts_hi.value()
            if hi > lo:
                try:
                    self._ts_span.extents = (lo, hi)
                    self._ts_canvas.draw_idle()
                except Exception:
                    pass
        self._schedule_replot_light()

    def _ts_gate_seconds(self, entry):
        """Translate the time-gate spinboxes (in the displayed, relative,
        selected-unit coordinates) into an ABSOLUTE (lo, hi) in seconds
        on entry.np_ts, or None when the gate is off / degenerate.

        The timestamp x-axis shows ``(ts - ts.min()) / unit_divisor``;
        invert that: ``ts_abs = ts.min() + value * unit_divisor``.
        """
        if not self._ts_enable.isChecked():
            return None
        lo = self._ts_lo.value()
        hi = self._ts_hi.value()
        if hi <= lo:
            return None
        ts = getattr(entry, "np_ts", None)
        if ts is None or len(ts) == 0:
            return None
        div = self._ts_unit_divisor()
        base = float(ts.min())
        return (base + lo * div, base + hi * div)

    # ── File loading ─────────────────────────────────────────────

    def _open_files(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        # Re-entrancy guard: processEvents() below pumps the event loop, so a
        # second Open click while loading would otherwise start a nested load.
        if getattr(self, "_loading_files", False):
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open ASDF or .vasdf Files",
            get_last_dir("data", "load"),
            "ASDF and virtual splits (*.asdf *.vasdf);;"
            "ASDF files (*.asdf);;"
            "Virtual splits (*.vasdf);;"
            "All files (*)")
        if not paths:
            return
        remember_last_dir("data", "load", paths[0])
        # Loading + dask .compute() runs on the GUI thread (a background worker
        # is the proper fix); until then, show a busy cursor and pump events
        # between files so the window doesn't look hung and the file list fills
        # progressively (code review 2026-06-02, preanalysis-load-blocks-gui).
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        self._loading_files = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for path in paths:
                if any(fe.filepath == path for fe in self._file_entries):
                    continue
                self._load_file(path)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
            self._loading_files = False
        self._replot()

    def _load_file(self, filepath):
        if is_vasdf_path(filepath):
            self._load_vasdf_file(filepath)
            return
        # DENIS-labelled merged ASDF? Route to MergedFileEntry
        # directly. The standard ``af['Run']`` / ``af['CoolerVoltage']``
        # read below would otherwise KeyError on the merged schema.
        try:
            from gui.analysis.merge import (
                is_merged_asdf, load_merged_asdf)
        except ImportError:
            is_merged_asdf = lambda _: False  # noqa: E731
            load_merged_asdf = lambda _: None  # noqa: E731
        if is_merged_asdf(filepath):
            try:
                md = load_merged_asdf(filepath)
            except ValueError as exc:
                # Surfaced by load_merged_asdf's consistency guard
                # (legacy bin_mode/domain mismatch bug). Tell the
                # user clearly instead of falling through to the
                # clstools path, which would crash anyway because
                # the new schema has no ``raw`` array.
                QMessageBox.warning(
                    self, "Merged ASDF inconsistency",
                    f"Cannot load this merged ASDF:\n\n{exc}")
                return
            if md is not None:
                self._load_merged_asdf_entry(filepath, md)
                return
        try:
            with asdf.open(filepath) as af:
                run_number = af['Run']
                cooler_v = af['CoolerVoltage'] * 10000
                date = af.tree.get('Date', None)
                laser_sp = af['LaserSetpoint']
                # code review 2026-06-02, file-mass-tooltip-misleading-zero:
                # None (not 0) when MassAMU is absent so the tooltip can
                # distinguish "missing" from a real value.
                mass_amu = af.tree.get('MassAMU', None)
                cal_set = np.array(af.tree.get('CalSet', []), dtype=float)
                cal_readback = np.array(af.tree.get('CalReadback', []), dtype=float) * 1000
        except Exception as e:
            QMessageBox.warning(self, "Load Error",
                                f"Failed to read {filepath}:\n{e}")
            return

        color_idx = len(self._file_entries) % len(DEFAULT_COLORS)
        color = DEFAULT_COLORS[color_idx]

        entry = FileEntry(filepath, run_number, cooler_v, date, laser_sp,
                          mass_amu, color,
                          dark_color=NEON_COLORS[color_idx % len(NEON_COLORS)])
        entry.color_index = color_idx   # stable neon slot for dark mode
        entry.set_dark_active(self._dark_mode)
        entry.np_cal_set = cal_set
        entry.np_cal_readback = cal_readback
        entry.toggled.connect(self._schedule_replot)
        entry.toggled.connect(self._refresh_master_check_state)
        entry.color_changed.connect(self._schedule_replot)
        entry.clicked.connect(self._select_file_entry)
        entry.removed.connect(self._on_entry_remove_requested)
        entry.reload_requested.connect(self._reload_file_entry)

        self._file_entries.append(entry)
        self._file_list_layout.addWidget(entry)
        self._refresh_master_check_state()

        self._load_cls_data(entry)
        entry.update_detail()

    def _load_merged_asdf_entry(self, filepath, merged_data):
        """Translate a loaded DENIS-merged ASDF into a MergedFileEntry.

        Used when the user picks Open... on an exported merge ASDF
        instead of a raw clstools run. The merged_data dict already
        has the shape ``compute_merged_spectrum`` produces, so we
        just shred it back into the constructor args ``MergedFileEntry``
        expects.

        Sets ``filepath`` to the user-picked ASDF path (not the
        synthetic ``[merged] <name>``) so a second Open of the same
        file is detected by the de-dup check in ``_open_files``, and
        so the YAML save records the real on-disk path.
        """
        name = merged_data.get("merged_name", "merged")
        mm = merged_data.get("merge_metadata", {}) or {}
        domain = ("frequency"
                  if merged_data.get("x_unit") == "MHz"
                  else "voltage")
        # ``source_info`` is a list of dicts; the export writes them via
        # ``per_run_audit``. Map back into the shape MergedFileEntry
        # expects.
        per_run = merged_data.get("per_run") or []
        source_runs = merged_data.get("source_runs") or []
        source_files = merged_data.get("source_files") or []
        source_info = []
        for i, rd in enumerate(per_run):
            source_info.append({
                "run_number": rd.get("run_num",
                                      source_runs[i]
                                      if i < len(source_runs) else None),
                "filepath":   rd.get("path",
                                      source_files[i]
                                      if i < len(source_files) else None),
                "cooler_v":   rd.get("cooler_v"),
                "laser_sp":   rd.get("laser_set") or rd.get("laser_sp"),
                "mass_amu":   rd.get("mass_amu"),
                "harmonic":   rd.get("harmonic"),
            })
        if not source_info:
            # Older exports may have only ``source_runs`` / ``source_files``.
            for rn, fp in zip(source_runs, source_files):
                source_info.append({
                    "run_number": rn, "filepath": fp,
                    "cooler_v": None, "laser_sp": None,
                    "mass_amu": None, "harmonic": None,
                })

        mfe = MergedFileEntry(
            name=name,
            merged_x=merged_data["x"],
            merged_y=merged_data["y"],
            merge_domain=domain,
            source_info=source_info,
            merge_cooler_v=mm.get("cooler_v"),
            merge_laser_sp=mm.get("laser_sp"),
            merge_mass_amu=mm.get("mass_amu"),
            merge_harmonic=mm.get("harmonic"),
            per_run=per_run,
        )
        # Override the synthetic "[merged] <name>" filepath with the
        # real on-disk path so the de-dup / save-config logic treats
        # this as a normal loaded file. The display label keeps the
        # ✦ <name> styling set in MergedFileEntry.__init__.
        mfe.filepath = filepath
        mfe.toggled.connect(self._schedule_replot)
        mfe.toggled.connect(self._refresh_master_check_state)
        mfe.color_changed.connect(self._schedule_replot)
        mfe.removed.connect(self._on_entry_remove_requested)
        mfe.clicked.connect(self._select_file_entry)
        # Wire the merged entry's View / Edit / Export menu signals to
        # the handlers that open the shared MergeDialog / viewer / export.
        mfe.view_requested.connect(self._view_merged_entry)
        mfe.edit_requested.connect(self._edit_merged_entry)
        mfe.export_requested.connect(self._export_merged_entry)
        mfe.color_index = len(self._file_entries)  # neon slot (dark mode)
        mfe.dark_color = NEON_COLORS[mfe.color_index % len(NEON_COLORS)]
        mfe.set_dark_active(self._dark_mode)
        self._file_entries.append(mfe)
        self._file_list_layout.addWidget(mfe)
        self._refresh_master_check_state()
        self._schedule_replot()

    def _load_vasdf_file(self, vasdf_path):
        """Load a virtual-split sidecar: read descriptor, then load
        the parent ASDF behind it and apply the V-gate + metadata
        overrides. Identity is the .vasdf path itself, so two splits
        of the same parent appear as independent runs everywhere.
        """
        try:
            desc = read_vasdf(vasdf_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Load Error",
                f"Failed to read .vasdf descriptor:\n{vasdf_path}\n{exc}")
            return

        parent_path = desc["parent_path"]
        if not os.path.isfile(parent_path):
            QMessageBox.warning(
                self, "Parent ASDF Missing",
                f"Parent ASDF not found:\n{parent_path}\n\n"
                f"This .vasdf references that path. Restore the parent "
                "ASDF or update the .vasdf and try again.")
            return

        try:
            with asdf.open(parent_path) as af:
                run_number = af['Run']
                asdf_cooler_v = af['CoolerVoltage'] * 10000
                date = af.tree.get('Date', None)
                asdf_laser_sp = af['LaserSetpoint']
                # code review 2026-06-02, file-mass-tooltip-misleading-zero:
                # None (not 0) when MassAMU is absent so the tooltip can
                # distinguish "missing" from a real value.
                mass_amu = af.tree.get('MassAMU', None)
                cal_set = np.array(
                    af.tree.get('CalSet', []), dtype=float)
                cal_readback = np.array(
                    af.tree.get('CalReadback', []), dtype=float) * 1000
        except Exception as exc:
            QMessageBox.warning(
                self, "Load Error",
                f"Failed to read parent ASDF:\n{parent_path}\n{exc}")
            return

        md = desc.get("metadata_override", {}) or {}
        cooler_v = float(md.get("cooler_v", asdf_cooler_v))
        laser_sp = float(md.get("laser_sp", asdf_laser_sp))

        color_idx = len(self._file_entries) % len(DEFAULT_COLORS)
        color = DEFAULT_COLORS[color_idx]

        _split_color_idx = color_idx
        entry = SplitFileEntry(
            vasdf_path=vasdf_path,
            parent_path=parent_path,
            source_id=desc["source_id"],
            split_lo=desc["split"]["lo"],
            split_hi=desc["split"]["hi"],
            cooler_v=cooler_v, date=date,
            laser_sp=laser_sp, mass_amu=mass_amu,
            label=desc.get("label"),
            metadata_override=md,
            color=color,
            dark_color=NEON_COLORS[_split_color_idx % len(NEON_COLORS)],
        )
        entry.color_index = _split_color_idx  # stable neon slot
        entry.set_dark_active(self._dark_mode)
        entry.np_cal_set = cal_set
        entry.np_cal_readback = cal_readback
        entry.toggled.connect(self._schedule_replot)
        entry.toggled.connect(self._refresh_master_check_state)
        entry.color_changed.connect(self._schedule_replot)
        entry.clicked.connect(self._select_file_entry)
        entry.removed.connect(self._on_entry_remove_requested)
        entry.reload_requested.connect(self._reload_file_entry)

        self._file_entries.append(entry)
        self._file_list_layout.addWidget(entry)
        self._refresh_master_check_state()

        # _load_cls_data picks up entry.parent_path automatically
        # since SplitFileEntry carries it.
        self._load_cls_data(entry)
        entry.update_detail()

    def _load_cls_data(self, entry):
        """Load ASDF via clstools, then cache all data as numpy arrays."""
        if _get_clstools() is None:
            QMessageBox.warning(
                self, "Missing dependency",
                "clstools is not installed. Cannot load ASDF data.\n"
                "Install with: pip install -e <path-to-cls_tools>")
            return

        # SplitFileEntry's filepath is the .vasdf descriptor itself;
        # the actual events live in the parent ASDF it points at.
        data_path = (entry.parent_path
                     if isinstance(entry, SplitFileEntry)
                     else entry.filepath)

        try:
            from gui.calibration import get_registry as _get_cal_registry
            from gui.calibration import load_run_calibrated, spec_fingerprint

            _cal_reg = _get_cal_registry()
            data = _get_clstools().CLSDataFrame()
            # Overwrite data.Cal with the run's chosen calibration BEFORE
            # Compute_Voltages, which reads Cal/Cal_order/VAccDiv and nothing
            # else. A split keys off its parent ASDF (data_path above), since
            # the calibration belongs to the file the events came from -- the
            # same rule the scan filter uses.
            load_run_calibrated(data, data_path, _cal_reg.to_dict())
            # Remember which calibration these arrays were built from, so
            # _on_calibrations_changed can reload only what actually changed.
            entry._cal_fingerprint = spec_fingerprint(
                _cal_reg.get(data_path))
            # The entry is fully constructed by now (a split knows its parent,
            # a merged entry knows it is merged), so the alert badge can
            # finally resolve its path and decide whether to warn.
            if getattr(entry, "cal_alert", None) is not None:
                entry.cal_alert.refresh()
            data.Compute_Voltages()
            # Keep the dataframe alive so compute_binned can re-bin on every
            # widget change. clstools uses dask, so data.Run stays lazy until
            # touched; memory cost is bounded by the numpy arrays we already
            # extract below.
            entry.cls_data = data
            # Underlying event data changed -> drop the per-entry binned-result
            # and frequency-prep caches so the next replot recomputes them.
            entry._bin_cache = None
            entry._freq_prep_key = None

            df = data.Sorted
            if hasattr(df, 'compute'):
                df = df.compute()

            # Cache numpy arrays for fast replot
            entry.np_v = df['V'].to_numpy()

            # Raw DAC voltage for grouping (events at same step)
            for col in ('DV', 'dv', 'DAC'):
                if col in df.columns:
                    entry.np_dv = df[col].to_numpy()
                    break
            if entry.np_dv is None:
                entry.np_dv = np.round(entry.np_v, 2)

            for col in ('TOF', 'T', 'tof'):
                if col in df.columns:
                    entry.np_tof = df[col].to_numpy()
                    break
            if entry.np_tof is None:
                entry.np_tof = np.zeros(len(entry.np_v))

            for col in ('TDC', 'CH', 'tdc', 'ch'):
                if col in df.columns:
                    entry.np_tdc = df[col].to_numpy()
                    break
            if entry.np_tdc is None:
                entry.np_tdc = np.ones(len(entry.np_v))

            for col in ('TS', 'ts', 'Timestamp'):
                if col in df.columns:
                    entry.np_ts = df[col].to_numpy()
                    break
            if entry.np_ts is None:
                entry.np_ts = np.zeros(len(entry.np_v))

            # Bunch column drives scan derivation -- it's monotonic
            # across the full run and integer-divides cleanly into
            # scan blocks. Optional: some legacy ASDFs may omit it.
            for col in ('Bunch', 'bunch', 'BunchNumber', 'bunch_number'):
                if col in df.columns:
                    entry.np_bunch = df[col].to_numpy()
                    break

            # Stash scan metadata (ScanningRanges / StepSize /
            # BunchesPerChannel). The first two live on the data
            # object; BunchesPerChannel only lives in the ASDF tree,
            # so we lift it via a metadata-only re-open.
            try:
                from gui.scan_filter import read_bunches_per_channel
                source_path = (getattr(entry, "parent_path", None)
                                or entry.filepath)
                entry.scan_meta = {
                    "scanning_ranges": getattr(data, "ScanningRanges", None),
                    "step_size": getattr(data, "Step_Size", None),
                    "bunches_per_channel":
                        read_bunches_per_channel(source_path),
                }
            except Exception:
                entry.scan_meta = None

            # Per-event cooler voltage in V: Vrfq * VCoolDiv + VCoolOffset.
            vcool_div = float(getattr(data, 'VCoolDiv', 10000) or 10000)
            vcool_off = float(getattr(data, 'VCoolOffset', 0) or 0)
            for col in ('Vrfq', 'vrfq', 'VCool', 'Vcool'):
                if col in df.columns:
                    entry.np_vcool = (
                        df[col].to_numpy() * vcool_div + vcool_off)
                    break

        except Exception as e:
            QMessageBox.warning(
                self, "Data Processing Error",
                f"Failed to process {entry.filepath}:\n{e}")

    def checked_file_entries(self):
        """Ticked file entries -- the single source of truth for what an
        Analysis-side import pulls. An entry without a checkbox counts as
        ticked."""
        return [fe for fe in self._file_entries
                if getattr(fe, 'check', None) is None or fe.check.isChecked()]

    def _remove_checked(self):
        """Remove the checked files from the list.

        Operates on the checked entries to match 'Merge Checked' (both act on
        the ticked files), renamed from the old 'Remove unchecked'.
        """
        to_remove = [fe for fe in self._file_entries
                     if fe.check.isChecked()]
        for fe in to_remove:
            self._file_entries.remove(fe)
            self._file_list_layout.removeWidget(fe)
            fe.deleteLater()
        self._refresh_master_check_state()
        self._replot()

    def _app_undo_stack(self):
        """The MainWindow's global QUndoStack, or None (headless/tests)."""
        try:
            return getattr(self.window(), "_undo_stack", None)
        except Exception:
            return None

    def _on_entry_remove_requested(self, entry):
        """User clicked 'Remove' on a file entry. Push an undoable command
        so Ctrl+Z restores it; fall back to a hard delete when there's no
        undo stack (headless tests)."""
        stack = self._app_undo_stack()
        if stack is None or entry not in self._file_entries:
            self._remove_single_entry(entry)
            return
        stack.push(_FileRemoveCommand(self, entry))

    def _do_soft_remove_entry(self, entry):
        """Detach a file entry WITHOUT deleting it (the undo command keeps
        it alive for restoration)."""
        if entry in self._file_entries:
            self._file_entries.remove(entry)
            self._file_list_layout.removeWidget(entry)
            entry.setParent(None)
            self._refresh_master_check_state()
            self._replot()

    def _do_reinsert_entry(self, entry, index):
        """Re-attach a soft-removed file entry at ``index``."""
        index = max(0, min(int(index), len(self._file_entries)))
        self._file_entries.insert(index, entry)
        self._file_list_layout.insertWidget(index, entry)
        entry.show()
        self._refresh_master_check_state()
        self._replot()

    def _remove_single_entry(self, entry):
        """Hard-remove a single file entry (no undo). Used by the
        merged->Analysis detach path and as the headless fallback."""
        if entry in self._file_entries:
            self._file_entries.remove(entry)
            self._file_list_layout.removeWidget(entry)
            entry.deleteLater()
            self._refresh_master_check_state()
            self._replot()

    def _on_master_check_clicked(self):
        """User clicked the master tickbox: force every entry to the
        opposite of "all currently checked". Mixed and all-unchecked
        states both round up to "check everything"."""
        if not self._file_entries:
            return
        all_checked = all(e.check.isChecked() for e in self._file_entries)
        new_state = not all_checked
        # Suppress per-entry replot triggers: do all the writes first,
        # then refresh the master visual + schedule one replot.
        for e in self._file_entries:
            e.check.blockSignals(True)
            e.check.setChecked(new_state)
            e.check.blockSignals(False)
        self._refresh_master_check_state()
        self._schedule_replot()

    def _refresh_master_check_state(self):
        """Drive the master tickbox from the per-entry checkboxes:
        all-checked -> Checked, all-unchecked -> Unchecked, mixed ->
        PartiallyChecked. Also disabled when the file list is empty."""
        self._master_check.blockSignals(True)
        try:
            n = len(self._file_entries)
            if n == 0:
                self._master_check.setEnabled(False)
                self._master_check.setCheckState(Qt.CheckState.Unchecked)
                return
            self._master_check.setEnabled(True)
            n_checked = sum(1 for e in self._file_entries
                             if e.check.isChecked())
            if n_checked == 0:
                self._master_check.setCheckState(Qt.CheckState.Unchecked)
            elif n_checked == n:
                self._master_check.setCheckState(Qt.CheckState.Checked)
            else:
                self._master_check.setCheckState(
                    Qt.CheckState.PartiallyChecked)
        finally:
            self._master_check.blockSignals(False)

    def _cal_peers(self):
        """Loaded runs offered as calibration borrow donors.

        Deduped on the ASDF path: two splits of the same parent share one
        calibration table, so offering both would be offering the same donor
        twice.
        """
        peers, seen = [], set()
        for e in self._file_entries:
            if getattr(e, "_is_merged", False):
                continue
            p = (e.parent_path if isinstance(e, SplitFileEntry)
                 else e.filepath)
            if not p:
                continue
            from gui.calibration import canonical_path
            key = canonical_path(p)
            if key in seen:
                continue
            seen.add(key)
            peers.append((f"run_{e.run_number}", p))
        return peers

    def _on_calibrations_changed(self):
        """Rebuild the runs whose voltage calibration actually changed.

        A scan filter only drops events at binning time, so invalidating the
        binned cache is enough for it. A calibration is different in kind: it
        decides what voltage every event *has*, so both the cached
        CLSDataFrame (which still holds the old ``Cal``) and ``entry.np_v``
        (derived from it) are stale and must be rebuilt from the ASDF.

        Gated on the spec fingerprint so that applying one calibration to
        fifty checked runs doesn't reload the forty-nine that were already
        correct. The registry's ``set_many`` keeps that to a single signal.
        """
        from gui.calibration import get_registry, spec_fingerprint
        reg = get_registry()

        stale = []
        for entry in self._file_entries:
            if getattr(entry, "_is_merged", False):
                continue          # pre-binned; no per-event data to recompute
            path = (entry.parent_path
                    if isinstance(entry, SplitFileEntry)
                    else entry.filepath)
            fp = spec_fingerprint(reg.get(path))
            if fp != getattr(entry, "_cal_fingerprint", None):
                stale.append(entry)
        if not stale:
            return

        # Each rebuild re-reads an ASDF, and "apply to all checked runs" can
        # make that fifty of them. Show progress rather than freezing the
        # window for a minute with no explanation, and let the user stop --
        # the registry already holds the new calibration either way, so an
        # interrupted pass just leaves the remaining runs to be rebuilt on
        # their next replot.
        from PySide6.QtWidgets import QProgressDialog

        progress = None
        if len(stale) > 2:
            progress = QProgressDialog(
                "Applying calibration…", "Stop", 0, len(stale), self)
            progress.setWindowTitle("Calibration")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(400)
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        failed = []
        try:
            for i, entry in enumerate(stale):
                if progress is not None:
                    if progress.wasCanceled():
                        break
                    progress.setLabelText(
                        f"Applying calibration to run_{entry.run_number}…")
                    progress.setValue(i)
                    QApplication.processEvents()
                try:
                    self._load_cls_data(entry)
                    entry.update_detail()
                except Exception as exc:
                    failed.append(
                        f"run_{entry.run_number}: {exc}")
        finally:
            if progress is not None:
                progress.setValue(len(stale))
                progress.close()
            else:
                QApplication.restoreOverrideCursor()

        # One dialog for the batch, not one per run: fifty modal pop-ups is
        # not a way to tell someone their donor file is missing.
        if failed:
            head = failed[:8]
            more = (f"\n…and {len(failed) - len(head)} more."
                    if len(failed) > len(head) else "")
            QMessageBox.warning(
                self, "Calibration",
                "Could not apply the calibration to "
                f"{len(failed)} run(s):\n\n" + "\n".join(head) + more)
        self._schedule_replot()

    def _reload_file_entry(self, entry):
        """Re-read the ASDF backing ``entry`` from disk and refresh
        the plot. Useful for ongoing scans that are still being
        written: pressing reload picks up the new events without
        having to remove and re-open the file.

        The header metadata (cooler, laser, mass, date) is treated
        as immutable for an ongoing scan; only the per-event arrays
        and the run-time label refresh. Merged entries never emit
        this signal (their reload button is hidden), so the entry
        passed in here is always a real ASDF-backed FileEntry or
        SplitFileEntry.
        """
        if _get_clstools() is None:
            QMessageBox.warning(
                self, "Reload",
                "clstools is required to reload ASDF data.")
            return
        # Indicate work in progress on the reload icon -- ASDFs
        # for long runs can take a second or two.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._load_cls_data(entry)
            entry.update_detail()
        except Exception as exc:
            QMessageBox.warning(
                self, "Reload failed",
                f"Could not re-read {entry.filepath}:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._schedule_replot()
        parent = self.window()
        if hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Reloaded: {entry.filepath}", 3000)

    def _select_file_entry(self, entry):
        """Highlight a single file entry, deselecting others.

        Cosmetic only: nothing in ``_replot`` reads ``entry.selected`` (it
        keys off ``check.isChecked()`` / colour / linestyle), so a highlight
        changes no plotted data and must NOT trigger a full re-bin + redraw of
        every checked entry. The cooler/laser auto-fill buttons read the
        highlighted entry on demand via ``_get_selected_entry``.
        """
        for fe in self._file_entries:
            fe.selected = (fe is entry)

    def _get_selected_entry(self):
        """Return the highlighted file entry, or first checked, or first."""
        for fe in self._file_entries:
            if fe.selected:
                return fe
        for fe in self._file_entries:
            if fe.check.isChecked():
                return fe
        return self._file_entries[0] if self._file_entries else None

    def _auto_fill_laser(self):
        """Copy laser setpoint from the selected file entry."""
        fe = self._get_selected_entry()
        if fe:
            self._laser_override.setValue(fe.laser_sp)

    def _auto_fill_cooler(self):
        """Copy cooler voltage from the selected file entry."""
        fe = self._get_selected_entry()
        if fe:
            self._cooler_override.setValue(fe.cooler_v)

    def _on_cooler_override_toggled(self, checked):
        """Enable/disable the cooler voltage field based on its tick."""
        self._cooler_override.setEnabled(checked)
        self._cooler_auto_btn.setEnabled(checked)
        self._schedule_replot()

    def _on_laser_override_toggled(self, checked):
        """Enable/disable the laser setpoint field based on its tick."""
        self._laser_override.setEnabled(checked)
        self._laser_auto_btn.setEnabled(checked)
        self._schedule_replot()

    def _default_file_for_axis(self):
        """Pick the file whose values to use when no specific entry is
        in scope (model overlays, merged entries shown in voltage axes).
        Selected first, then any checked, then any usable file. Skips
        merged and split entries — model overlays should anchor on a
        full-range parent run, not a sub-range."""
        skip = (MergedFileEntry, SplitFileEntry)
        for fe in self._file_entries:
            if isinstance(fe, skip):
                continue
            if getattr(fe, 'selected', False) and fe.cooler_v > 0:
                return fe
        for fe in self._file_entries:
            if isinstance(fe, skip):
                continue
            if fe.check.isChecked() and fe.cooler_v > 0:
                return fe
        for fe in self._file_entries:
            if isinstance(fe, skip):
                continue
            if fe.cooler_v > 0:
                return fe
        return None

    def _cooler_for_entry(self, entry):
        """Cooler voltage to use for this entry: override if its tick is
        on, else the file's own value (falling back to override if file
        value is missing)."""
        if self._cooler_override_enabled.isChecked():
            return self._cooler_override.value()
        val = getattr(entry, 'cooler_v', 0) or 0
        if val > 0:
            return val
        # code review 2026-06-02, cooler-laser-silent-default-fallback:
        # warn once per run that a fabricated default beam energy is used.
        fallback = self._cooler_override.value()
        self._warn_metadata_fallback(entry, "cooler voltage",
                                     f"{fallback:.2f} V")
        return fallback

    def _laser_for_entry(self, entry):
        """Laser setpoint to use for this entry: override if its tick is
        on, else the file's own value (falling back to override if file
        value is missing)."""
        if self._laser_override_enabled.isChecked():
            return self._laser_override.value()
        val = getattr(entry, 'laser_sp', 0) or 0
        if val > 0:
            return val
        # code review 2026-06-02, cooler-laser-silent-default-fallback:
        # warn once per run that a fabricated default setpoint is used.
        fallback = self._laser_override.value()
        self._warn_metadata_fallback(entry, "laser setpoint",
                                     f"{fallback:.6f} cm⁻¹")
        return fallback

    def _warn_metadata_fallback(self, entry, what, default_str):
        """Log once per run that missing/zero ``what`` metadata caused a
        fabricated default (``default_str``) to be substituted, so the
        beam-energy / frequency binning isn't silently built on guessed
        physics. code review 2026-06-02, cooler-laser-silent-default-
        fallback."""
        run = getattr(entry, 'run_number', None)
        key = (run, what)
        seen = getattr(self, '_metadata_fallback_warned', None)
        if seen is None:
            seen = set()
            self._metadata_fallback_warned = seen
        if key in seen:
            return
        seen.add(key)
        _log.warning(
            "run %s: missing/zero %s metadata; substituting default %s",
            run, what, default_str)

    def _default_cooler_laser(self):
        """Cooler/laser for contexts without a specific entry (model
        overlays, merged entries shown in voltage modes). Each parameter
        independently follows its own override tick."""
        fe = self._default_file_for_axis()
        if self._cooler_override_enabled.isChecked():
            cooler = self._cooler_override.value()
        elif fe is not None and fe.cooler_v > 0:
            cooler = fe.cooler_v
        else:
            cooler = self._cooler_override.value()
        if self._laser_override_enabled.isChecked():
            laser = self._laser_override.value()
        elif fe is not None and fe.laser_sp > 0:
            laser = fe.laser_sp
        else:
            laser = self._laser_override.value()
        return cooler, laser

    # ── Isotope label ────────────────────────────────────────────

    def _on_z_changed(self, z):
        self._update_isotope_label()
        self._refresh_mass_display()
        self._schedule_replot()

    def _on_a_changed(self, a):
        self._update_isotope_label()
        self._refresh_mass_display()
        self._schedule_replot()

    def _update_isotope_label(self):
        if _HAS_PERIODICTABLE:
            try:
                elem = periodictable.elements[self._z_spin.value()]
                self._isotope_label.setText(
                    f"{self._a_spin.value()}{elem.symbol}")
            except (KeyError, IndexError):
                self._isotope_label.setText("")
        else:
            self._isotope_label.setText("")

    def _lookup_mass(self):
        """Mass in amu from the periodictable database (or A as a fallback)."""
        if _HAS_PERIODICTABLE:
            try:
                iso = periodictable.elements[
                    self._z_spin.value()][self._a_spin.value()]
                return float(iso.mass)
            except (KeyError, IndexError, AttributeError):
                pass
        return float(self._a_spin.value())

    def _on_mass_override_toggled(self, checked):
        # When override turns off, snap back to the database value;
        # when it turns on, leave whatever number is currently shown
        # so the user can edit it. Read-only state mirrors the toggle.
        self._mass_spin.setReadOnly(not checked)
        self._mass_spin.setButtonSymbols(
            QDoubleSpinBox.ButtonSymbols.UpDownArrows if checked
            else QDoubleSpinBox.ButtonSymbols.NoButtons)
        if not checked:
            self._mass_spin.blockSignals(True)
            self._mass_spin.setValue(self._lookup_mass())
            self._mass_spin.blockSignals(False)
        self._schedule_replot()

    def _refresh_mass_display(self):
        """Update the displayed mass from the database (skipped when override is on)."""
        if self._mass_override.isChecked():
            return
        self._mass_spin.blockSignals(True)
        self._mass_spin.setValue(self._lookup_mass())
        self._mass_spin.blockSignals(False)

    def _get_isotope_mass(self):
        if self._mass_override.isChecked():
            return float(self._mass_spin.value())
        return self._lookup_mass()

    # ── Fast numpy-based data operations ─────────────────────────

    def _bin_spectrum(self, entry, pmt_gate, tof_gate, voltage_mode="calibrated"):
        """Legacy numpy-based binning. Used by _compute_merge only.

        The Pre-Analysis spectrum subplot now routes through
        gui.analysis.binning.compute_binned (see _replot). This method is
        kept because _compute_merge needs per-entry (v_means, counts) at
        the native DAC-step resolution to combine multiple runs onto a
        shared grid; rewriting that path is out of scope for the binning
        unification work.

        Returns (v_values, counts).

        voltage_mode:
            "raw"        - raw DAC set-voltages (for "Voltage" mode)
            "readback"   - CalSet→CalReadback mapped voltages (for
                           "Calibrated voltage" / "Calibrated beam energy")
            "calibrated" - clstools calibrated voltages (fallback)
        """
        mask = np.isin(entry.np_tdc, pmt_gate)
        if tof_gate:
            mask &= (entry.np_tof >= tof_gate[0]) & \
                     (entry.np_tof <= tof_gate[1])
        # Virtual-split entries carry a permanent raw-voltage gate so
        # the preview shows only the sub-range — keeps the displayed
        # spectrum in lock-step with what the fitter will see.
        if isinstance(entry, SplitFileEntry):
            mask &= (entry.np_dv >= entry.split_lo) & \
                     (entry.np_dv <= entry.split_hi)

        fv = entry.np_v[mask]
        fdv = entry.np_dv[mask]

        if len(fv) == 0:
            return np.array([]), np.array([])

        unique_dv, inv, counts = np.unique(
            fdv, return_inverse=True, return_counts=True)

        if voltage_mode == "raw":
            # Mean raw DAC voltage per step
            dv_sums = np.zeros(len(unique_dv))
            np.add.at(dv_sums, inv, fdv)
            v_out = dv_sums / counts
        elif voltage_mode == "readback":
            # Map each unique DAC voltage to its CalReadback value
            cal_set = entry.np_cal_set
            cal_rb = entry.np_cal_readback
            if cal_set is not None and len(cal_set) > 0:
                # For each unique DAC value, find nearest CalSet entry
                dv_means = np.zeros(len(unique_dv))
                np.add.at(dv_means, inv, fdv)
                dv_means = dv_means / counts
                idx = np.array([np.argmin(np.abs(cal_set - dv))
                                for dv in dv_means])
                v_out = cal_rb[idx]
            else:
                # Fallback to clstools calibrated
                v_sums = np.zeros(len(unique_dv))
                np.add.at(v_sums, inv, fv)
                v_out = v_sums / counts
        else:
            # clstools calibrated voltage
            v_sums = np.zeros(len(unique_dv))
            np.add.at(v_sums, inv, fv)
            v_out = v_sums / counts

        order = np.argsort(v_out)
        return v_out[order], counts[order]

    def _voltage_to_frequency(self, v_beam, mass_amu, harmonic, laser_cm1):
        """Convert beam energy voltage to frequency (MHz) via Doppler shift.

        v_beam is the total acceleration voltage (cooler - scanning), not the
        scanning voltage alone.
        """
        nu_laser_MHz = laser_cm1 * harmonic * C_LIGHT * 100.0 / 1e6
        beta = beta_from_voltage(v_beam, mass_amu, 1)
        return nu_seen_by_ion(nu_laser_MHz, beta, 'anti-collinear')

    # ── Plotting ─────────────────────────────────────────────────

    def _replot(self, spectrum_only=False):
        # ``spectrum_only`` is the interactive gate-drag fast path: while
        # the user drags a TOF or timestamp gate, only the spectrum
        # changes (the gate is a filter on the binned counts). The TOF and
        # timestamp histograms depend on the PMT gate alone, not on the
        # gate being dragged, and the SpanSelector already paints its own
        # live rectangle -- so we skip clearing/redrawing those two
        # canvases and the calibration panels entirely. Each costs a full
        # matplotlib figure draw (~150-300 ms); skipping them is what
        # makes the drag feel responsive. A full _replot() on mouse
        # release restores the gate overlays in their final position.
        self._replot_timer.stop()

        # _ax.clear() destroys the cached spectrum Line2D, so the fast-path
        # cache is invalid from here. The full path rebuilds it below once
        # the new line is drawn; the dask spectrum-only fallback leaves it
        # None (so the next gate move re-enters the fallback).
        self._fast = None
        self._fast_bg = None

        self._ax.clear()
        if not spectrum_only:
            self._tof_ax.clear()
            self._ts_ax.clear()

        xaxis_mode = self._xaxis_combo.currentText()
        harmonic = self._harmonic.value()
        offset = self._get_fundamental_cm()
        normalize = self._normalize.isChecked()
        mass = self._get_isotope_mass()
        pmt_gate = [i + 1 for i, cb in enumerate(self._channels)
                    if cb.isChecked()]
        tof_gate = None
        if self._tof_enable.isChecked():
            tof_gate = [self._tof_lo.value(), self._tof_hi.value()]

        # Fast-path eligibility: exactly one plain (non-merged, non-split)
        # loaded entry. Merged spectra have no per-event arrays; splits
        # carry a permanent v_gate the in-memory histogram doesn't model.
        # Built + self-checked below once the entry's line is drawn; any
        # other view leaves self._fast None and gate drags use the dask
        # spectrum-only fallback.
        _checked_loaded = [fe for fe in self._file_entries
                           if fe.check.isChecked() and fe.is_loaded]

        def _has_scan_exclusions(fe):
            # The gate-drag fast cache is built UNGATED and does not apply the
            # per-file scan filter, so a file with excluded scans would show
            # them in the single-file view (and mismatch the Analysis fit,
            # which keeps the exclusion). Disable the fast path then and fall
            # back to the dask spectrum path, which honors the filter (code
            # review 2026-06-02, fast-cache-ignores-scan-filter).
            from gui.scan_filter import get_registry as _sfreg
            p = getattr(fe, "filepath", None)
            return bool(p and _sfreg().get(p))

        _fast_one = (_checked_loaded[0] if (
            not spectrum_only and len(_checked_loaded) == 1
            and not isinstance(_checked_loaded[0],
                               (MergedFileEntry, SplitFileEntry))
            and not _has_scan_exclusions(_checked_loaded[0]))
            else None)

        _XLABEL_MAP = {
            "Voltage": "Scanning voltage (V)",
            "Calibrated voltage": "Calibrated voltage (V)",
            "Calibrated beam energy": "Beam energy (V)",
            "Wavenumber": "Wavenumber (cm$^{-1}$)",
            "Frequency": "Frequency (MHz)",
        }
        xlabel = _XLABEL_MAP.get(xaxis_mode, "Frequency (MHz)")

        if not pmt_gate:
            self._finalize_axes(xlabel, normalize, spectrum_only=spectrum_only)
            return

        for entry in self._file_entries:
            if not entry.check.isChecked() or not entry.is_loaded:
                continue

            # Merged entries: convert stored x to the current display axis
            if isinstance(entry, MergedFileEntry):
                cooler_v, laser_sp = self._default_cooler_laser()
                x = entry.merged_x.copy()
                y = entry.merged_y.copy()

                if entry.merge_domain == "voltage":
                    # stored x = scanning voltage (V)
                    if xaxis_mode == "Voltage":
                        xlabel = "Scanning voltage (V)"
                    elif xaxis_mode == "Calibrated voltage":
                        xlabel = "Calibrated voltage (V)"
                    elif xaxis_mode == "Calibrated beam energy":
                        x = cooler_v - x
                        xlabel = "Beam energy (V)"
                    elif xaxis_mode == "Wavenumber":
                        freq_MHz = self._voltage_to_frequency(
                            cooler_v - x, mass, harmonic, laser_sp)
                        x = freq_MHz * 1e6 / (C_LIGHT * 100.0) \
                            - offset * harmonic
                        xlabel = "Wavenumber (cm$^{-1}$)"
                    else:  # Frequency
                        freq_MHz = self._voltage_to_frequency(
                            cooler_v - x, mass, harmonic, laser_sp)
                        x = freq_MHz - offset * harmonic * C_LIGHT \
                            * 100.0 / 1e6
                        xlabel = "Frequency (MHz)"
                else:
                    # stored x = frequency (MHz, absolute)
                    if xaxis_mode in ("Voltage", "Calibrated voltage",
                                      "Calibrated beam energy"):
                        # Convert freq back to voltage via _mhz_to_xaxis
                        x = self._mhz_to_xaxis(
                            x, xaxis_mode, harmonic, offset,
                            mass, cooler_v, laser_sp)
                        if xaxis_mode == "Voltage":
                            xlabel = "Scanning voltage (V)"
                        elif xaxis_mode == "Calibrated voltage":
                            xlabel = "Calibrated voltage (V)"
                        else:
                            xlabel = "Beam energy (V)"
                    elif xaxis_mode == "Wavenumber":
                        x = x * 1e6 / (C_LIGHT * 100.0) \
                            - offset * harmonic
                        xlabel = "Wavenumber (cm$^{-1}$)"
                    else:  # Frequency
                        x = x - offset * harmonic * C_LIGHT * 100.0 / 1e6
                        xlabel = "Frequency (MHz)"

                if normalize and y.max() > 0:
                    y = y / y.max()
                self._ax.step(x, y, where='mid', color=self._display_color(entry),
                              alpha=entry.alpha, linestyle=entry.linestyle,
                              linewidth=1.5, label=entry.run_number)
                # TOF for a merged entry: per-event arrays were
                # collapsed at merge time, but ``compute_merged_spectrum``
                # captured a per-source TOF histogram in
                # ``per_run[i]["tof_index"/"tof_counts"]``. Overlay each
                # source's TOF so the user can still see the timing
                # structure that drove the chosen gate. PA-side
                # legacy merges (no per_run) just get an empty TOF
                # panel; the placeholder text below makes that clear.
                per_run = [] if spectrum_only else (
                    getattr(entry, "per_run", None) or [])
                for rd in per_run:
                    # ``rd.get(...) or []`` would raise on a numpy
                    # array (ambiguous truth value); branch on None
                    # explicitly and let ``len(arr) == 0`` handle the
                    # empty case below.
                    ti_raw = rd.get("tof_index")
                    tc_raw = rd.get("tof_counts")
                    if ti_raw is None or tc_raw is None:
                        continue
                    ti = np.asarray(ti_raw, dtype=float)
                    tc = np.asarray(tc_raw, dtype=float)
                    if len(ti) > 0 and len(tc) == len(ti):
                        self._tof_ax.step(
                            ti, tc, where='mid', alpha=0.6,
                            linewidth=1.2,
                            label=f"src {rd.get('run_num', '?')}")
                continue

            # Per-entry cooler/laser (override if on, else file's own)
            cooler_v = self._cooler_for_entry(entry)
            laser_sp = self._laser_for_entry(entry)

            # Build cfg for compute_binned.
            cfg = self._binning_cfg(entry, pmt_gate, tof_gate)
            bin_mode = cfg["bin_mode"]

            if entry.cls_data is None:
                continue

            # Per-file scan filter: the registry is keyed by the actual
            # ASDF path. Splits share their parent ASDF, so use the
            # parent_path for them. The context manager is a no-op when
            # the registry has no entry for this file.
            from gui.scan_filter import (
                filter_data_for_binning,
                gate_data_by_timestamp,
                get_registry as _get_sf_registry)
            sf_path = (getattr(entry, "parent_path", None)
                       if isinstance(entry, SplitFileEntry)
                       else entry.filepath)
            sf_excluded = _get_sf_registry().get(sf_path) if sf_path else set()
            # Time gate (absolute seconds) filters ONLY the spectrum, so
            # the user can watch counts evolve over a time window. Nested
            # inside the scan filter; both restore data.Run on exit.
            ts_gate = self._ts_gate_seconds(entry)

            # ── Per-entry binned-result cache ──────────────────────────
            # Re-binning a clstools/dask frame (and, in Frequency mode, the
            # per-event Compute_WL) is the dominant replot cost. Skip it for
            # any entry whose binning inputs are unchanged -- e.g. enabling a
            # 2nd file, toggling an override that affects only another entry,
            # changing the display axis without flipping bin mode, or a
            # cosmetic change. The key captures every input ``compute_binned``,
            # the scan-filter / time-gate context, and the Frequency-mode
            # physics prep depend on; ``_display_x`` runs AFTER the cache and
            # is cheap, so the display axis is intentionally NOT in the key.
            #
            # MAINTENANCE: this key MUST contain every input that changes the
            # *computed* binned result -- the physics interpretation
            # (cooler_v, laser_sp, mass, harmonic), all gates (pmt / tof /
            # time / scan-filter / v_gate), and every cfg binning field. If you
            # add a new per-file knob that feeds compute_binned or the
            # Frequency-mode prep (e.g. charge state, reference shift, a
            # cooler-correction mode), ADD IT TO THIS KEY or a stale plot can
            # be served. Inputs that only re-skin the display axis belong in
            # _display_x (recomputed on every replot), NOT here. The sibling
            # guard _freq_prep_key in _prepare_frequency_data must stay in sync.
            _tof_key = tuple(cfg["tof_gate"]) if cfg["tof_gate"] else None
            # The voltage calibration decides what V every event has, so it
            # feeds compute_binned as directly as any gate does. Fingerprinted
            # rather than embedded so the key stays hashable.
            from gui.calibration import get_registry as _get_cal_reg
            from gui.calibration import spec_fingerprint as _cal_fp
            _cal_path = (entry.parent_path
                         if isinstance(entry, SplitFileEntry)
                         else entry.filepath)
            _cal_key = _cal_fp(_get_cal_reg().get(_cal_path))
            _bin_key = (
                cfg["bin_mode"], cfg["x_column"], cfg["yerr_mode"],
                cfg["xerr_mode"], cfg["bin_definition"], cfg["bin_count"],
                cfg["bin_width_mhz"], cfg.get("step_multiple", 1),
                tuple(cfg["pmt_gate"]), _tof_key,
                cfg["v_gate"], ts_gate, frozenset(sf_excluded),
                round(cooler_v, 6), round(laser_sp, 9),
                round(mass, 9), int(harmonic), _cal_key)
            _cached = getattr(entry, "_bin_cache", None)
            if _cached is not None and _cached["key"] == _bin_key:
                x_bin = _cached["x"]
                y = _cached["y"].copy()   # copy: normalize mutates y below
                yerr = _cached["yerr"]
            else:
                # Frequency mode needs Compute_WL on the per-entry data with
                # the current cooler/laser/mass/harmonic (itself guarded).
                if bin_mode == "Frequency":
                    self._prepare_frequency_data(
                        entry, cooler_v, laser_sp, mass, harmonic)
                try:
                    with filter_data_for_binning(
                            entry.cls_data, sf_excluded, asdf_path=sf_path), \
                            gate_data_by_timestamp(entry.cls_data, ts_gate):
                        out = compute_binned(entry.cls_data, cfg)
                except Exception:
                    _label = (getattr(entry, "run_number", None)
                              or getattr(entry, "label", None) or "a file")
                    _log.warning("Binning failed for %s; its spectrum is "
                                 "omitted from the Pre-Analysis plot",
                                 _label, exc_info=True)
                    continue
                x_bin = out["x"]
                y = out["y"].astype(float)
                yerr = out["yerr"]
                entry._bin_cache = {
                    "key": _bin_key, "x": x_bin, "y": y.copy(), "yerr": yerr}

            if len(x_bin) == 0:
                continue

            # Convert bin centers to the user-selected display x-axis.
            x, xlabel = self._display_x(
                x_bin, bin_mode, xaxis_mode, entry,
                cooler_v, laser_sp, mass, harmonic, offset)
            if x is None:
                # Incompatible display × bin mode combination; Task 6's
                # combo restriction should normally prevent this.
                continue

            if normalize and y.max() > 0:
                ymax = y.max()
                y = y / ymax
                if yerr is not None:
                    yerr = yerr / ymax

            if yerr is None:
                _ln = self._ax.step(
                    x, y, where='mid', color=self._display_color(entry),
                    alpha=entry.alpha, linestyle=entry.linestyle,
                    linewidth=1.5,
                    label=f"run_{entry.run_number}")
                # Arm the gate-drag fast path for this single-entry view.
                if entry is _fast_one:
                    self._fast = self._try_build_fast_cache(
                        entry, cfg, bin_mode, _ln[0], normalize,
                        xaxis_mode, cooler_v, laser_sp, mass,
                        harmonic, offset)
            else:
                self._ax.errorbar(
                    x, y, yerr=yerr, fmt='o',
                    color=self._display_color(entry), alpha=entry.alpha,
                    linestyle=entry.linestyle, linewidth=1.0,
                    markersize=3, capsize=2,
                    label=f"run_{entry.run_number}")

            if spectrum_only:
                # Gate-drag fast path: TOF/timestamp histograms are
                # unchanged by the gate, so skip their per-event masks,
                # np.histogram calls, and stairs draws.
                continue

            # TOF histogram (pure numpy, 1 µs bins centered on integers)
            tof_mask = np.isin(entry.np_tdc, pmt_gate)
            tof_vals = entry.np_tof[tof_mask]
            if len(tof_vals) > 0:
                binsize = float(self._tof_binsize.value()) or 1.0
                tof_bins = np.arange(
                    tof_vals.min() - 0.5 * binsize,
                    tof_vals.max() + 0.5 * binsize,
                    binsize)
                if len(tof_bins) > 1:
                    tof_counts, _ = np.histogram(tof_vals, bins=tof_bins)
                    # stairs() draws a clean histogram outline straight
                    # from the bin EDGES -- no diagonal connectors
                    # between bins.
                    self._tof_ax.stairs(
                        tof_counts, tof_bins,
                        color=self._display_color(entry), alpha=entry.alpha,
                        linestyle=entry.linestyle, linewidth=1.5,
                        label=f"run_{entry.run_number}")

            # Timestamp histogram with configurable unit and bin size
            ts_mask = np.isin(entry.np_tdc, pmt_gate)
            ts_vals = entry.np_ts[ts_mask]
            if len(ts_vals) > 0:
                # Convert from raw seconds to display unit
                ts_unit = self._ts_unit.currentText()
                if ts_unit == "Minutes":
                    ts_display = (ts_vals - ts_vals.min()) / 60.0
                elif ts_unit == "Hours":
                    ts_display = (ts_vals - ts_vals.min()) / 3600.0
                elif ts_unit == "Days":
                    ts_display = (ts_vals - ts_vals.min()) / 86400.0
                else:  # Seconds
                    ts_display = ts_vals - ts_vals.min()

                binsize = self._ts_binsize.value()
                ts_bins = np.arange(
                    ts_display.min() - 0.5 * binsize,
                    ts_display.max() + 0.5 * binsize,
                    binsize)
                if len(ts_bins) > 1:
                    ts_counts, _ = np.histogram(ts_display, bins=ts_bins)
                    self._ts_ax.stairs(
                        ts_counts, ts_bins,
                        color=self._display_color(entry), alpha=entry.alpha * 0.7,
                        linestyle=entry.linestyle, linewidth=1.5,
                        label=f"run_{entry.run_number}")

        # Merged-only annotation: when every checked entry is a
        # MergedFileEntry, the TOF / Timestamp / Calibrations / Cooler
        # panels can't be reconstructed (per-event arrays were
        # collapsed at merge time). Drop a clear placeholder so the
        # empty panels aren't confusing. The TOF panel may still have
        # the per-source histogram overlay from the Merged branch
        # above; the placeholder only fires when even that's empty.
        checked = [fe for fe in self._file_entries
                   if fe.check.isChecked() and fe.is_loaded]
        only_merged = checked and all(
            isinstance(fe, MergedFileEntry) for fe in checked)
        if only_merged and not spectrum_only:
            if not self._tof_ax.lines:
                self._tof_ax.text(
                    0.5, 0.5,
                    "TOF not available for merged spectra\n"
                    "(per-event timing was collapsed at merge time;\n"
                    "exports from compute_merged_spectrum carry\n"
                    "per-source TOFs and would show here)",
                    transform=self._tof_ax.transAxes,
                    ha="center", va="center", fontsize=8,
                    color="#888", style="italic")
            self._ts_ax.text(
                0.5, 0.5,
                "Timestamp not available for merged spectra\n"
                "(per-event time was collapsed at merge time)",
                transform=self._ts_ax.transAxes,
                ha="center", va="center", fontsize=8,
                color="#888", style="italic")

        # Scan overlay: only meaningful when a single non-merged file
        # is shown; multi-file overlays would conflate scan numbers
        # across runs and confuse the right-click filter target.
        if not spectrum_only:
            self._maybe_draw_scan_overlay()

        self._finalize_axes(xlabel, normalize, spectrum_only=spectrum_only)

    # ── Scan-overlay helpers ────────────────────────────────────────

    def _on_ts_unit_changed(self):
        """Rescale the time-gate spinboxes so a display-unit change preserves
        the ABSOLUTE time window.

        The lo/hi/binsize spinboxes hold values in the selected unit, so
        switching Seconds->Minutes etc. would otherwise leave the numbers
        unchanged and silently move the gate to a 60x-different physical slice
        (code review 2026-06-02, ts-unit-change-shifts-gate). Rescale by
        old_div/new_div (so value*divisor, i.e. absolute seconds, is held
        constant) before replotting.
        """
        new_div = self._ts_unit_divisor()
        old_div = getattr(self, "_ts_prev_divisor", 1.0)
        if new_div > 0 and old_div != new_div:
            factor = old_div / new_div
            for sb in (self._ts_lo, self._ts_hi, self._ts_binsize):
                sb.blockSignals(True)
                sb.setValue(sb.value() * factor)
                sb.blockSignals(False)
        self._ts_prev_divisor = new_div
        self._schedule_replot()

    def _ts_unit_divisor(self):
        """Conversion factor from seconds to the user-picked timestamp
        unit. Mirrors the inline conversion in the histogram loop so
        the overlay axes always agree with the bars."""
        unit = self._ts_unit.currentText()
        if unit == "Minutes":
            return 60.0
        if unit == "Hours":
            return 3600.0
        if unit == "Days":
            return 86400.0
        return 1.0

    def _scan_overlay_target(self):
        """Pick the single non-merged checked entry that the overlay
        applies to, or None when the overlay can't be drawn cleanly.

        Returns the entry (a FileEntry or SplitFileEntry) when:
          - the "Show scans" tickbox is on
          - exactly one non-merged entry is currently checked
          - that entry has both the bunch column and scan_meta loaded

        Otherwise returns None. The caller treats None as "skip the
        overlay this frame".
        """
        if not self._ts_show_scans.isChecked():
            return None
        candidates = [fe for fe in self._file_entries
                      if fe.check.isChecked()
                      and not isinstance(fe, MergedFileEntry)
                      and fe.is_loaded
                      and getattr(fe, "np_bunch", None) is not None
                      and getattr(fe, "np_ts", None) is not None
                      and getattr(fe, "scan_meta", None)]
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _compute_scan_starts(self, entry):
        """Compute per-scan start timestamps (seconds since file start)
        plus the matching scan indices. Returns ``(starts, scan_idx)``
        where both arrays are aligned in scan order; an empty file or
        unfilled scan_meta returns ``([], [])``.

        Cached per entry keyed by ``np_bunch`` identity (set once at load and
        never mutated), so the per-event ``derive_scan_indices`` runs once
        instead of on every replot/overlay redraw."""
        cached = getattr(entry, "_scan_starts_cache", None)
        if cached is not None and cached[0] is entry.np_bunch:
            return cached[1], cached[2]
        from gui.scan_filter import derive_scan_indices
        meta = entry.scan_meta or {}
        idx = derive_scan_indices(
            entry.np_bunch,
            meta.get("scanning_ranges"),
            meta.get("step_size"),
            meta.get("bunches_per_channel"))
        if len(idx) == 0:
            entry._scan_starts_cache = (
                entry.np_bunch, np.array([]), np.array([]))
            return entry._scan_starts_cache[1], entry._scan_starts_cache[2]
        order = np.argsort(idx, kind="stable")
        i_sorted = idx[order]
        ts_sorted = entry.np_ts[order]
        edges = np.flatnonzero(np.diff(i_sorted)) + 1
        starts_at = np.concatenate(([0], edges))
        scan_nums = i_sorted[starts_at]
        # Start time of each scan = first event in that scan, relative
        # to the file's earliest event (matches the histogram x-axis).
        t0 = float(ts_sorted[0])
        starts = ts_sorted[starts_at] - t0
        entry._scan_starts_cache = (entry.np_bunch, starts, scan_nums)
        return starts, scan_nums

    def _on_show_scans_toggled(self, _checked):
        """Show/hide the scan overlay WITHOUT a full replot.

        The overlay lives only on the timestamp axis, so toggling it must not
        clear+redraw the spectrum / TOF / 6 calibration panels -- that full
        replot was the cause of the multi-second "enable scan view" freeze.
        Remove just the tracked overlay artists, redraw the overlay if now on,
        and repaint only the timestamp canvas."""
        for art in getattr(self, "_scan_overlay_artists", []):
            try:
                art.remove()
            except Exception:
                pass
        self._scan_overlay_artists = []
        self._maybe_draw_scan_overlay()  # no-op when toggled off (target None)
        self._ts_canvas.draw_idle()

    def _maybe_draw_scan_overlay(self):
        """Draw scan-boundary marks on the timestamp plot for the
        single non-merged file currently selected. No-op otherwise."""
        entry = self._scan_overlay_target()
        if entry is None:
            return
        # Track artists so the Show-scans toggle can remove just the overlay
        # (no full replot). In the full-replot path the axis was already
        # cleared, so the previous refs are dead -- reset the list here.
        self._scan_overlay_artists = []
        starts_s, scan_nums = self._compute_scan_starts(entry)
        if len(starts_s) == 0:
            return

        from gui.scan_filter import get_registry as _get_sf_registry
        scan_path = (getattr(entry, "parent_path", None)
                     or entry.filepath)
        excluded = _get_sf_registry().get(scan_path)

        divisor = self._ts_unit_divisor()
        starts_disp = starts_s / divisor

        # One vertical tick per scan; thin grey for included, thicker
        # red for excluded. Avoid one Line2D per scan when there are
        # hundreds of scans -- vlines is a single LineCollection.
        included_mask = np.array(
            [int(s) not in excluded for s in scan_nums])
        ylim = self._ts_ax.get_ylim()
        if np.any(included_mask):
            self._scan_overlay_artists.append(self._ts_ax.vlines(
                starts_disp[included_mask], ylim[0], ylim[1],
                colors="#888", linewidths=0.4, alpha=0.5,
                zorder=0))
        if np.any(~included_mask):
            self._scan_overlay_artists.append(self._ts_ax.vlines(
                starts_disp[~included_mask], ylim[0], ylim[1],
                colors="#c33", linewidths=0.9, alpha=0.7,
                zorder=0))
            # Shade excluded scans up to the next scan's start so the
            # excluded ranges read as bands, not isolated lines.
            for i, drop in enumerate(~included_mask):
                if not drop:
                    continue
                lo = starts_disp[i]
                hi = (starts_disp[i + 1]
                      if i + 1 < len(starts_disp) else ylim[1])
                self._scan_overlay_artists.append(self._ts_ax.axvspan(
                    lo, hi, color="#c33", alpha=0.10, zorder=0))

        # Scan-number labels only when there aren't too many to fit;
        # at 500 scans on a typical timestamp width every label would
        # collide. Show ~30 labels max, decimated uniformly.
        n = len(starts_disp)
        if n > 0:
            stride = max(1, n // 30)
            for i in range(0, n, stride):
                self._scan_overlay_artists.append(self._ts_ax.text(
                    starts_disp[i], ylim[1],
                    f"{int(scan_nums[i])}",
                    fontsize=6, color="#666",
                    ha="left", va="top", clip_on=True,
                    zorder=1))
        # Reapply ylim -- vlines and text can stretch the autoscaled
        # bounds, which would hide the histogram step plot.
        self._ts_ax.set_ylim(ylim)

    def _exclude_scans_in_view(self):
        """Add every scan whose start time is in the current x-axis
        range to the file's exclusion set. Pan/zoom in the timestamp
        plot first; the button reads the current xlim."""
        entry = self._scan_overlay_target()
        if entry is None:
            QMessageBox.information(
                self, "Exclude scans in view",
                "Turn on \"Show scans\" and check exactly one non-merged "
                "file before using this action.")
            return
        starts_s, scan_nums = self._compute_scan_starts(entry)
        if len(starts_s) == 0:
            QMessageBox.information(
                self, "Exclude scans in view",
                "No scans were derived for this file. Check that its "
                "ASDF has ScanningRanges and StepSize.")
            return
        divisor = self._ts_unit_divisor()
        starts_disp = starts_s / divisor

        xlim_lo, xlim_hi = self._ts_ax.get_xlim()
        mask = (starts_disp >= xlim_lo) & (starts_disp <= xlim_hi)
        new_excluded = {int(s) for s in scan_nums[mask]}
        if not new_excluded:
            QMessageBox.information(
                self, "Exclude scans in view",
                "No scan starts fall inside the current view range. "
                "Pan or zoom so at least one scan begins inside the "
                "visible window, then try again.")
            return

        from gui.scan_filter import get_registry as _get_sf_registry
        reg = _get_sf_registry()
        scan_path = (getattr(entry, "parent_path", None)
                     or entry.filepath)
        existing = reg.get(scan_path)
        # Count what's actually newly excluded so the status hint
        # doesn't double-count scans that were already filtered out.
        newly_added = new_excluded - existing
        merged = existing | new_excluded
        reg.set(scan_path, merged)
        # filters_changed (emitted inside reg.set) already retriggers
        # _schedule_replot, so no manual draw call needed.
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(
                f"Excluded {len(newly_added)} new scan(s) in view "
                f"({len(merged)} total now excluded for "
                f"{os.path.basename(scan_path)})", 4000)

    @staticmethod
    def _on_scroll_zoom(event):
        """Zoom the hovered plot in/out around the cursor on mouse-wheel.

        Scroll up = zoom in, scroll down = zoom out, centred on the
        cursor so the point under the pointer stays put. Works on
        whichever axes the cursor is over; no toolbar button needed.
        """
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        base = 1.3
        scale = (1.0 / base) if event.button == "up" else base
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xc, yc = event.xdata, event.ydata
        # Keep the cursor's data point fixed: shrink/grow each side by
        # `scale` about (xc, yc).
        ax.set_xlim(xc - (xc - x0) * scale, xc + (x1 - xc) * scale)
        ax.set_ylim(yc - (yc - y0) * scale, yc + (y1 - yc) * scale)
        ax.figure.canvas.draw_idle()

    @staticmethod
    def _apply_ticks(ax):
        """Denser major ticks + minor ticks on both axes of a plot.

        Called after the limits are finalised so the locators tick over
        the real data range. ~10 major divisions per axis (vs
        matplotlib's sparse default) with 5 minor subdivisions between
        them makes it easy to read positions off the spectrum / ToF /
        timestamp panes for initial-guess estimation.
        """
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=2.5)

    def _finalize_axes(self, xlabel, normalize, spectrum_only=False):
        """Set labels, draw models, refresh canvases.

        ``spectrum_only`` (gate-drag fast path) finalizes and redraws ONLY
        the spectrum axis/canvas, leaving the TOF and timestamp panels and
        their SpanSelectors untouched -- crucially, it must not tear down
        and recreate the SpanSelector the user is actively dragging.
        """
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel("Normalized" if normalize else "Counts")

        if spectrum_only:
            if any(fe.check.isChecked() for fe in self._file_entries):
                self._ax.legend(fontsize=8)
            self._update_models()
            if self._ax.has_data():
                self._ax.margins(x=0.0, y=0.05)
                self._ax.autoscale_view()
                self._apply_ticks(self._ax)
            self._apply_plot_theme([(self._fig, self._canvas)])
            self._apply_grids([(self._ax, self._canvas)])
            return

        self._tof_ax.set_xlabel("Time (\u00b5s)", fontsize=8)
        self._tof_ax.set_ylabel("Counts", fontsize=8)
        self._tof_ax.tick_params(labelsize=7)
        ts_unit_label = self._ts_unit.currentText().lower()
        self._ts_ax.set_xlabel(f"Time ({ts_unit_label})", fontsize=8)
        self._ts_ax.set_ylabel("Events", fontsize=8)
        self._ts_ax.tick_params(labelsize=7)

        if any(fe.check.isChecked() for fe in self._file_entries):
            self._ax.legend(fontsize=8)

        self._update_models()

        # Hug the data: no horizontal padding (spectrum drawn edge-to-
        # edge), a little vertical headroom so peaks aren't clipped.
        # Runs BEFORE _setup_tof_span so the interactive
        # SpanSelector rectangle and the gate axvspan (which span the
        # whole axes and would otherwise drag the autoscale past the
        # data) can't widen the TOF x-range.
        for ax in (self._ax, self._tof_ax, self._ts_ax):
            if ax.has_data():
                ax.margins(x=0.0, y=0.05)
                ax.autoscale_view()
                self._apply_ticks(ax)

        # Gate spans + SpanSelectors are added last, after the limits
        # are locked, so they decorate the panels without resizing them.
        self._setup_tof_span()
        self._setup_ts_span()

        # Any later change to the spectrum axes limits (scroll-zoom,
        # toolbar pan/zoom) outside a full replot invalidates the captured
        # fast-path blit background. ax.clear() wipes these callbacks, so
        # re-register every full replot.
        self._ax.callbacks.connect('xlim_changed', self._invalidate_fast_bg)
        self._ax.callbacks.connect('ylim_changed', self._invalidate_fast_bg)

        # Re-paint dark/light chrome (ax.clear() reset it) and draw the
        # spectrum, ToF and timestamp canvases.
        self._apply_plot_theme([
            (self._fig, self._canvas),
            (self._tof_fig, self._tof_canvas),
            (self._ts_fig, self._ts_canvas),
        ])
        # Re-apply grids (ax.clear() dropped them too).
        self._apply_grids()

        # Full replot: the calibration / cooler diagnostic panels are
        # always refreshed here. The expensive gate-drag path returns
        # early above (spectrum_only) and never reaches this point.
        self._replot_calibrations()

    def _replot_calibrations(self):
        """Update the calibration and cooler-voltage diagnostic plots."""
        self._cal_readback_ax.clear()
        self._cal_diff_ax.clear()
        self._cal_step_ax.clear()
        self._cal_cooler_ax.clear()
        self._cool_ohlc_ax.clear()
        self._cool_ripple_ax.clear()

        # Skip split children — cooler stability is a parent-run
        # property; a split would just plot the parent's full trace
        # again under a different label.
        cool_entries = [
            e for e in self._file_entries
            if e.check.isChecked()
            and not isinstance(e, SplitFileEntry)
            and e.np_vcool is not None and e.np_ts is not None
            and len(e.np_vcool) > 1
            and float(e.np_ts.max() - e.np_ts.min()) > 0
        ]
        n_cool = len(cool_entries)
        n_bins = int(self._cooler_bins.value())
        cool_stats = []  # populated below; drives status strip + y-clip

        for entry in self._file_entries:
            if not entry.check.isChecked():
                continue
            # Splits share their parent's calibration + cooler arrays;
            # plotting them would just duplicate the parent's curves.
            if isinstance(entry, SplitFileEntry):
                continue

            label = f"run_{entry.run_number}"
            color = entry.color

            cal_set = entry.np_cal_set
            cal_rb = entry.np_cal_readback
            if (cal_set is not None and cal_rb is not None
                    and len(cal_set) >= 2):
                # Readback vs Set
                self._cal_readback_ax.plot(
                    cal_set, cal_rb, color=color, linewidth=1.5,
                    label=label)
                # Difference (Readback - Set) vs Set
                self._cal_diff_ax.plot(
                    cal_set, cal_rb - cal_set, color=color, linewidth=1.5,
                    label=label)
                # Step size: diff(Readback) vs Set
                self._cal_step_ax.plot(
                    cal_set[1:], np.diff(cal_rb), color=color, linewidth=1.5,
                    label=label)

            if entry not in cool_entries:
                continue

            order = np.argsort(entry.np_ts)
            t_rel = entry.np_ts[order] - entry.np_ts[order][0]
            v = entry.np_vcool[order]

            # \u2500\u2500 Robust run-level statistics \u2500\u2500
            # V_ref is the median (immune to spikes). \u03c3_robust is the
            # MAD-derived \u03c3 (1.4826\u00b7MAD \u2248 Gaussian \u03c3 for clean data).
            v_ref = float(np.median(v))
            mad = float(np.median(np.abs(v - v_ref)))
            sigma_robust = 1.4826 * mad
            # Robust peak-to-peak: P95\u2212P5 strips the long Poisson-like tails.
            p5 = float(np.percentile(v, 5))
            p95 = float(np.percentile(v, 95))
            pp_robust = p95 - p5
            # True extremes (signed, relative to V_ref).
            idx_max = int(np.argmax(v))
            idx_min = int(np.argmin(v))
            v_max = float(v[idx_max])
            v_min = float(v[idx_min])
            t_max_pt = float(t_rel[idx_max])
            t_min_pt = float(t_rel[idx_min])
            dev_up = v_max - v_ref
            dev_dn = v_min - v_ref
            # Outliers: |v \u2212 V_ref| > 3\u00b7\u03c3_robust. Robust \u03c3 keeps the
            # threshold meaningful even when there are real spikes.
            if sigma_robust > 0:
                n_outliers = int(np.sum(
                    np.abs(v - v_ref) > 3.0 * sigma_robust))
            else:
                n_outliers = 0

            # \u2500\u2500 Top pane: raw cooler voltage \u2500\u2500
            t_line, v_line = t_rel, v
            if len(v_line) > 50000:
                stride = len(v_line) // 50000
                t_line = t_line[::stride]
                v_line = v_line[::stride]
            self._cal_cooler_ax.plot(
                t_line, v_line, color=color, linewidth=1.2, alpha=0.9,
                rasterized=True,
                label=(f"{label}  V_ref={v_ref:.2f} V  "
                       f"\u03c3={sigma_robust:.3f} V"))
            self._cal_cooler_ax.axhline(
                v_ref, color=color, linestyle="--",
                linewidth=1.0, alpha=0.7)

            # \u2500\u2500 Per-bin aggregates for the middle and bottom panes \u2500\u2500
            t_max = float(t_rel[-1])
            edges = np.linspace(0.0, t_max, n_bins + 1)
            bin_idx = np.searchsorted(edges, t_rel, side='right') - 1
            np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
            unique_bins, first_pos = np.unique(
                bin_idx, return_index=True)
            last_pos = np.r_[first_pos[1:] - 1, len(bin_idx) - 1]
            medians_dev = np.full(n_bins, np.nan)
            stds_bin = np.full(n_bins, np.nan)
            pp_bin = np.full(n_bins, np.nan)
            for ub, fp, lp in zip(unique_bins, first_pos, last_pos):
                seg = v[fp:lp + 1]
                medians_dev[ub] = float(np.median(seg)) - v_ref
                stds_bin[ub] = float(np.std(seg))
                if (lp - fp) >= 4:
                    pp_bin[ub] = float(
                        np.percentile(seg, 95) - np.percentile(seg, 5))
                else:
                    pp_bin[ub] = float(seg.max() - seg.min())

            valid = ~np.isnan(medians_dev)
            if not valid.any():
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            cx = centers[valid]
            md = medians_dev[valid]
            sd_b = stds_bin[valid]
            pp_b = pp_bin[valid]

            # \u2500\u2500 Middle pane: deviation from V_ref \u2500\u2500
            # Constant \u00b11\u03c3_robust and \u00b13\u03c3_robust horizontal bands give
            # immediate visual answers for "is the data inside \u00b1\u03c3" and
            # "are there points beyond \u00b13\u03c3".
            if sigma_robust > 0:
                self._cool_ohlc_ax.axhspan(
                    -3.0 * sigma_robust, 3.0 * sigma_robust,
                    color=color, alpha=0.07, linewidth=0)
                self._cool_ohlc_ax.axhspan(
                    -sigma_robust, sigma_robust,
                    color=color, alpha=0.15, linewidth=0)
            # Per-bin median trace = the "rolling median" / drift line.
            self._cool_ohlc_ax.plot(
                cx, md, color=color, linewidth=1.4, alpha=0.95,
                drawstyle="steps-mid",
                label=(f"{label}  \u03c3={sigma_robust:.3f} V  "
                       f"p\u2013p={pp_robust:.3f} V  "
                       f"max+={dev_up:+.3f} V  max\u2212={dev_dn:+.3f} V  "
                       f"spikes={n_outliers}"))
            # Zero line = run average (V_ref).
            self._cool_ohlc_ax.axhline(
                0.0, color="#555555", linestyle="--",
                linewidth=1.0, alpha=0.6)
            # Spike samples (|v\u2212V_ref| > 3\u00b7\u03c3_robust) as small dots.
            # The bin-median trace is robust against these, so without
            # the dots the deviation pane shows no sign of them \u2014 the
            # dots make the "Clip y" toggle do something visible: with
            # clip on they're off-screen, with clip off they expand the
            # y-axis into the spike range.
            if sigma_robust > 0 and n_outliers > 0:
                spike_mask = np.abs(v - v_ref) > 3.0 * sigma_robust
                t_sp = t_rel[spike_mask]
                v_sp = (v - v_ref)[spike_mask]
                # Cap dot count for very dense spike clusters.
                if len(t_sp) > 500:
                    pick = np.linspace(
                        0, len(t_sp) - 1, 500).astype(int)
                    t_sp = t_sp[pick]
                    v_sp = v_sp[pick]
                self._cool_ohlc_ax.plot(
                    t_sp, v_sp,
                    marker=".", linestyle="None",
                    color=color, markersize=3.5, alpha=0.65,
                    zorder=4)

            # \u2500\u2500 Bottom pane: ripple strength over time \u2500\u2500
            self._cool_ripple_ax.plot(
                cx, sd_b, color=color, linewidth=1.4, alpha=0.95,
                linestyle="-",
                label=f"{label}  RMS")
            self._cool_ripple_ax.plot(
                cx, pp_b, color=color, linewidth=1.4, alpha=0.85,
                linestyle="--",
                label=f"{label}  P95\u2212P5")

            # Stash for status strip + y-clip after the loop completes.
            cool_stats.append({
                "label": label,
                "v_ref": v_ref,
                "sigma": sigma_robust,
                "pp_rob": pp_robust,
                "spikes": n_outliers,
                "max_up": dev_up,
                "max_dn": dev_dn,
            })

        self._cal_readback_ax.set_ylabel("Readback (V)", fontsize=8)
        self._cal_readback_ax.tick_params(labelsize=7)
        self._cal_diff_ax.set_ylabel("Readback \u2212 Set (V)\n[offset/drift]", fontsize=8)
        self._cal_diff_ax.tick_params(labelsize=7)
        self._cal_step_ax.set_xlabel("Set voltage (V)", fontsize=8)
        self._cal_step_ax.set_ylabel("\u0394 Readback (V)\n[step uniformity]", fontsize=8)
        self._cal_step_ax.tick_params(labelsize=7)
        # Same grid + minor-tick treatment the cooler axes already get.
        for _cal_ax in (self._cal_readback_ax, self._cal_diff_ax,
                        self._cal_step_ax):
            _cal_ax.minorticks_on()
            _cal_ax.grid(True, which="major", linestyle="-",
                         linewidth=0.5, color="#bbbbbb", alpha=0.7)
            _cal_ax.grid(True, which="minor", linestyle=":",
                         linewidth=0.4, color="#cccccc", alpha=0.5)
        self._cal_cooler_ax.set_xlabel(
            "Time since run start (s)", fontsize=8)
        self._cal_cooler_ax.set_ylabel(
            "Cooler V (V)  [raw]", fontsize=8)
        self._cal_cooler_ax.tick_params(labelsize=7)
        self._cal_cooler_ax.ticklabel_format(
            useOffset=False, style="plain", axis="y")
        self._cal_cooler_ax.minorticks_on()
        self._cal_cooler_ax.grid(
            True, which="major", linestyle="-",
            linewidth=0.5, color="#bbbbbb", alpha=0.7)
        self._cal_cooler_ax.grid(
            True, which="minor", linestyle=":",
            linewidth=0.4, color="#cccccc", alpha=0.5)

        self._cool_ohlc_ax.set_xlabel(
            "Time since run start (s)", fontsize=8)
        self._cool_ohlc_ax.set_ylabel(
            "Deviation from run average (V)", fontsize=8)
        self._cool_ohlc_ax.tick_params(labelsize=7)
        self._cool_ohlc_ax.ticklabel_format(
            useOffset=False, style="plain", axis="y")
        self._cool_ohlc_ax.minorticks_on()
        self._cool_ohlc_ax.grid(
            True, which="major", linestyle="-",
            linewidth=0.5, color="#bbbbbb", alpha=0.7)
        self._cool_ohlc_ax.grid(
            True, which="minor", linestyle=":",
            linewidth=0.4, color="#cccccc", alpha=0.5)
        self._cool_ohlc_ax.relim()
        self._cool_ohlc_ax.autoscale_view()
        # Robust y-clip: keep normal ripple visible even when there's a
        # rare large spike. Clamp to ±4·max(σ_rob), but don't shrink the
        # axis below what the bin medians already need.
        if cool_stats and self._cooler_clip_y.isChecked():
            sigmas = [s["sigma"] for s in cool_stats if s["sigma"] > 0]
            if sigmas:
                lim = 4.0 * max(sigmas)
                cur_lo, cur_hi = self._cool_ohlc_ax.get_ylim()
                # Take max of (clip, autoscaled) on each side — never
                # crop the bin-median line off-screen.
                lo = max(cur_lo, -lim) if cur_lo < -lim else cur_lo
                hi = min(cur_hi, lim) if cur_hi > lim else cur_hi
                if hi > lo:
                    self._cool_ohlc_ax.set_ylim(lo, hi)

        self._cool_ripple_ax.set_xlabel(
            "Time since run start (s)", fontsize=8)
        self._cool_ripple_ax.set_ylabel(
            "Ripple amplitude (V)", fontsize=8)
        self._cool_ripple_ax.tick_params(labelsize=7)
        self._cool_ripple_ax.ticklabel_format(
            useOffset=False, style="plain", axis="y")
        self._cool_ripple_ax.minorticks_on()
        self._cool_ripple_ax.grid(
            True, which="major", linestyle="-",
            linewidth=0.5, color="#bbbbbb", alpha=0.7)
        self._cool_ripple_ax.grid(
            True, which="minor", linestyle=":",
            linewidth=0.4, color="#cccccc", alpha=0.5)
        # Ripple metrics are non-negative — clip the y-axis at 0.
        self._cool_ripple_ax.set_ylim(bottom=0)

        if any(fe.check.isChecked() for fe in self._file_entries):
            self._cal_readback_ax.legend(fontsize=7)
            if self._cal_cooler_ax.has_data():
                self._cal_cooler_ax.legend(fontsize=7, loc="upper right")
            if n_cool >= 1:
                self._cool_ohlc_ax.legend(fontsize=7, loc="upper right")
                self._cool_ripple_ax.legend(fontsize=7, loc="upper right")

        # MHz-per-volt slope at the current beam energy: converts the
        # cooler-voltage ripple into its approximate Doppler-shift
        # impact (∂V_beam/∂V_cool ≈ 1). None when physics params are
        # incomplete — the MHz column then shows "—".
        mhz_per_v = None
        try:
            _mass = self._mass_spin.value()
            _harm = self._harmonic.value()
            _laser = self._get_fundamental_cm()
            _v0 = cool_stats[0]["v_ref"] if cool_stats else 0.0
            if _mass > 0 and _harm > 0 and _laser > 0 and _v0 > 0:
                mhz_per_v = (self._voltage_to_frequency(
                                 _v0 + 0.5, _mass, _harm, _laser)
                             - self._voltage_to_frequency(
                                 _v0 - 0.5, _mass, _harm, _laser))
                if not np.isfinite(mhz_per_v):
                    mhz_per_v = None
        except Exception:
            mhz_per_v = None

        # Secondary y-axis on the deviation pane in ≈MHz — the scale a
        # spectroscopist actually judges ripple by. Recreated per replot
        # (ax.clear() does not remove child secondary axes).
        _old_secax = getattr(self, "_cool_mhz_secax", None)
        if _old_secax is not None:
            try:
                _old_secax.remove()
            except Exception:
                pass
            self._cool_mhz_secax = None
        if mhz_per_v and n_cool >= 1:
            try:
                _k = float(mhz_per_v)
                self._cool_mhz_secax = self._cool_ohlc_ax.secondary_yaxis(
                    "right",
                    functions=(lambda v, k=_k: v * k,
                               lambda m, k=_k: m / k))
                self._cool_mhz_secax.set_ylabel("≈ Δν (MHz)", fontsize=8)
                self._cool_mhz_secax.tick_params(labelsize=7)
            except Exception:
                self._cool_mhz_secax = None

        # Per-run summary table. Hidden when no runs contribute.
        if cool_stats:
            from PySide6.QtWidgets import QTableWidgetItem
            tbl = self._cooler_table
            tbl.setSortingEnabled(False)
            tbl.setRowCount(len(cool_stats))
            for row, s_ in enumerate(cool_stats):
                mhz_txt = (f"{abs(s_['sigma'] * mhz_per_v):.2f}"
                           if mhz_per_v else "—")
                cells = [
                    str(s_["label"]),
                    f"{s_['v_ref']:.2f}",
                    f"{s_['sigma']:.3f}",
                    f"{s_['pp_rob']:.3f}",
                    str(s_["spikes"]),
                    f"{s_['max_dn']:+.3f} / {s_['max_up']:+.3f}",
                    mhz_txt,
                ]
                for col, val in enumerate(cells):
                    item = QTableWidgetItem(val)
                    if col >= 1:
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight
                            | Qt.AlignmentFlag.AlignVCenter)
                    tbl.setItem(row, col, item)
            tbl.setSortingEnabled(True)
            tbl.setVisible(True)
        else:
            self._cooler_table.setRowCount(0)
            self._cooler_table.setVisible(False)

        # Merged-only placeholder: calibration tables and the cooler
        # ripple are sampled-over-time arrays the merge collapses
        # away. When every checked entry is a MergedFileEntry the
        # diagnostic panels would otherwise be silently empty -- the
        # text below makes the limitation explicit instead.
        checked = [fe for fe in self._file_entries
                   if fe.check.isChecked() and fe.is_loaded]
        only_merged = checked and all(
            isinstance(fe, MergedFileEntry) for fe in checked)
        if only_merged:
            note = ("Not available for merged spectra\n"
                    "(per-step calibration and per-event cooler\n"
                    "samples were collapsed at merge time)")
            for ax in (self._cal_readback_ax, self._cal_diff_ax,
                        self._cal_step_ax, self._cal_cooler_ax,
                        self._cool_ohlc_ax, self._cool_ripple_ax):
                if not ax.lines and not ax.collections:
                    ax.text(0.5, 0.5, note,
                             transform=ax.transAxes,
                             ha="center", va="center", fontsize=8,
                             color="#888", style="italic")

        # Re-apply the dark/light chrome (each render cleared the axes,
        # resetting facecolor/spines) and draw the six panels.
        self._apply_plot_theme([
            (self._cal_readback_fig, self._cal_readback_canvas),
            (self._cal_diff_fig, self._cal_diff_canvas),
            (self._cal_step_fig, self._cal_step_canvas),
            (self._cal_cooler_fig, self._cal_cooler_canvas),
            (self._cool_ohlc_fig, self._cool_ohlc_canvas),
            (self._cool_ripple_fig, self._cool_ripple_canvas),
        ])

    def _mhz_to_xaxis(self, freq_mhz, xaxis_mode, harmonic, offset,
                       mass, cooler_v, laser_sp):
        """Convert frequency in MHz to the current x-axis display units."""
        freq_mhz = np.asarray(freq_mhz, dtype=float)
        # Guard: voltage modes need valid physics parameters
        if xaxis_mode in ("Voltage", "Calibrated voltage",
                          "Calibrated beam energy"):
            if laser_sp <= 0 or mass <= 0 or harmonic <= 0:
                return freq_mhz  # fall back to MHz if params invalid
        if xaxis_mode in ("Voltage", "Calibrated voltage"):
            wn = freq_mhz * 1e6 / (C_LIGHT * 100.0)
            wn_laser = laser_sp * harmonic
            wn_rest = wn + offset * harmonic
            m_kg = mass * AMU_TO_KG
            E0 = m_kg * C_LIGHT**2
            V_beam = E0 / E_CHARGE * \
                     ((wn_laser**2 + wn_rest**2) /
                      (2 * wn_rest * wn_laser) - 1)
            return -(V_beam - cooler_v)
        elif xaxis_mode == "Calibrated beam energy":
            wn = freq_mhz * 1e6 / (C_LIGHT * 100.0)
            wn_laser = laser_sp * harmonic
            wn_rest = wn + offset * harmonic
            m_kg = mass * AMU_TO_KG
            E0 = m_kg * C_LIGHT**2
            V_beam = E0 / E_CHARGE * \
                     ((wn_laser**2 + wn_rest**2) /
                      (2 * wn_rest * wn_laser) - 1)
            return V_beam
        elif xaxis_mode == "Wavenumber":
            # The HFS model frequencies are ALREADY in the detuning frame
            # (df/centroid and hfs.pos() are relative to the transition,
            # centred near 0 — see the Voltage branch above, which *adds*
            # offset*harmonic to recover the absolute wavenumber). The data
            # axis (_display_x) reaches the same detuning frame by *sub-
            # tracting* offset*harmonic from absolute lab-frame values, so
            # the model must NOT subtract it again — just convert MHz→cm-1.
            # (Reverts the #49 easy-win, which wrongly treated the model as
            # absolute and shifted it ~the full transition off-screen.)
            return freq_mhz * 1e6 / (C_LIGHT * 100.0)
        else:  # Frequency (MHz)
            # Model frequencies are already detuning-relative (see the
            # Wavenumber branch note above), matching the _display_x
            # Frequency axis which subtracts the transition from absolute
            # values. No offset subtraction here, or the overlay lands
            # ~the full transition frequency off-screen.
            return freq_mhz

    def _update_models(self):
        for line in list(self._ax.lines):
            if hasattr(line, '_is_hfs_model'):
                line.remove()
        for ann in list(self._ax.texts):
            if hasattr(ann, '_is_hfs_model'):
                ann.remove()

        xaxis_mode = self._xaxis_combo.currentText()
        harmonic = self._harmonic.value()
        offset = self._get_fundamental_cm()
        mass = self._get_isotope_mass()
        cooler_v, laser_sp = self._default_cooler_laser()

        for i, panel in enumerate(self._model_panels):
            if not panel.isChecked():
                continue

            p = panel.get_model_params()

            # Validate quantum numbers before creating HFS model
            I = p["I"]
            Jl = p["Jl"]
            Ju = p["Ju"]
            # I, Jl, Ju must be non-negative half-integers
            if I < 0 or Jl < 0 or Ju < 0:
                continue
            if (2 * I) != int(2 * I) or (2 * Jl) != int(2 * Jl) \
                    or (2 * Ju) != int(2 * Ju):
                continue
            # Scale must be non-zero for a visible model
            if p["scale"] == 0:
                continue

            try:
                import satlas2  # lazy import (avoids satlas2 cost at startup)
                hfs = satlas2.HFS(
                    I=I, J=[Jl, Ju],
                    A=[p["Al"], p["Au"]],
                    B=[p["Bl"], p["Bu"]],
                    C=[0, 0],
                    df=p["centroid"],
                    scale=p["scale"],
                    racah=True,
                    fwhmg=p["fwhm_g"], fwhml=p["fwhm_l"],
                    name=f"hfs_preview_{i}",
                )
                bkg = satlas2.Polynomial([p["bkg"]], name=f"bkg_preview_{i}")

                # Apply per-peak amplitude overrides
                peak_overrides = panel.get_peak_overrides()
                for pk_label, pk_amp in peak_overrides.items():
                    amp_key = f"Amp{pk_label}"
                    if amp_key in hfs.params:
                        hfs.params[amp_key].value = pk_amp

                positions = np.array(hfs.pos())
                if len(positions) == 0:
                    continue
                x_min = positions.min() - 500
                x_max = positions.max() + 500
                x_model = np.linspace(x_min, x_max, 2000)

                y_model = hfs.f(x_model) + bkg.f(x_model)

                x_plot = self._mhz_to_xaxis(
                    x_model, xaxis_mode, harmonic, offset,
                    mass, cooler_v, laser_sp)

                m_color = panel.color   # active (light/dark) model colour
                line, = self._ax.plot(x_plot, y_model, color=m_color,
                                      linestyle=panel.linestyle,
                                      alpha=panel.alpha,
                                      linewidth=2, label=panel.model_name)
                line._is_hfs_model = True

                # Annotate individual peak positions when labels are on
                if panel._peak_labels_toggle.isChecked():
                    labels = list(hfs.lines)
                    peak_mhz = np.array(hfs.pos())
                    peak_x = self._mhz_to_xaxis(
                        peak_mhz, xaxis_mode, harmonic, offset,
                        mass, cooler_v, laser_sp)
                    peak_y = hfs.f(peak_mhz) + bkg.f(peak_mhz)
                    for px, py, lbl in zip(peak_x, peak_y, labels):
                        ann = self._ax.annotate(
                            lbl, xy=(px, py),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=7, color=m_color,
                            rotation=45,
                            arrowprops=dict(arrowstyle='-',
                                            color=m_color, lw=0.5),
                        )
                        ann._is_hfs_model = True

            except Exception:
                _log.warning("HFS model overlay '%s' could not be drawn",
                             getattr(panel, "name", "?"), exc_info=True)

        if self._model_panels:
            self._ax.legend(fontsize=8)
        self._ax.relim()
        self._ax.autoscale_view()
        # Model curves/labels live in the fast-path blit background, and
        # this also rescales the axes -- force a background recapture on
        # the next gate-drag update (the spectrum-line cache stays valid).
        self._fast_bg = None
        self._canvas.draw_idle()

    # ── HFS model management ─────────────────────────────────────

    def _add_model(self):
        self._model_counter += 1
        idx = self._model_counter - 1
        panel = HFSModelPanel(
            name=f"Model {self._model_counter}",
            color=["#000000", "#d62728", "#2ca02c", "#9467bd"][idx % 4],
            dark_color=NEON_MODEL_COLORS[idx % len(NEON_MODEL_COLORS)])
        panel.set_dark_active(self._dark_mode)
        panel.params_changed.connect(self._update_models)
        self._model_panels.append(panel)
        self._model_list_layout.addWidget(panel)
        self._update_models()

    def _duplicate_checked_models(self):
        """Duplicate every checked (ticked) HFS model panel in this tab.

        The model panels are checkable group boxes; ``isChecked()`` marks
        the active model(s). Each checked model is copied with a "(copy)"
        name suffix so the user can tweak a variant without re-entering
        parameters.
        """
        checked = [p for p in self._model_panels if p.isChecked()]
        if not checked:
            QMessageBox.information(
                self, "Duplicate checked",
                "No checked model to duplicate.\n"
                "Tick a model's checkbox to mark it first.")
            return
        for source in checked:
            config = source.to_dict()
            self._model_counter += 1
            new_name = f"{config.get('name', 'Model')} (copy)"
            config["name"] = new_name
            _midx = self._model_counter - 1
            panel = HFSModelPanel(
                name=new_name,
                color=["#000000", "#d62728", "#2ca02c", "#9467bd"][
                    _midx % 4],
                dark_color=NEON_MODEL_COLORS[_midx % len(NEON_MODEL_COLORS)])
            panel.from_dict(config)
            panel.set_dark_active(self._dark_mode)
            panel.params_changed.connect(self._update_models)
            self._model_panels.append(panel)
            self._model_list_layout.addWidget(panel)
        self._update_models()

    def _remove_checked_models(self):
        """Remove every checked (ticked) HFS model panel in this tab.
        Undoable: Ctrl+Z restores the removed model(s)."""
        checked = [p for p in self._model_panels if p.isChecked()]
        if not checked:
            QMessageBox.information(
                self, "Remove checked",
                "No checked model to remove.\n"
                "Tick a model's checkbox to mark it first.")
            return
        stack = self._app_undo_stack()
        if stack is None:
            for panel in checked:
                self._model_panels.remove(panel)
                self._model_list_layout.removeWidget(panel)
                panel.deleteLater()
            self._update_models()
            return
        items = [(p, self._model_panels.index(p)) for p in checked]
        stack.push(_ModelRemoveCommand(self, items))

    def _do_soft_remove_models(self, panels):
        """Detach model panels WITHOUT deleting them (undo keeps them)."""
        for panel in panels:
            if panel in self._model_panels:
                self._model_panels.remove(panel)
                self._model_list_layout.removeWidget(panel)
                panel.setParent(None)
        self._update_models()

    def _do_reinsert_models(self, items):
        """Re-attach soft-removed model panels at their original indices."""
        for panel, index in sorted(items, key=lambda t: t[1]):
            index = max(0, min(int(index), len(self._model_panels)))
            self._model_panels.insert(index, panel)
            self._model_list_layout.insertWidget(index, panel)
            panel.show()
        self._update_models()

    # ── Plot editor ──────────────────────────────────────────────

    def _show_spectrum_info(self):
        QMessageBox.information(
            self, "Spectrum Plots",
            "<b>TOF (Time-of-Flight)</b><br>"
            "Histogram of ion arrival times (in \u00b5s) for the selected "
            "PMT channels. Use the <i>Gate</i> checkbox and the span "
            "selector to restrict the spectrum to a specific TOF window, "
            "isolating the ion bunch from background.<br><br>"
            "<b>Spectrum</b><br>"
            "Counts per voltage/frequency step for the selected channels "
            "and TOF gate. The x-axis mode can be set in Plot Options:<br>"
            "\u2022 <b>Voltage</b> \u2013 raw scanning (DAC) voltage<br>"
            "\u2022 <b>Calibrated voltage</b> \u2013 readback-corrected "
            "scanning voltage<br>"
            "\u2022 <b>Calibrated beam energy</b> \u2013 cooler voltage "
            "minus calibrated scanning voltage<br>"
            "\u2022 <b>Wavenumber / Frequency</b> \u2013 Doppler-shifted "
            "laser frequency as seen by the ions<br><br>"
            "<b>Timestamp</b><br>"
            "Event rate over elapsed time since the start of the run. "
            "Use the <i>Unit</i> selector to switch between seconds, "
            "minutes, hours, or days, and adjust the <i>Bin</i> size "
            "to control the time resolution. Useful for spotting beam "
            "drops, target depletion, or unstable conditions.")

    def _show_calib_info(self):
        QMessageBox.information(
            self, "Calibration Plots",
            "<b>Readback (V)</b><br>"
            "Actual voltage measured by the readback system vs. the "
            "requested set voltage. Shows overall voltage response.<br><br>"
            "<b>Readback \u2212 Set (V) [offset/drift]</b><br>"
            "Difference between readback and set voltage at each scan step. "
            "Reveals systematic offsets and drift in the voltage supply "
            "across the scan range.<br><br>"
            "<b>\u0394 Readback (V) [step uniformity]</b><br>"
            "Change in readback voltage between consecutive scan steps. "
            "Ideally all steps should be equal. Variations indicate "
            "non-linearity or instability in the voltage supply.")

    def _show_cooler_info(self):
        QMessageBox.information(
            self, "Cooler Voltage",
            "Three stacked panes, all sharing the same time axis. "
            "<b>V_ref</b> is a robust run average \u2014 the median of the "
            "per-event cooler voltage \u2014 so spikes don't bias the "
            "baseline. <b>\u03c3_robust</b> = 1.4826\u00b7MAD is the MAD-derived "
            "Gaussian \u03c3, also robust against outliers. A <b>spike</b> "
            "is any sample with |v \u2212 V_ref| > 3\u00b7\u03c3_robust.<br><br>"
            "<b>Top: Cooler V (V) [raw]</b><br>"
            "Per-event cooler voltage <code>Vrfq \u00b7 VCoolDiv + VCoolOffset"
            "</code>, in absolute volts. The dashed horizontal line is "
            "<b>V_ref</b>. Runs above ~50k events are stride-decimated "
            "for responsive drawing \u2014 slow ripple is preserved, very "
            "high frequencies may alias.<br><br>"
            "<b>Middle: Deviation from run average (V)</b><br>"
            "Same data with V_ref subtracted; the dashed line at y = 0 "
            "<i>is</i> V_ref, not a target setpoint.<br>"
            "&nbsp;&nbsp;\u2022 <b>Shaded bands</b> = constant \u00b1\u03c3_robust "
            "(inner) and \u00b13\u03c3_robust (outer). Inside the inner band \u2248 "
            "Gaussian core; samples outside the outer band are spikes."
            "<br>"
            "&nbsp;&nbsp;\u2022 <b>Stepped line</b> = per-bin median "
            "deviation. Acts as a rolling-median drift indicator \u2014 if "
            "it wanders away from 0 over time, the cooler is drifting. "
            "Robust against spikes inside a bin, so the line stays "
            "clean.<br>"
            "&nbsp;&nbsp;\u2022 <b>Dots</b> = individual spike samples "
            "(beyond \u00b13\u00b7\u03c3_robust). The bin-median line absorbs spikes, "
            "so without these dots the deviation pane would show no "
            "trace of them at all. Capped at 500 dots per run for "
            "speed.<br><br>"
            "<b>Bottom: Ripple amplitude (V) [over time]</b><br>"
            "Two lines per run, both binned in time:<br>"
            "&nbsp;&nbsp;\u2022 <b>Solid</b> = per-bin RMS (local \u03c3).<br>"
            "&nbsp;&nbsp;\u2022 <b>Dashed</b> = per-bin P95\u2212P5 (robust "
            "peak-to-peak inside that bin).<br>"
            "Tells you whether the supply was stable across the whole "
            "run or only quiet in some regions. A spike fattens both "
            "metrics in its bin only \u2014 outside that bin both lines stay "
            "flat. Y-axis starts at 0.<br><br>"
            "<b>Status strip</b> above the plots shows V_ref, \u03c3_robust, "
            "P95\u2212P5, spike count, and signed extremes (max\u2212/max+) per "
            "run \u2014 judge run health at a glance without reading the "
            "per-pane legends.<br><br>"
            "<b>Clip y to \u00b14\u03c3</b> (header checkbox, default on) "
            "clamps the deviation pane's y-axis to \u00b14\u00b7\u03c3_robust. With "
            "the clip on, spike dots fall off-screen and the bin-median "
            "line plus \u03c3-bands fill the visible range. Uncheck to let "
            "the y-axis autoscale around the spike dots \u2014 useful when "
            "you want to see exactly how far the worst sample reached. "
            "Spike count and the max\u2212/max+ values are reported either "
            "way in the legend and status strip.<br><br>"
            "<b>Bins</b> (header spinbox) controls the time-bin count "
            "used by the middle pane's per-bin median and the bottom "
            "pane's per-bin RMS / P95\u2212P5. More bins = finer time "
            "resolution at the cost of fewer samples per bin.")

    def _open_plot_editor(self):
        from gui.shared_widgets import PlotEditorDialog
        if (self._plot_editor is not None
                and self._plot_editor.isVisible()):
            self._plot_editor.raise_()
            self._plot_editor.activateWindow()
            return
        self._plot_editor = PlotEditorDialog(
            self._fig, self._canvas, parent=self,
            extra_figures=[
                ("TOF", self._tof_fig, self._tof_canvas),
                ("Timestamp", self._ts_fig, self._ts_canvas),
            ])
        self._plot_editor.show()

    # ── Config save / load ───────────────────────────────────────

    def save_config(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pre-Analysis Configuration",
            get_last_dir("config", "save"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        if not path.lower().endswith(('.yaml', '.yml')):
            path += '.yaml'
        remember_last_dir("config", "save", path)

        d = self._build_config_dict()
        # The path-keyed registries live at the top level, beside the tab
        # section, exactly as MainWindow's full-project save writes them --
        # so a PA-only config and a full project agree on where they are.
        # They are added here rather than inside _build_config_dict() because
        # that dict is also embedded as the project's `preanalysis:` section,
        # and writing them there too would store the same registry twice.
        from gui.calibration import get_registry as _get_cal_registry
        from gui.scan_filter import get_registry as _get_sf_registry
        _cal_reg = _get_cal_registry()
        cal = _cal_reg.to_dict()
        if cal:
            d["calibrations"] = cal
        acks = _cal_reg.acks_to_list()
        if acks:
            d["calibration_acks"] = acks
        sf = _get_sf_registry().to_dict()
        if sf:
            d["scan_filters"] = sf
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
        parent = self.window()
        if hasattr(parent, 'statusBar'):
            parent.statusBar().showMessage(
                f"Pre-Analysis config saved: {path}")

    def load_config(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pre-Analysis Configuration",
            get_last_dir("config", "load"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        remember_last_dir("config", "load", path)
        self._load_config_from_path(path)

    # ── Merge checked files ─────────────────────────────────────

    def _merge_checked(self):
        """Merge all checked (non-merged) file entries into one spectrum.

        Delegates to the shared ``gui.analysis.merge.MergeDialog`` and its
        ``compute_merged_spectrum`` backend (clstools-based binning,
        per-file Doppler in frequency mode), so the merged_data shape is
        the same one the Analysis tab produces. A synthetic source_config
        is built here with the keys the merge path reads (physics
        parameters, gates, ref_freq) and nothing else.
        """
        entries = [fe for fe in self._file_entries
                   if fe.check.isChecked() and fe.is_loaded
                   and not getattr(fe, '_is_merged', False)]
        if len(entries) < 2:
            QMessageBox.information(
                self, "Merge",
                "Select at least 2 loaded, non-merged files to merge.")
            return

        # Pre-compute per-source raw values + effective means here so
        # the dialog can show sensible defaults and so we can carry the
        # full audit trail forward when the merge is accepted.
        harmonic = int(self._harmonic.value())
        tab_mass = float(self._get_isotope_mass())
        source_info = []
        for fe in entries:
            raw_cooler = float(getattr(fe, "cooler_v", 0.0) or 0.0)
            raw_laser = float(getattr(fe, "laser_sp", 0.0) or 0.0)
            raw_mass = float(getattr(fe, "mass_amu", 0.0) or 0.0)
            source_info.append({
                "run_number": str(fe.run_number),
                "filepath":   fe.filepath,
                "cooler_v":   raw_cooler,
                "laser_sp":   raw_laser,
                # Raw ASDF MassAMU for the audit trail only. May differ
                # from tab_mass (e.g. acquisition wrote a bare nuclear
                # mass without electron correction). The merge-level
                # ``mass_amu`` below uses tab_mass instead so the fit-
                # time projection matches PA's per-event display
                # (which also uses _get_isotope_mass()).
                "mass_amu":   raw_mass,
                "harmonic":   harmonic,
            })
        eff_coolers = [float(self._cooler_for_entry(fe)) for fe in entries]
        eff_lasers = [float(self._laser_for_entry(fe)) for fe in entries]
        # Mass: user's PA-tab choice is authoritative -- the per-event
        # PA display Doppler uses self._get_isotope_mass(), and the
        # merge's projection at fit time must match. Picking the mean
        # of source_info[i]["mass_amu"] would silently use whatever
        # the ASDF stored, which may not be what the user wants
        # (e.g. user override flagged on a different value).
        default_merge_metadata = {
            "cooler_v": (float(np.mean(eff_coolers))
                         if eff_coolers else None),
            "laser_sp": (float(np.mean(eff_lasers))
                         if eff_lasers else None),
            "mass_amu": tab_mass,
            "harmonic": harmonic,
        }

        # Synthetic source_config for the shared MergeDialog, in the
        # units the merge backend expects: ref_freq in Hz, gates in
        # their native clstools units. The dialog and
        # compute_merged_spectrum both read this dict; nothing in PA
        # reads it back.
        e_lo = self._e_lower.value()
        e_up = self._e_upper.value()
        ref_freq_hz = (e_up - e_lo) * 1e2 * C_LIGHT
        pmt_gate = [i + 1 for i, cb in enumerate(self._channels)
                    if cb.isChecked()]
        tof_gate = None
        if self._tof_enable.isChecked():
            tof_gate = [self._tof_lo.value(), self._tof_hi.value()]
        # Pick the dialog's default domain from the PA x-axis: if the
        # user is viewing in Frequency / Wavenumber, frequency merge
        # is almost certainly what they want.
        xaxis = self._xaxis_combo.currentText()
        default_domain = ("frequency"
                          if xaxis in ("Frequency", "Wavenumber")
                          else "voltage")
        source_config = {
            "cal_order":       1,
            "cooler_correction": "pbp",
            "tof_gate":        tof_gate,
            "pmt_gate":        pmt_gate,
            "v_gate":          None,
            "f_gate":          None,
            "mass":            tab_mass,
            "ref_freq":        ref_freq_hz,
            "harmonic":        harmonic,
            "ref_shift":       0.0,
            "noise_filter":    0,
            "cooler_override": (self._cooler_override.value()
                                if self._cooler_override_enabled.isChecked()
                                else 0),
            "laser_override":  (self._laser_override.value()
                                if self._laser_override_enabled.isChecked()
                                else 0),
            "bin_mode":        ("Raw Voltage"
                                if default_domain == "voltage"
                                else "Frequency"),
            "bin_definition":  self._bin_def_combo.currentText(),
            "bin_count":       int(self._bin_count_spin.value()),
            "bin_width_mhz":   float(self._bin_width_spin.value()),
            "x_column":        self._x_col_combo.currentText(),
            "yerr_mode":       self._yerr_combo.currentText(),
            "xerr_mode":       self._xerr_combo.currentText(),
        }

        # MergeDialog file_entries are dicts with these keys (Analysis
        # tab already passes per-file binning overrides; PA has none).
        dlg_entries = [
            {"path": fe.filepath,
             "run_number": fe.run_number,
             "binning_override": {}}
            for fe in entries
        ]

        # Thread per-file manual centroid offsets (set via the FileEntry
        # right-click "Set centroid offset…") into the dialog so a
        # frequency-domain merge applies them at the same point
        # compute_merged_spectrum applies the GP correction.
        manual_offsets_map = {
            fe.filepath: float(getattr(fe, "centroid_offset_mhz", 0.0))
            for fe in entries
            if getattr(fe, "centroid_offset_mhz", 0.0)
        }

        from gui.analysis.merge import MergeDialog
        dlg = MergeDialog(
            dlg_entries, source_config, parent=self,
            default_domain=default_domain,
            default_merge_metadata=default_merge_metadata,
            manual_offsets_map=manual_offsets_map)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        result = dlg.get_result()
        if not result:
            return

        # Convert the MergeDialog result dict into a MergedFileEntry.
        # Loading a PA-saved config goes through this same
        # _add_merged_entry, so the source_info / merge_* invariants are
        # preserved on round-trip.
        mp = result.get("merge_params", {})
        domain = mp.get("domain", "voltage")
        mm = result.get("merge_metadata", {}) or {}
        # User-edited dialog values override our default means. For
        # *frequency* merges the dialog hid the metadata panel and
        # those spinners are just stale pre-fills -- persisting them
        # would falsely suggest the user picked them. Drop to None so
        # the YAML audit trail matches the user's actual intent.
        if domain == "frequency":
            merge_cooler_v = None
            merge_laser_sp = None
            merge_mass_amu = None
            merge_harmonic = None
        else:
            merge_cooler_v = mm.get("cooler_v")
            merge_laser_sp = mm.get("laser_sp")
            merge_mass_amu = mm.get("mass_amu")
            merge_harmonic = (mm.get("harmonic") or harmonic)
        name = result.get(
            "merged_name",
            "merged_" + "_".join(
                str(s["run_number"]) for s in source_info))
        # Keep the per_run audit (TOF, timestamps, n_events, per-file
        # binned summaries) on the MergedFileEntry in-memory so the
        # Analysis-side bridge has more than just source_info to fill
        # in when the entry is later imported. Drop the per-source
        # x/y arrays before storing -- they're redundant with the
        # merged x/y and inflate memory usage for nothing.
        stripped_per_run = [
            {k: v for k, v in rd.items() if k not in ("x", "y")}
            for rd in (result.get("per_run") or [])
        ]
        self._add_merged_entry(
            name, result["x"], result["y"], domain, source_info,
            merge_cooler_v=merge_cooler_v,
            merge_laser_sp=merge_laser_sp,
            merge_mass_amu=merge_mass_amu,
            merge_harmonic=merge_harmonic,
            per_run=stripped_per_run or None)

    def _add_merged_entry(self, name, merged_x, merged_y, domain,
                          source_info,
                          merge_cooler_v=None, merge_laser_sp=None,
                          merge_mass_amu=None, merge_harmonic=None,
                          per_run=None,
                          color="#333333", checked=True,
                          alpha=1.0, linestyle="-"):
        """Create and add a MergedFileEntry to the file list."""
        mfe = MergedFileEntry(
            name=name,
            merged_x=merged_x,
            merged_y=merged_y,
            merge_domain=domain,
            source_info=source_info,
            merge_cooler_v=merge_cooler_v,
            merge_laser_sp=merge_laser_sp,
            merge_mass_amu=merge_mass_amu,
            merge_harmonic=merge_harmonic,
            per_run=per_run,
            color=color,
        )
        mfe.alpha = alpha
        mfe.linestyle = linestyle
        mfe.check.setChecked(checked)
        mfe.toggled.connect(self._schedule_replot)
        mfe.toggled.connect(self._refresh_master_check_state)
        mfe.color_changed.connect(self._schedule_replot)
        mfe.removed.connect(self._on_entry_remove_requested)
        mfe.clicked.connect(self._select_file_entry)
        # Wire the merged entry's View / Edit / Export menu signals to
        # the handlers that open the shared MergeDialog / viewer / export.
        mfe.view_requested.connect(self._view_merged_entry)
        mfe.edit_requested.connect(self._edit_merged_entry)
        mfe.export_requested.connect(self._export_merged_entry)
        mfe.color_index = len(self._file_entries)  # neon slot (dark mode)
        mfe.dark_color = NEON_COLORS[mfe.color_index % len(NEON_COLORS)]
        mfe.set_dark_active(self._dark_mode)
        self._file_entries.append(mfe)
        self._file_list_layout.addWidget(mfe)
        self._refresh_master_check_state()
        self._schedule_replot()

    # _merge_checked routes through
    # gui.analysis.merge.compute_merged_spectrum, which uses clstools'
    # binning so the PA and Analysis merge results agree exactly.
    # _bin_spectrum is retained because the PA spectrum plot (_replot)
    # still needs per-entry (v_means, counts) at native DAC-step
    # resolution.

    # ── PA-side merged-entry actions (View / Edit / Export) ──────
    #
    # Triggered by the MergedFileEntry right-click menu: View opens a
    # read-only plot, Edit re-opens the MergeDialog, Export writes a
    # merged ASDF.

    def _pa_synthetic_source_config(self):
        """Build the same synthetic source_config ``_merge_checked``
        passes into MergeDialog -- used by View and Edit so the
        dialog reads consistent physics across all three entry
        points (Merge / View / Edit) without duplicating the dict
        construction.
        """
        harmonic = int(self._harmonic.value())
        tab_mass = float(self._get_isotope_mass())
        e_lo = self._e_lower.value()
        e_up = self._e_upper.value()
        ref_freq_hz = (e_up - e_lo) * 1e2 * C_LIGHT
        pmt_gate = [i + 1 for i, cb in enumerate(self._channels)
                    if cb.isChecked()]
        tof_gate = None
        if self._tof_enable.isChecked():
            tof_gate = [self._tof_lo.value(), self._tof_hi.value()]
        return {
            "cal_order":       1,
            "cooler_correction": "pbp",
            "tof_gate":        tof_gate,
            "pmt_gate":        pmt_gate,
            "v_gate":          None,
            "f_gate":          None,
            "mass":            tab_mass,
            "ref_freq":        ref_freq_hz,
            "harmonic":        harmonic,
            "ref_shift":       0.0,
            "noise_filter":    0,
            "cooler_override": (self._cooler_override.value()
                                if self._cooler_override_enabled.isChecked()
                                else 0),
            "laser_override":  (self._laser_override.value()
                                if self._laser_override_enabled.isChecked()
                                else 0),
            "bin_mode":        ("Frequency"
                                if self._xaxis_combo.currentText()
                                    in ("Frequency", "Wavenumber")
                                else "Raw Voltage"),
            "bin_definition":  self._bin_def_combo.currentText(),
            "bin_count":       int(self._bin_count_spin.value()),
            "bin_width_mhz":   float(self._bin_width_spin.value()),
            "x_column":        self._x_col_combo.currentText(),
            "yerr_mode":       self._yerr_combo.currentText(),
            "xerr_mode":       self._xerr_combo.currentText(),
        }

    def _view_merged_entry(self, mfe):
        """Open the shared MergeViewDialog for a PA merged entry.

        When bin_mode is Frequency and the merge is voltage-domain, the
        dialog applies the V→F projection so the preview matches the
        eventual fit.
        """
        from gui.analysis.merge import MergeViewDialog
        try:
            md = mfe.to_merged_data()
        except Exception as exc:
            QMessageBox.warning(self, "View merged",
                                 f"Could not build view data:\n{exc}")
            return
        dlg = MergeViewDialog(
            md, parent=self,
            source_config=self._pa_synthetic_source_config())
        dlg.show()
        # Hold the reference so the non-modal dialog survives.
        self._merged_view_dlg = dlg

    def _edit_merged_entry(self, mfe):
        """Re-open the unified MergeDialog seeded with the existing
        merge so the user can adjust parameters and re-merge."""
        from gui.analysis.merge import MergeDialog
        # Re-source entries by file path from the loaded PA list.
        # Edit can only re-merge sources that are currently loaded
        # in PA -- otherwise there's nothing to recompute against.
        source_paths = {s.get("filepath")
                        for s in (mfe.source_info or [])
                        if s.get("filepath")}
        loaded = {fe.filepath: fe for fe in self._file_entries
                  if not isinstance(fe, MergedFileEntry)}
        present = [loaded[p] for p in source_paths if p in loaded]
        if len(present) < 2:
            QMessageBox.information(
                self, "Edit merged",
                "Re-merging needs the original source ASDFs to be "
                "loaded in this Pre-Analysis tab. Open them via "
                "File ▸ Open… and try again.")
            return
        dlg_entries = [
            {"path": fe.filepath, "run_number": fe.run_number,
             "binning_override": {}}
            for fe in present
        ]
        existing = mfe.to_merged_data()
        existing["merge_params"] = {
            "domain": ("frequency" if mfe.merge_domain == "frequency"
                       else "voltage"),
            "bin_step_mhz": None,
            "cooler_correction": "pbp",
            "centroid_correction": False,
            "merge_metadata": existing["merge_metadata"],
        }
        dlg = MergeDialog(
            dlg_entries, self._pa_synthetic_source_config(),
            parent=self,
            existing_result=existing,
            default_domain=("frequency"
                            if mfe.merge_domain == "frequency"
                            else "voltage"),
            default_merge_metadata=existing["merge_metadata"])
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        result = dlg.get_result()
        if not result:
            return
        # Replace the entry: simplest is to remove + add fresh, same
        # path the Analysis side uses. _remove_single_entry detaches
        # the widget; _add_merged_entry attaches a new one.
        self._remove_single_entry(mfe)
        mp = result.get("merge_params", {})
        domain = mp.get("domain", "voltage")
        mm = result.get("merge_metadata", {}) or {}
        if domain == "frequency":
            merge_cooler_v = None
            merge_laser_sp = None
            merge_mass_amu = None
            merge_harmonic = None
        else:
            merge_cooler_v = mm.get("cooler_v")
            merge_laser_sp = mm.get("laser_sp")
            merge_mass_amu = mm.get("mass_amu")
            merge_harmonic = (mm.get("harmonic")
                              or int(self._harmonic.value()))
        # Rebuild source_info from the dialog's input file_entries +
        # the original mfe.source_info (so per-source physics audit
        # is preserved when a source file is still loaded).
        source_info = []
        for e in dlg_entries:
            match = next(
                (s for s in (mfe.source_info or [])
                 if s.get("filepath") == e["path"]), None)
            if match:
                source_info.append(dict(match))
            else:
                source_info.append({
                    "run_number": str(e.get("run_number")),
                    "filepath":   e["path"],
                    "cooler_v":   None, "laser_sp": None,
                    "mass_amu":   None, "harmonic": None,
                })
        stripped_per_run = [
            {k: v for k, v in rd.items() if k not in ("x", "y")}
            for rd in (result.get("per_run") or [])
        ]
        name = result.get("merged_name", str(mfe.run_number))
        self._add_merged_entry(
            name, result["x"], result["y"], domain, source_info,
            merge_cooler_v=merge_cooler_v,
            merge_laser_sp=merge_laser_sp,
            merge_mass_amu=merge_mass_amu,
            merge_harmonic=merge_harmonic,
            per_run=stripped_per_run or None)

    def _export_merged_entry(self, mfe):
        """Export a PA merged entry as a DENIS-labelled ASDF.

        Uses the shared export_merged_asdf writer, so the resulting file
        can be re-loaded in either the Pre-Analysis or Analysis tab.
        """
        from gui.analysis.merge import export_merged_asdf
        from gui.shared_widgets import get_last_dir, remember_last_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Export merged ASDF",
            os.path.join(
                get_last_dir("data", "save"),
                f"{mfe.run_number}.asdf"),
            "ASDF files (*.asdf)")
        if not path:
            return
        if not path.lower().endswith(".asdf"):
            path += ".asdf"
        remember_last_dir("data", "save", path)
        try:
            export_merged_asdf(mfe.to_merged_data(), path)
        except ValueError as exc:
            # Consistency guard refused to write inconsistent data.
            QMessageBox.warning(
                self, "Export merged",
                f"Cannot export this merged spectrum:\n\n{exc}")
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Export merged",
                f"Export failed:\n{exc}")
            return
        parent = self.window()
        if hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Merged ASDF saved: {path}", 5000)

    # ── Save / Load config ───────────────────────────────────────

    def _build_config_dict(self):
        files = []
        merged_entries = []
        for fe in self._file_entries:
            if isinstance(fe, MergedFileEntry):
                merged_entries.append({
                    "name": fe.run_number,
                    "domain": fe.merge_domain,
                    "x": fe.merged_x.tolist(),
                    "y": fe.merged_y.tolist(),
                    "source_info": fe.source_info,
                    # Merge-level Doppler metadata. Optional; None is
                    # preserved so loaders that don't know about these
                    # keys still round-trip the entry intact.
                    "merge_cooler_v": fe.merge_cooler_v,
                    "merge_laser_sp": fe.merge_laser_sp,
                    "merge_mass_amu": fe.merge_mass_amu,
                    "merge_harmonic": fe.merge_harmonic,
                    "color": fe.light_color,
                    "dark_color": fe.dark_color,
                    "alpha": fe.alpha,
                    "linestyle": fe.linestyle,
                    "checked": fe.check.isChecked(),
                })
            else:
                # Real ASDFs and loaded .vasdf splits both serialize
                # as plain {path, ...} entries. The .vasdf path itself
                # carries the split descriptor — no extra YAML needed.
                files.append({
                    "path": fe.filepath,
                    "color": fe.light_color,
                    "dark_color": fe.dark_color,
                    "alpha": fe.alpha,
                    "linestyle": fe.linestyle,
                    "checked": fe.check.isChecked(),
                    # Persist the manual centroid offset. Optional /
                    # default 0 so older YAMLs without this key load
                    # unchanged.
                    "centroid_offset_mhz":
                        float(getattr(fe, "centroid_offset_mhz", 0.0)),
                })

        pmt_gate = [i + 1 for i, cb in enumerate(self._channels)
                    if cb.isChecked()]

        d = {
            "preanalysis": {
                "files": files,
                "merged_entries": merged_entries,
                "plot_options": {
                    "x_axis": self._xaxis_combo.currentText(),
                    "plot_layout": self._plot_layout_mode,
                    "dark_mode": self._dark_mode,
                    "grid_x": self._grid_x,
                    "grid_y": self._grid_y,
                    "e_lower": self._e_lower.value(),
                    "e_upper": self._e_upper.value(),
                    "harmonic": self._harmonic.value(),
                    "normalize": self._normalize.isChecked(),
                    "Z": self._z_spin.value(),
                    "A": self._a_spin.value(),
                    "mass_override": self._mass_override.isChecked(),
                    "mass_amu": self._mass_spin.value(),
                    "channels": pmt_gate,
                },
                "tof_gate": {
                    "enabled": self._tof_enable.isChecked(),
                    "lo": self._tof_lo.value(),
                    "hi": self._tof_hi.value(),
                    "binsize": float(self._tof_binsize.value()),
                },
                "cooler_voltage": self._cooler_override.value(),
                "laser_setpoint": self._laser_override.value(),
                "cooler_override": self._cooler_override_enabled.isChecked(),
                "laser_override": self._laser_override_enabled.isChecked(),
                "binning": {
                    "bin_mode": self._bin_mode_combo.currentText(),
                    "x_column": self._x_col_combo.currentText(),
                    "yerr_mode": self._yerr_combo.currentText(),
                    "xerr_mode": self._xerr_combo.currentText(),
                    "bin_definition": self._bin_def_combo.currentText(),
                    "bin_count": int(self._bin_count_spin.value()),
                    "bin_width_mhz": float(self._bin_width_spin.value()),
                    "step_multiple": int(self._spec_bin_mult.value()),
                },
                "models": [p.to_dict() for p in self._model_panels],
            }
        }
        return d

    def _load_config_from_path(self, path):
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load config:\n{e}")
            return

        # Registries first: the calibrations must be in force BEFORE the file
        # entries load, or every run would compute its voltages from the file
        # default and only pick the override up on some later replot. Merged
        # rather than replaced, so loading one config doesn't wipe filters or
        # calibrations belonging to files still open from another.
        from gui.calibration import get_registry as _get_cal_registry
        from gui.scan_filter import get_registry as _get_sf_registry
        if isinstance(raw.get("calibrations"), dict):
            _reg = _get_cal_registry()
            _merged = _reg.to_dict()
            _merged.update(raw["calibrations"])
            _reg.from_dict(_merged)
        if isinstance(raw.get("calibration_acks"), list):
            _reg = _get_cal_registry()
            _reg.acks_from_list(
                list(set(_reg.acks_to_list())
                     | set(raw["calibration_acks"])))
        if isinstance(raw.get("scan_filters"), dict):
            _sreg = _get_sf_registry()
            _smerged = _sreg.to_dict()
            _smerged.update(raw["scan_filters"])
            _sreg.from_dict(_smerged)

        cfg = raw.get("preanalysis", raw)
        self._restore_from_dict(cfg)

        parent = self.window()
        if hasattr(parent, 'statusBar'):
            parent.statusBar().showMessage(
                f"Pre-Analysis config loaded: {path}")

    def _restore_from_dict(self, cfg):
        """Populate the Pre-Analysis tab from a config dict."""
        for fe in list(self._file_entries):
            self._file_list_layout.removeWidget(fe)
            fe.deleteLater()
        self._file_entries.clear()
        self._refresh_master_check_state()

        for p in list(self._model_panels):
            self._model_list_layout.removeWidget(p)
            p.deleteLater()
        self._model_panels.clear()
        self._model_counter = 0

        opts = cfg.get("plot_options", {})
        idx = self._xaxis_combo.findText(opts.get("x_axis", "Frequency"))
        if idx >= 0:
            self._xaxis_combo.setCurrentIndex(idx)
        # Plot arrangement (no replot here — the caller replots once
        # after the whole config is applied).
        self._set_plot_layout(opts.get("plot_layout", "stacked"),
                              replot=False)
        # Dark-mode plots: fall back to the current (settings-derived)
        # state when the save predates this key.
        self._set_dark_mode(opts.get("dark_mode", self._dark_mode),
                            replot=False)
        self._set_grids(opts.get("grid_x", self._grid_x),
                        opts.get("grid_y", self._grid_y))
        # New-style: e_lower + e_upper. Legacy: single "offset" field.
        if "e_lower" in opts:
            self._e_lower.setValue(float(opts["e_lower"]))
            self._e_upper.setValue(float(opts["e_upper"]))
        elif "offset" in opts:
            # Legacy: offset was the fundamental (transition/harmonic).
            # Put it as E_upper with E_lower=0 so fundamental matches.
            harmonic = int(opts.get("harmonic", 2))
            self._e_lower.setValue(0.0)
            self._e_upper.setValue(float(opts["offset"]))
        self._harmonic.setValue(int(opts.get("harmonic", 2)))
        self._update_transition_labels()
        self._normalize.setChecked(bool(opts.get("normalize", False)))
        self._z_spin.setValue(int(opts.get("Z", 1)))
        self._a_spin.setValue(int(opts.get("A", 1)))
        # Mass override (default off; only respect saved value when on)
        override = bool(opts.get("mass_override", False))
        self._mass_override.setChecked(override)
        if override and "mass_amu" in opts:
            self._mass_spin.blockSignals(True)
            self._mass_spin.setValue(float(opts["mass_amu"]))
            self._mass_spin.blockSignals(False)
        else:
            self._refresh_mass_display()

        channels = opts.get("channels", [3, 4])
        for i, cb in enumerate(self._channels):
            cb.setChecked((i + 1) in channels)

        tof = cfg.get("tof_gate", {})
        self._tof_enable.setChecked(bool(tof.get("enabled", False)))
        self._tof_lo.setValue(float(tof.get("lo", 30)))
        self._tof_hi.setValue(float(tof.get("hi", 60)))
        self._tof_binsize.setValue(float(tof.get("binsize", 1.0)))

        self._cooler_override.setValue(
            float(cfg.get("cooler_voltage", 29977)))
        self._laser_override.setValue(
            float(cfg.get("laser_setpoint", 10920)))
        # Per-parameter override ticks (legacy "cooler_laser_override"
        # applies to both for back-compat with older configs).
        legacy = bool(cfg.get("cooler_laser_override", False))
        self._cooler_override_enabled.setChecked(
            bool(cfg.get("cooler_override", legacy)))
        self._laser_override_enabled.setChecked(
            bool(cfg.get("laser_override", legacy)))

        # Binning (Pre-Analysis defaults preserve the legacy look if
        # the YAML predates this section).
        binning = cfg.get("binning", {})
        self._bin_mode_combo.setCurrentText(
            binning.get("bin_mode", "Raw Voltage"))
        self._x_col_combo.setCurrentText(
            binning.get("x_column", "bins_center"))
        self._yerr_combo.setCurrentText(
            binning.get("yerr_mode", "None"))
        self._xerr_combo.setCurrentText(
            binning.get("xerr_mode", "None"))
        self._bin_def_combo.setCurrentText(
            binning.get("bin_definition", "Auto"))
        self._bin_count_spin.setValue(
            int(binning.get("bin_count", 100)))
        self._bin_width_spin.setValue(
            float(binning.get("bin_width_mhz", 10.0)))
        self._spec_bin_mult.setValue(
            int(binning.get("step_multiple", 1)))
        self._update_binning_enabled()

        for file_cfg in cfg.get("files", []):
            filepath = maybe_convert_path(file_cfg.get("path", ""))
            if not os.path.isfile(filepath):
                QMessageBox.warning(
                    self, "File Not Found",
                    f"Data file not found, skipping:\n{filepath}")
                continue
            _n_before = len(self._file_entries)
            self._load_file(filepath)
            # Apply per-file settings ONLY when a new entry was actually
            # appended. _load_file can fail silently (corrupt ASDF, bad .vasdf,
            # missing parent) and return without appending; the old
            # `if self._file_entries` non-emptiness check then misapplied this
            # file's color/checked/centroid-offset to a PRECEDING entry -- a
            # silent state corruption on load (code review 2026-06-02,
            # config-load-misattributes-file-settings).
            if len(self._file_entries) > _n_before:
                fe = self._file_entries[-1]
                fe.light_color = file_cfg.get("color",
                                               fe.light_color)
                fe.dark_color = file_cfg.get("dark_color",
                                              fe.dark_color)
                fe.alpha = file_cfg.get("alpha", 1.0)
                fe.linestyle = file_cfg.get("linestyle", "-")
                fe.check.setChecked(file_cfg.get("checked", True))
                # Restore manual centroid offset; defaults to 0 when
                # the key is missing (older YAML).
                fe.centroid_offset_mhz = float(
                    file_cfg.get("centroid_offset_mhz", 0.0))

        # `or []` covers two pathological inputs both seen in the wild:
        # a hand-edited YAML with `merged_entries: null`, and an older
        # config that drops the key entirely (the .get already handles
        # the latter, the `or` adds the former).
        for mcfg in (cfg.get("merged_entries") or []):
            # The merge_* keys are .get-defaulted so older YAMLs (no
            # merge_* keys, 2-list source_info entries) keep loading
            # unchanged. MergedFileEntry.__init__ normalizes the
            # source_info shape; the merge_* keys default to None
            # ("fall back to mean-of-sources at use time").
            self._add_merged_entry(
                name=mcfg.get("name", "merged"),
                merged_x=mcfg.get("x", []),
                merged_y=mcfg.get("y", []),
                domain=mcfg.get("domain", "voltage"),
                source_info=mcfg.get("source_info", []),
                merge_cooler_v=mcfg.get("merge_cooler_v"),
                merge_laser_sp=mcfg.get("merge_laser_sp"),
                merge_mass_amu=mcfg.get("merge_mass_amu"),
                merge_harmonic=mcfg.get("merge_harmonic"),
                color=mcfg.get("color", "#333333"),
                checked=mcfg.get("checked", True),
                alpha=mcfg.get("alpha", 1.0),
                linestyle=mcfg.get("linestyle", "-"),
            )

        for model_cfg in cfg.get("models", []):
            self._model_counter += 1
            # Seed the same light/dark defaults _add_model uses BEFORE
            # from_dict: a save that predates the dark-colour feature has
            # no "dark_color" key, and a bare HFSModelPanel() would fall
            # back to the light colour (black) -- so the first _replot()
            # below drew a black model curve on the black canvas, and the
            # neon fallback in _set_dark_mode only fixed the swatch later.
            _midx = self._model_counter - 1
            panel = HFSModelPanel(
                name=f"Model {self._model_counter}",
                color=["#000000", "#d62728", "#2ca02c", "#9467bd"][
                    _midx % 4],
                dark_color=NEON_MODEL_COLORS[_midx % len(NEON_MODEL_COLORS)])
            panel.from_dict(model_cfg)
            panel.set_dark_active(self._dark_mode)
            panel.params_changed.connect(self._update_models)
            self._model_panels.append(panel)
            self._model_list_layout.addWidget(panel)

        self._replot()
