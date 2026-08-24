"""Analysis-tab block widgets (Source, Model, Fitter, Output).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Defines the draggable block widgets that make up an analysis pipeline:
SourceBlock (file selection, physics, gates, binning, merging), ModelBlock
(HFS/lineshape parameters and constraints), FitterBlock (fit method, parameter
sharing, advanced constraints, MCMC), and OutputBlock (reports, plots,
diagnostics). Also houses the supporting binning/diagnostics dialogs and the
constraint-mode helpers shared across the parameter tables.

Depends on: cls_estimations.mass_lookup; gui.analysis.helpers, .binning,
.expr_validation, .naming, .vasdf; gui.shared_widgets. UI built on PySide6;
HFS models are built with satlas2 (imported lazily).
"""

import os
import numpy as np

# satlas2 is imported lazily where used (HFS model build) to keep app startup
# off the satlas2/lmfit import path.
from cls_estimations.mass_lookup import load_mass_table, get_mass, Z_TO_ELEMENT

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QRadioButton, QButtonGroup, QPushButton,
    QScrollArea, QToolButton, QMenu, QDialog, QDialogButtonBox, QTabWidget,
    QFileDialog, QMessageBox, QSizePolicy, QInputDialog,
    QProgressBar, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QTableWidget, QTableWidgetItem,
)
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint
from PySide6.QtGui import QDrag, QBrush, QColor

# clstools drags in pandas + scipy (~2 s of app startup), so it is
# imported on first use — data loading — not at module import.
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

from gui.analysis.helpers import (
    _make_double, _make_int, _make_analysis_spin, _BoundsButton,
    _show_scrollable_info, PopupPlotWindow,
)
from gui.analysis.binning import (
    BIN_DEFINITIONS, DEFAULT_BIN_DEFINITION,
    DEFAULT_BIN_COUNT, DEFAULT_BIN_WIDTH_MHZ,
    BINNING_OVERRIDE_KEYS, effective_binning_config,
    compute_binned,
)
from gui.analysis.expr_validation import (
    build_param_registry, infer_constraint_mode,
    validate_single_expression,
)
from gui.analysis.naming import NON_FIT_PARAMS, full_param_name
from gui.analysis.vasdf import is_vasdf_path, read_vasdf
from gui.dialog_style import make_plot_card, style_dialog
from gui.shared_widgets import _load_settings, _save_settings


# Constraint-mode dropdown values for the ModelBlock parameter table.
_CONSTRAINT_MODES = ("Free", "Fixed", "Equal", "Ratio", "Offset", "Custom")

# Per-mode background tints for the Mode combo, color-coding each
# parameter row's constraint type for at-a-glance identification.
_MODE_COLORS = {
    "Free":   "#1f6b30",   # green   — floats freely
    "Fixed":  "#8e2424",   # red     — held at value
    "Equal":  "#1d4f8a",   # blue    — locked to another param
    "Ratio":  "#5e2872",   # plum    — scaled multiple of another
    "Offset": "#a25c1f",   # orange  — additive offset to another
    "Custom": "#1a6b6b",   # teal    — free-form expression
}


def _apply_mode_combo_color(combo):
    """Tint a constraint Mode combo by its current text so the
    constraint type of every parameter row in the model table is
    identifiable at a glance.
    """
    bg = _MODE_COLORS.get(combo.currentText())
    if bg:
        combo.setStyleSheet(
            f"QComboBox {{ background-color: {bg}; color: white; }}")
    else:
        combo.setStyleSheet("")

# Column indices for the ModelBlock _param_table.
# The "Vary" state is encoded by Mode (only Mode=Free means vary=True),
# so there's no Vary column; saved configs still carry a `vary` field
# that's derived from Mode at save time.
_PT_PARAM = 0
_PT_VALUE = 1
_PT_BOUNDS = 2
_PT_MODE = 3
_PT_EXPR = 4


# Tooltip shown on a model-block expression cell when its content is
# valid (or empty). Replaced with the validator's message on errors.
_MODEL_EXPR_TOOLTIP_OK = (
    "Expression: bare parameter names from this model.\n"
    "Examples: Al * 0.5, FWHMG, centroid + 1250.\n"
    "For cross-run links, use the Fitter block's Advanced Constraints table."
)
# Tooltip shown on an Advanced Constraints expression cell when
# its content is valid (or empty).
_FITTER_EXPR_TOOLTIP_OK = (
    "Set this target's value with a full-name expression.\n"
    "Reference other parameters by their full lmfit name:\n"
    "    Run_<id>___<Model>___<param>\n\n"
    "Examples:\n"
    "    Run_7164___HFS_1___centroid + 1250\n"
    "    2 * Run_7164___HFS_1___FWHMG\n\n"
    "Leave blank for no constraint on this target.\n"
    "Filled rows override Parameter Sharing and ModelBlock\n"
    "expressions for the same target."
)

# Tooltip on the Parameter Sharing group box header.
_SHARING_GRP_TOOLTIP = (
    "Tie a bare parameter so the fit treats it as one variable\n"
    "across multiple runs/models.\n\n"
    "Use this when a physical quantity should be the same in\n"
    "every run -- e.g., a true hyperfine constant, or the\n"
    "spectrometer's Gaussian linewidth (FWHMG).\n\n"
    "For run-specific or run-pair relationships (e.g.,\n"
    "'Run B's centroid = Run A's centroid + offset'), use\n"
    "the Advanced Constraints table below."
)

# Tooltip on the Advanced Constraints group box header.
_CONSTRAINTS_GRP_TOOLTIP = (
    "Set explicit cross-run or cross-model relationships that\n"
    "Parameter Sharing can't express.\n\n"
    "Each row is one possible target (one source / model /\n"
    "param triple in your fit). Type a full-name expression in\n"
    "a row's Expression cell to constrain that target. Empty\n"
    "rows are ignored.\n\n"
    "Example: '2 * Run_7164___HFS_1___FWHMG' on the row for\n"
    "Run_7191's HFS_1 / FWHMG makes 7191's gaussian width\n"
    "twice 7164's at fit time.\n\n"
    "Constraints set here take precedence over Parameter\n"
    "Sharing and ModelBlock expressions for the same target."
)

# Tooltip on the Mode combo in each sharing-table row.
_SHARING_MODE_TOOLTIP = (
    "How widely to share this parameter:\n\n"
    "Model -- shared across every run that has the same model.\n"
    "    HFS_1's value becomes one variable across runs.\n"
    "    HFS_2 (if present) keeps its own independent variable.\n\n"
    "All -- shared across every model in every run.\n"
    "    HFS_1, HFS_2, and any other model with this parameter\n"
    "    all collapse to a single shared value.\n\n"
    "If your fit has only one model name (typical), Model and\n"
    "All are equivalent."
)

# Tooltip on the Share checkbox in each sharing-table row.
_SHARING_CHECKBOX_TOOLTIP = (
    "Tick to share this parameter as one variable across the\n"
    "runs/models selected by the Mode column on the right."
)
# Cell background applied to a constraint/expression cell whose
# contents fail validation.
_EXPR_ERROR_BG = QColor("#7a3030")
# Alias for the canonical NON_FIT_PARAMS set defined in
# gui.analysis.naming, the single source the validator, fitter, and
# UI all share so their notion of non-fit parameters stays in sync.
_NON_FIT_PARAMS_FOR_VALIDATION = NON_FIT_PARAMS


class _ConstraintModeDialog(QDialog):
    """Pick a target parameter (and optional value) for Equal / Ratio / Offset.

    Used by ModelBlock when the user changes the Mode column to one
    of the relational modes. Returns the target parameter name and,
    for Ratio/Offset, a numeric value.
    """

    def __init__(self, parent, mode, available_targets,
                 current_target=None, current_value=None):
        super().__init__(parent)
        self.setWindowTitle({
            "Equal": "Equal to...",
            "Ratio": "Ratio of...",
            "Offset": "Offset from...",
        }[mode])
        self._mode = mode

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._target = QComboBox()
        self._target.addItems(available_targets)
        if current_target and current_target in available_targets:
            self._target.setCurrentIndex(
                available_targets.index(current_target))
        form.addRow("Target parameter:", self._target)

        self._value = None
        if mode in ("Ratio", "Offset"):
            self._value = QDoubleSpinBox()
            self._value.setRange(-1e9, 1e9)
            self._value.setDecimals(6)
            if mode == "Ratio":
                self._value.setValue(
                    current_value if current_value is not None else 1.0)
                form.addRow("Ratio (target × this):", self._value)
            else:  # Offset
                self._value.setValue(
                    current_value if current_value is not None else 0.0)
                form.addRow("Offset (target + this):", self._value)

        layout.addLayout(form)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def result_data(self) -> tuple[str, float | None]:
        target = self._target.currentText()
        value = self._value.value() if self._value is not None else None
        return target, value
from gui.shared_widgets import get_plot_type_settings, maybe_convert_path


# ══════════════════════════════════════════════════════════════════
#  Pre-Analysis project chooser (for multi-tab Pre-Analysis)
# ══════════════════════════════════════════════════════════════════

def _choose_pa_project(parent_widget):
    """Return a PreAnalysisTab to pull from, prompting if multiple exist."""
    main_win = parent_widget.window()
    if not hasattr(main_win, 'preanalysis_tab'):
        return None
    container = main_win.preanalysis_tab
    # Legacy: container IS a PreAnalysisTab directly (no _projects attr)
    if not hasattr(container, '_projects'):
        return container
    projects = container._projects
    if not projects:
        QMessageBox.information(parent_widget, "Pre-Analysis",
                                "No Pre-Analysis projects exist.\n"
                                "Create one in the Pre-Analysis tab first.")
        return None
    if len(projects) == 1:
        return projects[0]
    # Multiple projects: show selection dialog
    names = [container._project_tabs.tabText(
                 container._project_tabs.indexOf(p))
             for p in projects]
    current = container._project_tabs.currentWidget()
    current_idx = projects.index(current) if current in projects else 0
    name, ok = QInputDialog.getItem(
        parent_widget, "Select Pre-Analysis Project",
        "Pull from which Pre-Analysis project?",
        names, current_idx, False)
    if not ok:
        return None
    return projects[names.index(name)]


# ══════════════════════════════════════════════════════════════════
#  Block Base Class
# ══════════════════════════════════════════════════════════════════

class AnalysisBlock(QGroupBox):
    """Base class for all analysis blocks (Source, Model, Fitter, Output)."""
    block_changed = Signal()
    remove_requested = Signal(object)

    BLOCK_TYPE = "Block"
    BLOCK_COLOR = "#555555"

    # Default width; subclasses can override.
    BLOCK_WIDTH = 340

    MIN_BLOCK_WIDTH = 220

    def __init__(self, name="", parent=None):
        super().__init__(parent)
        self._block_name = name or self.BLOCK_TYPE
        self.setTitle(f"{self.BLOCK_TYPE}: {self._block_name}")
        self.setCheckable(True)
        self.setChecked(True)
        # Scale block width by current font size vs default (9pt)
        from PySide6.QtWidgets import QApplication
        _base_pt = 9
        _cur_pt = QApplication.instance().font().pointSize() or _base_pt
        _scale = _cur_pt / _base_pt
        self._current_width = int(self.BLOCK_WIDTH * _scale)
        self.setFixedWidth(self._current_width)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._resizing = False
        self._resize_cursor_set = False
        self._resize_start_x = 0
        self._resize_start_width = 0

        self.setStyleSheet(f"""
            AnalysisBlock, SourceBlock, ModelBlock, FitterBlock, OutputBlock {{
                border: 2px solid {self.BLOCK_COLOR};
                border-radius: 5px;
                margin-top: 18px;
                padding-top: 6px;
            }}
            AnalysisBlock::title, SourceBlock::title, ModelBlock::title,
            FitterBlock::title, OutputBlock::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {self.BLOCK_COLOR};
                font-weight: bold;
            }}
        """)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(6, 8, 6, 6)
        self._main_layout.setSpacing(4)

        # Header row: drag handle + name edit + delete button
        header = QHBoxLayout()
        header.setSpacing(4)

        self._drag_handle = QToolButton()
        self._drag_handle.setText("\u2630")  # hamburger icon
        self._drag_handle.setToolTip("Drag to reorder")
        self._drag_handle.setFixedSize(22, 22)
        self._drag_handle.setStyleSheet(
            "QToolButton { border: none; font-size: 14px; color: #888; }"
            "QToolButton:hover { color: #ccc; }")
        self._drag_handle.installEventFilter(self)
        header.addWidget(self._drag_handle)

        self._name_edit = QLineEdit(self._block_name)
        self._name_edit.setPlaceholderText("Block name")
        self._name_edit.editingFinished.connect(self._on_name_changed)
        header.addWidget(self._name_edit)

        self._delete_btn = QToolButton()
        self._delete_btn.setText("\u2717")
        self._delete_btn.setToolTip("Remove this block")
        self._delete_btn.setFixedSize(22, 22)
        self._delete_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        header.addWidget(self._delete_btn)
        self._main_layout.addLayout(header)

        self._drag_start_pos = None

        # Content area (subclasses populate this)
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidget(self._content_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        self._main_layout.addWidget(scroll, 1)

        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked):
        self._content_widget.setEnabled(checked)
        self.block_changed.emit()

    def _on_name_changed(self):
        self._block_name = self._name_edit.text().strip() or self.BLOCK_TYPE
        self.setTitle(f"{self.BLOCK_TYPE}: {self._block_name}")
        self.block_changed.emit()

    @property
    def block_name(self):
        return self._block_name

    @property
    def is_enabled(self):
        return self.isChecked()

    def _near_right_edge(self, pos):
        """Return True if pos is within 6px of the right edge."""
        return pos.x() >= self.width() - 6

    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton and
                self._near_right_edge(event.position().toPoint())):
            self._resizing = True
            self._resize_start_x = event.globalPosition().toPoint().x()
            self._resize_start_width = self._current_width
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        from PySide6.QtWidgets import QApplication
        if self._resizing:
            dx = event.globalPosition().toPoint().x() - self._resize_start_x
            new_w = max(self.MIN_BLOCK_WIDTH, self._resize_start_width + dx)
            self._current_width = new_w
            self.setFixedWidth(new_w)
            event.accept()
            return
        # Change cursor when near right edge
        if self._near_right_edge(event.position().toPoint()):
            if not self._resize_cursor_set:
                QApplication.setOverrideCursor(
                    Qt.CursorShape.SizeHorCursor)
                self._resize_cursor_set = True
        else:
            if self._resize_cursor_set:
                QApplication.restoreOverrideCursor()
                self._resize_cursor_set = False
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        from PySide6.QtWidgets import QApplication
        if self._resizing:
            self._resizing = False
            self.releaseMouse()
            if self._resize_cursor_set:
                QApplication.restoreOverrideCursor()
                self._resize_cursor_set = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        from PySide6.QtWidgets import QApplication
        if self._resize_cursor_set and not self._resizing:
            QApplication.restoreOverrideCursor()
            self._resize_cursor_set = False
        super().leaveEvent(event)

    def to_dict(self):
        return {"type": self.BLOCK_TYPE, "name": self._block_name,
                "enabled": self.isChecked(), "width": self._current_width}

    def from_dict(self, d):
        self._block_name = d.get("name", self.BLOCK_TYPE)
        self._name_edit.setText(self._block_name)
        self.setTitle(f"{self.BLOCK_TYPE}: {self._block_name}")
        self.setChecked(d.get("enabled", True))
        if "width" in d:
            self._current_width = max(self.MIN_BLOCK_WIDTH, d["width"])
            self.setFixedWidth(self._current_width)

    def eventFilter(self, obj, event):
        """Start drag when the drag handle is pressed and moved."""
        if obj is self._drag_handle:
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_start_pos = event.globalPosition().toPoint()
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if (self._drag_start_pos is not None and
                        (event.globalPosition().toPoint() -
                         self._drag_start_pos).manhattanLength() > 10):
                    drag = QDrag(self)
                    mime = QMimeData()
                    mime.setText(self._block_name)
                    mime.setData("application/x-analysis-block",
                                 str(id(self)).encode())
                    drag.setMimeData(mime)
                    # Create a small pixmap preview
                    pm = self.grab().scaled(
                        120, 80, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    drag.setPixmap(pm)
                    drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
                    self._drag_start_pos = None
                    drag.exec(Qt.DropAction.MoveAction)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_start_pos = None
        return super().eventFilter(obj, event)


# ══════════════════════════════════════════════════════════════════
#  Per-File Binning Override Dialog
# ══════════════════════════════════════════════════════════════════


class FileBinningOverrideDialog(QDialog):
    """Edit a per-file binning override as a sparse patch.

    Each overridable key has an *Inherit* checkbox and a value widget.
    When Inherit is checked, the value widget is disabled and the key is
    *not* included in the resulting override dict — the file inherits
    from the Source block.
    """

    _BIN_MODES = ("Frequency", "Raw Voltage")
    _X_COLUMNS = ("bins_center", "Fmean")
    _YERR_MODES = ("None", "Poisson sqrt(y+1)", "Poisson sqrt(y)", "Model-based")
    _XERR_MODES = ("None", "From voltage std")

    def __init__(self, source_cfg, current_override,
                 run_label="this run", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Binning Override — {run_label}")
        style_dialog(self)
        self.setMinimumWidth(420)
        self._source_cfg = dict(source_cfg)

        from PySide6.QtWidgets import QDialogButtonBox
        outer = QVBoxLayout(self)
        intro = QLabel(
            "Each row controls one binning setting for THIS run only.\n"
            "  ☑ Use Source value  → this run follows the Source block "
            "(value in parentheses).\n"
            "  ☐ Use Source value  → this run uses the value on the right "
            "instead, frozen for this run.")
        intro.setStyleSheet("color: #aaa;")
        outer.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        outer.addLayout(form)

        # Row factory: returns (inherit_cb, value_widget, get_value, set_value)
        self._rows = {}

        def _inherit_cb(display_value):
            """Checkbox labelled 'Use Source (<current Source value>)'."""
            cb = QCheckBox(f"Use Source ({display_value})")
            cb.setToolTip(
                "Checked  → this run follows the Source block.\n"
                "           The value in parentheses is the current\n"
                "           Source value; future Source changes will\n"
                "           propagate to this run.\n\n"
                "Unchecked → this run uses the value on the right\n"
                "           instead. The value is frozen for this run\n"
                "           only and does not change if the Source\n"
                "           block is later edited.")
            # Highlight when override is active so the row stands out.
            def _style(chk):
                cb.setStyleSheet(
                    "" if chk
                    else "color: #ffb74d; font-weight: bold;")
            cb.toggled.connect(_style)
            _style(cb.isChecked())
            return cb

        def add_combo_row(key, label, items):
            src_val = str(source_cfg.get(key, items[0]))
            cb = _inherit_cb(src_val)
            w = QComboBox()
            w.addItems(items)
            w.setCurrentText(src_val)
            row = QHBoxLayout()
            row.addWidget(cb)
            row.addWidget(w, 1)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)
            cb.toggled.connect(lambda chk: w.setDisabled(chk))
            self._rows[key] = (cb, w,
                               lambda: w.currentText(),
                               lambda v: w.setCurrentText(str(v)))

        def add_int_row(key, label, lo, hi):
            src_val = int(source_cfg.get(key, lo))
            cb = _inherit_cb(str(src_val))
            w = _make_int(src_val, lo, hi)
            row = QHBoxLayout()
            row.addWidget(cb)
            row.addWidget(w, 1)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)
            cb.toggled.connect(lambda chk: w.setDisabled(chk))
            self._rows[key] = (cb, w, lambda: w.value(),
                               lambda v: w.setValue(int(v)))

        def add_double_row(key, label, lo, hi, decimals, step, unit=""):
            src_val = float(source_cfg.get(key, lo))
            shown = f"{src_val:g}{(' ' + unit) if unit else ''}"
            cb = _inherit_cb(shown)
            w = _make_double(src_val, lo, hi, decimals, step)
            row = QHBoxLayout()
            row.addWidget(cb)
            row.addWidget(w, 1)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)
            cb.toggled.connect(lambda chk: w.setDisabled(chk))
            self._rows[key] = (cb, w, lambda: w.value(),
                               lambda v: w.setValue(float(v)))

        add_combo_row("bin_mode", "Bin mode:", self._BIN_MODES)
        add_combo_row("x_column", "x values:", self._X_COLUMNS)
        add_combo_row("yerr_mode", "yerr mode:", self._YERR_MODES)
        add_combo_row("xerr_mode", "x-error:", self._XERR_MODES)
        add_combo_row("bin_definition", "Bin definition:",
                      list(BIN_DEFINITIONS))
        add_int_row("bin_count", "Bin count:", 1, 1000000)
        add_double_row("bin_width_mhz", "Bin width [MHz]:",
                       1e-4, 1e6, 4, 1.0, unit="MHz")
        add_int_row("step_multiple", "Bin multiple:", 1, 1000)

        # Pre-populate from current override (uncheck inherit + set value).
        for key, value in (current_override or {}).items():
            if key not in self._rows:
                continue
            cb, w, _get, _set = self._rows[key]
            cb.setChecked(False)
            try:
                _set(value)
            except Exception:
                pass

        # All others start as inherited.
        for key, (cb, w, _g, _s) in self._rows.items():
            if key not in (current_override or {}):
                cb.setChecked(True)
                w.setDisabled(True)

        # Live enable/disable of count/width based on effective state.
        def _refresh():
            eff_mode = (self._rows["bin_mode"][2]()
                        if not self._rows["bin_mode"][0].isChecked()
                        else self._source_cfg.get("bin_mode", "Frequency"))
            eff_def = (self._rows["bin_definition"][2]()
                       if not self._rows["bin_definition"][0].isChecked()
                       else self._source_cfg.get(
                           "bin_definition", DEFAULT_BIN_DEFINITION))
            count_inh = self._rows["bin_count"][0].isChecked()
            width_inh = self._rows["bin_width_mhz"][0].isChecked()
            mult_inh = self._rows["step_multiple"][0].isChecked()
            voltage = (eff_mode == "Raw Voltage")
            self._rows["bin_count"][1].setDisabled(
                count_inh or voltage or eff_def != "Fixed bin count")
            self._rows["bin_width_mhz"][1].setDisabled(
                width_inh or voltage or eff_def != "Fixed bin width")
            self._rows["step_multiple"][1].setDisabled(
                mult_inh or not (voltage or eff_def == "Per scan step"))
        for key in ("bin_mode", "bin_definition", "bin_count",
                    "bin_width_mhz", "step_multiple"):
            cb, w, _g, _s = self._rows[key]
            cb.toggled.connect(lambda *_: _refresh())
            if hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(lambda *_: _refresh())
        _refresh()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def get_override(self):
        """Return the sparse override dict (only non-inherited keys)."""
        out = {}
        for key, (cb, w, getter, _setter) in self._rows.items():
            if not cb.isChecked():
                out[key] = getter()
        return out


# ══════════════════════════════════════════════════════════════════
#  Combined Binning Dialog (Summary table + Diagnostics plot)
# ══════════════════════════════════════════════════════════════════


class BinningDialog(QDialog):
    """Tabbed binning view: Summary table, per-run Run-detail plot
    (spectrum with bin edges + optional raw-event rug, with occupancy
    and width health as text strips) and the cross-run Compare heatmap.

    `infos` is a list of (label, full_compute_binned_result_or_None,
    error_str_or_None). The result dict must include the diagnostics block
    when available; merged spectra omit it.
    """

    _COLUMNS = ("Run", "Source", "Mode", "Definition", "N bins",
                "dx median", "x range", "Override", "Empty",
                "Median counts", "Total counts")
    _SHADING_LIMIT = 300
    _EDGES_LIMIT = 500

    def __init__(self, infos, warnings_list, default_tab="summary",
                 default_run=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Binning")
        self.resize(960, 640)
        # Same visual language as the tabs and the other dialogs.
        style_dialog(self)
        from PySide6.QtWidgets import QDialogButtonBox

        self._infos = infos
        self._warnings = warnings_list

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(6)

        # Warnings header sits ABOVE the tabs so it's always visible.
        if warnings_list:
            warn_box = QGroupBox(f"Warnings ({len(warnings_list)})")
            warn_lay = QVBoxLayout(warn_box)
            warn_lay.setContentsMargins(8, 8, 8, 8)
            warn_lay.setSpacing(2)
            for w in warnings_list:
                color = "#ff7043" if w["level"] == "warning" else "#90caf9"
                tag = "WARN" if w["level"] == "warning" else "INFO"
                run_part = f" [{w['run']}]" if w.get("run") else ""
                lbl = QLabel(f"<span style='color:{color}; "
                             f"font-weight:bold;'>{tag}</span> "
                             f"{w['code']}{run_part}: {w['message']}")
                lbl.setWordWrap(True)
                warn_lay.addWidget(lbl)
            outer.addWidget(warn_box)
        else:
            outer.addWidget(QLabel(
                "<span style='color:#81c784;'>✓ No binning warnings."
                "</span>"))

        # Plottable rows: drop errored / merged (no diagnostics dict).
        plottable = [(label, res) for (label, res, err) in self._infos
                     if err is None and res and res.get("diagnostics")]
        self._plottable = dict(plottable)

        # Shared run picker — visible for all plot tabs (not Summary).
        self._picker_row = QWidget()
        prow = QHBoxLayout(self._picker_row)
        prow.setContentsMargins(0, 0, 0, 0)
        prow.addWidget(QLabel("Run:"))
        self._run_combo = QComboBox()
        for label, _res in plottable:
            self._run_combo.addItem(label)
        prow.addWidget(self._run_combo, 1)
        outer.addWidget(self._picker_row)
        self._picker_row.setVisible(False)   # Summary is the default tab

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_summary_tab(), "Summary")
        self._tabs.addTab(self._build_detail_tab(), "Run detail")
        self._tabs.addTab(self._build_compare_tab(), "Compare")
        outer.addWidget(self._tabs, 1)

        # Cross-tab dispatch: re-render only the visible plot.
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._run_combo.currentTextChanged.connect(
            lambda *_: self._refresh_active_plot())

        # Pick the requested run if it has a plot, otherwise the first.
        if default_run and default_run in self._plottable:
            idx = self._run_combo.findText(default_run)
            if idx >= 0:
                self._run_combo.setCurrentIndex(idx)

        if default_tab == "diagnostics":
            self._tabs.setCurrentIndex(1)
        else:
            # Summary is the default; no plot to render up front.
            self._on_tab_changed(self._tabs.currentIndex())

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.button(QDialogButtonBox.StandardButton.Close).clicked.connect(
            self.accept)
        outer.addWidget(btns)

    # ── Summary tab ────────────────────────────────────────────────

    def _build_summary_tab(self, parent=None):
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(2)
        tip = QLabel("<i>Double-click a row to jump to its diagnostics "
                     "plot.</i>")
        tip.setStyleSheet("color: #aaa;")
        lay.addWidget(tip)

        table = QTableWidget(len(self._infos), len(self._COLUMNS))
        table.setHorizontalHeaderLabels(list(self._COLUMNS))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        for row, (label, res, err) in enumerate(self._infos):
            if err is not None:
                table.setItem(row, 0, QTableWidgetItem(label))
                err_item = QTableWidgetItem(f"ERROR: {err}")
                err_item.setForeground(Qt.GlobalColor.red)
                table.setItem(row, 1, err_item)
                table.setSpan(row, 1, 1, len(self._COLUMNS) - 1)
                continue
            info = (res or {}).get("info", {})
            mode = info.get("bin_mode", "?")
            n = info.get("effective_n_bins", 0) or 0
            dx = (info.get("effective_bin_width_mhz")
                  or info.get("effective_bin_width_v"))
            unit = "MHz" if mode == "Frequency" else "V"
            x_min = info.get("x_min", 0) or 0
            x_max = info.get("x_max", 0) or 0
            ovr_keys = info.get("override_keys") or []
            # Fit-relevant occupancy stats instead of the old
            # Fallback/Verdict/Why noise (real problems already show
            # in the warnings header above the tabs).
            y_arr = np.asarray((res or {}).get("y", []), dtype=float)
            stats = self._occupancy_stats(y_arr)
            cells = [
                label,
                info.get("source", "?"),
                mode,
                info.get("bin_definition", "?"),
                str(n),
                f"{dx:.4f} {unit}" if dx else "—",
                f"{x_min:.2f} … {x_max:.2f} {unit}",
                ", ".join(ovr_keys) if ovr_keys else "—",
                (f"{stats['n_empty']}/{stats['n_in_scan']} "
                 f"({100 * stats['n_empty'] / stats['n_in_scan']:.0f}%)"
                 if stats["n_in_scan"] else "—"),
                f"{stats['median_nonzero']:.1f}",
                f"{stats['total']:.0f}",
            ]
            for c, val in enumerate(cells):
                it = QTableWidgetItem(val)
                if c == 7 and ovr_keys:
                    it.setForeground(Qt.GlobalColor.darkYellow)
                if (c == 8 and stats["n_in_scan"]
                        and stats["n_empty"]
                        > 0.2 * stats["n_in_scan"]):
                    it.setForeground(Qt.GlobalColor.darkYellow)
                table.setItem(row, c, it)
        table.resizeColumnsToContents()
        table.cellDoubleClicked.connect(self._jump_to_diagnostics)
        lay.addWidget(table, 1)
        self._summary_table = table
        return wrap

    @staticmethod
    def _occupancy_stats(y):
        """Gap-aware occupancy statistics for one run's binned counts.

        CLS scans cover a few narrow ranges around the HFS clusters;
        uniform grids fill the unvisited stretches with zero bins. A
        contiguous zero-run of length ≥ max(30, 5%·n) is treated as a
        SCAN GAP (outside the scan), everything else as real bins.
        Returns masks plus in-scan empty/low counts, the median
        non-zero occupancy and the total counts.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        empty_mask = (y == 0)
        low_mask = (y > 0) & (y < 10)
        gap_mask = np.zeros(n, dtype=bool)
        if n > 0:
            gap_threshold = max(30, int(round(0.05 * n)))
            i = 0
            while i < n:
                if empty_mask[i]:
                    j = i
                    while j < n and empty_mask[j]:
                        j += 1
                    if (j - i) >= gap_threshold:
                        gap_mask[i:j] = True
                    i = j
                else:
                    i += 1
        true_empty = empty_mask & ~gap_mask
        nonzero = y[y > 0]
        return {
            "gap_mask": gap_mask,
            "true_empty_mask": true_empty,
            "low_mask": low_mask,
            "n_gap": int(gap_mask.sum()),
            "n_empty": int(true_empty.sum()),
            "n_low": int(low_mask.sum()),
            "n_in_scan": max(1, n - int(gap_mask.sum())) if n else 0,
            "median_nonzero": (float(np.median(nonzero))
                               if len(nonzero) else 0.0),
            "max": float(y.max()) if n else 0.0,
            "total": float(y.sum()) if n else 0.0,
        }

    @staticmethod
    def _scale_strip_html(info):
        """One-line summary of the voltage↔rest-frame-frequency scale.

        Shows the raw-voltage bin size, the local 1 V → MHz conversion
        (dν/dV, with its across-scan spread when Doppler-nonlinear),
        and the bin size in the Doppler-shifted rest frame. Degrades
        gracefully when the scale couldn't be measured (e.g. a merged
        spectrum with no per-event F/DV frame)."""
        mpv = info.get("mhz_per_volt")
        bw_v = info.get("bin_width_v")
        bw_mhz = info.get("bin_width_rest_mhz")
        bits = []
        if bw_v is not None:
            bits.append(f"raw-voltage bin: <b>{bw_v:.4g} V</b>")
        if mpv:
            lo = info.get("mhz_per_volt_min")
            hi = info.get("mhz_per_volt_max")
            conv = f"1 V ≈ <b>{mpv:.4g} MHz</b>"
            # Flag Doppler nonlinearity across the scan (>5% spread).
            if lo and hi and mpv and (hi - lo) > 0.05 * mpv:
                conv += (f" <span style='color:#888;'>"
                         f"({lo:.4g}–{hi:.4g})</span>")
            bits.append(conv)
        if bw_mhz is not None:
            bits.append(f"rest-frame bin: <b>{bw_mhz:.4g} MHz</b>")
        if not bits:
            return ("<span style='color:#888;'>V↔MHz scale unavailable "
                    "(no per-event frequency/voltage data).</span>")
        return ("<span style='color:#90caf9;'>Scale:</span> "
                + " &nbsp;|&nbsp; ".join(bits))

    def _jump_to_diagnostics(self, row, _col):
        if 0 <= row < len(self._infos):
            label, res, err = self._infos[row]
            if err is not None or not res or not res.get("diagnostics"):
                return  # no plot available for this row
            idx = self._run_combo.findText(label)
            if idx >= 0:
                self._run_combo.setCurrentIndex(idx)
            self._tabs.setCurrentIndex(1)   # Run detail

    def _on_tab_changed(self, idx):
        """Show the picker only on the per-run detail tab; dispatch the
        render (Compare renders all runs at once)."""
        self._picker_row.setVisible(idx == 1 and bool(self._plottable))
        if not self._plottable:
            return
        self._refresh_active_plot()

    def _refresh_active_plot(self):
        idx = self._tabs.currentIndex()
        if idx == 1:
            self._render_detail()
        elif idx == 2:
            self._render_compare()

    def _current_res(self):
        """The compute_binned result for the currently picked run, or None."""
        if not self._plottable:
            return None
        return self._plottable.get(self._run_combo.currentText())

    # ── Run detail tab ─────────────────────────────────────────────

    def _build_detail_tab(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavToolbar)
        from matplotlib.figure import Figure

        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        ctrl_row = QHBoxLayout()
        self._shading_cb = QCheckBox("Show bin shading")
        self._shading_cb.toggled.connect(self._render_detail)
        ctrl_row.addWidget(self._shading_cb)
        self._rug_cb = QCheckBox("Show raw event rug")
        self._rug_cb.setToolTip(
            "Show a strip of raw event x-positions (sampled, max 20k)\n"
            "below the spectrum. Reveals event clustering or sparse\n"
            "regions that the binned view hides.")
        self._rug_cb.toggled.connect(self._render_detail)
        ctrl_row.addWidget(self._rug_cb)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._diag_title = QLabel()
        self._diag_title.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._diag_title)
        # Scan-voltage ↔ rest-frame-frequency conversion strip.
        self._diag_scale = QLabel()
        self._diag_scale.setTextFormat(Qt.TextFormat.RichText)
        self._diag_scale.setWordWrap(True)
        lay.addWidget(self._diag_scale)
        self._diag_warn = QLabel()
        self._diag_warn.setTextFormat(Qt.TextFormat.RichText)
        self._diag_warn.setWordWrap(True)
        lay.addWidget(self._diag_warn)

        self._diag_fig = Figure(figsize=(8.0, 4.5))
        self._diag_canvas = FigureCanvasQTAgg(self._diag_fig)
        lay.addWidget(make_plot_card(
            "Bin edges over the spectrum", self._diag_canvas,
            NavToolbar(self._diag_canvas, self), parent=self,
            subtitle="how the data was cut into bins"), 1)

        if not self._plottable:
            self._diag_title.setText(
                "<i>No diagnostics available "
                "(no checked runs produced bin edges).</i>")
        return wrap

    def _render_detail(self):
        res = self._current_res()
        if not res:
            return
        label = self._run_combo.currentText()
        info = res.get("info", {}) or {}
        diag = res.get("diagnostics", {}) or {}
        n_bins = len(diag.get("bin_centers", []))

        # Rebuild shading checkbox state for the newly-selected run.
        self._shading_cb.blockSignals(True)
        self._shading_cb.setEnabled(n_bins <= self._SHADING_LIMIT)
        if n_bins > self._SHADING_LIMIT:
            self._shading_cb.setChecked(False)
            self._shading_cb.setToolTip(
                f"Disabled because n_bins > {self._SHADING_LIMIT}; "
                f"shading would be visually noisy.")
        else:
            self._shading_cb.setChecked(True)
            self._shading_cb.setToolTip("")
        self._shading_cb.blockSignals(False)

        # Rug checkbox: disable if no raw sample was captured for this run.
        raw_sample = np.asarray(diag.get("raw_x_sample", []), dtype=float)
        raw_n = len(raw_sample)
        self._rug_cb.blockSignals(True)
        self._rug_cb.setEnabled(raw_n > 0)
        if raw_n == 0:
            self._rug_cb.setChecked(False)
            self._rug_cb.setToolTip(
                "Disabled: no raw event sample available for this run.")
        else:
            self._rug_cb.setToolTip(
                f"Toggle a strip of {raw_n} sampled raw event "
                f"x-positions below the spectrum.")
        self._rug_cb.blockSignals(False)

        x = np.asarray(res["x"], dtype=float)
        y = np.asarray(res["y"], dtype=float)
        yerr = res["yerr"]
        if callable(yerr):
            yerr = np.sqrt(np.maximum(y, 1.0))
        else:
            yerr = np.asarray(yerr, dtype=float)
        edges = np.asarray(diag.get("bin_edges", []), dtype=float)
        widths = np.asarray(diag.get("bin_widths", []), dtype=float)
        have_widths = widths.size > 0
        stats = self._occupancy_stats(y)
        unit = "MHz" if info.get("bin_mode") == "Frequency" else "V"
        rug_on = (self._rug_cb.isChecked() and raw_n > 0)

        # Title strip
        mode = info.get("bin_mode", "?")
        defn = info.get("bin_definition", "?")
        nb = info.get("effective_n_bins", "?")
        if mode == "Frequency":
            w = info.get("effective_bin_width_mhz")
            w_str = f"{w:.4f} MHz" if w else "—"
        else:
            w = info.get("effective_bin_width_v")
            w_str = f"{w:.4f} V" if w else "—"
        parts = [f"<b>{label}</b>", mode, defn,
                 f"{nb} bins", f"width {w_str}"]
        if stats["n_in_scan"]:
            parts.append(f"empty {stats['n_empty']}/{stats['n_in_scan']}")
        if stats["n_gap"]:
            parts.append(f"<span style='color:#888;'>scan-gap bins: "
                         f"{stats['n_gap']}</span>")
        if n_bins > self._EDGES_LIMIT:
            stride = max(1, n_bins // self._EDGES_LIMIT)
            parts.append(f"<i>showing every {stride}th edge</i>")
        self._diag_title.setText(" &nbsp;|&nbsp; ".join(parts))

        # Conversion strip: raw-voltage bin size, the 1 V → MHz factor,
        # and the bin size in the Doppler-shifted rest frame. dν/dV is
        # measured from this run's own gated events (median local
        # slope), so it reflects the actual scan.
        self._diag_scale.setText(self._scale_strip_html(info))

        # Aliasing checks only make sense on a uniform grid; per-step
        # and Raw Voltage widths vary with the scan by construction.
        per_step = (bool(info.get("per_step"))
                    or info.get("bin_mode") == "Raw Voltage")
        uniform_defs = ("Auto", "Fixed bin count", "Fixed bin width")
        aliasing_relevant = (not per_step
                             and info.get("bin_definition") in uniform_defs)

        # Warning strip — binning provenance + occupancy + widths.
        med = stats["median_nonzero"]
        mx = stats["max"]
        ratio = (mx / med) if med > 0 else 0.0
        msgs = []
        if info.get("fallback_used"):
            msgs.append("clstools default failed → sqrt(N) fallback used")
        if info.get("voltage_forced_auto"):
            msgs.append("Raw Voltage groups by unique DV; "
                        "bin_definition forced to Auto")
        req_w = info.get("requested_bin_width_mhz")
        eff_w = info.get("effective_bin_width_mhz")
        if req_w and eff_w and abs(eff_w - req_w) / req_w > 0.05:
            msgs.append(f"requested {req_w:.3f} MHz width; "
                        f"effective {eff_w:.3f} MHz "
                        f"({100 * (eff_w - req_w) / req_w:+.1f}%)")
        es = diag.get("edges_source", "")
        soft_note = ""
        if es and es not in ("clstools_intervals",
                              "midpoints_between_dv_values"):
            msgs.append(f"bin edges are estimated ({es})")
        elif es == "midpoints_between_dv_values":
            soft_note = ("<span style='color:#90caf9;'>Note:</span> Raw "
                         "Voltage groups by unique DV; vertical lines are "
                         "midpoints between centers, not real bin edges.")
        if (stats["n_empty"] > 0.20 * stats["n_in_scan"]
                and stats["n_empty"] >= 5):
            msgs.append(f"{stats['n_empty']} empty bins inside scan range "
                        f"({100*stats['n_empty']/stats['n_in_scan']:.0f}%)")
        if ratio >= 5.0 and mx >= 50:
            msgs.append(f"single bin dominates "
                        f"({mx:.0f} counts vs median {med:.0f})")
        if stats["n_low"] > 0.30 * stats["n_in_scan"] and stats["n_low"] >= 5:
            msgs.append(f"{stats['n_low']} low-count bins (&lt;10) — "
                        f"chi-square may be biased; consider Poisson LLH")
        narrow_mask = None
        if have_widths and aliasing_relevant:
            w_med = float(np.median(widths))
            w_mn = float(widths.min())
            w_mx = float(widths.max())
            if w_mn > 0 and (w_mx / w_mn) >= 1.5:
                msgs.append(f"width spread {w_mx / w_mn:.2f}× across bins")
            if w_med > 0:
                narrow_mask = widths < 0.5 * w_med
            if narrow_mask is not None and narrow_mask.any():
                msgs.append(
                    f"{int(narrow_mask.sum())} bin(s) narrower than half "
                    f"the median — a uniform grid can merge two scan "
                    f"steps (doubled-count spike); use Per scan step")
        if msgs:
            self._diag_warn.setText(
                f"<span style='color:#ffb74d;'>⚠ "
                f"{' · '.join(msgs)}</span>")
            self._diag_warn.show()
        elif soft_note:
            self._diag_warn.setText(soft_note)
            self._diag_warn.show()
        else:
            self._diag_warn.clear()
            self._diag_warn.hide()

        # Plot: spectrum with bin edges (+ optional raw-event rug).
        # Occupancy and width health show as numbers/warnings above —
        # per Arda, no extra panels.
        self._diag_fig.clear()
        if rug_on:
            axs = self._diag_fig.subplots(
                2, 1, sharex=True,
                gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05})
            ax, ax_rug = axs
        else:
            ax = self._diag_fig.add_subplot(111)
            ax_rug = None

        if self._shading_cb.isChecked() and n_bins <= self._SHADING_LIMIT:
            for i in range(0, len(edges) - 1, 2):
                ax.axvspan(edges[i], edges[i + 1],
                           facecolor="#888888", alpha=0.08, zorder=0)
        if len(edges) > 1:
            stride = 1 if n_bins <= self._EDGES_LIMIT else max(
                1, n_bins // self._EDGES_LIMIT)
            for e in edges[::stride]:
                ax.axvline(e, color="#999999", linewidth=0.5,
                           alpha=0.45, zorder=1)
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=3.5, color="black",
                    ecolor="#333333", capsize=1.5, zorder=2)
        ax.set_ylabel("Counts")
        ax.set_title(f"Binning diagnostics — {label}", fontsize=10)

        if ax_rug is not None:
            ax_rug.eventplot(raw_sample, lineoffsets=0.5,
                             linelengths=0.9, linewidths=0.4,
                             colors="#1976d2", alpha=0.4)
            ax_rug.set_yticks([])
            ax_rug.set_xlabel(res.get("x_label", f"bin center [{unit}]"))
            ax_rug.set_ylabel(f"raw\n(n={raw_n})",
                              fontsize=7, rotation=0,
                              labelpad=22, va="center")
            ax_rug.set_ylim(0, 1)
            for spine in ("top", "right", "left"):
                ax_rug.spines[spine].set_visible(False)
        else:
            ax.set_xlabel(res.get("x_label", f"bin center [{unit}]"))

        # tight_layout warns harmlessly when an eventplot axis is present;
        # the layout still renders correctly.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", category=UserWarning)
            self._diag_fig.tight_layout()
        self._diag_canvas.draw()

    # ── Compare tab (per-run heatmap) ──────────────────────────────

    def _build_compare_tab(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavToolbar)
        from matplotlib.figure import Figure

        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        ctrl = QHBoxLayout()
        self._cmp_log_cb = QCheckBox("log color scale")
        self._cmp_log_cb.setToolTip(
            "Use log10(counts + 1) for the heatmap color, useful when one\n"
            "run dominates or count range spans many decades.")
        self._cmp_log_cb.toggled.connect(self._render_compare)
        ctrl.addWidget(self._cmp_log_cb)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        self._cmp_title = QLabel()
        self._cmp_title.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._cmp_title)
        self._cmp_warn = QLabel()
        self._cmp_warn.setTextFormat(Qt.TextFormat.RichText)
        self._cmp_warn.setWordWrap(True)
        lay.addWidget(self._cmp_warn)

        self._cmp_fig = Figure(figsize=(8.0, 4.5))
        self._cmp_canvas = FigureCanvasQTAgg(self._cmp_fig)
        lay.addWidget(make_plot_card(
            "Run comparison", self._cmp_canvas,
            NavToolbar(self._cmp_canvas, self), parent=self,
            subtitle="the same binning across every checked run"), 1)

        if not self._plottable:
            self._cmp_title.setText("<i>No diagnostics available.</i>")
        return wrap

    def _render_compare(self):
        from gui.analysis.binning import rebin_to_common_grid
        if not self._plottable:
            return
        # All plottable runs must share bin_mode for the heatmap to make
        # sense — the x-axis units would differ otherwise.
        items = list(self._plottable.items())
        modes = {(res.get("info") or {}).get("bin_mode") for _, res in items}
        if len(modes) != 1:
            self._cmp_title.setText("")
            self._cmp_warn.setText(
                "<span style='color:#ffb74d;'>⚠ mixed bin modes "
                f"({modes}) — Compare requires all runs on the same "
                "axis (Frequency or Raw Voltage).</span>")
            self._cmp_warn.show()
            self._cmp_fig.clear()
            self._cmp_canvas.draw()
            return

        only_mode = next(iter(modes))
        unit = "MHz" if only_mode == "Frequency" else "V"
        per_source = [(np.asarray(res["x"], dtype=float),
                       np.asarray(res["y"], dtype=float))
                      for _, res in items]
        rebinned, centers, bw = rebin_to_common_grid(per_source)
        if len(centers) == 0:
            self._cmp_title.setText(
                "<i>No common-grid data to display.</i>")
            self._cmp_warn.clear(); self._cmp_warn.hide()
            self._cmp_fig.clear()
            self._cmp_canvas.draw()
            return

        labels = [lbl for lbl, _ in items]
        Z = np.array([yr for _, yr in rebinned])  # shape (n_runs, n_bins)

        n_runs, n_bins = Z.shape
        self._cmp_title.setText(
            f"<b>{n_runs} runs</b> on common grid: "
            f"{n_bins} bins @ {bw:.4f} {unit}")
        self._cmp_warn.clear(); self._cmp_warn.hide()

        self._cmp_fig.clear()
        ax = self._cmp_fig.add_subplot(111)
        if self._cmp_log_cb.isChecked():
            disp = np.log10(Z + 1.0)
            cb_label = "log10(counts + 1)"
        else:
            disp = Z
            cb_label = "counts"
        x_lo = centers[0] - 0.5 * bw
        x_hi = centers[-1] + 0.5 * bw
        im = ax.imshow(disp, aspect="auto", origin="lower",
                       extent=(x_lo, x_hi, -0.5, n_runs - 0.5),
                       cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(n_runs))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"{'Frequency' if only_mode == 'Frequency' else 'Voltage'}"
                      f" [{unit}]")
        ax.set_ylabel("Run")
        ax.set_title("Per-run counts on common grid", fontsize=10)
        cbar = self._cmp_fig.colorbar(im, ax=ax)
        cbar.set_label(cb_label)
        self._cmp_fig.tight_layout()
        self._cmp_canvas.draw()


# ══════════════════════════════════════════════════════════════════
#  Source Block
# ══════════════════════════════════════════════════════════════════

class SourceBlock(AnalysisBlock):
    """Source block: file selection, physics params, gates, binning."""
    BLOCK_TYPE = "Source"
    BLOCK_COLOR = "#2196F3"
    # Narrowed 476 → 420 (2026-07 feedback): the input fields had a
    # band of unused width. Width is persisted per project, so saved
    # blocks keep their stored width.
    BLOCK_WIDTH = 420

    def __init__(self, name="Source_1", parent=None):
        super().__init__(name, parent)
        layout = self._content_layout

        # Re-emit block_changed when the global scan filter mutates so
        # any Preview cache and the Analysis fitter both pick up the
        # exclusion. Subscribing here (per-instance) is fine because
        # SourceBlock instances are short-lived (rebuilt on project
        # load) and Qt drops the connection when the block is
        # destroyed.
        from gui.scan_filter import get_registry as _get_sf_registry
        _get_sf_registry().filters_changed.connect(
            self.block_changed.emit)

        # ── Files section ──
        files_grp = QGroupBox("Files")
        files_lay = QVBoxLayout(files_grp)
        files_lay.setContentsMargins(4, 8, 4, 4)
        files_lay.setSpacing(2)

        btn_row = QHBoxLayout()
        self._add_files_btn = QPushButton("+ Add Files")
        self._add_files_btn.setToolTip("Browse for ASDF run files to add")
        self._add_files_btn.clicked.connect(self._add_files)
        btn_row.addWidget(self._add_files_btn)
        self._pull_files_btn = QPushButton("Import from Pre-Analysis")
        self._pull_files_btn.setToolTip(
            "Pick a Pre-Analysis project and import everything at once:\n"
            "files, physics parameters, gates, and cooler/laser overrides.")
        self._pull_files_btn.clicked.connect(self._import_all_from_preanalysis)
        btn_row.addWidget(self._pull_files_btn)
        self._merge_btn = QPushButton("Merge Selected")
        self._merge_btn.setToolTip("Merge checked files into a single spectrum")
        self._merge_btn.clicked.connect(self._merge_selected)
        btn_row.addWidget(self._merge_btn)
        files_lay.addLayout(btn_row)

        # Master "check all" tickbox above the file list. Tri-state
        # (Checked / Unchecked / PartiallyChecked) reflects the per-file
        # boxes; clicking it forces every entry to checked-or-unchecked
        # so the user doesn't need a click per row on long lists.
        master_row = QHBoxLayout()
        master_row.setContentsMargins(2, 0, 2, 0)
        master_row.setSpacing(4)
        self._master_check = QCheckBox("Check all")
        self._master_check.setTristate(True)
        self._master_check.setEnabled(False)
        self._master_check.setToolTip(
            "Check or uncheck every file in this Source block at once.\n"
            "Mixed state (square) appears when only some files are checked.")
        self._master_check.clicked.connect(self._on_master_check_clicked)
        master_row.addWidget(self._master_check)
        master_row.addStretch()
        files_lay.addLayout(master_row)

        self._file_list_widget = QWidget()
        self._file_list_layout = QVBoxLayout(self._file_list_widget)
        self._file_list_layout.setContentsMargins(0, 0, 0, 0)
        self._file_list_layout.setSpacing(1)
        files_lay.addWidget(self._file_list_widget)

        self._file_entries = []  # list of dicts: {path, run_number, checkbox}
        layout.addWidget(files_grp)

        # A calibration can be changed from Pre-Analysis, from the overview,
        # or by loading a project -- keep the [cal] badges and the blinking
        # alerts honest wherever the change came from.
        from gui.calibration import get_registry as _get_cal_registry
        _get_cal_registry().calibrations_changed.connect(
            self._refresh_calibration_badges)

        # ── Physics Parameters ──
        phys_grp = QGroupBox("Physics Parameters")
        phys_form = QFormLayout(phys_grp)
        phys_form.setContentsMargins(4, 8, 4, 4)

        # Z + A row
        za_row = QHBoxLayout()
        za_row.setSpacing(6)
        za_row.addWidget(QLabel("Z:"))
        self._z_number = QSpinBox()
        self._z_number.setRange(1, 118)
        self._z_number.setValue(1)
        self._z_number.setToolTip("Atomic number Z")
        self._z_number.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._z_number.valueChanged.connect(self._on_za_changed)
        za_row.addWidget(self._z_number, 1)
        za_row.addWidget(QLabel("A:"))
        self._a_number = QSpinBox()
        self._a_number.setRange(1, 300)
        self._a_number.setValue(1)
        self._a_number.setToolTip("Mass number A")
        self._a_number.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._a_number.valueChanged.connect(self._on_za_changed)
        za_row.addWidget(self._a_number, 1)
        self._element_label = QLabel("")
        self._element_label.setStyleSheet(
            "color: #ddd; font-weight: bold; padding: 0 8px;")
        za_row.addWidget(self._element_label)
        phys_form.addRow(za_row)

        mass_row = QHBoxLayout()
        mass_row.setSpacing(6)
        self._mass_spin = _make_double(1.0, 0.001, 400.0, 6, 0.000001,
                                       tooltip="Atomic mass [amu] used for "
                                       "Doppler-to-frequency binning.\n"
                                       "Auto-filled from the IUPAC/AME table "
                                       "when Z and A change.\n"
                                       "Tick 'Override' to keep a custom value "
                                       "across Z/A edits.")
        self._mass_spin.setReadOnly(True)
        self._mass_spin.setButtonSymbols(
            QDoubleSpinBox.ButtonSymbols.NoButtons)
        mass_row.addWidget(self._mass_spin, 1)
        self._mass_override = QCheckBox("Override")
        self._mass_override.setToolTip(
            "Use the value typed above instead of the table lookup. "
            "When off, changing Z or A re-fills mass automatically.")
        self._mass_override.toggled.connect(self._on_mass_override_toggled)
        mass_row.addWidget(self._mass_override)
        phys_form.addRow("Mass [amu]:", mass_row)

        self._harmonic_spin = _make_int(4, 1, 8, tooltip="Laser harmonic")
        phys_form.addRow("Harmonic:", self._harmonic_spin)

        self._e_lower_spin = _make_double(0.0, -1e6, 1e6, 4, 0.001,
                                          tooltip="Lower atomic energy level [cm\u207b\u00b9].\n"
                                          "e.g. 3350.494 for Co ground state.\n"
                                          "ref = (E_upper \u2212 E_lower) \u00d7 c \u00d7 100")
        phys_form.addRow("E lower (cm\u207b\u00b9):", self._e_lower_spin)

        self._e_upper_spin = _make_double(0.0, -1e6, 1e6, 4, 0.001,
                                          tooltip="Upper atomic energy level [cm\u207b\u00b9].\n"
                                          "e.g. 47078.494 for Co excited state.\n"
                                          "Difference = full rest-frame transition.")
        phys_form.addRow("E upper (cm\u207b\u00b9):", self._e_upper_spin)
        layout.addWidget(phys_grp)

        # ── Gates ──
        gates_grp = QGroupBox("Gates")
        gates_lay = QVBoxLayout(gates_grp)
        gates_lay.setContentsMargins(6, 10, 6, 6)
        gates_lay.setSpacing(6)

        # TOF gate
        tof_row = QHBoxLayout()
        tof_row.setSpacing(6)
        self._tof_enable = QCheckBox("TOF gate:")
        self._tof_enable.setChecked(True)
        self._tof_enable.setToolTip(
            "Enable time-of-flight gate to select the ion bunch.\n"
            "Set lower and upper bounds in \u00b5s.")
        tof_row.addWidget(self._tof_enable)
        self._tof_lo = _make_double(38.0, 0, 1000, 1, 0.5)
        tof_row.addWidget(self._tof_lo, 1)
        tof_row.addWidget(QLabel("\u2013"))  # en-dash
        self._tof_hi = _make_double(44.0, 0, 1000, 1, 0.5)
        tof_row.addWidget(self._tof_hi, 1)
        gates_lay.addLayout(tof_row)

        # PMT channels
        pmt_row = QHBoxLayout()
        pmt_row.setSpacing(8)
        pmt_lbl = QLabel("PMT:")
        pmt_lbl.setToolTip("Select which PMT detector channels to include")
        pmt_row.addWidget(pmt_lbl)
        self._pmt_checks = []
        for i in range(1, 5):
            cb = QCheckBox(str(i))
            cb.setChecked(i >= 3)
            cb.setToolTip(f"Include PMT channel {i}")
            self._pmt_checks.append(cb)
            pmt_row.addWidget(cb)
        cb_dc = QCheckBox("DC")
        cb_dc.setToolTip("Include DC (continuous) channel")
        self._pmt_checks.append(cb_dc)
        pmt_row.addWidget(cb_dc)
        pmt_row.addStretch()
        gates_lay.addLayout(pmt_row)

        # V gate
        vgate_row = QHBoxLayout()
        vgate_row.setSpacing(6)
        self._vgate_enable = QCheckBox("V gate:")
        self._vgate_enable.setToolTip(
            "Voltage gate [min, max] on raw DV column")
        vgate_row.addWidget(self._vgate_enable)
        self._vgate_lo = _make_double(0.0, -1e6, 1e6, 1, 1.0)
        vgate_row.addWidget(self._vgate_lo, 1)
        vgate_row.addWidget(QLabel("\u2013"))
        self._vgate_hi = _make_double(0.0, -1e6, 1e6, 1, 1.0)
        vgate_row.addWidget(self._vgate_hi, 1)
        gates_lay.addLayout(vgate_row)

        # F gate (UI in MHz; converted to Hz for clstools in get_fgate)
        fgate_row = QHBoxLayout()
        fgate_row.setSpacing(6)
        self._fgate_enable = QCheckBox("F gate (MHz):")
        self._fgate_enable.setToolTip(
            "Frequency gate [min, max] in MHz, applied to the "
            "Doppler-corrected frequency (relative to the reference). "
            "Stored in Hz at the backend.")
        fgate_row.addWidget(self._fgate_enable)
        self._fgate_lo = _make_double(0.0, -1e6, 1e6, 3, 1.0)
        self._fgate_lo.setToolTip("Lower bound (MHz)")
        fgate_row.addWidget(self._fgate_lo, 1)
        fgate_row.addWidget(QLabel("\u2013"))
        self._fgate_hi = _make_double(0.0, -1e6, 1e6, 3, 1.0)
        self._fgate_hi.setToolTip("Upper bound (MHz)")
        fgate_row.addWidget(self._fgate_hi, 1)
        gates_lay.addLayout(fgate_row)

        # Noise filter
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        filter_row.addWidget(QLabel("Noise filter:"))
        self._noise_filter = QSpinBox()
        self._noise_filter.setRange(0, 100000)
        self._noise_filter.setValue(0)
        self._noise_filter.setToolTip(
            "Max events per timestamp (0=disabled). Removes noise bursts.")
        filter_row.addWidget(self._noise_filter, 1)
        gates_lay.addLayout(filter_row)

        layout.addWidget(gates_grp)

        # ── Binning ──
        bin_grp = QGroupBox("Binning")
        bin_form = QFormLayout(bin_grp)
        bin_form.setContentsMargins(4, 8, 4, 4)

        self._x_col_combo = QComboBox()
        self._x_col_combo.setToolTip(
            "Which per-bin x value to use as the abscissa of the spectrum\n"
            "passed to satlas2 (frequency mode only). Both come from\n"
            "clstools.Compute_Bins; the choice mostly matters in sparse "
            "bins.")
        self._x_col_combo.addItems(["bins_center", "Fmean"])
        self._x_col_combo.setItemData(
            0,
            "Midpoint of each frequency bin (the bin's geometric centre).\n"
            "Independent of how many events landed in the bin → stable and\n"
            "well-defined even for empty / near-empty bins.\n"
            "Pick this for: most fits. It's the safe default.",
            Qt.ItemDataRole.ToolTipRole)
        self._x_col_combo.setItemData(
            1,
            "Mean event frequency inside each bin.\n"
            "More accurate for the bin's true centroid when the spectrum\n"
            "varies steeply across a bin, but noisy when bins hold only\n"
            "a few counts (the mean is dominated by individual events).\n"
            "Pick this for: high-statistics scans where you want the\n"
            "mean rather than the bin midpoint to drive the fit.",
            Qt.ItemDataRole.ToolTipRole)
        bin_form.addRow("x values:", self._x_col_combo)

        self._yerr_combo = QComboBox()
        self._yerr_combo.setToolTip(
            "How per-bin y-uncertainties are computed for the fit.\n"
            "All three assume Poisson counting statistics; they differ in\n"
            "how the variance is estimated.")
        self._yerr_combo.addItems(["None", "Poisson sqrt(y+1)", "Poisson sqrt(y)",
                                   "Model-based"])
        # Seed new-block defaults from the app Settings ("Fitting Defaults").
        # These were saved but never read, so the controls looked effective but
        # did nothing (code review 2026-06-02, settings-fitting-defaults-write-
        # only). from_dict() overrides them for a loaded project, so this only
        # affects freshly-added blocks; setCurrentText is a no-op for an unknown
        # value, so a stale setting can't break the combo.
        _sd = _load_settings()
        self._x_col_combo.setCurrentText(_sd.get("x_column", "bins_center"))
        self._yerr_combo.setCurrentText(
            _sd.get("yerr_mode", "Poisson sqrt(y+1)"))
        self._yerr_combo.setItemData(
            0,
            "yerr = None — the spectrum is rendered without error bars\n"
            "and the fit treats every bin as having weight 1.\n"
            "Pick this for: quick visual inspection only. Not recommended\n"
            "for production fits — the χ² minimisation becomes a plain\n"
            "least-squares minimisation with no statistical interpretation.",
            Qt.ItemDataRole.ToolTipRole)
        self._yerr_combo.setItemData(
            1,
            "yerr = sqrt(y + 1)\n"
            "Avoids the zero-error pathology of empty bins and matches\n"
            "the Poisson sigma to ~1% for y ≥ 30. Slightly overestimates\n"
            "the error at very high counts. Low-count bins (y < 10) get\n"
            "a Bayesian Poisson interval correction on top.\n"
            "Pick this for: most fits. Simple, fast, no edge cases.",
            Qt.ItemDataRole.ToolTipRole)
        self._yerr_combo.setItemData(
            2,
            "yerr = sqrt(y)\n"
            "The textbook Poisson sigma. Empty bins (y = 0) would give\n"
            "yerr = 0 and divide-by-zero in the fit, so those are\n"
            "clamped to 1 as a workaround. Same low-count Bayesian\n"
            "interval correction as sqrt(y+1).\n"
            "Pick this for: completeness / comparison with sqrt(y+1).\n"
            "Generally not recommended — the zero-clamp is a hack.",
            Qt.ItemDataRole.ToolTipRole)
        self._yerr_combo.setItemData(
            3,
            "yerr = sqrt(model_at_x)  (recomputed every iteration)\n\n"
            "Instead of estimating the Poisson variance from the noisy\n"
            "observed counts y, satlas2 evaluates sqrt of the *expected*\n"
            "counts (the model evaluated at x) at every fit step. This\n"
            "removes the bias that sqrt(y) has toward downward\n"
            "fluctuations: a bin that randomly fluctuated low gets a\n"
            "small sqrt(y) → small error → pulls the fit toward the\n"
            "fluctuation. sqrt(model) doesn't have that bias because\n"
            "the weight tracks the expected mean, not the noisy sample.\n\n"
            "Trade-offs: marginally slower (yerr recomputed each step);\n"
            "the residual error bars depend on model parameters, so\n"
            "during fitting they change as the fit converges. The\n"
            "post-fit residual plot uses sqrt(max(y_fit, 1)) so empty\n"
            "bins stay finite.\n\n"
            "Pick this for: low-count Poisson data where you want the\n"
            "proper MLE-equivalent weighting. For high-count bins\n"
            "(≳30) this agrees with sqrt(y+1) to a fraction of a percent.",
            Qt.ItemDataRole.ToolTipRole)
        bin_form.addRow("yerr mode:", self._yerr_combo)

        self._bin_mode = QComboBox()
        self._bin_mode.setToolTip(
            "Domain in which events are binned to form the spectrum.")
        self._bin_mode.addItems(["Frequency", "Raw Voltage"])
        self._bin_mode.setItemData(
            0,
            "Convert each event's deceleration voltage to a Doppler-\n"
            "shifted frequency, then bin in MHz. Requires mass, the two\n"
            "energy levels, and harmonic (Physics Parameters above).\n"
            "Pick this for: any fit that depends on a physical line\n"
            "frequency — isotope shifts, hyperfine constants, FWHM in\n"
            "MHz, etc.",
            Qt.ItemDataRole.ToolTipRole)
        self._bin_mode.setItemData(
            1,
            "Bin events directly by their calibrated scanning voltage.\n"
            "No mass / energy levels / harmonic needed; clstools groups\n"
            "events by unique DV value. The x-axis is in volts.\n"
            "Pick this for: diagnostic plots, calibration checks, or\n"
            "when physics parameters are not yet known. Bin count and\n"
            "bin width controls do not apply in this mode.",
            Qt.ItemDataRole.ToolTipRole)
        bin_form.addRow("Bin mode:", self._bin_mode)

        self._xerr_combo = QComboBox()
        self._xerr_combo.setToolTip(
            "Optional per-bin x-axis uncertainty propagated into the fit.\n"
            "Only meaningful in Frequency mode; ignored otherwise.")
        self._xerr_combo.addItems(["None", "From voltage std"])
        self._xerr_combo.setItemData(
            0,
            "No x-axis error; satlas2 treats each bin's x as exact.\n"
            "Pick this for: most fits. The bin width is usually much\n"
            "smaller than line widths so x-error is negligible.",
            Qt.ItemDataRole.ToolTipRole)
        self._xerr_combo.setItemData(
            1,
            "Use the per-bin standard deviation of the deceleration\n"
            "voltage (Vstd from clstools) as the x-error, converted to\n"
            "MHz. satlas2 enlarges the effective y-error by\n"
            "sqrt(yerr² + (df/dx · xerr)²) at every iteration.\n"
            "Pick this for: scans with significant cooler-voltage\n"
            "jitter inside a bin, or when you want a more conservative\n"
            "uncertainty estimate near steep model features.",
            Qt.ItemDataRole.ToolTipRole)
        bin_form.addRow("x-error:", self._xerr_combo)

        self._bin_def_combo = QComboBox()
        self._bin_def_combo.setToolTip(
            "How the number of frequency bins is decided. Frequency mode\n"
            "only — Raw Voltage mode always groups by unique DV.")
        self._bin_def_combo.addItems(list(BIN_DEFINITIONS))
        # Default = the aliasing-safe per-step binning (same default the
        # Pre-Analysis spectrum uses); from_dict keeps a loaded project's
        # explicit choice.
        self._bin_def_combo.setCurrentText(DEFAULT_BIN_DEFINITION)
        self._bin_def_combo.setItemData(
            0,
            "clstools lays a UNIFORM grid of (maxF − minF) /\n"
            "Frequency_stepsize equal-width bins over the scan.\n"
            "Because the voltage→frequency map is Doppler-nonlinear,\n"
            "that grid sits at the aliasing point: where the step\n"
            "spacing dips below the bin width, two scan steps share a\n"
            "bin (a doubled-count spike) while other bins fall empty.\n"
            "Pick this for: comparison with legacy analyses only —\n"
            "'Per scan step' is the safe default.",
            Qt.ItemDataRole.ToolTipRole)
        self._bin_def_combo.setItemData(
            1,
            "Use exactly the value in 'Bin count' below, regardless of\n"
            "scan step or frequency range. Useful for comparing two runs\n"
            "with different scan resolutions on the same x grid.\n"
            "Caveat: very high counts give over-fine bins (mostly empty);\n"
            "very low counts give under-resolved peaks.",
            Qt.ItemDataRole.ToolTipRole)
        self._bin_def_combo.setItemData(
            2,
            "Target bin width in MHz (set in 'Bin width [MHz]' below).\n"
            "Two-pass: first call Compute_Bins to learn the gated\n"
            "frequency range, then re-bin with N = round(range / width).\n"
            "Caveat: clstools accepts an integer bin count, so the\n"
            "actual width drifts slightly from the requested value\n"
            "(reported in binning_summary.csv as effective_bin_width_mhz).\n"
            "A warning fires if the drift exceeds 5%.",
            Qt.ItemDataRole.ToolTipRole)
        self._bin_def_combo.setItemData(
            3,
            "One bin per raw scanning-voltage step, centred at the mean\n"
            "Doppler-shifted frequency of that step's own events — the\n"
            "frequency-domain equivalent of the raw voltage bins.\n"
            "Cannot alias: no uniform grid is involved, every step is\n"
            "its own bin. 'Bin multiple' groups N adjacent steps when\n"
            "coarser bins are wanted.\n"
            "Pick this for: most fits (the default; matches the\n"
            "Pre-Analysis spectrum view).",
            Qt.ItemDataRole.ToolTipRole)
        bin_form.addRow("Bin definition:", self._bin_def_combo)

        self._bin_count_spin = _make_int(
            DEFAULT_BIN_COUNT, 1, 1000000,
            tooltip=(
                "Exact number of frequency bins (Fixed bin count mode).\n"
                "Spread evenly across the gated min→max frequency range.\n"
                "Inactive in other modes; ignored in Raw Voltage."))
        bin_form.addRow("Bin count:", self._bin_count_spin)

        self._bin_width_spin = _make_double(
            DEFAULT_BIN_WIDTH_MHZ, 1e-4, 1e6, 4, 1.0,
            tooltip=(
                "Target bin width in MHz (Fixed bin width mode).\n"
                "Approximate: clstools converts width to an integer bin\n"
                "count, so the realised width may differ by a few percent.\n"
                "The actual width per run is in binning_summary.csv as\n"
                "effective_bin_width_mhz; a warning fires if the drift\n"
                "is larger than ~5%.\n"
                "Inactive in other modes; ignored in Raw Voltage."))
        bin_form.addRow("Bin width [MHz]:", self._bin_width_spin)

        self._step_mult_spin = _make_int(
            1, 1, 1000,
            tooltip=(
                "Group N adjacent native step bins into one (Per scan\n"
                "step and Raw Voltage modes). 1 = one bin per raw\n"
                "scanning-voltage step (the native resolution); 2 sums\n"
                "neighbouring steps pairwise, and so on. Grouping keeps\n"
                "the scan's own sampling, so it cannot alias the way a\n"
                "uniform grid does."))
        bin_form.addRow("Bin multiple:", self._step_mult_spin)

        # Enable/disable bin_count + bin_width based on definition + mode.
        self._bin_def_combo.currentTextChanged.connect(
            self._update_bin_def_controls)
        self._bin_mode.currentTextChanged.connect(
            self._update_bin_def_controls)
        # Domain-mismatch warning on merged entries follows the same
        # bin_mode change. Wired here (rather than inside the merged-
        # entry constructor) so the badge updates even when the entry
        # already exists and the user is the one flipping bin_mode.
        self._bin_mode.currentTextChanged.connect(
            lambda *_: self._refresh_merged_warning_badges())
        self._update_bin_def_controls()

        layout.addWidget(bin_grp)

        # ── Advanced Data Settings ──
        adv_grp = QGroupBox("Advanced Data Settings")
        adv_form = QFormLayout(adv_grp)
        adv_form.setContentsMargins(4, 8, 4, 4)

        # Calibration order (#38, #41)
        self._cal_order = QSpinBox()
        self._cal_order.setRange(1, 3)
        self._cal_order.setValue(1)
        self._cal_order.setToolTip(
            "Polynomial order for the LCR-voltage calibration done at file "
            "load.\n"
            "clstools fits CalSet (commanded values) vs CalReadback "
            "(measured)\n"
            "and applies the result to every event:\n"
            "    DV_cal = (a₀ + a₁·DV + a₂·DV² + a₃·DV³) · VAccDiv\n\n"
            "  1 = linear (slope + offset) — the default, almost always "
            "right.\n"
            "  2/3 = parabolic / cubic — only useful when the LCR readback\n"
            "        is visibly non-linear across the scan range.")
        adv_form.addRow("Cal. order:", self._cal_order)

        # Cooler correction mode (#42)
        self._cooler_corr = QComboBox()
        self._cooler_corr.addItems(["pbp", "mean"])
        self._cooler_corr.setToolTip(
            "How the cooler-voltage drift is subtracted from each event's\n"
            "post-deceleration voltage in clstools.Compute_Voltages:\n"
            "    V = Vcooler · VCoolDiv + VCoolOffset − DV_cal\n\n"
            "  pbp  = point-by-point: use the cooler voltage measured at "
            "each\n"
            "         event's timestamp. Tracks drift through the run.\n"
            "         Pick this when the cooler isn't perfectly stable.\n\n"
            "  mean = run-averaged cooler voltage for every event. Smooths\n"
            "         through cooler-readback noise but is blind to drift.\n"
            "         Pick this when the cooler is known stable and the\n"
            "         readback is noisy.")
        adv_form.addRow("Cooler corr.:", self._cooler_corr)

        # Reference shift (#37) — stored & displayed in MHz to match the
        # spectrum x-axis and the F-gate fields. clstools.Shift_Ref takes Hz,
        # so get_source_config converts MHz -> Hz at the boundary.
        self._ref_shift = _make_double(
            0.0, -1e9, 1e9, 6, 1.0,
            tooltip=(
                "Additive offset to the reference frequency that defines "
                "x = 0\n"
                "on the spectrum (clstools.Shift_Ref):\n"
                "    F = WN_to_f · WN − Reference\n\n"
                "  0      = use the rest-frame transition derived from\n"
                "           E upper − E lower (Physics Parameters above).\n"
                "  +value = pushes the spectrum to LOWER MHz.\n"
                "  −value = pushes the spectrum to HIGHER MHz.\n\n"
                "Use this to recenter when your line is offset from the\n"
                "literature reference, or when comparing isotopes against\n"
                "a shared reference frequency."))
        adv_form.addRow("Ref. shift [MHz]:", self._ref_shift)

        layout.addWidget(adv_grp)

        # ── Cooler / Laser Override ──
        self._ovr_enable = QCheckBox("Cooler / Laser Override")
        self._ovr_enable.setToolTip(
            "Force every file in this Source to use the same cooler voltage\n"
            "and/or laser setpoint instead of the values stored in each\n"
            "ASDF file's metadata. Affects clstools.Compute_Voltages "
            "(beam energy)\n"
            "and Compute_WL (Doppler shift).\n\n"
            "  Disabled → each file uses its own ASDF-recorded values.\n"
            "  Enabled  → ALL files use the override values below.\n\n"
            "Use when ASDF metadata is wrong (e.g. cooler offset mis-"
            "logged),\n"
            "or to test sensitivity of the fit to small changes in cooler/\n"
            "laser values.")
        layout.addWidget(self._ovr_enable)
        self._ovr_container = QWidget()
        ovr_form = QFormLayout(self._ovr_container)
        ovr_form.setContentsMargins(16, 4, 4, 4)
        self._cooler_spin = _make_double(0.0, 0, 1e6, 2, 1.0,
                                         tooltip="Total cooler voltage [V].\n"
                                         "ALL runs will use this fixed value for\n"
                                         "beam energy: V = override \u2212 DV_cal.")
        ovr_form.addRow("Cooler [V]:", self._cooler_spin)
        self._laser_spin = _make_double(0.0, 0, 1e6, 6, 0.0001,
                                        tooltip="Laser setpoint [cm\u207b\u00b9].\n"
                                        "ALL runs will use this fixed value for\n"
                                        "the Doppler shift calculation.")
        ovr_form.addRow("Laser [cm\u207b\u00b9]:", self._laser_spin)
        self._ovr_container.setVisible(False)
        self._ovr_enable.toggled.connect(self._ovr_container.setVisible)
        layout.addWidget(self._ovr_container)

        # ── Preview Navigation ──
        preview_nav = QHBoxLayout()
        self._prev_btn = QToolButton()
        self._prev_btn.setText("\u25c0")
        self._prev_btn.setFixedSize(24, 24)
        self._prev_btn.setToolTip("Previous file")
        self._prev_btn.clicked.connect(lambda: self._navigate_preview(-1))
        preview_nav.addWidget(self._prev_btn)

        self._preview_idx_label = QLabel("0 / 0")
        self._preview_idx_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        self._preview_idx_label.setFixedWidth(60)
        preview_nav.addWidget(self._preview_idx_label)

        self._next_btn = QToolButton()
        self._next_btn.setText("\u25b6")
        self._next_btn.setFixedSize(24, 24)
        self._next_btn.setToolTip("Next file")
        self._next_btn.clicked.connect(lambda: self._navigate_preview(1))
        preview_nav.addWidget(self._next_btn)

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setToolTip("Open preview for checked files")
        self._preview_btn.clicked.connect(self._open_preview)
        preview_nav.addWidget(self._preview_btn)

        self._run_info_btn = QPushButton("Meta")
        self._run_info_btn.setToolTip("Show metadata for current file")
        self._run_info_btn.clicked.connect(self._show_run_info)
        preview_nav.addWidget(self._run_info_btn)

        self._diag_btn = QPushButton("Run Stats")
        self._diag_btn.setToolTip(
            "Plot experimental parameters vs run number\n"
            "(cooler V, laser setpoint, DAQ time, events)")
        self._diag_btn.clicked.connect(self._show_source_diagnostics)
        preview_nav.addWidget(self._diag_btn)

        self._binning_btn = QPushButton("Binning")
        self._binning_btn.setToolTip(
            "Open the Binning view for every checked file:\n"
            "  • Summary tab — per-run table of mode, definition, bin count,\n"
            "    width, and occupancy (empty / median / total counts).\n"
            "  • Run detail tab — spectrum + bin edges for one run, with the\n"
            "    V↔MHz scale strip and any binning warnings.\n"
            "  • Compare tab — all runs re-binned onto one grid as a heatmap.\n"
            "Useful to spot mis-configured runs before running a fit.")
        self._binning_btn.clicked.connect(
            lambda: self._open_binning_dialog())
        preview_nav.addWidget(self._binning_btn)
        layout.addLayout(preview_nav)

        layout.addStretch()

        # Mass table (loaded once)
        try:
            self._mass_table = load_mass_table()
        except Exception:
            self._mass_table = {}

        # Preview / Meta navigation state
        self._preview_win = None
        self._meta_win = None
        self._meta_label = None  # QLabel inside _meta_win
        self._preview_index = 0

    # ── File management ──

    def _add_files(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select ASDF or .vasdf Files",
            get_last_dir("data", "load"),
            "ASDF and virtual splits (*.asdf *.vasdf);;"
            "ASDF files (*.asdf);;"
            "Virtual splits (*.vasdf);;"
            "All files (*)")
        if not paths:
            return
        remember_last_dir("data", "load", paths[0])
        for p in paths:
            self._add_file_entry(p)
        self.block_changed.emit()

    def _add_file_entry(self, filepath):
        # Branch on extension. .vasdf files carry split metadata that
        # the fitter needs to compose the V-gate; everything else flows
        # the same as a real ASDF.
        # DENIS-labelled merged ASDF? Route to _add_merged_entry so
        # the user can re-open an exported merge as a first-class
        # merged source-block entry instead of trying to run it
        # through clstools' cal-polynomial fit (which fails with
        # "data points must exceed order" on the binned merge data).
        try:
            from gui.analysis.merge import (
                is_merged_asdf, load_merged_asdf)
            if is_merged_asdf(filepath):
                try:
                    md = load_merged_asdf(filepath)
                except ValueError as exc:
                    QMessageBox.warning(
                        self, "Merged ASDF inconsistency",
                        f"Cannot load this merged ASDF:\n\n{exc}")
                    return
                if md is not None:
                    self._add_merged_entry(md)
                    return
        except ImportError:
            pass

        is_split = False
        parent_path = None
        source_id = None
        split_lo = None
        split_hi = None
        metadata_override = None
        run_num = "?"
        label_text = None

        if is_vasdf_path(filepath):
            try:
                desc = read_vasdf(filepath)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Bad .vasdf",
                    f"Failed to read descriptor:\n{filepath}\n{exc}")
                return
            is_split = True
            parent_path = desc["parent_path"]
            source_id = desc["source_id"]
            split_lo = desc["split"]["lo"]
            split_hi = desc["split"]["hi"]
            metadata_override = desc.get("metadata_override", {}) or {}
            run_num = source_id
            label_text = f"✂ {desc.get('label') or source_id}"
        else:
            base = os.path.basename(filepath)
            for part in base.replace(".asdf", "").split("_"):
                if part.isdigit():
                    run_num = part
                    break
            label_text = f"run_{run_num}"

        entry = {"path": filepath, "run_number": run_num,
                 "binning_override": {}}
        if is_split:
            entry["is_split"] = True
            entry["parent_path"] = parent_path
            entry["source_id"] = source_id
            entry["split_lo"] = split_lo
            entry["split_hi"] = split_hi
            entry["metadata_override"] = metadata_override

        row = QHBoxLayout()
        cb = QCheckBox()
        cb.setChecked(True)
        cb.toggled.connect(lambda: self.block_changed.emit())
        cb.toggled.connect(self._refresh_master_check_state)
        row.addWidget(cb)
        entry["checkbox"] = cb

        lbl = QLabel(label_text)
        # Tooltip exposes the actual file plus, for splits, the parent
        # ASDF and gate so the user can verify what they loaded.
        if is_split:
            mo_pieces = []
            if metadata_override.get("cooler_v") is not None:
                mo_pieces.append(
                    f"cooler {metadata_override['cooler_v']:.2f} V")
            if metadata_override.get("laser_sp") is not None:
                mo_pieces.append(
                    f"laser {metadata_override['laser_sp']:.6f} cm⁻¹")
            mo_text = ("\nOverride: " + ", ".join(mo_pieces)
                       if mo_pieces else "")
            lbl.setToolTip(
                f".vasdf: {filepath}\n"
                f"Parent: {parent_path}\n"
                f"V-gate: [{split_lo:.2f}, {split_hi:.2f}] V"
                f"{mo_text}")
        else:
            lbl.setToolTip(filepath)
        row.addWidget(lbl, 1)

        # Blinking "!" when this run's calibration outliers would actually move
        # its centroid. Merged spectra have no ASDF behind them, so nothing to
        # warn about; a split keys off its parent's calibration table. The path
        # is resolved lazily because the entry dict is still being filled in.
        from gui.calibration_alert import CalibrationAlertBadge
        alert = CalibrationAlertBadge(
            path_fn=lambda e=entry: (
                None if e.get("is_merged")
                else (e.get("parent_path") if e.get("is_split")
                      else e.get("path"))),
            run_label=f"run_{run_num}",
            physics_fn=lambda e=entry: self._cal_physics(e),
            peers_fn=self._cal_peers,
            cal_order_fn=lambda: self._cal_order.value(),
            parent=self)
        row.addWidget(alert)
        entry["cal_alert"] = alert

        badge = QLabel("[override]")
        badge.setStyleSheet(
            "color: #ffb74d; font-weight: bold; padding: 0 4px;")
        badge.hide()
        row.addWidget(badge)
        entry["badge_label"] = badge

        rm_btn = QToolButton()
        rm_btn.setText("\u2717")
        rm_btn.setFixedSize(18, 18)
        rm_btn.clicked.connect(lambda: self._remove_file_entry(entry, container))
        row.addWidget(rm_btn)

        container = QWidget()
        container.setLayout(row)
        container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        container.customContextMenuRequested.connect(
            lambda pos, e=entry, c=container:
                self._show_file_row_menu(e, c, pos))
        self._file_list_layout.addWidget(container)
        entry["widget"] = container
        self._file_entries.append(entry)
        self._refresh_master_check_state()
        self._update_nav_label()

    def _remove_file_entry(self, entry, widget):
        self._file_list_layout.removeWidget(widget)
        widget.deleteLater()
        if entry in self._file_entries:
            self._file_entries.remove(entry)
        # Keep index in bounds
        checked = self.get_checked_files()
        if checked:
            self._preview_index = min(self._preview_index, len(checked) - 1)
        else:
            self._preview_index = 0
        self._refresh_master_check_state()
        self._update_nav_label()
        self.block_changed.emit()

    def _on_master_check_clicked(self):
        """User clicked the master "Check all" tickbox: force every
        file entry to the opposite of "all currently checked". Mixed
        and all-unchecked states both round up to "check everything"."""
        checkboxes = [e["checkbox"] for e in self._file_entries
                      if "checkbox" in e]
        if not checkboxes:
            return
        all_checked = all(cb.isChecked() for cb in checkboxes)
        new_state = not all_checked
        # Suppress the per-entry block_changed emissions so we only
        # fire one consolidated change at the end.
        for cb in checkboxes:
            cb.blockSignals(True)
            cb.setChecked(new_state)
            cb.blockSignals(False)
        self._refresh_master_check_state()
        self.block_changed.emit()

    def _refresh_master_check_state(self):
        """Sync the master tickbox with the per-entry checkboxes:
        all-checked -> Checked, all-unchecked -> Unchecked, mixed ->
        PartiallyChecked. Disabled when the file list is empty.
        Entries lacking a ``checkbox`` key are ignored -- some tests
        stub _add_file_entry with bare dicts that bypass the real
        widget construction."""
        self._master_check.blockSignals(True)
        try:
            checkboxes = [e["checkbox"] for e in self._file_entries
                          if "checkbox" in e]
            n = len(checkboxes)
            if n == 0:
                self._master_check.setEnabled(False)
                self._master_check.setCheckState(Qt.CheckState.Unchecked)
                return
            self._master_check.setEnabled(True)
            n_checked = sum(1 for cb in checkboxes if cb.isChecked())
            if n_checked == 0:
                self._master_check.setCheckState(Qt.CheckState.Unchecked)
            elif n_checked == n:
                self._master_check.setCheckState(Qt.CheckState.Checked)
            else:
                self._master_check.setCheckState(
                    Qt.CheckState.PartiallyChecked)
        finally:
            self._master_check.blockSignals(False)

    # ── Voltage calibration ──

    def _calibrations(self):
        """Snapshot of the calibration registry for the load helpers.

        Every load path in this block routes through
        ``gui.calibration.load_run_calibrated`` with this map, so a preview,
        a diagnostic and the real fit all compute voltages from the same
        calibration. The fit worker gets the same snapshot handed to it
        explicitly (``project.py::_start_fit``), since a subprocess cannot
        reach the registry singleton.
        """
        from gui.calibration import get_registry
        return get_registry().to_dict()

    def _refresh_calibration_badges(self):
        for entry in self._file_entries:
            self._update_override_badge(entry)
            alert = entry.get("cal_alert")
            if alert is not None:
                alert.refresh()

    def _cal_physics(self, entry):
        """Doppler inputs so the calibration dialog can quote MHz, not volts.

        A residual in volts tells a physicist nothing; the same number as a
        centroid shift, set against the isotope shift being measured, tells
        them whether to care. Mass / harmonic come from the Source block; the
        laser setpoint and cooler voltage come from the file unless overridden.
        """
        from gui.calibration import read_beam_header
        cfg = self.get_source_config()
        path = (entry.get("parent_path") if entry.get("is_split")
                else entry.get("path"))
        laser = cfg.get("laser_override", 0) or 0
        cooler = cfg.get("cooler_override", 0) or 0
        if not (cfg.get("override_enabled") and laser and cooler):
            # Fall back to the run's own header.
            h_cooler, h_laser = read_beam_header(path) if path else (0.0, 0.0)
            laser = laser or h_laser
            cooler = cooler or h_cooler
        return {
            "mass_amu": cfg.get("mass", 0),
            "harmonic": cfg.get("harmonic", 2),
            "laser_cm": laser,
            "cooler_v": cooler,
        }

    def _cal_peers(self):
        """The other loaded runs, offered as borrow donors."""
        from gui.calibration import canonical_path
        peers, seen = [], set()
        for e in self._file_entries:
            if e.get("is_merged"):
                continue
            p = (e.get("parent_path") if e.get("is_split")
                 else e.get("path"))
            if not p:
                continue
            key = canonical_path(p)
            if key in seen:
                continue
            seen.add(key)
            peers.append((f"run_{e.get('run_number', '?')}", p))
        return peers

    def _open_calibration_dialog(self, entry):
        from gui.calibration_dialog import CalibrationDialog
        path = (entry.get("parent_path") if entry.get("is_split")
                else entry.get("path"))
        if not path:
            return
        CalibrationDialog(
            path,
            run_label=f"run_{entry.get('run_number', '?')}",
            cal_order=self._cal_order.value(),
            physics=self._cal_physics(entry),
            peers=self._cal_peers(),
            parent=self).exec()
        self._update_override_badge(entry)
        self.block_changed.emit()

    def _open_calibration_overview(self):
        from gui.calibration_dialog import CalibrationOverviewDialog
        peers = self._cal_peers()
        if not peers:
            QMessageBox.information(self, "Calibration",
                                    "No runs are loaded.")
            return
        phys = (self._cal_physics(self._file_entries[0])
                if self._file_entries else None)
        CalibrationOverviewDialog(
            peers, cal_order=self._cal_order.value(), physics=phys,
            parent=self).exec()
        self._refresh_calibration_badges()
        self.block_changed.emit()

    # ── Per-file binning overrides (right-click menu) ──

    def _update_override_badge(self, entry):
        """Show/hide the badge based on binning-override and calibration state.

        One badge covers both, with the text saying which: a run whose voltage
        axis was rebuilt from a different calibration is at least as worth
        flagging in the file list as one with a custom bin width.
        """
        badge = entry.get("badge_label")
        if badge is None:
            return
        ovr = entry.get("binning_override") or {}

        from gui.calibration import describe_spec, get_registry
        cal_path = (entry.get("parent_path") if entry.get("is_split")
                    else entry.get("path"))
        cal_spec = (get_registry().get(cal_path)
                    if cal_path and not entry.get("is_merged") else None)

        if not ovr and not cal_spec:
            badge.hide()
            badge.setToolTip("")
            return

        lines = []
        if cal_spec:
            lines.append(f"Voltage calibration: {describe_spec(cal_spec)}")
        if ovr:
            lines.append("Binning override:")
            for k in sorted(ovr.keys()):
                lines.append(f"  {k} = {ovr[k]}")

        if cal_spec and ovr:
            badge.setText("[cal+override]")
        elif cal_spec:
            badge.setText("[cal]")
        else:
            badge.setText("[override]")
        badge.setToolTip("\n".join(lines))
        badge.show()

    def _show_file_row_menu(self, entry, container, pos):
        """Right-click menu on a file row."""
        menu = QMenu(container)
        menu.setToolTipsVisible(True)
        is_merged = bool(entry.get("is_merged"))
        ovr = entry.get("binning_override") or {}
        edit_act = menu.addAction("Edit Binning Override...")
        freeze_act = menu.addAction("Freeze Current Source Binning Here")
        freeze_act.setToolTip(
            "Snapshots the current Source-block binning into an explicit\n"
            "per-file override. Later changes to the Source block will\n"
            "not propagate to this run.")
        copy_chk_act = menu.addAction("Copy This Override to Checked Runs")
        clear_act = menu.addAction("Clear Override")
        clear_act.setToolTip("Remove the override; this run inherits "
                             "from the Source block again.")
        menu.addSeparator()
        show_act = menu.addAction("Show Effective Binning")
        diag_act = menu.addAction("Show Binning Diagnostics")
        diag_act.setToolTip(
            "Open the diagnostics view (spectrum + bin edges) for this "
            "run only.")

        # Per-file scan filter. Merged spectra carry no per-event data,
        # so the action is disabled for them with a clarifying tooltip.
        menu.addSeparator()
        from gui.scan_filter import get_registry as _get_sf_registry
        sf_path = (entry.get("parent_path") if entry.get("is_split")
                   else entry.get("path"))
        n_excluded = (len(_get_sf_registry().get(sf_path))
                      if sf_path and not is_merged else 0)
        scan_label = (f"Filter scans...  ({n_excluded} excluded)"
                      if n_excluded else "Filter scans...")
        scan_act = menu.addAction(scan_label)
        scan_act.setToolTip(
            "Open the per-scan filter for this run. Excluded scans are "
            "dropped before binning and fitting.")

        # Per-run voltage calibration. Keyed on the parent ASDF for a split,
        # like the scan filter: the calibration belongs to the file the
        # events came from.
        from gui.calibration import get_registry as _get_cal_registry
        cal_spec = (_get_cal_registry().get(sf_path)
                    if sf_path and not is_merged else None)
        cal_act = menu.addAction(
            "Calibration...  (overridden)" if cal_spec else "Calibration...")
        cal_act.setToolTip(
            "Inspect this run's DAC->HV voltage calibration: the fit, the\n"
            "residuals, and what it costs in MHz. Exclude bad points, borrow\n"
            "another run's calibration, or enter coefficients by hand.")
        cal_all_act = menu.addAction("Calibration overview (all runs)...")
        cal_all_act.setToolTip(
            "Triage every loaded run's calibration at once, worst first.")

        if is_merged:
            for a in (edit_act, freeze_act, copy_chk_act, clear_act,
                      diag_act, scan_act, cal_act):
                a.setEnabled(False)
            cal_act.setToolTip(
                "Merged spectra are pre-binned; their voltage axis was fixed "
                "at merge time and cannot be re-calibrated.")
            edit_act.setToolTip("Merged spectra are pre-binned; "
                                "override does not apply.")
            diag_act.setToolTip("Merged spectra are pre-binned; "
                                "per-bin diagnostics are not available.")
            scan_act.setToolTip(
                "Merged spectra are pre-binned; per-event scans were "
                "collapsed at merge time and cannot be filtered here.")
        if not ovr:
            copy_chk_act.setEnabled(False)
            clear_act.setEnabled(False)
        chosen = menu.exec(container.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is edit_act:
            self._edit_file_binning_override(entry)
        elif chosen is freeze_act:
            self._copy_source_binning_to_entry(entry)
        elif chosen is copy_chk_act:
            self._copy_override_to_checked(entry)
        elif chosen is clear_act:
            self._clear_file_binning_override(entry)
        elif chosen is show_act:
            self._show_effective_binning(entry)
        elif chosen is diag_act:
            self._show_binning_diagnostics(entry)
        elif chosen is scan_act:
            from gui.scan_filter_dialog import ScanFilterDialog
            pmt_gate = self.get_pmt_gate() or [3, 4]
            ScanFilterDialog(
                sf_path, pmt_gate=pmt_gate, parent=self).exec()
        elif chosen is cal_act:
            self._open_calibration_dialog(entry)
        elif chosen is cal_all_act:
            self._open_calibration_overview()

    def _edit_file_binning_override(self, entry):
        # Splits show their source_id directly (e.g. "7509_A");
        # plain runs get the conventional "run_" prefix.
        rn = entry.get("run_number", "?")
        run_label = (str(rn) if entry.get("is_split")
                     else f"run_{rn}")
        dlg = FileBinningOverrideDialog(
            self.get_source_config(),
            entry.get("binning_override") or {},
            run_label=run_label,
            parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        entry["binning_override"] = dlg.get_override()
        self._update_override_badge(entry)
        self.block_changed.emit()

    def _copy_source_binning_to_entry(self, entry):
        """Snapshot every binning key from the Source block into this row."""
        src = self.get_source_config()
        entry["binning_override"] = {k: src[k] for k in BINNING_OVERRIDE_KEYS
                                     if k in src}
        self._update_override_badge(entry)
        self.block_changed.emit()

    def _copy_override_to_checked(self, entry):
        ovr = entry.get("binning_override") or {}
        if not ovr:
            return
        n = 0
        for e in self.get_checked_files():
            if e is entry or e.get("is_merged"):
                continue
            e["binning_override"] = dict(ovr)
            self._update_override_badge(e)
            n += 1
        self.block_changed.emit()
        QMessageBox.information(
            self, "Binning Override",
            f"Copied override to {n} other checked run(s).")

    def _clear_file_binning_override(self, entry):
        if not (entry.get("binning_override") or {}):
            return
        entry["binning_override"] = {}
        self._update_override_badge(entry)
        self.block_changed.emit()

    def _show_effective_binning(self, entry):
        from PySide6.QtWidgets import QDialogButtonBox
        eff = effective_binning_config(self.get_source_config(),
                                       entry.get("binning_override") or {})
        ovr_keys = sorted((entry.get("binning_override") or {}).keys())
        rows = ["<b>Run:</b> "
                f"{entry.get('run_number', '?')}"]
        if entry.get("is_merged"):
            rows.append("<i>(merged spectrum — override is ignored)</i>")
        rows.append("<b>Override keys:</b> " + (
            ", ".join(ovr_keys) if ovr_keys else "<i>none (inherited)</i>"))
        rows.append("<b>Effective binning config:</b>")
        rows.append("<pre style='margin:4px 0 0 0'>")
        for k in sorted(BINNING_OVERRIDE_KEYS):
            marker = " *" if k in ovr_keys else ""
            rows.append(f"  {k:<16} = {eff.get(k)}{marker}")
        rows.append("</pre>")
        if ovr_keys:
            rows.append("<i>* = overridden</i>")

        dlg = QDialog(self)
        dlg.setWindowTitle("Effective Binning")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(4)
        label = QLabel("<br>".join(rows))
        label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(label)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.adjustSize()
        dlg.exec()

    def get_checked_files(self):
        return [e for e in self._file_entries if e["checkbox"].isChecked()]

    # ── Run Merging ──

    def _merge_selected(self):
        """Open merge dialog for the checked files."""
        if _get_clstools() is None:
            QMessageBox.warning(self, "Merge", "clstools is required.")
            return
        checked = [e for e in self.get_checked_files()
                   if not e.get("is_merged")]
        if len(checked) < 2:
            QMessageBox.information(
                self, "Merge",
                "Select at least 2 loaded, non-merged files to merge.")
            return
        from gui.analysis.merge import MergeDialog
        corr_map = self._fetch_centroid_corrections_for_paths(
            [e["path"] for e in checked])
        src_cfg = self.get_source_config()
        # Pre-fill the dialog's merge-level Doppler panel with what
        # the fit will actually use for each source: the override
        # when its tick is on, else the mean of the per-file ASDF
        # cooler/laser values across the checked sources.
        #
        # ``get_source_config()`` always emits ``cooler_override``
        # as the spinbox value (no ``_ovr_enable`` gate), so reading
        # it directly would silently inject the override field's
        # default (e.g. 29977 V) into the dialog even when the
        # override checkbox is off. Gate explicitly here.
        ovr_on = self._ovr_enable.isChecked()
        ovr_cooler = float(self._cooler_spin.value()) if ovr_on else 0.0
        ovr_laser = float(self._laser_spin.value()) if ovr_on else 0.0
        if ovr_cooler > 0 and ovr_laser > 0:
            mean_cooler = ovr_cooler
            mean_laser = ovr_laser
        else:
            # Read CoolerVoltage / LaserSetpoint from each ASDF header
            # (cheap; no event-array load). ``CoolerVoltage`` is
            # stored as raw_value / 10000, so multiply by 10000 to
            # recover volts. ``LaserSetpoint`` is the cm⁻¹ value
            # directly.
            import asdf as _asdf
            coolers, lasers = [], []
            for e in checked:
                try:
                    with _asdf.open(e["path"]) as af:
                        cv = float(af.tree.get("CoolerVoltage", 0)) * 10000
                        ls = float(af.tree.get("LaserSetpoint", 0))
                except Exception:
                    continue
                if cv > 0:
                    coolers.append(cv)
                if ls > 0:
                    lasers.append(ls)
            mean_cooler = (float(np.mean(coolers)) if coolers
                           else (ovr_cooler or 0.0))
            mean_laser = (float(np.mean(lasers)) if lasers
                          else (ovr_laser or 0.0))
        default_merge_metadata = {
            "cooler_v": mean_cooler or None,
            "laser_sp": mean_laser or None,
            # Mass: the SourceBlock's chosen isotope mass is what the
            # rest of the fit pipeline uses for Doppler; use it
            # rather than reading per-file ASDF ``MassAMU`` (which
            # may be stale acquisition config).
            "mass_amu": (float(src_cfg.get("mass") or 0) or None),
            "harmonic": int(src_cfg.get("harmonic") or 2),
        }
        dlg = MergeDialog(checked, src_cfg, parent=self,
                           centroid_corrections_map=corr_map,
                           default_merge_metadata=default_merge_metadata)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            result = dlg.get_result()
            if result:
                self._add_merged_entry(result)

    def _fetch_centroid_corrections_for_paths(self, paths):
        """Walk up to the AnalysisProject and ask its parent for the
        IS-tab Reference Correction panel; pull per-file corrections
        for the given paths. Returns ``{}`` when the panel isn't
        reachable, the global apply toggle is off, or no corrector is
        fitted -- so the merge dialog can disable its checkbox
        cleanly. Logs the failure mode to stdout so a user wondering
        why the "Align centroids" checkbox is disabled has something
        to grep for.
        """
        # The merge dialog lives inside a SourceBlock, which is a
        # child of an AnalysisProject. The project has the panel
        # locator we already use at fit time.
        project = self._find_analysis_project()
        if project is None:
            print("[CentroidCorrection] merge: no AnalysisProject "
                  "found in parent chain; merge corrections "
                  "unavailable.", flush=True)
            return {}
        try:
            return project._build_corrections_map(paths)
        except Exception as exc:  # noqa: BLE001
            # Don't crash the merge dialog over a panel-side bug;
            # surface the cause so the user can debug it instead of
            # silently seeing the checkbox disabled.
            import traceback
            print(f"[CentroidCorrection] merge: corrections lookup "
                  f"raised {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
            return {}

    def _find_analysis_project(self):
        """Walk up parentWidget() to the enclosing AnalysisProject."""
        from gui.analysis.project import AnalysisProject
        p = self.parentWidget()
        while p is not None:
            if isinstance(p, AnalysisProject):
                return p
            p = p.parentWidget()
        return None

    def _add_merged_entry(self, merged_data):
        """Add a merged spectrum as a file entry in the list."""
        name = merged_data.get("merged_name", "merged")

        row = QHBoxLayout()
        cb = QCheckBox()
        cb.setChecked(True)
        cb.toggled.connect(lambda: self.block_changed.emit())
        cb.toggled.connect(self._refresh_master_check_state)
        row.addWidget(cb)

        # Domain-mismatch warning badge. Visibility is driven by
        # _refresh_merged_warning_badges (called here on add, and from
        # the Source-block bin_mode change signal). A voltage-merged
        # spectrum has lost the per-event Doppler info, so a Frequency
        # fit would silently fit voltage-axis data with a frequency
        # model -- this badge surfaces that mismatch the moment the
        # user picks Frequency.
        warn_lbl = QLabel("\u26a0")
        warn_lbl.setStyleSheet("color:#c97800; font-weight: bold;")
        warn_lbl.hide()
        row.addWidget(warn_lbl)

        lbl = QLabel(f"\u2726 {name}")
        lbl.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        lbl.setToolTip(
            f"Merged from: {', '.join(str(r) for r in merged_data.get('source_runs', []))}")
        row.addWidget(lbl, 1)

        # Right-click context menu button
        menu_btn = QToolButton()
        menu_btn.setText("\u2630")
        menu_btn.setFixedSize(22, 22)
        menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(menu_btn)
        menu.addAction("Edit...",
                        lambda md=merged_data: self._edit_merged(md))
        menu.addAction("View...",
                        lambda md=merged_data: self._view_merged(md))
        menu.addAction("Export ASDF...",
                        lambda md=merged_data: self._export_merged(md))
        menu.addSeparator()
        menu.addAction("Remove",
                        lambda md=merged_data: self._remove_merged(md))
        menu_btn.setMenu(menu)
        row.addWidget(menu_btn)

        container = QWidget()
        container.setLayout(row)
        self._file_list_layout.addWidget(container)

        entry = {
            "path": f"merged://{name}",
            "run_number": name,
            "checkbox": cb,
            "widget": container,
            "warn_label": warn_lbl,
            "is_merged": True,
            "merged_data": merged_data,
        }
        merged_data["_entry"] = entry  # back-reference for removal
        self._file_entries.append(entry)
        self._refresh_master_check_state()
        self._refresh_merged_warning_badges()
        self.block_changed.emit()

    def _refresh_merged_warning_badges(self):
        """Set the per-entry badge on each merged entry based on the
        Source block's current bin_mode.

        Three visual states:

        * **None** \u2014 merged x_unit matches bin_mode. Badge hidden.
        * **\u2139 (info)** \u2014 voltage merge + Frequency bin_mode. Phase 4
          will project the voltage axis onto MHz at fit time using the
          entry's ``merge_metadata``. Surface the cooler/laser/mass/
          harmonic values that will be used so the user can sanity-
          check before fitting (this was the Phase-1 \u26a0 trap until
          Phase 4 made the path correct).
        * **\u26a0 (warning)** \u2014 frequency merge + Raw-Voltage bin_mode.
          No projection in this direction; the entry can't be used.
        """
        bin_mode = self._bin_mode.currentText()
        for entry in self._file_entries:
            if not entry.get("is_merged"):
                continue
            warn = entry.get("warn_label")
            if warn is None:
                continue
            md = entry.get("merged_data") or {}
            x_unit = md.get("x_unit", "MHz")
            if x_unit == "V" and bin_mode == "Frequency":
                # Info: V\u2192F projection will run at fit time.
                mm = md.get("merge_metadata") or {}
                cooler = mm.get("cooler_v")
                laser = mm.get("laser_sp")
                mass = mm.get("mass_amu")
                harmonic = mm.get("harmonic")

                def _fmt(val, digits=3):
                    return f"{val:.{digits}f}" if val else "?"
                warn.setText("\u2139")
                warn.setStyleSheet(
                    "color:#4fc3f7; font-weight: bold;")
                warn.setToolTip(
                    "Voltage-merged spectrum: will be Doppler-projected "
                    "onto a rest-frame MHz axis at fit time using the "
                    "merge-level metadata.\n"
                    f"  cooler = {_fmt(cooler)} V\n"
                    f"  laser  = {_fmt(laser, 6)} cm\u207b\u00b9\n"
                    f"  mass   = {_fmt(mass, 4)} amu\n"
                    f"  harmonic = {harmonic if harmonic else '?'}")
                warn.show()
            elif x_unit == "MHz" and bin_mode == "Raw Voltage":
                # Warning: frequency merge can't be displayed/fit in V.
                warn.setText("\u26a0")
                warn.setStyleSheet(
                    "color:#c97800; font-weight: bold;")
                warn.setToolTip(
                    "Frequency-merged spectrum: stored on a MHz axis "
                    "and cannot be displayed in voltage. Switch "
                    "Source 'Bin mode' to 'Frequency' to use this "
                    "entry.")
                warn.show()
            else:
                warn.hide()
                warn.setToolTip("")

    def _edit_merged(self, merged_data):
        """Re-open merge dialog to adjust settings and re-merge."""
        entries = []
        for path in merged_data.get("source_files", []):
            for e in self._file_entries:
                if e.get("path") == path and not e.get("is_merged"):
                    entries.append(e)
                    break
        if len(entries) < 2:
            # Fall back: re-build entry list from paths
            entries = [{"path": p, "run_number": "?"} for p in
                       merged_data.get("source_files", [])]

        from gui.analysis.merge import MergeDialog
        corr_map = self._fetch_centroid_corrections_for_paths(
            [e["path"] for e in entries if e.get("path")])
        dlg = MergeDialog(entries, self.get_source_config(),
                          parent=self, existing_result=merged_data,
                          centroid_corrections_map=corr_map)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_result = dlg.get_result()
            if new_result:
                self._remove_merged(merged_data)
                self._add_merged_entry(new_result)

    def _view_merged(self, merged_data):
        """Open multi-tab viewer for the merged data. Pass the source
        config so the spectrum tab projects voltage→frequency when
        the block is in Frequency mode, matching the fit pipeline."""
        from gui.analysis.merge import MergeViewDialog
        dlg = MergeViewDialog(merged_data, parent=self,
                               source_config=self.get_source_config())
        dlg.show()

    def _export_merged(self, merged_data):
        """Export merged data as ASDF file."""
        from gui.shared_widgets import get_last_dir, remember_last_dir
        name = merged_data.get("merged_name", "merged")
        last_save = get_last_dir("data", "save")
        default_path = (os.path.join(last_save, f"{name}.asdf")
                        if last_save else f"{name}.asdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Merged ASDF",
            default_path,
            "ASDF files (*.asdf)")
        if not path:
            return
        remember_last_dir("data", "save", path)
        try:
            from gui.analysis.merge import export_merged_asdf
            export_merged_asdf(merged_data, path)
            QMessageBox.information(
                self, "Export", f"Merged ASDF saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _remove_merged(self, merged_data):
        """Remove a merged entry from the file list."""
        entry = merged_data.get("_entry")
        if entry and entry in self._file_entries:
            self._file_entries.remove(entry)
            w = entry.get("widget")
            if w:
                self._file_list_layout.removeWidget(w)
                w.deleteLater()
        self._refresh_master_check_state()
        self.block_changed.emit()

    def get_pmt_gate(self):
        gate = []
        for i, cb in enumerate(self._pmt_checks):
            if cb.isChecked():
                if i < 4:
                    gate.append(i + 1)
                else:
                    gate.append(0)  # DC channel
        return gate

    def get_tof_gate(self):
        if not self._tof_enable.isChecked():
            return None
        return [self._tof_lo.value(), self._tof_hi.value()]

    def get_ref_frequency(self):
        """Reference frequency [Hz] = (E_upper - E_lower) * c * 100.

        E_upper and E_lower are atomic energy levels in cm⁻¹.
        Their difference gives the full rest-frame transition
        wavenumber. No harmonic multiplication needed here because
        clstools handles the harmonic internally in the Doppler
        calculation (harmonic * Laser_set).
        """
        c = 299792458.0
        e_lo = self._e_lower_spin.value()
        e_up = self._e_upper_spin.value()
        return (e_up - e_lo) * 1e2 * c

    def get_vgate(self):
        if not self._vgate_enable.isChecked():
            return None
        return [self._vgate_lo.value(), self._vgate_hi.value()]

    def get_fgate(self):
        """Return the F gate as [lo_Hz, hi_Hz] for clstools.

        The UI shows MHz for convenience; we convert to Hz here because
        clstools' DataFrame.F column is in Hz.
        """
        if not self._fgate_enable.isChecked():
            return None
        return [self._fgate_lo.value() * 1e6,
                self._fgate_hi.value() * 1e6]

    def _on_za_changed(self):
        """Look up mass from IUPAC table when Z or A changes."""
        z = self._z_number.value()
        a = self._a_number.value()
        # Update element label
        sym = Z_TO_ELEMENT.get(z, "?")
        self._element_label.setText(f"{a}{sym}")
        # Look up mass (skipped when the user has locked it via Override)
        if not self._mass_override.isChecked() and self._mass_table:
            try:
                mass = get_mass(self._mass_table, z, a)
            except KeyError:
                # No entry -- use A as approximate mass
                mass = float(a)
            self._mass_spin.blockSignals(True)
            self._mass_spin.setValue(mass)
            self._mass_spin.blockSignals(False)
        self.block_changed.emit()

    def _on_mass_override_toggled(self, checked):
        """Lock/unlock the mass spinbox; reset to table value when unlocked."""
        self._mass_spin.setReadOnly(not checked)
        self._mass_spin.setButtonSymbols(
            QDoubleSpinBox.ButtonSymbols.UpDownArrows if checked
            else QDoubleSpinBox.ButtonSymbols.NoButtons)
        if not checked:
            self._on_za_changed()
        else:
            self.block_changed.emit()

    def _update_bin_def_controls(self):
        """Enable/disable Bin count and Bin width based on definition + mode.

        Raw Voltage forces Auto: clstools groups unique DV values, so neither
        a target count nor a target width is meaningful in voltage mode.
        """
        is_voltage = self._bin_mode.currentText() == "Raw Voltage"
        if is_voltage:
            self._bin_def_combo.setEnabled(False)
            self._bin_def_combo.setToolTip(
                "Raw Voltage groups unique DV values; bin count/width "
                "not applicable.\nFrequency mode required for explicit "
                "binning.")
            self._bin_count_spin.setEnabled(False)
            self._bin_width_spin.setEnabled(False)
            # Grouping N adjacent voltage steps stays meaningful.
            self._step_mult_spin.setEnabled(True)
            return
        self._bin_def_combo.setEnabled(True)
        self._bin_def_combo.setToolTip("How bin count is decided in "
                                       "Frequency mode")
        defn = self._bin_def_combo.currentText()
        self._bin_count_spin.setEnabled(defn == "Fixed bin count")
        self._bin_width_spin.setEnabled(defn == "Fixed bin width")
        self._step_mult_spin.setEnabled(defn == "Per scan step")

    def get_source_config(self):
        return {
            "Z": self._z_number.value(),
            "A": self._a_number.value(),
            "mass": self._mass_spin.value(),
            "harmonic": self._harmonic_spin.value(),
            "ref_freq": self.get_ref_frequency(),
            "tof_gate": self.get_tof_gate(),
            "pmt_gate": self.get_pmt_gate(),
            "v_gate": self.get_vgate(),
            "f_gate": self.get_fgate(),
            "noise_filter": self._noise_filter.value(),
            "x_column": self._x_col_combo.currentText(),
            "yerr_mode": self._yerr_combo.currentText(),
            "bin_mode": self._bin_mode.currentText(),
            "xerr_mode": self._xerr_combo.currentText(),
            "bin_definition": self._bin_def_combo.currentText(),
            "bin_count": self._bin_count_spin.value(),
            "bin_width_mhz": self._bin_width_spin.value(),
            "step_multiple": self._step_mult_spin.value(),
            "override_enabled": self._ovr_enable.isChecked(),
            "cooler_override": self._cooler_spin.value(),
            "laser_override": self._laser_spin.value(),
            "e_lower": self._e_lower_spin.value(),
            "e_upper": self._e_upper_spin.value(),
            "cal_order": self._cal_order.value(),
            "cooler_correction": self._cooler_corr.currentText(),
            # ref_shift is published in Hz because clstools.Shift_Ref takes
            # Hz; the UI shows MHz to match the spectrum axis.
            "ref_shift": self._ref_shift.value() * 1e6,
        }

    # ── Import from Pre-Analysis ──

    def _import_all_from_preanalysis(self):
        """Single-button import: pick a Pre-Analysis project once, then
        pull files, physics, gates, and cooler/laser overrides.
        """
        pa = _choose_pa_project(self)
        if pa is None:
            return

        # Import only the ticked Pre-Analysis entries. If the project has
        # entries but none are ticked, abort the whole import (don't copy
        # files, physics, gates, or overrides).
        all_entries = list(getattr(pa, '_file_entries', []) or [])
        if hasattr(pa, 'checked_file_entries'):
            ticked = list(pa.checked_file_entries())
        else:
            ticked = [
                fe for fe in all_entries
                if getattr(fe, 'check', None) is None or fe.check.isChecked()
            ]
        if all_entries and not ticked:
            QMessageBox.warning(
                self, "Import from Pre-Analysis",
                "No files are ticked in the selected Pre-Analysis project. "
                "Nothing was imported.")
            return

        self._import_files_from_preanalysis(pa)
        self._update_nav_label()

        # Physics parameters
        if hasattr(pa, '_e_lower'):
            self._e_lower_spin.setValue(pa._e_lower.value())
            self._e_upper_spin.setValue(pa._e_upper.value())
        elif hasattr(pa, '_offset'):
            # Legacy: pre-analysis stored a single "Transition" field
            # = (E_up - E_low) / harmonic. Convert back.
            harmonic = pa._harmonic.value() if hasattr(pa, '_harmonic') else 4
            transition_full = pa._offset.value() * harmonic
            self._e_lower_spin.setValue(0.0)
            self._e_upper_spin.setValue(transition_full)
        if hasattr(pa, '_harmonic'):
            self._harmonic_spin.setValue(pa._harmonic.value())
        if hasattr(pa, '_z_spin'):
            self._z_number.setValue(pa._z_spin.value())
        if hasattr(pa, '_a_spin'):
            self._a_number.setValue(pa._a_spin.value())
        if hasattr(pa, '_mass_override'):
            self._mass_override.setChecked(pa._mass_override.isChecked())
            if pa._mass_override.isChecked() and hasattr(pa, '_mass_spin'):
                self._mass_spin.blockSignals(True)
                self._mass_spin.setValue(pa._mass_spin.value())
                self._mass_spin.blockSignals(False)

        # Gates
        if hasattr(pa, '_tof_enable'):
            self._tof_enable.setChecked(pa._tof_enable.isChecked())
        if hasattr(pa, '_tof_lo'):
            self._tof_lo.setValue(pa._tof_lo.value())
        if hasattr(pa, '_tof_hi'):
            self._tof_hi.setValue(pa._tof_hi.value())
        if hasattr(pa, '_channels'):
            for i, cb in enumerate(pa._channels):
                if i < len(self._pmt_checks):
                    self._pmt_checks[i].setChecked(cb.isChecked())

        # Cooler / Laser overrides
        if hasattr(pa, '_cooler_override'):
            self._cooler_spin.setValue(pa._cooler_override.value())
        if hasattr(pa, '_laser_override'):
            self._laser_spin.setValue(pa._laser_override.value())

        self.block_changed.emit()

    def _import_files_from_preanalysis(self, pa):
        """Copy Pre-Analysis file entries into this Source block.

        Regular FileEntry / SplitFileEntry go through ``_add_file_entry``
        with the entry's real path. ``MergedFileEntry`` cannot — its
        ``filepath`` is the synthetic ``"[merged] <name>"`` placeholder
        that ``Load_Run`` rejects with ENOENT (the source of the
        "Preview Error: No such file or directory" report). Translate
        the per-entry x/y arrays into the same ``merged_data`` shape that
        the in-tab MergeDialog produces, then add it via
        ``_add_merged_entry`` so View / Edit / Export menus and the fit
        pipeline (which keys off ``is_merged``) all work.

        Frequency-domain Pre-Analysis merges store *absolute* lab-frame
        MHz; Analysis convention is MHz *relative to the rest-frame
        transition* (this is what ``compute_merged_spectrum`` produces
        via ``Compute_WL(ref=ref_freq)``). Shift on import so a merged
        entry round-tripped from Pre-Analysis fits the same model as one
        produced inside the Source block.
        """
        from gui.preanalysis_tab import MergedFileEntry
        from cls_estimations.constants import C_LIGHT
        try:
            e_lower = float(pa._e_lower.value())
            e_upper = float(pa._e_upper.value())
            harmonic = int(pa._harmonic.value())
        except (AttributeError, TypeError):
            e_lower = 0.0
            e_upper = 0.0
            harmonic = 2
        # Transition offset (MHz) that places the spectrum on the
        # relative Frequency display axis. Derivation: the display
        # frequency is x_absolute - (E_upper - E_lower)/harmonic
        # * harmonic * c * 100 / 1e6, in which the harmonic cancels,
        # leaving (E_upper - E_lower) * c * 100 / 1e6. Energies are in
        # cm⁻¹; c*100 converts cm⁻¹ to Hz, /1e6 to MHz.
        transition_mhz = (e_upper - e_lower) * C_LIGHT * 100.0 / 1e6

        for fe in getattr(pa, '_file_entries', []):
            # Import only ticked entries. The getattr guard keeps test
            # doubles without a real checkbox (whose .check is truthy or
            # absent) importing as before.
            chk = getattr(fe, 'check', None)
            if chk is not None and not chk.isChecked():
                continue
            if isinstance(fe, MergedFileEntry):
                self._add_merged_from_preanalysis(fe, transition_mhz)
            else:
                already = any(
                    e.get("path") == fe.filepath
                    for e in self._file_entries
                )
                if not already:
                    self._add_file_entry(fe.filepath)

    def _add_merged_from_preanalysis(self, mfe, transition_mhz):
        """Translate a Pre-Analysis ``MergedFileEntry`` to a Source-side
        merged entry. ``transition_mhz`` is subtracted from x in
        frequency-domain merges to convert absolute → ref-shifted MHz.

        Phase 2: the PA-side ``source_info`` is now a list of dicts
        (see ``_normalize_source_info``) carrying per-source physics
        parameters, and the entry carries merge-level Doppler
        metadata. Both flow into the Analysis-side ``merged_data`` so
        Phase 4's voltage-merge frequency-fit path can read them
        without re-importing from PA.
        """
        import numpy as np
        name = str(getattr(mfe, "run_number", "merged"))
        # Idempotent: skip if a merged entry with the same name is
        # already present.
        for e in self._file_entries:
            md = e.get("merged_data") or {}
            if e.get("is_merged") and md.get("merged_name") == name:
                return

        x = np.asarray(mfe.merged_x, dtype=float)
        y = np.asarray(mfe.merged_y, dtype=float)
        # Pre-Analysis merge doesn't store yerr; synthesize the Poisson
        # default sqrt(y), substituting 1.0 for empty (zero-count) bins.
        yerr = np.where(y > 0, np.sqrt(y), 1.0)

        domain = getattr(mfe, "merge_domain", "voltage")
        if domain == "frequency":
            x_unit = "MHz"
            x = x - transition_mhz
        else:
            x_unit = "V"

        # source_info is post-normalization (list of dicts). Accessing
        # ["run_number"] / ["filepath"] is safe regardless of whether
        # the PA YAML was Phase-1 (2-list) or Phase-2 (dict) shape.
        source_info = list(getattr(mfe, "source_info", []) or [])
        source_runs = [s.get("run_number") for s in source_info]
        source_files = [s.get("filepath") for s in source_info]

        # Prefer the rich in-memory per_run that PA captured at merge
        # time (TOF arrays, timestamps, dwell times, n_events) when
        # available; fall back to reconstructing skeletal per_run
        # from source_info for entries that were round-tripped via
        # YAML (which doesn't persist per_run).
        mfe_per_run = getattr(mfe, "per_run", None) or None
        if mfe_per_run:
            per_run = [dict(rd) for rd in mfe_per_run]
        else:
            per_run = [
                {
                    "run_num":   s.get("run_number"),
                    "path":      s.get("filepath"),
                    "cooler_v":  s.get("cooler_v"),
                    "laser_set": s.get("laser_sp"),
                    "mass_amu":  s.get("mass_amu"),
                    "harmonic":  s.get("harmonic"),
                }
                for s in source_info
            ]

        # merge_metadata is what Phase 4 reads to Doppler-shift a
        # voltage-merged spectrum at fit time. None means "fall back
        # to mean-of-sources" -- the fit path handles that.
        merge_metadata = {
            "cooler_v": getattr(mfe, "merge_cooler_v", None),
            "laser_sp": getattr(mfe, "merge_laser_sp", None),
            "mass_amu": getattr(mfe, "merge_mass_amu", None),
            "harmonic": getattr(mfe, "merge_harmonic", None),
        }

        merged_data = {
            "merged_name": name,
            "x": x,
            "y": y,
            "yerr": yerr,
            "x_unit": x_unit,
            "source_runs": source_runs,
            "source_files": source_files,
            "per_run": per_run,
            "bin_step_mhz": 0,
            "merge_metadata": merge_metadata,
            "metadata": {"imported_from": "preanalysis",
                         "merge_domain": domain},
        }
        self._add_merged_entry(merged_data)

    # ── Navigation helpers ──

    def _update_nav_label(self):
        """Refresh the index label to reflect current state."""
        checked = self.get_checked_files()
        if not checked:
            self._preview_idx_label.setText("0 / 0")
            return
        idx = max(0, min(self._preview_index, len(checked) - 1))
        self._preview_index = idx
        self._preview_idx_label.setText(f"{idx + 1} / {len(checked)}")

    def _is_win_open(self, win):
        """Return True if a QDialog reference is still alive and visible."""
        try:
            return win is not None and win.isVisible()
        except RuntimeError:
            return False

    # ── Data Preview (navigable) ──

    def _open_preview(self):
        """Open preview at the current index."""
        self._show_preview_at_index()

    def _navigate_preview(self, delta):
        """Move index to next/previous file; refresh open windows."""
        checked = self.get_checked_files()
        if not checked:
            return
        self._preview_index = max(
            0, min(self._preview_index + delta, len(checked) - 1))
        self._update_nav_label()
        # Refresh any open windows
        if self._is_win_open(self._preview_win):
            self._show_preview_at_index()
        if self._is_win_open(self._meta_win):
            self._refresh_meta_content()

    def _show_preview_at_index(self):
        """Load and display the file at _preview_index, mirroring all
        source block settings (bin mode, gates, yerr, overrides)."""
        if _get_clstools() is None:
            QMessageBox.warning(self, "Preview",
                                "clstools is required for data preview.")
            return
        checked = self.get_checked_files()
        if not checked:
            QMessageBox.information(self, "Preview",
                                    "No files selected for preview.")
            return
        idx = max(0, min(self._preview_index, len(checked) - 1))
        self._preview_index = idx
        self._update_nav_label()

        entry = checked[idx]
        # Merged entries carry pre-computed (x, y, yerr) — there's no
        # ASDF behind a "merged://" path. Forward to the View dialog
        # so a navigated-to merged entry doesn't crash Load_Run with
        # "[Errno 2] No such file or directory".
        if entry.get("is_merged"):
            from gui.analysis.merge import MergeViewDialog
            try:
                if self._preview_win is not None:
                    self._preview_win.close()
            except RuntimeError:
                pass
            # Pass source_config so a voltage-merged spectrum with
            # Source bin_mode=Frequency gets the V→F projection
            # applied in the preview, matching what the fit will use.
            self._preview_win = MergeViewDialog(
                entry["merged_data"], parent=self,
                source_config=self.get_source_config())
            self._preview_win.show()
            return

        filepath = entry["path"]
        is_split = bool(entry.get("is_split"))
        data_path = (entry["parent_path"] if is_split else filepath)
        split_md = (entry.get("metadata_override", {}) or {}
                    if is_split else {})
        try:
            from gui.analysis.binning import intersect_v_gate
            from gui.calibration import load_run_calibrated
            data = _get_clstools().CLSDataFrame()
            load_run_calibrated(data, data_path, self._calibrations(),
                                cal_order=self._cal_order.value())
            # Override precedence:
            # SourceBlock global override > per-split metadata > ASDF.
            src_cooler = (self._cooler_spin.value()
                          if self._ovr_enable.isChecked() else 0)
            cooler_override = (src_cooler if src_cooler > 0
                               else float(split_md.get("cooler_v", 0) or 0))
            if cooler_override > 0:
                data.VCoolDiv = 0
                data.VCoolOffset = cooler_override
            src_laser = (self._laser_spin.value()
                         if self._ovr_enable.isChecked() else 0)
            laser_override = (src_laser if src_laser > 0
                              else float(split_md.get("laser_sp", 0) or 0))
            if laser_override > 0:
                data.Laser_set = laser_override
            data.Compute_Voltages(
                cooler_correction=self._cooler_corr.currentText())
            nf = self._noise_filter.value()
            if nf > 0:
                data.apply_filter(filter_window=nf)

            # Build effective config FIRST so per-file bin_mode
            # override is honored when deciding the Frequency vs
            # Raw Voltage branch below. For splits, also compose
            # the source's V-gate with the split's range.
            cfg = effective_binning_config(
                self.get_source_config(),
                entry.get("binning_override") or {})
            if is_split:
                src_vg = cfg.get("v_gate")
                composed = intersect_v_gate(
                    tuple(src_vg) if src_vg else None,
                    (float(entry["split_lo"]),
                     float(entry["split_hi"])))
                if composed is None:
                    raise ValueError(
                        f"Split V-gate "
                        f"[{entry['split_lo']:.3f}, "
                        f"{entry['split_hi']:.3f}] does not overlap "
                        f"the Source-block V-gate {src_vg!r}; no "
                        "events would survive.")
                cfg = dict(cfg)
                cfg["v_gate"] = list(composed)
            bin_mode = cfg.get("bin_mode", "Frequency")
            if bin_mode != "Raw Voltage":
                ref_hz = self.get_ref_frequency()
                if ref_hz <= 0:
                    raise ValueError(
                        "Frequency binning requires E upper > E lower.\n"
                        "Set the atomic energy levels (cm⁻¹), or use "
                        "'Pull from Pre-Analysis' to import them.")
                # When cooler is overridden, VCoolDiv=0 breaks
                # Frequency_stepsize in Compute_WL (uses Vcool_init *
                # VCoolDiv for step size). Temporarily set Vcool_init
                # and VCoolDiv so the product gives the override value.
                if cooler_override > 0:
                    data.Vcool_init = cooler_override
                    data.VCoolDiv = 1
                data.Compute_WL(
                    Mass=self._mass_spin.value(),
                    ref=ref_hz,
                    harmonic=self._harmonic_spin.value(),
                )
                # Restore VCoolDiv=0 so it doesn't affect other code
                if cooler_override > 0:
                    data.VCoolDiv = 0
                # Compose with ref_hz and convert MHz->Hz: the spinbox is in
                # MHz and clstools.Shift_Ref takes Hz and *assigns* Reference
                # (overwrite, not add), so a bare MHz value would both wipe the
                # transition calibration and be off by 1e6, shifting the
                # previewed axis by ~the whole transition frequency (code
                # review 2026-06-02, preview-bare-ref-shift / -mhz-as-hz).
                rs_hz = self._ref_shift.value() * 1e6
                if rs_hz != 0.0:
                    data.Shift_Ref(ref=ref_hz + rs_hz)

            res = compute_binned(data, cfg)
            x, y, yerr = res["x"], res["y"], res["yerr"]
            x_label = res["x_label"]
            # Preview never uses callable yerr; fall back to sqrt(y+1)
            if res["use_callable_yerr"]:
                yerr = np.sqrt(y + 1)

            ps = get_plot_type_settings("preview")
            from matplotlib.figure import Figure
            fig = Figure(figsize=(ps["figsize_w"], ps["figsize_h"]))
            ax = fig.add_subplot(111)
            ax.errorbar(x, y, yerr=yerr, fmt=ps["data_fmt"],
                         ms=ps["data_ms"], capsize=ps["data_capsize"])
            ax.set_xlabel(x_label, fontsize=ps["label_size"])
            ax.set_ylabel("Counts", fontsize=ps["label_size"])
            run_num = checked[idx].get("run_number", "?")
            ax.set_title(f"Run {run_num}  ({idx + 1}/{len(checked)})",
                          fontsize=ps["title_size"])

            try:
                if self._preview_win is not None:
                    self._preview_win.close()
            except RuntimeError:
                pass
            self._preview_win = PopupPlotWindow(
                fig, title=f"Preview: Run {run_num}", parent=self)
            self._preview_win.show()
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def _show_run_info(self):
        """Open (or refresh) the non-modal metadata window for the current file."""
        self._refresh_meta_content(open_if_closed=True)

    def _build_meta_html(self, filepath, split_info=None):
        """Load a file and return metadata as HTML string.

        ``split_info`` (optional) carries the descriptor for a
        virtual-split entry: ``{parent_path, source_id, split_lo,
        split_hi, metadata_override}``. When provided we read
        ``parent_path`` (clstools can't parse a .vasdf), and prepend
        a small Virtual Split header to the output so the user sees
        the gate and per-side metadata too.
        """
        from gui.calibration import (
            canonical_path as _cal_canon, describe_spec, load_run_calibrated)
        load_path = (split_info["parent_path"]
                     if split_info else filepath)
        cal_map = self._calibrations()
        cal_spec = cal_map.get(_cal_canon(load_path))
        data = _get_clstools().CLSDataFrame()
        cal_res = load_run_calibrated(
            data, load_path, cal_map, cal_order=self._cal_order.value())
        data.Compute_Voltages(
            cooler_correction=self._cooler_corr.currentText())
        vcool = getattr(data, 'Vcool_init', None)
        vdiv = getattr(data, 'VCoolDiv', 1)
        vcool_v = f"{vcool * vdiv:.2f}" if vcool is not None else "?"
        # Cal is stored in clstools' raw convention; VAccDiv * c puts it back
        # in volts, where p0 is an offset in V and p1 a gain of ~1.
        cal_v = [f"{data.VAccDiv * c:.6f}"
                 for c in getattr(data, 'Cal', [])]
        # Which calibration produced those coefficients, and how well it fits.
        cal_source = describe_spec(cal_spec)
        cal_points = (f"{cal_res.n_points - len(cal_res.excluded)} used"
                      f" / {cal_res.n_points}")
        if cal_res.fit is not None:
            cal_points += f"  (sigma {cal_res.sigma_v:.4f} V)"
        ranges_str = ""
        for sr in getattr(data, 'ScanningRanges', []):
            ranges_str += f"  {sr[0]} V  to  {sr[1]} V\n"
        daq = getattr(data, 'DAQTStime', None)
        daq_str = f"{daq:.1f}" if daq is not None else "?"
        split_header = ""
        if split_info:
            md = split_info.get("metadata_override", {}) or {}
            split_header = (
                f"--- Virtual Split ---\n"
                f"Source ID:      {split_info.get('source_id', '?')}\n"
                f".vasdf:         {filepath}\n"
                f"Parent ASDF:    {load_path}\n"
                f"V-gate:         "
                f"[{split_info['split_lo']:.3f}, "
                f"{split_info['split_hi']:.3f}] V\n"
            )
            if md.get("cooler_v") is not None:
                split_header += (
                    f"Override V_cool:{md['cooler_v']:.3f} V\n")
            if md.get("laser_sp") is not None:
                split_header += (
                    f"Override Laser: {md['laser_sp']:.6f} cm⁻¹\n")
            if md.get("comment"):
                split_header += f"Comment:        {md['comment']}\n"
            split_header += "\n"
        info_text = (
            split_header
            + f"Run number:     {getattr(data, 'run_number', '?')}\n"
            f"Experiment:     {getattr(data, 'Experiment', '?')}\n"
            f"Date:           {getattr(data, 'Date', '?')}\n"
            f"File:           {load_path}\n"
            f"\n--- Voltage Settings ---\n"
            f"Cooler V div:   {vdiv}\n"
            f"Cooler V offset:{getattr(data, 'VCoolOffset', 0)}\n"
            f"LCR V div:      {getattr(data, 'VAccDiv', 1000)}\n"
            f"Init Cooler [V]:{vcool_v}\n"
            f"\n--- Calibration ---\n"
            f"Laser set [cm-1]: {getattr(data, 'Laser_set', '?')}\n"
            f"Cal source:     {cal_source}\n"
            f"Cal order:      {getattr(data, 'Cal_order', '?')}\n"
            f"Cal coeffs [V]: {cal_v}\n"
            f"Cal points:     {cal_points}\n"
            f"\n--- Scanning ---\n"
            f"Step size:      {getattr(data, 'Step_Size', '?')}\n"
            f"Dwell time:     {getattr(data, 'Dwell_Time', '?')}\n"
            f"Ranges:\n{ranges_str}"
            f"\n--- General ---\n"
            f"Events:         {getattr(data, 'Size', '?')}\n"
            f"DAQ time [s]:   {daq_str}\n"
            f"Start TS:       {getattr(data, 'TSstart', '?')}\n"
            f"Stop TS:        {getattr(data, 'TSstop', '?')}\n"
        )
        return f"<pre>{info_text}</pre>"

    def _build_merged_meta_html(self, merged_data):
        """Return an HTML metadata block for a merged spectrum entry.

        Merged entries have no ASDF file behind them, so the normal
        ``_build_meta_html`` path (which calls ``Load_Run`` on the
        synthetic ``merged://<name>`` path) would raise "Protocol not
        known: merged". This builder reads everything it needs out of
        ``merged_data`` directly.
        """
        name = merged_data.get("merged_name", "?")
        x_unit = merged_data.get("x_unit", "MHz")
        domain_label = "Frequency (rest-frame MHz)" if x_unit == "MHz" \
            else "Voltage (V)"
        x = np.asarray(merged_data.get("x", []), dtype=float)
        y = np.asarray(merged_data.get("y", []), dtype=float)
        n_bins = len(x)
        if n_bins > 1:
            x_lo, x_hi = float(x.min()), float(x.max())
            dx = float(np.median(np.diff(np.sort(x))))
        else:
            x_lo = x_hi = float(x[0]) if n_bins else 0.0
            dx = 0.0
        total = int(np.sum(y))
        bin_step = merged_data.get("bin_step_mhz", 0)

        source_runs = merged_data.get("source_runs", []) or []
        source_files = merged_data.get("source_files", []) or []
        per_run = merged_data.get("per_run", []) or []
        meta = merged_data.get("metadata", {}) or {}

        per_run_lines = ""
        if per_run:
            for rd in per_run:
                per_run_lines += (
                    f"  run {rd.get('run_num', '?')}  "
                    f"cooler={rd.get('cooler_v', '?')}  "
                    f"laser={rd.get('laser_set', '?')}\n")
        elif source_runs:
            for rn, fp in zip(
                    source_runs,
                    source_files + [""] * (len(source_runs)
                                            - len(source_files))):
                per_run_lines += f"  run {rn}  {fp}\n"

        meta_lines = ""
        for k in ("imported_from", "merge_domain"):
            if k in meta:
                meta_lines += f"{k:<16}: {meta[k]}\n"

        info_text = (
            f"--- Merged Spectrum ---\n"
            f"Name:           {name}\n"
            f"Axis:           {domain_label}\n"
            f"Bins:           {n_bins}\n"
            f"Bin step:       {bin_step}\n"
            f"x range:        [{x_lo:.3f}, {x_hi:.3f}] {x_unit}\n"
            f"Median Δx:      {dx:.4f} {x_unit}\n"
            f"Total counts:   {total}\n"
            f"\n--- Source Runs ({len(source_runs) or len(per_run)}) ---\n"
            f"{per_run_lines}"
        )
        if meta_lines:
            info_text += f"\n--- Metadata ---\n{meta_lines}"
        return f"<pre>{info_text}</pre>"

    def _open_or_update_meta_dialog(self, title, html):
        """Update the existing meta window or create a new non-modal one.

        Shared helper for merged and non-merged paths so the two
        branches don't drift in window geometry, margins, or text
        formatting.
        """
        if self._is_win_open(self._meta_win):
            self._meta_win.setWindowTitle(title)
            self._meta_label.setText(html)
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(520, 420)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(label)
        lay.addWidget(scroll, 1)
        self._meta_win = dlg
        self._meta_label = label
        dlg.show()

    def _refresh_meta_content(self, open_if_closed=False):
        """Refresh the metadata window content for the current index."""
        if _get_clstools() is None:
            QMessageBox.warning(self, "Run Info", "clstools is required.")
            return
        checked = self.get_checked_files()
        if not checked:
            QMessageBox.information(self, "Run Info", "No files selected.")
            return
        idx = max(0, min(self._preview_index, len(checked) - 1))
        entry = checked[idx]
        filepath = entry["path"]

        if not open_if_closed and not self._is_win_open(self._meta_win):
            return

        # Merged entries have no ASDF behind their synthetic
        # ``merged://<name>`` path -- route to a merged-info builder
        # that reads the in-memory dict directly.
        if entry.get("is_merged"):
            try:
                html = self._build_merged_meta_html(entry["merged_data"])
            except Exception as e:
                QMessageBox.critical(self, "Run Info Error", str(e))
                return
            self._open_or_update_meta_dialog(
                f"Run Info: {entry.get('run_number', 'merged')}", html)
            return

        split_info = None
        if entry.get("is_split"):
            split_info = {
                "parent_path": entry["parent_path"],
                "source_id": entry["source_id"],
                "split_lo": entry["split_lo"],
                "split_hi": entry["split_hi"],
                "metadata_override": entry.get("metadata_override", {}),
            }

        try:
            html = self._build_meta_html(filepath, split_info=split_info)
        except Exception as e:
            QMessageBox.critical(self, "Run Info Error", str(e))
            return

        title = (f"Run Info: {entry['source_id']}"
                 if entry.get("is_split")
                 else f"Run Info: {os.path.basename(filepath)}")
        self._open_or_update_meta_dialog(title, html)

    def _open_binning_dialog(self, focus_entry=None,
                              default_tab="summary"):
        """Open the unified Binning dialog (Summary + Diagnostics tabs).

        Re-runs the same load → Compute_Voltages → Compute_WL →
        compute_binned pipeline that the fit worker uses (with
        diagnostics enabled) so what the dialog shows matches what the
        fit will actually see. ``focus_entry`` pre-selects a row in the
        Diagnostics tab; ``default_tab`` opens the dialog on that tab.
        """
        if _get_clstools() is None:
            QMessageBox.warning(self, "Binning",
                                "clstools is required.")
            return
        checked = self.get_checked_files()
        if not checked:
            QMessageBox.information(self, "Binning",
                                    "No files selected.")
            return
        from gui.analysis.binning import (
            compute_binned, summarize_x, build_binning_warnings)

        focus_label = (f"run_{focus_entry.get('run_number', '?')}"
                       if focus_entry else None)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            src_cfg = self.get_source_config()
            infos = []   # list of (label, full_result_or_None, err_or_None)
            for entry in checked:
                label = f"run_{entry.get('run_number', '?')}"
                try:
                    if (entry.get("is_merged")
                            and "merged_data" in entry):
                        md = entry["merged_data"]
                        x_arr = np.asarray(md["x"], dtype=float)
                        s = summarize_x(x_arr)
                        x_unit = md.get("x_unit", "MHz")
                        info = {
                            "source": "merged",
                            "merged_name": md.get("merged_name"),
                            "bin_mode": ("Frequency" if x_unit == "MHz"
                                         else "Raw Voltage"),
                            "bin_definition": "Pre-merged",
                            "fallback_used": False,
                            "effective_n_bins": s["n_bins"],
                            "effective_bin_width_mhz": (
                                s["dx_median"] if x_unit == "MHz"
                                else None),
                            "effective_bin_width_v": (
                                s["dx_median"] if x_unit != "MHz"
                                else None),
                            "override_active": False,
                            "override_keys": [],
                            **s,
                        }
                        # No diagnostics for merged spectra (they're
                        # pre-binned; no authoritative bin edges).
                        full = {"x": x_arr,
                                "y": np.asarray(md["y"], dtype=float),
                                "yerr": np.asarray(md["yerr"], dtype=float),
                                "xerr": None,
                                "use_callable_yerr": False,
                                "x_label": ("Frequency [MHz]"
                                             if x_unit == "MHz"
                                             else "Voltage [V]"),
                                "info": info}
                    else:
                        data, eff_cfg = self._prepare_data_for_binning(
                            entry, src_cfg)
                        full = compute_binned(
                            data, eff_cfg,
                            include_diagnostics=True,
                            raw_event_sample_size=20000)
                        ovr = entry.get("binning_override") or {}
                        full["info"]["override_active"] = bool(ovr)
                        full["info"]["override_keys"] = sorted(ovr.keys())
                    infos.append((label, full, None))
                except Exception as exc:
                    infos.append((label, None, str(exc)))
        finally:
            QApplication.restoreOverrideCursor()

        warnings_list = build_binning_warnings(
            [(lbl, full["info"]) for lbl, full, _e in infos if full])
        BinningDialog(infos, warnings_list,
                      default_tab=default_tab,
                      default_run=focus_label,
                      parent=self).exec()

    def split_descriptor_for(self, entry):
        """The virtual-split descriptor for ``entry``, or None for a plain run.

        Same shape the fit worker and the auto-fitter are handed
        (``project.py``), so every consumer composes the split the same way.
        """
        if not entry.get("is_split"):
            return None
        return {
            "parent_path": entry["parent_path"],
            "source_id": entry["source_id"],
            "split_lo": float(entry["split_lo"]),
            "split_hi": float(entry["split_hi"]),
            "metadata_override": dict(entry.get("metadata_override") or {}),
        }

    def _prepare_data_for_binning(self, entry, src_cfg):
        """Load + Compute_Voltages + Compute_WL for one ASDF entry.

        Returns (data, eff_cfg) ready to feed into compute_binned.

        This used to be a hand-maintained copy of the fit worker's
        preprocessing, and it had drifted: it loaded ``entry["path"]``
        unconditionally, so a virtual split tried to hand clstools its
        ``.vasdf`` sidecar and died; it never applied the split's
        ``metadata_override``; and it never intersected the split's V-gate, so
        had the load succeeded the "diagnostics" would have shown the parent
        run's *whole* spectrum while claiming to describe one side of it.

        It now goes through ``prepare_run_data`` -- the same function the fit
        actually uses -- so "diagnostics and the fit see identical inputs" is
        true by construction rather than by vigilance.
        """
        from gui.analysis.pipeline import prepare_run_data

        override = entry.get("binning_override") or {}
        # Check the *effective* mode: a per-file override can flip this run to
        # Frequency even when the Source block is on Raw Voltage.
        eff = effective_binning_config(src_cfg, override)
        if (eff.get("bin_mode", "Frequency") != "Raw Voltage"
                and self.get_ref_frequency() <= 0):
            raise ValueError("Frequency binning requires E upper > E lower")

        return prepare_run_data(
            entry["path"], src_cfg,
            binning_override=override,
            split_descriptor=self.split_descriptor_for(entry),
            calibrations=self._calibrations(),
        )[:2]

    def _show_binning_diagnostics(self, entry=None):
        """Open the Binning dialog on the Diagnostics tab.

        Without ``entry`` the file at the current preview index is used,
        matching Preview / Meta semantics. With ``entry``, that run is
        pre-selected in the Diagnostics tab's run picker.
        """
        if entry is None:
            checked = self.get_checked_files()
            if checked:
                idx = max(0, min(self._preview_index, len(checked) - 1))
                entry = checked[idx]
        if entry is not None and entry.get("is_merged"):
            QMessageBox.information(
                self, "Binning",
                "Merged spectra are pre-binned; per-bin diagnostics "
                "are not available for them.")
            return
        self._open_binning_dialog(focus_entry=entry,
                                   default_tab="diagnostics")

    def _show_source_diagnostics(self):
        """Plot experimental parameters vs run number for all checked files."""
        if _get_clstools() is None:
            QMessageBox.warning(self, "Diagnostics",
                                "clstools is required.")
            return
        checked = [e for e in self.get_checked_files()
                   if not e.get("is_merged")]
        if len(checked) < 1:
            QMessageBox.information(self, "Diagnostics",
                                    "No files selected.")
            return
        try:
            from gui.calibration import load_run_calibrated
            cal_map = self._calibrations()
            runs, cooler_vs, laser_sps, daq_times, n_events = (
                [], [], [], [], [])
            for e in checked:
                # A virtual split's events live in the parent ASDF; clstools
                # cannot open the .vasdf sidecar itself. Per-split metadata
                # overrides win over the parent's header, so the plotted
                # cooler/laser are the ones the split is actually fitted with.
                md = (e.get("metadata_override") or {}
                      if e.get("is_split") else {})
                data_path = (e["parent_path"] if e.get("is_split")
                             else e["path"])
                data = _get_clstools().CLSDataFrame()
                load_run_calibrated(data, data_path, cal_map,
                                    cal_order=self._cal_order.value())
                rn = e.get("run_number", "?")
                runs.append(str(rn))
                cooler_vs.append(
                    float(md.get("cooler_v") or 0)
                    or getattr(data, "Vcool_init", 0)
                    * getattr(data, "VCoolDiv", 10000))
                laser_sps.append(
                    float(md.get("laser_sp") or 0)
                    or getattr(data, "Laser_set", 0))
                daq_times.append(getattr(data, "DAQTStime", 0))
                n_events.append(getattr(data, "Size", 0))

            ps = get_plot_type_settings("preview")
            from matplotlib.figure import Figure
            fig = Figure(figsize=(ps["figsize_w"], ps["figsize_h"] * 1.6))
            x = np.arange(len(runs))

            panels = [
                ("Cooler Voltage [V]", cooler_vs),
                ("Laser Setpoint [cm\u207b\u00b9]", laser_sps),
                ("DAQ Time [s]", daq_times),
                ("Events", n_events),
            ]
            for i, (ylabel, vals) in enumerate(panels):
                ax = fig.add_subplot(4, 1, i + 1)
                arr = np.array(vals, dtype=float)
                ax.plot(x, arr, "o-", ms=5, lw=1.5)
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                if len(arr) > 1 and std_val > 0:
                    ax.axhline(mean_val, ls="--", color="gray",
                               alpha=0.5)
                    ax.fill_between(
                        x, mean_val - std_val,
                        mean_val + std_val,
                        alpha=0.1, color="blue")
                # Annotate mean ± std in top-right corner
                if len(arr) >= 1:
                    ax.text(0.98, 0.92,
                            f"\u03bc = {mean_val:.6g}  "
                            f"\u03c3 = {std_val:.4g}",
                            transform=ax.transAxes, fontsize=7,
                            ha="right", va="top",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", alpha=0.7))
                ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=7)
                # Disable scientific offset on y-axis
                ax.ticklabel_format(axis="y", useOffset=False,
                                    style="plain")
                ax.yaxis.get_major_formatter().set_useOffset(False)
                if i < len(panels) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(x)
                    ax.set_xticklabels(runs, fontsize=7, rotation=45)
                    ax.set_xlabel("Run Number", fontsize=8)
            fig.suptitle("Run Statistics", fontsize=ps["title_size"])

            try:
                if hasattr(self, "_diag_win") and self._diag_win:
                    self._diag_win.close()
            except RuntimeError:
                pass
            self._diag_win = PopupPlotWindow(
                fig, title="Run Statistics", parent=self)
            self._diag_win.show()
        except Exception as e:
            QMessageBox.critical(self, "Run Stats Error", str(e))

    # ── Serialization ──

    def to_dict(self):
        d = super().to_dict()
        files_out = []
        for e in self._file_entries:
            # Merged entries round-trip through "merged_entries" below;
            # writing them here too caused a duplicate phantom file on
            # reload (the loader parsed "merged://merged_7735_7740" as
            # a run path and extracted the leading int).
            if e.get("is_merged"):
                continue
            fd = {"path": e["path"], "run_number": e["run_number"],
                  "checked": e["checkbox"].isChecked()}
            ovr = e.get("binning_override") or {}
            if ovr:
                fd["binning_override"] = dict(ovr)
            files_out.append(fd)
        d["files"] = files_out
        d["Z"] = self._z_number.value()
        d["A"] = self._a_number.value()
        d["mass"] = self._mass_spin.value()
        d["mass_override"] = self._mass_override.isChecked()
        d["harmonic"] = self._harmonic_spin.value()
        d["e_lower"] = self._e_lower_spin.value()
        d["e_upper"] = self._e_upper_spin.value()
        d["tof_gate"] = {"enabled": self._tof_enable.isChecked(),
                         "lo": self._tof_lo.value(), "hi": self._tof_hi.value()}
        d["pmt_gate"] = self.get_pmt_gate()
        d["v_gate"] = {"enabled": self._vgate_enable.isChecked(),
                        "lo": self._vgate_lo.value(),
                        "hi": self._vgate_hi.value()}
        # F gate: stored in MHz (matching the UI); legacy saves used Hz
        # under "lo"/"hi" — see from_dict for the fallback.
        d["f_gate"] = {"enabled": self._fgate_enable.isChecked(),
                        "lo_mhz": self._fgate_lo.value(),
                        "hi_mhz": self._fgate_hi.value()}
        d["noise_filter"] = self._noise_filter.value()
        d["x_column"] = self._x_col_combo.currentText()
        d["yerr_mode"] = self._yerr_combo.currentText()
        d["bin_mode"] = self._bin_mode.currentText()
        d["xerr_mode"] = self._xerr_combo.currentText()
        d["bin_definition"] = self._bin_def_combo.currentText()
        d["bin_count"] = self._bin_count_spin.value()
        d["bin_width_mhz"] = self._bin_width_spin.value()
        d["step_multiple"] = self._step_mult_spin.value()
        d["override_enabled"] = self._ovr_enable.isChecked()
        d["cooler_override"] = self._cooler_spin.value()
        d["laser_override"] = self._laser_spin.value()
        d["cal_order"] = self._cal_order.value()
        d["cooler_correction"] = self._cooler_corr.currentText()
        d["ref_shift_mhz"] = self._ref_shift.value()
        # Merged entries
        merged_list = []
        # Scalar per_run fields that round-trip through YAML. We drop
        # per-file x/y/TOF arrays (would balloon the workspace file);
        # the constituent overlay stays a fresh-merge-only feature.
        # The kept fields are what project_voltage_merge_to_frequency
        # consults for its mean fallback and spread-warning checks.
        _per_run_scalar_keys = (
            "run_num", "path",
            "cooler_v", "laser_set", "laser_sp",
            "mass_amu", "harmonic",
            "ts_start", "ts_stop", "n_events",
            "centroid_correction_mhz", "centroid_correction_sigma_mhz",
            "centroid_correction_mode", "manual_offset_mhz",
        )

        def _to_py(v):
            """Strip numpy scalar types so PyYAML's safe dumper can
            serialise them. ASDF metadata pulls fields like TSstart /
            Size in as np.int64 / np.float64, which would otherwise be
            tagged python/object and reject on safe_load."""
            if hasattr(v, "item") and not isinstance(
                    v, (str, bytes, list, tuple, dict)):
                try:
                    return v.item()
                except (AttributeError, ValueError):
                    return v
            return v

        for e in self._file_entries:
            if e.get("is_merged") and "merged_data" in e:
                md = e["merged_data"]
                per_run_out = [
                    {k: _to_py(v) for k, v in (rd or {}).items()
                     if k in _per_run_scalar_keys}
                    for rd in (md.get("per_run") or [])
                ]
                merge_meta = {k: _to_py(v) for k, v in
                              (md.get("merge_metadata") or {}).items()}
                merged_list.append({
                    "merged_name": md.get("merged_name"),
                    "source_files": md.get("source_files", []),
                    "source_runs": md.get("source_runs", []),
                    "merge_params": md.get("merge_params", {}),
                    "metadata": md.get("metadata", {}),
                    "x": md["x"].tolist() if hasattr(md["x"], "tolist")
                         else list(md["x"]),
                    "y": md["y"].tolist() if hasattr(md["y"], "tolist")
                         else list(md["y"]),
                    "yerr": md["yerr"].tolist() if hasattr(md["yerr"], "tolist")
                            else list(md["yerr"]),
                    "bin_step_mhz": _to_py(md.get("bin_step_mhz", 0)),
                    # Preserve unit + projection metadata so voltage
                    # merges round-trip through V→F at view/fit time.
                    # Without these, the view defaults x_unit to MHz
                    # and the voltage axis is silently mislabelled.
                    "x_unit": md.get("x_unit", "MHz"),
                    "merge_metadata": merge_meta,
                    "per_run": per_run_out,
                })
        d["merged_entries"] = merged_list
        return d

    def from_dict(self, d):
        super().from_dict(d)
        # Clear existing files
        for e in list(self._file_entries):
            if "widget" in e:
                e["widget"].deleteLater()
        self._file_entries.clear()
        self._refresh_master_check_state()
        for fd in d.get("files", []):
            path = maybe_convert_path(fd["path"])
            # Backward compat: older YAMLs (before the to_dict skip)
            # wrote merged entries into "files" too. Skip those here;
            # the real merged entry is rebuilt from "merged_entries".
            if isinstance(path, str) and path.startswith("merged://"):
                continue
            self._add_file_entry(path)
            if not fd.get("checked", True):
                self._file_entries[-1]["checkbox"].setChecked(False)
            ovr = fd.get("binning_override") or {}
            if ovr:
                # Filter to whitelisted keys to keep YAML self-healing if
                # someone hand-edits it with a now-removed key.
                self._file_entries[-1]["binning_override"] = {
                    k: v for k, v in ovr.items()
                    if k in BINNING_OVERRIDE_KEYS}
                self._update_override_badge(self._file_entries[-1])

        self._z_number.setValue(d.get("Z", 1))
        self._a_number.setValue(d.get("A", 1))
        # Set mass override flag BEFORE writing the value, so the Z/A
        # auto-fill triggered above doesn't clobber a saved custom mass.
        override = bool(d.get("mass_override", False))
        self._mass_override.setChecked(override)
        if override:
            self._mass_spin.blockSignals(True)
            self._mass_spin.setValue(d.get("mass", 1.0))
            self._mass_spin.blockSignals(False)
        else:
            # mass_override defaults to off => trust the lookup
            self._mass_spin.setValue(d.get("mass", 1.0))
        self._harmonic_spin.setValue(d.get("harmonic", 4))
        self._e_lower_spin.setValue(d.get("e_lower", 0.0))
        self._e_upper_spin.setValue(d.get("e_upper", 0.0))

        tof = d.get("tof_gate", {})
        self._tof_enable.setChecked(tof.get("enabled", True))
        self._tof_lo.setValue(tof.get("lo", 38.0))
        self._tof_hi.setValue(tof.get("hi", 44.0))

        pmt = d.get("pmt_gate", [3, 4])
        for i, cb in enumerate(self._pmt_checks):
            if i < 4:
                cb.setChecked((i + 1) in pmt)
            else:
                cb.setChecked(0 in pmt)

        idx = self._x_col_combo.findText(d.get("x_column", "bins_center"))
        if idx >= 0:
            self._x_col_combo.setCurrentIndex(idx)
        idx = self._yerr_combo.findText(d.get("yerr_mode", "Poisson sqrt(y+1)"))
        if idx >= 0:
            self._yerr_combo.setCurrentIndex(idx)
        vg = d.get("v_gate", {})
        self._vgate_enable.setChecked(vg.get("enabled", False))
        self._vgate_lo.setValue(vg.get("lo", 0.0))
        self._vgate_hi.setValue(vg.get("hi", 0.0))
        fg = d.get("f_gate", {})
        self._fgate_enable.setChecked(fg.get("enabled", False))
        if "lo_mhz" in fg or "hi_mhz" in fg:
            self._fgate_lo.setValue(float(fg.get("lo_mhz", 0.0)))
            self._fgate_hi.setValue(float(fg.get("hi_mhz", 0.0)))
        else:
            # Legacy saves: lo/hi were in Hz. Convert to MHz for the new UI.
            self._fgate_lo.setValue(float(fg.get("lo", 0.0)) / 1e6)
            self._fgate_hi.setValue(float(fg.get("hi", 0.0)) / 1e6)
        self._noise_filter.setValue(d.get("noise_filter", 0))
        idx = self._bin_mode.findText(d.get("bin_mode", "Frequency"))
        if idx >= 0:
            self._bin_mode.setCurrentIndex(idx)
        idx = self._xerr_combo.findText(d.get("xerr_mode", "None"))
        if idx >= 0:
            self._xerr_combo.setCurrentIndex(idx)
        # Legacy fallback is a literal "Auto", NOT the module default:
        # saves that predate the bin_definition key were produced when
        # Auto was the only behavior, and a load must reproduce them
        # even though new blocks now default to "Per scan step".
        idx = self._bin_def_combo.findText(
            d.get("bin_definition", "Auto"))
        if idx >= 0:
            self._bin_def_combo.setCurrentIndex(idx)
        self._bin_count_spin.setValue(int(d.get("bin_count",
                                                DEFAULT_BIN_COUNT)))
        self._bin_width_spin.setValue(float(d.get("bin_width_mhz",
                                                  DEFAULT_BIN_WIDTH_MHZ)))
        self._step_mult_spin.setValue(int(d.get("step_multiple", 1)))
        self._update_bin_def_controls()
        ovr_on = d.get("override_enabled",
                       d.get("cooler_override", 0) > 0 or
                       d.get("laser_override", 0) > 0)
        self._ovr_enable.setChecked(ovr_on)
        self._cooler_spin.setValue(d.get("cooler_override", 0.0))
        self._laser_spin.setValue(d.get("laser_override", 0.0))
        self._cal_order.setValue(d.get("cal_order", 1))
        idx = self._cooler_corr.findText(d.get("cooler_correction", "pbp"))
        if idx >= 0:
            self._cooler_corr.setCurrentIndex(idx)
        # ref_shift was originally stored in Hz; new saves use ref_shift_mhz.
        if "ref_shift_mhz" in d:
            self._ref_shift.setValue(float(d.get("ref_shift_mhz", 0.0)))
        else:
            self._ref_shift.setValue(float(d.get("ref_shift", 0.0)) / 1e6)
        # Restore merged entries
        for md in d.get("merged_entries", []):
            md["x"] = np.array(md["x"])
            md["y"] = np.array(md["y"])
            md["yerr"] = np.array(md["yerr"])
            self._add_merged_entry(md)
        # Defensive re-evaluation: bin_mode is restored earlier in
        # load_state, before any merged entry is added, so the per-add
        # refresh is already correct. This re-run guards against future
        # reorderings where bin_mode might be restored after the merged
        # loop.
        self._refresh_merged_warning_badges()
        # Reset navigation to first file
        self._preview_index = 0
        self._update_nav_label()


# ══════════════════════════════════════════════════════════════════
#  Model Block
# ══════════════════════════════════════════════════════════════════

class ModelBlock(AnalysisBlock):
    """Model block: HFS model parameters, background, peak amplitudes."""
    BLOCK_TYPE = "Model"
    BLOCK_COLOR = "#4CAF50"
    # Widened 400 → 470 (2026-07 feedback): at 400 the parameter
    # table's expressions column was pushed out of view. Width is
    # persisted per project, so saved blocks keep their stored width.
    BLOCK_WIDTH = 470

    def __init__(self, name="HFS_1", parent=None):
        super().__init__(name, parent)
        layout = self._content_layout

        # ── Model type & assignment ──
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        self._type_combo.setToolTip(
            "Model type:\n"
            "HFS = hyperfine structure (nuclear spins, A/B constants)\n"
            "Voigt = single Voigt peak + background\n"
            "Skewed Voigt = asymmetric Voigt peak\n"
            "Exponential Decay = amplitude \u00d7 exp(\u2212ln2 \u00d7 x/\u03c4)\n"
            "Piecewise Constant = step function\n"
            "Polynomial = background polynomial")
        self._type_combo.addItems(["HFS", "Voigt", "Skewed Voigt",
                                        "Exponential Decay",
                                        "Piecewise Constant", "Polynomial"])
        self._type_combo.setFixedWidth(90)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        top_row.addWidget(self._type_combo)
        top_row.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        # Disabled: per-model source assignment is not wired into the fitter
        # (it builds every model on each source, and a simultaneous fit uses
        # sources[0]), so an editable combo here advertised a routing the
        # engine ignores (code review 2026-06-02, model-source-combo-unused).
        self._source_combo.setToolTip(
            "Reserved — per-model source assignment is not yet wired into the "
            "fitter; every model is fit on the source block's run(s).")
        self._source_combo.setEnabled(False)
        self._source_combo.setFixedWidth(100)
        top_row.addWidget(self._source_combo)
        top_row.addStretch()
        layout.addLayout(top_row)

        # ── Import buttons ──
        import_row = QHBoxLayout()
        import_btn = QPushButton("Import from Pre-Analysis")
        import_btn.setToolTip(
            "Import HFS model parameters (I, J, A, B, centroid, etc.)\n"
            "from the Pre-Analysis tab's HFS model overlay")
        import_btn.clicked.connect(self._import_from_preanalysis)
        import_row.addWidget(import_btn)
        retrieve_btn = QPushButton("Retrieve Last Fit")
        retrieve_btn.setToolTip(
            "Load fitted parameter values from the last iteration\n"
            "as new initial guesses for the next fit")
        retrieve_btn.clicked.connect(self._retrieve_last_fit)
        import_row.addWidget(retrieve_btn)
        layout.addLayout(import_row)

        # ── HFS Settings ──
        self._hfs_settings = QGroupBox("HFS Settings")
        hfs_lay = QHBoxLayout(self._hfs_settings)
        hfs_lay.setContentsMargins(4, 8, 4, 4)
        # Peak label + combo are kept as widgets so saved configs with a
        # peak_shape field round-trip, but are hidden because satlas2
        # 0.1.10 supports Voigt only for HFS. Use the dedicated Skewed
        # Voigt block for skewed peaks.
        self._peak_label = QLabel("Peak:")
        hfs_lay.addWidget(self._peak_label)
        self._peak_combo = QComboBox()
        self._peak_combo.setToolTip(
            "Line profile shape for each HFS peak")
        self._peak_combo.addItems(["Voigt", "Gaussian", "Lorentzian",
                                        "PseudoVoigt", "Skewed Voigt",
                                        "CrystalBall"])
        self._peak_combo.setFixedWidth(90)
        hfs_lay.addWidget(self._peak_combo)
        self._peak_label.hide()
        self._peak_combo.hide()
        self._racah_check = QCheckBox("Racah")
        self._racah_check.setChecked(True)
        self._racah_check.setToolTip(
            "Use Racah intensity ratios (fixed by angular momentum).\n"
            "Uncheck to fit individual peak amplitudes freely.")
        self._racah_check.toggled.connect(self._on_racah_toggled)
        hfs_lay.addWidget(self._racah_check)
        hfs_lay.addStretch()
        layout.addWidget(self._hfs_settings)

        # ── Sidepeak Settings ──
        self._sidepeak_grp = QGroupBox("Sidepeaks")
        sp_lay = QHBoxLayout(self._sidepeak_grp)
        sp_lay.setContentsMargins(4, 8, 4, 4)
        sp_lay.addWidget(QLabel("N:"))
        self._sidepeak_n = QSpinBox()
        self._sidepeak_n.setRange(0, 20)
        self._sidepeak_n.setValue(0)
        self._sidepeak_n.setToolTip("Number of sidepeaks (0 = disabled)")
        sp_lay.addWidget(self._sidepeak_n)
        sp_lay.addWidget(QLabel("Offset:"))
        self._sidepeak_offset = _make_double(0.0, -1e6, 1e6, 2, 1.0,
                                              tooltip="Sidepeak offset [MHz]")
        self._sidepeak_offset.setFixedWidth(80)
        sp_lay.addWidget(self._sidepeak_offset)
        self._sidepeak_poisson = QCheckBox("Poisson")
        self._sidepeak_poisson.setToolTip(
            "Use Poisson distribution for sidepeak amplitudes")
        sp_lay.addWidget(self._sidepeak_poisson)
        sp_lay.addStretch()
        layout.addWidget(self._sidepeak_grp)

        # ── Parameter table ──
        self._param_table = QTableWidget()
        self._param_table.setColumnCount(5)
        self._param_table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Bounds", "Mode", "Expression"])
        self._param_table.horizontalHeader().setStretchLastSection(True)
        self._param_table.horizontalHeader().setSectionResizeMode(
            _PT_PARAM, QHeaderView.ResizeMode.ResizeToContents)
        self._param_table.horizontalHeader().setSectionResizeMode(
            _PT_VALUE, QHeaderView.ResizeMode.Fixed)
        self._param_table.setColumnWidth(_PT_VALUE, 110)
        self._param_table.horizontalHeader().setSectionResizeMode(
            _PT_BOUNDS, QHeaderView.ResizeMode.Fixed)
        self._param_table.setColumnWidth(_PT_BOUNDS, 110)
        self._param_table.horizontalHeader().setSectionResizeMode(
            _PT_MODE, QHeaderView.ResizeMode.Fixed)
        self._param_table.setColumnWidth(_PT_MODE, 90)
        self._param_table.verticalHeader().setVisible(False)
        self._param_table.setAlternatingRowColors(True)
        self._param_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._param_table.itemChanged.connect(self._on_param_item_changed)
        layout.addWidget(self._param_table, 1)  # stretch=1 to fill space

        # ── Peak Amplitudes (collapsible) ──
        self._peaks_toggle = QCheckBox("Show Peak Amplitudes")
        self._peaks_toggle.setToolTip(
            "Show per-peak amplitude table for manual control.\n"
            "Only active when Racah is unchecked.")
        self._peaks_toggle.toggled.connect(self._toggle_peaks)
        layout.addWidget(self._peaks_toggle)

        # Check All / Uncheck All buttons for peak amplitude vary
        self._peaks_btn_row = QWidget()
        pbr = QHBoxLayout(self._peaks_btn_row)
        pbr.setContentsMargins(0, 0, 0, 0)
        pbr.setSpacing(4)
        check_all = QPushButton("Check All")
        check_all.setFixedHeight(22)
        check_all.clicked.connect(lambda: self._set_all_peak_vary(True))
        pbr.addWidget(check_all)
        uncheck_all = QPushButton("Uncheck All")
        uncheck_all.setFixedHeight(22)
        uncheck_all.clicked.connect(lambda: self._set_all_peak_vary(False))
        pbr.addWidget(uncheck_all)
        pbr.addStretch()
        self._peaks_btn_row.setVisible(False)
        layout.addWidget(self._peaks_btn_row)

        self._peaks_table = QTableWidget()
        self._peaks_table.setColumnCount(3)
        self._peaks_table.setHorizontalHeaderLabels(
            ["Peak", "Amplitude", "Vary"])
        self._peaks_table.verticalHeader().setVisible(False)
        self._peaks_table.setAlternatingRowColors(True)
        self._peaks_table.setVisible(False)
        self._peaks_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._peaks_table)

        layout.addStretch()

        # ── Parameter data storage ──
        self._param_rows = []  # list of dicts for HFS params
        self._peak_lines = []  # list of HFS line labels

        # Build default HFS parameter table
        self._build_hfs_params()

    def _on_type_changed(self, text):
        is_hfs = (text == "HFS")
        self._hfs_settings.setVisible(is_hfs)
        self._sidepeak_grp.setVisible(is_hfs)
        self._peaks_toggle.setVisible(is_hfs)
        if not is_hfs:
            self._peaks_table.setVisible(False)
        if text == "HFS":
            self._build_hfs_params()
        elif text == "Voigt":
            self._build_voigt_params()
        elif text == "Skewed Voigt":
            self._build_skewvoigt_params()
        elif text == "Exponential Decay":
            self._build_expdecay_params()
        elif text == "Piecewise Constant":
            self._build_piecewise_params()
        elif text == "Polynomial":
            self._build_polynomial_params()
        self.block_changed.emit()

    def _build_hfs_params(self):
        """Build the HFS parameter table."""
        params = [
            ("I", 3.5, False, 0.0, 20.0),
            ("Jl", 4.5, False, 0.0, 20.0),
            ("Ju", 5.5, False, 0.0, 20.0),
            ("Al", 0.0, True, -1e6, 1e6),
            ("Au", 0.0, True, -1e6, 1e6),
            ("Bl", 0.0, True, -1e6, 1e6),
            ("Bu", 0.0, True, -1e6, 1e6),
            ("Cl", 0.0, False, -1e6, 1e6),
            ("Cu", 0.0, False, -1e6, 1e6),
            ("centroid", 0.0, True, -1e8, 1e8),
            ("scale", 100.0, True, 0.0, 1e8),
            ("FWHMG", 50.0, True, 0.0, 1e4),
            ("FWHML", 50.0, True, 0.0, 1e4),
            ("Bkg_p0", 0.0, True, -1e6, 1e6),
        ]
        self._set_param_table(params)

    def _build_voigt_params(self):
        params = [
            ("A", 100.0, True, 0.0, 1e8),
            ("mu", 0.0, True, -1e8, 1e8),
            ("FWHMG", 50.0, True, 0.0, 1e4),
            ("FWHML", 50.0, True, 0.0, 1e4),
            ("Bkg_p0", 0.0, True, -1e6, 1e6),
        ]
        self._set_param_table(params)

    def _build_skewvoigt_params(self):
        params = [
            ("A", 100.0, True, 0.0, 1e8),
            ("mu", 0.0, True, -1e8, 1e8),
            ("FWHMG", 50.0, True, 0.0, 1e4),
            ("FWHML", 50.0, True, 0.0, 1e4),
            ("Skew", 0.0, True, -10.0, 10.0),
            ("Bkg_p0", 0.0, True, -1e6, 1e6),
        ]
        self._set_param_table(params)

    def _build_expdecay_params(self):
        params = [
            ("amplitude", 100.0, True, 0.0, 1e8),
            ("halflife", 1.0, True, 0.001, 1e8),
        ]
        self._set_param_table(params)

    def _build_piecewise_params(self):
        params = [
            ("value0", 0.0, True, 0.0, 1e8),
            ("value1", 0.0, True, 0.0, 1e8),
            ("bound0", 0.0, False, -1e8, 1e8),
        ]
        self._set_param_table(params)

    def _build_polynomial_params(self):
        params = [
            ("p0", 0.0, True, -1e8, 1e8),
            ("p1", 0.0, True, -1e8, 1e8),
        ]
        self._set_param_table(params)

    def _set_param_table(self, params):
        """Populate parameter table from list of (name, value, vary, min, max)."""
        self._param_table.blockSignals(True)
        self._param_table.setRowCount(len(params))
        self._param_rows = []

        for i, (name, value, vary, pmin, pmax) in enumerate(params):
            # Name (read-only)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(i, _PT_PARAM, name_item)

            # Value (spinbox)
            val_spin = _make_analysis_spin(value, -1e12, 1e12, 4, 1.0)
            val_spin.valueChanged.connect(lambda v: self.block_changed.emit())
            self._param_table.setCellWidget(i, _PT_VALUE, val_spin)

            # Bounds button
            bounds_btn = _BoundsButton(pmin, pmax)
            bounds_btn.bounds_changed.connect(
                lambda: self.block_changed.emit())
            self._param_table.setCellWidget(i, _PT_BOUNDS, bounds_btn)

            # Mode (single source of truth for vary + expression).
            # Free = vary on, empty expression. Fixed = vary off, empty.
            # Equal/Ratio/Offset/Custom = vary off + expression.
            # Non-fit params (I/Jl/Ju) lock to Fixed.
            mode_combo = QComboBox()
            mode_combo.addItems(_CONSTRAINT_MODES)
            is_non_fit = name in _NON_FIT_PARAMS_FOR_VALIDATION
            if is_non_fit:
                mode_combo.setCurrentText("Fixed")
                mode_combo.setEnabled(False)
                mode_combo.setToolTip(
                    "Physics metadata; not fit. Mode locked to Fixed.")
            else:
                mode_combo.setCurrentText("Free" if vary else "Fixed")
                mode_combo.setToolTip(
                    "Free: parameter floats freely (vary).\n"
                    "Fixed: hold the current value (no vary).\n"
                    "Equal/Ratio/Offset: link to another local param.\n"
                    "Custom: write a free-form expression in the next column.")
            mode_combo.currentTextChanged.connect(
                lambda text, ri=i: self._on_mode_combo_changed(ri, text))
            # Live-tint the combo background by current mode for fast
            # visual scanning of the parameter table.
            mode_combo.currentTextChanged.connect(
                lambda _, mc=mode_combo: _apply_mode_combo_color(mc))
            _apply_mode_combo_color(mode_combo)
            self._param_table.setCellWidget(i, _PT_MODE, mode_combo)

            # Expression (text). Live validation sets a more specific
            # tooltip; this default is the success-state guidance.
            expr_item = QTableWidgetItem("")
            expr_item.setToolTip(_MODEL_EXPR_TOOLTIP_OK)
            self._param_table.setItem(i, _PT_EXPR, expr_item)

            self._param_rows.append({
                "name": name, "value": val_spin,
                "bounds": bounds_btn, "mode": mode_combo, "expr_row": i,
                # Remembered Free/Fixed preference, used when an
                # expression cell goes empty (constraint cleared, or
                # dialog cancelled). Initial state matches the
                # passed-in vary flag.
                "last_empty_mode": "Free" if vary else "Fixed",
            })

        self._param_table.blockSignals(False)
        # Re-validate cells (handles param-set changes when type switches
        # and any non-empty expression text loaded by from_dict).
        self._validate_all_expr_cells()

    def _on_param_item_changed(self, item):
        """Live-validate the Expression cell as the user edits."""
        if item.column() != _PT_EXPR:
            return
        self._validate_expr_cell(item)
        # Re-infer mode now that expression text changed.
        self._refresh_mode_combo(item.row())

    def _validate_expr_cell(self, item):
        """Color one Expression cell based on validate_single_expression."""
        expr = item.text()
        local = {row["name"] for row in self._param_rows
                 if row["name"] not in _NON_FIT_PARAMS_FOR_VALIDATION}
        issue = validate_single_expression(expr, "model", local)
        # Block signals while we change BackgroundRole on the item --
        # otherwise setBackground re-fires itemChanged and recurses.
        self._param_table.blockSignals(True)
        try:
            if issue is None:
                item.setBackground(QBrush())
                item.setToolTip(_MODEL_EXPR_TOOLTIP_OK)
            else:
                item.setBackground(_EXPR_ERROR_BG)
                item.setToolTip(issue.message)
        finally:
            self._param_table.blockSignals(False)

    def _validate_all_expr_cells(self):
        """Re-validate every Expression cell currently in the table."""
        for row in self._param_rows:
            it = self._param_table.item(row["expr_row"], _PT_EXPR)
            if it is not None:
                self._validate_expr_cell(it)
        # After validation, sync the Mode combos to the new state.
        for ri in range(len(self._param_rows)):
            self._refresh_mode_combo(ri)

    def _refresh_mode_combo(self, row_idx: int):
        """Re-derive the Mode for one row from its current expression
        and update the combo silently (no change-handler fire).

        Mode is the single source of truth for vary, so this helper
        runs whenever the expression text changes (or when a user
        cancels the Equal/Ratio/Offset dialog):

        - Non-empty expr -> Equal/Ratio/Offset/Custom (inferred from AST).
        - Empty expr -> use the row's remembered ``last_empty_mode``
          (Free or Fixed). That sticky preference is updated only
          when the user explicitly picks Free or Fixed from the combo,
          so cancelling a dialog or clearing a constraint expression
          reverts to whatever they had before, not a hard-coded
          default.
        """
        if not (0 <= row_idx < len(self._param_rows)):
            return
        row = self._param_rows[row_idx]
        if row["name"] in _NON_FIT_PARAMS_FOR_VALIDATION:
            return  # locked Fixed
        expr_item = self._param_table.item(row["expr_row"], _PT_EXPR)
        expr = (expr_item.text() if expr_item else "").strip()
        combo = row["mode"]
        current = combo.currentText()
        if expr:
            mode, _ = infer_constraint_mode(expr, vary=False)
        else:
            mode = row.get("last_empty_mode", "Fixed")
        if mode != current:
            combo.blockSignals(True)
            try:
                combo.setCurrentText(mode)
            finally:
                combo.blockSignals(False)
            # Signals were blocked above, so re-tint by hand.
            _apply_mode_combo_color(combo)

    def _on_mode_combo_changed(self, row_idx: int, new_mode: str):
        """User picked a Mode for `row_idx`. Apply or prompt as needed."""
        if not (0 <= row_idx < len(self._param_rows)):
            return
        row = self._param_rows[row_idx]
        pname = row["name"]
        if pname in _NON_FIT_PARAMS_FOR_VALIDATION:
            return  # locked Fixed; this signal shouldn't fire anyway

        expr_item = self._param_table.item(row["expr_row"], _PT_EXPR)
        if expr_item is None:
            return

        if new_mode in ("Free", "Fixed"):
            # Both are "no expression". Update the sticky preference
            # so future empty-expr states (cleared constraint,
            # cancelled dialog) revert here. Then clear the cell.
            row["last_empty_mode"] = new_mode
            self._set_row_expr(row_idx, "")
        elif new_mode == "Custom":
            # Don't change the expression -- user will edit it directly.
            pass
        elif new_mode in ("Equal", "Ratio", "Offset"):
            local = [r["name"] for r in self._param_rows
                     if r["name"] != pname
                     and r["name"] not in _NON_FIT_PARAMS_FOR_VALIDATION]
            if not local:
                QMessageBox.information(
                    self, f"{new_mode}",
                    "No other parameters available to link to.")
                self._refresh_mode_combo(row_idx)
                return
            # Pre-fill from current expression if it already matches.
            cur_expr = expr_item.text()
            cur_mode, cur_extras = infer_constraint_mode(
                cur_expr, vary=False)
            cur_target = (cur_extras.get("target")
                          if cur_mode == new_mode else None)
            cur_value = (cur_extras.get("value")
                         if cur_mode == new_mode else None)
            dlg = _ConstraintModeDialog(
                self, new_mode, local,
                current_target=cur_target, current_value=cur_value)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                target, value = dlg.result_data()
                if new_mode == "Equal":
                    expr = target
                elif new_mode == "Ratio":
                    expr = f"{value} * {target}"
                else:  # Offset
                    if value < 0:
                        expr = f"{target} - {abs(value)}"
                    else:
                        expr = f"{target} + {value}"
                self._set_row_expr(row_idx, expr)
            else:
                # Cancelled -- restore the displayed mode to current state.
                self._refresh_mode_combo(row_idx)

    def _set_row_expr(self, row_idx: int, expr: str):
        """Set the Expression cell text for `row_idx`. Validation and
        mode-combo refresh fire automatically via the itemChanged hook."""
        row = self._param_rows[row_idx]
        it = self._param_table.item(row["expr_row"], _PT_EXPR)
        if it is not None:
            it.setText(expr)

    def _on_racah_toggled(self, checked):
        if not checked and self._type_combo.currentText() == "HFS":
            self._peaks_toggle.setChecked(True)
            self._rebuild_peak_table()
        self.block_changed.emit()

    def _toggle_peaks(self, on):
        self._peaks_table.setVisible(on)
        self._peaks_btn_row.setVisible(on)
        if on:
            self._rebuild_peak_table()

    def _set_all_peak_vary(self, checked):
        """Set all peak amplitude Vary checkboxes to checked/unchecked."""
        for i in range(self._peaks_table.rowCount()):
            vary_widget = self._peaks_table.cellWidget(i, 2)
            if vary_widget:
                cb = vary_widget.findChild(QCheckBox)
                if cb:
                    cb.setChecked(checked)

    def _rebuild_peak_table(self):
        """Regenerate peak amplitude rows based on I, Jl, Ju."""
        I = self._get_param_value("I")
        Jl = self._get_param_value("Jl")
        Ju = self._get_param_value("Ju")
        if I is None or Jl is None or Ju is None:
            return
        try:
            import satlas2  # lazy import (avoids satlas2 cost at startup)
            hfs = satlas2.HFS(I=I, J=[Jl, Ju], A=[0, 0], B=[0, 0],
                              C=[0, 0], df=0, scale=1, racah=True,
                              fwhmg=50, fwhml=50)
        except Exception:
            return

        lines = hfs.lines
        self._peak_lines = list(lines)
        self._peaks_table.setRowCount(len(lines))
        for i, label in enumerate(lines):
            racah_val = hfs.params[f"Amp{label}"].value

            name_item = QTableWidgetItem(label)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._peaks_table.setItem(i, 0, name_item)

            amp_spin = _make_analysis_spin(racah_val, 0.0, 100.0, 6, 0.01)
            amp_spin.setEnabled(not self._racah_check.isChecked())
            self._peaks_table.setCellWidget(i, 1, amp_spin)

            vary_cb = QCheckBox()
            vary_cb.setChecked(not self._racah_check.isChecked())
            vary_w = QWidget()
            vary_lay = QHBoxLayout(vary_w)
            vary_lay.setContentsMargins(0, 0, 0, 0)
            vary_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vary_lay.addWidget(vary_cb)
            self._peaks_table.setCellWidget(i, 2, vary_w)

    def _get_param_value(self, name):
        for row in self._param_rows:
            if row["name"] == name:
                return row["value"].value()
        return None

    def _get_param_vary(self, name):
        for row in self._param_rows:
            if row["name"] == name:
                return row["mode"].currentText() == "Free"
        return False

    def apply_fixed_param(self, name, value, fixed):
        """Carry an imported parameter's value and fixed flag onto its row.

        The value is set BEFORE the Mode is changed so a Fixed parameter
        holds the imported value. ``fixed`` True -> Mode "Fixed" (no
        vary); False -> Mode "Free" (vary). No-op if the row is absent.
        """
        for row in self._param_rows:
            if row["name"] == name:
                row["value"].setValue(float(value))
                row["mode"].setCurrentText("Fixed" if fixed else "Free")
                return True
        return False

    def _get_param_min(self, name):
        for row in self._param_rows:
            if row["name"] == name:
                return row["bounds"].get_min()
        return -np.inf

    def _get_param_max(self, name):
        for row in self._param_rows:
            if row["name"] == name:
                return row["bounds"].get_max()
        return np.inf

    def _get_param_expr(self, name):
        for row in self._param_rows:
            if row["name"] == name:
                item = self._param_table.item(row["expr_row"], _PT_EXPR)
                return item.text() if item else ""
        return ""

    def get_model_config(self):
        """Return model configuration dict for fitting."""
        model_type = self._type_combo.currentText()
        config = {
            "type": model_type,
            "name": self._block_name,
            "source": self._source_combo.currentText(),
            "params": {},
            "prefunc": "None",  # kept for save-file compatibility
        }
        for row in self._param_rows:
            config["params"][row["name"]] = {
                "value": row["value"].value(),
                "vary": row["mode"].currentText() == "Free",
                "min": row["bounds"].get_min(),
                "max": row["bounds"].get_max(),
                "expr": self._param_table.item(row["expr_row"], _PT_EXPR).text()
                        if self._param_table.item(row["expr_row"], _PT_EXPR) else "",
            }
        if model_type == "HFS":
            peak_map = self._peak_combo.currentText().lower()
            _peak_name_map = {
                "skewed voigt": "skewvoigt",
                "pseudovoigt": "pseudovoigt",
                "crystalball": "crystalball",
            }
            peak_map = _peak_name_map.get(peak_map, peak_map)
            config["peak_shape"] = peak_map
            config["racah"] = self._racah_check.isChecked()
            # Sidepeak settings
            n_side = self._sidepeak_n.value()
            if n_side > 0:
                config["sidepeak_n"] = n_side
                config["sidepeak_offset"] = self._sidepeak_offset.value()
                config["sidepeak_poisson"] = self._sidepeak_poisson.isChecked()
            # Peak amplitudes
            if not self._racah_check.isChecked():
                config["peak_amplitudes"] = {}
                for i in range(self._peaks_table.rowCount()):
                    label_item = self._peaks_table.item(i, 0)
                    amp_widget = self._peaks_table.cellWidget(i, 1)
                    vary_widget = self._peaks_table.cellWidget(i, 2)
                    if label_item and amp_widget:
                        vary_cb = vary_widget.findChild(QCheckBox) if vary_widget else None
                        config["peak_amplitudes"][label_item.text()] = {
                            "value": amp_widget.value(),
                            "vary": vary_cb.isChecked() if vary_cb else True,
                        }
        return config

    def update_source_list(self, source_names):
        """Update the source combo box with available source block names."""
        # No-op when items are unchanged. Why: this is called from
        # _on_block_changed on every block_changed.emit(), including the
        # Auto-Fit Accept undo command, which only changes spinbox values.
        # On macOS Qt 6.6, clear()+addItems() from inside a QUndoCommand.redo()
        # signal chain can crash inside QListView::selectionChanged with
        # "index 0 beyond bounds for empty array" in libqcocoa.
        new_items = list(source_names)
        current_items = [self._source_combo.itemText(i)
                         for i in range(self._source_combo.count())]
        if new_items == current_items:
            return
        current = self._source_combo.currentText()
        self._source_combo.blockSignals(True)
        self._source_combo.clear()
        self._source_combo.addItems(new_items)
        idx = self._source_combo.findText(current)
        if idx >= 0:
            self._source_combo.setCurrentIndex(idx)
        self._source_combo.blockSignals(False)

    def _import_from_preanalysis(self):
        """Import model parameters from a Pre-Analysis HFS model."""
        pa = _choose_pa_project(self)
        if pa is None:
            return
        panels = getattr(pa, '_model_panels', [])
        if not panels:
            QMessageBox.information(self, "Import",
                                    "No HFS models found in Pre-Analysis tab.")
            return
        # If multiple models, let user choose
        if len(panels) == 1:
            panel = panels[0]
        else:
            from PySide6.QtWidgets import QInputDialog
            names = [p.model_name for p in panels]
            choice, ok = QInputDialog.getItem(
                self, "Select Model",
                "Import parameters from:", names, 0, False)
            if not ok:
                return
            panel = panels[names.index(choice)]
        mp = panel.get_model_params()
        # Map pre-analysis params to our table
        mapping = {
            "I": mp.get("I", 3.5),
            "Jl": mp.get("Jl", 4.5),
            "Ju": mp.get("Ju", 5.5),
            "Al": mp.get("Al", 0.0),
            "Au": mp.get("Au", 0.0),
            "Bl": mp.get("Bl", 0.0),
            "Bu": mp.get("Bu", 0.0),
            "centroid": mp.get("centroid", 0.0),
            "scale": mp.get("scale", 100.0),
            "FWHMG": mp.get("fwhm_g", 50.0),
            "FWHML": mp.get("fwhm_l", 50.0),
            "Bkg_p0": mp.get("bkg", 0.0),
        }
        for row in self._param_rows:
            if row["name"] in mapping:
                row["value"].setValue(mapping[row["name"]])

        # Carry the A/B hyperfine fixed state into the model block.
        # Pre-Analysis expresses "fix A/B" via the Al/Au and Bl/Bu ratio
        # locks; hyperfine_state() expands that to {name: (value, fixed)}.
        # apply_fixed_param sets the value first, then the Mode combo
        # ("Fixed" = vary off / "Free" = vary), so a constant the user
        # fixed in Pre-Analysis arrives Fixed here instead of Free.
        if hasattr(panel, "hyperfine_state"):
            for name, (value, fixed) in panel.hyperfine_state().items():
                self.apply_fixed_param(name, value, fixed)

        # Import peak amplitude overrides (non-Racah peaks)
        peak_overrides = panel.get_peak_overrides()
        if peak_overrides:
            # Uncheck Racah to enable free peak amplitudes;
            # this triggers _on_racah_toggled which rebuilds the peak table
            self._racah_check.setChecked(False)
            # Set imported amplitude values into the peak table
            for i in range(self._peaks_table.rowCount()):
                label_item = self._peaks_table.item(i, 0)
                amp_widget = self._peaks_table.cellWidget(i, 1)
                vary_widget = self._peaks_table.cellWidget(i, 2)
                if label_item and label_item.text() in peak_overrides:
                    if amp_widget:
                        amp_widget.setValue(peak_overrides[label_item.text()])
                    vary_cb = (vary_widget.findChild(QCheckBox)
                               if vary_widget else None)
                    if vary_cb:
                        vary_cb.setChecked(True)

        self.block_changed.emit()

    def _retrieve_last_fit(self):
        """Retrieve fitted parameter values from the last iteration."""
        # Walk up to find the parent AnalysisProject
        from gui.analysis.project import AnalysisProject
        project = self.parent()
        while project is not None and not isinstance(project, AnalysisProject):
            project = project.parent()
        if project is None or project._last_results is None:
            QMessageBox.information(
                self, "Retrieve",
                "No fit results available yet.\n"
                "Run a fit first, then retrieve the fitted values.")
            return

        results = project._last_results
        model_name = self._block_name

        # Collect fitted values from all runs for this model.
        # params_df columns: Source, Model, Parameter, Value, ...
        # The fitter sanitizes model names (dots -> _) before storing
        # them, so apply the same substitution to match rows below.
        import re
        safe_name = re.sub(r'[^A-Za-z0-9_]', '_', model_name)
        fitted = {}
        for r in results:
            if not r.get("success"):
                continue
            pdf = r.get("params_df", {})
            models = pdf.get("Model", [])
            params = pdf.get("Parameter", [])
            values = pdf.get("Value", [])
            bkg_name = f"{model_name}_bkg"
            safe_bkg = f"{safe_name}_bkg"
            for m, p, v in zip(models, params, values):
                if m == model_name or m == safe_name:
                    fitted.setdefault(p, []).append(float(v))
                elif m == bkg_name or m == safe_bkg:
                    # The background is fit as a SEPARATE model; collect its
                    # params under a "_bkg_" prefix so the Bkg_p0 row can find
                    # p0 below. Previously only the main model's rows were
                    # collected, so Retrieve Last Fit silently skipped Bkg_p0
                    # while reporting it updated N parameters (code review
                    # 2026-06-02, retrieve-last-fit-bkg-silent-miss).
                    fitted.setdefault(f"_bkg_{p}", []).append(float(v))

        if not fitted:
            QMessageBox.information(
                self, "Retrieve",
                f"No fitted parameters found for model '{model_name}'.\n"
                "Check that the model name matches.")
            return

        # Average across runs and apply to the parameter table
        import numpy as _np
        count = 0
        for row in self._param_rows:
            pname = row["name"]
            # Handle background parameter naming
            lookup = pname
            if pname == "Bkg_p0":
                # Background p0 collected from the separate _bkg model above.
                lookup = "_bkg_p0"
            if lookup in fitted:
                avg = float(_np.mean(fitted[lookup]))
                row["value"].setValue(avg)
                count += 1
            elif pname in fitted:
                avg = float(_np.mean(fitted[pname]))
                row["value"].setValue(avg)
                count += 1

        # Per-peak amplitudes: only fit when racah is off, but pull them
        # back regardless of the current toggle state — the user might
        # have re-enabled racah just to inspect ratios.
        is_hfs = self._type_combo.currentText() == "HFS"
        if is_hfs and self._peaks_table.rowCount() > 0:
            peak_count = 0
            for i in range(self._peaks_table.rowCount()):
                label_item = self._peaks_table.item(i, 0)
                amp_widget = self._peaks_table.cellWidget(i, 1)
                if not (label_item and amp_widget):
                    continue
                amp_key = f"Amp{label_item.text()}"
                if amp_key in fitted:
                    avg = float(_np.mean(fitted[amp_key]))
                    amp_widget.setValue(avg)
                    peak_count += 1
            count += peak_count

        self.block_changed.emit()
        QMessageBox.information(
            self, "Retrieve",
            f"Updated {count} parameters from last fit results.\n"
            f"Model: {model_name}")

    # ── Serialization ──

    def to_dict(self):
        d = super().to_dict()
        d["model_type"] = self._type_combo.currentText()
        d["source_assign"] = self._source_combo.currentText()
        d["peak_shape"] = self._peak_combo.currentText()
        d["racah"] = self._racah_check.isChecked()
        d["params"] = {}
        for row in self._param_rows:
            expr_item = self._param_table.item(row["expr_row"], _PT_EXPR)
            d["params"][row["name"]] = {
                "value": row["value"].value(),
                "vary": row["mode"].currentText() == "Free",
                "min": row["bounds"].get_min(),
                "max": row["bounds"].get_max(),
                "min_enabled": row["bounds"].is_min_enabled(),
                "max_enabled": row["bounds"].is_max_enabled(),
                "expr": expr_item.text() if expr_item else "",
            }
        # Sidepeak settings
        d["sidepeak_n"] = self._sidepeak_n.value()
        d["sidepeak_offset"] = self._sidepeak_offset.value()
        d["sidepeak_poisson"] = self._sidepeak_poisson.isChecked()
        d["peaks_shown"] = self._peaks_toggle.isChecked()
        # Save peak amplitudes
        d["peak_amplitudes"] = {}
        for i in range(self._peaks_table.rowCount()):
            label_item = self._peaks_table.item(i, 0)
            amp_widget = self._peaks_table.cellWidget(i, 1)
            vary_widget = self._peaks_table.cellWidget(i, 2)
            if label_item and amp_widget:
                vary_cb = vary_widget.findChild(QCheckBox) if vary_widget else None
                d["peak_amplitudes"][label_item.text()] = {
                    "value": amp_widget.value(),
                    "vary": vary_cb.isChecked() if vary_cb else True,
                }
        return d

    def from_dict(self, d):
        super().from_dict(d)
        idx = self._type_combo.findText(d.get("model_type", "HFS"))
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        idx = self._source_combo.findText(d.get("source_assign", ""))
        if idx >= 0:
            self._source_combo.setCurrentIndex(idx)
        idx = self._peak_combo.findText(d.get("peak_shape", "Voigt"))
        if idx >= 0:
            self._peak_combo.setCurrentIndex(idx)
        self._racah_check.setChecked(d.get("racah", True))
        # Sidepeak settings
        self._sidepeak_n.setValue(d.get("sidepeak_n", 0))
        self._sidepeak_offset.setValue(d.get("sidepeak_offset", 0.0))
        self._sidepeak_poisson.setChecked(d.get("sidepeak_poisson", False))
        # prefunc removed (Source block handles axis conversion)

        params = d.get("params", {})
        for row in self._param_rows:
            if row["name"] in params:
                p = params[row["name"]]
                row["value"].setValue(p.get("value", 0.0))
                row["bounds"].set_bounds(
                    p.get("min", -1e12), p.get("max", 1e12),
                    p.get("min_enabled"), p.get("max_enabled"))
                # Restore the row's "no-constraint preference" first
                # so that if the user later clears the saved
                # constraint expression, the mode reverts to the
                # original Free/Fixed choice rather than defaulting.
                # Then set the Mode combo for empty exprs (the
                # auto-inference triggered by setText handles
                # non-empty exprs correctly via _refresh_mode_combo).
                if row["name"] not in _NON_FIT_PARAMS_FOR_VALIDATION:
                    saved_vary = p.get("vary", True)
                    row["last_empty_mode"] = (
                        "Free" if saved_vary else "Fixed")
                    expr = (p.get("expr") or "").strip()
                    if not expr:
                        target_mode = row["last_empty_mode"]
                        combo = row["mode"]
                        combo.blockSignals(True)
                        try:
                            combo.setCurrentText(target_mode)
                        finally:
                            combo.blockSignals(False)
                expr_item = self._param_table.item(row["expr_row"], _PT_EXPR)
                if expr_item:
                    expr_item.setText(p.get("expr", ""))

        # The Mode tint is driven by ``currentTextChanged``, and the loop above
        # sets the combo with signals blocked (it has to: the handler applies
        # constraints and can raise dialogs). So nothing repainted the cells,
        # and a parameter saved as Fixed reloaded showing the word "Fixed" on
        # the green Free background it happened to be created with -- the
        # colour, which is what the eye actually reads, said the opposite of
        # the truth. Re-tint every row here rather than at each assignment, so
        # the expression-driven modes are covered too.
        for row in self._param_rows:
            _apply_mode_combo_color(row["mode"])

        self._peaks_toggle.setChecked(d.get("peaks_shown", False))

        # Restore peak amplitudes
        saved_amps = d.get("peak_amplitudes", {})
        if saved_amps and self._peaks_table.rowCount() > 0:
            for i in range(self._peaks_table.rowCount()):
                label_item = self._peaks_table.item(i, 0)
                if not label_item:
                    continue
                label = label_item.text()
                if label in saved_amps:
                    amp_w = self._peaks_table.cellWidget(i, 1)
                    vary_w = self._peaks_table.cellWidget(i, 2)
                    if amp_w:
                        amp_w.setValue(saved_amps[label].get("value", 0.0))
                    if vary_w:
                        cb = vary_w.findChild(QCheckBox)
                        if cb:
                            cb.setChecked(saved_amps[label].get("vary", True))


# ══════════════════════════════════════════════════════════════════
#  Fitter Block
# ══════════════════════════════════════════════════════════════════

class FitterBlock(AnalysisBlock):
    """Fitter block: fit mode, method, statistics, sharing/constraints,
    execution."""
    BLOCK_TYPE = "Fitter"
    BLOCK_COLOR = "#FF9800"
    BLOCK_WIDTH = 400

    fit_requested = Signal()
    auto_fit_requested = Signal()
    fit_cancel_requested = Signal()
    revert_requested = Signal()

    def __init__(self, name="Fitter_1", parent=None):
        super().__init__(name, parent)
        layout = self._content_layout

        # ── Fit Mode ──
        mode_grp = QGroupBox("Fit Mode")
        mode_lay = QVBoxLayout(mode_grp)
        mode_lay.setContentsMargins(4, 8, 4, 4)
        self._separate_radio = QRadioButton("Separate (per file)")
        self._separate_radio.setChecked(True)
        self._separate_radio.setToolTip(
            "Fit each file independently in parallel.\n"
            "Each run gets its own parameter set.")
        self._simultaneous_radio = QRadioButton("Simultaneous")
        self._simultaneous_radio.setToolTip(
            "Fit all files at once with a single fitter.\n"
            "Parameters can be linked across runs.")
        self._fit_mode_group = QButtonGroup(self)
        self._fit_mode_group.addButton(self._separate_radio)
        self._fit_mode_group.addButton(self._simultaneous_radio)
        mode_lay.addWidget(self._separate_radio)
        mode_lay.addWidget(self._simultaneous_radio)
        layout.addWidget(mode_grp)

        # ── Method & Statistics ──
        method_grp = QGroupBox("Method & Statistics")
        method_form = QFormLayout(method_grp)
        method_form.setContentsMargins(4, 8, 4, 4)

        self._method_combo = QComboBox()
        self._method_combo.setToolTip(
            "Minimisation algorithm:\n"
            "leastsq = Levenberg\u2013Marquardt (fast, gradient-based)\n"
            "least_squares = Trust Region Reflective (bounded)\n"
            "slsqp = Sequential Least Squares (constrained)\n"
            "emcee = MCMC sampling (Bayesian uncertainties)\n"
            "nelder = Nelder\u2013Mead simplex (derivative-free)\n"
            "powell = Powell's method (derivative-free)\n"
            "cobyla = Constrained optimisation (derivative-free)")
        self._method_combo.addItems([
            "leastsq", "least_squares", "slsqp", "emcee",
            "nelder", "powell", "cobyla",
        ])
        method_form.addRow("Method:", self._method_combo)

        self._stats_combo = QComboBox()
        self._stats_combo.setToolTip(
            "Cost function for the fit:\n"
            "Chi-square = weighted least squares (\u03a3(y\u2212f)\u00b2/\u03c3\u00b2)\n"
            "Gaussian LLH = Gaussian log-likelihood\n"
            "Poisson LLH = Poisson log-likelihood (best for low counts)")
        self._stats_combo.addItems([
            "Chi-square", "Gaussian LLH", "Poisson LLH",
        ])
        # Seed method/statistics from the app Settings ("Fitting Defaults");
        # these were saved but never read (code review 2026-06-02,
        # settings-fitting-defaults-write-only). from_dict() overrides for a
        # loaded project; setCurrentText no-ops an unknown value.
        _fd = _load_settings()
        self._method_combo.setCurrentText(_fd.get("default_method", "leastsq"))
        self._stats_combo.setCurrentText(
            _fd.get("default_statistics", "Chi-square"))
        method_form.addRow("Statistics:", self._stats_combo)

        self._scale_covar = QCheckBox("Scale covariance")
        # Default OFF: for counting data with absolute Poisson errors (the
        # usual case) scaling uncertainties by sqrt(reduced-chi-square)
        # double-counts the data's own noise estimate, and it disagrees with
        # the LLH path and the auto-fitter, which both force it off
        # (code review 2026-06-02, scale-covar-true-poisson-inflation).
        self._scale_covar.setChecked(False)
        self._scale_covar.setToolTip(
            "Scale parameter uncertainties by sqrt(reduced chi-square).\n"
            "Leave UNCHECKED for counting data with absolute Poisson errors\n"
            "(the usual case) -- scaling there double-counts the data's own\n"
            "noise. Likelihood fits force it off regardless.")
        method_form.addRow(self._scale_covar)
        layout.addWidget(method_grp)

        # ── Parameter Sharing (simultaneous mode) ──
        # Bare-parameter sharing across sources/models. shareModelParams
        # in satlas2 takes a bare name and shares it across all models
        # exposing that param.
        self._sharing_grp = QGroupBox("Parameter Sharing")
        self._sharing_grp.setVisible(False)
        self._sharing_grp.setToolTip(_SHARING_GRP_TOOLTIP)
        sharing_lay = QVBoxLayout(self._sharing_grp)
        sharing_lay.setContentsMargins(4, 8, 4, 4)
        self._sharing_table = QTableWidget()
        self._sharing_table.setColumnCount(3)
        self._sharing_table.setHorizontalHeaderLabels(
            ["Parameter", "Share", "Mode"])
        self._sharing_table.horizontalHeader().setStretchLastSection(True)
        self._sharing_table.verticalHeader().setVisible(False)
        self._sharing_table.setAlternatingRowColors(True)
        # Header tooltips: hover the column name to see what it means.
        self._sharing_table.horizontalHeaderItem(0).setToolTip(
            "Bare parameter name (Al, centroid, FWHMG, ...).\n"
            "One row per unique parameter across enabled models.")
        self._sharing_table.horizontalHeaderItem(1).setToolTip(
            _SHARING_CHECKBOX_TOOLTIP)
        self._sharing_table.horizontalHeaderItem(2).setToolTip(
            _SHARING_MODE_TOOLTIP)
        # Show ~7 rows by default; scroll within the table for more.
        self._sharing_table.setMinimumHeight(220)
        sharing_lay.addWidget(self._sharing_table)
        layout.addWidget(self._sharing_grp)

        # ── Advanced Constraints (simultaneous mode) ──
        # Full-name expression overrides for cross-run / cross-model
        # constraints that the bare-name sharing above can't express.
        # Target column displays a readable label; the underlying full
        # lmfit name (Run_<id>___<safe_model>___<param>) is stored in
        # the item's UserRole.
        self._expressions_grp = QGroupBox("Advanced Constraints")
        self._expressions_grp.setVisible(False)
        self._expressions_grp.setToolTip(_CONSTRAINTS_GRP_TOOLTIP)
        expr_lay = QVBoxLayout(self._expressions_grp)
        expr_lay.setContentsMargins(4, 8, 4, 4)
        self._expressions_table = QTableWidget()
        self._expressions_table.setColumnCount(2)
        self._expressions_table.setHorizontalHeaderLabels(
            ["Target", "Expression"])
        self._expressions_table.horizontalHeader().setStretchLastSection(True)
        self._expressions_table.verticalHeader().setVisible(False)
        self._expressions_table.setAlternatingRowColors(True)
        self._expressions_table.horizontalHeaderItem(0).setToolTip(
            "The parameter you want to constrain, shown as\n"
            "<source> / <model> / <param>. Read-only; one row\n"
            "per (source, model, param) combination in your fit.")
        self._expressions_table.horizontalHeaderItem(1).setToolTip(
            _FITTER_EXPR_TOOLTIP_OK)
        # Show ~10 rows by default; this table can hold many more
        # (one row per source × model × param). Wider Target column
        # so the readable label "Run_<id> / <Model> / <param>" fits.
        self._expressions_table.setMinimumHeight(300)
        self._expressions_table.setColumnWidth(0, 230)
        self._expressions_table.itemChanged.connect(
            self._on_expression_item_changed)
        expr_lay.addWidget(self._expressions_table)
        layout.addWidget(self._expressions_grp)

        # Full registry (set of valid Run_X___Model___param names) used
        # for live validation. Populated by update_expressions_table.
        self._full_registry: set[str] = set()

        # ── Common Grid (simultaneous mode) ──
        self._common_grid_grp = QGroupBox("Common Grid (simultaneous)")
        self._common_grid_grp.setVisible(False)
        cg_lay = QVBoxLayout(self._common_grid_grp)
        cg_lay.setContentsMargins(4, 8, 4, 4)
        cg_lay.setSpacing(2)

        self._common_grid_enable = QCheckBox(
            "Re-bin all sources onto one shared grid")
        self._common_grid_enable.setToolTip(
            "Default (off): each run is binned independently and "
            "satlas2's\n"
            "simultaneous fit sees runs with possibly different bin "
            "widths.\n"
            "Bins from runs with finer resolution carry more statistical\n"
            "weight per MHz than coarser runs.\n\n"
            "On: after per-run binning, every source's data is re-binned\n"
            "onto a common x grid (union of x ranges, width below or\n"
            "auto-derived). This balances statistical weight per bin\n"
            "across runs but discards a bit of resolution for runs that\n"
            "were originally finer than the common grid.\n\n"
            "Requires all sources to share the same bin mode (all\n"
            "Frequency or all Raw Voltage). Mixed modes fall back to\n"
            "independent grids with a warning.")
        cg_lay.addWidget(self._common_grid_enable)

        cg_form = QFormLayout()
        cg_form.setContentsMargins(0, 0, 0, 0)
        self._common_grid_width = _make_double(
            0.0, 0.0, 1e6, 4, 1.0,
            tooltip=("Common-grid bin width (MHz for Frequency mode, V "
                     "for Raw Voltage mode).\n0 = auto: median of per-run "
                     "dx values."))
        cg_form.addRow("Width [MHz / V]:", self._common_grid_width)
        cg_lay.addLayout(cg_form)
        self._common_grid_enable.toggled.connect(
            self._common_grid_width.setEnabled)
        self._common_grid_width.setEnabled(False)

        layout.addWidget(self._common_grid_grp)

        self._simultaneous_radio.toggled.connect(self._sharing_grp.setVisible)
        self._simultaneous_radio.toggled.connect(
            self._expressions_grp.setVisible)
        self._simultaneous_radio.toggled.connect(
            self._common_grid_grp.setVisible)

        # ── Priors ──
        self._priors_toggle = QCheckBox("Show Priors")
        self._priors_toggle.setToolTip(
            "Add Gaussian priors to constrain parameters.\n"
            "Each prior adds a penalty term: ((value \u2212 prior) / \u03c3)\u00b2")
        self._priors_toggle.toggled.connect(lambda on: self._priors_grp.setVisible(on))
        layout.addWidget(self._priors_toggle)

        self._priors_grp = QGroupBox("Gaussian Priors")
        self._priors_grp.setVisible(False)
        priors_lay = QVBoxLayout(self._priors_grp)
        priors_lay.setContentsMargins(4, 8, 4, 4)

        self._priors_table = QTableWidget()
        self._priors_table.setColumnCount(3)
        self._priors_table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Uncertainty"])
        # code review 2026-06-02, prior-param-name-silent-drop: surface the
        # required full-name format on the column header too.
        _param_hdr = self._priors_table.horizontalHeaderItem(0)
        if _param_hdr is not None:
            _param_hdr.setToolTip(
                "Full parameter name: Run_<id>___<Model>___<param>\n"
                "Example: Run_7164___HFS_1___centroid")
        self._priors_table.horizontalHeader().setStretchLastSection(True)
        self._priors_table.verticalHeader().setVisible(False)
        priors_lay.addWidget(self._priors_table)

        add_prior_btn = QPushButton("+ Add Prior")
        add_prior_btn.setToolTip("Add a Gaussian prior constraint row")
        add_prior_btn.clicked.connect(self._add_prior_row)
        priors_lay.addWidget(add_prior_btn)
        layout.addWidget(self._priors_grp)

        # ── MCMC Settings ──
        self._mcmc_toggle = QCheckBox("MCMC Settings")
        self._mcmc_toggle.setToolTip(
            "Show emcee MCMC sampler configuration.\n"
            "Only used when Method = emcee.")
        self._mcmc_toggle.toggled.connect(lambda on: self._mcmc_grp.setVisible(on))
        layout.addWidget(self._mcmc_toggle)

        self._mcmc_grp = QGroupBox("MCMC (emcee)")
        self._mcmc_grp.setVisible(False)
        mcmc_form = QFormLayout(self._mcmc_grp)
        mcmc_form.setContentsMargins(4, 8, 4, 4)

        self._nwalkers = _make_int(50, 2, 10000, tooltip="Number of walkers")
        mcmc_form.addRow("Walkers:", self._nwalkers)
        self._nsteps = _make_int(1000, 10, 1000000, tooltip="MCMC steps")
        mcmc_form.addRow("Steps:", self._nsteps)
        self._burn = _make_int(
            0, 0, 1000000,
            tooltip="Burn-in: discard the first N steps of every walker\n"
                    "before computing parameter values/uncertainties, so\n"
                    "the walkers' initial transient (before they reach the\n"
                    "stationary posterior) doesn't bias the result.\n"
                    "A common choice is ~2-3x the autocorrelation time.\n"
                    "0 = keep all steps.")
        mcmc_form.addRow("Burn-in:", self._burn)
        self._thin = _make_int(
            1, 1, 100000,
            tooltip="Thinning: after burn-in, keep only every Nth step to\n"
                    "reduce autocorrelation between the retained samples.\n"
                    "1 = keep every step.")
        mcmc_form.addRow("Thin:", self._thin)
        self._convergence = QCheckBox("Auto-convergence")
        self._convergence.setToolTip(
            "Stop early if the chain has converged.\n"
            "Checks autocorrelation time every N iterations.")
        mcmc_form.addRow(self._convergence)
        self._conv_iter = _make_int(50, 1, 1000,
                                    tooltip="Check convergence every N iterations")
        mcmc_form.addRow("Conv. iter:", self._conv_iter)
        self._conv_tau = _make_double(0.05, 0.001, 1.0, 3, 0.01,
                                      tooltip="Convergence threshold: stop when\n"
                                      "relative change in autocorrelation time < tau")
        mcmc_form.addRow("Conv. tau:", self._conv_tau)
        layout.addWidget(self._mcmc_grp)

        # ── Progress & Run ──
        run_grp = QGroupBox("Execution")
        run_lay = QVBoxLayout(run_grp)
        run_lay.setContentsMargins(4, 8, 4, 4)

        from gui.shared_widgets import lucide_icon
        from PySide6.QtWidgets import QSizePolicy

        # Row 1: Run Fit spans the whole width (the primary action).
        self._run_btn = QPushButton("\u25b6 Run Fit")
        self._run_btn.setStyleSheet("font-weight: bold;")
        self._run_btn.setToolTip("Start fitting with current settings")
        self._run_btn.setMinimumHeight(30)
        self._run_btn.clicked.connect(self.fit_requested.emit)
        run_lay.addWidget(self._run_btn)

        # Row 2: the secondary actions share the width equally. Each is
        # set to expand so its label always fits (no clipping) and the
        # three stay the same size.
        btn_row = QHBoxLayout()
        self._auto_btn = QPushButton(lucide_icon("crosshair"),
                                     "Find Parameters")
        self._auto_btn.setToolTip(
            "Open the Auto-Fitter: a multi-start parallel leastsq sweep.\n"
            "Many perturbed copies of the current parameters are fitted\n"
            "in parallel and ranked by chi-square, with a live preview.\n\n"
            "Accept the best result to seed new initial guesses, then\n"
            "run the real fit.")
        self._auto_btn.clicked.connect(self.auto_fit_requested.emit)
        self._stop_btn = QPushButton("\u25a0 Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip("Cancel the running fit")
        self._stop_btn.clicked.connect(self.fit_cancel_requested.emit)
        self._revert_btn = QPushButton("\u21b6 Revert")
        self._revert_btn.setToolTip(
            "Restore model parameters to pre-fit values")
        self._revert_btn.clicked.connect(self.revert_requested.emit)
        for _b in (self._auto_btn, self._stop_btn, self._revert_btn):
            _b.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Fixed)
            _b.setMinimumHeight(28)
            btn_row.addWidget(_b)
        run_lay.addLayout(btn_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        run_lay.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: gray;")
        run_lay.addWidget(self._status_label)
        layout.addWidget(run_grp)

        layout.addStretch()

    def _add_prior_row(self):
        row = self._priors_table.rowCount()
        self._priors_table.insertRow(row)
        # code review 2026-06-02, prior-param-name-silent-drop: document the
        # required full-name format; names that don't match an exact
        # Run_<id>___<Model>___<param> are silently ignored by the fitter.
        name_item = QTableWidgetItem("")
        name_item.setToolTip(
            "Full parameter name: Run_<id>___<Model>___<param>\n"
            "Example: Run_7164___HFS_1___centroid\n"
            "Names that don't match an existing parameter exactly are ignored.")
        self._priors_table.setItem(row, 0, name_item)
        val_spin = _make_analysis_spin(0.0, -1e12, 1e12, 4, 1.0)
        self._priors_table.setCellWidget(row, 1, val_spin)
        unc_spin = _make_analysis_spin(1.0, 0.0, 1e12, 4, 0.1)
        self._priors_table.setCellWidget(row, 2, unc_spin)

    def get_fitter_config(self):
        stats_map = {
            "Chi-square": {"llh": False, "llh_method": ""},
            "Gaussian LLH": {"llh": True, "llh_method": "gaussian"},
            "Poisson LLH": {"llh": True, "llh_method": "poisson"},
        }
        stats = stats_map.get(self._stats_combo.currentText(),
                              {"llh": False, "llh_method": ""})
        config = {
            "separate": self._separate_radio.isChecked(),
            "method": self._method_combo.currentText(),
            "llh": stats["llh"],
            "llh_method": stats["llh_method"],
            "scale_covar": self._scale_covar.isChecked(),
            "common_grid": self._common_grid_enable.isChecked(),
            "common_grid_width": self._common_grid_width.value(),
        }
        if self._method_combo.currentText() == "emcee":
            config["nwalkers"] = self._nwalkers.value()
            config["steps"] = self._nsteps.value()
            config["burn"] = self._burn.value()
            config["thin"] = self._thin.value()
            config["convergence"] = self._convergence.isChecked()
            config["convergence_iter"] = self._conv_iter.value()
            config["convergence_tau"] = self._conv_tau.value()

        # Sharing (for simultaneous): bare param names.
        config["shared_params"] = []       # shareModelParams
        config["shared_all_params"] = []   # shareParams (across ALL)
        for i in range(self._sharing_table.rowCount()):
            name_item = self._sharing_table.item(i, 0)
            share_w = self._sharing_table.cellWidget(i, 1)
            mode_w = self._sharing_table.cellWidget(i, 2)
            if not (name_item and share_w):
                continue
            share_cb = share_w.findChild(QCheckBox)
            if not (share_cb and share_cb.isChecked()):
                continue
            mode = mode_w.currentText() if mode_w else "Model"
            if mode == "All":
                config["shared_all_params"].append(name_item.text())
            else:
                config["shared_params"].append(name_item.text())

        # Expressions (for simultaneous): full lmfit names as keys.
        # Target full name lives in UserRole on column 0; column 0
        # text is just the readable display label.
        config["expressions"] = {}
        for i in range(self._expressions_table.rowCount()):
            target_item = self._expressions_table.item(i, 0)
            expr_item = self._expressions_table.item(i, 1)
            if not (target_item and expr_item):
                continue
            full_name = target_item.data(Qt.ItemDataRole.UserRole)
            text = expr_item.text().strip()
            if full_name and text:
                config["expressions"][full_name] = text

        # Priors
        config["priors"] = []
        for i in range(self._priors_table.rowCount()):
            name_item = self._priors_table.item(i, 0)
            val_w = self._priors_table.cellWidget(i, 1)
            unc_w = self._priors_table.cellWidget(i, 2)
            if name_item and name_item.text().strip() and val_w and unc_w:
                config["priors"].append({
                    "param": name_item.text(),
                    "value": val_w.value(),
                    "uncertainty": unc_w.value(),
                })
        return config

    def update_sharing_table(self, bare_param_names):
        """Populate the sharing table with bare parameter names.

        Bare names (e.g. ``Al``, ``centroid``) feed
        ``shareModelParams`` / ``shareParams`` directly. The user
        toggles per-row whether to share the param and chooses the
        mode (``Model`` for shareModelParams, ``All`` for shareParams).
        """
        new_names = list(bare_param_names)
        current = [
            (self._sharing_table.item(i, 0).text()
             if self._sharing_table.item(i, 0) is not None else "")
            for i in range(self._sharing_table.rowCount())
        ]
        if new_names == current \
                and not getattr(self, "_pending_sharing", None):
            return

        # Snapshot existing per-name {shared, mode} BEFORE rebuilding
        # so user-ticked rows survive when the bare-param set changes
        # (model added/removed/renamed, Piecewise param config
        # changed). Keys that disappear from the new name list are
        # silently dropped.
        existing: dict[str, dict] = {}
        for i in range(self._sharing_table.rowCount()):
            name_item = self._sharing_table.item(i, 0)
            share_w = self._sharing_table.cellWidget(i, 1)
            mode_w = self._sharing_table.cellWidget(i, 2)
            if not name_item:
                continue
            cb = share_w.findChild(QCheckBox) if share_w else None
            existing[name_item.text()] = {
                "shared": cb.isChecked() if cb else False,
                "mode": mode_w.currentText() if mode_w else "Model",
            }

        self._sharing_table.setRowCount(len(new_names))
        for i, name in enumerate(new_names):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(
                name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._sharing_table.setItem(i, 0, name_item)

            share_cb = QCheckBox()
            share_cb.setToolTip(_SHARING_CHECKBOX_TOOLTIP)
            share_w = QWidget()
            share_lay = QHBoxLayout(share_w)
            share_lay.setContentsMargins(0, 0, 0, 0)
            share_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            share_lay.addWidget(share_cb)
            self._sharing_table.setCellWidget(i, 1, share_w)

            mode_combo = QComboBox()
            mode_combo.addItems(["Model", "All"])
            mode_combo.setToolTip(_SHARING_MODE_TOOLTIP)
            self._sharing_table.setCellWidget(i, 2, mode_combo)

        # Merge: restore (a) existing in-flight user toggles snapshot
        # taken above, then (b) pending state from from_dict. The
        # from_dict pending wins on conflict because it represents
        # the most recent intent (project just opened) -- but in
        # practice `existing` is empty in the from_dict path, so
        # order only matters when the user does both at once.
        merged: dict[str, dict] = dict(existing)
        for s in (getattr(self, "_pending_sharing", None) or []):
            if "param" in s:
                merged[s["param"]] = {
                    "shared": s.get("shared", False),
                    "mode": s.get("mode", "Model"),
                }
        if merged:
            for i in range(self._sharing_table.rowCount()):
                name_item = self._sharing_table.item(i, 0)
                if not name_item:
                    continue
                s = merged.get(name_item.text())
                if not s:
                    continue
                share_w = self._sharing_table.cellWidget(i, 1)
                if share_w:
                    cb = share_w.findChild(QCheckBox)
                    if cb:
                        cb.setChecked(s.get("shared", False))
                mode_w = self._sharing_table.cellWidget(i, 2)
                if mode_w:
                    idx = mode_w.findText(s.get("mode", "Model"))
                    if idx >= 0:
                        mode_w.setCurrentIndex(idx)
            self._pending_sharing = []

    def update_expressions_table(self, source_names, model_configs):
        """Populate the Advanced Constraints table with one row per
        (source, model, param) target.

        Display column shows ``Run_<id> / <Model> / <param>``; the
        full lmfit name is stored in the item's UserRole. Pre-fit
        validator and live-cell validation both use this UserRole as
        the authoritative target.
        """
        targets = []  # (display_label, full_name)
        for src in source_names:
            for mc in model_configs:
                model_name = mc.get("name", "")
                for pname in mc.get("params", {}):
                    if pname in _NON_FIT_PARAMS_FOR_VALIDATION:
                        continue
                    full = full_param_name(src, model_name, pname)
                    label = full.replace("___", " / ")
                    targets.append((label, full))

        # Refresh full registry used by live-cell validation
        self._full_registry = {full for _, full in targets}

        # Snapshot every non-empty user-typed expression keyed by full
        # target name BEFORE rebuilding rows, so source/model edits
        # don't silently discard the user's work.
        existing: dict[str, str] = {}
        for i in range(self._expressions_table.rowCount()):
            target_item = self._expressions_table.item(i, 0)
            expr_item = self._expressions_table.item(i, 1)
            if not (target_item and expr_item):
                continue
            full = target_item.data(Qt.ItemDataRole.UserRole)
            text = expr_item.text().strip()
            if full and text:
                existing[full] = expr_item.text()

        # No-op when targets are unchanged AND no pending restore.
        # (Existing user expressions are preserved by virtue of not
        # rebuilding the rows.)
        current_full = []
        for i in range(self._expressions_table.rowCount()):
            target_item = self._expressions_table.item(i, 0)
            current_full.append(
                target_item.data(Qt.ItemDataRole.UserRole)
                if target_item else None)
        new_full = [full for _, full in targets]
        if new_full == current_full \
                and not getattr(self, "_pending_expressions", None):
            self._validate_all_expression_cells()
            return

        self._expressions_table.blockSignals(True)
        try:
            self._expressions_table.setRowCount(len(targets))
            for i, (label, full) in enumerate(targets):
                target_item = QTableWidgetItem(label)
                target_item.setFlags(
                    target_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                target_item.setData(Qt.ItemDataRole.UserRole, full)
                self._expressions_table.setItem(i, 0, target_item)
                self._expressions_table.setItem(i, 1, QTableWidgetItem(""))
        finally:
            self._expressions_table.blockSignals(False)

        # Restore (a) existing user-typed text the snapshot captured,
        # then (b) pending expressions loaded from from_dict.
        # from_dict pending wins on conflict because that's the most
        # recent intent (project just opened) -- but in practice
        # `existing` will be empty in the from_dict path, so order
        # only matters when the user does both at once.
        merged = dict(existing)
        merged.update(getattr(self, "_pending_expressions", None) or {})
        if merged:
            for i in range(self._expressions_table.rowCount()):
                t = self._expressions_table.item(i, 0)
                if not t:
                    continue
                full = t.data(Qt.ItemDataRole.UserRole)
                if full in merged:
                    self._expressions_table.item(i, 1).setText(
                        merged[full])
            self._pending_expressions = {}

        self._validate_all_expression_cells()

    def _on_expression_item_changed(self, item):
        """Live-validate the Expression cell as the user edits."""
        if item.column() != 1:
            return
        self._validate_expression_cell(item)

    def _validate_expression_cell(self, item):
        """Color one Advanced Constraints expression cell."""
        expr = item.text()
        issue = validate_single_expression(
            expr, "fitter", self._full_registry)
        self._expressions_table.blockSignals(True)
        try:
            if issue is None:
                item.setBackground(QBrush())
                item.setToolTip(_FITTER_EXPR_TOOLTIP_OK)
            else:
                item.setBackground(_EXPR_ERROR_BG)
                item.setToolTip(issue.message)
        finally:
            self._expressions_table.blockSignals(False)

    def _validate_all_expression_cells(self):
        for i in range(self._expressions_table.rowCount()):
            it = self._expressions_table.item(i, 1)
            if it is not None:
                self._validate_expression_cell(it)

    def set_progress(self, current, total, status_text=""):
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
        self._status_label.setText(status_text or f"{current}/{total}")

    def set_running(self, running):
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        if not running:
            self._status_label.setText("Ready")

    # ── Serialization ──

    def to_dict(self):
        d = super().to_dict()
        d["separate"] = self._separate_radio.isChecked()
        d["method"] = self._method_combo.currentText()
        d["statistics"] = self._stats_combo.currentText()
        d["scale_covar"] = self._scale_covar.isChecked()
        d["mcmc"] = {
            "nwalkers": self._nwalkers.value(),
            "steps": self._nsteps.value(),
            "burn": self._burn.value(),
            "thin": self._thin.value(),
            "convergence": self._convergence.isChecked(),
            "conv_iter": self._conv_iter.value(),
            "conv_tau": self._conv_tau.value(),
        }
        # Save priors
        priors = []
        for i in range(self._priors_table.rowCount()):
            name_item = self._priors_table.item(i, 0)
            val_w = self._priors_table.cellWidget(i, 1)
            unc_w = self._priors_table.cellWidget(i, 2)
            if name_item and val_w and unc_w:
                priors.append({"param": name_item.text(),
                               "value": val_w.value(),
                               "uncertainty": unc_w.value()})
        d["priors"] = priors
        # Sharing table state (bare param names).
        sharing = []
        for i in range(self._sharing_table.rowCount()):
            name_item = self._sharing_table.item(i, 0)
            share_w = self._sharing_table.cellWidget(i, 1)
            mode_w = self._sharing_table.cellWidget(i, 2)
            if not name_item:
                continue
            share_cb = share_w.findChild(QCheckBox) if share_w else None
            sharing.append({
                "param": name_item.text(),
                "shared": share_cb.isChecked() if share_cb else False,
                "mode": mode_w.currentText() if mode_w else "Model",
            })
        d["sharing"] = sharing

        # Advanced Constraints table state (full lmfit names as keys).
        expressions = {}
        for i in range(self._expressions_table.rowCount()):
            target_item = self._expressions_table.item(i, 0)
            expr_item = self._expressions_table.item(i, 1)
            if not (target_item and expr_item):
                continue
            full = target_item.data(Qt.ItemDataRole.UserRole)
            text = expr_item.text().strip()
            if full and text:
                expressions[full] = text
        d["expressions"] = expressions
        # Visibility toggles
        d["mcmc_toggle"] = self._mcmc_toggle.isChecked()
        d["priors_toggle"] = self._priors_toggle.isChecked()
        # Common-grid simultaneous mode
        d["common_grid"] = self._common_grid_enable.isChecked()
        d["common_grid_width"] = self._common_grid_width.value()
        return d

    def from_dict(self, d):
        super().from_dict(d)
        self._separate_radio.setChecked(d.get("separate", True))
        self._simultaneous_radio.setChecked(not d.get("separate", True))
        idx = self._method_combo.findText(d.get("method", "leastsq"))
        if idx >= 0:
            self._method_combo.setCurrentIndex(idx)
        idx = self._stats_combo.findText(d.get("statistics", "Chi-square"))
        if idx >= 0:
            self._stats_combo.setCurrentIndex(idx)
        self._scale_covar.setChecked(d.get("scale_covar", False))
        mcmc = d.get("mcmc", {})
        self._nwalkers.setValue(mcmc.get("nwalkers", 50))
        self._nsteps.setValue(mcmc.get("steps", 1000))
        self._burn.setValue(int(mcmc.get("burn", 0)))
        self._thin.setValue(int(mcmc.get("thin", 1)))
        self._convergence.setChecked(mcmc.get("convergence", False))
        self._conv_iter.setValue(mcmc.get("conv_iter", 50))
        self._conv_tau.setValue(mcmc.get("conv_tau", 0.05))
        for prior in d.get("priors", []):
            self._add_prior_row()
            row = self._priors_table.rowCount() - 1
            self._priors_table.item(row, 0).setText(prior.get("param", ""))
            self._priors_table.cellWidget(row, 1).setValue(prior.get("value", 0))
            self._priors_table.cellWidget(row, 2).setValue(
                prior.get("uncertainty", 1))
        # Restore sharing/expressions state (applied later when the
        # tables get populated by update_*_table). New schema uses
        # "sharing" + "expressions"; the old "linking" schema mixed
        # both with bare-name keys, which never functioned at fit
        # time -- we migrate the share parts (bare param after the
        # first colon if present) and drop expressions.
        sharing = d.get("sharing")
        expressions = d.get("expressions")
        if sharing is None and expressions is None and "linking" in d:
            legacy = d.get("linking", [])
            migrated_sharing = {}
            for lk in legacy:
                raw = (lk.get("param") or "").strip()
                if not raw:
                    continue
                # Old GUI stored "Model_1:Al"; take the bare suffix.
                bare = raw.split(":")[-1].strip() or raw
                if bare in migrated_sharing:
                    continue
                migrated_sharing[bare] = {
                    "param": bare,
                    "shared": bool(lk.get("shared", False)),
                    "mode": lk.get("mode", "Model"),
                }
            self._pending_sharing = list(migrated_sharing.values())
            self._pending_expressions = {}
        else:
            self._pending_sharing = list(sharing or [])
            self._pending_expressions = dict(expressions or {})
        # Visibility toggles
        self._mcmc_toggle.setChecked(d.get("mcmc_toggle", False))
        self._priors_toggle.setChecked(d.get("priors_toggle", False))
        # Common-grid simultaneous mode
        self._common_grid_enable.setChecked(d.get("common_grid", False))
        self._common_grid_width.setValue(float(d.get("common_grid_width",
                                                      0.0)))


# ══════════════════════════════════════════════════════════════════
#  Output Block
# ══════════════════════════════════════════════════════════════════

class OutputBlock(AnalysisBlock):
    """Output block: report, plots, tracking configuration."""
    BLOCK_TYPE = "Output"
    BLOCK_COLOR = "#9C27B0"
    # Default width for newly-added Output blocks. The width is
    # persisted per project, so existing saved Output blocks keep their
    # stored width on load and are unaffected by changes here.
    BLOCK_WIDTH = 387

    # Emitted by the "Re-apply outputs" button: regenerate the last fit's
    # output files (e.g. newly-ticked ToF plots) into the SAME iteration
    # directory, without re-running the fit.
    reapply_requested = Signal()

    def __init__(self, name="Output_1", parent=None):
        super().__init__(name, parent)
        layout = self._content_layout

        # ── Reports ──
        rep_grp = QGroupBox("Reports")
        rep_lay = QVBoxLayout(rep_grp)
        rep_lay.setContentsMargins(4, 8, 4, 4)
        self._report_check = QCheckBox("Fit report (text)")
        self._report_check.setChecked(True)
        self._report_check.setToolTip(
            "Save a text fit report with parameter values,\n"
            "uncertainties, and fit statistics")
        rep_lay.addWidget(self._report_check)

        corr_row = QHBoxLayout()
        self._show_correl = QCheckBox("Show correlations")
        # Default from the app Settings ("Fitting Defaults"); from_dict()
        # overrides for a loaded project (code review 2026-06-02,
        # settings-fitting-defaults-write-only).
        self._show_correl.setChecked(
            bool(_load_settings().get("show_correlations", True)))
        self._show_correl.setToolTip(
            "Include parameter correlations in the fit report")
        corr_row.addWidget(self._show_correl)
        corr_row.addWidget(QLabel("min:"))
        self._min_correl = _make_double(0.1, 0, 1, 2, 0.05,
                                        tooltip="Minimum correlation to display.\n"
                                        "Pairs below this threshold are hidden.")
        self._min_correl.setFixedWidth(70)
        corr_row.addWidget(self._min_correl)
        corr_row.addStretch()
        rep_lay.addLayout(corr_row)

        self._params_csv = QCheckBox("Parameters CSV")
        self._params_csv.setChecked(True)
        self._params_csv.setToolTip(
            "Save fitted parameters to a CSV table")
        rep_lay.addWidget(self._params_csv)
        self._metadata_csv = QCheckBox("Metadata CSV")
        self._metadata_csv.setChecked(True)
        self._metadata_csv.setToolTip(
            "Save fit metadata (run number, chi-square, etc.) to CSV")
        rep_lay.addWidget(self._metadata_csv)
        layout.addWidget(rep_grp)

        # ── Fit Plots ──
        plot_grp = QGroupBox("Fit Plots")
        plot_lay = QVBoxLayout(plot_grp)
        plot_lay.setContentsMargins(4, 8, 4, 4)
        self._fit_plots = QCheckBox("Individual fit plots")
        self._fit_plots.setChecked(True)
        self._fit_plots.setToolTip("Save a fit plot (data + fit curve) per run")
        plot_lay.addWidget(self._fit_plots)
        # ToF output: a checkbox enables a TYPE dropdown that chooses what
        # the ToF figure contains.
        tof_row = QHBoxLayout()
        self._tof_plots = QCheckBox("ToF plots")
        self._tof_plots.setChecked(False)
        self._tof_plots.setToolTip(
            "Save a time-of-flight plot per run, with the applied ToF gate\n"
            "highlighted. Pick what the figure contains in the dropdown.")
        tof_row.addWidget(self._tof_plots)
        self._tof_plot_type = QComboBox()
        self._tof_plot_type.addItems(
            ["ToF only", "ToF + spectrum", "ToF + spectrum + fit"])
        self._tof_plot_type.setToolTip(
            "What the ToF output figure contains:\n"
            "• ToF only — ToF histogram + gate\n"
            "• ToF + spectrum — adds the gated spectrum panel below\n"
            "• ToF + spectrum + fit — also overlays the fitted model")
        self._tof_plot_type.setEnabled(False)
        self._tof_plots.toggled.connect(self._tof_plot_type.setEnabled)
        tof_row.addWidget(self._tof_plot_type, 1)
        plot_lay.addLayout(tof_row)

        self._residual_panel = QCheckBox("Residual panel")
        self._residual_panel.setChecked(True)
        self._residual_panel.setToolTip(
            "Add a normalised residual panel below the fit plot")
        plot_lay.addWidget(self._residual_panel)
        self._hide_zero_bins = QCheckBox("Hide zero-count bins")
        self._hide_zero_bins.setToolTip(
            "Filter bins with y ≤ 0 from the plotted data markers,\n"
            "error bars, and residuals. The fit itself is unaffected —\n"
            "only the rendered fit plot. Useful when a scan covers a\n"
            "wide range with empty regions between peaks.")
        plot_lay.addWidget(self._hide_zero_bins)
        self._component_overlay = QCheckBox("Component overlay")
        self._component_overlay.setToolTip(
            "Overlay individual model components on the fit plot\n"
            "(e.g. separate HFS peaks)")
        self._component_overlay.toggled.connect(self._toggle_component_panel)
        plot_lay.addWidget(self._component_overlay)

        # Sub-checkboxes for individual components
        self._component_panel = QWidget()
        comp_panel_lay = QVBoxLayout(self._component_panel)
        comp_panel_lay.setContentsMargins(20, 0, 0, 0)
        comp_panel_lay.setSpacing(2)
        self._component_panel.setVisible(False)
        self._component_checks = {}  # {component_name: QCheckBox}
        plot_lay.addWidget(self._component_panel)

        # On-plot fit-values box: tick which fitted parameters get
        # printed as "Name = value ± err" in a corner box on each fit
        # plot (font size lives in Plot Options ▸ Fit Plot).
        self._values_on_plot = QCheckBox("Fit values on plot")
        self._values_on_plot.setToolTip(
            "Print selected fitted parameter values (value ± error) in "
            "a box on each fit plot.\nPick the parameters in the tree "
            "below; box font size is in Plot Options ▸ Fit Plot.")
        plot_lay.addWidget(self._values_on_plot)
        self._values_param_tree = QTreeWidget()
        self._values_param_tree.setHeaderHidden(True)
        self._values_param_tree.setRootIsDecorated(True)
        self._values_param_tree.setMinimumHeight(110)
        self._values_param_tree.setMaximumHeight(170)
        self._values_param_tree.setVisible(False)
        self._values_on_plot.toggled.connect(
            self._values_param_tree.setVisible)
        plot_lay.addWidget(self._values_param_tree)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self._plot_format = QComboBox()
        self._plot_format.setToolTip("Image format for saved plots")
        self._plot_format.addItems(["png", "pdf", "svg"])
        self._plot_format.setFixedWidth(60)
        fmt_row.addWidget(self._plot_format)
        fmt_row.addWidget(QLabel("DPI:"))
        self._plot_dpi = _make_int(200, 72, 600,
                                   tooltip="Resolution (dots per inch) for saved plots.\n"
                                   "200 = good screen quality, 300+ for print.")
        self._plot_dpi.setFixedWidth(70)
        fmt_row.addWidget(self._plot_dpi)
        fmt_row.addStretch()
        plot_lay.addLayout(fmt_row)
        layout.addWidget(plot_grp)

        # Detailed appearance for EVERY generated plot type — global
        # fonts / major+minor ticks / line widths plus per-type marker
        # styles, colors, alphas, shading. Same persisted settings as
        # Settings ▸ Plot Defaults; applies to plots generated after OK.
        self._plot_options_btn = QPushButton("Plot Options…")
        self._plot_options_btn.setToolTip(
            "Detailed appearance settings for every output plot type "
            "(fit, tracker, correlation, walk, χ² map, isotope shifts): "
            "ticks, line widths, colors, marker styles, shading, "
            "sizes.\nShared with Settings ▸ Plot Defaults; applies to "
            "plots generated from now on.")
        self._plot_options_btn.clicked.connect(self._open_plot_options)
        layout.addWidget(self._plot_options_btn)

        # ── Tracker Plots ──
        track_grp = QGroupBox("Tracker Plots (Separate mode)")
        track_lay = QVBoxLayout(track_grp)
        track_lay.setContentsMargins(4, 8, 4, 4)

        xaxis_row = QHBoxLayout()
        xaxis_row.addWidget(QLabel("x-axis:"))
        self._tracker_xaxis = QComboBox()
        self._tracker_xaxis.setToolTip(
            "X-axis for tracker plots:\n"
            "Run number = sequential run index\n"
            "Start time = DAQ start timestamp")
        self._tracker_xaxis.addItems(["Run number", "Start time"])
        self._tracker_xaxis.setFixedWidth(100)
        xaxis_row.addWidget(self._tracker_xaxis)
        xaxis_row.addStretch()
        track_lay.addLayout(xaxis_row)

        track_lay.addWidget(QLabel("Parameters to track:"))
        self._tracker_tree = QTreeWidget()
        self._tracker_tree.setHeaderHidden(True)
        self._tracker_tree.setRootIsDecorated(True)
        self._tracker_tree.setMinimumHeight(220)
        track_lay.addWidget(self._tracker_tree)
        layout.addWidget(track_grp)

        # ── Diagnostics ──
        diag_grp = QGroupBox("Diagnostics")
        diag_lay = QVBoxLayout(diag_grp)
        diag_lay.setContentsMargins(4, 8, 4, 4)
        self._chisq_map = QCheckBox("Chi-square map")
        self._chisq_map.setToolTip(
            "Scan each parameter around its best-fit value\n"
            "and plot \u03c7\u00b2 vs parameter. Works with any method.")
        diag_lay.addWidget(self._chisq_map)
        self._correl_plot = QCheckBox("Correlation plot (MCMC)")
        self._correl_plot.setToolTip(
            "Corner/triangle plot showing 2D parameter correlations\n"
            "from the MCMC chain. Requires Method = emcee.")
        diag_lay.addWidget(self._correl_plot)
        self._walk_plot = QCheckBox("Walk plot (MCMC)")
        self._walk_plot.setToolTip(
            "Trace plot showing walker trajectories per parameter.\n"
            "Useful for checking MCMC convergence. Requires Method = emcee.")
        diag_lay.addWidget(self._walk_plot)
        cb_row = QHBoxLayout()
        self._confidence_bands = QCheckBox("Confidence bands (MCMC)")
        self._confidence_bands.setToolTip(
            "Overlay 1\u03c3 confidence band on fit plots.\n"
            "Uses a thinned chain for speed.")
        cb_row.addWidget(self._confidence_bands)
        cb_row.addWidget(QLabel("Samples:"))
        self._band_samples = _make_int(
            200, 50, 5000,
            tooltip="Number of chain samples for confidence band.\n"
                    "More = smoother bands but slower.\n"
                    "200 is usually sufficient.")
        self._band_samples.setFixedWidth(70)
        cb_row.addWidget(self._band_samples)
        cb_row.addStretch()
        diag_lay.addLayout(cb_row)

        diag_lay.addWidget(QLabel("Parameters to plot:"))
        self._diag_param_tree = QTreeWidget()
        self._diag_param_tree.setHeaderHidden(True)
        self._diag_param_tree.setRootIsDecorated(True)
        self._diag_param_tree.setMinimumHeight(220)
        self._diag_param_tree.setMaximumHeight(300)
        diag_lay.addWidget(self._diag_param_tree)
        layout.addWidget(diag_grp)

        # ── Iteration ──
        iter_grp = QGroupBox("Iteration")
        iter_form = QFormLayout(iter_grp)
        iter_form.setContentsMargins(4, 8, 4, 4)
        self._iter_mode = QComboBox()
        self._iter_mode.setToolTip(
            "Auto = auto-increment iteration number (iter_001, iter_002, ...)\n"
            "Manual = use a custom label for this iteration")
        self._iter_mode.addItems(["Auto", "Manual"])
        iter_form.addRow("Mode:", self._iter_mode)
        self._iter_label = QLineEdit()
        self._iter_label.setPlaceholderText("Manual label...")
        self._iter_label.setToolTip("Custom iteration folder name (Manual mode only)")
        self._iter_label.setEnabled(False)
        self._iter_mode.currentTextChanged.connect(
            lambda t: self._iter_label.setEnabled(t == "Manual"))
        iter_form.addRow("Label:", self._iter_label)
        layout.addWidget(iter_grp)

        # ── Re-apply ──
        # Regenerate the output files for the LAST fit's iteration using
        # the current toggles, without re-fitting. Use case: you ran the
        # fit, then realised you wanted (say) ToF plots -- tick them and
        # Re-apply, and the existing iteration is updated in place rather
        # than spawning a new one.
        self._reapply_btn = QPushButton("Update iteration")
        self._reapply_btn.setToolTip(
            "Forgot a plot? Tick it above, then Update iteration\n"
            "(formerly 'Re-apply outputs'): regenerates the LAST fit's\n"
            "iteration with the current output options, WITHOUT re-running\n"
            "the fit — files are added to the existing iteration folder\n"
            "instead of creating a new one. Newly-ticked MCMC walk /\n"
            "correlation plots are rebuilt from the saved chains; only the\n"
            "chi-square map and confidence bands need a true re-run.\n"
            "Requires a fit to have been run in this session.")
        self._reapply_btn.setMinimumHeight(28)
        self._reapply_btn.clicked.connect(self.reapply_requested.emit)
        layout.addWidget(self._reapply_btn)

        layout.addStretch()

    def _toggle_component_panel(self, checked):
        self._component_panel.setVisible(
            checked and bool(self._component_checks))

    def update_component_list(self, model_names):
        """Rebuild per-component checkboxes from current model block names.

        model_names: list of (block_name, model_type) tuples for enabled models.
        """
        import re
        prev_checked = {n for n, cb in self._component_checks.items()
                        if cb.isChecked()}
        # Also restore from pending (saved project load)
        pending = getattr(self, "_pending_components", None)
        if pending is not None and not prev_checked:
            prev_checked = set(pending)
            self._pending_components = None

        # Clear old checkboxes
        lay = self._component_panel.layout()
        for cb in self._component_checks.values():
            lay.removeWidget(cb)
            cb.deleteLater()
        self._component_checks = {}

        for block_name, model_type in model_names:
            safe = re.sub(r'[^A-Za-z0-9_]', '_', block_name)
            # Main model component
            cb = QCheckBox(safe)
            cb.setChecked(safe in prev_checked if prev_checked else True)
            lay.addWidget(cb)
            self._component_checks[safe] = cb
            # Background component (HFS, Voigt, etc.)
            if model_type in ("HFS", "Voigt", "Skewed Voigt",
                              "Exponential Decay"):
                bkg_name = f"{safe}_bkg"
                cb_b = QCheckBox(bkg_name)
                cb_b.setChecked(
                    bkg_name in prev_checked if prev_checked else True)
                lay.addWidget(cb_b)
                self._component_checks[bkg_name] = cb_b

        self._component_panel.setVisible(
            self._component_overlay.isChecked()
            and bool(self._component_checks))

    def get_selected_components(self):
        """Return set of component names the user wants to overlay."""
        return {n for n, cb in self._component_checks.items()
                if cb.isChecked()}

    def update_tracker_params(self, model_params_dict):
        """Update tracker tree and diagnostic param tree with parameters
        grouped by model.
        model_params_dict: {model_name: [param_name, ...]}
        """
        # Preserve existing check states
        def _get_checked(tree):
            checked = set()
            root = tree.invisibleRootItem()
            for i in range(root.childCount()):
                mi = root.child(i)
                for j in range(mi.childCount()):
                    c = mi.child(j)
                    if c.checkState(0) == Qt.CheckState.Checked:
                        checked.add(f"{mi.text(0)}:{c.text(0)}")
            return checked

        prev_track = _get_checked(self._tracker_tree)
        prev_diag = _get_checked(self._diag_param_tree)
        prev_values = _get_checked(self._values_param_tree)

        # Apply pending states from from_dict (saved project load)
        pending_tracked = getattr(self, "_pending_tracked", {})
        if pending_tracked and not prev_track:
            for mname, plist in pending_tracked.items():
                for p in plist:
                    prev_track.add(f"{mname}:{p}")
            self._pending_tracked = {}

        pending_diag = getattr(self, "_pending_diag", [])
        if pending_diag and not prev_diag:
            # diag_params is a flat list of param names; match any model
            for model_name, params in model_params_dict.items():
                for p in params:
                    if p in pending_diag:
                        prev_diag.add(f"{model_name}:{p}")
            self._pending_diag = []

        pending_values = getattr(self, "_pending_values", [])
        if pending_values and not prev_values:
            prev_values.update(pending_values)  # stored as Model:param
            self._pending_values = []

        self._tracker_tree.clear()
        self._diag_param_tree.clear()
        self._values_param_tree.clear()
        for model_name, params in model_params_dict.items():
            # Tracker tree
            model_item = QTreeWidgetItem(self._tracker_tree, [model_name])
            model_item.setFlags(model_item.flags()
                                | Qt.ItemFlag.ItemIsAutoTristate
                                | Qt.ItemFlag.ItemIsUserCheckable)
            for param in params:
                child = QTreeWidgetItem(model_item, [param])
                child.setFlags(child.flags()
                               | Qt.ItemFlag.ItemIsUserCheckable)
                key = f"{model_name}:{param}"
                child.setCheckState(
                    0, Qt.CheckState.Checked if key in prev_track
                    else Qt.CheckState.Unchecked)
            model_item.setExpanded(True)

            # Diagnostic param tree (default: all checked)
            diag_item = QTreeWidgetItem(
                self._diag_param_tree, [model_name])
            diag_item.setFlags(diag_item.flags()
                               | Qt.ItemFlag.ItemIsAutoTristate
                               | Qt.ItemFlag.ItemIsUserCheckable)
            for param in params:
                child = QTreeWidgetItem(diag_item, [param])
                child.setFlags(child.flags()
                               | Qt.ItemFlag.ItemIsUserCheckable)
                key = f"{model_name}:{param}"
                # Default all checked if no previous state
                if prev_diag:
                    child.setCheckState(
                        0, Qt.CheckState.Checked if key in prev_diag
                        else Qt.CheckState.Unchecked)
                else:
                    child.setCheckState(0, Qt.CheckState.Checked)
            diag_item.setExpanded(True)

            # Values-on-plot tree (default: none checked — the box
            # only shows what the user explicitly picked)
            values_item = QTreeWidgetItem(
                self._values_param_tree, [model_name])
            values_item.setFlags(values_item.flags()
                                 | Qt.ItemFlag.ItemIsAutoTristate
                                 | Qt.ItemFlag.ItemIsUserCheckable)
            for param in params:
                child = QTreeWidgetItem(values_item, [param])
                child.setFlags(child.flags()
                               | Qt.ItemFlag.ItemIsUserCheckable)
                key = f"{model_name}:{param}"
                child.setCheckState(
                    0, Qt.CheckState.Checked if key in prev_values
                    else Qt.CheckState.Unchecked)
            values_item.setExpanded(True)

    def get_tracked_params(self):
        """Return {model_name: [checked_param_names]}."""
        result = {}
        root = self._tracker_tree.invisibleRootItem()
        for i in range(root.childCount()):
            model_item = root.child(i)
            model_name = model_item.text(0)
            checked = []
            for j in range(model_item.childCount()):
                child = model_item.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    checked.append(child.text(0))
            if checked:
                result[model_name] = checked
        return result

    def get_diag_params(self):
        """Return flat list of checked diagnostic parameter names."""
        checked = []
        root = self._diag_param_tree.invisibleRootItem()
        for i in range(root.childCount()):
            model_item = root.child(i)
            for j in range(model_item.childCount()):
                child = model_item.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    checked.append(child.text(0))
        return checked

    def get_values_params(self):
        """Return checked values-box params as "Model:param" keys."""
        checked = []
        root = self._values_param_tree.invisibleRootItem()
        for i in range(root.childCount()):
            model_item = root.child(i)
            for j in range(model_item.childCount()):
                child = model_item.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    checked.append(
                        f"{model_item.text(0)}:{child.text(0)}")
        return checked

    def _open_plot_options(self):
        """Open the detailed per-plot-type appearance dialog."""
        from gui.shared_widgets import PlotTypeOptionsDialog
        dlg = PlotTypeOptionsDialog(self)
        dlg.exec()

    def get_output_config(self):
        return {
            "report": self._report_check.isChecked(),
            "show_correl": self._show_correl.isChecked(),
            "min_correl": self._min_correl.value(),
            "params_csv": self._params_csv.isChecked(),
            "metadata_csv": self._metadata_csv.isChecked(),
            "fit_plots": self._fit_plots.isChecked(),
            "tof_plots": self._tof_plots.isChecked(),
            "tof_plot_type": self._tof_plot_type.currentText(),
            "residual_panel": self._residual_panel.isChecked(),
            "hide_zero_bins": self._hide_zero_bins.isChecked(),
            "component_overlay": self._component_overlay.isChecked(),
            "selected_components": sorted(self.get_selected_components()),
            "values_on_plot": self._values_on_plot.isChecked(),
            "values_params": self.get_values_params(),
            "plot_format": self._plot_format.currentText(),
            "plot_dpi": self._plot_dpi.value(),
            "tracker_xaxis": self._tracker_xaxis.currentText(),
            "tracked_params": self.get_tracked_params(),
            "chisq_map": self._chisq_map.isChecked(),
            "correl_plot": self._correl_plot.isChecked(),
            "walk_plot": self._walk_plot.isChecked(),
            "confidence_bands": self._confidence_bands.isChecked(),
            "band_samples": self._band_samples.value(),
            "diag_params": self.get_diag_params(),
            "iter_mode": self._iter_mode.currentText(),
            "iter_label": self._iter_label.text(),
        }

    # ── Serialization ──

    def to_dict(self):
        d = super().to_dict()
        d.update(self.get_output_config())
        return d

    def from_dict(self, d):
        super().from_dict(d)
        self._report_check.setChecked(d.get("report", True))
        self._show_correl.setChecked(d.get("show_correl", True))
        self._min_correl.setValue(d.get("min_correl", 0.1))
        self._params_csv.setChecked(d.get("params_csv", True))
        self._metadata_csv.setChecked(d.get("metadata_csv", True))
        self._fit_plots.setChecked(d.get("fit_plots", True))
        self._tof_plots.setChecked(d.get("tof_plots", False))
        idx = self._tof_plot_type.findText(
            d.get("tof_plot_type", "ToF only"))
        if idx >= 0:
            self._tof_plot_type.setCurrentIndex(idx)
        self._tof_plot_type.setEnabled(self._tof_plots.isChecked())
        self._residual_panel.setChecked(d.get("residual_panel", True))
        self._hide_zero_bins.setChecked(d.get("hide_zero_bins", False))
        self._component_overlay.setChecked(d.get("component_overlay", False))
        saved_comps = d.get("selected_components")
        if saved_comps is not None:
            self._pending_components = saved_comps
        idx = self._plot_format.findText(d.get("plot_format", "png"))
        if idx >= 0:
            self._plot_format.setCurrentIndex(idx)
        self._plot_dpi.setValue(d.get("plot_dpi", 200))
        idx = self._tracker_xaxis.findText(d.get("tracker_xaxis", "Run number"))
        if idx >= 0:
            self._tracker_xaxis.setCurrentIndex(idx)
        self._chisq_map.setChecked(d.get("chisq_map", False))
        self._correl_plot.setChecked(d.get("correl_plot", False))
        self._walk_plot.setChecked(d.get("walk_plot", False))
        self._confidence_bands.setChecked(d.get("confidence_bands", False))
        self._band_samples.setValue(d.get("band_samples", 200))
        idx = self._iter_mode.findText(d.get("iter_mode", "Auto"))
        if idx >= 0:
            self._iter_mode.setCurrentIndex(idx)
        self._iter_label.setText(d.get("iter_label", ""))
        self._values_on_plot.setChecked(d.get("values_on_plot", False))
        # Store tree states to apply when update_tracker_params populates them
        self._pending_tracked = d.get("tracked_params", {})
        self._pending_diag = d.get("diag_params", [])
        self._pending_values = d.get("values_params", [])
