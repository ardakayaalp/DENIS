"""Isotope shift analysis tab for the DENIS Analysis workspace.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Provides the permanent Isotope Shift tab inside the Analysis QTabWidget.
It collects fit centroids from multiple analysis projects, applies
systematic and Gaussian-process reference corrections, computes isotope
shifts (with full uncertainty propagation including inter-isotope
cross-covariance), and produces the comparison plot, shift table, and
log that can be exported or pushed to the Results tab.

Depends on: cls_estimations.isotope_shift (centroid extraction, weighted
averaging, shift and uncertainty computation), cls_estimations.plotting,
gui.analysis.helpers, gui.analysis.reference_correction_panel, and (lazily
imported) gui.analysis.blocks, gui.shared_widgets, gui.results_tab.
"""

import os
import csv
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea,
    QGroupBox, QFormLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QRadioButton, QButtonGroup, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QPlainTextEdit, QFileDialog, QMessageBox, QAbstractItemView,
    QToolButton, QColorDialog, QDialog, QDialogButtonBox,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush

from cls_estimations.isotope_shift import (
    extract_centroid, weighted_average, isotope_shift,
    cooler_voltage_systematic, compute_all_shifts,
    weighted_average_with_birge, gp_propagated_sigma,
    gp_cross_covariance, mixed_mode_propagated_sigma,
)
from cls_estimations.plotting import _get_color
from gui.analysis.helpers import _make_double, _make_int
from gui.analysis.reference_correction_panel import (
    ReferenceCorrectionPanel,
)


# ═══════════════════════════════════════════════════════════════════
#  Helper: per-isotope entry row data
# ═══════════════════════════════════════════════════════════════════

class _IsotopeEntry:
    """Lightweight container for one row in the isotope table."""
    __slots__ = (
        "select_cb", "project_combo", "label_edit", "a_spin",
        "run_combo", "centroid_item", "error_item", "ref_radio",
        "color_idx",
    )


def _color_button(color="#000000"):
    """Create a small color picker button."""
    btn = QToolButton()
    btn.setFixedSize(22, 22)
    btn.setStyleSheet(f"background-color: {color}; border: 1px solid grey;")
    btn._color = color
    return btn


def _centered_cell(widget):
    """Wrap a widget in a horizontally-centered container for use as a
    QTableWidget cell widget. Without this, checkboxes and radio buttons
    sit on the left edge of their cell instead of the column center."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addStretch(1)
    lay.addWidget(widget)
    lay.addStretch(1)
    return w


def _pick_color(btn, parent):
    """Open a color dialog and update the button."""
    c = QColorDialog.getColor(QColor(btn._color), parent)
    if c.isValid():
        btn._color = c.name()
        btn.setStyleSheet(
            f"background-color: {c.name()}; border: 1px solid grey;")


# ═══════════════════════════════════════════════════════════════════
#  IsotopeShiftTab
# ═══════════════════════════════════════════════════════════════════

class IsotopeShiftTab(QWidget):
    """Permanent tab inside the Analysis QTabWidget for isotope shift
    calculations, combined plotting, and shift table generation."""

    results_ready = Signal(str, list, dict)  # → AnalysisTab → ResultsTab

    def __init__(self, analysis_tab, parent=None):
        super().__init__(parent)
        self._analysis_tab = analysis_tab
        self._entries = []          # list of _IsotopeEntry
        self._last_shift_data = []  # result of compute_all_shifts
        self._last_log = ""
        self._ref_group = QButtonGroup(self)
        self._ref_group.setExclusive(True)
        # User-dragged positions for δν labels and arrows, keyed by a
        # stable string id (isotope label) so they survive replot,
        # save and reload. Filled by the Plot Editor's drag handlers
        # and re-applied in _build_plot. _is_label_pos[id] = (x, y);
        # _is_arrow_pos[id] = ((ax, ay), (bx, by)) in data coords.
        self._is_label_pos = {}
        self._is_arrow_pos = {}
        # Registry of the current figure's draggable δν artists, keyed
        # by id, handed to the Plot Editor so it can link a label to
        # its arrow (moving the label drags the arrow tail too).
        self._is_labels = {}
        self._is_arrows = {}

        # ── Main layout: horizontal splitter ──
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ════════════ LEFT PANEL (controls) ════════════
        # Two-subtab split: "Setup" carries the entries / systematics /
        # action buttons; "Reference Correction (GP)" hosts the GP
        # panel on its own. Keeping the action row in the Setup tab
        # means Compute Shifts is visible without scrolling past the
        # tall GP panel.
        left_tabs = QTabWidget()
        # Minimum width chosen so the Isotope Entries table's columns
        # (Sel / Project / Label / A / Runs / Centroid / Stat. Err.
        # / Ref) all fit without an inner horizontal scrollbar at
        # the default split.
        left_tabs.setMinimumWidth(744)

        # ── "Setup" subtab (entries + systematics + actions) ──
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(4, 4, 4, 4)

        # ── Study name ──
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Study name:"))
        self._study_name = QLineEdit("Isotope_Shift")
        self._study_name.setToolTip(
            "Name used when sending results to the Results tab")
        name_row.addWidget(self._study_name, 1)
        left_lay.addLayout(name_row)

        # ── Isotope table ──
        iso_grp = QGroupBox("Isotope Entries")
        iso_lay = QVBoxLayout(iso_grp)

        self._iso_table = QTableWidget(0, 8)
        self._iso_table.setHorizontalHeaderLabels([
            "Sel.", "Project", "Label", "A", "Runs",
            "Centroid (MHz)", "Stat. Err.", "Ref",
        ])
        h = self._iso_table.horizontalHeader()
        h.setStretchLastSection(False)
        for col in range(8):
            h.setSectionResizeMode(
                col, QHeaderView.ResizeMode.Interactive)
        h.resizeSection(0, 40)   # Sel.
        h.resizeSection(1, 110)  # Project
        h.resizeSection(2, 80)   # Label
        h.resizeSection(3, 70)   # A
        h.resizeSection(4, 130)  # Runs
        h.resizeSection(5, 130)  # Centroid (MHz)
        h.resizeSection(6, 90)   # Stat. Err.
        h.resizeSection(7, 40)   # Ref
        self._iso_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._iso_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._iso_table.verticalHeader().setVisible(False)
        # Give rows enough height to fully display the embedded combos
        # and spinboxes. Suppress the visual selection highlight
        # entirely: Qt's QTableWidget::item:selected only paints item
        # cells, NOT the cells hosting widgets (combo, spin, radio,
        # line edit), which produces a partial row highlight. The
        # selectionModel still works for _remove_selected_row's row
        # lookup -- the highlight is simply not drawn.
        self._iso_table.verticalHeader().setDefaultSectionSize(32)
        self._iso_table.setStyleSheet(
            "QTableWidget::item:selected { "
            "background-color: transparent; "
            "color: palette(text); }"
        )
        iso_lay.addWidget(self._iso_table)
        self._iso_table.currentCellChanged.connect(
            self._on_preview_row_changed)

        # Run preview (R25): the highlighted run's spectrum + fit and
        # its red. χ², so runs can be judged while picking them —
        # collapsible so it costs nothing when not wanted.
        self._preview_box = QGroupBox("Run preview")
        self._preview_box.setCheckable(True)
        self._preview_box.setChecked(True)
        self._preview_box.setToolTip(
            "Preview of the run selected in the highlighted row's Runs "
            "column: data + fitted curve + reduced χ². Untick the box "
            "to collapse.")
        _pv_lay = QVBoxLayout(self._preview_box)
        _pv_lay.setContentsMargins(4, 4, 4, 4)
        self._preview_fig = Figure(dpi=100, constrained_layout=True)
        self._preview_ax = self._preview_fig.add_subplot(111)
        self._preview_canvas = FigureCanvasQTAgg(self._preview_fig)
        self._preview_canvas.setFixedHeight(190)
        _pv_lay.addWidget(self._preview_canvas)
        self._preview_box.toggled.connect(
            self._preview_canvas.setVisible)
        iso_lay.addWidget(self._preview_box)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Row")
        add_btn.clicked.connect(self._add_row)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("- Remove Row")
        rm_btn.clicked.connect(self._remove_selected_row)
        btn_row.addWidget(rm_btn)
        refresh_btn = QPushButton("Refresh from Projects")
        refresh_btn.setToolTip(
            "Scan analysis projects and auto-populate / update rows")
        refresh_btn.clicked.connect(self._refresh_projects)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        iso_lay.addLayout(btn_row)

        # Give the entries group the layout stretch so the table grows
        # to fill available vertical space instead of leaving a gap at
        # the bottom of the panel.
        left_lay.addWidget(iso_grp, 1)

        # ── Systematic error ──
        sys_grp = QGroupBox("Systematic Error")
        sys_lay = QVBoxLayout(sys_grp)

        self._sys_manual_rb = QRadioButton("Manual")
        self._sys_manual_rb.setChecked(True)
        self._sys_manual_rb.setToolTip(
            "Apply a single user-defined systematic error to all isotopes")
        sys_manual_row = QHBoxLayout()
        sys_manual_row.addWidget(self._sys_manual_rb)
        self._sys_manual_spin = _make_double(
            0.0, 0.0, 1e6, 4, 0.1,
            tooltip="Systematic error applied to every centroid (MHz)")
        sys_manual_row.addWidget(self._sys_manual_spin)
        sys_manual_row.addWidget(QLabel("MHz"))
        sys_manual_row.addStretch()
        sys_lay.addLayout(sys_manual_row)

        self._sys_auto_rb = QRadioButton(
            "Auto from cooler voltage && laser setpoint")
        self._sys_auto_rb.setToolTip(
            "Estimate systematic error per isotope from the run-to-run\n"
            "jitter of cooler voltage and laser setpoint.\n"
            "Uses SEM = std/\u221aN (standard error of the mean).\n"
            "Projects with only 1 run get \u03c3=0 (flagged as unknown).")
        sys_lay.addWidget(self._sys_auto_rb)

        sys_auto_detail = QFormLayout()
        self._sys_charge = _make_int(1, 1, 10,
                                     tooltip="Ion charge state for "
                                     "Doppler shift calculation")
        sys_auto_detail.addRow("Charge:", self._sys_charge)
        self._sys_geometry = QComboBox()
        self._sys_geometry.addItems(["anti-collinear", "collinear"])
        self._sys_geometry.setToolTip(
            "Laser-ion geometry for Doppler shift calculation")
        sys_auto_detail.addRow("Geometry:", self._sys_geometry)
        sys_lay.addLayout(sys_auto_detail)

        # Auto-computed systematics display
        self._sys_details_text = QPlainTextEdit()
        self._sys_details_text.setReadOnly(True)
        self._sys_details_text.setMaximumHeight(160)
        self._sys_details_text.setPlaceholderText(
            "Auto-computed systematics will appear here after Compute.\n"
            "Shows per-project: cooler V mean/std/SEM, laser setpoint "
            "mean/std/SEM,\nDoppler sensitivity d(\u03bd)/d(V), and "
            "combined systematic error.")
        self._sys_details_text.setToolTip(
            "Detailed breakdown of systematic error estimation.\n"
            "Each project uses its OWN jitter (not the worst).\n"
            "SEM = std/\u221aN is used when averaging multiple runs.")
        sys_lay.addWidget(self._sys_details_text)

        left_lay.addWidget(sys_grp)

        # Plot options live in a separate modeless dialog so the left
        # panel only carries Isotope Entries + Systematic Error and the
        # entries table can grow into the freed vertical space.
        self._plot_options_dialog = self._build_plot_options_dialog()

        # ── Action buttons ──
        _btn_style = ("QPushButton { padding: 6px 16px; "
                      "font-weight: bold; }")
        act_row = QHBoxLayout()
        compute_btn = QPushButton("Compute Shifts")
        compute_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px 16px; }")
        compute_btn.clicked.connect(self._compute_shifts)
        act_row.addWidget(compute_btn)

        send_btn = QPushButton("Send to Results")
        send_btn.setStyleSheet(_btn_style)
        send_btn.clicked.connect(self._push_to_results)
        act_row.addWidget(send_btn)
        left_lay.addLayout(act_row)

        act_row2 = QHBoxLayout()
        plot_opts_btn = QPushButton("Plot Options...")
        plot_opts_btn.setToolTip(
            "Open the plot options window: layout, content, colors, "
            "alpha, z-order")
        plot_opts_btn.clicked.connect(self._show_plot_options)
        act_row2.addWidget(plot_opts_btn)
        csv_btn = QPushButton("Export Table CSV")
        csv_btn.clicked.connect(self._export_csv)
        act_row2.addWidget(csv_btn)
        log_btn = QPushButton("Save Log")
        log_btn.clicked.connect(self._save_log)
        act_row2.addWidget(log_btn)
        act_row2.addStretch()
        left_lay.addLayout(act_row2)

        setup_scroll.setWidget(left_widget)
        left_tabs.addTab(setup_scroll, "Setup")

        # ── "Reference Correction (GP)" subtab ──
        # The GP panel is tall (kernel form, ref list, diagnostic
        # plot, sample list, corrections table) so wrap it in its
        # own scroll area for narrow window heights.
        self._ref_corr_panel = ReferenceCorrectionPanel(self._analysis_tab)
        ref_scroll = QScrollArea()
        ref_scroll.setWidgetResizable(True)
        ref_scroll.setWidget(self._ref_corr_panel)
        left_tabs.addTab(ref_scroll, "Reference Correction (GP)")

        splitter.addWidget(left_tabs)

        # ════════════ RIGHT PANEL (output) ════════════
        right = QTabWidget()

        # ── Plot sub-tab ──
        plot_widget = QWidget()
        plot_lay2 = QVBoxLayout(plot_widget)
        plot_lay2.setContentsMargins(2, 2, 2, 2)
        self._figure = Figure(figsize=(10, 8))
        self._canvas = FigureCanvasQTAgg(self._figure)
        # Right-click editor must carry the δν arrow registry.
        self._canvas._plot_editor_opener = self._edit_plot
        self._toolbar = NavigationToolbar2QT(self._canvas, plot_widget)
        plot_lay2.addWidget(self._toolbar)
        plot_lay2.addWidget(self._canvas, 1)
        edit_btn = QPushButton("Edit Plot")
        edit_btn.clicked.connect(self._edit_plot)
        tb_row = QHBoxLayout()
        tb_row.addStretch()
        tb_row.addWidget(edit_btn)
        plot_lay2.addLayout(tb_row)
        right.addTab(plot_widget, "Plot")

        # ── Table sub-tab ──
        self._result_table = QTableWidget()
        self._result_table.setColumnCount(14)
        self._result_table.setHorizontalHeaderLabels([
            "Isotope", "A", "Centroid (MHz)",
            "\u03c3_fit", "\u03c3_scatter",
            "\u03c3_corr", "\u03c3_volt",
            "\u03b4\u03bd (MHz)",
            "\u03c3_\u03b4\u03bd(fit)",
            "\u03c3_\u03b4\u03bd(scatter)",
            "\u03c3_\u03b4\u03bd(corr)",
            "\u03c3_\u03b4\u03bd(sys)",
            "\u03c3_\u03b4\u03bd", "Ref",
        ])
        # Clarify the uncertainty budget so the component columns aren't
        # mis-summed: total = quad(fit, scatter, sys), and (corr) is a
        # SUB-component of (sys), not an independent addend. The fit-only
        # column was previously labelled '(stat)', which read as "the
        # statistical part of the total" yet excluded the Birge scatter
        # (code review 2026-06-02, is-table-stat-excludes-scatter).
        _budget_tips = {
            3: "Fit-only statistical uncertainty (inverse-variance; no Birge "
               "scatter inflation).",
            8: "delta-nu fit-only stat (was 'stat'); EXCLUDES the Birge "
               "scatter in the next column.",
            9: "Run-to-run (Birge) scatter contribution to delta-nu.",
            10: "Reference GP-correction contribution -- a SUB-component of "
                "sigma_delta-nu(sys); do NOT add it again on top of (sys).",
            11: "Systematic contribution to delta-nu (already includes the "
                "(corr) term).",
            12: "Total: sigma_delta-nu = quad(fit, scatter, sys).",
        }
        for _c, _tip in _budget_tips.items():
            _hi = self._result_table.horizontalHeaderItem(_c)
            if _hi is not None:
                _hi.setToolTip(_tip)
        rh = self._result_table.horizontalHeader()
        for col in range(14):
            rh.setSectionResizeMode(
                col, QHeaderView.ResizeMode.Interactive)
        for col in range(13):
            rh.setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents)
        rh.setStretchLastSection(False)
        right.addTab(self._result_table, "Table")

        # ── Log sub-tab ──
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        right.addTab(self._log_text, "Log")

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

    # ═══════════════════════════════════════════════════════════════
    #  Plot options dialog
    # ═══════════════════════════════════════════════════════════════

    def _build_plot_options_dialog(self):
        """Construct the modeless Plot Options dialog. All option widgets
        are stored as self attributes so the existing draw / save code
        that reads them continues to work unchanged."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Plot Options")
        dlg.setMinimumWidth(440)
        outer = QVBoxLayout(dlg)
        plot_lay = QFormLayout()
        outer.addLayout(plot_lay)

        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["Stacked Subplots", "Overlaid"])
        plot_lay.addRow("Layout:", self._layout_combo)

        self._content_combo = QComboBox()
        self._content_combo.addItems(["Data + Model", "Data Only"])
        plot_lay.addRow("Content:", self._content_combo)

        self._xaxis_combo = QComboBox()
        self._xaxis_combo.addItems(
            ["Relative to Reference", "Absolute"])
        plot_lay.addRow("X-axis:", self._xaxis_combo)

        self._centroid_cb = QCheckBox("Show centroid lines")
        self._centroid_cb.setChecked(True)
        self._centroid_cb.setToolTip(
            "Draw a dashed vertical line at each isotope's centroid")
        plot_lay.addRow(self._centroid_cb)

        self._ref_line_cb = QCheckBox(
            "Show reference centroid across all panels")
        self._ref_line_cb.setToolTip(
            "Draw the reference isotope's centroid as a dotted line\n"
            "on every subplot (useful with δν arrows)")
        plot_lay.addRow(self._ref_line_cb)

        self._arrows_cb = QCheckBox("Show δν arrows to reference")
        self._arrows_cb.setToolTip(
            "Draw double-headed arrows between each isotope's centroid\n"
            "and the reference centroid, labeled with δν value")
        plot_lay.addRow(self._arrows_cb)

        self._sort_cb = QCheckBox("Sort by mass number A")
        self._sort_cb.setChecked(True)
        self._sort_cb.setToolTip(
            "Order isotopes by mass number A (lightest on top)")
        plot_lay.addRow(self._sort_cb)

        self._hide_zero_cb = QCheckBox("Hide zero-count bins")
        self._hide_zero_cb.setToolTip(
            "Drop empty data bins from the plot AND break the fit\n"
            "model line over those gaps. Useful when a scan covers a\n"
            "much wider window than the actual peak structure.")
        plot_lay.addRow(self._hide_zero_cb)

        # Color options
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Model:"))
        self._model_color_btn = _color_button("#000000")
        self._model_color_btn.clicked.connect(
            lambda: _pick_color(self._model_color_btn, self))
        color_row.addWidget(self._model_color_btn)
        color_row.addWidget(QLabel("Centroid:"))
        self._centroid_color_btn = _color_button("#555555")
        self._centroid_color_btn.clicked.connect(
            lambda: _pick_color(self._centroid_color_btn, self))
        color_row.addWidget(self._centroid_color_btn)
        color_row.addWidget(QLabel("Ref:"))
        self._ref_color_btn = _color_button("#FF0000")
        self._ref_color_btn.clicked.connect(
            lambda: _pick_color(self._ref_color_btn, self))
        color_row.addWidget(self._ref_color_btn)
        color_row.addWidget(QLabel("Arrows:"))
        self._arrow_color_btn = _color_button("#333333")
        self._arrow_color_btn.clicked.connect(
            lambda: _pick_color(self._arrow_color_btn, self))
        color_row.addWidget(self._arrow_color_btn)
        color_row.addStretch()
        plot_lay.addRow("Colors:", color_row)

        # Data color pool
        self._color_pool = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        pool_row = QHBoxLayout()
        self._pool_container = QHBoxLayout()
        self._pool_btns = []
        self._rebuild_pool_buttons()
        pool_row.addLayout(self._pool_container)
        add_c_btn = QPushButton("+")
        add_c_btn.setFixedSize(22, 22)
        add_c_btn.setToolTip("Add a color to the pool")
        add_c_btn.clicked.connect(self._add_pool_color)
        pool_row.addWidget(add_c_btn)
        rm_c_btn = QPushButton("-")
        rm_c_btn.setFixedSize(22, 22)
        rm_c_btn.setToolTip("Remove the last color from the pool")
        rm_c_btn.clicked.connect(self._remove_pool_color)
        pool_row.addWidget(rm_c_btn)
        pool_row.addStretch()
        plot_lay.addRow("Data colors:", pool_row)

        # Alpha (transparency) per element
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Model:"))
        self._model_alpha = _make_double(0.85, 0.0, 1.0, 2, 0.05)
        self._model_alpha.setMinimumWidth(55)
        alpha_row.addWidget(self._model_alpha)
        alpha_row.addWidget(QLabel("Centroid:"))
        self._centroid_alpha = _make_double(0.6, 0.0, 1.0, 2, 0.05)
        self._centroid_alpha.setMinimumWidth(55)
        alpha_row.addWidget(self._centroid_alpha)
        alpha_row.addWidget(QLabel("Ref:"))
        self._ref_alpha = _make_double(0.7, 0.0, 1.0, 2, 0.05)
        self._ref_alpha.setMinimumWidth(55)
        alpha_row.addWidget(self._ref_alpha)
        alpha_row.addWidget(QLabel("Arrows:"))
        self._arrow_alpha = _make_double(1.0, 0.0, 1.0, 2, 0.05)
        self._arrow_alpha.setMinimumWidth(55)
        alpha_row.addWidget(self._arrow_alpha)
        plot_lay.addRow("Alpha:", alpha_row)

        # Z-order per element
        zorder_row = QHBoxLayout()
        zorder_row.addWidget(QLabel("Model:"))
        self._model_zorder = _make_int(3, 0, 10)
        self._model_zorder.setMinimumWidth(45)
        zorder_row.addWidget(self._model_zorder)
        zorder_row.addWidget(QLabel("Centroid:"))
        self._centroid_zorder = _make_int(1, 0, 10)
        self._centroid_zorder.setMinimumWidth(45)
        zorder_row.addWidget(self._centroid_zorder)
        zorder_row.addWidget(QLabel("Ref:"))
        self._ref_zorder = _make_int(1, 0, 10)
        self._ref_zorder.setMinimumWidth(45)
        zorder_row.addWidget(self._ref_zorder)
        zorder_row.addWidget(QLabel("Arrows:"))
        self._arrow_zorder = _make_int(5, 0, 10)
        self._arrow_zorder.setMinimumWidth(45)
        zorder_row.addWidget(self._arrow_zorder)
        plot_lay.addRow("Z-order:", zorder_row)

        # Modeless Close button so the user can keep the dialog open
        # while iterating on plot output.
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.hide)
        btns.accepted.connect(dlg.hide)
        outer.addWidget(btns)
        return dlg

    def _show_plot_options(self):
        self._plot_options_dialog.show()
        self._plot_options_dialog.raise_()
        self._plot_options_dialog.activateWindow()

    # ═══════════════════════════════════════════════════════════════
    #  Data color pool management
    # ═══════════════════════════════════════════════════════════════

    def _rebuild_pool_buttons(self):
        """Rebuild the color swatch buttons from self._color_pool."""
        # Clear existing
        for btn in self._pool_btns:
            btn.setParent(None)
        self._pool_btns.clear()
        for i, c in enumerate(self._color_pool):
            btn = QToolButton()
            btn.setFixedSize(18, 18)
            btn.setStyleSheet(
                f"background-color: {c}; border: 1px solid grey;")
            btn._color = c
            btn._idx = i
            btn.clicked.connect(
                lambda checked=False, b=btn: self._edit_pool_color(b))
            self._pool_container.addWidget(btn)
            self._pool_btns.append(btn)

    def _edit_pool_color(self, btn):
        c = QColorDialog.getColor(QColor(btn._color), self)
        if c.isValid():
            btn._color = c.name()
            btn.setStyleSheet(
                f"background-color: {c.name()}; border: 1px solid grey;")
            self._color_pool[btn._idx] = c.name()

    def _add_pool_color(self):
        c = QColorDialog.getColor(QColor("#000000"), self)
        if c.isValid():
            self._color_pool.append(c.name())
            self._rebuild_pool_buttons()

    def _remove_pool_color(self):
        if len(self._color_pool) > 1:
            self._color_pool.pop()
            self._rebuild_pool_buttons()

    def _get_data_color(self, idx):
        """Get a color from the pool, wrapping around."""
        if not self._color_pool:
            return "#1f77b4"
        return self._color_pool[idx % len(self._color_pool)]

    # ═══════════════════════════════════════════════════════════════
    #  Row management
    # ═══════════════════════════════════════════════════════════════

    def _add_row(self, project_name="", label="", A=0,
                 is_ref=False, run_sel="Weighted Average"):
        """Add a row to the isotope entries table."""
        row = self._iso_table.rowCount()
        self._iso_table.insertRow(row)

        e = _IsotopeEntry()
        e.color_idx = row

        # Column 0 - Select checkbox (centered in cell)
        e.select_cb = QCheckBox()
        e.select_cb.setChecked(True)
        # Refresh row fades whenever any select toggles, so unselected
        # rows visibly recede and the eye lands on the selected ones.
        e.select_cb.toggled.connect(self._refresh_row_fades)
        self._iso_table.setCellWidget(row, 0, _centered_cell(e.select_cb))

        # Column 1 - Project combo
        e.project_combo = QComboBox()
        self._populate_project_combo(e.project_combo, project_name)
        e.project_combo.currentTextChanged.connect(
            lambda _: self._on_project_changed(e))
        self._iso_table.setCellWidget(row, 1, e.project_combo)

        # Column 2 - Label
        e.label_edit = QLineEdit(label)
        # Give it a subtle frame so it reads as editable. Without
        # this, the dark-theme QLineEdit looks like static text
        # against the table background, while its Project / A /
        # Runs neighbors all have visible decorations (combo
        # arrows, spin arrows).
        e.label_edit.setStyleSheet(
            "QLineEdit { border: 1px solid palette(mid); "
            "padding: 2px 4px; }")
        e.label_edit.setToolTip(
            "Free-form label shown in the shift table and plots.")
        self._iso_table.setCellWidget(row, 2, e.label_edit)

        # Column 3 - A
        e.a_spin = _make_int(A, 1, 300)
        self._iso_table.setCellWidget(row, 3, e.a_spin)

        # Column 4 - Run selection
        e.run_combo = QComboBox()
        e.run_combo.addItem("Weighted Average")
        if run_sel != "Weighted Average":
            e.run_combo.addItem(run_sel)
            e.run_combo.setCurrentText(run_sel)
        e.run_combo.currentTextChanged.connect(
            lambda _t, ent=e: self._update_run_preview(ent))
        self._iso_table.setCellWidget(row, 4, e.run_combo)

        # Column 5 - Centroid (read-only)
        e.centroid_item = QTableWidgetItem("--")
        e.centroid_item.setFlags(
            e.centroid_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        e.centroid_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._iso_table.setItem(row, 5, e.centroid_item)

        # Column 6 - Error (read-only)
        e.error_item = QTableWidgetItem("--")
        e.error_item.setFlags(
            e.error_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        e.error_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._iso_table.setItem(row, 6, e.error_item)

        # Column 7 - Reference radio (centered in cell)
        e.ref_radio = QRadioButton()
        if is_ref:
            e.ref_radio.setChecked(True)
        self._ref_group.addButton(e.ref_radio)
        self._iso_table.setCellWidget(row, 7, _centered_cell(e.ref_radio))

        self._entries.append(e)
        self._on_project_changed(e)  # populate run combo
        # Set the initial fade state (no-op for a default-checked row,
        # but keeps things consistent if a row is added pre-unchecked).
        self._refresh_row_fades()
        return e

    def _on_preview_row_changed(self, row, _col=0, _prow=-1, _pcol=-1):
        """Preview the run of whichever table row gains focus."""
        if 0 <= row < len(self._entries):
            self._update_run_preview(self._entries[row])

    def _update_run_preview(self, entry):
        """Draw the entry's selected run — spectrum, fitted curve and
        reduced χ² — so runs can be judged while picking them."""
        ax = getattr(self, "_preview_ax", None)
        if ax is None or not self._preview_box.isChecked():
            return
        ax.clear()
        run_sel = entry.run_combo.currentText()
        project = self._find_project(entry.project_combo.currentText())
        results = (project._last_results or []) if project else []
        ok = [r for r in results if r.get("success")]
        r = None
        if run_sel and run_sel != "Weighted Average":
            for cand in ok:
                if str(cand.get("run_number")) == run_sel:
                    r = cand
                    break
        if r is None:
            ax.set_title(
                ("Weighted Average selected — pick a specific run to "
                 "preview its fit") if ok else "no fit results",
                fontsize=8)
            ax.tick_params(labelsize=6)
            self._preview_canvas.draw_idle()
            return
        try:
            x = np.asarray(r.get("x", []), dtype=float)
            y = np.asarray(r.get("y", []), dtype=float)
            yerr = r.get("yerr")
            if x.size and y.size:
                if (yerr is not None and np.ndim(yerr) == 1
                        and len(yerr) == len(y)):
                    ax.errorbar(x, y, yerr=np.asarray(yerr, dtype=float),
                                fmt=".", ms=3, lw=0.8, alpha=0.7,
                                color="#4477aa")
                else:
                    ax.plot(x, y, ".", ms=3, alpha=0.7, color="#4477aa")
            xs = np.asarray(r.get("x_smooth", []), dtype=float)
            ys = np.asarray(r.get("y_fit_smooth", []), dtype=float)
            if xs.size and ys.size:
                ax.plot(xs, ys, "-", lw=1.2, color="#cc3311")
        except Exception:
            pass
        title = f"run {run_sel}"
        redchi = (r.get("fit_quality") or {}).get("redchi")
        if redchi is not None:
            try:
                title += f"   red. χ² = {float(redchi):.3f}"
            except (TypeError, ValueError):
                pass
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        self._preview_canvas.draw_idle()

    def _refresh_row_fades(self):
        """Dim rows whose Sel. checkbox is unchecked so visual focus
        stays on the rows that actually contribute to the fit."""
        for row, ent in enumerate(self._entries):
            self._apply_row_fade(row, faded=not ent.select_cb.isChecked())

    def _apply_row_fade(self, row, faded):
        """Apply (or remove) a faded look to one row of the entry
        table. The Sel. checkbox in column 0 stays at full contrast
        so the user can always see and toggle it."""
        opacity = 0.35 if faded else 1.0
        for col in range(self._iso_table.columnCount()):
            widget = self._iso_table.cellWidget(row, col)
            if widget is None:
                # Plain item cell (centroid / error). Use foreground
                # brush for the dim look; clearing it on un-fade
                # restores the default palette text color.
                item = self._iso_table.item(row, col)
                if item is None:
                    continue
                if faded:
                    item.setForeground(QBrush(QColor(140, 140, 140)))
                else:
                    item.setForeground(QBrush())
                continue
            if col == 0:
                # Keep the Sel. checkbox itself fully visible.
                continue
            eff = widget.graphicsEffect()
            if not isinstance(eff, QGraphicsOpacityEffect):
                eff = QGraphicsOpacityEffect(widget)
                widget.setGraphicsEffect(eff)
            eff.setOpacity(opacity)

    def _remove_selected_row(self):
        rows = sorted(set(idx.row() for idx in
                          self._iso_table.selectedIndexes()), reverse=True)
        if not rows:
            return
        for r in rows:
            if r < len(self._entries):
                entry = self._entries.pop(r)
                self._ref_group.removeButton(entry.ref_radio)
            self._iso_table.removeRow(r)

    def _populate_project_combo(self, combo, current=""):
        """Fill a project combo box from analysis sample projects.

        Reference projects are excluded -- they don't represent
        isotopes to compute shifts on; they feed the GP corrector
        below.
        """
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")
        for p in self._analysis_tab._projects:
            if p.is_reference:
                continue
            combo.addItem(p._project_name)
        if current and combo.findText(current) >= 0:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_project_changed(self, entry):
        """When the project combo changes, update run combo and auto-fill."""
        project = self._find_project(entry.project_combo.currentText())
        entry.run_combo.blockSignals(True)
        entry.run_combo.clear()
        entry.run_combo.addItem("Weighted Average")

        if project:
            # Try in-memory results first, fall back to disk
            if not project._last_results:
                self._load_results_from_disk(project)

            if project._last_results:
                for r in project._last_results:
                    if r.get("success"):
                        entry.run_combo.addItem(
                            str(r.get("run_number", "?")))
            # Auto-fill A from source block
            src = self._get_source_block(project)
            if src:
                cfg = src.get_source_config()
                if not entry.label_edit.text():
                    entry.label_edit.setText(project._project_name)
                entry.a_spin.setValue(cfg.get("A", 0))
        entry.run_combo.blockSignals(False)

    def _find_project(self, name):
        """Find an AnalysisProject by name."""
        if not name:
            return None
        for p in self._analysis_tab._projects:
            if p._project_name == name:
                return p
        return None

    def _get_source_block(self, project):
        """Return the first SourceBlock in a project, or None."""
        from gui.analysis.blocks import SourceBlock
        for b in project._blocks:
            if isinstance(b, SourceBlock):
                return b
        return None

    # ═══════════════════════════════════════════════════════════════
    #  Load results from disk (when _last_results is None)
    # ═══════════════════════════════════════════════════════════════

    def _load_results_from_disk(self, project):
        """Load fit results from the latest iteration on disk into
        project._last_results so the IS tab can use them."""
        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()
        project_dir = os.path.join(base_dir, project._project_name)
        if not os.path.isdir(project_dir):
            return

        # Find latest iteration
        iters = sorted([d for d in os.listdir(project_dir)
                        if os.path.isdir(os.path.join(project_dir, d))
                        and d.startswith("iter_")])
        if not iters:
            return
        iter_name = iters[-1]
        iter_dir = os.path.join(project_dir, iter_name)

        # Load parameters.csv
        params_path = os.path.join(iter_dir, "parameters.csv")
        if not os.path.isfile(params_path):
            return

        try:
            import pandas as pd
            params_df = pd.read_csv(params_path)
        except Exception:
            return

        # Build results list from CSV + NPZ files
        results = []
        plots_dir = os.path.join(iter_dir, "plots")

        # Group parameters by run_number
        if "run_number" not in params_df.columns:
            return
        for run_num, grp in params_df.groupby("run_number"):
            result = {
                "success": True,
                "run_number": str(run_num),
                "run_file": "",
                "report": "",
                "params_df": {
                    "Parameter": grp["Parameter"].tolist(),
                    "Value": grp["Value"].tolist(),
                    "Stderr": grp.get("Error",
                                      grp.get("Stderr",
                                              pd.Series([0.0] * len(grp))
                                              )).tolist(),
                },
                "metadata_df": {},
                "x": [], "y": [], "yerr": [],
                "y_fit": [], "x_smooth": [], "y_fit_smooth": [],
                "residuals": [],
                "diagnostics": {},
                "fwhm": {},
                "peak_positions": {},
                "fit_quality": {},
                "run_metadata": {},
                "harmonic": 0,
            }
            # Try loading NPZ for x/y data
            npz_path = os.path.join(plots_dir, f"fit_run_{run_num}.npz")
            if os.path.isfile(npz_path):
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    result["x"] = data.get("x", np.array([])).tolist()
                    result["y"] = data.get("y", np.array([])).tolist()
                    result["yerr"] = data.get("yerr",
                                              np.array([])).tolist()
                    result["x_smooth"] = data.get(
                        "x_smooth", np.array([])).tolist()
                    result["y_fit_smooth"] = data.get(
                        "y_fit_smooth", np.array([])).tolist()
                except Exception:
                    pass

            # Try loading run metadata from metadata.csv
            meta_path = os.path.join(iter_dir, "metadata.csv")
            if os.path.isfile(meta_path):
                try:
                    meta_df = pd.read_csv(meta_path)
                    run_meta = meta_df[
                        meta_df["run_number"].astype(str) == str(run_num)]
                    if not run_meta.empty:
                        result["fit_quality"] = {
                            "redchi": run_meta.iloc[0].get(
                                "Reduced Chi-sq", None),
                        }
                except Exception:
                    pass

            # Try loading run summary for cooler_v / timestamps
            summary_path = os.path.join(iter_dir, "run_summary.csv")
            if os.path.isfile(summary_path):
                try:
                    summary_df = pd.read_csv(summary_path)
                    run_row = summary_df[
                        summary_df["run_number"].astype(str)
                        == str(run_num)]
                    if not run_row.empty:
                        rm = {}
                        # Numeric metadata; row column name -> rm key
                        numeric_cols = {
                            "cooler_v": "cooler_v",
                            "laser_set": "laser_set",
                            "laser_set_cm1": "laser_set",
                            "ts_start": "ts_start",
                            "ts_stop": "ts_stop",
                            "centroid_correction_mhz":
                                "centroid_correction_mhz",
                            "centroid_correction_sigma_mhz":
                                "centroid_correction_sigma_mhz",
                        }
                        # Bool / string fields -- track separately
                        # because pd.read_csv reads them differently.
                        if "centroid_correction_applied" in (
                                run_row.columns):
                            v = run_row.iloc[0][
                                "centroid_correction_applied"]
                            if pd.notna(v):
                                rm["centroid_correction_applied"] = (
                                    bool(v) and str(v).lower()
                                    not in ("false", "0", ""))
                        if "centroid_correction_mode" in (
                                run_row.columns):
                            v = run_row.iloc[0][
                                "centroid_correction_mode"]
                            if pd.notna(v) and str(v):
                                rm["centroid_correction_mode"] = str(v)
                        # Restore the per-constituent correction list
                        # for merged fits -- without this, σ_correction
                        # propagation degrades to the mean(μ) +
                        # RMS(σ) approximation after a save / reload
                        # cycle.
                        if "correction_constituents_json" in (
                                run_row.columns):
                            v = run_row.iloc[0][
                                "correction_constituents_json"]
                            if pd.notna(v) and str(v):
                                try:
                                    import json
                                    rm["correction_constituents"] = (
                                        json.loads(str(v)))
                                except (ValueError, TypeError):
                                    pass
                        for col, key in numeric_cols.items():
                            if col in run_row.columns:
                                v = run_row.iloc[0][col]
                                if pd.notna(v):
                                    rm[key] = float(v)
                        if "date" in run_row.columns:
                            v = run_row.iloc[0]["date"]
                            if pd.notna(v):
                                rm["date"] = str(v)
                        result["run_metadata"] = rm
                except Exception:
                    pass

            results.append(result)

        if results:
            project._last_results = results

    # ═══════════════════════════════════════════════════════════════
    #  Refresh from projects
    # ═══════════════════════════════════════════════════════════════

    def _refresh_projects(self):
        """Scan analysis projects and update / add rows.

        Reference projects (``is_reference=True``) are skipped from the
        isotope-shift table -- they feed the GP corrector instead, via
        the Reference Correction panel below.
        """
        existing_names = set()
        # Update existing rows
        for e in self._entries:
            self._populate_project_combo(
                e.project_combo, e.project_combo.currentText())
            self._on_project_changed(e)
            name = e.project_combo.currentText()
            if name:
                existing_names.add(name)

        # Add new rows for projects not yet in the table (sample only)
        for p in self._analysis_tab._projects:
            if p.is_reference:
                continue
            if p._project_name not in existing_names:
                src = self._get_source_block(p)
                A = src.get_source_config().get("A", 0) if src else 0
                is_ref = len(self._entries) == 0  # first added = reference
                self._add_row(
                    project_name=p._project_name,
                    label=p._project_name,
                    A=A,
                    is_ref=is_ref,
                )

        # Reference Correction panel needs the same refresh trigger.
        if hasattr(self, "_ref_corr_panel"):
            self._ref_corr_panel.refresh_projects()

        self._update_centroids()

    def _update_centroids(self):
        """Read centroid values from cached fit results."""
        for e in self._entries:
            project = self._find_project(e.project_combo.currentText())
            if not project or not project._last_results:
                e.centroid_item.setText("no fit")
                e.error_item.setText("--")
                continue

            ok_results = [r for r in project._last_results
                          if r.get("success")]
            if not ok_results:
                e.centroid_item.setText("failed")
                e.error_item.setText("--")
                continue

            run_sel = e.run_combo.currentText()
            if run_sel == "Weighted Average":
                vals, errs = [], []
                for r in ok_results:
                    c, s = extract_centroid(r.get("params_df", {}))
                    if c is not None:
                        vals.append(c)
                        errs.append(s)
                if vals:
                    c, s = weighted_average(vals, errs)
                    e.centroid_item.setText(f"{c:.4f}")
                    e.error_item.setText(f"{s:.4f}")
                else:
                    e.centroid_item.setText("N/A")
                    e.error_item.setText("--")
            else:
                # Specific run
                for r in ok_results:
                    if str(r.get("run_number", "")) == run_sel:
                        c, s = extract_centroid(r.get("params_df", {}))
                        if c is not None:
                            e.centroid_item.setText(f"{c:.4f}")
                            e.error_item.setText(f"{s:.4f}")
                        else:
                            e.centroid_item.setText("N/A")
                            e.error_item.setText("--")
                        break

    # ═══════════════════════════════════════════════════════════════
    #  Compute isotope shifts
    # ═══════════════════════════════════════════════════════════════

    def _compute_shifts(self):
        """Main computation: centroids -> systematics -> shifts -> plot + table."""
        self._update_centroids()

        # Collect selected entries
        selected = [e for e in self._entries if e.select_cb.isChecked()]
        if len(selected) < 2:
            QMessageBox.warning(self, "Isotope Shifts",
                                "Select at least 2 isotopes.")
            return

        # Find reference
        ref_entry = None
        for e in selected:
            if e.ref_radio.isChecked():
                ref_entry = e
                break
        if ref_entry is None:
            QMessageBox.warning(self, "Isotope Shifts",
                                "Select a reference isotope.")
            return

        log_lines = []

        # Build isotope_entries dict
        isotope_entries = {}
        # Per-isotope (timestamp_hours, weight) arrays for the runs
        # that contributed to each weighted-average centroid. Used by
        # the Phase-5 GP cross-covariance machinery below; left empty
        # for isotopes pinned to a single run.
        isotope_runs = {}
        run_metadata_by_project = {}
        mass_by_project = {}
        ref_key = None

        for e in selected:
            label = e.label_edit.text() or f"A={e.a_spin.value()}"
            project = self._find_project(e.project_combo.currentText())
            if not project or not project._last_results:
                log_lines.append(
                    f"WARNING: '{label}' has no fit results, skipping.")
                continue

            ok_results = [r for r in project._last_results
                          if r.get("success")]

            # Get centroid + per-run timestamps so the GP correction
            # can be propagated through the weighted average.
            run_sel = e.run_combo.currentText()
            iso_t_hours: list[float] = []
            iso_weights: list[float] = []
            iso_total_weight = 0.0  # full-centroid normalization (W_total)
            sigma_scatter = 0.0
            # A run participates in the GP-correction propagation
            # only if Phase D wrote centroid_correction_applied=True
            # for it -- the canonical "the spectrum was actually
            # shifted at fit time" signal. Within applied runs,
            # split by mode: Auto rows go through GP posterior cov;
            # Manual rows propagate the user-entered σ as
            # independent diagonal noise. Auto/Manual mode also
            # tells us how a run participates in inter-isotope
            # cross-cov (only Auto contributes).
            iso_auto_t: list[float] = []
            iso_auto_w: list[float] = []
            iso_manual_sigmas: list[float] = []
            iso_manual_w: list[float] = []
            # Merged-spectrum constituents: when a fit result has a
            # `correction_constituents` list, we expand it instead
            # of treating the merged result as a single Auto row,
            # so σ_correction reflects the per-file structure
            # (rather than a mean(μ) + RMS(σ) approximation).
            iso_merged_constituents: list[dict] = []
            if run_sel == "Weighted Average":
                vals, errs = [], []
                run_metas: list[dict] = []
                for r in ok_results:
                    c, s = extract_centroid(r.get("params_df", {}))
                    if c is None:
                        continue
                    vals.append(c)
                    errs.append(s if (s and s > 0) else 1.0)
                    run_metas.append(r.get("run_metadata") or {})
                if not vals:
                    log_lines.append(
                        f"WARNING: '{label}' no centroids found, skipping.")
                    continue
                centroid, sigma_fit, sigma_scatter, birge = (
                    weighted_average_with_birge(vals, errs))
                log_lines.append(
                    f"{label}: weighted average of {len(vals)} runs, "
                    f"centroid = {centroid:.4f} ± {sigma_fit:.4f} MHz "
                    f"(Birge ratio = {birge:.3f}; "
                    f"σ_scatter = {sigma_scatter:.4f} MHz)")
                full_weights = [1.0 / (s ** 2) for s in errs]
                iso_total_weight = float(sum(full_weights))
                for w, rm in zip(full_weights, run_metas):
                    if not bool(rm.get(
                            "centroid_correction_applied", False)):
                        continue
                    mode = rm.get("centroid_correction_mode")
                    constituents = rm.get("correction_constituents")
                    if mode == "merged" and constituents:
                        # Merged result: expand the per-constituent
                        # list and tag with a unique merged_id so the
                        # weight-split below groups correctly even
                        # when two unrelated merged fits happen to
                        # share the same centroid weight w (which
                        # otherwise collides their constituents into
                        # one group and undercounts each by 1/N).
                        merged_id = id(rm)
                        for c in constituents:
                            iso_merged_constituents.append({
                                **c,
                                "merged_weight": w,
                                "merged_id": merged_id,
                            })
                    elif mode == "merged":
                        # Legacy/intermediate-version save: marked
                        # merged but no constituents persisted (e.g.
                        # written between round 3 and round 4).
                        # Treat as Manual with the audit σ -- it's
                        # an independent uncertainty contribution,
                        # which is the closest honest fallback when
                        # we no longer know the GP's correlation
                        # structure across the merge.
                        log_lines.append(
                            f"WARNING: '{label}' merged result "
                            f"without correction_constituents -- "
                            f"using audit σ as a Manual-style row.")
                        iso_manual_sigmas.append(float(rm.get(
                            "centroid_correction_sigma_mhz", 0.0)
                                                       or 0.0))
                        iso_manual_w.append(w)
                    elif mode == "Manual":
                        iso_manual_sigmas.append(float(rm.get(
                            "centroid_correction_sigma_mhz", 0.0)
                                                       or 0.0))
                        iso_manual_w.append(w)
                    else:  # Auto (default for non-merged applied)
                        try:
                            ts_h = float(
                                rm.get("ts_start", 0) or 0) / 3600.0
                        except (TypeError, ValueError):
                            ts_h = 0.0
                        if ts_h > 0:
                            iso_auto_t.append(ts_h)
                            iso_auto_w.append(w)
                sigma = sigma_fit
            else:
                centroid, sigma = None, None
                run_meta = {}
                for r in ok_results:
                    if str(r.get("run_number", "")) == run_sel:
                        centroid, sigma = extract_centroid(
                            r.get("params_df", {}))
                        run_meta = r.get("run_metadata") or {}
                        break
                if centroid is None:
                    log_lines.append(
                        f"WARNING: '{label}' run {run_sel} has no centroid.")
                    continue
                log_lines.append(
                    f"{label}: run {run_sel}, "
                    f"centroid = {centroid:.4f} ± {sigma:.4f} MHz")
                full_w = (1.0 / max(sigma ** 2, 1e-30)
                          if sigma is not None and sigma > 0 else 1.0)
                iso_total_weight = float(full_w)
                if bool(run_meta.get(
                        "centroid_correction_applied", False)):
                    mode = run_meta.get("centroid_correction_mode")
                    constituents = run_meta.get(
                        "correction_constituents")
                    if mode == "merged" and constituents:
                        merged_id = id(run_meta)
                        for c in constituents:
                            iso_merged_constituents.append({
                                **c,
                                "merged_weight": full_w,
                                "merged_id": merged_id,
                            })
                    elif mode == "merged":
                        # Legacy save without constituents -- see
                        # the weighted-average path's comment.
                        log_lines.append(
                            f"WARNING: '{label}' merged result "
                            f"without correction_constituents -- "
                            f"using audit σ as a Manual-style row.")
                        iso_manual_sigmas.append(float(run_meta.get(
                            "centroid_correction_sigma_mhz", 0.0)
                                                       or 0.0))
                        iso_manual_w.append(full_w)
                    elif mode == "Manual":
                        iso_manual_sigmas.append(float(run_meta.get(
                            "centroid_correction_sigma_mhz", 0.0)
                                                       or 0.0))
                        iso_manual_w.append(full_w)
                    else:
                        ts_single = float(
                            run_meta.get("ts_start", 0) or 0)
                        if ts_single > 0:
                            iso_auto_t.append(ts_single / 3600.0)
                            iso_auto_w.append(full_w)

            isotope_entries[label] = {
                "centroid": centroid,
                "sigma_fit": sigma,
                "sigma_scatter": sigma_scatter,
                "sigma_correction": 0.0,    # filled below if corrector
                "sigma_voltage": 0.0,        # filled below
                "A": e.a_spin.value(),
            }
            # Expand merged-spectrum constituents into the Auto /
            # Manual pools. Each merged result's constituent weights
            # are split (proportional to n_events, equal fallback)
            # so they sum to that merged result's centroid weight --
            # otherwise σ_correction is over-counted by N×. Group
            # by merged_id (unique per merged result) rather than
            # merged_weight so two unrelated merged fits that
            # happen to share the same w don't get folded together
            # and undercounted.
            grouped: dict = {}
            for c in iso_merged_constituents:
                key = c.get("merged_id", id(c))
                grouped.setdefault(key, []).append(c)
            for _, group in grouped.items():
                merged_w = float(group[0].get("merged_weight", 1.0))
                events = [float(c.get("n_events", 0) or 0)
                          for c in group]
                total_events = sum(events)
                n_in_group = len(group)
                for c, n_e in zip(group, events):
                    if total_events > 0:
                        share = n_e / total_events
                    else:
                        share = 1.0 / n_in_group if n_in_group else 0.0
                    cw = merged_w * share
                    if c.get("centroid_correction_mode") == "Manual":
                        iso_manual_sigmas.append(float(c.get(
                            "centroid_correction_sigma_mhz", 0.0)
                                                       or 0.0))
                        iso_manual_w.append(cw)
                    else:
                        try:
                            ts_h = (float(c.get("ts_start", 0) or 0)
                                     / 3600.0)
                        except (TypeError, ValueError):
                            ts_h = 0.0
                        if ts_h > 0:
                            iso_auto_t.append(ts_h)
                            iso_auto_w.append(cw)
            isotope_runs[label] = {
                "auto_t": np.asarray(iso_auto_t, dtype=float),
                "auto_w": np.asarray(iso_auto_w, dtype=float),
                "manual_sigmas": np.asarray(
                    iso_manual_sigmas, dtype=float),
                "manual_w": np.asarray(iso_manual_w, dtype=float),
                "total_weight": iso_total_weight,
            }

            # Collect run metadata for systematic estimation
            pname = e.project_combo.currentText()
            run_metadata_by_project[pname] = [
                r.get("run_metadata", {}) for r in ok_results]
            src = self._get_source_block(project)
            if src:
                cfg = src.get_source_config()
                mass_by_project[pname] = cfg.get("mass", 0.0)

            if e is ref_entry:
                ref_key = label

        if ref_key is None or ref_key not in isotope_entries:
            QMessageBox.warning(self, "Isotope Shifts",
                                "Reference isotope has no valid centroid.")
            return

        # Systematic errors
        log_lines.append("")
        if self._sys_auto_rb.isChecked():
            # Auto-extract laser freq + harmonic from the first source block
            nu_laser = 0.0
            harmonic = 1
            for pname in mass_by_project:
                proj = self._find_project(pname)
                if proj:
                    src = self._get_source_block(proj)
                    if src:
                        cfg = src.get_source_config()
                        nu_laser = cfg.get("ref_freq", 0.0) / 1e6
                        harmonic = cfg.get("harmonic", 1) or 1
                        break
            sys_result = cooler_voltage_systematic(
                run_metadata_by_project, mass_by_project,
                self._sys_charge.value(),
                self._sys_geometry.currentText(),
                nu_laser, harmonic,
            )
            log_lines.append(sys_result["details"])
            self._sys_details_text.setPlainText(sys_result["details"])
            self._sys_details_text.moveCursor(
                self._sys_details_text.textCursor().MoveOperation.Start)
            # Map project-level systematics to isotope labels
            for e in selected:
                label = e.label_edit.text() or f"A={e.a_spin.value()}"
                pname = e.project_combo.currentText()
                if label in isotope_entries and pname in sys_result[
                        "systematic_errors"]:
                    isotope_entries[label]["sigma_voltage"] = (
                        sys_result["systematic_errors"][pname])
        else:
            manual_sys = self._sys_manual_spin.value()
            log_lines.append(
                f"Manual systematic error: {manual_sys:.4f} MHz (applied "
                f"to all isotopes)")
            self._sys_details_text.setPlainText(
                f"Manual: {manual_sys:.4f} MHz")
            for label in isotope_entries:
                isotope_entries[label]["sigma_voltage"] = manual_sys

        # Phase 5: correction propagation. Manual σ propagation does
        # NOT need the GP corrector -- it's just diagonal user-typed
        # σ². So we always run the per-isotope propagation, and only
        # gate Auto-row contributions and inter-isotope cross-cov on
        # a loaded corrector. This matters for disk-reloaded
        # projects (and for legacy mode="merged" rows that get
        # re-routed to the Manual pool) where centroid correction
        # metadata exists in run_metadata even though the panel's
        # GP isn't loaded yet.
        gp_cross_cov: dict[tuple[str, str], float] = {}
        corrector = (self._ref_corr_panel.corrector
                     if self._ref_corr_panel.apply_globally else None)
        any_corrected = any(
            (len(runs["auto_t"]) + len(runs["manual_sigmas"])) > 0
            for runs in isotope_runs.values())
        if any_corrected:
            log_lines.append("")
            log_lines.append("Correction σ propagation:")
            for label, runs in isotope_runs.items():
                n_auto = len(runs["auto_t"])
                n_manual = len(runs["manual_sigmas"])
                if n_auto == 0 and n_manual == 0:
                    continue
                if n_auto > 0 and corrector is None:
                    log_lines.append(
                        f"  {label}: skipped {n_auto} Auto rows -- "
                        f"no GP corrector loaded; propagating "
                        f"{n_manual} Manual rows only.")
                    auto_t_arg, auto_w_arg = (), ()
                else:
                    auto_t_arg = runs["auto_t"]
                    auto_w_arg = runs["auto_w"]
                try:
                    s_corr = mixed_mode_propagated_sigma(
                        corrector,
                        auto_t=auto_t_arg,
                        auto_w=auto_w_arg,
                        manual_sigmas=runs["manual_sigmas"],
                        manual_w=runs["manual_w"],
                        total_weight=runs["total_weight"])
                except Exception as exc:  # noqa: BLE001
                    log_lines.append(
                        f"  {label}: σ propagation failed "
                        f"({type(exc).__name__}: {exc})")
                    continue
                isotope_entries[label]["sigma_correction"] = s_corr
                log_lines.append(
                    f"  {label}: σ_correction = {s_corr:.4f} MHz "
                    f"({len(auto_t_arg)} Auto + {n_manual} Manual)")
            # Cross-cov uses ONLY the Auto pool and only when a
            # corrector is loaded. Manual rows are treated as
            # independent across isotopes (the user-typed σ has no
            # claim on inter-isotope correlation), and Auto rows
            # need the GP posterior to evaluate the cross term.
            ref_runs = isotope_runs.get(ref_key)
            if (corrector is not None and ref_runs is not None
                    and len(ref_runs["auto_t"])):
                for label, runs in isotope_runs.items():
                    if label == ref_key or len(runs["auto_t"]) == 0:
                        continue
                    try:
                        cov = gp_cross_covariance(
                            corrector,
                            runs["auto_t"], runs["auto_w"],
                            ref_runs["auto_t"], ref_runs["auto_w"],
                            total_weight_a=runs["total_weight"],
                            total_weight_b=ref_runs["total_weight"])
                    except Exception as exc:  # noqa: BLE001
                        log_lines.append(
                            f"  Cov({label}, {ref_key}) failed "
                            f"({type(exc).__name__}: {exc})")
                        continue
                    gp_cross_cov[(label, ref_key)] = cov
                    log_lines.append(
                        f"  Cov({label}, {ref_key}) = {cov:.4f} MHz²")

        # Compute shifts
        log_lines.append("")
        shift_data = compute_all_shifts(
            isotope_entries, ref_key, gp_cross_cov=gp_cross_cov)
        self._last_shift_data = shift_data

        log_lines.append(f"Reference isotope: {ref_key}")
        log_lines.append(
            f"{'Isotope':<12} {'A':>4}  {'Centroid':>12}  "
            f"{'sig_stat':>10}  {'sig_sys':>10}  "
            f"{'delta_nu':>12}  {'dnu_stat':>10}  "
            f"{'dnu_sys':>10}  {'dnu_tot':>10}")
        log_lines.append("-" * 105)
        for row in shift_data:
            ref_mark = " *" if row["is_reference"] else ""
            log_lines.append(
                f"{row['label']:<12} {row['A']:>4}  "
                f"{row['centroid']:>12.4f}  {row['sigma_stat']:>10.4f}  "
                f"{row['sigma_sys']:>10.4f}  {row['delta_nu']:>12.4f}  "
                f"{row['sigma_delta_nu_stat']:>10.4f}  "
                f"{row['sigma_delta_nu_sys']:>10.4f}  "
                f"{row['sigma_delta_nu']:>10.4f}{ref_mark}")

        self._last_log = "\n".join(log_lines)
        self._log_text.setPlainText(self._last_log)

        # Build table
        self._show_table(shift_data)

        # Build plot
        self._build_plot(shift_data, selected, ref_key)

    # ═══════════════════════════════════════════════════════════════
    #  Table display
    # ═══════════════════════════════════════════════════════════════

    def _show_table(self, shift_data):
        self._result_table.setRowCount(len(shift_data))
        for i, row in enumerate(shift_data):
            for col, val in enumerate([
                row["label"],
                str(row["A"]),
                f"{row['centroid']:.4f}",
                f"{row.get('sigma_fit', row.get('sigma_stat', 0.0)):.4f}",
                f"{row.get('sigma_scatter', 0.0):.4f}",
                f"{row.get('sigma_correction', 0.0):.4f}",
                f"{row.get('sigma_voltage', row.get('sigma_sys', 0.0)):.4f}",
                f"{row['delta_nu']:.4f}",
                f"{row['sigma_delta_nu_stat']:.4f}",
                f"{row.get('sigma_delta_nu_scatter', 0.0):.4f}",
                f"{row.get('sigma_delta_nu_correction', 0.0):.4f}",
                f"{row['sigma_delta_nu_sys']:.4f}",
                f"{row['sigma_delta_nu']:.4f}",
                "\u2713" if row["is_reference"] else "",
            ]):
                item = QTableWidgetItem(val)
                if col >= 2:  # right-align numeric columns
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
                self._result_table.setItem(i, col, item)

    # ═══════════════════════════════════════════════════════════════
    #  Plot generation
    # ═══════════════════════════════════════════════════════════════

    def _build_plot(self, shift_data, selected_entries, ref_key):
        """Build the combined isotope-shift comparison plot."""
        from gui.shared_widgets import get_plot_type_settings
        s = get_plot_type_settings("isotope_shift_plot")

        self._figure.clear()

        layout = self._layout_combo.currentText()
        content = self._content_combo.currentText()
        x_mode = self._xaxis_combo.currentText()
        show_centroids = self._centroid_cb.isChecked()
        show_ref_line = self._ref_line_cb.isChecked()
        show_arrows = self._arrows_cb.isChecked()
        sort_by_A = self._sort_cb.isChecked()
        hide_zero_bins = self._hide_zero_cb.isChecked()
        model_color = self._model_color_btn._color
        model_alpha = self._model_alpha.value()
        model_z = self._model_zorder.value()
        centroid_color = self._centroid_color_btn._color
        centroid_alpha = self._centroid_alpha.value()
        centroid_z = self._centroid_zorder.value()
        ref_line_color = self._ref_color_btn._color
        ref_alpha = self._ref_alpha.value()
        ref_z = self._ref_zorder.value()
        arrow_color = self._arrow_color_btn._color
        arrow_alpha = self._arrow_alpha.value()
        arrow_z = self._arrow_zorder.value()

        # Build ordered list of (shift_row, entry, project) tuples
        items = []
        for sd in shift_data:
            for e in selected_entries:
                lbl = e.label_edit.text() or f"A={e.a_spin.value()}"
                if lbl == sd["label"]:
                    project = self._find_project(
                        e.project_combo.currentText())
                    items.append((sd, e, project))
                    break

        if sort_by_A:
            items.sort(key=lambda t: t[0]["A"])

        n = len(items)
        if n == 0:
            return

        # Determine reference centroid for relative x-axis
        ref_centroid = 0.0
        for sd, _, _ in items:
            if sd["is_reference"]:
                ref_centroid = sd["centroid"]
                break

        stacked = "Stacked" in layout

        if stacked:
            axes = self._figure.subplots(
                n, 1, sharex=True,
                gridspec_kw={"hspace": 0.05})
            if n == 1:
                axes = [axes]
        else:
            ax_single = self._figure.add_subplot(111)
            axes = [ax_single] * n

        x_offset = ref_centroid if "Relative" in x_mode else 0.0
        ref_plot = ref_centroid - x_offset

        for idx, (sd, entry, project) in enumerate(items):
            ax = axes[idx]
            color = self._get_data_color(idx)
            label = sd["label"]

            # Get run data for plotting
            x_data, y_data, yerr_data = None, None, None
            x_smooth, y_smooth = None, None

            if project and project._last_results:
                run_sel = entry.run_combo.currentText()
                if run_sel == "Weighted Average":
                    for r in project._last_results:
                        if r.get("success"):
                            x_data = np.array(r.get("x", []))
                            y_data = np.array(r.get("y", []))
                            yerr_data = np.array(r.get("yerr", []))
                            x_smooth = np.array(r.get("x_smooth", []))
                            y_smooth = np.array(r.get("y_fit_smooth", []))
                            break
                else:
                    for r in project._last_results:
                        if (r.get("success") and
                                str(r.get("run_number", "")) == run_sel):
                            x_data = np.array(r.get("x", []))
                            y_data = np.array(r.get("y", []))
                            yerr_data = np.array(r.get("yerr", []))
                            x_smooth = np.array(r.get("x_smooth", []))
                            y_smooth = np.array(r.get("y_fit_smooth", []))
                            break

            if x_data is None or len(x_data) == 0:
                continue

            x_plot = x_data - x_offset
            x_sm_plot = (x_smooth - x_offset
                         if x_smooth is not None else None)
            centroid_plot = sd["centroid"] - x_offset

            # Apply zero-bin masking: drop empty data points and break
            # the smooth fit curve over those gaps.
            y_smooth_for_plot = y_smooth
            if hide_zero_bins:
                nz = np.asarray(y_data) > 0
                x_plot_masked = x_plot[nz]
                y_data_masked = y_data[nz]
                yerr_masked = (
                    yerr_data[nz]
                    if (yerr_data is not None
                        and np.ndim(yerr_data) > 0
                        and len(yerr_data) == len(y_data))
                    else yerr_data)
                if (x_sm_plot is not None and y_smooth is not None
                        and len(x_data) >= 2 and len(x_sm_plot) > 0):
                    xs = np.asarray(x_data, dtype=float)
                    edges = np.empty(len(xs) + 1)
                    edges[1:-1] = 0.5 * (xs[:-1] + xs[1:])
                    edges[0] = xs[0] - 0.5 * (xs[1] - xs[0])
                    edges[-1] = xs[-1] + 0.5 * (xs[-1] - xs[-2])
                    bin_idx = np.clip(
                        np.searchsorted(edges, x_smooth) - 1,
                        0, len(xs) - 1)
                    zero_mask = (np.asarray(y_data) == 0)[bin_idx]
                    y_smooth_for_plot = np.where(
                        zero_mask, np.nan, np.asarray(y_smooth))
                x_plot_used, y_data_used, yerr_used = (
                    x_plot_masked, y_data_masked, yerr_masked)
            else:
                x_plot_used, y_data_used, yerr_used = (
                    x_plot, y_data, yerr_data)

            # Plot data
            ax.errorbar(x_plot_used, y_data_used, yerr=yerr_used,
                        fmt=s.get("data_fmt", "."), color=color,
                        ms=s.get("data_ms", 5.0),
                        alpha=s.get("data_alpha", 0.7),
                        capsize=s.get("data_capsize", 3.0),
                        label=f"{label} data",
                        zorder=2)

            # Plot the smooth model fit curve in the configured color
            if ("Model" in content and x_sm_plot is not None and
                    y_smooth_for_plot is not None and len(x_sm_plot) > 0):
                ax.plot(x_sm_plot, y_smooth_for_plot, color=model_color,
                        lw=s.get("fit_lw", 2.0),
                        alpha=model_alpha,
                        label=f"{label} fit",
                        zorder=model_z)
                # Pale shade under the fitted curve (option in Plot
                # Options ▸ Isotope Shifts; "auto" = the model color).
                if s.get("shade_under_fit", False):
                    _shade_c = s.get("shade_color", "auto")
                    if not _shade_c or _shade_c == "auto":
                        _shade_c = model_color
                    ax.fill_between(
                        x_sm_plot, 0.0, y_smooth_for_plot,
                        color=_shade_c,
                        alpha=float(s.get("shade_alpha", 0.12)),
                        lw=0, zorder=1)

            # Centroid dashed line
            if show_centroids:
                ax.axvline(centroid_plot,
                           ls="--", color=centroid_color,
                           alpha=centroid_alpha, lw=1.0,
                           zorder=centroid_z)

            # Reference centroid line across all panels
            if show_ref_line and not sd["is_reference"]:
                ax.axvline(ref_plot,
                           ls=":", color=ref_line_color,
                           alpha=ref_alpha, lw=1.2,
                           zorder=ref_z)

            # Label — optionally boxed (Plot Options ▸ Isotope Shifts)
            # like the classic stacked-spectra figures.
            if stacked:
                ax.set_ylabel("Counts (a.u.)")
                _label_kwargs = dict(
                    transform=ax.transAxes,
                    fontsize=s.get("label_size", 12),
                    fontweight="bold")
                if s.get("boxed_labels", False):
                    _label_kwargs["bbox"] = dict(
                        boxstyle="square,pad=0.3", fc="white",
                        ec="black", lw=1.0, alpha=0.9)
                ax.text(0.02, 0.85, label, **_label_kwargs)
                if idx < n - 1:
                    ax.tick_params(labelbottom=False)
            else:
                ax.legend(fontsize=9, framealpha=0.7)

        # delta-nu arrows (connect each isotope centroid to reference).
        # Drawn as standalone FancyArrowPatch artists (not annotate's
        # implicit arrow) so each endpoint lives in ax.patches and is
        # independently grabbable by the Plot Editor; the label is a
        # separate text artist. Both carry a stable string id (the
        # isotope label) so dragged positions persist across replot.
        from matplotlib.patches import FancyArrowPatch
        self._is_labels = {}
        self._is_arrows = {}
        if show_arrows:
            # code review 2026-06-02, is-overlaid-arrow-ypos: in Overlaid
            # mode all arrows share axes[0]; staggering the y-position per
            # arrow index keeps 3+ isotopes' labels from overprinting.
            _arrow_k = 0
            for idx, (sd, _, _) in enumerate(items):
                if sd["is_reference"]:
                    continue
                aid = sd["label"]
                # Deleted in the plot editor: replot must not resurrect.
                _removed = getattr(self, "_removed_artist_gids", set())
                ax = axes[idx] if stacked else axes[0]
                c_plot = sd["centroid"] - x_offset
                y_lim = ax.get_ylim()
                if stacked:
                    y_pos = y_lim[1] * 0.9
                else:
                    y_pos = y_lim[1] * (0.9 - 0.08 * _arrow_k)
                _arrow_k += 1
                # Endpoints: tail at this isotope's centroid, head at
                # the reference. Honor any saved drag position.
                posA = (c_plot, y_pos)
                posB = (ref_plot, y_pos)
                saved_arrow = self._is_arrow_pos.get(aid)
                if saved_arrow is not None:
                    posA = tuple(saved_arrow[0])
                    posB = tuple(saved_arrow[1])
                if f"isarrow:{aid}" not in _removed:
                    arrow = FancyArrowPatch(
                        posA, posB,
                        arrowstyle="<->",
                        mutation_scale=12,
                        color=arrow_color,
                        alpha=arrow_alpha,
                        lw=s.get("arrow_lw", 1.5),
                        zorder=arrow_z)
                    arrow._is_id = aid
                    arrow.set_gid(f"isarrow:{aid}")
                    ax.add_patch(arrow)
                    self._is_arrows[aid] = arrow

                mid = ((posA[0] + posB[0]) / 2,
                       (posA[1] + posB[1]) / 2 * 1.02
                       if (posA[1] + posB[1]) else 0.0)
                saved_label = self._is_label_pos.get(aid)
                if saved_label is not None:
                    mid = tuple(saved_label)
                # Label the arrow with the isotope shift, its total
                # uncertainty in concise value(err) notation, AND the
                # unit \u2014 e.g. "\u03b4\u03bd = -136.5(14) MHz".
                if f"istext:{aid}" not in _removed:
                    from gui.analysis.helpers import format_value_error
                    _dnu_err = sd.get("sigma_delta_nu")
                    _dnu_txt = "\u03b4\u03bd = " + format_value_error(
                        sd["delta_nu"], _dnu_err, "MHz")
                    txt = ax.text(
                        mid[0], mid[1], _dnu_txt,
                        ha="center", va="bottom", fontsize=8,
                        color=arrow_color)
                    txt._is_id = aid
                    txt.set_gid(f"istext:{aid}")
                    self._is_labels[aid] = txt

        # X-axis label on bottom axes
        bottom_ax = axes[-1] if stacked else axes[0]
        if "Relative" in x_mode:
            bottom_ax.set_xlabel(
                f"Frequency relative to {ref_key} centroid (MHz)",
                fontsize=s.get("label_size", 12))
        else:
            bottom_ax.set_xlabel(
                "Frequency (MHz)", fontsize=s.get("label_size", 12))

        if not stacked:
            bottom_ax.set_ylabel("Counts (a.u.)")

        # tight_layout leaves a lot of empty whitespace around the
        # axes when the canvas is wider than the default figure
        # aspect ratio. Drive the margins directly so the spectra
        # actually fill the panel.
        self._figure.subplots_adjust(
            left=0.07, right=0.995, top=0.985, bottom=0.07,
            hspace=0.06)
        self._canvas.draw_idle()

    # ═══════════════════════════════════════════════════════════════
    #  Plot editing
    # ═══════════════════════════════════════════════════════════════

    def _edit_plot(self):
        # Modeless: a modal exec() blocks the canvas from receiving
        # mouse events, which breaks the editor's drag-to-reposition
        # mode. Holding a reference on self prevents GC.
        from gui.shared_widgets import PlotEditorDialog
        if (getattr(self, "_plot_editor", None) is not None
                and self._plot_editor.isVisible()):
            self._plot_editor.raise_()
            self._plot_editor.activateWindow()
            return
        self._plot_editor = PlotEditorDialog(
            self._figure, self._canvas, parent=self,
            arrow_registry={"labels": self._is_labels,
                            "arrows": self._is_arrows},
            on_position_changed=self._on_artist_moved)

        def _on_editor_closed(_r):
            # Artists deleted in the editor must stay deleted across
            # replots and travel with "send to Results".
            removed = getattr(self._plot_editor, "removed_gids", None)
            if removed:
                if not hasattr(self, "_removed_artist_gids"):
                    self._removed_artist_gids = set()
                self._removed_artist_gids |= set(removed)
            self._canvas.draw_idle()

        self._plot_editor.finished.connect(_on_editor_closed)
        self._plot_editor.show()

    def _on_artist_moved(self, kind, art_id, value):
        """Persist a δν label / arrow drag so it survives replot, save
        and reload. ``kind`` is "label" or "arrow"; ``value`` is the
        text position (x, y) or the arrow endpoints ((ax, ay),
        (bx, by)) in data coords."""
        if kind == "label":
            self._is_label_pos[art_id] = (float(value[0]), float(value[1]))
        elif kind == "arrow":
            (ax_, ay_), (bx_, by_) = value
            self._is_arrow_pos[art_id] = (
                (float(ax_), float(ay_)), (float(bx_), float(by_)))

    # ═══════════════════════════════════════════════════════════════
    #  Push to Results tab
    # ═══════════════════════════════════════════════════════════════

    def _shift_csv_header_and_rows(self):
        """Return ``(header, rows)`` for the isotope-shift CSV.

        The full Phase-5 column set, used by BOTH the auto-saved iteration
        artifact (_push_to_results) and the manual Export Table CSV
        (_export_csv) so the canonical on-disk file never silently drops the
        scatter / correction / covariance budget the tool computed
        (code review 2026-06-02, is-csv-column-mismatch).
        """
        header = [
            "Isotope", "A", "Centroid_MHz",
            # Phase 5: per-isotope sigma breakdown
            "Sigma_fit", "Sigma_scatter", "Sigma_correction", "Sigma_voltage",
            # Legacy aggregated columns kept for downstream tools
            "Sigma_stat", "Sigma_sys", "Sigma_total", "Delta_nu_MHz",
            "Sigma_dnu_stat", "Sigma_dnu_scatter", "Sigma_dnu_correction",
            "Sigma_dnu_sys", "Sigma_delta_nu", "Cov_with_reference_MHz2",
            "Is_reference",
        ]
        rows = []
        for row in self._last_shift_data:
            rows.append([
                row["label"], row["A"], row["centroid"],
                row.get("sigma_fit", row.get("sigma_stat", 0.0)),
                row.get("sigma_scatter", 0.0),
                row.get("sigma_correction", 0.0),
                row.get("sigma_voltage", row.get("sigma_sys", 0.0)),
                row["sigma_stat"], row["sigma_sys"], row["sigma_total"],
                row["delta_nu"], row["sigma_delta_nu_stat"],
                row.get("sigma_delta_nu_scatter", 0.0),
                row.get("sigma_delta_nu_correction", 0.0),
                row["sigma_delta_nu_sys"], row["sigma_delta_nu"],
                row.get("cov_with_reference_mhz2", 0.0),
                row["is_reference"],
            ])
        return header, rows

    def _push_to_results(self):
        """Save IS data and emit results_ready for the Results tab."""
        if not self._last_shift_data:
            QMessageBox.information(self, "Isotope Shifts",
                                    "Compute shifts first.")
            return

        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()
        study = self._study_name.text().strip() or "Isotope_Shift"
        project_name = f"IS_{study}"
        project_dir = os.path.join(base_dir, project_name)

        # Determine iteration
        iter_num = 1
        if os.path.isdir(project_dir):
            existing = sorted([d for d in os.listdir(project_dir)
                               if os.path.isdir(
                                   os.path.join(project_dir, d))])
            if existing:
                try:
                    last = int(existing[-1].split("_")[-1])
                    iter_num = last + 1
                except ValueError:
                    iter_num = len(existing) + 1
        iter_name = f"iter_{iter_num:03d}"
        iter_dir = os.path.join(project_dir, iter_name)
        os.makedirs(iter_dir, exist_ok=True)
        plots_dir = os.path.join(iter_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(iter_dir, "isotope_shifts.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Full Phase-5 column set (shared with the manual Export) so the
            # canonical artifact carries the complete error budget.
            _hdr, _rows = self._shift_csv_header_and_rows()
            w.writerow(_hdr)
            w.writerows(_rows)

        # Save log
        log_path = os.path.join(iter_dir, "computation_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(self._last_log)

        # Save fit report (same as log, for Results tab text display)
        report_path = os.path.join(iter_dir, "fit_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self._last_log)

        # Save plot as PNG
        png_path = os.path.join(plots_dir,
                                "isotope_shift_comparison.png")
        self._figure.savefig(png_path, dpi=150, bbox_inches="tight")

        # Save plot data as NPZ for re-rendering in Results tab
        npz_path = os.path.join(plots_dir,
                                "isotope_shift_comparison.npz")
        self._save_npz(npz_path)

        # Persist any user edits made in the Plot Editor (moved labels,
        # repositioned δν arrows, custom text colors) as a .style.json
        # sidecar — the Results tab applies it after re-rendering.
        try:
            from gui.results_tab import _extract_figure_style, _style_path
            import json as _json
            style = _extract_figure_style(self._figure)
            removed = getattr(self, "_removed_artist_gids", None)
            if removed:
                style["removed_gids"] = sorted(removed)
            with open(_style_path(npz_path), "w", encoding="utf-8") as f:
                _json.dump(style, f, indent=1)
        except Exception:
            pass

        # Build results list for the Results tab
        results = [{
            "success": True,
            "run_number": "IS_summary",
            "run_file": "",
            "report": self._last_log,
            "params_df": {},
            "metadata_df": {},
            "x": [], "y": [], "yerr": [],
            "y_fit": [], "x_smooth": [], "y_fit_smooth": [],
            "residuals": [],
            "diagnostics": {},
            "fwhm": {},
            "peak_positions": {},
            "fit_quality": {},
            "run_metadata": {},
            "harmonic": 0,
        }]
        output_config = {
            "report": True, "params_csv": True,
            "metadata_csv": False, "fit_plots": True,
            "iter_label": iter_name,
        }
        self.results_ready.emit(project_name, results, output_config)
        QMessageBox.information(
            self, "Isotope Shifts",
            f"Results saved to:\n{iter_dir}\n\n"
            f"Also sent to Results tab as '{project_name}'.")

    def _save_npz(self, path):
        """Save current plot data + every IS-tab style/toggle so the
        Results tab can re-render the figure identically.
        """
        save_dict = {
            "plot_type": "isotope_shift",
            "layout": self._layout_combo.currentText(),
            "content": self._content_combo.currentText(),
            "n_isotopes": len(self._last_shift_data),
            "labels": np.array([r["label"] for r in self._last_shift_data]),
            "x_mode": self._xaxis_combo.currentText(),
            "show_centroids": self._centroid_cb.isChecked(),
            "show_ref_line": self._ref_line_cb.isChecked(),
            "show_arrows": self._arrows_cb.isChecked(),
            "sort_by_A": self._sort_cb.isChecked(),
            "hide_zero_bins": self._hide_zero_cb.isChecked(),
            # Style: colors / alphas / zorders for every artist class.
            "model_color":     self._model_color_btn._color,
            "model_alpha":     self._model_alpha.value(),
            "model_zorder":    self._model_zorder.value(),
            "centroid_color":  self._centroid_color_btn._color,
            "centroid_alpha":  self._centroid_alpha.value(),
            "centroid_zorder": self._centroid_zorder.value(),
            "ref_line_color":  self._ref_color_btn._color,
            "ref_alpha":       self._ref_alpha.value(),
            "ref_zorder":      self._ref_zorder.value(),
            "arrow_color":     self._arrow_color_btn._color,
            "arrow_alpha":     self._arrow_alpha.value(),
            "arrow_zorder":    self._arrow_zorder.value(),
        }

        # Find reference index
        for i, r in enumerate(self._last_shift_data):
            if r["is_reference"]:
                save_dict["reference_idx"] = i
                break

        # Per-isotope data + flags
        for i, sd in enumerate(self._last_shift_data):
            save_dict[f"centroid_{i}"] = sd["centroid"]
            save_dict[f"color_{i}"] = self._get_data_color(i)
            save_dict[f"delta_nu_{i}"] = sd["delta_nu"]
            save_dict[f"sigma_delta_nu_{i}"] = float(
                sd.get("sigma_delta_nu") or 0.0)
            save_dict[f"A_{i}"] = sd.get("A", 0)
            save_dict[f"is_reference_{i}"] = bool(sd["is_reference"])

            # Find the matching entry and project to get raw data
            for e in self._entries:
                lbl = e.label_edit.text() or f"A={e.a_spin.value()}"
                if lbl != sd["label"]:
                    continue
                project = self._find_project(
                    e.project_combo.currentText())
                if not project or not project._last_results:
                    break
                run_sel = e.run_combo.currentText()
                for r in project._last_results:
                    if not r.get("success"):
                        continue
                    if (run_sel != "Weighted Average" and
                            str(r.get("run_number", "")) != run_sel):
                        continue
                    save_dict[f"x_{i}"] = np.array(r.get("x", []))
                    save_dict[f"y_{i}"] = np.array(r.get("y", []))
                    save_dict[f"yerr_{i}"] = np.array(r.get("yerr", []))
                    save_dict[f"x_smooth_{i}"] = np.array(
                        r.get("x_smooth", []))
                    save_dict[f"y_smooth_{i}"] = np.array(
                        r.get("y_fit_smooth", []))
                    break
                break

        np.savez(path, **save_dict)

    # ═══════════════════════════════════════════════════════════════
    #  Export / Save
    # ═══════════════════════════════════════════════════════════════

    def _export_csv(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        if not self._last_shift_data:
            QMessageBox.information(self, "Export",
                                    "Compute shifts first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Isotope Shift Table",
            get_last_dir("data", "save"),
            "CSV files (*.csv)")
        if not path:
            return
        remember_last_dir("data", "save", path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            _hdr, _rows = self._shift_csv_header_and_rows()
            w.writerow(_hdr)
            w.writerows(_rows)

    def _save_log(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        if not self._last_log:
            QMessageBox.information(self, "Save Log",
                                    "No log to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Computation Log",
            get_last_dir("data", "save"),
            "Text files (*.txt)")
        if not path:
            return
        remember_last_dir("data", "save", path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._last_log)

    # ═══════════════════════════════════════════════════════════════
    #  Serialization (save/load)
    # ═══════════════════════════════════════════════════════════════

    def to_dict(self):
        """Serialize the IS tab state to a dict for YAML save."""
        entries = []
        for e in self._entries:
            entries.append({
                "project_name": e.project_combo.currentText(),
                "label": e.label_edit.text(),
                "A": e.a_spin.value(),
                "selected": e.select_cb.isChecked(),
                "is_reference": e.ref_radio.isChecked(),
                "run_selection": e.run_combo.currentText(),
            })
        return {
            "study_name": self._study_name.text(),
            "entries": entries,
            "systematics": {
                "mode": ("auto" if self._sys_auto_rb.isChecked()
                         else "manual"),
                "manual_sigma_MHz": self._sys_manual_spin.value(),
                "charge": self._sys_charge.value(),
                "geometry": self._sys_geometry.currentText(),
            },
            "plot_options": {
                "layout": self._layout_combo.currentText(),
                "content": self._content_combo.currentText(),
                "x_axis": self._xaxis_combo.currentText(),
                "show_centroids": self._centroid_cb.isChecked(),
                "show_ref_line": self._ref_line_cb.isChecked(),
                "show_arrows": self._arrows_cb.isChecked(),
                "sort_by_A": self._sort_cb.isChecked(),
                "hide_zero_bins": self._hide_zero_cb.isChecked(),
                "model_color": self._model_color_btn._color,
                "centroid_color": self._centroid_color_btn._color,
                "ref_line_color": self._ref_color_btn._color,
                "arrow_color": self._arrow_color_btn._color,
                "data_color_pool": list(self._color_pool),
                "model_alpha": self._model_alpha.value(),
                "centroid_alpha": self._centroid_alpha.value(),
                "ref_alpha": self._ref_alpha.value(),
                "arrow_alpha": self._arrow_alpha.value(),
                "model_zorder": self._model_zorder.value(),
                "centroid_zorder": self._centroid_zorder.value(),
                "ref_zorder": self._ref_zorder.value(),
                "arrow_zorder": self._arrow_zorder.value(),
            },
            # User-dragged δν label / arrow positions (Edit Plot). Keyed
            # by isotope label so they re-apply after replot / reload.
            "label_positions": {
                k: list(v) for k, v in self._is_label_pos.items()},
            "arrow_positions": {
                k: [list(v[0]), list(v[1])]
                for k, v in self._is_arrow_pos.items()},
            "removed_artist_gids": sorted(
                getattr(self, "_removed_artist_gids", set())),
            "reference_correction": self._ref_corr_panel.to_dict(),
        }

    def from_dict(self, d):
        """Restore the IS tab state from a dict."""
        if not d:
            return
        self._study_name.setText(d.get("study_name", "Isotope_Shift"))

        # Clear existing rows
        while self._iso_table.rowCount() > 0:
            if self._entries:
                entry = self._entries.pop(0)
                self._ref_group.removeButton(entry.ref_radio)
            self._iso_table.removeRow(0)

        # Restore entries
        for ed in d.get("entries", []):
            e = self._add_row(
                project_name=ed.get("project_name", ""),
                label=ed.get("label", ""),
                A=ed.get("A", 0),
                is_ref=ed.get("is_reference", False),
                run_sel=ed.get("run_selection", "Weighted Average"),
            )
            e.select_cb.setChecked(ed.get("selected", True))

        # Restore systematics
        sys_d = d.get("systematics", {})
        if sys_d.get("mode") == "auto":
            self._sys_auto_rb.setChecked(True)
        else:
            self._sys_manual_rb.setChecked(True)
        self._sys_manual_spin.setValue(
            sys_d.get("manual_sigma_MHz", 0.0))
        self._sys_charge.setValue(sys_d.get("charge", 1))
        geo = sys_d.get("geometry", "anti-collinear")
        idx = self._sys_geometry.findText(geo)
        if idx >= 0:
            self._sys_geometry.setCurrentIndex(idx)
        # nu_laser_MHz no longer stored (auto-extracted from source blocks)

        # Restore plot options
        po = d.get("plot_options", {})
        idx = self._layout_combo.findText(
            po.get("layout", "Stacked Subplots"))
        if idx >= 0:
            self._layout_combo.setCurrentIndex(idx)
        idx = self._content_combo.findText(
            po.get("content", "Data + Model"))
        if idx >= 0:
            self._content_combo.setCurrentIndex(idx)
        idx = self._xaxis_combo.findText(
            po.get("x_axis", "Relative to Reference"))
        if idx >= 0:
            self._xaxis_combo.setCurrentIndex(idx)
        self._centroid_cb.setChecked(
            po.get("show_centroids", True))
        self._ref_line_cb.setChecked(
            po.get("show_ref_line", False))
        self._arrows_cb.setChecked(
            po.get("show_arrows", False))
        self._sort_cb.setChecked(
            po.get("sort_by_A", True))
        self._hide_zero_cb.setChecked(
            po.get("hide_zero_bins", False))

        # Restore colors
        for btn, key, default in [
            (self._model_color_btn, "model_color", "#000000"),
            (self._centroid_color_btn, "centroid_color", "#555555"),
            (self._ref_color_btn, "ref_line_color", "#FF0000"),
            (self._arrow_color_btn, "arrow_color", "#333333"),
        ]:
            c = po.get(key, default)
            btn._color = c
            btn.setStyleSheet(
                f"background-color: {c}; border: 1px solid grey;")
        # Restore data color pool
        saved_pool = po.get("data_color_pool")
        if saved_pool and isinstance(saved_pool, list):
            self._color_pool = list(saved_pool)
            self._rebuild_pool_buttons()
        # Restore alpha and z-order
        self._model_alpha.setValue(po.get("model_alpha", 0.85))
        self._centroid_alpha.setValue(po.get("centroid_alpha", 0.6))
        self._ref_alpha.setValue(po.get("ref_alpha", 0.7))
        self._arrow_alpha.setValue(po.get("arrow_alpha", 1.0))
        self._model_zorder.setValue(po.get("model_zorder", 3))
        self._centroid_zorder.setValue(po.get("centroid_zorder", 1))
        self._ref_zorder.setValue(po.get("ref_zorder", 1))
        self._arrow_zorder.setValue(po.get("arrow_zorder", 5))

        # Restore user-dragged δν label / arrow positions (Edit Plot).
        self._is_label_pos = {
            k: (float(v[0]), float(v[1]))
            for k, v in (d.get("label_positions") or {}).items()}
        self._is_arrow_pos = {
            k: ((float(v[0][0]), float(v[0][1])),
                (float(v[1][0]), float(v[1][1])))
            for k, v in (d.get("arrow_positions") or {}).items()}
        self._removed_artist_gids = set(
            d.get("removed_artist_gids") or [])

        # Restore Reference Correction panel (kernel/MCMC/saved corrector
        # + per-file corrections + checked-project state). Done last so
        # the panel sees the final project list.
        self._ref_corr_panel.from_dict(d.get("reference_correction") or {})
