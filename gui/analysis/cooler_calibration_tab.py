"""Cooler-voltage calibration from Yb hyperfine constants.

The cooler voltage sets the beam energy and so the whole Doppler-
corrected frequency axis. If the logged value is off by a constant,
every frequency is slightly wrong -- including a hyperfine A constant,
which is a frequency difference inside one spectrum. Two Yb isotopes
whose A constants are known to sub-Hz therefore measure that offset.

What this tab does, following the Yb II 369.42 nm calibration report:

* scans each Cooler Calibration project's own fit across a grid of
  assumed offsets (gui/analysis/cooler_scan.py), collecting A at each;
* fits A(dV) per run and per isotope, and finds where each isotope
  reproduces its literature value (cls_estimations/cooler_calibration);
* takes the crossing of the two isotope lines as the calibration, and
  the interval between their zero crossings as the range over which
  the analysis is repeated for the cooler-voltage systematic;
* applies the offset to the Source block of the chosen projects.

Layout mirrors the Reference Correction tab: controls on the left,
plot and tables on the right.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from cls_estimations.cooler_calibration import (
    LITERATURE_A, LITERATURE_SOURCE, ScanPoint, calibrate,
)
from gui.analysis.cooler_scan import CoolerScan

#: Parameters worth calibrating on. A_l is the report's choice: the
#: ground-state constant is the precisely known one.
TRACKED = ("Al", "Au")

ISOTOPES = ("171Yb", "173Yb")


def _wrap(text, width=54):
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def draw_calibration(fig, data):
    """Draw the calibration figure from a plain dict.

    Shared by this tab and the Results tab, so a sent figure is the
    figure that was on screen. Keys: per isotope a label, its points
    (dv, dev, sigma) and its pooled line (intercept, slope, band), and
    the crossing.
    """
    import matplotlib.pyplot as plt
    ax = fig.add_subplot(111)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.axhline(0.0, color="#555555", ls="--", lw=1.0, zorder=1,
               label="$A - A_{lit} = 0$")
    grid = np.asarray(data.get("grid", []), dtype=float)
    for i, iso in enumerate(data.get("isotopes", [])):
        c = colors[i % len(colors)]
        ax.errorbar(iso["dv"], iso["dev"], yerr=iso["sigma"], fmt="o",
                    ms=4.5, color=c, elinewidth=1.0, capsize=2.5,
                    zorder=3,
                    label=f"{iso['label']}  ($A_{{lit}}$ = "
                          f"{iso['literature']:.4f} MHz)")
        if grid.size and iso.get("slope") is not None:
            y = iso["intercept"] + iso["slope"] * grid
            ax.plot(grid, y, "-", color=c, lw=1.6, zorder=2)
            band = np.asarray(iso.get("band", []), dtype=float)
            if band.size == grid.size:
                ax.fill_between(grid, y - band, y + band, color=c,
                                alpha=0.22, lw=0, zorder=1)
    dv = data.get("dv")
    if dv is not None:
        sig = data.get("dv_sigma") or 0.0
        ax.plot([dv], [data.get("dv_y", 0.0)], "*", ms=16,
                color="#2e7d32", zorder=5,
                label=f"intersection: $\\Delta V$ = {dv:.2f} "
                      f"$\\pm$ {sig:.2f} V")
    lo, hi = data.get("scan_lo"), data.get("scan_hi")
    if lo is not None and hi is not None and lo != hi:
        ax.axvspan(lo, hi, color="#2e7d32", alpha=0.08, zorder=0,
                   label=f"systematic range [{lo:.2f}, {hi:.2f}] V")
    ax.set_xlabel("Cooler-voltage offset $\\Delta V$ (V)")
    ax.set_ylabel("$A - A_{lit}$ (MHz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(data.get("title", "Cooler-voltage calibration"))
    fig.tight_layout(pad=0.8)
    return ax


class CoolerCalibrationTab(QGroupBox):
    """The tab. ``corrector`` is the last CalibrationResult, or None."""

    results_ready = Signal(str, list, dict)
    #: (offset_v, sigma_v) whenever a calibration is applied.
    calibration_applied = Signal(float, float)

    def __init__(self, analysis_tab, parent=None):
        super().__init__("Cooler Calibration", parent)
        self._analysis_tab = analysis_tab
        self._points: list[ScanPoint] = []
        self._result = None
        self._scan = CoolerScan(self)
        self._scan.progress.connect(self._on_progress)
        self._scan.finished.connect(self._on_scan_done)
        self._scan.failed.connect(self._on_scan_failed)
        self._log_lines: list[str] = []
        self.setObjectName("coolerCalPanel")
        self.setStyleSheet(
            "QGroupBox#coolerCalPanel { margin-top: 18px; }"
            "QGroupBox#coolerCalPanel::title { top: 8px; }")
        self._build_ui()
        if hasattr(analysis_tab, "projects_changed"):
            analysis_tab.projects_changed.connect(self.refresh_projects)

    # ── UI ───────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        root.addWidget(self._main_splitter)

        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, 0, 0, 0)
        self._main_splitter.addWidget(left)

        col.addWidget(QLabel("Calibration projects:"))
        self._proj_table = QTableWidget(0, 4)
        self._proj_table.setHorizontalHeaderLabels(
            ["Use", "Project", "Isotope", "A_lit (MHz)"])
        self._proj_table.verticalHeader().setVisible(False)
        self._proj_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        # Widths measured from real widgets in the live font (see
        # _fit_project_columns): ResizeToContents cannot see what a
        # cell WIDGET needs, so the project names were elided to
        # "1_cal" and the isotope combo overlapped them.
        self._proj_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self._proj_table.horizontalHeader().setStretchLastSection(True)
        self._proj_table.setMaximumHeight(160)
        self._proj_table.setToolTip(_wrap(
            "The Cooler Calibration projects in this session. The "
            "isotope is guessed from each project's mass number; the "
            "literature A comes from " + LITERATURE_SOURCE + ". Edit "
            "the value only if you mean to use a different reference."))
        col.addWidget(self._proj_table)

        grid_box = QGroupBox("Offset scan")
        gl = QHBoxLayout(grid_box)
        gl.addWidget(QLabel("from"))
        self._dv_min = self._spin(-50.0, -5000.0, 5000.0)
        gl.addWidget(self._dv_min)
        gl.addWidget(QLabel("to"))
        self._dv_max = self._spin(50.0, -5000.0, 5000.0)
        gl.addWidget(self._dv_max)
        gl.addWidget(QLabel("V,"))
        self._dv_steps = QSpinBox()
        self._dv_steps.setRange(2, 201)
        self._dv_steps.setValue(15)
        self._dv_steps.setToolTip(_wrap(
            "How many offsets to fit at. Every run is fitted once per "
            "offset, so this multiplies the fitting time: 15 points "
            "over 5 runs is 75 fits."))
        gl.addWidget(self._dv_steps)
        gl.addWidget(QLabel("points"))
        gl.addStretch()
        gl.addWidget(QLabel("track"))
        self._param = QComboBox()
        self._param.addItems(TRACKED)
        self._param.setToolTip(_wrap(
            "Which fitted parameter is compared with literature. A_l "
            "(the lower/ground state constant) is the precisely known "
            "one and what the calibration report uses."))
        gl.addWidget(self._param)
        col.addWidget(grid_box)

        btns = QHBoxLayout()
        refresh = QPushButton("Refresh projects")
        refresh.clicked.connect(self.refresh_projects)
        btns.addWidget(refresh)
        self._run_btn = QPushButton("Run scan")
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 4px 12px; }")
        self._run_btn.setToolTip(_wrap(
            "Fit every ticked project at every offset in the grid, "
            "then work out where the isotopes agree with literature. "
            "Each project must already fit on its own."))
        self._run_btn.clicked.connect(self._start_scan)
        btns.addWidget(self._run_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._scan.stop)
        btns.addWidget(self._stop_btn)
        btns.addStretch()
        col.addLayout(btns)

        self._status = QLabel("(no scan yet)")
        self._status.setWordWrap(True)
        col.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        col.addWidget(self._progress)

        res_box = QGroupBox("Calibration")
        rl = QVBoxLayout(res_box)
        self._result_label = QLabel("—")
        self._result_label.setWordWrap(True)
        self._result_label.setToolTip(_wrap(
            "The offset where the two isotope lines cross, and the "
            "range between their zero crossings. Repeat the analysis "
            "across that range to get the cooler-voltage systematic "
            "on every extracted parameter."))
        rl.addWidget(self._result_label)
        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply offset to projects…")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply_dialog)
        apply_row.addWidget(self._apply_btn)
        self._send_btn = QPushButton("Send to Results")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_to_results)
        apply_row.addWidget(self._send_btn)
        apply_row.addStretch()
        rl.addLayout(apply_row)
        col.addWidget(res_box)
        col.addStretch()
        left.setMinimumWidth(460)

        right = QTabWidget()
        self._main_splitter.addWidget(right)
        self._figure = Figure(figsize=(7.0, 5.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(300)
        right.addTab(self._canvas, "Plot")

        self._run_table = QTableWidget(0, 7)
        self._run_table.setHorizontalHeaderLabels(
            ["Run", "Isotope", "b (MHz/V)", "σ_b", "a − A_lit (MHz)",
             "ΔV* (V)", "σ_ΔV*"])
        self._run_table.verticalHeader().setVisible(False)
        self._run_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._run_table.setToolTip(_wrap(
            "Per run: the slope of A against the assumed offset, how "
            "far the run sits from literature at zero offset, and the "
            "offset at which it would agree."))
        right.addTab(self._run_table, "Per run")

        self._sum_table = QTableWidget(0, 5)
        self._sum_table.setHorizontalHeaderLabels(
            ["Isotope", "N runs", "zero crossing (V)", "σ (V)",
             "weighted ⟨ΔV*⟩ (V)"])
        self._sum_table.verticalHeader().setVisible(False)
        self._sum_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        right.addTab(self._sum_table, "Summary")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        right.addTab(self._log, "Log")
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([520, 1180])
        self.refresh_projects()

    @staticmethod
    def _spin(value, lo, hi, decimals=2):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(value)
        s.setSingleStep(5.0)
        return s

    def showEvent(self, ev):
        super().showEvent(ev)
        # A cell WIDGET is positioned by the view, and one created
        # while the tab was hidden keeps the geometry of whatever the
        # columns were then -- the project name ended up painted
        # under the isotope combo. Rebuild once it is on screen.
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            QTimer.singleShot(0, self.refresh_projects)

    def changeEvent(self, ev):
        from PySide6.QtCore import QEvent
        super().changeEvent(ev)
        if ev.type() == QEvent.Type.FontChange and hasattr(
                self, "_proj_table"):
            # Zoom changes the font; the measured widths follow. A tick
            # later, so every widget has the new font first.
            self._iso_col_w = None
            QTimer.singleShot(0, self.refresh_projects)

    # ── projects ─────────────────────────────────────────────
    def calibration_projects(self):
        return [p for p in getattr(self._analysis_tab, "_projects", [])
                if getattr(p, "is_calibration", False)]

    @staticmethod
    def _mass_of(isotope):
        """The mass number in an isotope label: 171Yb -> 171."""
        import re
        m = re.match(r"(\d+)", str(isotope))
        return int(m.group(1)) if m else 0

    @classmethod
    def _guess_isotope(cls, project):
        """171Yb / 173Yb from the project's mass number, else its name.

        The mass must match in FULL: a project left at the default
        A = 1 once matched "171Yb" on a prefix, so every project was
        guessed as 171 and the second isotope silently vanished from
        the calibration.
        """
        try:
            from gui.analysis.blocks import SourceBlock
            for b in getattr(project, "_blocks", []) or []:
                if isinstance(b, SourceBlock):
                    a = int(b.get_source_config().get("A", 0) or 0)
                    for iso in ISOTOPES:
                        if a and a == cls._mass_of(iso):
                            return iso
                    break
        except Exception:      # noqa: BLE001
            pass
        name = str(getattr(project, "project_name", ""))
        for iso in ISOTOPES:
            if str(cls._mass_of(iso)) in name:
                return iso
        return ISOTOPES[0]

    def refresh_projects(self):
        """Re-read the calibration projects, keeping any edits."""
        kept = {}
        for r in range(self._proj_table.rowCount()):
            name = self._proj_table.item(r, 1).text()
            kept[name] = (
                self._proj_table.cellWidget(r, 0).isChecked(),
                self._proj_table.cellWidget(r, 2).currentText(),
                float(self._proj_table.item(r, 3).text()))
        projects = self.calibration_projects()
        # Rebuilt from empty. Replacing a cell widget leaves the old
        # one parented to the viewport, still visible at the geometry
        # it had -- after a few refreshes the table carried 20 widgets
        # for 4 cells, painting project names under isotope combos.
        for r in range(self._proj_table.rowCount()):
            for c in range(self._proj_table.columnCount()):
                old = self._proj_table.cellWidget(r, c)
                if old is not None:
                    self._proj_table.removeCellWidget(r, c)
                    old.setParent(None)
                    old.deleteLater()
        self._proj_table.setRowCount(0)
        self._proj_table.setRowCount(len(projects))
        for row, p in enumerate(projects):
            name = p.project_name
            use, iso, lit = kept.get(
                name, (True, self._guess_isotope(p), None))
            cb = QCheckBox()
            cb.setChecked(use)
            self._proj_table.setCellWidget(row, 0, cb)
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._proj_table.setItem(row, 1, item)
            combo = QComboBox()
            combo.addItems(ISOTOPES)
            combo.setCurrentText(iso)
            combo.currentTextChanged.connect(
                lambda t, r=row: self._on_isotope_changed(r, t))
            self._proj_table.setCellWidget(row, 2, combo)
            if lit is None:
                lit = LITERATURE_A.get(iso, 0.0)
            self._proj_table.setItem(
                row, 3, QTableWidgetItem(f"{lit:.10f}"))
        self._fit_project_columns()

    def _fit_project_columns(self):
        """Size the columns for the widgets they actually hold."""
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(self._proj_table.font())
        names = [self._proj_table.item(r, 1).text()
                 for r in range(self._proj_table.rowCount())] or ["Project"]
        # Measured once per font: a fresh QComboBox on every refresh
        # leaked one widget per call.
        if getattr(self, "_iso_col_w", None) is None:
            probe = QComboBox(self._proj_table)
            probe.setFont(self._proj_table.font())
            probe.addItems(ISOTOPES)
            self._iso_col_w = probe.sizeHint().width() + 8
            probe.setParent(None)
            probe.deleteLater()
        iso_w = self._iso_col_w
        widths = [
            max(34, fm.horizontalAdvance("Use") + 16),
            max(fm.horizontalAdvance(max(names, key=len)) + 18,
                fm.horizontalAdvance("Project") + 18),
            max(iso_w, fm.horizontalAdvance("Isotope") + 18),
            max(fm.horizontalAdvance("-3497.2400798500") + 18,
                fm.horizontalAdvance("A_lit (MHz)") + 18),
        ]
        for c, w in enumerate(widths):
            self._proj_table.setColumnWidth(c, w)
        # The pane has to be wide enough for the table plus the
        # scrollbar and both frames.
        parent = self.parentWidget()
        left = self._main_splitter.widget(0) if self._main_splitter else None
        if left is not None:
            left.setMinimumWidth(max(460, sum(widths) + 60))

    def _on_isotope_changed(self, row, iso):
        """The literature value follows the isotope unless it was
        deliberately changed."""
        item = self._proj_table.item(row, 3)
        if item is None:
            return
        try:
            current = float(item.text())
        except ValueError:
            current = None
        if current is None or current in LITERATURE_A.values():
            item.setText(f"{LITERATURE_A.get(iso, 0.0):.10f}")

    def _selected(self):
        """``([(project, isotope)], {isotope: A_lit})`` for ticked rows."""
        pairs, lits = [], {}
        by_name = {p.project_name: p
                   for p in self.calibration_projects()}
        for r in range(self._proj_table.rowCount()):
            if not self._proj_table.cellWidget(r, 0).isChecked():
                continue
            name = self._proj_table.item(r, 1).text()
            project = by_name.get(name)
            if project is None:
                continue
            iso = self._proj_table.cellWidget(r, 2).currentText()
            try:
                lits[iso] = float(self._proj_table.item(r, 3).text())
            except ValueError:
                lits[iso] = LITERATURE_A.get(iso, 0.0)
            pairs.append((project, iso))
        return pairs, lits

    def offsets(self):
        lo, hi = self._dv_min.value(), self._dv_max.value()
        return list(np.linspace(lo, hi, self._dv_steps.value()))

    # ── scan ─────────────────────────────────────────────────
    def _start_scan(self):
        pairs, lits = self._selected()
        if len(pairs) < 1:
            QMessageBox.warning(
                self, "Cooler Calibration",
                "Tick at least one Cooler Calibration project.\n\n"
                "Create one from '+ Create Analysis Project' and put "
                "the Yb runs in it.")
            return
        if len({iso for _p, iso in pairs}) < 2:
            QMessageBox.information(
                self, "Cooler Calibration",
                "Only one isotope is ticked. The scan will run and "
                "its zero crossing will be reported, but two isotopes "
                "are needed for a crossing -- which is the "
                "calibration.")
        if self._dv_min.value() == self._dv_max.value():
            QMessageBox.warning(self, "Cooler Calibration",
                                "The offset range has zero width.")
            return
        self._log_lines = [
            f"Scanning {len(pairs)} project(s) over "
            f"{self._dv_steps.value()} offsets "
            f"[{self._dv_min.value():+.2f}, "
            f"{self._dv_max.value():+.2f}] V, tracking "
            f"{self._param.currentText()}.",
            "Each project is fitted with its own settings; only the "
            "cooler offset changes.", ""]
        self._log.setPlainText("\n".join(self._log_lines))
        self._busy(True)
        try:
            self._scan.start(pairs, self.offsets(),
                             parameter=self._param.currentText())
        except (ValueError, RuntimeError) as exc:
            self._busy(False)
            QMessageBox.warning(self, "Cooler Calibration", str(exc))

    def _busy(self, on):
        self._run_btn.setEnabled(not on)
        self._stop_btn.setEnabled(on)
        self._progress.setVisible(on)

    def _on_progress(self, done, total, what):
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status.setText(f"Fitting {done}/{total} — {what}")

    def _on_scan_failed(self, message):
        self._busy(False)
        self._status.setText(f"Scan stopped: {message}")
        self._log_lines.append(f"FAILED: {message}")
        self._log.setPlainText("\n".join(self._log_lines))
        QMessageBox.warning(self, "Cooler Calibration", message)

    def _on_scan_done(self, points):
        self._busy(False)
        self._points = list(points)
        if not points:
            self._status.setText("Scan produced no usable fits.")
            return
        _pairs, lits = self._selected()
        self.compute(lits)

    def compute(self, literature=None):
        """Work the calibration out from the points already scanned."""
        self._result = calibrate(self._points, literature)
        self._populate_tables()
        self._render()
        res = self._result
        if res.dv is not None:
            self._status.setText(
                f"ΔV* = {res.dv:+.2f} ± {res.dv_sigma:.2f} V "
                f"from {len(self._points)} fits")
            rng = res.scan_range
            txt = (f"<b>ΔV* = {res.dv:+.3f} ± {res.dv_sigma:.3f} V</b>"
                   "<br>where the isotope lines cross")
            if rng:
                txt += (f"<br><br>Systematic scan range: "
                        f"[{rng[0]:+.3f}, {rng[1]:+.3f}] V "
                        f"(width {res.scan_width:.3f} V)")
            self._result_label.setText(txt)
            self._apply_btn.setEnabled(True)
        else:
            self._status.setText(res.note or "No crossing.")
            self._result_label.setText(res.note or "—")
            self._apply_btn.setEnabled(False)
        self._send_btn.setEnabled(True)
        self._log_lines.append(self._summary_text())
        self._log.setPlainText("\n".join(self._log_lines))

    def _summary_text(self):
        res = self._result
        if res is None:
            return ""
        out = []
        for iso in res.isotopes:
            out.append(
                f"{iso.isotope}: {len(iso.runs)} run(s), pooled zero "
                f"crossing {iso.zero:+.3f} ± {iso.zero_sigma:.3f} V"
                if iso.zero is not None else
                f"{iso.isotope}: no zero crossing")
            for r in iso.runs:
                if r.dv is None:
                    continue
                out.append(
                    f"   run {r.run}: b = {r.slope:+.4f} ± "
                    f"{r.slope_sigma:.4f} MHz/V, "
                    f"ΔV* = {r.dv:+.2f} ± {r.dv_sigma:.2f} V")
        if res.dv is not None:
            out.append(f"Intersection: {res.dv:+.3f} ± "
                       f"{res.dv_sigma:.3f} V")
        return "\n".join(out)

    # ── tables + plot ────────────────────────────────────────
    def _populate_tables(self):
        res = self._result
        rows = [r for iso in (res.isotopes if res else []) for r in iso.runs]
        self._run_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r.run, r.isotope, f"{r.slope:+.4f}",
                    f"{r.slope_sigma:.4f}",
                    f"{r.offset_deviation:+.3f} ± "
                    f"{r.offset_deviation_sigma:.3f}",
                    "—" if r.dv is None else f"{r.dv:+.2f}",
                    "—" if r.dv_sigma is None else f"{r.dv_sigma:.2f}"]
            for c, v in enumerate(vals):
                self._run_table.setItem(i, c, QTableWidgetItem(str(v)))

        isos = res.isotopes if res else []
        self._sum_table.setRowCount(len(isos))
        for i, iso in enumerate(isos):
            vals = [iso.isotope, str(len(iso.runs)),
                    "—" if iso.zero is None else f"{iso.zero:+.3f}",
                    "—" if iso.zero_sigma is None
                    else f"{iso.zero_sigma:.3f}",
                    "—" if iso.mean_dv is None else
                    f"{iso.mean_dv:+.3f} ± {iso.mean_dv_sigma:.3f}"]
            for c, v in enumerate(vals):
                self._sum_table.setItem(i, c, QTableWidgetItem(str(v)))

    def plot_data(self):
        """The figure's contents as a plain dict (see
        draw_calibration)."""
        res = self._result
        if res is None:
            return {}
        dvs = [p.dv for p in self._points]
        lo, hi = (min(dvs), max(dvs)) if dvs else (-1.0, 1.0)
        grid = np.linspace(lo, hi, 128)
        isos = []
        for iso in res.isotopes:
            pts = [p for p in self._points if p.isotope == iso.isotope]
            isos.append({
                "label": iso.isotope,
                "literature": iso.literature,
                "dv": [p.dv for p in pts],
                "dev": [p.value - iso.literature for p in pts],
                "sigma": [p.sigma for p in pts],
                "intercept": iso.line.intercept,
                "slope": iso.line.slope,
                "band": list(np.asarray(iso.line.sigma_at(grid))),
            })
        d = {"isotopes": isos, "grid": list(grid),
             "scan_lo": res.scan_lo, "scan_hi": res.scan_hi,
             "title": "Cooler-voltage calibration"}
        if res.dv is not None:
            d["dv"] = res.dv
            d["dv_sigma"] = res.dv_sigma
            d["dv_y"] = float(res.isotopes[0].line.value(res.dv))
        return d

    def _render(self):
        self._figure.clear()
        try:
            data = self.plot_data()
            if data:
                draw_calibration(self._figure, data)
        except Exception as exc:                 # noqa: BLE001
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot error:\n{exc}", ha="center",
                    va="center", transform=ax.transAxes)
        self._canvas.draw()

    # ── applying it ──────────────────────────────────────────
    def _apply_targets(self):
        return [p for p in getattr(self._analysis_tab, "_projects", [])
                if not getattr(p, "is_calibration", False)]

    def _apply_dialog(self):
        res = self._result
        if res is None or res.dv is None:
            return
        targets = self._apply_targets()
        if not targets:
            QMessageBox.information(
                self, "Cooler Calibration",
                "There are no sample or reference projects to apply "
                "it to yet.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Apply cooler offset")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            f"Set the Source-block cooler offset to "
            f"<b>{res.dv:+.3f} V</b> in:"))
        lst = QListWidget()
        lst.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        for p in targets:
            it = QListWidgetItem(self._describe(p))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, p.project_name)
            lst.addItem(it)
        lay.addWidget(lst)
        note = QLabel(
            "Their fits must be re-run afterwards: the offset changes "
            "the frequency axis, so existing centroids were taken in "
            "the old frame.")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        wanted = {lst.item(i).data(Qt.ItemDataRole.UserRole)
                  for i in range(lst.count())
                  if lst.item(i).checkState() == Qt.CheckState.Checked}
        done = self.apply_offset(res.dv, [p for p in targets
                                          if p.project_name in wanted])
        QMessageBox.information(
            self, "Cooler Calibration",
            f"Cooler offset {res.dv:+.3f} V set in {done} project(s).\n\n"
            f"Re-run their fits to use it.")

    @staticmethod
    def _describe(project):
        kind = "reference" if getattr(project, "is_reference", False) \
            else "sample"
        return f"{project.project_name}  ({kind})"

    def apply_offset(self, offset, projects):
        """Write *offset* into each project's Source block. Returns how
        many were changed."""
        from gui.analysis.blocks import SourceBlock
        n = 0
        for p in projects:
            for b in getattr(p, "_blocks", []) or []:
                if isinstance(b, SourceBlock):
                    b._cooler_offset.setValue(float(offset))
                    n += 1
                    break
        if n:
            sigma = (self._result.dv_sigma
                     if self._result is not None else 0.0) or 0.0
            self.calibration_applied.emit(float(offset), float(sigma))
        return n

    # ── Results ──────────────────────────────────────────────
    def _send_to_results(self):
        if self._result is None:
            return
        from gui.shared_widgets import get_analysis_dir
        base = get_analysis_dir()
        project_name = "Cooler_Calibration"
        pdir = os.path.join(base, project_name)
        iter_num = 1
        if os.path.isdir(pdir):
            existing = sorted(d for d in os.listdir(pdir)
                              if os.path.isdir(os.path.join(pdir, d)))
            if existing:
                try:
                    iter_num = int(existing[-1].split("_")[-1]) + 1
                except ValueError:
                    iter_num = len(existing) + 1
        iter_name = f"iter_{iter_num:03d}"
        idir = os.path.join(pdir, iter_name)
        plots = os.path.join(idir, "plots")
        os.makedirs(plots, exist_ok=True)

        res = self._result
        report = ["Cooler-voltage calibration", "=" * 40,
                  f"Tracked parameter: {self._param.currentText()}",
                  f"Literature values: {LITERATURE_SOURCE}",
                  f"Offsets: {self._dv_min.value():+.2f} .. "
                  f"{self._dv_max.value():+.2f} V in "
                  f"{self._dv_steps.value()} steps", ""]
        if res.dv is not None:
            report.append(f"Calibrated offset: {res.dv:+.4f} +/- "
                          f"{res.dv_sigma:.4f} V")
        if res.scan_range:
            report.append(
                f"Systematic scan range: [{res.scan_lo:+.4f}, "
                f"{res.scan_hi:+.4f}] V (width {res.scan_width:.4f} V)")
        report += ["", self._summary_text()]
        with open(os.path.join(idir, "fit_report.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(report))

        with open(os.path.join(idir, "per_run.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "isotope", "slope_mhz_per_v",
                        "slope_sigma", "a_minus_lit_mhz",
                        "a_minus_lit_sigma", "dv_star_v",
                        "dv_star_sigma"])
            for iso in res.isotopes:
                for r in iso.runs:
                    w.writerow([r.run, r.isotope, f"{r.slope:.6f}",
                                f"{r.slope_sigma:.6f}",
                                f"{r.offset_deviation:.6f}",
                                f"{r.offset_deviation_sigma:.6f}",
                                "" if r.dv is None else f"{r.dv:.6f}",
                                "" if r.dv_sigma is None
                                else f"{r.dv_sigma:.6f}"])

        with open(os.path.join(idir, "scan_points.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "isotope", "cooler_offset_v",
                        "value_mhz", "sigma_mhz"])
            for p in self._points:
                w.writerow([p.run, p.isotope, f"{p.dv:.6f}",
                            f"{p.value:.6f}", f"{p.sigma:.6f}"])

        png = os.path.join(plots, "cooler_calibration.png")
        try:
            self._figure.savefig(png, dpi=150, bbox_inches="tight")
        except Exception:                        # noqa: BLE001
            pass
        np.savez(os.path.join(plots, "cooler_calibration.npz"),
                 plot_type="cooler_calibration",
                 payload=np.array(self.plot_data(), dtype=object))

        results = [{
            "success": True, "run_number": "cooler_calibration",
            "run_file": "", "report": "\n".join(report),
            "params_df": {}, "metadata_df": {},
            "x": [], "y": [], "yerr": [], "y_fit": [],
            "x_smooth": [], "y_fit_smooth": [], "residuals": [],
            "diagnostics": {}, "fwhm": {}, "peak_positions": {},
            "fit_quality": {}, "run_metadata": {}, "harmonic": 0,
        }]
        self.results_ready.emit(
            project_name, results,
            {"report": True, "params_csv": False, "metadata_csv": False,
             "fit_plots": True, "iter_label": iter_name,
             "iter_name": iter_name})
        QMessageBox.information(
            self, "Cooler Calibration",
            f"Calibration written to:\n{idir}\n\nAlso sent to the "
            f"Results tab as '{project_name}'.")

    # ── save / load ──────────────────────────────────────────
    def ui_layout(self):
        from gui.ui_layout import collect
        return collect(main=getattr(self, "_main_splitter", None))

    def apply_ui_layout(self, d):
        from gui.ui_layout import restore
        restore(d, main=getattr(self, "_main_splitter", None))

    def to_dict(self):
        rows = []
        for r in range(self._proj_table.rowCount()):
            rows.append({
                "project": self._proj_table.item(r, 1).text(),
                "use": self._proj_table.cellWidget(r, 0).isChecked(),
                "isotope": self._proj_table.cellWidget(r, 2).currentText(),
                "literature": float(self._proj_table.item(r, 3).text()),
            })
        return {
            "dv_min": self._dv_min.value(),
            "dv_max": self._dv_max.value(),
            "dv_steps": self._dv_steps.value(),
            "parameter": self._param.currentText(),
            "rows": rows,
            # The scan is the expensive part: keep its points so a
            # reopened session shows the calibration without refitting.
            "points": [{"run": p.run, "isotope": p.isotope,
                        "dv": p.dv, "value": p.value,
                        "sigma": p.sigma} for p in self._points],
        }

    def from_dict(self, d):
        if not d:
            return
        self._dv_min.setValue(float(d.get("dv_min", -50.0)))
        self._dv_max.setValue(float(d.get("dv_max", 50.0)))
        self._dv_steps.setValue(int(d.get("dv_steps", 15)))
        idx = self._param.findText(str(d.get("parameter", "Al")))
        if idx >= 0:
            self._param.setCurrentIndex(idx)
        self.refresh_projects()
        wanted = {r.get("project"): r for r in (d.get("rows") or [])}
        for row in range(self._proj_table.rowCount()):
            name = self._proj_table.item(row, 1).text()
            saved = wanted.get(name)
            if not saved:
                continue
            self._proj_table.cellWidget(row, 0).setChecked(
                bool(saved.get("use", True)))
            self._proj_table.cellWidget(row, 2).setCurrentText(
                str(saved.get("isotope", ISOTOPES[0])))
            self._proj_table.item(row, 3).setText(
                f"{float(saved.get('literature', 0.0)):.10f}")
        self._points = [
            ScanPoint(run=str(p.get("run", "?")),
                      isotope=str(p.get("isotope", "")),
                      dv=float(p.get("dv", 0.0)),
                      value=float(p.get("value", 0.0)),
                      sigma=float(p.get("sigma", 0.0)))
            for p in (d.get("points") or [])]
        if self._points:
            _pairs, lits = self._selected()
            self.compute(lits)

    def load_from_file(self, path):
        """Adopt the calibration stored in another session's save."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        section = ((raw.get("analysis") or {}).get("cooler_calibration")
                   if isinstance(raw.get("analysis"), dict) else None)
        section = section or raw.get("cooler_calibration")
        if not section:
            raise ValueError(
                f"{os.path.basename(path)} holds no cooler calibration.")
        self.from_dict(section)
        return self._result
