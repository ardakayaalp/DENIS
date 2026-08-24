"""NIST ASD Browser window (Tools ▸ NIST ASD Browser…).

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Single-instance tool window owned by the main window. Three tabs over
one fetched spectrum: Levels (filter/table/diagram), Lines
(filter/table/stick spectrum) and Scheme Finder (search config, ranked
results, scheme diagram). Fetching and searching run in QThreads;
tables fill from the offline cache first. The whole UI state —
including the ranked results — serializes via to_dict()/from_dict()
into the unified DENIS save file.

Depends on: gui.nist_asd.{data,models,search,plotting},
gui.shared_widgets (spin factories); PySide6, matplotlib.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QPushButton, QScrollArea, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QPlainTextEdit, QAbstractItemView,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from gui.nist_asd import data as nd
from gui.nist_asd import plotting as nplot
from gui.nist_asd.models import Laser, RankedScheme
from gui.nist_asd.search import SchemeSearchConfig, SchemeSearcher


class _FetchWorker(QThread):
    """Fetch lines + levels for one spectrum (cache-first)."""
    done = Signal(object, object)          # lines_df, levels_df
    error = Signal(str)

    def __init__(self, spectrum: str, refresh: bool, parent=None):
        super().__init__(parent)
        self._spectrum = spectrum
        self._refresh = refresh

    def run(self):
        try:
            lines = nd.get_lines(self._spectrum, refresh=self._refresh)
            levels = nd.get_levels(self._spectrum,
                                   refresh=self._refresh)
            self.done.emit(lines, levels)
        except Exception as e:                     # noqa: BLE001
            self.error.emit(str(e))


class _SearchWorker(QThread):
    """Run a SchemeSearcher off the GUI thread (isobar tables are
    resolved in here too, cache-first)."""
    progress = Signal(str)
    done = Signal(object)                  # list[RankedScheme]
    error = Signal(str)

    def __init__(self, cfg: SchemeSearchConfig, lines, levels,
                 parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._lines = lines
        self._levels = levels
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            isobars = {}
            for iso in self._cfg.isobars:
                name = str(iso.get("spectrum", "")).strip()
                if not name:
                    continue
                self.progress.emit(f"Loading isobar {name}...")
                try:
                    isobars[name] = nd.get_lines(name)
                except Exception as e:             # noqa: BLE001
                    self.progress.emit(
                        f"Warning: isobar {name} unavailable ({e}).")
            searcher = SchemeSearcher(
                self._cfg, self._lines, self._levels,
                isobar_lines=isobars,
                progress=self.progress.emit,
                is_cancelled=lambda: self._cancel)
            self.done.emit(searcher.run())
        except Exception as e:                     # noqa: BLE001
            self.error.emit(str(e))


def _num_item(value, text=None):
    it = QTableWidgetItem()
    if value is None or (isinstance(value, float)
                         and not np.isfinite(value)):
        it.setText("")
        it.setData(Qt.ItemDataRole.UserRole, float("-inf"))
    else:
        it.setText(text if text is not None else str(value))
        it.setData(Qt.ItemDataRole.UserRole, float(value))
    return it


class NistAsdWindow(QWidget):
    """The NIST ASD browser + scheme finder tool window."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("NIST ASD Browser")
        self.resize(1380, 880)

        self._lines = pd.DataFrame()
        self._levels = pd.DataFrame()
        self._results: list[RankedScheme] = []
        self._fetch_worker = None
        self._search_worker = None

        outer = QVBoxLayout(self)

        # ── Top bar ──────────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel("Spectrum:"))
        self.spectrum_edit = QLineEdit("Si I")
        self.spectrum_edit.setMaximumWidth(110)
        self.spectrum_edit.setToolTip(
            "NIST spectrum notation: element + ionization stage in "
            "Roman numerals, e.g. 'Si I', 'Ca II', 'Nb I'.")
        top.addWidget(self.spectrum_edit)
        self.fetch_btn = QPushButton("Fetch")
        self.fetch_btn.setToolTip(
            "Load this spectrum — from the local cache when present, "
            "otherwise from NIST (needs internet once).")
        self.fetch_btn.clicked.connect(lambda: self._fetch(False))
        top.addWidget(self.fetch_btn)
        self.refresh_btn = QPushButton("Refresh from NIST")
        self.refresh_btn.setToolTip(
            "Ignore the local cache and re-download this spectrum's "
            "lines and levels from NIST ASD.")
        self.refresh_btn.clicked.connect(lambda: self._fetch(True))
        top.addWidget(self.refresh_btn)
        top.addSpacing(14)
        top.addWidget(QLabel("Wavelengths:"))
        self.medium_combo = QComboBox()
        self.medium_combo.addItems(["Air", "Vacuum"])
        self.medium_combo.setToolTip(
            "Display / laser-matching medium. NIST data is fetched in "
            "vacuum; air values use the standard NIST dispersion "
            "formula locally.")
        self.medium_combo.currentTextChanged.connect(
            lambda _t: self._refresh_all_views())
        top.addWidget(self.medium_combo)
        top.addStretch()
        self.cache_label = QLabel("No spectrum loaded")
        self.cache_label.setObjectName("sectionNote")
        top.addWidget(self.cache_label)
        outer.addLayout(top)

        # ── Tabs ─────────────────────────────────────────────────
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_levels_tab(), "Levels")
        self.tabs.addTab(self._build_lines_tab(), "Lines")
        self.tabs.addTab(self._build_scheme_tab(), "Scheme Finder")

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    # ── Levels tab ──────────────────────────────────────────────

    def _build_levels_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        filt = QHBoxLayout()
        filt.addWidget(QLabel("Config contains:"))
        self.lv_conf_edit = QLineEdit()
        self.lv_conf_edit.setMaximumWidth(120)
        filt.addWidget(self.lv_conf_edit)
        filt.addWidget(QLabel("Term contains:"))
        self.lv_term_edit = QLineEdit()
        self.lv_term_edit.setMaximumWidth(90)
        filt.addWidget(self.lv_term_edit)
        filt.addWidget(QLabel("E (cm⁻¹):"))
        self.lv_emin = QDoubleSpinBox()
        self.lv_emin.setRange(0, 1e7)
        self.lv_emin.setDecimals(1)
        filt.addWidget(self.lv_emin)
        filt.addWidget(QLabel("–"))
        self.lv_emax = QDoubleSpinBox()
        self.lv_emax.setRange(0, 1e7)
        self.lv_emax.setDecimals(1)
        self.lv_emax.setValue(0.0)
        self.lv_emax.setSpecialValueText("no max")
        filt.addWidget(self.lv_emax)
        self.lv_meta_check = QCheckBox("Metastables only")
        self.lv_meta_check.setToolTip(
            "Show only excited levels with no fast allowed (E1) "
            "decay — candidate scheme starting points.")
        filt.addWidget(self.lv_meta_check)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._fill_levels_table)
        filt.addWidget(apply_btn)
        filt.addStretch()
        self.lv_add_start_btn = QPushButton("Add as starting level")
        self.lv_add_start_btn.setToolTip(
            "Add the selected level to the Scheme Finder's "
            "starting-levels list.")
        self.lv_add_start_btn.clicked.connect(self._add_start_from_levels)
        filt.addWidget(self.lv_add_start_btn)
        lay.addLayout(filt)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.levels_table = QTableWidget()
        self.levels_table.setColumnCount(5)
        self.levels_table.setHorizontalHeaderLabels(
            ["Level (cm⁻¹)", "Configuration", "Term", "J",
             "Unc (cm⁻¹)"])
        self.levels_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.levels_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.levels_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.levels_table.setSortingEnabled(True)
        split.addWidget(self.levels_table)

        plot_host = QWidget()
        pl = QVBoxLayout(plot_host)
        pl.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Plot:"))
        self.lv_plot_combo = QComboBox()
        self.lv_plot_combo.addItems(
            ["Level diagram", "Decay channels (selected level)",
             "Lifetimes vs energy"])
        self.lv_plot_combo.setToolTip(
            "Level diagram — Grotrian view of the filtered levels.\n"
            "Decay channels — every decay of the SELECTED table row "
            "with Aki, λ and branching ratio (pick your detection "
            "channel here).\n"
            "Lifetimes — τ = 1/ΣAki of every upper level; long-lived "
            "outliers are metastable candidates.")
        self.lv_plot_combo.currentIndexChanged.connect(
            lambda _i: self._draw_levels_plot())
        view_row.addWidget(self.lv_plot_combo)
        view_row.addStretch()
        pl.addLayout(view_row)
        # constrained layout re-fits the axes on every draw, so the
        # figure always fills the canvas (no dead white margins).
        self.levels_fig = Figure(figsize=(5, 5), layout="constrained")
        self.levels_canvas = FigureCanvasQTAgg(self.levels_fig)
        # Hover a level bar → full properties as a tooltip.
        self.levels_canvas.mpl_connect("motion_notify_event",
                                       self._on_levels_hover)
        pl.addWidget(NavigationToolbar2QT(self.levels_canvas,
                                          plot_host))
        pl.addWidget(self.levels_canvas, 1)
        split.addWidget(plot_host)
        split.setSizes([620, 700])
        lay.addWidget(split, 1)
        self.levels_table.currentCellChanged.connect(
            lambda *_a: self._on_level_selection_changed())
        return w

    def _filtered_levels(self) -> pd.DataFrame:
        df = self._levels
        if df is None or df.empty:
            return pd.DataFrame()
        if self.lv_meta_check.isChecked():
            df = nd.find_metastable_states(df, self._lines)
            if df.empty:
                return df
        conf = self.lv_conf_edit.text().strip()
        if conf:
            df = df[df["Configuration"].str.contains(conf, case=False,
                                                     regex=False)]
        term = self.lv_term_edit.text().strip()
        if term:
            df = df[df["Term"].str.contains(term, case=False,
                                            regex=False)]
        emin = self.lv_emin.value()
        if emin > 0:
            df = df[df["Level (cm-1)"] >= emin]
        emax = self.lv_emax.value()
        if emax > 0:
            df = df[df["Level (cm-1)"] <= emax]
        return df

    def _fill_levels_table(self):
        df = self._filtered_levels()
        t = self.levels_table
        t.setSortingEnabled(False)
        t.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            e = float(row["Level (cm-1)"])
            it = _num_item(e, f"{e:.3f}")
            t.setItem(r, 0, it)
            t.setItem(r, 1, QTableWidgetItem(
                str(row.get("Configuration", ""))))
            t.setItem(r, 2, QTableWidgetItem(str(row.get("Term", ""))))
            t.setItem(r, 3, QTableWidgetItem(str(row.get("J", ""))))
            unc = row.get("Uncertainty (cm-1)")
            t.setItem(r, 4, _num_item(
                None if pd.isna(unc) else float(unc),
                "" if pd.isna(unc) else f"{float(unc):g}"))
        t.setSortingEnabled(True)
        self._draw_levels_plot(df)
        self.status_label.setText(f"{len(df)} level(s) shown")

    def _selected_level_energy(self):
        row = self.levels_table.currentRow()
        if row < 0:
            return None
        item = self.levels_table.item(row, 0)
        if item is None:
            return None
        try:
            return float(item.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return None

    def _on_level_selection_changed(self):
        # Only the decay-channels view depends on the selection.
        if self.lv_plot_combo.currentIndex() == 1:
            self._draw_levels_plot()

    def _draw_levels_plot(self, df=None):
        ax = (self.levels_fig.axes[0] if self.levels_fig.axes
              else self.levels_fig.add_subplot(111))
        mode = self.lv_plot_combo.currentIndex()
        if mode == 1:
            lvl = self._selected_level_energy()
            if lvl is None:
                ax.clear()
                ax.text(0.5, 0.5, "Select a level in the table",
                        ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                nplot.plot_decay_channels(ax, self._lines, lvl,
                                          self._medium())
        elif mode == 2:
            nplot.plot_lifetimes(ax, self._lines)
        else:
            nplot.plot_levels(
                ax, df if df is not None else self._filtered_levels(),
                lines_df=self._lines)
        self.levels_canvas.draw_idle()

    def _on_levels_hover(self, event):
        """Tooltip with the full level properties when hovering a bar
        in the level diagram."""
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QToolTip
        if event.inaxes is None:
            QToolTip.hideText()
            return
        for line in event.inaxes.lines:
            info = getattr(line, "_nist_info", None)
            if info:
                hit, _ = line.contains(event)
                if hit:
                    QToolTip.showText(QCursor.pos(), info,
                                      self.levels_canvas)
                    return
        QToolTip.hideText()

    def _add_start_from_levels(self):
        row = self.levels_table.currentRow()
        if row < 0:
            return
        item = self.levels_table.item(row, 0)
        if item is None:
            return
        energy = float(item.data(Qt.ItemDataRole.UserRole))
        r = self.start_table.rowCount()
        self.start_table.insertRow(r)
        self.start_table.setItem(r, 0, _num_item(energy,
                                                 f"{energy:.3f}"))
        self.start_table.setItem(r, 1, QTableWidgetItem("1.0"))
        self.tabs.setCurrentIndex(2)

    # ── Lines tab ───────────────────────────────────────────────

    def _build_lines_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        filt = QHBoxLayout()
        filt.addWidget(QLabel("λ (nm):"))
        self.ln_wlmin = QDoubleSpinBox()
        self.ln_wlmin.setRange(0, 1e6)
        self.ln_wlmin.setDecimals(2)
        filt.addWidget(self.ln_wlmin)
        filt.addWidget(QLabel("–"))
        self.ln_wlmax = QDoubleSpinBox()
        self.ln_wlmax.setRange(0, 1e6)
        self.ln_wlmax.setDecimals(2)
        self.ln_wlmax.setSpecialValueText("no max")
        filt.addWidget(self.ln_wlmax)
        filt.addWidget(QLabel("Min Aki:"))
        self.ln_aki_edit = QLineEdit()
        self.ln_aki_edit.setMaximumWidth(90)
        self.ln_aki_edit.setPlaceholderText("e.g. 1e6")
        self.ln_aki_edit.setToolTip(
            "Minimum Einstein A coefficient (s⁻¹); empty = no cut.")
        filt.addWidget(self.ln_aki_edit)
        filt.addWidget(QLabel("Type:"))
        self.ln_type_combo = QComboBox()
        self.ln_type_combo.addItems(["Any", "Allowed (E1)", "Forbidden"])
        filt.addWidget(self.ln_type_combo)
        filt.addWidget(QLabel("Involves level:"))
        self.ln_level_edit = QLineEdit()
        self.ln_level_edit.setMaximumWidth(100)
        self.ln_level_edit.setPlaceholderText("cm⁻¹ (empty = all)")
        self.ln_level_edit.setToolTip(
            "Keep only transitions with this level (±0.1 cm⁻¹) as "
            "lower or upper state.")
        filt.addWidget(self.ln_level_edit)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._fill_lines_table)
        filt.addWidget(apply_btn)
        filt.addStretch()
        lay.addLayout(filt)

        split = QSplitter(Qt.Orientation.Vertical)
        self.lines_table = QTableWidget()
        cols = ["λ (nm)", "Aki (s⁻¹)", "Acc", "Ei (cm⁻¹)",
                "Ek (cm⁻¹)", "Lower (conf term J)",
                "Upper (conf term J)", "Type"]
        self.lines_table.setColumnCount(len(cols))
        self.lines_table.setHorizontalHeaderLabels(cols)
        self.lines_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.lines_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lines_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.lines_table.setSortingEnabled(True)
        split.addWidget(self.lines_table)

        plot_host = QWidget()
        pl = QVBoxLayout(plot_host)
        pl.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Plot:"))
        self.ln_plot_combo = QComboBox()
        self.ln_plot_combo.addItems(
            ["Stick spectrum", "Line density"])
        self.ln_plot_combo.setToolTip(
            "Stick spectrum — λ vs log Aki of the filtered lines.\n"
            "Line density — transitions per wavelength bin: spectral "
            "congestion / contamination risk at a glance.")
        self.ln_plot_combo.currentIndexChanged.connect(
            lambda _i: self._draw_lines_plot())
        view_row.addWidget(self.ln_plot_combo)
        view_row.addWidget(QLabel("Bin (nm):"))
        self.ln_bin_spin = QDoubleSpinBox()
        self.ln_bin_spin.setRange(0.1, 1000.0)
        self.ln_bin_spin.setValue(10.0)
        self.ln_bin_spin.setDecimals(1)
        self.ln_bin_spin.valueChanged.connect(
            lambda _v: self._draw_lines_plot())
        view_row.addWidget(self.ln_bin_spin)
        view_row.addSpacing(12)

        # Manual axis limits (0/0 on a pair = leave that axis alone).
        def _lim_spin(lo, hi):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(2)
            s.setValue(0.0)
            s.setMaximumWidth(90)
            return s

        view_row.addWidget(QLabel("X:"))
        self.ln_xmin = _lim_spin(0.0, 1e6)
        view_row.addWidget(self.ln_xmin)
        view_row.addWidget(QLabel("–"))
        self.ln_xmax = _lim_spin(0.0, 1e6)
        view_row.addWidget(self.ln_xmax)
        view_row.addWidget(QLabel("Y:"))
        self.ln_ymin = _lim_spin(-1e6, 1e6)
        view_row.addWidget(self.ln_ymin)
        view_row.addWidget(QLabel("–"))
        self.ln_ymax = _lim_spin(-1e6, 1e6)
        view_row.addWidget(self.ln_ymax)
        set_btn = QPushButton("Set limits")
        set_btn.setToolTip(
            "Apply the manual X/Y ranges to the plot. A pair left at "
            "0 – 0 keeps that axis unchanged.")
        set_btn.clicked.connect(
            lambda _c=False: self._apply_lines_limits(draw=True))
        view_row.addWidget(set_btn)
        auto_btn = QPushButton("Auto")
        auto_btn.setToolTip("Reset both axes to automatic limits.")
        auto_btn.clicked.connect(self._auto_lines_limits)
        view_row.addWidget(auto_btn)
        view_row.addStretch()
        pl.addLayout(view_row)
        self.lines_fig = Figure(figsize=(8, 3), layout="constrained")
        self.lines_canvas = FigureCanvasQTAgg(self.lines_fig)
        pl.addWidget(NavigationToolbar2QT(self.lines_canvas,
                                          plot_host))
        pl.addWidget(self.lines_canvas, 1)
        split.addWidget(plot_host)
        split.setSizes([480, 340])
        lay.addWidget(split, 1)
        return w

    def _medium(self) -> str:
        return ("air" if self.medium_combo.currentText() == "Air"
                else "vacuum")

    def _filtered_lines(self) -> pd.DataFrame:
        df = self._lines
        if df is None or df.empty:
            return pd.DataFrame()
        medium = self._medium()
        df = df.copy()
        # Display wavelength: Ritz/ΔE when known, observed otherwise —
        # observed-only NIST lines keep a visible λ in the table.
        df["_wl"] = df.apply(
            lambda r: nd.display_wavelength_nm(r, medium), axis=1)
        if self.ln_wlmin.value() > 0:
            df = df[df["_wl"] >= self.ln_wlmin.value()]
        if self.ln_wlmax.value() > 0:
            df = df[df["_wl"] <= self.ln_wlmax.value()]
        aki_txt = self.ln_aki_edit.text().strip()
        if aki_txt:
            try:
                df = df[df["Aki(s^-1)"] >= float(aki_txt)]
            except ValueError:
                pass
        typ = self.ln_type_combo.currentText()
        if typ.startswith("Allowed"):
            df = df[df["Type"] == "E1"]
        elif typ == "Forbidden":
            df = df[~df["Type"].isin(["E1", ""])]
        lvl_txt = self.ln_level_edit.text().strip()
        if lvl_txt:
            try:
                lvl = float(lvl_txt)
                m = (((df["Ei(cm-1)"] - lvl).abs() < 0.1)
                     | ((df["Ek(cm-1)"] - lvl).abs() < 0.1))
                df = df[m]
            except ValueError:
                pass
        return df

    def _fill_lines_table(self):
        df = self._filtered_lines()
        t = self.lines_table
        t.setSortingEnabled(False)
        t.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            wl = row.get("_wl")
            t.setItem(r, 0, _num_item(
                None if pd.isna(wl) else float(wl),
                "" if pd.isna(wl) else f"{float(wl):.4f}"))
            aki = row.get("Aki(s^-1)")
            t.setItem(r, 1, _num_item(
                None if pd.isna(aki) else float(aki),
                "" if pd.isna(aki) else f"{float(aki):.3e}"))
            t.setItem(r, 2, QTableWidgetItem(str(row.get("Acc", ""))))
            for c, key in ((3, "Ei(cm-1)"), (4, "Ek(cm-1)")):
                v = row.get(key)
                t.setItem(r, c, _num_item(
                    None if pd.isna(v) else float(v),
                    "" if pd.isna(v) else f"{float(v):.3f}"))
            low = " ".join(str(row.get(k, "") or "")
                           for k in ("conf_i", "term_i", "J_i"))
            upp = " ".join(str(row.get(k, "") or "")
                           for k in ("conf_k", "term_k", "J_k"))
            t.setItem(r, 5, QTableWidgetItem(low.strip()))
            t.setItem(r, 6, QTableWidgetItem(upp.strip()))
            t.setItem(r, 7, QTableWidgetItem(str(row.get("Type", ""))))
        t.setSortingEnabled(True)
        self._draw_lines_plot(df)
        self.status_label.setText(f"{len(df)} line(s) shown")

    def _draw_lines_plot(self, df=None):
        if df is None:
            df = self._filtered_lines()
        ax = (self.lines_fig.axes[0] if self.lines_fig.axes
              else self.lines_fig.add_subplot(111))
        if self.ln_plot_combo.currentIndex() == 1:
            nplot.plot_line_density(ax, df, self._medium(),
                                    self.ln_bin_spin.value())
        else:
            nplot.plot_lines_stick(ax, df, self._medium())
        self._apply_lines_limits(draw=False)
        self.lines_canvas.draw_idle()

    def _auto_lines_limits(self):
        """Clear the manual ranges and rescale both axes."""
        for s in (self.ln_xmin, self.ln_xmax,
                  self.ln_ymin, self.ln_ymax):
            s.setValue(0.0)
        self._draw_lines_plot()

    def _apply_lines_limits(self, draw=True):
        """Manual x/y ranges for the Lines-tab plot; a 0–0 pair keeps
        that axis automatic."""
        if not self.lines_fig.axes:
            return
        ax = self.lines_fig.axes[0]
        xmin, xmax = self.ln_xmin.value(), self.ln_xmax.value()
        if xmax > xmin:
            ax.set_xlim(xmin, xmax)
        ymin, ymax = self.ln_ymin.value(), self.ln_ymax.value()
        if ymax > ymin:
            ax.set_ylim(ymin, ymax)
        if draw:
            self.lines_canvas.draw_idle()

    # ── Scheme tab ──────────────────────────────────────────────

    def _build_scheme_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        split = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(split)

        # Left: configuration
        cfg_scroll = QScrollArea()
        cfg_scroll.setWidgetResizable(True)
        cfg_w = QWidget()
        cfg = QVBoxLayout(cfg_w)

        start_grp = QGroupBox("Starting levels")
        sg = QVBoxLayout(start_grp)
        self.auto_discover_check = QCheckBox(
            "Auto-discover (ground + metastables)")
        self.auto_discover_check.setToolTip(
            "Ignore the list below; start from the ground state and "
            "every metastable state (no fast E1 decay).")
        sg.addWidget(self.auto_discover_check)
        self.start_table = QTableWidget(0, 2)
        self.start_table.setHorizontalHeaderLabels(
            ["Level (cm⁻¹)", "Weight"])
        self.start_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.start_table.setMaximumHeight(140)
        sg.addWidget(self.start_table)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_start_row)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(lambda: self.start_table.removeRow(
            self.start_table.currentRow()))
        btn_row.addWidget(rm_btn)
        btn_row.addStretch()
        sg.addLayout(btn_row)
        self.auto_discover_check.toggled.connect(
            self.start_table.setDisabled)
        cfg.addWidget(start_grp)

        laser_grp = QGroupBox("Lasers")
        lg = QVBoxLayout(laser_grp)
        self.broad_check = QCheckBox("Broad wavelength range instead")
        self.broad_check.setToolTip(
            "Search any transition inside one wavelength window "
            "instead of specific laser systems.")
        lg.addWidget(self.broad_check)
        broad_row = QHBoxLayout()
        broad_row.addWidget(QLabel("Range (nm):"))
        self.broad_min = QDoubleSpinBox()
        self.broad_min.setRange(0, 1e5)
        self.broad_min.setDecimals(1)
        self.broad_min.setValue(200.0)
        broad_row.addWidget(self.broad_min)
        broad_row.addWidget(QLabel("–"))
        self.broad_max = QDoubleSpinBox()
        self.broad_max.setRange(0, 1e5)
        self.broad_max.setDecimals(1)
        self.broad_max.setValue(1000.0)
        broad_row.addWidget(self.broad_max)
        broad_row.addStretch()
        lg.addLayout(broad_row)
        self.laser_table = QTableWidget(0, 4)
        self.laser_table.setHorizontalHeaderLabels(
            ["Name", "Center (nm)", "± (nm)", "Role"])
        self.laser_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.laser_table.setMaximumHeight(140)
        lg.addWidget(self.laser_table)
        lbtn = QHBoxLayout()
        ladd = QPushButton("Add laser")
        ladd.clicked.connect(lambda: self._add_laser_row())
        lbtn.addWidget(ladd)
        lrm = QPushButton("Remove")
        lrm.clicked.connect(lambda: self.laser_table.removeRow(
            self.laser_table.currentRow()))
        lbtn.addWidget(lrm)
        lbtn.addStretch()
        lg.addLayout(lbtn)
        self.broad_check.toggled.connect(self.laser_table.setDisabled)
        cfg.addWidget(laser_grp)

        par_grp = QGroupBox("Search parameters")
        pf = QFormLayout(par_grp)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 3)
        self.steps_spin.setValue(2)
        self.steps_spin.setToolTip(
            "Number of laser excitation steps (the last step is the "
            "probe; earlier steps are pumps).")
        pf.addRow("Excitation steps:", self.steps_spin)
        self.aki_pump_edit = QLineEdit()
        self.aki_pump_edit.setPlaceholderText("no cut")
        pf.addRow("Min Aki, pump (s⁻¹):", self.aki_pump_edit)
        self.aki_probe_edit = QLineEdit()
        self.aki_probe_edit.setPlaceholderText("no cut")
        pf.addRow("Min Aki, probe (s⁻¹):", self.aki_probe_edit)
        self.aki_det_edit = QLineEdit()
        self.aki_det_edit.setPlaceholderText("no cut")
        pf.addRow("Min Aki, detection (s⁻¹):", self.aki_det_edit)
        self.br_spin = QDoubleSpinBox()
        self.br_spin.setRange(0.0, 100.0)
        self.br_spin.setValue(1.0)
        self.br_spin.setDecimals(1)
        self.br_spin.setToolTip(
            "Minimum branching ratio of the watched fluorescence "
            "channel (% of all decays of the final level).")
        pf.addRow("Min branching (%):", self.br_spin)
        self.det_combo = QComboBox()
        self.det_combo.addItems(["any", "different", "same"])
        self.det_combo.setToolTip(
            "Constraint between detection and probe wavelengths: "
            "'different' = background-free detection away from the "
            "probe; 'same' = resonant detection.")
        pf.addRow("Detection λ vs probe:", self.det_combo)
        self.det_prox = QDoubleSpinBox()
        self.det_prox.setRange(0.01, 100.0)
        self.det_prox.setValue(1.0)
        pf.addRow("Proximity (nm):", self.det_prox)
        self.orb_probe_edit = QLineEdit()
        self.orb_probe_edit.setPlaceholderText(
            "e.g. s->p, p->d (empty = any)")
        self.orb_probe_edit.setToolTip(
            "Restrict the PROBE step's orbital jump; comma-separated "
            "alternatives.")
        pf.addRow("Probe orbital filter:", self.orb_probe_edit)
        cfg.addWidget(par_grp)

        iso_grp = QGroupBox("Isobaric contaminants")
        ig = QVBoxLayout(iso_grp)
        self.iso_table = QTableWidget(0, 2)
        self.iso_table.setHorizontalHeaderLabels(
            ["Spectrum", "Proximity (nm)"])
        self.iso_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.iso_table.setMaximumHeight(110)
        ig.addWidget(self.iso_table)
        ibtn = QHBoxLayout()
        iadd = QPushButton("Add")
        iadd.clicked.connect(lambda: self._add_iso_row())
        ibtn.addWidget(iadd)
        irm = QPushButton("Remove")
        irm.clicked.connect(lambda: self.iso_table.removeRow(
            self.iso_table.currentRow()))
        ibtn.addWidget(irm)
        ibtn.addStretch()
        ig.addLayout(ibtn)
        cfg.addWidget(iso_grp)

        run_row = QHBoxLayout()
        self.find_btn = QPushButton("Find Schemes")
        self.find_btn.clicked.connect(self._run_search)
        run_row.addWidget(self.find_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_search)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch()
        cfg.addLayout(run_row)
        self.search_status = QLabel("")
        self.search_status.setWordWrap(True)
        self.search_status.setObjectName("sectionNote")
        cfg.addWidget(self.search_status)
        cfg.addStretch()
        cfg_scroll.setWidget(cfg_w)
        cfg_scroll.setMinimumWidth(330)
        cfg_scroll.setMaximumWidth(430)
        split.addWidget(cfg_scroll)

        # Right: results
        right = QSplitter(Qt.Orientation.Vertical)
        self.results_table = QTableWidget()
        cols = ["#", "Score", "Steps λ (nm)", "Steps Aki",
                "Detect λ (nm)", "BR (%)", "⚠", "Notes"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.results_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.currentCellChanged.connect(
            lambda *_a: self._show_selected_scheme())
        right.addWidget(self.results_table)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        plot_host = QWidget()
        pl = QVBoxLayout(plot_host)
        pl.setContentsMargins(0, 0, 0, 0)
        self.scheme_fig = Figure(figsize=(5, 4), layout="constrained")
        self.scheme_canvas = FigureCanvasQTAgg(self.scheme_fig)
        pl.addWidget(NavigationToolbar2QT(self.scheme_canvas,
                                          plot_host))
        pl.addWidget(self.scheme_canvas, 1)
        bottom.addWidget(plot_host)
        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        bottom.addWidget(self.detail_text)
        bottom.setSizes([600, 330])
        right.addWidget(bottom)
        right.setSizes([330, 470])

        export_host = QWidget()
        eh = QVBoxLayout(export_host)
        eh.setContentsMargins(0, 0, 0, 0)
        erow = QHBoxLayout()
        erow.addStretch()
        self.export_btn = QPushButton("Export results CSV…")
        self.export_btn.clicked.connect(self._export_results)
        erow.addWidget(self.export_btn)
        eh.addLayout(erow)
        eh.addWidget(right, 1)
        split.addWidget(export_host)
        split.setSizes([380, 980])
        return w

    def _add_start_row(self, level=0.0, weight=1.0):
        r = self.start_table.rowCount()
        self.start_table.insertRow(r)
        self.start_table.setItem(r, 0, _num_item(float(level),
                                                 f"{float(level):.3f}"))
        self.start_table.setItem(r, 1,
                                 QTableWidgetItem(str(weight)))

    def _add_laser_row(self, name=None, center=400.0, rng=10.0,
                       role="ANY"):
        r = self.laser_table.rowCount()
        self.laser_table.insertRow(r)
        self.laser_table.setItem(
            r, 0, QTableWidgetItem(name or f"Laser {r + 1}"))
        self.laser_table.setItem(r, 1, _num_item(float(center),
                                                 f"{float(center):g}"))
        self.laser_table.setItem(r, 2, _num_item(float(rng),
                                                 f"{float(rng):g}"))
        combo = QComboBox()
        combo.addItems(["ANY", "PUMP", "PROBE"])
        combo.setCurrentText(role if role in ("ANY", "PUMP", "PROBE")
                             else "ANY")
        self.laser_table.setCellWidget(r, 3, combo)

    def _add_iso_row(self, spectrum="", prox=2.0):
        r = self.iso_table.rowCount()
        self.iso_table.insertRow(r)
        self.iso_table.setItem(r, 0, QTableWidgetItem(spectrum))
        self.iso_table.setItem(r, 1, QTableWidgetItem(str(prox)))

    # ── Fetch flow ──────────────────────────────────────────────

    def _fetch(self, refresh: bool):
        spectrum = self.spectrum_edit.text().strip()
        if not spectrum:
            return
        if self._fetch_worker is not None and \
                self._fetch_worker.isRunning():
            return
        self.fetch_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.cache_label.setText(
            ("Downloading from NIST…" if refresh
             else "Loading…") + f" ({spectrum})")
        self._fetch_worker = _FetchWorker(spectrum, refresh, self)
        self._fetch_worker.done.connect(self._on_fetch_done)
        self._fetch_worker.error.connect(self._on_fetch_error)
        self._fetch_worker.finished.connect(
            lambda: (self.fetch_btn.setEnabled(True),
                     self.refresh_btn.setEnabled(True)))
        self._fetch_worker.start()

    def _on_fetch_done(self, lines, levels):
        self.set_data(lines, levels)

    def _on_fetch_error(self, msg):
        self.cache_label.setText("Fetch failed")
        QMessageBox.warning(self, "NIST ASD", f"Fetch failed:\n{msg}")

    def set_data(self, lines: pd.DataFrame, levels: pd.DataFrame):
        """Install fetched tables and refresh every view."""
        self._lines = lines if lines is not None else pd.DataFrame()
        self._levels = levels if levels is not None else pd.DataFrame()
        spectrum = self.spectrum_edit.text().strip()
        info = nd.cache_info(spectrum, "lines")
        if info:
            import datetime as _dt
            when = _dt.datetime.fromtimestamp(
                info.get("fetched_at", 0)).strftime("%Y-%m-%d %H:%M")
            self.cache_label.setText(
                f"{spectrum}: {len(self._lines)} lines / "
                f"{len(self._levels)} levels (cached {when})")
        else:
            self.cache_label.setText(
                f"{spectrum}: {len(self._lines)} lines / "
                f"{len(self._levels)} levels")
        self._refresh_all_views()

    def load_cached_spectrum(self) -> bool:
        """Offline restore: fill from the disk cache if present."""
        spectrum = self.spectrum_edit.text().strip()
        lines = nd.load_cache(spectrum, "lines")
        if lines is None:
            return False
        levels = nd.load_cache(spectrum, "levels")
        if levels is None:
            levels = nd.derive_levels_from_lines(lines)
        self.set_data(lines, levels)
        return True

    def _refresh_all_views(self):
        if not self._lines.empty or not self._levels.empty:
            self._fill_levels_table()
            self._fill_lines_table()
        self._show_selected_scheme()

    # ── Search flow ─────────────────────────────────────────────

    def collect_search_config(self) -> SchemeSearchConfig:
        cfg = SchemeSearchConfig()
        cfg.spectrum = self.spectrum_edit.text().strip()
        cfg.medium = self._medium()
        cfg.auto_discover = self.auto_discover_check.isChecked()
        levels = []
        for r in range(self.start_table.rowCount()):
            try:
                lvl = float(self.start_table.item(r, 0).data(
                    Qt.ItemDataRole.UserRole))
                wt_item = self.start_table.item(r, 1)
                wt = float(wt_item.text()) if wt_item else 1.0
                levels.append({"level": lvl, "weight": wt})
            except (TypeError, ValueError, AttributeError):
                continue
        cfg.starting_levels = levels or [{"level": 0.0, "weight": 1.0}]
        if self.broad_check.isChecked():
            cfg.lasers = []
            cfg.broad_min_nm = self.broad_min.value() or None
            cfg.broad_max_nm = self.broad_max.value() or None
        else:
            lasers = []
            for r in range(self.laser_table.rowCount()):
                try:
                    name_item = self.laser_table.item(r, 0)
                    c = float(self.laser_table.item(r, 1).data(
                        Qt.ItemDataRole.UserRole))
                    rng = float(self.laser_table.item(r, 2).data(
                        Qt.ItemDataRole.UserRole))
                    combo = self.laser_table.cellWidget(r, 3)
                    role = (combo.currentText()
                            if isinstance(combo, QComboBox) else "ANY")
                    lasers.append(Laser(
                        name_item.text() if name_item else f"L{r+1}",
                        c, rng, role).to_dict())
                except (TypeError, ValueError, AttributeError):
                    continue
            cfg.lasers = lasers
        cfg.num_steps = self.steps_spin.value()

        def _flt(edit):
            txt = edit.text().strip()
            try:
                return float(txt) if txt else None
            except ValueError:
                return None

        cfg.aki_min_pump = _flt(self.aki_pump_edit)
        cfg.aki_min_probe = _flt(self.aki_probe_edit)
        cfg.aki_min_detect = _flt(self.aki_det_edit)
        cfg.min_branching_pct = self.br_spin.value()
        cfg.detection_constraint = self.det_combo.currentText()
        cfg.detection_proximity_nm = self.det_prox.value()
        orb = [s.strip() for s in
               self.orb_probe_edit.text().split(",") if s.strip()]
        cfg.orbital_filter_probe = orb
        isobars = []
        for r in range(self.iso_table.rowCount()):
            s_item = self.iso_table.item(r, 0)
            p_item = self.iso_table.item(r, 1)
            name = s_item.text().strip() if s_item else ""
            if not name:
                continue
            try:
                prox = float(p_item.text()) if p_item else 1.0
            except ValueError:
                prox = 1.0
            isobars.append({"spectrum": name, "proximity_nm": prox})
        cfg.isobars = isobars
        return cfg

    def _run_search(self):
        if self._lines.empty:
            QMessageBox.information(
                self, "NIST ASD",
                "Fetch a spectrum first (top bar).")
            return
        if self._search_worker is not None and \
                self._search_worker.isRunning():
            return
        cfg = self.collect_search_config()
        self.find_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.search_status.setText("Searching…")
        self._search_worker = _SearchWorker(
            cfg, self._lines, self._levels, self)
        self._search_worker.progress.connect(self.search_status.setText)
        self._search_worker.done.connect(self._on_search_done)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.finished.connect(
            lambda: (self.find_btn.setEnabled(True),
                     self.cancel_btn.setEnabled(False)))
        self._search_worker.start()

    def _cancel_search(self):
        if self._search_worker is not None:
            self._search_worker.cancel()
            self.search_status.setText("Cancelling…")

    def _on_search_done(self, ranked):
        self.set_results(list(ranked))
        self.search_status.setText(
            f"{len(self._results)} scheme(s) found.")

    def _on_search_error(self, msg):
        self.search_status.setText("Search failed")
        QMessageBox.warning(self, "NIST ASD",
                            f"Scheme search failed:\n{msg}")

    # ── Results ─────────────────────────────────────────────────

    def set_results(self, ranked: list):
        self._results = ranked
        t = self.results_table
        medium = self._medium()
        t.setRowCount(len(ranked))
        for r, rs in enumerate(ranked):
            t.setItem(r, 0, _num_item(r + 1, str(r + 1)))
            t.setItem(r, 1, _num_item(rs.score, f"{rs.score:.2f}"))
            wls, akis = [], []
            for step in rs.scheme.steps:
                wl = nd.wavelength_nm(step.transition, medium)
                wls.append(f"{wl:.2f}")
                aki = step.transition.get("Aki(s^-1)")
                akis.append(f"{aki:.1e}" if aki else "?")
            t.setItem(r, 2, QTableWidgetItem(" → ".join(wls)))
            t.setItem(r, 3, QTableWidgetItem(" → ".join(akis)))
            det_wl = nd.wavelength_nm(rs.scheme.detection, medium)
            t.setItem(r, 4, _num_item(det_wl, f"{det_wl:.2f}"))
            t.setItem(r, 5, _num_item(
                rs.scheme.branching_ratio * 100,
                f"{rs.scheme.branching_ratio * 100:.1f}"))
            t.setItem(r, 6, QTableWidgetItem(
                "⚠" if rs.issues else ""))
            t.setItem(r, 7, QTableWidgetItem("; ".join(rs.warnings)))
        if ranked:
            t.selectRow(0)
        else:
            self._show_selected_scheme()

    def _show_selected_scheme(self):
        row = self.results_table.currentRow()
        ax = (self.scheme_fig.axes[0] if self.scheme_fig.axes
              else self.scheme_fig.add_subplot(111))
        if row < 0 or row >= len(self._results):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            self.scheme_canvas.draw_idle()
            self.detail_text.setPlainText("")
            return
        rs = self._results[row]
        nplot.plot_scheme(ax, rs, self._medium())
        self.scheme_canvas.draw_idle()
        self.detail_text.setPlainText(self._describe(rs))

    def _describe(self, rs: RankedScheme) -> str:
        medium = self._medium()
        out = [f"Score: {rs.score:.3f}",
               f"Start level: {rs.scheme.start_level:.2f} cm-1 "
               f"(weight {rs.scheme.weight:g})", ""]
        for i, step in enumerate(rs.scheme.steps):
            t = step.transition
            wl = nd.wavelength_nm(t, medium)
            out.append(
                f"Step {i + 1}: {t.get('Ei(cm-1)'):.2f} -> "
                f"{t.get('Ek(cm-1)'):.2f} cm-1")
            out.append(f"  λ({medium}) = {wl:.4f} nm   "
                       f"Aki = {t.get('Aki(s^-1)'):.3e} s-1"
                       + (f"   laser: {step.laser}" if step.laser
                          else ""))
            out.append(f"  {t.get('conf_i')} {t.get('term_i')} "
                       f"(J={t.get('J_i')})  ->  {t.get('conf_k')} "
                       f"{t.get('term_k')} (J={t.get('J_k')})")
        d = rs.scheme.detection
        wl_d = nd.wavelength_nm(d, medium)
        out.append("")
        out.append(f"Detection: {d.get('Ek(cm-1)'):.2f} -> "
                   f"{d.get('Ei(cm-1)'):.2f} cm-1")
        out.append(f"  λ({medium}) = {wl_d:.4f} nm   BR = "
                   f"{rs.scheme.branching_ratio * 100:.2f} %")
        if rs.warnings:
            out.append("")
            out.append("Warnings: " + "; ".join(rs.warnings))
        for iss in rs.issues:
            out.append(f"  ⚠ {iss.isobar}: fluorescence at "
                       f"{iss.isobar_wl_nm:.3f} nm near detection "
                       f"({iss.primary_wl_nm:.3f} nm)")
        return "\n".join(out)

    def _export_results(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ranked schemes", "nist_schemes.csv",
            "CSV files (*.csv)")
        if not path:
            return
        medium = self._medium()
        rows = []
        for i, rs in enumerate(self._results):
            row = {"rank": i + 1, "score": rs.score,
                   "start_level_cm1": rs.scheme.start_level,
                   "branching_pct":
                       rs.scheme.branching_ratio * 100.0,
                   "contaminated": bool(rs.issues),
                   "notes": "; ".join(rs.warnings)}
            for j, step in enumerate(rs.scheme.steps):
                t = step.transition
                row[f"step{j+1}_wl_nm"] = nd.wavelength_nm(t, medium)
                row[f"step{j+1}_Aki"] = t.get("Aki(s^-1)")
                row[f"step{j+1}_Ei"] = t.get("Ei(cm-1)")
                row[f"step{j+1}_Ek"] = t.get("Ek(cm-1)")
                row[f"step{j+1}_laser"] = step.laser
            row["detect_wl_nm"] = nd.wavelength_nm(
                rs.scheme.detection, medium)
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        self.status_label.setText(
            f"Exported {len(rows)} scheme(s) to "
            f"{os.path.basename(path)}")

    # ── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "spectrum": self.spectrum_edit.text().strip(),
            "medium": self._medium(),
            "levels_filters": {
                "conf": self.lv_conf_edit.text(),
                "term": self.lv_term_edit.text(),
                "emin": self.lv_emin.value(),
                "emax": self.lv_emax.value(),
                "metastables": self.lv_meta_check.isChecked(),
            },
            "lines_filters": {
                "wlmin": self.ln_wlmin.value(),
                "wlmax": self.ln_wlmax.value(),
                "aki": self.ln_aki_edit.text(),
                "type": self.ln_type_combo.currentText(),
                "level": self.ln_level_edit.text(),
            },
            "search_config": self.collect_search_config().to_dict(),
            "results": [r.to_dict() for r in self._results[:200]],
            "selected": (self.results_table.currentRow()
                         if self._results else None),
        }

    def from_dict(self, d: dict) -> None:
        if not d:
            return
        self.spectrum_edit.setText(str(d.get("spectrum", "Si I")))
        self.medium_combo.setCurrentText(
            "Air" if d.get("medium", "air") == "air" else "Vacuum")
        lf = d.get("levels_filters", {})
        self.lv_conf_edit.setText(str(lf.get("conf", "")))
        self.lv_term_edit.setText(str(lf.get("term", "")))
        self.lv_emin.setValue(float(lf.get("emin", 0.0)))
        self.lv_emax.setValue(float(lf.get("emax", 0.0)))
        self.lv_meta_check.setChecked(bool(lf.get("metastables",
                                                  False)))
        nf = d.get("lines_filters", {})
        self.ln_wlmin.setValue(float(nf.get("wlmin", 0.0)))
        self.ln_wlmax.setValue(float(nf.get("wlmax", 0.0)))
        self.ln_aki_edit.setText(str(nf.get("aki", "")))
        idx = self.ln_type_combo.findText(str(nf.get("type", "Any")))
        if idx >= 0:
            self.ln_type_combo.setCurrentIndex(idx)
        self.ln_level_edit.setText(str(nf.get("level", "")))

        cfg = SchemeSearchConfig.from_dict(
            d.get("search_config", {}))
        self.auto_discover_check.setChecked(cfg.auto_discover)
        self.start_table.setRowCount(0)
        for item in cfg.starting_levels:
            self._add_start_row(item.get("level", 0.0),
                                item.get("weight", 1.0))
        self.broad_check.setChecked(not cfg.lasers)
        if cfg.broad_min_nm:
            self.broad_min.setValue(float(cfg.broad_min_nm))
        if cfg.broad_max_nm:
            self.broad_max.setValue(float(cfg.broad_max_nm))
        self.laser_table.setRowCount(0)
        for laser in cfg.laser_objects():
            self._add_laser_row(laser.name, laser.center_nm,
                                laser.range_nm, laser.role)
        self.steps_spin.setValue(cfg.num_steps)
        self.aki_pump_edit.setText(
            "" if cfg.aki_min_pump is None else f"{cfg.aki_min_pump:g}")
        self.aki_probe_edit.setText(
            "" if cfg.aki_min_probe is None
            else f"{cfg.aki_min_probe:g}")
        self.aki_det_edit.setText(
            "" if cfg.aki_min_detect is None
            else f"{cfg.aki_min_detect:g}")
        self.br_spin.setValue(float(cfg.min_branching_pct))
        idx = self.det_combo.findText(cfg.detection_constraint)
        if idx >= 0:
            self.det_combo.setCurrentIndex(idx)
        self.det_prox.setValue(float(cfg.detection_proximity_nm))
        self.orb_probe_edit.setText(
            ", ".join(cfg.orbital_filter_probe))
        self.iso_table.setRowCount(0)
        for iso in cfg.isobars:
            self._add_iso_row(str(iso.get("spectrum", "")),
                              float(iso.get("proximity_nm", 1.0)))

        # Results restore offline (schemes are self-contained dicts).
        results = [RankedScheme.from_dict(r)
                   for r in d.get("results", [])]
        self.load_cached_spectrum()   # tables refill if cache present
        self.set_results(results)
        sel = d.get("selected")
        if results and isinstance(sel, int) \
                and 0 <= sel < len(results):
            self.results_table.selectRow(sel)
