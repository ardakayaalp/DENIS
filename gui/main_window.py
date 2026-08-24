#!/usr/bin/env python3
"""DENIS main application window, settings dialog, and entry point.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds the top-level QMainWindow hosting the Estimate, Pre-Analysis,
Analysis, and Results tabs, plus the menu bar, application-wide undo
stack, zoom, and the unified YAML save/load system. Also defines the
Settings dialog and the ``main()`` entry point (UI scaling, session
logging, single-instance guard, dark Fusion palette, and splash).

Depends on: gui.shared_widgets (workers, dialogs, settings I/O, icons),
gui.estimate_tab, gui.preanalysis_container, gui.analysis,
gui.results_tab, and lazily gui.split_editor, gui.scan_filter,
gui.missing_files, gui.manual.viewer, cls_estimations.plotting.
"""

__version__ = "1.0.0"

import sys
import os
import logging
import tempfile
from datetime import datetime

os.environ["QT_API"] = "pyside6"
import matplotlib
matplotlib.use("QtAgg")

import yaml

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QDialogButtonBox, QGroupBox,
    QLineEdit, QPushButton, QLabel, QScrollArea, QWidget, QSplashScreen,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QAction, QKeySequence, QShortcut, QIcon, QPixmap, QFont,
    QUndoStack, QColor,
)

from gui.shared_widgets import (
    WorkerThread, SchmidtCalculatorDialog, ShellConfigDialog,
    _load_settings, _save_settings, _DEFAULT_PLOT_SETTINGS,
    _DEFAULT_PLOT_TYPE_SETTINGS, apply_plot_settings, StatusBarManager,
    lucide_icon, _AppShortcutFilter, _TooltipWrapFilter, maybe_convert_path,
)
from gui.estimate_tab import EstimateTab
from gui.preanalysis_container import PreAnalysisContainer
from gui.analysis import AnalysisTab
from gui.results_tab import ResultsTab


def _icon(name):
    """Shortcut for lucide_icon in this module."""
    return lucide_icon(name)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"DENIS v{__version__}")
        _ico = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "icons", "denis.ico")
        if os.path.exists(_ico):
            self.setWindowIcon(QIcon(_ico))
        else:
            self.setWindowIcon(_icon("atom"))
        self.resize(1200, 800)

        self._config_path = None
        # Fingerprint of the last-saved (or freshly loaded) app state;
        # closeEvent only prompts when the current state differs.
        self._saved_fingerprint = None
        # Reference to the last estimate run ({run_dir, config_name,
        # file_tag, palette}); saved with the config so loading it can
        # restore the plots/log without re-running.
        self._last_estimate_run = None
        self._worker = None
        self._schmidt_dialog = None
        self._shell_dialog = None
        self._unit_converter_dialog = None
        self._shg_calculator_dialog = None
        self._sfg_dfg_dialog = None
        self._quick_plot_dialog = None
        self._manual_window = None
        # NIST ASD browser: persistent instance (holds fetched tables
        # and results); created lazily. State loaded from a save file
        # before the first open parks in _pending_nist_state.
        self._nist_browser = None
        self._pending_nist_state = None

        # Zoom state
        app = QApplication.instance()
        self._base_font_pt = app.font().pointSize() or 9
        settings = _load_settings()
        self._zoom_level = max(-5, min(10, settings.get("zoom_level", 0)))

        # Central widget with top-level tabs
        self.tabs = QTabWidget()
        # objectName lets the stylesheet target ONLY this tab bar, so the
        # nested tab widgets inside each section keep their default look.
        self.tabs.setObjectName("MainTabs")
        # Don't draw the QTabBar's default "base" stripe under the
        # tabs — it creates a visible seam between the tab bar and
        # the rolodex pane below.
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.setDocumentMode(True)
        # The rolodex sheet lives in gui.theme so it can follow the
        # active theme; re-applied by the Settings dialog on switch.
        self._apply_main_tabs_style()
        self.setCentralWidget(self.tabs)

        # Tab 1: Estimate
        self.estimate_tab = EstimateTab()
        self.estimate_tab.run_requested.connect(self._on_run)
        self.estimate_tab.run_tab.load_requested.connect(
            self._on_load_estimation)
        self.tabs.addTab(self.estimate_tab, _icon("sigma"), "Estimate")

        # Tab 2: Pre-Analysis
        self.preanalysis_tab = PreAnalysisContainer()
        self.tabs.addTab(self.preanalysis_tab, _icon("eye"), "Pre-Analysis")

        # Tab 3: Analysis
        self.analysis_tab = AnalysisTab()
        self.tabs.addTab(self.analysis_tab, _icon("line-chart"), "Analysis")

        # Tab 4: Results
        self.results_tab = ResultsTab()
        self.tabs.addTab(self.results_tab, _icon("bar-chart-2"), "Results")

        # Status bar with manager (create before signal connections)
        self.statusBar().showMessage("Ready")
        self.status = StatusBarManager(self.statusBar())

        # Wire Analysis results and progress to main window
        self.analysis_tab.results_ready.connect(self._on_analysis_results)
        # Stop the "analysis finished" status-bar flash once the user
        # actually opens the Results tab.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.analysis_tab.fit_progress.connect(
            lambda c, t, s: self.status.show_progress(
                c, t, f"Fitting: {s}"))

        # Undo/Redo stack
        self._undo_stack = QUndoStack(self)
        self._undo_stack.setUndoLimit(200)

        # Application-wide Ctrl+C/V/Z/Y + spinbox undo tracking. The
        # filter snapshots spinbox values on focus-in and pushes
        # undo commands on focus-out, so newly created spinboxes
        # (added isotopes, new analysis blocks, etc.) are picked up
        # automatically without any periodic rescan.
        self._shortcut_filter = _AppShortcutFilter(
            self._undo_stack, parent=self)
        QApplication.instance().installEventFilter(self._shortcut_filter)

        # Word-wrap long plain-text tooltips into a sensible box (app-wide),
        # so multi-sentence tips don't render as one very long single line.
        self._tooltip_filter = _TooltipWrapFilter(self)
        QApplication.instance().installEventFilter(self._tooltip_filter)

        # Menu bar
        self._create_menus()

        # Apply persisted zoom (after all tabs are created)
        if self._zoom_level != 0:
            self._apply_zoom()

        # Snapshot the pristine state so closing an untouched session
        # never prompts to save.
        self._mark_saved()

    def _create_menus(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # ── Save / Save As / Load ──
        # "Save" writes to the currently loaded file without a dialog
        # (falling through to Save As when no file is loaded yet);
        # "Save As..." always opens the file dialog.
        save_action = QAction(_icon("save"), "Save", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save)
        file_menu.addAction(save_action)

        save_as = QAction(_icon("save"), "Save As...", self)
        save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as.triggered.connect(self._save_all)
        file_menu.addAction(save_as)

        load_all = QAction(_icon("folder-open"), "Load All...", self)
        load_all.setShortcut(QKeySequence("Ctrl+Shift+O"))
        load_all.triggered.connect(self._load_all)
        file_menu.addAction(load_all)

        file_menu.addSeparator()

        # ── Save / Load Tab (explicit per-tab export/import) ──
        save_tab = QAction(_icon("file-down"), "Save Tab As...", self)
        save_tab.triggered.connect(self._save_tab)
        file_menu.addAction(save_tab)

        load_tab = QAction(_icon("file-up"), "Load Tab...", self)
        load_tab.setShortcut(QKeySequence("Ctrl+O"))
        load_tab.triggered.connect(self._load_tab)
        file_menu.addAction(load_tab)

        file_menu.addSeparator()

        new_window = QAction(_icon("file-plus"), "New Window", self)
        new_window.setToolTip(
            "Open another clean DENIS instance in its own window. The new "
            "instance skips the single-instance guard, so both windows "
            "run independently.")
        new_window.triggered.connect(self._new_window)
        file_menu.addAction(new_window)

        file_menu.addSeparator()

        settings_action = QAction(_icon("settings"), "Settings...", self)
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction(_icon("x"), "Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")

        undo_action = self._undo_stack.createUndoAction(self, "Undo")
        undo_action.setIcon(_icon("undo"))
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        edit_menu.addAction(undo_action)

        redo_action = self._undo_stack.createRedoAction(self, "Redo")
        redo_action.setIcon(_icon("refresh-cw"))
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()
        split_file_action = QAction(_icon("pencil"), "Split File…", self)
        split_file_action.setToolTip(
            "Open the virtual-split editor: pick an ASDF, place a "
            "voltage cut, and save .vasdf sidecar files for each "
            "sub-range. Load the .vasdf files later in the "
            "Pre-Analysis or Analysis tab to treat them as "
            "independent runs.")
        split_file_action.triggered.connect(self._open_split_file_editor)
        edit_menu.addAction(split_file_action)

        # View menu
        view_menu = menu_bar.addMenu("View")

        zoom_in_action = QAction(_icon("zoom-in"), "Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence("Ctrl+="))
        zoom_in_action.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction(_icon("zoom-out"), "Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_action.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction(_icon("rotate-ccw"), "Reset Zoom", self)
        zoom_reset_action.setShortcut(QKeySequence("Ctrl+0"))
        zoom_reset_action.triggered.connect(self._zoom_reset)
        view_menu.addAction(zoom_reset_action)

        view_menu.addSeparator()
        self._zoom_label_action = QAction("Zoom: 100%", self)
        self._zoom_label_action.setEnabled(False)
        view_menu.addAction(self._zoom_label_action)

        # Zoom is applied after all tabs are created (see __init__)

        # Run menu
        run_menu = menu_bar.addMenu("Run")

        open_output = QAction(_icon("folder-open"), "Open Output Folder", self)
        open_output.triggered.connect(self._open_output)
        run_menu.addAction(open_output)

        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")
        schmidt_action = QAction(_icon("target"), "Schmidt Moment Calculator", self)
        schmidt_action.setStatusTip(
            "Calculate nuclear magnetic moments from the shell model "
            "(single or two-particle coupling)")
        schmidt_action.triggered.connect(self._open_schmidt_calculator)
        tools_menu.addAction(schmidt_action)
        shell_action = QAction(_icon("atom"), "Shell Configuration Plotter", self)
        shell_action.setStatusTip(
            "Visualize nucleon shell filling, parity, and coupling "
            "rules for odd-odd nuclei")
        shell_action.triggered.connect(self._open_shell_plotter)
        tools_menu.addAction(shell_action)

        tools_menu.addSeparator()
        converter_action = QAction(_icon("ruler"), "Unit Converter", self)
        converter_action.setStatusTip(
            "Convert between wavelength and frequency units: "
            "nm, cm\u207b\u00b9, MHz, GHz, THz, eV")
        converter_action.triggered.connect(self._open_unit_converter)
        tools_menu.addAction(converter_action)
        shg_action = QAction(_icon("flask-conical"), "SHG Crystal Angle Calculator", self)
        shg_action.setStatusTip(
            "Compute phase-matching angles for second-harmonic "
            "generation in BBO, KDP, LBO crystals")
        shg_action.triggered.connect(self._open_shg_calculator)
        tools_menu.addAction(shg_action)
        sfg_dfg_action = QAction(_icon("merge"), "SFG / DFG Calculator", self)
        sfg_dfg_action.setStatusTip(
            "Compute the sum- or difference-frequency wavelength "
            "from two input wavelengths")
        sfg_dfg_action.triggered.connect(self._open_sfg_dfg_calculator)
        tools_menu.addAction(sfg_dfg_action)
        tools_menu.addSeparator()
        quickplot_action = QAction(_icon("line-chart"), "Quick Plot", self)
        quickplot_action.setStatusTip(
            "Quick data plotter: load CSV/TSV files or paste data "
            "and plot instantly")
        quickplot_action.triggered.connect(self._open_quick_plot)
        tools_menu.addAction(quickplot_action)

        tools_menu.addSeparator()
        nist_action = QAction(_icon("search"), "NIST ASD Browser…",
                              self)
        nist_action.setStatusTip(
            "Browse NIST Atomic Spectra Database levels and lines, "
            "search and rank multi-step laser excitation schemes, "
            "and plot levels, transitions and schemes")
        nist_action.triggered.connect(self._open_nist_browser)
        tools_menu.addAction(nist_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        manual_action = QAction(_icon("circle-help"), "Documentation", self)
        manual_action.setShortcut(QKeySequence.StandardKey.HelpContents)  # F1
        manual_action.setStatusTip(
            "Open the interactive user manual (contents, equations, "
            "cross-linked topics)")
        manual_action.triggered.connect(self._open_manual)
        help_menu.addAction(manual_action)
        help_menu.addSeparator()
        about_action = QAction(_icon("atom"), "About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ══════════════════════════════════════════════════════════════
    #  Edit > Split File… launcher
    # ══════════════════════════════════════════════════════════════

    def _open_split_file_editor(self):
        """Open the standalone virtual-split editor as a modal dialog."""
        from gui.split_editor import SplitFileEditor
        dlg = SplitFileEditor(self)
        dlg.exec()

    # ══════════════════════════════════════════════════════════════
    #  Zoom
    # ══════════════════════════════════════════════════════════════

    def _zoom_in(self):
        if self._zoom_level < 10:
            self._zoom_level += 1
            self._apply_zoom()

    def _zoom_out(self):
        if self._zoom_level > -5:
            self._zoom_level -= 1
            self._apply_zoom()

    def _zoom_reset(self):
        self._zoom_level = 0
        self._apply_zoom()

    def _apply_main_tabs_style(self):
        """(Re-)apply the theme-matched sheet for the top-level tab
        bar. Called at construction and again on a live theme switch —
        the widget-level sheet would otherwise keep the old theme."""
        from gui.theme import main_tabs_qss
        self.tabs.setStyleSheet(main_tabs_qss())

    def _apply_zoom(self):
        new_size = max(6, self._base_font_pt + self._zoom_level)
        scale = new_size / self._base_font_pt
        app = QApplication.instance()
        font = app.font()
        font.setPointSize(new_size)
        app.setFont(font)

        # Force ALL widgets to recalculate sizes with the new font
        for w in app.allWidgets():
            w.setFont(font)
            w.updateGeometry()

        # Scale analysis block widths (fixed-width in scroll area)
        try:
            for proj in self.analysis_tab._projects:
                for block in proj._blocks:
                    new_w = int(block.BLOCK_WIDTH * scale)
                    block.setFixedWidth(max(new_w, int(
                        block.MIN_BLOCK_WIDTH * scale)))
                    block._current_width = new_w
        except (AttributeError, TypeError):
            pass

        # Re-measure the estimate tab's global-params band: the fonts
        # just changed, so the content's layout minimum did too (the
        # ParametersTab constructor measures the same way).
        try:
            pt = self.estimate_tab.params_tab
            need = pt.global_params.preferred_width()
            pt.global_params.setMinimumWidth(need)
            pt.global_params.setMaximumWidth(need + 44)
        except (AttributeError, TypeError):
            pass

        pct = int(100 * scale)
        self._zoom_label_action.setText(f"Zoom: {pct}%")
        self.statusBar().showMessage(f"Zoom: {pct}%", 2000)
        settings = _load_settings()
        settings["zoom_level"] = self._zoom_level
        _save_settings(settings)

    # ══════════════════════════════════════════════════════════════
    #  Unified Save / Load System
    # ══════════════════════════════════════════════════════════════
    #
    # YAML format:
    #   estimate:       { ... }      # Estimate tab state
    #   preanalysis:    { ... }      # Pre-Analysis tab state
    #   analysis:       { ... }      # Analysis tab state (projects)
    #
    # "Save All" writes all three keys.
    # "Save Tab" writes only the active tab's key.
    # "Load All" / "Load Tab" both call the same loader which
    #   populates whichever keys are found in the file.
    # ══════════════════════════════════════════════════════════════

    def _build_estimate_dict(self):
        """Serialize the Estimate tab into a dict."""
        et = self.estimate_tab
        d = et.params_tab.global_params.to_dict()
        d["isotopes"] = [
            p.to_dict() for p in et.params_tab.isotope_list.panels
        ]
        ref_panel = et.params_tab.isotope_list.get_reference_panel()
        if ref_panel is not None:
            d["reference"] = {
                "A": ref_panel.iso_A.value(),
                "I": ref_panel.iso_I.value(),
                "mu": ref_panel.iso_mu.value(),
                "Q": ref_panel.iso_Q.value(),
                "A_lower_MHz": ref_panel.hfs_Al.value(),
                "B_lower_MHz": ref_panel.hfs_Bl.value(),
                "A_upper_MHz": ref_panel.hfs_Au.value(),
                "B_upper_MHz": ref_panel.hfs_Bu.value(),
            }
        # Reference to the last run's output folder so loading this save
        # can restore the plots/log without re-running the estimation.
        if self._last_estimate_run:
            d["last_run"] = dict(self._last_estimate_run)
        return d

    def _build_preanalysis_dict(self):
        """Serialize the Pre-Analysis tab into a dict."""
        return self.preanalysis_tab._build_config_dict()

    def _build_analysis_dict(self):
        """Serialize the Analysis tab into a dict."""
        return {
            "projects": [p.to_dict(include_iterations=True)
                         for p in self.analysis_tab._projects],
            "isotope_shifts": self.analysis_tab._is_tab.to_dict(),
        }

    def _restore_estimate(self, data):
        """Populate the Estimate tab from a dict."""
        et = self.estimate_tab
        et.params_tab.global_params.from_dict(data)
        et.params_tab.isotope_list.clear_all()
        for iso_dict in data.get("isotopes", []):
            et.params_tab.isotope_list.add_isotope(iso_dict)
        ref = data.get("reference", {})
        if ref:
            found = et.params_tab.isotope_list.set_reference_from_config(ref)
            if not found:
                QMessageBox.information(
                    self, "Reference Not Found",
                    f"The reference isotope (A={ref.get('A')}) was not found "
                    "in the isotopes list.\nPlease add it or mark another "
                    "isotope as the reference.")
        spectrum_only = et.params_tab.global_params.is_spectrum_only()
        et.params_tab.isotope_list.set_spectrum_only(spectrum_only)
        # Restore the previous run's outputs (plots, peak table, log)
        # when the save recorded one and its folder still exists.
        last_run = data.get("last_run")
        if isinstance(last_run, dict):
            self._restore_estimate_outputs(last_run)

    def _restore_preanalysis(self, data):
        """Populate the Pre-Analysis tab from a dict."""
        # _load_config_from_path expects the raw YAML with a
        # "preanalysis" key OR a flat dict. We wrap it.
        self.preanalysis_tab._restore_from_dict(data)

    def _restore_analysis(self, data):
        """Populate the Analysis tab from a dict."""
        # Close existing projects and remove IS tab
        at = self.analysis_tab
        is_tab = at._is_tab
        while at._project_tabs.count() > 0:
            widget = at._project_tabs.widget(0)
            at._project_tabs.removeTab(0)
            if widget is is_tab:
                continue  # don't delete, just remove from tab bar
            if widget in at._projects:
                at._projects.remove(widget)
            widget.deleteLater()
        at._is_tab_visible = False
        # Re-add projects (_add_project will re-show IS tab)
        for pd in data.get("projects", []):
            at._add_project(
                pd.get("project_name", "Project"), config=pd)
        # Restore Isotope Shifts tab state
        is_data = data.get("isotope_shifts")
        if is_data:
            is_tab.from_dict(is_data)
        # Repopulate the Results tab with exactly the iterations this
        # save recorded (no whole-output-directory scan — that stays
        # behind the explicit "Refresh All" button).
        restored = {}
        for p in at._projects:
            names = getattr(p, "_session_iterations", None) or []
            if names:
                restored[p.project_name] = list(names)
        if restored:
            missing = self.results_tab.restore_iterations(restored)
            if missing:
                self.status.show(
                    f"{len(missing)} saved iteration folder(s) no longer "
                    f"exist: {', '.join(missing[:4])}"
                    + ("..." if len(missing) > 4 else ""))

    # ── Dirty tracking ──

    def _build_save_dict(self):
        """The full app state exactly as Save writes it (all tabs +
        the shared registries)."""
        d = {
            "estimate": self._build_estimate_dict(),
            "preanalysis": self._build_preanalysis_dict(),
            "analysis": self._build_analysis_dict(),
        }
        # Per-file scan filters live alongside the tab sections so a
        # filter set in one tab is visible to the other on reload.
        from gui.scan_filter import get_registry
        sf = get_registry().to_dict()
        if sf:
            d["scan_filters"] = sf
        # Same for the per-run voltage calibrations: a run's calibration is a
        # property of the file, not of one tab's analysis, so it lives beside
        # the tab sections rather than inside one of them.
        from gui.calibration import get_registry as _get_cal_registry
        _cal_reg = _get_cal_registry()
        cal = _cal_reg.to_dict()
        if cal:
            d["calibrations"] = cal
        # Which calibration warnings the user has read and dismissed. Kept out
        # of `calibrations` on purpose: that map is the calibration itself and
        # is what the fit subprocess is handed, and whether a human has looked
        # at a warning has no business influencing a fit.
        acks = _cal_reg.acks_to_list()
        if acks:
            d["calibration_acks"] = acks
        # NIST ASD browser: window state incl. ranked schemes. When
        # the tool was never opened this session, pass through any
        # state that came in with a loaded save file so load→save
        # round-trips don't drop it.
        if self._nist_browser is not None:
            d["nist_asd"] = self._nist_browser.to_dict()
        elif self._pending_nist_state:
            d["nist_asd"] = self._pending_nist_state
        return d

    def _state_fingerprint(self):
        """Deterministic hash of the current save-state, or None when
        the state can't be serialized (treated as dirty)."""
        try:
            import hashlib
            text = yaml.dump(self._build_save_dict(), sort_keys=True,
                             default_flow_style=True, allow_unicode=True)
            return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        except Exception:
            logging.getLogger("denis").exception(
                "state fingerprint failed; treating session as dirty")
            return None

    def _mark_saved(self):
        self._saved_fingerprint = self._state_fingerprint()

    def _is_dirty(self):
        current = self._state_fingerprint()
        if current is None or self._saved_fingerprint is None:
            return True
        return current != self._saved_fingerprint

    # ── Close confirmation ──

    def closeEvent(self, event):
        """Prompt to save before closing — but only when something
        actually changed since the last save/load."""
        try:
            dirty = self._is_dirty()
        except Exception:
            dirty = True
        if not dirty:
            event.accept()
            return

        name = (os.path.basename(self._config_path)
                if self._config_path else None)
        msg = QMessageBox(self)
        msg.setWindowTitle("Close DENIS")
        msg.setText(f"Save changes to {name} before closing?" if name
                    else "Save your changes before closing?")
        msg.setIcon(QMessageBox.Icon.Question)
        save_btn = msg.addButton("Save",
                                 QMessageBox.ButtonRole.AcceptRole)
        discard_btn = msg.addButton("Don't Save",
                                    QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(cancel_btn)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == save_btn:
            # Only close once the save has actually written a file. If the
            # user cancels the file dialog, _save() returns False and we
            # keep the window open rather than silently discarding state.
            event.accept() if self._save() else event.ignore()
        elif clicked == discard_btn:
            event.accept()
        else:
            event.ignore()

    # ── Config-dialog directory persistence ──

    def _config_dialog_start_dir(self, action):
        """Return the starting directory for a config Save/Load dialog.

        ``action`` is ``"save"`` or ``"load"``; we keep separate keys
        so users who load from one folder and save to another don't
        have the two locations clobber each other. Falls back to
        ``./configs/`` when the remembered path doesn't exist anymore
        (folder moved, USB unplugged, fresh install).
        """
        key = ("last_config_save_dir" if action == "save"
               else "last_config_load_dir")
        settings = _load_settings()
        d = settings.get(key) or ""
        if d and os.path.isdir(d):
            return d
        return os.path.join(os.getcwd(), "configs")

    def _remember_config_dir(self, action, path):
        """Persist ``dirname(path)`` under the per-action settings key
        so the next Save/Load opens in the same place."""
        key = ("last_config_save_dir" if action == "save"
               else "last_config_load_dir")
        new_dir = os.path.dirname(os.path.abspath(path))
        if not new_dir:
            return
        settings = _load_settings()
        if settings.get(key) == new_dir:
            return
        settings[key] = new_dir
        _save_settings(settings)

    # ── Save / Save As ──

    def _write_save_file(self, path):
        """Write the full app state to ``path`` and record it as the
        current config. Returns True; raises only on I/O errors."""
        self._write_yaml(path, self._build_save_dict())
        self._config_path = path
        self.setWindowTitle(
            f"DENIS v{__version__} - {os.path.basename(path)}")
        self._mark_saved()
        return True

    def _save(self):
        """Save to the currently loaded file without a dialog; falls
        through to Save As when no file is loaded yet. Returns True if
        a file was written."""
        path = self._config_path
        if path and os.path.isdir(os.path.dirname(os.path.abspath(path))):
            try:
                self._write_save_file(path)
            except Exception as e:
                QMessageBox.critical(
                    self, "Save failed",
                    f"Could not write {path}:\n{e}")
                return False
            self.status.show(f"Saved to {os.path.basename(path)}")
            return True
        return self._save_all()

    def _save_all(self):
        """Save As: pick a file, then write all tabs to it. Returns True
        if a file was written, False if the user cancelled the dialog (so
        closeEvent can avoid discarding unsaved state)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save All Configuration",
            self._config_dialog_start_dir("save"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return False
        if not path.lower().endswith(('.yaml', '.yml')):
            path += '.yaml'
        self._remember_config_dir("save", path)
        self._write_save_file(path)
        self.status.show(f"All tabs saved to {os.path.basename(path)}")
        return True

    # ── Load All ──

    def _load_all(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            self._config_dialog_start_dir("load"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        self._remember_config_dir("load", path)
        self._load_from_path(path)

    # ── Save Tab ──

    def _save_tab(self):
        """Write the current tab to a YAML config. Returns True if a file
        was written, False if the tab can't be saved or the user cancelled
        (so closeEvent keeps the window open rather than losing state)."""
        current = self.tabs.currentWidget()
        tab_name = self._tab_key(current)
        if tab_name is None:
            QMessageBox.information(
                self, "Save Tab",
                "This tab does not support saving.")
            return False

        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {self.tabs.tabText(self.tabs.currentIndex())}",
            self._config_dialog_start_dir("save"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return False
        if not path.lower().endswith(('.yaml', '.yml')):
            path += '.yaml'
        self._remember_config_dir("save", path)

        # Merge with existing file contents if present
        existing = {}
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                pass
        existing[tab_name] = self._build_tab_dict(tab_name)
        # Save Tab also refreshes the scan_filters registry section so
        # a per-tab save doesn't silently drop filter edits made
        # earlier in this session. Empty registry is omitted so older
        # YAMLs that never had this key stay clean.
        from gui.scan_filter import get_registry
        sf = get_registry().to_dict()
        if sf:
            existing["scan_filters"] = sf
        elif "scan_filters" in existing:
            existing.pop("scan_filters")
        from gui.calibration import get_registry as _get_cal_registry
        _cal_reg = _get_cal_registry()
        cal = _cal_reg.to_dict()
        if cal:
            existing["calibrations"] = cal
        elif "calibrations" in existing:
            existing.pop("calibrations")
        acks = _cal_reg.acks_to_list()
        if acks:
            existing["calibration_acks"] = acks
        elif "calibration_acks" in existing:
            existing.pop("calibration_acks")
        self._write_yaml(path, existing)
        self.status.show(
            f"{tab_name.title()} saved to {os.path.basename(path)}")
        return True

    # ── Load Tab ──

    def _load_tab(self):
        current = self.tabs.currentWidget()
        tab_name = self._tab_key(current)
        if tab_name is None:
            QMessageBox.information(self, "Load Tab",
                                    "This tab does not support loading.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            self._config_dialog_start_dir("load"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        self._remember_config_dir("load", path)
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load config:\n{e}")
            return
        if not isinstance(raw, dict):
            QMessageBox.critical(self, "Error",
                                 "Invalid config file format.")
            return

        if not self._resolve_missing_files(raw, path):
            return

        # Per-tab Load Tab: also refresh the scan-filter registry if
        # the file carries one. Other tabs' filter entries are left
        # alone (we merge rather than replace) so the user doesn't
        # lose work on an unrelated open file.
        if isinstance(raw.get("scan_filters"), dict):
            from gui.scan_filter import get_registry
            reg = get_registry()
            current = reg.to_dict()
            current.update(raw["scan_filters"])
            reg.from_dict(current)

        # Voltage calibrations: merge, not replace, for the same reason.
        if isinstance(raw.get("calibrations"), dict):
            from gui.calibration import get_registry as _get_cal_registry
            cal_reg = _get_cal_registry()
            merged = cal_reg.to_dict()
            merged.update(raw["calibrations"])
            cal_reg.from_dict(merged)
        if isinstance(raw.get("calibration_acks"), list):
            from gui.calibration import get_registry as _get_cal_registry
            cal_reg = _get_cal_registry()
            cal_reg.acks_from_list(
                list(set(cal_reg.acks_to_list())
                     | set(raw["calibration_acks"])))

        # Handle legacy estimate format at top level
        if tab_name == "estimate" and tab_name not in raw:
            if self._looks_like_estimate(raw):
                self._restore_estimate(raw)
                self.status.show(
                    f"Loaded Estimate from {os.path.basename(path)}")
                return

        if tab_name not in raw:
            available = ", ".join(raw.keys()) or "(none)"
            QMessageBox.information(
                self, "Load Tab",
                f"No '{tab_name}' data found in this file.\n"
                f"Available sections: {available}")
            return

        restore = {"estimate": self._restore_estimate,
                   "preanalysis": self._restore_preanalysis,
                   "analysis": self._restore_analysis}.get(tab_name)
        if restore:
            restore(raw[tab_name])
        self.status.show(
            f"Loaded {tab_name.title()} from {os.path.basename(path)}")

    # ── Shared helpers ──

    def _tab_key(self, widget):
        """Return the YAML key for a tab widget."""
        if widget is self.estimate_tab:
            return "estimate"
        if widget is self.preanalysis_tab:
            return "preanalysis"
        if widget is self.analysis_tab:
            return "analysis"
        return None

    def _build_tab_dict(self, key):
        """Build the dict for a single tab key."""
        if key == "estimate":
            return self._build_estimate_dict()
        if key == "preanalysis":
            return self._build_preanalysis_dict()
        if key == "analysis":
            return self._build_analysis_dict()
        return {}

    def _write_yaml(self, path, d):
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)

    def _resolve_missing_files(self, raw, yaml_path):
        """Run the missing-file dialog flow on a parsed YAML dict.

        Mutates ``raw`` in place: located paths are rewritten and
        unresolved entries are pruned. If any path was located, the
        YAML on disk is rewritten so the next load doesn't re-prompt.

        Returns ``True`` if the load should proceed, ``False`` if the
        user cancelled the missing-file dialog (whole load aborts).
        """
        from gui.missing_files import (
            scan_missing_paths, remap_paths_inplace,
            prune_missing_paths_inplace,
        )

        missing = scan_missing_paths(raw)
        if not missing:
            return True

        # Summary dialog with the list of missing files. Keep the
        # preview short so a 50-file mismatch doesn't blow out the
        # dialog height.
        preview = "\n".join(f"  • {p}" for p in missing[:8])
        if len(missing) > 8:
            preview += f"\n  ...and {len(missing) - 8} more"
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Missing Files")
        msg.setText(
            f"{len(missing)} file(s) referenced in this configuration "
            f"could not be found on disk.")
        msg.setInformativeText(
            f"{preview}\n\nLocate them one by one, skip the missing "
            f"entries (everything else loads as usual), or cancel "
            f"the load.")
        locate_btn = msg.addButton(
            "Locate Files...", QMessageBox.ButtonRole.AcceptRole)
        # Skip is the implicit "anything other than Locate or Cancel"
        # branch -- we only need to install the button, not its handle.
        msg.addButton("Skip Missing",
                      QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(locate_btn)
        msg.exec()
        clicked = msg.clickedButton()

        # Closing the dialog via the window "X" returns clickedButton()==None,
        # which previously fell through to the "Skip Missing" prune-everything
        # branch -- silently stripping every missing reference. Treat the
        # ambiguous close as Cancel (abort the load) instead (code review
        # 2026-06-02, missing-files-dialog-close-x-prunes-everything).
        if clicked is cancel_btn or clicked is None:
            return False

        remap = {}
        if clicked is locate_btn:
            # The next dialog should land somewhere helpful: prefer
            # the parent dir of the just-located file (sibling files
            # often live together), otherwise the parent of the
            # original missing path, otherwise the YAML's directory.
            yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
            last_dir = yaml_dir
            for orig in missing:
                start_dir = (last_dir or
                             os.path.dirname(maybe_convert_path(orig))
                             or yaml_dir)
                ext = os.path.splitext(orig)[1].lower()
                if ext in (".asdf", ".vasdf"):
                    flt = (f"Same type (*{ext});;"
                           f"ASDF/VASDF (*.asdf *.vasdf);;All files (*)")
                else:
                    flt = "All files (*)"
                new_path, _ = QFileDialog.getOpenFileName(
                    self, f"Locate: {os.path.basename(orig)}",
                    start_dir, flt)
                if not new_path:
                    # Cancelling a per-file dialog stops the loop;
                    # everything not yet located stays missing and is
                    # pruned below so the load continues with whatever
                    # was already resolved.
                    break
                remap[orig] = new_path
                last_dir = os.path.dirname(new_path)

        n_replaced = remap_paths_inplace(raw, remap)
        # Prune entries whose paths the user didn't locate. Whether
        # they came via "Skip Missing" or by stopping the locate loop
        # early, the in-memory dict shouldn't carry broken file
        # references into the tab restorers.
        unresolved = {p for p in missing if p not in remap}
        prune_missing_paths_inplace(raw, unresolved)

        # Auto-save the YAML only when we actually rewrote a path.
        # Skipped entries stay on disk so the user can reattempt the
        # locate flow next time (e.g. after remounting a network share).
        if n_replaced > 0:
            try:
                self._write_yaml(yaml_path, raw)
            except Exception as e:
                QMessageBox.warning(
                    self, "Could not update YAML",
                    f"Located {n_replaced} file(s) for this load, but "
                    f"writing the updated paths back to\n{yaml_path}\n"
                    f"failed:\n{e}\n\nThe load itself will still "
                    f"proceed with the new paths.")
            else:
                self.status.show(
                    f"Located {n_replaced} of {len(missing)} missing "
                    f"file(s); paths updated in "
                    f"{os.path.basename(yaml_path)}.")
        elif unresolved:
            self.status.show(
                f"Skipped {len(unresolved)} missing file(s); "
                f"YAML on disk left unchanged.")

        return True

    def _load_from_path(self, path):
        """Load a YAML file and populate whichever tabs are present."""
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load config:\n{e}")
            return
        if not isinstance(raw, dict):
            QMessageBox.critical(self, "Error",
                                 "Invalid config file format.")
            return

        if not self._resolve_missing_files(raw, path):
            return

        # Restore the global scan-filter registry BEFORE the tabs
        # rebuild themselves -- file entries created during tab restore
        # may want to read their initial filter state from the registry.
        from gui.scan_filter import get_registry
        get_registry().from_dict(raw.get("scan_filters") or {})

        # Same ordering rule for the calibrations: they must be in force
        # before the tabs load their runs, or Pre-Analysis would compute every
        # voltage from the file default and only fix it on the next replot.
        from gui.calibration import get_registry as _get_cal_registry
        _cal_reg = _get_cal_registry()
        _cal_reg.from_dict(raw.get("calibrations") or {})
        _cal_reg.acks_from_list(raw.get("calibration_acks") or [])

        loaded_tabs = []

        # Detect and load Estimate data
        if "estimate" in raw:
            self._restore_estimate(raw["estimate"])
            loaded_tabs.append("Estimate")
        elif self._looks_like_estimate(raw):
            # Legacy: top-level estimate format (has "element" key etc.)
            self._restore_estimate(raw)
            loaded_tabs.append("Estimate")

        # Detect and load Pre-Analysis data
        if "preanalysis" in raw:
            self._restore_preanalysis(raw["preanalysis"])
            loaded_tabs.append("Pre-Analysis")

        # Detect and load Analysis data
        if "analysis" in raw:
            self._restore_analysis(raw["analysis"])
            loaded_tabs.append("Analysis")

        # NIST ASD browser state: apply live when the window exists,
        # else park it for the first open.
        if "nist_asd" in raw:
            if self._nist_browser is not None:
                self._nist_browser.from_dict(raw["nist_asd"])
            else:
                self._pending_nist_state = raw["nist_asd"]
            loaded_tabs.append("NIST ASD")

        if not loaded_tabs:
            QMessageBox.information(
                self, "Load Config",
                "No recognized tab data found in this file.\n\n"
                "Expected keys: 'estimate', 'preanalysis', 'analysis'")
            return

        self._config_path = path
        self.setWindowTitle(
            f"DENIS v{__version__} - {os.path.basename(path)}")
        # A freshly loaded file IS the saved state — closing without
        # further edits must not prompt.
        self._mark_saved()
        self.status.show(
            f"Loaded: {', '.join(loaded_tabs)} from {os.path.basename(path)}")

    def _looks_like_estimate(self, d):
        """Check if a dict looks like a legacy top-level estimate config."""
        return any(k in d for k in ("element", "Z", "transition", "isotopes"))

    # ── Run ──────────────────────────────────────────────────────

    def _on_run(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Busy",
                                "A calculation is already running.")
            return

        et = self.estimate_tab

        # Validate
        gp = et.params_tab.global_params
        if not gp.element_edit.text().strip():
            QMessageBox.warning(self, "Validation Error",
                                "Element symbol is required.")
            return
        if len(et.params_tab.isotope_list.panels) == 0:
            QMessageBox.warning(self, "Validation Error",
                                "At least one isotope is required.")
            return
        if et.params_tab.isotope_list.get_reference_panel() is None:
            QMessageBox.warning(self, "Validation Error",
                                "No reference isotope selected.\n"
                                "Please mark one isotope as the reference.")
            return

        # Build config and write to temp file
        yaml_dict = self._build_estimate_dict()
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            tmp_path = os.path.join(
                tempfile.gettempdir(), f"cls_{timestamp}.yaml")
            with open(tmp_path, "w") as tmp:
                yaml.dump(yaml_dict, tmp, default_flow_style=False,
                          sort_keys=False, allow_unicode=True)
            self._tmp_config = tmp_path
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to write temp config:\n{e}")
            return

        # Get run options
        opts = et.run_tab.get_options()

        # Clear log
        et.run_tab.log_text.clear()
        # Switch to the Estimate tab (single surface — the run column
        # with its log is always visible there).
        self.tabs.setCurrentWidget(et)

        # Store config name for plot loading
        self._run_config_name = os.path.splitext(
            os.path.basename(self._tmp_config))[0]

        # Build isotope file tag for plot discovery
        seen = []
        for p in et.params_tab.isotope_list.panels:
            label = p.iso_label.text()
            if label and label not in seen:
                seen.append(label)
        self._run_file_tag = "_" + "_".join(seen) if seen else ""

        # Disable run button
        et.run_tab.run_btn.setEnabled(False)
        self.status.show_persistent("Estimation running...")

        # Start worker
        self._worker = WorkerThread(
            config_path=self._tmp_config,
            output_dir=opts["output_dir"],
            no_plot=opts["no_plot"],
            palette=opts["palette"],
        )
        self._worker.log_line.connect(et.run_tab.append_log)
        self._worker.finished_ok.connect(self._on_run_finished)
        self._worker.finished_error.connect(self._on_run_error)
        self._worker.start()

    def _on_run_finished(self, output_dir):
        et = self.estimate_tab
        et.run_tab.run_btn.setEnabled(True)
        self.status.show("Estimation complete")

        # Clean up temp file
        if hasattr(self, "_tmp_config") and os.path.exists(self._tmp_config):
            os.unlink(self._tmp_config)

        # Determine actual output subdirectory
        config_name = self._run_config_name
        actual_output = os.path.join(output_dir, config_name)
        if not os.path.isdir(actual_output):
            actual_output = output_dir

        # Pass plot data for native rendering
        if self._worker and self._worker.plot_results:
            from cls_estimations.plotting import set_palette
            set_palette(et.run_tab.palette_combo.currentText())
            et.plots_tab.set_plot_data(self._worker.plot_results)
            et.plots_tab.set_peak_data(self._worker.all_peaks or [])

        et.plots_tab.display_results(
            actual_output, config_name,
            self._run_file_tag,
            et.run_tab.palette_combo.currentText(),
        )

        # Persist the plotted arrays next to the PDFs and remember the
        # run so Save records it and a later load can restore the plots
        # without re-running (and without re-bloating the output dir).
        palette = et.run_tab.palette_combo.currentText()
        try:
            self._persist_estimate_outputs(actual_output)
        except Exception:
            logging.getLogger("denis").exception(
                "could not persist estimate arrays to %s", actual_output)
        self._last_estimate_run = {
            "run_dir": actual_output,
            "config_name": config_name,
            "file_tag": self._run_file_tag,
            "palette": palette,
        }

    def _persist_estimate_outputs(self, run_dir):
        """Write the in-memory run results into the run folder:
        ``estimate_results.npz`` (arrays for native re-rendering) and
        ``peaks.csv`` (human-readable peak table)."""
        if not (self._worker and self._worker.plot_results):
            return
        import numpy as np
        np.savez_compressed(
            os.path.join(run_dir, "estimate_results.npz"),
            plot_results=np.array(self._worker.plot_results, dtype=object),
            all_peaks=np.array(self._worker.all_peaks or [], dtype=object),
            version=1)
        peaks = self._worker.all_peaks or []
        if peaks:
            import pandas as pd
            pd.DataFrame(peaks).to_csv(
                os.path.join(run_dir, "peaks.csv"), index=False)

    def _restore_estimate_outputs(self, last_run):
        """Repopulate the estimate plots / peak table / log from a
        recorded run folder. Quietly skips whatever is missing."""
        run_dir = last_run.get("run_dir") or ""
        if not os.path.isdir(run_dir):
            self.status.show(
                f"Previous estimate output not found: {run_dir}")
            return
        et = self.estimate_tab
        palette = last_run.get("palette") or "default"
        config_name = last_run.get("config_name") or ""
        file_tag = last_run.get("file_tag") or ""
        self._last_estimate_run = {
            "run_dir": run_dir, "config_name": config_name,
            "file_tag": file_tag, "palette": palette,
        }
        idx = et.run_tab.palette_combo.findText(palette)
        if idx >= 0:
            et.run_tab.palette_combo.setCurrentIndex(idx)
        et.run_tab.output_dir.setText(os.path.dirname(run_dir) or run_dir)

        npz_path = os.path.join(run_dir, "estimate_results.npz")
        if os.path.isfile(npz_path):
            try:
                import numpy as np
                # Context manager: NpzFile sits in a reference cycle, so
                # without an explicit close the .npz stays locked until
                # cyclic GC runs (breaks folder deletion on Windows).
                with np.load(npz_path, allow_pickle=True) as data:
                    plot_results = list(data["plot_results"])
                    all_peaks = list(data["all_peaks"])
                from cls_estimations.plotting import set_palette
                set_palette(palette)
                et.plots_tab.set_plot_data(plot_results)
                et.plots_tab.set_peak_data(all_peaks)
            except Exception:
                logging.getLogger("denis").exception(
                    "could not load estimate arrays from %s", npz_path)
        et.plots_tab.display_results(run_dir, config_name, file_tag,
                                     palette)

        # Bring the run log back too.
        try:
            logs = sorted(f for f in os.listdir(run_dir)
                          if f.endswith(".log"))
            if logs:
                with open(os.path.join(run_dir, logs[-1]), "r",
                          errors="replace") as f:
                    et.run_tab.log_text.setPlainText(f.read())
        except Exception:
            pass
        self.status.show(
            f"Restored previous estimate run: {os.path.basename(run_dir)}")

    def _on_load_estimation(self):
        """Pick a previous estimation run folder and restore its plots,
        peak table and log for viewing — no re-run, no new outputs."""
        from gui.shared_widgets import get_estimates_dir
        start = (self.estimate_tab.run_tab.output_dir.text().strip()
                 or get_estimates_dir())
        d = QFileDialog.getExistingDirectory(
            self, "Load Estimation Run", start)
        if not d:
            return
        cfg_name = os.path.basename(os.path.normpath(d))
        # The run folder IS the config name; the file tag (isotope
        # labels) is recovered from the artifact names inside.
        import glob as _glob
        file_tag = ""
        found_artifact = False
        for pat, prefix in (
                (f"{cfg_name}_hfs_spectra*.pdf",
                 f"{cfg_name}_hfs_spectra"),
                (f"{cfg_name}_cls_estimate*.log",
                 f"{cfg_name}_cls_estimate")):
            hits = _glob.glob(os.path.join(d, pat))
            if hits:
                base = os.path.splitext(os.path.basename(hits[0]))[0]
                file_tag = base[len(prefix):]
                found_artifact = True
                break
        if not (found_artifact or os.path.isfile(
                os.path.join(d, "estimate_results.npz"))):
            # Most likely the estimates ROOT was picked instead of a
            # run folder inside it — say so instead of silently
            # showing an empty plot window.
            QMessageBox.information(
                self, "Load Estimation",
                f"'{cfg_name}' does not look like an estimation run "
                "folder.\n\nPick one run folder (e.g. "
                "cls_<date>_<time>) inside the output directory — it "
                "holds the run's PDFs, log and results file.")
            return
        self._restore_estimate_outputs({
            "run_dir": d,
            "config_name": cfg_name,
            "file_tag": file_tag,
            "palette":
                self.estimate_tab.run_tab.palette_combo.currentText(),
        })

    def _new_window(self):
        """Launch another independent DENIS instance (skips the
        single-instance guard via --new-window)."""
        from PySide6.QtCore import QProcess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(root, "gui.py")
        res = QProcess.startDetached(
            sys.executable, [script, "--new-window"], root)
        ok = res[0] if isinstance(res, tuple) else bool(res)
        if not ok:
            QMessageBox.warning(
                self, "New Window",
                f"Could not start a new DENIS instance:\n"
                f"{sys.executable} {script}")

    def _on_run_error(self, error_msg):
        et = self.estimate_tab
        et.run_tab.run_btn.setEnabled(True)
        self.status.show("Estimation failed -- see error dialog")
        # code review 2026-06-02, worker-thread-exceptions-not-logged:
        # also record the failure in the session log so a trace survives
        # after the dialog is dismissed.
        logging.getLogger("denis").error(
            "Estimation run failed: %s", error_msg)
        QMessageBox.critical(self, "Calculation Error", error_msg)

        # Clean up temp file
        if hasattr(self, "_tmp_config") and os.path.exists(self._tmp_config):
            os.unlink(self._tmp_config)

    # ── Misc ─────────────────────────────────────────────────────

    def _on_tab_changed(self, _index):
        """When the Results tab is opened, stop the finished-analysis
        status-bar flash (the user is now looking)."""
        if self.tabs.currentWidget() is self.results_tab:
            self.status.stop_flash()

    def _on_analysis_results(self, project_name, results, output_config):
        """Forward analysis results to the Results tab.

        We deliberately do NOT switch to the Results tab — the user may
        still be working in Analysis. Instead, flash the status bar (and
        the freshly finished iteration is highlighted in the Results
        tree) until the user opens the Results tab to look.
        """
        self.results_tab.add_results(project_name, results, output_config)
        ok = [r for r in results if r.get("success")]
        msg = (f"Analysis '{project_name}' complete -- "
               f"{len(ok)}/{len(results)} fits succeeded"
               f"  ▸ click the Results tab to view"
               f"  (forgot a plot? Output ▸ Update iteration)")
        redchis = [r["fit_quality"]["redchi"] for r in ok
                   if r.get("fit_quality", {}).get("redchi") is not None]
        if redchis:
            avg = sum(redchis) / len(redchis)
            msg += f" | avg red. chi-sq = {avg:.3f}"
        # Flash until the user actually opens the Results tab.
        if self.tabs.currentWidget() is self.results_tab:
            self.status.show(msg, timeout_ms=15000)
        else:
            self.status.flash(msg)

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()

    def _open_output(self):
        """Open the unified root output directory (with ``estimates/``
        and ``analysis/`` subfolders inside). Resolves from the
        ``output_directory`` setting so it tracks the user's
        configured root, not the per-config Estimate field."""
        from gui.shared_widgets import get_output_root
        output_dir = get_output_root()
        # Auto-create so first-time users hitting Run ▸ Open Output
        # don't get the "doesn't exist yet" warning before they've
        # run anything. Cheap.
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            pass
        if os.path.isdir(output_dir):
            import subprocess
            import platform
            if platform.system() == "Linux":
                subprocess.Popen(["xdg-open", output_dir])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["explorer", os.path.normpath(output_dir)])
        else:
            QMessageBox.information(
                self, "Info",
                f"Output directory does not exist yet:\n{output_dir}")

    def _open_schmidt_calculator(self):
        if (self._schmidt_dialog is not None
                and self._schmidt_dialog.isVisible()):
            self._schmidt_dialog.raise_()
            self._schmidt_dialog.activateWindow()
            return
        self._schmidt_dialog = SchmidtCalculatorDialog(parent=self)
        self._schmidt_dialog.show()

    def _open_shell_plotter(self):
        if (self._shell_dialog is not None
                and self._shell_dialog.isVisible()):
            self._shell_dialog.raise_()
            self._shell_dialog.activateWindow()
            return
        self._shell_dialog = ShellConfigDialog(parent=self)
        self._shell_dialog.show()

    def _open_unit_converter(self):
        from gui.shared_widgets import UnitConverterDialog
        if (self._unit_converter_dialog is not None
                and self._unit_converter_dialog.isVisible()):
            self._unit_converter_dialog.raise_()
            self._unit_converter_dialog.activateWindow()
            return
        self._unit_converter_dialog = UnitConverterDialog(parent=self)
        self._unit_converter_dialog.show()

    def _open_shg_calculator(self):
        from gui.shared_widgets import SHGCalculatorDialog
        if (self._shg_calculator_dialog is not None
                and self._shg_calculator_dialog.isVisible()):
            self._shg_calculator_dialog.raise_()
            self._shg_calculator_dialog.activateWindow()
            return
        self._shg_calculator_dialog = SHGCalculatorDialog(parent=self)
        self._shg_calculator_dialog.show()

    def _open_sfg_dfg_calculator(self):
        from gui.shared_widgets import SFGDFGCalculatorDialog
        if (self._sfg_dfg_dialog is not None
                and self._sfg_dfg_dialog.isVisible()):
            self._sfg_dfg_dialog.raise_()
            self._sfg_dfg_dialog.activateWindow()
            return
        self._sfg_dfg_dialog = SFGDFGCalculatorDialog(parent=self)
        self._sfg_dfg_dialog.show()

    def _open_quick_plot(self):
        from gui.shared_widgets import QuickPlotDialog
        if (self._quick_plot_dialog is not None
                and self._quick_plot_dialog.isVisible()):
            self._quick_plot_dialog.raise_()
            self._quick_plot_dialog.activateWindow()
            return
        self._quick_plot_dialog = QuickPlotDialog(parent=self)
        self._quick_plot_dialog.show()

    def _open_nist_browser(self):
        """Show the (single) NIST ASD browser window; on first open,
        apply state parked by a save-file load, else refill from the
        offline cache."""
        if self._nist_browser is None:
            from gui.nist_asd.tab import NistAsdWindow
            self._nist_browser = NistAsdWindow(parent=self)
            if self._pending_nist_state:
                self._nist_browser.from_dict(self._pending_nist_state)
                self._pending_nist_state = None
            else:
                self._nist_browser.load_cached_spectrum()
        self._nist_browser.show()
        self._nist_browser.raise_()
        self._nist_browser.activateWindow()

    def _open_manual(self):
        from gui.manual.viewer import ManualWindow
        if (self._manual_window is not None
                and self._manual_window.isVisible()):
            self._manual_window.raise_()
            self._manual_window.activateWindow()
            return
        self._manual_window = ManualWindow(parent=self)
        self._manual_window.show()

    def _show_about(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("About DENIS")
        dlg.setFixedWidth(600)

        outer = QVBoxLayout(dlg)
        body = QHBoxLayout()

        # Logo (vertically centered)
        _logo = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "icons", "denis_512.png")
        if os.path.exists(_logo):
            logo_label = QLabel()
            logo_label.setPixmap(QPixmap(_logo).scaled(
                180, 180, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            body.addWidget(logo_label, 0, Qt.AlignmentFlag.AlignVCenter)

        # Text
        text_label = QLabel(
            f"<h3>DENIS  v{__version__}</h3>"
            "<p><b>D</b>oppler <b>E</b>stimation and "
            "<b>N</b>umerical <b>I</b>nference for "
            "<b>S</b>pectroscopy</p>"
            "<p>A comprehensive tool for Collinear Laser Spectroscopy:<br>"
            "estimation, pre-analysis, fitting, and results.</p>"
            "<p>Fitting &amp; spectrum estimation powered by "
            '<a href="https://iks-nm.github.io/satlas2/index.html">'
            "satlas2</a>.<br>"
            "IGISOL-style ASDF file reading via "
            '<a href="https://github.com/andry3vi/cls_tools">'
            "cls_tools</a>.</p>"
            '<p>Developer: <a href="https://ardakayaalp.com/">'
            "Arda Kayaalp</a><br>"
            'Email: <a href="mailto:arda.kayaalp@kuleuven.be">'
            "arda.kayaalp@kuleuven.be</a></p>"
            "<p>GUI built with PySide6 + matplotlib.</p>")
        text_label.setWordWrap(True)
        text_label.setTextFormat(Qt.TextFormat.RichText)
        text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction)
        text_label.setOpenExternalLinks(True)
        body.addWidget(text_label, 1)

        outer.addLayout(body)

        from PySide6.QtWidgets import QDialogButtonBox
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dlg.accept)
        outer.addWidget(btn_box)
        dlg.exec()


# ══════════════════════════════════════════════════════════════════
#  Settings Dialog
# ══════════════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    """Application settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DENIS Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(700)
        # Fusion compresses QSpinBox/QDoubleSpinBox to a height that
        # clips the digit baselines in the Plot Defaults rows. Bump the
        # minimum height + add small vertical margins so adjacent rows
        # don't touch (scoped to this dialog). Avoid `padding` here: it
        # breaks Fusion's native frame on the focused widget.
        self.setStyleSheet(
            "QSpinBox, QDoubleSpinBox { "
            "min-height: 24px; margin-top: 2px; margin-bottom: 2px; }"
        )
        outer_layout = QVBoxLayout(self)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, 1)

        settings = _load_settings()

        # ── General ──
        gen_grp = QGroupBox("General")
        gen_form = QFormLayout(gen_grp)

        from gui.shared_widgets import _get_settings_path
        settings_path = _get_settings_path()
        settings_dir_row = QHBoxLayout()
        self._settings_dir = QLineEdit(os.path.dirname(settings_path))
        self._settings_dir.setReadOnly(True)
        self._settings_dir.setToolTip(
            "Directory where DENIS stores its settings file.\n"
            f"Current: {settings_path}")
        settings_dir_row.addWidget(self._settings_dir)
        open_dir_btn = QPushButton("Open")
        # Auto-size to the label plus button padding so the caption stays
        # unclipped when the UI font is zoomed.
        open_dir_btn.setToolTip("Open settings folder in file explorer")
        open_dir_btn.clicked.connect(
            lambda: self._open_folder(os.path.dirname(settings_path)))
        settings_dir_row.addWidget(open_dir_btn)
        gen_form.addRow("Settings directory:", settings_dir_row)

        self._auto_path = QCheckBox("Auto-convert paths (Linux/Windows)")
        self._auto_path.setChecked(settings.get("auto_path_conversion", True))
        self._auto_path.setToolTip(
            "Automatically convert file paths when loading save files "
            "created on a different OS (e.g. Linux paths on Windows).")
        gen_form.addRow(self._auto_path)

        from gui.theme import THEMES
        self._theme_combo = QComboBox()
        for key, label in THEMES.items():
            self._theme_combo.addItem(label, key)
        idx = self._theme_combo.findData(
            settings.get("ui_theme", "dark"))
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)
        self._theme_combo.setToolTip(
            "Application look. 'Dark' is the standard DENIS theme; "
            "'Classic 98 (dark)' is a retro look — dark colors with "
            "Windows-98 style 3D bevels, sharp corners and a bitmap "
            "(pixel) font. Applies immediately on OK.")
        gen_form.addRow("Theme:", self._theme_combo)

        self._ui_scale = QDoubleSpinBox()
        self._ui_scale.setRange(0.5, 3.0)
        self._ui_scale.setSingleStep(0.25)
        self._ui_scale.setDecimals(2)
        self._ui_scale.setValue(settings.get("ui_scale", 0.0))
        self._ui_scale.setToolTip(
            "UI scaling factor. Set 0 for automatic (system DPI). "
            "Try 1.25-1.50 for QHD screens, 1.75-2.0 for 4K. "
            "Requires restart to take effect.")
        self._ui_scale.setSpecialValueText("Auto")
        self._ui_scale.setMinimum(0.0)
        gen_form.addRow("UI scale (restart):", self._ui_scale)

        self._save_session_log = QCheckBox("Verbose session log (restart)")
        self._save_session_log.setChecked(
            settings.get("save_session_log", False))
        self._save_session_log.setToolTip(
            "A session log is always written to the 'logs' folder next to "
            "the application (the newest 20 are kept). This toggle raises "
            "the log level to DEBUG for extra diagnostic detail. "
            "Takes effect on the next launch.")
        gen_form.addRow(self._save_session_log)

        layout.addWidget(gen_grp)

        # ── Performance ──
        perf_grp = QGroupBox("Performance")
        perf_form = QFormLayout(perf_grp)
        cpu_count = os.cpu_count() or 4
        perf_form.addRow("Available CPU cores:", QLabel(str(cpu_count)))
        self._max_cores = QSpinBox()
        self._max_cores.setRange(1, cpu_count)
        self._max_cores.setValue(
            settings.get("max_cores", max(1, cpu_count - 2)))
        perf_form.addRow("Max cores for fitting:", self._max_cores)
        layout.addWidget(perf_grp)

        # ── Output ──
        out_grp = QGroupBox("Output")
        out_form = QFormLayout(out_grp)

        dir_row = QVBoxLayout()
        self._output_dir = QLineEdit(
            settings.get("output_directory", "./output"))
        self._output_dir.setToolTip(
            "Root output directory. Estimate runs land under "
            "<root>/estimates/ and Analysis runs under "
            "<root>/analysis/. The 'Run ▸ Open Output Folder' menu "
            "opens this root.")
        dir_row.addWidget(self._output_dir)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        dir_row.addWidget(browse_btn)
        out_form.addRow("Output directory:", dir_row)

        self._iter_mode = QComboBox()
        self._iter_mode.addItems(["Auto", "Manual"])
        idx = self._iter_mode.findText(
            settings.get("iteration_mode", "Auto"))
        if idx >= 0:
            self._iter_mode.setCurrentIndex(idx)
        out_form.addRow("Iteration mode:", self._iter_mode)

        self._plot_format = QComboBox()
        self._plot_format.addItems(["png", "pdf", "svg"])
        idx = self._plot_format.findText(
            settings.get("plot_format", "png"))
        if idx >= 0:
            self._plot_format.setCurrentIndex(idx)
        out_form.addRow("Plot format:", self._plot_format)

        self._plot_dpi = QSpinBox()
        self._plot_dpi.setRange(72, 600)
        self._plot_dpi.setValue(settings.get("plot_dpi", 200))
        out_form.addRow("Plot DPI:", self._plot_dpi)
        layout.addWidget(out_grp)

        # ── Fitting Defaults ──
        fit_grp = QGroupBox("Fitting Defaults")
        fit_form = QFormLayout(fit_grp)

        self._default_method = QComboBox()
        self._default_method.addItems([
            "leastsq", "least_squares", "slsqp", "emcee",
            "nelder", "powell", "cobyla"])
        idx = self._default_method.findText(
            settings.get("default_method", "leastsq"))
        if idx >= 0:
            self._default_method.setCurrentIndex(idx)
        fit_form.addRow("Default method:", self._default_method)

        self._default_stats = QComboBox()
        self._default_stats.addItems([
            "Chi-square", "Gaussian LLH", "Poisson LLH"])
        idx = self._default_stats.findText(
            settings.get("default_statistics", "Chi-square"))
        if idx >= 0:
            self._default_stats.setCurrentIndex(idx)
        fit_form.addRow("Default statistics:", self._default_stats)

        self._show_correl = QCheckBox()
        self._show_correl.setChecked(
            settings.get("show_correlations", True))
        fit_form.addRow("Show correlations:", self._show_correl)

        self._default_yerr = QComboBox()
        # Include "None" to match the SourceBlock's YERR_MODES; it was omitted
        # here so the Settings choice set was inconsistent with the real combo
        # (code review 2026-06-02, settings-fitting-defaults-write-only).
        self._default_yerr.addItems([
            "None", "Poisson sqrt(y+1)", "Poisson sqrt(y)", "Model-based"])
        idx = self._default_yerr.findText(
            settings.get("yerr_mode", "Poisson sqrt(y+1)"))
        if idx >= 0:
            self._default_yerr.setCurrentIndex(idx)
        fit_form.addRow("Default yerr:", self._default_yerr)

        self._default_xcol = QComboBox()
        self._default_xcol.addItems(["bins_center", "Fmean"])
        idx = self._default_xcol.findText(
            settings.get("x_column", "bins_center"))
        if idx >= 0:
            self._default_xcol.setCurrentIndex(idx)
        fit_form.addRow("Default x column:", self._default_xcol)
        layout.addWidget(fit_grp)

        # ── Plot Defaults ──
        plot_grp = QGroupBox("Plot Defaults")
        plot_outer = QVBoxLayout(plot_grp)

        # Global rcParams tab + per-plot-type tabs
        plot_tabs = QTabWidget()
        plot_tabs.setMinimumHeight(420)

        # --- Global tab ---
        global_w = QWidget()
        plot_form = QFormLayout(global_w)
        plot_form.setVerticalSpacing(10)
        plot_form.setContentsMargins(8, 12, 8, 8)
        user_plot = settings.get("plot_defaults", {})

        def _pd(key):
            return user_plot.get(key, _DEFAULT_PLOT_SETTINGS.get(key, 0))

        self._pd_line_w = QDoubleSpinBox()
        self._pd_line_w.setRange(0.5, 10.0); self._pd_line_w.setSingleStep(0.5)
        self._pd_line_w.setValue(_pd("lines.linewidth"))
        plot_form.addRow("Line width:", self._pd_line_w)

        self._pd_marker = QDoubleSpinBox()
        self._pd_marker.setRange(0.5, 20.0); self._pd_marker.setSingleStep(0.5)
        self._pd_marker.setValue(_pd("lines.markersize"))
        plot_form.addRow("Marker size:", self._pd_marker)

        self._pd_font = QSpinBox()
        self._pd_font.setRange(6, 30); self._pd_font.setValue(int(_pd("font.size")))
        plot_form.addRow("Font size:", self._pd_font)

        self._pd_label = QSpinBox()
        self._pd_label.setRange(6, 30); self._pd_label.setValue(int(_pd("axes.labelsize")))
        plot_form.addRow("Axis label size:", self._pd_label)

        self._pd_title = QSpinBox()
        self._pd_title.setRange(6, 30); self._pd_title.setValue(int(_pd("axes.titlesize")))
        plot_form.addRow("Title size:", self._pd_title)

        self._pd_axw = QDoubleSpinBox()
        self._pd_axw.setRange(0.2, 5.0); self._pd_axw.setSingleStep(0.1)
        self._pd_axw.setValue(_pd("axes.linewidth"))
        plot_form.addRow("Axes line width:", self._pd_axw)

        self._pd_tick_lbl = QSpinBox()
        self._pd_tick_lbl.setRange(6, 24); self._pd_tick_lbl.setValue(int(_pd("xtick.labelsize")))
        plot_form.addRow("Tick label size:", self._pd_tick_lbl)

        self._pd_tick_maj = QDoubleSpinBox()
        self._pd_tick_maj.setRange(1.0, 15.0); self._pd_tick_maj.setSingleStep(0.5)
        self._pd_tick_maj.setValue(_pd("xtick.major.size"))
        plot_form.addRow("Major tick size:", self._pd_tick_maj)

        self._pd_tick_min = QDoubleSpinBox()
        self._pd_tick_min.setRange(0.5, 10.0); self._pd_tick_min.setSingleStep(0.5)
        self._pd_tick_min.setValue(_pd("xtick.minor.size"))
        plot_form.addRow("Minor tick size:", self._pd_tick_min)

        self._pd_tick_w = QDoubleSpinBox()
        self._pd_tick_w.setRange(0.2, 3.0); self._pd_tick_w.setSingleStep(0.1)
        self._pd_tick_w.setValue(_pd("xtick.major.width"))
        plot_form.addRow("Tick width:", self._pd_tick_w)

        self._pd_legend = QSpinBox()
        self._pd_legend.setRange(6, 24); self._pd_legend.setValue(int(_pd("legend.fontsize")))
        plot_form.addRow("Legend font size:", self._pd_legend)

        self._pd_capsize = QDoubleSpinBox()
        self._pd_capsize.setRange(0.0, 10.0); self._pd_capsize.setSingleStep(0.5)
        self._pd_capsize.setValue(_pd("errorbar.capsize"))
        plot_form.addRow("Errorbar cap size:", self._pd_capsize)

        self._pd_figdpi = QSpinBox()
        self._pd_figdpi.setRange(72, 300); self._pd_figdpi.setValue(int(_pd("figure.dpi")))
        plot_form.addRow("Figure DPI:", self._pd_figdpi)

        self._pd_savedpi = QSpinBox()
        self._pd_savedpi.setRange(72, 600); self._pd_savedpi.setValue(int(_pd("savefig.dpi")))
        plot_form.addRow("Save DPI:", self._pd_savedpi)

        plot_tabs.addTab(global_w, "Global")

        # --- Per-plot-type tabs ---
        user_pt = settings.get("plot_type_defaults", {})
        self._pt_widgets = {}  # {plot_type: {key: widget}}

        _PT_LABELS = {
            "fit_plot": "Fit Plot",
            "walk_plot": "Walk Plot",
            "correlation_plot": "Correlation",
            "chisq_map": "Chi-sq Map",
            "tracker_plot": "Tracker",
            "preview": "Preview / TOF",
        }

        from gui.shared_widgets import (
            get_plot_type_tooltip, get_plot_type_sections,
            get_plot_type_label,
        )

        # code review 2026-06-02, settings-int-spinbox-range-cap-1e6:
        # count-threshold keys carry raw photon counts, which can exceed
        # 1e6 for high-stats data; give those the full int32 range.
        _COUNT_THRESHOLD_KEYS = {"low_count_max", "med_count_max"}

        def _make_widget(default_v, val, tip, key=None):
            """Build the right kind of widget for a settings entry."""
            # Order matters: bool is a subclass of int, so check it
            # FIRST or it would silently render as a 1-200 spinbox.
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
                # Counts can be very large; allow up to 1e6 so
                # threshold spinboxes are usable for high-stats data.
                # Count-threshold keys get the full int32 range so even
                # very high-stats values are not silently clamped.
                w.setRange(
                    0, 2_147_483_647 if key in _COUNT_THRESHOLD_KEYS
                    else 1_000_000)
                w.setValue(int(val))
            elif isinstance(default_v, str):
                w = QLineEdit(str(val))
            else:
                return None
            if tip:
                w.setToolTip(tip)
            return w

        for pt_key, pt_label in _PT_LABELS.items():
            defaults = _DEFAULT_PLOT_TYPE_SETTINGS.get(pt_key, {})
            user_vals = user_pt.get(pt_key, {})
            sections = get_plot_type_sections(pt_key)
            widgets = {}
            tab_w = QWidget()

            if sections:
                # Sectioned layout: each section becomes a QGroupBox
                # holding a QFormLayout. Falls back to flat for any keys
                # not explicitly listed in `sections`.
                outer = QVBoxLayout(tab_w)
                outer.setContentsMargins(8, 12, 8, 8)
                outer.setSpacing(10)
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
                        note_label.setStyleSheet(
                            "color: #aaa; font-style: italic;")
                        grp_lay.addWidget(note_label)
                    form = QFormLayout()
                    form.setVerticalSpacing(8)
                    grp_lay.addLayout(form)
                    for k in sec.get("keys", []):
                        if k not in defaults:
                            continue
                        listed.add(k)
                        default_v = defaults[k]
                        val = user_vals.get(k, default_v)
                        tip = get_plot_type_tooltip(pt_key, k)
                        w = _make_widget(default_v, val, tip, key=k)
                        if w is None:
                            continue
                        nice = get_plot_type_label(pt_key, k)
                        label_w = QLabel(f"{nice}:")
                        if tip:
                            label_w.setToolTip(tip)
                        form.addRow(label_w, w)
                        widgets[k] = w
                    outer.addWidget(grp)

                # Stragglers: any keys defined in defaults but not put
                # in any section land in a "Other" group at the bottom.
                others = [k for k in defaults if k not in listed]
                if others:
                    grp = QGroupBox("Other")
                    form = QFormLayout(grp)
                    form.setVerticalSpacing(8)
                    form.setContentsMargins(8, 12, 8, 8)
                    for k in others:
                        default_v = defaults[k]
                        val = user_vals.get(k, default_v)
                        tip = get_plot_type_tooltip(pt_key, k)
                        w = _make_widget(default_v, val, tip, key=k)
                        if w is None:
                            continue
                        nice = get_plot_type_label(pt_key, k)
                        label_w = QLabel(f"{nice}:")
                        if tip:
                            label_w.setToolTip(tip)
                        form.addRow(label_w, w)
                        widgets[k] = w
                    outer.addWidget(grp)
                outer.addStretch()
            else:
                # Flat layout for plot types without a section spec
                form = QFormLayout(tab_w)
                form.setVerticalSpacing(10)
                form.setContentsMargins(8, 12, 8, 8)
                for k, default_v in defaults.items():
                    val = user_vals.get(k, default_v)
                    tip = get_plot_type_tooltip(pt_key, k)
                    w = _make_widget(default_v, val, tip, key=k)
                    if w is None:
                        continue
                    nice = get_plot_type_label(pt_key, k)
                    label_w = QLabel(f"{nice}:")
                    if tip:
                        label_w.setToolTip(tip)
                    form.addRow(label_w, w)
                    widgets[k] = w

            self._pt_widgets[pt_key] = widgets

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(tab_w)
            plot_tabs.addTab(scroll, pt_label)

        plot_outer.addWidget(plot_tabs)

        reset_btn = QPushButton("Reset All Plot Defaults")
        reset_btn.clicked.connect(self._reset_plot_defaults)
        plot_outer.addWidget(reset_btn)

        layout.addWidget(plot_grp)

        # ── Buttons (outside scroll area) ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer_layout.addWidget(buttons)

    def _open_folder(self, path):
        """Open a folder in the system file explorer."""
        import subprocess, sys as _sys
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        if _sys.platform == "win32":
            os.startfile(path)
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._output_dir.text())
        if d:
            self._output_dir.setText(d)

    def _reset_plot_defaults(self):
        """Reset all plot settings to built-in defaults."""
        d = _DEFAULT_PLOT_SETTINGS
        self._pd_line_w.setValue(d["lines.linewidth"])
        self._pd_marker.setValue(d["lines.markersize"])
        self._pd_font.setValue(int(d["font.size"]))
        self._pd_label.setValue(int(d["axes.labelsize"]))
        self._pd_title.setValue(int(d["axes.titlesize"]))
        self._pd_axw.setValue(d["axes.linewidth"])
        self._pd_tick_lbl.setValue(int(d["xtick.labelsize"]))
        self._pd_tick_maj.setValue(d["xtick.major.size"])
        self._pd_tick_min.setValue(d["xtick.minor.size"])
        self._pd_tick_w.setValue(d["xtick.major.width"])
        self._pd_legend.setValue(int(d["legend.fontsize"]))
        self._pd_capsize.setValue(d["errorbar.capsize"])
        self._pd_figdpi.setValue(int(d["figure.dpi"]))
        self._pd_savedpi.setValue(int(d["savefig.dpi"]))
        # Reset per-plot-type
        for pt_key, widgets in self._pt_widgets.items():
            defaults = _DEFAULT_PLOT_TYPE_SETTINGS.get(pt_key, {})
            for k, w in widgets.items():
                dv = defaults.get(k)
                if dv is None:
                    continue
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(dv))
                elif isinstance(w, (QDoubleSpinBox, QSpinBox)):
                    w.setValue(dv)
                elif isinstance(w, QLineEdit):
                    w.setText(str(dv))

    def _save(self):
        settings = _load_settings()  # preserve existing keys
        old_scale = settings.get("ui_scale", 0.0)
        new_scale = self._ui_scale.value()
        # code review 2026-06-02, ui-scale-spinbox-sub-0.5-reachable:
        # the spinbox minimum is 0.0 (Auto), so a typed value in (0, 0.5)
        # is reachable and would launch the app at an unreadably small
        # scale. Coerce that range back to 0.0 (Auto).
        if 0.0 < new_scale < 0.5:
            new_scale = 0.0
            self._ui_scale.setValue(new_scale)
        old_save_log = settings.get("save_session_log", False)
        new_save_log = self._save_session_log.isChecked()
        old_theme = settings.get("ui_theme", "dark")
        new_theme = self._theme_combo.currentData() or "dark"
        settings.update({
            "auto_path_conversion": self._auto_path.isChecked(),
            "ui_theme": new_theme,
            "ui_scale": new_scale,
            "save_session_log": new_save_log,
            "max_cores": self._max_cores.value(),
            "output_directory": self._output_dir.text(),
            "iteration_mode": self._iter_mode.currentText(),
            "plot_format": self._plot_format.currentText(),
            "plot_dpi": self._plot_dpi.value(),
            "default_method": self._default_method.currentText(),
            "default_statistics": self._default_stats.currentText(),
            "show_correlations": self._show_correl.isChecked(),
            "yerr_mode": self._default_yerr.currentText(),
            "x_column": self._default_xcol.currentText(),
            "plot_defaults": {
                "lines.linewidth": self._pd_line_w.value(),
                "lines.markersize": self._pd_marker.value(),
                "font.size": self._pd_font.value(),
                "axes.labelsize": self._pd_label.value(),
                "axes.titlesize": self._pd_title.value(),
                "axes.linewidth": self._pd_axw.value(),
                "xtick.labelsize": self._pd_tick_lbl.value(),
                "ytick.labelsize": self._pd_tick_lbl.value(),
                "xtick.major.size": self._pd_tick_maj.value(),
                "ytick.major.size": self._pd_tick_maj.value(),
                "xtick.minor.size": self._pd_tick_min.value(),
                "ytick.minor.size": self._pd_tick_min.value(),
                "xtick.major.width": self._pd_tick_w.value(),
                "ytick.major.width": self._pd_tick_w.value(),
                "legend.fontsize": self._pd_legend.value(),
                "errorbar.capsize": self._pd_capsize.value(),
                "figure.dpi": self._pd_figdpi.value(),
                "savefig.dpi": self._pd_savedpi.value(),
            },
        })
        # Per-plot-type settings
        pt_out = {}
        for pt_key, widgets in self._pt_widgets.items():
            pt_d = {}
            for k, w in widgets.items():
                if isinstance(w, QCheckBox):
                    pt_d[k] = w.isChecked()
                elif isinstance(w, QDoubleSpinBox):
                    pt_d[k] = w.value()
                elif isinstance(w, QSpinBox):
                    pt_d[k] = w.value()
                elif isinstance(w, QLineEdit):
                    pt_d[k] = w.text()
            pt_out[pt_key] = pt_d
        settings["plot_type_defaults"] = pt_out

        _save_settings(settings)
        apply_plot_settings()

        # Theme switches apply live: re-theme the app and refresh the
        # main window's widget-level tab sheet (which would otherwise
        # keep the previous theme's colors).
        if new_theme != old_theme:
            from gui.theme import apply_theme
            app = QApplication.instance()
            if app is not None:
                apply_theme(app, new_theme)
            mw = self.parent()
            if mw is not None and hasattr(mw, "_apply_main_tabs_style"):
                mw._apply_main_tabs_style()

        restart_msgs = []
        if old_scale != new_scale:
            restart_msgs.append("UI scale")
        if old_save_log != new_save_log:
            restart_msgs.append("session log capture")
        if restart_msgs:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Restart Required",
                f"Changed settings ({', '.join(restart_msgs)}) take "
                "effect on the next launch.\n"
                "Please restart the application.")
        self.accept()


# ══════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════
def main():
    # Apply UI scale BEFORE QApplication is created
    _startup_settings = _load_settings()
    ui_scale = _startup_settings.get("ui_scale", 0.0)
    if ui_scale > 0:
        # QT_SCALE_FACTOR must be set before QApplication init
        os.environ["QT_SCALE_FACTOR"] = str(ui_scale)
    else:
        # Auto: let Qt handle DPI awareness natively
        os.environ.pop("QT_SCALE_FACTOR", None)

    # ── Session logging / observability (always on) ──
    # A dated, pruned session log under <app>/logs captures the stdout/stderr
    # banner, Python warnings, Qt messages, and uncaught exceptions (GUI and
    # worker threads) on every launch -- so a freeze or silent crash leaves a
    # trace even on a double-click (pythonw) launch with no console. The
    # Settings "save session log" toggle now selects verbose (DEBUG) output
    # rather than enabling logging at all, removing the old enable-then-restart
    # trap where a crash before the toggle was set left no record.
    from gui import session_log
    _app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _log_file = session_log.install(
        _app_root,
        verbose=bool(_startup_settings.get("save_session_log", False)))
    # Qt's own warnings/errors bypass the Python tee; route them to the log
    # before QApplication is created.
    session_log.install_qt_handler()

    # ── Session header (printed verbatim, like a CLI banner) ──
    import platform as _platform
    print("=" * 60)
    print(f"DENIS v{__version__}")
    print("=" * 60)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {_platform.system()} {_platform.release()} "
          f"({_platform.machine()})")
    print(f"OS:         {_platform.platform()}")
    print(f"App dir:    {_app_root}")
    if _log_file:
        print(f"Log file:   {_log_file}")
    # Versions come from the installed-package metadata, NOT from
    # importing the modules: __import__("satlas2") here cost ~2 s of
    # every startup (satlas2 → lmfit/emcee) just to print one line,
    # defeating the lazy-import work everywhere else.
    from importlib import metadata as _im
    for dist_name, label in (("PySide6", "PySide6"),
                             ("matplotlib", "matplotlib"),
                             ("numpy", "numpy"),
                             ("satlas2", "satlas2"),
                             ("clstools", "clstools")):
        try:
            print(f"{label:<11} {_im.version(dist_name)}")
        except Exception:
            print(f"{label:<11} NOT FOUND")
    print("-" * 60)

    app = QApplication(sys.argv)
    app.setApplicationName("DENIS")

    # ── Single-instance guard ──
    # A second launch (double-clicking the shortcut, or impatient repeat
    # clicks while the first instance is still loading) should raise the
    # existing window instead of starting another process. The first instance
    # owns a per-user named local socket; later launches connect to it, hand
    # off any file argument, and exit.
    #
    # File > New Window passes --new-window: that instance skips the guard
    # entirely (neither probes nor listens), so it runs independently and
    # never hijacks or receives file hand-offs.
    _new_window_flag = "--new-window" in sys.argv[1:]
    _file_args = [a for a in sys.argv[1:]
                  if not a.startswith("-") and os.path.isfile(a)]
    _single_server = None
    if not _new_window_flag:
        from PySide6.QtNetwork import QLocalServer, QLocalSocket
        import getpass
        try:
            _srv_name = f"DENIS-singleinstance-{getpass.getuser()}"
        except Exception:
            _srv_name = "DENIS-singleinstance"
        _probe = QLocalSocket()
        _probe.connectToServer(_srv_name)
        if _probe.waitForConnected(300):
            _payload = (os.path.abspath(_file_args[0])
                        if _file_args else "")
            _probe.write(_payload.encode("utf-8"))
            _probe.waitForBytesWritten(300)
            _probe.disconnectFromServer()
            return  # an instance is already running; don't open another
        # First instance: own the name (clearing any stale socket from a
        # prior crash) and listen for later launches.
        QLocalServer.removeServer(_srv_name)
        _single_server = QLocalServer()
        _single_server.listen(_srv_name)

    # Fusion style + themed palette + the app-wide stylesheet + the
    # wheel-focus guard, all owned by gui.theme so every window, dialog
    # and context menu shares one look. (Without the forced Fusion
    # style, macOS Aqua renders the dark stylesheets unreadable.)
    # The theme name persists under settings key "ui_theme".
    from gui.theme import apply_theme
    apply_theme(app, _startup_settings.get("ui_theme", "dark"))

    # Right-click on ANY matplotlib canvas → "Edit plot…" (universal
    # entry to the plot editor; surfaces with richer wiring override
    # via canvas._plot_editor_opener).
    from gui.shared_widgets import install_plot_editor_access
    install_plot_editor_access(app)

    # App icon: prefer custom DENIS ico, fall back to Lucide atom
    _icons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "icons")
    _ico_path = os.path.join(_icons_dir, "denis.ico")
    if os.path.exists(_ico_path):
        app.setWindowIcon(QIcon(_ico_path))
    else:
        app.setWindowIcon(lucide_icon("atom"))

    # ── Splash screen ──
    # File > New Window (--new-window) skips the splash: the user asked
    # for another window of an app that is already up, so just open it.
    _splash_path = os.path.join(_icons_dir, "denis_512.png")
    splash = None
    if os.path.exists(_splash_path) and not _new_window_flag:
        splash_pix = QPixmap(_splash_path)
        # Build splash with version text
        from PySide6.QtGui import QPainter
        canvas = QPixmap(540, 600)
        canvas.fill(QColor(24, 24, 30))
        painter = QPainter(canvas)
        # Center the logo
        x_off = (540 - splash_pix.width()) // 2
        painter.drawPixmap(x_off, 20, splash_pix)
        # Title and version below logo
        painter.setPen(QColor(220, 220, 220))
        font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 545, 540, 30,
                         Qt.AlignmentFlag.AlignCenter, "DENIS")
        font2 = QFont("Segoe UI", 10)
        painter.setFont(font2)
        painter.setPen(QColor(160, 160, 170))
        painter.drawText(0, 572, 540, 20,
                         Qt.AlignmentFlag.AlignCenter,
                         f"v{__version__} — Doppler Estimation and "
                         "Numerical Inference for Spectroscopy")
        painter.end()

        splash = QSplashScreen(canvas)
        splash.show()
        app.processEvents()

    # Apply user's plot defaults (or built-in defaults) before any Figure creation
    apply_plot_settings()

    window = MainWindow()
    # Start maximized so the UI uses the full available monitor work area (not
    # true fullscreen -- the title bar and restore button remain).
    #
    # Size the window to the screen work area BEFORE showing. Calling
    # showMaximized() on a window whose normal geometry is still the small
    # resize(1200, 800) makes Qt/Windows apply the maximized *frame* (so the
    # title-bar buttons sit at the real screen edge) while the child layout is
    # computed against the old narrow width -- the splitters overflow and the
    # right-hand panels are clipped until a manual restore+maximize forces a
    # real resize. Pre-sizing to the work area means the first (and only)
    # layout pass already runs at the final maximized width, so nothing is
    # clipped. showMaximized() then just sets the maximized window state.
    from PySide6.QtGui import QGuiApplication
    _screen = window.screen() or QGuiApplication.primaryScreen()
    if _screen is not None:
        window.setGeometry(_screen.availableGeometry())
    window.showMaximized()

    if splash is not None:
        splash.finish(window)

    # Raise the existing window when a second launch pokes the local server.
    def _on_second_instance():
        conn = _single_server.nextPendingConnection()
        if conn is None:
            return
        window.setWindowState(
            (window.windowState() & ~Qt.WindowState.WindowMinimized)
            | Qt.WindowState.WindowActive)
        window.show()
        window.raise_()
        window.activateWindow()

        def _maybe_load():
            data = bytes(conn.readAll()).decode("utf-8", "ignore").strip()
            if data and os.path.isfile(data):
                window._load_from_path(data)
            conn.disconnectFromServer()
        conn.readyRead.connect(_maybe_load)
        conn.disconnected.connect(conn.deleteLater)
    if _single_server is not None:
        _single_server.newConnection.connect(_on_second_instance)

    if _file_args:
        window._load_from_path(_file_args[0])

    # Force a clean restore->maximize cycle once the event loop is running.
    # Calling showMaximized() before exec() (and building tabs from a loaded
    # config before exec()) leaves the central layout sized against stale
    # geometry: the title-bar buttons sit on the real maximized frame, but the
    # tab content overflows the right edge until the user manually
    # restores+maximizes. Reproduce that toggle here. The normal geometry was
    # pre-set to the screen work area above, so showNormal() looks almost
    # identical to maximized and the flicker is negligible; the two distinct
    # geometries generate the real resize events that re-lay-out every tab.
    def _settle_maximized_layout():
        if window.windowState() & Qt.WindowState.WindowMaximized:
            window.showNormal()
            QTimer.singleShot(0, window.showMaximized)
    QTimer.singleShot(0, _settle_maximized_layout)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
