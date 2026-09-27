"""Top-level Analysis tab that manages analysis-project subtabs.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Hosts the Analysis tab's toolbar and the QTabWidget of analysis projects
plus the permanent Isotope Shifts tab. Handles creating, renaming,
closing, and Sample/Reference conversion of projects, keeps the Isotope
Shifts tab pinned last, forwards fit results and progress to the Results
tab, and saves/loads the whole analysis configuration as YAML.

Depends on: gui.analysis.project (AnalysisProject),
gui.analysis.isotope_shift_tab (IsotopeShiftTab), and gui.shared_widgets
(last-directory helpers, imported lazily).
"""

import yaml

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QMessageBox, QFileDialog, QInputDialog,
    QTabBar, QMenu,
)
from PySide6.QtCore import Qt, Signal

from gui.analysis.project import AnalysisProject
from gui.analysis.isotope_shift_tab import IsotopeShiftTab


class AnalysisTab(QWidget):
    """Analysis tab with project subtabs and block-based pipeline."""
    results_ready = Signal(str, list, dict)  # forward to Results tab
    fit_progress = Signal(int, int, str)     # current, total, status text
    # Emitted whenever the project list mutates (project added, closed,
    # or an existing project's reference flag changed). Subscribers
    # like the Reference Correction panel use it to keep their own
    # views in sync.
    projects_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        create_btn = QPushButton("+ Create Analysis Project")
        create_menu = QMenu(create_btn)
        sample_act = create_menu.addAction(
            "Sample Project",
            lambda: self._create_project(is_reference=False))
        sample_act.setToolTip(
            "Standard fit pipeline. Use for the isotope you want to "
            "measure; isotope shifts are computed from these.")
        ref_act = create_menu.addAction(
            "Reference Project",
            lambda: self._create_project(is_reference=True))
        ref_act.setToolTip(
            "Fit pipeline tagged as a reference -- its fitted "
            "centroids feed the Gaussian-process drift correction in "
            "the Isotope Shifts tab. Use for scans of a stable "
            "reference isotope taken throughout the campaign.")
        cal_act = create_menu.addAction(
            "Cooler Calibration Project",
            lambda: self._create_project(is_calibration=True))
        cal_act.setToolTip(
            "Fit pipeline for the Yb runs that calibrate the cooler "
            "voltage. Its hyperfine A constants are compared with "
            "their literature values across a grid of assumed cooler "
            "offsets in the Cooler Calibration tab, which is where "
            "the offset -- and its systematic range -- comes from. "
            "Not an isotope: it stays out of the shift table.")
        # Tooltips on QAction don't show in the menu by default --
        # enable hover tooltips so they're discoverable.
        create_menu.setToolTipsVisible(True)
        create_btn.setMenu(create_menu)
        toolbar.addWidget(create_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Project tabs
        self._project_tabs = QTabWidget()
        self._project_tabs.setTabsClosable(True)
        self._project_tabs.tabCloseRequested.connect(self._close_project)
        self._project_tabs.setMovable(True)
        self._project_tabs.tabBarDoubleClicked.connect(self._rename_project)
        tab_bar = self._project_tabs.tabBar()
        tab_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tab_bar.customContextMenuRequested.connect(self._tab_context_menu)
        layout.addWidget(self._project_tabs, 1)

        self._projects = []

        # Permanent Isotope Shifts tab (non-closable, hidden until projects exist)
        self._is_tab = IsotopeShiftTab(analysis_tab=self)
        self._is_tab.results_ready.connect(self.results_ready.emit)
        self._is_tab_visible = False
        # The GP panel is built by the IS tab (which owns its signals
        # and its save/load) but shown as a sibling of the isotope
        # projects rather than buried in a sub-tab (2026-09-20).
        self._gp_tab = self._is_tab._ref_corr_panel
        # Cooler Calibration: shown only while a calibration project
        # exists, because it has nothing to act on otherwise.
        from gui.analysis.cooler_calibration_tab import (
            CoolerCalibrationTab)
        self._cal_tab = CoolerCalibrationTab(analysis_tab=self)
        self._cal_tab.results_ready.connect(self.results_ready.emit)
        self._cal_tab_visible = False
        # Systematics: always there once projects exist. It acts on
        # the iterations already on disk, so it has something to show
        # whether or not a calibration was run this session.
        from gui.analysis.systematics_tab import SystematicsTab
        self._sys_tab = SystematicsTab(analysis_tab=self)
        self._sys_tab.results_ready.connect(self.results_ready.emit)
        self.projects_changed.connect(self._sys_tab.refresh_projects)

    def _create_project(self, is_reference=False, is_calibration=False):
        if is_calibration:
            title = "New Cooler Calibration Project"
            default = f"Yb_cal_{len(self._projects) + 1}"
            name, ok = QInputDialog.getText(
                self, title, "Project name:", text=default)
            if not ok or not name.strip():
                return
            self._add_project(name.strip(), is_calibration=True)
            return
        if is_reference:
            title = "New Reference Project"
            default = f"Reference_{len(self._projects) + 1}"
        else:
            title = "New Analysis Project"
            default = f"Project_{len(self._projects) + 1}"
        name, ok = QInputDialog.getText(
            self, title, "Project name:", text=default)
        if not ok or not name.strip():
            return
        self._add_project(name.strip(), is_reference=is_reference)

    #: The tabs that are not isotope projects: permanent, unclosable
    #: and always to the right of the projects.
    def _special_tabs(self):
        tabs = [self._is_tab, self._gp_tab]
        if self._cal_tab_visible:
            tabs.append(self._cal_tab)
        tabs.append(self._sys_tab)
        return tuple(tabs)

    def _wants_cooler_tab(self):
        return any(getattr(p, "is_calibration", False)
                   for p in self._projects)

    def _sync_cooler_tab(self):
        """Add or remove the Cooler Calibration tab to match the
        projects. Called wherever the project list changes."""
        want = self._wants_cooler_tab()
        if want and not self._cal_tab_visible:
            idx = self._project_tabs.addTab(self._cal_tab,
                                            "Cooler Calibration")
            self._project_tabs.tabBar().setTabButton(
                idx, QTabBar.ButtonPosition.RightSide, None)
            self._cal_tab_visible = True
            self._cal_tab.refresh_projects()
        elif not want and self._cal_tab_visible:
            idx = self._project_tabs.indexOf(self._cal_tab)
            if idx >= 0:
                self._project_tabs.removeTab(idx)
            self._cal_tab_visible = False

    def _is_special_tab(self, widget):
        """True for the Isotope Shifts / GP tabs, which cannot be
        closed, renamed or flagged as a reference project."""
        return widget in self._special_tabs()

    def _ensure_is_tab_last(self):
        """Keep the Isotope Shifts and GP tabs visible and last, in
        that order."""
        if not self._is_tab_visible:
            for widget, title in ((self._is_tab, "Isotope Shifts"),
                                  (self._gp_tab,
                                   "Reference Correction (GP)"),
                                  (self._sys_tab, "Systematics")):
                idx = self._project_tabs.addTab(widget, title)
                # No close button: they are part of the workspace,
                # not a project the user added.
                self._project_tabs.tabBar().setTabButton(
                    idx, QTabBar.ButtonPosition.RightSide, None)
            self._is_tab_visible = True
            self._sync_cooler_tab()
            return
        self._sync_cooler_tab()
        # Already there -- just push them back to the end, in order.
        for widget in self._special_tabs():
            cur = self._project_tabs.indexOf(widget)
            last = self._project_tabs.count() - 1
            if cur >= 0 and cur != last:
                self._project_tabs.tabBar().moveTab(cur, last)

    def _add_project(self, name, is_reference=False, config=None,
                     is_calibration=False):
        project = AnalysisProject(name, is_reference=is_reference,
                                  is_calibration=is_calibration)
        if config:
            project.from_dict(config)
        project.results_ready.connect(self.results_ready.emit)
        if hasattr(project, 'fit_progress'):
            project.fit_progress.connect(self.fit_progress.emit)
        self._projects.append(project)
        idx = self._project_tabs.addTab(project, self._tab_text(project))
        self._project_tabs.setCurrentIndex(idx)
        self._ensure_is_tab_last()
        from gui.theme import style_project_tab_bar
        style_project_tab_bar(self._project_tabs)
        self.projects_changed.emit()
        return project

    def _tab_text(self, project):
        """Tab title, with what kind of project it is after the name."""
        if getattr(project, "is_calibration", False):
            return f"{project.project_name} (Cooler Cal.)"
        if getattr(project, "is_reference", False):
            return f"{project.project_name} (Reference)"
        return project.project_name

    def _rename_project(self, index):
        """Double-click on a project tab opens an inline rename dialog.

        Skips the Isotope Shifts tab.
        """
        if index < 0:
            return
        widget = self._project_tabs.widget(index)
        if self._is_special_tab(widget) or widget not in self._projects:
            return
        old_name = widget.project_name
        name, ok = QInputDialog.getText(
            self, "Rename Project", "New project name:", text=old_name)
        if not ok:
            return
        name = name.strip()
        if not name or name == old_name:
            return
        widget._project_name = name
        self._project_tabs.setTabText(index, self._tab_text(widget))
        from gui.theme import style_project_tab_bar
        style_project_tab_bar(self._project_tabs)
        self.projects_changed.emit()

    #: The three kinds a project can be, with the label of the action
    #: that converts TO it and what that means. A calibration project
    #: used to be a one-way street: created from the menu, and the
    #: context menu only ever offered "reference". Arda asked for the
    #: full set (2026-09-25) -- the runs are the same files either way,
    #: and which role they play is a decision that changes.
    PROJECT_KINDS = (
        ("sample", "Convert to Sample Project",
         "An ordinary isotope project: its fitted centroids are "
         "corrected by the reference drift model and enter the "
         "isotope shifts."),
        ("reference", "Convert to Reference Project",
         "Its fitted centroids feed the Gaussian-process drift "
         "correction in the Isotope Shifts tab."),
        ("calibration", "Convert to Cooler Calibration Project",
         "Yb runs whose hyperfine A constants pin the cooler-voltage "
         "offset. Kept out of the isotope-shift table and the GP "
         "lists, and picked up by the Cooler Calibration tab."),
    )

    @staticmethod
    def project_kind(project):
        """Which of PROJECT_KINDS this project is."""
        if getattr(project, "is_calibration", False):
            return "calibration"
        if getattr(project, "is_reference", False):
            return "reference"
        return "sample"

    def _tab_context_menu(self, pos):
        """Right-click menu on a project tab: convert between the three
        kinds, plus rename / close. Skips the permanent tabs."""
        tab_bar = self._project_tabs.tabBar()
        index = tab_bar.tabAt(pos)
        if index < 0:
            return
        widget = self._project_tabs.widget(index)
        if self._is_special_tab(widget) or widget not in self._projects:
            return

        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        current = self.project_kind(widget)
        for kind, label, tip in self.PROJECT_KINDS:
            if kind == current:
                continue
            act = menu.addAction(label)
            act.setToolTip(tip + " Keeps all existing blocks and fits.")
            act.triggered.connect(
                lambda _=False, i=index, k=kind: self._set_project_kind(i, k))
        menu.addSeparator()
        rename_act = menu.addAction("Rename...")
        rename_act.triggered.connect(
            lambda _=False, i=index: self._rename_project(i))
        close_act = menu.addAction("Close")
        close_act.triggered.connect(
            lambda _=False, i=index: self._close_project(i))
        menu.exec(tab_bar.mapToGlobal(pos))

    def _set_project_kind(self, index, kind):
        """Make the project at ``index`` a sample, reference or
        calibration project, and refresh everything that reads those
        flags: the tab title, the Cooler Calibration tab's existence
        and its project table, and the GP / isotope-shift lists.
        """
        if index < 0 or kind not in dict(
                (k, lbl) for k, lbl, _t in self.PROJECT_KINDS):
            return
        widget = self._project_tabs.widget(index)
        if self._is_special_tab(widget) or widget not in self._projects:
            return
        widget._is_reference = (kind == "reference")
        widget._is_calibration = (kind == "calibration")
        self._project_tabs.setTabText(index, self._tab_text(widget))
        # The calibration tab exists only while a calibration project
        # does, and it lists them by name.
        self._sync_cooler_tab()
        if self._cal_tab_visible:
            self._cal_tab.refresh_projects()
        self.projects_changed.emit()

    def _toggle_project_reference(self, index):
        """Flip between sample and reference. Kept because it is the
        older name for what the context menu used to do."""
        widget = self._project_tabs.widget(index)
        if self._is_special_tab(widget) or widget not in self._projects:
            return
        self._set_project_kind(
            index,
            "sample" if self.project_kind(widget) == "reference"
            else "reference")

    def _close_project(self, index):
        widget = self._project_tabs.widget(index)
        if self._is_special_tab(widget):
            return  # Isotope Shifts / GP are permanent
        reply = QMessageBox.question(
            self, "Close Project",
            f"Close project '{self._project_tabs.tabText(index)}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._project_tabs.removeTab(index)
        if widget in self._projects:
            self._projects.remove(widget)
        widget.deleteLater()
        # Hide both permanent tabs when no projects remain -- they
        # have nothing to act on.
        if not self._projects and self._is_tab_visible:
            for w in self._special_tabs():
                idx = self._project_tabs.indexOf(w)
                if idx >= 0:
                    self._project_tabs.removeTab(idx)
            self._is_tab_visible = False
        self.projects_changed.emit()

    def save_config(self):
        """Save all project configurations."""
        from gui.shared_widgets import get_last_dir, remember_last_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Configuration",
            get_last_dir("config", "save"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        if not path.lower().endswith(('.yaml', '.yml')):
            path += '.yaml'
        remember_last_dir("config", "save", path)
        d = {"analysis": {
            "projects": [p.to_dict() for p in self._projects],
            "isotope_shifts": self._is_tab.to_dict(),
            "cooler_calibration": self._cal_tab.to_dict(),
            "systematics": self._sys_tab.to_dict(),
        }}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)

    def load_config(self):
        """Load project configurations."""
        from gui.shared_widgets import get_last_dir, remember_last_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Analysis Configuration",
            get_last_dir("config", "load"),
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        remember_last_dir("config", "load", path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load config:\n{e}")
            return
        cfg = raw.get("analysis", raw)
        # Same cost as a full session load -- every run named in the
        # file is opened -- so say what is happening here too.
        import os
        from gui.load_progress import begin as _begin_progress
        from gui.load_progress import report
        defs = cfg.get("projects", [])
        # Painting frozen while the tabs are rebuilt -- see the same
        # guard in MainWindow._load_from_path.
        self.setUpdatesEnabled(False)
        try:
            with _begin_progress(
                    self, os.path.basename(path),
                    [("projects", "Analysis projects", 1)]):
                for i, pd in enumerate(defs):
                    name = pd.get("project_name", "Project")
                    report(name, i, len(defs))
                    self._add_project(name, config=pd)
        finally:
            self.setUpdatesEnabled(True)
            self.update()
        is_data = cfg.get("isotope_shifts")
        if is_data:
            self._is_tab.from_dict(is_data)
        cal_data = cfg.get("cooler_calibration")
        if cal_data:
            self._sync_cooler_tab()
            self._cal_tab.from_dict(cal_data)
        sys_data = cfg.get("systematics")
        if sys_data:
            self._sys_tab.from_dict(sys_data)
