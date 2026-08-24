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

    def _create_project(self, is_reference=False):
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

    def _ensure_is_tab_last(self):
        """Ensure the Isotope Shifts tab is visible and always last."""
        if self._is_tab_visible:
            # Move to end if not already there
            cur = self._project_tabs.indexOf(self._is_tab)
            last = self._project_tabs.count() - 1
            if cur >= 0 and cur != last:
                self._project_tabs.tabBar().moveTab(cur, last)
        else:
            is_idx = self._project_tabs.addTab(
                self._is_tab, "Isotope Shifts")
            self._project_tabs.tabBar().setTabButton(
                is_idx, QTabBar.ButtonPosition.RightSide, None)
            self._is_tab_visible = True

    def _add_project(self, name, is_reference=False, config=None):
        project = AnalysisProject(name, is_reference=is_reference)
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
        """Tab title with a (Reference) suffix for reference projects."""
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
        if widget is self._is_tab or widget not in self._projects:
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

    def _tab_context_menu(self, pos):
        """Right-click menu on a project tab. Offers sample/reference
        conversion plus the existing rename/close actions. Skips the
        Isotope Shifts tab."""
        tab_bar = self._project_tabs.tabBar()
        index = tab_bar.tabAt(pos)
        if index < 0:
            return
        widget = self._project_tabs.widget(index)
        if widget is self._is_tab or widget not in self._projects:
            return

        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        if widget.is_reference:
            convert_act = menu.addAction("Convert to Sample Project")
            convert_act.setToolTip(
                "Untag this project as a reference. Its fitted centroids "
                "will no longer feed the Gaussian-process drift correction.")
        else:
            convert_act = menu.addAction("Convert to Reference Project")
            convert_act.setToolTip(
                "Tag this project as a reference. Its fitted centroids "
                "will feed the Gaussian-process drift correction in the "
                "Isotope Shifts tab. Keeps all existing blocks and fits.")
        convert_act.triggered.connect(
            lambda _=False, i=index: self._toggle_project_reference(i))
        menu.addSeparator()
        rename_act = menu.addAction("Rename...")
        rename_act.triggered.connect(
            lambda _=False, i=index: self._rename_project(i))
        close_act = menu.addAction("Close")
        close_act.triggered.connect(
            lambda _=False, i=index: self._close_project(i))
        menu.exec(tab_bar.mapToGlobal(pos))

    def _toggle_project_reference(self, index):
        """Flip the is_reference flag on the project at ``index`` and
        refresh dependent UI (tab title, Reference Correction panel)."""
        if index < 0:
            return
        widget = self._project_tabs.widget(index)
        if widget is self._is_tab or widget not in self._projects:
            return
        widget._is_reference = not widget._is_reference
        self._project_tabs.setTabText(index, self._tab_text(widget))
        self.projects_changed.emit()

    def _close_project(self, index):
        widget = self._project_tabs.widget(index)
        if widget is self._is_tab:
            return  # Cannot close the Isotope Shifts tab
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
        # Hide IS tab when no projects remain
        if not self._projects and self._is_tab_visible:
            is_idx = self._project_tabs.indexOf(self._is_tab)
            if is_idx >= 0:
                self._project_tabs.removeTab(is_idx)
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
        for pd in cfg.get("projects", []):
            self._add_project(pd.get("project_name", "Project"), config=pd)
        is_data = cfg.get("isotope_shifts")
        if is_data:
            self._is_tab.from_dict(is_data)
