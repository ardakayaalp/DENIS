"""Multi-project container for the Pre-Analysis tab.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Hosts a closable/movable tab bar of independent ``PreAnalysisTab``
projects, with toolbar actions to create, rename, and close them, and
handles serializing every project to (and restoring from) the session
YAML, including the legacy single-project flat layout.

Depends on: gui.preanalysis_tab (PreAnalysisTab).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QMessageBox, QInputDialog,
)

from gui.preanalysis_tab import PreAnalysisTab


class PreAnalysisContainer(QWidget):
    """Wrapper that holds multiple Pre-Analysis project tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        create_btn = QPushButton("+ Create Pre-Analysis")
        create_btn.clicked.connect(self._create_project)
        toolbar.addWidget(create_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Project tabs
        self._project_tabs = QTabWidget()
        self._project_tabs.setTabsClosable(True)
        self._project_tabs.tabCloseRequested.connect(self._close_project)
        self._project_tabs.setMovable(True)
        self._project_tabs.tabBarDoubleClicked.connect(self._rename_project)
        layout.addWidget(self._project_tabs, 1)

        self._projects = []
        # Session-global plot layout (mirrored onto every project).
        self._plot_layout_mode = "stacked"
        # Session-global dark-mode plots. Seed from the global settings
        # file so a fresh session opens the way the user last left it.
        try:
            from gui.shared_widgets import _load_settings
            self._pa_dark_mode = bool(
                _load_settings().get("pa_dark_plots", False))
        except Exception:
            self._pa_dark_mode = False
        # Session-global spectrum-tab grid toggles.
        self._pa_grid_x = False
        self._pa_grid_y = False

    # ── Project management ──

    def _sync_plot_layout(self, mode):
        """Apply one plot-layout mode to every open project (the
        setting is session-global, not per-project)."""
        self._plot_layout_mode = mode
        for p in self._projects:
            p._set_plot_layout(mode, replot=True)

    def _sync_dark_mode(self, on):
        """Apply one dark-mode choice to every open project (the setting
        is session-global, not per-project)."""
        self._pa_dark_mode = bool(on)
        for p in self._projects:
            # Skip the project that already flipped (the one that emitted
            # the change) so it isn't redrawn twice.
            if p._dark_mode == bool(on):
                continue
            p._set_dark_mode(on, replot=True)

    def _sync_grids(self, x_on, y_on):
        """Apply one grid choice to every open project (session-global)."""
        self._pa_grid_x = bool(x_on)
        self._pa_grid_y = bool(y_on)
        for p in self._projects:
            if p._grid_x == bool(x_on) and p._grid_y == bool(y_on):
                continue   # already there (e.g. the originating project)
            p._set_grids(x_on, y_on)

    def _create_project(self):
        name, ok = QInputDialog.getText(
            self, "New Pre-Analysis Project", "Project name:",
            text=f"PA_{len(self._projects) + 1}")
        if not ok or not name.strip():
            return
        self._add_project(name.strip())

    def _add_project(self, name, config=None):
        project = PreAnalysisTab()
        if config:
            project._restore_from_dict(config)
            # A restored project brings its saved layout / dark-mode
            # along; adopt them as the session modes so all stay in step.
            self._plot_layout_mode = project._plot_layout_mode
            self._pa_dark_mode = project._dark_mode
            self._pa_grid_x = project._grid_x
            self._pa_grid_y = project._grid_y
        self._projects.append(project)
        # Layout + dark mode + grids are session-global: new projects
        # adopt the current modes, and any project's switch re-syncs all.
        project._set_plot_layout(self._plot_layout_mode, replot=False)
        project._set_dark_mode(self._pa_dark_mode, replot=False)
        project._set_grids(self._pa_grid_x, self._pa_grid_y)
        project.layout_mode_changed.connect(self._sync_plot_layout)
        project.dark_mode_changed.connect(self._sync_dark_mode)
        project.grid_changed.connect(self._sync_grids)
        idx = self._project_tabs.addTab(project, name)
        self._project_tabs.setCurrentIndex(idx)
        from gui.theme import style_project_tab_bar
        style_project_tab_bar(self._project_tabs)
        return project

    def _close_project(self, index):
        reply = QMessageBox.question(
            self, "Close Project",
            f"Close pre-analysis '{self._project_tabs.tabText(index)}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        widget = self._project_tabs.widget(index)
        self._project_tabs.removeTab(index)
        if widget in self._projects:
            self._projects.remove(widget)
        widget.deleteLater()

    def _rename_project(self, index):
        if index < 0:
            return
        old_name = self._project_tabs.tabText(index)
        name, ok = QInputDialog.getText(
            self, "Rename Project", "New name:", text=old_name)
        if ok and name.strip():
            self._project_tabs.setTabText(index, name.strip())
            from gui.theme import style_project_tab_bar
            style_project_tab_bar(self._project_tabs)

    def current_project(self):
        """Return the currently active PreAnalysisTab, or None."""
        w = self._project_tabs.currentWidget()
        if isinstance(w, PreAnalysisTab):
            return w
        return None

    # ── Serialization ──

    def _build_config_dict(self):
        """Serialize all projects for YAML save."""
        projects = []
        for i, p in enumerate(self._projects):
            d = p._build_config_dict().get("preanalysis", {})
            d["name"] = self._project_tabs.tabText(
                self._project_tabs.indexOf(p))
            projects.append(d)
        return {"projects": projects}

    def _restore_from_dict(self, data):
        """Restore projects from a config dict (handles legacy format)."""
        # Clear existing
        while self._project_tabs.count() > 0:
            widget = self._project_tabs.widget(0)
            self._project_tabs.removeTab(0)
            if widget in self._projects:
                self._projects.remove(widget)
            widget.deleteLater()

        if "projects" in data:
            # New multi-project format
            for pd in data["projects"]:
                pd = dict(pd)  # copy to avoid mutating saved data
                name = pd.pop("name", f"PA_{len(self._projects) + 1}")
                self._add_project(name, config=pd)
        else:
            # Legacy: flat dict with files, plot_options, etc.
            self._add_project("PA_1", config=data)
        # Layout + dark mode are session-global: saves written before
        # that rule (or hand-edited) may carry differing per-project
        # modes — align everyone to the adopted session mode.
        self._sync_plot_layout(self._plot_layout_mode)
        self._sync_dark_mode(self._pa_dark_mode)
        self._sync_grids(self._pa_grid_x, self._pa_grid_y)
