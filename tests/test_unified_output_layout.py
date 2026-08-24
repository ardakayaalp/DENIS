"""Output directory layout and the Results-tab project-filtered refresh.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Verifies the shared_widgets helpers that locate the output root (from
settings, defaulting to ./output) and its fixed estimates/ and analysis/
subfolders, and the Results tab's refresh paths: Refresh All scans every
project under <root>/analysis, while Refresh Project scans only the
Analysis tab's active project (resolved by walking up to the main
window's analysis_tab).

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_unified_output_layout -v

Depends on: gui.shared_widgets (get_output_root, get_estimates_dir,
get_analysis_dir, _load_settings), gui.results_tab (ResultsTab);
PySide6 for QApplication.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np  # noqa: F401  (imported by GUI modules under test)

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class OutputDirHelpersTests(unittest.TestCase):
    """Helpers in shared_widgets are the single source of truth for
    where outputs land. Settings drives the root; subfolders are
    fixed."""

    def test_root_uses_setting_or_default(self):
        from gui import shared_widgets as sw
        with patch.object(sw, "_load_settings",
                            return_value={"output_directory": "/x/root"}):
            self.assertEqual(sw.get_output_root(), "/x/root")
        with patch.object(sw, "_load_settings", return_value={}):
            self.assertEqual(sw.get_output_root(), "./output")

    def test_subdirs_are_under_root(self):
        from gui import shared_widgets as sw
        with patch.object(sw, "_load_settings",
                            return_value={"output_directory": "/x/root"}):
            self.assertEqual(sw.get_estimates_dir(),
                              os.path.join("/x/root", "estimates"))
            self.assertEqual(sw.get_analysis_dir(),
                              os.path.join("/x/root", "analysis"))


class ResultsTabRefreshFilterTests(unittest.TestCase):
    """The Refresh Project path scans only the active project's
    subdirectory under ``<root>/analysis``. Refresh All scans every
    project."""

    def setUp(self):
        _ensure_app()

    def _make_layout(self, root, projects):
        """Build a fake ``<root>/analysis/<project>/iter_001/`` tree
        for each project name."""
        analysis_dir = os.path.join(root, "analysis")
        for name in projects:
            os.makedirs(
                os.path.join(analysis_dir, name, "iter_001"),
                exist_ok=True)
        return analysis_dir

    def test_refresh_all_picks_up_every_project(self):
        from gui.results_tab import ResultsTab

        with tempfile.TemporaryDirectory() as root:
            self._make_layout(root, ["projA", "projB", "projC"])
            tab = ResultsTab()
            with patch("gui.shared_widgets._load_settings",
                        return_value={"output_directory": root}):
                tab._refresh_from_disk()
            self.assertEqual(
                sorted(tab._results_data.keys()),
                ["projA", "projB", "projC"])

    def test_refresh_with_project_filter_picks_up_only_that_project(self):
        from gui.results_tab import ResultsTab

        with tempfile.TemporaryDirectory() as root:
            self._make_layout(root, ["projA", "projB", "projC"])
            tab = ResultsTab()
            with patch("gui.shared_widgets._load_settings",
                        return_value={"output_directory": root}):
                tab._refresh_from_disk(project_filter="projB")
            self.assertEqual(
                list(tab._results_data.keys()), ["projB"])
            self.assertIn("iter_001", tab._results_data["projB"])

    def test_refresh_active_project_resolves_active_via_analysis_tab(self):
        """``_refresh_active_project`` walks up to the main window's
        ``analysis_tab`` and uses the current QTabWidget child to
        find the active project name. Mocks the chain so the test
        doesn't depend on the full AnalysisTab construction."""
        from gui.results_tab import ResultsTab

        with tempfile.TemporaryDirectory() as root:
            self._make_layout(root, ["projA", "projB"])
            tab = ResultsTab()

            # Fake main window with an analysis_tab whose
            # _project_tabs.currentWidget returns a project with
            # _project_name = "projB".
            fake_project = MagicMock()
            fake_project._project_name = "projB"
            fake_tabs = MagicMock()
            fake_tabs.currentWidget.return_value = fake_project
            fake_analysis = MagicMock()
            fake_analysis._project_tabs = fake_tabs
            fake_window = MagicMock()
            fake_window.analysis_tab = fake_analysis

            with patch.object(tab, "window", return_value=fake_window), \
                    patch("gui.shared_widgets._load_settings",
                           return_value={"output_directory": root}):
                tab._refresh_active_project()

            self.assertEqual(
                list(tab._results_data.keys()), ["projB"])


if __name__ == "__main__":
    unittest.main()
