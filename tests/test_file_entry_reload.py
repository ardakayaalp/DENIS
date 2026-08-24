"""Pre-Analysis FileEntry per-file reload button and its handler.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

A FileEntry exposes a reload button (and reload_requested signal) so
users watching ongoing scans can pick up new events without removing and
re-adding the file. The button is hidden on a MergedFileEntry, which has
no on-disk ASDF to re-read. PreAnalysisTab._reload_file_entry re-reads
the ASDF via _load_cls_data, then refreshes the detail label and the
spectrum plot.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_file_entry_reload -v

Depends on: gui.preanalysis_tab (FileEntry, MergedFileEntry,
PreAnalysisTab); PySide6 for QApplication.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class FileEntryReloadButtonTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_file_entry_has_reload_button_and_signal(self):
        from gui.preanalysis_tab import FileEntry

        fe = FileEntry(
            filepath="/data/run_X.asdf",
            run_number="X", cooler_v=29895.3, date="2026-05-13",
            laser_sp=15975.4150, mass_amu=47.95)
        self.assertTrue(hasattr(fe, "reload_btn"))
        self.assertTrue(hasattr(fe, "reload_requested"))
        # The reload button must be visible on a regular file entry.
        self.assertTrue(fe.reload_btn.isVisibleTo(fe))

    def test_merged_file_entry_hides_reload_button(self):
        """MergedFileEntry has no on-disk ASDF backing it; the
        reload button would have no consumer there. Hidden so the
        UI doesn't tease an action that would crash."""
        from gui.preanalysis_tab import MergedFileEntry

        mfe = MergedFileEntry(
            name="m", merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]), merge_domain="voltage",
            source_info=[("a", "/p/a"), ("b", "/p/b")])
        # Inherited from FileEntry but hidden in MergedFileEntry.__init__.
        self.assertTrue(hasattr(mfe, "reload_btn"))
        self.assertFalse(mfe.reload_btn.isVisibleTo(mfe))

    def test_clicking_reload_emits_signal_with_entry(self):
        from gui.preanalysis_tab import FileEntry

        fe = FileEntry(
            filepath="/data/run_X.asdf",
            run_number="X", cooler_v=29895.3, date="2026-05-13",
            laser_sp=15975.4150, mass_amu=47.95)
        captured = []
        fe.reload_requested.connect(captured.append)
        fe.reload_btn.click()
        self.assertEqual(captured, [fe])


class PaReloadHandlerTests(unittest.TestCase):
    """PA's ``_reload_file_entry`` re-reads the ASDF via
    ``_load_cls_data`` then refreshes the detail label and the
    spectrum plot. Mocks the heavy clstools work to keep the test
    headless."""

    def setUp(self):
        _ensure_app()

    def test_reload_calls_load_cls_data_and_refresh(self):
        from gui.preanalysis_tab import PreAnalysisTab, FileEntry

        pa = PreAnalysisTab()
        fe = FileEntry(
            filepath="/data/run_X.asdf",
            run_number="X", cooler_v=29895.3, date="2026-05-13",
            laser_sp=15975.4150, mass_amu=47.95)
        pa._file_entries.append(fe)
        # Force clstools to "available" for the path we exercise.
        # (_get_clstools() replaced the old _HAS_CLSTOOLS flag when the
        # import went lazy for startup perf; any truthy module works.)
        with patch("gui.preanalysis_tab._get_clstools",
                   return_value=object()), \
                patch.object(pa, "_load_cls_data") as mock_load, \
                patch.object(fe, "update_detail") as mock_update, \
                patch.object(pa, "_schedule_replot") as mock_replot:
            pa._reload_file_entry(fe)
            mock_load.assert_called_once_with(fe)
            mock_update.assert_called_once()
            mock_replot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
