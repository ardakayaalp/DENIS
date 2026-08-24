"""Tests that an Analysis import pulls only the TICKED Pre-Analysis entries.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Verifies that SourceBlock._import_files_from_preanalysis honours each
Pre-Analysis file entry's checkbox: unticked entries are skipped,
import-all aborts with a warning when nothing is ticked, and
PreAnalysisTab.checked_file_entries is the single filter both paths
share (treating an entry with no checkbox as ticked).

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_import_only_checked -v

Depends on: gui.analysis.blocks (SourceBlock, module-level
_choose_pa_project / QMessageBox), gui.preanalysis_tab (PreAnalysisTab);
PySide6 for QApplication and a real QCheckBox.
"""

import unittest
from unittest.mock import MagicMock

from PySide6.QtWidgets import QApplication, QCheckBox


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class _FakeFE:
    """isinstance(_FakeFE(), MergedFileEntry) is False, so these flow
    through the regular-file import branch."""
    pass


def _fake_entry(path, checked=True):
    """A minimal Pre-Analysis file entry: a real QCheckBox + a filepath.

    A real QCheckBox (not a MagicMock) makes .check.isChecked() a true
    boolean, exercising the real ticked/unticked branch.
    """
    fe = _FakeFE()
    fe.filepath = path
    cb = QCheckBox()
    cb.setChecked(checked)
    fe.check = cb
    return fe


def _stub_add_file_entry(sb):
    """Stub SourceBlock._add_file_entry to avoid filesystem access while
    still producing entries shaped like the real ones (a 'checkbox' key
    is required by get_checked_files / _update_nav_label)."""
    def _add(p):
        cb = QCheckBox()
        cb.setChecked(True)
        sb._file_entries.append(
            {"path": p, "is_merged": False, "checkbox": cb,
             "run_number": "?"})
    sb._add_file_entry = MagicMock(side_effect=_add)


class _FakePA:
    """Fake PreAnalysisTab exposing _file_entries + checked_file_entries."""

    def __init__(self, entries):
        self._file_entries = entries
        self._e_lower = MagicMock(value=MagicMock(return_value=0.0))
        self._e_upper = MagicMock(value=MagicMock(return_value=25068.222))
        self._harmonic = MagicMock(value=MagicMock(return_value=2))

    def checked_file_entries(self):
        return [fe for fe in self._file_entries
                if getattr(fe, 'check', None) is None
                or fe.check.isChecked()]


class ImportOnlyCheckedTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_unchecked_entry_is_skipped(self):
        from gui.analysis.blocks import SourceBlock

        e1 = _fake_entry("/data/run_1.asdf", checked=True)
        e2 = _fake_entry("/data/run_2.asdf", checked=False)
        pa = _FakePA([e1, e2])

        sb = SourceBlock()
        _stub_add_file_entry(sb)
        sb._import_files_from_preanalysis(pa)

        paths = [e["path"] for e in sb._file_entries]
        self.assertEqual(paths, ["/data/run_1.asdf"])

    def test_real_filter_imports_only_ticked(self):
        from gui.analysis.blocks import SourceBlock

        e1 = _fake_entry("/data/a.asdf", checked=False)
        e2 = _fake_entry("/data/b.asdf", checked=True)
        e3 = _fake_entry("/data/c.asdf", checked=False)
        pa = _FakePA([e1, e2, e3])

        sb = SourceBlock()
        _stub_add_file_entry(sb)
        sb._import_files_from_preanalysis(pa)

        paths = [e["path"] for e in sb._file_entries]
        self.assertEqual(paths, ["/data/b.asdf"])

    def test_import_all_aborts_when_nothing_ticked(self):
        import gui.analysis.blocks as blocks_mod
        from gui.analysis.blocks import SourceBlock

        e1 = _fake_entry("/data/a.asdf", checked=False)
        e2 = _fake_entry("/data/b.asdf", checked=False)
        pa = _FakePA([e1, e2])

        sb = SourceBlock()
        _stub_add_file_entry(sb)

        orig_choose = blocks_mod._choose_pa_project
        orig_warning = blocks_mod.QMessageBox.warning
        warned = {}

        def _fake_warning(parent, title, text, *a, **k):
            warned["title"] = title
            warned["text"] = text
            return None

        blocks_mod._choose_pa_project = lambda parent: pa
        blocks_mod.QMessageBox.warning = staticmethod(_fake_warning)
        try:
            sb._import_all_from_preanalysis()
        finally:
            blocks_mod._choose_pa_project = orig_choose
            blocks_mod.QMessageBox.warning = orig_warning

        # Nothing imported; warning shown.
        self.assertEqual(sb._file_entries, [])
        self.assertEqual(sb._add_file_entry.call_count, 0)
        self.assertIn("ticked", warned.get("text", "").lower())

    def test_import_all_runs_when_something_ticked(self):
        import gui.analysis.blocks as blocks_mod
        from gui.analysis.blocks import SourceBlock

        e1 = _fake_entry("/data/a.asdf", checked=True)
        e2 = _fake_entry("/data/b.asdf", checked=False)
        pa = _FakePA([e1, e2])

        sb = SourceBlock()
        _stub_add_file_entry(sb)

        orig_choose = blocks_mod._choose_pa_project
        blocks_mod._choose_pa_project = lambda parent: pa
        try:
            sb._import_all_from_preanalysis()
        finally:
            blocks_mod._choose_pa_project = orig_choose

        paths = [e["path"] for e in sb._file_entries]
        self.assertEqual(paths, ["/data/a.asdf"])


class CheckedFileEntriesHelperTests(unittest.TestCase):
    """PreAnalysisTab.checked_file_entries() is the single source of
    truth for which entries an import pulls."""

    def setUp(self):
        _ensure_app()

    def test_helper_filters_unticked(self):
        from gui.preanalysis_tab import PreAnalysisTab

        # The method only reads _file_entries, so a bare instance is fine.
        pa = PreAnalysisTab.__new__(PreAnalysisTab)
        e1 = _fake_entry("/x/1", checked=True)
        e2 = _fake_entry("/x/2", checked=False)
        e3 = _fake_entry("/x/3", checked=True)
        pa._file_entries = [e1, e2, e3]

        out = PreAnalysisTab.checked_file_entries(pa)
        self.assertEqual([fe.filepath for fe in out], ["/x/1", "/x/3"])

    def test_helper_treats_missing_checkbox_as_ticked(self):
        from gui.preanalysis_tab import PreAnalysisTab

        pa = PreAnalysisTab.__new__(PreAnalysisTab)
        no_cb = MagicMock()
        no_cb.filepath = "/x/nocb"
        no_cb.check = None
        pa._file_entries = [no_cb]

        out = PreAnalysisTab.checked_file_entries(pa)
        self.assertEqual([fe.filepath for fe in out], ["/x/nocb"])


if __name__ == "__main__":
    unittest.main()
