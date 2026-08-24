"""Unit tests for Results-tab checkboxes, bulk delete, and notes cascade.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers the Results-tab hierarchy-aware checkboxes (cascade down,
partial-parent state), Select-all/Unselect-all, bulk "Remove selected"
deletion of checked subtrees, and the orphaned-notes cascade on delete.
Runs headless by forcing the offscreen Qt platform and reusing any
existing QApplication.

Tree shape (built by ResultsTab._rebuild_tree):
    Project
        |- 'Notes'          (project-level note, type=notes)
        |- iter_NNN
        |     |- 'Notes'    (iteration-level note, type=notes)
        |     |- <file items...>
        ...
A project's direct children are its own Notes item plus its iterations;
an iteration's direct children are its Notes item plus any file items.

Depends on: gui.results_tab (ResultsTab), gui.shared_widgets; PySide6
and pytest fixtures for the headless QApplication.
"""
import os

import pytest

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt

from gui.results_tab import ResultsTab
import gui.shared_widgets as shared_widgets


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _silence_dialogs(monkeypatch):
    """Auto-answer the confirmation dialogs so tests never block."""
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: None))


def _build_tab(tmp_path, monkeypatch, layout):
    """Create a ResultsTab whose analysis dir is `tmp_path`, populated
    with `layout` = {project: [iter_name, ...]} including on-disk dirs
    and notes files. Returns (tab, base_dir).
    """
    base_dir = str(tmp_path)
    # _rebuild_tree / add_results pull get_analysis_dir lazily from the
    # module, so patch it there.
    monkeypatch.setattr(shared_widgets, "get_analysis_dir",
                        lambda: base_dir)

    tab = ResultsTab()
    for project, iters in layout.items():
        project_dir = os.path.join(base_dir, project)
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, "_project_notes.txt"),
                  "w", encoding="utf-8") as f:
            f.write(f"notes for {project}")
        tab._results_data[project] = {}
        for iter_name in iters:
            iter_dir = os.path.join(project_dir, iter_name)
            os.makedirs(iter_dir, exist_ok=True)
            with open(os.path.join(iter_dir, "_iter_notes.txt"),
                      "w", encoding="utf-8") as f:
                f.write(f"notes for {project}/{iter_name}")
            tab._results_data[project][iter_name] = {
                "results": [], "output_config": {},
                "directory": iter_dir,
            }
    tab._rebuild_tree()
    return tab, base_dir


def _project_item(tab, name):
    for i in range(tab._tree.topLevelItemCount()):
        it = tab._tree.topLevelItem(i)
        data = it.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and data.get("project") == name \
                and "iteration" not in data:
            return it
    raise AssertionError(f"project item {name!r} not found")


def _iteration_item(tab, project, iteration):
    pit = _project_item(tab, project)
    for c in range(pit.childCount()):
        ch = pit.child(c)
        data = ch.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and data.get("iteration") == iteration:
            return ch
    raise AssertionError(
        f"iteration item {project}/{iteration} not found")


# ── (a) notes cascade on delete ────────────────────────────────────

def test_delete_iteration_removes_its_notes(tmp_path, monkeypatch, qapp):
    tab, base = _build_tab(tmp_path, monkeypatch,
                           {"P": ["iter_001", "iter_002"]})
    iter_dir = os.path.join(base, "P", "iter_001")
    note = os.path.join(iter_dir, "_iter_notes.txt")
    assert os.path.isfile(note)

    tab._delete_iteration("P", "iter_001", iter_dir, confirm=False)

    assert not os.path.exists(iter_dir)
    assert not os.path.exists(note)
    # Other iteration's note intact.
    assert os.path.isfile(
        os.path.join(base, "P", "iter_002", "_iter_notes.txt"))


def test_delete_iteration_scrubs_note_when_rmtree_skipped(
        tmp_path, monkeypatch, qapp):
    # The genuine orphan the old code missed: when os.path.isdir(iter_dir)
    # is False the rmtree branch is skipped, but a stray _iter_notes.txt
    # still sits there. The explicit cascade must remove it.
    tab, base = _build_tab(tmp_path, monkeypatch, {"P": ["iter_001"]})
    iter_dir = os.path.join(base, "P", "iter_001")
    note = os.path.join(iter_dir, "_iter_notes.txt")

    real_isdir = os.path.isdir
    monkeypatch.setattr(
        os.path, "isdir",
        lambda p: False if os.path.normpath(p) == os.path.normpath(
            iter_dir) else real_isdir(p))
    assert not os.path.isdir(iter_dir)   # rmtree branch skipped
    assert real_isdir(iter_dir)          # but the note is really there

    tab._delete_iteration("P", "iter_001", iter_dir, confirm=False)

    assert not os.path.exists(note)      # cascade scrubbed the orphan
    assert "P" not in tab._results_data  # model entry cleaned up


def test_delete_iteration_notes_helper_removes_note(
        tmp_path, monkeypatch, qapp):
    # Direct unit test of the Part-A cascade helper.
    tab, base = _build_tab(tmp_path, monkeypatch, {"P": ["iter_001"]})
    orphan_iter = os.path.join(base, "P", "orphan")
    os.makedirs(orphan_iter, exist_ok=True)
    note = os.path.join(orphan_iter, "_iter_notes.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write("orphan note")
    tab._delete_iteration_notes(orphan_iter)
    assert not os.path.exists(note)
    # Idempotent: a second call (file already gone) does not raise.
    tab._delete_iteration_notes(orphan_iter)


def test_delete_project_removes_all_child_notes(
        tmp_path, monkeypatch, qapp):
    tab, base = _build_tab(tmp_path, monkeypatch,
                           {"P": ["iter_001", "iter_002"]})
    proj_note = os.path.join(base, "P", "_project_notes.txt")
    i1_note = os.path.join(base, "P", "iter_001", "_iter_notes.txt")
    i2_note = os.path.join(base, "P", "iter_002", "_iter_notes.txt")
    assert all(os.path.isfile(p) for p in (proj_note, i1_note, i2_note))

    tab._delete_project("P", os.path.join(base, "P"), confirm=False)

    assert not os.path.exists(os.path.join(base, "P"))
    for p in (proj_note, i1_note, i2_note):
        assert not os.path.exists(p)
    assert "P" not in tab._results_data


# ── (b) cascade down ───────────────────────────────────────────────

def test_checking_project_checks_all_descendants(
        tmp_path, monkeypatch, qapp):
    tab, _ = _build_tab(tmp_path, monkeypatch,
                        {"P": ["iter_001", "iter_002"]})
    pit = _project_item(tab, "P")
    pit.setCheckState(0, Qt.CheckState.Checked)

    assert pit.checkState(0) == Qt.CheckState.Checked

    def _all_checked(item):
        if item.checkState(0) != Qt.CheckState.Checked:
            return False
        return all(_all_checked(item.child(c))
                   for c in range(item.childCount()))
    assert _all_checked(pit)


# ── (c) partial parent when not all children checked ───────────────

def test_partial_parent_when_not_all_children_checked(
        tmp_path, monkeypatch, qapp):
    # A project's direct children are its own Notes item PLUS its
    # iterations, so 'all children checked' means notes + every
    # iteration. Leaving any one unchecked keeps the parent partial.
    tab, _ = _build_tab(tmp_path, monkeypatch,
                        {"P": ["iter_001", "iter_002", "iter_003"]})
    pit = _project_item(tab, "P")
    n = pit.childCount()
    assert n >= 2  # project-notes + iterations

    # Check all but the last direct child -> parent PartiallyChecked.
    for c in range(n - 1):
        pit.child(c).setCheckState(0, Qt.CheckState.Checked)
    assert pit.checkState(0) == Qt.CheckState.PartiallyChecked

    # Checking the final child promotes the parent to fully Checked.
    pit.child(n - 1).setCheckState(0, Qt.CheckState.Checked)
    assert pit.checkState(0) == Qt.CheckState.Checked


def test_partial_parent_at_iteration_level(
        tmp_path, monkeypatch, qapp):
    # Checking one of several iterations leaves the project partial.
    # Checking an iteration also cascades its Notes child to Checked.
    tab, _ = _build_tab(tmp_path, monkeypatch,
                        {"P": ["iter_001", "iter_002"]})
    pit = _project_item(tab, "P")
    it1 = _iteration_item(tab, "P", "iter_001")
    it1.setCheckState(0, Qt.CheckState.Checked)
    assert it1.checkState(0) == Qt.CheckState.Checked
    assert it1.child(0).checkState(0) == Qt.CheckState.Checked
    assert pit.checkState(0) == Qt.CheckState.PartiallyChecked


# ── Select all / Unselect all ──────────────────────────────────────

def test_select_all_then_unselect_all(tmp_path, monkeypatch, qapp):
    tab, _ = _build_tab(tmp_path, monkeypatch,
                        {"P": ["iter_001"], "Q": ["iter_001"]})
    tab._set_all_checks(True)
    for item in tab._iter_all_items():
        assert item.checkState(0) == Qt.CheckState.Checked
    tab._set_all_checks(False)
    for item in tab._iter_all_items():
        assert item.checkState(0) == Qt.CheckState.Unchecked


# ── (d) Remove selected deletes exactly the checked subtrees ───────

def test_remove_selected_deletes_checked_subtrees(
        tmp_path, monkeypatch, qapp):
    tab, base = _build_tab(tmp_path, monkeypatch, {
        "P": ["iter_001", "iter_002"],
        "Q": ["iter_001"],
    })
    # Tick whole project P + just iteration Q/iter_001.
    _project_item(tab, "P").setCheckState(0, Qt.CheckState.Checked)
    _iteration_item(tab, "Q", "iter_001").setCheckState(
        0, Qt.CheckState.Checked)

    tab._remove_selected()

    # P gone entirely.
    assert not os.path.exists(os.path.join(base, "P"))
    assert "P" not in tab._results_data
    # Q's only iteration gone (so Q dropped from memory), but Q's
    # project note file is untouched (we only ticked the iteration).
    assert not os.path.exists(os.path.join(base, "Q", "iter_001"))
    assert "Q" not in tab._results_data
    assert os.path.isfile(os.path.join(base, "Q", "_project_notes.txt"))


def test_remove_selected_dedupes_project_over_iteration(
        tmp_path, monkeypatch, qapp):
    # A checked project AND its (cascade-checked) iterations: the project
    # delete must absorb the iterations without erroring on a second pass.
    tab, base = _build_tab(tmp_path, monkeypatch,
                           {"P": ["iter_001", "iter_002"]})
    _project_item(tab, "P").setCheckState(0, Qt.CheckState.Checked)
    assert _iteration_item(tab, "P", "iter_001").checkState(0) \
        == Qt.CheckState.Checked

    tab._remove_selected()  # must not raise

    assert not os.path.exists(os.path.join(base, "P"))
    assert "P" not in tab._results_data


def test_remove_selected_leaves_unchecked_alone(
        tmp_path, monkeypatch, qapp):
    tab, base = _build_tab(tmp_path, monkeypatch, {
        "P": ["iter_001"],
        "Q": ["iter_001", "iter_002"],
    })
    _iteration_item(tab, "Q", "iter_001").setCheckState(
        0, Qt.CheckState.Checked)

    tab._remove_selected()

    # Only Q/iter_001 deleted; everything else intact.
    assert not os.path.exists(os.path.join(base, "Q", "iter_001"))
    assert os.path.isdir(os.path.join(base, "Q", "iter_002"))
    assert os.path.isdir(os.path.join(base, "P", "iter_001"))
    assert "P" in tab._results_data
    assert "iter_002" in tab._results_data["Q"]
    assert "iter_001" not in tab._results_data["Q"]


def test_remove_selected_no_selection_is_noop(
        tmp_path, monkeypatch, qapp):
    tab, base = _build_tab(tmp_path, monkeypatch, {"P": ["iter_001"]})
    tab._remove_selected()  # nothing ticked -> info dialog, no delete
    assert os.path.isdir(os.path.join(base, "P", "iter_001"))
    assert "P" in tab._results_data
