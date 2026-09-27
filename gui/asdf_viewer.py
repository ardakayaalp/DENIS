"""Tools ▸ ASDF Viewer: the raw contents of run files, for inspection.

Drop .asdf (or .vasdf split) files on the window, or right-click a file
in the Pre-Analysis list or an Analysis Source block and choose "View
ASDF…". Shown exactly as stored, nothing converted:

* **Header** -- every top-level entry that is not an array (Run, Date,
  CoolerVoltage, LaserSetpoint, MassAMU, ScanningRanges, raw_header,
  ...), with a note where DENIS reads a value differently from how it
  is stored (CoolerVoltage is kept divided by 10 000).
* **Tables** -- every array: a 2-D array as its own table, columns
  named from its ``<name>_header`` list (``raw`` -> the event table:
  timestamp, voltage, bunch_number, channel, time, cooler), and the
  1-D arrays of equal length side by side (CalSet next to
  CalReadback). Below it, per-column min / max / mean and the distinct
  values when there are few (which PMT channels fired, say).
* **Tree** -- the whole ASDF tree, including the asdf library and
  history entries, arrays summarised by shape and type.

Hover a column header or a header key for what it is and its unit
(cls_estimations.asdf_contents.COLUMN_DOCS / FIELD_DOCS); hover a
timestamp or cooler cell for its converted reading.

Files load in the background, so the window never freezes: a few files
one after another (process start-up would cost more than it saves),
larger drops on up to 8 worker processes (Settings ▸ Max cores caps it
lower). Measured on 412 runs (2026-09-27): 8.1 s one by one, 1.65 s on
8 processes; threads barely help, the ASDF header parsing being pure
Python. The workers import only numpy and asdf
(cls_estimations.asdf_contents), not the GUI.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import (
    QAbstractTableModel, QCoreApplication, QElapsedTimer, QModelIndex, Qt,
    QThread, Signal,
)
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton,
    QSplitter, QStackedWidget, QTableView, QTableWidget, QTableWidgetItem,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from cls_estimations.asdf_contents import (  # noqa: F401 (re-exported)
    COLUMN_DOCS, FIELD_DOCS, MAX_LISTED_VALUES, cell_note, column_stats,
    number_text, read_asdf_file, type_text, value_text,
)

FILE_FILTER = "ASDF run files (*.asdf *.vasdf);;All files (*)"

#: At or above either, a drop loads on worker processes; below, one
#: after another on a background thread.
PROCESS_MIN_FILES = 30
PROCESS_MIN_BYTES = 300e6
#: More workers only add start-up time: 412 runs took 1.65 s on 8
#: processes but 3.95 s on 22, the first result arriving after 2.65 s
#: (measured 2026-09-27; one by one: 8.1 s).
MAX_WORKERS = 8

#: Where DENIS reads a stored value differently -- the Header "Note".
_NOTES = {
    "CoolerVoltage": lambda v: (
        f"stored / 10 000: DENIS reads {cell_note('CoolerVoltage', v)}"),
    "LaserSetpoint": lambda v: "cm⁻¹",
    "MassAMU": lambda v: "u (the Source block's isotope mass is used "
                         "for the Doppler shift)",
}


# ── reading ────────────────────────────────────────────────────────────

def resolve(path):
    """``(path, the .asdf actually read, split descriptor or None)``."""
    from gui.analysis.vasdf import is_vasdf_path, read_vasdf
    path = os.path.normpath(str(path))
    if is_vasdf_path(path):
        split = read_vasdf(path)
        return path, os.path.normpath(split["parent_path"]), split
    return path, path, None


def read_asdf(path):
    """Everything the viewer shows for one file, read in this thread.

    Returns read_asdf_file()'s dict plus ``path`` (as given, a .vasdf
    included) and ``split`` (its descriptor, or None).
    """
    path, source, split = resolve(path)
    info = read_asdf_file(source)
    info.update(path=path, split=split)
    return info


def _max_workers():
    try:
        from gui.shared_widgets import _load_settings
        wanted = int(_load_settings().get("max_cores") or 0)
    except Exception:                                     # noqa: BLE001
        wanted = 0
    cpus = os.cpu_count() or 1
    return max(1, min(wanted or max(1, cpus - 2), cpus, MAX_WORKERS))


class AsdfLoader(QThread):
    """Reads files off the GUI thread; one ``loaded`` signal per file.

    ``jobs`` is [(path, source, split)]. Few or small files are read one
    after another here; a big drop goes to a process pool, results
    arriving as each file finishes.
    """

    loaded = Signal(str, object, str)       # path, info or None, error

    def __init__(self, jobs, parent=None):
        super().__init__(parent)
        self.jobs = list(jobs)
        self._stop = False
        total = sum(os.path.getsize(s) for _p, s, _sp in self.jobs
                    if os.path.isfile(s))
        self.use_processes = (len(self.jobs) >= PROCESS_MIN_FILES
                              or total >= PROCESS_MIN_BYTES)

    def stop(self):
        self._stop = True

    def _emit(self, path, split, fn):
        try:
            info = fn()
        except Exception as exc:                          # noqa: BLE001
            self.loaded.emit(path, None, f"{type(exc).__name__}: {exc}")
            return
        info.update(path=path, split=split)
        self.loaded.emit(path, info, "")

    def run(self):
        if not self.use_processes:
            for path, source, split in self.jobs:
                if self._stop:
                    return
                self._emit(path, split, lambda s=source: read_asdf_file(s))
            return
        from concurrent.futures import ProcessPoolExecutor, as_completed
        workers = min(_max_workers(), len(self.jobs))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(read_asdf_file, source): (path, split)
                       for path, source, split in self.jobs}
            for fut in as_completed(futures):
                if self._stop:
                    pool.shutdown(wait=False, cancel_futures=True)
                    return
                path, split = futures[fut]
                if fut.exception() is not None:
                    # Retried once here before calling the file
                    # unreadable: a worker can fail for reasons that are
                    # not the file's (one did, once, and not again).
                    source = next(s for p, s, _sp in self.jobs if p == path)
                    self._emit(path, split,
                               lambda s=source: read_asdf_file(s))
                else:
                    self._emit(path, split, fut.result)


# ── table model ────────────────────────────────────────────────────────

class ArrayTableModel(QAbstractTableModel):
    """A 2-D array as a table, read on demand (100 000+ rows are fine).

    Column headers carry their documentation as a tooltip, and a
    timestamp or cooler cell its converted reading.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = np.empty((0, 0))
        self._columns = []

    def set_table(self, data, columns):
        self.beginResetModel()
        self._data = data
        self._columns = list(columns)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else int(self._data.shape[0])

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return number_text(self._data[index.row(), index.column()])
        if role == Qt.ItemDataRole.ToolTipRole:
            return cell_note(self._columns[index.column()],
                             self._data[index.row(), index.column()]) or None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation,
                   role=Qt.ItemDataRole.DisplayRole):
        horizontal = orientation == Qt.Orientation.Horizontal
        if horizontal and section >= len(self._columns):
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._columns[section] if horizontal else str(section)
        if role == Qt.ItemDataRole.ToolTipRole and horizontal:
            name = self._columns[section]
            return COLUMN_DOCS.get(name) or FIELD_DOCS.get(name)
        return None


def _copy_selection(view):
    """Selected cells as tab-separated text (pastes into a spreadsheet)."""
    idx = view.selectionModel().selectedIndexes()
    if not idx:
        return ""
    rows = sorted({i.row() for i in idx})
    cols = sorted({i.column() for i in idx})
    model = view.model()
    lines = ["\t".join(str(model.headerData(c, Qt.Orientation.Horizontal))
                       for c in cols)]
    for r in rows:
        lines.append("\t".join(
            model.data(model.index(r, c)) or "" for c in cols))
    text = "\n".join(lines)
    QGuiApplication.clipboard().setText(text)
    return text


def _doc(name):
    return COLUMN_DOCS.get(name) or FIELD_DOCS.get(name) or ""


# ── window ─────────────────────────────────────────────────────────────

class AsdfViewer(QDialog):
    """Non-modal inspector for raw ASDF run files."""

    files_changed = Signal()
    loading_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("asdf_viewer")
        self.setWindowTitle("ASDF Viewer")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowMinMaxButtonsHint)
        self.setAcceptDrops(True)
        self._files = {}                 # path -> info
        self._errors = {}                # path -> message
        self._pending = set()            # paths still loading
        self._loaders = []
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop_loading)

        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self._browse)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_current)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        hint = QLabel("Drop .asdf or .vasdf files here")
        hint.setStyleSheet("color: gray;")
        self.status_label = QLabel()
        for w in (self.open_button, self.remove_button, self.clear_button):
            bar.addWidget(w)
        bar.addSpacing(12)
        bar.addWidget(hint)
        bar.addStretch()
        bar.addWidget(self.status_label)
        lay.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._show_current)
        split.addWidget(self.file_list)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        rlay.addWidget(self.path_label)
        self.stack = QStackedWidget()
        self._empty_text = (
            "No file.\n\nDrop .asdf files here, use Open…, or "
            "right-click a run in Pre-Analysis or in an Analysis Source "
            "block and choose “View ASDF…”.")
        self.empty_label = QLabel(self._empty_text)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.stack.addWidget(self.empty_label)
        self.tabs = QTabWidget()
        self.stack.addWidget(self.tabs)
        rlay.addWidget(self.stack, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 4)
        fm0 = self.fontMetrics()
        split.setSizes([fm0.horizontalAdvance(
            "Run 00000 — run_00000.asdf") + 30,
            fm0.horizontalAdvance("M") * 100])
        lay.addWidget(split, 1)

        # Header
        self.header_table = QTableWidget(0, 4)
        self.header_table.setHorizontalHeaderLabels(
            ["Key", "Value", "Type", "Note"])
        self.header_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.header_table.verticalHeader().setVisible(False)
        _hh = self.header_table.horizontalHeader()
        # Key and Type fit their text; Value and Note share the rest, so
        # a long value is elided (full text in its tooltip) instead of
        # pushing the other columns off screen.
        for col, mode in ((0, QHeaderView.ResizeMode.ResizeToContents),
                          (1, QHeaderView.ResizeMode.Stretch),
                          (2, QHeaderView.ResizeMode.ResizeToContents),
                          (3, QHeaderView.ResizeMode.Stretch)):
            _hh.setSectionResizeMode(col, mode)
        self.tabs.addTab(self.header_table, "Header")

        # Tables
        tables_page = QWidget()
        tl = QVBoxLayout(tables_page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Table:"))
        self.table_combo = QComboBox()
        self.table_combo.currentIndexChanged.connect(self._show_table)
        row.addWidget(self.table_combo, 1)
        self.table_info = QLabel()
        row.addWidget(self.table_info)
        tl.addLayout(row)
        tsplit = QSplitter(Qt.Orientation.Vertical)
        self.model = ArrayTableModel(self)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        # Size columns from the header and the first rows only: scanning
        # every one of 100 000+ events would stall on each table switch.
        self.table_view.horizontalHeader().setResizeContentsPrecision(200)
        self.table_view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self._table_menu)
        tsplit.addWidget(self.table_view)
        self.stats_table = QTableWidget(0, 5)
        self.stats_table.setHorizontalHeaderLabels(
            ["Column", "Min", "Max", "Mean", "Distinct values"])
        self.stats_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        tsplit.addWidget(self.stats_table)
        tsplit.setStretchFactor(0, 3)
        tl.addWidget(tsplit, 1)
        self.tabs.addTab(tables_page, "Tables")

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Key", "Value", "Type"])
        for col, mode in ((0, QHeaderView.ResizeMode.ResizeToContents),
                          (1, QHeaderView.ResizeMode.Stretch),
                          (2, QHeaderView.ResizeMode.ResizeToContents)):
            self.tree.header().setSectionResizeMode(col, mode)
        self.tree.header().setStretchLastSection(False)
        self.tabs.addTab(self.tree, "Tree")

        fm = self.fontMetrics()
        screen = (parent.screen() if parent is not None
                  else self.screen()).availableGeometry()
        self.resize(min(fm.horizontalAdvance("M") * 140,
                        int(screen.width() * 0.8)),
                    min(fm.height() * 46, int(screen.height() * 0.8)))
        self._refresh_buttons()

    # ── files ───────────────────────────────────────────────────────
    def paths(self):
        return [self.file_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.file_list.count())]

    def add_files(self, paths, select=True):
        """List the files at once and load them in the background.

        A file already listed is just selected. Returns the paths that
        were queued for loading; watch ``loading_finished`` or call
        wait_until_loaded() for the results.
        """
        jobs = []
        last = None
        for raw_path in paths:
            try:
                path, source, split = resolve(raw_path)
            except Exception as exc:                      # noqa: BLE001
                path = os.path.normpath(str(raw_path))
                if path not in self._index():
                    self._errors[path] = f"{type(exc).__name__}: {exc}"
                    self._add_item(path)
                last = path
                continue
            last = path
            if path in self._index():
                continue
            self._pending.add(path)
            self._add_item(path)
            jobs.append((path, source, split))
        if jobs:
            loader = AsdfLoader(jobs, self)
            loader.loaded.connect(self._on_loaded)
            loader.finished.connect(lambda ld=loader: self._loader_done(ld))
            self._loaders.append(loader)
            loader.start()
        if select and last is not None:
            self.select(last)
        self._refresh_buttons()
        self._refresh_status()
        self.files_changed.emit()
        return [j[0] for j in jobs]

    def _index(self):
        return set(self.paths())

    def _add_item(self, path):
        item = QListWidgetItem(self._item_text(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(self._item_tip(path))
        self.file_list.addItem(item)

    def _item_text(self, path):
        base = os.path.basename(path)
        if path in self._pending:
            return f"⏳ {base}"
        if path in self._errors:
            return f"⚠ {base}"
        run = dict(self._files[path]["header"]).get("Run")
        return f"Run {run} — {base}" if run is not None else base

    def _item_tip(self, path):
        if path in self._errors:
            return f"{path}\n\n{self._errors[path]}"
        return path

    def _on_loaded(self, path, info, error):
        if path not in self._pending:          # removed or cleared meanwhile
            return
        self._pending.discard(path)
        if info is None:
            self._errors[path] = error
        else:
            self._files[path] = info
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText(self._item_text(path))
                item.setToolTip(self._item_tip(path))
                if i == self.file_list.currentRow():
                    self._show_current(i)
                break
        self._refresh_status()

    def _loader_done(self, loader):
        if loader in self._loaders:
            self._loaders.remove(loader)
        loader.deleteLater()
        if not self._loaders:
            self._refresh_status()
            self.loading_finished.emit()

    def is_loading(self):
        return bool(self._pending) or bool(self._loaders)

    def wait_until_loaded(self, timeout_ms=60000):
        """Pump events until every queued file has arrived (tests,
        scripts). Returns False on timeout."""
        timer = QElapsedTimer()
        timer.start()
        while self.is_loading():
            QCoreApplication.processEvents()
            for ld in list(self._loaders):
                ld.wait(20)
            if timer.elapsed() > timeout_ms:
                return False
        QCoreApplication.processEvents()
        return True

    def _refresh_status(self):
        n = len(self._pending)
        if not n:
            self.status_label.clear()
            return
        procs = any(ld.use_processes for ld in self._loaders)
        self.status_label.setText(
            f"Loading {n} file{'s' if n != 1 else ''}…"
            + (f"  ({_max_workers()} processes)" if procs else ""))

    def failed(self):
        return dict(self._errors)

    def select(self, path):
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(Qt.ItemDataRole.UserRole) == path:
                self.file_list.setCurrentRow(i)
                return

    def remove_current(self):
        row = self.file_list.currentRow()
        if row < 0:
            return
        path = self.file_list.item(row).data(Qt.ItemDataRole.UserRole)
        self._files.pop(path, None)
        self._errors.pop(path, None)
        self._pending.discard(path)
        self.file_list.takeItem(row)
        if self.file_list.count() == 0:
            self._show_current(-1)
        self._refresh_buttons()
        self._refresh_status()
        self.files_changed.emit()

    def clear(self):
        for ld in self._loaders:
            ld.stop()
        self._files.clear()
        self._errors.clear()
        self._pending.clear()
        self.file_list.clear()
        self._show_current(-1)
        self._refresh_buttons()
        self._refresh_status()
        self.files_changed.emit()

    def stop_loading(self, wait_ms=5000):
        """Stop every loader and wait for its thread to end. Closing the
        window only hides it (it is reused) and loading carries on; the
        app quitting must not destroy a running thread, which crashes."""
        for ld in list(self._loaders):
            ld.stop()
        for ld in list(self._loaders):
            ld.wait(wait_ms)

    def _refresh_buttons(self):
        has = self.file_list.count() > 0
        self.remove_button.setEnabled(has)
        self.clear_button.setEnabled(has)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open ASDF files", "", FILE_FILTER)
        if paths:
            self.add_files(paths)

    # ── drag and drop ───────────────────────────────────────────────
    @staticmethod
    def _dropped_paths(mime):
        if not mime.hasUrls():
            return []
        return [u.toLocalFile() for u in mime.urls()
                if u.isLocalFile() and u.toLocalFile().lower().endswith(
                    (".asdf", ".vasdf"))]

    def dragEnterEvent(self, event):
        if self._dropped_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, event):
        paths = self._dropped_paths(event.mimeData())
        if paths:
            self.add_files(paths)
            event.acceptProposedAction()

    # ── showing ─────────────────────────────────────────────────────
    def current_info(self):
        row = self.file_list.currentRow()
        if row < 0:
            return None
        return self._files.get(
            self.file_list.item(row).data(Qt.ItemDataRole.UserRole))

    def _show_current(self, row):
        if row < 0:
            self.path_label.clear()
            self.empty_label.setText(self._empty_text)
            self.stack.setCurrentWidget(self.empty_label)
            return
        path = self.file_list.item(row).data(Qt.ItemDataRole.UserRole)
        if path in self._pending:
            self.path_label.setText(path)
            self.empty_label.setText("Loading…")
            self.stack.setCurrentWidget(self.empty_label)
            return
        if path in self._errors:
            self.path_label.setText(path)
            self.empty_label.setText(
                f"Could not read this file:\n\n{self._errors[path]}")
            self.stack.setCurrentWidget(self.empty_label)
            return
        info = self._files[path]
        text = f"{path}  ({info['size_bytes'] / 1e6:.2f} MB)"
        if info["split"]:
            sp = info["split"].get("split") or {}
            text = (f"{path}\nVirtual split of {info['parent_path']}, "
                    f"{sp.get('axis', 'raw_voltage')} "
                    f"[{sp.get('lo')}, {sp.get('hi')}] — shown: the "
                    f"parent file, unsplit")
        self.path_label.setText(text)
        self.stack.setCurrentWidget(self.tabs)
        self._fill_header(info)
        self._fill_tables(info)
        self._fill_tree(info)

    def _fill_header(self, info):
        rows = info["header"]
        t = self.header_table
        t.setRowCount(len(rows))
        for r, (key, value) in enumerate(rows):
            note = _NOTES.get(key)
            if key.endswith("_header") and key[:-7] in info["tree"]:
                note_text = f"column names of '{key[:-7]}'"
            else:
                try:
                    note_text = note(value) if note else ""
                except Exception:                           # noqa: BLE001
                    note_text = ""
            doc = FIELD_DOCS.get(key, "")
            if key.endswith("_header"):
                doc = (f"Column names of the '{key[:-7]}' table; hover a "
                       f"column header in the Tables tab for what each is.")
            for c, txt in enumerate((key, value_text(value),
                                     type_text(value), note_text)):
                item = QTableWidgetItem(txt)
                # The key explains the entry; every other cell shows its
                # own full text, which may be elided.
                item.setToolTip(doc if (c == 0 and doc) else txt)
                t.setItem(r, c, item)

    def _fill_tables(self, info):
        self.table_combo.blockSignals(True)
        self.table_combo.clear()
        for tab in info["tables"]:
            n, m = tab["data"].shape
            self.table_combo.addItem(f"{tab['name']}  ({n:,} × {m})")
        self.table_combo.blockSignals(False)
        self._show_table(0 if info["tables"] else -1)

    def _show_table(self, index):
        info = self.current_info()
        if info is None or index < 0 or index >= len(info["tables"]):
            self.model.set_table(np.empty((0, 0)), [])
            self.stats_table.setRowCount(0)
            self.table_info.clear()
            return
        tab = info["tables"][index]
        self.model.set_table(tab["data"], tab["columns"])
        self.table_view.resizeColumnsToContents()
        n, m = tab["data"].shape
        self.table_info.setText(
            f"{n:,} rows × {m} columns"
            + ("  — one row per event" if tab["name"] == "raw" else ""))
        stats = column_stats(tab["data"])
        self.stats_table.setRowCount(len(stats))
        for r, (name, (lo, hi, mean, distinct)) in enumerate(
                zip(tab["columns"], stats)):
            dtext = (", ".join(number_text(v) for v in distinct)
                     if isinstance(distinct, list) else f"{distinct:,}")
            for c, txt in enumerate((name, number_text(lo), number_text(hi),
                                     number_text(mean), dtext)):
                item = QTableWidgetItem(txt)
                if c == 0 and _doc(name):
                    item.setToolTip(_doc(name))
                elif c in (1, 2, 3) and cell_note(name, txt):
                    item.setToolTip(cell_note(name, txt))
                self.stats_table.setItem(r, c, item)
        self.stats_table.resizeColumnsToContents()

    def _fill_tree(self, info):
        self.tree.clear()

        def add(parent, key, value):
            if isinstance(value, dict):
                item = QTreeWidgetItem(parent,
                                       [key, "", f"dict[{len(value)}]"])
                for k, v in value.items():
                    add(item, str(k), v)
            elif isinstance(value, list) and value and all(
                    isinstance(v, (dict, list)) for v in value):
                item = QTreeWidgetItem(parent, [key, "", type_text(value)])
                for i, v in enumerate(value):
                    add(item, f"[{i}]", v)
            else:
                item = QTreeWidgetItem(parent, [key, value_text(value),
                                                type_text(value)])
            if parent is self.tree.invisibleRootItem() and _doc(key):
                item.setToolTip(0, _doc(key))
        for k, v in info["tree"].items():
            add(self.tree.invisibleRootItem(), k, v)
        self.tree.expandToDepth(0)

    def _table_menu(self, pos):
        menu = QMenu(self.table_view)
        copy_act = menu.addAction("Copy selection")
        export_act = menu.addAction("Export table as CSV…")
        chosen = menu.exec(self.table_view.viewport().mapToGlobal(pos))
        if chosen is copy_act:
            _copy_selection(self.table_view)
        elif chosen is export_act:
            self.export_current_table()

    def export_current_table(self, path=None):
        info = self.current_info()
        idx = self.table_combo.currentIndex()
        if info is None or idx < 0:
            return None
        tab = info["tables"][idx]
        if path is None:
            base = os.path.splitext(os.path.basename(info["path"]))[0]
            path, _ = QFileDialog.getSaveFileName(
                self, "Export table", f"{base}_{tab['name'].split()[0]}.csv",
                "CSV (*.csv)")
            if not path:
                return None
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(tab["columns"]) + "\n")
            for row in tab["data"]:
                fh.write(",".join(number_text(v) for v in row) + "\n")
        return path


def open_in_viewer(paths, anchor=None):
    """Open the viewer with ``paths`` added and selected.

    Uses the main window's single viewer when ``anchor`` sits in one (so
    files opened from different tabs collect in one window), else a
    standalone one.
    """
    top = anchor.window() if anchor is not None else None
    if top is not None and hasattr(top, "open_asdf_viewer"):
        return top.open_asdf_viewer(paths)
    viewer = AsdfViewer(top)
    viewer.add_files(paths)
    viewer.show()
    viewer.raise_()
    return viewer
