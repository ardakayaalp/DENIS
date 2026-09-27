"""The Results tab's CSV viewer: shaded by run, sortable, filterable.

A parameters.csv holds one row per parameter per run -- 32 rows a run
on the Ge sets, nearly a thousand for a campaign -- in one long
undifferentiated list. Reading it meant counting rows to find where
one run ended, and there was no way to ask the two questions it is
opened for: "just the centroids" and "which run has the worst
error". So:

* **Shading by run.** Every run gets its own subtle tint, keyed to
  the run itself rather than to its row position, so the tint still
  identifies the run after a sort. Adjacent runs never share a hue.
* **Sorting.** Click a header. Numbers sort as numbers (-265 before
  10, which a text sort gets wrong), text sorts case-insensitively,
  blanks and NaN go last. Right-click a header to reset the order.
* **Filtering.** A quick filter on one column -- comma-separated
  terms, OR'ed: a term that is exactly one of the column's values
  matches that value only ("Al" does not match "False"), anything
  else matches as a substring ("Amp" gives every Amp*). And a
  checklist of the column's values from the header's right-click
  menu, like a spreadsheet's filter.

The text on screen is the file's own formatting (floats to 10
significant figures, as before), so nothing a user copies differs
from the file.

Depends on: PySide6, pandas (only for the DataFrame passed in).
"""
from __future__ import annotations

import math

from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPushButton,
    QTableView, QVBoxLayout, QWidget, QWidgetAction,
)

#: Columns that identify a run, in order of preference.
RUN_COLUMNS = ("run_number", "run", "run_id", "Run", "run_num",
               "Source", "source")

#: Above this many distinct values a column's checklist would be a
#: wall of numbers; the quick text filter is the better tool there.
CHECKLIST_LIMIT = 500

#: Tint strength over the table's base colour.
TINT_ALPHA = 0.20
_N_HUES = 8
#: Hue order that keeps neighbours far apart on the colour wheel, so
#: two adjacent runs never read as the same colour.
_HUE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def cell_text(val):
    """The on-screen text for a cell -- the file's own formatting."""
    if isinstance(val, float):
        return f"{val:.10g}"
    return str(val)


def sort_key(val):
    """Numbers as numbers, then text, then blanks / NaN."""
    if isinstance(val, bool):
        return (1, str(val).lower())
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return (2, 0.0)
        return (0, float(val))
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return (2, 0.0)
    try:
        return (0, float(s))
    except ValueError:
        return (1, s.lower())


def run_column(columns):
    """Index of the column that identifies a run, or None."""
    names = list(columns)
    for want in RUN_COLUMNS:
        if want in names:
            return names.index(want)
    return None


def tints(base, n=_N_HUES, alpha=TINT_ALPHA):
    """*n* subtle tints of *base*, ordered so neighbours differ most."""
    out = []
    for i in _HUE_ORDER[:n] if n <= len(_HUE_ORDER) else range(n):
        hue = QColor.fromHsv(int((i * 360 / n + 200) % 360), 150, 230)
        out.append(QColor(
            round(base.red() * (1 - alpha) + hue.red() * alpha),
            round(base.green() * (1 - alpha) + hue.green() * alpha),
            round(base.blue() * (1 - alpha) + hue.blue() * alpha)))
    return out


class DataFrameModel(QAbstractTableModel):
    """A read-only table over a DataFrame, with a tint per run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cols: list[str] = []
        self._raw: list[list] = []          # original values
        self._text: list[list[str]] = []    # display strings
        self._run_col: int | None = None
        self._row_tint: list[int] = []      # palette index per row
        self._palette: list[QColor] = []
        self._marked: set[int] = set()      # columns with a filter on
        self.shade = True

    # ── data ──
    def set_frame(self, df):
        self.beginResetModel()
        self._cols = [str(c) for c in df.columns]
        self._raw = df.values.tolist()
        self._text = [[cell_text(v) for v in row] for row in self._raw]
        self._run_col = run_column(self._cols)
        self._marked = set()
        # Tints assigned in order of first appearance, so the file's
        # own run order walks through the palette.
        self._row_tint = []
        if self._run_col is not None:
            seen: dict[str, int] = {}
            for row in self._text:
                key = row[self._run_col]
                if key not in seen:
                    seen[key] = len(seen)
                self._row_tint.append(seen[key])
        self.endResetModel()

    def set_palette(self, colors):
        self._palette = list(colors)
        self._emit_all()

    def set_marked(self, cols):
        """Mark the headers of the columns that are filtering."""
        self._marked = set(cols)
        if self._cols:
            self.headerDataChanged.emit(
                Qt.Orientation.Horizontal, 0, len(self._cols) - 1)

    def set_shade(self, on):
        self.shade = bool(on)
        self._emit_all()

    def _emit_all(self):
        if self._text:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._text) - 1, len(self._cols) - 1),
                [Qt.ItemDataRole.BackgroundRole])

    @property
    def run_column(self):
        return self._run_col

    def column_names(self):
        return list(self._cols)

    def text(self, row, col):
        return self._text[row][col]

    def sort_value(self, row, col):
        return sort_key(self._raw[row][col])

    def column_texts(self, col):
        return [r[col] for r in self._text]

    # ── Qt model API ──
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._text)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._cols)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if role in (Qt.ItemDataRole.DisplayRole,
                    Qt.ItemDataRole.ToolTipRole):
            return self._text[r][c]
        if (role == Qt.ItemDataRole.BackgroundRole and self.shade
                and self._palette and self._row_tint):
            return self._palette[self._row_tint[r] % len(self._palette)]
        return None

    def headerData(self, section, orientation,
                   role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if section >= len(self._cols):
                return None
            mark = " ▼" if section in self._marked else ""
            return self._cols[section] + mark
        return str(section + 1)


class CsvFilterProxy(QSortFilterProxyModel):
    """Per-column value checklists, one quick text filter, and a sort
    that knows numbers from text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._allowed: dict[int, set[str]] = {}    # col -> kept values
        self._quick_col: int | None = None
        self._quick_terms: list[str] = []
        self._quick_exact: set[str] = set()

    # ── filters ──
    def _begin(self):
        """Qt 6.10 deprecated invalidateFilter() for a begin/end pair
        around the change; older Qt only has the former."""
        if hasattr(self, "beginFilterChange"):
            self.beginFilterChange()

    def _end(self):
        if hasattr(self, "endFilterChange"):
            rows = getattr(getattr(QSortFilterProxyModel, "Direction",
                                   None), "Rows", None)
            if rows is not None:
                self.endFilterChange(rows)
            else:
                self.endFilterChange()
        else:
            self.invalidateFilter()

    def set_allowed(self, col, values):
        """Keep only rows whose *col* text is in *values*; None clears."""
        self._begin()
        if values is None:
            self._allowed.pop(col, None)
        else:
            self._allowed[col] = set(values)
        self._end()

    def allowed(self, col):
        return self._allowed.get(col)

    def set_quick(self, col, text):
        """Comma-separated terms on one column, OR'ed.

        A term equal (case-insensitively) to one of the column's values
        matches that value exactly, so "Al" is Al and not every cell
        containing "al" ("False", "Model_1_bkg"...). Any other term is
        a substring match, so "Amp" finds every Amp*.
        """
        self._begin()
        self._quick_col = col
        self._quick_terms = [t.strip().lower()
                             for t in (text or "").split(",")
                             if t.strip()]
        src = self.sourceModel()
        present = ({v.strip().lower() for v in src.column_texts(col)}
                   if (src is not None and col is not None
                       and 0 <= col < src.columnCount()) else set())
        self._quick_exact = {t for t in self._quick_terms if t in present}
        self._end()

    def clear_filters(self):
        self._begin()
        self._allowed.clear()
        self._quick_terms = []
        self._quick_exact = set()
        self._end()

    def active(self):
        return bool(self._allowed) or bool(self._quick_terms)

    def filterAcceptsRow(self, row, parent):
        src = self.sourceModel()
        for col, keep in self._allowed.items():
            if src.text(row, col) not in keep:
                return False
        if self._quick_terms and self._quick_col is not None:
            cell = src.text(row, self._quick_col).strip().lower()
            ok = False
            for t in self._quick_terms:
                if (cell == t) if t in self._quick_exact else (t in cell):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def lessThan(self, left, right):
        src = self.sourceModel()
        return (src.sort_value(left.row(), left.column())
                < src.sort_value(right.row(), right.column()))


class CsvTableView(QWidget):
    """Filter bar + table + row count. ``load(df)`` shows a frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = DataFrameModel(self)
        self._proxy = CsvFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setDynamicSortFilter(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(3)

        bar = QHBoxLayout()
        # The viewer page has no margins of its own; without these the
        # label sits flush against the pane edge.
        bar.setContentsMargins(4, 3, 4, 0)
        bar.addWidget(QLabel("Filter"))
        self._col_combo = QComboBox()
        self._col_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._col_combo.setToolTip("Column the quick filter looks at.")
        bar.addWidget(self._col_combo)
        self._quick = QLineEdit()
        self._quick.setClearButtonEnabled(True)
        self._quick.setPlaceholderText("e.g.  centroid, Al   (comma = or)")
        self._quick.setToolTip(
            "Comma-separated terms; a row is kept if the column matches "
            "any of them. A term that is exactly one of the column's "
            "values matches only that value (Al, not False); anything "
            "else matches as part of the text (Amp finds every Amp*).\n\n"
            "Right-click a column header for a checklist of its values "
            "and for sorting. Click a header to sort by it.")
        bar.addWidget(self._quick, 1)
        self._shade_cb = QCheckBox("Shade by run")
        self._shade_cb.setChecked(True)
        self._shade_cb.setToolTip(
            "Give every run its own tint. The tint belongs to the run, "
            "not the row, so it still identifies the run after sorting.")
        bar.addWidget(self._shade_cb)
        self._clear_btn = QPushButton("Clear filters")
        self._clear_btn.setToolTip("Remove every filter and the sort.")
        bar.addWidget(self._clear_btn)
        self._count = QLabel("")
        bar.addWidget(self._count)
        root.addLayout(bar)

        self._view = QTableView()
        self._view.setModel(self._proxy)
        self._view.setSortingEnabled(True)
        self._view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems)
        self._view.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self._view.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSortIndicatorShown(True)
        hh.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hh.customContextMenuRequested.connect(self._header_menu)
        hh.setToolTip("Click to sort. Right-click for filters.")
        root.addWidget(self._view, 1)

        self._col_combo.currentIndexChanged.connect(self._apply_quick)
        self._quick.textChanged.connect(self._apply_quick)
        self._shade_cb.toggled.connect(self._set_shade)
        self._clear_btn.clicked.connect(self.clear_filters)
        self._proxy.rowsInserted.connect(self._update_count)
        self._proxy.rowsRemoved.connect(self._update_count)
        self._proxy.modelReset.connect(self._update_count)
        self._proxy.layoutChanged.connect(self._update_count)

    # ── public ──
    @property
    def view(self):
        return self._view

    @property
    def proxy(self):
        return self._proxy

    @property
    def model(self):
        return self._model

    def load(self, df):
        """Show *df*. Filters are cleared -- they belong to one file --
        but the quick filter's COLUMN is kept when the new file has
        it, so stepping through iterations' parameters.csv keeps
        showing the same thing."""
        keep_col = self._col_combo.currentText()
        keep_text = self._quick.text()
        self._view.setSortingEnabled(False)
        self._proxy.clear_filters()
        self._model.set_frame(df)
        self._model.set_palette(tints(self._view.palette().base().color()))
        self._model.set_shade(self._shade_cb.isChecked()
                              and self._model.run_column is not None)
        self._view.setAlternatingRowColors(not self._model.shade)
        self._shade_cb.setEnabled(self._model.run_column is not None)

        cols = self._model.column_names()
        self._col_combo.blockSignals(True)
        self._col_combo.clear()
        self._col_combo.addItems(cols)
        if keep_col in cols:
            self._col_combo.setCurrentText(keep_col)
        elif "Parameter" in cols:
            self._col_combo.setCurrentText("Parameter")
        self._col_combo.blockSignals(False)
        # The previous file's filter text carries over only when its
        # column does: "centroid" on Parameter means the same thing in
        # the next iteration's parameters.csv.
        if keep_col not in cols:
            keep_text = ""
        self._quick.blockSignals(True)
        self._quick.setText(keep_text)
        self._quick.blockSignals(False)
        self._apply_quick()

        self._proxy.sort(-1)                 # the file's own order
        self._view.horizontalHeader().setSortIndicator(
            -1, Qt.SortOrder.AscendingOrder)
        self._view.setSortingEnabled(True)
        self._update_count()

    def clear_filters(self):
        self._quick.blockSignals(True)
        self._quick.clear()
        self._quick.blockSignals(False)
        self._proxy.clear_filters()
        self._view.setSortingEnabled(False)
        self._proxy.sort(-1)
        self._view.horizontalHeader().setSortIndicator(
            -1, Qt.SortOrder.AscendingOrder)
        self._view.setSortingEnabled(True)
        self._update_header_marks()
        self._update_count()

    # ── internals ──
    def _apply_quick(self, *_):
        col = self._col_combo.currentIndex()
        self._proxy.set_quick(col if col >= 0 else None,
                              self._quick.text())
        self._update_count()

    def _set_shade(self, on):
        self._model.set_shade(on and self._model.run_column is not None)
        self._view.setAlternatingRowColors(not self._model.shade)

    def _update_count(self, *_):
        total = self._model.rowCount()
        shown = self._proxy.rowCount()
        if total == 0:
            self._count.setText("")
        elif shown == total:
            self._count.setText(f"{total} rows")
        else:
            self._count.setText(f"{shown} of {total} rows")
        self._clear_btn.setEnabled(
            self._proxy.active() or self._proxy.sortColumn() >= 0)

    def _update_header_marks(self):
        """A mark on every header whose checklist is filtering."""
        self._model.set_marked(
            c for c in range(self._model.columnCount())
            if self._proxy.allowed(c) is not None)

    def _header_menu(self, pos):
        col = self._view.horizontalHeader().logicalIndexAt(pos)
        if col < 0:
            return
        name = self._model.column_names()[col]
        menu = QMenu(self)
        menu.addAction("Sort ascending",
                       lambda: self._view.sortByColumn(
                           col, Qt.SortOrder.AscendingOrder))
        menu.addAction("Sort descending",
                       lambda: self._view.sortByColumn(
                           col, Qt.SortOrder.DescendingOrder))
        menu.addAction("File order (no sort)", self._reset_sort)
        menu.addSeparator()
        menu.addAction(f"Quick-filter on '{name}'",
                       lambda: (self._col_combo.setCurrentIndex(col),
                                self._quick.setFocus()))
        values = sorted(set(self._model.column_texts(col)), key=sort_key)
        if len(values) <= CHECKLIST_LIMIT:
            menu.addSeparator()
            menu.addAction(self._checklist_action(menu, col, values))
        else:
            act = menu.addAction(
                f"({len(values)} distinct values -- use the quick "
                f"filter)")
            act.setEnabled(False)
        if self._proxy.allowed(col) is not None:
            menu.addAction(f"Clear filter on '{name}'",
                           lambda: (self._proxy.set_allowed(col, None),
                                    self._update_header_marks(),
                                    self._update_count()))
        menu.exec(self._view.horizontalHeader().mapToGlobal(pos))

    def _reset_sort(self):
        self._view.setSortingEnabled(False)
        self._proxy.sort(-1)
        self._view.horizontalHeader().setSortIndicator(
            -1, Qt.SortOrder.AscendingOrder)
        self._view.setSortingEnabled(True)
        self._update_count()

    def _checklist_action(self, menu, col, values):
        """A searchable checklist of *col*'s values, inside the menu."""
        w = QWidget(menu)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 4, 6, 4)
        search = QLineEdit()
        search.setPlaceholderText("Search values")
        search.setClearButtonEnabled(True)
        lay.addWidget(search)
        lst = QListWidget()
        lst.setMinimumHeight(min(260, 22 * (len(values) + 1)))
        current = self._proxy.allowed(col)
        for v in values:
            it = QListWidgetItem(v if v.strip() else "(blank)")
            it.setData(Qt.ItemDataRole.UserRole, v)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(
                Qt.CheckState.Checked
                if current is None or v in current
                else Qt.CheckState.Unchecked)
            lst.addItem(it)
        lay.addWidget(lst)
        row = QHBoxLayout()
        b_all, b_none, b_ok = (QPushButton("All"), QPushButton("None"),
                               QPushButton("Apply"))
        for b in (b_all, b_none, b_ok):
            row.addWidget(b)
        lay.addLayout(row)

        def visible_items():
            return [lst.item(i) for i in range(lst.count())
                    if not lst.item(i).isHidden()]

        def set_all(state):
            for it in visible_items():
                it.setCheckState(state)

        def on_search(text):
            t = text.strip().lower()
            for i in range(lst.count()):
                it = lst.item(i)
                it.setHidden(bool(t) and t not in it.text().lower())

        def apply():
            keep = {lst.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(lst.count())
                    if lst.item(i).checkState() == Qt.CheckState.Checked}
            # Everything ticked is no filter at all.
            self._proxy.set_allowed(
                col, None if len(keep) == len(values) else keep)
            self._update_header_marks()
            self._update_count()
            menu.close()

        search.textChanged.connect(on_search)
        b_all.clicked.connect(lambda: set_all(Qt.CheckState.Checked))
        b_none.clicked.connect(lambda: set_all(Qt.CheckState.Unchecked))
        b_ok.clicked.connect(apply)
        act = QWidgetAction(menu)
        act.setDefaultWidget(w)
        return act
