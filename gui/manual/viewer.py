"""The DENIS documentation window (Help > Documentation, F1).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Defines ManualWindow, a modeless QMainWindow with a left contents tree and a
right rich-text browser. Navigation history (back/forward) is managed explicitly
so links, tree clicks, and the Home button share one consistent stack. Equation
and diagram images are loaded from the on-disk cache via a loadResource override,
and embedded images are rescaled to fit the pane width.

Depends on: gui.manual.structure, gui.manual.content, gui.shared_widgets
(optional lucide icons); built on PySide6.
"""

import os

from PySide6.QtCore import Qt, QUrl, QSize
from PySide6.QtGui import (
    QImage, QAction, QKeySequence, QDesktopServices, QIcon,
    QTextCursor, QTextImageFormat,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTextBrowser, QToolBar, QLineEdit, QLabel, QVBoxLayout,
)

from gui.manual import structure
from gui.manual import content as content_mod

try:
    from gui.shared_widgets import lucide_icon
except Exception:  # pragma: no cover - icons are cosmetic
    def lucide_icon(_name):
        return QIcon()


# Document-level stylesheet (Qt rich-text CSS subset) applied to every page.
_DOC_CSS = """
body { color: #d4d4d4; font-family: 'Segoe UI', sans-serif; }
h1 { color: #ffffff; font-size: 20px; }
h2 { color: #7fb5e0; font-size: 15px; }
h3 { color: #9ac4e6; font-size: 13px; }
p, li, td { font-size: 13px; line-height: 140%; }
a { color: #4ea1d5; text-decoration: none; }
code { color: #d7a36a; font-family: Consolas, 'DejaVu Sans Mono', monospace; }
ul { margin-left: 6px; }
"""

_BROWSER_QSS = "QTextBrowser { background: #232629; border: none; padding: 6px 14px; }"
_TREE_QSS = (
    "QTreeWidget { background: #1f2123; border: none; outline: 0; }"
    "QTreeWidget::item { padding: 3px 2px; }"
    "QTreeWidget::item:selected { background: #2f6da3; color: #ffffff; }"
    "QTreeWidget::item:hover { background: #2a2d30; }"
)

HOME_PAGE = "introduction"


class _ContentBrowser(QTextBrowser):
    """QTextBrowser that loads cached PNGs, never auto-follows links, and keeps
    embedded images scaled to the available width so wide figures/diagrams are
    never clipped (Qt rich text has no responsive-image CSS)."""

    # Left+right document padding (_BROWSER_QSS) plus a small safety margin.
    _WIDTH_MARGIN = 36

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.document().setDefaultStyleSheet(_DOC_CSS)
        self.setStyleSheet(_BROWSER_QSS)
        self._native_cache = {}

    def loadResource(self, type_, url):
        path = url.toLocalFile()
        if path and os.path.exists(path):
            img = QImage(path)
            if not img.isNull():
                return img
        return super().loadResource(type_, url)

    def _native_size(self, name):
        """Pixel (width, height) of an image src, cached."""
        if name not in self._native_cache:
            path = QUrl(name).toLocalFile() or name
            img = QImage(path)
            self._native_cache[name] = (
                (img.width(), img.height()) if not img.isNull() else (0, 0))
        return self._native_cache[name]

    def rescale_images(self):
        """Fit every embedded image to the pane width, preserving aspect ratio
        and never upscaling beyond native resolution."""
        avail = self.viewport().width() - self._WIDTH_MARGIN
        if avail <= 0:
            return
        doc = self.document()
        # Collect first (mutating formats while iterating invalidates fragments).
        targets = []
        block = doc.begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid() and frag.charFormat().isImageFormat():
                    fmt = frag.charFormat().toImageFormat()
                    targets.append((frag.position(), frag.length(), fmt))
                it += 1
            block = block.next()
        if not targets:
            return
        for pos, length, fmt in targets:
            nw, nh = self._native_size(fmt.name())
            if nw <= 0 or nh <= 0:
                continue
            if nw > avail:
                scale = avail / nw
                fmt.setWidth(avail)
                fmt.setHeight(round(nh * scale))
            else:
                fmt.setWidth(nw)
                fmt.setHeight(nh)
            cur = QTextCursor(doc)
            cur.setPosition(pos)
            cur.setPosition(pos + length, QTextCursor.MoveMode.KeepAnchor)
            cur.setCharFormat(fmt)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale_images()


class ManualWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DENIS — Documentation")
        self.setWindowIcon(lucide_icon("circle-help"))
        self.resize(1080, 720)

        self._history = []
        self._idx = -1
        self._syncing = False
        self._id_to_item = {}

        self._build_toolbar()

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet(_TREE_QSS)
        self.tree.setMinimumWidth(240)
        self._build_tree()
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        splitter.addWidget(self.tree)

        self.browser = _ContentBrowser()
        self.browser.anchorClicked.connect(self._on_anchor)
        splitter.addWidget(self.browser)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 800])
        self.setCentralWidget(splitter)

        self.navigate(HOME_PAGE)

    # ── construction ──────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_back = QAction(lucide_icon("chevron-left"), "Back", self)
        self.act_back.setShortcut(QKeySequence.StandardKey.Back)
        self.act_back.triggered.connect(self.go_back)
        tb.addAction(self.act_back)

        self.act_fwd = QAction(lucide_icon("chevron-right"), "Forward", self)
        self.act_fwd.setShortcut(QKeySequence.StandardKey.Forward)
        self.act_fwd.triggered.connect(self.go_forward)
        tb.addAction(self.act_fwd)

        self.act_home = QAction(lucide_icon("atom"), "Contents", self)
        self.act_home.triggered.connect(lambda: self.navigate(HOME_PAGE))
        tb.addAction(self.act_home)

        tb.addSeparator()
        tb.addWidget(QLabel(" Search: "))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter contents…  (Enter: find in page)")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(320)
        self.search.textChanged.connect(self._filter_tree)
        self.search.returnPressed.connect(self._find_in_page)
        tb.addWidget(self.search)

    def _build_tree(self):
        for sec in structure.TOC:
            top = QTreeWidgetItem([sec["title"]])
            top.setData(0, Qt.ItemDataRole.UserRole, sec["id"])
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self._id_to_item[sec["id"]] = top
            for child in sec.get("children", []):
                it = QTreeWidgetItem([child["title"]])
                it.setData(0, Qt.ItemDataRole.UserRole, child["id"])
                top.addChild(it)
                self._id_to_item[child["id"]] = it
            self.tree.addTopLevelItem(top)

    # ── navigation ────────────────────────────────────────────────

    def navigate(self, page_id, record=True):
        body = content_mod.get_page_html(page_id)
        self.browser.setHtml(body)
        self.browser.rescale_images()
        self.browser.verticalScrollBar().setValue(0)
        if record:
            del self._history[self._idx + 1:]
            self._history.append(page_id)
            self._idx = len(self._history) - 1
        self._sync_tree(page_id)
        self._update_nav_actions()

    def go_back(self):
        if self._idx > 0:
            self._idx -= 1
            self.navigate(self._history[self._idx], record=False)

    def go_forward(self):
        if self._idx < len(self._history) - 1:
            self._idx += 1
            self.navigate(self._history[self._idx], record=False)

    def _update_nav_actions(self):
        self.act_back.setEnabled(self._idx > 0)
        self.act_fwd.setEnabled(self._idx < len(self._history) - 1)

    def _on_anchor(self, url):
        s = url.toString()
        if s.startswith("page:"):
            self.navigate(s[len("page:"):])
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
        elif s:
            self.navigate(s)

    # ── tree <-> browser sync ─────────────────────────────────────

    def _on_tree_selection(self):
        if self._syncing:
            return
        items = self.tree.selectedItems()
        if not items:
            return
        page_id = items[0].data(0, Qt.ItemDataRole.UserRole)
        if page_id:
            self.navigate(page_id)

    def _sync_tree(self, page_id):
        item = self._id_to_item.get(page_id)
        if item is None:
            return
        self._syncing = True
        try:
            if item.parent():
                item.parent().setExpanded(True)
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)
        finally:
            self._syncing = False

    # ── search ────────────────────────────────────────────────────

    def _filter_tree(self, text):
        text = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            child_match = False
            for j in range(top.childCount()):
                c = top.child(j)
                m = (text in c.text(0).lower()) if text else True
                c.setHidden(not m)
                child_match = child_match or m
            top_match = (not text) or (text in top.text(0).lower()) or child_match
            top.setHidden(not top_match)
            if text and top_match:
                top.setExpanded(True)

    def _find_in_page(self):
        text = self.search.text().strip()
        if text and not self.browser.find(text):
            # wrap to top and retry once
            cursor = self.browser.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.browser.setTextCursor(cursor)
            self.browser.find(text)
