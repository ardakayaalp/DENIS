"""Results tab: browse, view, edit and export analysis outputs.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Provides the DENIS Results tab, a project/iteration tree on the left and a
multi-page viewer on the right (text, table, live matplotlib plot, and a
notes editor). Plots are re-rendered live from saved .npz data so they stay
editable, with per-figure style overrides persisted to .style.json. Also
handles bulk selection, deletion of projects/iterations, per-scope notes,
and export of individual items or whole iteration directories.

Depends on: gui.shared_widgets (plot editor, plot-type settings, analysis
directory and last-dir helpers, icons), gui.analysis.helpers (scrollable
info dialog), cls_estimations.plotting (color cycle); built on PySide6 and
matplotlib (QtAgg backend).
"""

import os
import json
import numpy as np

# pandas is imported where used (one read_csv call): at module level it
# cost ~0.7 s of every app startup.

from gui import session_log

_log = session_log.get_logger("results")

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import to_hex
from matplotlib.container import ErrorbarContainer

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
    QTreeWidgetItem, QLabel, QPushButton, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QStackedWidget, QSizePolicy, QDialog,
    QDialogButtonBox, QScrollArea, QMenu, QCheckBox,
    QComboBox, QSpinBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon, QBrush, QColor

from gui.shared_widgets import (
    PlotEditorDialog, get_plot_type_settings,
    fit_plot_settings_for_counts, lucide_icon,
)


# ── Figure style extraction / application ──────────────────────────

def _style_path(npz_path):
    """Return the .style.json path for a given .npz path."""
    return npz_path.rsplit('.', 1)[0] + '.style.json'


def fit_run_sort_key(fname):
    """Sort ``fit_run_<n>.<ext>`` figures by numeric run number so the
    Results tree lists them in the same order as the fit report (e.g.
    fit_run_2 before fit_run_10), instead of lexically.

    Non-matching filenames keep a stable alphabetical order after the
    numeric ones.
    """
    label = fname
    if fname.startswith("fit_run_"):
        label = fname[len("fit_run_"):].rsplit(".", 1)[0]
    digits = ""
    for ch in str(label):
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return (0, int(digits), str(fname)) if digits else (1, 0, str(fname))


def _extract_figure_style(fig):
    """Capture all editable style properties from a matplotlib figure."""
    from matplotlib.collections import (
        PathCollection, PolyCollection, LineCollection)
    style = {
        "figsize": [fig.get_figwidth(), fig.get_figheight()],
        "suptitle": fig._suptitle.get_text() if fig._suptitle else None,
        "suptitle_size": (fig._suptitle.get_fontsize()
                          if fig._suptitle else None),
        "axes": [],
    }
    # Capture subplot adjust params
    sp = fig.subplotpars
    style["subplotpars"] = {
        "left": sp.left, "right": sp.right,
        "top": sp.top, "bottom": sp.bottom,
    }

    for ax in fig.axes:
        ax_style = {
            "title": ax.get_title(),
            "title_size": float(ax.title.get_fontsize()),
            "title_pos": list(ax.title.get_position()),
            "xlabel": ax.get_xlabel(),
            "xlabel_size": float(ax.xaxis.label.get_fontsize()),
            "xlabel_pos": list(ax.xaxis.label.get_position()),
            "ylabel": ax.get_ylabel(),
            "ylabel_size": float(ax.yaxis.label.get_fontsize()),
            "ylabel_pos": list(ax.yaxis.label.get_position()),
            "xlim": list(ax.get_xlim()),
            "ylim": list(ax.get_ylim()),
            "grid": bool(ax.xaxis.get_gridlines()
                         and ax.xaxis.get_gridlines()[0].get_visible()),
            "artists": [],
        }
        # Tick label size
        tl = ax.xaxis.get_ticklabels()
        ax_style["tick_size"] = float(tl[0].get_fontsize()) if tl else 10.0

        # Collect errorbar child IDs to skip
        eb_ids = set()
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                dl, caps, bars = container
                if dl is not None:
                    eb_ids.add(id(dl))
                for c in (caps or []):
                    eb_ids.add(id(c))
                for b in (bars or []):
                    eb_ids.add(id(b))
                # Save errorbar style from data line
                a = {"kind": "errorbar", "label": container.get_label()}
                if dl is not None:
                    a.update({
                        "color": to_hex(dl.get_color()),
                        "linewidth": float(dl.get_linewidth()),
                        "linestyle": dl.get_linestyle(),
                        "marker": str(dl.get_marker()),
                        "markersize": float(dl.get_markersize()),
                        "alpha": dl.get_alpha() if dl.get_alpha()
                        is not None else 1.0,
                        "visible": dl.get_visible(),
                    })
                ax_style["artists"].append(a)

        # Lines (skip errorbar children)
        for line in ax.get_lines():
            if id(line) in eb_ids:
                continue
            ax_style["artists"].append({
                "kind": "line",
                "label": line.get_label(),
                "color": to_hex(line.get_color()),
                "linewidth": float(line.get_linewidth()),
                "linestyle": line.get_linestyle(),
                "marker": str(line.get_marker()),
                "markersize": float(line.get_markersize()),
                "alpha": (line.get_alpha() if line.get_alpha()
                          is not None else 1.0),
                "visible": line.get_visible(),
            })

        # Collections (skip errorbar children)
        for coll in ax.collections:
            if id(coll) in eb_ids:
                continue
            if isinstance(coll, PathCollection):
                fc = coll.get_facecolor()
                a = {
                    "kind": "scatter",
                    "label": coll.get_label(),
                    "color": to_hex(fc[0]) if len(fc) > 0 else "#000000",
                    "alpha": (float(coll.get_alpha())
                              if coll.get_alpha() is not None else 1.0),
                    "visible": coll.get_visible(),
                }
                sizes = coll.get_sizes()
                a["size"] = float(sizes[0]) if len(sizes) > 0 else 36.0
                ax_style["artists"].append(a)
            elif isinstance(coll, PolyCollection):
                fc = coll.get_facecolor()
                ax_style["artists"].append({
                    "kind": "fill",
                    "label": coll.get_label(),
                    "color": to_hex(fc[0]) if len(fc) > 0 else "#000000",
                    "alpha": (float(coll.get_alpha())
                              if coll.get_alpha() is not None else 1.0),
                    "visible": coll.get_visible(),
                })
            elif isinstance(coll, LineCollection):
                cs = coll.get_colors()
                ax_style["artists"].append({
                    "kind": "linecoll",
                    "label": coll.get_label(),
                    "color": to_hex(cs[0]) if len(cs) > 0 else "#000000",
                    "alpha": (float(coll.get_alpha())
                              if coll.get_alpha() is not None else 1.0),
                    "visible": coll.get_visible(),
                })

        # Text labels and annotation arrows. Captured AFTER the
        # standard artists so re-application can match texts by index
        # within ax.texts (Annotations are also in this list).
        from matplotlib.text import Annotation
        from matplotlib.transforms import IdentityTransform
        texts = []
        for t in list(ax.texts):
            entry = {
                "is_annotation": isinstance(t, Annotation),
                "gid": t.get_gid(),
                "text": t.get_text(),
                "color": to_hex(t.get_color()),
                "alpha": (float(t.get_alpha())
                          if t.get_alpha() is not None else 1.0),
                "visible": bool(t.get_visible()),
                "fontsize": float(t.get_fontsize()),
            }
            try:
                entry["position"] = list(t.get_position())
            except Exception:
                entry["position"] = None
            # Tag the transform so re-application puts the position
            # in the right coord system (axes-fraction labels vs
            # data-coord δν annotations).
            try:
                entry["transform"] = (
                    "axes" if t.get_transform() is ax.transAxes
                    else "data")
            except Exception:
                entry["transform"] = "data"
            if isinstance(t, Annotation):
                try:
                    entry["xy"] = list(t.xy)
                except Exception:
                    entry["xy"] = None
                # Arrow patch tint
                ap = t.arrow_patch
                if ap is not None:
                    try:
                        entry["arrow_color"] = to_hex(ap.get_edgecolor())
                        entry["arrow_lw"] = float(ap.get_linewidth())
                    except Exception:
                        pass
            texts.append(entry)
        ax_style["texts"] = texts

        # δν arrows are standalone FancyArrowPatch artists (both the IS
        # tab and the Results renderer draw them that way, gid-tagged),
        # which live in ax.patches — the old text-only capture lost
        # them entirely, so drags/tints never round-tripped.
        from matplotlib.patches import FancyArrowPatch
        arrows = []
        for p in ax.patches:
            if not isinstance(p, FancyArrowPatch):
                continue
            entry = {
                "gid": p.get_gid(),
                "visible": bool(p.get_visible()),
                "alpha": (float(p.get_alpha())
                          if p.get_alpha() is not None else 1.0),
            }
            try:
                entry["color"] = to_hex(p.get_edgecolor())
            except Exception:
                pass
            try:
                entry["lw"] = float(p.get_linewidth())
            except Exception:
                pass
            pos = getattr(p, "_posA_posB", None)
            if pos is not None and len(pos) == 2:
                try:
                    entry["posA"] = [float(pos[0][0]), float(pos[0][1])]
                    entry["posB"] = [float(pos[1][0]), float(pos[1][1])]
                except Exception:
                    pass
            arrows.append(entry)
        ax_style["arrows"] = arrows

        style["axes"].append(ax_style)
    return style


def _apply_figure_style(fig, style):
    """Apply saved style to a matplotlib figure after rendering."""
    from matplotlib.collections import (
        PathCollection, PolyCollection, LineCollection)

    if "suptitle" in style and style["suptitle"] is not None:
        fig.suptitle(style["suptitle"],
                     fontsize=style.get("suptitle_size", 10))

    for ax_idx, ax in enumerate(fig.axes):
        if ax_idx >= len(style.get("axes", [])):
            break
        s = style["axes"][ax_idx]

        # Title. Every property is applied ONLY when present in the
        # style dict, so a partial style (from a selective copy/paste)
        # changes just the chosen properties instead of resetting the
        # rest to defaults.
        if "title" in s or "title_size" in s:
            ax.set_title(s.get("title", ax.get_title()),
                         fontsize=s.get("title_size",
                                        ax.title.get_fontsize()))
        if "title_pos" in s:
            ax.title.set_position(s["title_pos"])

        # X label
        if "xlabel" in s or "xlabel_size" in s:
            ax.set_xlabel(s.get("xlabel", ax.get_xlabel()),
                          fontsize=s.get("xlabel_size",
                                         ax.xaxis.label.get_fontsize()))
        if "xlabel_pos" in s:
            ax.xaxis.label.set_position(s["xlabel_pos"])

        # Y label
        if "ylabel" in s or "ylabel_size" in s:
            ax.set_ylabel(s.get("ylabel", ax.get_ylabel()),
                          fontsize=s.get("ylabel_size",
                                         ax.yaxis.label.get_fontsize()))
        if "ylabel_pos" in s:
            ax.yaxis.label.set_position(s["ylabel_pos"])

        # Limits
        if "xlim" in s:
            ax.set_xlim(s["xlim"])
        if "ylim" in s:
            ax.set_ylim(s["ylim"])

        # Tick size
        if "tick_size" in s:
            ax.tick_params(labelsize=s["tick_size"])

        # Grid
        if "grid" in s:
            ax.grid(bool(s["grid"]))

        # Apply artist styles positionally
        saved_artists = s.get("artists", [])
        # Build current artist list in same order as extraction
        current_artists = []
        eb_ids = set()
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                dl, caps, bars = container
                if dl is not None:
                    eb_ids.add(id(dl))
                for c in (caps or []):
                    eb_ids.add(id(c))
                for b in (bars or []):
                    eb_ids.add(id(b))
                current_artists.append((container, "errorbar"))
        for line in ax.get_lines():
            if id(line) in eb_ids:
                continue
            current_artists.append((line, "line"))
        for coll in ax.collections:
            if id(coll) in eb_ids:
                continue
            if isinstance(coll, PathCollection):
                current_artists.append((coll, "scatter"))
            elif isinstance(coll, PolyCollection):
                current_artists.append((coll, "fill"))
            elif isinstance(coll, LineCollection):
                current_artists.append((coll, "linecoll"))

        for i, (artist, kind) in enumerate(current_artists):
            if i >= len(saved_artists):
                break
            sa = saved_artists[i]
            if sa.get("kind") != kind:
                continue  # mismatch, skip

            if kind == "line":
                if "color" in sa:
                    artist.set_color(sa["color"])
                if "linewidth" in sa:
                    artist.set_linewidth(sa["linewidth"])
                if "linestyle" in sa:
                    artist.set_linestyle(sa["linestyle"])
                if "marker" in sa:
                    artist.set_marker(sa["marker"])
                if "markersize" in sa:
                    artist.set_markersize(sa["markersize"])
                if "alpha" in sa:
                    artist.set_alpha(sa["alpha"])
                if "visible" in sa:
                    artist.set_visible(sa["visible"])
                if "label" in sa:
                    artist.set_label(sa["label"])

            elif kind == "errorbar":
                dl = artist[0]
                if dl is not None:
                    if "color" in sa:
                        dl.set_color(sa["color"])
                        for cap in (artist[1] or []):
                            cap.set_color(sa["color"])
                        for bar in (artist[2] or []):
                            bar.set_color(sa["color"])
                    if "linewidth" in sa:
                        dl.set_linewidth(sa["linewidth"])
                    if "linestyle" in sa:
                        dl.set_linestyle(sa["linestyle"])
                    if "marker" in sa:
                        dl.set_marker(sa["marker"])
                    if "markersize" in sa:
                        dl.set_markersize(sa["markersize"])
                    if "alpha" in sa:
                        dl.set_alpha(sa["alpha"])
                        for cap in (artist[1] or []):
                            cap.set_alpha(sa["alpha"])
                        for bar in (artist[2] or []):
                            bar.set_alpha(sa["alpha"])
                    if "visible" in sa:
                        dl.set_visible(sa["visible"])
                        for cap in (artist[1] or []):
                            cap.set_visible(sa["visible"])
                        for bar in (artist[2] or []):
                            bar.set_visible(sa["visible"])
                if "label" in sa:
                    artist.set_label(sa["label"])

            elif kind == "scatter":
                if "color" in sa:
                    artist.set_facecolors(sa["color"])
                    artist.set_edgecolors(sa["color"])
                if "size" in sa:
                    artist.set_sizes([sa["size"]])
                if "alpha" in sa:
                    artist.set_alpha(sa["alpha"])
                if "visible" in sa:
                    artist.set_visible(sa["visible"])

            elif kind == "fill":
                if "color" in sa:
                    artist.set_facecolor(sa["color"])
                if "alpha" in sa:
                    artist.set_alpha(sa["alpha"])
                if "visible" in sa:
                    artist.set_visible(sa["visible"])

            elif kind == "linecoll":
                if "color" in sa:
                    artist.set_color(sa["color"])
                if "alpha" in sa:
                    artist.set_alpha(sa["alpha"])
                if "visible" in sa:
                    artist.set_visible(sa["visible"])

        # Apply text / annotation positions and styles. Prefer matching
        # by gid (set by the IS/Results renderers); index within
        # ax.texts is only the legacy fallback for old sidecars.
        from matplotlib.text import Annotation
        saved_texts = s.get("texts") or []
        current_texts = list(ax.texts)
        by_gid = {t.get_gid(): t for t in current_texts if t.get_gid()}
        for i, st in enumerate(saved_texts):
            gid = st.get("gid")
            if gid and gid in by_gid:
                t = by_gid[gid]
            elif i < len(current_texts):
                t = current_texts[i]
            else:
                continue
            if bool(st.get("is_annotation")) != isinstance(t, Annotation):
                # Old index-matched sidecars could smear a δν label onto
                # an annotation arrow (the doubled-arrow bug); a kind
                # mismatch now skips cleanly instead of applying.
                continue
            if "text" in st:
                try:
                    t.set_text(st["text"])
                except Exception:
                    pass
            if "color" in st:
                try:
                    t.set_color(st["color"])
                except Exception:
                    pass
            if "alpha" in st and st["alpha"] is not None:
                try:
                    t.set_alpha(float(st["alpha"]))
                except Exception:
                    pass
            if "visible" in st:
                try:
                    t.set_visible(bool(st["visible"]))
                except Exception:
                    pass
            if "fontsize" in st:
                try:
                    t.set_fontsize(float(st["fontsize"]))
                except Exception:
                    pass
            pos = st.get("position")
            if pos is not None and len(pos) == 2:
                try:
                    t.set_position((float(pos[0]), float(pos[1])))
                except Exception:
                    pass
            if isinstance(t, Annotation):
                xy = st.get("xy")
                if xy is not None and len(xy) == 2:
                    try:
                        t.xy = (float(xy[0]), float(xy[1]))
                    except Exception:
                        pass
                ap = t.arrow_patch
                if ap is not None:
                    if "arrow_color" in st:
                        try:
                            ap.set_color(st["arrow_color"])
                        except Exception:
                            pass
                    if "arrow_lw" in st:
                        try:
                            ap.set_linewidth(float(st["arrow_lw"]))
                        except Exception:
                            pass

        # δν arrow patches (FancyArrowPatch), matched by gid.
        from matplotlib.patches import FancyArrowPatch
        saved_arrows = s.get("arrows") or []
        cur_arrows = {p.get_gid(): p for p in ax.patches
                      if isinstance(p, FancyArrowPatch) and p.get_gid()}
        for sa in saved_arrows:
            p = cur_arrows.get(sa.get("gid"))
            if p is None:
                continue
            if sa.get("posA") and sa.get("posB"):
                try:
                    p.set_positions(tuple(sa["posA"]), tuple(sa["posB"]))
                except Exception:
                    pass
            if "color" in sa:
                try:
                    p.set_color(sa["color"])
                except Exception:
                    pass
            if "lw" in sa:
                try:
                    p.set_linewidth(float(sa["lw"]))
                except Exception:
                    pass
            if sa.get("alpha") is not None:
                try:
                    p.set_alpha(float(sa["alpha"]))
                except Exception:
                    pass
            if "visible" in sa:
                try:
                    p.set_visible(bool(sa["visible"]))
                except Exception:
                    pass

    # Artists the user deleted in the plot editor. The renderers
    # regenerate their default annotations on every render, so removal
    # must be re-applied by gid or deleted arrows would resurrect.
    removed = set(style.get("removed_gids") or [])
    if removed:
        for ax in fig.axes:
            for art in (list(ax.texts) + list(ax.patches)
                        + list(ax.lines)):
                if art.get_gid() in removed:
                    try:
                        art.remove()
                    except Exception:
                        pass


# ── Plot-style copy/paste: property categories ───────────────────────
#
# Each category maps to the top-level and/or per-axis keys of the style
# dict produced by _extract_figure_style. Copy/paste dialogs let the
# user tick which categories to carry.
_STYLE_CATEGORIES = [
    ("figure", "Figure size, margins & suptitle"),
    ("title_labels_text", "Title & axis-label text"),
    ("font_sizes", "Font sizes (title, labels, ticks)"),
    ("limits", "Axis limits (x / y)"),
    ("grid", "Grid on / off"),
    ("label_positions", "Title / axis-label positions"),
    ("artist_styles", "Line & marker styles (colour, width…)"),
    ("texts_annotations", "Text labels & annotation arrows"),
]
_STYLE_TOP_KEYS = {
    "figure": ["figsize", "subplotpars", "suptitle", "suptitle_size"],
}
_STYLE_AXIS_KEYS = {
    "title_labels_text": ["title", "xlabel", "ylabel"],
    "font_sizes": ["title_size", "xlabel_size", "ylabel_size", "tick_size"],
    "limits": ["xlim", "ylim"],
    "grid": ["grid"],
    "label_positions": ["title_pos", "xlabel_pos", "ylabel_pos"],
    "artist_styles": ["artists"],
    "texts_annotations": ["texts", "arrows"],
}


def _filter_style(style, categories):
    """Return a copy of ``style`` keeping only the chosen categories."""
    cats = set(categories)
    out = {}
    for cat in cats:
        for k in _STYLE_TOP_KEYS.get(cat, []):
            if k in style:
                out[k] = style[k]
    axes_in = style.get("axes", [])
    if axes_in:
        axes_out = []
        for ax in axes_in:
            a = {}
            for cat in cats:
                for k in _STYLE_AXIS_KEYS.get(cat, []):
                    if k in ax:
                        a[k] = ax[k]
            axes_out.append(a)
        if any(axes_out):
            out["axes"] = axes_out
    return out


def _style_categories_present(style):
    """Which categories actually carry data in ``style`` (so a paste
    dialog only offers what was copied)."""
    present = []
    axes = style.get("axes", [])
    for cat, _label in _STYLE_CATEGORIES:
        has = any(k in style for k in _STYLE_TOP_KEYS.get(cat, []))
        if not has:
            for ax in axes:
                if any(k in ax for k in _STYLE_AXIS_KEYS.get(cat, [])):
                    has = True
                    break
        if has:
            present.append(cat)
    return present


class StylePropertyDialog(QDialog):
    """Checklist of style-property categories to copy or paste."""

    def __init__(self, categories, parent=None, title="Select properties",
                 intro=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        from gui.dialog_style import style_dialog
        try:
            style_dialog(self)
        except Exception:
            pass
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        if intro:
            note = QLabel(intro)
            note.setWordWrap(True)
            lay.addWidget(note)
        self._boxes = {}
        labels = dict(_STYLE_CATEGORIES)
        for cat in categories:
            cb = QCheckBox(labels.get(cat, cat))
            cb.setChecked(True)
            self._boxes[cat] = cb
            lay.addWidget(cb)
        row = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn = QPushButton("None")
        none_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        lay.addLayout(row)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _set_all(self, on):
        for cb in self._boxes.values():
            cb.setChecked(on)

    def selected(self):
        return [c for c, cb in self._boxes.items() if cb.isChecked()]


class PlotCsvExportDialog(QDialog):
    """Choose which columns and formatting to write when exporting a
    plot's underlying data as CSV."""

    _DELIMS = [("Comma  ,", ","), ("Tab  \\t", "\t"),
               ("Semicolon  ;", ";")]

    def __init__(self, column_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export plot data as CSV")
        from gui.dialog_style import style_dialog
        try:
            style_dialog(self)
        except Exception:
            pass
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.addWidget(QLabel("<b>Columns to export</b>"))
        self._boxes = {}
        for name in column_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self._boxes[name] = cb
            lay.addWidget(cb)
        row = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn = QPushButton("None")
        none_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        lay.addLayout(row)

        opts = QHBoxLayout()
        self._header_cb = QCheckBox("Header row")
        self._header_cb.setChecked(True)
        opts.addWidget(self._header_cb)
        opts.addWidget(QLabel("Delimiter:"))
        self._delim_combo = QComboBox()
        for label, _ in self._DELIMS:
            self._delim_combo.addItem(label)
        opts.addWidget(self._delim_combo)
        opts.addWidget(QLabel("Decimals:"))
        self._prec_spin = QSpinBox()
        self._prec_spin.setRange(0, 12)
        self._prec_spin.setValue(6)
        opts.addWidget(self._prec_spin)
        opts.addStretch()
        lay.addLayout(opts)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _set_all(self, on):
        for cb in self._boxes.values():
            cb.setChecked(on)

    def options(self):
        return {
            "columns": [n for n, cb in self._boxes.items()
                        if cb.isChecked()],
            "header": self._header_cb.isChecked(),
            "delimiter": self._DELIMS[self._delim_combo.currentIndex()][1],
            "precision": self._prec_spin.value(),
        }


class ResultsTab(QWidget):
    """Results tab: tree browser + viewer for analysis outputs."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: project tree ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Analysis Results"))

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.itemClicked.connect(self._on_item_clicked)
        # Hierarchy-aware tri-state checkboxes (cascade down on a parent
        # toggle, recompute the parent up from its children). The
        # reentrancy guard stops our own setCheckState() writes from
        # re-entering _on_item_check_changed.
        self._updating_checks = False
        # Iterations that finished this session and haven't been viewed
        # yet — highlighted in the tree until the user opens them.
        self._fresh_iters = set()
        self._tree.itemChanged.connect(self._on_item_check_changed)
        self._tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(
            self._on_tree_context_menu)
        left_layout.addWidget(self._tree)

        tree_btn_row = QHBoxLayout()
        refresh_all_btn = QPushButton("Refresh All")
        refresh_all_btn.setToolTip(
            "Re-scan the output directory and surface results for "
            "EVERY project found on disk (including projects not "
            "open in the Analysis tab).")
        # clicked emits bool (the button's checked state); use a
        # lambda to swallow it so it isn't passed as project_filter.
        refresh_all_btn.clicked.connect(lambda _=False: self._refresh_from_disk())
        tree_btn_row.addWidget(refresh_all_btn)
        refresh_project_btn = QPushButton("Refresh Project")
        refresh_project_btn.setToolTip(
            "Re-scan the output directory but only load results for "
            "the project currently active in the Analysis tab.")
        refresh_project_btn.clicked.connect(
            lambda _=False: self._refresh_active_project())
        tree_btn_row.addWidget(refresh_project_btn)
        collapse_btn = QPushButton("Collapse All")
        collapse_btn.setToolTip("Collapse all projects to show only top-level names")
        collapse_btn.clicked.connect(self._tree.collapseAll)
        tree_btn_row.addWidget(collapse_btn)
        left_layout.addLayout(tree_btn_row)

        # ── Bulk-selection row (checkboxes) ──
        select_btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.setToolTip("Tick every project, iteration and item")
        select_all_btn.clicked.connect(lambda _=False: self._set_all_checks(True))
        select_btn_row.addWidget(select_all_btn)
        unselect_all_btn = QPushButton("Unselect All")
        unselect_all_btn.setToolTip("Clear every tick")
        unselect_all_btn.clicked.connect(
            lambda _=False: self._set_all_checks(False))
        select_btn_row.addWidget(unselect_all_btn)
        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.setToolTip(
            "Delete every ticked project / iteration from disk (and its "
            "notes), after a confirmation.")
        remove_selected_btn.clicked.connect(
            lambda _=False: self._remove_selected())
        select_btn_row.addWidget(remove_selected_btn)
        left_layout.addLayout(select_btn_row)

        splitter.addWidget(left)

        # ── Right panel: viewer ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._title_label = QLabel("Select an item to view")
        self._title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self._title_label)

        self._viewer_stack = QStackedWidget()

        # Page 0: placeholder
        placeholder = QLabel("No item selected")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #888;")
        self._viewer_stack.addWidget(placeholder)

        # Page 1: text viewer
        self._text_viewer = QPlainTextEdit()
        self._text_viewer.setReadOnly(True)
        self._text_viewer.setFont(QFont("Courier", 10))
        self._viewer_stack.addWidget(self._text_viewer)

        # Page 2: table viewer
        self._table_viewer = QTableWidget()
        self._table_viewer.setAlternatingRowColors(True)
        self._table_viewer.horizontalHeader().setStretchLastSection(False)
        self._table_viewer.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._viewer_stack.addWidget(self._table_viewer)

        # Page 3: plot viewer
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_figure = Figure(figsize=(8, 5), dpi=100,
                                     constrained_layout=False)
        self._plot_figure.subplots_adjust(
            left=0.06, right=0.99, top=0.96, bottom=0.10)
        self._plot_canvas = FigureCanvasQTAgg(self._plot_figure)
        # Right-click "Edit plot…" routes through the Results
        # editor (style.json persistence), not a bare dialog.
        self._plot_canvas._plot_editor_opener = self._edit_plot
        # tight_layout stores margins as FRACTIONS computed at the figure's
        # size when it ran. If the window is later resized/maximized the
        # figure grows but the fractions stay fixed -> the (absolute) label
        # band stops shrinking and leaves slack on the left/bottom. Re-fit
        # the layout on every canvas resize so the plot keeps hugging the
        # border at any window size. Guarded by _plot_tight_layout so the
        # full-bleed image page is left alone.
        self._plot_tight_layout = False
        self._plot_canvas.mpl_connect(
            "resize_event", self._on_plot_canvas_resize)
        self._plot_toolbar = NavigationToolbar2QT(self._plot_canvas, self)
        plot_layout.addWidget(self._plot_toolbar)
        plot_layout.addWidget(self._plot_canvas, 1)
        self._viewer_stack.addWidget(plot_widget)

        # Page 4: notes editor (free-text per project / iteration)
        notes_widget = QWidget()
        notes_layout = QVBoxLayout(notes_widget)
        notes_layout.setContentsMargins(0, 0, 0, 0)
        self._notes_editor = QPlainTextEdit()
        self._notes_editor.setPlaceholderText(
            "Write notes about this project or iteration here. "
            "They are saved automatically.")
        self._notes_editor.textChanged.connect(self._on_notes_text_changed)
        notes_layout.addWidget(self._notes_editor, 1)
        notes_btn_row = QHBoxLayout()
        self._notes_status = QLabel("")
        self._notes_status.setStyleSheet("color: #888;")
        notes_btn_row.addWidget(self._notes_status)
        notes_btn_row.addStretch()
        save_notes_btn = QPushButton("Save Now")
        save_notes_btn.clicked.connect(lambda: self._save_notes(force=True))
        notes_btn_row.addWidget(save_notes_btn)
        notes_layout.addLayout(notes_btn_row)
        self._viewer_stack.addWidget(notes_widget)
        self._notes_path = None
        self._notes_dirty = False
        self._notes_save_timer = QTimer(self)
        self._notes_save_timer.setSingleShot(True)
        self._notes_save_timer.setInterval(800)  # ms after last keystroke
        self._notes_save_timer.timeout.connect(self._save_notes)

        right_layout.addWidget(self._viewer_stack, 1)

        # Buttons
        btn_row = QHBoxLayout()
        self._edit_plot_btn = QPushButton("Edit Plot")
        self._edit_plot_btn.clicked.connect(self._edit_plot)
        self._edit_plot_btn.setEnabled(False)
        btn_row.addWidget(self._edit_plot_btn)

        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._export_current)
        btn_row.addWidget(self._export_btn)

        self._export_all_btn = QPushButton("Export All")
        self._export_all_btn.setToolTip(
            "Copy the entire selected iteration directory to a chosen location")
        self._export_all_btn.clicked.connect(self._export_all)
        btn_row.addWidget(self._export_all_btn)
        btn_row.addStretch()
        info_btn = QPushButton(lucide_icon("circle-help"), "Info")
        info_btn.setFixedWidth(75)
        info_btn.clicked.connect(self._show_info)
        btn_row.addWidget(info_btn)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([280, 700])
        layout.addWidget(splitter)

        # Internal data storage
        # {project_name: {iter_name: {type: ..., data: ...}}}
        self._results_data = {}
        self._current_item_data = None
        self._current_tree_item = None
        # In-memory clipboard for copy/paste of a plot's saved style
        # (fonts, sizes, label positions, colours…) between plots.
        self._style_clipboard = None

    def add_results(self, project_name, results, output_config):
        """Add results from a completed analysis."""
        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()
        project_dir = os.path.join(base_dir, project_name)

        # Determine iteration name
        if os.path.isdir(project_dir):
            existing = sorted([d for d in os.listdir(project_dir)
                               if os.path.isdir(
                                   os.path.join(project_dir, d))])
            iter_name = existing[-1] if existing else "iter_001"
        else:
            iter_name = "iter_001"

        # Store in memory
        if project_name not in self._results_data:
            self._results_data[project_name] = {}
        self._results_data[project_name][iter_name] = {
            "results": results,
            "output_config": output_config,
            "directory": os.path.join(project_dir, iter_name),
        }

        # Mark this iteration as freshly finished so _rebuild_tree
        # highlights it until the user views it once.
        self._fresh_iters.add((project_name, iter_name))

        self._rebuild_tree()
        return (project_name, iter_name)

    def _project_hue(self, project_name):
        """Stable hue (0-359) derived from the project name. Same name
        always maps to the same hue across sessions.
        """
        import hashlib
        h = int(hashlib.md5(project_name.encode("utf-8")).hexdigest()[:8],
                16)
        return h % 360

    def _project_brushes(self, project_name, iter_index=None):
        """Return (background, foreground) brushes for a project / iter.
        Subtle: low-saturation tint on the dark theme background. Iter
        rows shift saturation/lightness slightly so iterations of the
        same project read as a 'shade' of the project color without
        becoming distracting.
        """
        hue = self._project_hue(project_name)
        if iter_index is None:
            # Project header — slightly stronger tint, slightly lighter
            bg = QColor.fromHsl(hue, 60, 58)   # ~23% lightness
            fg = QColor.fromHsl(hue, 90, 200)  # mild pastel text
        else:
            # Iteration — same hue, dimmer + slightly cycled
            offset = (iter_index % 4) * 4      # 0 / 4 / 8 / 12
            bg = QColor.fromHsl(hue, 50, 42 + offset)
            fg = QColor.fromHsl(hue, 70, 175)
        return QBrush(bg), QBrush(fg)

    def _rebuild_tree(self):
        """Rebuild the tree widget from stored data."""
        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()
        # Guard so setting checkable flags / states below doesn't fire
        # _on_item_check_changed for every node during the rebuild.
        self._updating_checks = True
        self._tree.clear()
        for project_name in sorted(self._results_data.keys()):
            project_item = QTreeWidgetItem(self._tree, [project_name])
            project_item.setExpanded(True)
            project_dir = os.path.join(base_dir, project_name)
            project_item.setData(0, Qt.ItemDataRole.UserRole, {
                "project": project_name, "dir": project_dir,
            })
            # Subtle project-color highlight (header)
            p_bg, p_fg = self._project_brushes(project_name)
            project_item.setBackground(0, p_bg)
            project_item.setForeground(0, p_fg)
            f = project_item.font(0)
            f.setBold(True)
            project_item.setFont(0, f)

            # Project-level notes
            proj_notes_path = os.path.join(
                project_dir, "_project_notes.txt")
            proj_notes_item = QTreeWidgetItem(project_item, ["📝 Notes"])
            proj_notes_item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "notes", "path": proj_notes_path,
                "scope": "project", "label": project_name,
            })

            for iter_idx, iter_name in enumerate(sorted(
                    self._results_data[project_name].keys())):
                iter_data = self._results_data[project_name][iter_name]
                is_fresh = (project_name, iter_name) in self._fresh_iters
                label = f"● {iter_name}  (NEW)" if is_fresh else iter_name
                iter_item = QTreeWidgetItem(project_item, [label])
                iter_dir = iter_data.get("directory", "")
                iter_item.setData(0, Qt.ItemDataRole.UserRole, {
                    "project": project_name, "iteration": iter_name,
                    "dir": iter_dir, "iter_index": iter_idx,
                })
                # Subtle iteration shading — same hue as the project
                i_bg, i_fg = self._project_brushes(
                    project_name, iter_index=iter_idx)
                iter_item.setBackground(0, i_bg)
                iter_item.setForeground(0, i_fg)
                if is_fresh:
                    # Fresh-iteration highlight: bold + a bright amber
                    # tint that stands out against the project shading,
                    # cleared once the user views it (see _clear_fresh).
                    fnt = iter_item.font(0)
                    fnt.setBold(True)
                    iter_item.setFont(0, fnt)
                    iter_item.setBackground(0, QBrush(QColor(120, 85, 10)))
                    iter_item.setForeground(0, QBrush(QColor(255, 214, 110)))
                    iter_item.setExpanded(True)

                # Iteration-level notes (always offered, even if dir missing)
                iter_notes_path = os.path.join(iter_dir, "_iter_notes.txt")
                iter_notes_item = QTreeWidgetItem(iter_item, ["📝 Notes"])
                iter_notes_item.setData(0, Qt.ItemDataRole.UserRole, {
                    "type": "notes", "path": iter_notes_path,
                    "scope": "iteration",
                    "label": f"{project_name} / {iter_name}",
                })

                # Add file entries
                if os.path.isdir(iter_dir):
                    # Fit report
                    report_path = os.path.join(iter_dir, "fit_report.txt")
                    if os.path.isfile(report_path):
                        child = QTreeWidgetItem(iter_item, ["Fit Report"])
                        child.setData(0, Qt.ItemDataRole.UserRole, {
                            "type": "text", "path": report_path})

                    # Parameters CSV
                    params_path = os.path.join(iter_dir, "parameters.csv")
                    if os.path.isfile(params_path):
                        child = QTreeWidgetItem(iter_item, ["Parameters"])
                        child.setData(0, Qt.ItemDataRole.UserRole, {
                            "type": "csv", "path": params_path})

                    # Metadata CSV
                    meta_path = os.path.join(iter_dir, "metadata.csv")
                    if os.path.isfile(meta_path):
                        child = QTreeWidgetItem(iter_item, ["Metadata"])
                        child.setData(0, Qt.ItemDataRole.UserRole, {
                            "type": "csv", "path": meta_path})

                    # Binning Summary CSV
                    binsum_path = os.path.join(iter_dir,
                                               "binning_summary.csv")
                    if os.path.isfile(binsum_path):
                        child = QTreeWidgetItem(iter_item,
                                                ["Binning Summary"])
                        child.setData(0, Qt.ItemDataRole.UserRole, {
                            "type": "csv", "path": binsum_path})

                    # Binning Warnings log (plain text; .log is the
                    # new format, .json is the legacy back-compat).
                    # Skip the tree entry when the warnings list was
                    # empty (zero-byte .log file or empty .json).
                    binwarn_log = os.path.join(iter_dir,
                                                "binning_warnings.log")
                    binwarn_json = os.path.join(iter_dir,
                                                 "binning_warnings.json")
                    if os.path.isfile(binwarn_log):
                        try:
                            _has_warnings = (
                                os.path.getsize(binwarn_log) > 0)
                        except OSError:
                            _has_warnings = True
                        binwarn_path = binwarn_log
                    elif os.path.isfile(binwarn_json):
                        try:
                            import json
                            with open(binwarn_json,
                                       encoding="utf-8") as _wf:
                                _has_warnings = bool(json.load(_wf))
                        except Exception:
                            _has_warnings = True
                        binwarn_path = binwarn_json
                    else:
                        _has_warnings = False
                        binwarn_path = None
                    if _has_warnings and binwarn_path:
                        child = QTreeWidgetItem(iter_item,
                                                ["Binning Warnings"])
                        child.setData(0, Qt.ItemDataRole.UserRole, {
                            "type": "text", "path": binwarn_path})

                    # Plots (prefer .npz for editable live plots)
                    plots_dir = os.path.join(iter_dir, "plots")
                    if os.path.isdir(plots_dir):
                        seen_bases = set()
                        for fname in sorted(os.listdir(plots_dir),
                                            key=fit_run_sort_key):
                            if fname.lower().endswith('.npz'):
                                base = fname[:-4]
                                seen_bases.add(base)
                                display = base + ".png"
                                child = QTreeWidgetItem(
                                    iter_item, [display])
                                child.setData(
                                    0, Qt.ItemDataRole.UserRole, {
                                        "type": "editable_plot",
                                        "path": os.path.join(
                                            plots_dir, fname)})
                        for fname in sorted(os.listdir(plots_dir),
                                            key=fit_run_sort_key):
                            base = os.path.splitext(fname)[0]
                            if (fname.lower().endswith(
                                    ('.png', '.pdf', '.svg'))
                                    and base not in seen_bases):
                                child = QTreeWidgetItem(
                                    iter_item, [fname])
                                child.setData(
                                    0, Qt.ItemDataRole.UserRole, {
                                        "type": "image",
                                        "path": os.path.join(
                                            plots_dir, fname)})

                    # Tracking (prefer .npz for editable live plots)
                    track_dir = os.path.join(iter_dir, "tracking")
                    if os.path.isdir(track_dir):
                        seen_bases = set()
                        for fname in sorted(os.listdir(track_dir)):
                            fpath = os.path.join(track_dir, fname)
                            if fname.lower().endswith('.npz'):
                                base = fname[:-4]
                                seen_bases.add(base)
                                display = base + ".png"
                                child = QTreeWidgetItem(
                                    iter_item, [display])
                                child.setData(
                                    0, Qt.ItemDataRole.UserRole, {
                                        "type": "editable_plot",
                                        "path": fpath})
                        for fname in sorted(os.listdir(track_dir)):
                            fpath = os.path.join(track_dir, fname)
                            base = os.path.splitext(fname)[0]
                            if (fname.lower().endswith(
                                    ('.png', '.pdf', '.svg'))
                                    and base not in seen_bases):
                                child = QTreeWidgetItem(
                                    iter_item, [fname])
                                child.setData(
                                    0, Qt.ItemDataRole.UserRole, {
                                        "type": "image", "path": fpath})
                            elif fname.lower().endswith('.csv'):
                                child = QTreeWidgetItem(
                                    iter_item, [fname])
                                child.setData(
                                    0, Qt.ItemDataRole.UserRole, {
                                        "type": "csv", "path": fpath})

                    # Diagnostics (prefer .npz data for live plots)
                    diag_dir = os.path.join(iter_dir, "diagnostics")
                    if os.path.isdir(diag_dir):
                        seen = set()
                        for fname in sorted(os.listdir(diag_dir)):
                            fpath = os.path.join(diag_dir, fname)
                            if fname.lower().endswith('.npz'):
                                base = fname[:-4]  # strip .npz
                                seen.add(base)
                                # Show with image extension in tree
                                display = base + ".png"
                                child = QTreeWidgetItem(
                                    iter_item, [display])
                                child.setData(
                                    0, Qt.ItemDataRole.UserRole, {
                                        "type": "diagnostic_plot",
                                        "path": fpath})
                        # Fallback: show PNGs without .npz
                        for fname in sorted(os.listdir(diag_dir)):
                            fpath = os.path.join(diag_dir, fname)
                            base = os.path.splitext(fname)[0]
                            if (fname.lower().endswith(
                                    ('.png', '.pdf', '.svg'))
                                    and base not in seen):
                                child = QTreeWidgetItem(
                                    iter_item, [fname])
                                child.setData(
                                    0, Qt.ItemDataRole.UserRole, {
                                        "type": "image", "path": fpath})

                iter_item.setExpanded(True)

        # Make every node checkable (tri-state for parents) and start
        # unchecked. Done after the tree is fully built so children
        # exist when we tag parents as auto-tristate.
        self._make_items_checkable()
        self._updating_checks = False

    def _iter_all_items(self):
        """Yield every QTreeWidgetItem in the tree (depth-first)."""
        stack = [self._tree.topLevelItem(i)
                 for i in range(self._tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            yield item
            for c in range(item.childCount()):
                stack.append(item.child(c))

    def _make_items_checkable(self):
        """Tag every tree item with a user checkbox in column 0.

        We do NOT use Qt's ItemIsAutoTristate: it would override our
        cascade (e.g. forcing a freshly-checked iteration back to
        partial because its own helper children are still unchecked).
        Parent partial/checked state is computed manually in
        _recompute_parent_check instead. Caller must hold the
        _updating_checks guard so the initial Unchecked writes don't
        re-enter the itemChanged handler.
        """
        for item in self._iter_all_items():
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)

    def _on_item_check_changed(self, item, column):
        """Cascade checkbox state down to children and recompute the
        parent state up. Guarded against our own writes re-entering.
        """
        if column != 0 or self._updating_checks:
            return
        self._updating_checks = True
        self._tree.blockSignals(True)
        try:
            state = item.checkState(0)
            if state != Qt.CheckState.PartiallyChecked:
                # User clicked a checkbox (not a Qt-computed partial):
                # push that state onto all descendants.
                self._set_subtree_check(item, state)
            # Recompute every ancestor from its children.
            parent = item.parent()
            while parent is not None:
                self._recompute_parent_check(parent)
                parent = parent.parent()
        finally:
            self._tree.blockSignals(False)
            self._updating_checks = False

    def _set_subtree_check(self, item, state):
        """Set `item` and all its descendants to `state`."""
        item.setCheckState(0, state)
        for c in range(item.childCount()):
            self._set_subtree_check(item.child(c), state)

    def _recompute_parent_check(self, parent):
        """Set `parent` to Checked / Unchecked / PartiallyChecked based
        on its immediate children's check states.
        """
        n = parent.childCount()
        if n == 0:
            return
        checked = 0
        partial = False
        for c in range(n):
            cs = parent.child(c).checkState(0)
            if cs == Qt.CheckState.Checked:
                checked += 1
            elif cs == Qt.CheckState.PartiallyChecked:
                partial = True
        if partial or 0 < checked < n:
            parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
        elif checked == n:
            parent.setCheckState(0, Qt.CheckState.Checked)
        else:
            parent.setCheckState(0, Qt.CheckState.Unchecked)

    def _set_all_checks(self, checked):
        """Select-all / Unselect-all helper for every node."""
        state = (Qt.CheckState.Checked if checked
                 else Qt.CheckState.Unchecked)
        self._updating_checks = True
        self._tree.blockSignals(True)
        try:
            for item in self._iter_all_items():
                item.setCheckState(0, state)
        finally:
            self._tree.blockSignals(False)
            self._updating_checks = False

    def _remove_selected(self):
        """Delete every checked project / iteration (deduped to the
        top-most checked ancestor) via the existing single-delete path,
        after a confirmation dialog.
        """
        # Collect the top-most checked structural nodes (projects and
        # iterations). A checked project subsumes its checked
        # iterations, so we don't descend into a checked node.
        projects = []      # (project, project_dir)
        iterations = []    # (project, iteration, iter_dir)

        def _walk(item):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            is_project = (isinstance(data, dict) and "project" in data
                          and "iteration" not in data
                          and "type" not in data)
            is_iteration = (isinstance(data, dict) and "project" in data
                            and "iteration" in data
                            and "type" not in data)
            if item.checkState(0) == Qt.CheckState.Checked and (
                    is_project or is_iteration):
                if is_project:
                    projects.append(
                        (data["project"], data.get("dir", "")))
                else:
                    iterations.append((data["project"],
                                       data["iteration"],
                                       data.get("dir", "")))
                return  # don't descend: parent delete covers children
            for c in range(item.childCount()):
                _walk(item.child(c))

        for i in range(self._tree.topLevelItemCount()):
            _walk(self._tree.topLevelItem(i))

        if not projects and not iterations:
            QMessageBox.information(
                self, "Remove Selected",
                "Nothing is ticked. Use the checkboxes to mark "
                "projects or iterations to delete.")
            return

        parts = []
        if projects:
            parts.append(f"{len(projects)} project(s)")
        if iterations:
            parts.append(f"{len(iterations)} iteration(s)")
        reply = QMessageBox.question(
            self, "Remove Selected",
            "Permanently delete " + " and ".join(parts)
            + " from disk (including their notes)?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete via the existing single-delete path (confirm=False so
        # we don't re-prompt per node). Projects first so a project
        # delete absorbs any of its iterations that were also collected;
        # skip iterations whose project is already gone.
        for project, project_dir in projects:
            self._delete_project(project, project_dir, confirm=False)
        for project, iteration, iter_dir in iterations:
            if project not in self._results_data:
                continue  # project (and this iteration) already removed
            if iteration not in self._results_data.get(project, {}):
                continue
            self._delete_iteration(project, iteration, iter_dir,
                                   confirm=False)

    def _clear_fresh_for_item(self, item):
        """If *item* belongs to a freshly finished iteration, clear that
        iteration's highlight (it's been viewed). Walks up to the
        iteration node, since children (Fit Report, plots, …) don't
        carry the project/iteration keys themselves.
        """
        if not self._fresh_iters:
            return
        node = item
        while node is not None:
            d = node.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(d, dict) and d.get("project") and d.get("iteration"):
                key = (d["project"], d["iteration"])
                if key in self._fresh_iters:
                    self._fresh_iters.discard(key)
                    # Restore the iteration row's normal look IN PLACE
                    # (rebuilding the tree here would delete the clicked
                    # item out from under the caller).
                    node.setText(0, d["iteration"])
                    fnt = node.font(0)
                    fnt.setBold(False)
                    node.setFont(0, fnt)
                    i_bg, i_fg = self._project_brushes(
                        d["project"], iter_index=d.get("iter_index", 0))
                    node.setBackground(0, i_bg)
                    node.setForeground(0, i_fg)
                return
            node = node.parent()

    def _on_item_clicked(self, item, column):
        """Display the selected item in the viewer."""
        # Flush any pending notes before switching away.
        self._save_notes(force=True)

        # Viewing an iteration (or anything under it) clears its
        # "freshly finished" highlight. Done before the early return so
        # clicking the iteration row itself counts as viewing it.
        self._clear_fresh_for_item(item)

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or not isinstance(data, dict) or "type" not in data:
            return

        self._current_item_data = data
        self._current_tree_item = item
        item_type = data["type"]
        filepath = data.get("path", "")
        self._title_label.setText(os.path.basename(filepath))

        if item_type == "notes":
            self._load_notes(filepath, label=data.get("label", ""))
            self._edit_plot_btn.setEnabled(False)
            return

        if item_type == "text":
            try:
                with open(filepath, "r", encoding="utf-8",
                          errors="replace") as f:
                    self._text_viewer.setPlainText(f.read())
            except Exception as e:
                self._text_viewer.setPlainText(f"Error reading file:\n{e}")
            self._viewer_stack.setCurrentIndex(1)
            self._edit_plot_btn.setEnabled(False)

        elif item_type == "csv":
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                self._table_viewer.setRowCount(len(df))
                self._table_viewer.setColumnCount(len(df.columns))
                self._table_viewer.setHorizontalHeaderLabels(
                    list(df.columns))
                for i in range(len(df)):
                    for j in range(len(df.columns)):
                        val = df.iloc[i, j]
                        # code review 2026-06-02, results-csv-float-format-loss:
                        # use higher precision so displayed values match file
                        text = f"{val:.10g}" if isinstance(
                            val, float) else str(val)
                        self._table_viewer.setItem(
                            i, j, QTableWidgetItem(text))
            except Exception as e:
                self._text_viewer.setPlainText(
                    f"Error reading CSV:\n{e}")
                self._viewer_stack.setCurrentIndex(1)
                return
            self._viewer_stack.setCurrentIndex(2)
            self._edit_plot_btn.setEnabled(False)

        elif item_type == "image":
            self._plot_figure.clear()
            # Full-bleed static image: don't let the resize handler impose
            # a tight_layout (it would reintroduce label-band margins).
            self._plot_tight_layout = False
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(filepath)
                ax = self._plot_figure.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                self._plot_figure.subplots_adjust(
                    left=0, right=1, top=1, bottom=0)
                self._plot_canvas.draw()
            except Exception as e:
                ax = self._plot_figure.add_subplot(111)
                ax.text(0.5, 0.5, f"Cannot display:\n{e}",
                        ha='center', va='center',
                        transform=ax.transAxes)
                self._plot_canvas.draw()
            self._viewer_stack.setCurrentIndex(3)
            self._edit_plot_btn.setEnabled(True)

        elif item_type in ("diagnostic_plot", "editable_plot"):
            self._plot_figure.clear()
            try:
                self._render_from_npz(filepath)
            except Exception as e:
                ax = self._plot_figure.add_subplot(111)
                ax.text(0.5, 0.5, f"Cannot render:\n{e}",
                        ha='center', va='center',
                        transform=ax.transAxes)
            self._plot_canvas.draw()
            self._viewer_stack.setCurrentIndex(3)
            self._edit_plot_btn.setEnabled(True)

    def _apply_plot_layout(self):
        """Fit the plot figure's subplots tight against the canvas at the
        CURRENT figure size, then pull single-column stacks together.

        Re-runnable: called after each render and on every canvas resize so
        the margins (which tight_layout stores as size-dependent fractions)
        always match the live window size instead of going slack when the
        window grows.
        """
        fig = self._plot_figure
        if not fig.axes:
            return
        # tight_layout doesn't reserve space for a suptitle, so leave a
        # strip for it (walk / chi-sq / correl plots use one).
        suptitle = getattr(fig, "_suptitle", None)
        has_suptitle = bool(suptitle and suptitle.get_text())
        top_rect = 0.94 if has_suptitle else 1.0
        try:
            # right edge at 0.975 (not 1.0) leaves a small breathing-room
            # margin so the axes/legend don't sit flush against the border.
            fig.tight_layout(pad=0.1, h_pad=0.3, w_pad=0.3,
                             rect=[0, 0, 0.975, top_rect])
        except Exception:
            return
        # Pull stacked/sharex panels (fit+residual, stacked isotope shift,
        # walk, parameters) close together. Done AFTER tight_layout because
        # a gridspec hspace would otherwise disable tight_layout's margin
        # hugging; setting it here only adjusts the inter-panel gap, leaving
        # the tight outer margins intact. Restricted to single-COLUMN stacks
        # (all axes share one x-position) so the correlation-matrix grid and
        # horizontal slice plots keep their normal spacing.
        axs = fig.axes
        if len(axs) > 1:
            x0s = {round(a.get_position().x0, 3) for a in axs}
            # Only pull together a single-column SHAREX stack (fit+residual,
            # walk, parameters), where the x-label lives on the bottom
            # panel alone. Independent stacked panels with their own
            # x-axes (e.g. ToF + spectrum: time vs frequency) keep normal
            # spacing so the upper x-label/ticks don't collide with the
            # lower panel's title.
            n_xlabels = sum(1 for a in axs if a.get_xlabel())
            if len(x0s) == 1 and n_xlabels <= 1:
                fig.subplots_adjust(hspace=0.05)

    def _on_plot_canvas_resize(self, _event):
        """Re-fit the tight layout when the canvas (window) resizes, so the
        plot keeps hugging the border at any size. No-op for the full-bleed
        image page. The re-layout doesn't change canvas geometry, so it
        can't re-trigger this handler."""
        if not self._plot_tight_layout:
            return
        if self._viewer_stack.currentIndex() != 3:
            return
        self._apply_plot_layout()
        self._plot_canvas.draw_idle()

    def _render_from_npz(self, npz_path):
        """Render a live matplotlib plot from a .npz data file."""
        data = np.load(npz_path, allow_pickle=True)
        plot_type = str(data.get("plot_type", ""))
        fig = self._plot_figure

        if plot_type == "tof":
            self._render_tof_plot(fig, data)
        elif plot_type == "tof_spectrum":
            self._render_tof_spectrum_plot(fig, data)
        elif plot_type == "fit":
            self._render_fit_plot(fig, data)
        elif plot_type == "tracker":
            self._render_tracker_plot(fig, data)
        elif plot_type == "walk":
            self._render_walk_plot(fig, data)
        elif plot_type == "correl":
            self._render_correl_plot(fig, data)
        elif plot_type == "chisq":
            self._render_chisq_plot(fig, data)
        elif plot_type == "isotope_shift":
            self._render_isotope_shift_plot(fig, data)

        # Hug the canvas; re-fits automatically on resize via
        # _on_plot_canvas_resize.
        self._plot_tight_layout = True
        self._apply_plot_layout()

        # Apply saved style overrides if they exist
        sp = _style_path(npz_path)
        if os.path.isfile(sp):
            try:
                with open(sp, 'r', encoding="utf-8") as f:
                    style = json.load(f)
                _apply_figure_style(fig, style)
            except Exception:
                _log.warning("Could not apply saved plot style %s; "
                             "showing the unstyled figure", sp, exc_info=True)

    def _render_tof_plot(self, fig, data):
        """Render a saved ToF plot (.npz) live: histogram + gate overlay."""
        centers = np.asarray(data.get("centers", []), dtype=float)
        counts = np.asarray(data.get("counts", []), dtype=float)
        run_num = str(data.get("run_num", "?"))
        ax = fig.subplots(1, 1)
        if centers.size and counts.size:
            edges = np.empty(centers.size + 1, dtype=float)
            edges[:-1] = centers - 0.5
            edges[-1] = centers[-1] + 0.5
            ax.stairs(counts, edges, color="tab:green", linewidth=1.5,
                      label=f"Run {run_num}")
            gate = np.asarray(data.get("tof_gate", []), dtype=float)
            if gate.size == 2 and gate[1] > gate[0]:
                ax.axvspan(gate[0], gate[1], alpha=0.2, color="orange",
                           label=f"ToF gate [{gate[0]:g}, {gate[1]:g}] µs")
            ax.set_xlim(float(centers.min() - 0.5),
                        float(centers.max() + 0.5))
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Counts")
        ax.set_title(f"ToF — Run {run_num}", loc="left")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.margins(y=0.05)

    def _render_tof_spectrum_plot(self, fig, data):
        """Render a saved ToF+spectrum plot (.npz) live: top = gated ToF
        histogram + gate overlay, bottom = gated spectrum (+ optional
        fitted-model overlay)."""
        from gui.shared_widgets import count_errorbar_pairs
        centers = np.asarray(data.get("centers", []), dtype=float)
        counts = np.asarray(data.get("counts", []), dtype=float)
        x = np.asarray(data.get("x", []), dtype=float)
        y = np.asarray(data.get("y", []), dtype=float)
        run_num = str(data.get("run_num", "?"))
        gate = np.asarray(data.get("tof_gate", []), dtype=float)
        yerr = np.asarray(data.get("yerr", []), dtype=float)
        x_smooth = np.asarray(data.get("x_smooth", []), dtype=float)
        y_fit_smooth = np.asarray(data.get("y_fit_smooth", []), dtype=float)
        with_fit = bool(data.get("with_fit", True))
        x_label = str(data.get("x_label", "Frequency (MHz)"))

        ax_tof, ax_spec = fig.subplots(2, 1)

        # Top: ToF + gate
        if centers.size and counts.size:
            edges = np.empty(centers.size + 1, dtype=float)
            edges[:-1] = centers - 0.5
            edges[-1] = centers[-1] + 0.5
            ax_tof.stairs(counts, edges, color="tab:green", linewidth=1.5,
                          label=f"Run {run_num}")
            ax_tof.set_xlim(float(centers.min() - 0.5),
                            float(centers.max() + 0.5))
        if gate.size == 2 and gate[1] > gate[0]:
            ax_tof.axvspan(float(gate[0]), float(gate[1]), alpha=0.2,
                           color="orange",
                           label=f"ToF gate [{gate[0]:g}, {gate[1]:g}] µs")
        ax_tof.set_xlabel("Time (µs)")
        ax_tof.set_ylabel("Counts")
        ax_tof.set_title(f"ToF — Run {run_num}", loc="left")
        ax_tof.legend(fontsize=8)
        ax_tof.grid(True, alpha=0.3)
        ax_tof.margins(y=0.05)

        # Bottom: gated spectrum (+ optional fit)
        s = fit_plot_settings_for_counts(
            float(np.max(y)) if y.size else 0.0)
        yerr_c = (count_errorbar_pairs(y, yerr)
                  if yerr.size == y.size else None)
        ax_spec.errorbar(
            x, y, yerr=yerr_c, fmt=s["data_fmt"], ms=s["data_ms"],
            alpha=s["data_alpha"], capsize=s["data_capsize"],
            elinewidth=s.get("data_elinewidth", 0.5),
            ecolor=s.get("data_ecolor", "0.55"), label="Data")
        if with_fit and x_smooth.size and y_fit_smooth.size:
            ax_spec.plot(x_smooth, y_fit_smooth, color=s["fit_color"],
                         lw=s["fit_lw"], alpha=s["fit_alpha"], label="Fit")
        ax_spec.set_xlabel(x_label)
        ax_spec.set_ylabel("Counts")
        ax_spec.set_title(f"Gated spectrum — Run {run_num}", loc="left")
        ax_spec.legend(fontsize=8)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.margins(y=0.05)
        if x.size:
            ax_spec.set_xlim(float(np.min(x)), float(np.max(x)))

    def _render_fit_plot(self, fig, data):
        run_num = str(data.get("run_num", "?"))
        x = data["x"]
        y = data["y"]
        # Auto-pick marker/alpha defaults from the highest data bin so
        # low-stats runs render with bigger, more opaque markers.
        try:
            peak_count = float(np.max(np.asarray(y))) if len(y) else 0.0
        except Exception:
            peak_count = 0.0
        s = fit_plot_settings_for_counts(peak_count)
        yerr = data["yerr"]
        x_smooth = data["x_smooth"]
        y_fit_smooth = data["y_fit_smooth"]
        residuals = data["residuals"]
        show_resid = bool(data.get("show_resid", True))
        hide_zero_bins = bool(data.get("hide_zero_bins", False))

        # Plot-time filter: drop zero-count bins from markers/residuals
        # AND mask the smooth fit/component curves over those bins, so
        # the line breaks where there is no data.
        y_fit_smooth_plot = y_fit_smooth
        comp_mask = None
        if hide_zero_bins:
            import numpy as _np
            ya_y = _np.asarray(y)
            nz = ya_y > 0
            x_plot, y_plot = _np.asarray(x)[nz], ya_y[nz]
            ya = _np.asarray(yerr)
            yerr_plot = ya[nz] if ya.ndim > 0 and len(ya) == len(y) else yerr
            ra = _np.asarray(residuals)
            res_plot = ra[nz] if len(ra) == len(y) else residuals
            xs = _np.asarray(x, dtype=float)
            xs_smooth = _np.asarray(x_smooth, dtype=float)
            if len(xs) >= 2 and len(xs_smooth) > 0:
                edges = _np.empty(len(xs) + 1)
                edges[1:-1] = 0.5 * (xs[:-1] + xs[1:])
                edges[0] = xs[0] - 0.5 * (xs[1] - xs[0])
                edges[-1] = xs[-1] + 0.5 * (xs[-1] - xs[-2])
                bin_idx = _np.clip(
                    _np.searchsorted(edges, xs_smooth) - 1,
                    0, len(xs) - 1)
                comp_mask = (ya_y == 0)[bin_idx]
                y_fit_smooth_plot = _np.where(
                    comp_mask, _np.nan, _np.asarray(y_fit_smooth))
        else:
            x_plot, y_plot, yerr_plot, res_plot = x, y, yerr, residuals

        if show_resid:
            # NB: do NOT set hspace in gridspec_kw -- a manual hspace makes
            # the figure "incompatible with tight_layout" (it bails with a
            # warning and leaves the wide initial subplots_adjust margins,
            # wasting canvas on the left/bottom). _render_from_npz tightens
            # the inter-panel gap via subplots_adjust(hspace=...) AFTER
            # tight_layout instead, keeping the hugged outer margins.
            ax_main, ax_res = fig.subplots(
                2, 1, sharex=True,
                gridspec_kw={"height_ratios": [3, 1]})
        else:
            ax_main = fig.subplots(1, 1)
            ax_res = None

        # Clip the lower error bar at y=0 so error bars never dip below
        # the physical zero-count floor.
        from gui.shared_widgets import count_errorbar_pairs
        yerr_plot_clipped = count_errorbar_pairs(y_plot, yerr_plot)
        ax_main.errorbar(x_plot, y_plot, yerr=yerr_plot_clipped,
                         fmt=s["data_fmt"],
                         ms=s["data_ms"], alpha=s["data_alpha"],
                         capsize=s["data_capsize"],
                         elinewidth=s.get("data_elinewidth", 0.5),
                         ecolor=s.get("data_ecolor", "0.55"),
                         label='Data')
        ax_main.plot(x_smooth, y_fit_smooth_plot, color=s["fit_color"],
                     lw=s["fit_lw"], alpha=s["fit_alpha"], label='Fit')
        # Shaded area under the fit curve (matches the saved output
        # plot; "auto" = fit-line color at low alpha).
        if s.get("shade_under_fit", False):
            _shade_c = s.get("shade_color", "auto")
            if not _shade_c or _shade_c == "auto":
                _shade_c = s["fit_color"]
            ax_main.fill_between(
                x_smooth, 0.0, y_fit_smooth_plot, color=_shade_c,
                alpha=float(s.get("shade_alpha", 0.12)), lw=0, zorder=1)
        # Per-model component overlay
        # Always add component lines so Edit Plot can toggle them;
        # initial visibility follows the component_overlay flag from config
        import matplotlib.pyplot as _plt
        _comp_colors = _plt.rcParams['axes.prop_cycle'].by_key()['color']
        comp_names = list(data.get("component_names", []))
        overlay_on = bool(data.get("component_overlay", False))
        sel_comps = list(data.get("selected_components", []))
        import numpy as _np_local
        for ci, cname in enumerate(comp_names):
            ckey = f"component_{ci}"
            if ckey in data:
                visible = overlay_on and (
                    not sel_comps or str(cname) in sel_comps)
                cc = _comp_colors[(ci + 1) % len(_comp_colors)]
                cdata_arr = _np_local.asarray(data[ckey])
                if comp_mask is not None and len(cdata_arr) == len(comp_mask):
                    cdata_arr = _np_local.where(
                        comp_mask, _np_local.nan, cdata_arr)
                ax_main.plot(x_smooth, cdata_arr, color=cc,
                             lw=s["fit_lw"] * 0.7, alpha=0.7, ls='--',
                             label=str(cname), visible=visible)
        # Confidence bands from MCMC
        if "band_x" in data:
            bx = data["band_x"]
            ax_main.fill_between(bx, data["band_lo"], data["band_hi"],
                                 color=s["fit_color"], alpha=s["band_alpha"],
                                 label='1\u03c3 band')
        ax_main.set_ylabel("Counts", fontsize=s["label_size"])
        ax_main.set_title(f"Run {run_num}", loc='left',
                          fontsize=s["title_size"])
        # On-plot fit-values box, drawn verbatim from the lines the
        # output writer pre-formatted into the .npz (absent in old
        # files → no box).
        vlines = data.get("values_lines")
        if vlines is not None and len(vlines):
            ax_main.text(
                0.98, 0.98, "\n".join(str(v) for v in vlines),
                transform=ax_main.transAxes, ha="right", va="top",
                fontsize=s.get("values_box_fontsize", 11),
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="black", lw=0.8, alpha=0.9),
                zorder=6)
        # Legend only for visible artists
        handles, labels = ax_main.get_legend_handles_labels()
        vis_h, vis_l = [], []
        for h, lb in zip(handles, labels):
            if getattr(h, 'get_visible', lambda: True)():
                vis_h.append(h)
                vis_l.append(lb)
        if vis_h:
            ax_main.legend(vis_h, vis_l, loc="upper left")
        ax_main.grid(True, alpha=s["grid_alpha"])
        # Hug the DATA x-range. margins()/autoscale would include the fit
        # curve, which is evaluated on a wider grid than the data and so
        # left an empty margin on both sides. Pin x to the data extent
        # instead (residual panel follows via sharex). Keep a little y
        # headroom so peaks/error bars aren't clipped.
        ax_main.margins(y=0.05)
        ax_main.autoscale_view()
        _xp = np.asarray(x_plot, dtype=float)
        if _xp.size > 0:
            ax_main.set_xlim(float(np.min(_xp)), float(np.max(_xp)))

        # Pick the x-axis label from the saved bin mode (the .npz
        # carries `bin_mode` from the original fit's binning_info).
        bm = str(data.get("bin_mode") or "Frequency")
        x_label = ("Voltage (V)" if bm == "Raw Voltage"
                   else "Frequency (MHz)")
        if ax_res is not None:
            ax_res.errorbar(x_plot, res_plot, yerr=1, fmt=s["resid_fmt"],
                            ms=s["resid_ms"], alpha=s["data_alpha"],
                            capsize=s.get("data_capsize", 0.0),
                            elinewidth=s.get("data_elinewidth", 0.5),
                            ecolor=s.get("data_ecolor", "0.55"))
            ax_res.axhline(0, color=s["fit_color"], ls='--', alpha=0.5)
            ax_res.set_ylabel(r"Res. ($\sigma$)", fontsize=s["label_size"])
            ax_res.set_xlabel(x_label, fontsize=s["label_size"])
            ax_res.grid(True, alpha=s["grid_alpha"])
        else:
            ax_main.set_xlabel(x_label, fontsize=s["label_size"])

    def _render_tracker_plot(self, fig, data):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        ts = get_plot_type_settings("tracker_plot")
        model_name = str(data.get("model_name", ""))
        params = list(data["params"])
        xlabel = str(data.get("xlabel", "Run number"))
        x_mode = str(data.get("x_mode", xlabel))
        use_time_axis = (x_mode == "Start time")
        n_params = len(params)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        axes = fig.subplots(n_params, 1, sharex=True, squeeze=False)

        for j, param in enumerate(params):
            ax = axes[j, 0]
            color = colors[j % len(colors)]
            runs = data.get(f"runs_{j}", np.array([]))
            vals = data.get(f"vals_{j}", np.array([]))
            errs = data.get(f"errs_{j}", np.array([]))
            times = data.get(f"times_{j}", np.array([]))
            if len(runs) == 0:
                ax.text(0.5, 0.5, f"No data for {param}",
                        ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                continue
            times_arr = np.asarray(times) if len(times) else np.array([])
            if use_time_axis and times_arr.size and np.any(times_arr > 0):
                x_plot = np.array([
                    mdates.date2num(datetime.fromtimestamp(float(t)))
                    for t in times_arr
                ])
                date_axis = True
            else:
                x_plot = runs
                date_axis = False
            # Academic style (mirrors the saved tracker output):
            # unconnected points; weighted-mean line + horizontal ±1σ
            # band instead of the old point-connecting polygon.
            vals = np.asarray(vals, dtype=float)
            errs = np.asarray(errs, dtype=float)
            ax.errorbar(x_plot, vals, yerr=errs,
                        fmt=ts["marker_fmt"], color=color,
                        ms=ts["marker_ms"], capsize=ts["capsize"],
                        elinewidth=ts.get("elinewidth", 1.2), zorder=3)
            if ts.get("connect_runs", False):
                ax.plot(x_plot, vals, color=color, lw=0.8,
                        alpha=0.5, zorder=2)
            if ts.get("mean_line", True) and len(vals) > 1:
                w = np.where(errs > 0,
                             1.0 / np.maximum(errs, 1e-300) ** 2, 0.0)
                if np.any(w > 0):
                    mean = float(np.sum(w * vals) / np.sum(w))
                    mean_err = float(1.0 / np.sqrt(np.sum(w)))
                else:
                    mean = float(np.mean(vals))
                    mean_err = 0.0
                mc = ts.get("mean_color", "0.25")
                ax.axhline(mean, color=mc, ls=ts.get("mean_ls", "--"),
                           lw=1.2, zorder=1,
                           label="Weighted mean $\\pm 1\\sigma$")
                if mean_err > 0:
                    ax.axhspan(mean - mean_err, mean + mean_err,
                               color=mc, alpha=ts["band_alpha"],
                               lw=0, zorder=0)
                if j == 0:
                    ax.legend(loc="best")
            from gui.analysis.helpers import param_axis_label
            ax.set_ylabel(param_axis_label(param),
                          fontsize=ts["label_size"])
            ax.set_title(f"{model_name}: {param[:1].upper()}{param[1:]}",
                         loc='left', fontsize=ts["title_size"])
            ax.grid(True, alpha=ts["grid_alpha"])
            if date_axis:
                ax.xaxis_date()
                ax.set_xticks(x_plot)
                date_strs = [
                    datetime.fromtimestamp(float(t)).strftime(
                        "%Y-%m-%d\n%H:%M")
                    for t in times_arr
                ]
                if len(x_plot) > 2:
                    span = float(x_plot[-1] - x_plot[0])
                    min_gap = span / 6.0
                    last_shown = x_plot[0]
                    for i in range(1, len(x_plot) - 1):
                        if (x_plot[i] - last_shown) < min_gap:
                            date_strs[i] = ""
                        else:
                            last_shown = x_plot[i]
                ax.set_xticklabels(date_strs, rotation=30, ha="right")
            else:
                ax.set_xticks(x_plot)
        axes[-1, 0].set_xlabel(xlabel, fontsize=ts["label_size"])

    def _render_walk_plot(self, fig, data):
        ws = get_plot_type_settings("walk_plot")
        run_num = str(data.get("run_num", "?"))
        labels = list(data["labels"])
        chain = data["chain"]
        n_var = len(labels)
        axes = fig.subplots(n_var, 1, sharex=True, squeeze=False)
        from gui.analysis.helpers import param_axis_label
        for i, label in enumerate(labels):
            ax = axes[i, 0]
            ax.plot(chain[:, :, i], alpha=ws["trace_alpha"],
                    lw=ws["trace_lw"])
            ax.set_ylabel(param_axis_label(label),
                          fontsize=ws["label_size"])
            ax.tick_params(labelsize=ws["tick_size"])
        axes[-1, 0].set_xlabel("Step", fontsize=ws["label_size"])
        fig.suptitle(f"Walk Plot \u2014 Run {run_num}",
                     fontsize=ws["title_size"])

    def _render_correl_plot(self, fig, data):
        cs = get_plot_type_settings("correlation_plot")
        run_num = str(data.get("run_num", "?"))
        labels = list(data["labels"])
        flat = data["flatchain"]
        n_var = len(labels)
        axes = fig.subplots(n_var, n_var)
        if n_var == 1:
            axes = np.array([[axes]])
        for i in range(n_var):
            for j in range(n_var):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    ax.hist(flat[:, i], bins=cs["hist_bins"],
                            color=cs["hist_color"],
                            alpha=cs["hist_alpha"], density=True)
                else:
                    ax.scatter(flat[:, j], flat[:, i],
                               s=cs["scatter_s"],
                               alpha=cs["scatter_alpha"],
                               color=cs["scatter_color"])
                from gui.analysis.helpers import param_axis_label
                if i == n_var - 1:
                    ax.set_xlabel(param_axis_label(labels[j]),
                                  fontsize=cs["label_size"])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(param_axis_label(labels[i]),
                                  fontsize=cs["label_size"])
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=cs["tick_size"])
        fig.suptitle(f"Correlation \u2014 Run {run_num}",
                     fontsize=cs["title_size"])

    def _render_chisq_plot(self, fig, data):
        qs = get_plot_type_settings("chisq_map")
        run_num = str(data.get("run_num", "?"))
        labels = list(data["labels"])
        scans = data["scans"]
        chisq = data["chisq"]
        best = data["best"]
        n_var = len(labels)
        axes = fig.subplots(1, n_var, squeeze=False)
        from gui.analysis.helpers import param_axis_label
        for idx in range(n_var):
            ax = axes[0, idx]
            ax.plot(scans[idx], chisq[idx], qs["line_fmt"],
                    ms=qs["line_ms"])
            ax.axvline(best[idx], color=qs["best_color"],
                       ls=qs["best_ls"], alpha=0.5)
            ax.set_xlabel(param_axis_label(labels[idx]),
                          fontsize=qs["label_size"])
            ax.set_ylabel(r"$\chi^2$", fontsize=qs["label_size"])
            ax.tick_params(labelsize=qs["tick_size"])
        fig.suptitle(f"Chi-sq Map \u2014 Run {run_num}",
                     fontsize=qs["title_size"])

    def _render_isotope_shift_plot(self, fig, data):
        """Render an isotope-shift comparison plot from saved .npz data.

        Reconstructs the figure that was sent to Results: per-isotope data,
        model fits, centroid lines, an optional reference line, and optional
        delta-nu arrows, in either stacked-subplot or single-axes layout.
        Every per-artist style is read from the .npz with fallbacks to the
        global plot-type defaults.
        """
        from cls_estimations.plotting import _get_color
        s = get_plot_type_settings("isotope_shift_plot")

        # Helper: numpy 0-d arrays come back from np.load as np.ndarray;
        # unbox them to native Python types so str/bool/float compare ok.
        def _g(key, default):
            if key not in data:
                return default
            v = data[key]
            try:
                v = v.item()
            except (AttributeError, ValueError):
                pass
            return v

        layout = str(_g("layout", "Stacked Subplots"))
        content = str(_g("content", "Data + Model"))
        n = int(_g("n_isotopes", 0))
        if n == 0:
            return
        labels = list(data.get("labels", []))
        labels = [str(x.item() if hasattr(x, 'item') else x) for x in labels]
        ref_idx = int(_g("reference_idx", 0))
        show_centroids = bool(_g("show_centroids", True))
        show_ref_line = bool(_g("show_ref_line", False))
        show_arrows = bool(_g("show_arrows", False))
        sort_by_A = bool(_g("sort_by_A", True))
        hide_zero_bins = bool(_g("hide_zero_bins", False))
        x_mode = str(_g("x_mode", "Relative to Reference"))
        stacked = "Stacked" in layout

        # Style overrides \u2014 fall back to plot-type defaults
        model_color = str(_g("model_color", "#000000"))
        model_alpha = float(_g("model_alpha", s.get("fit_alpha", 0.85)))
        model_z = int(_g("model_zorder", 3))
        centroid_color = str(_g("centroid_color", "#555555"))
        centroid_alpha = float(_g("centroid_alpha",
                                   s.get("centroid_alpha", 0.5)))
        centroid_z = int(_g("centroid_zorder", 1))
        ref_line_color = str(_g("ref_line_color", "#FF0000"))
        ref_alpha = float(_g("ref_alpha", 0.7))
        ref_z = int(_g("ref_zorder", 1))
        arrow_color = str(_g("arrow_color",
                             s.get("arrow_color", "#333333")))
        arrow_alpha = float(_g("arrow_alpha", 1.0))
        arrow_z = int(_g("arrow_zorder", 5))

        # Reference centroid
        ref_centroid = float(_g(f"centroid_{ref_idx}", 0.0))
        x_offset = ref_centroid if "Relative" in x_mode else 0.0

        # Build ordered isotope index list (sort by A if requested)
        order = list(range(n))
        if sort_by_A:
            order.sort(key=lambda j: float(_g(f"A_{j}", j)))

        if stacked:
            # hspace omitted on purpose (breaks tight_layout); the gap is
            # re-tightened post-tight_layout in _render_from_npz.
            axes = fig.subplots(n, 1, sharex=True)
            if n == 1:
                axes = [axes]
        else:
            ax_single = fig.add_subplot(111)
            axes = [ax_single] * n

        for plot_pos, i in enumerate(order):
            ax = axes[plot_pos]
            # Saved color falls back to the matplotlib cycle if absent.
            color = str(_g(f"color_{i}", _get_color(i)))
            label = labels[i] if i < len(labels) else f"#{i}"
            centroid = float(_g(f"centroid_{i}", 0.0))
            centroid_plot = centroid - x_offset
            is_ref = bool(_g(f"is_reference_{i}", i == ref_idx))

            x_key, y_key = f"x_{i}", f"y_{i}"
            yerr_key = f"yerr_{i}"
            xs_key, ys_key = f"x_smooth_{i}", f"y_smooth_{i}"

            x_data = np.asarray(data[x_key]) if x_key in data else None
            y_data = np.asarray(data[y_key]) if y_key in data else None
            yerr_data = (np.asarray(data[yerr_key])
                         if yerr_key in data else None)
            x_smooth = (np.asarray(data[xs_key])
                        if xs_key in data else None)
            y_smooth = (np.asarray(data[ys_key])
                        if ys_key in data else None)

            # Apply zero-bin masking: drop empty data points and break
            # the smooth fit curve over those gaps.
            y_smooth_for_plot = y_smooth
            if hide_zero_bins and y_data is not None and len(y_data) > 0:
                nz = y_data > 0
                x_data_used = x_data[nz]
                y_data_used = y_data[nz]
                yerr_used = (yerr_data[nz]
                             if (yerr_data is not None
                                 and yerr_data.ndim > 0
                                 and len(yerr_data) == len(y_data))
                             else yerr_data)
                if (x_smooth is not None and y_smooth is not None
                        and len(x_data) >= 2 and len(x_smooth) > 0):
                    xs = x_data.astype(float)
                    edges = np.empty(len(xs) + 1)
                    edges[1:-1] = 0.5 * (xs[:-1] + xs[1:])
                    edges[0] = xs[0] - 0.5 * (xs[1] - xs[0])
                    edges[-1] = xs[-1] + 0.5 * (xs[-1] - xs[-2])
                    bin_idx = np.clip(
                        np.searchsorted(edges, x_smooth) - 1,
                        0, len(xs) - 1)
                    zero_mask = (y_data == 0)[bin_idx]
                    y_smooth_for_plot = np.where(
                        zero_mask, np.nan, y_smooth.astype(float))
            else:
                x_data_used, y_data_used, yerr_used = (
                    x_data, y_data, yerr_data)

            # Plot data (with offset)
            if x_data_used is not None and len(x_data_used) > 0:
                ax.errorbar(x_data_used - x_offset, y_data_used,
                            yerr=yerr_used,
                            fmt=s.get("data_fmt", "."), color=color,
                            ms=s.get("data_ms", 5.0),
                            alpha=s.get("data_alpha", 0.7),
                            capsize=s.get("data_capsize", 3.0),
                            label=f"{label} data", zorder=2)

            # Plot model fit in the model_color (typically black)
            if ("Model" in content and x_smooth is not None
                    and y_smooth_for_plot is not None
                    and len(x_smooth) > 0):
                ax.plot(x_smooth - x_offset, y_smooth_for_plot,
                        color=model_color,
                        lw=s.get("fit_lw", 2.0),
                        alpha=model_alpha,
                        label=f"{label} fit", zorder=model_z)
                # Pale shade under the fit (mirrors the IS tab).
                if s.get("shade_under_fit", False):
                    _shade_c = s.get("shade_color", "auto")
                    if not _shade_c or _shade_c == "auto":
                        _shade_c = model_color
                    ax.fill_between(
                        x_smooth - x_offset, 0.0, y_smooth_for_plot,
                        color=_shade_c,
                        alpha=float(s.get("shade_alpha", 0.12)),
                        lw=0, zorder=1)

            # Per-isotope centroid line (own color, not data color)
            if show_centroids:
                ax.axvline(centroid_plot,
                           ls=s.get("centroid_ls", "--"),
                           color=centroid_color,
                           alpha=centroid_alpha, lw=1.0, zorder=centroid_z)

            # Reference centroid drawn on every panel (skip on ref itself)
            if show_ref_line and not is_ref:
                ref_plot = ref_centroid - x_offset
                ax.axvline(ref_plot,
                           ls=":", color=ref_line_color,
                           alpha=ref_alpha, lw=1.2, zorder=ref_z)

            if stacked:
                ax.set_ylabel("Counts (a.u.)")
                _label_kwargs = dict(
                    transform=ax.transAxes,
                    fontsize=s.get("label_size", 12),
                    fontweight="bold")
                if s.get("boxed_labels", False):
                    _label_kwargs["bbox"] = dict(
                        boxstyle="square,pad=0.3", fc="white",
                        ec="black", lw=1.0, alpha=0.9)
                ax.text(0.02, 0.85, label, **_label_kwargs)
                if plot_pos < n - 1:
                    ax.tick_params(labelbottom=False)
            else:
                ax.legend(fontsize=9, framealpha=0.7)

        # \u03b4\u03bd arrows (connect each isotope centroid to the reference)
        if show_arrows:
            ref_plot = ref_centroid - x_offset
            # code review 2026-06-02, is-overlaid-arrow-ypos: in Overlaid mode
            # all arrows share one axes, so stagger their y-position by the
            # k-th drawn arrow to stop labels overprinting (bounded fraction).
            k = 0
            for plot_pos, i in enumerate(order):
                if bool(_g(f"is_reference_{i}", i == ref_idx)):
                    continue
                ax = axes[plot_pos] if stacked else axes[0]
                c_plot = float(_g(f"centroid_{i}", 0.0)) - x_offset
                y_lim = ax.get_ylim()
                if stacked:
                    y_pos = y_lim[1] * 0.9
                else:
                    y_pos = y_lim[1] * max(0.4, 0.9 - 0.08 * k)
                k += 1
                dnu = float(_g(f"delta_nu_{i}", 0.0))
                _err = float(_g(f"sigma_delta_nu_{i}", 0.0) or 0.0)
                aid = str(labels[i]) if i < len(labels) else str(i)
                # Same artist types + gids as the IS tab's own renderer
                # (FancyArrowPatch + separate text). The old ax.annotate
                # version made the two renderers' artist inventories
                # differ, so the index-matched style sidecar smeared the
                # \u03b4\u03bd label onto the annotation arrow \u2014 the doubled-arrow
                # bug when a plot was sent to Results.
                from matplotlib.patches import FancyArrowPatch
                arrow = FancyArrowPatch(
                    (c_plot, y_pos), (ref_plot, y_pos),
                    arrowstyle="<->", mutation_scale=12,
                    color=arrow_color, alpha=arrow_alpha,
                    lw=s.get("arrow_lw", 1.5), zorder=arrow_z)
                arrow.set_gid(f"isarrow:{aid}")
                ax.add_patch(arrow)
                from gui.analysis.helpers import format_value_error
                txt = ax.text(
                    (c_plot + ref_plot) / 2, y_pos * 1.02,
                    "\u03b4\u03bd = " + format_value_error(
                        dnu, _err if _err > 0 else None, "MHz"),
                    ha="center", va="bottom", fontsize=8,
                    color=arrow_color)
                txt.set_gid(f"istext:{aid}")

        bottom_ax = axes[-1] if stacked else axes[0]
        if "Relative" in x_mode:
            ref_label = labels[ref_idx] if ref_idx < len(labels) else "ref"
            bottom_ax.set_xlabel(
                f"Frequency relative to {ref_label} centroid (MHz)",
                fontsize=s.get("label_size", 12))
        else:
            bottom_ax.set_xlabel("Frequency (MHz)",
                                 fontsize=s.get("label_size", 12))
        if not stacked:
            bottom_ax.set_ylabel("Counts (a.u.)")

    def _edit_plot(self):
        """Open plot editor for the current plot.

        Opens modeless so mouse events still reach the canvas — that's
        what makes the editor's drag-to-reposition mode work. Style is
        saved to disk when the dialog closes via the `finished` signal.
        """
        if not (self._current_item_data and self._current_item_data.get(
                "type") in ("image", "diagnostic_plot", "editable_plot")):
            return
        if (getattr(self, "_edit_dialog", None) is not None
                and self._edit_dialog.isVisible()):
            self._edit_dialog.raise_()
            self._edit_dialog.activateWindow()
            return
        try:
            source_path = self._current_item_data.get("path", "")
            self._edit_dialog = PlotEditorDialog(
                self._plot_figure, self._plot_canvas, parent=self,
                source_path=source_path)

            def _on_closed(_result, sp_path=source_path):
                try:
                    self._plot_canvas.draw()
                    if sp_path and sp_path.lower().endswith('.npz'):
                        style = _extract_figure_style(self._plot_figure)
                        # Deletions made in the editor: merge with any
                        # previously-saved removals so they accumulate
                        # instead of resetting per session.
                        removed = set()
                        try:
                            with open(_style_path(sp_path), 'r',
                                      encoding="utf-8") as f:
                                removed |= set(json.load(f).get(
                                    "removed_gids") or [])
                        except Exception:
                            pass
                        removed |= set(getattr(
                            self._edit_dialog, "removed_gids", None)
                            or [])
                        if removed:
                            style["removed_gids"] = sorted(removed)
                        with open(_style_path(sp_path), 'w',
                                  encoding="utf-8") as f:
                            json.dump(style, f, indent=1)
                except Exception:
                    _log.warning("Failed to save plot style edits for %s",
                                 sp_path, exc_info=True)

            self._edit_dialog.finished.connect(_on_closed)
            self._edit_dialog.show()
        except Exception:
            _log.warning("Could not open the plot editor", exc_info=True)

    def _export_current(self):
        """Export the currently viewed item."""
        if not self._current_item_data:
            return
        item_type = self._current_item_data.get("type", "")
        filepath = self._current_item_data.get("path", "")

        from gui.shared_widgets import get_last_dir, remember_last_dir

        # For plot types, use Save As dialog with the figure
        if item_type in ("image", "diagnostic_plot", "editable_plot"):
            if not self._plot_figure.axes:
                return
            base = os.path.splitext(os.path.basename(filepath))[0]
            last_save = get_last_dir("data", "save")
            default_path = (os.path.join(last_save, f"{base}.png")
                            if last_save else f"{base}.png")
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot As", default_path,
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)")
            if not save_path:
                return
            remember_last_dir("data", "save", save_path)
            self._plot_figure.savefig(save_path, dpi=300,
                                      bbox_inches="tight")
            QMessageBox.information(
                self, "Export", f"Saved to:\n{save_path}")
            return

        # For text/csv, copy the file
        if not filepath or not os.path.isfile(filepath):
            return
        last_save = get_last_dir("data", "save")
        default_path = (os.path.join(last_save, os.path.basename(filepath))
                        if last_save else os.path.basename(filepath))
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export File",
            default_path,
            "All files (*)")
        if not save_path:
            return
        remember_last_dir("data", "save", save_path)
        import shutil
        try:
            shutil.copy2(filepath, save_path)
            QMessageBox.information(
                self, "Export", f"Saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export:\n{e}")

    def _export_all(self):
        """Export the entire current iteration directory."""
        from gui.shared_widgets import get_last_dir, remember_last_dir
        # Find the iteration directory from the current selection
        item = self._tree.currentItem()
        if not item:
            QMessageBox.information(self, "Export All",
                                    "No iteration selected.")
            return
        # Walk up to find iteration-level item (depth 1)
        while item.parent() and item.parent().parent():
            item = item.parent()
        iter_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not iter_data or "dir" not in iter_data:
            QMessageBox.information(self, "Export All",
                                    "Select an iteration to export.")
            return
        # A project header also carries a "dir" (the whole project tree but no
        # "iteration" key), so exporting it here would silently copy EVERY
        # iteration -- contradicting "export the current iteration". Require an
        # iteration (code review 2026-06-02, results-export-all-project-row).
        if "iteration" not in iter_data:
            QMessageBox.information(
                self, "Export All",
                "Select an iteration to export (not the project header) -- "
                "expand the project and pick one.")
            return
        src_dir = iter_data["dir"]
        if not os.path.isdir(src_dir):
            QMessageBox.warning(self, "Export All",
                                f"Directory not found:\n{src_dir}")
            return
        dest = QFileDialog.getExistingDirectory(
            self, "Export All - Choose Destination",
            get_last_dir("data", "save"))
        if not dest:
            return
        remember_last_dir("data", "save", dest)
        import shutil
        dest_name = os.path.join(dest, os.path.basename(src_dir))
        try:
            shutil.copytree(src_dir, dest_name)
            QMessageBox.information(
                self, "Export All", f"Exported to:\n{dest_name}")
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed:\n{e}")

    # ── Right-click delete (project / iteration) ──

    @staticmethod
    def _is_styleable_plot(data):
        """A tree item whose look is saved in a sidecar .style.json —
        the editable live plots backed by an .npz."""
        return (isinstance(data, dict)
                and data.get("type") in ("editable_plot", "diagnostic_plot")
                and str(data.get("path", "")).lower().endswith(".npz"))

    def _checked_styleable_paths(self):
        """.npz paths of every ticked editable plot in the tree."""
        paths = []
        for item in self._iter_all_items():
            if item.checkState(0) != Qt.CheckState.Checked:
                continue
            d = item.data(0, Qt.ItemDataRole.UserRole)
            if self._is_styleable_plot(d):
                paths.append(d["path"])
        return paths

    def _on_tree_context_menu(self, point):
        """Right-click menu. Plot rows get Copy/Paste plot style;
        project and iteration rows get Delete. Other rows (notes, a
        single fit_report.txt) get nothing, so they can't be nuked.
        """
        item = self._tree.itemAt(point)
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return

        # ── Editable plot: copy/paste style ──
        if self._is_styleable_plot(data):
            self._plot_style_context_menu(point, data["path"])
            return

        # Iteration row → has both project + iteration; project row →
        # has project but NOT iteration.
        is_project = "project" in data and "iteration" not in data \
            and "type" not in data
        is_iteration = "project" in data and "iteration" in data \
            and "type" not in data
        if not (is_project or is_iteration):
            return

        menu = QMenu(self)
        if is_iteration:
            label = (f"Delete iteration “{data['iteration']}”…")
        else:
            label = f"Delete project “{data['project']}” (all iterations)…"
        action = menu.addAction(label)
        chosen = menu.exec(self._tree.viewport().mapToGlobal(point))
        if chosen is not action:
            return
        if is_iteration:
            self._delete_iteration(data["project"], data["iteration"],
                                   data.get("dir", ""))
        else:
            self._delete_project(data["project"], data.get("dir", ""))

    def _plot_style_context_menu(self, point, npz_path):
        """Copy this plot's saved style, or paste a previously copied
        one onto it (and, optionally, every ticked plot)."""
        menu = QMenu(self)
        copy_act = menu.addAction("Copy plot style")
        have_clip = bool(self._style_clipboard)
        src = (self._style_clipboard or {}).get("source", "")
        paste_act = menu.addAction(
            f"Paste plot style{f' (from {src})' if src else ''}")
        paste_act.setEnabled(have_clip)
        checked = self._checked_styleable_paths()
        paste_checked_act = None
        if have_clip and len(checked) > 1:
            paste_checked_act = menu.addAction(
                f"Paste style to {len(checked)} checked plots")
        menu.addSeparator()
        csv_act = menu.addAction("Export data as CSV…")
        chosen = menu.exec(self._tree.viewport().mapToGlobal(point))
        if chosen is None:
            return
        if chosen is copy_act:
            self._copy_plot_style(npz_path)
        elif chosen is paste_act:
            self._paste_plot_style([npz_path])
        elif chosen is paste_checked_act:
            self._paste_plot_style(checked)
        elif chosen is csv_act:
            self._export_plot_csv(npz_path)

    def _copy_plot_style(self, npz_path):
        sp = _style_path(npz_path)
        if not os.path.isfile(sp):
            QMessageBox.information(
                self, "Copy plot style",
                "This plot has no saved edits yet.\n\nOpen it, adjust "
                "it in the Plot Editor, close the editor to save the "
                "edits, then copy.")
            return
        try:
            with open(sp, "r", encoding="utf-8") as f:
                style = json.load(f)
        except Exception:
            QMessageBox.warning(
                self, "Copy plot style",
                "Could not read this plot's saved style.")
            return
        # Deletions are per-plot (gid-specific); copy only the look.
        style.pop("removed_gids", None)
        # Let the user narrow WHICH properties travel.
        present = _style_categories_present(style)
        if not present:
            QMessageBox.information(
                self, "Copy plot style",
                "This plot's saved style has no recognised properties.")
            return
        dlg = StylePropertyDialog(
            present, self, title="Copy plot style",
            intro="Tick the properties to copy from "
                  f"<b>{os.path.basename(npz_path)}</b>:")
        if not dlg.exec():
            return
        chosen = dlg.selected()
        if not chosen:
            return
        self._style_clipboard = {
            "style": _filter_style(style, chosen),
            "source": os.path.basename(npz_path)}

    def _paste_plot_style(self, npz_paths):
        if not self._style_clipboard:
            return
        style = self._style_clipboard.get("style", {})
        # Let the user narrow WHICH of the copied properties to apply.
        present = _style_categories_present(style)
        if not present:
            return
        n = len(npz_paths)
        dlg = StylePropertyDialog(
            present, self, title="Paste plot style",
            intro=(f"Tick the properties to apply to {n} "
                   f"plot{'s' if n != 1 else ''}:"))
        if not dlg.exec():
            return
        chosen = dlg.selected()
        if not chosen:
            return
        style = _filter_style(style, chosen)
        applied = 0
        for npz in npz_paths:
            sp = _style_path(npz)
            merged = dict(style)
            # Preserve the TARGET's own deletions — pasted style changes
            # the look, not which artists this plot removed.
            try:
                if os.path.isfile(sp):
                    with open(sp, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if existing.get("removed_gids"):
                        merged["removed_gids"] = existing["removed_gids"]
            except Exception:
                pass
            try:
                with open(sp, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=1)
                applied += 1
            except Exception:
                _log.warning("Could not write style to %s", sp,
                             exc_info=True)
        # Refresh the viewer if the currently shown plot was a target.
        cur = (self._current_item_data or {}).get("path", "")
        if cur in npz_paths and self._current_tree_item is not None:
            self._on_item_clicked(self._current_tree_item, 0)
        QMessageBox.information(
            self, "Paste plot style",
            f"Applied the copied style to {applied} "
            f"plot{'s' if applied != 1 else ''}.")

    # ── CSV export of a plot's underlying data ────────────────────

    @staticmethod
    def _plot_csv_columns(plot_type, data):
        """Build an ordered {name: 1-D array} of everything the plot
        stores, so the user can export data points, errors, the model
        and residuals. Ragged columns (e.g. the dense fit curve) are
        allowed — the writer pads them."""
        cols = {}

        def arr(key):
            v = data.get(key)
            if v is None:
                return None
            try:
                a = np.asarray(v, dtype=float).ravel()
            except (ValueError, TypeError):
                return None   # non-numeric (e.g. a label string)
            return a if a.size else None

        if plot_type == "fit":
            x, y, yerr = arr("x"), arr("y"), arr("yerr")
            res = arr("residuals")
            if x is not None:
                cols["x"] = x
            if y is not None:
                cols["y"] = y
            if yerr is not None:
                cols["y_err"] = yerr
            if (res is not None and y is not None
                    and len(res) == len(y)):
                cols["model"] = y - res       # residual = data − model
                cols["residual"] = res
                if yerr is not None:
                    cols["residual_err"] = yerr
            xs, ys = arr("x_smooth"), arr("y_fit_smooth")
            if xs is not None and ys is not None:
                cols["x_fit_curve"] = xs
                cols["y_fit_curve"] = ys
            return cols

        if plot_type in ("tof", "tof_spectrum"):
            for k in ("centers", "counts", "x", "y", "yerr",
                      "x_smooth", "y_fit_smooth"):
                a = arr(k)
                if a is not None:
                    cols[k] = a
            return cols

        if plot_type == "tracker":
            params = list(data.get("params", []))
            for j, p in enumerate(params):
                for suffix, key in (("run", f"runs_{j}"),
                                    ("value", f"vals_{j}"),
                                    ("err", f"errs_{j}")):
                    a = arr(key)
                    if a is not None:
                        cols[f"{p}_{suffix}"] = a
            return cols

        if plot_type == "chisq":
            labels = list(data.get("labels", []))
            scans = data.get("scans")
            chisq = data.get("chisq")
            if scans is not None and chisq is not None:
                scans = np.asarray(scans, dtype=float)
                chisq = np.asarray(chisq, dtype=float)
                for j, lab in enumerate(labels):
                    try:
                        cols[f"{lab}_scan"] = scans[j].ravel()
                        cols[f"{lab}_chisq"] = chisq[j].ravel()
                    except Exception:
                        pass
            return cols

        # Generic fallback: every 1-D numeric array in the npz.
        keys = data.files if hasattr(data, "files") else list(data.keys())
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            try:
                raw = np.asarray(v, dtype=float)
            except (ValueError, TypeError):
                continue
            if raw.ndim == 1 and raw.size:
                cols[k] = raw
        return cols

    def _export_plot_csv(self, npz_path):
        if not os.path.isfile(npz_path):
            QMessageBox.warning(self, "Export CSV",
                                "The plot's data file is missing.")
            return
        try:
            data = np.load(npz_path, allow_pickle=True)
            plot_type = str(data.get("plot_type", ""))
        except Exception:
            QMessageBox.warning(self, "Export CSV",
                                "Could not read the plot's data file.")
            return
        cols = self._plot_csv_columns(plot_type, data)
        if not cols:
            QMessageBox.information(
                self, "Export CSV",
                "This plot has no tabular data to export.")
            return
        dlg = PlotCsvExportDialog(list(cols.keys()), self)
        if not dlg.exec():
            return
        opts = dlg.options()
        chosen = [c for c in cols if c in opts["columns"]]
        if not chosen:
            return

        from gui.shared_widgets import get_last_dir, remember_last_dir
        base = os.path.splitext(os.path.basename(npz_path))[0]
        last = get_last_dir("data", "save")
        default_path = (os.path.join(last, f"{base}.csv")
                        if last else f"{base}.csv")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export plot data as CSV", default_path,
            "CSV files (*.csv);;All files (*)")
        if not save_path:
            return
        remember_last_dir("data", "save", save_path)

        import csv
        prec = opts["precision"]
        nrows = max(len(cols[c]) for c in chosen)

        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            return f"{v:.{prec}f}"

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=opts["delimiter"])
                if opts["header"]:
                    w.writerow(chosen)
                for i in range(nrows):
                    row = []
                    for c in chosen:
                        a = cols[c]
                        row.append(fmt(a[i]) if i < len(a) else "")
                    w.writerow(row)
        except Exception as exc:
            QMessageBox.critical(self, "Export CSV",
                                 f"Could not write the CSV:\n{exc}")
            return
        QMessageBox.information(self, "Export CSV",
                                f"Exported {len(chosen)} column(s), "
                                f"{nrows} row(s) to:\n{save_path}")

    def _delete_iteration(self, project, iteration, iter_dir,
                          confirm=True):
        if confirm:
            msg = (f"Delete iteration <b>{iteration}</b> of "
                   f"<b>{project}</b>?<br><br>"
                   f"This removes the directory from disk:<br>"
                   f"<tt>{iter_dir}</tt>")
            reply = QMessageBox.question(
                self, "Delete iteration", msg,
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        # Disk first; if that fails the in-memory tree stays so the user
        # can retry once the lock / permission is resolved.
        if iter_dir and os.path.isdir(iter_dir):
            try:
                import shutil
                shutil.rmtree(iter_dir)
            except Exception as e:
                QMessageBox.critical(
                    self, "Delete failed",
                    f"Could not remove iteration directory:\n{e}")
                return
        # Clean up the iteration's notes file. Normally it lives inside
        # iter_dir and is gone with the rmtree above, but remove it
        # explicitly so a stray note (e.g. iter_dir missing on disk)
        # never outlives the iteration it belongs to.
        self._delete_iteration_notes(iter_dir)
        # In-memory cleanup
        proj = self._results_data.get(project)
        if proj is not None and iteration in proj:
            del proj[iteration]
            if not proj:
                # Iteration list empty → also drop the project entry so
                # the tree doesn't show an empty header.
                del self._results_data[project]
        # Editor / preview state may still point at a file we just
        # nuked; clear it so the next click doesn't try to read it.
        self._current_item_data = None
        self._notes_path = None
        self._notes_dirty = False
        self._rebuild_tree()

    def _delete_project(self, project, project_dir, confirm=True):
        if confirm:
            n_iters = len(self._results_data.get(project, {}))
            msg = (f"Delete project <b>{project}</b> "
                   f"and all {n_iters} iteration(s)?<br><br>"
                   f"This removes the directory from disk:<br>"
                   f"<tt>{project_dir}</tt>")
            reply = QMessageBox.question(
                self, "Delete project", msg,
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        # Capture the iteration dirs before we drop the in-memory entry
        # so we can scrub their notes too even if the rmtree is skipped.
        iter_dirs = [iv.get("directory", "") for iv
                     in self._results_data.get(project, {}).values()]
        if project_dir and os.path.isdir(project_dir):
            try:
                import shutil
                shutil.rmtree(project_dir)
            except Exception as e:
                QMessageBox.critical(
                    self, "Delete failed",
                    f"Could not remove project directory:\n{e}")
                return
        # Cascade notes cleanup: the project note plus every iteration
        # note (in case any iter_dir lived outside project_dir or the
        # rmtree was skipped because the dir was already gone).
        self._delete_project_notes(project_dir)
        for iter_dir in iter_dirs:
            self._delete_iteration_notes(iter_dir)
        if project in self._results_data:
            del self._results_data[project]
        self._current_item_data = None
        self._notes_path = None
        self._notes_dirty = False
        self._rebuild_tree()

    # ── Notes cascade cleanup (Part A: orphaned-notes fix) ──

    def _delete_iteration_notes(self, iter_dir):
        """Remove the iteration-level notes file for `iter_dir`."""
        if not iter_dir:
            return
        self._remove_file_quietly(
            os.path.join(iter_dir, "_iter_notes.txt"))

    def _delete_project_notes(self, project_dir):
        """Remove the project-level notes file for `project_dir`."""
        if not project_dir:
            return
        self._remove_file_quietly(
            os.path.join(project_dir, "_project_notes.txt"))

    @staticmethod
    def _remove_file_quietly(path):
        """os.remove that ignores a missing file or permission error."""
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass

    # ── Notes (project / iteration) ──

    def _load_notes(self, path, label=""):
        """Load notes from `path` into the editor and switch to the notes
        page. Suppresses the autosave timer while populating.
        """
        self._notes_path = path
        self._notes_dirty = False
        text = ""
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = ""
                _log.warning("Could not read notes file %s; showing an empty "
                             "notes pane (existing notes are not overwritten "
                             "unless you edit and save)", path, exc_info=True)
        # block textChanged so loading doesn't mark as dirty
        self._notes_editor.blockSignals(True)
        self._notes_editor.setPlainText(text)
        self._notes_editor.blockSignals(False)
        if label:
            self._title_label.setText(f"Notes – {label}")
        if path:
            self._notes_status.setText(f"Saved → {path}")
        else:
            self._notes_status.setText("")
        self._viewer_stack.setCurrentIndex(4)

    def _on_notes_text_changed(self):
        if self._notes_path is None:
            return
        self._notes_dirty = True
        self._notes_status.setText("Editing… (autosaves)")
        self._notes_save_timer.start()

    def _save_notes(self, force=False):
        if self._notes_path is None:
            return
        if not self._notes_dirty and not force:
            return
        text = self._notes_editor.toPlainText()
        # Avoid writing an empty file when the path doesn't exist yet and
        # the user hasn't typed anything.
        if not text and not os.path.isfile(self._notes_path):
            self._notes_dirty = False
            return
        try:
            os.makedirs(os.path.dirname(self._notes_path), exist_ok=True)
            with open(self._notes_path, "w", encoding="utf-8") as f:
                f.write(text)
            self._notes_dirty = False
            self._notes_status.setText(f"Saved → {self._notes_path}")
        except Exception as e:
            self._notes_status.setText(f"Save failed: {e}")

    def _show_info(self):
        text = (
            "<h3>Results Viewer</h3>"
            "Browse and inspect all output produced by the Analysis tab. "
            "Results are organised by project and iteration in the tree "
            "on the left.<br><br>"

            "<h4>Tree Structure</h4>"
            "\u2022 <b>Project</b> \u2192 <b>Iteration</b> \u2192 individual "
            "output files.<br>"
            "\u2022 Items are grouped by category: Fit Report, Parameters, "
            "Metadata, fit plots, tracker plots/CSV, diagnostic plots.<br><br>"

            "<h4>Viewer Panels</h4>"
            "\u2022 <b>Text</b> \u2013 Fit reports and other text files are "
            "displayed in a monospace viewer.<br>"
            "\u2022 <b>Table</b> \u2013 CSV files (parameters, metadata, "
            "tracker) are shown in an interactive table with columns sized "
            "to content.<br>"
            "\u2022 <b>Plot</b> \u2013 Fit plots, tracker plots, and "
            "diagnostic plots (chi-square map, correlation, walk plot) are "
            "rendered as interactive matplotlib figures with a navigation "
            "toolbar (zoom, pan, save).<br><br>"

            "<h4>Edit Plot</h4>"
            "Opens the Plot Editor for the current plot, where you can:<br>"
            "\u2022 Adjust axis limits, titles, label positions, and tick "
            "sizes.<br>"
            "\u2022 Change artist properties: colour, line width, line style, "
            "marker, marker size, alpha, and visibility.<br>"
            "\u2022 Supports all artist types: lines, error bars, scatter "
            "points, fills, vertical/horizontal lines, bar charts.<br>"
            "\u2022 Edits are saved as <tt>.style.json</tt> alongside the "
            "plot data and persist across sessions.<br>"
            "\u2022 Use <i>Overwrite Original</i> to update the PNG on disk, "
            "or <i>Save As</i> to export as PNG, PDF, or SVG.<br><br>"

            "<h4>Export</h4>"
            "Saves the currently viewed item:<br>"
            "\u2022 Plots \u2013 opens a save dialog (PNG / PDF / SVG).<br>"
            "\u2022 Text / CSV \u2013 copies the file to a chosen "
            "location.<br><br>"

            "<h4>Refresh</h4>"
            "• <b>Refresh All</b> – re-scan the output directory for every "
            "project on disk (picks up results from other sessions or added "
            "manually).<br>"
            "• <b>Refresh Project</b> – re-scan only the project active in "
            "the Analysis tab."
        )
        from gui.analysis.helpers import _show_scrollable_info
        _show_scrollable_info(self, "Results Tab \u2013 Information", text)

    def restore_iterations(self, mapping):
        """Register saved-session iterations without a whole-dir scan.

        ``mapping`` is ``{project_name: [iter names]}`` as recorded in a
        save file. Only those directories are stat-checked and added —
        unrelated projects/iterations in the output directory stay out
        of the tree (that is what the explicit *Refresh All* button is
        for). Returns the list of ``project/iter`` entries whose folder
        no longer exists so the caller can tell the user.
        """
        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()
        missing = []
        for project_name, iter_names in (mapping or {}).items():
            project_dir = os.path.join(base_dir, str(project_name))
            for iter_name in iter_names:
                iter_dir = os.path.join(project_dir, str(iter_name))
                if not os.path.isdir(iter_dir):
                    missing.append(f"{project_name}/{iter_name}")
                    continue
                self._results_data.setdefault(str(project_name), {})[
                    str(iter_name)] = {
                        "results": [],
                        "output_config": {},
                        "directory": iter_dir,
                }
        self._rebuild_tree()
        return missing

    def _refresh_from_disk(self, project_filter=None):
        """Scan the analysis output directory for results.

        Parameters
        ----------
        project_filter : str or None
            When None (the default, hooked up to the *Refresh All*
            button) every project subdirectory is loaded. When set,
            only that single project name is scanned -- used by
            *Refresh Project* so a user with many on-disk projects
            doesn't have to wait for every one to be re-walked.
        """
        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()

        if not os.path.isdir(base_dir):
            QMessageBox.information(
                self, "Refresh",
                f"Output directory not found:\n{base_dir}")
            return

        if project_filter is not None:
            project_dir = os.path.join(base_dir, project_filter)
            if not os.path.isdir(project_dir):
                QMessageBox.information(
                    self, "Refresh",
                    f"No results directory for project "
                    f"'{project_filter}' under:\n{base_dir}")
                return
            project_names = [project_filter]
        else:
            project_names = os.listdir(base_dir)

        for project_name in project_names:
            project_dir = os.path.join(base_dir, project_name)
            if not os.path.isdir(project_dir):
                continue
            if project_name not in self._results_data:
                self._results_data[project_name] = {}

            for iter_name in os.listdir(project_dir):
                iter_dir = os.path.join(project_dir, iter_name)
                if not os.path.isdir(iter_dir):
                    continue
                if iter_name not in self._results_data[project_name]:
                    self._results_data[project_name][iter_name] = {
                        "results": [],
                        "output_config": {},
                        "directory": iter_dir,
                    }

        self._rebuild_tree()

    def _refresh_active_project(self):
        """Refresh only the project currently active in the Analysis
        tab. Falls back to a friendly message if no project is open."""
        main = self.window()
        analysis_tab = getattr(main, "analysis_tab", None)
        if analysis_tab is None:
            QMessageBox.information(
                self, "Refresh",
                "No Analysis tab is open in this window.")
            return
        project = analysis_tab._project_tabs.currentWidget()
        if project is None:
            QMessageBox.information(
                self, "Refresh",
                "No analysis project is currently active. Open or "
                "create one in the Analysis tab first.")
            return
        name = getattr(project, "_project_name", None) or \
            getattr(project, "project_name", None)
        if not name:
            QMessageBox.information(
                self, "Refresh",
                "Active analysis project has no name yet -- save the "
                "project first.")
            return
        self._refresh_from_disk(project_filter=name)
