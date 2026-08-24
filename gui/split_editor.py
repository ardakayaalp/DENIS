"""Standalone editor for creating ``.vasdf`` virtual-split sidecars.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Invoked from ``Edit > Split File…`` in the main menu. A standalone tool
for turning a parent ASDF run into one or more virtual sub-runs the user
can then load in either the Pre-Analysis or Analysis tab; it needs no
open project. The editor reads the parent ASDF's raw scanning voltage
events, shows them as a histogram, lets the user place a draggable cut
marker, edits per-side metadata (Source ID, cooler V, laser setpoint,
free-form comment), and writes one ``.vasdf`` file per side next to the
parent ASDF. The parent ASDF is never modified.

Depends on: gui.analysis.vasdf (descriptor read/write), gui.shared_widgets
(settings + last-dir helpers); reads ASDF events via clstools and draws
with matplotlib.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import asdf

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QLocale
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QDoubleSpinBox, QPushButton, QFileDialog,
    QMessageBox, QDialogButtonBox, QWidget,
)

try:
    import clstools as cls
    _HAS_CLSTOOLS = True
except ImportError:
    _HAS_CLSTOOLS = False

from gui.analysis.vasdf import (
    VASDF_EXT, write_vasdf, read_vasdf, is_vasdf_path,
    default_vasdf_paths,
)
from gui.shared_widgets import _load_settings, _save_settings


def _make_double(value=0.0, lo=-1e8, hi=1e8, decimals=3, step=0.1,
                 suffix=""):
    sb = QDoubleSpinBox()
    sb.setLocale(QLocale(QLocale.Language.English,
                         QLocale.Country.UnitedStates))
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    return sb


class SplitFileEditor(QDialog):
    """Standalone window for creating virtual-split sidecar files.

    Workflow: open ASDF → place cut → edit per-side metadata →
    save ``.vasdf`` files. Optionally open an existing ``.vasdf``
    to re-edit it (loads its parent ASDF and pre-fills the cut and
    metadata fields).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Split File — Virtual Splits")
        self.resize(820, 700)

        self._asdf_path: Optional[str] = None
        self._np_dv: Optional[np.ndarray] = None
        self._parent_cooler_v = 0.0
        self._parent_laser_sp = 0.0
        self._run_number = ""
        self._dv_min = 0.0
        self._dv_max = 1.0
        self._cut_value = 0.5
        self._dragging = False
        # If the user opened an existing .vasdf — and we find its
        # sibling next to it — we remember both paths so "Save"
        # overwrites them in place rather than writing new files.
        # ``[left_path, right_path]``; entries default to None.
        self._editing_vasdf_paths: list = [None, None]

        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        # ── File-picker row ──────────────────────────────────────
        pick_row = QHBoxLayout()
        self._open_asdf_btn = QPushButton("Open ASDF…")
        self._open_asdf_btn.clicked.connect(self._on_open_asdf)
        pick_row.addWidget(self._open_asdf_btn)
        self._open_vasdf_btn = QPushButton("Open existing .vasdf…")
        self._open_vasdf_btn.setToolTip(
            "Load an existing .vasdf to re-edit its cut and metadata. "
            "The parent ASDF is loaded automatically.")
        self._open_vasdf_btn.clicked.connect(self._on_open_vasdf)
        pick_row.addWidget(self._open_vasdf_btn)
        pick_row.addStretch()
        self._file_label = QLabel(
            "<span style='color:#888'>No file loaded.</span>")
        pick_row.addWidget(self._file_label)
        main.addLayout(pick_row)

        # ── Cut spinbox row ──────────────────────────────────────
        cut_row = QHBoxLayout()
        cut_row.addWidget(QLabel("Cut at raw voltage:"))
        self._cut_spin = _make_double(0.0, -1e6, 1e6, 3, 0.1, " V")
        self._cut_spin.valueChanged.connect(self._on_cut_changed)
        self._cut_spin.setEnabled(False)
        cut_row.addWidget(self._cut_spin)
        cut_row.addStretch()
        main.addLayout(cut_row)

        # ── Spectrum plot (np_dv histogram + draggable cut) ──────
        self._fig = Figure(figsize=(7, 3), dpi=100)
        self._fig.subplots_adjust(
            left=0.08, right=0.99, top=0.95, bottom=0.18)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("Raw scanning voltage (V)", fontsize=9)
        self._ax.set_ylabel("Counts", fontsize=9)
        self._ax.tick_params(labelsize=8)
        self._ax.grid(True, alpha=0.3)
        self._canvas = FigureCanvasQTAgg(self._fig)
        main.addWidget(self._canvas, 2)

        # Placeholders — replaced once a file is loaded.
        self._fill_left = None
        self._fill_right = None
        self._cut_line = None

        self._canvas.mpl_connect(
            "button_press_event", self._on_press)
        self._canvas.mpl_connect(
            "button_release_event", self._on_release)
        self._canvas.mpl_connect(
            "motion_notify_event", self._on_motion)

        # ── Per-side metadata forms ─────────────────────────────
        sides = QHBoxLayout()
        sides.setSpacing(8)
        self._left = self._build_side_form("Left side  →  _A")
        self._right = self._build_side_form("Right side  →  _B")
        sides.addWidget(self._left["group"], 1)
        sides.addWidget(self._right["group"], 1)
        main.addLayout(sides)

        # ── Counts label ────────────────────────────────────────
        self._counts_label = QLabel("")
        self._counts_label.setStyleSheet("color: #888; padding: 2px;")
        main.addWidget(self._counts_label)

        # ── Buttons ─────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close)
        self._save_btn = btns.addButton(
            "Save Split Files…",
            QDialogButtonBox.ButtonRole.AcceptRole)
        self._save_btn.setEnabled(False)
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

        if not _HAS_CLSTOOLS:
            QMessageBox.warning(
                self, "Missing dependency",
                "clstools is not installed — the editor will only be "
                "able to read .vasdf descriptors, not the parent ASDF "
                "events needed for the spectrum view.")

    # ── Side form helper ─────────────────────────────────────────

    def _build_side_form(self, title):
        grp = QGroupBox(title)
        form = QFormLayout(grp)
        form.setContentsMargins(8, 8, 8, 8)
        form.setVerticalSpacing(4)
        sid = QLineEdit("")
        sid.setToolTip(
            "Source ID — appears as the trailing token of the source "
            "name (Run_<this>) and in the .vasdf filename. No 'run_' "
            "prefix; the convention is <run>_<suffix> e.g. 7509_A.")
        form.addRow("Source ID:", sid)
        cooler_spin = _make_double(0.0, -1e8, 1e8, 3, 1.0, " V")
        form.addRow("Cooler:", cooler_spin)
        laser_spin = _make_double(0.0, 0.0, 1e8, 6, 0.0001, " cm⁻¹")
        form.addRow("Laser:", laser_spin)
        comment = QLineEdit("")
        comment.setPlaceholderText("optional, e.g. isotope label")
        form.addRow("Comment:", comment)
        return {"group": grp, "sid": sid, "cooler": cooler_spin,
                "laser": laser_spin, "comment": comment}

    # ── File loading ────────────────────────────────────────────

    def _on_open_asdf(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ASDF File", get_last_dir("data", "load"),
            "ASDF files (*.asdf);;All files (*)")
        if not path:
            return
        remember_last_dir("data", "load", path)
        self._editing_vasdf_paths = [None, None]
        self._load_asdf_into_editor(path)

    def _on_open_vasdf(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Open .vasdf File", get_last_dir("data", "load"),
            "Virtual splits (*.vasdf);;All files (*)")
        if not path:
            return
        remember_last_dir("data", "load", path)
        try:
            desc = read_vasdf(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Bad .vasdf",
                f"Failed to read descriptor:\n{exc}")
            return
        parent_path = desc["parent_path"]
        if not os.path.isfile(parent_path):
            QMessageBox.critical(
                self, "Parent ASDF missing",
                f"Parent ASDF not found:\n{parent_path}\n\n"
                "Move the .vasdf or restore the parent and try again.")
            return
        if not self._load_asdf_into_editor(parent_path):
            return

        # Look for a sibling .vasdf next to `path` that points at the
        # same parent and adjoins this one — that's the other half
        # of the original two-way split. If we find exactly one, both
        # sides get pre-filled and Save will overwrite both files in
        # place. If we find none (or multiple ambiguous), fall back
        # to the single-side pre-fill.
        sibling_path, sibling_desc = self._find_sibling_vasdf(path, desc)

        eps = 1e-9
        if sibling_desc is not None:
            this_lo = desc["split"]["lo"]
            this_hi = desc["split"]["hi"]
            sib_lo = sibling_desc["split"]["lo"]
            sib_hi = sibling_desc["split"]["hi"]
            this_is_left = (this_hi <= sib_lo + eps)
            cut = this_hi if this_is_left else sib_hi
            left_desc = desc if this_is_left else sibling_desc
            right_desc = sibling_desc if this_is_left else desc
            left_path = path if this_is_left else sibling_path
            right_path = sibling_path if this_is_left else path

            self._cut_spin.blockSignals(True)
            self._cut_spin.setValue(cut)
            self._cut_spin.blockSignals(False)
            self._cut_value = cut
            self._update_shading()
            self._update_counts()
            self._fill_side(self._left, left_desc)
            self._fill_side(self._right, right_desc)
            self._editing_vasdf_paths = [left_path, right_path]
            return

        # Single-sided pre-fill: place cut at the saved range's hi
        # (so the saved side becomes "left"); leave the other side
        # at parent defaults so the user can re-split.
        lo = desc["split"]["lo"]
        hi = desc["split"]["hi"]
        if lo <= self._dv_min + eps and hi >= self._dv_max - eps:
            cut = 0.5 * (self._dv_min + self._dv_max)
        else:
            cut = hi
        self._cut_spin.blockSignals(True)
        self._cut_spin.setValue(cut)
        self._cut_spin.blockSignals(False)
        self._cut_value = cut
        self._update_shading()
        self._update_counts()
        side = self._left if hi <= cut + eps else self._right
        self._fill_side(side, desc)
        # Remember which slot to overwrite on Save.
        if hi <= cut + eps:
            self._editing_vasdf_paths = [path, None]
        else:
            self._editing_vasdf_paths = [None, path]

    def _find_sibling_vasdf(self, this_path, this_desc):
        """Look for another .vasdf next to ``this_path`` that points
        at the same parent ASDF and adjoins ``this_desc``'s range.

        Returns ``(sibling_path, sibling_desc)`` or ``(None, None)``
        if zero or multiple candidates are found.
        """
        try:
            siblings = [
                os.path.join(os.path.dirname(this_path), name)
                for name in os.listdir(os.path.dirname(this_path) or ".")
                if name.lower().endswith(".vasdf")
            ]
        except OSError:
            return None, None
        candidates = []
        eps = 1e-6
        for sp in siblings:
            if os.path.normcase(os.path.abspath(sp)) == \
                    os.path.normcase(os.path.abspath(this_path)):
                continue
            try:
                d = read_vasdf(sp)
            except Exception:
                continue
            if os.path.normcase(os.path.abspath(d["parent_path"])) != \
                    os.path.normcase(os.path.abspath(
                        this_desc["parent_path"])):
                continue
            this_hi = float(this_desc["split"]["hi"])
            this_lo = float(this_desc["split"]["lo"])
            sib_hi = float(d["split"]["hi"])
            sib_lo = float(d["split"]["lo"])
            adjoins = (abs(this_hi - sib_lo) < eps
                       or abs(sib_hi - this_lo) < eps)
            if adjoins:
                candidates.append((sp, d))
        if len(candidates) == 1:
            return candidates[0]
        return None, None

    def _fill_side(self, side, desc):
        """Populate one side form from a vasdf descriptor."""
        side["sid"].setText(desc.get("source_id", ""))
        md = desc.get("metadata_override", {}) or {}
        if "cooler_v" in md:
            side["cooler"].setValue(float(md["cooler_v"]))
        if "laser_sp" in md:
            side["laser"].setValue(float(md["laser_sp"]))
        if md.get("comment"):
            side["comment"].setText(str(md["comment"]))

    def _load_asdf_into_editor(self, asdf_path: str) -> bool:
        """Read parent ASDF metadata + np_dv events; populate UI.

        Returns True on success; on failure shows a warning and
        returns False without mutating editor state.
        """
        if not _HAS_CLSTOOLS:
            QMessageBox.critical(
                self, "Missing dependency",
                "clstools is required to read ASDF event data.")
            return False

        # 1. Quick metadata read via asdf — cheap, gives run_number etc.
        run_number = ""
        cooler_v = 0.0
        laser_sp = 0.0
        try:
            with asdf.open(asdf_path) as af:
                run_number = str(af.tree.get("Run", ""))
                cooler_v = float(af.tree.get("CoolerVoltage", 0)) * 10000.0
                laser_sp = float(af.tree.get("LaserSetpoint", 0))
        except Exception as exc:
            QMessageBox.critical(
                self, "ASDF read error",
                f"Could not read ASDF metadata:\n{exc}")
            return False

        # 2. Full event load via clstools for the np_dv array.
        try:
            from gui.calibration import get_registry as _get_cal_registry
            from gui.calibration import load_run_calibrated
            data = cls.CLSDataFrame()
            # Apply the run's calibration before Compute_Voltages, so the
            # gate the user drags here is in the same volts the fit will use.
            load_run_calibrated(data, asdf_path,
                                _get_cal_registry().to_dict())
            data.Compute_Voltages()
            df = data.Sorted
            if hasattr(df, "compute"):
                df = df.compute()
        except Exception as exc:
            QMessageBox.critical(
                self, "ASDF processing error",
                f"Could not process ASDF events:\n{exc}")
            return False

        np_dv = None
        for col in ("DV", "dv", "DAC"):
            if col in df.columns:
                np_dv = df[col].to_numpy()
                break
        if np_dv is None or len(np_dv) == 0:
            QMessageBox.critical(
                self, "No event data",
                "ASDF file has no DV column — cannot draw spectrum.")
            return False

        np_dv = np.asarray(np_dv, dtype=float)
        self._asdf_path = asdf_path
        self._np_dv = np_dv
        self._run_number = run_number or os.path.splitext(
            os.path.basename(asdf_path))[0].split("_")[-1]
        self._parent_cooler_v = cooler_v
        self._parent_laser_sp = laser_sp
        self._dv_min = float(np_dv.min())
        self._dv_max = float(np_dv.max())
        self._cut_value = 0.5 * (self._dv_min + self._dv_max)

        # Reset UI — fresh defaults + redraw spectrum.
        self._file_label.setText(
            f"<b>{os.path.basename(asdf_path)}</b> "
            f"<span style='color:#888'>({len(np_dv):,} events)</span>")
        self._cut_spin.blockSignals(True)
        self._cut_spin.setRange(self._dv_min, self._dv_max)
        self._cut_spin.setValue(self._cut_value)
        self._cut_spin.setEnabled(True)
        self._cut_spin.blockSignals(False)

        # Default per-side metadata = parent values; suggested IDs.
        rn = self._run_number or "split"
        self._left["sid"].setText(f"{rn}_A")
        self._right["sid"].setText(f"{rn}_B")
        for side in (self._left, self._right):
            side["cooler"].setValue(cooler_v)
            side["laser"].setValue(laser_sp)
            side["comment"].setText("")

        self._draw_spectrum()
        self._save_btn.setEnabled(True)
        return True

    def _draw_spectrum(self):
        self._ax.clear()
        n_bins = min(400, max(50, int(np.sqrt(len(self._np_dv)))))
        counts, edges = np.histogram(self._np_dv, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self._ax.step(
            centers, counts, where="mid",
            color="#1f77b4", lw=1.0)
        self._ax.set_xlabel("Raw scanning voltage (V)", fontsize=9)
        self._ax.set_ylabel("Counts", fontsize=9)
        self._ax.tick_params(labelsize=8)
        self._ax.grid(True, alpha=0.3)
        self._fill_left = self._ax.axvspan(
            self._dv_min, self._cut_value,
            color="#1f77b4", alpha=0.08, linewidth=0)
        self._fill_right = self._ax.axvspan(
            self._cut_value, self._dv_max,
            color="#ff7f0e", alpha=0.08, linewidth=0)
        self._cut_line = self._ax.axvline(
            self._cut_value, color="#d62728", lw=1.6)
        self._canvas.draw_idle()
        self._update_counts()

    def _update_shading(self):
        if self._fill_left is None:
            return
        self._fill_left.remove()
        self._fill_right.remove()
        self._fill_left = self._ax.axvspan(
            self._dv_min, self._cut_value,
            color="#1f77b4", alpha=0.08, linewidth=0)
        self._fill_right = self._ax.axvspan(
            self._cut_value, self._dv_max,
            color="#ff7f0e", alpha=0.08, linewidth=0)
        self._cut_line.set_xdata([self._cut_value, self._cut_value])
        self._canvas.draw_idle()

    def _on_cut_changed(self, value):
        self._cut_value = float(value)
        self._update_shading()
        self._update_counts()

    def _set_cut_from_event(self, event):
        if (event.inaxes is not self._ax
                or event.xdata is None
                or self._np_dv is None):
            return
        x = max(self._dv_min, min(self._dv_max, float(event.xdata)))
        self._cut_spin.blockSignals(True)
        self._cut_spin.setValue(x)
        self._cut_spin.blockSignals(False)
        self._cut_value = x
        self._update_shading()
        self._update_counts()

    def _on_press(self, event):
        if (event.inaxes is self._ax
                and event.button == 1
                and self._np_dv is not None):
            self._dragging = True
            self._set_cut_from_event(event)

    def _on_release(self, event):
        self._dragging = False

    def _on_motion(self, event):
        if self._dragging:
            self._set_cut_from_event(event)

    def _update_counts(self):
        if self._np_dv is None or self._np_dv.size == 0:
            self._counts_label.setText("")
            return
        n_left = int(np.sum(self._np_dv <= self._cut_value))
        n_right = int(np.sum(self._np_dv > self._cut_value))
        total = n_left + n_right
        self._counts_label.setText(
            f"Counts:  Left = {n_left:,}    Right = {n_right:,}    "
            f"(total {total:,})")

    # ── Save ────────────────────────────────────────────────────

    def _on_save(self):
        if self._np_dv is None or not self._asdf_path:
            QMessageBox.information(
                self, "Nothing to save",
                "Open an ASDF file first.")
            return
        sid_a = self._left["sid"].text().strip()
        sid_b = self._right["sid"].text().strip()
        if not sid_a or not sid_b:
            QMessageBox.warning(
                self, "Source ID required",
                "Both sides need a non-empty Source ID.")
            return
        if sid_a == sid_b:
            QMessageBox.warning(
                self, "Source IDs must differ",
                f"Both sides have Source ID '{sid_a}'.")
            return

        # Default save dir = parent's directory; default names use
        # the trailing token of each source_id as the filename suffix
        # (so 7509_A → run_7509_A.vasdf next to run_7509.asdf).
        suffix_a = sid_a.rsplit("_", 1)[-1] if "_" in sid_a else sid_a
        suffix_b = sid_b.rsplit("_", 1)[-1] if "_" in sid_b else sid_b
        defaults = default_vasdf_paths(
            self._asdf_path, [suffix_a, suffix_b])
        # If the user is re-editing existing .vasdf files, default the
        # save targets to those paths so Save overwrites them in place.
        if self._editing_vasdf_paths[0]:
            defaults[0] = self._editing_vasdf_paths[0]
        if self._editing_vasdf_paths[1]:
            defaults[1] = self._editing_vasdf_paths[1]

        from gui.shared_widgets import get_last_dir, remember_last_dir
        # If the user has a previously-remembered save dir, prefer it
        # over the parent-directory default — but only when the dialog's
        # default still points at the parent (i.e. the user hasn't
        # already opened the existing .vasdf for re-editing).
        last_save = get_last_dir("data", "save")
        if last_save and not self._editing_vasdf_paths[0]:
            defaults[0] = os.path.join(last_save, os.path.basename(defaults[0]))
        if last_save and not self._editing_vasdf_paths[1]:
            defaults[1] = os.path.join(last_save, os.path.basename(defaults[1]))
        path_a, _ = QFileDialog.getSaveFileName(
            self, "Save left split as",
            defaults[0],
            f"Virtual splits (*{VASDF_EXT});;All files (*)")
        if not path_a:
            return
        if not path_a.lower().endswith(VASDF_EXT):
            path_a += VASDF_EXT
        remember_last_dir("data", "save", path_a)

        path_b, _ = QFileDialog.getSaveFileName(
            self, "Save right split as",
            defaults[1],
            f"Virtual splits (*{VASDF_EXT});;All files (*)")
        if not path_b:
            return
        if not path_b.lower().endswith(VASDF_EXT):
            path_b += VASDF_EXT
        remember_last_dir("data", "save", path_b)

        try:
            write_vasdf(
                path_a,
                parent_path=self._asdf_path,
                source_id=sid_a, label=sid_a,
                lo=self._dv_min, hi=self._cut_value,
                metadata_override=self._side_md(self._left),
            )
            write_vasdf(
                path_b,
                parent_path=self._asdf_path,
                source_id=sid_b, label=sid_b,
                lo=self._cut_value, hi=self._dv_max,
                metadata_override=self._side_md(self._right),
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Write error",
                f"Failed to write .vasdf files:\n{exc}")
            return

        QMessageBox.information(
            self, "Splits saved",
            "Saved:\n"
            f"  • {os.path.basename(path_a)}\n"
            f"  • {os.path.basename(path_b)}\n\n"
            "Open them from the Pre-Analysis or Analysis tab "
            "through the regular file dialog.")
        self.accept()

    @staticmethod
    def _side_md(side):
        md = {
            "cooler_v": float(side["cooler"].value()),
            "laser_sp": float(side["laser"].value()),
        }
        comment = side["comment"].text().strip()
        if comment:
            md["comment"] = comment
        return md
