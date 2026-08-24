"""Voltage-calibration diagnostics: see the fit, fix it, share the fix.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Two dialogs, reachable from the file row's right-click menu in both
Pre-Analysis and Analysis:

``CalibrationDialog``
    One run. Readback-vs-set with the fitted polynomial on top, residuals
    underneath, and -- the part that actually decides anything -- what the
    choice costs in **MHz**. Volts of residual mean nothing to a
    spectroscopist; a 47 MHz centroid shift against a 300 MHz isotope
    shift means everything. Points are clickable, and the calibration can
    instead be borrowed from another run or typed in by hand.

``CalibrationOverviewDialog``
    Every loaded run at a glance, worst first. This is what makes the
    app's "change nothing by default, but flag it" policy practical: with
    a campaign of hundreds of runs, nobody is going to open a diagnostic
    for each one, so the triage has to come to them.

Both write to the ``gui.calibration`` registry, so a fix made in
Pre-Analysis is the fix the Analysis fit uses, and vice versa.

Depends on: gui.calibration, matplotlib, PySide6.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from gui.calibration import (
    DEFAULT_ORDER,
    DEFAULT_VACC_DIV,
    FLAG_MIN_IMPACT_V,
    MAX_ORDER,
    CalibrationError,
    calibration_shift_v,
    canonical_path,
    describe_spec,
    diagnose,
    fit_calibration,
    get_registry,
    read_beam_header,
    read_cal_points,
    resolve_calibration,
    shift_mhz_over_scan,
    suspect_outliers,
)
from gui.dialog_style import make_plot_card, style_dialog

# Rejection rules, in the order the combo shows them.
_RULES: List[Tuple[str, str]] = [
    ("none", "Keep every point"),
    ("first_n", "Drop the first N points"),
    ("sigma", "Drop residual outliers (n·σ)"),
    ("manual", "Only the points I unticked"),
]


def _fmt_mhz(v: float) -> str:
    return f"{v:+.1f} MHz" if abs(v) >= 0.05 else "negligible"


class CalibrationDialog(QDialog):
    """Diagnose and override one run's voltage calibration.

    ``physics`` (mass_amu / laser_cm / harmonic / cooler_v) is optional: with
    it the dialog can quote the impact in MHz, without it it falls back to
    volts. Callers that have a Source block or the Pre-Analysis physics inputs
    should always pass it -- the MHz number is the whole point.

    ``peers`` is ``[(label, path), ...]`` of the other runs currently loaded,
    offered as borrow donors.
    """

    def __init__(self, path: str, *, run_label: str = "",
                 cal_order: int = DEFAULT_ORDER,
                 physics: Optional[dict] = None,
                 peers: Optional[Sequence[Tuple[str, str]]] = None,
                 parent=None):
        super().__init__(parent)
        self._path = path
        self._physics = physics or {}
        self._peers = [p for p in (peers or []) if canonical_path(p[1])
                       != canonical_path(path)]
        self._base_order = int(cal_order)
        self._reg = get_registry()

        run_label = run_label or os.path.basename(path)
        self.setWindowTitle(f"Voltage calibration — {run_label}")
        self.resize(1020, 860)

        try:
            self._set_v, self._read_raw = read_cal_points(path)
        except Exception as exc:
            self._set_v = np.array([])
            self._read_raw = np.array([])
            QMessageBox.warning(
                self, "Calibration",
                f"Could not read the calibration table from\n{path}\n\n{exc}")

        self._n = int(len(self._set_v))
        # Hand-picked exclusions. Seeded from the saved spec so reopening the
        # dialog shows what is actually in force, not a blank slate.
        self._excluded: set[int] = set()
        self._result = None
        self._building = True

        # The untouched calibration -- what the file gives you with no override
        # at all. Everything the dialog says is measured against this, and it
        # is drawn alongside the corrected one so "did my fix do anything?" is
        # answered by looking rather than by remembering.
        self._base_fit = None
        if self._n:
            try:
                self._base_fit = fit_calibration(
                    self._set_v, self._read_raw, self._base_order, (),
                    DEFAULT_VACC_DIV)
            except CalibrationError:
                pass

        self._build_ui(run_label)
        self._load_saved_spec()
        self._building = False
        self._on_mode_changed()   # settles the enabled states, then recomputes

    # ── UI ───────────────────────────────────────────────────

    def _build_ui(self, run_label: str):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavToolbar)
        from matplotlib.figure import Figure

        style_dialog(self)
        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        hdr = QLabel(
            f"<b>{run_label}</b> &nbsp;·&nbsp; {self._n} calibration points"
            f" &nbsp;·&nbsp; <span style='color:#8a8a90'>{self._path}</span>")
        hdr.setWordWrap(True)
        hdr.setToolTip(
            "The DAC→HV calibration table stored in this run's ASDF: for each\n"
            "voltage the scanner was *told* to produce (Set), the voltage that\n"
            "was actually *measured* (Readback). A polynomial through these\n"
            "pairs is what converts every event's DAC value into a real\n"
            "voltage — and hence into a frequency.")
        outer.addWidget(hdr)

        body = QHBoxLayout()
        body.setSpacing(8)
        outer.addLayout(body, 1)

        # ---- left: plots ----
        # Three panels, not two: the curve, the residuals you *would* have had,
        # and the residuals you have now. The two residual sets need their own
        # y-scales -- a settling glitch puts them ~30x apart, so sharing an axis
        # would flatten the corrected one into the zero line and hide exactly
        # the thing the user is trying to see.
        self._fig = Figure(figsize=(7.0, 7.4))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setToolTip(
            "Top: the measured calibration points and the polynomial through\n"
            "them. Middle and bottom: how far each point sits from that line.\n\n"
            "Points are coloured by the order they were taken, dark → light,\n"
            "so a supply that was still settling at the start of the sweep\n"
            "shows up as dark points off the line.\n\n"
            "Click any point to include or exclude it from the fit.")
        self._ax_fit, self._ax_res0, self._ax_res1 = self._fig.subplots(
            3, 1, sharex=True, height_ratios=[2.2, 1.0, 1.0])
        self._fig.subplots_adjust(hspace=0.12, left=0.16, right=0.97,
                                  top=0.96, bottom=0.08)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        body.addWidget(
            make_plot_card("Calibration fit", self._canvas,
                           NavToolbar(self._canvas, self), parent=self,
                           subtitle="click a point to exclude it"),
            3)

        # ---- right: controls ----
        right = QVBoxLayout()
        right.setSpacing(6)
        body.addLayout(right, 2)

        mode_box = QGroupBox("Where the calibration comes from")
        mode_lay = QVBoxLayout(mode_box)
        mode_lay.setSpacing(4)
        self._mode_group = QButtonGroup(self)

        self._rb_fit = QRadioButton("Fit this run's own points")
        self._rb_fit.setToolTip(
            "Fit a polynomial through this run's calibration table.\n"
            "This is the normal case — use it, and drop any bad points below.")
        self._rb_borrow = QRadioButton("Copy another run's calibration")
        self._rb_borrow.setToolTip(
            "Use a different run's coefficients for this one.\n\n"
            "Useful when this run's calibration table is unusable but a run\n"
            "taken close to it in time is fine — the DAC→HV hardware is the\n"
            "same, so its calibration still describes this beam.")
        self._rb_coeffs = QRadioButton("Type the coefficients in")
        self._rb_coeffs.setToolTip(
            "Enter the polynomial by hand, e.g. from an external calibration.")
        for i, rb in enumerate((self._rb_fit, self._rb_borrow,
                                self._rb_coeffs)):
            self._mode_group.addButton(rb, i)
            mode_lay.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
            if i == 0:
                # --- fit controls, indented under the radio ---
                form = QFormLayout()
                form.setContentsMargins(20, 2, 0, 6)
                self._order = QSpinBox()
                self._order.setRange(1, MAX_ORDER)
                self._order.setValue(self._base_order)
                self._order.setToolTip(
                    "Order of the polynomial fitted through the points.\n\n"
                    "1 — a straight line: a gain and an offset. Almost always\n"
                    "    the right answer; the DAC→HV chain is linear.\n"
                    "2 or 3 — only if the residuals show a real curve. A high\n"
                    "    order will happily bend itself through noise.")
                self._order.valueChanged.connect(self._recompute)
                form.addRow("Polynomial order:", self._order)

                self._rule = QComboBox()
                for key, label in _RULES:
                    self._rule.addItem(label, key)
                self._rule.setToolTip(
                    "Which points to leave OUT of the fit.\n\n"
                    "Keep every point — the default. Nothing is dropped.\n\n"
                    "Drop the first N — matches the physical failure: the HV\n"
                    "    supply is still settling at the start of the sweep, so\n"
                    "    the first few readbacks are simply wrong.\n\n"
                    "Drop residual outliers (n·σ) — a statistical cut. Leave\n"
                    "    'iterative' ticked: a single pass computes σ from data\n"
                    "    that still contains the outliers, so a cluster of bad\n"
                    "    points inflates σ enough to hide inside its own cut.\n\n"
                    "Only the points I unticked — your explicit list. The mode\n"
                    "    switches to this automatically the moment you click a\n"
                    "    point, so what was dropped is never ambiguous later.")
                self._rule.currentIndexChanged.connect(self._on_rule_changed)
                form.addRow("Drop points:", self._rule)

                self._drop_first = QSpinBox()
                self._drop_first.setRange(0, 999)
                self._drop_first.setValue(3)
                self._drop_first.setToolTip(
                    "How many points at the START of the calibration sweep to\n"
                    "discard — these are the dark-coloured ones on the plot.")
                self._drop_first.valueChanged.connect(self._recompute)
                form.addRow("How many:", self._drop_first)

                sig_row = QHBoxLayout()
                sig_row.setContentsMargins(0, 0, 0, 0)
                self._sigma = QDoubleSpinBox()
                self._sigma.setRange(0.5, 10.0)
                self._sigma.setSingleStep(0.5)
                self._sigma.setValue(3.0)
                self._sigma.setToolTip(
                    "Drop any point further than this many standard deviations\n"
                    "from the fitted line. 3 is a sane default; 2 reproduces\n"
                    "the cut clstools applies internally.")
                self._sigma.valueChanged.connect(self._recompute)
                sig_row.addWidget(self._sigma)
                self._iterative = QCheckBox("iterative (recommended)")
                self._iterative.setChecked(True)
                self._iterative.setToolTip(
                    "Re-fit after each cut, and judge the scatter robustly.\n\n"
                    "Leave this ON. Without it the cut is computed once, from\n"
                    "data that still contains the bad points — so a run with\n"
                    "three bad points typically catches only the worst one and\n"
                    "leaves the calibration still wrong.")
                self._iterative.toggled.connect(self._recompute)
                sig_row.addWidget(self._iterative)
                sig_holder = QWidget()
                sig_holder.setLayout(sig_row)
                form.addRow("Cut at:", sig_holder)

                self._ignore_intercept = QCheckBox(
                    "Force zero intercept (p₀ = 0)")
                self._ignore_intercept.setToolTip(
                    "Constrain the calibration offset p0 to exactly 0, so the\n"
                    "line passes through the origin and only the gain (p1, …)\n"
                    "is fitted.\n\n"
                    "Use this when the DAC→HV chain is known to have no DC\n"
                    "offset and you do not want a few volts of apparent offset\n"
                    "absorbing real curvature or a settling glitch. The offset\n"
                    "is dropped from the fit, not fitted and then zeroed, so\n"
                    "the slope is the true through-origin gain.")
                self._ignore_intercept.toggled.connect(self._recompute)
                form.addRow("Constrain:", self._ignore_intercept)

                mode_lay.addLayout(form)

            elif i == 1:
                brow = QHBoxLayout()
                brow.setContentsMargins(20, 2, 0, 6)
                self._donor = QComboBox()
                for label, p in self._peers:
                    self._donor.addItem(label, p)
                if not self._peers:
                    self._donor.addItem("(no other runs loaded)", None)
                self._donor.setToolTip(
                    "The run to take the calibration from. Pick one recorded\n"
                    "close in time — the hardware must not have changed in\n"
                    "between. You are warned if its sweep does not cover the\n"
                    "voltage range this run scans.")
                self._donor.currentIndexChanged.connect(self._recompute)
                brow.addWidget(self._donor, 1)
                browse = QPushButton("Browse…")
                browse.setToolTip("Pick any ASDF run on disk as the donor.")
                browse.clicked.connect(self._browse_donor)
                brow.addWidget(browse)
                mode_lay.addLayout(brow)

            else:
                crow = QHBoxLayout()
                crow.setContentsMargins(20, 2, 0, 6)
                self._coeff_edits = []
                for j in range(MAX_ORDER + 1):
                    e = QLineEdit()
                    e.setPlaceholderText(f"p{j}")
                    e.setToolTip(
                        "Calibrated voltage = p0 + p1·DV + p2·DV² + …\n\n"
                        "p0 is an offset in volts.\n"
                        "p1 is a gain — normally very close to 1.\n"
                        "Leave p2 / p3 empty for a straight line.")
                    e.editingFinished.connect(self._recompute)
                    crow.addWidget(e)
                    self._coeff_edits.append(e)
                mode_lay.addLayout(crow)

        self._rb_fit.setChecked(True)
        right.addWidget(mode_box)

        # ---- stats + impact ----
        result_box = QGroupBox("Resulting calibration")
        result_lay = QVBoxLayout(result_box)
        result_lay.setSpacing(5)

        self._stats = QLabel()
        self._stats.setWordWrap(True)
        self._stats.setTextFormat(Qt.TextFormat.RichText)
        self._stats.setToolTip(
            "The polynomial this dialog will apply, and how tightly the\n"
            "calibration points sit on it.\n\n"
            "σ is the typical distance of a point from the line, in volts.\n"
            "Smaller is better — it means the calibration describes the\n"
            "hardware well.")
        result_lay.addWidget(self._stats)

        self._impact = QLabel()
        self._impact.setWordWrap(True)
        self._impact.setTextFormat(Qt.TextFormat.RichText)
        self._impact.setStyleSheet(
            "padding:7px; border:1px solid #4a4a4f; border-radius:3px;"
            "background-color:#2a2a2e;")
        self._impact.setToolTip(
            "The only number that decides anything.\n\n"
            "Volts of residual mean nothing on their own. This is what your\n"
            "choice does to the measured frequency — set it against the\n"
            "isotope shift you are trying to measure. If it says 'negligible',\n"
            "this run's calibration is not your problem.")
        result_lay.addWidget(self._impact)
        right.addWidget(result_box)

        # ---- point table ----
        self._table = QTableWidget(self._n, 4)
        self._table.setToolTip(
            "Every calibration point. Untick one to leave it out of the fit —\n"
            "the same as clicking it on the plot.\n\n"
            "'Residual' is how far that point sits from the line, in volts.")
        self._table.setAlternatingRowColors(True)
        self._table.setHorizontalHeaderLabels(
            ["Use", "#", "Set [V]", "Residual [V]"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self._checks: List[QCheckBox] = []
        for i in range(self._n):
            cb = QCheckBox()
            cb.setChecked(True)
            cb.toggled.connect(
                lambda _on, idx=i: self._on_check_toggled(idx))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hl.addWidget(cb)
            self._table.setCellWidget(i, 0, holder)
            self._checks.append(cb)
            self._table.setItem(i, 1, QTableWidgetItem(str(i)))
            self._table.setItem(
                i, 2, QTableWidgetItem(f"{self._set_v[i]:.2f}"))
            self._table.setItem(i, 3, QTableWidgetItem(""))
        right.addWidget(self._table, 1)

        # ---- buttons ----
        btns = QDialogButtonBox()
        self._apply_btn = btns.addButton(
            "Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        self._apply_btn.setToolTip(
            "Use this calibration for this run everywhere — the Pre-Analysis\n"
            "spectrum, the Analysis fit, merges and splits. Saved with the\n"
            "project.")
        self._apply_all_btn = btns.addButton(
            "Apply to other runs…", QDialogButtonBox.ButtonRole.ActionRole)
        self._apply_all_btn.setToolTip(
            "Give other loaded runs the same calibration. Useful when a whole\n"
            "shift of runs shares the same settling problem.")
        self._reset_btn = btns.addButton(
            "Reset to file default", QDialogButtonBox.ButtonRole.ResetRole)
        self._reset_btn.setToolTip(
            "Throw away the override and go back to fitting every point in\n"
            "the file, exactly as it was recorded.")
        btns.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._apply_btn.clicked.connect(self._apply)
        self._apply_all_btn.clicked.connect(self._apply_to_others)
        self._reset_btn.clicked.connect(self._reset)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    # ── state <-> UI ─────────────────────────────────────────

    def _load_saved_spec(self):
        """Seed the controls from whatever override is already in force."""
        spec = self._reg.get(self._path)
        if not spec:
            return
        mode = spec.get("mode")
        self._order.setValue(int(spec.get("order", self._base_order)))
        if mode == "borrow":
            self._rb_borrow.setChecked(True)
            idx = self._donor.findData(spec.get("donor"))
            if idx < 0:
                self._donor.addItem(
                    os.path.basename(spec["donor"]), spec["donor"])
                idx = self._donor.count() - 1
            self._donor.setCurrentIndex(idx)
        elif mode == "coeffs":
            self._rb_coeffs.setChecked(True)
            for e, c in zip(self._coeff_edits, spec.get("coeffs_v", [])):
                e.setText(f"{c:.8g}")
        else:
            self._rb_fit.setChecked(True)
            rule = spec.get("reject", "none")
            self._rule.setCurrentIndex(
                max(0, self._rule.findData(rule)))
            self._sigma.setValue(float(spec.get("sigma", 3.0)))
            self._iterative.setChecked(bool(spec.get("iterative", True)))
            self._drop_first.setValue(int(spec.get("drop_first", 0)))
            self._ignore_intercept.setChecked(
                bool(spec.get("ignore_intercept", False)))
            self._excluded = {int(i) for i in (spec.get("excluded") or ())
                              if 0 <= int(i) < self._n}

    def _current_spec(self) -> Optional[dict]:
        if self._rb_borrow.isChecked():
            donor = self._donor.currentData()
            if not donor:
                return None
            return {"mode": "borrow", "donor": donor,
                    "order": self._order.value()}

        if self._rb_coeffs.isChecked():
            coeffs = []
            for e in self._coeff_edits:
                txt = e.text().strip()
                if not txt:
                    break
                try:
                    coeffs.append(float(txt))
                except ValueError:
                    return None
            if len(coeffs) < 2:
                return None
            return {"mode": "coeffs", "coeffs_v": coeffs}

        rule = self._rule.currentData()
        return {
            "mode": "fit",
            "order": self._order.value(),
            "reject": rule,
            "sigma": self._sigma.value(),
            "iterative": self._iterative.isChecked(),
            "drop_first": self._drop_first.value(),
            "ignore_intercept": self._ignore_intercept.isChecked(),
            "excluded": sorted(self._excluded),
        }

    def _on_mode_changed(self):
        # Radio 'toggled' fires while _build_ui is still wiring widgets up,
        # before the later ones exist. The post-build call re-runs this once
        # everything is in place.
        if self._building:
            return
        is_fit = self._rb_fit.isChecked()
        self._order.setEnabled(is_fit or self._rb_borrow.isChecked())
        self._rule.setEnabled(is_fit)
        rule = self._rule.currentData()
        self._drop_first.setEnabled(is_fit and rule == "first_n")
        self._sigma.setEnabled(is_fit and rule == "sigma")
        self._iterative.setEnabled(is_fit and rule == "sigma")
        self._ignore_intercept.setEnabled(is_fit)
        self._donor.setEnabled(self._rb_borrow.isChecked())
        for e in self._coeff_edits:
            e.setEnabled(self._rb_coeffs.isChecked())
        self._table.setEnabled(is_fit)
        if not self._building:
            self._recompute()

    def _on_rule_changed(self):
        rule = self._rule.currentData()
        self._drop_first.setEnabled(rule == "first_n")
        self._sigma.setEnabled(rule == "sigma")
        self._iterative.setEnabled(rule == "sigma")
        if not self._building:
            self._recompute()

    def _materialize_to_manual(self):
        """Freeze the current effective exclusions into an explicit list.

        The moment the user touches a point by hand, the rule becomes their
        list. Anything else is ambiguous -- you cannot half-apply a σ cut and
        half-hand-pick and still say, six months later, which points were
        dropped and why.
        """
        if self._result is not None:
            self._excluded = set(self._result.excluded)
        self._rule.blockSignals(True)
        self._rule.setCurrentIndex(self._rule.findData("manual"))
        self._rule.blockSignals(False)
        self._on_rule_changed()

    def _toggle_point(self, idx: int):
        if not self._rb_fit.isChecked() or not (0 <= idx < self._n):
            return
        if self._rule.currentData() != "manual":
            self._materialize_to_manual()
        if idx in self._excluded:
            self._excluded.discard(idx)
        else:
            self._excluded.add(idx)
        self._recompute()

    def _on_pick(self, event):
        ind = getattr(event, "ind", None)
        if ind is None or not len(ind):
            return
        self._toggle_point(int(ind[0]))

    def _on_check_toggled(self, idx: int):
        if self._building:
            return
        want_used = self._checks[idx].isChecked()
        is_used = self._result is None or idx not in self._result.excluded
        if want_used != is_used:
            self._toggle_point(idx)

    def _browse_donor(self):
        start = os.path.dirname(self._path)
        p, _ = QFileDialog.getOpenFileName(
            self, "Borrow calibration from…", start, "ASDF runs (*.asdf)")
        if not p:
            return
        if self._donor.findData(p) < 0:
            self._donor.addItem(os.path.basename(p), p)
        self._donor.setCurrentIndex(self._donor.findData(p))
        self._rb_borrow.setChecked(True)

    # ── recompute + draw ─────────────────────────────────────

    def _recompute(self):
        if self._building or self._n == 0:
            return
        spec = self._current_spec()
        cal_map = dict(self._reg.to_dict())
        key = canonical_path(self._path)
        if spec is None:
            cal_map.pop(key, None)
        else:
            cal_map[key] = spec

        try:
            self._result = resolve_calibration(
                self._path, cal_map, cal_order=self._order.value())
            self._error = ""
        except CalibrationError as exc:
            self._result = None
            self._error = str(exc)

        self._sync_table()
        self._draw()
        self._update_stats()
        self._apply_btn.setEnabled(self._result is not None)

    def _sync_table(self):
        excl = set(self._result.excluded) if self._result else set()
        resid = (self._result.fit.residuals_v
                 if self._result and self._result.fit is not None else None)
        for i in range(self._n):
            cb = self._checks[i]
            cb.blockSignals(True)
            cb.setChecked(i not in excl)
            cb.blockSignals(False)
            item = self._table.item(i, 3)
            if resid is not None and i < len(resid):
                item.setText(f"{resid[i]:+.4f}")
            else:
                item.setText("—")

    def _draw(self):
        for ax in (self._ax_fit, self._ax_res0, self._ax_res1):
            ax.clear()

        if self._result is None or self._result.fit is None:
            self._ax_fit.text(0.5, 0.5, self._error or "no calibration table",
                              ha="center", va="center", color="#c33",
                              transform=self._ax_fit.transAxes, wrap=True)
            self._canvas.draw_idle()
            return

        fit = self._result.fit
        used = fit.used_mask
        x, y, r = fit.set_v, fit.read_v, fit.residuals_v
        base = self._base_fit
        # Colour by acquisition order so a start-of-sweep failure -- the whole
        # reason this dialog exists -- is visible at a glance.
        order_c = np.arange(self._n)
        xs = np.linspace(float(x.min()), float(x.max()), 200)

        # ── 1. the calibration curves, before and after ──
        self._ax_fit.scatter(x[used], y[used], c=order_c[used],
                             cmap="viridis", s=38, zorder=3, picker=6,
                             label="used")
        if (~used).any():
            self._ax_fit.scatter(x[~used], y[~used], marker="x", s=70,
                                 c="#e53935", zorder=4, picker=6,
                                 label="excluded")
        # The two curves are genuinely close on an absolute scale -- a 1.3 V
        # disagreement across a 400 V sweep is a third of a percent, which is
        # invisible here and worth 50 MHz downstream. So label them with their
        # coefficients: that keeps them distinguishable when they overlap, and
        # the residual panels below are the magnified view.
        if base is not None:
            self._ax_fit.plot(
                xs, base.predict_v(xs), "--", color="#9e9e9e", lw=1.3,
                zorder=2,
                label=(f"file's own: p0={base.coeffs_v[0]:+.3f} V, "
                       f"p1={base.coeffs_v[1]:.5f}"
                       if len(base.coeffs_v) > 1 else "file's own"))
        res = self._result
        self._ax_fit.plot(
            xs, res.predict_v(xs), "-", color="#1e88e5", lw=1.6, zorder=3,
            label=(f"yours: p0={res.coeffs_v[0]:+.3f} V, "
                   f"p1={res.coeffs_v[1]:.5f}"
                   if len(res.coeffs_v) > 1 else "yours"))
        self._ax_fit.set_ylabel("Readback [V]", fontsize=9)
        self._ax_fit.tick_params(labelsize=8, labelbottom=False)
        self._ax_fit.legend(fontsize=8, loc="best")
        self._ax_fit.grid(alpha=0.25)

        # ── 2. how far each point sits from the FILE's own calibration ──
        self._ax_res0.axhline(0, color="#888", lw=0.9, zorder=1)
        if base is not None:
            rb = base.residuals_v
            sb = base.sigma_v
            if sb > 0:
                self._ax_res0.axhspan(-2 * sb, 2 * sb, color="#9e9e9e",
                                      alpha=0.15, zorder=0)
            self._ax_res0.scatter(x, rb, c=order_c, cmap="viridis", s=30,
                                  zorder=3, picker=6)
            self._ax_res0.set_ylabel("Distance from\nfile's own line [V]",
                                     fontsize=8)
            self._ax_res0.text(
                0.985, 0.93, f"typical scatter σ = {sb:.4f} V", ha="right",
                va="top", fontsize=8, color="#616161",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85),
                transform=self._ax_res0.transAxes)
        else:
            self._ax_res0.text(0.5, 0.5, "no baseline fit", ha="center",
                               va="center", fontsize=8, color="#888",
                               transform=self._ax_res0.transAxes)
        self._ax_res0.tick_params(labelsize=8, labelbottom=False)
        self._ax_res0.grid(alpha=0.25)

        # ── 3. how far each point sits from YOUR calibration ──
        # Same points, measured against the line you are about to apply. With
        # no override the two panels are identical, which is confusing to look
        # at unless it is said out loud -- so say it.
        self._ax_res1.axhline(0, color="#888", lw=0.9, zorder=1)
        s = fit.sigma_v
        if s > 0:
            self._ax_res1.axhspan(-2 * s, 2 * s, color="#1e88e5", alpha=0.12,
                                  zorder=0)
        self._ax_res1.scatter(x[used], r[used], c=order_c[used],
                              cmap="viridis", s=30, zorder=3, picker=6)
        if (~used).any():
            self._ax_res1.scatter(x[~used], r[~used], marker="x", s=64,
                                  c="#e53935", zorder=4, picker=6)
        self._ax_res1.text(
            0.985, 0.93, f"typical scatter σ = {s:.4f} V", ha="right",
            va="top", fontsize=8, color="#1565c0",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85),
            transform=self._ax_res1.transAxes)

        unchanged = (base is not None
                     and not res.excluded
                     and np.allclose(res.coeffs_v, base.coeffs_v,
                                     rtol=1e-12, atol=0.0))
        if unchanged:
            self._ax_res1.text(
                0.5, 0.06,
                "same as above — you haven't changed anything yet",
                ha="center", va="bottom", fontsize=8, color="#8a8a90",
                style="italic", transform=self._ax_res1.transAxes)

        self._ax_res1.set_xlabel("Voltage the scanner was set to [V]",
                                 fontsize=9)
        self._ax_res1.set_ylabel("Distance from\nyour line [V]", fontsize=8)
        self._ax_res1.tick_params(labelsize=8)
        self._ax_res1.grid(alpha=0.25)

        self._canvas.draw_idle()

    def _update_stats(self):
        if self._result is None:
            self._stats.setText(
                f"<span style='color:#c33'>{self._error}</span>")
            self._impact.setText("")
            return

        res = self._result
        fit = res.fit
        coef = " &nbsp; ".join(
            f"p{i} = {c:.6g}" for i, c in enumerate(res.coeffs_v))
        lines = [
            "<span style='color:#9a9aa0'>calibrated voltage "
            "= p0 + p1·DV + …</span>",
            f"<b>{coef}</b>",
        ]
        if fit is not None:
            if res.mode in ("borrow", "coeffs"):
                # Nothing was fitted to this run's points -- the coefficients
                # came from elsewhere. What the residuals show here is whether
                # they *describe* this run, which is a different question, and
                # saying "n of n points used" would misrepresent it.
                lines.append(
                    f"<b>{fit.n_points} points</b> compared against it "
                    f"&nbsp;·&nbsp; typical scatter σ = {fit.sigma_v:.4f} V "
                    f"&nbsp;·&nbsp; worst {fit.max_abs_v:.4f} V"
                    f"<br><span style='color:#9a9aa0'>Nothing was fitted to "
                    f"this run — the scatter just says whether the borrowed "
                    f"line still describes it.</span>")
            else:
                lines.append(
                    f"<b>{fit.n_used} of {fit.n_points} points used</b> "
                    f"&nbsp;·&nbsp; typical scatter σ = {fit.sigma_v:.4f} V "
                    f"&nbsp;·&nbsp; worst {fit.max_abs_v:.4f} V")
        if res.mode == "borrow":
            lines.append(self._borrow_note())
        self._stats.setText("<br>".join(lines))
        self._impact.setText(self._impact_text())

    def _borrow_note(self) -> str:
        """Warn when the donor's sweep doesn't cover this run's."""
        try:
            d_set, _ = read_cal_points(self._result.donor)
        except Exception:
            return ""
        if not len(d_set) or not self._n:
            return ""
        if (d_set.min() <= self._set_v.min()
                and d_set.max() >= self._set_v.max()):
            return (f"Borrowed from "
                    f"<b>{os.path.basename(self._result.donor)}</b>.")
        return (f"<span style='color:#fb8c00'>⚠ "
                f"{os.path.basename(self._result.donor)} was only calibrated "
                f"over {d_set.min():.0f}–{d_set.max():.0f} V, but this run "
                f"scans {self._set_v.min():.0f}–{self._set_v.max():.0f} V — "
                f"part of the range is extrapolated.</span>")

    def _impact_text(self) -> str:
        """What this choice costs, against the file's untouched calibration.

        The headline of the dialog. A residual in volts is not actionable; a
        centroid shift in MHz, next to the isotope shift being measured, is.
        """
        if self._result is None or not self._n:
            return ""
        base = self._base_fit
        if base is None:
            return ""

        dv_v = calibration_shift_v(
            self._result.coeffs_v, base.coeffs_v,
            float(self._set_v.min()), float(self._set_v.max()))

        if dv_v <= 0.0:
            return ("<b>What this changes</b><br>"
                    "<span style='color:#9a9aa0'>Nothing yet — this is the "
                    "file's own calibration. Exclude a point, or pick a rule, "
                    "to see the effect.</span>")

        head = ("<b>What this changes</b><br>"
                f"moves the voltage by up to <b>{dv_v:.3f} V</b> "
                f"across the sweep")

        p = self._physics
        if not all(p.get(k) for k in
                   ("mass_amu", "laser_cm", "cooler_v", "harmonic")):
            return head + ("<br><span style='color:#9a9aa0'>Set the mass, "
                           "laser and cooler voltage to see this in "
                           "MHz.</span>")

        try:
            _dv, dnu = shift_mhz_over_scan(
                self._result.coeffs_v, base.coeffs_v,
                float(self._set_v.min()), float(self._set_v.max()),
                cooler_v=float(p["cooler_v"]), laser_cm=float(p["laser_cm"]),
                mass_amu=float(p["mass_amu"]),
                harmonic=int(p["harmonic"]))
        except Exception:
            return head

        mid = float(dnu[len(dnu) // 2])
        tilt = float(dnu.max() - dnu.min())
        big = abs(mid) > 1.0 or tilt > 1.0
        colour = "#ffa726" if big else "#9a9aa0"
        return (
            head
            + f"<br><span style='color:{colour}'>which moves your measured "
              f"<b>centroid by {_fmt_mhz(mid)}</b></span>"
            + (f"<br><span style='color:{colour}'>…and <b>tilts</b> the "
               f"frequency axis by {tilt:.1f} MHz across the scan, so peaks "
               f"move by different amounts depending where they sit</span>"
               if tilt > 1.0 else ""))

    # ── actions ──────────────────────────────────────────────

    def _apply(self):
        spec = self._current_spec()
        if self._result is None:
            return
        self._reg.set(self._path, spec)
        self.accept()

    def _apply_to_others(self):
        """Push this calibration onto other runs.

        For a 'fit' spec this copies the *rule* (drop the first 3, say), which
        is what you want when a whole shift of runs shares a settling problem.
        Hand-picked indices are copied too, but they only mean the same thing
        if the tables line up -- so say so rather than pretending otherwise.
        """
        if self._result is None or not self._peers:
            QMessageBox.information(
                self, "Calibration",
                "No other runs are loaded to copy this calibration to.")
            return
        spec = self._current_spec()

        dlg = _ApplyToRunsDialog(self._peers, describe_spec(spec), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        targets = dlg.selected()
        if not targets:
            return

        # One signal for the whole batch: subscribers reload the runs they
        # touch, and firing per-file would reload each of them N times.
        self._reg.set_many({p: dict(spec) for p in targets})
        self._reg.set(self._path, spec)
        QMessageBox.information(
            self, "Calibration",
            f"Applied to {len(targets)} other run(s).")
        self.accept()

    def _reset(self):
        self._reg.clear(self._path)
        self.accept()


class _ApplyToRunsDialog(QDialog):
    """Tick the runs that should inherit this calibration."""

    def __init__(self, peers: Sequence[Tuple[str, str]], summary: str,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply calibration to other runs")
        self.resize(430, 470)
        style_dialog(self)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Apply <b>{summary}</b> to:"))

        self._boxes: List[Tuple[QCheckBox, str]] = []
        for label, path in peers:
            cb = QCheckBox(label)
            cb.setToolTip(path)
            lay.addWidget(cb)
            self._boxes.append((cb, path))
        lay.addStretch(1)

        note = QLabel(
            "<span style='color:#888'>A rule (e.g. “drop the first 3”) "
            "transfers cleanly. Hand-picked point numbers only mean the same "
            "thing if the runs' calibration tables line up.</span>")
        note.setWordWrap(True)
        lay.addWidget(note)

        row = QHBoxLayout()
        all_btn = QPushButton("Select all")
        all_btn.clicked.connect(
            lambda: [cb.setChecked(True) for cb, _ in self._boxes])
        row.addWidget(all_btn)
        none_btn = QPushButton("Select none")
        none_btn.clicked.connect(
            lambda: [cb.setChecked(False) for cb, _ in self._boxes])
        row.addWidget(none_btn)
        row.addStretch(1)
        lay.addLayout(row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def selected(self) -> List[str]:
        return [p for cb, p in self._boxes if cb.isChecked()]


class CalibrationOverviewDialog(QDialog):
    """Triage every loaded run's calibration at once, worst first.

    The app's default is to change nothing and flag instead, which is only
    workable if the flags find you. Opening a per-run diagnostic across a
    campaign of hundreds of runs is not something anyone will do, so the
    triage comes here: sort by what the calibration is actually costing, in
    MHz, and go fix the top of the list.
    """

    def __init__(self, runs: Sequence[Tuple[str, str]], *,
                 cal_order: int = DEFAULT_ORDER,
                 physics: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voltage calibration — overview")
        self.resize(920, 560)
        style_dialog(self)
        self._runs = list(runs)
        self._physics = physics or {}
        self._cal_order = cal_order

        lay = QVBoxLayout(self)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["", "Run", "Points", "σ [V]", "Cost", "Calibration"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch)
        self._table.cellDoubleClicked.connect(self._open_row)
        lay.addWidget(self._table, 1)

        hint = QLabel("<span style='color:#888'>Double-click a run to open "
                      "its calibration diagnostic.</span>")
        lay.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        exp = btns.addButton("Export…", QDialogButtonBox.ButtonRole.ActionRole)
        imp = btns.addButton("Import…", QDialogButtonBox.ButtonRole.ActionRole)
        exp.setToolTip(
            "Write the calibration overrides to a standalone YAML, so the same\n"
            "fixes can be reused in another project without redoing them.")
        imp.setToolTip(
            "Merge calibration overrides from a YAML (or from another\n"
            "project file) into this session.")
        exp.clicked.connect(self._export)
        imp.clicked.connect(self._import)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        lay.addWidget(btns)

        get_registry().calibrations_changed.connect(self._refresh)
        self._refresh()

    # ── import / export ──────────────────────────────────────

    def _export(self):
        """Write the overrides to a standalone YAML.

        The calibration of a run is a property of the file, but the registry
        is saved inside a project. Without this, a second project over the
        same data starts from scratch and quietly re-fits every bad run with
        its uncorrected calibration.
        """
        import yaml

        cal = get_registry().to_dict()
        if not cal:
            QMessageBox.information(
                self, "Export calibrations",
                "There are no calibration overrides to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export calibrations", "calibrations.yaml",
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        if not path.lower().endswith((".yaml", ".yml")):
            path += ".yaml"
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump({"calibrations": cal}, f, default_flow_style=False,
                          sort_keys=False, allow_unicode=True)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Export calibrations",
            f"Exported {len(cal)} calibration override(s) to\n{path}")

    def _import(self):
        """Merge overrides from a YAML (a calibration export, or a project).

        Merged rather than replaced: importing one campaign's fixes must not
        silently drop the ones already in force for another.
        """
        import yaml

        path, _ = QFileDialog.getOpenFileName(
            self, "Import calibrations", "",
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        incoming = (raw or {}).get("calibrations") if isinstance(raw, dict) \
            else None
        if not isinstance(incoming, dict) or not incoming:
            QMessageBox.warning(
                self, "Import calibrations",
                f"No 'calibrations:' section found in\n{path}")
            return

        reg = get_registry()
        before = set(reg.to_dict())
        merged = reg.to_dict()
        merged.update(incoming)
        reg.from_dict(merged)

        # Report what actually landed, not what was in the file. `from_dict`
        # silently drops entries `normalize_spec` cannot make sense of (a
        # hand-edited YAML, a spec from a future version), and claiming to have
        # imported an override that was quietly discarded is exactly how a run
        # ends up fitted with the wrong calibration while the user believes it
        # was fixed.
        after = set(reg.to_dict())
        landed = after - before
        rejected = len(incoming) - len(
            [p for p in incoming if canonical_path(p) in after])

        known = {canonical_path(p) for _lbl, p in self._runs}
        hit = sum(1 for p in incoming if canonical_path(p) in known
                  and canonical_path(p) in after)

        msg = [f"Imported {len(landed)} override(s) from {len(incoming)} "
               f"entries in the file.",
               f"{hit} match a run loaded here; the rest are kept and will "
               f"apply when those runs are loaded."]
        if rejected:
            msg.append("")
            msg.append(f"⚠ {rejected} entr{'y was' if rejected == 1 else 'ies were'} "
                       f"not understood and had to be discarded.")
        QMessageBox.information(self, "Import calibrations", "\n".join(msg))

    def _refresh(self):
        cal_map = get_registry().to_dict()
        rows = []
        for label, path in self._runs:
            d = diagnose(path, cal_map, run_number=label,
                         cal_order=self._cal_order)
            rows.append((d, self._cost_mhz(d, cal_map)))

        # Worst first: that is the entire job of this table.
        rows.sort(key=lambda t: (-(t[1] if t[1] is not None else t[0].impact_v),
                                 t[0].run_number))

        self._table.setRowCount(len(rows))
        n_flagged = 0
        for r, (d, cost) in enumerate(rows):
            flag = "⚠" if d.flagged else ("✓" if not d.error else "!")
            if d.flagged:
                n_flagged += 1
            item = QTableWidgetItem(flag)
            item.setData(Qt.ItemDataRole.UserRole, d.path)
            self._table.setItem(r, 0, item)
            self._table.setItem(r, 1, QTableWidgetItem(d.run_number))
            # All points are in use unless an override says otherwise; the
            # suspects are what a fix *would* drop, not what has been dropped.
            pts = str(d.n_points)
            if d.suspect and not d.has_override:
                pts += f"  ({len(d.suspect)} suspect)"
            self._table.setItem(r, 2, QTableWidgetItem(pts))
            self._table.setItem(r, 3, QTableWidgetItem(
                "—" if np.isnan(d.sigma_v) else f"{d.sigma_v:.4f}"))
            if d.error:
                cost_txt = "error"
            elif cost is not None:
                cost_txt = _fmt_mhz(cost) if d.suspect else "—"
            else:
                cost_txt = (f"{d.impact_v:.3f} V" if d.suspect else "—")
            self._table.setItem(r, 4, QTableWidgetItem(cost_txt))
            self._table.setItem(r, 5, QTableWidgetItem(
                d.error or d.spec_summary))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch)

        if n_flagged:
            self._summary.setText(
                f"<b style='color:#fb8c00'>{n_flagged} of {len(rows)} run(s) "
                f"have calibration outliers worth acting on.</b> Nothing has "
                f"been changed — open a run to decide.")
        else:
            self._summary.setText(
                f"<b>All {len(rows)} run(s) look fine.</b> No calibration "
                f"outlier would move a centroid by more than "
                f"{FLAG_MIN_IMPACT_V:.2f} V.")

    def _cost_mhz(self, d, cal_map) -> Optional[float]:
        """The flagged outliers' cost in MHz, if the physics is known.

        Beam energy and laser setpoint are read from **this** run's header
        rather than taken from the shared ``physics`` dict: they are per-run
        values, and a table that quoted every row's cost using the first run's
        cooler voltage would be confidently wrong for all the others. Mass and
        harmonic do come from the caller -- those belong to the analysis, not
        the file.
        """
        p = self._physics
        if (not d.suspect or d.error
                or not all(p.get(k) for k in ("mass_amu", "harmonic"))):
            return None
        cooler_v, laser_cm = read_beam_header(d.path)
        if not (cooler_v and laser_cm):
            return None
        try:
            set_v, read_raw = read_cal_points(d.path)
            cur = resolve_calibration(d.path, cal_map,
                                      cal_order=self._cal_order)
            trimmed = fit_calibration(set_v, read_raw, self._cal_order,
                                      d.suspect, DEFAULT_VACC_DIV)
            _dv, dnu = shift_mhz_over_scan(
                cur.coeffs_v, trimmed.coeffs_v,
                float(set_v.min()), float(set_v.max()),
                cooler_v=cooler_v, laser_cm=laser_cm,
                mass_amu=float(p["mass_amu"]), harmonic=int(p["harmonic"]))
            return float(dnu[len(dnu) // 2])
        except Exception:
            return None

    def _open_row(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        label = self._table.item(row, 1).text()
        # This run's own beam energy and laser, not the shared dict's.
        cooler_v, laser_cm = read_beam_header(path)
        phys = dict(self._physics)
        if cooler_v:
            phys["cooler_v"] = cooler_v
        if laser_cm:
            phys["laser_cm"] = laser_cm
        CalibrationDialog(
            path, run_label=label, cal_order=self._cal_order,
            physics=phys, peers=self._runs, parent=self).exec()
        self._refresh()

    def closeEvent(self, event):
        """Drop the registry subscription.

        Qt keeps this dialog alive through its parent after ``exec()`` returns,
        so a live connection would make every later calibration change re-run
        a full ``diagnose`` sweep over every run -- in a dead window nobody is
        looking at, once per stale dialog ever opened.
        """
        try:
            get_registry().calibrations_changed.disconnect(self._refresh)
        except (RuntimeError, TypeError):
            pass          # already gone, or never connected
        super().closeEvent(event)
