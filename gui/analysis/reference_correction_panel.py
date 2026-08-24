"""Reference Correction panel embedded in the Isotope Shifts tab.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Trains a Gaussian-process corrector on time-stamped reference-isotope
centroids drawn from selected Reference Projects, displays the diagnostic
plot, and exposes a per-run correction table for selected Sample Projects.
The panel collects the reference observations, fits the GP, evaluates
μ_GP(t_run) for each sample run, and stores those numbers so the merge /
fit pipeline can subtract them after the Doppler conversion. The
merge-time application itself lives in merge.py / fitting.py, not here.

Depends on: cls_estimations.reference_correction, gui.analysis.blocks;
uses NumPy, matplotlib, and PySide6 (with a background QThread fit worker).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPlainTextEdit, QMessageBox, QSpinBox, QToolButton,
)

from cls_estimations.reference_correction import (
    ReferenceCorrector, ReferenceObservation, SUPPORTED_KERNELS,
)


_KERNEL_LABELS = {
    "rbf": "RBF + WhiteNoise",
    "matern": "Matérn(5/2) + WhiteNoise",
    "thesis": "Composite (slow RBF + Periodic·Matérn52 + WhiteNoise)",
}

_MODE_AUTO = "Auto"
_MODE_MANUAL = "Manual"
_MODE_OFF = "Off"
_MODES = (_MODE_AUTO, _MODE_MANUAL, _MODE_OFF)


@dataclass
class _FileCorrection:
    """One per-file correction row in the panel's table.

    Auto values are populated by ``Compute per-file corrections``
    from the GP at the run's timestamp; Manual values are populated
    by user edits when ``mode == Manual``. They live in separate
    fields so toggling mode preserves both sets and never silently
    pulls a manual entry into Auto's display.

    The ``value_mhz`` / ``sigma_mhz`` properties resolve which pair
    a caller sees based on the current ``mode``.
    """
    project: str
    run_number: str
    file_path: str
    ts_start: float
    t_hours: float
    mode: str = _MODE_AUTO
    auto_value_mhz: float = 0.0
    auto_sigma_mhz: float = 0.0
    manual_value_mhz: float = 0.0
    manual_sigma_mhz: float = 0.0

    @property
    def value_mhz(self) -> float:
        return (self.manual_value_mhz if self.mode == _MODE_MANUAL
                else self.auto_value_mhz)

    @property
    def sigma_mhz(self) -> float:
        return (self.manual_sigma_mhz if self.mode == _MODE_MANUAL
                else self.auto_sigma_mhz)


# ──────────────────────────────────────────────────────────────────
#  Background worker for fit()
# ──────────────────────────────────────────────────────────────────

class _FitWorker(QThread):
    """Run ``corrector.fit(observations)`` off the GUI thread.

    PyTensor's compile cache is process-global; we only ever start ONE
    worker at a time. The main panel disables the Fit button while
    a worker is alive.
    """

    done = Signal()
    failed = Signal(str)

    def __init__(self, corrector: ReferenceCorrector,
                 observations: list[ReferenceObservation], parent=None):
        super().__init__(parent)
        # Public so the panel's _on_fit_done slot can pull the fitted
        # corrector out without reaching into a private attribute.
        self.corrector = corrector
        self._observations = observations
        self.error_message: str | None = None

    def run(self):
        try:
            self.corrector.fit(self._observations)
            self.done.emit()
        except Exception as e:
            self.error_message = f"{type(e).__name__}: {e}"
            self.failed.emit(self.error_message)


# ──────────────────────────────────────────────────────────────────
#  ReferenceCorrectionPanel
# ──────────────────────────────────────────────────────────────────

class ReferenceCorrectionPanel(QGroupBox):
    """Group box that owns the GP-correction workflow.

    Signals
    -------
    corrections_changed : str, dict
        Emitted whenever the per-file correction table changes for a
        sample project. ``str`` is the sample-project name; ``dict``
        is keyed by ``file_path`` and contains
        ``{"correction_mhz", "sigma_mhz", "mode"}``.
    """

    corrections_changed = Signal(str, dict)

    def __init__(self, analysis_tab, parent=None):
        super().__init__("Reference Correction (GP)", parent)
        self._analysis_tab = analysis_tab
        self._corrector: ReferenceCorrector | None = None
        self._fit_worker: _FitWorker | None = None
        # `_fit_busy` is the source of truth for "fit in progress".
        # `_fit_worker.isRunning()` would race with the queued
        # `done`/`failed` slot dispatch (the worker reports done before
        # the slot has cleared `_fit_worker`).
        self._fit_busy = False
        # corrections[sample_project_name][file_path] -> _FileCorrection
        self._corrections: dict[str, dict[str, _FileCorrection]] = {}
        self._build_ui()
        # Auto-refresh when the user creates / closes a project so the
        # ref / sample lists don't go stale -- AnalysisTab is expected
        # to emit this signal.
        if hasattr(analysis_tab, "projects_changed"):
            analysis_tab.projects_changed.connect(self._on_projects_changed)

    # ── UI construction ─────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # ── Header: help button ──
        header_row = QHBoxLayout()
        header_row.addStretch(1)
        help_btn = QToolButton()
        help_btn.setText("?")
        help_btn.setToolTip("Show the workflow guide for this panel")
        help_btn.setFixedSize(22, 22)
        help_btn.clicked.connect(self._show_help)
        header_row.addWidget(help_btn)
        outer.addLayout(header_row)

        # ── Row 1: kernel + MCMC + apply toggle ──
        top_form = QFormLayout()
        top_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._kernel_combo = QComboBox()
        for k in SUPPORTED_KERNELS:
            self._kernel_combo.addItem(_KERNEL_LABELS[k], userData=k)
        self._kernel_combo.setToolTip(
            "Kernel for the GP. RBF is the simplest and fastest; "
            "Matérn handles small jumps; Composite is a 3-term "
            "kernel (slow RBF + diurnal Periodic·Matérn52 + "
            "WhiteNoise) for long campaigns with day/night lock "
            "cycles.")
        top_form.addRow("Kernel:", self._kernel_combo)

        self._mcmc_cb = QCheckBox(
            "Also run MCMC (slow; for diagnostic posterior)")
        self._mcmc_cb.setToolTip(
            "If checked, run pm.sample after find_MAP. The MCMC "
            "trace is summarized for display but predict()/cov() "
            "still use the MAP point.")
        top_form.addRow("", self._mcmc_cb)

        self._mcmc_draws = QSpinBox()
        self._mcmc_draws.setRange(100, 50000)
        self._mcmc_draws.setValue(1000)
        self._mcmc_draws.setSingleStep(100)
        self._mcmc_draws.setToolTip("MCMC draws after tuning")
        top_form.addRow("MCMC draws:", self._mcmc_draws)

        self._apply_global_cb = QCheckBox("Apply correction at fit time")
        self._apply_global_cb.setChecked(True)
        self._apply_global_cb.setToolTip(
            "Global toggle: when off, downstream merging/fitting "
            "ignores all per-file corrections regardless of mode.")
        top_form.addRow("", self._apply_global_cb)

        outer.addLayout(top_form)

        # ── Reference projects ──
        outer.addWidget(QLabel("Reference projects:"))
        self._ref_list = QListWidget()
        self._ref_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._ref_list.setMaximumHeight(110)
        outer.addWidget(self._ref_list)

        ref_btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh project list")
        refresh_btn.clicked.connect(self.refresh_projects)
        ref_btn_row.addWidget(refresh_btn)
        self._fit_btn = QPushButton("Fit GP")
        self._fit_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 4px 12px; }")
        self._fit_btn.clicked.connect(self._fit_clicked)
        ref_btn_row.addWidget(self._fit_btn)
        ref_btn_row.addStretch()
        outer.addLayout(ref_btn_row)

        self._status = QLabel("(no GP fit yet)")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        # ── Diagnostic plot ──
        self._figure = Figure(figsize=(7, 3.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(220)
        outer.addWidget(self._canvas)

        # ── Sample projects ──
        outer.addWidget(QLabel("Apply to sample projects:"))
        self._sample_list = QListWidget()
        self._sample_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._sample_list.setMaximumHeight(110)
        outer.addWidget(self._sample_list)

        sample_btn_row = QHBoxLayout()
        compute_btn = QPushButton("Compute per-file corrections")
        compute_btn.clicked.connect(self._compute_corrections_clicked)
        sample_btn_row.addWidget(compute_btn)
        sample_btn_row.addStretch()
        outer.addLayout(sample_btn_row)

        # ── Per-file corrections table ──
        self._corr_table = QTableWidget(0, 7)
        self._corr_table.setHorizontalHeaderLabels(
            ["Project", "Run", "t (h)", "Mode",
             "μ (MHz)", "σ (MHz)", "|μ|/σ"])
        h = self._corr_table.horizontalHeader()
        for col in range(7):
            h.setSectionResizeMode(
                col, QHeaderView.ResizeMode.Interactive)
        h.resizeSection(0, 130)
        h.resizeSection(1, 70)
        h.resizeSection(2, 70)
        h.resizeSection(3, 90)
        h.resizeSection(4, 100)
        h.resizeSection(5, 100)
        h.resizeSection(6, 70)
        self._corr_table.verticalHeader().setVisible(False)
        self._corr_table.setMinimumHeight(150)
        self._corr_table.itemChanged.connect(self._table_item_changed)
        # Click-to-sort: clicking a column header sorts on that
        # column. Internal repopulates disable sorting briefly to
        # avoid row-index reshuffling while inserting (see
        # _populate_table).
        self._corr_table.setSortingEnabled(True)
        outer.addWidget(self._corr_table)

    # ── Public refresh ───────────────────────────────────────

    @classmethod
    def _pytensor_cxx_note(cls, *, _probe=None) -> str:
        """Return a one-line GUI hint when PyTensor has no C compiler
        on PATH. Cached after the first call so repeated fits don't
        re-import pytensor every click. Returns empty string when a
        compiler is available -- the production-speed case stays
        silent.

        ``_probe`` is a test seam: a callable returning the cxx
        config string, used to bypass the real ``import pytensor``
        in unit tests. Probe-driven calls also bypass and refresh
        the class-level cache so each test starts from a clean
        slate.
        """
        if _probe is None:
            cached = getattr(cls, "_cxx_note_cache", None)
            if cached is not None:
                return cached
        note = ""
        try:
            if _probe is not None:
                cxx = _probe()
            else:
                import pytensor
                cxx = pytensor.config.cxx
            if not (cxx or "").strip():
                note = (
                    "  ⚠ PyTensor C compiler not detected -- using "
                    "pure-Python fallback (10–100× slower). On "
                    "Windows: install MSVC Build Tools or "
                    "`conda install m2w64-toolchain`.")
        except Exception:  # noqa: BLE001
            # PyTensor not importable means PyMC isn't installed
            # either; ReferenceCorrector.fit will surface that
            # error itself, no need to duplicate here.
            pass
        if _probe is None:
            cls._cxx_note_cache = note
        return note

    def _show_help(self):
        """Show a plain-language workflow guide for the GP-correction
        panel, written for a CLS physicist who has not read the thesis."""
        QMessageBox.information(
            self, "Reference Correction — guide",
            "<h3>What this panel does</h3>"
            "<p>Long campaigns drift: the wavemeter readout and laser "
            "lock wander by tens of MHz over hours-to-days. If you "
            "treat the reference frequency as a single number, that "
            "drift biases isotope-shift results by the difference "
            "between the reference time and the sample time. This "
            "panel fits a Gaussian process to time-stamped reference-"
            "isotope scans and subtracts the inferred drift from each "
            "sample run.</p>"
            "<h4>Step-by-step</h4>"
            "<ol>"
            "<li><b>Create a Reference Project.</b> Add ASDFs of the "
            "stable reference isotope, set up its HFS model + fitter, "
            "run the fit. Each fitted centroid feeds the GP.</li>"
            "<li><b>Pick a kernel</b> (top of this panel). RBF for "
            "smooth slow drift; Matérn for occasional small jumps; "
            "Composite for long campaigns with diurnal lock "
            "cycles.</li>"
            "<li><b>Click Fit GP.</b> The diagnostic plot below shows "
            "scatter of the reference centroids, the MAP curve, and "
            "1σ/2σ bands.</li>"
            "<li><b>Tick the Sample Project(s)</b> you want corrected, "
            "click Compute per-file corrections. The table populates "
            "with μ_GP and σ_GP for each run.</li>"
            "<li><b>Mode</b> per row: Auto = use μ_GP. Manual = type "
            "your own μ and σ (e.g. a published value). Off = skip "
            "this run.</li>"
            "<li><b>Re-fit the Sample Project.</b> The fit reads this "
            "table and subtracts the correction from the binned "
            "frequency axis. Console logs which files got corrected.</li>"
            "</ol>"
            "<h4>Reading the table</h4>"
            "<p>The <b>|μ|/σ</b> column flags rows where the GP "
            "thinks the run is far from the smooth drift it learned. "
            "Amber (>2σ) is worth a look; red (>3σ) usually means "
            "either the run is genuinely off, or your kernel is too "
            "smooth to capture the local behavior. Off rows are "
            "shown grey so you can see at a glance which files are "
            "skipped.</p>"
            "<h4>Save / load</h4>"
            "<p>The trained GP, the table, and the per-row Mode "
            "choices all round-trip through the project YAML. "
            "Saving without first fitting the GP is fine; the table "
            "is empty on reload and you'll be told to refit.</p>")

    def _on_projects_changed(self):
        """Called when AnalysisTab adds or closes a project. Refresh
        the lists and drop cached corrections for projects that no
        longer exist (otherwise stale entries linger through the YAML
        round-trip and the fit pipeline would apply ghost corrections)."""
        live_names = {p.project_name for p in self._analysis_tab._projects}
        stale = [k for k in self._corrections if k not in live_names]
        for k in stale:
            del self._corrections[k]
        if stale:
            self._populate_table()
        self.refresh_projects()

    def refresh_projects(self, *, restore_state: dict | None = None):
        """Re-scan ``analysis_tab._projects`` and rebuild the Reference
        / Sample lists.

        Parameters
        ----------
        restore_state : dict or None
            If given, ``{"ref": [names], "sam": [names]}`` overrides the
            current checkbox state. Used by ``from_dict`` after a
            project save/load. Empty lists mean "explicitly nothing
            checked", distinct from ``None`` (no preference).
        """
        if restore_state is not None:
            prev_ref = set(restore_state.get("ref", []))
            prev_sam = set(restore_state.get("sam", []))
            explicit = True
        elif self._ref_list.count() > 0 or self._sample_list.count() > 0:
            prev_ref = self._checked_names(self._ref_list)
            prev_sam = self._checked_names(self._sample_list)
            explicit = True
        else:
            prev_ref = set()
            prev_sam = set()
            explicit = False  # first run -> default check all reference

        self._ref_list.clear()
        self._sample_list.clear()
        for p in self._analysis_tab._projects:
            if p.is_reference:
                n_obs = len(p.get_reference_observations())
                if n_obs == 0:
                    txt = f"{p.project_name}  (0 obs — fit first)"
                else:
                    txt = f"{p.project_name}  ({n_obs} obs)"
                item = QListWidgetItem(txt)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if n_obs == 0:
                    item.setToolTip(
                        "This reference project has no fitted "
                        "centroids yet. Open the project tab, run "
                        "its fit pipeline, then refresh here.")
                checked = ((p.project_name in prev_ref)
                           if explicit else True)
                item.setCheckState(
                    Qt.CheckState.Checked if checked
                    else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, p.project_name)
                self._ref_list.addItem(item)
            else:
                item = QListWidgetItem(p.project_name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                checked = (p.project_name in prev_sam) if explicit else False
                item.setCheckState(
                    Qt.CheckState.Checked if checked
                    else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, p.project_name)
                self._sample_list.addItem(item)

    # ── Fit GP ─────────────────────────────────────────────

    def _fit_clicked(self):
        if self._fit_busy:
            QMessageBox.information(
                self, "Fit in progress",
                "A GP fit is already running. Wait for it to finish.")
            return

        # Gather observations from all checked reference projects.
        ref_names = self._checked_names(self._ref_list)
        if not ref_names:
            QMessageBox.warning(
                self, "No reference projects",
                "Check at least one reference project (and make sure "
                "it has been fit first so its centroids are available).")
            return
        all_obs: list[ReferenceObservation] = []
        for p in self._analysis_tab._projects:
            if p.project_name in ref_names:
                rows = p.get_reference_observations()
                for r in rows:
                    all_obs.append(ReferenceObservation(
                        t=r["ts_start"] / 3600.0,    # hours since epoch
                        centroid=r["centroid_mhz"],
                        sigma=r["sigma_mhz"],
                        label=f"{p.project_name}/{r['label']}",
                        include=r.get("include", True),
                    ))
        if len(all_obs) < 2:
            QMessageBox.warning(
                self, "Not enough observations",
                f"Need ≥ 2 reference observations to fit the GP "
                f"(got {len(all_obs)}). Re-run the reference project's "
                f"fit pipeline first.")
            return

        kernel = self._kernel_combo.currentData() or "rbf"
        run_mcmc = self._mcmc_cb.isChecked()
        n_samples = self._mcmc_draws.value()
        rc = ReferenceCorrector(
            kernel=kernel, run_mcmc=run_mcmc, n_samples=n_samples)

        self._fit_busy = True
        self._set_busy(True)
        # Paint the base status FIRST so the user sees "Fitting GP…"
        # immediately, then append the C-compiler hint -- that
        # subroutine triggers a (cached) `import pytensor` which
        # can take a couple seconds on the first call. Order:
        # (1) paint base text, (2) flush via processEvents so the
        # QLabel actually repaints, (3) append the cxx note (no-op
        # on machines that have a compiler).
        from PySide6.QtCore import QCoreApplication
        base = (f"Fitting GP ({kernel}, {len(all_obs)} obs)... "
                f"this may take 20-90 s.")
        self._status.setText(base)
        QCoreApplication.processEvents()
        cxx_note = self._pytensor_cxx_note()
        if cxx_note:
            self._status.setText(base + cxx_note)

        # Direct slot connections (not lambdas) so Qt's auto-disconnect
        # on receiver deletion takes care of cleanup if the panel is
        # closed mid-fit. The worker carries `rc` itself.
        self._fit_worker = _FitWorker(rc, all_obs, self)
        self._fit_worker.done.connect(self._on_fit_done)
        self._fit_worker.failed.connect(self._on_fit_failed)
        self._fit_worker.finished.connect(self._fit_worker.deleteLater)
        self._fit_worker.start()

    def _on_fit_done(self):
        worker = self._fit_worker
        self._fit_worker = None
        self._fit_busy = False
        if worker is None:
            return
        rc = worker.corrector
        self._fit_done(rc)

    def _on_fit_failed(self, msg: str):
        self._fit_worker = None
        self._fit_busy = False
        self._fit_failed(msg)

    def _fit_done(self, rc: ReferenceCorrector):
        self._set_busy(False)
        self._corrector = rc
        hp = rc.hyperparameters
        self._status.setText(
            f"GP fit OK ({rc.kernel}; n={len(rc.observations)}). "
            f"σ_n = {hp.sigma_n:.3g} MHz" + (
                f", ℓ = {hp.ell:.3g}, η = {hp.eta:.3g}"
                if rc.kernel in ("rbf", "matern") else
                f", ℓ_slow = {hp.length_slow:.3g}, "
                f"P_fast = {hp.period_fast:.3g}"
            ))
        # Stale per-file numbers belong to the previous corrector.
        # Clearing them forces the user to recompute, which is cheap
        # and avoids silently-wrong displays.
        if self._corrections:
            self._corrections = {}
            self._populate_table()
            self._status.setText(
                self._status.text()
                + " — old corrections cleared; recompute to refresh.")
        self._render_diagnostic_plot()

    def _fit_failed(self, msg: str):
        self._set_busy(False)
        self._corrector = None
        self._status.setText(f"GP fit failed: {msg}")
        # Stale plot would mislead the user about the current state.
        self._figure.clear()
        self._canvas.draw()
        QMessageBox.critical(self, "Fit error", msg)

    def _set_busy(self, busy: bool):
        self._fit_btn.setEnabled(not busy)
        self._kernel_combo.setEnabled(not busy)

    def _render_diagnostic_plot(self):
        if self._corrector is None:
            return
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        try:
            self._corrector.diagnostic_plot(ax=ax, n_grid=300, t_unit="h")
        except Exception as e:
            ax.text(0.5, 0.5,
                    f"Plot error:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
        self._figure.tight_layout()
        self._canvas.draw()

    # ── Compute per-file corrections ───────────────────────

    def _compute_corrections_clicked(self):
        if self._corrector is None or not self._corrector.is_fit:
            QMessageBox.warning(
                self, "No GP fit",
                "Fit the GP first, then compute corrections.")
            return

        sample_names = self._checked_names(self._sample_list)
        if not sample_names:
            QMessageBox.warning(
                self, "No sample projects",
                "Check at least one sample project to apply the "
                "correction to.")
            return

        new_corrections: dict[str, dict[str, _FileCorrection]] = {}
        skipped_no_fit: list[str] = []
        skipped_no_ts: list[str] = []
        skipped_all_failed: list[str] = []
        for p in self._analysis_tab._projects:
            if p.project_name not in sample_names or p.is_reference:
                continue
            results = getattr(p, "_last_results", None)
            if not results:
                skipped_no_fit.append(p.project_name)
                continue
            project_map: dict[str, _FileCorrection] = {}
            had_success = False
            for r in results:
                if not r.get("success"):
                    continue
                had_success = True
                fpath = r.get("run_file") or ""
                run_num = str(r.get("run_number", "") or "")
                meta = r.get("run_metadata") or {}
                try:
                    ts_start = float(meta.get("ts_start", 0) or 0)
                except (TypeError, ValueError):
                    ts_start = 0.0
                if ts_start <= 0:
                    skipped_no_ts.append(f"{p.project_name}/{run_num}")
                    continue
                t_hours = ts_start / 3600.0
                mu, sigma = self._corrector.predict(t_hours)
                fc = _FileCorrection(
                    project=p.project_name,
                    run_number=run_num,
                    file_path=fpath,
                    ts_start=ts_start,
                    t_hours=t_hours,  # rebased below
                    mode=_MODE_AUTO,
                    auto_value_mhz=float(mu[0]),
                    auto_sigma_mhz=float(sigma[0]),
                )
                # Preserve mode + manual edits if the user already
                # interacted with this row in a previous compute.
                # Auto values are always overwritten with the fresh
                # GP prediction.
                old = self._corrections.get(p.project_name, {}).get(fpath)
                if old is not None:
                    fc.mode = old.mode
                    fc.manual_value_mhz = old.manual_value_mhz
                    fc.manual_sigma_mhz = old.manual_sigma_mhz
                project_map[fpath] = fc
            if not had_success:
                skipped_all_failed.append(p.project_name)
                continue
            if project_map:
                new_corrections[p.project_name] = project_map

        # Re-base t_hours so the column shows "hours since first run"
        # (small numbers) rather than "hours since 1970" (~470000).
        all_ts = [fc.ts_start for fmap in new_corrections.values()
                  for fc in fmap.values()]
        if all_ts:
            t0_seconds = min(all_ts)
            for fmap in new_corrections.values():
                for fc in fmap.values():
                    fc.t_hours = (fc.ts_start - t0_seconds) / 3600.0

        self._corrections = new_corrections
        self._populate_table()
        self._emit_corrections_for_all()

        msgs = []
        if skipped_no_fit:
            msgs.append(
                "No fit results yet -- fit the sample project first: "
                + ", ".join(skipped_no_fit))
        if skipped_all_failed:
            msgs.append(
                "All runs in the project failed to fit; nothing to "
                "correct: " + ", ".join(skipped_all_failed))
        if skipped_no_ts:
            msgs.append(
                "Skipped (no ts_start in run_metadata): "
                + ", ".join(skipped_no_ts[:8])
                + (f" (+{len(skipped_no_ts) - 8} more)"
                   if len(skipped_no_ts) > 8 else "")
                + ". This typically means the project was fit before "
                "Phase A added ts_start capture; re-run the fit and "
                "click Compute again.")
        if msgs:
            QMessageBox.information(
                self, "Corrections computed", "\n\n".join(msgs))

    def _populate_table(self):
        # Disable sorting during the rebuild so cell setItem() calls
        # don't reorder partially-populated rows. Restore it after.
        was_sorting = self._corr_table.isSortingEnabled()
        self._corr_table.setSortingEnabled(False)
        self._corr_table.blockSignals(True)
        try:
            self._corr_table.setRowCount(0)
            for proj, fmap in self._corrections.items():
                for fc in fmap.values():
                    row = self._corr_table.rowCount()
                    self._corr_table.insertRow(row)
                    # Project / Run / t -- read-only, sort numerically
                    # for the t column via _NumericItem.
                    self._corr_table.setItem(
                        row, 0, _ro_item(fc.project))
                    self._corr_table.setItem(
                        row, 1, _ro_item(fc.run_number))
                    self._corr_table.setItem(
                        row, 2, _NumericItem(fc.t_hours, fmt=".3f",
                                              read_only=True))
                    # Mode dropdown
                    mode_combo = QComboBox()
                    mode_combo.addItems(_MODES)
                    mode_combo.setCurrentText(fc.mode)
                    mode_combo.currentTextChanged.connect(
                        lambda txt, p=fc.project, fp=fc.file_path:
                        self._mode_changed(p, fp, txt))
                    self._corr_table.setCellWidget(row, 3, mode_combo)
                    # μ -- editable in Manual mode
                    val_item = _NumericItem(fc.value_mhz, fmt=".6g")
                    val_item.setData(
                        Qt.ItemDataRole.UserRole,
                        (fc.project, fc.file_path, "value"))
                    if fc.mode != _MODE_MANUAL:
                        val_item.setFlags(
                            val_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._corr_table.setItem(row, 4, val_item)
                    # σ -- editable in Manual mode: a manually-typed
                    # correction may have a different uncertainty than
                    # the GP's σ_GP, e.g. 0 if the user pulled the value
                    # from a published number.
                    sig_item = _NumericItem(fc.sigma_mhz, fmt=".3g")
                    sig_item.setData(
                        Qt.ItemDataRole.UserRole,
                        (fc.project, fc.file_path, "sigma"))
                    if fc.mode != _MODE_MANUAL:
                        sig_item.setFlags(
                            sig_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._corr_table.setItem(row, 5, sig_item)
                    # |μ|/σ -- a quick "is this an outlier?" cue
                    ratio = (abs(fc.value_mhz) / fc.sigma_mhz
                             if fc.sigma_mhz > 0 else float("inf"))
                    ratio_item = _NumericItem(
                        ratio, fmt=".2f", read_only=True)
                    self._corr_table.setItem(row, 6, ratio_item)
                    # Color-code rows by |μ|/σ. Amber at >2σ ("worth
                    # a look"), red-ish at >3σ ("almost certainly an
                    # outlier"). Off rows get a faint grey overlay so
                    # users see at a glance that those files are
                    # being skipped.
                    self._color_row(row, fc.mode, ratio)
        finally:
            self._corr_table.blockSignals(False)
            self._corr_table.setSortingEnabled(was_sorting)

    def _color_row(self, row: int, mode: str, ratio: float):
        if mode == _MODE_OFF:
            color = QColor(180, 180, 180, 40)   # grey, "skipped"
        elif ratio > 3.0:
            color = QColor(244, 67, 54, 70)     # red-ish, ">3σ"
        elif ratio > 2.0:
            color = QColor(255, 193, 7, 70)     # amber,  ">2σ"
        else:
            color = QColor(0, 0, 0, 0)          # transparent
        for col in range(self._corr_table.columnCount()):
            item = self._corr_table.item(row, col)
            if item is not None:
                item.setBackground(color)
        # Mode column hosts a QComboBox via setCellWidget, which has
        # no QTableWidgetItem -- color its background via stylesheet
        # so the row tint isn't broken by a white gap at column 3.
        widget = self._corr_table.cellWidget(row, 3)
        if widget is not None:
            r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
            if a == 0:
                widget.setStyleSheet("")  # default theme
            else:
                widget.setStyleSheet(
                    f"QComboBox {{ background-color: "
                    f"rgba({r}, {g}, {b}, {a}); }}")

    def _mode_changed(self, project: str, file_path: str, mode: str):
        fmap = self._corrections.get(project)
        if not fmap or file_path not in fmap:
            return
        fc = fmap[file_path]
        # When the user switches to Manual for the first time on a
        # row, seed the manual fields from the current Auto values
        # so the cell editor opens with the GP prediction visible
        # rather than literal 0.0. Switching back to Auto leaves
        # the manual values intact (they're independently
        # persisted), and a future Manual switch re-uses them.
        if (mode == _MODE_MANUAL
                and fc.manual_value_mhz == 0.0
                and fc.manual_sigma_mhz == 0.0):
            fc.manual_value_mhz = fc.auto_value_mhz
            fc.manual_sigma_mhz = fc.auto_sigma_mhz
        fc.mode = mode
        # Repopulate so the value cell's editable flag updates.
        self._populate_table()
        self._emit_corrections(project)

    def _table_item_changed(self, item):
        ud = item.data(Qt.ItemDataRole.UserRole)
        if not ud:
            return
        project, file_path, kind = ud
        if kind not in ("value", "sigma"):
            return
        try:
            v = float(item.text())
        except ValueError:
            return
        fmap = self._corrections.get(project)
        if not fmap or file_path not in fmap:
            return
        fc = fmap[file_path]
        if fc.mode != _MODE_MANUAL:
            return
        # Suspend sorting while we touch sibling cells: Qt re-sorts
        # on every setText/setBackground when sortingEnabled, which
        # would shuffle the row out from under the user's cursor
        # mid-edit if they're sorted by μ or |μ|/σ.
        was_sorting = self._corr_table.isSortingEnabled()
        self._corr_table.setSortingEnabled(False)
        try:
            if kind == "value":
                fc.manual_value_mhz = v
            elif kind == "sigma":
                if v < 0:
                    # Negative σ is meaningless; reject silently.
                    # Cell text will be re-rendered on the next
                    # repopulate.
                    return
                fc.manual_sigma_mhz = v
            self._emit_corrections(project)
            # Re-color the row so the |μ|/σ cue tracks the edit.
            ratio = (abs(fc.value_mhz) / fc.sigma_mhz
                     if fc.sigma_mhz > 0 else float("inf"))
            self._color_row(item.row(), fc.mode, ratio)
            # Update |μ|/σ cell text too.
            ratio_item = self._corr_table.item(item.row(), 6)
            if ratio_item is not None:
                ratio_item.setText(f"{ratio:.2f}")
        finally:
            self._corr_table.setSortingEnabled(was_sorting)

    # ── Outbound: corrections_changed signal ───────────────

    def _emit_corrections(self, project: str):
        fmap = self._corrections.get(project, {})
        out = {}
        for fpath, fc in fmap.items():
            out[fpath] = {
                "correction_mhz": fc.value_mhz,
                "sigma_mhz": fc.sigma_mhz,
                "mode": fc.mode,
            }
        self.corrections_changed.emit(project, out)

    def _emit_corrections_for_all(self):
        for project in self._corrections:
            self._emit_corrections(project)

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _checked_names(list_widget: QListWidget) -> set[str]:
        out: set[str] = set()
        for i in range(list_widget.count()):
            it = list_widget.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                name = it.data(Qt.ItemDataRole.UserRole)
                if name:
                    out.add(str(name))
        return out

    @staticmethod
    def _first_source_block(project):
        from gui.analysis.blocks import SourceBlock
        for b in project._blocks:
            if isinstance(b, SourceBlock):
                return b
        return None

    # ── Save / load ──────────────────────────────────────────

    def to_dict(self) -> dict:
        d: dict = {
            "kernel": self._kernel_combo.currentData() or "rbf",
            "run_mcmc": self._mcmc_cb.isChecked(),
            "mcmc_draws": self._mcmc_draws.value(),
            "apply_global": self._apply_global_cb.isChecked(),
            "checked_reference_projects": sorted(
                self._checked_names(self._ref_list)),
            "checked_sample_projects": sorted(
                self._checked_names(self._sample_list)),
            "corrector": (self._corrector.to_dict()
                          if self._corrector is not None else None),
            "corrections": {
                project: {
                    fpath: {
                        "run_number": fc.run_number,
                        "ts_start": fc.ts_start,
                        "t_hours": fc.t_hours,
                        "mode": fc.mode,
                        # Persist auto and manual fields separately
                        # so a Manual->Auto->Manual round-trip
                        # preserves the user's typed numbers.
                        "auto_value_mhz": fc.auto_value_mhz,
                        "auto_sigma_mhz": fc.auto_sigma_mhz,
                        "manual_value_mhz": fc.manual_value_mhz,
                        "manual_sigma_mhz": fc.manual_sigma_mhz,
                        # Legacy compound fields kept for back-compat
                        # readers; new code should prefer the
                        # auto_/manual_ pair.
                        "value_mhz": fc.value_mhz,
                        "sigma_mhz": fc.sigma_mhz,
                    } for fpath, fc in fmap.items()
                } for project, fmap in self._corrections.items()
            },
        }
        return d

    def from_dict(self, d: dict):
        if not d:
            return
        kernel = d.get("kernel", "rbf")
        idx = self._kernel_combo.findData(kernel)
        if idx >= 0:
            self._kernel_combo.setCurrentIndex(idx)
        self._mcmc_cb.setChecked(bool(d.get("run_mcmc", False)))
        self._mcmc_draws.setValue(int(d.get("mcmc_draws", 1000)))
        self._apply_global_cb.setChecked(bool(d.get("apply_global", True)))

        # The project lists need refresh_projects() called by the
        # parent after all projects are restored.

        corr_state = d.get("corrector")
        corrections_in_yaml = bool(d.get("corrections"))
        if corr_state:
            try:
                self._corrector = ReferenceCorrector.from_dict(corr_state)
                self._render_diagnostic_plot()
                hp = self._corrector.hyperparameters
                self._status.setText(
                    f"Loaded GP fit ({self._corrector.kernel}; "
                    f"n={len(self._corrector.observations)}). "
                    f"σ_n = {hp.sigma_n:.3g} MHz")
            except Exception as e:  # noqa: BLE001
                self._status.setText(f"Could not restore corrector: {e}")
                self._corrector = None
        elif corrections_in_yaml:
            # Corrections were saved but the corrector wasn't. Without
            # a corrector, _build_corrections_map gates them out at
            # fit time anyway -- so showing the stale table would
            # mislead users into thinking the fit will use them.
            # Drop the corrections and tell the user.
            self._status.setText(
                "Saved corrections from a previous session were "
                "discarded because no GP fit was stored. Re-fit the "
                "GP and click Compute corrections.")
            d = dict(d)
            d["corrections"] = {}

        self._corrections = {}
        for project, fmap in (d.get("corrections") or {}).items():
            inner: dict[str, _FileCorrection] = {}
            for fpath, c in fmap.items():
                mode = str(c.get("mode", _MODE_AUTO))
                # Back-compat: old saves only had value_mhz/sigma_mhz
                # (the mode-resolved view). Map them onto the new
                # auto_ / manual_ pairs by mode so a Manual row's
                # typed numbers don't end up labeled as Auto.
                legacy_v = float(c.get("value_mhz", 0.0))
                legacy_s = float(c.get("sigma_mhz", 0.0))
                auto_v = float(c.get(
                    "auto_value_mhz",
                    legacy_v if mode != _MODE_MANUAL else 0.0))
                auto_s = float(c.get(
                    "auto_sigma_mhz",
                    legacy_s if mode != _MODE_MANUAL else 0.0))
                manual_v = float(c.get(
                    "manual_value_mhz",
                    legacy_v if mode == _MODE_MANUAL else 0.0))
                manual_s = float(c.get(
                    "manual_sigma_mhz",
                    legacy_s if mode == _MODE_MANUAL else 0.0))
                inner[fpath] = _FileCorrection(
                    project=project,
                    run_number=str(c.get("run_number", "")),
                    file_path=fpath,
                    ts_start=float(c.get("ts_start", 0.0)),
                    t_hours=float(c.get("t_hours", 0.0)),
                    mode=mode,
                    auto_value_mhz=auto_v,
                    auto_sigma_mhz=auto_s,
                    manual_value_mhz=manual_v,
                    manual_sigma_mhz=manual_s,
                )
            self._corrections[project] = inner
        self._populate_table()

        # Repopulate the Ref/Sample lists honoring the saved checks.
        self.refresh_projects(restore_state={
            "ref": list(d.get("checked_reference_projects", [])),
            "sam": list(d.get("checked_sample_projects", [])),
        })

    # ── Public read accessors ───────────────────────────────

    @property
    def apply_globally(self) -> bool:
        return self._apply_global_cb.isChecked()

    @property
    def corrector(self) -> ReferenceCorrector | None:
        """The fitted GP corrector, or None when nothing is loaded.
        The merge / fit consumers (merge.py / fitting.py) call
        ``corrector.predict(t/3600)`` to get μ_GP(t) for files that
        weren't in the precomputed per-file table (e.g. newly added
        runs or one-off fits)."""
        return self._corrector

    def get_correction(self, project_name: str, file_path: str
                       ) -> dict | None:
        """Return ``{correction_mhz, sigma_mhz, mode}`` for one run, or
        None if the panel hasn't computed one. Mode ``"Off"`` means
        callers should NOT apply a correction; ``"Auto"``/``"Manual"``
        both yield a usable value (with the global toggle layered on
        top by callers)."""
        fmap = self._corrections.get(project_name)
        if not fmap:
            return None
        fc = fmap.get(file_path)
        if fc is None:
            return None
        return {
            "correction_mhz": fc.value_mhz,
            "sigma_mhz": fc.sigma_mhz,
            "mode": fc.mode,
        }


def _ro_item(text: str) -> QTableWidgetItem:
    """Read-only QTableWidgetItem."""
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts as a number even when the
    displayed text is formatted (e.g. ``%.6g``). Without this Qt
    sorts the column lexicographically and ``"10"`` < ``"2"``.

    Comparisons read both sides from ``text()`` (not a cached
    ``_value``) so Manual-mode edits via the cell editor are
    reflected immediately in subsequent sorts.
    """

    def __init__(self, value: float, *, fmt: str = ".6g",
                 read_only: bool = False):
        super().__init__(format(value, fmt))
        if read_only:
            self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except (ValueError, TypeError):
            return super().__lt__(other)
