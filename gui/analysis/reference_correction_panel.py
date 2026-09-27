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

import os
import re
import sys
import textwrap
from dataclasses import dataclass

from gui.analysis.gp_frame import canonical_obs_key

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtCore import Qt, QElapsedTimer, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QSplitter, QWidget,
    QPushButton, QComboBox, QCheckBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPlainTextEdit, QMessageBox, QProgressBar, QSpinBox, QToolButton,
)

from cls_estimations.reference_correction import (
    ReferenceCorrector, ReferenceObservation, SUPPORTED_KERNELS,
    OBS_CAPSIZE, OBS_CAPTHICK, OBS_ELINEWIDTH, OBS_MARKER_SIZE,
    GP_LEGEND_SIZE, style_gp_axes,
)


_KERNEL_LABELS = {
    "rbf": "RBF + WhiteNoise",
    "matern": "Matérn(5/2) + WhiteNoise",
    "thesis": "Composite (slow RBF + Periodic·Matérn52 + WhiteNoise)",
}

#: Short display names. ``thesis`` is the on-disk kernel key -- it
#: names the composite kernel of van den Borne 2025 and saved projects
#: are keyed by it -- but it says nothing to someone reading a status
#: line, so no user-facing string prints the raw key. Use
#: :func:`_kernel_name` (2026-09-20).
_KERNEL_SHORT = {
    "rbf": "RBF",
    "matern": "Matérn-5/2",
    "thesis": "Composite",
}


def _kernel_name(key: str) -> str:
    """Display name for a kernel key, for status text and reports."""
    return _KERNEL_SHORT.get(key, str(key))


def _wrap_tip(text: str, width: int = 54) -> str:
    """Hard-wrap a tooltip so Qt shows a readable block, not a ribbon.

    QToolTip word-wraps rich text only, and then at up to half the
    screen width, so a paragraph-length tip became a single line
    running most of the way across a wide monitor. Plain-text tooltips
    honour newlines, so filling the text here gives the same compact
    box the short tips elsewhere in the app get for free.
    """
    return "\n".join(
        textwrap.fill(para.strip(), width) if para.strip() else ""
        for para in str(text).split("\n"))


#: ``{"<path>!<mtime>": (ts_start, ts_stop)}``. Reading two cells out
#: of an ASDF is cheap (tens of ms) but not free, and Compute is
#: clicked often.
_TS_START_CACHE: dict[str, float] = {}


def _asdf_ts_start(path: str) -> float:
    """First event timestamp of an ASDF run, in epoch seconds."""
    return _asdf_ts_span(path)[0]


def _asdf_ts_span(path: str) -> tuple[float, float]:
    """``(first, last)`` event timestamp of an ASDF run, in seconds.

    clstools sets ``TSstart`` to the minimum of the event ``TS``
    column; events are written in acquisition order, so that is
    ``raw[0, 0]``. Reading that single cell lazily costs a few tens of
    ms against a full ``Load_Run`` of the whole event array, which is
    what lets a run that has never been fitted still be placed on the
    GP's time axis. Checked equal to the ``ts_start`` merge.py records
    for every run of the Ge campaign.

    The last is needed because a fitted centroid reflects the drift
    averaged over the whole acquisition, not the drift at the instant
    the run began -- see ``ReferenceCorrector.predict_interval``.

    Returns ``(0.0, 0.0)`` for anything unreadable -- a missing file,
    a ``.vasdf`` descriptor (no events of its own; callers pass its
    parent) or a synthetic ``merged://`` path.
    """
    if not path or path.startswith("merged://"):
        return (0.0, 0.0)
    try:
        key = f"{path}!{os.path.getmtime(path)}"
    except OSError:
        return (0.0, 0.0)
    hit = _TS_START_CACHE.get(key)
    if hit is not None:
        return hit
    span = (0.0, 0.0)
    try:
        import asdf
        with asdf.open(path, lazy_load=True) as af:
            raw = af.tree["raw"]
            span = (float(np.asarray(raw[0, 0])),
                    float(np.asarray(raw[-1, 0])))
    except Exception:  # noqa: BLE001
        span = (0.0, 0.0)
    _TS_START_CACHE[key] = span
    return span


#: The "  [m:ss]" the fit clock appends, so a tick can replace its
#: own previous output instead of stacking on it.
_ELAPSED_RE = re.compile(r"\s*\[\d+:\d\d\]$")

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
    #: End of the acquisition, epoch seconds. 0.0 when unknown, in
    #: which case the correction falls back to a point prediction at
    #: ts_start. LAST in the field list on purpose: inserting it
    #: mid-dataclass reassigned every positional construction.
    ts_stop: float = 0.0

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
    #: Emitted when a GP fit finishes, so the systematic scan can
    #: carry on with the next link of its chain. (The interactive
    #: path ignores them; they exist for a caller with no user.)
    gp_fit_done = Signal()
    gp_fit_failed = Signal(str)
    #: (project_name, results, output_config) -- same shape the Isotope
    #: Shifts tab emits, relayed to the Results tab by the IS tab.
    results_ready = Signal(str, list, dict)

    def __init__(self, analysis_tab, parent=None):
        super().__init__("Reference Correction (GP)", parent)
        # As a tab page the box's title lands hard against the tab
        # bar. Both themes already give QGroupBox margin-top: 10px and
        # place the title with subcontrol-origin: margin at the TOP of
        # that margin -- so a larger margin alone only drops the frame
        # line and leaves the title glued to the tabs (the first
        # attempt, 2026-09-21, set 10px and changed nothing). Frame
        # and title move down together: margin +8, title top +8.
        # Scoped by object name, so the theme still draws the rest.
        self.setObjectName("gpPanel")
        self.setStyleSheet(
            "QGroupBox#gpPanel { margin-top: 18px; }"
            "QGroupBox#gpPanel::title { top: 8px; }")
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
        # Labels ("<project>/<run>") the user has ticked off the GP
        # training set. Held by label rather than by index so it
        # survives re-scans, re-fits and reordering.
        self._excluded_obs: set[str] = set()
        # The exclusions in force when the stored GP was fitted. The
        # "re-fit" nag compares against this: a non-empty exclusion
        # set is not stale, a set that DISAGREES with the fit is.
        self._fitted_excluded: set[str] | None = None
        # Set while a list is being repopulated: clearing and refilling
        # a QListWidget emits itemChanged per row, which would other-
        # wise be read as the user unticking every run in turn.
        self._obs_guard = False
        self._build_ui()
        # Auto-refresh when the user creates / closes a project so the
        # ref / sample lists don't go stale -- AnalysisTab is expected
        # to emit this signal.
        if hasattr(analysis_tab, "projects_changed"):
            analysis_tab.projects_changed.connect(self._on_projects_changed)

    # ── UI construction ─────────────────────────────────────

    def _build_ui(self):
        """Controls on the left, plots on the right.

        ``outer`` still names the control column, so every widget
        below reads as it always did; only the two plot widgets go
        right. The splitter handle is kept so the division can be
        dragged and saved like the Isotope Shifts one.
        """
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        # A collapsed pane stays collapsed through every resize; the
        # plot must never be able to reach 0 px (2026-09-22).
        self._main_splitter.setChildrenCollapsible(False)
        root.addWidget(self._main_splitter)

        _left = QWidget()
        outer = QVBoxLayout(_left)
        outer.setContentsMargins(0, 0, 0, 0)
        self._main_splitter.addWidget(_left)

        _right = QWidget()
        plots = QVBoxLayout(_right)
        plots.setContentsMargins(0, 0, 0, 0)
        self._main_splitter.addWidget(_right)
        # The plots take the extra room when the tab is widened; the
        # control column keeps the width its tables need.
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        # Wide enough for the corrections table's seven columns
        # (130+70+70+90+100+100+70) without an inner scrollbar, which
        # is the same reasoning as the Isotope Shifts left column.
        _left.setMinimumWidth(640)
        self._main_splitter.setSizes([660, 1240])

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
        self._kernel_combo.setToolTip(_wrap_tip(
            "Kernel for the GP. RBF is the simplest and fastest; "
            "Matérn handles small jumps; Composite is a 3-term "
            "kernel (slow RBF + diurnal Periodic·Matérn52 + "
            "WhiteNoise) for long campaigns with day/night lock "
            "cycles."))
        top_form.addRow("Kernel:", self._kernel_combo)

        self._mcmc_cb = QCheckBox(
            "Also run MCMC (slow; for diagnostic posterior)")
        self._mcmc_cb.setToolTip(_wrap_tip(
            "Run pm.sample after find_MAP, to see how well the data "
            "pins down the kernel hyperparameters.\n\n"
            "This does NOT change the correction: predict() and "
            "cov() evaluate at the MAP point either way, so every "
            "mu and sigma comes out identical. What you get is a "
            "posterior spread per hyperparameter and an r-hat "
            "convergence check, reported on the status line after "
            "the fit.\n\n"
            "It is slow -- minutes on the pure-Python PyTensor "
            "path -- so leave it off for routine work."))
        top_form.addRow("", self._mcmc_cb)

        self._mcmc_draws = QSpinBox()
        self._mcmc_draws.setRange(100, 50000)
        self._mcmc_draws.setValue(1000)
        self._mcmc_draws.setSingleStep(100)
        self._mcmc_draws.setToolTip(_wrap_tip(
            "How many posterior samples pm.sample keeps after its "
            "500 tuning steps. More draws resolve the hyperparameter "
            "posterior more finely and take proportionally longer.\n\n"
            "Does nothing unless 'Also run MCMC' is ticked, and "
            "never changes the correction itself -- the drift curve "
            "is always evaluated at the MAP point."))
        self._mcmc_draws.setEnabled(self._mcmc_cb.isChecked())
        self._mcmc_cb.toggled.connect(self._mcmc_draws.setEnabled)
        top_form.addRow("MCMC draws:", self._mcmc_draws)

        self._apply_global_cb = QCheckBox("Apply correction at fit time")
        self._apply_global_cb.setChecked(True)
        self._apply_global_cb.setToolTip(_wrap_tip(
            "Global toggle: when off, downstream merging/fitting "
            "ignores all per-file corrections regardless of mode."))
        top_form.addRow("", self._apply_global_cb)

        outer.addLayout(top_form)

        # ── Reference projects ──
        outer.addWidget(QLabel("Reference projects:"))
        self._ref_list = QListWidget()
        self._ref_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._ref_list.setMaximumHeight(110)
        self._ref_list.itemChanged.connect(self._on_ref_item_changed)
        outer.addWidget(self._ref_list)

        # ── Reference runs (per-observation opt-out) ──
        self._obs_lbl = QLabel("Reference runs:")
        outer.addWidget(self._obs_lbl)
        self._obs_list = QListWidget()
        self._obs_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._obs_list.setMaximumHeight(150)
        self._obs_list.setToolTip(_wrap_tip(
            "Every fitted centroid from the ticked reference "
            "projects. Untick a run to keep it out of the GP "
            "training set -- for a run whose calibration or fit you "
            "do not trust. It stays on the plot, drawn grey, so the "
            "exclusion is visible. Re-fit the GP for it to take "
            "effect."))
        self._obs_list.itemChanged.connect(self._on_obs_item_changed)
        outer.addWidget(self._obs_list)

        ref_btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh project list")
        refresh_btn.setToolTip(
            "Re-scan the Analysis tab for Reference and Sample projects.")
        refresh_btn.clicked.connect(self.refresh_projects)
        ref_btn_row.addWidget(refresh_btn)
        self._fit_btn = QPushButton("Fit GP")
        self._fit_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 4px 12px; }")
        self._fit_btn.setToolTip(_wrap_tip(
            "Train the Gaussian process on the fitted centroids of the "
            "ticked reference projects. Each successful reference-run "
            "fit becomes one observation; the result is the drift curve "
            "plotted below."))
        self._fit_btn.clicked.connect(self._fit_clicked)
        ref_btn_row.addWidget(self._fit_btn)
        self._send_btn = QPushButton("Send to Results")
        self._send_btn.setToolTip(_wrap_tip(
            "Write the GP diagnostic plot to the Results tab (as a live, "
            "re-renderable plot plus a PNG and a CSV of the reference "
            "centroids). Enabled once a GP has been fitted."))
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_plot_to_results)
        # Lives on the plot row now (see the view controls): it
        # sends the view, so it belongs next to it.
        ref_btn_row.addStretch()
        outer.addLayout(ref_btn_row)

        self._status = QLabel("(no GP fit yet)")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)
        # The GP runs off the GUI thread and takes 20-90 s (minutes on
        # the pure-Python PyTensor fallback). Without a moving bar the
        # only sign it is working is a static status line, which is
        # indistinguishable from a fit that never started.
        self._fit_progress = QProgressBar()
        self._fit_progress.setRange(0, 0)      # indeterminate: no ETA
        self._fit_progress.setTextVisible(False)
        self._fit_progress.setMaximumHeight(8)
        self._fit_progress.setVisible(False)
        outer.addWidget(self._fit_progress)
        # find_MAP exposes no iteration count, so the bar can only
        # bounce; the ticking clock is what actually answers "is it
        # still going?" on a multi-minute pure-Python fit.
        self._fit_clock = QElapsedTimer()
        self._fit_tick = QTimer(self)
        self._fit_tick.setInterval(1000)
        self._fit_tick.timeout.connect(self._update_fit_elapsed)

        # ── Diagnostic plot ──
        self._show_corrected = QCheckBox("Corrected centroids")
        self._show_corrected.setChecked(True)
        self._show_corrected.setToolTip(_wrap_tip(
            "Add a second panel showing every fitted run of every "
            "project in the drift-free frame, one colour per "
            "isotope. This is the check that the correction worked: "
            "the reference isotope should scatter about zero, and "
            "each sample should sit at its isotope shift. A "
            "reference sitting down on the drift curve instead of on "
            "the zero line means it was not corrected."))
        self._show_corrected.toggled.connect(self._render_diagnostic_plot)

        self._show_excluded = QCheckBox("Excluded points")
        self._show_excluded.setChecked(True)
        self._show_excluded.setToolTip(_wrap_tip(
            "Draw the ticked-off runs as grey crosses. Turn it off "
            "for the publication view: only the points the curve was "
            "actually fitted to. It changes the picture, never the "
            "fit."))
        self._show_excluded.toggled.connect(self._render_diagnostic_plot)

        self._show_residuals = QCheckBox("Residuals (σ)")
        self._show_residuals.setToolTip(_wrap_tip(
            "Add a panel of standardised residuals, "
            "(y − μ) / √(σ² + σ_n²), "
            "where σ_n is the fitted white-noise term: the pull "
            "of each measurement against the curve. Scatter well "
            "outside ±1 means the kernel cannot follow the "
            "drift; everything crushed inside ±0.2 means it is "
            "chasing noise."))
        self._show_residuals.toggled.connect(self._render_diagnostic_plot)

        self._t_unit = QComboBox()
        self._t_unit.addItem("hours", 1.0)
        self._t_unit.addItem("minutes", 60.0)
        self._t_unit.setToolTip(_wrap_tip(
            "X-axis unit for every panel. A display choice only -- "
            "the GP is always fitted in hours, so switching costs "
            "nothing and never needs a re-fit."))
        self._t_unit.currentIndexChanged.connect(
            self._render_diagnostic_plot)

        self._show_ref_avg = QCheckBox("Ref. average")
        self._show_ref_avg.setToolTip(_wrap_tip(
            "Draw a dashed line in the corrected panel at the "
            "reference isotope's inverse-variance weighted average, "
            "excluded runs left out. In the drift-free frame the "
            "reference belongs at 0, so this line's offset from 0 is "
            "how far the GP mean sits from the reference data it was "
            "fitted to."))
        self._show_ref_avg.toggled.connect(self._render_diagnostic_plot)

        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(self._show_corrected)
        view_row.addWidget(self._show_ref_avg)
        view_row.addWidget(self._show_excluded)
        view_row.addWidget(self._show_residuals)
        view_row.addStretch()
        view_row.addWidget(QLabel("Time:"))
        view_row.addWidget(self._t_unit)
        view_row.addWidget(self._send_btn)
        plots.addLayout(view_row)
        self._figure = Figure(figsize=(6.5, 6.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(340)
        self._canvas.setMouseTracking(True)
        self._canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        plots.addWidget(self._canvas, 1)

        # ── Sample projects ──
        _sample_lbl = QLabel("Apply to sample projects:")
        _sample_lbl.setToolTip(_wrap_tip(
            "The reference projects above are corrected too, without "
            "being listed here -- every centroid that reaches the "
            "isotope-shift table has to sit in the same drift-free "
            "frame."))
        outer.addWidget(_sample_lbl)
        self._sample_list = QListWidget()
        self._sample_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._sample_list.setMaximumHeight(110)
        outer.addWidget(self._sample_list)

        sample_btn_row = QHBoxLayout()
        compute_btn = QPushButton("Compute per-file corrections")
        compute_btn.setToolTip(_wrap_tip(
            "Evaluate the fitted GP at each run of the ticked sample "
            "projects and fill the table below with the drift "
            "correction for that run. The runs do NOT have to be "
            "fitted first -- a correction only needs each run's "
            "timestamp -- so this can be done before merging, which is "
            "what enables 'Align centroids before merging' in the "
            "merge dialog.\n\n"
            "The ticked REFERENCE projects are corrected as well, "
            "automatically. A corrected sample centroid is already "
            "its shift against the reference, so leaving the "
            "reference on its raw scale would subtract the reference "
            "level a second time -- and make every isotope shift "
            "depend on which reference run you picked as Ref."))
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
        _col_tips = [
            "The sample project this run belongs to.",
            "Run number of the measurement being corrected.",
            "When this run was taken, in hours since the FIRST "
            "measurement of the whole campaign. This is the GP's time "
            "axis -- the correction below is the drift the GP predicts "
            "at this moment.",
            "How this run's correction is chosen. "
            "Auto = use the GP prediction at t. "
            "Manual = type your own value and uncertainty (e.g. a "
            "published number). "
            "Off = leave this run uncorrected and exclude it from the "
            "correction budget.",
            "The drift correction itself, in MHz. It is SUBTRACTED from "
            "this run's frequency axis after Doppler binning, so a "
            "fitted centroid is reported in the drift-free reference "
            "frame. Editable only in Manual mode.",
            "Uncertainty on that correction, in MHz (the GP's 1-sigma "
            "at this time). It propagates into the sigma_corr column of "
            "the isotope-shift table rather than into the fit itself. "
            "Editable only in Manual mode.",
            "How large the correction is compared with its own "
            "uncertainty. Below 2 the drift is barely resolved; the row "
            "turns amber above 2 and red above 3, where the correction "
            "is significant and worth sanity-checking against the "
            "diagnostic plot.",
        ]
        for _c, _tip in enumerate(_col_tips):
            _hi = self._corr_table.horizontalHeaderItem(_c)
            if _hi is not None:
                _hi.setToolTip(_wrap_tip(_tip))
        self._corr_table.setToolTip(_wrap_tip(
            "One row per run of the ticked sample projects -- including "
            "runs that have not been fitted yet, since a correction "
            "only needs the run's timestamp. Each row says how far the "
            "reference drifted when that run was taken, and therefore "
            "how much is subtracted from its frequency axis. Hover a "
            "column header for details."))
        self._corr_table.verticalHeader().setVisible(False)
        self._corr_table.setMinimumHeight(150)
        self._corr_table.itemChanged.connect(self._table_item_changed)
        # Click-to-sort: clicking a column header sorts on that
        # column. Internal repopulates disable sorting briefly to
        # avoid row-index reshuffling while inserting (see
        # _populate_table).
        self._corr_table.setSortingEnabled(True)
        outer.addWidget(self._corr_table, 1)

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
                    "  ⚠ No C compiler (g++) on PATH, so PyMC "
                    "computes in pure Python: same answer, a few "
                    "times slower for a MAP fit and much slower for "
                    "MCMC. Harmless -- see the ? button.")
        except Exception:  # noqa: BLE001
            # PyTensor not importable means PyMC isn't installed
            # either; ReferenceCorrector.fit will surface that
            # error itself, no need to duplicate here.
            pass
        if _probe is None:
            cls._cxx_note_cache = note
        return note

    @staticmethod
    def _help_html():
        """The guide text, kept separate from the dialog so it
        can be read back and asserted on."""
        return (
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
            "<li><b>Untick any bad reference runs</b> in the "
            "Reference runs list. One unreliable centroid drags the "
            "whole curve: the GP has no notion of an outlier, so a "
            "run sitting 200 MHz off its neighbours is fitted rather "
            "than discounted, and the slow amplitude inflates to "
            "cover it. An excluded run stays on the plot as a grey "
            "cross, so the figure records the decision. Typical "
            "reasons: an invalid calibration stream, a failed lock, "
            "a fit you do not trust.</li>"
            "<li><b>Click Fit GP.</b> The diagnostic plot below shows "
            "scatter of the reference centroids, the MAP curve, and "
            "1σ/2σ bands.</li>"
            "<li><b>Tick the Sample Project(s)</b> you want corrected, "
            "click Compute per-file corrections. The table populates "
            "with μ_GP and σ_GP for each run. The sample runs do "
            "<i>not</i> have to be fitted first -- a correction only "
            "needs the run's timestamp, so you can do this before "
            "merging or fitting anything.</li>"
            "<li><b>Mode</b> per row: Auto = use μ_GP. Manual = type "
            "your own μ and σ (e.g. a published value). Off = skip "
            "this run.</li>"
            "<li><b>Re-fit the Sample Project.</b> The fit reads this "
            "table and subtracts the correction from the binned "
            "frequency axis. Console logs which files got corrected.</li>"
            "</ol>"
            "<h4>Merging corrected runs</h4>"
            "<p>The merge dialog's <b>Align centroids before merging</b> "
            "box only lights up once this table has a row for every "
            "file going into the merge -- it shifts each file onto the "
            "drift-free frame <i>before</i> they are histogrammed "
            "together, which is what stops the drift smearing the "
            "merged peak. So: fit the GP, compute the corrections, "
            "then merge. A merged entry has no row of its own; it "
            "inherits the corrections of the runs inside it.</p>"
            "<h4>Reading the table</h4>"
            "<p>The <b>|μ|/σ</b> column flags rows where the GP "
            "thinks the run is far from the smooth drift it learned. "
            "Amber (>2σ) is worth a look; red (>3σ) usually means "
            "either the run is genuinely off, or your kernel is too "
            "smooth to capture the local behavior. Off rows are "
            "shown grey so you can see at a glance which files are "
            "skipped.</p>"
            "<h4>The \"g++ not detected\" warning</h4>"
            "<p>PyMC does its arithmetic through PyTensor, which "
            "normally compiles the model to C. With no compiler it "
            "falls back to pure Python. <b>The answer is "
            "identical</b> either way &mdash; nothing is wrong with "
            "your data and nothing needs re-doing.</p>"
            "<p>How much slower depends entirely on what you run. "
            "Measured on a 13-run Composite fit, MAP took 17.5 s in "
            "pure Python against 7-11 s compiled: about 2x, because "
            "a GP this small spends most of its time in SciPy's "
            "optimiser and NumPy linear algebra, which are compiled "
            "on both paths. <b>MCMC is what really pays</b>, since "
            "it evaluates the model thousands of times &mdash; that "
            "is the combination that can look like a hang. Note too "
            "that the first fit after installing a compiler is slow "
            "(~110 s here) while PyTensor builds and caches the C "
            "code; later fits are fast.</p>"
            "<p><b>It has to be g++.</b> PyTensor runs "
            "<code>g++ -v</code> and, failing that, looks inside the "
            "Python environment at "
            "<code>Scripts\\Library\\mingw-w64\\bin\\g++</code>. It "
            "has no MSVC code path at all, so <i>Visual Studio and "
            "the MSVC Build Tools will not satisfy it</i> no matter "
            "how complete the install.</p>"
            "<p>On Windows, either of these puts MinGW-w64 g++ on "
            "PATH:</p>"
            "<ul>"
            "<li><code>choco install mingw</code> (admin shell)</li>"
            "<li><code>winget install "
            "BrechtSanders.WinLibs.POSIX.UCRT</code></li>"
            "</ul>"
            "<p>Restart DENIS afterwards: the check runs once per "
            "session and the result is cached.</p>"
            "<h4>Save / load</h4>"
            "<p>The trained GP, the table, and the per-row Mode "
            "choices all round-trip through the project YAML. "
            "Saving without first fitting the GP is fine; the table "
            "is empty on reload and you'll be told to refit.</p>")

    def _show_help(self):
        """Show a plain-language workflow guide for the GP-correction
        panel, written for a CLS physicist who has not read the thesis."""
        QMessageBox.information(
            self, "Reference Correction — guide",
            self._help_html())

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

        # Clearing and refilling emits itemChanged per row; without
        # the guard each one would be read as the user reticking a
        # project and would rebuild the run list N times over.
        self._obs_guard = True
        self._ref_list.clear()
        self._sample_list.clear()
        for p in self._analysis_tab._projects:
            # Cooler Calibration projects are neither a drift reference
            # nor a sample to correct.
            if getattr(p, "is_calibration", False):
                continue
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
        self._obs_guard = False
        self._refresh_obs_list()

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
        all_obs = self._gather_observations()
        n_inc = sum(1 for o in all_obs if o.include)
        if n_inc < 2:
            # Distinguish "you have not fitted anything" from "you
            # have ticked almost everything off", which are different
            # mistakes with different remedies.
            if all_obs and n_inc < len(all_obs):
                QMessageBox.warning(
                    self, "Not enough observations",
                    f"Need ≥ 2 reference observations to fit the "
                    f"GP, but only {n_inc} of {len(all_obs)} are "
                    f"ticked in the Reference runs list.")
            else:
                QMessageBox.warning(
                    self, "Not enough observations",
                    f"Need ≥ 2 reference observations to fit the "
                    f"GP (got {len(all_obs)}). Re-run the reference "
                    f"project's fit pipeline first.")
            return

        kernel = self._kernel_combo.currentData() or "rbf"
        run_mcmc = self._mcmc_cb.isChecked()
        n_samples = self._mcmc_draws.value()
        # MAP alone on the pure-Python backend is slow but finishes.
        # MAP + several thousand MCMC draws on it is the case that
        # looks like a hang, so say so before the user waits it out.
        if run_mcmc and self._pytensor_cxx_note():
            if QMessageBox.question(
                    self, "This will take a while",
                    f"MCMC is on ({n_samples} draws) and there is "
                    f"no C compiler, so PyMC is running in pure "
                    f"Python. MCMC evaluates the model thousands of "
                    f"times, which is exactly the case a missing "
                    f"compiler hurts most. Expect many minutes, and "
                    f"the window will be sluggish throughout.\n\n"
                    f"MCMC only adds posterior spreads for the "
                    f"hyperparameters; it does not change the "
                    f"correction. Untick it for a fit that takes "
                    f"seconds.\n\nRun it anyway?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return
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
        base = (f"Fitting GP ({_kernel_name(kernel)} kernel, "
                f"{len(all_obs)} obs)... this may take 20-90 s.")
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
        self._quiet = False
        self.gp_fit_done.emit()
        hp = rc.hyperparameters
        self._status.setText(
            f"GP fit OK ({_kernel_name(rc.kernel)} kernel; "
            f"n={len(rc.observations)}). "
            f"σ_n = {hp.sigma_n:.3g} MHz" + (
                f", ℓ = {hp.ell:.3g}, η = {hp.eta:.3g}"
                if rc.kernel in ("rbf", "matern") else
                f", ℓ_slow = {hp.length_slow:.3g}, "
                f"P_fast = {hp.period_fast:.3g}"
            ))
        summary = getattr(rc, "mcmc_summary", None)
        if summary:
            # Report the spread the sampler found and the worst
            # r_hat. Without this the MCMC run produced nothing the
            # user could see (2026-09-21).
            import math
            bits, worst_rhat = [], 0.0
            for name, row in summary.items():
                sd = row.get("sd")
                if sd is not None and math.isfinite(sd):
                    bits.append(f"{name} ±{sd:.3g}")
                rh = row.get("r_hat")
                if rh is not None and math.isfinite(rh):
                    worst_rhat = max(worst_rhat, rh)
            tail = ", ".join(bits[:4])
            if tail:
                self._status.setText(
                    self._status.text()
                    + f"  MCMC posterior sd: {tail}"
                    + (f"; max r\u0302 = {worst_rhat:.3f}"
                       if worst_rhat else ""))
            lines = ["MCMC posterior (diagnostic only -- the "
                     "correction uses the MAP point):", ""]
            for name, row in summary.items():
                lines.append(
                    f"{name:>12}  mean {row.get('mean', float('nan')):.6g}"
                    f"  sd {row.get('sd', float('nan')):.4g}"
                    f"  95% HDI [{row.get('hdi_2.5%', float('nan')):.6g},"
                    f" {row.get('hdi_97.5%', float('nan')):.6g}]"
                    f"  r-hat {row.get('r_hat', float('nan')):.3f}")
            self._status.setToolTip("\n".join(lines))
        else:
            self._status.setToolTip("")

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
        # The stored GP and the tick list now agree again, so drop
        # the "re-fit the GP" nag.
        self._fitted_excluded = set(self._excluded_obs)
        self._update_obs_label()

    # ── driving the panel from the systematic scan ──────────
    def start_gp_fit_for_scan(self):
        """Refit the GP on the reference project's current results.

        The cooler offset acts on the voltages, before the Doppler
        conversion; this correction acts in the rest frame after it.
        So every assumed offset has its own reference centroids and
        its own drift model -- carrying one offset's GP over to
        another would correct the samples into a frame that does not
        belong to them.

        MAP only, whatever the MCMC box says: the sampler adds
        posterior spreads for the hyperparameters and does not change
        the correction, and a scan cannot afford it once per offset.

        Returns True when a fit started; the caller then waits for
        :attr:`gp_fit_done` or :attr:`gp_fit_failed`.
        """
        if self._fit_busy:
            return False
        all_obs = self._gather_observations()
        if sum(1 for o in all_obs if o.include) < 2:
            return False
        kernel = self._kernel_combo.currentData() or "rbf"
        rc = ReferenceCorrector(kernel=kernel, run_mcmc=False)
        self._quiet = True
        self._fit_busy = True
        self._set_busy(True)
        self._status.setText(
            f"Refitting GP for the systematic scan "
            f"({_kernel_name(kernel)}, {len(all_obs)} obs)...")
        self._fit_worker = _FitWorker(rc, all_obs, self)
        self._fit_worker.done.connect(self._on_fit_done)
        self._fit_worker.failed.connect(self._on_fit_failed)
        self._fit_worker.finished.connect(self._fit_worker.deleteLater)
        self._fit_worker.start()
        return True

    def recompute_corrections(self):
        """Re-evaluate the per-file corrections from the current GP.

        The headless half of "Compute per-file corrections": same
        work, no dialogs and no wait cursor, for the scan to call
        between refitting the GP and fitting the samples.
        """
        if self._corrector is None or not self._corrector.is_fit:
            return 0
        wanted = (self._checked_names(self._sample_list)
                  | self._checked_names(self._ref_list))
        new_corrections, _no_runs, _no_ts = self._collect_corrections(
            wanted)
        self._corrections = new_corrections
        self._populate_table()
        self._emit_corrections_for_all()
        return sum(len(v) for v in new_corrections.values())

    def _on_plot_hover(self, event):
        """Name the nearest plotted point under the cursor.

        Hit-tested in DISPLAY space, so the tolerance is a constant
        number of pixels whatever the axes are zoomed to -- a data
        tolerance would grow and shrink with the view.
        """
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QToolTip
        if event.inaxes is None or not getattr(self, "_hover_points", None):
            QToolTip.hideText()
            return
        best, best_d2 = None, 14.0 ** 2       # pixels
        for ax, x, y, text in self._hover_points:
            if ax is not event.inaxes:
                continue
            px, py = ax.transData.transform((x, y))
            d2 = (px - event.x) ** 2 + (py - event.y) ** 2
            if d2 < best_d2:
                best, best_d2 = text, d2
        if best is None:
            QToolTip.hideText()
            return
        QToolTip.showText(QCursor.pos(), best, self._canvas)

    def _send_plot_to_results(self):
        """Write the GP diagnostic plot into the Results tab.

        Same artifact layout the Isotope Shifts tab uses (project dir +
        iter_NNN + plots/), so the Results tree, the Plot Editor and
        Export All all work on it unchanged. The .npz carries
        plot_type="gp_reference" for live re-rendering.
        """
        import csv
        import os

        if self._corrector is None:
            QMessageBox.information(
                self, "Reference Correction",
                "Fit the GP first.")
            return
        try:
            arrays = self._corrector.diagnostic_arrays(n_grid=400)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Reference Correction",
                                f"Could not build the plot data:\n{e}")
            return

        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()
        project_name = "GP_Reference_Correction"
        project_dir = os.path.join(base_dir, project_name)

        iter_num = 1
        if os.path.isdir(project_dir):
            existing = sorted(d for d in os.listdir(project_dir)
                              if os.path.isdir(os.path.join(project_dir, d)))
            if existing:
                try:
                    iter_num = int(existing[-1].split("_")[-1]) + 1
                except ValueError:
                    iter_num = len(existing) + 1
        iter_name = f"iter_{iter_num:03d}"
        iter_dir = os.path.join(project_dir, iter_name)
        plots_dir = os.path.join(iter_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        rc = self._corrector
        hp = rc.hyperparameters
        n_fit = sum(1 for o in rc.observations if o.include)
        n_off = len(rc.observations) - n_fit
        lines = [
            "Reference centroid correction (Gaussian process)",
            "=" * 52,
            f"Kernel:        {_kernel_name(rc.kernel)}",
            f"Observations:  {n_fit} fitted"
            + (f", {n_off} excluded" if n_off else ""),
            f"sigma_n:       {hp.sigma_n:.6g} MHz",
        ]
        if rc.kernel in ("rbf", "matern"):
            lines += [f"ell:           {hp.ell:.6g}",
                      f"eta:           {hp.eta:.6g}"]
        else:
            lines += [f"ell_slow:      {hp.length_slow:.6g}",
                      f"P_fast:        {hp.period_fast:.6g}"]
        lines += ["",
                  "Reference centroids used to train the GP are in "
                  "reference_centroids.csv."]
        report = "\n".join(lines)
        with open(os.path.join(iter_dir, "fit_report.txt"), "w",
                  encoding="utf-8") as f:
            f.write(report)

        csv_path = os.path.join(iter_dir, "reference_centroids.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_hours_since_first", "centroid_mhz", "sigma_mhz"])
            for t, y, e in zip(arrays["t_train"], arrays["y_train"],
                               arrays["yerr_train"]):
                w.writerow([f"{t:.6f}", f"{y:.6f}", f"{e:.6f}"])

        png_path = os.path.join(plots_dir, "gp_reference_correction.png")
        try:
            self._figure.savefig(png_path, dpi=150, bbox_inches="tight")
        except Exception:
            pass
        # The dict the panel just drew, so the Results tab re-draws
        # exactly this view: unit, residuals, excluded points and all.
        d = self._plot_data()
        d["title"] = (f"GP reference correction "
                      f"({_kernel_name(rc.kernel)}, n={n_fit})")
        npz_path = os.path.join(plots_dir, "gp_reference_correction.npz")
        np.savez(npz_path, plot_type="gp_reference", **d)

        results = [{
            "success": True, "run_number": "GP_reference",
            "run_file": "", "report": report,
            "params_df": {}, "metadata_df": {},
            "x": [], "y": [], "yerr": [],
            "y_fit": [], "x_smooth": [], "y_fit_smooth": [],
            "residuals": [], "diagnostics": {}, "fwhm": {},
            "peak_positions": {}, "fit_quality": {},
            "run_metadata": {}, "harmonic": 0,
        }]
        output_config = {
            "report": True, "params_csv": False, "metadata_csv": False,
            "fit_plots": True, "iter_label": iter_name,
            "iter_name": iter_name,
        }
        self.results_ready.emit(project_name, results, output_config)
        QMessageBox.information(
            self, "Reference Correction",
            f"GP plot saved to:\n{iter_dir}\n\n"
            f"Also sent to the Results tab as '{project_name}'.")

    def _fit_failed(self, msg: str):
        self._set_busy(False)
        self._corrector = None
        if hasattr(self, "_send_btn"):
            self._send_btn.setEnabled(False)
        self._status.setText(f"GP fit failed: {msg}")
        # Stale plot would mislead the user about the current state.
        self._figure.clear()
        self._canvas.draw()
        quiet = getattr(self, "_quiet", False)
        self._quiet = False
        self.gp_fit_failed.emit(str(msg))
        if quiet:
            # Driven by the systematic scan: a modal box here stops
            # the scan dead, waiting for a click nobody will give.
            return
        QMessageBox.critical(self, "Fit error", msg)

    def _set_busy(self, busy: bool):
        self._fit_btn.setEnabled(not busy)
        self._kernel_combo.setEnabled(not busy)
        # The fit is already off the GUI thread, so an unresponsive
        # window here is the GIL, not a blocked event loop: the
        # pure-Python PyTensor path is interpreted end to end and the
        # worker holds the interpreter between switch intervals (5 ms
        # by default). Shortening it while fitting costs a few percent
        # of throughput and buys a window that keeps repainting.
        if busy:
            if getattr(self, "_old_switch", None) is None:
                self._old_switch = sys.getswitchinterval()
                sys.setswitchinterval(0.0005)
        elif getattr(self, "_old_switch", None) is not None:
            sys.setswitchinterval(self._old_switch)
            self._old_switch = None
        if hasattr(self, "_fit_progress"):
            self._fit_progress.setVisible(busy)
        if hasattr(self, "_fit_tick"):
            if busy:
                self._fit_clock.restart()
                self._fit_tick.start()
            else:
                self._fit_tick.stop()

    def _update_fit_elapsed(self):
        """Append a mm:ss counter to the status line while fitting.

        The counter is stripped off and re-appended each tick rather
        than added to a snapshot, so _set_busy can be called before
        the status text is final (it disables the Fit button, which
        has to happen before the processEvents() that paints it) and
        the counters still never stack.
        """
        secs = int(self._fit_clock.elapsed() / 1000)
        base = _ELAPSED_RE.sub("", self._status.text())
        self._status.setText(f"{base}  [{secs // 60}:{secs % 60:02d}]")

    def _render_diagnostic_plot(self):
        # Single choke point for "a GP plot now exists" -- reached from
        # _fit_done and from_dict alike, so the Send button follows.
        if hasattr(self, "_send_btn"):
            self._send_btn.setEnabled(self._corrector is not None)
        if self._corrector is None:
            return
        from gui.analysis import gp_figure
        self._figure.clear()
        # (axes, x, y, text) per plotted point, for the hover tip.
        self._hover_points = []
        try:
            d = self._plot_data()
            axes = gp_figure.draw(self._figure, d)
            self._hover_points = self._hover_from(d, axes)
        except Exception as e:  # noqa: BLE001
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot error:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
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
        # The reference projects get corrections too, without being
        # ticked anywhere: a corrected sample centroid is already
        # nu_A - nu_ref(t_A), so leaving the reference on its raw
        # scale made the isotope-shift subtraction take the reference
        # level off twice -- and made the answer depend on which
        # reference run was picked as the Ref row (2026-09-20).
        wanted = set(sample_names) | self._checked_names(self._ref_list)

        # Dating a run that was never fitted means opening its ASDF
        # (one lazy cell, ~20 ms, cached) -- unnoticeable for a
        # handful of runs, a visible pause for a campaign-sized
        # project, so say that something is happening.
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            new_corrections, skipped_no_runs, skipped_no_ts = (
                self._collect_corrections(wanted))
        finally:
            QApplication.restoreOverrideCursor()
            QCoreApplication.processEvents()

        self._corrections = new_corrections
        self._populate_table()
        self._emit_corrections_for_all()
        self._report_correction_skips(
            skipped_no_runs, skipped_no_ts)

    def _collect_corrections(self, wanted_names):
        """Evaluate the GP once per dated run of the named projects.

        ``wanted_names`` covers the ticked sample projects AND the
        ticked reference projects -- every centroid that will appear
        in the isotope-shift table has to end up in the same
        drift-free frame, or the shift subtracts the reference level
        twice.

        Returns ``(corrections, projects_with_no_dated_run,
        runs_that_could_not_be_dated)``.
        """
        new_corrections: dict[str, dict[str, _FileCorrection]] = {}
        skipped_no_runs: list[str] = []
        skipped_no_ts: list[str] = []
        for p in self._analysis_tab._projects:
            if p.project_name not in wanted_names:
                continue
            stamps, no_ts = self._run_timestamps(p)
            skipped_no_ts.extend(
                f"{p.project_name}/{lbl}" for lbl in no_ts)
            if not stamps:
                skipped_no_runs.append(p.project_name)
                continue
            project_map: dict[str, _FileCorrection] = {}
            # Chronological, so the table reads as a timeline by
            # default (the user can still click any column to sort).
            for fpath, (ts_start, ts_stop, run_num) in sorted(
                    stamps.items(), key=lambda kv: kv[1][0]):
                t_hours = ts_start / 3600.0
                # The fitted centroid reflects the drift averaged over
                # the whole acquisition, so subtract the GP averaged
                # over the same window rather than its value at the
                # instant the run began (2026-09-21). Worth ~2 MHz on
                # a 5-minute run and ~4.4 MHz on a 33-minute one,
                # against 4-6 MHz statistical errors.
                mu_v, sig_v = self._corrector.predict_interval(
                    t_hours, ts_stop / 3600.0 if ts_stop else t_hours)
                fc = _FileCorrection(
                    project=p.project_name,
                    run_number=run_num,
                    file_path=fpath,
                    ts_start=ts_start,
                    ts_stop=ts_stop,
                    t_hours=t_hours,  # rebased below
                    mode=_MODE_AUTO,
                    auto_value_mhz=mu_v,
                    auto_sigma_mhz=sig_v,
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
        return new_corrections, skipped_no_runs, skipped_no_ts

    def _report_correction_skips(self, skipped_no_runs, skipped_no_ts):
        """Tell the user about anything Compute could not place on the
        GP's time axis. Silent when everything landed."""
        msgs = []
        if skipped_no_runs:
            msgs.append(
                "No runs with a readable timestamp -- add ASDF files to "
                "the project's Source block first: "
                + ", ".join(skipped_no_runs))
        if skipped_no_ts:
            msgs.append(
                "Skipped (no start timestamp): "
                + ", ".join(skipped_no_ts[:8])
                + (f" (+{len(skipped_no_ts) - 8} more)"
                   if len(skipped_no_ts) > 8 else "")
                + ". The GP places a run on its time axis by the "
                "first event's timestamp; these files could not be "
                "read, or carry no events.")
        if msgs:
            QMessageBox.information(
                self, "Corrections computed", "\n\n".join(msgs))

    def _corrected_centroids(self):
        """Every fitted run of every project in the drift-free frame.

        ``{label: {"t", "y", "yerr", "run", "raw", "g", "fly",
        "excl", "is_ref"}}``: ``y`` is the corrected centroid, ``raw``
        the one before any GP correction, ``g`` the drift level it was
        corrected by, ``fly`` whether that happened here rather than
        at fit time, ``excl`` whether the run is excluded from the GP.

        Computed through gui.analysis.gp_frame rather than read off
        params_df. The old version assumed the fit had already
        subtracted the GP; when it had not -- none of the 29 T02 runs
        had -- the panel labelled "Corrected" was plotting raw
        centroids. Times are rebased on the GP's first training point
        so the panels share one x axis.

        Returns ``{}`` when no GP is loaded -- the corrected frame is
        defined by it.
        """
        rc = self._corrector
        if rc is None or not rc.is_fit:
            return {}
        from cls_estimations.isotope_shift import extract_centroid
        from gui.analysis.gp_frame import frame_centroid, observation_key
        import math

        t0_h = float(getattr(rc, "_t0", 0.0))
        out: dict[str, dict] = {}
        for p in getattr(self._analysis_tab, "_projects", []) or []:
            label = p.project_name
            if (not getattr(p, "_last_results", None)
                    and hasattr(p, "load_results_from_disk")):
                try:
                    p.load_results_from_disk()
                except Exception:  # noqa: BLE001
                    pass
            is_ref = bool(getattr(p, "is_reference", False))
            for r in getattr(p, "_last_results", None) or []:
                if not r.get("success"):
                    continue
                c, sig = extract_centroid(
                    r.get("params_df", {}) or {},
                    source_name=r.get("source_name"))
                if c is None or not math.isfinite(c):
                    continue
                meta = r.get("run_metadata") or {}
                try:
                    ts = float(meta.get("ts_start", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if ts <= 0:
                    continue
                fr = frame_centroid(rc, p, r, float(c))
                if fr["corrected"] is None:
                    continue
                d = out.setdefault(label, {
                    "t": [], "y": [], "yerr": [], "run": [],
                    "raw": [], "g": [], "fly": [], "excl": [],
                    "is_ref": is_ref})
                d["t"].append(ts / 3600.0 - t0_h)
                d["y"].append(float(fr["corrected"]))
                d["yerr"].append(
                    float(sig) if sig and math.isfinite(sig) else 0.0)
                d["run"].append(str(r.get("run_number", "") or "?"))
                d["raw"].append(float(fr["raw"]))
                d["g"].append(float(fr["g"]))
                d["fly"].append(bool(fr["on_the_fly"]))
                d["excl"].append(
                    is_ref and observation_key(p, r) in self._excluded_obs)
        return {k: v for k, v in out.items() if v["t"]}

    def _plot_data(self):
        """Everything the figure shows, as flat arrays.

        The same dict is drawn here, saved by Send to Results and
        re-drawn by the Results tab, so the sent figure is the one on
        screen by construction (gui/analysis/gp_figure.py).
        """
        rc = self._corrector
        arr = rc.diagnostic_arrays(n_grid=300)
        scale = float(self._t_unit.currentData() or 1.0)
        obs = list(rc.observations)
        inc = [o for o in obs if o.include]
        exc = [o for o in obs if not o.include]
        d = {
            "t_train": arr["t_train"], "y_train": arr["y_train"],
            "yerr_train": arr["yerr_train"],
            "t_grid": arr["t_grid"], "mu": arr["mu"],
            "sigma": arr["sigma"],
            "t_excl": arr.get("t_excluded", np.zeros(0)),
            "y_excl": arr.get("y_excluded", np.zeros(0)),
            "yerr_excl": arr.get("yerr_excluded", np.zeros(0)),
            "train_labels": np.array([o.label for o in inc], dtype=object),
            "excl_labels": np.array([o.label for o in exc], dtype=object),
            "t_scale": scale,
            "t_unit": "min" if scale != 1.0 else "h",
            "show_excluded": self._show_excluded.isChecked(),
            "show_residuals": self._show_residuals.isChecked(),
            "show_corrected": self._show_corrected.isChecked(),
            "show_ref_avg": self._show_ref_avg.isChecked(),
        }

        # Residuals, for the fitted points and -- same formula -- for
        # the excluded ones, so hiding or showing them is only a
        # display choice.
        t_r, r = rc.residuals()
        mu_r, _ = rc.predict(np.asarray(t_r) + rc._t0)
        d["res_t"], d["res_r"] = np.asarray(t_r), np.asarray(r)
        d["res_mu"] = np.asarray(mu_r, dtype=float)
        t_e = np.asarray(d["t_excl"], dtype=float)
        if t_e.size:
            mu_e, _ = rc.predict(t_e + rc._t0)
            sn = float(getattr(rc.hyperparameters, "sigma_n", 0.0) or 0.0)
            y_e = np.asarray(d["y_excl"], dtype=float)
            e_e = np.asarray(d["yerr_excl"], dtype=float)
            d["res_t_excl"] = t_e
            d["res_mu_excl"] = np.asarray(mu_e, dtype=float)
            d["res_r_excl"] = (y_e - d["res_mu_excl"]) / np.sqrt(
                e_e ** 2 + sn ** 2)

        # Corrected centroids, flattened: label table + index per point.
        from cls_estimations.isotope_shift import weighted_average
        series = self._corrected_centroids() \
            if self._show_corrected.isChecked() else {}
        cols = {k: [] for k in ("index", "t", "y", "yerr", "raw", "g",
                                "fly", "excl", "run")}
        labels, ref_lbl, ref_avg, ref_err = [], [], [], []
        for i, (lbl, s) in enumerate(sorted(series.items())):
            labels.append(lbl)
            n = len(s["t"])
            cols["index"].extend([i] * n)
            for k in ("t", "y", "yerr", "raw", "g", "fly", "excl",
                      "run"):
                cols[k].extend(s[k])
            if s.get("is_ref"):
                keep = [(y, e) for y, e, x in zip(s["y"], s["yerr"],
                                                  s["excl"])
                        if not x and e > 0]
                if keep:
                    a, ea = weighted_average([k[0] for k in keep],
                                             [k[1] for k in keep])
                    ref_lbl.append(lbl)
                    ref_avg.append(float(a))
                    ref_err.append(float(ea))
        d.update({
            "corr_labels": np.array(labels, dtype=object),
            "corr_index": np.asarray(cols["index"], dtype=int),
            "corr_t": np.asarray(cols["t"], dtype=float),
            "corr_y": np.asarray(cols["y"], dtype=float),
            "corr_yerr": np.asarray(cols["yerr"], dtype=float),
            "corr_raw": np.asarray(cols["raw"], dtype=float),
            "corr_g": np.asarray(cols["g"], dtype=float),
            "corr_fly": np.asarray(cols["fly"], dtype=bool),
            "corr_excl": np.asarray(cols["excl"], dtype=bool),
            "corr_run": np.array(cols["run"], dtype=object),
            "ref_avg_labels": np.array(ref_lbl, dtype=object),
            "ref_avg": np.asarray(ref_avg, dtype=float),
            "ref_avg_err": np.asarray(ref_err, dtype=float),
        })
        return d

    @staticmethod
    def _hover_from(d, axes):
        """``[(ax, x, y, text)]`` for every point drawn from *d*."""
        scale = float(d.get("t_scale", 1.0) or 1.0)
        unit = str(d.get("t_unit", "h"))
        prec = 1 if scale != 1.0 else 3
        show_ex = bool(d.get("show_excluded", True))
        pts = []

        def when(t):
            return f"t = {float(t) * scale:.{prec}f} {unit}"

        ax = axes["drift"]
        for t, y, lbl in zip(d["t_train"], d["y_train"],
                             d["train_labels"]):
            pts.append((ax, float(t) * scale, float(y),
                        f"{lbl}\n{float(y):.2f} MHz at {when(t)}"))
        from gui.analysis.gp_figure import edge_y

        def pinned(axis, y):
            """Where an excluded point is drawn, and a note if it was
            pinned to the frame edge."""
            yd, mk = edge_y(axis, float(y))
            return yd, ("  (off scale)" if mk else "")

        if show_ex:
            for t, y, lbl in zip(d["t_excl"], d["y_excl"],
                                 d["excl_labels"]):
                yd, off = pinned(ax, y)
                pts.append((ax, float(t) * scale, yd,
                            f"{lbl}  (excluded){off}\n"
                            f"{float(y):.2f} MHz at {when(t)}"))

        ax_r = axes.get("res")
        if ax_r is not None:
            rows = list(zip(d["res_t"], d["res_r"], d["res_mu"],
                            d["y_train"], d["train_labels"],
                            [""] * len(d["res_t"])))
            if show_ex and "res_r_excl" in d:
                rows += list(zip(d["res_t_excl"], d["res_r_excl"],
                                 d["res_mu_excl"], d["y_excl"],
                                 d["excl_labels"],
                                 ["  (excluded)"] * len(d["res_t_excl"])))
            for t, r, mu, y, lbl, tag in rows:
                r_draw, off = (pinned(ax_r, r) if tag
                               else (float(r), ""))
                tag = tag + off
                pts.append((ax_r, float(t) * scale, r_draw,
                            f"{lbl}{tag}\n"
                            f"residual {float(r):+.2f}\u03c3\n"
                            f"original {float(y):.2f} MHz\n"
                            f"GP {float(mu):.2f} MHz\n"
                            f"original \u2212 GP "
                            f"{float(y) - float(mu):+.2f} MHz"))

        ax_c = axes.get("corr")
        if ax_c is not None:
            labels = [str(v) for v in d["corr_labels"]]
            for i, t, y, e, raw, g, fly, ex, run in zip(
                    d["corr_index"], d["corr_t"], d["corr_y"],
                    d["corr_yerr"], d["corr_raw"], d["corr_g"],
                    d["corr_fly"], d["corr_excl"], d["corr_run"]):
                if ex and not show_ex:
                    continue
                where = "here" if fly else "at fit time"
                tag = "  (excluded)" if ex else ""
                y_draw = float(y)
                if ex:
                    y_draw, off = pinned(ax_c, y)
                    tag += off
                pts.append((ax_c, float(t) * scale, y_draw,
                            f"{labels[int(i)]}  run {run}{tag}\n"
                            f"original {float(raw):.2f} MHz\n"
                            f"GP {float(g):.2f} MHz ({where})\n"
                            f"corrected {float(y):.2f} \u00b1 "
                            f"{float(e):.2f} MHz\n{when(t)}"))
        return pts

    @staticmethod
    def _run_timestamps(project):
        """``({path: (ts_start, ts_stop, run_number)}, [undated])`` for
        every run of *project*.

        A GP correction needs nothing but a timestamp; the run does
        NOT have to have been fitted. That matters for two cases the
        old fit-results-only lookup could never serve (2026-09-20):

        * a project fitted on a MERGED entry. Its fit result is keyed
          ``merged://<name>``, so the constituent ASDFs got no rows
          and the merge dialog's "Align centroids before merging"
          stayed greyed out -- the one place the correction most
          needs to be available, since merging is what bakes it in.
        * a project the user wants to align BEFORE fitting at all,
          which is the natural order: correct, merge, then fit.

        Timestamps come from the cheapest source that has them: the
        last fit results, then a merged entry's per-run audit table
        (both already in memory), then one lazy cell read from the
        ASDF itself. ``merged://`` paths are deliberately excluded --
        a merged spectrum inherits the corrections of its
        constituents at merge time (see ``_merged_run_metadata`` in
        fitting.py), so a row of its own would be dead weight.
        """
        stamps: dict[str, tuple[float, str]] = {}
        no_ts: list[str] = []

        def _put(path, ts, run_num, ts_end=0) -> bool:
            if not path or str(path).startswith("merged://"):
                return False
            try:
                ts = float(ts or 0)
                ts_end = float(ts_end or 0)
            except (TypeError, ValueError):
                return False
            if ts <= 0:
                return False
            if ts_end <= ts:
                ts_end = 0.0        # unknown; caller falls back to a point
            stamps.setdefault(path, (ts, ts_end, str(run_num or "")))
            return True

        results = getattr(project, "_last_results", None)
        if not results and hasattr(project, "load_results_from_disk"):
            try:
                project.load_results_from_disk()
            except Exception:  # noqa: BLE001
                pass
            results = getattr(project, "_last_results", None)
        for r in results or []:
            if not r.get("success"):
                continue
            _meta = r.get("run_metadata") or {}
            _put(r.get("run_file"), _meta.get("ts_start"),
                 r.get("run_number"), _meta.get("ts_stop"))

        try:
            from gui.analysis.blocks import SourceBlock
            blocks = project._get_blocks_by_type(SourceBlock)
        except Exception:  # noqa: BLE001
            blocks = []

        # Merged entries first: their per-run audit table already
        # holds each constituent's path + ts_start, so taking them
        # here saves an ASDF read in the pass below.
        for blk in blocks:
            for entry in getattr(blk, "_file_entries", []) or []:
                if not entry.get("is_merged"):
                    continue
                md = entry.get("merged_data") or {}
                for pr in md.get("per_run") or []:
                    _put(pr.get("path"), pr.get("ts_start"),
                         pr.get("run_num"), pr.get("ts_stop"))

        for blk in blocks:
            for entry in getattr(blk, "_file_entries", []) or []:
                if entry.get("is_merged"):
                    continue
                path = entry.get("path")
                if not path or path in stamps:
                    continue
                run_num = entry.get("run_number", "")
                # A .vasdf split is a descriptor; its events (and so
                # its timestamps) live in the parent ASDF.
                src = entry.get("parent_path") or path
                _t0, _t1 = _asdf_ts_span(src)
                if not _put(path, _t0, run_num, _t1):
                    no_ts.append(str(run_num or os.path.basename(path)))

        return stamps, no_ts

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
                    mode_combo.setToolTip(
                        "Auto: use the GP prediction for this run.  "
                        "Manual: type your own correction and sigma.  "
                        "Off: leave this run uncorrected (it is then "
                        "excluded from the correction budget).")
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

    # ── Reference observations ─────────────────────────────

    def _gather_observations(self) -> list[ReferenceObservation]:
        """Every reference centroid from the ticked projects, each
        carrying its include flag.

        Shared by the fit and by the run list, so the list cannot
        disagree with what the GP is actually trained on.
        """
        out: list[ReferenceObservation] = []
        ref_names = self._checked_names(self._ref_list)
        for p in self._analysis_tab._projects:
            if p.project_name not in ref_names:
                continue
            for r in p.get_reference_observations():
                label = canonical_obs_key(
                    f"{p.project_name}/{r['label']}")
                out.append(ReferenceObservation(
                    t=r["ts_start"] / 3600.0,    # hours since epoch
                    centroid=r["centroid_mhz"],
                    sigma=r["sigma_mhz"],
                    label=label,
                    include=(label not in self._excluded_obs
                             and bool(r.get("include", True))),
                ))
        out.sort(key=lambda o: o.t)
        return out

    def _refresh_obs_list(self):
        """Repopulate the per-run tick list from the ticked projects.

        The GP is only as good as the runs it trains on, and one bad
        centroid drags the whole curve: a reference run whose
        calibration stream was invalid can sit 200 MHz off its
        neighbours, and the fit will bend to reach it rather than
        treat it as noise. Until now there was no way to say so
        (2026-09-21).
        """
        self._obs_guard = True
        try:
            self._obs_list.clear()
            obs = self._gather_observations()
            # Prefix with the project only when there is more than one
            # -- "74Ge/run_7961" is noise when 74Ge is the only source.
            multi = len(self._checked_names(self._ref_list)) > 1
            for o in obs:
                run = o.label.split("/", 1)[-1]
                shown = o.label if multi else run
                it = QListWidgetItem(
                    f"{shown}     {o.centroid:+.1f} ± "
                    f"{o.sigma:.1f} MHz")
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(
                    Qt.CheckState.Checked if o.include
                    else Qt.CheckState.Unchecked)
                it.setData(Qt.ItemDataRole.UserRole, o.label)
                it.setToolTip(_wrap_tip(
                    f"{o.label}: centroid {o.centroid:+.3f} MHz, "
                    f"σ {o.sigma:.3f} MHz. Untick to drop this "
                    f"run from the GP training set; it stays on the "
                    f"plot in grey. Re-fit the GP afterwards."))
                self._obs_list.addItem(it)
        finally:
            self._obs_guard = False
        self._update_obs_label()

    def _update_obs_label(self):
        n = self._obs_list.count()
        n_off = sum(
            1 for i in range(n)
            if self._obs_list.item(i).checkState()
            != Qt.CheckState.Checked)
        txt = f"Reference runs ({n} total"
        txt += f", {n_off} excluded):" if n_off else "):"
        listed = {str(self._obs_list.item(i).data(
            Qt.ItemDataRole.UserRole)) for i in range(n)}
        if (self._corrector is not None
                and self._fitted_excluded is not None
                and (self._excluded_obs & listed)
                != (self._fitted_excluded & listed)):
            # Stale means the ticks and the fitted curve DISAGREE.
            # Keying it off "anything excluded" left the nag up after
            # a perfectly good re-fit, which teaches the user to
            # ignore it.
            txt += "   — re-fit the GP"
        self._obs_lbl.setText(txt)

    def _on_ref_item_changed(self, _item):
        """A project was ticked or unticked -- its runs come or go."""
        if self._obs_guard:
            return
        self._refresh_obs_list()

    def _on_obs_item_changed(self, item):
        if self._obs_guard:
            return
        label = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not label:
            return
        if item.checkState() == Qt.CheckState.Checked:
            self._excluded_obs.discard(label)
        else:
            self._excluded_obs.add(label)
        self._update_obs_label()
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

    def ui_layout(self):
        from gui.ui_layout import collect
        return collect(main=getattr(self, "_main_splitter", None))

    def apply_ui_layout(self, d):
        from gui.ui_layout import restore
        restore(d, main=getattr(self, "_main_splitter", None))

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
            # By label, not index: the run list is rebuilt from
            # whatever projects are loaded, in timestamp order, so an
            # index would silently point at a different run.
            "excluded_observations": sorted(self._excluded_obs),
            "corrector": (self._corrector.to_dict()
                          if self._corrector is not None else None),
            "corrections": {
                project: {
                    fpath: {
                        "run_number": fc.run_number,
                        "ts_start": fc.ts_start,
                        "ts_stop": fc.ts_stop,
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
        # Restored before the lists are built: refresh_projects() ->
        # _refresh_obs_list() reads this to set the tick states.
        # Normalised on the way in: keys saved before 2026-09-22 end
        # in a file name that a disk-loaded result does not have.
        self._excluded_obs = {
            canonical_obs_key(x)
            for x in (d.get("excluded_observations") or [])}

        # The project lists need refresh_projects() called by the
        # parent after all projects are restored.

        corr_state = d.get("corrector")
        corrections_in_yaml = bool(d.get("corrections"))
        if corr_state:
            try:
                self._corrector = ReferenceCorrector.from_dict(corr_state)
                self._render_diagnostic_plot()
                hp = self._corrector.hyperparameters
                # The corrector remembers which runs it was fitted
                # without, so the "re-fit" nag is accurate from the
                # moment the file opens rather than after the first
                # tick.
                self._fitted_excluded = {
                    canonical_obs_key(o.label)
                    for o in self._corrector.observations
                    if not o.include}
                _n_fitted = sum(
                    1 for o in self._corrector.observations if o.include)
                self._status.setText(
                    f"Loaded GP fit "
                    f"({_kernel_name(self._corrector.kernel)} kernel; "
                    f"n={_n_fitted}). "
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
                    ts_stop=float(c.get("ts_stop", 0.0) or 0.0),
                    t_hours=float(c.get("t_hours", 0.0)),
                    mode=mode,
                    auto_value_mhz=auto_v,
                    auto_sigma_mhz=auto_s,
                    manual_value_mhz=manual_v,
                    manual_sigma_mhz=manual_s,
                )
            self._corrections[project] = inner
        self._populate_table()
        # A restored GP with an EMPTY table looks like the panel
        # failed to load its settings. It usually means the GP was
        # re-fitted last session -- which deliberately clears stale
        # per-file numbers -- and saved before Compute was clicked
        # again. Say so rather than showing a blank table under a
        # confident "Loaded GP fit" line (2026-09-20).
        if self._corrector is not None and not self._corrections:
            self._status.setText(
                self._status.text()
                + "  No per-file corrections stored yet -- tick the "
                  "sample projects and click Compute per-file "
                  "corrections.")

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

    def corrections_for_paths(self, file_paths) -> dict:
        """Path-keyed corrections across EVERY sample project.

        The Pre-Analysis tab merges files without knowing which
        AnalysisProject they belong to, so it cannot use
        :meth:`get_correction` (which is keyed by project). A path that
        somehow carries corrections in two projects is SKIPPED rather
        than guessed at -- silently picking one would bias the merge.
        """
        if not self.apply_globally or self._corrector is None:
            return {}
        wanted = set(file_paths or ())
        hits: dict[str, list] = {}
        for fmap in self._corrections.values():
            for path, fc in fmap.items():
                if path in wanted:
                    hits.setdefault(path, []).append(fc)
        out = {}
        for path, found in hits.items():
            if len(found) != 1:
                continue        # ambiguous: same run in two projects
            fc = found[0]
            out[path] = {
                "correction_mhz": fc.value_mhz,
                "sigma_mhz": fc.sigma_mhz,
                "mode": fc.mode,
            }
        return out

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
