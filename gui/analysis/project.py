"""Analysis project widget driving the block-based fit pipeline.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Defines AnalysisProject, the QWidget that holds one project subtab in the
Analysis tab: a horizontal, drag-reorderable pipeline of Source, Model,
Fitter, and Output blocks. It gathers each block's config, validates
parameter expressions, runs the fit in a worker thread, applies the
auto-fitter and reference centroid corrections, and writes all output
artifacts (fit reports, CSVs, fit/ToF/tracker/diagnostic plots, MCMC
chains) into per-iteration directories. Reference projects also expose
fitted centroids as observations for the GP drift corrector.

Depends on: gui.shared_widgets, gui.analysis.* (helpers, blocks, fitting,
naming, expr_validation, plus auto_fitter, binning, merge on demand),
gui.scan_filter, and cls_estimations.isotope_shift; uses PySide6,
numpy/pandas, matplotlib, and satlas2.
"""

import os
import numpy as np
import yaml

from gui import session_log

_log = session_log.get_logger("project")

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QScrollArea, QMessageBox, QMenu,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QUndoCommand

from gui.shared_widgets import (
    _load_settings, get_plot_type_settings, fit_plot_settings_for_counts,
    lucide_icon,
)
from gui.analysis.helpers import _show_scrollable_info
from gui.analysis.blocks import (
    AnalysisBlock, SourceBlock, ModelBlock, FitterBlock, OutputBlock,
)
from gui.analysis.fitting import (
    _build_models_on_source, FitWorkerThread,
)
from gui.analysis.naming import (
    NON_FIT_PARAMS, is_fit_param_name,
    source_name_for_path, source_name_for_merged,
    source_name_for_descriptor,
)
from gui.analysis.expr_validation import validate_configs


# ══════════════════════════════════════════════════════════════════
#  Pre-fit expression validation dialogs
# ══════════════════════════════════════════════════════════════════

def run_sort_key(run_number):
    """Natural sort key for a run number: numeric runs ascending, then
    non-numeric labels (e.g. merged names) last, alphabetically.

    Used so the fit report, the per-run CSVs, and the saved figures all
    follow the same numeric run ordering instead of completion order.
    """
    text = str(run_number)
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return (0, int(digits), text) if digits else (1, 0, text)


def _format_issue_lines(issues):
    return "\n".join(f"• {i.location_str()}: {i.message}" for i in issues)


def _show_validation_errors(parent, errors, warnings_):
    """Blocking dialog. Lists errors that must be fixed before fitting.
    Warnings are shown as context but don't change the outcome."""
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Cannot fit: expression errors")
    msg.setText("The following expression problem(s) must be fixed "
                "before fitting:")
    detail = _format_issue_lines(errors)
    if warnings_:
        detail += ("\n\nWarnings (would be shown if errors were "
                   "resolved):\n")
        detail += _format_issue_lines(warnings_)
    msg.setInformativeText(detail)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()


def _confirm_validation_warnings(parent, warnings_):
    """Non-blocking confirmation. Returns True if the user explicitly
    chose to continue. Default button is Cancel."""
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Expression warnings")
    msg.setText("The configuration has warnings. Review before fitting:")
    msg.setInformativeText(_format_issue_lines(warnings_))
    cont_btn = msg.addButton("Continue anyway",
                             QMessageBox.ButtonRole.AcceptRole)
    cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
    msg.setDefaultButton(cancel_btn)
    msg.exec()
    return msg.clickedButton() is cont_btn


# ══════════════════════════════════════════════════════════════════
#  Undo command for batch spinbox changes (Auto-Fit accept, etc.)
# ══════════════════════════════════════════════════════════════════

class _BatchSetValueCommand(QUndoCommand):
    """Undo/redo a batch of spinbox value changes in one step."""

    def __init__(self, changes, blocks):
        super().__init__("Auto-Fit Accept")
        # changes: list of (QDoubleSpinBox, old_value, new_value)
        self._changes = changes
        self._blocks = blocks  # affected ModelBlock set for block_changed

    def redo(self):
        for widget, _, new_val in self._changes:
            widget.blockSignals(True)
            widget.setValue(new_val)
            widget.blockSignals(False)
            if hasattr(widget, '_undo_prev'):
                widget._undo_prev = new_val
        for block in self._blocks:
            block.block_changed.emit()

    def undo(self):
        for widget, old_val, _ in self._changes:
            widget.blockSignals(True)
            widget.setValue(old_val)
            widget.blockSignals(False)
            if hasattr(widget, '_undo_prev'):
                widget._undo_prev = old_val
        for block in self._blocks:
            block.block_changed.emit()


# ══════════════════════════════════════════════════════════════════
#  Block Container with drag-and-drop reordering
# ══════════════════════════════════════════════════════════════════

class _BlockContainer(QWidget):
    """Horizontal container that accepts drag-and-drop block reordering."""
    blocks_reordered = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 20, 4, 4)
        self._layout.setSpacing(8)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._drop_indicator_idx = -1

    def layout(self):
        return self._layout

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-analysis-block"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-analysis-block"):
            event.acceptProposedAction()
            # Determine insertion index based on x position
            pos_x = event.position().x()
            idx = self._layout.count()
            for i in range(self._layout.count()):
                w = self._layout.itemAt(i).widget()
                if w and pos_x < w.x() + w.width() / 2:
                    idx = i
                    break
            self._drop_indicator_idx = idx
            self.update()

    def dragLeaveEvent(self, event):
        self._drop_indicator_idx = -1
        self.update()

    def dropEvent(self, event):
        self._drop_indicator_idx = -1
        self.update()
        if not event.mimeData().hasFormat("application/x-analysis-block"):
            return
        # Find the dragged block by id
        block_id = int(event.mimeData().data(
            "application/x-analysis-block").data().decode())
        dragged = None
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if w and id(w) == block_id:
                dragged = w
                break
        if dragged is None:
            return

        # Determine target index
        pos_x = event.position().x()
        target_idx = self._layout.count()
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if w and pos_x < w.x() + w.width() / 2:
                target_idx = i
                break

        # Remove and re-insert
        self._layout.removeWidget(dragged)
        self._layout.insertWidget(target_idx, dragged)
        event.acceptProposedAction()
        self.blocks_reordered.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_indicator_idx >= 0:
            from PySide6.QtGui import QPainter, QPen
            painter = QPainter(self)
            pen = QPen(QColor("#2196F3"), 3)
            painter.setPen(pen)
            # Draw vertical line at insertion point
            if self._drop_indicator_idx < self._layout.count():
                w = self._layout.itemAt(self._drop_indicator_idx).widget()
                if w:
                    x = w.x() - 4
                    painter.drawLine(x, 4, x, self.height() - 4)
            else:
                # After last widget
                if self._layout.count() > 0:
                    w = self._layout.itemAt(
                        self._layout.count() - 1).widget()
                    if w:
                        x = w.x() + w.width() + 4
                        painter.drawLine(x, 4, x, self.height() - 4)
            painter.end()


# ══════════════════════════════════════════════════════════════════
#  Analysis Project Widget (one per project subtab)
# ══════════════════════════════════════════════════════════════════

class AnalysisProject(QWidget):
    """A single analysis project containing a horizontal block pipeline."""
    results_ready = Signal(str, list, dict)  # project_name, results, config
    fit_progress = Signal(int, int, str)     # current, total, status

    def __init__(self, name="Project", is_reference=False, parent=None):
        super().__init__(parent)
        self._project_name = name
        self._is_reference = bool(is_reference)
        self._blocks = []
        self._fit_worker = None
        self._last_results = None
        self._last_output_config = None
        # Directory of the most recently saved iteration; "Re-apply
        # outputs" regenerates files into this same folder instead of
        # allocating a new iter_NNN.
        self._last_iter_dir = None
        # Iteration names this project wrote during the session (or
        # restored from a save file). Saved with the project so loading
        # the file can repopulate the Results tab with exactly these
        # iterations instead of rescanning the whole output directory.
        self._session_iterations = []
        settings = _load_settings()
        default_cores = max(1, os.cpu_count() - 2) if os.cpu_count() else 1
        self._n_cores = settings.get("max_cores", default_cores)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ── Block scroll area (horizontal) ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._block_container = _BlockContainer()
        self._block_container.blocks_reordered.connect(
            self._sync_blocks_from_layout)
        self._block_layout = self._block_container.layout()

        scroll.setWidget(self._block_container)
        main_layout.addWidget(scroll, 1)

        # ── Bottom bar ──
        bottom = QHBoxLayout()
        info_btn = QPushButton(lucide_icon("circle-help"), "Info")
        info_btn.setFixedWidth(75)
        info_btn.setToolTip("Show the analysis pipeline guide")
        info_btn.clicked.connect(self._show_info)
        bottom.addWidget(info_btn)
        bottom.addStretch()
        self._add_block_btn = QPushButton("+ Add Block")
        menu = QMenu(self._add_block_btn)
        menu.addAction("Source", lambda: self.add_block("Source"))
        menu.addAction("Model", lambda: self.add_block("Model"))
        menu.addAction("Fitter", lambda: self.add_block("Fitter"))
        menu.addAction("Output", lambda: self.add_block("Output"))
        self._add_block_btn.setMenu(menu)
        bottom.addWidget(self._add_block_btn)
        main_layout.addLayout(bottom)

        # Add default blocks
        self.add_block("Source")
        self.add_block("Model")
        self.add_block("Fitter")
        self.add_block("Output")

    def _show_info(self):
        text = (
            "<h3>Analysis Pipeline</h3>"
            "Build a fitting pipeline from left to right using four block "
            "types. Each block can be enabled/disabled with its checkbox "
            "and reordered by dragging. Hover over any control for a tooltip.<br><br>"

            "<h4>Source Block</h4>"
            "\u2022 <b>Files</b> \u2013 Add ASDF run files to analyse. "
            "Use <i>Import from Pre-Analysis</i> to import the current selection, "
            "or <i>Merge Selected</i> to combine runs into a single spectrum.<br>"
            "\u2022 <b>Z / A / Mass</b> \u2013 Proton number, mass number, and "
            "atomic mass (amu). Mass auto-fills from IUPAC/AME table.<br>"
            "\u2022 <b>Harmonic</b> \u2013 Laser harmonic multiplier "
            "(e.g.&nbsp;4 for fourth harmonic).<br>"
            "\u2022 <b>E lower / E upper</b> \u2013 Lower- and upper-state "
            "energies [cm\u207b\u00b9], used for Doppler conversion.<br>"
            "\u2022 <b>TOF gate</b> \u2013 Time-of-flight window [\u00b5s] to "
            "select the ion bunch.<br>"
            "\u2022 <b>PMT channels</b> \u2013 Select which detector channels "
            "to include (1\u20134 and DC).<br>"
            "\u2022 <b>V gate / F gate</b> \u2013 Optional voltage and "
            "frequency gates to filter data.<br>"
            "\u2022 <b>Noise filter</b> \u2013 Remove timestamps with excess "
            "event counts (0 = disabled).<br>"
            "\u2022 <b>Binning</b> \u2013 x-axis quantity, y-error mode, "
            "frequency vs raw voltage, and optional x-error.<br>"
            "\u2022 <b>Advanced</b> \u2013 Calibration order, cooler correction "
            "mode (pbp/mean), and reference frequency shift.<br>"
            "\u2022 <b>Cooler / Laser Override</b> \u2013 Force a single "
            "cooler voltage or laser setpoint for all runs (0 = auto from file).<br>"
            "\u2022 <b>Preview</b> \u2013 Quick-look at the spectrum. "
            "<b>Meta</b> shows run metadata. "
            "<b>Run Stats</b> plots experimental parameters vs run number.<br><br>"

            "<h4>Model Block</h4>"
            "\u2022 <b>Type</b> \u2013 HFS, Voigt, Skewed Voigt, Exponential Decay, "
            "Piecewise Constant, or Polynomial.<br>"
            "\u2022 <b>Source</b> \u2013 Which Source block provides the data.<br>"
            "\u2022 <b>Import</b> \u2013 Pull HFS parameters (I, J, A, B, centroid) "
            "from Pre-Analysis.<br>"
            "\u2022 <b>HFS settings</b> \u2013 Racah intensity ratios; uncheck "
            "<b>Racah</b> to fit peak amplitudes freely (use the Skewed Voigt "
            "model type for asymmetric peaks).<br>"
            "\u2022 <b>Sidepeaks</b> \u2013 Add N charge-exchange sidepeaks with "
            "an offset and optional Poisson distribution.<br>"
            "\u2022 <b>Parameter table</b> \u2013 Set initial values, choose each "
            "parameter's <i>Mode</i> (Free / Fixed / Equal / Ratio / Offset / "
            "Custom), set <i>Bounds</i> [min, max], or write an "
            "<i>Expression</i>.<br>"

            "<h4>Fitter Block</h4>"
            "\u2022 <b>Fit mode</b> \u2013 <i>Separate</i> fits each file "
            "independently in parallel; <i>Simultaneous</i> fits all files "
            "at once with optional parameter sharing or advanced "
            "constraints.<br>"
            "\u2022 <b>Method</b> \u2013 leastsq (fast), least_squares (bounded), "
            "emcee (MCMC/Bayesian), nelder/powell/cobyla (derivative-free), "
            "slsqp (constrained).<br>"
            "\u2022 <b>Statistics</b> \u2013 Chi-square, Gaussian log-likelihood, "
            "or Poisson log-likelihood (best for low counts).<br>"
            "\u2022 <b>Scale covariance</b> \u2013 Scale uncertainties by "
            "\u221a(reduced \u03c7\u00b2). Disable for likelihood fits.<br>"
            "\u2022 <b>Parameter Sharing</b> (simultaneous) \u2013 Share "
            "a bare parameter (e.g. <i>Al</i>) across same-named models "
            "or all sources.<br>"
            "\u2022 <b>Advanced Constraints</b> (simultaneous) \u2013 "
            "Set explicit cross-run/cross-model expressions on full "
            "parameter targets.<br>"
            "\u2022 <b>Priors</b> \u2013 Gaussian prior constraints: adds "
            "((value \u2212 prior) / \u03c3)\u00b2 penalty.<br>"
            "\u2022 <b>MCMC settings</b> \u2013 Walkers, steps, and optional "
            "auto-convergence based on autocorrelation time.<br><br>"

            "<h4>Output Block</h4>"
            "\u2022 <b>Reports</b> \u2013 Text fit report with optional "
            "correlations, parameters CSV, and metadata CSV.<br>"
            "\u2022 <b>Fit plots</b> \u2013 Per-run plots with data, fit curve, "
            "residuals, and component overlays. Choose format and DPI.<br>"
            "\u2022 <b>Tracker</b> \u2013 Plot selected parameters vs run number "
            "or start time across iterations.<br>"
            "\u2022 <b>Diagnostics</b> \u2013 Chi-square map (any method), "
            "walk plot and correlation plot (emcee only), confidence bands "
            "(1\u03c3 shaded region from thinned MCMC chain).<br>"
            "\u2022 <b>Iteration</b> \u2013 <i>Auto</i> increments "
            "(iter_001, iter_002, ...); <i>Manual</i> uses a custom label.<br><br>"

            "<h4>Sample vs. Reference projects</h4>"
            "\u2022 The <b>+ Create Analysis Project</b> menu offers two "
            "kinds of project. <b>Sample Project</b> is the standard "
            "fit pipeline. <b>Reference Project</b> tags a project as a "
            "set of reference-isotope scans whose fitted centroids feed "
            "the Gaussian-process drift correction. Reference-project "
            "tabs show a <i>(Reference)</i> suffix.<br><br>"

            "<h4>Isotope Shifts Tab</h4>"
            "\u2022 Appears after creating projects. Select fitted projects, "
            "designate a reference isotope, and compute isotope shifts with "
            "propagated statistical and systematic errors.<br>"
            "\u2022 Systematic errors estimated from per-project cooler voltage "
            "and laser setpoint jitter (SEM).<br>"
            "\u2022 Plot modes: stacked subplots or overlaid spectra with "
            "centroid lines and \u0394\u03bd arrows.<br>"
            "\u2022 Send results (plot + CSV + log) to the Results tab.<br><br>"

            "<h4>Tips</h4>"
            "\u2022 Use <b>+ Add Block</b> for multi-component fits "
            "(e.g.&nbsp;HFS + Polynomial background).<br>"
            "\u2022 Blocks can be removed with the \u2717 button.<br>"
            "\u2022 Drag blocks to reorder them.<br>"
            "\u2022 Results appear in the <b>Results</b> tab and are saved to "
            "<tt>&lt;output_root&gt;/analysis/ProjectName/iter_NNN/</tt>.<br>"
            "\u2022 Use <b>Revert</b> to restore parameters to their pre-fit values."
        )
        _show_scrollable_info(self, "Analysis Tab \u2013 Guide", text)

    def add_block(self, block_type, config=None):
        """Add a new block of the given type."""
        counts = sum(1 for b in self._blocks
                     if b.BLOCK_TYPE == block_type) + 1
        default_name = f"{block_type}_{counts}"

        if block_type == "Source":
            block = SourceBlock(default_name)
        elif block_type == "Model":
            block = ModelBlock(default_name)
        elif block_type == "Fitter":
            block = FitterBlock(default_name)
            block.fit_requested.connect(self._on_fit_requested)
            block.auto_fit_requested.connect(self._on_auto_fit_requested)
            block.fit_cancel_requested.connect(self._on_fit_cancel)
            block.revert_requested.connect(self._revert_fit)
        elif block_type == "Output":
            block = OutputBlock(default_name)
            block.reapply_requested.connect(self._reapply_outputs)
        else:
            return

        block.remove_requested.connect(self._remove_block)
        block.block_changed.connect(self._on_block_changed)

        if config:
            block.from_dict(config)

        self._blocks.append(block)
        self._block_layout.addWidget(block)
        self._on_block_changed()
        return block

    def _sync_blocks_from_layout(self):
        """Re-sync self._blocks list to match the visual layout order."""
        new_order = []
        for i in range(self._block_layout.count()):
            w = self._block_layout.itemAt(i).widget()
            if w and isinstance(w, AnalysisBlock):
                new_order.append(w)
        self._blocks = new_order
        self._on_block_changed()

    def _remove_block(self, block):
        if block in self._blocks:
            self._blocks.remove(block)
            self._block_layout.removeWidget(block)
            block.deleteLater()
            self._on_block_changed()

    def _on_block_changed(self):
        """Sync source names into model combos, update tracker params, etc."""
        block_source_names = [b.block_name for b in self._blocks
                              if isinstance(b, SourceBlock) and b.is_enabled]
        for b in self._blocks:
            if isinstance(b, ModelBlock):
                b.update_source_list(block_source_names)

        # Sharing table: bare param union across enabled models.
        bare_param_set: set[str] = set()
        model_params_grouped = {}
        model_configs = []
        for b in self._blocks:
            if isinstance(b, ModelBlock) and b.is_enabled:
                cfg = b.get_model_config()
                model_configs.append(cfg)
                mt = cfg.get("type", "")
                param_names = [p for p in cfg["params"]
                               if is_fit_param_name(mt, p)]
                model_params_grouped[b.block_name] = param_names
                bare_param_set.update(param_names)
        bare_param_names = sorted(bare_param_set)

        # Expressions table: full registry of (source × model × param)
        # targets. Source names come from the FIRST enabled SourceBlock
        # only -- which is exactly what _on_fit_start uses (sources[0]).
        # Iterating all enabled SourceBlocks here would offer targets
        # that the fitter never instantiates.
        fit_source_names: list[str] = []
        first_source_block = next(
            (b for b in self._blocks
             if isinstance(b, SourceBlock) and b.is_enabled),
            None,
        )
        if first_source_block is not None:
            for entry in first_source_block.get_checked_files():
                if entry.get("is_merged") and "merged_data" in entry:
                    fit_source_names.append(
                        source_name_for_merged(entry["merged_data"]))
                else:
                    fit_source_names.append(
                        source_name_for_path(entry["path"]))

        # Collect model names + types for component overlay checkboxes
        model_name_types = [(b.block_name, b.get_model_config()["type"])
                            for b in self._blocks
                            if isinstance(b, ModelBlock) and b.is_enabled]

        for b in self._blocks:
            if isinstance(b, FitterBlock):
                b.update_sharing_table(bare_param_names)
                b.update_expressions_table(
                    fit_source_names, model_configs)
            if isinstance(b, OutputBlock):
                b.update_tracker_params(model_params_grouped)
                b.update_component_list(model_name_types)

    def _get_blocks_by_type(self, block_type_cls):
        return [b for b in self._blocks
                if isinstance(b, block_type_cls) and b.is_enabled]

    def _on_auto_fit_requested(self):
        """Open the Auto-Fitter dialog respecting the Fitter block mode."""
        from gui.analysis.auto_fitter import AutoFitterDialog

        sources = self._get_blocks_by_type(SourceBlock)
        models = self._get_blocks_by_type(ModelBlock)
        if not sources or not models:
            QMessageBox.warning(self, "Auto-Fitter",
                                "Need at least one Source and one Model block.")
            return

        source_block = sources[0]
        checked = source_block.get_checked_files()
        if not checked:
            QMessageBox.warning(self, "Auto-Fitter",
                                "No files selected in the Source block.")
            return

        # Get fitter config for mode and sharing
        fitters = self._get_blocks_by_type(FitterBlock)
        fitter_config = fitters[0].get_fitter_config() if fitters else {}
        is_simultaneous = not fitter_config.get("separate", True)

        source_config = source_block.get_source_config()

        # Load source data depending on mode
        sources_data = []
        files_to_load = checked if is_simultaneous else checked[:1]
        # AutoFit's initial guesses are biased by μ_GP if we feed
        # the unshifted axis -- ask the panel for the same per-file
        # corrections the real fit will apply, then subtract them
        # post-binning inside _load_data_for_autofit.
        autofit_corrections = self._build_corrections_map(
            [entry["path"] for entry in files_to_load])
        for entry in files_to_load:
            run_file = entry["path"]
            merged = entry.get("merged_data") if entry.get(
                "is_merged") else None
            split_desc = None
            if entry.get("is_split"):
                split_desc = {
                    "parent_path": entry["parent_path"],
                    "source_id": entry["source_id"],
                    "split_lo": float(entry["split_lo"]),
                    "split_hi": float(entry["split_hi"]),
                    "metadata_override": dict(
                        entry.get("metadata_override") or {}),
                }
            try:
                x, y, yerr = self._load_data_for_autofit(
                    run_file, source_config, merged,
                    entry.get("binning_override"),
                    centroid_correction=autofit_corrections.get(
                        run_file),
                    split_descriptor=split_desc)
            except Exception as e:
                QMessageBox.critical(self, "Auto-Fitter",
                                     f"Failed to load {run_file}:\n{e}")
                return
            name = os.path.splitext(os.path.basename(run_file))[0]
            if entry.get("is_merged"):
                name = entry.get("merged_data", {}).get(
                    "merged_name", name)
            elif split_desc is not None:
                # Use the split's source_id so the AutoFitter dialog
                # labels the spectra distinctly when both halves are
                # loaded simultaneously.
                name = split_desc["source_id"]
            sources_data.append(
                {"name": name, "x": x, "y": y, "yerr": yerr})

        model_configs = [m.get_model_config() for m in models]

        dlg = AutoFitterDialog(
            sources_data, model_configs, fitter_config, parent=self)
        dlg._model_blocks = models  # store reference for accept
        dlg.parameters_accepted.connect(self._on_autofit_accepted)
        self._auto_fitter_dialog = dlg  # prevent GC
        dlg.show()

    def _on_autofit_accepted(self, result):
        """Slot for auto-fitter parameters_accepted signal."""
        try:
            dlg = self._auto_fitter_dialog
            if dlg is None:
                return
            models = getattr(dlg, '_model_blocks', [])
            self._apply_autofit_result(result, models)
        except Exception as e:
            print(f"[AutoFit] ERROR in accept slot: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _load_data_for_autofit(self, run_file, source_config, merged_data,
                               binning_override=None,
                               centroid_correction=None,
                               split_descriptor=None):
        """Load and bin a single run, return (x, y, yerr).

        Per-file ``binning_override`` (sparse patch on top of
        ``source_config``) is applied so the auto-fitter sees the
        same binning the real fit will use.

        ``centroid_correction`` (dict with ``correction_mhz`` /
        ``mode``) is the same shape ``_fit_single_run`` consumes;
        when given (and bin_mode is Frequency, mode is Auto/Manual),
        the binned x-axis is shifted by ``-correction_mhz`` so the
        AutoFitter's initial-guess centroid lives in the same frame
        the eventual fit will produce.

        ``split_descriptor`` (virtual-split feature) is the per-file
        descriptor for a ``.vasdf`` entry. When non-None we load
        ``parent_path`` instead of ``run_file``, compose the V-gate
        with the split's [split_lo, split_hi], and apply the
        per-split cooler/laser overrides with the same precedence
        as the real fit (SourceBlock global > per-split > ASDF).
        """
        import numpy as np
        from gui.analysis.binning import compute_binned
        from gui.analysis.pipeline import prepare_run_data

        if merged_data is not None:
            x = np.asarray(merged_data["x"], dtype=float)
            y = np.asarray(merged_data["y"], dtype=float)
            # Phase 4: a voltage-merged spectrum fitted in frequency
            # mode needs its bin centers Doppler-projected onto the
            # rest-frame MHz axis using the user-supplied
            # merge_metadata. The autofitter initial guesses live in
            # frequency space when bin_mode=="Frequency", so the
            # projection has to happen here.
            if (merged_data.get("x_unit") == "V"
                    and source_config.get("bin_mode") == "Frequency"):
                from gui.analysis.merge import (
                    project_voltage_merge_to_frequency)
                x, y, yerr, _warnings = (
                    project_voltage_merge_to_frequency(
                        merged_data, source_config))
                return x, y, yerr
            return x, y, np.sqrt(y + 1)

        # Shared load/interpret/Doppler prep so the auto-fitter's seed lives
        # in exactly the frame the real fit produces. Routing through
        # prepare_run_data also applies the composed Shift_Ref(ref_freq +
        # ref_shift) -- this path previously ran Compute_WL with no Shift_Ref,
        # so a non-zero Reference shift left the auto-fit seed offset from the
        # fit (code review 2026-06-02, autofitter-no-ref-shift).
        # In-process, so the registry singleton is reachable; the fit worker
        # gets a snapshot instead (see _start_fit).
        from gui.calibration import get_registry as _get_cal_registry

        data, eff_cfg, _meta = prepare_run_data(
            run_file, source_config,
            binning_override=binning_override,
            split_descriptor=split_descriptor,
            calibrations=_get_cal_registry().to_dict())
        bin_mode = eff_cfg.get("bin_mode", "Frequency")

        res = compute_binned(data, eff_cfg)
        # Apply the GP centroid correction post-binning so the
        # auto-fitter's centroid initial guess matches the frame
        # the real fit will produce: subtract correction_mhz from the
        # binned x-axis for Auto/Manual modes (skipped for Raw Voltage).
        if (centroid_correction
                and isinstance(centroid_correction, dict)
                and centroid_correction.get("mode")
                    in ("Auto", "Manual")
                and bin_mode != "Raw Voltage"):
            try:
                _corr_mhz = float(centroid_correction.get(
                    "correction_mhz", 0.0))
            except (TypeError, ValueError):
                _corr_mhz = 0.0
            if _corr_mhz != 0.0:
                res["x"] = res["x"] - _corr_mhz
        x, y = res["x"], res["y"]
        yerr = res["yerr"]
        if res["use_callable_yerr"]:
            yerr = np.sqrt(y + 1)
        return x, y, yerr

    def _apply_autofit_result(self, result, model_blocks):
        """Push auto-fit parameters back to Model blocks (undoable)."""
        changes = []       # (widget, old_val, new_val)
        affected = set()   # ModelBlock instances that changed

        for key, val in result.items():
            # key is 2-tuple (midx, pname) or 3-tuple (midx, pname, src_idx)
            if len(key) == 3:
                midx, pname, src_idx = key
                if src_idx != 0:
                    continue   # only apply first source's values to blocks
            else:
                midx, pname = key
            if midx >= len(model_blocks):
                continue
            block = model_blocks[midx]
            if pname.startswith("Amp"):
                line = pname[3:]
                for i in range(block._peaks_table.rowCount()):
                    label_item = block._peaks_table.item(i, 0)
                    if label_item and label_item.text() == line:
                        w = block._peaks_table.cellWidget(i, 1)
                        if w:
                            changes.append((w, w.value(), val))
                            affected.add(block)
                        break
            else:
                for row in block._param_rows:
                    if row["name"] == pname:
                        w = row["value"]
                        changes.append((w, w.value(), val))
                        affected.add(block)
                        break

        if not changes:
            print("[AutoFit] No matching parameters found", flush=True)
            return

        # Push as a single undoable command
        main_win = self.window()
        undo_stack = getattr(main_win, '_undo_stack', None)
        if undo_stack:
            cmd = _BatchSetValueCommand(changes, affected)
            undo_stack.push(cmd)   # push() calls redo() which sets values
        else:
            # Fallback: apply directly without undo support
            for w, _, new_val in changes:
                w.setValue(new_val)
            for block in affected:
                block.block_changed.emit()

        print(f"[AutoFit] Applied {len(changes)} parameters to "
              f"{len(affected)} model block(s)", flush=True)

    def _on_fit_requested(self):
        """Gather configs from all blocks and start fitting."""
        sources = self._get_blocks_by_type(SourceBlock)
        models = self._get_blocks_by_type(ModelBlock)
        fitters = self._get_blocks_by_type(FitterBlock)
        outputs = self._get_blocks_by_type(OutputBlock)

        if not sources:
            QMessageBox.warning(self, "Fit Error",
                                "No enabled Source block found.")
            return
        if not models:
            QMessageBox.warning(self, "Fit Error",
                                "No enabled Model block found.")
            return
        if not fitters:
            QMessageBox.warning(self, "Fit Error",
                                "No enabled Fitter block found.")
            return

        source_block = sources[0]
        fitter_block = fitters[0]
        output_block = outputs[0] if outputs else None

        checked_files = source_block.get_checked_files()
        if not checked_files:
            QMessageBox.warning(self, "Fit Error",
                                "No files selected in the Source block.")
            return

        run_files = [e["path"] for e in checked_files]
        merged_data_map = {}
        split_descriptors_map = {}
        file_overrides_map = {}
        for e in checked_files:
            if e.get("is_merged") and "merged_data" in e:
                # `merged_data["_entry"]` is a back-reference to the
                # GUI row (holds a QCheckBox + QWidget) used by the
                # "Remove" menu. The fit pool ships merged_data to a
                # child process via pickle, and Qt objects can't be
                # pickled. Strip the GUI hook on the way out.
                md = e["merged_data"]
                merged_data_map[e["path"]] = {
                    k: v for k, v in md.items() if k != "_entry"}
            if e.get("is_split"):
                split_descriptors_map[e["path"]] = {
                    "parent_path": e["parent_path"],
                    "source_id": e["source_id"],
                    "split_lo": float(e["split_lo"]),
                    "split_hi": float(e["split_hi"]),
                    "metadata_override": dict(
                        e.get("metadata_override") or {}),
                }
            ovr = e.get("binning_override") or {}
            if ovr and not e.get("is_merged"):
                file_overrides_map[e["path"]] = dict(ovr)
        source_config = source_block.get_source_config()

        # Validate physics parameters
        if source_config["mass"] <= 0:
            QMessageBox.warning(self, "Fit Error",
                                "Mass must be positive.")
            return
        if source_config["ref_freq"] == 0:
            QMessageBox.warning(self, "Fit Error",
                                "Reference frequency is zero. Check energy "
                                "levels (upper and lower must differ).")
            return
        if source_config["harmonic"] <= 0:
            QMessageBox.warning(self, "Fit Error",
                                "Harmonic must be positive.")
            return

        model_configs = [m.get_model_config() for m in models]
        fitter_config = fitter_block.get_fitter_config()

        if output_block:
            fitter_config["show_correl"] = output_block._show_correl.isChecked()
            fitter_config["min_correl"] = output_block._min_correl.value()

        output_config = output_block.get_output_config() if output_block else {}

        # Validate parameter expressions before any heavy lifting.
        source_names = []
        for p in run_files:
            if p in merged_data_map:
                source_names.append(
                    source_name_for_merged(merged_data_map[p]))
            elif p in split_descriptors_map:
                source_names.append(source_name_for_descriptor({
                    "kind": "virtual_split",
                    "source_id": split_descriptors_map[p]["source_id"],
                }))
            else:
                source_names.append(source_name_for_path(p))
        fit_mode = ("separate"
                    if fitter_config.get("separate", True)
                    else "simultaneous")
        issues = validate_configs(
            model_configs, fitter_config, source_names, fit_mode)
        errors = [i for i in issues if i.level == "error"]
        warnings_ = [i for i in issues if i.level == "warning"]
        if errors:
            _show_validation_errors(self, errors, warnings_)
            return
        if warnings_ and not _confirm_validation_warnings(self, warnings_):
            return

        # Snapshot model parameters before fitting (for revert)
        self._pre_fit_snapshot = [m.to_dict() for m in models]

        # Start worker thread
        fitter_block.set_running(True)
        fitter_block.set_progress(0, len(run_files), "Starting...")

        # Reference centroid correction (Phase D): the IS-tab panel
        # owns the GP corrector and per-file overrides. We snapshot the
        # corrections in the main thread so the worker (which may run
        # in subprocesses) doesn't have to navigate the widget tree.
        corrections_map = self._build_corrections_map(run_files)

        # Snapshot per-file scan exclusions from the global registry
        # in this (parent) thread; subprocess workers can't reach the
        # registry singleton, so we hand them the map up front.
        from gui.scan_filter import get_registry as _get_sf_registry
        sf_reg = _get_sf_registry()
        scan_filters_map = {}
        for entry in checked_files:
            if entry.get("is_merged"):
                continue
            key = (entry.get("parent_path")
                   if entry.get("is_split") else entry.get("path"))
            if not key:
                continue
            ex = sf_reg.get(key)
            if ex:
                scan_filters_map[key] = ex

        # Same story for the voltage calibrations, with one difference: the
        # whole registry travels, not just the checked runs. A "borrow" spec
        # names a donor run that need not itself be part of this fit, and the
        # worker has to be able to resolve it.
        from gui.calibration import get_registry as _get_cal_registry
        calibrations_map = _get_cal_registry().to_dict()

        self._fit_worker = FitWorkerThread(
            run_files=run_files,
            source_config=source_config,
            model_configs=model_configs,
            fitter_config=fitter_config,
            output_config=output_config,
            n_cores=self._n_cores,
            merged_data_map=merged_data_map,
            split_descriptors_map=split_descriptors_map,
            file_overrides_map=file_overrides_map,
            centroid_corrections_map=corrections_map,
            scan_filters_map=scan_filters_map,
            calibrations_map=calibrations_map,
        )
        self._fit_worker.progress.connect(
            lambda c, t, s: fitter_block.set_progress(c, t, s))
        self._fit_worker.progress.connect(self.fit_progress.emit)
        self._fit_worker.fit_done.connect(self._on_single_fit_done)
        self._fit_worker.all_done.connect(
            lambda results: self._on_all_fits_done(
                results, fitter_block, output_config))
        self._fit_worker.error.connect(
            lambda msg: self._on_fit_error(msg, fitter_block))
        # Safety net: if thread terminates (e.g., forced cancel), re-enable UI
        self._fit_worker.finished.connect(
            lambda: fitter_block.set_running(False))
        self._fit_worker.start()

    def _on_fit_cancel(self):
        if self._fit_worker:
            self._fit_worker.cancel()
            # Re-enable UI immediately -- don't wait for all_done
            fitters = self._get_blocks_by_type(FitterBlock)
            for fb in fitters:
                fb.set_running(False)
                fb.set_progress(0, 0, "Cancelled")

    def _on_single_fit_done(self, result):
        """Called when a single run fit completes."""
        pass  # Could update a live status display

    def _on_all_fits_done(self, results, fitter_block, output_config):
        """Called when all fits are complete."""
        print(f"[Project] _on_all_fits_done: {len(results)} results")
        # Order runs by run number so the fit report, the per-run CSVs,
        # and the saved figures all follow the same numeric ordering
        # (rather than the nondeterministic ProcessPool completion order).
        results = sorted(
            results, key=lambda r: run_sort_key(r.get("run_number", "")))
        fitter_block.set_running(False)
        # Surface the centroid-correction summary in the GUI status
        # line. Phase D writes centroid_correction_applied=True iff
        # the spectrum was actually shifted (or a Manual override
        # was honored) at fit time. Using that flag means: Raw
        # Voltage fits don't false-positive (their x-axis is in
        # volts, no MHz shift can land), and Manual rows with σ=0
        # don't false-negative (the user provided a confident value
        # and the shift still happened).
        n_attempted = sum(
            1 for r in results
            if r.get("success") and bool(
                (r.get("run_metadata") or {}).get(
                    "centroid_correction_applied", False)))
        n_ok = sum(1 for r in results if r.get("success"))
        if n_attempted:
            done_text = (f"Complete — GP correction applied to "
                         f"{n_attempted}/{n_ok} runs")
        else:
            done_text = "Complete"
        fitter_block.set_progress(len(results), len(results), done_text)

        # Save results to disk
        try:
            self._save_results(results, output_config)
        except Exception as e:
            import traceback
            print(f"[Project] _save_results error: {e}")
            traceback.print_exc()

        # Cache for Isotope Shift tab
        self._last_results = results
        self._last_output_config = output_config

        # Emit to Results tab
        self.results_ready.emit(self._project_name, results, output_config)

    def _rebuild_chain_diagnostics(self, results, output_config):
        """Backfill walk/correlation data from the persisted MCMC chains.

        At fit time those arrays are only built when their boxes were
        ticked; "Update iteration" lets the user tick them afterwards, so
        rebuild the same structures by re-reading the chain ``.h5``
        (original path, else the copy in the iteration's ``chains/``).
        There is no live fitter here, so the vary-mask is recovered from
        the chain itself: fixed parameters never move, their column
        spread is exactly zero.

        Returns the artifact names that were requested but genuinely
        need a re-run (chi-square map / confidence bands need the live
        fitter; walk/correlation without any saved chain).
        """
        want_walk = bool(output_config.get("walk_plot"))
        want_correl = bool(output_config.get("correl_plot"))
        skipped = set()
        ok_results = [r for r in results if r.get("success")]
        if output_config.get("chisq_map") and any(
                not (r.get("diagnostics") or {}).get("chisq_data")
                for r in ok_results):
            skipped.add("chi-square map")
        if output_config.get("confidence_bands") and any(
                not (r.get("diagnostics") or {}).get("confidence_bands")
                for r in ok_results):
            skipped.add("confidence bands")
        if not (want_walk or want_correl):
            return sorted(skipped)

        diag_sel = output_config.get("diag_params", [])
        for r in ok_results:
            diag = r.setdefault("diagnostics", {})
            need_walk = want_walk and not diag.get("walk_data")
            need_correl = want_correl and not diag.get("correl_data")
            if not (need_walk or need_correl):
                continue
            chain_file = r.get("chain_file")
            if not (chain_file and os.path.isfile(chain_file)):
                cand = os.path.join(
                    self._last_iter_dir or "", "chains",
                    f"chain_run_{r.get('run_number')}.h5")
                chain_file = cand if os.path.isfile(cand) else None
            if not chain_file:
                skipped.add("walk/correlation plots (no saved MCMC "
                            "chain — leastsq fit?)")
                continue
            try:
                import h5py
                with h5py.File(chain_file, "r") as hf:
                    g = hf["mcmc"]
                    chain = g["chain"][:]       # (steps, walkers, ndim)
                    labels = list(g.attrs["labels"])
                labels = [l.decode() if isinstance(l, bytes) else str(l)
                          for l in labels]
                spread = chain.reshape(-1, chain.shape[-1]).std(axis=0)
                var_mask = [i for i, s in enumerate(spread) if s > 0]
                var_labels = [labels[i].split("___")[-1] for i in var_mask]
                var_chain = chain[:, :, var_mask]
                if diag_sel:
                    sel = [j for j, vl in enumerate(var_labels)
                           if vl in diag_sel]
                    if sel:
                        var_labels = [var_labels[j] for j in sel]
                        var_chain = var_chain[:, :, sel]
                if need_walk:
                    diag["walk_data"] = {"labels": var_labels,
                                         "chain": var_chain}
                if need_correl:
                    flat = var_chain.reshape((-1, len(var_labels)))
                    diag["correl_data"] = {"labels": var_labels,
                                           "flatchain": flat}
            except Exception:
                session_log.get_logger("analysis").exception(
                    "chain re-read failed for run %s (%s)",
                    r.get("run_number"), chain_file)
                skipped.add("walk/correlation plots (chain unreadable)")
        return sorted(skipped)

    def _reapply_outputs(self):
        """Update the LAST fit's iteration: regenerate its output files
        with the CURRENT output options, WITHOUT re-running the fit.

        Use case: the fit ran, then the user realises an output was
        missing (e.g. ToF or tracker plots) -- tick it and Update
        iteration, and the same iteration is updated in place rather
        than spawning a new one. Works because the per-run result dicts
        (including the always-computed ToF histogram) are cached in
        ``self._last_results`` for the session; newly-ticked MCMC
        walk/correlation plots are rebuilt from the persisted chains.
        Enabling outputs ADDS their files; previously-written files for
        now-disabled outputs are left untouched.
        """
        if not self._last_results:
            QMessageBox.information(
                self, "Update iteration",
                "No fit results from this session to update.\n\n"
                "Run a fit first, then tick the outputs you want (e.g. "
                "ToF plots) and Update iteration to add them.")
            return
        if not self._last_iter_dir or not os.path.isdir(self._last_iter_dir):
            QMessageBox.warning(
                self, "Update iteration",
                "The last iteration folder could not be found.\n"
                "Re-run the fit to create it.")
            return
        outputs = self._get_blocks_by_type(OutputBlock)
        output_block = outputs[0] if outputs else None
        if output_block is None:
            QMessageBox.warning(self, "Update iteration",
                                "No enabled Output block found.")
            return
        output_config = output_block.get_output_config()
        skipped = self._rebuild_chain_diagnostics(
            self._last_results, output_config)
        try:
            self._save_results(self._last_results, output_config,
                               iter_dir_override=self._last_iter_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Update iteration",
                f"Failed to regenerate outputs:\n{e}")
            return
        self._last_output_config = output_config
        # Refresh the Results tab so the regenerated files appear.
        self.results_ready.emit(
            self._project_name, self._last_results, output_config)
        iter_name = os.path.basename(os.path.normpath(self._last_iter_dir))
        msg = f"Outputs updated for {iter_name} (no re-fit)."
        if skipped:
            msg += ("\n\nNot regenerable without a re-run: "
                    + "; ".join(skipped) + ".")
        QMessageBox.information(self, "Update iteration", msg)

    def _on_fit_error(self, msg, fitter_block):
        fitter_block.set_running(False)
        # code review 2026-06-02, worker-thread-exceptions-not-logged
        _log.error("Simultaneous fit failed: %s", msg)
        QMessageBox.critical(self, "Fit Error", msg)

    def _revert_fit(self):
        """Restore model parameters to their pre-fit values."""
        if not hasattr(self, '_pre_fit_snapshot') or not self._pre_fit_snapshot:
            QMessageBox.information(self, "Revert",
                                    "No pre-fit snapshot available.")
            return
        models = self._get_blocks_by_type(ModelBlock)
        for model_block, snapshot in zip(models, self._pre_fit_snapshot):
            model_block.from_dict(snapshot)
        QMessageBox.information(self, "Revert",
                                "Model parameters reverted to pre-fit values.")

    def _save_results(self, results, output_config, iter_dir_override=None):
        """Save fit results to the output directory.

        ``iter_dir_override`` (used by Re-apply outputs) writes into an
        EXISTING iteration directory instead of allocating a new
        iter_NNN, so output toggles can be regenerated in place.
        """
        import json
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from gui.analysis.binning import (
            build_binning_warnings, format_binning_warnings_text)

        # Determine output directory. Unified layout:
        # ``<output_root>/analysis/<project>/iter_NNN/``.
        from gui.shared_widgets import get_analysis_dir
        base_dir = get_analysis_dir()
        project_dir = os.path.join(base_dir, self._project_name)

        if iter_dir_override:
            iter_dir = iter_dir_override
            iter_name = os.path.basename(os.path.normpath(iter_dir))
        else:
            # Determine iteration
            iter_mode = output_config.get("iter_mode", "Auto")
            if iter_mode == "Auto":
                existing = [d for d in os.listdir(project_dir)
                            if d.startswith("iter_")] if os.path.isdir(
                                project_dir) else []
                nums = []
                for d in existing:
                    try:
                        nums.append(int(d.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
                next_num = max(nums, default=0) + 1
                iter_name = f"iter_{next_num:03d}"
            else:
                iter_name = output_config.get("iter_label", "manual") or "manual"

        iter_dir = os.path.join(project_dir, iter_name)
        # Remember for Re-apply outputs.
        self._last_iter_dir = iter_dir
        if iter_name not in self._session_iterations:
            self._session_iterations.append(iter_name)
        plots_dir = os.path.join(iter_dir, "plots")
        tracking_dir = os.path.join(iter_dir, "tracking")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(tracking_dir, exist_ok=True)

        # Save project config snapshot
        project_config = self.to_dict()
        with open(os.path.join(iter_dir, "config_snapshot.yaml"), "w") as f:
            yaml.dump(project_config, f, default_flow_style=False,
                      sort_keys=False)

        # Build binning warnings once (used by fit_report + json file).
        per_run_infos = [(str(r.get("run_number", "?")), r.get("binning_info"))
                         for r in results if r.get("success")]
        binning_warnings = build_binning_warnings(per_run_infos)
        # Human-readable log carrying the same warnings as the
        # fit_report.txt header. Written as plain UTF-8 text so cm⁻¹ /
        # σ / χ² render literally in the Results-tab text viewer rather
        # than as escaped \uXXXX sequences.
        warn_log_path = os.path.join(iter_dir, "binning_warnings.log")
        warn_text = format_binning_warnings_text(binning_warnings)
        if warn_text:
            with open(warn_log_path, "w", encoding="utf-8") as f:
                f.write(warn_text.lstrip("\n"))
        else:
            # Touch an empty file so Results-tab logic can still
            # detect "warnings were checked, none fired" vs "this
            # iteration is from a pre-Phase-4 run".
            open(warn_log_path, "w", encoding="utf-8").close()

        # Save fit reports. UTF-8 because the binning-warning header
        # routinely embeds Unicode glyphs (cm⁻¹, σ, χ², …). The default
        # cp1252 on Windows would crash on those.
        if output_config.get("report", True):
            with open(os.path.join(iter_dir, "fit_report.txt"), "w",
                       encoding="utf-8") as f:
                warn_text = format_binning_warnings_text(binning_warnings)
                if warn_text:
                    f.write(warn_text + "\n")
                for r in results:
                    if r.get("success"):
                        f.write(f"\n{'='*20} RUN {r['run_number']} "
                                f"{'='*20}\n")
                        rm = r.get("run_metadata", {})
                        if rm:
                            f.write(f"  Cooler V: {rm.get('cooler_v', '?'):.2f}"
                                    f"  |  Laser: {rm.get('laser_set', '?')}"
                                    f" cm-1  |  Harmonic: "
                                    f"{r.get('harmonic', '?')}\n")
                        # Surface the GP centroid correction at the
                        # top of the per-run report so a user reading
                        # fit_report.txt sees it without opening
                        # run_summary.csv. Only printed when the fit
                        # was actually corrected (the `applied` flag
                        # is the honest signal -- Raw Voltage runs
                        # never appear here even if the panel had a
                        # correction queued for them).
                        if bool(rm.get(
                                "centroid_correction_applied", False)):
                            corr_mhz = float(rm.get(
                                "centroid_correction_mhz", 0.0) or 0.0)
                            corr_sig = float(rm.get(
                                "centroid_correction_sigma_mhz", 0.0)
                                              or 0.0)
                            mode = rm.get("centroid_correction_mode")
                            # Label the σ correctly per mode:
                            # - Auto:    GP posterior σ at this t
                            # - Manual:  user-typed σ
                            # - merged:  share-weighted audit value
                            #            over the constituent σs --
                            #            the IS-tab propagation can
                            #            do better when it knows the
                            #            GP correlates the
                            #            constituents.
                            if mode == "Manual":
                                sigma_label = "σ_manual"
                            elif mode == "merged":
                                sigma_label = "σ_audit"
                            else:
                                sigma_label = "σ_GP"
                            note = ("audit value; per-constituent "
                                    "propagation is used for "
                                    "isotope-shift uncertainty"
                                    if mode == "merged" else
                                    "applied to centroid below")
                            f.write(
                                f"  GP centroid correction: "
                                f"{corr_mhz:+.4f} MHz "
                                f"({sigma_label} = {corr_sig:.4f} "
                                f"MHz, mode={mode!s}; {note}).\n")
                        f.write(r.get("report", ""))
                        # FWHM and peak positions
                        if r.get("fwhm"):
                            f.write("\n--- FWHM ---\n")
                            for mname, fwhm_d in r["fwhm"].items():
                                f.write(f"  {mname}: {fwhm_d['value']:.4f}"
                                        f" +/- {fwhm_d['uncertainty']:.4f}"
                                        f" MHz\n")
                        if r.get("peak_positions"):
                            f.write("\n--- Peak Positions [MHz] ---\n")
                            for mname, positions in r["peak_positions"].items():
                                pos_str = ", ".join(f"{p:.4f}"
                                                    for p in positions)
                                f.write(f"  {mname}: [{pos_str}]\n")
                    else:
                        f.write(f"\n{'='*20} RUN {r.get('run_number', '?')} "
                                f"FAILED {'='*20}\n")
                        f.write(r.get("error", "Unknown error") + "\n")

        # Save parameter CSV
        if output_config.get("params_csv", True):
            all_params = []
            for r in results:
                if r.get("success") and "params_df" in r:
                    df = pd.DataFrame(r["params_df"])
                    df["run_number"] = r["run_number"]
                    all_params.append(df)
            if all_params:
                params_out = pd.concat(all_params)
                # Rename and reorder columns for clarity
                params_out = params_out.rename(columns={
                    "Stderr": "Error", "Minimum": "Min", "Maximum": "Max",
                })
                col_order = ["run_number", "Parameter", "Value", "Error",
                             "Vary", "Min", "Max", "Source", "Model",
                             "Expression"]
                col_order = [c for c in col_order if c in params_out.columns]
                params_out[col_order].to_csv(
                    os.path.join(iter_dir, "parameters.csv"), index=False)

        # Save metadata CSV
        if output_config.get("metadata_csv", True):
            all_meta = []
            for r in results:
                if r.get("success") and "metadata_df" in r:
                    df = pd.DataFrame(r["metadata_df"])
                    df["run_number"] = r["run_number"]
                    all_meta.append(df)
            if all_meta:
                meta_out = pd.concat(all_meta)
                # Rename columns for clarity
                meta_out = meta_out.rename(columns={
                    "Aic": "AIC", "Bic": "BIC",
                    "Redchi": "Reduced Chi-sq",
                    "Chisquare": "Chi-sq",
                    "Function evaluations": "Func. evals",
                })
                col_order = ["run_number", "Source", "Fitting method",
                             "Data points", "Variables", "Chi-sq",
                             "Reduced Chi-sq", "AIC", "BIC",
                             "Func. evals", "Message"]
                col_order = [c for c in col_order if c in meta_out.columns]
                meta_out[col_order].to_csv(
                    os.path.join(iter_dir, "metadata.csv"), index=False)

        # Save run summary CSV with experimental parameters
        run_summary = []
        binning_summary = []
        for r in results:
            if r.get("success"):
                rm = r.get("run_metadata", {})
                fq = r.get("fit_quality", {})
                bi = r.get("binning_info") or {}
                # JSON-encode the merged-spectrum per-file
                # correction list so the IS tab can rebuild proper
                # per-constituent propagation after a disk reload.
                # Empty string when there are no constituents (the
                # result wasn't a merged fit, or no per-file
                # corrections were applied at merge time).
                constituents = rm.get("correction_constituents") or []
                if constituents:
                    constituents_json = json.dumps(
                        constituents, separators=(",", ":"))
                else:
                    constituents_json = ""
                run_summary.append({
                    "run_number": r["run_number"],
                    "cooler_v": rm.get("cooler_v"),
                    "laser_set_cm1": rm.get("laser_set"),
                    "harmonic": r.get("harmonic"),
                    "daq_time_s": rm.get("daq_time"),
                    "n_events": rm.get("n_events"),
                    "ts_start": rm.get("ts_start"),
                    "ts_stop": rm.get("ts_stop"),
                    "date": rm.get("date"),
                    "centroid_correction_applied":
                        bool(rm.get("centroid_correction_applied",
                                     False)),
                    "centroid_correction_mode":
                        rm.get("centroid_correction_mode") or "",
                    "centroid_correction_mhz":
                        rm.get("centroid_correction_mhz"),
                    "centroid_correction_sigma_mhz":
                        rm.get("centroid_correction_sigma_mhz"),
                    "correction_constituents_json": constituents_json,
                    "redchi": fq.get("redchi"),
                    "chisqr": fq.get("chisqr"),
                    "bin_mode": bi.get("bin_mode"),
                    "bin_definition": bi.get("bin_definition"),
                    "effective_n_bins": bi.get("effective_n_bins"),
                    "effective_bin_width_mhz": bi.get("effective_bin_width_mhz"),
                    "binning_fallback": bi.get("fallback_used"),
                })
                # Trimmed to the fit-relevant binning facts. Dropped the
                # noise: constant "source"=clstools, the requested_*
                # values (redundant with effective_*), dx_min/dx_max and
                # the internal voltage_forced_auto / override_active /
                # common_grid_* flags. What survives is what actually
                # describes how each run was binned.
                binning_summary.append({
                    "run_number": r["run_number"],
                    "bin_mode": bi.get("bin_mode"),
                    "bin_definition": bi.get("bin_definition"),
                    "effective_n_bins": bi.get("effective_n_bins"),
                    "effective_bin_width_mhz": bi.get(
                        "effective_bin_width_mhz"),
                    "effective_bin_width_v": bi.get("effective_bin_width_v"),
                    "x_min": bi.get("x_min"),
                    "x_max": bi.get("x_max"),
                    "override_keys": ",".join(bi.get("override_keys") or []),
                    "fallback_used": bi.get("fallback_used"),
                })
        if run_summary:
            pd.DataFrame(run_summary).to_csv(
                os.path.join(iter_dir, "run_summary.csv"), index=False)
        if binning_summary:
            pd.DataFrame(binning_summary).to_csv(
                os.path.join(iter_dir, "binning_summary.csv"), index=False)

        # Weighted average of key parameters (#25)
        if len(results) > 1:
            try:
                import satlas2 as _sat
                successful = [r for r in results if r.get("success")]
                if len(successful) > 1:
                    all_p = pd.concat([pd.DataFrame(r["params_df"])
                                       for r in successful])
                    # Group by Parameter name and compute weighted average
                    wavg_rows = []
                    col_val = next((c for c in all_p.columns
                                    if "value" in c.lower()), None)
                    col_err = next((c for c in all_p.columns
                                    if c.lower() in ("stderr", "error")),
                                   None)
                    col_par = next((c for c in all_p.columns
                                    if "param" in c.lower()), None)
                    if col_val and col_err and col_par:
                        for pname, grp in all_p.groupby(col_par):
                            vals = grp[col_val].dropna().values
                            errs = grp[col_err].dropna().values
                            if len(vals) > 1 and len(errs) == len(vals):
                                mask = errs > 0
                                if mask.sum() > 1:
                                    avg, unc = _sat.weightedAverage(
                                        vals[mask], errs[mask])
                                    wavg_rows.append({
                                        "Parameter": pname,
                                        "WeightedAvg": float(avg),
                                        "Uncertainty": float(unc),
                                        "N": int(mask.sum()),
                                    })
                    if wavg_rows:
                        pd.DataFrame(wavg_rows).to_csv(
                            os.path.join(iter_dir, "weighted_averages.csv"),
                            index=False)
            except Exception:
                _log.warning("Failed to write weighted_averages.csv",
                             exc_info=True)

        # Save fit plots
        fmt = output_config.get("plot_format", "png")
        dpi = output_config.get("plot_dpi", 200)
        if output_config.get("fit_plots", True):
            for r in results:
                if not r.get("success"):
                    continue
                self._save_fit_plot(r, plots_dir, fmt, dpi, output_config)

        # ToF plots. The type dropdown selects ToF-only, ToF+spectrum, or
        # ToF+spectrum+fit (all show the applied gate highlighted).
        if output_config.get("tof_plots", False):
            tof_type = output_config.get("tof_plot_type", "ToF only")
            for r in results:
                if not r.get("success") or not r.get("tof_hist"):
                    continue
                if tof_type == "ToF only":
                    self._save_tof_plot(r, plots_dir, fmt, dpi)
                else:
                    self._save_tof_spectrum_plot(
                        r, plots_dir, fmt, dpi,
                        with_fit=(tof_type == "ToF + spectrum + fit"))

        # Save tracker plots
        tracked = output_config.get("tracked_params", {})
        if tracked and len(results) > 1:
            self._save_tracker_plots(results, tracked, tracking_dir,
                                     fmt, dpi, output_config)

        # Save diagnostic plots from raw data
        diag_dir = os.path.join(iter_dir, "diagnostics")
        has_diags = any(r.get("diagnostics") for r in results
                        if r.get("success"))
        if has_diags:
            os.makedirs(diag_dir, exist_ok=True)
            for r in results:
                if not r.get("success"):
                    continue
                diags = r.get("diagnostics", {})
                run_num = r["run_number"]
                self._save_diagnostic_plots(diags, run_num, diag_dir,
                                            fmt, dpi)

        # Persist MCMC chain files (#46)
        chains_dir = os.path.join(iter_dir, "chains")
        for r in results:
            if not r.get("success"):
                continue
            chain_file = r.get("chain_file")
            if chain_file and os.path.isfile(chain_file):
                os.makedirs(chains_dir, exist_ok=True)
                import shutil
                dest = os.path.join(
                    chains_dir,
                    f"chain_run_{r['run_number']}.h5")
                shutil.copy2(chain_file, dest)

        # Parameter comparison table (#28)
        successful = [r for r in results if r.get("success")]
        if len(successful) > 1:
            try:
                comp_rows = []
                for r in successful:
                    row = {"run_number": r["run_number"]}
                    fq = r.get("fit_quality", {})
                    row["redchi"] = fq.get("redchi")
                    for mname, fwhm_d in r.get("fwhm", {}).items():
                        row[f"{mname}_FWHM"] = fwhm_d.get("value")
                        row[f"{mname}_FWHM_err"] = fwhm_d.get("uncertainty")
                    df = pd.DataFrame(r.get("params_df", {}))
                    col_par = next((c for c in df.columns
                                    if "param" in c.lower()), None)
                    col_val = next((c for c in df.columns
                                    if "value" in c.lower()), None)
                    col_err = next((c for c in df.columns
                                    if c.lower() in ("stderr", "error")),
                                   None)
                    if col_par and col_val:
                        for _, prow in df.iterrows():
                            pn = str(prow[col_par])
                            row[pn] = prow[col_val]
                            if col_err:
                                row[f"{pn}_err"] = prow[col_err]
                    comp_rows.append(row)
                if comp_rows:
                    pd.DataFrame(comp_rows).to_csv(
                        os.path.join(iter_dir, "parameter_comparison.csv"),
                        index=False)
            except Exception:
                _log.warning("Failed to write parameter_comparison.csv",
                             exc_info=True)

    def _save_tof_plot(self, result, plots_dir, fmt, dpi):
        """Save a time-of-flight plot for one run: the ToF histogram with
        the applied ToF gate shaded on top, so post-processing users can
        judge the beam's width in time. Also writes a .npz sidecar so the
        Results tab can re-render / Edit-Plot it live.
        """
        import matplotlib.pyplot as plt

        hist = result.get("tof_hist") or {}
        centers = np.asarray(hist.get("centers", []), dtype=float)
        counts = np.asarray(hist.get("counts", []), dtype=float)
        if centers.size == 0 or counts.size == 0:
            return
        run_num = result["run_number"]
        gate = result.get("tof_gate")

        # Reconstruct bin edges from the 1 µs centres for a clean stairs
        # outline: each edge sits half a bin (0.5 µs) either side of a centre.
        edges = np.empty(centers.size + 1, dtype=float)
        edges[:-1] = centers - 0.5
        edges[-1] = centers[-1] + 0.5

        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.stairs(counts, edges, color="tab:green", linewidth=1.5,
                  label=f"Run {run_num}")
        if gate and len(gate) == 2 and gate[1] > gate[0]:
            ax.axvspan(gate[0], gate[1], alpha=0.2, color="orange",
                       label=f"ToF gate [{gate[0]:g}, {gate[1]:g}] µs")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Counts")
        ax.set_title(f"ToF — Run {run_num}", loc="left")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.margins(y=0.05)
        ax.autoscale_view()
        ax.set_xlim(float(centers.min() - 0.5), float(centers.max() + 0.5))
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(plots_dir, f"tof_run_{run_num}.{fmt}"),
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        np.savez(
            os.path.join(plots_dir, f"tof_run_{run_num}.npz"),
            plot_type="tof", run_num=run_num,
            centers=centers, counts=counts,
            tof_gate=(np.asarray(gate, dtype=float)
                      if gate else np.array([])))

    def _save_tof_spectrum_plot(self, result, plots_dir, fmt, dpi,
                                with_fit):
        """Two-panel figure per run: top = gated ToF histogram with the
        ToF gate highlighted (like the standalone ToF plot); bottom = the
        gated spectrum in the fit's x-axis, optionally with the fitted
        model overlaid. Writes a .npz sidecar for live Results rendering.
        """
        import matplotlib.pyplot as plt
        from gui.shared_widgets import count_errorbar_pairs

        hist = result.get("tof_hist") or {}
        centers = np.asarray(hist.get("centers", []), dtype=float)
        counts = np.asarray(hist.get("counts", []), dtype=float)
        x = np.asarray(result.get("x", []), dtype=float)
        y = np.asarray(result.get("y", []), dtype=float)
        if centers.size == 0 or counts.size == 0 or x.size == 0:
            return
        run_num = result["run_number"]
        gate = result.get("tof_gate")
        yerr = np.asarray(result.get("yerr", []), dtype=float)
        x_smooth = np.asarray(result.get("x_smooth", []), dtype=float)
        y_fit_smooth = np.asarray(result.get("y_fit_smooth", []),
                                  dtype=float)
        bin_mode = str((result.get("binning_info") or {}).get("bin_mode")
                       or "Frequency")
        x_label = ("Voltage (V)" if bin_mode == "Raw Voltage"
                   else "Frequency (MHz)")
        s = fit_plot_settings_for_counts(
            float(np.max(y)) if y.size else 0.0)

        edges = np.empty(centers.size + 1, dtype=float)
        edges[:-1] = centers - 0.5
        edges[-1] = centers[-1] + 0.5

        fig, (ax_tof, ax_spec) = plt.subplots(
            2, 1, figsize=(7.0, 6.0))

        # ── Top: ToF + gate ──
        ax_tof.stairs(counts, edges, color="tab:green", linewidth=1.5,
                      label=f"Run {run_num}")
        if gate and len(gate) == 2 and gate[1] > gate[0]:
            ax_tof.axvspan(gate[0], gate[1], alpha=0.2, color="orange",
                           label=f"ToF gate [{gate[0]:g}, {gate[1]:g}] µs")
        ax_tof.set_xlabel("Time (µs)")
        ax_tof.set_ylabel("Counts")
        ax_tof.set_title(f"ToF — Run {run_num}", loc="left")
        ax_tof.legend(fontsize=8)
        ax_tof.grid(True, alpha=0.3)
        ax_tof.margins(y=0.05)
        ax_tof.autoscale_view()
        ax_tof.set_xlim(float(centers.min() - 0.5),
                        float(centers.max() + 0.5))

        # ── Bottom: gated spectrum (+ optional fit) ──
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
        ax_spec.autoscale_view()
        ax_spec.set_xlim(float(np.min(x)), float(np.max(x)))

        fig.tight_layout(pad=0.4)
        fig.savefig(
            os.path.join(plots_dir, f"tof_spectrum_run_{run_num}.{fmt}"),
            dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        np.savez(
            os.path.join(plots_dir, f"tof_spectrum_run_{run_num}.npz"),
            plot_type="tof_spectrum", run_num=run_num,
            centers=centers, counts=counts,
            tof_gate=(np.asarray(gate, dtype=float)
                      if gate else np.array([])),
            x=x, y=y,
            yerr=(yerr if yerr.size == y.size else np.array([])),
            x_smooth=x_smooth, y_fit_smooth=y_fit_smooth,
            with_fit=bool(with_fit), bin_mode=bin_mode, x_label=x_label)

    def _save_fit_plot(self, result, plots_dir, fmt, dpi, output_config):
        """Generate and save a fit plot with optional residuals."""
        import matplotlib.pyplot as plt

        x = np.array(result["x"])
        y = np.array(result["y"])
        # Pick fit-plot defaults based on the highest data bin so that
        # low-stats fits get bigger, more opaque markers automatically.
        peak_count = float(np.max(y)) if len(y) else 0.0
        s = fit_plot_settings_for_counts(peak_count)
        yerr = np.array(result["yerr"])
        x_smooth = np.array(result["x_smooth"])
        y_fit_smooth = np.array(result["y_fit_smooth"])
        y_fit = np.array(result["y_fit"])
        residuals = np.array(result["residuals"])

        show_resid = output_config.get("residual_panel", True)
        hide_zero_bins = output_config.get("hide_zero_bins", False)
        fw, fh = s["figsize_w"], s["figsize_h"]

        # Plot-time filter: optionally drop zero-count bins from the
        # markers, error bars, residuals, AND from the fit-line overlay
        # so empty regions show no curve. The fit itself is already done;
        # this only affects how the plot is rendered.
        y_fit_smooth_plot = y_fit_smooth
        comp_mask = None
        if hide_zero_bins:
            nz = np.asarray(y) > 0
            x_plot, y_plot = x[nz], y[nz]
            yerr_plot = (yerr[nz] if np.ndim(yerr) > 0 and len(yerr) == len(y)
                         else yerr)
            res_plot = residuals[nz] if len(residuals) == len(y) else residuals
            # NaN-mask the smooth fit curve over zero-count bins. Build
            # bin edges as midpoints between adjacent data points; assign
            # each x_smooth sample to its nearest data bin via searchsorted;
            # set fit value to NaN where that bin is empty (matplotlib
            # breaks the line at NaNs).
            if len(x) >= 2 and len(x_smooth) > 0:
                xs = np.asarray(x, dtype=float)
                edges = np.empty(len(xs) + 1)
                edges[1:-1] = 0.5 * (xs[:-1] + xs[1:])
                edges[0] = xs[0] - 0.5 * (xs[1] - xs[0])
                edges[-1] = xs[-1] + 0.5 * (xs[-1] - xs[-2])
                bin_idx = np.clip(
                    np.searchsorted(edges, x_smooth) - 1,
                    0, len(xs) - 1)
                comp_mask = (np.asarray(y) == 0)[bin_idx]
                y_fit_smooth_plot = np.where(
                    comp_mask, np.nan, y_fit_smooth)
        else:
            x_plot, y_plot, yerr_plot, res_plot = x, y, yerr, residuals

        if show_resid:
            fig, (ax_main, ax_res) = plt.subplots(
                2, 1, figsize=(fw, fh), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
        else:
            fig, ax_main = plt.subplots(figsize=(fw, fh * 0.7))
            ax_res = None

        # Clip the lower error bar at y=0 -- histogram counts can't be
        # negative, and the symmetric sqrt(y+1) bar at y=0 otherwise
        # reaches to y=−1 which is visually nonsense.
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
        # Shaded area under the fit curve (publication style). "auto"
        # shade color = the fit-line color at low alpha.
        if s.get("shade_under_fit", False):
            _shade_c = s.get("shade_color", "auto")
            if not _shade_c or _shade_c == "auto":
                _shade_c = s["fit_color"]
            ax_main.fill_between(
                x_smooth, 0.0, y_fit_smooth_plot, color=_shade_c,
                alpha=float(s.get("shade_alpha", 0.12)), lw=0, zorder=1)
        # Per-model component overlay
        overlay_on = output_config.get("component_overlay", False)
        selected_comps = set(output_config.get("selected_components", []))
        if overlay_on:
            import matplotlib.pyplot as _plt
            _comp_colors = _plt.rcParams['axes.prop_cycle'].by_key()['color']
            comp = result.get("component_curves", {})
            for ci, (cname, cdata) in enumerate(comp.items()):
                if selected_comps and cname not in selected_comps:
                    continue
                cc = _comp_colors[(ci + 1) % len(_comp_colors)]
                cdata_arr = np.array(cdata)
                if comp_mask is not None and len(cdata_arr) == len(comp_mask):
                    cdata_arr = np.where(comp_mask, np.nan, cdata_arr)
                ax_main.plot(x_smooth, cdata_arr, color=cc,
                             lw=s["fit_lw"] * 0.7, alpha=0.7, ls='--',
                             label=cname)
        # Confidence bands from MCMC
        bands = result.get("diagnostics", {}).get("confidence_bands")
        if bands:
            bx = np.array(bands["x"])
            ax_main.fill_between(bx, bands["lo_1sigma"], bands["hi_1sigma"],
                                 color=s["fit_color"], alpha=s["band_alpha"],
                                 label='1\u03c3 band')
        ax_main.set_ylabel("Counts", fontsize=s["label_size"])
        ax_main.set_title(f"Run {result['run_number']}", loc='left',
                          fontsize=s["title_size"])
        # On-plot fit-values box (params picked in the Output block).
        values_lines = []
        if output_config.get("values_on_plot") and "params_df" in result:
            from gui.shared_widgets import format_fit_value_lines
            values_lines = format_fit_value_lines(
                result["params_df"],
                output_config.get("values_params") or [])
            if values_lines:
                ax_main.text(
                    0.98, 0.98, "\n".join(values_lines),
                    transform=ax_main.transAxes, ha="right", va="top",
                    fontsize=s.get("values_box_fontsize", 11),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="black", lw=0.8, alpha=0.9),
                    zorder=6)
        ax_main.legend(loc="upper left")
        ax_main.grid(True, alpha=s["grid_alpha"])
        # Hug the DATA x-range. margins()/autoscale would include the fit
        # curve, which is evaluated on a wider grid than the data and so
        # left an empty margin on both sides. Pin x to the data extent
        # instead (residual panel follows via sharex). Keep a little y
        # headroom so peaks/error bars aren't clipped.
        ax_main.margins(y=0.05)
        ax_main.autoscale_view()
        if len(x_plot) > 0:
            ax_main.set_xlim(float(np.min(x_plot)), float(np.max(x_plot)))

        # Pick the x-axis label from the actual bin mode used at fit
        # time; fall back to Frequency (the historical default) when
        # the result lacks the binning_info breadcrumb.
        bm = (result.get("binning_info", {}).get("bin_mode")
              or "Frequency")
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
            plt.setp(ax_main.get_xticklabels(), visible=False)
        else:
            ax_main.set_xlabel(x_label, fontsize=s["label_size"])

        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(plots_dir,
                                 f"fit_run_{result['run_number']}.{fmt}"),
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Save raw data as .npz for live re-rendering in Results tab.
        # bin_mode goes in so the Results tab can label the x-axis
        # correctly for Raw Voltage fits (was hard-coded MHz).
        npz_kwargs = dict(
            plot_type="fit",
            run_num=result["run_number"],
            bin_mode=bm,
            # Pre-formatted values-box lines: the Results tab re-render
            # draws them verbatim (old .npz without this key → no box).
            # Unicode dtype so the .npz stays loadable without pickle.
            values_lines=np.array(values_lines, dtype=str),
            x=x, y=y, yerr=yerr,
            x_smooth=x_smooth, y_fit_smooth=y_fit_smooth,
            y_fit=y_fit, residuals=residuals,
            show_resid=show_resid,
            hide_zero_bins=hide_zero_bins,
            component_overlay=overlay_on)
        # Store which components were selected (empty = all)
        if selected_comps:
            npz_kwargs["selected_components"] = sorted(selected_comps)
        # Per-model component curves (always stored for Edit Plot access)
        comp = result.get("component_curves", {})
        if comp:
            npz_kwargs["component_names"] = list(comp.keys())
            for i, (cname, cdata) in enumerate(comp.items()):
                npz_kwargs[f"component_{i}"] = np.asarray(cdata)
        if bands:
            npz_kwargs["band_x"] = np.asarray(bands["x"])
            npz_kwargs["band_lo"] = np.asarray(bands["lo_1sigma"])
            npz_kwargs["band_hi"] = np.asarray(bands["hi_1sigma"])
        np.savez(os.path.join(plots_dir,
                              f"fit_run_{result['run_number']}.npz"),
                 **npz_kwargs)

    def _save_diagnostic_plots(self, diags, run_num, diag_dir, fmt, dpi):
        """Generate diagnostic plots from raw data and save PNGs + .npz."""
        import matplotlib.pyplot as plt

        # Walk plot
        if "walk_data" in diags:
            ws = get_plot_type_settings("walk_plot")
            wd = diags["walk_data"]
            labels = wd["labels"]
            chain = np.asarray(wd["chain"])  # (steps, walkers, ndim)
            n_var = len(labels)
            fig, axes = plt.subplots(
                n_var, 1,
                figsize=(ws["figsize_w"], ws["panel_h"] * n_var),
                sharex=True, squeeze=False)
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
            fig.tight_layout(pad=0.3)
            base = f"walk_plot_run_{run_num}"
            fig.savefig(os.path.join(diag_dir, f"{base}.{fmt}"),
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            np.savez_compressed(os.path.join(diag_dir, f"{base}.npz"),
                                labels=np.array(labels, dtype=object),
                                chain=chain,
                                plot_type="walk", run_num=str(run_num))

        # Correlation / corner plot
        if "correl_data" in diags:
            cs = get_plot_type_settings("correlation_plot")
            cd = diags["correl_data"]
            labels = cd["labels"]
            flat = np.asarray(cd["flatchain"])
            n_var = len(labels)
            cell = cs["cell_size"]
            fig, axes = plt.subplots(n_var, n_var,
                                     figsize=(cell * n_var, cell * n_var))
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
            fig.tight_layout(pad=0.3)
            base = f"correl_plot_run_{run_num}"
            fig.savefig(os.path.join(diag_dir, f"{base}.{fmt}"),
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            np.savez_compressed(os.path.join(diag_dir, f"{base}.npz"),
                                labels=np.array(labels, dtype=object),
                                flatchain=flat,
                                plot_type="correl", run_num=str(run_num))

        # Chi-square map
        if "chisq_data" in diags:
            qs = get_plot_type_settings("chisq_map")
            scans = diags["chisq_data"]
            n_var = len(scans)
            fig, axes = plt.subplots(
                1, n_var,
                figsize=(qs["panel_w"] * n_var, qs["panel_h"]),
                squeeze=False)
            all_labels, all_scans, all_chisq, all_best = [], [], [], []
            for idx, sd in enumerate(scans):
                ax = axes[0, idx]
                ax.plot(sd["scan"], sd["chisq"], qs["line_fmt"],
                        ms=qs["line_ms"])
                ax.axvline(sd["best"], color=qs["best_color"],
                           ls=qs["best_ls"], alpha=0.5)
                from gui.analysis.helpers import param_axis_label
                ax.set_xlabel(param_axis_label(sd["label"]),
                              fontsize=qs["label_size"])
                ax.set_ylabel(r"$\chi^2$", fontsize=qs["label_size"])
                ax.tick_params(labelsize=qs["tick_size"])
                all_labels.append(sd["label"])
                all_scans.append(sd["scan"])
                all_chisq.append(sd["chisq"])
                all_best.append(sd["best"])
            fig.suptitle(f"Chi-sq Map \u2014 Run {run_num}",
                         fontsize=qs["title_size"])
            fig.tight_layout(pad=0.3)
            base = f"chisq_map_run_{run_num}"
            fig.savefig(os.path.join(diag_dir, f"{base}.{fmt}"),
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            np.savez_compressed(os.path.join(diag_dir, f"{base}.npz"),
                                labels=np.array(all_labels, dtype=object),
                                scans=np.array(all_scans),
                                chisq=np.array(all_chisq),
                                best=np.array(all_best),
                                plot_type="chisq", run_num=str(run_num))

    def _save_tracker_plots(self, results, tracked_params, tracking_dir,
                            fmt, dpi, output_config):
        """Generate parameter tracking plots."""
        import matplotlib.pyplot as plt
        import pandas as pd

        successful = [r for r in results if r.get("success")]
        if not successful:
            return

        x_mode = output_config.get("tracker_xaxis", "Run number")
        use_time_axis = (x_mode == "Start time")
        if use_time_axis:
            import matplotlib.dates as mdates
            from datetime import datetime

        for model_name, param_list in tracked_params.items():
            if not param_list:
                continue

            ts = get_plot_type_settings("tracker_plot")
            n_params = len(param_list)
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            fig, axes = plt.subplots(
                n_params, 1,
                figsize=(ts["figsize_w"], ts["panel_h"] * n_params),
                sharex=True, squeeze=False)

            for j, param in enumerate(param_list):
                ax = axes[j, 0]
                color = colors[j % len(colors)]
                runs = []
                vals = []
                errs = []
                times = []

                for r in successful:
                    df = pd.DataFrame(r["params_df"])
                    # Find matching row
                    col_model = next(
                        (c for c in df.columns if "model" in c.lower()), None)
                    col_param = next(
                        (c for c in df.columns if "param" in c.lower()), None)
                    col_value = next(
                        (c for c in df.columns if "value" in c.lower()), None)
                    col_unc = next(
                        (c for c in df.columns
                         if any(s in c.lower()
                                for s in ["stderr", "unc", "err"])), None)

                    if not all([col_model, col_param, col_value]):
                        continue

                    sel = df[
                        df[col_model].astype(str).str.contains(
                            model_name, case=False, na=False) &
                        df[col_param].astype(str).str.fullmatch(
                            param, case=False)
                    ]
                    if sel.empty:
                        continue

                    runs.append(str(r["run_number"]))
                    vals.append(float(sel.iloc[0][col_value]))
                    err = 0.0
                    if col_unc and pd.notna(sel.iloc[0][col_unc]):
                        err = float(sel.iloc[0][col_unc])
                    errs.append(err)
                    t_sec = (r.get("run_metadata") or {}).get("ts_start", 0)
                    times.append(float(t_sec or 0))

                if not runs:
                    ax.text(0.5, 0.5, f"No data for {param}",
                            ha='center', va='center',
                            transform=ax.transAxes, color='gray')
                    continue

                # Cast run identifiers to int when they all look
                # numeric (the common case); fall back to strings if
                # any are virtual-split source_ids like "7509_A". The
                # latter renders as a categorical axis.
                try:
                    runs_arr = np.array([int(r) for r in runs])
                except (ValueError, TypeError):
                    runs_arr = np.array(runs, dtype=object)
                vals_arr = np.array(vals)
                errs_arr = np.array(errs)
                times_arr = np.array(times)

                # Sort by the chosen x-axis so line segments connect
                # in monotonic order (otherwise we get bow-tie shapes
                # when results come back out of run order).
                if use_time_axis and np.any(times_arr > 0):
                    order = np.argsort(times_arr)
                    x_plot = np.array([
                        mdates.date2num(datetime.fromtimestamp(t))
                        for t in times_arr[order]
                    ])
                else:
                    if runs_arr.dtype == object:
                        # Strings like "7509_A": sort by leading int.
                        try:
                            keys = [int(str(s).split("_")[0])
                                    for s in runs_arr]
                            order = np.argsort(keys)
                        except (ValueError, TypeError):
                            order = np.arange(len(runs_arr))
                    else:
                        order = np.argsort(runs_arr)
                    x_plot = runs_arr[order]
                runs_arr = runs_arr[order]
                vals_arr = vals_arr[order]
                errs_arr = errs_arr[order]
                times_arr = times_arr[order]

                # Academic style: unconnected points with caps; the
                # old ".-" line + polygon band connecting error bars
                # read as noise. A weighted-mean line with a HORIZONTAL
                # ±1σ band shows the stability instead.
                ax.errorbar(x_plot, vals_arr, yerr=errs_arr,
                            fmt=ts["marker_fmt"], color=color,
                            ms=ts["marker_ms"], capsize=ts["capsize"],
                            elinewidth=ts.get("elinewidth", 1.2),
                            zorder=3)
                if ts.get("connect_runs", False):
                    ax.plot(x_plot, vals_arr, color=color, lw=0.8,
                            alpha=0.5, zorder=2)
                if ts.get("mean_line", True) and len(vals_arr) > 1:
                    w = np.where(errs_arr > 0, 1.0 /
                                 np.maximum(errs_arr, 1e-300) ** 2, 0.0)
                    if np.any(w > 0):
                        mean = float(np.sum(w * vals_arr) / np.sum(w))
                        mean_err = float(1.0 / np.sqrt(np.sum(w)))
                    else:
                        mean = float(np.mean(vals_arr))
                        mean_err = 0.0
                    mc = ts.get("mean_color", "0.25")
                    ax.axhline(mean, color=mc,
                               ls=ts.get("mean_ls", "--"), lw=1.2,
                               zorder=1,
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
                ax.set_title(
                    f"{model_name}: {param[:1].upper()}{param[1:]}",
                    loc='left', fontsize=ts["title_size"])
                ax.grid(True, alpha=ts["grid_alpha"])
                if use_time_axis and np.any(times_arr > 0):
                    ax.xaxis_date()
                    ax.set_xticks(x_plot)
                    date_strs = [
                        datetime.fromtimestamp(t).strftime(
                            "%Y-%m-%d\n%H:%M")
                        for t in times_arr
                    ]
                    # Hide labels for points too close to their
                    # previous neighbour so multi-day plots with a
                    # cluster of nearby runs don't overlap. Tick marks
                    # remain at every data point.
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

            axes[-1, 0].set_xlabel(
                output_config.get("tracker_xaxis", "Run number"),
                fontsize=ts["label_size"])
            fig.tight_layout(pad=0.3)
            safe_name = model_name.replace(" ", "_")
            fig.savefig(os.path.join(tracking_dir,
                                     f"tracker_{safe_name}.{fmt}"),
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            # Save raw data as .npz for live re-rendering in Results tab
            tracker_npz = {"plot_type": "tracker",
                           "model_name": model_name,
                           "params": np.array(param_list, dtype=object),
                           "xlabel": x_mode,
                           "x_mode": x_mode}
            for j, param in enumerate(param_list):
                t_runs, t_vals, t_errs, t_times = [], [], [], []
                for r in successful:
                    df_r = pd.DataFrame(r["params_df"])
                    cm = next((c for c in df_r.columns
                               if "model" in c.lower()), None)
                    cp = next((c for c in df_r.columns
                               if "param" in c.lower()), None)
                    cv = next((c for c in df_r.columns
                               if "value" in c.lower()), None)
                    cu = next((c for c in df_r.columns
                               if any(s in c.lower()
                                      for s in ["stderr", "unc", "err"])),
                              None)
                    if not all([cm, cp, cv]):
                        continue
                    s = df_r[
                        df_r[cm].astype(str).str.contains(
                            model_name, case=False, na=False) &
                        df_r[cp].astype(str).str.fullmatch(
                            param, case=False)]
                    if s.empty:
                        continue
                    t_runs.append(str(r["run_number"]))
                    t_vals.append(float(s.iloc[0][cv]))
                    e = 0.0
                    if cu and pd.notna(s.iloc[0][cu]):
                        e = float(s.iloc[0][cu])
                    t_errs.append(e)
                    t_sec = (r.get("run_metadata") or {}).get("ts_start", 0)
                    t_times.append(float(t_sec or 0))
                # Cast run identifiers to int when all are numeric;
                # fall back to strings otherwise (split source_ids).
                try:
                    t_runs_arr = np.array([int(r) for r in t_runs])
                except (ValueError, TypeError):
                    t_runs_arr = np.array(t_runs, dtype=object)
                # Sort so the renderer also gets monotonic order.
                t_times_arr = np.array(t_times)
                if use_time_axis and np.any(t_times_arr > 0):
                    order = np.argsort(t_times_arr)
                else:
                    if t_runs_arr.dtype == object:
                        try:
                            keys = [int(str(s).split("_")[0])
                                    for s in t_runs_arr]
                            order = np.argsort(keys)
                        except (ValueError, TypeError):
                            order = np.arange(len(t_runs_arr))
                    else:
                        order = np.argsort(t_runs_arr)
                t_runs_arr = t_runs_arr[order]
                t_vals = list(np.array(t_vals)[order])
                t_errs = list(np.array(t_errs)[order])
                t_times_arr = t_times_arr[order]
                tracker_npz[f"runs_{j}"] = t_runs_arr
                tracker_npz[f"times_{j}"] = t_times_arr
                tracker_npz[f"vals_{j}"] = np.array(t_vals)
                tracker_npz[f"errs_{j}"] = np.array(t_errs)
            np.savez(os.path.join(tracking_dir,
                                  f"tracker_{safe_name}.npz"),
                     **tracker_npz)

        # Save single consolidated tracker CSV with all tracked params
        # Columns: run_number, param1, param1_error, param2, param2_error, ...
        all_tracked = []  # list of (model_name, param_name) tuples
        for mn, pl in tracked_params.items():
            for p in pl:
                all_tracked.append((mn, p))

        if all_tracked:
            csv_rows = []
            for r in successful:
                df = pd.DataFrame(r["params_df"])
                col_model = next(
                    (c for c in df.columns if "model" in c.lower()), None)
                col_param = next(
                    (c for c in df.columns if "param" in c.lower()), None)
                col_value = next(
                    (c for c in df.columns if "value" in c.lower()), None)
                col_unc = next(
                    (c for c in df.columns
                     if any(s in c.lower()
                            for s in ["stderr", "unc", "err"])), None)
                if not all([col_model, col_param, col_value]):
                    continue
                row = {"run_number": r["run_number"]}
                for mn, p in all_tracked:
                    sel = df[
                        df[col_model].astype(str).str.contains(
                            mn, case=False, na=False) &
                        df[col_param].astype(str).str.fullmatch(
                            p, case=False)
                    ]
                    if not sel.empty:
                        row[p] = float(sel.iloc[0][col_value])
                        err = 0.0
                        if col_unc and pd.notna(sel.iloc[0][col_unc]):
                            err = float(sel.iloc[0][col_unc])
                        row[f"{p}_error"] = err
                    else:
                        row[p] = np.nan
                        row[f"{p}_error"] = np.nan
                csv_rows.append(row)
            if csv_rows:
                # Build column order: run_number, then param/error pairs
                cols = ["run_number"]
                for _, p in all_tracked:
                    cols.extend([p, f"{p}_error"])
                pd.DataFrame(csv_rows)[cols].to_csv(
                    os.path.join(tracking_dir, "tracker.csv"),
                    index=False)

    # ── Serialization ──

    def to_dict(self, include_iterations=False):
        """Serialize the pipeline config.

        ``include_iterations`` adds the session's iteration names —
        wanted in the app save file (so Results can restore them on
        load) but NOT in the per-iteration ``config_snapshot.yaml``,
        which should describe the pipeline alone.
        """
        blocks_out = []
        for i, b in enumerate(self._blocks):
            bd = b.to_dict()
            bd["order"] = i
            blocks_out.append(bd)
        out = {
            "project_name": self._project_name,
            "is_reference": self._is_reference,
            "blocks": blocks_out,
        }
        if include_iterations and self._session_iterations:
            out["iterations"] = list(self._session_iterations)
        return out

    def from_dict(self, d):
        self._project_name = d.get("project_name", "Project")
        self._is_reference = bool(d.get("is_reference", False))
        self._session_iterations = [
            str(x) for x in (d.get("iterations") or [])]
        # Clear existing blocks
        for b in list(self._blocks):
            self._remove_block(b)

        block_defs = d.get("blocks", [])

        # Sort by explicit order if present; otherwise use default
        # type-based ordering: Source -> Model -> Fitter -> Output
        _TYPE_ORDER = {"Source": 0, "Model": 1, "Fitter": 2, "Output": 3}
        has_order = any("order" in bd for bd in block_defs)
        if has_order:
            block_defs = sorted(block_defs, key=lambda bd: bd.get("order", 0))
        else:
            block_defs = sorted(
                block_defs,
                key=lambda bd: _TYPE_ORDER.get(bd.get("type", ""), 99))

        for bd in block_defs:
            self.add_block(bd["type"], config=bd)

    @property
    def project_name(self):
        return self._project_name

    @property
    def is_reference(self):
        return self._is_reference

    def _find_reference_correction_panel(self):
        """Walk up the widget tree to locate the IsotopeShiftTab's
        Reference Correction panel. Returns None if the panel is not
        installed (older saves, headless contexts, etc.)."""
        p = self.parentWidget()
        while p is not None:
            is_tab = getattr(p, "_is_tab", None)
            if is_tab is not None:
                return getattr(is_tab, "_ref_corr_panel", None)
            p = p.parentWidget()
        return None

    def _build_corrections_map(self, run_files):
        """Build a ``{file_path: {correction_mhz, sigma_mhz}}`` dict
        from the Reference Correction panel for the current project.

        Honors the panel's global Apply toggle and per-row Mode
        ("Off" rows are skipped). Returns ``{}`` when the panel has
        no fitted corrector or the user is opted out. A short
        diagnostic line is printed for each call so the user can see
        which files got corrected and which were silently skipped."""
        panel = self._find_reference_correction_panel()
        if panel is None:
            print(f"[CentroidCorrection] {self._project_name}: no IS-tab "
                  f"panel reachable from project; no correction applied.",
                  flush=True)
            return {}
        if not getattr(panel, "apply_globally", False):
            print(f"[CentroidCorrection] {self._project_name}: panel "
                  f"global toggle is off; no correction applied.",
                  flush=True)
            return {}
        if getattr(panel, "corrector", None) is None:
            print(f"[CentroidCorrection] {self._project_name}: no GP "
                  f"corrector fitted; no correction applied.",
                  flush=True)
            return {}
        out: dict[str, dict] = {}
        skipped_no_entry: list[str] = []
        skipped_off: list[str] = []
        for fpath in run_files:
            c = panel.get_correction(self._project_name, fpath)
            if not c:
                skipped_no_entry.append(os.path.basename(fpath))
                continue
            mode = c.get("mode", "Auto")
            if mode == "Off":
                skipped_off.append(os.path.basename(fpath))
                continue
            try:
                mhz = float(c.get("correction_mhz", 0.0))
                sig = float(c.get("sigma_mhz", 0.0))
            except (TypeError, ValueError):
                continue
            out[fpath] = {
                "correction_mhz": mhz,
                "sigma_mhz": sig,
                "mode": mode,
            }
        print(f"[CentroidCorrection] {self._project_name}: "
              f"applied to {len(out)}/{len(run_files)} files "
              f"(skipped: no entry={len(skipped_no_entry)}, "
              f"mode=Off={len(skipped_off)}).",
              flush=True)
        if skipped_no_entry:
            print(f"[CentroidCorrection] {self._project_name}: files "
                  f"without precomputed correction (re-click 'Compute "
                  f"per-file corrections' in the IS panel after adding "
                  f"new runs): {', '.join(skipped_no_entry[:8])}"
                  + (f" (+{len(skipped_no_entry) - 8} more)"
                     if len(skipped_no_entry) > 8 else ""),
                  flush=True)
        return out

    def get_reference_observations(self):
        """Per-file (timestamp, centroid, sigma) tuples for a reference
        project's last fit results.

        The output is the input table consumed by the GP centroid
        corrector. Sample (non-reference) projects return an empty list.
        Observations with non-finite or non-positive sigma, non-finite
        timestamps or centroid, or a missing/zero ``ts_start`` are
        skipped: the GP needs positive variance and finite times.

        For simultaneous fits, every per-run result shares one global
        ``params_df``; centroids are disambiguated by filtering on the
        ``Source`` column derived from the run file name.

        Returns
        -------
        list[dict]
            Each entry has keys:

            - ``run_number`` (str)
            - ``label`` (str): human-readable label (basename of run file
              if available, else run_number)
            - ``run_file`` (str): absolute path to the source ASDF (may be
              empty for merged spectra)
            - ``ts_start`` (float): seconds since epoch from ASDF
              ``TSstart``
            - ``centroid_mhz`` (float)
            - ``sigma_mhz`` (float)
            - ``include`` (bool): default True; the GP UI flips this to
              exclude outliers
        """
        if not self._is_reference or not self._last_results:
            return []
        import math
        from cls_estimations.isotope_shift import extract_centroid
        from gui.analysis.naming import source_name_for_path

        obs = []
        for r in self._last_results:
            if not r.get("success"):
                continue
            params = r.get("params_df", {}) or {}
            run_file = r.get("run_file") or ""
            # Prefer the satlas2 source name stamped on the result by
            # the fitter -- merged spectra and virtual splits can't be
            # recovered from `run_file` (a merged "path" is the synthetic
            # `merged://<name>` key, and a split's name uses `source_id`,
            # not the file's basename). Fall back to deriving from
            # `run_file` only for legacy results that predate the stamp.
            src_name = r.get("source_name")
            if not src_name and run_file:
                src_name = source_name_for_path(run_file)
            c, sigma = extract_centroid(params, source_name=src_name)
            if c is None or not math.isfinite(c):
                continue
            if (sigma is None or not math.isfinite(sigma) or sigma <= 0):
                continue
            meta = r.get("run_metadata") or {}
            ts = meta.get("ts_start", 0)
            try:
                ts_f = float(ts)
            except (TypeError, ValueError):
                ts_f = 0.0
            if ts_f <= 0 or not math.isfinite(ts_f):
                continue
            run_num = str(r.get("run_number", "") or "")
            label = (os.path.basename(run_file)
                     if run_file else (run_num or "run"))
            obs.append({
                "run_number": run_num,
                "label": label,
                "run_file": run_file,
                "ts_start": ts_f,
                "centroid_mhz": float(c),
                "sigma_mhz": float(sigma),
                "include": True,
            })
        return obs
