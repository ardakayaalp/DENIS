"""Auto-Fitter dialog: multi-start local optimizer for HFS fitting.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Generates N random perturbations of the user's initial guess, runs a
satlas2 leastsq fit on each in parallel (ProcessPoolExecutor), ranks the
results by chi-square, and lets the user preview, refine around the best,
and accept a result back into the Model blocks. Supports both single-source
and simultaneous multi-source fits with shared parameters and expression
ties, built with the same model builder the regular fitter uses.

Depends on: gui.analysis.fitting, gui.analysis.naming,
gui.analysis.expr_validation, gui.shared_widgets; uses satlas2,
NumPy, matplotlib, and PySide6.
"""

import bisect
import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QWidget, QProgressBar,
)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# code review 2026-06-02, worker-thread-exceptions-not-logged
from gui import session_log

_log = session_log.get_logger("auto_fitter")


# ══════════════════════════════════════════════════════════════════
#  Standalone fit function (runs in subprocess)
# ══════════════════════════════════════════════════════════════════

def _build_models_on_source(source, model_configs, satlas2):
    """Build satlas2 models on a source from model_configs. Returns model_map."""
    model_map = {}
    for mc in model_configs:
        safe = re.sub(r'[^A-Za-z0-9_]', '_', mc["name"])
        params = mc["params"]
        mt = mc["type"]

        if mt == "HFS":
            hfs = satlas2.HFS(
                I=params["I"]["value"],
                J=[params["Jl"]["value"], params["Ju"]["value"]],
                A=[params["Al"]["value"], params["Au"]["value"]],
                B=[params["Bl"]["value"], params["Bu"]["value"]],
                C=[params.get("Cl", {}).get("value", 0),
                   params.get("Cu", {}).get("value", 0)],
                df=params["centroid"]["value"],
                scale=params["scale"]["value"],
                fwhmg=params["FWHMG"]["value"],
                fwhml=params["FWHML"]["value"],
                racah=mc.get("racah", True),
                name=safe,
            )
            for pname_v in ["Al", "Au", "Bl", "Bu", "Cl", "Cu",
                            "centroid", "scale", "FWHMG", "FWHML"]:
                if pname_v in params:
                    hfs.params[pname_v].vary = params[pname_v].get(
                        "vary", True)

            if not mc.get("racah", True):
                peak_amps = mc.get("peak_amplitudes", {})
                for line in hfs.lines:
                    amp_key = f"Amp{line}"
                    if line in peak_amps:
                        hfs.params[amp_key].value = \
                            peak_amps[line]["value"]
                        hfs.params[amp_key].vary = \
                            peak_amps[line].get("vary", True)
                    else:
                        hfs.params[amp_key].vary = False

            source.addModel(hfs)
            model_map[safe] = hfs

            bkg_val = params.get("Bkg_p0", {}).get("value", 0)
            bkg = satlas2.Polynomial([bkg_val], name=f"{safe}_bkg")
            bkg.params["p0"].vary = params.get(
                "Bkg_p0", {}).get("vary", True)
            source.addModel(bkg)
            model_map[f"{safe}_bkg"] = bkg

        elif mt == "Voigt":
            voigt = satlas2.Voigt(
                params["A"]["value"], params["mu"]["value"],
                params["FWHMG"]["value"], params["FWHML"]["value"],
                name=safe)
            for pname_v in ["A", "mu", "FWHMG", "FWHML"]:
                if pname_v in params:
                    voigt.params[pname_v].vary = params[pname_v].get(
                        "vary", True)
            source.addModel(voigt)
            model_map[safe] = voigt
            bkg_val = params.get("Bkg_p0", {}).get("value", 0)
            bkg = satlas2.Polynomial([bkg_val], name=f"{safe}_bkg")
            bkg.params["p0"].vary = params.get(
                "Bkg_p0", {}).get("vary", True)
            source.addModel(bkg)
            model_map[f"{safe}_bkg"] = bkg

    return model_map


def _set_param(model_map, mc_name, pname, val):
    """Set a parameter value on the correct model in model_map."""
    safe = re.sub(r'[^A-Za-z0-9_]', '_', mc_name)
    if pname == "Bkg_p0":
        m = model_map.get(f"{safe}_bkg")
        if m:
            m.params["p0"].value = val
    else:
        m = model_map.get(safe)
        if m and pname in m.params:
            m.params[pname].value = val


def _get_param(model_map, mc_name, pname):
    """Read a fitted parameter value from model_map."""
    safe = re.sub(r'[^A-Za-z0-9_]', '_', mc_name)
    if pname == "Bkg_p0":
        m = model_map.get(f"{safe}_bkg")
        return float(m.params["p0"].value) if m else 0.0
    m = model_map.get(safe)
    if m and pname in m.params:
        return float(m.params[pname].value)
    return 0.0


def _run_one_fit(sources_data, model_configs, perturbed_values,
                 vary_keys, shared_params=None, shared_all_params=None,
                 fitter_config=None):
    """Build models, apply perturbed values, fit, return result.

    This runs in a subprocess via ProcessPoolExecutor.

    Parameters
    ----------
    sources_data : list of dict
        Each dict has keys "name", "x", "y", "yerr".  Length 1 for separate
        mode, >1 for simultaneous.
    shared_params : list of str or None
        Model-local parameter names shared across sources
        (fitter.shareModelParams). Ignored when len(sources_data) == 1.
    shared_all_params : list of str or None
        'All'-mode (full-name) shares routed through fitter.shareParams.
    fitter_config : dict or None
        The fitter-block config. When given, the fit honors the user's
        objective (method / scale_covar / Poisson-Gaussian llh) and applies
        the fitter-block expressions and Gaussian priors, so the auto-fit
        explores the same space the real fit will. None reduces to the
        previous leastsq / scale_covar=False behaviour.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        import satlas2
        import numpy as _np

        multi = len(sources_data) > 1

        # Build one source per data set with the shared model builder
        # fitting._build_models_on_source so parameter bounds (min/max),
        # every model type, and sidepeaks are identical to a regular Run.
        # The per-source model map is read back from ``src.models`` (a
        # list of (name, model) tuples) so it is complete regardless of
        # the builder's return value, letting the Auto-Fitter iterate
        # only the non-fixed parameters over the full model set.
        from gui.analysis.fitting import (
            _build_models_on_source as _build_canonical)
        from gui.analysis.naming import full_param_name, is_fit_param_name
        from gui.analysis.expr_validation import lower_model_expression

        fitter = satlas2.Fitter()
        sources = []
        model_maps = []   # one model_map per source
        for sd in sources_data:
            src = satlas2.Source(
                sd["x"], sd["y"], yerr=sd["yerr"], name=sd["name"])
            _build_canonical(src, model_configs, satlas2)
            mmap = {name: model for name, model in src.models}
            fitter.addSource(src)
            sources.append(src)
            model_maps.append(mmap)

        # Share parameters across sources (simultaneous mode). 'All'-mode
        # shares go through shareParams (full-name scope, higher priority);
        # model-local shares through shareModelParams -- mirroring the regular
        # simultaneous fit. The old code unioned both into shareModelParams,
        # mis-routing 'All' shares (code review 2026-06-02).
        if multi and shared_all_params:
            for pname in shared_all_params:
                try:
                    fitter.shareParams(pname)
                except Exception:
                    pass
        if multi and shared_params:
            for pname in shared_params:
                try:
                    fitter.shareModelParams(pname)
                except Exception:
                    pass

        # Apply the model-block expression ties: bare model-local names
        # are lowered to full lmfit names per source. Without this the
        # auto fit explores an unconstrained model and its accepted
        # result could fail to reproduce on a regular Run.
        for src in sources:
            for mc in model_configs:
                mt = mc.get("type", "")
                local_params = {pn for pn in mc["params"]
                                if is_fit_param_name(mt, pn)}
                for pname, pdata in mc["params"].items():
                    if not is_fit_param_name(mt, pname):
                        continue
                    expr = (pdata.get("expr") or "").strip()
                    if not expr:
                        continue
                    try:
                        fitter.setExpr(
                            full_param_name(src.name, mc["name"], pname),
                            lower_model_expression(
                                expr, src.name, mc["name"], local_params))
                    except Exception:
                        pass

        # Fitter-block expressions (already full names) + Gaussian priors,
        # mirroring the regular fit so the auto-fit explores the SAME
        # constrained space (code review 2026-06-02, autofitter dropped
        # priors/fitter-expressions). Failures are collected and returned so
        # the dialog can surface them instead of silently fitting unconstrained.
        fcfg = fitter_config or {}
        expr_errors = []
        for _full, _expr in (fcfg.get("expressions") or {}).items():
            try:
                fitter.setExpr(_full, _expr)
            except Exception as _ex:
                expr_errors.append(f"{_full}: {_ex}")
        for _prior in (fcfg.get("priors") or []):
            _parts = str(_prior.get("param", "")).split("___")
            if len(_parts) == 3:
                try:
                    fitter.setParamPrior(
                        _parts[0], _parts[1], _parts[2],
                        _prior["value"], _prior["uncertainty"])
                except Exception:
                    pass

        # Apply perturbed values
        for vk, val in zip(vary_keys, perturbed_values):
            if len(vk) == 3:
                midx, pname, src_idx = vk
                _set_param(model_maps[src_idx],
                           model_configs[midx]["name"], pname, val)
            else:
                midx, pname = vk
                # Shared: set on first source (sharing propagates), or
                # single-source mode
                _set_param(model_maps[0],
                           model_configs[midx]["name"], pname, val)

        # Fit honoring the user's objective: when Poisson/Gaussian LLH is
        # selected, the auto-fitter must minimize (and rank by) the SAME
        # likelihood the real fit will, not plain chi2 -- for low-count bins
        # the two minima differ (code review 2026-06-02, autofitter-ignores-
        # objective). emcee is a sampler, not a point objective, so the quick
        # multi-start falls back to leastsq. fitter_config None reduces to the
        # previous leastsq / scale_covar=False behaviour.
        fit_kwargs = {"method": fcfg.get("method", "leastsq"),
                      "scale_covar": fcfg.get("scale_covar", False)}
        if fit_kwargs["method"] == "emcee":
            fit_kwargs["method"] = "leastsq"
        if fcfg.get("llh"):
            fit_kwargs["llh"] = True
            fit_kwargs["llh_method"] = fcfg.get("llh_method")
            fit_kwargs["scale_covar"] = False
        fitter.fit(**fit_kwargs)

        chi2 = float(fitter.chisqr) if hasattr(fitter, 'chisqr') else 1e30
        redchi = float(fitter.redchi) if hasattr(fitter, 'redchi') else 1e30

        # Extract fitted values matching vary_keys order
        fitted_vals = []
        for vk in vary_keys:
            if len(vk) == 3:
                midx, pname, src_idx = vk
                fitted_vals.append(
                    _get_param(model_maps[src_idx],
                               model_configs[midx]["name"], pname))
            else:
                midx, pname = vk
                fitted_vals.append(
                    _get_param(model_maps[0],
                               model_configs[midx]["name"], pname))

        # Per-source smooth curves for plotting
        per_source = []
        for si, (src, mmap) in enumerate(zip(sources, model_maps)):
            xd = _np.asarray(sources_data[si]["x"])
            xs = _np.linspace(xd.min(), xd.max(), 500)
            ys = src.evaluate(xs)
            comps = {}
            for mname, model in mmap.items():
                if mname.endswith("_bkg"):
                    continue
                try:
                    comps[mname] = model.f(xs).tolist()
                except Exception:
                    pass
            per_source.append({
                "x_smooth": xs.tolist(),
                "y_smooth": ys.tolist(),
                "components": comps,
            })

        # Backward-compatible top-level keys (from first source)
        return {
            "success": True,
            "chi2": chi2,
            "redchi": redchi,
            "fitted_vals": fitted_vals,
            "x_smooth": per_source[0]["x_smooth"],
            "y_smooth": per_source[0]["y_smooth"],
            "components": per_source[0]["components"],
            "per_source": per_source,
            "expr_errors": expr_errors,
        }
    except Exception as e:
        return {"success": False, "chi2": 1e30, "redchi": 1e30,
                "error": str(e), "fitted_vals": list(perturbed_values),
                "x_smooth": [], "y_smooth": [], "components": {},
                "per_source": []}


# ══════════════════════════════════════════════════════════════════
#  Worker Thread
# ══════════════════════════════════════════════════════════════════

class _MultiStartWorker(QThread):
    """Run N perturbed leastsq fits in parallel."""

    # fit_index, completed_count, total_count, result_dict
    result_ready = Signal(int, int, int, object)
    all_done = Signal()
    error = Signal(str)

    def __init__(self, sources_data, model_configs, vary_keys,
                 center_values, spread_frac, n_starts, n_cores,
                 shared_params=None, shared_all_params=None,
                 fitter_config=None, parent=None):
        super().__init__(parent)
        self.sources_data = sources_data
        self.model_configs = model_configs
        self.vary_keys = vary_keys
        self.center = np.array(center_values)
        self.spread_frac = spread_frac
        self.n_starts = n_starts
        self.n_cores = n_cores
        self.shared_params = shared_params or []
        self.shared_all_params = shared_all_params or []
        self.fitter_config = fitter_config or {}
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            rng = np.random.default_rng()
            n = len(self.center)

            # Generate perturbed starting points
            perturbations = []
            perturbations.append(self.center.copy())  # always include original
            for _ in range(self.n_starts - 1):
                p = self.center.copy()
                for j in range(n):
                    spread = abs(self.center[j]) * self.spread_frac + 1.0
                    p[j] = rng.normal(self.center[j], spread)
                perturbations.append(p)

            # Run fits in parallel
            completed = 0

            with ProcessPoolExecutor(max_workers=self.n_cores) as pool:
                futures = {}
                for i, pvals in enumerate(perturbations):
                    if self._cancelled:
                        break
                    f = pool.submit(
                        _run_one_fit,
                        self.sources_data, self.model_configs,
                        pvals, self.vary_keys, self.shared_params,
                        self.shared_all_params, self.fitter_config)
                    futures[f] = i

                for future in as_completed(futures):
                    if self._cancelled:
                        break
                    fit_idx = futures[future]
                    completed += 1
                    try:
                        result = future.result(timeout=60)
                    except Exception as e:
                        result = {"success": False, "chi2": 1e30,
                                  "redchi": 1e30, "error": str(e),
                                  "fitted_vals": [], "x_smooth": [],
                                  "y_smooth": [], "components": {}}

                    self.result_ready.emit(
                        fit_idx, completed, self.n_starts, result)

            self.all_done.emit()

        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════
#  Dialog
# ══════════════════════════════════════════════════════════════════

class AutoFitterDialog(QDialog):
    """Auto-Fitter v2: multi-start local optimizer with live visualization."""

    parameters_accepted = Signal(object)

    def __init__(self, sources_data, model_configs, fitter_config=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Fitter")
        self.resize(1100, 850)
        self.setMinimumSize(800, 600)
        self.setModal(False)

        try:
            from gui.shared_widgets import lucide_icon
            self.setWindowIcon(lucide_icon("crosshair"))
        except Exception:
            pass

        # Store source data (list of {"name","x","y","yerr"} dicts)
        self._sources_data = sources_data
        self._multi = len(sources_data) > 1
        # Backward-compat aliases for single-source plotting
        self._x = np.asarray(sources_data[0]["x"], dtype=float)
        self._y = np.asarray(sources_data[0]["y"], dtype=float)
        self._yerr = np.asarray(sources_data[0]["yerr"], dtype=float)
        self._model_configs = model_configs
        self._fitter_config = fitter_config or {}
        self._worker = None
        self._results = []        # successful results, sorted by chi2
        self._failed_results = []
        self._selected_idx = 0
        self._round = 0
        self._n_completed = 0
        self._auto_plot = True    # auto-update plot with best result
        self._user_pinned = None  # result object the user clicked on

        # Shared params from the fitter-block linking config. Keep 'Model'
        # shares (shareModelParams) and 'All' shares (shareParams) SEPARATE so
        # the worker routes each correctly -- they have different scope and
        # priority in satlas2; the old union mis-routed 'All' shares through
        # shareModelParams (code review 2026-06-02).
        self._shared_params = list(
            self._fitter_config.get("shared_params", []))
        self._shared_all_params = list(
            self._fitter_config.get("shared_all_params", []))
        shared_set = ((set(self._shared_params) | set(self._shared_all_params))
                      if self._multi else set())
        n_sources = len(sources_data)

        # Build vary keys and initial values
        self._vary_keys = []   # (midx, pname) or (midx, pname, src_idx)
        self._init_values = []
        from gui.analysis.naming import NON_FIT_PARAMS as _SKIP
        for i, mc in enumerate(model_configs):
            for pname, pdata in mc["params"].items():
                if pname in _SKIP:
                    continue
                if not pdata.get("vary", False):
                    continue
                if (pdata.get("expr") or "").strip():
                    continue  # expression-tied params are derived, not fit
                if self._multi and pname not in shared_set:
                    for si in range(n_sources):
                        self._vary_keys.append((i, pname, si))
                        self._init_values.append(pdata.get("value", 0))
                else:
                    self._vary_keys.append((i, pname))
                    self._init_values.append(pdata.get("value", 0))
            # Peak amplitude parameters (HFS with racah off)
            if mc.get("type") == "HFS" and not mc.get("racah", True):
                for line, amp_data in mc.get("peak_amplitudes", {}).items():
                    if amp_data.get("vary", False):
                        pname = f"Amp{line}"
                        if self._multi and pname not in shared_set:
                            for si in range(n_sources):
                                self._vary_keys.append((i, pname, si))
                                self._init_values.append(
                                    amp_data.get("value", 0))
                        else:
                            self._vary_keys.append((i, pname))
                            self._init_values.append(
                                amp_data.get("value", 0))
        self._center = np.array(self._init_values, dtype=float)
        self._ndata = sum(len(sd["x"]) for sd in sources_data)

        root = QVBoxLayout(self)
        root.setSpacing(4)

        # ── Controls ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        ctrl.addWidget(QLabel("Starts:"))
        self._n_starts = QSpinBox()
        self._n_starts.setRange(5, 500)
        self._n_starts.setValue(30)
        self._n_starts.setToolTip(
            "Number of random perturbations to try.\n"
            "Each runs a fast leastsq fit. More = better coverage.")
        ctrl.addWidget(self._n_starts)

        ctrl.addWidget(QLabel("Spread:"))
        self._spread = QDoubleSpinBox()
        self._spread.setRange(0.01, 2.0)
        self._spread.setDecimals(2)
        self._spread.setValue(0.30)
        self._spread.setSingleStep(0.05)
        self._spread.setToolTip(
            "Perturbation spread as fraction of parameter value.\n"
            "0.30 = each parameter varied by +/-30% of its value.\n"
            "Larger = wider search, smaller = fine refinement.")
        ctrl.addWidget(self._spread)

        ctrl.addWidget(QLabel("Cores:"))
        self._n_cores = QSpinBox()
        self._n_cores.setRange(1, os.cpu_count() or 4)
        self._n_cores.setValue(max(1, (os.cpu_count() or 4) - 2))
        self._n_cores.setToolTip("CPU cores for parallel fits")
        ctrl.addWidget(self._n_cores)

        ctrl.addStretch()

        self._start_btn = QPushButton("\u25b6 Start")
        self._start_btn.setStyleSheet("font-weight: bold;")
        self._start_btn.clicked.connect(self._start_fresh)
        ctrl.addWidget(self._start_btn)
        self._refine_btn = QPushButton("\u25b6 Refine")
        self._refine_btn.setToolTip(
            "Generate new perturbations around the selected best\n"
            "result with tighter spread. Narrows the search.")
        self._refine_btn.setEnabled(False)
        self._refine_btn.clicked.connect(self._refine)
        ctrl.addWidget(self._refine_btn)
        self._stop_btn = QPushButton("\u25a0 Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self._stop_btn)
        root.addLayout(ctrl)

        # ── Progress + Core activity toggle ──
        prog_row = QHBoxLayout()
        self._cores_toggle = QPushButton("Show Core Activity")
        self._cores_toggle.setCheckable(True)
        self._cores_toggle.setChecked(False)
        # Minimum row height so the button label and the progress bar
        # render fully without clipping the text.
        self._cores_toggle.setMinimumHeight(28)
        self._cores_toggle.toggled.connect(self._toggle_cores)
        prog_row.addWidget(self._cores_toggle)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMinimumHeight(28)
        prog_row.addWidget(self._progress, 1)
        self._status = QLabel("Ready")
        prog_row.addWidget(self._status)
        root.addLayout(prog_row)

        self._cores_panel = QWidget()
        self._cores_grid = QGridLayout(self._cores_panel)
        self._cores_grid.setContentsMargins(4, 2, 4, 2)
        self._cores_grid.setSpacing(4)
        self._cores_panel.setVisible(False)
        self._core_bars = []
        root.addWidget(self._cores_panel)

        # ── Plot visibility checkboxes ──
        vis_row = QHBoxLayout()
        vis_row.setSpacing(12)
        vis_row.addWidget(QLabel("Show:"))
        self._show_data = QCheckBox("Data")
        self._show_data.setChecked(True)
        self._show_data.toggled.connect(self._redraw_current)
        vis_row.addWidget(self._show_data)
        self._show_fit = QCheckBox("Total fit")
        self._show_fit.setChecked(True)
        self._show_fit.toggled.connect(self._redraw_current)
        vis_row.addWidget(self._show_fit)
        self._show_components = QCheckBox("Model components")
        self._show_components.setChecked(True)
        self._show_components.toggled.connect(self._redraw_current)
        vis_row.addWidget(self._show_components)
        vis_row.addStretch()
        root.addLayout(vis_row)

        # ── Plot ──
        # Wider top/bottom margins so the title and xlabel don't clip
        # when the Core Activity panel is expanded and squeezes the
        # canvas vertically. Minimum canvas height keeps the plot
        # readable in that compressed state.
        self._fig = Figure(dpi=100)
        self._fig.subplots_adjust(
            left=0.06, right=0.99, top=0.92, bottom=0.16)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumHeight(360)
        root.addWidget(self._canvas, 1)

        # ── Results table ──
        res_grp = QGroupBox("Results (ranked by \u03c7\u00b2)")
        res_lay = QVBoxLayout(res_grp)
        res_lay.setContentsMargins(4, 8, 4, 4)

        self._result_label = QLabel("")
        res_lay.addWidget(self._result_label)

        self._res_table = QTableWidget()
        self._res_table.setColumnCount(4)
        self._res_table.setHorizontalHeaderLabels(
            ["Rank", "\u03c7\u00b2", "\u03c7\u00b2_red", "Parameters"])
        self._res_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch)
        self._res_table.verticalHeader().setVisible(False)
        self._res_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._res_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._res_table.setMaximumHeight(180)
        self._res_table.currentCellChanged.connect(self._on_result_selected)
        self._res_table.doubleClicked.connect(self._show_param_detail)
        self._res_table.setToolTip(
            "Click to preview, double-click to see full parameters")
        res_lay.addWidget(self._res_table)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._accept_btn = QPushButton("Accept Selected")
        self._accept_btn.setEnabled(False)
        self._accept_btn.setStyleSheet("font-weight: bold;")
        self._accept_btn.setToolTip(
            "Push the selected result's parameters to the Model blocks")
        self._accept_btn.clicked.connect(self._accept)
        btn_row.addWidget(self._accept_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        res_lay.addLayout(btn_row)
        root.addWidget(res_grp)

        # Initial plot
        self._draw_data()

    # ── Core activity panel ──

    def _toggle_cores(self, checked):
        self._cores_panel.setVisible(checked)
        self._cores_toggle.setText(
            "Hide Core Activity" if checked else "Show Core Activity")

    def _build_core_bars(self, n_cores, n_fits):
        """Create per-core progress bars for the current round."""
        # Clear old widgets
        while self._cores_grid.count():
            item = self._cores_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._core_bars = []
        self._core_labels = []

        # Calculate fits assigned to each core slot
        fits_per_core = [0] * n_cores
        for i in range(n_fits):
            fits_per_core[i % n_cores] += 1

        # Grid: each core = label + bar, arranged in N columns
        cols = min(n_cores, 4)
        label_width = 55 if n_cores < 10 else 65
        for k in range(n_cores):
            grid_row = k // cols
            grid_col = k % cols
            # Use a sub-column pair: 2*col for label, 2*col+1 for bar
            lbl = QLabel(f"Core {k + 1}:")
            lbl.setFixedWidth(label_width)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                             | Qt.AlignmentFlag.AlignVCenter)
            bar = QProgressBar()
            bar.setRange(0, fits_per_core[k])
            bar.setValue(0)
            bar.setFormat("%v/%m")
            bar.setFixedHeight(16)
            self._cores_grid.addWidget(lbl, grid_row, grid_col * 2)
            self._cores_grid.addWidget(bar, grid_row, grid_col * 2 + 1)
            self._core_bars.append(bar)
            self._core_labels.append(lbl)

        # Make bar columns stretch equally, labels stay fixed
        for c in range(cols):
            self._cores_grid.setColumnStretch(c * 2, 0)
            self._cores_grid.setColumnStretch(c * 2 + 1, 1)

    # ── Plotting ──

    def _source_colors(self):
        """Return a color list for multi-source plots."""
        import matplotlib.pyplot as plt
        return plt.rcParams['axes.prop_cycle'].by_key()['color']

    def _draw_data(self):
        self._ax.clear()
        if self._multi:
            colors = self._source_colors()
            for si, sd in enumerate(self._sources_data):
                cc = colors[si % len(colors)]
                self._ax.errorbar(
                    sd["x"], sd["y"], yerr=sd["yerr"],
                    fmt='.', ms=3, alpha=0.5, capsize=1,
                    color=cc, label=sd["name"], zorder=1)
        else:
            self._ax.errorbar(
                self._x, self._y, yerr=self._yerr,
                fmt='.', ms=3, alpha=0.5, capsize=1,
                color='black', label='Data', zorder=1)
        self._ax.set_xlabel("Frequency (MHz)")
        self._ax.set_ylabel("Counts")
        self._ax.legend(fontsize=8)
        self._ax.grid(True, alpha=0.2)
        self._hug_x_to_data()
        self._canvas.draw_idle()

    def _hug_x_to_data(self):
        """Pin the x-axis to the DATA extent.

        The fit curve (x_smooth) is evaluated on a wider grid than the
        data, so matplotlib's autoscale otherwise leaves empty margin on
        both sides. Use the union of all source x-ranges.
        """
        xs = [np.asarray(sd["x"], dtype=float)
              for sd in self._sources_data
              if len(sd.get("x", [])) > 0]
        if xs:
            lo = min(float(a.min()) for a in xs)
            hi = max(float(a.max()) for a in xs)
            if hi > lo:
                self._ax.set_xlim(lo, hi)

    def _redraw_current(self):
        """Redraw with current visibility settings."""
        if self._results and self._selected_idx < len(self._results):
            self._draw_result(self._results[self._selected_idx])
        else:
            self._draw_data()

    def _draw_result(self, result):
        """Draw data + a specific fit result."""
        self._ax.clear()
        per_source = result.get("per_source", [])

        if self._multi and per_source:
            colors = self._source_colors()
            for si, sd in enumerate(self._sources_data):
                cc = colors[si % len(colors)]
                if self._show_data.isChecked():
                    self._ax.errorbar(
                        sd["x"], sd["y"], yerr=sd["yerr"],
                        fmt='.', ms=3, alpha=0.5, capsize=1,
                        color=cc, label=sd["name"], zorder=1)
                if si < len(per_source) and self._show_fit.isChecked():
                    ps = per_source[si]
                    self._ax.plot(
                        np.array(ps["x_smooth"]),
                        np.array(ps["y_smooth"]),
                        color=cc, lw=1.5, ls='-', alpha=0.9,
                        label=f'Fit ({sd["name"]})', zorder=3)
        else:
            # Single source path
            if self._show_data.isChecked():
                self._ax.errorbar(
                    self._x, self._y, yerr=self._yerr,
                    fmt='.', ms=3, alpha=0.5, capsize=1,
                    color='black', label='Data', zorder=1)

            if result.get("x_smooth") and result.get("y_smooth"):
                x_s = np.array(result["x_smooth"])
                y_s = np.array(result["y_smooth"])

                if self._show_fit.isChecked():
                    self._ax.plot(x_s, y_s, 'r-', lw=1.5,
                                 label='Fit', zorder=3)

                if self._show_components.isChecked():
                    import matplotlib.pyplot as plt
                    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    for ci, (cname, cdata) in enumerate(
                            result.get("components", {}).items()):
                        cc = colors[(ci + 1) % len(colors)]
                        self._ax.plot(x_s, np.array(cdata), color=cc,
                                     lw=1, ls='--', alpha=0.7,
                                     label=cname, zorder=2)

        chi2 = result.get("chi2", 0)
        redchi = result.get("redchi", 0)
        src_label = f"  ({len(self._sources_data)} sources)" if self._multi else ""
        self._ax.set_title(
            f"Round {self._round}  |  "
            f"\u03c7\u00b2 = {chi2:.1f}  |  "
            f"\u03c7\u00b2_red = {redchi:.4f}{src_label}",
            fontsize=10, loc='left')
        self._ax.set_xlabel("Frequency (MHz)")
        self._ax.set_ylabel("Counts")
        self._ax.legend(fontsize=7)
        self._ax.grid(True, alpha=0.2)
        self._hug_x_to_data()
        self._canvas.draw_idle()

    # ── Start / Refine / Stop ──

    def _start_fresh(self):
        """Start from the original initial guesses."""
        self._round = 0
        self._center = np.array(self._init_values, dtype=float)
        self._run_round()

    def _refine(self):
        """Start a new round centered on the selected best result."""
        if not self._results:
            return
        idx = self._selected_idx
        if idx < len(self._results):
            best = self._results[idx]
            if best.get("fitted_vals"):
                self._center = np.array(best["fitted_vals"], dtype=float)
                # Halve the spread for refinement
                self._spread.setValue(
                    max(0.02, self._spread.value() * 0.5))
        self._run_round()

    def _run_round(self):
        if not self._vary_keys:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Auto-Fitter",
                "No parameters to vary. Check 'Vary' in Model blocks.")
            return

        self._round += 1
        self._results = []
        self._failed_results = []
        self._n_completed = 0
        self._selected_idx = 0
        self._auto_plot = True
        self._user_pinned = None

        n_cores = self._n_cores.value()
        n_fits = self._n_starts.value()
        self._build_core_bars(n_cores, n_fits)

        # Clear table
        self._res_table.setRowCount(0)
        self._result_label.setText("")

        self._start_btn.setEnabled(False)
        self._refine_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._accept_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText(f"Round {self._round}: running...")

        self._worker = _MultiStartWorker(
            sources_data=self._sources_data,
            model_configs=self._model_configs,
            vary_keys=self._vary_keys,
            center_values=self._center.tolist(),
            spread_frac=self._spread.value(),
            n_starts=n_fits,
            n_cores=n_cores,
            shared_params=self._shared_params if self._multi else None,
            shared_all_params=(self._shared_all_params
                               if self._multi else None),
            fitter_config=self._fitter_config,
        )
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        if self._worker:
            self._worker.cancel()
        self._stop_btn.setEnabled(False)
        self._status.setText("Stopping...")

    # ── Live result callbacks ──

    def _on_result_ready(self, fit_idx, completed, total, result):
        """Handle a single completed fit — update core bars, table, plot."""
        self._n_completed = completed

        # Overall progress
        pct = int(completed / max(total, 1) * 100)
        self._progress.setValue(pct)

        # Core slot bar
        if self._core_bars:
            slot = fit_idx % len(self._core_bars)
            bar = self._core_bars[slot]
            bar.setValue(bar.value() + 1)
            # Turn green when this core finishes all its assigned fits
            if bar.value() >= bar.maximum():
                bar.setStyleSheet(
                    "QProgressBar::chunk { background-color: #4CAF50; }")
                self._core_labels[slot].setStyleSheet("color: #4CAF50;")

        # Insert into sorted results (successful only)
        if result.get("success"):
            chi2 = result.get("chi2", 1e30)
            keys = [r["chi2"] for r in self._results]
            idx = bisect.bisect_left(keys, chi2)
            self._results.insert(idx, result)
        else:
            self._failed_results.append(result)

        # Status text
        n_success = len(self._results)
        best = self._results[0] if self._results else {}
        redchi = best.get("redchi", 0)
        self._status.setText(
            f"Round {self._round}: {completed}/{total}  |  "
            f"{n_success} converged  |  "
            f"best \u03c7\u00b2_red = {redchi:.4f}")

        # Live results table
        self._update_results_table()

        # Enable accept/refine as soon as we have a result
        if self._results:
            self._accept_btn.setEnabled(True)
            self._refine_btn.setEnabled(True)

        # Update plot periodically with best (unless user pinned a result)
        if completed <= 3 or completed % 5 == 0 or completed == total:
            if not self._auto_plot and self._user_pinned:
                self._draw_result(self._user_pinned)
            elif self._results:
                self._draw_result(self._results[0])

    def _update_results_table(self):
        """Rebuild the top-10 results table from the sorted list."""
        show = self._results[:10]

        self._res_table.blockSignals(True)
        self._res_table.setRowCount(len(show))
        for row, r in enumerate(show):
            self._res_table.setItem(
                row, 0, QTableWidgetItem(f"#{row + 1}"))
            self._res_table.setItem(
                row, 1, QTableWidgetItem(f"{r.get('chi2', 0):.2f}"))
            self._res_table.setItem(
                row, 2, QTableWidgetItem(f"{r.get('redchi', 0):.4f}"))
            # Compact parameter summary
            vals = r.get("fitted_vals", [])
            summary_parts = []
            for vk, v in zip(self._vary_keys, vals):
                midx, pname = vk[0], vk[1]
                mc = self._model_configs[midx]
                suffix = f"[{vk[2]}]" if len(vk) == 3 else ""
                summary_parts.append(
                    f"{mc['name']}.{pname}{suffix}={v:.2f}")
            self._res_table.setItem(
                row, 3, QTableWidgetItem("  ".join(summary_parts)))
        self._res_table.blockSignals(False)

        # Maintain selection: follow the best (row 0) so the user watches
        # the auto-fitter converge, UNLESS they pinned a specific result by
        # clicking it. The selectRow is signal-blocked so this programmatic
        # move doesn't fire _on_result_selected (which would falsely pin
        # the current best and freeze the selection on it).
        target_row = 0
        if self._user_pinned is not None:
            for i, r in enumerate(show):
                if r is self._user_pinned:
                    target_row = i
                    break
            else:
                # Pinned result fell out of top 10 — release pin
                self._user_pinned = None
        self._selected_idx = target_row
        self._res_table.blockSignals(True)
        self._res_table.selectRow(target_row)
        self._res_table.blockSignals(False)

    def _on_all_done(self):
        """Final cleanup after all fits complete."""
        # If no successful results, show failed ones for debugging
        if not self._results and self._failed_results:
            self._results = self._failed_results[:5]
            self._update_results_table()

        total = self._n_completed
        n_success = len([r for r in self._results if r.get("success", True)])
        best = self._results[0] if self._results else {}
        chi2 = best.get("chi2", 0)
        redchi = best.get("redchi", 0)
        nvarys = len(self._vary_keys)

        self._progress.setValue(100)
        self._status.setText(
            f"Round {self._round} done  |  "
            f"{n_success}/{total} converged  |  "
            f"best \u03c7\u00b2_red = {redchi:.4f}")
        self._result_label.setText(
            f"<b>Round {self._round}:</b>  "
            f"{n_success} of {total} fits converged.  "
            f"Best \u03c7\u00b2 = {chi2:.2f},  "
            f"\u03c7\u00b2_red = {redchi:.4f}  "
            f"({nvarys} params, {self._ndata} data points)")

        # Final table and plot
        self._auto_plot = True
        self._user_pinned = None
        self._update_results_table()
        if self._results:
            self._selected_idx = 0
            self._res_table.blockSignals(True)
            self._res_table.selectRow(0)
            self._res_table.blockSignals(False)
            self._draw_result(self._results[0])

        self._start_btn.setEnabled(True)
        self._refine_btn.setEnabled(bool(self._results))
        self._stop_btn.setEnabled(False)
        self._accept_btn.setEnabled(bool(self._results))
        self._worker = None

    def _on_error(self, msg):
        # code review 2026-06-02, worker-thread-exceptions-not-logged
        _log.error("Auto-fitter failed: %s", msg)
        self._status.setText(f"Error: {msg}")
        self._start_btn.setEnabled(True)
        self._refine_btn.setEnabled(bool(self._results))
        self._stop_btn.setEnabled(False)
        self._worker = None

    def _on_result_selected(self, row, col, prev_row, prev_col):
        if 0 <= row < len(self._results):
            self._selected_idx = row
            self._auto_plot = False
            self._user_pinned = self._results[row]
            self._draw_result(self._results[row])

    def _show_param_detail(self, index):
        """Show a popup dialog with full parameter details for the clicked row."""
        row = index.row()
        if row < 0 or row >= len(self._results):
            return
        result = self._results[row]
        vals = result.get("fitted_vals", [])
        chi2 = result.get("chi2", 0)
        redchi = result.get("redchi", 0)

        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, \
            QTableWidgetItem, QHeaderView, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Result #{row + 1}  |  "
                           f"\u03c7\u00b2_red = {redchi:.4f}")
        dlg.resize(500, 400)
        lay = QVBoxLayout(dlg)

        tbl = QTableWidget()
        tbl.setColumnCount(3)
        tbl.setHorizontalHeaderLabels(["Model", "Parameter", "Value"])
        tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setRowCount(len(self._vary_keys))

        for r, (vk, v) in enumerate(zip(self._vary_keys, vals)):
            midx, pname = vk[0], vk[1]
            mc = self._model_configs[midx]
            suffix = f" [src {vk[2]}]" if len(vk) == 3 else ""
            tbl.setItem(r, 0, QTableWidgetItem(mc["name"] + suffix))
            tbl.setItem(r, 1, QTableWidgetItem(pname))
            tbl.setItem(r, 2, QTableWidgetItem(f"{v:.6f}"))

        lay.addWidget(tbl)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

    # ── Accept ──

    def _accept(self):
        if not self._results or self._selected_idx >= len(self._results):
            print("[AutoFit] _accept: no results or invalid index", flush=True)
            return
        best = self._results[self._selected_idx]
        vals = best.get("fitted_vals", [])
        result = {}
        for vk, v in zip(self._vary_keys, vals):
            result[tuple(vk)] = float(v)
        print(f"[AutoFit] Emitting {len(result)} params from "
              f"result #{self._selected_idx + 1}", flush=True)
        self.parameters_accepted.emit(result)
        self._status.setText(
            f"Accepted result #{self._selected_idx + 1}!")
