# DENIS

**D**oppler **E**stimation and **N**umerical **I**nference for **S**pectroscopy

A comprehensive tool for **Collinear Laser Spectroscopy (CLS)**: run-time estimation, pre-analysis, SATLAS2-based fitting, and results visualization.

**Version:** 1.0.0
**Developer:** [Arda Kayaalp](https://ardakayaalp.com/) (arda.kayaalp@kuleuven.be)
**Framework:** PySide6 (Qt6) + matplotlib + satlas2 + clstools

---

## Quick Start

The recommended install path uses [`uv`](https://docs.astral.sh/uv/) — it bootstraps Python itself, the project venv, and every dependency in one step. No existing Python install is required.

```bash
# Get the code
git clone https://github.com/ardakayaalp/DENIS.git
cd DENIS

# First-time install: bootstraps uv (if missing), runs `uv sync`,
# and creates a desktop shortcut. Run with no args for an interactive
# menu, or pass a subcommand to skip the prompt.
install.bat            # Windows  -- menu
install.bat full       # Windows  -- uv + dependencies + shortcut
install.bat uv         # Windows  -- install uv only
install.bat shortcut   # Windows  -- create desktop shortcut only

./install.sh           # Linux / macOS  -- menu
./install.sh full      # Linux / macOS  -- full install
./install.sh uv        # Linux / macOS  -- uv only
./install.sh shortcut  # Linux / macOS  -- shortcut only

# Launch from the desktop shortcut, or from a terminal:
uv run python gui.py

# CLI estimation
uv run python cls_estimate.py config.yaml --output-dir ./output

# Load a save file on startup
uv run python gui.py my_config.yaml
```

If `uv` is already installed and the wrapper scripts are not needed:

```bash
uv sync
uv run python gui.py
```

---

## Features

### Estimate Tab
- Run-time estimation for CLS experiments
- HFS spectrum simulation in voltage space
- Isotope shift and resonance voltage calculations
- Multi-isotope comparison with configurable parameters
- **Single-surface layout** (2026-07): three resizable columns — global parameters, isotope list (position-coded color stripes), and Run Options + plots + log; the ⛶ button expands the plots over the whole tab
- **Run persistence**: every run writes `estimate_results.npz` + `peaks.csv` into its `cls_<timestamp>` folder; the save file records the run so plots/peak table/log restore on load without re-running
- **Load Estimation...**: reopen any previous run folder for viewing — older, PDF-only runs are rendered page-by-page onto the live matplotlib canvas so pan/zoom/save still work
- Half-integer J quantum numbers in the log (e.g. `J=5/2->7/2`)
- **Peak List** view: sortable table of all peaks (isotope, state, shift, voltage, intensity)
- Both **intensity-ordered and voltage-ordered** peak tables emitted to the log automatically

### Pre-Analysis Tab
- Load and visualize ASDF data files from CLS experiments
- TOF histogram with interactive gate selection and an adjustable **TOF bin size** (default 1 µs, down to 1 ns)
- Spectrum display in voltage, frequency, or wavenumber, binned **per scan step** by default (no uniform-grid aliasing); a header **Bin: N × step** spin groups adjacent steps
- **Layout switcher** ("3 row stacked" / "2 row stacked"), **X/Y grid toggles**, and a **dark-mode plot toggle** (black canvas, neon line colors) — all session-global and persisted
- Default run colors from the classic **Windows 98 palette**; every file and model carries **separate light + dark colors** with a dual-mode picker (preset palettes, hex/RGB entry, saved custom slots)
- Timestamp overview for run diagnostics, with per-scan overlay and *Exclude scans in view*
- Calibration visualization (readback vs set, step uniformity)
- **Voltage-calibration diagnostics & per-run overrides** (right-click a run -> *Calibration...*) -- see below
- Cooler-voltage stability diagnostics: robust σ (1.4826·MAD), spike detection, ripple panes, a sortable per-run summary table, and a **≈ Δν (MHz)** secondary axis
- HFS model overlay with adjustable parameters and **peak label annotations**
- Multi-file overlay with per-run color coding, **opacity**, and **line style** controls
- **Spectrum merging**: merge checked files in voltage or Doppler-shifted frequency domain
- E lower / E upper energy level inputs with auto-computed transition and fundamental
- **Mass display + override**: shows the looked-up mass in amu next to Z/A; tick **Override** to supply a custom amu (e.g. an AME2020 value not in the periodictable database)

### Analysis Tab
- Block-based fitting pipeline: Source -> Model -> Fitter -> Output
- **Models:** HFS (with sidepeaks, Racah amplitudes), Voigt, Skewed Voigt, Exponential Decay, Piecewise Constant, Polynomial
- **Fit methods:** leastsq, least_squares, slsqp, emcee (MCMC), nelder, powell, cobyla
- **Statistics:** Chi-square, Gaussian LLH, Poisson LLH
- Separate (per-file) and simultaneous (multi-file) fitting modes
- Parameter sharing (shareModelParams / shareParams) and expression constraints
- MCMC confidence bands, walk plots, correlation plots, chi-square maps
- Run merging: combine multiple ASDF files into a single spectrum
- **Mass override**: source block has a "Mass [amu]" spinbox auto-filled from the IUPAC/AME table; tick **Override** to keep a custom value across Z/A edits
- **F gate in MHz**: source-block frequency gate is entered in MHz for convenience (converted to Hz at the clstools boundary)
- **Isotope Shift tab**: compute shifts with propagated statistical and systematic errors. Per-isotope σ broken out into `σ_fit` / `σ_scatter` (Birge inflation) / `σ_correction` (from the GP, see below) / `σ_voltage`. When two isotopes share the same GP corrector, their corrected centroids are correlated through the GP and the cross-covariance is subtracted from the shift variance — shifts shrink for isotopes measured close in time and approach quadrature for well-separated measurements
- **Reference Centroid Correction (Gaussian Process)**: time-dependent drift correction following van den Borne (2025). Fit reference-isotope scans in a dedicated **Reference Project**, train a GP (RBF / Matérn(5/2) / 3-component composite kernel via PyMC v5) on the resulting `(t, centroid, σ)` table, and apply per-run μ_GP shifts to the binned MHz axis post-Doppler. UI in the Isotope Shifts tab: kernel selector, diagnostic plot reproducing the thesis Fig. B.2 layout (errorbars + MAP + 1σ/2σ bands), per-file table with Auto / Manual / Off mode, |μ|/σ outlier coloring, sortable columns. Merge dialog and AutoFitter both consume the same correction so initial guesses and merged spectra land in the corrected frame
- Data preview with navigable file browser
- Fit quality indicators (FWHM, peak positions, reduced chi-square)
- Weighted averages and parameter comparison across iterations
- **Update iteration**: regenerate the last fit's output artifacts into its existing `iter_NNN` folder with the current toggles, without re-fitting — newly-ticked walk/correlation plots are rebuilt from the saved MCMC chains; only the chi-square map and confidence bands need a true re-run
- **Fit values on plot**: an on-plot values box ("Name = value ± err") with a per-parameter picker tree; shade-under-fit and per-plot-type styling (including Isotope Shifts) live in the Output block's **Plot Options…** dialog

#### Explicit binning + per-file overrides

The Source block exposes the binning policy as first-class controls instead of relying on clstools defaults:
- **Bin definition**: `Per scan step` (the default: one bin per native DAC voltage step, Doppler-projected in frequency mode — immune to the uniform-grid aliasing that produces doubled-count spikes), `Auto` (clstools' uniform grid at roughly one bin per step — can alias when the grid beats against the step spacing), `Fixed bin count`, or `Fixed bin width [MHz]` (frequency mode only; clstools accepts integer bin counts so the realised width is approximate).
- **Bin multiple**: group N adjacent native steps into one bin (count-weighted centers, summed counts) — available for `Per scan step` and Raw Voltage binning; Pre-Analysis exposes the same engine as its `Bin: N × step` header spin.
- **Bin count / Bin width** spinboxes activate based on the selected definition.
- All binning logic — bin definition, x_column, yerr mode (incl. low-count Poisson interval correction), xerr mode, fallback handling — lives in `gui/analysis/binning.py::compute_binned(data, cfg)`. Every fit / preview / merge / autofit path goes through this single helper.

The previous silent `max(100, sqrt(N))` fallback (used when `Compute_Bins` raised) is now reported on the per-run result as `binning_info["fallback_used"]` and logged by the worker.

**Per-file overrides** (right-click any run row in the Source block):
- *Edit Binning Override...* opens a dialog with one row per overridable key. Each row has a `Use Source (current_value)` checkbox — checked = inherit from the Source block, unchecked = freeze a different value for this run only.
- *Freeze Current Source Binning Here* snapshots the current Source values into an explicit override (no longer follows future Source changes).
- *Copy This Override to Checked Runs* propagates the sparse override to other checked rows.
- *Show Effective Binning* shows the merged (Source + override) config for that run.
- An orange `[override]` badge marks rows that have an active override; YAML stores overrides sparsely under `files[].binning_override`.
- Only binning keys (`bin_mode, x_column, yerr_mode, xerr_mode, bin_definition, bin_count, bin_width_mhz`) are overridable per file; physics, gates, and laser/cooler overrides remain Source-level.

**Pre-fit and post-fit summaries:**
- The Source block's **Binning** button opens a three-tab view of the actual binning result for every checked file:
  - *Summary* tab: per-run table with fit-relevant occupancy columns — **Empty a/b (%)** with a scan-gap-aware denominator (contiguous zero runs ≥ max(30, 5%·n) count as scan gaps, not empties; highlighted when in-scan empties exceed 20%), **Median counts**, and **Total counts** — plus mode/definition/bin-count columns. Double-click a row to jump to its plot.
  - *Run detail* tab: the spectrum with vertical bin edges (optional alternating shading ≤ 300 bins, optional **raw event rug** from up to 20k sampled gated events). Bin/occupancy health surfaces as a title strip plus a merged warning strip (fallback, forced-Auto, requested-vs-effective width, estimated edges, in-scan empties, dominant bin, Poisson advice, width spread, and a **narrow-bin aliasing flag** — aliasing checks apply to uniform grids only, since per-step / Raw-Voltage widths vary by construction). A blue **Scale:** line reports the raw-voltage bin size, the measured **1 V ≈ N MHz** conversion (median of adjacent-step slopes from the run's own gated events, with the min–max spread), and the equivalent rest-frame bin width in MHz.
  - *Compare* tab: per-run heatmap. All checked runs are re-binned onto a common grid (the same helper that powers the simultaneous-fit Common Grid mode) and rendered as `imshow` with rows = runs and columns = common bins. Optional log color scale. Skipped with a clear message if runs have mixed bin modes.
  - Right-click any non-merged file row → *Show Binning Diagnostics* opens the same dialog directly on the Run detail tab pre-selected to that run.
- Each fit iteration writes `binning_summary.csv` with per-run results and `binning_warnings.log` with the same warnings text the `fit_report.txt` header carries (codes: `BINNING_FALLBACK`, `EFFECTIVE_WIDTH_MISMATCH`, `RAW_VOLTAGE_FORCED_AUTO`, `DX_SPREAD_LARGE`, `COMMON_GRID_USED`, `VOLTAGE_TO_FREQUENCY_PROJECTION`, `MERGE_COOLER_SPREAD`, `MERGE_LASER_SPREAD`, `MANUAL_OFFSET_IGNORED_VOLTAGE_MERGE`). Pre-Phase-4 iterations wrote a `binning_warnings.json` — Results tab still reads either.
- `fit_report.txt` gets a `Binning Warnings` block at the top when any warnings fire; `run_summary.csv` gains `bin_mode, bin_definition, effective_n_bins, effective_bin_width_mhz, binning_fallback` columns.

**Common-grid simultaneous fits (Fitter block, simultaneous mode only):**
- Default off: each source keeps its own per-run bin grid. Runs with finer resolution carry more statistical weight per MHz.
- On: after per-run binning, every source is re-binned onto one shared x grid (union of x ranges; width either user-set or auto = median of per-run dx). Counts are conserved per source — statistics are not merged across runs, only the x grid is aligned.
- Requires all sources to share the same bin mode (all Frequency or all Raw Voltage); mixed modes fall back to independent grids with a warning.
- xerr (if any) is dropped on rebin.
- `binning_summary.csv` gets `common_grid_used` and `common_grid_width` columns; an info-level `COMMON_GRID_USED` warning surfaces in the report.

### Results Tab
- Browse all output organized by project and iteration (fresh iterations highlighted with a (NEW) marker)
- Live re-rendering of fit plots, tracker plots, diagnostics using plot settings (fit-values box, shade-under-fit, and academic tracker styling re-render from the saved `.npz` data)
- Plot editor for customizing saved figures; edits persist in a `.style.json` sidecar (gid-matched annotations, accumulated removals)
- **Copy / paste plot style** between plots with a property checklist (figure size, fonts, limits, grid, label positions, artist styles, annotations) — bulk paste to all checked plots
- **Export data as CSV**: right-click any `.npz`-backed plot to export its arrays (data, errors, model, residuals, dense fit curve) with column selection, header toggle, delimiter, and precision controls
- Project / iteration **notes** with autosave; bulk delete via checkboxes
- Export individual items or entire iterations (Export All)
- On load, exactly the iterations recorded in the save file are restored; **Refresh All** re-scans the whole output directory

### Tools Menu
- **Schmidt Moment Calculator**: single-particle and two-particle coupling
- **Shell Configuration Plotter**: nucleon shell filling diagrams
- **Unit Converter**: nm, cm^-1, MHz, GHz, THz, eV (CODATA 2018)
- **SHG Crystal Angle Calculator**: Type I phase-matching for BBO, KDP, LBO; wavelength → angle and angle → wavelength modes
- **SFG / DFG Calculator**: sum / difference frequency from two input wavelengths in any of the six units; result shown simultaneously in all six. Math goes through wavenumbers so mixed-unit inputs Just Work
- **Quick Plot**: load CSV/TSV, paste data, pick columns, plot instantly
- **NIST ASD Browser**: full in-app NIST Atomic Spectra Database browser + multi-step excitation **scheme finder** — levels/lines tables with filters, Grotrian-style level diagrams (lifetime-encoded opacity, hover tooltips), decay-channel and line-density plots, air/vacuum toggle (vacuum canonical, air computed locally), a cache-first offline store under `settings/nist_cache`, ranked 1–3-step scheme search with named-laser roles (PUMP/PROBE/ANY), branching-ratio and isobar-contamination checks, CSV export, and save-file persistence (`nist_asd` key)

### UI Features
- **Two themes**: the default dark theme, plus **Classic 98 (dark)** — Windows-98-style 3D bevels and sharp corners in dark colors with a pixel font (Settings ▸ General ▸ Theme, applies live)
- **Save model with dirty tracking**: Save (Ctrl+S, no dialog once a file is loaded), Save As... (Ctrl+Shift+S), Save Tab As...; closing a clean session never prompts, a dirty one offers Save / Don't Save / Cancel
- **New Window**: File ▸ New Window opens an independent instance (skips the single-instance guard and the splash)
- **Session outputs restore on load**: Estimate runs and Analysis iterations recorded in the save file come back without re-running; a whole-directory rescan stays behind Results ▸ Refresh All
- **Plot editor everywhere**: right-click any matplotlib canvas ▸ *Edit plot…* — six tabs (Figure, Axes, Artists, Add, Legend, Export) with real font pickers + bold/italic + live preview, add/remove text/annotations/lines/spans (numeric coordinates or click-to-place), per-artist position/rotation/z-order, linear/log scales, and Ctrl+Z / Ctrl+Y undo inside the editor
- **Zoom**: Ctrl+= / Ctrl+- / Ctrl+0 with persistent zoom level
- **Undo/Redo**: Ctrl+Z / Ctrl+Y track every spinbox value change app-wide (typed values, arrow keys, mouse wheel, arrow buttons; auto-detects newly created spinboxes)
- **Clipboard on spinboxes**: Ctrl+C copies the numeric value, Ctrl+V parses the clipboard text and applies it -- no need to select the text first
- **Wheel-focus guard**: scrolling over an unfocused spinbox/combo/slider scrolls the page instead of changing the value — click a field first to wheel-edit it
- **Lucide icons** throughout menus, tabs, and tool dialogs (white, MIT license)
- **Custom logo**: DENIS D-shaped CLS beamline drawing as window icon and splash screen
- **Splash screen** on startup with logo and version; fast warm start (~1.4 s) via lazy heavy imports

### Settings
- **Theme** selector: Dark (default) or Classic 98 (dark), applied live on OK
- Default plot settings with per-plot-type customization (Fit / Walk / Correlation / Chi-sq Map / Tracker / Preview-TOF, plus Isotope Shifts via the Output block's *Plot Options…* dialog)
- UI scale factor for high-DPI screens (QT_SCALE_FACTOR)
- Auto path conversion for Linux/Windows cross-platform workflows
- **Session log**: always written to `logs/denis_<timestamp>.log` (verbatim `stdout`+`stderr` plus a structured logger and crash hook; newest 20 kept). The **Verbose session log** toggle raises the level to DEBUG on the next launch
- Settings stored at `settings/settings.yaml` in the app directory (auto-migrated from old paths)
- Smart save/load with tab-level merging

---

## Dependencies

Pinned in `pyproject.toml` + `uv.lock` and resolved automatically by `uv sync`. For reference:

- **Python 3.10+** (downloaded by `uv` if not present)
- **PySide6** -- Qt6 GUI framework
- **matplotlib** -- Scientific plotting
- **numpy**, **pandas** -- Numerical computation
- **satlas2** -- Hyperfine structure fitting (v0.2.8+)
- **clstools** -- CLS data loading and processing; not on PyPI, pulled from [github.com/andry3vi/cls_tools](https://github.com/andry3vi/cls_tools) via `[tool.uv.sources]`, **pinned to `f6b9d7`** (adds `filter_calibration` and `ignore_intercept` upstream). DENIS overwrites `data.Cal` after every `Load_Run` (see [Voltage calibration](#voltage-calibration)), so its results do not depend on the pinned version -- both upstream controls are reimplemented in DENIS's own calibration fit where they actually reach the numbers.
- **PyYAML** -- Configuration persistence
- **asdf** -- ASDF data file format
- **lmfit** -- Fitting backend (used by satlas2)
- **emcee** -- MCMC sampling (optional, for emcee method)
- **pymc** -- PyMC v5 (with PyTensor backend, scipy, arviz). Required by the GP reference-centroid correction; the rest of the app loads fine without it but `Fit GP` raises a clear error
- **periodictable** -- Element lookup

---

## File Structure

```
DENIS/
|
|-- gui.py                          # GUI entry point
|-- cls_estimate.py                 # CLI entry point
|-- pyproject.toml                  # uv project + deps (Python, PySide6, satlas2, clstools, ...)
|-- uv.lock                         # Pinned dependency snapshot (committed)
|-- install.bat                     # Windows: menu / CLI for uv install, deps, and shortcut
|-- install.sh                      # Linux / macOS: same
|                                   # (subcommands: full | uv | shortcut)
|-- make_release.bat                # Zip a versioned release into share/
|-- .gitattributes                  # *.sh LF / *.bat CRLF
|
|-- icons/
|   |-- denis.ico                   # App icon (7 sizes: 16-256px)
|   |-- denis_512.png               # Splash screen logo
|   |-- denis_logo.svg              # Source logo (hand-drawn CLS beamline D)
|   |-- lucide/                     # Lucide icons (MIT, white stroke)
|       |-- atom.svg, sigma.svg, merge.svg, ...
|
|-- cls_estimations/                # Core computation library
|   |-- constants.py                # Physical constants (c, amu, e)
|   |-- config_parser.py            # YAML config loading
|   |-- doppler.py                  # Relativistic Doppler shift calculations
|   |-- hfs_model.py                # Hyperfine structure model (estimation)
|   |-- isotope_shift.py            # Isotope shift calculation, Birge inflation, GP cross-covariance, mixed Auto/Manual propagation
|   |-- reference_correction.py     # ReferenceCorrector GP class (PyMC v5; RBF / Matérn / 3-component thesis kernel)
|   |-- mass_lookup.py              # IUPAC/AME atomic mass table lookup
|   |-- plotting.py                 # Matplotlib plot helpers and palettes
|   |-- schmidt.py                  # Nuclear shell model (Schmidt moments)
|   |-- statistics.py               # Signal rate and timing estimation
|   |-- IUPAC-atomic-masses.csv     # CIAAW/AME mass table (2021)
|
|-- gui/                            # PySide6 GUI modules
|   |-- main_window.py              # Main window, menus, settings, dirty tracking, undo stack
|   |-- theme.py                    # Dark + Classic 98 themes, wheel-focus guard
|   |-- session_log.py              # Always-on session log (tee + crash hook)
|   |-- estimate_tab.py             # Estimate tab (single surface: parameters, run, plots, peak list)
|   |-- preanalysis_tab.py          # Pre-Analysis tab (data viewer, HFS models, merging, dark plots)
|   |-- preanalysis_container.py    # Multi-project Pre-Analysis wrapper
|   |-- results_tab.py              # Results tab (browser, live plots, style clipboard, CSV export)
|   |-- calibration.py              # Voltage-calibration fit, per-run overrides, cross-tab registry
|   |-- calibration_dialog.py       # Calibration diagnostic (fit + residuals + MHz cost) and overview
|   |-- calibration_alert.py        # Blinking "!" on flagged runs; click -> cost in MHz -> acknowledge
|   |-- scan_filter.py              # Per-scan exclusion registry (shared across tabs)
|   |-- scan_filter_dialog.py       # Filter scans... dialog
|   |-- split_editor.py             # Edit > Split File (.vasdf virtual splits)
|   |-- missing_files.py            # Locate-or-skip flow for moved data files
|   |-- shared_widgets.py           # Common widgets, plot editor, tool dialogs, settings I/O
|   |-- dialog_style.py             # Legacy styling shim (rules now live in theme.py)
|   |-- analysis/                       # Analysis tab (modular package)
|   |   |-- blocks.py                   # Source, Model, Fitter, Output block widgets + binning dialog
|   |   |-- binning.py                  # compute_binned: per-scan-step / uniform grids, yerr modes, warnings
|   |   |-- fitting.py                  # Model building and fit workers
|   |   |-- auto_fitter.py              # Multi-start parallel leastsq Auto-Fitter
|   |   |-- expr_validation.py          # Expression validator (namespaces, cycles)
|   |   |-- project.py                  # Analysis project orchestration (sample + reference projects)
|   |   |-- pipeline.py                 # Shared prepare_run_data() for every fit path
|   |   |-- tab.py                      # Top-level analysis tab with project tabs
|   |   |-- merge.py                    # Run merging dialog (honors GP correction when available)
|   |   |-- isotope_shift_tab.py        # Isotope-shift analysis + the GP Reference Correction panel embed
|   |   |-- reference_correction_panel.py  # GP centroid-correction UI (kernel pick, diagnostic plot, per-file table)
|   |   |-- helpers.py                  # Shared helpers, spinbox classes
|   |   |-- naming.py, vasdf.py         # Run naming; .vasdf sidecar I/O
|   |-- manual/                     # In-app manual (Help > Documentation, F1)
|   |   |-- structure.py, viewer.py, builder.py, content/, figs/
|   |-- nist_asd/                   # Tools > NIST ASD Browser
|       |-- data.py, models.py, search.py, plotting.py, tab.py
|
|-- settings/                       # User settings (auto-created)
|   |-- settings.yaml               # UI preferences, plot defaults, zoom, theme, etc.
|   |-- nist_cache/                 # Cached NIST spectra (Si I pre-seeded for offline first open)
|
|-- logs/                           # Session logs (always written; newest 20 kept)
|   |-- denis_2026-04-28_143015.log # Verbatim stdout+stderr copy of the session
|
|-- configs/                        # Example YAML configuration files (example_*.yaml)
|-- docs/
|   |-- manual/                     # LaTeX user manual
|       |-- manual.tex, preamble.tex, references.bib
|       |-- sections/               # 9 numbered sections + 3 appendices
|       |-- figs/                   # Screenshots
|       |-- build.bat, build.sh     # latexmk wrappers (MiKTeX / TeX Live)
|-- tests/                          # Headless-safe pytest suite (offscreen Qt)
```

---

## Configuration

Settings are stored at `settings/settings.yaml` (inside the app directory) and include:
- UI theme (`ui_theme`: `dark` or `win98`)
- Performance (max CPU cores for parallel fitting)
- Output directory and iteration mode
- Default fitting method, statistics, yerr mode
- Plot defaults (global rcParams + per-plot-type: fit, walk, correlation, chi-sq, tracker, preview, isotope shifts)
- UI scale factor (0 = auto, or 1.25/1.5/2.0 for high-DPI)
- Auto path conversion toggle (Linux/Windows)
- Zoom level persistence; Pre-Analysis dark-plot state (`pa_dark_plots`) and saved custom line colors (`pa_custom_colors`)

---

## Data Format

The toolkit reads **ASDF** files produced by the IGISOL CLS DAQ system. Each file contains:
- Raw event data (timestamp, scanning voltage, bunch, PMT channel, TOF, cooler voltage)
- Calibration data (set/readback voltage pairs)
- Metadata (run number, experiment, date, laser setpoint, dwell time, scanning ranges)

Processing pipeline: `Load_Run` -> **apply calibration** -> `Compute_Voltages` -> `Compute_WL` -> `Compute_Bins` (via clstools)

---

## Voltage calibration

Each run carries a DAC->HV calibration table (`CalSet` -> `CalReadback`). A polynomial
through it turns every event's DAC value into a real voltage, so if the table is wrong the
whole frequency axis is wrong. The usual failure is the HV supply still settling over the
first points of the calibration sweep: the fit gets pulled toward them, and **every**
voltage in the run is biased.

This is a first-order systematic, not a rounding detail. At 30 kV with mass 51, **1 V of
calibration error is ~18 MHz** of centroid shift -- comparable to the isotope shifts being
measured. Worse, an offset error and a gain error cancel mid-sweep and add at the edges, so
a bad calibration **tilts** the frequency axis rather than translating it, moving peaks by
different amounts depending on where they sit in the scan.

**DENIS owns the calibration fit** (`gui/calibration.py`): every load path calls
`load_run_calibrated`, which runs `Load_Run` and then overwrites `data.Cal` before
`Compute_Voltages` reads it. Two consequences:

- The numbers **do not depend on which clstools is installed** (upstream `c0334c0` added an
  on-by-default 2σ cut that silently drops points; the previous pin did not).
- **With no override, the default is a plain polynomial over every calibration point** --
  bit-for-bit the unfiltered clstools fit.

> **⚠ This changed the default on machines whose `clstools` was the newer, filtering build.**
> Before this feature, `Load_Run(filter_calibration=True)` was silently dropping every
> calibration point beyond 2σ of the residuals. DENIS now switches that filter off and
> keeps all points unless *you* say otherwise, so a run with a calibration outlier can fit
> to a different centroid than it did previously — by design: rejection is now an explicit,
> recorded choice rather than a silent one. Affected runs are exactly the ones the overview
> flags with ⚠. If you have results you need to reproduce bit-for-bit against the old
> behaviour, set those runs to `Reject: n·σ`, `σ = 2`, non-iterative, which reproduces the
> old filter exactly.

### Using it

A run whose calibration outliers would actually move its centroid gets a **blinking `!`**
on its row, in both Pre-Analysis and Analysis. Click it: you get the outlier count, the
centroid shift in MHz, the tilt across the scan, and a choice of *Open calibration...* or
*Keep as-is*. Either way the blinking stops for good -- a warning that keeps nagging after
it has been read is one you learn to click past. The acknowledgement is saved with the
project.

Right-click a run -> **Calibration...** for the full diagnostic.

- **Three panels** -- the calibration curve with the **uncorrected** (file) and **corrected**
  (in force) polynomials overlaid and labelled with their coefficients; residuals against
  the uncorrected calibration; and residuals against the one in force. The two residual
  panels are the before/after: a settling glitch puts them ~30x apart in sigma, so they get
  their own y-scales. Points are coloured by acquisition order, which makes a start-of-sweep
  failure obvious at a glance.
- **What it costs in MHz.** The headline number: volts of residual are not actionable, a
  centroid shift next to your isotope shift is.
- **Exclude points** -- click them, or apply a rule: *drop the first N* (matches the physical
  failure) or *n·σ residual outliers*. Prefer the iterative form of the σ rule: a single pass
  computes σ from data that still contains the outliers, so a cluster of bad points inflates
  σ enough to hide inside the cut -- which is exactly what clstools' own 2σ filter does.
- **Borrow another run's calibration** -- run 1001's coefficients applied to run 1002. Warns
  if the donor's sweep does not cover this run's range.
- **Enter coefficients by hand** -- `DV_cal[V] = p0 + p1·DV + ...`, where `p0` is an offset in
  volts and `p1` a gain of ~1.
- **Calibration overview** -- every loaded run, worst first, so a campaign can be triaged at
  a glance instead of one dialog at a time.

By default nothing is dropped: runs whose outliers would actually move a centroid are
**flagged** (the blinking `!`, a `[cal]` badge, a ⚠ in the overview), and you decide. Note
the flag keys on *physical impact*, not statistical significance -- on a clean 20-point
table, Gaussian noise routinely throws a point past 3σ, and a warning that fires on good
runs is worse than none. Every drop is recorded in the fit report and in merged files'
per-run provenance.

The calibration is applied where clstools applies it -- between `Load_Run` and
`Compute_Voltages` -- so it reaches **everything**: the Frequency and Raw Voltage spectra,
the calibrated-voltage / beam-energy / wavenumber display axes, virtual splits (which key
off their parent ASDF), merges, the auto-fitter, and the fit itself.

Overrides are a property of the *file*, not of an analysis, so they live in a process-wide
registry shared by every tab and persist under the project YAML's top-level `calibrations:`
key -- a sibling of `scan_filters:`.

---

## Documentation

- **In-app manual**: Help ▸ Documentation (F1) — an interactive, cross-linked reference covering every tab, tool, and file format
- **PDF manual**: [`docs/manual/manual.pdf`](docs/manual/manual.pdf) (LaTeX sources in `docs/manual/`; rebuild with `build.bat` / `build.sh`, needs MiKTeX or TeX Live)

---

## License and Citation

DENIS is released under the [MIT License](LICENSE). If you publish results produced with it, please cite DENIS (see [`CITATION.cff`](CITATION.cff) — GitHub's *Cite this repository* button) along with **satlas2**, **clstools**, and the CIAAW/AME mass table for the underlying methods and data.

---

## References

- **satlas2:** [iks-nm.github.io/satlas2](https://iks-nm.github.io/satlas2/index.html) -- W. Gins et al., KU Leuven
- **clstools:** [github.com/andry3vi/cls_tools](https://github.com/andry3vi/cls_tools) -- A. Raggio, JYFL
- **IUPAC masses:** [ciaaw.org](https://www.ciaaw.org/) -- CIAAW/AME recommended nuclide masses (2021)
- **Lucide icons:** [lucide.dev](https://lucide.dev/) -- MIT/ISC license
