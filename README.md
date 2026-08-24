# DENIS

[![DOI](https://zenodo.org/badge/1344933579.svg)](https://doi.org/10.5281/zenodo.22081266)

**D**oppler **E**stimation and **N**umerical **I**nference for **S**pectroscopy

A comprehensive tool for **Collinear Laser Spectroscopy (CLS)**: run-time estimation, pre-analysis, SATLAS2-based fitting, and results visualization.

**Version:** 1.0.0
**Developer:** [Arda Kayaalp](https://ardakayaalp.com/) (arda.kayaalp@kuleuven.be)
**Framework:** PySide6 (Qt6) + matplotlib + satlas2 + clstools

---

## Quick Start

The recommended install path uses [`uv`](https://docs.astral.sh/uv/): it bootstraps Python itself, the project venv, and every dependency in one step. No existing Python install is required.

```bash
# Get the code
git clone https://github.com/ardakayaalp/DENIS.git
cd DENIS

# First-time install (bootstraps uv, runs `uv sync`, creates a desktop
# shortcut). No args = interactive menu; subcommands: full | uv | shortcut
install.bat full       # Windows
./install.sh full      # Linux / macOS

# Launch from the desktop shortcut, or from a terminal:
uv run python gui.py

# CLI estimation / load a save file on startup
uv run python cls_estimate.py config.yaml --output-dir ./output
uv run python gui.py my_config.yaml
```

If `uv` is already installed: `uv sync && uv run python gui.py`.

---

## Features

### Estimate Tab
- Run-time estimation and HFS spectrum simulation in voltage space; isotope shift and resonance voltage calculations for multi-isotope plans
- Single-surface layout: global parameters, isotope list (color-striped), and Run Options + plots + log; an expand toggle covers the whole tab with the plots
- Runs persist (`estimate_results.npz` + `peaks.csv` per `cls_<timestamp>` folder) and restore on load without re-running; **Load Estimation...** reopens any previous run, rendering PDF-only legacy runs onto the live canvas
- Sortable **Peak List** view; intensity-ordered and voltage-ordered peak tables in the log

### Pre-Analysis Tab
- Load and overlay ASDF runs with per-run color, opacity, and line style; TOF histogram with interactive gate and adjustable bin size (down to 1 ns); timestamp overview with per-scan exclusion
- Spectrum in voltage, frequency, or wavenumber, binned **per scan step** by default (immune to uniform-grid aliasing); a header **Bin: N x step** spin groups adjacent steps
- Layout switcher (3-row / 2-row), X/Y grid toggles, and a **dark-mode plot toggle** (black canvas, neon lines); default run colors from the classic Windows 98 palette, with separate light + dark colors per file and model
- HFS model overlay with sliders, ratio locks, per-peak Racah/Free/Linked amplitudes, and peak labels
- Spectrum merging in voltage or Doppler-shifted frequency domain; virtual file splitting (`.vasdf`); per-file centroid offsets
- Calibration and cooler-voltage diagnostics (robust statistics, spike detection, per-run summary table with a MHz-impact column)
- **Voltage-calibration diagnostics and per-run overrides** -- see below

### Analysis Tab
- Block-based fitting pipeline: Source -> Model -> Fitter -> Output
- **Models:** HFS (sidepeaks, Racah), Voigt, Skewed Voigt, Exponential Decay, Piecewise Constant, Polynomial. **Methods:** leastsq, least_squares, slsqp, emcee (MCMC), nelder, powell, cobyla. **Statistics:** Chi-square, Gaussian LLH, Poisson LLH
- Separate (per-file, parallel) and simultaneous fitting; opt-in parameter sharing, expression constraints with live validation, Gaussian priors, MCMC diagnostics (walk/correlation plots, confidence bands, chi-square maps)
- **Binning as first-class controls**: `Per scan step` (default), `Auto`, `Fixed bin count`, `Fixed bin width`; a `Bin multiple` spin groups adjacent steps; per-file binning overrides via right-click; optional common-grid re-binning for simultaneous fits
- A three-tab **Binning dialog** (Summary occupancy table, Run detail with bin edges + a measured 1 V ~ N MHz scale strip + aliasing warnings, Compare heatmap); binning warnings land in the fit report and `binning_summary.csv`
- Multi-start parallel leastsq **Auto-Fitter** to seed initial guesses; **Update iteration** regenerates outputs into the existing iteration without re-fitting (walk/correlation rebuilt from saved chains)
- **Isotope Shift tab**: weighted averages with Birge inflation, systematic errors from cooler/laser jitter, a full per-shift error budget, and a run preview panel
- **Reference Centroid Correction (GP)**: time-dependent drift correction following van den Borne (2025). Fit reference scans in a Reference Project, train a GP (RBF / Matern(5/2) / composite kernel via PyMC v5), and apply per-run corrections consistently in fits, merges, and the Auto-Fitter; correlated corrections partially cancel in shifts via the GP cross-covariance
- On-plot fit-values box, shade-under-fit, and per-plot-type styling via the Output block's **Plot Options...** dialog

### Results Tab
- Browse everything by project and iteration (fresh results marked NEW); live re-rendering of fit, tracker, and diagnostic plots from their `.npz` data
- Plot edits persist in `.style.json` sidecars; **copy/paste plot style** between plots with a property checklist (bulk paste supported)
- **Export data as CSV** (data, errors, model, residuals, fit curve) with column/delimiter/precision control; export single items or whole iterations
- Project and iteration notes with autosave; on load, exactly the saved iterations are restored (Refresh All re-scans the directory)

### Tools Menu
- **Schmidt Moment Calculator** (single- and two-particle), **Shell Configuration Plotter** (spherical shell model, Nordheim rule), **Unit Converter** (nm, cm^-1, MHz, GHz, THz, eV), **SHG Crystal Angle Calculator** (Type I, BBO/KDP/LBO), **SFG / DFG Calculator** (mixed-unit inputs), **Quick Plot** (CSV/TSV or pasted data)
- **NIST ASD Browser**: in-app NIST Atomic Spectra Database browser + multi-step excitation **scheme finder** -- level/line tables and diagrams (lifetime-encoded opacity), air/vacuum toggle, offline cache, ranked 1-3-step scheme search with laser roles, branching-ratio and isobar-contamination checks, CSV export, save-file persistence

### UI & Settings
- Two themes: default dark and **Classic 98 (dark)**, applied live; Lucide icons; splash screen
- Editor-style save model with dirty tracking (Save / Save As / Save Tab As; clean sessions close silently); **New Window** opens an independent instance
- **Plot editor everywhere**: right-click any canvas -> *Edit plot...* -- six tabs with real font pickers, add/remove annotations and lines, per-artist position/rotation/z-order, log scales, and its own Ctrl+Z/Ctrl+Y undo
- App-wide spinbox undo/redo and Ctrl+C/Ctrl+V; wheel-focus guard (scrolling never edits an unfocused field); zoom (Ctrl+= / Ctrl+- / Ctrl+0)
- Settings in `settings/settings.yaml`: theme, plot defaults (global + per-plot-type), UI scale, path auto-conversion, output/fitting defaults
- Session log always written to `logs/denis_<timestamp>.log` (newest 20 kept, crash hook included); a Verbose toggle raises it to DEBUG

---

## Dependencies

Pinned in `pyproject.toml` + `uv.lock` and resolved automatically by `uv sync`. For reference:

- **Python 3.10+** (downloaded by `uv` if not present)
- **PySide6** -- Qt6 GUI framework
- **matplotlib**, **numpy**, **pandas** -- plotting and numerics
- **satlas2** -- hyperfine structure fitting (v0.2.8+)
- **clstools** -- CLS data loading; not on PyPI, pulled from [github.com/andry3vi/cls_tools](https://github.com/andry3vi/cls_tools) via `[tool.uv.sources]`, pinned to `f6b9d7`. DENIS overwrites `data.Cal` after every `Load_Run` (see [Voltage calibration](#voltage-calibration)), so results do not depend on the pinned version
- **PyYAML**, **asdf**, **lmfit**, **emcee**, **periodictable**
- **pymc** -- PyMC v5, required only by the GP reference-centroid correction

---

## Repository Layout

```
DENIS/
|-- gui.py                  # GUI entry point
|-- cls_estimate.py         # CLI entry point
|-- pyproject.toml, uv.lock # uv project + pinned dependencies
|-- install.bat, install.sh # installers (full | uv | shortcut)
|-- icons/                  # app icon, logo, Lucide icons (MIT)
|-- cls_estimations/        # core computation library (Doppler, HFS,
|                           #   Schmidt, isotope shifts, GP corrector,
|                           #   IUPAC/AME mass table)
|-- gui/                    # PySide6 modules: main window + themes,
|   |-- analysis/           #   the four tabs, calibration machinery,
|   |-- manual/             #   in-app manual (Help > Documentation),
|   |-- nist_asd/           #   NIST ASD Browser
|-- settings/               # user settings + pre-seeded NIST cache
|-- configs/                # example YAML configurations
|-- docs/manual/            # LaTeX manual (manual.pdf pre-built)
|-- tests/                  # headless pytest suite (offscreen Qt)
```

---

## Data Format

The toolkit reads **ASDF** files produced by the IGISOL CLS DAQ system: raw event data (timestamp, scanning voltage, bunch, PMT channel, TOF, cooler voltage), calibration tables (set/readback pairs), and run metadata.

Processing pipeline: `Load_Run` -> **apply calibration** -> `Compute_Voltages` -> `Compute_WL` -> `Compute_Bins` (via clstools)

---

## Voltage calibration

Each run carries a DAC->HV calibration table; a polynomial through it turns every event's DAC value into a real voltage, so a bad table biases the whole frequency axis. At 30 kV with mass 51, **1 V of calibration error is ~18 MHz** of centroid shift, and an offset+gain error *tilts* the axis rather than translating it.

**DENIS owns the calibration fit** (`gui/calibration.py`): every load path overwrites `data.Cal` before `Compute_Voltages`, so the numbers do not depend on the installed clstools build, and by default **nothing is dropped** (upstream clstools silently applies a 2-sigma cut; DENIS turns it off and makes rejection an explicit, recorded choice).

- Runs whose calibration outliers would actually move their centroid get a **blinking `!`** badge (Pre-Analysis and Analysis); clicking it reports the cost in MHz and the acknowledgement is saved with the project
- Right-click a run -> **Calibration...**: fit-vs-file overlay, residuals before/after, cost in MHz, point exclusion (click, drop-first-N, or iterative n-sigma), a zero-intercept (p0 = 0) option, borrowing another run's calibration, or manual coefficients
- **Calibration overview...** triages every loaded run at once, worst first
- Overrides are per-file, shared across tabs, and persist under the save file's top-level `calibrations:` key; every drop is recorded in the fit report

Reproducing old filtered results bit-for-bit: set the flagged runs to `Reject: n-sigma`, sigma = 2, non-iterative.

---

## Documentation

- **In-app manual**: Help > Documentation (F1) -- an interactive, cross-linked reference covering every tab, tool, and file format
- **PDF manual**: [`docs/manual/manual.pdf`](docs/manual/manual.pdf) (LaTeX sources alongside; rebuild with `build.bat` / `build.sh`)

---

## License and Citation

DENIS is released under the [MIT License](LICENSE). Releases are archived on Zenodo: cite v1.0.0 via [10.5281/zenodo.22081267](https://doi.org/10.5281/zenodo.22081267), or all versions via the concept DOI [10.5281/zenodo.22081266](https://doi.org/10.5281/zenodo.22081266) (also in [`CITATION.cff`](CITATION.cff), surfaced as GitHub's *Cite this repository* button). Please also cite **satlas2**, **clstools**, and the CIAAW/AME mass table for the underlying methods and data.

---

## References

- **satlas2:** [iks-nm.github.io/satlas2](https://iks-nm.github.io/satlas2/index.html) -- W. Gins et al., KU Leuven
- **clstools:** [github.com/andry3vi/cls_tools](https://github.com/andry3vi/cls_tools) -- A. Raggio, JYFL
- **IUPAC masses:** [ciaaw.org](https://www.ciaaw.org/) -- CIAAW/AME recommended nuclide masses (2021)
- **Lucide icons:** [lucide.dev](https://lucide.dev/) -- MIT/ISC license
