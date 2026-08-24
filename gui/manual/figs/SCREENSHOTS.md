# DENIS manual — screenshots needed

> **2026-08-24 status:** every PNG in this folder was captured 2026-06-02,
> *before* the 2026-07 UI overhaul (single-surface Estimate, 2-pane PA,
> corner Layout/Grid/dark controls, new File menu, themes, NIST browser).
> The manual *text* is current; the images are the remaining refresh —
> recapture the checklist below against the new UI in a real window
> (offscreen renders have no fonts). New shots to add while at it:
> the NIST ASD Browser (Levels + Scheme Finder), the PA dark-mode plots,
> the 3-tab Binning dialog, and the six-tab Plot Editor with its Add tab.

Drop captured images in this folder. Use the **suggested filename** for each (or
any name — they can be remapped during embedding). Each "captures" line is one
distinct screenshot to take; the `fills:` note lists the manual page slot(s) that
image serves (one capture can fill several pages — it gets copied per slot).

Take each shot on the developer's own data; placeholder slots show until the file
is added. ~49 distinct captures.

Legend:  `[ ]` to capture · `[x]` done · `[gen]` generated, no screenshot needed.

---

## Global UI & dialogs

- [ ] **main_window.png** — main window on launch: four-tab rolodex bar (Estimate,
  Pre-Analysis, Analysis, Results) + File/Edit/View/Run/Tools/Help menu bar + status bar.
  `fills: ui_main_window.png, main_window.png`
- [ ] **menu_file.png** — File menu expanded (Save All / Load All / Save Tab / Load Tab /
  Settings / Exit) showing keyboard shortcuts. `fills: menu_file_expanded.png, settings_file_menu_saveload.png`
- [ ] **missing_files_dialog.png** — the Missing Files dialog (Locate Files… / Skip Missing /
  Cancel). `fills: config_missing_files_dialog.png, settings_missing_files_dialog.png`
- [ ] **undo_reveal.png** — an undo in progress: DENIS switched to the affected tab and
  flashed an amber highlight on the reverted spinbox. `fills: ui_undo_reveal.png`
- [ ] **plot_editor.png** — Plot Editor dialog (Figure / Axes / Artists / Legend / Export
  tabs) beside a live figure, "Click to select / drag on plot" visible.
  `fills: ui_plot_editor.png, results_plot_editor.png`
- [ ] **status_bar.png** — status-bar states: a normal timestamped message, the embedded
  fitting progress bar, and the amber "analysis complete — click Results" flash. `fills: ui_status_bar_states.png`
- [ ] **about_dialog.png** — Help → About (logo, version, D-E-N-I-S expansion, satlas2 /
  cls_tools links, contact). `fills: help_about_dialog.png, citation_about_dialog.png`
- [ ] **install_menu.png** — the installer's interactive terminal menu ([1] Full, [2] uv,
  [3] shortcut). `fills: troubleshooting_install_menu.png, install_menu.png`

## Estimate tab

- [ ] **estimate_three_columns.png** — Parameters & Run with all three columns (globals |
  isotope list | Run Options & Log) and ≥1 isotope panel added. `fills: estimate_layout_three_columns.png, qs_estimate_params.png`
- [ ] **estimate_laser_modes.png** — Laser group in Anchor-to-isotope mode (setpoint field
  greyed). `fills: estimate_laser_modes.png`
- [ ] **estimate_isotope_list.png** — column 2: "+ Add Isotope", count, two isotope panels,
  one marked Reference (blue border). `fills: estimate_isotope_list.png`
- [ ] **estimate_isotope_expanded.png** — an isotope panel expanded: nuclear fields
  (A/amu, Label, I, μ, Q, isotope shift) + isomer + Schmidt sub-panels. `fills: estimate_isotope_panel_expanded.png`
- [ ] **estimate_run_log.png** — column 3: Run Options (output dir + Browse, palette, Generate
  plots), the bold Run Estimation button, syntax-highlighted log with Grand Total + peak list.
  `fills: estimate_run_options_log.png, qs_estimate_output.png`
- [ ] **estimate_plots.png** — Plots sub-tab, View = Individual Spectra, dashed peak markers,
  toolbar + Edit Plot / Open PDF. `fills: estimate_plots_subtab.png` (also reused for physics-simulated-spectrum)

## Pre-Analysis tab

- [ ] **preanalysis_overview.png** — full 3-column layout (Data Files + Plot Options |
  Spectrum/Calibrations/Cooler tabs | HFS Models). `fills: preanalysis_overview_layout.png, qs_preanalysis.png`
- [ ] **preanalysis_file_list.png** — Data Files list: tri-state Check-all, Remove unchecked /
  Merge Checked, FileEntry rows (tickbox, swatch, reload, run name, V/λ/t line). `fills: preanalysis_file_list.png`
- [ ] **preanalysis_spectrum_panels.png** — Spectrum tab's three stacked panels (TOF /
  Spectrum / Timestamp) with toolbars + Timestamp Unit/Bin controls. `fills: preanalysis_spectrum_panels.png`
- [ ] **preanalysis_tof_gate.png** — orange TOF gate dragged over the ion bunch; Spectrum
  re-binning live below. `fills: preanalysis_tof_gate.png`
- [ ] **preanalysis_xaxis_modes.png** — same spectrum in Voltage vs Frequency, x-axis
  dropdown open (all five entries). `fills: preanalysis_xaxis_modes.png`
- [ ] **preanalysis_plot_options.png** — Plot Options group (x-axis combo, E lower/upper +
  Harmonic, Z/A + label, Mass + Override, Channels, Normalize, Cooler/Laser overrides). `fills: preanalysis_plot_options.png`
- [ ] **preanalysis_hfs_panel.png** — an HFS Models panel expanded (header, I/Jl/Ju, Al/Au/Bl/Bu
  with Fix ratio locks, value+slider+limits, Peak Amplitudes). `fills: preanalysis_hfs_panel.png`
- [ ] **preanalysis_scan_filter.png** — Filter Scans dialog (per-scan table + Check/Uncheck +
  Exclude rate < threshold). `fills: preanalysis_scan_filter_dialog.png`
- [ ] **preanalysis_scan_overlay.png** — Timestamp plot with Show scans: grey included / red
  excluded bands + "Exclude scans in view". `fills: preanalysis_scan_overlay.png`
- [ ] **preanalysis_merge_dialog.png** — the Merge dialog (domain combo voltage/frequency +
  bin step) with the synthetic merged entry behind. `fills: preanalysis_merge_dialog.png`
- [ ] **preanalysis_cooler_voltage.png** — Cooler Voltage tab (raw / deviation with σ bands /
  ripple, status strip, Clip y + Bins). `fills: preanalysis_cooler_voltage.png`
- [ ] **split_editor.png** — Split File editor: raw-voltage histogram, draggable red cut,
  blue/orange sides, per-side metadata forms, counts line. `fills: preanalysis_split_editor.png, splitfile_editor_overview.png`

## Analysis tab

- [ ] **analysis_create_project.png** — the "+ Create Analysis Project" menu (Sample /
  Reference) with a (Reference)-suffixed tab and the permanent Isotope Shifts tab. `fills: analysis_create_project_menu.png`
- [ ] **analysis_pipeline.png** — the Source → Model → Fitter → Output pipeline (colour-coded
  borders, enable checks, drag handles, + Add Block). `fills: analysis_pipeline_blocks.png, qs_analysis_pipeline.png`
- [ ] **analysis_source_block.png** — Source block expanded (Files list + Check-all, Physics
  Parameters, Gates, Binning, Preview/Meta/Run Stats toolbar). `fills: analysis_source_block.png`
- [ ] **analysis_model_block.png** — Model block (Type = HFS, Racah/Sidepeak, param table with
  Value/Bounds/colour-coded Mode combos + a red invalid Expression cell). `fills: analysis_model_block.png`
- [ ] **analysis_fitter_block.png** — Fitter block in Simultaneous mode (Method & Statistics,
  Parameter Sharing, Advanced Constraints, Common Grid, Run Fit / Find Parameters / Stop / Revert).
  `fills: analysis_fitter_simultaneous.png`
- [ ] **analysis_output_block.png** — Output block (Reports, Fit Plots + plot-type dropdown +
  Format/DPI, Tracker Plots, Diagnostics, Iteration Auto/Manual, Re-apply outputs). `fills: analysis_output_block.png`
- [ ] **analysis_autofitter.png** — Auto-Fitter window mid-sweep (Starts/Spread/Cores,
  χ²-ranked results table, live fit plot, Core Activity). `fills: analysis_autofitter_window.png`
- [ ] **analysis_isotope_shifts.png** — Isotope Shifts tab (Isotope Entries + Systematic Error
  on left; comparison plot with centroid lines + δν arrows and the shift table on right).
  `fills: analysis_isotope_shifts_tab.png`
- [ ] **analysis_reference_gp.png** — Reference Correction (GP) panel (kernel/MCMC/apply row,
  project checklists, scatter + MAP curve + 1/2σ bands, corrections table). ==> discard it for now, I will put an example and take a screenshot later.
  `fills: analysis_reference_gp_panel.png` (also reused for physics-gp)
- [ ] **analysis_fit_progress.png** — a fit in progress: Fitter block progress bar + status,
  Stop enabled / Run Fit disabled. `fills: analysis_fit_progress.png`=> not necessary for now.

## Results tab

- [ ] **results_overview.png** — the Results tab: tree on left, live fit plot + toolbar on
  right, Edit Plot / Export / Export All / Info row. `fills: results_layout_overview.png, qs_results.png`
- [ ] **results_tree.png** — the tree: hue-tinted project header, a bold-amber (NEW) iteration,
  expanded Fit Report / Parameters / Metadata / plot items. `fills: results_tree_fresh_iter.png`
- [ ] **results_live_plot.png** — a live fit plot from a .npz sidecar (clipped error bars, fit
  line, residual panel sharing x, navigation toolbar). `fills: results_live_fit_plot.png` ==> not necessary for now
- [ ] **results_export_all.png** — Export All: destination picker + the copied iter_xxx folder
  (reports, CSVs, plots, .npz, .style.json). `fills: results_export_all.png` => not necessary for now

## Tools (open each calculator with the example inputs)

- [ ] **tools_schmidt.png** — Schmidt Moment Calculator, Two-particle mode (Type/Orbital ×2,
  Total I, g_s quench, live μ / g / Description). `fills: tools_schmidt_two_particle.png`
- [ ] **tools_shell_config.png** — Shell Configuration Plotter for ⁷²Ge (config strings,
  predicted Jπ, red-proton/blue-neutron shell ladder with magic-number gaps).
  `fills: tools_shell_config_72ge.png` (also reused for physics-schmidt-single)
- [ ] **tools_unit_converter.png** — Unit Converter, 532 nm entered, six output fields filled.
  `fills: tools_unit_converter_532nm.png`
- [ ] **tools_shg.png** — SHG Crystal Angle Calculator, Wavelength→Angle, BBO, 1064 nm →
  phase-matching angle + 532 nm. `fills: tools_shg_bbo_1064.png`
- [ ] **tools_sfg_dfg.png** — SFG/DFG Calculator, 1064 + 532 nm, Sum (SFG), six-unit output.
  `fills: tools_sfg_dfg_1064_532.png`
- [ ] **tools_quick_plot.png** — Quick Plot, CSV loaded + Line chart (X/Y dropdowns, Type/Color,
  title/grid/legend, embedded canvas). `fills: tools_quick_plot_csv_line.png`

## Settings

- [ ] **settings_overview.png** — the Settings dialog (General / Performance / Output / Fitting
  Defaults / Plot Defaults groups, OK/Cancel). `fills: settings_dialog_overview.png`
- [ ] **settings_plot_defaults_fitplot.png** — Plot Defaults → Fit Plot sub-tab (Auto-sizing
  thresholds + Low/Medium/High region groups). `fills: settings_plotdefaults_fitplot.png`
- [ ] **settings_session_log.png** — Settings General group with the "Save session log
  (restart)" checkbox + tooltip. `fills: troubleshooting_session_log_toggle.png`
  (File menu, Missing Files dialog, and Split File editor here reuse menu_file.png /
  missing_files_dialog.png / split_editor.png above.) => no need

## Physics Reference (figures) => no need for now

- [gen] **timing curve** `t = σ²(S+B)/S²` — generated as a matplotlib diagram. `fills: physics_timing_curve.png` ✓
- [ ] physics-simulated-spectrum — **reuses** estimate_plots.png. `fills: physics_simulated_spectrum.png`
- [ ] physics-schmidt-single (orbital ladder) — **reuses** tools_shell_config.png, OR ask me
  to generate it. `fills: physics_schmidt_orbitals.png`
- [ ] physics-gp — **reuses** analysis_reference_gp.png. `fills: physics_gp_diagnostic.png`

---

## Status (embedding pass, 2026-06-02)

**Embedded ✓** — 39 provided captures copied into 44 page slots; rendering live.

**Deferred to text (per your notes)** — these pages now show a short italic
"(Figure deferred — …)" description instead of a figure; drop the image in and
ask me to re-embed to restore the figure:
- `analysis_reference_gp_panel.png`  (Analysis ▸ Reference Correction GP)
- `analysis_fit_progress.png`        (Analysis ▸ Running fits)
- `results_live_fit_plot.png`        (Results ▸ Live plots)
- `results_export_all.png`           (Results ▸ Export)
- Physics figures (simulated-spectrum, schmidt-orbitals, gp-diagnostic) — no
  figure blocks were ever placed, so nothing shows; add later if wanted.

**Resolved (2026-06-02):**
- `main_window.png` ✓ embedded (fills `ui_main_window.png` + `main_window.png`).
- `analysis_merge_dialog.png` — intentionally **not used**; the Analysis Merge
  dialog is identical to the Pre-Analysis one, so that page now links to
  `pa-merging` in text instead of showing a duplicate screenshot.

All referenced screenshot slots are now filled — no placeholders remain.
