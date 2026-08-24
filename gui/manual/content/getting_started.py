"""Authored manual pages for the 'Getting Started' section.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds the HTML for the introduction, installation, launching, four-tab tour,
and per-tab quick-start pages. UI labels here are verified against the current
code (install.bat / install.sh, pyproject.toml, gui.py, cls_estimate.py, and the
four tab modules). Screenshots are referenced by name via R.screenshot(); until
the image file is added under gui/manual/figs/ it renders as a placeholder slot.

Depends on: gui.manual.render.
"""

from gui.manual import render as R

# Command snippets and paths are defined here (not inline in the f-strings)
# so no backslash ever appears inside an f-string expression — that is a
# SyntaxError on Python 3.10/3.11 (the project's minimum), and "\\n" inside an
# expression would render a literal \n rather than a line break.
_FULL_INSTALL = (
    "install.bat full           REM Windows\n"
    "./install.sh full          # macOS / Linux"
)
_MANUAL_INSTALL = (
    "uv sync                    # install Python + dependencies into .venv\n"
    "uv run python gui.py       # launch"
)
_UV_PATH_WIN = r"%USERPROFILE%\.local\bin"
_UV_PATH_NIX = "~/.local/bin"


def introduction():
    return f"""
<h1>What DENIS Is</h1>

<p><b>DENIS</b> &mdash; <b>D</b>oppler <b>E</b>stimation and <b>N</b>umerical
<b>I</b>nference for <b>S</b>pectroscopy &mdash; is a desktop application for
<i>collinear laser spectroscopy</i> (CLS). It supports the full measurement
workflow: planning how much beam time a measurement needs, inspecting raw
spectra as they come off the experiment, fitting hyperfine structure, and
reporting the extracted nuclear observables.</p>

<p>DENIS is built with PySide6 (the Qt GUI toolkit) and matplotlib, uses the
<b>satlas2</b> library for spectrum modelling and fitting, and reads
IGISOL-style ASDF data files through <b>clstools</b>.</p>

{R.diagram("four-tab-map",
           "The four tabs follow the natural order of a CLS campaign.")}

<h2>The four tabs</h2>
<ul>
  <li><b>{R.link("tab-estimate", "Estimate")}</b> &mdash; predict beam time and
      simulate a hyperfine spectrum from atomic and nuclear inputs, before any
      beam is taken.</li>
  <li><b>{R.link("tab-preanalysis", "Pre-Analysis")}</b> &mdash; load ASDF runs,
      gate on time-of-flight, bin, switch between voltage and Doppler-frequency
      axes, and overlay a model to check the data live.</li>
  <li><b>{R.link("tab-analysis", "Analysis")}</b> &mdash; build a fitting pipeline
      (Source &rarr; Model &rarr; Fitter &rarr; Output), run satlas2 fits, and
      combine centroids into isotope shifts.</li>
  <li><b>{R.link("tab-results", "Results")}</b> &mdash; browse every artefact a
      fit produced, re-style plots, write notes, and export.</li>
</ul>

<h2>Who it is for</h2>
<p>DENIS is aimed at CLS experimenters &mdash; both for planning runs at the
beamline and for analysing the resulting spectra afterwards. A reader new to the
app should start with {R.link("installation", "Installation")} and the
{R.link("quickstart-estimate", "Quick Start")}; the
{R.link("physics-reference", "Physics Reference")} appendix collects every
formula the program evaluates.</p>

{R.related(["installation", "launching-denis", "four-tab-tour", "general-ui",
            "physics-reference"])}
"""


def installation():
    return f"""
<h1>Installation</h1>

<p>DENIS runs on Windows, macOS, and Linux. You do <b>not</b> need to install
Python yourself &mdash; the installer uses <b>uv</b> (the Astral package
manager) to provision a private Python and all dependencies into a project-local
<code>.venv</code>. Python 3.10 or newer is required (uv handles this for you).</p>

<h2>Recommended: the installer</h2>
<p>From the project folder, run the installer for your platform. With no
argument it shows an interactive menu; pass <code>full</code> to do everything
non-interactively.</p>

{R.code(_FULL_INSTALL, "One-shot full install")}

<p>A <b>full</b> install performs three steps:</p>
<ol>
  <li><b>Install uv</b> (via the official Astral installer) if it is not already
      on your <code>PATH</code>.</li>
  <li><b>uv sync</b> &mdash; resolve and install Python plus every dependency
      into <code>.venv</code>.</li>
  <li><b>Create a desktop shortcut</b> that launches the app.</li>
</ol>

<p>The menu also offers these as separate actions (<code>uv</code> /
<code>shortcut</code>), e.g. to (re)create just the shortcut once
<code>.venv</code> exists.</p>

{R.warning("If uv was just installed, your current terminal may not see it yet "
           "&mdash; uv installs to <code>" + _UV_PATH_WIN + "</code> "
           "(Windows) or <code>" + _UV_PATH_NIX + "</code> (macOS/Linux). Open a "
           "<b>new</b> terminal and re-run the installer. See "
           + R.link("troubleshooting-common", "Troubleshooting") + ".")}

<h2>Manual install</h2>
<p>If you prefer to drive uv yourself:</p>
{R.code(_MANUAL_INSTALL, "Manual setup with uv")}

<h2>The desktop shortcut</h2>
<p>The shortcut points at <code>gui.py</code> with the project's interpreter:</p>
<ul>
  <li><b>Windows</b> &mdash; uses <code>pythonw.exe</code> so no console window
      appears (falls back to <code>python.exe</code> if <code>pythonw</code> is
      missing). Icon: <code>icons/denis.ico</code>.</li>
  <li><b>macOS</b> &mdash; a <code>DENIS.command</code> launcher on the Desktop.</li>
  <li><b>Linux</b> &mdash; a <code>DENIS.desktop</code> entry.</li>
</ul>

{R.screenshot("install_menu.png",
              "The installer's interactive menu (run install.bat / ./install.sh "
              "with no argument).")}

<h2>Dependencies</h2>
<p>Resolved automatically by <code>uv sync</code>: PySide6, matplotlib, numpy,
pandas, PyYAML, tabulate, satlas2, asdf, periodictable, h5py, lmfit, emcee,
clstools, and pymc. Most come from PyPI; <b>clstools</b> is pulled from its
upstream git repository
({R.extlink("https://github.com/andry3vi/cls_tools", "github.com/andry3vi/cls_tools")}).</p>

{R.related(["launching-denis", "troubleshooting-common", "citation-license"])}
"""


def launching_denis():
    return f"""
<h1>Launching DENIS</h1>

<p>There are two ways to start the graphical application:</p>
<ul>
  <li><b>Desktop shortcut</b> &mdash; double-click the <i>DENIS</i> shortcut
      created during {R.link("installation", "installation")}.</li>
  <li><b>Terminal</b> &mdash; from the project root, run:</li>
</ul>

{R.code("uv run python gui.py", "Launch from a terminal")}

<p>After a brief splash screen the {R.link("four-tab-tour", "main window")}
opens with the four tabs across the top, the Estimate tab selected by
default.</p>

<h2>Headless estimation (command line)</h2>
<p>The beam-time estimator also has a command-line entry point that reads the
same YAML configuration the {R.link("tab-estimate", "Estimate")} tab saves,
runs the calculation without the GUI, and writes the report and plots to disk:</p>

{R.code("python cls_estimate.py <config.yaml> [--output-dir DIR] [--no-plot]",
        "cls_estimate.py — headless estimator")}

<p>This is handy for scripting many estimates or running on a machine without a
display. See {R.link("estimation-yaml-schema", "the estimation YAML schema")}
for the configuration format.</p>

{R.note("On Windows, launching from the shortcut uses <code>pythonw.exe</code> "
        "so no console window appears. If a black console window pops up and "
        "closes, see " + R.link("troubleshooting-common", "Troubleshooting") + ".")}

{R.related(["installation", "four-tab-tour", "estimation-yaml-schema",
            "tab-estimate"])}
"""


def four_tab_tour():
    return f"""
<h1>The Four-Tab Tour</h1>

<p>The whole application lives in one window. A tab bar across the top switches
between the four stages of a measurement; a {R.link("menu-bar", "menu bar")}
(File, Edit, View, Run, Tools, Help) above it holds global actions such as
saving/loading configurations, undo/redo, zoom, and the standalone
{R.link("tools", "calculator tools")}.</p>

{R.screenshot("main_window.png",
              "The DENIS main window after launch, with the four top-level tabs "
              "and the Estimate tab selected.")}

<h2>Estimate</h2>
<p>{R.link("tab-estimate", "Estimate")} is where you go <i>before</i> taking
beam. Define the element, transition, beam/laser parameters and a list of
isotopes, then run to get per-peak signal rates and a predicted total run time,
plus a simulated hyperfine spectrum. One surface, three columns: parameters,
isotopes, and run/plots/log.</p>

<h2>Pre-Analysis</h2>
<p>{R.link("tab-preanalysis", "Pre-Analysis")} is the data viewer. Load one or
more ASDF runs, gate on time-of-flight, bin the spectrum, flip the x-axis
between scanning voltage and Doppler-shifted frequency, and overlay trial HFS
models &mdash; everything you need to sanity-check data and seed a fit.</p>

<h2>Analysis</h2>
<p>{R.link("tab-analysis", "Analysis")} is where spectra are fitted. Each fit is
a project built from a left-to-right block pipeline &mdash;
{R.link("an-source-block", "Source")} &rarr;
{R.link("an-model-block", "Model")} &rarr;
{R.link("an-fitter", "Fitter")} &rarr;
{R.link("an-output-block", "Output")}. A permanent
{R.link("an-isotope-shifts", "Isotope Shifts")} sub-tab combines centroids
across projects.</p>

<h2>Results</h2>
<p>{R.link("tab-results", "Results")} collects everything a fit produced in a
project/iteration tree: fit reports, parameter tables, and live plots you can
re-style and export.</p>

{R.related(["tab-estimate", "tab-preanalysis", "tab-analysis", "tab-results",
            "menu-bar", "general-ui"])}
"""


def quickstart_estimate():
    return f"""
<h1>Quick Start: Estimate Beam Time</h1>

<p>This first walkthrough plans a measurement. No example data is needed &mdash;
you supply atomic and nuclear inputs and DENIS predicts the run time and a
simulated spectrum.</p>

<h2>Steps</h2>
<ol>
  <li>Open the <b>{R.link("tab-estimate", "Estimate")}</b> tab. It is a
      three-column layout: parameter groups on the left, the isotope list in the
      middle, run options, plots, and the log on the right.</li>
  <li>Fill the left-column groups: <b>{R.link("est-element-transition",
      "Element &amp; Transition")}</b> (symbol/Z, lower and upper levels in
      cm{R.code_ref('⁻¹')}, and their J), <b>{R.link("est-beam", "Beam")}</b>
      (voltage in kV, charge, geometry), <b>{R.link("est-laser", "Laser")}</b>
      (harmonic and setpoint), and <b>{R.link("est-linewidth-scan",
      "Linewidth &amp; Scan")}</b> (FWHM, scan range, step, dwell).</li>
  <li>In the middle column, click <b>+ Add Isotope</b> for each isotope and fill
      its mass number A, spin I, moments &mu; and Q, isotope shift, yield, and
      efficiency. Tick one isotope's <b>Reference</b> box &mdash; its A/B
      constants scale the others.</li>
  <li>Make sure the <b>{R.link("est-statistics", "Statistics (Timing Mode)")}</b>
      group is enabled to get timing (leave it off for a spectrum-only
      simulation), and set the required &sigma;.</li>
  <li>Click <b>Run Estimation</b> (right column). The log streams per-peak
      signal rates, the on-peak time for the requested &sigma;, the number of
      sweeps to cover the scan range, and a grand-total run time.</li>
  <li>The plot panel above the log shows the simulated voltage spectrum for
      each isotope over the scan range; use the <b>View</b> drop-down for the
      combined overview or the sortable peak list, and the expand toggle to
      view the plots full-tab.</li>
</ol>

{R.screenshot("qs_estimate_params.png",
              "Estimate tab: parameter groups filled in, one isotope panel "
              "expanded, marked as Reference.")}

{R.screenshot("qs_estimate_output.png",
              "After Run Estimation: the log report (right column) and the "
              "simulated spectrum on the Plots sub-tab.")}

{R.tip("The on-peak measurement time scales as t = &sigma;&sup2;(S+B)/S&sup2; "
       "in the signal rate S and background B &mdash; halving the signal "
       "quadruples the time. See " + R.link("timing-statistics",
       "Timing / Statistics") + " for the full timing model.")}

{R.related(["tab-estimate", "est-statistics", "timing-statistics",
            "quickstart-preanalysis"])}
"""


def quickstart_preanalysis():
    return f"""
<h1>Quick Start: Pre-Analyse a Spectrum</h1>

<p>Once you have a measured run, the {R.link("tab-preanalysis", "Pre-Analysis")}
tab lets you gate, bin, and inspect it &mdash; and overlay a trial model to seed
the fit.</p>

<h2>Steps</h2>
<ol>
  <li>Open <b>Pre-Analysis</b> and click <b>Open...</b> in the left file list.
      Select an IGISOL-style ASDF run. It appears in the list; click it to load
      its spectra into the centre panels.</li>
  <li>On the <b>Spectrum</b> tab, drag the <b>ToF gate</b> to isolate the ion
      bunch from background. After clicking inside a numeric box, the keyboard
      arrows step the digit under the cursor &mdash; coarse digit for big jumps,
      fine digit for small ones. The spectrum re-bins live as the gate
      moves.</li>
  <li>To see the spectrum in the rest frame, change the x-axis mode from
      <b>Voltage</b> to <b>Frequency</b> (or Wavenumber). DENIS applies the
      relativistic {R.link("physics-doppler", "Doppler shift")} per event.</li>
  <li>Click <b>+ Add Model</b> (right panel) to overlay a trial HFS model: set
      the centroid, linewidth, and A/B constants and adjust until the model line
      positions sit near the visible peaks. This gives a clean starting point
      for the fitter.</li>
</ol>

{R.screenshot("qs_preanalysis.png",
              "Pre-Analysis with a run loaded: ToF gate (top), binned spectrum "
              "with an HFS model overlaid, file list and model editor on the "
              "sides.")}

{R.related(["tab-preanalysis", "pa-tof-time-gate", "pa-xaxis-modes",
            "pa-hfs-overlay", "physics-doppler", "quickstart-analysis"])}
"""


def quickstart_analysis():
    return f"""
<h1>Quick Start: Build and Run an Analysis Project</h1>

<p>The {R.link("tab-analysis", "Analysis")} tab fits spectra. Each fit is a
project assembled from a pipeline of blocks.</p>

<h2>Steps</h2>
<ol>
  <li>Open <b>Analysis</b> and click <b>+ Create Analysis Project</b>. Choose
      <b>Sample Project</b> for a standard fit (or
      {R.link("an-projects", "Reference Project")} for reference-isotope scans).
      The project opens in its own sub-tab.</li>
  <li>Build the pipeline left to right (drag a block's handle to reorder, use
      <b>+ Add Block</b> for more):
    <ul>
      <li><b>{R.link("an-source-block", "Source")}</b> &mdash; one per run file;
          set the binning axis, range, and ToF gate (reuse the values from
          Pre-Analysis).</li>
      <li><b>{R.link("an-model-block", "Model")}</b> &mdash; typically one per
          isotopomer; set initial A, B, amplitudes, FWHMs, centroid, and set
          the <b>Mode</b> of the parameters that should float to
          <b>Free</b>.</li>
      <li><b>{R.link("an-fitter", "Fitter")}</b> &mdash; pick the optimizer
          (leastsq, least_squares, slsqp, emcee, &hellip;), the statistic
          (chi-square / Poisson LLH / Gaussian LLH), and the y-error model.</li>
      <li><b>{R.link("an-output-block", "Output")}</b> &mdash; enable the
          reports and plots you want (data + fit, residuals, walk/correlation
          plots for MCMC, &chi;&sup2; map).</li>
    </ul>
  </li>
  <li>If the initial guess is poor, click <b>Find Parameters</b> on the Fitter
      block to open the {R.link("an-autofitter", "Auto-Fitter")} &mdash; a
      multi-start parallel <code>leastsq</code> sweep that lets you compare
      candidates and accept one back into the model in a single undoable
      step.</li>
  <li>Click <b>Run Fit</b> on the Fitter block. The status bar reports progress;
      on completion the results are pushed to the
      {R.link("tab-results", "Results")} tab.</li>
</ol>

{R.screenshot("qs_analysis_pipeline.png",
              "An Analysis project showing the Source -> Model -> Fitter -> "
              "Output block pipeline.")}

{R.related(["tab-analysis", "an-pipeline", "an-fitter", "an-autofitter",
            "physics-fit-objectives", "quickstart-results"])}
"""


def quickstart_results():
    return f"""
<h1>Quick Start: Read Results and Compute Isotope Shifts</h1>

<p>After a fit, the {R.link("tab-results", "Results")} tab is where you read,
re-style, and export everything it produced &mdash; and the
{R.link("an-isotope-shifts", "Isotope Shifts")} tab combines fitted centroids.</p>

<h2>Reading a fit</h2>
<ol>
  <li>Open <b>Results</b>. The left tree lists each project, and under it one
      folder per iteration (<code>iter_001</code>, &hellip;) containing the
      <b>Fit Report</b>, <b>Parameters</b>, <b>Metadata</b>, and plot items.</li>
  <li>Click an item to view it: text reports in a monospace box, parameter
      tables as a grid, and plots as live matplotlib figures (including the
      reduced &chi;&sup2; fit-quality summary).</li>
  <li>Click <b>Edit Plot</b> to adjust styling (fonts, colours, limits); the
      changes are saved next to the plot data as a <code>.style.json</code> so
      they survive reloading. Use <b>Export</b> (or <b>Export All</b>) to save
      figures as PNG, PDF, or SVG.</li>
</ol>

<h2>Computing isotope shifts</h2>
<ol>
  <li>Back in <b>Analysis</b>, open the permanent <b>Isotope Shifts</b>
      sub-tab.</li>
  <li>Click <b>Refresh from Projects</b> to populate one row per fitted project,
      then designate a reference isotope.</li>
  <li>Click <b>Compute Shifts</b>. The right pane shows the combined plot, the
      shift table (with propagated statistical and systematic errors), and a
      log of how each shift was computed.</li>
  <li><b>Export Table CSV</b> writes the table to disk; <b>Send to Results</b>
      pushes the figure to the Results tab.</li>
</ol>

{R.screenshot("qs_results.png",
              "Results tab after a completed fit: the project/iteration tree on "
              "the left, a data+fit+residuals plot on the right.")}

{R.related(["tab-results", "res-tree", "res-export", "an-isotope-shifts",
            "physics-isotope-shift"])}
"""


PAGES = {
    "introduction": introduction,
    "installation": installation,
    "launching-denis": launching_denis,
    "four-tab-tour": four_tab_tour,
    "quickstart-estimate": quickstart_estimate,
    "quickstart-preanalysis": quickstart_preanalysis,
    "quickstart-analysis": quickstart_analysis,
    "quickstart-results": quickstart_results,
}
