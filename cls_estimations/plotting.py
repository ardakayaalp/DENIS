"""Hyperfine-spectrum plotting helpers and figure styling.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Renders predicted HFS spectra for the run-time estimation: per-isotope
subplots and a combined overview, saved as PDFs. Defines the LaTeX
serif plot style and selectable colour palettes, and uses the
thread-safe object-oriented matplotlib API so figures can be built
from a worker thread.

Depends on: standard library and third-party packages only (matplotlib,
numpy).
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

# Use the OO matplotlib API (Figure + FigureCanvasAgg) for the figures
# below instead of plt.subplots(). pyplot routes through the GUI backend
# selected at startup (QtAgg in the desktop app), and macOS Cocoa
# refuses to be touched off the main thread -- which is exactly what
# happens here, since the estimation runs in a worker QThread. The OO
# path bypasses pyplot's figure manager entirely and is thread-safe.
# plt.rcParams.update() below is still safe; rcParams is a global dict.

# ── Classic LaTeX serif style ───────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["CMU Serif", "Latin Modern Roman", "DejaVu Serif",
                           "Times New Roman", "Times"],
    "mathtext.fontset":   "cm",
    "axes.unicode_minus": False,
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "xtick.minor.size":   3,
    "ytick.minor.size":   3,
    "xtick.top":          True,
    "ytick.right":        True,
    "legend.fontsize":    9,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,       # square corners
})

# ── Color palettes ──────────────────────────────────────────────────
PALETTES = {
    "default": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ],
    "pastel": [
        "#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b",
        "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf",
    ],
    "vibrant": [
        "#e60049", "#0bb4ff", "#50e991", "#e6d800",
        "#9b19f5", "#ffa300", "#dc0ab4", "#00bfa0",
    ],
    "muted": [
        "#4878d0", "#ee854a", "#6acc64", "#d65f5f",
        "#956cb4", "#8c613c", "#dc7ec0", "#797979",
    ],
    "high_contrast": [
        "#023eff", "#ff7c00", "#1ac938", "#e8000b",
        "#8b2be2", "#9f4800", "#f14cc1", "#a3a3a3",
    ],
}

_active_palette = "default"


def set_palette(name):
    """Select the active color palette by name."""
    global _active_palette
    if name not in PALETTES:
        raise ValueError(
            f"Unknown palette '{name}'. Choose from: {', '.join(PALETTES)}"
        )
    _active_palette = name


def _get_color(index):
    """Return the colour at *index* in the active palette (wraps around)."""
    pal = PALETTES[_active_palette]
    return pal[index % len(pal)]


# ── Plotting helpers ────────────────────────────────────────────────

def plot_isotope_spectrum(ax, dV_array, intensity_array, label,
                          measured_peak_dVs=None, color=None, alpha=1.0,
                          linestyle="-"):
    """Plot a single HFS spectrum on given axes."""
    ax.plot(dV_array, intensity_array, label=label,
            color=color, alpha=alpha, linestyle=linestyle, linewidth=1.0)
    if measured_peak_dVs is not None:
        for v in measured_peak_dVs:
            ax.axvline(v, color=color or "red", linestyle="--",
                       alpha=0.45 * alpha, linewidth=0.7)
    ax.set_ylabel("Normalised intensity")


def plot_all_cases(all_results, output_dir, file_tag="", config_name=""):
    """One subplot per isotope/isomer case, saved as hfs_spectra<tag>.pdf."""
    import os

    n = len(all_results)
    if n == 0:
        return

    subplot_entries = []
    for res in all_results:
        subplot_entries.append(("gs", res))
        if "isomer_label" in res and not res.get("plot_with_gs", True):
            subplot_entries.append(("isomer_only", res))

    n_plots = len(subplot_entries)
    fig = Figure(figsize=(10, 3.5 * n_plots))
    FigureCanvasAgg(fig)
    axes = [fig.add_subplot(n_plots, 1, i + 1) for i in range(n_plots)]

    color_idx = 0
    plot_idx = 0
    for entry_type, res in subplot_entries:
        ax = axes[plot_idx]
        c_gs = _get_color(color_idx)

        if entry_type == "gs":
            plot_isotope_spectrum(
                ax, res["dV_array"], res["intensity_array"],
                res["label"], res.get("measured_peak_dVs"),
                color=c_gs,
            )
            if "isomer_label" in res and res.get("plot_with_gs", True):
                c_iso = _get_color(color_idx + 1)
                plot_isotope_spectrum(
                    ax, res["isomer_dV"], res["isomer_intensity"],
                    res["isomer_label"], res.get("isomer_measured_dVs"),
                    color=c_iso, alpha=0.5,
                )
        else:  # isomer_only
            c_iso = _get_color(color_idx + 1)
            plot_isotope_spectrum(
                ax, res["isomer_dV"], res["isomer_intensity"],
                res["isomer_label"], res.get("isomer_measured_dVs"),
                color=c_iso,
            )

        ax.set_xlabel("Voltage offset $\\Delta V$ (V)")
        ax.legend()
        plot_idx += 1
        color_idx += 2

    fig.tight_layout()
    prefix = f"{config_name}_" if config_name else ""
    path = os.path.join(output_dir, f"{prefix}hfs_spectra{file_tag}.pdf")
    fig.savefig(path)
    return path


def plot_combined_overview(all_results, output_dir, file_tag="", config_name=""):
    """All isotopes/isomers overlaid on one plot, saved as overview.pdf."""
    import os

    fig = Figure(figsize=(12, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    color_idx = 0

    for res in all_results:
        c = _get_color(color_idx)
        ax.plot(res["dV_array"], res["intensity_array"],
                color=c, label=res["label"], linewidth=1.0)
        if res.get("measured_peak_dVs"):
            for v in res["measured_peak_dVs"]:
                ax.axvline(v, color=c, linestyle="--", alpha=0.35, linewidth=0.7)

        if "isomer_label" in res:
            c2 = _get_color(color_idx + 1)
            ax.plot(res["isomer_dV"], res["isomer_intensity"],
                    color=c2, linestyle="-", alpha=0.5,
                    label=res["isomer_label"], linewidth=1.0)
            if res.get("isomer_measured_dVs"):
                for v in res["isomer_measured_dVs"]:
                    ax.axvline(v, color=c2, linestyle="--", alpha=0.25, linewidth=0.7)

        color_idx += 2

    ax.set_xlabel("Voltage offset $\\Delta V$ (V)")
    ax.set_ylabel("Normalised intensity")
    ax.legend()
    fig.tight_layout()
    prefix = f"{config_name}_" if config_name else ""
    path = os.path.join(output_dir, f"{prefix}overview{file_tag}.pdf")
    fig.savefig(path)
    return path
