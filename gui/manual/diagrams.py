"""Schematic diagrams for the manual, drawn with matplotlib and cached to PNG.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds the manual's schematic figures (the four-tab workflow strip, the
collinear/anti-collinear Doppler geometry, and the on-peak timing curve) and
caches each as a PNG. Diagrams are theme-aware (light strokes/text on a
transparent background) and keyed by name + DIAGRAM_VERSION, so bumping the
version invalidates stale cached images after a drawing is edited. Uses the
object-oriented Figure API (no pyplot) to avoid touching global matplotlib state.

Depends on: standard library and third-party packages only (numpy, matplotlib).
"""

import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")

DIAGRAM_VERSION = 2

FG = "#e6e6e6"        # primary text/stroke
ACCENT = "#4ea1d5"    # blue accent
MUTED = "#9aa0a6"     # secondary text
FILL = "#3a3f44"      # box fill


def _path(name):
    return os.path.join(_CACHE_DIR, f"diag_{name}_v{DIAGRAM_VERSION}.png")


def _save(fig, name):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    p = _path(name)
    FigureCanvasAgg(fig)
    fig.savefig(p, dpi=160, transparent=True, bbox_inches="tight",
                pad_inches=0.05)
    return p


def _box(ax, x, y, w, h, title, sub=None, edge=ACCENT):
    # clip_on=False so artists are never cropped by the axes view limits;
    # bbox_inches='tight' then captures their full extent.
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=edge, facecolor=FILL, clip_on=False))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title,
            ha="center", va="center", color=FG, fontsize=11, fontweight="bold",
            clip_on=False)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                color=MUTED, fontsize=8.5, clip_on=False)


def _arrow(ax, x0, y0, x1, y1, color=FG, style="-|>", lw=1.6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2, clip_on=False))


def four_tab_map():
    """Estimate -> Pre-Analysis -> Analysis -> Results workflow strip."""
    name = "four-tab-map"
    if os.path.exists(_path(name)):
        return _path(name)
    fig = Figure(figsize=(8.7, 1.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 87); ax.set_ylim(0, 17)
    tabs = [("Estimate", "plan beam time"),
            ("Pre-Analysis", "inspect spectra"),
            ("Analysis", "fit HFS"),
            ("Results", "view & export")]
    w, h, gap, x = 17, 9, 5.5, 1.5
    for i, (t, s) in enumerate(tabs):
        _box(ax, x, 4, w, h, t, s)
        if i < len(tabs) - 1:
            _arrow(ax, x + w, 8.5, x + w + gap, 8.5)
        x += w + gap
    return _save(fig, name)


def doppler_geometry():
    """Collinear vs anti-collinear laser/ion-beam geometry schematic."""
    name = "doppler-geometry"
    if os.path.exists(_path(name)):
        return _path(name)
    fig = Figure(figsize=(8.4, 3.1))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 84); ax.set_ylim(0, 31)

    def row(y, label, laser_from_right):
        # beam axis
        ax.plot([6, 60], [y, y], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                clip_on=False)
        # ion moving right (velocity v)
        ax.add_patch(Circle((20, y), 1.6, facecolor=ACCENT, edgecolor=FG,
                            lw=1.0, clip_on=False))
        _arrow(ax, 22, y, 30, y, color=ACCENT)
        ax.text(26, y + 2.4, r"ion  $\vec{v}$", color=ACCENT, fontsize=9,
                ha="center", clip_on=False)
        # laser photon
        if laser_from_right:
            _arrow(ax, 58, y, 40, y, color="#e06c5a")  # head-on
            ax.text(50, y + 2.4, r"laser $\nu_{\mathrm{lab}}$", color="#e06c5a",
                    fontsize=9, ha="center", clip_on=False)
        else:
            _arrow(ax, 8, y - 3.0, 38, y - 3.0, color="#e0b85a")
            ax.text(24, y - 5.2, r"laser $\nu_{\mathrm{lab}}$", color="#e0b85a",
                    fontsize=9, ha="center", clip_on=False)
        ax.text(63, y, label, color=FG, fontsize=9.5, va="center",
                ha="left", fontweight="bold", clip_on=False)

    row(22, "anti-collinear\n(head-on)", laser_from_right=True)
    row(8, "collinear\n(co-propagating)", laser_from_right=False)
    return _save(fig, name)


def timing_curve():
    """On-peak integration time t = sigma^2 (S+B) / S^2 vs signal rate S, for a
    few background rates B (sigma = 3, the default required significance)."""
    name = "timing-curve"
    if os.path.exists(_path(name)):
        return _path(name)
    fig = Figure(figsize=(6.2, 3.4))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.13, 0.17, 0.84, 0.74])
    ax.set_facecolor("none")
    sigma = 3.0
    S = np.linspace(2.0, 200.0, 400)
    for B, color, lbl in [(0, ACCENT, "B = 0"),
                          (10, "#e0b85a", "B = 10 Hz"),
                          (50, "#e06c5a", "B = 50 Hz")]:
        ax.plot(S, sigma**2 * (S + B) / S**2, color=color, lw=1.8, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("signal rate  S  (Hz)", color=FG, fontsize=10)
    ax.set_ylabel("on-peak time  t  (s)", color=FG, fontsize=10)
    ax.set_title(r"$t = \sigma^2 (S+B)/S^2\quad(\sigma=3)$", color=FG, fontsize=11)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    leg = ax.legend(facecolor="none", edgecolor=MUTED, labelcolor=FG, fontsize=8)
    leg.get_frame().set_alpha(0.0)
    return _save(fig, name)


# Registry so content/render can resolve a diagram by string name.
DIAGRAMS = {
    "four-tab-map": four_tab_map,
    "doppler-geometry": doppler_geometry,
    "timing-curve": timing_curve,
}


def render(name):
    """Render a registered diagram by name; return the cached PNG path."""
    fn = DIAGRAMS.get(name)
    if fn is None:
        raise KeyError(f"unknown diagram: {name!r}")
    return fn()
