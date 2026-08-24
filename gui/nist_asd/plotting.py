"""Matplotlib renderers for the NIST ASD browser.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Three axes-level renderers (the caller owns the figure/canvas):

- :func:`plot_levels` — Grotrian-style diagram: one column per term,
  horizontal level bars, optional highlights;
- :func:`plot_lines_stick` — stick spectrum, wavelength vs log10(Aki),
  allowed vs forbidden colored;
- :func:`plot_scheme` — one ranked scheme: involved levels, upward
  excitation arrows (λ + Aki labels), dashed downward detection arrow
  (λ + BR%).

White publication-style figures per app convention; rcParams (incl.
minor ticks) come from the app's plot defaults.

Depends on: gui.nist_asd.data; matplotlib, numpy, pandas.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from gui.nist_asd import data as nd


def _fade(color, strength: float, floor: float = 0.25):
    """Pre-blend ``color`` toward white by channel strength instead of
    using alpha: translucent FancyArrowPatches double-draw where the
    shaft meets the head triangle, and the overlap shows through.
    An OPAQUE faded color has the same visual weight without the
    artifact. ``strength`` in [0, 1]; ``floor`` = weakest visibility."""
    a = floor + (1.0 - floor) * min(max(float(strength), 0.0), 1.0)
    r, g, b = to_rgb(color)
    return (1 - a * (1 - r), 1 - a * (1 - g), 1 - a * (1 - b))

_ALLOWED_COLOR = "#1f77b4"
_FORBIDDEN_COLOR = "#d62728"
_STEP_COLORS = ["#e6702e", "#c02942", "#7b3294"]   # pump→probe shades
_DETECT_COLOR = "#9467bd"

# Sharp, slim triangular arrow head (used by the decay and scheme
# diagrams; the default "-|>" head is stubby).
_ARROWSTYLE = "-|>,head_width=0.32,head_length=0.95"
_ARROW_MUTATION = 16


def _config_inner(config) -> str | None:
    """Mathtext body (no $) for a NIST configuration, or None."""
    if not isinstance(config, str) or not config.strip() \
            or config == "nan":
        return None

    def _parent(m):
        inner = m.group(1)
        inner = re.sub(r"<([^>]+)>", r"_{\1}", inner)
        inner = re.sub(r"^(\d+)", r"^{\1}", inner)
        # Odd parity: textbook superscript "o", not NIST's ASCII "*".
        inner = inner.replace("*", "^{o}")
        return "(" + inner + ")"

    s = re.sub(r"\(([^)]*)\)", _parent, config.strip())
    s = re.sub(r"(\d+[spdfgh])(\d+)", r"\1^{\2}", s)
    # jK-coupling K values outside parentheses: '...6s.<1/2>' → a
    # subscript rather than a literal <1/2>.
    s = re.sub(r"\.?<([^>]+)>", r"_{\1}", s)
    return s.replace(".", "\\,")


def _term_inner(term) -> str | None:
    """Mathtext body (no $) for a NIST term symbol, or None."""
    if not isinstance(term, str) or not term.strip() or term == "nan":
        return None
    t = term.strip()
    prefix = ""
    m = re.match(r"^([a-z])\s+(.*)$", t)
    if m:
        prefix, t = m.group(1) + "\\;", m.group(2)
    t = re.sub(r"^(\d+)", r"^{\1}", t)
    # Odd parity: textbook superscript "o", not NIST's ASCII "*".
    return prefix + t.replace("*", "^{o}")


def format_config_label(config: str) -> str:
    """NIST configuration → textbook mathtext: '3s2.3p2' →
    ``$3s^{2}3p^{2}$``; jj parents like '(2P*<3/2>)' become
    superscripted terms with a J subscript."""
    inner = _config_inner(config)
    return f"${inner}$" if inner else "?"


def format_term_label(term: str) -> str:
    """NIST term → textbook mathtext: '3P*' → ``$^{3}P^{*}$``,
    'a 3P*' keeps its letter prefix, '2[3/2]' → ``$^{2}[3/2]$``."""
    inner = _term_inner(term)
    return f"${inner}$" if inner else ""


def format_level_label(conf, term, j=None) -> str:
    """Full textbook level label: config + term with a J subscript,
    e.g. ``$3s^{2}3p^{2}\\;^{3}P_{2}$``."""
    ci = _config_inner(conf)
    ti = _term_inner(term)
    parts = []
    if ci:
        parts.append(ci)
    if ti:
        j = str(j).strip() if j is not None else ""
        parts.append(ti + (f"_{{{j}}}" if j and j != "nan" else ""))
    if not parts:
        return ""
    return "$" + "\\;".join(parts) + "$"


def dense_ticks(ax, x: bool = True, y: bool = True) -> None:
    """Adaptive, readable tick density: MaxNLocator majors + auto
    minors. The locators recompute on every zoom/pan, so the axis
    stays readable at any magnification."""
    if x:
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if y:
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=14, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def level_lifetimes(lines_df: pd.DataFrame) -> dict:
    """{level energy (rounded 2) → radiative lifetime τ = 1/ΣAki}."""
    t = nd.transitions_only(lines_df)
    if t is None or t.empty or "Aki(s^-1)" not in t.columns:
        return {}
    t = t.dropna(subset=["Aki(s^-1)"])
    t = t[t["Aki(s^-1)"] > 0]
    if t.empty:
        return {}
    sums = t.groupby(t["Ek(cm-1)"].round(2))["Aki(s^-1)"].sum()
    return {float(e): 1.0 / float(a) for e, a in sums.items()}


def _lifetime_alphas(energies, taus: dict) -> list:
    """Opacity per level: longer-lived → denser. Levels with no known
    decay (ground state, unknowns) count as effectively infinite and
    get full opacity; the rest map log τ onto [0.30, 1.0]."""
    logt = []
    for e in energies:
        tau = taus.get(round(float(e), 2))
        logt.append(None if tau is None else np.log10(tau))
    finite = [v for v in logt if v is not None]
    if not finite:
        return [1.0] * len(energies)
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    return [1.0 if v is None
            else 0.30 + 0.70 * (v - lo) / span
            for v in logt]


def _config_columns(levels_df: pd.DataFrame) -> list:
    """Unique configurations ordered by their lowest level energy."""
    cols = []
    for conf, grp in levels_df.groupby(
            levels_df["Configuration"].fillna("")):
        cols.append((float(grp["Level (cm-1)"].min()), conf or "?"))
    return [c for _, c in sorted(cols)]


def plot_levels(ax, levels_df: pd.DataFrame, highlight=None,
                max_configs: int = 12,
                lines_df: pd.DataFrame | None = None) -> int:
    """Level diagram: one column per CONFIGURATION with textbook
    mathtext labels; bars carry their term (+J) when few enough to
    stay readable. When ``lines_df`` is given, each level's OPACITY
    encodes its radiative lifetime (longer-lived → denser) and every
    bar carries a ``_nist_info`` payload for hover tooltips. Returns
    the number of levels drawn."""
    ax.clear()
    if levels_df is None or levels_df.empty:
        ax.text(0.5, 0.5, "No levels to plot", ha="center",
                va="center", transform=ax.transAxes, color="gray")
        ax.set_xticks([])
        return 0
    df = levels_df.dropna(subset=["Level (cm-1)"])
    configs = _config_columns(df)
    if len(configs) > max_configs:
        keep = set(configs[:max_configs])
        df = df[df["Configuration"].fillna("?").isin(keep)]
        configs = configs[:max_configs]
    xpos = {c: i for i, c in enumerate(configs)}
    hl = {round(float(h), 2) for h in (highlight or [])}
    annotate_terms = len(df) <= 40
    taus = level_lifetimes(lines_df) if lines_df is not None else {}
    energies = df["Level (cm-1)"].tolist()
    alphas = (_lifetime_alphas(energies, taus) if taus
              else [1.0] * len(energies))
    n = 0
    for (_, row), alpha in zip(df.iterrows(), alphas):
        conf = row.get("Configuration") or "?"
        x = xpos.get(conf)
        if x is None:
            continue
        e = float(row["Level (cm-1)"])
        is_hl = round(e, 2) in hl
        color = "#d62728" if is_hl else "#333333"
        (line,) = ax.plot([x - 0.38, x + 0.38], [e, e], color=color,
                          lw=2.6 if is_hl else 1.8,
                          alpha=1.0 if is_hl else alpha,
                          solid_capstyle="butt")
        # Hover payload (shown as a tooltip by the browser window).
        tau = taus.get(round(e, 2))
        unc = row.get("Uncertainty (cm-1)")
        info = [f"E = {e:,.3f} cm⁻¹"]
        if pd.notna(unc) and str(unc) != "":
            info.append(f"unc = {unc} cm⁻¹")
        info.append(f"{row.get('Configuration', '')}  "
                    f"{row.get('Term', '')}  J={row.get('J', '')}")
        info.append("τ = ∞ / unknown (no tabulated decay)"
                    if tau is None else f"τ = {tau:.3e} s")
        line._nist_info = "\n".join(info)
        if annotate_terms:
            term = format_term_label(str(row.get("Term") or ""))
            j = str(row.get("J") or "")
            label = term + (f"$_{{{j}}}$" if term and j else "")
            if label:
                ax.text(x + 0.42, e, label, fontsize=9,
                        va="center", ha="left", color="#555555")
        n += 1
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([format_config_label(c) for c in configs],
                       rotation=35, ha="right", fontsize=12)
    ax.set_ylabel("Energy (cm$^{-1}$)")
    ax.set_xlim(-0.7, len(configs) - 0.2 + (0.7 if annotate_terms
                                            else 0.0))
    ax.margins(y=0.05)
    ax.grid(True, axis="y", alpha=0.25)
    dense_ticks(ax, x=False, y=True)

    # Zoom-adaptive bar thickness: zooming into a narrow energy band
    # thickens the visible bars so they stay prominent. Registered on
    # the axes (cla() resets it, so no accumulation across redraws).
    level_lines = [l for l in ax.lines
                   if getattr(l, "_nist_info", None)]
    for l in level_lines:
        l._base_lw = l.get_linewidth()
    y0, y1 = ax.get_ylim()
    full_span = abs(y1 - y0) or 1.0

    def _on_ylim(ax_):
        lo, hi = ax_.get_ylim()
        vis = abs(hi - lo) or 1.0
        scale = min(max(full_span / vis, 1.0) ** 0.35, 2.6)
        for l in level_lines:
            l.set_linewidth(l._base_lw * scale)

    ax.callbacks.connect("ylim_changed", _on_ylim)
    return n


def plot_lines_stick(ax, lines_df: pd.DataFrame,
                     medium: str = "vacuum") -> int:
    """Stick spectrum of transitions with known Aki: wavelength vs
    log10(Aki). Returns the number of sticks drawn."""
    ax.clear()
    t = nd.transitions_only(lines_df) if lines_df is not None \
        else pd.DataFrame()
    t = t.dropna(subset=["Aki(s^-1)"]) if not t.empty else t
    if t.empty:
        ax.text(0.5, 0.5, "No transitions with Aki to plot",
                ha="center", va="center", transform=ax.transAxes,
                color="gray")
        ax.set_xticks([])
        return 0
    wl = t.apply(lambda r: nd.wavelength_nm(r, medium), axis=1)
    ok = np.isfinite(wl) & (t["Aki(s^-1)"] > 0)
    t, wl = t[ok], wl[ok]
    logA = np.log10(t["Aki(s^-1)"].astype(float))
    allowed = t["Type"].fillna("E1").isin(["E1", ""])
    for mask, color, label in (
            (allowed, _ALLOWED_COLOR, "Allowed (E1)"),
            (~allowed, _FORBIDDEN_COLOR, "Forbidden")):
        if mask.any():
            ax.vlines(wl[mask], -8, logA[mask], color=color, lw=1.0,
                      alpha=0.8, label=label)
    lo = float(np.floor(logA.min())) - 1
    ax.set_ylim(bottom=max(lo, -8))
    unit = "air" if medium == "air" else "vac"
    ax.set_xlabel(f"Wavelength, {unit} (nm)")
    ax.set_ylabel(r"log$_{10}$ A$_{ki}$ (s$^{-1}$)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    dense_ticks(ax)
    return int(len(t))


def _level_label(t: dict, side: str) -> str:
    return format_level_label(t.get(f"conf_{side}"),
                              t.get(f"term_{side}"),
                              t.get(f"J_{side}"))


def plot_scheme(ax, ranked, medium: str = "vacuum") -> None:
    """Draw one RankedScheme as an excitation/detection diagram.

    Level bars and their labels live in the SAME (data) coordinate
    system: the exact energy sits at the right end of each bar, the
    configuration/term at the left end, so nothing drifts away from
    its level. No score in the title (the table carries it)."""
    ax.clear()
    scheme = ranked.scheme
    # Involved levels: every step's endpoints + the detection floor.
    levels = {}   # energy -> label

    def _add(e, label):
        if e is None:
            return
        e = float(e)
        if e not in levels or (label and not levels[e]):
            levels[e] = label

    for step in scheme.steps:
        t = step.transition
        _add(t.get("Ei(cm-1)"), _level_label(t, "i"))
        _add(t.get("Ek(cm-1)"), _level_label(t, "k"))
    det = scheme.detection
    _add(det.get("Ei(cm-1)"), _level_label(det, "i"))

    span = max(levels) - min(levels) if len(levels) > 1 else 1.0
    # Near-degenerate levels (fine structure) would overprint their
    # labels — stagger: the lower of a close pair labels BELOW its bar.
    ordered = sorted(levels)
    below = set()
    for a, b in zip(ordered, ordered[1:]):
        if (b - a) < span * 0.05:
            below.add(a)
    for e, label in levels.items():
        ax.hlines(e, 0.05, 0.95, color="#333333", lw=1.8)
        off = -span * 0.012 if e in below else span * 0.012
        va = "top" if e in below else "bottom"
        # Exact energy hugging the bar's right end.
        ax.text(0.95, e + off, f"{e:,.1f} cm$^{{-1}}$",
                va=va, ha="right", fontsize=11, color="#222222")
        if label:
            ax.text(0.05, e + off, label, fontsize=10,
                    color="#555555", va=va, ha="left")

    # Excitation arrows, one x-slot per step
    n_steps = len(scheme.steps)
    for i, step in enumerate(scheme.steps):
        t = step.transition
        e0, e1 = float(t["Ei(cm-1)"]), float(t["Ek(cm-1)"])
        x = 0.18 + 0.42 * (i / max(n_steps, 1))
        color = _STEP_COLORS[i % len(_STEP_COLORS)]
        ax.annotate(
            "", xy=(x, e1), xytext=(x, e0),
            arrowprops=dict(arrowstyle=_ARROWSTYLE, color=color,
                            mutation_scale=_ARROW_MUTATION, lw=2.2),
        )
        wl = nd.wavelength_nm(t, medium)
        aki = t.get("Aki(s^-1)")
        label = f"{wl:.2f} nm"
        if aki:
            label += f"\nA={aki:.2e}"
        if step.laser:
            label += f"\n[{step.laser}]"
        ax.text(x + 0.015, (e0 + e1) / 2, label, fontsize=10,
                color=color, va="center")

    # Detection (fluorescence) arrow, dashed, rightmost slot
    e_top = float(det["Ek(cm-1)"])
    e_lo = float(det["Ei(cm-1)"])
    xd = 0.74
    ax.annotate(
        "", xy=(xd, e_lo), xytext=(xd, e_top),
        arrowprops=dict(arrowstyle=_ARROWSTYLE, color=_DETECT_COLOR,
                        mutation_scale=_ARROW_MUTATION,
                        lw=2.0, linestyle="--"),
    )
    wl_d = nd.wavelength_nm(det, medium)
    ax.text(xd + 0.015, (e_top + e_lo) / 2,
            f"{wl_d:.2f} nm\nBR {scheme.branching_ratio * 100:.1f}%",
            fontsize=10, color=_DETECT_COLOR, va="center")

    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.set_ylabel("Energy (cm$^{-1}$)")
    ax.margins(y=0.1)
    dense_ticks(ax, x=False, y=True)
    if ranked.issues:
        ax.set_title("⚠ possible isobaric contamination",
                     fontsize=10, loc="left", color="#b02020")


def plot_decay_channels(ax, lines_df: pd.DataFrame, level_cm1: float,
                        medium: str = "vacuum",
                        tolerance: float = 0.1) -> int:
    """Diagnostics: the selected level with its decay channels drawn
    as a level diagram — the upper level on top, each destination
    level below at its true energy, one downward arrow per channel
    with opacity and width scaled by its strength, labeled λ + BR%.
    Returns the number of channels drawn."""
    ax.clear()
    t = nd.transitions_only(lines_df) if lines_df is not None \
        else pd.DataFrame()
    if not t.empty:
        t = t[(t["Ek(cm-1)"] - float(level_cm1)).abs() < tolerance]
        t = t.dropna(subset=["Aki(s^-1)"])
        t = t[t["Aki(s^-1)"] > 0]
    if t is None or t.empty:
        ax.text(0.5, 0.5, "No decay channels with known Aki",
                ha="center", va="center", transform=ax.transAxes,
                color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        return 0
    # One channel per destination, strongest first for slot layout.
    t = t.sort_values("Aki(s^-1)", ascending=False)
    total = float(t["Aki(s^-1)"].sum())
    aki_max = float(t["Aki(s^-1)"].max())
    e_up = float(level_cm1)
    n = len(t)

    # y floor: zero when any channel actually reaches down toward the
    # ground region; when EVERY destination sits high, starting at 0
    # compresses all structure into a sliver at the top — crop below
    # the lowest destination instead.
    min_dest = float(t["Ei(cm-1)"].min())
    if min_dest <= 0.35 * e_up:
        y0 = 0.0
    else:
        y0 = max(0.0, min_dest - 0.20 * (e_up - min_dest + 1.0))
    span_guess = max(e_up - y0, 1.0)

    # Upper level: full-width bar with textbook labels.
    up_row = t.iloc[0]
    up_label = format_level_label(up_row.get("conf_k"),
                                  up_row.get("term_k"),
                                  up_row.get("J_k"))
    ax.hlines(e_up, 0.04, 0.96, color="#222222", lw=2.6)
    ax.text(0.04, e_up + span_guess * 0.015, up_label, fontsize=11,
            ha="left", va="bottom", color="#333333")
    ax.text(0.96, e_up + span_guess * 0.015,
            f"{e_up:,.1f} cm$^{{-1}}$", fontsize=11, ha="right",
            va="bottom", color="#222222")

    # Destination slots spread across the width, strongest leftmost.
    xs = np.linspace(0.14, 0.88, n) if n > 1 else np.array([0.5])
    half = min(0.40 / max(n, 1), 0.085)
    for slot, (_, row) in enumerate(t.iterrows()):
        e_lo = float(row["Ei(cm-1)"])
        aki = float(row["Aki(s^-1)"])
        br = aki / total * 100.0
        frac = aki / aki_max
        x = float(xs[slot])
        typ = str(row.get("Type") or "E1")
        base_color = _ALLOWED_COLOR if typ in ("E1", "") else \
            _FORBIDDEN_COLOR
        ax.hlines(e_lo, x - half, x + half, color="#333333", lw=1.8)
        lo_label = format_level_label(row.get("conf_i"),
                                      row.get("term_i"),
                                      row.get("J_i"))
        # Vertical (orthogonal) labels: configuration over the left
        # half, energy over the right half. Destinations close to the
        # upper level hang their labels BELOW the bar instead, so
        # nothing pokes past the upper level into the title.
        dy = span_guess * 0.012
        above = e_lo < y0 + 0.72 * span_guess
        y_lab = e_lo + dy if above else e_lo - dy
        va = "bottom" if above else "top"
        if lo_label:
            ax.text(x - half * 0.55, y_lab, lo_label,
                    fontsize=10, rotation=90, ha="center",
                    va=va, color="#555555")
        ax.text(x + half * 0.55, y_lab,
                f"{e_lo:,.1f} cm$^{{-1}}$", fontsize=10,
                rotation=90, ha="center", va=va,
                color="#333333")
        # Arrow: width + an OPAQUE white-blended fade follow the
        # channel strength — true alpha exposes the shaft/head
        # overlap inside FancyArrowPatch (the head shows the line
        # running through it).
        color = _fade(base_color, frac)
        ax.annotate(
            "", xy=(x, e_lo), xytext=(x, e_up),
            arrowprops=dict(arrowstyle=_ARROWSTYLE, color=color,
                            mutation_scale=_ARROW_MUTATION,
                            lw=1.0 + 2.6 * frac),
        )
        # λ/BR: one vertical line hanging BELOW the destination bar —
        # empty space by construction in every slot, so it cannot
        # collide with bars, arrows or level labels (short arrows made
        # mid-arrow placement collide). Near the floor there is no
        # room below → it rises ABOVE the bar instead, between the
        # config and energy columns.
        # λ/BR: vertical, centered on the arrow's midpoint (offset
        # sideways just enough to clear the widest shaft). Clamped so
        # a label longer than a SHORT arrow tucks under the upper
        # level instead of poking into the title.
        wl = nd.wavelength_nm(row, medium)
        lam_text = f"{wl:.2f} nm · BR {br:.1f}%"
        fig = ax.figure
        axes_px = max(fig.get_figheight() * fig.get_dpi() * 0.80, 1.0)
        half_len = (0.5 * len(lam_text) * 10 * 0.80
                    / axes_px * span_guess * 1.12
                    + span_guess * 0.012)
        y_c = 0.5 * (e_up + e_lo)
        y_c = min(y_c, e_up - half_len)
        y_c = max(y_c, y0 + half_len)
        ax.text(x + 0.014, y_c, lam_text, fontsize=10,
                rotation=90, ha="center", va="center",
                color=_fade(base_color, frac, floor=0.55))

    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.set_ylabel("Energy (cm$^{-1}$)")
    ax.set_ylim(y0, e_up + span_guess * 0.12)
    dense_ticks(ax, x=False, y=True)
    tau = 1.0 / total
    ax.set_title(f"Decays of {e_up:,.2f} cm$^{{-1}}$ — "
                 f"τ ≈ {tau:.3e} s", fontsize=12, loc="left")
    return int(n)


def plot_lifetimes(ax, lines_df: pd.DataFrame) -> int:
    """Diagnostics: radiative lifetime τ = 1/ΣAki of every upper level
    vs its energy (log τ). Long-lived outliers are metastable
    candidates. Returns the number of levels plotted."""
    ax.clear()
    t = nd.transitions_only(lines_df) if lines_df is not None \
        else pd.DataFrame()
    if not t.empty:
        t = t.dropna(subset=["Aki(s^-1)"])
        t = t[t["Aki(s^-1)"] > 0]
    if t is None or t.empty:
        ax.text(0.5, 0.5, "No transitions with Aki",
                ha="center", va="center", transform=ax.transAxes,
                color="gray")
        ax.set_xticks([])
        return 0
    sums = t.groupby(t["Ek(cm-1)"].round(2))["Aki(s^-1)"].sum()
    energies = sums.index.to_numpy(dtype=float)
    taus = 1.0 / sums.to_numpy(dtype=float)
    ax.scatter(energies, taus, s=48, color=_ALLOWED_COLOR, alpha=0.75,
               edgecolors="#123a5c", linewidths=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Upper level energy (cm$^{-1}$)")
    ax.set_ylabel(r"Radiative lifetime $\tau = 1/\Sigma A_{ki}$ (s)")
    ax.grid(True, alpha=0.25)
    dense_ticks(ax, y=False)   # y is logarithmic — keep log ticks
    return int(len(energies))


def plot_line_density(ax, lines_df: pd.DataFrame,
                      medium: str = "vacuum",
                      bin_nm: float = 10.0) -> int:
    """Diagnostics: number of transitions per wavelength bin — spectral
    congestion at a glance. Returns the number of lines binned."""
    ax.clear()
    t = nd.transitions_only(lines_df) if lines_df is not None \
        else pd.DataFrame()
    if t is None or t.empty:
        ax.text(0.5, 0.5, "No transitions to bin", ha="center",
                va="center", transform=ax.transAxes, color="gray")
        ax.set_xticks([])
        return 0
    wl = t.apply(lambda r: nd.wavelength_nm(r, medium), axis=1)
    wl = wl[np.isfinite(wl)]
    if wl.empty:
        ax.set_xticks([])
        return 0
    # Cap the bin COUNT: a small bin width over NIST's full IR range
    # (tens of thousands of nm) would otherwise create hundreds of
    # thousands of bar patches and freeze the UI. stairs() draws the
    # whole histogram as ONE artist.
    span = float(wl.max() - wl.min()) or 1.0
    eff_bin = max(float(bin_nm), span / 2000.0)
    lo = np.floor(wl.min() / eff_bin) * eff_bin
    hi = np.ceil(wl.max() / eff_bin) * eff_bin
    edges = np.arange(lo, hi + eff_bin, eff_bin)
    counts, edges = np.histogram(wl, bins=edges)
    # Bars with visible edges so single bins stay distinguishable —
    # safe now that the bin COUNT is capped at 2000 (the old freeze
    # came from an uncapped count, not from bar patches per se).
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=eff_bin, color=_ALLOWED_COLOR,
           alpha=0.85, edgecolor="#123a5c", linewidth=0.5)
    unit = "air" if medium == "air" else "vac"
    ax.set_xlabel(f"Wavelength, {unit} (nm)")
    # Compact ylabel; the widened-bin note goes INSIDE the axes.
    ax.set_ylabel(f"Lines / {eff_bin:.3g} nm")
    if eff_bin > float(bin_nm):
        ax.text(0.99, 0.97,
                f"bin widened {bin_nm:g} → {eff_bin:.3g} nm "
                "(≤2000 bins)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#666666")
    ax.grid(True, alpha=0.25)
    dense_ticks(ax)
    return int(len(wl))
