"""The GP diagnostic figure, drawn from flat arrays.

One function draws it for both places it appears: the live Reference
Correction panel and the Results tab, which re-renders it from the
.npz written by "Send to Results". There used to be two copies of this
drawing code, and they drifted. The Results copy knew nothing about
the time unit, the residual panel or excluded points, so the figure
you sent was not the figure you were looking at. Now the panel builds
the dict, draws it, and saves that same dict.

Everything is a flat array or a scalar, because np.savez cannot hold
ragged or nested data. The lower panel's per-project grouping travels
as a label table plus one label index per point. Missing keys fall
back to what older saves implied, so an .npz written before this
module still renders.

Times are in hours since the GP's first training point. ``t_scale``
converts them for display (60 for minutes) and is never baked into
the stored arrays.
"""
from __future__ import annotations

import numpy as np

EXCLUDED_COLOR = "#9e9e9e"
#: How far inside the frame an off-scale point is pinned, as a
#: fraction of the axis span.
EDGE_INSET = 0.05


def edge_y(ax, y):
    """``(y_drawn, marker)`` for a point on *ax*: itself when inside
    the y-range, pinned just inside the edge with a triangle when not.
    Shared by the drawing and the hover tips so the tip sits on the
    marker."""
    lo, hi = ax.get_ylim()
    inset = EDGE_INSET * (hi - lo)
    if y > hi:
        return hi - inset, "^"
    if y < lo:
        return lo + inset, "v"
    return y, None


def _fix_range(ax, lows, highs, pad_frac=0.08):
    """Pin the y-range to the fitted data, so excluded points drawn
    afterwards cannot stretch it."""
    lows = np.concatenate([np.ravel(a) for a in lows if np.size(a)]) \
        if any(np.size(a) for a in lows) else np.zeros(0)
    highs = np.concatenate([np.ravel(a) for a in highs if np.size(a)]) \
        if any(np.size(a) for a in highs) else np.zeros(0)
    lows, highs = lows[np.isfinite(lows)], highs[np.isfinite(highs)]
    if not lows.size or not highs.size:
        return
    lo, hi = float(lows.min()), float(highs.max())
    pad = pad_frac * (hi - lo) if hi > lo else 1.0
    ax.set_ylim(lo - pad, hi + pad)


def _draw_excluded(ax, t, y, yerr, *, fmt_style):
    """Excluded points: in range with their error bars, off range as
    edge triangles without them (an error bar at the frame edge would
    be a lie about where the point is)."""
    inside, pinned = [], []
    for i, yi in enumerate(y):
        yd, mk = edge_y(ax, float(yi))
        (pinned if mk else inside).append((i, yd, mk))
    if inside:
        idx = [i for i, *_ in inside]
        ax.errorbar(np.asarray(t)[idx], np.asarray(y)[idx],
                    yerr=(np.asarray(yerr)[idx] if np.size(yerr)
                          else None),
                    **fmt_style)
    for i, yd, mk in pinned:
        ax.plot([t[i]], [yd], mk, ms=fmt_style.get("markersize", 7),
                color=EXCLUDED_COLOR, zorder=5, clip_on=False)


REF_AVG_LEGEND = "{label} weighted avg {avg:+.2f} ± {err:.2f} MHz"


def _val(d, key, default=None):
    """A scalar from *d*, unboxing the 0-d arrays np.load returns."""
    if key not in d:
        return default
    v = d[key]
    try:
        return v.item()
    except (AttributeError, ValueError):
        return v


def _arr(d, key):
    if key not in d or d[key] is None:
        return np.zeros(0)
    a = np.asarray(d[key])
    return a.astype(float) if a.dtype != object else a


def _labels(d, key):
    if key not in d or d[key] is None:
        return []
    return [str(v) for v in np.asarray(d[key], dtype=object).ravel()]


def layout(d):
    """Which panels this dict draws: ``["drift", "res"?, "corr"?]``."""
    panels = ["drift"]
    if bool(_val(d, "show_residuals", False)) and _arr(d, "res_t").size:
        panels.append("res")
    # Older saves carry no show_corrected flag; the lower panel was
    # drawn whenever there was data for it.
    if bool(_val(d, "show_corrected", True)) and _arr(d, "corr_t").size:
        panels.append("corr")
    return panels


def draw(fig, d):
    """Draw the GP diagnostic onto *fig*. Returns ``{panel: ax}``.

    The caller owns clearing the figure and drawing the canvas.
    """
    from cls_estimations.reference_correction import (
        MAP_LEGEND_EXCLUDED, MAP_LEGEND_OBS, MAP_LINE_ALPHA,
        MAP_LINE_COLOR, MAP_LINE_WIDTH, OBS_CAPSIZE, OBS_CAPTHICK,
        OBS_ELINEWIDTH, OBS_MARKER_SIZE, style_gp_axes,
    )

    scale = float(_val(d, "t_scale", 1.0) or 1.0)
    unit = str(_val(d, "t_unit", "h") or "h")
    show_excl = bool(_val(d, "show_excluded", True))
    panels = layout(d)
    heights = {"drift": 3, "res": 1, "corr": 2}

    if len(panels) == 1:
        # A height-ratio gridspec is not tight_layout compatible, and
        # the single panel is laid out with tight_layout below.
        axs = [fig.add_subplot(111)]
    else:
        axs = [a for row in fig.subplots(
            len(panels), 1, sharex=True, squeeze=False,
            gridspec_kw={"height_ratios": [heights[p] for p in panels],
                         "hspace": 0.08}) for a in row]
    out = dict(zip(panels, axs))

    # ── Drift model ──
    ax = out["drift"]
    t_tr = _arr(d, "t_train")
    if t_tr.size:
        ax.errorbar(t_tr * scale, _arr(d, "y_train"),
                    yerr=_arr(d, "yerr_train"),
                    fmt="k.", markersize=OBS_MARKER_SIZE,
                    elinewidth=OBS_ELINEWIDTH, capsize=OBS_CAPSIZE,
                    capthick=OBS_CAPTHICK, label=MAP_LEGEND_OBS)
    t_g, mu, sig = _arr(d, "t_grid"), _arr(d, "mu"), _arr(d, "sigma")
    if t_g.size and mu.size and sig.size:
        ax.fill_between(t_g * scale, mu - 2 * sig, mu + 2 * sig,
                        color="#aac6e0", alpha=0.6,
                        label=r"2-$\sigma$ interval")
        ax.fill_between(t_g * scale, mu - sig, mu + sig,
                        color="#f3c98a", alpha=0.85,
                        label=r"1-$\sigma$ interval")
        ax.plot(t_g * scale, mu, color=MAP_LINE_COLOR,
                alpha=MAP_LINE_ALPHA, linewidth=MAP_LINE_WIDTH,
                label="MAP")
    t_ex = _arr(d, "t_excl")
    if show_excl and t_ex.size:
        y_tr, e_tr = _arr(d, "y_train"), _arr(d, "yerr_train")
        _fix_range(ax, [y_tr - e_tr, mu - 2 * sig],
                   [y_tr + e_tr, mu + 2 * sig])
        # On top of the bands, in grey: shown, not used.
        _draw_excluded(ax, t_ex * scale, _arr(d, "y_excl"),
                       _arr(d, "yerr_excl"), fmt_style=dict(
                           fmt="x", markersize=OBS_MARKER_SIZE + 1,
                           color=EXCLUDED_COLOR,
                           mew=OBS_ELINEWIDTH + 0.4,
                           elinewidth=OBS_ELINEWIDTH,
                           capsize=OBS_CAPSIZE, capthick=OBS_CAPTHICK,
                           zorder=5))
        # A proxy, so the legend entry exists whether the point is
        # in range or pinned to the edge.
        ax.errorbar([], [], yerr=[], fmt="x",
                    markersize=OBS_MARKER_SIZE + 1,
                    color=EXCLUDED_COLOR, mew=OBS_ELINEWIDTH + 0.4,
                    label=MAP_LEGEND_EXCLUDED)
    ax.set_ylabel(str(_val(d, "ylabel", "Centroid (MHz)")))
    title = str(_val(d, "title", "") or "")
    if title:
        ax.set_title(title)
    style_gp_axes(ax, legend_kw={"loc": "best"})

    # ── Residuals ──
    if "res" in out:
        ax_r = out["res"]
        ax_r.axhspan(-1.0, 1.0, color="#c9c9c9", alpha=0.35, zorder=0)
        ax_r.axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
        r = _arr(d, "res_r")
        ax_r.plot(_arr(d, "res_t") * scale, r, "o",
                  ms=OBS_MARKER_SIZE - 1, mfc="none",
                  mew=OBS_ELINEWIDTH, color="#37474f", zorder=3)
        # Range from the FITTED residuals only; an excluded point is
        # typically excluded for being far off, and would otherwise
        # squash every fitted residual onto the zero line.
        finite = r[np.isfinite(r)] if r.size else r
        lim = max(2.0, float(np.max(np.abs(finite))) * 1.25) \
            if finite.size else 2.0
        ax_r.set_ylim(-lim, lim)
        r_ex = _arr(d, "res_r_excl")
        if show_excl and r_ex.size:
            t_rx = _arr(d, "res_t_excl") * scale
            for ti, ri in zip(t_rx, r_ex):
                yd, mk = edge_y(ax_r, float(ri))
                ax_r.plot([ti], [yd], mk or "x",
                          ms=OBS_MARKER_SIZE, mew=OBS_ELINEWIDTH + 0.4,
                          color=EXCLUDED_COLOR, zorder=3,
                          clip_on=False)
        ax_r.set_ylabel("Res. (σ)")
        style_gp_axes(ax_r, legend=False)

    # ── Corrected centroids ──
    if "corr" in out:
        import matplotlib.pyplot as plt
        ax_c = out["corr"]
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        labels = _labels(d, "corr_labels")
        idx = _arr(d, "corr_index").astype(int)
        c_t, c_y = _arr(d, "corr_t"), _arr(d, "corr_y")
        c_e = _arr(d, "corr_yerr")
        excl = (_arr(d, "corr_excl").astype(bool)
                if "corr_excl" in d else np.zeros(c_t.size, bool))
        for i, lbl in enumerate(labels):
            color = palette[i % len(palette)]
            m = (idx == i) & ~excl
            if m.any():
                ax_c.errorbar(
                    c_t[m] * scale, c_y[m],
                    yerr=c_e[m] if c_e.size else None,
                    fmt="o", ms=OBS_MARKER_SIZE + 1, mfc="none",
                    mew=OBS_ELINEWIDTH + 0.3, color=color,
                    elinewidth=OBS_ELINEWIDTH, capsize=OBS_CAPSIZE,
                    capthick=OBS_CAPTHICK, label=lbl, zorder=2)
        if show_excl and excl.any():
            keep = ~excl
            e_k = c_e[keep] if c_e.size else np.zeros(int(keep.sum()))
            avgs = (_arr(d, "ref_avg")
                    if bool(_val(d, "show_ref_avg", False))
                    else np.zeros(0))
            _fix_range(ax_c, [c_y[keep] - e_k, avgs],
                       [c_y[keep] + e_k, avgs])
            _draw_excluded(
                ax_c, c_t[excl] * scale, c_y[excl],
                c_e[excl] if c_e.size else np.zeros(0),
                fmt_style=dict(fmt="x", markersize=OBS_MARKER_SIZE + 1,
                               mew=OBS_ELINEWIDTH + 0.4,
                               color=EXCLUDED_COLOR,
                               elinewidth=OBS_ELINEWIDTH,
                               capsize=OBS_CAPSIZE,
                               capthick=OBS_CAPTHICK, zorder=2))
        # The reference's own weighted average, if asked for. It is
        # where the reference actually sits in the corrected frame --
        # ~0 when the GP describes it well, and its offset from 0 is
        # the GP mean's bias against the data.
        if bool(_val(d, "show_ref_avg", False)):
            for lbl, avg, err in zip(_labels(d, "ref_avg_labels"),
                                     _arr(d, "ref_avg"),
                                     _arr(d, "ref_avg_err")):
                if not np.isfinite(avg):
                    continue
                color = (palette[labels.index(lbl) % len(palette)]
                         if lbl in labels else "#555555")
                ax_c.axhline(avg, ls="--", lw=1.4, color=color,
                             zorder=1, label=REF_AVG_LEGEND.format(
                                 label=lbl, avg=avg, err=err))
        ax_c.set_ylabel("Corrected centroid (MHz)")
        n_leg = len(ax_c.get_legend_handles_labels()[0])
        style_gp_axes(ax_c, legend=n_leg > 0,
                      legend_kw={"ncol": min(n_leg, 3) or 1,
                                 "loc": "best"})

    for a in axs[:-1]:
        a.set_xlabel("")
    axs[-1].set_xlabel(str(_val(
        d, "xlabel", f"Timestamp [{unit}] (since first measurement)")))

    if len(axs) == 1:
        fig.tight_layout(pad=0.6)
    else:
        # tight_layout cannot handle the shared-x gridspec; place the
        # panels directly so they stay glued on one x axis.
        fig.subplots_adjust(left=0.085, right=0.985, top=0.985,
                            bottom=0.085, hspace=0.07)
    return out
