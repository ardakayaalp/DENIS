"""Walk (trace) and correlation (corner) plots of an emcee chain.

One renderer for the two places that draw them -- the figure written
into the iteration folder (project.py) and the live plot in the Results
tab (results_tab.py) -- so the saved file and the live view cannot
drift apart.

The chain arrives as ``(steps, walkers, ndim)`` for the walk plot and
already burned-in and flattened, ``(samples, ndim)``, for the corner
plot: the corner plot shows the posterior the reported values were
computed from, the walk plot shows the whole run with the burn-in cut
marked on it.
"""
import numpy as np

#: The 2-D credible regions of the corner plot, in sigma. For a 2-D
#: Gaussian the probability inside the n-sigma contour is
#: 1 - exp(-n^2 / 2): 11.8, 39.3, 67.5 and 86.5 % -- not the 68 / 95 %
#: of a 1-D interval. Same convention as the `corner` package.
CREDIBLE_SIGMAS = (0.5, 1.0, 1.5, 2.0)

#: Colour scales offered for the filled regions. All run light -> dark,
#: so the most probable (inner) region is always the darkest.
CONTOUR_CMAPS = ("Blues", "Reds", "Greens", "Purples", "Oranges",
                 "Greys", "PuBu", "BuPu", "YlGnBu", "YlOrBr")

CORNER_STYLES = ("Credible regions", "Scatter")
FONT_FAMILIES = ("serif", "sans-serif")

# Where on the colour scale the fills and the lines are taken from.
_FILL_RANGE = (0.22, 0.78)
_LINE_AT = 0.95


def credible_mass(sigma):
    """Probability inside the ``sigma`` contour of a 2-D Gaussian."""
    return 1.0 - float(np.exp(-0.5 * float(sigma) ** 2))


def posterior_slice(chain, burn=0, thin=1):
    """``(chain[burn::thin], burn_used)`` -- the samples the numbers use.

    The same rule as the burn-in recompute in fitting.py: asking to
    discard every step keeps the last two, and a slice too short to
    give percentiles falls back to the whole chain (burn_used 0).
    """
    chain = np.asarray(chain)
    burn = max(0, int(burn or 0))
    thin = max(1, int(thin or 1))
    if burn == 0 and thin == 1:
        return chain, 0
    nsteps = chain.shape[0]
    if burn >= nsteps:
        burn = max(0, nsteps - 2)
    sliced = chain[burn::thin]
    if sliced.shape[0] < 2:
        return chain, 0
    return sliced, burn


def density_thresholds(H, masses):
    """Density levels enclosing each probability mass, ascending.

    Sort the (smoothed) histogram from the densest bin down and walk
    the cumulative sum: the level for mass m is the density of the last
    bin still inside m. Returns None for an empty histogram. Levels are
    made strictly increasing, which contourf requires.
    """
    flat = np.sort(np.asarray(H, dtype=float).ravel())[::-1]
    cum = np.cumsum(flat)
    if cum.size == 0 or cum[-1] <= 0:
        return None
    cum = cum / cum[-1]
    levels = np.empty(len(masses))
    for k, m in enumerate(masses):
        inside = flat[cum <= m]
        levels[k] = inside[-1] if inside.size else flat[0]
    levels.sort()
    for k in range(1, len(levels)):
        if levels[k] <= levels[k - 1]:
            levels[k] = levels[k - 1] * (1.0 + 1e-6) + 1e-12
    return levels


def _colormap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[str(name)]
    except (KeyError, ValueError):
        return matplotlib.colormaps[CONTOUR_CMAPS[0]]


def region_colours(cmap_name):
    """``(fills outer -> inner, line colour)`` for a colour scale."""
    cmap = _colormap(cmap_name)
    fills = [cmap(v) for v in np.linspace(*_FILL_RANGE,
                                          len(CREDIBLE_SIGMAS))]
    return fills, cmap(_LINE_AT)


def _decimals(err):
    """Decimals that show an uncertainty to two significant figures,
    never fewer than two."""
    err = abs(float(err))
    if not np.isfinite(err) or err == 0:
        return 2
    return int(min(8, max(2, 1 - np.floor(np.log10(err)))))


def value_title(name, median, up, down):
    """``name = median^{+up}_{-down}`` as mathtext."""
    d = _decimals(min(abs(up), abs(down)) or max(abs(up), abs(down)))
    return (f"{name} = ${median:.{d}f}"
            f"^{{+{abs(up):.{d}f}}}_{{-{abs(down):.{d}f}}}$")


def _sigma_label(s):
    return f"{s:g}$\\sigma$ ({100.0 * credible_mass(s):.1f} %)"


def _ranges(flat):
    """A robust plotting range per column: 0.5-99.5 percentile, padded.

    Robust so that a walker that strayed during the transient cannot
    squeeze the whole posterior into one bin."""
    out = []
    for col in flat.T:
        lo, hi = np.percentile(col, [0.5, 99.5])
        if not hi > lo:
            span = max(abs(lo), 1.0) * 1e-6
            lo, hi = lo - span, hi + span
        pad = 0.05 * (hi - lo)
        out.append((lo - pad, hi + pad))
    return out


# ── Walk plot ────────────────────────────────────────────────────────

def draw_walk(fig, labels, chain, ws, run_num="?", burn=0):
    """Trace of every walker per parameter, burn-in cut marked.

    ``ws`` is the walk_plot style (shared_widgets defaults merged with
    the user's). The dashed line sits at the first step the reported
    values use; everything left of it was discarded.
    """
    from gui.analysis.helpers import param_axis_label
    chain = np.asarray(chain)
    n_var = len(labels)
    axes = fig.subplots(n_var, 1, sharex=True, squeeze=False)
    burn = int(burn or 0)
    show = (bool(ws.get("show_burn_line", True))
            and 0 < burn < chain.shape[0])
    colour = ws.get("burn_line_color", "#000000")
    for i, label in enumerate(labels):
        ax = axes[i, 0]
        ax.plot(chain[:, :, i], alpha=ws["trace_alpha"],
                lw=ws["trace_lw"])
        if show:
            ax.axvline(burn, color=colour, ls="--",
                       lw=float(ws.get("burn_line_width", 1.5)),
                       zorder=5, gid="burn_in_line")
        ax.set_ylabel(param_axis_label(label), fontsize=ws["label_size"])
        ax.tick_params(labelsize=ws["tick_size"])
    if show:
        top = axes[0, 0]
        top.annotate(
            f"burn-in: {burn} steps", xy=(burn, 1.0),
            xycoords=top.get_xaxis_transform(), xytext=(4, -3),
            textcoords="offset points", ha="left", va="top",
            fontsize=ws["tick_size"], color=colour, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.75))
    axes[-1, 0].set_xlabel("Step", fontsize=ws["label_size"])
    fig.suptitle(f"Walk Plot — Run {run_num}",
                 fontsize=ws["title_size"])
    return axes


# ── Correlation (corner) plot ────────────────────────────────────────

def draw_corner(fig, labels, flat, cs, run_num="?"):
    """Corner plot of the posterior samples ``flat`` (samples, ndim).

    ``cs["style"]`` picks "Credible regions" (filled 2-D regions, step
    histograms with the 16 / 50 / 84 % lines, value titles) or the
    earlier "Scatter" rendering.
    """
    flat = np.asarray(flat, dtype=float)
    if flat.ndim == 1:
        flat = flat[:, None]
    if flat.shape[1] != len(labels):
        raise ValueError(f"{flat.shape[1]} chain columns for "
                         f"{len(labels)} labels")
    if cs.get("style", CORNER_STYLES[0]) == "Scatter":
        return _draw_corner_scatter(fig, labels, flat, cs, run_num)
    return _draw_corner_regions(fig, labels, flat, cs, run_num)


def _draw_corner_scatter(fig, labels, flat, cs, run_num):
    """The corner plot as it was before the credible-region style."""
    from gui.analysis.helpers import param_axis_label
    n_var = len(labels)
    axes = fig.subplots(n_var, n_var, squeeze=False)
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
    fig.suptitle(f"Correlation — Run {run_num}",
                 fontsize=cs["title_size"])
    return axes


def _regions(ax, x, y, rx, ry, cs, fills, line):
    """Filled 2-D credible regions of (x, y) on ``ax``."""
    bins = max(5, int(cs.get("contour_bins", 30)))
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[rx, ry])
    smooth = float(cs.get("smooth", 1.0) or 0.0)
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, smooth)
    levels = density_thresholds(
        H, [credible_mass(s) for s in CREDIBLE_SIGMAS])
    if levels is None or not H.max() > 0:
        ax.plot(x, y, ",", color=line, alpha=0.3)
        return None
    # One empty bin on every side, so no region is cut open at the
    # edge of the range.
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    dx, dy = xe[1] - xe[0], ye[1] - ye[0]
    xp = np.concatenate([[xc[0] - dx], xc, [xc[-1] + dx]])
    yp = np.concatenate([[yc[0] - dy], yc, [yc[-1] + dy]])
    Hp = np.zeros((H.shape[0] + 2, H.shape[1] + 2))
    Hp[1:-1, 1:-1] = H
    top = Hp.max() * (1.0 + 1e-4)
    filled = ax.contourf(xp, yp, Hp.T, levels=list(levels) + [top],
                         colors=fills, antialiased=True)
    ax.contour(xp, yp, Hp.T, levels=levels, colors=[line],
               linewidths=float(cs.get("contour_lw", 1.0)))
    return filled


def _draw_corner_regions(fig, labels, flat, cs, run_num):
    from matplotlib.patches import Patch
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator
    from gui.analysis.helpers import param_axis_label

    n = len(labels)
    fills, line = region_colours(cs.get("contour_cmap", CONTOUR_CMAPS[0]))
    font = cs.get("font_family", FONT_FAMILIES[0])
    if font not in FONT_FAMILIES:
        font = FONT_FAMILIES[0]
    mfont = "stix" if font == "serif" else "dejavusans"
    text = dict(fontfamily=font, math_fontfamily=mfont)
    frame = float(cs.get("frame_lw", 1.5))
    label_size = cs.get("label_size", 9)
    tick_size = cs.get("tick_size", 8)
    ticks = max(2, int(cs.get("max_ticks", 3)))
    rng = _ranges(flat)

    axes = fig.subplots(n, n, squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            x = flat[:, j]
            if i == j:
                ax.hist(x, bins=int(cs.get("hist_bins", 40)), range=rng[j],
                        histtype="step", color=line,
                        lw=float(cs.get("hist_lw", 1.5)))
                lo, med, hi = np.percentile(x, [15.87, 50.0, 84.13])
                for q in (lo, med, hi):
                    ax.axvline(q, ls="--", color=line,
                               lw=float(cs.get("quantile_lw", 1.0)))
                ax.set_title(value_title(param_axis_label(labels[j]), med,
                                         hi - med, med - lo),
                             fontsize=label_size, **text)
                ax.set_yticks([])
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                _regions(ax, x, flat[:, i], rng[j], rng[i], cs, fills, line)
                ax.set_ylim(rng[i])
                ax.yaxis.set_major_locator(MaxNLocator(ticks, prune="lower"))
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlim(rng[j])
            ax.xaxis.set_major_locator(MaxNLocator(ticks, prune="lower"))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            for spine in ax.spines.values():
                spine.set_linewidth(frame)
            ax.tick_params(which="both", direction="in", top=True,
                           right=True, labelsize=tick_size,
                           labelfontfamily=font)
            if i == n - 1:
                ax.set_xlabel(param_axis_label(labels[j]),
                              fontsize=label_size, **text)
                ax.tick_params(axis="x", labelrotation=45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(param_axis_label(labels[i]),
                              fontsize=label_size, **text)
            else:
                ax.tick_params(labelleft=False)

    if n >= 2:
        # In the empty upper triangle, anchored to its top-right cell
        # so it moves with the grid and never covers a panel.
        handles = [Patch(facecolor=f, edgecolor=line, lw=1.0)
                   for f in reversed(fills)]
        leg = fig.legend(
            handles, [_sigma_label(s) for s in CREDIBLE_SIGMAS],
            title="2-D credible regions", loc="upper right",
            bbox_to_anchor=(1.0, 1.0), bbox_transform=axes[0, n - 1].transAxes,
            frameon=True, fancybox=False, fontsize=label_size,
            title_fontsize=label_size + 1, prop={"family": font,
                                                 "size": label_size})
        leg.get_frame().set_linewidth(frame)
        leg.get_frame().set_edgecolor("black")
        for t in leg.get_texts():
            t.set_math_fontfamily(mfont)
        leg.get_title().set_fontfamily(font)
        leg.set_gid("credible_regions_legend")
    fig.suptitle(f"Correlation — Run {run_num}",
                 fontsize=cs.get("title_size", 12), fontfamily=font)
    return axes
