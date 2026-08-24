"""Render equations to cached PNG images for the manual.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Renders equations via real LaTeX (usetex) when a TeX distribution is present,
otherwise via matplotlib's built-in mathtext engine (no system LaTeX required).
It never mutates matplotlib's global ``rcParams``: the math font is selected
per-Text via ``math_fontfamily`` and usetex is enabled only inside an
``rc_context``, so the app's own plots are unaffected. Equations are rendered
light-on-transparent for the dark documentation background and cached on disk by
content hash so repeated views are instant.

Depends on: standard library and third-party packages only (matplotlib).
"""

import hashlib
import os
import re
import shutil

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")
# Committed, shipped equation images (rendered with real LaTeX on a machine
# that has it, so end users get LaTeX quality without needing a TeX install).
_EQ_DIR = os.path.join(os.path.dirname(__file__), "figs", "eq")

# Defaults tuned for the dark documentation theme.
DEFAULT_COLOR = "#e6e6e6"
DEFAULT_FONTSIZE = 17
DEFAULT_DPI = 160
MATH_FONTFAMILY = "cm"  # Computer Modern look; alternatives: 'stix', 'dejavusans'

# True LaTeX rendering (usetex) needs a TeX distribution + dvipng. Detected
# once; when present, equations render via real LaTeX, else via mathtext.
LATEX_AVAILABLE = bool(shutil.which("latex") and shutil.which("dvipng"))
LATEX_DPI = 200  # crisper than mathtext; the viewer scales images down to fit


def _ensure_cache():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(key):
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"eq_{h}.png")


def normalize_mathtext(latex):
    """Best-effort conversion of survey LaTeX to mathtext-safe source.

    matplotlib mathtext lacks ``\\dfrac``/``\\tfrac`` (use ``\\frac``),
    ``\\text{}`` (use ``\\mathrm{}``), and environments such as
    ``\\begin{cases}``. This handles the common substitutions; piecewise
    equations should instead be hand-curated as separate lines in
    ``equations.py``. Used only as a fallback for un-curated equations.
    """
    s = latex
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    def _roman(m):
        inner = m.group(1).replace(" ", r"\ ")
        return r"\mathrm{" + inner + "}"

    s = re.sub(r"\\text\{([^}]*)\}", _roman, s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", _roman, s)
    # mathtext has no \left[ ... \right] spacing issues, but \! / \, / \; ok.
    return s


# Delimiter-sizing commands mathtext does not support (drop them, keep the
# delimiter). Longer/more-specific names first so plain \big/\Big don't leave
# a stray suffix.
_BIG_CMDS = ("\\bigg", "\\Bigg", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr",
             "\\big", "\\Big")


def _mathtext_safe(s):
    """Coerce a line into the mathtext subset, defensively (covers imperfectly
    curated equations): \\dfrac->\\frac, drop \\big* sizing, \\text{}->\\mathrm{},
    and strip a bare \\text (a common bad escape of hyphenated words)."""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for cmd in _BIG_CMDS:
        s = s.replace(cmd, "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", s)
    s = re.sub(r"\\text(?![A-Za-z{])", "", s)
    return s


def _plainify(s):
    """Last-resort: strip math markup so a broken equation still renders as
    readable (non-math) text rather than crashing."""
    s = s.replace("$", "")
    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return s.strip() or "(equation)"


def _render_to(body, path, color, fontsize, dpi):
    fig = Figure()
    fig.patch.set_alpha(0.0)
    fig.text(0.0, 0.0, body, fontsize=fontsize, color=color,
             math_fontfamily=MATH_FONTFAMILY, ha="left", va="bottom")
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.06)


def _render_latex(body, path, color, fontsize, dpi):
    """Render via real LaTeX (usetex). rc_context keeps text.usetex / preamble
    local to this call so the app's own matplotlib plots are unaffected."""
    with matplotlib.rc_context({"text.usetex": True,
                                "text.latex.preamble": r"\usepackage{amsmath}"}):
        fig = Figure()
        fig.patch.set_alpha(0.0)
        fig.text(0.0, 0.0, body, fontsize=fontsize, color=color,
                 ha="left", va="bottom")
        FigureCanvasAgg(fig)
        fig.savefig(path, dpi=dpi, transparent=True,
                    bbox_inches="tight", pad_inches=0.08)


def committed_equation_path(name):
    """Return the shipped figs/eq/<name>.png if it exists, else None."""
    p = os.path.join(_EQ_DIR, name + ".png")
    return p if os.path.exists(p) else None


def render_equation(key, latex_lines, mathtext_lines, color=DEFAULT_COLOR,
                    fontsize=DEFAULT_FONTSIZE, dpi=DEFAULT_DPI):
    """Render an equation, preferring real LaTeX (usetex) and falling back to
    mathtext, then to plain text. Cached on disk by content."""
    _ensure_cache()
    cache_key = (f"{key}|{latex_lines}|{mathtext_lines}|{color}|{fontsize}"
                 f"|{LATEX_AVAILABLE}")
    path = _cache_path(cache_key)
    if os.path.exists(path):
        return path
    if LATEX_AVAILABLE and latex_lines:
        try:
            _render_latex("\n".join(latex_lines), path, color, fontsize, LATEX_DPI)
            return path
        except Exception:
            pass  # fall through to mathtext
    safe = "\n".join(_mathtext_safe(line) for line in mathtext_lines)
    try:
        _render_to(safe, path, color, fontsize, dpi)
    except Exception:
        _render_to("\n".join(_plainify(line) for line in mathtext_lines),
                   path, color, fontsize, dpi)
    return path


def build_committed_equation(name, latex_lines, mathtext_lines,
                             color=DEFAULT_COLOR, fontsize=DEFAULT_FONTSIZE):
    """Render one equation to the committed figs/eq/<name>.png. Returns the
    engine actually used ('latex' | 'mathtext' | 'plain')."""
    os.makedirs(_EQ_DIR, exist_ok=True)
    path = os.path.join(_EQ_DIR, name + ".png")
    if LATEX_AVAILABLE and latex_lines:
        try:
            _render_latex("\n".join(latex_lines), path, color, fontsize, LATEX_DPI)
            return "latex"
        except Exception:
            pass
    try:
        _render_to("\n".join(_mathtext_safe(l) for l in mathtext_lines),
                   path, color, fontsize, DEFAULT_DPI)
        return "mathtext"
    except Exception:
        _render_to("\n".join(_plainify(l) for l in mathtext_lines),
                   path, color, fontsize, DEFAULT_DPI)
        return "plain"


def render_lines(lines, color=DEFAULT_COLOR, fontsize=DEFAULT_FONTSIZE,
                 dpi=DEFAULT_DPI):
    """Render a list of mathtext lines (each like ``r"$\\beta = ...$"``)
    stacked vertically into a single cached PNG. Returns the file path.

    Each line is sanitized into the mathtext subset; if it still fails to parse
    it is rendered as plain text so a malformed equation never crashes the
    viewer. A line may also be plain (non-``$``) text.
    """
    if isinstance(lines, str):
        lines = [lines]
    key = f"{chr(10).join(lines)}|{color}|{fontsize}|{dpi}|{MATH_FONTFAMILY}"
    _ensure_cache()
    path = _cache_path(key)
    if os.path.exists(path):
        return path

    safe_body = "\n".join(_mathtext_safe(line) for line in lines)
    try:
        _render_to(safe_body, path, color, fontsize, dpi)
    except Exception:
        plain = "\n".join(_plainify(line) for line in lines)
        _render_to(plain, path, color, fontsize, dpi)
    return path


def clear_cache():
    """Delete all cached rendered images (regenerated on next view)."""
    if not os.path.isdir(_CACHE_DIR):
        return
    for f in os.listdir(_CACHE_DIR):
        if f.endswith(".png"):
            try:
                os.remove(os.path.join(_CACHE_DIR, f))
            except OSError:
                pass
