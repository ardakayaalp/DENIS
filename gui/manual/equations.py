"""Curated equation sources for the manual, keyed by catalog name.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Holds the hand-verified, mathtext-safe equation sources (each a list of mathtext
lines rendered stacked) and resolves an equation name to either rendered PNG or
its source lines. Curated lines use only constructs matplotlib mathtext supports
(``\\frac`` not ``\\dfrac``, ``\\mathrm`` not ``\\text``, no ``cases``
environment — piecewise relations are written as separate lines and labelled in
the surrounding prose). Names not curated here fall back to ``normalize_mathtext``
applied to the raw survey LaTeX in ``structure.EQUATION_LATEX``.

Depends on: gui.manual.mathrender, gui.manual.structure.
"""

import json
import os

from gui.manual import mathrender
from gui.manual.structure import EQUATION_LATEX

# Curated equations (mathtext-safe) loaded from the JSON sidecar. The
# hand-written entries below take precedence where the same name appears in both.
_CURATED_JSON = os.path.join(os.path.dirname(__file__), "equations_curated.json")
try:
    with open(_CURATED_JSON, encoding="utf-8") as _f:
        _GENERATED = json.load(_f)
except (OSError, ValueError):
    _GENERATED = {}

# name -> list of mathtext lines (verified against the cited implementation)
_HANDWRITTEN = {
    # cls_estimations/doppler.py:15  beta_from_voltage
    "beta-from-voltage": [
        r"$\beta = \sqrt{\,1 - \left(\dfrac{E_0}{qeV + E_0}\right)^{2}}$".replace(r"\dfrac", r"\frac"),
        r"$E_0 = m c^{2}$",
    ],
    # cls_estimations/doppler.py:33  voltage_from_beta
    "voltage-from-beta": [
        r"$\gamma = \frac{1}{\sqrt{1-\beta^{2}}}\,,\qquad V = \frac{(\gamma-1)\,m c^{2}}{q e}$",
    ],
    # cls_estimations/doppler.py:47  nu_seen_by_ion  (two geometries, labelled in prose)
    "doppler-seen": [
        r"$\nu_{\mathrm{seen}} = \nu_{\mathrm{lab}}\,\sqrt{\dfrac{1+\beta}{1-\beta}}$".replace(r"\dfrac", r"\frac"),
        r"$\nu_{\mathrm{seen}} = \nu_{\mathrm{lab}}\,\sqrt{\dfrac{1-\beta}{1+\beta}}$".replace(r"\dfrac", r"\frac"),
    ],
    # cls_estimations/doppler.py:60  transition_frequency_MHz
    "transition-frequency": [
        r"$\nu_0 = (E_{\mathrm{upper}} - E_{\mathrm{lower}})\,\cdot c$",
    ],
    # cls_estimations/doppler.py:135 laser_frequency_for_resonance
    "laser-frequency-resonance": [
        r"$\nu_{\mathrm{laser}} = \nu_{\mathrm{rest}}\,\sqrt{\frac{1-\beta}{1+\beta}}$",
        r"$\nu_{\mathrm{laser}} = \nu_{\mathrm{rest}}\,\sqrt{\frac{1+\beta}{1-\beta}}$",
    ],
    # cls_estimations/doppler.py:217 generate_voltage_spectrum
    "fwhm-to-sigma": [
        r"$\sigma = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2}}$",
    ],
}

# Generated curated set fills the catalog; hand-written entries take precedence.
CURATED = {**_GENERATED, **_HANDWRITTEN}


def get(name):
    """Return the mathtext-safe line list for an equation name (fallback path).

    Curated entries win; otherwise fall back to a normalized version of the raw
    survey LaTeX (which may be imperfect for piecewise/`cases` equations).
    """
    if name in CURATED:
        return CURATED[name]
    raw = EQUATION_LATEX.get(name)
    if raw is None:
        return [r"$\mathrm{?}$"]
    return ["$" + mathrender.normalize_mathtext(raw) + "$"]


def latex_source(name):
    """Best LaTeX lines for real (usetex) rendering.

    Uses the full survey LaTeX (with \\dfrac etc.) when it has no ``cases``
    environment — matplotlib's usetex cannot render ``cases`` — otherwise falls
    back to the curated, already-split mathtext-safe lines (also valid LaTeX).
    """
    raw = EQUATION_LATEX.get(name)
    if raw and "\\begin{cases}" not in raw and "\\cases" not in raw:
        return ["$" + raw + "$"]
    return get(name)


def render(name, **kwargs):
    """Render an equation by name to a PNG; return the file path.

    A committed (shipped) figs/eq/<name>.png wins, so end users get the
    LaTeX-rendered image without needing a TeX install. Otherwise render at
    runtime (usetex if available, else mathtext)."""
    committed = mathrender.committed_equation_path(name)
    if committed:
        return committed
    return mathrender.render_equation(
        key=name, latex_lines=latex_source(name), mathtext_lines=get(name),
        **kwargs)


def is_curated(name):
    return name in CURATED
