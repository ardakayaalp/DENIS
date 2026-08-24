"""Authored manual pages for the 'Physics Reference' appendix.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds the HTML for the physics-reference pages, currently the relativistic
Doppler-shift page that documents the speed-from-voltage and frequency-seen-by-
ion relations and embeds the matching backend source. Equations and diagrams are
pulled from the manual's catalog via the render helpers.

Depends on: gui.manual.render.
"""

from gui.manual import render as R

# Source snippet quoting cls_estimations/doppler.py (nu_seen_by_ion); kept in
# sync with the live implementation it documents.
_NU_SEEN_SNIPPET = '''def nu_seen_by_ion(nu_laser_MHz, beta, geometry):
    """Frequency the ion sees in its rest frame from a lab-frame laser."""
    geom = _normalize_geometry(geometry)
    if geom == "anti-collinear":
        return nu_laser_MHz * np.sqrt((1.0 + beta) / (1.0 - beta))
    else:
        return nu_laser_MHz * np.sqrt((1.0 - beta) / (1.0 + beta))'''


def physics_doppler():
    return f"""
<h1>Relativistic Doppler Shift</h1>

<p>The Doppler transform is the heart of CLS: an ion accelerated to a few tens
of keV moves fast enough ({'β'} ~ 10{R.code_ref('⁻³')}) that the
laser frequency it <i>sees</i> in its rest frame differs measurably from the
lab-frame frequency, and that difference is tuned by scanning the acceleration
voltage. DENIS uses these relations throughout the
{R.link("tab-estimate", "Estimate")} and {R.link("tab-preanalysis",
"Pre-Analysis")} tabs.</p>

<h2>Speed from acceleration voltage</h2>
<p>An ion of rest mass <i>m</i> and charge state <i>q</i> accelerated through a
potential <i>V</i> gains kinetic energy <i>qeV</i>. Its relativistic speed
{'β'} = <i>v/c</i> follows from energy conservation:</p>

{R.equation("beta-from-voltage")}

<p>where <i>E</i>{R.code_ref('₀')} = <i>mc</i>{R.code_ref('²')} is the
rest energy. The inverse &mdash; the voltage that produces a given {'β'}
&mdash; is</p>

{R.equation("voltage-from-beta")}

<h2>Frequency seen by the ion</h2>
<p>The lab-frame laser frequency {'ν'}{R.code_ref('lab')} is shifted to the
rest-frame frequency {'ν'}{R.code_ref('seen')} the ion experiences. The sign
depends on geometry &mdash; whether the laser meets the ion head-on or travels
with it:</p>

{R.diagram("doppler-geometry",
           "Anti-collinear (head-on) blue-shifts; collinear red-shifts.")}

{R.equation("doppler-seen",
            "Top: anti-collinear (head-on). Bottom: collinear (co-propagating).")}

<p>This is exactly how the backend implements it &mdash; one branch per
geometry:</p>

{R.code(_NU_SEEN_SNIPPET, "cls_estimations/doppler.py")}

<h2>Symbols</h2>
<table cellpadding="4" cellspacing="0">
  <tr><td><b>{'β'}</b></td><td>ion speed as a fraction of <i>c</i></td></tr>
  <tr><td><b>{'γ'}</b></td><td>Lorentz factor, 1/&radic;(1&minus;{'β'}&sup2;)</td></tr>
  <tr><td><b>{'ν'}{R.code_ref('lab')}</b></td><td>laser frequency in the lab frame (MHz)</td></tr>
  <tr><td><b>{'ν'}{R.code_ref('seen')}</b></td><td>frequency in the ion rest frame (MHz)</td></tr>
  <tr><td><b>q, V, m</b></td><td>charge state, acceleration voltage (V), ion mass</td></tr>
</table>

<p>Geometry strings are normalised so any label containing &ldquo;anti&rdquo;
counts as anti-collinear; everything else is treated as collinear
({R.code_ref('_normalize_geometry')}).</p>

{R.related(["est-beam", "est-laser", "pa-xaxis-modes", "physics-resonance",
            "physics-simulated-spectrum"])}
"""


PAGES = {
    "physics-doppler": physics_doppler,
}
