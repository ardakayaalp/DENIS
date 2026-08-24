"""Authored manual pages for the 'Estimate' tab section.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds the HTML for the Estimate-tab reference pages, currently the Beam page
covering acceleration voltage, charge state, and geometry and how they feed the
relativistic Doppler model that drives spectrum simulation and scan timing.

Depends on: gui.manual.render.
"""

from gui.manual import render as R


def est_beam():
    return f"""
<h1>Beam (Voltage, Charge, Geometry)</h1>

<p>The <b>Beam</b> group in the Estimate tab sets the three quantities that fix
how acceleration voltage maps onto rest-frame frequency. Together they drive the
relativistic {R.link("physics-doppler", "Doppler model")} used to simulate the
hyperfine spectrum and to predict scan timing.</p>

<h2>Controls</h2>
<table cellpadding="5" cellspacing="0">
  <tr>
    <td><b>Voltage (kV)</b></td>
    <td>Ion-beam acceleration voltage. Default <b>30&nbsp;kV</b>; range
        0.001&ndash;200&nbsp;kV, three decimals. Typical CLS beams are
        30&ndash;60&nbsp;kV.</td>
  </tr>
  <tr>
    <td><b>Charge state</b></td>
    <td>Ion charge <i>q</i> (number of electrons removed); integer 1&ndash;10.
        Enters the kinetic energy as <i>qeV</i>.</td>
  </tr>
  <tr>
    <td><b>Geometry</b></td>
    <td><b>anti-collinear</b> (head-on, default) or <b>collinear</b>
        (co-propagating). This picks the sign of the Doppler shift.</td>
  </tr>
</table>

<p>The voltage entered in kV is converted to volts internally and fed, with the
charge state and the ion mass, into the speed relation</p>

{R.equation("beta-from-voltage")}

<p>Scanning the voltage sweeps {'β'}, which Doppler-shifts the fixed lab-frame
laser across the resonance &mdash; see {R.link("physics-doppler",
"Relativistic Doppler Shift")} for the frequency the ion actually sees, and
{R.link("est-laser", "Laser (Setpoint vs Anchor-to-Isotope)")} for how the
lab-frame frequency is chosen.</p>

{R.note("Voltage here is the <i>full</i> acceleration potential. In the "
        "Pre-Analysis and Analysis tabs the effective beam voltage is the "
        "cooler voltage minus the scanning voltage; see "
        + R.link("pa-xaxis-modes", "X-Axis Modes") + ".")}

{R.related(["physics-doppler", "est-laser", "est-linewidth-scan",
            "est-element-transition"])}
"""


PAGES = {
    "est-beam": est_beam,
}
