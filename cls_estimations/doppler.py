"""Relativistic Doppler kinematics for collinear laser spectroscopy.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Converts between ion-beam acceleration voltage, relativistic beta, and the
laser frequency seen by an ion in its rest frame (collinear and anti-collinear
geometries), solves for the resonance voltage / required laser frequency, and
builds simulated hyperfine spectra in voltage space. These relations underpin
the Estimate tab's spectrum simulation and the Pre-Analysis frequency axis.

Depends on: cls_estimations.constants (physical constants and unit factors).
"""

import numpy as np
from .constants import C_LIGHT, AMU_TO_KG, E_CHARGE, CM_TO_HZ


def _normalize_geometry(geom):
    """Normalise geometry string to 'anti-collinear' or 'collinear'."""
    g = geom.strip()
    for h in ["\u2013", "\u2014", "\u2012"]:  # en-dash, em-dash, figure-dash
        g = g.replace(h, "-")
    if "anti" in g.lower():
        return "anti-collinear"
    return "collinear"


def beta_from_voltage(V, mass_amu, charge):
    """Relativistic beta = v/c from acceleration voltage.

    Parameters
    ----------
    V : float or array  – acceleration voltage [V]
    mass_amu : float     – atomic mass [amu]
    charge : int         – charge state (number of e)
    """
    V = np.asarray(V, dtype=float)
    m = mass_amu * AMU_TO_KG
    E0 = m * C_LIGHT**2
    Etot = charge * E_CHARGE * V + E0
    beta2 = 1.0 - (E0**2) / (Etot**2)
    beta2 = np.clip(beta2, 0.0, 1.0 - 1e-14)
    return np.sqrt(beta2)


def voltage_from_beta(beta, mass_amu, charge):
    """Acceleration voltage from relativistic beta."""
    if np.any(beta <= 0) or np.any(beta >= 1):
        return None
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    m = mass_amu * AMU_TO_KG
    E0 = m * C_LIGHT**2
    Etot = gamma * E0
    V = (Etot - E0) / (charge * E_CHARGE)
    if np.any(V <= 0):
        return None
    return V


def nu_seen_by_ion(nu_laser_MHz, beta, geometry):
    """Frequency the ion sees in its rest frame from a lab-frame laser.

    Anti-collinear (head-on):   nu_seen = nu_lab * sqrt((1+beta)/(1-beta))
    Collinear (same direction): nu_seen = nu_lab * sqrt((1-beta)/(1+beta))
    """
    geom = _normalize_geometry(geometry)
    if geom == "anti-collinear":
        return nu_laser_MHz * np.sqrt((1.0 + beta) / (1.0 - beta))
    else:
        return nu_laser_MHz * np.sqrt((1.0 - beta) / (1.0 + beta))


def transition_frequency_MHz(lower_cm, upper_cm):
    """Transition frequency in MHz from lower/upper energy levels in cm^-1."""
    return (upper_cm - lower_cm) * CM_TO_HZ / 1e6


def voltage_to_frequency_array(v_array, cooler_v, laser_cm1,
                                mass_amu, harmonic, ref_freq_hz=0.0,
                                geometry='anti-collinear', charge=1):
    """Convert binned scanning-voltage centers to rest-frame MHz.

    Re-projects an already-binned voltage spectrum onto a rest-frame
    frequency axis, used when a voltage-merged spectrum is fitted in
    frequency mode: once binned, per-event Doppler information is gone, so a
    single set of merge-level Doppler parameters (cooler/laser/mass/harmonic)
    is applied to every bin center. Counts are unchanged; only the x-axis
    interpretation changes. Operates on a binned array rather than raw
    events, using one (cooler, laser, mass, harmonic) set for the whole
    spectrum.

    Parameters
    ----------
    v_array : 1-D array
        Scanning-voltage bin centers (DAC volts, lab frame).
    cooler_v : float
        Full acceleration voltage for the merged spectrum (V).
    laser_cm1 : float
        Lab-frame laser setpoint (cm^-1).
    mass_amu : float
        Ion mass in atomic mass units.
    harmonic : int
        Laser harmonic. nu_laser_seen_at_ion = harmonic * laser_cm1 * c.
    ref_freq_hz : float, optional
        If non-zero, subtract this rest-frame reference (Hz) so the
        returned values are detuning MHz, matching the convention
        ``Compute_WL(ref=ref_freq)`` produces. Default 0 -> absolute
        rest-frame MHz.
    geometry : str, optional
        'anti-collinear' (default, what CLS uses) or 'collinear'.
    charge : int, optional
        Ion charge state (default 1).

    Returns
    -------
    f_mhz : 1-D array
        Rest-frame frequency (MHz), one entry per input voltage bin.

    Notes
    -----
    The single-cooler/single-laser assumption is the whole point of the
    voltage-merge-into-frequency-fit path: the user has accepted that
    the source files were close enough to share one Doppler shift.
    Phase 4's spread-warning at fit time surfaces cases where they
    weren't, but never blocks.
    """
    v_array = np.asarray(v_array, dtype=float)
    nu_laser_mhz = laser_cm1 * harmonic * C_LIGHT * 100.0 / 1e6
    beam_v = cooler_v - v_array
    # Clamp to avoid beta_from_voltage's NaN tail when a stray scan
    # bin pushes beam_v below 1 V (which would happen if cooler_v is
    # accidentally less than the scan range -- typically misconfigured
    # merge metadata).
    beam_v = np.clip(beam_v, 1.0, None)
    beta = beta_from_voltage(beam_v, mass_amu, charge)
    nu_seen_mhz = nu_seen_by_ion(nu_laser_mhz, beta, geometry)
    if ref_freq_hz:
        nu_seen_mhz = nu_seen_mhz - ref_freq_hz / 1e6
    return nu_seen_mhz


def laser_frequency_for_resonance(nu_rest_MHz, voltage_V, mass_amu, charge, geometry):
    """Compute laser frequency needed for resonance at a given acceleration voltage.

    Inverse of resonance_voltage(): given the ion rest-frame transition frequency
    and the acceleration voltage, return the lab-frame laser frequency that will
    be Doppler-shifted into resonance.

    Parameters
    ----------
    nu_rest_MHz : float – transition rest-frame frequency [MHz]
    voltage_V : float   – acceleration voltage [V]
    mass_amu : float     – atomic mass [amu]
    charge : int         – charge state (number of e)
    geometry : str       – 'anti-collinear' or 'collinear'

    Returns
    -------
    float – required lab-frame laser frequency [MHz]
    """
    geom = _normalize_geometry(geometry)
    beta = beta_from_voltage(voltage_V, mass_amu, charge)
    if geom == "anti-collinear":
        # nu_seen = nu_laser * sqrt((1+beta)/(1-beta))  =>  nu_laser = nu_rest * sqrt((1-beta)/(1+beta))
        return nu_rest_MHz * np.sqrt((1.0 - beta) / (1.0 + beta))
    else:
        # nu_seen = nu_laser * sqrt((1-beta)/(1+beta))  =>  nu_laser = nu_rest * sqrt((1+beta)/(1-beta))
        return nu_rest_MHz * np.sqrt((1.0 + beta) / (1.0 - beta))


def resonance_voltage(nu_rest_MHz, nu_laser_MHz, mass_amu, charge, geometry):
    """Solve for the acceleration voltage at which the ion is resonant.

    Returns V_acc in volts.
    """
    geom = _normalize_geometry(geometry)
    r = nu_laser_MHz / nu_rest_MHz
    r2 = r * r

    if geom == "anti-collinear":
        beta = (1.0 - r2) / (1.0 + r2)
    else:
        beta = (r2 - 1.0) / (r2 + 1.0)

    if beta <= 0 or beta >= 1:
        return None
    return voltage_from_beta(beta, mass_amu, charge)


def generate_voltage_spectrum(
    peak_offsets_MHz,
    peak_intensities,
    centroid_MHz,
    nu_laser_MHz,
    mass_amu,
    charge,
    V0,
    geometry,
    fwhm_MHz,
    v_range,
    v_steps=1000,
):
    """Generate a simulated HFS spectrum in voltage space.

    Parameters
    ----------
    peak_offsets_MHz : array – HFS offsets from centroid [MHz]
    peak_intensities : array – normalised intensities
    centroid_MHz : float – centroid rest-frame frequency [MHz] (nu0 + isotope_shift)
    nu_laser_MHz : float – lab-frame laser frequency [MHz]
    mass_amu : float – atomic mass [amu]
    charge : int – charge state
    V0 : float – beam voltage [V]
    geometry : str
    fwhm_MHz : float – linewidth [MHz]
    v_range : tuple(float,float) – (dV_min, dV_max) voltage offset range around V0
    v_steps : int

    Returns
    -------
    dV_array : 1-D array – voltage offsets [V]
    intensity_array : 1-D array – normalised intensity
    """
    sigma = fwhm_MHz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dV = np.linspace(v_range[0], v_range[1], v_steps)
    Vacc = V0 - dV
    Vacc = np.clip(Vacc, 1.0, None)

    m_kg = mass_amu * AMU_TO_KG
    E0 = m_kg * C_LIGHT**2
    Etot = charge * E_CHARGE * Vacc + E0
    beta2 = 1.0 - (E0**2) / (Etot**2)
    beta2 = np.clip(beta2, 0.0, 1.0 - 1e-14)
    beta = np.sqrt(beta2)

    geom = _normalize_geometry(geometry)

    y = np.zeros_like(dV)
    for offset, amp in zip(peak_offsets_MHz, peak_intensities):
        nu_rest = centroid_MHz + offset
        if geom == "anti-collinear":
            nu_seen = nu_laser_MHz * np.sqrt((1.0 + beta) / (1.0 - beta))
        else:
            nu_seen = nu_laser_MHz * np.sqrt((1.0 - beta) / (1.0 + beta))
        det = nu_seen - nu_rest
        y += amp * np.exp(-(det**2) / (2.0 * sigma**2))

    if y.max() > 0:
        y /= y.max()
    return dV, y
