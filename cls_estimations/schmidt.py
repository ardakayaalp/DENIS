"""Schmidt single-particle nuclear magnetic-moment calculations.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Provides single-particle Schmidt moment estimates for isotopes whose
experimental nuclear moments are unknown, used to predict HFS A
constants. The calculations use the single-particle shell model
(with optional g_s quenching) and a weak-coupling addition rule for
two-nucleon configurations, backed by a table of shell-model orbitals.

Depends on: standard library and third-party packages only (typing).
"""
from typing import Optional, Dict, Tuple


# Nuclear shell model orbital data
ORBITALS = [
    # ── Magic 2 ──────────────────────────────────────────────────
    {"name": "1s1/2",  "capacity":  2, "l": 0, "j":  0.5, "magic_gap_after": True,  "magic_number": 2},
    # ── Magic 8 ──────────────────────────────────────────────────
    {"name": "1p3/2",  "capacity":  4, "l": 1, "j":  1.5, "magic_gap_after": False},
    {"name": "1p1/2",  "capacity":  2, "l": 1, "j":  0.5, "magic_gap_after": True,  "magic_number": 8},
    # ── Magic 20 ─────────────────────────────────────────────────
    {"name": "1d5/2",  "capacity":  6, "l": 2, "j":  2.5, "magic_gap_after": False},
    {"name": "2s1/2",  "capacity":  2, "l": 0, "j":  0.5, "magic_gap_after": False},
    {"name": "1d3/2",  "capacity":  4, "l": 2, "j":  1.5, "magic_gap_after": True,  "magic_number": 20},
    # ── Magic 28 ─────────────────────────────────────────────────
    {"name": "1f7/2",  "capacity":  8, "l": 3, "j":  3.5, "magic_gap_after": True,  "magic_number": 28},
    # ── Magic 50 ─────────────────────────────────────────────────
    {"name": "2p3/2",  "capacity":  4, "l": 1, "j":  1.5, "magic_gap_after": False},
    {"name": "1f5/2",  "capacity":  6, "l": 3, "j":  2.5, "magic_gap_after": False},
    {"name": "2p1/2",  "capacity":  2, "l": 1, "j":  0.5, "magic_gap_after": False},
    {"name": "1g9/2",  "capacity": 10, "l": 4, "j":  4.5, "magic_gap_after": True,  "magic_number": 50},
    # ── Magic 82 ─────────────────────────────────────────────────
    {"name": "1g7/2",  "capacity":  8, "l": 4, "j":  3.5, "magic_gap_after": False},
    {"name": "2d5/2",  "capacity":  6, "l": 2, "j":  2.5, "magic_gap_after": False},
    {"name": "2d3/2",  "capacity":  4, "l": 2, "j":  1.5, "magic_gap_after": False},
    {"name": "3s1/2",  "capacity":  2, "l": 0, "j":  0.5, "magic_gap_after": False},
    {"name": "1h11/2", "capacity": 12, "l": 5, "j":  5.5, "magic_gap_after": True,  "magic_number": 82},
    # ── Magic 126 ────────────────────────────────────────────────
    {"name": "1h9/2",  "capacity": 10, "l": 5, "j":  4.5, "magic_gap_after": False},
    {"name": "2f7/2",  "capacity":  8, "l": 3, "j":  3.5, "magic_gap_after": False},
    {"name": "2f5/2",  "capacity":  6, "l": 3, "j":  2.5, "magic_gap_after": False},
    {"name": "3p3/2",  "capacity":  4, "l": 1, "j":  1.5, "magic_gap_after": False},
    {"name": "3p1/2",  "capacity":  2, "l": 1, "j":  0.5, "magic_gap_after": False},
    {"name": "1i13/2", "capacity": 14, "l": 6, "j":  6.5, "magic_gap_after": True,  "magic_number": 126},
    # ── Predicted magic 184 (ordering per diagram) ───────────────
    {"name": "2g9/2",  "capacity": 10, "l": 4, "j":  4.5, "magic_gap_after": False},  # 136
    {"name": "3d5/2",  "capacity":  6, "l": 2, "j":  2.5, "magic_gap_after": False},  # 142
    {"name": "1i11/2", "capacity": 12, "l": 6, "j":  5.5, "magic_gap_after": False},  # 154
    {"name": "2g7/2",  "capacity":  8, "l": 4, "j":  3.5, "magic_gap_after": False},  # 162
    {"name": "4s1/2",  "capacity":  2, "l": 0, "j":  0.5, "magic_gap_after": False},  # 164
    {"name": "3d3/2",  "capacity":  4, "l": 2, "j":  1.5, "magic_gap_after": False},  # 168
    {"name": "1j15/2", "capacity": 16, "l": 7, "j":  7.5, "magic_gap_after": True,  "magic_number": 184},
]

# Free nucleon g-factors
G_S_P = 5.586   # Proton spin g-factor
G_S_N = -3.826  # Neutron spin g-factor
G_L_P = 1.0     # Proton orbital g-factor
G_L_N = 0.0     # Neutron orbital g-factor


def get_orbital(name: str) -> Optional[Dict]:
    """Get orbital data by name (e.g., '2p1/2')."""
    for orb in ORBITALS:
        if orb["name"] == name:
            return orb
    return None


def schmidt_moment(l: int, j: float, *, proton: bool, g_s_factor: float = 1.0) -> float:
    """Calculate Schmidt single-particle magnetic moment.

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number
    j : float
        Total angular momentum (l ± 1/2)
    proton : bool
        True for proton, False for neutron
    g_s_factor : float, optional
        Quenching factor for g_s (default 1.0 = no quenching)

    Returns
    -------
    float
        Magnetic moment in nuclear magnetons

    Notes
    -----
    Schmidt formulas:
    - For j = l + 1/2 (spin aligned):
        μ_p = j - 1/2 + g_s/2
        μ_n = g_s/2
    - For j = l - 1/2 (spin anti-aligned):
        μ_p = (j/(j+1)) * (j + 3/2 - g_s/2)
        μ_n = -(j/(j+1)) * g_s/2
    """
    if proton:
        g_s = G_S_P * g_s_factor
        if j == (l + 0.5):
            return j - 0.5 + (g_s / 2.0)
        elif j == (l - 0.5):
            return (j / (j + 1.0)) * (j + 1.5 - g_s / 2.0)
    else:
        g_s = G_S_N * g_s_factor
        if j == (l + 0.5):
            return g_s / 2.0
        elif j == (l - 0.5):
            return -(j / (j + 1.0)) * (g_s / 2.0)
    raise ValueError(f"Invalid (l={l}, j={j}) combination for Schmidt calculation")


def schmidt_moment_from_orbital(orbital_name: str, *, proton: bool,
                                  g_s_factor: float = 1.0) -> float:
    """Calculate Schmidt moment from orbital name.

    Parameters
    ----------
    orbital_name : str
        Orbital name (e.g., '2p1/2', '1f7/2')
    proton : bool
        True for proton, False for neutron
    g_s_factor : float, optional
        Quenching factor for g_s (default 1.0)

    Returns
    -------
    float
        Magnetic moment in nuclear magnetons
    """
    orb = get_orbital(orbital_name)
    if orb is None:
        raise ValueError(f"Unknown orbital: {orbital_name}")
    return schmidt_moment(orb["l"], orb["j"], proton=proton, g_s_factor=g_s_factor)


def g_factor_from_mu_j(mu: float, j: float) -> float:
    """Calculate g-factor from magnetic moment and spin.

    g = μ / j
    """
    if j == 0:
        raise ValueError("j must be nonzero to calculate g-factor")
    return mu / j


def two_particle_moment(orbital1: str, proton1: bool,
                        orbital2: str, proton2: bool,
                        I: float, *, g_s_factor: float = 1.0) -> float:
    """Calculate magnetic moment for two unpaired nucleons using addition rule.

    Uses the weak-coupling addition rule:
        g(I) = 0.5*(g1 + g2) + 0.5*(g1 - g2) * [j1(j1+1) - j2(j2+1)] / [I(I+1)]
        μ(I) = g(I) * I

    Parameters
    ----------
    orbital1 : str
        First nucleon orbital (e.g., '2p3/2')
    proton1 : bool
        True if first nucleon is proton
    orbital2 : str
        Second nucleon orbital
    proton2 : bool
        True if second nucleon is proton
    I : float
        Total nuclear spin
    g_s_factor : float, optional
        Quenching factor (default 1.0)

    Returns
    -------
    float
        Total magnetic moment in nuclear magnetons
    """
    if I is None or I <= 0:
        raise ValueError("Total spin I must be positive for two-particle calculation")

    orb1 = get_orbital(orbital1)
    orb2 = get_orbital(orbital2)
    if orb1 is None or orb2 is None:
        raise ValueError(f"Unknown orbital: {orbital1 if orb1 is None else orbital2}")

    mu1 = schmidt_moment(orb1["l"], orb1["j"], proton=proton1, g_s_factor=g_s_factor)
    mu2 = schmidt_moment(orb2["l"], orb2["j"], proton=proton2, g_s_factor=g_s_factor)

    j1, j2 = orb1["j"], orb2["j"]
    g1 = g_factor_from_mu_j(mu1, j1)
    g2 = g_factor_from_mu_j(mu2, j2)

    gI = 0.5 * (g1 + g2) + 0.5 * (g1 - g2) * (
        (j1 * (j1 + 1.0) - j2 * (j2 + 1.0)) / (I * (I + 1.0))
    )
    return gI * I


def calculate_schmidt_moment(valence_spec: dict, *, g_s_factor: float = 1.0) -> Tuple[float, str]:
    """Calculate Schmidt moment from valence specification.

    Parameters
    ----------
    valence_spec : dict
        Specification with keys:
        - For single particle: 'type' ('proton' or 'neutron'), 'orbital' (str)
        - For two particles: 'type1', 'orbital1', 'type2', 'orbital2', 'I' (total spin)
    g_s_factor : float, optional
        Quenching factor (default 1.0)

    Returns
    -------
    mu : float
        Calculated magnetic moment
    description : str
        Human-readable description of calculation
    """
    # Single particle case
    if 'orbital' in valence_spec and 'type' in valence_spec:
        particle_type = valence_spec['type']
        orbital = valence_spec['orbital']
        proton = (particle_type.lower() in ['proton', 'pi', 'p'])

        mu = schmidt_moment_from_orbital(orbital, proton=proton, g_s_factor=g_s_factor)
        desc = f"{particle_type} in {orbital} (g_s_factor={g_s_factor:.3f})"
        return mu, desc

    # Two particle case
    elif all(k in valence_spec for k in ['type1', 'orbital1', 'type2', 'orbital2', 'I']):
        proton1 = (valence_spec['type1'].lower() in ['proton', 'pi', 'p'])
        proton2 = (valence_spec['type2'].lower() in ['proton', 'pi', 'p'])

        mu = two_particle_moment(
            valence_spec['orbital1'], proton1,
            valence_spec['orbital2'], proton2,
            valence_spec['I'],
            g_s_factor=g_s_factor
        )
        desc = (f"{valence_spec['type1']}:{valence_spec['orbital1']} + "
                f"{valence_spec['type2']}:{valence_spec['orbital2']}, I={valence_spec['I']} "
                f"(g_s_factor={g_s_factor:.3f})")
        return mu, desc

    else:
        raise ValueError(
            "valence_spec must contain either "
            "('type', 'orbital') for single particle or "
            "('type1', 'orbital1', 'type2', 'orbital2', 'I') for two particles"
        )
