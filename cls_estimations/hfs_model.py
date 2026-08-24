"""Hyperfine-structure line positions, intensities, and A/B scaling.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Computes HFS line offsets (MHz from centroid) and normalised
intensities for a given spin and set of A/B constants via satlas2, and
scales the magnetic-dipole (A) and electric-quadrupole (B) hyperfine
constants from a reference isotope using its known moments. Provides
the spectrum model used by the estimation and plotting code.

Depends on: standard library and third-party packages only (numpy
always; satlas2 imported lazily).
"""
import numpy as np
# satlas2 is imported lazily inside compute_hfs_lines so that importing this
# module (and the cls_estimate chain that pulls it in) does NOT load
# satlas2/lmfit/scipy/emcee at application startup.


def compute_hfs_lines(I, J_lower, J_upper, A_lower, B_lower, A_upper, B_upper):
    """Compute HFS line positions (MHz offsets from centroid) and normalised intensities.

    If I == 0, returns a single line at offset 0 with intensity 1.
    """
    if I == 0:
        return np.array([0.0]), np.array([1.0])

    import satlas2 as sat  # lazy: first call pays the import, off startup path

    hfs = sat.HFS(
        I=I,
        J=[J_lower, J_upper],
        A=[A_lower, A_upper],
        B=[B_lower, B_upper],
        C=[0.0, 0.0],
        df=0,
        name="hfs_tmp",
    )
    offsets = np.array(hfs.pos(), dtype=float)
    intens_dict = hfs.intensities
    ints = []
    for key in hfs.lines:
        p = intens_dict[f"Amp{key}"]
        try:
            val = float(p.value)
        except (AttributeError, TypeError):
            val = float(p)
        ints.append(val)

    intens = np.array(ints, dtype=float)
    s = intens.sum()
    if s > 0:
        intens /= s
    return offsets, intens


def scale_A(A_ref, mu_ref, I_ref, mu, I):
    """Scale magnetic dipole HFS constant A from a reference isotope.

    A_scaled = (mu * I_ref * A_ref) / (mu_ref * I)
    """
    if mu_ref == 0:
        raise ValueError("Reference magnetic moment mu_ref cannot be zero for A scaling")
    if I == 0:
        return 0.0
    return (mu * I_ref * A_ref) / (mu_ref * I)


def scale_B(B_ref, Q_ref, Q):
    """Scale electric quadrupole HFS constant B from a reference isotope.

    B_scaled = (Q * B_ref) / Q_ref
    """
    if Q_ref == 0:
        if Q == 0:
            return 0.0
        raise ValueError("Reference quadrupole moment Q_ref cannot be zero for B scaling (with Q != 0)")
    return (Q * B_ref) / Q_ref
