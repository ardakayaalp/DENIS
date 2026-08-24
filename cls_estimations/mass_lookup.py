"""Element symbol/Z mapping and IUPAC nuclide mass-table lookup.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Holds the element-symbol-to-atomic-number table (and its inverse) and
loads the bundled CIAAW/IUPAC atomic-mass CSV into a ``{(Z, A): mass}``
dictionary, with a lookup helper that reports the available mass
numbers when an isotope is missing. Supplies isotope masses to the
Doppler and estimation calculations.

Depends on: standard library and third-party packages only (csv, re, os).
"""
import csv
import re
import os

ELEMENT_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
    "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
    "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
    "Og": 118,
}

Z_TO_ELEMENT = {z: sym for sym, z in ELEMENT_Z.items()}

_DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "cls_estimations", "IUPAC-atomic-masses.csv",
)
def load_mass_table(csv_path=None):
    """Load CIAAW/IUPAC nuclide mass table.  Returns dict {(Z,A): mass_amu}."""
    if csv_path is None:
        csv_path = _DEFAULT_CSV
    mass_table = {}
    with open(csv_path, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() == "nuclide":
                break
        for row in reader:
            if len(row) < 2:
                continue
            nuclide = row[0].strip()
            mass_str = row[1].strip()
            if not mass_str:
                continue
            try:
                mass_u = float(mass_str)
            except ValueError:
                continue
            m = re.match(r"(\d+)([A-Za-z]+)", nuclide)
            if not m:
                continue
            A = int(m.group(1))
            symbol = m.group(2)
            Z = ELEMENT_Z.get(symbol)
            if Z is None:
                continue
            mass_table[(Z, A)] = mass_u
    return mass_table


def get_mass(table, Z, A):
    """Look up mass in amu.  Raises KeyError if not found."""
    key = (int(Z), int(A))
    if key not in table:
        element = Z_TO_ELEMENT.get(int(Z), "?")
        available_A = sorted(a for (z, a) in table if z == int(Z))
        hint = (f" Available A for {element} (Z={Z}): {available_A}"
                if available_A else "")
        raise KeyError(
            f"Isotope not found: Z={Z} ({element}), A={A}.{hint}")
    return table[key]
