"""YAML experiment-configuration parser and dataclass schema.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Defines the typed dataclasses (transition, beam, laser, reference,
scan, isotope, isomer, Schmidt valence) that describe a run-time
estimation case, and loads/validates a YAML file into a ``FullConfig``.
Enforces mutually exclusive options (e.g. laser setpoint vs. anchor,
single isomer yield specification) and derives effective background
rates from continuous-rate/ToF-gate blocks.

Depends on: standard library and third-party packages only (dataclasses,
typing, PyYAML).
"""
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class TransitionConfig:
    lower_level_cm: float
    upper_level_cm: float
    J_lower: float
    J_upper: float


@dataclass
class BeamConfig:
    voltage_kV: float
    charge_state: int
    geometry: str


@dataclass
class LaserConfig:
    harmonic: int
    setpoint_cm: Optional[float] = None
    anchor_isotope: Optional[str] = None
    anchor_dV: Optional[float] = None
    anchor_state: Optional[str] = None  # "gs" or "isomer"


@dataclass
class ReferenceConfig:
    A: int
    I: float
    mu: float
    Q: float
    A_lower_MHz: float
    B_lower_MHz: float
    A_upper_MHz: float
    B_upper_MHz: float


@dataclass
class ScanConfig:
    voltage_range_V: float
    step_size_V: Optional[float] = None
    dwell_time_ms: Optional[float] = None


@dataclass
class SchmidtValenceConfig:
    """Configuration for Schmidt moment calculation."""
    # Single particle: type, orbital
    # Two particles: type1, orbital1, type2, orbital2, I
    type: Optional[str] = None
    orbital: Optional[str] = None
    type1: Optional[str] = None
    orbital1: Optional[str] = None
    type2: Optional[str] = None
    orbital2: Optional[str] = None
    I: Optional[float] = None


@dataclass
class IsomerConfig:
    I: float
    mu: float
    Q: float
    isotope_shift_MHz: float
    peaks_to_measure: Optional[int] = None
    production_ratio: Optional[float] = None
    yield_ions_per_sec: Optional[float] = None
    spin_distribution_sigma: Optional[float] = None
    plot_with_gs: bool = True
    A_lower_MHz: Optional[float] = None
    B_lower_MHz: Optional[float] = None
    A_upper_MHz: Optional[float] = None
    B_upper_MHz: Optional[float] = None
    schmidt_valence: Optional[SchmidtValenceConfig] = None


@dataclass
class IsotopeConfig:
    A: int
    label: str
    I: float
    mu: float
    Q: float
    isotope_shift_MHz: float
    yield_ions_per_sec: Optional[float] = None
    spectroscopic_efficiency: Optional[float] = None
    background_rate_Hz: Optional[float] = None
    peaks_to_measure: Optional[int] = None
    A_lower_MHz: Optional[float] = None
    B_lower_MHz: Optional[float] = None
    A_upper_MHz: Optional[float] = None
    B_upper_MHz: Optional[float] = None
    schmidt_valence: Optional[SchmidtValenceConfig] = None
    isomer: Optional[IsomerConfig] = None
    uniform_timing: Optional[bool] = None
    background_continuous_Hz: Optional[float] = None
    background_tof_gate_us: Optional[float] = None


@dataclass
class FullConfig:
    element: str
    Z: int
    transition: TransitionConfig
    beam: BeamConfig
    laser: LaserConfig
    linewidth_fwhm_MHz: float
    reference: ReferenceConfig
    scan: ScanConfig
    isotopes: list  # list[IsotopeConfig]
    spectrum_only: bool = False
    required_sigma: Optional[float] = None
    g_s_factor: float = 1.0  # Schmidt g_s quenching factor
    uniform_timing: bool = False  # global default for uniform peak timing


def _parse_schmidt_valence(d):
    """Parse Schmidt valence configuration."""
    if d is None:
        return None
    return SchmidtValenceConfig(
        type=d.get("type"),
        orbital=d.get("orbital"),
        type1=d.get("type1"),
        orbital1=d.get("orbital1"),
        type2=d.get("type2"),
        orbital2=d.get("orbital2"),
        I=float(d["I"]) if "I" in d else None,
    )


def _parse_isomer(d, spectrum_only=False):
    if d is None:
        return None
    has_yield = "yield_ions_per_sec" in d
    has_ratio = "production_ratio" in d
    has_spin_sigma = "spin_distribution_sigma" in d
    if not spectrum_only:
        n_yield_specs = sum([has_yield, has_ratio, has_spin_sigma])
        if n_yield_specs == 0:
            raise ValueError(
                "Isomer must specify exactly one of: 'yield_ions_per_sec' (independent yield), "
                "'production_ratio' (fraction of g.s. yield), or "
                "'spin_distribution_sigma' (auto-calculate from spin statistics)."
            )
        if n_yield_specs > 1:
            raise ValueError(
                "Isomer specifies multiple yield methods. Choose exactly one of: "
                "'yield_ions_per_sec', 'production_ratio', or 'spin_distribution_sigma'."
            )
    return IsomerConfig(
        I=float(d["I"]),
        mu=float(d["mu"]),
        Q=float(d["Q"]),
        isotope_shift_MHz=float(d["isotope_shift_MHz"]),
        peaks_to_measure=int(d["peaks_to_measure"]) if "peaks_to_measure" in d else None,
        production_ratio=float(d["production_ratio"]) if has_ratio else None,
        yield_ions_per_sec=float(d["yield_ions_per_sec"]) if has_yield else None,
        spin_distribution_sigma=float(d["spin_distribution_sigma"]) if has_spin_sigma else None,
        plot_with_gs=bool(d.get("plot_with_gs", True)),
        A_lower_MHz=float(d["A_lower_MHz"]) if "A_lower_MHz" in d else None,
        B_lower_MHz=float(d["B_lower_MHz"]) if "B_lower_MHz" in d else None,
        A_upper_MHz=float(d["A_upper_MHz"]) if "A_upper_MHz" in d else None,
        B_upper_MHz=float(d["B_upper_MHz"]) if "B_upper_MHz" in d else None,
        schmidt_valence=_parse_schmidt_valence(d.get("schmidt_valence")),
    )


def _parse_isotope(d, dwell_time_ms, spectrum_only=False):
    # Determine effective background rate
    bg_continuous_Hz = None
    bg_tof_gate_us = None
    background_rate_Hz = None

    if not spectrum_only:
        bg_block = d.get("background")
        if bg_block is not None:
            bg_continuous_Hz = float(bg_block["continuous_rate_Hz"])
            bg_tof_gate_us = float(bg_block["tof_gate_us"])
            dwell_time_s = dwell_time_ms / 1000.0
            tof_gate_s = bg_tof_gate_us * 1e-6
            background_rate_Hz = bg_continuous_Hz * tof_gate_s / dwell_time_s
        elif "background_rate_Hz" in d:
            background_rate_Hz = float(d["background_rate_Hz"])
        else:
            raise ValueError(
                f"Isotope {d.get('label', d.get('A', '?'))}: must specify either "
                "'background_rate_Hz' or 'background: {continuous_rate_Hz, tof_gate_us}'."
            )

    return IsotopeConfig(
        A=int(d["A"]),
        label=str(d["label"]),
        I=float(d["I"]),
        mu=float(d["mu"]),
        Q=float(d["Q"]),
        isotope_shift_MHz=float(d["isotope_shift_MHz"]),
        yield_ions_per_sec=float(d["yield_ions_per_sec"]) if "yield_ions_per_sec" in d else None,
        spectroscopic_efficiency=float(d["spectroscopic_efficiency"]) if "spectroscopic_efficiency" in d else None,
        background_rate_Hz=background_rate_Hz,
        peaks_to_measure=int(d["peaks_to_measure"]) if "peaks_to_measure" in d else None,
        A_lower_MHz=float(d["A_lower_MHz"]) if "A_lower_MHz" in d else None,
        B_lower_MHz=float(d["B_lower_MHz"]) if "B_lower_MHz" in d else None,
        A_upper_MHz=float(d["A_upper_MHz"]) if "A_upper_MHz" in d else None,
        B_upper_MHz=float(d["B_upper_MHz"]) if "B_upper_MHz" in d else None,
        schmidt_valence=_parse_schmidt_valence(d.get("schmidt_valence")),
        isomer=_parse_isomer(d.get("isomer"), spectrum_only=spectrum_only),
        uniform_timing=bool(d["uniform_timing"]) if "uniform_timing" in d else None,
        background_continuous_Hz=bg_continuous_Hz,
        background_tof_gate_us=bg_tof_gate_us,
    )


def load_config(yaml_path):
    """Load and validate a YAML configuration file."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    t = raw["transition"]
    transition = TransitionConfig(
        lower_level_cm=float(t["lower_level_cm"]),
        upper_level_cm=float(t["upper_level_cm"]),
        J_lower=float(t["J_lower"]),
        J_upper=float(t["J_upper"]),
    )

    b = raw["beam"]
    beam = BeamConfig(
        voltage_kV=float(b["voltage_kV"]),
        charge_state=int(b["charge_state"]),
        geometry=str(b["geometry"]),
    )

    la = raw["laser"]
    has_setpoint = "setpoint_cm" in la
    has_anchor = "anchor" in la
    if has_setpoint and has_anchor:
        raise ValueError(
            "laser: specify either 'setpoint_cm' or 'anchor', not both."
        )
    if not has_setpoint and not has_anchor:
        raise ValueError(
            "laser: must specify either 'setpoint_cm' or 'anchor'."
        )
    if has_anchor:
        anc = la["anchor"]
        laser = LaserConfig(
            harmonic=int(la["harmonic"]),
            anchor_isotope=str(anc["isotope"]),
            anchor_dV=float(anc["dV"]),
            anchor_state=str(anc.get("state", "gs")),
        )
    else:
        laser = LaserConfig(
            harmonic=int(la["harmonic"]),
            setpoint_cm=float(la["setpoint_cm"]),
        )

    r = raw["reference"]
    reference = ReferenceConfig(
        A=int(r["A"]),
        I=float(r["I"]),
        mu=float(r["mu"]),
        Q=float(r["Q"]),
        A_lower_MHz=float(r["A_lower_MHz"]),
        B_lower_MHz=float(r["B_lower_MHz"]),
        A_upper_MHz=float(r["A_upper_MHz"]),
        B_upper_MHz=float(r["B_upper_MHz"]),
    )

    sc = raw["scan"]
    scan = ScanConfig(
        voltage_range_V=float(sc["voltage_range_V"]),
        step_size_V=float(sc["step_size_V"]) if "step_size_V" in sc else None,
        dwell_time_ms=float(sc["dwell_time_ms"]) if "dwell_time_ms" in sc else None,
    )

    lw = raw["linewidth"]
    fwhm = float(lw["fwhm_MHz"])

    stats = raw.get("statistics")
    if stats is not None:
        sigma = float(stats["required_sigma"])
        uniform_timing = bool(stats.get("uniform_timing", False))
        spectrum_only = False
    else:
        sigma = None
        uniform_timing = False
        spectrum_only = True

    isotopes = [_parse_isotope(iso, scan.dwell_time_ms, spectrum_only=spectrum_only)
                for iso in raw["isotopes"]]

    # Optional g_s_factor for Schmidt moment quenching
    g_s_factor = float(raw.get("g_s_factor", 1.0))

    return FullConfig(
        element=str(raw["element"]),
        Z=int(raw["Z"]),
        transition=transition,
        beam=beam,
        laser=laser,
        linewidth_fwhm_MHz=fwhm,
        reference=reference,
        scan=scan,
        isotopes=isotopes,
        spectrum_only=spectrum_only,
        required_sigma=sigma,
        g_s_factor=g_s_factor,
        uniform_timing=uniform_timing,
    )
