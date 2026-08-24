"""Data models for the NIST ASD scheme finder.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Plain, YAML-friendly containers: transitions are carried as small
dicts (a fixed key subset extracted from the NIST lines table), so a
RankedScheme serializes losslessly into the unified DENIS save file
and restores offline without refetching NIST.

Depends on: gui.nist_asd.data (wavelength helper); pandas only for
row-to-dict extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# The transition fields persisted with a scheme (everything the UI,
# report and diagram need — nothing else from the NIST row).
TRANSITION_KEYS = (
    "ritz_wl_vac(nm)", "Aki(s^-1)", "Ei(cm-1)", "Ek(cm-1)",
    "conf_i", "term_i", "J_i", "conf_k", "term_k", "J_k",
    "Type", "Acc",
)


def transition_dict(row) -> dict:
    """Extract the persisted subset from a NIST lines row
    (pandas Series or mapping)."""
    get = row.get if hasattr(row, "get") else row.__getitem__
    out = {}
    for k in TRANSITION_KEYS:
        v = get(k, None)
        if pd.isna(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out


@dataclass
class Laser:
    """An experimental laser system.

    ``role``: "PUMP" (never drives the final step), "PROBE" (only the
    final step), or "ANY". Wavelengths are in the search's configured
    medium.
    """
    name: str
    center_nm: float
    range_nm: float
    role: str = "ANY"

    @property
    def wl_min(self) -> float:
        return self.center_nm - self.range_nm

    @property
    def wl_max(self) -> float:
        return self.center_nm + self.range_nm

    def can_drive(self, wl_nm: float) -> bool:
        return self.wl_min <= wl_nm <= self.wl_max

    def to_dict(self) -> dict:
        return {"name": self.name, "center_nm": float(self.center_nm),
                "range_nm": float(self.range_nm), "role": self.role}

    @classmethod
    def from_dict(cls, d: dict) -> "Laser":
        return cls(name=str(d.get("name", "laser")),
                   center_nm=float(d.get("center_nm", 0.0)),
                   range_nm=float(d.get("range_nm", 0.0)),
                   role=str(d.get("role", "ANY")))


@dataclass
class SchemeStep:
    """One laser-driven upward transition of a scheme."""
    transition: dict
    laser: str | None = None      # laser name, or None in broad mode

    def to_dict(self) -> dict:
        return {"transition": dict(self.transition),
                "laser": self.laser}

    @classmethod
    def from_dict(cls, d: dict) -> "SchemeStep":
        return cls(transition=dict(d.get("transition", {})),
                   laser=d.get("laser"))


@dataclass
class Scheme:
    """A complete excitation + detection pathway."""
    steps: list                    # list[SchemeStep], ground → top
    detection: dict                # decay transition (fluorescence)
    branching_ratio: float
    start_level: float
    weight: float = 1.0

    @property
    def final_level(self) -> float:
        return float(self.steps[-1].transition.get("Ek(cm-1)") or 0.0)

    def signature(self) -> tuple:
        """Dedup key: the physical pathway, independent of lasers."""
        parts = []
        for s in self.steps:
            t = s.transition
            parts.append((round(float(t.get("Ei(cm-1)") or 0), 2),
                          round(float(t.get("Ek(cm-1)") or 0), 2)))
        d = self.detection
        parts.append((round(float(d.get("Ek(cm-1)") or 0), 2),
                      round(float(d.get("Ei(cm-1)") or 0), 2)))
        return tuple(parts)

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps],
                "detection": dict(self.detection),
                "branching_ratio": float(self.branching_ratio),
                "start_level": float(self.start_level),
                "weight": float(self.weight)}

    @classmethod
    def from_dict(cls, d: dict) -> "Scheme":
        return cls(
            steps=[SchemeStep.from_dict(s) for s in d.get("steps", [])],
            detection=dict(d.get("detection", {})),
            branching_ratio=float(d.get("branching_ratio", 0.0)),
            start_level=float(d.get("start_level", 0.0)),
            weight=float(d.get("weight", 1.0)))


@dataclass
class IsobarIssue:
    """A potential isobaric-contamination overlap."""
    isobar: str
    primary_wl_nm: float
    isobar_wl_nm: float
    isobar_aki: float | None = None

    def to_dict(self) -> dict:
        return {"isobar": self.isobar,
                "primary_wl_nm": float(self.primary_wl_nm),
                "isobar_wl_nm": float(self.isobar_wl_nm),
                "isobar_aki": (None if self.isobar_aki is None
                               else float(self.isobar_aki))}

    @classmethod
    def from_dict(cls, d: dict) -> "IsobarIssue":
        return cls(isobar=str(d.get("isobar", "?")),
                   primary_wl_nm=float(d.get("primary_wl_nm", 0.0)),
                   isobar_wl_nm=float(d.get("isobar_wl_nm", 0.0)),
                   isobar_aki=d.get("isobar_aki"))


@dataclass
class RankedScheme:
    """A scored scheme plus its warnings and contamination issues."""
    scheme: Scheme
    score: float
    issues: list = field(default_factory=list)    # list[IsobarIssue]
    warnings: list = field(default_factory=list)  # list[str]

    def __lt__(self, other):
        return self.score < other.score

    def to_dict(self) -> dict:
        return {"scheme": self.scheme.to_dict(),
                "score": float(self.score),
                "issues": [i.to_dict() for i in self.issues],
                "warnings": list(self.warnings)}

    @classmethod
    def from_dict(cls, d: dict) -> "RankedScheme":
        return cls(
            scheme=Scheme.from_dict(d.get("scheme", {})),
            score=float(d.get("score", 0.0)),
            issues=[IsobarIssue.from_dict(i)
                    for i in d.get("issues", [])],
            warnings=[str(w) for w in d.get("warnings", [])])
