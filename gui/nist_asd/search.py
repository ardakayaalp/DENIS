"""Multi-step excitation scheme search over NIST ASD line data.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Qt-free search engine combining the best of both standalone
prototypes: recursive N-step upward pathfinding with named lasers
(each used at most once; PUMP never drives the final step, PROBE only
the final step) or a broad wavelength window; per-role Aki min/max and
orbital filters; decay evaluation with branching ratios; detection
same/different-wavelength constraint; isobaric-contamination checks;
log-scoring; physical-pathway dedup.

Score = log10(weight) + Σ log10(Aki_step) + log10(BR)
        − 5 · (# isobar issues)

The engine takes plain callables for progress/cancel so the UI can run
it inside a QThread without this module importing Qt.

Depends on: gui.nist_asd.data, gui.nist_asd.models; pandas, numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gui.nist_asd import data as nd
from gui.nist_asd.models import (
    IsobarIssue, Laser, RankedScheme, Scheme, SchemeStep,
    transition_dict,
)


@dataclass
class SchemeSearchConfig:
    """All knobs of a scheme search (YAML round-trippable)."""
    spectrum: str = "Si I"
    # [{"level": cm-1, "weight": relative population}, ...]
    starting_levels: list = field(default_factory=lambda: [
        {"level": 0.0, "weight": 1.0}])
    auto_discover: bool = False
    metastable_aki_threshold: float = 1.0e4
    lasers: list = field(default_factory=list)      # list[Laser dicts]
    broad_min_nm: float | None = None               # used when no lasers
    broad_max_nm: float | None = None
    num_steps: int = 1                              # 1..3
    aki_min_pump: float | None = None
    aki_min_probe: float | None = None
    aki_min_detect: float | None = None
    min_branching_pct: float = 1.0
    detection_constraint: str = "any"   # any | same | different
    detection_proximity_nm: float = 1.0
    orbital_filter_pump: list = field(default_factory=list)
    orbital_filter_probe: list = field(default_factory=list)
    # [{"spectrum": "P I", "proximity_nm": 2.0}, ...]
    isobars: list = field(default_factory=list)
    medium: str = "air"                 # laser wavelengths medium
    level_tolerance: float = 0.1
    max_results: int = 500

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["lasers"] = [l.to_dict() if isinstance(l, Laser) else dict(l)
                       for l in self.lasers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SchemeSearchConfig":
        cfg = cls()
        for k, v in (d or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.num_steps = max(1, min(3, int(cfg.num_steps)))
        return cfg

    def laser_objects(self) -> list:
        return [l if isinstance(l, Laser) else Laser.from_dict(l)
                for l in self.lasers]


class SchemeSearcher:
    """Runs one scheme search over pre-fetched line/level tables."""

    def __init__(self, config: SchemeSearchConfig,
                 lines_df: pd.DataFrame,
                 levels_df: pd.DataFrame,
                 isobar_lines: dict | None = None,
                 progress=None, is_cancelled=None):
        self.cfg = config
        self.lines = nd.transitions_only(lines_df)
        self.levels = levels_df
        self.isobar_lines = isobar_lines or {}
        self._progress = progress or (lambda msg: None)
        self._cancelled = is_cancelled or (lambda: False)
        self._lasers = config.laser_objects()
        self._use_lasers = bool(self._lasers)
        self._paths: list = []

    # ── public API ──────────────────────────────────────────────────

    def run(self) -> list:
        """Full search → ranked, deduped, sorted list[RankedScheme]."""
        starts = self._resolve_starting_levels()
        self._progress(f"Searching from {len(starts)} starting "
                       f"level(s), {self.cfg.num_steps} step(s)...")
        self._paths = []
        for lv in starts:
            if self._cancelled():
                return []
            self._climb([], self._lasers, float(lv["level"]),
                        float(lv.get("weight", 1.0)))
        self._progress(f"{len(self._paths)} excitation path(s) found; "
                       "evaluating decay channels...")
        schemes = []
        for path, weight in self._paths:
            if self._cancelled():
                return []
            schemes.extend(self._evaluate_decays(path, weight))
        self._progress(f"{len(schemes)} candidate scheme(s); "
                       "scoring and ranking...")
        ranked = self._rank(schemes)
        if len(ranked) > self.cfg.max_results:
            self._progress(
                f"Keeping top {self.cfg.max_results} of "
                f"{len(ranked)} schemes.")
            ranked = ranked[:self.cfg.max_results]
        return ranked

    # ── search internals ────────────────────────────────────────────

    def _resolve_starting_levels(self) -> list:
        out = []
        if self.cfg.auto_discover:
            out.append({"level": 0.0, "weight": 1.0})
            meta = nd.find_metastable_states(
                self.levels, self.lines,
                self.cfg.metastable_aki_threshold)
            for _, r in meta.iterrows():
                out.append({"level": float(r["Level (cm-1)"]),
                            "weight": 1.0})
            self._progress(f"Auto-discovery: ground state + "
                           f"{len(out) - 1} metastable state(s).")
        else:
            for item in self.cfg.starting_levels:
                lvl = float(item.get("level", 0.0))
                actual, diff = nd.find_closest_level(self.levels, lvl)
                if np.isfinite(actual) and diff <= 1.0:
                    lvl = actual
                elif diff > 1.0:
                    self._progress(
                        f"Warning: level {lvl:.2f} cm-1 not tabulated "
                        f"(closest {actual:.2f}); using input value.")
                out.append({"level": lvl,
                            "weight": float(item.get("weight", 1.0))})
        # Unique by energy
        seen, uniq = set(), []
        for item in out:
            key = round(item["level"], 2)
            if key not in seen:
                seen.add(key)
                uniq.append(item)
        return uniq

    def _climb(self, path: list, lasers_left: list,
               level: float, weight: float) -> None:
        if self._cancelled():
            return
        if len(path) == self.cfg.num_steps:
            self._paths.append((list(path), weight))
            return
        is_last = (len(path) + 1) == self.cfg.num_steps
        role = "probing" if is_last else "pumping"
        ups = self.lines[
            (self.lines["Ei(cm-1)"] - level).abs()
            < self.cfg.level_tolerance]
        ups = ups[ups["Ek(cm-1)"] > ups["Ei(cm-1)"]]
        ups = self._apply_role_filters(ups, role)
        for _, trans in ups.iterrows():
            wl = nd.wavelength_nm(trans, self.cfg.medium)
            if not np.isfinite(wl):
                continue
            if self._use_lasers:
                for i, laser in enumerate(lasers_left):
                    if laser.role == "PUMP" and is_last:
                        continue
                    if laser.role == "PROBE" and not is_last:
                        continue
                    if not laser.can_drive(wl):
                        continue
                    step = SchemeStep(transition_dict(trans),
                                      laser.name)
                    self._climb(path + [step],
                                lasers_left[:i] + lasers_left[i + 1:],
                                float(trans["Ek(cm-1)"]), weight)
            else:
                if not self._in_broad_range(wl):
                    continue
                step = SchemeStep(transition_dict(trans), None)
                self._climb(path + [step], lasers_left,
                            float(trans["Ek(cm-1)"]), weight)

    def _in_broad_range(self, wl: float) -> bool:
        lo, hi = self.cfg.broad_min_nm, self.cfg.broad_max_nm
        return ((lo is None or wl >= lo)
                and (hi is None or wl <= hi))

    def _apply_role_filters(self, df: pd.DataFrame,
                            role: str) -> pd.DataFrame:
        aki_min = (self.cfg.aki_min_probe if role == "probing"
                   else self.cfg.aki_min_pump)
        if aki_min:
            df = df[df["Aki(s^-1)"] >= float(aki_min)]
        orb = (self.cfg.orbital_filter_probe if role == "probing"
               else self.cfg.orbital_filter_pump)
        if orb:
            targets = set(orb)
            mask = df.apply(
                lambda r: nd.orbital_transition(
                    r.get("conf_i", ""), r.get("conf_k", ""))
                in targets, axis=1)
            df = df[mask]
        return df

    def _evaluate_decays(self, path: list, weight: float) -> list:
        final_level = float(path[-1].transition["Ek(cm-1)"])
        decays = self.lines[
            (self.lines["Ek(cm-1)"] - final_level).abs()
            < self.cfg.level_tolerance].dropna(subset=["Aki(s^-1)"])
        if decays.empty:
            return []
        # BR denominator: EVERY known decay channel of the level —
        # the detection-Aki cut only selects which channel we watch.
        total_aki = float(decays["Aki(s^-1)"].sum())
        if total_aki <= 0:
            return []
        watch = decays
        if self.cfg.aki_min_detect:
            watch = watch[watch["Aki(s^-1)"]
                          >= float(self.cfg.aki_min_detect)]
        probe_wl = nd.wavelength_nm(path[-1].transition,
                                    self.cfg.medium)
        out = []
        for _, dec in watch.iterrows():
            br = float(dec["Aki(s^-1)"]) / total_aki
            if br * 100.0 < self.cfg.min_branching_pct:
                continue
            det_wl = nd.wavelength_nm(dec, self.cfg.medium)
            if not self._passes_detection_constraint(probe_wl, det_wl):
                continue
            out.append(Scheme(
                steps=list(path), detection=transition_dict(dec),
                branching_ratio=br,
                start_level=float(
                    path[0].transition.get("Ei(cm-1)") or 0.0),
                weight=weight))
        return out

    def _passes_detection_constraint(self, probe_wl: float,
                                     det_wl: float) -> bool:
        mode = self.cfg.detection_constraint
        if mode not in ("same", "different"):
            return True
        if not (np.isfinite(probe_wl) and np.isfinite(det_wl)):
            return False
        same = abs(probe_wl - det_wl) < self.cfg.detection_proximity_nm
        return same if mode == "same" else not same

    # ── ranking ─────────────────────────────────────────────────────

    def _rank(self, schemes: list) -> list:
        ranked, seen = [], set()
        for scheme in schemes:
            sig = scheme.signature()
            if sig in seen:
                continue
            seen.add(sig)
            issues = self._isobar_issues(scheme)
            score, warnings = self._score(scheme, issues)
            if score <= -90:
                continue    # fatal flaw (missing Aki on a step)
            ranked.append(RankedScheme(scheme, score, issues, warnings))
        ranked.sort(reverse=True)
        return ranked

    def _score(self, scheme: Scheme, issues: list):
        warnings = []
        score = (math.log10(scheme.weight)
                 if scheme.weight > 0 else -10.0)
        for i, step in enumerate(scheme.steps):
            aki = step.transition.get("Aki(s^-1)")
            if aki is None or not np.isfinite(aki) or aki <= 0:
                warnings.append(f"Missing Aki on step {i + 1}")
                return -100.0, warnings
            score += math.log10(aki)
        br = scheme.branching_ratio
        score += math.log10(br) if br > 0 else -10.0
        if issues:
            n_spectra = len({i.isobar for i in issues})
            warnings.append(
                f"Possible contamination from {n_spectra} isobar(s)")
            score -= 5.0 * len(issues)
        return score, warnings

    def _isobar_issues(self, scheme: Scheme) -> list:
        issues = []
        det_wl = nd.wavelength_nm(scheme.detection, self.cfg.medium)
        if not np.isfinite(det_wl):
            return issues
        for iso in self.cfg.isobars:
            name = iso.get("spectrum", "")
            prox = float(iso.get("proximity_nm", 1.0))
            iso_df = self.isobar_lines.get(name)
            if iso_df is None or iso_df.empty:
                continue
            iso_t = nd.transitions_only(iso_df)
            wls = iso_t.apply(
                lambda r: nd.wavelength_nm(r, self.cfg.medium), axis=1)
            # Contamination needs BOTH: one of our lasers can excite
            # the isobar AND the isobar fluoresces near our detection.
            excitable = False
            for step in scheme.steps:
                if step.laser is None:
                    excitable = self._use_lasers is False and any(
                        self._in_broad_range(w) for w in wls
                        if np.isfinite(w))
                else:
                    laser = next((l for l in self._lasers
                                  if l.name == step.laser), None)
                    if laser is not None:
                        excitable = bool(np.any(
                            (wls >= laser.wl_min)
                            & (wls <= laser.wl_max)))
                if excitable:
                    break
            if not excitable:
                continue
            near = iso_t[(wls - det_wl).abs() <= prox]
            for idx, row in near.iterrows():
                issues.append(IsobarIssue(
                    isobar=name, primary_wl_nm=float(det_wl),
                    isobar_wl_nm=float(wls.loc[idx]),
                    isobar_aki=(float(row["Aki(s^-1)"])
                                if pd.notna(row.get("Aki(s^-1)"))
                                else None)))
        return issues
