"""NIST ASD scheme search: pathfinding, filters, scoring, round-trips.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Exercises gui.nist_asd.search on a SYNTHETIC four-level system where
every expected scheme and score is known analytically:

    0 ──500 nm──> 20000 ──400 nm──> 45000
    0 ──222 nm───────────────────-> 45000
    45000 decays: → 0 (Aki 8e7, BR 0.8) and → 20000 (Aki 2e7, BR 0.2)
    100 cm-1 is metastable (only a slow M1 decay).

Pins: 2-step search with PUMP/PROBE roles; each-laser-used-once and
role placement; broad-wavelength mode; per-role Aki cuts; min-BR cut;
detection same/different constraint; isobar contamination flag +
score penalty; physical-pathway dedup across laser assignments;
weighted-mean score formula; auto-discovery of metastables; config and
RankedScheme dict round-trips.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_nist_asd_search.py -q

Depends on: gui.nist_asd.{data,models,search}; pandas, numpy.
"""

import math
import unittest

import pandas as pd

from gui.nist_asd import data as nd
from gui.nist_asd.models import Laser, RankedScheme
from gui.nist_asd.search import SchemeSearchConfig, SchemeSearcher


def _line(Ei, Ek, aki, conf_i, conf_k, typ=""):
    return {"obs_wl_vac(nm)": None, "ritz_wl_vac(nm)": None,
            "intens": "", "Aki(s^-1)": aki, "Acc": "A",
            "Ei(cm-1)": Ei, "Ek(cm-1)": Ek,
            "conf_i": conf_i, "term_i": "X", "J_i": "1",
            "conf_k": conf_k, "term_k": "Y", "J_k": "1",
            "Type": typ}


def _system():
    lines = nd.clean_lines_df(pd.DataFrame([
        _line(0.0, 20000.0, 5.0e7, "3s2.3p2", "3s2.3p.4s"),
        _line(20000.0, 45000.0, 2.0e7, "3s2.3p.4s", "3s2.3p.4p"),
        _line(0.0, 45000.0, 8.0e7, "3s2.3p2", "3s2.3p.4p"),
        _line(0.0, 100.0, 1.0e-2, "3s2.3p2", "3s2.3p2", typ="M1"),
    ]))
    levels = nd.derive_levels_from_lines(lines)
    return lines, levels


def _two_step_cfg(**kw):
    cfg = SchemeSearchConfig(
        spectrum="Xx I",
        starting_levels=[{"level": 0.0, "weight": 1.0}],
        lasers=[Laser("pump", 500.0, 5.0, "PUMP").to_dict(),
                Laser("probe", 400.0, 5.0, "PROBE").to_dict()],
        num_steps=2, min_branching_pct=1.0, medium="vacuum")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TwoStepSearchTests(unittest.TestCase):
    def test_finds_ladder_and_scores_it(self):
        lines, levels = _system()
        ranked = SchemeSearcher(_two_step_cfg(), lines, levels).run()
        # Two detection channels of the same excitation path.
        self.assertEqual(len(ranked), 2)
        best = ranked[0]
        self.assertAlmostEqual(best.scheme.branching_ratio, 0.8)
        self.assertEqual(best.scheme.steps[0].laser, "pump")
        self.assertEqual(best.scheme.steps[1].laser, "probe")
        expected = (math.log10(5e7) + math.log10(2e7)
                    + math.log10(0.8))
        self.assertAlmostEqual(best.score, expected, places=6)
        # Ranked descending.
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_roles_are_enforced(self):
        lines, levels = _system()
        # A single PROBE-only laser wide enough for both steps: it may
        # not drive step 1, so no 2-step path can complete.
        cfg = _two_step_cfg(lasers=[
            Laser("wide", 450.0, 60.0, "PROBE").to_dict()])
        self.assertEqual(SchemeSearcher(cfg, lines, levels).run(), [])
        # Same laser with role ANY: allowed on step 1, but each laser
        # is used at most once -> still no complete 2-step path.
        cfg = _two_step_cfg(lasers=[
            Laser("wide", 450.0, 60.0, "ANY").to_dict()])
        self.assertEqual(SchemeSearcher(cfg, lines, levels).run(), [])

    def test_detection_constraint_different(self):
        lines, levels = _system()
        cfg = _two_step_cfg(detection_constraint="different",
                            detection_proximity_nm=1.0)
        ranked = SchemeSearcher(cfg, lines, levels).run()
        # The BR-0.2 channel re-emits along the probe (same lambda)
        # and is rejected; only the 222 nm channel survives.
        self.assertEqual(len(ranked), 1)
        self.assertAlmostEqual(ranked[0].scheme.branching_ratio, 0.8)

    def test_min_branching_cut(self):
        lines, levels = _system()
        cfg = _two_step_cfg(min_branching_pct=50.0)
        ranked = SchemeSearcher(cfg, lines, levels).run()
        self.assertEqual([round(r.scheme.branching_ratio, 2)
                          for r in ranked], [0.8])

    def test_probe_aki_cut_removes_path(self):
        lines, levels = _system()
        cfg = _two_step_cfg(aki_min_probe=3.0e7)
        self.assertEqual(SchemeSearcher(cfg, lines, levels).run(), [])

    def test_orbital_filter_on_probe(self):
        lines, levels = _system()
        cfg = _two_step_cfg(orbital_filter_probe=["s->p"])
        self.assertEqual(len(SchemeSearcher(cfg, lines,
                                            levels).run()), 2)
        cfg = _two_step_cfg(orbital_filter_probe=["s->d"])
        self.assertEqual(SchemeSearcher(cfg, lines, levels).run(), [])

    def test_dedup_across_laser_assignments(self):
        lines, levels = _system()
        # Two interchangeable ANY lasers that can each drive either
        # step -> the same physical pathway appears once, not twice.
        cfg = _two_step_cfg(lasers=[
            Laser("A", 450.0, 60.0, "ANY").to_dict(),
            Laser("B", 450.0, 60.0, "ANY").to_dict()])
        ranked = SchemeSearcher(cfg, lines, levels).run()
        sigs = [r.scheme.signature() for r in ranked]
        self.assertEqual(len(sigs), len(set(sigs)))
        self.assertEqual(len(ranked), 2)   # two detection channels


class BroadRangeAndOneStepTests(unittest.TestCase):
    def test_broad_range_two_step(self):
        lines, levels = _system()
        cfg = _two_step_cfg(lasers=[], broad_min_nm=350.0,
                            broad_max_nm=550.0)
        ranked = SchemeSearcher(cfg, lines, levels).run()
        self.assertEqual(len(ranked), 2)
        self.assertIsNone(ranked[0].scheme.steps[0].laser)

    def test_one_step_probe_within_range(self):
        lines, levels = _system()
        cfg = _two_step_cfg(lasers=[], broad_min_nm=350.0,
                            broad_max_nm=550.0, num_steps=1)
        ranked = SchemeSearcher(cfg, lines, levels).run()
        # Only 0->20000 (500 nm) is in range; its sole decay channel
        # returns on the same line with BR 1.
        self.assertEqual(len(ranked), 1)
        self.assertAlmostEqual(ranked[0].scheme.branching_ratio, 1.0)
        self.assertAlmostEqual(
            float(ranked[0].scheme.steps[0].transition["Ek(cm-1)"]),
            20000.0)


class IsobarAndDiscoveryTests(unittest.TestCase):
    def test_isobar_contamination_penalty(self):
        lines, levels = _system()
        # Isobar excitable by the pump (line at 500 nm) AND fluorescing
        # right at our 222.22 nm detection wavelength.
        det_wl = 1e7 / 45000.0
        iso = nd.clean_lines_df(pd.DataFrame([
            _line(0.0, 20000.0, 1.0e7, "3s2.3p2", "3s2.3p.4s"),
            _line(0.0, 1e7 / det_wl, 5.0e6, "3s2.3p2", "3s2.3p.4p"),
        ]))
        cfg = _two_step_cfg(
            isobars=[{"spectrum": "Yy I", "proximity_nm": 1.0}],
            detection_constraint="different")
        clean = SchemeSearcher(cfg, lines, levels).run()
        dirty = SchemeSearcher(cfg, lines, levels,
                               isobar_lines={"Yy I": iso}).run()
        self.assertEqual(len(clean), 1)
        self.assertEqual(len(dirty), 1)
        self.assertEqual(len(dirty[0].issues), 1)
        self.assertEqual(dirty[0].issues[0].isobar, "Yy I")
        self.assertAlmostEqual(clean[0].score - dirty[0].score, 5.0)
        self.assertTrue(any("contamination" in w
                            for w in dirty[0].warnings))

    def test_auto_discovery_includes_metastable(self):
        lines, levels = _system()
        cfg = _two_step_cfg(auto_discover=True,
                            metastable_aki_threshold=1.0e4)
        searcher = SchemeSearcher(cfg, lines, levels)
        starts = searcher._resolve_starting_levels()
        vals = sorted(s["level"] for s in starts)
        self.assertIn(0.0, vals)
        self.assertIn(100.0, vals)

    def test_weight_shifts_score(self):
        lines, levels = _system()
        cfg = _two_step_cfg(
            starting_levels=[{"level": 0.0, "weight": 0.1}])
        ranked = SchemeSearcher(cfg, lines, levels).run()
        expected = (-1.0 + math.log10(5e7) + math.log10(2e7)
                    + math.log10(0.8))
        self.assertAlmostEqual(ranked[0].score, expected, places=6)


class RoundTripTests(unittest.TestCase):
    def test_config_round_trip(self):
        cfg = _two_step_cfg(isobars=[{"spectrum": "P I",
                                      "proximity_nm": 2.0}],
                            orbital_filter_probe=["s->p"])
        back = SchemeSearchConfig.from_dict(cfg.to_dict())
        self.assertEqual(back.to_dict(), cfg.to_dict())
        self.assertEqual([l.name for l in back.laser_objects()],
                         ["pump", "probe"])

    def test_ranked_scheme_round_trip(self):
        lines, levels = _system()
        ranked = SchemeSearcher(_two_step_cfg(), lines, levels).run()
        d = ranked[0].to_dict()
        back = RankedScheme.from_dict(d)
        self.assertEqual(back.to_dict(), d)
        self.assertAlmostEqual(back.scheme.branching_ratio,
                               ranked[0].scheme.branching_ratio)
        self.assertEqual(back.scheme.signature(),
                         ranked[0].scheme.signature())


if __name__ == "__main__":
    unittest.main()
