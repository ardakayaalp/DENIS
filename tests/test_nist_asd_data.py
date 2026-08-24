"""NIST ASD data layer: parsing, cache, derived physics.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the M1 contracts of gui.nist_asd.data against trimmed LIVE
captures from 2026-07-25 (tests/data/nist_si_i_{lines,levels}.tsv):

- tab-delimited parsing (quoted cells, trailing-tab ghost column);
- cleaning: numeric coercion, blank Type → "E1" (ASD labels only
  forbidden lines), observed-only rows kept, transitions_only();
- error-page detection raises NistFetchError (no silent HTML parse);
- CSV cache round-trip with meta, offline load;
- derive_levels_from_lines fallback;
- vacuum→air conversion (NIST five-figure formula) and
  wavelength_nm() fallback to 1e7/ΔE;
- orbital parsing/classification incl. jj-coupling configs with
  parentheses;
- metastable finder honors the blank-Type-is-E1 normalization.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_nist_asd_data.py -q

Depends on: gui.nist_asd.data (Qt-free), pandas, numpy.
"""

import math
import os
import tempfile
import unittest

import pandas as pd

from gui.nist_asd import data as nd

_HERE = os.path.dirname(os.path.abspath(__file__))
_LINES_TSV = os.path.join(_HERE, "data", "nist_si_i_lines.tsv")
_LEVELS_TSV = os.path.join(_HERE, "data", "nist_si_i_levels.tsv")


def _lines_df():
    with open(_LINES_TSV, encoding="utf-8") as f:
        return nd.clean_lines_df(nd.parse_tab_delimited(f.read()))


def _levels_df():
    with open(_LEVELS_TSV, encoding="utf-8") as f:
        return nd.clean_levels_df(nd.parse_tab_delimited(f.read()))


class ParsingTests(unittest.TestCase):
    def test_lines_parse_and_clean(self):
        df = _lines_df()
        # Ghost trailing column dropped, headers unquoted.
        self.assertNotIn("Unnamed: 16", df.columns)
        for col in ("ritz_wl_vac(nm)", "Aki(s^-1)", "Ei(cm-1)",
                    "Ek(cm-1)"):
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), col)
        # Observed-only rows kept in the browse table...
        self.assertTrue(df["Ei(cm-1)"].isna().any())
        # ...but excluded from the physics view.
        t = nd.transitions_only(df)
        self.assertFalse(t["Ei(cm-1)"].isna().any())
        self.assertGreater(len(df), len(t))

    def test_blank_type_normalized_to_e1(self):
        df = _lines_df()
        self.assertIn("E1", set(df["Type"]))
        self.assertNotIn("", set(df["Type"]))
        # The fixture carries real forbidden lines too.
        self.assertIn("E2", set(df["Type"]))

    def test_levels_parse(self):
        df = _levels_df()
        self.assertIn("Level (cm-1)", df.columns)
        self.assertNotIn("Prefix", df.columns)
        self.assertAlmostEqual(float(df.iloc[0]["Level (cm-1)"]), 0.0)
        # Si I ground term fine structure present.
        self.assertTrue((df["Level (cm-1)"] - 77.115).abs().min() < 1e-6)

    def test_error_page_raises(self):
        html = ("<html><body>NIST ASD : Input Error Error Message: "
                "Unknown parameter</body></html>")
        with self.assertRaises(nd.NistFetchError) as ctx:
            nd._raise_if_error_page(html, "Xx I")
        self.assertIn("Unknown parameter", str(ctx.exception))


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_offline(self):
        df = _lines_df()
        with tempfile.TemporaryDirectory() as d:
            path = nd.save_cache("Si I", "lines", df, directory=d)
            self.assertTrue(os.path.isfile(path))
            back = nd.load_cache("Si I", "lines", directory=d)
            self.assertEqual(len(back), len(df))
            # Cleaning is re-applied on load (Type normalization).
            self.assertIn("E1", set(back["Type"]))
            info = nd.cache_info("Si I", "lines", directory=d)
            self.assertEqual(info["rows"], len(df))
            self.assertEqual(info["spectrum"], "Si I")
            # get_lines() must not touch the network on a warm cache.
            got = nd.get_lines("Si I", refresh=False, directory=d)
            self.assertEqual(len(got), len(df))

    def test_load_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(nd.load_cache("Nb II", "lines",
                                            directory=d))


class DerivedTests(unittest.TestCase):
    def test_derive_levels_from_lines(self):
        levels = nd.derive_levels_from_lines(_lines_df())
        self.assertIn("Level (cm-1)", levels.columns)
        vals = levels["Level (cm-1)"].tolist()
        self.assertEqual(vals, sorted(vals))
        self.assertIn(0.0, vals)                      # ground state
        self.assertEqual(len(set(vals)), len(vals))   # unique

    def test_vac_to_air_matches_known_value(self):
        # NIST: 589.158 nm (vac) ≈ 588.995 nm (air) for the Na D2 line.
        self.assertAlmostEqual(nd.vac_to_air(589.158), 588.995,
                               places=2)
        self.assertTrue(math.isnan(nd.vac_to_air(float("nan"))))

    def test_wavelength_nm_prefers_ritz_then_energy(self):
        row = {"ritz_wl_vac(nm)": 251.43156,
               "Ei(cm-1)": 0.0, "Ek(cm-1)": 39760.285}
        self.assertAlmostEqual(nd.wavelength_nm(row, "vacuum"),
                               251.43156)
        self.assertLess(nd.wavelength_nm(row, "air"),
                        nd.wavelength_nm(row, "vacuum"))
        row_no_ritz = {"ritz_wl_vac(nm)": float("nan"),
                       "Ei(cm-1)": 0.0, "Ek(cm-1)": 40000.0}
        self.assertAlmostEqual(nd.wavelength_nm(row_no_ritz, "vacuum"),
                               1e7 / 40000.0)

    def test_orbital_parsing_and_classification(self):
        self.assertEqual(nd.parse_orbital("3s2.3p.4d"), "d")
        self.assertEqual(nd.parse_orbital("3s2.3p2"), "p")
        self.assertIsNone(nd.parse_orbital(""))
        self.assertEqual(nd.orbital_transition("3s2.3p2", "3s2.3p.4s"),
                         "p->s")
        self.assertEqual(nd.orbital_transition("3s2.3p2", "3s2.3p.4d"),
                         "p->d")
        # jj-coupling parent term in parentheses parses fine.
        self.assertEqual(
            nd.orbital_transition("3s2.3p2",
                                  "3s2.3p.(2P*<3/2>).4f"), "p->f")
        self.assertIsNone(nd.orbital_transition("", "3s2.3p.4s"))

    def test_metastable_finder_blank_type_semantics(self):
        levels = pd.DataFrame({
            "Level (cm-1)": [0.0, 6298.85, 40000.0],
            "Configuration": ["3s2.3p2", "3s2.3p2", "3s2.3p.4s"],
            "Term": ["3P", "1D", "3P*"], "J": ["0", "2", "1"],
        })
        lines = nd.clean_lines_df(pd.DataFrame({
            # 1D decays only via a slow E2 line -> metastable;
            # 40000 decays via a fast blank-Type (=E1) line -> not.
            "Ei(cm-1)": [0.0, 0.0],
            "Ek(cm-1)": [6298.85, 40000.0],
            "Aki(s^-1)": [1.0, 7.0e7],
            "Type": ["E2", ""],
            "conf_i": ["3s2.3p2"] * 2, "term_i": ["3P"] * 2,
            "J_i": ["0"] * 2,
            "conf_k": ["3s2.3p2", "3s2.3p.4s"],
            "term_k": ["1D", "3P*"], "J_k": ["2", "1"],
        }))
        meta = nd.find_metastable_states(levels, lines,
                                         fast_aki_threshold=1.0e4)
        self.assertEqual(len(meta), 1)
        self.assertAlmostEqual(float(meta.iloc[0]["Level (cm-1)"]),
                               6298.85)

    def test_find_closest_level(self):
        levels = _levels_df()
        lvl, diff = nd.find_closest_level(levels, 77.0)
        self.assertAlmostEqual(lvl, 77.115)
        self.assertLess(diff, 0.2)


if __name__ == "__main__":
    unittest.main()
