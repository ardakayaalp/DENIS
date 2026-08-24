"""Phase 5.3 (UI overhaul): Update iteration — post-run plot backfill.

Date:    2026-07-24
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins ``AnalysisProject._rebuild_chain_diagnostics``: newly-ticked MCMC
walk / correlation plots are rebuilt from the persisted chain ``.h5``
(original path or the iteration's ``chains/`` copy), with the vary-mask
recovered from the chain itself (fixed parameters have zero spread);
artifacts that genuinely need a re-run (chi-square map, confidence
bands, or walk/correlation with no saved chain) are reported, not
silently dropped.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_update_iteration -v

Depends on: gui.analysis.project (AnalysisProject); PySide6, numpy, h5py.
"""

import os
import tempfile
import unittest

import numpy as np

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _write_chain(path, steps=20, walkers=8):
    """A fake satlas2-style chain: 3 params, the middle one FIXED
    (constant column). Labels carry the source___param convention."""
    import h5py
    rng = np.random.default_rng(42)
    chain = np.empty((steps, walkers, 3))
    chain[:, :, 0] = rng.normal(0.0, 1.0, (steps, walkers))
    chain[:, :, 1] = 5.0                       # fixed -> zero spread
    chain[:, :, 2] = rng.normal(-2.0, 0.5, (steps, walkers))
    with h5py.File(path, "w") as hf:
        g = hf.create_group("mcmc")
        g.create_dataset("chain", data=chain)
        g.attrs["labels"] = ["src___centroid", "src___scale", "FWHMG"]
    return chain


class RebuildChainDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.analysis.project import AnalysisProject
        self.proj = AnalysisProject("U")

    def tearDown(self):
        self.proj.deleteLater()

    def test_walk_and_correl_rebuilt_from_chain(self):
        with tempfile.TemporaryDirectory() as td:
            chain_path = os.path.join(td, "chain_run_1.h5")
            _write_chain(chain_path)
            r = {"success": True, "run_number": 1,
                 "chain_file": chain_path, "diagnostics": {}}
            skipped = self.proj._rebuild_chain_diagnostics(
                [r], {"walk_plot": True, "correl_plot": True,
                      "chisq_map": True})
            # Chi-square map needs the live fitter -> reported.
            self.assertTrue(any("chi-square" in s for s in skipped))
            wd = r["diagnostics"]["walk_data"]
            # Fixed param (zero spread) dropped; source prefix stripped.
            self.assertEqual(wd["labels"], ["centroid", "FWHMG"])
            self.assertEqual(wd["chain"].shape, (20, 8, 2))
            cd = r["diagnostics"]["correl_data"]
            self.assertEqual(cd["flatchain"].shape, (160, 2))

    def test_falls_back_to_iteration_chains_copy(self):
        with tempfile.TemporaryDirectory() as td:
            iter_dir = os.path.join(td, "iter_001")
            os.makedirs(os.path.join(iter_dir, "chains"))
            _write_chain(os.path.join(iter_dir, "chains",
                                      "chain_run_7.h5"))
            self.proj._last_iter_dir = iter_dir
            r = {"success": True, "run_number": 7,
                 "chain_file": None, "diagnostics": {}}
            skipped = self.proj._rebuild_chain_diagnostics(
                [r], {"walk_plot": True})
            self.assertEqual(skipped, [])
            self.assertIn("walk_data", r["diagnostics"])

    def test_no_chain_is_reported_not_silent(self):
        r = {"success": True, "run_number": 3,
             "chain_file": None, "diagnostics": {}}
        self.proj._last_iter_dir = None
        skipped = self.proj._rebuild_chain_diagnostics(
            [r], {"walk_plot": True})
        self.assertTrue(any("no saved MCMC chain" in s for s in skipped))
        self.assertNotIn("walk_data", r["diagnostics"])

    def test_diag_param_filter_applies(self):
        with tempfile.TemporaryDirectory() as td:
            chain_path = os.path.join(td, "c.h5")
            _write_chain(chain_path)
            r = {"success": True, "run_number": 1,
                 "chain_file": chain_path, "diagnostics": {}}
            self.proj._rebuild_chain_diagnostics(
                [r], {"walk_plot": True, "diag_params": ["FWHMG"]})
            self.assertEqual(r["diagnostics"]["walk_data"]["labels"],
                             ["FWHMG"])

    def test_existing_data_left_untouched(self):
        sentinel = {"labels": ["x"], "chain": np.zeros((1, 1, 1))}
        r = {"success": True, "run_number": 1, "chain_file": None,
             "diagnostics": {"walk_data": sentinel}}
        skipped = self.proj._rebuild_chain_diagnostics(
            [r], {"walk_plot": True})
        self.assertIs(r["diagnostics"]["walk_data"], sentinel)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
