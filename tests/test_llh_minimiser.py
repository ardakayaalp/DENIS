"""Picking a minimiser that can actually minimise a likelihood.

"I tried to fit Yb171 with leastsquare with chi square and poisson, chi
square fits but poisson cannot. The counts on the plots looks weird."
(Arda, 2026-09-25)

Nothing was wrong with the data. Least squares minimises a residual
*vector*; a likelihood is a single number, so the two cannot be
combined. satlas2 does not say so -- `core.py` swaps in SLSQP behind
the user's back:

    if llh or method.lower() == 'emcee':
        ...
        if method.lower() in ['leastsq', 'least_squares']:
            method = 'slsqp'

and on run 7507 SLSQP walked off to centroid 1.0e12 MHz, A_l -9659,
redchi 1.1e8, reporting success. The model was then drawn hundreds of
GHz away from the spectrum, which is what "the counts look weird" was.
Nelder-Mead on the same run gives A_l 12637.36 against the 12637.23 of
Arda's own emcee pipeline.

So DENIS substitutes a working minimiser instead of a broken one, and
puts a line in the fit report saying it did.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_llh_minimiser.py -q
"""
import inspect
import unittest

from gui.analysis.fitting import (
    LLH_FALLBACK_METHOD, resolve_llh_method)


def _cfg(**kw):
    cfg = {"method": "leastsq", "llh": False, "llh_method": "poisson"}
    cfg.update(kw)
    return cfg


class ChiSquareIsLeftAloneTests(unittest.TestCase):
    """Without a likelihood every method is legitimate."""

    def test_least_squares_stays(self):
        method, note = resolve_llh_method(_cfg())
        self.assertEqual(method, "leastsq")
        self.assertEqual(note, "")

    def test_any_other_method_stays(self):
        for name in ("least_squares", "nelder", "powell", "emcee", "slsqp"):
            with self.subTest(method=name):
                method, note = resolve_llh_method(_cfg(method=name))
                self.assertEqual(method, name)
                self.assertEqual(note, "")

    def test_a_missing_llh_key_is_chi_square(self):
        method, note = resolve_llh_method({"method": "leastsq"})
        self.assertEqual((method, note), ("leastsq", ""))


class LikelihoodSubstitutionTests(unittest.TestCase):

    def test_leastsq_is_replaced(self):
        method, note = resolve_llh_method(_cfg(llh=True))
        self.assertEqual(method, LLH_FALLBACK_METHOD)
        self.assertTrue(note)

    def test_least_squares_is_replaced(self):
        method, _note = resolve_llh_method(
            _cfg(llh=True, method="least_squares"))
        self.assertEqual(method, LLH_FALLBACK_METHOD)

    def test_the_spelling_does_not_matter(self):
        for name in ("LeastSq", "LEASTSQ", "Least_Squares"):
            with self.subTest(method=name):
                method, note = resolve_llh_method(_cfg(llh=True, method=name))
                self.assertEqual(method, LLH_FALLBACK_METHOD)
                self.assertTrue(note)

    def test_a_missing_method_key_defaults_to_leastsq(self):
        """The Fitter block's default, so this is the common case."""
        method, note = resolve_llh_method({"llh": True,
                                           "llh_method": "poisson"})
        self.assertEqual(method, LLH_FALLBACK_METHOD)
        self.assertTrue(note)

    def test_the_fallback_is_a_scalar_minimiser(self):
        self.assertIn(LLH_FALLBACK_METHOD, ("nelder", "powell"))

    def test_a_scalar_minimiser_is_kept(self):
        for name in ("nelder", "powell", "emcee", "slsqp", "cobyla"):
            with self.subTest(method=name):
                method, note = resolve_llh_method(_cfg(llh=True, method=name))
                self.assertEqual(method, name)
                self.assertEqual(note, "")

    def test_the_note_names_the_substitute_and_the_alternatives(self):
        _method, note = resolve_llh_method(_cfg(llh=True))
        self.assertIn("leastsq", note)
        self.assertIn(LLH_FALLBACK_METHOD, note)
        self.assertIn("powell", note)
        self.assertIn("emcee", note)

    def test_a_gaussian_likelihood_counts_too(self):
        """It is `llh`, not the llh_method, that decides."""
        method, _note = resolve_llh_method(
            _cfg(llh=True, llh_method="gaussian"))
        self.assertEqual(method, LLH_FALLBACK_METHOD)


class BothFitPathsUseItTests(unittest.TestCase):
    """A fit that bypassed the resolver would be back to SLSQP."""

    def _source(self, name):
        import gui.analysis.fitting as f
        obj = getattr(f, name)
        return inspect.getsource(obj)

    def test_the_per_run_fit_resolves_the_method(self):
        src = self._source("_fit_single_run")
        self.assertIn("resolve_llh_method(fitter_config)", src)
        head = src[:src.index("resolve_llh_method")]
        self.assertNotIn('fit_kwargs = {"method"', head)

    def test_the_simultaneous_fit_resolves_the_method(self):
        from gui.analysis.fitting import FitWorkerThread
        src = inspect.getsource(FitWorkerThread._run_simultaneous)
        self.assertIn("resolve_llh_method(self.fitter_config)", src)

    def test_both_paths_put_the_note_in_the_report(self):
        from gui.analysis.fitting import FitWorkerThread
        for src in (self._source("_fit_single_run"),
                    inspect.getsource(FitWorkerThread._run_simultaneous)):
            with self.subTest():
                self.assertIn("[Minimiser]", src)

    def test_the_resolved_method_is_what_is_passed_to_satlas2(self):
        for src in (self._source("_fit_single_run"),
                    inspect.getsource(
                        __import__("gui.analysis.fitting", fromlist=["x"])
                        .FitWorkerThread._run_simultaneous)):
            with self.subTest():
                self.assertIn('"method": _method', src)


class Satlas2StillDoesItTests(unittest.TestCase):
    """If satlas2 ever stops substituting, this workaround can go."""

    def test_satlas2_swaps_leastsq_for_slsqp_under_a_likelihood(self):
        try:
            from satlas2.core import Fitter
        except Exception as exc:                            # noqa: BLE001
            self.skipTest(f"satlas2 unavailable: {exc}")
        src = inspect.getsource(Fitter.fit)
        self.assertIn("slsqp", src)
        self.assertIn("['leastsq', 'least_squares']", src)


class FitterBlockHintTests(unittest.TestCase):
    """The block says which minimiser will run before it runs."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _block(self, method="leastsq", statistics="Chi-square"):
        from gui.analysis.blocks import FitterBlock
        block = FitterBlock()
        self.addCleanup(block.deleteLater)
        # Never rely on the app settings for the starting point: they
        # carry the user own Fitting Defaults.
        block._method_combo.setCurrentText(method)
        block._stats_combo.setCurrentText(statistics)
        return block

    def test_chi_square_shows_nothing(self):
        block = self._block()
        self.assertFalse(block._llh_method_hint.isVisible())

    def test_a_likelihood_with_leastsq_is_flagged(self):
        block = self._block(statistics="Poisson LLH")
        self.assertTrue(block._llh_method_hint.isVisibleTo(block))
        text = block._llh_method_hint.text()
        self.assertIn("cannot minimise a likelihood", text)
        self.assertIn(LLH_FALLBACK_METHOD, text)

    def test_the_substitution_warning_clears_when_the_method_is_fixed(self):
        block = self._block(statistics="Gaussian LLH")
        self.assertIn("cannot minimise", block._llh_method_hint.text())
        block._method_combo.setCurrentText(LLH_FALLBACK_METHOD)
        self.assertNotIn("cannot minimise", block._llh_method_hint.text())
        # What stays is the error-bar warning (test_llh_error_correction).
        self.assertIn("√2", block._llh_method_hint.text())

    def test_emcee_under_a_likelihood_needs_no_warning(self):
        block = self._block(statistics="Poisson LLH")
        block._method_combo.setCurrentText("emcee")
        self.assertFalse(block._llh_method_hint.isVisibleTo(block))

    def test_the_flag_clears_when_the_statistics_is_fixed(self):
        block = self._block(statistics="Poisson LLH")
        block._stats_combo.setCurrentText("Chi-square")
        self.assertFalse(block._llh_method_hint.isVisibleTo(block))

    def test_loading_a_project_flags_it_too(self):
        block = self._block()
        block.from_dict({"method": "leastsq",
                         "statistics": "Poisson LLH"})
        self.assertTrue(block._llh_method_hint.isVisibleTo(block))

    def test_the_saved_config_keeps_what_the_user_picked(self):
        """The substitution belongs to the fit, not to the session."""
        block = self._block(statistics="Poisson LLH")
        cfg = block.get_fitter_config()
        self.assertEqual(cfg["method"], "leastsq")
        self.assertTrue(cfg["llh"])
        self.assertEqual(block.to_dict()["method"], "leastsq")
        # ... and the fit resolves it there.
        self.assertEqual(resolve_llh_method(cfg)[0], LLH_FALLBACK_METHOD)

if __name__ == "__main__":
    unittest.main()
