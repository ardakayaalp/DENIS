"""Tests for WP3: the Auto-Fitter honours the user's fit objective + constraints.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Before WP3 the Auto-Fitter always ran plain chi2 leastsq and dropped priors,
fitter-block expressions, and 'All'-mode sharing, so an accepted seed was not a
fixed point of a fit run under Poisson/Gaussian LLH. _run_one_fit now takes the
fitter_config and forwards the objective (method / llh / llh_method) and applies
fitter-block expressions / priors. These tests spy on satlas2.Fitter.fit to
prove the objective actually reaches the optimizer, confirm back-compat when no
config is given, and smoke-test the fitter-block expression path.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_wp3_autofitter_objective -v

Depends on: gui.analysis.auto_fitter (_run_one_fit); requires satlas2.
"""

import numpy as np
import pytest

satlas2 = pytest.importorskip("satlas2")

from gui.analysis.auto_fitter import _run_one_fit
from gui.analysis.naming import full_param_name


def _voigt_config():
    def p(value, vary=True, expr=""):
        return {"value": value, "vary": vary, "min": None, "max": None,
                "expr": expr}
    return {"type": "Voigt", "name": "M1", "params": {
        "A": p(8.0), "mu": p(0.0), "FWHMG": p(40.0), "FWHML": p(40.0),
        "Bkg_p0": p(1.0)}}


def _make_data():
    x = np.linspace(-200.0, 200.0, 120)
    v = satlas2.Voigt(8.0, 0.0, 40.0, 40.0, name="M1")
    y = v.f(x) + 5.0
    return {"name": "Run_1", "x": x, "y": y,
            "yerr": np.sqrt(np.abs(y) + 1.0)}


_VK = [(0, "A"), (0, "mu"), (0, "FWHMG"), (0, "FWHML")]
_INIT = [6.0, 5.0, 30.0, 30.0]


def _spy_fit(monkeypatch):
    captured = {}
    orig = satlas2.Fitter.fit

    def spy(self, *a, **kw):
        captured.clear()
        captured.update(kw)
        return orig(self, *a, **kw)

    monkeypatch.setattr(satlas2.Fitter, "fit", spy)
    return captured


def test_llh_objective_is_forwarded_to_the_fit(monkeypatch):
    captured = _spy_fit(monkeypatch)
    res = _run_one_fit(
        [_make_data()], [_voigt_config()], _INIT, _VK,
        fitter_config={"method": "leastsq", "llh": True,
                       "llh_method": "poisson"})
    assert res["success"], res.get("error")
    assert captured.get("llh") is True
    assert captured.get("llh_method") == "poisson"
    assert captured.get("scale_covar") is False  # forced under llh


def test_no_config_keeps_legacy_chi2_leastsq(monkeypatch):
    captured = _spy_fit(monkeypatch)
    res = _run_one_fit([_make_data()], [_voigt_config()], _INIT, _VK)
    assert res["success"], res.get("error")
    assert captured.get("method") == "leastsq"
    assert captured.get("scale_covar") is False
    assert "llh" not in captured  # chi2 path, no likelihood objective


def test_emcee_method_falls_back_to_leastsq_for_search(monkeypatch):
    captured = _spy_fit(monkeypatch)
    res = _run_one_fit(
        [_make_data()], [_voigt_config()], _INIT, _VK,
        fitter_config={"method": "emcee"})
    assert res["success"], res.get("error")
    assert captured.get("method") == "leastsq"


def test_fitter_block_expression_is_applied(monkeypatch):
    # A fitter-block expression (full names) must be applied, not dropped.
    # Spy on setExpr to confirm it is called with the configured tie.
    seen = []
    orig = satlas2.Fitter.setExpr

    def spy(self, name, expr):
        seen.append((name, expr))
        return orig(self, name, expr)

    monkeypatch.setattr(satlas2.Fitter, "setExpr", spy)
    mu_full = full_param_name("Run_1", "M1", "mu")
    res = _run_one_fit(
        [_make_data()], [_voigt_config()],
        [6.0, 30.0, 30.0], [(0, "A"), (0, "FWHMG"), (0, "FWHML")],
        fitter_config={"expressions": {mu_full: "0.0"}})
    assert res["success"], res.get("error")
    assert any(n == mu_full and e == "0.0" for n, e in seen)


if __name__ == "__main__":
    import unittest
    unittest.main()
