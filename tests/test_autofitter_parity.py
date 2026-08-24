"""Auto-Fitter builds models through the canonical fitter builder.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

The Auto-Fitter must construct models with fitting._build_models_on_source,
the same builder the regular Run uses, so it honours parameter bounds,
every model type, and model-block expression ties. The final test
asserts the user-facing contract: a parameter set the Auto-Fitter
accepts is a fixed point of the regular fitter (re-fitting from those
values does not move them and the Al=Au expression tie still holds).
These tests drive the subprocess entry point ``_run_one_fit`` directly
(headless, no Qt) so they exercise the real build + fit path.

Depends on: gui.analysis.auto_fitter (_run_one_fit),
gui.analysis.fitting (_build_models_on_source), gui.analysis.naming,
gui.analysis.expr_validation; requires the satlas2 fitting package.
"""
import numpy as np
import pytest

satlas2 = pytest.importorskip("satlas2")

from gui.analysis.auto_fitter import _run_one_fit


def _voigt_config(amin=None, amax=None, mu_expr=""):
    """A single-Voigt model config matching ModelBlock.get_model_config()."""
    def p(value, vary=True, mn=None, mx=None, expr=""):
        return {"value": value, "vary": vary, "min": mn, "max": mx,
                "expr": expr}
    return {
        "type": "Voigt", "name": "M1",
        "params": {
            "A": p(8.0, mn=amin, mx=amax),
            "mu": p(0.0, expr=mu_expr),
            "FWHMG": p(40.0),
            "FWHML": p(40.0),
            "Bkg_p0": p(1.0),
        },
    }


def _make_data(a=8.0, mu=0.0, fg=40.0, fl=40.0, bkg=1.0):
    x = np.linspace(-200.0, 200.0, 120)
    v = satlas2.Voigt(a, mu, fg, fl, name="M1")
    y = v.f(x) + bkg
    yerr = np.sqrt(np.abs(y) + 1.0)
    return {"name": "Run_1", "x": x, "y": y, "yerr": yerr}


def test_fit_runs_and_converges_via_canonical_builder():
    """A plain Voigt fit through _run_one_fit succeeds and recovers the
    amplitude (proves the canonical builder path is wired and works)."""
    sd = _make_data(a=8.0)
    cfg = _voigt_config()
    vary_keys = [(0, "A"), (0, "mu"), (0, "FWHMG"), (0, "FWHML")]
    init = [6.0, 5.0, 30.0, 30.0]
    res = _run_one_fit([sd], [cfg], init, vary_keys)
    assert res["success"], res.get("error")
    fitted = dict(zip([vk[1] for vk in vary_keys], res["fitted_vals"]))
    assert abs(fitted["A"] - 8.0) < 1.0


def test_parameter_bounds_are_respected():
    """The canonical builder applies min/max. Capping A at 3 must keep the
    fitted amplitude within the bound even though the true value is 8."""
    sd = _make_data(a=8.0)
    cfg = _voigt_config(amax=3.0)
    vary_keys = [(0, "A"), (0, "mu"), (0, "FWHMG"), (0, "FWHML")]
    init = [2.0, 0.0, 40.0, 40.0]
    res = _run_one_fit([sd], [cfg], init, vary_keys)
    assert res["success"], res.get("error")
    fitted = dict(zip([vk[1] for vk in vary_keys], res["fitted_vals"]))
    # Small numerical slack for the optimizer's bound handling.
    assert fitted["A"] <= 3.0 + 1e-6


def test_expression_tie_does_not_crash_and_fits():
    """An expression-tied parameter (mu = 0*A i.e. pinned) must be applied
    via setExpr without breaking the fit. Smoke test of the tie path."""
    sd = _make_data(mu=0.0)
    cfg = _voigt_config(mu_expr="0.0 + 0.0")  # constant expr -> mu derived
    # An expression-tied parameter is excluded from vary_keys.
    vary_keys = [(0, "A"), (0, "FWHMG"), (0, "FWHML")]
    init = [6.0, 30.0, 30.0]
    res = _run_one_fit([sd], [cfg], init, vary_keys)
    assert res["success"], res.get("error")
    fitted = dict(zip([vk[1] for vk in vary_keys], res["fitted_vals"]))
    assert np.isfinite(fitted["A"])


def test_autofit_result_reproduces_in_regular_fitter():
    """The user-facing contract: parameters the Auto-Fitter accepts must
    be a fixed point of the regular fitter. Auto-fit an HFS model with an
    Al=Au expression tie, then seed the regular fitter (same canonical
    builder + setExpr) from those values; it must not move and the tie
    must hold."""
    from gui.analysis.fitting import _build_models_on_source
    from gui.analysis.naming import full_param_name
    from gui.analysis.expr_validation import lower_model_expression

    x = np.linspace(-3000.0, 3000.0, 300)
    truth = satlas2.HFS(I=1.5, J=[0.5, 1.5], A=[210.0, 210.0], B=[0.0, 0.0],
                        C=[0, 0], df=15.0, scale=220.0, fwhmg=70.0,
                        fwhml=70.0, name="M1")
    y = truth.f(x) + 5.0
    yerr = np.sqrt(np.abs(y) + 1.0)
    sd = {"name": "Run_1", "x": x, "y": y, "yerr": yerr}

    def pp(value, vary=True, expr=""):
        return {"value": value, "vary": vary, "min": None, "max": None,
                "expr": expr}

    def cfg(au, df, sc, fg, fl):
        return {"type": "HFS", "name": "M1", "racah": True, "params": {
            "I": pp(1.5, vary=False), "Jl": pp(0.5, vary=False),
            "Ju": pp(1.5, vary=False), "Al": pp(au, expr="Au"),
            "Au": pp(au), "Bl": pp(0.0, vary=False),
            "Bu": pp(0.0, vary=False), "centroid": pp(df),
            "scale": pp(sc), "FWHMG": pp(fg), "FWHML": pp(fl),
            "Bkg_p0": pp(5.0)}}

    vk = [(0, "Au"), (0, "centroid"), (0, "scale"), (0, "FWHMG"),
          (0, "FWHML")]
    auto = _run_one_fit([sd], [cfg(150.0, 0.0, 200.0, 80.0, 80.0)],
                        [150.0, 0.0, 200.0, 80.0, 80.0], vk)
    assert auto["success"], auto.get("error")
    af = dict(zip([k[1] for k in vk], auto["fitted_vals"]))
    # Auto-fit recovered the truth.
    assert abs(af["Au"] - 210.0) < 1.0
    assert abs(af["centroid"] - 15.0) < 1.0

    # Regular fitter seeded from the auto result reproduces it.
    src = satlas2.Source(x, y, yerr=yerr, name="Run_1")
    _build_models_on_source(
        src, [cfg(af["Au"], af["centroid"], af["scale"],
                  af["FWHMG"], af["FWHML"])], satlas2)
    f = satlas2.Fitter()
    f.addSource(src)
    f.setExpr(full_param_name("Run_1", "M1", "Al"),
              lower_model_expression(
                  "Au", "Run_1", "M1",
                  {"Au", "centroid", "scale", "FWHMG", "FWHML"}))
    f.fit(method="leastsq", scale_covar=False)
    m = dict(src.models)["M1"]
    assert abs(m.params["Au"].value - af["Au"]) < 1e-3
    assert abs(m.params["centroid"].value - af["centroid"]) < 1e-3
    # The expression tie held through both fits.
    assert abs(m.params["Al"].value - m.params["Au"].value) < 1e-9
