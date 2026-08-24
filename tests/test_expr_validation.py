"""Headless tests for the model/fitter constraint-expression validator.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Validates gui.analysis.expr_validation: building the parameter
registry, lowering bare model-local names to fully qualified
Source___Model___param names, classifying constraint expressions
(Free/Fixed/Equal/Ratio/Offset/Custom), and reporting parse errors,
unknown names, self-references, and dependency cycles across the
model-local and fitter-global scopes. Runs headless from the project
root with the project interpreter.

Depends on: gui.analysis.expr_validation, gui.analysis.naming.
"""

import unittest

from gui.analysis.expr_validation import (
    Issue,
    build_param_registry,
    infer_constraint_mode,
    lower_model_expression,
    validate_configs,
    validate_single_expression,
)


def _hfs_model(name="HFS_1", *, expr_map=None, vary_map=None):
    """Convenience: build a minimal HFS-shaped model_config dict.

    `expr_map` / `vary_map` are {param_name: expr} / {param_name: bool};
    omitted params get expr="" / vary=True.
    """
    expr_map = expr_map or {}
    vary_map = vary_map or {}
    params = ["I", "Jl", "Ju", "Al", "Au", "Bl", "Bu",
              "centroid", "scale", "FWHMG", "FWHML", "Bkg_p0"]
    return {
        "type": "HFS",
        "name": name,
        "source": "src",
        "params": {
            p: {
                "value": 0.0, "min": -1e9, "max": 1e9,
                "vary": vary_map.get(p, True),
                "expr": expr_map.get(p, ""),
            }
            for p in params
        },
    }


def _errors(issues):
    return [i for i in issues if i.level == "error"]


def _warnings(issues):
    return [i for i in issues if i.level == "warning"]


class ParamRegistryTests(unittest.TestCase):
    def test_full_names_built_for_each_source(self):
        models = [_hfs_model("HFS_1")]
        reg = build_param_registry(models, ["Run_7164", "Run_7191"])
        self.assertIn("Run_7164___HFS_1___Au", reg)
        self.assertIn("Run_7191___HFS_1___Au", reg)

    def test_bkg_p0_uses_bkg_suffix_remap(self):
        reg = build_param_registry([_hfs_model("HFS_1")], ["Run_7164"])
        self.assertIn("Run_7164___HFS_1_bkg___p0", reg)
        self.assertNotIn("Run_7164___HFS_1___Bkg_p0", reg)

    def test_non_fit_params_excluded(self):
        reg = build_param_registry([_hfs_model("HFS_1")], ["Run_7164"])
        self.assertNotIn("Run_7164___HFS_1___I", reg)
        self.assertNotIn("Run_7164___HFS_1___Jl", reg)
        self.assertNotIn("Run_7164___HFS_1___Ju", reg)

    def test_safe_name_normalises_special_chars(self):
        reg = build_param_registry(
            [_hfs_model("HFS 1.alt")], ["Run_7164"])
        self.assertIn("Run_7164___HFS_1_alt___Au", reg)


class ModelLocalNamespaceTests(unittest.TestCase):
    def test_clean_config_has_no_issues(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "0.6 * Al"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, fitter_config={}, source_names=["Run_7164"],
            fit_mode="separate")
        self.assertEqual(issues, [])

    def test_empty_expr_is_skipped(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "   "})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        self.assertEqual(issues, [])

    def test_parse_error_is_reported(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "0.6 *"})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        errs = _errors(issues)
        self.assertEqual(len(errs), 1)
        self.assertIn("parse", errs[0].message.lower())
        self.assertEqual(errs[0].location, ("model", "HFS_1", "Au"))

    def test_unknown_name_is_reported_with_available_list(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "0.6 * NotAParam"})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        errs = _errors(issues)
        self.assertEqual(len(errs), 1)
        self.assertIn("NotAParam", errs[0].message)
        self.assertIn("Available in this model", errs[0].message)

    def test_full_name_in_model_expr_is_rejected_in_separate(self):
        models = [_hfs_model("HFS_1",
                             expr_map={"Au": "Run_7164___HFS_1___Al * 0.6"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        errs = _errors(issues)
        self.assertEqual(len(errs), 1)
        self.assertIn("full parameter names", errs[0].message)
        self.assertIn("Fitter block", errs[0].message)

    def test_full_name_in_model_expr_is_rejected_in_simultaneous(self):
        models = [_hfs_model("HFS_1",
                             expr_map={"Au": "Run_7164___HFS_1___Al * 0.6"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164", "Run_7191"], "simultaneous")
        errs = _errors(issues)
        self.assertTrue(any("full parameter names" in e.message
                            for e in errs))

    def test_self_reference_is_reported(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "Au * 2"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        # Both self-reference and a length-1 cycle may be reported; we
        # just need the self-reference signal to be present.
        self.assertTrue(any(
            "references its own parameter" in e.message
            for e in _errors(issues)
        ))

    def test_vary_true_with_expr_emits_warning(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "0.6 * Al"},
                             vary_map={"Au": True})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        self.assertEqual(_errors(issues), [])
        warns = _warnings(issues)
        self.assertTrue(any("Vary=True" in w.message for w in warns))

    def test_global_names_are_recognised(self):
        models = [_hfs_model("HFS_1",
                             expr_map={"Au": "ratio_A * Al"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate",
            global_names={"ratio_A"})
        self.assertEqual(_errors(issues), [])

    def test_asteval_builtins_recognised(self):
        models = [_hfs_model("HFS_1",
                             expr_map={"Au": "exp(Al / pi)"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        self.assertEqual(_errors(issues), [])

    def test_two_cycle_in_one_model_detected(self):
        models = [_hfs_model("HFS_1",
                             expr_map={"Au": "0.5 * Bu", "Bu": "2 * Au"},
                             vary_map={"Au": False, "Bu": False})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        cycle_errs = [e for e in _errors(issues)
                      if "Cycle" in e.message]
        self.assertEqual(len(cycle_errs), 1)
        self.assertIn("Au", cycle_errs[0].message)
        self.assertIn("Bu", cycle_errs[0].message)


class FitterFullNamespaceTests(unittest.TestCase):
    def test_clean_fitter_expression(self):
        models = [_hfs_model("HFS_1")]
        fitter_cfg = {"expressions": {
            "Run_7191___HFS_1___centroid":
                "Run_7164___HFS_1___centroid + 100.0",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164", "Run_7191"], "simultaneous")
        self.assertEqual(_errors(issues), [])

    def test_target_does_not_exist(self):
        models = [_hfs_model("HFS_1")]
        fitter_cfg = {"expressions": {
            "Run_9999___HFS_1___centroid": "Run_7164___HFS_1___centroid",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164"], "simultaneous")
        errs = _errors(issues)
        self.assertEqual(len(errs), 1)
        self.assertIn("does not exist", errs[0].message)

    def test_unknown_dependency_in_fitter_expr(self):
        models = [_hfs_model("HFS_1")]
        fitter_cfg = {"expressions": {
            "Run_7191___HFS_1___centroid": "Run_7164___HFS_1___nope",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164", "Run_7191"], "simultaneous")
        errs = _errors(issues)
        self.assertTrue(any("Unknown parameter" in e.message for e in errs))


class SimultaneousModeBehaviourTests(unittest.TestCase):
    def test_simultaneous_accepts_model_expressions(self):
        # The fitter now expands model-level expressions across all
        # sources in simultaneous mode (same as separate mode), so
        # there is no warning -- the configuration is fully valid.
        models = [_hfs_model("HFS_1", expr_map={"Au": "0.6 * Al"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164", "Run_7191"], "simultaneous")
        self.assertEqual(issues, [])

    def test_separate_mode_accepts_model_expressions(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "0.6 * Al"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        self.assertEqual(issues, [])


class CycleDetectionTests(unittest.TestCase):
    def test_cycle_across_model_and_fitter_scope(self):
        # Cross-scope cycle that's only real in Simultaneous mode
        # (fitter expressions are not applied in Separate mode).
        # Model: Au = 0.5 * Bu, lowered onto Run_7164.
        # Fitter: Run_7164___HFS_1___Bu = 2 * Run_7164___HFS_1___Au.
        # Cycle: Au -> Bu -> Au.
        models = [_hfs_model("HFS_1",
                             expr_map={"Au": "0.5 * Bu"},
                             vary_map={"Au": False})]
        fitter_cfg = {"expressions": {
            "Run_7164___HFS_1___Bu": "2 * Run_7164___HFS_1___Au",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164"], "simultaneous")
        cycle_errs = [e for e in _errors(issues)
                      if "Cycle" in e.message]
        self.assertEqual(len(cycle_errs), 1)

    def test_no_false_cycle_when_fitter_overrides_model(self):
        # Bug #5 regression: validator used to keep model-block edges
        # for a target even when a fitter-block expression overrode it.
        # Model: Au = 0.5 * Bu (lowered: edges Au -> {Bu})
        # Model: Bu = Au + 1 (lowered: edges Bu -> {Au})  — model cycle!
        # Fitter: Run_7164___HFS_1___Au = 0.5 * Cl
        #         (overrides Au; real edges Au -> {Cl})
        # At fit time the model expr for Au is dead, so the cycle
        # Au -> Bu -> Au does NOT exist. Validator should agree.
        models = [_hfs_model("HFS_1",
                             expr_map={"Au": "0.5 * Bu",
                                       "Bu": "Au + 1"},
                             vary_map={"Au": False, "Bu": False})]
        fitter_cfg = {"expressions": {
            "Run_7164___HFS_1___Au": "0.5 * Run_7164___HFS_1___Cl",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164"], "simultaneous")
        # No cycle: the fitter's Au override breaks the model loop.
        cycle_errs = [e for e in _errors(issues)
                      if "Cycle" in e.message]
        self.assertEqual(cycle_errs, [],
                         f"unexpected cycles: {[e.message for e in cycle_errs]}")

    def test_separate_mode_skips_strict_fitter_validation(self):
        # Bug #3 regression: stale fitter expressions (e.g. left over
        # after switching from Simultaneous to Separate) used to error
        # if their target was no longer in the registry. They should
        # warn only.
        models = [_hfs_model("HFS_1")]
        fitter_cfg = {"expressions": {
            "Run_9999___HFS_1___Au": "Run_7164___HFS_1___Au",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164"], "separate")
        self.assertEqual(_errors(issues), [],
                         "Separate mode must not error on stale fitter exprs")
        self.assertTrue(_warnings(issues),
                        "Separate mode should warn about ignored exprs")


class LowerModelExpressionTests(unittest.TestCase):
    LOCAL = {"Al", "Au", "Bl", "Bu", "centroid", "scale",
             "FWHMG", "FWHML", "Bkg_p0"}

    def _lower(self, expr):
        return lower_model_expression(
            expr, "Run_7164", "HFS_1", self.LOCAL)

    def test_simple_bare_name_is_rewritten(self):
        self.assertEqual(self._lower("Al"), "Run_7164___HFS_1___Al")

    def test_arithmetic_expression_rewritten(self):
        # ast.unparse may insert spaces; compare on normalized eval
        out = self._lower("Al * 0.5")
        self.assertIn("Run_7164___HFS_1___Al", out)
        self.assertIn("0.5", out)

    def test_multiple_locals_rewritten(self):
        out = self._lower("0.5 * (Al + Au)")
        self.assertIn("Run_7164___HFS_1___Al", out)
        self.assertIn("Run_7164___HFS_1___Au", out)

    def test_asteval_builtin_passes_through(self):
        out = self._lower("exp(Al / pi)")
        self.assertIn("Run_7164___HFS_1___Al", out)
        self.assertIn("pi", out)
        # `pi` is bare, not a full name
        self.assertNotIn("Run_7164___HFS_1___pi", out)
        # `exp` is a function call, not rewritten
        self.assertIn("exp(", out)

    def test_unknown_global_passes_through(self):
        # Globals are not known to the lowerer; they're left alone.
        out = self._lower("ratio_A * Al")
        self.assertIn("ratio_A", out)
        self.assertNotIn("Run_7164___HFS_1___ratio_A", out)
        self.assertIn("Run_7164___HFS_1___Al", out)

    def test_bkg_p0_uses_bkg_remap(self):
        # Lowering Bkg_p0 must go through the same _bkg___p0 remap
        # full_param_name uses, so the rewritten reference matches the
        # name lmfit will actually expose.
        out = lower_model_expression(
            "Bkg_p0 * 2", "Run_7164", "HFS_1", self.LOCAL)
        self.assertIn("Run_7164___HFS_1_bkg___p0", out)

    def test_safe_model_name_normalisation(self):
        out = lower_model_expression(
            "Al", "Run_7164", "HFS 1.alt", {"Al"})
        self.assertEqual(out, "Run_7164___HFS_1_alt___Al")

    def test_round_trips_through_asteval(self):
        # Whatever we emit must be parseable as an expression -- lmfit
        # would otherwise reject it. Use ast.parse(mode='eval') as a
        # cheap proxy for asteval's parser.
        import ast as _ast
        for raw in ["Al", "Al*0.5", "0.5 * (Al + Au)",
                    "exp(Al / pi)", "Al + ratio_A", "Bkg_p0 + 1"]:
            with self.subTest(raw=raw):
                out = self._lower(raw)
                _ast.parse(out, mode="eval")  # raises if malformed


class ValidateSingleExpressionTests(unittest.TestCase):
    LOCAL = {"Al", "Au", "Bl", "Bu", "centroid", "scale",
             "FWHMG", "FWHML", "Bkg_p0"}
    REGISTRY = {
        "Run_7164___HFS_1___Al", "Run_7164___HFS_1___Au",
        "Run_7164___HFS_1___centroid",
        "Run_7191___HFS_1___Al", "Run_7191___HFS_1___Au",
    }

    def test_empty_returns_none(self):
        self.assertIsNone(validate_single_expression(
            "", "model", self.LOCAL))
        self.assertIsNone(validate_single_expression(
            "   ", "model", self.LOCAL))

    def test_model_valid_bare_name(self):
        self.assertIsNone(validate_single_expression(
            "Al * 0.5", "model", self.LOCAL))

    def test_model_unknown_bare_name(self):
        issue = validate_single_expression(
            "Al + Nope", "model", self.LOCAL)
        self.assertIsNotNone(issue)
        self.assertIn("Unknown name", issue.message)

    def test_model_rejects_full_name(self):
        issue = validate_single_expression(
            "Run_7164___HFS_1___Al * 0.5", "model", self.LOCAL)
        self.assertIsNotNone(issue)
        self.assertIn("bare names", issue.message)
        self.assertIn("Fitter block", issue.message)

    def test_model_parse_error(self):
        issue = validate_single_expression("Al *", "model", self.LOCAL)
        self.assertIsNotNone(issue)
        self.assertIn("parse", issue.message.lower())

    def test_model_builtins_pass(self):
        self.assertIsNone(validate_single_expression(
            "exp(Al / pi)", "model", self.LOCAL))

    def test_model_globals_pass(self):
        self.assertIsNone(validate_single_expression(
            "ratio_A * Al", "model", self.LOCAL,
            global_names={"ratio_A"}))

    def test_fitter_valid_full_name(self):
        self.assertIsNone(validate_single_expression(
            "Run_7164___HFS_1___Al * 0.5", "fitter", self.REGISTRY))

    def test_fitter_rejects_bare_name(self):
        issue = validate_single_expression(
            "Al * 0.5", "fitter", self.REGISTRY)
        self.assertIsNotNone(issue)
        self.assertIn("bare name", issue.message)

    def test_fitter_unknown_full_name(self):
        issue = validate_single_expression(
            "Run_9999___HFS_1___Al", "fitter", self.REGISTRY)
        self.assertIsNotNone(issue)
        self.assertIn("does not exist", issue.message)

    def test_fitter_empty_registry_accepts_full_name_shape(self):
        # When the registry hasn't been computed yet (e.g. no models
        # configured), accept anything that looks like a full name.
        # The pre-fit modal will catch real mismatches.
        self.assertIsNone(validate_single_expression(
            "Run_7164___HFS_1___Al", "fitter", set()))


class InferConstraintModeTests(unittest.TestCase):
    def test_empty_vary_true_is_free(self):
        self.assertEqual(infer_constraint_mode("", True), ("Free", {}))
        self.assertEqual(infer_constraint_mode("   ", True), ("Free", {}))

    def test_empty_vary_false_is_fixed(self):
        self.assertEqual(infer_constraint_mode("", False), ("Fixed", {}))

    def test_single_name_is_equal(self):
        mode, extras = infer_constraint_mode("Al", False)
        self.assertEqual(mode, "Equal")
        self.assertEqual(extras["target"], "Al")

    def test_const_times_name_is_ratio(self):
        mode, extras = infer_constraint_mode("0.5 * Al", False)
        self.assertEqual(mode, "Ratio")
        self.assertEqual(extras, {"target": "Al", "value": 0.5})

    def test_name_times_const_is_ratio(self):
        mode, extras = infer_constraint_mode("Al * 0.5", False)
        self.assertEqual(mode, "Ratio")
        self.assertEqual(extras, {"target": "Al", "value": 0.5})

    def test_negative_ratio(self):
        mode, extras = infer_constraint_mode("-0.5 * Al", False)
        self.assertEqual(mode, "Ratio")
        self.assertEqual(extras, {"target": "Al", "value": -0.5})

    def test_name_plus_const_is_offset(self):
        mode, extras = infer_constraint_mode("centroid + 1250", False)
        self.assertEqual(mode, "Offset")
        self.assertEqual(extras, {"target": "centroid", "value": 1250})

    def test_name_minus_const_is_negative_offset(self):
        mode, extras = infer_constraint_mode("centroid - 100", False)
        self.assertEqual(mode, "Offset")
        self.assertEqual(extras, {"target": "centroid", "value": -100})

    def test_complex_expression_is_custom(self):
        mode, _ = infer_constraint_mode("0.5 * Al + 0.1", False)
        self.assertEqual(mode, "Custom")

    def test_function_call_is_custom(self):
        mode, _ = infer_constraint_mode("exp(Al)", False)
        self.assertEqual(mode, "Custom")

    def test_two_names_is_custom(self):
        # "Al + Bu" doesn't fit Equal/Ratio/Offset (Offset wants const).
        mode, _ = infer_constraint_mode("Al + Bu", False)
        self.assertEqual(mode, "Custom")

    def test_parse_error_is_custom(self):
        mode, _ = infer_constraint_mode("Al *", False)
        self.assertEqual(mode, "Custom")


class LowerModelExpressionSafetyTests(unittest.TestCase):
    LOCAL = {"Al", "Au", "Bl", "Bu", "centroid"}

    def test_lambda_rejected(self):
        with self.assertRaises(ValueError):
            lower_model_expression(
                "(lambda x: x + Al)(0)",
                "Run_7164", "HFS_1", self.LOCAL)

    def test_list_comprehension_rejected(self):
        with self.assertRaises(ValueError):
            lower_model_expression(
                "sum([Al for _ in range(3)])",
                "Run_7164", "HFS_1", self.LOCAL)

    def test_generator_expression_rejected(self):
        with self.assertRaises(ValueError):
            lower_model_expression(
                "sum(Al for _ in range(3))",
                "Run_7164", "HFS_1", self.LOCAL)


class AstevalBuiltinsTests(unittest.TestCase):
    LOCAL = {"Al", "Au", "Bl", "Bu", "centroid"}

    def test_numpy_trig_aliases_pass(self):
        for fn in ("arctan", "arcsin", "arccos", "arctan2"):
            with self.subTest(fn=fn):
                args = "Al, Au" if fn == "arctan2" else "Al"
                issue = validate_single_expression(
                    f"{fn}({args})", "model", self.LOCAL)
                self.assertIsNone(issue)

    def test_degrees_radians_pass(self):
        self.assertIsNone(validate_single_expression(
            "degrees(Al) + radians(Au)", "model", self.LOCAL))

    def test_hypot_passes(self):
        self.assertIsNone(validate_single_expression(
            "hypot(Al, Au)", "model", self.LOCAL))


class SeparateModeFitterExpressionWarningTests(unittest.TestCase):
    def test_separate_mode_warns_on_fitter_expressions(self):
        models = [_hfs_model("HFS_1")]
        fitter_cfg = {"expressions": {
            "Run_7164___HFS_1___Au": "0.5 * Run_7164___HFS_1___Al",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164"], "separate")
        # No errors -- the expression is otherwise valid.
        self.assertEqual(_errors(issues), [])
        warns = _warnings(issues)
        self.assertTrue(warns)
        self.assertTrue(any(
            "Separate" in w.message and "ignored" in w.message
            for w in warns
        ))

    def test_simultaneous_mode_no_warning(self):
        models = [_hfs_model("HFS_1")]
        fitter_cfg = {"expressions": {
            "Run_7164___HFS_1___Au": "0.5 * Run_7164___HFS_1___Al",
        }}
        issues = validate_configs(
            models, fitter_cfg, ["Run_7164", "Run_7191"], "simultaneous")
        self.assertEqual(issues, [])

    def test_separate_mode_no_warning_when_no_fitter_expressions(self):
        models = [_hfs_model("HFS_1", expr_map={"Au": "0.5 * Al"},
                             vary_map={"Au": False})]
        issues = validate_configs(
            models, {}, ["Run_7164"], "separate")
        self.assertEqual(issues, [])


class PiecewiseFilterTests(unittest.TestCase):
    """satlas2.PiecewiseConstant only exposes value* as fit params;
    bound* is a fixed numpy array on the model. Registry, sharing
    union, and validation must filter accordingly."""

    def _piecewise_model(self):
        return {
            "type": "Piecewise Constant", "name": "PC_1", "source": "src",
            "params": {
                "value0": {"value": 0.0, "vary": True, "expr": "",
                           "min": 0, "max": 1e8},
                "value1": {"value": 0.0, "vary": True, "expr": "",
                           "min": 0, "max": 1e8},
                "bound0": {"value": 0.0, "vary": False, "expr": "",
                           "min": -1e8, "max": 1e8},
            },
        }

    def test_registry_excludes_bounds(self):
        reg = build_param_registry([self._piecewise_model()], ["Run_X"])
        self.assertIn("Run_X___PC_1___value0", reg)
        self.assertIn("Run_X___PC_1___value1", reg)
        self.assertNotIn("Run_X___PC_1___bound0", reg)

    def test_validation_rejects_bound_reference(self):
        # A model expression on a Piecewise bound* should fail because
        # bound0 isn't a fittable param of this model type.
        model = self._piecewise_model()
        # Simulate a user expression on value0 that references bound0
        model["params"]["value0"]["expr"] = "bound0 + 1"
        issues = validate_configs(
            [model], {}, ["Run_X"], "separate")
        errs = _errors(issues)
        self.assertTrue(errs)
        self.assertTrue(any("Unknown name" in e.message for e in errs))


class NamingSanitizationTests(unittest.TestCase):
    """Sanitize hyphens, dots, etc. so lmfit's identifier check accepts."""

    def test_path_with_hyphen_sanitized(self):
        from gui.analysis.naming import source_name_for_path
        # The trailing token after the last underscore is the "run id".
        # Hyphens in that token must become underscores.
        self.assertEqual(
            source_name_for_path("/data/run_7164-a.asdf"),
            "Run_7164_a")

    def test_merged_data_with_hyphens_sanitized(self):
        from gui.analysis.naming import source_name_for_merged
        # User-edited merge labels often look like 'merged-7164-7191'.
        # Sanitize so lmfit's add_many doesn't choke on the hyphens.
        self.assertEqual(
            source_name_for_merged({"merged_name": "merged-7164-7191"}),
            "Run_merged_7164_7191")

    def test_merged_data_with_dots_sanitized(self):
        from gui.analysis.naming import source_name_for_merged
        self.assertEqual(
            source_name_for_merged({"merged_name": "Yb.171.scan"}),
            "Run_Yb_171_scan")

    def test_full_name_round_trip_with_sanitized_source(self):
        from gui.analysis.naming import (
            source_name_for_merged, full_param_name,
        )
        src = source_name_for_merged({"merged_name": "merged-7164-7191"})
        full = full_param_name(src, "HFS_1", "Au")
        # Full name must be a valid Python identifier (lmfit's
        # `valid_symbol_name` requirement). Triple underscores are
        # allowed since they're underscore characters.
        self.assertTrue(full.replace("___", "_").isidentifier(),
                        f"non-identifier: {full!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
