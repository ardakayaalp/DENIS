"""Headless validator for analysis-tab parameter expressions.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Parses and checks the parameter constraint expressions a user types in the
Model and Fitter blocks, reporting parse errors, unknown names, self- and
cyclic references, and mode-specific issues as structured Issue objects.
It also lowers bare model-block expressions to full lmfit names and
classifies (vary, expr) pairs into UI constraint modes. The validator is
pure -- no Qt or satlas2 imports -- so it can be unit-tested headless.

Two expression namespaces exist by design:

- **Model-block expressions** (per-model, in the Model block parameter
  table) reference *bare* parameter names within that model
  (``Au``, ``Al``, ``FWHMG``, ...), plus optional global variables
  and asteval builtins. Referencing a full ``Run_X___...`` name from
  a model block is an error regardless of fit mode.

- **Fitter-block expressions** (in the Advanced Constraints table,
  simultaneous fits only) reference *full* parameter names of the
  form ``Run_<source>___<safe_model>___<param>``.

Cycle detection runs over the union: model-block expressions are
expanded to every source for dependency walking, matching how the
fitter lowers those expressions before passing them to satlas2.

Depends on: gui.analysis.naming (NON_FIT_PARAMS, full_param_name,
is_fit_param_name).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Literal

from gui.analysis.naming import (
    NON_FIT_PARAMS, full_param_name, is_fit_param_name,
)


# Re-export under the local module-private name used by existing
# call sites; the canonical definition lives in naming.py.
_NON_FIT_PARAMS = NON_FIT_PARAMS

# Identifiers asteval makes available by default. Extending this is
# safe (it only widens what we *don't* flag as unresolved -- asteval
# itself remains the final arbiter at fit time). Includes the numpy
# trig aliases lmfit's asteval table accepts.
_ASTEVAL_BUILTINS = frozenset({
    "True", "False", "None",
    "abs", "max", "min", "sum", "len", "round",
    "pi", "e", "inf", "nan",
    "exp", "log", "log10", "log2", "sqrt",
    "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2",
    "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "arcsinh", "arccosh", "arctanh",
    "degrees", "radians",
    "hypot", "copysign",
    "expm1", "log1p",
    "floor", "ceil", "trunc",
    "pow", "fabs",
})

# AST node types that `lower_model_expression` will refuse to rewrite
# because they introduce locally-bound names (lambda args,
# comprehension iterators, walrus targets) that the rewriter can't
# distinguish from references to model parameters.
_UNSAFE_LOWERING_NODES = (
    ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.NamedExpr,
)


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    location: tuple    # ("model", model_name, param_name) or ("fitter", full_name)
    message: str
    expr: str

    def location_str(self) -> str:
        if self.location and self.location[0] == "model":
            return f"Model '{self.location[1]}', parameter '{self.location[2]}'"
        if self.location and self.location[0] == "fitter":
            return f"Advanced constraint on '{self.location[1]}'"
        return "(unknown)"


def build_param_registry(
    model_configs: list[dict],
    source_names: Iterable[str],
) -> set[str]:
    """All full lmfit parameter names that will exist at fit time.

    Mirrors the construction in ``fitting.py`` (including the
    ``Bkg_p0`` -> ``..._bkg___p0`` remap, via ``full_param_name``)
    and skips parameters that are not fittable for their model type
    (HFS metadata I/Jl/Ju, Piecewise Constant ``bound*``).
    """
    names: set[str] = set()
    for src in source_names:
        for mc in model_configs:
            mt = mc.get("type", "")
            for pname in mc.get("params", {}):
                if not is_fit_param_name(mt, pname):
                    continue
                names.add(full_param_name(src, mc["name"], pname))
    return names


def _ast_names(expr: str) -> set[str]:
    """Identifiers appearing as bare names in an expression."""
    tree = ast.parse(expr, mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


class _ModelExprLowerer(ast.NodeTransformer):
    """Rewrite ast.Name nodes whose id is a local model param to the
    corresponding full lmfit name. Other identifiers pass through."""

    def __init__(self, local_params: set[str], source_name: str,
                 model_name: str):
        self._local = local_params
        self._source = source_name
        self._model = model_name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._local:
            return ast.copy_location(
                ast.Name(
                    id=full_param_name(self._source, self._model, node.id),
                    ctx=node.ctx),
                node,
            )
        return node


def lower_model_expression(
    expr: str,
    source_name: str,
    model_name: str,
    local_params: Iterable[str],
) -> str:
    """Rewrite bare param references in a model-block expression to
    full lmfit names for one source.

    Only identifiers in ``local_params`` are substituted. asteval
    builtins, user globals, and any other identifiers are left alone --
    asteval will resolve them at fit time.

    Raises ``ValueError`` for expressions containing constructs that
    bind local names (lambdas, comprehensions, walrus). Those would
    let a bound name shadow a parameter and the rewriter can't tell
    the two apart by AST shape alone.

    Example::

        lower_model_expression('Al*0.5', 'Run_7164', 'HFS_1', {'Al', 'Au'})
        -> 'Run_7164___HFS_1___Al * 0.5'
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, _UNSAFE_LOWERING_NODES):
            raise ValueError(
                f"Cannot lower model expression containing "
                f"{type(node).__name__}: {expr!r}")
    new_tree = _ModelExprLowerer(
        set(local_params), source_name, model_name).visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def _looks_like_full_name(name: str) -> bool:
    """Heuristic: identifier matches the Run_<id>___...___... shape."""
    return name.startswith("Run_") and "___" in name


def _const_value(node: ast.AST) -> float | int | None:
    """Numeric value of a Constant or UnaryOp(USub, Constant). Else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if (isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float))):
            return -node.operand.value
    return None


def infer_constraint_mode(
    expr: str, vary: bool,
) -> tuple[str, dict]:
    """Classify a (vary, expr) pair into a constraint mode for the UI.

    Returns ``(mode, extras)`` where ``mode`` is one of
    ``Free``, ``Fixed``, ``Equal``, ``Ratio``, ``Offset``, ``Custom``
    and ``extras`` carries the structural fields useful for round-trip
    (``target``: bare param name; ``value``: numeric constant).

    Anything that doesn't fit a recognised pattern (parse error,
    multi-operand, function calls, ...) falls back to ``Custom`` so
    that existing user expressions are never silently rewritten.
    """
    expr = (expr or "").strip()
    if not expr:
        return ("Free" if vary else "Fixed"), {}
    try:
        body = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return "Custom", {}

    # Equal: a single bare name.
    if isinstance(body, ast.Name):
        return "Equal", {"target": body.id}

    if isinstance(body, ast.BinOp):
        op = body.op
        l, r = body.left, body.right

        # Ratio: <const> * <name>  or  <name> * <const>
        if isinstance(op, ast.Mult):
            if isinstance(r, ast.Name) and (v := _const_value(l)) is not None:
                return "Ratio", {"target": r.id, "value": v}
            if isinstance(l, ast.Name) and (v := _const_value(r)) is not None:
                return "Ratio", {"target": l.id, "value": v}

        # Offset: <name> + <const>  or  <name> - <const>
        if isinstance(op, (ast.Add, ast.Sub)):
            if isinstance(l, ast.Name) and (v := _const_value(r)) is not None:
                signed = v if isinstance(op, ast.Add) else -v
                return "Offset", {"target": l.id, "value": signed}

    return "Custom", {}


def validate_single_expression(
    expr: str,
    scope: Literal["model", "fitter"],
    visible_names: set[str],
    global_names: set[str] | None = None,
) -> Issue | None:
    """Lightweight per-cell validation for live UI feedback.

    Returns ``None`` if the expression is empty or valid, or one
    :class:`Issue` describing the first problem found.

    Compared to :func:`validate_configs`, this only does:

    - parse check
    - visible-name resolution

    Cycle detection, cross-config consistency, vary/expr conflicts,
    and simultaneous-mode warnings are *not* checked here -- the
    pre-fit modal handles those. Live UI is meant to be cheap and
    forgiving while the user is mid-edit.

    Parameters
    ----------
    expr
        The expression text from the cell.
    scope
        ``"model"`` for the per-model parameter table (bare local
        names), ``"fitter"`` for the Advanced Constraints table
        (full ``Run_<source>___<model>___<param>`` names).
    visible_names
        Names that are valid references in this scope. For ``"model"``,
        the set of bare local parameter names in the same model. For
        ``"fitter"``, the full registry (or a syntactic-shape allowance,
        if the caller hasn't computed it yet).
    global_names
        Optional user-defined globals (the planned globals table);
        always treated as resolvable.
    """
    expr = (expr or "").strip()
    if not expr:
        return None
    globals_ = global_names or set()
    loc = ("model", "<cell>", "<cell>") if scope == "model" \
        else ("fitter", "<cell>")

    try:
        refs = _ast_names(expr)
    except SyntaxError as e:
        return Issue(
            level="error", location=loc, expr=expr,
            message=f"Cannot parse: {e.msg}.",
        )

    for r in refs:
        if r in _ASTEVAL_BUILTINS or r in globals_:
            continue
        if scope == "model":
            if _looks_like_full_name(r):
                return Issue(
                    level="error", location=loc, expr=expr,
                    message=("Use bare names here (e.g. 'Al'); full "
                             "Run_X___... names belong in the Fitter "
                             "block's Advanced Constraints table."),
                )
            if r not in visible_names:
                return Issue(
                    level="error", location=loc, expr=expr,
                    message=f"Unknown name '{r}'.",
                )
        else:  # fitter scope
            if not _looks_like_full_name(r):
                return Issue(
                    level="error", location=loc, expr=expr,
                    message=("Advanced Constraints expressions need full "
                             f"parameter names; '{r}' is a bare name."),
                )
            if visible_names and r not in visible_names:
                return Issue(
                    level="error", location=loc, expr=expr,
                    message=f"Parameter '{r}' does not exist.",
                )

    return None


def validate_configs(
    model_configs: list[dict],
    fitter_config: dict,
    source_names: list[str],
    fit_mode: Literal["separate", "simultaneous"],
    global_names: set[str] | None = None,
) -> list[Issue]:
    """Return a flat list of issues across model-block and fitter-block exprs.

    An empty list means the configuration is OK to fit.
    """
    issues: list[Issue] = []
    globals_ = global_names or set()
    full_registry = build_param_registry(model_configs, source_names)

    # Dependency graph keyed by full lmfit name, populated as we
    # successfully validate each expression. Cycle detection runs at
    # the end over the union of model-block (expanded across sources)
    # and fitter-block edges.
    dep_graph: dict[str, set[str]] = {}

    # ── Model-block expressions ──
    for mc in model_configs:
        model_name = mc.get("name", "")
        model_type = mc.get("type", "")
        local_params = {p for p in mc.get("params", {})
                        if is_fit_param_name(model_type, p)}
        for pname, pdata in mc.get("params", {}).items():
            if not is_fit_param_name(model_type, pname):
                continue
            expr = (pdata.get("expr") or "").strip()
            if not expr:
                continue
            loc = ("model", model_name, pname)

            try:
                refs = _ast_names(expr)
            except SyntaxError as e:
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message=f"Cannot parse expression: {e.msg}.",
                ))
                continue

            full_refs = sorted(r for r in refs if _looks_like_full_name(r))
            unresolved = sorted(
                r for r in refs
                if not _looks_like_full_name(r)
                and r not in _ASTEVAL_BUILTINS
                and r not in globals_
                and r not in local_params
            )

            if full_refs:
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message=("Model-level expressions must use bare "
                             "parameter names (e.g. 'Al'), not full "
                             f"parameter names. Found: "
                             f"{', '.join(full_refs)}. Move cross-run "
                             "constraints to the Fitter block's "
                             "Advanced Constraints table."),
                ))
            if unresolved:
                avail = ", ".join(sorted(local_params)) or "(none)"
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message=(f"Unknown name(s): {', '.join(unresolved)}. "
                             f"Available in this model: {avail}."),
                ))

            if pname in refs:
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message=(f"Expression references its own parameter "
                             f"'{pname}'."),
                ))

            if pdata.get("vary", False):
                issues.append(Issue(
                    level="warning", location=loc, expr=expr,
                    message=("Parameter has Vary=True and a non-empty "
                             "expression; lmfit will silently set "
                             "Vary=False."),
                ))

            # Build dep edges (in full-name space) only when this
            # expression has no resolution errors. Otherwise garbage
            # edges would create spurious cycle reports.
            if not full_refs and not unresolved:
                local_deps = {r for r in refs if r in local_params}
                for src in source_names:
                    this = full_param_name(src, model_name, pname)
                    dep_graph[this] = {
                        full_param_name(src, model_name, r)
                        for r in local_deps
                    }

    # ── Fitter-block expressions ──
    # In Separate mode, fitter-block expressions are not applied at fit
    # time, so we skip strict validation entirely and emit a warning
    # for each non-empty entry. This avoids errors on stale targets
    # when a user toggles modes back and forth, and surfaces the
    # silent-drop clearly. Strict validation only runs in Simultaneous
    # mode, where the expressions actually take effect.
    fitter_exprs: dict[str, str] = fitter_config.get("expressions", {}) or {}
    if fit_mode == "separate":
        for full_name, raw_expr in fitter_exprs.items():
            expr = (raw_expr or "").strip()
            if not expr:
                continue
            issues.append(Issue(
                level="warning",
                location=("fitter", full_name), expr=expr,
                message=("Advanced Constraints are only applied in "
                         "Simultaneous fit mode. This expression will "
                         "be ignored under the current Separate mode."),
            ))
    else:
        for full_name, raw_expr in fitter_exprs.items():
            expr = (raw_expr or "").strip()
            if not expr:
                continue
            loc = ("fitter", full_name)

            if full_name not in full_registry:
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message=("Target parameter does not exist. Check "
                             "that the run, model, and parameter names "
                             "match the currently selected runs and "
                             "models."),
                ))
                continue

            try:
                refs = _ast_names(expr)
            except SyntaxError as e:
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message=f"Cannot parse expression: {e.msg}.",
                ))
                continue

            unresolved = sorted(
                r for r in refs
                if r not in _ASTEVAL_BUILTINS
                and r not in globals_
                and r not in full_registry
            )
            if unresolved:
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message=(
                        f"Unknown parameter(s): {', '.join(unresolved)}."),
                ))

            if full_name in refs:
                issues.append(Issue(
                    level="error", location=loc, expr=expr,
                    message="Expression references its own parameter.",
                ))

            # Fitter-block expressions REPLACE any model-block
            # edges for the same target. satlas2's setExpr is
            # last-write-wins (priority 1 over sharing/model
            # expressions, per core.py:50-77), so the model
            # expression for this target is dead. Keeping the old
            # edge would fabricate cycles that don't exist at fit
            # time. We do this even when the fitter expression has
            # unresolved refs -- the unresolved error is reported
            # separately, and the user's override intent is already
            # established by the presence of the expression.
            deps = {r for r in refs if r in full_registry}
            dep_graph[full_name] = deps

    # ── Cycle detection ──
    for cycle in _find_cycles(dep_graph):
        head = cycle[0]
        chain = " -> ".join(cycle + [head])
        issues.append(Issue(
            level="error", location=("fitter", head), expr="",
            message=f"Cycle in expression dependencies: {chain}.",
        ))

    return issues


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Cycles in a directed graph. Each cycle returned as a list of
    node names in traversal order. Self-loops are returned as length-1.
    Cycles are deduplicated by rotation-invariant canonical form.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in graph}
    path: list[str] = []
    cycles: list[list[str]] = []

    def visit(n: str) -> None:
        c = color.get(n, BLACK)
        if c == GRAY:
            i = path.index(n)
            cycles.append(path[i:])
            return
        if c == BLACK:
            return
        color[n] = GRAY
        path.append(n)
        for m in graph.get(n, ()):
            visit(m)
        path.pop()
        color[n] = BLACK

    for start in list(graph.keys()):
        if color[start] == WHITE:
            visit(start)

    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for cyc in cycles:
        if not cyc:
            continue
        i = cyc.index(min(cyc))
        rot = tuple(cyc[i:] + cyc[:i])
        if rot not in seen:
            seen.add(rot)
            unique.append(list(rot))
    return unique
