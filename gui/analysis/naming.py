"""Pure naming helpers for the analysis fit pipeline.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Single source of truth for the lmfit parameter and source naming
convention used by the fitter and validator. Builds source names from
run files, merged-spectrum descriptors, and virtual-split descriptors,
composes full parameter names, and decides which model parameters are
actually fittable. This module deliberately avoids any Qt or satlas2
imports so it can be used from headless contexts (the validator, tests).

Convention
----------
Source name:
    Run_<run_id>

    where ``<run_id>`` is the trailing token of the ASDF filename
    (e.g. ``run_7164.asdf`` -> ``7164``) for standard runs, or
    ``merged_data["merged_name"]`` for pre-merged spectra.

Full parameter name:
    <source>___<safe_model>___<param>

    Triple-underscore separator. ``<safe_model>`` is the model block
    name with non-alphanumeric characters replaced by ``_``. The
    ``Bkg_p0`` parameter is remapped to ``<source>___<safe_model>_bkg___p0``
    because satlas2 builds backgrounds as a separate sub-model.

Depends on: standard library only (os, re).
"""

import os
import re


# Parameters present on a model in fitting.py / blocks.py / validator
# but never fit (kept fixed by physics): I, Jl, Ju on HFS models.
# Single source of truth so the fitter, validator, and UI agree.
NON_FIT_PARAMS = frozenset({"I", "Jl", "Ju"})


def is_fit_param_name(model_type: str, param_name: str) -> bool:
    """Whether ``param_name`` becomes an lmfit Parameter at fit time
    for a model of the given ``model_type``.

    Returns False for:
    - HFS metadata: ``I``, ``Jl``, ``Ju`` (fixed by physics across
      every model type that exposes them).
    - PiecewiseConstant ``bound*`` keys: satlas2 only exposes
      ``value*`` as lmfit parameters; ``bound`` values are stored
      as a fixed numpy array on the model and are not fittable.
      See ``_build_models_on_source`` for the construction.

    Returns True otherwise -- the param flows through to lmfit and
    can legitimately appear in expressions, sharing, and constraints.
    """
    if param_name in NON_FIT_PARAMS:
        return False
    if model_type == "Piecewise Constant" and param_name.startswith("bound"):
        return False
    return True


_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def safe_model_name(name: str) -> str:
    """Sanitize a model block name for use in an lmfit parameter name."""
    return _SAFE_RE.sub("_", name)


def _safe_id(token: str) -> str:
    """Sanitize an arbitrary string into a Python-identifier-safe token.

    Replaces any non-[A-Za-z0-9_] character with ``_``. lmfit's
    ``valid_symbol_name`` rejects anything else, so source names
    derived from filenames or user-provided merge labels must pass
    through this before being concatenated into a full lmfit name.
    """
    return _SAFE_RE.sub("_", token)


def source_name_for_path(run_file_path: str) -> str:
    """Source name for a standard ASDF run loaded from disk.

    Sanitized so files with hyphens or other non-identifier characters
    in the trailing token still produce a valid lmfit name.
    """
    base = os.path.basename(run_file_path)
    run_id = base.replace(".asdf", "").split("_")[-1]
    return f"Run_{_safe_id(run_id)}"


def source_name_for_merged(merged_data: dict) -> str:
    """Source name for a pre-merged spectrum descriptor.

    User-edited merge labels often contain hyphens or other
    non-identifier characters (e.g. ``merged-7164-7191``). Those
    would crash lmfit's ``add_many`` later, so sanitize here at
    the single source of truth for source naming.
    """
    run_id = merged_data.get("merged_name", "merged")
    return f"Run_{_safe_id(run_id)}"


def source_name_for_descriptor(desc: dict) -> str:
    """Source name for any run-source descriptor.

    Descriptor shape::

        {"kind": "asdf",          "path": ".../run_7509.asdf"}
        {"kind": "merged",        "merged_data": {...}}
        {"kind": "virtual_split", "source_id": "7509_A",
                                  "parent_path": ".../run_7509.asdf"}

    The canonical ``source_id`` for a virtual split is the parent's
    run id with a short suffix (``_A``, ``_B``, ``_2``, ...), with no
    leading ``run_`` — this function prepends ``Run_`` itself, so a
    ``source_id`` of ``7509_A`` becomes ``Run_7509_A``. Passing in
    ``run_7509_A`` would produce the unwanted ``Run_run_7509_A``.
    """
    kind = desc.get("kind", "asdf")
    if kind == "merged":
        return source_name_for_merged(desc.get("merged_data", desc))
    if kind == "virtual_split":
        return f"Run_{_safe_id(desc['source_id'])}"
    return source_name_for_path(desc["path"])


def full_param_name(source_name: str, model_name: str,
                    param_name: str) -> str:
    """Compose the full lmfit parameter name for one (source, model, param)."""
    safe = safe_model_name(model_name)
    if param_name == "Bkg_p0":
        return f"{source_name}___{safe}_bkg___p0"
    return f"{source_name}___{safe}___{param_name}"
