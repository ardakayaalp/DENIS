"""Missing-file detection and remapping for session YAML loads.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

When a session YAML refers to ASDF / VASDF data files that no longer
exist on disk (renamed folder, moved Nextcloud share, switched OS),
the loader prompts the user to either locate each missing file or
skip them. This module provides the pure helpers that walk the YAML
structure, scan for missing paths, remap them, and prune skipped
entries; the dialog flow lives in ``main_window``.

Path locations covered:
    preanalysis.files[].path
    analysis.projects[].blocks[type=Source].files[].path

Synthetic paths (``merged://...``) are ignored -- they're not on disk.
``maybe_convert_path`` is applied before the existence check so a
saved POSIX path round-trips on Windows and vice versa.

Depends on: gui.shared_widgets (maybe_convert_path).
"""

import os

from gui.shared_widgets import maybe_convert_path


def _iter_pa_files_lists(pa):
    """Yield each Pre-Analysis ``files`` list in ``pa``.

    Pre-Analysis is multi-project: the live save format is
    ``preanalysis.projects[].files[]``. Legacy single-project saves
    keep ``preanalysis.files[]`` directly under ``preanalysis``;
    PreAnalysisContainer._restore_from_dict still loads those, so we
    have to scan them too.
    """
    if isinstance(pa.get("projects"), list):
        for proj in pa["projects"]:
            if isinstance(proj, dict) and isinstance(
                    proj.get("files"), list):
                yield proj["files"]
    if isinstance(pa.get("files"), list):
        # Legacy flat layout (PA-1-only YAMLs predating the container).
        yield pa["files"]


def _iter_path_holders(yaml_dict):
    """Yield ``(holder_dict, key)`` pairs for every file-path slot in
    the YAML. The caller can read or rewrite ``holder[key]`` in place.

    Synthetic ``merged://`` paths and non-Source analysis blocks are
    skipped here so callers don't need to filter again.
    """
    pa = yaml_dict.get("preanalysis") or {}
    for files_list in _iter_pa_files_lists(pa):
        for fcfg in files_list:
            if isinstance(fcfg, dict) and isinstance(
                    fcfg.get("path"), str):
                if not fcfg["path"].startswith("merged://"):
                    yield fcfg, "path"

    an = yaml_dict.get("analysis") or {}
    for proj in an.get("projects") or []:
        for block in proj.get("blocks") or []:
            if block.get("type") != "Source":
                continue
            for fcfg in block.get("files") or []:
                if isinstance(fcfg, dict) and isinstance(
                        fcfg.get("path"), str):
                    if not fcfg["path"].startswith("merged://"):
                        yield fcfg, "path"


def scan_missing_paths(yaml_dict):
    """Return the unique missing file paths referenced in ``yaml_dict``.

    Order of first appearance is preserved so the locate-files dialog
    walks them top-down (Pre-Analysis first, then Analysis projects in
    order). Paths are returned as the original strings stored in the
    YAML -- ``maybe_convert_path`` is only applied for the existence
    check, not the return value, so the caller's remap dict stays
    keyed on what's actually written to disk.
    """
    seen = set()
    out = []
    for holder, key in _iter_path_holders(yaml_dict):
        original = holder[key]
        if original in seen:
            continue
        if not os.path.isfile(maybe_convert_path(original)):
            seen.add(original)
            out.append(original)
    return out


def remap_paths_inplace(yaml_dict, remap):
    """Rewrite path slots in-place using ``remap`` (old -> new).

    Returns the number of slots replaced. Slots whose current value
    isn't a key in ``remap`` are left alone, which is what we want
    when ``remap`` only covers a subset (the user located 3 of 5).

    Also rewrites the path-*keyed* top-level registries (``scan_filters``,
    ``calibrations``), which ``_iter_path_holders`` cannot reach because
    their paths are dict keys rather than values. Without this, relocating a
    moved project silently drops every scan filter and voltage calibration
    attached to it -- the run would load fine and quietly compute its
    voltages from the file default, which is precisely the silent-wrong-number
    failure the calibration feature exists to eliminate.
    """
    if not remap:
        return 0
    n = 0
    for holder, key in _iter_path_holders(yaml_dict):
        if holder[key] in remap:
            holder[key] = remap[holder[key]]
            n += 1
    n += _remap_registry_keys_inplace(yaml_dict, remap)
    return n


def _remap_registry_keys_inplace(yaml_dict, remap):
    """Rewrite the path-keyed registry sections. Returns slots replaced.

    ``remap`` is keyed by the raw strings stored in the YAML's *file-entry*
    slots (e.g. ``C:/Users/x/Desktop/run_7735.asdf``), but the registries key
    themselves through ``canonical_path`` (``os.path.normcase`` + ``abspath``),
    which on Windows lowercases and flips the separators
    (``c:\\users\\x\\desktop\\run_7735.asdf``). A direct ``remap.get(path)``
    therefore never matches, and the section would be silently left pointing at
    the dead path -- the run then loads fine and quietly computes its voltages
    from the uncorrected calibration. So match canonically, on both sides.
    """
    from gui.calibration import canonical_path

    if not remap:
        return 0
    # Canonical view of the remap, so a registry key in any spelling resolves.
    canon = {canonical_path(old): new for old, new in remap.items()}

    def _lookup(path):
        if not isinstance(path, str):
            return None
        if path in remap:
            return remap[path]
        return canon.get(canonical_path(path))

    n = 0
    # ``calibration_acks`` is a plain list of paths, not a dict -- a relocated
    # run whose ack was left behind would start blinking its warning again on
    # every project load, which trains the user to ignore the warning.
    acks = yaml_dict.get("calibration_acks")
    if isinstance(acks, list):
        rebuilt_acks = []
        for p in acks:
            new_p = _lookup(p)
            if new_p is None:
                rebuilt_acks.append(p)
            else:
                rebuilt_acks.append(new_p)
                n += 1
        yaml_dict["calibration_acks"] = rebuilt_acks

    for section in ("scan_filters", "calibrations"):
        reg = yaml_dict.get(section)
        if not isinstance(reg, dict):
            continue
        rebuilt = {}
        for path, payload in reg.items():
            new_path = _lookup(path)
            if new_path is None:
                new_path = path
            else:
                n += 1
            # A borrow spec points at a *donor* run, which may itself have
            # moved. Miss this and the calibration resolves to a dead path and
            # fails the fit outright.
            if isinstance(payload, dict):
                new_donor = _lookup(payload.get("donor"))
                if new_donor is not None:
                    payload = dict(payload)
                    payload["donor"] = new_donor
                    n += 1
            rebuilt[new_path] = payload
        yaml_dict[section] = rebuilt
    return n


def prune_missing_paths_inplace(yaml_dict, missing):
    """Drop file entries whose ``path`` is in the ``missing`` set.

    Used after the locate flow to remove the entries the user opted
    to skip (either via the top-level Skip button or by cancelling
    mid-locate). Doesn't touch the on-disk YAML; only the in-memory
    dict the loader is about to consume.
    """
    if not missing:
        return

    def _filter(files_list):
        return [
            f for f in files_list
            if not (isinstance(f, dict) and f.get("path") in missing)
        ]

    pa = yaml_dict.get("preanalysis") or {}
    if isinstance(pa.get("projects"), list):
        for proj in pa["projects"]:
            if isinstance(proj, dict) and isinstance(
                    proj.get("files"), list):
                proj["files"] = _filter(proj["files"])
    if isinstance(pa.get("files"), list):
        pa["files"] = _filter(pa["files"])

    an = yaml_dict.get("analysis") or {}
    for proj in an.get("projects") or []:
        for block in proj.get("blocks") or []:
            if block.get("type") != "Source":
                continue
            if isinstance(block.get("files"), list):
                block["files"] = _filter(block["files"])
