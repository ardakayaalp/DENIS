"""Read/write helpers for ``.vasdf`` virtual-split sidecar files.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

A ``.vasdf`` is a tiny YAML pointer next to the parent ASDF:

    kind: virtual_split
    version: 1
    parent_path: /path/to/run_7509.asdf
    source_id: 7509_A
    label: 7509_A
    split:
      axis: raw_voltage
      lo: -120.0
      hi: 35.0
    metadata_override:
      cooler_v: 29912.19
      laser_sp: 10920.1234
      comment: 173Yb peak

The parent ASDF is never touched. Loaders detect the ``.vasdf``
extension, read the descriptor, load the parent ASDF, and apply the
V-gate at every consumer (preview binning, fitter binning). This module
provides the write/read/normalize/path-suggestion functions for those
sidecars; it persists and reads back only the metadata-override keys it
knows how to interpret.

Depends on: standard library and third-party packages only (os, typing,
yaml).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml


VASDF_EXT = ".vasdf"
VASDF_KIND = "virtual_split"
VASDF_VERSION = 1


def is_vasdf_path(path: Optional[str]) -> bool:
    """True if ``path`` ends with the ``.vasdf`` extension (case-insensitive)."""
    if not path:
        return False
    return os.path.splitext(path)[1].lower() == VASDF_EXT


def write_vasdf(
    path: str,
    parent_path: str,
    source_id: str,
    label: str,
    lo: float,
    hi: float,
    metadata_override: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a ``.vasdf`` sidecar at ``path``.

    The descriptor is intentionally tiny (no array data) so the
    parent ASDF is the single source of truth for the events. The
    file's identity in every UI map is its own path, ``path``.
    """
    desc = {
        "kind": VASDF_KIND,
        "version": VASDF_VERSION,
        "parent_path": str(parent_path),
        "source_id": str(source_id),
        "label": str(label or source_id),
        "split": {
            "axis": "raw_voltage",
            "lo": float(lo),
            "hi": float(hi),
        },
    }
    if metadata_override:
        # Persist only the keys we know how to interpret. Unknown keys
        # are silently dropped so a stray field can't poison loading.
        clean = {}
        for key in ("cooler_v", "laser_sp", "comment"):
            if key in metadata_override:
                clean[key] = metadata_override[key]
        if clean:
            desc["metadata_override"] = clean

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(desc, f, default_flow_style=False, sort_keys=False,
                       allow_unicode=True)


def read_vasdf(path: str) -> Dict[str, Any]:
    """Read a ``.vasdf`` sidecar and return a normalized descriptor.

    Returned shape::

        {
            "kind": "virtual_split",
            "parent_path": str,
            "source_id": str,
            "label": str,
            "split": {"axis": "raw_voltage", "lo": float, "hi": float},
            "metadata_override": {... known keys only ...},
        }

    Raises ``ValueError`` for missing required fields or wrong kind.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    kind = raw.get("kind")
    if kind != VASDF_KIND:
        raise ValueError(
            f"Not a virtual_split descriptor (kind={kind!r}): {path}")

    parent_path = raw.get("parent_path")
    if not parent_path:
        raise ValueError(f"Missing parent_path in {path}")
    source_id = raw.get("source_id")
    if not source_id:
        raise ValueError(f"Missing source_id in {path}")

    split = raw.get("split") or {}
    if "lo" not in split or "hi" not in split:
        raise ValueError(f"Missing split.lo / split.hi in {path}")

    md = raw.get("metadata_override") or {}
    clean_md = {}
    for key in ("cooler_v", "laser_sp", "comment"):
        if key in md:
            clean_md[key] = md[key]

    return {
        "kind": VASDF_KIND,
        "parent_path": str(parent_path),
        "source_id": str(source_id),
        "label": str(raw.get("label") or source_id),
        "split": {
            "axis": str(split.get("axis", "raw_voltage")),
            "lo": float(split["lo"]),
            "hi": float(split["hi"]),
        },
        "metadata_override": clean_md,
    }


def default_vasdf_paths(parent_asdf_path: str, suffixes):
    """Suggest default ``<parent_base>_<suffix>.vasdf`` paths next to
    the parent.

    ``suffixes`` is an iterable of short side identifiers (e.g.
    ``["A", "B"]``); pass only the suffix that should be appended to
    the parent's basename, not the full source_id (which usually
    already contains the run number, so passing ``"7509_A"`` would
    duplicate it).

    Convenience for the editor's "Save…" default — callers may save
    anywhere.
    """
    base = os.path.basename(parent_asdf_path)
    if base.lower().endswith(".asdf"):
        base = base[:-len(".asdf")]
    parent_dir = os.path.dirname(os.path.abspath(parent_asdf_path))
    return [os.path.join(parent_dir, f"{base}_{s}{VASDF_EXT}")
            for s in suffixes]
