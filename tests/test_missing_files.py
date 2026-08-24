"""Tests for the YAML missing-file scan / remap / prune helpers.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Exercises gui.missing_files against synthetic session-YAML dicts that
mix the legacy single-project Pre-Analysis layout, the live multi-
project layout, and Analysis Source blocks. Verifies that scanning
surfaces every missing on-disk path (ignoring merged:// entries),
remapping rewrites located paths in place, and pruning drops the paths
the user could not locate.

Depends on: gui.missing_files (scan_missing_paths, remap_paths_inplace,
prune_missing_paths_inplace).
"""

import os
import tempfile
import unittest

from gui.missing_files import (
    scan_missing_paths, remap_paths_inplace,
    prune_missing_paths_inplace,
)


def _yaml(pa_files=None, pa_projects=None, an_blocks=None):
    """Build a minimal session-YAML dict for the helpers to chew on.

    ``pa_files`` populates the legacy single-project layout
    (``preanalysis.files``); ``pa_projects`` populates the multi-
    project layout (``preanalysis.projects[].files``) -- the live
    save format from PreAnalysisContainer._build_config_dict.
    """
    out = {}
    pa = {}
    if pa_files is not None:
        pa["files"] = pa_files
    if pa_projects is not None:
        pa["projects"] = pa_projects
    if pa:
        out["preanalysis"] = pa
    if an_blocks is not None:
        out["analysis"] = {"projects": [{"blocks": an_blocks}]}
    return out


class ScanMissingPathsTests(unittest.TestCase):

    def setUp(self):
        # One real on-disk file so "exists" cases use a path that
        # actually exists. NamedTemporaryFile + delete=False so it
        # survives the close on Windows; tearDown removes it.
        f = tempfile.NamedTemporaryFile(
            suffix=".asdf", delete=False)
        f.close()
        self.real = f.name
        self.addCleanup(os.unlink, self.real)
        # A path guaranteed not to exist.
        self.fake = os.path.join(
            tempfile.gettempdir(), "denis_does_not_exist_001.asdf")
        self.fake2 = os.path.join(
            tempfile.gettempdir(), "denis_does_not_exist_002.asdf")

    def test_empty_dict(self):
        self.assertEqual(scan_missing_paths({}), [])

    def test_no_missing(self):
        d = _yaml(pa_files=[{"path": self.real}])
        self.assertEqual(scan_missing_paths(d), [])

    def test_pa_missing(self):
        d = _yaml(pa_files=[{"path": self.fake}])
        self.assertEqual(scan_missing_paths(d), [self.fake])

    def test_pa_projects_layer(self):
        # The live PA save format wraps files under projects[]. A
        # scanner that only looked at preanalysis.files[] would miss
        # everything and the user would still see "File Not Found"
        # popups during _restore_from_dict after locating.
        d = _yaml(pa_projects=[
            {"name": "PA_1", "files": [{"path": self.fake}]},
            {"name": "PA_2", "files": [{"path": self.fake2}]},
        ])
        self.assertEqual(
            scan_missing_paths(d), [self.fake, self.fake2])

    def test_pa_legacy_and_projects_both_scanned(self):
        # Defensive: a hand-edited YAML that mixes both layouts
        # (shouldn't happen in practice) should still surface every
        # missing path.
        d = _yaml(
            pa_files=[{"path": self.fake}],
            pa_projects=[{"name": "PA_1",
                           "files": [{"path": self.fake2}]}],
        )
        self.assertEqual(
            sorted(scan_missing_paths(d)),
            sorted([self.fake, self.fake2]))

    def test_analysis_missing_only_source_blocks(self):
        # A non-Source block listing files is malformed YAML, but the
        # scanner should skip it rather than crash. Only Source-typed
        # blocks contribute paths -- mirrors what the loader does.
        d = _yaml(an_blocks=[
            {"type": "Source", "files": [{"path": self.fake}]},
            {"type": "Model",  "files": [{"path": self.fake2}]},
        ])
        self.assertEqual(scan_missing_paths(d), [self.fake])

    def test_merged_paths_ignored(self):
        d = _yaml(pa_files=[
            {"path": "merged://run_7164_7165"},
            {"path": self.fake},
        ])
        self.assertEqual(scan_missing_paths(d), [self.fake])

    def test_dedupes_across_sections(self):
        d = _yaml(
            pa_files=[{"path": self.fake}],
            an_blocks=[{"type": "Source",
                        "files": [{"path": self.fake}]}],
        )
        self.assertEqual(scan_missing_paths(d), [self.fake])

    def test_preserves_order(self):
        d = _yaml(
            pa_files=[{"path": self.fake}, {"path": self.real}],
            an_blocks=[{"type": "Source",
                        "files": [{"path": self.fake2}]}],
        )
        self.assertEqual(scan_missing_paths(d), [self.fake, self.fake2])

    def test_handles_none_lists(self):
        # YAML with `files: null` (hand-edits) or missing keys
        # shouldn't crash the scanner.
        self.assertEqual(scan_missing_paths(
            {"preanalysis": {"files": None}}), [])
        self.assertEqual(scan_missing_paths(
            {"analysis": {"projects": None}}), [])
        self.assertEqual(scan_missing_paths(
            {"analysis": {"projects": [{"blocks": None}]}}), [])


class RemapPathsInplaceTests(unittest.TestCase):

    def test_empty_remap_noop(self):
        d = _yaml(pa_files=[{"path": "/x/a.asdf"}])
        self.assertEqual(remap_paths_inplace(d, {}), 0)
        self.assertEqual(d["preanalysis"]["files"][0]["path"], "/x/a.asdf")

    def test_partial_remap_only_replaces_matches(self):
        # User located 1 of 2 missing files. The second stays as-is
        # so the prune step can drop it.
        d = _yaml(pa_files=[
            {"path": "/x/a.asdf"},
            {"path": "/x/b.asdf"},
        ])
        n = remap_paths_inplace(d, {"/x/a.asdf": "/y/a.asdf"})
        self.assertEqual(n, 1)
        self.assertEqual(
            [f["path"] for f in d["preanalysis"]["files"]],
            ["/y/a.asdf", "/x/b.asdf"])

    def test_remap_across_sections(self):
        d = _yaml(
            pa_files=[{"path": "/x/a.asdf"}],
            an_blocks=[{"type": "Source",
                        "files": [{"path": "/x/a.asdf"}]}],
        )
        n = remap_paths_inplace(d, {"/x/a.asdf": "/y/a.asdf"})
        self.assertEqual(n, 2)
        self.assertEqual(
            d["preanalysis"]["files"][0]["path"], "/y/a.asdf")
        self.assertEqual(
            d["analysis"]["projects"][0]["blocks"][0]["files"][0]["path"],
            "/y/a.asdf")

    def test_remap_pa_projects_layer(self):
        # The bug that prompted this test set: located paths went into
        # remap, but remap_paths_inplace only updated the legacy
        # preanalysis.files layer and never touched
        # preanalysis.projects[].files. Result: the YAML was rewritten
        # with the new path on the legacy slot (which is empty), and
        # the projects[] slot still pointed at the missing file --
        # PA's own per-file warning popup then complained.
        d = _yaml(pa_projects=[
            {"name": "PA_1", "files": [{"path": "/x/a.asdf"}]},
            {"name": "PA_2", "files": [{"path": "/x/b.asdf"},
                                         {"path": "/x/a.asdf"}]},
        ])
        n = remap_paths_inplace(d, {"/x/a.asdf": "/y/a.asdf"})
        # Two slots reference /x/a.asdf (PA_1's only file + PA_2's
        # second), so two replacements happen.
        self.assertEqual(n, 2)
        self.assertEqual(
            d["preanalysis"]["projects"][0]["files"][0]["path"],
            "/y/a.asdf")
        self.assertEqual(
            d["preanalysis"]["projects"][1]["files"][1]["path"],
            "/y/a.asdf")
        # Untouched entry stays as-is.
        self.assertEqual(
            d["preanalysis"]["projects"][1]["files"][0]["path"],
            "/x/b.asdf")


class PruneMissingPathsInplaceTests(unittest.TestCase):

    def test_drops_only_matching_entries(self):
        d = _yaml(pa_files=[
            {"path": "/x/keep.asdf"},
            {"path": "/x/drop.asdf"},
        ])
        prune_missing_paths_inplace(d, {"/x/drop.asdf"})
        self.assertEqual(
            [f["path"] for f in d["preanalysis"]["files"]],
            ["/x/keep.asdf"])

    def test_prunes_pa_projects_layer(self):
        d = _yaml(pa_projects=[
            {"name": "PA_1", "files": [{"path": "/x/keep.asdf"},
                                          {"path": "/x/drop.asdf"}]},
        ])
        prune_missing_paths_inplace(d, {"/x/drop.asdf"})
        self.assertEqual(
            [f["path"] for f in
             d["preanalysis"]["projects"][0]["files"]],
            ["/x/keep.asdf"])

    def test_prunes_analysis_source_blocks(self):
        d = _yaml(an_blocks=[{
            "type": "Source",
            "files": [{"path": "/x/a.asdf"}, {"path": "/x/b.asdf"}],
        }])
        prune_missing_paths_inplace(d, {"/x/a.asdf"})
        self.assertEqual(
            [f["path"] for f in
             d["analysis"]["projects"][0]["blocks"][0]["files"]],
            ["/x/b.asdf"])

    def test_empty_missing_noop(self):
        d = _yaml(pa_files=[{"path": "/x/a.asdf"}])
        prune_missing_paths_inplace(d, set())
        self.assertEqual(len(d["preanalysis"]["files"]), 1)


if __name__ == "__main__":
    unittest.main()
