"""Unit tests for merged-entry handling in the Analysis SourceBlock.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Locks in four merge-handling behaviors:
 1. ``_refresh_meta_content`` on a merged entry must NOT call
    ``Load_Run`` on the synthetic ``merged://`` path (which raises
    "Protocol not known: merged").
 2. ``_build_merged_meta_html`` produces HTML the Run-Info dialog can
    display.
 3. The Source-block badge appears on a voltage-merged entry when
    Source bin_mode is Frequency (info icon, Doppler-projected), and
    hides when bin_mode is Raw Voltage; the reverse holds for
    frequency-merged entries.
 4. The badge tracks live bin_mode changes via currentTextChanged.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_phase1_merge_fixes -v

Depends on: gui.analysis.blocks (SourceBlock); PySide6 for the
QApplication fixture.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class BuildMergedMetaHtmlTests(unittest.TestCase):
    """The HTML must reflect what's in merged_data without touching the
    filesystem -- merged spectra have no ASDF behind them."""

    def setUp(self):
        _ensure_app()

    def test_voltage_merged_meta_html_has_voltage_axis_and_runs(self):
        from gui.analysis.blocks import SourceBlock

        merged_data = {
            "merged_name": "merged_7735_7740",
            "x": np.array([-50.0, 0.0, 50.0, 100.0]),
            "y": np.array([2.0, 5.0, 11.0, 4.0]),
            "yerr": np.array([1.4, 2.2, 3.3, 2.0]),
            "x_unit": "V",
            "source_runs": ["7735", "7740"],
            "source_files": ["/data/run_7735.asdf",
                             "/data/run_7740.asdf"],
            "per_run": [],
            "bin_step_mhz": 0,
            "metadata": {"imported_from": "preanalysis",
                         "merge_domain": "voltage"},
        }

        sb = SourceBlock()
        html = sb._build_merged_meta_html(merged_data)

        self.assertIn("merged_7735_7740", html)
        self.assertIn("Voltage (V)", html)  # axis label
        self.assertIn("Bins:           4", html)
        # Total counts = 2 + 5 + 11 + 4 = 22
        self.assertIn("Total counts:   22", html)
        # Range
        self.assertIn("-50.000", html)
        self.assertIn("100.000", html)
        # Source runs
        self.assertIn("7735", html)
        self.assertIn("7740", html)
        # Metadata block
        self.assertIn("imported_from", html)
        self.assertIn("preanalysis", html)
        self.assertIn("merge_domain", html)

    def test_frequency_merged_meta_html_says_rest_frame(self):
        from gui.analysis.blocks import SourceBlock

        merged_data = {
            "merged_name": "merged_freq",
            "x": np.array([-100.0, 0.0, 100.0]),
            "y": np.array([1.0, 7.0, 2.0]),
            "yerr": np.array([1.0, 2.6, 1.4]),
            "x_unit": "MHz",
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "per_run": [],
            "bin_step_mhz": 100,
            "metadata": {},
        }

        sb = SourceBlock()
        html = sb._build_merged_meta_html(merged_data)

        self.assertIn("Frequency (rest-frame MHz)", html)
        self.assertIn("MHz", html)
        self.assertNotIn("Voltage (V)", html)


class RefreshMetaContentRoutingTests(unittest.TestCase):
    """The early-return branch in _refresh_meta_content must take over
    for merged entries -- the existing _build_meta_html path calls
    Load_Run on the synthetic merged:// path and raises "Protocol not
    known: merged"."""

    def setUp(self):
        _ensure_app()

    def test_merged_entry_does_not_call_build_meta_html(self):
        from gui.analysis.blocks import SourceBlock

        sb = SourceBlock()
        # Inject a merged entry directly.
        merged_data = {
            "merged_name": "merged_X",
            "x": np.array([0.0, 1.0]),
            "y": np.array([1.0, 2.0]),
            "yerr": np.array([1.0, 1.4]),
            "x_unit": "MHz",
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "per_run": [],
            "bin_step_mhz": 0,
        }
        sb._add_merged_entry(merged_data)
        # Force the meta window open via the open_if_closed path.
        sb._preview_index = 0

        with patch.object(sb, "_build_meta_html") as m_build, \
                patch.object(sb, "_build_merged_meta_html",
                              wraps=sb._build_merged_meta_html) as m_merged:
            sb._refresh_meta_content(open_if_closed=True)

        # The ASDF-loading path must not have been entered.
        m_build.assert_not_called()
        m_merged.assert_called_once()
        # And the merged HTML builder got the right dict.
        passed = m_merged.call_args.args[0]
        self.assertEqual(passed["merged_name"], "merged_X")

    def test_navigate_merged_then_nonmerged_reuses_meta_window(self):
        """The Meta window should be reused across merged↔non-merged
        navigation: only its title and text update, not the dialog
        itself. Mocks _build_meta_html since we have no real ASDF.
        """
        from gui.analysis.blocks import SourceBlock

        sb = SourceBlock()

        # Inject a merged entry plus a regular (mocked) entry.
        merged_data = {
            "merged_name": "merged_X",
            "x": np.array([0.0, 1.0]),
            "y": np.array([1.0, 2.0]),
            "yerr": np.array([1.0, 1.4]),
            "x_unit": "MHz",
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "per_run": [],
            "bin_step_mhz": 0,
        }
        sb._add_merged_entry(merged_data)
        # Hand-construct a regular entry whose path will be passed to
        # the mocked _build_meta_html.
        from PySide6.QtWidgets import QCheckBox
        sb._file_entries.append({
            "path": "/data/run_X.asdf",
            "run_number": "X",
            "checkbox": QCheckBox(),
            "is_merged": False,
        })
        sb._file_entries[-1]["checkbox"].setChecked(True)

        with patch.object(sb, "_build_meta_html",
                           return_value="<pre>NORMAL_HTML</pre>"):
            # Start on the merged entry: opens the window.
            sb._preview_index = 0
            sb._refresh_meta_content(open_if_closed=True)
            first_win = sb._meta_win
            self.assertIsNotNone(first_win)
            self.assertIn("merged_X", first_win.windowTitle())

            # Navigate to the non-merged entry: window is reused.
            sb._preview_index = 1
            sb._refresh_meta_content(open_if_closed=False)
            self.assertIs(sb._meta_win, first_win)
            self.assertIn("run_X.asdf", first_win.windowTitle())
            self.assertIn("NORMAL_HTML", sb._meta_label.text())

            # Navigate back to merged: window still reused, title flips.
            sb._preview_index = 0
            sb._refresh_meta_content(open_if_closed=False)
            self.assertIs(sb._meta_win, first_win)
            self.assertIn("merged_X", first_win.windowTitle())


class MergedWarningBadgeTests(unittest.TestCase):
    """Voltage merge in a Frequency-mode Source must surface a ⚠ badge.
    The badge follows live bin_mode changes via the existing
    currentTextChanged signal -- no need to re-import the entry."""

    def setUp(self):
        _ensure_app()

    def test_voltage_merge_with_frequency_bin_mode_shows_info_badge(self):
        """Voltage-merge + Frequency bin_mode now produces an INFO
        badge (ℹ) — Phase 4's V→F projection makes the fit work, so
        the user just needs to know the projection will happen and
        with which metadata."""
        from gui.analysis.blocks import SourceBlock

        sb = SourceBlock()
        sb._bin_mode.setCurrentText("Frequency")
        merged_data = {
            "merged_name": "merged_V",
            "x": np.array([0.0, 50.0, 100.0]),
            "y": np.array([1.0, 2.0, 3.0]),
            "yerr": np.array([1.0, 1.4, 1.7]),
            "x_unit": "V",
            "source_runs": ["a", "b"],
            "source_files": ["/p/a", "/p/b"],
            "per_run": [],
            "bin_step_mhz": 0,
            "merge_metadata": {"cooler_v": 29895.4,
                                "laser_sp": 15975.4150,
                                "mass_amu": 47.95,
                                "harmonic": 2},
        }
        sb._add_merged_entry(merged_data)
        entry = sb._file_entries[-1]
        self.assertTrue(entry["warn_label"].isVisibleTo(entry["widget"]))
        # Info icon and tooltip refer to the projection.
        self.assertEqual(entry["warn_label"].text(), "ℹ")
        self.assertIn("Doppler-projected",
                       entry["warn_label"].toolTip())
        # Tooltip surfaces the merge metadata so the user can verify.
        self.assertIn("29895", entry["warn_label"].toolTip())

    def test_voltage_merge_with_raw_voltage_bin_mode_hides_badge(self):
        from gui.analysis.blocks import SourceBlock

        sb = SourceBlock()
        sb._bin_mode.setCurrentText("Raw Voltage")
        merged_data = {
            "merged_name": "merged_V",
            "x": np.array([0.0, 50.0]),
            "y": np.array([1.0, 2.0]),
            "yerr": np.array([1.0, 1.4]),
            "x_unit": "V",
            "source_runs": ["a"],
            "source_files": ["/p/a"],
            "per_run": [],
            "bin_step_mhz": 0,
        }
        sb._add_merged_entry(merged_data)
        entry = sb._file_entries[-1]
        self.assertFalse(entry["warn_label"].isVisibleTo(entry["widget"]))

    def test_badge_updates_when_bin_mode_changes(self):
        from gui.analysis.blocks import SourceBlock

        sb = SourceBlock()
        sb._bin_mode.setCurrentText("Raw Voltage")
        merged_data = {
            "merged_name": "merged_V",
            "x": np.array([0.0, 50.0]),
            "y": np.array([1.0, 2.0]),
            "yerr": np.array([1.0, 1.4]),
            "x_unit": "V",
            "source_runs": ["a"],
            "source_files": ["/p/a"],
            "per_run": [],
            "bin_step_mhz": 0,
        }
        sb._add_merged_entry(merged_data)
        entry = sb._file_entries[-1]
        self.assertFalse(entry["warn_label"].isVisibleTo(entry["widget"]))

        sb._bin_mode.setCurrentText("Frequency")
        # Signal-driven refresh.
        self.assertTrue(entry["warn_label"].isVisibleTo(entry["widget"]))

    def test_frequency_merge_with_voltage_mode_also_warns(self):
        from gui.analysis.blocks import SourceBlock

        sb = SourceBlock()
        sb._bin_mode.setCurrentText("Raw Voltage")
        merged_data = {
            "merged_name": "merged_F",
            "x": np.array([-100.0, 0.0, 100.0]),
            "y": np.array([1.0, 7.0, 2.0]),
            "yerr": np.array([1.0, 2.6, 1.4]),
            "x_unit": "MHz",
            "source_runs": ["a"],
            "source_files": ["/p/a"],
            "per_run": [],
            "bin_step_mhz": 100,
        }
        sb._add_merged_entry(merged_data)
        entry = sb._file_entries[-1]
        self.assertTrue(entry["warn_label"].isVisibleTo(entry["widget"]))
        self.assertIn("Frequency-merged",
                       entry["warn_label"].toolTip())


# ``_PreAnalysisMergeDialog`` has been deleted; PA now routes through
# the shared ``gui.analysis.merge.MergeDialog``. The default-domain and
# bin-step Auto-vs-Manual behaviors are tested against that new dialog.


if __name__ == "__main__":
    unittest.main()
