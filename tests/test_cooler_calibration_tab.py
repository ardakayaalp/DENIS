"""The Cooler Calibration tab: project type, calibration, applying it.

Covers the parts a user touches: a third kind of project that is NOT
an isotope, a tab that appears with it, the calibration computed from
scanned points, and writing the resulting offset into the Source
blocks of the projects that need it.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_cooler_calibration_tab.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from cls_estimations.cooler_calibration import (  # noqa: E402
    LITERATURE_A, ScanPoint,
)


def _points(zero_171=-33.46, zero_173=-14.00):
    """Scan points whose isotope lines cross literature where asked."""
    a171, a173 = LITERATURE_A["171Yb"], LITERATURE_A["173Yb"]
    out = []
    for run in ("7547", "7507"):
        for dv in (-50.0, -25.0, 0.0, 25.0, 50.0):
            out.append(ScanPoint(run, "171Yb", dv,
                                 a171 - 0.21 * (dv - zero_171), 0.5))
    for run in ("7502",):
        for dv in (-50.0, -25.0, 0.0, 25.0, 50.0):
            out.append(ScanPoint(run, "173Yb", dv,
                                 a173 + 0.058 * (dv - zero_173), 0.1))
    return out


class ProjectTypeTests(unittest.TestCase):

    def _tab(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        return at

    def test_a_calibration_project_says_so_on_its_tab(self):
        at = self._tab()
        p = at._add_project("Yb_cal", is_calibration=True)
        self.assertTrue(p.is_calibration)
        self.assertIn("Cooler Cal.", at._tab_text(p))

    def test_the_flag_survives_a_save(self):
        at = self._tab()
        p = at._add_project("Yb_cal", is_calibration=True)
        d = p.to_dict()
        self.assertTrue(d["is_calibration"])
        q = at._add_project("restored", config=d)
        self.assertTrue(q.is_calibration)
        self.assertIn("Cooler Cal.", at._tab_text(q))

    def test_the_tab_comes_and_goes_with_the_projects(self):
        at = self._tab()
        at._add_project("sample")
        self.assertFalse(at._cal_tab_visible)
        at._add_project("Yb_cal", is_calibration=True)
        self.assertTrue(at._cal_tab_visible)
        titles = [at._project_tabs.tabText(i)
                  for i in range(at._project_tabs.count())]
        self.assertIn("Cooler Calibration", titles)

    def test_it_is_not_an_isotope(self):
        """It calibrates the voltage; it has no shift to measure, so
        it must not appear in the Isotope Shifts table or in the GP's
        project lists."""
        at = self._tab()
        at._add_project("70Ge")
        at._add_project("Yb_cal", is_calibration=True)
        ist = at._is_tab
        ist._refresh_projects()
        names = [e.project_combo.currentText() for e in ist._entries]
        self.assertNotIn("Yb_cal", names)
        panel = at._gp_tab
        panel.refresh_projects()
        listed = [panel._sample_list.item(i).text()
                  for i in range(panel._sample_list.count())]
        listed += [panel._ref_list.item(i).text()
                   for i in range(panel._ref_list.count())]
        self.assertFalse(any("Yb_cal" in t for t in listed))


class CalibrationTabTests(unittest.TestCase):

    def _tab(self, names=("Yb171_cal", "Yb173_cal")):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        for n in names:
            at._add_project(n, is_calibration=True)
        cal = at._cal_tab
        cal.refresh_projects()
        return at, cal

    def _row(self, cal, name):
        for r in range(cal._proj_table.rowCount()):
            if cal._proj_table.item(r, 1).text() == name:
                return r
        raise AssertionError(f"{name} not listed")

    def test_it_lists_the_calibration_projects_only(self):
        at, cal = self._tab()
        at._add_project("70Ge")
        cal.refresh_projects()
        listed = [cal._proj_table.item(r, 1).text()
                  for r in range(cal._proj_table.rowCount())]
        self.assertEqual(sorted(listed), ["Yb171_cal", "Yb173_cal"])

    def test_the_isotope_is_guessed_and_brings_its_literature_value(self):
        _at, cal = self._tab()
        r = self._row(cal, "Yb173_cal")
        self.assertEqual(cal._proj_table.cellWidget(r, 2).currentText(),
                         "173Yb")
        self.assertAlmostEqual(
            float(cal._proj_table.item(r, 3).text()),
            LITERATURE_A["173Yb"], places=6)

    def test_changing_the_isotope_updates_the_literature_value(self):
        _at, cal = self._tab(names=("Yb171_cal",))
        r = self._row(cal, "Yb171_cal")
        cal._proj_table.cellWidget(r, 2).setCurrentText("173Yb")
        self.assertAlmostEqual(
            float(cal._proj_table.item(r, 3).text()),
            LITERATURE_A["173Yb"], places=6)

    def test_a_typed_literature_value_is_not_overwritten(self):
        """Someone using a different reference must not have it
        silently replaced."""
        _at, cal = self._tab(names=("Yb171_cal",))
        r = self._row(cal, "Yb171_cal")
        cal._proj_table.item(r, 3).setText("12345.6789000000")
        cal._proj_table.cellWidget(r, 2).setCurrentText("173Yb")
        self.assertAlmostEqual(
            float(cal._proj_table.item(r, 3).text()), 12345.6789,
            places=4)

    def test_refreshing_does_not_leak_cell_widgets(self):
        """Replacing a cell widget leaves the old one parented to the
        viewport, still visible at its old geometry. After a few
        refreshes the table carried 20 widgets for 4 cells and painted
        project names underneath isotope combos."""
        from PySide6.QtWidgets import QCheckBox, QComboBox
        _at, cal = self._tab()
        for _ in range(5):
            cal.refresh_projects()
        live = [c for c in cal._proj_table.viewport().children()
                if isinstance(c, (QComboBox, QCheckBox))]
        # one checkbox + one isotope combo per row, and nothing else
        self.assertEqual(len(live), 2 * cal._proj_table.rowCount())

    def test_the_offset_grid(self):
        _at, cal = self._tab()
        cal._dv_min.setValue(-50.0)
        cal._dv_max.setValue(50.0)
        cal._dv_steps.setValue(15)
        offs = cal.offsets()
        self.assertEqual(len(offs), 15)
        self.assertAlmostEqual(offs[0], -50.0)
        self.assertAlmostEqual(offs[-1], 50.0)

    def test_computing_from_scanned_points(self):
        _at, cal = self._tab()
        cal._points = _points()
        cal.compute()
        self.assertIsNotNone(cal._result)
        self.assertAlmostEqual(cal._result.dv, -29.25, delta=0.1)
        self.assertAlmostEqual(cal._result.scan_lo, -33.46, delta=0.05)
        self.assertAlmostEqual(cal._result.scan_hi, -14.00, delta=0.05)
        # ...and it reaches the screen.
        self.assertIn("-29.2", cal._result_label.text().replace("−", "-"))
        self.assertEqual(cal._run_table.rowCount(), 3)   # 2 + 1 runs
        self.assertEqual(cal._sum_table.rowCount(), 2)   # 2 isotopes
        self.assertTrue(cal._apply_btn.isEnabled())

    def test_the_plot_data_carries_both_isotopes_and_the_crossing(self):
        _at, cal = self._tab()
        cal._points = _points()
        cal.compute()
        d = cal.plot_data()
        self.assertEqual([i["label"] for i in d["isotopes"]],
                         ["171Yb", "173Yb"])
        self.assertAlmostEqual(d["dv"], -29.25, delta=0.1)
        self.assertEqual(len(d["isotopes"][0]["band"]), len(d["grid"]))

    def test_the_figure_renders(self):
        _at, cal = self._tab()
        cal._points = _points()
        cal.compute()
        self.assertTrue(cal._figure.axes)
        labels = cal._figure.axes[0].get_legend_handles_labels()[1]
        self.assertTrue(any("intersection" in t for t in labels))

    def test_one_isotope_alone_reports_why(self):
        _at, cal = self._tab(names=("Yb171_cal",))
        cal._points = [p for p in _points() if p.isotope == "171Yb"]
        cal.compute()
        self.assertIsNone(cal._result.dv)
        self.assertFalse(cal._apply_btn.isEnabled())
        self.assertIn("two isotopes", cal._result_label.text())


class ApplyTests(unittest.TestCase):

    def _setup(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        at._add_project("Yb171_cal", is_calibration=True)
        at._add_project("Yb173_cal", is_calibration=True)
        sample = at._add_project("70Ge")
        ref = at._add_project("74Ge", is_reference=True)
        cal = at._cal_tab
        cal.refresh_projects()
        cal._points = _points()
        cal.compute()
        return at, cal, sample, ref

    @staticmethod
    def _offset(project):
        from gui.analysis.blocks import SourceBlock
        for b in project._blocks:
            if isinstance(b, SourceBlock):
                return b._cooler_offset.value()
        return None

    def test_it_writes_the_offset_into_the_source_blocks(self):
        _at, cal, sample, ref = self._setup()
        n = cal.apply_offset(cal._result.dv, [sample, ref])
        self.assertEqual(n, 2)
        # The field keeps millivolts. On a 30 kV beam that is a 3e-8
        # relative effect, far below the ~1.7 V the calibration itself
        # is uncertain by.
        for p in (sample, ref):
            self.assertAlmostEqual(self._offset(p), cal._result.dv,
                                   places=3)

    def test_the_field_keeps_millivolts(self):
        """Enough that rounding never matters next to the
        calibration's own uncertainty."""
        _at, cal, sample, _ref = self._setup()
        cal.apply_offset(-29.2485074626, [sample])
        self.assertAlmostEqual(self._offset(sample), -29.249, places=6)

    def test_the_offset_then_travels_with_the_project(self):
        """It has to survive a save, or the analysis silently reverts
        to the uncalibrated frame."""
        _at, cal, sample, _ref = self._setup()
        cal.apply_offset(-29.25, [sample])
        cfg = sample.to_dict()
        from gui.analysis.blocks import SourceBlock
        src = next(b for b in sample._blocks
                   if isinstance(b, SourceBlock))
        blocks = cfg.get("blocks") or []
        saved = next(b for b in blocks
                     if b.get("type") == src.BLOCK_TYPE)
        self.assertAlmostEqual(saved["cooler_offset_v"], -29.25,
                               places=6)

    def test_the_calibration_projects_are_not_targets(self):
        """Applying the offset to the Yb runs that measured it would
        move the very spectra the calibration was derived from."""
        _at, cal, _s, _r = self._setup()
        names = [p.project_name for p in cal._apply_targets()]
        self.assertNotIn("Yb171_cal", names)
        self.assertIn("70Ge", names)

    def test_applying_announces_itself(self):
        _at, cal, sample, _ref = self._setup()
        seen = []
        cal.calibration_applied.connect(
            lambda v, s: seen.append((v, s)))
        cal.apply_offset(-29.25, [sample])
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0][0], -29.25)


class PersistenceTests(unittest.TestCase):

    def _tab(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        at._add_project("Yb171_cal", is_calibration=True)
        at._add_project("Yb173_cal", is_calibration=True)
        cal = at._cal_tab
        cal.refresh_projects()
        return at, cal

    def test_a_round_trip_keeps_the_calibration(self):
        """The scan is the expensive part -- 75 fits. Reopening a
        session must not mean running it again."""
        _at, cal = self._tab()
        cal._dv_min.setValue(-40.0)
        cal._dv_steps.setValue(9)
        cal._points = _points()
        cal.compute()
        d = cal.to_dict()

        _at2, cal2 = self._tab()
        cal2.from_dict(d)
        self.assertEqual(cal2._dv_min.value(), -40.0)
        self.assertEqual(cal2._dv_steps.value(), 9)
        self.assertEqual(len(cal2._points), len(cal._points))
        self.assertAlmostEqual(cal2._result.dv, cal._result.dv,
                               places=6)

    def test_an_empty_save_is_harmless(self):
        _at, cal = self._tab()
        cal.from_dict({})
        cal.from_dict(None)

    def test_a_calibration_can_be_taken_from_another_session(self):
        """'If there is no calibration in this session, load the one
        from the session that has it.'"""
        import tempfile
        import yaml
        _at, cal = self._tab()
        cal._points = _points()
        cal.compute()
        path = os.path.join(tempfile.mkdtemp(), "other.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"analysis": {"cooler_calibration": cal.to_dict()}},
                      f)

        _at2, cal2 = self._tab()
        res = cal2.load_from_file(path)
        self.assertAlmostEqual(res.dv, cal._result.dv, places=6)

    def test_a_file_without_one_says_so(self):
        import tempfile
        import yaml
        _at, cal = self._tab()
        path = os.path.join(tempfile.mkdtemp(), "plain.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"analysis": {"projects": []}}, f)
        with self.assertRaises(ValueError):
            cal.load_from_file(path)


if __name__ == "__main__":
    unittest.main()
