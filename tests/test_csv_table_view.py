"""The Results tab's CSV viewer: run shading, sorting, filtering.

A parameters.csv is ~32 rows per run in one long list. The questions
it is opened for -- "just the centroids", "which run has the worst
error" -- had no answer short of reading every row.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_csv_table_view.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui.csv_table_view import (  # noqa: E402
    CsvTableView, run_column, sort_key, tints,
)


def _params():
    """Three runs of a parameters.csv, the shape Arda's are."""
    rows = []
    for run, c, ce in ((7986, -265.5975, 3.690030056),
                       (7996, -254.4764951, 3.757566601),
                       (8004, -270.1, 2.5)):
        for par, val, err in (("centroid", c, ce), ("Al", 0.0, 0.0),
                              ("FWHMG", 110.5, 29.6), ("p0", 12.2, 0.61)):
            rows.append({"run_number": run, "Parameter": par,
                         "Value": val, "Error": err, "Vary": par != "Al",
                         "Model": "Model_1_bkg" if par == "p0"
                         else "Model_1"})
    return pd.DataFrame(rows)


def _shown(tv, col):
    """Visible texts of *col*, in on-screen order."""
    px = tv.proxy
    c = tv.model.column_names().index(col)
    return [px.data(px.index(r, c)) for r in range(px.rowCount())]


class SortKeyTests(unittest.TestCase):

    def test_numbers_sort_as_numbers(self):
        """A text sort puts '-265' after '10'."""
        vals = ["10", "-265.5975", "3.69", "100000000"]
        self.assertEqual(sorted(vals, key=sort_key),
                         ["-265.5975", "3.69", "10", "100000000"])

    def test_blanks_and_nan_go_last(self):
        vals = [float("nan"), 2.0, "nan", "", "abc", 1.0]
        out = sorted(vals, key=sort_key)
        self.assertEqual(out[:2], [1.0, 2.0])
        self.assertEqual(out[2], "abc")


class FilterTests(unittest.TestCase):

    def _view(self):
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        return tv

    def test_the_quick_filter_defaults_to_the_parameter_column(self):
        tv = self._view()
        self.assertEqual(tv._col_combo.currentText(), "Parameter")

    def test_just_the_centroids(self):
        tv = self._view()
        tv._quick.setText("centroid")
        self.assertEqual(_shown(tv, "Parameter"), ["centroid"] * 3)
        self.assertIn("3 of 12", tv._count.text())

    def test_an_exact_value_does_not_match_as_a_substring(self):
        """'Al' is a parameter. As a substring it would also match
        'False' in Vary and nothing sensible; with the Model column
        selected, 'Model_1' must not drag in 'Model_1_bkg'."""
        tv = self._view()
        tv._quick.setText("Al")
        self.assertEqual(set(_shown(tv, "Parameter")), {"Al"})
        tv._col_combo.setCurrentText("Model")
        tv._quick.setText("Model_1")
        self.assertEqual(set(_shown(tv, "Model")), {"Model_1"})

    def test_a_partial_term_matches_as_a_substring(self):
        tv = self._view()
        tv._quick.setText("FWHM")
        self.assertEqual(set(_shown(tv, "Parameter")), {"FWHMG"})

    def test_terms_are_ored(self):
        tv = self._view()
        tv._quick.setText("centroid, Al")
        self.assertEqual(set(_shown(tv, "Parameter")), {"centroid", "Al"})
        self.assertEqual(tv.proxy.rowCount(), 6)

    def test_the_checklist_filter(self):
        tv = self._view()
        c = tv.model.column_names().index("run_number")
        tv.proxy.set_allowed(c, {"7996"})
        self.assertEqual(set(_shown(tv, "run_number")), {"7996"})
        tv.proxy.set_allowed(c, None)
        self.assertEqual(tv.proxy.rowCount(), 12)

    def test_filters_combine(self):
        tv = self._view()
        c = tv.model.column_names().index("run_number")
        tv.proxy.set_allowed(c, {"7986", "8004"})
        tv._quick.setText("centroid")
        self.assertEqual(tv.proxy.rowCount(), 2)

    def test_clear_filters_restores_everything(self):
        tv = self._view()
        tv._quick.setText("centroid")
        tv.view.sortByColumn(3, Qt.SortOrder.DescendingOrder)
        tv.clear_filters()
        self.assertEqual(tv.proxy.rowCount(), 12)
        self.assertEqual(tv._quick.text(), "")
        self.assertEqual(tv.proxy.sortColumn(), -1)


class SortTests(unittest.TestCase):

    def test_runs_by_centroid_error(self):
        """The example asked for: centroids only, worst error first."""
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        tv._quick.setText("centroid")
        err = tv.model.column_names().index("Error")
        tv.view.sortByColumn(err, Qt.SortOrder.DescendingOrder)
        self.assertEqual(_shown(tv, "run_number"), ["7996", "7986", "8004"])

    def test_a_negative_value_sorts_numerically(self):
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        tv._quick.setText("centroid")
        val = tv.model.column_names().index("Value")
        tv.view.sortByColumn(val, Qt.SortOrder.AscendingOrder)
        self.assertEqual(_shown(tv, "Value"),
                         ["-270.1", "-265.5975", "-254.4764951"])

    def test_a_new_file_opens_in_file_order(self):
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        tv.view.sortByColumn(3, Qt.SortOrder.DescendingOrder)
        tv.load(_params())
        self.assertEqual(tv.proxy.sortColumn(), -1)
        self.assertEqual(_shown(tv, "run_number")[:4], ["7986"] * 4)


class ShadingTests(unittest.TestCase):

    def _bg(self, tv, proxy_row):
        px = tv.proxy
        return px.data(px.index(proxy_row, 0),
                       Qt.ItemDataRole.BackgroundRole)

    def test_each_run_has_its_own_tint(self):
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        a, b, c = self._bg(tv, 0), self._bg(tv, 4), self._bg(tv, 8)
        self.assertEqual(self._bg(tv, 3), a)          # same run
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)

    def test_the_tint_follows_the_run_through_a_sort(self):
        """Keyed to the run, not the row -- so after sorting by error
        the colour still says which run a row is."""
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        colour_of = {}
        for r in range(tv.proxy.rowCount()):
            run = tv.proxy.data(tv.proxy.index(r, 0))
            colour_of.setdefault(run, self._bg(tv, r))
        tv.view.sortByColumn(3, Qt.SortOrder.DescendingOrder)
        for r in range(tv.proxy.rowCount()):
            run = tv.proxy.data(tv.proxy.index(r, 0))
            self.assertEqual(self._bg(tv, r), colour_of[run])

    def test_shading_can_be_turned_off(self):
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        tv._shade_cb.setChecked(False)
        self.assertIsNone(self._bg(tv, 0))
        self.assertTrue(tv.view.alternatingRowColors())

    def test_a_file_without_a_run_column_is_not_shaded(self):
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
        self.assertIsNone(self._bg(tv, 0))
        self.assertFalse(tv._shade_cb.isEnabled())

    def test_neighbouring_tints_are_far_apart(self):
        """Adjacent runs must never read as the same colour."""
        from PySide6.QtGui import QColor
        ts = tints(QColor(30, 30, 30))
        for a, b in zip(ts, ts[1:]):
            d = sum(abs(x - y) for x, y in zip(a.getRgb()[:3],
                                               b.getRgb()[:3]))
            self.assertGreater(d, 15)

    def test_run_column_detection(self):
        self.assertEqual(run_column(["a", "run_number", "Source"]), 1)
        self.assertEqual(run_column(["Source", "x"]), 0)
        self.assertIsNone(run_column(["x", "y"]))


class CarryOverTests(unittest.TestCase):

    def test_the_quick_filter_survives_moving_to_the_next_file(self):
        """Stepping through iterations' parameters.csv keeps showing
        the centroids."""
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        tv._quick.setText("centroid")
        tv.load(_params())
        self.assertEqual(tv._quick.text(), "centroid")
        self.assertEqual(tv.proxy.rowCount(), 3)

    def test_but_not_onto_a_file_without_that_column(self):
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(_params())
        tv._quick.setText("centroid")
        tv.load(pd.DataFrame({"x": [1, 2]}))
        self.assertEqual(tv._quick.text(), "")
        self.assertEqual(tv.proxy.rowCount(), 2)


class ResultsTabWiringTests(unittest.TestCase):

    def test_the_results_tab_uses_it(self):
        from gui.results_tab import ResultsTab
        rt = ResultsTab()
        self.addCleanup(rt.deleteLater)
        self.assertIsInstance(rt._table_viewer, CsvTableView)

    def test_the_file_formatting_is_kept(self):
        """Floats to 10 significant figures, as the plain viewer did
        (code review 2026-06-02) -- what is copied matches the file."""
        tv = CsvTableView()
        self.addCleanup(tv.deleteLater)
        tv.load(pd.DataFrame({"Error": [3.690030056123456]}))
        self.assertEqual(tv.proxy.data(tv.proxy.index(0, 0)),
                         "3.690030056")


if __name__ == "__main__":
    unittest.main()
