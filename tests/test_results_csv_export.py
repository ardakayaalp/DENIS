"""Results tab: export a plot's underlying data as CSV.

Date:    2026-07-26
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the CSV export: the column builder pulls data points + errors,
the model and residuals (and the dense fit curve) from a fit .npz;
ragged columns are padded; the options dialog controls header,
delimiter, precision and which columns are written.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_results_csv_export.py -q

Depends on: gui.results_tab; PySide6; numpy.
"""

import os
import tempfile
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _fit_npz(path):
    x = np.linspace(-100, 100, 11)
    y = 50 + 10 * np.exp(-(x / 30) ** 2)
    yerr = np.sqrt(y)
    resid = np.linspace(-1, 1, 11)
    xs = np.linspace(-100, 100, 200)
    ys = 50 + 10 * np.exp(-(xs / 30) ** 2)
    np.savez(path, plot_type="fit", x=x, y=y, yerr=yerr,
             residuals=resid, x_smooth=xs, y_fit_smooth=ys, run_num="2")
    return x, y, yerr, resid


class CsvColumnBuilderTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.results_tab import ResultsTab
        self.rt = ResultsTab()
        self.d = tempfile.mkdtemp()
        self.npz = os.path.join(self.d, "fit_run_2.npz")
        self.x, self.y, self.yerr, self.resid = _fit_npz(self.npz)

    def test_fit_columns_include_data_errors_model_residuals(self):
        data = np.load(self.npz, allow_pickle=True)
        cols = self.rt._plot_csv_columns("fit", data)
        for name in ("x", "y", "y_err", "model", "residual",
                     "residual_err", "x_fit_curve", "y_fit_curve"):
            self.assertIn(name, cols)
        # model = data − residual; residual_err mirrors y_err.
        self.assertTrue(np.allclose(cols["model"], self.y - self.resid))
        self.assertTrue(np.allclose(cols["residual_err"], self.yerr))

    def test_generic_fallback_for_unknown_type(self):
        p = os.path.join(self.d, "misc.npz")
        np.savez(p, plot_type="mystery", a=np.arange(5.0),
                 b=np.ones(5), grid=np.zeros((3, 3)))
        data = np.load(p, allow_pickle=True)
        cols = self.rt._plot_csv_columns("mystery", data)
        self.assertIn("a", cols)
        self.assertIn("b", cols)
        self.assertNotIn("grid", cols)   # 2-D dropped


class CsvWriterTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from gui.results_tab import ResultsTab
        import gui.results_tab as R
        self.R = R
        self.rt = ResultsTab()
        self.d = tempfile.mkdtemp()
        self.npz = os.path.join(self.d, "fit_run_2.npz")
        _fit_npz(self.npz)
        self.out = os.path.join(self.d, "out.csv")
        # Auto-accept the export dialog (all columns, defaults) and the
        # save-path dialog; silence the confirmation.
        self._exec = R.PlotCsvExportDialog.exec
        R.PlotCsvExportDialog.exec = lambda _self: 1
        self._save = R.QFileDialog.getSaveFileName
        R.QFileDialog.getSaveFileName = staticmethod(
            lambda *a, **k: (self.out, ""))
        self._info = R.QMessageBox.information
        R.QMessageBox.information = staticmethod(lambda *a, **k: None)

    def tearDown(self):
        self.R.PlotCsvExportDialog.exec = self._exec
        self.R.QFileDialog.getSaveFileName = self._save
        self.R.QMessageBox.information = self._info

    def test_export_writes_padded_csv(self):
        self.rt._export_plot_csv(self.npz)
        self.assertTrue(os.path.isfile(self.out))
        with open(self.out) as f:
            lines = f.read().strip().splitlines()
        # Header + 200 rows (padded to the dense fit curve length).
        self.assertEqual(lines[0].split(",")[:3], ["x", "y", "y_err"])
        self.assertEqual(len(lines) - 1, 200)
        # Row 20 is past the 11 data points: data columns blank, the
        # dense fit-curve columns still filled.
        row = lines[21].split(",")
        self.assertEqual(row[0], "")          # x (len 11) padded
        self.assertNotEqual(row[6], "")       # x_fit_curve (len 200)

    def test_precision_and_delimiter_options(self):
        import gui.results_tab as R

        def opts_patch(_self):
            return {"columns": ["x", "y"], "header": False,
                    "delimiter": ";", "precision": 2}
        orig = R.PlotCsvExportDialog.options
        R.PlotCsvExportDialog.options = opts_patch
        try:
            self.rt._export_plot_csv(self.npz)
        finally:
            R.PlotCsvExportDialog.options = orig
        with open(self.out) as f:
            first = f.readline().strip()
        # No header, semicolon delimiter, 2 decimals.
        parts = first.split(";")
        self.assertEqual(len(parts), 2)
        self.assertRegex(parts[0], r"^-?\d+\.\d{2}$")


if __name__ == "__main__":
    unittest.main()
