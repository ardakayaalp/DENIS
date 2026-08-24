"""LineColorDialog: dual light/dark picker with hex/RGB + custom slots.

Date:    2026-07-26
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the enriched colour picker used by Pre-Analysis files and models:
a hex or R,G,B text box sets each mode's colour, and custom colours the
user saves round-trip through the (patched) settings store. The real
settings file is never touched by these tests.

Run from the project root:

    .venv/Scripts/python.exe -m pytest tests/test_line_color_dialog.py -q

Depends on: gui.preanalysis_tab; PySide6.
"""

import unittest

from PySide6.QtWidgets import QApplication, QLineEdit


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class LineColorDialogTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        import gui.preanalysis_tab as P
        self.P = P
        # Isolate from the real settings file.
        self._saved = {"colors": ["#ABCDEF"]}
        self._orig_load = P._load_custom_colors
        self._orig_save = P._save_custom_colors
        P._load_custom_colors = lambda: list(self._saved["colors"])
        P._save_custom_colors = lambda c: self._saved.update(
            {"colors": list(c)})

    def tearDown(self):
        self.P._load_custom_colors = self._orig_load
        self.P._save_custom_colors = self._orig_save

    def test_initial_colours(self):
        dlg = self.P.LineColorDialog("#000080", "#00E5FF")
        self.assertEqual(dlg.colors(), ("#000080", "#00E5FF"))
        self.assertEqual(dlg._customs, ["#ABCDEF"])

    def test_hex_entry_sets_light(self):
        dlg = self.P.LineColorDialog("#000080", "#00E5FF")
        edit = dlg.findChildren(QLineEdit)[0]
        edit.setText("#123456")
        edit.editingFinished.emit()
        self.assertEqual(dlg.colors()[0].lower(), "#123456")

    def test_rgb_entry_sets_dark(self):
        dlg = self.P.LineColorDialog("#000080", "#00E5FF")
        dark_edit = dlg.findChildren(QLineEdit)[1]
        dark_edit.setText("255,0,128")
        dark_edit.editingFinished.emit()
        self.assertEqual(dlg.colors()[1].lower(), "#ff0080")

    def test_invalid_text_ignored(self):
        dlg = self.P.LineColorDialog("#000080", "#00E5FF")
        edit = dlg.findChildren(QLineEdit)[0]
        edit.setText("not-a-colour")
        edit.editingFinished.emit()
        self.assertEqual(dlg.colors()[0], "#000080")   # unchanged

    def test_custom_save_round_trips(self):
        dlg = self.P.LineColorDialog("#111111", "#222222")
        # Simulate the "＋" save of the dark colour.
        dlg._customs.insert(0, dlg._cur(True))
        self.P._save_custom_colors(dlg._customs)
        self.assertIn("#222222", self._saved["colors"])
        # A fresh dialog picks the saved custom up.
        dlg2 = self.P.LineColorDialog("#111111", "#222222")
        self.assertIn("#222222", dlg2._customs)


if __name__ == "__main__":
    unittest.main()
