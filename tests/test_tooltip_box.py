"""Tooltips sit as a compact box by the pointer -- headers included.

Arda (2026-09-27), on the ASDF viewer's column-header tooltip that ran
across the screen: "make it square, do not make the single row and span
wider area in the screen, confine them in a rectangular or square boxes
around the mouse pointer". Column headers had bypassed the app's tooltip
wrapper entirely (a header has no cells, so the item lookup found
nothing); and wrapped tooltips had no width target.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_tooltip_box.py -q
"""
import os
import re
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPoint  # noqa: E402
from PySide6.QtGui import QHelpEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QTableView, QToolTip  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui import shared_widgets as sw  # noqa: E402

LONG = ("Voltage-calibration points: the voltage read back at each "
        "setpoint, divided by 1,000 (x 1,000 = V). The fit of CalReadback "
        "against CalSet turns a setpoint into the applied scan voltage.")


class WidthTests(unittest.TestCase):

    def test_the_width_grows_with_the_text_within_limits(self):
        self.assertEqual(sw.tooltip_width_chars(10), sw.TIP_MIN_CHARS)
        self.assertEqual(sw.tooltip_width_chars(100_000), sw.TIP_MAX_CHARS)
        self.assertLess(sw.tooltip_width_chars(300),
                        sw.tooltip_width_chars(700))

    def test_about_twice_as_wide_as_tall(self):
        """In characters: n chars at width w make n / w lines."""
        for n in (300, 600, 900):
            w = sw.tooltip_width_chars(n)
            lines = n / w
            # A character is about twice as tall as wide on screen.
            ratio = w / (2 * lines)
            with self.subTest(chars=n):
                self.assertTrue(1.0 <= ratio <= 3.0, ratio)

    def test_a_long_one_line_tooltip_is_boxed(self):
        out = sw.reflow_tooltip(LONG)
        m = re.search(r'<table width="(\d+)"', out)
        self.assertIsNotNone(m, out)
        self.assertIn("CalReadback", out)

    def test_short_ones_are_left_alone(self):
        self.assertIsNone(sw.reflow_tooltip("Drag to reorder"))


class HeaderTooltipTests(unittest.TestCase):
    """The column-header path through the app-wide wrapper."""

    def setUp(self):
        from gui.asdf_viewer import ArrayTableModel
        self.model = ArrayTableModel()
        self.model.set_table(np.zeros((3, 2)), ["time", "CalReadback"])
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.resize(400, 200)
        self.addCleanup(self.view.deleteLater)
        self.shown = []
        orig = QToolTip.showText
        QToolTip.showText = staticmethod(
            lambda pos, text, w=None, *a: self.shown.append(text))
        self.addCleanup(setattr, QToolTip, "showText", orig)

    def _hover_header(self, section):
        header = self.view.horizontalHeader()
        x = header.sectionViewportPosition(section) + 4
        ev = QHelpEvent(QEvent.Type.ToolTip, QPoint(x, 4), QPoint(x, 4))
        return sw._TooltipWrapFilter().eventFilter(header.viewport(), ev)

    def test_a_header_tooltip_is_wrapped(self):
        self.assertTrue(self._hover_header(1))
        self.assertEqual(len(self.shown), 1)
        self.assertIn("<table width=", self.shown[0])
        self.assertIn("CalReadback", self.shown[0])

    def test_each_section_gets_its_own_text(self):
        self._hover_header(0)
        self.assertIn("Time of flight", self.shown[0])

    def test_a_section_without_a_tooltip_is_left_to_qt(self):
        self.model.set_table(np.zeros((3, 1)), ["no_doc_for_this"])
        self.assertFalse(self._hover_header(0))
        self.assertEqual(self.shown, [])


if __name__ == "__main__":
    unittest.main()
