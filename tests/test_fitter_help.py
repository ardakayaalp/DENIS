"""The "?" guide next to the Fitter block's Method & Statistics.

Arda asked (2026-09-25) for a square "?" button that opens a pop-up
with practical detail on the methods in that block, "so user can make
more informed decision which one to use when". The guide is data
(fitter_help.METHODS / STATISTICS), so these tests can insist that
every choice the combos offer is explained -- a method added to the
combo without a line in the guide fails here.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_fitter_help.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui.analysis import fitter_help  # noqa: E402
from gui.analysis.fitter_help import (  # noqa: E402
    METHODS, STATISTICS, build_html, show_fitter_help)
from gui.analysis.fitting import LLH_FALLBACK_METHOD  # noqa: E402


def _combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


class ButtonTests(unittest.TestCase):

    def setUp(self):
        from gui.analysis.blocks import FitterBlock
        self.block = FitterBlock()
        self.addCleanup(self.block.deleteLater)
        self.btn = self.block._method_help_btn

    def test_it_is_a_square_question_mark(self):
        self.assertEqual(self.btn.text(), "?")
        hint = self.btn.sizeHint()
        self.assertEqual(hint.width(), hint.height())

    def test_it_is_exactly_as_tall_as_the_method_combo(self):
        self.assertEqual(self.btn.sizeHint().height(),
                         self.block._method_combo.sizeHint().height())

    def test_it_sits_in_the_method_and_statistics_group(self):
        group = self.btn.parentWidget()
        while group is not None and not group.inherits("QGroupBox"):
            group = group.parentWidget()
        self.assertIsNotNone(group)
        self.assertIn("Statistics", group.title())

    def test_it_says_what_it_does(self):
        self.assertIn("guide", self.btn.toolTip())

    def test_clicking_opens_the_guide_without_blocking(self):
        self.btn.click()
        dlg = fitter_help._dialog
        self.addCleanup(dlg.hide)
        self.assertIsInstance(dlg, QDialog)
        self.assertTrue(dlg.isVisible())
        self.assertFalse(dlg.isModal())

    def test_a_second_click_reuses_the_open_guide(self):
        self.btn.click()
        first = fitter_help._dialog
        self.btn.click()
        self.addCleanup(first.hide)
        self.assertIs(fitter_help._dialog, first)


class CoverageTests(unittest.TestCase):
    """Every choice in the combos is explained."""

    @classmethod
    def setUpClass(cls):
        from gui.analysis.blocks import FitterBlock
        cls.block = FitterBlock()
        cls.methods = _combo_items(cls.block._method_combo)
        cls.stats = _combo_items(cls.block._stats_combo)
        cls.html = build_html()

    @classmethod
    def tearDownClass(cls):
        cls.block.deleteLater()

    def test_every_method_has_a_row(self):
        self.assertEqual(sorted(m[0] for m in METHODS), sorted(self.methods))

    def test_every_statistic_has_a_row(self):
        self.assertEqual(sorted(s[0] for s in STATISTICS),
                         sorted(self.stats))

    def test_every_choice_appears_in_the_guide(self):
        for name in self.methods + self.stats:
            with self.subTest(name=name):
                self.assertIn(name, self.html)

    def test_the_likelihood_substitute_is_named_correctly(self):
        leastsq = next(m for m in METHODS if m[0] == "leastsq")
        self.assertIn(LLH_FALLBACK_METHOD, " ".join(leastsq))

    def test_it_covers_the_error_bars_and_burn_in(self):
        for topic in ("Error bars", "Burn-in", "Scale covariance",
                      "walk plot"):
            with self.subTest(topic=topic):
                self.assertIn(topic, self.html)

    def test_the_header_colour_follows_the_theme(self):
        self.assertIn('bgcolor="#123456"', build_html(header_bg="#123456"))


class ManualLinkTests(unittest.TestCase):

    def test_it_opens_the_fitter_page(self):
        seen = []

        class _Manual:
            def navigate(self, page_id, record=True):
                seen.append(page_id)

        class _Top:
            _manual_window = _Manual()

            def _open_manual(self):
                seen.append("opened")

        dlg = fitter_help.FitterHelpDialog()
        self.addCleanup(dlg.deleteLater)
        dlg.parent = lambda: type("P", (), {"window": lambda s: _Top()})()
        dlg.open_manual()
        self.assertEqual(seen, ["opened", fitter_help.MANUAL_PAGE])

    def test_the_page_exists_in_the_manual(self):
        from gui.manual.content.generated import PAGES_DATA
        self.assertIn(fitter_help.MANUAL_PAGE, PAGES_DATA)


class ShowTests(unittest.TestCase):

    def test_show_without_an_anchor_still_works(self):
        dlg = show_fitter_help(None)
        self.addCleanup(dlg.hide)
        self.assertTrue(dlg.isVisible())


if __name__ == "__main__":
    unittest.main()
