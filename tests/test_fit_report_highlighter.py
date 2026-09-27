"""Fit-report syntax highlighting.

The point of the highlighter is not decoration. A fit report buries
three things that decide whether the fit is usable -- the relative
error on each parameter, the correlations between them, and reduced
chi-square -- in lines that are otherwise formatted identically. So
the tests assert on what colour a *value* gets, not merely that some
colour was applied.

Formats are read back off the block layout, which is what Qt actually
paints, rather than trusting that highlightBlock was called.
"""
import unittest

from PySide6.QtGui import QColor, QTextDocument

from gui.fit_report_highlighter import (
    CORR_DEGENERATE, CORR_HIGH, FitReportHighlighter, MAX_CHARS,
    PCT_FAIR, PCT_GOOD, PCT_POOR, build_formats, corr_key,
    looks_like_fit_report, pct_key, rchi_key,
)


def _colors(line: str, dark: bool = True):
    """``{char_index: '#rrggbb'}`` for one highlighted line."""
    doc = QTextDocument()
    hl = FitReportHighlighter(doc, dark=dark)
    doc.setPlainText(line)
    hl.rehighlight()
    out = {}
    block = doc.firstBlock()
    for r in block.layout().formats():
        name = r.format.foreground().color().name()
        for i in range(r.start, r.start + r.length):
            out[i] = name
    return out


def _color_of(line: str, needle: str, dark: bool = True):
    """Colour of the first character of *needle* within *line*."""
    idx = line.index(needle)
    return _colors(line, dark).get(idx)


class ThresholdTests(unittest.TestCase):
    """The value->severity mapping, independent of Qt."""

    def test_percent_bands(self):
        self.assertEqual(pct_key(2.35), "pct_good")
        self.assertEqual(pct_key(PCT_GOOD - 0.01), "pct_good")
        self.assertEqual(pct_key(19.66), "pct_fair")
        self.assertEqual(pct_key(PCT_FAIR), "pct_poor")
        self.assertEqual(pct_key(80.29), "pct_poor")
        self.assertEqual(pct_key(PCT_POOR), "pct_bad")
        self.assertEqual(pct_key(671509.73), "pct_bad")

    def test_the_runaway_errors_are_flagged(self):
        """354992519857.76% is not a large uncertainty, it is a
        parameter the data does not constrain. It must not land in
        the same bucket as 25%."""
        self.assertEqual(pct_key(354992519857.76), "pct_bad")
        self.assertNotEqual(pct_key(354992519857.76), pct_key(25.42))

    def test_correlation_bands_use_magnitude(self):
        """Sign is irrelevant to degeneracy; -1.0 is as bad as +1.0."""
        self.assertEqual(corr_key(1.0), "corr_degenerate")
        self.assertEqual(corr_key(-1.0), "corr_degenerate")
        self.assertEqual(corr_key(CORR_DEGENERATE), "corr_degenerate")
        self.assertEqual(corr_key(-0.9791), "corr_degenerate")
        self.assertEqual(corr_key(CORR_HIGH), "corr_high")
        self.assertEqual(corr_key(-0.8584), "corr_high")
        self.assertEqual(corr_key(0.1455), "corr_low")

    def test_reduced_chi_square_bands(self):
        """1 is the target. Far above means the model does not fit;
        far below means the errors are overestimated -- both worth
        seeing, so neither is 'good'."""
        self.assertEqual(rchi_key(0.78438531), "pct_good")
        self.assertEqual(rchi_key(1.52611209), "pct_good")
        self.assertEqual(rchi_key(3.02239433), "pct_poor")
        self.assertEqual(rchi_key(19.7318461), "pct_bad")
        self.assertEqual(rchi_key(0.01), "pct_bad")


class DetectionTests(unittest.TestCase):
    """The same viewer shows metadata dumps and binning summaries.
    Parameter-line rules would mangle those, so highlighting is
    switched by content."""

    def test_a_report_is_recognised(self):
        self.assertTrue(looks_like_fit_report(
            "[[Fit Statistics]]\n    chi-square = 1.0\n"))
        self.assertTrue(looks_like_fit_report("[[Variables]]\n"))

    def test_other_text_is_not(self):
        self.assertFalse(looks_like_fit_report(
            "run_number,centroid,error\n7934,-181.1,4.2\n"))
        self.assertFalse(looks_like_fit_report(""))
        self.assertFalse(looks_like_fit_report("Binning: 5 MHz\n"))

    def test_the_marker_may_be_below_the_first_line(self):
        """Reports open with a run banner, not the section header."""
        body = "\n=========== RUN 7934 ===========\n" + \
               "  Cooler V: 1.0\n[[Fit Statistics]]\n"
        self.assertTrue(looks_like_fit_report(body))


class ParameterLineTests(unittest.TestCase):

    LINE = ("    Run_7934___Model_1___centroid:     -181.140600 "
            "+/- 4.25328305 (2.35%) (init = -181.1406)")
    BAD = ("    Run_7934___Model_1___Al:            13.1820000 "
           "+/- 88518.4121 (671509.73%) (init = 13.182)")

    def test_a_good_error_is_green_and_a_hopeless_one_is_red(self):
        fmt = build_formats(True)
        self.assertEqual(_color_of(self.LINE, "(2.35%)"),
                         QColor(fmt["pct_good"].foreground().color()).name())
        self.assertEqual(_color_of(self.BAD, "(671509.73%)"),
                         QColor(fmt["pct_bad"].foreground().color()).name())
        self.assertNotEqual(_color_of(self.LINE, "(2.35%)"),
                            _color_of(self.BAD, "(671509.73%)"))

    def test_the_repeated_prefix_is_dimmed_and_the_leaf_is_not(self):
        """Run_7934___Model_1___ is identical on every line of the
        block and carries nothing once you know which run you are
        reading; 'centroid' is what the eye is hunting for."""
        c = _colors(self.LINE)
        prefix = c[self.LINE.index("Run_7934")]
        leaf = c[self.LINE.index("centroid")]
        self.assertNotEqual(prefix, leaf)
        fmt = build_formats(True)
        self.assertEqual(leaf, fmt["name"].foreground().color().name())

    def test_a_bkg_parameter_leaf_is_found_too(self):
        """Background params use a different separator depth
        (Model_1_bkg___p0); the leaf must still be p0."""
        line = ("    Run_7934___Model_1_bkg___p0:        3.53560000 "
                "+/- 0.89887614 (25.42%) (init = 3.5356)")
        fmt = build_formats(True)
        self.assertEqual(_color_of(line, "p0:"),
                         fmt["name"].foreground().color().name())

    def test_init_and_fixed_are_de_emphasised(self):
        fmt = build_formats(True)
        self.assertEqual(_color_of(self.LINE, "(init"),
                         fmt["init"].foreground().color().name())
        fixed = "    Run_7934___Model_1___Au:            0 (fixed)"
        self.assertEqual(_color_of(fixed, "(fixed)"),
                         fmt["fixed"].foreground().color().name())

    def test_a_fixed_parameter_gets_no_error_colouring(self):
        """It has no uncertainty to judge, so colouring it would be
        inventing information."""
        fixed = "    Run_7934___Model_1___Au:            0 (fixed)"
        c = _colors(fixed)
        fmt = build_formats(True)
        bad = fmt["pct_bad"].foreground().color().name()
        self.assertNotIn(bad, c.values())

    def test_scientific_notation_errors_still_parse(self):
        """The worst fits report 6.4292e+11, and that is exactly when
        the colour matters most."""
        line = ("    Run_7957___Model_1___centroid:     -198.033574 "
                "+/- 6.4292e+11 (324653921356.18%) (init = -181.1406)")
        fmt = build_formats(True)
        self.assertEqual(_color_of(line, "(324653921356.18%)"),
                         fmt["pct_bad"].foreground().color().name())


class CorrelationLineTests(unittest.TestCase):

    DEGEN = ("    C(Run_7934___Model_1___Al, Run_7934___Model_1___Bl)"
             "          = +1.0000")
    WEAK = ("    C(Run_7934___Model_1___centroid, "
            "Run_7934___Model_1___scale) = +0.1223")

    def test_a_degenerate_pair_is_red(self):
        """C(Al, Bl) = +1.0000 is the line that explains the
        671509% above it -- the two parameters trade off at no cost,
        so neither is determined."""
        fmt = build_formats(True)
        self.assertEqual(_color_of(self.DEGEN, "+1.0000"),
                         fmt["corr_degenerate"].foreground().color().name())

    def test_a_weak_correlation_is_not(self):
        fmt = build_formats(True)
        self.assertEqual(_color_of(self.WEAK, "+0.1223"),
                         fmt["corr_low"].foreground().color().name())

    def test_both_parameter_names_are_split(self):
        """Two full paths per line is most of the line; without
        dimming them the coefficient is lost."""
        c = _colors(self.DEGEN)
        fmt = build_formats(True)
        leaf = fmt["name"].foreground().color().name()
        self.assertEqual(c[self.DEGEN.index("Al,")], leaf)
        self.assertEqual(c[self.DEGEN.index("Bl)")], leaf)

    def test_a_negative_strong_correlation_is_flagged(self):
        line = ("    C(Run_7943___Model_1___FWHML, "
                "Run_7943___Model_1_bkg___p0) = -0.9791")
        fmt = build_formats(True)
        self.assertEqual(_color_of(line, "-0.9791"),
                         fmt["corr_degenerate"].foreground().color().name())


class StructureTests(unittest.TestCase):

    def test_run_banner(self):
        line = "==================== RUN 7934 ===================="
        fmt = build_formats(True)
        c = _colors(line)
        self.assertEqual(c[0], fmt["rule"].foreground().color().name())
        self.assertEqual(c[line.index("RUN")],
                         fmt["run"].foreground().color().name())

    def test_section_header_and_its_aside(self):
        line = "[[Correlations]] (unreported correlations are < 0.100)"
        fmt = build_formats(True)
        c = _colors(line)
        self.assertEqual(c[0], fmt["section"].foreground().color().name())
        self.assertEqual(c[line.index("(unreported")],
                         fmt["section_note"].foreground().color().name())

    def test_subsection_rule(self):
        fmt = build_formats(True)
        self.assertEqual(_color_of("--- Peak Positions [MHz] ---", "Peak"),
                         fmt["subsection"].foreground().color().name())

    def test_reduced_chi_square_is_coloured_by_value(self):
        good = "    reduced chi-square = 0.78438531"
        bad = "    reduced chi-square = 19.7318461"
        fmt = build_formats(True)
        self.assertEqual(_color_of(good, "0.78438531"),
                         fmt["pct_good"].foreground().color().name())
        self.assertEqual(_color_of(bad, "19.7318461"),
                         fmt["pct_bad"].foreground().color().name())

    def test_plain_chi_square_is_not_judged(self):
        """chi-square alone means nothing without the dof; only the
        reduced form has a target of 1."""
        line = "    chi-square         = 1460.15661"
        fmt = build_formats(True)
        self.assertEqual(_color_of(line, "1460.15661"),
                         fmt["value"].foreground().color().name())

    def test_the_metadata_strip_splits_keys_from_values(self):
        line = ("  Cooler V: 29906.94  |  Laser: 18429.999991940065 "
                "cm-1  |  Harmonic: 2")
        fmt = build_formats(True)
        self.assertEqual(_color_of(line, "29906.94"),
                         fmt["value"].foreground().color().name())
        self.assertEqual(_color_of(line, "Cooler"),
                         fmt["key"].foreground().color().name())


class SafetyTests(unittest.TestCase):

    def test_blank_lines_are_untouched(self):
        self.assertEqual(_colors(""), {})
        self.assertEqual(_colors("      "), {})

    def test_disabling_it_paints_nothing(self):
        """Switched off for non-reports; it must then be inert rather
        than merely subtle."""
        doc = QTextDocument()
        hl = FitReportHighlighter(doc)
        hl.enabled = False
        doc.setPlainText("[[Variables]]")
        hl.rehighlight()
        self.assertEqual(doc.firstBlock().layout().formats(), [])

    def test_a_malformed_percentage_does_not_raise(self):
        """Reports are machine-written, but a truncated or corrupt
        file must still display."""
        for line in ("    x___y: 1 +/- 2 (nan%) (init = 0)",
                     "    x___y: 1 +/- 2 (%) (init = 0)",
                     "    C(a, b) = notanumber",
                     "[[", "]]", "=== RUN ===", "--- ---"):
            with self.subTest(line=line):
                _colors(line)          # must not raise

    def test_size_guard_is_sane(self):
        """Big enough for a campaign report, small enough that the
        viewer stays responsive."""
        self.assertGreater(MAX_CHARS, 100_000)

    def test_light_palette_differs_from_dark(self):
        """Both shipped themes are dark, but the light table must be
        real rather than a copy, or a light theme would be unreadable
        on day one."""
        d, l = build_formats(True), build_formats(False)
        self.assertEqual(set(d), set(l))
        self.assertNotEqual(d["value"].foreground().color().name(),
                            l["value"].foreground().color().name())

    def test_every_format_key_is_used_by_some_rule(self):
        """A key nothing references is dead colour.

        Scans the whole module, not just the class: the severity
        keys (pct_*, corr_*) are produced by the module-level
        pct_key/corr_key/rchi_key helpers and never appear as
        literals inside highlightBlock.
        """
        import inspect
        from gui import fit_report_highlighter as frh
        src = inspect.getsource(frh)
        body = src[src.index("def pct_key"):]
        for key in build_formats(True):
            with self.subTest(key=key):
                self.assertIn(f'"{key}"', body)


if __name__ == "__main__":
    unittest.main()


class ViewerIntegrationTests(unittest.TestCase):
    """The Results text viewer shares one document between fit
    reports, metadata dumps and binning summaries, so the highlighter
    has to be switched per file rather than merely attached."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _tab(self):
        from gui.results_tab import ResultsTab
        tab = ResultsTab()
        self.addCleanup(tab.deleteLater)
        return tab

    def test_the_viewer_owns_exactly_one_highlighter(self):
        """Building one per selection would leave a live highlighter
        behind for every item ever clicked, each still reformatting
        the document on repaint."""
        tab = self._tab()
        hl = tab._report_highlighter
        self.assertIsNotNone(hl)
        self.assertIs(hl.document(), tab._text_viewer.document())
        tab._set_report_text("[[Variables]]\n    a___b: 1 (fixed)")
        tab._set_report_text("plain text")
        self.assertIs(tab._report_highlighter, hl)

    def test_it_switches_on_for_a_report_and_off_for_anything_else(self):
        tab = self._tab()
        tab._set_report_text("plain notes, no markers")
        self.assertFalse(tab._report_highlighter.enabled)
        tab._set_report_text("[[Fit Statistics]]\n    chi-square = 1.0")
        self.assertTrue(tab._report_highlighter.enabled)
        tab._set_report_text("Binning: 5 MHz")
        self.assertFalse(tab._report_highlighter.enabled)

    def test_the_text_is_intact_either_way(self):
        """Highlighting must never alter the document -- this is the
        file the user exports."""
        tab = self._tab()
        body = ("[[Variables]]\n    Run_1___M___centroid: -181.14 "
                "+/- 4.25 (2.35%) (init = -181.14)")
        tab._set_report_text(body)
        self.assertEqual(tab._text_viewer.toPlainText(), body)

    def test_an_oversized_report_is_shown_unhighlighted(self):
        """A campaign-wide report can run to megabytes; rehighlighting
        every block on every scroll is not worth a colour."""
        tab = self._tab()
        big = "[[Variables]]\n" + ("    a___b: 1 +/- 2 (3%)\n"
                                  * (MAX_CHARS // 24 + 100))
        self.assertGreater(len(big), MAX_CHARS)
        tab._set_report_text(big)
        self.assertFalse(tab._report_highlighter.enabled)
