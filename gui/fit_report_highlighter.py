"""Syntax highlighting for lmfit fit reports in the Results viewer.

A fit report is mostly boilerplate. Every parameter line repeats the
run and model it belongs to -- ``Run_7934___Model_1___centroid`` is
28 characters of which 8 carry information -- and the numbers that
decide whether a fit is any good are scattered among numbers that do
not matter. Reading one means scanning for three things:

* **the relative error**. ``(2.35%)`` is a measured parameter;
  ``(671509.73%)`` is a parameter the data does not constrain at all,
  and it looks exactly the same in monospace grey.
* **the correlations**. ``C(Al, Bl) = +1.0000`` means those two are
  perfectly degenerate and the fit has no unique solution -- that
  single line explains the 671509% above it.
* **reduced chi-square**. 0.78 and 19.73 are two very different
  outcomes sitting in identically formatted lines.

So this does not colour by token type alone. Percentages, correlation
coefficients and reduced chi-square are coloured **by value**, on
thresholds that mean something physically, so a bad fit is visible
from across the room rather than after reading every line.

Both shipped themes are dark, but the palette is chosen from the
widget's own background lightness so a light theme would still be
legible.
"""
from __future__ import annotations

import re

from PySide6.QtGui import (
    QColor, QFont, QSyntaxHighlighter, QTextCharFormat,
)

# ── Thresholds ───────────────────────────────────────────────
# Relative parameter error, per cent. A Gaussian 1-sigma of 30% is
# already a parameter you would not quote; past 100% the fit has not
# determined it at all, and the giant values (10^5 %+) mean the
# covariance matrix is effectively singular.
PCT_GOOD = 10.0
PCT_FAIR = 30.0
PCT_POOR = 100.0

# |r| between two parameters. Past 0.95 they are degenerate in
# practice: the fit can trade one against the other at no cost, which
# is what produces the runaway errors above.
CORR_HIGH = 0.80
CORR_DEGENERATE = 0.95

# Reduced chi-square. Around 1 is the target; well under means the
# errors are overestimated (or the model is overfitting), well over
# means the model does not describe the data.
RCHI_LO, RCHI_HI = 0.5, 2.0
RCHI_BAD = 5.0

#: Documents larger than this are shown unhighlighted. A report
#: covering a whole campaign can run to megabytes, and rehighlighting
#: every block on every scroll is not worth a colour.
MAX_CHARS = 2_000_000

#: Markers that identify a document as an lmfit fit report. Checked
#: against the head of the text so the highlighter is not attached to
#: metadata dumps or binning summaries.
_REPORT_MARKERS = ("[[Fit Statistics]]", "[[Variables]]",
                   "[[Correlations]]")


def looks_like_fit_report(text: str) -> bool:
    """True if *text* is an lmfit report worth highlighting."""
    head = text[:8000]
    return any(m in head for m in _REPORT_MARKERS)


# ── Patterns ─────────────────────────────────────────────────
# Anchored and ordered cheapest-first; highlightBlock runs per line
# on every repaint of the viewport.
_RE_RUN = re.compile(r"^(=+)(\s*RUN\s+)(\S+)(\s*)(=+)\s*$")
_RE_SECTION = re.compile(r"^\[\[([^\]]+)\]\](.*)$")
_RE_SUBSECTION = re.compile(r"^(-{2,})( .+? )(-{2,})\s*$")
_RE_STAT = re.compile(r"^(\s+)(#?\s*[A-Za-z][^=]*?)(\s*=\s*)(.+)$")
_RE_PARAM = re.compile(r"^(\s+)([A-Za-z_]\w*)(:)(\s+)(.*)$")
_RE_CORR = re.compile(
    r"^(\s+)(C\()([^,]+)(,\s*)([^)]+)(\)\s*)(=\s*)([+-]?[\d.]+)\s*$")
_RE_META = re.compile(r"([A-Za-z][A-Za-z ]*?):\s*(\S+)")

# Inside a parameter's value tail.
_RE_PCT = re.compile(r"\(([\d.]+(?:e[+-]?\d+)?)%\)")
_RE_PM = re.compile(r"\+/-")
_RE_INIT = re.compile(r"\(init\s*=\s*[^)]*\)")
_RE_FIXED = re.compile(r"\(fixed\)")
_RE_NUMBER = re.compile(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?")


def _c(spec, *, bold=False, italic=False):
    f = QTextCharFormat()
    f.setForeground(QColor(spec))
    if bold:
        f.setFontWeight(QFont.Weight.Bold)
    if italic:
        f.setFontItalic(True)
    return f


def build_formats(dark: bool = True) -> dict:
    """The colour table. Kept a plain dict so tests can assert on it
    and a theme can swap it wholesale."""
    if dark:
        return {
            "rule": _c("#546E7A"),            # the ==== and ---- runs
            "run": _c("#4DD0E1", bold=True),  # RUN 7934
            "section": _c("#FFB74D", bold=True),   # [[Variables]]
            "section_note": _c("#8D6E63"),    # its trailing remark
            "subsection": _c("#CE93D8", bold=True),  # --- FWHM ---
            "key": _c("#90CAF9"),             # stat / metadata names
            "prefix": _c("#546E7A"),          # Run_7934___Model_1___
            "name": _c("#A5D6A7", bold=True),  # ...centroid
            "punct": _c("#78909C"),
            "value": _c("#ECEFF1"),
            "pm": _c("#78909C"),
            "err": _c("#B0BEC5"),
            "init": _c("#607D8B", italic=True),
            "fixed": _c("#78909C", italic=True),
            "pct_good": _c("#66BB6A"),
            "pct_fair": _c("#B0BEC5"),
            "pct_poor": _c("#FFA726", bold=True),
            "pct_bad": _c("#EF5350", bold=True),
            "corr_low": _c("#90A4AE"),
            "corr_high": _c("#FFA726", bold=True),
            "corr_degenerate": _c("#EF5350", bold=True),
        }
    return {
        "rule": _c("#B0BEC5"),
        "run": _c("#00697A", bold=True),
        "section": _c("#B25E00", bold=True),
        "section_note": _c("#8D6E63"),
        "subsection": _c("#7B1FA2", bold=True),
        "key": _c("#1565C0"),
        "prefix": _c("#90A4AE"),
        "name": _c("#2E7D32", bold=True),
        "punct": _c("#607D8B"),
        "value": _c("#212121"),
        "pm": _c("#607D8B"),
        "err": _c("#455A64"),
        "init": _c("#78909C", italic=True),
        "fixed": _c("#607D8B", italic=True),
        "pct_good": _c("#2E7D32"),
        "pct_fair": _c("#455A64"),
        "pct_poor": _c("#E65100", bold=True),
        "pct_bad": _c("#C62828", bold=True),
        "corr_low": _c("#607D8B"),
        "corr_high": _c("#E65100", bold=True),
        "corr_degenerate": _c("#C62828", bold=True),
    }


def pct_key(pct: float) -> str:
    """Format key for a relative error, per cent."""
    if pct < PCT_GOOD:
        return "pct_good"
    if pct < PCT_FAIR:
        return "pct_fair"
    if pct < PCT_POOR:
        return "pct_poor"
    return "pct_bad"


def corr_key(r: float) -> str:
    """Format key for a correlation coefficient."""
    a = abs(r)
    if a >= CORR_DEGENERATE:
        return "corr_degenerate"
    if a >= CORR_HIGH:
        return "corr_high"
    return "corr_low"


def rchi_key(v: float) -> str:
    """Format key for reduced chi-square. Reuses the percentage
    colours: the reader only has to learn one green/amber/red."""
    if RCHI_LO <= v <= RCHI_HI:
        return "pct_good"
    if v > RCHI_BAD or v < RCHI_LO / 5.0:
        return "pct_bad"
    return "pct_poor"


class FitReportHighlighter(QSyntaxHighlighter):
    """Colour an lmfit report by structure *and* by value.

    Attach to a ``QPlainTextEdit``'s document; it re-runs per block as
    Qt repaints, so every rule is a single anchored regex on one line.
    """

    def __init__(self, document, dark: bool = True):
        super().__init__(document)
        self.fmt = build_formats(dark)
        self.enabled = True

    def set_dark(self, dark: bool):
        self.fmt = build_formats(dark)
        self.rehighlight()

    # ── Helpers ──
    def _apply(self, start, length, key):
        if length > 0:
            self.setFormat(start, length, self.fmt[key])

    def _value_tail(self, offset: int, tail: str):
        """The part after ``name:`` -- value, error, percentage,
        init and fixed markers."""
        if _RE_FIXED.search(tail):
            # "0 (fixed)": the number is not a result, so it is not
            # dressed up as one.
            m = _RE_FIXED.search(tail)
            self._apply(offset, m.start(), "value")
            self._apply(offset + m.start(), len(m.group(0)), "fixed")
            return

        # init block first: it contains an "=" and digits that the
        # later number pass would otherwise recolour.
        init = _RE_INIT.search(tail)
        init_span = init.span() if init else (len(tail), len(tail))

        pm = _RE_PM.search(tail)
        pct = _RE_PCT.search(tail)

        body_end = min(init_span[0],
                       pct.start() if pct else init_span[0])
        if pm:
            self._apply(offset, pm.start(), "value")
            self._apply(offset + pm.start(), len(pm.group(0)), "pm")
            self._apply(offset + pm.end(),
                        max(0, body_end - pm.end()), "err")
        else:
            self._apply(offset, body_end, "value")

        if pct:
            try:
                val = float(pct.group(1))
            except ValueError:
                val = 0.0
            self._apply(offset + pct.start(), len(pct.group(0)),
                        pct_key(val))
        if init:
            self._apply(offset + init_span[0],
                        init_span[1] - init_span[0], "init")

    # ── Qt entry point ──
    def highlightBlock(self, text: str):
        if not self.enabled or not text.strip():
            return

        m = _RE_RUN.match(text)
        if m:
            self._apply(m.start(1), len(m.group(1)), "rule")
            self._apply(m.start(2), len(m.group(2)), "run")
            self._apply(m.start(3), len(m.group(3)), "run")
            self._apply(m.start(5), len(m.group(5)), "rule")
            return

        m = _RE_SECTION.match(text)
        if m:
            self._apply(0, m.end(1) + 2, "section")
            self._apply(m.start(2), len(m.group(2)), "section_note")
            return

        m = _RE_SUBSECTION.match(text)
        if m:
            self._apply(m.start(1), len(m.group(1)), "rule")
            self._apply(m.start(2), len(m.group(2)), "subsection")
            self._apply(m.start(3), len(m.group(3)), "rule")
            return

        m = _RE_CORR.match(text)
        if m:
            self._apply(m.start(2), len(m.group(2)), "punct")
            for g in (3, 5):
                self._name_span(m.start(g), m.group(g))
            self._apply(m.start(4), len(m.group(4)), "punct")
            self._apply(m.start(6), len(m.group(6)), "punct")
            self._apply(m.start(7), len(m.group(7)), "punct")
            try:
                r = float(m.group(8))
            except ValueError:
                r = 0.0
            self._apply(m.start(8), len(m.group(8)), corr_key(r))
            return

        m = _RE_PARAM.match(text)
        if m:
            self._name_span(m.start(2), m.group(2))
            self._apply(m.start(3), 1, "punct")
            self._value_tail(m.start(5), m.group(5))
            return

        m = _RE_STAT.match(text)
        if m:
            self._apply(m.start(2), len(m.group(2)), "key")
            self._apply(m.start(3), len(m.group(3)), "punct")
            key = m.group(2).strip().lstrip("# ").strip()
            val = m.group(4)
            if key == "reduced chi-square":
                try:
                    self._apply(m.start(4), len(val),
                                rchi_key(float(val)))
                    return
                except ValueError:
                    pass
            self._apply(m.start(4), len(val), "value")
            return

        # The run's metadata strip: "Cooler V: ... | Laser: ... | ..."
        if "|" in text and ":" in text:
            for mm in _RE_META.finditer(text):
                self._apply(mm.start(1), len(mm.group(1)), "key")
                self._apply(mm.start(2), len(mm.group(2)), "value")
            return

        # Anything else: at least make the numbers legible (the
        # "Model_1: [-72.39, -220.69, ...]" peak-position lines).
        for mm in _RE_NUMBER.finditer(text):
            self._apply(mm.start(), len(mm.group(0)), "value")

    def _name_span(self, start: int, name: str):
        """``Run_7934___Model_1___centroid`` -- dim the path, light up
        the leaf. The prefix is the same on every line of the block
        and carries no information once you know which run you are
        reading; the leaf is the parameter you are looking for."""
        idx = name.rfind("___")
        if idx >= 0:
            self._apply(start, idx + 3, "prefix")
            self._apply(start + idx + 3, len(name) - idx - 3, "name")
        else:
            self._apply(start, len(name), "name")
