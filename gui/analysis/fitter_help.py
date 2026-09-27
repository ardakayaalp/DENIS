"""The "?" guide next to the Fitter block's Method & Statistics.

Practical help for choosing a minimiser and a cost function: what each
one does, whether it gives error bars, and when to use it. The tables
are data (METHODS, STATISTICS) so a test can check that every entry of
the Fitter block's combos is explained here -- a method added to the
combo without a line in this guide fails that test.

The advice comes from checks run on real DENIS fits (2026-09-25):
Poisson + leastsq silently running SLSQP, the bias of data-based
errors at low counts, reported errors against the true scatter, and
the effect of burn-in on emcee's error bars.
"""
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QPushButton, QTextBrowser, QToolButton,
    QVBoxLayout,
)

from gui.analysis.fitting import LLH_FALLBACK_METHOD

MANUAL_PAGE = "an-fitter"

#: (combo text, what it minimises, uses the Source yerr?, use it for)
STATISTICS = (
    ("Chi-square",
     "&Sigma; ((y &minus; f) / &sigma;)&sup2;, with &sigma; from the "
     "Source block's <b>yerr mode</b>.",
     "Yes &mdash; &sigma; is each bin's weight.",
     "Most spectra: fast with <code>leastsq</code>, and its error bars "
     "match the real scatter when yerr is sqrt(y+1) or Model-based and "
     "Scale covariance is off. At low counts a fitted background sits "
     "up to ~1 count per bin low with sqrt(y+1) (Model-based: ~0.5 "
     "high); centroids are unaffected."),
    ("Gaussian LLH",
     "&minus;&frac12; &Sigma; ((y &minus; f) / &sigma;)&sup2; &mdash; "
     "satlas2 leaves out the ln &sigma; term.",
     "Yes.",
     "Nothing over Chi-square: the same best fit, and it needs a slower "
     "scalar minimiser. It is what <code>emcee</code> samples when "
     "Statistics is Chi-square."),
    ("Poisson LLH",
     "&Sigma; (f &minus; y ln f): the exact likelihood of counted events.",
     "No &mdash; a Poisson variance equals its mean, so the model "
     "already sets each bin's weight.",
     "Low counts (many bins under ~10): unbiased values at any count "
     "level. Needs raw counts (DENIS bins are) and a model above zero "
     "everywhere, so keep the background positive."),
)

#: (combo text, what it is, error bars, speed, notes)
METHODS = (
    ("leastsq",
     "Levenberg&ndash;Marquardt.",
     "From the covariance matrix at the minimum.",
     "Fast",
     "The default for Chi-square. Cannot minimise a likelihood: with "
     f"Poisson or Gaussian LLH DENIS runs <code>{LLH_FALLBACK_METHOD}"
     "</code> instead and says so in the report."),
    ("least_squares",
     "Trust-region reflective (scipy); handles bounds natively.",
     "From the covariance matrix.",
     "Fast",
     "An alternative to leastsq, e.g. when parameters press against "
     "their Min/Max. Chi-square only, as leastsq."),
    ("slsqp",
     "Sequential least-squares programming.",
     "Numerical estimate (see below).",
     "Medium",
     "Not recommended for spectra: fitting a Yb hyperfine spectrum "
     "with a likelihood, it reported success at a centroid of "
     "10<sup>12</sup> MHz."),
    ("emcee",
     "MCMC sampler: maps the whole posterior with an ensemble of "
     "walkers.",
     "Half the 15.87&ndash;84.13 % spread of the posterior, after "
     "burn-in.",
     "Slow",
     "The most trustworthy error bars, also for asymmetric or strongly "
     "correlated parameters. Costs walkers &times; steps model "
     "evaluations (50 &times; 1000 = 50&nbsp;000). Needs a burn-in &mdash; "
     "see below."),
    ("nelder",
     "Nelder&ndash;Mead simplex; needs no derivatives.",
     "Numerical estimate (see below).",
     "Medium",
     "The minimiser for a Poisson LLH point fit. Also worth a try when "
     "leastsq stalls."),
    ("powell",
     "Powell's direction-set method; needs no derivatives.",
     "Numerical estimate (see below).",
     "Medium",
     "Same role as nelder."),
    ("cobyla",
     "Constrained optimisation by linear approximation.",
     "Numerical estimate (see below).",
     "Slow",
     "Rarely needed here."),
)

RECIPES = (
    ("Everyday spectra (most bins above ~10&ndash;20 counts)",
     "Chi-square + <code>leastsq</code>, yerr sqrt(y+1) or Model-based, "
     "Scale covariance off."),
    ("Low counts (many bins below ~10)",
     "Poisson LLH + <code>nelder</code>. For quick error bars tick "
     "<b>Correct likelihood error bars</b>; for the final ones run "
     "Poisson LLH + <code>emcee</code> with a burn-in."),
    ("Final numbers, or asymmetric / correlated parameters",
     "<code>emcee</code>, started from a converged point fit, with the "
     "burn-in set from the walk plot."),
    ("A fit that will not converge",
     "<b>Find Parameters</b> (Auto-Fitter) first; nelder or powell can "
     "help when leastsq stalls."),
    ("Avoid",
     "<code>slsqp</code> for spectra; yerr <b>None</b> for real fits "
     "(unweighted, and its errors are meaningless); Scale covariance "
     "with absolute Poisson errors."),
)


def _table(headers, rows, header_bg, widths=None):
    widths = widths or [None] * len(headers)
    head = "".join(
        f'<th align="left" bgcolor="{header_bg}"'
        + (f' width="{w}%"' if w else "") + f">{h}</th>"
        for h, w in zip(headers, widths))
    body = "".join(
        "<tr>" + "".join(f'<td valign="top">{c}</td>' for c in row) + "</tr>"
        for row in rows)
    return ('<table border="1" cellspacing="0" cellpadding="4" '
            f'width="100%"><tr>{head}</tr>{body}</table>')


def build_html(header_bg="#dddddd"):
    """The whole guide as rich text for a QTextBrowser."""
    recipes = "".join(f"<li><b>{w}:</b> {what}</li>" for w, what in RECIPES)
    stats = _table(("Statistics", "Minimises", "Uses yerr?", "Use it for"),
                   STATISTICS, header_bg, widths=(14, 28, 20, 38))
    methods = _table(("Method", "What it is", "Error bars", "Speed",
                      "Notes"), METHODS, header_bg,
                     widths=(17, 23, 20, 8, 32))
    return f"""
<h3>Which should I pick?</h3>
<ul>{recipes}</ul>

<h3>Statistics &mdash; what is minimised</h3>
{stats}

<h3>Method &mdash; how it is minimised</h3>
{methods}

<h3>Error bars</h3>
<ul>
<li><b>leastsq / least_squares</b>: from the covariance matrix. With
<b>Scale covariance</b> off (the default) they are absolute, and correct
when yerr is the true &sigma;. Scale covariance multiplies them by
&radic;(reduced &chi;&sup2;) &mdash; only for errors that are not
absolute. It is forced off for a likelihood.</li>
<li><b>emcee</b>: from the posterior after burn-in. With enough burn-in
they agree with leastsq on a well-behaved fit.</li>
<li><b>nelder, powell, cobyla, slsqp</b>: lmfit estimates the errors
numerically, as 2 &times; the inverse curvature at the minimum. Right for
Chi-square. Under a likelihood they come out &radic;2 (41 %) too large:
the factor 2 assumes a &chi;&sup2;, while satlas2 minimises &minus;ln L,
half of one. The Fitter block warns about it, and <b>Correct likelihood
error bars (&divide;&radic;2)</b> &mdash; off by default &mdash; divides
them by &radic;2; corrected, they agree with emcee to a few percent.
The report says which was done, and says so too when the estimate
failed and no error could be given.</li>
</ul>

<h3>emcee settings (MCMC Settings)</h3>
<ul>
<li><b>Walkers</b>: at least twice the number of free parameters;
50 covers most fits.</li>
<li><b>Burn-in</b>: the walkers start in a tight ball at the Model
block's values and need a while to spread over the posterior. Those
steps must be discarded, or the error bars come out too wide (10&ndash;40
% on a Yb spectrum). Set it where the traces in the <b>walk plot</b>
have settled &mdash; the dashed line marks it &mdash; typically 5&ndash;10
autocorrelation times (&tau;, listed in the report's [[Burn-in]]
block once a burn-in is set).</li>
<li><b>Steps</b>: enough for ~50 &tau; after the burn-in.</li>
<li><b>Thin</b>: keeps every n-th step. It only saves memory; it does
not improve the estimate.</li>
<li>Starting emcee from a converged leastsq or nelder fit shortens the
burn-in.</li>
</ul>
"""


class FitterHelpDialog(QDialog):
    """Non-modal, so it can stay open while the combos are changed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fitter_help_dialog")
        self.setWindowTitle("Choosing a method and statistics")
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        bg = self.palette().color(QPalette.ColorRole.AlternateBase).name()
        self.browser.setHtml(build_html(header_bg=bg))
        lay.addWidget(self.browser, 1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.manual_button = QPushButton("Full manual page")
        self.manual_button.setToolTip("Help ▸ Documentation, at the "
                                      "Fitter block page")
        box.addButton(self.manual_button,
                      QDialogButtonBox.ButtonRole.ActionRole)
        box.rejected.connect(self.close)
        self.manual_button.clicked.connect(self.open_manual)
        lay.addWidget(box)
        # Wide enough for the tables in a monospace theme font, but
        # never larger than the screen.
        fm = self.fontMetrics()
        w, h = fm.horizontalAdvance("M") * 104, fm.height() * 50
        screen = (parent.screen() if parent is not None
                  else self.screen()).availableGeometry()
        self.resize(min(w, int(screen.width() * 0.9)),
                    min(h, int(screen.height() * 0.9)))

    def open_manual(self):
        """Help ▸ Documentation, at the Fitter page."""
        top = self.parent().window() if self.parent() is not None else None
        win = None
        if top is not None and hasattr(top, "_open_manual"):
            top._open_manual()
            win = getattr(top, "_manual_window", None)
        else:
            from gui.manual.viewer import ManualWindow
            win = ManualWindow(parent=self)
            win.show()
            self._manual_window = win
        if win is not None and hasattr(win, "navigate"):
            win.navigate(MANUAL_PAGE)
        return win


class SquareToolButton(QToolButton):
    """A tool button as tall as ``match`` (usually the combo beside it)
    and exactly as wide, so it stays square at every zoom level."""

    def __init__(self, text, match=None, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setAutoRaise(False)
        self._match = match

    def sizeHint(self):
        if self._match is not None:
            # Exactly the combo's height: a taller button would make its
            # row taller than the ones around it.
            side = self._match.sizeHint().height()
        else:
            s = super().sizeHint()
            side = max(s.width(), s.height())
        return QSize(side, side)

    def minimumSizeHint(self):
        return self.sizeHint()


_dialog = None


def show_fitter_help(anchor=None):
    """Show the guide, reusing the open one rather than stacking copies."""
    global _dialog
    try:
        from shiboken6 import isValid
    except ImportError:                                 # pragma: no cover
        def isValid(obj):
            return obj is not None
    if _dialog is None or not isValid(_dialog):
        parent = anchor.window() if anchor is not None else None
        _dialog = FitterHelpDialog(parent)
        _dialog.setWindowFlag(Qt.WindowType.Window, True)
    _dialog.show()
    _dialog.raise_()
    _dialog.activateWindow()
    return _dialog
