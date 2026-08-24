"""Shared look-and-feel for the pop-out dialogs.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

The tabs have a settled visual language: a white publication-style plot
sits inside a dark *card* whose header strip carries the panel title on
the left and a compact matplotlib toolbar on the right, and the
surrounding controls sit in bordered, blue-titled group boxes.

Since 2026-07-24 the *language itself* lives in ``gui.theme``: its
application-wide stylesheet covers the group boxes, tables, tabs and
plot-card object names that used to be defined here, so every dialog
gets the look without opting in. This module keeps the layout helpers
(:func:`make_plot_card`, :func:`stabilize_toolbar`,
:func:`section_note`) and stays importable for the existing callers;
``style_dialog`` is now a no-op kept for compatibility.

Note the plots stay **white**. That is deliberate and app-wide (see
``shared_widgets._STATIC_PLOT_STYLE``: CMU Serif, ticks-in, framed
legend) -- these figures are publication output, and a dark diagnostic
plot would be the thing that looked out of place.

Depends on: PySide6, gui.theme.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from gui.theme import ACCENT_TITLE as ACCENT

#: Kept for compatibility; the rules moved into gui.theme.build_qss().
DIALOG_QSS = ""


def style_dialog(dialog) -> None:
    """No-op since the global theme covers dialogs; kept for callers."""
    return None


def stabilize_toolbar(toolbar, label_width: int = 180) -> None:
    """Pin a matplotlib toolbar's footprint and right-anchor its buttons.

    ``locLabel`` has an Expanding size policy and a Fixed-policy toolbar
    takes its size from sizeHint, so the toolbar grew and shrank on every
    mouse move and jiggled the header. Same fix the tabs already apply
    (``preanalysis_tab._stabilize_toolbar``); duplicated rather than
    imported to keep this module free of tab dependencies.
    """
    label = getattr(toolbar, "locLabel", None)
    if label is None:
        return
    label.setFixedWidth(label_width)
    label.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    actions = toolbar.actions()
    if len(actions) < 2:
        return
    loc_action = actions[-1]      # matplotlib appends locLabel last
    toolbar.removeAction(loc_action)
    toolbar.insertAction(actions[0], loc_action)


def make_plot_card(title: str, canvas, toolbar, parent=None,
                   subtitle: str = "") -> QFrame:
    """A plot in the app's card: dark header strip, title left, toolbar right.

    The same construction the Pre-Analysis panels use, so a dialog's plot and
    a tab's plot are visibly the same kind of object.
    """
    card = QFrame(parent)
    card.setObjectName("plotCard")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(4, 3, 4, 4)
    lay.setSpacing(3)

    hdr = QHBoxLayout()
    hdr.setContentsMargins(4, 1, 2, 0)
    hdr.setSpacing(8)

    lbl = QLabel(title)
    lbl.setObjectName("plotCardTitle")
    hdr.addWidget(lbl)
    if subtitle:
        sub = QLabel(subtitle)
        sub.setObjectName("sectionNote")
        hdr.addWidget(sub)
    hdr.addStretch(1)

    if toolbar is not None:
        toolbar.setMaximumHeight(24)
        toolbar.setSizePolicy(QSizePolicy.Policy.Fixed,
                              QSizePolicy.Policy.Fixed)
        stabilize_toolbar(toolbar)
        hdr.addWidget(toolbar)

    lay.addLayout(hdr)
    lay.addWidget(canvas, 1)
    return card


def section_note(text: str, parent=None) -> QLabel:
    """A muted, wrapping explanatory line."""
    lbl = QLabel(text, parent)
    lbl.setObjectName("sectionNote")
    lbl.setWordWrap(True)
    lbl.setTextFormat(Qt.TextFormat.RichText)
    return lbl
