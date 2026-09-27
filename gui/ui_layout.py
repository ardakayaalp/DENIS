"""Persistence for the adjustable window geometry (splitter positions).

Date:    2026-09-20
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Column widths and panel splits are tuned for a particular screen, so
throwing them away on every reload makes the app feel disposable. Each
tab exposes ``ui_layout()`` / ``apply_ui_layout(d)``; MainWindow stores
the union under a top-level ``ui_layout`` key of the save file, beside
the tab sections (it is window state, not analysis state).

Sizes are plain int lists. A stored layout is applied only when the
splitter still has the same number of panes, so a file saved before a
pane was added or removed is IGNORED rather than mis-applied -- an
older save must never be able to collapse a panel to zero width.

A restore onto a splitter that is not yet visible is DEFERRED until it
is shown. Tabs other than the current one are hidden pages whose
layouts have not run, so their splitters still carry a construction-
time width; setSizes there scales the stored split down to that
width, a minimum-width pane wins, and the other pane is squeezed to 0.
A collapsed QSplitter pane then stays collapsed through every later
resize -- which is how the Isotope Shifts plot vanished (2026-09-22).

Depends on: PySide6 (duck-typed; any QSplitter-like object works).
"""


def splitter_sizes(sp):
    """Current pane sizes of *sp*, or None if there is nothing useful.

    None (rather than []) so callers can drop the key entirely and keep
    the save file free of noise for splitters that never existed.
    """
    if sp is None:
        return None
    try:
        sizes = [int(s) for s in sp.sizes()]
    except Exception:
        return None
    # A hidden or not-yet-laid-out splitter reports all zeros; storing
    # that would restore an invisible panel on the next load.
    if not sizes or sum(sizes) <= 0:
        return None
    return sizes


#: No shown pane is ever restored narrower than this. Zero is a
#: collapsed panel, not a layout anyone chose.
MIN_PANE = 40


def _pane_shown(sp, i):
    """False only for a pane the app has hidden on purpose, which
    legitimately reports 0. Unknown (duck-typed) means shown."""
    try:
        return bool(sp.widget(i).isVisibleTo(sp))
    except Exception:
        return True


def _ready(sp):
    """True when *sp* is on screen with a real width. A duck-typed
    stand-in without the Qt API counts as ready."""
    try:
        return bool(sp.isVisible()) and int(sp.width()) > 0
    except Exception:
        return True


def _mark_restored(sp):
    """Tell the owning tab a saved layout is in charge, so it does not
    lay its own default split over the top."""
    try:
        sp.setProperty("ui_layout_restored", True)
    except Exception:
        pass


def _defer(sp, sizes):
    """Apply *sizes* once *sp* is first shown."""
    from PySide6.QtCore import QEvent, QObject, QTimer

    pending = getattr(sp, "_ui_layout_pending", None)
    if pending is not None:
        pending.sizes = sizes       # a later load wins
        return

    class _Pending(QObject):
        def __init__(self, splitter, sz):
            super().__init__(splitter)
            self.sizes = sz

        def eventFilter(self, obj, ev):
            if (ev.type() in (QEvent.Type.Show, QEvent.Type.Resize)
                    and _ready(obj)):
                obj.removeEventFilter(self)
                obj._ui_layout_pending = None
                sz = self.sizes
                # After the splitter has finished its own show/resize
                # handling, or that handling redistributes over us.
                QTimer.singleShot(0, lambda: obj.setSizes(sz))
                self.deleteLater()
            return False

    p = _Pending(sp, sizes)
    sp._ui_layout_pending = p
    sp.installEventFilter(p)


def apply_splitter_sizes(sp, sizes):
    """Restore *sizes* onto *sp*. Returns True when it was applied or
    scheduled.

    No-ops unless the pane count still matches, the sizes are positive
    and no shown pane is narrower than MIN_PANE -- see the module
    docstring. Deferred until *sp* is visible.
    """
    if sp is None or not sizes:
        return False
    try:
        sizes = [int(s) for s in sizes]
    except (TypeError, ValueError):
        return False
    if len(sizes) != sp.count() or sum(sizes) <= 0:
        return False
    if any(s < MIN_PANE and _pane_shown(sp, i)
           for i, s in enumerate(sizes)):
        return False
    _mark_restored(sp)
    if not _ready(sp):
        try:
            _defer(sp, sizes)
            return True
        except Exception:
            pass          # no Qt event loop: apply now, best effort
    try:
        sp.setSizes(sizes)
    except Exception:
        return False
    return True


def collect(**splitters):
    """Build a layout dict from ``name=splitter`` pairs, dropping the
    ones with nothing to store."""
    out = {}
    for name, sp in splitters.items():
        sizes = splitter_sizes(sp)
        if sizes:
            out[name] = sizes
    return out


def restore(d, **splitters):
    """Apply a layout dict built by :func:`collect`."""
    if not isinstance(d, dict):
        return
    for name, sp in splitters.items():
        apply_splitter_sizes(sp, d.get(name))
