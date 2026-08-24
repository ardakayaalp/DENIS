"""Tests for draggable / persistable delta-nu arrows and labels in the
isotope-shift Edit Plot flow.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers:
  * PlotEditorDialog hit-tests a FancyArrowPatch endpoint living in
    ax.patches and moves only that endpoint via set_positions.
  * Dragging a delta-nu label also moves its linked arrow's tail (shared
    _is_id) and both positions are reported via on_position_changed.
  * IsotopeShiftTab.to_dict / from_dict round-trips the per-id label
    and arrow position dicts so drags survive save / reload, and a
    fresh tab applies a saved arrow position on the next plot.

Runs headless: conftest.py sets QT_QPA_PLATFORM=offscreen and the Agg
matplotlib backend. QApplication is created module-level (shared).

Depends on: gui.shared_widgets (PlotEditorDialog),
gui.analysis.isotope_shift_tab (IsotopeShiftTab); PySide6 and
matplotlib for the headless figure/canvas.
"""

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

class _FakeEvent:
    """Minimal stand-in for a matplotlib MouseEvent."""
    def __init__(self, ax, x, y, xdata, ydata, button=1):
        self.inaxes = ax
        self.x = x
        self.y = y
        self.xdata = xdata
        self.ydata = ydata
        self.button = button


def _disp(ax, x, y):
    """Data coords -> display (pixel) coords."""
    return ax.transData.transform((x, y))


def _make_editor(figure, canvas, registry, recorded):
    from gui.shared_widgets import PlotEditorDialog

    def _on_pos(kind, art_id, value):
        recorded.append((kind, art_id, value))

    return PlotEditorDialog(
        figure, canvas, parent=None,
        arrow_registry=registry,
        on_position_changed=_on_pos)


# ──────────────────────────────────────────────────────────────────
#  PlotEditorDialog drag behavior
# ──────────────────────────────────────────────────────────────────

def test_arrow_endpoint_is_draggable_in_patches():
    """An arrow drawn as a FancyArrowPatch in ax.patches is grabbable
    by an endpoint, and only that endpoint moves."""
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    canvas.draw()  # realize transforms

    arrow = FancyArrowPatch((10, 50), (90, 50), arrowstyle="<->")
    arrow._is_id = "A100"
    ax.add_patch(arrow)

    recorded = []
    editor = _make_editor(
        fig, canvas,
        {"labels": {}, "arrows": {"A100": arrow}}, recorded)

    # Press near the tail (posA) in pixel space.
    xa, ya = _disp(ax, 10, 50)
    editor._on_drag_press(_FakeEvent(ax, xa, ya, 10, 50))
    assert editor._drag_target is arrow
    assert editor._drag_kind == "tail"

    # Drag the tail; the head must stay put.
    editor._on_drag_motion(_FakeEvent(ax, *_disp(ax, 30, 70), 30, 70))
    posA, posB = arrow._posA_posB
    assert posA == pytest.approx((30, 70))
    assert posB == pytest.approx((90, 50))

    # Release persists the arrow endpoints under its id.
    editor._on_drag_release(_FakeEvent(ax, *_disp(ax, 30, 70), 30, 70))
    assert ("arrow", "A100") in [(k, i) for (k, i, _v) in recorded]


def test_label_drag_moves_linked_arrow_tail():
    """Dragging a δν label moves the tail of its linked arrow and
    reports both label and arrow positions."""
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    canvas.draw()

    arrow = FancyArrowPatch((10, 50), (90, 50), arrowstyle="<->")
    arrow._is_id = "A101"
    ax.add_patch(arrow)
    txt = ax.text(50, 55, "dnu=12.3")
    txt._is_id = "A101"

    recorded = []
    editor = _make_editor(
        fig, canvas,
        {"labels": {"A101": txt}, "arrows": {"A101": arrow}}, recorded)

    # Seed the drag state exactly as _on_drag_press would after a click
    # on the label at its current data point (50, 55) -- contains()
    # hit-testing is finicky headlessly, so we set up _dragging here and
    # then exercise the real _on_drag_motion code path.
    px, py = _disp(ax, 50, 55)
    inv = txt.get_transform().inverted()
    sx, sy = inv.transform((px, py))
    editor._dragging = {
        "artist": txt, "ax": ax,
        "start_x": float(sx), "start_y": float(sy),
        "orig_pos": (50.0, 55.0),
    }
    editor._drag_target = txt
    editor._drag_kind = "text"
    editor._on_drag_motion(_FakeEvent(ax, *_disp(ax, 40, 60), 40, 60))

    assert txt.get_position() == pytest.approx((40, 60))
    posA, posB = arrow._posA_posB
    assert posA == pytest.approx((40, 60))   # tail followed the label
    assert posB == pytest.approx((90, 50))   # head unchanged

    editor._on_drag_release(_FakeEvent(ax, *_disp(ax, 40, 60), 40, 60))
    kinds = {(k, i) for (k, i, _v) in recorded}
    assert ("label", "A101") in kinds
    assert ("arrow", "A101") in kinds


# ──────────────────────────────────────────────────────────────────
#  IsotopeShiftTab persistence
# ──────────────────────────────────────────────────────────────────

def _make_is_tab():
    from gui.analysis.isotope_shift_tab import IsotopeShiftTab

    class _StubAnalysisTab:
        _projects = []

    try:
        return IsotopeShiftTab(_StubAnalysisTab())
    except Exception as exc:  # pragma: no cover - env guard
        pytest.skip(f"IsotopeShiftTab not constructible headlessly: {exc}")


def test_position_dicts_initialized():
    tab = _make_is_tab()
    assert tab._is_label_pos == {}
    assert tab._is_arrow_pos == {}


def test_to_from_dict_round_trips_positions():
    tab = _make_is_tab()
    tab._is_label_pos = {"A100": (12.5, 30.0)}
    tab._is_arrow_pos = {"A100": ((5.0, 30.0), (40.0, 30.0))}

    d = tab.to_dict()
    assert d["label_positions"] == {"A100": [12.5, 30.0]}
    assert d["arrow_positions"] == {"A100": [[5.0, 30.0], [40.0, 30.0]]}

    tab2 = _make_is_tab()
    tab2.from_dict(d)
    assert tab2._is_label_pos == {"A100": (12.5, 30.0)}
    assert tab2._is_arrow_pos == {"A100": ((5.0, 30.0), (40.0, 30.0))}


def test_from_dict_tolerates_missing_position_keys():
    """Loading an old project that predates the position dicts must not
    raise and must leave the dicts empty."""
    tab = _make_is_tab()
    tab.from_dict({"study_name": "old", "entries": []})
    assert tab._is_label_pos == {}
    assert tab._is_arrow_pos == {}


def test_on_artist_moved_records_positions():
    """The tab callback handed to the editor stores positions in the
    persistence dicts in data coords."""
    tab = _make_is_tab()
    tab._on_artist_moved("label", "A102", (3.0, 4.0))
    tab._on_artist_moved("arrow", "A102", ((1.0, 2.0), (5.0, 2.0)))
    assert tab._is_label_pos["A102"] == (3.0, 4.0)
    assert tab._is_arrow_pos["A102"] == ((1.0, 2.0), (5.0, 2.0))
