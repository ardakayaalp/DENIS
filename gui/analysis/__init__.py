"""Public API surface for the CLS Toolkit Analysis package.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Re-exports the commonly-used analysis symbols so that
``from gui.analysis import AnalysisTab`` (and the widgets, blocks, fitting
helpers, and project class) resolve directly off the package, hiding the
internal module layout from callers.

Depends on: gui.analysis.tab, gui.analysis.helpers, gui.analysis.blocks,
gui.analysis.fitting, gui.analysis.project, gui.analysis.isotope_shift_tab.
"""

from gui.analysis.tab import AnalysisTab
from gui.analysis.helpers import (
    _show_scrollable_info,
    _color_icon,
    _make_double,
    _make_int,
    _make_analysis_spin,
    _AnalysisSpinBox,
    _BoundsButton,
    PopupPlotWindow,
)
from gui.analysis.blocks import (
    AnalysisBlock,
    SourceBlock,
    ModelBlock,
    FitterBlock,
    OutputBlock,
)
from gui.analysis.fitting import (
    _build_models_on_source,
    _fit_single_run,
    FitWorkerThread,
)
from gui.analysis.project import AnalysisProject
from gui.analysis.isotope_shift_tab import IsotopeShiftTab

__all__ = [
    "AnalysisTab",
    "AnalysisProject",
    "_show_scrollable_info",
    "_color_icon",
    "_make_double",
    "_make_int",
    "_make_analysis_spin",
    "_AnalysisSpinBox",
    "_BoundsButton",
    "PopupPlotWindow",
    "AnalysisBlock",
    "SourceBlock",
    "ModelBlock",
    "FitterBlock",
    "OutputBlock",
    "_build_models_on_source",
    "_fit_single_run",
    "FitWorkerThread",
    "IsotopeShiftTab",
]
