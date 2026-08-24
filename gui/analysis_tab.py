"""Backward-compatibility shim for the analysis tab module.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Re-exports the public API of the ``gui.analysis`` package so that existing
``from gui.analysis_tab import ...`` statements keep working after the
analysis tab was split into a sub-package.

Depends on: gui.analysis, gui.analysis.tab, gui.analysis.helpers,
gui.shared_widgets
"""

import warnings as _warnings

# Nothing in this repo imports this shim (AnalysisTab is imported from
# gui.analysis directly); it exists only for external user scripts that still
# do `from gui.analysis_tab import ...`. Emit a DeprecationWarning so any such
# use is visible and can migrate to `gui.analysis`, while keeping the shim so
# those scripts don't break (code review 2026-06-02, dead-analysis-tab-shim).
_warnings.warn(
    "gui.analysis_tab is a compatibility shim; import from 'gui.analysis' "
    "instead.", DeprecationWarning, stacklevel=2)

# Re-export the full public API from the new sub-package
from gui.analysis import *  # noqa: F401,F403

# Explicit re-exports for the symbols known to be imported externally
from gui.analysis.tab import AnalysisTab  # noqa: F811
from gui.analysis.helpers import _show_scrollable_info  # noqa: F811
from gui.shared_widgets import _load_settings, _save_settings  # noqa: F401
