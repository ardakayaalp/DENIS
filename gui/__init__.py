"""DENIS GUI package — exposes the application entry point.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Package initializer for the DENIS graphical interface. Re-exports the
``main`` launcher so the application can be started via ``gui.main``.

Depends on: gui.main_window
"""

from gui.main_window import main

__all__ = ["main"]
