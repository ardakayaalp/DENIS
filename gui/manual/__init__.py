"""In-UI interactive user manual package for DENIS.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Provides the modeless documentation window (Help > Documentation, F1): a
left-pane contents tree and a right-pane rich-text viewer supporting hyperlinks,
back/forward navigation, matplotlib-rendered equations, schematic diagrams, code
snippets, and figures. The window is constructed lazily by gui.main_window, so
importing this package has no hard PySide6 dependency until the manual is opened.

Sub-modules: structure (auto-generated TOC + per-page plan), mathrender (cached
mathtext/LaTeX equation PNGs), equations (curated equation sources), diagrams
(matplotlib schematics), render and builder (HTML page-building helpers),
content/ (authored page registry with a planned-page fallback), and viewer (the
ManualWindow widget).

Depends on: standard library and third-party packages only (sub-modules are
imported lazily by the viewer, not at package import time).
"""
