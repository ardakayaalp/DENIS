#!/usr/bin/env python3
"""DENIS GUI launcher script.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Top-level entry point for the DENIS desktop application. Invokes the GUI
``main`` function when run as a script.

Depends on: gui
"""

from gui import main

if __name__ == "__main__":
    main()
