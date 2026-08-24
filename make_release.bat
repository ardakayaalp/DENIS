@echo off
REM ─────────────────────────────────────────────────────────────
REM  DENIS - Package a versioned release snapshot under share/
REM
REM  Usage:  make_release.bat [version]
REM  Output: share\v<version>\  (full runnable copy of the app,
REM          minus build artefacts, caches, user data, and git
REM          metadata).
REM ─────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

if "%~1"=="" (
    set /p "VERSION=Enter version (e.g. 1.0.0): "
) else (
    set "VERSION=%~1"
)

if "!VERSION!"=="" (
    echo ERROR: Version required.
    exit /b 1
)

set "DEST=share\v!VERSION!"

echo.
echo ===========================================================
echo   DENIS - Packaging release v!VERSION!
echo   Destination: !DEST!\
echo ===========================================================
echo.

if exist "!DEST!" (
    set /p "OVERWRITE=Destination already exists. Overwrite? [y/N]: "
    if /i not "!OVERWRITE!"=="y" (
        echo Aborted.
        exit /b 1
    )
    rmdir /s /q "!DEST!"
)

REM ── Top-level files (runnable entry points + install scripts) ──
REM  install.bat/.sh handle uv install, dependency sync, AND shortcut
REM  creation via a menu / CLI subcommand -- the separate
REM  create_shortcut.* scripts were folded in.
robocopy . "!DEST!" ^
    cls_estimate.py gui.py ^
    pyproject.toml uv.lock ^
    README.md ^
    install.bat install.sh ^
    /NJH /NJS /NDL /NFL /NP

REM ── Source packages (skip __pycache__ and the regenerable manual image
REM    cache). gui\manual\ -- the Documentation viewer, content, committed
REM    equation images (gui\manual\figs\eq) and screenshots (gui\manual\figs)
REM    -- ships in full via /E; only gui\manual\_cache (runtime-rendered,
REM    regenerated on first open) is skipped. ──
robocopy cls_estimations "!DEST!\cls_estimations" /E /XD __pycache__ /NJH /NJS /NDL /NFL /NP
robocopy gui             "!DEST!\gui"             /E /XD __pycache__ _cache /XF SCREENSHOTS.md /NJH /NJS /NDL /NFL /NP

REM ── Static assets ──
robocopy icons "!DEST!\icons" /E /NJH /NJS /NDL /NFL /NP

REM ── docs/ : skip dev-only material (superpowers plans/specs, the LaTeX
REM    manual's PLAN.md + build scripts, and build artefacts). The
REM    user-facing manual now lives IN-APP (Help -> Documentation), so only
REM    non-dev docs assets are shipped. ──
robocopy docs "!DEST!\docs" /E /XD build __pycache__ superpowers /XF PLAN.md build.bat build.sh /NJH /NJS /NDL /NFL /NP

REM ── Example configs only (skip user-specific YAMLs) ──
robocopy configs "!DEST!\configs" example_*.yaml /NJH /NJS /NDL /NFL /NP

REM Robocopy returns 0-7 for various flavours of success; only 8+ is failure.
REM Easier sanity check: make sure the entry-point file landed.
if not exist "!DEST!\cls_estimate.py" (
    echo.
    echo ERROR: snapshot creation failed -- entry point missing.
    exit /b 1
)

echo.
echo ===========================================================
echo   Snapshot created: !DEST!\
echo.
echo   Recipient should:
echo     1. cd into !DEST!\
echo     2. run "install.bat" (Windows) or "./install.sh" (Linux/macOS)
echo        - menu offers: Full install / uv only / shortcut only
echo        - or pass the action as an arg, e.g. "install.bat full"
echo     3. launch via the desktop shortcut, or
echo        uv run python gui.py
echo ===========================================================
exit /b 0
