@echo off
REM ─────────────────────────────────────────────────────────────
REM  Build the DENIS user manual PDF (Windows + MiKTeX).
REM  Output: build\manual.pdf
REM ─────────────────────────────────────────────────────────────
setlocal
cd /d "%~dp0"

where latexmk >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: latexmk not found on PATH.
    echo Install MiKTeX and ensure latexmk is available, then re-run.
    pause
    exit /b 1
)

latexmk -pdf -interaction=nonstopmode -outdir=build manual.tex
set RC=%ERRORLEVEL%
if %RC% neq 0 (
    echo.
    echo Build FAILED. See build\manual.log for details.
    pause
    exit /b %RC%
)

echo.
echo ===========================================================
echo   PDF built: build\manual.pdf
echo ===========================================================
exit /b 0
