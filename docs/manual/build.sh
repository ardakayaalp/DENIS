#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Build the DENIS user manual PDF (Linux/macOS + TeX Live).
#  Output: build/manual.pdf
# ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

if ! command -v latexmk &>/dev/null; then
    echo "ERROR: latexmk not found on PATH."
    echo "Install TeX Live / MacTeX and ensure latexmk is available."
    exit 1
fi

latexmk -pdf -interaction=nonstopmode -outdir=build manual.tex
rc=$?
if [ "$rc" -ne 0 ]; then
    echo
    echo "Build FAILED. See build/manual.log for details."
    exit "$rc"
fi

echo
echo "==========================================================="
echo "  PDF built: build/manual.pdf"
echo "==========================================================="
