#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  DENIS - Installer (Linux/macOS)
#  Doppler Estimation and Numerical Inference for Spectroscopy
#
#  Usage:
#    ./install.sh            Interactive menu
#    ./install.sh full       Full install (uv + dependencies + shortcut)
#    ./install.sh uv         Install uv only
#    ./install.sh shortcut   Create desktop shortcut only
# ─────────────────────────────────────────────────────────────

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

echo "==========================================================="
echo "  DENIS - Doppler Estimation and Numerical Inference"
echo "==========================================================="
echo

install_uv() {
    if command -v uv &>/dev/null; then
        echo "uv already installed: $(uv --version)"
        return 0
    fi
    echo "uv was not found on PATH."
    echo
    read -rp "Install uv now via the official Astral installer? [Y/n]: " ans
    case "$ans" in
        n|N)
            echo
            echo "Cannot continue without uv. Install manually from"
            echo "  https://docs.astral.sh/uv/getting-started/installation/"
            return 1
            ;;
    esac
    echo
    echo "Installing uv..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo
        echo "ERROR: uv installation failed."
        return 1
    fi
    # Pick up the install for this shell -- uv installs to ~/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo
        echo "uv was installed but is not yet visible in this shell's PATH."
        echo "Open a NEW terminal and re-run ./install.sh."
        return 1
    fi
    return 0
}

sync_deps() {
    echo
    echo "-----------------------------------------------------------"
    echo "  Installing Python + dependencies via uv sync..."
    echo "-----------------------------------------------------------"
    echo
    if ! uv sync; then
        echo
        echo "ERROR: uv sync failed. See output above."
        return 1
    fi
    echo
    echo "Dependencies installed into .venv"
    return 0
}

create_shortcut() {
    local GUI_PY="$APP_DIR/gui.py"
    local ICON="$APP_DIR/icons/denis_512.png"
    local PYTHON="$APP_DIR/.venv/bin/python"
    if [ ! -x "$PYTHON" ]; then
        echo
        echo "ERROR: Project venv not found at .venv/"
        echo 'Run "./install.sh full" first to set up uv and dependencies.'
        return 1
    fi
    local TARGET="$HOME/Desktop"
    mkdir -p "$TARGET"
    local SC
    case "$(uname -s)" in
        Darwin)
            SC="$TARGET/DENIS.command"
            cat > "$SC" <<EOF
#!/usr/bin/env bash
cd "$APP_DIR"
exec "$PYTHON" "$GUI_PY"
EOF
            ;;
        *)
            SC="$TARGET/DENIS.desktop"
            cat > "$SC" <<EOF
[Desktop Entry]
Type=Application
Name=DENIS
Comment=Doppler Estimation and Numerical Inference for Spectroscopy
Exec=$PYTHON $GUI_PY
Path=$APP_DIR
Icon=$ICON
Terminal=false
Categories=Science;Physics;
EOF
            ;;
    esac
    chmod +x "$SC"
    echo "Shortcut created: $SC"
    echo "Python:           $PYTHON"
    return 0
}

MODE="${1:-}"
DO_UV=0; DO_SYNC=0; DO_SC=0
case "$MODE" in
    full)     DO_UV=1; DO_SYNC=1; DO_SC=1 ;;
    uv)       DO_UV=1 ;;
    shortcut) DO_SC=1 ;;
    "")
        echo "Choose an action:"
        echo "  [1] Full install (uv + dependencies + desktop shortcut)"
        echo "  [2] Install uv only"
        echo "  [3] Create desktop shortcut only (assumes .venv exists)"
        echo "  [q] Quit"
        echo
        read -rp "Selection [1/2/3/q]: " CHOICE
        case "$CHOICE" in
            1) DO_UV=1; DO_SYNC=1; DO_SC=1 ;;
            2) DO_UV=1 ;;
            3) DO_SC=1 ;;
            q|Q) exit 0 ;;
            *) echo "Invalid choice."; exit 1 ;;
        esac
        ;;
    *)
        echo "Unknown option '$MODE'. Use: full | uv | shortcut"
        exit 1
        ;;
esac

[ "$DO_UV"   -eq 1 ] && { install_uv      || exit 1; }
[ "$DO_SYNC" -eq 1 ] && { sync_deps       || exit 1; }
[ "$DO_SC"   -eq 1 ] && { create_shortcut || exit 1; }

echo
echo "==========================================================="
echo "  Done. Launch from terminal:  uv run python gui.py"
echo "==========================================================="
