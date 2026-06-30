#!/usr/bin/env bash
# =============================================================================
# DOOM-Killer Setup Script
# Installs system dependencies and prepares the Python virtual environment.
#
# Usage (run once after cloning):
#   chmod +x setup.sh && ./setup.sh
# =============================================================================

set -euo pipefail

# ── Terminal colours ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}    $*"; }
success() { echo -e "${GREEN}[OK]${NC}      $*"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC}   $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo
echo -e "${BOLD}DOOM-Killer — Environment Setup${NC}"
echo    "──────────────────────────────────────────────────────────────"
echo

# ── 0. Refuse to run as root ─────────────────────────────────────────────────
if [[ "${EUID}" -eq 0 ]]; then
    error "Do not run this script as root. Run as your normal user; sudo will be invoked automatically where required."
fi

# ── 1. OS check ──────────────────────────────────────────────────────────────
info "Checking operating system..."
if [[ "$(uname -s)" != "Linux" ]]; then
    error "DOOM-Killer requires a Linux host with eBPF support (detected: $(uname -s))."
fi
success "Linux detected."

# ── 2. Python 3 version check ────────────────────────────────────────────────
info "Checking Python 3 installation..."
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install it with: sudo apt-get install -y python3 python3-venv"
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)'; then
    success "Python ${PY_VER} detected (>= 3.8 required)."
else
    error "Python >= 3.8 is required. Detected ${PY_VER}."
fi

# ── 3. System package dependencies ───────────────────────────────────────────
info "Checking system dependencies..."
MISSING_PKGS=()

# bpftrace — required for eBPF tracepoint probes
if ! command -v bpftrace &>/dev/null; then
    MISSING_PKGS+=("bpftrace")
fi

# stdbuf ships in coreutils; check for the binary directly
if ! command -v stdbuf &>/dev/null; then
    MISSING_PKGS+=("coreutils")
fi

# ab (Apache Bench) — required by the data factory for load generation
if ! command -v ab &>/dev/null; then
    MISSING_PKGS+=("apache2-utils")
fi

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing missing packages: ${MISSING_PKGS[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING_PKGS[@]}"
    success "System packages installed."
else
    success "All system packages are already present."
fi

# ── 4. Docker availability check ─────────────────────────────────────────────
info "Checking for Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker not found. The daemon monitors Docker containers — install Docker before running the daemon."
    warn "Install guide: https://docs.docker.com/engine/install/"
else
    success "Docker detected."
fi

# ── 5. Python virtual environment ────────────────────────────────────────────
VENV_DIR="${SCRIPT_DIR}/.venv"
info "Setting up Python virtual environment..."
if [[ -d "${VENV_DIR}" ]]; then
    info ".venv/ already exists — skipping creation."
else
    python3 -m venv "${VENV_DIR}"
    success "Virtual environment created at .venv/"
fi

# ── 6. Python package installation ───────────────────────────────────────────
info "Installing Python dependencies from requirements.txt..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
success "Python dependencies installed."

# ── 7. Pre-trained model verification ────────────────────────────────────────
MODEL_PATH="${SCRIPT_DIR}/models/doom_model.onnx"
info "Checking for pre-trained ONNX model..."
if [[ -f "${MODEL_PATH}" ]]; then
    success "Pre-trained model found at models/doom_model.onnx."
else
    warn "No pre-trained model found. Train one before running the daemon:"
    warn "  Step 1 (harvest):  sudo .venv/bin/python doom_killer.py harvest-healthy"
    warn "  Step 2 (train):         .venv/bin/python doom_killer.py train"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo -e "${GREEN}${BOLD}Setup complete.${NC}"
echo
echo -e "  Run the daemon (monitors all Docker containers):"
echo -e "    ${BOLD}sudo .venv/bin/python doom_killer.py run${NC}"
echo
echo -e "  Monitor a specific container:"
echo -e "    ${BOLD}sudo .venv/bin/python doom_killer.py run --target <container-name>${NC}"
echo
