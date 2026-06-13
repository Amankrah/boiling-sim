#!/usr/bin/env bash
# ============================================================================
# Phase 5 Bootstrap (Linux / macOS / WSL): Rust toolchain + maturin install
# ============================================================================
# Run AFTER scripts/setup_wsl_env.sh has finished (driver, CUDA Toolkit, uv,
# Python 3.11 venv). Picks up where Phase 0 left off and gets the Rust
# extension (`sim_core`) building into the venv.
#
# Usage (from project root):
#   bash scripts/bootstrap.sh
#
# What this does:
#   1. Installs rustup if missing.
#   2. The rust-toolchain.toml at the repo root pins channel 1.85; rustup
#      auto-installs that channel + clippy + rustfmt on first cargo call.
#   3. Verifies CUDA_PATH is set (cuda-kernels/build.rs requires it on
#      Windows; on Linux the build script searches /usr/local/cuda).
#   4. Runs `uv pip install -e .` which invokes maturin to build sim_core
#      and drop the extension into the editable install.
# ============================================================================

set -euo pipefail

info() { printf '\033[32m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m  %s\n' "$*"; }
fail() { printf '\033[31m[ERR ]\033[0m %s\n' "$*"; exit 1; }

cd "$(dirname "$0")/.."

# ---- 1. Rustup ----
if ! command -v rustup >/dev/null 2>&1; then
    info "Installing rustup (will auto-install Rust 1.85 from rust-toolchain.toml)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain none --profile minimal
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
else
    info "rustup: $(rustup --version)"
fi

info "Warming Rust toolchain (rust-toolchain.toml pins 1.85)..."
cargo --version

# ---- 2. CUDA toolkit reachability ----
if [[ -z "${CUDA_PATH:-}" ]]; then
    for candidate in /usr/local/cuda /usr/lib/cuda; do
        if [[ -d "$candidate/lib64" ]]; then
            export CUDA_PATH="$candidate"
            info "Derived CUDA_PATH=$CUDA_PATH"
            break
        fi
    done
fi
if [[ -z "${CUDA_PATH:-}" ]]; then
    warn "CUDA Toolkit not found at /usr/local/cuda or /usr/lib/cuda."
    warn "Install nvidia-cuda-toolkit (Linux) or set CUDA_PATH explicitly."
    fail "Aborting bootstrap."
fi
info "CUDA_PATH: $CUDA_PATH"

# ---- 3. venv check ----
if [[ ! -d .venv ]]; then
    fail "No .venv found. Run scripts/setup_wsl_env.sh first."
fi

# ---- 4. Editable install via maturin ----
info "Installing boilingsim (editable) -- maturin will build sim_core..."
uv pip install -e .

# ---- 5. Smoke verify ----
info "Smoke-verifying sim_core import..."
./.venv/bin/python -c "
from sim_core.props import MaterialProps
mp = MaterialProps.from_json('data/materials.json')
assert mp.beta_100c == 7.5e-4, f'beta_100c got {mp.beta_100c}'
print('sim_core.props OK:', mp)
"

info "========================================"
info "  Phase 5 bootstrap complete."
info "  Run pytest python/tests/ to verify the full suite."
info "========================================"
