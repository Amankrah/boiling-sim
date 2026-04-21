#!/usr/bin/env bash
# ============================================================================
# Phase 0: WSL2 Ubuntu Environment Setup for boiling-sim
# ============================================================================
# Run INSIDE WSL2 Ubuntu 24.04 after:
#   1. Updating the NVIDIA driver on Windows host to 560+
#   2. Creating C:\Users\<you>\.wslconfig with memory=240GB
#   3. Running 'wsl --shutdown' and relaunching Ubuntu
#
# Usage:  cd /path/to/boiling-sim && bash scripts/setup_wsl_env.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

info()  { echo -e "\033[0;32m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; exit 1; }

# ---- Pre-flight: GPU check ----
info "Checking GPU visibility from WSL..."
if ! nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Install NVIDIA driver on Windows host only."
fi

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
info "Detected NVIDIA driver: $DRIVER_VER"
if [ "$DRIVER_MAJOR" -lt 560 ]; then
    error "Driver $DRIVER_VER too old. Need >= 560.28 for CUDA 12.6."
fi
info "Driver version OK (>= 560)."

# ---- Step 3: System packages ----
info "Installing system packages..."
sudo apt update
info "Repairing broken/partial installs (required before adding build deps)..."
sudo apt-get --fix-broken install -y
sudo apt install -y \
    build-essential git curl pkg-config libssl-dev \
    cmake ninja-build python3-dev python3-venv \
    libblas-dev liblapack-dev libhdf5-dev \
    clang lld unzip

# ---- Step 4: CUDA Toolkit (12.6+; skip if nvcc 12.x or newer already on PATH) ----
# Lambda / distro packages may ship 12.8 as "nvidia-cuda-toolkit"; do not stack NVIDIA's 12.6 on top.
if nvcc --version 2>/dev/null | grep -qE 'release 1[2-9]\.|release [2-9][0-9]\.'; then
    info "CUDA compiler already present; skipping apt CUDA install."
    nvcc --version | grep release || true
else
    info "Installing CUDA Toolkit 12.6 (WSL-specific)..."
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-6
    cd "$REPO_ROOT"

    if ! grep -q 'cuda-12.6' ~/.bashrc; then
        {
            echo ''
            echo '# CUDA 12.6'
            echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH'
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}'
        } >> ~/.bashrc
    fi
    export PATH=/usr/local/cuda-12.6/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
fi

info "Verifying nvcc..."
nvcc --version || error "nvcc not found"

# CUDA smoke test
info "CUDA hello-world smoke test..."
HELLO_CU=$(mktemp /tmp/hello_cuda_XXXX.cu)
cat > "$HELLO_CU" << 'CUDAEOF'
#include <cstdio>
__global__ void hi() { printf("GPU thread %d alive\n", threadIdx.x); }
int main() { hi<<<1, 4>>>(); cudaDeviceSynchronize(); return 0; }
CUDAEOF
nvcc "$HELLO_CU" -o /tmp/hello_cuda && /tmp/hello_cuda
rm -f "$HELLO_CU" /tmp/hello_cuda
info "CUDA smoke test passed!"

# ---- Step 5: Python with uv ----
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_ROOT"
if [ ! -d .venv ]; then
    info "Creating Python 3.11 venv..."
    uv venv --python 3.11
fi

info "Installing Python dependencies..."
source .venv/bin/activate
uv pip install \
    "warp-lang[examples]" \
    numpy scipy matplotlib h5py \
    pyvista trimesh pygmsh meshio \
    usd-core \
    fastapi "uvicorn[standard]" websockets \
    zstandard pyyaml pydantic \
    pytest pytest-benchmark

# ---- Step 6: Rust ----
if ! command -v rustc &>/dev/null; then
    info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
rustup default stable
rustup component add clippy rustfmt
info "Installing cargo tools..."
cargo install cargo-watch cargo-nextest maturin 2>/dev/null || true

# ---- Step 7: Node.js 20 + pnpm ----
if ! command -v node &>/dev/null; then
    info "Installing Node.js 20 LTS..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs
    sudo npm install -g pnpm
else
    info "Node.js already installed: $(node --version)"
fi

# ---- Init git ----
cd "$REPO_ROOT"
if [ ! -d .git ]; then
    git init
fi

# ---- Summary ----
echo ""
info "========================================"
info "  Phase 0 Environment Setup Complete!"
info "========================================"
echo ""
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Driver: $DRIVER_VER"
CUDA_VER=$(nvcc --version 2>&1 | grep release | sed 's/.*release //' | sed 's/,.*//')
echo "  CUDA:   $CUDA_VER"
echo "  Python: $(python --version 2>&1)"
echo "  Rust:   $(rustc --version 2>&1)"
echo "  Node:   $(node --version 2>&1)"
echo ""
info "Next steps:"
info "  1. source .venv/bin/activate"
info "  2. python -m warp.examples.core.example_sph"
info "  3. python -m warp.examples.fem.example_diffusion"
info "  4. cargo build --release"
info "  5. pytest python/tests/"
info "  6. Fill in benchmarks/baseline.md"
