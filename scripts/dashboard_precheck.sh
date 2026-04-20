#!/usr/bin/env bash
# Phase 6 solver-container entrypoint precheck.
#
# Mitigates "Risk 1" from the Phase 6 plan: WSL2 + Docker GPU
# passthrough frequently fails silently -- Warp initializes, reports
# "0 CUDA devices visible", and falls back to CPU. Users don't notice
# until three minutes into a run when s/sim-s numbers are suspicious.
#
# This script fails LOUDLY at container start if either nvidia-smi is
# missing or Warp can't see a CUDA device, pointing at the WSL2 / NVIDIA
# Container Toolkit docs in GETTING_STARTED.md.

set -euo pipefail

echo "[precheck] Checking nvidia-smi visibility..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<EOF
[precheck] FAIL: nvidia-smi binary not found in this container.

The solver container requires the NVIDIA Container Toolkit on the host.
On Linux:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
On WSL2:
  Ensure your Windows-side NVIDIA driver is current, then install the
  toolkit inside WSL and restart Docker Desktop.

See GETTING_STARTED.md, section "Dashboard / Phase 6 deployment".
EOF
    exit 1
fi

# A clean `nvidia-smi -L` run confirms at least one GPU is visible.
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[precheck] FAIL: nvidia-smi is present but returned no devices." >&2
    nvidia-smi || true
    exit 1
fi
echo "[precheck] nvidia-smi:"
nvidia-smi -L

echo "[precheck] Checking Warp CUDA visibility..."
python - <<'PYEOF'
import sys
try:
    import warp as wp
except ImportError as e:
    sys.stderr.write(
        "[precheck] FAIL: warp-lang not importable inside the container. "
        f"{e}\n"
        "This means pip install -e .[dashboard] failed earlier. Check the "
        "Dockerfile.solver build log.\n"
    )
    sys.exit(1)

wp.init()
n = wp.get_cuda_device_count() if hasattr(wp, "get_cuda_device_count") else len(wp.get_devices())
if n == 0:
    sys.stderr.write(
        "[precheck] FAIL: Warp initialised but sees zero CUDA devices.\n"
        "Symptoms: solver would run on CPU at 100x real-time. Refusing to proceed.\n"
        "Fix: verify GPU passthrough in docker-compose.yml (deploy.resources.\n"
        "reservations.devices) and that the NVIDIA Container Toolkit is\n"
        "installed on the host. See GETTING_STARTED.md.\n"
    )
    sys.exit(1)
print(f"[precheck] Warp sees {n} CUDA device(s). Ready.")
PYEOF

echo "[precheck] All checks passed. Launching run_dashboard.py."
exec python scripts/run_dashboard.py \
    --config "${BOILINGSIM_CONFIG:-configs/scenarios/default.yaml}" \
    --duration "${BOILINGSIM_DURATION:-0}" \
    --dx-mm "${BOILINGSIM_DX_MM:-2.0}" \
    --pressure-iters "${BOILINGSIM_PRESSURE_ITERS:-100}" \
    --snapshot-hz "${BOILINGSIM_SNAPSHOT_HZ:-30}" \
    --ingest-host "${BOILINGSIM_INGEST_HOST:-127.0.0.1}" \
    --ingest-port "${BOILINGSIM_INGEST_PORT:-8765}" \
    --control-host "${BOILINGSIM_CONTROL_HOST:-127.0.0.1}" \
    --control-port "${BOILINGSIM_CONTROL_PORT:-8766}"
