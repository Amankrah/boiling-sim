# Getting Started — Windows Native Setup

This project is set up for **Windows-native development** on the Lambda Vector workstation. WSL2 is not required.

## Environment Status Check

| Component | Needed | Your Status |
|-----------|--------|-------------|
| NVIDIA driver | 560+ | ✅ 595.97 |
| VS 2019/2022 Build Tools (MSVC) | Required by nvcc | ✅ VS 2019 BuildTools |
| CUDA Toolkit 12.6 | Required | ❌ **install next** |
| Python 3.11 | Required (Warp ≤3.12) | ❌ will be installed by uv |
| Rust stable | 1.75+ | ✅ 1.90 |
| Node.js 20+ | Required for dashboard | ✅ 22.13.1 |
| pnpm | Required for dashboard | ✅ 10.2.1 |
| uv | Recommended | ❌ will be installed |

## Step 1 — Install CUDA Toolkit 12.6 (Manual)

Download and run:
- **URL:** https://developer.nvidia.com/cuda-12-6-0-download-archive
- **Select:** Windows → x86_64 → 11 → **exe (local)**
- **Size:** ~3 GB installer, ~6 GB installed
- **Install type:** Express (recommended)

After install, **close and reopen PowerShell** so the `CUDA_PATH` environment variable and the updated `PATH` take effect.

Verify:
```powershell
nvcc --version
# Should print: Cuda compilation tools, release 12.6, ...
echo $env:CUDA_PATH
# Should print: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

## Step 2 — Run the Setup Script

From the project root:
```powershell
cd C:\Users\Windows\Desktop\Dev_Projects\boiling-sim
.\scripts\setup_windows_env.ps1
```

The script will:
1. Verify driver, MSVC, and CUDA Toolkit
2. Compile and run a CUDA hello-world kernel
3. Install `uv` if missing
4. Create a Python 3.11 venv at `.venv\`
5. Install Warp and all Python dependencies
6. Install cargo tools (`cargo-watch`, `cargo-nextest`)
7. Initialize git

If the script exits with an error, fix the reported issue and re-run — it's idempotent.

## Step 3 — Verify Everything Works

```powershell
# Activate venv (do this each new shell)
.\.venv\Scripts\Activate.ps1

# 1. Warp SPH example (records fluid throughput)
python -m warp.examples.core.example_sph

# 2. Warp FEM example (records FE solver throughput)
python -m warp.examples.fem.example_diffusion

# 3. Rust + CUDA smoke test
cargo build --release
cargo test --release -p cuda-kernels

# 4. Python smoke tests
pytest python\tests\
```

All four should pass. Record the throughput numbers in `benchmarks\baseline.md`.

## Step 4 — Phase 0 Exit

When all of these are true, Phase 0 is complete and you can move to Phase 1:

- [ ] `nvidia-smi` shows RTX 6000 Ada, driver 560+
- [ ] `nvcc --version` shows 12.6
- [ ] Warp SPH and FEM examples run
- [ ] `cargo test -p cuda-kernels` passes
- [ ] `pytest python\tests\` passes
- [ ] `benchmarks\baseline.md` filled in with measured throughput

## Directory Layout

```
boiling-sim/
├── Cargo.toml              Rust workspace
├── pyproject.toml          Python project
├── package.json            Node workspace
├── crates/
│   ├── sim-core/           Rust orchestration + PyO3
│   ├── cuda-kernels/       Hand-written CUDA (.cu files)
│   └── ws-server/          WebSocket streaming server
├── python/boilingsim/      Main Python package (geometry, fluid, thermal, boiling, nutrient, pipeline)
├── python/tests/           Pytest smoke tests
├── configs/scenarios/      YAML scenario definitions
├── data/materials.json     Material properties
├── benchmarks/             Performance baselines
├── scripts/                Setup + utility scripts
├── web/                    React/R3F dashboard (Phase 6)
└── docs/                   Planning documents (already in repo root)
```

## Phase Plan

| Phase | Duration | Goal |
|-------|----------|------|
| **0** | 2 weeks | **Environment + baseline benchmarks (YOU ARE HERE)** |
| 1 | 3 weeks | Parametric USD scene (pot + water + carrot) |
| 2 | 5 weeks | Single-phase CFD + conjugate heat transfer |
| 3 | 6 weeks | Nucleate boiling + bubbles |
| 4 | 4 weeks | Carrot nutrient retention coupling |
| 5 | 4 weeks | Rust + custom CUDA acceleration |
| 6 | 5 weeks | Live 3D dashboard |
| 7 | optional | Omniverse Kit migration |

See `multiphysics_boiling_developer_guide.md` for full technical detail.
