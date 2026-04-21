# Boiling Sim

**GPU-accelerated multiphysics 3D boiling simulation** with coupled thermal, fluid, and nutrient retention modeling.

A carrot-boiling case study built with NVIDIA Warp, CUDA, and Rust — developed at the SASEL Lab, McGill University.

## Overview

This project simulates the full physics of boiling a carrot in a steel pot: natural convection in water, conjugate heat transfer through the pot wall, nucleate bubble dynamics, and nutrient degradation inside the carrot. The simulation runs on GPU via NVIDIA Warp kernels with optional hand-written CUDA and Rust acceleration, and streams results to a live 3D web dashboard.

### Key capabilities

- **Conjugate heat transfer** — Coupled solid (pot wall) and fluid (water) thermal solve on a staggered MAC grid with implicit diffusion
- **Natural convection** — Boussinesq-approximated Navier-Stokes with semi-Lagrangian advection and pressure projection
- **Nucleate boiling** — Onset-of-nucleate-boiling detection, bubble nucleation, growth, and departure via Lagrangian particle tracking
- **Nutrient retention** — First-order Arrhenius degradation kinetics for beta-carotene and vitamin C inside a tetrahedral FE carrot mesh
- **GPU acceleration** — All compute kernels run on NVIDIA GPUs via Warp, with optional CUDA/Rust paths for critical hotspots
- **Live 3D dashboard** — WebSocket-streamed visualization with React and React Three Fiber

## Project structure

```
boiling-sim/
├── Cargo.toml              # Rust workspace
├── pyproject.toml           # Python project (boilingsim package)
├── package.json             # Node workspace for dashboard
├── crates/
│   ├── sim-core/            # Rust orchestration + PyO3 bindings
│   ├── cuda-kernels/        # Hand-written CUDA (.cu files)
│   └── ws-server/           # WebSocket streaming server
├── python/
│   └── boilingsim/          # Main Python package
│       ├── geometry.py      # USD scene generation (pot, water, carrot)
│       ├── fluid.py         # Navier-Stokes solver (MAC grid)
│       ├── thermal.py       # Conjugate heat transfer solver
│       ├── boiling.py       # Nucleate boiling + bubble dynamics
│       ├── nutrient.py      # Nutrient degradation kinetics
│       ├── pipeline.py      # Multi-phase simulation orchestrator
│       ├── config.py        # Configuration management
│       └── scenario.py      # YAML scenario runner
├── python/tests/            # Pytest test suite
├── configs/scenarios/       # YAML scenario definitions
├── data/materials.json      # Material properties database
├── benchmarks/              # Performance baselines and validation
├── scripts/                 # Setup and utility scripts
└── web/                     # React/R3F dashboard (Phase 6)
```

## Prerequisites

| Component | Version |
|-----------|---------|
| NVIDIA GPU | Ada Lovelace or newer (tested on RTX 6000 Ada 48 GB) |
| NVIDIA Driver | 560+ |
| CUDA Toolkit | 12.6+ (12.6 baseline; 12.8 from Ubuntu/Lambda apt is OK) |
| Python | 3.11 (required by Warp, must be < 3.13) |
| Rust | 1.75+ |
| Node.js | 20+ |
| OS | Windows 11 (native) or Linux via WSL2 |

## Quick start

### Windows (native)

1. **Install CUDA Toolkit 12.6** — Download from the [NVIDIA CUDA 12.6 Archive](https://developer.nvidia.com/cuda-12-6-0-download-archive) and run the installer (Express install is fine).

2. **Run the setup script** from the repo root in PowerShell:

   ```powershell
   cd boiling-sim
   .\scripts\setup_windows_env.ps1
   ```

   This checks the toolchain, creates a Python 3.11 virtualenv, and installs Python dependencies. For day-to-day work in a new shell, you can dot-source `.\scripts\activate_dev_env.ps1` so MSVC, CUDA, and the venv are on `PATH`.

3. **Activate the environment**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install the `boilingsim` package (editable)** so the `boiling-sim-scenario` CLI is available:

   ```powershell
   uv pip install -e ".[dev]"
   ```

5. **Verify**

   ```powershell
   pytest python\tests\
   cargo test --release -p cuda-kernels
   ```

### Linux / WSL2 (Ubuntu)

Use this when developing inside **WSL2** or **native Ubuntu** with an NVIDIA GPU visible in the distro (`nvidia-smi` works) and a **560+** driver on the host (for WSL2: install the driver on Windows only).

1. From your clone (any path), run:

   ```bash
   cd /path/to/boiling-sim
   bash scripts/setup_wsl_env.sh
   ```

   The script updates packages, runs `apt-get --fix-broken install` to clear broken/partial installs, installs build dependencies, then installs **CUDA 12.6** from NVIDIA’s **WSL Ubuntu** repo **only if** `nvcc` is missing or older than CUDA 12 (for example, **Lambda’s `nvidia-cuda-toolkit` 12.8** is detected and left alone). It then installs **uv**, a Python 3.11 **`.venv`**, Rust tooling, and Node.js 20.

2. **If `apt` fails with unmet dependencies** (common on Ubuntu 22.04 when security updates are pending), repair and retry:

   ```bash
   sudo apt update
   sudo apt --fix-broken install
   ```

   Accept the proposed upgrades (for example `libssl3`, `libcurl4`, `libfreerdp2-2`). Then run `bash scripts/setup_wsl_env.sh` again.

3. **Benign noise during CUDA install** — You may see `head: cannot open '/etc/ssl/certs/java/cacerts'` while `ca-certificates-java` runs; the postinst usually still completes and registers certificates.

4. **`apt autoremove` suggestions** — After NVIDIA driver changes, apt may list old `libnvidia-*` packages as “no longer required.” Review before running `sudo apt autoremove` so you do not remove packages you still need for graphics or compute.

5. **Activate and install the package**

   ```bash
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

6. **Verify**

   ```bash
   pytest python/tests/
   cargo test --release -p cuda-kernels
   cargo build --release
   ```

## Usage

### Run a scenario

```powershell
boiling-sim-scenario configs/scenarios/default.yaml
```

```bash
boiling-sim-scenario configs/scenarios/default.yaml
```

### Run individual scripts

```powershell
python scripts/run_heating.py
```

```bash
python scripts/run_heating.py
```

## Development phases

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Environment + baseline benchmarks | Complete |
| 1 | Parametric USD scene (pot + water + carrot) | Complete |
| 2 | Single-phase CFD + conjugate heat transfer | In progress |
| 3 | Nucleate boiling + bubble dynamics | Planned |
| 4 | Carrot nutrient retention coupling | Planned |
| 5 | Rust + custom CUDA acceleration | Planned |
| 6 | Live 3D dashboard | Planned |
| 7 | Omniverse Kit migration (optional) | Planned |

## Testing

```powershell
# Python tests
pytest python/tests/

# Rust/CUDA tests
cargo test --release -p cuda-kernels

# Benchmarks
pytest python/tests/ --benchmark-only
```

## Documentation

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — Setup walkthrough for the Lambda Vector workstation
- [`multiphysics_boiling_developer_guide.md`](multiphysics_boiling_developer_guide.md) — Full technical guide with equations, data structures, and implementation steps
- [`benchmarks/`](benchmarks/) — Validation results and performance baselines

## Tech stack

- **GPU compute**: [NVIDIA Warp](https://github.com/NVIDIA/warp) (Python GPU kernels), CUDA 12.6
- **Numerics**: NumPy, SciPy, warp.fem
- **Geometry**: Trimesh, PyGmsh, MeshIO, OpenUSD
- **Visualization**: PyVista, Matplotlib
- **Systems**: Rust (orchestration + PyO3), Tokio + Axum (WebSocket server)
- **Dashboard**: React, React Three Fiber, Three.js
- **Config**: YAML scenarios, Pydantic models

## License

All rights reserved. This is a research project of the SASEL Lab at McGill University.
