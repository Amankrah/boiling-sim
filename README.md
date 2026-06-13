# Boiling Sim

**GPU-accelerated multiphysics 3D boiling simulation** with coupled thermal, fluid, and nutrient retention modeling.

A carrot-boiling case study built with NVIDIA Warp, CUDA, and Rust — developed at the SASEL Lab, McGill University.

## Overview

This project simulates the full physics of boiling a carrot in a steel pot: natural convection in water, conjugate heat transfer through the pot wall, nucleate bubble dynamics, and nutrient degradation inside the carrot. The simulation runs on GPU via NVIDIA Warp kernels, with a hand-written CUDA + Rust acceleration path for the pressure-projection hotspot ([Phase 5](#phase-5-rust-acceleration)), and streams results to a live 3D web dashboard.

### Key capabilities

- **Conjugate heat transfer** — Coupled solid (pot wall) and fluid (water) thermal solve on a staggered MAC grid with implicit diffusion
- **Natural convection** — Boussinesq-approximated Navier-Stokes with semi-Lagrangian advection and pressure projection
- **Nucleate boiling** — Onset-of-nucleate-boiling detection, bubble nucleation, growth, and departure via Lagrangian particle tracking
- **Nutrient retention** — First-order Arrhenius degradation kinetics for beta-carotene and vitamin C inside a tetrahedral FE carrot mesh
- **GPU acceleration** — Warp kernels for the full pipeline plus an opt-in hand-written CUDA kernel for pressure projection (`BOILINGSIM_USE_RUST_PRESSURE=1`) — measured **−38 % per-projection / −21 % end-to-end** vs Warp baseline on RTX 6000 Ada
- **Live 3D dashboard** — WebSocket-streamed visualization with React and React Three Fiber

## Phase 5: Rust acceleration

The pressure Poisson solver ships in two flavors:

| Path | Default | Where it lives | Status |
| --- | --- | --- | --- |
| Warp JIT (Python) | yes | [`fluid.py:jacobi_pressure_step`](python/boilingsim/fluid.py) | canonical reference, always available |
| Hand-written CUDA via Rust | opt-in (`BOILINGSIM_USE_RUST_PRESSURE=1`) | [`crates/cuda-kernels/src/jacobi_pressure.cu`](crates/cuda-kernels/src/jacobi_pressure.cu) | shared-memory tiled, coalesced, FMA-on |

Parity is validated by [`test_pressure_parity.py`](python/tests/test_pressure_parity.py): 1-step diff at rtol=1e-5 over 8 random configurations, plus a full 200-iteration projection on a realistic pot scenario at max_rel_diff < 1e-4. Bit-exact mode is available for debugging: rebuild with `BOILINGSIM_FMAD=false uv pip install -e .` to force `--fmad=false` on the CUDA build.

Measured perf on RTX 6000 Ada, dx=2mm, 200 pressure iterations (see [`benchmarks/phase5_pre_baseline/`](benchmarks/phase5_pre_baseline/)):

| Pot material | s/sim-s (Warp) | s/sim-s (Rust) | End-to-end | pressure_projection |
| --- | --- | --- | --- | --- |
| Steel 304 | 2.99 | 2.37 | **−20.8 %** | **−37.8 %** |
| Aluminum | 3.04 | 2.47 | **−18.9 %** | **−36.9 %** |
| Copper | 2.68 | 2.27 | **−15.4 %** | **−36.2 %** |

The Rust path is opt-in for now: it requires the maturin build to succeed (`uv pip install -e .` after `scripts/bootstrap.ps1` / `scripts/bootstrap.sh`) and a CUDA driver matching cudarc's `cuda-12060` feature. To force the JIT-compiled Warp path even when the Rust extension is built, leave `BOILINGSIM_USE_RUST_PRESSURE` unset (default).

GPU arch is configurable via `BOILINGSIM_GPU_ARCH` at build time (default `compute_89,sm_89` for RTX 6000 Ada). The `cuda-kernels` build script reads this env var when nvcc compiles the .cu files; non-Ada GPUs need a matching arch override.

### Phase 5.5: Jacobi kernel tuning sprint

Three levers tried — `TILE_K` 8 → 32 (warp-coalesced k-axis), `__launch_bounds__(512, 2)` for occupancy hint, `TILE_I` 4 → 8 for 1024-thread blocks. Honest result: **no statistically-significant per-projection improvement** above the M3 baseline. Run-to-run variance is ±2 % and the measured deltas (−2.3 % from Lever 1, −0.7 % from Lever 2, +10 % regression from Lever 3) fall inside that band. The code kept the TILE_K=32 + `__launch_bounds__` config as the more textbook-correct shape even without measurable perf delta. Re-evaluate at the production dx=0.5mm grid where the kernel's larger working set may surface different bottlenecks.

### Phase 5.6: Scatter kernels (latent_heat, momentum, water_alpha)

Three additional scatter kernels ported as hand-written CUDA + Rust:

- [`crates/cuda-kernels/src/scatter_latent_heat.cu`](crates/cuda-kernels/src/scatter_latent_heat.cu) — trilinear 8-cell energy sink (atomic_add with negative deposits)
- [`crates/cuda-kernels/src/scatter_momentum.cu`](crates/cuda-kernels/src/scatter_momentum.cu) — trilinear 8-face buoyancy injection onto z-faces
- [`crates/cuda-kernels/src/reduce_water_alpha.cu`](crates/cuda-kernels/src/reduce_water_alpha.cu) — VOF alpha reduction + clamp finalizer

Bubble struct ABI matches Warp's `wp.struct Bubble` (56 bytes, validated by [`python/tests/test_scatter_parity.py`](python/tests/test_scatter_parity.py)) — 11 parity tests covering ABI smoke, sum-conservation, per-cell parity at <1e-5 max relative diff, and the BC short-circuits all pass. Gated behind `BOILINGSIM_USE_RUST_SCATTER=1`.

**Default OFF, with reason.** Empirical perf on the dev grid (10 sim-s, dx=2mm, 100k bubbles, steel pot):

| Path | s/sim-s | Delta vs Warp |
| --- | --- | --- |
| Pure Warp baseline | 2.91 | — |
| `BOILINGSIM_USE_RUST_PRESSURE=1` only | 2.46 | **−15.6 %** |
| Both flags ON | 2.59 | **−10.9 %** |

The scatter ports are at kernel-level parity with Warp but add ~0.15 ms/step of Python→Rust FFI overhead (three extra dispatches per step plus CAI lookups), which outweighs the kernel-level savings at 10k active bubbles. Turning the flag ON is a net regression versus pressure-only today.

### Phase 5.7: `step_update_bubbles` port

The dominant per-bubble Warp kernel (193 lines, eight phases: Mikic-Rohsenow growth, Plesset-Zwick condensation with embedded latent-heat scatter, Fritz departure, terminal-slip advection, vent/solid deactivation) ported to [`crates/cuda-kernels/src/update_bubbles.cu`](crates/cuda-kernels/src/update_bubbles.cu). Wired behind the same `BOILINGSIM_USE_RUST_SCATTER=1` env flag, so one toggle turns all four Rust bubble kernels on together.

Parity validated by [`python/tests/test_update_bubbles_parity.py`](python/tests/test_update_bubbles_parity.py) — five gates, all green:

1. Per-thread Bubble field bit-exact (radius / position / velocity / departure_radius at rtol=1e-6; active / site_cleared integer-exact).
2. `needs_fragment` + `slot_claim` flag counts match exactly.
3. `site_active` final-set equivalence (Rust uses `atomicCAS` to fix the Warp race; the set of cleared sites is identical).
4. T-scatter sum-conservation and per-cell RMS within float-precision noise.
5. 20-step multi-step integration smoke (active-set + total-T-integral parity).

**Honest perf with all flags ON** (`BOILINGSIM_USE_RUST_PRESSURE=1 BOILINGSIM_USE_RUST_SCATTER=1`):

| Material | s/sim-s (Warp) | s/sim-s (all flags) | Delta vs Warp |
| --- | --- | --- | --- |
| Steel 304 | 2.91 | 2.57 | **−11.7 %** |
| Aluminum | 3.04 | 2.50 | **−17.9 %** |
| Copper | 2.68 | 2.44 | **−8.9 %** |

The plan hypothesised that porting `update_bubbles` would amortise the M5.6 FFI overhead and deliver a 30-40 % kernel-level win on step_bubbles. The hypothesis was empirically wrong on both counts: the Warp JIT was already efficient on the atomic-heavy bubble kernels, so the hand-written CUDA `update_bubbles` lands roughly at parity (step_bubbles drops only ~0.1-0.2 ms vs M5.6-only). The combined `BOILINGSIM_USE_RUST_SCATTER` flag therefore remains a small regression vs `BOILINGSIM_USE_RUST_PRESSURE` alone — useful for code-coverage / debugging via a fully-Rust path but not a perf win.

Pressure-only stays the canonical fast-path: it's the cleanest single-flag win (-15.6 % steel, no scatter-side FFI overhead) and the only Rust path with a measurable end-to-end perf delta. The four bubble kernels are kept on for future work (Phase 6 CG solver would need them already wired, and the validation harness is reusable). See [`refactored-swimming-boot.md`](../../.claude/plans/refactored-swimming-boot.md) for the full Phase 5.7 plan and the honest perf retrospective.

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
