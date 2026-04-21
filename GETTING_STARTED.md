# Getting Started

The **primary** walkthrough below targets **Windows-native** development on the Lambda Vector workstation (no WSL2 required). If you work on **native Ubuntu** (including the same machine with Linux installed, or Lambda’s Jammy image with `archive.lambdalabs.com` enabled), use **[Native Ubuntu / Lambda (bash)](#native-ubuntu--lambda-bash)** first, then pick up verification steps that apply to both OSes.

## Native Ubuntu / Lambda (bash)

Use this path when `nvcc` comes from **`nvidia-cuda-toolkit`** / **`nvidia-cuda-dev`** (Lambda or Ubuntu repos). Those stacks often report **CUDA 12.8** (or another 12.x), which is **compatible** with this project; documentation elsewhere may still say “12.6” as the reference baseline.

### Check the compiler (not PowerShell syntax)

In **bash**, verify:

```bash
nvcc --version
# Example: Cuda compilation tools, release 12.8, V12.8.93
```

**Common mistake:** `echo $env:CUDA_PATH` is **PowerShell**. In bash it does not read `CUDA_PATH`. Use:

```bash
echo "${CUDA_PATH:-<unset>}"
```

Seeing **`<unset>`** here is **normal** for apt-based CUDA: nothing is wrong. If `which nvcc` works (often `/usr/bin/nvcc`), Python/Warp and the setup script are fine. You only need `CUDA_PATH` if a tool explicitly requires it (Windows builds always should set it; on Linux it is optional).

### Rust `cuda-kernels` (Linux library path)

`crates/cuda-kernels/build.rs` looks for CUDA libraries in this order: **`CUDA_PATH/lib64`** if set, else **`/usr/local/cuda/lib64`**, else **`/usr/lib/cuda/lib64`** (typical for `nvidia-cuda-dev`). So an unset `CUDA_PATH` is usually fine.

If **`cargo build -p cuda-kernels`** still cannot find `libcudart`, install the dev packages or point the build at your toolkit root:

```bash
export CUDA_PATH=/usr/lib/cuda   # or wherever `dpkg -L nvidia-cuda-dev` puts libcudart
```

As a last resort on Ubuntu:

```bash
sudo mkdir -p /usr/local && sudo ln -sfn /usr/lib/cuda /usr/local/cuda
```

### Finish environment setup

From your clone:

```bash
bash scripts/setup_wsl_env.sh
```

The script **does not** install NVIDIA’s separate 12.6 meta-package if **`nvcc` already reports CUDA 12.x or newer** (so it will not fight Lambda’s 12.8). Then:

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest python/tests/
cargo build --release
cargo test --release -p cuda-kernels
```

---

## Windows Native Setup

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

## Dashboard / Phase 6 deployment

The Phase 6 live dashboard ships as three services wired by
`docker-compose.yml`:

- **solver** — Python + Warp, produces msgpack snapshots at 30 Hz.
- **ws-server** — Rust Axum relay (`/stream` WebSocket, TCP ingest on
  8765, TCP control forward on 8766).
- **web** — nginx serving the Vite production build; proxies `/stream`
  to the ws-server container so the browser sees a same-origin URL.

### Local dev (no Docker)

Each service runs standalone in its own terminal. Use the **repository root** as the working directory (there is no `backend/` folder).

**Windows (PowerShell):**

```powershell
# Terminal 1 -- Rust relay
cargo run -p ws-server --release

# Terminal 2 -- Python producer (warm-started default scenario)
python scripts\run_dashboard.py --config configs\scenarios\default.yaml

# Terminal 3 -- Vite dev server
cd web
npm install --include=dev
npm run dev
```

**Linux / macOS (bash):** use **`/`** in paths. Backslashes merge path segments (`scripts\run_...` becomes `scriptsrun_...`).

```bash
# Terminal 1
cargo run -p ws-server --release

# Terminal 2 (venv active: source .venv/bin/activate)
python scripts/run_dashboard.py --config configs/scenarios/default.yaml

# Terminal 3
cd web
npm install --include=dev
npm run dev
```

Open http://localhost:3000 (or the Vite “Network” URL). In dev, the UI
connects to **`ws://<same-host>:8080/stream`** automatically so it matches
`cargo run -p ws-server`, and **`/api/*`** (Results tab artefacts) is proxied
to the same relay on port 8080. Override WebSocket URL with `VITE_WS_URL` in
`web/.env.development` if needed.

### Containerised (single command)

On a GPU host with the NVIDIA Container Toolkit installed:

```bash
docker compose up --build
```

Then open http://localhost:3000 — nginx proxies the WebSocket to the
ws-server container; the solver hits it over the internal
`boiling-net` bridge.

The solver container runs `scripts/dashboard_precheck.sh` before
launching the simulation. If `nvidia-smi` isn't reachable or Warp
fails to see a CUDA device (the classic WSL2 + Docker silent CPU
fallback), the container exits with a clear error instead of
quietly going 100× slower.

### Side-by-side comparison demos

The share-link mechanism encodes scenario parameters + camera pose
(but not simulation time; see the Phase-6 plan non-goals). To run a
side-by-side material comparison in the same browser session, open
two browser windows against the same ws-server:

```
http://localhost:3000/?hf=30000&mat=steel_304&cd=25&cl=50
http://localhost:3000/?hf=30000&mat=copper&cd=25&cl=50
```

Each window drives the same solver via the shared WebSocket; loading
the second URL kicks a `set_material` control message and triggers a
rebuild. A few seconds later both windows are streaming the same
step cadence so you can compare wall temperature and nutrient
retention between pot materials in real time. This is the intended
demo pattern for donor/partner walkthroughs.

