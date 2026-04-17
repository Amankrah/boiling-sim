# Phase 0 Baseline Benchmarks

## System Info

| Component | Value |
|-----------|-------|
| Date | 2026-04-16 |
| GPU | NVIDIA RTX 6000 Ada Generation |
| VRAM | 48 GB (49140 MiB) |
| Driver Version | 595.97 |
| CUDA Toolkit (installed) | 12.6, V12.6.20 |
| CUDA Runtime (Warp bundled) | 12.9 |
| Warp Version | 1.12.1 |
| Python Version | 3.11.15 |
| Rust Version | 1.90.0 |
| cudarc Version | 0.12 (cuda-12060 feature) |
| MSVC | VS 2019 BuildTools, MSVC 14.29.30133 |
| Node.js | 22.13.1 |
| pnpm | 10.2.1 |
| OS | Windows 11 Pro 26200.8246 (native, no WSL) |
| CPU | AMD Threadripper PRO 7975WX (64 threads) |
| RAM | 256 GB |

## Warp SPH Example

Command: `python -m warp.examples.core.example_sph`

| Metric | Value |
|--------|-------|
| Step time | ~7.5 ms (range: 6.97 – 8.46) |
| Render time | ~10.4 ms (range: 9.76 – 11.61) |
| Total per frame | ~18 ms |
| Effective FPS | ~55 fps |
| Output | `example_sph.usd` |

**Expected on RTX 6000 Ada (per guide):** 80-120 fps at 10M particles. Default example particle count unverified — render overhead dominates at small N.

## Warp FEM Diffusion Example

Command: `python -m warp.examples.fem.example_diffusion`

| Metric | Value |
|--------|-------|
| Kernel modules compiled | 29 |
| Module compile time | ~7.2 s total (one-time, then cached) |
| CG final error | 5.7e-3 (tol 2.1e-2) |
| CG iterations to converge | 63 |
| Outcome | OK |

## Rust + CUDA (cudarc) — Vector Add

Command: `cargo test --release -p cuda-kernels`

| Metric | Value |
|--------|-------|
| `test_cuda_device_available` | PASS |
| `test_vector_add_compiles` | PASS (build.rs → nvcc → cl.exe → sm_89) |
| `test_round_trip_device_memory` | PASS (1M f32 elements, <5s) |
| Kernel launch benchmark | **DEFERRED TO PHASE 5** |

**Expected on RTX 6000 Ada (per guide):** 900+ GB/s for saturated vector-add (card peak 960 GB/s). Will measure in Phase 5 when cudarc module-launch API is wired up.

## Python Smoke Tests

Command: `pytest python/tests/`

| Test | Result |
|------|--------|
| test_import_boilingsim | PASS |
| test_warp_available | PASS |
| test_warp_kernel_launches | PASS |
| test_materials_json_valid | PASS |
| test_default_config_loads | PASS |
| test_numpy_scipy_available | PASS |
| **Total** | **6/6 passed** |

## Phase 0 Acceptance

- [x] `nvidia-smi` shows RTX 6000 Ada with 48 GB, driver 595.97
- [x] CUDA 12.6 compiles and runs the hello-world kernel
- [x] Warp SPH and FEM examples run; throughput recorded
- [x] Rust program compiles CUDA kernel via nvcc + cl.exe
- [x] `cargo build --release` succeeds for all workspace crates
- [x] `cargo test --release -p cuda-kernels` passes
- [x] `pytest python/tests/` passes 6/6
- [x] Full repo scaffold with materials.json, scenario config, tests

**Phase 0 complete.** Ready to proceed to Phase 1 (Geometry and Parametric Scene).

## Known Limitations / TODO

1. **Vector-add GB/s not measured.** Scaffold rounds-trips memory but defers kernel launch to Phase 5 when cudarc module-loading API is wired.
2. **Warp SPH particle count** not extracted from example — could be queried for precise throughput numbers.
3. **`.wslconfig`** not applied (not using WSL).
4. `cc` crate "Compiler family detection failed" warnings — harmless; caused by nvcc wrapping cl.exe such that the standard C-compiler probe macros aren't defined. Does not affect compilation correctness.
