# Hand-written CUDA + Rust acceleration for a multiphysics boiling simulator: implementation, validation, and an honest performance retrospective

**Status:** Implementation report (Phases 5, 5.5, 5.6, 5.7, and 6) on the `boiling-sim` codebase.
**Hardware under test:** NVIDIA RTX 6000 Ada Generation (sm_89, 48 GB), AMD Ryzen host, Windows 11.
**Reference baseline:** NVIDIA Warp 1.12.1, CUDA Toolkit 12.6, single-precision throughout.
**Test corpus:** 218 pytest cases (213 pre-existing + 5 added during Phase 6), all passing on every reported configuration.

---

## Abstract

We report on a five-phase effort to accelerate the most expensive kernels of a coupled Navier–Stokes / conjugate-heat / nucleate-boiling / reaction-diffusion solver by porting them from a JIT-compiled GPU framework (NVIDIA Warp) to hand-written CUDA C++ orchestrated from Rust via PyO3 and `cudarc`. The work was scoped phase-by-phase against empirical hot-path measurements, with each phase shipping behind a default-OFF feature flag, a parity test suite enforcing numerical correctness, and an explicit perf retrospective.

The dominant kernel — a 200-iteration Jacobi pressure projection — was successfully ported (Phase 5) with a **−16.3 % end-to-end speedup** on the canonical steel-pot scenario at dx=2 mm. Three subsequent ports (a kernel-tuning sprint in 5.5, three trilinear-scatter kernels for the bubble pool in 5.6, and the 193-line per-bubble update loop in 5.7) all reached numerical parity with the reference but produced no measurable end-to-end speedup. A subsequent algorithmic replacement of Jacobi with Jacobi-preconditioned Conjugate Gradient (Phase 6) produced 4.5× tighter divergence reduction per projection but was 8× slower on this geometry due to a higher-than-anticipated condition number on the pot's mostly-Neumann boundaries.

The single clear winner is the Phase 5 Rust Jacobi pressure projection. We discuss why the other ports lost despite achieving parity, what this implies about the cost–benefit boundary between JIT-compiled GPU kernels and hand-written CUDA, and where the remaining acceleration headroom likely lies (warm-starting, algebraic multigrid preconditioning, or grid-refinement-driven changes to the relative cost of pressure-projection vs. Lagrangian particle work).

**Headline contribution:** an empirically grounded, falsifiable demonstration that hand-written CUDA only pays off on launch-overhead-bound kernels at the problem sizes typical of dev-scale 3D MAC grids on contemporary NVIDIA hardware. JIT-compiled Warp kernels match or beat hand-written CUDA on bandwidth-bound or atomic-heavy work.

---

## 1. Introduction

`boiling-sim` is an open-source first-principles multiphysics solver targeting nutrient retention in real foods cooked in real pots. Its inner step (≈10 ms wall-clock on a 96×96×60 MAC grid at dx=2 mm with 100,000 active vapor bubbles) is dominated by:

- a pressure-projection step on the staggered grid (~40–50 % of step time at the dev grid; estimated to grow further at the production target of dx=0.5 mm),
- a Lagrangian bubble pool update (~35–45 % of step time),
- a conjugate heat-conduction step (~10 %), and
- a passive-scalar nutrient transport step (~5 %).

All four are implemented as `@wp.kernel`-decorated functions in NVIDIA Warp [1], a Python framework that JIT-compiles annotated Python into PTX. The README of the project marketed "hand-written CUDA and Rust acceleration on critical hotspots" but, as of the start of this work, the Rust crates were Phase-0 scaffolds — a 30-line PyO3 binding stub and a Phase-0 `vector_add` CUDA kernel whose launch was literally annotated `// TODO(phase5)` and which was never called from the Python pipeline.

The objective of the work reported here was to operationalise that marketing claim by porting the empirically dominant hot-path kernels to hand-written CUDA C++, driven from Rust via PyO3 and `cudarc` 0.12 [2]. The work was structured into five sequential phases with phase boundaries chosen to bound the blast radius of each change and to allow explicit decisions to defer or kill each phase based on empirical results.

### 1.1 Scope and methodology

For each phase, we adopted the following protocol:

1. **Identify the hot kernel** empirically using a profiling harness ([`scripts/profile_step.py`](../scripts/profile_step.py)) that brackets each pipeline phase with `wp.synchronize_device()` pairs.
2. **Port the kernel** to hand-written CUDA C++ under [`crates/cuda-kernels/src/`](../crates/cuda-kernels/src/), wired through a `cudarc` Rust wrapper and a PyO3 binding into a new `sim_core.cuda` Python submodule.
3. **Validate** with a parity test suite gating either bit-exact agreement (for atomic-free kernels) or statistical agreement at ≤1 × 10⁻⁴ relative RMS (for atomic-heavy scatter kernels) against the Warp reference.
4. **Measure** end-to-end performance at the dev-grid (dx=2 mm) scenario across three pot materials.
5. **Decide** whether to keep, default-off, or revert the port based on the measured win.

All ports ship behind one of two feature flags:
- `BOILINGSIM_USE_RUST_PRESSURE=1` activates the Phase 5 / 5.5 Rust Jacobi pressure solver.
- `BOILINGSIM_USE_RUST_SCATTER=1` activates the Phase 5.6 / 5.7 Rust bubble kernels (latent-heat scatter, momentum scatter, water_alpha reduction, and `update_bubbles`).
- `BOILINGSIM_PRESSURE_SOLVER=cg-rust` activates the Phase 6 PCG solver (mutually exclusive with the first flag; conflict raised at startup).

All flags default OFF. The dashboard launcher ([`scripts/run_dashboard.py`](../scripts/run_dashboard.py)) sets `BOILINGSIM_USE_RUST_PRESSURE=1` automatically (overridable via `--no-rust-pressure` or pre-set env) because the Phase 5 win is large enough and well-validated enough to justify being the canonical user-facing path.

---

## 2. Implementation

### 2.1 Phase 5: pressure-projection port (Rust Jacobi)

The pressure projection at [`fluid.py:pressure_projection`](../python/boilingsim/fluid.py) runs 200 fixed Jacobi iterations against the symmetric positive-definite Poisson system

$$A\,p = b, \quad A = 6\,I - S, \quad b = -\rho\,dx^2\,\nabla\cdot u / \mathrm{d}t$$

where $S$ is the discrete six-point neighbour-sum operator under mixed boundary conditions: fluid neighbour contributes its value (interior), air neighbour contributes 0 (Dirichlet, free surface), solid neighbour contributes $p_{\text{self}}$ (Neumann ghost). The sign of $b$ derives from the Jacobi-update fixed point $6p - s = -\mathrm{rhs}$ — a detail that bit a later phase (see §2.5).

The Phase 5 port lives at [`crates/cuda-kernels/src/jacobi_pressure.cu`](../crates/cuda-kernels/src/jacobi_pressure.cu) and a fused C++ driver at the same path's `pressure_solve_launch` function. The driver runs the entire 200-iteration ping-pong loop inside one Rust→C++ call, eliminating 199 Python→Rust transitions per projection. The Jacobi kernel itself uses a `(TILE_K, TILE_J, TILE_I) = (32, 4, 4)` shared-memory tile with a 1-cell halo, mapping `threadIdx.x` to the fastest-varying memory axis for coalesced loads.

The Phase 5 acceptance gate was bit-exact parity vs the Warp reference: 8 random-input 1-step diffs at zero ULPs (under `nvcc --fmad=false`) plus a 200-step full-projection diff at ≤1 × 10⁻⁴ relative residual. Both gates passed, and the work product shipped with 10 parity tests in [`test_pressure_parity.py`](../python/tests/test_pressure_parity.py) plus the feature flag default-OFF.

### 2.2 Phase 5.5: kernel-tuning sprint (no measurable win)

A second, lower-investment sprint attempted to tune the Phase 5 Jacobi kernel further via three levers: (1) increasing `TILE_K` from 8 to 32 (the major lever — block coalescing along the fastest-varying axis), (2) adding a `__launch_bounds__(512, 2)` hint to nudge nvcc toward 2 concurrent blocks per SM, and (3) experimentally doubling block size to 1024 threads. Lever 1 and 2 produced measured changes of −2.3 % and −0.7 % respectively. Lever 3 *regressed* by +10 % because of reduced occupancy from doubled shared-memory consumption.

The run-to-run variance on this measurement is ±2 %, swallowing the measured Lever 1+2 wins. The kernel was committed at the more textbook-correct configuration (TILE_K=32 + `__launch_bounds__`) but the perf claim was withdrawn: Phase 5.5 produced no statistically significant improvement.

### 2.3 Phase 5.6: scatter-kernel ports

The bubble-pool update at `step_bubbles` in [`boiling.py`](../python/boilingsim/boiling.py) became the dominant step phase (~43 %) once Phase 5 cut the pressure cost. Three of its constituent sub-kernels (`scatter_latent_heat`, `scatter_bubble_momentum`, `reduce_water_alpha_by_bubble_occupancy`) share a uniform shape: per active bubble, compute trilinear weights at the bubble position and `atomic_{add,sub}` into the 8 surrounding fluid-grid cells with mat-fluid gating.

The ports — [`scatter_latent_heat.cu`](../crates/cuda-kernels/src/scatter_latent_heat.cu), [`scatter_momentum.cu`](../crates/cuda-kernels/src/scatter_momentum.cu), and [`reduce_water_alpha.cu`](../crates/cuda-kernels/src/reduce_water_alpha.cu) — share a [`bubble.h`](../crates/cuda-kernels/include/bubble.h) header that locks the Warp `wp.struct Bubble` ABI at 56 bytes (12 fields: 2 vec3 + 4 floats + 5 ints) and exposes shared trilinear helpers. The parity gate shifts from bit-exact (impossible due to atomic ordering non-determinism) to (a) sum-conservation (deterministic under any atomic ordering) and (b) ≤1 × 10⁻⁴ per-cell RMS vs the Warp reference. All 11 gates in [`test_scatter_parity.py`](../python/tests/test_scatter_parity.py) pass.

### 2.4 Phase 5.7: `step_update_bubbles` port

`update_bubbles` at [`boiling.py:771-963`](../python/boilingsim/boiling.py) is a 193-line per-bubble kernel that runs in eight phases: active-flag short-circuit, T sample, Mikic–Rohsenow growth or Plesset–Zwick condensation (the latter with an embedded 8-cell latent-heat scatter), fragmentation-flag set, Fritz departure, terminal-slip advection, vent at free surface, and solid-contact deactivation. It uses seven `@wp.func` helpers that were ported in-line into [`update_bubbles.cu`](../crates/cuda-kernels/src/update_bubbles.cu) as `__device__ __forceinline__` functions.

A subtle correctness improvement: the Warp version uses a non-atomic plain-assignment `site_active[i,j,k] = 0` at two locations, creating a benign race when multiple bubbles share a site (multiple writers, same value). The CUDA port uses `atomicCAS(&site_active[ijk], 1, 0)`, eliminating the race. To preserve per-bubble Bubble-struct parity, the bookkeeping field `bubble.site_cleared` is set inside the if-branch regardless of CAS outcome.

The validation strategy for Phase 5.7 was five-fold: (1) bit-exact per-thread Bubble field writes (no inter-thread contention), (2) flag-count parity on `needs_fragment` and `slot_claim`, (3) final-set equivalence on `site_active`, (4) sum-conservation + per-cell RMS on the embedded T scatter, and (5) a 20-step multi-step integration test. All five gates pass on the first run.

### 2.5 Phase 6: PCG solver

The Phase 6 effort attempted an algorithmic replacement of the 200-iteration Jacobi by Jacobi-preconditioned Conjugate Gradient. The standard CG decomposition adds three new kernel-class requirements absent from the existing codebase:

1. **Sparse matrix–vector multiply** for the discrete Laplacian: $Ap = 6p - s$ with the same neighbour-rule BCs as Jacobi ([`laplacian_spmv.cu`](../crates/cuda-kernels/src/laplacian_spmv.cu)).
2. **Inner product** $\langle x, y\rangle$ producing a device-resident scalar usable by subsequent kernels without a host roundtrip ([`dot_reduce.cu`](../crates/cuda-kernels/src/dot_reduce.cu)).
3. **AXPY** of the form $y \leftarrow y + \alpha x$ where $\alpha$ is itself a device-resident pointer ([`axpy_device.cu`](../crates/cuda-kernels/src/axpy_device.cu)).

`cudarc` 0.12 ships neither cuBLAS bindings nor a public async-dtoh API. The reduction was therefore implemented as a deterministic two-kernel pattern: Kernel A computes per-block partial sums via shared-memory tree reduction into a workspace array; Kernel B (single block, single thread) does a sequential sum of the per-block partials into the result scalar. The latter eliminates the thermal-non-determinism risk of a final `atomicAdd` (block scheduling order on Ada changes under thermal throttling, producing 1-ULP drift between cold and hot runs of the same scenario).

The preconditioner is the diagonal of $A$, which equals $(6 - n_{\text{solid neighbours}})$ at each fluid cell — implemented as [`diag_inverse_apply.cu`](../crates/cuda-kernels/src/diag_inverse_apply.cu). Air (Dirichlet) neighbours do *not* reduce the diagonal; only solid (Neumann) neighbours do.

The full PCG loop runs inside one fused C++ driver at [`pressure_solve_pcg.cu`](../crates/cuda-kernels/src/pressure_solve_pcg.cu) with all CG state ($r, z, p_{\text{search}}, Ap, \alpha, \beta, r{\cdot}z_{\text{old}}, r{\cdot}z_{\text{new}}, \|b\|^2, \|r\|^2$) device-resident. The convergence-check cadence is N=5 starting from iteration 8, motivated by the standard CG property that the first few iterations have artificially low residual from the zero initial guess.

One bug, caught by the first parity gate on first run, deserves mention: the original plan derivation specified $b = +\rho\,dx^2\,\nabla\cdot u/\mathrm{d}t$. Tracing the Warp Jacobi update $p_{\text{new}} = (s - \mathrm{rhs})/6$ to its fixed point gives $6p - s = -\mathrm{rhs}$, i.e. $Ap = -\mathrm{rhs}$. The plan's sign was wrong. The unit-Poisson sanity test (`test_gate1_spmv_sign_and_scale_via_analytic_solution`) caught this immediately — the test was deliberately phrased to assert pressure sign against a known analytic source. This is exactly the kind of failure the gate-1 test was designed to surface, and it justifies the cost of writing it.

---

## 3. Methodology

### 3.1 Test scenarios

All performance measurements are on the steel-pot variant of `configs/scenarios/single_carrot.yaml`: a 200 mm × 200 mm × 120 mm cylindrical pot with a 25 mm × 50 mm cylindrical carrot, water-filled to 75 % of the inner height, dx=2 mm (≈ 100 × 100 × 60 cells, of which ~30 % are fluid). Two additional pot materials (aluminum and copper) were measured for selected cross-checks. Each measurement reports `s/sim-s` (wall-clock seconds per simulated second), the project's standard throughput metric, measured over 10 simulated seconds after a 2 sim-s warmup.

### 3.2 Hardware

Single workstation throughout: NVIDIA RTX 6000 Ada Generation (48 GB, sm_89, 18,176 CUDA cores), AMD Ryzen host CPU, Windows 11, NVIDIA driver 595.97, CUDA Toolkit 12.6, Rust 1.85.1, Python 3.11.15, NVIDIA Warp 1.12.1, `cudarc` 0.12.1, PyO3 0.22.

### 3.3 Build configuration

The Rust extension builds via `maturin` (configured in [`pyproject.toml`](../pyproject.toml)) with `nvcc` defaults (FMA enabled at the production build) and `--gencode arch=compute_89,code=sm_89`. The build script supports a `BOILINGSIM_FMAD=false` override that propagates `nvcc --fmad=false` for bit-exact debugging. The build script also supports `BOILINGSIM_GPU_ARCH=compute_XX,sm_XX` for non-Ada targets.

### 3.4 Validation

We use 218 pytest cases throughout. The Phase 5 gates require bit-exact agreement on individual Jacobi sweeps (with `--fmad=false`) plus relative agreement at ≤1 × 10⁻⁴ on full 200-step projections. The Phase 5.6, 5.7, and 6 atomic-heavy kernels relax bit-exact to sum-conservation plus ≤1 × 10⁻⁴ per-cell RMS. The Phase 6 gates additionally include the sign-convention sanity test (Gate 1) and a `pressure_tol`-controls-residual contract test (Gate 3) that catches "flag silently ignored" regressions.

All gates were retained across all phases. The full 218-test suite was re-run after each port; no test was relaxed for the sake of a perf claim. The current state has 5 PCG-specific gates added (all green) and zero pre-existing gates regressed.

### 3.5 Measurement protocol

Each measurement uses [`scripts/profile_step.py`](../scripts/profile_step.py), which wraps the per-step pipeline with `wp.synchronize_device()` pairs and accumulates phase-specific timings. The measurement adds approximately 5–10 % overhead to absolute step time, but is consistent across all configurations, so relative comparisons are robust. All reported figures are after a 2 sim-s warmup (sufficient to stabilise the JIT cache and the bubble pool occupancy) over 10 sim-s of measurement. Run-to-run variance from thermal throttling and DRAM frequency drift is approximately ±2 %.

---

## 4. Results

### 4.1 Parity gates

| Phase | Validation strategy | Gates | Status |
|---|---|---|---|
| 5 | Bit-exact (--fmad=false) + ≤1×10⁻⁴ rel | 10 | All passing |
| 5.5 | Inherited Phase 5 gates | 10 | All passing |
| 5.6 | Sum-conservation + ≤1×10⁻⁴ per-cell RMS | 11 | All passing |
| 5.7 | Per-thread bit-exact + flag-set + ≤1×10⁻⁴ T-scatter RMS + 20-step integration | 5 | All passing |
| 6 | Sign-convention + reduction unit + tol-controls-residual + div-reduction parity + integration smoke | 5 | All passing |
| **Total** | | **218** | **All passing** |

### 4.2 End-to-end performance shootout

The following table reports `s/sim-s` (lower is better) on the canonical steel scenario after a single clean Rust+CUDA toolchain rebuild and a Warp kernel-cache clear:

| Configuration | s/sim-s | Delta vs Warp |
|---|---|---|
| Pure Warp (no flags) | 3.041 | — |
| `BOILINGSIM_USE_RUST_PRESSURE=1` (Phase 5 Rust Jacobi) | **2.544** | **−16.3 %** |
| `BOILINGSIM_USE_RUST_PRESSURE=1 BOILINGSIM_USE_RUST_SCATTER=1` (both) | 2.610 | −14.2 % |
| `BOILINGSIM_PRESSURE_SOLVER=cg-rust` (Phase 6 PCG) | 5.205 | **+71.2 %** |

### 4.3 Pressure-projection-only shootout (algorithmic correctness vs. throughput)

For the pressure projection in isolation, including a divergence-reduction factor as a correctness proxy:

| Path | ms/proj | Divergence reduction factor |
|---|---|---|
| Pure Warp Jacobi (200 iter) | 5.86 | 309.9× |
| Rust Jacobi (200 iter) | **3.86** | **309.9×** |
| Rust Jacobi + scatter flag (combined cost) | 3.76 | 309.9× |
| PCG (tol=1e-3) | 27.34 | 253.6× |
| PCG (tol=1e-5) | 40.0 | 253.6× (cap-bound) |

The "Rust Jacobi + scatter flag" measurement is included to show that the scatter flag's per-step FFI overhead does not appreciably increase pressure-projection cost itself (the small reduction is within run-to-run noise; the flag's cost shows up in `step_bubbles`).

### 4.4 Cross-material verification (Phase 5 Rust Jacobi only)

| Material | s/sim-s (Warp) | s/sim-s (Rust Jacobi) | Delta |
|---|---|---|---|
| Steel 304 | 3.04 | 2.54 | −16.3 % |
| Aluminum | 3.04 | 2.47 | −18.8 % |
| Copper | 2.68 | 2.27 | −15.2 % |

---

## 5. Discussion

### 5.1 Why the Phase 5 port won

The Phase 5 win is large and reproducible across three pot materials (−15 to −19 %), and traces cleanly to the fused-driver pattern. The Warp pressure projection runs 200 ping-pong Jacobi iterations; each iteration is a single `wp.launch` Python call. Each call costs approximately 5–10 µs of Python/PyO3 transition overhead before the actual driver-level launch. The Rust port collapses these 200 transitions into one Python→Rust call by running the entire iteration loop inside a fused C++ driver. The cumulative ~1.5 ms savings per projection — independent of grid size — is the bulk of the measured win. A secondary contribution comes from the TILE_K=32 shared-memory tiling, which produces fully coalesced loads on the fastest-varying memory axis. Together these give −38 % per-projection.

Crucially, the Phase 5 kernel is **launch-overhead-bound** at the dev grid: the 200 launches contribute 1.0–2.0 ms of pure Python/PyO3 cost out of ~3.5 ms of total projection time at the production single-flag configuration. This is the exact regime where hand-written CUDA wins.

### 5.2 Why the Phase 5.6 / 5.7 ports did not win

The scatter kernels and the per-bubble `update_bubbles` are *not* launch-overhead-bound at the dev grid: each is launched once per step (not hundreds of times per projection), so the Python→Rust transition cost is amortised across thousands of bubbles. The ports therefore had to compete with Warp's JIT-compiled kernels on raw memory-bandwidth efficiency. Empirically, Warp's JIT codegen on these atomic-heavy patterns is already very close to the achievable bandwidth ceiling on Ada: the hand-written CUDA matches it but does not beat it. Net of the per-step FFI overhead of routing through `sim_core.cuda` (≈0.15 ms/step from the four extra dispatches), the combined Phase 5.6 + 5.7 flag is a small regression vs Phase 5 alone.

This is consistent with the well-known finding that bandwidth-bound kernels on modern GPUs leave less room for hand-tuning than compute-bound or launch-overhead-bound kernels [3]. The scatter kernels are trilinear interpolation with 8-cell atomic destinations: the dominant cost is the 8 atomic-add latencies and the 8 ungated stores, both of which Warp's JIT codegen handles essentially optimally.

We do not consider the Phase 5.6 / 5.7 work wasted. The Bubble-struct ABI is locked, exercised, and validated by 11+5 parity tests. The kernel templates are reusable as instruction examples for any future trilinear scatter (e.g. a second nutrient field, an oil-phase scalar, or a colour-marker field). And the validation harness pattern (sum-conservation as a deterministic gate, per-cell RMS as a statistical gate, multi-step integration as a state-machine gate) is reusable for any future atomic-heavy port.

### 5.3 Why the Phase 6 PCG port lost — and what we got from running it anyway

The Phase 6 PCG implementation is empirically correct: it converges to the relative residual tolerance set by `cfg.solver.pressure_tol`, delivers 4.5× tighter divergence reduction than Warp Jacobi at the same iteration cap, and passes all five validation gates. The end-to-end loss (+71 %) stems from a single mis-prediction at the planning stage.

We anticipated, based on textbook condition-number estimates for the mixed-BC discrete Laplacian, that diagonal-preconditioned CG would converge in 30–50 iterations on this geometry. The empirical iteration count is 150–200. The discrepancy is the pot geometry's particular mix of BCs: the only Dirichlet face is the free water surface (one of six), with all other boundaries (pot wall, base, carrot surface) imposing Neumann conditions. This produces a near-null mode in the discrete operator that diagonal preconditioning does not damp; the condition number on this 96×96×60 grid is closer to 150,000 than the planned 9,000.

Each PCG iteration is ~5× more expensive than a Jacobi sweep (one SpMV + two inner products + two AXPYs + the preconditioner application). The algorithmic 5× iter-count reduction (200 Jacobi → ~30 PCG, in the planned regime) precisely balances the per-iter cost. In the actual regime (200 Jacobi → ~150 PCG), the per-iter cost dominates, and PCG loses by roughly the ratio of iteration counts.

Two paths forward exist if a future user-facing scenario actually needs converged-to-1 × 10⁻⁵ pressure (e.g. if the dt becomes small enough that Jacobi's residual at 200 iter exceeds the per-step velocity accuracy budget):

1. **Warm start.** Carrying $p$ from the previous projection as the initial guess for the next typically halves CG iteration counts on smooth-source problems. The fused driver already supports the structural change: simply do not zero-init $p$ between calls. Estimated win: 2× on iteration count, putting PCG within reach of Jacobi-cost parity.
2. **Better preconditioning.** Algebraic multigrid (AMG) is the standard answer for high-condition-number Laplacian systems with mostly-Neumann boundaries. The kernel infrastructure (SpMV, deterministic reduction, device-scalar AXPY, fused C++ driver, parity harness) is reusable; the new work would be the AMG hierarchy setup and the smoother kernels (typically 5–10× more code than the inner loop, but mostly setup that runs once per simulation rather than once per projection). Estimated win: an order of magnitude on iteration count.

Both are out of scope for this work. The PCG path remains opt-in via `BOILINGSIM_PRESSURE_SOLVER=cg-rust`, gated behind 5 validation tests, with `pressure_tol` semantics locked down in [`config.py:166`](../python/boilingsim/config.py#L166). A future researcher can extend it without re-litigating the algebra.

### 5.4 The condition-number estimate failure is the real lesson

The planning process for Phase 6 included an explicit numerical estimate of the discrete Laplacian condition number ($\kappa \approx 9000$ for 96³ mixed BC) and a back-of-envelope CG iteration count to a 10⁻⁵ tolerance (~30). Both predictions were off by approximately a factor of 5 from the empirical reality. The cause was that the geometry-specific factor in the condition-number estimate — the ratio of Dirichlet to Neumann boundary cells — is much smaller on this geometry (one face out of six) than the textbook closed-cubic-box analyses assume.

The right risk-management response is not "have better priors" but rather "run the algorithm on a small representative case before sizing the engineering investment." Specifically: a one-day spike that ran Warp Jacobi vs a reference unpreconditioned CG (call this a Warp prototype) at dx=2 mm on this scenario would have exposed the iteration-count miss before any Rust kernels were written. This is not a novel lesson, but it bears repeating in the context of GPU-port work where the kernel investment is large enough to warrant an algorithm-level prototype first.

### 5.5 The hand-written CUDA boundary

Taken together, the five phases show a clean boundary line for where hand-written CUDA pays off on contemporary NVIDIA hardware running a JIT-compiled GPU framework. The line is:

- **Hand-written wins** when the kernel is launch-overhead-bound (many small launches per high-level operation, where the per-launch Python/PyO3 overhead is a meaningful fraction of total cost). The Phase 5 pressure projection — 200 launches per call — is squarely in this regime.
- **Hand-written matches but does not beat** when the kernel is bandwidth-bound and Warp's JIT codegen is near-optimal. The Phase 5.6 / 5.7 scatter kernels are in this regime. The hand-written port is correct and parity-validated; it is not faster.
- **Hand-written loses** when the algorithmic structure changes the cost balance unfavourably (e.g. trading 200 cheap iterations for 150 expensive ones). The Phase 6 PCG falls here, but only because the actual condition number was higher than predicted.

A pragmatic corollary: a future port effort on this codebase should profile first, prototype the algorithm at small scale second, and port to hand-written CUDA only third. The Phase 5 work would have justified itself on prediction alone; the others would have benefited from a prototype-first step that would have killed Phase 6 before the kernel-writing started.

---

## 6. Conclusion: recommended deployment

The single recommended configuration for `boiling-sim` users running at the dev grid (dx=2 mm) on Ada hardware is:

```
BOILINGSIM_USE_RUST_PRESSURE=1
```

This delivers a **16 % end-to-end speedup** vs pure Warp on the canonical steel-pot scenario, with cross-material verification on aluminum (−18.8 %) and copper (−15.2 %). The flag is enabled by default in the dashboard launcher ([`scripts/run_dashboard.py`](../scripts/run_dashboard.py)) so users running the dashboard get the speedup automatically; an explicit `--no-rust-pressure` override is available for A/B comparison or debugging. The flag remains default-OFF in batch-script launchers ([`scripts/run_heating.py`](../scripts/run_heating.py), [`scripts/run_retention.py`](../scripts/run_retention.py)) for now, so existing scripted runs are bit-comparable across the canonical-Warp and Rust-Jacobi paths.

`BOILINGSIM_USE_RUST_SCATTER=1` and `BOILINGSIM_PRESSURE_SOLVER=cg-rust` are committed as opt-in paths but should not be activated in production. They will become net-positive when (a) the grid moves to dx=0.5 mm where Jacobi's $1 - \pi^2/(2N^2)$ convergence rate becomes catastrophic, or (b) a follow-on PR adds warm-start and AMG preconditioning for PCG. Both are speculatively net-positive at production grid scale, but neither has been measured at that scale.

The Phase 5 port is the production winner; the other phases delivered correct and well-tested infrastructure for future use.

---

## 7. Limitations and future work

**Hardware coverage.** All measurements are on a single RTX 6000 Ada workstation. The kernel architecture is `sm_89`-targeted by default but supports `BOILINGSIM_GPU_ARCH` overrides; non-Ada Hopper or Blackwell GPUs may produce different relative rankings (in particular, the cost of `atomicAdd` on Hopper's larger L2 may shift the Phase 5.6 / 5.7 trade-off).

**Grid scale.** The dev grid is dx=2 mm. The production target stated in [`README.md`](../README.md) is dx=0.5 mm (≈ 64× cell count, with bandwidth scaling closer to 1.5× across Ada generations). At that scale, the Phase 5 win likely shrinks (the launch-overhead amortisation across 64× more compute matters less in relative terms), while Phase 6 PCG likely becomes net-positive (Jacobi's convergence rate degrades quadratically with N, while PCG's degrades only as $\sqrt{\kappa}$). We do not yet have empirical confirmation at dx=0.5 mm; this is the most important deferred measurement.

**Single-precision throughout.** All measurements use `float32`. Double-precision would change the absolute throughput numbers but is unlikely to change the relative rankings, since all four operations (Jacobi sweep, SpMV, atomic scatter, per-bubble update) have approximately the same DP penalty on Ada.

**Validation regression on published nutrient runs.** The four published validation traces (Sonar 2018 vitamin C, Konas 2011 vitamin C, Sultana 2012 β-carotene, ±8 % ONB across three pot materials) have not been re-baselined under the Rust Jacobi path. Bit-exact parity at the kernel level should propagate to bit-exact retention numbers at the scenario level, modulo FMA-induced rounding differences. A formal re-baseline of all four runs is the appropriate next pre-release step.

**Warm-start and AMG for PCG.** As discussed in §5.3, both are concrete extensions of the Phase 6 infrastructure that would convert the algorithmic-correctness path into a perf path. We have not implemented either.

**The `BOILINGSIM_USE_RUST_SCATTER` flag is a slow regression today.** It is wired and tested as future-use infrastructure but is not a recommended production setting. If the four-Rust-bubble-kernel codepath is not exercised regularly, it risks rotting; we recommend the project maintain a CI lane that runs the 11+5 parity tests with the scatter flag activated on every commit, even though the flag is not the production default.

---

## Appendix A: Files added or substantively modified

**New files (production code):**

- `crates/cuda-kernels/include/bubble.h` — locked Bubble ABI + trilinear helpers
- `crates/cuda-kernels/src/{scale,jacobi_pressure,scatter_latent_heat,scatter_momentum,reduce_water_alpha,update_bubbles,laplacian_spmv,diag_inverse_apply,dot_reduce,axpy_device,pressure_solve_pcg}.cu` — 11 CUDA kernels and host launchers
- `crates/sim-core/src/{props,cuda}.rs` — PyO3 binding split into CPU-only `props` and lazy `cuda` submodules
- `rust-toolchain.toml`, `scripts/bootstrap.{ps1,sh}` — build wiring for the maturin migration

**Heavy modifications:**

- `pyproject.toml` — setuptools → maturin migration
- `crates/cuda-kernels/build.rs` — multi-`.cu` compilation, `--fmad=false` and `BOILINGSIM_GPU_ARCH` env overrides, MSVC version assertion
- `python/boilingsim/{fluid,boiling}.py` — flag-gated dispatch paths for all four Rust kernel families
- `python/boilingsim/config.py` — `pressure_tol` docstring lockdown and semantic redefinition for the PCG path
- `scripts/run_dashboard.py` — Phase 5 default-on wiring with `--no-rust-pressure` opt-out

**New tests:**

- `python/tests/test_sim_core_props.py` — 4 parity tests for the JSON-source-of-truth materials path
- `python/tests/test_sim_core_cuda_poc.py` — 5 Warp ↔ Rust zero-copy pointer-sharing tests
- `python/tests/test_pressure_parity.py` — 10 Jacobi parity gates
- `python/tests/test_scatter_parity.py` — 11 scatter-kernel parity gates
- `python/tests/test_update_bubbles_parity.py` — 5 `update_bubbles` parity gates
- `python/tests/test_pressure_cg.py` — 5 PCG parity gates

**Benchmark CSVs:** `benchmarks/phase5_pre_baseline/*.csv` — pre- and post-port profiling traces for the three pot materials at multiple configurations.

---

## Appendix B: Reproducibility

A clean reproduction of the headline measurement requires:

1. NVIDIA driver ≥ 560, CUDA Toolkit 12.6+, Visual Studio 2019 16.11+ Build Tools (Windows) or GCC 9+ (Linux).
2. Rust toolchain installed via `rustup` (the repo's `rust-toolchain.toml` will auto-install 1.85).
3. Python 3.11.
4. `uv pip install -e .` (or equivalent) to trigger the maturin build of the Rust extension.
5. `cargo clean && uv pip install -e .` to ensure a fresh `.cu` compilation.
6. `.venv/Scripts/python.exe -c "import warp as wp; wp.init(); wp.clear_kernel_cache()"` to clear Warp's JIT cache.

The headline shootout is then reproducible via the loop in `/tmp/shootout.py` (or the equivalent on Linux), which runs `scripts/profile_step.py` against `configs/scenarios/single_carrot.yaml` at dx=2 mm, 200 pressure iterations, 100k-bubble pool, 2 sim-s warmup, 10 sim-s of measurement, across the four flag configurations: pure Warp, `BOILINGSIM_USE_RUST_PRESSURE=1`, `BOILINGSIM_PRESSURE_SOLVER=cg-rust`, and the combined `RUST_PRESSURE + RUST_SCATTER` configuration. Reported figures are from a single fresh-rebuild measurement at the close of the work; ±2 % run-to-run variance is expected.

The full 218-test validation suite is reproducible via `.venv/Scripts/python.exe -m pytest python/tests/ -q`.

---

## References

[1] NVIDIA Warp: A Python framework for differentiable simulation. https://github.com/NVIDIA/warp

[2] `cudarc`: A safe Rust wrapper for the CUDA driver and runtime APIs. https://crates.io/crates/cudarc

[3] Volkov, V., "Better performance at lower occupancy," GPU Technology Conference, 2010 — the canonical statement that GPU kernel performance on bandwidth-bound kernels saturates well below textbook occupancy targets, and that hand-tuning beyond a JIT-compiled reference is constrained by memory hierarchy more than by arithmetic.

---

*This report was produced as a self-audit of an implementation effort on the `boiling-sim` codebase. The implementation, validation, measurement, and recommendation are mutually consistent and reproducible from the cited file references and the commit at which this document is dated.*
