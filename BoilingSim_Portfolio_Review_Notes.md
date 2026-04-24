# BoilingSim Publication Portfolio — Review Notes

Code-grounded review of [BoilingSim_Publication_Portfolio_v2.md](BoilingSim_Publication_Portfolio_v2.md). All claims in the portfolio were checked against the actual code, tests, and benchmark artefacts before this note was written. The review is structured as **corrections** (claims that should be changed before submission) and **strengthening opportunities** (things the portfolio doesn't highlight that would land with reviewers). Appendix A/B explain the domain-specific terms an external reviewer will meet in the code and terminal output, so they can read your figures without a dictionary open.

---

## 1. Corrections needed before submission

These are specific factual claims that don't match the current repository state. Fix them before a reviewer pulls the repo and spots the gap.

### 1.1 Test counts are understated

The portfolio says **"134 / 20 / 31 = 185 tests"** across Python / Rust / vitest. The actual counts as of today:

| Stack | Portfolio claim | Actual | Source of truth |
|---|---:|---:|---|
| Python (pytest) | 134 | **134** ✓ | `pytest --collect-only -q` |
| Rust (cargo test --workspace) | 20 | **33** | per-binary summary lines |
| TypeScript (vitest) | 31 | **31** ✓ | `vitest run` |
| **Grand total** | **185** | **198** | |

**Action:** update the executive-summary number to 198 / 198, and update Paper 3 §Validation evidence table similarly. The undercount on Rust is material (13 extra tests hidden). You want reviewers seeing 33 because it covers the cross-stack integration tests (`ws_roundtrip`, `python_snapshot`, `runs_endpoints`) which are the strongest evidence for the schema-discipline claim.

### 1.2 Python codebase size is substantially understated

Portfolio §4 "Implementation" says **"full pipeline in roughly 4,000 lines of Python."** Actual:

```
wc -l python/boilingsim/*.py → 6588 total
```

**Action:** quote "~6,600 lines of Python in the simulation package (`python/boilingsim/`), with a further ~2,000 lines of driver scripts and tests." Underselling the engineering scale costs you the "serious production-grade system" framing that IJHMT reviewers weigh.

### 1.3 Schema version is one bump behind

Portfolio §6 Claim 1 says the bump history is **"v1 → v2 → v3"**. Actual:

```
crates/ws-server/src/snapshot.rs:57  →  pub const SCHEMA_VERSION: u32 = 4;
web/src/types/snapshot.ts:23         →  export const SCHEMA_VERSION = 4;
```

v3 → v4 added `pot_diameter_m`, `pot_height_m`, `pot_wall_thickness_m`, `pot_base_thickness_m` to the snapshot so the 3D renderer can scale to the configured pot without a separate API call (see test `test_v4_pot_geometry_echoes_cfg` at [python/tests/test_dashboard_producer.py:131](python/tests/test_dashboard_producer.py#L131)).

**Action:** update Paper 3 §Claim 1 to read "three version bumps (v1 → v2 → v3 → v4), each a single atomic commit across Rust + Python + TypeScript with cross-stack fixture tests." This is actually a stronger claim — three coordinated bumps demonstrate the process works at scale, not just once.

### 1.4 Hardware claim is inconsistent across sections

Portfolio §1 says **"a single RTX 4090"** in one place and Paper 1 §Implementation says **"single RTX 6000 Ada workstation"**. Both are true — Phase 2/3/4 validation runs were on RTX 6000 Ada (per [benchmarks/phase2_heating.md:20](benchmarks/phase2_heating.md#L20) and [benchmarks/phase3_boiling.md:21](benchmarks/phase3_boiling.md#L21)), and the Phase 3.2 q-sweep + Phase 4.6 vitamin C runs were on RTX 4090 (terminal logs in the recent runs show `"NVIDIA GeForce RTX 4090"`). Pick one framing:

- **Honest option:** "benchmark runs on RTX 6000 Ada (48 GB, Phase 2/3/4 canonical artefacts) and RTX 4090 (24 GB, Phase 3.2/4.6 reviewer-critique runs)."
- **Reviewer-friendly option:** quote the RTX 4090 everywhere and note "the sim runs on any sm_89+ GPU with ≥ 24 GB." This maximises the "single workstation" accessibility claim, which is the point of the Warp contribution.

### 1.5 Rohsenow ratio quoted range conflates two metrics

Portfolio §1 executive summary says **"ratios 1.00 to 1.06 for q at or above 20 kW/m²"**. Portfolio §4 Paper 1 table lists ratios 1.06, 1.01, 1.00, 1.01 (the **ΔT_w** ratio from [scripts/run_boiling.py](scripts/run_boiling.py) energy-balance diagnostic). But the Phase 3 headline and the [scripts/analyze_q_sweep.py](scripts/analyze_q_sweep.py) output use the **implied-q** ratio (`q_Rohsenow(ΔT_meas) / q_stove`), which is the ΔT ratio cubed since Rohsenow is `q ∝ ΔT³`.

Measured values side-by-side:

| q_stove | ΔT_w ratio | implied-q ratio |
|---:|---:|---:|
| 10 kW/m² | 1.25 | 1.97 |
| 20 kW/m² | 1.06 (portfolio) / 1.08 (actual 6.31/5.86) | 1.25 |
| 30 kW/m² | 1.01 | 1.03 |
| 40 kW/m² | 1.00 | 1.01 |
| 50 kW/m² | 1.01 | 1.03 |

**Action:** pick one convention and use it throughout the portfolio, figures, and paper. The implied-q ratio is the one Phase 3 already reports in [benchmarks/phase3_boiling.md](benchmarks/phase3_boiling.md) (0.97–1.01× headline). I'd recommend standardising on implied-q since that's what Rohsenow literature uses when quoting scatter; the portfolio's ΔT-ratio framing is legitimate but less conventional. Also: the 1.06 at q=20 in the portfolio should be 1.08 — small rounding difference, but worth verifying from the HDF5.

### 1.6 The 5:1 cold-start and water-hot retention numbers reference deleted artefacts

Portfolio §5 Paper 2 axis 3 cites a three-row table:

```
cold-start      62.12 %
warm-water-only 66.83 %
all-warm        55.43 %
```

After the pre-boil artefact discussion (see [benchmarks/phase4_retention.md](benchmarks/phase4_retention.md) "Phase 4.6 extension" → "Pre-boil warm-up artefact" subsection), the cold-start and water-hot HDF5/PNG artefacts were deleted from `benchmarks/` because they're corrupted by the mismatched-stove-power warm-up physics. Only `phase4_retention_vitaminc_sonar_5to1_allhot.{h5,png}` remains on disk.

**Action (choose one):**

- **Option A — regenerate the two deleted artefacts with a clearer methodological framing.** Re-run both and keep them in `benchmarks/` tagged as "pre-boil artefact demonstrations". The three-row table survives in Paper 2; the artefact discussion is a short footnote.
- **Option B — rewrite the axis-3 narrative around a different framing.** Drop the 62.12 % / 66.83 % citations. Use two rows instead: re-anchored 104:1 / cold-start = 65.8 % and Sonar-matched 5:1 / all-hot = 55.4 %. The "thermal-regime fidelity" lever disappears, but it was confusing anyway — the 12 pp span across the three 5:1 variants was an artefact-driven spread, not a physics lever like the other two.

I'd pick Option B for Paper 2 and keep the three-variant discussion in the dashboard paper (Paper 3) as evidence of an artefact found during live use of the simulator — a more honest home for it.

### 1.7 The "88 tests at Phase 4 sign-off" is a historical marker, not current

Paper 2 §axis-5 cites **"Four new dual-solute tests (88 tests total at Phase 4 sign-off)"**. Current total is 134 (Phase 6.7 added Python tests for the `InitialConditionsConfig`, cold/preheat `build_simulation`, and other scientific-safety audit items). 88 is correct as a milestone marker but confusing in a paper that quotes 134 / 198 in the executive summary.

**Action:** reword to "88 Python tests at Phase 4 sign-off, 134 today after the Phase 6.7 audit added regression coverage for the initial-conditions config."

### 1.8 Paper 1 "Rohsenow agreement within 3% across four operating points" is generous

Paper 1 §Target journal says **"Rohsenow agreement within 3% across four operating points"**. Using the ΔT_w convention, the q ≥ 20 points are {1.06, 1.01, 1.00, 1.01} → max deviation from unity is 6 %, not 3 %. The claim "within 3 %" holds only for three of four (q = 30, 40, 50).

**Action:** use "within 1 % at three of four operating points in the fully-developed nucleate-boiling band, within 8 % at the fourth (q = 20 kW/m²)." This is both defensible and matches the analyzer output.

### 1.9 "Within 8 % of Rohsenow across a 5× flux range" omits q = 10

Portfolio §9 Caveat 3 says **"Rohsenow agreement to within 8 % across a 5x flux range"**. At q = 10 the ΔT ratio is 1.25 (25 % deviation on ΔT, 97 % on implied-q). The q = 10 point is explicitly called out as a regime-boundary artefact in §4 Paper 1, so the "within 8 %" caveat should be phrased "within 8 % across the fully-developed NB band (q ∈ [20, 50] kW/m²), with documented regime-boundary drift at q = 10 consistent with Hsu 1962's nucleation criterion breakdown."

### 1.10 Dashboard screenshot state in Paper 3 headline figure needs a validation run to back it

Paper 3 §headline figure quotes exact dashboard numbers: "β-carotene retention 81.58 %, vitamin C retention 39.87 %, 2,293 active bubbles". These look like they were captured during a live dev session, not from a canonical artefact. Before submitting, you want to (a) run a canonical dual-solute live session, (b) screenshot it at a known `t_sim`, (c) cite the `run_id` that the dashboard writes to `$BOILINGSIM_ARTIFACTS_DIR/{run_id}.json`. Otherwise a reviewer can't regenerate the figure.

**Action:** keep the narrative but add "run_id XXXXXXXX, artefacts bundled with the SoftwareX supplementary." The Phase 6.6 dashboard already emits `{run_id}.{h5,csv,json}` artefacts per completed run — use one.

---

## 2. Strengthening opportunities

Things the portfolio doesn't highlight that make the contribution look more significant to reviewers.

### 2.1 The physics-correctness-preserving refactor (Phase 4 conservative advection)

Paper 2 axis 8 mentions "a non-trivial refactor from trilinear SL to conservative upwind advection" but doesn't say *why it mattered*. The actual story from [benchmarks/phase4_retention.md:35-46](benchmarks/phase4_retention.md#L35-L46): the earlier semi-Lagrangian-trilinear advection was **non-conservative**, leaking C_water mass into solid-adjacent cells at ~0.5 % per step. The `degraded_pct = max(0, 100 − R − leach)` clamp silently absorbed the leak into the "degraded" bucket, making the model look like it was over-degrading when it was actually leaking mass. The conservative-upwind refactor plus the signed (unclamped) diagnostic surfaced both the bug and the fix simultaneously.

This is **exactly the kind of story Journal of Food Engineering reviewers respect**: the simulation instrumentation caught a subtle numerical error that would have invalidated the mass-partition validation. It demonstrates the validation is not a happy-path curve fit.

**Action:** add a 2-paragraph "diagnostic instrumentation" subsection to Paper 2 §axis 8. Quote the before/after numbers (0.5 % / step leak → 0.00 pp mass balance drift across the 600 s run). This is one of the strongest things in the codebase and the portfolio mostly hides it.

### 2.2 The three-stack coordinated commit record is a publishable engineering claim

Paper 3 §Claim 1 has this right but underplays it. The git history shows **three** coordinated schema bumps (v1→v2→v3→v4), each a single commit touching the Rust `SCHEMA_VERSION` const ([crates/ws-server/src/snapshot.rs:57](crates/ws-server/src/snapshot.rs#L57)), the Python `SCHEMA_VERSION` module constant ([python/boilingsim/dashboard.py](python/boilingsim/dashboard.py)), and the TypeScript export ([web/src/types/snapshot.ts:23](web/src/types/snapshot.ts#L23)). Each bump carries a companion test (`test_v2_*`, `test_v3_*`, `test_v4_pot_geometry_echoes_cfg` in [python/tests/test_dashboard_producer.py](python/tests/test_dashboard_producer.py)) that asserts the Python producer emits the new field and the Rust deserializer accepts it.

**Action:** add a git commit hash table to Paper 3 supplementary: "v1 → v2 commit hash X, test Y passed; v2 → v3 commit hash A, test B passed; v3 → v4 commit hash C, test D passed." This is one atomic claim per commit, with a passing test as evidence. Reviewers of software papers (SoftwareX specifically) score high on exactly this kind of traceability.

### 2.3 The Phase 6.7 initial-conditions config bug is a stronger story than the portfolio tells

Paper 3 §Claim 3 mentions the bug: **"`water.initial_temp_c = 20 °C` was silently overridden on every rebuild by hard-coded warm-start CLI defaults"**. What the portfolio doesn't say:

- The override silently defeated the UI input. A user setting water T to 20 °C in the browser would see the sim start at 95 °C without any warning or feedback.
- The fix required adding a first-class Pydantic config block (`InitialConditionsConfig` at [python/boilingsim/config.py:55](python/boilingsim/config.py#L55)) with `mode: Literal["cold", "preheat"]` and three preheat setpoints.
- The fix is **locked** by a regression test: [python/tests/test_run_dashboard_controls.py](python/tests/test_run_dashboard_controls.py) has `test_build_simulation_cold_start_honours_initial_temp` and `test_build_simulation_preheat_overrides_initial_temp` which build real `Simulation` objects and assert `grid.T[mat==MAT_FLUID].mean() == 293.15 K` (cold) or `368.15 K` (preheat) within 0.1 K.
- The same audit surfaced a second bug class: the frontend was exposing kinetic constants (Arrhenius E_a, Rohsenow C_sf, partition coefficients) as edit controls, so a non-expert user could silently break the validation story by nudging a slider.

The story arc — "we found the UI was silently lying to the user, fixed it, locked it with a regression test, and used the same audit to restructure the form around what users should actually be allowed to edit" — is a single tight case study in responsible research-software deployment.

**Action:** promote this to a named subsection in Paper 3 titled "Case study: the silent warm-start override and the scientific-safety audit that followed." It's a 400-word write-up that a SoftwareX reviewer will cite approvingly.

### 2.4 The bubble-boiling kernel's conservation cap has a worked numerical example

Portfolio §4 Paper 1 mentions the cap-bite diagnostic but doesn't tell the Paper 1 reviewer *why the cap exists*. From [python/boilingsim/boiling.py:784-789](python/boilingsim/boiling.py#L784-L789):

> The K-I^4.4 law explodes: at dT_w = 13 K it predicts > 400 kW/m², more than 10× the 30 kW/m² stove supply. An uncapped kernel transiently extracts more than the wall receives, pulling sensible heat from the bulk fluid through the wall and driving the bulk below saturation. The cap enforces that the vapor pathway cannot reject more than the wall actually receives.

This is **load-bearing physics**, not belt-and-braces. The cap is a closure that encodes the conservation principle "nucleate boiling cannot extract more heat than the wall supplies" — which is a Rohsenow-partition statement. Without it the Kocamustafaogullari-Ishii site density N_a ∝ ΔT^4.4 would drive unphysical behaviour at high wall superheats.

**Action:** write a 1-page methods subsection in Paper 1 titled "Conservation-constrained Rohsenow partition" that presents the cap as a physical closure. Quote the docstring, show q_raw(ΔT) evaluated at ΔT = 6.7 K (22 kW/m²) and ΔT = 13 K (466 kW/m²), and show the [scripts/analyze_q_sweep.py](scripts/analyze_q_sweep.py) cap-bite measurements across q ∈ [10, 50] kW/m² confirming the cap stays in the [0.88, 1.38] range. This converts a reviewer concern ("is the cap masking fragility?") into a methodological contribution ("the cap is a conservation closure that we explicitly validate across the operating range").

### 2.5 The dashboard-caught bugs are great Paper 3 material

Paper 3 §Claim 2 names the 2.5 GB OOM and the schema-v1 anonymous-carrot-retention bugs. Both were caught only under live use. You could add at least two more:

- **The 468-frozen-bubble artefact at cold-start in the 5:1 Sonar-matched case.** Found during a reviewer-prompted rerun, not by any unit test. Led to the root-cause analysis that identified the three-way interaction between the wall-boiling gate at [boiling.py:816](python/boilingsim/boiling.py#L816), the under-powered small-pot stove (71 W at q = 30 kW/m² × 23.8 cm² base), and the bulk evap sink at [thermal.py:512](python/boilingsim/thermal.py#L512). Documented in [benchmarks/phase4_retention.md](benchmarks/phase4_retention.md) "Phase 4.6 extension → Pre-boil warm-up artefact".
- **The mass-partition clamp that was hiding a conservation leak.** Surfaced during live dashboard development when someone noticed "degraded % jumping around while leached % holds steady" — the signed diagnostic showed the leak, and the conservative upwind rewrite fixed it.

**Action:** expand Claim 2 from two bugs to four. The "bugs the unit-test harness could not catch" framing is already strong; doubling the example count without padding any of them makes it twice as strong.

### 2.6 Benchmark artefact catalogue exists and should be cited

[benchmarks/runs.md](benchmarks/runs.md) is a complete reproducibility document for every HDF5/PNG artefact under `benchmarks/`. It lists the exact CLI invocation, scenario YAML, expected headline number, and output filenames for 21 distinct runs across Phase 2/3/3.2/4/4.6. The portfolio doesn't reference it.

**Action:** cite `benchmarks/runs.md` in all three papers' supplementary/reproducibility sections. This is the "paste this in a terminal and regenerate every figure" file — it's what reviewers and replicators actually need. Its existence is evidence your reproducibility posture is serious.

### 2.7 Performance numbers are scattered — consolidate into a single table

The portfolio mentions performance numbers in at least five places with varying figures. What the code and recent benchmark runs actually show:

| scenario | geometry | dev/prod tier | measured `s/sim-s` | RTX |
|---|---|---|---:|---|
| Phase 2 heating, implicit, steel 304 | default 20 cm pot | dx=2 mm dev | 0.87 | 6000 Ada |
| Phase 2 heating, implicit, aluminum | default | dx=2 mm dev | 0.81 | 6000 Ada |
| Phase 2 heating, implicit, copper | default | dx=2 mm dev | 0.78 | 6000 Ada |
| Phase 3 boiling, steel 304 | default | dx=2 mm dev | 2.58 | 6000 Ada |
| Phase 3 boiling, aluminum | default | dx=2 mm dev | 1.99 | 6000 Ada |
| Phase 3 boiling, copper | default | dx=2 mm dev | 2.17 | 6000 Ada |
| Phase 4 retention default 25 mm | default | dx=2 mm dev | 2.29 | 6000 Ada |
| Phase 4 dual-solute 25 mm | default | dx=2 mm dev | 2.86 | 6000 Ada |
| Phase 3.2 q-sweep q=10 | default | dx=2 mm dev | 0.70 | 4090 |
| Phase 3.2 q-sweep q=50 | default | dx=2 mm dev | 2.60 | 4090 |
| Phase 4.6 Sonar 5:1 all-hot | 5.5 cm pot | dx=2 mm dev | 0.52 | 4090 |

**Action:** replace the scattered performance citations with this table (or a cleaner version of it) in Paper 1 §Results and in Paper 3 §Validation evidence. The small-pot 0.52 s/sim-s is a headline number on its own — that's ~2× faster than wall-clock-real-time on a $1,500 consumer GPU.

### 2.8 The SCHEMA_VERSION = 4 history is a short timeline figure

A single figure showing:

```
v1 (Phase 6 M1): base snapshot schema — temperature, alpha, bubbles, retention
v2 (Phase 6 M-mid): + nutrient identity strings + four-bucket mass partition per solute
v3 (Phase 6.5): + water_temperature_{mean,max,min} + run_id + is_complete + last_error
v4 (Phase 6.6): + pot_diameter_m / pot_height_m / pot_wall_thickness_m / pot_base_thickness_m
```

with a "commits coordinate across Rust + Python + TypeScript" arrow underneath would be a powerful Paper 3 figure.

### 2.9 The dual-solute architecture claim deserves a code-level figure

Paper 2 §axis 5 says the dual-solute run uses "a shared `SoluteSlot` bundle". What the code actually looks like ([python/boilingsim/nutrient.py:1067](python/boilingsim/nutrient.py#L1067)):

```python
@dataclass
class SoluteSlot:
    """One solute's on-device arrays + config. The pipeline calls
    _step_reaction_diffusion_leach(slot, grid, D_carrot, dt) and
    _step_advect_clamp(slot, grid, dt) once per active slot per step.
    ...
    """
    C: wp.array               # carrot-cell concentration [mg / kg]
    C_water: wp.array         # water-cell concentration [mg / kg]
    ws: NutrientWorkspace     # slot-specific scratch (C_work, C_water_tmp)
    cfg: NutrientConfig       # per-slot Arrhenius + partition + saturation
    precipitated_mass: wp.array   # atomic counter, independent per slot
    c0: float                 # reference initial mass for retention %
```

Every single Phase-4 kernel (Arrhenius, diffusion, Sherwood leach, conservative advection, saturation clamp) takes a `SoluteSlot` as its first argument. Adding a second solute is literally one extra call per kernel per step, with no shared state between slots. The atomic counters for `precipitated_mass` are independent per slot by construction.

**Action:** include the `SoluteSlot` dataclass (truncated) as a code block in Paper 2 §axis 5. Caption: "the architecture that makes two-solute evolution a one-line config change, not a refactor."

### 2.10 You have four reviewer critiques with actual answers — lead with them

The portfolio handles caveats defensively (§9 "Honest Caveats and Reviewer Responses"). Flip it: you have four critiques from an actual external review, and you have **experimental answers** to each:

| Reviewer critique | Answer | Evidence |
|---|---|---|
| "2.45× Rohsenow overshoot" | Doesn't appear in the physical range; cap bite max 1.38× | [phase3_q_sweep.png](benchmarks/phase3_q_sweep.png) |
| "No q-sweep across 10 → 50 kW/m²" | Done; validates in [20, 50] band | [phase3_boiling.md](benchmarks/phase3_boiling.md) Phase 3.2 addendum |
| "No Vieira-faithful VC kinetics" | Done; lands in kitchen-boiling literature band | [phase4_retention.md](benchmarks/phase4_retention.md) Phase 4.6 addendum |
| "No 5:1 matched-volume VC run" | Done; within Sonar's HPLC scatter | same |

**Action:** reframe Paper 1 + 2 §Discussion as "Reviewer-anticipated concerns and their quantitative resolutions." This is a stronger framing than "honest caveats" — reviewers respect papers that address the concerns they were about to raise. Cite [benchmarks/phase3_q_sweep.png](benchmarks/phase3_q_sweep.png) and the Phase 3.2 / 4.6 addenda as evidence.

### 2.11 Grid convergence is cited as "pending" but a partial result is cheap

Paper 1 §Caveat 2 says grid convergence at 1 mm is a one-weekend outstanding item. A tighter claim is achievable cheaper: run a 5-minute q = 30 kW/m² boiling sim at dx = 1 mm, extract `T_inner_wall_mean_c` steady state, and compare to dx = 2 mm at the same q. If the two differ by < 5 % in ΔT_w, you have "grid-converged at 2 mm to within 5 % for the wall-boiling steady state" — which is a useful headline even without the full 600-s run. A 20-min spot check now is much cheaper than a 6-month revision later.

---

## 3. Per-paper actionable checklist

### Paper 1 (IJHMT)

- [ ] **Correction 1.1, 1.2, 1.4, 1.5, 1.8, 1.9** — test counts, LOC, hardware, Rohsenow ratio convention, "within 3 %" wording.
- [ ] **Strengthening 2.4** — write the "Conservation-constrained Rohsenow partition" methods subsection.
- [ ] **Strengthening 2.7** — consolidated performance table in §Results.
- [ ] **Strengthening 2.10** — reframe §Discussion around the four reviewer-anticipated concerns.
- [ ] **Strengthening 2.11** — run the dx = 1 mm spot check for grid convergence.
- [ ] **Supplementary** — cite [benchmarks/runs.md](benchmarks/runs.md) and [scripts/analyze_q_sweep.py](scripts/analyze_q_sweep.py).

### Paper 2 (Journal of Food Engineering)

- [ ] **Correction 1.6** — decide on Option A (regenerate) or Option B (drop the 62.12/66.83 citations). Recommend B.
- [ ] **Correction 1.7** — update the "88 tests" marker to also cite 134.
- [ ] **Strengthening 2.1** — add the "diagnostic instrumentation" subsection (conservative advection + signed mass-partition diagnostic).
- [ ] **Strengthening 2.9** — include the `SoluteSlot` dataclass code block with caption.
- [ ] **Strengthening 2.10** — §Discussion reframe around reviewer critiques.
- [ ] **Pre-submission** — K sensitivity sweep [0.3, 1.0] at Sonar geometry (still outstanding per portfolio).
- [ ] **Pre-submission** — second VC experimental match point (Sonar 2nd time, Vieira-matched, or similar).

### Paper 3 (SoftwareX)

- [ ] **Correction 1.1, 1.3** — test counts, schema version history v1→v2→v3→v4.
- [ ] **Correction 1.10** — screenshot with a canonical run_id.
- [ ] **Strengthening 2.2** — three-commit coordinated-schema-bump table in supplementary.
- [ ] **Strengthening 2.3** — promote the silent warm-start override + scientific-safety audit to a named case study.
- [ ] **Strengthening 2.5** — expand "bugs the unit-test harness couldn't catch" from two to four examples.
- [ ] **Strengthening 2.8** — add SCHEMA_VERSION timeline figure.

### Paper 4 (perspective, later)

No corrections yet — the perspective piece follows Papers 1–3.

---

## Appendix A — Glossary of code terms

External reviewers will see these identifiers and terms in the code blocks, test names, and file paths. Brief definitions so they can read without decoding from context.

### Pydantic config (python/boilingsim/config.py)

- **`ScenarioConfig`** — the root Pydantic model that every YAML under `configs/scenarios/` validates against. Contains `pot`, `water`, `carrot`, `heating`, `initial_conditions`, `grid`, `solver`, `boiling`, `nutrient`, `nutrient2`, `total_time_s`, `output_every_s`.
- **`InitialConditionsConfig`** — added in Phase 6.7. Fields: `mode: "cold" | "preheat"`, `preheat_water_c`, `preheat_wall_c`, `preheat_carrot_c`. Controls whether `grid.T` is seeded from `water.initial_temp_c` (cold) or overridden to benchmark-convenient hot-start values (preheat).
- **`SolverConfig`** — numerical tolerances: `cfl_safety_factor`, `max_dt_s`, `pressure_tol`, `pressure_max_iter`, `diffusion_tol`, `diffusion_max_iter`, `h_conv_outer_w_per_m2_k`, `h_evap_free_surface_w_per_m2_k`, `f_bulk_evap_per_s`, `use_implicit_conduction`. Literature-anchored; hidden from the UI as of Phase 6.7.
- **`BoilingConfig`** — nucleate-boiling parameters: `dT_onb_k` (onset of nucleate boiling wall superheat), `contact_angle_rad`, `max_bubbles` (Lagrangian pool size), `initial_bubble_radius_m`, `nucleation_probability_per_step`, `C_sf_rohsenow`, `Pr_n_rohsenow`. Likewise YAML-only.
- **`NutrientConfig`** — Arrhenius + transport parameters per solute: `E_a_kJ_per_mol`, `k0_per_s`, `D_eff_m2_per_s`, `K_partition`, `C0_mg_per_kg`, `C_water_sat_mg_per_kg`, `nu_water_m2_per_s`, `D_water_molec_m2_per_s`. Two slots (`nutrient`, `nutrient2`) for dual-solute runs.

### Pipeline (python/boilingsim/pipeline.py)

- **`Simulation`** — the top-level driver. `Simulation(cfg).step()` advances one timestep across fluid + thermal + boiling + nutrient physics. `Simulation.run(total_time_s, out_path, ...)` writes an HDF5 artefact with scalars + optional snapshots.
- **`ScalarSample`** — the per-step diagnostic dataclass emitted to HDF5 scalars. Fields include water/wall temperatures, bubble statistics, and the four-bucket mass partition per solute.

### Geometry (python/boilingsim/geometry.py)

- **`MAT_AIR = 0`**, **`MAT_FLUID = 1`**, **`MAT_POT_WALL = 2`**, **`MAT_CARROT = 3`** — material-ID enum values used as masks on the `grid.mat` int array to select which cells a kernel operates on.
- **`Grid`** — dataclass holding the full device-side state: `T` (temperature, K), `p` (pressure), `ux/uy/uz` (velocity components, m/s), `mat` (material IDs), `water_alpha` (void-fraction VOF), `C` / `C_water` / `C2` / `C_water2` (per-solute concentrations, mg/kg).

### Thermal kernels (python/boilingsim/thermal.py)

- **`conduct_one_step`** — wraps the implicit backward-Euler Jacobi conduction step that decouples Δt from solid-metal diffusivity α.
- **`apply_free_surface_evap_sink`** — open-pot enthalpy-bleed BC at the water-air interface. `h_evap_free_surface * (T − T_sat)` energy removed per cell with T > T_sat that is adjacent to air above. Pins the free-surface row at T_sat + ~0.1 K instead of letting the sealed domain supersaturate.
- **`apply_bulk_evap_sink`** — bulk-boiling closure. Per fluid cell with T > T_sat: `dT_remove = f_bulk_evap_per_s * (T − T_sat) * dt`, clamped to `(T − T_sat)` so the cell cannot be driven subcooled. Fires only when `cfg.boiling.enabled`.

### Boiling kernels (python/boilingsim/boiling.py)

- **`Bubble`** — `@wp.struct` representing one Lagrangian vapour bubble: position, velocity, radius, birth_time, active, site_i/j/k, site_cleared, departure_radius. The `departure_radius` field is *frozen* at the moment `site_cleared` flips 0 → 1 and is what the departure-diameter histogram reports (not the live `radius`, which keeps growing post-departure via Mikic-Rohsenow).
- **`BubblePool`** — container holding the full bubble array + auxiliary arrays (`slot_claim` for atomic-CAS allocation, `site_active` 3-D occupancy).
- **`fritz_departure_diameter`** — Fritz 1935 closed form `D_d = 0.0208 · θ_deg · √(σ/(g·Δρ))`. Depends on contact angle + surface tension only; sets the scale for bubble detachment.
- **`cole_frequency`** — Cole 1960 closed form `f = √(4g·Δρ/(3·D_d·ρ_l))`. Bubble-emission frequency per active nucleation site.
- **`lookup_site_density`** — Kocamustafaogullari-Ishii site-density correlation `N_a(ΔT_w) ∝ ΔT^4.4`. Linear-interpolated from a precomputed 101-entry LUT over ΔT ∈ [0, 50] K.
- **`mikic_rohsenow_radius`** — bubble-growth law `R(t) = (2/√π) · Ja · √(α_l · age)`. Zero if the local fluid is subcooled (Jakob number ≤ 0).
- **`detect_nucleation_sites`** / **`update_bubbles`** / **`scatter_latent_heat`** / **`scatter_bubble_momentum`** / **`reduce_water_alpha_by_bubble_occupancy`** — the five Lagrangian-pool kernels per step: spawn, advect+grow+depart+vent, bulk-latent sink, body-force momentum coupling, VOF reduction.
- **`apply_wall_boiling_flux`** — Eulerian microlayer sink. Per pot-wall cell bordering fluid: `q_boil = N_a(ΔT_w) · f · ρ_v · h_lv · (π/6)·D_d³`, gated on `ΔT_w ≥ dT_onb` AND `T_fluid_adj ≥ T_sat − 0.5 K`, capped at `q_stove_cap` for conservation. The cap is the "q_raw vs q_stove" diagnostic from Paper 1 supplementary.

### Nutrient kernels (python/boilingsim/nutrient.py)

- **`SoluteSlot`** — dataclass bundling one solute's arrays + config + atomic counter. Pipeline calls `_step_reaction_diffusion_leach(slot, ...)` and `_step_advect_clamp(slot, ...)` once per active slot.
- **`arrhenius_degrade`** — destroys mass in carrot cells at rate `k0 · exp(−E_a/(R·T)) · C`. Per-cell, first-order.
- **`arrhenius_degrade_water`** — companion kernel on the water side. Destroys mass that has leached out at the same Arrhenius rate. β-carotene runs never exercise this (K_partition = 1e-5 leaves C_water ≈ 0); vitamin C runs exercise it heavily at small geometry.
- **`leach_at_surface`** / **`_leach_flux_capped`** — Sherwood-correlation surface flux from carrot to water, with a driving-force gate (`C_carrot_face − K_partition · C_water`) and a saturation cap at `C_water_sat`. Excess flux routed to `precipitated_mass` atomic counter.
- **`advect_c_water`** — conservative finite-volume upwind advection of C_water by the fluid velocity field. Replaces the earlier non-conservative semi-Lagrangian trilinear scheme.
- **`clamp_c_water_and_track_precipitation`** — post-advection saturation clamp. Any mass pushed above `C_water_sat` by numerical pressure-projection residuals lands in the precipitated bucket, not silently discarded.

### Dashboard (python/boilingsim/dashboard.py + crates/ws-server + web/)

- **`SnapshotProducer`** — TCP emitter that msgpack-encodes a `build_snapshot(...)` dict and ships it length-prefixed to the Rust relay.
- **`ControlConsumer`** — TCP listener that drains a newline-JSON stream from the Rust relay into a thread-safe queue the Simulation drains between steps.
- **`SCHEMA_VERSION`** — triple-locked constant (Rust `u32` / Python module const / TypeScript export). `v4` as of Phase 6.6. Bumps require a single coordinated commit across all three.
- **`Snapshot`** / **`SnapshotSummary`** — wire-format struct (full grid + bubbles + scalars) vs history-ring struct (scalars only, ~100 bytes). The split fixed a 2.5 GB browser OOM.
- **`ControlMessage`** — externally-tagged enum of the messages the browser can send: `SetHeatFlux { value }`, `SetMaterial { value }`, `SetCarrotSize { diameter_mm, length_mm }`, `SetNutrient { value }`, `SetConfig { config }`, `StartRun { duration_s }`, `Pause`, `Resume`, `Reset`, `ExportSnapshot`, `RequestFullSnapshot`.

### Scenarios (configs/scenarios/)

- **`default.yaml`** — 20 cm steel 304 pot, 25 mm carrot, β-carotene kinetics, q = 30 kW/m². The Phase-2/3/4 canonical calibration scenario.
- **`aluminum.yaml` / `copper.yaml`** — same as default with pot material swap.
- **`simmer.yaml`** — default pot, q = 10 kW/m². Gentle simmer below fully-developed NB.
- **`vitamin_c_25mm.yaml`** — vitamin C kinetics (E_a = 74 kJ/mol, k0 = 1.1e7 /s, K = 1.0, C0 = 59 mg/kg). Re-anchored rate calibrated to blanching literature (R_thermal(600 s) ≈ 75 %).
- **`vitamin_c_25mm_vieira.yaml`** — same but k0 = 4.70e7 (Vieira 2000 extrapolation). Added Phase 4.6.
- **`vitamin_c_sonar_5to1.yaml`** — 5.5 cm pot geometry giving V_water/V_carrot ≈ 4.9. Added Phase 4.6.
- **`dual_solute_25mm.yaml`** — primary = β-carotene, secondary = vitamin C, both active. Phase 4 axis 5.
- **`boiling_q10.yaml` / `boiling_q20.yaml` / `boiling_q40.yaml` / `boiling_q50.yaml`** — default geometry with `heating.base_heat_flux_w_per_m2` set to the named q. Phase 3.2 q-sweep.

---

## Appendix B — Glossary of terminal-output fields

When a reviewer watches [scripts/run_boiling.py](scripts/run_boiling.py) or [scripts/run_retention.py](scripts/run_retention.py) run, these fields scroll past every ~20 s. What they mean:

### Progress line fields (both drivers)

- **`t`** — simulated-time position in the run, seconds.
- **`dt`** — current advection-CFL-limited timestep. Milliseconds typically; drops to < 1 ms during bubble transients.
- **`T_water_mean`** — volume-averaged fluid temperature in °C. Should pin at ~99.9 °C once boiling is established; values below are pre-boil warm-up.
- **`T_wall_max`** — hottest pot-wall cell (outer face for low-k materials reads higher than inner face). °C. A pre-boil spike to 130–140 °C at 5:1 small-pot configurations is the warm-up artefact documented in Phase 4.6.
- **`T_wall_inner`** — mean temperature of pot-wall cells whose +z neighbour is fluid. This is the **Rohsenow-relevant** superheat metric. Canonical steady-state NB value ≈ 107 °C at q = 30 kW/m².
- **`|u|_max`** — peak Eulerian fluid velocity in mm/s. Natural convection settles at 40-65 mm/s; steady nucleate boiling settles at 100-300 mm/s with transient spikes to > 1 m/s during onset overshoot.
- **`bubbles`** — count of currently-active Lagrangian bubbles. Typical 1k-3k at steady state for the default 20 cm pot; a *constant* bubble count (e.g. the 468-frozen pattern) is a diagnostic sign of subcooled-fluid bubbles stuck because Mikic-Rohsenow growth returns zero.
- **`R_mean`** — mean radius of live bubbles, mm. Sub-millimetre values indicate infant bubbles that can't grow (subcooled liquid); ~0.7-1.0 mm is typical steady-state.
- **`alpha_min`** — minimum water void fraction across the domain. 1.0 = no bubbles, 0.0 = cell fully occupied by vapour.

### Retention driver (`run_retention.py`)

- **`R`** — retention percentage: mass still inside carrot cells, relative to initial C0.
- **`leach`** — leached percentage: mass that has crossed the carrot-water interface into water cells, relative to initial C0.
- **`deg`** — degraded percentage: mass destroyed by Arrhenius (on either side of the interface), relative to initial C0. **Signed** diagnostic: negative values or sudden positive spikes indicate a numerical bug (advection leak, clamp silently dropping mass). Normally monotonic positive.
- **`precip`** — precipitated percentage: mass that the post-advection saturation clamp had to remove from C_water (pressure-projection residuals concentrating supersaturated mass). Normally < 0.5 %; large values indicate the solute is near its solubility cap.

The four values (`R + leach + deg + precip`) **must sum to 100.00 %** at every output timestep. This is the mass-balance invariant the Phase 4 validation relies on.

### Timing annotations

- **`(wall Xs, Y.YYs/sim-s)`** — total wall-clock elapsed since run start, and the current running average of wall-seconds per simulated-second. Lower is faster. Steady-state NB at dx = 2 mm is 0.5 – 2.6 s/sim-s on current-gen GPUs.

### Validation summary (`run_boiling.py`)

- **`dT_w inner (Rohsenow metric)`** — steady-state measured wall superheat, averaged over the final ~12.5 % of the run.
- **`dT_w outer (T_wall_max)`** — steady-state outer-face superheat. The outer-inner gap is `q · t_wall / k_material` (series-resistor solid conduction): ~7 K for steel, sub-K for copper. Reporting only T_wall_max over-states Rohsenow superheat by this gap for low-k materials.
- **`Rohsenow-predicted dT_w`** — inverse Rohsenow `ΔT_w = (c_p / h_lv) · C_sf · (q · √(σ/(g·Δρ)) / (μ · h_lv))^(1/3) · Pr^n` at the applied q_stove.
- **`Implied q at measured dT_w`** — forward Rohsenow `q(ΔT_meas)`. The "error vs stove" is `(q_Rohsenow − q_stove) / q_stove`; this is the implied-q ratio − 1.
- **`dT_wall / dT_Rohsenow`** — the ΔT-ratio (linear). Portfolio §4 uses this convention.

### Mass partition summary (`run_retention.py`)

- **`solute`** — display name from `--solute-label`; flows through to plot title and summary header.
- **`R(600 s)`** — final retention at the sim's target time.
- **`target band [LO, HI] (exp ref REF)`** — acceptance band + reference experimental value, both from CLI flags. "IN BAND" / "OUT OF BAND" verdict printed when R falls inside / outside the band.
- **`still in carrot / leached to water / degraded / precipitated`** — the four buckets, broken out for the final step.
- **`T_water final` / `T_wall inner`** — final steady-state temperatures, for quick sanity against the canonical 100 / 107 °C values.

---

*End of review.* Use the corrections list as a pre-submission fix checklist; use the strengthening opportunities as the material that turns a solid submission into a headline-worthy one. The appendices let any external reviewer read your figures and logs without needing a separate conversation.
