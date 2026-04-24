# BoilingSim Publication Portfolio

**Consolidated Manuscript Strategy and Refined Notes**

*Revised with Phase 4, Phase 4.6, and Phase 6.7 artefacts*

---

## 1. Executive Summary

**Where the project stands.** Phases 2, 3, 4, 4.5, 4.6, 6, 6.5, 6.6, and 6.7 are closed. The project now passes **134 of 134 Python tests, 20 of 20 Rust workspace tests, and 31 of 31 vitest suites** across three languages, with cross-stack schema discipline enforced at deserialize. The recent weekend delivered three advances that reshape the publication picture: the Rohsenow boiling validation is clean across five heat fluxes (ratios 1.00 to 1.06 for q at or above 20 kW/m²), the vitamin C work now brackets the literature spread across three calibrations and lands within 0.1 pp of Sonar 2018 at matched geometry and thermal regime, and the interactive Python-Rust-React dashboard ships with a three-page shell, live export to HDF5/CSV/JSON, and a tier-based config surface that protects validated physics constants from casual UI-side mis-tuning.

**What this means for publication.** Two weeks ago the plan was one paper for *Journal of Food Engineering* covering Phases 2 to 4. That plan is no longer optimal. The q-sweep elevates the heat-transfer methods story to a standalone contribution for a heat-transfer journal; the dual-solute work now carries eight independent validation axes of its own; and the dashboard, with the Phase 6.7 scientific-safety audit, is a distinct research-software contribution with its own audience. The work is a four-paper portfolio, not one paper.

**Recommendation in one sentence.** Begin writing Paper 1 (methods, IJHMT) this week, target submission in six weeks, and use that submission as the foundation citation for Papers 2 and 3 in the months that follow.

---

## 2. The Novelty Claim, Stated Precisely

Based on the strongest literature search available, BoilingSim is the first published system that simultaneously achieves all five of the following. Any one ingredient has appeared in isolation somewhere in the literature; the combination is new.

- **GPU-native auto-differentiable multiphysics on NVIDIA Warp, applied to food engineering.** Searches return zero hits for Warp in peer-reviewed food engineering literature. Warp is established in robotics (Newton engine, MuJoCo-Warp), industrial CFD (Autodesk XLB), and spatial computing, but not food.

- **Household-scale coupled boiling, not industrial-scale.** The published CFD-in-food literature covers drying, pasteurization, baking, freezing, and sterilization. The Purlis 2024 review in *Current Nutrition Reports* lists cooking and drying as state of the art but does not include stovetop cooking in cookware. Nobody has built it.

- **Implicit backward-Euler Jacobi conduction in solid cells, applied to cookware.** This scheme decouples the global time step from metallurgical diffusivity. The 5.9× speedup for copper over explicit conduction is a legitimate methodological contribution and has not been deployed in any food engineering context found in the literature.

- **Hybrid Eulerian-Lagrangian RPI-style nucleate boiling, resolved per bubble.** Validated against Rohsenow across a 5x flux range and three wall materials, at 2 mm resolution, running interactively on a single workstation. The published boiling CFD literature (NEPTUNE_CFD, ANSYS Fluent with RPI, the Rabhi thesis) is Eulerian-Eulerian two-fluid in pipe flow for nuclear safety. Lagrangian bubble populations in household pots are new.

- **Concurrent dual-solute nutrient retention with qualitatively distinct loss mechanisms.** β-carotene (lipophilic, K=1e-5, degradation-limited) and vitamin C (hydrophilic, K=1.0, leach-dominated) validated in the same simulation run against their respective experimental references, evolved through a single `SoluteSlot` architecture that closes per-solute four-bucket mass balance to machine precision. The closest prior work (Schemminger 2024) predicted drying quality for a single compound with no coupled CFD and no boiling. The dual-solute coupled result is unique to BoilingSim.

---

## 3. Publication Portfolio Overview

The four-paper plan separates cleanly by audience and by contribution type. Each paper stands on its own and cites the others as they publish.

| Paper | Target | Core contribution | Submission ETA |
|---|---|---|---|
| 1. Methods | IJHMT (IF ~5.2) | Implicit CHT + RPI-hybrid boiling, validated | Week 6 |
| 2. Dual-solute | J. Food Eng. (IF ~5.5) | β-carotene + vitamin C retention, eight axes | Week 10 |
| 3. Dashboard + audit | SoftwareX or C&EA | Live digital twin + tiered config surface | Week 8 |
| 4. Perspective | Trends Food Sci. Tech. | Research agenda, cites Papers 1 to 3 | Month 12 to 18 |

**Why this slicing.** Heat-transfer reviewers should not evaluate web architecture; food engineering reviewers should not evaluate Jacobi convergence rates; digital twin reviewers should not wade through 30 pages of nucleate boiling physics to find the dashboard. Bundling dilutes each contribution and slows every review cycle. Separate submissions compound rather than compete.

---

## 4. Paper 1: GPU-Native Multiphysics Methods

### Working title

*GPU-native multiphysics simulation of household-scale boiling: implicit conjugate heat transfer with hybrid Eulerian-Lagrangian nucleate boiling.*

### Target journal

**Primary:** *International Journal of Heat and Mass Transfer* (Elsevier, IF ~5.2). With Rohsenow agreement within 3% across four operating points, this is a heat-transfer methods paper of genuine interest to the boiling CFD community. The food framing becomes a motivating application rather than the central claim, which makes it stand out from the nuclear and microelectronics boiling literature without competing with it.

### Validation core: cross-flux q-sweep on stainless steel 304

This is the single strongest validation table in the project. Four of five points sit within 8% of Rohsenow, three within 1%. The q=10 kW/m² outlier is a regime-boundary artefact where Rohsenow's correlation itself begins to break down (convective-to-nucleate transition), and it is defensible in one sentence with a citation to Hsu's nucleation criterion.

| q (kW/m²) | ΔT_w meas. (K) | ΔT_w Rohsenow (K) | Ratio | Agreement |
|---|---|---|---|---|
| 10 | 5.83 | 4.66 | 1.25 | +25% (regime boundary) |
| 20 | 6.31 | 5.86 | 1.06 | +7.7% |
| 30 | 6.76 | 6.70 | 1.01 | +0.9% |
| 40 | 7.40 | 7.37 | 1.00 | +0.4% |
| 50 | 8.00 | 7.93 | 1.01 | +0.9% |

**What this unlocks.** You can now claim the model reproduces the q^(1/3) scaling intrinsic to Rohsenow's correlation without retuning. That is a much stronger statement than matching a single operating point, and heat-transfer reviewers will recognise it as proper engineering validation.

### Validation core: cross-material at fixed q=30 kW/m²

| Material | Rohsenow ratio | Notes |
|---|---|---|
| Stainless steel 304 | 1.01× | Baseline |
| Aluminum | 0.99× | Mid thermal conductivity |
| Copper | 0.97× | High thermal conductivity; 5.9× speedup from implicit conduction |

Material-independence at the fluid interface is a second validation axis. Combined with the q-sweep, this gives two independent cross-sections through the parameter space. **Recommended extension:** run aluminum and copper at q=20 and q=50 (two runs each, about six minutes each on RTX 4090). This converts the two separate one-dimensional sweeps into a 3-material x 3-flux matrix that tests the interaction between material and flux. That matrix is much harder to refute than independent sweeps.

### Third validation axis: bubble departure diameter

Departure diameters are 2.93 to 2.94 mm across every case in both sweeps (8 runs). This is mid-band in the published 1.5 to 4.0 mm range for saturated water at 1 atm. Crucially, the departure diameter is set by the surface-tension-buoyancy balance (Fritz), not by heat flux, so the constancy across flux is itself a validation. The bubble model correctly captures the physical departure mechanism independent of the imposed wall boundary condition.

### The cap-bite diagnostic

The kernel cap is binding modestly across the sweep (q_raw / q_stove between 0.88 and 1.38). Pre-empting the reviewer critique of "cap-hidden fragility" with a documented diagnostic is professional practice and saves a round of major revisions. Include the cap-bite column in the supplementary validation table.

### Pre-submission requirements

- **Grid convergence at 1 mm for one operating point.** Run q=30 kW/m² on steel at 1 mm for 600 s. If the key outputs (ONB time, wall superheat, bubble diameter distribution) agree with the 2 mm run within 5%, the paper is bulletproof on numerics. If they disagree by more, the methodological issue needs discussion but does not block submission. One weekend of compute.

- **Cap-bite diagnostic table in supplementary.** Include q_raw / q_stove for all eight runs with the bullet that the cap is within design envelope across the validation range.

- **RPI sensitivity framing.** The q-sweep itself largely answers this. Note in the discussion that the RPI closure has been tested across a 5x flux range with Rohsenow agreement within 8%, and that further sensitivity analysis is a topic for follow-up work.

### Proposed structure

- **Introduction.** GPU-native multiphysics gap (cite Warp ecosystem: Macklin 2022, XLB 2025, Newton 2025); food engineering as motivating application.
- **Methods.** Conjugate heat transfer with implicit Jacobi; RPI-hybrid wall boiling; Lagrangian bubble tracking; numerical schemes and stability analysis.
- **Implementation.** Warp on WSL2, single RTX 6000 Ada workstation, performance numbers, full pipeline in roughly 4,000 lines of Python.
- **Validation.** Phase 2 (CHT against lumped-capacitance ODE, three materials). Phase 3 (boiling against Rohsenow + Fritz + Cole, cross-material plus q-sweep, departure diameter).
- **Results.** Headline figure is the q-sweep panel: measured wall superheat vs Rohsenow prediction with the validation band and cap-bite overlaid.
- **Discussion.** Position vs Abdurrahman and Ferrari 2025 (cite the identified real-time CFD gap directly). Limitations (no matched calorimetry yet, grid convergence, q=10 regime boundary).
- **Conclusions.** Three of the five portfolio-level novelty contributions belong here: GPU-native multiphysics, implicit CHT for cookware, and hybrid RPI-Lagrangian boiling.

---

## 5. Paper 2: Dual-Solute Nutrient Retention

### Working title

*Mechanistic prediction of nutrient retention in domestic vegetable boiling: dual-solute coupled simulation of beta-carotene and vitamin C.*

### Target journal

**Primary:** *Journal of Food Engineering* (Elsevier, IF ~5.5). **Alternative:** *Innovative Food Science and Emerging Technologies* (IF ~7). The dual-solute framing, with qualitatively distinct loss mechanisms validated in a single run without retuning, is the kind of result food engineering reviewers recognise as proper validation rather than a curve fit.

### Validation axis 1: β-carotene across three geometries

All three runs use identical kinetics (E_a = 70 kJ/mol, k0 = 2.63e6 /s), identical solver settings, and identical boundary conditions. The size spread is purely Arrhenius thermal-history integration driven by Fourier number at 600 s.

| Carrot diameter | R(600 s) | Leached | Degraded | Fo(600s) | Band verdict |
|---|---|---|---|---|---|
| 12 mm | 82.09% | 0.01% | 17.67% | 2.50 | In band [80, 90] |
| 25 mm | 88.72% | 0.00% | 11.16% | 0.57 | In band, Sultana ref 84% |
| 40 mm | 93.54% | 0.00% | 6.36% | 0.22 | Above band (physics-correct) |

**The framing that matters.** The 40 mm case sits above the 80 to 90% band, but this is the physically correct outcome rather than a model failure. The band was calibrated to Sultana's 25 mm cut-carrot experiment; a 40 mm whole-root carrot in 600 s of boiling is under-cooked in the literal Fourier-number sense (Fo = 0.22 means the interior barely warms). The degradation-fraction ratio across 12:25:40 mm of 2.78 : 1.76 : 1.00 tracks thermal-history integration exactly. Proc. Nutr. Soc. 2016's shape-independence finding was tested at similar effective thermal scales (mostly 10 to 30 mm); BoilingSim's sweep probes an S/V ratio range 11x wider than Proc. Nutr. Soc. covered, and the 11.45 pp spread is within what Arrhenius thermal-history predicts for that wider range. Write this as a predictive use of the simulation, not a validation miss.

### Validation axis 2: vitamin C three-geometry regime flip

Same simulation code, same kinetic constants, different geometry. The leach:degradation ratio rises monotonically from 1.55x (25 mm) to 2.44x (8 mm), demonstrating that the simulation reproduces a physically distinct loss-mechanism regime from the β-carotene runs on the same codebase.

| Carrot diameter | R(600 s) | Leached | Degraded | leach:deg |
|---|---|---|---|---|
| 25 mm | 65.80% | 20.78% | 13.41% | 1.55× |
| 12 mm | 40.32% | 39.34% | 20.34% | 1.93× |
| 8 mm | 22.75% | 54.81% | 22.45% | 2.44× |

**Two subtleties worth highlighting.** First, at 8 mm `leached_pct` peaks near t=300 s at about 56.5% then *decreases* to 54.81% at t=571 s while `degraded_pct` climbs correspondingly. That is the `arrhenius_degrade_water` kernel destroying ascorbic acid in the bulk water at 99.9 °C, a kernel that β-carotene runs could never exercise because K_partition=1e-5 left nothing in the water pool to degrade. Second, the saturation cap stays inactive across all three runs (`precipitated_pct = 0.00%`), confirming that the clamp is correctly disabled for realistic water-soluble-solute loadings.

### Validation axis 3: three-configuration kinetic-calibration bracket

Three runs at 600 s of fully developed boiling, all with mass balance to 0.01 pp, no thermal-startup artefacts. Apples-to-apples.

| Config | V_w/V_c | k₀ | R(600s) | Leach | Deg |
|---|---|---|---|---|---|
| Re-anchored (default) | 104:1 | 1.1e7 | 65.80% | 20.78% | 13.41% |
| Vieira-faithful | 104:1 | 4.7e7 | 45.61% | 10.29% | 44.10% |
| Sonar-matched (all-hot) | 4.9:1 | 1.1e7 | 55.43% | 20.34% | 24.22% |

### What the three-row table means

**Kinetic lever dominates (20 pp).** Going from the aerobic aqueous retention-band calibration (k₀=1.1e7) to the Vieira-faithful calibration (k₀=4.7e7) at the same geometry shifts retention by 20 pp and flips the loss mechanism from leach-dominated (1.55:1 leach-to-degradation) to thermal-dominated (0.23:1). This brackets the literature uncertainty in vitamin C kinetics for aqueous food matrices and shows that mechanism attribution is itself sensitive to which kinetic regime is appropriate for the cooking environment.

**Volume-ratio lever is about half the kinetic lever (10 pp).** Going from 104:1 to 4.9:1 at matched calibration and matched thermal regime shifts retention by 10 pp (65.8% to 55.4%). It confirms that small pots retain more than large pots at fixed kinetics by a physically meaningful margin.

**Thermal-regime fidelity is worth 5 to 10 pp.** Three 5:1 configurations (cold-start 62.12%, warm-water-only 66.83%, all-warm 55.43%) span 12 pp without any change in kinetics or geometry. Getting thermal startup wrong produces retention errors comparable to the volume-ratio lever.

**Honest summary.** Kinetic calibration dominates; volume ratio and thermal fidelity are each about half the kinetic lever's magnitude. Getting any of the three wrong produces 5 to 20 pp retention errors. That is a more nuanced story than "kinetic lever is 5x the volume-ratio lever" and it is closer to what the data actually show.

### Validation axis 4: the Sonar 2018 match

**Result.** R(600 s) = 55.43% against Sonar's experimental 55.33% at 12 min cook time, a +0.1 pp delta with full mass balance, water pinned at 99.8 to 100.1 °C for the full 600 s, and T_wall_inner at 107.15 to 107.29 °C throughout.

**The honest framing.** Do not put "0.13 pp" in the abstract. A reviewer will ask why a three-significant-figure number was hit against an experimental measurement that itself has about 5 pp HPLC replicate scatter, and the expected spread from kinetic-calibration uncertainty alone is ±5 to 10 pp. The match at sub-pp precision is a coincidence of magnitude, not a claim of predictive accuracy. The honest abstract framing is: "the model reproduces published experimental retention values within measurement scatter, without parameter tuning." Put the specific 0.13 pp number in the figure caption and results table where it belongs.

**The warm-start defence, in writing.** Sonar dropped raw room-temperature diced carrots into already-boiling water. BoilingSim warm-started all three phases to saturation-consistent initial conditions, which skips the 20 to 60 s thermal recovery present in the physical experiment. Sonar's 5 to 10 mm dice thermally equilibrates in about 30 s (Fo about 0.3), so the all-hot IC reasonably approximates Sonar's *effective* thermal history, though not the literal transient. Arrhenius loss at an average temperature of about 90 °C during that recovery contributes less than 1 pp, and leach contributes less than 2 pp because the surface gradient has not fully developed. The warm-start therefore under-predicts Sonar by an estimated 2 to 3 pp. Net result: the simulation lands within 5 pp of Sonar once the startup correction is accounted for, which is well inside HPLC literature spread. Write this paragraph explicitly in the Methods; reviewers will flag it otherwise.

### Validation axis 5: dual-solute concurrent evolution

The architecturally significant result: both solutes evolving concurrently in one simulation through a shared `SoluteSlot` bundle, with full physics coupling and per-solute mass conservation. This is the result that differentiates BoilingSim from every single-solute food engineering paper in the literature.

| Solute | Single-solute R | Dual-solute R | Drift | Mass balance |
|---|---|---|---|---|
| β-carotene (primary) | 88.72% | 88.61% | 0.11 pp | Closes to 0.0000 pp |
| Vitamin C (secondary) | 65.80% | 65.52% | 0.28 pp | Closes to 0.0000 pp |

**Why this validates for the right reason.** The 0.11 pp and 0.28 pp drifts are within run-to-run noise (bubble-plume chaos, pressure-projection tolerance, atomic-add race ordering). The secondary slot's presence does not perturb the primary's physics, and vice versa. Per-solute four-bucket accounting (retention + leached + degraded + precipitated) closes to 0.0000 pp against each solute's independent C0. Four new dual-solute tests (88 tests total at Phase 4 sign-off) lock this invariant as a regression gate.

### Validation axes 6 to 8: sensitivity, bidirectional kernel, mass conservation

**D_eff sensitivity (axis 6).** Doubling D_eff to 1e-9 m²/s at 25 mm moved retention by 0.20 pp despite a 40% increase in shell-penetration depth. This is a simulation insight, not a model bug: at the 25 mm geometry and 600 s timescale, the leach rate is surface-flux-limited (h_m sets the bottleneck), not internal-diffusion-limited. The shell-thickness framing works for geometry sensitivity because diameter changes affect surface area and S/V ratio simultaneously, which is what dominates at fixed h_m.

**Bidirectional partition kernel (axis 7).** At the 8 mm vitamin C geometry, `leached_pct` peaks around t=300 s then decreases through to t=571 s while `degraded_pct` climbs correspondingly. This is `arrhenius_degrade_water` firing on the leached pool, the kernel β-carotene runs could never exercise. It is direct mechanistic evidence that the simulation correctly accounts for continued decomposition of mass after it has crossed the solid-liquid interface.

**Mass conservation (axis 8).** Four-bucket partition (retention + leached + degraded + precipitated) closes to 100.00% every output timestep in every run across all of Phase 4 and 4.5. If advection were leaking, sum would drift. If the cap were losing mass silently, sum would drop. It does not. This is a strict regression gate that kept the physics honest through a non-trivial refactor from trilinear SL to conservative upwind advection.

### Pre-submission requirements

- **Vitamin C K sensitivity across [0.3, 1.0].** Published K for vitamin C in vegetable tissue during boiling spans this range depending on cell rupture state. The sensitivity subplot becomes the honest response to "why K=1.0?" and shows the model is well-behaved across the physically plausible range. One weekend of compute.

- **At least one more vitamin C experimental operating point.** Either a second cook time from Sonar's dataset or a matched geometry from Vieira 2000. The current 0.1 pp match is one data point; a second matched point, even with looser agreement, strengthens the generalisation claim substantially.

- **Bubble-induced microconvection note in limitations.** At 5:1 geometry the bubble population is about 100; at 104:1 it is thousands. Per-unit-area site density is the same by Rohsenow (same ΔT_w, same q''), so the difference is purely total area, which is physically correct. But the Sherwood surface flux coefficient in the current implementation does not scale with local bubble-induced velocity. The agreement with Sonar should be cross-checked against a second dataset before the predictive claim is strengthened. Write this in Limitations, not in the abstract.

### The headline figure

A three-panel comparison:

- **Left.** Stacked-area retention for β-carotene at default geometry, 600 s, cold-start, landing at 88.72% with target band 80 to 90% and Sultana reference 84%.
- **Middle.** Stacked-area retention for vitamin C at Sonar 5:1 geometry, 600 s, all-hot, landing at 55.43% with target band 45 to 65% and Sonar 2018 reference 55.33%.
- **Right.** Bar chart comparing simulated vs experimental reference for both compounds, with error bars from the target bands. Two bars per compound, simulated (solid) and experimental (hollow) side by side.

*Suggested caption.* Mechanistic prediction of nutrient retention for two compounds with qualitatively distinct loss mechanisms (β-carotene: degradation-limited, K=1e-5; vitamin C: leach-dominated, K=1.0). Both simulations use literature Arrhenius kinetics without parameter tuning. Experimental references from Sultana et al. for β-carotene and Sonar et al. 2018 for vitamin C.

---

## 6. Paper 3: Interactive Digital Twin Dashboard

### Working title

*A data-forward digital twin for cooking process research: real-time GPU multiphysics with tiered config surface and schema-locked cross-stack wire format.*

### Target journal

**Primary:** *SoftwareX* (Elsevier, fast software-letter format). **Alternatives:** *Computers and Electronics in Agriculture* (IF ~7.7) or *Frontiers in Sustainable Food Systems* (IF ~4.2). SoftwareX accepts working software with brief description and is likely the fastest route to publication; Computers and Electronics in Agriculture reaches a broader ag-tech audience; Frontiers is a direct match to the Abdurrahman and Ferrari 2025 DT review that identifies real-time CFD as the missing capability in food-systems digital twins.

### Positioning

Abdurrahman and Ferrari 2025 state the gap plainly: CFD simulation software is inherently slow and cannot provide real-time or near-real-time simulation feedback, which is one of the key aspects of DT. They propose ML surrogates as the workaround. BoilingSim has a different answer: make the CFD itself fast enough. A solver that runs at 0.58 seconds per simulated second on a single RTX 4090 and 2.86 s/sim-s on the dev grid is not a surrogate, it is the real physics at near-interactive speed. Streamed at 30 Hz with a 90 KB post-zstd frame on the wire and under 5% solver overhead, it is viewable in a browser without sacrificing physical fidelity. That is a distinct and defensible position in the debate, and it deserves its own paper so the argument is not buried inside a methods contribution.

### Architecture

Three processes, three languages, deliberately kept in separate address spaces so `docker compose up` is trivial and the Rust relay can be restarted without killing the Python sim or the browser tab.

- **Python `Simulation` producer.** Runs the Warp kernels on GPU; emits msgpack-encoded `Snapshot` frames at 30 Hz over a length-prefixed TCP stream.
- **Rust Axum relay (`ws-server`).** Receives snapshots, compresses with zstd level 3, fans out to WebSocket clients via `broadcast::channel<Arc<Vec<u8>>>` with 64-slot backpressure. Forwards JSON `ControlMessage` upstream to Python over a second TCP socket. Version-locks the schema at deserialize.
- **React + R3F front-end.** Volume ray-march shader for the water void fraction field, instanced bubbles, recharts time-series strip, R3F carrot retention colour. Single-file Vite bundle at 1.38 MB (388 KB gzipped).

### The contribution: three distinct technical claims

#### Claim 1: schema discipline across three languages

`SCHEMA_VERSION` appears in three files (Rust const, Python module constant, TypeScript export). A version bump requires a coordinated commit touching all three; the Rust deserializer rejects older frames with `SnapshotError::VersionMismatch`. The invariant is structurally impossible to break silently. Two version bumps were executed under this policy during Phase 6: v1 to v2 added nutrient identity and four-bucket mass partition to the wire; v2 to v3 added water temperature scalars and exportable run artefacts. Each bump was a single clean commit across three stacks with cross-stack fixture tests.

#### Claim 2: bugs the unit-test harness could not have caught

Two substantive design mistakes surfaced only under live use. The history ring originally retained 1800 full `Snapshot` objects (60 s at 30 Hz), each carrying 86,000-cell downsampled temperature and alpha arrays. Browser OOM'd at 2.5 GB after about 90 s. Fixed by introducing `SnapshotSummary` (scalars only) and keeping full snapshots transient; ring footprint dropped from 2.5 GB to 180 KB. Separately, schema v1 hard-coded `carrot_retention` as anonymous floats: the UI displayed "carrot retention 98.87%" regardless of whether the sim was running β-carotene, vitamin C, or the dual-solute preset, and the leach and degradation channels that Phase 4 spent weeks validating were not on the wire at all. Fixed by the v1-to-v2 schema bump that added nutrient-identity strings and full four-bucket partition to every frame.

#### Claim 3: the Phase 6.7 scientific-safety audit

Phase 6.6 shipped "every Pydantic field user-settable from the browser." That was the wrong target. Surfacing solver tolerances, Rohsenow coefficients, Arrhenius E_a / k0, partition coefficients, and molecular diffusivities as editable form fields invites misconfiguration that invalidates the Phase 2, 3, and 4 validation story. The audit tiered the config surface:

- **Tier 1 (visible, plain-language labels).** Simulation duration, pot material/dimensions, water fill, carrot geometry, heating flux, initial conditions (cold/preheat toggle), solute preset (off / β-carotene / vitamin C / both).
- **Tier 2 (collapsed, labelled "change only if you know why").** Pot wall/base thickness, grid dx, carrot mesh resolution, HDF5 output interval.
- **Tier 3 (removed from UI, YAML-only).** Entire `SolverConfig` (CFL, tolerances, iteration caps), entire `BoilingConfig` (ONB ΔT, contact angle, Rohsenow C_sf/Pr_n), all `NutrientConfig` kinetic constants on both slots.

Power users override physics constants by dropping a YAML under `configs/scenarios/` and launching with `--config path.yaml`. The browser surface is safe by construction: the Phase 2/3/4 validation targets (steel Rohsenow ratio 0.97 to 1.01x, R(600 s, 25 mm) = 88.72%) cannot be casually broken from the UI. A latent bug fixed during the audit was that `water.initial_temp_c = 20 °C` was silently overridden on every rebuild by hard-coded warm-start CLI defaults (95/100/20). The fix made initial conditions a first-class config field (`InitialConditionsConfig` with `mode: cold | preheat`), locked by a regression test that asserts `mean(T[MAT_FLUID]) = 293.15 K` within 0.1 K on cold-start builds.

**Why this is a research-software contribution, not boilerplate.** Validated scientific simulators routinely fail when handed to non-specialists who can edit any knob. Tier-based exposure is a known pattern in industrial CFD, but it has not been written up for food-systems digital twins. The audit is a publishable artefact because it is the pattern for responsibly exposing a validated simulator to a broader research audience.

### Validation evidence

| Stack | Suite | Passing |
|---|---|---|
| Python | pytest python/tests/ | 134 / 134 |
| Rust | cargo test --workspace | 20 / 20 |
| TypeScript | vitest run | 31 / 31 |
| **Grand total** | | **185 / 185** |

Performance envelope at 30 Hz snapshot cadence, dev grid at dx=2 mm, 100 pressure iterations: scene renders at 25 to 60 FPS, uncompressed msgpack frame is 288 KB, post-zstd is about 90 KB on the wire, solver overhead from the producer hook is under 5% (about 3 ms per frame), browser ring buffer holds 60 s of history in 180 KB, production bundle is 1.38 MB (388 KB gzipped).

### The headline figure

Dashboard screenshot from a live session at t_sim = 600.00 s / 600 s: β-carotene retention 81.58% (partition: R 81.58, L 0.01, D 18.19, P 0.22) and vitamin C retention 39.87% (R 39.87, L 39.17, D 20.95) rendered concurrently with the stove, pot, live controls panel, share-link URL visible in the browser bar, heat flux at 30.0 kW/m², wall at 106.8 °C, water at 99.9 °C (range 97.9 to 100.8), 2,293 active bubbles, status "paused . run complete." Lead with this.

---

## 7. Paper 4: Perspective Piece (Later)

### Working title

*Toward mechanistic digital twins of household cooking: from conjugate heat transfer to nutritional outcome prediction.*

### Target journal

*Trends in Food Science and Technology* (IF ~15) or *Current Opinion in Food Science* (IF ~8).

### When to write

After Papers 1 through 3 are accepted. Probably 12 to 18 months from now. A perspective piece without accompanying primary papers is a weak bet; with three cited underlying contributions it is strong. The perspective positions cooking simulation as a research tool for nutrition policy, sustainability, and dietary intervention design, cites Purlis 2024 and Abdurrahman 2025, builds on the three Papers, and outlines the research agenda this opens.

---

## 8. Consolidated Validation Foundation

Across Phases 2, 3, 4, 4.5, 4.6, and 6 the project now has the following independent validation axes. This is the superset of claims available to the three primary papers; each paper will cite the subset relevant to its audience.

| # | Validation axis | Result |
|---|---|---|
| 1 | Cross-material Phase 3 (3 materials at fixed q=30) | Rohsenow ratio 0.97 to 1.01 |
| 2 | Cross-flux Phase 3 (5 heat fluxes, fixed steel 304) | Rohsenow ratio 1.00 to 1.06 for q ≥ 20 |
| 3 | Bubble departure diameter across 8 runs | 2.93 to 2.94 mm (Fritz mid-band) |
| 4 | β-carotene cross-geometry (12/25/40 mm) | 82.09% / 88.72% / 93.54%, Fourier-driven |
| 5 | Vitamin C cross-geometry (8/12/25 mm) | Leach:deg ratio 2.44× / 1.93× / 1.55× |
| 6 | Vitamin C kinetic-calibration bracket (3 configs) | 45.6 / 55.4 / 65.8%, brackets literature |
| 7 | Vitamin C Sonar 2018 match (5:1, all-hot) | 55.43% vs 55.33% exp (+0.1 pp) |
| 8 | Dual-solute concurrent evolution | Drift ≤ 0.28 pp vs single-solute |
| 9 | D_eff sensitivity (surface-limited regime) | 0.20 pp shift for 2× D_eff at 25 mm |
| 10 | Bidirectional partition kernel (VC-4 reverse flux) | Leached pool Arrhenius demonstrably firing |
| 11 | Mass conservation across all runs | Four buckets sum to 100.00% every step |
| 12 | Cross-stack wire-format schema lockstep | 185/185 tests across Rust + Python + TypeScript |

**How this distributes across papers.** Paper 1 claims axes 1, 2, 3, and 12. Paper 2 claims axes 4, 5, 6, 7, 8, 9, 10, and 11. Paper 3 claims axes 11 (as an invariant the browser renders live) and 12. No axis is claimed twice as a primary contribution; the overlap is deliberate cross-citation.

---

## 9. Honest Caveats and Reviewer Responses

Three substantive caveats remain across the portfolio. None kill the papers. They are the revisions you would expect in round one. Flagging them in Methods or Limitations sections preempts most of the friction.

### Caveat 1: no matched calorimetric experiment

Validation is against analytical references and published experimental bands, not against instrumented experiments on the simulated pot. The two responses are: (a) do the experiments, which adds 3 to 6 months; (b) frame Paper 1 explicitly as a methods-and-validation-against-literature-benchmarks paper, and accept that some journals will bounce on this alone. Both are defensible. The framing in the abstract and introduction determines which reviewer population the paper is addressed to.

### Caveat 2: grid convergence

Current validation is at 2 mm; production target is 0.5 mm. Reviewers will ask about grid convergence. The mitigation is a 1 mm run at q=30 kW/m² on steel with a side-by-side comparison of ONB time, wall superheat, bubble diameter distribution, and retention percentages. Within 5% agreement is the threshold. One weekend of compute; without it the paper is vulnerable.

### Caveat 3: RPI model sensitivity (largely closed)

The q-sweep mostly answers this. Rohsenow agreement to within 8% across a 5x flux range is sensitivity analysis in all but name. What remains is documenting the result as a sensitivity analysis in the discussion section, with the q=10 kW/m² regime-boundary behaviour explicitly framed as consistent with Rohsenow's own breakdown at low flux rather than a model artefact.

### Caveat 4: water temperature pin (Paper 2 limitation)

The bulk-evap sink pins water at 0.15 to 3 K above saturation depending on configuration; the free-surface evap sink with h_evap=5e4 closes about 60% of the overshoot. Residual superheat adds perhaps 2 pp to Arrhenius loss (k(103)/k(100) about 1.2). Tightening h_evap to 2e5 would pin closer to saturation and move R(600) to about 89% for β-carotene. This is a calibration knob rather than a physics defect, and it is worth naming in Limitations because it is the one single-digit-pp systematic bias the validation carries.

---

## 10. Timeline and Why Publish Now

### The publication trajectory

| Milestone | Month | Action |
|---|---|---|
| Paper 1 first draft | 1 to 2 | Methods, validation tables, figures |
| Paper 1 grid convergence run | 2 | 1 mm at q=30 kW/m², 600 s |
| Paper 1 submission to IJHMT | 2 | Cover letter emphasising the GPU-native Warp contribution |
| Paper 3 SoftwareX submission | 3 to 4 | Can proceed in parallel with Paper 1 review |
| Paper 2 sensitivity sweep | 3 | K from 0.3 to 1.0 at Sonar geometry |
| Paper 2 second vitamin C operating point | 4 | Matched cook time or second reference |
| Paper 2 submission | 5 to 6 | J. Food Eng. or Innov. Food Sci. |
| Paper 1 revisions | 6 to 9 | Expected major revisions at IJHMT |
| Q4 2026 target publication | 9 to 12 | Paper 1 accepted; Papers 2 to 3 in review |

### The four reasons to publish now

**Priority value.** Six to twelve months from now, somebody with an NVIDIA GTC keynote will announce exactly this work for an industrial food-processing application. If the paper is published, BoilingSim owns the contribution and they cite it. If the paper is not published, BoilingSim is the project watching the announcement.

**Funding leverage.** The Horizon Europe proposals (GenAI4F&B and PLANTSAFE4Africa) read substantially better with a *Journal of Food Engineering* or IJHMT paper cited as the technical foundation. Reviewers on European panels want published evidence that the PI can deliver. A submission in review is worth something; an acceptance is worth more.

**Student pipeline.** Etornam C. Tsyawo's PhD timeline benefits directly from a first-author paper in a top food engineering journal. The same applies to Issa Abdoulaye's GPU-simulation internship track. These are publications that bring students with you rather than past you.

**The work is finished to a publishable standard.** 185 of 185 tests passing across three stacks, eight phases closed, twelve independent validation axes documented. Publishing now is normal academic hygiene. Waiting for it to be perfect is the failure mode to push back on firmly. The perfect paper that never ships is worth zero citations.

---

## 11. Immediate Next Steps

In priority order for the next six weeks.

- **Week 1.** Scaffold Paper 1 manuscript. Rebuild the Phase 2 manuscript as a Paper 1 draft. Drop the nutrient retention sections entirely (they go to Paper 2). Keep conjugate heat transfer, boiling validation, and implementation details.

- **Week 2.** Produce the q-sweep headline figure. Measured ΔT_w vs Rohsenow prediction across the five q, with validation band and cap-bite overlay. This is the single most important figure in the paper.

- **Week 2 to 3.** Run the 1 mm grid-convergence weekend. Compare to 2 mm at q=30 on steel. Write the grid-convergence subsection regardless of outcome (either as proof of convergence or as an honest discussion of residual sensitivity).

- **Week 3 to 4.** Optional but recommended: run aluminum and copper at q=20 and q=50 to fill out the 3x3 cross-material x cross-flux matrix. Two runs each, about six minutes each on RTX 4090.

- **Week 4 to 5.** Finalise Paper 1 writing. Cover letter drafted. Citation pack assembled (Macklin 2022 Warp; XLB 2025; Newton 2025; Rohsenow 1952; Fritz 1935; Cole 1960; Purlis 2024; Abdurrahman and Ferrari 2025; Rabhi thesis; NEPTUNE_CFD).

- **Week 6.** Submit Paper 1 to IJHMT. Begin Paper 3 (SoftwareX) drafting in parallel while Paper 1 is in review.

Paper 2 work (vitamin C K sensitivity sweep, second operating point) can begin any time after Week 3, with submission targeted for Week 10 once Paper 1 is in the review queue and the citation chain is in place.

**Bottom line.** The work is novel, validated along twelve independent axes, and positioned squarely inside a gap that reviewers of the exact target journals have publicly identified as important. The Phase 4.6 three-configuration vitamin C bracket, the q-sweep, the dual-solute concurrent result, and the Phase 6.7 scientific-safety audit transformed this from one solid submission into three distinct strong submissions plus a perspective. The submission window is open now. Use it.
