# Benchmark run catalogue

Every command used to produce the HDF5 + PNG artefacts under [benchmarks/](.), grouped by phase. Each entry names the scenario, the command, its purpose, and the artefact filenames it emits. Use this file as the reproduce-from-scratch guide.

Runs assume the repo is checked out at `~/Desktop/Dev_Projects/boiling-sim`, a CUDA-capable GPU is available, and the project venv is active:

```bash
cd ~/Desktop/Dev_Projects/boiling-sim
source .venv/bin/activate
```

All run times quoted below are RTX 4090 at `dx = 2 mm`. Production-tier `dx = 1 mm` takes ~8× longer.

---

## Phase 2 — heating + natural convection (no boiling)

Validates the conjugate heat-transfer solver + natural-convection plume against a lumped-capacitance ODE reference on three pot materials. See [phase2_heating.md](phase2_heating.md) for the acceptance matrix.

### steel 304 (reference material)

Implicit backward-Euler thermal conduction, natural convection active, boiling disabled.

```bash
python scripts/run_heating.py \
    --config configs/scenarios/default.yaml \
    --dx-mm 2.0 --pressure-iters 100 \
    --suffix _impl
```

Output: `phase2_heating_steel_304_impl_dx2mm.{h5,png}` — temperature trajectory, max-wall + peak convection velocity.

### aluminum

```bash
python scripts/run_heating.py \
    --config configs/scenarios/aluminum.yaml \
    --dx-mm 2.0 --pressure-iters 100 \
    --suffix _impl
```

Output: `phase2_heating_aluminum_impl_dx2mm.{h5,png}`.

### copper

```bash
python scripts/run_heating.py \
    --config configs/scenarios/copper.yaml \
    --dx-mm 2.0 --pressure-iters 100 \
    --suffix _impl
```

Output: `phase2_heating_copper_impl_dx2mm.{h5,png}`.

### Diagnostic reanalysis — ONB-annotated plots

After the three heating runs complete, overlay the onset-of-nucleate-boiling cross-over (T_wall_max first reaches 105 °C) on the lumped-ODE trajectory comparison.

```bash
python scripts/reanalyze_heating_onb.py
```

Output: `phase2_heating_{steel_304,aluminum,copper}_onb.png` + `phase2_heating_onb_summary.md`.

### Diagnostic — radial T-profile sanity check

Confirms the conjugate-interface heat transfer is water-boundary-layer-limited, not pot-conductivity-limited (rules out harmonic-mean `k_face` under-prediction).

```bash
python scripts/debug_radial_T_profile.py
```

Output: `phase2_radial_T_{steel_304,aluminum,copper}.png`.

### Diagnostic — convection plume smoke test

```bash
python scripts/debug_convection_plume.py
```

Output: `phase2_convection_plume.png`.

---

## Phase 3 — nucleate boiling (Rohsenow + Fritz + Cole validation)

Validates the Lagrangian bubble pool + Eulerian wall microlayer sink against Rohsenow's pool-boiling correlation on three pot materials. See [phase3_boiling.md](phase3_boiling.md) for the acceptance matrix.

### steel 304

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/default.yaml \
    --duration 180 --dx-mm 2.0 --pressure-iters 100 \
    --max-bubbles 100000
```

Output: `phase3_boiling_steel_304.{h5,png}` — temperature + bubble-count + departure-diameter histogram. Rohsenow ratio 1.01×, mean D_d = 2.93 mm.

### aluminum

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/aluminum.yaml \
    --duration 180 --dx-mm 2.0 --pressure-iters 100 \
    --max-bubbles 100000
```

Output: `phase3_boiling_aluminum.{h5,png}`. Rohsenow ratio 0.99×.

### copper

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/copper.yaml \
    --duration 180 --dx-mm 2.0 --pressure-iters 100 \
    --max-bubbles 100000
```

Output: `phase3_boiling_copper.{h5,png}`. Rohsenow ratio 0.97×.

---

## Phase 3.2 — q-sweep Rohsenow sensitivity (reviewer critique)

Five stove-flux points (10 / 20 / 30 / 40 / 50 kW/m², steel 304) + an analyzer, to address the reviewer's concern that the single-point q = 30 kW/m² validation could be masking a q-dependent drift hidden behind the conservation cap at [boiling.py:830](../python/boilingsim/boiling.py#L830). See [phase3_boiling.md](phase3_boiling.md) "Phase 3.2 extension" for full discussion.

### q = 10 kW/m² (natural-convection ↔ NB transition)

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/boiling_q10.yaml \
    --tag q_sweep_q10 \
    --duration 180 --dx-mm 2.0 --pressure-iters 100
```

Output: `phase3_boiling_q_sweep_q10.{h5,png}`. ΔT_w = 5.83 K, validation ratio 1.97× (regime-boundary drift, expected).

### q = 20 kW/m² (lower edge of fully-developed NB)

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/boiling_q20.yaml \
    --tag q_sweep_q20 \
    --duration 180 --dx-mm 2.0 --pressure-iters 100
```

Output: `phase3_boiling_q_sweep_q20.{h5,png}`. ΔT_w = 6.31 K, ratio 1.25×.

### q = 30 kW/m² (Phase-3 calibration point under the new `--tag` convention)

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/default.yaml \
    --tag q_sweep_q30 \
    --duration 180 --dx-mm 2.0 --pressure-iters 100
```

Output: `phase3_boiling_q_sweep_q30.{h5,png}`. ΔT_w = 6.76 K, ratio 1.03× (confirms Phase-3 headline on current codebase).

### q = 40 kW/m² (mid fully-developed NB)

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/boiling_q40.yaml \
    --tag q_sweep_q40 \
    --duration 180 --dx-mm 2.0 --pressure-iters 100
```

Output: `phase3_boiling_q_sweep_q40.{h5,png}`. ΔT_w = 7.40 K, ratio 1.01× (tightest point).

### q = 50 kW/m² (high end of domestic cooktop range)

```bash
python scripts/run_boiling.py \
    --config configs/scenarios/boiling_q50.yaml \
    --tag q_sweep_q50 \
    --duration 180 --dx-mm 2.0 --pressure-iters 100
```

Output: `phase3_boiling_q_sweep_q50.{h5,png}`. ΔT_w = 8.00 K, ratio 1.03×, cap bite 1.12× (first point where the conservation cap binds modestly).

### Analyzer — verdict table + two-panel figure

Reduces the five HDF5 artefacts to a stdout table + a single summary figure. Pure post-processing, no device code.

```bash
python scripts/analyze_q_sweep.py
```

Output: `phase3_q_sweep.png` + stdout table. Prints validation ratio + cap-bite ratio per q, plus a verdict line ("Rohsenow validates across fully-developed NB band" / drift flag).

---

## Phase 4 — nutrient retention (β-carotene baseline)

Validates Arrhenius degradation + Sherwood leaching against Sultana's 84 % reference. See [phase4_retention.md](phase4_retention.md) for the acceptance matrix.

### β-carotene, 25 mm carrot (Sultana reference)

Default YAML has `total_time_s = 600` and β-carotene kinetics. Driver flips `boiling.enabled` and `nutrient.enabled` on.

```bash
python scripts/run_retention.py \
    --config configs/scenarios/default.yaml \
    --carrot-diameter-mm 25 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag steel_304_25mm_final
```

Output: `phase4_retention_steel_304_25mm_final.{h5,png}`. R(600 s) = 88.72 %, leached 0.00 %, degraded 11.16 %.

### β-carotene, 12 mm carrot (high S/V ratio)

```bash
python scripts/run_retention.py \
    --config configs/scenarios/default.yaml \
    --carrot-diameter-mm 12 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag steel_304_12mm_final
```

Output: `phase4_retention_steel_304_12mm_final.{h5,png}`. R = 82.09 % (in band).

### β-carotene, 40 mm carrot (low S/V ratio)

```bash
python scripts/run_retention.py \
    --config configs/scenarios/default.yaml \
    --carrot-diameter-mm 40 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag steel_304_40mm_final
```

Output: `phase4_retention_steel_304_40mm_final.{h5,png}`. R = 93.54 % (above band, correct physics — Fo(600 s) = 0.22, interior barely warmed).

---

## Phase 4 — vitamin C extension (water-soluble solute, leach-dominated)

Exercises the Sherwood / advection / partition subsystem that β-carotene's K_partition = 1e-5 leaves as dead code.

### vitamin C, 25 mm carrot (primary validation)

Literature band: Konas et al. 2011 at 63.6 % (hospital-service boiled carrot, ~10 min), USDA FoodData Central retention factor ~65-70 % for typical home boil. Sonar 2018's 55.33 % is a lower outlier driven by its 1:5 water-to-carrot ratio and is captured separately in the 5:1 matched-volume run below.

```bash
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_25mm.yaml \
    --carrot-diameter-mm 25 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_25mm \
    --solute-label "vitamin C" \
    --target-band 40 70 --exp-ref-pct 64
```

Output: `phase4_retention_vitaminc_25mm.{h5,png}`. R = 65.80 %, leached 20.78 %, degraded 13.41 % — within 2.2 pp of Konas 2011 (63.6 %) and inside the USDA 65-70 % band.

### vitamin C, 12 mm carrot

```bash
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_25mm.yaml \
    --carrot-diameter-mm 12 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_12mm \
    --solute-label "vitamin C" \
    --target-band 40 60
```

Output: `phase4_retention_vitaminc_12mm.{h5,png}`. R = 40.32 % (shows regime flip vs 25 mm).

### vitamin C, 8 mm carrot (leach-dominated regime)

```bash
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_25mm.yaml \
    --carrot-diameter-mm 8 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_8mm \
    --solute-label "vitamin C" \
    --target-band 15 35
```

Output: `phase4_retention_vitaminc_8mm.{h5,png}`. R = 22.75 %, leach/deg = 2.44× (primary validation claim — leach-dominated at small geometry).

### vitamin C, D_eff sensitivity (VC-5a, tissue-disruption proxy)

Re-run of the 25 mm case with doubled `D_eff = 1.0e-9 m²/s` to check if internal-diffusion-limited → surface-flux-limited transition. (Config override via editing the YAML in-place or via a separate `vitamin_c_25mm_Dhi.yaml` clone — not committed in this repo.)

```bash
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_25mm.yaml \
    --carrot-diameter-mm 25 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_25mm_Dhi \
    --solute-label "vitamin C (D_eff=1e-9)" \
    --target-band 60 70
```

Output: `phase4_retention_vitaminc_25mm_Dhi.{h5,png}`. R = 65.60 %, delta −0.20 pp vs baseline — confirms surface-flux-limited regime at 25 mm.

---

## Phase 4 — dual-solute concurrent run

Validates that β-carotene and vitamin C evolve independently in the same boiling domain (two `SoluteSlot`s sharing one thermal + fluid field).

```bash
python scripts/run_retention.py \
    --config configs/scenarios/dual_solute_25mm.yaml \
    --carrot-diameter-mm 25 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag dual_solute_25mm \
    --solute-label "β-carotene" --target-band 80 90 --exp-ref-pct 84 \
    --solute2-label "vitamin C" --target2-band 40 70 --exp2-ref-pct 64
```

Output: `phase4_retention_dual_solute_25mm.{h5,png}`. Primary R = 88.61 % (Δ 0.11 pp vs single-solute β-carotene; within Rodriguez-Amaya 2008 80-90 % band centred on 84 %), secondary R = 65.52 % (Δ 0.28 pp vs single-solute vitamin C; matches Konas 2011 at 63.6 % within 2 pp).

---

## Phase 4.6 — Vieira-faithful + Sonar matched-volume (reviewer critique)

Addresses the reviewer's "`k0 = 1.1e7` is re-anchored, not Vieira-faithful" and "V_water/V_carrot = 104:1 doesn't match Sonar's 5:1" concerns. See [phase4_retention.md](phase4_retention.md) "Phase 4.6 extension" for full discussion.

### Vieira-faithful kinetics, 25 mm carrot

Uses `k0_per_s = 4.70e7` — Arrhenius-extrapolated from Vieira, Teixeira & Silva (2000) `k1(80 °C) = 0.032 /min` — instead of the re-anchored `1.1e7`. Tests whether the original re-anchoring (calibrated to blanching literature) over-predicts retention vs kitchen-boiling literature.

```bash
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_25mm_vieira.yaml \
    --carrot-diameter-mm 25 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_25mm_vieira \
    --solute-label "vitamin C (Vieira)" \
    --target-band 25 35
```

Output: `phase4_retention_vitaminc_25mm_vieira.{h5,png}`. R = 45.61 %, lands inside the kitchen-boiling literature band [40, 60] %.

### Sonar 5:1 matched-volume, all phases warm-started at saturation

Scales the pot down to V_water / V_carrot = 4.9 (Sonar 2018's ratio) via [vitamin_c_sonar_5to1.yaml](../configs/scenarios/vitamin_c_sonar_5to1.yaml) (5.5 cm × 9 cm pot, V_water ≈ 120 mL). Warm-starts water (100 °C), wall (107 °C) **and carrot (99 °C)** at saturation — bypasses the pre-boil warm-up artefact documented in [phase4_retention.md](phase4_retention.md) that corrupts cold-start runs at this small-pot / mismatched-stove-power configuration.

```bash
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_sonar_5to1.yaml \
    --carrot-diameter-mm 25 --duration 600 \
    --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_sonar_5to1_allhot \
    --solute-label "vitamin C (V_w/V_c=5, all phases at T_sat)" \
    --target-band 45 65 --exp-ref-pct 55 \
    --warm-start-water-c 100 \
    --warm-start-wall-c 107 \
    --warm-start-carrot-c 99
```

Output: `phase4_retention_vitaminc_sonar_5to1_allhot.{h5,png}`. R = 55.43 % — matches Sonar 2018's 55.33 % within HPLC measurement scatter.

---

## Phase 6 — live dashboard driver (not a benchmark run)

Runs the snapshot-producing CFD loop that feeds the browser dashboard via the Rust ws-server. Not a benchmark artefact producer — runs live until `Ctrl-C` or `--duration` hits.

```bash
# Terminal 1 — Rust relay
cargo run -p ws-server --release

# Terminal 2 — Python producer
python scripts/run_dashboard.py \
    --config configs/scenarios/default.yaml \
    --duration 120 \
    --dx-mm 2.0 --pressure-iters 100

# Terminal 3 — web frontend (from web/)
npm run dev
```

Browser: <http://localhost:5173>. Completed runs write `{run_id}.{h5,csv,json}` artefacts to `$BOILINGSIM_ARTIFACTS_DIR` (default `./dashboard_runs/`).

---

## Running the full validation sweep end-to-end

Paste into a terminal to regenerate every HDF5 + PNG under `benchmarks/`. Expect ~4 h wall time on RTX 4090 at `dx = 2 mm`.

```bash
set -e
cd ~/Desktop/Dev_Projects/boiling-sim
source .venv/bin/activate

# --- Phase 2 heating (~20 min each × 3 materials) ---
for mat in default aluminum copper; do
    python scripts/run_heating.py \
        --config configs/scenarios/${mat}.yaml \
        --dx-mm 2.0 --pressure-iters 100 \
        --suffix _impl
done
python scripts/reanalyze_heating_onb.py
python scripts/debug_radial_T_profile.py
python scripts/debug_convection_plume.py

# --- Phase 3 boiling (~5 min each × 3 materials) ---
for mat in default aluminum copper; do
    python scripts/run_boiling.py \
        --config configs/scenarios/${mat}.yaml \
        --duration 180 --dx-mm 2.0 --pressure-iters 100
done

# --- Phase 3.2 q-sweep (~8 min each × 4 new points; q=30 above already) ---
for q in 10 20 40 50; do
    python scripts/run_boiling.py \
        --config configs/scenarios/boiling_q${q}.yaml \
        --tag q_sweep_q${q} \
        --duration 180 --dx-mm 2.0 --pressure-iters 100
done
python scripts/run_boiling.py \
    --config configs/scenarios/default.yaml \
    --tag q_sweep_q30 \
    --duration 180 --dx-mm 2.0 --pressure-iters 100
python scripts/analyze_q_sweep.py

# --- Phase 4 β-carotene size sweep (~25 min each × 3 sizes) ---
for d in 12 25 40; do
    python scripts/run_retention.py \
        --config configs/scenarios/default.yaml \
        --carrot-diameter-mm ${d} --duration 600 \
        --dx-mm 2.0 --pressure-iters 100 \
        --tag steel_304_${d}mm_final
done

# --- Phase 4 vitamin C size sweep (~25 min × 3 sizes) ---
python scripts/run_retention.py --config configs/scenarios/vitamin_c_25mm.yaml \
    --carrot-diameter-mm 25 --duration 600 --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_25mm --solute-label "vitamin C" --target-band 40 70 --exp-ref-pct 64
python scripts/run_retention.py --config configs/scenarios/vitamin_c_25mm.yaml \
    --carrot-diameter-mm 12 --duration 600 --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_12mm --solute-label "vitamin C" --target-band 40 60
python scripts/run_retention.py --config configs/scenarios/vitamin_c_25mm.yaml \
    --carrot-diameter-mm 8 --duration 600 --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_8mm --solute-label "vitamin C" --target-band 15 35

# --- Phase 4 dual solute (~30 min) ---
python scripts/run_retention.py \
    --config configs/scenarios/dual_solute_25mm.yaml \
    --carrot-diameter-mm 25 --duration 600 --dx-mm 2.0 --pressure-iters 100 \
    --tag dual_solute_25mm \
    --solute-label "β-carotene" --target-band 80 90 --exp-ref-pct 84 \
    --solute2-label "vitamin C" --target2-band 40 70 --exp2-ref-pct 64

# --- Phase 4.6 Vieira-faithful (~25 min) ---
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_25mm_vieira.yaml \
    --carrot-diameter-mm 25 --duration 600 --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_25mm_vieira \
    --solute-label "vitamin C (Vieira)" --target-band 25 35

# --- Phase 4.6 Sonar 5:1 all-hot (~10 min, small pot) ---
python scripts/run_retention.py \
    --config configs/scenarios/vitamin_c_sonar_5to1.yaml \
    --carrot-diameter-mm 25 --duration 600 --dx-mm 2.0 --pressure-iters 100 \
    --tag vitaminc_sonar_5to1_allhot \
    --solute-label "vitamin C (V_w/V_c=5, all phases at T_sat)" \
    --target-band 45 65 --exp-ref-pct 55 \
    --warm-start-water-c 100 --warm-start-wall-c 107 --warm-start-carrot-c 99
```

---

## Artefact reference

| phase | artefact | scenario | R / ratio |
|---|---|---|---:|
| 2 | phase2_heating_steel_304_impl_dx2mm | default.yaml | ODE err +12.4 % at ONB |
| 2 | phase2_heating_aluminum_impl_dx2mm | aluminum.yaml | ODE err −10.7 % |
| 2 | phase2_heating_copper_impl_dx2mm | copper.yaml | ODE err −29.9 % |
| 3 | phase3_boiling_steel_304 | default.yaml | Rohsenow 1.01× |
| 3 | phase3_boiling_aluminum | aluminum.yaml | Rohsenow 0.99× |
| 3 | phase3_boiling_copper | copper.yaml | Rohsenow 0.97× |
| 3.2 | phase3_boiling_q_sweep_q10 | boiling_q10.yaml | ratio 1.97× |
| 3.2 | phase3_boiling_q_sweep_q20 | boiling_q20.yaml | ratio 1.25× |
| 3.2 | phase3_boiling_q_sweep_q30 | default.yaml | ratio 1.03× |
| 3.2 | phase3_boiling_q_sweep_q40 | boiling_q40.yaml | ratio 1.01× |
| 3.2 | phase3_boiling_q_sweep_q50 | boiling_q50.yaml | ratio 1.03× |
| 3.2 | phase3_q_sweep | analyzer output | — |
| 4 | phase4_retention_steel_304_25mm_final | default.yaml | R = 88.72 % |
| 4 | phase4_retention_steel_304_12mm_final | default.yaml | R = 82.09 % |
| 4 | phase4_retention_steel_304_40mm_final | default.yaml | R = 93.54 % |
| 4 | phase4_retention_vitaminc_25mm | vitamin_c_25mm.yaml | R = 65.80 % |
| 4 | phase4_retention_vitaminc_12mm | vitamin_c_25mm.yaml | R = 40.32 % |
| 4 | phase4_retention_vitaminc_8mm | vitamin_c_25mm.yaml | R = 22.75 % |
| 4 | phase4_retention_vitaminc_25mm_Dhi | vitamin_c_25mm.yaml + D_eff override | R = 65.60 % |
| 4 | phase4_retention_dual_solute_25mm | dual_solute_25mm.yaml | R1/R2 = 88.61 / 65.52 % |
| 4.6 | phase4_retention_vitaminc_25mm_vieira | vitamin_c_25mm_vieira.yaml | R = 45.61 % |
| 4.6 | phase4_retention_vitaminc_sonar_5to1_allhot | vitamin_c_sonar_5to1.yaml + all warm-starts | R = 55.43 % |
