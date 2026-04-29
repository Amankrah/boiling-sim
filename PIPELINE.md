# Boiling-Sim Pipeline — Step-by-Step Technical Walkthrough

This document traces a single end-to-end execution of the multiphysics boiling
simulator, from a YAML scenario on disk through the GPU step loop, the Rust
relay, and the React dashboard. It is intended for a new developer joining the
codebase: each section names the concrete file, function, and (where useful)
the equation being solved.

> Code references use repo-relative paths so they are clickable from the IDE,
> e.g. [pipeline.py:271](python/boilingsim/pipeline.py#L271).

---

## 0. System Map

There are three processes that talk over TCP/WebSocket:

```
┌───────────────────────────┐    msgpack frames     ┌──────────────────────┐    zstd ws frames    ┌───────────────────────┐
│  Python (GPU producer)    │  ──── tcp/8765  ───▶  │  Rust ws-server      │  ──── ws/8080 ────▶  │  React dashboard      │
│  scripts/run_dashboard.py │                       │  (axum + tokio)      │                      │  (R3F + Recharts)     │
│  boilingsim/pipeline.py   │  ◀── tcp/8766 ─────   │  fan-out broker      │  ◀── ws/8080 ───── │  control messages     │
└───────────────────────────┘   newline-JSON        └──────────────────────┘                      └───────────────────────┘
                                control messages                │
                                                                │  GET /api/runs/...
                                                                ▼
                                                      ./dashboard_runs/{run_id}.{h5,csv,json}
```

The same Python solver is also driven offline by
[`scripts/run_heating.py`](scripts/run_heating.py),
[`scripts/run_boiling.py`](scripts/run_boiling.py), and
[`scripts/run_retention.py`](scripts/run_retention.py); those write HDF5
benchmark artefacts directly and never touch the network stack.

---

## 1. Scenario Input — YAML to validated config

**Entry point:**
[`boilingsim.config.load_scenario`](python/boilingsim/config.py#L346)

A run begins by reading a YAML file under
[`configs/scenarios/`](configs/scenarios/) (e.g.
[default.yaml](configs/scenarios/default.yaml)). The file is parsed by
`yaml.safe_load` and validated by Pydantic through nested
[`BaseModel`](python/boilingsim/config.py#L22) classes:

| Section            | Class                                                                  | Holds                                                          |
| ------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| `pot`              | [`PotConfig`](python/boilingsim/config.py#L22)                         | diameter, height, wall/base thickness, material name           |
| `water`            | [`WaterConfig`](python/boilingsim/config.py#L38)                       | fill fraction, initial temp                                    |
| `carrot`           | [`CarrotConfig`](python/boilingsim/config.py#L43)                      | diameter, length, position, β-carotene loading                 |
| `heating`          | [`HeatingConfig`](python/boilingsim/config.py#L50)                     | base heat-flux W/m², ambient T                                 |
| `initial_conditions` | [`InitialConditionsConfig`](python/boilingsim/config.py#L55)         | `cold` vs `preheat` start mode                                 |
| `grid`             | [`GridConfig`](python/boilingsim/config.py#L78)                        | dx (m), carrot tet-mesh resolution                             |
| `solver`           | [`SolverConfig`](python/boilingsim/config.py#L83)                      | CFL safety, pressure / diffusion tols, BE switch, evap knobs   |
| `boiling`          | [`BoilingConfig`](python/boilingsim/config.py#L126)                    | nucleation, fragmentation, coalescence, Rohsenow C_sf          |
| `nutrient` / `nutrient2` | [`NutrientConfig`](python/boilingsim/config.py#L221)             | Arrhenius + Sherwood + partition coefficients per solute       |

Cross-field validators
([`_carrot_fits_inside_pot`](python/boilingsim/config.py#L330),
[`_nutrient2_requires_primary`](python/boilingsim/config.py#L321),
[`_wall_thinner_than_radius`](python/boilingsim/config.py#L29)) reject
geometrically impossible setups before any GPU work happens.

The result is a single `ScenarioConfig` value passed by reference everywhere
downstream.

---

## 2. Geometry — voxelising the pot

**Entry point:**
[`build_pot_geometry(cfg, device="cuda:0")`](python/boilingsim/geometry.py#L270)

The simulation operates on a **MAC (Marker-And-Cell) staggered grid**: scalars
live at cell centres, velocities live on cell faces. The
[`Grid`](python/boilingsim/geometry.py#L196) dataclass owns every Warp array.

### 2.1 Grid sizing

[`compute_grid_dims`](python/boilingsim/geometry.py#L246) takes the pot's outer
diameter / height plus a 4-cell pad and rounds up to even `nx, ny`. With the
default `dx = 1 mm` and a 20×12 cm pot, this yields ≈ 5 M cells (≈ 6 GB VRAM
for 8 fields). The world-space `origin` puts the pot centred at `(x=0, y=0)`
with `z = 0` at the stove face.

### 2.2 Allocations

For each cell-centred field (`pot_sdf`, `water_alpha`, `T`, `p`, `mat`) and
each MAC face-velocity field (`ux : (nx+1, ny, nz)`, `uy`, `uz`) Warp
allocates a zero-initialised GPU array.

### 2.3 Static field population (Warp kernels)

| Kernel                                                                       | What it stamps into the grid                                                                       |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [`build_pot_sdf`](python/boilingsim/geometry.py#L62)                         | Signed distance to the pot wall (outer cylinder minus inner cylinder); `< 0` inside steel.         |
| [`init_water_volume_fraction`](python/boilingsim/geometry.py#L89)            | `α = 1` for cells inside the cavity below `water_line_z`, else `0`.                                |
| [`populate_material_ids`](python/boilingsim/geometry.py#L122)                | Stamps `MAT_FLUID / MAT_POT_WALL / MAT_AIR / MAT_CARROT` per cell with documented precedence.      |
| [`initialize_temperature`](python/boilingsim/geometry.py#L165)               | Sets `T(x)` per material from the YAML (`water.initial_temp_c`, `heating.ambient_temp_c`, etc.).   |

If `cfg.boiling.enabled`, a Lagrangian
[`BubblePool`](python/boilingsim/boiling.py#L242) is allocated next; if
`cfg.nutrient.enabled`, the per-cell concentration arrays `C` (carrot) and
`C_water` are allocated and seeded by
[`initialize_nutrient_field`](python/boilingsim/nutrient.py#L220).

The result: a fully populated `Grid` ready for time-stepping. Velocity and
pressure start at zero.

---

## 3. Material Properties

**Entry point:**
[`MaterialProps.from_scenario`](python/boilingsim/thermal.py#L58)

Reads [`data/materials.json`](data/materials.json) (a JSON file that supports
`#` comments via [`json_hash_comments.py`](python/boilingsim/json_hash_comments.py))
and packs four ordered float arrays — one entry per material ID:

```
rho   = [ρ_water, ρ_pot, ρ_air, ρ_carrot]   kg/m³
c_p   = [c_water, c_pot, c_air, c_carrot]   J/(kg·K)
k     = [k_water, k_pot, k_air, k_carrot]   W/(m·K)
```

Each array is mirrored to a Warp `wp.array` (`*_wp`) so kernels can index by
material ID without a Python roundtrip.

---

## 4. Workspaces — pre-allocated kernel scratch

| Workspace                                                                                              | Owned arrays                                                              |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| [`FluidWorkspace`](python/boilingsim/fluid.py#L500) (allocated by `allocate_fluid_workspace`)          | divergence, pressure ping-pong, advection scratch for u/v/w/T, `u_max_scalar` |
| [`ThermalWorkspace`](python/boilingsim/thermal.py#L530)                                                | three face-flux arrays, BE Jacobi old/iter buffers                        |
| [`NutrientWorkspace`](python/boilingsim/nutrient.py#L100)                                              | `C_work`, `C_water_tmp`, `precipitated_mass`, plus matching `*2` for the secondary solute |

All scratch is allocated once at `Simulation.__init__` (lines
[100–129](python/boilingsim/pipeline.py#L100-L129)) and reused every step.

---

## 5. The Time-Step Loop

**Entry point:**
[`Simulation.step()`](python/boilingsim/pipeline.py#L271)

Each call advances the coupled system by `dt`. The order matters; reordering
breaks one of the conservation invariants documented in the docstring.

### 5.1 Compute Δt — CFL-bounded timestep

[`Simulation.compute_dt`](python/boilingsim/pipeline.py#L233)

```
u_max  = max |u| over all MAC faces                          (advection CFL)
dt_cfl = dx / max(u_max, 1e-8)
dt_cap = cfg.solver.max_dt_s / cfg.solver.cfl_safety_factor
```

- Backward-Euler conduction is unconditionally stable, so the diffusion
  bound is dropped when `cfg.solver.use_implicit_conduction` (the default).
- When nutrient physics is on, `dx² / (6·D_eff)` is added as a fourth
  bound (rarely binds at dev resolutions).
- The final `dt = cfl_safety_factor · min(...)`.

`u_max` involves a host roundtrip; `BOILINGSIM_DT_REFRESH=N` lets you cache
it for `N` steps when smooth velocity fields make that safe.

### 5.2 Phase 1–2 — Semi-Lagrangian advection of u and T

`advect_all` ([fluid.py](python/boilingsim/fluid.py)) orchestrates four kernels:

1. [`extend_temperature_into_solids`](python/boilingsim/fluid.py#L338) — copies
   each non-fluid cell's T from a 6-neighbour fluid lookup so that a
   trilinear backtrace which grazes a solid cell never reads garbage.
2. [`advect_temperature`](python/boilingsim/fluid.py#L382) — for each fluid
   cell, samples velocity at the cell centre, traces back along
   `p_back = p − u·dt`, then trilinearly samples `T_ext` there and writes
   into `T_new`. Solids keep their real `T` (not `T_ext`).
3. [`advect_ux / advect_uy / advect_uz`](python/boilingsim/fluid.py#L415) — the
   same backtrace on the three face-velocity grids.

Because the grid is staggered, each velocity component lives on its own
lattice with its own `(ox, oy, oz)` half-integer offset; the trilinear
sampler in [`_tri_sample`](python/boilingsim/fluid.py#L37) compensates for
that.

### 5.3 Phase 3 — Boussinesq buoyancy

[`apply_buoyancy`](python/boilingsim/fluid.py#L169):

```
u_z += dt · g · β · (T_face − T_ref)            on internal fluid z-faces
```

with `β = 2.07e-4 K⁻¹` (water at 25 °C) and `T_ref = water.initial_temp_c`.
This is the only momentum source in single-phase mode and drives natural
convection.

### 5.4 Phase 3b — Bubble update (when `boiling.enabled`)

[`step_bubbles`](python/boilingsim/boiling.py#L1940) runs five sub-passes:

1. **`step_update_bubbles`** ([`update_bubbles`](python/boilingsim/boiling.py#L645) kernel) —
   for every active bubble:
   - Mikic-Rohsenow growth: `R(t) = √(7/π · Ja²·α_l·age)`
   - Fritz departure check: `R ≥ R_d = 0.0208·θ·√(σ/(g·Δρ))`
   - Cole frequency vent on age > `1/f_d`
   - Terminal-slip rise + advection by sampled fluid velocity
2. **`step_fragment_bubbles`** — Rayleigh-Taylor split into two equal-volume
   daughters when `R > fragmentation_radius_m`.
3. **`step_coalesce_bubbles`** — spatial-hash O(N) merge pass with bin size
   `2·R_max`. Two close bubbles fuse into a volume-conserving daughter.
4. **`step_scatter_latent_heat`** — bulk RPI sink: each bubble pulls
   `h_lv·dV/dt` of enthalpy from the surrounding superheated fluid.
5. **`step_scatter_momentum`** — vapour-rise reaction force on the
   adjacent fluid faces (drives bulk circulation).
6. **`step_reduce_water_alpha`** — resets `water_alpha` from the static
   baseline and subtracts each bubble's volume fraction (used by the
   3D renderer to show vapour pockets).
7. **`step_nucleation`** — [`detect_nucleation_sites`](python/boilingsim/boiling.py#L403)
   scans every inner-pot-wall cell whose `+z` neighbour is fluid; samples
   `N_a(ΔT_w)` from the precomputed Kocamustafaogullari-Ishii table
   ([`build_nucleation_table`](python/boilingsim/boiling.py#L90)) and
   probabilistically spawns new bubbles.

### 5.5 Phase 4 — Conjugate heat conduction + boundary sources

[`conduct_one_step`](python/boilingsim/thermal.py#L598) chooses between
explicit Euler and backward-Euler Jacobi based on
`cfg.solver.use_implicit_conduction`:

**Backward-Euler (default).**
[`apply_implicit_conduction_update`](python/boilingsim/thermal.py#L226) sweeps
every non-air cell with the Jacobi update

```
(1 + Σγ) · T_new = T_old + Σγ · T_nbr
γ_face = dt · k_face / (ρ·c_p·dx²)            (fluid–fluid / conjugate)
γ_face = dt · h_conv·dx / (ρ·c_p·dx²)         (solid ↔ air, with T_nbr → T_amb)
```

`k_face` is the **harmonic mean** of the two cell conductivities — this is
what makes conjugate heat transfer correct across a steel–water interface
(arithmetic mean would over-conduct). Air–solid faces are replaced by a
Newton-cooling effective `k = h_conv·dx`. `cfg.solver.diffusion_max_iter`
sweeps later, the result is in `grid.T`.

**Explicit Euler.** Three flux kernels
([`heat_conduction_flux_x/y/z`](python/boilingsim/thermal.py#L123)) followed
by [`apply_conduction_update`](python/boilingsim/thermal.py#L187) — used only
when `use_implicit_conduction = False`.

After the diffusion solve, three boundary kernels apply:

- **`apply_base_heat_flux`** ([thermal.py:357](python/boilingsim/thermal.py#L357)) —
  injects `q_base [W/m²]` into the *single layer* of pot-wall cells whose
  `−z` neighbour is air (the stove contact surface).
  `ΔT = q_base · dt / (ρ·c_p·dx)`.
- **`apply_evaporative_cooling`** (Phase 2 placeholder; only when boiling
  is off) — temperature-gated linear ramp from 0 at 85 °C to
  `0.1·q_base` at 100 °C on free-surface cells.
- When boiling is on, two extra sinks run instead:
  - **`apply_free_surface_evap_sink`** — pins the top fluid row to T_sat
    (modelling the open-pot enthalpy bleed our sealed domain otherwise
    misses).
  - **`apply_bulk_evap_sink`** — Newton relaxation `dT/dt = −f·(T−T_sat)`
    on every superheated fluid cell (lumps bulk nucleation that the
    wall-anchored bubble pool can't capture).

### 5.6 Phase 4b — Wall boiling (when `boiling.enabled`)

[`step_wall_boiling_flux`](python/boilingsim/boiling.py#L1814) calls
[`apply_wall_boiling_flux`](python/boilingsim/boiling.py#L1310). This is the
microlayer-evaporation sink: pot-wall cells at *active* nucleation sites
lose enthalpy proportional to the local superheat, capped at `q_stove` for
energy conservation. Together with the Lagrangian bulk scatter this is the
RPI partition: `q_total = q_nb + q_conv + q_quench`.

### 5.7 Phase 4c — Nutrient reaction-diffusion-leaching

When `cfg.nutrient.enabled`,
[`_step_reaction_diffusion_leach`](python/boilingsim/nutrient.py#L1139) runs
once per active solute slot:

1. **`arrhenius_degrade`** on `C` (carrot field) —
   `C ← C·exp(−k0·exp(−E_a/RT)·dt)` for every carrot cell.
2. **`arrhenius_degrade_water`** on `C_water` — same kinetics on solute
   that has already leached out (otherwise we'd over-retain post-leach
   mass).
3. **`diffuse_nutrient_explicit`** — explicit central-difference Laplacian
   inside the carrot, ping-ponged through `C_work`. Stability bound
   `dt ≤ dx²/(6·D_eff)` is asserted.
4. **`leach_at_surface`** — Sherwood mass-transfer at every face between
   a carrot cell and a fluid cell: `J = h_m · (C_carrot − C_water/K_p)`
   with `h_m = Sh·D_water/L`, `Sh = 2 + 0.6·Re^0.5·Sc^0.33` (Frössling).
   Output capped by the aqueous solubility `C_water_sat_mg_per_kg`.

Dual-solute mode runs the same sequence on the secondary `slot.C2 / C_water2`
arrays driven by `cfg.nutrient2`.

### 5.8 Phase 5 — No-slip + pressure projection

[`enforce_no_slip`](python/boilingsim/fluid.py#L153) zeros every MAC face that
touches a non-fluid cell. Done **twice** — once before pressure projection
(so the divergence we measure is on the constrained field) and once after
(numerical drift guard).

[`pressure_projection`](python/boilingsim/fluid.py) is the heart of
incompressible flow:

1. **`compute_divergence`** — `∇·u` at every fluid cell-centre.
2. **Jacobi sweeps** of `∇²p = (ρ/dt)·∇·u` via
   [`jacobi_pressure_step`](python/boilingsim/fluid.py#L242), with mixed BCs:
   - fluid neighbour → use its pressure
   - air neighbour (free surface) → Dirichlet `p = 0`
   - solid neighbour → Neumann `∂p/∂n = 0` (ghost cell `= self`)
3. **`subtract_pressure_gradient_x/y/z`** — `u ← u − (dt/ρ)·∇p` on every
   internal fluid face, leaving a divergence-free velocity field.

Iterations bounded by `cfg.solver.pressure_max_iter` (default 200).

### 5.9 Phase 6 — Passive-scalar advection (water-side solute)

After projection, the water-side solute concentration `C_water` is advected
by the freshly divergence-free velocity using a **conservative upwind**
scheme ([`advect_c_water`](python/boilingsim/nutrient.py#L753)) and then
clamped at `C_water_sat_mg_per_kg` by
[`clamp_c_water_and_track_precipitation`](python/boilingsim/nutrient.py#L884).
Mass clipped by the cap is credited to a `precipitated_mass` atomic counter
so the four-channel partition (retention + leached + degraded + precipitated)
sums to 100 % every sample — giving an in-built mass-conservation alarm.

### 5.10 Bookkeeping

`self.t += dt; self.step_count += 1`. Per-phase wall times accumulate when
`BOILINGSIM_PROFILE=1` (each phase wraps in
[`_profile_phase`](python/boilingsim/pipeline.py#L188), bracketed by
`wp.synchronize_device`).

---

## 6. Diagnostics — `ScalarSample`

**Entry point:**
[`Simulation.sample_scalars`](python/boilingsim/pipeline.py#L385)

Once per N steps the driver pulls a flat snapshot of integral metrics:

| Field                            | Source                                                                |
| -------------------------------- | --------------------------------------------------------------------- |
| `T_mean / max / min_water_c`     | mean / max / min over `_water_mask`                                   |
| `T_max_wall_c`                   | max over `_wall_mask`                                                 |
| `T_inner_wall_mean_c / max_c`    | restricted to the fluid-contact face (Rohsenow-relevant superheat)    |
| `u_max_mps`                      | reduction kernel in `compute_max_velocity`                            |
| `n_active_bubbles`, `*_R_mm`     | host-side reduction over the bubble pool                              |
| `alpha_min`                      | min `water_alpha` over fluid cells (1.0 = no vapour)                  |
| `retention_pct / leached_pct / degraded_pct / precipitated_pct` | 100·Σ C_carrot/Σ C₀, etc. — must sum to 100 |
| `retention2_pct / ...`           | identical four-channel partition for the secondary solute             |

`ScalarSample` is a `@dataclass` — every field is float-typed and serialisable.

---

## 7. Two Run Modes

### 7.1 Offline benchmark mode

[`Simulation.run`](python/boilingsim/pipeline.py#L513) drives the loop until
`total_time_s`, sampling scalars every N steps and writing **HDF5** with
`scalars/*` (full time series), `snapshots/T` (downsampled volumetric
temperature), `bubble_snapshots/{radii_m, positions_m}` (variable-length
datasets), and `meta` attributes. This is what
[`scripts/run_heating.py`](scripts/run_heating.py) (Phase-2 sensible heating
vs lumped-capacitance ODE), [`scripts/run_boiling.py`](scripts/run_boiling.py)
(Phase-3 Rohsenow check + Fritz departure histogram), and
[`scripts/run_retention.py`](scripts/run_retention.py) (Phase-4 nutrient
retention vs published kinetics) all consume.

### 7.2 Live-dashboard mode

[`scripts/run_dashboard.py`](scripts/run_dashboard.py) replaces the run loop
with its own variant that:

- starts **paused** at `t=0`;
- drains incoming control messages from
  [`ControlConsumer`](python/boilingsim/dashboard.py#L397) (background TCP
  client connected to the Rust relay's port 8766) before each step;
- classifies them: *live-editable* (`set_heat_flux`) mutates the cfg in
  place; *rebuild-triggering* (`set_material`, `set_carrot_size`,
  `set_nutrient`, `set_config`, `reset`) tears down and recreates the
  whole `Simulation` after sending a "rebuilding" marker frame;
- runs `sim.step()`;
- emits a snapshot via [`SnapshotProducer.send_snapshot`](python/boilingsim/dashboard.py#L353)
  on a 30 Hz cadence (`snapshot_hz` flag);
- accumulates `ScalarSample`s in a bounded
  [`ScalarHistory`](python/boilingsim/run_writer.py#L43) ring;
- when `t_sim >= total_time_s` (or the user clicks Finish), calls
  [`write_run_artefacts`](python/boilingsim/run_writer.py#L102) which writes
  three sibling files under `dashboard_runs/`:
  - `{run_id}.h5` — full scalar time-series + parameter echo (HDF5)
  - `{run_id}.csv` — one row per sample (cheap to fetch from the browser)
  - `{run_id}.json` — final-state summary, acceptance gates, mass balance,
    plus the entire `ScenarioConfig` (`cfg.model_dump_json()`) so a future
    reader can reproduce the exact run.

---

## 8. Wire Format — Python ↔ Rust

### 8.1 Snapshot frame (Python → Rust → browser)

**Schema:** `SCHEMA_VERSION = 4` enforced on both sides.
- Python builder: [`build_snapshot`](python/boilingsim/dashboard.py#L135)
- Rust mirror: [`Snapshot`](crates/ws-server/src/snapshot.rs#L93)
- TS mirror: `web/src/types/snapshot.ts`

Encoding: **MessagePack** (`msgpack.packb(..., use_bin_type=True)`),
length-prefixed (`u32 big-endian`) on TCP.

Key payload fields:

| Field                                          | Purpose                                                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| `version`, `t_sim`, `step`                     | identification + clock                                                                |
| `is_rebuilding`, `is_paused`, `is_complete`    | lifecycle gates the UI uses to draw banners                                           |
| `grid`, `grid_ds`                              | full + 2× downsampled grid dims                                                       |
| `temperature[]`                                | downsampled `T` in Celsius, C-order (k-fastest)                                       |
| `alpha[]`                                      | downsampled water VOF (1 = water, 0 = vapour/air)                                     |
| `bubbles[]`                                    | `[{position, radius}]` for every active bubble                                        |
| `carrot_retention / leached / degraded / precipitated` | primary solute mass partition (sum ≈ 100 %)                                   |
| `carrot_*2`                                    | secondary solute partition                                                            |
| `wall_temperature_mean / wall_heat_flux`       | the two scalars displayed in the heat-flux ring                                       |
| `water_temperature_mean / max / min`           | live water row on the Live page                                                       |
| `pot_diameter_m / pot_height_m / ...`          | echo of `cfg.pot` so the 3D renderer can scale to match (added in v4)                 |
| `run_id`, `total_time_s`, `last_error`         | run lifecycle; `last_error` carries Pydantic validation failures back from `set_config` |

Volumes are downsampled by [`_downsample_halves`](python/boilingsim/dashboard.py#L88)
(stride-2 subsample) — keeps frames near 300 KB for a 5 M-cell grid.

### 8.2 Control frame (browser → Rust → Python)

JSON (chosen for inspectability), parsed by
[`ControlMessage`](crates/ws-server/src/control.rs#L17), an externally tagged
serde enum with variants `set_heat_flux`, `set_material`, `set_carrot_size`,
`set_nutrient`, `set_config`, `start_run`, `pause`, `resume`, `reset`,
`finalize`, `export_snapshot`, `request_full_snapshot`. The Rust forwarder
re-serialises each with [`to_json_line`](crates/ws-server/src/control.rs#L66)
so the Python consumer can simply split the TCP stream on `\n` and call
`json.loads`.

---

## 9. Rust Relay (`crates/ws-server/`)

**Entry point:** [`main.rs`](crates/ws-server/src/main.rs)

A single `tokio` runtime spawns three concurrent tasks all sharing one
[`AppState`](crates/ws-server/src/app.rs) (two `tokio::sync::broadcast`
channels):

```
ingest::run        listens on 127.0.0.1:8765   (Python → Rust)
control_forward    listens on 127.0.0.1:8766   (Rust → Python)
axum HTTP/WS       listens on 0.0.0.0:8080     (browser)
```

### 9.1 Ingest

[`ingest::handle_producer`](crates/ws-server/src/ingest.rs#L47):

1. Wraps the TCP stream in a `LengthDelimitedCodec` (max 16 MB, big-endian
   `u32` header).
2. For each frame, calls `Snapshot::from_msgpack_bytes` *just to validate
   the version* — payload bytes themselves are not re-serialised.
3. Pushes the raw bytes (wrapped in `Arc<Vec<u8>>`) into the
   `state.snapshots` broadcast channel. One allocation, fanned out to N
   subscribers.

### 9.2 WebSocket

[`ws::handle_socket`](crates/ws-server/src/ws.rs#L42) per client:

- Spawns a snapshot task that subscribes to the broadcast, zstd-encodes
  each frame at level 3, and writes `Message::Binary` to the socket.
  Compression happens **per-client** rather than once globally because
  the common case is exactly one client (the dashboard) and it keeps the
  ingest hot path uncompressed for future debug taps.
- Spawns a control task that reads `Message::Text`, parses
  `ControlMessage`, and pushes onto the `state.controls` broadcast.

Backpressure: the snapshot channel has capacity 64 (≈ 2 s at 30 Hz). A
client lagging by ≥ 5 frames triggers a `warn!`; lagging by less is
silently absorbed (browser GC / paint stalls are normal).

### 9.3 Control forwarder

[`control_forward::handle_consumer`](crates/ws-server/src/control_forward.rs#L39)
subscribes to `state.controls` and writes each as a newline-terminated JSON
line to the Python socket. Disconnects discard pending messages by design
(stale heat-flux commands shouldn't auto-replay when Python reconnects).

### 9.4 Run-artefact HTTP routes

[`runs.rs`](crates/ws-server/src/runs.rs) exposes:

```
GET /api/runs                       → JSON list of completed runs
GET /api/runs/{id}/summary.json     → stream summary (or "latest" alias)
GET /api/runs/{id}/scalars.csv      → stream the CSV
GET /api/runs/{id}/data.h5          → stream the raw HDF5
```

Run IDs are validated as 32-char lowercase hex (uuid simple format) or
the literal `"latest"`. Path resolution is via `artefact_dir_with_source()`
which prefers `BOILINGSIM_ARTIFACTS_DIR`, then walks up to the workspace
root looking for `[workspace]` in `Cargo.toml`, then falls back to
cwd-relative `./dashboard_runs`.

---

## 10. Web Frontend (`web/src/`)

**Entry point:** [`web/src/App.tsx`](web/src/App.tsx)

### 10.1 The single WebSocket — `useSnapshot`

[`hooks/useSnapshot.ts`](web/src/hooks/useSnapshot.ts) opens *one*
WebSocket at App level (so tab switches don't reconnect) and on every
message:

1. Treats the `ArrayBuffer` as zstd-compressed bytes.
2. Decompresses via [`fzstd`](https://www.npmjs.com/package/fzstd) (pure-JS).
3. msgpack-decodes via `@msgpack/msgpack` into a typed `Snapshot`.
4. Rejects mismatched `version`.
5. Sets the latest-snapshot React state and appends a `summarizeSnapshot()`
   *summary* (≈ 100 B) to a 1800-deep ring used by Recharts. The full
   snapshot (≈ 700 KB of arrays) is never retained between frames — only
   the previous-frame React state which the reconciler drops on update.
6. Bumps `historyVersion` at most every 200 ms (5 Hz) to throttle chart
   re-renders.

`sendCommand(cmd)` simply `JSON.stringify`s a `ControlMessage` and writes
it as a text frame.

### 10.2 Three pages, one router

[`hooks/usePage.ts`](web/src/hooks/usePage.ts) exposes a `?page=` URL
parameter. App.tsx renders one of:

- **`LivePage`** ([web/src/pages/LivePage.tsx](web/src/pages/LivePage.tsx))
  — `BoilingScene` (R3F canvas: `Pot`, `WaterVolume`, `Bubbles`,
  `CarrotMesh`, `Stove`, `GradientBackground`) + `SceneOverlay` +
  `ControlPanel` (heat-flux slider, material picker, carrot size) +
  `TimeSeriesPanel` (Recharts strip).
- **`ConfigPage`** ([web/src/pages/ConfigPage.tsx](web/src/pages/ConfigPage.tsx))
  — `ConfigForm` builds a full `ScenarioConfig` JSON blob, sends it via
  `set_config` + `start_run`. Validation errors come back via
  `snapshot.last_error`.
- **`ResultsPage`** ([web/src/pages/ResultsPage.tsx](web/src/pages/ResultsPage.tsx))
  — uses [`useRunArtefacts`](web/src/hooks/useRunArtefacts.ts) to fetch
  `/api/runs/latest/summary.json` and `.../scalars.csv` in parallel,
  then renders the Phase-4-style `ResultsReport` (HeatUpStorylineCard,
  BoilingVigorCard, FinalPartitionDonutCard, NutrientLossRateCard).

### 10.3 URL share state

`share.ts` round-trips a `{params, camera}` object through `?s=base64(json)`,
so a deep-linked URL reproduces the exact slider / orbit-control state on
reload — and the seeding effect at App.tsx:80 pushes those params back to
Python on the first WS open so Python and the URL never disagree.

---

## 11. Lifecycle of a Single User-Driven Run

Walking through the dashboard happy path:

1. `docker compose up` brings up `ws-server`, `solver`, `web` (or you start
   them locally: `cargo run -p ws-server`, `python scripts/run_dashboard.py`,
   `npm run dev`).
2. User opens `http://localhost:3000`. The browser establishes a single
   WebSocket to `/stream`. The Rust relay logs `ws client connected`.
3. Python is paused at `t=0`. The relay forwards the first snapshot frame
   (which carries `is_paused=true, t_sim=0`) and the UI shows
   `IdleBanner`.
4. User clicks **Configure** → fills in pot/water/carrot/heating fields →
   clicks **Apply & Start Run** with `duration = 600 s`.
5. `ConfigForm` issues `{type:"set_config", config:{...}}` followed by
   `{type:"start_run", duration_s:600}` over the WS.
6. The Rust WS handler parses both, broadcasts onto `state.controls`, and
   the control-forwarder pipes them as newline-JSON to Python.
7. Python's `ControlConsumer.drain()` returns both. `set_config` triggers
   a full `Simulation` rebuild (after sending a "rebuilding" marker frame
   so the browser shows a spinner); `start_run` flips `paused = False` and
   resets `total_time_s`.
8. The step loop now runs at GPU speed; every 1/30 s
   `SnapshotProducer.send_snapshot` emits a new msgpack frame; the relay
   broadcasts; the browser decompresses → decodes → re-renders the
   `BoilingScene` + charts.
9. The user can drag the heat-flux slider mid-run. `ControlPanel` debounces
   to ~10 Hz and emits `set_heat_flux`; Python applies it live without
   rebuilding. The continuous-mirror effect at App.tsx:113 keeps the
   slider in lock-step with `snapshot.wall_heat_flux`.
10. At `t_sim ≥ 600 s`, the dashboard driver's `finalize_run()` calls
    `write_run_artefacts`, emits a final snapshot with `is_complete=true`,
    and pauses the loop.
11. The browser sees `is_complete=true`, lights up the "New results" badge
    on the Results tab, and the `ResultsPage` mount (when clicked) fetches
    the three artefacts via `/api/runs/latest/...` and renders the
    Phase-4 report.

---

## 12. File Map — Where to Look First

| You want to change…                       | Open this first                                                              |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| Stove power, pot dimensions, carrot       | [`configs/scenarios/*.yaml`](configs/scenarios/)                              |
| Add a new YAML field                      | [`config.py`](python/boilingsim/config.py) (Pydantic) + the consumer site     |
| How the SDF / material IDs are stamped    | [`geometry.py`](python/boilingsim/geometry.py)                                |
| Pressure projection / SL advection        | [`fluid.py`](python/boilingsim/fluid.py)                                      |
| BE Jacobi conduction / Newton cooling     | [`thermal.py`](python/boilingsim/thermal.py)                                  |
| Bubble physics (nucleation, Mikic, Fritz) | [`boiling.py`](python/boilingsim/boiling.py)                                  |
| Arrhenius / Fick / Sherwood               | [`nutrient.py`](python/boilingsim/nutrient.py)                                |
| The step ordering itself                  | [`pipeline.py:271`](python/boilingsim/pipeline.py#L271) (`Simulation.step`)   |
| HDF5 / CSV / JSON layout                  | [`run_writer.py`](python/boilingsim/run_writer.py)                            |
| Wire schema (msgpack)                     | [`dashboard.py`](python/boilingsim/dashboard.py) + [`snapshot.rs`](crates/ws-server/src/snapshot.rs) |
| Add a new control message type            | [`control.rs`](crates/ws-server/src/control.rs) + handler in [`run_dashboard.py`](scripts/run_dashboard.py) |
| HTTP run-artefact route                   | [`runs.rs`](crates/ws-server/src/runs.rs)                                     |
| WebSocket plumbing                        | [`ws.rs`](crates/ws-server/src/ws.rs)                                         |
| Frontend WebSocket / msgpack hook         | [`hooks/useSnapshot.ts`](web/src/hooks/useSnapshot.ts)                        |
| 3D rendering (R3F)                        | [`components/BoilingScene.tsx`](web/src/components/BoilingScene.tsx)          |
| Results-page report cards                 | [`components/ResultsReport/`](web/src/components/ResultsReport/)              |

---

## 13. Glossary of Invariants

These properties are checked by tests and/or asserted at runtime; if you
break one, you've likely introduced a bug:

- **MAC staggering.** Velocities live on faces, scalars at centres. Every
  kernel that mixes them must use the right `(ox, oy, oz)` half-integer
  offsets in `_tri_sample`.
- **Material precedence in `populate_material_ids`.** Pot wall → carrot
  → water → air. Reordering would let carrot cells overwrite the wall.
- **Pressure projection BCs.** Fluid neighbour = use value, air = `p=0`,
  solid = ghost equals self. Same divisor (6) regardless because the
  Neumann ghost contributes `p_self` to the sum.
- **No-slip enforced twice per step.** Once before projection, once after.
- **Mass partition sums to 100 %.** `retention + leached + degraded +
  precipitated = 100`. A drift indicates the upwind advection / clamp /
  Arrhenius/Sherwood pipeline lost or invented mass; `degraded_pct` is
  intentionally signed so this is visible in plots.
- **Schema version.** Bumping `SCHEMA_VERSION` requires touching
  `dashboard.py`, `snapshot.rs`, and `web/src/types/snapshot.ts` in the
  same commit.
- **Reset on rebuild.** Every `set_material / set_carrot_size / set_config /
  reset` issues a brand-new `run_id`, clears history, and resets
  `is_complete`. Don't reuse the previous `run_id` — artefact filenames
  collide.

---

## 14. Quick Performance Notes

- Default `dx = 1 mm` ≈ 5 M cells, ~6 GB VRAM, ~3–5 ms/step on RTX 6000 Ada.
- `BOILINGSIM_PROFILE=1` enables per-phase wall timing
  ([`Simulation._profile_phase`](python/boilingsim/pipeline.py#L188)).
- `BOILINGSIM_DT_REFRESH=N` skips the `u_max` host-sync for `N` steps —
  use only with smooth velocity fields.
- Pressure projection dominates step time at this resolution; raise
  `cfg.solver.pressure_max_iter` cautiously (200 is the sweet spot in
  practice).
- The browser receives ≈ 300 KB/frame at 30 Hz (~9 MB/s). zstd at
  level 3 inflates to ~80 KB on the wire.

---

That is the full pipeline: **YAML → Pydantic config → Warp grid + workspaces →
GPU step loop (advect → buoyancy → bubbles → conduction + BCs → wall-boil →
nutrient → no-slip → pressure → no-slip → C_water advect) → ScalarSample →
HDF5 (offline) or msgpack/zstd (live) → Rust broadcast relay → WebSocket →
React/R3F dashboard → HDF5/CSV/JSON artefacts on completion.**
