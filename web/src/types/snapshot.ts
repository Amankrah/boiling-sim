// TypeScript mirror of crates/ws-server/src/snapshot.rs.
//
// Keep this file in lockstep with the Rust struct AND
// python/boilingsim/dashboard.py -- a schema bump is a cross-stack
// commit. SCHEMA_VERSION must equal the Rust `SCHEMA_VERSION` const.
//
// v2 (superseded): adds full Phase-4 four-bucket mass partition
// (retention + leached + degraded + precipitated) for both solutes
// + nutrient display names + a `set_nutrient` ControlMessage.
//
// v3 (superseded — Phase 6.6 data-forward upgrade): surfaces water
// temperature (mean/max/min) that Phase 4's pipeline already
// computed but that was never on the wire. Adds run_id (uuid per
// rebuild), total_time_s (target duration), is_complete (true once
// the run finished and artefacts were written), last_error
// (Pydantic validation failures from set_config messages).
//
// v4 (superseded): echoes the running sim's pot geometry
// (diameter, height, wall + base thickness) so the 3D <Pot> component
// can scale to match whatever the user picked on the Configuration
// page. Previously the pot was hardcoded at 20 cm × 12 cm regardless
// of cfg.pot.
//
// v5 (superseded): the temperature and alpha fields arrive as
// msgpack `bin` chunks (raw little-endian f32 bytes) rather than
// `array<f32>`. The decoder in useSnapshot.ts reinterprets the
// Uint8Array payload as a Float32Array before the snapshot reaches
// React state.
//
// v6 (this — multi-carrot pose): adds carrot pose / quantity fields
// so the dashboard can render N cylinders laying flat in the pot
// (realistic stew) rather than one hardcoded vertical procedural
// cylinder. carrot_count is the instance count, carrot_axis is the
// cylinder axis (0=x, 1=y, 2=z), carrot_centres is one anchor per
// instance, and carrot_total_mass_g is the derived UX quantity for
// the Config page's live readout.
export const SCHEMA_VERSION = 6;

export interface GridMeta {
  nx: number;
  ny: number;
  nz: number;
  dx: number;
  origin: [number, number, number];
}

export interface BubbleState {
  position: [number, number, number];
  radius: number;
}

export interface Snapshot {
  version: number;
  t_sim: number;
  step: number;
  is_rebuilding: boolean;
  is_paused: boolean;
  grid: GridMeta;
  grid_ds: GridMeta;
  // v5: msgpack `bin` on the wire; useSnapshot.ts converts the
  // decoded Uint8Array into a Float32Array view before exposing
  // the snapshot to React. Length equals
  // `grid_ds.nx * grid_ds.ny * grid_ds.nz`.
  temperature: Float32Array;
  alpha: Float32Array;
  bubbles: BubbleState[];
  // --- nutrient identity (v2) ---
  nutrient_primary_name: string;
  nutrient_secondary_name: string;
  // --- primary solute four-bucket mass partition (v2) ---
  carrot_retention: number;
  carrot_leached: number;
  carrot_degraded: number;
  carrot_precipitated: number;
  // --- secondary solute four-bucket mass partition (v2) ---
  carrot_retention2: number;
  carrot_leached2: number;
  carrot_degraded2: number;
  carrot_precipitated2: number;
  carrot_surface_c: number[];
  carrot_surface_c2: number[];
  wall_temperature_mean: number;
  wall_heat_flux: number;
  // --- v3: thermal detail + run metadata ---
  water_temperature_mean: number;
  water_temperature_max: number;
  water_temperature_min: number;
  run_id: string;
  total_time_s: number;
  is_complete: boolean;
  last_error: string;
  // --- v4: pot geometry echo (metres) ---
  pot_diameter_m: number;
  pot_height_m: number;
  pot_wall_thickness_m: number;
  pot_base_thickness_m: number;
  // --- v6: carrot pose / quantity ---
  /** Number of carrot instances; ``carrot_centres`` has this length. */
  carrot_count: number;
  /** Cylinder axis: 0=x, 1=y, 2=z. x/y mean horizontal, z is vertical. */
  carrot_axis: number;
  carrot_diameter_m: number;
  carrot_length_m: number;
  /** World-space anchor per carrot instance. For axis=2 (z) this is
   *  the cylinder base; for axis 0/1 (x or y) it's the cylinder centre. */
  carrot_centres: [number, number, number][];
  /** Total carrot mass in grams (count·π·(d/2)²·L·ρ_carrot, ρ≈1040). */
  carrot_total_mass_g: number;
}

/**
 * Scalar-only projection of a snapshot, retained in the browser's
 * history ring for time-series plots. Strips the volume arrays
 * (`temperature`, `alpha`, `carrot_surface_c*`) and the bubble list
 * -- those are orders of magnitude larger than we need for plotting
 * and hoarding them for 60 s of 30 Hz frames blows the JS heap past
 * "not enough memory to open this page".
 */
export interface SnapshotSummary {
  t_sim: number;
  step: number;
  is_rebuilding: boolean;
  is_paused: boolean;
  bubbles_count: number;
  // Primary solute four-bucket partition.
  carrot_retention: number;
  carrot_leached: number;
  carrot_degraded: number;
  carrot_precipitated: number;
  // Secondary solute four-bucket partition.
  carrot_retention2: number;
  carrot_leached2: number;
  carrot_degraded2: number;
  carrot_precipitated2: number;
  // Nutrient identity (short strings; cheap to retain in the ring).
  nutrient_primary_name: string;
  nutrient_secondary_name: string;
  wall_temperature_mean: number;
  wall_heat_flux: number;
  // v3: water T mean is needed for the Live-view overlay, the new
  // Recharts water-T line in the Results trajectory plot, and the
  // "water T pinned at saturation" exit-check audit. Max/min kept
  // off the summary to keep the ring light; they're available on
  // the full snapshot state for the SceneOverlay's range display.
  water_temperature_mean: number;
  // Run metadata is constant within a run but cheap to retain so
  // the plots can label traces and the Results page can compute
  // progress from summaries alone.
  run_id: string;
  total_time_s: number;
  is_complete: boolean;
}

export function summarizeSnapshot(s: Snapshot): SnapshotSummary {
  return {
    t_sim: s.t_sim,
    step: s.step,
    is_rebuilding: s.is_rebuilding,
    is_paused: s.is_paused,
    bubbles_count: s.bubbles.length,
    carrot_retention: s.carrot_retention,
    carrot_leached: s.carrot_leached,
    carrot_degraded: s.carrot_degraded,
    carrot_precipitated: s.carrot_precipitated,
    carrot_retention2: s.carrot_retention2,
    carrot_leached2: s.carrot_leached2,
    carrot_degraded2: s.carrot_degraded2,
    carrot_precipitated2: s.carrot_precipitated2,
    nutrient_primary_name: s.nutrient_primary_name,
    nutrient_secondary_name: s.nutrient_secondary_name,
    wall_temperature_mean: s.wall_temperature_mean,
    wall_heat_flux: s.wall_heat_flux,
    water_temperature_mean: s.water_temperature_mean,
    run_id: s.run_id,
    total_time_s: s.total_time_s,
    is_complete: s.is_complete,
  };
}

export type NutrientPreset = "beta_carotene" | "vitamin_c" | "both";

/**
 * Shape of the config blob sent on a `set_config` message. The
 * Python side Pydantic-validates it; any Pydantic field name is
 * acceptable here. We keep it `unknown`-ish (JSON value) rather
 * than mirroring every Pydantic model so the TS type stays loose
 * and the Configuration form drives its own typed form state
 * independent of this wire shape.
 */
export type ScenarioConfigJson = Record<string, unknown>;

export type ControlMessage =
  | { type: "set_heat_flux"; value: number }
  | { type: "set_material"; value: "steel_304" | "copper" | "aluminum" }
  | { type: "set_carrot_size"; diameter_mm: number; length_mm: number }
  | { type: "set_nutrient"; value: NutrientPreset }
  // Phase 6.6 v3: staged full-config apply from the Configuration page.
  | { type: "set_config"; config: ScenarioConfigJson }
  // Phase 6.6 v3: begin a timed run. Usually follows `set_config`.
  | { type: "start_run"; duration_s: number }
  // Phase 6.6 v3: emit artefacts mid-run without stopping.
  | { type: "export_snapshot" }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "reset" }
  // Stop the run mid-flight, write the partial-history artefacts, and
  // flip is_complete so the Results page becomes available. Distinct
  // from export_snapshot (which keeps stepping) and reset (which
  // discards everything).
  | { type: "finalize" }
  | { type: "request_full_snapshot" };

export type ConnectionState = "connecting" | "open" | "closed" | "error";
