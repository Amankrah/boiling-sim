// TypeScript mirror of boilingsim.config (Pydantic) models. Shapes
// match the JSON that `ScenarioConfig.model_dump(mode="json")`
// emits; Pydantic is authoritative on validation, so these types
// stay structural rather than trying to encode every `gt=0` /
// `le=1.0` constraint.

export type MaterialName = "steel_304" | "cast_iron" | "aluminum" | "copper";

export interface PotDraft {
  diameter_m: number;
  height_m: number;
  wall_thickness_m: number;
  base_thickness_m: number;
  material: MaterialName;
}

export interface WaterDraft {
  fill_fraction: number;
  initial_temp_c: number;
}

export interface CarrotDraft {
  diameter_m: number;
  length_m: number;
  position: [number, number, number];
  /** Number of identical carrot instances (1-64). count=1 keeps the
   *  legacy single-carrot behaviour; >1 triggers auto-placement. */
  count: number;
  /** Cylinder axis. "z" is legacy vertical; "x"/"y" are horizontal. */
  axis: "x" | "y" | "z";
  initial_beta_carotene_mg_per_100g: number;
}

export interface HeatingDraft {
  base_heat_flux_w_per_m2: number;
  ambient_temp_c: number;
}

export type InitialConditionsMode = "cold" | "preheat";

export interface InitialConditionsDraft {
  mode: InitialConditionsMode;
  preheat_water_c: number;
  preheat_wall_c: number;
  preheat_carrot_c: number;
}

/** Which solute(s) the pipeline tracks. `off` disables both slots;
 *  `both` enables β-carotene on slot 1 and vitamin C on slot 2 for the
 *  dual-solute validation. Picking a key fully populates the
 *  Arrhenius / diffusion / partition / solubility knobs from the
 *  canonical literature values below. */
export type SoluteKey = "off" | "beta_carotene" | "vitamin_c" | "both";

export interface GridDraft {
  dx_m: number;
  carrot_mesh_resolution: number;
}

export interface SolverDraft {
  cfl_safety_factor: number;
  max_dt_s: number;
  pressure_tol: number;
  pressure_max_iter: number;
  diffusion_tol: number;
  diffusion_max_iter: number;
  h_conv_outer_w_per_m2_k: number;
  h_evap_free_surface_w_per_m2_k: number;
  f_bulk_evap_per_s: number;
  use_implicit_conduction: boolean;
}

export interface BoilingDraft {
  enabled: boolean;
  dT_onb_k: number;
  contact_angle_rad: number;
  max_bubbles: number;
  initial_bubble_radius_m: number;
  nucleation_probability_per_step: number;
  C_sf_rohsenow: number;
  Pr_n_rohsenow: number;
}

export interface NutrientDraft {
  enabled: boolean;
  E_a_kJ_per_mol: number;
  k0_per_s: number;
  D_eff_m2_per_s: number;
  K_partition: number;
  C_water_sat_mg_per_kg: number;
  C0_mg_per_kg: number;
  nu_water_m2_per_s: number;
  D_water_molec_m2_per_s: number;
}

export interface SimulationDraft {
  total_time_s: number;
  output_every_s: number;
}

export interface ScenarioDraft {
  pot: PotDraft;
  water: WaterDraft;
  carrot: CarrotDraft;
  heating: HeatingDraft;
  initial_conditions: InitialConditionsDraft;
  grid: GridDraft;
  solver: SolverDraft;
  boiling: BoilingDraft;
  nutrient: NutrientDraft;
  nutrient2: NutrientDraft;
  simulation: SimulationDraft;
}

/** Canonical per-solute kinetic constants. One source of truth for the
 *  frontend preset dropdown; matches the `_BETA_CAROTENE` / `_VITAMIN_C`
 *  dicts in scripts/run_dashboard.py. */
export const BETA_CAROTENE_PRESET: NutrientDraft = {
  enabled: true,
  E_a_kJ_per_mol: 70.0,
  k0_per_s: 2.63e6,
  D_eff_m2_per_s: 2.0e-10,
  K_partition: 1.0e-5,
  C_water_sat_mg_per_kg: 6.0e-3,
  C0_mg_per_kg: 83.0,
  nu_water_m2_per_s: 2.94e-7,
  D_water_molec_m2_per_s: 1.0e-9,
};

export const VITAMIN_C_PRESET: NutrientDraft = {
  enabled: true,
  E_a_kJ_per_mol: 74.0,
  k0_per_s: 1.1e7,
  D_eff_m2_per_s: 5.0e-10,
  K_partition: 1.0,
  C_water_sat_mg_per_kg: 1.0e6,
  C0_mg_per_kg: 59.0,
  nu_water_m2_per_s: 2.94e-7,
  D_water_molec_m2_per_s: 1.0e-9,
};

const DISABLED_NUTRIENT: NutrientDraft = {
  ...BETA_CAROTENE_PRESET,
  enabled: false,
};

/** Map a solute dropdown choice onto the (nutrient, nutrient2) pair
 *  the backend expects. The disabled slot carries β-carotene defaults
 *  so `model_validate` passes if the user flips it on later. */
export function soluteKeyToNutrients(
  key: SoluteKey,
): { nutrient: NutrientDraft; nutrient2: NutrientDraft } {
  switch (key) {
    case "off":
      return { nutrient: DISABLED_NUTRIENT, nutrient2: DISABLED_NUTRIENT };
    case "beta_carotene":
      return { nutrient: BETA_CAROTENE_PRESET, nutrient2: DISABLED_NUTRIENT };
    case "vitamin_c":
      return { nutrient: VITAMIN_C_PRESET, nutrient2: DISABLED_NUTRIENT };
    case "both":
      return { nutrient: BETA_CAROTENE_PRESET, nutrient2: VITAMIN_C_PRESET };
  }
}

/** Reverse: figure out which solute preset a draft currently matches,
 *  so the dropdown can show the right default. Falls back to `off`
 *  when either slot has been customised via YAML. */
export function nutrientsToSoluteKey(
  n: NutrientDraft,
  n2: NutrientDraft,
): SoluteKey {
  const n2On = n2.enabled;
  if (!n.enabled) return "off";
  const isBeta = (d: NutrientDraft) =>
    Math.abs(d.K_partition - 1.0e-5) < 1e-9 && Math.abs(d.E_a_kJ_per_mol - 70.0) < 1e-6;
  const isVitC = (d: NutrientDraft) =>
    Math.abs(d.K_partition - 1.0) < 1e-9 && Math.abs(d.E_a_kJ_per_mol - 74.0) < 1e-6;
  if (n2On && isBeta(n) && isVitC(n2)) return "both";
  if (!n2On && isBeta(n)) return "beta_carotene";
  if (!n2On && isVitC(n)) return "vitamin_c";
  return "off";
}

/** Default values mirroring the Pydantic defaults in
 *  python/boilingsim/config.py. Kept here for the Configuration
 *  page's "Reset to defaults" button and the initial form state
 *  before the user loads a preset. */
export const DEFAULT_DRAFT: ScenarioDraft = {
  pot: {
    diameter_m: 0.20,
    height_m: 0.12,
    wall_thickness_m: 0.003,
    base_thickness_m: 0.005,
    material: "steel_304",
  },
  water: {
    fill_fraction: 0.75,
    initial_temp_c: 20.0,
  },
  carrot: {
    diameter_m: 0.025,
    length_m: 0.05,
    position: [0.0, 0.0, 0.03],
    count: 1,
    axis: "z",
    initial_beta_carotene_mg_per_100g: 8.3,
  },
  heating: {
    base_heat_flux_w_per_m2: 80000.0,
    ambient_temp_c: 22.0,
  },
  initial_conditions: {
    mode: "cold",
    preheat_water_c: 95.0,
    preheat_wall_c: 100.0,
    preheat_carrot_c: 20.0,
  },
  grid: {
    dx_m: 0.002,
    carrot_mesh_resolution: 40,
  },
  solver: {
    cfl_safety_factor: 0.4,
    max_dt_s: 0.1,
    pressure_tol: 1.0e-5,
    pressure_max_iter: 100,
    diffusion_tol: 1.0e-4,
    diffusion_max_iter: 15,
    h_conv_outer_w_per_m2_k: 10.0,
    h_evap_free_surface_w_per_m2_k: 5.0e4,
    f_bulk_evap_per_s: 1.0,
    use_implicit_conduction: true,
  },
  boiling: {
    enabled: true,
    dT_onb_k: 5.0,
    contact_angle_rad: 1.0,
    max_bubbles: 100_000,
    initial_bubble_radius_m: 1.0e-5,
    nucleation_probability_per_step: 0.1,
    C_sf_rohsenow: 0.013,
    Pr_n_rohsenow: 1.0,
  },
  nutrient: {
    enabled: true,
    E_a_kJ_per_mol: 70.0,
    k0_per_s: 2.63e6,
    D_eff_m2_per_s: 2.0e-10,
    K_partition: 1.0e-5,
    C_water_sat_mg_per_kg: 6.0e-3,
    C0_mg_per_kg: 83.0,
    nu_water_m2_per_s: 2.94e-7,
    D_water_molec_m2_per_s: 1.0e-9,
  },
  nutrient2: {
    enabled: false,
    E_a_kJ_per_mol: 74.0,
    k0_per_s: 1.1e7,
    D_eff_m2_per_s: 5.0e-10,
    K_partition: 1.0,
    C_water_sat_mg_per_kg: 1.0e6,
    C0_mg_per_kg: 59.0,
    nu_water_m2_per_s: 2.94e-7,
    D_water_molec_m2_per_s: 1.0e-9,
  },
  simulation: {
    total_time_s: 600.0,
    output_every_s: 0.1,
  },
};

/** Named presets matching the YAML scenarios + the NUTRIENT_PRESETS
 *  dict in scripts/run_dashboard.py. */
export const PRESETS: Record<string, { label: string; draft: () => ScenarioDraft }> = {
  default: {
    label: "Default (β-carotene, 25 mm, steel 304)",
    draft: () => structuredClone(DEFAULT_DRAFT),
  },
  vitamin_c: {
    label: "Vitamin C (25 mm, steel 304)",
    draft: () => {
      const d = structuredClone(DEFAULT_DRAFT);
      d.nutrient = {
        enabled: true,
        E_a_kJ_per_mol: 74.0,
        k0_per_s: 1.1e7,
        D_eff_m2_per_s: 5.0e-10,
        K_partition: 1.0,
        C_water_sat_mg_per_kg: 1.0e6,
        C0_mg_per_kg: 59.0,
        nu_water_m2_per_s: 2.94e-7,
        D_water_molec_m2_per_s: 1.0e-9,
      };
      return d;
    },
  },
  dual_solute: {
    label: "Dual solute (β-carotene + vitamin C)",
    draft: () => {
      const d = structuredClone(DEFAULT_DRAFT);
      d.nutrient2.enabled = true;
      return d;
    },
  },
  simmer: {
    label: "Gentle simmer (10 kW/m²)",
    draft: () => {
      const d = structuredClone(DEFAULT_DRAFT);
      d.heating.base_heat_flux_w_per_m2 = 10_000.0;
      return d;
    },
  },
  copper: {
    label: "Copper pot",
    draft: () => {
      const d = structuredClone(DEFAULT_DRAFT);
      d.pot.material = "copper";
      return d;
    },
  },
  aluminum: {
    label: "Aluminium pot",
    draft: () => {
      const d = structuredClone(DEFAULT_DRAFT);
      d.pot.material = "aluminum";
      return d;
    },
  },
};

/** Convert a ScenarioDraft to the Pydantic-expected JSON blob. The
 *  only transformation: Python expects `nutrient` and `nutrient2`
 *  keys on ScenarioConfig, and a separate `total_time_s` /
 *  `output_every_s` on the top level (not nested under
 *  `simulation`). */
export function draftToScenarioJson(
  d: ScenarioDraft,
): Record<string, unknown> {
  return {
    pot: d.pot,
    water: d.water,
    carrot: d.carrot,
    heating: d.heating,
    initial_conditions: d.initial_conditions,
    grid: d.grid,
    solver: d.solver,
    boiling: d.boiling,
    nutrient: d.nutrient,
    nutrient2: d.nutrient2,
    total_time_s: d.simulation.total_time_s,
    output_every_s: d.simulation.output_every_s,
  };
}
