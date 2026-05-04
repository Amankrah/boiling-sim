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
  /** Quantity-input mode. "dimensions" lets the user set length_m
   *  directly (legacy); "mass" derives length_m from target_mass_g
   *  so users can specify "I want 200 g of carrots". */
  mass_mode: "dimensions" | "mass";
  /** Required iff mass_mode="mass". Total grams across all instances. */
  target_mass_g: number | null;
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

/** M7: one extra ingredient (M4+ ``cfg.extra_ingredients[k]``).
 *
 *  Carries the same geometry knobs as ``CarrotDraft`` plus its own
 *  display name, density, and per-ingredient nutrient profiles. The
 *  ``nutrients`` list (M8) holds an arbitrary number of solutes per
 *  ingredient; the first two get translated to the legacy
 *  ``nutrient`` / ``nutrient2`` slots on the Python side, the rest go
 *  into ``extra_nutrients``. */
export interface ExtraIngredientDraft {
  /** Display name; drives the 3D scene's color palette. */
  name: string;
  density_kg_per_m3: number;
  diameter_m: number;
  length_m: number;
  position: [number, number, number];
  count: number;
  axis: "x" | "y" | "z";
  mass_mode: "dimensions" | "mass";
  target_mass_g: number | null;
  /** M8: variable-N nutrients per ingredient. Empty list = no
   *  per-ingredient nutrient kinetics for this extra. */
  nutrients: NamedNutrientDraft[];
}

/** A NutrientDraft with an explicit display name. The name doubles
 *  as the dict key in the clean YAML schema and as the identifier
 *  couplings reference (``ingredient.nutrient`` dotted IDs). */
export interface NamedNutrientDraft extends NutrientDraft {
  name: string;
}

/** M7: nutrient-nutrient coupling (Sakai 1987 protective-effect model).
 *  Each coupling references its protector + protected by dotted
 *  ``ingredient.nutrient`` identifiers. */
export interface CouplingDraft {
  /** ``"carrot.vitamin_c"`` etc. Empty string = unset (skipped). */
  protector: string;
  protected: string;
  enabled: boolean;
  eta: number;
  c_ref_mg_per_kg: number;
  eta_max: number;
}

export interface ScenarioDraft {
  pot: PotDraft;
  water: WaterDraft;
  carrot: CarrotDraft;
  /** M7: ingredients beyond the legacy carrot. Empty by default. */
  extra_ingredients: ExtraIngredientDraft[];
  /** M7: protective couplings between solute slots. Empty by default. */
  couplings: CouplingDraft[];
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
    mass_mode: "dimensions",
    target_mass_g: null,
    initial_beta_carotene_mg_per_100g: 8.3,
  },
  extra_ingredients: [],
  couplings: [],
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
 *  `simulation`).
 *
 *  M7+M8: extras with their nutrient lists are emitted as
 *  ``extra_ingredients[]`` with ``nutrient`` (slot 0), ``nutrient2``
 *  (slot 1), and ``extra_nutrients`` (slot 2+) blocks. Couplings are
 *  emitted as ``nutrient_couplings[]`` with the coupling block's
 *  dotted ``protector`` / ``protected`` identifiers split into
 *  ``protector_ingredient`` + ``protector_nutrient_name`` etc. Empty
 *  extras / couplings lists serialize as empty arrays which the
 *  Pydantic side accepts cleanly. */
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
    extra_ingredients: d.extra_ingredients.map(extraToJson),
    nutrient_couplings: d.couplings
      .filter((c) => c.protector && c.protected)
      .map((c) => couplingToJson(c, d)),
    total_time_s: d.simulation.total_time_s,
    output_every_s: d.simulation.output_every_s,
  };
}

/** Translate one extra-ingredient draft into the legacy Pydantic
 *  shape. The first nutrient becomes ``nutrient`` (primary), the
 *  second becomes ``nutrient2`` (secondary), the rest land in
 *  ``extra_nutrients[]``. */
function extraToJson(e: ExtraIngredientDraft): Record<string, unknown> {
  const geom: Record<string, unknown> = {
    name: e.name,
    density_kg_per_m3: e.density_kg_per_m3,
    diameter_m: e.diameter_m,
    length_m: e.length_m,
    position: e.position,
    count: e.count,
    axis: e.axis,
    mass_mode: e.mass_mode,
    target_mass_g: e.target_mass_g,
  };
  const nuts = e.nutrients;
  if (nuts.length >= 1) geom.nutrient = stripName(nuts[0]);
  if (nuts.length >= 2) geom.nutrient2 = stripName(nuts[1]);
  if (nuts.length >= 3) {
    geom.extra_nutrients = nuts.slice(2).map(stripName);
  }
  return geom;
}

function stripName(n: NamedNutrientDraft): NutrientDraft & { name: string } {
  // Python's NutrientConfig has ``name`` as a real field (M6), so
  // we just pass it through. Keeping the helper for symmetry.
  return { ...n };
}

/** Translate one coupling draft into the legacy Pydantic shape.
 *  Splits the dotted ``protector`` (e.g. ``carrot.vitamin_c``) into
 *  its ingredient + nutrient parts. The Python translator picks the
 *  right slot literal vs nutrient-name field based on whether the
 *  nutrient sits in the legacy 2-slot pair or in ``extra_nutrients``. */
function couplingToJson(
  c: CouplingDraft,
  d: ScenarioDraft,
): Record<string, unknown> {
  const [protI, protN] = (c.protector ?? "").split(".");
  const [tgtI, tgtN] = (c.protected ?? "").split(".");
  // Resolve slot via name lookup against the draft itself so the JSON
  // matches what the Pydantic resolver expects on the receive side.
  const out: Record<string, unknown> = {
    enabled: c.enabled,
    eta: c.eta,
    c_ref_mg_per_kg: c.c_ref_mg_per_kg,
    eta_max: c.eta_max,
    protector_ingredient: protI ?? "",
    protected_ingredient: tgtI ?? "",
    ...slotIdentForDraft(d, protI, protN, "protector"),
    ...slotIdentForDraft(d, tgtI, tgtN, "protected"),
  };
  return out;
}

/** Resolve ``ingredient.nutrient`` against a draft's ingredient list
 *  to either the legacy slot literal ("primary"/"secondary") or the
 *  M8 ``protector_nutrient_name`` field. Falls back to ``primary`` if
 *  the lookup fails (the Python side will surface a clearer error
 *  during validation). */
function slotIdentForDraft(
  d: ScenarioDraft,
  ingName: string | undefined,
  nutName: string | undefined,
  kind: "protector" | "protected",
): Record<string, string> {
  const out: Record<string, string> = {};
  if (!ingName || !nutName) return out;
  // Ingredient 0 (legacy carrot): nutrient slot 0/1 only.
  if (ingName === "carrot" || !d.extra_ingredients.find((e) => e.name === ingName)) {
    // Ingredient 0's nutrient names live on d.nutrient.name / d.nutrient2.name
    // (added by M6 schema). Fall back to "primary" if the field hasn't been
    // populated yet.
    out[`${kind}_slot`] = nutName === "vitamin_c" ? "secondary" : "primary";
    return out;
  }
  // Extra ingredient: find the nutrient by name in its list.
  const extra = d.extra_ingredients.find((e) => e.name === ingName);
  if (!extra) return out;
  const idx = extra.nutrients.findIndex((n) => n.name === nutName);
  if (idx === 0) {
    out[`${kind}_slot`] = "primary";
  } else if (idx === 1) {
    out[`${kind}_slot`] = "secondary";
  } else if (idx >= 2) {
    out[`${kind}_nutrient_name`] = nutName;
    out[`${kind}_slot`] = "primary"; // placeholder; Python uses the name field
  }
  return out;
}

/** Factory for an "Add ingredient" button. Defaults to a small
 *  potato with a single starch nutrient, off (so adding the row
 *  doesn't accidentally enable nutrient kernels). */
export function makeBlankIngredient(): ExtraIngredientDraft {
  return {
    name: "potato",
    density_kg_per_m3: 1080.0,
    diameter_m: 0.030,
    length_m: 0.040,
    position: [0.0, -0.040, 0.040],
    count: 1,
    axis: "x",
    mass_mode: "dimensions",
    target_mass_g: null,
    nutrients: [],
  };
}

/** Factory for an "Add nutrient" button on an ingredient. Defaults
 *  to a disabled β-carotene preset with an empty name (user must
 *  pick a name before applying). */
export function makeBlankNutrient(name = ""): NamedNutrientDraft {
  return {
    name,
    enabled: false,
    E_a_kJ_per_mol: 60.0,
    k0_per_s: 1.0e6,
    D_eff_m2_per_s: 2.0e-10,
    K_partition: 1.0,
    C_water_sat_mg_per_kg: 1.0e3,
    C0_mg_per_kg: 100.0,
    nu_water_m2_per_s: 2.94e-7,
    D_water_molec_m2_per_s: 1.0e-9,
  };
}

/** Factory for an "Add coupling" button. Empty protector/protected
 *  fields (user must fill them in via dropdowns). */
export function makeBlankCoupling(): CouplingDraft {
  return {
    protector: "",
    protected: "",
    enabled: true,
    eta: 0.5,
    c_ref_mg_per_kg: 5.0,
    eta_max: 0.5,
  };
}
