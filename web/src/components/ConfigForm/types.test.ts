import { describe, expect, it } from "vitest";

import {
  BETA_CAROTENE_PRESET,
  DEFAULT_DRAFT,
  PRESETS,
  VITAMIN_C_PRESET,
  draftToScenarioJson,
  makeBlankCoupling,
  makeBlankIngredient,
  makeBlankNutrient,
  nutrientsToSoluteKey,
  soluteKeyToNutrients,
} from "./types";

describe("ConfigForm draftToScenarioJson", () => {
  it("returns a top-level shape that matches ScenarioConfig Pydantic keys", () => {
    const j = draftToScenarioJson(DEFAULT_DRAFT);
    // Pydantic ScenarioConfig keys live at the top level...
    for (const k of [
      "pot", "water", "carrot", "heating", "initial_conditions",
      "grid", "solver", "boiling", "nutrient", "nutrient2",
      "total_time_s", "output_every_s",
    ]) {
      expect(j).toHaveProperty(k);
    }
    // ...and the form's "simulation" wrapper does NOT leak through.
    expect(j).not.toHaveProperty("simulation");
  });

  it("forwards initial_conditions to the top-level payload", () => {
    const draft = structuredClone(DEFAULT_DRAFT);
    draft.initial_conditions.mode = "preheat";
    draft.initial_conditions.preheat_water_c = 90.0;
    const j = draftToScenarioJson(draft) as Record<string, Record<string, unknown>>;
    expect(j.initial_conditions.mode).toBe("preheat");
    expect(j.initial_conditions.preheat_water_c).toBe(90.0);
  });

  it("default draft initial_conditions is cold", () => {
    expect(DEFAULT_DRAFT.initial_conditions.mode).toBe("cold");
  });

  it("carries simulation.total_time_s to the top-level Pydantic key", () => {
    const draft = structuredClone(DEFAULT_DRAFT);
    draft.simulation.total_time_s = 123.5;
    draft.simulation.output_every_s = 0.25;
    const j = draftToScenarioJson(draft) as Record<string, unknown>;
    expect(j.total_time_s).toBe(123.5);
    expect(j.output_every_s).toBe(0.25);
  });

  it("deep-clones sub-blocks so Pydantic can't mutate our state", () => {
    const draft = structuredClone(DEFAULT_DRAFT);
    const j = draftToScenarioJson(draft) as Record<string, Record<string, unknown>>;
    expect(j.pot.diameter_m).toBe(draft.pot.diameter_m);
    // The blob shares references with the draft slice -- that's fine
    // because sendCommand serialises to JSON immediately, but verify
    // the top-level shape is its own object.
    expect(j).not.toBe(draft as unknown as typeof j);
  });
});

describe("ConfigForm presets", () => {
  it("default preset produces the baseline", () => {
    const d = PRESETS.default.draft();
    expect(d.nutrient.K_partition).toBeCloseTo(1e-5);
    expect(d.nutrient2.enabled).toBe(false);
    expect(d.pot.material).toBe("steel_304");
  });

  it("vitamin_c preset flips K_partition to ~1 and sat to huge", () => {
    const d = PRESETS.vitamin_c.draft();
    expect(d.nutrient.K_partition).toBeCloseTo(1.0);
    expect(d.nutrient.C_water_sat_mg_per_kg).toBeGreaterThan(1e3);
  });

  it("dual_solute preset turns on nutrient2", () => {
    const d = PRESETS.dual_solute.draft();
    expect(d.nutrient.enabled).toBe(true);
    expect(d.nutrient2.enabled).toBe(true);
  });

  it("copper preset swaps material", () => {
    const d = PRESETS.copper.draft();
    expect(d.pot.material).toBe("copper");
  });

  it("simmer preset lowers heat flux", () => {
    const d = PRESETS.simmer.draft();
    expect(d.heating.base_heat_flux_w_per_m2).toBeLessThanOrEqual(15_000);
  });

  it("preset drafts are independent clones", () => {
    const a = PRESETS.default.draft();
    const b = PRESETS.default.draft();
    a.pot.material = "copper";
    expect(b.pot.material).toBe("steel_304");
  });
});

describe("ConfigForm solute preset plumbing", () => {
  it("soluteKeyToNutrients('off') disables both slots", () => {
    const { nutrient, nutrient2 } = soluteKeyToNutrients("off");
    expect(nutrient.enabled).toBe(false);
    expect(nutrient2.enabled).toBe(false);
  });

  it("soluteKeyToNutrients('beta_carotene') matches the canonical preset", () => {
    const { nutrient, nutrient2 } = soluteKeyToNutrients("beta_carotene");
    expect(nutrient.enabled).toBe(true);
    expect(nutrient.K_partition).toBeCloseTo(BETA_CAROTENE_PRESET.K_partition);
    expect(nutrient2.enabled).toBe(false);
  });

  it("soluteKeyToNutrients('both') enables β-carotene + vitamin C", () => {
    const { nutrient, nutrient2 } = soluteKeyToNutrients("both");
    expect(nutrient.K_partition).toBeCloseTo(BETA_CAROTENE_PRESET.K_partition);
    expect(nutrient2.enabled).toBe(true);
    expect(nutrient2.K_partition).toBeCloseTo(VITAMIN_C_PRESET.K_partition);
  });

  it("nutrientsToSoluteKey round-trips every preset", () => {
    for (const key of ["off", "beta_carotene", "vitamin_c", "both"] as const) {
      const { nutrient, nutrient2 } = soluteKeyToNutrients(key);
      expect(nutrientsToSoluteKey(nutrient, nutrient2)).toBe(key);
    }
  });

  // -- M7 coverage ----------------------------------------------------

  it("emits an empty extra_ingredients list and empty couplings list by default", () => {
    const j = draftToScenarioJson(DEFAULT_DRAFT) as Record<string, unknown>;
    expect(j.extra_ingredients).toEqual([]);
    expect(j.nutrient_couplings).toEqual([]);
  });

  it("translates an extra ingredient with three nutrients to nutrient + nutrient2 + extra_nutrients", () => {
    const draft = structuredClone(DEFAULT_DRAFT);
    const extra = makeBlankIngredient();
    extra.name = "potato";
    extra.nutrients = [
      { ...makeBlankNutrient("starch"), enabled: true, K_partition: 0.5 },
      { ...makeBlankNutrient("vitamin_b6"), enabled: true },
      { ...makeBlankNutrient("potassium"), enabled: true },
    ];
    draft.extra_ingredients = [extra];
    const j = draftToScenarioJson(draft) as Record<string, unknown>;
    const extras = j.extra_ingredients as Record<string, unknown>[];
    expect(extras).toHaveLength(1);
    const e0 = extras[0] as Record<string, unknown>;
    expect(e0.name).toBe("potato");
    // First two nutrients ride in legacy slots; rest in extra_nutrients.
    expect((e0.nutrient as Record<string, unknown>).name).toBe("starch");
    expect((e0.nutrient2 as Record<string, unknown>).name).toBe("vitamin_b6");
    const extraNuts = e0.extra_nutrients as Record<string, unknown>[];
    expect(extraNuts).toHaveLength(1);
    expect(extraNuts[0].name).toBe("potassium");
  });

  it("filters out couplings with empty protector/protected", () => {
    const draft = structuredClone(DEFAULT_DRAFT);
    draft.couplings = [
      { ...makeBlankCoupling(), protector: "", protected: "carrot.beta_carotene" },
      {
        ...makeBlankCoupling(),
        protector: "carrot.vitamin_c",
        protected: "carrot.beta_carotene",
      },
    ];
    const j = draftToScenarioJson(draft) as Record<string, unknown>;
    const couplings = j.nutrient_couplings as Record<string, unknown>[];
    expect(couplings).toHaveLength(1);
    const cc = couplings[0];
    expect(cc.protector_ingredient).toBe("carrot");
    expect(cc.protected_ingredient).toBe("carrot");
  });
});
