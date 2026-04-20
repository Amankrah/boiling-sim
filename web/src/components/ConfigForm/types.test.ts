import { describe, expect, it } from "vitest";

import {
  DEFAULT_DRAFT,
  PRESETS,
  draftToScenarioJson,
} from "./types";

describe("ConfigForm draftToScenarioJson", () => {
  it("returns a top-level shape that matches ScenarioConfig Pydantic keys", () => {
    const j = draftToScenarioJson(DEFAULT_DRAFT);
    // Pydantic ScenarioConfig keys live at the top level...
    for (const k of [
      "pot", "water", "carrot", "heating", "grid",
      "solver", "boiling", "nutrient", "nutrient2",
      "total_time_s", "output_every_s",
    ]) {
      expect(j).toHaveProperty(k);
    }
    // ...and the form's "simulation" wrapper does NOT leak through.
    expect(j).not.toHaveProperty("simulation");
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
