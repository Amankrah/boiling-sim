// Unit tests for the milestone + loss-rate helpers in derived.ts.
// These power the Heat-up storyline and Nutrient loss-rate cards;
// the math is the only part that's interesting to test (the
// Recharts plumbing is visual).

import { describe, expect, it } from "vitest";

import type { RunSummary, ScalarRow } from "../../hooks/useRunArtefacts";

import {
  computeLossRate,
  finalPartition,
  findMilestones,
  hasSecondarySolute,
} from "./derived";

/** Build a ScalarRow with all fields zero except those overridden.
 *  Keeps test cases focused on the columns each helper actually
 *  reads. */
function row(overrides: Partial<ScalarRow>): ScalarRow {
  return {
    t: 0,
    dt: 0,
    T_mean_water_c: 0,
    T_max_water_c: 0,
    T_min_water_c: 0,
    T_max_wall_c: 0,
    T_inner_wall_mean_c: 0,
    T_inner_wall_max_c: 0,
    u_max_mps: 0,
    n_active_bubbles: 0,
    mean_bubble_R_mm: 0,
    mean_departed_bubble_R_mm: 0,
    max_bubble_R_mm: 0,
    alpha_min: 1,
    retention_pct: 100,
    leached_pct: 0,
    degraded_pct: 0,
    precipitated_pct: 0,
    retention2_pct: 100,
    leached2_pct: 0,
    degraded2_pct: 0,
    precipitated2_pct: 0,
    ...overrides,
  };
}

describe("findMilestones", () => {
  it("locates t_sat at the first sample crossing 99 C", () => {
    const rows = [
      row({ t: 0, T_mean_water_c: 25 }),
      row({ t: 10, T_mean_water_c: 70 }),
      row({ t: 20, T_mean_water_c: 99.2 }),
      row({ t: 30, T_mean_water_c: 99.9 }),
    ];
    const m = findMilestones(rows);
    expect(m.tSat).toBe(20);
  });

  it("locates t_first_bubble at the first sample with n>=1", () => {
    const rows = [
      row({ t: 0, n_active_bubbles: 0 }),
      row({ t: 5, n_active_bubbles: 0 }),
      row({ t: 10, n_active_bubbles: 1 }),
      row({ t: 15, n_active_bubbles: 12 }),
    ];
    const m = findMilestones(rows);
    expect(m.tFirstBubble).toBe(10);
  });

  it("locates t_first_loss at the first sample retention<99.5%", () => {
    const rows = [
      row({ t: 0, retention_pct: 100 }),
      row({ t: 10, retention_pct: 99.9 }),
      row({ t: 20, retention_pct: 99.4 }),
      row({ t: 30, retention_pct: 95.0 }),
    ];
    const m = findMilestones(rows);
    expect(m.tFirstLoss).toBe(20);
  });

  it("returns undefined when a threshold is never crossed", () => {
    const rows = [
      row({ t: 0, T_mean_water_c: 25, n_active_bubbles: 0 }),
      row({ t: 10, T_mean_water_c: 30, n_active_bubbles: 0 }),
    ];
    const m = findMilestones(rows);
    expect(m.tSat).toBeUndefined();
    expect(m.tFirstBubble).toBeUndefined();
    expect(m.tFirstLoss).toBeUndefined();
  });

  it("handles an empty rows array without throwing", () => {
    const m = findMilestones([]);
    expect(m).toEqual({});
  });
});

describe("computeLossRate", () => {
  it("returns positive rate when retention is decreasing", () => {
    // retention drops 10 % over 10 s -> 1 %/s -> 60 %/min.
    const rows = [
      row({ t: 0, retention_pct: 100 }),
      row({ t: 5, retention_pct: 95 }),
      row({ t: 10, retention_pct: 90 }),
    ];
    const out = computeLossRate(rows, false);
    expect(out).toHaveLength(3);
    // The middle sample sees a clean central diff: (90 - 100) / 10 = -1 %/s,
    // negated -> +1 %/s -> +60 %/min.
    expect(out[1].rate1).toBeCloseTo(60, 4);
    // No secondary requested.
    expect(out[1].rate2).toBeUndefined();
  });

  it("returns near-zero rate when retention is flat", () => {
    const rows = [
      row({ t: 0, retention_pct: 100 }),
      row({ t: 5, retention_pct: 100 }),
      row({ t: 10, retention_pct: 100 }),
      row({ t: 15, retention_pct: 100 }),
    ];
    const out = computeLossRate(rows, false);
    for (const r of out) {
      expect(r.rate1).toBeCloseTo(0, 4);
    }
  });

  it("includes secondary rate when requested", () => {
    const rows = [
      row({ t: 0, retention_pct: 100, retention2_pct: 100 }),
      row({ t: 5, retention_pct: 99, retention2_pct: 90 }),
      row({ t: 10, retention_pct: 98, retention2_pct: 80 }),
    ];
    const out = computeLossRate(rows, true);
    // Secondary is dropping 10x faster than primary, so rate2 ~= 10 * rate1.
    expect(out[1].rate2).toBeDefined();
    expect(out[1].rate2!).toBeGreaterThan(out[1].rate1 * 5);
  });

  it("handles a single-row CSV by returning a zero rate", () => {
    const out = computeLossRate([row({ t: 0, retention_pct: 100 })], false);
    expect(out).toEqual([{ t: 0, rate1: 0 }]);
  });

  it("returns [] for an empty input", () => {
    expect(computeLossRate([], false)).toEqual([]);
  });
});

describe("finalPartition + hasSecondarySolute", () => {
  function makeSummary(final: Record<string, number>, names = { p: "β-carotene", s: "" }): RunSummary {
    return {
      run_id: "r",
      schema_version: 4,
      n_samples: 1,
      wall_clock_s: 1,
      t_sim_total_s: 1,
      step_count: 1,
      s_per_sim_s: 1,
      snapshot_cadence_hz: 1,
      nutrient_primary_name: names.p,
      nutrient_secondary_name: names.s,
      final,
      acceptance: [],
      mass_balance: { max_abs_drift_pct: 0, final_sum_pct: 100 },
      parameters: {},
    };
  }

  it("extracts the primary slot's four buckets", () => {
    const s = makeSummary({
      retention_pct: 88,
      leached_pct: 5,
      degraded_pct: 6,
      precipitated_pct: 1,
    });
    expect(finalPartition(s, "primary")).toEqual({
      retention: 88,
      leached: 5,
      degraded: 6,
      precipitated: 1,
    });
  });

  it("falls back to 0 when a bucket is missing or non-finite", () => {
    const s = makeSummary({ retention_pct: 99 });
    expect(finalPartition(s, "primary")).toEqual({
      retention: 99,
      leached: 0,
      degraded: 0,
      precipitated: 0,
    });
  });

  it("hasSecondarySolute is true when the secondary nutrient name is set", () => {
    const s = makeSummary({}, { p: "β-carotene", s: "vitamin C" });
    expect(hasSecondarySolute(s)).toBe(true);
  });

  it("hasSecondarySolute is true when secondary retention has dropped", () => {
    const s = makeSummary(
      {
        retention2_pct: 80,
        leached2_pct: 0,
      },
      { p: "β-carotene", s: "" },
    );
    expect(hasSecondarySolute(s)).toBe(true);
  });

  it("hasSecondarySolute is false on a primary-only run", () => {
    const s = makeSummary(
      {
        retention2_pct: 100,
        leached2_pct: 0,
      },
      { p: "β-carotene", s: "" },
    );
    expect(hasSecondarySolute(s)).toBe(false);
  });
});
