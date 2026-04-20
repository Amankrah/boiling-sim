import { describe, expect, it } from "vitest";

import { parseScalarsCsv } from "./useRunArtefacts";

describe("parseScalarsCsv", () => {
  it("parses the run-writer's fixed column layout", () => {
    const csv = [
      "t,dt,T_mean_water_c,T_max_water_c,T_min_water_c,T_max_wall_c,T_inner_wall_mean_c,T_inner_wall_max_c,u_max_mps,n_active_bubbles,mean_bubble_R_mm,mean_departed_bubble_R_mm,max_bubble_R_mm,alpha_min,retention_pct,leached_pct,degraded_pct,precipitated_pct,retention2_pct,leached2_pct,degraded2_pct,precipitated2_pct",
      "0.0,0.001,95.0,95.5,94.5,107.0,106.5,107.2,0.1,0,0,0,0,1.0,100.0,0.0,0.0,0.0,100.0,0.0,0.0,0.0",
      "10.0,0.002,99.5,100.2,99.0,107.1,106.6,107.3,0.2,10,0.3,0.0,0.5,0.95,99.8,0.0,0.2,0.0,100.0,0.0,0.0,0.0",
    ].join("\n");
    const rows = parseScalarsCsv(csv);
    expect(rows).toHaveLength(2);
    expect(rows[0].t).toBe(0);
    expect(rows[0].T_mean_water_c).toBeCloseTo(95.0);
    expect(rows[1].retention_pct).toBeCloseTo(99.8);
    expect(rows[1].leached_pct).toBeCloseTo(0);
  });

  it("tolerates trailing empty lines", () => {
    const csv = "t,dt,T_mean_water_c,T_max_water_c,T_min_water_c,T_max_wall_c,T_inner_wall_mean_c,T_inner_wall_max_c,u_max_mps,n_active_bubbles,mean_bubble_R_mm,mean_departed_bubble_R_mm,max_bubble_R_mm,alpha_min,retention_pct,leached_pct,degraded_pct,precipitated_pct,retention2_pct,leached2_pct,degraded2_pct,precipitated2_pct\n0.0,0.001,95,95,95,107,106,107,0,0,0,0,0,1,100,0,0,0,100,0,0,0\n\n";
    const rows = parseScalarsCsv(csv);
    expect(rows).toHaveLength(1);
  });

  it("returns empty array on header-only input", () => {
    expect(parseScalarsCsv("t,dt")).toEqual([]);
    expect(parseScalarsCsv("")).toEqual([]);
  });
});
