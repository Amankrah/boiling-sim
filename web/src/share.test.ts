import { describe, expect, it } from "vitest";

import {
  DEFAULT_SHARE_STATE,
  decodeShareState,
  encodeShareState,
  type ShareState,
} from "./share";

describe("share URL encode/decode", () => {
  it("round-trips the default state", () => {
    const encoded = encodeShareState(DEFAULT_SHARE_STATE);
    const decoded = decodeShareState(encoded);
    expect(decoded.params).toEqual(DEFAULT_SHARE_STATE.params);
    for (let i = 0; i < 3; i++) {
      expect(decoded.camera.position[i]).toBeCloseTo(
        DEFAULT_SHARE_STATE.camera.position[i],
        4,
      );
      expect(decoded.camera.target[i]).toBeCloseTo(
        DEFAULT_SHARE_STATE.camera.target[i],
        4,
      );
    }
  });

  it("round-trips a non-default copper + 45 kW configuration", () => {
    const state: ShareState = {
      params: {
        heatFluxWPerM2: 45_000,
        material: "copper",
        carrotDiameterMm: 30,
        carrotLengthMm: 60,
      },
      camera: {
        position: [0.42, -0.11, 0.22],
        target: [0.01, 0.0, 0.05],
      },
    };
    const encoded = encodeShareState(state);
    const decoded = decodeShareState(encoded);
    expect(decoded.params).toEqual(state.params);
    for (let i = 0; i < 3; i++) {
      expect(decoded.camera.position[i]).toBeCloseTo(state.camera.position[i], 3);
      expect(decoded.camera.target[i]).toBeCloseTo(state.camera.target[i], 3);
    }
  });

  it("falls back to defaults for missing fields", () => {
    const decoded = decodeShareState("?hf=12345");
    expect(decoded.params.heatFluxWPerM2).toBe(12345);
    expect(decoded.params.material).toBe(DEFAULT_SHARE_STATE.params.material);
    expect(decoded.params.carrotDiameterMm).toBe(
      DEFAULT_SHARE_STATE.params.carrotDiameterMm,
    );
    expect(decoded.camera).toEqual(DEFAULT_SHARE_STATE.camera);
  });

  it("rejects bogus material values by falling back to the default", () => {
    const decoded = decodeShareState("?mat=unobtainium");
    expect(decoded.params.material).toBe(DEFAULT_SHARE_STATE.params.material);
  });

  it("rejects NaN / non-numeric fields", () => {
    const decoded = decodeShareState("?hf=notanumber&cx=oops");
    expect(decoded.params.heatFluxWPerM2).toBe(
      DEFAULT_SHARE_STATE.params.heatFluxWPerM2,
    );
    expect(decoded.camera.position[0]).toBe(DEFAULT_SHARE_STATE.camera.position[0]);
  });

  it("drops simulation-time-like keys silently (semantic scope)", () => {
    // The dev-guide example URL includes `t=300`. Our implementation
    // intentionally ignores it -- share links encode scene + camera
    // only, not sim time. See plan non-goals.
    const decoded = decodeShareState("?hf=30000&t=300");
    expect(decoded.params.heatFluxWPerM2).toBe(30000);
    // Decoded state has no t_sim field, and defaults still apply.
    expect("t_sim" in (decoded as unknown as Record<string, unknown>)).toBe(false);
  });

  it("emits a URLSearchParams-parseable querystring (no leading ?)", () => {
    const q = encodeShareState(DEFAULT_SHARE_STATE);
    expect(q.startsWith("?")).toBe(false);
    const params = new URLSearchParams(q);
    expect(params.has("hf")).toBe(true);
    expect(params.has("mat")).toBe(true);
    expect(params.has("cd")).toBe(true);
    expect(params.has("cl")).toBe(true);
    expect(params.has("cx")).toBe(true);
    expect(params.has("cy")).toBe(true);
    expect(params.has("cz")).toBe(true);
    expect(params.has("cfx")).toBe(true);
    expect(params.has("cfy")).toBe(true);
    expect(params.has("cfz")).toBe(true);
    expect(params.has("t")).toBe(false); // sim time must NOT be encoded
  });
});
