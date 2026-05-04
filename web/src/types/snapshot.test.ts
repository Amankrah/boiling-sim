// M4 unit test: the TypeScript decode path (fzstd -> msgpack) must
// correctly round-trip a real Python-produced snapshot fixture into
// the `Snapshot` interface with every field present and sane.
//
// Fixture is `target/sample_snapshot.mp` emitted by
// scripts/capture_sample_snapshot.py. The Rust integration test
// already asserts rmp-serde can decode it; this test asserts
// @msgpack/msgpack can too.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { decode as msgpackDecode } from "@msgpack/msgpack";
import { describe, expect, it } from "vitest";

import { SCHEMA_VERSION, type Snapshot } from "./snapshot";

const FIXTURE = resolve(__dirname, "../../../target/sample_snapshot.mp");

describe("TypeScript snapshot decode path", () => {
  it("decodes the Python-produced msgpack fixture into a well-typed Snapshot", () => {
    let bytes: Uint8Array;
    try {
      bytes = new Uint8Array(readFileSync(FIXTURE));
    } catch {
      // Fixture missing -- developer hasn't run capture_sample_snapshot
      // yet. Skip rather than hard-fail so first-time clones work.
      console.warn(
        `[snapshot.test] SKIP: fixture ${FIXTURE} missing; run scripts/capture_sample_snapshot.py`,
      );
      return;
    }

    // v5 wire format: temperature/alpha arrive as Uint8Array (msgpack
    // bin). The hook decoder converts them into Float32Arrays before
    // exposing the snapshot to React; this test mirrors that step.
    const raw = msgpackDecode(bytes) as Omit<
      Snapshot,
      "temperature" | "alpha"
    > & { temperature: Uint8Array; alpha: Uint8Array };
    expect(raw.version).toBe(SCHEMA_VERSION);
    expect(typeof raw.t_sim).toBe("number");
    expect(typeof raw.step).toBe("number");
    expect(typeof raw.is_rebuilding).toBe("boolean");
    expect(typeof raw.is_paused).toBe("boolean");

    // Grid metadata.
    expect(raw.grid_ds.nx).toBe(raw.grid.nx >> 1);
    expect(raw.grid_ds.ny).toBe(raw.grid.ny >> 1);
    expect(raw.grid_ds.nz).toBe(raw.grid.nz >> 1);

    // Buffer sizes: bin payload is 4 bytes/cell.
    const expectedLen = raw.grid_ds.nx * raw.grid_ds.ny * raw.grid_ds.nz;
    expect(raw.temperature.byteLength).toBe(expectedLen * 4);
    expect(raw.alpha.byteLength).toBe(expectedLen * 4);

    // Reinterpret bytes as f32 (test-side mirror of f32ArrayFromBytes).
    const tBuf = new ArrayBuffer(raw.temperature.byteLength);
    new Uint8Array(tBuf).set(raw.temperature);
    const temperature = new Float32Array(tBuf);
    const aBuf = new ArrayBuffer(raw.alpha.byteLength);
    new Uint8Array(aBuf).set(raw.alpha);
    const alpha = new Float32Array(aBuf);
    expect(temperature.length).toBe(expectedLen);
    expect(alpha.length).toBe(expectedLen);

    // Retention fields make physical sense.
    expect(raw.carrot_retention).toBeGreaterThanOrEqual(0);
    expect(raw.carrot_retention).toBeLessThanOrEqual(100.5);
    expect(raw.carrot_retention2).toBeGreaterThanOrEqual(0);
    expect(raw.carrot_retention2).toBeLessThanOrEqual(100.5);

    // Temperature should be in Celsius (sane range for a warm-start sim).
    // Math.min(...) on a 1.4M-element Float32Array can blow the call stack;
    // use a fold instead.
    let tMin = Infinity;
    let tMax = -Infinity;
    for (let i = 0; i < temperature.length; i++) {
      const v = temperature[i];
      if (v < tMin) tMin = v;
      if (v > tMax) tMax = v;
    }
    expect(tMin).toBeGreaterThanOrEqual(-5);
    expect(tMax).toBeLessThanOrEqual(200);

    // v4: pot geometry echo. Defaults come from the scenario YAML
    // (0.20 m diameter, 0.12 m height, 3 mm wall, 5 mm base); allow a
    // broad sanity band.
    expect(raw.pot_diameter_m).toBeGreaterThan(0.05);
    expect(raw.pot_diameter_m).toBeLessThan(0.60);
    expect(raw.pot_height_m).toBeGreaterThan(0.02);
    expect(raw.pot_height_m).toBeLessThan(0.50);
    expect(raw.pot_wall_thickness_m).toBeGreaterThan(0);
    expect(raw.pot_wall_thickness_m).toBeLessThan(raw.pot_diameter_m / 2);
    expect(raw.pot_base_thickness_m).toBeGreaterThan(0);
    expect(raw.pot_base_thickness_m).toBeLessThan(raw.pot_height_m);

    // v7: per-instance retention. Default fixture is 3 horizontal
    // carrots at t=5 steps -- per-instance vector should be populated
    // (length == carrot_count) and every entry near 100% (negligible
    // degradation in 5 steps).
    if (raw.carrot_count > 1) {
      expect(raw.carrot_retention_per_instance).toHaveLength(raw.carrot_count);
      for (const r of raw.carrot_retention_per_instance) {
        expect(r).toBeGreaterThan(99.0);
        expect(r).toBeLessThanOrEqual(100.5);
      }
    }
  });

  // NB: fzstd is decompress-only; the inverse (zstd encode) lives on the
  // Rust side and is covered by crates/ws-server/tests/ws_roundtrip.rs,
  // which wires up fake producer -> ws-server -> real WS client and
  // verifies that ws client decodes the zstd-compressed msgpack
  // successfully.
});
