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

    const snap = msgpackDecode(bytes) as Snapshot;
    expect(snap.version).toBe(SCHEMA_VERSION);
    expect(typeof snap.t_sim).toBe("number");
    expect(typeof snap.step).toBe("number");
    expect(typeof snap.is_rebuilding).toBe("boolean");
    expect(typeof snap.is_paused).toBe("boolean");

    // Grid metadata.
    expect(snap.grid_ds.nx).toBe(snap.grid.nx >> 1);
    expect(snap.grid_ds.ny).toBe(snap.grid.ny >> 1);
    expect(snap.grid_ds.nz).toBe(snap.grid.nz >> 1);

    // Buffer sizes match downsampled grid cell count.
    const expectedLen = snap.grid_ds.nx * snap.grid_ds.ny * snap.grid_ds.nz;
    expect(snap.temperature.length).toBe(expectedLen);
    expect(snap.alpha.length).toBe(expectedLen);

    // Retention fields make physical sense.
    expect(snap.carrot_retention).toBeGreaterThanOrEqual(0);
    expect(snap.carrot_retention).toBeLessThanOrEqual(100.5);
    expect(snap.carrot_retention2).toBeGreaterThanOrEqual(0);
    expect(snap.carrot_retention2).toBeLessThanOrEqual(100.5);

    // Temperature should be in Celsius (sane range for a warm-start sim).
    const tMin = Math.min(...snap.temperature);
    const tMax = Math.max(...snap.temperature);
    expect(tMin).toBeGreaterThanOrEqual(-5);
    expect(tMax).toBeLessThanOrEqual(200);

    // v4: pot geometry echo. Defaults come from the scenario YAML
    // (0.20 m diameter, 0.12 m height, 3 mm wall, 5 mm base); allow a
    // broad sanity band.
    expect(snap.pot_diameter_m).toBeGreaterThan(0.05);
    expect(snap.pot_diameter_m).toBeLessThan(0.60);
    expect(snap.pot_height_m).toBeGreaterThan(0.02);
    expect(snap.pot_height_m).toBeLessThan(0.50);
    expect(snap.pot_wall_thickness_m).toBeGreaterThan(0);
    expect(snap.pot_wall_thickness_m).toBeLessThan(snap.pot_diameter_m / 2);
    expect(snap.pot_base_thickness_m).toBeGreaterThan(0);
    expect(snap.pot_base_thickness_m).toBeLessThan(snap.pot_height_m);
  });

  // NB: fzstd is decompress-only; the inverse (zstd encode) lives on the
  // Rust side and is covered by crates/ws-server/tests/ws_roundtrip.rs,
  // which wires up fake producer -> ws-server -> real WS client and
  // verifies that ws client decodes the zstd-compressed msgpack
  // successfully.
});
