// Verify packVolumeData correctly transposes Python C-order
// (nx, ny, nz) data into Three.js Data3DTexture order (x fastest,
// z slowest) while packing alpha+temperature into an RG byte buffer.
//
// Gate for M5: if this transpose is wrong, the volume renders
// axis-flipped and debugging by eye is miserable. We assert the map
// unit-cell-by-unit-cell on a small 2x3x4 fixture.

import { describe, expect, it } from "vitest";

import { packVolumeData } from "./WaterVolume";

describe("packVolumeData", () => {
  it("transposes (nx, ny, nz) C-order into Three.js Data3DTexture order", () => {
    // Tiny 2x3x4 grid (24 voxels). Python-side index:
    //   src = i*ny*nz + j*nz + k  (k fastest, i slowest)
    // so we can construct a temperature array where the value at src
    // encodes (i, j, k) and then assert the transpose landed it at
    // the correct Three.js texture coordinate (x=i, y=j, z=k).
    const nx = 2;
    const ny = 3;
    const nz = 4;
    const n = nx * ny * nz;
    const temperature = new Float32Array(n);
    const alpha = new Float32Array(n);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          const src = i * ny * nz + j * nz + k;
          // Encode (i, j, k) into temperature in a way that maps cleanly
          // into the [20, 100] Celsius band packVolumeData expects:
          //   T = 20 + 10*i + 3*j + 0.5*k
          // This stays inside [20, 33] so the normalised byte is < 32.
          temperature[src] = 20 + 10 * i + 3 * j + 0.5 * k;
          alpha[src] = 0.5; // mid-range so post-quantisation byte is ~127
        }
      }
    }
    const out = new Uint8Array(2 * n);
    packVolumeData(temperature, alpha, nx, ny, nz, out);

    // Now probe specific voxels and check the buffer landed them in
    // Three.js order: dst index = 2 * (x + y*nx + z*nx*ny).
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          const dst = 2 * (i + j * nx + k * nx * ny);
          // Alpha encoded as byte: 0.5 * 255 = 127.
          expect(out[dst + 0]).toBe(127);
          // Temperature normalised: (T - 20) / 80, then * 255.
          const expectedTn = (10 * i + 3 * j + 0.5 * k) / 80;
          const expectedByte = Math.max(0, Math.min(255, (expectedTn * 255) | 0));
          expect(out[dst + 1]).toBe(expectedByte);
        }
      }
    }
  });

  it("saturates the normalisation at the [20, 100] C band edges", () => {
    const out = new Uint8Array(2);
    // Hot value above 100 C should saturate at 255.
    packVolumeData([130], [1], 1, 1, 1, out);
    expect(out[0]).toBe(255); // alpha
    expect(out[1]).toBe(255); // temperature byte at saturation
    // Cold below 20 C should saturate at 0.
    const out2 = new Uint8Array(2);
    packVolumeData([0], [1], 1, 1, 1, out2);
    expect(out2[1]).toBe(0);
  });

  it("rejects mis-sized output buffers loudly", () => {
    expect(() =>
      packVolumeData([20, 20, 20], [1, 1, 1], 1, 1, 3, new Uint8Array(4)),
    ).toThrow(/out buffer length/);
  });

  it("rejects mismatched input lengths loudly", () => {
    expect(() =>
      packVolumeData([20, 20], [1, 1, 1], 1, 1, 3, new Uint8Array(6)),
    ).toThrow(/temperature length/);
    expect(() =>
      packVolumeData([20, 20, 20], [1, 1], 1, 1, 3, new Uint8Array(6)),
    ).toThrow(/alpha length/);
  });
});
