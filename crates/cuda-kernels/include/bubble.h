// Phase 5.6: shared Bubble struct + trilinear scatter helpers.
//
// The struct layout MUST match Warp's `@wp.struct class Bubble` at
// `python/boilingsim/boiling.py:38-58`. Field order, types, and sizes are
// validated by the parity tests in `python/tests/test_scatter_parity.py` --
// if a Warp upgrade changes the struct packing rule (today: natural-align,
// no padding), these tests fail loudly and we update both sides together.
//
// Memory layout reminder: Warp `wp.array3d` uses C-contiguous indexing
// `(i, j, k) -> (i * ny + j) * nz + k`. Same as the Jacobi kernel.

#ifndef BOILINGSIM_BUBBLE_H
#define BOILINGSIM_BUBBLE_H

#include <cuda_runtime.h>

// Total: 56 bytes (matches Warp's `wp.struct Bubble` packing).
// Members are all 4-byte aligned so the natural-align rule produces no
// padding -- a future Warp version that adds explicit alignment hints
// would need this layout updated. The parity tests catch any drift.
struct Bubble {
    float position[3];   // m, world-space centre
    float velocity[3];   // m/s
    float radius;        // m
    float birth_time;    // s
    int   active;        // 1 = live, 0 = empty slot
    int   site_i;        // nucleation-site grid index
    int   site_j;
    int   site_k;
    int   site_cleared;
    float departure_radius;
};

static_assert(sizeof(Bubble) == 56,
              "Bubble struct must be 56 bytes to match Warp's wp.struct Bubble layout");

__device__ __forceinline__ int bubble_idx3(int i, int j, int k, int ny, int nz) {
    return (i * ny + j) * nz + k;
}

// Trilinear scatter weight at fractional position (tx, ty, tz) in [0, 1)^3.
// Returns the 8 corner weights (w000..w111) that sum to 1. Mirrors the
// hand-unrolled trilinear in `scatter_latent_heat` at boiling.py:1403-1410
// exactly so the per-cell deposit is bit-equal up to the atomic-ordering
// non-determinism inherent to concurrent atomic_sub.
struct TrilinearWeights {
    float w000, w100, w010, w110, w001, w101, w011, w111;
};

__device__ __forceinline__ TrilinearWeights trilinear_weights(
    float tx, float ty, float tz
) {
    TrilinearWeights w;
    w.w000 = (1.0f - tx) * (1.0f - ty) * (1.0f - tz);
    w.w100 = tx         * (1.0f - ty) * (1.0f - tz);
    w.w010 = (1.0f - tx) * ty         * (1.0f - tz);
    w.w110 = tx         * ty         * (1.0f - tz);
    w.w001 = (1.0f - tx) * (1.0f - ty) * tz;
    w.w101 = tx         * (1.0f - ty) * tz;
    w.w011 = (1.0f - tx) * ty         * tz;
    w.w111 = tx         * ty         * tz;
    return w;
}

// Cell-centre trilinear sample of a float field. Mirror of
// `_sample_cell_scalar` at boiling.py:633.
__device__ __forceinline__ float sample_cell_scalar(
    const float* field, int nx, int ny, int nz,
    float px, float py, float pz,
    float ox, float oy, float oz,
    float dx
) {
    float fx = (px - ox) / dx - 0.5f;
    float fy = (py - oy) / dx - 0.5f;
    float fz = (pz - oz) / dx - 0.5f;

    fx = fminf(fmaxf(fx, 0.0f), (float)(nx - 1) - 1.0e-6f);
    fy = fminf(fmaxf(fy, 0.0f), (float)(ny - 1) - 1.0e-6f);
    fz = fminf(fmaxf(fz, 0.0f), (float)(nz - 1) - 1.0e-6f);

    int i0 = (int)fx, j0 = (int)fy, k0 = (int)fz;
    float tx = fx - (float)i0;
    float ty = fy - (float)j0;
    float tz = fz - (float)k0;

    float c000 = field[bubble_idx3(i0,     j0,     k0,     ny, nz)];
    float c100 = field[bubble_idx3(i0 + 1, j0,     k0,     ny, nz)];
    float c010 = field[bubble_idx3(i0,     j0 + 1, k0,     ny, nz)];
    float c110 = field[bubble_idx3(i0 + 1, j0 + 1, k0,     ny, nz)];
    float c001 = field[bubble_idx3(i0,     j0,     k0 + 1, ny, nz)];
    float c101 = field[bubble_idx3(i0 + 1, j0,     k0 + 1, ny, nz)];
    float c011 = field[bubble_idx3(i0,     j0 + 1, k0 + 1, ny, nz)];
    float c111 = field[bubble_idx3(i0 + 1, j0 + 1, k0 + 1, ny, nz)];

    float c00 = c000 * (1.0f - tx) + c100 * tx;
    float c10 = c010 * (1.0f - tx) + c110 * tx;
    float c01 = c001 * (1.0f - tx) + c101 * tx;
    float c11 = c011 * (1.0f - tx) + c111 * tx;
    float c0 = c00 * (1.0f - ty) + c10 * ty;
    float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

#endif  // BOILINGSIM_BUBBLE_H
