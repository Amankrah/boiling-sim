// Phase 5.6 M5.6.C: scatter bubble VOF occupancy reduction onto water_alpha
// + finalizer that clamps alpha to [0, 1].
//
// Direct mirror of `reduce_water_alpha_by_bubble_occupancy` + `clamp_alpha_nonnegative`
// at python/boilingsim/boiling.py:1620 and :1687. Same atomic-sub trilinear
// pattern as scatter_latent_heat with a cell-centred (0.5, 0.5, 0.5) offset
// and mat-fluid gating.

#include "../include/bubble.h"

namespace {

constexpr float FOUR_THIRDS_PI = 4.18879020478639098f;

}  // anonymous namespace

extern "C" __global__
void reduce_water_alpha_kernel(
    const Bubble* __restrict__ bubbles,
    int n_bubbles,
    float* water_alpha,
    const int* __restrict__ mat,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float dx,
    int mat_fluid
) {
    const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_idx >= n_bubbles) return;

    const Bubble bubble = bubbles[b_idx];
    if (bubble.active == 0) return;

    const float R = bubble.radius;
    const float V_b = FOUR_THIRDS_PI * R * R * R;
    const float V_cell = dx * dx * dx;
    const float occ_ref = V_b / V_cell;

    float fx = (bubble.position[0] - ox) / dx - 0.5f;
    float fy = (bubble.position[1] - oy) / dx - 0.5f;
    float fz = (bubble.position[2] - oz) / dx - 0.5f;
    fx = fminf(fmaxf(fx, 0.0f), (float)(nx - 1) - 1.0e-6f);
    fy = fminf(fmaxf(fy, 0.0f), (float)(ny - 1) - 1.0e-6f);
    fz = fminf(fmaxf(fz, 0.0f), (float)(nz - 1) - 1.0e-6f);

    const int i0 = (int)fx, j0 = (int)fy, k0 = (int)fz;
    const float tx = fx - (float)i0;
    const float ty = fy - (float)j0;
    const float tz = fz - (float)k0;
    const TrilinearWeights w = trilinear_weights(tx, ty, tz);

    #define SCATTER(di, dj, dk, weight)                                       \
        do {                                                                   \
            const int ii = i0 + (di);                                         \
            const int jj = j0 + (dj);                                         \
            const int kk = k0 + (dk);                                         \
            if (mat[bubble_idx3(ii, jj, kk, ny, nz)] == mat_fluid) {          \
                atomicAdd(&water_alpha[bubble_idx3(ii, jj, kk, ny, nz)],      \
                          -occ_ref * (weight));                                \
            }                                                                  \
        } while (0)

    SCATTER(0, 0, 0, w.w000);
    SCATTER(1, 0, 0, w.w100);
    SCATTER(0, 1, 0, w.w010);
    SCATTER(1, 1, 0, w.w110);
    SCATTER(0, 0, 1, w.w001);
    SCATTER(1, 0, 1, w.w101);
    SCATTER(0, 1, 1, w.w011);
    SCATTER(1, 1, 1, w.w111);

    #undef SCATTER
}

extern "C" __global__
void clamp_alpha_kernel(float* water_alpha, int n_cells) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    const float a = water_alpha[idx];
    if (a < 0.0f) {
        water_alpha[idx] = 0.0f;
    } else if (a > 1.0f) {
        water_alpha[idx] = 1.0f;
    }
}

extern "C" cudaError_t reduce_water_alpha_launch(
    const Bubble* bubbles,
    int n_bubbles,
    float* water_alpha,
    const int* mat,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float dx,
    int mat_fluid
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return cudaSuccess;
    if (n_bubbles > 0) {
        const int block = 256;
        const int grid = (n_bubbles + block - 1) / block;
        reduce_water_alpha_kernel<<<grid, block>>>(
            bubbles, n_bubbles, water_alpha, mat,
            nx, ny, nz, ox, oy, oz, dx, mat_fluid
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    // Always run the clamp; mirrors the Warp path which clamps even when
    // n_active == 0 (the scatter may leave a tiny rounding residual).
    const int n_cells = nx * ny * nz;
    const int block = 256;
    const int grid = (n_cells + block - 1) / block;
    clamp_alpha_kernel<<<grid, block>>>(water_alpha, n_cells);
    return cudaGetLastError();
}
