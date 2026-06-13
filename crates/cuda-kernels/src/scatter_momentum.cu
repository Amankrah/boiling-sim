// Phase 5.6 M5.6.B: scatter each bubble's excess-buoyancy upward force
// across the 8 surrounding z-faces of the MAC grid.
//
// Direct mirror of `scatter_bubble_momentum` at python/boilingsim/boiling.py:1538.
// Same atomic_add pattern as scatter_latent_heat -- parity gate is statistical,
// not bit-exact.
//
// Key differences from scatter_latent_heat:
//   * Target field is `uz` (z-face velocities) at shape (nx, ny, nz+1).
//   * Cell-centre offset is (0.5, 0.5, 0.0) -- z is NOT offset because
//     uz lives on z-faces.
//   * Material guard checks the UPPER cell of the face (mat[i,j,k]) instead
//     of the face cell itself.

#include "../include/bubble.h"

namespace {

constexpr float FOUR_THIRDS_PI = 4.18879020478639098f;  // 4/3 * pi

}  // anonymous namespace

extern "C" __global__
void scatter_momentum_kernel(
    const Bubble* __restrict__ bubbles,
    int n_bubbles,
    float* uz,                   // atomic_add target; mat shape (nx, ny, nz+1)
    const int* __restrict__ mat, // shape (nx, ny, nz)
    int nx, int ny, int nz,      // mat shape
    int uz_nx, int uz_ny, int uz_nz,  // uz shape: (nx, ny, nz+1)
    float ox, float oy, float oz,
    float dx,
    float dt,
    float rho_l, float rho_v, float g_mag,
    int mat_fluid
) {
    const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_idx >= n_bubbles) return;

    const Bubble bubble = bubbles[b_idx];
    if (bubble.active == 0) return;

    const float R = bubble.radius;
    const float V_b = FOUR_THIRDS_PI * R * R * R;
    const float F_total = V_b * (rho_l - rho_v) * g_mag;
    const float cell_volume = dx * dx * dx;
    const float dUz_ref = F_total * dt / (rho_l * cell_volume);

    // uz lives at (i+0.5, j+0.5, k)*dx world positions -> z-face offset (0.5, 0.5, 0.0).
    float fx = (bubble.position[0] - ox) / dx - 0.5f;
    float fy = (bubble.position[1] - oy) / dx - 0.5f;
    float fz = (bubble.position[2] - oz) / dx;
    fx = fminf(fmaxf(fx, 0.0f), (float)(uz_nx - 1) - 1.0e-6f);
    fy = fminf(fmaxf(fy, 0.0f), (float)(uz_ny - 1) - 1.0e-6f);
    fz = fminf(fmaxf(fz, 0.0f), (float)(uz_nz - 1) - 1.0e-6f);

    const int i0 = (int)fx, j0 = (int)fy, k0 = (int)fz;
    const float tx = fx - (float)i0;
    const float ty = fy - (float)j0;
    const float tz = fz - (float)k0;
    const TrilinearWeights w = trilinear_weights(tx, ty, tz);

    // Each scatter checks (a) k0 valid for the face (k0>0 for lower set or
    // k0+1<nz for upper set), (b) i/j in bounds, (c) the face's UPPER cell
    // is fluid. Mirror the boiling.py:1601-1616 guards exactly.

    #define UZ_IDX(i, j, k) (((i) * uz_ny + (j)) * uz_nz + (k))
    #define MAT_IDX(i, j, k) (((i) * ny + (j)) * nz + (k))

    // k = k0 layer (faces between cell k0-1 and k0; mat lookup at [i,j,k0]):
    if (k0 > 0 && mat[MAT_IDX(i0, j0, k0)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0, j0, k0)], dUz_ref * w.w000);
    }
    if (k0 > 0 && i0 + 1 < nx && mat[MAT_IDX(i0 + 1, j0, k0)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0 + 1, j0, k0)], dUz_ref * w.w100);
    }
    if (k0 > 0 && j0 + 1 < ny && mat[MAT_IDX(i0, j0 + 1, k0)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0, j0 + 1, k0)], dUz_ref * w.w010);
    }
    if (k0 > 0 && i0 + 1 < nx && j0 + 1 < ny
        && mat[MAT_IDX(i0 + 1, j0 + 1, k0)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0 + 1, j0 + 1, k0)], dUz_ref * w.w110);
    }
    // k = k0+1 layer (faces between cell k0 and k0+1; mat lookup at [i,j,k0+1]):
    if (k0 + 1 < nz && mat[MAT_IDX(i0, j0, k0 + 1)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0, j0, k0 + 1)], dUz_ref * w.w001);
    }
    if (k0 + 1 < nz && i0 + 1 < nx
        && mat[MAT_IDX(i0 + 1, j0, k0 + 1)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0 + 1, j0, k0 + 1)], dUz_ref * w.w101);
    }
    if (k0 + 1 < nz && j0 + 1 < ny
        && mat[MAT_IDX(i0, j0 + 1, k0 + 1)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0, j0 + 1, k0 + 1)], dUz_ref * w.w011);
    }
    if (k0 + 1 < nz && i0 + 1 < nx && j0 + 1 < ny
        && mat[MAT_IDX(i0 + 1, j0 + 1, k0 + 1)] == mat_fluid) {
        atomicAdd(&uz[UZ_IDX(i0 + 1, j0 + 1, k0 + 1)], dUz_ref * w.w111);
    }

    #undef UZ_IDX
    #undef MAT_IDX
}

extern "C" cudaError_t scatter_momentum_launch(
    const Bubble* bubbles,
    int n_bubbles,
    float* uz,
    const int* mat,
    int nx, int ny, int nz,
    int uz_nx, int uz_ny, int uz_nz,
    float ox, float oy, float oz,
    float dx, float dt,
    float rho_l, float rho_v, float g_mag,
    int mat_fluid
) {
    if (n_bubbles <= 0) return cudaSuccess;
    const int block = 256;
    const int grid = (n_bubbles + block - 1) / block;
    scatter_momentum_kernel<<<grid, block>>>(
        bubbles, n_bubbles, uz, mat,
        nx, ny, nz, uz_nx, uz_ny, uz_nz,
        ox, oy, oz, dx, dt, rho_l, rho_v, g_mag, mat_fluid
    );
    return cudaGetLastError();
}
