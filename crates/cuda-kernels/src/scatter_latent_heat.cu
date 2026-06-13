// Phase 5.6 M5.6.A: scatter latent-heat sink from each growing bubble into
// the 8 surrounding water cells via trilinear weights.
//
// Direct mirror of `scatter_latent_heat` at python/boilingsim/boiling.py:1328.
// The mat-gated `atomic_sub` pattern means bit-exact parity with Warp is
// impossible (atomic ordering is non-deterministic) -- the parity gate is
// statistical agreement within <1e-4 RMS plus exact sum-conservation.

#include "../include/bubble.h"

namespace {

constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float PI     = 3.14159265358979323846f;

}  // anonymous namespace

extern "C" __global__
void scatter_latent_heat_kernel(
    const Bubble* __restrict__ bubbles,
    int n_bubbles,
    float* T,                    // mutated via atomicAdd (no __restrict__ -- atomics)
    const int* __restrict__ mat,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float dx,
    float dt,
    float current_time,
    float rho_l, float cp_l,
    float rho_v, float h_lv,
    float T_sat_k,
    int mat_fluid
) {
    const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_idx >= n_bubbles) return;

    const Bubble bubble = bubbles[b_idx];
    if (bubble.active == 0) return;

    const float age = current_time - bubble.birth_time;
    if (age <= 1.0e-6f) return;

    // Temperature gate: only scatter when the local liquid is superheated.
    const float T_local = sample_cell_scalar(
        T, nx, ny, nz,
        bubble.position[0], bubble.position[1], bubble.position[2],
        ox, oy, oz, dx
    );
    if (T_local <= T_sat_k) return;

    // Analytic dR/dt from the Mikic-Rohsenow monotonic growth law (R ~ sqrt(t)).
    const float dR_dt = bubble.radius / (2.0f * age);
    if (dR_dt <= 0.0f) return;

    // Total latent-heat power and energy extracted this step.
    const float Q_b = rho_v * h_lv * 4.0f * PI
                    * bubble.radius * bubble.radius * dR_dt;
    const float E_step = Q_b * dt;

    const float cell_volume = dx * dx * dx;
    const float dT_ref = E_step / (rho_l * cp_l * cell_volume);

    // Trilinear cell-centre indexing.
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

    // Scatter via atomic_add with NEGATIVE deposits (mirrors Warp's atomic_sub).
    // The mat == mat_fluid gate matches boiling.py:1413-1428 exactly so the
    // non-fluid corners are skipped on both paths.
    #define SCATTER(di, dj, dk, weight)                                      \
        do {                                                                  \
            const int ii = i0 + (di);                                        \
            const int jj = j0 + (dj);                                        \
            const int kk = k0 + (dk);                                        \
            if (mat[bubble_idx3(ii, jj, kk, ny, nz)] == mat_fluid) {         \
                atomicAdd(&T[bubble_idx3(ii, jj, kk, ny, nz)],               \
                          -dT_ref * (weight));                                \
            }                                                                 \
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

extern "C" cudaError_t scatter_latent_heat_launch(
    const Bubble* bubbles,
    int n_bubbles,
    float* T,
    const int* mat,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float dx, float dt, float current_time,
    float rho_l, float cp_l, float rho_v, float h_lv, float T_sat_k,
    int mat_fluid
) {
    if (n_bubbles <= 0 || nx <= 0 || ny <= 0 || nz <= 0) return cudaSuccess;
    const int block = 256;
    const int grid = (n_bubbles + block - 1) / block;
    scatter_latent_heat_kernel<<<grid, block>>>(
        bubbles, n_bubbles, T, mat,
        nx, ny, nz, ox, oy, oz, dx, dt, current_time,
        rho_l, cp_l, rho_v, h_lv, T_sat_k, mat_fluid
    );
    return cudaGetLastError();
}
