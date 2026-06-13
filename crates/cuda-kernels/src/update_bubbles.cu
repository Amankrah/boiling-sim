// Phase 5.7: per-bubble update kernel.
//
// Direct mirror of `update_bubbles` at python/boilingsim/boiling.py:771-965.
// Eight phases per bubble (active check, T sample, Mikic-Rohsenow growth OR
// Plesset-Zwick condensation w/ embedded 8-cell atomic_add scatter to T,
// fragmentation flag, Fritz departure, terminal-slip advection, free-surface
// vent, solid-contact deactivation).
//
// Parity strategy (see test_update_bubbles_parity.py):
//   - Per-thread Bubble field writes: bit-exact vs Warp (same arithmetic
//     order, same FMA discipline -- but build.rs default has --fmad enabled
//     so we accept up to ~1 ULP per FMA across the helpers).
//   - T scatter via atomicAdd: statistical parity (sum-conservation
//     deterministic, per-cell ordering non-deterministic).
//   - needs_fragment, slot_claim: per-thread bit-exact.
//   - site_active: USES atomicCAS (changes Warp's racy plain-assign to
//     strictly serial). Bubble.site_cleared is still set inside the
//     if-branch regardless of CAS outcome so the per-thread bubble field
//     parity gate holds. See plan: this is the chosen user behaviour.

#include "../include/bubble.h"

namespace {

// 2/sqrt(pi), the Mikic-Rohsenow / Plesset-Zwick prefactor.
constexpr float MR_PREFACTOR = 1.1283791670955126f;
constexpr float FOUR_PI_OVER_3 = 4.18879020478639098f;  // 4*pi/3
constexpr float DEG_PER_RAD    = 57.29577951308232f;

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Helper functions (mirror boiling.py wp.funcs)
// ---------------------------------------------------------------------------

// Mirror of mikic_rohsenow_radius at boiling.py:185-201.
__device__ __forceinline__ float mikic_rohsenow_radius(
    float age_s, float T_local_k, float T_sat_k,
    float rho_l, float rho_v, float cp_l, float k_l, float h_lv
) {
    float dT = T_local_k - T_sat_k;
    if (dT <= 0.0f) return 0.0f;
    float Ja = rho_l * cp_l * dT / (rho_v * h_lv);
    float alpha_l = k_l / (rho_l * cp_l);
    return MR_PREFACTOR * Ja * sqrtf(alpha_l * age_s);
}

// Mirror of _condensation_decrement at boiling.py:204-238.
__device__ __forceinline__ float condensation_decrement(
    float R, float T_local_k, float T_sat_k,
    float rho_l, float rho_v, float cp_l, float k_l, float h_lv,
    float dt
) {
    float dT_sub = T_sat_k - T_local_k;
    if (dT_sub <= 0.0f) return 0.0f;
    if (R <= 0.0f) return 0.0f;
    float Ja_sub = rho_l * cp_l * dT_sub / (rho_v * h_lv);
    float alpha_l = k_l / (rho_l * cp_l);
    float rate = MR_PREFACTOR * Ja_sub * alpha_l / R;
    return rate * dt;
}

// Mirror of fritz_departure_diameter at boiling.py:137-146.
__device__ __forceinline__ float fritz_departure_diameter(
    float theta_rad, float sigma, float g_mag, float rho_l, float rho_v
) {
    float theta_deg = theta_rad * DEG_PER_RAD;
    return 0.0208f * theta_deg * sqrtf(sigma / (g_mag * (rho_l - rho_v)));
}

// Mirror of terminal_slip_velocity at boiling.py:156-182.
__device__ __forceinline__ float terminal_slip_velocity(float R) {
    float v_pow = 391.0f * powf(R, 1.26f);
    return (v_pow > 0.22f) ? 0.22f : v_pow;
}

// Mirror of _mat_at_point at boiling.py:749-763.
__device__ __forceinline__ int mat_at_point(
    const int* mat, int mat_nx, int mat_ny, int mat_nz,
    float px, float py, float pz,
    float ox, float oy, float oz,
    float dx
) {
    float fx = (px - ox) / dx - 0.5f;
    float fy = (py - oy) / dx - 0.5f;
    float fz = (pz - oz) / dx - 0.5f;
    int i = (int)(fx + 0.5f);
    int j = (int)(fy + 0.5f);
    int k = (int)(fz + 0.5f);
    if (i < 0) i = 0; if (i > mat_nx - 1) i = mat_nx - 1;
    if (j < 0) j = 0; if (j > mat_ny - 1) j = mat_ny - 1;
    if (k < 0) k = 0; if (k > mat_nz - 1) k = mat_nz - 1;
    return mat[bubble_idx3(i, j, k, mat_ny, mat_nz)];
}

// Trilinear sample of one MAC-grid axis at an offset position. Used by
// sample_face_u below for each of (ux, uy, uz). The offset (fx_off, fy_off,
// fz_off) is the cell-centre vs face-centre offset, which differs per axis.
// Mirror of the per-axis blocks in _sample_face_u at boiling.py:677-746.
__device__ __forceinline__ float sample_face_component(
    const float* field, int fnx, int fny, int fnz,
    float px, float py, float pz,
    float ox, float oy, float oz,
    float dx,
    float fx_off, float fy_off, float fz_off
) {
    float fx = (px - ox) / dx - fx_off;
    float fy = (py - oy) / dx - fy_off;
    float fz = (pz - oz) / dx - fz_off;
    fx = fminf(fmaxf(fx, 0.0f), (float)(fnx - 1) - 1.0e-6f);
    fy = fminf(fmaxf(fy, 0.0f), (float)(fny - 1) - 1.0e-6f);
    fz = fminf(fmaxf(fz, 0.0f), (float)(fnz - 1) - 1.0e-6f);
    int i0 = (int)fx, j0 = (int)fy, k0 = (int)fz;
    float tx = fx - (float)i0;
    float ty = fy - (float)j0;
    float tz = fz - (float)k0;
    // Warp uses the unrolled-summation form (see boiling.py:700-705): each
    // pair on a row is added, then multiplied by (1-ty)(1-tz) etc. Preserve
    // this exact sum order for bit-parity.
    float c000 = field[bubble_idx3(i0,     j0,     k0,     fny, fnz)];
    float c100 = field[bubble_idx3(i0 + 1, j0,     k0,     fny, fnz)];
    float c010 = field[bubble_idx3(i0,     j0 + 1, k0,     fny, fnz)];
    float c110 = field[bubble_idx3(i0 + 1, j0 + 1, k0,     fny, fnz)];
    float c001 = field[bubble_idx3(i0,     j0,     k0 + 1, fny, fnz)];
    float c101 = field[bubble_idx3(i0 + 1, j0,     k0 + 1, fny, fnz)];
    float c011 = field[bubble_idx3(i0,     j0 + 1, k0 + 1, fny, fnz)];
    float c111 = field[bubble_idx3(i0 + 1, j0 + 1, k0 + 1, fny, fnz)];
    return ((c000 * (1.0f - tx) + c100 * tx) * (1.0f - ty) * (1.0f - tz)
          + (c010 * (1.0f - tx) + c110 * tx) *         ty  * (1.0f - tz)
          + (c001 * (1.0f - tx) + c101 * tx) * (1.0f - ty) *         tz
          + (c011 * (1.0f - tx) + c111 * tx) *         ty  *         tz);
}

// Mirror of _sample_face_u at boiling.py:677-746. ux/uy/uz each have a
// different shape -- caller passes three triples.
__device__ __forceinline__ void sample_face_u(
    const float* ux, int ux_nx, int ux_ny, int ux_nz,
    const float* uy, int uy_nx, int uy_ny, int uy_nz,
    const float* uz, int uz_nx, int uz_ny, int uz_nz,
    float px, float py, float pz,
    float ox, float oy, float oz,
    float dx,
    float* out_u, float* out_v, float* out_w
) {
    // x-face: offset (0.0, 0.5, 0.5).
    *out_u = sample_face_component(
        ux, ux_nx, ux_ny, ux_nz,
        px, py, pz, ox, oy, oz, dx,
        0.0f, 0.5f, 0.5f
    );
    // y-face: offset (0.5, 0.0, 0.5).
    *out_v = sample_face_component(
        uy, uy_nx, uy_ny, uy_nz,
        px, py, pz, ox, oy, oz, dx,
        0.5f, 0.0f, 0.5f
    );
    // z-face: offset (0.5, 0.5, 0.0).
    *out_w = sample_face_component(
        uz, uz_nx, uz_ny, uz_nz,
        px, py, pz, ox, oy, oz, dx,
        0.5f, 0.5f, 0.0f
    );
}

// ---------------------------------------------------------------------------
// Main kernel: per-bubble update
// ---------------------------------------------------------------------------

extern "C" __global__
void update_bubbles_kernel(
    Bubble* bubbles,
    int n_bubbles,
    int* slot_claim,
    int* site_active,
    int* needs_fragment,
    const float* T,
    const int* mat,
    int nx, int ny, int nz,             // T / mat / site_active shape
    const float* ux,
    int ux_nx, int ux_ny, int ux_nz,
    const float* uy,
    int uy_nx, int uy_ny, int uy_nz,
    const float* uz,
    int uz_nx, int uz_ny, int uz_nz,
    float ox, float oy, float oz,
    float dx, float dt,
    float current_time,
    float water_line_z,
    float T_sat_k,
    float rho_l, float rho_v,
    float cp_l, float k_l, float h_lv,
    float sigma,
    float theta_rad, float g_mag,
    float R_seed, float R_frag, float R_max,
    int mat_fluid
) {
    const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_idx >= n_bubbles) return;

    Bubble bubble = bubbles[b_idx];
    if (bubble.active == 0) return;

    // ----- Phase 1: trilinear T sample at bubble centre -----
    float T_local = sample_cell_scalar(
        T, nx, ny, nz,
        bubble.position[0], bubble.position[1], bubble.position[2],
        ox, oy, oz, dx
    );

    if (T_local > T_sat_k) {
        // ----- Phase 2A: Mikic-Rohsenow growth -----
        float age = current_time - bubble.birth_time;
        if (age < 0.0f) age = 0.0f;
        float R_target = mikic_rohsenow_radius(
            age, T_local, T_sat_k, rho_l, rho_v, cp_l, k_l, h_lv
        );
        if (R_target > bubble.radius) {
            bubble.radius = R_target;
        }
    } else {
        // ----- Phase 2B: condensation + embedded latent-heat scatter -----
        float R_old = bubble.radius;
        float dR = condensation_decrement(
            R_old, T_local, T_sat_k,
            rho_l, rho_v, cp_l, k_l, h_lv, dt
        );
        float R_new_raw = R_old - dR;
        bool fully_condensed = (R_new_raw <= R_seed);
        float R_new = fully_condensed ? R_seed : R_new_raw;

        // Volume lost this step and embedded 8-cell atomicAdd scatter to T.
        float V_lost = FOUR_PI_OVER_3 * (R_old * R_old * R_old - R_new * R_new * R_new);
        if (V_lost > 0.0f) {
            float E_release = rho_v * h_lv * V_lost;
            float cell_volume = dx * dx * dx;
            float dT_ref = E_release / (rho_l * cp_l * cell_volume);

            float fx = (bubble.position[0] - ox) / dx - 0.5f;
            float fy = (bubble.position[1] - oy) / dx - 0.5f;
            float fz = (bubble.position[2] - oz) / dx - 0.5f;
            fx = fminf(fmaxf(fx, 0.0f), (float)(nx - 1) - 1.0e-6f);
            fy = fminf(fmaxf(fy, 0.0f), (float)(ny - 1) - 1.0e-6f);
            fz = fminf(fmaxf(fz, 0.0f), (float)(nz - 1) - 1.0e-6f);
            int i0 = (int)fx, j0 = (int)fy, k0 = (int)fz;
            float tx = fx - (float)i0, ty = fy - (float)j0, tz = fz - (float)k0;
            TrilinearWeights w = trilinear_weights(tx, ty, tz);

            #define SCATTER_T(di, dj, dk, weight)                              \
                do {                                                            \
                    int ii = i0 + (di), jj = j0 + (dj), kk = k0 + (dk);        \
                    if (mat[bubble_idx3(ii, jj, kk, ny, nz)] == mat_fluid) {   \
                        atomicAdd(&((float*)T)[bubble_idx3(ii, jj, kk, ny, nz)], \
                                  dT_ref * (weight));                           \
                    }                                                            \
                } while (0)

            SCATTER_T(0, 0, 0, w.w000);
            SCATTER_T(1, 0, 0, w.w100);
            SCATTER_T(0, 1, 0, w.w010);
            SCATTER_T(1, 1, 0, w.w110);
            SCATTER_T(0, 0, 1, w.w001);
            SCATTER_T(1, 0, 1, w.w101);
            SCATTER_T(0, 1, 1, w.w011);
            SCATTER_T(1, 1, 1, w.w111);

            #undef SCATTER_T
        }

        if (fully_condensed) {
            // Clear site (with atomicCAS to fix the Warp race) and deactivate.
            if (bubble.site_cleared == 0 && bubble.site_i >= 0) {
                // atomicCAS swaps site_active from 1 -> 0 if it was 1,
                // returning the old value. The bubble.site_cleared flag is
                // set to 1 regardless of the CAS outcome (preserves Warp's
                // per-bubble semantic "I think I cleared this site").
                atomicCAS(&site_active[bubble_idx3(
                    bubble.site_i, bubble.site_j, bubble.site_k, ny, nz
                )], 1, 0);
                bubble.site_cleared = 1;
            }
            bubble.active = 0;
            bubble.radius = 0.0f;
            bubbles[b_idx] = bubble;
            slot_claim[b_idx] = 0;
            return;
        } else {
            bubble.radius = R_new;
        }
    }

    // ----- Phase 3: fragmentation flag + R_max cap -----
    if (bubble.radius > R_frag) {
        needs_fragment[b_idx] = 1;
    }
    if (bubble.radius > R_max) {
        bubble.radius = R_max;
    }

    // ----- Phase 4: Fritz departure check -----
    float D_d = fritz_departure_diameter(theta_rad, sigma, g_mag, rho_l, rho_v);
    bool departed = (2.0f * bubble.radius >= D_d);

    if (departed) {
        if (bubble.site_cleared == 0 && bubble.site_i >= 0) {
            atomicCAS(&site_active[bubble_idx3(
                bubble.site_i, bubble.site_j, bubble.site_k, ny, nz
            )], 1, 0);
            bubble.site_cleared = 1;
            bubble.departure_radius = bubble.radius;
        }
    }

    // ----- Phase 5: terminal-slip advection (departed bubbles only) -----
    if (departed) {
        float u_val, v_val, w_val;
        sample_face_u(
            ux, ux_nx, ux_ny, ux_nz,
            uy, uy_nx, uy_ny, uy_nz,
            uz, uz_nx, uz_ny, uz_nz,
            bubble.position[0], bubble.position[1], bubble.position[2],
            ox, oy, oz, dx,
            &u_val, &v_val, &w_val
        );
        float slip = terminal_slip_velocity(bubble.radius);
        bubble.velocity[0] = u_val;
        bubble.velocity[1] = v_val;
        bubble.velocity[2] = w_val + slip;
        bubble.position[0] = bubble.position[0] + bubble.velocity[0] * dt;
        bubble.position[1] = bubble.position[1] + bubble.velocity[1] * dt;
        bubble.position[2] = bubble.position[2] + bubble.velocity[2] * dt;
    }

    // ----- Phase 6: vent at free surface -----
    if (bubble.position[2] >= water_line_z) {
        bubble.active = 0;
        bubbles[b_idx] = bubble;
        slot_claim[b_idx] = 0;
        return;
    }

    // ----- Phase 7: solid-contact deactivation -----
    int m_here = mat_at_point(
        mat, nx, ny, nz,
        bubble.position[0], bubble.position[1], bubble.position[2],
        ox, oy, oz, dx
    );
    if (m_here != mat_fluid) {
        bubble.active = 0;
        bubbles[b_idx] = bubble;
        slot_claim[b_idx] = 0;
        return;
    }

    // ----- Phase 8: final commit -----
    bubbles[b_idx] = bubble;
}

extern "C" cudaError_t update_bubbles_launch(
    Bubble* bubbles,
    int n_bubbles,
    int* slot_claim,
    int* site_active,
    int* needs_fragment,
    const float* T,
    const int* mat,
    int nx, int ny, int nz,
    const float* ux, int ux_nx, int ux_ny, int ux_nz,
    const float* uy, int uy_nx, int uy_ny, int uy_nz,
    const float* uz, int uz_nx, int uz_ny, int uz_nz,
    float ox, float oy, float oz,
    float dx, float dt,
    float current_time,
    float water_line_z,
    float T_sat_k,
    float rho_l, float rho_v,
    float cp_l, float k_l, float h_lv,
    float sigma,
    float theta_rad, float g_mag,
    float R_seed, float R_frag, float R_max,
    int mat_fluid
) {
    if (n_bubbles <= 0) return cudaSuccess;
    const int block = 256;
    const int grid = (n_bubbles + block - 1) / block;
    update_bubbles_kernel<<<grid, block>>>(
        bubbles, n_bubbles, slot_claim, site_active, needs_fragment,
        T, mat, nx, ny, nz,
        ux, ux_nx, ux_ny, ux_nz,
        uy, uy_nx, uy_ny, uy_nz,
        uz, uz_nx, uz_ny, uz_nz,
        ox, oy, oz, dx, dt, current_time, water_line_z,
        T_sat_k, rho_l, rho_v, cp_l, k_l, h_lv, sigma,
        theta_rad, g_mag, R_seed, R_frag, R_max,
        mat_fluid
    );
    return cudaGetLastError();
}
