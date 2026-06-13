// Phase 6 K1: laplacian SpMV for PCG.
//
// Computes `Ap = 6*p - s` (no 1/dx^2 scaling) where:
//   - 6 is the unit-diagonal of the discrete Poisson Laplacian
//   - s = sum of 6 neighbour pressures, with the same BC rules as
//     `_pressure_neighbour` in fluid.py:222-239:
//       fluid neighbour: use its pressure value
//       air neighbour: 0 (Dirichlet p=0 at free surface)
//       solid neighbour: p_self (Neumann ghost; effectively reduces the
//         row's diagonal by 1 per solid neighbour)
//
// SIGN AND SCALING (locked in the plan, do NOT change without re-deriving):
//   The Jacobi update at fluid.py:275 is `p_new = (s - rhs) / 6` where
//   `rhs = rho * dx*dx * div_u / dt`. Rearranging: `6*p - s = rhs`. The
//   linear system is therefore `A*p = b` with `A = 6*I - S` and
//   `b = rho * dx*dx * div_u / dt`. NO `1/dx^2` rescaling on either side --
//   we keep the unit-diagonal-of-6 form so the SpMV is bit-identical with
//   Jacobi's `Ax - b` residual.
//
// Non-fluid cells: write Ap = 0 (same convention as Jacobi which writes 0
// to p_new for non-fluid cells at fluid.py:262).
//
// Threading: identical to jacobi_pressure.cu after Phase 5.5 tuning.
//   TILE_K=32 (warp-coalesced k axis) x TILE_J=4 x TILE_I=4 = 512 threads.
//   __launch_bounds__(512, 2) keeps 2 blocks per SM for occupancy.

#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int spmv_idx3(int i, int j, int k, int ny, int nz) {
    return (i * ny + j) * nz + k;
}

// Mirror of `_pressure_neighbour` in fluid.py:222-239. Same as the helper in
// jacobi_pressure.cu (anonymous namespace there); duplicated here to keep
// each .cu file self-contained at compile time.
__device__ __forceinline__ float spmv_neighbour(
    int m_nbr, float p_nbr, float p_self,
    int mat_fluid, int mat_air
) {
    if (m_nbr == mat_fluid) return p_nbr;
    if (m_nbr == mat_air)   return 0.0f;
    return p_self;
}

}  // anonymous namespace

constexpr int SPMV_TILE_K = 32;
constexpr int SPMV_TILE_J = 4;
constexpr int SPMV_TILE_I = 4;
constexpr int SPMV_SH_K = SPMV_TILE_K + 2;
constexpr int SPMV_SH_J = SPMV_TILE_J + 2;
constexpr int SPMV_SH_I = SPMV_TILE_I + 2;

__device__ __forceinline__ int spmv_sh_idx(int si, int sj, int sk) {
    return (si * SPMV_SH_J + sj) * SPMV_SH_K + sk;
}

extern "C" __global__
__launch_bounds__(SPMV_TILE_I * SPMV_TILE_J * SPMV_TILE_K, 2)
void laplacian_spmv_kernel(
    float* __restrict__ Ap,
    const float* __restrict__ p,
    const int*   __restrict__ mat,
    int nx, int ny, int nz,
    int mat_fluid, int mat_air
) {
    __shared__ float sh_p[SPMV_SH_I * SPMV_SH_J * SPMV_SH_K];
    __shared__ int   sh_m[SPMV_SH_I * SPMV_SH_J * SPMV_SH_K];

    const int tile_k0 = blockIdx.x * SPMV_TILE_K;
    const int tile_j0 = blockIdx.y * SPMV_TILE_J;
    const int tile_i0 = blockIdx.z * SPMV_TILE_I;

    // Cooperative load of the (TILE+2)^3 halo-inclusive block.
    const int tid = threadIdx.x
                  + threadIdx.y * SPMV_TILE_K
                  + threadIdx.z * SPMV_TILE_K * SPMV_TILE_J;
    const int total = SPMV_SH_I * SPMV_SH_J * SPMV_SH_K;
    const int block_threads = SPMV_TILE_I * SPMV_TILE_J * SPMV_TILE_K;
    for (int idx = tid; idx < total; idx += block_threads) {
        const int sk = idx % SPMV_SH_K;
        const int sj = (idx / SPMV_SH_K) % SPMV_SH_J;
        const int si = idx / (SPMV_SH_K * SPMV_SH_J);
        const int gi = tile_i0 + si - 1;
        const int gj = tile_j0 + sj - 1;
        const int gk = tile_k0 + sk - 1;
        if (gi >= 0 && gi < nx && gj >= 0 && gj < ny && gk >= 0 && gk < nz) {
            const int g = spmv_idx3(gi, gj, gk, ny, nz);
            sh_p[idx] = p[g];
            sh_m[idx] = mat[g];
        } else {
            sh_p[idx] = 0.0f;
            sh_m[idx] = mat_air;
        }
    }
    __syncthreads();

    const int lk = threadIdx.x;
    const int lj = threadIdx.y;
    const int li = threadIdx.z;
    const int i = tile_i0 + li;
    const int j = tile_j0 + lj;
    const int k = tile_k0 + lk;
    if (i >= nx || j >= ny || k >= nz) return;

    const int si = li + 1;
    const int sj = lj + 1;
    const int sk = lk + 1;
    const int center_sh = spmv_sh_idx(si, sj, sk);

    if (sh_m[center_sh] != mat_fluid) {
        Ap[spmv_idx3(i, j, k, ny, nz)] = 0.0f;
        return;
    }

    const float p_self = sh_p[center_sh];

    // Same neighbour order as fluid.py:268-272 / jacobi_pressure.cu:178-185.
    const float p_left  = spmv_neighbour(sh_m[spmv_sh_idx(si - 1, sj, sk)],
                                          sh_p[spmv_sh_idx(si - 1, sj, sk)],
                                          p_self, mat_fluid, mat_air);
    const float p_right = spmv_neighbour(sh_m[spmv_sh_idx(si + 1, sj, sk)],
                                          sh_p[spmv_sh_idx(si + 1, sj, sk)],
                                          p_self, mat_fluid, mat_air);
    const float p_down  = spmv_neighbour(sh_m[spmv_sh_idx(si, sj - 1, sk)],
                                          sh_p[spmv_sh_idx(si, sj - 1, sk)],
                                          p_self, mat_fluid, mat_air);
    const float p_up    = spmv_neighbour(sh_m[spmv_sh_idx(si, sj + 1, sk)],
                                          sh_p[spmv_sh_idx(si, sj + 1, sk)],
                                          p_self, mat_fluid, mat_air);
    const float p_back  = spmv_neighbour(sh_m[spmv_sh_idx(si, sj, sk - 1)],
                                          sh_p[spmv_sh_idx(si, sj, sk - 1)],
                                          p_self, mat_fluid, mat_air);
    const float p_front = spmv_neighbour(sh_m[spmv_sh_idx(si, sj, sk + 1)],
                                          sh_p[spmv_sh_idx(si, sj, sk + 1)],
                                          p_self, mat_fluid, mat_air);

    const float s = p_left + p_right + p_down + p_up + p_back + p_front;
    // Ap = 6*p - s. Unit diagonal, no dx^2.
    Ap[spmv_idx3(i, j, k, ny, nz)] = 6.0f * p_self - s;
}

extern "C" cudaError_t laplacian_spmv_launch(
    float* Ap,
    const float* p,
    const int* mat,
    int nx, int ny, int nz,
    int mat_fluid, int mat_air
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return cudaSuccess;
    const dim3 block(SPMV_TILE_K, SPMV_TILE_J, SPMV_TILE_I);
    const dim3 grid(
        (nz + SPMV_TILE_K - 1) / SPMV_TILE_K,
        (ny + SPMV_TILE_J - 1) / SPMV_TILE_J,
        (nx + SPMV_TILE_I - 1) / SPMV_TILE_I
    );
    laplacian_spmv_kernel<<<grid, block>>>(
        Ap, p, mat, nx, ny, nz, mat_fluid, mat_air
    );
    return cudaGetLastError();
}
