// Phase 5 M2: hand-written Jacobi pressure kernel.
//
// Direct mirror of `python/boilingsim/fluid.py:jacobi_pressure_step` -- the
// goal is bit-tolerant parity against the Warp reference, NOT raw speed.
// Sum order, neighbour-lookup branch, BC handling, and the divisor are all
// preserved exactly. If you change the arithmetic here you break the M2
// acceptance gate; see `python/tests/test_pressure_parity.py`.
//
// Threading model:
//   * One thread per cell (i, j, k).
//   * 3D block of (BX, BY, BZ) = (8, 8, 8) = 512 threads.
//   * Grid covers (nx, ny, nz) with ceil-div.
//   * No shared memory tiling in M2 -- bit-tolerance comes first. Tiling
//     is deferred until M3 perf measurement says it's worth the complexity.
//
// Array layout:
//   Warp's wp.array3d is C-contiguous: stride = (ny*nz, nz, 1). Indexing
//   `arr[i, j, k]` maps to `arr_ptr[(i * ny + j) * nz + k]`.

#include <cuda_runtime.h>

namespace {

// Match the geometry constants from python/boilingsim/geometry.py.
// (Passed in by the caller too, but a `__device__ __forceinline__` neighbour
// helper reads them as kernel parameters anyway.)

__device__ __forceinline__ int idx3(int i, int j, int k, int ny, int nz) {
    return (i * ny + j) * nz + k;
}

/// Neighbour-pressure lookup under mixed BCs.
///   * fluid neighbour: use its value (interior).
///   * air neighbour (free surface): Dirichlet p = 0.
///   * solid neighbour (pot wall / carrot): Neumann ∂p/∂n = 0 → ghost = p_self.
///
/// Mirror of `_pressure_neighbour` in fluid.py.
__device__ __forceinline__ float pressure_neighbour(
    int m_nbr, float p_nbr, float p_self,
    int mat_fluid, int mat_air
) {
    if (m_nbr == mat_fluid) return p_nbr;
    if (m_nbr == mat_air)   return 0.0f;
    return p_self;
}

} // anonymous namespace

// Tile dimensions for the shared-memory kernel.
//
// Memory layout note: arrays are stored as ``(i, j, k) -> (i*ny + j)*nz + k``,
// so ``k`` is the fastest-varying global dimension. To get coalesced reads
// for consecutive threads in a warp, threadIdx.x must therefore map to
// the ``k`` axis (not ``i``). The TILE_K, TILE_J, TILE_I dim names below
// reflect this -- TILE_K corresponds to block.x.
//
// Phase 5.5 Lever 1: TILE_K = 32 makes each warp = one full row in k =
// exactly one 128-byte cache line for global LDG/STG of div_u/p_new. This
// is the textbook coalesced-stencil pattern, replacing the previous 8x8x8
// shape where each warp's 32 threads spanned four 128-byte lines.
//
// (4, 4, 32) = 512 threads keeps us under the 1024 cap; halo block is
// (4+2)(4+2)(32+2) = 1224 cells * (4+4) = 9.8 KB shared mem per block,
// still fitting 2 concurrent blocks per SM on Ada (100 KB shmem/SM).
constexpr int TILE_K = 32;
constexpr int TILE_J = 4;
constexpr int TILE_I = 4;
constexpr int SH_K = TILE_K + 2;
constexpr int SH_J = TILE_J + 2;
constexpr int SH_I = TILE_I + 2;

__device__ __forceinline__ int sh_idx(int si, int sj, int sk) {
    return (si * SH_J + sj) * SH_K + sk;
}

// Phase 5.5 Lever 2: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor).
// Hints nvcc to keep register usage low enough that 2 blocks can co-reside on
// each SM. At 512 threads/block and 9.8 KB shmem/block, 2 blocks = 1024 threads
// (16 warps) per SM — solid occupancy for memory-latency hiding on Ada.
extern "C" __global__
__launch_bounds__(TILE_I * TILE_J * TILE_K, 2)
void jacobi_pressure_kernel(
    float* __restrict__ p_new,
    const float* __restrict__ p_old,
    const float* __restrict__ div_u,
    const int*   __restrict__ mat,
    int nx, int ny, int nz,
    float dx, float dt, float rho,
    int mat_fluid, int mat_air
) {
    __shared__ float sh_p[SH_I * SH_J * SH_K];
    __shared__ int   sh_m[SH_I * SH_J * SH_K];

    // Tile origin in global coords. blockIdx.x walks the k axis (fastest-
    // varying in memory) so consecutive blocks address consecutive cache
    // lines; blockIdx.z walks the slow i axis.
    const int tile_k0 = blockIdx.x * TILE_K;
    const int tile_j0 = blockIdx.y * TILE_J;
    const int tile_i0 = blockIdx.z * TILE_I;

    // Cooperative load of the (TILE+2)^3 halo-inclusive block into shared mem.
    // 1000 cells, 512 threads → each thread loads 1-2 cells. Out-of-bounds
    // halo positions get (0.0, mat_air) so they match the geometry's outer
    // air ring (Dirichlet p=0 at free surface).
    const int tid = threadIdx.x
                  + threadIdx.y * TILE_K
                  + threadIdx.z * TILE_K * TILE_J;
    const int total = SH_I * SH_J * SH_K;
    const int block_threads = TILE_I * TILE_J * TILE_K;
    for (int idx = tid; idx < total; idx += block_threads) {
        const int sk = idx % SH_K;
        const int sj = (idx / SH_K) % SH_J;
        const int si = idx / (SH_K * SH_J);
        const int gi = tile_i0 + si - 1;
        const int gj = tile_j0 + sj - 1;
        const int gk = tile_k0 + sk - 1;
        if (gi >= 0 && gi < nx && gj >= 0 && gj < ny && gk >= 0 && gk < nz) {
            const int g = idx3(gi, gj, gk, ny, nz);
            sh_p[idx] = p_old[g];
            sh_m[idx] = mat[g];
        } else {
            sh_p[idx] = 0.0f;
            sh_m[idx] = mat_air;
        }
    }
    __syncthreads();

    // Per-thread cell. threadIdx.x -> k for coalesced global reads of
    // p_new / div_u.
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
    const int center_sh = sh_idx(si, sj, sk);

    if (sh_m[center_sh] != mat_fluid) {
        p_new[idx3(i, j, k, ny, nz)] = 0.0f;
        return;
    }

    const float p_self = sh_p[center_sh];

    // 6 neighbour lookups from shared memory. Each is a single LDS.32
    // (3.5 ns) instead of a global LDG.32 (~300 ns uncached / ~20 ns cached).
    const float p_left  = pressure_neighbour(
        sh_m[sh_idx(si - 1, sj, sk)],
        sh_p[sh_idx(si - 1, sj, sk)],
        p_self, mat_fluid, mat_air);
    const float p_right = pressure_neighbour(
        sh_m[sh_idx(si + 1, sj, sk)],
        sh_p[sh_idx(si + 1, sj, sk)],
        p_self, mat_fluid, mat_air);
    const float p_down  = pressure_neighbour(
        sh_m[sh_idx(si, sj - 1, sk)],
        sh_p[sh_idx(si, sj - 1, sk)],
        p_self, mat_fluid, mat_air);
    const float p_up    = pressure_neighbour(
        sh_m[sh_idx(si, sj + 1, sk)],
        sh_p[sh_idx(si, sj + 1, sk)],
        p_self, mat_fluid, mat_air);
    const float p_back  = pressure_neighbour(
        sh_m[sh_idx(si, sj, sk - 1)],
        sh_p[sh_idx(si, sj, sk - 1)],
        p_self, mat_fluid, mat_air);
    const float p_front = pressure_neighbour(
        sh_m[sh_idx(si, sj, sk + 1)],
        sh_p[sh_idx(si, sj, sk + 1)],
        p_self, mat_fluid, mat_air);

    // Sum order EXACTLY matches fluid.py:273 (left + right + down + up + back + front).
    // Any reordering risks float non-associativity and breaks parity.
    const float s = p_left + p_right + p_down + p_up + p_back + p_front;
    const float rhs = rho * dx * dx * div_u[idx3(i, j, k, ny, nz)] / dt;
    p_new[idx3(i, j, k, ny, nz)] = (s - rhs) / 6.0f;
}

extern "C" cudaError_t jacobi_pressure_launch(
    float* p_new,
    const float* p_old,
    const float* div_u,
    const int* mat,
    int nx, int ny, int nz,
    float dx, float dt, float rho,
    int mat_fluid, int mat_air
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return cudaSuccess;
    // block.x maps to the k axis (fastest-varying in memory). grid is laid
    // out (k, j, i) so blockIdx.x walks tiles along k -> coalesced LDG.32 on
    // global reads of div_u and writes of p_new.
    const dim3 block(TILE_K, TILE_J, TILE_I);
    const dim3 grid(
        (nz + TILE_K - 1) / TILE_K,
        (ny + TILE_J - 1) / TILE_J,
        (nx + TILE_I - 1) / TILE_I
    );
    jacobi_pressure_kernel<<<grid, block>>>(
        p_new, p_old, div_u, mat,
        nx, ny, nz, dx, dt, rho, mat_fluid, mat_air
    );
    return cudaGetLastError();
}

// Phase 5 M3: fused pressure-solve loop.
//
// Runs `n_iter` Jacobi sweeps on the same device with ping-pong between
// `p` and `p_tmp`. On entry, `p` is the initial guess (zero for the first
// projection call -- the caller zeros it). On exit, the converged pressure
// lives in `p` regardless of whether `n_iter` is odd or even (the launcher
// handles the final memcpy so the caller's contract matches Warp's path).
//
// Why this is the M3 win: instead of 200 round-trips through
// Python → PyO3 → Rust → cudarc → driver, we make ONE round-trip and let
// the driver queue all 200 launches asynchronously on the default stream.
// Driver-side launch overhead is ~1-2 μs vs the ~10 μs of Python/PyO3
// per call, so we save ~1.5 ms per projection at 200 iterations.
//
// CUDA Graph capture is a separate (potentially additive) optimization
// that can be layered on top of this loop in a later milestone if the
// driver-side launch overhead is still in the way. For the Phase 5 perf
// gate (≥30 % end-to-end reduction), measurements drive whether we need it.
extern "C" cudaError_t pressure_solve_launch(
    float* p,
    float* p_tmp,
    const float* div_u,
    const int* mat,
    int nx, int ny, int nz,
    float dx, float dt, float rho,
    int n_iter,
    int mat_fluid, int mat_air
) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || n_iter <= 0) return cudaSuccess;
    // block.x maps to the k axis (fastest-varying in memory). grid is laid
    // out (k, j, i) so blockIdx.x walks tiles along k -> coalesced LDG.32 on
    // global reads of div_u and writes of p_new.
    const dim3 block(TILE_K, TILE_J, TILE_I);
    const dim3 grid(
        (nz + TILE_K - 1) / TILE_K,
        (ny + TILE_J - 1) / TILE_J,
        (nx + TILE_I - 1) / TILE_I
    );

    // Ping-pong: iter 0 reads p, writes p_tmp; iter 1 reads p_tmp, writes p; ...
    // After n_iter sweeps, the latest write is to p_tmp when n_iter is odd
    // and to p when n_iter is even. We copy p_tmp -> p in the odd case so
    // the caller always reads the final pressure from `p`.
    for (int it = 0; it < n_iter; ++it) {
        float* dst = (it & 1) == 0 ? p_tmp : p;
        const float* src = (it & 1) == 0 ? p : p_tmp;
        jacobi_pressure_kernel<<<grid, block>>>(
            dst, src, div_u, mat,
            nx, ny, nz, dx, dt, rho, mat_fluid, mat_air
        );
    }

    if ((n_iter & 1) == 1) {
        cudaError_t err = cudaMemcpyAsync(
            p, p_tmp, sizeof(float) * (size_t)nx * (size_t)ny * (size_t)nz,
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) return err;
    }
    return cudaGetLastError();
}
