// Phase 6 K2: diagonal-Jacobi preconditioner application.
//
// Computes `z[i] = r[i] / diag(A)[i]` for each fluid cell.
//
// `diag(A)[i]` for the discrete Laplacian with mixed BCs (see plan section
// "Three bugs in the original draft, fixed before writing code"):
//   - Start at 6 (one per stencil neighbour direction).
//   - For each NEUMANN (solid) neighbour, subtract 1 because the ghost
//     `p_self` ends up on the diagonal: `6p - sum_with_p_self_for_solid =
//     (6 - num_solid_nbrs)*p_self - sum_non_solid`.
//   - Dirichlet (air) neighbours do NOT reduce the diagonal -- they
//     contribute 0 to the off-diagonal but the row still has 6 outgoing
//     stencil arms.
//
// For non-fluid cells: z is set to 0 (Ap stays at 0 for them too in the
// SpMV; r will be 0 for them in the CG residual, so 0/anything is fine).
//
// One-pass per-cell kernel, no shared memory needed. Cost dominated by the
// per-cell load of 6 mat values.

#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int dia_idx3(int i, int j, int k, int ny, int nz) {
    return (i * ny + j) * nz + k;
}

// Material id at (i, j, k) with out-of-bounds treated as MAT_AIR (matches
// the SpMV BC convention). Reused across the diagonal computation.
__device__ __forceinline__ int dia_mat_or_air(
    const int* mat, int i, int j, int k,
    int nx, int ny, int nz, int mat_air
) {
    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
        return mat_air;
    }
    return mat[dia_idx3(i, j, k, ny, nz)];
}

}  // anonymous namespace

extern "C" __global__
void diag_inverse_apply_kernel(
    float* __restrict__ z,
    const float* __restrict__ r,
    const int*   __restrict__ mat,
    int nx, int ny, int nz,
    int mat_fluid, int mat_air
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const int center = dia_idx3(i, j, k, ny, nz);
    if (mat[center] != mat_fluid) {
        z[center] = 0.0f;
        return;
    }

    // Count solid neighbours (not fluid, not air -> solid).
    int n_solid = 0;
    const int m_l = dia_mat_or_air(mat, i - 1, j, k, nx, ny, nz, mat_air);
    const int m_r = dia_mat_or_air(mat, i + 1, j, k, nx, ny, nz, mat_air);
    const int m_d = dia_mat_or_air(mat, i, j - 1, k, nx, ny, nz, mat_air);
    const int m_u = dia_mat_or_air(mat, i, j + 1, k, nx, ny, nz, mat_air);
    const int m_b = dia_mat_or_air(mat, i, j, k - 1, nx, ny, nz, mat_air);
    const int m_f = dia_mat_or_air(mat, i, j, k + 1, nx, ny, nz, mat_air);

    if (m_l != mat_fluid && m_l != mat_air) ++n_solid;
    if (m_r != mat_fluid && m_r != mat_air) ++n_solid;
    if (m_d != mat_fluid && m_d != mat_air) ++n_solid;
    if (m_u != mat_fluid && m_u != mat_air) ++n_solid;
    if (m_b != mat_fluid && m_b != mat_air) ++n_solid;
    if (m_f != mat_fluid && m_f != mat_air) ++n_solid;

    const float diag = 6.0f - (float)n_solid;
    // diag is always >= 1 in practice (a cell surrounded by solids in 5+
    // directions is not a fluid cell). Guard against the pathological case
    // of a 1-cell fluid pocket surrounded entirely by solids (diag=0) by
    // falling back to z = r in that case -- mathematically the row is then
    // singular and the cell is decoupled from the rest of the system.
    if (diag <= 0.0f) {
        z[center] = r[center];
    } else {
        z[center] = r[center] / diag;
    }
}

extern "C" cudaError_t diag_inverse_apply_launch(
    float* z,
    const float* r,
    const int* mat,
    int nx, int ny, int nz,
    int mat_fluid, int mat_air
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return cudaSuccess;
    // 8x8x8 = 512 threads, simple shape (the kernel is memory-bandwidth
    // bound on the mat reads, not compute bound).
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (nz + 7) / 8,
        (ny + 7) / 8,
        (nx + 7) / 8
    );
    diag_inverse_apply_kernel<<<grid, block>>>(
        z, r, mat, nx, ny, nz, mat_fluid, mat_air
    );
    return cudaGetLastError();
}
