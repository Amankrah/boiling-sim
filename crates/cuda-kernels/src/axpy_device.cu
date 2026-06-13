// Phase 6 K4 + K5: axpy-family kernels with DEVICE-RESIDENT scalar
// coefficients.
//
// The key trick is the coefficient pointer: `alpha_ptr` and `beta_ptr` are
// device pointers into 1-element f32 buffers computed by earlier kernels
// in the CG loop (e.g. `alpha = rzold / pAp` via `compute_alpha_kernel`).
// The CG driver never reads them to host -- they stay GPU-resident across
// the full inner loop, so we save ~30 dtoh syncs per projection.
//
// K4: axpy_device   -- y = alpha * x + y         (with alpha negation flag)
// K5: scaled_axpy   -- p = z + beta * p_old      (CG search-direction update)
// Also: divide_scalars -- result = num / den    (device-side α and β
//                                                computation)

#include <cuda_runtime.h>

// Standard axpy with device-resident scalar. `negate_alpha=true` computes
// `y = -alpha*x + y` (used for the CG residual update `r = r - alpha*Ap`).
extern "C" __global__
void axpy_device_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ alpha_ptr,
    int n,
    int negate_alpha
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float alpha = negate_alpha != 0 ? -(*alpha_ptr) : (*alpha_ptr);
    y[idx] = alpha * x[idx] + y[idx];
}

extern "C" cudaError_t axpy_device_launch(
    float* y,
    const float* x,
    const float* alpha_ptr,
    int n,
    int negate_alpha
) {
    if (n <= 0) return cudaSuccess;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    axpy_device_kernel<<<grid, block>>>(y, x, alpha_ptr, n, negate_alpha);
    return cudaGetLastError();
}

// CG search-direction update: p_out = z + beta * p_old. Allows in-place
// when p_out == p_old (the common case in CG); the read of p_old happens
// before the write to p_out for the same index, so a single thread reading
// and writing the same address is safe.
extern "C" __global__
void scaled_axpy_kernel(
    float* __restrict__ p_out,
    const float* __restrict__ z,
    const float* __restrict__ p_old,
    const float* __restrict__ beta_ptr,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float beta = *beta_ptr;
    p_out[idx] = z[idx] + beta * p_old[idx];
}

extern "C" cudaError_t scaled_axpy_launch(
    float* p_out,
    const float* z,
    const float* p_old,
    const float* beta_ptr,
    int n
) {
    if (n <= 0) return cudaSuccess;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    scaled_axpy_kernel<<<grid, block>>>(p_out, z, p_old, beta_ptr, n);
    return cudaGetLastError();
}

// Single-thread kernel to compute `result = num / den` (or zero if den is
// zero) from two device-resident scalars. Used for `alpha = rzold / pAp`
// and `beta = rznew / rzold` in the CG loop.
extern "C" __global__
void divide_scalars_kernel(
    const float* num_ptr,
    const float* den_ptr,
    float* result_ptr
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float den = *den_ptr;
    *result_ptr = (den == 0.0f) ? 0.0f : (*num_ptr / den);
}

extern "C" cudaError_t divide_scalars_launch(
    const float* num_ptr,
    const float* den_ptr,
    float* result_ptr
) {
    divide_scalars_kernel<<<1, 1>>>(num_ptr, den_ptr, result_ptr);
    return cudaGetLastError();
}

// Compute `b = (rho * dx*dx / dt) * div_u` -- the CG right-hand side.
// Non-fluid cells get b = 0 (same convention as the SpMV and Jacobi).
extern "C" __global__
void compute_b_kernel(
    float* __restrict__ b,
    const float* __restrict__ div_u,
    const int* __restrict__ mat,
    int nx, int ny, int nz,
    float scale,                   // = rho * dx * dx / dt
    int mat_fluid
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    const int idx = (i * ny + j) * nz + k;
    if (mat[idx] != mat_fluid) {
        b[idx] = 0.0f;
    } else {
        b[idx] = scale * div_u[idx];
    }
}

extern "C" cudaError_t compute_b_launch(
    float* b,
    const float* div_u,
    const int* mat,
    int nx, int ny, int nz,
    float scale,
    int mat_fluid
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return cudaSuccess;
    const dim3 block(8, 8, 8);
    const dim3 grid((nz + 7) / 8, (ny + 7) / 8, (nx + 7) / 8);
    compute_b_kernel<<<grid, block>>>(b, div_u, mat, nx, ny, nz, scale, mat_fluid);
    return cudaGetLastError();
}

// Zero a device buffer of length n. Wraps cudaMemsetAsync so the CG driver
// can use a uniform launcher API for "set this buffer to zero".
extern "C" cudaError_t zero_buffer_launch(float* buf, int n) {
    if (n <= 0) return cudaSuccess;
    return cudaMemsetAsync(buf, 0, sizeof(float) * (size_t)n);
}
