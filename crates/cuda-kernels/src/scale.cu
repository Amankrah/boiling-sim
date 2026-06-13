// Phase 5 M1: pointer-sharing PoC kernel.
//
// In-place scalar multiply of a device float buffer. The whole point is to
// validate that a raw device pointer obtained from Warp on the Python side
// (via ``wp.array.__cuda_array_interface__["data"][0]``) round-trips into a
// Rust-launched kernel without an explicit copy and writes back through the
// same buffer the Python side reads.

#include <cuda_runtime.h>

extern "C" __global__
void scale_inplace_kernel(float* a, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= scale;
    }
}

// Host wrapper called from Rust. Synchronous on the default stream so the
// PoC has trivial completion semantics; M2 / M3 introduce explicit streams
// + CUDA Graphs once the FFI shape is proven.
extern "C" cudaError_t scale_inplace_launch(float* d_a, int n, float scale) {
    if (n <= 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    scale_inplace_kernel<<<blocks, threads>>>(d_a, scale, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    return cudaDeviceSynchronize();
}
