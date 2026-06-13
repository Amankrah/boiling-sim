// Phase 6 K3: deterministic device-resident dot product reduction.
//
// Two-kernel pattern:
//   Kernel A (`dot_partial_kernel`): each block reduces its assigned cells
//     into one block-local sum via warp-shuffle + shared memory, writes
//     to `workspace[blockIdx.x]`.
//   Kernel B (`dot_final_kernel`): a SINGLE block, SINGLE thread does a
//     sequential sum over the per-block partials in `workspace` and writes
//     the result to `result[0]`. Deterministic by construction -- no
//     atomicAdd, no inter-block ordering.
//
// Per the Plan agent stress-test: `atomicAdd` ordering varies thermally on
// Ada (block scheduling changes with throttling state). The sequential
// final reduce eliminates this flake. Cost: 256 sequential adds takes
// ~256 * 4 cycles ~= 1024 cycles ~= 500 ns on Ada. Negligible.
//
// Used for `r.dot(r)`, `r.dot(z)`, `p.dot(Ap)`, `b.dot(b)` in the CG loop.

#include <cuda_runtime.h>

namespace {

// Block size MUST be a power of two for the shared-memory tree reduction
// below; 256 is a good fit (8 warps, fits all blocks across all SMs).
constexpr int DOT_BLOCK = 256;

// Hard cap on the number of partial-sum slots. At 256 cells per block and
// 256 partial slots, the kernel handles up to 256*256 = 65536 cells per
// launch; for larger grids, the per-block sum loop covers multiple
// "block-sized chunks" so we never run out of workspace slots. See the
// strided loop in `dot_partial_kernel` below.
constexpr int DOT_MAX_PARTIALS = 1024;

}  // anonymous namespace

// Compute per-block partial dot products. Output: workspace[blockIdx.x].
extern "C" __global__
void dot_partial_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    int n,
    float* __restrict__ workspace
) {
    __shared__ float sdata[DOT_BLOCK];
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * blockDim.x;
    const int stride = gridDim.x * blockDim.x;

    // Each thread accumulates a private partial across a strided sweep of
    // the array. Stride is gridDim*blockDim so per-thread chunks don't
    // overlap and we cover every element exactly once.
    float thread_sum = 0.0f;
    for (int idx = block_offset + tid; idx < n; idx += stride) {
        thread_sum += x[idx] * y[idx];
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Standard shared-memory tree reduction down to sdata[0].
    for (int s = DOT_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        workspace[blockIdx.x] = sdata[0];
    }
}

// Single block, single thread sequential reduction over per-block partials.
// `n_partials` is the number of valid entries in `workspace` (i.e. the
// grid dim used for the partial kernel).
extern "C" __global__
void dot_final_kernel(
    const float* __restrict__ workspace,
    int n_partials,
    float* __restrict__ result
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float sum = 0.0f;
    for (int i = 0; i < n_partials; ++i) {
        sum += workspace[i];
    }
    result[0] = sum;
}

// Compute Sum_i x[i] * y[i] into result[0] (a device-resident 1-element
// float buffer). `workspace` must point to at least DOT_MAX_PARTIALS
// floats of scratch device memory; the caller owns this allocation.
extern "C" cudaError_t dot_launch(
    const float* x,
    const float* y,
    int n,
    float* workspace,
    float* result
) {
    if (n <= 0) {
        // Zero-init the result without launching kernels.
        return cudaMemsetAsync(result, 0, sizeof(float));
    }
    // Cap grid dim so we don't exceed the workspace capacity. The strided
    // loop inside dot_partial_kernel covers the entire array even when
    // gridDim*blockDim < n.
    int n_blocks = (n + DOT_BLOCK - 1) / DOT_BLOCK;
    if (n_blocks > DOT_MAX_PARTIALS) n_blocks = DOT_MAX_PARTIALS;

    dot_partial_kernel<<<n_blocks, DOT_BLOCK>>>(x, y, n, workspace);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    dot_final_kernel<<<1, 1>>>(workspace, n_blocks, result);
    return cudaGetLastError();
}
