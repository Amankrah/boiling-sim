// Phase 6 driver: fused Preconditioned Conjugate Gradient pressure solve.
//
// Replaces the 200-iter Jacobi loop with a PCG loop that runs the entire
// inner iteration sequence (SpMV, two reductions, two axpys, preconditioner)
// inside one Rust→C++ call. Algorithm:
//
//   b = (rho*dx^2/dt) * div_u            (only fluid cells)
//   bsq = b.dot(b)                       (device-resident; one host copy at start)
//   p_out = 0                            (cold-start initial guess)
//   r = b - A*p_out = b
//   z = M^-1 * r
//   p_search = z
//   rzold = r.dot(z)
//   for iter in 0..max_iter:
//     Ap = A * p_search
//     pAp = p_search.dot(Ap)
//     alpha = rzold / pAp
//     p_out += alpha * p_search
//     r     -= alpha * Ap
//     if iter >= 8 and iter % 5 == 0:
//       rsq = r.dot(r)
//       if rsq < tol^2 * bsq: break
//     z = M^-1 * r
//     rznew = r.dot(z)
//     beta = rznew / rzold
//     p_search = z + beta * p_search
//     rzold = rznew
//
// SIGN AND SCALING (verified in plan & laplacian_spmv.cu): A = 6*I - S, where
// S is the neighbour-sum operator with the same _pressure_neighbour BC rules
// as fluid.py:222-239. b = rho*dx^2*div_u/dt. No 1/dx^2 anywhere -- the
// system is algebraically identical to Jacobi's `Ax = rhs` arithmetic.
//
// Convergence check cadence (N=5 from iter >= 8): early iterations have
// artificially low residual due to zero initial guess, so we skip until
// iter 8 to avoid premature exit. Every 5 iter after that does one
// dtoh_sync_copy of a 1-element buffer (~10 us) so total sync cost on a
// 30-iter solve is ~50 us, <2% of typical projection budget.
//
// Stream policy: all kernels run on the default stream (cudarc 0.12 has no
// public async API). The convergence-check dtoh therefore drains the default
// stream. If Phase 7 introduces cross-stream concurrency for the energy
// equation, the convergence check would break that overlap; document there.

#include <cuda_runtime.h>

// Forward declarations of the helper launchers from sibling .cu files.
extern "C" cudaError_t laplacian_spmv_launch(
    float* Ap, const float* p, const int* mat,
    int nx, int ny, int nz, int mat_fluid, int mat_air
);
extern "C" cudaError_t diag_inverse_apply_launch(
    float* z, const float* r, const int* mat,
    int nx, int ny, int nz, int mat_fluid, int mat_air
);
extern "C" cudaError_t dot_launch(
    const float* x, const float* y, int n,
    float* workspace, float* result
);
extern "C" cudaError_t axpy_device_launch(
    float* y, const float* x, const float* alpha_ptr,
    int n, int negate_alpha
);
extern "C" cudaError_t scaled_axpy_launch(
    float* p_out, const float* z, const float* p_old,
    const float* beta_ptr, int n
);
extern "C" cudaError_t divide_scalars_launch(
    const float* num_ptr, const float* den_ptr, float* result_ptr
);
extern "C" cudaError_t compute_b_launch(
    float* b, const float* div_u, const int* mat,
    int nx, int ny, int nz, float scale, int mat_fluid
);
extern "C" cudaError_t zero_buffer_launch(float* buf, int n);

// Convergence-check cadence and starting iteration. Per plan & Plan-agent
// recommendation: check every 5 iter starting from iter 8.
constexpr int CG_CHECK_EVERY = 5;
constexpr int CG_FIRST_CHECK = 8;

// PCG driver. All workspace buffers are caller-allocated on the device and
// must be at least `n = nx*ny*nz` floats each (except `dot_workspace` which
// must be at least DOT_MAX_PARTIALS=1024 floats from dot_reduce.cu, and the
// 6 scalar buffers which are 1 float each).
//
// Returns the actual iteration count via `iter_count_host` so callers can
// log it. The final pressure lands in `p_out`.
extern "C" cudaError_t pressure_solve_pcg_launch(
    // Pressure field (input: initial guess, output: solved pressure)
    float* p_out,
    // Inputs
    const float* div_u,
    const int*   mat,
    int nx, int ny, int nz,
    float dx, float dt, float rho,
    int mat_fluid, int mat_air,
    // Convergence parameters
    float pressure_tol,         // relative residual tolerance ||r||/||b|| < tol
    int   max_iter,
    // Workspace device buffers, each of length n = nx*ny*nz
    float* ws_b,
    float* ws_r,
    float* ws_z,
    float* ws_p_search,
    float* ws_Ap,
    // Dot-product reduction workspace (>= 1024 floats)
    float* dot_workspace,
    // Six device-resident scalars (1 float each)
    float* dev_alpha,
    float* dev_beta,
    float* dev_rzold,
    float* dev_rznew,
    float* dev_bsq,
    float* dev_rsq,
    float* dev_pAp,
    // Output: actual iteration count (host side, for logging)
    int* iter_count_host
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (iter_count_host) *iter_count_host = 0;
        return cudaSuccess;
    }
    const int n = nx * ny * nz;
    cudaError_t err;

    // Step 1: b = -(rho * dx*dx / dt) * div_u, zero on non-fluid cells.
    //
    // Sign derivation (re-verified after Gate 1 failure):
    //   Jacobi update: p_new = (s - rhs) / 6 where rhs = rho*dx^2*div_u/dt.
    //   At convergence: 6*p = s - rhs  =>  6*p - s = -rhs.
    //   The linear system is therefore A*p = b with:
    //     A = 6*I - S   (matches laplacian_spmv: Ap = 6p - s)
    //     b = -rhs = -(rho * dx*dx / dt) * div_u
    //   The NEGATIVE sign on b was missing from the original plan; without
    //   it CG converges to -p_jacobi (sign-flipped solution).
    const float b_scale = -rho * dx * dx / dt;
    err = compute_b_launch(ws_b, div_u, mat, nx, ny, nz, b_scale, mat_fluid);
    if (err != cudaSuccess) return err;

    // Step 2: bsq = b.dot(b). Read to host so we can compare rsq against it
    // without an extra dtoh on every convergence check.
    err = dot_launch(ws_b, ws_b, n, dot_workspace, dev_bsq);
    if (err != cudaSuccess) return err;
    float bsq_host = 0.0f;
    err = cudaMemcpy(&bsq_host, dev_bsq, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;

    // Edge case: bsq == 0 means div_u is zero (already incompressible).
    // Set p_out to zero and exit immediately.
    if (bsq_host == 0.0f) {
        err = zero_buffer_launch(p_out, n);
        if (iter_count_host) *iter_count_host = 0;
        return err;
    }
    const float tol_sq_bsq = pressure_tol * pressure_tol * bsq_host;

    // Step 3: p_out = 0 (cold start).
    err = zero_buffer_launch(p_out, n);
    if (err != cudaSuccess) return err;

    // Step 4: r = b (since A * 0 = 0).
    err = cudaMemcpyAsync(ws_r, ws_b, sizeof(float) * (size_t)n,
                          cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;

    // Step 5: z = M^-1 * r.
    err = diag_inverse_apply_launch(ws_z, ws_r, mat, nx, ny, nz, mat_fluid, mat_air);
    if (err != cudaSuccess) return err;

    // Step 6: p_search = z.
    err = cudaMemcpyAsync(ws_p_search, ws_z, sizeof(float) * (size_t)n,
                          cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;

    // Step 7: rzold = r . z.
    err = dot_launch(ws_r, ws_z, n, dot_workspace, dev_rzold);
    if (err != cudaSuccess) return err;

    // ---- Main CG iteration loop ----
    int iter = 0;
    for (iter = 0; iter < max_iter; ++iter) {
        // Ap = A * p_search.
        err = laplacian_spmv_launch(ws_Ap, ws_p_search, mat,
                                    nx, ny, nz, mat_fluid, mat_air);
        if (err != cudaSuccess) return err;

        // pAp = p_search . Ap.
        err = dot_launch(ws_p_search, ws_Ap, n, dot_workspace, dev_pAp);
        if (err != cudaSuccess) return err;

        // alpha = rzold / pAp (device-side divide).
        err = divide_scalars_launch(dev_rzold, dev_pAp, dev_alpha);
        if (err != cudaSuccess) return err;

        // p_out += alpha * p_search.
        err = axpy_device_launch(p_out, ws_p_search, dev_alpha, n, /*negate=*/0);
        if (err != cudaSuccess) return err;

        // r -= alpha * Ap.
        err = axpy_device_launch(ws_r, ws_Ap, dev_alpha, n, /*negate=*/1);
        if (err != cudaSuccess) return err;

        // Convergence check (cadence: every CG_CHECK_EVERY from iter >= CG_FIRST_CHECK).
        if (iter >= CG_FIRST_CHECK && (iter % CG_CHECK_EVERY) == 0) {
            err = dot_launch(ws_r, ws_r, n, dot_workspace, dev_rsq);
            if (err != cudaSuccess) return err;
            float rsq_host = 0.0f;
            err = cudaMemcpy(&rsq_host, dev_rsq, sizeof(float),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return err;
            if (rsq_host < tol_sq_bsq) {
                ++iter;  // count the iteration we just finished
                break;
            }
        }

        // z = M^-1 * r.
        err = diag_inverse_apply_launch(ws_z, ws_r, mat, nx, ny, nz,
                                        mat_fluid, mat_air);
        if (err != cudaSuccess) return err;

        // rznew = r . z.
        err = dot_launch(ws_r, ws_z, n, dot_workspace, dev_rznew);
        if (err != cudaSuccess) return err;

        // beta = rznew / rzold.
        err = divide_scalars_launch(dev_rznew, dev_rzold, dev_beta);
        if (err != cudaSuccess) return err;

        // p_search = z + beta * p_search (in-place).
        err = scaled_axpy_launch(ws_p_search, ws_z, ws_p_search, dev_beta, n);
        if (err != cudaSuccess) return err;

        // rzold = rznew (device-side scalar copy).
        err = cudaMemcpyAsync(dev_rzold, dev_rznew, sizeof(float),
                              cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return err;
    }

    if (iter_count_host) *iter_count_host = iter;
    return cudaGetLastError();
}
