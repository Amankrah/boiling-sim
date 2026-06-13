//! CUDA kernel bindings for the boiling simulation.
//!
//! This crate compiles hand-written `.cu` kernels via nvcc (see [`build.rs`])
//! and exposes safe-ish Rust wrappers around them. The launcher functions
//! live in the .cu files as `extern "C"` host functions; Rust just calls
//! them with the right device pointers + sizes.
//!
//! Phase 5 M1 (this milestone): `scale_inplace` for the Warp ↔ cudarc
//! pointer-sharing PoC. M2 adds the tiled Jacobi pressure kernel. M3 adds
//! the CUDA Graph launcher.

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

extern "C" {
    #[allow(dead_code)]
    fn vector_add(a: *const f32, b: *const f32, c: *mut f32, n: i32);

    /// Phase 5 M1 launcher. Multiplies `n` floats at device pointer `d_a`
    /// by `scale` in place. Synchronous on the default stream. Returns a
    /// `cudaError_t` (0 == success).
    fn scale_inplace_launch(d_a: *mut f32, n: i32, scale: f32) -> i32;

    /// Phase 5 M2 launcher. One Jacobi sweep of ∇²p = (ρ/dt)·∇·u on the
    /// MAC grid. Mirror of [`jacobi_pressure_step`](python/boilingsim/fluid.py).
    /// Operates on the externally-allocated buffers `p_new` (destination),
    /// `p_old` (source), `div_u` (RHS), `mat` (material IDs). Asynchronous
    /// on the default stream -- M3 wraps a sequence of these in a CUDA
    /// Graph; this launcher doesn't sync so the next iteration's launch
    /// queues immediately. Returns the post-launch cudaGetLastError().
    fn jacobi_pressure_launch(
        p_new: *mut f32,
        p_old: *const f32,
        div_u: *const f32,
        mat: *const i32,
        nx: i32,
        ny: i32,
        nz: i32,
        dx: f32,
        dt: f32,
        rho: f32,
        mat_fluid: i32,
        mat_air: i32,
    ) -> i32;

    /// Phase 5 M3 launcher. Runs the FULL 200-Jacobi-iteration loop with
    /// ping-pong inside CUDA C++. Eliminates the Python→PyO3→Rust round-trip
    /// per iteration. Final pressure ends up in `p` regardless of n_iter
    /// parity (an in-Rust memcpy handles the odd-iter case). Asynchronous;
    /// caller syncs at the end.
    fn pressure_solve_launch(
        p: *mut f32,
        p_tmp: *mut f32,
        div_u: *const f32,
        mat: *const i32,
        nx: i32,
        ny: i32,
        nz: i32,
        dx: f32,
        dt: f32,
        rho: f32,
        n_iter: i32,
        mat_fluid: i32,
        mat_air: i32,
    ) -> i32;

    /// Phase 5.6 M5.6.A launcher: trilinear scatter of latent-heat sink
    /// from each growing bubble into the 8 surrounding water cells. The
    /// `bubbles` pointer must point to an array of 56-byte Bubble structs
    /// (layout defined in `cuda-kernels/include/bubble.h`, matched to
    /// Warp's `wp.struct Bubble`). Output T field is mutated via atomicAdd
    /// with negative deposits (mirrors Warp's atomic_sub).
    fn scatter_latent_heat_launch(
        bubbles: *const u8,  // raw byte pointer; sized at 56 * n_bubbles
        n_bubbles: i32,
        T: *mut f32,
        mat: *const i32,
        nx: i32,
        ny: i32,
        nz: i32,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dt: f32,
        current_time: f32,
        rho_l: f32,
        cp_l: f32,
        rho_v: f32,
        h_lv: f32,
        T_sat_k: f32,
        mat_fluid: i32,
    ) -> i32;

    /// Phase 5.6 M5.6.B launcher: trilinear scatter of bubble buoyancy
    /// upward force onto the 8 surrounding z-faces. uz shape differs from
    /// mat shape (uz has nz+1 elements in the k dim because MAC z-faces
    /// live between cells); pass both shapes explicitly.
    fn scatter_momentum_launch(
        bubbles: *const u8,
        n_bubbles: i32,
        uz: *mut f32,
        mat: *const i32,
        nx: i32, ny: i32, nz: i32,
        uz_nx: i32, uz_ny: i32, uz_nz: i32,
        ox: f32, oy: f32, oz: f32,
        dx: f32, dt: f32,
        rho_l: f32, rho_v: f32, g_mag: f32,
        mat_fluid: i32,
    ) -> i32;

    /// Phase 5.6 M5.6.C launcher: scatter bubble VOF occupancy reduction
    /// onto `water_alpha`, then clamp to [0, 1] in a follow-on kernel.
    fn reduce_water_alpha_launch(
        bubbles: *const u8,
        n_bubbles: i32,
        water_alpha: *mut f32,
        mat: *const i32,
        nx: i32, ny: i32, nz: i32,
        ox: f32, oy: f32, oz: f32,
        dx: f32,
        mat_fluid: i32,
    ) -> i32;

    /// Phase 6 launcher: full PCG pressure-solve loop in CUDA C++. Inputs
    /// are `div_u` and `mat`; output is `p_out`. The driver allocates no
    /// device memory of its own -- the caller passes in pre-allocated
    /// workspace buffers (5x `n` floats, 1024-float dot workspace, 7
    /// 1-float device scalars). See pressure_solve_pcg.cu for the layout.
    fn pressure_solve_pcg_launch(
        p_out: *mut f32,
        div_u: *const f32,
        mat: *const i32,
        nx: i32, ny: i32, nz: i32,
        dx: f32, dt: f32, rho: f32,
        mat_fluid: i32, mat_air: i32,
        pressure_tol: f32, max_iter: i32,
        ws_b: *mut f32,
        ws_r: *mut f32,
        ws_z: *mut f32,
        ws_p_search: *mut f32,
        ws_ap: *mut f32,
        dot_workspace: *mut f32,
        dev_alpha: *mut f32,
        dev_beta: *mut f32,
        dev_rzold: *mut f32,
        dev_rznew: *mut f32,
        dev_bsq: *mut f32,
        dev_rsq: *mut f32,
        dev_pap: *mut f32,
        iter_count_host: *mut i32,
    ) -> i32;

    /// Phase 5.7 launcher: per-bubble update (Mikic-Rohsenow growth or
    /// Plesset-Zwick condensation w/ embedded T scatter, Fritz departure,
    /// terminal-slip advection, vent/solid deactivation). Mirror of
    /// `update_bubbles` at python/boilingsim/boiling.py:771-965.
    ///
    /// Mutates: bubbles[], slot_claim[], site_active[][][] (via atomicCAS),
    /// needs_fragment[], T[][][] (via atomicAdd on condensation scatter).
    fn update_bubbles_launch(
        bubbles: *mut u8,
        n_bubbles: i32,
        slot_claim: *mut i32,
        site_active: *mut i32,
        needs_fragment: *mut i32,
        T: *const f32,            // pointer is const but kernel does atomicAdd into it
        mat: *const i32,
        nx: i32, ny: i32, nz: i32,
        ux: *const f32, ux_nx: i32, ux_ny: i32, ux_nz: i32,
        uy: *const f32, uy_nx: i32, uy_ny: i32, uy_nz: i32,
        uz: *const f32, uz_nx: i32, uz_ny: i32, uz_nz: i32,
        ox: f32, oy: f32, oz: f32,
        dx: f32, dt: f32,
        current_time: f32,
        water_line_z: f32,
        t_sat_k: f32,
        rho_l: f32, rho_v: f32,
        cp_l: f32, k_l: f32, h_lv: f32,
        sigma: f32,
        theta_rad: f32, g_mag: f32,
        r_seed: f32, r_frag: f32, r_max: f32,
        mat_fluid: i32,
    ) -> i32;
}

/// In-place scale a device buffer of `n` floats by `scale`.
///
/// # Safety
///
/// `device_ptr` must point to at least `n * size_of::<f32>()` bytes of valid
/// device memory in the current primary context. cudarc's [`CudaDevice::new`]
/// retains the primary context, which is the same context Warp's runtime API
/// allocates against — so a pointer obtained from `wp.array.__cuda_array_interface__`
/// is safe to pass here as long as the same device id is used on both sides.
///
/// The `_device` argument is kept alive (not just borrowed) so the primary
/// context stays retained across the kernel launch. It is otherwise unused;
/// the launcher in `scale.cu` writes through `device_ptr` directly.
pub unsafe fn scale_inplace_raw(
    _device: &Arc<CudaDevice>,
    device_ptr: u64,
    n: usize,
    scale: f32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let n_i32: i32 = n
        .try_into()
        .map_err(|_| anyhow!("scale_inplace_raw: n={n} overflows i32"))?;
    let rc = scale_inplace_launch(device_ptr as *mut f32, n_i32, scale);
    if rc != 0 {
        return Err(anyhow!("scale_inplace_launch returned cudaError_t {rc}"));
    }
    Ok(())
}

/// Phase 5 M2 entry point. One Jacobi sweep of the pressure Poisson
/// equation on a 3D MAC grid.
///
/// # Safety
///
/// All four device pointers must point to at least `nx * ny * nz` elements
/// of valid device memory in the current primary context. `p_new` and `p_old`
/// must not overlap (the kernel is non-in-place to avoid races). See
/// [`scale_inplace_raw`] for the broader pointer-lifetime contract.
///
/// Synchronous behavior: this is currently a blocking call -- the launcher
/// does a `cudaGetLastError` but no explicit synchronize. The caller is
/// expected to chain many sweeps; M3 introduces CUDA Graph capture to fuse
/// 200 launches into one driver-side submission.
#[allow(clippy::too_many_arguments)]
pub unsafe fn jacobi_pressure_step_raw(
    _device: &Arc<CudaDevice>,
    p_new_ptr: u64,
    p_old_ptr: u64,
    div_u_ptr: u64,
    mat_ptr: u64,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f32,
    dt: f32,
    rho: f32,
    mat_fluid: i32,
    mat_air: i32,
) -> Result<()> {
    let to_i32 = |v: usize| -> Result<i32> {
        v.try_into().map_err(|_| anyhow!("dim {v} overflows i32"))
    };
    let rc = jacobi_pressure_launch(
        p_new_ptr as *mut f32,
        p_old_ptr as *const f32,
        div_u_ptr as *const f32,
        mat_ptr as *const i32,
        to_i32(nx)?,
        to_i32(ny)?,
        to_i32(nz)?,
        dx,
        dt,
        rho,
        mat_fluid,
        mat_air,
    );
    if rc != 0 {
        return Err(anyhow!("jacobi_pressure_launch returned cudaError_t {rc}"));
    }
    Ok(())
}

/// Phase 5 M3 entry point. Runs the entire 200-iteration Jacobi pressure
/// solve in a single fused Rust→C++ call.
///
/// # Safety
///
/// `p_ptr` and `p_tmp_ptr` must point to disjoint `nx*ny*nz` f32 buffers.
/// `p_ptr` carries the initial guess on entry and the final pressure on
/// exit. `div_u_ptr` and `mat_ptr` are read-only. All four buffers live
/// in the primary context for `device.ordinal()`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn pressure_solve_raw(
    _device: &Arc<CudaDevice>,
    p_ptr: u64,
    p_tmp_ptr: u64,
    div_u_ptr: u64,
    mat_ptr: u64,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f32,
    dt: f32,
    rho: f32,
    n_iter: usize,
    mat_fluid: i32,
    mat_air: i32,
) -> Result<()> {
    let to_i32 = |v: usize| -> Result<i32> {
        v.try_into().map_err(|_| anyhow!("dim {v} overflows i32"))
    };
    let rc = pressure_solve_launch(
        p_ptr as *mut f32,
        p_tmp_ptr as *mut f32,
        div_u_ptr as *const f32,
        mat_ptr as *const i32,
        to_i32(nx)?,
        to_i32(ny)?,
        to_i32(nz)?,
        dx,
        dt,
        rho,
        to_i32(n_iter)?,
        mat_fluid,
        mat_air,
    );
    if rc != 0 {
        return Err(anyhow!("pressure_solve_launch returned cudaError_t {rc}"));
    }
    Ok(())
}

/// Phase 5.6 M5.6.A entry point. Scatter latent-heat sink across the
/// fluid grid from a pool of active bubbles.
///
/// # Safety
///
/// `bubbles_ptr` must point to at least `n_bubbles * 56` bytes of valid
/// device memory laid out as an array of `Bubble` structs (see
/// `cuda-kernels/include/bubble.h`). `T_ptr` and `mat_ptr` must each
/// point to `nx*ny*nz` elements. All pointers must live in the primary
/// context of `device.ordinal()`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_latent_heat_raw(
    _device: &Arc<CudaDevice>,
    bubbles_ptr: u64,
    n_bubbles: usize,
    t_ptr: u64,
    mat_ptr: u64,
    nx: usize,
    ny: usize,
    nz: usize,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dt: f32,
    current_time: f32,
    rho_l: f32,
    cp_l: f32,
    rho_v: f32,
    h_lv: f32,
    t_sat_k: f32,
    mat_fluid: i32,
) -> Result<()> {
    let to_i32 = |v: usize| -> Result<i32> {
        v.try_into().map_err(|_| anyhow!("dim {v} overflows i32"))
    };
    let rc = scatter_latent_heat_launch(
        bubbles_ptr as *const u8,
        to_i32(n_bubbles)?,
        t_ptr as *mut f32,
        mat_ptr as *const i32,
        to_i32(nx)?,
        to_i32(ny)?,
        to_i32(nz)?,
        ox,
        oy,
        oz,
        dx,
        dt,
        current_time,
        rho_l,
        cp_l,
        rho_v,
        h_lv,
        t_sat_k,
        mat_fluid,
    );
    if rc != 0 {
        return Err(anyhow!("scatter_latent_heat_launch returned cudaError_t {rc}"));
    }
    Ok(())
}

/// Phase 5.6 M5.6.B entry point. Scatter bubble buoyancy momentum across
/// the fluid z-faces.
///
/// # Safety
///
/// Same contract as [`scatter_latent_heat_raw`]: `bubbles_ptr` is a 56-byte
/// stride struct array of size `n_bubbles`; `uz_ptr` has `uz_nx*uz_ny*uz_nz`
/// f32 elements; `mat_ptr` has `nx*ny*nz` i32 elements. uz_nz is typically
/// nz+1 for MAC z-faces.
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_momentum_raw(
    _device: &Arc<CudaDevice>,
    bubbles_ptr: u64,
    n_bubbles: usize,
    uz_ptr: u64,
    mat_ptr: u64,
    nx: usize, ny: usize, nz: usize,
    uz_nx: usize, uz_ny: usize, uz_nz: usize,
    ox: f32, oy: f32, oz: f32,
    dx: f32, dt: f32,
    rho_l: f32, rho_v: f32, g_mag: f32,
    mat_fluid: i32,
) -> Result<()> {
    let to_i32 = |v: usize| -> Result<i32> {
        v.try_into().map_err(|_| anyhow!("dim {v} overflows i32"))
    };
    let rc = scatter_momentum_launch(
        bubbles_ptr as *const u8,
        to_i32(n_bubbles)?,
        uz_ptr as *mut f32,
        mat_ptr as *const i32,
        to_i32(nx)?, to_i32(ny)?, to_i32(nz)?,
        to_i32(uz_nx)?, to_i32(uz_ny)?, to_i32(uz_nz)?,
        ox, oy, oz, dx, dt,
        rho_l, rho_v, g_mag,
        mat_fluid,
    );
    if rc != 0 {
        return Err(anyhow!("scatter_momentum_launch returned cudaError_t {rc}"));
    }
    Ok(())
}

/// Phase 5.6 M5.6.C entry point. Scatter VOF alpha reduction + clamp to [0, 1].
#[allow(clippy::too_many_arguments)]
pub unsafe fn reduce_water_alpha_raw(
    _device: &Arc<CudaDevice>,
    bubbles_ptr: u64,
    n_bubbles: usize,
    alpha_ptr: u64,
    mat_ptr: u64,
    nx: usize, ny: usize, nz: usize,
    ox: f32, oy: f32, oz: f32,
    dx: f32,
    mat_fluid: i32,
) -> Result<()> {
    let to_i32 = |v: usize| -> Result<i32> {
        v.try_into().map_err(|_| anyhow!("dim {v} overflows i32"))
    };
    let rc = reduce_water_alpha_launch(
        bubbles_ptr as *const u8,
        to_i32(n_bubbles)?,
        alpha_ptr as *mut f32,
        mat_ptr as *const i32,
        to_i32(nx)?, to_i32(ny)?, to_i32(nz)?,
        ox, oy, oz, dx,
        mat_fluid,
    );
    if rc != 0 {
        return Err(anyhow!("reduce_water_alpha_launch returned cudaError_t {rc}"));
    }
    Ok(())
}

/// Phase 6 entry point: PCG pressure solve. Runs the entire iterative
/// solver inside one Rust→C++ call. Workspace buffers are caller-owned;
/// the launcher does no allocations.
///
/// # Safety
///
/// All pointer arguments must point to valid device memory in the primary
/// context of `device.ordinal()`. Workspace buffers must each carry at
/// least the documented number of floats (see `pressure_solve_pcg.cu`).
/// `iter_count_host` must point to a valid host i32 the launcher can
/// write through.
#[allow(clippy::too_many_arguments)]
pub unsafe fn pressure_solve_pcg_raw(
    _device: &Arc<CudaDevice>,
    p_out_ptr: u64,
    div_u_ptr: u64,
    mat_ptr: u64,
    nx: usize, ny: usize, nz: usize,
    dx: f32, dt: f32, rho: f32,
    mat_fluid: i32, mat_air: i32,
    pressure_tol: f32, max_iter: usize,
    ws_b_ptr: u64,
    ws_r_ptr: u64,
    ws_z_ptr: u64,
    ws_p_search_ptr: u64,
    ws_ap_ptr: u64,
    dot_workspace_ptr: u64,
    dev_alpha_ptr: u64,
    dev_beta_ptr: u64,
    dev_rzold_ptr: u64,
    dev_rznew_ptr: u64,
    dev_bsq_ptr: u64,
    dev_rsq_ptr: u64,
    dev_pap_ptr: u64,
) -> Result<usize> {
    let to_i32 = |v: usize| -> Result<i32> {
        v.try_into().map_err(|_| anyhow!("dim {v} overflows i32"))
    };
    let mut iter_count: i32 = 0;
    let rc = pressure_solve_pcg_launch(
        p_out_ptr as *mut f32,
        div_u_ptr as *const f32,
        mat_ptr as *const i32,
        to_i32(nx)?, to_i32(ny)?, to_i32(nz)?,
        dx, dt, rho,
        mat_fluid, mat_air,
        pressure_tol, to_i32(max_iter)?,
        ws_b_ptr as *mut f32,
        ws_r_ptr as *mut f32,
        ws_z_ptr as *mut f32,
        ws_p_search_ptr as *mut f32,
        ws_ap_ptr as *mut f32,
        dot_workspace_ptr as *mut f32,
        dev_alpha_ptr as *mut f32,
        dev_beta_ptr as *mut f32,
        dev_rzold_ptr as *mut f32,
        dev_rznew_ptr as *mut f32,
        dev_bsq_ptr as *mut f32,
        dev_rsq_ptr as *mut f32,
        dev_pap_ptr as *mut f32,
        &mut iter_count as *mut i32,
    );
    if rc != 0 {
        return Err(anyhow!("pressure_solve_pcg_launch returned cudaError_t {rc}"));
    }
    Ok(iter_count.max(0) as usize)
}

/// Phase 5.7 entry point. Per-bubble update kernel: growth, condensation,
/// Fritz departure, advection, vent/solid deactivation.
///
/// # Safety
///
/// `bubbles_ptr` points to `n_bubbles * 56` bytes of Bubble structs. `T_ptr`
/// is treated as mutable by the kernel (atomicAdd on the condensation scatter
/// path) -- pass the raw pointer here, the C signature accepts const but the
/// kernel uses atomicAdd so the caller must hold a mutable reference upstream.
/// All grid pointers must live in the primary context of `device.ordinal()`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn update_bubbles_raw(
    _device: &Arc<CudaDevice>,
    bubbles_ptr: u64,
    n_bubbles: usize,
    slot_claim_ptr: u64,
    site_active_ptr: u64,
    needs_fragment_ptr: u64,
    t_ptr: u64,
    mat_ptr: u64,
    nx: usize, ny: usize, nz: usize,
    ux_ptr: u64, ux_nx: usize, ux_ny: usize, ux_nz: usize,
    uy_ptr: u64, uy_nx: usize, uy_ny: usize, uy_nz: usize,
    uz_ptr: u64, uz_nx: usize, uz_ny: usize, uz_nz: usize,
    ox: f32, oy: f32, oz: f32,
    dx: f32, dt: f32,
    current_time: f32,
    water_line_z: f32,
    t_sat_k: f32,
    rho_l: f32, rho_v: f32,
    cp_l: f32, k_l: f32, h_lv: f32,
    sigma: f32,
    theta_rad: f32, g_mag: f32,
    r_seed: f32, r_frag: f32, r_max: f32,
    mat_fluid: i32,
) -> Result<()> {
    let to_i32 = |v: usize| -> Result<i32> {
        v.try_into().map_err(|_| anyhow!("dim {v} overflows i32"))
    };
    let rc = update_bubbles_launch(
        bubbles_ptr as *mut u8,
        to_i32(n_bubbles)?,
        slot_claim_ptr as *mut i32,
        site_active_ptr as *mut i32,
        needs_fragment_ptr as *mut i32,
        t_ptr as *const f32,
        mat_ptr as *const i32,
        to_i32(nx)?, to_i32(ny)?, to_i32(nz)?,
        ux_ptr as *const f32, to_i32(ux_nx)?, to_i32(ux_ny)?, to_i32(ux_nz)?,
        uy_ptr as *const f32, to_i32(uy_nx)?, to_i32(uy_ny)?, to_i32(uy_nz)?,
        uz_ptr as *const f32, to_i32(uz_nx)?, to_i32(uz_ny)?, to_i32(uz_nz)?,
        ox, oy, oz, dx, dt, current_time, water_line_z,
        t_sat_k, rho_l, rho_v, cp_l, k_l, h_lv, sigma,
        theta_rad, g_mag, r_seed, r_frag, r_max,
        mat_fluid,
    );
    if rc != 0 {
        return Err(anyhow!("update_bubbles_launch returned cudaError_t {rc}"));
    }
    Ok(())
}

/// Scaffold: allocate host & device buffers for vector_add and round-trip them.
///
/// Returns the elapsed time in seconds. The actual kernel launch is deferred
/// to Phase 5; this function verifies that cudarc device memory ops work.
pub fn run_vector_add_benchmark(device: &Arc<CudaDevice>, n: usize) -> Result<f64> {
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let _a_dev = device.htod_sync_copy(&a_host)?;
    let _b_dev = device.htod_sync_copy(&b_host)?;
    let c_dev: CudaSlice<f32> = device.alloc_zeros(n)?;

    device.synchronize()?;
    let start = std::time::Instant::now();
    // TODO(phase5): launch kernel via cudarc module API.
    device.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();

    let _c_host = device.dtoh_sync_copy(&c_dev)?;
    Ok(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_device_available() {
        let _device = CudaDevice::new(0).expect("No CUDA device found");
    }

    #[test]
    fn test_vector_add_compiles() {
        // Build succeeding means nvcc compiled vector_add.cu via build.rs.
    }

    #[test]
    fn test_round_trip_device_memory() {
        let device = CudaDevice::new(0).expect("No CUDA device found");
        let elapsed = run_vector_add_benchmark(&device, 1_000_000).unwrap();
        assert!(elapsed < 5.0, "Round trip took too long: {elapsed}s");
    }

    #[test]
    fn test_scale_inplace_doubles_values() {
        // Allocate a device buffer of 1s via cudarc, run the kernel,
        // copy back, assert all 2s.
        let device = CudaDevice::new(0).expect("No CUDA device found");
        let host: Vec<f32> = vec![1.0; 1024];
        let dev: CudaSlice<f32> = device
            .htod_sync_copy(&host)
            .expect("htod copy failed");
        // cudarc 0.12 exposes the raw device pointer via .device_ptr()
        let raw_ptr = *dev.device_ptr();
        unsafe {
            scale_inplace_raw(&device, raw_ptr, 1024, 2.0)
                .expect("scale_inplace failed");
        }
        let back: Vec<f32> = device.dtoh_sync_copy(&dev).expect("dtoh copy failed");
        assert_eq!(back.len(), 1024);
        for (i, v) in back.iter().enumerate() {
            assert_eq!(*v, 2.0, "index {i} got {v}, expected 2.0");
        }
    }
}
