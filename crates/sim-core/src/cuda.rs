//! CUDA-side handles for the Phase 5 Rust acceleration path.
//!
//! Registered as the `sim_core.cuda` submodule. Importing this submodule
//! does NOT eagerly open a CUDA device — the binding is registered, but
//! the first call to `SimCore::new(device_id)` is what touches the driver.
//! That way `from sim_core import props` works on a Mac dev machine even
//! though `sim_core.cuda.SimCore(0)` would fail there.

use cudarc::driver::CudaDevice;
use pyo3::prelude::*;
use std::sync::Arc;

/// Thin handle around a `cudarc::driver::CudaDevice`. M1 extends this with
/// the Warp ↔ cudarc pointer-sharing PoC (`scale_array`); M2 wires in the
/// hand-written Jacobi pressure kernel; M3 fuses the 200 launches into a
/// single CUDA Graph.
#[pyclass(module = "sim_core.cuda")]
pub struct SimCore {
    device: Arc<CudaDevice>,
    #[pyo3(get)]
    device_id: usize,
}

#[pymethods]
impl SimCore {
    #[new]
    fn new(device_id: usize) -> PyResult<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "CUDA device {device_id} unavailable: {e}. \
                 If you are on a CPU-only machine, use sim_core.props instead \
                 of sim_core.cuda."
            ))
        })?;
        Ok(Self { device, device_id })
    }

    fn device_info(&self) -> PyResult<String> {
        Ok(format!("CUDA device {} ready", self.device_id))
    }

    /// Phase 5 M1: in-place scale of an external device buffer.
    ///
    /// The caller passes the raw device pointer as a Python int -- typically
    /// taken from a Warp array via ``arr.__cuda_array_interface__["data"][0]``
    /// (the public CUDA Array Interface protocol). The kernel multiplies
    /// every float at that buffer by ``scale`` in place. Synchronous.
    ///
    /// This validates the Phase 5 FFI contract end-to-end: Python allocates,
    /// Rust mutates, Python reads back, without any host-side copy.
    fn scale_array(&self, device_ptr: u64, n: usize, scale: f32) -> PyResult<()> {
        // SAFETY: the caller guarantees ``device_ptr`` points to at least
        // ``n * size_of::<f32>()`` bytes of valid device memory in the primary
        // context for ``self.device_id``. Warp's allocator and cudarc's
        // ``CudaDevice::new(id)`` both target the primary context for the
        // same id, so a pointer produced by Warp on device 0 is safe to use
        // here when this ``SimCore`` was created with device_id=0. If the ids
        // mismatch this call is undefined behaviour -- there is no cheap way
        // to verify the binding belongs to the device. Document this on the
        // Python side and enforce by convention in tests.
        unsafe {
            cuda_kernels::scale_inplace_raw(&self.device, device_ptr, n, scale)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
        }
        Ok(())
    }

    /// Phase 5 M2: one Jacobi sweep of the pressure Poisson equation.
    ///
    /// Direct mirror of :py:func:`boilingsim.fluid.jacobi_pressure_step`.
    /// All four device pointers come in as Python ints (raw CUDA addresses,
    /// typically from `arr.__cuda_array_interface__["data"][0]` on the
    /// caller's Warp arrays). The kernel writes `p_new` and reads everything
    /// else; the caller is responsible for the ping-pong swap between
    /// iterations.
    ///
    /// The launch is asynchronous on the default stream so the caller can
    /// queue the full 200-iteration loop without per-iteration sync stalls.
    /// Pair with ``wp.synchronize_device("cuda:0")`` on the Python side
    /// before reading results.
    #[pyo3(signature = (
        p_new_ptr, p_old_ptr, div_u_ptr, mat_ptr,
        nx, ny, nz, dx, dt, rho,
        mat_fluid=0, mat_air=2,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn jacobi_pressure_step(
        &self,
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
    ) -> PyResult<()> {
        // SAFETY: same pointer-lifetime contract as scale_array. Caller
        // guarantees all four buffers live in the primary context of this
        // device_id and carry at least nx*ny*nz elements of the right type.
        unsafe {
            cuda_kernels::jacobi_pressure_step_raw(
                &self.device,
                p_new_ptr,
                p_old_ptr,
                div_u_ptr,
                mat_ptr,
                nx,
                ny,
                nz,
                dx,
                dt,
                rho,
                mat_fluid,
                mat_air,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    /// Phase 5 M3: fused 200-iteration pressure solve.
    ///
    /// Runs the entire Jacobi loop in CUDA C++ so only one Python→Rust
    /// transition happens per `pressure_projection` call instead of 200.
    /// On entry, `p` carries the initial guess; on exit, it carries the
    /// converged pressure (the C++ launcher handles the odd-iter memcpy
    /// internally). Asynchronous; pair with ``wp.synchronize_device`` on
    /// the Python side.
    #[pyo3(signature = (
        p_ptr, p_tmp_ptr, div_u_ptr, mat_ptr,
        nx, ny, nz, dx, dt, rho, n_iter,
        mat_fluid=0, mat_air=2,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pressure_solve(
        &self,
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
    ) -> PyResult<()> {
        // SAFETY: same pointer-lifetime contract as scale_array and
        // jacobi_pressure_step. p and p_tmp must be disjoint buffers; the
        // C++ launcher does ping-pong + a final memcpy to canonicalize.
        unsafe {
            cuda_kernels::pressure_solve_raw(
                &self.device,
                p_ptr, p_tmp_ptr, div_u_ptr, mat_ptr,
                nx, ny, nz, dx, dt, rho, n_iter,
                mat_fluid, mat_air,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    /// Phase 5.6 M5.6.A: scatter latent-heat sink across the fluid grid.
    ///
    /// Mirrors :py:func:`boilingsim.boiling.step_scatter_latent_heat` with
    /// the same arguments threaded through. The bubble pool's raw device
    /// pointer (from `wp.array(dtype=Bubble).__cuda_array_interface__`)
    /// comes in as a `u64`; on the C++ side it is interpreted as an array
    /// of 56-byte Bubble structs whose layout is locked in
    /// `crates/cuda-kernels/include/bubble.h` and validated by the parity
    /// tests in `python/tests/test_scatter_parity.py`.
    #[pyo3(signature = (
        bubbles_ptr, n_bubbles,
        t_ptr, mat_ptr,
        nx, ny, nz,
        ox, oy, oz, dx, dt, current_time,
        rho_l, cp_l, rho_v, h_lv, t_sat_k,
        mat_fluid=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn scatter_latent_heat(
        &self,
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
    ) -> PyResult<()> {
        // SAFETY: same pointer-lifetime contract as the other kernels.
        // bubbles_ptr must point to >= n_bubbles * 56 bytes of valid device
        // memory; T and mat must each be nx*ny*nz f32/i32 buffers.
        unsafe {
            cuda_kernels::scatter_latent_heat_raw(
                &self.device,
                bubbles_ptr, n_bubbles,
                t_ptr, mat_ptr,
                nx, ny, nz,
                ox, oy, oz, dx, dt, current_time,
                rho_l, cp_l, rho_v, h_lv, t_sat_k,
                mat_fluid,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    /// Phase 5.6 M5.6.B: scatter bubble momentum to z-faces.
    ///
    /// `uz` shape carries an extra k row vs `mat` (MAC z-faces), so both
    /// shapes are passed explicitly.
    #[pyo3(signature = (
        bubbles_ptr, n_bubbles,
        uz_ptr, mat_ptr,
        nx, ny, nz,
        uz_nx, uz_ny, uz_nz,
        ox, oy, oz, dx, dt,
        rho_l, rho_v, g_mag,
        mat_fluid=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn scatter_momentum(
        &self,
        bubbles_ptr: u64, n_bubbles: usize,
        uz_ptr: u64, mat_ptr: u64,
        nx: usize, ny: usize, nz: usize,
        uz_nx: usize, uz_ny: usize, uz_nz: usize,
        ox: f32, oy: f32, oz: f32,
        dx: f32, dt: f32,
        rho_l: f32, rho_v: f32, g_mag: f32,
        mat_fluid: i32,
    ) -> PyResult<()> {
        unsafe {
            cuda_kernels::scatter_momentum_raw(
                &self.device,
                bubbles_ptr, n_bubbles,
                uz_ptr, mat_ptr,
                nx, ny, nz, uz_nx, uz_ny, uz_nz,
                ox, oy, oz, dx, dt,
                rho_l, rho_v, g_mag,
                mat_fluid,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    /// Phase 5.6 M5.6.C: scatter bubble VOF alpha reduction + clamp.
    #[pyo3(signature = (
        bubbles_ptr, n_bubbles,
        alpha_ptr, mat_ptr,
        nx, ny, nz,
        ox, oy, oz, dx,
        mat_fluid=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn reduce_water_alpha(
        &self,
        bubbles_ptr: u64, n_bubbles: usize,
        alpha_ptr: u64, mat_ptr: u64,
        nx: usize, ny: usize, nz: usize,
        ox: f32, oy: f32, oz: f32, dx: f32,
        mat_fluid: i32,
    ) -> PyResult<()> {
        unsafe {
            cuda_kernels::reduce_water_alpha_raw(
                &self.device,
                bubbles_ptr, n_bubbles,
                alpha_ptr, mat_ptr,
                nx, ny, nz,
                ox, oy, oz, dx,
                mat_fluid,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    /// Phase 5.7: per-bubble update kernel.
    #[pyo3(signature = (
        bubbles_ptr, n_bubbles,
        slot_claim_ptr, site_active_ptr, needs_fragment_ptr,
        t_ptr, mat_ptr,
        nx, ny, nz,
        ux_ptr, ux_nx, ux_ny, ux_nz,
        uy_ptr, uy_nx, uy_ny, uy_nz,
        uz_ptr, uz_nx, uz_ny, uz_nz,
        ox, oy, oz, dx, dt,
        current_time, water_line_z,
        t_sat_k,
        rho_l, rho_v, cp_l, k_l, h_lv, sigma,
        theta_rad, g_mag,
        r_seed, r_frag, r_max,
        mat_fluid=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn update_bubbles(
        &self,
        bubbles_ptr: u64, n_bubbles: usize,
        slot_claim_ptr: u64, site_active_ptr: u64, needs_fragment_ptr: u64,
        t_ptr: u64, mat_ptr: u64,
        nx: usize, ny: usize, nz: usize,
        ux_ptr: u64, ux_nx: usize, ux_ny: usize, ux_nz: usize,
        uy_ptr: u64, uy_nx: usize, uy_ny: usize, uy_nz: usize,
        uz_ptr: u64, uz_nx: usize, uz_ny: usize, uz_nz: usize,
        ox: f32, oy: f32, oz: f32, dx: f32, dt: f32,
        current_time: f32, water_line_z: f32,
        t_sat_k: f32,
        rho_l: f32, rho_v: f32, cp_l: f32, k_l: f32, h_lv: f32, sigma: f32,
        theta_rad: f32, g_mag: f32,
        r_seed: f32, r_frag: f32, r_max: f32,
        mat_fluid: i32,
    ) -> PyResult<()> {
        unsafe {
            cuda_kernels::update_bubbles_raw(
                &self.device,
                bubbles_ptr, n_bubbles,
                slot_claim_ptr, site_active_ptr, needs_fragment_ptr,
                t_ptr, mat_ptr,
                nx, ny, nz,
                ux_ptr, ux_nx, ux_ny, ux_nz,
                uy_ptr, uy_nx, uy_ny, uy_nz,
                uz_ptr, uz_nx, uz_ny, uz_nz,
                ox, oy, oz, dx, dt, current_time, water_line_z,
                t_sat_k, rho_l, rho_v, cp_l, k_l, h_lv, sigma,
                theta_rad, g_mag, r_seed, r_frag, r_max,
                mat_fluid,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    /// Phase 6: PCG pressure solver.
    ///
    /// Mirrors :py:func:`boilingsim.fluid.pressure_projection` (Jacobi
    /// reference). Runs the entire CG loop in CUDA C++, with all state
    /// (residual, search direction, alpha, beta, etc.) device-resident.
    /// Returns the iteration count so the Python side can log convergence.
    ///
    /// All workspace pointers are caller-allocated. Layout:
    ///   - ws_b, ws_r, ws_z, ws_p_search, ws_ap: each `nx*ny*nz` floats
    ///   - dot_workspace: at least 1024 floats
    ///   - 7 device scalars: 1 float each (alpha, beta, rzold, rznew, bsq, rsq, pAp)
    #[pyo3(signature = (
        p_out_ptr, div_u_ptr, mat_ptr,
        nx, ny, nz,
        dx, dt, rho,
        pressure_tol, max_iter,
        ws_b_ptr, ws_r_ptr, ws_z_ptr, ws_p_search_ptr, ws_ap_ptr,
        dot_workspace_ptr,
        dev_alpha_ptr, dev_beta_ptr, dev_rzold_ptr, dev_rznew_ptr,
        dev_bsq_ptr, dev_rsq_ptr, dev_pap_ptr,
        mat_fluid=0, mat_air=2,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pressure_solve_pcg(
        &self,
        p_out_ptr: u64,
        div_u_ptr: u64,
        mat_ptr: u64,
        nx: usize, ny: usize, nz: usize,
        dx: f32, dt: f32, rho: f32,
        pressure_tol: f32, max_iter: usize,
        ws_b_ptr: u64, ws_r_ptr: u64, ws_z_ptr: u64,
        ws_p_search_ptr: u64, ws_ap_ptr: u64,
        dot_workspace_ptr: u64,
        dev_alpha_ptr: u64, dev_beta_ptr: u64,
        dev_rzold_ptr: u64, dev_rznew_ptr: u64,
        dev_bsq_ptr: u64, dev_rsq_ptr: u64, dev_pap_ptr: u64,
        mat_fluid: i32, mat_air: i32,
    ) -> PyResult<usize> {
        unsafe {
            cuda_kernels::pressure_solve_pcg_raw(
                &self.device,
                p_out_ptr, div_u_ptr, mat_ptr,
                nx, ny, nz,
                dx, dt, rho,
                mat_fluid, mat_air,
                pressure_tol, max_iter,
                ws_b_ptr, ws_r_ptr, ws_z_ptr, ws_p_search_ptr, ws_ap_ptr,
                dot_workspace_ptr,
                dev_alpha_ptr, dev_beta_ptr,
                dev_rzold_ptr, dev_rznew_ptr,
                dev_bsq_ptr, dev_rsq_ptr, dev_pap_ptr,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }
    }

    fn __repr__(&self) -> String {
        format!("SimCore(device_id={})", self.device_id)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimCore>()?;
    Ok(())
}
