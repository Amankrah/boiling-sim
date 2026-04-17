"""Thermal conduction with conjugate heat transfer.

Solves the energy equation

    ρ·c_p · ∂T/∂t = ∇·(k ∇T) + S_T

over the three-domain grid using a harmonic-mean face conductivity
(guide §2.3, §2.5). This module is Phase 2, Milestone B: conduction only —
no advection, that lands in Milestone C.

Boundary conditions implemented here:
  * Stove base heat flux on bottom pot cells  (:func:`apply_base_heat_flux`).
  * Newton cooling on outer pot wall           (baked into the flux kernel
    via a special h_conv·dx effective-k at pot-wall ↔ air interfaces).
  * Evaporative cooling on water surface cells (:func:`apply_evaporative_cooling`),
    placeholder at ``0.1·q_base`` until Phase 3 adds the proper Stefan condition.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import warp as wp

from .config import ScenarioConfig
from .geometry import (
    MAT_AIR,
    MAT_CARROT,
    MAT_FLUID,
    MAT_POT_WALL,
    Grid,
)


# ---------------------------------------------------------------------------
# Material properties  (per-material-ID, on device)
# ---------------------------------------------------------------------------


@dataclass
class MaterialProps:
    """Per-material thermophysical properties (indexed by material ID).

    Arrays have length 4, ordered [fluid, pot_wall, air, carrot] matching the
    MAT_* constants. All units SI: kg/m³, J/(kg·K), W/(m·K).
    """

    rho: np.ndarray
    c_p: np.ndarray
    k: np.ndarray
    rho_wp: wp.array
    cp_wp: wp.array
    k_wp: wp.array

    @classmethod
    def from_scenario(
        cls,
        cfg: ScenarioConfig,
        device: str = "cuda:0",
        materials_path: str | pathlib.Path | None = None,
    ) -> "MaterialProps":
        """Load the four material properties arrays from materials.json."""
        if materials_path is None:
            materials_path = (
                pathlib.Path(__file__).resolve().parents[2] / "data" / "materials.json"
            )
        data = json.loads(pathlib.Path(materials_path).read_text())

        water = data["water"]
        pot = data[cfg.pot.material]
        carrot = data["carrot"]

        # Air values: dry-air at 25°C, 1 atm.
        air_rho = 1.184
        air_cp = 1005.0
        air_k = 0.0262

        rho = np.array([water["rho_ref"], pot["rho"], air_rho, carrot["rho"]], dtype=np.float32)
        c_p = np.array([water["c_p"], pot["c_p"], air_cp, carrot["c_p"]], dtype=np.float32)
        k = np.array([water["k"], pot["k"], air_k, carrot["k"]], dtype=np.float32)

        return cls(
            rho=rho, c_p=c_p, k=k,
            rho_wp=wp.array(rho, dtype=float, device=device),
            cp_wp=wp.array(c_p, dtype=float, device=device),
            k_wp=wp.array(k, dtype=float, device=device),
        )


# ---------------------------------------------------------------------------
# Conduction flux kernels
# ---------------------------------------------------------------------------
#
# For each face we compute an outgoing flux  F = -k_face · (T_right - T_left) / dx
# (units W/m²). At a solid ↔ air interface we replace the harmonic mean with
# an effective k_face = h_conv · dx so that F = h_conv · (T_solid - T_air),
# which is Newton cooling with coefficient h_conv.


@wp.func
def _k_face(
    mat_left: int,
    mat_right: int,
    k_arr: wp.array(dtype=float),
    dx: float,
    h_conv_air: float,
    mat_air: int,
) -> float:
    # Air ↔ solid face → Newton cooling (effective k so F = h_conv·ΔT)
    if (mat_left == mat_air) != (mat_right == mat_air):
        return h_conv_air * dx
    # Standard harmonic mean
    kl = k_arr[mat_left]
    kr = k_arr[mat_right]
    return 2.0 * kl * kr / (kl + kr + 1.0e-12)


@wp.kernel
def heat_conduction_flux_x(
    flux_x: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    k_arr: wp.array(dtype=float),
    dx: float,
    h_conv_air: float,
    mat_air: int,
):
    """Compute x-face heat fluxes (shape (nx+1, ny, nz))."""
    i, j, k = wp.tid()

    # Boundary faces are adiabatic (zero flux) — no neighbour to sample.
    if i == 0 or i == flux_x.shape[0] - 1:
        flux_x[i, j, k] = 0.0
        return

    m_l = mat[i - 1, j, k]
    m_r = mat[i, j, k]
    k_f = _k_face(m_l, m_r, k_arr, dx, h_conv_air, mat_air)
    flux_x[i, j, k] = -k_f * (T[i, j, k] - T[i - 1, j, k]) / dx


@wp.kernel
def heat_conduction_flux_y(
    flux_y: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    k_arr: wp.array(dtype=float),
    dx: float,
    h_conv_air: float,
    mat_air: int,
):
    i, j, k = wp.tid()
    if j == 0 or j == flux_y.shape[1] - 1:
        flux_y[i, j, k] = 0.0
        return
    m_l = mat[i, j - 1, k]
    m_r = mat[i, j, k]
    k_f = _k_face(m_l, m_r, k_arr, dx, h_conv_air, mat_air)
    flux_y[i, j, k] = -k_f * (T[i, j, k] - T[i, j - 1, k]) / dx


@wp.kernel
def heat_conduction_flux_z(
    flux_z: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    k_arr: wp.array(dtype=float),
    dx: float,
    h_conv_air: float,
    mat_air: int,
):
    i, j, k = wp.tid()
    if k == 0 or k == flux_z.shape[2] - 1:
        flux_z[i, j, k] = 0.0
        return
    m_l = mat[i, j, k - 1]
    m_r = mat[i, j, k]
    k_f = _k_face(m_l, m_r, k_arr, dx, h_conv_air, mat_air)
    flux_z[i, j, k] = -k_f * (T[i, j, k] - T[i, j, k - 1]) / dx


@wp.kernel
def apply_conduction_update(
    T: wp.array3d(dtype=float),
    flux_x: wp.array3d(dtype=float),
    flux_y: wp.array3d(dtype=float),
    flux_z: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    rho_arr: wp.array(dtype=float),
    cp_arr: wp.array(dtype=float),
    dx: float,
    dt: float,
    mat_air: int,
):
    """Explicit-Euler cell-centred temperature update.

    dT = dt / (ρ·c_p·dx) · Σ(F_in - F_out)

    Air cells (``mat == mat_air``) are skipped — they have a low heat
    capacity that makes them stability-limiting out of all proportion to
    their physical role. Newton cooling at solid↔air faces is handled by
    the flux kernels directly (``h_conv·dx`` effective k), so the solid
    side still receives the correct heat loss even though the air cell
    itself doesn't update.
    """
    i, j, k = wp.tid()
    m = mat[i, j, k]
    if m == mat_air:
        return
    rho = rho_arr[m]
    cp = cp_arr[m]

    net = (
        (flux_x[i, j, k] - flux_x[i + 1, j, k])
        + (flux_y[i, j, k] - flux_y[i, j + 1, k])
        + (flux_z[i, j, k] - flux_z[i, j, k + 1])
    )
    T[i, j, k] = T[i, j, k] + dt * net / (rho * cp * dx)


@wp.kernel
def apply_implicit_conduction_update(
    T_dst: wp.array3d(dtype=float),
    T_iter: wp.array3d(dtype=float),
    T_old: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    k_arr: wp.array(dtype=float),
    rho_arr: wp.array(dtype=float),
    cp_arr: wp.array(dtype=float),
    dx: float,
    dt: float,
    h_conv_air: float,
    T_amb_k: float,
    mat_air: int,
    mat_fluid: int,
):
    """One Jacobi sweep of the backward-Euler conduction system.

    The BE system for each non-air cell is
        (1 + Σ γ) · T_new = T_old + Σ γ · T_nbr
    where γ = dt · k_face / (ρ·c_p·dx²) for fluid↔fluid or conjugate
    faces, and γ = dt · h_conv / (ρ·c_p·dx) at solid↔air faces (with the
    neighbour fixed to T_amb as a Dirichlet condition). Jacobi uses the
    previous iterate's T values for the off-diagonal sum.

    Unconditionally stable regardless of α or Δt. Air cells don't update.
    """
    i, j, k = wp.tid()
    m_self = mat[i, j, k]
    if m_self == mat_air:
        return

    rho = rho_arr[m_self]
    cp = cp_arr[m_self]
    k_self = k_arr[m_self]
    inv_rho_cp = 1.0 / (rho * cp)

    nx = mat.shape[0]
    ny = mat.shape[1]
    nz = mat.shape[2]

    rhs_sum = 0.0
    diag_extra = 0.0

    # -x face
    if i > 0:
        m_nbr = mat[i - 1, j, k]
        if m_nbr == mat_air:
            gamma = dt * h_conv_air * inv_rho_cp / dx
            rhs_sum = rhs_sum + gamma * T_amb_k
            diag_extra = diag_extra + gamma
        else:
            k_nbr = k_arr[m_nbr]
            k_face = 2.0 * k_self * k_nbr / (k_self + k_nbr + 1.0e-12)
            gamma = dt * k_face * inv_rho_cp / (dx * dx)
            rhs_sum = rhs_sum + gamma * T_iter[i - 1, j, k]
            diag_extra = diag_extra + gamma
    # +x face
    if i < nx - 1:
        m_nbr = mat[i + 1, j, k]
        if m_nbr == mat_air:
            gamma = dt * h_conv_air * inv_rho_cp / dx
            rhs_sum = rhs_sum + gamma * T_amb_k
            diag_extra = diag_extra + gamma
        else:
            k_nbr = k_arr[m_nbr]
            k_face = 2.0 * k_self * k_nbr / (k_self + k_nbr + 1.0e-12)
            gamma = dt * k_face * inv_rho_cp / (dx * dx)
            rhs_sum = rhs_sum + gamma * T_iter[i + 1, j, k]
            diag_extra = diag_extra + gamma
    # -y face
    if j > 0:
        m_nbr = mat[i, j - 1, k]
        if m_nbr == mat_air:
            gamma = dt * h_conv_air * inv_rho_cp / dx
            rhs_sum = rhs_sum + gamma * T_amb_k
            diag_extra = diag_extra + gamma
        else:
            k_nbr = k_arr[m_nbr]
            k_face = 2.0 * k_self * k_nbr / (k_self + k_nbr + 1.0e-12)
            gamma = dt * k_face * inv_rho_cp / (dx * dx)
            rhs_sum = rhs_sum + gamma * T_iter[i, j - 1, k]
            diag_extra = diag_extra + gamma
    # +y face
    if j < ny - 1:
        m_nbr = mat[i, j + 1, k]
        if m_nbr == mat_air:
            gamma = dt * h_conv_air * inv_rho_cp / dx
            rhs_sum = rhs_sum + gamma * T_amb_k
            diag_extra = diag_extra + gamma
        else:
            k_nbr = k_arr[m_nbr]
            k_face = 2.0 * k_self * k_nbr / (k_self + k_nbr + 1.0e-12)
            gamma = dt * k_face * inv_rho_cp / (dx * dx)
            rhs_sum = rhs_sum + gamma * T_iter[i, j + 1, k]
            diag_extra = diag_extra + gamma
    # -z face
    if k > 0:
        m_nbr = mat[i, j, k - 1]
        if m_nbr == mat_air:
            gamma = dt * h_conv_air * inv_rho_cp / dx
            rhs_sum = rhs_sum + gamma * T_amb_k
            diag_extra = diag_extra + gamma
        else:
            k_nbr = k_arr[m_nbr]
            k_face = 2.0 * k_self * k_nbr / (k_self + k_nbr + 1.0e-12)
            gamma = dt * k_face * inv_rho_cp / (dx * dx)
            rhs_sum = rhs_sum + gamma * T_iter[i, j, k - 1]
            diag_extra = diag_extra + gamma
    # +z face
    if k < nz - 1:
        m_nbr = mat[i, j, k + 1]
        if m_nbr == mat_air:
            gamma = dt * h_conv_air * inv_rho_cp / dx
            rhs_sum = rhs_sum + gamma * T_amb_k
            diag_extra = diag_extra + gamma
        else:
            k_nbr = k_arr[m_nbr]
            k_face = 2.0 * k_self * k_nbr / (k_self + k_nbr + 1.0e-12)
            gamma = dt * k_face * inv_rho_cp / (dx * dx)
            rhs_sum = rhs_sum + gamma * T_iter[i, j, k + 1]
            diag_extra = diag_extra + gamma

    T_dst[i, j, k] = (T_old[i, j, k] + rhs_sum) / (1.0 + diag_extra)


# ---------------------------------------------------------------------------
# Boundary conditions as direct volumetric source terms
# ---------------------------------------------------------------------------


@wp.kernel
def apply_base_heat_flux(
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    rho_arr: wp.array(dtype=float),
    cp_arr: wp.array(dtype=float),
    origin: wp.vec3,
    dx: float,
    dt: float,
    q_base: float,
    mat_pot_wall: int,
):
    """Inject a uniform heat flux q_base [W/m²] at the stove interface.

    Applied to pot-wall cells whose ``mat[i,j,k-1] == air`` — i.e., cells
    that sit directly above the air padding below the pot. This is
    precisely the single layer that *is* the stove contact surface;
    applying q_base to multiple layers would multiply the intended
    surface flux by the number of layers.

    Cell update: ΔT = q_base · dx² · dt / (ρ·c_p·dx³) = q_base · dt / (ρ·c_p·dx).
    """
    i, j, k = wp.tid()
    if mat[i, j, k] != mat_pot_wall:
        return
    # k-1 must exist and be an air cell (2 = MAT_AIR).
    if k == 0:
        return
    if mat[i, j, k - 1] != 2:  # MAT_AIR
        return
    m = mat[i, j, k]
    T[i, j, k] = T[i, j, k] + dt * q_base / (rho_arr[m] * cp_arr[m] * dx)


@wp.kernel
def apply_evaporative_cooling(
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    rho_arr: wp.array(dtype=float),
    cp_arr: wp.array(dtype=float),
    origin: wp.vec3,
    dx: float,
    dt: float,
    q_evap: float,
    water_line_z: float,
    mat_fluid: int,
    T_onset_k: float,
    T_sat_k: float,
):
    """Temperature-gated placeholder evaporative cooling at the free surface.

    The loss ramps from 0 at ``T_onset_k`` to ``q_evap`` at ``T_sat_k``
    (linear). Below ``T_onset`` the sink is zero, so cold water doesn't get
    cooled further. Replaced by a Stefan-condition-driven mass sink in
    Phase 3.
    """
    i, j, k = wp.tid()
    if mat[i, j, k] != mat_fluid:
        return
    z = origin[2] + (float(k) + 0.5) * dx
    if z < water_line_z - dx or z > water_line_z:
        return
    T_cell = T[i, j, k]
    frac = wp.clamp((T_cell - T_onset_k) / (T_sat_k - T_onset_k), 0.0, 1.0)
    if frac <= 0.0:
        return
    m = mat[i, j, k]
    T[i, j, k] = T_cell - dt * q_evap * frac / (rho_arr[m] * cp_arr[m] * dx)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class ThermalWorkspace:
    """Face-flux arrays + implicit-BE scratch, reused across conduction steps."""

    flux_x: wp.array
    flux_y: wp.array
    flux_z: wp.array
    T_old_snapshot: wp.array   # (nx, ny, nz) T at start of step (for BE)
    T_iter_alt: wp.array       # (nx, ny, nz) Jacobi ping-pong buffer


def allocate_thermal_workspace(grid: Grid, device: str = "cuda:0") -> ThermalWorkspace:
    nx, ny, nz = grid.shape
    return ThermalWorkspace(
        flux_x=wp.zeros((nx + 1, ny, nz), dtype=float, device=device),
        flux_y=wp.zeros((nx, ny + 1, nz), dtype=float, device=device),
        flux_z=wp.zeros((nx, ny, nz + 1), dtype=float, device=device),
        T_old_snapshot=wp.zeros((nx, ny, nz), dtype=float, device=device),
        T_iter_alt=wp.zeros((nx, ny, nz), dtype=float, device=device),
    )


def compute_max_dt_conduction(
    props: MaterialProps, dx: float, safety: float = 0.8,
    exclude_air: bool = True,
) -> float:
    """Explicit-Euler stability: dt ≤ dx² / (2·d·α_max). d=3 spatial dims.

    ``exclude_air=True`` drops the air material's α from the max. Air has a
    low ρ (1.2 kg/m³) giving it a misleadingly high α despite carrying almost
    no heat; Newton cooling at solid-air interfaces (baked into the flux
    kernel as h_conv·dx) handles the physical heat loss separately.
    """
    alpha = props.k / (props.rho * props.c_p)
    if exclude_air:
        # Air is index 2 (MAT_AIR). Mask it out.
        alpha_no_air = np.concatenate([alpha[:2], alpha[3:]])
        alpha_max = float(alpha_no_air.max())
    else:
        alpha_max = float(alpha.max())
    return safety * dx ** 2 / (2.0 * 3.0 * alpha_max)


def _launch_flux_kernels(grid: Grid, props: MaterialProps, ws: ThermalWorkspace,
                          T_field: wp.array, h_conv: float, device: str) -> None:
    """Compute all three MAC-face heat fluxes from ``T_field``."""
    nx, ny, nz = grid.shape
    dx = grid.dx
    wp.launch(
        heat_conduction_flux_x,
        dim=(nx + 1, ny, nz),
        inputs=[ws.flux_x, T_field, grid.mat, props.k_wp, dx, h_conv, MAT_AIR],
        device=device,
    )
    wp.launch(
        heat_conduction_flux_y,
        dim=(nx, ny + 1, nz),
        inputs=[ws.flux_y, T_field, grid.mat, props.k_wp, dx, h_conv, MAT_AIR],
        device=device,
    )
    wp.launch(
        heat_conduction_flux_z,
        dim=(nx, ny, nz + 1),
        inputs=[ws.flux_z, T_field, grid.mat, props.k_wp, dx, h_conv, MAT_AIR],
        device=device,
    )


def conduct_one_step(
    grid: Grid,
    props: MaterialProps,
    ws: ThermalWorkspace,
    cfg: ScenarioConfig,
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Advance the temperature field by one conduction step with BCs applied.

    Dispatches to explicit-Euler or Jacobi backward-Euler depending on
    ``cfg.solver.use_implicit_conduction``.
    """
    nx, ny, nz = grid.shape
    dx = grid.dx
    h_conv = cfg.solver.h_conv_outer_w_per_m2_k

    if cfg.solver.use_implicit_conduction:
        # ---- Backward-Euler Jacobi ----
        # Snapshot T_old (the right-hand side of the BE system).
        wp.copy(ws.T_old_snapshot, grid.T)
        T_amb_k = cfg.heating.ambient_temp_c + 273.15
        n_iter = cfg.solver.diffusion_max_iter
        for it in range(n_iter):
            src = grid.T if (it % 2 == 0) else ws.T_iter_alt
            dst = ws.T_iter_alt if (it % 2 == 0) else grid.T
            wp.launch(
                apply_implicit_conduction_update,
                dim=(nx, ny, nz),
                inputs=[dst, src, ws.T_old_snapshot, grid.mat, props.k_wp,
                        props.rho_wp, props.cp_wp, dx, dt, h_conv, T_amb_k,
                        MAT_AIR, MAT_FLUID],
                device=device,
            )
        if n_iter % 2 == 1:
            wp.copy(grid.T, ws.T_iter_alt)
    else:
        # ---- Explicit Euler ----
        _launch_flux_kernels(grid, props, ws, grid.T, h_conv, device)
        wp.launch(
            apply_conduction_update,
            dim=(nx, ny, nz),
            inputs=[grid.T, ws.flux_x, ws.flux_y, ws.flux_z, grid.mat,
                    props.rho_wp, props.cp_wp, dx, dt, MAT_AIR],
            device=device,
        )

    # 3. Source terms.
    wp.launch(
        apply_base_heat_flux,
        dim=(nx, ny, nz),
        inputs=[grid.T, grid.mat, props.rho_wp, props.cp_wp,
                wp.vec3(*grid.origin), dx, dt,
                cfg.heating.base_heat_flux_w_per_m2,
                MAT_POT_WALL],
        device=device,
    )

    h_inner = cfg.pot.height_m - cfg.pot.base_thickness_m
    water_line_z = cfg.pot.base_thickness_m + cfg.water.fill_fraction * h_inner
    q_evap = 0.1 * cfg.heating.base_heat_flux_w_per_m2  # placeholder until Phase 3
    T_onset_k = 85.0 + 273.15  # K — evaporation ramp begins here
    T_sat_k = 100.0 + 273.15   # K — full q_evap at boiling
    wp.launch(
        apply_evaporative_cooling,
        dim=(nx, ny, nz),
        inputs=[grid.T, grid.mat, props.rho_wp, props.cp_wp,
                wp.vec3(*grid.origin), dx, dt,
                q_evap, water_line_z, MAT_FLUID,
                T_onset_k, T_sat_k],
        device=device,
    )
