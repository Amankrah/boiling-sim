"""Incompressible Navier-Stokes on a MAC grid with Boussinesq buoyancy.

Phase 2, Milestone C: everything needed for natural convection in a closed pot.

Step sequence (driven from ``pipeline.py`` in Milestone D):

    1. ``advect_velocity`` — semi-Lagrangian backtrace on each MAC face
    2. ``advect_temperature`` — SL backtrace on cell centres (solids skip)
    3. ``apply_buoyancy``    — Boussinesq source on z-faces
    4. thermal.conduct_one_step (Milestone B)
    5. ``enforce_no_slip``   — zero velocity on faces adjacent to solids
    6. ``pressure_projection`` — enforce ∇·u = 0 in fluid cells
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .config import ScenarioConfig
from .geometry import MAT_AIR, MAT_FLUID, Grid


# ---------------------------------------------------------------------------
# Trilinear sampling helpers
# ---------------------------------------------------------------------------
#
# Each scalar / face-velocity field lives on a lattice offset from the cell
# corner. ``offset`` is the half-integer shift per axis:
#   cell centres: (0.5, 0.5, 0.5)
#   x-faces:      (0.0, 0.5, 0.5)
#   y-faces:      (0.5, 0.0, 0.5)
#   z-faces:      (0.5, 0.5, 0.0)


@wp.func
def _tri_sample(
    field: wp.array3d(dtype=float),
    p: wp.vec3,
    origin: wp.vec3,
    dx: float,
    ox: float,
    oy: float,
    oz: float,
) -> float:
    """Trilinear-interpolate ``field`` at world-space point ``p``.

    ``ox/oy/oz`` are the half-integer offsets defining the lattice this field
    lives on (0 for face-aligned, 0.5 for cell-centre).
    """
    fx = (p[0] - origin[0]) / dx - ox
    fy = (p[1] - origin[1]) / dx - oy
    fz = (p[2] - origin[2]) / dx - oz

    # Clamp to valid interpolation range (one less than the upper axis bound).
    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]
    fx = wp.clamp(fx, 0.0, float(nx - 1) - 1.0e-6)
    fy = wp.clamp(fy, 0.0, float(ny - 1) - 1.0e-6)
    fz = wp.clamp(fz, 0.0, float(nz - 1) - 1.0e-6)

    i0 = int(fx)
    j0 = int(fy)
    k0 = int(fz)
    tx = fx - float(i0)
    ty = fy - float(j0)
    tz = fz - float(k0)

    c000 = field[i0, j0, k0]
    c100 = field[i0 + 1, j0, k0]
    c010 = field[i0, j0 + 1, k0]
    c110 = field[i0 + 1, j0 + 1, k0]
    c001 = field[i0, j0, k0 + 1]
    c101 = field[i0 + 1, j0, k0 + 1]
    c011 = field[i0, j0 + 1, k0 + 1]
    c111 = field[i0 + 1, j0 + 1, k0 + 1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@wp.func
def _sample_velocity_at(
    p: wp.vec3,
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
) -> wp.vec3:
    """Return the full (u, v, w) vector at world-space point ``p``."""
    u = _tri_sample(ux, p, origin, dx, 0.0, 0.5, 0.5)
    v = _tri_sample(uy, p, origin, dx, 0.5, 0.0, 0.5)
    w = _tri_sample(uz, p, origin, dx, 0.5, 0.5, 0.0)
    return wp.vec3(u, v, w)


# ---------------------------------------------------------------------------
# No-slip enforcement
# ---------------------------------------------------------------------------


@wp.kernel
def enforce_no_slip_ux(
    ux: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    mat_fluid: int,
):
    """Zero x-face velocities touching a solid cell (no-slip at walls)."""
    i, j, k = wp.tid()
    if i == 0 or i == ux.shape[0] - 1:
        ux[i, j, k] = 0.0
        return
    if mat[i - 1, j, k] != mat_fluid or mat[i, j, k] != mat_fluid:
        ux[i, j, k] = 0.0


@wp.kernel
def enforce_no_slip_uy(
    uy: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    mat_fluid: int,
):
    i, j, k = wp.tid()
    if j == 0 or j == uy.shape[1] - 1:
        uy[i, j, k] = 0.0
        return
    if mat[i, j - 1, k] != mat_fluid or mat[i, j, k] != mat_fluid:
        uy[i, j, k] = 0.0


@wp.kernel
def enforce_no_slip_uz(
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    mat_fluid: int,
):
    i, j, k = wp.tid()
    if k == 0 or k == uz.shape[2] - 1:
        uz[i, j, k] = 0.0
        return
    if mat[i, j, k - 1] != mat_fluid or mat[i, j, k] != mat_fluid:
        uz[i, j, k] = 0.0


def enforce_no_slip(grid: Grid, device: str = "cuda:0") -> None:
    """Zero all MAC faces that touch a non-fluid cell."""
    nx, ny, nz = grid.shape
    wp.launch(enforce_no_slip_ux, dim=(nx + 1, ny, nz),
              inputs=[grid.ux, grid.mat, MAT_FLUID], device=device)
    wp.launch(enforce_no_slip_uy, dim=(nx, ny + 1, nz),
              inputs=[grid.uy, grid.mat, MAT_FLUID], device=device)
    wp.launch(enforce_no_slip_uz, dim=(nx, ny, nz + 1),
              inputs=[grid.uz, grid.mat, MAT_FLUID], device=device)


# ---------------------------------------------------------------------------
# Boussinesq buoyancy
# ---------------------------------------------------------------------------


@wp.kernel
def apply_buoyancy(
    uz: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dt: float,
    g_mag: float,
    beta: float,
    T_ref: float,
    mat_fluid: int,
):
    """Add Δt · β · g · (T − T_ref) to uz on internal fluid z-faces.

    ``g_mag`` is the scalar gravitational magnitude (+9.81). Warm fluid
    (T > T_ref) rises, giving a positive increment to the upward-pointing
    z-face velocity.
    """
    i, j, k = wp.tid()
    if k == 0 or k == uz.shape[2] - 1:
        return
    if mat[i, j, k - 1] != mat_fluid or mat[i, j, k] != mat_fluid:
        return
    T_face = 0.5 * (T[i, j, k - 1] + T[i, j, k])
    uz[i, j, k] = uz[i, j, k] + dt * g_mag * beta * (T_face - T_ref)


# ---------------------------------------------------------------------------
# Divergence + pressure projection (Jacobi)
# ---------------------------------------------------------------------------


@wp.kernel
def compute_divergence(
    div_u: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    mat_fluid: int,
):
    """Cell-centred ∇·u on the MAC grid. Non-fluid cells set to 0."""
    i, j, k = wp.tid()
    if mat[i, j, k] != mat_fluid:
        div_u[i, j, k] = 0.0
        return
    dudx = (ux[i + 1, j, k] - ux[i, j, k]) / dx
    dvdy = (uy[i, j + 1, k] - uy[i, j, k]) / dx
    dwdz = (uz[i, j, k + 1] - uz[i, j, k]) / dx
    div_u[i, j, k] = dudx + dvdy + dwdz


@wp.func
def _pressure_neighbour(
    m_nbr: int,
    p_nbr: float,
    p_self: float,
    mat_fluid: int,
    mat_air: int,
) -> float:
    """Neighbour pressure for the Jacobi stencil under mixed BCs.

    * fluid neighbour: use its value (interior).
    * air neighbour (free surface): Dirichlet p=0.
    * solid neighbour (pot wall / carrot): Neumann ∂p/∂n=0 → ghost = self.
    """
    if m_nbr == mat_fluid:
        return p_nbr
    if m_nbr == mat_air:
        return 0.0
    return p_self  # Neumann at solid wall


@wp.kernel
def jacobi_pressure_step(
    p_new: wp.array3d(dtype=float),
    p_old: wp.array3d(dtype=float),
    div_u: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    dt: float,
    rho: float,
    mat_fluid: int,
    mat_air: int,
):
    """One Jacobi sweep of ∇²p = (ρ/dt)·div_u with mixed BCs.

    Neumann at solid walls (no-flow faces), Dirichlet p=0 at the free
    surface (air cells above the water). The stencil divisor stays 6
    because the Neumann ghost cell contributes ``p_self`` to the sum.
    """
    i, j, k = wp.tid()
    if mat[i, j, k] != mat_fluid:
        p_new[i, j, k] = 0.0
        return

    p_self = p_old[i, j, k]
    p_left = _pressure_neighbour(mat[i - 1, j, k], p_old[i - 1, j, k], p_self, mat_fluid, mat_air)
    p_right = _pressure_neighbour(mat[i + 1, j, k], p_old[i + 1, j, k], p_self, mat_fluid, mat_air)
    p_down = _pressure_neighbour(mat[i, j - 1, k], p_old[i, j - 1, k], p_self, mat_fluid, mat_air)
    p_up = _pressure_neighbour(mat[i, j + 1, k], p_old[i, j + 1, k], p_self, mat_fluid, mat_air)
    p_back = _pressure_neighbour(mat[i, j, k - 1], p_old[i, j, k - 1], p_self, mat_fluid, mat_air)
    p_front = _pressure_neighbour(mat[i, j, k + 1], p_old[i, j, k + 1], p_self, mat_fluid, mat_air)

    s = p_left + p_right + p_down + p_up + p_back + p_front
    rhs = rho * dx * dx * div_u[i, j, k] / dt
    p_new[i, j, k] = (s - rhs) / 6.0


@wp.kernel
def subtract_pressure_gradient_x(
    ux: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    dt: float,
    rho: float,
    mat_fluid: int,
):
    i, j, k = wp.tid()
    if i == 0 or i == ux.shape[0] - 1:
        return
    if mat[i - 1, j, k] != mat_fluid or mat[i, j, k] != mat_fluid:
        return
    ux[i, j, k] = ux[i, j, k] - dt / rho * (p[i, j, k] - p[i - 1, j, k]) / dx


@wp.kernel
def subtract_pressure_gradient_y(
    uy: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    dt: float,
    rho: float,
    mat_fluid: int,
):
    i, j, k = wp.tid()
    if j == 0 or j == uy.shape[1] - 1:
        return
    if mat[i, j - 1, k] != mat_fluid or mat[i, j, k] != mat_fluid:
        return
    uy[i, j, k] = uy[i, j, k] - dt / rho * (p[i, j, k] - p[i, j - 1, k]) / dx


@wp.kernel
def subtract_pressure_gradient_z(
    uz: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    dt: float,
    rho: float,
    mat_fluid: int,
):
    i, j, k = wp.tid()
    if k == 0 or k == uz.shape[2] - 1:
        return
    if mat[i, j, k - 1] != mat_fluid or mat[i, j, k] != mat_fluid:
        return
    uz[i, j, k] = uz[i, j, k] - dt / rho * (p[i, j, k] - p[i, j, k - 1]) / dx


# ---------------------------------------------------------------------------
# Semi-Lagrangian advection
# ---------------------------------------------------------------------------


@wp.kernel
def extend_temperature_into_solids(
    T_ext: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    mat_fluid: int,
):
    """For each non-fluid cell, copy the T of the nearest fluid neighbour.

    This produces a ``T_ext`` field that is smooth across fluid-solid
    boundaries, so the semi-Lagrangian trilinear sampler never reads a
    wildly different T just because the backtrace grazed a solid cell.
    Fluid cells keep their own T.
    """
    i, j, k = wp.tid()
    if mat[i, j, k] == mat_fluid:
        T_ext[i, j, k] = T[i, j, k]
        return

    # Search 6-neighbourhood for any fluid cell; use its T.
    nx = T.shape[0]
    ny = T.shape[1]
    nz = T.shape[2]
    if i > 0 and mat[i - 1, j, k] == mat_fluid:
        T_ext[i, j, k] = T[i - 1, j, k]
        return
    if i < nx - 1 and mat[i + 1, j, k] == mat_fluid:
        T_ext[i, j, k] = T[i + 1, j, k]
        return
    if j > 0 and mat[i, j - 1, k] == mat_fluid:
        T_ext[i, j, k] = T[i, j - 1, k]
        return
    if j < ny - 1 and mat[i, j + 1, k] == mat_fluid:
        T_ext[i, j, k] = T[i, j + 1, k]
        return
    if k > 0 and mat[i, j, k - 1] == mat_fluid:
        T_ext[i, j, k] = T[i, j, k - 1]
        return
    if k < nz - 1 and mat[i, j, k + 1] == mat_fluid:
        T_ext[i, j, k] = T[i, j, k + 1]
        return
    # No fluid neighbour: keep the cell's own T.
    T_ext[i, j, k] = T[i, j, k]


@wp.kernel
def advect_temperature(
    T_new: wp.array3d(dtype=float),
    T_true: wp.array3d(dtype=float),
    T_ext: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    mat_fluid: int,
):
    """Semi-Lagrangian advection of cell-centred T in fluid cells.

    ``T_ext`` is the extended field (from
    :func:`extend_temperature_into_solids`) used only for trilinear sampling
    so the sampler can't read a wildly different material's T across a
    boundary. Non-fluid cells in the output keep their *real* T
    (``T_true``) — not the propagated value from T_ext.
    """
    i, j, k = wp.tid()
    if mat[i, j, k] != mat_fluid:
        T_new[i, j, k] = T_true[i, j, k]
        return

    p = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx
    u = _sample_velocity_at(p, ux, uy, uz, origin, dx)
    p_back = p - u * dt
    T_new[i, j, k] = _tri_sample(T_ext, p_back, origin, dx, 0.5, 0.5, 0.5)


@wp.kernel
def advect_ux(
    ux_new: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    mat_fluid: int,
):
    """Semi-Lagrangian advection of u_x on the x-face grid."""
    i, j, k = wp.tid()
    if i == 0 or i == ux.shape[0] - 1:
        ux_new[i, j, k] = 0.0
        return
    if mat[i - 1, j, k] != mat_fluid and mat[i, j, k] != mat_fluid:
        ux_new[i, j, k] = ux[i, j, k]
        return

    p = origin + wp.vec3(float(i), float(j) + 0.5, float(k) + 0.5) * dx
    u = _sample_velocity_at(p, ux, uy, uz, origin, dx)
    p_back = p - u * dt
    ux_new[i, j, k] = _tri_sample(ux, p_back, origin, dx, 0.0, 0.5, 0.5)


@wp.kernel
def advect_uy(
    uy_new: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    mat_fluid: int,
):
    i, j, k = wp.tid()
    if j == 0 or j == uy.shape[1] - 1:
        uy_new[i, j, k] = 0.0
        return
    if mat[i, j - 1, k] != mat_fluid and mat[i, j, k] != mat_fluid:
        uy_new[i, j, k] = uy[i, j, k]
        return

    p = origin + wp.vec3(float(i) + 0.5, float(j), float(k) + 0.5) * dx
    u = _sample_velocity_at(p, ux, uy, uz, origin, dx)
    p_back = p - u * dt
    uy_new[i, j, k] = _tri_sample(uy, p_back, origin, dx, 0.5, 0.0, 0.5)


@wp.kernel
def advect_uz(
    uz_new: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    mat_fluid: int,
):
    i, j, k = wp.tid()
    if k == 0 or k == uz.shape[2] - 1:
        uz_new[i, j, k] = 0.0
        return
    if mat[i, j, k - 1] != mat_fluid and mat[i, j, k] != mat_fluid:
        uz_new[i, j, k] = uz[i, j, k]
        return

    p = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k)) * dx
    u = _sample_velocity_at(p, ux, uy, uz, origin, dx)
    p_back = p - u * dt
    uz_new[i, j, k] = _tri_sample(uz, p_back, origin, dx, 0.5, 0.5, 0.0)


# ---------------------------------------------------------------------------
# Python-side drivers (orchestrate kernel launches)
# ---------------------------------------------------------------------------


@dataclass
class FluidWorkspace:
    """Scratch arrays for fluid projection and advection."""

    div_u: wp.array            # (nx, ny, nz)
    p_tmp: wp.array            # (nx, ny, nz) Jacobi ping-pong buffer
    ux_tmp: wp.array           # (nx+1, ny, nz) SL velocity advection buffer
    uy_tmp: wp.array           # (nx, ny+1, nz)
    uz_tmp: wp.array           # (nx, ny, nz+1)
    T_tmp: wp.array            # (nx, ny, nz) SL temperature advection output
    T_ext: wp.array            # (nx, ny, nz) T extended into non-fluid cells
    u_max_scalar: wp.array     # (1,) GPU-side reduction target for compute_max_velocity


def allocate_fluid_workspace(grid: Grid, device: str = "cuda:0") -> FluidWorkspace:
    nx, ny, nz = grid.shape
    return FluidWorkspace(
        div_u=wp.zeros((nx, ny, nz), dtype=float, device=device),
        p_tmp=wp.zeros((nx, ny, nz), dtype=float, device=device),
        ux_tmp=wp.zeros((nx + 1, ny, nz), dtype=float, device=device),
        uy_tmp=wp.zeros((nx, ny + 1, nz), dtype=float, device=device),
        uz_tmp=wp.zeros((nx, ny, nz + 1), dtype=float, device=device),
        T_tmp=wp.zeros((nx, ny, nz), dtype=float, device=device),
        T_ext=wp.zeros((nx, ny, nz), dtype=float, device=device),
        u_max_scalar=wp.zeros(1, dtype=float, device=device),
    )


def pressure_projection(
    grid: Grid,
    ws: FluidWorkspace,
    cfg: ScenarioConfig,
    dt: float,
    rho: float,
    device: str = "cuda:0",
) -> float:
    """Enforce incompressibility: solve ∇²p = (ρ/dt)·∇·u then correct u.

    Returns the final max |∇·u| over fluid cells (a diagnostic).
    """
    nx, ny, nz = grid.shape
    dx = grid.dx

    # Zero the initial pressure guess for a clean start.
    grid.p.zero_()
    ws.p_tmp.zero_()

    # RHS from current velocity divergence
    wp.launch(
        compute_divergence,
        dim=(nx, ny, nz),
        inputs=[ws.div_u, grid.ux, grid.uy, grid.uz, grid.mat, dx, MAT_FLUID],
        device=device,
    )

    # Jacobi iterations (ping-pong p ↔ p_tmp).
    # After N iterations, the result lives in p_tmp when N is odd, in grid.p
    # when N is even.
    for it in range(cfg.solver.pressure_max_iter):
        src = grid.p if (it % 2 == 0) else ws.p_tmp
        dst = ws.p_tmp if (it % 2 == 0) else grid.p
        wp.launch(
            jacobi_pressure_step,
            dim=(nx, ny, nz),
            inputs=[dst, src, ws.div_u, grid.mat, dx, dt, rho, MAT_FLUID, MAT_AIR],
            device=device,
        )
    if cfg.solver.pressure_max_iter % 2 == 1:
        # Odd N → last write was to p_tmp; canonicalize into grid.p.
        wp.copy(grid.p, ws.p_tmp)

    # Correct velocities
    wp.launch(
        subtract_pressure_gradient_x,
        dim=(nx + 1, ny, nz),
        inputs=[grid.ux, grid.p, grid.mat, dx, dt, rho, MAT_FLUID],
        device=device,
    )
    wp.launch(
        subtract_pressure_gradient_y,
        dim=(nx, ny + 1, nz),
        inputs=[grid.uy, grid.p, grid.mat, dx, dt, rho, MAT_FLUID],
        device=device,
    )
    wp.launch(
        subtract_pressure_gradient_z,
        dim=(nx, ny, nz + 1),
        inputs=[grid.uz, grid.p, grid.mat, dx, dt, rho, MAT_FLUID],
        device=device,
    )

    # Recompute divergence as a diagnostic and return max |div|.
    wp.launch(
        compute_divergence,
        dim=(nx, ny, nz),
        inputs=[ws.div_u, grid.ux, grid.uy, grid.uz, grid.mat, dx, MAT_FLUID],
        device=device,
    )
    wp.synchronize_device(device)
    return float(abs(ws.div_u.numpy()).max())


def advect_all(
    grid: Grid,
    ws: FluidWorkspace,
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Semi-Lagrangian advection of u, T. Reads from grid.*, writes into tmp,
    then swaps pointers so grid.* holds the advected fields."""
    nx, ny, nz = grid.shape
    origin = wp.vec3(*grid.origin)
    dx = grid.dx

    # Velocity: semi-Lagrangian into tmp arrays
    wp.launch(advect_ux, dim=(nx + 1, ny, nz),
              inputs=[ws.ux_tmp, grid.ux, grid.uy, grid.uz, grid.mat, origin, dx, dt, MAT_FLUID],
              device=device)
    wp.launch(advect_uy, dim=(nx, ny + 1, nz),
              inputs=[ws.uy_tmp, grid.ux, grid.uy, grid.uz, grid.mat, origin, dx, dt, MAT_FLUID],
              device=device)
    wp.launch(advect_uz, dim=(nx, ny, nz + 1),
              inputs=[ws.uz_tmp, grid.ux, grid.uy, grid.uz, grid.mat, origin, dx, dt, MAT_FLUID],
              device=device)

    # Temperature: first extend T into neighbouring non-fluid cells so the
    # SL trilinear sampler can't read wildly different material Ts, then
    # run semi-Lagrangian backtrace.
    wp.launch(extend_temperature_into_solids, dim=(nx, ny, nz),
              inputs=[ws.T_ext, grid.T, grid.mat, MAT_FLUID],
              device=device)
    wp.launch(advect_temperature, dim=(nx, ny, nz),
              inputs=[ws.T_tmp, grid.T, ws.T_ext, grid.ux, grid.uy, grid.uz, grid.mat,
                      origin, dx, dt, MAT_FLUID],
              device=device)

    # Swap: advected fields live in the workspace; copy back into grid.
    wp.copy(grid.ux, ws.ux_tmp)
    wp.copy(grid.uy, ws.uy_tmp)
    wp.copy(grid.uz, ws.uz_tmp)
    wp.copy(grid.T, ws.T_tmp)


def apply_buoyancy_step(
    grid: Grid,
    cfg: ScenarioConfig,
    dt: float,
    beta: float,
    T_ref_k: float,
    device: str = "cuda:0",
) -> None:
    """Add the Boussinesq source to the vertical face velocities."""
    nx, ny, nz = grid.shape
    wp.launch(
        apply_buoyancy,
        dim=(nx, ny, nz + 1),
        inputs=[grid.uz, grid.T, grid.mat, dt, 9.81, beta, T_ref_k, MAT_FLUID],
        device=device,
    )


@wp.kernel
def _zero_scalar(out: wp.array(dtype=float)):
    out[0] = float(0.0)


@wp.kernel
def _atomic_max_abs(field: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    wp.atomic_max(out, 0, wp.abs(field[i, j, k]))


def compute_max_velocity(grid: Grid, ws: "FluidWorkspace | None" = None) -> float:
    """Return max |u| across all MAC faces.

    When ``ws`` is provided, does a GPU-side reduction into
    ``ws.u_max_scalar`` (three tiny kernel launches + one 4-byte readback)
    — this is what ``compute_dt`` hits every step. Without ``ws``, falls
    back to the legacy three-full-field ``.numpy()`` roundtrip so external
    callers (tests, ad-hoc diagnostics) don't have to thread a workspace
    around.

    The GPU path collapses the per-step cost from ~50 ms (3 × ~16 MB
    PCIe transfer + host ``abs().max()``) to sub-millisecond, which
    matters for production 600 s runs at 0.5 mm where ``compute_dt``
    fires ~600 k times.
    """
    if ws is not None:
        device = ws.u_max_scalar.device
        wp.launch(_zero_scalar, dim=1, inputs=[ws.u_max_scalar], device=device)
        wp.launch(_atomic_max_abs, dim=grid.ux.shape,
                  inputs=[grid.ux, ws.u_max_scalar], device=device)
        wp.launch(_atomic_max_abs, dim=grid.uy.shape,
                  inputs=[grid.uy, ws.u_max_scalar], device=device)
        wp.launch(_atomic_max_abs, dim=grid.uz.shape,
                  inputs=[grid.uz, ws.u_max_scalar], device=device)
        return float(ws.u_max_scalar.numpy()[0])

    ux = grid.ux.numpy()
    uy = grid.uy.numpy()
    uz = grid.uz.numpy()
    return float(max(abs(ux).max(), abs(uy).max(), abs(uz).max()))


def compute_cfl_dt(grid: Grid, cfg: ScenarioConfig, ws: "FluidWorkspace | None" = None) -> float:
    """CFL-limited Δt from current max velocity."""
    u_max = compute_max_velocity(grid, ws=ws)
    if u_max < 1e-8:
        return cfg.solver.max_dt_s
    return cfg.solver.cfl_safety_factor * grid.dx / u_max
