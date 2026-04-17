"""Tests for the Phase 2 Milestone C fluid solver."""

import math

import numpy as np
import pytest
import warp as wp

from boilingsim.config import ScenarioConfig, load_scenario
from boilingsim.fluid import (
    advect_all,
    allocate_fluid_workspace,
    apply_buoyancy_step,
    compute_divergence,
    enforce_no_slip,
    pressure_projection,
)
from boilingsim.geometry import (
    MAT_AIR,
    MAT_FLUID,
    MAT_POT_WALL,
    Grid,
    build_pot_geometry,
)


# ---------------------------------------------------------------------------
# Helpers: small synthetic grids for unit tests
# ---------------------------------------------------------------------------


def _make_all_fluid_grid(
    nx: int, ny: int, nz: int, dx: float = 0.002, device: str = "cuda:0"
) -> Grid:
    """Build a Grid where every cell is MAT_FLUID (no pot walls)."""
    mat_np = np.full((nx, ny, nz), MAT_FLUID, dtype=np.int32)
    T_np = np.full((nx, ny, nz), 293.15, dtype=np.float32)
    origin = (0.0, 0.0, 0.0)
    return Grid(
        nx=nx, ny=ny, nz=nz, dx=dx, origin=origin,
        pot_sdf=wp.zeros((nx, ny, nz), dtype=float, device=device),
        water_alpha=wp.zeros((nx, ny, nz), dtype=float, device=device),
        T=wp.array(T_np, dtype=float, device=device),
        p=wp.zeros((nx, ny, nz), dtype=float, device=device),
        mat=wp.array(mat_np, dtype=int, device=device),
        ux=wp.zeros((nx + 1, ny, nz), dtype=float, device=device),
        uy=wp.zeros((nx, ny + 1, nz), dtype=float, device=device),
        uz=wp.zeros((nx, ny, nz + 1), dtype=float, device=device),
    )


def _make_walled_grid(
    nx: int, ny: int, nz: int, dx: float = 0.002, device: str = "cuda:0"
) -> Grid:
    """Fluid interior surrounded by a one-cell MAT_POT_WALL shell."""
    mat_np = np.full((nx, ny, nz), MAT_POT_WALL, dtype=np.int32)
    mat_np[1:-1, 1:-1, 1:-1] = MAT_FLUID
    T_np = np.full((nx, ny, nz), 293.15, dtype=np.float32)
    origin = (0.0, 0.0, 0.0)
    return Grid(
        nx=nx, ny=ny, nz=nz, dx=dx, origin=origin,
        pot_sdf=wp.zeros((nx, ny, nz), dtype=float, device=device),
        water_alpha=wp.zeros((nx, ny, nz), dtype=float, device=device),
        T=wp.array(T_np, dtype=float, device=device),
        p=wp.zeros((nx, ny, nz), dtype=float, device=device),
        mat=wp.array(mat_np, dtype=int, device=device),
        ux=wp.zeros((nx + 1, ny, nz), dtype=float, device=device),
        uy=wp.zeros((nx, ny + 1, nz), dtype=float, device=device),
        uz=wp.zeros((nx, ny, nz + 1), dtype=float, device=device),
    )


def _div_u_max(grid: Grid, ws) -> float:
    """Return max |∇·u| in fluid cells."""
    nx, ny, nz = grid.shape
    wp.launch(
        compute_divergence,
        dim=(nx, ny, nz),
        inputs=[ws.div_u, grid.ux, grid.uy, grid.uz, grid.mat, grid.dx, MAT_FLUID],
    )
    wp.synchronize()
    return float(abs(ws.div_u.numpy()).max())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fluid_module_imports():
    from boilingsim import fluid  # noqa: F401


def test_enforce_no_slip_zeros_boundary_faces():
    """A walled grid with a random interior velocity: no-slip leaves zero at all
    faces touching a wall cell."""
    nx, ny, nz = 8, 8, 8
    grid = _make_walled_grid(nx, ny, nz)

    # Inject random velocities everywhere.
    grid.ux.assign(np.random.default_rng(0).standard_normal(grid.ux.shape).astype(np.float32))
    grid.uy.assign(np.random.default_rng(1).standard_normal(grid.uy.shape).astype(np.float32))
    grid.uz.assign(np.random.default_rng(2).standard_normal(grid.uz.shape).astype(np.float32))

    enforce_no_slip(grid)

    mat_np = grid.mat.numpy()
    ux_np = grid.ux.numpy()
    # ux[i,j,k] is between cells (i-1,j,k) and (i,j,k).
    # It must be zero if either side is not MAT_FLUID.
    for i in range(1, nx):
        for j in range(ny):
            for k in range(nz):
                if mat_np[i - 1, j, k] != MAT_FLUID or mat_np[i, j, k] != MAT_FLUID:
                    assert ux_np[i, j, k] == 0.0, f"ux[{i},{j},{k}] should be 0"


def test_divergence_free_after_projection():
    """Random velocity field → project → max |∇·u| in fluid drops by orders of
    magnitude and falls below the tolerance."""
    nx, ny, nz = 16, 16, 16
    grid = _make_walled_grid(nx, ny, nz)
    ws = allocate_fluid_workspace(grid)

    rng = np.random.default_rng(42)
    grid.ux.assign(rng.standard_normal(grid.ux.shape).astype(np.float32) * 0.1)
    grid.uy.assign(rng.standard_normal(grid.uy.shape).astype(np.float32) * 0.1)
    grid.uz.assign(rng.standard_normal(grid.uz.shape).astype(np.float32) * 0.1)

    enforce_no_slip(grid)
    before = _div_u_max(grid, ws)

    cfg = ScenarioConfig()
    # Pump up iterations to confirm full convergence on this tiny grid.
    # Jacobi needs ~1000 sweeps for a 16³ grid to reach 1e-4 residual.
    cfg.solver.pressure_max_iter = 1500
    max_div = pressure_projection(grid, ws, cfg, dt=0.01, rho=997.0)

    assert before > 1.0, f"initial div too small to be a real test ({before})"
    assert max_div < 1e-3, f"max |div| after projection = {max_div:.3e} (> 1e-3)"
    assert max_div < before * 1e-3, (
        f"projection should reduce div by at least 3 orders of magnitude "
        f"(before={before:.3e}, after={max_div:.3e})"
    )


def test_advection_bounded_total_temperature_drift():
    """SL advection is not strictly conservative, but a div-free velocity
    field should keep Σ T drift small (a few %) over many steps.
    """
    nx, ny, nz = 12, 12, 12
    grid = _make_walled_grid(nx, ny, nz)  # boundary walls prevent flux leak
    ws = allocate_fluid_workspace(grid)

    # Non-uniform initial T (cosine bump in x) over fluid cells.
    T_np = np.full((nx, ny, nz), 293.15, dtype=np.float32)
    for i in range(1, nx - 1):
        T_np[i, :, :] = 293.15 + 30.0 * math.cos(math.pi * (i - nx / 2) / nx)
    grid.T.assign(T_np)

    # Seed a random velocity and project to div-free.
    rng = np.random.default_rng(0)
    grid.ux.assign(rng.standard_normal(grid.ux.shape).astype(np.float32) * 0.05)
    grid.uy.assign(rng.standard_normal(grid.uy.shape).astype(np.float32) * 0.05)
    grid.uz.assign(rng.standard_normal(grid.uz.shape).astype(np.float32) * 0.05)
    enforce_no_slip(grid)
    cfg = ScenarioConfig()
    cfg.solver.pressure_max_iter = 1500
    pressure_projection(grid, ws, cfg, dt=0.01, rho=997.0)

    mat_np = grid.mat.numpy()
    fluid_mask = mat_np == MAT_FLUID

    T_before = grid.T.numpy()
    sum_T_before = float(T_before[fluid_mask].sum())

    for _ in range(20):
        advect_all(grid, ws, dt=0.01)

    T_after = grid.T.numpy()
    sum_T_after = float(T_after[fluid_mask].sum())
    rel_drift = abs(sum_T_after - sum_T_before) / abs(sum_T_before)
    # Semi-Lagrangian is non-conservative but with div-free flow the drift
    # should be within a few percent over 20 steps.
    assert rel_drift < 0.02, (
        f"Sum T drift {rel_drift:.3e} (before={sum_T_before:.3f}, after={sum_T_after:.3f})"
    )


def test_buoyancy_pushes_warm_cells_upward():
    """Zero velocity; warmer T near the bottom → after one buoyancy step,
    vertical face velocities there are positive."""
    nx, ny, nz = 10, 10, 10
    grid = _make_all_fluid_grid(nx, ny, nz)

    T_ref = 293.15
    T_np = np.full((nx, ny, nz), T_ref, dtype=np.float32)
    T_np[:, :, :3] = T_ref + 10.0  # bottom 3 layers are 10 K warmer
    grid.T.assign(T_np)

    # One buoyancy step with a strong β for easy measurement.
    apply_buoyancy_step(grid, ScenarioConfig(), dt=0.1, beta=2.07e-4, T_ref_k=T_ref)

    uz_np = grid.uz.numpy()
    # Check z-faces in the warm region (k=1,2,3)
    warm_faces = uz_np[:, :, 1:4]
    assert (warm_faces > 0).all(), (
        f"some warm-region z-faces are not positive: min = {warm_faces.min()}"
    )
    # Expected magnitude: dt · g · β · ΔT = 0.1 · 9.81 · 2.07e-4 · 10 = 2.03e-3 m/s
    expected = 0.1 * 9.81 * 2.07e-4 * 10.0
    assert abs(warm_faces.mean() - expected) < 0.2 * expected, (
        f"buoyancy magnitude off: got {warm_faces.mean():.3e}, expected {expected:.3e}"
    )


def test_hydrostatic_balance_closed_box():
    """Isothermal closed box: after projection, uz should stay zero (no buoyancy).

    Verifies that the projection + no-slip combination doesn't manufacture
    spurious motion when there's no forcing.
    """
    nx, ny, nz = 12, 12, 12
    grid = _make_walled_grid(nx, ny, nz)
    ws = allocate_fluid_workspace(grid)

    # Zero velocity, uniform T; one buoyancy step keeps uz = 0.
    apply_buoyancy_step(grid, ScenarioConfig(), dt=0.01, beta=2.07e-4, T_ref_k=293.15)
    enforce_no_slip(grid)
    max_div = pressure_projection(grid, ws, ScenarioConfig(), dt=0.01, rho=997.0)

    assert max_div < 1e-6
    u_max = max(abs(grid.ux.numpy()).max(), abs(grid.uy.numpy()).max(), abs(grid.uz.numpy()).max())
    assert u_max < 1e-6, f"spurious velocity after no-forcing step: {u_max}"


def test_real_pot_projection_reduces_divergence():
    """Sanity check on the full pot geometry: seed random velocity, project,
    divergence drops substantially. This is the 208³ production grid —
    default cfg 200 Jacobi iterations gives partial convergence which is all
    we need for a single-step sanity check.
    """
    cfg = ScenarioConfig()
    grid = build_pot_geometry(cfg)
    ws = allocate_fluid_workspace(grid)

    rng = np.random.default_rng(11)
    grid.ux.assign(rng.standard_normal(grid.ux.shape).astype(np.float32) * 0.01)
    grid.uy.assign(rng.standard_normal(grid.uy.shape).astype(np.float32) * 0.01)
    grid.uz.assign(rng.standard_normal(grid.uz.shape).astype(np.float32) * 0.01)

    enforce_no_slip(grid)
    before = _div_u_max(grid, ws)

    max_div = pressure_projection(grid, ws, cfg, dt=0.01, rho=997.0)
    # A 208³ grid needs many more Jacobi iterations for full convergence; with
    # the default 200, we only get partial divergence reduction. The full
    # pipeline in Milestone D will bump iteration counts or switch to CG.
    assert max_div < before, "projection made divergence worse"
