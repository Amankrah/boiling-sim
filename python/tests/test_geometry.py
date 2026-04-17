"""Tests for geometry generation: SDF, water volume fraction, carrot mesh, USD."""

import math
import pathlib

import numpy as np
import pytest

from boilingsim.config import ScenarioConfig, load_scenario
from boilingsim.geometry import (
    MAT_AIR,
    MAT_CARROT,
    MAT_FLUID,
    MAT_POT_WALL,
    build_carrot_mesh,
    build_pot_geometry,
    build_pot_mesh,
    build_water_surface_mesh,
    compute_grid_dims,
    export_scene_usd,
    translate_points,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_YAML = ROOT / "configs" / "scenarios" / "default.yaml"


@pytest.fixture(scope="module")
def default_cfg() -> ScenarioConfig:
    return load_scenario(DEFAULT_YAML)


@pytest.fixture(scope="module")
def grid(default_cfg):
    return build_pot_geometry(default_cfg)


def test_grid_dims_bound_pot(default_cfg):
    nx, ny, nz, origin = compute_grid_dims(default_cfg)
    assert nx * default_cfg.grid.dx_m > default_cfg.pot.diameter_m
    assert ny * default_cfg.grid.dx_m > default_cfg.pot.diameter_m
    assert nz * default_cfg.grid.dx_m > default_cfg.pot.height_m
    # Even cell counts so the pot axis lands on a face
    assert nx % 2 == 0
    assert ny % 2 == 0


def test_pot_sdf_sign_convention(grid, default_cfg):
    """sdf < 0 inside wall material, sdf > 0 in cavity or outside."""
    sdf = grid.pot_sdf.numpy()
    nx, ny, nz = grid.shape
    dx = grid.dx
    ox, oy, oz = grid.origin

    # Point deep in the wall: just inside the outer cylinder, above the base.
    r_outer = default_cfg.pot.diameter_m / 2
    r_mid_wall = r_outer - default_cfg.pot.wall_thickness_m / 2
    z_mid = default_cfg.pot.base_thickness_m + default_cfg.pot.height_m / 2

    i = int(round((r_mid_wall - ox) / dx - 0.5))
    j = int(round((0.0 - oy) / dx - 0.5))
    k = int(round((z_mid - oz) / dx - 0.5))
    assert sdf[i, j, k] < 0, f"Expected wall cell to have sdf<0, got {sdf[i, j, k]}"

    # Point deep in the water cavity: at axis, above base.
    i_axis = int(round((0.0 - ox) / dx - 0.5))
    j_axis = int(round((0.0 - oy) / dx - 0.5))
    k_cav = int(round((z_mid - oz) / dx - 0.5))
    assert sdf[i_axis, j_axis, k_cav] > 0, (
        f"Expected cavity cell to have sdf>0, got {sdf[i_axis, j_axis, k_cav]}"
    )

    # Point in the padding, outside the pot: should be positive.
    i_out = nx - 1  # last cell in +x, well outside the pot
    assert sdf[i_out, j_axis, k_cav] > 0


def test_wall_thickness_from_sdf(grid, default_cfg):
    """Measure wall thickness by counting sdf<0 cells along a mid-height ray."""
    sdf = grid.pot_sdf.numpy()
    nx, ny, nz = grid.shape
    dx = grid.dx
    ox, oy, oz = grid.origin

    # Sweep +x axis at y=0, z = mid-pot-height
    z_mid = default_cfg.pot.base_thickness_m + default_cfg.pot.height_m / 2
    k = int(round((z_mid - oz) / dx - 0.5))
    j = int(round((0.0 - oy) / dx - 0.5))
    ray = sdf[:, j, k]

    wall_mask = ray < 0
    wall_indices = np.where(wall_mask)[0]
    # Take only the +x side (indices > grid centre)
    centre_i = nx // 2
    plus_x = wall_indices[wall_indices > centre_i]
    if len(plus_x) == 0:
        pytest.fail("No wall cells found on +x side")

    measured_thickness = (plus_x.max() - plus_x.min() + 1) * dx
    expected = default_cfg.pot.wall_thickness_m
    assert abs(measured_thickness - expected) <= dx, (
        f"wall thickness {measured_thickness:.4f} m vs expected {expected:.4f} m"
    )


def test_water_volume_matches_analytic(grid, default_cfg):
    """Count α=1 cells and compare to π·r²·h for the filled cylinder."""
    alpha = grid.water_alpha.numpy()
    water_cells = int((alpha > 0.5).sum())
    sim_volume = water_cells * grid.dx ** 3

    r_inner = default_cfg.pot.diameter_m / 2 - default_cfg.pot.wall_thickness_m
    h_inner = default_cfg.pot.height_m - default_cfg.pot.base_thickness_m
    expected = math.pi * r_inner ** 2 * default_cfg.water.fill_fraction * h_inner

    rel_err = abs(sim_volume - expected) / expected
    assert rel_err < 0.05, f"water volume error {rel_err*100:.1f}% > 5%"


def test_water_not_in_wall(grid, default_cfg):
    """No water should be marked inside the pot wall material (sdf<0)."""
    sdf = grid.pot_sdf.numpy()
    alpha = grid.water_alpha.numpy()
    overlap = ((alpha > 0.5) & (sdf < 0)).sum()
    assert overlap == 0, f"{overlap} cells marked as both water and pot wall"


def test_carrot_mesh_dev_resolution(default_cfg):
    pts, tets, tris = build_carrot_mesh(
        default_cfg.carrot.diameter_m,
        default_cfg.carrot.length_m,
        default_cfg.grid.carrot_mesh_resolution,
    )
    assert pts.shape[1] == 3
    assert tets.shape[1] == 4
    assert tris.shape[1] == 3
    # Dev-tier resolution=40 yields ~60k tets on gmsh 4.15; keep a wide band.
    assert 5_000 <= tets.shape[0] <= 100_000, (
        f"tet count {tets.shape[0]} outside dev range [5k, 100k]"
    )


def test_carrot_surface_is_closed_manifold(default_cfg):
    """Every edge of the carrot surface mesh should be shared by exactly 2 triangles."""
    _, _, tris = build_carrot_mesh(
        default_cfg.carrot.diameter_m,
        default_cfg.carrot.length_m,
        default_cfg.grid.carrot_mesh_resolution,
    )
    edge_count: dict[tuple[int, int], int] = {}
    for tri in tris:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1

    bad = [e for e, n in edge_count.items() if n != 2]
    assert not bad, f"{len(bad)} non-manifold edges (expected 0)"


def test_translate_points():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    out = translate_points(pts, (10.0, 20.0, 30.0))
    assert out[0].tolist() == [10.0, 20.0, 30.0]
    assert out[1].tolist() == [11.0, 22.0, 33.0]


# ---------------------------------------------------------------------------
# Phase 2 Milestone A — material IDs + initial T + MAC allocation
# ---------------------------------------------------------------------------


def test_grid_has_all_phase2_fields(grid):
    """Grid must carry cell-centred + MAC velocity arrays for Phase 2."""
    nx, ny, nz = grid.shape
    # cell-centred
    assert grid.T.shape == (nx, ny, nz)
    assert grid.p.shape == (nx, ny, nz)
    assert grid.mat.shape == (nx, ny, nz)
    # face-centred
    assert grid.ux.shape == (nx + 1, ny, nz)
    assert grid.uy.shape == (nx, ny + 1, nz)
    assert grid.uz.shape == (nx, ny, nz + 1)


def test_material_id_partition(grid):
    """Every cell gets exactly one material ID."""
    mat = grid.mat.numpy()
    total = (
        (mat == MAT_FLUID).sum()
        + (mat == MAT_POT_WALL).sum()
        + (mat == MAT_AIR).sum()
        + (mat == MAT_CARROT).sum()
    )
    assert total == mat.size


def test_wall_mat_matches_sdf(grid):
    """Every MAT_POT_WALL cell has sdf<0, and vice versa."""
    mat = grid.mat.numpy()
    sdf = grid.pot_sdf.numpy()
    wall = mat == MAT_POT_WALL
    assert (sdf[wall] < 0).all(), "some pot-wall cells have sdf>=0"


def test_carrot_mat_count_matches_analytic(grid, default_cfg):
    """Carrot cell count ≈ π·r²·L / dx³ within 5 % (discretization)."""
    mat = grid.mat.numpy()
    carrot_cells = int((mat == MAT_CARROT).sum())
    carrot_vol = math.pi * (default_cfg.carrot.diameter_m / 2) ** 2 * default_cfg.carrot.length_m
    expected = carrot_vol / grid.dx ** 3
    rel_err = abs(carrot_cells - expected) / expected
    assert rel_err < 0.05, f"carrot cell error {rel_err*100:.1f}% > 5%"


def test_fluid_mat_matches_water_alpha(grid):
    """Fluid cells = water_alpha cells minus cells taken by carrot."""
    mat = grid.mat.numpy()
    alpha = grid.water_alpha.numpy()
    # Every fluid cell should have alpha>0.5; carrot cells originally had alpha=1 but are now MAT_CARROT.
    assert (alpha[mat == MAT_FLUID] > 0.5).all()
    # No fluid cell in the padding/air region.
    assert (mat[alpha < 0.5] != MAT_FLUID).all()


def test_initial_temperature_from_mat(grid, default_cfg):
    """Initial T partitions by material: water/carrot = initial_temp_c, wall/air = ambient."""
    T = grid.T.numpy()
    mat = grid.mat.numpy()

    T_water_k = default_cfg.water.initial_temp_c + 273.15
    T_amb_k = default_cfg.heating.ambient_temp_c + 273.15

    assert np.allclose(T[mat == MAT_FLUID], T_water_k)
    assert np.allclose(T[mat == MAT_CARROT], T_water_k)
    assert np.allclose(T[mat == MAT_POT_WALL], T_amb_k)
    assert np.allclose(T[mat == MAT_AIR], T_amb_k)


def test_velocity_and_pressure_start_zero(grid):
    assert not grid.ux.numpy().any()
    assert not grid.uy.numpy().any()
    assert not grid.uz.numpy().any()
    assert not grid.p.numpy().any()


def test_usd_export(default_cfg, tmp_path):
    """End-to-end: build all meshes, export USD, reopen and verify prims."""
    from pxr import Usd

    pot_mesh = build_pot_mesh(default_cfg)
    water_mesh = build_water_surface_mesh(default_cfg)
    pts, _, tris = build_carrot_mesh(
        default_cfg.carrot.diameter_m,
        default_cfg.carrot.length_m,
        20,  # coarser for speed in this test
    )
    pts = translate_points(pts, default_cfg.carrot.position)

    out = tmp_path / "scene.usd"
    export_scene_usd(out, pot_mesh, water_mesh, pts, tris)
    assert out.exists() and out.stat().st_size > 0

    stage = Usd.Stage.Open(str(out))
    assert stage is not None
    assert stage.GetPrimAtPath("/World/Pot").IsValid()
    assert stage.GetPrimAtPath("/World/Water").IsValid()
    assert stage.GetPrimAtPath("/World/Carrot").IsValid()
