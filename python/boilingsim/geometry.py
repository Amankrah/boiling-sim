"""Geometry generation for the boiling simulation.

Builds three things from a :class:`ScenarioConfig`:

1. A **pot signed-distance field** sampled onto a Cartesian grid.
   Convention: ``sdf < 0`` inside pot wall material, ``sdf > 0`` in the water
   cavity or outside. Downstream solvers rely on this sign.

2. A **water volume fraction** field (``alpha`` in [0, 1]) that marks liquid
   cells at t=0.

3. A **carrot tetrahedral mesh** via pygmsh/gmsh, plus its surface triangles
   wrapped as a :class:`wp.Mesh` for collision and boundary queries.

Also provides USD export of a simple visualization scene (pot shell, water
disk, carrot surface).
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .config import ScenarioConfig


wp.init()


# ---------------------------------------------------------------------------
# Material IDs  (Phase 2)
# ---------------------------------------------------------------------------

MAT_FLUID = 0
MAT_POT_WALL = 1
MAT_AIR = 2
MAT_CARROT = 3


# ---------------------------------------------------------------------------
# Warp SDF kernels  (guide §1.2, §1.3)
# ---------------------------------------------------------------------------


@wp.func
def _sdf_cylinder(p: wp.vec3, r: float, h: float) -> float:
    """Signed distance to an axis-aligned capped cylinder at the origin,
    radius ``r``, extending from z=0 to z=h."""
    d_xy = wp.length(wp.vec2(p[0], p[1])) - r
    d_z = wp.abs(p[2] - h * 0.5) - h * 0.5

    out = wp.length(wp.vec2(wp.max(d_xy, 0.0), wp.max(d_z, 0.0)))
    ins = wp.min(wp.max(d_xy, d_z), 0.0)
    return out + ins


@wp.kernel
def build_pot_sdf(
    sdf: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    r_outer: float,
    r_inner: float,
    h_outer: float,
    h_inner: float,
    base_thickness: float,
):
    """Populate ``sdf`` with the pot's signed distance.

    ``sdf[i,j,k] < 0``  — cell centre is inside the pot wall material.
    ``sdf[i,j,k] > 0``  — cell centre is in the water cavity or outside.
    """
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx

    d_outer = _sdf_cylinder(p, r_outer, h_outer)
    inner_offset = wp.vec3(0.0, 0.0, base_thickness)
    d_inner = _sdf_cylinder(p - inner_offset, r_inner, h_inner)

    # max(d_outer, -d_inner) = outer minus inner; negative = in wall material.
    sdf[i, j, k] = wp.max(d_outer, -d_inner)


@wp.kernel
def init_water_volume_fraction(
    alpha: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    r_inner: float,
    water_line_z: float,
    base_thickness: float,
):
    """Mark cells inside the pot's inner cavity and below the water line as liquid.

    Uses an analytic inner-cylinder test rather than the pot SDF, because the
    SDF is also positive outside the pot — we only want the *inner* cavity.
    """
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx

    r_xy = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    inside_cavity = r_xy < r_inner
    above_base = p[2] > base_thickness
    below_water = p[2] < water_line_z

    if inside_cavity and above_base and below_water:
        alpha[i, j, k] = 1.0
    else:
        alpha[i, j, k] = 0.0


# ---------------------------------------------------------------------------
# Material-ID voxelization  (Phase 2)
# ---------------------------------------------------------------------------


@wp.kernel
def populate_material_ids(
    mat: wp.array3d(dtype=int),
    pot_sdf: wp.array3d(dtype=float),
    water_alpha: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    carrot_centres: wp.array(dtype=wp.vec3),
    carrot_count: int,
    carrot_axis: int,    # 0=x, 1=y, 2=z
    carrot_radius: float,
    carrot_length: float,
    mat_fluid: int,
    mat_pot_wall: int,
    mat_air: int,
    mat_carrot: int,
):
    """Stamp each cell with a material ID.

    Precedence (highest first): pot wall → carrot → water → air.

    Multi-carrot: ``carrot_centres`` is a length-``carrot_count`` array
    of vec3 anchor points. For ``carrot_axis == 2`` (legacy z), the
    anchor is the *base* of the cylinder (extends +length along +z);
    for axis 0/1 (horizontal x or y), the anchor is the *centre* and
    the cylinder extends ±length/2 along that axis.
    """
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx

    # Pot wall wins if we're in solid pot material.
    if pot_sdf[i, j, k] < 0.0:
        mat[i, j, k] = mat_pot_wall
        return

    # Carrot test against each instance. count is small (<=64) and the
    # cell is exited early on the first hit, so this stays cheap.
    half_len = carrot_length * float(0.5)
    for c in range(carrot_count):
        rel = p - carrot_centres[c]
        if carrot_axis == 0:
            # Horizontal along +x: anchor is centre, axial range ±L/2.
            r_perp = wp.sqrt(rel[1] * rel[1] + rel[2] * rel[2])
            along = rel[0]
            if r_perp < carrot_radius and along > -half_len and along < half_len:
                mat[i, j, k] = mat_carrot
                return
        elif carrot_axis == 1:
            # Horizontal along +y.
            r_perp = wp.sqrt(rel[0] * rel[0] + rel[2] * rel[2])
            along = rel[1]
            if r_perp < carrot_radius and along > -half_len and along < half_len:
                mat[i, j, k] = mat_carrot
                return
        else:
            # Vertical (legacy): anchor is the base, axial range [0, L].
            r_perp = wp.sqrt(rel[0] * rel[0] + rel[1] * rel[1])
            along = rel[2]
            if r_perp < carrot_radius and along > float(0.0) and along < carrot_length:
                mat[i, j, k] = mat_carrot
                return

    # Water (liquid phase).
    if water_alpha[i, j, k] > 0.5:
        mat[i, j, k] = mat_fluid
        return

    # Everything else is air (padding, above water line, etc.).
    mat[i, j, k] = mat_air


@wp.kernel
def initialize_temperature(
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    T_water_k: float,
    T_pot_k: float,
    T_carrot_k: float,
    T_air_k: float,
    mat_fluid: int,
    mat_pot_wall: int,
    mat_air: int,
    mat_carrot: int,
):
    """Set T per cell from the material ID (temperatures in Kelvin)."""
    i, j, k = wp.tid()
    m = mat[i, j, k]
    if m == mat_fluid:
        T[i, j, k] = T_water_k
    elif m == mat_pot_wall:
        T[i, j, k] = T_pot_k
    elif m == mat_carrot:
        T[i, j, k] = T_carrot_k
    else:
        T[i, j, k] = T_air_k


# ---------------------------------------------------------------------------
# Grid wrapper
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    """MAC staggered grid holding all Phase 2 field arrays.

    Cell-centred fields (shape ``(nx, ny, nz)``):
        ``pot_sdf``, ``water_alpha``, ``T``, ``p``, ``mat``

    Face-centred velocities (MAC):
        ``ux``: ``(nx + 1, ny, nz)``
        ``uy``: ``(nx, ny + 1, nz)``
        ``uz``: ``(nx, ny, nz + 1)``

    Phase-3 optional: ``bubbles`` is a :class:`BubblePool` (defined in
    :mod:`boilingsim.boiling`). Only populated when ``cfg.boiling.enabled``.
    """

    nx: int
    ny: int
    nz: int
    dx: float
    origin: tuple[float, float, float]
    pot_sdf: wp.array
    water_alpha: wp.array
    T: wp.array
    p: wp.array
    mat: wp.array
    ux: wp.array
    uy: wp.array
    uz: wp.array
    bubbles: Any = None  # BubblePool | None — typed Any to avoid circular import
    # Phase-3 Milestone D: baseline water α (without bubbles). Evolving
    # ``water_alpha`` is reset from this each bubble step, then reduced by
    # the scatter of bubble volume fractions. Populated by build_pot_geometry.
    water_alpha_base: Any = None
    # Phase-4 Milestone A: beta-carotene concentration (mg/kg) on carrot cells.
    # Cell-centred scalar, allocated when ``cfg.nutrient.enabled``. ``C_water``
    # (Phase-4 Milestone C) will hold the water-side passive scalar tracking
    # leached mass; allocated lazily when leaching activates.
    C: Any = None
    C_water: Any = None
    # Phase-4 dual-solute extension: optional second solute, evolved
    # concurrently in the same domain, allocated when ``cfg.nutrient2.enabled``.
    C2: Any = None
    C_water2: Any = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)


def compute_grid_dims(
    cfg: ScenarioConfig, pad_cells: int = 4
) -> tuple[int, int, int, tuple[float, float, float]]:
    """Return (nx, ny, nz, origin) for a grid that bounds the pot with a pad."""
    dx = cfg.grid.dx_m
    r_outer = cfg.pot.diameter_m / 2
    h_outer = cfg.pot.height_m

    pad = pad_cells * dx
    x_extent = 2 * r_outer + 2 * pad
    y_extent = 2 * r_outer + 2 * pad
    z_extent = h_outer + 2 * pad

    nx = int(math.ceil(x_extent / dx))
    ny = int(math.ceil(y_extent / dx))
    nz = int(math.ceil(z_extent / dx))

    nx += nx % 2
    ny += ny % 2

    origin = (-nx * dx / 2, -ny * dx / 2, -pad)
    return nx, ny, nz, origin


def build_pot_geometry(cfg: ScenarioConfig, device: str = "cuda:0") -> Grid:
    """Allocate the full MAC grid and populate static fields.

    Allocates:
      * cell-centred: ``pot_sdf``, ``water_alpha``, ``T``, ``p``, ``mat``
      * face-centred MAC velocities: ``ux``, ``uy``, ``uz``

    Populates the pot SDF, water α, material IDs, and initial T at rest.
    Velocity and pressure start at zero.
    """
    nx, ny, nz, origin = compute_grid_dims(cfg)
    dx = cfg.grid.dx_m

    # Cell-centred fields
    pot_sdf = wp.zeros((nx, ny, nz), dtype=float, device=device)
    water_alpha = wp.zeros((nx, ny, nz), dtype=float, device=device)
    T = wp.zeros((nx, ny, nz), dtype=float, device=device)
    p = wp.zeros((nx, ny, nz), dtype=float, device=device)
    mat = wp.zeros((nx, ny, nz), dtype=int, device=device)

    # MAC face velocities
    ux = wp.zeros((nx + 1, ny, nz), dtype=float, device=device)
    uy = wp.zeros((nx, ny + 1, nz), dtype=float, device=device)
    uz = wp.zeros((nx, ny, nz + 1), dtype=float, device=device)

    # Geometry parameters
    r_outer = cfg.pot.diameter_m / 2
    r_inner = r_outer - cfg.pot.wall_thickness_m
    h_outer = cfg.pot.height_m
    h_inner = h_outer - cfg.pot.base_thickness_m
    base_thickness = cfg.pot.base_thickness_m
    water_height = cfg.water.fill_fraction * h_inner
    water_line_z = base_thickness + water_height

    # Auto-place N carrots from the config (count + axis + anchor).
    # count==1 returns [position] unchanged so legacy single-carrot
    # scenarios voxelize identically.
    from .config import auto_place_carrots
    centres = auto_place_carrots(
        count=cfg.carrot.count,
        axis=cfg.carrot.axis,
        anchor=cfg.carrot.position,
        diameter_m=cfg.carrot.diameter_m,
        length_m=cfg.carrot.length_m,
        inner_radius=r_inner,
        base_thickness=base_thickness,
        water_top_z=water_line_z,
    )
    carrot_centres_np = np.array(centres, dtype=np.float32)
    carrot_centres = wp.array(carrot_centres_np, dtype=wp.vec3, device=device)
    carrot_count_int = len(centres)
    carrot_axis_int = {"x": 0, "y": 1, "z": 2}[cfg.carrot.axis]
    carrot_radius = cfg.carrot.diameter_m / 2
    carrot_length = cfg.carrot.length_m

    # ---- Kernels ----
    wp.launch(
        build_pot_sdf,
        dim=(nx, ny, nz),
        inputs=[
            pot_sdf,
            wp.vec3(*origin),
            dx,
            r_outer,
            r_inner,
            h_outer,
            h_inner,
            base_thickness,
        ],
        device=device,
    )
    wp.launch(
        init_water_volume_fraction,
        dim=(nx, ny, nz),
        inputs=[
            water_alpha,
            wp.vec3(*origin),
            dx,
            r_inner,
            water_line_z,
            base_thickness,
        ],
        device=device,
    )
    wp.launch(
        populate_material_ids,
        dim=(nx, ny, nz),
        inputs=[
            mat,
            pot_sdf,
            water_alpha,
            wp.vec3(*origin),
            dx,
            carrot_centres,
            carrot_count_int,
            carrot_axis_int,
            carrot_radius,
            carrot_length,
            MAT_FLUID,
            MAT_POT_WALL,
            MAT_AIR,
            MAT_CARROT,
        ],
        device=device,
    )
    wp.launch(
        initialize_temperature,
        dim=(nx, ny, nz),
        inputs=[
            T,
            mat,
            cfg.water.initial_temp_c + 273.15,
            cfg.heating.ambient_temp_c + 273.15,
            cfg.water.initial_temp_c + 273.15,  # carrot equilibrated with water
            cfg.heating.ambient_temp_c + 273.15,
            MAT_FLUID,
            MAT_POT_WALL,
            MAT_AIR,
            MAT_CARROT,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    grid = Grid(
        nx=nx, ny=ny, nz=nz, dx=dx, origin=origin,
        pot_sdf=pot_sdf, water_alpha=water_alpha,
        T=T, p=p, mat=mat,
        ux=ux, uy=uy, uz=uz,
    )

    # Phase 3 optional: allocate bubble pool + α baseline if boiling is enabled.
    if cfg.boiling.enabled:
        # Local import to avoid circular dependency at module-load time.
        from .boiling import allocate_bubble_pool
        grid.bubbles = allocate_bubble_pool(cfg, grid, device=device)
        # Snapshot the static water mask so bubble-occupancy VOF can reset α each step.
        grid.water_alpha_base = wp.zeros((nx, ny, nz), dtype=float, device=device)
        wp.copy(grid.water_alpha_base, grid.water_alpha)

    # Phase 4 optional: allocate nutrient concentration field on carrot cells
    # and the water-side passive scalar that tracks leached mass (Milestone C).
    if cfg.nutrient.enabled:
        from .nutrient import initialize_nutrient_field
        grid.C = wp.zeros((nx, ny, nz), dtype=float, device=device)
        grid.C_water = wp.zeros((nx, ny, nz), dtype=float, device=device)
        initialize_nutrient_field(grid, cfg, device=device)

    # Phase 4 dual-solute extension: optional second solute.
    if cfg.nutrient2.enabled:
        from .nutrient import initialize_nutrient_field
        grid.C2 = wp.zeros((nx, ny, nz), dtype=float, device=device)
        grid.C_water2 = wp.zeros((nx, ny, nz), dtype=float, device=device)
        initialize_nutrient_field(
            grid, cfg, device=device,
            target_C=grid.C2,
            C0_override=cfg.nutrient2.C0_mg_per_kg,
        )

    return grid


# ---------------------------------------------------------------------------
# Carrot tet mesh  (guide §1.4)
# ---------------------------------------------------------------------------


def build_carrot_mesh(
    diameter_m: float, length_m: float, resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a tet mesh for a cylindrical carrot aligned with +z.

    The ``resolution`` knob sets the characteristic element size; larger
    values produce denser meshes.

    Returns
    -------
    points        : (N, 3) float32 vertex coordinates at the origin
    tets          : (M, 4) int32 tet→vertex indices
    surface_tris  : (K, 3) int32 surface triangle→vertex indices
    """
    import pygmsh

    radius = diameter_m / 2
    with pygmsh.occ.Geometry() as geom:
        geom.add_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, length_m], radius)
        geom.characteristic_length_max = length_m / resolution
        mesh = geom.generate_mesh(dim=3)

    points = np.asarray(mesh.points, dtype=np.float32)
    tets = np.asarray(
        mesh.cells_dict.get("tetra", np.zeros((0, 4), dtype=np.int32)),
        dtype=np.int32,
    )
    tris = np.asarray(
        mesh.cells_dict.get("triangle", np.zeros((0, 3), dtype=np.int32)),
        dtype=np.int32,
    )
    return points, tets, tris


def translate_points(points: np.ndarray, offset: tuple[float, float, float]) -> np.ndarray:
    """Return a translated copy of ``points``."""
    return points + np.asarray(offset, dtype=points.dtype)


def make_carrot_warp_mesh(
    points: np.ndarray, surface_tris: np.ndarray, device: str = "cuda:0"
) -> wp.Mesh:
    """Wrap the carrot surface as a :class:`wp.Mesh` for collision queries."""
    return wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(surface_tris.flatten(), dtype=int, device=device),
    )


# ---------------------------------------------------------------------------
# Visualization meshes for USD export  (guide §1.5)
# ---------------------------------------------------------------------------


@dataclass
class TriMesh:
    """Minimal triangle-mesh container for USD export."""
    points: np.ndarray  # (N, 3) float
    faces: np.ndarray   # (M, 3) int


def _ring(radius: float, z: float, n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])


def build_pot_mesh(cfg: ScenarioConfig, n_segments: int = 64) -> TriMesh:
    """Build a hollow-cylinder pot shell as concentric annular strips.

    Assembled by hand as outer shell + inner shell + top rim + base — no CSG.
    """
    r_outer = cfg.pot.diameter_m / 2
    r_inner = r_outer - cfg.pot.wall_thickness_m
    h = cfg.pot.height_m
    bt = cfg.pot.base_thickness_m

    outer_bot = _ring(r_outer, 0.0, n_segments)
    outer_top = _ring(r_outer, h, n_segments)
    inner_bot = _ring(r_inner, bt, n_segments)
    inner_top = _ring(r_inner, h, n_segments)

    points = np.vstack([outer_bot, outer_top, inner_bot, inner_top]).astype(np.float32)
    N = n_segments
    OB, OT, IB, IT = 0, N, 2 * N, 3 * N

    faces: list[list[int]] = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # Outer shell (outward-facing)
        faces.append([OB + i, OB + j, OT + i])
        faces.append([OT + i, OB + j, OT + j])
        # Inner shell (inward-facing)
        faces.append([IB + i, IT + i, IB + j])
        faces.append([IT + i, IT + j, IB + j])
        # Top rim (connects outer_top to inner_top)
        faces.append([OT + i, OT + j, IT + i])
        faces.append([IT + i, OT + j, IT + j])
        # Inner floor of base (connects inner_bot to outer_bot at z=bt)
        # — omitted; the visible base is the underside fan below.

    centre_idx = len(points)
    points = np.vstack([points, np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])
    for i in range(n_segments):
        j = (i + 1) % n_segments
        faces.append([centre_idx, OB + j, OB + i])  # underside fan

    return TriMesh(points=points, faces=np.asarray(faces, dtype=np.int32))


def build_water_surface_mesh(cfg: ScenarioConfig, n_segments: int = 64) -> TriMesh:
    """A flat disk at the water line, radius = pot inner radius."""
    r_inner = cfg.pot.diameter_m / 2 - cfg.pot.wall_thickness_m
    h_inner = cfg.pot.height_m - cfg.pot.base_thickness_m
    z = cfg.pot.base_thickness_m + cfg.water.fill_fraction * h_inner

    theta = np.linspace(0.0, 2 * np.pi, n_segments, endpoint=False)
    ring = np.column_stack(
        [r_inner * np.cos(theta), r_inner * np.sin(theta), np.full(n_segments, z)]
    )
    centre = np.array([[0.0, 0.0, z]], dtype=np.float32)
    points = np.vstack([ring, centre]).astype(np.float32)

    centre_idx = n_segments
    faces = np.array(
        [[centre_idx, i, (i + 1) % n_segments] for i in range(n_segments)],
        dtype=np.int32,
    )
    return TriMesh(points=points, faces=faces)


# ---------------------------------------------------------------------------
# USD export  (guide §1.5)
# ---------------------------------------------------------------------------


def export_scene_usd(
    path: str | pathlib.Path,
    pot_mesh: TriMesh,
    water_mesh: TriMesh,
    carrot_points: np.ndarray,
    carrot_surface_tris: np.ndarray,
) -> None:
    """Write pot, water, and carrot meshes to a USD stage (z-up, meters)."""
    from pxr import Usd, UsdGeom

    path = pathlib.Path(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    def _add(name: str, points: np.ndarray, faces: np.ndarray) -> None:
        prim = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        prim.CreatePointsAttr([tuple(map(float, p)) for p in points])
        prim.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        prim.CreateFaceVertexCountsAttr([3] * len(faces))

    _add("Pot", pot_mesh.points, pot_mesh.faces)
    _add("Water", water_mesh.points, water_mesh.faces)
    _add("Carrot", carrot_points, carrot_surface_tris)

    stage.GetRootLayer().Save()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def estimate_vram_mb(nx: int, ny: int, nz: int, n_fields: int = 8) -> float:
    """Rough VRAM estimate in MB for ``n_fields`` float32 fields on the grid."""
    return (nx * ny * nz * 4 * n_fields) / (1024 * 1024)
