"""Scenario CLI: build a USD scene from a YAML config.

Usage:
    python -m boilingsim.scenario --config configs/scenarios/default.yaml --output scene.usd
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

from .config import load_scenario
from .geometry import (
    MAT_AIR,
    MAT_CARROT,
    MAT_FLUID,
    MAT_POT_WALL,
    build_carrot_mesh,
    build_pot_geometry,
    build_pot_mesh,
    build_water_surface_mesh,
    compute_grid_dims,
    estimate_vram_mb,
    export_scene_usd,
    translate_points,
)


def _run_steady_heat(grid, cfg, duration_s: float, device: str,
                      with_bubbles: bool = False) -> None:
    """Milestone B exit check: conduction-only heating, report mean T history.

    Phase 3 Milestone A: pass ``with_bubbles=True`` to also run the nucleation
    detection kernel every step and print the bubble count alongside T.
    """
    from .geometry import MAT_FLUID, MAT_POT_WALL
    from .thermal import (
        MaterialProps,
        allocate_thermal_workspace,
        compute_max_dt_conduction,
        conduct_one_step,
    )
    import warp as wp

    props = MaterialProps.from_scenario(cfg, device=device)
    ws = allocate_thermal_workspace(grid, device=device)
    dt = cfg.solver.cfl_safety_factor * compute_max_dt_conduction(props, grid.dx)
    dt = min(dt, cfg.solver.max_dt_s)
    n_steps = int(duration_s / dt)

    print(f"\n  --- steady-heat run: {duration_s:.0f}s sim @ dt={dt*1000:.3f}ms = {n_steps} steps ---")
    if with_bubbles:
        if grid.bubbles is None:
            print("    WARNING: --with-bubbles requested but boiling.enabled=False in config; no pool allocated.")
            with_bubbles = False
        else:
            print(f"    Phase-3 nucleation enabled (pool = {grid.bubbles.max_bubbles:,} slots)")
    mat_np = grid.mat.numpy()
    water_mask = mat_np == MAT_FLUID
    wall_mask = mat_np == MAT_POT_WALL

    if with_bubbles:
        from .boiling import step_nucleation

    t0 = time.perf_counter()
    checkpoint_every = max(1, n_steps // 10)
    for step in range(n_steps):
        conduct_one_step(grid, props, ws, cfg, dt, device=device)
        if with_bubbles:
            step_nucleation(grid, grid.bubbles, cfg, dt,
                            sim_time=step * dt, step_count=step, device=device)
        if step % checkpoint_every == 0 or step == n_steps - 1:
            wp.synchronize()
            T_np = grid.T.numpy()
            t_sim = (step + 1) * dt
            if with_bubbles:
                n_bubbles = grid.bubbles.count_active()
                print(
                    f"    t={t_sim:6.2f}s  "
                    f"T_water mean={T_np[water_mask].mean()-273.15:6.2f}C  "
                    f"T_wall max={T_np[wall_mask].max()-273.15:6.2f}C  "
                    f"bubbles={n_bubbles:,}"
                )
            else:
                print(
                    f"    t={t_sim:6.2f}s  "
                    f"T_water mean={T_np[water_mask].mean()-273.15:6.2f}C  "
                    f"T_wall max={T_np[wall_mask].max()-273.15:6.2f}C"
                )
    wall = time.perf_counter() - t0
    print(f"  wall time: {wall:.2f}s  ({wall/duration_s:.3f}s/sim-s)")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a USD scene (pot + water + carrot) from a scenario YAML."
    )
    p.add_argument("--config", type=pathlib.Path, required=True, help="Path to scenario YAML")
    p.add_argument("--output", type=pathlib.Path, required=True, help="Output .usd file")
    p.add_argument("--device", default="cuda:0", help="Warp device (default: cuda:0)")
    p.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip SDF grid generation; only export visualization meshes.",
    )
    p.add_argument(
        "--steady-heat",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Run conduction-only heating for N seconds after building the scene "
             "and report mean water / wall temperatures. Milestone B exit check.",
    )
    p.add_argument(
        "--with-bubbles",
        action="store_true",
        help="Enable Phase-3 bubble pool + nucleation detection during --steady-heat. "
             "Requires boiling.enabled=true in the YAML (or set here).",
    )
    p.add_argument(
        "--with-nutrient",
        action="store_true",
        help="Enable Phase-4 carrot beta-carotene physics (Arrhenius degradation, "
             "in-carrot diffusion, Sherwood leaching, water-side scalar advection). "
             "Equivalent to setting nutrient.enabled=true in the YAML.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = load_scenario(args.config)
    # CLI flag can force boiling on (overrides YAML)
    if args.with_bubbles and not cfg.boiling.enabled:
        cfg.boiling.enabled = True
    if args.with_nutrient and not cfg.nutrient.enabled:
        cfg.nutrient.enabled = True
    print(f"Loaded scenario from {args.config}")
    print(f"  pot:    {cfg.pot.material}  D={cfg.pot.diameter_m*100:.1f}cm  H={cfg.pot.height_m*100:.1f}cm")
    print(f"  water:  fill={cfg.water.fill_fraction:.0%}  T0={cfg.water.initial_temp_c}°C")
    print(f"  carrot: D={cfg.carrot.diameter_m*1000:.1f}mm  L={cfg.carrot.length_m*1000:.1f}mm")

    nx, ny, nz, origin = compute_grid_dims(cfg)
    cells = nx * ny * nz
    print(f"  grid:   {nx} x {ny} x {nz} = {cells:,} cells @ dx={cfg.grid.dx_m*1000:.2f}mm")
    print(f"          origin={origin}, est VRAM (8 fields): {estimate_vram_mb(nx, ny, nz):.1f} MB")

    grid = None
    if not args.skip_grid:
        t0 = time.perf_counter()
        grid = build_pot_geometry(cfg, device=args.device)
        dt = time.perf_counter() - t0
        mat = grid.mat.numpy()
        T = grid.T.numpy()
        fluid = int((mat == MAT_FLUID).sum())
        wall = int((mat == MAT_POT_WALL).sum())
        carrot = int((mat == MAT_CARROT).sum())
        air = int((mat == MAT_AIR).sum())
        print(f"  grid build:  {dt:.2f}s")
        print(f"  materials:   fluid={fluid:,}  wall={wall:,}  carrot={carrot:,}  air={air:,}")
        print(f"  T range:     [{T.min()-273.15:.1f}, {T.max()-273.15:.1f}] °C")
        print(f"  solver:      CFL×{cfg.solver.cfl_safety_factor}  max_dt={cfg.solver.max_dt_s}s  "
              f"h_conv={cfg.solver.h_conv_outer_w_per_m2_k} W/m²K")

    if args.steady_heat is not None and grid is not None:
        _run_steady_heat(grid, cfg, duration_s=args.steady_heat,
                         device=args.device, with_bubbles=args.with_bubbles)

    # Visualization meshes
    t0 = time.perf_counter()
    pot_mesh = build_pot_mesh(cfg)
    water_mesh = build_water_surface_mesh(cfg)
    print(f"  pot mesh:    {len(pot_mesh.points):,} verts, {len(pot_mesh.faces):,} tris")
    print(f"  water mesh:  {len(water_mesh.points):,} verts, {len(water_mesh.faces):,} tris")

    print(f"  building carrot tet mesh (resolution={cfg.grid.carrot_mesh_resolution})...")
    c_pts, c_tets, c_tris = build_carrot_mesh(
        cfg.carrot.diameter_m,
        cfg.carrot.length_m,
        cfg.grid.carrot_mesh_resolution,
    )
    # Multi-carrot: replicate the canonical mesh at each auto-placement
    # centre. For axis="z" the centre is the cylinder base (legacy);
    # for axis="x"/"y" it's the centroid -- but ``build_carrot_mesh``
    # generates a vertical cylinder anchored at the origin extending
    # +z, so we must rotate per-instance for horizontal axes.
    from .config import auto_place_carrots
    inner_radius = cfg.pot.diameter_m / 2 - cfg.pot.wall_thickness_m
    water_height = cfg.water.fill_fraction * (cfg.pot.height_m - cfg.pot.base_thickness_m)
    water_top_z = cfg.pot.base_thickness_m + water_height
    centres = auto_place_carrots(
        count=cfg.carrot.count,
        axis=cfg.carrot.axis,
        anchor=cfg.carrot.position,
        diameter_m=cfg.carrot.diameter_m,
        length_m=cfg.carrot.length_m,
        inner_radius=inner_radius,
        base_thickness=cfg.pot.base_thickness_m,
        water_top_z=water_top_z,
    )
    c_pts_per_instance = [
        _orient_and_translate_carrot(c_pts, cfg.carrot.axis, cfg.carrot.length_m, centre)
        for centre in centres
    ]
    print(
        f"  carrot:      {len(c_pts):,} verts, {len(c_tets):,} tets, "
        f"{len(c_tris):,} surface tris × {len(centres)} instance(s)"
    )
    print(f"  mesh build:  {time.perf_counter()-t0:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_scene_usd(args.output, pot_mesh, water_mesh, c_pts_per_instance, c_tris)
    print(f"USD scene written: {args.output.resolve()}")
    return 0


def _orient_and_translate_carrot(
    canonical_pts: np.ndarray,
    axis: str,
    length_m: float,
    centre: tuple[float, float, float],
) -> np.ndarray:
    """Apply axis-aware orientation + translation to one carrot mesh.

    The canonical mesh is a vertical cylinder anchored at origin extending
    +z by ``length_m``. The runtime kernel interprets ``centre`` as:
      * axis="z" → base of the cylinder; cylinder extends [cz, cz+L].
      * axis="x" / "y" → centre of the cylinder; cylinder extends ±L/2.

    We mirror that in mesh space so the USD export matches the rasterized
    physics geometry.
    """
    pts = canonical_pts.copy()
    if axis == "z":
        # Legacy: anchor is the base; canonical mesh already extends +z
        # from origin, so just translate.
        return translate_points(pts, centre)
    # Horizontal: rotate so cylinder axis aligns with +x or +y, then
    # shift so the cylinder's midpoint sits at ``centre``.
    half = length_m / 2.0
    pts[:, 2] -= half  # centre cylinder on origin in z (range [-L/2, +L/2])
    if axis == "x":
        # Map (x, y, z) → (z, y, -x): rotate about +y by +90°.
        rotated = np.column_stack([pts[:, 2], pts[:, 1], -pts[:, 0]]).astype(pts.dtype)
    else:  # axis == "y"
        # Map (x, y, z) → (x, z, -y): rotate about +x by -90°.
        rotated = np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]]).astype(pts.dtype)
    return translate_points(rotated, centre)


if __name__ == "__main__":
    raise SystemExit(main())
