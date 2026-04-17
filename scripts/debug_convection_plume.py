"""Milestone C smoke test: heat a small spot at the bottom of a water-only
cube, run a short time-step loop (advection + buoyancy + thermal conduction
+ pressure projection), and save a cross-section showing a rising plume.

Not the full pot scenario — this is a clean test of whether the fluid
solver actually produces natural-convection behaviour.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

from boilingsim.config import ScenarioConfig  # noqa: E402
from boilingsim.fluid import (  # noqa: E402
    advect_all,
    allocate_fluid_workspace,
    apply_buoyancy_step,
    enforce_no_slip,
    pressure_projection,
)
from boilingsim.geometry import MAT_FLUID, MAT_POT_WALL, Grid  # noqa: E402
from boilingsim.thermal import (  # noqa: E402
    MaterialProps,
    allocate_thermal_workspace,
    compute_max_dt_conduction,
    heat_conduction_flux_x,
    heat_conduction_flux_y,
    heat_conduction_flux_z,
    apply_conduction_update,
)


def build_hot_spot_grid(nx: int, ny: int, nz: int, dx: float,
                         T_bg: float, T_hot: float, device: str = "cuda:0") -> Grid:
    """A closed walled box of water with a hot spot at the bottom centre."""
    mat_np = np.full((nx, ny, nz), MAT_POT_WALL, dtype=np.int32)
    mat_np[1:-1, 1:-1, 1:-1] = MAT_FLUID

    T_np = np.full((nx, ny, nz), T_bg, dtype=np.float32)
    # Hot spot: central 4x4 region in x-y, bottom 2 layers in z (inside fluid).
    cx, cy = nx // 2, ny // 2
    T_np[cx - 2:cx + 2, cy - 2:cy + 2, 1:3] = T_hot

    return Grid(
        nx=nx, ny=ny, nz=nz, dx=dx, origin=(0.0, 0.0, 0.0),
        pot_sdf=wp.zeros((nx, ny, nz), dtype=float, device=device),
        water_alpha=wp.zeros((nx, ny, nz), dtype=float, device=device),
        T=wp.array(T_np, dtype=float, device=device),
        p=wp.zeros((nx, ny, nz), dtype=float, device=device),
        mat=wp.array(mat_np, dtype=int, device=device),
        ux=wp.zeros((nx + 1, ny, nz), dtype=float, device=device),
        uy=wp.zeros((nx, ny + 1, nz), dtype=float, device=device),
        uz=wp.zeros((nx, ny, nz + 1), dtype=float, device=device),
    )


def conduct_pure(grid: Grid, props: MaterialProps, ws_t, dt: float):
    """Pure conduction step (no boundary sources) — thermal.py conduct_one_step
    without stove/evap; used because our test box has no real BCs."""
    nx, ny, nz = grid.shape
    dx = grid.dx
    wp.launch(heat_conduction_flux_x, dim=(nx + 1, ny, nz),
              inputs=[ws_t.flux_x, grid.T, grid.mat, props.k_wp, dx, 10.0, 2])
    wp.launch(heat_conduction_flux_y, dim=(nx, ny + 1, nz),
              inputs=[ws_t.flux_y, grid.T, grid.mat, props.k_wp, dx, 10.0, 2])
    wp.launch(heat_conduction_flux_z, dim=(nx, ny, nz + 1),
              inputs=[ws_t.flux_z, grid.T, grid.mat, props.k_wp, dx, 10.0, 2])
    wp.launch(apply_conduction_update, dim=(nx, ny, nz),
              inputs=[grid.T, ws_t.flux_x, ws_t.flux_y, ws_t.flux_z, grid.mat,
                      props.rho_wp, props.cp_wp, dx, dt])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--output", type=pathlib.Path,
                    default=ROOT / "benchmarks" / "phase2_convection_plume.png")
    args = ap.parse_args()

    nx, ny, nz = 32, 32, 48
    dx = 0.002  # 2 mm cells → box is 64×64×96 mm
    T_bg, T_hot = 293.15, 343.15  # 20°C background, 70°C hot spot

    grid = build_hot_spot_grid(nx, ny, nz, dx, T_bg, T_hot)

    cfg = ScenarioConfig()
    cfg.solver.pressure_max_iter = 600  # enough for 32³-ish grid

    props = MaterialProps.from_scenario(cfg)
    ws_fluid = allocate_fluid_workspace(grid)
    ws_thermal = allocate_thermal_workspace(grid)

    # Pick a time step: thermal CFL dominates water conduction; velocity CFL
    # will kick in once plume starts moving.
    dt_thermal = 0.5 * compute_max_dt_conduction(props, dx)
    dt = min(dt_thermal, 0.1)
    beta = 2.07e-4  # water near 25°C
    print(f"step dt = {dt*1000:.2f} ms; running {args.steps} steps "
          f"(t_final = {args.steps*dt:.2f} s)")

    for step in range(args.steps):
        advect_all(grid, ws_fluid, dt)
        apply_buoyancy_step(grid, cfg, dt, beta=beta, T_ref_k=T_bg)
        conduct_pure(grid, props, ws_thermal, dt)
        enforce_no_slip(grid)
        pressure_projection(grid, ws_fluid, cfg, dt, rho=997.0)
        enforce_no_slip(grid)

        if step % 10 == 0:
            wp.synchronize()
            u_max = max(abs(grid.ux.numpy()).max(),
                        abs(grid.uy.numpy()).max(),
                        abs(grid.uz.numpy()).max())
            T_np = grid.T.numpy()
            print(f"  step {step:4d}  t={step*dt:.3f}s  |u|_max={u_max*1000:.2f} mm/s  "
                  f"T range=[{T_np.min()-273.15:.1f}, {T_np.max()-273.15:.1f}] °C")

    wp.synchronize()
    T_np = grid.T.numpy()
    ux_np = grid.ux.numpy()
    uy_np = grid.uy.numpy()
    uz_np = grid.uz.numpy()

    # Cell-centred velocity magnitude for visualization
    ux_cc = 0.5 * (ux_np[:-1, :, :] + ux_np[1:, :, :])
    uy_cc = 0.5 * (uy_np[:, :-1, :] + uy_np[:, 1:, :])
    uz_cc = 0.5 * (uz_np[:, :, :-1] + uz_np[:, :, 1:])
    u_mag = np.sqrt(ux_cc**2 + uy_cc**2 + uz_cc**2)

    # xz-slice through y = ny//2
    j = ny // 2
    T_slice = (T_np[:, j, :] - 273.15).T
    u_slice = u_mag[:, j, :].T
    uz_slice = uz_cc[:, j, :].T

    extent = (0, nx * dx, 0, nz * dx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(T_slice, origin="lower", extent=extent, cmap="inferno",
                         vmin=T_bg - 273.15, vmax=T_hot - 273.15, aspect="equal")
    plt.colorbar(im0, ax=axes[0], label="T [°C]")
    axes[0].set_title("Temperature  (y=mid slice)")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("z [m]")

    im1 = axes[1].imshow(u_slice * 1000, origin="lower", extent=extent,
                         cmap="viridis", aspect="equal")
    plt.colorbar(im1, ax=axes[1], label="|u| [mm/s]")
    axes[1].set_title("Velocity magnitude")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("z [m]")

    vm = np.abs(uz_slice).max() * 1000 + 1e-9
    im2 = axes[2].imshow(uz_slice * 1000, origin="lower", extent=extent,
                         cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="equal")
    plt.colorbar(im2, ax=axes[2], label="u_z [mm/s]")
    axes[2].set_title("Vertical velocity (red=up)")
    axes[2].set_xlabel("x [m]"); axes[2].set_ylabel("z [m]")

    fig.suptitle(
        f"Convection plume smoke test — t={args.steps*dt:.2f}s, "
        f"64×64×96 mm box, hot spot at base"
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
