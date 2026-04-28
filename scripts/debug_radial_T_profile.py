"""Plot radial T profile through the pot wall + water, at a fixed z,
to verify that conjugate-interface heat transfer looks sane.

A correctly-behaving sim should show:
  * Near-isothermal copper wall (α high → small ΔT across 3 mm of wall)
  * Noticeable gradient in steel wall (α lower → a few K across the wall)
  * Thin thermal boundary layer in the water near the wall (water α is low,
    so a ~1-2 mm BL carries the temperature drop from wall to bulk)

If the copper wall shows > ~10 K across its thickness at a sane heat flux,
that would indicate the harmonic-mean k_face is under-predicting heat
transfer at the strong-contrast (copper ↔ water) interface.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boilingsim.config import load_scenario  # noqa: E402


def plot_radial_at_t(
    h5_path: pathlib.Path,
    config_path: pathlib.Path,
    target_t_s: float,
    out_path: pathlib.Path,
) -> None:
    cfg = load_scenario(config_path)
    r_outer = cfg.pot.diameter_m / 2
    r_inner = r_outer - cfg.pot.wall_thickness_m

    with h5py.File(h5_path, "r") as f:
        snap_t = np.asarray(f["snapshots/t"])
        snap_T = np.asarray(f["snapshots/T"])
        nx = int(f["meta"].attrs["nx"])
        ny = int(f["meta"].attrs["ny"])
        nz = int(f["meta"].attrs["nz"])
        dx = float(f["meta"].attrs["dx_m"])
        material = str(f["meta"].attrs["pot_material"])

    # Pick the snapshot closest to target_t_s
    idx = int(np.argmin(np.abs(snap_t - target_t_s)))
    actual_t = float(snap_t[idx])
    T_field = snap_T[idx]  # shape (nx, ny, nz) in Kelvin

    # Reconstruct the origin (same formula as compute_grid_dims).
    # Grid is centred on the pot axis in x, y; z starts from a small pad.
    origin_x = -(nx * dx) / 2
    origin_y = -(ny * dx) / 2
    origin_z = -4 * dx  # pad_cells=4 in compute_grid_dims

    # Radial profile at y=0 (j = ny // 2), z = mid-water.
    z_mid_water = cfg.pot.base_thickness_m + 0.5 * (cfg.pot.height_m - cfg.pot.base_thickness_m) * cfg.water.fill_fraction
    k_slice = int((z_mid_water - origin_z) / dx)
    j_slice = ny // 2

    # x runs from origin_x to origin_x + nx*dx; we want r from 0 to r_outer+pad.
    x_centres = origin_x + (np.arange(nx) + 0.5) * dx
    # Radial coordinate is |x| when we sample the centreline y=0.
    # Take the +x half (right of pot centre).
    i_centre = np.argmin(np.abs(x_centres))
    r_grid = x_centres[i_centre:]
    T_grid_c = T_field[i_centre:, j_slice, k_slice] - 273.15  # to °C

    # Build the figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- Left: T(r) with material regions shaded ----
    axes[0].plot(r_grid * 1000, T_grid_c, "o-", markersize=3, color="tab:red")
    axes[0].axvspan(0, r_inner * 1000, color="lightblue", alpha=0.3, label="water")
    axes[0].axvspan(r_inner * 1000, r_outer * 1000, color="lightgray", alpha=0.5, label=f"{material} wall")
    axes[0].axvspan(r_outer * 1000, r_grid.max() * 1000, color="wheat", alpha=0.3, label="air")
    axes[0].axvline(r_inner * 1000, color="k", linestyle=":", alpha=0.6)
    axes[0].axvline(r_outer * 1000, color="k", linestyle=":", alpha=0.6)
    axes[0].set_xlabel("radial distance r [mm]")
    axes[0].set_ylabel("T [°C]")
    axes[0].set_title(
        f"{material} @ t={actual_t:.1f}s, z={z_mid_water*1000:.1f}mm (mid-water)"
    )
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    # Zoom on the interface region (water-wall-air transition)
    mask = (r_grid * 1000 > (r_inner * 1000 - 8)) & (r_grid * 1000 < (r_outer * 1000 + 4))
    axes[1].plot(r_grid[mask] * 1000, T_grid_c[mask], "o-", markersize=4, color="tab:red")
    axes[1].axvline(r_inner * 1000, color="k", linestyle=":", alpha=0.6, label=f"r_inner = {r_inner*1000:.1f} mm")
    axes[1].axvline(r_outer * 1000, color="k", linestyle=":", alpha=0.6, label=f"r_outer = {r_outer*1000:.1f} mm")
    axes[1].axvspan(r_inner * 1000, r_outer * 1000, color="lightgray", alpha=0.5)
    axes[1].set_xlabel("radial distance r [mm]")
    axes[1].set_ylabel("T [°C]")
    axes[1].set_title("interface zoom: water BL → wall → air")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    # Print the key numbers.
    # Cell-center grid alignment: r_grid is sorted ascending. The first cell
    # with center >= r_inner is the inner-wall cell; the last cell with
    # center < r_outer is the outer-wall cell. We deliberately do NOT use
    # argmin-to-r_outer here — when wall_thickness ≈ dx the closest cell
    # ties between the last wall cell and the first air cell, and floating-
    # point rounding can land in air (T = T_ambient), producing a spurious
    # "ΔT across wall = T_wall − T_ambient" instead of the actual conduction
    # drop. searchsorted picks the wall cell unambiguously.
    i_inner = int(np.searchsorted(r_grid, r_inner, side="left"))
    i_outer = int(np.searchsorted(r_grid, r_outer, side="left")) - 1
    if i_outer < i_inner:
        # Wall is unresolved at this dx (< 1 cell). Collapse to a single cell
        # so the report still prints something sensible.
        i_outer = i_inner
    i_mid = (i_inner + i_outer) // 2
    n_wall_cells = i_outer - i_inner + 1
    print(f"\n=== {material} radial profile @ t={actual_t:.1f}s ===")
    print(f"  wall resolution: {n_wall_cells} cell{'s' if n_wall_cells != 1 else ''} "
          f"across {(r_outer-r_inner)*1000:.1f}mm wall (dx={dx*1000:.1f}mm)")
    print(f"  T at r=0 (water core):           {T_grid_c[0]:.2f} °C")
    print(f"  T at r=r_inner-dx (water BL):    {T_grid_c[max(i_inner-1, 0)]:.2f} °C")
    print(f"  T at r=r_inner (inner wall):     {T_grid_c[i_inner]:.2f} °C")
    print(f"  T at r=mid-wall:                 {T_grid_c[i_mid]:.2f} °C")
    print(f"  T at r=r_outer-dx (outer wall):  {T_grid_c[i_outer]:.2f} °C")
    print(f"  T at r=r_outer+dx (outer air):   "
          f"{T_grid_c[i_outer+1]:.2f} °C  (ambient sanity check)")
    dT_across_wall = T_grid_c[i_inner] - T_grid_c[i_outer]
    print(f"  ΔT across wall (inner-outer):    {dT_across_wall:+.2f} K  "
          f"({'small for copper OK' if abs(dT_across_wall) < 5 else 'check this'})")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-t", type=float, default=300.0,
                    help="Target simulation time to probe (will pick nearest snapshot).")
    args = ap.parse_args()

    bench_dir = ROOT / "benchmarks"
    config_dir = ROOT / "configs" / "scenarios"

    scenarios = [
        ("steel_304", "default.yaml", bench_dir / "phase2_heating_steel_304_impl.h5"),
        ("aluminum", "aluminum.yaml", bench_dir / "phase2_heating_aluminum_impl.h5"),
        ("copper", "copper.yaml", bench_dir / "phase2_heating_copper_impl.h5"),
    ]

    for material, yaml_name, h5_path in scenarios:
        if not h5_path.exists():
            print(f"skip {material}: {h5_path.name} not found")
            continue
        out_path = bench_dir / f"phase2_radial_T_{material}.png"
        plot_radial_at_t(h5_path, config_dir / yaml_name, args.target_t, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
