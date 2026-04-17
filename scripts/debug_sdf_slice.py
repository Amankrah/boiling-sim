"""Render a cross-section of the pot SDF for visual sanity-checking.

Produces a 3-panel PNG (xz-slice at y=0) showing:
  1. Raw SDF with zero-contour
  2. Sign mask (wall material vs cavity vs outside)
  3. Water volume fraction alpha

Usage:
    python scripts/debug_sdf_slice.py --config configs/scenarios/default.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Make the boilingsim package importable when running as a script.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boilingsim.config import load_scenario  # noqa: E402
from boilingsim.geometry import build_pot_geometry  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, default=ROOT / "benchmarks" / "phase1_pot_slice.png")
    args = p.parse_args()

    cfg = load_scenario(args.config)
    grid = build_pot_geometry(cfg)
    sdf = grid.pot_sdf.numpy()
    alpha = grid.water_alpha.numpy()
    nx, ny, nz = grid.shape
    dx = grid.dx
    ox, oy, oz = grid.origin

    j = ny // 2  # slice through y=0
    sdf_slice = sdf[:, j, :].T  # shape (nz, nx) for imshow with x horizontal, z vertical
    alpha_slice = alpha[:, j, :].T
    sign_slice = np.sign(sdf_slice)  # -1 wall, 0 boundary, +1 cavity/outside

    extent = (ox, ox + nx * dx, oz, oz + nz * dx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sdf_slice, origin="lower", extent=extent, cmap="RdBu", vmin=-0.02, vmax=0.02, aspect="equal")
    axes[0].contour(sdf_slice, levels=[0.0], colors="k", linewidths=1, extent=extent)
    axes[0].set_title("SDF  (blue<0=wall, red>0=cavity/outside)")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("z [m]")

    axes[1].imshow(sign_slice, origin="lower", extent=extent, cmap="coolwarm", aspect="equal")
    axes[1].set_title("sign(SDF)")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("z [m]")

    axes[2].imshow(alpha_slice, origin="lower", extent=extent, cmap="Blues", aspect="equal")
    axes[2].set_title("water α (1=liquid)")
    axes[2].set_xlabel("x [m]"); axes[2].set_ylabel("z [m]")

    title = (
        f"Pot cross-section @ y=0   dx={dx*1000:.2f}mm   "
        f"{cfg.pot.material}   wall={cfg.pot.wall_thickness_m*1000:.1f}mm"
    )
    fig.suptitle(title)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"Saved slice figure: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
