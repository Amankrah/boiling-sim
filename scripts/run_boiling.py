"""Run a full nucleate-boiling scenario and validate against the Rohsenow
correlation and published bubble-departure statistics (Phase 3 Milestone E).

Produces, per scenario, under ``benchmarks/``:
  - phase3_boiling_<material>.h5   : scalars + bubble snapshots (HDF5)
  - phase3_boiling_<material>.png  : three-panel summary figure
  - stdout                          : Rohsenow check and bubble-stats summary
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boilingsim.config import load_scenario  # noqa: E402
from boilingsim.pipeline import Simulation  # noqa: E402
from boilingsim.geometry import MAT_FLUID, MAT_POT_WALL  # noqa: E402


# Water properties (Phase 2/3 constants; match data/materials.json)
T_SAT_K = 373.15
RHO_L = 997.0
RHO_V = 0.598
CP_L = 4186.0
H_LV = 2.257e6
SIGMA = 0.0589
MU_L = 2.82e-4      # water dynamic viscosity at 100 C
K_L = 0.606
PR_L = MU_L * CP_L / K_L   # ~ 1.95
G = 9.81


def rohsenow_wall_superheat_from_q(q_w_per_m2: float, C_sf: float, Pr_n: float) -> float:
    """Invert the Rohsenow correlation to get delta_T_wall from q_wall.

        c_p,l * delta_T_w / h_lv = C_sf * ( q_w * sqrt(sigma/(g*drho)) / (mu_l * h_lv) )^0.33 * Pr_l^n

    Solve for delta_T_w.
    """
    drho = RHO_L - RHO_V
    capillary = math.sqrt(SIGMA / (G * drho))
    bracket = q_w_per_m2 * capillary / (MU_L * H_LV)
    rhs = C_sf * (bracket ** 0.33) * (PR_L ** Pr_n)
    return rhs * H_LV / CP_L


def rohsenow_q_from_superheat(delta_T_k: float, C_sf: float, Pr_n: float) -> float:
    """Invert the same Rohsenow correlation for q_wall given delta_T_w."""
    drho = RHO_L - RHO_V
    capillary = math.sqrt(SIGMA / (G * drho))
    # c_p,l * dT / h_lv = C_sf * X^0.33 * Pr^n, where X = q*cap/(mu*h_lv)
    # X = ( (c_p*dT/h_lv) / (C_sf * Pr^n) )^(1/0.33)
    lhs = CP_L * delta_T_k / H_LV
    X = (lhs / (C_sf * (PR_L ** Pr_n))) ** (1.0 / 0.33)
    q = X * MU_L * H_LV / capillary
    return q


def load_scalars(h5: h5py.File) -> dict:
    g = h5["scalars"]
    return {k: np.asarray(g[k]) for k in g.keys()}


def collect_bubble_radii(h5: h5py.File, last_n_snaps: int = 20) -> np.ndarray:
    """Concatenate bubble radii over the last `last_n_snaps` snapshots."""
    if "bubble_snapshots" not in h5:
        return np.array([])
    radii_ds = h5["bubble_snapshots/radii_m"]
    n_snaps = len(radii_ds)
    start = max(0, n_snaps - last_n_snaps)
    all_r = []
    for i in range(start, n_snaps):
        r = np.asarray(radii_ds[i])
        if r.size:
            all_r.append(r)
    if not all_r:
        return np.array([])
    return np.concatenate(all_r)


def plot_summary(h5_path: pathlib.Path, out_path: pathlib.Path,
                  material: str, C_sf: float, Pr_n: float,
                  q_base_w_per_m2: float) -> dict:
    """Produce the Milestone-E 3-panel summary figure and return validation stats."""
    with h5py.File(h5_path, "r") as f:
        sc = load_scalars(f)
        # Radii from the second half of the run (after transient)
        radii = collect_bubble_radii(f, last_n_snaps=10)
        t_final = float(sc["t"][-1])

    # ---- Rohsenow-based check ----
    # Rohsenow's dT_w is defined at the *fluid-contact face* (inner wall).
    # For low-k materials (steel), the heater face reads q*L/k hotter than
    # the inner face, so T_max_wall_c over-reports the boiling superheat.
    # Use T_inner_wall_mean_c -- the average temperature on pot-wall cells
    # whose +z neighbor is fluid -- as the Rohsenow-relevant metric.
    # Fall back to T_max_wall_c for older HDF5 files without this dataset.
    half = len(sc["t"]) // 2
    if "T_inner_wall_mean_c" in sc:
        inner_series = sc["T_inner_wall_mean_c"][half:]
        dT_w_meas_k = float(np.mean(inner_series[-max(len(inner_series)//4, 1):]) - 100.0)
    else:
        dT_w_meas_k = float(np.max(sc["T_max_wall_c"][half:]) - 100.0)
    dT_w_meas_k = max(dT_w_meas_k, 0.01)
    # Also capture the raw outer-face value for contrast reporting.
    dT_w_outer_k = float(np.max(sc["T_max_wall_c"][half:]) - 100.0)
    # Predicted delta_T_w from Rohsenow at our stove q"
    dT_w_rohsenow_k = rohsenow_wall_superheat_from_q(q_base_w_per_m2, C_sf, Pr_n)
    # And the inverse: Rohsenow q" predicted for the measured delta_T_w
    q_rohsenow = rohsenow_q_from_superheat(dT_w_meas_k, C_sf, Pr_n)
    q_err_pct = 100.0 * (q_rohsenow - q_base_w_per_m2) / q_base_w_per_m2

    # ---- Bubble departure diameter ----
    mean_D_mm = float(2.0 * radii.mean() * 1000.0) if radii.size else 0.0
    median_D_mm = float(2.0 * np.median(radii) * 1000.0) if radii.size else 0.0

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Temperatures over time
    axes[0].plot(sc["t"], sc["T_mean_water_c"], "-", color="tab:blue", label="water mean")
    axes[0].plot(sc["t"], sc["T_max_wall_c"], "-", color="tab:red", label="wall max")
    axes[0].axhline(100.0, ls=":", color="k", alpha=0.4, label="T_sat = 100 C")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("T [C]")
    axes[0].set_title(f"{material}: temperatures")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # (b) Bubble count and velocity
    ax2 = axes[1]
    ax2.plot(sc["t"], sc["n_active_bubbles"], "-", color="tab:green", label="n active bubbles")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("bubble count", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.grid(alpha=0.3)
    ax2_r = ax2.twinx()
    ax2_r.plot(sc["t"], sc["u_max_mps"] * 1000.0, "-", color="tab:orange", alpha=0.7, label="|u|_max")
    ax2_r.set_ylabel("|u|_max [mm/s]", color="tab:orange")
    ax2_r.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_title(f"{material}: bubble population & peak velocity")

    # (c) Bubble departure diameter histogram
    ax3 = axes[2]
    if radii.size:
        D_mm = 2.0 * radii * 1000.0
        ax3.hist(D_mm, bins=40, color="tab:purple", edgecolor="white", alpha=0.8)
        ax3.axvspan(1.5, 4.0, color="tab:green", alpha=0.15, label="published range")
        ax3.axvline(mean_D_mm, color="k", ls="--", label=f"mean = {mean_D_mm:.2f} mm")
    ax3.set_xlabel("2R = bubble diameter [mm]")
    ax3.set_ylabel("count")
    ax3.set_title(f"{material}: bubble diameter distribution")
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"Phase 3 nucleate boiling validation — {material} @ q={q_base_w_per_m2/1000:.0f} kW/m^2"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "t_final_s": t_final,
        "dT_w_measured_k": dT_w_meas_k,
        "dT_w_outer_k": dT_w_outer_k,
        "dT_w_rohsenow_k": dT_w_rohsenow_k,
        "q_rohsenow_implied_w_m2": q_rohsenow,
        "q_rohsenow_err_pct": q_err_pct,
        "mean_departure_D_mm": mean_D_mm,
        "median_departure_D_mm": median_D_mm,
        "n_radii_sampled": int(radii.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--dx-mm", type=float, default=2.0)
    ap.add_argument("--max-bubbles", type=int, default=100_000)
    ap.add_argument("--pressure-iters", type=int, default=100)
    ap.add_argument("--snapshot-every-s", type=float, default=10.0)
    ap.add_argument("--warm-start-water-c", type=float, default=95.0,
                    help="Pre-heat water to this T before the run (default: 95 C).")
    ap.add_argument("--warm-start-wall-c", type=float, default=100.0)
    ap.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "benchmarks")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_scenario(args.config)
    cfg.grid.dx_m = args.dx_mm / 1000.0
    cfg.solver.pressure_max_iter = args.pressure_iters
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = args.max_bubbles

    material = cfg.pot.material
    print(f"=== Phase 3 boiling validation: {material} ===")
    print(f"  config       : {args.config}")
    print(f"  duration     : {args.duration:.1f} s")
    print(f"  dx           : {args.dx_mm:.2f} mm")
    print(f"  max bubbles  : {args.max_bubbles:,}")
    print(f"  C_sf         : {cfg.boiling.C_sf_rohsenow}")
    print(f"  q_stove      : {cfg.heating.base_heat_flux_w_per_m2/1000:.0f} kW/m^2")
    print(f"  Pr_L         : {PR_L:.3f}")

    # --- Rohsenow reference ---
    dT_w_predict = rohsenow_wall_superheat_from_q(
        cfg.heating.base_heat_flux_w_per_m2,
        cfg.boiling.C_sf_rohsenow, cfg.boiling.Pr_n_rohsenow,
    )
    print(f"  Rohsenow predicts dT_w = {dT_w_predict:.2f} K "
          f"at q = {cfg.heating.base_heat_flux_w_per_m2/1000:.0f} kW/m^2")

    # --- Run simulation ---
    sim = Simulation(cfg, device=args.device)
    # Warm-start to the target water and wall temperatures.
    T_np = sim.grid.T.numpy()
    mat_np = sim.grid.mat.numpy()
    T_np[mat_np == MAT_FLUID] = args.warm_start_water_c + 273.15
    T_np[mat_np == MAT_POT_WALL] = args.warm_start_wall_c + 273.15
    sim.grid.T.assign(T_np)

    print(f"\n  === running {args.duration:.0f} s of boiling ===")
    out_h5 = args.out_dir / f"phase3_boiling_{material}.h5"
    t0 = time.perf_counter()
    scalars = sim.run(
        total_time_s=args.duration,
        out_path=out_h5,
        scalar_every_n_steps=20,
        snapshot_every_s=args.snapshot_every_s,
        progress_every_s=20.0,
    )
    wall = time.perf_counter() - t0
    print(f"\n  sim done: {wall:.1f} s wall, {sim.step_count} steps, "
          f"{wall/args.duration:.2f} s/sim-s")

    # --- Validation plots + summary ---
    plot_path = args.out_dir / f"phase3_boiling_{material}.png"
    stats = plot_summary(
        out_h5, plot_path, material,
        C_sf=cfg.boiling.C_sf_rohsenow,
        Pr_n=cfg.boiling.Pr_n_rohsenow,
        q_base_w_per_m2=cfg.heating.base_heat_flux_w_per_m2,
    )

    print("\n=== Validation summary ===")
    print(f"  dT_w inner (Rohsenow metric):{stats['dT_w_measured_k']:.2f} K")
    print(f"  dT_w outer (T_wall_max):     {stats['dT_w_outer_k']:.2f} K  "
          f"(outer-inner gap = {stats['dT_w_outer_k'] - stats['dT_w_measured_k']:.2f} K "
          f"from q*L/k in solid wall)")
    print(f"  Rohsenow-predicted dT_w:    {stats['dT_w_rohsenow_k']:.2f} K")
    print(f"  Implied q at measured dT_w: {stats['q_rohsenow_implied_w_m2']/1000:.2f} kW/m^2 "
          f"(error vs stove: {stats['q_rohsenow_err_pct']:+.1f}%)")
    print(f"  mean departure D:           {stats['mean_departure_D_mm']:.2f} mm "
          f"(published range 1.5 - 4.0 mm)")
    print(f"  median departure D:         {stats['median_departure_D_mm']:.2f} mm")
    print(f"  bubbles sampled in histogram: {stats['n_radii_sampled']:,}")
    print(f"  final: T_water={scalars[-1].T_mean_water_c:.2f} C, "
          f"T_wall_max={scalars[-1].T_max_wall_c:.2f} C, "
          f"bubbles={scalars[-1].n_active_bubbles:,}")
    print(f"  plot: {plot_path}")

    # --- Energy balance diagnostic ---
    A_base = math.pi * (cfg.pot.diameter_m / 2) ** 2
    P_stove = cfg.heating.base_heat_flux_w_per_m2 * A_base
    dT_w_final = stats["dT_w_measured_k"]
    print(f"\n=== Energy balance diagnostic ===")
    print(f"  P_stove  = {cfg.heating.base_heat_flux_w_per_m2/1000:.0f} kW/m^2 * "
          f"{A_base*1e4:.1f} cm^2 = {P_stove:.0f} W")
    print(f"  dT_wall  = {dT_w_final:.1f} K  (target: {stats['dT_w_rohsenow_k']:.1f} K)")
    ratio = dT_w_final / max(stats["dT_w_rohsenow_k"], 0.01)
    print(f"  dT_wall / dT_Rohsenow = {ratio:.2f}  (target: 0.7 - 1.3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
