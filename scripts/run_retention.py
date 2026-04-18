"""Phase 4 Milestone E validation driver.

Runs a warm-started boiling + nutrient simulation for a configurable
duration and carrot diameter, writes HDF5 + a three-panel summary plot,
and reports retention vs. the dev-guide's target band [80 %, 90 %]
(experimental reference ~84 %) for a 25 mm reference carrot after 600 s.

Per scenario, under ``benchmarks/``:
  - phase4_retention_<tag>.h5   : scalars + carrot/water C snapshots
  - phase4_retention_<tag>.png  : 3-panel retention / T / mass-balance plot
  - stdout                       : retention + mass-balance summary
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
from boilingsim.geometry import MAT_CARROT, MAT_FLUID, MAT_POT_WALL  # noqa: E402
from boilingsim.pipeline import Simulation  # noqa: E402


# -------- Helpers --------------------------------------------------------------


def load_scalars(h5: h5py.File) -> dict:
    g = h5["scalars"]
    return {k: np.asarray(g[k]) for k in g.keys()}


def plot_retention_summary(
    h5_path: pathlib.Path,
    out_path: pathlib.Path,
    tag: str,
    C0: float,
    target_band: tuple[float, float] = (80.0, 90.0),
) -> dict:
    """Three-panel figure: (a) retention vs time with target band, (b) water
    and wall temperatures, (c) total carrot/water mass (mass-balance check).
    Returns a dict of validation stats.
    """
    with h5py.File(h5_path, "r") as f:
        sc = load_scalars(f)

    t = sc["t"]
    R = sc["retention_pct"]
    Tw = sc["T_mean_water_c"]
    Twall = sc["T_inner_wall_mean_c"] if "T_inner_wall_mean_c" in sc else sc["T_max_wall_c"]

    # Phase-4 instrumentation: leached vs degraded breakdown (Phase-4-prime).
    have_breakdown = "leached_pct" in sc and "degraded_pct" in sc
    leached = sc["leached_pct"] if have_breakdown else None
    degraded = sc["degraded_pct"] if have_breakdown else None

    # Final-state values (averaged over last 5 % of the series for noise).
    tail = max(1, len(R) // 20)
    R_final = float(R[-tail:].mean())
    leached_final = float(leached[-tail:].mean()) if have_breakdown else 0.0
    degraded_final = float(degraded[-tail:].mean()) if have_breakdown else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Retention as a stacked-area mass-conservation breakdown:
    #     bottom = still in carrot (R), middle = leached into water, top = degraded.
    #     Three bands sum to 100 % at every t. Lets you tell at a glance
    #     whether retention loss is mostly Arrhenius or mostly leaching.
    axes[0].axhspan(target_band[0], target_band[1], color="tab:green", alpha=0.10,
                     label=f"target band [{target_band[0]:.0f}%,{target_band[1]:.0f}%]")
    axes[0].axhline(84.0, ls=":", color="k", alpha=0.5, label="exp ref 84%")
    if have_breakdown:
        axes[0].fill_between(t, 0, R, color="tab:green", alpha=0.55,
                              label=f"in carrot ({R_final:.1f}%)")
        axes[0].fill_between(t, R, R + leached, color="tab:blue", alpha=0.55,
                              label=f"leached ({leached_final:.1f}%)")
        axes[0].fill_between(t, R + leached, R + leached + degraded,
                              color="tab:red", alpha=0.45,
                              label=f"degraded ({degraded_final:.1f}%)")
    axes[0].plot(t, R, "-", color="tab:purple", lw=1.5, label="R(t) measured")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("mass fraction of initial [%]")
    axes[0].set_title(f"{tag}: beta-carotene mass partition")
    axes[0].set_ylim(0, 102)
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(alpha=0.3)

    # (b) Temperatures
    axes[1].plot(t, Tw, "-", color="tab:blue", label="water mean")
    axes[1].plot(t, Twall, "-", color="tab:red", label="wall inner")
    axes[1].axhline(100.0, ls=":", color="k", alpha=0.4, label="T_sat")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("T [C]")
    axes[1].set_title(f"{tag}: temperatures")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # (c) Bubble count (as proxy for boiling vigour) + peak velocity
    ax3 = axes[2]
    if "n_active_bubbles" in sc:
        ax3.plot(t, sc["n_active_bubbles"], "-", color="tab:green", label="bubbles")
        ax3.set_ylabel("n active bubbles", color="tab:green")
        ax3.tick_params(axis="y", labelcolor="tab:green")
    if "u_max_mps" in sc:
        ax3_r = ax3.twinx()
        ax3_r.plot(t, sc["u_max_mps"] * 1000.0, "-",
                   color="tab:orange", alpha=0.7, label="|u|_max")
        ax3_r.set_ylabel("|u|_max [mm/s]", color="tab:orange")
        ax3_r.tick_params(axis="y", labelcolor="tab:orange")
    ax3.set_xlabel("time [s]")
    ax3.set_title(f"{tag}: boiling vigour")
    ax3.grid(alpha=0.3)

    fig.suptitle(f"Phase 4 retention validation - {tag}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "R_final_pct": R_final,
        "leached_final_pct": leached_final,
        "degraded_final_pct": degraded_final,
        "have_breakdown": have_breakdown,
        "target_band_lo": target_band[0],
        "target_band_hi": target_band[1],
        "in_band": target_band[0] <= R_final <= target_band[1],
        "T_water_final_c": float(Tw[-1]),
        "T_wall_final_c": float(Twall[-1]),
    }


# -------- Main -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--duration", type=float, default=600.0,
                    help="Simulated time in seconds (dev-guide target: 600 s).")
    ap.add_argument("--dx-mm", type=float, default=2.0)
    ap.add_argument("--carrot-diameter-mm", type=float, default=None,
                    help="Override carrot diameter. Default: use YAML value.")
    ap.add_argument("--max-bubbles", type=int, default=100_000)
    ap.add_argument("--pressure-iters", type=int, default=100)
    ap.add_argument("--snapshot-every-s", type=float, default=30.0)
    ap.add_argument("--warm-start-water-c", type=float, default=95.0,
                    help="Pre-heat water to this T before the run (default: 95 C, "
                         "matches experimental boiling setup).")
    ap.add_argument("--warm-start-wall-c", type=float, default=100.0)
    ap.add_argument("--warm-start-carrot-c", type=float, default=20.0,
                    help="Carrot initial temperature -- default 20 C, matching "
                         "'dropped into boiling water' cooking protocol.")
    ap.add_argument("--tag", type=str, default=None,
                    help="Output filename tag. Default: <material>_<D_carrot>mm.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "benchmarks")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_scenario(args.config)
    cfg.grid.dx_m = args.dx_mm / 1000.0
    cfg.solver.pressure_max_iter = args.pressure_iters
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = args.max_bubbles
    cfg.nutrient.enabled = True

    if args.carrot_diameter_mm is not None:
        cfg.carrot.diameter_m = args.carrot_diameter_mm / 1000.0

    tag = args.tag
    if tag is None:
        tag = f"{cfg.pot.material}_{int(round(cfg.carrot.diameter_m*1000))}mm"

    material = cfg.pot.material
    q_stove_kw = cfg.heating.base_heat_flux_w_per_m2 / 1000.0
    print(f"=== Phase 4 retention validation: {tag} ===")
    print(f"  config         : {args.config}")
    print(f"  duration       : {args.duration:.1f} s")
    print(f"  dx             : {args.dx_mm:.2f} mm")
    print(f"  carrot D       : {cfg.carrot.diameter_m*1000:.1f} mm")
    print(f"  carrot L       : {cfg.carrot.length_m*1000:.1f} mm")
    print(f"  pot material   : {material}")
    print(f"  q_stove        : {q_stove_kw:.0f} kW/m^2")
    print(f"  Arrhenius E_a  : {cfg.nutrient.E_a_kJ_per_mol:.1f} kJ/mol")
    print(f"  Arrhenius k0   : {cfg.nutrient.k0_per_s:.2e} /s")
    print(f"  D_eff          : {cfg.nutrient.D_eff_m2_per_s:.2e} m^2/s")
    print(f"  K_partition    : {cfg.nutrient.K_partition:.3f}")
    print(f"  C0             : {cfg.nutrient.C0_mg_per_kg:.1f} mg/kg")
    print(f"  warm-start     : water={args.warm_start_water_c}C, "
          f"wall={args.warm_start_wall_c}C, carrot={args.warm_start_carrot_c}C")

    # --- Build and warm-start ---
    sim = Simulation(cfg, device=args.device)
    T_np = sim.grid.T.numpy()
    mat_np = sim.grid.mat.numpy()
    T_np[mat_np == MAT_FLUID] = args.warm_start_water_c + 273.15
    T_np[mat_np == MAT_POT_WALL] = args.warm_start_wall_c + 273.15
    T_np[mat_np == MAT_CARROT] = args.warm_start_carrot_c + 273.15
    sim.grid.T.assign(T_np)

    # --- Run ---
    print(f"\n  === running {args.duration:.0f} s of boiling + nutrient ===")
    out_h5 = args.out_dir / f"phase4_retention_{tag}.h5"
    t0 = time.perf_counter()
    # scalar_every_n_steps = 100 rather than 20: sample_scalars does 5
    # synchronising GPU->host array copies per call (T, bubbles struct,
    # water_alpha, C, C_water) and the former default at dt ~ 2 ms meant
    # ~7000 sync points per 600 s sim = ~0.8 s/sim-s regression vs Phase 3.
    # 100-step cadence still gives ~200 ms sim resolution in the HDF5
    # trace, plenty for retention plots and mass-balance checks.
    scalars = sim.run(
        total_time_s=args.duration,
        out_path=out_h5,
        scalar_every_n_steps=100,
        snapshot_every_s=args.snapshot_every_s,
        progress_every_s=30.0,
    )
    wall = time.perf_counter() - t0
    print(f"\n  sim done: {wall:.1f} s wall, {sim.step_count} steps, "
          f"{wall/args.duration:.2f} s/sim-s")

    # --- Plots + validation summary ---
    plot_path = args.out_dir / f"phase4_retention_{tag}.png"
    stats = plot_retention_summary(
        out_h5, plot_path, tag,
        C0=cfg.nutrient.C0_mg_per_kg,
    )

    print("\n=== Retention summary ===")
    print(f"  R({args.duration:.0f} s)         = {stats['R_final_pct']:.2f} %")
    print(f"  target band       = [{stats['target_band_lo']:.0f} %, "
          f"{stats['target_band_hi']:.0f} %] (exp ref 84 %)")
    status = "IN BAND" if stats["in_band"] else "OUT OF BAND"
    print(f"  => {status}")
    if stats["have_breakdown"]:
        print(f"  -- mass partition (sums to ~100%) --")
        print(f"  still in carrot   = {stats['R_final_pct']:6.2f} %")
        print(f"  leached to water  = {stats['leached_final_pct']:6.2f} %  "
              f"(Sherwood surface flux)")
        print(f"  degraded          = {stats['degraded_final_pct']:6.2f} %  "
              f"(Arrhenius + small advection numerical loss)")
    print(f"  T_water final     = {stats['T_water_final_c']:.2f} C")
    print(f"  T_wall inner      = {stats['T_wall_final_c']:.2f} C")
    print(f"  plot              : {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
