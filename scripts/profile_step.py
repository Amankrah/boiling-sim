"""Per-phase wall-time breakdown of ``Simulation.step`` (Phase-7 M1).

Wraps each step phase with a ``wp.synchronize_device`` pair so we can
attribute GPU work to its caller. Anchors all later optimization work
(M2–M5) in measured cost share rather than guesses.

Usage:
    python scripts/profile_step.py \
        --config configs/scenarios/default.yaml \
        --duration 30 --warmup 2 --dx-mm 2.0 --pressure-iters 100 \
        --max-bubbles 100000

Output:
    - stdout: ranked per-phase table
    - benchmarks/profile_step_breakdown.csv (overwritten each run)

Caveats:
    The synchronize-device pairs add ~5–10 % to step wall time -- the
    measured fractions are accurate, but the absolute s/sim-s number
    will be larger than a non-profiled run. Document the relative
    breakdown, not the total.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Enable profiling BEFORE importing the pipeline so Simulation.__init__
# picks it up.
os.environ["BOILINGSIM_PROFILE"] = "1"

from boilingsim.config import load_scenario  # noqa: E402
from boilingsim.geometry import MAT_FLUID, MAT_POT_WALL  # noqa: E402
from boilingsim.pipeline import Simulation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path,
                    default=ROOT / "configs" / "scenarios" / "default.yaml")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Sim seconds to profile after warmup (default 30).")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="Sim seconds to discard before measuring (default 2).")
    ap.add_argument("--dx-mm", type=float, default=2.0)
    ap.add_argument("--max-bubbles", type=int, default=100_000)
    ap.add_argument("--pressure-iters", type=int, default=100)
    ap.add_argument("--warm-start-water-c", type=float, default=95.0)
    ap.add_argument("--warm-start-wall-c", type=float, default=100.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", type=pathlib.Path,
                    default=ROOT / "benchmarks" / "profile_step_breakdown.csv")
    args = ap.parse_args()

    cfg = load_scenario(args.config)
    cfg.grid.dx_m = args.dx_mm / 1000.0
    cfg.solver.pressure_max_iter = args.pressure_iters
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = args.max_bubbles

    print(f"=== profile_step ({cfg.pot.material}) ===")
    print(f"  config        : {args.config}")
    print(f"  warmup / meas : {args.warmup:.1f} s / {args.duration:.1f} s")
    print(f"  dx            : {args.dx_mm:.2f} mm")
    print(f"  pressure iter : {args.pressure_iters}")
    print(f"  max bubbles   : {args.max_bubbles:,}")

    sim = Simulation(cfg, device=args.device)
    if not sim._profile_enabled:
        print("ERROR: Simulation did not see BOILINGSIM_PROFILE=1.")
        return 1

    # Warm-start to avoid measuring the cold-water heat-up regime --
    # we want the steady-boiling phase since that's where the user's
    # 6.74 s/sim-s baseline lives.
    T_np = sim.grid.T.numpy()
    mat_np = sim.grid.mat.numpy()
    T_np[mat_np == MAT_FLUID] = args.warm_start_water_c + 273.15
    T_np[mat_np == MAT_POT_WALL] = args.warm_start_wall_c + 273.15
    sim.grid.T.assign(T_np)

    # Phase 1: warmup -- profiling is on but we'll reset() before measurement.
    print(f"  warming up {args.warmup:.1f} sim-s ...")
    target_t = sim.t + args.warmup
    while sim.t < target_t:
        sim.step()

    sim.reset_profile()
    t_wall0 = time.perf_counter()
    t_sim0 = sim.t
    target_t = sim.t + args.duration
    print(f"  measuring {args.duration:.1f} sim-s ...")
    while sim.t < target_t:
        sim.step()
    t_wall = time.perf_counter() - t_wall0
    t_sim = sim.t - t_sim0
    n_steps = sim._profile_n

    rows = sim.profile_summary()
    if not rows:
        print("ERROR: no profile rows captured.")
        return 1

    sim_s_per_wall_s = t_sim / t_wall if t_wall > 0 else 0.0
    wall_s_per_sim_s = t_wall / t_sim if t_sim > 0 else 0.0
    total_phase_s = sum(r[1] for r in rows)
    profile_overhead_pct = 100.0 * (1.0 - total_phase_s / t_wall) if t_wall > 0 else 0.0

    print()
    print(f"  wall            = {t_wall:.2f} s   ({wall_s_per_sim_s:.3f} s/sim-s)")
    print(f"  sim-s measured  = {t_sim:.2f} s")
    print(f"  steps           = {n_steps:,}  ({1000.0 * t_wall / max(n_steps, 1):.3f} ms/step)")
    print(f"  rate            = {sim_s_per_wall_s:.4f} sim-s / wall-s")
    print(f"  phase-sum / wall = {100.0 * total_phase_s / t_wall:.1f} %"
          f"  (residual {profile_overhead_pct:.1f} % is python overhead + sync stalls)")
    print()
    print(f"  {'phase':<28} {'total_s':>10} {'mean_ms/step':>14} {'frac_pct':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*14} {'-'*10}")
    for name, total_s, mean_ms, frac_pct in rows:
        print(f"  {name:<28} {total_s:>10.3f} {mean_ms:>14.4f} {frac_pct:>9.2f}%")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "total_s", "mean_ms_per_step", "frac_pct"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.4f}"])
    print(f"\n  wrote {args.csv}")

    top5 = sum(r[3] for r in rows[:5])
    if top5 < 85.0:
        print(f"\n  WARNING: top-5 phases sum to {top5:.1f} % (<85 %); "
              f"breakdown may be missing a phase.")
    else:
        print(f"\n  top-5 phases sum to {top5:.1f} % (gate >=85 %).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
