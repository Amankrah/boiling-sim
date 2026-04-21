"""Run a full heating scenario end-to-end and validate against a lumped-capacitance
ODE reference (Phase 2, Milestone D).

Produces:
  benchmarks/phase2_heating_<material>.h5     — raw HDF5 time series + snapshots
  benchmarks/phase2_heating_<material>.png    — sim vs lumped plot
  stdout: time-to-95C for sim and lumped
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
from scipy.integrate import solve_ivp  # noqa: E402

from boilingsim.config import ScenarioConfig, load_scenario  # noqa: E402
from boilingsim.json_hash_comments import loads_json_with_hash_comments  # noqa: E402
from boilingsim.pipeline import Simulation  # noqa: E402


# ---------------------------------------------------------------------------
# Lumped-capacitance reference
# ---------------------------------------------------------------------------


def lumped_capacitance_ode(cfg: ScenarioConfig) -> dict:
    """One-body lumped reference: dT/dt = (Q_stove − Q_loss(T) − Q_evap) / C_sys.

    Parameters
    ----------
    cfg
        Scenario config. Used to compute masses, areas, heat capacities.

    Returns
    -------
    dict with params, time grid, T(t), and time-to-95C.
    """
    # Load material properties (we don't want the GPU-side MaterialProps here).
    materials = loads_json_with_hash_comments(
        (ROOT / "data" / "materials.json").read_text(encoding="utf-8")
    )
    water = materials["water"]
    pot = materials[cfg.pot.material]

    r_outer = cfg.pot.diameter_m / 2
    r_inner = r_outer - cfg.pot.wall_thickness_m
    h = cfg.pot.height_m
    bt = cfg.pot.base_thickness_m
    h_inner = h - bt

    # Water mass
    water_height = cfg.water.fill_fraction * h_inner
    V_water = math.pi * r_inner ** 2 * water_height
    m_water = water["rho_ref"] * V_water

    # Pot mass: outer cylinder volume minus the inner cavity.
    V_pot = math.pi * r_outer ** 2 * h - math.pi * r_inner ** 2 * h_inner
    m_pot = pot["rho"] * V_pot

    # Heat capacities
    C_water = m_water * water["c_p"]
    C_pot = m_pot * pot["c_p"]
    C_sys = C_water + C_pot

    # Heating power (stove flux × base area)
    A_base = math.pi * r_outer ** 2
    P_stove = cfg.heating.base_heat_flux_w_per_m2 * A_base

    # External area for Newton cooling (sides + top rim + base underside).
    A_side = 2.0 * math.pi * r_outer * h
    A_top_rim = math.pi * (r_outer ** 2 - r_inner ** 2)
    A_base_bot = math.pi * r_outer ** 2  # in our grid the base also has air below
    A_ext = A_side + A_top_rim + A_base_bot
    h_conv = cfg.solver.h_conv_outer_w_per_m2_k
    T_amb_k = cfg.heating.ambient_temp_c + 273.15

    # Evaporative sink (0.1 · q_base on the water free surface, gated 85→100C)
    A_surface = math.pi * r_inner ** 2
    P_evap_max = 0.1 * cfg.heating.base_heat_flux_w_per_m2 * A_surface
    T_onset_k = 85.0 + 273.15
    T_sat_k = 100.0 + 273.15

    # ODE: dT/dt = (P_stove - h·A_ext·(T-T_amb) - P_evap(T)) / C_sys
    def rhs(_t, y):
        T = y[0]
        Q_loss = h_conv * A_ext * (T - T_amb_k)
        frac = max(0.0, min(1.0, (T - T_onset_k) / (T_sat_k - T_onset_k)))
        P_evap = P_evap_max * frac
        return [(P_stove - Q_loss - P_evap) / C_sys]

    T0 = cfg.water.initial_temp_c + 273.15
    t_max = cfg.total_time_s
    sol = solve_ivp(rhs, (0.0, t_max), [T0], max_step=5.0, dense_output=True)
    t_grid = np.linspace(0.0, t_max, 1000)
    T_grid = sol.sol(t_grid)[0]

    # Time to reach 95 C (368.15 K) — interpolate from the ODE solution.
    target_k = 95.0 + 273.15
    if T_grid.max() >= target_k:
        idx = int(np.argmax(T_grid >= target_k))
        t_95 = t_grid[idx]
    else:
        # Solve analytically: T(t) = T_amb + (T0 - T_amb - P_net/(h*A))·exp(-t/tau) + P_net/(h*A)
        # For convenience, just return None and let the caller extrapolate.
        t_95 = None

    return {
        "m_water_kg": m_water,
        "m_pot_kg": m_pot,
        "C_sys_j_per_k": C_sys,
        "P_stove_w": P_stove,
        "P_evap_max_w": P_evap_max,
        "h_conv_A_ext_w_per_k": h_conv * A_ext,
        "A_ext_m2": A_ext,
        "t": t_grid,
        "T_k": T_grid,
        "t_to_95c_s": t_95,
    }


# ---------------------------------------------------------------------------
# Plot + compare
# ---------------------------------------------------------------------------


def plot_heating(
    scenario_name: str,
    sim_t: np.ndarray,
    sim_T_mean: np.ndarray,
    sim_T_max_wall: np.ndarray,
    sim_u_max_mmps: np.ndarray,
    lumped: dict,
    out_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: mean water T — sim vs lumped
    axes[0].plot(lumped["t"], lumped["T_k"] - 273.15, "--", color="gray", label="Lumped capacitance")
    axes[0].plot(sim_t, sim_T_mean, "-", color="tab:red", label="Sim (mean water)")
    axes[0].axhline(95.0, ls=":", color="k", alpha=0.5)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("T [C]")
    axes[0].set_title(f"{scenario_name}: mean water T")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Middle: max wall T
    axes[1].plot(sim_t, sim_T_max_wall, "-", color="tab:orange")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("T [C]")
    axes[1].set_title("max wall T")
    axes[1].grid(alpha=0.3)

    # Right: peak velocity
    axes[2].plot(sim_t, sim_u_max_mmps, "-", color="tab:blue")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("|u|_max [mm/s]")
    axes[2].set_title("peak convection velocity")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--duration", type=float, default=None,
                    help="Override total_time_s from config.")
    ap.add_argument("--pressure-iters", type=int, default=None,
                    help="Override solver.pressure_max_iter.")
    ap.add_argument("--dx-mm", type=float, default=None,
                    help="Override grid.dx_m (in millimetres) for faster runs.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "benchmarks")
    ap.add_argument("--suffix", type=str, default="",
                    help="Extra suffix for output filenames.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_scenario(args.config)
    if args.duration is not None:
        cfg.total_time_s = args.duration
    if args.pressure_iters is not None:
        cfg.solver.pressure_max_iter = args.pressure_iters
    if args.dx_mm is not None:
        cfg.grid.dx_m = args.dx_mm / 1000.0

    material = cfg.pot.material
    print(f"=== Heating scenario: {material} ===")
    print(f"  config   : {args.config}")
    print(f"  duration : {cfg.total_time_s:.1f}s")
    print(f"  grid dx  : {cfg.grid.dx_m*1000:.2f}mm")

    # Lumped reference
    lumped = lumped_capacitance_ode(cfg)
    print(f"  lumped: m_water={lumped['m_water_kg']:.3f}kg  m_pot={lumped['m_pot_kg']:.3f}kg  "
          f"C={lumped['C_sys_j_per_k']/1000:.1f}kJ/K  P_stove={lumped['P_stove_w']:.1f}W  "
          f"P_evap_max={lumped['P_evap_max_w']:.1f}W (gated 85-100C)")
    if lumped["t_to_95c_s"] is not None:
        print(f"  lumped: time-to-95C = {lumped['t_to_95c_s']:.1f}s ({lumped['t_to_95c_s']/60:.2f}min)")
    else:
        print(f"  lumped: does not reach 95C within {cfg.total_time_s}s — T_final={lumped['T_k'][-1]-273.15:.2f}C")

    # Run simulation
    print(f"\n  === running simulation ===")
    sim = Simulation(cfg, device=args.device)
    tag = f"{material}{args.suffix}"
    out_h5 = args.out_dir / f"phase2_heating_{tag}.h5"
    wall_t0 = time.perf_counter()
    scalars = sim.run(
        total_time_s=cfg.total_time_s,
        out_path=out_h5,
        scalar_every_n_steps=10,
        snapshot_every_s=60.0,
        progress_every_s=30.0,
    )
    wall = time.perf_counter() - wall_t0
    print(f"\n  sim done: {wall:.1f}s wall time, {sim.step_count} steps, "
          f"{wall/cfg.total_time_s:.3f} s/sim-s")

    # Load scalars
    sim_t = np.array([s.t for s in scalars])
    sim_T_mean = np.array([s.T_mean_water_c for s in scalars])
    sim_T_max_wall = np.array([s.T_max_wall_c for s in scalars])
    sim_u_max_mmps = np.array([s.u_max_mps for s in scalars]) * 1000

    # Time-to-95 from sim
    if sim_T_mean.max() >= 95.0:
        idx = int(np.argmax(sim_T_mean >= 95.0))
        sim_t95 = float(sim_t[idx])
        print(f"  sim: time-to-95C = {sim_t95:.1f}s ({sim_t95/60:.2f}min)")
    else:
        sim_t95 = None
        print(f"  sim: does not reach 95C — T_final={sim_T_mean[-1]:.2f}C")

    # Comparison summary
    print("\n=== Summary ===")
    print(f"  {'':20s}{'sim':>10s}{'lumped':>10s}{'err':>10s}")
    # ONB-capped comparison (Phase 2 is only valid up to onset of nucleate boiling).
    T_ONB_C = 105.0
    above = sim_T_max_wall >= T_ONB_C
    if above.any():
        idx = int(np.argmax(above))
        if idx > 0:
            y0, y1 = sim_T_max_wall[idx - 1], sim_T_max_wall[idx]
            t0, t1 = sim_t[idx - 1], sim_t[idx]
            frac = (T_ONB_C - y0) / (y1 - y0) if y1 != y0 else 0.0
            t_onb = float(t0 + frac * (t1 - t0))
        else:
            t_onb = float(sim_t[0])
        sim_T_at_onb = float(np.interp(t_onb, sim_t, sim_T_mean))
        lumped_T_at_onb = float(np.interp(t_onb, lumped["t"], lumped["T_k"] - 273.15))
        err_onb = 100.0 * (sim_T_at_onb - lumped_T_at_onb) / lumped_T_at_onb
        print(f"  {'t_ONB_s':20s}{t_onb:10.1f}   (T_wall hits {T_ONB_C}C)")
        print(f"  {'T_water_at_ONB':20s}{sim_T_at_onb:10.2f}{lumped_T_at_onb:10.2f}{err_onb:9.2f}%")
    print(f"  {'T_final_c':20s}{sim_T_mean[-1]:10.2f}{lumped['T_k'][-1]-273.15:10.2f}"
          f"{100.0*(sim_T_mean[-1]-(lumped['T_k'][-1]-273.15))/(lumped['T_k'][-1]-273.15):9.2f}%")
    if sim_t95 is not None and lumped["t_to_95c_s"] is not None:
        err = 100.0 * (sim_t95 - lumped["t_to_95c_s"]) / lumped["t_to_95c_s"]
        print(f"  {'t_to_95c_s':20s}{sim_t95:10.1f}{lumped['t_to_95c_s']:10.1f}{err:9.2f}%")

    # Plot
    plot_path = args.out_dir / f"phase2_heating_{tag}.png"
    plot_heating(tag, sim_t, sim_T_mean, sim_T_max_wall, sim_u_max_mmps, lumped, plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
