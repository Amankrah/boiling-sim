"""Re-analyze existing Phase 2 heating HDF5 files with an ONB cap.

Phase 2 physics (sensible heating + natural convection, no boiling) is only
valid while T_wall < T_sat + ΔT_ONB ≈ 105 °C. Past that the lumped-capacitance
ODE is also invalid (real system would boil, latent heat kicks in). So the
fair validation window is ``[0, t_ONB]`` where ``t_ONB`` is the first time
``max(T_wall) > 105 °C``.

Usage:
    python scripts/reanalyze_heating_onb.py
Produces:
    benchmarks/phase2_heating_{material}_onb.png     per-material plots
    benchmarks/phase2_heating_onb_summary.md          combined table
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boilingsim.config import load_scenario  # noqa: E402
from run_heating import lumped_capacitance_ode  # noqa: E402


T_ONB_C = 105.0  # Wall temperature where boiling kicks in (T_sat + 5 K ONB margin)


def find_first_crossing(t: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Linearly interpolate the first t where y(t) crosses `threshold`."""
    above = y >= threshold
    if not above.any():
        return None
    idx = int(np.argmax(above))
    if idx == 0:
        return float(t[0])
    # Linear interp between (t[idx-1], y[idx-1]) and (t[idx], y[idx])
    y0, y1 = y[idx - 1], y[idx]
    t0, t1 = t[idx - 1], t[idx]
    if y1 == y0:
        return float(t[idx])
    frac = (threshold - y0) / (y1 - y0)
    return float(t0 + frac * (t1 - t0))


def analyze_one(h5_path: pathlib.Path, config_path: pathlib.Path) -> dict:
    cfg = load_scenario(config_path)
    lumped = lumped_capacitance_ode(cfg)

    with h5py.File(h5_path, "r") as f:
        t = np.asarray(f["scalars/t"])
        T_water = np.asarray(f["scalars/T_mean_water_c"])
        T_wall_max = np.asarray(f["scalars/T_max_wall_c"])

    t_onb = find_first_crossing(t, T_wall_max, T_ONB_C)

    # Sim T_water at t_onb (interp)
    if t_onb is None:
        return {
            "material": cfg.pot.material,
            "t_onb_s": None,
            "T_water_sim_c": float(T_water[-1]),
            "T_water_lumped_c": float(lumped["T_k"][-1] - 273.15),
            "T_wall_max_c": float(T_wall_max[-1]),
            "t_final": float(t[-1]),
            "t": t, "T_water": T_water, "T_wall_max": T_wall_max,
            "lumped_t": lumped["t"],
            "lumped_T_c": lumped["T_k"] - 273.15,
        }

    sim_T_at_onb = float(np.interp(t_onb, t, T_water))
    lumped_T_at_onb = float(np.interp(t_onb, lumped["t"], lumped["T_k"] - 273.15))
    err_pct = 100.0 * (sim_T_at_onb - lumped_T_at_onb) / lumped_T_at_onb

    return {
        "material": cfg.pot.material,
        "t_onb_s": t_onb,
        "T_water_sim_at_onb": sim_T_at_onb,
        "T_water_lumped_at_onb": lumped_T_at_onb,
        "err_pct_at_onb": err_pct,
        "T_water_sim_final": float(T_water[-1]),
        "T_wall_max_final": float(T_wall_max[-1]),
        "t_final": float(t[-1]),
        "t": t, "T_water": T_water, "T_wall_max": T_wall_max,
        "lumped_t": lumped["t"],
        "lumped_T_c": lumped["T_k"] - 273.15,
    }


def plot_one(result: dict, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: mean water T vs time, sim + lumped, with ONB cutoff line
    axes[0].plot(result["lumped_t"], result["lumped_T_c"], "--",
                 color="gray", label="Lumped capacitance")
    axes[0].plot(result["t"], result["T_water"], "-",
                 color="tab:red", label="Sim (mean water)")
    if result["t_onb_s"] is not None:
        t_onb = result["t_onb_s"]
        axes[0].axvline(t_onb, color="k", linestyle=":", alpha=0.6)
        axes[0].annotate(
            f"ONB\nt={t_onb:.0f}s",
            xy=(t_onb, 20), xytext=(t_onb + 20, 25),
            arrowprops=dict(arrowstyle="->", alpha=0.5),
            fontsize=9,
        )
        sim_T = result["T_water_sim_at_onb"]
        lump_T = result["T_water_lumped_at_onb"]
        axes[0].plot([t_onb], [sim_T], "ro", markersize=7, zorder=5)
        axes[0].plot([t_onb], [lump_T], "o", color="gray", markersize=7, zorder=5)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("T [C]")
    axes[0].set_title(f"{result['material']}: mean water T (Phase 2 valid until ONB)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: max wall T vs time with T_ONB line
    axes[1].plot(result["t"], result["T_wall_max"], "-", color="tab:orange")
    axes[1].axhline(T_ONB_C, color="k", linestyle=":", alpha=0.6,
                    label=f"T_ONB = {T_ONB_C:.0f} C")
    if result["t_onb_s"] is not None:
        axes[1].axvline(result["t_onb_s"], color="k", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("T [C]")
    axes[1].set_title("max wall T  (ONB when this crosses 105C)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    bench_dir = ROOT / "benchmarks"
    config_dir = ROOT / "configs" / "scenarios"

    scenarios = [
        ("steel_304", "default.yaml"),
        ("aluminum", "aluminum.yaml"),
        ("copper", "copper.yaml"),
    ]

    results = []
    summary_lines: list[str] = []
    summary_lines.append("# Phase 2 Heating Validation — ONB-Capped Summary\n")
    summary_lines.append(
        "Phase 2 covers sensible heating + natural convection only. "
        "The simulation and the lumped-capacitance ODE are both valid "
        f"up to the onset of nucleate boiling (T_wall > {T_ONB_C:.0f}°C). "
        "Validation compares T_water at t_ONB.\n")
    summary_lines.append(
        "| Material | t_ONB (s) | T_wall @ t_ONB | T_water sim | T_water lumped | Error |")
    summary_lines.append(
        "|---|---:|---:|---:|---:|---:|")

    for material, yaml_name in scenarios:
        h5_file = bench_dir / f"phase2_heating_{material}_dx2mm.h5"
        if not h5_file.exists():
            print(f"skip {material}: {h5_file.name} not found")
            continue
        config_file = config_dir / yaml_name
        result = analyze_one(h5_file, config_file)
        plot_path = bench_dir / f"phase2_heating_{material}_onb.png"
        plot_one(result, plot_path)
        results.append(result)

        if result["t_onb_s"] is None:
            summary_lines.append(
                f"| {material} | **never** | {result['T_wall_max_final']:.1f}°C "
                f"(at t={result['t_final']:.0f}s) | — | — | Phase 2 stayed valid entire run |"
            )
        else:
            summary_lines.append(
                f"| {material} | {result['t_onb_s']:.0f} "
                f"| {T_ONB_C:.0f}°C "
                f"| {result['T_water_sim_at_onb']:.1f}°C "
                f"| {result['T_water_lumped_at_onb']:.1f}°C "
                f"| {result['err_pct_at_onb']:+.2f}% |"
            )

        print(f"\n=== {material} ===")
        print(f"  t_ONB = {result['t_onb_s']}")
        if result["t_onb_s"] is not None:
            print(f"  sim T_water @ ONB    = {result['T_water_sim_at_onb']:.2f} C")
            print(f"  lumped T_water @ ONB = {result['T_water_lumped_at_onb']:.2f} C")
            print(f"  error                = {result['err_pct_at_onb']:+.2f}%")
        print(f"  sim T_water final    = {result['T_water_sim_final']:.2f} C")
        print(f"  sim T_wall_max final = {result['T_wall_max_final']:.2f} C")

    summary_path = bench_dir / "phase2_heating_onb_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
