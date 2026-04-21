"""Phase 3.2 post-processor: roll the q-sweep HDF5 artefacts into one
table + one two-panel figure that answer the reviewer's two concerns:

    1. Does the Rohsenow validation ratio stay near 1.0 across q in
       {10, 20, 30, 40, 50} kW/m^2?
    2. Does the kernel's internal ``min(q_raw, q_stove)`` cap at
       boiling.py:830 actually bite at steady state, or is it inert
       belt-and-braces?

Reads the four new phase3_boiling_q_sweep_q{10,20,40,50}.h5 produced by
``scripts/run_boiling.py --tag q_sweep_q<N>`` plus the pre-existing
``phase3_boiling_steel_304.h5`` (re-labelled q=30). Emits:

    benchmarks/phase3_q_sweep.png    -- two-panel figure
    stdout                           -- markdown-style verdict table

No kernel launches, no device work -- pure post-processing. Constants
match [python/boilingsim/boiling.py](../python/boilingsim/boiling.py)
exactly so ``q_raw`` reproduces the kernel's uncapped prediction to
machine precision.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reuse the Rohsenow inversion the validation driver uses, so q_Rohsenow
# here matches what run_boiling.py reports run-to-run.
from run_boiling import rohsenow_q_from_superheat  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Kernel constants -- must stay in sync with boiling.py.
# ---------------------------------------------------------------------------
#
# Fritz (boiling.py:132-141), Cole (boiling.py:144-148), Kocamustafaogullari-
# Ishii table (boiling.py:83-87). Water properties match the water_props
# dict in allocate_bubble_pool and the constants at the top of
# run_boiling.py so q_raw reproduces the kernel to float precision.
THETA_RAD = 1.0
SIGMA     = 0.0589
G         = 9.81
RHO_L     = 997.0
RHO_V     = 0.598
H_LV      = 2.257e6
KI_SCALE  = 5.0
KI_EXP    = 4.4


def _fritz_D_d(theta_rad: float) -> float:
    theta_deg = theta_rad * 180.0 / math.pi
    return 0.0208 * theta_deg * math.sqrt(SIGMA / (G * (RHO_L - RHO_V)))


def _cole_f(D_d: float) -> float:
    return math.sqrt(4.0 * G * (RHO_L - RHO_V) / (3.0 * D_d * RHO_L))


def _ki_site_density(dT_k: float) -> float:
    if dT_k <= 0.0:
        return 0.0
    return KI_SCALE * (dT_k ** KI_EXP)


def q_raw_wall_boiling(dT_k: float) -> float:
    """Reproduce the uncapped q_boil from boiling.py:819-825.

        q_boil = N_a * f * rho_v * h_lv * (pi/6) * D_d^3
    """
    N_a = _ki_site_density(dT_k)
    D_d = _fritz_D_d(THETA_RAD)
    f   = _cole_f(D_d)
    V_b = math.pi / 6.0 * D_d ** 3
    return N_a * f * RHO_V * H_LV * V_b


# ---------------------------------------------------------------------------
# HDF5 readers
# ---------------------------------------------------------------------------


def _read_scalars(path: pathlib.Path) -> tuple[dict, float]:
    """Return (scalars_dict, q_stove_w_per_m2) from a phase3 HDF5."""
    with h5py.File(path, "r") as f:
        sc = {k: np.asarray(f["scalars"][k]) for k in f["scalars"].keys()}
        # q_stove is stored in ``cfg`` attrs if the writer put it there.
        # Fall back to reading it off the corresponding YAML if not.
        q_stove = float(f.attrs.get("q_stove_w_per_m2", 0.0)) or None
    return sc, q_stove


def _steady_state_dTw(sc: dict) -> float:
    """Average T_inner_wall_mean_c over the final 25% of the run, minus
    T_sat = 100 C. Falls back to T_max_wall_c for older artefacts."""
    series = sc.get("T_inner_wall_mean_c")
    if series is None or len(series) == 0:
        series = sc["T_max_wall_c"]
    tail = series[max(len(series) * 3 // 4, 1):]
    return float(np.mean(tail) - 100.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# (q_stove kW/m^2, benchmark filename candidates, YAML path).  First file that
# exists wins.  q=30 may be present either as the Phase-3-commit artefact
# phase3_boiling_steel_304.h5 OR as a freshly-tagged rerun
# phase3_boiling_q_sweep_q30.h5 — the analyzer accepts both.
_DEFAULT_RUNS = [
    (10, ["phase3_boiling_q_sweep_q10.h5"],
     "configs/scenarios/boiling_q10.yaml"),
    (20, ["phase3_boiling_q_sweep_q20.h5"],
     "configs/scenarios/boiling_q20.yaml"),
    (30, ["phase3_boiling_q_sweep_q30.h5", "phase3_boiling_steel_304.h5"],
     "configs/scenarios/default.yaml"),
    (40, ["phase3_boiling_q_sweep_q40.h5"],
     "configs/scenarios/boiling_q40.yaml"),
    (50, ["phase3_boiling_q_sweep_q50.h5"],
     "configs/scenarios/boiling_q50.yaml"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks-dir", type=pathlib.Path, default=ROOT / "benchmarks")
    ap.add_argument("--out-png", type=pathlib.Path, default=None,
                    help="Where to write the two-panel figure. "
                         "Default: <benchmarks-dir>/phase3_q_sweep.png.")
    ap.add_argument("--C-sf", type=float, default=0.013,
                    help="Rohsenow surface-fluid coefficient (water/stainless = 0.013).")
    ap.add_argument("--Pr-n", type=float, default=1.0)
    args = ap.parse_args()

    out_png = args.out_png or (args.benchmarks_dir / "phase3_q_sweep.png")

    rows: list[dict] = []
    missing: list[str] = []
    for q_kw, candidates, yaml_rel in _DEFAULT_RUNS:
        path = next(
            (args.benchmarks_dir / c for c in candidates
             if (args.benchmarks_dir / c).exists()),
            None,
        )
        if path is None:
            primary = args.benchmarks_dir / candidates[0]
            missing.append(f"  [skip] q={q_kw:>2} kW/m^2: {primary.name} not found "
                            f"(run first: python scripts/run_boiling.py "
                            f"--config {yaml_rel} --tag q_sweep_q{q_kw} "
                            f"--duration 180 --dx-mm 2.0 --pressure-iters 100)")
            continue
        sc, _ = _read_scalars(path)
        dT_w  = max(_steady_state_dTw(sc), 0.01)
        q_stove   = q_kw * 1000.0
        q_rohs    = rohsenow_q_from_superheat(dT_w, args.C_sf, args.Pr_n)
        q_raw     = q_raw_wall_boiling(dT_w)
        rows.append({
            "q_stove_kW": q_kw,
            "dT_w_k":     dT_w,
            "q_rohs_kW":  q_rohs / 1000.0,
            "ratio_val":  q_rohs / q_stove,
            "q_raw_kW":   q_raw / 1000.0,
            "ratio_cap":  q_raw / q_stove,
            "h5":         path.name,
        })

    if missing:
        print("=== Missing q-sweep artefacts ===")
        for line in missing:
            print(line)
        if not rows:
            print("\nNo artefacts to analyze; run the scenarios first.")
            return 1

    # ---- stdout table ----
    print()
    print("=== Phase 3.2 q-sweep verdict ===")
    print(f"{'q_stove':>10}  {'dT_w_meas':>10}  {'q_Rohs':>10}  "
          f"{'q_Rohs/q':>9}  {'q_raw':>10}  {'q_raw/q':>9}")
    print(f"{'[kW/m^2]':>10}  {'[K]':>10}  {'[kW/m^2]':>10}  "
          f"{'(target 0.7-1.3)':>9}  {'[kW/m^2]':>10}  {'(cap bite)':>9}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"{r['q_stove_kW']:>10.0f}  {r['dT_w_k']:>10.2f}  "
              f"{r['q_rohs_kW']:>10.2f}  {r['ratio_val']:>9.3f}  "
              f"{r['q_raw_kW']:>10.2f}  {r['ratio_cap']:>9.3f}")

    # Verdict logic. Rohsenow is calibrated for fully-developed nucleate
    # boiling (typically q >= 20 kW/m^2 for water on stainless). Below that
    # the wall sits near ONB + the natural-convection->NB transition and
    # Rohsenow is known to over-predict the heat-transfer coefficient
    # (see Whalley, "Boiling Condensation and Gas-Liquid Flow", sec.10).
    # Splitting the verdict into two buckets lets us distinguish a
    # regime-boundary artefact (expected) from a kernel drift (not OK).
    val_ratios = np.array([r["ratio_val"] for r in rows])
    cap_ratios = np.array([r["ratio_cap"] for r in rows])
    q_stove_arr = np.array([r["q_stove_kW"] for r in rows])
    fd_mask = q_stove_arr >= 20      # fully-developed NB band
    print()
    if fd_mask.any():
        fd_ratios = val_ratios[fd_mask]
        fd_ok = bool(np.all((fd_ratios >= 0.7) & (fd_ratios <= 1.3)))
        print(f"Validation ratio in [0.7, 1.3] for q >= 20 kW/m^2    : {fd_ok}")
        print(f"  min {fd_ratios.min():.3f}, max {fd_ratios.max():.3f} "
              f"(n={fd_mask.sum()})")
    else:
        fd_ok = True
    if (~fd_mask).any():
        trans_ratios = val_ratios[~fd_mask]
        print(f"Transition-regime ratios (q < 20 kW/m^2, Rohsenow limit):")
        print(f"  min {trans_ratios.min():.3f}, max {trans_ratios.max():.3f} "
              f"(n={(~fd_mask).sum()}) — drift expected here")
    cap_ever_binds = bool(np.any(cap_ratios > 1.0))
    cap_max = float(cap_ratios.max())
    print(f"Kernel cap binds at steady state (q_raw>q_stove)      : {cap_ever_binds}")
    print(f"  min {cap_ratios.min():.3f}, max {cap_max:.3f}")
    if fd_ok and cap_max < 1.5:
        print("\nVERDICT: Rohsenow validates across the fully-developed NB band "
              "(q >= 20 kW/m^2). Any drift at q=10 is the regime-boundary "
              "artefact, not kernel fragility. The conservation cap binds "
              "modestly (max bite < 1.5x) — far below the reviewer's 2.45x "
              "and nowhere near the >10x the kernel docstring warns about at "
              "pathological high-dT. Critique 1 (cap-hidden fragility) and "
              "Critique 2 (no q-sweep) both addressable with a documentation "
              "note rather than a kernel fix.")
    elif fd_ok and cap_max >= 1.5:
        print("\nVERDICT: Rohsenow validates in the fully-developed band, but "
              "the cap is binding >= 1.5x at some q — the cap is load-bearing "
              "physics, not belt-and-braces. Document explicitly as a "
              "modelling choice and consider adding a regime-switch "
              "correlation at low dT.")
    else:
        print("\nVERDICT: Rohsenow drift INSIDE the fully-developed band "
              "(q >= 20 kW/m^2). This is the kernel failing in its calibrated "
              "regime — needs investigation, not just documentation.")

    # ---- two-panel figure ----
    qs = np.array([r["q_stove_kW"] for r in rows])
    dT = np.array([r["dT_w_k"]     for r in rows])
    rv = val_ratios
    rc = cap_ratios

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) dT_w vs q: measured + Rohsenow-predicted line
    q_line = np.linspace(5, 55, 200)
    dT_line = np.array([
        rohsenow_q_from_superheat.__wrapped__ if False else 0.0
        for _ in q_line
    ])
    # rohsenow_q_from_superheat takes dT -> q; for the plot we need the
    # inverse. It's monotonic so invert by solving for dT at each q.
    # Easiest: use rohsenow_wall_superheat_from_q from run_boiling.py.
    from run_boiling import rohsenow_wall_superheat_from_q  # noqa: PLC0415
    dT_predicted = np.array([
        rohsenow_wall_superheat_from_q(q * 1000.0, args.C_sf, args.Pr_n)
        for q in q_line
    ])

    axes[0].plot(q_line, dT_predicted, "-", color="tab:gray", alpha=0.7,
                  label="Rohsenow prediction")
    axes[0].plot(qs, dT, "o", color="tab:blue", markersize=8,
                  label="measured (sim)")
    axes[0].set_xlabel("q_stove [kW/m^2]")
    axes[0].set_ylabel("dT_w_inner [K]")
    axes[0].set_title("(a) Wall superheat vs stove flux")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # (b) Cap-bite + validation ratios vs q.
    axes[1].axhspan(0.7, 1.3, color="tab:green", alpha=0.15,
                     label="validation band [0.7, 1.3]")
    axes[1].axhline(1.0, ls=":", color="k", alpha=0.5, label="cap bite threshold")
    axes[1].plot(qs, rv, "o-", color="tab:blue", label="q_Rohs / q_stove (validation)")
    axes[1].plot(qs, rc, "s-", color="tab:red",  label="q_raw / q_stove (cap bite)")
    axes[1].set_xlabel("q_stove [kW/m^2]")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("(b) Validation vs cap-bite across the sweep")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Phase 3.2 q-sweep — steel 304, dT_w measured at fluid-contact face")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nplot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
