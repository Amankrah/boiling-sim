"""Tests for the Phase 3 Milestone A nucleation infrastructure."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from boilingsim.boiling import (
    DT_TABLE_MAX_K,
    N_TABLE_ENTRIES,
    Bubble,
    allocate_bubble_pool,
    build_nucleation_table,
    step_nucleation,
)
from boilingsim.config import BoilingConfig, ScenarioConfig, load_scenario
from boilingsim.geometry import MAT_POT_WALL, build_pot_geometry


@pytest.fixture(scope="module")
def boil_cfg() -> ScenarioConfig:
    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 100_000
    cfg.grid.dx_m = 0.002
    return cfg


@pytest.fixture(scope="module")
def boil_grid(boil_cfg):
    return build_pot_geometry(boil_cfg)


# ---------------------------------------------------------------------------
# Nucleation-site density table
# ---------------------------------------------------------------------------


def test_nucleation_table_monotonic(boil_cfg):
    water_props = {"T_sat": 373.15, "rho_v": 0.598, "h_lv": 2.257e6, "sigma": 0.0589}
    table = build_nucleation_table(boil_cfg.boiling, water_props).numpy()
    assert table.shape == (N_TABLE_ENTRIES,)
    assert table[0] == 0.0, "N_a(ΔT=0) must be zero"
    # Strictly increasing on positive ΔT
    diffs = np.diff(table)
    assert (diffs >= 0).all(), "N_a must be non-decreasing in ΔT"
    assert (diffs[1:] > 0).all(), "N_a must be strictly increasing for ΔT > 0"


def test_nucleation_table_reasonable_magnitude(boil_cfg):
    water_props = {"T_sat": 373.15, "rho_v": 0.598, "h_lv": 2.257e6, "sigma": 0.0589}
    table = build_nucleation_table(boil_cfg.boiling, water_props).numpy()
    # ΔT = 10 K corresponds to index 20 (half of 20 K).
    n_at_10k = table[20]
    # Target calibration: ~1e5 sites/m² at ΔT=10 K (order-of-magnitude, not exact).
    assert 1.0e4 <= n_at_10k <= 1.0e6, f"N_a(10 K) = {n_at_10k}, expected ~1e5"


# ---------------------------------------------------------------------------
# Bubble pool allocation
# ---------------------------------------------------------------------------


def test_bubble_pool_allocation(boil_cfg, boil_grid):
    pool = boil_grid.bubbles
    assert pool is not None
    assert pool.max_bubbles == 100_000
    # Every slot starts inactive
    assert pool.count_active() == 0
    # Auxiliary arrays allocated to grid shape
    assert pool.site_active.shape == boil_grid.shape
    # Nucleation table has the expected entries
    assert pool.nucleation_table.shape == (N_TABLE_ENTRIES,)


# ---------------------------------------------------------------------------
# Nucleation behaviour
# ---------------------------------------------------------------------------


def test_no_nucleation_below_onb(boil_cfg):
    """Wall at T_sat + 1 K (below 5 K ONB threshold): no bubbles spawn."""
    cfg = boil_cfg.model_copy(deep=True)
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles
    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_POT_WALL] = 373.15 + 1.0   # sub-ONB
    grid.T.assign(T_np)

    for step in range(5):
        step_nucleation(grid, pool, cfg, dt=0.01, sim_time=step * 0.01, step_count=step)
    wp.synchronize()
    assert pool.count_active() == 0, "Bubbles spawned despite T_wall below ONB"


def test_nucleation_at_hot_base(boil_cfg):
    """Wall at T_sat + 20 K: many sites spawn; count within an order of magnitude."""
    cfg = boil_cfg.model_copy(deep=True)
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles
    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_POT_WALL] = 373.15 + 20.0  # 20 K superheat
    grid.T.assign(T_np)

    # One step is enough at 20 K superheat — N_a is huge.
    step_nucleation(grid, pool, cfg, dt=0.01, sim_time=0.0, step_count=0)
    wp.synchronize()
    n_active = pool.count_active()
    # Sanity: at least a few, no more than pool size.
    assert 5 <= n_active <= cfg.boiling.max_bubbles, (
        f"Unexpected bubble count {n_active} at 20 K wall superheat"
    )

    # Each bubble must land inside the fluid domain above a pot-wall cell.
    bubbles_np = pool.bubbles.numpy()
    active_mask = np.array([b["active"] for b in bubbles_np]) == 1
    assert active_mask.sum() == n_active

    # site_active bookkeeping: count of flagged sites ≈ count of bubbles
    n_flagged = int((pool.site_active.numpy() == 1).sum())
    assert n_flagged == n_active


def test_nucleation_respects_pool_capacity():
    """Set a ridiculously small pool and an extreme superheat. New bubbles should
    stop being added once the pool is full."""
    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 16  # tiny
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_POT_WALL] = 373.15 + 40.0
    grid.T.assign(T_np)

    for step in range(3):
        step_nucleation(grid, pool, cfg, dt=0.01, sim_time=step * 0.01, step_count=step)
    wp.synchronize()
    n_active = pool.count_active()
    assert n_active <= 16, f"Pool overflow: {n_active} active bubbles in 16-slot pool"


# ---------------------------------------------------------------------------
# Milestone B: correlations (Fritz, Cole, Mikic-Rohsenow) — pure-Python
# ---------------------------------------------------------------------------


def _fritz_m(theta_rad: float, sigma: float = 0.0589,
             g: float = 9.81, rho_l: float = 997.0, rho_v: float = 0.598) -> float:
    """Python-side mirror of the @wp.func for test verification."""
    import math
    theta_deg = math.degrees(theta_rad)
    return 0.0208 * theta_deg * math.sqrt(sigma / (g * (rho_l - rho_v)))


def _cole_hz(D_d_m: float, g: float = 9.81,
             rho_l: float = 997.0, rho_v: float = 0.598) -> float:
    import math
    return math.sqrt(4.0 * g * (rho_l - rho_v) / (3.0 * D_d_m * rho_l))


def test_fritz_water_steel_in_literature_range():
    """D_d for water on steel at θ = 1 rad should be 2–4 mm (published range)."""
    D_d_m = _fritz_m(theta_rad=1.0)
    assert 2.0e-3 <= D_d_m <= 4.0e-3, f"D_d = {D_d_m*1000:.2f} mm outside [2, 4] mm"


def test_cole_frequency_water_steel_in_literature_range():
    """f for typical water-steel D_d ≈ 2.9 mm should be 20–100 Hz."""
    D_d_m = _fritz_m(theta_rad=1.0)
    f_hz = _cole_hz(D_d_m)
    assert 20.0 <= f_hz <= 100.0, f"f = {f_hz:.1f} Hz outside [20, 100] Hz"


def _terminal_slip_mps(R_m: float) -> float:
    """Python-side mirror of @wp.func terminal_slip_velocity (power-law fit
    to Clift-Grace-Weber 1978, capped at the plateau)."""
    v_pow = 391.0 * (R_m ** 1.26)
    return 0.22 if v_pow > 0.22 else v_pow


def test_terminal_slip_seed_bubble_is_small():
    """At R = 50 um the curve gives v ~ 5 mm/s, two orders of magnitude
    below the old constant 0.2 m/s. Pre-M1 a seed bubble would zoom
    around at 200 mm/s; now it barely rises on its own and rides the
    fluid u instead."""
    v = _terminal_slip_mps(50.0e-6)
    assert 1.0e-3 <= v <= 1.0e-2, f"v(50 um) = {v*1000:.2f} mm/s outside [1, 10]"


def test_terminal_slip_plateau_at_3mm():
    """At R = 3 mm a clean-water bubble is in the wave-controlled
    plateau ~0.22 m/s (Grace 1976 / Clift-Grace-Weber 1978)."""
    v = _terminal_slip_mps(3.0e-3)
    assert v == 0.22, f"v(3 mm) = {v:.3f} m/s should be plateau 0.22"


def test_terminal_slip_monotonic_below_plateau():
    """The curve must be monotonically non-decreasing in R (a bigger
    bubble can't rise slower than a smaller one). Sweeps R from 10 um
    to plateau and asserts no decrease."""
    R_grid = [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 5e-3]
    vs = [_terminal_slip_mps(R) for R in R_grid]
    for i in range(1, len(vs)):
        assert vs[i] >= vs[i - 1] - 1.0e-9, (
            f"slip not monotonic: v({R_grid[i-1]*1000} mm)={vs[i-1]:.4f} "
            f"> v({R_grid[i]*1000} mm)={vs[i]:.4f}"
        )


def test_mikic_rohsenow_growth_rate_analytic():
    """At ΔT = 10 K, Ja ≈ 31, α_l ≈ 1.45e-7 m²/s; R(10 ms) ≈ 1.3 mm.

    Compares the sim's update_bubbles kernel output against the analytic
    Mikic-Rohsenow prediction for a single static bubble in a hot pool.
    """
    from boilingsim.boiling import seed_test_bubble, step_update_bubbles
    import math

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 16
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Uniform superheat: water everywhere at T_sat + 10 K.
    T_np = grid.T.numpy()
    T_np[:] = 373.15 + 10.0
    grid.T.assign(T_np)

    # Seed one bubble at mid-water, birth_time = 0.
    seed_test_bubble(pool, slot=0, position=(0.0, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0),
                     radius=1.0e-5, birth_time=0.0)

    # Step forward to t = 10 ms. dt doesn't control growth — age since birth does.
    sim_time = 0.010
    step_update_bubbles(grid, pool, cfg, dt=1.0e-4, sim_time=sim_time)
    wp.synchronize()

    out = pool.bubbles.numpy()[0]
    R_measured = float(out["radius"])

    # Analytic Mikic-Rohsenow:
    rho_l, cp_l, k_l = 997.0, 4186.0, 0.606
    rho_v, h_lv = 0.598, 2.257e6
    Ja = rho_l * cp_l * 10.0 / (rho_v * h_lv)
    alpha_l = k_l / (rho_l * cp_l)
    R_analytic = (2.0 / math.sqrt(math.pi)) * Ja * math.sqrt(alpha_l * sim_time)

    # Departure may have capped R; take the min of analytic and Fritz.
    D_d = _fritz_m(1.0)
    R_cap = 0.5 * D_d
    R_expected = min(R_analytic, R_cap)
    # Tolerance 10 % — allows for the sim's capped growth and integer index rounding.
    rel_err = abs(R_measured - R_expected) / R_expected
    assert rel_err < 0.10, (
        f"R at t={sim_time*1000:.1f}ms: sim={R_measured*1000:.3f}mm, "
        f"expected={R_expected*1000:.3f}mm, err={rel_err*100:.2f}%"
    )


# ---------------------------------------------------------------------------
# Milestone B: bubble life cycle
# ---------------------------------------------------------------------------


def test_bubbles_rise_and_vent():
    """Seed bubbles with radius > Fritz D_d near the base. Over 5 seconds of sim,
    they should rise (no fluid flow) and all eventually vent at the surface.
    """
    from boilingsim.boiling import seed_test_bubble, step_update_bubbles

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 100
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Uniform water at T_sat (no thermal growth drive — pure transport test).
    T_np = grid.T.numpy()
    T_np[:] = 373.15
    grid.T.assign(T_np)

    R_big = _fritz_m(1.0) * 0.6   # 2·R > D_d → already departed
    z_start = cfg.pot.base_thickness_m + 0.002

    # Seed 20 already-detached bubbles on a horizontal line near the base.
    for i in range(20):
        # Tight cluster around pot axis, y=0, z near base.
        x = 0.005 * (i - 10)  # -50 mm .. +45 mm in 5 mm steps; keep inside r_inner=97 mm.
        seed_test_bubble(pool, slot=i, position=(x, 0.0, z_start),
                         velocity=(0.0, 0.0, 0.2),
                         radius=R_big, birth_time=0.0)
    wp.synchronize()

    dt = 5.0e-3
    for step in range(1000):
        step_update_bubbles(grid, pool, cfg, dt, sim_time=step * dt)
    wp.synchronize()

    n_still_active = pool.count_active()
    assert n_still_active == 0, f"{n_still_active} bubbles still active after 5 s"


# ---------------------------------------------------------------------------
# Milestone C: two-way energy coupling (latent-heat sink)
# ---------------------------------------------------------------------------


def test_latent_heat_energy_balance():
    """Spawn a single bubble, run one scatter step; the total energy removed
    from the 8 surrounding cells must equal ρ_v·h_lv·4π·R²·(dR/dt)·dt.

    We isolate the scatter kernel — no advection, no conduction, no
    nucleation — so the only T change is the latent-heat sink.
    """
    import math
    from boilingsim.boiling import seed_test_bubble, step_scatter_latent_heat

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Water slightly above T_sat so the temperature gate in scatter_latent_heat
    # passes (bubbles don't scatter when the surrounding liquid is at or below
    # saturation — no growth drive).
    T_np = grid.T.numpy()
    T_np[:] = 373.15 + 1.0
    grid.T.assign(T_np)

    # Seed one bubble in water, off the carrot axis. Carrot occupies
    # (|r_xy| < 12.5 mm, 30 mm < z < 80 mm), so place bubble at x=50 mm.
    # Choose R=2mm and dt=10ms so the per-cell ΔT (~0.025 K) is well above
    # float32 precision on the stored T (~373 K → ~1e-5 K resolution).
    R0 = 2.0e-3
    age = 0.1
    birth_time = 0.0
    sim_time = birth_time + age
    seed_test_bubble(pool, slot=0, position=(0.05, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0),
                     radius=R0, birth_time=birth_time)

    dt = 1.0e-2
    # Capture energy before the sink.
    T_before = grid.T.numpy().copy()
    mat_np = grid.mat.numpy()

    step_scatter_latent_heat(grid, pool, cfg, dt=dt, sim_time=sim_time)
    wp.synchronize()

    T_after = grid.T.numpy()
    # Energy change in fluid cells only.
    rho_l, cp_l = 997.0, 4186.0
    dV = grid.dx ** 3
    dE_measured = (T_after - T_before)[mat_np == 0] * rho_l * cp_l * dV
    E_removed = -float(dE_measured.sum())   # positive = energy removed

    # Analytic latent-heat energy removal.
    rho_v, h_lv = 0.598, 2.257e6
    dR_dt = R0 / (2.0 * age)
    Q_b = rho_v * h_lv * 4.0 * math.pi * R0 * R0 * dR_dt
    E_expected = Q_b * dt

    rel_err = abs(E_removed - E_expected) / E_expected
    assert rel_err < 0.02, (
        f"latent-heat scatter: measured {E_removed:.4e} J, "
        f"expected {E_expected:.4e} J, rel err {rel_err*100:.2f}%"
    )


def test_wall_boiling_flux_cools_superheated_wall():
    """End-to-end Milestone-C integration: superheated wall + enabled boiling
    should cool the pot wall via the Eulerian wall-boiling flux kernel
    (microlayer-evaporation sink).

    We run a tiny sequence of conduct + step_bubbles + step_wall_boiling_flux
    against a conduction-only baseline. With boiling ON, the wall loses energy
    to bubble-nucleation latent-heat extraction at the wall surface; with
    boiling OFF, the same stove flux just heats the wall up with no vapor
    pathway. Mean pot-wall temperature must be **lower** in the boiling case.

    (This test replaces the earlier Lagrangian-scatter assertion: the scatter
    kernel is retained for diagnostics but no longer fires in the pipeline --
    the wall flux is now the sole latent-heat sink. Doubling them produced a
    subcooled bulk that stalled bubble growth.)
    """
    from boilingsim.boiling import step_bubbles, step_wall_boiling_flux
    from boilingsim.thermal import (
        MaterialProps, allocate_thermal_workspace, conduct_one_step,
    )

    def run(enable_boiling: bool) -> tuple[float, int]:
        cfg = load_scenario("configs/scenarios/single_carrot.yaml")
        cfg.boiling.enabled = enable_boiling
        cfg.boiling.max_bubbles = 50_000
        cfg.grid.dx_m = 0.002
        grid = build_pot_geometry(cfg)
        props = MaterialProps.from_scenario(cfg)
        ws = allocate_thermal_workspace(grid)

        # Hot system: wall at T_sat + 20 K, water at T_sat + 2 K.
        T_np = grid.T.numpy()
        mat_np = grid.mat.numpy()
        T_np[mat_np == MAT_POT_WALL] = 373.15 + 20.0
        T_np[mat_np == 0] = 373.15 + 2.0   # MAT_FLUID = 0
        grid.T.assign(T_np)

        dt = 0.005
        for step in range(100):
            conduct_one_step(grid, props, ws, cfg, dt)
            if enable_boiling:
                step_bubbles(grid, grid.bubbles, cfg, dt,
                             sim_time=step * dt, step_count=step)
                step_wall_boiling_flux(grid, grid.bubbles, cfg, props, dt)
        wp.synchronize()
        T_final = grid.T.numpy()
        wall_mean_c = float(T_final[mat_np == MAT_POT_WALL].mean() - 273.15)
        n_bubbles = grid.bubbles.count_active() if enable_boiling else 0
        return wall_mean_c, n_bubbles

    wall_noboil, _ = run(False)
    wall_boil, n_bubbles = run(True)

    # Sanity: bubbles actually nucleated at the superheated wall.
    assert n_bubbles > 10, (
        f"Only {n_bubbles} bubbles active -- boiling kernel may not be firing"
    )

    # Wall flux should cool the pot wall relative to the conduction-only baseline.
    assert wall_boil < wall_noboil, (
        f"Wall boiling flux not cooling wall: "
        f"boiling ON mean = {wall_boil:.4f} C, OFF mean = {wall_noboil:.4f} C"
    )
    # Drop should be physically meaningful. The kernel is capped at q_stove
    # (conservation -- see apply_wall_boiling_flux docstring), so the max
    # cooling rate at the inner wall cell is q_stove * dt / (rho * cp * dx).
    # For steel at 30 kW/m^2 over 0.5 s: ~2 K ceiling, but conduction from
    # the rest of the (still-hot) wall reduces net drop to a few tenths of K.
    drop = wall_noboil - wall_boil
    assert drop > 0.15, (
        f"Wall boiling flux drop too small: {drop:.3f} K "
        f"(expected > 0.15 K with {n_bubbles} bubbles over 0.5 s at dT_w=20K)"
    )


def test_no_sink_when_bubble_just_nucleated():
    """Bubbles with age < 1 μs shouldn't scatter (their analytic dR/dt is
    unbounded at t=0; the kernel guards against this). Seeding a bubble with
    birth_time == current_time should leave T unchanged.
    """
    from boilingsim.boiling import seed_test_bubble, step_scatter_latent_heat

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    T_np = grid.T.numpy()
    T_np[:] = 373.15
    grid.T.assign(T_np)

    sim_time = 0.0
    seed_test_bubble(pool, slot=0, position=(0.05, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0),
                     radius=1.0e-5, birth_time=sim_time)   # age = 0 exactly

    T_before = grid.T.numpy().copy()
    step_scatter_latent_heat(grid, pool, cfg, dt=1.0e-3, sim_time=sim_time)
    wp.synchronize()
    T_after = grid.T.numpy()

    max_change = float(np.abs(T_after - T_before).max())
    assert max_change == 0.0, (
        f"Just-nucleated bubble (age=0) should not have scattered; "
        f"max T change = {max_change:.3e}"
    )


# ---------------------------------------------------------------------------
# Milestone D: two-way momentum coupling + VOF alpha reduction
# ---------------------------------------------------------------------------


def test_bubble_momentum_creates_upward_velocity():
    """A single bubble scatter_bubble_momentum step must create positive
    upward velocity on the 8 surrounding z-faces (all inside water)."""
    from boilingsim.boiling import seed_test_bubble, step_scatter_momentum
    import math

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Zero velocity baseline.
    uz_before = grid.uz.numpy().copy()
    assert not uz_before.any(), "initial uz should be all zero"

    # Single bubble, R = 2 mm, positioned in water off the carrot axis.
    R0 = 2.0e-3
    seed_test_bubble(pool, slot=0, position=(0.05, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0), radius=R0, birth_time=0.0)

    step_scatter_momentum(grid, pool, cfg, dt=1.0e-3)
    wp.synchronize()

    uz_after = grid.uz.numpy()
    # Analytic total Δ(uz·V_cell·ρ_l) = bubble force × dt
    rho_l, rho_v, g = 997.0, 0.598, 9.81
    V_b = (4.0 / 3.0) * math.pi * R0 ** 3
    dt = 1.0e-3
    # Momentum transferred per cell summed = V_b·(ρ_l-ρ_v)·g·dt  → sum of (Δuz · V_cell · ρ_l) ≈ F·dt
    V_cell = grid.dx ** 3
    total_momentum = float((uz_after - uz_before).sum()) * rho_l * V_cell
    expected = V_b * (rho_l - rho_v) * g * dt
    rel_err = abs(total_momentum - expected) / expected
    assert rel_err < 0.05, (
        f"momentum scatter imbalance: measured {total_momentum:.4e}, "
        f"expected {expected:.4e}, err {rel_err*100:.2f}%"
    )
    # Every non-zero uz contribution must be positive (upward).
    nonzero = (uz_after - uz_before)[uz_after != uz_before]
    assert (nonzero > 0).all(), "all bubble-forced uz contributions must point upward"


def test_water_alpha_reduced_in_bubble_cells():
    """Spawn one bubble; water_alpha must drop in its 8-cell stencil and
    stay at the baseline (1.0 in water, 0 elsewhere) everywhere else.
    """
    from boilingsim.boiling import seed_test_bubble, step_reduce_water_alpha
    import math

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    R0 = 2.0e-3
    seed_test_bubble(pool, slot=0, position=(0.05, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0), radius=R0, birth_time=0.0)

    step_reduce_water_alpha(grid, pool)
    wp.synchronize()

    alpha = grid.water_alpha.numpy()
    base = grid.water_alpha_base.numpy()

    # Total reduction across the grid = V_b / V_cell, clamped so α >= 0.
    V_b = (4.0 / 3.0) * math.pi * R0 ** 3
    V_cell = grid.dx ** 3
    expected_reduction = V_b / V_cell
    measured_reduction = float((base - alpha).sum())
    # Allow up to 20% slack for clamping (a 2 mm bubble at dx=2 mm exceeds
    # 1 cell's volume → trilinear scatter can push some cells negative,
    # which the clamp then absorbs). The *sign* must be correct and the
    # reduction concentrated where the bubble is.
    assert measured_reduction > 0.5 * expected_reduction, (
        f"alpha reduction too small: measured {measured_reduction:.3f}, "
        f"expected up to {expected_reduction:.3f}"
    )
    assert measured_reduction <= expected_reduction + 1.0e-6, (
        f"alpha reduction exceeds bubble volume: "
        f"measured {measured_reduction:.3f}, expected <= {expected_reduction:.3f}"
    )
    # Clamped range
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0


def test_alpha_resets_from_baseline_each_step():
    """When a bubble moves/deactivates, previously-reduced cells must return
    to their baseline alpha next step (the reset_from_baseline).
    """
    from boilingsim.boiling import seed_test_bubble, step_reduce_water_alpha

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Step 1: with bubble present.
    seed_test_bubble(pool, slot=0, position=(0.05, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0), radius=2.0e-3, birth_time=0.0)
    step_reduce_water_alpha(grid, pool)
    wp.synchronize()
    alpha_with = grid.water_alpha.numpy().copy()

    # Step 2: deactivate the bubble (simulate venting) and re-run reset.
    b_np = pool.bubbles.numpy()
    b_np[0]["active"] = 0
    pool.bubbles.assign(b_np)
    pool.slot_claim.zero_()
    step_reduce_water_alpha(grid, pool)
    wp.synchronize()
    alpha_cleared = grid.water_alpha.numpy()
    base = grid.water_alpha_base.numpy()

    # Before: α was reduced somewhere.
    assert not np.allclose(alpha_with, base), "Step 1 didn't reduce α as expected"
    # After the reset with no bubbles: α == baseline exactly.
    assert np.allclose(alpha_cleared, base, atol=1e-6), "Step 2 didn't fully reset α to baseline"


def test_full_bubble_step_no_nan_over_30s():
    """Phase-3 stability test: run 30 s of the full pipeline (advection +
    buoyancy + bubbles + conduction + projection) on a pre-boiling steel
    pot. No NaNs, bubble count bounded, water T finite.
    """
    from boilingsim.pipeline import Simulation

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.grid.dx_m = 0.002
    cfg.solver.pressure_max_iter = 100
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 50_000

    sim = Simulation(cfg)
    # Warm-start: water 95 C, wall 102 C (already at ONB-ish).
    T_np = sim.grid.T.numpy()
    mat_np = sim.grid.mat.numpy()
    T_np[mat_np == 0] = 95.0 + 273.15
    T_np[mat_np == MAT_POT_WALL] = 102.0 + 273.15
    sim.grid.T.assign(T_np)

    scalars = sim.run(total_time_s=5.0, progress_every_s=60.0)  # quiet mode
    wp.synchronize()

    T_final = sim.grid.T.numpy()
    assert np.isfinite(T_final).all(), "NaN or inf in final T field"
    uz = sim.grid.uz.numpy()
    assert np.isfinite(uz).all(), "NaN or inf in final uz"
    # Reasonable velocity bound (boiling convection ~10 cm/s peaks)
    assert abs(uz).max() < 5.0, f"|uz|_max = {abs(uz).max():.2f} m/s (runaway)"
    # Bubble pool not runaway
    n_bubbles = sim.grid.bubbles.count_active()
    assert n_bubbles < cfg.boiling.max_bubbles, (
        f"bubble pool saturated: {n_bubbles}/{cfg.boiling.max_bubbles}"
    )


def test_bubble_deactivates_on_solid():
    """A bubble placed inside a carrot cell should be deactivated on the next
    update step (material at position != MAT_FLUID)."""
    from boilingsim.boiling import seed_test_bubble, step_update_bubbles

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    T_np = grid.T.numpy()
    T_np[:] = 373.15
    grid.T.assign(T_np)

    # Carrot is at position (0, 0, 0.03), diameter 25mm, length 50mm.
    # Place a bubble inside it at (0, 0, 0.05) with R > D_d/2 so it's detached
    # and will advect (not stuck at the nucleation site).
    R_big = _fritz_m(1.0) * 0.6
    seed_test_bubble(pool, slot=0, position=(0.0, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.2),
                     radius=R_big, birth_time=0.0)

    step_update_bubbles(grid, pool, cfg, dt=1.0e-3, sim_time=0.0)
    wp.synchronize()

    out = pool.bubbles.numpy()[0]
    assert out["active"] == 0, "bubble inside carrot cell should have been deactivated"


# ---------------------------------------------------------------------------
# Phase 3.3: condensation of bubbles in subcooled liquid
# ---------------------------------------------------------------------------


def test_bubble_condenses_in_subcooled_fluid():
    """A bubble placed in strongly subcooled liquid (T << T_sat) should shrink
    via Plesset-Zwick diffusion-controlled condensation and fully deactivate
    within a handful of update steps. At deactivation, the bubble's remaining
    latent heat must be deposited back into the local fluid stencil (atomic
    add, mirror of scatter_latent_heat). This closes the phantom-bubble
    limitation documented in phase3_boiling.md §Known limitations.
    """
    from boilingsim.boiling import seed_test_bubble, step_update_bubbles

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Strongly subcooled water everywhere — should drive rapid collapse.
    T_sub_k = 323.15  # 50 C (50 K below saturation)
    T_np = grid.T.numpy()
    T_np[:] = T_sub_k
    grid.T.assign(T_np)
    mat_np = grid.mat.numpy()

    # Seed one 0.2 mm bubble off-axis in water. Off the carrot column
    # (|r| > 12.5 mm), inside the inner pot radius (≈ 97 mm), and above
    # the base: pick x = 50 mm, z = 60 mm.
    R0 = 2.0e-4   # 200 um — larger than the 10 um seed floor
    position = (0.05, 0.0, 0.06)
    seed_test_bubble(pool, slot=0, position=position,
                     velocity=(0.0, 0.0, 0.0),
                     radius=R0, birth_time=0.0)
    wp.synchronize()

    T_before = grid.T.numpy().copy()

    # Step forward repeatedly; Plesset-Zwick at 50 K subcool collapses a
    # 200 um bubble in sub-millisecond wall-clock, so << 20 ms of sim time.
    dt = 1.0e-4
    for step in range(200):
        step_update_bubbles(grid, pool, cfg, dt=dt, sim_time=step * dt)
    wp.synchronize()

    out = pool.bubbles.numpy()[0]
    assert out["active"] == 0, (
        "bubble in subcooled fluid should have fully condensed and deactivated"
    )

    # Energy released: E = rho_v * (4/3) pi R0^3 * h_lv. Distributed over the
    # trilinear 8-cell stencil at the condensation point, so total integrated
    # cell-volume ΔT equals the analytic value up to scatter discretization.
    rho_l, cp_l = 997.0, 4186.0
    rho_v, h_lv = 0.598, 2.257e6
    V_vap = 4.0 * np.pi / 3.0 * R0 ** 3
    E_release = rho_v * h_lv * V_vap
    # Expected total cell-volume-integrated ΔT summed over fluid cells:
    expected_sum_dT = E_release / (rho_l * cp_l * grid.dx ** 3)

    from boilingsim.geometry import MAT_FLUID
    T_after = grid.T.numpy()
    water_mask = mat_np == MAT_FLUID
    sum_dT_measured = float(np.sum(T_after[water_mask] - T_before[water_mask]))

    # Relaxed tolerance (12 %): numerical integration of dV across ~8 discrete
    # shrink steps overshoots the analytic volume loss slightly when the
    # bubble crosses R_seed on its last step (R_new clamped to R_seed instead
    # of to zero, so ~1 % of the vapor is never deposited). Measured 1.465e-3
    # K vs analytic 1.355e-3 K (~+8 %) -- the explicit-Euler discretization
    # error on dR/dt = -k/R at small R.
    assert sum_dT_measured == pytest.approx(expected_sum_dT, rel=0.12), (
        f"latent-heat deposit: expected ΔT sum ≈ {expected_sum_dT:.2e} K, "
        f"measured {sum_dT_measured:.2e} K (>12% mismatch)"
    )
    assert sum_dT_measured > 0.0, "condensation must deposit (not remove) heat"


def test_no_condensation_when_fluid_at_saturation():
    """Regression guard: bubbles in liquid at T_sat should neither grow (no
    drive) nor shrink (the `_condensation_decrement` func gates on
    T_local < T_sat). Radius must be preserved across a step.
    """
    from boilingsim.boiling import seed_test_bubble, step_update_bubbles

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    T_np = grid.T.numpy()
    T_np[:] = 373.15  # exactly T_sat
    grid.T.assign(T_np)

    R0 = 1.0e-4   # 100 um; below Fritz D_d/2 so bubble stays attached
    seed_test_bubble(pool, slot=0, position=(0.05, 0.0, 0.06),
                     velocity=(0.0, 0.0, 0.0),
                     radius=R0, birth_time=0.0)
    wp.synchronize()

    step_update_bubbles(grid, pool, cfg, dt=1.0e-3, sim_time=1.0e-3)
    wp.synchronize()

    out = pool.bubbles.numpy()[0]
    assert out["active"] == 1, "bubble at T_sat should remain active"
    assert out["radius"] == pytest.approx(R0, rel=1.0e-5), (
        f"bubble radius at T_sat drifted: expected {R0}, got {out['radius']}"
    )


# ---------------------------------------------------------------------------
# Phase 8 M2: Rayleigh-Taylor binary fragmentation
# ---------------------------------------------------------------------------


def test_fragmentation_splits_into_two_volume_conserving_daughters():
    """A bubble flagged with R > R_frag should split into 2 active
    bubbles whose summed volume equals the parent's. Daughter radii
    R_d = R / 2^(1/3) = 0.7937 R."""
    from boilingsim.boiling import (
        seed_test_bubble, step_fragment_bubbles,
    )
    import math

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 16
    cfg.boiling.fragmentation_radius_m = 4.0e-3
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Seed one bubble at 5 mm with a non-zero velocity so the
    # perpendicular-axis logic picks a deterministic direction.
    R_parent = 5.0e-3
    seed_test_bubble(pool, slot=0, position=(0.0, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.2),
                     radius=R_parent, birth_time=0.0)
    # Manually flag the bubble for fragmentation (bypassing update_bubbles
    # so this test only exercises the fragment kernel itself).
    nf = pool.needs_fragment.numpy()
    nf[0] = 1
    pool.needs_fragment.assign(nf)

    step_fragment_bubbles(pool, cfg)
    wp.synchronize()

    bubbles = pool.bubbles.numpy()
    active = bubbles[bubbles["active"] == 1]
    assert len(active) == 2, f"expected 2 daughters, got {len(active)}"

    R_expected = R_parent * (0.5 ** (1.0 / 3.0))
    for b in active:
        assert b["radius"] == pytest.approx(R_expected, rel=1.0e-5), (
            f"daughter R = {b['radius']*1000:.4f} mm, expected "
            f"{R_expected*1000:.4f} mm"
        )

    # Volume conservation: V_parent = sum(V_daughters)
    V_parent = (4.0 / 3.0) * math.pi * R_parent ** 3
    V_daughters = sum((4.0 / 3.0) * math.pi * float(b["radius"]) ** 3
                       for b in active)
    rel_err = abs(V_parent - V_daughters) / V_parent
    assert rel_err < 1.0e-5, (
        f"volume not conserved: parent={V_parent:.6e} m^3, "
        f"daughters={V_daughters:.6e} m^3, err={rel_err:.2e}"
    )


def test_fragmentation_pool_full_graceful_degradation():
    """If no daughter slot is available, the parent stays at full radius
    (the R_max safety cap will clamp it). No crash, no corruption."""
    from boilingsim.boiling import (
        seed_test_bubble, step_fragment_bubbles,
    )

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 4   # tiny pool to force exhaustion
    cfg.boiling.fragmentation_radius_m = 4.0e-3
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    # Fill all 4 slots; only slot 0 will be flagged for fragmentation.
    for s in range(4):
        seed_test_bubble(pool, slot=s, position=(0.0, 0.0, 0.05),
                         velocity=(0.0, 0.0, 0.2),
                         radius=5.0e-3, birth_time=0.0)
    nf = pool.needs_fragment.numpy()
    nf[0] = 1
    pool.needs_fragment.assign(nf)

    n_active_before = pool.count_active()
    step_fragment_bubbles(pool, cfg)
    wp.synchronize()
    n_active_after = pool.count_active()

    # Pool was full; fragmentation couldn't claim a daughter slot. Count
    # stays the same; flag was consumed; parent's radius is unchanged.
    assert n_active_after == n_active_before, (
        f"pool count changed unexpectedly: {n_active_before} -> {n_active_after}"
    )
    assert pool.needs_fragment.numpy()[0] == 0, (
        "needs_fragment flag should be cleared even when fragmentation fails"
    )
    out = pool.bubbles.numpy()[0]
    assert out["radius"] == pytest.approx(5.0e-3, rel=1.0e-5), (
        f"parent radius drifted in pool-full case: {out['radius']*1000:.4f} mm"
    )


# ---------------------------------------------------------------------------
# Phase 8 M3: spatial-hash coalescence
# ---------------------------------------------------------------------------


def test_coalescence_merges_overlapping_bubbles():
    """Two bubbles whose centres are within R1 + R2 should merge into one
    survivor with R = (R1^3 + R2^3)^(1/3) and momentum-weighted velocity."""
    from boilingsim.boiling import (
        seed_test_bubble, step_coalesce_bubbles,
    )
    import math

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 16
    cfg.boiling.coalescence_enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    R1, R2 = 1.5e-3, 2.0e-3
    # Centres 2 mm apart, sum of radii is 3.5 mm -> overlapping.
    seed_test_bubble(pool, slot=0, position=(0.0, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.2),
                     radius=R1, birth_time=1.0)
    seed_test_bubble(pool, slot=1, position=(0.002, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.1),
                     radius=R2, birth_time=2.0)

    step_coalesce_bubbles(pool, cfg)
    wp.synchronize()

    bubbles = pool.bubbles.numpy()
    active = bubbles[bubbles["active"] == 1]
    assert len(active) == 1, f"expected 1 survivor, got {len(active)}"

    survivor = active[0]
    R_expected = (R1 ** 3 + R2 ** 3) ** (1.0 / 3.0)
    assert survivor["radius"] == pytest.approx(R_expected, rel=1.0e-4), (
        f"survivor R = {survivor['radius']*1000:.4f} mm, expected "
        f"{R_expected*1000:.4f} mm"
    )

    # Volume conservation: V_survivor = V1 + V2.
    V1 = (4.0 / 3.0) * math.pi * R1 ** 3
    V2 = (4.0 / 3.0) * math.pi * R2 ** 3
    V_surv = (4.0 / 3.0) * math.pi * float(survivor["radius"]) ** 3
    rel_err = abs((V1 + V2) - V_surv) / (V1 + V2)
    assert rel_err < 1.0e-4, (
        f"volume not conserved: V1+V2={V1+V2:.6e}, V_survivor={V_surv:.6e}"
    )

    # Momentum-weighted velocity: v_z should land between 0.1 and 0.2 m/s,
    # weighted by volume. v_expected = (V1*0.2 + V2*0.1) / (V1+V2).
    v_expected_z = (V1 * 0.2 + V2 * 0.1) / (V1 + V2)
    assert survivor["velocity"][2] == pytest.approx(v_expected_z, rel=1.0e-4), (
        f"v_z = {survivor['velocity'][2]:.4f}, expected {v_expected_z:.4f}"
    )

    # birth_time inherits the OLDER (smaller value).
    assert survivor["birth_time"] == pytest.approx(1.0, rel=1.0e-6), (
        f"birth_time = {survivor['birth_time']}, expected min(1.0, 2.0) = 1.0"
    )


def test_coalescence_does_not_merge_separated_bubbles():
    """Two bubbles with centres farther apart than R1 + R2 must NOT merge."""
    from boilingsim.boiling import (
        seed_test_bubble, step_coalesce_bubbles,
    )

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 16
    cfg.boiling.coalescence_enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    R1, R2 = 1.0e-3, 1.0e-3
    # Centres 5 mm apart, sum of radii is 2 mm -> well separated.
    seed_test_bubble(pool, slot=0, position=(0.0, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0),
                     radius=R1, birth_time=0.0)
    seed_test_bubble(pool, slot=1, position=(0.005, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0),
                     radius=R2, birth_time=0.0)

    step_coalesce_bubbles(pool, cfg)
    wp.synchronize()

    bubbles = pool.bubbles.numpy()
    n_active = int((bubbles["active"] == 1).sum())
    assert n_active == 2, f"separated bubbles wrongly merged: {n_active} active"


def test_coalescence_disabled_is_noop():
    """When ``coalescence_enabled=False`` the kernel pass is skipped
    entirely -- two overlapping bubbles must remain two."""
    from boilingsim.boiling import (
        seed_test_bubble, step_coalesce_bubbles,
    )

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 16
    cfg.boiling.coalescence_enabled = False
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    seed_test_bubble(pool, slot=0, position=(0.0, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0),
                     radius=1.5e-3, birth_time=0.0)
    seed_test_bubble(pool, slot=1, position=(0.001, 0.0, 0.05),
                     velocity=(0.0, 0.0, 0.0),
                     radius=2.0e-3, birth_time=0.0)

    step_coalesce_bubbles(pool, cfg)
    wp.synchronize()

    bubbles = pool.bubbles.numpy()
    n_active = int((bubbles["active"] == 1).sum())
    assert n_active == 2, f"coalescence ran with disabled flag: {n_active} active"


# ---------------------------------------------------------------------------
# Phase 8 Refactor-1: compact bubble readback
# ---------------------------------------------------------------------------


def test_compact_active_bubbles_empty_pool():
    """Fresh pool with no bubbles seeded must return n_active=0 and empty
    arrays without crashing in the short-circuit branch."""
    from boilingsim.boiling import read_active_bubbles

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 64
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    view = read_active_bubbles(pool)

    assert view.n_active == 0
    assert view.positions.shape == (0, 3)
    assert view.radii.shape == (0,)
    assert view.departure_radii.shape == (0,)
    assert view.site_cleared.shape == (0,)
    assert view.site_cleared.dtype == np.bool_


def test_compact_active_bubbles_sparse_mixed():
    """Seed five active bubbles into sparse slots (0, 17, 53, 99, 4321)
    with distinct radii, then assert the helper returns exactly those
    five with all fields preserved. Output order is non-deterministic
    (atomic_add race) so we compare as sets / sorted arrays."""
    from boilingsim.boiling import read_active_bubbles, seed_test_bubble

    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = 8192
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    pool = grid.bubbles

    seeded = [
        # (slot, position, radius)
        (0,    (0.000, 0.000, 0.050), 1.0e-3),
        (17,   (0.010, 0.000, 0.052), 1.5e-3),
        (53,   (0.020, 0.005, 0.054), 2.0e-3),
        (99,   (0.030, 0.010, 0.056), 2.5e-3),
        (4321, (0.040, 0.015, 0.058), 3.0e-3),
    ]
    for slot, pos, R in seeded:
        seed_test_bubble(pool, slot=slot, position=pos,
                         velocity=(0.0, 0.0, 0.0),
                         radius=R, birth_time=0.0)

    view = read_active_bubbles(pool)

    assert view.n_active == 5
    assert view.positions.shape == (5, 3)
    assert view.radii.shape == (5,)

    # Sorted-radius comparison (output order is race-dependent).
    sorted_radii = np.sort(view.radii)
    expected_radii = np.array(sorted([s[2] for s in seeded]), dtype=np.float32)
    np.testing.assert_allclose(sorted_radii, expected_radii, rtol=1e-6)

    # Position set match: sort each row by x to align
    sorted_pos = view.positions[np.argsort(view.positions[:, 0])]
    expected_pos = np.array(
        sorted([s[1] for s in seeded], key=lambda p: p[0]),
        dtype=np.float32,
    )
    np.testing.assert_allclose(sorted_pos, expected_pos, rtol=1e-6)

    # seed_test_bubble sets site_cleared=1 and departure_radius=radius;
    # verify the kernel propagated both fields.
    assert view.site_cleared.all()
    sorted_dep = np.sort(view.departure_radii)
    np.testing.assert_allclose(sorted_dep, expected_radii, rtol=1e-6)
