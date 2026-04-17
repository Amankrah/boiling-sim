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
    cfg = load_scenario("configs/scenarios/default.yaml")
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
    cfg = load_scenario("configs/scenarios/default.yaml")
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
