"""Nucleate boiling sub-model: Lagrangian bubble pool, nucleation site
detection, Mikic-Rohsenow growth, Fritz departure, Cole frequency.

Phase 3 Milestone A: data structures + nucleation detection only.
Physics stepping (Mikic-Rohsenow, Fritz, Cole) lands in Milestone B;
two-way coupling in C/D; validation sweep in E.

Conventions
-----------
All units SI (m, kg, s, K). Bubble radii stored in metres.

Material IDs: ``MAT_POT_WALL = 1`` (nucleation happens only on cells
whose +z neighbour is MAT_FLUID, i.e. the inner pot-wall top layer that
sits in direct contact with water).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .config import BoilingConfig, ScenarioConfig
from .geometry import MAT_FLUID, MAT_POT_WALL, Grid


# ---------------------------------------------------------------------------
# Lagrangian bubble data structure
# ---------------------------------------------------------------------------


@wp.struct
class Bubble:
    """One Lagrangian vapor bubble.

    ``active != 0`` marks a live bubble. Inactive slots are skipped by
    every kernel and are candidates for reuse by :func:`spawn_bubble`.
    """
    position: wp.vec3    # m, world-space position of bubble centre
    velocity: wp.vec3    # m/s, bubble-centre velocity
    radius: float        # m
    birth_time: float    # s, simulation time at nucleation
    active: int          # 1 = live, 0 = empty slot
    site_i: int          # nucleation-site grid index (for site-active bookkeeping)
    site_j: int
    site_k: int


# ---------------------------------------------------------------------------
# Nucleation-site density lookup table (Kocamustafaogullari-Ishii)
# ---------------------------------------------------------------------------
#
# N_a(ΔT_w) = (1 / D_c^2) · F(ρ*) · (ΔT_w)^{4.4}    [sites / m²]
#
# D_c is the critical cavity diameter; F(ρ*) is a density-ratio function.
# For engineering use we tabulate N_a directly on [0, 50] K with 101 entries
# (dT from 0 K to 50 K in 0.5 K steps). Kernel calls do linear interpolation.

_N_ENTRIES = 101
_DT_MAX_K = 50.0


def _kocamustafaogullari_ishii_site_density(dT_k: float, cfg: BoilingConfig,
                                              water_props: dict) -> float:
    """Return N_a [sites / m²] at wall superheat dT_k.

    Simplified engineering form (per dev-guide §2.5 line 164):
        N_a = F / D_c^2 · (ΔT_w)^4.4
    with F absorbing the ρ* dependence. The D_c is estimated from the
    contact angle and surface tension via
        D_c ≈ 4·σ·T_sat / (h_lv·ρ_v·ΔT_w)  (Hsu 1962)
    but because the 4.4 exponent dominates, a rough D_c estimate is fine.
    We target a density near 10⁵ sites/m² at ΔT=10 K (typical pool boiling).
    """
    if dT_k <= 0.0:
        return 0.0
    # Calibrated so N_a(10 K) ≈ 1e5 sites/m² for water on steel (order-of-magnitude).
    scale = 5.0
    return scale * (dT_k ** 4.4)


def build_nucleation_table(cfg: BoilingConfig, water_props: dict,
                             device: str = "cuda:0") -> wp.array:
    """Precompute N_a(ΔT_w) on a uniform grid [0, 50] K as a Warp array.

    Kernels interpolate linearly with ``dT_k / dT_max * (N_entries - 1)``.
    """
    table_np = np.array(
        [_kocamustafaogullari_ishii_site_density(
            i * _DT_MAX_K / (_N_ENTRIES - 1), cfg, water_props)
         for i in range(_N_ENTRIES)],
        dtype=np.float32,
    )
    return wp.array(table_np, dtype=float, device=device)


@wp.func
def lookup_site_density(table: wp.array(dtype=float), dT_k: float,
                         dT_max: float, n_entries: int) -> float:
    """Linearly interpolate site-density table at wall superheat ``dT_k``."""
    if dT_k <= 0.0:
        return 0.0
    clamped = wp.clamp(dT_k, 0.0, dT_max - 1.0e-6)
    idx_f = clamped / dT_max * float(n_entries - 1)
    i0 = int(idx_f)
    i1 = i0 + 1
    if i1 >= n_entries:
        return table[n_entries - 1]
    frac = idx_f - float(i0)
    return table[i0] * (1.0 - frac) + table[i1] * frac


# ---------------------------------------------------------------------------
# BubblePool: device-side pool + site-active bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class BubblePool:
    """Container for the Lagrangian bubble pool and its auxiliary arrays."""

    bubbles: wp.array                 # shape (max_bubbles,) of Bubble struct
    site_active: wp.array             # int3d, (nx, ny, nz); 1 = bubble at this nucleation cell
    active_count: wp.array            # int1d, length 1 — atomic-add target for new bubbles
    nucleation_table: wp.array        # float1d, length _N_ENTRIES; N_a(ΔT) LUT
    max_bubbles: int
    cfg: BoilingConfig

    def count_active(self) -> int:
        """Return host-side count of currently-active bubbles.

        Scans the pool via a Warp kernel and host-copies the result.
        Used for diagnostics; not to be called every step.
        """
        nx, ny, nz = 0, 0, 0  # unused; kept for symmetry
        sum_buf = wp.zeros(1, dtype=int, device=self.bubbles.device)
        wp.launch(
            _count_active_kernel,
            dim=self.max_bubbles,
            inputs=[self.bubbles, sum_buf],
            device=self.bubbles.device,
        )
        return int(sum_buf.numpy()[0])


def allocate_bubble_pool(cfg: ScenarioConfig, grid: Grid,
                          device: str = "cuda:0") -> BubblePool:
    """Allocate the bubble pool + nucleation bookkeeping arrays on the device."""
    boiling = cfg.boiling
    water_props = {"T_sat": 373.15, "rho_v": 0.598, "h_lv": 2.257e6,
                   "sigma": 0.0589}  # read from materials.json in full build

    bubbles = wp.zeros(boiling.max_bubbles, dtype=Bubble, device=device)
    nx, ny, nz = grid.shape
    site_active = wp.zeros((nx, ny, nz), dtype=int, device=device)
    active_count = wp.zeros(1, dtype=int, device=device)
    table = build_nucleation_table(boiling, water_props, device=device)

    return BubblePool(
        bubbles=bubbles,
        site_active=site_active,
        active_count=active_count,
        nucleation_table=table,
        max_bubbles=boiling.max_bubbles,
        cfg=boiling,
    )


# ---------------------------------------------------------------------------
# Diagnostic kernels (cheap, used by tests)
# ---------------------------------------------------------------------------


@wp.kernel
def _count_active_kernel(bubbles: wp.array(dtype=Bubble),
                          sum_buf: wp.array(dtype=int)):
    b = wp.tid()
    if bubbles[b].active == 1:
        wp.atomic_add(sum_buf, 0, 1)


# ---------------------------------------------------------------------------
# Nucleation-site detection
# ---------------------------------------------------------------------------


@wp.kernel
def detect_nucleation_sites(
    bubbles: wp.array(dtype=Bubble),
    active_count: wp.array(dtype=int),
    site_active: wp.array3d(dtype=int),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    nucleation_table: wp.array(dtype=float),
    origin: wp.vec3,
    dx: float,
    # params
    T_sat_k: float,
    dT_onb: float,
    dT_max: float,
    n_table: int,
    initial_radius: float,
    nucleation_prob_per_step: float,
    dt: float,
    current_time: float,
    max_bubbles: int,
    mat_pot_wall: int,
    mat_fluid: int,
    # per-step pseudo-random seed (int) — rotated each call to give varying patterns
    seed: int,
):
    """Scan pot-wall cells that border fluid; spawn new bubbles where wall
    superheat exceeds ΔT_onb AND the site is not already occupied.

    Spawn probability per step = site_density * cell_area * dt *
    nucleation_prob_per_step. Because site density can be very high
    (10⁵/m²), and each cell face is 4×10⁻⁶ m² at dx=2 mm, per-cell per-step
    spawn rate is ~1 at 20 K superheat — so we clamp to probabilistic
    spawning using a cheap hash-random.
    """
    i, j, k = wp.tid()

    # Skip non-pot-wall cells
    if mat[i, j, k] != mat_pot_wall:
        return
    # Must border a fluid cell above (nucleates into water)
    if k + 1 >= T.shape[2]:
        return
    if mat[i, j, k + 1] != mat_fluid:
        return
    # Skip if a bubble is already at this site
    if site_active[i, j, k] != 0:
        return

    # Wall superheat at this cell
    T_wall = T[i, j, k]
    dT = T_wall - T_sat_k
    if dT < dT_onb:
        return

    # Local site density (sites/m²) → expected sites this step this cell
    N_a = lookup_site_density(nucleation_table, dT, dT_max, n_table)
    face_area = dx * dx
    expected_count = N_a * face_area * dt * nucleation_prob_per_step

    # Simple hash for pseudorandom unit-float in [0,1)
    h = seed ^ (i * 73856093)
    h = h ^ (j * 19349663)
    h = h ^ (k * 83492791)
    h = h ^ (h >> 13)
    h = h * 1274126177
    rnd = float(h & 16777215) / 16777216.0  # 24-bit mantissa

    if rnd >= expected_count:
        return

    # Atomically claim a bubble slot
    slot = wp.atomic_add(active_count, 0, 1)
    if slot >= max_bubbles:
        # Pool full — unclaim to keep counter bounded
        wp.atomic_sub(active_count, 0, 1)
        return

    # Position the bubble at the centre of the fluid cell above this wall cell
    p = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k + 1) + 0.5) * dx

    b = Bubble()
    b.position = p
    b.velocity = wp.vec3(0.0, 0.0, 0.0)
    b.radius = initial_radius
    b.birth_time = current_time
    b.active = 1
    b.site_i = i
    b.site_j = j
    b.site_k = k
    bubbles[slot] = b
    site_active[i, j, k] = 1


# ---------------------------------------------------------------------------
# Python-side driver
# ---------------------------------------------------------------------------


def step_nucleation(
    grid: Grid,
    pool: BubblePool,
    cfg: ScenarioConfig,
    dt: float,
    sim_time: float,
    step_count: int,
    device: str = "cuda:0",
) -> None:
    """Milestone-A entry point: spawn new bubbles where wall is superheated.

    Safe to call every step. Cheap — one kernel pass over (nx, ny, nz).
    """
    nx, ny, nz = grid.shape
    T_sat_k = 373.15  # water saturation at 1 atm

    wp.launch(
        detect_nucleation_sites,
        dim=(nx, ny, nz),
        inputs=[
            pool.bubbles,
            pool.active_count,
            pool.site_active,
            grid.T,
            grid.mat,
            pool.nucleation_table,
            wp.vec3(*grid.origin),
            grid.dx,
            T_sat_k,
            cfg.boiling.dT_onb_k,
            _DT_MAX_K,
            _N_ENTRIES,
            cfg.boiling.initial_bubble_radius_m,
            cfg.boiling.nucleation_probability_per_step,
            dt,
            sim_time,
            pool.max_bubbles,
            MAT_POT_WALL,
            MAT_FLUID,
            step_count + 1,  # seed, rotated each step
        ],
        device=device,
    )


# Expose module-level constants for tests
N_TABLE_ENTRIES = _N_ENTRIES
DT_TABLE_MAX_K = _DT_MAX_K
