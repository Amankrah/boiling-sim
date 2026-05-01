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
    radius: float        # m, current (possibly still-growing) bubble radius
    birth_time: float    # s, simulation time at nucleation
    active: int          # 1 = live, 0 = empty slot
    site_i: int          # nucleation-site grid index (for site-active bookkeeping)
    site_j: int
    site_k: int
    site_cleared: int    # 1 after departure clears site_active; preserves site_i/j/k
    departure_radius: float  # m, frozen copy of radius at the moment site_cleared flips 0 -> 1.
                             # Diagnostics (departure-diameter histogram) must use this, not
                             # ``radius`` -- a detached bubble keeps growing via Mikic-Rohsenow
                             # as it rises, so `radius` alone is the age-weighted population,
                             # not the Fritz-departure population.


# ---------------------------------------------------------------------------
# Nucleation-site density lookup table (Kocamustafaogullari-Ishii)
# ---------------------------------------------------------------------------
#
# N_a(delta T_w) = (1 / D_c^2) * F(rho*) * (delta T_w)^{4.4}    [sites / m^2]
#
# D_c is the critical cavity diameter; F(rho*) is a density-ratio function.
# For engineering use we tabulate N_a directly on [0, 50] K with 101 entries
# (dT from 0 K to 50 K in 0.5 K steps). Kernel calls do linear interpolation.

_N_ENTRIES = 101
_DT_MAX_K = 50.0


def _kocamustafaogullari_ishii_site_density(dT_k: float, cfg: BoilingConfig,
                                              water_props: dict) -> float:
    """Return N_a [sites / m^2] at wall superheat dT_k.

    Simplified engineering form (per dev-guide sec.2.5 line 164):
        N_a = F / D_c^2 * (delta T_w)^4.4
    with F absorbing the rho* dependence. The D_c is estimated from the
    contact angle and surface tension via
        D_c ~ 4*sigma*T_sat / (h_lv*rho_v*delta T_w)  (Hsu 1962)
    but because the 4.4 exponent dominates, a rough D_c estimate is fine.
    We target a density near 10^5 sites/m^2 at delta T=10 K (typical pool boiling).
    """
    if dT_k <= 0.0:
        return 0.0
    # Calibrated so N_a(10 K) ~ 1e5 sites/m^2 for water on steel (order-of-magnitude).
    scale = 5.0
    return scale * (dT_k ** 4.4)


def build_nucleation_table(cfg: BoilingConfig, water_props: dict,
                             device: str = "cuda:0") -> wp.array:
    """Precompute N_a(delta T_w) on a uniform grid [0, 50] K as a Warp array.

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
# Correlations: Fritz departure diameter, Cole frequency, Mikic-Rohsenow
# ---------------------------------------------------------------------------
#
# The 0.0208 coefficient in Fritz's 1935 original uses the contact angle in
# DEGREES. The dev-guide sec.2.5 text says "radians"; that would be wrong by a
# factor 57.3. We honour the guide's radian input but convert internally so
# the numerical result matches published values (~2.5 mm for water on steel
# at theta ~ 1 rad).


@wp.func
def fritz_departure_diameter(theta_rad: float, sigma: float, g_mag: float,
                              rho_l: float, rho_v: float) -> float:
    """D_d [m] = 0.0208 * theta[deg] * sqrt(sigma / (g*(rho_l - rho_v))).

    Ref: Fritz (1935). At water-on-steel conditions (theta ~ 1 rad = 57.3deg,
    sigma ~ 0.059 N/m, delta rho ~ 996 kg/m^3) this gives ~2.9 mm.
    """
    theta_deg = theta_rad * 57.29577951308232
    return 0.0208 * theta_deg * wp.sqrt(sigma / (g_mag * (rho_l - rho_v)))


@wp.func
def cole_frequency(D_d: float, g_mag: float,
                    rho_l: float, rho_v: float) -> float:
    """f [Hz] = sqrt(4*g*(rho_l - rho_v) / (3*D_d*rho_l)).  Cole (1960)."""
    return wp.sqrt(4.0 * g_mag * (rho_l - rho_v) / (3.0 * D_d * rho_l))


@wp.func
def terminal_slip_velocity(R: float) -> float:
    """Radius-dependent terminal rise velocity for a water bubble.

    Replaces the constant 0.2 m/s slip used through Phase 3, which was
    the Grace-1976 plateau for ~2.5 mm bubbles. Real terminal velocity
    spans 2-3 orders of magnitude across the radii in our pool: at
    R=50 um, v ~ 5 mm/s; at R=1 mm, v ~ 220 mm/s; plateau above that.

    Power-law fit to clean-water-bubble data (Clift-Grace-Weber 1978,
    figure 7.3) anchored on (R=50 um, v=5 mm/s) and (R=1 mm, v=220 mm/s):
        v(R) = 391 * R^1.26  [m/s, R in m]
    capped at the wave-controlled plateau v_plateau = 0.22 m/s for
    R >= 1 mm. The exponent 1.26 is gentler than Stokes (R^2) because
    real bubbles transition out of Stokes drag around Re ~ 1 (R ~ 50 um);
    the curve absorbs the Re correction empirically rather than solving
    the drag balance self-consistently.

    Anchor on water at 100 C; the same form is within ~30 % for water
    at 25-100 C, which is good enough for our visualization purposes.
    For other fluids (e.g. milk), the plateau and exponent would shift
    with the Bond / Morton numbers; not in scope.
    """
    v_pow = 391.0 * wp.pow(R, 1.26)
    if v_pow > 0.22:
        return 0.22
    return v_pow


@wp.func
def mikic_rohsenow_radius(age_s: float, T_local_k: float, T_sat_k: float,
                           rho_l: float, rho_v: float, cp_l: float,
                           k_l: float, h_lv: float) -> float:
    """R(t) = (2/sqrtpi) * Ja * sqrt(alpha_l * t).

    Jakob number Ja = rho_l c_p_l (T - T_sat) / (rho_v h_lv).
    Thermal diffusivity alpha_l = k_l / (rho_l c_p_l).
    For T_local <= T_sat returns 0 (no growth drive).
    """
    dT = T_local_k - T_sat_k
    if dT <= 0.0:
        return 0.0
    Ja = rho_l * cp_l * dT / (rho_v * h_lv)
    alpha_l = k_l / (rho_l * cp_l)
    # 2/sqrt(pi) ~ 1.1283791670955126
    return 1.1283791670955126 * Ja * wp.sqrt(alpha_l * age_s)


@wp.func
def _condensation_decrement(
    R: float,
    T_local_k: float,
    T_sat_k: float,
    rho_l: float,
    rho_v: float,
    cp_l: float,
    k_l: float,
    h_lv: float,
    dt: float,
) -> float:
    """Plesset-Zwick-style diffusion-controlled condensation shrinkage.

    Symmetric counterpart to :func:`mikic_rohsenow_radius`: a vapor bubble in
    subcooled liquid shrinks at
        dR/dt = -(2/sqrt(pi)) * Ja_sub * alpha_l / R
    where ``Ja_sub = rho_l * c_p,l * (T_sat - T_local) / (rho_v * h_lv)``
    (positive when the local liquid is subcooled). Returns the *positive*
    radius decrement for this step; caller subtracts it from the current
    radius. Returns 0 at or above saturation (no condensation drive). The
    1/R dependence means the rate accelerates as the bubble shrinks -- once
    condensation starts, full collapse typically completes within 1-2 dt
    steps for bubbles below ~100 um in strongly subcooled liquid.
    """
    dT_sub = T_sat_k - T_local_k
    if dT_sub <= 0.0:
        return 0.0
    if R <= 0.0:
        return 0.0
    Ja_sub = rho_l * cp_l * dT_sub / (rho_v * h_lv)
    alpha_l = k_l / (rho_l * cp_l)
    # 2/sqrt(pi) ~ 1.1283791670955126
    rate = 1.1283791670955126 * Ja_sub * alpha_l / R   # [m/s]
    return rate * dt


# ---------------------------------------------------------------------------
# BubblePool: device-side pool + site-active bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class BubblePool:
    """Container for the Lagrangian bubble pool and its auxiliary arrays."""

    bubbles: wp.array                 # shape (max_bubbles,) of Bubble struct
    slot_claim: wp.array              # int1d, length max_bubbles; per-slot claim flag (0=free, 1=claimed)
    site_active: wp.array             # int3d, (nx, ny, nz); 1 = bubble at this nucleation cell
    nucleation_table: wp.array        # float1d, length _N_ENTRIES; N_a(delta T) LUT
    needs_fragment: wp.array          # int1d, length max_bubbles; 1 = flagged for fragmentation this step (M2)
    # Phase 8 M3: spatial-hash coalescence workspace.
    bin_counts: wp.array              # int1d, (n_bins,); bubble count per bin this step
    bin_lists: wp.array               # int2d, (n_bins, max_per_bin); bubble indices per bin
    merge_target: wp.array            # int1d, (max_bubbles,); 0 = not absorbed, else absorber_idx + 1
    n_bins_x: int
    n_bins_y: int
    n_bins_z: int
    bin_size_m: float
    bin_origin: tuple                 # (ox, oy, oz) world-space origin of the bin grid
    # Phase 8 Refactor-1: compact-readback workspace (read_active_bubbles).
    view_positions: wp.array          # (max_bubbles, 3) float32, dense scatter target
    view_radii: wp.array              # (max_bubbles,) float32
    view_departure_radii: wp.array    # (max_bubbles,) float32
    view_site_cleared: wp.array       # (max_bubbles,) int32
    view_n_active: wp.array           # (1,) int32, atomic counter zeroed each readback
    max_bubbles: int
    cfg: BoilingConfig

    def count_active(self) -> int:
        """Return host-side count of currently-active bubbles.

        Scans the pool via a Warp kernel and host-copies the result.
        Used for diagnostics; not to be called every step.
        """
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
    slot_claim = wp.zeros(boiling.max_bubbles, dtype=int, device=device)
    needs_fragment = wp.zeros(boiling.max_bubbles, dtype=int, device=device)
    nx, ny, nz = grid.shape
    site_active = wp.zeros((nx, ny, nz), dtype=int, device=device)
    table = build_nucleation_table(boiling, water_props, device=device)

    # M3 spatial-hash workspace: covers the full grid domain in
    # ``coalescence_bin_size_m`` cubes. Coverage is generous (1-2 bins
    # of slack on each side) since bubbles outside the fluid region
    # would have been deactivated by ``update_bubbles``' wall check.
    bin_size = float(boiling.coalescence_bin_size_m)
    domain_x = float(nx) * grid.dx
    domain_y = float(ny) * grid.dx
    domain_z = float(nz) * grid.dx
    n_bins_x = max(1, int(domain_x / bin_size) + 1)
    n_bins_y = max(1, int(domain_y / bin_size) + 1)
    n_bins_z = max(1, int(domain_z / bin_size) + 1)
    n_bins = n_bins_x * n_bins_y * n_bins_z
    max_per_bin = int(boiling.coalescence_max_per_bin)
    bin_counts = wp.zeros(n_bins, dtype=int, device=device)
    bin_lists = wp.zeros((n_bins, max_per_bin), dtype=int, device=device)
    merge_target = wp.zeros(boiling.max_bubbles, dtype=int, device=device)
    bin_origin = (
        float(grid.origin[0]),
        float(grid.origin[1]),
        float(grid.origin[2]),
    )

    # Phase 8 Refactor-1: compact-readback workspace. Sized once to the
    # full pool so the host transfer is over a small typed buffer rather
    # than the 60-byte/slot Bubble struct, and downstream Python uses
    # ``[:n_active]`` instead of a boolean mask.
    view_positions = wp.zeros((boiling.max_bubbles, 3), dtype=float, device=device)
    view_radii = wp.zeros(boiling.max_bubbles, dtype=float, device=device)
    view_departure_radii = wp.zeros(boiling.max_bubbles, dtype=float, device=device)
    view_site_cleared = wp.zeros(boiling.max_bubbles, dtype=int, device=device)
    view_n_active = wp.zeros(1, dtype=int, device=device)

    return BubblePool(
        bubbles=bubbles,
        slot_claim=slot_claim,
        site_active=site_active,
        nucleation_table=table,
        needs_fragment=needs_fragment,
        bin_counts=bin_counts,
        bin_lists=bin_lists,
        merge_target=merge_target,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        n_bins_z=n_bins_z,
        bin_size_m=bin_size,
        bin_origin=bin_origin,
        view_positions=view_positions,
        view_radii=view_radii,
        view_departure_radii=view_departure_radii,
        view_site_cleared=view_site_cleared,
        view_n_active=view_n_active,
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
# Phase 8 Refactor-1: compact bubble readback for dashboard / sample_scalars
# ---------------------------------------------------------------------------


@dataclass
class CompactBubbles:
    """Dense host-side view over the active bubbles.

    Emitted by :func:`read_active_bubbles`. ``positions`` is shaped
    ``(n_active, 3)``; the three radius/flag arrays are length
    ``n_active``. Output order is non-deterministic (the GPU compaction
    uses ``atomic_add`` to claim slots), so callers must not assume
    stability across runs.
    """
    n_active: int
    positions: np.ndarray
    radii: np.ndarray
    departure_radii: np.ndarray
    site_cleared: np.ndarray  # bool


_EMPTY_VEC3 = np.zeros((0, 3), dtype=np.float32)
_EMPTY_F32 = np.zeros(0, dtype=np.float32)
_EMPTY_BOOL = np.zeros(0, dtype=bool)


@wp.kernel
def _compact_active_bubbles_kernel(
    bubbles: wp.array(dtype=Bubble),
    view_positions: wp.array2d(dtype=float),
    view_radii: wp.array(dtype=float),
    view_departure_radii: wp.array(dtype=float),
    view_site_cleared: wp.array(dtype=int),
    view_n_active: wp.array(dtype=int),
):
    """Scatter active bubbles into dense view_* arrays + atomic count.

    One thread per pool slot. Inactive slots return early; active slots
    claim a unique output index via ``atomic_add`` and write the
    dashboard-relevant fields. Output ordering is race-dependent and
    therefore non-deterministic -- safe because every consumer
    (mean/max statistics, msgpack list, HDF5 unordered set) is
    order-independent.
    """
    b = wp.tid()
    if bubbles[b].active != 1:
        return
    idx = wp.atomic_add(view_n_active, 0, 1)
    view_positions[idx, 0] = bubbles[b].position[0]
    view_positions[idx, 1] = bubbles[b].position[1]
    view_positions[idx, 2] = bubbles[b].position[2]
    view_radii[idx] = bubbles[b].radius
    view_departure_radii[idx] = bubbles[b].departure_radius
    view_site_cleared[idx] = bubbles[b].site_cleared


def read_active_bubbles(pool: "BubblePool") -> CompactBubbles:
    """Return a dense host-side view of the currently active bubbles.

    Replaces the older idiom of ``pool.bubbles.numpy(); mask =
    bubbles['active'] == 1; ...`` which DMAs the full 60-byte/slot
    struct array even when only a few percent of the pool is active.
    The kernel scatters into pre-allocated workspace arrays, then we
    sync once on the n_active counter and slice ``[:n_active]`` on the
    host.
    """
    pool.view_n_active.zero_()
    wp.launch(
        _compact_active_bubbles_kernel,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles,
            pool.view_positions,
            pool.view_radii,
            pool.view_departure_radii,
            pool.view_site_cleared,
            pool.view_n_active,
        ],
        device=pool.bubbles.device,
    )
    n_active = int(pool.view_n_active.numpy()[0])
    if n_active == 0:
        return CompactBubbles(
            n_active=0,
            positions=_EMPTY_VEC3,
            radii=_EMPTY_F32,
            departure_radii=_EMPTY_F32,
            site_cleared=_EMPTY_BOOL,
        )
    return CompactBubbles(
        n_active=n_active,
        positions=pool.view_positions.numpy()[:n_active],
        radii=pool.view_radii.numpy()[:n_active],
        departure_radii=pool.view_departure_radii.numpy()[:n_active],
        site_cleared=pool.view_site_cleared.numpy()[:n_active].astype(bool),
    )


@wp.kernel
def _seed_test_bubble(
    bubbles: wp.array(dtype=Bubble),
    slot_claim: wp.array(dtype=int),
    slot: int,
    px: float, py: float, pz: float,
    vx: float, vy: float, vz: float,
    radius: float,
    birth_time: float,
):
    """Test-only helper: write a single active bubble into ``slot`` and
    mark the slot as claimed so ordinary deactivation paths free it later.
    """
    _ = wp.tid()  # kernel launched with dim=1
    b = Bubble()
    b.position = wp.vec3(px, py, pz)
    b.velocity = wp.vec3(vx, vy, vz)
    b.radius = radius
    b.birth_time = birth_time
    b.active = 1
    b.site_i = -1
    b.site_j = -1
    b.site_k = -1
    b.site_cleared = 1
    b.departure_radius = radius  # test bubbles are seeded post-departure
    bubbles[slot] = b
    slot_claim[slot] = 1


def seed_test_bubble(
    pool: BubblePool,
    slot: int,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0e-5,
    birth_time: float = 0.0,
    device: str = "cuda:0",
) -> None:
    """Convenience wrapper around :kernel:`_seed_test_bubble` for Python code."""
    wp.launch(
        _seed_test_bubble,
        dim=1,
        inputs=[
            pool.bubbles, pool.slot_claim, slot,
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2],
            radius, birth_time,
        ],
        device=device,
    )


# ---------------------------------------------------------------------------
# Nucleation-site detection
# ---------------------------------------------------------------------------


@wp.kernel
def detect_nucleation_sites(
    bubbles: wp.array(dtype=Bubble),
    slot_claim: wp.array(dtype=int),          # parallel to bubbles, atomic 0/1 flag
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
    # per-step pseudo-random seed (int) -- rotated each call to give varying patterns
    seed: int,
):
    """Scan pot-wall cells that border fluid; spawn new bubbles where wall
    superheat exceeds delta T_onb AND the site is not already occupied.

    Slot allocation uses a hash-seeded linear probe against ``slot_claim``,
    so vented bubbles' slots are correctly reused. No monotonic counter.
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

    # Local site density (sites/m^2) -> expected sites this step this cell
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

    # Find a free slot by linear-probing from a hash-derived starting index.
    # atomic_cas returns the old value; if it was 0 we atomically flipped to 1
    # and own the slot; if 1 it was already taken and we advance.
    start = int((h >> 7) & 2147483647) % max_bubbles
    slot = int(-1)   # explicit dynamic-int declaration so we can mutate in loop
    for attempt in range(16):
        cand = (start + attempt) % max_bubbles
        old = wp.atomic_cas(slot_claim, cand, 0, 1)
        if old == 0:
            slot = cand
            break
    if slot < 0:
        return  # pool full or contention; drop this nucleation, retry next step

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
    b.site_cleared = 0         # still attached; departure_radius will freeze on release
    b.departure_radius = 0.0   # sentinel: non-zero only after the 0 -> 1 flip
    bubbles[slot] = b
    site_active[i, j, k] = 1


# ---------------------------------------------------------------------------
# Sampling helpers for bubble kernels
# ---------------------------------------------------------------------------


@wp.func
def _sample_cell_scalar(
    field: wp.array3d(dtype=float),
    p: wp.vec3,
    origin: wp.vec3,
    dx: float,
) -> float:
    """Trilinear sample a cell-centred scalar at world-space point ``p``.

    Mirrors fluid._tri_sample with cell-centre offset (0.5, 0.5, 0.5).
    """
    fx = (p[0] - origin[0]) / dx - 0.5
    fy = (p[1] - origin[1]) / dx - 0.5
    fz = (p[2] - origin[2]) / dx - 0.5

    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]
    fx = wp.clamp(fx, 0.0, float(nx - 1) - 1.0e-6)
    fy = wp.clamp(fy, 0.0, float(ny - 1) - 1.0e-6)
    fz = wp.clamp(fz, 0.0, float(nz - 1) - 1.0e-6)

    i0 = int(fx); j0 = int(fy); k0 = int(fz)
    tx = fx - float(i0); ty = fy - float(j0); tz = fz - float(k0)

    c000 = field[i0, j0, k0]
    c100 = field[i0 + 1, j0, k0]
    c010 = field[i0, j0 + 1, k0]
    c110 = field[i0 + 1, j0 + 1, k0]
    c001 = field[i0, j0, k0 + 1]
    c101 = field[i0 + 1, j0, k0 + 1]
    c011 = field[i0, j0 + 1, k0 + 1]
    c111 = field[i0 + 1, j0 + 1, k0 + 1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@wp.func
def _sample_face_u(
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    p: wp.vec3,
    origin: wp.vec3,
    dx: float,
) -> wp.vec3:
    """Sample the full (u, v, w) velocity vector at point ``p`` on a MAC grid."""
    # x-face: offset (0.0, 0.5, 0.5)
    fx = (p[0] - origin[0]) / dx
    fy = (p[1] - origin[1]) / dx - 0.5
    fz = (p[2] - origin[2]) / dx - 0.5
    fx = wp.clamp(fx, 0.0, float(ux.shape[0] - 1) - 1.0e-6)
    fy = wp.clamp(fy, 0.0, float(ux.shape[1] - 1) - 1.0e-6)
    fz = wp.clamp(fz, 0.0, float(ux.shape[2] - 1) - 1.0e-6)
    i0 = int(fx); j0 = int(fy); k0 = int(fz)
    tx = fx - float(i0); ty = fy - float(j0); tz = fz - float(k0)
    c000 = ux[i0, j0, k0]; c100 = ux[i0 + 1, j0, k0]
    c010 = ux[i0, j0 + 1, k0]; c110 = ux[i0 + 1, j0 + 1, k0]
    c001 = ux[i0, j0, k0 + 1]; c101 = ux[i0 + 1, j0, k0 + 1]
    c011 = ux[i0, j0 + 1, k0 + 1]; c111 = ux[i0 + 1, j0 + 1, k0 + 1]
    u_val = (
        (c000 * (1.0 - tx) + c100 * tx) * (1.0 - ty) * (1.0 - tz)
        + (c010 * (1.0 - tx) + c110 * tx) * ty * (1.0 - tz)
        + (c001 * (1.0 - tx) + c101 * tx) * (1.0 - ty) * tz
        + (c011 * (1.0 - tx) + c111 * tx) * ty * tz
    )

    # y-face: offset (0.5, 0.0, 0.5)
    gx = (p[0] - origin[0]) / dx - 0.5
    gy = (p[1] - origin[1]) / dx
    gz = (p[2] - origin[2]) / dx - 0.5
    gx = wp.clamp(gx, 0.0, float(uy.shape[0] - 1) - 1.0e-6)
    gy = wp.clamp(gy, 0.0, float(uy.shape[1] - 1) - 1.0e-6)
    gz = wp.clamp(gz, 0.0, float(uy.shape[2] - 1) - 1.0e-6)
    i1 = int(gx); j1 = int(gy); k1 = int(gz)
    sx = gx - float(i1); sy = gy - float(j1); sz = gz - float(k1)
    d000 = uy[i1, j1, k1]; d100 = uy[i1 + 1, j1, k1]
    d010 = uy[i1, j1 + 1, k1]; d110 = uy[i1 + 1, j1 + 1, k1]
    d001 = uy[i1, j1, k1 + 1]; d101 = uy[i1 + 1, j1, k1 + 1]
    d011 = uy[i1, j1 + 1, k1 + 1]; d111 = uy[i1 + 1, j1 + 1, k1 + 1]
    v_val = (
        (d000 * (1.0 - sx) + d100 * sx) * (1.0 - sy) * (1.0 - sz)
        + (d010 * (1.0 - sx) + d110 * sx) * sy * (1.0 - sz)
        + (d001 * (1.0 - sx) + d101 * sx) * (1.0 - sy) * sz
        + (d011 * (1.0 - sx) + d111 * sx) * sy * sz
    )

    # z-face: offset (0.5, 0.5, 0.0)
    hx = (p[0] - origin[0]) / dx - 0.5
    hy = (p[1] - origin[1]) / dx - 0.5
    hz = (p[2] - origin[2]) / dx
    hx = wp.clamp(hx, 0.0, float(uz.shape[0] - 1) - 1.0e-6)
    hy = wp.clamp(hy, 0.0, float(uz.shape[1] - 1) - 1.0e-6)
    hz = wp.clamp(hz, 0.0, float(uz.shape[2] - 1) - 1.0e-6)
    i2 = int(hx); j2 = int(hy); k2 = int(hz)
    rx = hx - float(i2); ry = hy - float(j2); rz = hz - float(k2)
    e000 = uz[i2, j2, k2]; e100 = uz[i2 + 1, j2, k2]
    e010 = uz[i2, j2 + 1, k2]; e110 = uz[i2 + 1, j2 + 1, k2]
    e001 = uz[i2, j2, k2 + 1]; e101 = uz[i2 + 1, j2, k2 + 1]
    e011 = uz[i2, j2 + 1, k2 + 1]; e111 = uz[i2 + 1, j2 + 1, k2 + 1]
    w_val = (
        (e000 * (1.0 - rx) + e100 * rx) * (1.0 - ry) * (1.0 - rz)
        + (e010 * (1.0 - rx) + e110 * rx) * ry * (1.0 - rz)
        + (e001 * (1.0 - rx) + e101 * rx) * (1.0 - ry) * rz
        + (e011 * (1.0 - rx) + e111 * rx) * ry * rz
    )
    return wp.vec3(u_val, v_val, w_val)


@wp.func
def _mat_at_point(
    mat: wp.array3d(dtype=int),
    p: wp.vec3,
    origin: wp.vec3,
    dx: float,
) -> int:
    """Return the material ID of the cell that contains ``p`` (nearest cell)."""
    fx = (p[0] - origin[0]) / dx - 0.5
    fy = (p[1] - origin[1]) / dx - 0.5
    fz = (p[2] - origin[2]) / dx - 0.5
    i = wp.clamp(int(fx + 0.5), 0, mat.shape[0] - 1)
    j = wp.clamp(int(fy + 0.5), 0, mat.shape[1] - 1)
    k = wp.clamp(int(fz + 0.5), 0, mat.shape[2] - 1)
    return mat[i, j, k]


# ---------------------------------------------------------------------------
# update_bubbles: grow, depart, advect, vent
# ---------------------------------------------------------------------------


@wp.kernel
def update_bubbles(
    bubbles: wp.array(dtype=Bubble),
    slot_claim: wp.array(dtype=int),
    site_active: wp.array3d(dtype=int),
    needs_fragment: wp.array(dtype=int),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    dt: float,
    current_time: float,
    water_line_z: float,
    # Fluid / vapor properties
    T_sat_k: float,
    rho_l: float,
    rho_v: float,
    cp_l: float,
    k_l: float,
    h_lv: float,
    sigma: float,
    # Physics parameters
    theta_rad: float,
    g_mag: float,
    R_seed: float,
    R_frag: float,
    R_max: float,
    mat_fluid: int,
):
    """Advance one bubble: Mikic-Rohsenow growth in superheated liquid,
    Plesset-Zwick condensation shrinkage in subcooled liquid, Fritz departure,
    advect, vent at free surface or on contact with a solid.

    When a bubble deactivates (vented, entered solid, or fully condensed),
    clear ``slot_claim[b_idx]`` so the slot is reusable by the next nucleation
    step. Full condensation additionally clears the nucleation-site active
    flag and deposits the bubble's remaining latent heat back into the local
    8-cell fluid stencil via trilinear atomic-add (symmetric to
    :func:`scatter_latent_heat`'s atomic-sub path).
    """
    b_idx = wp.tid()
    bubble = bubbles[b_idx]
    if bubble.active == 0:
        return

    # Sample local liquid temperature at bubble centre.
    T_local = _sample_cell_scalar(T, bubble.position, origin, dx)

    if T_local > T_sat_k:
        # --- Superheated liquid: Mikic-Rohsenow monotonic growth. -----------
        age = current_time - bubble.birth_time
        if age < 0.0:
            age = 0.0
        R_target = mikic_rohsenow_radius(
            age, T_local, T_sat_k, rho_l, rho_v, cp_l, k_l, h_lv,
        )
        if R_target > bubble.radius:
            bubble.radius = R_target
    else:
        # --- Subcooled liquid: Plesset-Zwick condensation shrinkage. --------
        # Every step we compute the radius decrement, clamp to a floor at
        # R_seed, and scatter the latent heat of the *volume lost this step*
        # back into the local 8-cell fluid stencil (mirror of
        # scatter_latent_heat: atomic_add here, atomic_sub there). This
        # preserves energy conservation: every mg of vapor that condenses
        # deposits its h_lv into the fluid at the same step, not at the
        # final collapse event. When the clamped radius hits R_seed the
        # bubble deactivates; the site-active flag is cleared so a fresh
        # bubble can nucleate here next step.
        R_old = bubble.radius
        dR = _condensation_decrement(
            R_old, T_local, T_sat_k,
            rho_l, rho_v, cp_l, k_l, h_lv, dt,
        )
        R_new_raw = R_old - dR
        fully_condensed = (R_new_raw <= R_seed)
        R_new = R_seed if fully_condensed else R_new_raw

        # Volume lost this step (R_old > R_new guaranteed; R_old^3 - R_new^3).
        V_lost = (4.0 * 3.14159265358979 / 3.0) * (
            R_old * R_old * R_old - R_new * R_new * R_new
        )
        if V_lost > 0.0:
            E_release = rho_v * h_lv * V_lost
            cell_volume = dx * dx * dx
            dT_ref = E_release / (rho_l * cp_l * cell_volume)

            fx = (bubble.position[0] - origin[0]) / dx - 0.5
            fy = (bubble.position[1] - origin[1]) / dx - 0.5
            fz = (bubble.position[2] - origin[2]) / dx - 0.5
            nx = T.shape[0]
            ny = T.shape[1]
            nz = T.shape[2]
            fx = wp.clamp(fx, 0.0, float(nx - 1) - 1.0e-6)
            fy = wp.clamp(fy, 0.0, float(ny - 1) - 1.0e-6)
            fz = wp.clamp(fz, 0.0, float(nz - 1) - 1.0e-6)
            i0 = int(fx); j0 = int(fy); k0 = int(fz)
            tx = fx - float(i0); ty = fy - float(j0); tz = fz - float(k0)
            w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
            w100 = tx * (1.0 - ty) * (1.0 - tz)
            w010 = (1.0 - tx) * ty * (1.0 - tz)
            w110 = tx * ty * (1.0 - tz)
            w001 = (1.0 - tx) * (1.0 - ty) * tz
            w101 = tx * (1.0 - ty) * tz
            w011 = (1.0 - tx) * ty * tz
            w111 = tx * ty * tz
            if mat[i0, j0, k0] == mat_fluid:
                wp.atomic_add(T, i0, j0, k0, dT_ref * w000)
            if mat[i0 + 1, j0, k0] == mat_fluid:
                wp.atomic_add(T, i0 + 1, j0, k0, dT_ref * w100)
            if mat[i0, j0 + 1, k0] == mat_fluid:
                wp.atomic_add(T, i0, j0 + 1, k0, dT_ref * w010)
            if mat[i0 + 1, j0 + 1, k0] == mat_fluid:
                wp.atomic_add(T, i0 + 1, j0 + 1, k0, dT_ref * w110)
            if mat[i0, j0, k0 + 1] == mat_fluid:
                wp.atomic_add(T, i0, j0, k0 + 1, dT_ref * w001)
            if mat[i0 + 1, j0, k0 + 1] == mat_fluid:
                wp.atomic_add(T, i0 + 1, j0, k0 + 1, dT_ref * w101)
            if mat[i0, j0 + 1, k0 + 1] == mat_fluid:
                wp.atomic_add(T, i0, j0 + 1, k0 + 1, dT_ref * w011)
            if mat[i0 + 1, j0 + 1, k0 + 1] == mat_fluid:
                wp.atomic_add(T, i0 + 1, j0 + 1, k0 + 1, dT_ref * w111)

        if fully_condensed:
            if bubble.site_cleared == 0 and bubble.site_i >= 0:
                site_active[bubble.site_i, bubble.site_j, bubble.site_k] = 0
                bubble.site_cleared = 1
            bubble.active = 0
            bubble.radius = 0.0
            bubbles[b_idx] = bubble
            slot_claim[b_idx] = 0
            return
        else:
            bubble.radius = R_new

    # M2: flag for Rayleigh-Taylor fragmentation when R exceeds R_frag.
    # The actual split (volume-conserving binary breakup) is handled by
    # ``fragment_bubbles`` in a separate pass after this kernel, since
    # claiming a daughter slot requires atomic_cas on the same pool we're
    # iterating. R_max remains as a safety floor: if the daughter pool
    # is full and fragmentation can't fire, the cap kicks in instead so
    # bubbles never exceed R_max regardless of R_frag's value.
    if bubble.radius > R_frag:
        needs_fragment[b_idx] = 1
    if bubble.radius > R_max:
        bubble.radius = R_max

    # Departure check: 2*R >= Fritz D_d.
    D_d = fritz_departure_diameter(theta_rad, sigma, g_mag, rho_l, rho_v)
    departed = (2.0 * bubble.radius >= D_d)

    # Clear the site-active flag on departure so a new bubble can spawn there.
    # ``departure_radius`` is frozen at the 0 -> 1 transition so the
    # departure-diameter histogram reports the size at detachment, not the
    # post-departure grown size during rise.
    if departed:
        if bubble.site_cleared == 0 and bubble.site_i >= 0:
            site_active[bubble.site_i, bubble.site_j, bubble.site_k] = 0
            bubble.site_cleared = 1
            bubble.departure_radius = bubble.radius

    # Advect: departed bubbles get local fluid velocity + R-dependent
    # terminal slip (M1 of bubble-physics plan). Stokes regime for
    # sub-millimetre bubbles, plateau ~0.22 m/s for 1+ mm. Attached
    # bubbles stay at the nucleation site.
    if departed:
        u_fluid = _sample_face_u(ux, uy, uz, bubble.position, origin, dx)
        slip = terminal_slip_velocity(bubble.radius)
        bubble.velocity = wp.vec3(
            u_fluid[0],
            u_fluid[1],
            u_fluid[2] + slip,
        )
        bubble.position = bubble.position + bubble.velocity * dt

    # Vent at free surface.
    if bubble.position[2] >= water_line_z:
        bubble.active = 0
        bubbles[b_idx] = bubble
        slot_claim[b_idx] = 0
        return

    # Deactivate if bubble drifted into a solid (pot wall, carrot, air).
    m_here = _mat_at_point(mat, bubble.position, origin, dx)
    if m_here != mat_fluid:
        bubble.active = 0
        bubbles[b_idx] = bubble
        slot_claim[b_idx] = 0
        return

    bubbles[b_idx] = bubble


# ---------------------------------------------------------------------------
# Phase 8 M2: Rayleigh-Taylor binary fragmentation
# ---------------------------------------------------------------------------


@wp.kernel
def fragment_bubbles(
    bubbles: wp.array(dtype=Bubble),
    slot_claim: wp.array(dtype=int),
    needs_fragment: wp.array(dtype=int),
    max_bubbles: int,
    R_frag: float,
):
    """Binary breakup of bubbles flagged by ``update_bubbles`` for exceeding
    R_frag. Each parent splits into two equal-volume daughters at
    R_d = R / 2^(1/3) = 0.7937 * R, so total volume (and therefore vapor
    mass and latent-heat reservoir) is conserved exactly.

    Two-pass design avoids race conditions: ``update_bubbles`` flags the
    parent in ``needs_fragment``; this kernel runs after, so by the time
    we claim a daughter slot via atomic_cas no concurrent advection or
    growth touches the same pool entry.

    Daughter placement: offset the two daughters along a perpendicular to
    the parent's velocity (so they separate sideways from the rise
    direction, not vertically), each by R_d so their centres are 2*R_d
    apart -- just touching, not overlapping. Velocities get a small
    perpendicular jitter (5 % of |v|) so they diverge over time and don't
    immediately re-coalesce.

    Pool-full graceful degradation: if no free slot is found in 16
    linear-probe attempts, the parent stays at its (possibly capped)
    radius and ``needs_fragment`` is cleared. The R_max safety cap
    handles this case so bubbles never exceed R_max even when
    fragmentation is starved.
    """
    b_idx = wp.tid()
    if needs_fragment[b_idx] == 0:
        return
    needs_fragment[b_idx] = 0  # consume the flag

    bubble = bubbles[b_idx]
    if bubble.active == 0:
        return
    if bubble.radius <= R_frag:
        return  # state changed (condensed below threshold) before frag fired

    # Volume-conserving binary breakup: 2 * (4/3 * pi * R_d^3) = (4/3 * pi * R^3)
    # => R_d = R / 2^(1/3) = R * 0.7937005259840998
    R_daughter = bubble.radius * 0.7937005259840998

    # Perpendicular offset axis: cross product of velocity with the
    # world axis it's least aligned with, then normalize. Stationary
    # bubbles (|v| ~ 0) get a fixed +x offset so split is deterministic.
    v = bubble.velocity
    abs_vx = wp.abs(v[0])
    abs_vy = wp.abs(v[1])
    abs_vz = wp.abs(v[2])
    perp = wp.vec3(0.0, 0.0, 0.0)
    if abs_vx <= abs_vy and abs_vx <= abs_vz:
        perp = wp.vec3(0.0, -v[2], v[1])
    elif abs_vy <= abs_vz:
        perp = wp.vec3(-v[2], 0.0, v[0])
    else:
        perp = wp.vec3(-v[1], v[0], 0.0)
    perp_mag = wp.sqrt(perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2])
    if perp_mag < 1.0e-12:
        perp = wp.vec3(R_daughter, 0.0, 0.0)
        perp_mag = R_daughter
    inv_mag = R_daughter / perp_mag
    offset = wp.vec3(perp[0] * inv_mag, perp[1] * inv_mag, perp[2] * inv_mag)

    # Velocity jitter: 5 % of |v| in the perpendicular direction.
    v_mag = wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    j_scale = 0.05 * v_mag / R_daughter   # so |jitter| = 0.05 |v|
    jitter = wp.vec3(offset[0] * j_scale, offset[1] * j_scale, offset[2] * j_scale)

    # Try to claim a daughter slot via atomic_cas linear-probing. The
    # start index is hashed from b_idx so concurrent fragmentations
    # touch different starting points and rarely contend.
    h = b_idx * 73856093
    h = h ^ (h >> 13)
    h = h * 1274126177
    start = (h & 2147483647) % max_bubbles

    daughter_slot = int(-1)
    for attempt in range(16):
        cand = (start + attempt) % max_bubbles
        if cand != b_idx:
            old = wp.atomic_cas(slot_claim, cand, 0, 1)
            if old == 0:
                daughter_slot = cand
                break

    if daughter_slot < 0:
        return  # pool full; parent stays at full radius (R_max cap clamps it)

    # Daughter: parent state with offset position and jittered velocity.
    daughter = Bubble()
    daughter.position = bubble.position + offset
    daughter.velocity = wp.vec3(v[0] + jitter[0], v[1] + jitter[1], v[2] + jitter[2])
    daughter.radius = R_daughter
    daughter.birth_time = bubble.birth_time
    daughter.active = 1
    daughter.site_i = -1
    daughter.site_j = -1
    daughter.site_k = -1
    daughter.site_cleared = 1
    daughter.departure_radius = bubble.departure_radius
    bubbles[daughter_slot] = daughter

    # Parent: shrink, mirror-offset, mirror-jitter so the pair is
    # symmetric about the original centre.
    bubble.position = bubble.position - offset
    bubble.velocity = wp.vec3(v[0] - jitter[0], v[1] - jitter[1], v[2] - jitter[2])
    bubble.radius = R_daughter
    bubbles[b_idx] = bubble


# ---------------------------------------------------------------------------
# Phase 8 M3: spatial-hash coalescence
# ---------------------------------------------------------------------------


@wp.kernel
def reset_bin_counts(bin_counts: wp.array(dtype=int)):
    """Zero the per-step bin-count array. Launched at dim=n_bins."""
    bin_counts[wp.tid()] = 0


@wp.kernel
def reset_merge_targets(merge_target: wp.array(dtype=int)):
    """Zero the per-step merge_target array. Launched at dim=max_bubbles."""
    merge_target[wp.tid()] = 0


@wp.func
def _bin_index_clamped(p: wp.vec3, origin_x: float, origin_y: float,
                        origin_z: float, bin_size: float,
                        n_bins_x: int, n_bins_y: int,
                        n_bins_z: int) -> wp.vec3i:
    """Compute the 3D bin index for a world-space position, clamped to the
    valid bin range. Bubbles outside the bin grid get folded onto the
    nearest face -- safe because such bubbles will be deactivated by
    update_bubbles' wall check on the next pass."""
    bx = int((p[0] - origin_x) / bin_size)
    by = int((p[1] - origin_y) / bin_size)
    bz = int((p[2] - origin_z) / bin_size)
    if bx < 0:
        bx = 0
    if by < 0:
        by = 0
    if bz < 0:
        bz = 0
    if bx >= n_bins_x:
        bx = n_bins_x - 1
    if by >= n_bins_y:
        by = n_bins_y - 1
    if bz >= n_bins_z:
        bz = n_bins_z - 1
    return wp.vec3i(bx, by, bz)


@wp.kernel
def bin_bubbles(
    bubbles: wp.array(dtype=Bubble),
    bin_counts: wp.array(dtype=int),
    bin_lists: wp.array2d(dtype=int),
    origin_x: float,
    origin_y: float,
    origin_z: float,
    bin_size: float,
    n_bins_x: int,
    n_bins_y: int,
    n_bins_z: int,
    max_per_bin: int,
):
    """Pass A: each active bubble atomic-counts itself into its 3D bin and
    writes its slot index into bin_lists. Bin overflow (> max_per_bin
    bubbles in one bin) drops the excess silently -- those bubbles miss
    coalescence detection this step and try again next step."""
    b_idx = wp.tid()
    bubble = bubbles[b_idx]
    if bubble.active == 0:
        return
    b3 = _bin_index_clamped(
        bubble.position, origin_x, origin_y, origin_z, bin_size,
        n_bins_x, n_bins_y, n_bins_z,
    )
    flat = b3[0] + b3[1] * n_bins_x + b3[2] * n_bins_x * n_bins_y
    k = wp.atomic_add(bin_counts, flat, 1)
    if k < max_per_bin:
        bin_lists[flat, k] = b_idx


@wp.kernel
def find_merge_targets(
    bubbles: wp.array(dtype=Bubble),
    bin_counts: wp.array(dtype=int),
    bin_lists: wp.array2d(dtype=int),
    merge_target: wp.array(dtype=int),
    origin_x: float,
    origin_y: float,
    origin_z: float,
    bin_size: float,
    n_bins_x: int,
    n_bins_y: int,
    n_bins_z: int,
    max_per_bin: int,
):
    """Pass B: each active bubble checks its own bin and the 26 neighbour
    bins for overlapping bubbles. Atomic-CAS on merge_target[other_idx]
    claims the other as an absorbee. First-wins; concurrent attempts on
    the same target fall through and try the next overlap candidate.

    Each bubble can ABSORB many but be ABSORBED at most once (the CAS on
    its target slot ensures this). Chain merges A->B->C are deferred:
    if A claims B and C also wants to claim B, C just doesn't merge B
    this step. Loose but stable; total mass is never lost."""
    b_idx = wp.tid()
    me = bubbles[b_idx]
    if me.active == 0:
        return
    b3 = _bin_index_clamped(
        me.position, origin_x, origin_y, origin_z, bin_size,
        n_bins_x, n_bins_y, n_bins_z,
    )
    bx0 = b3[0]; by0 = b3[1]; bz0 = b3[2]
    # Iterate over own bin + 26 neighbours.
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                bx = bx0 + di
                by = by0 + dj
                bz = bz0 + dk
                if bx < 0 or by < 0 or bz < 0:
                    continue
                if bx >= n_bins_x or by >= n_bins_y or bz >= n_bins_z:
                    continue
                flat = bx + by * n_bins_x + bz * n_bins_x * n_bins_y
                cnt = bin_counts[flat]
                if cnt > max_per_bin:
                    cnt = max_per_bin
                for slot in range(cnt):
                    other_idx = bin_lists[flat, slot]
                    # Avoid self, and only attempt one direction of the pair
                    # (other_idx > b_idx) to halve the work and pair up
                    # decisively.
                    if other_idx <= b_idx:
                        continue
                    other = bubbles[other_idx]
                    if other.active == 0:
                        continue
                    # Overlap check: ||p_me - p_other|| <= R_me + R_other
                    dx_p = me.position[0] - other.position[0]
                    dy_p = me.position[1] - other.position[1]
                    dz_p = me.position[2] - other.position[2]
                    dist2 = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p
                    rsum = me.radius + other.radius
                    if dist2 > rsum * rsum:
                        continue
                    # Try to claim other_idx as an absorbee. CAS: only
                    # the first claimer wins; record b_idx + 1 (so 0 still
                    # means "not absorbed").
                    old = wp.atomic_cas(merge_target, other_idx, 0, b_idx + 1)
                    if old == 0:
                        # Won. Continue checking other candidates -- this
                        # bubble may absorb multiple in one step.
                        pass


@wp.kernel
def apply_merges_absorber_pass(
    bubbles: wp.array(dtype=Bubble),
    merge_target: wp.array(dtype=int),
    max_bubbles: int,
):
    """Pass C1: each non-absorbee bubble scans merge_target for absorbees
    pointing at it (= b_idx + 1) and folds their volume / momentum into
    its own state. Reads absorbee structs while they're still active --
    the absorbee deactivation runs in a separate kernel afterwards to
    avoid intra-kernel struct read/write races.

    O(N) inner scan per absorber. Most thread work is the absorbee /
    active == 0 early exit; the few absorbers do the linear scan."""
    b_idx = wp.tid()
    me = bubbles[b_idx]
    if me.active == 0:
        return
    if merge_target[b_idx] != 0:
        return  # I'm an absorbee, handled by absorbee_pass

    target_marker = b_idx + 1
    V_me = (4.0 / 3.0) * 3.14159265358979 * me.radius * me.radius * me.radius
    px = me.position[0] * V_me
    py = me.position[1] * V_me
    pz = me.position[2] * V_me
    vx = me.velocity[0] * V_me
    vy = me.velocity[1] * V_me
    vz = me.velocity[2] * V_me
    V_total = V_me
    earliest_birth = me.birth_time
    found_any = int(0)
    for j in range(max_bubbles):
        if merge_target[j] != target_marker:
            continue
        other = bubbles[j]
        Ro = other.radius
        Vo = (4.0 / 3.0) * 3.14159265358979 * Ro * Ro * Ro
        V_total = V_total + Vo
        px = px + other.position[0] * Vo
        py = py + other.position[1] * Vo
        pz = pz + other.position[2] * Vo
        vx = vx + other.velocity[0] * Vo
        vy = vy + other.velocity[1] * Vo
        vz = vz + other.velocity[2] * Vo
        if other.birth_time < earliest_birth:
            earliest_birth = other.birth_time
        found_any = 1
    if found_any == 0:
        return
    # Volume-conserving combine: R_new = (V_total / (4pi/3))^(1/3).
    R_new = wp.pow(V_total * 3.0 / (4.0 * 3.14159265358979), 1.0 / 3.0)
    me.radius = R_new
    inv_V = 1.0 / V_total
    me.position = wp.vec3(px * inv_V, py * inv_V, pz * inv_V)
    me.velocity = wp.vec3(vx * inv_V, vy * inv_V, vz * inv_V)
    me.birth_time = earliest_birth
    bubbles[b_idx] = me


@wp.kernel
def apply_merges_absorbee_pass(
    bubbles: wp.array(dtype=Bubble),
    slot_claim: wp.array(dtype=int),
    merge_target: wp.array(dtype=int),
    site_active: wp.array3d(dtype=int),
):
    """Pass C2: deactivate all absorbees flagged in merge_target. Runs
    AFTER the absorber pass so absorbers have already read the absorbees'
    state. Frees the pool slot and clears any wall site the absorbee
    was still attached to (rare; usually absorbees are post-departure
    bubbles in the bulk)."""
    b_idx = wp.tid()
    if merge_target[b_idx] == 0:
        return
    me = bubbles[b_idx]
    if me.active == 0:
        return
    if me.site_cleared == 0 and me.site_i >= 0:
        site_active[me.site_i, me.site_j, me.site_k] = 0
    me.active = 0
    me.radius = 0.0
    bubbles[b_idx] = me
    slot_claim[b_idx] = 0


# ---------------------------------------------------------------------------
# Milestone C: two-way energy coupling (latent-heat sink)
# ---------------------------------------------------------------------------


@wp.kernel
def scatter_latent_heat(
    bubbles: wp.array(dtype=Bubble),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    current_time: float,
    # Physics
    rho_l: float,
    cp_l: float,
    rho_v: float,
    h_lv: float,
    T_sat_k: float,
    mat_fluid: int,
):
    """Trilinearly scatter the latent-heat sink from each growing bubble into
    the 8 surrounding water cells.

    Per dev-guide sec.3.5: the instantaneous rate of latent-heat extraction from
    the bubble's neighbourhood is ``Q_dot_b = rho_v * h_lv * 4pi R^2 * dR/dt`` [W].
    The total energy removed this step is ``Q_dot_b * dt`` [J]. Distributed with
    trilinear weights ``w_ijk`` across 8 cells (which sum to 1), each cell's
    temperature drops by ``delta T_ijk = w_ijk * Q_dot_b * dt / (rho_l c_p,l dx^3)``.

    Only MAT_FLUID cells receive the sink (non-fluid cells in the support are
    skipped and their share is lost -- keeps the kernel robust near walls and
    the free surface). dR/dt is computed analytically from the Mikic-Rohsenow
    law ``R ? sqrtt`` as ``dR/dt = R / (2*age)``.

    Temperature-gated: bubbles surrounded by liquid at or below T_sat do not
    extract heat (no growth drive when the Jakob number is zero).
    """
    b_idx = wp.tid()
    bubble = bubbles[b_idx]
    if bubble.active == 0:
        return

    age = current_time - bubble.birth_time
    if age <= 1.0e-6:
        return  # just nucleated: no growth yet to scatter

    # Temperature gate: only scatter when the local liquid is superheated.
    T_local = _sample_cell_scalar(T, bubble.position, origin, dx)
    if T_local <= T_sat_k:
        return

    # Analytic dR/dt from Mikic-Rohsenow monotonic growth.
    dR_dt = bubble.radius / (2.0 * age)
    if dR_dt <= 0.0:
        return

    # Total latent-heat power and energy extracted this step.
    Q_b = rho_v * h_lv * 4.0 * 3.14159265358979 * bubble.radius * bubble.radius * dR_dt  # [W]
    E_step = Q_b * dt                                                                     # [J]
    # Convert to a reference temperature drop: delta T_ref = E_step / (rho_l c_p dx^3)
    cell_volume = dx * dx * dx
    dT_ref = E_step / (rho_l * cp_l * cell_volume)

    # Trilinear indexing at cell-centre lattice.
    fx = (bubble.position[0] - origin[0]) / dx - 0.5
    fy = (bubble.position[1] - origin[1]) / dx - 0.5
    fz = (bubble.position[2] - origin[2]) / dx - 0.5
    nx = T.shape[0]
    ny = T.shape[1]
    nz = T.shape[2]
    fx = wp.clamp(fx, 0.0, float(nx - 1) - 1.0e-6)
    fy = wp.clamp(fy, 0.0, float(ny - 1) - 1.0e-6)
    fz = wp.clamp(fz, 0.0, float(nz - 1) - 1.0e-6)

    i0 = int(fx); j0 = int(fy); k0 = int(fz)
    tx = fx - float(i0); ty = fy - float(j0); tz = fz - float(k0)

    # 8 corner weights.
    w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
    w100 = tx * (1.0 - ty) * (1.0 - tz)
    w010 = (1.0 - tx) * ty * (1.0 - tz)
    w110 = tx * ty * (1.0 - tz)
    w001 = (1.0 - tx) * (1.0 - ty) * tz
    w101 = tx * (1.0 - ty) * tz
    w011 = (1.0 - tx) * ty * tz
    w111 = tx * ty * tz

    # Atomic-subtract dT_ref * weight from each fluid cell corner.
    if mat[i0, j0, k0] == mat_fluid:
        wp.atomic_sub(T, i0, j0, k0, dT_ref * w000)
    if mat[i0 + 1, j0, k0] == mat_fluid:
        wp.atomic_sub(T, i0 + 1, j0, k0, dT_ref * w100)
    if mat[i0, j0 + 1, k0] == mat_fluid:
        wp.atomic_sub(T, i0, j0 + 1, k0, dT_ref * w010)
    if mat[i0 + 1, j0 + 1, k0] == mat_fluid:
        wp.atomic_sub(T, i0 + 1, j0 + 1, k0, dT_ref * w110)
    if mat[i0, j0, k0 + 1] == mat_fluid:
        wp.atomic_sub(T, i0, j0, k0 + 1, dT_ref * w001)
    if mat[i0 + 1, j0, k0 + 1] == mat_fluid:
        wp.atomic_sub(T, i0 + 1, j0, k0 + 1, dT_ref * w101)
    if mat[i0, j0 + 1, k0 + 1] == mat_fluid:
        wp.atomic_sub(T, i0, j0 + 1, k0 + 1, dT_ref * w011)
    if mat[i0 + 1, j0 + 1, k0 + 1] == mat_fluid:
        wp.atomic_sub(T, i0 + 1, j0 + 1, k0 + 1, dT_ref * w111)


# ---------------------------------------------------------------------------
# Milestone C-2: Eulerian wall boiling flux (microlayer evaporation)
# ---------------------------------------------------------------------------


@wp.kernel
def apply_wall_boiling_flux(
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    nucleation_table: wp.array(dtype=float),
    rho_arr: wp.array(dtype=float),
    cp_arr: wp.array(dtype=float),
    dx: float,
    dt: float,
    T_sat_k: float,
    dT_onb: float,
    dT_max: float,
    n_table: int,
    theta_rad: float,
    sigma: float,
    g_mag: float,
    rho_l: float,
    rho_v: float,
    h_lv: float,
    q_stove_cap: float,
    mat_pot_wall: int,
    mat_fluid: int,
):
    """Eulerian wall heat-flux model for nucleate boiling.

    For each pot-wall cell that borders fluid above, compute the boiling
    heat flux from the Kocamustafaogullari-Ishii site density, Fritz
    departure diameter, and Cole departure frequency:

        q_boil = N_a(dT_w) * f * rho_v * h_lv * (pi/6) * D_d^3   [W/m^2]

    This represents microlayer evaporation under bubbles at nucleation
    sites -- the dominant wall-cooling mechanism in nucleate boiling,
    accounting for 50-70 % of total heat transfer.  The heat is subtracted
    directly from the wall cell.

    Two caps enforce physical conservation:

    1. ``q_boil <= q_stove_cap``. The K-I^4.4 law explodes: at dT_w = 13 K it
       predicts >400 kW/m^2, more than 10x the 30 kW/m^2 stove supply. An
       uncapped kernel transiently extracts more than the wall receives,
       pulling sensible heat from the bulk fluid through the wall and
       driving the bulk below saturation. The cap enforces that the vapor
       pathway cannot reject more than the wall actually receives.

    2. ``dT_remove <= max(dT_w - dT_onb, 0)``. Keeps the wall above the
       nucleation-onset threshold so it stays at least ONB above the
       adjacent fluid. This prevents the wall cell from ever becoming a
       conductive sink for the fluid above it.
    """
    i, j, k = wp.tid()

    if mat[i, j, k] != mat_pot_wall:
        return
    if k + 1 >= T.shape[2]:
        return
    if mat[i, j, k + 1] != mat_fluid:
        return

    dT_w = T[i, j, k] - T_sat_k
    if dT_w < dT_onb:
        return

    # Physical gate: microlayer evaporation is impossible into subcooled
    # liquid -- bubbles would condense immediately. Require the adjacent
    # fluid cell to be at or within a small tolerance of saturation before
    # the vapor pathway activates. This ensures stove heat first warms the
    # bulk fluid to saturation via conduction + convection; only then does
    # nucleate boiling start rejecting heat at the wall.
    T_fluid_adj = T[i, j, k + 1]
    if T_fluid_adj < T_sat_k - 0.5:
        return

    N_a = lookup_site_density(nucleation_table, dT_w, dT_max, n_table)
    D_d = fritz_departure_diameter(theta_rad, sigma, g_mag, rho_l, rho_v)
    f = cole_frequency(D_d, g_mag, rho_l, rho_v)

    # pi/6 * D_d^3
    V_bubble = 0.5235987755982988 * D_d * D_d * D_d
    q_boil = N_a * f * rho_v * h_lv * V_bubble  # [W/m^2]

    # Conservation cap: wall cannot reject more than the stove supplies.
    # This is the nucleate-boiling partition of Rohsenow's q''_w; the
    # remaining wall heat reaches the fluid via conduction + convection.
    q_boil = wp.min(q_boil, q_stove_cap)

    m = mat[i, j, k]
    dT_remove = q_boil * dt / (rho_arr[m] * cp_arr[m] * dx)

    # Stay at least ONB above T_sat, so wall remains a net source for fluid.
    dT_max_remove = wp.max(dT_w - dT_onb, float(0.0))
    dT_remove = wp.min(dT_remove, dT_max_remove)

    T[i, j, k] = T[i, j, k] - dT_remove


# ---------------------------------------------------------------------------
# Milestone D: two-way momentum coupling + VOF alpha reduction
# ---------------------------------------------------------------------------


@wp.kernel
def scatter_bubble_momentum(
    bubbles: wp.array(dtype=Bubble),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    # Physics
    rho_l: float,
    rho_v: float,
    g_mag: float,
    mat_fluid: int,
):
    """Dev-guide sec.3.4: scatter each bubble's excess-buoyancy body force to
    the 8 nearest z-face velocities via trilinear weights.

    Per unit cell volume the force density is
        f_b = V_b * (rho_l - rho_v) * g  [N/m^3 of bubble, distributed trilinearly]
    The update to uz on a neighbouring z-face is
        delta uz = w * V_b * (rho_l - rho_v) * g * dt / (rho_l * V_cell)
    where ``w`` is that face's trilinear weight and V_cell = dx^3.
    Non-fluid z-faces (wall, air) are skipped so momentum stays in water.
    """
    b_idx = wp.tid()
    bubble = bubbles[b_idx]
    if bubble.active == 0:
        return

    # Per-bubble total upward force (N), distributed across 8 z-faces.
    V_b = 4.18879020478639 * bubble.radius * bubble.radius * bubble.radius  # 4/3*pi*R^3
    F_total = V_b * (rho_l - rho_v) * g_mag
    # Convert to delta uz contribution per unit weight:
    #   delta uz = F_total * w * dt / (rho_l * V_cell)
    cell_volume = dx * dx * dx
    dUz_ref = F_total * dt / (rho_l * cell_volume)

    # uz lives on z-faces at (i+0.5, j+0.5, k)*dx positions -> offset (0.5, 0.5, 0.0).
    fx = (bubble.position[0] - origin[0]) / dx - 0.5
    fy = (bubble.position[1] - origin[1]) / dx - 0.5
    fz = (bubble.position[2] - origin[2]) / dx
    nx_ = uz.shape[0]
    ny_ = uz.shape[1]
    nz_ = uz.shape[2]
    fx = wp.clamp(fx, 0.0, float(nx_ - 1) - 1.0e-6)
    fy = wp.clamp(fy, 0.0, float(ny_ - 1) - 1.0e-6)
    fz = wp.clamp(fz, 0.0, float(nz_ - 1) - 1.0e-6)

    i0 = int(fx); j0 = int(fy); k0 = int(fz)
    tx = fx - float(i0); ty = fy - float(j0); tz = fz - float(k0)

    w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
    w100 = tx * (1.0 - ty) * (1.0 - tz)
    w010 = (1.0 - tx) * ty * (1.0 - tz)
    w110 = tx * ty * (1.0 - tz)
    w001 = (1.0 - tx) * (1.0 - ty) * tz
    w101 = tx * (1.0 - ty) * tz
    w011 = (1.0 - tx) * ty * tz
    w111 = tx * ty * tz

    # A z-face at index (i, j, k) sits between cell (i, j, k-1) and (i, j, k).
    # Only scatter if at least one side is fluid (so momentum stays in water).
    # Use a simpler guard: require the face's upper cell (i, j, k) to be fluid.
    if k0 > 0 and mat[i0, j0, k0] == mat_fluid:
        wp.atomic_add(uz, i0, j0, k0, dUz_ref * w000)
    if k0 > 0 and i0 + 1 < mat.shape[0] and mat[i0 + 1, j0, k0] == mat_fluid:
        wp.atomic_add(uz, i0 + 1, j0, k0, dUz_ref * w100)
    if k0 > 0 and j0 + 1 < mat.shape[1] and mat[i0, j0 + 1, k0] == mat_fluid:
        wp.atomic_add(uz, i0, j0 + 1, k0, dUz_ref * w010)
    if k0 > 0 and i0 + 1 < mat.shape[0] and j0 + 1 < mat.shape[1] and mat[i0 + 1, j0 + 1, k0] == mat_fluid:
        wp.atomic_add(uz, i0 + 1, j0 + 1, k0, dUz_ref * w110)
    if k0 + 1 < mat.shape[2] and mat[i0, j0, k0 + 1] == mat_fluid:
        wp.atomic_add(uz, i0, j0, k0 + 1, dUz_ref * w001)
    if k0 + 1 < mat.shape[2] and i0 + 1 < mat.shape[0] and mat[i0 + 1, j0, k0 + 1] == mat_fluid:
        wp.atomic_add(uz, i0 + 1, j0, k0 + 1, dUz_ref * w101)
    if k0 + 1 < mat.shape[2] and j0 + 1 < mat.shape[1] and mat[i0, j0 + 1, k0 + 1] == mat_fluid:
        wp.atomic_add(uz, i0, j0 + 1, k0 + 1, dUz_ref * w011)
    if k0 + 1 < mat.shape[2] and i0 + 1 < mat.shape[0] and j0 + 1 < mat.shape[1] and mat[i0 + 1, j0 + 1, k0 + 1] == mat_fluid:
        wp.atomic_add(uz, i0 + 1, j0 + 1, k0 + 1, dUz_ref * w111)


@wp.kernel
def reduce_water_alpha_by_bubble_occupancy(
    bubbles: wp.array(dtype=Bubble),
    water_alpha: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    mat_fluid: int,
):
    """Reduce ``water_alpha`` in the 8 cells around each active bubble.

    The bubble's volume fraction within each cell is approximated as
    ``V_b / V_cell * w_ijk`` where w_ijk is the trilinear weight. The sum
    of weights is 1, so total reduction across the 8 cells equals
    ``V_b / V_cell`` -- the exact displacement by the bubble (if it fits in
    an 8-cell stencil). Final alpha is clamped to [0, 1].
    """
    b_idx = wp.tid()
    bubble = bubbles[b_idx]
    if bubble.active == 0:
        return

    V_b = 4.18879020478639 * bubble.radius * bubble.radius * bubble.radius
    V_cell = dx * dx * dx
    occ_ref = V_b / V_cell

    fx = (bubble.position[0] - origin[0]) / dx - 0.5
    fy = (bubble.position[1] - origin[1]) / dx - 0.5
    fz = (bubble.position[2] - origin[2]) / dx - 0.5
    nx = water_alpha.shape[0]
    ny = water_alpha.shape[1]
    nz = water_alpha.shape[2]
    fx = wp.clamp(fx, 0.0, float(nx - 1) - 1.0e-6)
    fy = wp.clamp(fy, 0.0, float(ny - 1) - 1.0e-6)
    fz = wp.clamp(fz, 0.0, float(nz - 1) - 1.0e-6)

    i0 = int(fx); j0 = int(fy); k0 = int(fz)
    tx = fx - float(i0); ty = fy - float(j0); tz = fz - float(k0)

    w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
    w100 = tx * (1.0 - ty) * (1.0 - tz)
    w010 = (1.0 - tx) * ty * (1.0 - tz)
    w110 = tx * ty * (1.0 - tz)
    w001 = (1.0 - tx) * (1.0 - ty) * tz
    w101 = tx * (1.0 - ty) * tz
    w011 = (1.0 - tx) * ty * tz
    w111 = tx * ty * tz

    # Only reduce alpha in water cells.
    if mat[i0, j0, k0] == mat_fluid:
        wp.atomic_sub(water_alpha, i0, j0, k0, occ_ref * w000)
    if mat[i0 + 1, j0, k0] == mat_fluid:
        wp.atomic_sub(water_alpha, i0 + 1, j0, k0, occ_ref * w100)
    if mat[i0, j0 + 1, k0] == mat_fluid:
        wp.atomic_sub(water_alpha, i0, j0 + 1, k0, occ_ref * w010)
    if mat[i0 + 1, j0 + 1, k0] == mat_fluid:
        wp.atomic_sub(water_alpha, i0 + 1, j0 + 1, k0, occ_ref * w110)
    if mat[i0, j0, k0 + 1] == mat_fluid:
        wp.atomic_sub(water_alpha, i0, j0, k0 + 1, occ_ref * w001)
    if mat[i0 + 1, j0, k0 + 1] == mat_fluid:
        wp.atomic_sub(water_alpha, i0 + 1, j0, k0 + 1, occ_ref * w101)
    if mat[i0, j0 + 1, k0 + 1] == mat_fluid:
        wp.atomic_sub(water_alpha, i0, j0 + 1, k0 + 1, occ_ref * w011)
    if mat[i0 + 1, j0 + 1, k0 + 1] == mat_fluid:
        wp.atomic_sub(water_alpha, i0 + 1, j0 + 1, k0 + 1, occ_ref * w111)


@wp.kernel
def clamp_alpha_nonnegative(water_alpha: wp.array3d(dtype=float)):
    """Clamp alpha to [0, 1] after the atomic-sub scatter in case of piling up."""
    i, j, k = wp.tid()
    a = water_alpha[i, j, k]
    if a < 0.0:
        water_alpha[i, j, k] = 0.0
    elif a > 1.0:
        water_alpha[i, j, k] = 1.0


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

    Safe to call every step. Cheap -- one kernel pass over (nx, ny, nz).
    """
    nx, ny, nz = grid.shape
    T_sat_k = 373.15  # water saturation at 1 atm

    wp.launch(
        detect_nucleation_sites,
        dim=(nx, ny, nz),
        inputs=[
            pool.bubbles,
            pool.slot_claim,
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


def step_update_bubbles(
    grid: Grid,
    pool: BubblePool,
    cfg: ScenarioConfig,
    dt: float,
    sim_time: float,
    device: str = "cuda:0",
) -> None:
    """Milestone-B entry point: grow every active bubble by Mikic-Rohsenow,
    check Fritz departure, advect, and vent.

    Runs one kernel pass over the full pool. Inactive bubbles return early.
    """
    T_sat_k = 373.15
    rho_l = 997.0
    rho_v = 0.598
    cp_l = 4186.0
    k_l = 0.606
    h_lv = 2.257e6
    sigma = 0.0589
    g_mag = 9.81

    # Water line z = base_thickness + fill_fraction * inner_height
    h_inner = cfg.pot.height_m - cfg.pot.base_thickness_m
    water_line_z = cfg.pot.base_thickness_m + cfg.water.fill_fraction * h_inner

    wp.launch(
        update_bubbles,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles,
            pool.slot_claim,
            pool.site_active,
            pool.needs_fragment,
            grid.T,
            grid.mat,
            grid.ux, grid.uy, grid.uz,
            wp.vec3(*grid.origin),
            grid.dx,
            dt,
            sim_time,
            water_line_z,
            T_sat_k, rho_l, rho_v, cp_l, k_l, h_lv, sigma,
            cfg.boiling.contact_angle_rad,
            g_mag,
            cfg.boiling.initial_bubble_radius_m,
            cfg.boiling.fragmentation_radius_m,
            cfg.boiling.max_bubble_radius_m,
            MAT_FLUID,
        ],
        device=device,
    )


def step_coalesce_bubbles(
    pool: BubblePool,
    cfg: ScenarioConfig,
    device: str = "cuda:0",
) -> None:
    """Phase 8 M3: spatial-hash coalescence.

    Three sub-kernels:
      1. ``bin_bubbles`` — each active bubble assigns itself to a 3D bin.
      2. ``find_merge_targets`` — each bubble scans its bin + 26 neighbours
         for overlap candidates; CAS to claim absorbees.
      3. ``apply_merges_absorber_pass`` + ``apply_merges_absorbee_pass`` —
         realize the merges (volume-conserving combine on the absorber,
         deactivate absorbees).

    Per-step cost: O(N) thanks to the spatial hash. Skipped when
    ``cfg.boiling.coalescence_enabled`` is False.
    """
    if not cfg.boiling.coalescence_enabled:
        return
    n_bins = pool.n_bins_x * pool.n_bins_y * pool.n_bins_z
    max_per_bin = int(cfg.boiling.coalescence_max_per_bin)

    # Reset per-step scratch.
    wp.launch(reset_bin_counts, dim=n_bins,
              inputs=[pool.bin_counts], device=device)
    wp.launch(reset_merge_targets, dim=pool.max_bubbles,
              inputs=[pool.merge_target], device=device)

    ox, oy, oz = pool.bin_origin
    bs = pool.bin_size_m

    # Pass A: bin every active bubble.
    wp.launch(
        bin_bubbles,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles, pool.bin_counts, pool.bin_lists,
            ox, oy, oz, bs,
            pool.n_bins_x, pool.n_bins_y, pool.n_bins_z,
            max_per_bin,
        ],
        device=device,
    )
    # Pass B: find overlapping pairs and claim absorbees.
    wp.launch(
        find_merge_targets,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles, pool.bin_counts, pool.bin_lists, pool.merge_target,
            ox, oy, oz, bs,
            pool.n_bins_x, pool.n_bins_y, pool.n_bins_z,
            max_per_bin,
        ],
        device=device,
    )
    # Pass C1: absorbers fold absorbees in.
    wp.launch(
        apply_merges_absorber_pass,
        dim=pool.max_bubbles,
        inputs=[pool.bubbles, pool.merge_target, pool.max_bubbles],
        device=device,
    )
    # Pass C2: deactivate absorbees (must run AFTER C1).
    wp.launch(
        apply_merges_absorbee_pass,
        dim=pool.max_bubbles,
        inputs=[pool.bubbles, pool.slot_claim, pool.merge_target,
                pool.site_active],
        device=device,
    )


def step_fragment_bubbles(
    pool: BubblePool,
    cfg: ScenarioConfig,
    device: str = "cuda:0",
) -> None:
    """Phase 8 M2: Rayleigh-Taylor binary breakup pass.

    Reads ``pool.needs_fragment`` (set by ``update_bubbles`` when a bubble
    crossed R_frag this step) and splits each flagged bubble into two
    equal-volume daughters. Pool-full graceful degradation; flag is reset
    after each pass so this kernel is idempotent if launched twice.
    """
    wp.launch(
        fragment_bubbles,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles,
            pool.slot_claim,
            pool.needs_fragment,
            pool.max_bubbles,
            cfg.boiling.fragmentation_radius_m,
        ],
        device=device,
    )


def step_scatter_latent_heat(
    grid: Grid,
    pool: BubblePool,
    cfg: ScenarioConfig,
    dt: float,
    sim_time: float,
    device: str = "cuda:0",
) -> None:
    """Milestone-C two-way energy coupling: remove latent-heat energy from
    water cells in the neighbourhood of every growing bubble.

    Must run AFTER :func:`step_update_bubbles` so radii are current, but
    BEFORE :func:`step_nucleation` so newly-spawned bubbles (age=0) don't
    scatter on their birth step (their dR/dt is the analytic early-growth
    slope which is unbounded at t=0; the ``age <= 1e-6`` guard handles this).
    """
    T_sat_k = 373.15
    rho_l = 997.0
    cp_l = 4186.0
    rho_v = 0.598
    h_lv = 2.257e6

    wp.launch(
        scatter_latent_heat,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles,
            grid.T,
            grid.mat,
            wp.vec3(*grid.origin),
            grid.dx,
            dt,
            sim_time,
            rho_l, cp_l, rho_v, h_lv,
            T_sat_k,
            MAT_FLUID,
        ],
        device=device,
    )


def step_wall_boiling_flux(
    grid: Grid,
    pool: BubblePool,
    cfg: ScenarioConfig,
    props,  # MaterialProps from thermal module (carries rho_wp, cp_wp)
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Eulerian wall boiling flux: directly cool pot-wall cells at nucleation
    sites via the Kocamustafaogullari-Ishii + Fritz + Cole model.

    Must be called AFTER :func:`thermal.conduct_one_step` so the stove flux
    is already applied.  The boiling flux is then the additional source term
    that removes microlayer-evaporation energy from the wall.
    """
    nx, ny, nz = grid.shape
    T_sat_k = 373.15
    rho_l = 997.0
    rho_v = 0.598
    h_lv = 2.257e6
    sigma = 0.0589
    g_mag = 9.81

    q_stove_cap = cfg.heating.base_heat_flux_w_per_m2

    wp.launch(
        apply_wall_boiling_flux,
        dim=(nx, ny, nz),
        inputs=[
            grid.T,
            grid.mat,
            pool.nucleation_table,
            props.rho_wp,
            props.cp_wp,
            grid.dx,
            dt,
            T_sat_k,
            cfg.boiling.dT_onb_k,
            _DT_MAX_K,
            _N_ENTRIES,
            cfg.boiling.contact_angle_rad,
            sigma,
            g_mag,
            rho_l,
            rho_v,
            h_lv,
            q_stove_cap,
            MAT_POT_WALL,
            MAT_FLUID,
        ],
        device=device,
    )


def step_scatter_momentum(
    grid: Grid,
    pool: BubblePool,
    cfg: ScenarioConfig,
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Milestone-D two-way momentum coupling: add each bubble's excess-buoyancy
    force to the nearby vertical face velocities. Must run AFTER update_bubbles
    (so radii are current) and should run alongside the Boussinesq buoyancy step.
    """
    rho_l = 997.0
    rho_v = 0.598
    g_mag = 9.81

    wp.launch(
        scatter_bubble_momentum,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles,
            grid.uz,
            grid.mat,
            wp.vec3(*grid.origin),
            grid.dx,
            dt,
            rho_l, rho_v, g_mag,
            MAT_FLUID,
        ],
        device=device,
    )


def step_reduce_water_alpha(
    grid: Grid,
    pool: BubblePool,
    device: str = "cuda:0",
) -> None:
    """Milestone-D VOF: reset ``water_alpha`` to the static water mask and
    then scatter each active bubble's volume fraction into its 8-cell stencil.

    Reversible: every step starts from the clean baseline, so bubbles that
    left a cell no longer claim alpha there.
    """
    if grid.water_alpha_base is None:
        return  # boiling disabled or pool not allocated

    nx, ny, nz = grid.shape
    # Reset alpha to baseline.
    wp.copy(grid.water_alpha, grid.water_alpha_base)

    wp.launch(
        reduce_water_alpha_by_bubble_occupancy,
        dim=pool.max_bubbles,
        inputs=[
            pool.bubbles,
            grid.water_alpha,
            grid.mat,
            wp.vec3(*grid.origin),
            grid.dx,
            MAT_FLUID,
        ],
        device=device,
    )
    # Clamp to [0, 1] after the sub scatter.
    wp.launch(
        clamp_alpha_nonnegative,
        dim=(nx, ny, nz),
        inputs=[grid.water_alpha],
        device=device,
    )


def step_bubbles(
    grid: Grid,
    pool: BubblePool,
    cfg: ScenarioConfig,
    dt: float,
    sim_time: float,
    step_count: int,
    device: str = "cuda:0",
) -> None:
    """Full Phase-3 bubble step (Milestones A-D), RPI-partitioned.

    Order matters:
      1. update_bubbles -- grow, depart, advect, vent (reads current T, u, alpha)
      2. scatter_latent_heat -- remove energy from superheated bulk fluid
      3. scatter_bubble_momentum -- push water upward around each bubble
      4. reduce_water_alpha -- reset VOF alpha from baseline and re-scatter
      5. step_nucleation -- spawn new bubbles on newly-superheated sites

    **Two-kernel RPI partition of the latent-heat sink:**

    * :func:`scatter_latent_heat` handles the *bulk* portion: bubbles rising
      through superheated liquid absorb latent heat at their current position.
      Self-gates on T_local > T_sat (no over-extraction into subcooled bulk),
      which keeps the mean fluid near saturation without going below.

    * :func:`step_wall_boiling_flux` (called from Simulation.step) handles the
      *wall* portion: microlayer evaporation directly cools pot-wall cells at
      nucleation sites. Capped at q_stove (conservation) and gated on adjacent
      fluid at saturation (microlayer can't fire into subcooled liquid).

    Together they reproduce the RPI heat-flux partition
    q_total = q_nb + q_conv + q_quench: wall kernel is q_nb, conduction to
    fluid is q_conv+q_quench, Lagrangian scatter caps the bulk that q_nb
    didn't ferry away directly. This is the combination the original Phase-3
    plan implicitly assumed but didn't prescribe -- without the wall kernel
    the wall runs away to ~155 C; without the bulk scatter the fluid goes
    subcooled because the wall kernel diverts all stove heat to vapor.
    """
    step_update_bubbles(grid, pool, cfg, dt, sim_time, device=device)
    step_fragment_bubbles(pool, cfg, device=device)
    step_coalesce_bubbles(pool, cfg, device=device)
    step_scatter_latent_heat(grid, pool, cfg, dt, sim_time, device=device)
    step_scatter_momentum(grid, pool, cfg, dt, device=device)
    step_reduce_water_alpha(grid, pool, device=device)
    step_nucleation(grid, pool, cfg, dt, sim_time, step_count, device=device)


# Expose module-level constants for tests
N_TABLE_ENTRIES = _N_ENTRIES
DT_TABLE_MAX_K = _DT_MAX_K
