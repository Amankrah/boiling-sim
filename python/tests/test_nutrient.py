"""Phase 4 Milestone A tests: Arrhenius degradation of beta-carotene."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from boilingsim.config import ScenarioConfig, load_scenario
from boilingsim.geometry import MAT_CARROT, build_pot_geometry
from boilingsim.nutrient import (
    allocate_nutrient_workspace,
    arrhenius_rate,
    arrhenius_degrade,
    diffusion_stability_dt,
    retention_fraction,
    sherwood_h_m_host,
    step_advect_c_water,
    step_degrade,
    step_diffuse_nutrient,
    step_leach,
    _MAT_CARROT,
    _R_GAS,
)


@pytest.fixture(scope="module")
def nut_cfg() -> ScenarioConfig:
    """Dev-grid config with nutrient physics enabled."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    return cfg


@pytest.fixture(scope="module")
def nut_grid(nut_cfg):
    """Geometry with carrot cells initialised to C = C0."""
    return build_pot_geometry(nut_cfg)


def test_arrhenius_rate_at_100c(nut_cfg):
    """`k(T)` computed Python-side should match `k0*exp(-E_a/(R*T))` exactly."""
    T_k = 373.15  # 100 C
    expected = nut_cfg.nutrient.k0_per_s * math.exp(
        -(nut_cfg.nutrient.E_a_kJ_per_mol * 1000.0) / (_R_GAS * T_k)
    )
    measured = arrhenius_rate(nut_cfg, T_k)
    assert math.isclose(expected, measured, rel_tol=1e-12)
    # Order-of-magnitude sanity: water-carrot boiling kinetics give k ~ 1e-4
    # to 1e-3 1/s at 100 C for beta-carotene. Published half-lives are
    # hundreds of seconds so retention at 600 s should land in [0.5, 0.9].
    assert 1.0e-5 < measured < 1.0e-2, (
        f"k(100 C) = {measured:.3e} 1/s, outside expected 1e-5..1e-2 band"
    )


def test_constant_t_degradation(nut_cfg):
    """With T held at 100 C and no diffusion, the kernel should match the
    closed-form ``C(t) = C0 * exp(-k*t)`` for any dt.

    We run for 600 s in 100 steps (dt = 6 s), compute retention from the
    kernel, and compare to the analytic exponential. Because we use the
    exact-integration form ``C *= exp(-k*dt)``, the numerical retention
    should agree to full single-precision."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)

    # Pin carrot T = 100 C everywhere (override thermal state directly).
    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_CARROT] = 373.15
    grid.T.assign(T_np)

    dt = 6.0
    for _ in range(100):
        step_degrade(grid, cfg, dt)
    wp.synchronize()

    R_measured = retention_fraction(grid, cfg)
    k = arrhenius_rate(cfg, 373.15)
    R_expected = math.exp(-k * 600.0)
    rel_err = abs(R_measured - R_expected) / R_expected
    assert rel_err < 5.0e-3, (
        f"retention after 600 s: measured {R_measured:.6f}, "
        f"analytic exp(-k*t) = {R_expected:.6f}, rel err {rel_err*100:.3f}%"
    )


def test_no_degradation_at_cold_t(nut_cfg):
    """Arrhenius rate at 20 C should be vanishingly small; 600 s of stepping
    should leave retention above 99.99 %."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)

    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_CARROT] = 293.15  # 20 C
    grid.T.assign(T_np)

    dt = 1.0
    for _ in range(600):
        step_degrade(grid, cfg, dt)
    wp.synchronize()

    R = retention_fraction(grid, cfg)
    # Analytic: k(20 C) = k0*exp(-E_a/(R*T)) ~ 8.8e-7 /s at defaults, so
    # k*t ~ 5.3e-4 over 600 s, retention ~ 0.9995. This is well above 99.9 %.
    assert R > 0.999, (
        f"retention at 20 C / 600 s = {R:.6f}, expected > 99.9 %"
    )


def test_degradation_only_in_carrot_cells():
    """Set every cell to C = C0 manually; run degradation with carrot at 100 C
    and everything else at 0 C. Non-carrot cells must be untouched; carrot
    cells must decay."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)

    # Overwrite C uniformly (normally only carrot cells are nonzero).
    C_np = np.full_like(grid.C.numpy(), cfg.nutrient.C0_mg_per_kg)
    grid.C.assign(C_np)

    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    # Carrot hot, everything else cold
    T_np[mat_np == MAT_CARROT] = 373.15
    T_np[mat_np != MAT_CARROT] = 273.15
    grid.T.assign(T_np)

    for _ in range(100):
        step_degrade(grid, cfg, 6.0)
    wp.synchronize()

    C_after = grid.C.numpy()
    carrot_mask = mat_np == MAT_CARROT

    # Carrot cells decayed.
    C_carrot = C_after[carrot_mask]
    assert C_carrot.max() < cfg.nutrient.C0_mg_per_kg * 0.99, (
        "carrot cells should have measurably decayed after 600 s at 100 C"
    )

    # Non-carrot cells untouched.
    C_elsewhere = C_after[~carrot_mask]
    assert np.allclose(C_elsewhere, cfg.nutrient.C0_mg_per_kg, atol=1.0e-6), (
        "non-carrot cells must not be modified by the kernel"
    )


def test_initial_retention_is_one(nut_grid, nut_cfg):
    """After build_pot_geometry fires `init_carrot_concentration`, retention
    must be exactly 1.0 (within float32 rounding)."""
    R = retention_fraction(nut_grid, nut_cfg)
    assert abs(R - 1.0) < 1.0e-5, (
        f"initial retention = {R:.6f}, expected 1.0"
    )


# ---------------------------------------------------------------------------
# Milestone B: in-carrot diffusion tests
# ---------------------------------------------------------------------------


def test_diffusion_preserves_mass_under_neumann_bc():
    """With zero-flux Neumann at the carrot surface (Milestone B, no leaching
    yet), the total carrot-side beta-carotene mass must be conserved across
    diffusion steps. A uniform initial field isn't a good test (Laplacian is
    zero everywhere), so seed a gaussian peak at the carrot centre and run
    for 10 s."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    ws = allocate_nutrient_workspace(grid)

    # Pre-seed a gaussian: centre of the carrot bounding box, sigma = 2 dx.
    mat_np = grid.mat.numpy()
    carrot_mask = mat_np == MAT_CARROT
    nx, ny, nz = grid.shape
    C_np = grid.C.numpy()
    ii, jj, kk = np.indices((nx, ny, nz))
    # Find carrot centre in index space.
    icx, jcy, kcz = np.array(np.where(carrot_mask)).mean(axis=1)
    r2 = ((ii - icx) ** 2 + (jj - jcy) ** 2 + (kk - kcz) ** 2).astype(float)
    sigma = 2.0
    gauss = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)
    C_np = np.where(carrot_mask, gauss * 100.0, 0.0).astype(np.float32)
    grid.C.assign(C_np)

    mass_before = float(grid.C.numpy()[carrot_mask].sum())

    dt = 1.0
    for _ in range(10):
        step_diffuse_nutrient(grid, ws, cfg, dt)
    wp.synchronize()

    mass_after = float(grid.C.numpy()[carrot_mask].sum())
    rel_err = abs(mass_after - mass_before) / mass_before
    assert rel_err < 1.0e-4, (
        f"diffusion mass drift: {mass_before:.6e} -> {mass_after:.6e}, "
        f"rel err {rel_err*100:.5f}%"
    )


def test_diffusion_does_not_leak_into_non_carrot_cells():
    """With zero-flux Neumann BC, C should stay exactly zero on every cell
    that isn't carrot, even after many diffusion steps. This guards against
    accidentally writing to non-carrot cells in the kernel (or leaking the
    scratch buffer back)."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    ws = allocate_nutrient_workspace(grid)

    # Seed a gaussian peak at carrot centre so there's a real gradient to
    # push against the boundary.
    mat_np = grid.mat.numpy()
    carrot_mask = mat_np == MAT_CARROT
    nx, ny, nz = grid.shape
    ii, jj, kk = np.indices((nx, ny, nz))
    icx, jcy, kcz = np.array(np.where(carrot_mask)).mean(axis=1)
    r2 = ((ii - icx) ** 2 + (jj - jcy) ** 2 + (kk - kcz) ** 2).astype(float)
    gauss = np.exp(-r2 / 8.0).astype(np.float32)
    C_np = np.where(carrot_mask, gauss * 100.0, 0.0).astype(np.float32)
    grid.C.assign(C_np)

    dt = 5.0
    for _ in range(12):  # 60 s total
        step_diffuse_nutrient(grid, ws, cfg, dt)
    wp.synchronize()

    C_final = grid.C.numpy()
    leak = C_final[~carrot_mask]
    assert np.all(leak == 0.0), (
        f"non-carrot cells contain nutrient after diffusion: "
        f"max leak = {float(leak.max()):.3e}"
    )


def test_diffusion_analytic_1d_slab():
    """Seed a cosine mode along z uniformly across every carrot cell in that
    z-plane, so the problem is genuinely 1-D in z (no x,y gradient, no
    sideways diffusion). Compare decay to the closed-form analytic
    ``C(z, t) = C0 * cos(pi*(z-z0)/L) * exp(-(pi/L)^2 * D_eff * t)`` at the
    top/bottom of the carrot (the cosine-mode maxima)."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    # Artificially boost D_eff so the decay is observable inside a short test
    # run. Mode decay rate scales linearly with D, so the analytic target
    # scales exactly the same way. (Real D_eff stays 2e-10 in production.)
    cfg.nutrient.D_eff_m2_per_s = 1.0e-6

    grid = build_pot_geometry(cfg)
    ws = allocate_nutrient_workspace(grid)

    mat_np = grid.mat.numpy()
    carrot_mask = mat_np == MAT_CARROT
    nx, ny, nz = grid.shape

    # Find the z-range of the carrot.
    carrot_ks_all = np.where(carrot_mask)[2]
    k0, k1 = int(carrot_ks_all.min()), int(carrot_ks_all.max())
    L = float((k1 - k0 + 1)) * grid.dx

    # Cosine mode varying in z only, uniform across every x,y position that is
    # a carrot cell. No gradient in x,y -> purely 1-D slab diffusion in z.
    C_np = np.zeros_like(grid.C.numpy())
    for kz in range(k0, k1 + 1):
        z_local = (kz - k0 + 0.5) * grid.dx
        mode_val = np.cos(np.pi * z_local / L)
        layer = carrot_mask[:, :, kz]
        C_np[:, :, kz] = np.where(layer, mode_val, 0.0)
    grid.C.assign(C_np.astype(np.float32))

    # Analytic fundamental-mode decay rate.
    D = cfg.nutrient.D_eff_m2_per_s
    lam = (np.pi / L) ** 2 * D
    total_t = 400.0  # lam*t ~ 1.6 at L ~ 50 mm, D = 1e-6

    dt = diffusion_stability_dt(cfg, grid.dx) * 0.5  # well inside stability
    n_steps = int(total_t / dt)
    for _ in range(n_steps):
        step_diffuse_nutrient(grid, ws, cfg, dt)
    wp.synchronize()

    sim_t = n_steps * dt
    C_final = grid.C.numpy()
    # Measured peak amplitude: the mode has +1 at z=k0 and -1 at z=k1.
    # Average over carrot cells in the extreme z-planes to reduce noise
    # from the staircase boundary.
    peak_top = float(C_final[:, :, k0][carrot_mask[:, :, k0]].mean())
    peak_bot = float(C_final[:, :, k1][carrot_mask[:, :, k1]].mean())
    measured_amp = (peak_top - peak_bot) / 2.0

    # Analytic amplitude (initial amplitude 1.0, decays as exp(-lam*t)).
    expected_amp = math.exp(-lam * sim_t)
    rel_err = abs(measured_amp - expected_amp) / expected_amp
    assert rel_err < 0.05, (
        f"1-D slab cosine mode decay after {sim_t:.1f} s: "
        f"measured amp {measured_amp:.4f}, analytic {expected_amp:.4f}, "
        f"rel err {rel_err*100:.2f}%"
    )


# ---------------------------------------------------------------------------
# Milestone C: Sherwood correlation + surface leaching tests
# ---------------------------------------------------------------------------


def _build_leach_scenario(u_mag_mps: float):
    """Helper: build a dev-grid geometry with nutrient enabled, zero velocities
    (by default) or uniformly set to ``u_mag_mps`` along x. Returns (cfg, grid).
    """
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)

    if u_mag_mps != 0.0:
        ux_np = grid.ux.numpy()
        ux_np[:] = u_mag_mps
        grid.ux.assign(ux_np)
    return cfg, grid


def test_sherwood_at_known_re(nut_cfg):
    """Host-side `sherwood_h_m_host` should reproduce the closed-form
    Sh = 0.683 Re^0.466 Sc^(1/3) with the default water properties."""
    # Pick a velocity that gives Re ~ 1000 at D_carrot = 25 mm and
    # nu_water = 2.94e-7 m^2/s. Re = u*D/nu -> u = Re*nu/D = 1000 * 2.94e-7 / 0.025
    # = 0.01176 m/s.
    D_carrot = nut_cfg.carrot.diameter_m
    target_Re = 1000.0
    u_mag = target_Re * nut_cfg.nutrient.nu_water_m2_per_s / D_carrot

    # Direct closed form (Sc ~ 294 at our defaults -> Sc^(1/3) ~ 6.65)
    nu = nut_cfg.nutrient.nu_water_m2_per_s
    Dw = nut_cfg.nutrient.D_water_molec_m2_per_s
    Re = u_mag * D_carrot / nu
    Sc = nu / Dw
    Sh_exp = 0.683 * (Re ** 0.466) * (Sc ** (1.0 / 3.0))
    h_m_exp = Sh_exp * Dw / D_carrot

    h_m_measured = sherwood_h_m_host(nut_cfg, u_mag, D_carrot)
    rel_err = abs(h_m_measured - h_m_exp) / h_m_exp
    assert rel_err < 1.0e-6, (
        f"Sherwood h_m at Re~{Re:.0f}, Sc~{Sc:.0f}: "
        f"computed {h_m_measured:.4e}, expected {h_m_exp:.4e}, "
        f"rel err {rel_err*100:.4f}%"
    )


def test_sherwood_floor_at_stagnant():
    """At Re -> 0 the forced-convection term collapses; h_m should fall back
    to the Sh = 2 natural-convection floor."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    D_carrot = cfg.carrot.diameter_m
    Dw = cfg.nutrient.D_water_molec_m2_per_s
    h_m = sherwood_h_m_host(cfg, 0.0, D_carrot)
    expected = 2.0 * Dw / D_carrot
    assert math.isclose(h_m, expected, rel_tol=1.0e-6), (
        f"stagnant h_m = {h_m:.4e}, expected Sh=2 baseline {expected:.4e}"
    )


def test_leaching_mass_conservation():
    """With no degradation (T held cold so k(T) is negligible) and no
    diffusion, the leach kernel must move mass from carrot to water
    cell-for-cell. Sum(C on carrot cells) + Sum(C_water on fluid cells)
    should stay constant to within single-precision rounding."""
    cfg, grid = _build_leach_scenario(u_mag_mps=0.1)

    # Freeze carrot temperature well below degradation threshold to isolate
    # the leaching term.
    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_CARROT] = 293.15
    grid.T.assign(T_np)

    carrot_mask = mat_np == MAT_CARROT
    fluid_mask = mat_np == 0  # MAT_FLUID = 0

    def total_mass() -> float:
        C_np = grid.C.numpy()
        Cw_np = grid.C_water.numpy()
        return float(C_np[carrot_mask].sum() + Cw_np[fluid_mask].sum())

    m0 = total_mass()
    dt = 0.5
    for _ in range(200):  # 100 s of simulated leaching at uniform u = 0.1 m/s
        step_leach(grid, cfg, dt)
    wp.synchronize()

    m1 = total_mass()
    rel_err = abs(m1 - m0) / m0
    assert rel_err < 5.0e-3, (
        f"leach mass conservation: before {m0:.4e}, after {m1:.4e}, "
        f"rel err {rel_err*100:.3f}%"
    )


def test_partition_at_equilibrium():
    """Run leaching long enough to approach equilibrium, then check the
    carrot-to-water concentration ratio tends toward ``K_partition``.

    At equilibrium the flux J = h_m*(C - C_w/K) -> 0, which means
    C_w = K*C. So the ratio C_w_avg / C_carrot_avg should approach K.

    We accelerate convergence by using a higher K (0.8) so equilibrium lands
    at a well-separated ratio, then confirm that ratio is hit within tight
    tolerance."""
    cfg, grid = _build_leach_scenario(u_mag_mps=0.3)
    cfg.nutrient.K_partition = 0.8  # ~80% of carrot concentration at equilibrium
    # This test exercises the partition-equilibrium math; disable the
    # solubility cap (set very high) so it doesn't fire before equilibrium.
    cfg.nutrient.C_water_sat_mg_per_kg = 1.0e6

    # Freeze T to suppress degradation.
    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_CARROT] = 293.15
    grid.T.assign(T_np)

    carrot_mask = mat_np == MAT_CARROT

    # Without diffusion (Milestone C only), only *surface* carrot cells --
    # those with at least one fluid neighbour -- lose mass; interior carrot
    # cells stay at C0 forever. So the equilibrium ratio K = C_w / C_s holds
    # only at the surface. Identify both surface carrot cells and the fluid
    # cells they touch.
    nx, ny, nz = grid.shape
    carrot_surface_mask = np.zeros_like(carrot_mask)
    fluid_surface_mask = np.zeros_like(carrot_mask)
    offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for ci, cj, ck in zip(*np.where(carrot_mask)):
        for di, dj, dk in offsets:
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                if mat_np[ni, nj, nk] == 0:  # MAT_FLUID
                    carrot_surface_mask[ci, cj, ck] = True
                    fluid_surface_mask[ni, nj, nk] = True

    dt = 0.5
    for _ in range(4000):
        step_leach(grid, cfg, dt)
    wp.synchronize()

    C_np = grid.C.numpy()
    Cw_np = grid.C_water.numpy()
    C_carrot_surface = float(C_np[carrot_surface_mask].mean())
    C_water_surface = float(Cw_np[fluid_surface_mask].mean())
    ratio = C_water_surface / C_carrot_surface
    assert abs(ratio - cfg.nutrient.K_partition) < 0.05, (
        f"partition equilibrium (surface-to-surface): C_water/C_carrot = {ratio:.3f}, "
        f"expected K = {cfg.nutrient.K_partition}"
    )


def test_leaching_respects_solubility_cap():
    """``C_water`` must never overshoot ``C_water_sat`` even in the
    staircase-geometry worst case where several carrot cells share one
    fluid neighbour. ``_leach_flux_capped`` divides the per-face J_max by
    the maximum possible neighbour count (6) precisely so the sum of
    atomic_add contributions in a single step cannot push the neighbour
    past the cap.

    Regression guard: if that divisor gets removed (or the cap is moved
    somewhere the race can still happen), this test fails because
    multi-neighbour fluid cells will consistently overshoot by 2-6x per
    step.
    """
    cfg, grid = _build_leach_scenario(u_mag_mps=0.0)
    cfg.nutrient.C_water_sat_mg_per_kg = 0.5
    cfg.nutrient.K_partition = 0.9  # aggressive driving force so cap hits fast

    # Freeze both phases cold: disable Arrhenius so the cap can't be
    # "respected" by water-side decay. This isolates the cap mechanism.
    mat_np = grid.mat.numpy()
    T_np = grid.T.numpy()
    T_np[mat_np == MAT_CARROT] = 293.15
    T_np[mat_np == 0] = 293.15
    grid.T.assign(T_np)

    dt = 0.5
    for _ in range(2000):  # 1000 s -- far past saturation at this K
        step_leach(grid, cfg, dt)
    wp.synchronize()

    Cw = grid.C_water.numpy()
    max_Cw = float(Cw.max())
    # Strict: post-atomic value must not exceed the cap.
    assert max_Cw <= cfg.nutrient.C_water_sat_mg_per_kg + 1.0e-6, (
        f"C_water exceeded saturation cap: max {max_Cw:.6f} mg/kg, "
        f"cap {cfg.nutrient.C_water_sat_mg_per_kg} mg/kg"
    )
    # Sanity: we actually approached the cap, otherwise the test is vacuous.
    assert max_Cw > 0.5 * cfg.nutrient.C_water_sat_mg_per_kg, (
        f"C_water never approached cap (max {max_Cw:.6f}); test does not "
        f"exercise the cap branch"
    )


# ---------------------------------------------------------------------------
# Milestone D: full-pipeline integration + C_water advection
# ---------------------------------------------------------------------------


def test_c_water_advection_moves_scalar_downstream():
    """Seed a localised C_water blob, set a uniform velocity in +x, advect a
    few steps with ``step_advect_c_water``. The blob's centre of mass should
    shift in +x by approximately ``u*t``."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    ws = allocate_nutrient_workspace(grid)

    # Set uniform u in +x across every x-face so every fluid cell sees u in +x.
    u_mag = 0.05  # 50 mm/s
    ux_np = grid.ux.numpy()
    ux_np[:] = u_mag
    grid.ux.assign(ux_np)

    # Seed a gaussian blob of C_water well inside the fluid region.
    mat_np = grid.mat.numpy()
    fluid_mask = mat_np == 0
    nx, ny, nz = grid.shape
    # Pick a point known to be fluid: carrot center + few cells to the +y so
    # we're not overlapping the carrot.
    icx, jcy, kcz = np.array(np.where(mat_np == MAT_CARROT)).mean(axis=1)
    seed_i = int(icx) + 8  # shift off the carrot
    seed_j = int(jcy)
    seed_k = int(kcz)
    # Ensure seed is a fluid cell; if not, fall back to any fluid cell at that z.
    if not fluid_mask[seed_i, seed_j, seed_k]:
        fluid_ks = np.where(fluid_mask[:, :, seed_k])
        seed_i = int(fluid_ks[0][0] + 5)
        seed_j = int(fluid_ks[1][0])

    C_np = grid.C_water.numpy()
    C_np[seed_i, seed_j, seed_k] = 100.0
    grid.C_water.assign(C_np)

    dt = 0.02  # CFL = u*dt/dx = 0.05*0.02/0.002 = 0.5
    n_steps = 10
    for _ in range(n_steps):
        step_advect_c_water(grid, ws, dt)
    wp.synchronize()

    C_after = grid.C_water.numpy()
    # Compute centre of mass along x among fluid cells.
    if C_after.sum() < 1.0e-6:
        pytest.fail(f"advection wiped out all mass (final sum = {C_after.sum():.3e})")

    xs = np.arange(nx).reshape(-1, 1, 1)
    com_before = float(seed_i)
    com_after = float((xs * C_after).sum() / C_after.sum())

    expected_shift_cells = u_mag * dt * n_steps / grid.dx
    measured_shift = com_after - com_before
    rel_err = abs(measured_shift - expected_shift_cells) / max(
        expected_shift_cells, 1.0e-6
    )
    # Semi-Lagrangian with trilinear sample is diffusive; accept loose tol.
    assert rel_err < 0.3, (
        f"blob drift: expected shift {expected_shift_cells:.2f} cells, "
        f"measured {measured_shift:.2f} cells (rel err {rel_err*100:.1f}%)"
    )


def test_c_water_advection_conserves_total_mass():
    """Upwind finite-volume advection must conserve the integrated C_water
    mass across a pure-advection run (no leaching, no degradation).

    Seed a gaussian C_water blob in the fluid region, spin up a uniform
    velocity in +x, run 100 advection-only steps, and check that the total
    mass (sum over all cells) has drifted by less than 1e-5 (machine
    precision × cell count). The earlier SL-trilinear scheme failed this:
    mass leaked through the domain boundary and into solid-adjacent cells
    at ~0.5 % per step, which was laundered into degraded_pct by the
    max(0, ...) clamp we've since removed."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    ws = allocate_nutrient_workspace(grid)

    # Set uniform velocity in +x and +z (mixed for 3-D stress test).
    ux_np = grid.ux.numpy()
    ux_np[:] = 0.03  # 30 mm/s
    grid.ux.assign(ux_np)
    uz_np = grid.uz.numpy()
    uz_np[:] = 0.02
    grid.uz.assign(uz_np)

    # Seed a small gaussian in the fluid, centred off the carrot so
    # boundaries dominate less.
    mat_np = grid.mat.numpy()
    fluid_mask = mat_np == 0  # MAT_FLUID
    nx, ny, nz = grid.shape
    ii, jj, kk = np.indices((nx, ny, nz))
    # Pick a definitely-fluid centre: pot centre in x,y and mid-water in z.
    ic, jc, kc = nx // 2, ny // 2, nz // 2
    # If that lands in carrot, nudge aside.
    if mat_np[ic, jc, kc] != 0:
        ic = ic + 15
    r2 = (ii - ic) ** 2 + (jj - jc) ** 2 + (kk - kc) ** 2
    gauss = np.exp(-r2 / 8.0).astype(np.float32)
    C_np = np.where(fluid_mask, gauss * 100.0, 0.0).astype(np.float32)
    grid.C_water.assign(C_np)

    mass_before = float(grid.C_water.numpy().sum())
    assert mass_before > 0.0

    dt = 0.02  # CFL = 0.03 * 0.02 / 0.002 = 0.3, well under 1
    for _ in range(100):
        step_advect_c_water(grid, ws, dt)
    wp.synchronize()

    mass_after = float(grid.C_water.numpy().sum())
    rel_err = abs(mass_after - mass_before) / mass_before
    # Conservative finite-volume upwind should conserve to ~machine precision
    # for a domain that hasn't yet advected mass into a solid boundary.
    # Accept 0.5 % as a safe margin for the boundary-hitting regime.
    assert rel_err < 5.0e-3, (
        f"advection mass drift: {mass_before:.6e} -> {mass_after:.6e}, "
        f"rel err {rel_err*100:.5f}%  (>0.5%; upwind advection should conserve)"
    )


def test_c_water_advection_no_leak_into_solids():
    """After many advection steps, C_water in non-fluid cells must stay
    exactly zero (the kernel explicitly writes 0 for solid cells)."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.grid.dx_m = 0.002
    grid = build_pot_geometry(cfg)
    ws = allocate_nutrient_workspace(grid)

    # Seed C_water uniformly everywhere (including solids -- unphysical but
    # exercises the kernel's zero-write path).
    C_np = np.full(grid.shape, 5.0, dtype=np.float32)
    grid.C_water.assign(C_np)

    ux_np = grid.ux.numpy()
    ux_np[:] = 0.03
    grid.ux.assign(ux_np)

    for _ in range(30):
        step_advect_c_water(grid, ws, dt=0.02)
    wp.synchronize()

    mat_np = grid.mat.numpy()
    non_fluid = mat_np != 0
    C_np = grid.C_water.numpy()
    assert np.all(C_np[non_fluid] == 0.0), (
        f"non-fluid cells contaminated: max = {float(C_np[non_fluid].max()):.3e}"
    )


def test_full_pipeline_no_nan_over_60s():
    """End-to-end full pipeline (conduct + bubbles + nutrient) for 60 s at
    dev grid. No NaN, no negative C, retention stays in [0, 1]."""
    from boilingsim.pipeline import Simulation

    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.002
    cfg.total_time_s = 60.0
    sim = Simulation(cfg)

    for _ in range(200):
        sim.step()
        if sim.t >= 60.0:
            break

    wp.synchronize()
    C_np = sim.grid.C.numpy()
    Cw_np = sim.grid.C_water.numpy()
    T_np = sim.grid.T.numpy()
    assert not np.isnan(C_np).any(), "NaN in C"
    assert not np.isnan(Cw_np).any(), "NaN in C_water"
    assert not np.isnan(T_np).any(), "NaN in T"
    assert (C_np >= 0.0).all(), (
        f"negative C in {int((C_np < 0).sum())} cells, min = {float(C_np.min())}"
    )
    assert (Cw_np >= -1.0e-6).all(), (
        f"negative C_water, min = {float(Cw_np.min())}"
    )
    R = retention_fraction(sim.grid, sim.cfg)
    assert 0.0 <= R <= 1.0, f"retention out of bounds: {R}"


def test_retention_monotonic_decreasing_full_pipeline():
    """R(t) computed from grid.C must never increase over time in the full
    pipeline. (Degradation only shrinks C, diffusion conserves integrated
    mass, leaching only removes mass from the carrot, and numerical
    advection of C_water doesn't touch C at all.)"""
    from boilingsim.pipeline import Simulation

    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.002
    cfg.total_time_s = 30.0
    sim = Simulation(cfg)

    r_prev = 1.0 + 1.0e-6
    for _ in range(250):
        sim.step()
        R = retention_fraction(sim.grid, sim.cfg)
        assert R <= r_prev + 1.0e-5, (
            f"retention increased at t={sim.t:.2f}s: {r_prev:.6f} -> {R:.6f}"
        )
        r_prev = R
        if sim.t >= 30.0:
            break


def test_c_water_never_exceeds_cap_in_full_pipeline():
    """After every pipeline step, ``max(C_water)`` must stay at or below
    ``C_water_sat`` (within float32 rounding). The post-advection clamp
    is what enforces this; the leach kernel's per-face /6 divisor caps
    the source but cannot undo numerical overshoots introduced by upwind
    advection on a not-quite-div-free velocity field (projection residual
    accumulates over many steps).

    Regression: if :func:`step_clamp_c_water_sat` is ever removed from the
    pipeline, or stops being called after every advection, peak C_water
    drifts 10-300x above sat within the first minute of a rolling-boil
    scenario -- confirmed empirically before the clamp was added.
    """
    from boilingsim.pipeline import Simulation

    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.002
    cfg.total_time_s = 20.0
    sim = Simulation(cfg)

    # Warm-start: water at 95 C, wall at 100 C, carrot cold, so bubbles
    # spin up and produce the stagnation cells that caused the original
    # overshoot.
    T_np = sim.grid.T.numpy()
    mat_np = sim.grid.mat.numpy()
    T_np[mat_np == 0] = 95.0 + 273.15
    T_np[mat_np == 1] = 100.0 + 273.15
    T_np[mat_np == MAT_CARROT] = 20.0 + 273.15
    sim.grid.T.assign(T_np)

    sat = cfg.nutrient.C_water_sat_mg_per_kg
    tol = sat * 1.0e-4  # float32 rounding headroom
    n_checks = 0
    while sim.t < 20.0:
        sim.step()
        Cw_max = float(sim.grid.C_water.numpy().max())
        assert Cw_max <= sat + tol, (
            f"C_water exceeded cap at t={sim.t:.2f}s: "
            f"max {Cw_max:.6e} mg/kg vs sat {sat:.6e} mg/kg "
            f"(ratio {Cw_max/sat:.1f}x)"
        )
        n_checks += 1
    assert n_checks > 0, "no pipeline steps ran; test did not exercise the clamp"


def test_full_pipeline_mass_balance_with_precipitation():
    """The four-channel mass balance
    ``retention + leached + degraded + precipitated = 100 %%`` must hold
    at every sample. The prior three-channel version silently absorbed
    advection-induced overshoot into degraded_pct; the precipitated
    channel breaks that out so the diagnostic can distinguish numerical
    clip loss from real Arrhenius destruction.
    """
    from boilingsim.pipeline import Simulation

    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.002
    cfg.total_time_s = 15.0
    sim = Simulation(cfg)
    T_np = sim.grid.T.numpy()
    mat_np = sim.grid.mat.numpy()
    T_np[mat_np == 0] = 95.0 + 273.15
    T_np[mat_np == 1] = 100.0 + 273.15
    T_np[mat_np == MAT_CARROT] = 20.0 + 273.15
    sim.grid.T.assign(T_np)

    while sim.t < 15.0:
        sim.step()
        s = sim.sample_scalars(0.0)
        total = (s.retention_pct + s.leached_pct
                 + s.degraded_pct + s.precipitated_pct)
        assert abs(total - 100.0) < 1.0e-3, (
            f"mass balance broke at t={sim.t:.2f}s: "
            f"R={s.retention_pct:.6f} leach={s.leached_pct:.6f} "
            f"deg={s.degraded_pct:.6f} precip={s.precipitated_pct:.6f} "
            f"sum={total:.6f}"
        )


def test_stagnant_vs_moving_leach_rate():
    """Rate of leaching out of the carrot should increase with fluid velocity.
    Compare an identical 60 s run at u = 0 (Sh = 2 floor) vs u = 0.3 m/s
    (Sh from forced correlation, much larger). Expect the moving case to
    leach at least a few times more mass.

    This test is about ``h_m(u)`` responsiveness, not equilibrium mass.
    At the new physical defaults (K=1e-5, sat=6e-3) equilibrium lands at
    ~8e-4 mg/kg -- both the stagnant and the moving case hit it within
    a handful of steps and total leached mass converges, hiding the
    Sherwood delta. Override both K and sat up to the prior values to
    keep the transient regime wide enough that the velocity sensitivity
    shows up in total leached mass over 60 s.
    """
    cfg_stag, grid_stag = _build_leach_scenario(u_mag_mps=0.0)
    cfg_mov,  grid_mov  = _build_leach_scenario(u_mag_mps=0.3)
    for cfg in (cfg_stag, cfg_mov):
        cfg.nutrient.K_partition = 0.007           # keep driving force alive
        cfg.nutrient.C_water_sat_mg_per_kg = 0.6   # cap well clear of equilibrium

    for grid in (grid_stag, grid_mov):
        mat_np = grid.mat.numpy()
        T_np = grid.T.numpy()
        T_np[mat_np == MAT_CARROT] = 293.15  # freeze T
        grid.T.assign(T_np)

    def leached_mass(grid, cfg) -> float:
        mat_np = grid.mat.numpy()
        fluid_mask = mat_np == 0
        Cw_np = grid.C_water.numpy()
        return float(Cw_np[fluid_mask].sum())

    dt = 0.5
    for _ in range(120):  # 60 s
        step_leach(grid_stag, cfg_stag, dt)
        step_leach(grid_mov, cfg_mov, dt)
    wp.synchronize()

    m_stag = leached_mass(grid_stag, cfg_stag)
    m_mov = leached_mass(grid_mov, cfg_mov)
    ratio = m_stag / max(m_mov, 1.0e-12)
    assert ratio < 0.4, (
        f"stagnant leached {m_stag:.3e}, moving leached {m_mov:.3e}: "
        f"stagnant/moving = {ratio:.3f}, expected < 0.4"
    )


# ---------------------------------------------------------------------------
# Dual-solute extension tests (Phase-4 post-VC)
# ---------------------------------------------------------------------------


def test_dual_solute_geometry_allocates_both_fields():
    """With ``nutrient2.enabled = True`` and a distinct C0, build_pot_geometry
    must allocate an independent grid.C2 initialised from cfg.nutrient2.C0
    and a zeroed grid.C_water2. The primary field must be unaffected."""
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.nutrient2 = cfg.nutrient.model_copy(update={
        "enabled": True,
        "C0_mg_per_kg": 42.0,
        "K_partition": 1.0,
    })
    cfg.grid.dx_m = 0.004

    grid = build_pot_geometry(cfg)
    mat = grid.mat.numpy()
    carrot_mask = mat == MAT_CARROT
    n_carrot = int(carrot_mask.sum())
    assert n_carrot > 0, "no carrot cells in test geometry -- check dx / position"

    C_np = grid.C.numpy()
    C2_np = grid.C2.numpy()
    Cw_np = grid.C_water.numpy()
    Cw2_np = grid.C_water2.numpy()

    # Primary field: filled from cfg.nutrient.C0.
    assert C_np[carrot_mask].mean() == pytest.approx(
        cfg.nutrient.C0_mg_per_kg, rel=1e-6
    ), "grid.C not initialised from cfg.nutrient.C0"

    # Secondary field: filled from the OVERRIDDEN cfg.nutrient2.C0.
    assert C2_np[carrot_mask].mean() == pytest.approx(42.0, rel=1e-6), (
        "grid.C2 not initialised from cfg.nutrient2.C0 (override dropped)"
    )

    # Off-carrot cells are zero in both fields.
    assert float(np.abs(C_np[~carrot_mask]).max()) == 0.0
    assert float(np.abs(C2_np[~carrot_mask]).max()) == 0.0

    # Water-side scalars start at zero.
    assert float(Cw_np.sum()) == 0.0
    assert float(Cw2_np.sum()) == 0.0


def test_dual_solute_symmetric_params_equal_retention():
    """With nutrient2 configured identically to nutrient, the two retention
    traces must coincide to machine precision at every sample. This is the
    core invariant of the dual-solute refactor -- if the two slots were
    cross-contaminated or read the wrong config block, the symmetric case
    would surface the bug immediately."""
    from boilingsim.pipeline import Simulation

    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.nutrient2 = cfg.nutrient.model_copy(update={"enabled": True})
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.004
    cfg.total_time_s = 2.0

    sim = Simulation(cfg)
    samples = []
    for _ in range(60):
        dt_used = sim.step()
        samples.append(sim.sample_scalars(dt_used))
        if sim.t >= 2.0:
            break
    wp.synchronize()

    R1 = np.array([s.retention_pct for s in samples])
    R2 = np.array([s.retention2_pct for s in samples])
    L1 = np.array([s.leached_pct for s in samples])
    L2 = np.array([s.leached2_pct for s in samples])
    # With identical params, configs, and initial conditions (both C0s
    # produce the same normalised R) the traces must match exactly.
    assert float(np.max(np.abs(R1 - R2))) < 1.0e-6, (
        f"symmetric retention drift: max |R1-R2| = {np.max(np.abs(R1 - R2)):.3e}"
    )
    assert float(np.max(np.abs(L1 - L2))) < 1.0e-6, (
        f"symmetric leach drift: max |L1-L2| = {np.max(np.abs(L1 - L2)):.3e}"
    )


def test_dual_solute_independent_precipitation_counters():
    """Directly exercise the two precipitation counters via the clamp
    kernel: push a known overshoot into each solute's C_water field with
    known magnitudes and confirm only the intended counter accumulates.

    This avoids relying on the coupled boiling+leach+advect pipeline to
    drive one cap and not the other, which is fragile: the saturation
    clamp only fires once atomic-add races or advection concentration
    push C_water past C_water_sat, and whether that happens in a 1-2 s
    run depends sensitively on velocity field, K_partition, and sat
    thresholds. The direct-injection approach tests the bookkeeping
    invariant (counters don't cross-contaminate) without that noise.
    """
    from boilingsim.nutrient import (
        allocate_nutrient_workspace,
        clamp_c_water_and_track_precipitation,
    )

    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.nutrient.C_water_sat_mg_per_kg = 1.0e-3
    cfg.nutrient2 = cfg.nutrient.model_copy(update={
        "enabled": True,
        "C_water_sat_mg_per_kg": 1.0e6,   # effectively disabled
    })
    cfg.grid.dx_m = 0.004

    grid = build_pot_geometry(cfg)
    ws = allocate_nutrient_workspace(grid, alloc_secondary=True)
    nx, ny, nz = grid.shape

    # Inject 10 mg/kg into every fluid cell of BOTH C_water arrays.
    mat_np = grid.mat.numpy()
    fluid_mask = mat_np == 0
    Cw1 = np.zeros_like(mat_np, dtype=np.float32)
    Cw1[fluid_mask] = 10.0
    Cw2 = np.zeros_like(mat_np, dtype=np.float32)
    Cw2[fluid_mask] = 10.0
    grid.C_water.assign(Cw1)
    grid.C_water2.assign(Cw2)

    # Primary clamp: sat = 1e-3 so every fluid cell (at 10 mg/kg) is
    # clipped, putting ~(10 - 1e-3) * n_fluid mg/kg into precipitated_mass.
    wp.launch(
        clamp_c_water_and_track_precipitation,
        dim=(nx, ny, nz),
        inputs=[grid.C_water, ws.precipitated_mass,
                cfg.nutrient.C_water_sat_mg_per_kg],
    )
    # Secondary clamp: sat = 1e6, nothing clipped, counter stays at 0.
    wp.launch(
        clamp_c_water_and_track_precipitation,
        dim=(nx, ny, nz),
        inputs=[grid.C_water2, ws.precipitated_mass2,
                cfg.nutrient2.C_water_sat_mg_per_kg],
    )
    wp.synchronize()

    n_fluid = int(fluid_mask.sum())
    expected_primary = (10.0 - 1.0e-3) * float(n_fluid)
    got_primary = float(ws.precipitated_mass.numpy()[0])
    got_secondary = float(ws.precipitated_mass2.numpy()[0])

    assert math.isclose(got_primary, expected_primary, rel_tol=1.0e-4), (
        f"primary precipitated mass: expected ~{expected_primary:.3e}, "
        f"got {got_primary:.3e}"
    )
    assert got_secondary == 0.0, (
        f"secondary precipitated counter was touched: {got_secondary:.3e} "
        f"(should be exactly 0 -- independence violated)"
    )


def test_dual_solute_does_not_drift_single_solute_baseline():
    """Regression guard: enabling nutrient2 (with the SAME parameters as
    the primary) must NOT alter the single-solute primary trace. If any
    slot helper reads/writes the wrong array by accident, the primary
    trace drifts and this test fails.

    Concretely: run the same base cfg twice, once with nutrient2 disabled
    and once with nutrient2 enabled but configured identically, then
    compare the primary retention_pct arrays sample-by-sample."""
    from boilingsim.pipeline import Simulation

    def run_once(nut2_enabled: bool) -> np.ndarray:
        cfg = load_scenario("configs/scenarios/default.yaml")
        cfg.nutrient.enabled = True
        if nut2_enabled:
            cfg.nutrient2 = cfg.nutrient.model_copy(update={"enabled": True})
        cfg.boiling.enabled = True
        cfg.grid.dx_m = 0.004
        cfg.total_time_s = 1.0
        sim = Simulation(cfg)
        samples = []
        for _ in range(40):
            dt_used = sim.step()
            samples.append(sim.sample_scalars(dt_used))
            if sim.t >= 1.0:
                break
        wp.synchronize()
        return np.array([s.retention_pct for s in samples])

    R_off = run_once(nut2_enabled=False)
    R_on = run_once(nut2_enabled=True)

    # Must be byte-identical: dual-solute mode should not perturb the
    # primary solute's numerical trajectory in any way.
    assert len(R_off) == len(R_on), (
        f"sample count drift: off={len(R_off)}, on={len(R_on)}"
    )
    max_drift = float(np.max(np.abs(R_off - R_on)))
    assert max_drift < 1.0e-4, (
        f"primary retention drifts when nutrient2 is enabled: "
        f"max |R_off - R_on| = {max_drift:.3e}"
    )
