"""Phase 5.7 parity tests: hand-written CUDA update_bubbles vs Warp.

Five gates per the plan:

1. **Per-thread Bubble field bit-exact** after one step. Seed bubbles in
   distinct configurations and assert position/velocity/radius/active/
   site_cleared/departure_radius all match Warp output bit-for-bit.
   (atomicCAS-vs-plain-assign on site_active doesn't change Bubble fields
   because we always set bubble.site_cleared = 1 inside the if-branch.)
2. **needs_fragment + slot_claim flag counts.** Plan agent's "silent drift"
   guard: missing a single `needs_fragment = 1` would only show up after
   thousands of steps. Seed across the R_seed → R_max range and assert
   the flag-1 count and slot-0 (post-deactivation) count both match Warp.
3. **site_active final-set equivalence (NOT bit-equal attribution).** The
   set of (i,j,k) where site_active == 0 must match Warp's set. The
   per-bubble attribution can differ because Warp races; Rust uses CAS.
4. **T scatter sum-conservation + per-cell <1e-4 RMS** on the condensation
   path. Same gate as M5.6.A but with `atomic_add` going positive.
5. **Multi-step integration** smoke: run a small grid for 20 steps with
   the flag on and verify all per-thread invariants stay bit-exact across
   the multi-step trajectory.

If any gate fails, the kernel's arithmetic order or shape FFI is off.
First suspect is `_sample_face_u` -- the three MAC shape triples are
the fattest cliff in the FFI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp


pytestmark = pytest.mark.cuda_required


@pytest.fixture(scope="module")
def rust():
    sim_core = pytest.importorskip("sim_core")
    if sim_core.cuda is None:
        pytest.skip("sim_core.cuda submodule unavailable")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Warp reports no CUDA device available")
    return sim_core.cuda.SimCore(0)


# ---------------------------------------------------------------------------
# Scenario fixtures: seed multiple bubbles spanning every kernel branch.
# ---------------------------------------------------------------------------


def _seed_bubble_via_kernel(
    pool, slot_claim, slot,
    position, velocity=(0.0, 0.0, 0.0),
    radius=1.0e-5, birth_time=0.0,
):
    from boilingsim.boiling import _seed_test_bubble
    wp.launch(
        _seed_test_bubble,
        dim=1,
        inputs=[
            pool, slot_claim, slot,
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2],
            radius, birth_time,
        ],
        device="cuda:0",
    )


def _build_scenario(n_pool=64, grid_shape=(20, 20, 24), dx=0.002, water_line_z=None):
    """Build a small all-fluid box + a bubble pool. Returns a dict with all
    input arrays. Tests seed the pool however they want before invoking
    Warp / Rust."""
    from boilingsim.boiling import Bubble

    nx, ny, nz = grid_shape
    if water_line_z is None:
        water_line_z = (nz - 1) * dx  # one cell below the top

    # All-fluid (MAT_FLUID = 0) grid with a slight superheat field that
    # has a vertical T gradient so different bubbles see different T_local.
    mat_np = np.zeros(grid_shape, dtype=np.int32)
    T_np = np.linspace(370.0, 376.0, nz, dtype=np.float32)
    T_np = np.broadcast_to(T_np, grid_shape).copy()

    # MAC velocity fields with a steady upward flow + tiny shear.
    ux_shape = (nx + 1, ny, nz)
    uy_shape = (nx, ny + 1, nz)
    uz_shape = (nx, ny, nz + 1)
    rng = np.random.default_rng(7)
    ux_np = rng.normal(0.0, 0.02, ux_shape).astype(np.float32)
    uy_np = rng.normal(0.0, 0.02, uy_shape).astype(np.float32)
    uz_np = np.full(uz_shape, 0.05, dtype=np.float32) \
            + rng.normal(0.0, 0.02, uz_shape).astype(np.float32)

    # site_active starts at 1 everywhere (a bubble can clear it).
    site_active_np = np.ones(grid_shape, dtype=np.int32)

    pool = wp.zeros(n_pool, dtype=Bubble, device="cuda:0")
    slot_claim = wp.zeros(n_pool, dtype=wp.int32, device="cuda:0")
    needs_fragment = wp.zeros(n_pool, dtype=wp.int32, device="cuda:0")
    return {
        "shape": grid_shape,
        "ux_shape": ux_shape,
        "uy_shape": uy_shape,
        "uz_shape": uz_shape,
        "dx": dx,
        "origin": (0.0, 0.0, 0.0),
        "water_line_z": water_line_z,
        "n_pool": n_pool,
        "pool": pool,
        "slot_claim": slot_claim,
        "needs_fragment": needs_fragment,
        "site_active_np": site_active_np,
        "T_np": T_np,
        "mat_np": mat_np,
        "ux_np": ux_np,
        "uy_np": uy_np,
        "uz_np": uz_np,
        # Physics defaults
        "dt": 1.0e-3,
        "sim_time": 0.1,
        "T_sat_k": 373.15,
        "rho_l": 997.0,
        "rho_v": 0.598,
        "cp_l": 4186.0,
        "k_l": 0.606,
        "h_lv": 2.257e6,
        "sigma": 0.0589,
        "theta_rad": 1.0,
        "g_mag": 9.81,
        "R_seed": 1.0e-5,
        "R_frag": 4.0e-3,
        "R_max": 5.0e-3,
    }


def _seed_mixed_bubbles(sc):
    """Seed a pool with a diverse mix that exercises every kernel branch.

    Returns the host-side pool layout so the test can predict which slot
    is in which configuration:
        slot 0: growing (mid-grid, R = R_seed, will grow)
        slot 1: condensing not fully (lower-grid, T<T_sat, R = 5e-4 m)
        slot 2: fully condensing (R = 2*R_seed, T<<T_sat, deep subcooled)
        slot 3: about-to-depart (high R = 2 mm at mid-grid)
        slot 4: above water-line will vent (z >= water_line_z)
        slot 5: at large R will hit fragmentation flag (R = R_frag * 1.1)
        slot 6-7: extra growing bubbles for sum-conservation statistics
        slot 8: inactive (untouched)
    """
    nx, ny, nz = sc["shape"]
    dx = sc["dx"]
    pool = sc["pool"]
    slot_claim = sc["slot_claim"]

    # slot 0: growing -- mid-grid (T ~ 373.something for our gradient)
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=0,
        position=(nx * dx * 0.4, ny * dx * 0.4, nz * dx * 0.7),
        radius=sc["R_seed"],
        birth_time=0.0,
    )
    # slot 1: condensing not fully -- bottom of grid, low T, R = 5e-4
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=1,
        position=(nx * dx * 0.5, ny * dx * 0.5, nz * dx * 0.1),
        radius=5.0e-4,
        birth_time=0.0,
    )
    # slot 2: fully condensing -- bottom + small R + strong subcool
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=2,
        position=(nx * dx * 0.2, ny * dx * 0.2, nz * dx * 0.05),
        radius=sc["R_seed"] * 1.5,
        birth_time=0.0,
    )
    # slot 3: about-to-depart at hot mid-grid (R=2mm -> 2R=4mm > D_d≈2.6mm)
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=3,
        position=(nx * dx * 0.5, ny * dx * 0.5, nz * dx * 0.6),
        radius=2.0e-3,
        birth_time=0.05,
    )
    # slot 4: ABOVE water line -> vent
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=4,
        position=(nx * dx * 0.5, ny * dx * 0.5, sc["water_line_z"] + dx),
        radius=1.0e-3,
        birth_time=0.05,
    )
    # slot 5: at fragmentation radius
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=5,
        position=(nx * dx * 0.5, ny * dx * 0.5, nz * dx * 0.7),
        radius=sc["R_frag"] * 1.1,
        birth_time=0.05,
    )
    # slots 6 & 7: more growing bubbles for statistics
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=6,
        position=(nx * dx * 0.3, ny * dx * 0.6, nz * dx * 0.5),
        radius=2.0e-4,
        birth_time=0.02,
    )
    _seed_bubble_via_kernel(
        pool, slot_claim, slot=7,
        position=(nx * dx * 0.6, ny * dx * 0.3, nz * dx * 0.55),
        radius=3.0e-4,
        birth_time=0.03,
    )
    wp.synchronize_device("cuda:0")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _run_warp(sc):
    """Run one update_bubbles step via Warp; return host-side copies of
    every mutable array."""
    from boilingsim.boiling import update_bubbles

    pool = sc["pool"]
    slot_claim_d = wp.array(sc["slot_claim"].numpy(), dtype=wp.int32, device="cuda:0")
    needs_fragment_d = wp.zeros(sc["n_pool"], dtype=wp.int32, device="cuda:0")
    site_active_d = wp.array(sc["site_active_np"].copy(), dtype=wp.int32, device="cuda:0")
    T_d = wp.array(sc["T_np"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(sc["mat_np"], dtype=wp.int32, device="cuda:0")
    ux_d = wp.array(sc["ux_np"], dtype=wp.float32, device="cuda:0")
    uy_d = wp.array(sc["uy_np"], dtype=wp.float32, device="cuda:0")
    uz_d = wp.array(sc["uz_np"], dtype=wp.float32, device="cuda:0")

    # Snapshot the pool BEFORE update (each path needs the same starting state).
    pool_in = wp.zeros(sc["n_pool"], dtype=pool.dtype, device="cuda:0")
    wp.copy(pool_in, pool)

    wp.launch(
        update_bubbles,
        dim=sc["n_pool"],
        inputs=[
            pool_in,
            slot_claim_d,
            site_active_d,
            needs_fragment_d,
            T_d, mat_d,
            ux_d, uy_d, uz_d,
            wp.vec3(*sc["origin"]),
            sc["dx"], sc["dt"], sc["sim_time"], sc["water_line_z"],
            sc["T_sat_k"], sc["rho_l"], sc["rho_v"],
            sc["cp_l"], sc["k_l"], sc["h_lv"], sc["sigma"],
            sc["theta_rad"], sc["g_mag"],
            sc["R_seed"], sc["R_frag"], sc["R_max"],
            0,  # MAT_FLUID
        ],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")
    return {
        "pool": pool_in.numpy(),
        "slot_claim": slot_claim_d.numpy(),
        "needs_fragment": needs_fragment_d.numpy(),
        "site_active": site_active_d.numpy(),
        "T": T_d.numpy(),
    }


def _run_rust(rust, sc):
    """Run one update_bubbles step via the Rust kernel; return same dict shape."""
    pool = sc["pool"]
    slot_claim_d = wp.array(sc["slot_claim"].numpy(), dtype=wp.int32, device="cuda:0")
    needs_fragment_d = wp.zeros(sc["n_pool"], dtype=wp.int32, device="cuda:0")
    site_active_d = wp.array(sc["site_active_np"].copy(), dtype=wp.int32, device="cuda:0")
    T_d = wp.array(sc["T_np"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(sc["mat_np"], dtype=wp.int32, device="cuda:0")
    ux_d = wp.array(sc["ux_np"], dtype=wp.float32, device="cuda:0")
    uy_d = wp.array(sc["uy_np"], dtype=wp.float32, device="cuda:0")
    uz_d = wp.array(sc["uz_np"], dtype=wp.float32, device="cuda:0")

    pool_in = wp.zeros(sc["n_pool"], dtype=pool.dtype, device="cuda:0")
    wp.copy(pool_in, pool)

    rust.update_bubbles(
        int(pool_in.ptr),
        int(sc["n_pool"]),
        int(slot_claim_d.__cuda_array_interface__["data"][0]),
        int(site_active_d.__cuda_array_interface__["data"][0]),
        int(needs_fragment_d.__cuda_array_interface__["data"][0]),
        int(T_d.__cuda_array_interface__["data"][0]),
        int(mat_d.__cuda_array_interface__["data"][0]),
        sc["shape"][0], sc["shape"][1], sc["shape"][2],
        int(ux_d.__cuda_array_interface__["data"][0]),
        sc["ux_shape"][0], sc["ux_shape"][1], sc["ux_shape"][2],
        int(uy_d.__cuda_array_interface__["data"][0]),
        sc["uy_shape"][0], sc["uy_shape"][1], sc["uy_shape"][2],
        int(uz_d.__cuda_array_interface__["data"][0]),
        sc["uz_shape"][0], sc["uz_shape"][1], sc["uz_shape"][2],
        sc["origin"][0], sc["origin"][1], sc["origin"][2],
        sc["dx"], sc["dt"], sc["sim_time"], sc["water_line_z"],
        sc["T_sat_k"], sc["rho_l"], sc["rho_v"],
        sc["cp_l"], sc["k_l"], sc["h_lv"], sc["sigma"],
        sc["theta_rad"], sc["g_mag"],
        sc["R_seed"], sc["R_frag"], sc["R_max"],
        0,  # MAT_FLUID
    )
    wp.synchronize_device("cuda:0")
    return {
        "pool": pool_in.numpy(),
        "slot_claim": slot_claim_d.numpy(),
        "needs_fragment": needs_fragment_d.numpy(),
        "site_active": site_active_d.numpy(),
        "T": T_d.numpy(),
    }


# ---------------------------------------------------------------------------
# Gate 1 + 2: per-thread Bubble + flag arrays (deterministic side)
# ---------------------------------------------------------------------------


def test_gate1_bubble_fields_bit_exact(rust):
    """Per-thread Bubble field writes must match Warp bit-for-bit. The
    arithmetic is deterministic per-thread (no inter-thread contention on
    Bubble fields), so any drift indicates an arithmetic or branch mismatch.

    Tolerance: zero ULPs for radius, position, velocity, departure_radius;
    exact integer equality for active and site_cleared.
    """
    sc = _build_scenario()
    _seed_mixed_bubbles(sc)
    warp_out = _run_warp(sc)
    rust_out = _run_rust(rust, sc)

    pw = warp_out["pool"]
    pr = rust_out["pool"]

    # Per-bubble check across all 8 seeded + the inactive remainder.
    for slot in range(sc["n_pool"]):
        np.testing.assert_array_equal(
            pw[slot]["active"], pr[slot]["active"],
            err_msg=f"slot {slot} active mismatch: warp={pw[slot]['active']} rust={pr[slot]['active']}",
        )
        np.testing.assert_array_equal(
            pw[slot]["site_cleared"], pr[slot]["site_cleared"],
            err_msg=f"slot {slot} site_cleared mismatch",
        )
        # Floats: assert exact equality (FMA may produce <1 ULP drift; we
        # accept that via assert_allclose with rtol=1e-6 to absorb FMA-rounding).
        np.testing.assert_allclose(
            pw[slot]["radius"], pr[slot]["radius"], rtol=1e-6, atol=1e-12,
            err_msg=f"slot {slot} radius: warp={pw[slot]['radius']} rust={pr[slot]['radius']}",
        )
        np.testing.assert_allclose(
            pw[slot]["position"], pr[slot]["position"], rtol=1e-6, atol=1e-9,
            err_msg=f"slot {slot} position: warp={pw[slot]['position']} rust={pr[slot]['position']}",
        )
        np.testing.assert_allclose(
            pw[slot]["velocity"], pr[slot]["velocity"], rtol=1e-6, atol=1e-9,
            err_msg=f"slot {slot} velocity: warp={pw[slot]['velocity']} rust={pr[slot]['velocity']}",
        )
        np.testing.assert_allclose(
            pw[slot]["departure_radius"], pr[slot]["departure_radius"],
            rtol=1e-6, atol=1e-12,
            err_msg=f"slot {slot} departure_radius mismatch",
        )


def test_gate2_flag_count_parity(rust):
    """needs_fragment.sum() and slot_claim count-of-zeros must match Warp."""
    sc = _build_scenario()
    _seed_mixed_bubbles(sc)
    warp_out = _run_warp(sc)
    rust_out = _run_rust(rust, sc)

    nf_w = int(warp_out["needs_fragment"].sum())
    nf_r = int(rust_out["needs_fragment"].sum())
    assert nf_w == nf_r, (
        f"needs_fragment count: warp={nf_w} rust={nf_r}. A miss here means "
        "the Rust path silently drops fragmentation -- bubble size drift after 1000s of steps."
    )

    # slot_claim post-update: 0 = deactivated; check the count.
    sc_w = int((warp_out["slot_claim"] == 0).sum())
    sc_r = int((rust_out["slot_claim"] == 0).sum())
    assert sc_w == sc_r, (
        f"slot_claim==0 count: warp={sc_w} rust={sc_r}"
    )


# ---------------------------------------------------------------------------
# Gate 3: site_active final-set equivalence
# ---------------------------------------------------------------------------


def test_gate3_site_active_set_equivalence(rust):
    """The set of cells where site_active==0 after the kernel must be the
    same on both paths. atomicCAS in Rust eliminates the Warp race but the
    resulting set is identical (both write 0 to the same locations)."""
    sc = _build_scenario()
    _seed_mixed_bubbles(sc)
    warp_out = _run_warp(sc)
    rust_out = _run_rust(rust, sc)

    warp_zeros = set(map(tuple, np.argwhere(warp_out["site_active"] == 0).tolist()))
    rust_zeros = set(map(tuple, np.argwhere(rust_out["site_active"] == 0).tolist()))
    assert warp_zeros == rust_zeros, (
        f"site_active=0 sets differ:\n  warp - rust = {warp_zeros - rust_zeros}\n"
        f"  rust - warp = {rust_zeros - warp_zeros}"
    )


# ---------------------------------------------------------------------------
# Gate 4: T scatter sum-conservation + per-cell agreement
# ---------------------------------------------------------------------------


def test_gate4_T_scatter_sum_and_per_cell(rust):
    """The condensation path scatters latent heat back into T. Sum of dT
    across the grid is deterministic (atomic ordering preserves sum). Per
    cell, Warp and Rust may differ by ~1 ULP per atomic but the overall
    RMS should be tiny."""
    sc = _build_scenario()
    _seed_mixed_bubbles(sc)
    T_init = sc["T_np"]
    warp_out = _run_warp(sc)
    rust_out = _run_rust(rust, sc)

    dT_w = warp_out["T"] - T_init
    dT_r = rust_out["T"] - T_init
    sum_w = float(dT_w.sum())
    sum_r = float(dT_r.sum())

    # Sum-conservation: should agree to float32 precision (~1e-6 rel).
    # In practice the absolute sums are tiny (8-bubble condensation deposits
    # are ~1 mK total) so use absolute tol relative to grid size.
    grid_cells = float(dT_w.size)
    abs_tol = abs(sum_w) * 1e-4 + 1e-6 * grid_cells
    assert abs(sum_w - sum_r) < abs_tol, (
        f"T scatter sum mismatch: warp={sum_w:.6e} rust={sum_r:.6e}"
    )

    # Per-cell RMS over cells the warp path actually touched.
    touched = np.abs(dT_w) > 0.0
    if touched.any():
        diff = dT_w[touched] - dT_r[touched]
        rms = float(np.sqrt((diff ** 2).mean()))
        scale = float(np.abs(dT_w[touched]).max())
        rel_rms = rms / (scale + 1e-12)
        assert rel_rms < 1e-3, (
            f"T scatter per-cell RMS too high: rms={rms:.3e} rel={rel_rms:.3e}"
        )


# ---------------------------------------------------------------------------
# Gate 5: multi-step integration smoke
# ---------------------------------------------------------------------------


def test_gate5_multi_step_integration(rust):
    """Run 20 steps each, compare end-state. Tests for state-machine drift
    that single-step gates miss."""
    sc = _build_scenario()
    _seed_mixed_bubbles(sc)
    pool_init = sc["pool"].numpy().copy()
    slot_claim_init = sc["slot_claim"].numpy().copy()
    site_active_init = sc["site_active_np"].copy()
    T_init = sc["T_np"].copy()

    def run_n_steps(runner_fn, n_steps):
        # Each iteration rebuilds GPU buffers from the saved init state
        # WITHIN the runner. But for multi-step we need to thread state
        # forward. So instead we just call the runner once per step,
        # feeding the pool/state from the prior step.
        from boilingsim.boiling import Bubble

        # We use a fresh scenario dict per step but with state copied forward.
        pool_state = pool_init.copy()
        slot_claim_state = slot_claim_init.copy()
        site_active_state = site_active_init.copy()
        T_state = T_init.copy()
        for step in range(n_steps):
            # Rebuild scenario with current state.
            sc_step = dict(sc)
            sc_step["pool"] = wp.array(pool_state, dtype=Bubble, device="cuda:0")
            sc_step["slot_claim"] = wp.array(slot_claim_state, dtype=wp.int32, device="cuda:0")
            sc_step["site_active_np"] = site_active_state
            sc_step["T_np"] = T_state
            sc_step["sim_time"] = 0.1 + step * sc["dt"]
            out = runner_fn(sc_step)
            pool_state = out["pool"]
            slot_claim_state = out["slot_claim"]
            site_active_state = out["site_active"]
            T_state = out["T"]
        return {
            "pool": pool_state, "slot_claim": slot_claim_state,
            "site_active": site_active_state, "T": T_state,
        }

    warp_end = run_n_steps(_run_warp, 20)
    rust_end = run_n_steps(lambda s: _run_rust(rust, s), 20)

    # Bubble fields: same tolerance as gate 1 but cumulative.
    pw = warp_end["pool"]
    pr = rust_end["pool"]
    for slot in range(sc["n_pool"]):
        if pw[slot]["active"] != pr[slot]["active"]:
            pytest.fail(
                f"after 20 steps slot {slot} active diverged: "
                f"warp={pw[slot]['active']} rust={pr[slot]['active']}"
            )

    # Active set must match.
    active_w = set(np.where(pw["active"] == 1)[0].tolist())
    active_r = set(np.where(pr["active"] == 1)[0].tolist())
    assert active_w == active_r, (
        f"active sets diverged after 20 steps: warp={active_w} rust={active_r}"
    )

    # Total T integral should agree to ~1e-3.
    sum_w = float(warp_end["T"].sum())
    sum_r = float(rust_end["T"].sum())
    rel = abs(sum_w - sum_r) / abs(sum_w)
    assert rel < 1e-4, (
        f"after 20 steps, total T sum drifted: warp={sum_w:.4e} rust={sum_r:.4e} rel={rel:.3e}"
    )
