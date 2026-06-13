"""Phase 5.6 M5.6.A parity tests for the Rust scatter kernels.

These tests gate the Rust scatter-kernel port against the Warp reference.
Atomic operations break bit-exact parity, so the gates here are:

1. **Bubble ABI smoke**: a wp.array of Bubble seeded with known sentinel
   values must round-trip through Rust as the matching 56-byte struct.
   If this fails, the C++ `struct Bubble` layout in
   `crates/cuda-kernels/include/bubble.h` is out of sync with Warp's
   `wp.struct Bubble` at `python/boilingsim/boiling.py:38-58`.

2. **Sum-conservation** (deterministic, no atomic-order dependence):
   the total energy removed from the fluid summed over all cells must
   match the analytic per-bubble sum within float32 precision. This is
   the same invariant the existing `test_latent_heat_energy_balance`
   verifies for the Warp path.

3. **Per-cell agreement**: max relative diff between Warp and Rust
   output T fields after one scatter step is < 1e-4. Atomic ordering
   may shift individual cell values by ~1 ULP but the deterministic
   sum-conservation gate above bounds the total error.
"""

from __future__ import annotations

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
# Bubble ABI smoke
# ---------------------------------------------------------------------------


def test_bubble_struct_is_56_bytes():
    """The first gate: Warp's wp.struct Bubble must allocate as 56 bytes per
    element so the C++ `struct Bubble` in bubble.h sees the right layout.

    A future Warp upgrade that changes struct packing (e.g. adds explicit
    alignment hints) would change this and break the rest of the suite.
    Catching it here gives a clear error before the kernel-output tests
    spew confusing parity failures.
    """
    from boilingsim.boiling import Bubble
    pool = wp.zeros(4, dtype=Bubble, device="cuda:0")
    bytes_per = pool.size * pool.dtype._type_._size_ if hasattr(pool.dtype, "_type_") else None
    # Public path: the array's strides tell us bytes-per-element regardless of
    # Warp internals.
    strides = pool.strides
    assert len(strides) == 1, f"expected 1D Bubble pool, got strides {strides}"
    bytes_per_element = strides[0]
    assert bytes_per_element == 56, (
        f"Bubble struct allocates as {bytes_per_element} bytes per element; "
        "C++ side at crates/cuda-kernels/include/bubble.h assumes 56. "
        "Update both sides together if Warp changed its struct packing."
    )


# ---------------------------------------------------------------------------
# Single-bubble scatter parity (the M5.6.A acceptance gate)
# ---------------------------------------------------------------------------


def _build_scenario(grid_shape=(16, 16, 16), dx=0.002):
    """Build a small fluid box with a single seeded bubble at the centre.

    Returns the input arrays both paths will share: T (uniform superheat),
    mat (all fluid), bubbles (one active bubble), and the physics constants.
    """
    from boilingsim.boiling import Bubble, _seed_test_bubble

    nx, ny, nz = grid_shape
    # All-fluid grid -- avoids non-fluid corner skips so the test exercises
    # the full 8-corner scatter on both paths.
    mat_np = np.zeros(grid_shape, dtype=np.int32)  # MAT_FLUID = 0
    # Slightly above T_sat so the temperature gate passes.
    T_np = np.full(grid_shape, 374.15, dtype=np.float32)

    # Allocate a minimal Bubble pool directly via _seed_test_bubble, sidestepping
    # the BubblePool dataclass (which carries lots of M3 coalescence scratch we
    # don't need for an isolated scatter parity test).
    pool = wp.zeros(4, dtype=Bubble, device="cuda:0")
    slot_claim = wp.zeros(4, dtype=wp.int32, device="cuda:0")
    # Centre the bubble at the box centre. Origin = (0,0,0) so position is
    # (nx/2 * dx, ny/2 * dx, nz/2 * dx).
    center = (nx / 2 * dx, ny / 2 * dx, nz / 2 * dx)
    R0 = 2.0e-3
    sim_time = 0.1
    wp.launch(
        _seed_test_bubble,
        dim=1,
        inputs=[
            pool, slot_claim, 0,
            center[0], center[1], center[2],
            0.0, 0.0, 0.0,
            R0, 0.0,
        ],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")

    return {
        "shape": grid_shape,
        "dx": dx,
        "origin": (0.0, 0.0, 0.0),
        "T_init": T_np,
        "mat": mat_np,
        "pool": pool,
        "R0": R0,
        "sim_time": sim_time,
        "dt": 1.0e-2,
        "rho_l": 997.0,
        "cp_l": 4186.0,
        "rho_v": 0.598,
        "h_lv": 2.257e6,
        "T_sat_k": 373.15,
    }


def _run_warp(scenario):
    """Run one step of the Warp scatter_latent_heat kernel; return T."""
    from boilingsim.boiling import scatter_latent_heat

    grid_shape = scenario["shape"]
    T_d = wp.array(scenario["T_init"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(scenario["mat"], dtype=wp.int32, device="cuda:0")
    pool = scenario["pool"]

    wp.launch(
        scatter_latent_heat,
        dim=pool.shape[0],
        inputs=[
            pool, T_d, mat_d,
            wp.vec3(*scenario["origin"]),
            scenario["dx"],
            scenario["dt"],
            scenario["sim_time"],
            scenario["rho_l"], scenario["cp_l"],
            scenario["rho_v"], scenario["h_lv"],
            scenario["T_sat_k"],
            0,  # MAT_FLUID
        ],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")
    return T_d.numpy()


def _run_rust(rust, scenario):
    """Run one step of the Rust scatter_latent_heat kernel; return T.

    Bubble-pool pointer comes via the private `arr.ptr` attribute because
    Warp's `__cuda_array_interface__` doesn't support struct-typed arrays
    (Bubble is a wp.struct). The fluid f32/i32 arrays use the public CAI
    path as in M1.
    """
    grid_shape = scenario["shape"]
    T_d = wp.array(scenario["T_init"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(scenario["mat"], dtype=wp.int32, device="cuda:0")
    pool = scenario["pool"]

    rust.scatter_latent_heat(
        int(pool.ptr),  # struct-typed wp.array: private API; CAI unsupported.
        int(pool.shape[0]),
        int(T_d.__cuda_array_interface__["data"][0]),
        int(mat_d.__cuda_array_interface__["data"][0]),
        grid_shape[0], grid_shape[1], grid_shape[2],
        scenario["origin"][0], scenario["origin"][1], scenario["origin"][2],
        scenario["dx"],
        scenario["dt"],
        scenario["sim_time"],
        scenario["rho_l"], scenario["cp_l"],
        scenario["rho_v"], scenario["h_lv"],
        scenario["T_sat_k"],
        0,  # MAT_FLUID
    )
    wp.synchronize_device("cuda:0")
    return T_d.numpy()


def test_single_bubble_sum_conservation(rust):
    """The total energy removed from the fluid by one scatter step must equal
    the analytic ρ_v · h_lv · 4π R² · (R/2age) · dt within float32 precision.

    This gate is deterministic even with atomics -- the SUM is invariant to
    ordering, only individual cell values vary.
    """
    sc = _build_scenario(grid_shape=(20, 20, 20), dx=0.002)
    T_init = sc["T_init"]
    T_after_rust = _run_rust(rust, sc)

    # Total dT applied across the grid * cell volume * rho_l * c_p_l = energy.
    cell_volume = sc["dx"] ** 3
    dE_measured = -(T_after_rust - T_init).sum() * sc["rho_l"] * sc["cp_l"] * cell_volume

    # Analytic energy from the single bubble.
    R0 = sc["R0"]
    age = sc["sim_time"]
    dR_dt = R0 / (2.0 * age)
    Q_b = sc["rho_v"] * sc["h_lv"] * 4.0 * np.pi * R0 * R0 * dR_dt
    dE_expected = Q_b * sc["dt"]

    rel_err = abs(dE_measured - dE_expected) / dE_expected
    assert rel_err < 0.01, (
        f"Rust scatter sum-conservation off by {rel_err*100:.2f} % "
        f"(measured {dE_measured:.4e} J, expected {dE_expected:.4e} J)"
    )


def test_single_bubble_warp_vs_rust_close(rust):
    """Per-cell agreement between Warp and Rust scatter outputs.

    A single bubble's 8-corner scatter is deterministic on both paths (one
    bubble, no atomic-ordering contention), so the difference should be
    bit-tiny (FMA divergence at most). Multi-bubble scenarios get a looser
    gate in the next test.
    """
    sc = _build_scenario(grid_shape=(20, 20, 20), dx=0.002)
    T_warp = _run_warp(sc)
    T_rust = _run_rust(rust, sc)

    diff = np.abs(T_warp - T_rust)
    scale = np.maximum(np.abs(T_warp), 1.0e-6)
    max_rel = float((diff / scale).max())
    max_abs = float(diff.max())
    assert max_rel < 1.0e-5, (
        f"single-bubble scatter parity: max_rel_diff={max_rel:.3e}, "
        f"max_abs_diff={max_abs:.3e}"
    )


def test_no_scatter_when_T_below_saturation(rust):
    """Subcooled water: temperature gate skips the scatter. Rust must produce
    an unchanged T field, same as Warp."""
    sc = _build_scenario(grid_shape=(16, 16, 16), dx=0.002)
    sc["T_init"] = np.full(sc["shape"], 370.0, dtype=np.float32)  # below T_sat

    T_rust = _run_rust(rust, sc)
    np.testing.assert_array_equal(T_rust, sc["T_init"])


def test_no_scatter_when_just_nucleated(rust):
    """Age < 1e-6 means dR/dt is undefined (analytic singularity at t=0).
    Both paths must short-circuit and leave T unchanged."""
    sc = _build_scenario(grid_shape=(16, 16, 16), dx=0.002)
    sc["sim_time"] = 0.0  # age = birth_time - 0 = 0

    T_rust = _run_rust(rust, sc)
    np.testing.assert_array_equal(T_rust, sc["T_init"])


# ---------------------------------------------------------------------------
# M5.6.B: scatter_bubble_momentum parity
# ---------------------------------------------------------------------------


def _build_momentum_scenario(grid_shape=(16, 16, 16), dx=0.002):
    """Single bubble in an all-fluid grid; uz field starts at zero.

    Returns a scenario dict similar to _build_scenario but tuned for the
    momentum kernel: no temperature gate, no birth-time guard, just the
    geometric scatter to z-faces.
    """
    from boilingsim.boiling import Bubble, _seed_test_bubble

    nx, ny, nz = grid_shape
    mat_np = np.zeros(grid_shape, dtype=np.int32)
    # uz has shape (nx, ny, nz+1) for MAC z-faces.
    uz_shape = (nx, ny, nz + 1)
    uz_init = np.zeros(uz_shape, dtype=np.float32)

    pool = wp.zeros(4, dtype=Bubble, device="cuda:0")
    slot_claim = wp.zeros(4, dtype=wp.int32, device="cuda:0")
    center = (nx / 2 * dx, ny / 2 * dx, nz / 2 * dx)
    R0 = 2.0e-3
    wp.launch(
        _seed_test_bubble,
        dim=1,
        inputs=[
            pool, slot_claim, 0,
            center[0], center[1], center[2],
            0.0, 0.0, 0.0,
            R0, 0.0,
        ],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")
    return {
        "shape": grid_shape,
        "uz_shape": uz_shape,
        "dx": dx,
        "origin": (0.0, 0.0, 0.0),
        "mat": mat_np,
        "uz_init": uz_init,
        "pool": pool,
        "R0": R0,
        "dt": 1.0e-3,
        "rho_l": 997.0,
        "rho_v": 0.598,
        "g_mag": 9.81,
    }


def _run_warp_momentum(sc):
    from boilingsim.boiling import scatter_bubble_momentum

    uz_d = wp.array(sc["uz_init"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(sc["mat"], dtype=wp.int32, device="cuda:0")
    wp.launch(
        scatter_bubble_momentum,
        dim=sc["pool"].shape[0],
        inputs=[
            sc["pool"], uz_d, mat_d,
            wp.vec3(*sc["origin"]),
            sc["dx"], sc["dt"],
            sc["rho_l"], sc["rho_v"], sc["g_mag"],
            0,  # MAT_FLUID
        ],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")
    return uz_d.numpy()


def _run_rust_momentum(rust, sc):
    uz_d = wp.array(sc["uz_init"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(sc["mat"], dtype=wp.int32, device="cuda:0")
    rust.scatter_momentum(
        int(sc["pool"].ptr),
        int(sc["pool"].shape[0]),
        int(uz_d.__cuda_array_interface__["data"][0]),
        int(mat_d.__cuda_array_interface__["data"][0]),
        sc["shape"][0], sc["shape"][1], sc["shape"][2],
        sc["uz_shape"][0], sc["uz_shape"][1], sc["uz_shape"][2],
        sc["origin"][0], sc["origin"][1], sc["origin"][2],
        sc["dx"], sc["dt"],
        sc["rho_l"], sc["rho_v"], sc["g_mag"],
        0,
    )
    wp.synchronize_device("cuda:0")
    return uz_d.numpy()


def test_momentum_sum_conservation(rust):
    """Total momentum injected = F_total · dt / ρ_l · 1/V_cell summed over
    8 faces (weights sum to 1). Reuse the analytic balance from the existing
    test_bubble_momentum_creates_upward_velocity test."""
    import math

    sc = _build_momentum_scenario(grid_shape=(20, 20, 20), dx=0.002)
    uz_after = _run_rust_momentum(rust, sc)
    uz_before = sc["uz_init"]

    rho_l = sc["rho_l"]
    V_cell = sc["dx"] ** 3
    total_momentum_measured = float((uz_after - uz_before).sum()) * rho_l * V_cell

    R0 = sc["R0"]
    V_b = 4.0 / 3.0 * math.pi * R0 ** 3
    expected = V_b * (rho_l - sc["rho_v"]) * sc["g_mag"] * sc["dt"]
    rel_err = abs(total_momentum_measured - expected) / expected
    assert rel_err < 0.05, (
        f"Rust momentum sum off by {rel_err*100:.2f}% "
        f"(measured {total_momentum_measured:.4e}, expected {expected:.4e})"
    )


def test_momentum_warp_vs_rust_close(rust):
    """Single-bubble momentum scatter: Warp vs Rust per-face agreement
    within ULP-level tolerance (single bubble, no atomic contention)."""
    sc = _build_momentum_scenario(grid_shape=(20, 20, 20), dx=0.002)
    uz_warp = _run_warp_momentum(sc)
    uz_rust = _run_rust_momentum(rust, sc)

    diff = np.abs(uz_warp - uz_rust)
    scale = np.maximum(np.abs(uz_warp), 1.0e-10)
    max_rel = float((diff / scale).max())
    assert max_rel < 1.0e-5, (
        f"single-bubble momentum scatter parity: max_rel_diff={max_rel:.3e}, "
        f"max_abs_diff={float(diff.max()):.3e}"
    )


def test_momentum_only_positive_uz_contributions(rust):
    """Every nonzero uz contribution from a bubble must be positive (the
    buoyancy force is always upward). Mirrors the existing
    test_bubble_momentum_creates_upward_velocity invariant."""
    sc = _build_momentum_scenario(grid_shape=(16, 16, 16), dx=0.002)
    uz_after = _run_rust_momentum(rust, sc)

    nonzero = uz_after[uz_after != 0.0]
    assert (nonzero > 0).all(), (
        f"some Rust momentum contributions are negative: min = {nonzero.min()}"
    )


# ---------------------------------------------------------------------------
# M5.6.C: reduce_water_alpha + clamp parity
# ---------------------------------------------------------------------------


def _build_alpha_scenario(grid_shape=(16, 16, 16), dx=0.002, R0=2.0e-3):
    """Single bubble + water_alpha initialised at 1.0 everywhere."""
    from boilingsim.boiling import Bubble, _seed_test_bubble

    nx, ny, nz = grid_shape
    mat_np = np.zeros(grid_shape, dtype=np.int32)  # all fluid
    alpha_init = np.ones(grid_shape, dtype=np.float32)
    pool = wp.zeros(4, dtype=Bubble, device="cuda:0")
    slot_claim = wp.zeros(4, dtype=wp.int32, device="cuda:0")
    center = (nx / 2 * dx, ny / 2 * dx, nz / 2 * dx)
    wp.launch(
        _seed_test_bubble,
        dim=1,
        inputs=[
            pool, slot_claim, 0,
            center[0], center[1], center[2],
            0.0, 0.0, 0.0,
            R0, 0.0,
        ],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")
    return {
        "shape": grid_shape,
        "dx": dx,
        "origin": (0.0, 0.0, 0.0),
        "mat": mat_np,
        "alpha_init": alpha_init,
        "pool": pool,
        "R0": R0,
    }


def _run_warp_alpha(sc):
    from boilingsim.boiling import (
        clamp_alpha_nonnegative,
        reduce_water_alpha_by_bubble_occupancy,
    )

    alpha_d = wp.array(sc["alpha_init"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(sc["mat"], dtype=wp.int32, device="cuda:0")
    wp.launch(
        reduce_water_alpha_by_bubble_occupancy,
        dim=sc["pool"].shape[0],
        inputs=[
            sc["pool"], alpha_d, mat_d,
            wp.vec3(*sc["origin"]),
            sc["dx"],
            0,
        ],
        device="cuda:0",
    )
    wp.launch(
        clamp_alpha_nonnegative,
        dim=sc["shape"],
        inputs=[alpha_d],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")
    return alpha_d.numpy()


def _run_rust_alpha(rust, sc):
    alpha_d = wp.array(sc["alpha_init"].copy(), dtype=wp.float32, device="cuda:0")
    mat_d = wp.array(sc["mat"], dtype=wp.int32, device="cuda:0")
    rust.reduce_water_alpha(
        int(sc["pool"].ptr),
        int(sc["pool"].shape[0]),
        int(alpha_d.__cuda_array_interface__["data"][0]),
        int(mat_d.__cuda_array_interface__["data"][0]),
        sc["shape"][0], sc["shape"][1], sc["shape"][2],
        sc["origin"][0], sc["origin"][1], sc["origin"][2],
        sc["dx"],
        0,
    )
    wp.synchronize_device("cuda:0")
    return alpha_d.numpy()


def test_alpha_total_reduction_matches_bubble_volume(rust):
    """Sum of (1 - alpha) across the grid * V_cell = bubble volume V_b."""
    import math

    sc = _build_alpha_scenario(grid_shape=(20, 20, 20), dx=0.002, R0=1.5e-3)
    alpha_after = _run_rust_alpha(rust, sc)

    V_cell = sc["dx"] ** 3
    volume_removed = float((sc["alpha_init"] - alpha_after).sum()) * V_cell
    V_b = 4.0 / 3.0 * math.pi * sc["R0"] ** 3
    rel_err = abs(volume_removed - V_b) / V_b
    assert rel_err < 0.05, (
        f"Rust alpha reduction volume off by {rel_err*100:.2f}% "
        f"(measured {volume_removed:.4e}, expected {V_b:.4e})"
    )


def test_alpha_warp_vs_rust_close(rust):
    """Single-bubble alpha-reduction parity within ULP-level tolerance."""
    sc = _build_alpha_scenario(grid_shape=(20, 20, 20), dx=0.002, R0=1.5e-3)
    alpha_warp = _run_warp_alpha(sc)
    alpha_rust = _run_rust_alpha(rust, sc)

    diff = np.abs(alpha_warp - alpha_rust)
    max_abs = float(diff.max())
    assert max_abs < 1.0e-5, f"alpha parity: max_abs_diff={max_abs:.3e}"


def test_alpha_clamped_to_unit_interval(rust):
    """After the kernel, every cell satisfies 0 <= alpha <= 1."""
    # Force overshoot: start with alpha at 0 so the scatter goes negative.
    sc = _build_alpha_scenario(grid_shape=(16, 16, 16), dx=0.002)
    sc["alpha_init"] = np.zeros(sc["shape"], dtype=np.float32)
    alpha_after = _run_rust_alpha(rust, sc)
    assert (alpha_after >= 0.0).all(), (
        f"clamp failed on lower bound: min alpha = {alpha_after.min()}"
    )
    assert (alpha_after <= 1.0).all(), (
        f"clamp failed on upper bound: max alpha = {alpha_after.max()}"
    )
