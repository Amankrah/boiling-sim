"""Phase 5 M1: Warp ↔ cudarc pointer-sharing PoC.

These tests prove the Phase 5 FFI memory contract end-to-end:

1. Python (Warp) allocates a CUDA device buffer.
2. Python hands the raw device pointer to Rust via the public
   ``__cuda_array_interface__["data"][0]`` protocol.
3. Rust launches a hand-written CUDA kernel that mutates the buffer in place.
4. Python reads back the same buffer and observes the mutation, with no
   host-side copy in the middle.

If any of these tests regresses, the Phase 5 acceleration path is broken
at its foundation -- M2 and M3 cannot proceed. Bisect against
[`crates/cuda-kernels/src/scale.cu`](../../crates/cuda-kernels/src/scale.cu)
and [`crates/sim-core/src/cuda.rs`](../../crates/sim-core/src/cuda.rs).
"""

from __future__ import annotations

import numpy as np
import pytest

import warp as wp


pytestmark = pytest.mark.cuda_required


@pytest.fixture(scope="module")
def sim_core_cuda():
    sim_core = pytest.importorskip("sim_core")
    if sim_core.cuda is None:
        pytest.skip("sim_core.cuda submodule unavailable -- CUDA driver missing")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Warp reports no CUDA device available")
    return sim_core.cuda


@pytest.fixture(scope="module")
def sim(sim_core_cuda):
    return sim_core_cuda.SimCore(0)


def test_warp_ptr_matches_cuda_array_interface():
    """``wp.array.ptr`` must equal ``__cuda_array_interface__["data"][0]``.

    Plan M1 safeguard: ``wp.array.ptr`` is a private Warp attribute that is
    stable today but could drift in a future Warp version. By passing the
    pointer through the public CAI protocol while also cross-checking
    against the private attribute, a desync between the two fails loudly
    here instead of silently corrupting GPU memory in a Rust kernel.
    """
    wp.init()
    arr = wp.array(np.ones(64, dtype=np.float32), device="cuda:0")
    ptr_private = int(arr.ptr)
    ptr_public = int(arr.__cuda_array_interface__["data"][0])
    assert ptr_private == ptr_public, (
        f"Warp's private .ptr ({ptr_private:#x}) and public "
        f"__cuda_array_interface__['data'][0] ({ptr_public:#x}) disagree -- "
        "Warp may have changed its array layout. Update the contract in "
        "crates/sim-core/src/cuda.rs::scale_array."
    )


def test_scale_array_doubles_in_place(sim):
    """End-to-end FFI smoke test: allocate ones via Warp, double via Rust,
    read 2s back. Uses integer scale (2.0) to keep the assertion bit-exact
    and remove any floating-point ambiguity from the round-trip."""
    n = 4096
    arr = wp.array(np.ones(n, dtype=np.float32), device="cuda:0")
    raw_ptr = int(arr.__cuda_array_interface__["data"][0])

    sim.scale_array(raw_ptr, n, 2.0)
    wp.synchronize_device("cuda:0")

    back = arr.numpy()
    assert back.dtype == np.float32
    assert back.shape == (n,)
    # Every value is exactly 2.0; bit-equal because we started at 1.0 and
    # multiplied by 2.0 (no rounding error).
    assert (back == 2.0).all(), (
        f"scale_array didn't double the buffer in place: "
        f"min={back.min()}, max={back.max()}, mean={back.mean()}"
    )


def test_scale_array_handles_arbitrary_factor(sim):
    """Real-valued scale: confirm the kernel applies the float arithmetic
    correctly (within float32 precision)."""
    n = 1024
    src = np.linspace(0.0, 1.0, n, dtype=np.float32)
    arr = wp.array(src, device="cuda:0")
    raw_ptr = int(arr.__cuda_array_interface__["data"][0])

    scale = np.float32(0.5)
    sim.scale_array(raw_ptr, n, float(scale))
    wp.synchronize_device("cuda:0")

    expected = src * scale
    back = arr.numpy()
    np.testing.assert_allclose(back, expected, rtol=0.0, atol=0.0)


def test_scale_array_zero_n_is_noop(sim):
    """n=0 must early-return without touching the buffer."""
    arr = wp.array(np.array([3.0, 4.0, 5.0], dtype=np.float32), device="cuda:0")
    raw_ptr = int(arr.__cuda_array_interface__["data"][0])
    sim.scale_array(raw_ptr, 0, 99.0)
    wp.synchronize_device("cuda:0")
    np.testing.assert_array_equal(arr.numpy(), [3.0, 4.0, 5.0])


def test_scale_array_idempotent_across_calls(sim):
    """Repeated calls compose: 2 * 3 * 5 = 30."""
    n = 256
    arr = wp.array(np.ones(n, dtype=np.float32), device="cuda:0")
    raw_ptr = int(arr.__cuda_array_interface__["data"][0])
    sim.scale_array(raw_ptr, n, 2.0)
    sim.scale_array(raw_ptr, n, 3.0)
    sim.scale_array(raw_ptr, n, 5.0)
    wp.synchronize_device("cuda:0")
    assert (arr.numpy() == 30.0).all()
