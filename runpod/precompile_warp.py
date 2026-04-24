"""Warp JIT pre-compile for Dockerfile.runpod.

Runs a 3-step coarse simulation so every GPU kernel is compiled + cached
before the container is pushed. Saves ~45 s on every cold pod start.

Exits 0 if successful OR if no CUDA device is visible at build time; the
caller in Dockerfile.runpod is responsible for not invoking this script
on GPU-less hosts (which would hang on ``wp.init()``).

Wrapped in try/except so a Warp internal error doesn't fail the build --
the worst case is we lose the ~45 s cold-start speedup; the image is
still fully functional.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import warp as wp
    except ImportError as e:
        print(f"[precompile] warp-lang not importable: {e}", file=sys.stderr)
        return 0  # Don't fail the build.

    try:
        wp.init()
    except Exception as e:
        print(f"[precompile] wp.init() failed: {e}", file=sys.stderr)
        return 0

    n = wp.get_cuda_device_count() if hasattr(wp, "get_cuda_device_count") else 0
    if n == 0:
        print("[precompile] No CUDA device visible; skipping JIT warm-up.")
        return 0

    try:
        from boilingsim.config import load_scenario
        from boilingsim.pipeline import Simulation
    except ImportError as e:
        print(f"[precompile] boilingsim not importable: {e}", file=sys.stderr)
        return 0

    try:
        cfg = load_scenario("configs/scenarios/default.yaml")
        cfg.grid.dx_m = 0.004  # coarse, just to touch every kernel
        cfg.boiling.enabled = True
        sim = Simulation(cfg, device="cuda:0")
        for _ in range(3):
            sim.step()
        print("[precompile] Warp JIT cache warmed over 3 coarse-grid steps.")
    except Exception as e:
        print(f"[precompile] Simulation warm-up failed: {e}", file=sys.stderr)
        # Still return 0 — image is usable, cold-start will JIT normally.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
