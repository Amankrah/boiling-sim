"""Produce `target/sample_snapshot.mp` for the Rust cross-stack test.

Run once after any schema bump in boilingsim.dashboard / ws-server::snapshot.
The Rust integration test crates/ws-server/tests/python_snapshot.rs loads
this file and verifies Rust can decode the Python producer's msgpack bytes.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from boilingsim.config import load_scenario  # noqa: E402
from boilingsim.dashboard import serialize_snapshot  # noqa: E402
from boilingsim.pipeline import Simulation  # noqa: E402


def main() -> int:
    cfg = load_scenario(ROOT / "configs" / "scenarios" / "default.yaml")
    cfg.nutrient.enabled = True
    cfg.nutrient2 = cfg.nutrient.model_copy(update={"enabled": True})
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.004

    sim = Simulation(cfg)
    # Five steps is enough to populate bubbles + non-trivial diagnostics
    # without spending real wall time.
    for _ in range(5):
        sim.step()

    out = ROOT / "target" / "sample_snapshot.mp"
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = serialize_snapshot(sim, step=5)
    out.write_bytes(buf)
    print(f"wrote {out} ({len(buf)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
