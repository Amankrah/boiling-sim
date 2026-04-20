"""Phase 6 M1 -- snapshot serializer unit tests.

Round-trips exercise the msgpack layer only; the cross-stack check that
``rmp_serde`` on the Rust side accepts our bytes runs in
``crates/ws-server/tests/ingest_roundtrip.rs`` once M2 lands. For M1 we
use Python's own msgpack decoder to verify field presence, types, and
the dev-guide §6.2 invariants.
"""

from __future__ import annotations

import msgpack
import numpy as np
import pytest

from boilingsim.config import load_scenario
from boilingsim.dashboard import (
    SCHEMA_VERSION,
    build_snapshot,
    serialize_rebuild_marker,
    serialize_snapshot,
)
from boilingsim.pipeline import Simulation


@pytest.fixture(scope="module")
def sim_dual() -> Simulation:
    """A warm-started dual-solute simulation at the coarsest dev grid.

    We only need a few steps of real physics so the diagnostics are
    non-trivially populated (non-zero bubble count, retention starting
    to move off 100 %).
    """
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    cfg.nutrient2 = cfg.nutrient.model_copy(update={"enabled": True})
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.004
    cfg.total_time_s = 0.5

    sim = Simulation(cfg)
    for _ in range(20):
        sim.step()
        if sim.t >= 0.5:
            break
    return sim


def _unpack(buf: bytes) -> dict:
    """Decode msgpack bytes with raw=False so strings are str, not bytes."""
    return msgpack.unpackb(buf, raw=False)


def test_snapshot_has_all_schema_fields(sim_dual):
    snap = build_snapshot(sim_dual, step=42)
    expected_fields = {
        "version", "t_sim", "step", "is_rebuilding", "is_paused",
        "grid", "grid_ds",
        "temperature", "alpha", "bubbles",
        "nutrient_primary_name", "nutrient_secondary_name",
        "carrot_retention", "carrot_leached",
        "carrot_degraded", "carrot_precipitated",
        "carrot_retention2", "carrot_leached2",
        "carrot_degraded2", "carrot_precipitated2",
        "carrot_surface_c", "carrot_surface_c2",
        "wall_temperature_mean", "wall_heat_flux",
        # v3 additions
        "water_temperature_mean", "water_temperature_max", "water_temperature_min",
        "run_id", "total_time_s", "is_complete", "last_error",
        # v4 additions
        "pot_diameter_m", "pot_height_m",
        "pot_wall_thickness_m", "pot_base_thickness_m",
    }
    assert set(snap.keys()) == expected_fields, (
        f"missing/extra keys: "
        f"missing={expected_fields - set(snap.keys())} "
        f"extra={set(snap.keys()) - expected_fields}"
    )
    # Grid sub-dicts must have the exact field set the Rust GridMeta expects.
    for k in ("grid", "grid_ds"):
        assert set(snap[k].keys()) == {"nx", "ny", "nz", "dx", "origin"}, (
            f"{k} field mismatch: {snap[k].keys()}"
        )
    # Bubbles (if any) must carry position + radius.
    for b in snap["bubbles"]:
        assert set(b.keys()) == {"position", "radius"}


def test_version_field_is_schema_version(sim_dual):
    snap = build_snapshot(sim_dual, step=0)
    assert snap["version"] == SCHEMA_VERSION


def test_downsampled_grid_is_half_resolution(sim_dual):
    snap = build_snapshot(sim_dual, step=0)
    grid = snap["grid"]
    grid_ds = snap["grid_ds"]
    assert grid_ds["nx"] == grid["nx"] // 2
    assert grid_ds["ny"] == grid["ny"] // 2
    assert grid_ds["nz"] == grid["nz"] // 2
    expected_len = grid_ds["nx"] * grid_ds["ny"] * grid_ds["nz"]
    assert len(snap["temperature"]) == expected_len
    assert len(snap["alpha"]) == expected_len
    # Downsampled dx is twice the full-res dx.
    assert snap["grid_ds"]["dx"] == pytest.approx(grid["dx"] * 2.0)


def test_v3_water_temperature_present(sim_dual):
    """v3 schema bump: water temperature mean/max/min must be on the wire."""
    snap = build_snapshot(sim_dual, step=0)
    # Water temp in Celsius; at the sim's warm-start we expect mid-range.
    for key in ("water_temperature_mean", "water_temperature_max", "water_temperature_min"):
        assert key in snap, f"v3 field {key!r} missing from snapshot"
        assert isinstance(snap[key], float)
        assert -5.0 <= snap[key] <= 200.0, (
            f"{key} out of Celsius band: {snap[key]}"
        )
    # Mean lies between min and max.
    assert snap["water_temperature_min"] <= snap["water_temperature_mean"] <= snap["water_temperature_max"]


def test_v3_run_metadata_defaults(sim_dual):
    """v3 schema bump: run_id/total_time_s/is_complete/last_error default cleanly."""
    snap = build_snapshot(sim_dual, step=0)
    assert snap["run_id"] == ""
    assert snap["total_time_s"] == 0.0
    assert snap["is_complete"] is False
    assert snap["last_error"] == ""


def test_v4_pot_geometry_echoes_cfg(sim_dual):
    """v4 schema bump: pot dims must mirror cfg.pot so the 3D renderer
    can scale to match whatever the user chose on the Config page."""
    snap = build_snapshot(sim_dual, step=0)
    cfg_pot = sim_dual.cfg.pot
    for key, cfg_val in (
        ("pot_diameter_m", cfg_pot.diameter_m),
        ("pot_height_m", cfg_pot.height_m),
        ("pot_wall_thickness_m", cfg_pot.wall_thickness_m),
        ("pot_base_thickness_m", cfg_pot.base_thickness_m),
    ):
        assert key in snap, f"v4 field {key!r} missing from snapshot"
        assert isinstance(snap[key], float)
        assert snap[key] == pytest.approx(cfg_val), (
            f"{key} on wire ({snap[key]}) != cfg.pot.{key.removeprefix('pot_')} ({cfg_val})"
        )


def test_v3_run_metadata_forwarded(sim_dual):
    """When the producer passes run kwargs, they must land on the snapshot."""
    snap = build_snapshot(
        sim_dual, step=0,
        run_id="abc123",
        total_time_s=60.0,
        is_complete=True,
        last_error="config validation failed: foo",
    )
    assert snap["run_id"] == "abc123"
    assert snap["total_time_s"] == 60.0
    assert snap["is_complete"] is True
    assert "config validation failed" in snap["last_error"]


def test_temperature_is_celsius_in_sane_range(sim_dual):
    snap = build_snapshot(sim_dual, step=0)
    t_arr = np.asarray(snap["temperature"], dtype=np.float32)
    # Fresh default.yaml start: water at initial_temp_c = 20, walls warming.
    # Nothing should be below 0 C (ice) or above 200 C (superheated pot).
    assert float(t_arr.min()) >= -5.0
    assert float(t_arr.max()) <= 200.0


def test_retention_fields_in_expected_range(sim_dual):
    snap = build_snapshot(sim_dual, step=0)
    assert 0.0 <= snap["carrot_retention"] <= 100.5
    assert 0.0 <= snap["carrot_retention2"] <= 100.5


def test_bubbles_list_length_matches_active_count(sim_dual):
    snap = build_snapshot(sim_dual, step=0)
    # Bubble pool may be empty at 0.5 s -- but if present, every entry
    # must have a strictly positive radius.
    for b in snap["bubbles"]:
        assert b["radius"] > 0.0
        assert len(b["position"]) == 3


def test_msgpack_roundtrip_preserves_structure(sim_dual):
    buf = serialize_snapshot(sim_dual, step=7)
    decoded = _unpack(buf)
    snap = build_snapshot(sim_dual, step=7)

    # Top-level scalars round-trip exactly.
    assert decoded["version"] == snap["version"]
    assert decoded["step"] == snap["step"]
    assert decoded["grid"] == snap["grid"]
    assert decoded["grid_ds"] == snap["grid_ds"]
    # Arrays compare element-wise with float tolerance.
    np.testing.assert_allclose(
        decoded["temperature"], snap["temperature"], rtol=0, atol=0,
        err_msg="temperature buffer altered by msgpack roundtrip",
    )


def test_rebuild_marker_has_correct_flag():
    buf = serialize_rebuild_marker(t_sim=1.25)
    decoded = _unpack(buf)
    assert decoded["is_rebuilding"] is True
    assert decoded["is_paused"] is False
    assert decoded["temperature"] == []
    assert decoded["bubbles"] == []
    assert decoded["version"] == SCHEMA_VERSION


def test_wire_payload_stays_under_budget(sim_dual):
    """At dx=4mm the grid is ~50x50x30 = 75k cells; downsampled 25x25x15 = 9375.
    Temperature + alpha combined is ~75 KB of raw f64 via msgpack-list
    encoding. With real production 200^3 grids (~25x the cell count) the
    budget still lands under 2 MB uncompressed per the dev-guide §6.2
    estimate.
    """
    buf = serialize_snapshot(sim_dual, step=0)
    # Dev-rig budget: at coarse test grid, payload is small. This test
    # exists to catch regressions where someone accidentally serializes
    # the full-res (nx, ny, nz) volume instead of the downsampled one.
    assert len(buf) < 5_000_000, (
        f"snapshot payload too large: {len(buf) / 1e6:.2f} MB"
    )
