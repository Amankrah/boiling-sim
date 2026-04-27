"""Phase 6.6 M2 unit tests for the run-artefact writer + ScalarHistory.

Covers the three artefact formats (HDF5 / CSV / JSON), the acceptance
gate logic, the ring-cap downsampling path, and the overall reset
behaviour.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Any

import h5py
import pytest

from boilingsim.config import load_scenario
from boilingsim.pipeline import ScalarSample
from boilingsim.run_writer import (
    SCALAR_CSV_FIELDS,
    ScalarHistory,
    write_run_artefacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    t: float,
    r: float = 99.0,
    leached: float = 0.0,
    degraded: float = 1.0,
    precip: float = 0.0,
    t_water: float = 99.9,
    step: int = 0,
) -> ScalarSample:
    # `step` is the sim-step counter at the moment the sample was
    # captured (Simulation.step_count). Defaults to 0 here so existing
    # call sites in this test module don't need to pass it; the new
    # test_step_count_propagates_to_summary test passes a non-zero
    # value to verify the JSON summary picks it up.
    return ScalarSample(
        t=t, dt=0.001, step=step,
        T_mean_water_c=t_water, T_max_water_c=t_water + 0.5, T_min_water_c=t_water - 0.5,
        T_max_wall_c=107.0, T_inner_wall_mean_c=106.5, T_inner_wall_max_c=107.2,
        u_max_mps=0.1,
        n_active_bubbles=42, mean_bubble_R_mm=0.6, mean_departed_bubble_R_mm=0.7,
        max_bubble_R_mm=1.0, alpha_min=0.0,
        retention_pct=r, leached_pct=leached, degraded_pct=degraded, precipitated_pct=precip,
    )


@pytest.fixture(scope="module")
def beta_carotene_cfg():
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    return cfg


# ---------------------------------------------------------------------------
# ScalarHistory
# ---------------------------------------------------------------------------


def test_scalar_history_appends_and_counts():
    h = ScalarHistory(target_duration_s=60.0, snapshot_hz=30.0)
    for i in range(100):
        h.append(_make_sample(t=i * 0.033))
    assert len(h) == 100
    assert h.samples[0].t == pytest.approx(0.0)
    assert h.samples[-1].t == pytest.approx(99 * 0.033)


def test_scalar_history_cap_downsamples_old_entries():
    """When the cap is exceeded, old entries get halved but recent ones
    stay dense. Concretely: cap = 50_000 at short durations, so a
    loop just past 50 000 should drop into the ~37 500 range
    (half of 25 000 old = 12 500 kept + latest 25 000 retained)."""
    h = ScalarHistory(target_duration_s=1.0, snapshot_hz=30.0)
    cap = h.cap
    t_final = (cap + 99) * 0.001
    for i in range(cap + 100):
        h.append(_make_sample(t=i * 0.001))
    # After overflow, the size should sit somewhere between cap/2 and
    # cap; never shoot above cap.
    assert cap // 2 <= len(h) <= cap
    # The last sample must still be the most recent one we pushed.
    assert h.samples[-1].t == pytest.approx(t_final)


def test_scalar_history_clear_resets():
    h = ScalarHistory(target_duration_s=60.0)
    for i in range(10):
        h.append(_make_sample(t=i * 0.1))
    h.clear()
    assert len(h) == 0


# ---------------------------------------------------------------------------
# Artefact writer -- CSV
# ---------------------------------------------------------------------------


def test_write_run_artefacts_csv_header_and_rows(beta_carotene_cfg, tmp_path: Path):
    h = ScalarHistory(target_duration_s=60.0)
    for i in range(5):
        h.append(_make_sample(t=i * 10.0, r=99.0 - i, leached=0.0, degraded=float(i)))
    h5_path, csv_path, json_path = write_run_artefacts(
        h, beta_carotene_cfg, run_id="test-run", out_dir=tmp_path,
        wall_clock_s=1.5,
        nutrient_primary_name="β-carotene",
    )
    assert csv_path.exists()
    # Header matches the canonical field list.
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert tuple(header) == SCALAR_CSV_FIELDS
    assert len(rows) == 5
    # Retention column tracks what we put in.
    retention_col = SCALAR_CSV_FIELDS.index("retention_pct")
    assert [float(r[retention_col]) for r in rows] == [99.0, 98.0, 97.0, 96.0, 95.0]
    # All three artefact paths exist.
    assert h5_path.exists()
    assert json_path.exists()


# ---------------------------------------------------------------------------
# Artefact writer -- HDF5
# ---------------------------------------------------------------------------


def test_write_run_artefacts_hdf5_has_expected_datasets(beta_carotene_cfg, tmp_path: Path):
    h = ScalarHistory(target_duration_s=60.0)
    for i in range(3):
        h.append(_make_sample(t=i))
    h5_path, _, _ = write_run_artefacts(
        h, beta_carotene_cfg, run_id="h5-test", out_dir=tmp_path,
        nutrient_primary_name="β-carotene",
    )
    with h5py.File(h5_path, "r") as f:
        # Top-level groups.
        assert "scalars" in f
        assert "meta" in f
        # Every canonical scalar field is a dataset of length 3.
        for name in SCALAR_CSV_FIELDS:
            assert name in f["scalars"], f"missing scalars/{name}"
            assert f["scalars"][name].shape == (3,)
        # Meta carries the run_id and echoed config.
        assert f["meta"].attrs["run_id"] == "h5-test"
        assert f["meta"].attrs["n_samples"] == 3
        assert f["meta"].attrs["schema_version"] == 3
        assert "scenario_config" in f["meta"]


# ---------------------------------------------------------------------------
# Artefact writer -- JSON summary + acceptance gates
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_summary_json_has_final_state_block(beta_carotene_cfg, tmp_path: Path):
    h = ScalarHistory(target_duration_s=60.0)
    for i in range(20):
        h.append(_make_sample(t=i * 30.0, r=90.0 - i * 0.1, t_water=99.9))
    _, _, json_path = write_run_artefacts(
        h, beta_carotene_cfg, run_id="json-test", out_dir=tmp_path,
        wall_clock_s=60.0, nutrient_primary_name="β-carotene",
    )
    summary = _read_json(json_path)
    assert summary["run_id"] == "json-test"
    assert summary["schema_version"] == 3
    assert summary["n_samples"] == 20
    assert summary["nutrient_primary_name"] == "β-carotene"
    final = summary["final"]
    assert "retention_pct" in final
    assert "T_water_mean_c" in final
    assert final["T_water_mean_c"] == pytest.approx(99.9, abs=1e-6)


def test_step_count_propagates_to_summary(beta_carotene_cfg, tmp_path: Path):
    """Phase 6.8: ScalarSample carries the sim's step counter, and
    write_run_artefacts pulls it into the JSON summary's `step_count`
    field. Before this fix, the writer used getattr(samples[-1], "step",
    0) on a ScalarSample that lacked a `step` attribute, so the field
    was always reported as 0 even after a 250k-step run.
    """
    h = ScalarHistory(target_duration_s=60.0)
    last_step = 0
    for i in range(15):
        # Deliberately monotonic-but-uneven steps -- mirrors the real
        # advection-CFL pattern where dt varies per step. We assert the
        # JSON summary picks up the FINAL sample's step value, not zero.
        last_step = i * 1000 + 7
        h.append(_make_sample(t=i * 40.0, t_water=99.9, step=last_step))
    _, _, json_path = write_run_artefacts(
        h, beta_carotene_cfg, run_id="step-test", out_dir=tmp_path,
        wall_clock_s=600.0, nutrient_primary_name="β-carotene",
    )
    summary = _read_json(json_path)
    assert summary["step_count"] == last_step, (
        f"summary['step_count'] = {summary['step_count']} but expected "
        f"{last_step} (the last ScalarSample's step field)"
    )


def test_summary_json_acceptance_gates_beta_carotene(beta_carotene_cfg, tmp_path: Path):
    """An in-band β-carotene run should pass every gate."""
    h = ScalarHistory(target_duration_s=60.0)
    # Samples that land R at 88 % (in the [80, 90] band); leached near
    # 0; degraded takes the rest; water at ~100 C; mass sum closes.
    for i in range(30):
        h.append(_make_sample(
            t=i * 20.0, r=88.0, leached=0.0, degraded=12.0, precip=0.0,
            t_water=99.95,
        ))
    _, _, json_path = write_run_artefacts(
        h, beta_carotene_cfg, run_id="accept", out_dir=tmp_path,
        nutrient_primary_name="β-carotene",
    )
    summary = _read_json(json_path)
    gates = summary["acceptance"]
    # At least: retention-in-band + mass-balance + water-at-sat + no-NaN.
    assert len(gates) >= 4
    for g in gates:
        assert g["passed"], f"gate failed: {g['name']} -- {g['detail']}"


def test_summary_json_acceptance_gates_flag_mass_drift(beta_carotene_cfg, tmp_path: Path):
    """A run where the four buckets don't sum to 100 must flag the
    mass-balance gate as failed."""
    h = ScalarHistory(target_duration_s=60.0)
    for i in range(10):
        # Intentionally break the invariant: sum = 95.
        h.append(_make_sample(
            t=i, r=80.0, leached=5.0, degraded=10.0, precip=0.0,
        ))
    _, _, json_path = write_run_artefacts(
        h, beta_carotene_cfg, run_id="mass-drift", out_dir=tmp_path,
        nutrient_primary_name="β-carotene",
    )
    summary = _read_json(json_path)
    mass_gate = next(g for g in summary["acceptance"] if "mass-balance" in g["name"].lower())
    assert mass_gate["passed"] is False


def test_summary_json_parameter_echo_present(beta_carotene_cfg, tmp_path: Path):
    h = ScalarHistory(target_duration_s=60.0)
    h.append(_make_sample(t=0.0))
    _, _, json_path = write_run_artefacts(
        h, beta_carotene_cfg, run_id="params", out_dir=tmp_path,
        nutrient_primary_name="β-carotene",
    )
    summary = _read_json(json_path)
    params = summary["parameters"]
    # Pot + nutrient sections must be present with their key fields.
    assert "pot" in params
    assert "nutrient" in params
    assert params["pot"]["diameter_m"] > 0
    assert params["nutrient"]["C0_mg_per_kg"] > 0


# ---------------------------------------------------------------------------
# Canonical field list matches ScalarSample's dataclass fields
# ---------------------------------------------------------------------------


def test_csv_fields_are_a_subset_of_scalar_sample():
    """Regression guard: if someone adds a field to ScalarSample they
    should add it to SCALAR_CSV_FIELDS too, and vice-versa."""
    sample_field_names = {f.name for f in dc_fields(ScalarSample)}
    for name in SCALAR_CSV_FIELDS:
        assert name in sample_field_names, (
            f"SCALAR_CSV_FIELDS references {name!r} which isn't a field on ScalarSample"
        )
