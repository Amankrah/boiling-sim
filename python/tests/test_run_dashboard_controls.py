"""Phase 6.6 M4: unit tests for the new control-message apply logic.

Exercises ``apply_control_rebuild`` in isolation (no Simulation, no
TCP sockets, no Rust relay). Focused on the new v3 semantics:
``set_config`` validates via Pydantic; ``set_nutrient`` returns a
clean error on unknown preset; ``set_material`` / ``set_carrot_size``
keep their pre-v3 behaviour intact.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from boilingsim.config import load_scenario


# ---------------------------------------------------------------------------
# Module loader -- scripts/run_dashboard.py isn't in python/boilingsim/, so
# we load it via the filesystem path. The same pattern capture_sample_snapshot
# uses in its own invocation shim.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dashboard_mod():
    root = pathlib.Path(__file__).resolve().parents[2]
    src = root / "scripts" / "run_dashboard.py"
    sys.path.insert(0, str(root / "python"))
    spec = importlib.util.spec_from_file_location("run_dashboard", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def base_cfg():
    cfg = load_scenario("configs/scenarios/default.yaml")
    cfg.nutrient.enabled = True
    return cfg


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------


def test_set_config_full_valid_applies(run_dashboard_mod, base_cfg):
    """Full ScenarioConfig JSON validates and replaces the cfg."""
    new_cfg_json = base_cfg.model_dump(mode="json")
    new_cfg_json["pot"]["material"] = "copper"
    new_cfg_json["total_time_s"] = 120.0
    new_cfg_json["heating"]["base_heat_flux_w_per_m2"] = 45000.0

    msg = {"type": "set_config", "config": new_cfg_json}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert err == ""
    assert returned.pot.material == "copper"
    assert returned.total_time_s == 120.0
    assert returned.heating.base_heat_flux_w_per_m2 == 45000.0


def test_set_config_invalid_returns_error_keeps_cfg(run_dashboard_mod, base_cfg):
    """Pydantic rejection: err non-empty, cfg unchanged."""
    bad = base_cfg.model_dump(mode="json")
    # diameter_m has gt=0; set it negative to force validation error.
    bad["pot"]["diameter_m"] = -0.1
    msg = {"type": "set_config", "config": bad}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert err != ""
    assert "set_config validation failed" in err
    # Original config is untouched -- caller sees same pot.diameter_m.
    assert returned is base_cfg
    assert returned.pot.diameter_m > 0


def test_set_config_missing_field_defaults_from_pydantic(run_dashboard_mod, base_cfg):
    """Pydantic fills in omitted fields from their defaults."""
    minimal = {"pot": {"material": "aluminum"}, "total_time_s": 60.0}
    msg = {"type": "set_config", "config": minimal}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert err == ""
    assert returned.pot.material == "aluminum"
    # Defaults filled in:
    assert returned.pot.diameter_m > 0
    assert returned.water.fill_fraction > 0
    # Total time came through.
    assert returned.total_time_s == 60.0


def test_set_config_rejects_non_object(run_dashboard_mod, base_cfg):
    msg = {"type": "set_config", "config": "not an object"}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert "config" in err and "not an object" in err
    assert returned is base_cfg


# ---------------------------------------------------------------------------
# set_material + set_carrot_size (unchanged pre-v3 path, regression guard)
# ---------------------------------------------------------------------------


def test_set_material_changes_pot_material(run_dashboard_mod, base_cfg):
    msg = {"type": "set_material", "value": "copper"}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert err == ""
    assert returned.pot.material == "copper"


def test_set_material_unknown_value_is_ignored(run_dashboard_mod, base_cfg):
    msg = {"type": "set_material", "value": "unobtainium"}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    # Pre-v3 behaviour: unknown materials silently dropped, no error.
    assert err == ""
    # Material stayed at default.
    assert returned.pot.material != "unobtainium"


def test_set_carrot_size_converts_mm_to_m(run_dashboard_mod, base_cfg):
    msg = {"type": "set_carrot_size", "diameter_mm": 30, "length_mm": 80}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert err == ""
    assert returned.carrot.diameter_m == pytest.approx(0.030)
    assert returned.carrot.length_m == pytest.approx(0.080)


# ---------------------------------------------------------------------------
# set_nutrient preset
# ---------------------------------------------------------------------------


def test_set_nutrient_vitamin_c_preset(run_dashboard_mod, base_cfg):
    msg = {"type": "set_nutrient", "value": "vitamin_c"}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert err == ""
    # Vitamin C signature: K_partition == 1.0, C_water_sat >> 1.
    assert returned.nutrient.K_partition == pytest.approx(1.0)
    assert returned.nutrient.C_water_sat_mg_per_kg >= 1.0e3


def test_set_nutrient_unknown_preset_returns_error(run_dashboard_mod, base_cfg):
    msg = {"type": "set_nutrient", "value": "unobtainium"}
    returned, err = run_dashboard_mod.apply_control_rebuild(base_cfg, msg)
    assert err != ""
    assert "unknown" in err.lower()
    assert returned is base_cfg


# ---------------------------------------------------------------------------
# apply_control_live routes new message types correctly
# ---------------------------------------------------------------------------


def test_apply_control_live_set_config_schedules_rebuild(run_dashboard_mod, base_cfg):
    """Pseudo-sim (just the cfg attribute) -- set_config always
    schedules a rebuild via `apply_control_live` returning True."""

    class _FakeSim:
        def __init__(self, cfg):
            self.cfg = cfg

    sim = _FakeSim(base_cfg)
    msg = {"type": "set_config", "config": {"pot": {"material": "copper"}}}
    assert run_dashboard_mod.apply_control_live(sim, msg) is True


def test_apply_control_live_start_run_handled_elsewhere(run_dashboard_mod, base_cfg):
    """start_run is NOT a rebuild trigger -- the main loop applies
    it directly. `apply_control_live` must return False."""

    class _FakeSim:
        def __init__(self, cfg):
            self.cfg = cfg

    sim = _FakeSim(base_cfg)
    msg = {"type": "start_run", "duration_s": 300.0}
    assert run_dashboard_mod.apply_control_live(sim, msg) is False


def test_apply_control_live_set_heat_flux_mutates(run_dashboard_mod, base_cfg):
    class _FakeSim:
        def __init__(self, cfg):
            self.cfg = cfg

    sim = _FakeSim(base_cfg)
    msg = {"type": "set_heat_flux", "value": 42000.0}
    assert run_dashboard_mod.apply_control_live(sim, msg) is False
    assert sim.cfg.heating.base_heat_flux_w_per_m2 == 42000.0
