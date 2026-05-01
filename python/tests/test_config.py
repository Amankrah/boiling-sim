"""Tests for the Pydantic scenario configuration schema."""

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from boilingsim.config import (
    CarrotConfig,
    InitialConditionsConfig,
    PotConfig,
    ScenarioConfig,
    WaterConfig,
    load_scenario,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_YAML = ROOT / "configs" / "scenarios" / "default.yaml"


def test_default_yaml_loads():
    cfg = load_scenario(DEFAULT_YAML)
    assert cfg.pot.material == "steel_304"
    assert cfg.pot.diameter_m == 0.20
    assert cfg.water.fill_fraction == 0.75
    assert cfg.carrot.diameter_m == 0.025
    assert cfg.total_time_s == 900.0


def test_defaults_valid():
    cfg = ScenarioConfig()
    assert cfg.pot.diameter_m > 0
    assert 0 < cfg.water.fill_fraction < 1


def test_negative_diameter_rejected():
    with pytest.raises(ValidationError):
        PotConfig(diameter_m=-0.1)


def test_fill_fraction_out_of_range():
    with pytest.raises(ValidationError):
        WaterConfig(fill_fraction=1.2)
    with pytest.raises(ValidationError):
        WaterConfig(fill_fraction=0.0)


def test_wall_thicker_than_radius_rejected():
    with pytest.raises(ValidationError):
        PotConfig(diameter_m=0.01, wall_thickness_m=0.02)


def test_carrot_outside_pot_rejected():
    with pytest.raises(ValidationError):
        ScenarioConfig(carrot=CarrotConfig(position=(0.5, 0.0, 0.03)))


def test_carrot_above_water_line_rejected():
    with pytest.raises(ValidationError):
        ScenarioConfig(carrot=CarrotConfig(length_m=0.2, position=(0.0, 0.0, 0.03)))


def test_yaml_roundtrip(tmp_path: pathlib.Path):
    original = ScenarioConfig()
    path = tmp_path / "rt.yaml"
    path.write_text(yaml.safe_dump(original.model_dump()))
    roundtrip = load_scenario(path)
    assert roundtrip.model_dump() == original.model_dump()


def test_initial_conditions_default_is_cold():
    cfg = ScenarioConfig()
    assert cfg.initial_conditions.mode == "cold"
    assert cfg.initial_conditions.preheat_water_c == 95.0
    assert cfg.initial_conditions.preheat_wall_c == 100.0
    assert cfg.initial_conditions.preheat_carrot_c == 20.0


def test_initial_conditions_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        InitialConditionsConfig(mode="sideways")


def test_initial_conditions_preheat_range_enforced():
    with pytest.raises(ValidationError):
        InitialConditionsConfig(mode="preheat", preheat_water_c=200.0)


# ---------------------------------------------------------------------------
# v6: multi-carrot count + axis + auto-placement
# ---------------------------------------------------------------------------


def test_carrot_count_default_is_one():
    """Backward-compat: every existing benchmark/test relies on
    count=1 unless they explicitly opt into multi-carrot."""
    cfg = CarrotConfig()
    assert cfg.count == 1
    assert cfg.axis == "z"


def test_carrot_count_axis_validates_in_pot():
    """3 horizontal carrots auto-place inside the default pot."""
    cfg = ScenarioConfig.model_validate({
        "carrot": {
            "count": 3,
            "axis": "x",
            "diameter_m": 0.025,
            "length_m": 0.06,
            "position": [0.0, 0.0, 0.04],
        }
    })
    assert cfg.carrot.count == 3
    # ~91.9 g for 3 × 60 mm × 25 mm carrots at ρ=1040.
    assert 80 < cfg.carrot.total_mass_g() < 105


def test_overcrowded_pot_rejected_at_validation():
    """64 large carrots cannot all fit inside a 20 cm pot."""
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate({
            "carrot": {
                "count": 64,
                "axis": "x",
                "diameter_m": 0.040,  # 40 mm dia, 64 of them packed: way too wide
                "length_m": 0.060,
                "position": [0.0, 0.0, 0.040],
            }
        })


def test_carrot_axis_invalid_value_rejected():
    """Pydantic Literal must reject anything outside {x,y,z}."""
    with pytest.raises(ValidationError):
        CarrotConfig(axis="diagonal")  # type: ignore[arg-type]
