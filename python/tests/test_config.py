"""Tests for the Pydantic scenario configuration schema."""

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from boilingsim.config import (
    CarrotConfig,
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
    assert cfg.total_time_s == 600.0


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
