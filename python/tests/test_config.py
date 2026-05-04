"""Tests for the Pydantic scenario configuration schema."""

import math
import pathlib

import pytest
import yaml
from pydantic import ValidationError

from boilingsim.config import (
    CarrotConfig,
    ExtraIngredientConfig,
    InitialConditionsConfig,
    PotConfig,
    ScenarioConfig,
    WaterConfig,
    load_scenario,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_YAML = ROOT / "configs" / "scenarios" / "single_carrot.yaml"


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


# ---------------------------------------------------------------------------
# M2: gram-as-primary input
# ---------------------------------------------------------------------------


def test_mass_mode_derives_length():
    """mass_mode='mass' with 200 g target / 4 carrots / 25 mm dia derives
    length ≈ 0.0979 m (volume = 200/4/1040 = 4.808e-5 m³, length = vol /
    π / (0.0125)²)."""
    cfg = CarrotConfig(
        mass_mode="mass",
        target_mass_g=200.0,
        count=4,
        diameter_m=0.025,
    )
    # Derived length ≈ 0.0979 m; computed mass round-trips to 200.00 g.
    assert abs(cfg.length_m - 0.0979) < 1.0e-3
    assert abs(cfg.total_mass_g() - 200.0) < 1.0e-2


def test_mass_mode_with_unfittable_target_rejects():
    """1 kg of carrots cannot fit in a 200 mm diameter pot at count=2 +
    25 mm dia (each would need ~979 mm length, far exceeding the inner
    diameter). The fits-in-pot validator must surface this clearly."""
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate({
            "carrot": {
                "mass_mode": "mass",
                "target_mass_g": 1000.0,  # 1 kg
                "count": 2,
                "axis": "x",
                "diameter_m": 0.025,
                "position": [0.0, 0.0, 0.040],
            }
        })


def test_dimensions_mode_unchanged_back_compat():
    """Default mass_mode='dimensions' preserves explicit length_m and
    just exposes total_mass_g() for display. No legacy YAML is broken."""
    cfg = CarrotConfig(diameter_m=0.025, length_m=0.05, count=1)
    assert cfg.mass_mode == "dimensions"
    assert cfg.target_mass_g is None
    assert cfg.length_m == 0.05  # untouched
    # Sanity: derived display mass is consistent with the volume formula.
    expected_g = math.pi * (0.0125 ** 2) * 0.05 * 1040.0 * 1000.0
    assert abs(cfg.total_mass_g() - expected_g) < 1.0e-3


def test_mass_mode_requires_target_mass_g():
    """mass_mode='mass' without a target raises a clear validation error."""
    with pytest.raises(ValidationError):
        CarrotConfig(mass_mode="mass")


# ---------------------------------------------------------------------------
# M4: multi-ingredient
# ---------------------------------------------------------------------------


def test_legacy_carrot_yaml_loads_without_extras():
    """Existing YAML with no extra_ingredients block continues to load
    cleanly; n_ingredients == 1; iter_ingredients yields a single tuple."""
    cfg = load_scenario(DEFAULT_YAML)
    assert cfg.n_ingredients == 1
    assert cfg.extra_ingredients == []
    ings = cfg.iter_ingredients()
    assert len(ings) == 1
    geom, nut, nut2 = ings[0]
    assert geom is cfg.carrot
    assert nut is cfg.nutrient
    assert nut2 is cfg.nutrient2


def test_extra_ingredients_iter_order():
    """iter_ingredients yields legacy carrot first, then extras in
    declaration order. Drives voxel ingredient_id assignment."""
    cfg = ScenarioConfig.model_validate({
        "extra_ingredients": [
            {
                "name": "potato",
                "count": 2,
                "axis": "x",
                "diameter_m": 0.025,
                "length_m": 0.040,
                "position": [0.0, -0.040, 0.040],
            },
            {
                "name": "onion",
                "count": 1,
                "axis": "z",
                "diameter_m": 0.030,
                "length_m": 0.030,
                "position": [0.040, 0.040, 0.020],
            },
        ]
    })
    assert cfg.n_ingredients == 3
    names = [g.name for g, _, _ in cfg.iter_ingredients()]
    assert names == ["carrot", "potato", "onion"]


def test_extra_ingredient_density_drives_mass():
    """Per-ingredient density flows through to total_mass_g(). Potato
    at 1080 kg/m³ vs default carrot at 1040 -- different mass for the
    same volume."""
    same_volume = ExtraIngredientConfig(
        name="potato",
        diameter_m=0.025,
        length_m=0.05,
        count=1,
        axis="z",
        density_kg_per_m3=1080.0,
    )
    carrot_default = CarrotConfig(diameter_m=0.025, length_m=0.05, count=1)
    # Same volume; mass ratio == density ratio.
    ratio = same_volume.total_mass_g() / carrot_default.total_mass_g()
    assert abs(ratio - 1080.0 / 1040.0) < 1.0e-3


def test_extra_ingredient_must_fit_in_pot():
    """Auto-placement still validates each extra ingredient's instances
    against the inner-pot radius and water line."""
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate({
            "extra_ingredients": [{
                "name": "giant",
                "count": 1,
                "axis": "z",
                "diameter_m": 0.050,
                "length_m": 0.500,  # 500 mm -- way taller than the 120 mm pot
                "position": [0.0, 0.0, 0.040],
            }]
        })


# ---------------------------------------------------------------------------
# M6: streamlined YAML schema (clean ingredients[] + couplings[] form)
# ---------------------------------------------------------------------------


def _clean_form_dict() -> dict:
    """Hand-built clean YAML form used by several M6 tests so each can
    tweak just the field it cares about."""
    return {
        "ingredients": [
            {
                "name": "carrot",
                "count": 3,
                "axis": "x",
                "diameter_m": 0.025,
                "length_m": 0.060,
                "position": [0.0, 0.0, 0.040],
                "nutrients": {
                    "beta_carotene": {
                        "enabled": True,
                        "C0_mg_per_kg": 83.0,
                        "K_partition": 1.0e-5,
                    },
                    "vitamin_c": {
                        "enabled": True,
                        "C0_mg_per_kg": 60.0,
                        "K_partition": 1.0,
                    },
                },
            },
            {
                "name": "potato",
                "count": 2,
                "axis": "x",
                "diameter_m": 0.030,
                "length_m": 0.040,
                "position": [0.0, -0.050, 0.025],
                "density_kg_per_m3": 1080,
                "nutrients": {
                    "starch": {
                        "enabled": True,
                        "C0_mg_per_kg": 200000.0,
                        "K_partition": 0.5,
                    },
                },
            },
        ],
        "couplings": [{
            "protector": "carrot.vitamin_c",
            "protected": "carrot.beta_carotene",
            "eta": 0.5,
            "c_ref_mg_per_kg": 5.0,
        }],
    }


def test_clean_yaml_form_translates_to_legacy_layout():
    """The clean ``ingredients[]`` form translates to the same legacy
    ``cfg.carrot`` + ``cfg.nutrient`` + ``cfg.nutrient2`` +
    ``cfg.extra_ingredients`` shape every Python call site already
    expects. Spot-check the key fields."""
    cfg = ScenarioConfig.model_validate(_clean_form_dict())
    # Ingredient 0 -> top-level legacy fields.
    assert cfg.carrot.name == "carrot"
    assert cfg.carrot.count == 3
    assert cfg.carrot.axis == "x"
    assert cfg.nutrient.name == "beta_carotene"
    assert cfg.nutrient.enabled is True
    assert cfg.nutrient.K_partition == 1.0e-5
    assert cfg.nutrient2.name == "vitamin_c"
    assert cfg.nutrient2.enabled is True
    # Ingredient 1 -> extras[0].
    assert len(cfg.extra_ingredients) == 1
    extra = cfg.extra_ingredients[0]
    assert extra.name == "potato"
    assert extra.nutrient.name == "starch"
    assert extra.nutrient.K_partition == 0.5
    # Coupling resolved from dotted id.
    assert len(cfg.nutrient_couplings) == 1
    cc = cfg.nutrient_couplings[0]
    assert cc.protector_ingredient == "carrot"
    assert cc.protector_slot == "secondary"  # vitamin_c is the 2nd nutrient key
    assert cc.protected_ingredient == "carrot"
    assert cc.protected_slot == "primary"


def test_legacy_yaml_form_still_loads():
    """The 14 untouched legacy scenario YAMLs continue to load through
    ``load_scenario`` without translation interference."""
    legacy_paths = [
        ROOT / "configs" / "scenarios" / "single_carrot.yaml",
        ROOT / "configs" / "scenarios" / "boiling_q20.yaml",
        ROOT / "configs" / "scenarios" / "household_boil.yaml",
    ]
    for p in legacy_paths:
        cfg = load_scenario(p)
        # Legacy form: top-level carrot block populates cfg.carrot directly,
        # n_ingredients == 1 (no extras).
        assert cfg.n_ingredients == 1, f"{p.name} should be single-ingredient"
        assert cfg.extra_ingredients == []


def test_clean_form_default_yaml_loads_and_renders_three_ingredients():
    """The dashboard's default scenario uses the new clean form. Exercises
    the full translation: 3 ingredients (carrot + potato + onion), one
    coupling, every nutrient .name populated."""
    cfg = load_scenario(ROOT / "configs" / "scenarios" / "default.yaml")
    assert cfg.n_ingredients == 3
    names = [g.name for g, _, _ in cfg.iter_ingredients()]
    assert names == ["carrot", "potato", "onion"]
    # The coupling resolves correctly.
    assert len(cfg.nutrient_couplings) == 1


def test_hybrid_legacy_and_clean_yaml_rejected():
    """A YAML mixing top-level ``carrot:`` (legacy) AND ``ingredients:``
    (clean) is ambiguous and must be rejected with a clear error."""
    with pytest.raises(ValidationError) as exc:
        ScenarioConfig.model_validate({
            "carrot": {"count": 1},
            "ingredients": [{"name": "carrot", "nutrients": {}}],
        })
    assert "ambiguous" in str(exc.value).lower()


def test_three_nutrients_on_ingredient_zero_rejected():
    """M8: ingredient 0 (legacy ``cfg.carrot``) is still capped at 2
    nutrients because it stores them on top-level ``cfg.nutrient`` /
    ``cfg.nutrient2``. A clean-form YAML declaring 3 nutrients on the
    first ingredient must error out and tell the user to move that
    ingredient to ``extra_ingredients`` (i.e., declare it second).
    Extras can carry any number of nutrients via ``extra_nutrients``."""
    data = _clean_form_dict()
    data["ingredients"][0]["nutrients"]["lutein"] = {"enabled": True}
    with pytest.raises(ValidationError) as exc:
        ScenarioConfig.model_validate(data)
    assert "is capped at 2" in str(exc.value)


def test_three_nutrients_on_extra_accepted():
    """M8: extras can carry any number of nutrients. Verify a 3-nutrient
    extra translates into the legacy form with two nutrients in
    primary/secondary slots and the third in ``extra_nutrients``."""
    data = _clean_form_dict()
    # The fixture's potato extra has just ``starch``; pad it out so the
    # third declared nutrient genuinely lands in ``extra_nutrients``.
    data["ingredients"][1]["nutrients"]["vitamin_b6"] = {
        "enabled": True, "C0_mg_per_kg": 30.0,
    }
    data["ingredients"][1]["nutrients"]["potassium"] = {
        "enabled": True, "C0_mg_per_kg": 4000.0,
    }
    cfg = ScenarioConfig.model_validate(data)
    extra = cfg.extra_ingredients[0]
    # First two nutrients ride in legacy primary/secondary slots; the
    # third (``potassium``) goes into ``extra_nutrients``.
    assert extra.nutrient.name == "starch"
    assert extra.nutrient2.name == "vitamin_b6"
    assert len(extra.extra_nutrients) == 1
    assert extra.extra_nutrients[0].name == "potassium"
    # all_nutrients() flattens them in order.
    assert [n.name for n in extra.all_nutrients()] == [
        "starch", "vitamin_b6", "potassium",
    ]


def test_coupling_with_unknown_nutrient_rejected():
    """A coupling referencing an ingredient.nutrient pair that wasn't
    declared in ``ingredients[]`` raises a clear error at validation time."""
    data = _clean_form_dict()
    data["couplings"][0]["protector"] = "carrot.no_such_nutrient"
    with pytest.raises(ValidationError) as exc:
        ScenarioConfig.model_validate(data)
    assert "doesn't resolve" in str(exc.value)
