"""Scenario configuration schema.

Pydantic models that validate a YAML scenario file and produce the typed
configuration passed into geometry generation and the solver pipeline.

All dimensions are SI (meters, seconds, Celsius for user-facing temperatures,
Kelvin internally downstream).
"""

from __future__ import annotations

import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


MaterialName = Literal["steel_304", "cast_iron", "aluminum", "copper"]


class PotConfig(BaseModel):
    diameter_m: float = Field(0.20, gt=0.0, description="Outer diameter of the pot (m)")
    height_m: float = Field(0.12, gt=0.0, description="Outer height of the pot (m)")
    wall_thickness_m: float = Field(0.003, gt=0.0)
    base_thickness_m: float = Field(0.005, gt=0.0)
    material: MaterialName = "steel_304"

    @model_validator(mode="after")
    def _wall_thinner_than_radius(self) -> "PotConfig":
        if self.wall_thickness_m >= self.diameter_m / 2:
            raise ValueError("wall_thickness_m must be less than pot radius")
        if self.base_thickness_m >= self.height_m:
            raise ValueError("base_thickness_m must be less than pot height")
        return self


class WaterConfig(BaseModel):
    fill_fraction: float = Field(0.75, gt=0.0, lt=1.0)
    initial_temp_c: float = Field(20.0, ge=0.0, le=100.0)


class CarrotConfig(BaseModel):
    diameter_m: float = Field(0.025, gt=0.0)
    length_m: float = Field(0.05, gt=0.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.03)
    initial_beta_carotene_mg_per_100g: float = Field(8.3, ge=0.0)


class HeatingConfig(BaseModel):
    base_heat_flux_w_per_m2: float = Field(30000.0, ge=0.0)
    ambient_temp_c: float = Field(22.0, ge=-50.0, le=100.0)


class InitialConditionsConfig(BaseModel):
    """How the grid temperature field is seeded at t=0.

    ``cold`` (default) honours ``water.initial_temp_c`` end-to-end: water,
    pot wall and carrot all start at the configured water temperature
    (the carrot is assumed to have equilibrated with the water at load
    time; pot wall equilibrates too since the stove hasn't fired yet).
    Air stays at ``heating.ambient_temp_c``.

    ``preheat`` overrides the grid T field after construction with the
    ``preheat_*_c`` values. Intended for benchmark scripts that want to
    skip the 5-10 min warming transient and start the interesting physics
    (nucleate boiling, nutrient degradation) immediately. Phase-3 and
    Phase-4 benchmarks use this path with the historical 95 / 100 / 20 °C
    defaults.
    """

    mode: Literal["cold", "preheat"] = "cold"
    preheat_water_c: float = Field(95.0, ge=0.0, le=105.0)
    preheat_wall_c: float = Field(100.0, ge=0.0, le=120.0)
    preheat_carrot_c: float = Field(20.0, ge=0.0, le=100.0)


class GridConfig(BaseModel):
    dx_m: float = Field(0.001, gt=0.0)
    carrot_mesh_resolution: int = Field(40, gt=0)


class SolverConfig(BaseModel):
    """Phase 2 CFD + thermal solver parameters."""

    cfl_safety_factor: float = Field(0.4, gt=0.0, le=0.5)
    max_dt_s: float = Field(0.1, gt=0.0, description="Hard cap on Δt regardless of CFL")
    pressure_tol: float = Field(1e-5, gt=0.0)
    pressure_max_iter: int = Field(200, gt=0)
    diffusion_tol: float = Field(1e-4, gt=0.0)
    diffusion_max_iter: int = Field(15, gt=0)
    h_conv_outer_w_per_m2_k: float = Field(10.0, ge=0.0, description="Newton cooling coefficient on outer pot wall")
    h_evap_free_surface_w_per_m2_k: float = Field(
        5.0e4, ge=0.0,
        description="Open-pot free-surface enthalpy-bleed coefficient. Fires only when "
                    "cfg.boiling.enabled and a fluid cell is adjacent to air above. "
                    "A real boiling pot is latent-heat-pinned at T_sat by vapour exit; "
                    "our sealed domain needs this bookkeeping term to reproduce that. "
                    "Default 5e4 W/m^2/K pins the free-surface row to T_sat + ~0.1 K "
                    "at the 30 kW/m^2 stove default. Deeper fluid still drifts via "
                    "bulk-to-surface transport lag -- see ``f_bulk_evap_per_s`` for "
                    "the matching volumetric closure.",
    )
    f_bulk_evap_per_s: float = Field(
        1.0, ge=0.0,
        description="Bulk-boiling closure for the sealed computational domain. Every "
                    "fluid cell with T > T_sat sees its superheat decay at rate "
                    "``f_bulk_evap_per_s`` [1/s] -- a lumped model for the bulk "
                    "nucleation pathway that the wall-anchored bubble pool does not "
                    "capture (real water above saturation inside the column flashes "
                    "to steam throughout its volume, not just at the fluid-air "
                    "interface). Per-cell update: dT_remove = f*(T-T_sat)*dt, clamped "
                    "to (T-T_sat) so a cell cannot be driven subcooled. Fires only "
                    "when cfg.boiling.enabled. Default 1.0 /s (~1 s e-folding) matches "
                    "the order of magnitude of real pot thermal response; set 0 to "
                    "disable and recover the surface-only sink. Values above ~5 /s "
                    "start damping legitimate buoyancy plumes.",
    )
    use_implicit_conduction: bool = Field(
        True,
        description="Backward-Euler Jacobi for thermal conduction. Unconditionally "
                    "stable so Δt is only bounded by advection CFL (not α_solid).",
    )


class BoilingConfig(BaseModel):
    """Phase 3 nucleate-boiling parameters.

    When ``enabled=False`` the simulation falls back to the Phase-2
    placeholder evaporative-cooling kernel (temperature-gated). When True,
    a Lagrangian bubble pool is allocated and the latent-heat sink, vapor
    momentum back-reaction, and VOF α reduction all kick in.
    """

    enabled: bool = Field(
        False,
        description="Master switch. Off in default.yaml for backwards compat with Phase-2 validations.",
    )
    dT_onb_k: float = Field(
        5.0, gt=0.0,
        description="Wall superheat (T_wall − T_sat) threshold for onset of nucleate boiling.",
    )
    contact_angle_rad: float = Field(
        1.0, gt=0.0, le=3.14159,
        description="Bubble-wall contact angle (water on steel ≈ 0.7–1.4 rad).",
    )
    max_bubbles: int = Field(
        100_000, gt=0,
        description="Lagrangian particle pool size. 100k for dev, 1M for production.",
    )
    initial_bubble_radius_m: float = Field(
        1.0e-5, gt=0.0,
        description="Seed radius at nucleation (10 um is typical cavity mouth size).",
    )
    max_bubble_radius_m: float = Field(
        5.0e-3, gt=0.0,
        description=(
            "Safety floor on bubble radius. Mikic-Rohsenow growth is "
            "monotonic in age and unbounded; without a cap, a bubble that "
            "gets stuck in a stagnation zone keeps inflating until it vents. "
            "5 mm is the published Rayleigh-Taylor fragmentation threshold "
            "for water at 1 atm (Levich 1962, Clift-Grace-Weber 1978). "
            "With fragmentation enabled (fragmentation_radius_m below the "
            "cap), this cap should rarely fire."
        ),
    )
    fragmentation_radius_m: float = Field(
        4.0e-3, gt=0.0,
        description=(
            "Bubble radius at which a Rayleigh-Taylor breakup event splits "
            "the bubble into two equal-volume daughters (R_d = R / 2^(1/3) "
            "= R * 0.794). Set below max_bubble_radius_m so the split fires "
            "before the safety cap clamps. Real bubbles in water at 1 atm "
            "fragment in the 5-7 mm range; 4 mm is conservative."
        ),
    )
    coalescence_enabled: bool = Field(
        True,
        description=(
            "Master switch for the spatial-hash coalescence pass. When two "
            "bubble centres come within R1+R2 they merge into a single "
            "volume-conserving bubble (R = (R1^3+R2^3)^(1/3), momentum-"
            "weighted velocity). Costs ~3 extra kernel launches per step; "
            "expected to be < 5 % of step time at typical pool sizes."
        ),
    )
    coalescence_bin_size_m: float = Field(
        12.0e-3, gt=0.0,
        description=(
            "Spatial-hash bin edge length for the coalescence pass. Should "
            "be at least 2 * max_bubble_radius_m so that any pair of "
            "overlapping bubbles is in the same bin or in immediate "
            "neighbours. Default 12 mm = 2.4x the 5 mm cap."
        ),
    )
    coalescence_max_per_bin: int = Field(
        64, gt=0,
        description=(
            "Per-bin capacity for the spatial-hash bubble lookup table. "
            "If a bin overflows (more bubbles than this in one bin) the "
            "extras simply skip coalescence detection that step -- they "
            "get another shot next step. With 12 mm bins and 1.5 mm "
            "bubbles, max physical packing is ~30; 64 leaves comfortable "
            "headroom."
        ),
    )
    nucleation_probability_per_step: float = Field(
        0.1, gt=0.0, le=1.0,
        description="Scales Cole frequency f to convert site-active rate into per-step spawn probability.",
    )
    C_sf_rohsenow: float = Field(
        0.013, gt=0.0,
        description="Rohsenow surface-fluid coefficient. 0.013 for water on stainless steel.",
    )
    Pr_n_rohsenow: float = Field(
        1.0, gt=0.0,
        description="Rohsenow Prandtl exponent n. Use 1.0 for water.",
    )


class NutrientConfig(BaseModel):
    """Phase 4 beta-carotene reaction-diffusion-leaching parameters.

    When ``enabled=False`` the simulation skips all nutrient physics. When
    True, a voxelised concentration field ``C`` is allocated on carrot cells
    (initial value ``C0_mg_per_kg``) and evolved by Arrhenius degradation
    (Milestone A), molecular diffusion (Milestone B), and Sherwood-correlation
    surface leaching into a water-side passive scalar ``C_water`` (Milestone C).

    Defaults match dev-guide sec.4 / data/materials.json:49-67 for beta-carotene
    in carrot:
        E_a = 70 kJ/mol      — activation energy
        k0  = 2.63e6 /s      — pre-exponential factor
        D_eff = 2e-10 m^2/s  — effective diffusivity in carrot tissue
        K_partition = 0.3    — carrot/water equilibrium ratio
        C0  = 83 mg/kg       — initial beta-carotene loading
    """

    enabled: bool = Field(
        False,
        description="Master switch. Off in default.yaml so Phase-0/1/2/3 regression tests unaffected.",
    )
    E_a_kJ_per_mol: float = Field(
        70.0, gt=0.0,
        description="Arrhenius activation energy for beta-carotene thermal degradation.",
    )
    k0_per_s: float = Field(
        2.63e6, gt=0.0,
        description="Arrhenius pre-exponential factor.",
    )
    D_eff_m2_per_s: float = Field(
        2.0e-10, gt=0.0,
        description="Effective diffusivity of beta-carotene in carrot tissue.",
    )
    K_partition: float = Field(
        1.0e-5, gt=0.0,
        description="Equilibrium partition coefficient: C_water / C_carrot at equilibrium. "
                    "For bare beta-carotene in pure water this is 1e-4 to 1e-6 depending on "
                    "temperature and tissue state (Treszczanowicz et al. 1998 measured carotene "
                    "distribution between organic solvent and water in this range). The prior "
                    "default of 0.007 modelled a 'moderately lipophilic' carotenoid ester, not "
                    "bare beta-carotene; at our pot's 107:1 water:carrot volume ratio that value "
                    "allowed ~75 %% of the carrot to dissolve before equilibrium, which blew "
                    "past the dev-guide [80, 90] %% retention band. 1e-5 puts equilibrium "
                    "C_water at K*C0 ~ 8e-4 mg/kg so leaching self-throttles at <1 %% of C0 "
                    "and retention is correctly dominated by Arrhenius degradation. For water-"
                    "soluble vitamins (C, folate) override to 0.5-2.0; for a carrot-in-oil "
                    "emulsion, raise toward 0.007 (oil phase carries the carotene).",
    )
    C_water_sat_mg_per_kg: float = Field(
        6.0e-3, gt=0.0,
        description="Absolute saturation concentration of the solute in water (mg per kg "
                    "water). Hard cap on the water-side concentration that the leaching "
                    "kernel will produce, regardless of partition coefficient or driving "
                    "force. Default 6e-3 mg/kg matches beta-carotene aqueous solubility at "
                    "~100 C (order ~6 ug/L; Craft & Soares 1992 report ~0.6 ug/L at 20 C, "
                    "scaled up an order of magnitude for boiling). The prior default of 0.6 "
                    "mg/kg was 100x too high -- that is the 20 C *micro*gram figure misread "
                    "as milligrams. Set very high (e.g. 1e6) for water-soluble nutrients "
                    "where solubility never limits.",
    )
    C0_mg_per_kg: float = Field(
        83.0, ge=0.0,
        description="Initial beta-carotene concentration in carrot cells (mg per kg carrot tissue).",
    )
    # --- Milestone C (Sherwood leaching) water-side transport properties ---
    nu_water_m2_per_s: float = Field(
        2.94e-7, gt=0.0,
        description="Kinematic viscosity of water at ~100 C, for Reynolds number.",
    )
    D_water_molec_m2_per_s: float = Field(
        1.0e-9, gt=0.0,
        description="Molecular diffusivity of beta-carotene in water, for Schmidt number "
                    "and the h_m = Sh * D / L conversion.",
    )


class ScenarioConfig(BaseModel):
    pot: PotConfig = Field(default_factory=PotConfig)
    water: WaterConfig = Field(default_factory=WaterConfig)
    carrot: CarrotConfig = Field(default_factory=CarrotConfig)
    heating: HeatingConfig = Field(default_factory=HeatingConfig)
    initial_conditions: InitialConditionsConfig = Field(
        default_factory=InitialConditionsConfig,
    )
    grid: GridConfig = Field(default_factory=GridConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    boiling: BoilingConfig = Field(default_factory=BoilingConfig)
    nutrient: NutrientConfig = Field(default_factory=NutrientConfig)
    nutrient2: NutrientConfig = Field(
        default_factory=NutrientConfig,
        description="Optional second solute, evolved concurrently in the same "
                    "boiling domain. Disabled by default; when enabled the "
                    "primary nutrient must also be enabled. Used for the "
                    "dual-solute validation (beta-carotene + vitamin C in the "
                    "same pot).",
    )
    total_time_s: float = Field(600.0, gt=0.0)
    output_every_s: float = Field(0.1, gt=0.0)

    @model_validator(mode="after")
    def _nutrient2_requires_primary(self) -> "ScenarioConfig":
        if self.nutrient2.enabled and not self.nutrient.enabled:
            raise ValueError(
                "nutrient2.enabled=True requires nutrient.enabled=True "
                "(primary solute must be active before a secondary is added)"
            )
        return self

    @model_validator(mode="after")
    def _carrot_fits_inside_pot(self) -> "ScenarioConfig":
        inner_radius = self.pot.diameter_m / 2 - self.pot.wall_thickness_m
        water_height = self.water.fill_fraction * (self.pot.height_m - self.pot.base_thickness_m)

        cx, cy, cz = self.carrot.position
        carrot_r = self.carrot.diameter_m / 2
        if (cx ** 2 + cy ** 2) ** 0.5 + carrot_r > inner_radius:
            raise ValueError("carrot center + radius exceeds pot inner radius")
        if cz < self.pot.base_thickness_m:
            raise ValueError("carrot sits below the pot base")
        if cz + self.carrot.length_m > self.pot.base_thickness_m + water_height:
            raise ValueError("carrot top extends above the water line")
        return self


def load_scenario(path: str | pathlib.Path) -> ScenarioConfig:
    """Load + validate a YAML scenario file."""
    path = pathlib.Path(path)
    data = yaml.safe_load(path.read_text())
    return ScenarioConfig.model_validate(data)
