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
        description="Seed radius at nucleation (10 μm is typical cavity mouth size).",
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


class ScenarioConfig(BaseModel):
    pot: PotConfig = Field(default_factory=PotConfig)
    water: WaterConfig = Field(default_factory=WaterConfig)
    carrot: CarrotConfig = Field(default_factory=CarrotConfig)
    heating: HeatingConfig = Field(default_factory=HeatingConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    boiling: BoilingConfig = Field(default_factory=BoilingConfig)
    total_time_s: float = Field(600.0, gt=0.0)
    output_every_s: float = Field(0.1, gt=0.0)

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
