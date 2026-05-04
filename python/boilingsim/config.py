"""Scenario configuration schema.

Pydantic models that validate a YAML scenario file and produce the typed
configuration passed into geometry generation and the solver pipeline.

All dimensions are SI (meters, seconds, Celsius for user-facing temperatures,
Kelvin internally downstream).
"""

from __future__ import annotations

import math
import pathlib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# Carrot tissue density in kg/m^3 (close to water; varieties range
# 1010-1080). Used to derive total_mass_g from geometry for the Config
# UI's quantity feedback.
RHO_CARROT_KG_PER_M3 = 1040.0


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
    # M4: human-readable ingredient name. Drives the 3D scene's color
    # palette ("carrot"=orange, "potato"=cream, "onion"=yellow). Legacy
    # scenarios default to "carrot" so back-compat YAML is unchanged.
    name: str = Field("carrot", min_length=1, max_length=32)
    # M4: tissue density, kg/m³. Controls the derived total_mass_g
    # readout. Default 1040 ≈ carrot. Potato is closer to 1080; onion
    # 950. Stays a per-ingredient knob so future "I'm cooking 200 g of
    # potatoes" UX is honest about mass-to-volume conversion.
    density_kg_per_m3: float = Field(1040.0, gt=0.0)
    diameter_m: float = Field(0.025, gt=0.0)
    length_m: float = Field(0.05, gt=0.0)
    # Anchor point. Semantics depend on ``axis``:
    #   axis="z" (legacy): bottom of the carrot; cylinder extends +z by length_m.
    #   axis="x" or "y":   centre of the carrot; cylinder extends ±length_m/2.
    position: tuple[float, float, float] = (0.0, 0.0, 0.03)
    # Number of identical carrots to place in the pot. count=1 keeps the
    # legacy single-carrot behaviour (used by every existing benchmark
    # scenario). count>1 triggers deterministic auto-placement around
    # ``position`` (see ``auto_place_carrots``).
    count: int = Field(1, ge=1, le=64)
    # Cylinder axis. "z" is legacy (vertical carrot, standing on its end).
    # "x" or "y" gives the realistic horizontal stew orientation.
    axis: Literal["x", "y", "z"] = "z"
    # M2: how the user specifies quantity.
    #   "dimensions" (default, legacy): user sets diameter_m + length_m;
    #                 total mass is derived for display only.
    #   "mass":        user sets target_mass_g + count + diameter_m;
    #                 length_m is derived to match the target. The Config
    #                 UI flips Length to read-only in this mode.
    mass_mode: Literal["dimensions", "mass"] = "dimensions"
    # Required when ``mass_mode == "mass"``. Total mass in grams across
    # all instances (e.g. 200 g => 4 carrots × 50 g each at count=4).
    # ``None`` is the default and is valid only with mass_mode="dimensions".
    target_mass_g: float | None = Field(default=None, gt=0.0)
    initial_beta_carotene_mg_per_100g: float = Field(8.3, ge=0.0)

    @model_validator(mode="after")
    def _derive_length_from_mass(self) -> "CarrotConfig":
        """When ``mass_mode == "mass"``, recompute ``length_m`` from
        ``target_mass_g`` so the cylinder volume matches the user's
        target. Runs before ``ScenarioConfig._carrot_fits_inside_pot``
        so the derived length is bound-checked just like a user-set one.
        """
        if self.mass_mode == "mass":
            if self.target_mass_g is None:
                raise ValueError(
                    "mass_mode='mass' requires target_mass_g to be set"
                )
            r = self.diameter_m / 2.0
            per_carrot_mass_kg = (self.target_mass_g / 1000.0) / float(self.count)
            per_carrot_volume_m3 = per_carrot_mass_kg / self.density_kg_per_m3
            derived_length_m = per_carrot_volume_m3 / (math.pi * r * r)
            if derived_length_m <= 0.0:
                raise ValueError(
                    f"derived length_m {derived_length_m:.6f} m is non-positive; "
                    "check target_mass_g, count, and diameter_m"
                )
            # ``length_m`` is a Field on a frozen model after validation, so
            # use object.__setattr__ to bypass Pydantic's immutability and
            # re-record. Pydantic v2 supports this for in-place derivation.
            object.__setattr__(self, "length_m", derived_length_m)
        return self

    def total_mass_g(self) -> float:
        """Aggregate carrot mass in grams.

        Used by the dashboard for quantity feedback: when the user adjusts
        count/diameter/length on the Config page, the UI shows the
        derived total in real time so they know "I'm cooking 200 g".
        Uses ``density_kg_per_m3`` so per-ingredient density (potato vs
        carrot vs onion) flows through correctly.
        """
        r = self.diameter_m / 2.0
        volume_m3 = self.count * math.pi * r * r * self.length_m
        return volume_m3 * self.density_kg_per_m3 * 1000.0  # kg -> g


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
    # M6: human-readable nutrient identifier. Populated automatically from
    # the dict key in the new clean YAML form (``ingredients[*].nutrients.<name>:``).
    # Used by:
    #   * Coupling resolution -- ``protector: carrot.vitamin_c`` looks up the
    #     slot whose ``name == "vitamin_c"`` inside ingredient ``carrot``.
    #   * Dashboard display labels (``_classify_nutrient`` checks this before
    #     falling back to the K_partition / C_water_sat sniffer).
    # Default empty string preserves legacy behaviour.
    name: str = Field("", max_length=64)
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


class NutrientCouplingConfig(BaseModel):
    """One protective coupling between two solute slots.

    M5: real cooking shows a vitamin-C-protects-β-carotene effect --
    AA in solution scavenges peroxyl radicals that would otherwise
    cleave the carotenoid polyene chain. While AA is present at
    typical fresh-vegetable concentrations (~50 mg/kg water), β-carotene
    degrades roughly 0.5× as fast as in AA-depleted broths (Sakai et al.,
    1987, J Food Process Preserv 11: 197; Ordóñez-Santos et al., 2020,
    Int J Food Sci Tech 55: 201; Henneberry & Reid, "Protective effect
    of ascorbic acid in degradation of β-carotene...", 2002 vol).

    Mathematical form (host-side, per step):
        protection_factor = max(1 - eta * c_protector_mean / c_ref,
                                1 - eta_max)
        k_eff = cfg.k0 * protection_factor
    where ``c_protector_mean`` is the spatially-averaged water-side
    concentration of the protector (typically ascorbic acid). ``eta`` is
    the slope of protection per unit reference concentration; ``c_ref``
    sets the scale where protection saturates; ``eta_max`` caps the
    protection at a sensible value (default 0.5 = 50 % rate reduction).

    Slots are identified by ingredient name (matches ``cfg.carrot.name``
    or any ``cfg.extra_ingredients[k].name``) plus slot index ("primary"
    for ``cfg.nutrient`` / ``extra.nutrient``, "secondary" for
    ``cfg.nutrient2`` / ``extra.nutrient2``).
    """

    enabled: bool = True
    protector_ingredient: str = Field(
        "carrot",
        description="Ingredient name carrying the protective solute (e.g. "
                    "vitamin C). Must match ``cfg.carrot.name`` or one of "
                    "``cfg.extra_ingredients[*].name``.",
    )
    protector_slot: Literal["primary", "secondary"] = Field(
        "secondary",
        description="Which solute slot on the protector ingredient. Vitamin C "
                    "is conventionally the secondary solute on a carrot in "
                    "this codebase's dual-solute scenarios.",
    )
    protected_ingredient: str = Field(
        "carrot",
        description="Ingredient name whose degradation is being slowed.",
    )
    protected_slot: Literal["primary", "secondary"] = Field(
        "primary",
        description="Which solute slot on the protected ingredient (typically "
                    "the carotenoid as primary).",
    )
    eta: float = Field(
        0.5, ge=0.0, le=100.0,
        description="Protection slope (dimensionless). At "
                    "c_protector_mean = c_ref the protection equals eta "
                    "(typically 0.5 → 50 %% rate reduction). The eta_max "
                    "cap prevents runaway even at large eta.",
    )
    c_ref_mg_per_kg: float = Field(
        50.0, gt=0.0,
        description="Reference protector concentration (mg/kg water) at which "
                    "protection reaches ``eta``. Default ~50 mg/kg matches "
                    "typical fresh-vegetable AA water-side levels after a "
                    "boil.",
    )
    eta_max: float = Field(
        0.5, ge=0.0, le=0.99,
        description="Hard cap on protection: degradation rate is never "
                    "reduced below ``(1 - eta_max) * k0``. Default 0.5 keeps "
                    "the model conservative (real fresh-vegetable broths "
                    "rarely show >50 %% rate suppression).",
    )
    # M8: optional nutrient-name overrides used when an extra ingredient's
    # nutrient sits past the legacy primary/secondary pair (i.e., in
    # ``ExtraIngredientConfig.extra_nutrients``). When non-empty, these
    # take precedence over ``protector_slot`` / ``protected_slot`` and
    # the resolver looks up the nutrient by name in the ingredient's
    # ``all_nutrients()`` list. Auto-populated by the YAML translator
    # when the dotted identifier resolves to a 3rd+ nutrient slot.
    protector_nutrient_name: str = Field(
        "", max_length=64,
        description="Nutrient name on the protector ingredient (M8 N-slot "
                    "extras). Empty -> use ``protector_slot`` literal.",
    )
    protected_nutrient_name: str = Field(
        "", max_length=64,
        description="Nutrient name on the protected ingredient (M8 N-slot "
                    "extras). Empty -> use ``protected_slot`` literal.",
    )


class ExtraIngredientConfig(CarrotConfig):
    """One additional ingredient beyond the legacy ``cfg.carrot``.

    M4: ``cfg.extra_ingredients`` is a list of these. Each carries the
    same geometry knobs as ``CarrotConfig`` plus its own primary +
    secondary nutrient profile, so a stew can have carrot + potato +
    onion with different β-carotene / starch / sulfide kinetics.

    Why a separate type instead of extending ``CarrotConfig`` itself
    with ``nutrient`` fields: the legacy single-carrot path keeps the
    top-level ``cfg.nutrient`` / ``cfg.nutrient2`` blocks for ingredient
    0 (~190 call sites read ``cfg.nutrient.*`` directly). Folding nutrients
    into ``CarrotConfig`` would force a YAML-format break we don't need.
    Extras carry their nutrients on their own Pydantic block so the
    legacy YAML keeps working unchanged.
    """

    # Override defaults sensibly for "another ingredient":
    name: str = Field("potato", min_length=1, max_length=32)
    density_kg_per_m3: float = Field(1080.0, gt=0.0)
    # By default an extra ingredient stays out of the way of the legacy
    # carrot at (0, 0, 0.03). Specify a different position in YAML.
    position: tuple[float, float, float] = (0.0, 0.030, 0.040)

    nutrient: NutrientConfig = Field(default_factory=NutrientConfig)
    nutrient2: NutrientConfig = Field(default_factory=NutrientConfig)
    # M8: nutrients beyond the legacy primary/secondary pair. The 2-slot
    # cap was a Pydantic-shape artefact, not a kernel limitation -- the
    # SoluteSlot allocator already handles arbitrary slot counts. Each
    # entry here gets its own C / C_water / scratch arrays at pipeline
    # init and pumps through the standard reaction-diffusion-leach
    # kernels just like ``nutrient`` / ``nutrient2``. Couplings reference
    # these slots by nutrient name (the ``NutrientConfig.name`` field).
    extra_nutrients: list[NutrientConfig] = Field(
        default_factory=list,
        description="Nutrients beyond the legacy ``nutrient`` / ``nutrient2`` "
                    "pair. Each gets its own SoluteSlot. Available only on "
                    "extras (ingredient 0 / cfg.carrot stays at the 2-slot cap "
                    "via the legacy top-level cfg.nutrient / cfg.nutrient2 "
                    "blocks; this asymmetry will be unified in a future Python-"
                    "tree refactor).",
    )

    def all_nutrients(self) -> list[NutrientConfig]:
        """Unified list of every nutrient slot on this extra ingredient,
        ordered: ``nutrient`` (primary), ``nutrient2`` (secondary, if
        enabled), then ``extra_nutrients[*]`` in declared order. Used by
        the slot allocator + per-ingredient diagnostic loops."""
        out: list[NutrientConfig] = []
        if self.nutrient.enabled:
            out.append(self.nutrient)
        if self.nutrient2.enabled:
            out.append(self.nutrient2)
        out.extend(n for n in self.extra_nutrients if n.enabled)
        return out

    @model_validator(mode="after")
    def _extra_nutrient2_requires_primary(self) -> "ExtraIngredientConfig":
        if self.nutrient2.enabled and not self.nutrient.enabled:
            raise ValueError(
                f"extra_ingredient '{self.name}': nutrient2.enabled=True "
                "requires nutrient.enabled=True"
            )
        # extra_nutrients beyond the pair also require the primary to be on,
        # for the same "primary anchors the ingredient's solute physics"
        # contract. (Internally each extra slot is independent, but the
        # API contract is symmetric.)
        if self.extra_nutrients and not self.nutrient.enabled:
            raise ValueError(
                f"extra_ingredient '{self.name}': extra_nutrients require "
                "nutrient.enabled=True"
            )
        return self


def _translate_clean_yaml_to_legacy(data: dict) -> dict:
    """M6: translate the clean unified YAML form into the legacy
    internal representation.

    Clean form (user-facing):
        ingredients:
          - name: carrot
            count: 3
            axis: x
            ...
            nutrients:
              beta_carotene: { enabled: true, ... }
              vitamin_c:     { enabled: true, ... }
          - name: potato
            ...
        couplings:
          - protector: carrot.vitamin_c
            protected: carrot.beta_carotene
            ...

    Legacy form (internal, what every Pydantic model + 130+ Python
    call sites already expect):
        carrot: { ...geometry... }
        nutrient:  { name: beta_carotene, ... }
        nutrient2: { name: vitamin_c, ... }
        extra_ingredients:
          - { name: potato, ..., nutrient: {...}, nutrient2: {...} }
        nutrient_couplings:
          - protector_ingredient: carrot
            protector_slot: secondary
            protected_ingredient: carrot
            protected_slot: primary
            ...

    The translation runs once at YAML load time; downstream code never
    sees the clean form.

    Constraints (raise ``ValueError`` on violation, surfaces as a
    Pydantic ValidationError to the user):
      * At least one ingredient.
      * Max 2 ``nutrients`` per ingredient (current internal cap).
      * No collision with the legacy form (``carrot``, ``nutrient``,
        ``nutrient2``, ``extra_ingredients``, ``nutrient_couplings``
        keys must NOT be present alongside ``ingredients`` /
        ``couplings``).
      * Coupling dotted identifiers (``ingredient_name.nutrient_name``)
        must resolve to a declared (ingredient, nutrient) pair.
    """
    if not isinstance(data, dict):
        return data
    if "ingredients" not in data:
        return data  # legacy form -- pass through unchanged

    # Reject hybrids: presence of any legacy carrot/nutrient block
    # alongside the new ingredients[] form is ambiguous.
    forbidden_legacy_keys = (
        "carrot", "nutrient", "nutrient2",
        "extra_ingredients", "nutrient_couplings",
    )
    collisions = [k for k in forbidden_legacy_keys if k in data]
    if collisions:
        raise ValueError(
            "ScenarioConfig schema is ambiguous: the clean ``ingredients`` "
            f"form cannot be combined with legacy keys {collisions!r}. "
            "Use one form or the other."
        )

    # Working copy so we don't mutate the caller's dict.
    out = {k: v for k, v in data.items() if k != "ingredients" and k != "couplings"}
    raw_ingredients = data["ingredients"]
    raw_couplings = data.get("couplings", [])

    if not isinstance(raw_ingredients, list) or len(raw_ingredients) == 0:
        raise ValueError(
            "ScenarioConfig.ingredients must be a non-empty list "
            "(at least one ingredient required)."
        )

    # Build a per-ingredient nutrient-name → slot ("primary"/"secondary")
    # lookup so coupling identifiers can be resolved below. Also do the
    # 2-nutrient cap enforcement here.
    nutrient_lookup: list[tuple[str, dict[str, str]]] = []  # [(ing_name, {nut_name: slot})]

    def _split_one_ingredient(
        raw_ing: dict, idx: int, allow_extra: bool
    ) -> tuple[dict, dict, dict, list[dict]]:
        """Return (geometry_dict, primary_nut_dict, secondary_nut_dict,
        extra_nut_dicts).

        ``allow_extra`` controls whether nutrients past the second key
        spill into ``extra_nut_dicts`` (True for extras / M8) or trigger
        a ValidationError (False for ingredient 0 / legacy carrot, which
        is still capped at 2 by the top-level ``cfg.nutrient`` /
        ``cfg.nutrient2`` Pydantic shape).
        """
        if not isinstance(raw_ing, dict):
            raise ValueError(
                f"ingredients[{idx}] must be a dict; got {type(raw_ing).__name__}"
            )
        # Pop the nutrients key out of the geometry dict.
        nutrients = raw_ing.get("nutrients", {})
        if not isinstance(nutrients, dict):
            raise ValueError(
                f"ingredients[{idx}].nutrients must be a dict keyed by "
                "nutrient name (e.g. ``beta_carotene:`` or ``vitamin_c:``); "
                f"got {type(nutrients).__name__}"
            )
        if not allow_extra and len(nutrients) > 2:
            raise ValueError(
                f"ingredients[{idx}].nutrients has {len(nutrients)} entries; "
                "ingredient 0 (the legacy ``cfg.carrot``) is capped at 2 "
                "tracked nutrients (primary + secondary) by the top-level "
                "``cfg.nutrient`` / ``cfg.nutrient2`` Pydantic shape. Move "
                "this ingredient to ``extra_ingredients[]`` (i.e., declare it "
                "second in the list) to use more than 2 nutrients."
            )
        # Geometry is everything else.
        geom = {k: v for k, v in raw_ing.items() if k != "nutrients"}
        nut_keys = list(nutrients.keys())
        primary_dict: dict = {}
        secondary_dict: dict = {}
        extra_nut_dicts: list[dict] = []
        slot_for_name: dict[str, str] = {}
        if len(nut_keys) >= 1:
            n0_name = nut_keys[0]
            primary_dict = dict(nutrients[n0_name]) if isinstance(nutrients[n0_name], dict) else {}
            primary_dict.setdefault("name", n0_name)
            slot_for_name[n0_name] = "primary"
        if len(nut_keys) >= 2:
            n1_name = nut_keys[1]
            secondary_dict = dict(nutrients[n1_name]) if isinstance(nutrients[n1_name], dict) else {}
            secondary_dict.setdefault("name", n1_name)
            slot_for_name[n1_name] = "secondary"
        # M8: nutrients beyond #2 land in extra_nutrients (extras only).
        for nk_idx in range(2, len(nut_keys)):
            nk_name = nut_keys[nk_idx]
            d = dict(nutrients[nk_name]) if isinstance(nutrients[nk_name], dict) else {}
            d.setdefault("name", nk_name)
            extra_nut_dicts.append(d)
            slot_for_name[nk_name] = nk_name   # name-based slot id (not "primary"/"secondary")
        ing_name = geom.get("name", "carrot" if idx == 0 else f"ingredient_{idx}")
        nutrient_lookup.append((ing_name, slot_for_name))
        return geom, primary_dict, secondary_dict, extra_nut_dicts

    # Ingredient 0 → top-level carrot + nutrient + nutrient2 (capped at 2).
    geom0, primary0, secondary0, extra_nuts0 = _split_one_ingredient(
        raw_ingredients[0], 0, allow_extra=False,
    )
    assert not extra_nuts0  # 2-cap enforced above for idx=0
    out["carrot"] = geom0
    if primary0:
        out["nutrient"] = primary0
    if secondary0:
        out["nutrient2"] = secondary0

    # Ingredients 1..N → extra_ingredients[]. Each extra carries its
    # geometry inline plus its own nutrient / nutrient2 / extra_nutrients.
    extras: list[dict] = []
    for idx in range(1, len(raw_ingredients)):
        geom_i, primary_i, secondary_i, extra_nuts_i = _split_one_ingredient(
            raw_ingredients[idx], idx, allow_extra=True,
        )
        extra_dict = dict(geom_i)
        if primary_i:
            extra_dict["nutrient"] = primary_i
        if secondary_i:
            extra_dict["nutrient2"] = secondary_i
        if extra_nuts_i:
            extra_dict["extra_nutrients"] = extra_nuts_i
        extras.append(extra_dict)
    if extras:
        out["extra_ingredients"] = extras

    # Resolve couplings.
    if raw_couplings:
        if not isinstance(raw_couplings, list):
            raise ValueError(
                "ScenarioConfig.couplings must be a list of coupling dicts."
            )
        translated: list[dict] = []
        for ci, cc in enumerate(raw_couplings):
            if not isinstance(cc, dict):
                raise ValueError(f"couplings[{ci}] must be a dict.")
            translated.append(_resolve_coupling_identifiers(cc, ci, nutrient_lookup))
        out["nutrient_couplings"] = translated

    return out


def _resolve_coupling_identifiers(
    cc: dict,
    ci: int,
    nutrient_lookup: list[tuple[str, dict[str, str]]],
) -> dict:
    """Translate a clean-form coupling dict (with ``protector`` /
    ``protected`` dotted identifiers) into the legacy form
    (``protector_ingredient`` + ``protector_slot``, etc.).

    Accepts both forms in the same coupling dict: if the legacy keys
    are already present, they pass through unchanged.
    """
    out = dict(cc)
    for kind in ("protector", "protected"):
        ident = out.pop(kind, None)
        if ident is None:
            continue
        if not isinstance(ident, str) or "." not in ident:
            raise ValueError(
                f"couplings[{ci}].{kind} = {ident!r} must be a dotted "
                "identifier of the form ``ingredient_name.nutrient_name``."
            )
        ing_name, nut_name = ident.split(".", 1)
        # Look up the slot in the nutrient_lookup table built during
        # ingredient translation.
        slot = None
        for (lookup_ing, slot_for_name) in nutrient_lookup:
            if lookup_ing == ing_name:
                slot = slot_for_name.get(nut_name)
                break
        if slot is None:
            raise ValueError(
                f"couplings[{ci}].{kind} = {ident!r} doesn't resolve to any "
                "declared (ingredient, nutrient) pair in ``ingredients``."
            )
        out[f"{kind}_ingredient"] = ing_name
        # If the resolver mapped this nutrient to a name (M8 N-slot
        # extras), record it explicitly and pin the legacy slot literal
        # to "primary" as a placeholder (it'll be ignored at runtime
        # when the name field is non-empty).
        if slot in ("primary", "secondary"):
            out[f"{kind}_slot"] = slot
        else:
            out[f"{kind}_nutrient_name"] = slot
            out[f"{kind}_slot"] = "primary"   # placeholder; runtime uses the name
    return out


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
    # M4: ingredients beyond the legacy carrot (ingredient 0). Each
    # extra carries its own geometry + primary + secondary nutrient
    # profile so a stew can have carrot + potato + onion with different
    # kinetics. Order matters: the first extra becomes ingredient 1
    # (instance_id = 2), the second becomes ingredient 2, etc. Default
    # empty -- legacy single-carrot scenarios load unchanged.
    extra_ingredients: list[ExtraIngredientConfig] = Field(
        default_factory=list,
        description="Additional ingredients beyond cfg.carrot (ingredient 0). "
                    "Each carries its own geometry + nutrient profile.",
    )
    # M5: nutrient-nutrient saturation / antioxidant interactions.
    nutrient_couplings: list[NutrientCouplingConfig] = Field(
        default_factory=list,
        description="Protective couplings between solute slots. Empty list "
                    "(default) leaves every slot's degradation kinetics "
                    "independent. See NutrientCouplingConfig for the model.",
    )
    total_time_s: float = Field(600.0, gt=0.0)
    output_every_s: float = Field(0.1, gt=0.0)

    def iter_ingredients(self) -> "list[tuple[CarrotConfig, NutrientConfig, NutrientConfig]]":
        """Yield (geometry, nutrient, nutrient2) tuples in voxel-id order.

        Ingredient 0 is the legacy ``cfg.carrot`` paired with the
        top-level ``cfg.nutrient`` / ``cfg.nutrient2``. Ingredients 1..N
        are pulled from ``cfg.extra_ingredients`` in declared order.
        Unified iteration so kernels can dispatch per slot without
        knowing whether they're walking ingredient 0 or N.
        """
        out: list[tuple[CarrotConfig, NutrientConfig, NutrientConfig]] = [
            (self.carrot, self.nutrient, self.nutrient2),
        ]
        for extra in self.extra_ingredients:
            out.append((extra, extra.nutrient, extra.nutrient2))
        return out

    @property
    def n_ingredients(self) -> int:
        """Number of ingredients including the legacy carrot."""
        return 1 + len(self.extra_ingredients)

    @model_validator(mode="before")
    @classmethod
    def _accept_clean_yaml_form(cls, data: Any) -> Any:
        """M6: front door for the clean unified YAML form.

        When a YAML file uses the new ``ingredients[]`` + ``couplings[]``
        layout, this pre-validator translates it into the legacy
        ``carrot`` + ``nutrient`` + ``nutrient2`` + ``extra_ingredients``
        + ``nutrient_couplings`` shape that the rest of the codebase
        expects. Files using the legacy form pass through unchanged.
        """
        if isinstance(data, dict):
            return _translate_clean_yaml_to_legacy(data)
        return data

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
        water_top_z = self.pot.base_thickness_m + water_height

        # Validate ingredient 0 (legacy carrot) and every extra ingredient
        # against the same fits-in-pot rule. Each ingredient's
        # ``count`` instances are auto-placed first, then every centre
        # is bounds-checked against the inner pot cylinder + water line.
        for ing_idx, (geom, _nut, _nut2) in enumerate(self.iter_ingredients()):
            _check_ingredient_fits_in_pot(
                ing_idx=ing_idx,
                geom=geom,
                inner_radius=inner_radius,
                base_thickness=self.pot.base_thickness_m,
                water_top_z=water_top_z,
            )
        return self


def _check_ingredient_fits_in_pot(
    ing_idx: int,
    geom: CarrotConfig,
    inner_radius: float,
    base_thickness: float,
    water_top_z: float,
) -> None:
    """Bounds-check every auto-placed instance of one ingredient.

    Factored out of ``ScenarioConfig._carrot_fits_inside_pot`` so the
    legacy carrot (ingredient 0) and every entry in
    ``cfg.extra_ingredients`` use exactly the same rules.
    """
    carrot_r = geom.diameter_m / 2.0
    carrot_L = geom.length_m
    axis = geom.axis

    centres = auto_place_carrots(
        count=geom.count,
        axis=axis,
        anchor=geom.position,
        diameter_m=geom.diameter_m,
        length_m=carrot_L,
        inner_radius=inner_radius,
        base_thickness=base_thickness,
        water_top_z=water_top_z,
    )

    name_label = geom.name if ing_idx > 0 else "carrot"
    for inst_idx, (cx, cy, cz) in enumerate(centres):
        tag = f"{name_label}[{inst_idx}]" if geom.count > 1 else name_label
        if axis == "z":
            if (cx ** 2 + cy ** 2) ** 0.5 + carrot_r > inner_radius:
                raise ValueError(f"{tag} center + radius exceeds pot inner radius")
            if cz < base_thickness:
                raise ValueError(f"{tag} sits below the pot base")
            if cz + carrot_L > water_top_z:
                raise ValueError(f"{tag} top extends above the water line")
        elif axis == "x":
            far_x = abs(cx) + carrot_L / 2.0
            far_y = abs(cy) + carrot_r
            if (far_x ** 2 + far_y ** 2) ** 0.5 > inner_radius:
                raise ValueError(f"{tag} (axis=x) sweeps outside pot inner radius")
            if cz - carrot_r < base_thickness:
                raise ValueError(f"{tag} bottom sits below the pot base")
            if cz + carrot_r > water_top_z:
                raise ValueError(f"{tag} top extends above the water line")
        else:  # axis == "y"
            far_x = abs(cx) + carrot_r
            far_y = abs(cy) + carrot_L / 2.0
            if (far_x ** 2 + far_y ** 2) ** 0.5 > inner_radius:
                raise ValueError(f"{tag} (axis=y) sweeps outside pot inner radius")
            if cz - carrot_r < base_thickness:
                raise ValueError(f"{tag} bottom sits below the pot base")
            if cz + carrot_r > water_top_z:
                raise ValueError(f"{tag} top extends above the water line")


def auto_place_carrots(
    count: int,
    axis: Literal["x", "y", "z"],
    anchor: tuple[float, float, float],
    diameter_m: float,
    length_m: float,
    inner_radius: float,
    base_thickness: float,
    water_top_z: float,
) -> list[tuple[float, float, float]]:
    """Compute deterministic centres for ``count`` identical carrots.

    ``anchor`` is the user-supplied ``cfg.carrot.position``: the base
    point for axis="z" (legacy single-carrot semantics) or the centroid
    of the placement region for axis="x" / "y".

    Strategy:
      * count==1: return [anchor] unchanged. Preserves every existing
        single-carrot scenario without needing YAML edits.
      * axis=="z" (vertical): distribute centres on a circumferential
        ring around (anchor.x, anchor.y) at z=anchor.z, sized to fit
        inside the pot.
      * axis in {"x", "y"} (horizontal): stack centres along the
        perpendicular horizontal axis at z=anchor.z, with 1.2x carrot
        diameter spacing between centres so the cylinders don't
        intersect.

    Raises ``ValueError`` when count is too large to fit.
    """
    if count <= 1:
        return [anchor]
    cx0, cy0, cz0 = anchor
    r = diameter_m / 2.0
    if axis == "z":
        # Ring placement around the anchor.
        ring_r = inner_radius - r * 1.5
        if ring_r < r:
            raise ValueError(
                f"pot too narrow for {count} vertical carrots "
                f"(need ring radius > {r:.3f} m, have {ring_r:.3f} m)"
            )
        return [
            (
                cx0 + ring_r * math.cos(2.0 * math.pi * i / count),
                cy0 + ring_r * math.sin(2.0 * math.pi * i / count),
                cz0,
            )
            for i in range(count)
        ]
    # Horizontal: stack along the perpendicular horizontal axis.
    # spacing = 1.2 * diameter between centres (20% gap).
    spacing = 1.2 * diameter_m
    half_span = (count - 1) * spacing / 2.0
    centres: list[tuple[float, float, float]] = []
    for i in range(count):
        offset = i * spacing - half_span
        if axis == "x":
            cx, cy = cx0, cy0 + offset
        else:  # axis == "y"
            cx, cy = cx0 + offset, cy0
        centres.append((cx, cy, cz0))
    return centres


def load_scenario(path: str | pathlib.Path) -> ScenarioConfig:
    """Load + validate a YAML scenario file."""
    path = pathlib.Path(path)
    data = yaml.safe_load(path.read_text())
    return ScenarioConfig.model_validate(data)
