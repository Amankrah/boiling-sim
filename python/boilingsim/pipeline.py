"""Phase 2 coupled simulation pipeline.

Wires the Milestone-C fluid solver and Milestone-B thermal solver together
per dev-guide §2.2 into a :class:`Simulation` driver. Writes HDF5 output
with a scalar time series every step and downsampled full-field snapshots
on a slower cadence.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .config import ScenarioConfig
from .fluid import (
    FluidWorkspace,
    advect_all,
    allocate_fluid_workspace,
    apply_buoyancy_step,
    compute_max_velocity,
    enforce_no_slip,
    pressure_projection,
)
from .geometry import MAT_CARROT, MAT_FLUID, MAT_POT_WALL, Grid, build_pot_geometry
from .thermal import (
    MaterialProps,
    ThermalWorkspace,
    allocate_thermal_workspace,
    compute_max_dt_conduction,
    conduct_one_step,
)


# ---------------------------------------------------------------------------
# Simulation state container
# ---------------------------------------------------------------------------


@dataclass
class ScalarSample:
    t: float
    dt: float
    # Sim step counter at the moment this sample was captured. Independent
    # of `t` because dt varies per step (advection-CFL bound). Drivers like
    # run_dashboard.py and the run-summary writer use this for the STEPS
    # field on the Results page; offline HDF5 readers can use it to align
    # samples with kernel-launch indices for performance analysis.
    step: int
    T_mean_water_c: float
    T_max_water_c: float
    T_min_water_c: float
    T_max_wall_c: float
    T_inner_wall_mean_c: float   # Fluid-contact face avg (Rohsenow-relevant metric)
    T_inner_wall_max_c: float    # Hottest cell on the fluid-contact face
    u_max_mps: float
    # Phase-3 Milestone E diagnostics (zero when boiling disabled)
    n_active_bubbles: int = 0
    mean_bubble_R_mm: float = 0.0
    mean_departed_bubble_R_mm: float = 0.0  # mean R of detached (site_cleared == 1) active bubbles
    max_bubble_R_mm: float = 0.0
    alpha_min: float = 1.0                   # min water_alpha anywhere (0 = bubble-saturated)
    # Phase-4 Milestone A diagnostic (100 = no degradation, 0 = complete loss)
    retention_pct: float = 100.0
    # Phase-4 instrumentation: mass-partition diagnostic. The four channels
    # sum to 100 %% when physics is correct. degraded_pct is *signed*: a
    # sudden negative value indicates mass creation (numerical artefact),
    # a sudden positive spike after a stable trajectory indicates mass
    # destruction. Do NOT clamp to [0, 100] -- that hides bugs, which
    # happened in the earlier non-conservative SL advection run.
    leached_pct: float = 0.0
    degraded_pct: float = 0.0
    # Cumulative mass clipped out of C_water by the post-advection
    # saturation clamp (physical solubility cap). Normally near zero; grows
    # when pressure-projection residual ∇·u concentrates mass at stagnation
    # cells past C_water_sat. See clamp_c_water_and_track_precipitation.
    precipitated_pct: float = 0.0
    # Phase-4 dual-solute extension: mass partition for the *second* solute
    # evolved concurrently in the same domain. Defaults (100/0/0/0) match the
    # single-solute case so HDF5 traces from nutrient2-disabled runs are
    # unchanged in meaning.
    retention2_pct: float = 100.0
    leached2_pct: float = 0.0
    degraded2_pct: float = 0.0
    precipitated2_pct: float = 0.0
    # M3 per-instance retention diagnostics. Empty when nutrient is
    # disabled or carrot.count==1 (no point in a length-1 vector when
    # the aggregate scalar already covers it). Length == cfg.carrot.count
    # otherwise; entry i is the percent of carrot[i]'s initial mass
    # still inside its voxel mask. Per-instance leached/degraded/
    # precipitated are deferred -- they require attribution physics
    # (tracking which carrot a dissolved molecule came from), and the
    # aggregate scalars above already characterise the mass-balance
    # picture across all instances. Per-instance *retention* is the
    # diagnostic users actually want for "is the carrot near the wall
    # cooked through?".
    retention_per_instance: list[float] = field(default_factory=list)
    retention2_per_instance: list[float] = field(default_factory=list)
    # M4 per-ingredient retention. One entry per ingredient (cfg.carrot
    # at index 0; cfg.extra_ingredients[k-1] at index k). Length =
    # cfg.n_ingredients when populated; empty when nutrient is disabled
    # or only one ingredient is present (the aggregate scalars cover
    # the legacy single-ingredient case). Same shared-nutrient-profile
    # caveat as the per-instance vector: every ingredient leaches into
    # the same C/C_water field today; per-ingredient nutrient kinetics
    # is a future M4-extended.
    retention_per_ingredient: list[float] = field(default_factory=list)
    retention2_per_ingredient: list[float] = field(default_factory=list)
    # M4 ingredient names (parallel to retention_per_ingredient). Used by
    # the dashboard to label per-ingredient lines / 3D colors. Length =
    # cfg.n_ingredients when populated.
    ingredient_names: list[str] = field(default_factory=list)


class Simulation:
    """Coupled CFD + thermal pipeline for a boiling-sim scenario."""

    def __init__(self, cfg: ScenarioConfig, device: str = "cuda:0") -> None:
        self.cfg = cfg
        self.device = device

        self.grid: Grid = build_pot_geometry(cfg, device=device)
        self.props = MaterialProps.from_scenario(cfg, device=device)
        self.ws_fluid: FluidWorkspace = allocate_fluid_workspace(self.grid, device=device)
        self.ws_thermal: ThermalWorkspace = allocate_thermal_workspace(self.grid, device=device)
        # Phase-4 Milestone B: ping-pong buffer for explicit in-carrot diffusion.
        # When ``cfg.nutrient2.enabled`` the workspace also carries the second
        # solute's independent scratch (C_work2 / C_water_tmp2) and atomic
        # counter (precipitated_mass2), so a single ``ws_nutrient`` handle
        # covers both slots.
        self.ws_nutrient = None
        self.primary_slot = None
        self.secondary_slot = None
        # M4-extended: each extra ingredient with an enabled nutrient
        # gets its own SoluteSlot. ``self.extra_slots`` is a list of
        # per-ingredient slot lists: ``self.extra_slots[k]`` is the
        # list of SoluteSlot for ``cfg.extra_ingredients[k]`` in slot
        # order (primary, secondary if enabled, then ``extra_nutrients``
        # in declared order). M8: variable-length, not capped at 2.
        self.extra_slots: list[list[Any]] = []
        if cfg.nutrient.enabled and self.grid.C is not None:
            from .nutrient import (
                allocate_nutrient_workspace,
                make_primary_slot,
                make_secondary_slot,
            )
            self.ws_nutrient = allocate_nutrient_workspace(
                self.grid, device=device,
                alloc_secondary=cfg.nutrient2.enabled,
            )
            self.primary_slot = make_primary_slot(
                self.grid, cfg, self.ws_nutrient,
            )
            if cfg.nutrient2.enabled and self.grid.C2 is not None:
                self.secondary_slot = make_secondary_slot(
                    self.grid, cfg, self.ws_nutrient,
                )

        # M4-extended: allocate per-extra-ingredient solute slots. Each
        # extra with ``nutrient.enabled`` gets its own C / C_water /
        # workspace / precipitated_mass arrays, with kernels gated on
        # ``ingredient_id == k+1`` so they only modify their own
        # ingredient's voxels and dump dissolved mass into their own
        # water field.
        if (
            self.grid.ingredient_id is not None
            and len(cfg.extra_ingredients) > 0
        ):
            from .nutrient import _allocate_ingredient_slot
            for ext_idx, extra in enumerate(cfg.extra_ingredients):
                ingredient_idx = ext_idx + 1   # 0 is legacy carrot
                slots = _allocate_ingredient_slot(
                    self.grid, cfg, extra, ingredient_idx, device=device,
                )
                self.extra_slots.append(slots)

        # M5: resolve nutrient-nutrient coupling specs into pairs of
        # actual SoluteSlot references. Each entry is (coupling_cfg,
        # protector_slot, protected_slot).
        #
        # Resolution rules:
        #   * Coupling with ``enabled: false`` -- silently skipped.
        #   * Coupling with a typo in the *ingredient* name -- HARD ERROR.
        #     Catches misspellings like ``carrott.vitamin_c``; never the
        #     fault of a disabled nutrient since the ingredient itself is
        #     declared in cfg.
        #   * Coupling whose ingredient exists but whose slot is empty
        #     (the named nutrient has ``enabled: false``) -- silently
        #     skipped. This lets the dashboard's default.yaml ship a
        #     coupling block alongside disabled nutrients without
        #     crashing; users opt into the coupling by enabling both
        #     nutrients on the participating ingredients.
        self._resolved_couplings: list[tuple[Any, Any, Any]] = []
        for cc in cfg.nutrient_couplings:
            if not cc.enabled:
                continue
            # First check: does the named ingredient exist at all?
            ingredient_names = {self.cfg.carrot.name} | {
                e.name for e in self.cfg.extra_ingredients
            }
            for kind, ing in (
                ("protector", cc.protector_ingredient),
                ("protected", cc.protected_ingredient),
            ):
                if ing not in ingredient_names:
                    raise ValueError(
                        f"nutrient_coupling {kind} references unknown "
                        f"ingredient {ing!r}; declared ingredients are "
                        f"{sorted(ingredient_names)}"
                    )
            # M8: when the coupling carries an explicit nutrient name
            # (3rd+ slot on an extra ingredient), the resolver matches
            # by name; otherwise it falls back to the primary/secondary
            # literal.
            protector = self._resolve_slot(
                cc.protector_ingredient, cc.protector_slot,
                nutrient_name=cc.protector_nutrient_name,
            )
            protected = self._resolve_slot(
                cc.protected_ingredient, cc.protected_slot,
                nutrient_name=cc.protected_nutrient_name,
            )
            if protector is None or protected is None:
                # Targets exist as ingredients but their nutrients are
                # disabled -- skip this coupling, no harm done.
                continue
            self._resolved_couplings.append((cc, protector, protected))

        # Water-specific constants (Phase 2 uses constant properties).
        self.rho_water = float(self.props.rho[MAT_FLUID])
        self.beta_water = 2.07e-4  # 1/K near 25 °C (water)
        self.T_ref_k = cfg.water.initial_temp_c + 273.15

        # Precompute host-side masks for diagnostics (avoid GPU→CPU roundtrip each step).
        self._mat_host = self.grid.mat.numpy()
        self._water_mask = self._mat_host == MAT_FLUID
        self._wall_mask = self._mat_host == MAT_POT_WALL
        self._carrot_mask = self._mat_host == MAT_CARROT
        self._n_carrot = int(self._carrot_mask.sum())
        # M3: per-instance carrot masks for retention diagnostics.
        # ``self._instance_id_host`` mirrors ``grid.instance_id`` on host
        # (cells outside carrots are 0; carrot cells carry c+1 within
        # whichever ingredient claimed the cell -- M4 makes instance_id
        # per-ingredient, not global).
        # M4: ``self._ingredient_id_host`` carries the 1-based ingredient
        # label (0 outside, k+1 for ingredient k). Combined as
        # ``(ingredient_id == k+1) & (instance_id == c+1)`` to disambiguate
        # "carrot 0 instance 0" from "potato 0 instance 0".
        if self.grid.instance_id is not None:
            self._instance_id_host = self.grid.instance_id.numpy()
        else:
            self._instance_id_host = None
        if self.grid.ingredient_id is not None:
            self._ingredient_id_host = self.grid.ingredient_id.numpy()
        else:
            self._ingredient_id_host = None

        # M4: per-ingredient masks. ``_per_ingredient_masks[k]`` is the
        # boolean voxel mask for ingredient k as a whole (all instances
        # combined). Length == cfg.n_ingredients. Used by per-ingredient
        # retention reductions in sample_scalars.
        self._per_ingredient_masks: list[np.ndarray] = []
        self._n_per_ingredient: list[int] = []
        if self._ingredient_id_host is not None:
            for k in range(cfg.n_ingredients):
                m = self._ingredient_id_host == (k + 1)
                self._per_ingredient_masks.append(m)
                self._n_per_ingredient.append(int(m.sum()))

        # M3: per-instance masks for ingredient 0 ONLY (legacy carrot).
        # Multi-ingredient scenarios get per-ingredient retention from
        # ``_per_ingredient_masks`` instead. Per-(ingredient, instance)
        # diagnostics are deferred -- the count×n_ingredients matrix
        # explodes wire-format size and "is this carrot cooked?" is
        # already answered by per-ingredient totals at typical count<=8.
        if self._instance_id_host is not None and self._ingredient_id_host is not None:
            ing0_mask = self._ingredient_id_host == 1
            n_carrots = int(cfg.carrot.count)
            self._instance_masks = [
                ing0_mask & (self._instance_id_host == (c + 1))
                for c in range(n_carrots)
            ]
            self._n_per_instance = [int(m.sum()) for m in self._instance_masks]
        else:
            self._instance_masks = []
            self._n_per_instance = []
        # Inner-wall (fluid-contact-face) mask: pot-wall cells whose +z neighbor
        # is fluid. This is the Rohsenow-relevant boiling surface. For low-k
        # pot materials the heater face is several K hotter than this face due
        # to the q*L/k drop across the solid, so T_wall_max (all pot-wall cells)
        # over-reports the boiling superheat -- see phase3_boiling.md.
        mh = self._mat_host
        inner = np.zeros_like(self._wall_mask)
        inner[:, :, :-1] = (mh[:, :, :-1] == MAT_POT_WALL) & (mh[:, :, 1:] == MAT_FLUID)
        self._inner_wall_mask = inner

        self.t: float = 0.0
        self.step_count: int = 0

        # Per-phase profiling. Opt-in via env var so production runs pay
        # zero overhead; when enabled, each step phase is wrapped in a
        # ``wp.synchronize_device`` pair so per-kernel cost is measured
        # without overlap. The synchronize pairs add ~5–10 % to step
        # wall time -- documented and consistent across runs.
        self._profile_enabled: bool = bool(
            int(os.environ.get("BOILINGSIM_PROFILE", "0") or "0")
        )
        self._profile_acc: dict[str, float] = {}
        self._profile_n: int = 0

        # M2: u_max readback caching for compute_dt. The host sync at
        # ``ws.u_max_scalar.numpy()[0]`` was 14 % of step time in the M1
        # profile. Caching for K>1 steps WAS theoretically safe (cfl=0.4
        # gives 2.5x headroom) but in live dual-solute dashboard runs
        # transient u_max bursts at bubble departure can violate CFL
        # under K=8: C_water advection oscillates (negative cells trip
        # the clamp kernel and inflate precipitated_pct), bubble
        # positions overshoot outside the pot, and scatter_latent_heat
        # over-subtracts T to <-200 C. Default K=1 (no caching) is the
        # safe fallback; opt in via ``BOILINGSIM_DT_REFRESH=N`` only on
        # workloads with smooth u_max trajectories.
        self._dt_refresh_every: int = max(
            1, int(os.environ.get("BOILINGSIM_DT_REFRESH", "1") or "1")
        )
        self._cached_u_max: float = 0.0
        # Force refresh on the first compute_dt call.
        self._steps_since_dt_refresh: int = self._dt_refresh_every

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _profile_phase(self, name: str):
        """Context manager that times a step phase when profiling is on.

        When ``BOILINGSIM_PROFILE`` is unset, this is a no-op (single
        Python ``yield``); the only cost is the ``with`` statement
        itself, ~0.5 µs per phase per step. When set, it brackets the
        phase with ``wp.synchronize_device`` so the measured elapsed
        time captures the actual GPU work, not just kernel-launch
        overhead.
        """
        if not self._profile_enabled:
            yield
            return
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            wp.synchronize_device(self.device)
            self._profile_acc[name] = (
                self._profile_acc.get(name, 0.0)
                + (time.perf_counter() - t0)
            )

    def reset_profile(self) -> None:
        """Clear accumulated phase timings (does not toggle the flag)."""
        self._profile_acc.clear()
        self._profile_n = 0

    def profile_summary(self) -> list[tuple[str, float, float, float]]:
        """Return per-phase timings sorted by total cost (descending).

        Each row is ``(name, total_s, mean_ms_per_step, frac_pct_of_total)``.
        Empty list when profiling is disabled or no steps have run.
        """
        if not self._profile_enabled or self._profile_n == 0:
            return []
        total = sum(self._profile_acc.values())
        rows: list[tuple[str, float, float, float]] = []
        for name, t in sorted(self._profile_acc.items(), key=lambda kv: -kv[1]):
            mean_ms = 1000.0 * t / self._profile_n
            frac_pct = 100.0 * t / total if total > 0 else 0.0
            rows.append((name, t, mean_ms, frac_pct))
        return rows

    def compute_dt(self) -> float:
        """Return a stable timestep from active stability constraints.

        With ``use_implicit_conduction=True`` (default) the thermal-diffusion
        limit is dropped — BE is unconditionally stable — so Δt is bounded
        by advection CFL and the user-set ``max_dt_s``. When nutrient physics
        is enabled the explicit in-carrot diffusion adds ``dx^2/(6*D_eff)``
        as an additional upper bound; at the dev-grid defaults this is
        ~6700 s and never binds, but a very fine dx can reach it, at which
        point we clamp here instead of letting ``step_diffuse_nutrient``
        raise.

        u_max is refreshed every ``self._dt_refresh_every`` steps to skip
        the host-sync readback on most steps (M2). The cached value is
        used in between; the cfl_safety_factor headroom absorbs the
        small mismatch.
        """
        if self._steps_since_dt_refresh >= self._dt_refresh_every:
            self._cached_u_max = compute_max_velocity(self.grid, ws=self.ws_fluid)
            self._steps_since_dt_refresh = 1
        else:
            self._steps_since_dt_refresh += 1
        u_max = self._cached_u_max
        dt_cfl = self.grid.dx / max(u_max, 1.0e-8)
        dt_cap = self.cfg.solver.max_dt_s / self.cfg.solver.cfl_safety_factor
        if self.cfg.solver.use_implicit_conduction:
            dt = min(dt_cfl, dt_cap)
        else:
            dt_thermal = compute_max_dt_conduction(self.props, self.grid.dx, safety=1.0)
            dt = min(dt_thermal, dt_cfl, dt_cap)
        if self.cfg.nutrient.enabled:
            from .nutrient import diffusion_stability_dt, _diffusion_stability_dt_D
            dt = min(dt, diffusion_stability_dt(self.cfg, self.grid.dx))
            if self.cfg.nutrient2.enabled:
                dt = min(dt, _diffusion_stability_dt_D(
                    self.grid.dx, self.cfg.nutrient2.D_eff_m2_per_s))
        return self.cfg.solver.cfl_safety_factor * dt

    def _resolve_slot(
        self,
        ingredient_name: str,
        slot_name: str,
        nutrient_name: str = "",
    ) -> Any:
        """Look up a SoluteSlot by ingredient name + slot identity.

        ``nutrient_name`` (M8) takes precedence over ``slot_name`` when
        non-empty: the resolver searches the ingredient's slot list
        for one whose ``cfg_nutrient.name`` matches. This handles
        couplings that target a 3rd+ nutrient on an extra ingredient
        (beyond the legacy primary/secondary pair). When empty, falls
        back to the literal ``slot_name`` ("primary"/"secondary").

        Returns ``None`` if the slot doesn't exist (nutrient disabled,
        name typo, etc.); the caller decides whether to error or skip.
        """
        # Ingredient 0 (legacy carrot) lives on (primary_slot, secondary_slot).
        if ingredient_name == self.cfg.carrot.name:
            if nutrient_name:
                # Match by nutrient name on the legacy carrot's two slots.
                if (self.primary_slot is not None
                        and self.primary_slot.cfg_nutrient.name == nutrient_name):
                    return self.primary_slot
                if (self.secondary_slot is not None
                        and self.secondary_slot.cfg_nutrient.name == nutrient_name):
                    return self.secondary_slot
                return None
            if slot_name == "primary":
                return self.primary_slot
            return self.secondary_slot
        # Extras: search by name.
        for ext_idx, extra in enumerate(self.cfg.extra_ingredients):
            if extra.name != ingredient_name:
                continue
            slots = self.extra_slots[ext_idx]
            if nutrient_name:
                for s in slots:
                    if s.cfg_nutrient.name == nutrient_name:
                        return s
                return None
            # Slot literal -> index in the slot list. Slot 0 is "primary",
            # slot 1 is "secondary"; couplings with M8-style nutrient_name
            # don't reach this branch.
            idx = 0 if slot_name == "primary" else 1
            if idx < len(slots):
                return slots[idx]
            return None
        return None

    def _apply_couplings(self) -> None:
        """M5: refresh ``slot.rate_multiplier`` for every protected slot
        from the current state of its protectors' water-side
        concentrations.

        Called once at the top of ``step()`` before the reaction-
        diffusion-leach kernels. Multiple couplings on the same
        protected slot multiply (independent scavengers); a single
        coupling factor is bounded below by ``1 - eta_max``.
        """
        # Reset every slot's multiplier so a coupling that *was* active
        # last step but is disabled this step doesn't carry stale state.
        all_slots: list[Any] = [self.primary_slot, self.secondary_slot]
        for slots_for_extra in self.extra_slots:
            all_slots.extend(slots_for_extra)
        for s in all_slots:
            if s is not None:
                s.rate_multiplier = 1.0
        if not self._resolved_couplings:
            return
        # Compute mean water-side protector concentrations once per
        # protector slot (the same protector may apply to multiple
        # protected slots).
        protector_means: dict[int, float] = {}
        for _cc, protector, _protected in self._resolved_couplings:
            key = id(protector)
            if key in protector_means:
                continue
            if protector.C_water is None or not self._water_mask.any():
                protector_means[key] = 0.0
                continue
            cw = protector.C_water.numpy()
            protector_means[key] = float(cw[self._water_mask].mean())
        # Apply each coupling. Multiplicative composition lets two
        # couplings on the same protected slot stack (e.g. AA + tocopherol).
        for cc, protector, protected in self._resolved_couplings:
            c_mean = protector_means[id(protector)]
            factor = 1.0 - cc.eta * (c_mean / cc.c_ref_mg_per_kg)
            factor = max(factor, 1.0 - cc.eta_max)
            protected.rate_multiplier *= factor

    def step(self, dt: float | None = None) -> float:
        """Advance the simulation by one step. Returns the dt used."""
        if dt is None:
            with self._profile_phase("compute_dt"):
                dt = self.compute_dt()

        # 1-2. Semi-Lagrangian advection of velocity and temperature.
        with self._profile_phase("advect_all"):
            advect_all(self.grid, self.ws_fluid, dt, device=self.device)

        # 3. Boussinesq buoyancy on z-faces.
        with self._profile_phase("buoyancy"):
            apply_buoyancy_step(
                self.grid, self.cfg, dt,
                beta=self.beta_water, T_ref_k=self.T_ref_k, device=self.device,
            )

        # 3b. Phase-3 Milestone B: advance bubbles (growth + departure + rise + vent),
        #     then detect new nucleation. Decoupled from fluid for this milestone —
        #     momentum + latent-heat feedback lands in Milestones C and D.
        if self.cfg.boiling.enabled and self.grid.bubbles is not None:
            from .boiling import step_bubbles
            with self._profile_phase("step_bubbles"):
                step_bubbles(
                    self.grid, self.grid.bubbles, self.cfg, dt,
                    sim_time=self.t, step_count=self.step_count, device=self.device,
                )

        # 4. Conjugate heat diffusion + all boundary sources (stove, Newton, evap).
        with self._profile_phase("conduct_one_step"):
            conduct_one_step(self.grid, self.props, self.ws_thermal, self.cfg, dt, device=self.device)

        # 4b. Phase-3: Eulerian wall boiling flux (microlayer evaporation).
        #     Directly cools pot-wall cells at nucleation sites, proportional to
        #     local superheat. This is the dominant wall-cooling mechanism that
        #     the Lagrangian scatter alone cannot provide (it acts on mid-fluid).
        if self.cfg.boiling.enabled and self.grid.bubbles is not None:
            from .boiling import step_wall_boiling_flux
            with self._profile_phase("wall_boiling_flux"):
                step_wall_boiling_flux(
                    self.grid, self.grid.bubbles, self.cfg, self.props, dt,
                    device=self.device,
                )

        # 4c. Phase-4 Milestones A+B+C: Arrhenius degradation + in-carrot
        #     diffusion + Sherwood-flux surface leaching into the water-side
        #     passive scalar C_water. Order: degrade (reaction), then diffuse
        #     (internal transport with zero-flux BC), then leach (opens the
        #     surface, flux is driven by C_carrot - C_water/K_partition, scales
        #     with h_m = Sh*D_water/D_carrot at the current face velocity).
        if self.cfg.nutrient.enabled and self.grid.C is not None:
            assert self.ws_nutrient is not None and self.primary_slot is not None, (
                "nutrient.enabled with grid.C allocated but ws_nutrient / "
                "primary_slot is None: diffusion would be silently skipped. "
                "Check Simulation.__init__."
            )
            from .nutrient import _step_reaction_diffusion_leach
            D_carrot = self.cfg.carrot.diameter_m
            # M5: refresh per-slot rate_multiplier from the current
            # water-side protector concentrations BEFORE launching the
            # degrade kernels. No-op when cfg.nutrient_couplings is empty.
            with self._profile_phase("nutrient_couplings"):
                self._apply_couplings()
            with self._profile_phase("nutrient_react_diff_leach"):
                _step_reaction_diffusion_leach(
                    self.primary_slot, self.grid, D_carrot, dt, device=self.device,
                )
                if self.secondary_slot is not None:
                    _step_reaction_diffusion_leach(
                        self.secondary_slot, self.grid, D_carrot, dt, device=self.device,
                    )
                # M4-extended + M8: each extra ingredient's slots pump
                # independently through the standard reaction-diffusion-
                # leach sequence. Slot count per ingredient is variable
                # (M8 ``extra_nutrients`` allows >2).
                for slots_for_extra in self.extra_slots:
                    for slot in slots_for_extra:
                        _step_reaction_diffusion_leach(
                            slot, self.grid, slot.D_carrot, dt,
                            device=self.device,
                        )

        # 5. No-slip on solid faces before projection.
        with self._profile_phase("no_slip_pre"):
            enforce_no_slip(self.grid, device=self.device)

        # 6. Pressure projection — enforces ∇·u = 0 in fluid.
        with self._profile_phase("pressure_projection"):
            pressure_projection(
                self.grid, self.ws_fluid, self.cfg, dt,
                rho=self.rho_water, device=self.device,
            )

        # 7. Re-enforce no-slip (pressure subtraction doesn't touch solid faces,
        #    but this guards against drift from numerical error).
        with self._profile_phase("no_slip_post"):
            enforce_no_slip(self.grid, device=self.device)

        # 8. Phase-4 Milestone D: advect the water-side beta-carotene passive
        #    scalar using the freshly projected (divergence-free) velocity
        #    field, then clamp to the physical solubility cap. The clamp is
        #    necessary because our pressure_projection runs to tol=1e-5, not
        #    machine precision, so there is a small residual ∇·u that breaks
        #    the upwind scheme's monotone property. Over ~90k steps this
        #    accumulates at stagnation cells and produces 10-300x cap
        #    excursions (bulk fluid mean stays correct, but peak pixels
        #    violate physical solubility). Excess mass is credited to
        #    precipitated_pct so the mass-balance triple still closes at
        #    100 %%.
        if (self.cfg.nutrient.enabled
                and self.grid.C_water is not None
                and self.ws_nutrient is not None
                and self.primary_slot is not None):
            from .nutrient import _step_advect_clamp
            with self._profile_phase("nutrient_advect_clamp"):
                _step_advect_clamp(self.primary_slot, self.grid, dt, device=self.device)
                if self.secondary_slot is not None:
                    _step_advect_clamp(self.secondary_slot, self.grid, dt, device=self.device)
                # M4-extended + M8: every slot on every extra advects + clamps.
                for slots_for_extra in self.extra_slots:
                    for slot in slots_for_extra:
                        _step_advect_clamp(slot, self.grid, dt, device=self.device)

        self.t += dt
        self.step_count += 1
        if self._profile_enabled:
            self._profile_n += 1
        return dt

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sample_scalars(self, dt_last: float) -> ScalarSample:
        """Capture mean/min/max temperatures, peak velocity, and bubble stats."""
        T = self.grid.T.numpy()
        T_w = T[self._water_mask]
        T_wall = T[self._wall_mask] if self._wall_mask.any() else T
        if self._inner_wall_mask.any():
            T_inner = T[self._inner_wall_mask]
            T_inner_mean_c = float(T_inner.mean() - 273.15)
            T_inner_max_c = float(T_inner.max() - 273.15)
        else:
            T_inner_mean_c = float(T_wall.max() - 273.15)
            T_inner_max_c = float(T_wall.max() - 273.15)
        u_max = compute_max_velocity(self.grid, ws=self.ws_fluid)

        # Phase-3 bubble diagnostics. Phase-8-Refactor-1: compact readback
        # via :func:`read_active_bubbles` -- DMAs ~24 bytes/slot of typed
        # float arrays instead of the 60-byte/slot Bubble struct, and
        # avoids the host-side boolean mask over ``max_bubbles``.
        n_active = 0
        mean_R_mm = 0.0
        mean_departed_R_mm = 0.0
        max_R_mm = 0.0
        alpha_min = 1.0
        if self.grid.bubbles is not None:
            from .boiling import read_active_bubbles
            view = read_active_bubbles(self.grid.bubbles)
            n_active = view.n_active
            if n_active > 0:
                mean_R_mm = float(view.radii.mean() * 1000.0)
                max_R_mm = float(view.radii.max() * 1000.0)
                if view.site_cleared.any():
                    # Use the frozen departure_radius so this reports the
                    # Fritz-departure population mean, not post-departure growth.
                    R_dep = view.departure_radii[view.site_cleared]
                    mean_departed_R_mm = float(R_dep.mean() * 1000.0)
            if self.grid.water_alpha_base is not None:
                alpha_min = float(self.grid.water_alpha.numpy()[self._water_mask].min())

        # Phase-4 Milestone A: retention fraction from carrot concentration field.
        # Plus mass-conservation breakdown: still-in-carrot vs leached vs
        # degraded vs precipitated. Inlined here (rather than calling the
        # public retention_fraction / water_pool_fraction / precipitated_
        # fraction helpers) so we reuse the cached self._carrot_mask and
        # self._n_carrot instead of doing three extra grid.mat.numpy()
        # roundtrips per sample.
        retention_pct = 100.0
        leached_pct = 0.0
        degraded_pct = 0.0
        precipitated_pct = 0.0
        if self.grid.C is not None and self._n_carrot > 0:
            C0 = self.cfg.nutrient.C0_mg_per_kg
            ref_mass = C0 * float(self._n_carrot)
            C_np = self.grid.C.numpy()
            retention_pct = 100.0 * float(C_np[self._carrot_mask].sum()) / ref_mass
            if self.grid.C_water is not None:
                Cw_np = self.grid.C_water.numpy()
                leached_pct = 100.0 * float(Cw_np.sum()) / ref_mass
            if self.ws_nutrient is not None:
                precipitated_pct = (
                    100.0
                    * float(self.ws_nutrient.precipitated_mass.numpy()[0])
                    / ref_mass
                )
            # Signed residual -- do NOT clamp. Oscillation or a negative
            # value means the advection scheme is creating/destroying mass
            # (numerical artefact); we want that visible, not hidden.
            degraded_pct = (
                100.0 - retention_pct - leached_pct - precipitated_pct
            )

        # --- Dual-solute: secondary mass partition ---
        # Same invariant as primary (sum to ~100 every sample), computed
        # independently from grid.C2 / grid.C_water2 / precipitated_mass2
        # against cfg.nutrient2.C0_mg_per_kg. Defaults (100/0/0/0) carry
        # through when nutrient2 is disabled, leaving HDF5 back-compatible
        # for single-solute traces at the cost of four extra always-emitted
        # datasets.
        retention2_pct = 100.0
        leached2_pct = 0.0
        degraded2_pct = 0.0
        precipitated2_pct = 0.0
        if self.grid.C2 is not None and self._n_carrot > 0:
            C0_2 = self.cfg.nutrient2.C0_mg_per_kg
            ref_mass2 = C0_2 * float(self._n_carrot)
            C2_np = self.grid.C2.numpy()
            retention2_pct = 100.0 * float(C2_np[self._carrot_mask].sum()) / ref_mass2
            if self.grid.C_water2 is not None:
                Cw2_np = self.grid.C_water2.numpy()
                leached2_pct = 100.0 * float(Cw2_np.sum()) / ref_mass2
            if self.ws_nutrient is not None and self.ws_nutrient.precipitated_mass2 is not None:
                precipitated2_pct = (
                    100.0
                    * float(self.ws_nutrient.precipitated_mass2.numpy()[0])
                    / ref_mass2
                )
            degraded2_pct = (
                100.0 - retention2_pct - leached2_pct - precipitated2_pct
            )

        # M4 per-ingredient retention. Compute only when there's more
        # than one ingredient (single-ingredient scenarios are covered
        # by the aggregate scalars above).
        # M4-extended: each extra ingredient has its OWN C field via
        # self.extra_slots; ingredient 0 still reads grid.C. Per-
        # ingredient C0 comes from each ingredient's own NutrientConfig
        # so retention is normalised against the right starting mass.
        retention_per_ingredient: list[float] = []
        retention2_per_ingredient: list[float] = []
        ingredient_names: list[str] = []
        if (
            self.cfg.n_ingredients > 1
            and self.grid.C is not None
            and len(self._per_ingredient_masks) == self.cfg.n_ingredients
        ):
            for k, (mask, n_cells) in enumerate(
                zip(self._per_ingredient_masks, self._n_per_ingredient, strict=True)
            ):
                if k == 0:
                    ingredient_names.append(self.cfg.carrot.name)
                    C_np = self.grid.C.numpy()
                    C0 = self.cfg.nutrient.C0_mg_per_kg
                    C2_np = self.grid.C2.numpy() if self.grid.C2 is not None else None
                    C0_2 = self.cfg.nutrient2.C0_mg_per_kg if C2_np is not None else 0.0
                else:
                    extra = self.cfg.extra_ingredients[k - 1]
                    ingredient_names.append(extra.name)
                    slots_for_extra = self.extra_slots[k - 1]
                    # M4-extended/M8: slots_for_extra is variable-length.
                    # Per-ingredient retention diagnostics still report
                    # only the first two slots (primary + secondary)
                    # since the wire format's per-ingredient retention
                    # vector is a 2-tuple. M8 nutrients beyond #2 stay
                    # invisible at the ingredient-level diagnostic but
                    # still leach + degrade kernel-side.
                    primary = slots_for_extra[0] if len(slots_for_extra) >= 1 else None
                    secondary = slots_for_extra[1] if len(slots_for_extra) >= 2 else None
                    if primary is None:
                        # Extra has no enabled nutrient -- fall back to
                        # the shared cfg.nutrient pool for the diagnostic.
                        C_np = self.grid.C.numpy()
                        C0 = self.cfg.nutrient.C0_mg_per_kg
                    else:
                        C_np = primary.C.numpy()
                        C0 = primary.cfg_nutrient.C0_mg_per_kg
                    if secondary is not None:
                        C2_np = secondary.C.numpy()
                        C0_2 = secondary.cfg_nutrient.C0_mg_per_kg
                    else:
                        C2_np = None
                        C0_2 = 0.0

                if n_cells == 0 or C0 <= 0.0:
                    retention_per_ingredient.append(100.0)
                else:
                    ref = C0 * float(n_cells)
                    retention_per_ingredient.append(
                        100.0 * float(C_np[mask].sum()) / ref
                    )
                if C2_np is not None and C0_2 > 0.0 and n_cells > 0:
                    ref2 = C0_2 * float(n_cells)
                    retention2_per_ingredient.append(
                        100.0 * float(C2_np[mask].sum()) / ref2
                    )

        # M3: per-instance retention (within ingredient 0 only). Skip
        # when count==1 (aggregate scalar covers it).
        retention_per_instance: list[float] = []
        retention2_per_instance: list[float] = []
        n_carrots = len(self._instance_masks)
        if n_carrots > 1 and self.grid.C is not None:
            C0 = self.cfg.nutrient.C0_mg_per_kg
            C_np = self.grid.C.numpy()
            for c, (mask, n_cells) in enumerate(
                zip(self._instance_masks, self._n_per_instance, strict=True)
            ):
                if n_cells == 0:
                    retention_per_instance.append(100.0)
                    continue
                ref = C0 * float(n_cells)
                if ref > 0.0:
                    retention_per_instance.append(
                        100.0 * float(C_np[mask].sum()) / ref
                    )
                else:
                    retention_per_instance.append(100.0)
            if self.grid.C2 is not None:
                C0_2 = self.cfg.nutrient2.C0_mg_per_kg
                C2_np = self.grid.C2.numpy()
                for mask, n_cells in zip(
                    self._instance_masks, self._n_per_instance, strict=True
                ):
                    if n_cells == 0:
                        retention2_per_instance.append(100.0)
                        continue
                    ref = C0_2 * float(n_cells)
                    if ref > 0.0:
                        retention2_per_instance.append(
                            100.0 * float(C2_np[mask].sum()) / ref
                        )
                    else:
                        retention2_per_instance.append(100.0)

        return ScalarSample(
            t=self.t,
            dt=dt_last,
            step=self.step_count,
            T_mean_water_c=float(T_w.mean() - 273.15),
            T_max_water_c=float(T_w.max() - 273.15),
            T_min_water_c=float(T_w.min() - 273.15),
            T_max_wall_c=float(T_wall.max() - 273.15),
            T_inner_wall_mean_c=T_inner_mean_c,
            T_inner_wall_max_c=T_inner_max_c,
            u_max_mps=float(u_max),
            n_active_bubbles=n_active,
            mean_bubble_R_mm=mean_R_mm,
            mean_departed_bubble_R_mm=mean_departed_R_mm,
            max_bubble_R_mm=max_R_mm,
            alpha_min=alpha_min,
            retention_pct=retention_pct,
            leached_pct=leached_pct,
            degraded_pct=degraded_pct,
            precipitated_pct=precipitated_pct,
            retention2_pct=retention2_pct,
            leached2_pct=leached2_pct,
            degraded2_pct=degraded2_pct,
            precipitated2_pct=precipitated2_pct,
            retention_per_instance=retention_per_instance,
            retention2_per_instance=retention2_per_instance,
            retention_per_ingredient=retention_per_ingredient,
            retention2_per_ingredient=retention2_per_ingredient,
            ingredient_names=ingredient_names,
        )

    # ------------------------------------------------------------------
    # Run loop with HDF5 logging
    # ------------------------------------------------------------------

    def run(
        self,
        total_time_s: float,
        out_path: pathlib.Path | None = None,
        scalar_every_n_steps: int = 20,
        snapshot_every_s: float = 60.0,
        progress_every_s: float = 10.0,
    ) -> list[ScalarSample]:
        """Time-integrate up to ``total_time_s`` and return the scalar trace.

        If ``out_path`` is given, writes HDF5 with:
          * ``scalars/*``        — per-sample arrays (t, dt, T_*, u_max)
          * ``snapshots/NN/T``   — full 3-D T field (float32) every snapshot
          * ``meta``             — grid dims, dx, material counts
        """
        import h5py  # local import keeps module import cheap

        scalars: list[ScalarSample] = []
        last_progress = -1e9
        last_snapshot_t = -1e9
        snapshots_T: list[np.ndarray] = []
        snapshot_times: list[float] = []
        bubble_radii_snaps: list[np.ndarray] = []    # radii at each snapshot
        bubble_positions_snaps: list[np.ndarray] = []  # (N, 3) at each snapshot

        wall_t0 = time.perf_counter()

        while self.t < total_time_s:
            dt = self.step()

            if self.step_count % scalar_every_n_steps == 0 or self.t >= total_time_s:
                scalars.append(self.sample_scalars(dt))

            # Full-field + bubble snapshot cadence
            if self.t - last_snapshot_t >= snapshot_every_s:
                wp.synchronize_device(self.device)
                snapshots_T.append(self.grid.T.numpy().astype(np.float32))
                snapshot_times.append(self.t)
                if self.grid.bubbles is not None:
                    from .boiling import read_active_bubbles
                    view = read_active_bubbles(self.grid.bubbles)
                    # Departure-diameter histogram: restrict to bubbles that
                    # have actually detached from a wall site and use the
                    # frozen ``departure_radius`` rather than the live
                    # ``radius`` (which keeps growing during rise). This
                    # eliminates the near-zero spike caused by counting
                    # in-flight infant bubbles as if they were departures.
                    detached = view.site_cleared
                    bubble_radii_snaps.append(
                        view.departure_radii[detached].astype(np.float32)
                    )
                    bubble_positions_snaps.append(
                        view.positions[detached].astype(np.float32)
                    )
                last_snapshot_t = self.t

            if self.t - last_progress >= progress_every_s and scalars:
                s = scalars[-1]
                wall = time.perf_counter() - wall_t0
                extra = ""
                if s.n_active_bubbles > 0:
                    extra = (f"  bubbles={s.n_active_bubbles:,}  "
                             f"R_mean={s.mean_bubble_R_mm:.2f}mm  "
                             f"alpha_min={s.alpha_min:.3f}")
                if self.cfg.nutrient.enabled:
                    extra += (f"  R={s.retention_pct:5.2f}%"
                               f"  leach={s.leached_pct:.2f}%"
                               f"  deg={s.degraded_pct:.2f}%"
                               f"  precip={s.precipitated_pct:.2f}%")
                print(
                    f"  t={self.t:7.2f}s  dt={dt*1000:5.2f}ms  "
                    f"T_water_mean={s.T_mean_water_c:6.2f}C  "
                    f"T_wall_max={s.T_max_wall_c:6.2f}C  "
                    f"T_wall_inner={s.T_inner_wall_mean_c:6.2f}C  "
                    f"|u|_max={s.u_max_mps*1000:6.2f}mm/s"
                    f"{extra}  "
                    f"(wall {wall:.1f}s, {wall/max(self.t,1e-6):.3f}s/sim-s)"
                )
                last_progress = self.t

        # Final snapshot
        wp.synchronize_device(self.device)
        snapshots_T.append(self.grid.T.numpy().astype(np.float32))
        snapshot_times.append(self.t)
        if self.grid.bubbles is not None:
            from .boiling import read_active_bubbles
            view = read_active_bubbles(self.grid.bubbles)
            detached = view.site_cleared
            bubble_radii_snaps.append(view.departure_radii[detached].astype(np.float32))
            bubble_positions_snaps.append(view.positions[detached].astype(np.float32))

        if out_path is not None:
            out_path = pathlib.Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as f:
                # scalars
                g = f.create_group("scalars")
                g.create_dataset("t", data=np.array([s.t for s in scalars]))
                g.create_dataset("dt", data=np.array([s.dt for s in scalars]))
                g.create_dataset("step", data=np.array([s.step for s in scalars]))
                g.create_dataset("T_mean_water_c", data=np.array([s.T_mean_water_c for s in scalars]))
                g.create_dataset("T_max_water_c", data=np.array([s.T_max_water_c for s in scalars]))
                g.create_dataset("T_min_water_c", data=np.array([s.T_min_water_c for s in scalars]))
                g.create_dataset("T_max_wall_c", data=np.array([s.T_max_wall_c for s in scalars]))
                g.create_dataset("T_inner_wall_mean_c", data=np.array([s.T_inner_wall_mean_c for s in scalars]))
                g.create_dataset("T_inner_wall_max_c", data=np.array([s.T_inner_wall_max_c for s in scalars]))
                g.create_dataset("u_max_mps", data=np.array([s.u_max_mps for s in scalars]))
                # Phase-3 Milestone E: bubble diagnostics time series
                g.create_dataset("n_active_bubbles", data=np.array([s.n_active_bubbles for s in scalars]))
                g.create_dataset("mean_bubble_R_mm", data=np.array([s.mean_bubble_R_mm for s in scalars]))
                g.create_dataset("mean_departed_R_mm", data=np.array([s.mean_departed_bubble_R_mm for s in scalars]))
                g.create_dataset("max_bubble_R_mm", data=np.array([s.max_bubble_R_mm for s in scalars]))
                g.create_dataset("alpha_min", data=np.array([s.alpha_min for s in scalars]))
                # Phase-4 Milestone A: retention_pct time series
                g.create_dataset("retention_pct", data=np.array([s.retention_pct for s in scalars]))
                # Phase-4 instrumentation: leaching vs degradation breakdown
                g.create_dataset("leached_pct", data=np.array([s.leached_pct for s in scalars]))
                g.create_dataset("degraded_pct", data=np.array([s.degraded_pct for s in scalars]))
                g.create_dataset("precipitated_pct", data=np.array([s.precipitated_pct for s in scalars]))
                # Phase-4 dual-solute extension: second solute mass partition.
                # Emitted unconditionally so HDF5 schema is stable whether or
                # not nutrient2 is enabled; when disabled these traces are
                # flat (100 / 0 / 0 / 0) and carry negligible storage.
                g.create_dataset("retention2_pct", data=np.array([s.retention2_pct for s in scalars]))
                g.create_dataset("leached2_pct", data=np.array([s.leached2_pct for s in scalars]))
                g.create_dataset("degraded2_pct", data=np.array([s.degraded2_pct for s in scalars]))
                g.create_dataset("precipitated2_pct", data=np.array([s.precipitated2_pct for s in scalars]))
                # snapshots
                sg = f.create_group("snapshots")
                sg.create_dataset("t", data=np.array(snapshot_times))
                sg.create_dataset("T", data=np.stack(snapshots_T, axis=0), compression="gzip")
                # bubble snapshots (jagged — use variable-length datasets)
                if bubble_radii_snaps:
                    bg = f.create_group("bubble_snapshots")
                    bg.create_dataset("t", data=np.array(snapshot_times[-len(bubble_radii_snaps):]))
                    vlen_f32 = h5py.vlen_dtype(np.float32)
                    rads_ds = bg.create_dataset(
                        "radii_m", (len(bubble_radii_snaps),), dtype=vlen_f32,
                    )
                    for i, r in enumerate(bubble_radii_snaps):
                        rads_ds[i] = r
                    pos_ds = bg.create_dataset(
                        "positions_m", (len(bubble_positions_snaps),), dtype=vlen_f32,
                    )
                    # flatten (N, 3) → (N*3,) so vlen works
                    for i, p in enumerate(bubble_positions_snaps):
                        pos_ds[i] = p.ravel()
                # meta
                m = f.create_group("meta")
                m.attrs["nx"], m.attrs["ny"], m.attrs["nz"] = self.grid.shape
                m.attrs["dx_m"] = self.grid.dx
                m.attrs["pot_material"] = self.cfg.pot.material
                m.attrs["q_base_w_per_m2"] = self.cfg.heating.base_heat_flux_w_per_m2
                m.attrs["boiling_enabled"] = self.cfg.boiling.enabled

        return scalars
