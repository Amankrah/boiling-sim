"""Phase 2 coupled simulation pipeline.

Wires the Milestone-C fluid solver and Milestone-B thermal solver together
per dev-guide §2.2 into a :class:`Simulation` driver. Writes HDF5 output
with a scalar time series every step and downsampled full-field snapshots
on a slower cadence.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

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

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

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
        """
        u_max = compute_max_velocity(self.grid, ws=self.ws_fluid)
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

    def step(self, dt: float | None = None) -> float:
        """Advance the simulation by one step. Returns the dt used."""
        if dt is None:
            dt = self.compute_dt()

        # 1-2. Semi-Lagrangian advection of velocity and temperature.
        advect_all(self.grid, self.ws_fluid, dt, device=self.device)

        # 3. Boussinesq buoyancy on z-faces.
        apply_buoyancy_step(
            self.grid, self.cfg, dt,
            beta=self.beta_water, T_ref_k=self.T_ref_k, device=self.device,
        )

        # 3b. Phase-3 Milestone B: advance bubbles (growth + departure + rise + vent),
        #     then detect new nucleation. Decoupled from fluid for this milestone —
        #     momentum + latent-heat feedback lands in Milestones C and D.
        if self.cfg.boiling.enabled and self.grid.bubbles is not None:
            from .boiling import step_bubbles
            step_bubbles(
                self.grid, self.grid.bubbles, self.cfg, dt,
                sim_time=self.t, step_count=self.step_count, device=self.device,
            )

        # 4. Conjugate heat diffusion + all boundary sources (stove, Newton, evap).
        conduct_one_step(self.grid, self.props, self.ws_thermal, self.cfg, dt, device=self.device)

        # 4b. Phase-3: Eulerian wall boiling flux (microlayer evaporation).
        #     Directly cools pot-wall cells at nucleation sites, proportional to
        #     local superheat. This is the dominant wall-cooling mechanism that
        #     the Lagrangian scatter alone cannot provide (it acts on mid-fluid).
        if self.cfg.boiling.enabled and self.grid.bubbles is not None:
            from .boiling import step_wall_boiling_flux
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
            _step_reaction_diffusion_leach(
                self.primary_slot, self.grid, D_carrot, dt, device=self.device,
            )
            if self.secondary_slot is not None:
                _step_reaction_diffusion_leach(
                    self.secondary_slot, self.grid, D_carrot, dt, device=self.device,
                )

        # 5. No-slip on solid faces before projection.
        enforce_no_slip(self.grid, device=self.device)

        # 6. Pressure projection — enforces ∇·u = 0 in fluid.
        pressure_projection(
            self.grid, self.ws_fluid, self.cfg, dt,
            rho=self.rho_water, device=self.device,
        )

        # 7. Re-enforce no-slip (pressure subtraction doesn't touch solid faces,
        #    but this guards against drift from numerical error).
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
            _step_advect_clamp(self.primary_slot, self.grid, dt, device=self.device)
            if self.secondary_slot is not None:
                _step_advect_clamp(self.secondary_slot, self.grid, dt, device=self.device)

        self.t += dt
        self.step_count += 1
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

        # Phase-3 bubble diagnostics — cheap host roundtrip over the bubble pool.
        n_active = 0
        mean_R_mm = 0.0
        mean_departed_R_mm = 0.0
        max_R_mm = 0.0
        alpha_min = 1.0
        if self.grid.bubbles is not None:
            bubbles = self.grid.bubbles.bubbles.numpy()
            active_mask = bubbles["active"] == 1
            n_active = int(active_mask.sum())
            if n_active > 0:
                R = bubbles["radius"][active_mask]
                mean_R_mm = float(R.mean() * 1000.0)
                max_R_mm = float(R.max() * 1000.0)
                detached_mask = bubbles["site_cleared"][active_mask] == 1
                if detached_mask.any():
                    # Use the frozen departure_radius so this reports the
                    # Fritz-departure population mean, not post-departure growth.
                    R_dep = bubbles["departure_radius"][active_mask][detached_mask]
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

        return ScalarSample(
            t=self.t,
            dt=dt_last,
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
                    bs = self.grid.bubbles.bubbles.numpy()
                    # Departure-diameter histogram: restrict to bubbles that
                    # have actually detached from a wall site and use the
                    # frozen ``departure_radius`` rather than the live
                    # ``radius`` (which keeps growing during rise). This
                    # eliminates the near-zero spike caused by counting
                    # in-flight infant bubbles as if they were departures.
                    mask = (bs["active"] == 1) & (bs["site_cleared"] == 1)
                    bubble_radii_snaps.append(
                        bs["departure_radius"][mask].astype(np.float32)
                    )
                    bubble_positions_snaps.append(bs["position"][mask].astype(np.float32))
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
            bs = self.grid.bubbles.bubbles.numpy()
            mask = (bs["active"] == 1) & (bs["site_cleared"] == 1)
            bubble_radii_snaps.append(bs["departure_radius"][mask].astype(np.float32))
            bubble_positions_snaps.append(bs["position"][mask].astype(np.float32))

        if out_path is not None:
            out_path = pathlib.Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as f:
                # scalars
                g = f.create_group("scalars")
                g.create_dataset("t", data=np.array([s.t for s in scalars]))
                g.create_dataset("dt", data=np.array([s.dt for s in scalars]))
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
