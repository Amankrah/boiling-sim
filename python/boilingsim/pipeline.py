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
from .geometry import MAT_FLUID, MAT_POT_WALL, Grid, build_pot_geometry
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
    u_max_mps: float


class Simulation:
    """Coupled CFD + thermal pipeline for a boiling-sim scenario."""

    def __init__(self, cfg: ScenarioConfig, device: str = "cuda:0") -> None:
        self.cfg = cfg
        self.device = device

        self.grid: Grid = build_pot_geometry(cfg, device=device)
        self.props = MaterialProps.from_scenario(cfg, device=device)
        self.ws_fluid: FluidWorkspace = allocate_fluid_workspace(self.grid, device=device)
        self.ws_thermal: ThermalWorkspace = allocate_thermal_workspace(self.grid, device=device)

        # Water-specific constants (Phase 2 uses constant properties).
        self.rho_water = float(self.props.rho[MAT_FLUID])
        self.beta_water = 2.07e-4  # 1/K near 25 °C (water)
        self.T_ref_k = cfg.water.initial_temp_c + 273.15

        # Precompute host-side masks for diagnostics (avoid GPU→CPU roundtrip each step).
        self._mat_host = self.grid.mat.numpy()
        self._water_mask = self._mat_host == MAT_FLUID
        self._wall_mask = self._mat_host == MAT_POT_WALL

        self.t: float = 0.0
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def compute_dt(self) -> float:
        """Return a stable timestep from active stability constraints.

        With ``use_implicit_conduction=True`` (default) the thermal-diffusion
        limit is dropped — BE is unconditionally stable — so Δt is bounded
        by advection CFL and the user-set ``max_dt_s``.
        """
        u_max = compute_max_velocity(self.grid)
        dt_cfl = self.grid.dx / max(u_max, 1.0e-8)
        dt_cap = self.cfg.solver.max_dt_s / self.cfg.solver.cfl_safety_factor
        if self.cfg.solver.use_implicit_conduction:
            dt = min(dt_cfl, dt_cap)
        else:
            dt_thermal = compute_max_dt_conduction(self.props, self.grid.dx, safety=1.0)
            dt = min(dt_thermal, dt_cfl, dt_cap)
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

        # 4. Conjugate heat diffusion + all boundary sources (stove, Newton, evap).
        conduct_one_step(self.grid, self.props, self.ws_thermal, self.cfg, dt, device=self.device)

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

        self.t += dt
        self.step_count += 1
        return dt

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sample_scalars(self, dt_last: float) -> ScalarSample:
        """Capture mean/min/max temperatures and peak velocity (host roundtrip)."""
        T = self.grid.T.numpy()
        T_w = T[self._water_mask]
        T_wall = T[self._wall_mask] if self._wall_mask.any() else T
        u_max = compute_max_velocity(self.grid)
        return ScalarSample(
            t=self.t,
            dt=dt_last,
            T_mean_water_c=float(T_w.mean() - 273.15),
            T_max_water_c=float(T_w.max() - 273.15),
            T_min_water_c=float(T_w.min() - 273.15),
            T_max_wall_c=float(T_wall.max() - 273.15),
            u_max_mps=float(u_max),
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

        wall_t0 = time.perf_counter()

        while self.t < total_time_s:
            dt = self.step()

            if self.step_count % scalar_every_n_steps == 0 or self.t >= total_time_s:
                scalars.append(self.sample_scalars(dt))

            # Full-field snapshot cadence
            if self.t - last_snapshot_t >= snapshot_every_s:
                wp.synchronize_device(self.device)
                snapshots_T.append(self.grid.T.numpy().astype(np.float32))
                snapshot_times.append(self.t)
                last_snapshot_t = self.t

            if self.t - last_progress >= progress_every_s and scalars:
                s = scalars[-1]
                wall = time.perf_counter() - wall_t0
                print(
                    f"  t={self.t:7.2f}s  dt={dt*1000:5.2f}ms  "
                    f"T_water_mean={s.T_mean_water_c:6.2f}°C  "
                    f"T_wall_max={s.T_max_wall_c:6.2f}°C  "
                    f"|u|_max={s.u_max_mps*1000:6.2f}mm/s  "
                    f"(wall {wall:.1f}s, {wall/max(self.t,1e-6):.3f}s/sim-s)"
                )
                last_progress = self.t

        # Final snapshot
        wp.synchronize_device(self.device)
        snapshots_T.append(self.grid.T.numpy().astype(np.float32))
        snapshot_times.append(self.t)

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
                g.create_dataset("u_max_mps", data=np.array([s.u_max_mps for s in scalars]))
                # snapshots
                sg = f.create_group("snapshots")
                sg.create_dataset("t", data=np.array(snapshot_times))
                sg.create_dataset("T", data=np.stack(snapshots_T, axis=0), compression="gzip")
                # meta
                m = f.create_group("meta")
                m.attrs["nx"], m.attrs["ny"], m.attrs["nz"] = self.grid.shape
                m.attrs["dx_m"] = self.grid.dx
                m.attrs["pot_material"] = self.cfg.pot.material
                m.attrs["q_base_w_per_m2"] = self.cfg.heating.base_heat_flux_w_per_m2

        return scalars
