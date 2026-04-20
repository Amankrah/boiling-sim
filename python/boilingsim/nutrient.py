"""Carrot nutrient retention: reaction-diffusion-leaching for beta-carotene.

Phase 4 replaces this stub with the full RPI-style chemistry:

* Milestone A (this file now): Arrhenius degradation ``dC/dt = -k(T)*C`` on
  carrot cells, with the exact-integration update ``C *= exp(-k(T)*dt)``
  (unconditionally stable regardless of dt).
* Milestone B: in-carrot diffusion (zero-flux Neumann at the carrot surface)
  via a 7-point explicit stencil.
* Milestone C: Sherwood-correlation surface leaching into a water-side
  passive scalar ``C_water``.
* Milestone D: advection of ``C_water`` with the existing velocity field,
  integration into ``Simulation.step``.
* Milestone E: validation sweep against the dev-guide's R(600 s) in [80%, 90%]
  target for a 25 mm carrot.

Physical constants (dev-guide sec.4 / data/materials.json:49-67):

    E_a / R       - activation energy (J/mol) over gas constant (J/mol/K)
    k0            - pre-exponential factor (1/s), default 2.63e6
    D_eff         - effective diffusivity in carrot (m^2/s), default 2e-10
    K_partition   - carrot-vs-water equilibrium ratio, default 1e-5
                    (bare beta-carotene in pure water, Treszczanowicz 1998).
                    Raise for water-soluble nutrients (vitamin C ~1.0) or
                    when an oil phase carries the carotene (~0.007).
    C_water_sat   - aqueous solubility cap in mg / kg water, default 6e-3
                    (beta-carotene at 100 C ~6 ug/L). Not usually binding
                    at the K=1e-5 default because partition equilibrium hits
                    first; kept as a hard physical upper bound for
                    robustness against parameter sweeps.
    C0            - initial beta-carotene loading (mg/kg), default 83

Concentration units: mg of beta-carotene per kg of carrot tissue. The
absolute scale doesn't matter for the decay + diffusion kinetics (which
are linear in C); retention is always reported as C / C0.

Why these defaults give retention in [80, 90] %% over 600 s: at K = 1e-5
and our 107:1 water:carrot volume ratio, the maximum mass that can leach
before equilibrium is ~0.1 %% of C0. The Phase-4 retention budget is
therefore spent almost entirely on Arrhenius degradation, which at
k(100 C) ~ 4.2e-4 /s gives R(600 s) = exp(-0.25) ~ 78 %% for a
uniformly-heated carrot and somewhat higher in practice because the
carrot core lags water temperature for the first minute. The prior
default K=0.007 allowed up to 75 %% of the carrot to dissolve at
equilibrium -- a mis-modelling of 'beta-carotene in pure water' that
was really 'beta-carotene in a vegetable-oil emulsion'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .config import ScenarioConfig
    from .geometry import Grid


# Universal gas constant in J / (mol K).
_R_GAS = 8.31446261815324

# Material ID for carrot cells (must match geometry.py).
_MAT_CARROT = 3


# ---------------------------------------------------------------------------
# Milestone A: initialization + Arrhenius degradation + retention diagnostic
# ---------------------------------------------------------------------------


@wp.kernel
def init_carrot_concentration(
    C: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    C0: float,
    mat_carrot: int,
):
    """Set ``C = C0`` on every carrot cell, ``C = 0`` everywhere else.

    Launched over the full (nx, ny, nz) domain at geometry-build time when
    ``cfg.nutrient.enabled``. The water-side passive scalar ``C_water``
    (allocated in Milestone C) starts at zero regardless.
    """
    i, j, k = wp.tid()
    if mat[i, j, k] == mat_carrot:
        C[i, j, k] = C0
    else:
        C[i, j, k] = 0.0


# ---------------------------------------------------------------------------
# Workspace: ping-pong buffer for explicit diffusion
# ---------------------------------------------------------------------------


@dataclass
class NutrientWorkspace:
    """Scratch arrays for the nutrient pipeline.

    * ``C_work``            -- ping-pong buffer for explicit in-carrot
                               diffusion (Milestone B).
    * ``C_water_tmp``       -- destination buffer for semi-Lagrangian /
                               upwind advection of the water-side passive
                               scalar (Milestone D).
    * ``precipitated_mass`` -- single-element GPU scalar (length-1 array)
                               accumulating the cumulative mass clipped by
                               the post-advection saturation clamp. Reads
                               back as ``precipitated_pct`` in the diagnostic
                               triple-sum (retention + leached + degraded +
                               precipitated = 100 %%). See
                               :func:`clamp_c_water_and_track_precipitation`.
    """

    C_work: wp.array            # (nx, ny, nz) -- diffusion ping-pong
    C_water_tmp: wp.array       # (nx, ny, nz) -- advection destination
    precipitated_mass: wp.array  # (1,) -- cumulative clipped mass
    # Dual-solute extension: independent scratch + counter for the secondary
    # solute. Allocated only when ``cfg.nutrient2.enabled``; otherwise None.
    # Kept on the same workspace rather than a second NutrientWorkspace so the
    # single ``ws_nutrient`` handle in Simulation covers both slots.
    C_work2: Any = None
    C_water_tmp2: Any = None
    precipitated_mass2: Any = None


def allocate_nutrient_workspace(
    grid: "Grid",
    device: str = "cuda:0",
    alloc_secondary: bool = False,
) -> NutrientWorkspace:
    """Allocate ping-pong buffers for explicit diffusion + SL advection, plus
    the cumulative-precipitation counter. When ``alloc_secondary`` is True,
    also allocates independent scratch + counter for the second solute.
    """
    nx, ny, nz = grid.shape
    ws = NutrientWorkspace(
        C_work=wp.zeros((nx, ny, nz), dtype=float, device=device),
        C_water_tmp=wp.zeros((nx, ny, nz), dtype=float, device=device),
        precipitated_mass=wp.zeros(1, dtype=float, device=device),
    )
    if alloc_secondary:
        ws.C_work2 = wp.zeros((nx, ny, nz), dtype=float, device=device)
        ws.C_water_tmp2 = wp.zeros((nx, ny, nz), dtype=float, device=device)
        ws.precipitated_mass2 = wp.zeros(1, dtype=float, device=device)
    return ws


@wp.kernel
def arrhenius_degrade(
    C: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    k0: float,
    E_a_over_R: float,
    dt: float,
    mat_carrot: int,
):
    """Apply exact-integration Arrhenius first-order decay on carrot cells.

    Per-cell update: ``C *= exp(-k(T)*dt)`` with ``k(T) = k0*exp(-E_a/(R*T))``.
    The exact-integration form is unconditionally stable regardless of dt, so
    the reaction term never constrains the global timestep (which stays
    bounded by advection CFL per the Phase-3 pattern).

    Non-carrot cells are skipped (C is already 0 there from init; degradation
    on C=0 is a no-op but we guard anyway).
    """
    i, j, k = wp.tid()
    if mat[i, j, k] != mat_carrot:
        return
    T_k = T[i, j, k]
    if T_k <= 0.0:
        return  # guard against uninitialized / negative temperatures
    k_rate = k0 * wp.exp(-E_a_over_R / T_k)
    C[i, j, k] = C[i, j, k] * wp.exp(-k_rate * dt)


@wp.kernel
def arrhenius_degrade_water(
    C_water: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    k0: float,
    E_a_over_R: float,
    dt: float,
    mat_fluid: int,
):
    """Arrhenius degradation on the leached water-side pool.

    Once beta-carotene has left the carrot into the water phase it is still
    exposed to ~100 C boiling water and continues to decompose at the
    Arrhenius rate; this is the dominant loss pathway in the simmered-soup
    literature. The earlier Phase 4 implementation only degraded the carrot
    side, which left leached mass as an artificial long-term reservoir in
    the bulk fluid.

    Same exact-integration form as :func:`arrhenius_degrade`, applied on
    every fluid cell using the local T. Sub-saturation fluid cells (T <
    ~60 C) see essentially zero rate and are a no-op.
    """
    i, j, k = wp.tid()
    if mat[i, j, k] != mat_fluid:
        return
    T_k = T[i, j, k]
    if T_k <= 0.0:
        return
    k_rate = k0 * wp.exp(-E_a_over_R / T_k)
    C_water[i, j, k] = C_water[i, j, k] * wp.exp(-k_rate * dt)


# ---------------------------------------------------------------------------
# Python-side drivers
# ---------------------------------------------------------------------------


def initialize_nutrient_field(
    grid: "Grid",
    cfg: "ScenarioConfig",
    device: str = "cuda:0",
    target_C: Any = None,
    C0_override: float | None = None,
) -> None:
    """Populate a concentration array from C0 on carrot cells.

    By default targets ``grid.C`` and uses ``cfg.nutrient.C0_mg_per_kg`` --
    preserving the single-solute contract. For the dual-solute extension
    pass ``target_C=grid.C2`` and ``C0_override=cfg.nutrient2.C0_mg_per_kg``
    to populate the secondary field. Called from
    :func:`geometry.build_pot_geometry`.
    """
    C_arr = target_C if target_C is not None else grid.C
    if C_arr is None:
        raise RuntimeError(
            "target concentration array not allocated; cannot initialize "
            "nutrient field"
        )
    C0 = C0_override if C0_override is not None else cfg.nutrient.C0_mg_per_kg
    nx, ny, nz = grid.shape
    wp.launch(
        init_carrot_concentration,
        dim=(nx, ny, nz),
        inputs=[C_arr, grid.mat, C0, _MAT_CARROT],
        device=device,
    )


def step_degrade(
    grid: "Grid",
    cfg: "ScenarioConfig",
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Advance nutrient concentration by ``dt`` via Arrhenius degradation
    on **both** phases: the carrot-side ``grid.C`` (primary reservoir) and
    the water-side ``grid.C_water`` (leached pool). Leached solute in
    100 C water keeps decomposing; skipping it biases the retention
    diagnostic toward "leached" vs "degraded" and artificially preserves
    mass that in reality would have vanished.
    """
    if grid.C is None:
        return  # nutrient disabled or not yet allocated
    from .geometry import MAT_FLUID as _MF

    nx, ny, nz = grid.shape
    E_a_j_per_mol = cfg.nutrient.E_a_kJ_per_mol * 1000.0
    E_a_over_R = E_a_j_per_mol / _R_GAS

    # Carrot-side primary reservoir.
    wp.launch(
        arrhenius_degrade,
        dim=(nx, ny, nz),
        inputs=[
            grid.C,
            grid.T,
            grid.mat,
            cfg.nutrient.k0_per_s,
            E_a_over_R,
            dt,
            _MAT_CARROT,
        ],
        device=device,
    )

    # Leached pool in the water phase.
    if grid.C_water is not None:
        wp.launch(
            arrhenius_degrade_water,
            dim=(nx, ny, nz),
            inputs=[
                grid.C_water,
                grid.T,
                grid.mat,
                cfg.nutrient.k0_per_s,
                E_a_over_R,
                dt,
                _MF,
            ],
            device=device,
        )


def retention_fraction(
    grid: "Grid",
    cfg: "ScenarioConfig",
) -> float:
    """Return the fraction of initial beta-carotene remaining in the carrot.

    Definition: ``R = (sum C over carrot cells) / (C0 * N_carrot_cells)``
    where N_carrot_cells is the number of cells with ``mat == MAT_CARROT``.
    Volume-weighted, so this is the retention fraction even if the carrot
    straddles the staircase grid unevenly.

    Returns 1.0 if the nutrient field is not allocated (disabled scenario).
    Values outside [0, 1] indicate a bug or numerical drift.
    """
    if grid.C is None:
        return 1.0
    mat_np = grid.mat.numpy()
    mask = mat_np == _MAT_CARROT
    n_carrot = int(mask.sum())
    if n_carrot == 0:
        return 1.0
    C_np = grid.C.numpy()
    total_mass = float(C_np[mask].sum())
    reference = cfg.nutrient.C0_mg_per_kg * float(n_carrot)
    if reference <= 0.0:
        return 1.0
    return total_mass / reference


def arrhenius_rate(cfg: "ScenarioConfig", T_k: float) -> float:
    """Return ``k(T) = k0 * exp(-E_a / (R*T))`` in 1/s for host-side unit tests."""
    E_a_j_per_mol = cfg.nutrient.E_a_kJ_per_mol * 1000.0
    return cfg.nutrient.k0_per_s * math.exp(-E_a_j_per_mol / (_R_GAS * T_k))


def water_pool_fraction(grid: "Grid", cfg: "ScenarioConfig") -> float:
    """Fraction of initial carrot beta-carotene mass that has been **leached**
    into the water-side passive scalar.

    Computed as ``sum(C_water over fluid cells) / (C0 * N_carrot_cells)``.
    Concentration units cancel because both C and C_water use the same
    (mg/kg, "abstract concentration"), and cell volumes are uniform.

    This + :func:`retention_fraction` partition every β-carotene atom by its
    current location:

        retention_fraction      -> fraction still in carrot
        water_pool_fraction     -> fraction leached, sitting in fluid C_water
        degraded_fraction       -> 1 - retention - water_pool
                                    (Arrhenius destruction + small numerical
                                     leakage from C_water SL advection at
                                     solid boundaries)

    Returns 0.0 when the C_water field is not allocated (Phase-3-only run
    or nutrient disabled).
    """
    if grid.C_water is None or grid.C is None:
        return 0.0
    mat_np = grid.mat.numpy()
    n_carrot = int((mat_np == _MAT_CARROT).sum())
    if n_carrot == 0:
        return 0.0
    Cw = grid.C_water.numpy()
    # Sum over all cells -- non-fluid cells are explicitly zeroed by the
    # advection kernel, so this is identical to summing over fluid cells but
    # also catches any drift / numerical leakage at boundaries.
    total_water_mass = float(Cw.sum())
    initial_carrot_mass = cfg.nutrient.C0_mg_per_kg * float(n_carrot)
    if initial_carrot_mass <= 0.0:
        return 0.0
    return total_water_mass / initial_carrot_mass


# ---------------------------------------------------------------------------
# Milestone B: in-carrot diffusion (explicit 7-point stencil)
# ---------------------------------------------------------------------------


@wp.kernel
def diffuse_nutrient_explicit(
    C_in: wp.array3d(dtype=float),
    C_out: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    D_eff: float,
    dx: float,
    dt: float,
    mat_carrot: int,
):
    """Explicit 7-point Laplacian update with zero-flux Neumann BC at the
    carrot surface.

        C_new = C + (D_eff*dt / dx^2) * Sum_{neighbour in carrot}(C_n - C)

    Faces to non-carrot cells contribute zero flux -- we simply omit them
    from the sum, which is equivalent to mirroring the cell's own value on
    the outside of the boundary. This makes the carrot a closed domain for
    diffusion: mass is conserved inside until Milestone C's leaching kernel
    opens the surface.

    Stability: diffusion number <= 1/6 requires
        dt <= dx^2 / (6 * D_eff).
    At dx = 2 mm and D_eff = 2e-10 m^2/s this is ~6700 s -- always satisfied
    by the advection-CFL dt that dominates the real pipeline (~ms).
    """
    i, j, k = wp.tid()

    # Non-carrot cells: copy through unchanged (C should already be 0).
    if mat[i, j, k] != mat_carrot:
        C_out[i, j, k] = C_in[i, j, k]
        return

    c_self = C_in[i, j, k]
    lap = float(0.0)

    # +x
    if i + 1 < C_in.shape[0] and mat[i + 1, j, k] == mat_carrot:
        lap = lap + (C_in[i + 1, j, k] - c_self)
    # -x
    if i >= 1 and mat[i - 1, j, k] == mat_carrot:
        lap = lap + (C_in[i - 1, j, k] - c_self)
    # +y
    if j + 1 < C_in.shape[1] and mat[i, j + 1, k] == mat_carrot:
        lap = lap + (C_in[i, j + 1, k] - c_self)
    # -y
    if j >= 1 and mat[i, j - 1, k] == mat_carrot:
        lap = lap + (C_in[i, j - 1, k] - c_self)
    # +z
    if k + 1 < C_in.shape[2] and mat[i, j, k + 1] == mat_carrot:
        lap = lap + (C_in[i, j, k + 1] - c_self)
    # -z
    if k >= 1 and mat[i, j, k - 1] == mat_carrot:
        lap = lap + (C_in[i, j, k - 1] - c_self)

    coeff = D_eff * dt / (dx * dx)
    C_out[i, j, k] = c_self + coeff * lap


def _diffusion_stability_dt_D(dx: float, D_eff: float) -> float:
    """Primitive-argument 7-point explicit-diffusion stability bound.

    Used by the dual-solute pipeline which needs the minimum of both solutes'
    bounds without threading full cfg through a helper.
    """
    return (dx * dx) / (6.0 * D_eff)


def diffusion_stability_dt(cfg: "ScenarioConfig", dx: float) -> float:
    """Return the explicit-diffusion stability bound ``dx^2 / (6*D_eff)``.

    At our dev-grid parameters (dx = 2 mm, D_eff = 2e-10 m^2/s) this is
    ~6700 s, well above any realistic solver dt. Included as a safety
    guard that :func:`step_diffuse_nutrient` asserts against.
    """
    return _diffusion_stability_dt_D(dx, cfg.nutrient.D_eff_m2_per_s)


# ---------------------------------------------------------------------------
# Milestone C: Sherwood correlation + surface leaching
# ---------------------------------------------------------------------------


@wp.func
def sherwood_h_m(
    u_mag: float,
    D_carrot: float,
    nu_water: float,
    D_water_molec: float,
) -> float:
    """Mass-transfer coefficient ``h_m [m/s]`` from the forced-convection
    Ranz-Marshall / Whitaker-style Sherwood correlation:

        Sh = 0.683 * Re^0.466 * Sc^(1/3)
        h_m = Sh * D_water_molec / D_carrot

    with ``Re = |u|*D_carrot / nu_water`` and ``Sc = nu_water / D_water_molec``.
    A floor at Sh = 2 represents the diffusion-limited (stagnant-fluid)
    baseline: for a sphere in quiescent fluid the analytic solution is Sh=2.
    """
    Re = u_mag * D_carrot / nu_water
    Sc = nu_water / D_water_molec
    Sh_forced = float(0.0)
    if Re > 0.0:
        Sh_forced = 0.683 * wp.pow(Re, 0.466) * wp.pow(Sc, 0.33333333)
    Sh = wp.max(float(2.0), Sh_forced)
    return Sh * D_water_molec / D_carrot


@wp.func
def _freestream_u_mag(
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    i: int, j: int, k: int,
    di: int, dj: int, dk: int,
    mat_fluid: int,
) -> float:
    """Sample the cell-centre fluid |u| N cells off-surface in direction
    (di, dj, dk) so the no-slip boundary at the carrot face doesn't collapse
    the Reynolds number.

    Tries N = 3, 2, 1 cells off; picks the *first* one that is genuinely
    fluid (no wall, no carrot, no air). If none of the three are fluid
    (thin channel / corner), returns 0 and the forced-convection branch of
    the Sherwood correlation drops out, leaving the Sh = 2 floor.

    This is what fixes the previously-observed Re~0 collapse: the kernel
    was reading ``ux[neighbour]`` which is exactly the no-slip face at the
    carrot-fluid interface, plus one far face that carries the real flow.
    """
    nx = ux.shape[0]
    ny = uy.shape[1]
    nz = uz.shape[2]
    # N = 3 preferred, fall back to 2, then 1.
    for n in range(3, 0, -1):
        si = i + di * n
        sj = j + dj * n
        sk = k + dk * n
        if si < 0 or si >= mat.shape[0]:
            continue
        if sj < 0 or sj >= mat.shape[1]:
            continue
        if sk < 0 or sk >= mat.shape[2]:
            continue
        if mat[si, sj, sk] != mat_fluid:
            continue
        # Guard MAC face indices too.
        if si + 1 >= nx or sj + 1 >= ny or sk + 1 >= nz:
            continue
        u_c_x = float(0.5) * (ux[si, sj, sk] + ux[si + 1, sj, sk])
        u_c_y = float(0.5) * (uy[si, sj, sk] + uy[si, sj + 1, sk])
        u_c_z = float(0.5) * (uz[si, sj, sk] + uz[si, sj, sk + 1])
        return wp.sqrt(u_c_x * u_c_x + u_c_y * u_c_y + u_c_z * u_c_z)
    return float(0.0)


@wp.func
def _leach_flux_capped(
    c_self: float,
    c_water: float,
    h_m: float,
    K_partition: float,
    C_water_sat: float,
    coeff: float,
) -> float:
    """Sherwood driving-force flux with solubility cap and no-condensation gate.

    Returns the per-face flux J [mg/m^2/s] (multiplied later by coeff = dt/dx
    to get the per-cell dC update). Three guards:

    * If the neighbour is already at saturation, return 0 (no more dissolves).
    * Compute J = h_m * (c_self - c_water/K). If negative, return 0
      (no solute condensation back into the carrot in Phase 4).
    * Cap J at (C_water_sat - c_water)/(6*coeff) so the worst-case sum of
      atomic_add contributions from up to 6 carrot neighbours sharing this
      fluid cell lands at most at C_water_sat. Earlier versions capped at
      (C_water_sat - c_water)/coeff per face, which is correct for a single
      contributor but races when multiple carrot cells target the same
      fluid neighbour: each reads the same pre-update c_water, each approves
      the full-magnitude flux, and the atomic_add sums them past sat. The
      /6 divisor turns the cap into a hard pre-atomic guarantee at the cost
      of slowing the final ~1/6 of the saturation approach.
    """
    if c_water >= C_water_sat:
        return float(0.0)
    J = h_m * (c_self - c_water / K_partition)
    if J <= 0.0:
        return float(0.0)
    J_max = (C_water_sat - c_water) / (coeff * float(6.0))
    return wp.min(J, J_max)


@wp.kernel
def leach_at_surface(
    C: wp.array3d(dtype=float),
    C_water: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    dt: float,
    D_carrot: float,
    nu_water: float,
    D_water_molec: float,
    K_partition: float,
    C_water_sat: float,
    mat_fluid: int,
    mat_carrot: int,
):
    """Flux-transport solute across carrot-fluid faces with two physical caps.

    For each carrot cell (i, j, k), loop over its 6 face neighbours. Wherever
    the neighbour is fluid, compute the Sherwood-correlation mass-transfer
    coefficient from the *neighbour's* cell-centre velocity magnitude, then
    apply the driving-force flux

        J [mg/m^2/s] = h_m * (C_carrot - C_water_neighbour / K_partition)

    Two physical caps prevent unphysical bulk-water sinks for poorly-soluble
    solutes:

    1. **Solubility cap.** Once C_water_neighbour reaches the absolute
       saturation concentration ``C_water_sat`` (e.g. ~0.6 mg/L for beta-
       carotene at 100 C), no more solute can dissolve regardless of the
       partition driving force. The kernel skips faces where the neighbour
       is already saturated and clamps J so the post-step C_water cannot
       overshoot C_water_sat.
    2. **No condensation.** J is clamped to be non-negative -- if K_partition
       and the local C_water make the driving force negative, this would
       physically imply solute precipitating back into the carrot, a Phase 5
       feature.

    Mass conservation: per-face dC_carrot = -dC_water_neighbour so the only
    way mass leaves the system is via the solubility cap (when atomic-add
    races push C_water above C_water_sat in a single step, which the
    test_leaching_mass_conservation tolerance absorbs).
    """
    i, j, k = wp.tid()

    if mat[i, j, k] != mat_carrot:
        return

    c_self = C[i, j, k]
    delta_self = float(0.0)
    coeff = dt / dx  # converts flux [mg/m^2/s] to dC [same units as C] per face

    # Helper inlined once per face. Sample fluid-cell-centre |u| via MAC averaging.

    # +x face
    if i + 1 < C.shape[0] and mat[i + 1, j, k] == mat_fluid:
        ni = i + 1
        umag = _freestream_u_mag(ux, uy, uz, mat, i, j, k, 1, 0, 0, mat_fluid)
        h_m = sherwood_h_m(umag, D_carrot, nu_water, D_water_molec)
        J = _leach_flux_capped(c_self, C_water[ni, j, k], h_m,
                                 K_partition, C_water_sat, coeff)
        delta_self -= J * coeff
        wp.atomic_add(C_water, ni, j, k, J * coeff)
    # -x
    if i >= 1 and mat[i - 1, j, k] == mat_fluid:
        ni = i - 1
        umag = _freestream_u_mag(ux, uy, uz, mat, i, j, k, -1, 0, 0, mat_fluid)
        h_m = sherwood_h_m(umag, D_carrot, nu_water, D_water_molec)
        J = _leach_flux_capped(c_self, C_water[ni, j, k], h_m,
                                 K_partition, C_water_sat, coeff)
        delta_self -= J * coeff
        wp.atomic_add(C_water, ni, j, k, J * coeff)
    # +y
    if j + 1 < C.shape[1] and mat[i, j + 1, k] == mat_fluid:
        nj = j + 1
        umag = _freestream_u_mag(ux, uy, uz, mat, i, j, k, 0, 1, 0, mat_fluid)
        h_m = sherwood_h_m(umag, D_carrot, nu_water, D_water_molec)
        J = _leach_flux_capped(c_self, C_water[i, nj, k], h_m,
                                 K_partition, C_water_sat, coeff)
        delta_self -= J * coeff
        wp.atomic_add(C_water, i, nj, k, J * coeff)
    # -y
    if j >= 1 and mat[i, j - 1, k] == mat_fluid:
        nj = j - 1
        umag = _freestream_u_mag(ux, uy, uz, mat, i, j, k, 0, -1, 0, mat_fluid)
        h_m = sherwood_h_m(umag, D_carrot, nu_water, D_water_molec)
        J = _leach_flux_capped(c_self, C_water[i, nj, k], h_m,
                                 K_partition, C_water_sat, coeff)
        delta_self -= J * coeff
        wp.atomic_add(C_water, i, nj, k, J * coeff)
    # +z
    if k + 1 < C.shape[2] and mat[i, j, k + 1] == mat_fluid:
        nk = k + 1
        umag = _freestream_u_mag(ux, uy, uz, mat, i, j, k, 0, 0, 1, mat_fluid)
        h_m = sherwood_h_m(umag, D_carrot, nu_water, D_water_molec)
        J = _leach_flux_capped(c_self, C_water[i, j, nk], h_m,
                                 K_partition, C_water_sat, coeff)
        delta_self -= J * coeff
        wp.atomic_add(C_water, i, j, nk, J * coeff)
    # -z
    if k >= 1 and mat[i, j, k - 1] == mat_fluid:
        nk = k - 1
        umag = _freestream_u_mag(ux, uy, uz, mat, i, j, k, 0, 0, -1, mat_fluid)
        h_m = sherwood_h_m(umag, D_carrot, nu_water, D_water_molec)
        J = _leach_flux_capped(c_self, C_water[i, j, nk], h_m,
                                 K_partition, C_water_sat, coeff)
        delta_self -= J * coeff
        wp.atomic_add(C_water, i, j, nk, J * coeff)

    C[i, j, k] = c_self + delta_self


def sherwood_h_m_host(
    cfg: "ScenarioConfig",
    u_mag: float,
    D_carrot: float,
) -> float:
    """Host-side Sherwood h_m for unit tests. Same formula as the wp.func."""
    nu = cfg.nutrient.nu_water_m2_per_s
    Dw = cfg.nutrient.D_water_molec_m2_per_s
    Re = u_mag * D_carrot / nu
    Sc = nu / Dw
    Sh_forced = 0.0 if Re <= 0.0 else 0.683 * (Re ** 0.466) * (Sc ** (1.0 / 3.0))
    Sh = max(2.0, Sh_forced)
    return Sh * Dw / D_carrot


# ---------------------------------------------------------------------------
# Milestone D: semi-Lagrangian advection of the water-side passive scalar
# ---------------------------------------------------------------------------


@wp.func
def _tri_sample_cc(
    field: wp.array3d(dtype=float),
    p: wp.vec3,
    origin: wp.vec3,
    dx: float,
) -> float:
    """Trilinear sample a cell-centred scalar at world-space point ``p``.
    Same body as ``boiling._sample_cell_scalar`` -- duplicated here to
    keep the nutrient module self-contained without a cross-module wp.func
    import."""
    fx = (p[0] - origin[0]) / dx - 0.5
    fy = (p[1] - origin[1]) / dx - 0.5
    fz = (p[2] - origin[2]) / dx - 0.5
    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]
    fx = wp.clamp(fx, float(0.0), float(nx - 1) - 1.0e-6)
    fy = wp.clamp(fy, float(0.0), float(ny - 1) - 1.0e-6)
    fz = wp.clamp(fz, float(0.0), float(nz - 1) - 1.0e-6)
    i0 = int(fx); j0 = int(fy); k0 = int(fz)
    tx = fx - float(i0); ty = fy - float(j0); tz = fz - float(k0)
    c000 = field[i0, j0, k0]
    c100 = field[i0 + 1, j0, k0]
    c010 = field[i0, j0 + 1, k0]
    c110 = field[i0 + 1, j0 + 1, k0]
    c001 = field[i0, j0, k0 + 1]
    c101 = field[i0 + 1, j0, k0 + 1]
    c011 = field[i0, j0 + 1, k0 + 1]
    c111 = field[i0 + 1, j0 + 1, k0 + 1]
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@wp.kernel
def advect_c_water(
    C_water: wp.array3d(dtype=float),
    C_water_new: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    origin: wp.vec3,
    dx: float,
    dt: float,
    mat_fluid: int,
):
    """First-order upwind finite-volume advection of the water passive scalar.

    Flux formulation guarantees strict mass conservation at machine precision
    across the C_water field: each face contributes an equal-and-opposite
    update to its two adjacent cells. At solid boundaries the face flux is
    zero (scalar cannot advect *into* or *out of* a solid cell).

    Replaces the earlier SL-trilinear scheme, which was diffusive and
    non-conservative -- it bled mass into solid-neighbour cells every step
    (the leak was laundered into the "degraded" diagnostic by the now-fixed
    ``max(0, ...)`` clamp in sample_scalars). CFL condition: |u|*dt/dx <= 1,
    comfortably satisfied at cfg.solver.cfl_safety_factor = 0.4.
    """
    i, j, k = wp.tid()

    if mat[i, j, k] != mat_fluid:
        C_water_new[i, j, k] = float(0.0)
        return

    c_self = C_water[i, j, k]
    d_C = float(0.0)
    alpha = dt / dx  # converts (u * C) [mg*m/kg/s] to ΔC [mg/kg] per face

    # ---- x direction ----
    # Left face at index i (velocity ux[i,j,k])
    u_L = ux[i, j, k]
    if i >= 1 and mat[i - 1, j, k] == mat_fluid:
        # Both sides are fluid -> real face flux.
        if u_L >= 0.0:
            flux_L = u_L * C_water[i - 1, j, k]  # flow from left neighbour into me
        else:
            flux_L = u_L * c_self  # flow from me into left neighbour (negative)
        d_C = d_C + alpha * flux_L  # face flux IN to this cell (positive if u_L > 0)
    # else: solid boundary -> zero flux (do nothing)

    # Right face at index i+1 (velocity ux[i+1,j,k])
    u_R = ux[i + 1, j, k]
    if i + 1 < mat.shape[0] and mat[i + 1, j, k] == mat_fluid:
        if u_R >= 0.0:
            flux_R = u_R * c_self   # flow from me to right neighbour
        else:
            flux_R = u_R * C_water[i + 1, j, k]  # flow from right neighbour into me
        d_C = d_C - alpha * flux_R  # face flux OUT of this cell

    # ---- y direction ----
    u_L = uy[i, j, k]
    if j >= 1 and mat[i, j - 1, k] == mat_fluid:
        if u_L >= 0.0:
            flux_L = u_L * C_water[i, j - 1, k]
        else:
            flux_L = u_L * c_self
        d_C = d_C + alpha * flux_L

    u_R = uy[i, j + 1, k]
    if j + 1 < mat.shape[1] and mat[i, j + 1, k] == mat_fluid:
        if u_R >= 0.0:
            flux_R = u_R * c_self
        else:
            flux_R = u_R * C_water[i, j + 1, k]
        d_C = d_C - alpha * flux_R

    # ---- z direction ----
    u_L = uz[i, j, k]
    if k >= 1 and mat[i, j, k - 1] == mat_fluid:
        if u_L >= 0.0:
            flux_L = u_L * C_water[i, j, k - 1]
        else:
            flux_L = u_L * c_self
        d_C = d_C + alpha * flux_L

    u_R = uz[i, j, k + 1]
    if k + 1 < mat.shape[2] and mat[i, j, k + 1] == mat_fluid:
        if u_R >= 0.0:
            flux_R = u_R * c_self
        else:
            flux_R = u_R * C_water[i, j, k + 1]
        d_C = d_C - alpha * flux_R

    C_water_new[i, j, k] = c_self + d_C


def step_advect_c_water(
    grid: "Grid",
    ws: NutrientWorkspace,
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Advance ``grid.C_water`` by one upwind flux-form advection step.

    Uses ``ws.C_water_tmp`` as the scratch destination then copies back into
    ``grid.C_water``. Called from :class:`Simulation.step` right after
    pressure_projection so the velocity field is divergence-free.
    """
    if grid.C_water is None:
        return
    from .geometry import MAT_FLUID as _MF

    nx, ny, nz = grid.shape
    wp.launch(
        advect_c_water,
        dim=(nx, ny, nz),
        inputs=[
            grid.C_water,
            ws.C_water_tmp,
            grid.ux,
            grid.uy,
            grid.uz,
            grid.mat,
            wp.vec3(*grid.origin),
            grid.dx,
            dt,
            _MF,
        ],
        device=device,
    )
    wp.copy(grid.C_water, ws.C_water_tmp)


@wp.kernel
def clamp_c_water_and_track_precipitation(
    C_water: wp.array3d(dtype=float),
    precipitated_total: wp.array(dtype=float),
    C_water_sat: float,
):
    """Clip each fluid cell's ``C_water`` to ``[0, C_water_sat]`` and
    accumulate the clipped mass into a single-cell global counter so the
    run-wide mass balance still closes.

    Why this exists. Upwind finite-volume advection is monotone (i.e.
    ``max(C_water)`` cannot grow) only when the transporting velocity is
    exactly divergence-free. Our pressure-projection Jacobi solver is run
    to ``pressure_tol = 1e-5`` (not machine precision), so every step
    leaves a small residual ``∇·u``. Over ~90k steps at ``dt ~ 1 ms`` this
    accumulates at stagnation cells (wall cul-de-sacs, bubble-plume
    convergence zones) and locally concentrates ``C_water`` 10-300x above
    ``C_water_sat``. The fluid-mean stays far below the cap -- this is a
    peak-pixel artefact, not a bulk mass issue -- but it violates the
    physical solubility limit that the leach kernel's per-face cap is
    supposed to enforce system-wide.

    Physical interpretation. When local concentration exceeds solubility,
    the solute precipitates out of solution (forms solid particles that
    sediment or cling to solids). This kernel models that as an
    irreversible sink: clipped mass leaves ``C_water`` and is accounted as
    ``precipitated_pct`` in the diagnostic triple. Mass balance remains
    exact: ``retention + leached + degraded + precipitated = 100 %%``.

    Also handles the rare ``C_water < 0`` case from float accumulation at
    near-zero cells; negative mass gets zeroed and its magnitude is added
    to the precipitated counter (a tiny positive sink).
    """
    i, j, k = wp.tid()
    c = C_water[i, j, k]
    if c > C_water_sat:
        wp.atomic_add(precipitated_total, 0, c - C_water_sat)
        C_water[i, j, k] = C_water_sat
    elif c < float(0.0):
        wp.atomic_add(precipitated_total, 0, -c)
        C_water[i, j, k] = float(0.0)


def step_clamp_c_water_sat(
    grid: "Grid",
    ws: NutrientWorkspace,
    cfg: "ScenarioConfig",
    device: str = "cuda:0",
) -> None:
    """Run :func:`clamp_c_water_and_track_precipitation` across the full
    grid. Idempotent after a cell is clipped (second pass is a no-op)."""
    if grid.C_water is None:
        return
    nx, ny, nz = grid.shape
    wp.launch(
        clamp_c_water_and_track_precipitation,
        dim=(nx, ny, nz),
        inputs=[
            grid.C_water,
            ws.precipitated_mass,
            cfg.nutrient.C_water_sat_mg_per_kg,
        ],
        device=device,
    )


def precipitated_fraction(
    grid: "Grid",
    ws: NutrientWorkspace,
    cfg: "ScenarioConfig",
) -> float:
    """Cumulative fraction of initial carrot beta-carotene mass that has
    been clipped out of ``C_water`` by the saturation clamp.

    Returns 0.0 when nutrient is disabled or no carrot cells exist.
    """
    if grid.C_water is None or ws is None:
        return 0.0
    mat_np = grid.mat.numpy()
    n_carrot = int((mat_np == _MAT_CARROT).sum())
    if n_carrot == 0:
        return 0.0
    total = float(ws.precipitated_mass.numpy()[0])
    initial = cfg.nutrient.C0_mg_per_kg * float(n_carrot)
    if initial <= 0.0:
        return 0.0
    return total / initial


def step_leach(
    grid: "Grid",
    cfg: "ScenarioConfig",
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Advance carrot and water-side concentrations by one Sherwood-flux step.

    Called from :class:`Simulation.step` after :func:`step_diffuse_nutrient`
    when ``cfg.nutrient.enabled``. Reads ``grid.ux/uy/uz`` (current velocity),
    writes both ``grid.C`` (carrot cells) and ``grid.C_water`` (fluid cells
    that touch the carrot surface).
    """
    if grid.C is None or grid.C_water is None:
        return
    from .geometry import MAT_CARROT as _MC, MAT_FLUID as _MF

    nx, ny, nz = grid.shape
    D_carrot = cfg.carrot.diameter_m
    wp.launch(
        leach_at_surface,
        dim=(nx, ny, nz),
        inputs=[
            grid.C,
            grid.C_water,
            grid.ux,
            grid.uy,
            grid.uz,
            grid.mat,
            grid.dx,
            dt,
            D_carrot,
            cfg.nutrient.nu_water_m2_per_s,
            cfg.nutrient.D_water_molec_m2_per_s,
            cfg.nutrient.K_partition,
            cfg.nutrient.C_water_sat_mg_per_kg,
            _MF,
            _MC,
        ],
        device=device,
    )


def step_diffuse_nutrient(
    grid: "Grid",
    ws: NutrientWorkspace,
    cfg: "ScenarioConfig",
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Advance ``grid.C`` by one explicit diffusion step.

    Uses ``ws.C_work`` as the scratch output buffer, then copies the result
    back into ``grid.C``. Cost: one kernel launch + one wp.copy per step
    (cudaMemcpyAsync inside Warp, negligible vs. the Laplacian kernel).

    Stability: asserts that ``dt`` satisfies ``dt <= dx^2/(6 D_eff)``.
    This should never fire at any reasonable pipeline dt; it's a guardrail
    against someone handing in a huge explicit dt by mistake.
    """
    if grid.C is None:
        return

    dt_max = diffusion_stability_dt(cfg, grid.dx)
    if dt > dt_max:
        raise RuntimeError(
            f"explicit nutrient diffusion: dt={dt:.3e} s exceeds stability "
            f"bound {dt_max:.3e} s (dx={grid.dx:.3e} m, "
            f"D_eff={cfg.nutrient.D_eff_m2_per_s:.3e} m^2/s)"
        )

    nx, ny, nz = grid.shape
    wp.launch(
        diffuse_nutrient_explicit,
        dim=(nx, ny, nz),
        inputs=[
            grid.C,
            ws.C_work,
            grid.mat,
            cfg.nutrient.D_eff_m2_per_s,
            grid.dx,
            dt,
            _MAT_CARROT,
        ],
        device=device,
    )
    wp.copy(grid.C, ws.C_work)


# ---------------------------------------------------------------------------
# Dual-solute extension: slot bundle + composite per-slot helpers
# ---------------------------------------------------------------------------


@dataclass
class SoluteSlot:
    """All arrays and parameters needed to evolve one solute's concentration
    pair (carrot-side ``C``, water-side ``C_water``) through the full Phase-4
    reaction-diffusion-leach-advect-clamp pipeline.

    Bundling these lets the pipeline call a single pair of per-slot helpers
    (:func:`_step_reaction_diffusion_leach`, :func:`_step_advect_clamp`) once
    per active solute rather than duplicating five step_* function calls.

    Fields mirror the arrays threaded through the single-solute step_*
    functions:

    * ``C``                  -- carrot-side concentration (``grid.C`` or ``grid.C2``).
    * ``C_water``            -- water-side passive scalar (``grid.C_water`` or ``grid.C_water2``).
    * ``C_work``             -- diffusion ping-pong scratch (from workspace).
    * ``C_water_tmp``        -- advection destination scratch (from workspace).
    * ``precipitated_mass``  -- length-1 cumulative clipped-mass counter.
    * ``cfg_nutrient``       -- the per-slot ``NutrientConfig`` (k0, E_a, D_eff,
                                K_partition, C_water_sat, nu_water,
                                D_water_molec, C0_mg_per_kg) for this solute.
    """

    C: Any
    C_water: Any
    C_work: Any
    C_water_tmp: Any
    precipitated_mass: Any
    cfg_nutrient: Any


def make_primary_slot(
    grid: "Grid",
    cfg: "ScenarioConfig",
    ws: NutrientWorkspace,
) -> SoluteSlot:
    """Build the primary :class:`SoluteSlot` from ``grid.C``/``grid.C_water``,
    the existing workspace scratch buffers, and ``cfg.nutrient``."""
    return SoluteSlot(
        C=grid.C,
        C_water=grid.C_water,
        C_work=ws.C_work,
        C_water_tmp=ws.C_water_tmp,
        precipitated_mass=ws.precipitated_mass,
        cfg_nutrient=cfg.nutrient,
    )


def make_secondary_slot(
    grid: "Grid",
    cfg: "ScenarioConfig",
    ws: NutrientWorkspace,
) -> SoluteSlot:
    """Build the secondary :class:`SoluteSlot` from ``grid.C2``/``grid.C_water2``
    and the secondary scratch buffers on the workspace."""
    return SoluteSlot(
        C=grid.C2,
        C_water=grid.C_water2,
        C_work=ws.C_work2,
        C_water_tmp=ws.C_water_tmp2,
        precipitated_mass=ws.precipitated_mass2,
        cfg_nutrient=cfg.nutrient2,
    )


def _step_reaction_diffusion_leach(
    slot: SoluteSlot,
    grid: "Grid",
    D_carrot: float,
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Arrhenius degrade (carrot + water) + in-carrot diffusion + Sherwood leach
    for a single solute slot. Same sequence as the single-solute pipeline's
    ``step_degrade + step_diffuse_nutrient + step_leach``, but reading from
    ``slot.*`` instead of the hard-coded primary arrays/cfg. ``D_carrot`` is
    ``cfg.carrot.diameter_m`` -- the shared geometry length scale that both
    solutes' Sherwood correlations use.
    """
    if slot.C is None:
        return
    from .geometry import MAT_FLUID as _MF

    nx, ny, nz = grid.shape
    cfg_n = slot.cfg_nutrient
    E_a_j_per_mol = cfg_n.E_a_kJ_per_mol * 1000.0
    E_a_over_R = E_a_j_per_mol / _R_GAS

    # --- Arrhenius degrade (carrot) ---
    wp.launch(
        arrhenius_degrade,
        dim=(nx, ny, nz),
        inputs=[slot.C, grid.T, grid.mat, cfg_n.k0_per_s, E_a_over_R, dt, _MAT_CARROT],
        device=device,
    )
    # --- Arrhenius degrade (water pool) ---
    if slot.C_water is not None:
        wp.launch(
            arrhenius_degrade_water,
            dim=(nx, ny, nz),
            inputs=[slot.C_water, grid.T, grid.mat, cfg_n.k0_per_s, E_a_over_R, dt, _MF],
            device=device,
        )

    # --- Explicit in-carrot diffusion (with stability guard) ---
    dt_max = _diffusion_stability_dt_D(grid.dx, cfg_n.D_eff_m2_per_s)
    if dt > dt_max:
        raise RuntimeError(
            f"explicit nutrient diffusion: dt={dt:.3e} s exceeds stability "
            f"bound {dt_max:.3e} s (dx={grid.dx:.3e} m, "
            f"D_eff={cfg_n.D_eff_m2_per_s:.3e} m^2/s)"
        )
    wp.launch(
        diffuse_nutrient_explicit,
        dim=(nx, ny, nz),
        inputs=[slot.C, slot.C_work, grid.mat, cfg_n.D_eff_m2_per_s, grid.dx, dt, _MAT_CARROT],
        device=device,
    )
    wp.copy(slot.C, slot.C_work)

    # --- Sherwood leach ---
    if slot.C_water is not None:
        wp.launch(
            leach_at_surface,
            dim=(nx, ny, nz),
            inputs=[
                slot.C,
                slot.C_water,
                grid.ux,
                grid.uy,
                grid.uz,
                grid.mat,
                grid.dx,
                dt,
                D_carrot,
                cfg_n.nu_water_m2_per_s,
                cfg_n.D_water_molec_m2_per_s,
                cfg_n.K_partition,
                cfg_n.C_water_sat_mg_per_kg,
                _MF,
                _MAT_CARROT,
            ],
            device=device,
        )


def _step_advect_clamp(
    slot: SoluteSlot,
    grid: "Grid",
    dt: float,
    device: str = "cuda:0",
) -> None:
    """Conservative upwind advection of ``slot.C_water`` + post-advect
    saturation clamp with precipitation accounting. Same as the single-solute
    ``step_advect_c_water + step_clamp_c_water_sat``.
    """
    if slot.C_water is None:
        return
    from .geometry import MAT_FLUID as _MF

    nx, ny, nz = grid.shape
    wp.launch(
        advect_c_water,
        dim=(nx, ny, nz),
        inputs=[
            slot.C_water,
            slot.C_water_tmp,
            grid.ux,
            grid.uy,
            grid.uz,
            grid.mat,
            wp.vec3(*grid.origin),
            grid.dx,
            dt,
            _MF,
        ],
        device=device,
    )
    wp.copy(slot.C_water, slot.C_water_tmp)

    wp.launch(
        clamp_c_water_and_track_precipitation,
        dim=(nx, ny, nz),
        inputs=[
            slot.C_water,
            slot.precipitated_mass,
            slot.cfg_nutrient.C_water_sat_mg_per_kg,
        ],
        device=device,
    )
