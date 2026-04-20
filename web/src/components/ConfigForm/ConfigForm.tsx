// Configuration form orchestrator. Owns the full ScenarioDraft
// state, renders the 9 Pydantic sections as collapsible cards, and
// batches a `set_config` + `start_run` pair on Apply. Server-side
// Pydantic validation errors land on `snapshot.last_error` and are
// surfaced in the sticky apply bar.

import { useState } from "react";

import type { ControlMessage, Snapshot } from "../../types/snapshot";
import { Button } from "../ui/Button";
import { Checkbox } from "../ui/Checkbox";
import { NumberInput } from "../ui/NumberInput";
import { Select } from "../ui/Select";
import { Slider } from "../ui/Slider";

import { FieldRow } from "./FieldRow";
import { Section } from "./Section";
import {
  DEFAULT_DRAFT,
  PRESETS,
  draftToScenarioJson,
  type MaterialName,
  type ScenarioDraft,
} from "./types";

interface Props {
  snapshot: Snapshot | null;
  sendCommand: (cmd: ControlMessage) => void;
  onApplied: () => void;
}

const MATERIAL_OPTIONS: { value: MaterialName; label: string }[] = [
  { value: "steel_304", label: "Stainless 304" },
  { value: "cast_iron", label: "Cast iron" },
  { value: "copper", label: "Copper" },
  { value: "aluminum", label: "Aluminium" },
];

const PRESET_OPTIONS = Object.entries(PRESETS).map(([value, { label }]) => ({
  value,
  label,
}));

export function ConfigForm({ snapshot, sendCommand, onApplied }: Props) {
  const [draft, setDraft] = useState<ScenarioDraft>(() =>
    structuredClone(DEFAULT_DRAFT),
  );
  const [applying, setApplying] = useState(false);
  const [presetKey, setPresetKey] = useState<string>("default");

  // Server-rejected configs land here via the Snapshot.last_error
  // field the Python producer sets after a failed Pydantic validate.
  const serverError = snapshot?.last_error ?? "";

  // Shallow helpers so the section renderers don't repeat spread
  // boilerplate for every field.
  const updatePot = (patch: Partial<ScenarioDraft["pot"]>) =>
    setDraft((d) => ({ ...d, pot: { ...d.pot, ...patch } }));
  const updateWater = (patch: Partial<ScenarioDraft["water"]>) =>
    setDraft((d) => ({ ...d, water: { ...d.water, ...patch } }));
  const updateCarrot = (patch: Partial<ScenarioDraft["carrot"]>) =>
    setDraft((d) => ({ ...d, carrot: { ...d.carrot, ...patch } }));
  const updateHeating = (patch: Partial<ScenarioDraft["heating"]>) =>
    setDraft((d) => ({ ...d, heating: { ...d.heating, ...patch } }));
  const updateGrid = (patch: Partial<ScenarioDraft["grid"]>) =>
    setDraft((d) => ({ ...d, grid: { ...d.grid, ...patch } }));
  const updateSolver = (patch: Partial<ScenarioDraft["solver"]>) =>
    setDraft((d) => ({ ...d, solver: { ...d.solver, ...patch } }));
  const updateBoiling = (patch: Partial<ScenarioDraft["boiling"]>) =>
    setDraft((d) => ({ ...d, boiling: { ...d.boiling, ...patch } }));
  const updateNutrient = (patch: Partial<ScenarioDraft["nutrient"]>) =>
    setDraft((d) => ({ ...d, nutrient: { ...d.nutrient, ...patch } }));
  const updateNutrient2 = (patch: Partial<ScenarioDraft["nutrient2"]>) =>
    setDraft((d) => ({ ...d, nutrient2: { ...d.nutrient2, ...patch } }));
  const updateSimulation = (patch: Partial<ScenarioDraft["simulation"]>) =>
    setDraft((d) => ({ ...d, simulation: { ...d.simulation, ...patch } }));

  const handleApply = () => {
    setApplying(true);
    try {
      sendCommand({ type: "set_config", config: draftToScenarioJson(draft) });
      sendCommand({ type: "start_run", duration_s: draft.simulation.total_time_s });
    } finally {
      // Tiny defer lets the outgoing WS frames ship before App.tsx
      // flips the page.
      window.setTimeout(() => {
        setApplying(false);
        onApplied();
      }, 120);
    }
  };

  const handleResetDefaults = () => {
    setDraft(structuredClone(DEFAULT_DRAFT));
    setPresetKey("default");
  };

  const handleLoadPreset = (key: string) => {
    const preset = PRESETS[key];
    if (!preset) return;
    setDraft(preset.draft());
    setPresetKey(key);
  };

  return (
    <div className="config-form">
      <header className="config-form__head">
        <div>
          <h2>Configuration</h2>
          <p>
            Edit every scenario field, then click <strong>Apply &amp; Start
            Run</strong>. The Python producer validates via Pydantic; any
            rejection shows up at the bottom of this page and the current
            simulation keeps running.
          </p>
        </div>
        <div className="config-form__preset-row">
          <span className="hint">Preset</span>
          <Select<string>
            ariaLabel="load scenario preset"
            value={presetKey}
            options={PRESET_OPTIONS}
            onChange={handleLoadPreset}
          />
        </div>
      </header>

      {/* --- Run (prominent first: duration is the most common edit) --- */}
      <Section title="Simulation" subtitle="duration + sampling">
        <FieldRow label="Total time" hint="Stops the run and writes artefacts at this simulated time. 0 = run forever.">
          <NumberInput
            label="t"
            ariaLabel="total simulation time in seconds"
            value={draft.simulation.total_time_s}
            min={0}
            max={36_000}
            step={10}
            unit="s"
            onCommit={(v) => updateSimulation({ total_time_s: v })}
          />
        </FieldRow>
        <FieldRow label="Output interval" hint="Only used by the offline HDF5 writer; dashboard uses snapshot-Hz independently.">
          <NumberInput
            label="dt"
            ariaLabel="HDF5 output interval in seconds"
            value={draft.simulation.output_every_s}
            min={0.01}
            max={60}
            step={0.01}
            unit="s"
            onCommit={(v) => updateSimulation({ output_every_s: v })}
          />
        </FieldRow>
      </Section>

      {/* --- Pot --- */}
      <Section title="Pot" subtitle="geometry + material">
        <FieldRow label="Material">
          <Select<MaterialName>
            ariaLabel="pot material"
            value={draft.pot.material}
            options={MATERIAL_OPTIONS}
            onChange={(v) => updatePot({ material: v })}
          />
        </FieldRow>
        <FieldRow label="Diameter">
          <NumberInput
            label="∅"
            ariaLabel="pot diameter in metres"
            value={draft.pot.diameter_m}
            min={0.05}
            max={0.50}
            step={0.005}
            unit="m"
            onCommit={(v) => updatePot({ diameter_m: v })}
          />
        </FieldRow>
        <FieldRow label="Height">
          <NumberInput
            label="h"
            ariaLabel="pot height in metres"
            value={draft.pot.height_m}
            min={0.05}
            max={0.40}
            step={0.005}
            unit="m"
            onCommit={(v) => updatePot({ height_m: v })}
          />
        </FieldRow>
        <FieldRow label="Wall thickness">
          <NumberInput
            label="w"
            ariaLabel="pot wall thickness in metres"
            value={draft.pot.wall_thickness_m}
            min={0.0005}
            max={0.02}
            step={0.0005}
            unit="m"
            onCommit={(v) => updatePot({ wall_thickness_m: v })}
          />
        </FieldRow>
        <FieldRow label="Base thickness">
          <NumberInput
            label="b"
            ariaLabel="pot base thickness in metres"
            value={draft.pot.base_thickness_m}
            min={0.001}
            max={0.03}
            step={0.001}
            unit="m"
            onCommit={(v) => updatePot({ base_thickness_m: v })}
          />
        </FieldRow>
      </Section>

      {/* --- Water --- */}
      <Section title="Water" subtitle="fill level + starting temperature">
        <FieldRow label="Fill fraction" hint="Fraction of the pot interior filled with water (0-1).">
          <Slider
            ariaLabel="water fill fraction"
            value={Math.round(draft.water.fill_fraction * 100)}
            min={10}
            max={100}
            step={5}
            onChange={(v) => updateWater({ fill_fraction: v / 100 })}
          />
          <span className="field-row__value">
            {(draft.water.fill_fraction * 100).toFixed(0)} %
          </span>
        </FieldRow>
        <FieldRow label="Initial temperature">
          <NumberInput
            label="T"
            ariaLabel="initial water temperature"
            value={draft.water.initial_temp_c}
            min={0}
            max={100}
            step={1}
            unit="°C"
            onCommit={(v) => updateWater({ initial_temp_c: v })}
          />
        </FieldRow>
      </Section>

      {/* --- Carrot --- */}
      <Section title="Carrot" subtitle="geometry + position">
        <FieldRow label="Diameter">
          <NumberInput
            label="∅"
            ariaLabel="carrot diameter in millimetres"
            value={draft.carrot.diameter_m * 1000}
            min={5}
            max={80}
            step={1}
            unit="mm"
            onCommit={(mm) => updateCarrot({ diameter_m: mm / 1000 })}
          />
        </FieldRow>
        <FieldRow label="Length">
          <NumberInput
            label="L"
            ariaLabel="carrot length in millimetres"
            value={draft.carrot.length_m * 1000}
            min={10}
            max={200}
            step={1}
            unit="mm"
            onCommit={(mm) => updateCarrot({ length_m: mm / 1000 })}
          />
        </FieldRow>
        <FieldRow label="Position Z">
          <NumberInput
            label="z"
            ariaLabel="carrot bottom z-coordinate in metres"
            value={draft.carrot.position[2]}
            min={0.005}
            max={0.15}
            step={0.005}
            unit="m"
            onCommit={(z) =>
              updateCarrot({
                position: [draft.carrot.position[0], draft.carrot.position[1], z],
              })
            }
          />
        </FieldRow>
        <FieldRow label="β-carotene C₀" hint="Reference loading; dashboard nutrient presets override this.">
          <NumberInput
            label="C"
            ariaLabel="initial beta-carotene loading"
            value={draft.carrot.initial_beta_carotene_mg_per_100g}
            min={0}
            max={20}
            step={0.1}
            unit="mg/100g"
            onCommit={(v) =>
              updateCarrot({ initial_beta_carotene_mg_per_100g: v })
            }
          />
        </FieldRow>
      </Section>

      {/* --- Heating --- */}
      <Section title="Heating" subtitle="stove flux + ambient">
        <FieldRow label="Base heat flux">
          <NumberInput
            label="q"
            ariaLabel="base heat flux in kilowatts per square metre"
            value={draft.heating.base_heat_flux_w_per_m2 / 1000}
            min={0}
            max={60}
            step={1}
            unit="kW/m²"
            onCommit={(kw) =>
              updateHeating({ base_heat_flux_w_per_m2: kw * 1000 })
            }
          />
        </FieldRow>
        <FieldRow label="Ambient">
          <NumberInput
            label="T"
            ariaLabel="ambient air temperature"
            value={draft.heating.ambient_temp_c}
            min={-20}
            max={60}
            step={1}
            unit="°C"
            onCommit={(v) => updateHeating({ ambient_temp_c: v })}
          />
        </FieldRow>
      </Section>

      {/* --- Grid --- */}
      <Section title="Grid" subtitle="spatial resolution" defaultOpen={false}>
        <FieldRow label="dx">
          <NumberInput
            label="dx"
            ariaLabel="grid spacing in millimetres"
            value={draft.grid.dx_m * 1000}
            min={0.5}
            max={10}
            step={0.5}
            unit="mm"
            onCommit={(mm) => updateGrid({ dx_m: mm / 1000 })}
          />
        </FieldRow>
        <FieldRow label="Carrot mesh resolution">
          <NumberInput
            label="n"
            ariaLabel="carrot surface mesh resolution"
            value={draft.grid.carrot_mesh_resolution}
            min={10}
            max={200}
            step={5}
            onCommit={(v) =>
              updateGrid({ carrot_mesh_resolution: Math.round(v) })
            }
          />
        </FieldRow>
      </Section>

      {/* --- Solver --- */}
      <Section title="Solver" subtitle="CFL, tolerances, sink coefficients" defaultOpen={false}>
        <FieldRow label="CFL safety">
          <Slider
            ariaLabel="cfl safety factor"
            value={Math.round(draft.solver.cfl_safety_factor * 100)}
            min={10}
            max={50}
            step={5}
            onChange={(v) => updateSolver({ cfl_safety_factor: v / 100 })}
          />
          <span className="field-row__value">
            {draft.solver.cfl_safety_factor.toFixed(2)}
          </span>
        </FieldRow>
        <FieldRow label="Max dt">
          <NumberInput
            label="dt"
            ariaLabel="max timestep in seconds"
            value={draft.solver.max_dt_s}
            min={0.001}
            max={1.0}
            step={0.01}
            unit="s"
            onCommit={(v) => updateSolver({ max_dt_s: v })}
          />
        </FieldRow>
        <FieldRow label="Pressure tol" hint="Jacobi convergence threshold.">
          <NumberInput
            label="ε"
            ariaLabel="pressure tolerance"
            value={draft.solver.pressure_tol}
            min={1e-8}
            max={1e-2}
            step={1e-6}
            onCommit={(v) => updateSolver({ pressure_tol: v })}
          />
        </FieldRow>
        <FieldRow label="Pressure max iters">
          <NumberInput
            label="N"
            ariaLabel="pressure max iterations"
            value={draft.solver.pressure_max_iter}
            min={10}
            max={1000}
            step={10}
            onCommit={(v) =>
              updateSolver({ pressure_max_iter: Math.round(v) })
            }
          />
        </FieldRow>
        <FieldRow label="Diffusion tol">
          <NumberInput
            label="ε"
            ariaLabel="diffusion tolerance"
            value={draft.solver.diffusion_tol}
            min={1e-8}
            max={1e-2}
            step={1e-5}
            onCommit={(v) => updateSolver({ diffusion_tol: v })}
          />
        </FieldRow>
        <FieldRow label="Diffusion max iters">
          <NumberInput
            label="N"
            ariaLabel="diffusion max iterations"
            value={draft.solver.diffusion_max_iter}
            min={5}
            max={200}
            step={1}
            onCommit={(v) =>
              updateSolver({ diffusion_max_iter: Math.round(v) })
            }
          />
        </FieldRow>
        <FieldRow label="h_conv outer" hint="Newton cooling on outside of the pot wall.">
          <NumberInput
            label="h"
            ariaLabel="outer convective coefficient"
            value={draft.solver.h_conv_outer_w_per_m2_k}
            min={0}
            max={500}
            step={1}
            unit="W/m²K"
            onCommit={(v) => updateSolver({ h_conv_outer_w_per_m2_k: v })}
          />
        </FieldRow>
        <FieldRow label="h_evap free surface" hint="Bigger values pin bulk water closer to T_sat.">
          <NumberInput
            label="h"
            ariaLabel="free-surface evaporation coefficient"
            value={draft.solver.h_evap_free_surface_w_per_m2_k}
            min={0}
            max={5e6}
            step={1e4}
            unit="W/m²K"
            onCommit={(v) =>
              updateSolver({ h_evap_free_surface_w_per_m2_k: v })
            }
          />
        </FieldRow>
        <FieldRow label="f_bulk_evap">
          <NumberInput
            label="f"
            ariaLabel="bulk evaporation fraction per second"
            value={draft.solver.f_bulk_evap_per_s}
            min={0}
            max={100}
            step={0.1}
            unit="/s"
            onCommit={(v) => updateSolver({ f_bulk_evap_per_s: v })}
          />
        </FieldRow>
        <FieldRow label="Implicit conduction">
          <Checkbox
            label="Use implicit conduction"
            hint="Backward-Euler; unconditionally stable, slower per step."
            checked={draft.solver.use_implicit_conduction}
            onChange={(checked) =>
              updateSolver({ use_implicit_conduction: checked })
            }
          />
        </FieldRow>
      </Section>

      {/* --- Boiling --- */}
      <Section title="Boiling" subtitle="bubble pool + Rohsenow" defaultOpen={false}>
        <FieldRow label="Enabled">
          <Checkbox
            label="Nucleate boiling active"
            checked={draft.boiling.enabled}
            onChange={(checked) => updateBoiling({ enabled: checked })}
          />
        </FieldRow>
        <FieldRow label="ONB superheat">
          <NumberInput
            label="ΔT"
            ariaLabel="nucleation onset superheat"
            value={draft.boiling.dT_onb_k}
            min={0.1}
            max={30}
            step={0.5}
            unit="K"
            onCommit={(v) => updateBoiling({ dT_onb_k: v })}
          />
        </FieldRow>
        <FieldRow label="Contact angle">
          <NumberInput
            label="θ"
            ariaLabel="contact angle in radians"
            value={draft.boiling.contact_angle_rad}
            min={0.1}
            max={3.14}
            step={0.05}
            unit="rad"
            onCommit={(v) => updateBoiling({ contact_angle_rad: v })}
          />
        </FieldRow>
        <FieldRow label="Max bubbles">
          <NumberInput
            label="N"
            ariaLabel="max bubble pool size"
            value={draft.boiling.max_bubbles}
            min={100}
            max={1_000_000}
            step={1000}
            onCommit={(v) => updateBoiling({ max_bubbles: Math.round(v) })}
          />
        </FieldRow>
        <FieldRow label="Initial radius">
          <NumberInput
            label="r"
            ariaLabel="initial bubble radius in micrometres"
            value={draft.boiling.initial_bubble_radius_m * 1e6}
            min={1}
            max={1000}
            step={1}
            unit="µm"
            onCommit={(um) =>
              updateBoiling({ initial_bubble_radius_m: um * 1e-6 })
            }
          />
        </FieldRow>
        <FieldRow label="Nucleation probability">
          <NumberInput
            label="p"
            ariaLabel="nucleation probability per step"
            value={draft.boiling.nucleation_probability_per_step}
            min={0}
            max={1}
            step={0.01}
            onCommit={(v) =>
              updateBoiling({ nucleation_probability_per_step: v })
            }
          />
        </FieldRow>
        <FieldRow label="C_sf Rohsenow">
          <NumberInput
            label="C"
            ariaLabel="rohsenow surface factor"
            value={draft.boiling.C_sf_rohsenow}
            min={0.001}
            max={0.1}
            step={0.001}
            onCommit={(v) => updateBoiling({ C_sf_rohsenow: v })}
          />
        </FieldRow>
        <FieldRow label="Pr_n Rohsenow">
          <NumberInput
            label="n"
            ariaLabel="rohsenow prandtl exponent"
            value={draft.boiling.Pr_n_rohsenow}
            min={0.5}
            max={2.0}
            step={0.1}
            onCommit={(v) => updateBoiling({ Pr_n_rohsenow: v })}
          />
        </FieldRow>
      </Section>

      {/* --- Nutrient (primary) --- */}
      <Section title="Nutrient — primary" subtitle="β-carotene / vitamin C / etc.">
        <FieldRow label="Enabled">
          <Checkbox
            label="Track primary solute"
            checked={draft.nutrient.enabled}
            onChange={(checked) => updateNutrient({ enabled: checked })}
          />
        </FieldRow>
        <FieldRow label="E_a">
          <NumberInput
            label="E"
            ariaLabel="activation energy"
            value={draft.nutrient.E_a_kJ_per_mol}
            min={10}
            max={200}
            step={1}
            unit="kJ/mol"
            onCommit={(v) => updateNutrient({ E_a_kJ_per_mol: v })}
          />
        </FieldRow>
        <FieldRow label="k₀">
          <NumberInput
            label="k"
            ariaLabel="Arrhenius prefactor"
            value={draft.nutrient.k0_per_s}
            min={1}
            max={1e12}
            step={1e5}
            unit="/s"
            onCommit={(v) => updateNutrient({ k0_per_s: v })}
          />
        </FieldRow>
        <FieldRow label="D_eff">
          <NumberInput
            label="D"
            ariaLabel="effective diffusivity in carrot"
            value={draft.nutrient.D_eff_m2_per_s}
            min={1e-12}
            max={1e-7}
            step={1e-11}
            unit="m²/s"
            onCommit={(v) => updateNutrient({ D_eff_m2_per_s: v })}
          />
        </FieldRow>
        <FieldRow label="K partition" hint="≈1e-5 for β-carotene, ≈1.0 for water-soluble.">
          <NumberInput
            label="K"
            ariaLabel="partition coefficient"
            value={draft.nutrient.K_partition}
            min={1e-7}
            max={10}
            step={0.001}
            onCommit={(v) => updateNutrient({ K_partition: v })}
          />
        </FieldRow>
        <FieldRow label="C_water sat">
          <NumberInput
            label="C"
            ariaLabel="water-side saturation concentration"
            value={draft.nutrient.C_water_sat_mg_per_kg}
            min={1e-4}
            max={1e8}
            step={1}
            unit="mg/kg"
            onCommit={(v) => updateNutrient({ C_water_sat_mg_per_kg: v })}
          />
        </FieldRow>
        <FieldRow label="C₀">
          <NumberInput
            label="C"
            ariaLabel="initial carrot concentration"
            value={draft.nutrient.C0_mg_per_kg}
            min={0}
            max={1e4}
            step={1}
            unit="mg/kg"
            onCommit={(v) => updateNutrient({ C0_mg_per_kg: v })}
          />
        </FieldRow>
      </Section>

      {/* --- Nutrient 2 (secondary, opt-in) --- */}
      <Section
        title="Nutrient — secondary"
        subtitle="dual-solute track"
        defaultOpen={false}
      >
        <FieldRow label="Enabled">
          <Checkbox
            label="Track a second solute concurrently"
            hint="Requires primary nutrient enabled."
            checked={draft.nutrient2.enabled}
            onChange={(checked) => updateNutrient2({ enabled: checked })}
          />
        </FieldRow>
        <FieldRow label="E_a">
          <NumberInput
            label="E"
            ariaLabel="secondary activation energy"
            value={draft.nutrient2.E_a_kJ_per_mol}
            min={10}
            max={200}
            step={1}
            unit="kJ/mol"
            onCommit={(v) => updateNutrient2({ E_a_kJ_per_mol: v })}
          />
        </FieldRow>
        <FieldRow label="k₀">
          <NumberInput
            label="k"
            ariaLabel="secondary Arrhenius prefactor"
            value={draft.nutrient2.k0_per_s}
            min={1}
            max={1e12}
            step={1e5}
            unit="/s"
            onCommit={(v) => updateNutrient2({ k0_per_s: v })}
          />
        </FieldRow>
        <FieldRow label="K partition">
          <NumberInput
            label="K"
            ariaLabel="secondary partition coefficient"
            value={draft.nutrient2.K_partition}
            min={1e-7}
            max={10}
            step={0.001}
            onCommit={(v) => updateNutrient2({ K_partition: v })}
          />
        </FieldRow>
        <FieldRow label="C₀">
          <NumberInput
            label="C"
            ariaLabel="secondary initial concentration"
            value={draft.nutrient2.C0_mg_per_kg}
            min={0}
            max={1e4}
            step={1}
            unit="mg/kg"
            onCommit={(v) => updateNutrient2({ C0_mg_per_kg: v })}
          />
        </FieldRow>
      </Section>

      {/* --- Sticky apply bar --- */}
      <div className="apply-bar">
        {serverError ? (
          <div className="apply-bar__error" role="alert">
            {serverError}
          </div>
        ) : (
          <div className="apply-bar__status">
            Ready. Apply runs through Pydantic and rebuilds the sim at
            <code> t = 0</code>.
          </div>
        )}
        <Button variant="ghost" onClick={handleResetDefaults}>
          Reset to defaults
        </Button>
        <Button
          variant="primary"
          disabled={applying}
          onClick={handleApply}
        >
          {applying ? "Applying…" : "Apply & Start Run"}
        </Button>
      </div>
    </div>
  );
}
