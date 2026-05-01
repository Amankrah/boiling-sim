// Configuration form orchestrator. Owns the full ScenarioDraft
// state, renders the 9 Pydantic sections as collapsible cards, and
// batches a `set_config` + `start_run` pair on Apply. Server-side
// Pydantic validation errors land on `snapshot.last_error` and are
// surfaced in the sticky apply bar.

import { useState } from "react";

import type { ControlMessage, Snapshot } from "../../types/snapshot";
import { Button } from "../ui/Button";
import { NumberInput } from "../ui/NumberInput";
import { Select } from "../ui/Select";
import { Slider } from "../ui/Slider";

import { FieldRow } from "./FieldRow";
import { Section } from "./Section";
import {
  DEFAULT_DRAFT,
  PRESETS,
  draftToScenarioJson,
  nutrientsToSoluteKey,
  soluteKeyToNutrients,
  type InitialConditionsMode,
  type MaterialName,
  type ScenarioDraft,
  type SoluteKey,
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

const INITIAL_CONDITIONS_OPTIONS: { value: InitialConditionsMode; label: string }[] = [
  { value: "cold", label: "Cold start (use water temperature below)" },
  { value: "preheat", label: "Preheated (skip the 5–10 min warming transient)" },
];

const SOLUTE_OPTIONS: { value: SoluteKey; label: string }[] = [
  { value: "off", label: "Off (no nutrient tracking)" },
  { value: "beta_carotene", label: "β-carotene (lipophilic, degradation-dominated)" },
  { value: "vitamin_c", label: "Vitamin C (water-soluble, leach-dominated)" },
  { value: "both", label: "β-carotene + vitamin C (dual solute)" },
];

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
  const updateInitialConditions = (
    patch: Partial<ScenarioDraft["initial_conditions"]>,
  ) =>
    setDraft((d) => ({
      ...d,
      initial_conditions: { ...d.initial_conditions, ...patch },
    }));
  const updateGrid = (patch: Partial<ScenarioDraft["grid"]>) =>
    setDraft((d) => ({ ...d, grid: { ...d.grid, ...patch } }));
  const updateSimulation = (patch: Partial<ScenarioDraft["simulation"]>) =>
    setDraft((d) => ({ ...d, simulation: { ...d.simulation, ...patch } }));

  const handleSoluteChange = (key: SoluteKey) => {
    const { nutrient, nutrient2 } = soluteKeyToNutrients(key);
    setDraft((d) => ({ ...d, nutrient, nutrient2 }));
  };
  const currentSolute: SoluteKey = nutrientsToSoluteKey(
    draft.nutrient,
    draft.nutrient2,
  );

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
            Set the experiment knobs, then click <strong>Apply &amp; Start
            Run</strong>. Kinetic constants (Arrhenius, Rohsenow, partition
            coefficients, solver tolerances) load from literature values tied
            to the chosen material and solute — override them by dropping a
            YAML into <code>configs/scenarios/</code>. Server-side Pydantic
            rejections surface at the bottom of this page.
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
      <Section title="Simulation" subtitle="how long to run">
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
      <Section title="Carrot" subtitle="geometry + count + orientation">
        <FieldRow label="Count">
          <NumberInput
            label="N"
            ariaLabel="number of carrot instances"
            value={draft.carrot.count}
            min={1}
            max={64}
            step={1}
            unit=""
            onCommit={(n) => updateCarrot({ count: Math.round(n) })}
          />
        </FieldRow>
        <FieldRow label="Axis">
          <Select
            ariaLabel="carrot cylinder axis"
            value={draft.carrot.axis}
            options={[
              { value: "x", label: "x (horizontal)" },
              { value: "y", label: "y (horizontal)" },
              { value: "z", label: "z (vertical, legacy)" },
            ]}
            onChange={(axis) => updateCarrot({ axis })}
          />
        </FieldRow>
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
        <FieldRow label="Total mass">
          {/* Derived (count · π · (d/2)² · L · ρ_carrot, ρ ≈ 1040 kg/m³).
              Displayed live so users know "I'm cooking N grams". */}
          <span className="text-secondary" aria-live="polite">
            {(
              draft.carrot.count *
              Math.PI *
              Math.pow(draft.carrot.diameter_m / 2, 2) *
              draft.carrot.length_m *
              1040 *
              1000
            ).toFixed(1)}{" "}
            g
          </span>
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

      {/* --- Initial conditions --- */}
      <Section
        title="Initial conditions"
        subtitle="how the pot starts at t = 0"
      >
        <FieldRow
          label="Start state"
          hint="Cold start simulates filling a room-temperature pot and turning the stove on. Preheated skips the 5–10 min warming transient — useful for focusing on boiling and nutrient kinetics."
        >
          <Select<InitialConditionsMode>
            ariaLabel="initial conditions mode"
            value={draft.initial_conditions.mode}
            options={INITIAL_CONDITIONS_OPTIONS}
            onChange={(v) => updateInitialConditions({ mode: v })}
          />
        </FieldRow>
        {draft.initial_conditions.mode === "preheat" && (
          <>
            <FieldRow label="Preheat water" hint="Typical 95 °C: near saturation but still sub-cooled.">
              <NumberInput
                label="T"
                ariaLabel="preheat water temperature"
                value={draft.initial_conditions.preheat_water_c}
                min={0}
                max={105}
                step={1}
                unit="°C"
                onCommit={(v) =>
                  updateInitialConditions({ preheat_water_c: v })
                }
              />
            </FieldRow>
            <FieldRow label="Preheat pot wall">
              <NumberInput
                label="T"
                ariaLabel="preheat pot wall temperature"
                value={draft.initial_conditions.preheat_wall_c}
                min={0}
                max={120}
                step={1}
                unit="°C"
                onCommit={(v) =>
                  updateInitialConditions({ preheat_wall_c: v })
                }
              />
            </FieldRow>
            <FieldRow label="Preheat carrot">
              <NumberInput
                label="T"
                ariaLabel="preheat carrot temperature"
                value={draft.initial_conditions.preheat_carrot_c}
                min={0}
                max={100}
                step={1}
                unit="°C"
                onCommit={(v) =>
                  updateInitialConditions({ preheat_carrot_c: v })
                }
              />
            </FieldRow>
          </>
        )}
      </Section>

      {/* --- Solute (preset-driven; physics constants hidden) --- */}
      <Section
        title="Solute"
        subtitle="which nutrient(s) to track"
      >
        <FieldRow
          label="Solute"
          hint="Kinetic constants (E_a, k₀, D_eff, K_partition, C_water_sat) load from literature values for the chosen solute. Drop a custom YAML into configs/scenarios/ to override."
        >
          <Select<SoluteKey>
            ariaLabel="solute preset"
            value={currentSolute}
            options={SOLUTE_OPTIONS}
            onChange={handleSoluteChange}
          />
        </FieldRow>
      </Section>

      {/* --- Advanced --- */}
      <Section
        title="Advanced"
        subtitle="pot construction + grid resolution — change only if you know why"
        defaultOpen={false}
      >
        <FieldRow label="Pot wall thickness">
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
        <FieldRow label="Pot base thickness">
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
        <FieldRow label="Grid dx" hint="2 mm is the dev tier used across Phase-2/3/4 benchmarks; finer dx → sharper boundary layers at higher cost.">
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
        <FieldRow label="Output interval" hint="Only affects offline HDF5; the dashboard stream runs at its own snapshot cadence.">
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
