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
  makeBlankCoupling,
  makeBlankIngredient,
  makeBlankNutrient,
  nutrientsToSoluteKey,
  soluteKeyToNutrients,
  type CouplingDraft,
  type ExtraIngredientDraft,
  type InitialConditionsMode,
  type MaterialName,
  type NamedNutrientDraft,
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

  // M7: extras + couplings list helpers. Mutators clone the array so
  // React's identity-based change detection picks up updates.
  const addExtra = () =>
    setDraft((d) => ({
      ...d,
      extra_ingredients: [...d.extra_ingredients, makeBlankIngredient()],
    }));
  const removeExtra = (idx: number) =>
    setDraft((d) => ({
      ...d,
      extra_ingredients: d.extra_ingredients.filter((_, i) => i !== idx),
    }));
  const updateExtra = (
    idx: number,
    patch: Partial<ExtraIngredientDraft>,
  ) =>
    setDraft((d) => ({
      ...d,
      extra_ingredients: d.extra_ingredients.map((e, i) =>
        i === idx ? { ...e, ...patch } : e,
      ),
    }));
  const addNutrientToExtra = (extraIdx: number) =>
    setDraft((d) => ({
      ...d,
      extra_ingredients: d.extra_ingredients.map((e, i) =>
        i === extraIdx
          ? { ...e, nutrients: [...e.nutrients, makeBlankNutrient()] }
          : e,
      ),
    }));
  const removeNutrientFromExtra = (extraIdx: number, nutIdx: number) =>
    setDraft((d) => ({
      ...d,
      extra_ingredients: d.extra_ingredients.map((e, i) =>
        i === extraIdx
          ? { ...e, nutrients: e.nutrients.filter((_, j) => j !== nutIdx) }
          : e,
      ),
    }));
  const updateNutrientOnExtra = (
    extraIdx: number,
    nutIdx: number,
    patch: Partial<NamedNutrientDraft>,
  ) =>
    setDraft((d) => ({
      ...d,
      extra_ingredients: d.extra_ingredients.map((e, i) =>
        i === extraIdx
          ? {
              ...e,
              nutrients: e.nutrients.map((n, j) =>
                j === nutIdx ? { ...n, ...patch } : n,
              ),
            }
          : e,
      ),
    }));
  const addCoupling = () =>
    setDraft((d) => ({ ...d, couplings: [...d.couplings, makeBlankCoupling()] }));
  const removeCoupling = (idx: number) =>
    setDraft((d) => ({
      ...d,
      couplings: d.couplings.filter((_, i) => i !== idx),
    }));
  const updateCoupling = (idx: number, patch: Partial<CouplingDraft>) =>
    setDraft((d) => ({
      ...d,
      couplings: d.couplings.map((c, i) => (i === idx ? { ...c, ...patch } : c)),
    }));

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
        <FieldRow label="Quantity input">
          {/* M2: choose how the user expresses size. "dimensions" lets
              the user set Length directly (legacy); "mass" makes Length
              read-only and derives it from a target gram amount. */}
          <Select
            ariaLabel="quantity input mode"
            value={draft.carrot.mass_mode}
            options={[
              { value: "dimensions", label: "Specify dimensions" },
              { value: "mass", label: "Specify mass (g)" },
            ]}
            onChange={(mode) => {
              if (mode === "mass") {
                // Seed target_mass_g from the current derived total so
                // the switch is non-destructive.
                const currentMassG =
                  draft.carrot.count *
                  Math.PI *
                  Math.pow(draft.carrot.diameter_m / 2, 2) *
                  draft.carrot.length_m *
                  1040 *
                  1000;
                updateCarrot({
                  mass_mode: "mass",
                  target_mass_g: Math.round(currentMassG * 10) / 10,
                });
              } else {
                updateCarrot({ mass_mode: "dimensions", target_mass_g: null });
              }
            }}
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
        {draft.carrot.mass_mode === "mass" ? (
          <FieldRow label="Target mass">
            <NumberInput
              label="m"
              ariaLabel="target carrot mass in grams"
              value={draft.carrot.target_mass_g ?? 0}
              min={5}
              max={5000}
              step={10}
              unit="g"
              onCommit={(g) => updateCarrot({ target_mass_g: g })}
            />
          </FieldRow>
        ) : (
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
        )}
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
        <FieldRow
          label={
            draft.carrot.mass_mode === "mass"
              ? "Derived length"
              : "Total mass"
          }
        >
          {/* In dimensions mode this shows derived total mass; in mass
              mode it shows the derived per-instance length so users see
              both sides of the round-trip. */}
          <span className="text-secondary" aria-live="polite">
            {draft.carrot.mass_mode === "mass" ? (
              <>
                {(() => {
                  const targetKg = (draft.carrot.target_mass_g ?? 0) / 1000;
                  const r = draft.carrot.diameter_m / 2;
                  const perCarrotKg = targetKg / Math.max(1, draft.carrot.count);
                  const lengthM = perCarrotKg / 1040 / (Math.PI * r * r);
                  return (lengthM * 1000).toFixed(1);
                })()}{" "}
                mm
              </>
            ) : (
              <>
                {(
                  draft.carrot.count *
                  Math.PI *
                  Math.pow(draft.carrot.diameter_m / 2, 2) *
                  draft.carrot.length_m *
                  1040 *
                  1000
                ).toFixed(1)}{" "}
                g
              </>
            )}
          </span>
        </FieldRow>
      </Section>

      {/* --- M7: Extra Ingredients --- */}
      <Section
        title="Extra Ingredients"
        subtitle={
          draft.extra_ingredients.length === 0
            ? "(none) — add a potato, onion, etc. with its own nutrients"
            : `${draft.extra_ingredients.length} extra ingredient${draft.extra_ingredients.length === 1 ? "" : "s"}`
        }
      >
        {draft.extra_ingredients.map((extra, idx) => (
          <ExtraIngredientCard
            key={`extra-${idx}`}
            extra={extra}
            extraIdx={idx}
            onUpdate={(patch) => updateExtra(idx, patch)}
            onRemove={() => removeExtra(idx)}
            onAddNutrient={() => addNutrientToExtra(idx)}
            onRemoveNutrient={(nutIdx) => removeNutrientFromExtra(idx, nutIdx)}
            onUpdateNutrient={(nutIdx, patch) =>
              updateNutrientOnExtra(idx, nutIdx, patch)
            }
          />
        ))}
        <FieldRow label="Add">
          <Button onClick={addExtra}>+ Add ingredient</Button>
        </FieldRow>
      </Section>

      {/* --- M7: Nutrient-Nutrient Couplings --- */}
      <Section
        title="Couplings"
        subtitle={
          draft.couplings.length === 0
            ? "(none) — e.g. vitamin C protects β-carotene"
            : `${draft.couplings.length} active coupling${draft.couplings.length === 1 ? "" : "s"}`
        }
      >
        {draft.couplings.map((coupling, idx) => (
          <CouplingCard
            key={`coupling-${idx}`}
            coupling={coupling}
            onUpdate={(patch) => updateCoupling(idx, patch)}
            onRemove={() => removeCoupling(idx)}
          />
        ))}
        <FieldRow label="Add">
          <Button onClick={addCoupling}>+ Add coupling</Button>
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


/* ---------------------------------------------------------------------- */
/* M7 sub-components                                                       */
/* ---------------------------------------------------------------------- */

interface ExtraIngredientCardProps {
  extra: ExtraIngredientDraft;
  extraIdx: number;
  onUpdate: (patch: Partial<ExtraIngredientDraft>) => void;
  onRemove: () => void;
  onAddNutrient: () => void;
  onRemoveNutrient: (nutIdx: number) => void;
  onUpdateNutrient: (nutIdx: number, patch: Partial<NamedNutrientDraft>) => void;
}

/** One extra-ingredient card. Geometry knobs at top, nutrient sub-list
 *  below. Nutrient cards each carry a name + the kinetic params. */
function ExtraIngredientCard({
  extra,
  extraIdx,
  onUpdate,
  onRemove,
  onAddNutrient,
  onRemoveNutrient,
  onUpdateNutrient,
}: ExtraIngredientCardProps) {
  return (
    <div className="extra-ingredient-card">
      <FieldRow label={`Ingredient ${extraIdx + 1}`}>
        <input
          className="text-input"
          aria-label="ingredient name"
          type="text"
          value={extra.name}
          onChange={(e) => onUpdate({ name: e.target.value })}
        />
        <Button onClick={onRemove}>Remove</Button>
      </FieldRow>
      <FieldRow label="Count">
        <NumberInput
          label="N"
          ariaLabel="extra ingredient count"
          value={extra.count}
          min={1}
          max={64}
          step={1}
          unit=""
          onCommit={(n) => onUpdate({ count: Math.round(n) })}
        />
      </FieldRow>
      <FieldRow label="Axis">
        <Select
          ariaLabel="extra ingredient cylinder axis"
          value={extra.axis}
          options={[
            { value: "x", label: "x (horizontal)" },
            { value: "y", label: "y (horizontal)" },
            { value: "z", label: "z (vertical)" },
          ]}
          onChange={(axis) => onUpdate({ axis })}
        />
      </FieldRow>
      <FieldRow label="Diameter">
        <NumberInput
          label="∅"
          ariaLabel="extra ingredient diameter in mm"
          value={extra.diameter_m * 1000}
          min={5}
          max={80}
          step={1}
          unit="mm"
          onCommit={(mm) => onUpdate({ diameter_m: mm / 1000 })}
        />
      </FieldRow>
      <FieldRow label="Length">
        <NumberInput
          label="L"
          ariaLabel="extra ingredient length in mm"
          value={extra.length_m * 1000}
          min={10}
          max={200}
          step={1}
          unit="mm"
          onCommit={(mm) => onUpdate({ length_m: mm / 1000 })}
        />
      </FieldRow>
      <FieldRow label="Density">
        <NumberInput
          label="ρ"
          ariaLabel="ingredient tissue density in kg per cubic metre"
          value={extra.density_kg_per_m3}
          min={500}
          max={1500}
          step={10}
          unit="kg/m³"
          onCommit={(v) => onUpdate({ density_kg_per_m3: v })}
        />
      </FieldRow>
      <FieldRow label="Position Z">
        <NumberInput
          label="z"
          ariaLabel="extra ingredient z coordinate in metres"
          value={extra.position[2]}
          min={0.005}
          max={0.15}
          step={0.005}
          unit="m"
          onCommit={(z) =>
            onUpdate({ position: [extra.position[0], extra.position[1], z] })
          }
        />
      </FieldRow>

      {/* Nutrients sub-list */}
      <div className="extra-ingredient-card__nutrients">
        <FieldRow
          label={`Nutrients (${extra.nutrients.length})`}
        >
          <Button onClick={onAddNutrient}>+ Add nutrient</Button>
        </FieldRow>
        {extra.nutrients.map((nut, nutIdx) => (
          <NutrientCard
            key={`extra-${extraIdx}-nut-${nutIdx}`}
            nut={nut}
            onUpdate={(patch) => onUpdateNutrient(nutIdx, patch)}
            onRemove={() => onRemoveNutrient(nutIdx)}
          />
        ))}
      </div>
    </div>
  );
}


interface NutrientCardProps {
  nut: NamedNutrientDraft;
  onUpdate: (patch: Partial<NamedNutrientDraft>) => void;
  onRemove: () => void;
}

/** One nutrient card inside an extra ingredient. Exposes the user-
 *  tunable kinetic params; everything else (nu_water, D_water_molec)
 *  defaults from makeBlankNutrient(). */
function NutrientCard({ nut, onUpdate, onRemove }: NutrientCardProps) {
  return (
    <div className="nutrient-card">
      <FieldRow label="Name">
        <input
          className="text-input"
          aria-label="nutrient name"
          type="text"
          value={nut.name}
          placeholder="e.g. starch, beta_carotene"
          onChange={(e) => onUpdate({ name: e.target.value })}
        />
        <Button onClick={onRemove}>Remove</Button>
      </FieldRow>
      <FieldRow label="Enabled">
        <Select
          ariaLabel="nutrient enabled"
          value={nut.enabled ? "on" : "off"}
          options={[
            { value: "on", label: "On" },
            { value: "off", label: "Off" },
          ]}
          onChange={(v) => onUpdate({ enabled: v === "on" })}
        />
      </FieldRow>
      <FieldRow label="C₀ (mg/kg)">
        <NumberInput
          label="C₀"
          ariaLabel="initial nutrient concentration mg per kg ingredient"
          value={nut.C0_mg_per_kg}
          min={0}
          max={1.0e6}
          step={1}
          unit="mg/kg"
          onCommit={(v) => onUpdate({ C0_mg_per_kg: v })}
        />
      </FieldRow>
      <FieldRow label="K_partition">
        <NumberInput
          label="K"
          ariaLabel="partition coefficient water over ingredient"
          value={nut.K_partition}
          min={1.0e-7}
          max={10.0}
          step={1.0e-3}
          unit=""
          onCommit={(v) => onUpdate({ K_partition: v })}
        />
      </FieldRow>
      <FieldRow label="E_a (kJ/mol)">
        <NumberInput
          label="Eₐ"
          ariaLabel="Arrhenius activation energy"
          value={nut.E_a_kJ_per_mol}
          min={10}
          max={200}
          step={1}
          unit="kJ/mol"
          onCommit={(v) => onUpdate({ E_a_kJ_per_mol: v })}
        />
      </FieldRow>
    </div>
  );
}


interface CouplingCardProps {
  coupling: CouplingDraft;
  onUpdate: (patch: Partial<CouplingDraft>) => void;
  onRemove: () => void;
}

function CouplingCard({ coupling, onUpdate, onRemove }: CouplingCardProps) {
  return (
    <div className="coupling-card">
      <FieldRow label="Protector">
        <input
          className="text-input"
          aria-label="protector ingredient.nutrient identifier"
          type="text"
          value={coupling.protector}
          placeholder="e.g. carrot.vitamin_c"
          onChange={(e) => onUpdate({ protector: e.target.value })}
        />
        <Button onClick={onRemove}>Remove</Button>
      </FieldRow>
      <FieldRow label="Protected">
        <input
          className="text-input"
          aria-label="protected ingredient.nutrient identifier"
          type="text"
          value={coupling.protected}
          placeholder="e.g. carrot.beta_carotene"
          onChange={(e) => onUpdate({ protected: e.target.value })}
        />
      </FieldRow>
      <FieldRow label="Enabled">
        <Select
          ariaLabel="coupling enabled"
          value={coupling.enabled ? "on" : "off"}
          options={[
            { value: "on", label: "On" },
            { value: "off", label: "Off" },
          ]}
          onChange={(v) => onUpdate({ enabled: v === "on" })}
        />
      </FieldRow>
      <FieldRow label="η (slope)">
        <NumberInput
          label="η"
          ariaLabel="protection slope eta"
          value={coupling.eta}
          min={0}
          max={5}
          step={0.05}
          unit=""
          onCommit={(v) => onUpdate({ eta: v })}
        />
      </FieldRow>
      <FieldRow label="C_ref (mg/kg)">
        <NumberInput
          label="C_ref"
          ariaLabel="reference protector concentration"
          value={coupling.c_ref_mg_per_kg}
          min={0.1}
          max={1.0e4}
          step={0.5}
          unit="mg/kg"
          onCommit={(v) => onUpdate({ c_ref_mg_per_kg: v })}
        />
      </FieldRow>
      <FieldRow label="η_max">
        <NumberInput
          label="η_max"
          ariaLabel="protection cap"
          value={coupling.eta_max}
          min={0}
          max={0.99}
          step={0.05}
          unit=""
          onCommit={(v) => onUpdate({ eta_max: v })}
        />
      </FieldRow>
    </div>
  );
}
