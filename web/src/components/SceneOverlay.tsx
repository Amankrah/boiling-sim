// Top-right telemetry card over the scene. Primary hero metric is
// the retention percentage of whichever nutrient the sim is tracking
// (labelled with the solute's display name). Below it: the full
// Phase-4 four-bucket partition (retention + leached + degraded +
// precipitated), which sums to ~100 % and is the validated
// mass-balance invariant. When the secondary solute is active, its
// partition sits below the primary's in a second mini-table.

import type { Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot;
  /** Layout mode. Default "overlay" keeps the legacy glassmorphic
   *  card floating over the 3D scene; "sidebar" strips the absolute
   *  positioning + backdrop blur so the card drops into the Live
   *  page's right rail alongside the control panel. */
  variant?: "overlay" | "sidebar";
}

interface Partition {
  label: string;
  retention: number;
  leached: number;
  degraded: number;
  precipitated: number;
}

function partitions(snapshot: Snapshot): Partition[] {
  const out: Partition[] = [];
  const primaryActive =
    snapshot.nutrient_primary_name !== "" ||
    snapshot.carrot_retention < 99.99 ||
    snapshot.carrot_leached > 0 ||
    snapshot.carrot_degraded > 0;
  if (primaryActive) {
    out.push({
      label: snapshot.nutrient_primary_name || "primary",
      retention: snapshot.carrot_retention,
      leached: snapshot.carrot_leached,
      degraded: snapshot.carrot_degraded,
      precipitated: snapshot.carrot_precipitated,
    });
  }
  const secondaryActive =
    snapshot.nutrient_secondary_name !== "" ||
    snapshot.carrot_retention2 < 99.99 ||
    snapshot.carrot_leached2 > 0 ||
    snapshot.carrot_degraded2 > 0;
  if (secondaryActive) {
    out.push({
      label: snapshot.nutrient_secondary_name || "secondary",
      retention: snapshot.carrot_retention2,
      leached: snapshot.carrot_leached2,
      degraded: snapshot.carrot_degraded2,
      precipitated: snapshot.carrot_precipitated2,
    });
  }
  return out;
}

export function SceneOverlay({ snapshot, variant = "overlay" }: Props) {
  const parts = partitions(snapshot);
  const hero = parts[0] ?? null;
  const className =
    variant === "sidebar"
      ? "scene-overlay scene-overlay--sidebar"
      : "scene-overlay";

  return (
    <aside className={className} aria-label="simulation telemetry">
      {hero ? (
        <>
          <span className="scene-overlay__hero-label">
            {hero.label} retention
          </span>
          <div className="scene-overlay__hero">
            {hero.retention.toFixed(2)}
            <span
              style={{
                fontSize: "var(--text-base)",
                color: "var(--text-2)",
                marginLeft: 4,
              }}
            >
              %
            </span>
          </div>
        </>
      ) : null}

      {parts.map((p, i) => (
        <div key={`${p.label}-${i}`} className="overlay-partition">
          {i > 0 ? (
            <div className="overlay-partition__title">{p.label}</div>
          ) : null}
          <PartitionRow
            label="retention"
            value={p.retention}
            tone="r"
            hideWhenZero={false}
          />
          <PartitionRow
            label="leached"
            value={p.leached}
            tone="l"
            hideWhenZero={false}
          />
          <PartitionRow
            label="degraded"
            value={p.degraded}
            tone="d"
            hideWhenZero={false}
          />
          <PartitionRow
            label="precipitated"
            value={p.precipitated}
            tone="p"
            hideWhenZero
          />
          <PartitionSum partition={p} />
        </div>
      ))}

      <div className="scene-overlay__divider" />

      <div className="scene-overlay__row">
        <span>t_sim</span>
        <span className="value">
          {snapshot.t_sim.toFixed(2)} s
          {snapshot.total_time_s > 0
            ? ` / ${snapshot.total_time_s.toFixed(0)} s`
            : ""}
        </span>
      </div>
      {snapshot.total_time_s > 0 ? (
        <div
          className="scene-overlay__progress"
          role="progressbar"
          aria-label="run progress"
          aria-valuemin={0 as number}
          aria-valuemax={100 as number}
          aria-valuenow={
            Math.round(
              Math.min(
                100,
                (100 * snapshot.t_sim) / snapshot.total_time_s,
              ),
            ) as number
          }
          data-pct={Math.min(
            100,
            (100 * snapshot.t_sim) / snapshot.total_time_s,
          )}
        >
          <div
            className="scene-overlay__progress-fill"
            style={{
              width: `${Math.min(
                100,
                (100 * snapshot.t_sim) / snapshot.total_time_s,
              )}%`,
            }}
          />
        </div>
      ) : null}
      <div className="scene-overlay__row">
        <span>step</span>
        <span className="value">{snapshot.step}</span>
      </div>
      <div className="scene-overlay__row">
        <span>water T</span>
        <span className="value">
          {snapshot.water_temperature_mean.toFixed(1)} °C
        </span>
      </div>
      <div className="scene-overlay__row scene-overlay__row--sub">
        <span>water range</span>
        <span className="value">
          {snapshot.water_temperature_min.toFixed(1)} –{" "}
          {snapshot.water_temperature_max.toFixed(1)} °C
        </span>
      </div>
      <div className="scene-overlay__row">
        <span>wall T</span>
        <span className="value">
          {snapshot.wall_temperature_mean.toFixed(1)} °C
        </span>
      </div>
      <div className="scene-overlay__row">
        <span>heat flux</span>
        <span className="value">
          {(snapshot.wall_heat_flux / 1000).toFixed(1)} kW/m²
        </span>
      </div>
      <div className="scene-overlay__row">
        <span>bubbles</span>
        <span className="value">{snapshot.bubbles.length}</span>
      </div>
      {snapshot.is_paused ? (
        <div
          className="scene-overlay__row"
          style={{ color: "var(--accent-cool)", marginTop: 4 }}
        >
          <span>paused</span>
          <span>·</span>
        </div>
      ) : null}
      {snapshot.is_complete ? (
        <div
          className="scene-overlay__row"
          style={{ color: "var(--accent-success)", marginTop: 4 }}
        >
          <span>run complete</span>
          <span>✓</span>
        </div>
      ) : null}
    </aside>
  );
}

function PartitionRow({
  label,
  value,
  tone,
  hideWhenZero,
}: {
  label: string;
  value: number;
  tone: "r" | "l" | "d" | "p";
  hideWhenZero: boolean;
}) {
  if (hideWhenZero && value < 0.005) return null;
  return (
    <div className={`overlay-partition__row overlay-partition__row--${tone}`}>
      <span>{label}</span>
      <span className="value">{value.toFixed(2)} %</span>
    </div>
  );
}

function PartitionSum({ partition: p }: { partition: Partition }) {
  const sum =
    p.retention + p.leached + p.degraded + p.precipitated;
  const drift = Math.abs(sum - 100);
  if (drift < 0.5) return null; // invariant holding, don't clutter the UI
  return (
    <div className="overlay-partition__row overlay-partition__row--warn">
      <span>sum</span>
      <span className="value">{sum.toFixed(2)} %</span>
    </div>
  );
}
