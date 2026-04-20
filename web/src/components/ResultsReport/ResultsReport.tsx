// Phase-4-style report rendered from the live run artefacts. Header
// sections mirror `benchmarks/phase4_retention.md`:
//
//   1. Headline        -- big retention number + band pill
//   2. Parameters      -- applied config echo
//   3. Trajectory      -- table at t = 0/10/.../100 % of sim
//   4. Time series     -- stacked mass partition + water/wall T + bubbles
//   5. Mass balance    -- |sum - 100| over time
//   6. Exit-check      -- auto-generated gate list from acceptance[]
//   7. Performance     -- wall clock, s/sim-s, steps, cadence
//   8. Downloads       -- HDF5 / CSV / JSON
//
// Every chart uses Recharts with colours from the design tokens, so
// the visual matches the Live page's time-series strip.

import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useTokenColors } from "../../hooks/useTokenColor";
import type {
  RunArtefacts,
  RunSummary,
  ScalarRow,
} from "../../hooks/useRunArtefacts";
import { Button } from "../ui/Button";

interface Props {
  artefacts: RunArtefacts;
  onStartNewRun?: () => void;
}

export function ResultsReport({ artefacts, onStartNewRun }: Props) {
  const { summary, scalars, runId } = artefacts;
  return (
    <div className="results-report">
      <Headline summary={summary} />
      <DownloadButtons runId={runId} onStartNewRun={onStartNewRun} />
      <ExitCheckAudit summary={summary} />
      <div className="report-grid">
        <MassPartitionCard scalars={scalars} summary={summary} which="primary" />
        {isSecondaryActive(summary) ? (
          <MassPartitionCard scalars={scalars} summary={summary} which="secondary" />
        ) : null}
        <ThermalCard scalars={scalars} />
        <MassBalanceCard scalars={scalars} summary={summary} />
        <BubblesCard scalars={scalars} />
        <PerformanceCard summary={summary} />
      </div>
      <TrajectoryTable scalars={scalars} summary={summary} />
      <ParametersTable summary={summary} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Headline
// ---------------------------------------------------------------------------

function Headline({ summary }: { summary: RunSummary }) {
  const retention = summary.final.retention_pct ?? 0;
  const nutrient = summary.nutrient_primary_name || "nutrient";
  const band = expectedBand(nutrient);
  const classification = band
    ? retention < band[0]
      ? { label: "Below band", tone: "warn" }
      : retention > band[1]
        ? { label: "Above band", tone: "warn" }
        : { label: "In band", tone: "success" }
    : null;
  return (
    <header className="report-headline">
      <div className="report-headline__labels">
        <span className="report-headline__label">Final retention</span>
        <span className="report-headline__nutrient">{nutrient}</span>
      </div>
      <div className="report-headline__value">
        {retention.toFixed(2)}
        <span className="report-headline__unit">%</span>
      </div>
      {classification ? (
        <span className={`report-pill report-pill--${classification.tone}`}>
          {classification.label}
          {band ? ` (target ${band[0]}–${band[1]} %)` : null}
        </span>
      ) : null}
      <div className="report-headline__meta mono">
        t = {summary.t_sim_total_s.toFixed(1)} s · {summary.n_samples} samples
      </div>
    </header>
  );
}

function expectedBand(nutrientName: string): [number, number] | null {
  const n = nutrientName.toLowerCase();
  if (n.includes("caroten")) return [80, 90];
  if (n.includes("vitamin")) return [55, 80];
  return null;
}

// ---------------------------------------------------------------------------
// Download buttons
// ---------------------------------------------------------------------------

function DownloadButtons({
  runId,
  onStartNewRun,
}: {
  runId: string;
  onStartNewRun?: () => void;
}) {
  const baseHref = `/api/runs/${encodeURIComponent(runId)}`;
  return (
    <div className="report-downloads">
      <a className="btn btn--primary" href={`${baseHref}/data.h5`} download>
        Download HDF5
      </a>
      <a className="btn" href={`${baseHref}/scalars.csv`} download>
        Download CSV
      </a>
      <a className="btn" href={`${baseHref}/summary.json`} download>
        Download summary JSON
      </a>
      {onStartNewRun ? (
        <span style={{ flex: 1 }} />
      ) : null}
      {onStartNewRun ? (
        <Button variant="ghost" onClick={onStartNewRun}>
          Configure another run →
        </Button>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exit-check audit
// ---------------------------------------------------------------------------

function ExitCheckAudit({ summary }: { summary: RunSummary }) {
  const gates = summary.acceptance ?? [];
  const passed = gates.filter((g) => g.passed).length;
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">Exit-check audit</span>
        <span className="report-card__value">
          {passed} / {gates.length} passed
        </span>
      </div>
      <div className="report-card__body">
        <ul className="gate-list">
          {gates.map((g, i) => (
            <li key={`${g.name}-${i}`} className="gate-row">
              <span
                className={`gate-mark gate-mark--${g.passed ? "ok" : "no"}`}
                aria-hidden
              >
                {g.passed ? "✓" : "✗"}
              </span>
              <span className="gate-name">{g.name}</span>
              <span className="gate-detail mono">{g.detail}</span>
            </li>
          ))}
          {gates.length === 0 ? (
            <li className="gate-row">
              <span className="hint">No gates recorded.</span>
            </li>
          ) : null}
        </ul>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Mass partition stacked area (per solute)
// ---------------------------------------------------------------------------

function isSecondaryActive(summary: RunSummary): boolean {
  if (summary.nutrient_secondary_name && summary.nutrient_secondary_name !== "") return true;
  const r2 = summary.final.retention2_pct ?? 100;
  const l2 = summary.final.leached2_pct ?? 0;
  return r2 < 99.99 || l2 > 0.01;
}

function MassPartitionCard({
  scalars,
  summary,
  which,
}: {
  scalars: ScalarRow[];
  summary: RunSummary;
  which: "primary" | "secondary";
}) {
  const colors = useTokenColors({
    r1: "--plot-r1",
    r2: "--plot-r2",
    wall: "--plot-wall",
    bubbles: "--plot-bubbles",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
  });
  const data = useMemo(() => {
    return scalars.map((s) => ({
      t: s.t,
      retention: which === "primary" ? s.retention_pct : s.retention2_pct,
      leached: which === "primary" ? s.leached_pct : s.leached2_pct,
      degraded: which === "primary" ? s.degraded_pct : s.degraded2_pct,
      precipitated:
        which === "primary" ? s.precipitated_pct : s.precipitated2_pct,
    }));
  }, [scalars, which]);
  const title =
    which === "primary"
      ? `${summary.nutrient_primary_name || "primary"} — mass partition`
      : `${summary.nutrient_secondary_name || "secondary"} — mass partition`;
  const final =
    which === "primary"
      ? {
          r: summary.final.retention_pct ?? 0,
          l: summary.final.leached_pct ?? 0,
          d: summary.final.degraded_pct ?? 0,
        }
      : {
          r: summary.final.retention2_pct ?? 0,
          l: summary.final.leached2_pct ?? 0,
          d: summary.final.degraded2_pct ?? 0,
        };
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">{title}</span>
        <span className="report-card__value mono">
          R {final.r.toFixed(1)} · L {final.l.toFixed(1)} · D {final.d.toFixed(1)} %
        </span>
      </div>
      <div className="report-card__body report-card__body--chart">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 6, left: 0 }}>
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis dataKey="t" stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} unit="s" />
            <YAxis domain={[0, 100]} stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} />
            <Tooltip
              contentStyle={tooltipStyle(colors)}
              formatter={(v: number, name: string) => [`${v.toFixed(2)} %`, name]}
              labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`}
            />
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 2 }} iconSize={8} />
            <Area type="monotone" dataKey="retention" stackId="p" stroke={colors.r1} fill={colors.r1} fillOpacity={0.55} isAnimationActive={false} />
            <Area type="monotone" dataKey="leached" stackId="p" stroke={colors.r2} fill={colors.r2} fillOpacity={0.55} isAnimationActive={false} />
            <Area type="monotone" dataKey="degraded" stackId="p" stroke={colors.wall} fill={colors.wall} fillOpacity={0.45} isAnimationActive={false} />
            <Area type="monotone" dataKey="precipitated" stackId="p" stroke={colors.bubbles} fill={colors.bubbles} fillOpacity={0.45} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Thermal: water T (mean / min / max) + inner wall T + T_sat line
// ---------------------------------------------------------------------------

function ThermalCard({ scalars }: { scalars: ScalarRow[] }) {
  const colors = useTokenColors({
    wall: "--plot-wall",
    flux: "--plot-flux",
    cool: "--accent-cool",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
  });
  const data = useMemo(() => {
    return scalars.map((s) => ({
      t: s.t,
      T_water: s.T_mean_water_c,
      T_water_max: s.T_max_water_c,
      T_water_min: s.T_min_water_c,
      T_wall: s.T_inner_wall_mean_c,
    }));
  }, [scalars]);
  const last = data.length > 0 ? data[data.length - 1] : null;
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">Temperatures (°C)</span>
        <span className="report-card__value mono">
          {last ? `water ${last.T_water.toFixed(1)} · wall ${last.T_wall.toFixed(1)}` : "—"}
        </span>
      </div>
      <div className="report-card__body report-card__body--chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 6, left: 0 }}>
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis dataKey="t" stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} unit="s" />
            <YAxis stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} />
            <ReferenceLine y={100} stroke={colors.cool} strokeDasharray="4 4" label={{ value: "T_sat", fill: colors.cool, fontSize: 10 }} />
            <Tooltip
              contentStyle={tooltipStyle(colors)}
              formatter={(v: number, name: string) => [`${v.toFixed(2)} °C`, name]}
              labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} iconSize={8} />
            <Line type="monotone" dataKey="T_water" name="water mean" stroke={colors.cool} strokeWidth={1.6} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="T_water_max" name="water max" stroke={colors.cool} strokeDasharray="3 3" strokeWidth={1.0} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="T_water_min" name="water min" stroke={colors.cool} strokeDasharray="1 3" strokeWidth={1.0} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="T_wall" name="wall inner" stroke={colors.wall} strokeWidth={1.6} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Mass-balance drift
// ---------------------------------------------------------------------------

function MassBalanceCard({
  scalars,
  summary,
}: {
  scalars: ScalarRow[];
  summary: RunSummary;
}) {
  const colors = useTokenColors({
    warn: "--accent-warn",
    success: "--accent-success",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
  });
  const data = useMemo(() => {
    return scalars.map((s) => {
      const sum =
        s.retention_pct + s.leached_pct + s.degraded_pct + s.precipitated_pct;
      return { t: s.t, drift: Math.abs(sum - 100) };
    });
  }, [scalars]);
  const maxDrift = summary.mass_balance?.max_abs_drift_pct ?? 0;
  const gateOk = maxDrift < 0.5;
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">Mass-balance drift (primary)</span>
        <span
          className="report-card__value mono"
          style={{ color: gateOk ? colors.success : colors.warn }}
        >
          max |sum − 100| = {maxDrift.toFixed(4)} pp
        </span>
      </div>
      <div className="report-card__body report-card__body--chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 6, left: 0 }}>
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis dataKey="t" stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} unit="s" />
            <YAxis stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} />
            <ReferenceLine y={0.5} stroke={colors.warn} strokeDasharray="4 4" label={{ value: "0.5 pp gate", fill: colors.warn, fontSize: 10 }} />
            <Tooltip
              contentStyle={tooltipStyle(colors)}
              formatter={(v: number) => [`${v.toFixed(4)} pp`, "|sum-100|"]}
              labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`}
            />
            <Line type="monotone" dataKey="drift" stroke={colors.success} strokeWidth={1.4} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Bubbles
// ---------------------------------------------------------------------------

function BubblesCard({ scalars }: { scalars: ScalarRow[] }) {
  const colors = useTokenColors({
    bubbles: "--plot-bubbles",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
  });
  const data = useMemo(
    () => scalars.map((s) => ({ t: s.t, n: s.n_active_bubbles })),
    [scalars],
  );
  const last = data.length > 0 ? data[data.length - 1].n : 0;
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">Active bubbles</span>
        <span className="report-card__value mono">{last.toLocaleString()}</span>
      </div>
      <div className="report-card__body report-card__body--chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 6, left: 0 }}>
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis dataKey="t" stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} unit="s" />
            <YAxis stroke={colors.axis} tick={{ fontSize: 10, fill: colors.axis }} allowDecimals={false} />
            <Tooltip
              contentStyle={tooltipStyle(colors)}
              formatter={(v: number) => [v.toLocaleString(), "count"]}
              labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`}
            />
            <Line type="monotone" dataKey="n" stroke={colors.bubbles} strokeWidth={1.4} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Performance card (static numbers)
// ---------------------------------------------------------------------------

function PerformanceCard({ summary }: { summary: RunSummary }) {
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">Performance</span>
      </div>
      <div className="report-card__body">
        <dl className="kv-list">
          <KV label="Wall clock" value={`${summary.wall_clock_s.toFixed(1)} s`} />
          <KV label="Sim time" value={`${summary.t_sim_total_s.toFixed(1)} s`} />
          <KV
            label="s / sim-s"
            value={summary.s_per_sim_s > 0 ? summary.s_per_sim_s.toFixed(2) : "—"}
          />
          <KV label="Steps" value={summary.step_count.toLocaleString()} />
          <KV
            label="Snapshot cadence"
            value={`${summary.snapshot_cadence_hz.toFixed(1)} Hz`}
          />
          <KV label="Samples retained" value={summary.n_samples.toLocaleString()} />
        </dl>
      </div>
    </section>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="kv">
      <dt>{label}</dt>
      <dd className="mono">{value}</dd>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trajectory table at 10% intervals
// ---------------------------------------------------------------------------

function TrajectoryTable({
  scalars,
  summary,
}: {
  scalars: ScalarRow[];
  summary: RunSummary;
}) {
  const rows = useMemo(() => pickTrajectoryRows(scalars), [scalars]);
  const sec = isSecondaryActive(summary);
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">Trajectory</span>
        <span className="report-card__value hint">
          Sampled at 0 / 10 / … / 100 % of total run
        </span>
      </div>
      <div className="report-card__body">
        <div className="report-table-wrap">
          <table className="report-table">
            <thead>
              <tr>
                <th>t (s)</th>
                <th>T_water (°C)</th>
                <th>T_wall (°C)</th>
                <th colSpan={4}>
                  {summary.nutrient_primary_name || "primary"}
                </th>
                {sec ? (
                  <th colSpan={4}>
                    {summary.nutrient_secondary_name || "secondary"}
                  </th>
                ) : null}
              </tr>
              <tr className="subhead">
                <th />
                <th />
                <th />
                <th>R</th>
                <th>L</th>
                <th>D</th>
                <th>P</th>
                {sec ? (
                  <>
                    <th>R</th>
                    <th>L</th>
                    <th>D</th>
                    <th>P</th>
                  </>
                ) : null}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td className="mono">{r.t.toFixed(1)}</td>
                  <td className="mono">{r.T_mean_water_c.toFixed(2)}</td>
                  <td className="mono">{r.T_inner_wall_mean_c.toFixed(2)}</td>
                  <td className="mono">{r.retention_pct.toFixed(2)}</td>
                  <td className="mono">{r.leached_pct.toFixed(2)}</td>
                  <td className="mono">{r.degraded_pct.toFixed(2)}</td>
                  <td className="mono">{r.precipitated_pct.toFixed(2)}</td>
                  {sec ? (
                    <>
                      <td className="mono">{r.retention2_pct.toFixed(2)}</td>
                      <td className="mono">{r.leached2_pct.toFixed(2)}</td>
                      <td className="mono">{r.degraded2_pct.toFixed(2)}</td>
                      <td className="mono">{r.precipitated2_pct.toFixed(2)}</td>
                    </>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function pickTrajectoryRows(scalars: ScalarRow[]): ScalarRow[] {
  if (scalars.length === 0) return [];
  const total = scalars[scalars.length - 1].t;
  if (total <= 0) return [scalars[0]];
  const out: ScalarRow[] = [];
  for (let pct = 0; pct <= 10; pct++) {
    const target = (total * pct) / 10;
    // Find the first sample with t >= target; fall back to the last
    // sample for pct=10.
    const found = scalars.find((s) => s.t >= target) ?? scalars[scalars.length - 1];
    out.push(found);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Parameters echo
// ---------------------------------------------------------------------------

function ParametersTable({ summary }: { summary: RunSummary }) {
  const params = summary.parameters ?? {};
  // Flatten one level: { section: { key: value } } -> rows.
  const rows: Array<{ section: string; key: string; value: string }> = [];
  for (const [section, sub] of Object.entries(params)) {
    if (sub && typeof sub === "object" && !Array.isArray(sub)) {
      for (const [k, v] of Object.entries(sub as Record<string, unknown>)) {
        rows.push({
          section,
          key: k,
          value: formatParamValue(v),
        });
      }
    } else {
      rows.push({
        section: "",
        key: section,
        value: formatParamValue(sub),
      });
    }
  }
  return (
    <section className="report-card">
      <div className="report-card__head">
        <span className="report-card__title">Applied parameters</span>
        <span className="report-card__value hint">
          Echo from ScenarioConfig.model_dump
        </span>
      </div>
      <div className="report-card__body">
        <div className="report-table-wrap">
          <table className="report-table report-table--params">
            <thead>
              <tr>
                <th>Section</th>
                <th>Field</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td>{r.section}</td>
                  <td className="mono">{r.key}</td>
                  <td className="mono">{r.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function formatParamValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(3);
    return String(v);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return `[${v.map(formatParamValue).join(", ")}]`;
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tooltipStyle(
  colors: Record<"bg" | "border" | "text1", string>,
): React.CSSProperties {
  return {
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    color: colors.text1,
    fontSize: 12,
  };
}
