// Time-series strip: four cards (wall T, heat flux, retention,
// bubbles) side-by-side across the bottom of the dashboard. Each
// card has a compact header (title + current value) and a Recharts
// line chart body. Series colors come from CSS tokens via
// useTokenColors so the chart stays in lockstep with the design
// system.

import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useTokenColors } from "../hooks/useTokenColor";
import type { SnapshotSummary } from "../types/snapshot";

interface Props {
  history: SnapshotSummary[];
  historyVersion: number;
  /** Max points rendered per chart. */
  maxPoints?: number;
}

interface ChartPoint {
  t: number;
  T_wall: number;
  heat_flux_kw: number;
  retention1: number;
  leached1: number;
  degraded1: number;
  precipitated1: number;
  retention2: number;
  leached2: number;
  degraded2: number;
  precipitated2: number;
  bubbles: number;
}

function downsample(history: SnapshotSummary[], maxPoints: number): ChartPoint[] {
  const n = history.length;
  if (n === 0) return [];
  const stride = Math.max(1, Math.floor(n / maxPoints));
  const out: ChartPoint[] = [];
  for (let i = 0; i < n; i += stride) {
    const s = history[i];
    out.push({
      t: s.t_sim,
      T_wall: s.wall_temperature_mean,
      heat_flux_kw: s.wall_heat_flux / 1000.0,
      retention1: s.carrot_retention,
      leached1: s.carrot_leached,
      degraded1: s.carrot_degraded,
      precipitated1: s.carrot_precipitated,
      retention2: s.carrot_retention2,
      leached2: s.carrot_leached2,
      degraded2: s.carrot_degraded2,
      precipitated2: s.carrot_precipitated2,
      bubbles: s.bubbles_count,
    });
  }
  return out;
}

function latestNutrientName(
  history: SnapshotSummary[],
  which: "primary" | "secondary",
): string {
  for (let i = history.length - 1; i >= 0; i--) {
    const s = history[i];
    const name =
      which === "primary"
        ? s.nutrient_primary_name
        : s.nutrient_secondary_name;
    if (name) return name;
  }
  return which;
}

function hasSecondSolute(history: SnapshotSummary[]): boolean {
  for (let i = history.length - 1; i >= 0 && i >= history.length - 10; i--) {
    if (history[i].carrot_retention2 < 99.99) return true;
  }
  return false;
}

export function TimeSeriesPanel({
  history,
  historyVersion,
  maxPoints = 240,
}: Props) {
  const colors = useTokenColors({
    wall: "--plot-wall",
    flux: "--plot-flux",
    r1: "--plot-r1",
    r2: "--plot-r2",
    bubbles: "--plot-bubbles",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
  });

  const data = useMemo(
    () => downsample(history, maxPoints),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [historyVersion, maxPoints],
  );

  const showSecond = useMemo(
    () => hasSecondSolute(history),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [historyVersion],
  );

  if (data.length === 0) {
    return (
      <>
        <PlotCard title="Wall inner T" value="— °C">
          <Empty />
        </PlotCard>
        <PlotCard title="Heat flux" value="— kW/m²">
          <Empty />
        </PlotCard>
        <PlotCard title="Mass partition" value="—">
          <Empty />
        </PlotCard>
        <PlotCard title="Bubbles" value="—">
          <Empty />
        </PlotCard>
      </>
    );
  }

  const last = data[data.length - 1];

  return (
    <>
      <PlotCard title="Wall inner T" value={`${last.T_wall.toFixed(1)} °C`}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={chartMargin}>
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis dataKey="t" tick={{ ...tickStyle, fill: colors.axis }} stroke={colors.axis} unit="s" />
            <YAxis domain={["auto", "auto"]} tick={{ ...tickStyle, fill: colors.axis }} stroke={colors.axis} />
            <Tooltip
              contentStyle={tooltipStyle(colors)}
              labelFormatter={fmtTime}
              formatter={(v: number) => [`${v.toFixed(2)} °C`, "T_wall"]}
            />
            <Line type="monotone" dataKey="T_wall" stroke={colors.wall} strokeWidth={1.6} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </PlotCard>

      <PlotCard title="Heat flux" value={`${last.heat_flux_kw.toFixed(1)} kW/m²`}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={chartMargin}>
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis dataKey="t" tick={{ ...tickStyle, fill: colors.axis }} stroke={colors.axis} unit="s" />
            <YAxis tick={{ ...tickStyle, fill: colors.axis }} stroke={colors.axis} />
            <Tooltip
              contentStyle={tooltipStyle(colors)}
              labelFormatter={fmtTime}
              formatter={(v: number) => [`${v.toFixed(1)} kW/m²`, "q"]}
            />
            <Line type="monotone" dataKey="heat_flux_kw" stroke={colors.flux} strokeWidth={1.6} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </PlotCard>

      <PlotCard
        title={
          showSecond
            ? `${latestNutrientName(history, "primary")} mass partition`
            : `${latestNutrientName(history, "primary")} mass partition`
        }
        value={
          <span className="plot-card__value--split">
            <span style={{ color: colors.r1 }}>R {last.retention1.toFixed(1)}%</span>
            <span style={{ color: colors.r2 }}>L {last.leached1.toFixed(1)}%</span>
            <span style={{ color: colors.wall }}>D {last.degraded1.toFixed(1)}%</span>
          </span>
        }
      >
        <StackedPartition
          data={data}
          colors={colors}
          rKey="retention1"
          lKey="leached1"
          dKey="degraded1"
          pKey="precipitated1"
        />
      </PlotCard>

      {showSecond ? (
        <PlotCard
          title={`${latestNutrientName(history, "secondary")} mass partition`}
          value={
            <span className="plot-card__value--split">
              <span style={{ color: colors.r1 }}>R {last.retention2.toFixed(1)}%</span>
              <span style={{ color: colors.r2 }}>L {last.leached2.toFixed(1)}%</span>
              <span style={{ color: colors.wall }}>D {last.degraded2.toFixed(1)}%</span>
            </span>
          }
        >
          <StackedPartition
            data={data}
            colors={colors}
            rKey="retention2"
            lKey="leached2"
            dKey="degraded2"
            pKey="precipitated2"
          />
        </PlotCard>
      ) : null}

      <PlotCard title="Active bubbles" value={last.bubbles.toLocaleString()}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={chartMargin}>
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis dataKey="t" tick={{ ...tickStyle, fill: colors.axis }} stroke={colors.axis} unit="s" />
            <YAxis tick={{ ...tickStyle, fill: colors.axis }} stroke={colors.axis} allowDecimals={false} />
            <Tooltip
              contentStyle={tooltipStyle(colors)}
              labelFormatter={fmtTime}
              formatter={(v: number) => [v.toLocaleString(), "count"]}
            />
            <Line type="monotone" dataKey="bubbles" stroke={colors.bubbles} strokeWidth={1.6} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </PlotCard>
    </>
  );
}

// ------- helpers ----------------------------------------------------

/** Stacked-area mass partition for a single solute. Mirrors the
 *  [benchmarks/phase4_retention.md](../../../benchmarks/phase4_retention.md)
 *  validation figures: `retention` in green at the bottom, `leached` in
 *  blue on top, `degraded` in red above that, `precipitated` (usually
 *  near zero) in violet at the top. The four buckets stack to ~100 %,
 *  which is the Phase 4 mass-balance invariant. */
function StackedPartition({
  data,
  colors,
  rKey,
  lKey,
  dKey,
  pKey,
}: {
  data: ChartPoint[];
  colors: Record<string, string>;
  rKey: keyof ChartPoint;
  lKey: keyof ChartPoint;
  dKey: keyof ChartPoint;
  pKey: keyof ChartPoint;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={chartMargin}>
        <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="t"
          tick={{ ...tickStyle, fill: colors.axis }}
          stroke={colors.axis}
          unit="s"
        />
        <YAxis
          domain={[0, 100]}
          tick={{ ...tickStyle, fill: colors.axis }}
          stroke={colors.axis}
        />
        <Tooltip
          contentStyle={tooltipStyle(colors)}
          labelFormatter={fmtTime}
          formatter={(v: number, name: string) => [`${v.toFixed(2)} %`, name]}
        />
        <Legend wrapperStyle={{ fontSize: 10, paddingTop: 2 }} iconSize={8} />
        <Area
          type="monotone"
          dataKey={rKey}
          name="retention"
          stackId="partition"
          stroke={colors.r1}
          fill={colors.r1}
          fillOpacity={0.55}
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey={lKey}
          name="leached"
          stackId="partition"
          stroke={colors.r2}
          fill={colors.r2}
          fillOpacity={0.55}
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey={dKey}
          name="degraded"
          stackId="partition"
          stroke={colors.wall}
          fill={colors.wall}
          fillOpacity={0.45}
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey={pKey}
          name="precip."
          stackId="partition"
          stroke={colors.bubbles}
          fill={colors.bubbles}
          fillOpacity={0.45}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function PlotCard({
  title,
  value,
  children,
}: {
  title: string;
  value: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="plot-card">
      <div className="plot-card__head">
        <span className="plot-card__title">{title}</span>
        <span className="plot-card__value">{value}</span>
      </div>
      <div className="plot-card__body">{children}</div>
    </div>
  );
}

function Empty() {
  return <div className="plot-empty">no data yet</div>;
}

function fmtTime(value: number | string): string {
  if (typeof value === "number") return `t = ${value.toFixed(2)} s`;
  return String(value);
}

const chartMargin = { top: 6, right: 10, bottom: 6, left: 0 };
const tickStyle = { fontSize: 10 };

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
