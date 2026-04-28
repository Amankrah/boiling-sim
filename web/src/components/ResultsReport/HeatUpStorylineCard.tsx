// Heat-up storyline -- one line chart of bulk water mean temperature
// over time, annotated with three vertical reference lines marking
// the run's three key milestones:
//
//   * t_sat          -- when the water first hit ~saturation (99 deg C)
//   * t_first_bubble -- when the first nucleation bubble appeared
//   * t_first_loss   -- when retention first dropped below 99.5 %
//
// Above the chart, three small "milestone pills" repeat those times
// in seconds so the reader doesn't have to mouse over the lines.
//
// Helpers in derived.ts so the same milestone search can drive the
// boiling-vigor and loss-rate cards.

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ScalarRow } from "../../hooks/useRunArtefacts";
import { useTokenColors } from "../../hooks/useTokenColor";

import { ChartCard } from "./ChartCard";
import { findMilestones, type Milestones } from "./derived";

interface Props {
  scalars: ScalarRow[];
}

export function HeatUpStorylineCard({ scalars }: Props) {
  const colors = useTokenColors({
    cool: "--accent-cool",
    success: "--accent-success",
    warn: "--accent-warn",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
    text2: "--text-2",
    text3: "--text-3",
  });

  const milestones = useMemo(() => findMilestones(scalars), [scalars]);

  const data = useMemo(
    () =>
      scalars.map((s) => ({ t: s.t, T_water: s.T_mean_water_c })),
    [scalars],
  );

  // Drop any milestone whose timestamp falls outside the visible
  // X-axis range. Recharts otherwise clips the ReferenceLine to the
  // chart edge and renders its label as a vertical column of glyphs
  // stacked on the y-axis -- which is what produced the garbled
  // "78320312"-looking label when finalize was clicked on a run whose
  // history had been trimmed past the early milestones.
  const tMin = data.length > 0 ? data[0].t : -Infinity;
  const tMax = data.length > 0 ? data[data.length - 1].t : Infinity;
  const inRange = (t: number | undefined): t is number =>
    t !== undefined && t >= tMin && t <= tMax;
  const tSat = inRange(milestones.tSat) ? milestones.tSat : undefined;
  const tFirstBubble = inRange(milestones.tFirstBubble)
    ? milestones.tFirstBubble
    : undefined;
  const tFirstLoss = inRange(milestones.tFirstLoss)
    ? milestones.tFirstLoss
    : undefined;

  return (
    <ChartCard
      name="heat_up_storyline"
      title="Heat-up storyline"
      subtitle="Water mean T with run milestones"
      rightSlot={<MilestonePills milestones={milestones} colors={colors} />}
    >
      <div className="report-card__body--chart" style={{ height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 8, right: 16, bottom: 6, left: 0 }}
          >
            <CartesianGrid stroke={colors.grid} strokeDasharray="3 3" />
            <XAxis
              dataKey="t"
              stroke={colors.axis}
              tick={{ fontSize: 10, fill: colors.axis }}
              unit="s"
            />
            <YAxis
              stroke={colors.axis}
              tick={{ fontSize: 10, fill: colors.axis }}
              domain={[20, "dataMax + 5"]}
            />
            <ReferenceLine
              y={100}
              stroke={colors.cool}
              strokeDasharray="4 4"
              label={{ value: "T_sat", fill: colors.cool, fontSize: 10 }}
            />
            {tSat !== undefined ? (
              <ReferenceLine
                x={tSat}
                stroke={colors.cool}
                strokeDasharray="2 4"
                label={{
                  value: "sat",
                  position: "top",
                  fill: colors.cool,
                  fontSize: 10,
                }}
              />
            ) : null}
            {tFirstBubble !== undefined ? (
              <ReferenceLine
                x={tFirstBubble}
                stroke={colors.success}
                strokeDasharray="2 4"
                label={{
                  value: "1st bubble",
                  position: "top",
                  fill: colors.success,
                  fontSize: 10,
                }}
              />
            ) : null}
            {tFirstLoss !== undefined ? (
              <ReferenceLine
                x={tFirstLoss}
                stroke={colors.warn}
                strokeDasharray="2 4"
                label={{
                  value: "loss begins",
                  position: "top",
                  fill: colors.warn,
                  fontSize: 10,
                }}
              />
            ) : null}
            <Tooltip
              contentStyle={{
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                color: colors.text1,
                fontSize: 12,
              }}
              formatter={(v: number) => [`${v.toFixed(2)} °C`, "water mean"]}
              labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`}
            />
            <Line
              type="monotone"
              dataKey="T_water"
              stroke={colors.cool}
              strokeWidth={1.7}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

function MilestonePills({
  milestones,
  colors,
}: {
  milestones: Milestones;
  colors: Record<string, string>;
}) {
  const pills: Array<{ label: string; t: number; color: string }> = [];
  if (milestones.tSat !== undefined) {
    pills.push({ label: "sat", t: milestones.tSat, color: colors.cool });
  }
  if (milestones.tFirstBubble !== undefined) {
    pills.push({
      label: "1st bub",
      t: milestones.tFirstBubble,
      color: colors.success,
    });
  }
  if (milestones.tFirstLoss !== undefined) {
    pills.push({
      label: "loss",
      t: milestones.tFirstLoss,
      color: colors.warn,
    });
  }
  if (pills.length === 0) return null;
  return (
    <div className="milestone-pills">
      {pills.map((p) => (
        <span
          key={p.label}
          className="milestone-pill mono"
          style={{ color: p.color, borderColor: p.color }}
          title={`${p.label} reached at t = ${p.t.toFixed(2)} s`}
        >
          {p.label} {p.t.toFixed(1)}s
        </span>
      ))}
    </div>
  );
}
