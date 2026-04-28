// Nutrient loss-rate card -- d/dt(retention_pct) per solute, smoothed,
// converted to %/min. Shows when the carrot is leaking nutrient
// fastest; spikes typically appear near the saturation transition
// (when convection kicks in) and decay as the surface concentration
// depletes.
//
// Uses computeLossRate from derived.ts, which central-differences
// the retention column and applies a 3-sample moving average to
// suppress 30 Hz snapshot noise without blurring the saturation
// transient.

import { useMemo } from "react";
import {
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

import type { ScalarRow, RunSummary } from "../../hooks/useRunArtefacts";
import { useTokenColors } from "../../hooks/useTokenColor";

import { ChartCard } from "./ChartCard";
import { computeLossRate, findMilestones, hasSecondarySolute } from "./derived";

interface Props {
  scalars: ScalarRow[];
  summary: RunSummary;
}

export function NutrientLossRateCard({ scalars, summary }: Props) {
  const colors = useTokenColors({
    r1: "--plot-r1",
    r2: "--plot-r2",
    cool: "--accent-cool",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
  });

  const includeSecondary = hasSecondarySolute(summary);
  const milestones = useMemo(() => findMilestones(scalars), [scalars]);
  const data = useMemo(
    () => computeLossRate(scalars, includeSecondary),
    [scalars, includeSecondary],
  );

  const primaryName = summary.nutrient_primary_name || "primary";
  const secondaryName = summary.nutrient_secondary_name || "secondary";

  // See HeatUpStorylineCard for the rationale -- a milestone outside
  // the data's t range produces a garbled clipped label on the y-axis.
  const tMin = data.length > 0 ? data[0].t : -Infinity;
  const tMax = data.length > 0 ? data[data.length - 1].t : Infinity;
  const tSat =
    milestones.tSat !== undefined
    && milestones.tSat >= tMin
    && milestones.tSat <= tMax
      ? milestones.tSat
      : undefined;

  return (
    <ChartCard
      name="nutrient_loss_rate"
      title="Nutrient loss-rate (%/min)"
      subtitle="d/dt(retention) -- positive when nutrient is leaving the carrot"
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
            {/* `%/min` unit moved into the card title (above) to keep
                this y-axis as a clean tick column. The previous
                `label={{ value: "%/min", angle: -90, position:
                "insideLeft" }}` overlapped the tick labels — at small
                rate scales (0.0..0.8 %/min) Recharts spaced the
                ticks tightly enough that the rotated unit-text and
                the numeric tick labels rendered on top of each other,
                producing the apparent "0.8 / 0.8 / 0.8 / 0.8" stack. */}
            <YAxis
              stroke={colors.axis}
              tick={{ fontSize: 10, fill: colors.axis }}
              width={42}
            />
            <ReferenceLine
              y={0}
              stroke={colors.axis}
              strokeOpacity={0.5}
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
            <Tooltip
              contentStyle={{
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                color: colors.text1,
                fontSize: 12,
              }}
              formatter={(v: number, name: string) => [
                `${v.toFixed(3)} %/min`,
                name,
              ]}
              labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} iconSize={8} />
            <Line
              type="monotone"
              dataKey="rate1"
              name={primaryName}
              stroke={colors.r1}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
            />
            {includeSecondary ? (
              <Line
                type="monotone"
                dataKey="rate2"
                name={secondaryName}
                stroke={colors.r2}
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}
