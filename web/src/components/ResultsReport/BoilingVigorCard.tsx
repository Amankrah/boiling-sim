// Boiling vigor -- three overlaid traces showing how vigorous the
// boiling is over time:
//
//   * active bubble count            (left  y-axis, integers)
//   * mean active-bubble radius (mm) (right y-axis, mm)
//   * max water velocity   (mm/s)    (right y-axis, mm/s)
//
// `u_max_mps` is converted from m/s to mm/s on the way in so it
// shares the right axis with the bubble-radius trace without a
// huge dynamic-range mismatch (typical pot boiling u_max is <1 m/s,
// bubble R is <2 mm; both fit on a 0..2000 mm-style axis).
//
// A vertical reference line at the saturation milestone anchors the
// "before / after boiling" phases. Reuses findMilestones from
// derived.ts.

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

import type { ScalarRow } from "../../hooks/useRunArtefacts";
import { useTokenColors } from "../../hooks/useTokenColor";

import { ChartCard } from "./ChartCard";
import { findMilestones } from "./derived";

interface Props {
  scalars: ScalarRow[];
}

export function BoilingVigorCard({ scalars }: Props) {
  const colors = useTokenColors({
    bubbles: "--plot-bubbles",
    flux: "--plot-flux",
    cool: "--accent-cool",
    warn: "--accent-warn",
    grid: "--plot-grid",
    axis: "--plot-axis",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
  });

  const milestones = useMemo(() => findMilestones(scalars), [scalars]);

  const data = useMemo(
    () =>
      scalars.map((s) => ({
        t: s.t,
        n: s.n_active_bubbles,
        meanRmm: Number.isFinite(s.mean_bubble_R_mm) ? s.mean_bubble_R_mm : 0,
        // Convert m/s to mm/s for axis-share with bubble radius.
        umax_mmps: Number.isFinite(s.u_max_mps) ? s.u_max_mps * 1000 : 0,
      })),
    [scalars],
  );

  return (
    <ChartCard
      name="boiling_vigor"
      title="Boiling vigor"
      subtitle="Bubble count + mean radius + max velocity"
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
              yAxisId="count"
              orientation="left"
              stroke={colors.bubbles}
              tick={{ fontSize: 10, fill: colors.axis }}
              allowDecimals={false}
            />
            <YAxis
              yAxisId="dyn"
              orientation="right"
              stroke={colors.flux}
              tick={{ fontSize: 10, fill: colors.axis }}
            />
            {milestones.tSat !== undefined ? (
              <ReferenceLine
                yAxisId="count"
                x={milestones.tSat}
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
              formatter={(value: number, name: string) => {
                if (name === "bubble count")
                  return [value.toLocaleString(), name];
                if (name === "mean R")
                  return [`${value.toFixed(3)} mm`, name];
                if (name === "max u") return [`${value.toFixed(1)} mm/s`, name];
                return [value, name];
              }}
              labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} iconSize={8} />
            <Line
              yAxisId="count"
              type="monotone"
              dataKey="n"
              name="bubble count"
              stroke={colors.bubbles}
              strokeWidth={1.4}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="dyn"
              type="monotone"
              dataKey="meanRmm"
              name="mean R"
              stroke={colors.flux}
              strokeWidth={1.4}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="dyn"
              type="monotone"
              dataKey="umax_mmps"
              name="max u"
              stroke={colors.warn}
              strokeWidth={1.2}
              strokeDasharray="3 3"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}
