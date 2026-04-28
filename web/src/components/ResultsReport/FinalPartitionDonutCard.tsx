// Final-partition donut -- one Recharts PieChart per active solute
// showing the end-state distribution of the four mass buckets
// (retention / leached / degraded / precipitated). Easier to scan
// than the time-stacked area when the user just wants to know
// "where did the nutrient go?".
//
// Centre label per donut shows the nutrient name + final retention %
// so the same artefact can be lifted directly into a slide.

import { useMemo } from "react";
import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import type { RunSummary } from "../../hooks/useRunArtefacts";
import { useTokenColors } from "../../hooks/useTokenColor";

import { ChartCard } from "./ChartCard";
import { finalPartition, hasSecondarySolute } from "./derived";

interface Props {
  summary: RunSummary;
}

export function FinalPartitionDonutCard({ summary }: Props) {
  const colors = useTokenColors({
    r1: "--plot-r1",
    r2: "--plot-r2",
    wall: "--plot-wall",
    bubbles: "--plot-bubbles",
    bg: "--bg-0",
    border: "--border-subtle",
    text1: "--text-1",
    text2: "--text-2",
    text3: "--text-3",
  });

  const includeSecondary = hasSecondarySolute(summary);
  const primaryName = summary.nutrient_primary_name || "primary";
  const secondaryName = summary.nutrient_secondary_name || "secondary";

  return (
    <ChartCard
      name="final_partition"
      title="Final composition"
      subtitle="Where the nutrient ended up at run completion"
    >
      <div
        className="report-card__body--chart"
        style={{
          height: includeSecondary ? 280 : 240,
          display: "flex",
          gap: 8,
        }}
      >
        <Donut
          slot="primary"
          summary={summary}
          name={primaryName}
          colors={colors}
        />
        {includeSecondary ? (
          <Donut
            slot="secondary"
            summary={summary}
            name={secondaryName}
            colors={colors}
          />
        ) : null}
      </div>
    </ChartCard>
  );
}

function Donut({
  slot,
  summary,
  name,
  colors,
}: {
  slot: "primary" | "secondary";
  summary: RunSummary;
  name: string;
  colors: Record<string, string>;
}) {
  const partition = useMemo(() => finalPartition(summary, slot), [summary, slot]);
  const data = [
    { name: "retention", value: partition.retention, fill: colors.r1 },
    { name: "leached", value: partition.leached, fill: colors.r2 },
    { name: "degraded", value: partition.degraded, fill: colors.wall },
    { name: "precipitated", value: partition.precipitated, fill: colors.bubbles },
  ];
  return (
    <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            innerRadius="55%"
            outerRadius="85%"
            stroke={colors.bg}
            strokeWidth={1}
            isAnimationActive={false}
          >
            {data.map((d) => (
              <Cell key={d.name} fill={d.fill} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: colors.bg,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              color: colors.text1,
              fontSize: 12,
            }}
            formatter={(v: number, n: string) => [`${v.toFixed(2)} %`, n]}
          />
          <Legend
            wrapperStyle={{ fontSize: 10 }}
            iconSize={8}
            verticalAlign="bottom"
          />
        </PieChart>
      </ResponsiveContainer>
      {/* Centre label -- rendered as an overlaid div so it doesn't
          scale weirdly with the donut radius the way Recharts'
          built-in label rendering does. */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          pointerEvents: "none",
          textAlign: "center",
          paddingBottom: 30,
        }}
      >
        <span
          style={{
            color: colors.text3,
            fontSize: 10,
            textTransform: "uppercase",
            letterSpacing: 0.4,
          }}
        >
          {name}
        </span>
        <span
          style={{
            color: colors.text1,
            fontSize: 22,
            fontWeight: 600,
            lineHeight: 1.1,
          }}
        >
          {partition.retention.toFixed(1)}%
        </span>
        <span style={{ color: colors.text3, fontSize: 10 }}>retained</span>
      </div>
    </div>
  );
}
