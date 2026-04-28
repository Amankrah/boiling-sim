// Pure-data helpers shared by the Results-page chart cards. Extracted
// out of the card components so the math is unit-testable and so the
// same milestone helpers can colour-code multiple cards (e.g. heat-up
// storyline + boiling vigor both use `tSat` as a reference line).

import type { ScalarRow, RunSummary } from "../../hooks/useRunArtefacts";

// ---------------------------------------------------------------------------
// Milestones: discrete time markers extracted from the time-series.
// All fields optional -- a short or low-flux run may never cross
// every threshold.
// ---------------------------------------------------------------------------

export interface Milestones {
  /** Seconds at which the bulk water mean temperature first hits the
   *  near-saturation threshold (default 99 deg C). */
  tSat?: number;
  /** Seconds of the first sample with at least one active bubble. */
  tFirstBubble?: number;
  /** Seconds at which retention first dropped below 99.5 % (i.e. the
   *  carrot has lost more than 0.5 percentage points of nutrient). */
  tFirstLoss?: number;
}

const DEFAULT_T_SAT_C = 99.0;
/** A "first loss" milestone fires as soon as retention drops by 0.5 pp;
 *  earlier than the original 99 % threshold so it captures the
 *  nutrient transient rather than the steady-state. */
const DEFAULT_LOSS_THRESHOLD_PCT = 99.5;

export function findMilestones(
  scalars: ScalarRow[],
  opts: {
    saturationC?: number;
    retentionThresholdPct?: number;
  } = {},
): Milestones {
  const saturationC = opts.saturationC ?? DEFAULT_T_SAT_C;
  const retentionThresholdPct =
    opts.retentionThresholdPct ?? DEFAULT_LOSS_THRESHOLD_PCT;

  let tSat: number | undefined;
  let tFirstBubble: number | undefined;
  let tFirstLoss: number | undefined;

  for (const row of scalars) {
    if (
      tSat === undefined &&
      Number.isFinite(row.T_mean_water_c) &&
      row.T_mean_water_c >= saturationC
    ) {
      tSat = row.t;
    }
    if (
      tFirstBubble === undefined &&
      Number.isFinite(row.n_active_bubbles) &&
      row.n_active_bubbles >= 1
    ) {
      tFirstBubble = row.t;
    }
    if (
      tFirstLoss === undefined &&
      Number.isFinite(row.retention_pct) &&
      row.retention_pct < retentionThresholdPct
    ) {
      tFirstLoss = row.t;
    }
    if (
      tSat !== undefined &&
      tFirstBubble !== undefined &&
      tFirstLoss !== undefined
    ) {
      break;
    }
  }
  return { tSat, tFirstBubble, tFirstLoss };
}

// ---------------------------------------------------------------------------
// Loss rate: d/dt(retention_pct) per solute, smoothed.
// ---------------------------------------------------------------------------

export interface LossRateRow {
  t: number;
  /** Loss rate in %/min for the primary solute. Positive = nutrient
   *  is leaving the carrot (retention falling). */
  rate1: number;
  /** Same for secondary solute, when active. */
  rate2?: number;
}

/** Central-difference d/dt of a numeric column on a sorted ScalarRow
 *  list, then convert to %/min and apply a 3-sample symmetric moving
 *  average to smooth out 30 Hz snapshot noise without blurring the
 *  saturation transient. Boundary points use one-sided diffs. */
function diffSmooth(
  rows: ScalarRow[],
  key: keyof ScalarRow,
): number[] {
  const n = rows.length;
  if (n < 2) return rows.map(() => 0);

  const raw = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    if (i === 0) {
      const dt = rows[1].t - rows[0].t;
      raw[i] = dt > 0 ? -((rows[1][key] as number) - (rows[0][key] as number)) / dt : 0;
    } else if (i === n - 1) {
      const dt = rows[n - 1].t - rows[n - 2].t;
      raw[i] = dt > 0 ? -((rows[n - 1][key] as number) - (rows[n - 2][key] as number)) / dt : 0;
    } else {
      const dt = rows[i + 1].t - rows[i - 1].t;
      raw[i] = dt > 0 ? -((rows[i + 1][key] as number) - (rows[i - 1][key] as number)) / dt : 0;
    }
    // Convert from %/s to %/min so the y-axis is human-friendly.
    raw[i] = raw[i] * 60;
    if (!Number.isFinite(raw[i])) raw[i] = 0;
  }

  // 3-sample moving average. Endpoints reuse the nearest-pair mean.
  const smooth = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    if (i === 0) smooth[i] = (raw[0] + raw[1]) / 2;
    else if (i === n - 1) smooth[i] = (raw[n - 2] + raw[n - 1]) / 2;
    else smooth[i] = (raw[i - 1] + raw[i] + raw[i + 1]) / 3;
  }
  return smooth;
}

export function computeLossRate(
  rows: ScalarRow[],
  includeSecondary: boolean,
): LossRateRow[] {
  if (rows.length === 0) return [];
  const r1 = diffSmooth(rows, "retention_pct");
  const r2 = includeSecondary ? diffSmooth(rows, "retention2_pct") : null;
  return rows.map((row, i) => {
    const out: LossRateRow = { t: row.t, rate1: r1[i] };
    if (r2) out.rate2 = r2[i];
    return out;
  });
}

// ---------------------------------------------------------------------------
// Final partition extracted from the summary JSON for the donut chart.
// ---------------------------------------------------------------------------

export interface FinalPartition {
  retention: number;
  leached: number;
  degraded: number;
  precipitated: number;
}

export function finalPartition(
  summary: RunSummary,
  slot: "primary" | "secondary",
): FinalPartition {
  const f = summary.final;
  const num = (k: string): number => {
    const v = f[k];
    return typeof v === "number" && Number.isFinite(v) ? v : 0;
  };
  if (slot === "primary") {
    return {
      retention: num("retention_pct"),
      leached: num("leached_pct"),
      degraded: num("degraded_pct"),
      precipitated: num("precipitated_pct"),
    };
  }
  return {
    retention: num("retention2_pct"),
    leached: num("leached2_pct"),
    degraded: num("degraded2_pct"),
    precipitated: num("precipitated2_pct"),
  };
}

/** True when the secondary solute has any non-default activity --
 *  matches the heuristic already in ResultsReport.tsx so the new
 *  cards stay aligned with the rest of the page. */
export function hasSecondarySolute(summary: RunSummary): boolean {
  if (
    summary.nutrient_secondary_name &&
    summary.nutrient_secondary_name !== ""
  ) {
    return true;
  }
  const r2 = summary.final.retention2_pct ?? 100;
  const l2 = summary.final.leached2_pct ?? 0;
  return r2 < 99.99 || l2 > 0.01;
}
