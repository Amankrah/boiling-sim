// Fetcher for the Phase 6.6 run artefacts served by the Rust relay
// at `/api/runs/{id}/{summary.json, scalars.csv, data.h5}`.
//
// The Results page mounts this hook with id="latest". On mount it
// fetches both summary + CSV in parallel, parses the CSV, and
// exposes a typed bundle plus a refresh callback. A "new run
// complete" signal from the Live view (via a `reloadKey` prop)
// causes a refetch without remounting the page.

import { useCallback, useEffect, useState } from "react";

/** Shape of the summary.json emitted by python/boilingsim/run_writer.py. */
export interface RunSummary {
  run_id: string;
  schema_version: number;
  n_samples: number;
  wall_clock_s: number;
  t_sim_total_s: number;
  step_count: number;
  s_per_sim_s: number;
  snapshot_cadence_hz: number;
  nutrient_primary_name: string;
  nutrient_secondary_name: string;
  final: Record<string, number>;
  acceptance: Array<{ name: string; passed: boolean; detail: string }>;
  mass_balance: { max_abs_drift_pct: number; final_sum_pct: number };
  parameters: Record<string, unknown>;
}

/** One row of the scalars.csv — keys mirror `SCALAR_CSV_FIELDS`. */
export interface ScalarRow {
  t: number;
  dt: number;
  T_mean_water_c: number;
  T_max_water_c: number;
  T_min_water_c: number;
  T_max_wall_c: number;
  T_inner_wall_mean_c: number;
  T_inner_wall_max_c: number;
  u_max_mps: number;
  n_active_bubbles: number;
  mean_bubble_R_mm: number;
  mean_departed_bubble_R_mm: number;
  max_bubble_R_mm: number;
  alpha_min: number;
  retention_pct: number;
  leached_pct: number;
  degraded_pct: number;
  precipitated_pct: number;
  retention2_pct: number;
  leached2_pct: number;
  degraded2_pct: number;
  precipitated2_pct: number;
}

export interface RunArtefacts {
  runId: string;
  summary: RunSummary;
  scalars: ScalarRow[];
}

export type ArtefactStatus =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "ready"; artefacts: RunArtefacts }
  | { state: "empty" } // 404 from /latest: no completed runs yet.
  | { state: "error"; message: string };

interface UseRunArtefactsOptions {
  /** Either "latest" (alias) or a concrete 32-char hex run id. */
  runId: string;
  /** Bump to force a refetch without remounting. */
  reloadKey?: number | string;
}

export function useRunArtefacts(options: UseRunArtefactsOptions): {
  status: ArtefactStatus;
  refresh: () => void;
} {
  const { runId, reloadKey } = options;
  const [status, setStatus] = useState<ArtefactStatus>({ state: "idle" });

  const fetchArtefacts = useCallback(async () => {
    setStatus({ state: "loading" });
    try {
      const [summaryRes, scalarsRes] = await Promise.all([
        fetch(`/api/runs/${runId}/summary.json`, { cache: "no-store" }),
        fetch(`/api/runs/${runId}/scalars.csv`, { cache: "no-store" }),
      ]);
      if (summaryRes.status === 404) {
        setStatus({ state: "empty" });
        return;
      }
      if (!summaryRes.ok || !scalarsRes.ok) {
        setStatus({
          state: "error",
          message: `HTTP ${summaryRes.status} / ${scalarsRes.status}`,
        });
        return;
      }
      const summary = (await summaryRes.json()) as RunSummary;
      const csvText = await scalarsRes.text();
      const scalars = parseScalarsCsv(csvText);
      setStatus({
        state: "ready",
        artefacts: { runId: summary.run_id, summary, scalars },
      });
    } catch (err) {
      setStatus({
        state: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [runId]);

  useEffect(() => {
    void fetchArtefacts();
  }, [fetchArtefacts, reloadKey]);

  return { status, refresh: () => void fetchArtefacts() };
}

/** Hand-rolled CSV reader. The run-writer emits a fixed column set
 *  with no quoted fields / no embedded commas, so `line.split(",")`
 *  is sufficient. Header-row drives key names. Non-finite values
 *  become NaN on the way through `parseFloat`. */
export function parseScalarsCsv(text: string): ScalarRow[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length < 2) return [];
  const header = lines[0].split(",").map((s) => s.trim());
  const out: ScalarRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const fields = lines[i].split(",");
    const row: Partial<ScalarRow> = {};
    for (let c = 0; c < header.length; c++) {
      const key = header[c] as keyof ScalarRow;
      const raw = fields[c] ?? "";
      (row[key] as number) = parseFloat(raw);
    }
    out.push(row as ScalarRow);
  }
  return out;
}
