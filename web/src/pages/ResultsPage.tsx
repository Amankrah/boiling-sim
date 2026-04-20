// Results view: Phase-4-style report rendered from the latest
// completed run's summary.json + scalars.csv. Fetches on mount and
// refetches whenever a new run completes (detected via
// snapshot.run_id + is_complete going true).

import { ResultsReport } from "../components/ResultsReport/ResultsReport";
import { Button } from "../components/ui/Button";
import { useRunArtefacts } from "../hooks/useRunArtefacts";
import type { Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot | null;
  onGotoConfig: () => void;
}

export function ResultsPage({ snapshot, onGotoConfig }: Props) {
  // Bumping the reloadKey forces useRunArtefacts to refetch. Key on
  // run_id when is_complete so each newly-finished run pulls fresh
  // artefacts; while in-flight we key on a constant so the hook
  // doesn't thrash.
  const reloadKey =
    snapshot?.is_complete && snapshot.run_id
      ? `complete:${snapshot.run_id}`
      : "idle";

  const { status, refresh } = useRunArtefacts({ runId: "latest", reloadKey });

  if (status.state === "loading" || status.state === "idle") {
    return (
      <div className="results-layout">
        <div className="config-placeholder">
          <h2 style={{ fontSize: "var(--text-lg)", fontWeight: 600, margin: 0 }}>
            Loading results…
          </h2>
          <p className="muted">Fetching /api/runs/latest/…</p>
        </div>
      </div>
    );
  }

  if (status.state === "empty") {
    return (
      <div className="results-layout">
        <div className="config-placeholder">
          <h2 style={{ fontSize: "var(--text-lg)", fontWeight: 600, margin: 0 }}>
            No completed run yet
          </h2>
          <p className="muted" style={{ maxWidth: 560 }}>
            Configure a run from the <strong>Config</strong> tab and click
            <strong> Apply &amp; Start Run</strong> with a non-zero duration.
            Results land here automatically when the sim reaches
            <code> total_time_s</code> and artefacts are written.
          </p>
          <div
            className="control-row"
            style={{ marginTop: "var(--space-3)" }}
          >
            <Button variant="primary" onClick={onGotoConfig}>
              Go to Config →
            </Button>
            <Button onClick={refresh}>Check again</Button>
          </div>
        </div>
      </div>
    );
  }

  if (status.state === "error") {
    return (
      <div className="results-layout">
        <div className="config-placeholder">
          <h2 style={{ fontSize: "var(--text-lg)", fontWeight: 600, margin: 0 }}>
            Failed to load results
          </h2>
          <p className="muted mono" style={{ maxWidth: 720 }}>
            {status.message}
          </p>
          <Button onClick={refresh}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="results-layout">
      <ResultsReport
        artefacts={status.artefacts}
        onStartNewRun={onGotoConfig}
      />
    </div>
  );
}
