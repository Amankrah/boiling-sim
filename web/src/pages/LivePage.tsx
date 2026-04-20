// Live view: 3D scene (hero) + compact control panel + time-series
// strip. Extracted from the monolithic App.tsx in Phase 6.6 M5 so
// the new Config / Results pages can share a common shell.
//
// This component is deliberately dumb: all state lives in App, and
// it takes exactly the props the live UI needs. No data fetching
// here (WebSocket stays at App level so the connection survives
// across tab switches).

import { BoilingScene, type CameraPose } from "../components/BoilingScene";
import { ControlPanel } from "../components/ControlPanel";
import { RebuildBanner } from "../components/RebuildBanner";
import { SceneOverlay } from "../components/SceneOverlay";
import { TimeSeriesPanel } from "../components/TimeSeriesPanel";
import type { UseSnapshotReturn } from "../hooks/useSnapshot";
import type { ShareableParams } from "../share";
import type { ControlMessage, Snapshot } from "../types/snapshot";

interface Props {
  snapshot: Snapshot | null;
  history: UseSnapshotReturn["history"];
  historyVersion: number;
  sendCommand: (cmd: ControlMessage) => void;
  params: ShareableParams;
  onParamsChange: (next: ShareableParams) => void;
  initialCamera: CameraPose;
  onCameraChange: (pose: CameraPose) => void;
  onCopyShareLink: () => void | Promise<void>;
  onOpenConfig: () => void;
  showDebug: boolean;
}

export function LivePage({
  snapshot,
  history,
  historyVersion,
  sendCommand,
  params,
  onParamsChange,
  initialCamera,
  onCameraChange,
  onCopyShareLink,
  onOpenConfig,
  showDebug,
}: Props) {
  return (
    <div className="live-layout">
      <section className="app__scene">
        {snapshot ? (
          <BoilingScene
            snapshot={snapshot}
            initialCamera={initialCamera}
            onCameraChange={onCameraChange}
            showStats={showDebug}
          />
        ) : (
          <div className="scene-placeholder">
            Waiting for first snapshot from the Rust relay…
          </div>
        )}
        {snapshot?.is_rebuilding ? <RebuildBanner /> : null}
      </section>

      <aside className="app__controls">
        {snapshot ? (
          <SceneOverlay snapshot={snapshot} variant="sidebar" />
        ) : null}
        <ControlPanel
          snapshot={snapshot}
          params={params}
          onParamsChange={onParamsChange}
          sendCommand={sendCommand}
          onCopyShareLink={onCopyShareLink}
          onOpenConfig={onOpenConfig}
        />
      </aside>

      <section className="app__plots">
        <TimeSeriesPanel history={history} historyVersion={historyVersion} />
      </section>
    </div>
  );
}
