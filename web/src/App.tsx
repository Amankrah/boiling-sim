// Phase 6.6 router shell. App.tsx owns the shared data layer (the
// single WebSocket via useSnapshot, the share-state params/camera,
// the debug toggle) and picks which page to render based on the
// `usePage` hook. Keeping the WebSocket at App level means switching
// tabs doesn't disconnect or reset the snapshot stream.
//
//   +----------------------------+
//   | TopBar (brand + nav tabs)  |
//   +----------------------------+
//   | <app__main>                |
//   |   LivePage | ConfigPage |  |
//   |   ResultsPage              |
//   +----------------------------+

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { TopBar } from "./components/TopBar";
import { usePage } from "./hooks/usePage";
import { useSnapshot } from "./hooks/useSnapshot";
import { ConfigPage } from "./pages/ConfigPage";
import { LivePage } from "./pages/LivePage";
import { ResultsPage } from "./pages/ResultsPage";
import {
  DEFAULT_SHARE_STATE,
  buildShareUrl,
  decodeShareState,
  pushShareState,
  type ShareState,
  type ShareableParams,
} from "./share";
import type { CameraPose } from "./components/BoilingScene";

const WS_URL = (() => {
  const fromEnv = (import.meta as unknown as { env?: Record<string, string> }).env
    ?.VITE_WS_URL;
  if (fromEnv) return fromEnv;
  if (typeof window === "undefined") return "ws://localhost:8080/stream";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/stream`;
})();

export function App() {
  const {
    snapshot,
    history,
    historyVersion,
    connectionState,
    sendCommand,
    frameCount,
    lastFrameAt,
    lastError,
  } = useSnapshot({ url: WS_URL });

  const { page, setPage } = usePage();

  // Share state is authoritative: read from URL on mount, write back
  // on every change via pushShareState (which preserves `?page=`).
  const initialShareState = useMemo<ShareState>(() => decodeShareState(), []);
  const [params, setParams] = useState<ShareableParams>(initialShareState.params);
  const [camera, setCamera] = useState<CameraPose>(initialShareState.camera);
  const [showDebug, setShowDebug] = useState(false);

  // Seed Python with URL-derived params once the WS is open, for
  // fields that differ from the scenario default.
  const seededRef = useRef(false);
  useEffect(() => {
    if (seededRef.current || connectionState !== "open") return;
    seededRef.current = true;
    const d = DEFAULT_SHARE_STATE.params;
    if (params.heatFluxWPerM2 !== d.heatFluxWPerM2) {
      sendCommand({ type: "set_heat_flux", value: params.heatFluxWPerM2 });
    }
    if (params.material !== d.material) {
      sendCommand({ type: "set_material", value: params.material });
    }
    if (
      params.carrotDiameterMm !== d.carrotDiameterMm ||
      params.carrotLengthMm !== d.carrotLengthMm
    ) {
      sendCommand({
        type: "set_carrot_size",
        diameter_mm: params.carrotDiameterMm,
        length_mm: params.carrotLengthMm,
      });
    }
  }, [connectionState, params, sendCommand]);

  useEffect(() => {
    pushShareState({ params, camera });
  }, [params, camera]);

  const handleCameraChange = useCallback((pose: CameraPose) => {
    setCamera(pose);
  }, []);

  const handleCopyShareLink = useCallback(async () => {
    const url = buildShareUrl({ params, camera });
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      window.open(url, "_blank", "noopener,noreferrer");
    }
  }, [params, camera]);

  // "New results ready" badge on the Results tab lights up the first
  // time we see is_complete=true. Cleared when the user visits the
  // Results page.
  const [resultsReady, setResultsReady] = useState(false);
  useEffect(() => {
    if (snapshot?.is_complete && page !== "results") {
      setResultsReady(true);
    }
    if (page === "results") {
      setResultsReady(false);
    }
  }, [snapshot?.is_complete, page]);

  return (
    <div className="app">
      <TopBar
        wsUrl={WS_URL}
        connectionState={connectionState}
        frameCount={frameCount}
        lastFrameAt={lastFrameAt}
        lastError={lastError}
        showDebug={showDebug}
        onToggleDebug={() => setShowDebug((v) => !v)}
        page={page}
        onPageChange={setPage}
        resultsReady={resultsReady}
      />
      <div className="app__main">
        {page === "live" ? (
          <LivePage
            snapshot={snapshot}
            history={history}
            historyVersion={historyVersion}
            sendCommand={sendCommand}
            params={params}
            onParamsChange={setParams}
            initialCamera={initialShareState.camera}
            onCameraChange={handleCameraChange}
            onCopyShareLink={handleCopyShareLink}
            onOpenConfig={() => setPage("config")}
            showDebug={showDebug}
          />
        ) : page === "config" ? (
          <ConfigPage
            snapshot={snapshot}
            params={params}
            sendCommand={sendCommand}
            onDone={() => setPage("live")}
          />
        ) : (
          <ResultsPage
            snapshot={snapshot}
            onGotoConfig={() => setPage("config")}
          />
        )}
      </div>

      {showDebug && snapshot ? (
        <pre className="debug-drawer">
          {JSON.stringify(
            {
              version: snapshot.version,
              t_sim: snapshot.t_sim,
              step: snapshot.step,
              run_id: snapshot.run_id,
              total_time_s: snapshot.total_time_s,
              is_complete: snapshot.is_complete,
              last_error: snapshot.last_error,
              grid: snapshot.grid,
              grid_ds: snapshot.grid_ds,
              temperature_len: snapshot.temperature.length,
              alpha_len: snapshot.alpha.length,
              bubbles_count: snapshot.bubbles.length,
              carrot_retention: snapshot.carrot_retention,
              carrot_retention2: snapshot.carrot_retention2,
              water_temperature_mean: snapshot.water_temperature_mean,
              wall_temperature_mean: snapshot.wall_temperature_mean,
              wall_heat_flux: snapshot.wall_heat_flux,
            },
            null,
            2,
          )}
        </pre>
      ) : null}
    </div>
  );
}
