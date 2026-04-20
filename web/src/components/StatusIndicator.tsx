// Live-connection indicator. A coloured dot + label + frame-count
// meter, living in the top bar. Replaces the old absolute-positioned
// ConnectionPill so the status can share chrome with the brand and
// WS URL.
//
// Stale-frame detection: even if the WebSocket reports `open`, we
// flag "stale" when the newest frame is older than STALE_THRESHOLD_MS
// (default 2 s). That's the common symptom of a dead Python producer
// or a stalled Rust relay -- the socket stays up while the data
// stops.

import { useEffect, useState } from "react";

import type { ConnectionState } from "../types/snapshot";

const STALE_THRESHOLD_MS = 2000;
const POLL_MS = 500;

interface Props {
  state: ConnectionState;
  frameCount: number;
  lastFrameAt: number | null;
  error: string | null;
}

export function StatusIndicator({
  state,
  frameCount,
  lastFrameAt,
  error,
}: Props) {
  // `lastFrameAt` only updates on each new frame. To detect the
  // *absence* of frames we need a periodic re-render. One cheap
  // timer per mount.
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(performance.now()), POLL_MS);
    return () => window.clearInterval(id);
  }, []);

  const ageMs = lastFrameAt !== null ? (now - lastFrameAt) | 0 : null;
  const isStale =
    state === "open" && ageMs !== null && ageMs > STALE_THRESHOLD_MS;

  const effectiveState: ConnectionState | "stale" = isStale ? "stale" : state;
  const label =
    effectiveState === "open"
      ? `live · ${frameCount} frames`
      : effectiveState === "stale"
        ? `stalled · ${Math.round((ageMs ?? 0) / 1000)} s`
        : effectiveState === "error"
          ? "error"
          : effectiveState;
  const title =
    error ??
    (ageMs !== null
      ? `last frame: ${ageMs} ms ago`
      : "waiting for frames…");
  return (
    <span className={`status status--${effectiveState}`} title={title}>
      <span className="status__dot" aria-hidden />
      <span>{label}</span>
    </span>
  );
}
