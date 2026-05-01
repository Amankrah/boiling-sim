// useSnapshot -- single WebSocket subscription + msgpack + fzstd
// decoder + auto-reconnect. Returns the latest snapshot, connection
// state, and a sendCommand callback for dispatching ControlMessages
// back to the Rust relay (which forwards them on to Python).

import { decode as msgpackDecode } from "@msgpack/msgpack";
import { decompress as fzstdDecompress } from "fzstd";
import { useCallback, useEffect, useRef, useState } from "react";

import type {
  ConnectionState,
  ControlMessage,
  Snapshot,
  SnapshotSummary,
} from "../types/snapshot";
import { SCHEMA_VERSION, summarizeSnapshot } from "../types/snapshot";

/**
 * Reinterpret a msgpack-decoded `Uint8Array` (raw little-endian f32
 * bytes) as a `Float32Array`. Copies into a freshly-allocated
 * `ArrayBuffer` because msgpack-decoder Uint8Arrays are slices into a
 * larger buffer whose byteOffset is not guaranteed to be 4-byte
 * aligned (a `Float32Array` view requires alignment). The copy runs
 * at memcpy speed (~tens of µs for a 2.7 MB field at 15 Hz).
 *
 * Browsers are uniformly little-endian on x86 + ARM64 / Apple Silicon,
 * which matches the Python producer's `numpy.tobytes()` layout. We do
 * not handle big-endian hosts -- the dashboard would render garbled
 * temperatures and the issue would be obvious immediately.
 */
function f32ArrayFromBytes(buf: Uint8Array): Float32Array {
  const aligned = new ArrayBuffer(buf.byteLength);
  new Uint8Array(aligned).set(buf);
  return new Float32Array(aligned);
}

export interface UseSnapshotOptions {
  url: string;
  /** Max snapshots held in the history ring (for Recharts in M6). */
  historyLen?: number;
  /** Reconnect delay in ms after a non-clean close. */
  reconnectDelayMs?: number;
}

export interface UseSnapshotReturn {
  snapshot: Snapshot | null;
  /**
   * Scalar-only history ring for Recharts. Full snapshots carry
   * ~700 KB of volume+alpha arrays each -- retaining 1800 of them
   * (60 s at 30 Hz) blows the JS heap past 2 GB and Chrome refuses
   * the tab with "not enough memory to open this page". See
   * summarizeSnapshot in types/snapshot.ts.
   */
  history: SnapshotSummary[];
  /**
   * Monotonic integer bumped once per received snapshot. Downstream
   * components memoize chart data on `historyVersion` so they re-render
   * with fresh history without forcing the entire component tree to
   * rebuild every frame. See TimeSeriesPanel for the intended use.
   */
  historyVersion: number;
  connectionState: ConnectionState;
  sendCommand: (cmd: ControlMessage) => void;
  /** Diagnostics surfaced in the connection pill / devtools. */
  frameCount: number;
  lastFrameAt: number | null;
  lastError: string | null;
}

export function useSnapshot(options: UseSnapshotOptions): UseSnapshotReturn {
  const { url, historyLen = 1800, reconnectDelayMs = 1000 } = options;

  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const [frameCount, setFrameCount] = useState(0);
  const [lastFrameAt, setLastFrameAt] = useState<number | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  // Ring buffer for plots. Summaries only -- the volume arrays on a
  // full Snapshot are ~700 KB each and keeping 1800 of them blows the
  // JS heap past 2 GB (Chrome throws "not enough memory to open this
  // page"). Summaries are <100 bytes each, so 1800 of them = ~180 KB
  // -- trivial -- while still carrying every field the time-series
  // plots read.
  const historyRef = useRef<SnapshotSummary[]>([]);
  const [historyVersion, setHistoryVersion] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  // Throttle historyVersion bumps so Recharts re-memos at ~5 Hz, not
  // 30 Hz. Without this throttle, five chart cards re-render per
  // incoming snapshot, pinning the main thread and backing up the
  // WebSocket broadcast channel (observed as 5-10 `ws client lagged`
  // bursts on the server side). Plots don't need 30 Hz updates --
  // retention and wall-T change on sim-second timescales.
  const lastHistoryBumpMsRef = useRef<number>(0);

  const sendCommand = useCallback((cmd: ControlMessage) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    } else {
      // Stash a warning for devtools without spamming the UI.
      console.warn("[dashboard] sendCommand while ws not open:", cmd);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      if (!mountedRef.current) return;
      setConnectionState("connecting");
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionState("open");
        setLastError(null);
      };

      ws.onmessage = (ev: MessageEvent) => {
        if (!(ev.data instanceof ArrayBuffer)) {
          // Control / text frames aren't expected from the server.
          return;
        }
        try {
          const compressed = new Uint8Array(ev.data);
          const mpBytes = fzstdDecompress(compressed);
          // v5: temperature/alpha arrive as Uint8Array (msgpack bin).
          // Decode into a typed object whose float fields are real
          // Float32Arrays before the snapshot reaches React state.
          const raw = msgpackDecode(mpBytes) as Omit<
            Snapshot,
            "temperature" | "alpha"
          > & {
            temperature: Uint8Array;
            alpha: Uint8Array;
          };
          if (raw.version !== SCHEMA_VERSION) {
            setLastError(
              `schema version mismatch: got ${raw.version}, expected ${SCHEMA_VERSION}. Rebuild your client.`,
            );
            return;
          }
          const decoded: Snapshot = {
            ...raw,
            temperature: f32ArrayFromBytes(raw.temperature),
            alpha: f32ArrayFromBytes(raw.alpha),
          };
          setSnapshot(decoded);
          setFrameCount((n) => n + 1);
          setLastFrameAt(performance.now());
          // Append a SUMMARY to the ring, not the full snapshot. The
          // full snapshot is held only in React state (`snapshot`
          // above) for the latest frame; React's reconciler drops
          // the previous one on each update so the huge volume
          // arrays stay transient.
          const hist = historyRef.current;
          hist.push(summarizeSnapshot(decoded));
          if (hist.length > historyLen) {
            hist.splice(0, hist.length - historyLen);
          }
          // Throttle the historyVersion bump to ~5 Hz (200 ms) so
          // Recharts only re-memos + re-renders the plot strip at a
          // rate the main thread can actually sustain. At 30 Hz
          // WebSocket arrival this is 6x less chart work. Plot values
          // update smoothly enough at 5 Hz (temperature and retention
          // change on sim-second timescales); volume + bubbles still
          // update every frame via the full `snapshot` state.
          const nowMs = performance.now();
          if (nowMs - lastHistoryBumpMsRef.current >= 200) {
            lastHistoryBumpMsRef.current = nowMs;
            setHistoryVersion((v) => (v + 1) & 0x7fff_ffff);
          }
        } catch (err) {
          setLastError(
            `frame decode failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      };

      ws.onerror = (ev: Event) => {
        setLastError("websocket error");
        setConnectionState("error");
        // Let onclose handle reconnect scheduling to avoid duplicate timers.
        void ev;
      };

      ws.onclose = () => {
        setConnectionState("closed");
        wsRef.current = null;
        if (mountedRef.current) {
          reconnectTimerRef.current = window.setTimeout(connect, reconnectDelayMs);
        }
      };
    };

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [url, reconnectDelayMs, historyLen]);

  // historyVersion is the change trigger downstream components key
  // their useMemo on; the ref holds the actual ring.
  const history = historyRef.current;

  return {
    snapshot,
    history,
    historyVersion,
    connectionState,
    sendCommand,
    frameCount,
    lastFrameAt,
    lastError,
  };
}
