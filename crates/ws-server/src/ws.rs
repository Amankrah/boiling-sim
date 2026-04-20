//! Axum WebSocket upgrade handler.
//!
//! For each connected client, this spawns a lightweight tokio task
//! that:
//!
//! 1. Subscribes to the snapshot broadcast channel, zstd-compresses
//!    each frame at level 3, and writes it as `Message::Binary`.
//! 2. Reads incoming `Message::Text` frames as
//!    [`crate::control::ControlMessage`] JSON and forwards them onto
//!    the control broadcast channel (which the control forwarder
//!    drains toward Python).
//!
//! Zstd compression happens per-client rather than once in the ingest
//! task because (a) the overwhelmingly common case is exactly one
//! connected client (the dashboard), (b) compression cost is ~1 ms
//! per 300 KB frame at level 3 -- trivial -- and (c) it keeps the
//! ingest-path hot bytes uncompressed so a future metrics tap can
//! inspect them without re-decompressing.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures_util::{sink::SinkExt, stream::StreamExt};
use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, info, warn};

use crate::app::AppState;
use crate::control::ControlMessage;

/// zstd compression level. Level 3 is the dev-guide spec and strikes
/// a ~4x ratio on dense temperature/alpha float arrays with sub-ms
/// CPU. Don't raise without profiling.
const ZSTD_LEVEL: i32 = 3;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    info!("ws client connected");

    let (mut ws_tx, mut ws_rx) = socket.split();
    let mut snap_rx = state.snapshots.subscribe();
    let control_tx = state.controls.clone();

    // Forward snapshots -> client as compressed binary frames.
    let snapshot_task = tokio::spawn(async move {
        loop {
            match snap_rx.recv().await {
                Ok(raw) => {
                    let compressed = match zstd::encode_all(&raw[..], ZSTD_LEVEL) {
                        Ok(c) => c,
                        Err(e) => {
                            warn!("zstd encode failed, dropping frame: {e}");
                            continue;
                        }
                    };
                    if let Err(e) = ws_tx.send(Message::Binary(compressed)).await {
                        debug!("ws send failed (client gone): {e}");
                        break;
                    }
                }
                Err(RecvError::Lagged(skipped)) => {
                    // Single-frame lag is normal backpressure when the
                    // client's zstd decode + paint cycle momentarily
                    // slips behind the 30 Hz producer; it doesn't
                    // warrant a warning-per-occurrence (produces log
                    // spam during browser GC / paint stalls). Log
                    // only when the burst is big enough to be a
                    // genuine user-visible hitch.
                    if skipped >= 5 {
                        warn!("ws client lagged, dropped {skipped} snapshot(s)");
                    }
                }
                Err(RecvError::Closed) => break,
            }
        }
    });

    // Read client control messages -> control broadcast.
    let control_task = tokio::spawn(async move {
        while let Some(msg) = ws_rx.next().await {
            let msg = match msg {
                Ok(m) => m,
                Err(e) => {
                    debug!("ws recv error: {e}");
                    break;
                }
            };
            match msg {
                Message::Text(text) => match ControlMessage::from_json(&text) {
                    Ok(cmd) => {
                        // `send` only errors when there are zero
                        // subscribers; that means the Python control
                        // forwarder isn't up yet. We log and drop --
                        // retaining stale control messages would mean
                        // surprise parameter changes when Python
                        // reconnects.
                        if state.controls.send(cmd.clone()).is_err() {
                            debug!("no control subscribers; dropping {:?}", cmd);
                        }
                    }
                    Err(e) => {
                        warn!("malformed control message: {e} (raw={text})");
                    }
                },
                Message::Binary(_) => {
                    warn!("unexpected binary frame from client (ignored)");
                }
                Message::Ping(_) | Message::Pong(_) => {}
                Message::Close(_) => break,
            }
        }
    });

    // When either half finishes (client closes, send fails), wind down.
    let _ = tokio::join!(snapshot_task, control_task);
    // control_tx is only used for subscriber-presence lookups; drop
    // explicitly for clarity.
    drop(control_tx);
    info!("ws client disconnected");
}
