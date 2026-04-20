//! Shared state passed to every Axum handler + background task.
//!
//! There are two broadcast channels:
//!
//! * `snapshots` -- raw msgpack bytes straight from the Python ingest
//!   listener, wrapped in `Arc` so all WS clients share a single
//!   allocation per frame. Compression happens inside the WS handler
//!   (once per client) but the hot bytes stay shared.
//!
//! * `controls` -- `ControlMessage`s received from any WS client,
//!   fanned out to every Python control-forwarder connection. In
//!   practice there's exactly one Python subscriber, but the channel
//!   form lets us add a second (e.g. a debug tap) without rewiring.

use std::sync::Arc;

use tokio::sync::broadcast;

use crate::control::ControlMessage;

/// Maximum number of queued snapshots per subscriber before the
/// broadcast channel lags. At 30 Hz × 64-deep this buys ~2 s of
/// backpressure headroom before slow clients get dropped messages --
/// a clean UX signal ("frames stuttered") rather than a crash.
pub const SNAPSHOT_CHANNEL_CAPACITY: usize = 64;

/// Control messages are tiny and infrequent; 256 is overkill but free.
pub const CONTROL_CHANNEL_CAPACITY: usize = 256;

/// Cloneable handle carried through the Axum router + background tasks.
#[derive(Clone)]
pub struct AppState {
    pub snapshots: broadcast::Sender<Arc<Vec<u8>>>,
    pub controls: broadcast::Sender<ControlMessage>,
}

impl AppState {
    pub fn new() -> Self {
        let (snapshots, _) = broadcast::channel(SNAPSHOT_CHANNEL_CAPACITY);
        let (controls, _) = broadcast::channel(CONTROL_CHANNEL_CAPACITY);
        Self { snapshots, controls }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
