//! ws-server library: snapshot wire format, TCP ingest pipeline,
//! control-message forwarder, and WebSocket handler.
//!
//! The binary entrypoint lives in `src/main.rs` and wires these
//! modules into a Tokio runtime.

pub mod app;
pub mod control;
pub mod control_forward;
pub mod ingest;
pub mod runs;
pub mod snapshot;
pub mod ws;

pub use app::AppState;
