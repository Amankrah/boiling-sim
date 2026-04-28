//! Phase 6 Rust relay: Python producer -> Rust fan-out -> browser WS.
//!
//! Three background tasks share a single [`AppState`]:
//!
//! * TCP ingest on 127.0.0.1:8765 (msgpack snapshot frames from Python)
//! * TCP control forwarder on 127.0.0.1:8766 (newline-JSON control
//!   messages from browser toward Python)
//! * Axum HTTP + WebSocket server on 0.0.0.0:8080 (`/health`, `/stream`)

use std::net::SocketAddr;

use axum::{routing::get, Router};
use tracing::info;
use tracing_subscriber::EnvFilter;

use ws_server::{
    app::AppState, control_forward, ingest, runs, ws::ws_handler,
};

const HTTP_ADDR: &str = "0.0.0.0:8080";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let state = AppState::new();

    // Background: TCP ingest + control forwarder.
    {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = ingest::run(ingest::DEFAULT_INGEST_ADDR, state).await {
                tracing::error!("ingest task exited: {e}");
            }
        });
    }
    {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) =
                control_forward::run(control_forward::DEFAULT_CONTROL_ADDR, state).await
            {
                tracing::error!("control forwarder exited: {e}");
            }
        });
    }

    // Foreground: HTTP + WebSocket + Phase 6.6 run artefact endpoints.
    let app = Router::new()
        .route("/health", get(health))
        .route("/stream", get(ws_handler))
        .route("/api/runs", get(runs::list_runs))
        .route("/api/runs/:run_id/summary.json", get(runs::get_summary))
        .route("/api/runs/:run_id/scalars.csv", get(runs::get_scalars))
        .route("/api/runs/:run_id/data.h5", get(runs::get_hdf5))
        .with_state(state);

    let addr: SocketAddr = HTTP_ADDR.parse()?;
    info!("ws-server listening on http://{addr}");
    info!("  websocket: ws://{addr}/stream");
    info!("  ingest   : {}", ingest::DEFAULT_INGEST_ADDR);
    info!("  control  : {}", control_forward::DEFAULT_CONTROL_ADDR);
    {
        // Resolve + canonicalise so the operator can verify at a glance
        // that ws-server and the Python producer are reading/writing
        // the same physical directory. The historical
        // `artefacts: ./dashboard_runs` line hid a recurring footgun
        // where ws-server's cwd-relative default drifted from Python's
        // absolute project-root default — see runs::artefact_dir docs.
        let (path, source) = runs::artefact_dir_with_source();
        let resolved = path.canonicalize().unwrap_or(path);
        info!("  artefacts: {} ({})", resolved.display(), source.label());
    }

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "boiling-sim ws-server alive"
}
