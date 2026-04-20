//! End-to-end test for the Phase 6.6 run-artefact HTTP endpoints.
//! Spins up an Axum server pointed at a tempdir with fixture files,
//! hits each endpoint via reqwest-style tokio-tungstenite (we just
//! use `reqwest` via its fluent client API bundled in dev-deps).

use std::fs;
use std::sync::{Mutex, OnceLock};

use axum::{routing::get, Router};
use tokio::net::TcpListener;

use ws_server::app::AppState;
use ws_server::runs;

/// Serialise access to the global `BOILINGSIM_ARTIFACTS_DIR` env var
/// so parallel Cargo tests don't clobber each other's tmpdirs. Each
/// test acquires this lock once, sets the env, starts its server,
/// runs the assertions, and releases on drop.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Spin up a local server on an ephemeral port; set
/// BOILINGSIM_ARTIFACTS_DIR so the handlers read from the tmpdir.
async fn spin_up_with_dir(dir: &std::path::Path) -> String {
    std::env::set_var("BOILINGSIM_ARTIFACTS_DIR", dir);
    let state = AppState::new();
    let app = Router::new()
        .route("/api/runs", get(runs::list_runs))
        .route("/api/runs/:run_id/summary.json", get(runs::get_summary))
        .route("/api/runs/:run_id/scalars.csv", get(runs::get_scalars))
        .route("/api/runs/:run_id/data.h5", get(runs::get_hdf5))
        .with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    format!("http://{addr}")
}

fn write_fixture(dir: &std::path::Path, run_id: &str, nutrient: &str, t_sim: f64) {
    fs::write(
        dir.join(format!("{run_id}.json")),
        serde_json::json!({
            "run_id": run_id,
            "schema_version": 3,
            "n_samples": 5,
            "t_sim_total_s": t_sim,
            "nutrient_primary_name": nutrient,
            "final": {"retention_pct": 88.7},
            "acceptance": [],
            "mass_balance": {"max_abs_drift_pct": 0.01},
            "parameters": {},
        }).to_string(),
    ).unwrap();
    fs::write(
        dir.join(format!("{run_id}.csv")),
        "t,dt,retention_pct\n0.0,0.001,100.0\n1.0,0.001,99.5\n",
    ).unwrap();
    fs::write(
        dir.join(format!("{run_id}.h5")),
        b"\x89HDF\r\n\x1a\n_fake_h5_payload_",
    ).unwrap();
}

#[tokio::test]
async fn list_runs_returns_sorted_entries() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    // Two fixtures, different mtimes — the second should come first
    // in the sort.
    write_fixture(tmp.path(), "11111111111111111111111111111111", "β-carotene", 600.0);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    write_fixture(tmp.path(), "22222222222222222222222222222222", "vitamin C", 300.0);

    let base = spin_up_with_dir(tmp.path()).await;
    let body = reqwest::get(format!("{base}/api/runs")).await.unwrap().text().await.unwrap();
    let entries: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["run_id"], "22222222222222222222222222222222");
    assert_eq!(entries[0]["nutrient_primary_name"], "vitamin C");
    assert_eq!(entries[1]["run_id"], "11111111111111111111111111111111");
}

#[tokio::test]
async fn summary_endpoint_streams_json() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    write_fixture(tmp.path(), "aabbccddeeff00112233445566778899", "β-carotene", 120.0);
    let base = spin_up_with_dir(tmp.path()).await;
    let resp = reqwest::get(format!("{base}/api/runs/aabbccddeeff00112233445566778899/summary.json")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "application/json"
    );
    let j: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(j["run_id"], "aabbccddeeff00112233445566778899");
    assert_eq!(j["schema_version"], 3);
}

#[tokio::test]
async fn csv_endpoint_serves_text_csv() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    write_fixture(tmp.path(), "aabbccddeeff00112233445566778899", "β-carotene", 120.0);
    let base = spin_up_with_dir(tmp.path()).await;
    let resp = reqwest::get(format!("{base}/api/runs/aabbccddeeff00112233445566778899/scalars.csv")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap().to_string();
    assert!(ct.starts_with("text/csv"));
    let body = resp.text().await.unwrap();
    assert!(body.starts_with("t,dt,retention_pct"));
}

#[tokio::test]
async fn hdf5_endpoint_serves_binary() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    write_fixture(tmp.path(), "aabbccddeeff00112233445566778899", "β-carotene", 120.0);
    let base = spin_up_with_dir(tmp.path()).await;
    let resp = reqwest::get(format!("{base}/api/runs/aabbccddeeff00112233445566778899/data.h5")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "application/x-hdf5"
    );
    let bytes = resp.bytes().await.unwrap();
    // The HDF5 file starts with the \x89HDF magic number.
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
}

#[tokio::test]
async fn latest_endpoint_resolves_to_newest() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    write_fixture(tmp.path(), "11111111111111111111111111111111", "older", 60.0);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    write_fixture(tmp.path(), "22222222222222222222222222222222", "newer", 90.0);

    let base = spin_up_with_dir(tmp.path()).await;
    let resp = reqwest::get(format!("{base}/api/runs/latest/summary.json")).await.unwrap();
    // Read headers BEFORE consuming the body (reqwest::Response::json takes self).
    let cc = resp
        .headers()
        .get("cache-control")
        .map(|v| v.to_str().unwrap_or("").to_string())
        .unwrap_or_default();
    assert!(cc.contains("no-store"), "latest alias must be uncacheable; got {cc:?}");
    let j: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(j["run_id"], "22222222222222222222222222222222");
}

#[tokio::test]
async fn latest_returns_404_when_empty() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let base = spin_up_with_dir(tmp.path()).await;
    let resp = reqwest::get(format!("{base}/api/runs/latest/summary.json")).await.unwrap();
    assert_eq!(resp.status(), 404);
    let j: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(j["status"], "no_completed_run");
}

#[tokio::test]
async fn malformed_run_id_rejected() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let base = spin_up_with_dir(tmp.path()).await;
    // Path traversal attempt.
    let resp = reqwest::get(format!("{base}/api/runs/..%2fsecrets/summary.json")).await.unwrap();
    assert_eq!(resp.status(), 400);
    // Too-short id.
    let resp2 = reqwest::get(format!("{base}/api/runs/abc/summary.json")).await.unwrap();
    assert_eq!(resp2.status(), 400);
}
