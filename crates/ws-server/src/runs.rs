//! HTTP endpoints serving Phase 6.6 run artefacts.
//!
//! Python writes `{run_id}.h5`, `{run_id}.csv`, `{run_id}.json` to a
//! directory (configurable via `BOILINGSIM_ARTIFACTS_DIR`, default
//! `./dashboard_runs`). This module exposes four read-only routes:
//!
//! ```text
//! GET /api/runs                       -> list all completed runs (JSON)
//! GET /api/runs/{run_id}/summary.json -> stream the summary JSON
//! GET /api/runs/{run_id}/scalars.csv  -> stream the scalar time-series
//! GET /api/runs/{run_id}/data.h5      -> stream the HDF5 file
//! GET /api/runs/latest/...            -> alias to the most recently modified run
//! ```
//!
//! Run IDs are validated as 32-character lowercase hex (the
//! `uuid::Uuid::simple()` format Python writes) or the literal string
//! `"latest"` — keeps path traversal safe.

use std::path::PathBuf;

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tracing::{debug, warn};

use crate::app::AppState;

/// Default artefact directory if `BOILINGSIM_ARTIFACTS_DIR` is unset.
pub const DEFAULT_ARTEFACT_DIR: &str = "./dashboard_runs";

/// Returns the resolved artefact directory, honouring the env var.
pub fn artefact_dir() -> PathBuf {
    std::env::var("BOILINGSIM_ARTIFACTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ARTEFACT_DIR))
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RunEntry {
    pub run_id: String,
    /// Seconds since UNIX epoch the summary.json was last modified.
    /// Browser uses this to sort and to detect "new run available".
    pub created_at: f64,
    /// Simulated-time total (seconds) from the summary JSON.
    pub t_sim_total_s: f64,
    /// Nutrient label from the summary JSON (β-carotene / vitamin C / etc.).
    pub nutrient_primary_name: String,
}

/// Validate `run_id` so we never path-traverse. Accepts either:
///   - the literal `latest` sentinel, or
///   - a 32-char lowercase hex string (uuid4.hex).
fn validate_run_id(candidate: &str) -> Result<(), StatusCode> {
    if candidate == "latest" {
        return Ok(());
    }
    if candidate.len() != 32 || !candidate.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(())
}

/// Resolve `run_id == "latest"` to the most-recently-modified JSON in
/// the artefact directory. Returns 404 if the directory is empty.
fn resolve_latest(dir: &std::path::Path) -> Option<String> {
    let mut best: Option<(std::time::SystemTime, String)> = None;
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if stem.len() != 32 {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        match &best {
            None => best = Some((mtime, stem)),
            Some((best_mtime, _)) if mtime > *best_mtime => {
                best = Some((mtime, stem))
            }
            _ => {}
        }
    }
    best.map(|(_, id)| id)
}

/// `GET /api/runs` — enumerate completed runs.
///
/// Scans the artefact dir for `{run_id}.json` files, opens each, picks
/// out a few headline fields, and returns them as an array sorted by
/// `created_at` descending (newest first). Malformed / unparseable
/// summaries are silently skipped rather than failing the whole
/// listing.
pub async fn list_runs(State(_state): State<AppState>) -> Response {
    let dir = artefact_dir();
    let mut entries = Vec::<RunEntry>::new();

    let read = match std::fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return Json(entries).into_response(),
    };

    for item in read.flatten() {
        let path = item.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if stem.len() != 32 {
            continue;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                debug!("skip {}: read failed {}", path.display(), e);
                continue;
            }
        };
        let summary: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                debug!("skip {}: json parse {}", path.display(), e);
                continue;
            }
        };
        let created_at = item
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        entries.push(RunEntry {
            run_id: stem,
            created_at,
            t_sim_total_s: summary
                .get("t_sim_total_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            nutrient_primary_name: summary
                .get("nutrient_primary_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        });
    }
    entries.sort_by(|a, b| b.created_at.partial_cmp(&a.created_at).unwrap_or(std::cmp::Ordering::Equal));
    Json(entries).into_response()
}

/// Shared helper for the three streaming endpoints.
async fn serve_artefact(
    run_id: &str,
    suffix: &str,
    content_type: &str,
    content_disposition_filename: &str,
) -> Response {
    if let Err(code) = validate_run_id(run_id) {
        return (code, "invalid run_id").into_response();
    }
    let dir = artefact_dir();
    let resolved_id = if run_id == "latest" {
        match resolve_latest(&dir) {
            Some(id) => id,
            None => {
                return (StatusCode::NOT_FOUND, Json(serde_json::json!({
                    "status": "no_completed_run",
                    "detail": "No completed runs in the artefact directory yet.",
                }))).into_response();
            }
        }
    } else {
        run_id.to_string()
    };
    let path = dir.join(format!("{resolved_id}.{suffix}"));
    let mut file = match tokio::fs::File::open(&path).await {
        Ok(f) => f,
        Err(e) => {
            warn!("artefact {} not found: {}", path.display(), e);
            return (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "status": "not_found",
                "run_id": resolved_id,
                "detail": e.to_string(),
            }))).into_response();
        }
    };
    // Stream the file contents. Artefacts are small (<100 MB even for
    // long runs), so a single buffered read is simpler than a chunked
    // stream and the HTTP client can still receive it progressively.
    let mut buf = Vec::new();
    if let Err(e) = file.read_to_end(&mut buf).await {
        warn!("artefact {} read failed: {}", path.display(), e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "read failed").into_response();
    }
    let download_name = format!("{content_disposition_filename}").replace("{id}", &resolved_id);
    let cache_header = if run_id == "latest" {
        // The "latest" alias resolves at request time; never cache.
        "no-store"
    } else {
        // Concrete run IDs are immutable artefacts.
        "public, max-age=31536000, immutable"
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CACHE_CONTROL, cache_header)
        .header(
            header::CONTENT_DISPOSITION,
            format!("inline; filename=\"{download_name}\""),
        )
        .body(Body::from(buf))
        .expect("response builds")
}

/// `GET /api/runs/{run_id}/summary.json`
pub async fn get_summary(AxumPath(run_id): AxumPath<String>) -> Response {
    serve_artefact(
        &run_id, "json", "application/json",
        "{id}-summary.json",
    ).await
}

/// `GET /api/runs/{run_id}/scalars.csv`
pub async fn get_scalars(AxumPath(run_id): AxumPath<String>) -> Response {
    serve_artefact(
        &run_id, "csv", "text/csv; charset=utf-8",
        "{id}-scalars.csv",
    ).await
}

/// `GET /api/runs/{run_id}/data.h5`
pub async fn get_hdf5(AxumPath(run_id): AxumPath<String>) -> Response {
    serve_artefact(
        &run_id, "h5", "application/x-hdf5",
        "{id}-data.h5",
    ).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_run_id_accepts_latest() {
        assert!(validate_run_id("latest").is_ok());
    }

    #[test]
    fn validate_run_id_accepts_lowercase_hex32() {
        assert!(validate_run_id("08d96318a3b34ef6915ae7306f41d22e").is_ok());
    }

    #[test]
    fn validate_run_id_rejects_short_or_wrong_charset() {
        assert!(validate_run_id("shortish").is_err());
        assert!(validate_run_id("UPPERCASE0000000000000000000000A").is_err());
        // Path traversal attempt.
        assert!(validate_run_id("../secrets/creds.env").is_err());
        // Exactly 32 chars but with a non-hex char.
        assert!(validate_run_id("0000000000000000000000000000000z").is_err());
    }
}
