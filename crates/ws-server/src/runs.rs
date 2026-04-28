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

/// Cwd-relative fallback when neither the env var nor a workspace
/// root can be located. Last-resort path used by binaries that have
/// been `cargo install`-ed outside a workspace; in normal dev/CI we
/// always hit one of the higher-priority branches in [`artefact_dir`].
pub const DEFAULT_ARTEFACT_DIR: &str = "./dashboard_runs";

/// Names how [`artefact_dir`] resolved its path. Surfaced in the
/// startup log so a glance at the terminal tells the user whether
/// they're hitting the env-var override, the workspace-root walk-up,
/// or the cwd fallback (the last of which usually means something is
/// misconfigured).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtefactDirSource {
    EnvVar,
    WorkspaceRoot,
    CwdFallback,
}

impl ArtefactDirSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::EnvVar => "via BOILINGSIM_ARTIFACTS_DIR",
            Self::WorkspaceRoot => "via workspace root walk-up",
            Self::CwdFallback => {
                "via cwd fallback -- set BOILINGSIM_ARTIFACTS_DIR for explicit control"
            }
        }
    }
}

/// Walk upward from `start` looking for a `Cargo.toml` whose contents
/// contain `[workspace]`. Returns the directory holding that file (the
/// workspace root) on success. None if no such file is found before
/// reaching the filesystem root, or if any IO error trips the search.
///
/// This is the same pattern `cargo` itself uses to find the workspace
/// from any nested directory. We deliberately keep walking past member
/// crate `Cargo.toml`s (which lack `[workspace]`) — those files exist
/// at e.g. `crates/ws-server/Cargo.toml` and would otherwise short-
/// circuit the walk to the wrong directory.
fn find_workspace_root_from(start: &std::path::Path) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        let candidate = cur.join("Cargo.toml");
        if candidate.is_file() {
            if let Ok(text) = std::fs::read_to_string(&candidate) {
                if text.contains("[workspace]") {
                    return Some(cur);
                }
            }
        }
        if !cur.pop() {
            return None;
        }
    }
}

/// Returns the resolved artefact directory along with the mechanism
/// that produced it. Resolution priority:
///
///   1. `BOILINGSIM_ARTIFACTS_DIR` env var (explicit override).
///   2. `<workspace_root>/dashboard_runs` discovered by walking up
///      from the current working directory looking for the workspace
///      root `Cargo.toml`. This is the new robust default that lets
///      ws-server be launched from any cwd inside the workspace
///      (project root, `web/`, `crates/ws-server/`, ...) and still
///      resolve to the same artefact directory Python writes to.
///   3. Cwd-relative `./dashboard_runs` as the last-resort fallback
///      for binaries shipped outside a Cargo workspace.
pub fn artefact_dir_with_source() -> (PathBuf, ArtefactDirSource) {
    if let Ok(p) = std::env::var("BOILINGSIM_ARTIFACTS_DIR") {
        return (PathBuf::from(p), ArtefactDirSource::EnvVar);
    }
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(root) = find_workspace_root_from(&cwd) {
            return (root.join("dashboard_runs"), ArtefactDirSource::WorkspaceRoot);
        }
    }
    (PathBuf::from(DEFAULT_ARTEFACT_DIR), ArtefactDirSource::CwdFallback)
}

/// Returns the resolved artefact directory, honouring the env var
/// then falling back to the workspace-root walk-up. Preferred call
/// site for HTTP handlers that don't care about the source label;
/// the startup logger uses [`artefact_dir_with_source`] instead so
/// it can surface "which mechanism resolved this path" in the log.
pub fn artefact_dir() -> PathBuf {
    artefact_dir_with_source().0
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

    #[test]
    fn find_workspace_root_walks_up_past_member_crates() {
        // Layout: tmp/Cargo.toml ([workspace]) + tmp/crates/foo/Cargo.toml
        // (no [workspace]) + tmp/web/. From tmp/web/ and tmp/crates/foo/
        // the walk-up should return tmp/, NOT tmp/crates/foo/.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/foo\"]\n",
        ).unwrap();
        std::fs::create_dir_all(root.join("crates/foo")).unwrap();
        std::fs::write(
            root.join("crates/foo/Cargo.toml"),
            "[package]\nname = \"foo\"\nversion = \"0.0.0\"\n",
        ).unwrap();
        std::fs::create_dir_all(root.join("web")).unwrap();

        // Canonicalise so the equality compares post-symlink paths.
        let root_canon = root.canonicalize().unwrap();
        for start_subdir in ["", "web", "crates/foo"] {
            let start = root.join(start_subdir);
            let found = find_workspace_root_from(&start)
                .expect("walk should find workspace root")
                .canonicalize()
                .unwrap();
            assert_eq!(
                found, root_canon,
                "walk from {start:?} should land at workspace root",
            );
        }
    }

    #[test]
    fn find_workspace_root_returns_none_outside_workspace() {
        // tempdir contains no Cargo.toml at all.
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(find_workspace_root_from(tmp.path()).is_none());
    }

    #[test]
    fn artefact_dir_env_var_takes_priority() {
        // Preserve any inherited env var so we don't poison subsequent
        // tests in the same process.
        let prior = std::env::var("BOILINGSIM_ARTIFACTS_DIR").ok();
        std::env::set_var("BOILINGSIM_ARTIFACTS_DIR", "/tmp/explicit-override");
        let (path, source) = artefact_dir_with_source();
        assert_eq!(path, PathBuf::from("/tmp/explicit-override"));
        assert_eq!(source, ArtefactDirSource::EnvVar);
        // Restore.
        match prior {
            Some(v) => std::env::set_var("BOILINGSIM_ARTIFACTS_DIR", v),
            None => std::env::remove_var("BOILINGSIM_ARTIFACTS_DIR"),
        }
    }
}
