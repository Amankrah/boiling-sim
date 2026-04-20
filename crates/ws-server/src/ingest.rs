//! TCP listener on 127.0.0.1:8765 that accepts length-prefixed
//! msgpack-encoded [`crate::snapshot::Snapshot`] frames from the
//! Python producer.
//!
//! Wire format: `u32 big-endian length || msgpack payload`, one frame
//! per snapshot. `tokio_util::codec::LengthDelimitedCodec` handles
//! framing; we decode only the header (version) for validation and
//! push the full payload bytes into the broadcast channel so WS
//! clients and any debug consumer share a single allocation per frame.
//!
//! Design: a fresh connection replaces any previous producer. That
//! matches the intended topology (one Python process producing, one
//! Rust process relaying) and avoids mixing snapshots from two sims.

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::StreamExt;
use tokio::net::TcpListener;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};
use tracing::{debug, info, warn};

use crate::app::AppState;
use crate::snapshot::Snapshot;

pub const DEFAULT_INGEST_ADDR: &str = "127.0.0.1:8765";

/// Spawn the ingest TCP listener. The returned future runs until
/// shutdown (binding failure is propagated back as an error).
pub async fn run(addr: &str, state: AppState) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("ingest listener bound on {addr}");

    loop {
        let (socket, peer) = listener.accept().await?;
        info!("producer connected from {peer}");
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_producer(socket, peer, state).await {
                warn!("producer {peer} disconnected: {e}");
            }
        });
    }
}

async fn handle_producer(
    socket: tokio::net::TcpStream,
    peer: SocketAddr,
    state: AppState,
) -> anyhow::Result<()> {
    // 16 MB frame cap: well above the dev-guide 2 MB downsampled budget
    // yet keeps a single malformed length header from exhausting memory.
    let codec = LengthDelimitedCodec::builder()
        .length_field_type::<u32>()
        .big_endian()
        .max_frame_length(16 * 1024 * 1024)
        .new_codec();
    let mut reader = FramedRead::new(socket, codec);

    let mut frame_count: u64 = 0;
    while let Some(frame) = reader.next().await {
        let bytes: Bytes = frame?.freeze();
        // Validate the header (version) synchronously so we surface
        // mismatches loudly rather than forwarding them to clients.
        if let Err(e) = Snapshot::from_msgpack_bytes(&bytes) {
            warn!("rejected snapshot from {peer}: {e}");
            continue;
        }
        // Push raw msgpack bytes into the broadcast channel; the WS
        // handler compresses once per client on send. `send` only errs
        // when there are zero subscribers -- that's fine, we keep the
        // producer running so clients can join mid-stream.
        let _ = state.snapshots.send(Arc::new(bytes.to_vec()));
        frame_count = frame_count.saturating_add(1);
        if frame_count % 300 == 0 {
            debug!("ingested {frame_count} frames from {peer}");
        }
    }
    info!("producer {peer} closed after {frame_count} frames");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;
    use tokio::time::{timeout, Duration};

    use crate::snapshot::{BubbleState, GridMeta, Snapshot, SCHEMA_VERSION};

    fn small_snapshot(step: u64) -> Snapshot {
        Snapshot {
            version: SCHEMA_VERSION,
            t_sim: step as f32 * 0.01,
            step,
            is_rebuilding: false,
            is_paused: false,
            grid: GridMeta { nx: 4, ny: 4, nz: 4, dx: 0.002, origin: [0.0; 3] },
            grid_ds: GridMeta { nx: 2, ny: 2, nz: 2, dx: 0.004, origin: [0.0; 3] },
            temperature: vec![95.0; 8],
            alpha: vec![1.0; 8],
            bubbles: vec![BubbleState { position: [0.0; 3], radius: 1e-4 }],
            nutrient_primary_name: "β-carotene".into(),
            nutrient_secondary_name: String::new(),
            carrot_retention: 99.0,
            carrot_leached: 0.0,
            carrot_degraded: 1.0,
            carrot_precipitated: 0.0,
            carrot_retention2: 100.0,
            carrot_leached2: 0.0,
            carrot_degraded2: 0.0,
            carrot_precipitated2: 0.0,
            carrot_surface_c: vec![],
            carrot_surface_c2: vec![],
            wall_temperature_mean: 100.0,
            wall_heat_flux: 30_000.0,
            water_temperature_mean: 99.9,
            water_temperature_max: 100.2,
            water_temperature_min: 99.4,
            run_id: "test-run".into(),
            total_time_s: 60.0,
            is_complete: false,
            last_error: String::new(),
        }
    }

    /// Pick an ephemeral port by binding with port 0 and returning the
    /// bound address. Avoids flaky overlap with tests running in
    /// parallel.
    async fn bind_ephemeral() -> (TcpListener, String) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().unwrap().to_string();
        (listener, addr)
    }

    /// Thin shim: run the ingest accept-loop on a pre-bound listener so
    /// we don't race on port acquisition.
    async fn run_ingest_on(listener: TcpListener, state: AppState) {
        loop {
            let (socket, peer) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => return,
            };
            let st = state.clone();
            tokio::spawn(async move {
                let _ = handle_producer(socket, peer, st).await;
            });
        }
    }

    #[tokio::test]
    async fn ingest_forwards_valid_frames_to_broadcast() {
        let state = AppState::new();
        let mut rx = state.snapshots.subscribe();
        let (listener, addr) = bind_ephemeral().await;
        tokio::spawn(run_ingest_on(listener, state.clone()));

        let mut stream = TcpStream::connect(&addr).await.expect("connect");
        let snap = small_snapshot(1);
        let bytes = snap.to_msgpack_bytes().expect("encode");
        let len = (bytes.len() as u32).to_be_bytes();
        stream.write_all(&len).await.unwrap();
        stream.write_all(&bytes).await.unwrap();
        stream.flush().await.unwrap();

        let received = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("broadcast within timeout")
            .expect("channel open");
        let decoded = Snapshot::from_msgpack_bytes(&received).expect("decode");
        assert_eq!(decoded.step, 1);
    }

    #[tokio::test]
    async fn ingest_rejects_version_mismatch_without_propagating() {
        let state = AppState::new();
        let mut rx = state.snapshots.subscribe();
        let (listener, addr) = bind_ephemeral().await;
        tokio::spawn(run_ingest_on(listener, state.clone()));

        let mut stream = TcpStream::connect(&addr).await.expect("connect");

        // Bad frame first: wrong version.
        let mut bad = small_snapshot(0);
        bad.version = SCHEMA_VERSION + 1;
        let bad_bytes = bad.to_msgpack_bytes().unwrap();
        stream.write_all(&(bad_bytes.len() as u32).to_be_bytes()).await.unwrap();
        stream.write_all(&bad_bytes).await.unwrap();

        // Good frame second: correct version.
        let good = small_snapshot(42);
        let good_bytes = good.to_msgpack_bytes().unwrap();
        stream.write_all(&(good_bytes.len() as u32).to_be_bytes()).await.unwrap();
        stream.write_all(&good_bytes).await.unwrap();
        stream.flush().await.unwrap();

        // First received broadcast must be the GOOD one -- bad one was dropped.
        let received = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("broadcast within timeout")
            .expect("channel open");
        let decoded = Snapshot::from_msgpack_bytes(&received).unwrap();
        assert_eq!(decoded.step, 42, "bad frame leaked to broadcast");
    }
}
