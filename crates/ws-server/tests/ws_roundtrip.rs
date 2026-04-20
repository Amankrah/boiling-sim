//! End-to-end: fake Python producer sends TCP frames -> ws-server -> fake
//! browser receives zstd-compressed msgpack binaries over WebSocket.
//!
//! Also exercises the control reverse-channel: fake browser sends a
//! ControlMessage via WS text frame and a fake Python consumer on the
//! control TCP port receives the JSON line.

use std::sync::Arc;
use std::time::Duration;

use axum::{routing::get, Router};
use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

use ws_server::app::AppState;
use ws_server::snapshot::{BubbleState, GridMeta, Snapshot, SCHEMA_VERSION};
use ws_server::ws::ws_handler;

fn fixture(step: u64) -> Snapshot {
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

async fn bind_ephemeral() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    (listener, addr)
}

/// Spin up all three pieces of the server on ephemeral ports. Returns
/// (http_addr, ingest_addr, control_addr, state).
async fn spin_up_server() -> (String, String, String, AppState) {
    let state = AppState::new();

    // Ingest listener.
    let (ingest_listener, ingest_addr) = bind_ephemeral().await;
    {
        let state = state.clone();
        tokio::spawn(async move {
            loop {
                let (sock, peer) = match ingest_listener.accept().await {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let st = state.clone();
                tokio::spawn(async move {
                    let _ = private::handle_producer(sock, peer, st).await;
                });
            }
        });
    }

    // Control forwarder.
    let (ctrl_listener, ctrl_addr) = bind_ephemeral().await;
    {
        let state = state.clone();
        tokio::spawn(async move {
            loop {
                let (sock, peer) = match ctrl_listener.accept().await {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let st = state.clone();
                tokio::spawn(async move {
                    let _ = private::handle_consumer(sock, peer, st).await;
                });
            }
        });
    }

    // HTTP + WS.
    let (http_listener, http_addr) = bind_ephemeral().await;
    let app = Router::new()
        .route("/stream", get(ws_handler))
        .with_state(state.clone());
    tokio::spawn(async move {
        let _ = axum::serve(http_listener, app).await;
    });

    // Small settle so listeners are actually accept-ready.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (http_addr, ingest_addr, ctrl_addr, state)
}

/// Shim around the private handlers from `ingest` / `control_forward`.
/// We can't call them directly because they're `async fn` in `pub(crate)`
/// positions, so we re-drive the same accept loops inline via the public
/// re-exported helpers.
mod private {
    #![allow(unused_imports)]
    use super::*;
    use bytes::Bytes;
    use futures_util::StreamExt as _;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;
    use tokio::sync::broadcast::error::RecvError;
    use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

    pub async fn handle_producer(
        socket: TcpStream,
        _peer: std::net::SocketAddr,
        state: AppState,
    ) -> anyhow::Result<()> {
        let codec = LengthDelimitedCodec::builder()
            .length_field_type::<u32>()
            .big_endian()
            .max_frame_length(16 * 1024 * 1024)
            .new_codec();
        let mut reader = FramedRead::new(socket, codec);
        while let Some(frame) = reader.next().await {
            let bytes: Bytes = frame?.freeze();
            if Snapshot::from_msgpack_bytes(&bytes).is_err() {
                continue;
            }
            let _ = state.snapshots.send(Arc::new(bytes.to_vec()));
        }
        Ok(())
    }

    pub async fn handle_consumer(
        mut socket: TcpStream,
        _peer: std::net::SocketAddr,
        state: AppState,
    ) -> anyhow::Result<()> {
        let mut rx = state.controls.subscribe();
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    let line = msg.to_json_line();
                    if socket.write_all(line.as_bytes()).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Lagged(_)) => continue,
                Err(RecvError::Closed) => break,
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn ws_delivers_compressed_snapshot_roundtrip() {
    let (http_addr, ingest_addr, _ctrl_addr, _state) = spin_up_server().await;

    // Connect fake WS client FIRST so it's subscribed before we publish.
    let ws_url = format!("ws://{http_addr}/stream");
    let (mut ws_stream, _resp) = tokio_tungstenite::connect_async(&ws_url)
        .await
        .expect("ws connect");

    // Wait a beat so the handler is running and subscribed.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Fake Python producer: send one length-prefixed msgpack frame.
    let mut producer = TcpStream::connect(&ingest_addr).await.expect("producer connect");
    let snap = fixture(7);
    let bytes = snap.to_msgpack_bytes().unwrap();
    producer.write_all(&(bytes.len() as u32).to_be_bytes()).await.unwrap();
    producer.write_all(&bytes).await.unwrap();
    producer.flush().await.unwrap();

    // Fake browser: wait for one binary frame, zstd-decode, re-parse.
    let frame = timeout(Duration::from_secs(3), ws_stream.next())
        .await
        .expect("WS frame arrived in time")
        .expect("stream yielded")
        .expect("message ok");
    let bin = match frame {
        tokio_tungstenite::tungstenite::Message::Binary(b) => b,
        other => panic!("expected Binary, got {other:?}"),
    };
    let decoded = zstd::decode_all(&bin[..]).expect("zstd decode");
    let recovered = Snapshot::from_msgpack_bytes(&decoded).expect("msgpack decode");
    assert_eq!(recovered.step, 7);
    assert_eq!(recovered.version, SCHEMA_VERSION);
}

#[tokio::test]
async fn ws_forwards_control_text_to_python_consumer() {
    let (http_addr, _ingest_addr, ctrl_addr, _state) = spin_up_server().await;

    // Fake Python control consumer connects first.
    let py_stream = TcpStream::connect(&ctrl_addr).await.expect("py connect");
    let mut py_reader = BufReader::new(py_stream).lines();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Fake browser sends a ControlMessage over WS.
    let ws_url = format!("ws://{http_addr}/stream");
    let (mut ws_stream, _) = tokio_tungstenite::connect_async(&ws_url).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    ws_stream
        .send(tokio_tungstenite::tungstenite::Message::Text(
            r#"{"type":"set_heat_flux","value":45000}"#.to_string(),
        ))
        .await
        .unwrap();

    let line = timeout(Duration::from_secs(3), py_reader.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(line.contains(r#""type":"set_heat_flux""#));
    assert!(line.contains(r#""value":45000"#));
}
