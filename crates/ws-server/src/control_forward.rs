//! TCP listener on 127.0.0.1:8766 that streams browser-originated
//! [`ControlMessage`]s, one per newline-delimited JSON line, to the
//! Python simulation process.
//!
//! Python opens a client connection as soon as it starts the dashboard
//! loop (see `scripts/run_dashboard.py` in M3). Rust holds the write
//! side, the broadcast subscriber side, and forwards every message it
//! receives from the WS handler. If Python disconnects (sim crash /
//! restart), the forwarder closes its subscriber and re-arms for the
//! next connection -- no control messages are retained across
//! disconnects, by design (a stale heat-flux command from five
//! minutes ago shouldn't auto-apply when Python comes back).

use std::net::SocketAddr;

use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, info, warn};

use crate::app::AppState;

pub const DEFAULT_CONTROL_ADDR: &str = "127.0.0.1:8766";

pub async fn run(addr: &str, state: AppState) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("control forwarder listening on {addr}");
    loop {
        let (socket, peer) = listener.accept().await?;
        info!("python control consumer connected from {peer}");
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_consumer(socket, peer, state).await {
                warn!("control consumer {peer} dropped: {e}");
            }
        });
    }
}

async fn handle_consumer(
    mut socket: TcpStream,
    peer: SocketAddr,
    state: AppState,
) -> anyhow::Result<()> {
    let mut rx = state.controls.subscribe();
    let mut forwarded: u64 = 0;
    loop {
        match rx.recv().await {
            Ok(msg) => {
                let line = msg.to_json_line();
                if let Err(e) = socket.write_all(line.as_bytes()).await {
                    debug!("control consumer {peer} write failed: {e}");
                    break;
                }
                forwarded = forwarded.saturating_add(1);
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                warn!(
                    "control consumer {peer} lagged by {skipped} messages; \
                     continuing with next"
                );
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        }
    }
    info!("control consumer {peer} closed after forwarding {forwarded} messages");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::net::TcpStream;
    use tokio::time::{timeout, Duration};

    use crate::control::ControlMessage;

    async fn bind_ephemeral() -> (TcpListener, String) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        (listener, addr)
    }

    async fn run_forwarder_on(listener: TcpListener, state: AppState) {
        loop {
            let (socket, peer) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => return,
            };
            let st = state.clone();
            tokio::spawn(async move {
                let _ = handle_consumer(socket, peer, st).await;
            });
        }
    }

    #[tokio::test]
    async fn forwards_control_messages_as_json_lines() {
        let state = AppState::new();
        let (listener, addr) = bind_ephemeral().await;
        tokio::spawn(run_forwarder_on(listener, state.clone()));

        let stream = TcpStream::connect(&addr).await.unwrap();
        let mut reader = BufReader::new(stream).lines();

        // Give the spawned consumer a moment to subscribe before we
        // publish, so the first message isn't lost to a race between
        // `accept` and `subscribe`.
        tokio::time::sleep(Duration::from_millis(50)).await;

        state
            .controls
            .send(ControlMessage::SetHeatFlux { value: 45_000.0 })
            .expect("subscriber present");
        state
            .controls
            .send(ControlMessage::Pause)
            .expect("subscriber present");

        let line1 = timeout(Duration::from_secs(2), reader.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let line2 = timeout(Duration::from_secs(2), reader.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        assert!(line1.contains(r#""type":"set_heat_flux""#));
        assert!(line1.contains(r#""value":45000"#));
        assert_eq!(line2, r#"{"type":"pause"}"#);
    }
}
