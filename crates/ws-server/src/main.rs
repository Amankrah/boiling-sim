use axum::{routing::get, Router};
use std::net::SocketAddr;

async fn health() -> &'static str {
    "boiling-sim ws-server alive"
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/health", get(health));
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("ws-server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
