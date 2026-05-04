"""Phase 6 M3 -- integration tests for ``SnapshotProducer`` and
``ControlConsumer`` against a fake Rust relay.

The fake relay is a tiny threaded TCP server (stdlib socketserver)
that:
    - on ingest: reads length-prefixed msgpack frames and stashes them
      in a list
    - on control: accepts a connection and writes JSON lines provided
      by the test

This lets us verify Python's outbound wire format and inbound parser
without spinning up the real Rust binary.
"""

from __future__ import annotations

import json
import socket
import socketserver
import struct
import threading
import time
from typing import Any

import msgpack
import pytest

from boilingsim.config import load_scenario
from boilingsim.dashboard import (
    SCHEMA_VERSION,
    ControlConsumer,
    SnapshotProducer,
)
from boilingsim.pipeline import Simulation


# -------- Fake servers ---------------------------------------------------


class _IngestHandler(socketserver.BaseRequestHandler):
    """Reads u32-BE length-prefixed msgpack frames into `server.frames`."""

    def handle(self) -> None:  # type: ignore[override]
        sock: socket.socket = self.request
        while True:
            header = self._recv_exact(sock, 4)
            if not header:
                return
            (length,) = struct.unpack(">I", header)
            payload = self._recv_exact(sock, length)
            if payload is None:
                return
            self.server.frames.append(payload)  # type: ignore[attr-defined]

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None if not buf else buf + b"\x00" * (n - len(buf))
            buf += chunk
        return buf


class _IngestServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    # Handler threads must be daemons or pytest teardown blocks until
    # every client connection closes on its own schedule.
    daemon_threads = True

    def __init__(self, addr: tuple[str, int]) -> None:
        super().__init__(addr, _IngestHandler)
        self.frames: list[bytes] = []


class _ControlHandler(socketserver.BaseRequestHandler):
    """Writes each queued string to its client. The server exposes a
    ``send_line`` helper the test calls from the main thread."""

    def handle(self) -> None:  # type: ignore[override]
        sock: socket.socket = self.request
        self.server.active_clients.append(sock)  # type: ignore[attr-defined]
        try:
            # Keep the connection open until the test tears the server down.
            while not self.server.stop_event.is_set():  # type: ignore[attr-defined]
                time.sleep(0.05)
        finally:
            try:
                sock.close()
            except OSError:
                pass


class _ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr: tuple[str, int]) -> None:
        super().__init__(addr, _ControlHandler)
        self.active_clients: list[socket.socket] = []
        self.stop_event = threading.Event()

    def send_line(self, line: str) -> None:
        data = (line.rstrip("\n") + "\n").encode("utf-8")
        for c in list(self.active_clients):
            try:
                c.sendall(data)
            except OSError:
                pass


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -------- Fixtures -------------------------------------------------------


@pytest.fixture
def ingest_server():
    port = _ephemeral_port()
    server = _IngestServer(("127.0.0.1", port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, port
    server.shutdown()
    server.server_close()


@pytest.fixture
def control_server():
    port = _ephemeral_port()
    server = _ControlServer(("127.0.0.1", port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, port
    server.stop_event.set()
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def small_sim() -> Simulation:
    cfg = load_scenario("configs/scenarios/single_carrot.yaml")
    cfg.nutrient.enabled = True
    cfg.boiling.enabled = True
    cfg.grid.dx_m = 0.004
    cfg.total_time_s = 0.5
    sim = Simulation(cfg)
    for _ in range(5):
        sim.step()
        if sim.t >= 0.2:
            break
    return sim


# -------- Tests: SnapshotProducer ---------------------------------------


def test_producer_sends_valid_length_prefixed_frames(ingest_server, small_sim):
    server, port = ingest_server
    producer = SnapshotProducer(addr=("127.0.0.1", port))

    n_sent = 0
    for step in range(5):
        if producer.send_snapshot(small_sim, step=step):
            n_sent += 1

    # Give the handler thread a moment to finish buffering.
    deadline = time.monotonic() + 2.0
    while len(server.frames) < n_sent and time.monotonic() < deadline:
        time.sleep(0.02)
    producer.close()

    assert n_sent > 0, "producer sent no frames -- fake server not reachable"
    assert len(server.frames) == n_sent, (
        f"expected {n_sent} frames at server, got {len(server.frames)}"
    )
    # Every frame must msgpack-decode and carry the schema version.
    for frame in server.frames:
        decoded = msgpack.unpackb(frame, raw=False)
        assert decoded["version"] == SCHEMA_VERSION


def test_producer_survives_relay_not_running():
    """With no server on the target port, ``send_snapshot`` must drop
    the frame and return False -- not raise, not block."""
    dead_port = _ephemeral_port()  # bind+release so no one's listening
    producer = SnapshotProducer(
        addr=("127.0.0.1", dead_port),
        reconnect_backoff_s=0.05,
    )
    # Using raw bytes so we don't need a Simulation fixture for this test.
    ok1 = producer.send_bytes(b"irrelevant, not delivered")
    assert ok1 is False
    assert producer.frames_dropped == 1
    # Second call inside the backoff window also drops silently.
    ok2 = producer.send_bytes(b"also not delivered")
    assert ok2 is False
    assert producer.frames_dropped == 2


def test_producer_reconnects_after_server_comes_up_late(small_sim):
    """Scenario: Python starts before the Rust relay. Early sends drop,
    later sends succeed once the relay becomes listenable.

    This directly exercises the backoff-then-retry path without the
    cross-platform hazards of a socket-alive-check after peer shutdown
    (on Windows, sendall can briefly appear to succeed on a dead
    socket, which makes an "old-server-dies" scenario unreliable to
    assert on).
    """
    # Pick a port nobody is listening on yet.
    port = _ephemeral_port()

    producer = SnapshotProducer(
        addr=("127.0.0.1", port),
        reconnect_backoff_s=0.05,
    )
    # First send: no server, must drop.
    assert producer.send_snapshot(small_sim, step=0) is False
    assert producer.frames_dropped == 1

    # Bring the server up.
    server = _IngestServer(("127.0.0.1", port))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Retry until we land a send. Backoff is 50 ms so ~10 attempts.
    recovered = False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if producer.send_snapshot(small_sim, step=1):
            recovered = True
            break
        time.sleep(0.1)
    assert recovered, (
        f"producer never reconnected; frames_dropped={producer.frames_dropped}"
    )

    # Let the server finish reading the framed payload.
    deadline = time.monotonic() + 2.0
    while not server.frames and time.monotonic() < deadline:
        time.sleep(0.02)
    producer.close()
    server.shutdown()
    server.server_close()
    assert len(server.frames) >= 1


# -------- Tests: ControlConsumer -----------------------------------------


def test_consumer_parses_newline_json_lines(control_server):
    server, port = control_server
    consumer = ControlConsumer(
        addr=("127.0.0.1", port),
        reconnect_backoff_s=0.05,
    )
    consumer.start()
    # Give the consumer thread a moment to connect.
    time.sleep(0.3)

    server.send_line(json.dumps({"type": "set_heat_flux", "value": 40000}))
    server.send_line(json.dumps({"type": "pause"}))

    # Poll until both messages arrive.
    deadline = time.monotonic() + 2.0
    seen: list[dict[str, Any]] = []
    while len(seen) < 2 and time.monotonic() < deadline:
        seen.extend(consumer.drain())
        time.sleep(0.05)
    consumer.stop()

    assert len(seen) == 2, f"expected 2 messages, got {seen}"
    assert seen[0]["type"] == "set_heat_flux"
    assert seen[0]["value"] == 40000
    assert seen[1]["type"] == "pause"


def test_consumer_ignores_malformed_lines(control_server):
    server, port = control_server
    consumer = ControlConsumer(
        addr=("127.0.0.1", port),
        reconnect_backoff_s=0.05,
    )
    consumer.start()
    time.sleep(0.3)

    # Bad JSON, missing "type", good JSON.
    server.send_line("not-json")
    server.send_line('{"value": 1}')
    server.send_line(json.dumps({"type": "reset"}))

    deadline = time.monotonic() + 2.0
    seen: list[dict[str, Any]] = []
    while not seen and time.monotonic() < deadline:
        seen.extend(consumer.drain())
        time.sleep(0.05)
    consumer.stop()

    assert len(seen) == 1
    assert seen[0]["type"] == "reset"


def test_consumer_survives_relay_not_running():
    """No server on target port -> consumer thread stays alive, drain()
    returns empty list each call."""
    dead_port = _ephemeral_port()
    consumer = ControlConsumer(
        addr=("127.0.0.1", dead_port),
        reconnect_backoff_s=0.05,
    )
    consumer.start()
    time.sleep(0.2)
    assert consumer.drain() == []
    consumer.stop()
