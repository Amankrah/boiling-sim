"""Phase 6 dashboard glue: snapshot serializer (M1) + TCP producer +
control consumer (M3).

The module is split into three independent concerns:

    1. Snapshot builders (``build_snapshot``, ``serialize_snapshot``,
       ``serialize_rebuild_marker``) -- pure functions, unit-testable
       without a running Rust relay.

    2. ``SnapshotProducer`` -- maintains a TCP connection to the Rust
       relay's ingest port (default 127.0.0.1:8765) and writes
       length-prefixed msgpack frames. Auto-reconnects on failure so a
       Rust restart doesn't kill the Python sim.

    3. ``ControlConsumer`` -- background thread reading newline-JSON
       ``ControlMessage`` lines from the Rust relay's control port
       (default 127.0.0.1:8766) and pushing parsed dicts onto a
       thread-safe queue that the sim loop drains between steps.

Wire format: **MessagePack** (``msgpack`` package, ``use_bin_type=True``)
with field names matching [crates/ws-server/src/snapshot.rs](../../../crates/ws-server/src/snapshot.rs)
exactly. The Rust ``Snapshot::from_msgpack_bytes`` rejects any payload
whose ``version`` field doesn't match ``SCHEMA_VERSION``; Python bumps
must be coordinated with the Rust bump in the same commit.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np

if TYPE_CHECKING:
    from .pipeline import Simulation


# ---------------------------------------------------------------------------
# Wire-format version. MUST stay in lockstep with crates/ws-server/src/snapshot.rs
# (see the CHANGELOG comment at the top of that file for v1 -> v2 changes).
# ---------------------------------------------------------------------------
SCHEMA_VERSION: int = 6


# Display names for the Phase-4-validated solutes. Keyed to the
# NutrientConfig fields that identify them; used both on the wire
# (`nutrient_primary_name`) and by the UI for labels. The matching
# numeric presets live in scripts/run_dashboard.py so this module
# stays pure data-contract.
NUTRIENT_DISPLAY_NAMES: dict[str, str] = {
    "beta_carotene": "β-carotene",
    "vitamin_c": "vitamin C",
}


def _classify_nutrient(nutrient_cfg: Any) -> str:
    """Best-effort identification of which canonical nutrient a
    NutrientConfig block represents, by parameter signature.

    Uses the two distinguishing knobs -- K_partition and
    C_water_sat_mg_per_kg -- because both β-carotene and vitamin C are
    identified exclusively by those two in our validated scenarios
    (1e-5 / 6e-3 for β-carotene, 1.0 / 1e6 for vitamin C). Returns the
    display name; falls back to "nutrient" if the signature doesn't
    match any preset (e.g. user edited the YAML)."""
    if not getattr(nutrient_cfg, "enabled", False):
        return ""
    K = float(getattr(nutrient_cfg, "K_partition", 0.0))
    sat = float(getattr(nutrient_cfg, "C_water_sat_mg_per_kg", 0.0))
    if K < 1.0e-3 and sat < 1.0:
        return NUTRIENT_DISPLAY_NAMES["beta_carotene"]
    if K >= 0.5 and sat >= 1.0e3:
        return NUTRIENT_DISPLAY_NAMES["vitamin_c"]
    return "nutrient"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _downsample_halves(field: np.ndarray) -> np.ndarray:
    """Stride-2 subsample of a (nx, ny, nz) cell-centred field.

    Cheap, non-overlapping -- good enough for volume rendering at 30 Hz.
    Keeps the C-contiguous (nx, ny, nz) layout that
    ``Snapshot::temperature`` documents (k fastest, i slowest).
    """
    return np.ascontiguousarray(field[::2, ::2, ::2], dtype=np.float32)


_AXIS_TO_INT = {"x": 0, "y": 1, "z": 2}


def _axis_to_int(axis: str) -> int:
    """Map cfg.carrot.axis literal to the wire-format integer (0=x, 1=y, 2=z)."""
    return _AXIS_TO_INT.get(axis, 2)


def _carrot_centres(cfg: Any) -> list[list[float]]:
    """Compute carrot instance centres from cfg using the same auto-placement
    routine geometry.py uses to voxelize. Returned as a length-``count`` list
    of ``[x, y, z]`` lists (msgpack-friendly)."""
    from .config import auto_place_carrots
    inner_radius = cfg.pot.diameter_m / 2 - cfg.pot.wall_thickness_m
    water_height = cfg.water.fill_fraction * (
        cfg.pot.height_m - cfg.pot.base_thickness_m
    )
    water_top_z = cfg.pot.base_thickness_m + water_height
    centres = auto_place_carrots(
        count=cfg.carrot.count,
        axis=cfg.carrot.axis,
        anchor=cfg.carrot.position,
        diameter_m=cfg.carrot.diameter_m,
        length_m=cfg.carrot.length_m,
        inner_radius=inner_radius,
        base_thickness=cfg.pot.base_thickness_m,
        water_top_z=water_top_z,
    )
    return [[float(c[0]), float(c[1]), float(c[2])] for c in centres]


def _grid_meta(nx: int, ny: int, nz: int, dx: float, origin: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "dx": float(dx),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
    }


def _active_bubbles(sim: "Simulation") -> list[dict[str, Any]]:
    """Extract active bubbles as a list of {position, radius} dicts.

    Returns an empty list when ``cfg.boiling.enabled`` is False or the
    pool is empty. Inactive pool slots are filtered out so the wire
    payload stays proportional to on-screen bubble count, not the
    ``max_bubbles`` allocation.
    """
    if sim.grid.bubbles is None:
        return []
    from .boiling import read_active_bubbles  # local import: avoids cycle
    view = read_active_bubbles(sim.grid.bubbles)
    if view.n_active == 0:
        return []
    return [
        {"position": [float(p[0]), float(p[1]), float(p[2])], "radius": float(r)}
        for p, r in zip(view.positions, view.radii, strict=True)
    ]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_snapshot(
    sim: "Simulation",
    step: int,
    *,
    is_rebuilding: bool = False,
    is_paused: bool = False,
    run_id: str = "",
    total_time_s: float = 0.0,
    is_complete: bool = False,
    last_error: str = "",
    sample: Any = None,
) -> dict[str, Any]:
    """Build the Python dict that ``msgpack.packb`` serializes into a
    ``Snapshot``-compatible wire buffer.

    Split out from :func:`serialize_snapshot` so callers can inspect
    field values (tests, debug logging) without re-decoding the msgpack
    bytes.

    v3 additions: ``run_id``, ``total_time_s``, ``is_complete``,
    ``last_error``. All have safe defaults so callers that don't yet
    track run metadata still produce valid v3 frames.

    v4 additions: ``pot_diameter_m``, ``pot_height_m``,
    ``pot_wall_thickness_m``, ``pot_base_thickness_m`` -- pot geometry
    echoed from ``cfg.pot`` so the 3D renderer can scale to match the
    currently-simulated pot instead of hardcoding 20 cm x 12 cm.

    Perf: callers can pass a pre-computed ``sample`` (a ``ScalarSample``)
    to skip the 5-10 GPU readbacks ``sample_scalars`` performs --
    important when the dashboard loop already sampled this tick for the
    scalar history. Pass ``None`` (default) to recompute internally,
    which keeps tests and ad-hoc callers working.
    """
    grid = sim.grid
    cfg = sim.cfg
    nx, ny, nz = grid.shape
    origin = tuple(float(v) for v in grid.origin)  # type: ignore[assignment]

    # Temperature stored internally in Kelvin; browser wants Celsius.
    T_k = grid.T.numpy()
    T_c = (T_k - 273.15).astype(np.float32)
    alpha = (
        grid.water_alpha.numpy().astype(np.float32)
        if grid.water_alpha is not None
        else np.ones_like(T_c, dtype=np.float32)
    )

    T_ds = _downsample_halves(T_c).ravel(order="C")
    alpha_ds = _downsample_halves(alpha).ravel(order="C")
    nx_ds, ny_ds, nz_ds = (nx // 2, ny // 2, nz // 2)

    # Mass-partition retention percentages via sample_scalars. Reuse a
    # pre-computed sample when the caller has one (~5-10 ms of GPU
    # readbacks saved per snapshot tick).
    if sample is None:
        sample = sim.sample_scalars(dt_last=0.0)

    wall_heat_flux = float(getattr(cfg.heating, "base_heat_flux_w_per_m2", 0.0))

    return {
        "version": SCHEMA_VERSION,
        "t_sim": float(sim.t),
        "step": int(step),
        "is_rebuilding": bool(is_rebuilding),
        "is_paused": bool(is_paused),
        "grid": _grid_meta(nx, ny, nz, grid.dx, origin),
        "grid_ds": _grid_meta(nx_ds, ny_ds, nz_ds, grid.dx * 2.0, origin),
        # v5: raw little-endian f32 bytes. msgpack ``use_bin_type=True``
        # encodes ``bytes`` as a ``bin`` chunk (1 + len bytes overhead),
        # which is ~30x cheaper to produce than the per-cell Python-float
        # allocation .tolist() did on a 692k-cell field. Browser-side
        # decode reinterprets the Uint8Array buffer as a Float32Array.
        "temperature": T_ds.tobytes(),
        "alpha": alpha_ds.tobytes(),
        "bubbles": _active_bubbles(sim),
        # --- nutrient identity (v2) ---
        "nutrient_primary_name": _classify_nutrient(cfg.nutrient),
        "nutrient_secondary_name": _classify_nutrient(
            getattr(cfg, "nutrient2", None)
        ),
        # --- primary solute mass partition (v2) ---
        "carrot_retention": float(sample.retention_pct),
        "carrot_leached": float(sample.leached_pct),
        "carrot_degraded": float(sample.degraded_pct),
        "carrot_precipitated": float(sample.precipitated_pct),
        # --- secondary solute mass partition (v2) ---
        "carrot_retention2": float(sample.retention2_pct),
        "carrot_leached2": float(sample.leached2_pct),
        "carrot_degraded2": float(sample.degraded2_pct),
        "carrot_precipitated2": float(sample.precipitated2_pct),
        # Surface-C extraction will ship alongside a real tet carrot
        # mesh in a future pass; ship empty vecs to preserve schema.
        "carrot_surface_c": [],
        "carrot_surface_c2": [],
        "wall_temperature_mean": float(sample.T_inner_wall_mean_c),
        "wall_heat_flux": wall_heat_flux,
        # --- v3: water thermal detail + run metadata ---
        "water_temperature_mean": float(sample.T_mean_water_c),
        "water_temperature_max": float(sample.T_max_water_c),
        "water_temperature_min": float(sample.T_min_water_c),
        "run_id": str(run_id),
        "total_time_s": float(total_time_s),
        "is_complete": bool(is_complete),
        "last_error": str(last_error),
        # --- v4: pot geometry echo ---
        "pot_diameter_m": float(cfg.pot.diameter_m),
        "pot_height_m": float(cfg.pot.height_m),
        "pot_wall_thickness_m": float(cfg.pot.wall_thickness_m),
        "pot_base_thickness_m": float(cfg.pot.base_thickness_m),
        # --- v6: carrot pose / quantity ---
        # The browser draws N cylinders -- one per centre -- with the
        # given diameter / length / axis. ``carrot_total_mass_g`` is the
        # derived UX quantity (count * pi * (d/2)^2 * length * rho_carrot).
        # Per-instance retention is a future extension; for now all
        # instances share the aggregate ``carrot_retention*`` scalars.
        "carrot_count": int(cfg.carrot.count),
        "carrot_axis": _axis_to_int(cfg.carrot.axis),
        "carrot_diameter_m": float(cfg.carrot.diameter_m),
        "carrot_length_m": float(cfg.carrot.length_m),
        "carrot_centres": _carrot_centres(cfg),
        "carrot_total_mass_g": float(cfg.carrot.total_mass_g()),
    }


def serialize_snapshot(
    sim: "Simulation",
    step: int,
    *,
    is_rebuilding: bool = False,
    is_paused: bool = False,
    run_id: str = "",
    total_time_s: float = 0.0,
    is_complete: bool = False,
    last_error: str = "",
    sample: Any = None,
) -> bytes:
    """Msgpack-encode a snapshot for transmission to the Rust ws-server.

    The v3 kwargs (run_id / total_time_s / is_complete / last_error)
    are forwarded to :func:`build_snapshot`; all have safe empty
    defaults so callers that haven't wired up run tracking still
    produce valid frames.
    """
    return msgpack.packb(
        build_snapshot(
            sim, step,
            is_rebuilding=is_rebuilding,
            is_paused=is_paused,
            run_id=run_id,
            total_time_s=total_time_s,
            is_complete=is_complete,
            last_error=last_error,
            sample=sample,
        ),
        use_bin_type=True,
        # Floats are default 64-bit in msgpack; we want 32-bit to match
        # Rust's f32 on the wire. msgpack-python honours dtype on ndarray
        # inputs but we've already converted to lists. Single-precision
        # retention / wall temps end up as f64 -- rmp_serde will accept
        # f64 for f32 fields (wide->narrow is handled via serde) at the
        # cost of ~15 % payload growth on the retention fields, which is
        # negligible against the volume arrays.
    )


DEFAULT_INGEST_ADDR: tuple[str, int] = ("127.0.0.1", 8765)
DEFAULT_CONTROL_ADDR: tuple[str, int] = ("127.0.0.1", 8766)


class SnapshotProducer:
    """Maintains a TCP connection to the Rust ingest port and writes
    length-prefixed msgpack frames.

    Write semantics: best-effort. A failed write closes the socket and
    schedules a reconnect on the next call. The sim loop is never
    blocked by retry logic -- if the relay is down, we drop frames on
    the floor and the browser sees a gap. That's the correct behaviour
    for a live-view tool: missing frames are less bad than a stalled
    solver.

    Thread-safety: the producer is driven from the sim-loop thread
    only. Do not share across threads.
    """

    def __init__(
        self,
        addr: tuple[str, int] = DEFAULT_INGEST_ADDR,
        *,
        reconnect_backoff_s: float = 1.0,
    ) -> None:
        self.addr = addr
        self.reconnect_backoff_s = reconnect_backoff_s
        self._sock: socket.socket | None = None
        self._next_connect_at: float = 0.0
        self._frames_sent: int = 0
        self._frames_dropped: int = 0

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    def _ensure_connected(self) -> bool:
        if self._sock is not None:
            return True
        if time.monotonic() < self._next_connect_at:
            return False
        try:
            s = socket.create_connection(self.addr, timeout=1.0)
            s.settimeout(None)
            self._sock = s
            return True
        except OSError:
            # Relay not listening yet (or died). Back off but don't raise.
            self._next_connect_at = time.monotonic() + self.reconnect_backoff_s
            return False

    def send_bytes(self, payload: bytes) -> bool:
        """Send a single length-prefixed frame. Returns True on success,
        False if the relay was unreachable and the frame was dropped."""
        if not self._ensure_connected():
            self._frames_dropped += 1
            return False
        assert self._sock is not None
        header = struct.pack(">I", len(payload))
        try:
            self._sock.sendall(header + payload)
        except OSError:
            # Peer closed or network hiccup. Drop the current socket;
            # the next send attempts a fresh connect.
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
            self._next_connect_at = time.monotonic() + self.reconnect_backoff_s
            self._frames_dropped += 1
            return False
        self._frames_sent += 1
        return True

    def send_snapshot(
        self,
        sim: "Simulation",
        step: int,
        *,
        is_rebuilding: bool = False,
        is_paused: bool = False,
        run_id: str = "",
        total_time_s: float = 0.0,
        is_complete: bool = False,
        last_error: str = "",
        sample: Any = None,
    ) -> bool:
        buf = serialize_snapshot(
            sim, step,
            is_rebuilding=is_rebuilding,
            is_paused=is_paused,
            run_id=run_id,
            total_time_s=total_time_s,
            is_complete=is_complete,
            last_error=last_error,
            sample=sample,
        )
        return self.send_bytes(buf)

    def send_rebuild_marker(
        self,
        t_sim: float = 0.0,
        run_id: str = "",
        total_time_s: float = 0.0,
    ) -> bool:
        return self.send_bytes(
            serialize_rebuild_marker(
                t_sim=t_sim, run_id=run_id, total_time_s=total_time_s,
            ),
        )

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class ControlConsumer:
    """Background thread that reads newline-JSON ``ControlMessage`` lines
    from the Rust relay's control port and pushes parsed dicts onto a
    thread-safe queue.

    Usage::

        cc = ControlConsumer()
        cc.start()
        while running:
            sim.step()
            for msg in cc.drain():
                apply(msg)

    Reconnect policy matches ``SnapshotProducer``: best-effort, back off
    silently when the relay is unreachable. An orderly ``stop()`` joins
    the reader thread.
    """

    def __init__(
        self,
        addr: tuple[str, int] = DEFAULT_CONTROL_ADDR,
        *,
        reconnect_backoff_s: float = 1.0,
    ) -> None:
        self.addr = addr
        self.reconnect_backoff_s = reconnect_backoff_s
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="dashboard-control-consumer", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def drain(self) -> list[dict[str, Any]]:
        """Pop every message currently queued. Non-blocking."""
        msgs: list[dict[str, Any]] = []
        while True:
            try:
                msgs.append(self._queue.get_nowait())
            except queue.Empty:
                return msgs

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                sock = socket.create_connection(self.addr, timeout=1.0)
            except OSError:
                # Relay not ready; back off and retry.
                self._stop_flag.wait(self.reconnect_backoff_s)
                continue
            sock.settimeout(0.25)  # periodic poll so stop_flag is honoured
            buf = b""
            try:
                while not self._stop_flag.is_set():
                    try:
                        chunk = sock.recv(4096)
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, _, buf = buf.partition(b"\n")
                        if not line.strip():
                            continue
                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if isinstance(msg, dict) and "type" in msg:
                            self._queue.put(msg)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass


def serialize_rebuild_marker(
    t_sim: float = 0.0,
    run_id: str = "",
    total_time_s: float = 0.0,
) -> bytes:
    """Produce a minimal "rebuilding..." frame for the browser to show a
    spinner while the Python producer tears down and reconstructs the
    ``Simulation`` (material / carrot-size / reset control messages).

    All field values are zeros / empties; ``is_rebuilding = True`` is
    the signal. Schema stays the same so the browser uses the same
    decoder. Optional ``run_id`` + ``total_time_s`` let the browser
    keep the progress bar reading the post-rebuild target.
    """
    payload = {
        "version": SCHEMA_VERSION,
        "t_sim": float(t_sim),
        "step": 0,
        "is_rebuilding": True,
        "is_paused": False,
        "grid": _grid_meta(0, 0, 0, 0.0, (0.0, 0.0, 0.0)),
        "grid_ds": _grid_meta(0, 0, 0, 0.0, (0.0, 0.0, 0.0)),
        # v5: temperature/alpha are msgpack `bin` (empty here for the
        # rebuild marker so the Rust deserializer accepts the frame).
        "temperature": b"",
        "alpha": b"",
        "bubbles": [],
        "nutrient_primary_name": "",
        "nutrient_secondary_name": "",
        "carrot_retention": 100.0,
        "carrot_leached": 0.0,
        "carrot_degraded": 0.0,
        "carrot_precipitated": 0.0,
        "carrot_retention2": 100.0,
        "carrot_leached2": 0.0,
        "carrot_degraded2": 0.0,
        "carrot_precipitated2": 0.0,
        "carrot_surface_c": [],
        "carrot_surface_c2": [],
        "wall_temperature_mean": 0.0,
        "wall_heat_flux": 0.0,
        # v3
        "water_temperature_mean": 0.0,
        "water_temperature_max": 0.0,
        "water_temperature_min": 0.0,
        "run_id": str(run_id),
        "total_time_s": float(total_time_s),
        "is_complete": False,
        "last_error": "",
        # v4 -- pot geometry echo; zeros are fine during a rebuild since
        # the client will get the true values on the next real snapshot.
        "pot_diameter_m": 0.0,
        "pot_height_m": 0.0,
        "pot_wall_thickness_m": 0.0,
        "pot_base_thickness_m": 0.0,
        # v6 -- carrot pose / quantity. Same rationale: zeros during
        # rebuild; the next snapshot carries the real values.
        "carrot_count": 0,
        "carrot_axis": 2,
        "carrot_diameter_m": 0.0,
        "carrot_length_m": 0.0,
        "carrot_centres": [],
        "carrot_total_mass_g": 0.0,
    }
    return msgpack.packb(payload, use_bin_type=True)
