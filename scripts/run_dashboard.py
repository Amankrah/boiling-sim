"""Phase 6 dashboard driver: warm-started live boiling simulation that
publishes msgpack snapshots to the Rust ws-server at a configurable
cadence and applies browser-originated control messages between steps.

Mirrors the CLI surface of [scripts/run_retention.py](run_retention.py)
for scenario / grid / warm-start flags. Unlike ``run_retention.py``, this
script owns its own step loop (rather than calling ``Simulation.run``)
because control messages that swap material / carrot size trigger a
full Simulation rebuild which the HDF5-emitting ``run`` method isn't
designed for.

Flow per iteration::

    drain ControlConsumer queue -> apply to cfg / schedule rebuild
    if rebuild pending:
        send rebuild marker
        tear down + rebuild Simulation
    sim.step()
    if step_count % snapshot_interval == 0:
        send snapshot

Example::

    # Terminal 1: Rust relay
    cargo run -p ws-server --release
    # Terminal 2: Python producer
    python scripts/run_dashboard.py --config configs/scenarios/default.yaml --duration 60
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import uuid
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from boilingsim.config import load_scenario  # noqa: E402
from boilingsim.dashboard import (  # noqa: E402
    DEFAULT_CONTROL_ADDR,
    DEFAULT_INGEST_ADDR,
    ControlConsumer,
    SnapshotProducer,
    _classify_nutrient,  # noqa: PLC2701 - internal helper kept module-local
)
from boilingsim.geometry import MAT_CARROT, MAT_FLUID, MAT_POT_WALL  # noqa: E402
from boilingsim.pipeline import Simulation  # noqa: E402
from boilingsim.run_writer import ScalarHistory, write_run_artefacts  # noqa: E402

KNOWN_MATERIALS = {"steel_304", "copper", "aluminum"}

# ---------------------------------------------------------------------------
# Nutrient presets -- matches configs/scenarios/{default,vitamin_c_25mm,
# dual_solute_25mm}.yaml so the dashboard lets users flip solute without
# restarting the Python process. Each key is a valid `set_nutrient`
# value; the value is the (nutrient, nutrient2) patch applied on top of
# the base scenario config.
# ---------------------------------------------------------------------------

_BETA_CAROTENE: dict = {
    "enabled": True,
    "E_a_kJ_per_mol": 70.0,
    "k0_per_s": 2.63e6,
    "D_eff_m2_per_s": 2.0e-10,
    "K_partition": 1.0e-5,
    "C0_mg_per_kg": 83.0,
    "C_water_sat_mg_per_kg": 6.0e-3,
}

_VITAMIN_C: dict = {
    "enabled": True,
    "E_a_kJ_per_mol": 74.0,
    "k0_per_s": 1.1e7,
    "D_eff_m2_per_s": 5.0e-10,
    "K_partition": 1.0,
    "C0_mg_per_kg": 59.0,
    "C_water_sat_mg_per_kg": 1.0e6,
}

_DISABLED_NUTRIENT: dict = {"enabled": False}

NUTRIENT_PRESETS: dict[str, tuple[dict, dict]] = {
    # (nutrient_patch, nutrient2_patch)
    "beta_carotene": (_BETA_CAROTENE, _DISABLED_NUTRIENT),
    "vitamin_c": (_VITAMIN_C, _DISABLED_NUTRIENT),
    "both": (_BETA_CAROTENE, _VITAMIN_C),
}


def apply_nutrient_preset(cfg, key: str) -> bool:
    """Patch `cfg.nutrient` (+ `cfg.nutrient2`) with the named preset
    via Pydantic model_copy. Returns True on success.
    """
    if key not in NUTRIENT_PRESETS:
        return False
    patch1, patch2 = NUTRIENT_PRESETS[key]
    cfg.nutrient = cfg.nutrient.model_copy(update=patch1)
    cfg.nutrient2 = cfg.nutrient2.model_copy(update=patch2)
    return True


def build_simulation(cfg, *, device: str) -> Simulation:
    """Construct a Simulation honouring ``cfg.initial_conditions``.

    ``mode == "cold"`` returns the bare Simulation; ``geometry.py`` already
    seeds the T field from ``cfg.water.initial_temp_c`` and
    ``cfg.heating.ambient_temp_c``. ``mode == "preheat"`` additionally
    overwrites water / pot-wall / carrot cells with the preheat setpoints
    — the historical warm-start path used by benchmark scripts to skip
    the warming transient.
    """
    sim = Simulation(cfg, device=device)
    ic = cfg.initial_conditions
    if ic.mode == "preheat":
        T = sim.grid.T.numpy()
        mat = sim.grid.mat.numpy()
        T[mat == MAT_FLUID] = ic.preheat_water_c + 273.15
        T[mat == MAT_POT_WALL] = ic.preheat_wall_c + 273.15
        T[mat == MAT_CARROT] = ic.preheat_carrot_c + 273.15
        sim.grid.T.assign(T)
    return sim


def apply_control_live(sim: Simulation, msg: dict[str, Any]) -> bool:
    """Apply a control message that does NOT require a rebuild. Returns
    True if the caller should trigger a rebuild instead.

    Live-editable: ``set_heat_flux`` mutates the scalar on cfg.heating.
    Rebuild-triggering: ``set_material``, ``set_carrot_size``,
    ``set_nutrient``, ``reset``, ``set_config`` (v3).
    Handled directly in the main loop (NOT here):
    ``pause``/``resume``/``start_run``/``export_snapshot``/
    ``request_full_snapshot``.
    """
    kind = msg.get("type")
    if kind == "set_heat_flux":
        value = float(msg.get("value", 0.0))
        sim.cfg.heating.base_heat_flux_w_per_m2 = value
        return False
    if kind in (
        "set_material", "set_carrot_size", "set_nutrient", "reset", "set_config",
    ):
        return True
    return False


def apply_control_rebuild(cfg, msg: dict[str, Any]) -> tuple[Any, str]:
    """Apply a rebuild-triggering control message to the config.

    Returns ``(new_cfg, error_str)``. On success, ``error_str`` is
    empty and ``new_cfg`` is the updated config to use for the next
    Simulation rebuild. On failure (v3 ``set_config`` with invalid
    JSON), ``error_str`` carries the Pydantic validation error and
    ``new_cfg`` is the **unchanged** cfg -- the caller should surface
    the error via ``Snapshot.last_error`` and skip the rebuild.
    """
    # Local import so test harnesses that don't exercise the rebuild
    # path don't pay for the Pydantic import graph.
    from boilingsim.config import ScenarioConfig  # noqa: PLC0415

    kind = msg.get("type")
    if kind == "set_material":
        value = str(msg.get("value", ""))
        if value in KNOWN_MATERIALS:
            cfg.pot.material = value
        return cfg, ""
    if kind == "set_carrot_size":
        diameter_mm = float(msg.get("diameter_mm", cfg.carrot.diameter_m * 1000.0))
        length_mm = float(msg.get("length_mm", cfg.carrot.length_m * 1000.0))
        cfg.carrot.diameter_m = diameter_mm / 1000.0
        cfg.carrot.length_m = length_mm / 1000.0
        return cfg, ""
    if kind == "set_nutrient":
        value = str(msg.get("value", ""))
        applied = apply_nutrient_preset(cfg, value)
        if not applied:
            return cfg, f"unknown nutrient preset '{value}'"
        return cfg, ""
    if kind == "set_config":
        # Full ScenarioConfig JSON blob from the Configuration page.
        # Pydantic's model_validate does all the range-checking; if it
        # raises, we return the stringified error for the UI.
        cfg_json = msg.get("config")
        if not isinstance(cfg_json, dict):
            return cfg, "set_config: `config` field missing or not an object"
        try:
            validated = ScenarioConfig.model_validate(cfg_json)
        except Exception as e:  # pydantic.ValidationError + anything else
            return cfg, f"set_config validation failed: {e}"
        return validated, ""
    # "reset" uses the current cfg unchanged.
    return cfg, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Simulated seconds; 0 means run forever (or until Ctrl-C).")
    ap.add_argument("--dx-mm", type=float, default=2.0)
    ap.add_argument("--pressure-iters", type=int, default=100)
    ap.add_argument("--max-bubbles", type=int, default=100_000)
    ap.add_argument("--snapshot-hz", type=float, default=30.0,
                    help="Target snapshot cadence. 30 is the dashboard default.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ingest-host", default=DEFAULT_INGEST_ADDR[0])
    ap.add_argument("--ingest-port", type=int, default=DEFAULT_INGEST_ADDR[1])
    ap.add_argument("--control-host", default=DEFAULT_CONTROL_ADDR[0])
    ap.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_ADDR[1])
    ap.add_argument(
        "--artefacts-dir", type=pathlib.Path, default=None,
        help="Directory where completed runs write {run_id}.{h5,csv,json}. "
             "Default: env BOILINGSIM_ARTIFACTS_DIR if set, else "
             "./dashboard_runs.",
    )
    args = ap.parse_args()

    artefacts_dir = args.artefacts_dir or pathlib.Path(
        __import__("os").environ.get(
            "BOILINGSIM_ARTIFACTS_DIR", str(ROOT / "dashboard_runs"),
        )
    )
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_scenario(args.config)
    cfg.grid.dx_m = args.dx_mm / 1000.0
    cfg.solver.pressure_max_iter = args.pressure_iters
    cfg.boiling.enabled = True
    cfg.boiling.max_bubbles = args.max_bubbles
    cfg.nutrient.enabled = True
    # Dual-solute is an opt-in via the YAML (nutrient2.enabled = true).
    # We leave it untouched so e.g. dual_solute_25mm.yaml lights up both
    # solutes automatically, and default.yaml keeps single-solute.

    producer = SnapshotProducer(addr=(args.ingest_host, args.ingest_port))
    consumer = ControlConsumer(addr=(args.control_host, args.control_port))
    consumer.start()

    sim = build_simulation(cfg, device=args.device)

    snapshot_interval_s = 1.0 / max(args.snapshot_hz, 0.1)
    last_snapshot_at: float = -1.0
    step_count: int = 0
    # Start paused so the GPU isn't burning cycles on a sim nobody's
    # watching yet. The browser flips us live when the user clicks
    # Resume on the Live page or Apply & Start Run on the Config page
    # (which sends `start_run`, see the drain handler below).
    paused: bool = True
    last_progress_at = time.perf_counter()

    # Run lifecycle state (Phase 6.6 v3):
    #   run_id       -- uuid minted per Simulation rebuild, shows up on
    #                   the wire + in the artefact filenames.
    #   total_time_s -- target duration; 0 means "run forever".
    #   history      -- bounded ScalarSample ring that gets persisted
    #                   as HDF5 / CSV / JSON on completion.
    #   run_start_wall -- wall-clock reference to report s/sim-s.
    #   is_complete  -- true once the artefact writer has emitted the
    #                   three files; sim is paused until the user
    #                   triggers a new run.
    run_id: str = uuid.uuid4().hex
    total_time_s: float = max(args.duration, 0.0)
    history = ScalarHistory(
        target_duration_s=total_time_s if total_time_s > 0 else 600.0,
        snapshot_hz=args.snapshot_hz,
    )
    run_start_wall = time.perf_counter()
    is_complete = False
    last_error = ""

    print("=== Phase 6 live dashboard producer ===")
    print(f"  config       : {args.config}")
    print(f"  dx           : {args.dx_mm:.2f} mm")
    print(f"  snapshot Hz  : {args.snapshot_hz:.1f}")
    print(f"  artefacts dir: {artefacts_dir}")
    print(f"  total_time_s : {total_time_s} (0 = run forever)")
    print(f"  ingest  -> tcp://{args.ingest_host}:{args.ingest_port}")
    print(f"  control <- tcp://{args.control_host}:{args.control_port}")
    print("  paused at t=0 -- click Resume on the Live page or "
          "'Apply & Start Run' on the Config page to begin stepping")
    print("  Ctrl-C to stop")

    def reset_run(fresh_sim: Simulation) -> tuple[str, ScalarHistory, float, bool]:
        """Reset run-scoped state after a Simulation rebuild."""
        return (
            uuid.uuid4().hex,
            ScalarHistory(
                target_duration_s=total_time_s if total_time_s > 0 else 600.0,
                snapshot_hz=args.snapshot_hz,
            ),
            time.perf_counter(),
            False,
        )

    def finalize_run() -> None:
        """Write artefacts for the current history, mark complete, emit
        the completion snapshot. Shared by the auto-complete (duration
        reached) and `finalize` (user-stopped mid-run) code paths so
        both produce identical output. No-op if history is empty."""
        nonlocal is_complete, paused, last_snapshot_at, last_error
        if len(history) == 0:
            print("  [finalize] skipping: history empty")
            return
        wall_clock = time.perf_counter() - run_start_wall
        try:
            h5_path, csv_path, json_path = write_run_artefacts(
                history, cfg, run_id, artefacts_dir,
                wall_clock_s=wall_clock,
                nutrient_primary_name=_classify_nutrient(cfg.nutrient),
                nutrient_secondary_name=_classify_nutrient(
                    getattr(cfg, "nutrient2", None),
                ),
            )
        except Exception as e:
            print(f"  [finalize] failed to write artefacts: {e}")
            last_error = f"finalize failed: {e}"
            return
        print(
            f"  [complete] run_id={run_id} wall={wall_clock:.1f}s "
            f"artefacts: {h5_path.name}, {csv_path.name}, {json_path.name}"
        )
        is_complete = True
        paused = True  # stop stepping until user triggers a new run
        # Emit the completion snapshot immediately so the browser gets
        # the is_complete=True signal without waiting for the cadence.
        producer.send_snapshot(
            sim, step=step_count,
            is_rebuilding=False,
            is_paused=paused,
            run_id=run_id,
            total_time_s=total_time_s,
            is_complete=True,
            last_error=last_error,
        )
        last_snapshot_at = time.perf_counter()

    try:
        while True:
            # 1. Drain control messages -> classify each.
            rebuild_pending = False
            latest_rebuild_msg: dict[str, Any] | None = None
            duration_change: float | None = None  # from start_run
            do_export = False
            do_finalize_now = False
            for msg in consumer.drain():
                kind = msg.get("type")
                if kind == "pause":
                    paused = True
                    continue
                if kind == "resume":
                    paused = False
                    continue
                if kind == "start_run":
                    # Begin a new timed run (usually after set_config).
                    # Deferred: applied AFTER any pending rebuild so the
                    # new history's cap matches the new duration.
                    # Also unpauses so the very first run after launch
                    # (where the loop boots in the paused-at-t=0 idle
                    # state) actually starts stepping.
                    duration_change = float(msg.get("duration_s", 600.0))
                    paused = False
                    continue
                if kind == "export_snapshot":
                    # Write artefacts mid-run without resetting state.
                    do_export = True
                    continue
                if kind == "finalize":
                    # User clicked "Finish & save": stop the run NOW,
                    # write artefacts from the partial history, flip
                    # is_complete so the Results page becomes available.
                    do_finalize_now = True
                    continue
                if apply_control_live(sim, msg):
                    rebuild_pending = True
                    latest_rebuild_msg = msg

            # 2. start_run affects the upcoming rebuild's history cap,
            #    so apply its duration change before reset_run() runs.
            #    Also clears `is_complete` so stepping can resume
            #    without a rebuild (if the user just wants a fresh
            #    countdown on the current cfg).
            if duration_change is not None:
                total_time_s = duration_change
                is_complete = False
                last_error = ""

            # 3. Rebuild if set_material/size/nutrient/config/reset fired.
            if rebuild_pending:
                producer.send_rebuild_marker(
                    t_sim=sim.t, run_id=run_id, total_time_s=total_time_s,
                )
                if latest_rebuild_msg is not None:
                    new_cfg, err = apply_control_rebuild(cfg, latest_rebuild_msg)
                    if err:
                        # Validation or preset-lookup failed. Surface
                        # the error to the browser and SKIP the rebuild
                        # so the current sim keeps running.
                        print(f"  [control] {err}")
                        last_error = err
                        continue
                    cfg = new_cfg
                    last_error = ""
                sim = build_simulation(cfg, device=args.device)
                step_count = 0
                last_snapshot_at = -1.0
                run_id, history, run_start_wall, is_complete = reset_run(sim)
                # If start_run arrived alongside set_config, the
                # duration update already landed in step 2; reset_run
                # re-created history with the current total_time_s so
                # everything lines up.
                continue

            # 4. Mid-run export_snapshot -- write artefacts without
            #    resetting state. Useful for "save what we have" mid-run.
            if do_export and len(history) > 0:
                wall = time.perf_counter() - run_start_wall
                try:
                    h5_path, csv_path, json_path = write_run_artefacts(
                        history, cfg, run_id, artefacts_dir,
                        wall_clock_s=wall,
                        nutrient_primary_name=_classify_nutrient(cfg.nutrient),
                        nutrient_secondary_name=_classify_nutrient(
                            getattr(cfg, "nutrient2", None),
                        ),
                    )
                    print(
                        f"  [export] run_id={run_id} "
                        f"artefacts: {h5_path.name}, {csv_path.name}, {json_path.name}"
                    )
                except Exception as e:
                    print(f"  [export] failed: {e}")
                    last_error = f"export failed: {e}"

            # 3a. User-triggered finalize: stop NOW, write artefacts.
            if do_finalize_now and not is_complete:
                finalize_run()

            # 3b. Auto-complete on duration reached.
            if (
                total_time_s > 0.0
                and sim.t >= total_time_s
                and not is_complete
            ):
                finalize_run()

            # 4. Step (skip if paused or complete, but keep the snapshot
            #    cadence so the browser stays responsive).
            if not paused and not is_complete:
                sim.step()
                step_count += 1
                # Buffer scalars at the snapshot cadence (not step
                # cadence) to match HDF5 density expectations.
                now = time.perf_counter()
                if now - last_snapshot_at >= snapshot_interval_s:
                    sample = sim.sample_scalars(dt_last=0.0)
                    history.append(sample)

            # 5. Emit snapshot on the chosen cadence.
            now = time.perf_counter()
            if now - last_snapshot_at >= snapshot_interval_s:
                producer.send_snapshot(
                    sim, step=step_count,
                    is_rebuilding=False,
                    is_paused=paused,
                    run_id=run_id,
                    total_time_s=total_time_s,
                    is_complete=is_complete,
                    last_error=last_error,
                )
                last_snapshot_at = now

            # 6. Progress log every ~5 seconds.
            if now - last_progress_at > 5.0:
                print(
                    f"  t_sim={sim.t:7.2f}s  step={step_count:6d}  "
                    f"sent={producer.frames_sent:6d}  dropped={producer.frames_dropped:4d}"
                    f"{'  [PAUSED]' if paused else ''}"
                    f"{'  [COMPLETE]' if is_complete else ''}"
                )
                last_progress_at = now

            # Small sleep when complete-and-idle so we're not hot-spinning.
            if is_complete:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nCtrl-C -- shutting down cleanly")
    finally:
        consumer.stop()
        producer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
