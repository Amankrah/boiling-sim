"""Scalar history buffering + HDF5/CSV/JSON artefact writing for the
live dashboard.

At 30 Hz × 600 s = 18 000 ScalarSamples × ~30 fields × 8 bytes ≈ 4 MB.
Always-on accumulation is trivial; the producer appends on every
`sample_scalars()` call (which it was calling anyway for snapshot
emission) and persists three artefacts on run completion:

    {run_id}.h5    -- full scalar time-series + parameter echo
                      (pandas/h5py/MATLAB friendly; mirrors the Phase-4
                      layout used by `Simulation.run(out_path=...)`)
    {run_id}.csv   -- one row per sample, cheap for the browser's
                      Results page to fetch and render
    {run_id}.json  -- final-state summary + acceptance gates, driving
                      the Phase-4-style report on the Results page

The cap keeps memory bounded if the user manually extends a run past
the target duration; downsampling drops every other old entry rather
than losing the head or tail.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from .config import ScenarioConfig
    from .pipeline import ScalarSample


# ---------------------------------------------------------------------------
# ScalarHistory -- bounded append-only ring
# ---------------------------------------------------------------------------


class ScalarHistory:
    """Bounded buffer of `ScalarSample`s accumulated over a single
    simulation run. Reset on rebuild; persisted on completion.

    The cap is generous -- we keep every sample for the target
    duration and a ~2× margin, then start halving. Worst case memory
    at 50 000 × 30 fields × 8 bytes ≈ 12 MB.
    """

    def __init__(self, target_duration_s: float, snapshot_hz: float = 30.0) -> None:
        # 2× safety margin + 50 k floor so ad-hoc "duration = 0" (run
        # forever) stays bounded.
        self.cap = max(50_000, int(2 * max(target_duration_s, 60.0) * snapshot_hz))
        self._samples: list[ScalarSample] = []

    @property
    def samples(self) -> list[ScalarSample]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def append(self, sample: "ScalarSample") -> None:
        self._samples.append(sample)
        # If we blew past the cap, drop every other old entry. Keeps
        # the latest samples at full resolution and only thins out
        # pre-cap history.
        if len(self._samples) > self.cap:
            half = self.cap // 2
            old = self._samples[:-half]
            recent = self._samples[-half:]
            downsampled = old[::2]
            self._samples = downsampled + recent

    def clear(self) -> None:
        self._samples.clear()


# ---------------------------------------------------------------------------
# CSV field list -- canonical column order (also the HDF5 dataset names)
# ---------------------------------------------------------------------------

SCALAR_CSV_FIELDS: tuple[str, ...] = (
    "t", "dt",
    "T_mean_water_c", "T_max_water_c", "T_min_water_c",
    "T_max_wall_c", "T_inner_wall_mean_c", "T_inner_wall_max_c",
    "u_max_mps",
    "n_active_bubbles", "mean_bubble_R_mm", "mean_departed_bubble_R_mm",
    "max_bubble_R_mm", "alpha_min",
    "retention_pct", "leached_pct", "degraded_pct", "precipitated_pct",
    "retention2_pct", "leached2_pct", "degraded2_pct", "precipitated2_pct",
)


# ---------------------------------------------------------------------------
# Artefact writer
# ---------------------------------------------------------------------------


def write_run_artefacts(
    history: ScalarHistory,
    cfg: "ScenarioConfig",
    run_id: str,
    out_dir: Path | str,
    *,
    wall_clock_s: float = 0.0,
    nutrient_primary_name: str = "",
    nutrient_secondary_name: str = "",
) -> tuple[Path, Path, Path]:
    """Persist the run as HDF5 + CSV + JSON under ``out_dir``.

    Returns ``(h5_path, csv_path, json_path)``. Caller is responsible
    for making the paths visible to the Rust relay (in docker-compose
    via a shared volume; in dev via a shared filesystem path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"{run_id}.h5"
    csv_path = out_dir / f"{run_id}.csv"
    json_path = out_dir / f"{run_id}.json"

    samples = history.samples
    _write_csv(csv_path, samples)
    _write_hdf5(h5_path, samples, cfg, run_id)
    _write_summary_json(
        json_path, samples, cfg, run_id,
        wall_clock_s=wall_clock_s,
        nutrient_primary_name=nutrient_primary_name,
        nutrient_secondary_name=nutrient_secondary_name,
    )
    return h5_path, csv_path, json_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, samples: list["ScalarSample"]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SCALAR_CSV_FIELDS)
        for s in samples:
            writer.writerow([getattr(s, k) for k in SCALAR_CSV_FIELDS])


def _write_hdf5(
    path: Path,
    samples: list["ScalarSample"],
    cfg: "ScenarioConfig",
    run_id: str,
) -> None:
    with h5py.File(path, "w") as f:
        g = f.create_group("scalars")
        for key in SCALAR_CSV_FIELDS:
            values = [getattr(s, key) for s in samples]
            g.create_dataset(key, data=np.array(values))
        meta = f.create_group("meta")
        meta.attrs["run_id"] = run_id
        meta.attrs["n_samples"] = len(samples)
        meta.attrs["schema_version"] = 3
        # Echo the ScenarioConfig as JSON so a future reader can
        # reconstruct what was run without hunting for the YAML.
        try:
            cfg_json = cfg.model_dump_json()
        except Exception:  # pragma: no cover - defensive, Pydantic always works
            cfg_json = "{}"
        meta.create_dataset(
            "scenario_config", data=np.bytes_(cfg_json.encode("utf-8")),
        )


def _write_summary_json(
    path: Path,
    samples: list["ScalarSample"],
    cfg: "ScenarioConfig",
    run_id: str,
    *,
    wall_clock_s: float,
    nutrient_primary_name: str,
    nutrient_secondary_name: str,
) -> None:
    """Final-state summary + acceptance gates + parameter echo.

    The Results page renders a Phase-4-style report directly from
    this JSON; field names are chosen to read well on the client side.
    """
    summary: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": 3,
        "n_samples": len(samples),
        "wall_clock_s": float(wall_clock_s),
        "t_sim_total_s": float(samples[-1].t) if samples else 0.0,
        "step_count": int(getattr(samples[-1], "step", 0)) if samples else 0,
        "s_per_sim_s": (
            float(wall_clock_s / samples[-1].t)
            if samples and samples[-1].t > 0 else 0.0
        ),
        "snapshot_cadence_hz": (
            float(len(samples) / samples[-1].t)
            if samples and samples[-1].t > 0 else 0.0
        ),
        "nutrient_primary_name": nutrient_primary_name,
        "nutrient_secondary_name": nutrient_secondary_name,
        "final": _final_state(samples),
        "acceptance": _acceptance_gates(samples, nutrient_primary_name),
        "mass_balance": _mass_balance_stats(samples),
        "parameters": _parameter_echo(cfg),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _final_state(samples: list["ScalarSample"]) -> dict[str, float]:
    if not samples:
        return {}
    last = samples[-1]
    # Average the last 5 % of samples for noise reduction (matches
    # scripts/run_retention.py convention).
    tail = max(1, len(samples) // 20)
    window = samples[-tail:]
    def mean(field: str) -> float:
        return float(np.mean([getattr(s, field) for s in window]))
    return {
        "t_sim_s": float(last.t),
        "T_water_mean_c": mean("T_mean_water_c"),
        "T_water_max_c": mean("T_max_water_c"),
        "T_water_min_c": mean("T_min_water_c"),
        "T_wall_inner_mean_c": mean("T_inner_wall_mean_c"),
        "T_wall_inner_max_c": mean("T_inner_wall_max_c"),
        "n_active_bubbles": int(last.n_active_bubbles),
        "retention_pct": mean("retention_pct"),
        "leached_pct": mean("leached_pct"),
        "degraded_pct": mean("degraded_pct"),
        "precipitated_pct": mean("precipitated_pct"),
        "retention2_pct": mean("retention2_pct"),
        "leached2_pct": mean("leached2_pct"),
        "degraded2_pct": mean("degraded2_pct"),
        "precipitated2_pct": mean("precipitated2_pct"),
    }


def _acceptance_gates(
    samples: list["ScalarSample"],
    nutrient_primary_name: str,
) -> list[dict[str, Any]]:
    """Auto-generated checklist matching the style of
    benchmarks/phase4_retention.md "Exit-check audit".
    """
    if not samples:
        return []

    last = samples[-1]

    # Phase-4-validated reference bands. Kept conservative (wider than
    # the dev-guide) so small deviations don't falsely fail the gate.
    BETA_CAROTENE_BAND = (80.0, 90.0)
    VITAMIN_C_BAND = (55.0, 80.0)
    is_beta_carotene = "caroten" in nutrient_primary_name.lower()
    is_vitamin_c = "vitamin" in nutrient_primary_name.lower()
    target_band: tuple[float, float] | None = (
        BETA_CAROTENE_BAND if is_beta_carotene
        else VITAMIN_C_BAND if is_vitamin_c
        else None
    )

    # Mass-balance invariant: sum across all four buckets, worst drift
    # over the whole run.
    sums_primary = [
        s.retention_pct + s.leached_pct + s.degraded_pct + s.precipitated_pct
        for s in samples
    ]
    mass_drift_primary = max(abs(s - 100.0) for s in sums_primary)

    water_t_deviation = max(abs(s.T_mean_water_c - 100.0) for s in samples[-max(1, len(samples) // 4):])

    gates: list[dict[str, Any]] = [
        {
            "name": "Mass-balance invariant (primary solute)",
            "passed": mass_drift_primary < 0.5,
            "detail": f"max |sum - 100| = {mass_drift_primary:.3f} pp (gate < 0.5 pp)",
        },
        {
            "name": "Water temperature pinned at saturation",
            "passed": water_t_deviation < 2.0,
            "detail": (
                f"|T_water - 100| over final quarter: max "
                f"{water_t_deviation:.2f} K (gate < 2 K)"
            ),
        },
        {
            "name": "No NaN in retention trace",
            "passed": all(
                not np.isnan(s.retention_pct) and not np.isnan(s.T_mean_water_c)
                for s in samples
            ),
            "detail": "retention + water T finite at every sample",
        },
    ]

    if target_band is not None:
        lo, hi = target_band
        r_final = float(np.mean([s.retention_pct for s in samples[-max(1, len(samples) // 20):]]))
        gates.insert(0, {
            "name": f"Retention in expected band [{lo:.0f}, {hi:.0f}] %",
            "passed": lo <= r_final <= hi,
            "detail": f"R(t_end) = {r_final:.2f} % (target [{lo:.0f}, {hi:.0f}])",
        })

    # Secondary solute mass balance only reported when it's actually active.
    sec_active = any(
        s.retention2_pct < 99.99 or s.leached2_pct > 0.01
        for s in samples
    )
    if sec_active:
        sums_sec = [
            s.retention2_pct + s.leached2_pct + s.degraded2_pct + s.precipitated2_pct
            for s in samples
        ]
        drift_sec = max(abs(s - 100.0) for s in sums_sec)
        gates.append({
            "name": "Mass-balance invariant (secondary solute)",
            "passed": drift_sec < 0.5,
            "detail": f"max |sum - 100| = {drift_sec:.3f} pp (gate < 0.5 pp)",
        })

    _ = last  # silence unused if the gate set above didn't reference it
    return gates


def _mass_balance_stats(samples: list["ScalarSample"]) -> dict[str, float]:
    if not samples:
        return {"max_abs_drift_pct": 0.0}
    sums = [
        s.retention_pct + s.leached_pct + s.degraded_pct + s.precipitated_pct
        for s in samples
    ]
    return {
        "max_abs_drift_pct": float(max(abs(s - 100.0) for s in sums)),
        "final_sum_pct": float(sums[-1]),
    }


def _parameter_echo(cfg: "ScenarioConfig") -> dict[str, Any]:
    """Echo the applied scenario config as a plain-JSON dict for the
    Results page to render into tables. Any pydantic model will
    serialise via model_dump; we keep the shape flat per section.
    """
    try:
        return cfg.model_dump(mode="json")
    except Exception:  # pragma: no cover
        return {}


# Suppress the unused-import hint for asdict -- kept intentionally in
# the import list as a hook for future per-field overrides.
_ = asdict
