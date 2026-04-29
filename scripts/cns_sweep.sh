#!/usr/bin/env bash
# 2026 CNS Halifax conference run sweep.
#
# Drives 7 retention runs that together tell three slide stories:
#   (1) Power sweep at one material (steel) -- q ∈ {30, 60, 90} kW/m²
#   (2) Cold start vs hot drop-in (Sultana protocol) at q = 60 kW/m²
#   (3) Material sweep at one power -- steel/Al/Cu at q = 80 kW/m²
#
# Each run writes:
#   benchmarks/phase4_retention_<tag>.png   -- 3-panel headline plot
#   benchmarks/phase4_retention_<tag>.h5    -- full HDF5 time series + snapshots
#
# Skips runs whose PNG already exists, so re-running picks up where it left
# off if anything is interrupted.
#
# Wall time on RTX 4090: ~25-40 min per 900 s sim, so ~3-4 h end-to-end.
# Run unattended; the script prints per-run banners.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

run_retention() {
  local config="$1" tag="$2" water_c="$3" wall_c="$4" label="$5"
  local out="benchmarks/phase4_retention_${tag}.png"

  if [[ -f "$out" ]]; then
    echo ">>> SKIP  $tag  (already have $out)"
    return 0
  fi

  echo
  echo "============================================================"
  echo ">>> $label"
  echo ">>> config=$config  tag=$tag  warm-start water=${water_c}C wall=${wall_c}C carrot=20C"
  echo "============================================================"

  local t0
  t0=$(date +%s)

  python scripts/run_retention.py \
    --config "$config" \
    --duration 900 \
    --dx-mm 2.0 \
    --pressure-iters 100 \
    --warm-start-water-c "$water_c" \
    --warm-start-wall-c  "$wall_c" \
    --warm-start-carrot-c 20 \
    --tag "$tag" \
    --solute-label "beta-carotene" \
    --target-band 80 90 \
    --exp-ref-pct 84 \
    --snapshot-every-s 30

  local elapsed=$(( $(date +%s) - t0 ))
  echo ">>> $tag finished in $((elapsed / 60))m $((elapsed % 60))s"
}

# ---- (1) Power sweep, cold start, steel ----------------------------------
run_retention configs/scenarios/cns_q30.yaml \
  cns_q30_cold 20 20 \
  "Power sweep: q=30 kW/m² (simmer), cold start"

run_retention configs/scenarios/cns_demo.yaml \
  cns_q60_cold 20 20 \
  "Power sweep: q=60 kW/m² (medium-high), cold start  -- anchor"

run_retention configs/scenarios/cns_q90.yaml \
  cns_q90_cold 20 20 \
  "Power sweep: q=90 kW/m² (max-boil), cold start"

# ---- (2) Cold vs hot drop-in at q=60 kW/m² -------------------------------
# Sultana / Vieira experimental protocol: drop a cold carrot into already-
# boiling water. Use the same cns_demo.yaml but warm-start water=95C
# wall=100C (the historic Phase-4 retention setup).
run_retention configs/scenarios/cns_demo.yaml \
  cns_q60_hotdrop 95 100 \
  "Hot drop-in: q=60 kW/m², water=95C wall=100C (Sultana protocol)"

# ---- (3) Material sweep at q=80 kW/m² (real-world configs) ---------------
run_retention configs/scenarios/default.yaml \
  realworld_steel_q80 20 20 \
  "Material sweep: steel 304 at q=80 kW/m² (real-world)"

run_retention configs/scenarios/aluminum.yaml \
  realworld_aluminum_q80 20 20 \
  "Material sweep: aluminum at q=80 kW/m² (real-world)"

run_retention configs/scenarios/copper.yaml \
  realworld_copper_q80 20 20 \
  "Material sweep: copper at q=80 kW/m² (real-world)"

echo
echo "============================================================"
echo "All CNS sweep runs complete."
echo "Plots:    benchmarks/phase4_retention_{cns_q30_cold,cns_q60_cold,cns_q90_cold,cns_q60_hotdrop,realworld_steel_q80,realworld_aluminum_q80,realworld_copper_q80}.png"
echo "Raw HDF5: same names, .h5 suffix"
echo "============================================================"
