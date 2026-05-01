//! Wire format for Python -> Rust -> browser streaming.
//!
//! # Version policy
//!
//! `SCHEMA_VERSION` is the single source of truth for the snapshot layout.
//! A given ws-server binary accepts snapshots with `version == SCHEMA_VERSION`
//! ONLY. Older or newer versions are rejected at deserialization with
//! [`SnapshotError::VersionMismatch`]. Bumping the version requires a
//! coordinated commit touching this file, [`python/boilingsim/dashboard.py`],
//! and the TypeScript mirror under [`web/src/types/snapshot.ts`]. Document
//! every bump in CHANGELOG.md. Do not add compatibility shims -- the
//! Python solver, Rust relay, and browser are shipped together.
//!
//! # Serialization
//!
//! MessagePack (`rmp-serde`) on the wire. The Rust struct has a flat field
//! layout with no enums in the hot path, so msgpack encodes it as a map of
//! primitive-or-array values that `@msgpack/msgpack` on the JS side decodes
//! directly into typed objects. JSON is reserved for the JSON-typed
//! [`crate::control::ControlMessage`] stream, where readability outweighs
//! bandwidth.

use serde::{Deserialize, Serialize};

/// Current wire-format version. Bump requires a cross-stack commit.
///
/// # Changelog
///
/// **v1** (Phase 6 initial): retention only, anonymous `carrot_retention*`.
///
/// **v2** (superseded): adds full mass partition (`leached_pct`,
/// `degraded_pct`, `precipitated_pct`) for both solutes and names each
/// solute explicitly (`nutrient_primary_name`, `nutrient_secondary_name`).
/// The browser now renders all four buckets as a stacked area and labels
/// retention with the actual compound name. Required after user feedback
/// that the dashboard was mis-labelling β-carotene as "carrot retention"
/// and hiding the leaching / degradation channels that Phase 4 validated.
///
/// **v3** (superseded — Phase 6.6 data-forward upgrade): surfaces water
/// temperature (mean / max / min) that Phase 4's `ScalarSample` has
/// computed since day one but was never routed onto the wire. Adds
/// `run_id` (UUID per rebuild), `total_time_s` (the user-selected
/// target duration), `is_complete` (true when sim hit `t_sim >=
/// total_time_s` and artefacts were written), and `last_error` (surfaces
/// Pydantic validation failures from the new `set_config` control
/// message back to the Configuration page). The Live view gains a
/// water-T row and a progress bar; the Results page fetches HDF5/CSV/JSON
/// artefacts via the new `/api/runs/*` endpoints.
///
/// **v4** (superseded): echoes the live pot geometry so the
/// 3D renderer can scale the procedural pot to whatever dimensions
/// the running simulation actually uses. Previously the 3D pot was
/// hardcoded at 20 cm × 12 cm regardless of the Config page's
/// pot-section settings; now `pot_diameter_m`, `pot_height_m`,
/// `pot_wall_thickness_m`, and `pot_base_thickness_m` flow from
/// Python's `cfg.pot` onto the wire and drive `<Pot>`'s props.
///
/// **v5** (superseded): replaces the `Vec<f32>` arrays for
/// `temperature` and `alpha` with msgpack `bin` (raw little-endian f32
/// bytes). At dx = 2 mm and the realistic-pot default config the
/// downsampled fields are 692k cells each, and `numpy.tolist()` was
/// allocating 1.4 M Python floats per snapshot at 15 Hz — measured at
/// 9.79 ms per field, ~30 % of the snapshot budget. `tobytes()` runs
/// in ~0.5 ms (19× faster). Browser-side decode reinterprets the
/// `Uint8Array` payload as a `Float32Array`. Endianness is fixed at
/// little-endian — the assumption holds for x86 and ARM hosts.
///
/// **v6** (this — multi-carrot pose): adds carrot pose and quantity
/// fields so the dashboard can render N cylinders laying flat in the
/// pot rather than the single hardcoded vertical procedural cylinder.
/// The aggregate retention scalars (`carrot_retention*`) are unchanged
/// — per-instance retention is a future feature requiring labelled
/// cells. New fields: `carrot_count`, `carrot_axis` (0=x, 1=y, 2=z),
/// `carrot_diameter_m`, `carrot_length_m`, `carrot_centres` (length =
/// count, world-space anchor per instance), `carrot_total_mass_g`
/// (derived: count·π·(d/2)²·L·ρ_carrot, displayed live in the Config
/// page).
pub const SCHEMA_VERSION: u32 = 6;

/// Errors surfaced by [`Snapshot::from_msgpack_bytes`].
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("msgpack decode failed: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error(
        "snapshot version mismatch: got {got}, expected {expected}. See CHANGELOG.md \
         for the upgrade path; old clients must be rebuilt alongside the server."
    )]
    VersionMismatch { got: u32, expected: u32 },
}

/// Full + downsampled grid dimensions.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GridMeta {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub dx: f32,
    /// World-space coordinate of the (0, 0, 0) cell centre.
    pub origin: [f32; 3],
}

/// Lagrangian bubble state for instanced rendering.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BubbleState {
    pub position: [f32; 3],
    pub radius: f32,
}

/// One frame of simulation state on the wire.
///
/// Fields map 1:1 to the developer-guide §6.2 schema, plus the dual-solute
/// extension (`*_2` fields) shipped in Phase 4.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Snapshot {
    /// Must equal [`SCHEMA_VERSION`]; rejected otherwise.
    pub version: u32,
    /// Simulated time in seconds since the start of the run.
    pub t_sim: f32,
    /// Solver step count since run start. `u64::MAX` is reserved as a
    /// sentinel -- don't produce it from live runs.
    pub step: u64,
    /// True when the Python producer is between simulations (material or
    /// carrot-size change, or reset). The browser paints a spinner until
    /// the next `is_rebuilding = false` frame arrives.
    pub is_rebuilding: bool,
    /// True when the user has hit the Pause button. Producer still streams
    /// frames at the normal cadence so the UI stays responsive.
    pub is_paused: bool,

    /// Full-resolution grid metadata (used by the carrot mesh + pot geometry).
    pub grid: GridMeta,
    /// Downsampled grid metadata (`nx/2 x ny/2 x nz/2`). The volume renderer
    /// reads this.
    pub grid_ds: GridMeta,

    /// Downsampled temperature field in Celsius. C-contiguous on
    /// `(nx, ny, nz)` = `(grid_ds.nx, grid_ds.ny, grid_ds.nz)` -- i.e. the
    /// k (z) axis is the fastest, i (x) axis slowest. Linear index
    /// `idx = i*ny*nz + j*nz + k`.
    ///
    /// v5: raw little-endian f32 bytes (msgpack `bin` chunk). Length in
    /// bytes equals `4 * grid_ds.nx * grid_ds.ny * grid_ds.nz`. The
    /// Rust relay never reads the values; we only decode it to validate
    /// the version field, then forward the raw frame to clients.
    #[serde(with = "serde_bytes")]
    pub temperature: Vec<u8>,
    /// Downsampled water void-fraction in [0, 1], same layout as `temperature`.
    /// v5: raw little-endian f32 bytes. See `temperature`.
    #[serde(with = "serde_bytes")]
    pub alpha: Vec<u8>,

    /// Active bubbles only (inactive pool slots filtered out by the producer).
    pub bubbles: Vec<BubbleState>,

    /// Human-readable name of the primary nutrient (e.g. "β-carotene",
    /// "vitamin C"). Empty string when the nutrient block is disabled.
    pub nutrient_primary_name: String,
    /// Human-readable name of the secondary nutrient. Empty when
    /// `nutrient2` is disabled.
    pub nutrient_secondary_name: String,

    /// **Primary solute mass partition.** Four buckets always sum to
    /// ~100 %: the Phase-4 validation invariant. The UI renders them
    /// as a stacked-area plot matching `benchmarks/phase4_retention.md`.
    pub carrot_retention: f32,
    pub carrot_leached: f32,
    pub carrot_degraded: f32,
    pub carrot_precipitated: f32,

    /// **Secondary solute mass partition.** Same semantics; defaults to
    /// `(100, 0, 0, 0)` when `nutrient2` is disabled.
    pub carrot_retention2: f32,
    pub carrot_leached2: f32,
    pub carrot_degraded2: f32,
    pub carrot_precipitated2: f32,

    /// Per-carrot-voxel surface concentration for the primary solute
    /// (Phase 4: mg/kg). Empty vec until surface extraction lands.
    pub carrot_surface_c: Vec<f32>,
    /// Per-carrot-voxel surface concentration for the secondary solute.
    /// Empty vec until surface extraction lands.
    pub carrot_surface_c2: Vec<f32>,

    pub wall_temperature_mean: f32,
    pub wall_heat_flux: f32,

    // -------- v3: thermal detail + run metadata --------
    /// Bulk water mean temperature in Celsius (Phase-4 `T_mean_water_c`).
    /// The single most useful thermal value the user wants to see --
    /// is the pot actually at a boil?
    pub water_temperature_mean: f32,
    /// Hottest water cell (C). Spikes next to the pot wall during
    /// transient heat-up; plateaus at T_sat during a stable boil.
    pub water_temperature_max: f32,
    /// Coldest water cell (C). Near the cold carrot in the first
    /// ~60 s; collapses to T_sat once convection mixes.
    pub water_temperature_min: f32,

    /// UUID assigned per Simulation rebuild. Appears in the run
    /// artefact filenames (`{run_id}.h5 / .csv / .json`) and in the
    /// `/api/runs/{run_id}/*` HTTP endpoints.
    pub run_id: String,
    /// Target simulated-time duration for this run. When `t_sim >=
    /// total_time_s` the Python producer writes artefacts, emits a
    /// completion snapshot with `is_complete = true`, and pauses
    /// stepping until a new `start_run` control message arrives.
    /// Zero means "run indefinitely" -- duration-less exploration mode.
    pub total_time_s: f32,
    /// Set on the first snapshot after artefact writing completes.
    /// The Live page reads this to surface a "Run complete -- View
    /// results" banner.
    pub is_complete: bool,
    /// Latest Pydantic validation error from a `set_config` control
    /// message, or empty string if none. Cleared on the next
    /// successful rebuild. The Configuration page renders this
    /// inline when non-empty.
    pub last_error: String,

    // -------- v4: pot geometry echo --------
    /// Pot outer diameter in metres (Python `cfg.pot.diameter_m`).
    /// Drives `<Pot diameterM>` in the 3D scene so the rendered pot
    /// matches whatever size the simulation is actually using.
    pub pot_diameter_m: f32,
    /// Pot outer height in metres (lip to base).
    pub pot_height_m: f32,
    /// Wall thickness in metres. Sets the gap between the pot's
    /// outer shell and its inner cooking cavity.
    pub pot_wall_thickness_m: f32,
    /// Base thickness in metres. Raises the inner floor off the
    /// cooktop by this much.
    pub pot_base_thickness_m: f32,

    // -------- v6: carrot pose / quantity --------
    /// Number of carrot instances in the pot (matches Python
    /// `cfg.carrot.count`). The dashboard renders one cylinder per
    /// instance; ``carrot_centres`` has this many entries.
    pub carrot_count: u32,
    /// Cylinder axis (0=x, 1=y, 2=z). Drives the `<CarrotMesh>`
    /// rotation and the auto-placement layout: x/y mean horizontal
    /// (carrots lay flat — realistic stew), z is the legacy vertical.
    pub carrot_axis: u8,
    /// Per-carrot diameter (metres). All instances share one shape.
    pub carrot_diameter_m: f32,
    /// Per-carrot length (metres). All instances share one shape.
    pub carrot_length_m: f32,
    /// World-space anchor per instance. For axis = 2 (z), the anchor
    /// is the *base* of the cylinder; for axis 0/1 (x or y), it's the
    /// *centre*. Length equals ``carrot_count``.
    pub carrot_centres: Vec<[f32; 3]>,
    /// Total carrot mass in grams, derived from count + dimensions
    /// and a fixed density (~1040 kg/m³). Displayed live in the Config
    /// page so the user knows "I'm cooking 200 g".
    pub carrot_total_mass_g: f32,
}

impl Snapshot {
    /// Decode a msgpack-encoded snapshot and validate its version.
    pub fn from_msgpack_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let snap: Snapshot = rmp_serde::from_slice(bytes)?;
        if snap.version != SCHEMA_VERSION {
            return Err(SnapshotError::VersionMismatch {
                got: snap.version,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(snap)
    }

    /// Encode a snapshot to msgpack bytes. Used by tests and by the
    /// ws-server's own ingest-roundtrip diagnostics.
    pub fn to_msgpack_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec_named(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_snapshot() -> Snapshot {
        Snapshot {
            version: SCHEMA_VERSION,
            t_sim: 1.25,
            step: 300,
            is_rebuilding: false,
            is_paused: false,
            grid: GridMeta {
                nx: 100,
                ny: 100,
                nz: 60,
                dx: 0.002,
                origin: [-0.1, -0.1, 0.0],
            },
            grid_ds: GridMeta {
                nx: 50,
                ny: 50,
                nz: 30,
                dx: 0.004,
                origin: [-0.1, -0.1, 0.0],
            },
            // v5 raw-bytes fields: 4 bytes/cell little-endian f32.
            // Filled with arbitrary repeating bytes since the Rust
            // relay never reinterprets these as floats; we only check
            // length.
            temperature: vec![0x42u8; 50 * 50 * 30 * 4],
            alpha: vec![0x3Fu8; 50 * 50 * 30 * 4],
            bubbles: vec![
                BubbleState { position: [0.01, 0.0, 0.02], radius: 1.0e-4 },
                BubbleState { position: [-0.01, 0.0, 0.03], radius: 2.0e-4 },
            ],
            nutrient_primary_name: "β-carotene".into(),
            nutrient_secondary_name: "vitamin C".into(),
            carrot_retention: 88.72,
            carrot_leached: 0.00,
            carrot_degraded: 11.16,
            carrot_precipitated: 0.12,
            carrot_retention2: 65.80,
            carrot_leached2: 21.03,
            carrot_degraded2: 13.17,
            carrot_precipitated2: 0.00,
            carrot_surface_c: vec![],
            carrot_surface_c2: vec![],
            wall_temperature_mean: 106.8,
            wall_heat_flux: 30_000.0,
            // v3
            water_temperature_mean: 99.88,
            water_temperature_max: 103.1,
            water_temperature_min: 97.4,
            run_id: "e3b0c44298fc1c149afbf4c8996fb924".into(),
            total_time_s: 600.0,
            is_complete: false,
            last_error: String::new(),
            // v4
            pot_diameter_m: 0.20,
            pot_height_m: 0.12,
            pot_wall_thickness_m: 0.003,
            pot_base_thickness_m: 0.005,
            // v6: 3 horizontal carrots in the default pot.
            carrot_count: 3,
            carrot_axis: 0,
            carrot_diameter_m: 0.025,
            carrot_length_m: 0.060,
            carrot_centres: vec![
                [0.0, -0.030, 0.040],
                [0.0,  0.000, 0.040],
                [0.0,  0.030, 0.040],
            ],
            carrot_total_mass_g: 91.9,
        }
    }

    #[test]
    fn roundtrip_is_lossless() {
        let snap = fixture_snapshot();
        let bytes = snap.to_msgpack_bytes().expect("encode");
        let back = Snapshot::from_msgpack_bytes(&bytes).expect("decode");
        assert_eq!(back.version, snap.version);
        assert_eq!(back.step, snap.step);
        assert_eq!(back.grid, snap.grid);
        assert_eq!(back.grid_ds, snap.grid_ds);
        assert_eq!(back.temperature.len(), snap.temperature.len());
        assert_eq!(back.alpha.len(), snap.alpha.len());
        assert_eq!(back.bubbles, snap.bubbles);
        assert!((back.carrot_retention - snap.carrot_retention).abs() < 1e-6);
        assert!((back.carrot_retention2 - snap.carrot_retention2).abs() < 1e-6);
    }

    #[test]
    fn grid_ds_is_half_resolution() {
        let snap = fixture_snapshot();
        assert_eq!(snap.grid_ds.nx, snap.grid.nx / 2);
        assert_eq!(snap.grid_ds.ny, snap.grid.ny / 2);
        assert_eq!(snap.grid_ds.nz, snap.grid.nz / 2);
        // v5: temperature/alpha hold 4 bytes per cell (little-endian f32).
        let expected_bytes =
            (snap.grid_ds.nx * snap.grid_ds.ny * snap.grid_ds.nz) as usize * 4;
        assert_eq!(snap.temperature.len(), expected_bytes);
        assert_eq!(snap.alpha.len(), expected_bytes);
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let mut snap = fixture_snapshot();
        snap.version = SCHEMA_VERSION + 1;
        let bytes = snap.to_msgpack_bytes().expect("encode");
        let err = Snapshot::from_msgpack_bytes(&bytes).expect_err("expected version rejection");
        match err {
            SnapshotError::VersionMismatch { got, expected } => {
                assert_eq!(got, SCHEMA_VERSION + 1);
                assert_eq!(expected, SCHEMA_VERSION);
            }
            SnapshotError::Decode(e) => panic!("expected version mismatch, got decode error: {e}"),
        }
    }

    #[test]
    fn version_zero_is_rejected() {
        let mut snap = fixture_snapshot();
        snap.version = 0;
        let bytes = snap.to_msgpack_bytes().expect("encode");
        assert!(matches!(
            Snapshot::from_msgpack_bytes(&bytes),
            Err(SnapshotError::VersionMismatch { got: 0, .. })
        ));
    }

    #[test]
    fn retention_fields_are_in_expected_range() {
        let snap = fixture_snapshot();
        assert!(snap.carrot_retention >= 0.0 && snap.carrot_retention <= 100.5);
        assert!(snap.carrot_retention2 >= 0.0 && snap.carrot_retention2 <= 100.5);
    }
}
