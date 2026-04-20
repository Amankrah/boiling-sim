//! Cross-stack verification that a msgpack buffer produced by the
//! Python `build_snapshot` serializer round-trips through the Rust
//! deserializer with every field preserved.
//!
//! The `target/sample_snapshot.mp` fixture is captured by running
//! `scripts/capture_sample_snapshot.py` once and committed for
//! reproducibility; regenerate when the schema bumps.

use std::fs;
use std::path::PathBuf;

use ws_server::snapshot::{Snapshot, SCHEMA_VERSION};

fn fixture_path() -> PathBuf {
    // Workspace root is two levels above the crate dir.
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(crate_dir)
        .join("..")
        .join("..")
        .join("target")
        .join("sample_snapshot.mp")
}

#[test]
fn python_msgpack_deserializes_and_matches_schema() {
    let path = fixture_path();
    if !path.exists() {
        // Developer hasn't captured the fixture yet. Skip rather than
        // hard-fail so first-time clones of the repo aren't stuck.
        // See scripts/capture_sample_snapshot.py.
        eprintln!(
            "SKIP: fixture {} not found; run scripts/capture_sample_snapshot.py",
            path.display()
        );
        return;
    }

    let bytes = fs::read(&path).expect("read fixture");
    let snap = Snapshot::from_msgpack_bytes(&bytes).expect("decode Python snapshot");

    // Version match (also trivially enforced by `from_msgpack_bytes`).
    assert_eq!(snap.version, SCHEMA_VERSION);

    // Downsampled grid is half-resolution.
    assert_eq!(snap.grid_ds.nx, snap.grid.nx / 2);
    assert_eq!(snap.grid_ds.ny, snap.grid.ny / 2);
    assert_eq!(snap.grid_ds.nz, snap.grid.nz / 2);

    // Buffer length matches the downsampled cell count.
    let expected_len = (snap.grid_ds.nx * snap.grid_ds.ny * snap.grid_ds.nz) as usize;
    assert_eq!(
        snap.temperature.len(),
        expected_len,
        "temperature buffer length != nx_ds*ny_ds*nz_ds"
    );
    assert_eq!(snap.alpha.len(), expected_len);

    // Retention fields are the mass-partition percentages -- must sit in
    // [0, 100+eps]. At t < 1 s on a fresh default.yaml sim these should
    // still be close to 100 %.
    assert!(
        snap.carrot_retention >= 0.0 && snap.carrot_retention <= 100.5,
        "carrot_retention out of band: {}",
        snap.carrot_retention
    );
    assert!(
        snap.carrot_retention2 >= 0.0 && snap.carrot_retention2 <= 100.5,
        "carrot_retention2 out of band: {}",
        snap.carrot_retention2
    );

    // Sanity-check temperature range: solver runs in Kelvin but the
    // producer converts to Celsius. Nothing should be below 0 C or
    // above 200 C on a fresh warm-start.
    let t_min = snap
        .temperature
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let t_max = snap
        .temperature
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        t_min >= -5.0 && t_max <= 200.0,
        "temperature out of Celsius band: [{t_min}, {t_max}]"
    );
}
