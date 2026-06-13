//! Rust acceleration entry point for boiling-sim (Phase 5).
//!
//! Exposes two submodules to Python:
//!
//! * [`sim_core.props`](props) — CPU-only mirror of
//!   [`boilingsim.thermal.MaterialProps`]. Reads `data/materials.json`
//!   directly so the JSON is the single source of truth across both
//!   languages.
//! * [`sim_core.cuda`](cuda) — GPU handles + (in upcoming milestones)
//!   the hand-written Jacobi pressure kernel and CUDA Graph launcher.
//!   Lazily opens the CUDA driver on first `SimCore::new(...)`.

use pyo3::prelude::*;

mod cuda;
mod props;

#[pymodule]
fn sim_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Build the two submodules and add them to the parent.
    let props_mod = PyModule::new_bound(py, "props")?;
    props::register(&props_mod)?;
    m.add_submodule(&props_mod)?;

    let cuda_mod = PyModule::new_bound(py, "cuda")?;
    cuda::register(&cuda_mod)?;
    m.add_submodule(&cuda_mod)?;

    // PyO3 quirk: a submodule registered via add_submodule alone is
    // discoverable as `sim_core.props` but is NOT importable via
    // `from sim_core.props import X` unless we also register it in
    // `sys.modules`. The Python shim at `python/sim_core/__init__.py`
    // works around this for the package-level import, but the explicit
    // sys.modules entry here makes the C-extension hierarchy match
    // Python's expectations for IDE autocompletion and pickling.
    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("sim_core.props", &props_mod)?;
    modules.set_item("sim_core.cuda", &cuda_mod)?;

    Ok(())
}
