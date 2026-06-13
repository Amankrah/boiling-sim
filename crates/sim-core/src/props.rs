//! CPU-only mirror of `boilingsim.thermal.MaterialProps`.
//!
//! Reads `data/materials.json` directly (with the same `#` / `//` line-comment
//! handling as the Python loader at [`json_hash_comments.py`]) so the JSON
//! stays a single source of truth across both languages. No CUDA dependency
//! here — this module is importable on a CUDA-less machine, which is what
//! lets the test suite cover the loader on CI runners that don't have a GPU.

use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Strip `#` / `//` line comments from a JSON document, preserving `//`
/// sequences that appear inside string literals. Exact mirror of
/// `python/boilingsim/json_hash_comments.py:strip_hash_comments` so a
/// future test can diff their outputs byte-for-byte if drift is ever
/// suspected.
fn strip_hash_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (idx, line) in text.lines().enumerate() {
        let mut buf = String::with_capacity(line.len());
        let mut chars = line.chars().peekable();
        let mut in_string = false;
        let mut escape = false;
        while let Some(c) = chars.next() {
            if escape {
                buf.push(c);
                escape = false;
                continue;
            }
            if in_string {
                if c == '\\' {
                    escape = true;
                    buf.push(c);
                } else if c == '"' {
                    in_string = false;
                    buf.push(c);
                } else {
                    buf.push(c);
                }
                continue;
            }
            if c == '"' {
                in_string = true;
                buf.push(c);
                continue;
            }
            if c == '/' && chars.peek().copied() == Some('/') {
                break;
            }
            if c == '#' {
                break;
            }
            buf.push(c);
        }
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(buf.trim_end());
    }
    out
}

#[derive(Debug, Deserialize)]
struct WaterBlock {
    rho_ref: f64,
    rho_l_100c: f64,
    c_p: f64,
    k: f64,
    mu_25c: f64,
    mu_100c: f64,
    sigma: f64,
    beta_25c: f64,
    beta_100c: f64,
    #[serde(rename = "T_sat")]
    t_sat: f64,
    h_lv: f64,
    rho_vapor: f64,
}

#[derive(Debug, Deserialize)]
struct ConstantsBlock {
    #[serde(rename = "R_gas")]
    _r_gas: f64,
    g: f64,
}

#[derive(Debug, Deserialize)]
struct MaterialsRoot {
    water: WaterBlock,
    constants: ConstantsBlock,
}

/// Subset of `boilingsim.thermal.MaterialProps` carrying the
/// saturation / hot-state water scalars (Boussinesq, bubble kernels, Sherwood).
/// Pot-material and carrot rho/c_p/k arrays stay on the Python side for
/// now — they require a `cfg.pot.material` choice and the Phase-5 kernels
/// don't read them.
///
/// Equivalent Python class: [`boilingsim.thermal.MaterialProps`].
#[pyclass(frozen, module = "sim_core.props")]
pub struct MaterialProps {
    #[pyo3(get)]
    pub rho_ref: f64,
    #[pyo3(get)]
    pub rho_l_100c: f64,
    #[pyo3(get)]
    pub c_p: f64,
    #[pyo3(get)]
    pub k: f64,
    #[pyo3(get)]
    pub mu_25c: f64,
    #[pyo3(get)]
    pub mu_100c: f64,
    #[pyo3(get)]
    pub sigma: f64,
    #[pyo3(get)]
    pub beta_25c: f64,
    #[pyo3(get)]
    pub beta_100c: f64,
    #[pyo3(get)]
    pub T_sat_k: f64,
    #[pyo3(get)]
    pub h_lv: f64,
    #[pyo3(get)]
    pub rho_v: f64,
    #[pyo3(get)]
    pub g: f64,
}

#[pymethods]
impl MaterialProps {
    /// Load materials.json (with `#`/`//` line comments) and return the
    /// MaterialProps. Mirrors :py:meth:`boilingsim.thermal.MaterialProps.from_scenario`
    /// for the JSON-side fields only.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Self> {
        Self::load_from_path(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Computed property: ν_water at 100 °C = mu_100c / rho_l_100c.
    /// Source of truth for the Sherwood ν used by the leach kernel.
    #[getter]
    fn nu_water_100c(&self) -> f64 {
        self.mu_100c / self.rho_l_100c
    }

    fn __repr__(&self) -> String {
        format!(
            "MaterialProps(beta_100c={:.4e}, T_sat_k={:.2}, sigma={:.4}, \
             nu_water_100c={:.4e})",
            self.beta_100c,
            self.T_sat_k,
            self.sigma,
            self.nu_water_100c(),
        )
    }
}

impl MaterialProps {
    fn load_from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let stripped = strip_hash_comments(&raw);
        let root: MaterialsRoot = serde_json::from_str(&stripped)
            .with_context(|| format!("failed to parse {} as MaterialsRoot", path.display()))?;
        let w = root.water;
        let c = root.constants;
        if w.rho_l_100c <= 0.0 || w.beta_100c <= 0.0 || w.mu_100c <= 0.0 {
            return Err(anyhow!(
                "materials.json water block has non-positive saturation property \
                 (rho_l_100c={}, beta_100c={}, mu_100c={}) -- expected post-Phase-5 schema",
                w.rho_l_100c,
                w.beta_100c,
                w.mu_100c
            ));
        }
        Ok(MaterialProps {
            rho_ref: w.rho_ref,
            rho_l_100c: w.rho_l_100c,
            c_p: w.c_p,
            k: w.k,
            mu_25c: w.mu_25c,
            mu_100c: w.mu_100c,
            sigma: w.sigma,
            beta_25c: w.beta_25c,
            beta_100c: w.beta_100c,
            T_sat_k: w.t_sat,
            h_lv: w.h_lv,
            rho_v: w.rho_vapor,
            g: c.g,
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MaterialProps>()?;
    Ok(())
}
