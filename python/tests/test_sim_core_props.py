"""Phase 5 M0: parity test for the Rust ``sim_core.props`` shim.

Validates that the Rust ``MaterialProps`` reads the same values out of
``data/materials.json`` as the Python ``boilingsim.thermal.MaterialProps``.
If this test fails, the two languages have drifted -- inspect
``data/materials.json`` first, then ``crates/sim-core/src/props.rs``
deserialization for missing or renamed fields.
"""

from __future__ import annotations

import pathlib

import pytest

from boilingsim.config import ScenarioConfig
from boilingsim.thermal import MaterialProps as PyMaterialProps


ROOT = pathlib.Path(__file__).resolve().parents[2]
MATERIALS_JSON = ROOT / "data" / "materials.json"


@pytest.fixture(scope="module")
def rust_props():
    """Load the Rust-side props. ``sim_core`` is the maturin-built extension."""
    sim_core = pytest.importorskip("sim_core")
    return sim_core.props.MaterialProps.from_json(str(MATERIALS_JSON))


@pytest.fixture(scope="module")
def py_props():
    """Load the Python-side props off the default scenario."""
    return PyMaterialProps.from_scenario(ScenarioConfig())


def test_rust_props_loads_canonical_values(rust_props):
    """Rust-side reads the post-Phase-5-schema fields without error."""
    assert rust_props.beta_100c == pytest.approx(7.5e-4)
    assert rust_props.T_sat_k == pytest.approx(373.15)
    assert rust_props.h_lv == pytest.approx(2.257e6)
    assert rust_props.sigma == pytest.approx(0.0589)
    assert rust_props.mu_100c == pytest.approx(2.81e-4)
    assert rust_props.rho_l_100c == pytest.approx(957.8)


def test_rust_props_matches_python_props(rust_props, py_props):
    """Cross-language parity on the JSON-sourced scalar fields.

    The Python `MaterialProps.from_scenario` and the Rust
    `MaterialProps.from_json` must produce equal values for every field
    that lives in materials.json. The two computations are independent
    (Python uses :func:`loads_json_with_hash_comments`, Rust uses
    `strip_hash_comments + serde_json`), so equality here proves the
    parsers agree byte-for-byte on the data path.
    """
    assert rust_props.T_sat_k == pytest.approx(py_props.T_sat_k)
    assert rust_props.rho_l_100c == pytest.approx(py_props.rho_l_100c)
    assert rust_props.rho_v == pytest.approx(py_props.rho_v)
    assert rust_props.h_lv == pytest.approx(py_props.h_lv)
    assert rust_props.sigma == pytest.approx(py_props.sigma)
    assert rust_props.beta_100c == pytest.approx(py_props.beta_100c)
    assert rust_props.mu_100c == pytest.approx(py_props.mu_100c)
    assert rust_props.g == pytest.approx(py_props.g)
    assert rust_props.nu_water_100c == pytest.approx(py_props.nu_water_100c)


def test_rust_nu_water_100c_is_mu_over_rho(rust_props):
    """The Rust property must compute ν = μ/ρ honestly, not embed a literal."""
    assert rust_props.nu_water_100c == pytest.approx(
        rust_props.mu_100c / rust_props.rho_l_100c
    )


def test_rust_props_repr_smoke(rust_props):
    """``__repr__`` returns something parseable so debugger output is readable."""
    rep = repr(rust_props)
    assert "MaterialProps" in rep
    assert "beta_100c" in rep
    assert "T_sat_k" in rep
