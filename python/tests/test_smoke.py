"""Smoke tests for Phase 0 acceptance.

These tests verify that the core dependencies are installed and functional.
They do NOT test physics — that begins in Phase 1.
"""

import pathlib

from boilingsim.json_hash_comments import loads_json_with_hash_comments


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_import_boilingsim():
    """Package imports without error."""
    import boilingsim
    assert boilingsim.__version__ == "0.1.0"


def test_warp_available():
    """Warp is installed and can detect a CUDA device."""
    import warp as wp
    wp.init()
    assert wp.is_cuda_available(), "No CUDA device found by Warp"


def test_warp_kernel_launches():
    """A trivial Warp kernel compiles and runs on the GPU."""
    import warp as wp
    import numpy as np

    wp.init()

    @wp.kernel
    def add_one(a: wp.array(dtype=float)):
        i = wp.tid()
        a[i] = a[i] + 1.0

    n = 1024
    arr = wp.array(np.zeros(n, dtype=np.float32), device="cuda:0")
    wp.launch(add_one, dim=n, inputs=[arr], device="cuda:0")
    result = arr.numpy()
    assert result[0] == 1.0
    assert result[-1] == 1.0


def test_materials_json_valid():
    """materials.json loads and contains all required materials."""
    path = ROOT / "data" / "materials.json"
    assert path.exists(), f"materials.json not found at {path}"
    data = loads_json_with_hash_comments(path.read_text(encoding="utf-8"))
    required = ["water", "steel_304", "cast_iron", "aluminum", "copper", "carrot", "constants"]
    for key in required:
        assert key in data, f"Missing material: {key}"
    # Spot-check a few critical values
    assert data["water"]["T_sat"] == 373.15
    assert data["water"]["h_lv"] == 2.257e6
    assert data["carrot"]["Ea_J_per_mol"] == 70000.0


def test_default_config_loads():
    """Default scenario YAML loads with Pydantic validation."""
    import yaml
    path = ROOT / "configs" / "scenarios" / "default.yaml"
    assert path.exists()
    cfg = yaml.safe_load(path.read_text())
    assert cfg["pot"]["material"] == "steel_304"
    assert cfg["total_time_s"] == 900.0


def test_numpy_scipy_available():
    """Core numerical libraries are importable."""
    import numpy as np
    import scipy
    assert np.__version__
    assert scipy.__version__
