"""Rust acceleration package for boiling-sim (Phase 5).

This Python module is a thin shim around the maturin-built Rust extension
that lives at ``sim_core/sim_core.{pyd,so}``. The Rust binding registers
two submodules:

* ``sim_core.props`` — CPU-only, always importable. Mirrors
  :class:`boilingsim.thermal.MaterialProps` and reads
  ``data/materials.json`` directly so the JSON stays a single source of
  truth across both languages.
* ``sim_core.cuda`` — lazily initialised. Holds the GPU-backed kernels
  (Phase 5 pressure solver, future bubble scatter, etc). Importing this
  submodule on a machine without CUDA raises ``ImportError`` with a
  clear message; everything else stays usable.

The split is deliberate: it lets unit tests of :mod:`sim_core.props`
run on CPU-only CI lanes (and Macs) without forcing every contributor
to install the CUDA Toolkit just to read the JSON.
"""

from __future__ import annotations

# Maturin installs the compiled extension at ``sim_core/sim_core.{pyd,so}``.
# Re-export every symbol it registers at top level so callers can write
# ``from sim_core import props`` or ``from sim_core.props import MaterialProps``
# without an extra hop through ``sim_core.sim_core``.
from .sim_core import *  # noqa: F401,F403

# Public submodule handles: explicit re-exports for IDE autocompletion +
# clean ``from sim_core.props import ...`` import semantics.
from .sim_core import props  # noqa: F401

try:  # pragma: no cover - the cuda submodule is exercised by GPU CI lane
    from .sim_core import cuda  # noqa: F401
except ImportError:
    # CUDA submodule registration failed (no driver, mismatched runtime,
    # or running on a CUDA-less platform). props stays usable; cuda
    # access raises a clearer error at call sites.
    cuda = None  # type: ignore[assignment]
