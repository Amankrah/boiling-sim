# Multiphysics 3D Boiling Simulation: Developer Guide

**A Step-by-Step Build Guide for GPU-Accelerated Coupled Thermal, Fluid, and Nutrient Retention Modeling**

*Carrot Boiling Case Study with NVIDIA Warp, CUDA, and Rust*

Prepared for: Emmanuel A. Kwofie, SASEL Lab, McGill University
Target machine: Lambda Vector workstation (Windows 11 Pro 25H2, Threadripper PRO 7975WX, 256 GB RAM, RTX 6000 Ada 48 GB)
Companion to: `multiphysics_boiling_simulation_plan.docx`

---

## Table of Contents

1. [How to Use This Guide](#how-to-use-this-guide)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Phase 0: Environment and Toolchain](#phase-0-environment-and-toolchain)
4. [Phase 1: Geometry and Parametric Scene](#phase-1-geometry-and-parametric-scene)
5. [Phase 2: Single-Phase CFD with Conjugate Heat Transfer](#phase-2-single-phase-cfd-with-conjugate-heat-transfer)
6. [Phase 3: Nucleate Boiling and Vapor Generation](#phase-3-nucleate-boiling-and-vapor-generation)
7. [Phase 4: Carrot Nutrient Retention Coupling](#phase-4-carrot-nutrient-retention-coupling)
8. [Phase 5: Rust and Custom CUDA Acceleration](#phase-5-rust-and-custom-cuda-acceleration)
9. [Phase 6: Live 3D Dashboard](#phase-6-live-3d-dashboard)
10. [Phase 7: Optional Omniverse Kit Migration](#phase-7-optional-omniverse-kit-migration)
11. [Appendix: Material Properties and Constants](#appendix-material-properties-and-constants)

---

## How to Use This Guide

This guide is the working companion to the project plan. Every phase contains four things: the scientific model with equations, the data structures and code scaffolding, the implementation steps in order, and the validation you must run before moving on. Do not skip the validation. A simulation that looks right is not the same as a simulation that is right.

All commands target the Lambda Vector workstation running **Windows 11 Pro 25H2 with WSL2 Ubuntu 24.04** as the primary development environment, with CUDA 12.6 or later, Python 3.11, Rust 1.75 or later, and Node.js 20. WSL2 is used because the Warp examples, warp.fem module, Rust CUDA crates, and every reference build in this domain assume Linux conventions. A Windows-native fallback is documented for the dashboard front end in Phase 6, since Node.js and the browser work equally well on Windows. Code snippets are illustrative and intended as starting points, not drop-in implementations. Every kernel needs profiling and unit testing before it enters the main pipeline.

### Hardware Budget

The RTX 6000 Ada Generation provides 48 GB of ECC GDDR6 on the same Ada Lovelace architecture as the RTX 4090 (`sm_89`), which means kernels compile identically but run with double the memory headroom. Combined with 256 GB of host RAM and 64 CPU threads, the baseline simulation resolution moves from 2 mm grid spacing to **0.5 mm**, and the bubble particle pool moves from 100,000 to 1,000,000. The carrot FE mesh can reach 200,000 tetrahedra without pressure on VRAM.

Repository layout is assumed to be:

```
boiling-sim/
├── Cargo.toml
├── pyproject.toml
├── package.json
├── crates/
│   ├── sim-core/            # Rust orchestration
│   ├── cuda-kernels/        # Hand-written CUDA (built with nvcc)
│   └── ws-server/           # WebSocket streaming
├── python/
│   ├── boilingsim/
│   │   ├── geometry.py
│   │   ├── fluid.py
│   │   ├── thermal.py
│   │   ├── boiling.py
│   │   ├── nutrient.py
│   │   └── pipeline.py
│   └── tests/
├── web/
│   ├── src/
│   └── public/
├── configs/
│   └── scenarios/           # YAML/JSON scenario definitions
├── data/
│   ├── materials.json
│   └── validation/
└── scripts/
```

---

## Mathematical Foundations

Before touching code, read this section once. Every phase refers back to these equations.

### 2.1 Fluid Dynamics: Navier-Stokes with Boussinesq Buoyancy

The water is treated as incompressible before vapor appears. The velocity field **u**(x, t) and pressure field p(x, t) satisfy:

**Continuity:**

$$
\nabla \cdot \mathbf{u} = 0
$$

**Momentum (Boussinesq approximation):**

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho_0}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}\,\beta(T - T_0)
$$

Where ρ₀ is a reference density (997 kg/m³ at 25 °C), ν is kinematic viscosity, **g** is gravity (−9.81 ẑ), β is the thermal expansion coefficient of water (≈ 2.07 × 10⁻⁴ 1/K near room temperature, rising to about 7.5 × 10⁻⁴ 1/K near 100 °C), and T₀ is the reference temperature.

We solve this with a **projection method** on a staggered Cartesian grid (MAC grid): velocities live on cell faces, pressure and temperature live at cell centers. The time step splits into:

1. **Advection step (semi-Lagrangian):** trace each grid velocity back along its own streamline and interpolate.
2. **Diffusion step (implicit):** solve (I − νΔt∇²)**u*** = **u**ⁿ⁺¹/² with a Jacobi or conjugate gradient sweep.
3. **Buoyancy step:** add Δt · **g** · β(T − T₀) to vertical velocity.
4. **Pressure projection:** solve ∇²p = (1/Δt) ∇ · **u*** and subtract Δt∇p.

The Poisson solve for pressure is the dominant cost and is the first kernel to profile.

### 2.2 Heat Transfer: Three-Domain Energy Equation

The energy equation holds in water, pot wall, and carrot with domain-specific ρ, c_p, k:

$$
\rho c_p \left( \frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T \right) = \nabla \cdot (k \nabla T) + S_T
$$

In solid domains (pot, carrot interior in the Eulerian sense) **u** = 0 and the advection term drops. S_T is a volumetric source, used for electrical heating of the pot base and for phase-change latent heat.

### 2.3 Conjugate Heat Transfer Interface Conditions

At the fluid-solid interface Γ (inner pot surface, carrot surface):

**Temperature continuity (Dirichlet):**

$$
T_f\big|_{\Gamma} = T_s\big|_{\Gamma}
$$

**Heat flux continuity (Neumann):**

$$
k_f \left(\nabla T_f \cdot \mathbf{n}\right)\big|_{\Gamma} = k_s \left(\nabla T_s \cdot \mathbf{n}\right)\big|_{\Gamma}
$$

In practice these are enforced through ghost cells or through a harmonic mean of thermal conductivities at the interface face:

$$
k_{\text{face}} = \frac{2 \, k_f \, k_s}{k_f + k_s}
$$

### 2.4 Phase Change: Volume of Fluid with Stefan Condition

A phase indicator α ∈ [0, 1] marks liquid (α = 1) and vapor (α = 0). Its transport equation with a phase-change source is:

$$
\frac{\partial \alpha}{\partial t} + \nabla \cdot (\alpha \mathbf{u}) = -\frac{\dot{m}}{\rho_l}
$$

Where ṁ is the interfacial mass flux per unit volume (kg/m³/s). At the liquid-vapor interface the **Stefan condition** relates ṁ to the heat flux jump:

$$
\dot{m} \, h_{lv} = k_l \left(\nabla T_l \cdot \mathbf{n}\right) - k_v \left(\nabla T_v \cdot \mathbf{n}\right)
$$

With h_lv = 2.257 × 10⁶ J/kg (latent heat of vaporization of water at 100 °C).

For household cookware we do **not** resolve the microlayer directly. We use a hybrid model described in Section 2.5.

### 2.5 Nucleate Boiling Sub-Model

**Onset of Nucleate Boiling (ONB):** bubbles begin forming when the wall superheat exceeds a threshold given by Hsu's criterion. A simplified practical criterion is:

$$
\Delta T_{\text{ONB}} = T_w - T_{\text{sat}} \geq 5 \text{ K}
$$

**Active Nucleation Site Density (Kocamustafaogullari-Ishii, 1983):**

$$
N_a = \frac{1}{D_c^2} \, F(\rho^*) \, (\Delta T_w)^{4.4}
$$

Where D_c is the critical cavity diameter, F(ρ*) is a density-ratio function, and ΔT_w is wall superheat in Kelvin. For engineering use we will tabulate N_a(ΔT_w) directly.

**Bubble Departure Diameter (Fritz, 1935):**

$$
D_d = 0.0208 \, \theta \sqrt{\frac{\sigma}{g(\rho_l - \rho_v)}}
$$

Where θ is the contact angle in radians (typically 0.7 to 1.4 for water on steel), σ is the surface tension of water (≈ 0.059 N/m at 100 °C), and ρ_l, ρ_v are liquid and vapor densities.

**Bubble Departure Frequency (Cole, 1960):**

$$
f = \sqrt{\frac{4 g (\rho_l - \rho_v)}{3 D_d \rho_l}}
$$

**Bubble Growth (Rayleigh-Plesset, simplified thermal growth):**

For thermally-driven growth in superheated liquid, the Mikic-Rohsenow model gives the radius R(t):

$$
R(t) = \frac{2}{\sqrt{\pi}} \, Ja \sqrt{\alpha_l \, t}
$$

Where Ja is the Jakob number and α_l is the thermal diffusivity of the liquid:

$$
Ja = \frac{\rho_l c_{p,l} (T_l - T_{\text{sat}})}{\rho_v h_{lv}}, \qquad \alpha_l = \frac{k_l}{\rho_l c_{p,l}}
$$

**Rohsenow Correlation (validation target for wall heat flux in fully developed nucleate boiling):**

$$
\frac{c_{p,l} \Delta T_w}{h_{lv}} = C_{sf} \left[\frac{q''_w}{\mu_l h_{lv}} \sqrt{\frac{\sigma}{g(\rho_l - \rho_v)}}\right]^{0.33} Pr_l^{n}
$$

With C_sf ≈ 0.013 for water on stainless steel, n ≈ 1.0 for water.

### 2.6 Beta-Carotene Degradation Kinetics (Carrot Model)

Inside the carrot the beta-carotene concentration C(x, t) evolves under coupled **reaction-diffusion**:

$$
\frac{\partial C}{\partial t} = D_{\text{eff}} \nabla^2 C - k(T) \, C
$$

With **first-order Arrhenius** temperature dependence:

$$
k(T) = k_0 \, \exp\!\left(-\frac{E_a}{R T}\right)
$$

Where E_a ≈ 66 to 79 kJ/mol for carotenoids in carrot tissue (per Koca et al., Knockaert et al.), R = 8.314 J/mol·K, T in Kelvin, and k_0 is the pre-exponential factor calibrated per sample.

**Effective diffusivity** D_eff for water-soluble carotenoid fragments in carrot tissue is roughly 1 × 10⁻¹⁰ to 5 × 10⁻¹⁰ m²/s depending on temperature and tissue damage state.

**Surface leaching boundary condition** at the carrot-water interface Γ_c:

$$
-D_{\text{eff}} \, (\nabla C \cdot \mathbf{n})\big|_{\Gamma_c} = h_m \left( C\big|_{\Gamma_c} - K_p \, C_{\text{water}} \right)
$$

Where h_m is the mass transfer coefficient (depends on local water velocity via a Sherwood correlation), K_p is the partition coefficient between carrot matrix and water (typically K_p < 1 for hydrophobic carotenoids), and C_water is the concentration in the surrounding water.

**Integrated Retention Percentage:**

$$
R(t) = \frac{\int_{V_c} C(x, t) \, dV}{\int_{V_c} C(x, 0) \, dV} \times 100\%
$$

This is the quantity to compare against the 84% experimental and 80-90% model predictions from the existing SASEL poster.

### 2.7 Numerical Stability Constraints

**CFL condition for advection:**

$$
\Delta t \leq \frac{C_{\text{CFL}} \, \Delta x}{|u|_{\max}}, \qquad C_{\text{CFL}} \leq 0.5
$$

**Diffusion stability (explicit):**

$$
\Delta t \leq \frac{\Delta x^2}{2 \, d \, \max(\nu, \alpha)}
$$

Where d is the spatial dimension (3 here). This is usually the binding constraint near hot walls with high thermal diffusivity; use an implicit diffusion solve to relax it.

**Surface tension (capillary) time step:**

$$
\Delta t \leq \sqrt{\frac{(\rho_l + \rho_v) \, \Delta x^3}{4 \pi \sigma}}
$$

For the final pipeline, pick Δt = min of all three with a safety factor of 0.8.

---

## Phase 0: Environment and Toolchain

**Duration:** 2 weeks. **Goal:** verified stack with baseline kernel benchmarks on the Lambda Vector.

The Lambda Vector runs Windows 11 Pro 25H2. The primary development environment is WSL2 Ubuntu 24.04 because every tool in our stack (Warp, warp.fem, cudarc, nvcc build pipelines, USD tooling) is best supported there. Windows retains the native role for the web browser, dashboard testing, and any Omniverse Kit streaming client in Phase 7.

### 0.1 Windows Host: NVIDIA Driver and WSL2

On the Windows host, run PowerShell as Administrator.

```powershell
# Install the NVIDIA RTX Enterprise Driver for RTX 6000 Ada from nvidia.com/drivers
# Verify with:
nvidia-smi
# Should report: NVIDIA RTX 6000 Ada Generation, 49140 MiB, driver 555+ or later

# Enable WSL2 and install Ubuntu 24.04
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2

# After reboot and first Ubuntu login:
wsl --update
wsl --status   # confirm default version = 2
```

**Critical rule:** install the NVIDIA driver *only* on the Windows host. Do **not** install `nvidia-*` packages inside WSL. The Windows driver exposes the GPU to WSL automatically through `/usr/lib/wsl/lib/libcuda.so.1`.

### 0.2 WSL2 Ubuntu: System Packages

Drop into the Ubuntu shell (`wsl -d Ubuntu-24.04`) and install:

```bash
sudo apt update && sudo apt install -y \
    build-essential git curl pkg-config libssl-dev \
    cmake ninja-build python3-dev python3-venv \
    libblas-dev liblapack-dev libhdf5-dev \
    clang lld unzip

# Verify GPU passthrough from inside WSL
nvidia-smi
```

Confirm you see the RTX 6000 Ada Generation with 49 GB, driver version 555 or later. If nvidia-smi is not found, the Windows driver was installed incorrectly or WSL needs a `wsl --shutdown` then a fresh launch.

### 0.3 CUDA Toolkit Inside WSL

Use the WSL-specific CUDA package, never the generic Linux one:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version    # should report CUDA 12.6
# Smoke test: compile and run a trivial kernel
cat > /tmp/hello.cu <<'EOF'
#include <cstdio>
__global__ void hi() { printf("GPU thread %d alive\n", threadIdx.x); }
int main() { hi<<<1, 4>>>(); cudaDeviceSynchronize(); return 0; }
EOF
nvcc /tmp/hello.cu -o /tmp/hello && /tmp/hello
```

### 0.4 Python Environment with uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd ~ && git init boiling-sim && cd boiling-sim
uv venv --python 3.11
source .venv/bin/activate
uv pip install \
    warp-lang[examples] \
    numpy scipy matplotlib h5py \
    pyvista trimesh pygmsh meshio \
    usd-core \
    fastapi "uvicorn[standard]" websockets \
    zstandard pyyaml pydantic \
    pytest pytest-benchmark
```

### 0.5 Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
rustup default stable
rustup component add clippy rustfmt

cargo install cargo-watch cargo-nextest maturin
```

Add to a root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/sim-core", "crates/cuda-kernels", "crates/ws-server"]

[workspace.dependencies]
cudarc = { version = "0.12", features = ["cuda-12060"] }
libloading = "0.8"
pyo3 = { version = "0.22", features = ["extension-module"] }
tokio = { version = "1", features = ["full"] }
axum = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
zstd = "0.13"
anyhow = "1"
cc = "1"
```

### 0.6 Node.js for Dashboard

Two options. Run inside WSL for a single unified environment:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g pnpm
```

Or install natively on Windows using the LTS installer from nodejs.org and run the front end in PowerShell while the solver runs in WSL. The WebSocket server listens on `0.0.0.0:8080` and WSL exposes it to Windows via `localhost` automatically.

### 0.7 Verify Warp Works

```bash
python -m warp.examples.core.example_sph
python -m warp.examples.fem.example_diffusion
```

Both render to USD files in the current directory. View them in Windows with USD Composer (free download from NVIDIA) or convert to PNG with `usdrecord` and open from the WSL filesystem at `\\wsl$\Ubuntu-24.04\home\<user>\`. Record throughput (particles per second for SPH, DoF per second for FEM) in `benchmarks/baseline.md`.

Expected rough baseline on the RTX 6000 Ada: SPH at 10 million particles should step at 80 to 120 fps, versus 40 to 60 fps on the RTX 4090.

### 0.8 Verify Rust + CUDA Works

Create `crates/cuda-kernels/build.rs`:

```rust
fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode").flag("arch=compute_89,code=sm_89")  // Ada Lovelace (RTX 6000 Ada, 4090, 4080, L40)
        .file("src/vector_add.cu")
        .compile("vector_add");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
```

With a `src/vector_add.cu`:

```cuda
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

Add a Rust binding and run a smoke test that launches the kernel on 10⁶ elements and compares against a CPU reference. Record the GB/s achieved. On the RTX 6000 Ada expect 900+ GB/s for a saturated vector-add, which is close to the card's 960 GB/s memory bandwidth ceiling.

### 0.9 WSL2 Memory and File-System Notes

Two gotchas specific to WSL2 on a 256 GB host:

**1. Default WSL memory cap.** WSL2 caps RAM at 50% of the host by default. To give the simulation access to the full 256 GB, create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=240GB
processors=64
swap=0
```

Then from PowerShell: `wsl --shutdown` and restart.

**2. Disk I/O.** Keep the project in the WSL native filesystem (`~/boiling-sim`), not on `/mnt/c/`. The `/mnt/c/` path crosses the 9P protocol bridge to NTFS and can be 20x slower for small-file operations like Git and Python imports.

### 0.10 Phase 0 Acceptance

Before leaving this phase you must have:

- `nvidia-smi` inside WSL shows the RTX 6000 Ada with 48 GB and driver 555+.
- CUDA 12.6 compiles and runs a hello-world kernel.
- Warp SPH and FEM examples run and throughput is recorded.
- A Rust program successfully launches a CUDA kernel through cudarc at near-peak bandwidth.
- `.wslconfig` grants at least 240 GB RAM and all 64 threads to WSL.
- A `benchmarks/baseline.md` file recording measured throughput of each sample kernel, with driver version, CUDA version, and date.

---

## Phase 1: Geometry and Parametric Scene

**Duration:** 3 weeks. **Goal:** a parametric USD scene (pot + water + carrot) queryable from Warp.

### 1.1 Configuration Schema

Create `python/boilingsim/config.py` with a Pydantic schema:

```python
from pydantic import BaseModel, Field
from typing import Literal

class PotConfig(BaseModel):
    diameter_m: float = 0.20           # 20 cm
    height_m: float = 0.12             # 12 cm
    wall_thickness_m: float = 0.003    # 3 mm
    base_thickness_m: float = 0.005    # 5 mm
    material: Literal["steel_304", "cast_iron", "aluminum", "copper"] = "steel_304"

class WaterConfig(BaseModel):
    fill_fraction: float = 0.75        # of pot height
    initial_temp_c: float = 20.0

class CarrotConfig(BaseModel):
    diameter_m: float = 0.025          # 2.5 cm
    length_m: float = 0.05             # 5 cm
    position: tuple[float, float, float] = (0.0, 0.0, 0.03)
    initial_beta_carotene_mg_per_100g: float = 8.3

class HeatingConfig(BaseModel):
    base_heat_flux_w_per_m2: float = 30000.0   # ~1000 W on a 20 cm base
    ambient_temp_c: float = 22.0

class GridConfig(BaseModel):
    dx_m: float = 0.0005               # 0.5 mm production resolution on 48 GB VRAM
    carrot_mesh_resolution: int = 80   # tets per axis, ~200k total

class ScenarioConfig(BaseModel):
    pot: PotConfig
    water: WaterConfig
    carrot: CarrotConfig
    heating: HeatingConfig
    grid: GridConfig
    total_time_s: float = 600.0
    output_every_s: float = 0.1
```

### 1.2 Pot Geometry via Signed Distance Fields

Represent the pot as the boolean difference of two cylinders, then sample into a level set on the grid. In `python/boilingsim/geometry.py`:

```python
import warp as wp
import numpy as np

@wp.func
def sdf_cylinder(p: wp.vec3, r: float, h: float) -> float:
    d_xy = wp.length(wp.vec2(p[0], p[1])) - r
    d_z = wp.abs(p[2] - h * 0.5) - h * 0.5
    outside = wp.length(wp.vec2(wp.max(d_xy, 0.0), wp.max(d_z, 0.0)))
    inside = wp.min(wp.max(d_xy, d_z), 0.0)
    return outside + inside

@wp.kernel
def build_pot_sdf(
    sdf: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    r_outer: float,
    r_inner: float,
    h_outer: float,
    h_inner: float,
    base_thickness: float,
):
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i), float(j), float(k)) * dx

    d_outer = sdf_cylinder(p, r_outer, h_outer)
    inner_offset = wp.vec3(0.0, 0.0, base_thickness)
    d_inner = sdf_cylinder(p - inner_offset, r_inner, h_inner)

    # Pot = outer minus inner
    sdf[i, j, k] = wp.max(d_outer, -d_inner)
```

Validate: raymarch the SDF in a separate debug script and render to PNG. Confirm the wall thickness and base thickness are correct.

### 1.3 Water Domain

Water occupies the pot interior up to the fill fraction. Its volume fraction α(x, 0) = 1 where the cell is inside the pot and below the water line, else 0:

```python
@wp.kernel
def init_water_volume_fraction(
    alpha: wp.array3d(dtype=float),
    pot_sdf: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    water_line_z: float,
    base_thickness: float,
):
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i), float(j), float(k)) * dx
    inside_pot = pot_sdf[i, j, k] > 0.0
    below_water = (p[2] > base_thickness) and (p[2] < water_line_z)
    alpha[i, j, k] = 1.0 if (inside_pot and below_water) else 0.0
```

### 1.4 Carrot Tetrahedral Mesh

Use `pygmsh` or `gmsh` to generate a tet mesh for the carrot cylinder, then load into a Warp mesh:

```python
import pygmsh
import numpy as np

def build_carrot_mesh(diameter: float, length: float, resolution: int):
    with pygmsh.occ.Geometry() as geom:
        cyl = geom.add_cylinder([0, 0, 0], [0, 0, length], diameter / 2)
        geom.characteristic_length_max = length / resolution
        mesh = geom.generate_mesh(dim=3)
    # Extract points and tet connectivity
    points = mesh.points.astype(np.float32)
    tets = mesh.cells_dict["tetra"].astype(np.int32)
    surface_tris = mesh.cells_dict["triangle"].astype(np.int32)
    return points, tets, surface_tris
```

Wrap the carrot surface as a Warp mesh so the fluid solver can query it for the immersed-boundary coupling:

```python
surface_points, surface_indices = get_surface_mesh(carrot_points, carrot_tets)
carrot_mesh = wp.Mesh(
    points=wp.array(surface_points, dtype=wp.vec3),
    indices=wp.array(surface_indices.flatten(), dtype=int),
)
```

### 1.5 Export to USD

```python
from pxr import Usd, UsdGeom, Gf

def export_scene_usd(path, pot_mesh, water_surface_mesh, carrot_mesh):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    for name, mesh in [("Pot", pot_mesh), ("Water", water_surface_mesh), ("Carrot", carrot_mesh)]:
        usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        usd_mesh.CreatePointsAttr(mesh.points.tolist())
        usd_mesh.CreateFaceVertexIndicesAttr(mesh.indices.tolist())
        usd_mesh.CreateFaceVertexCountsAttr([3] * (len(mesh.indices) // 3))
    stage.GetRootLayer().Save()
```

Open the resulting `.usd` in USD Composer (or just `pyvista.read`) and confirm visually.

### 1.6 Phase 1 Acceptance

- `scenario.py --config configs/scenarios/default.yaml` produces a USD file.
- Wall thickness is visually correct when cross-sectioned.
- Carrot tet count is in the 100,000 to 250,000 range (production) or 10,000 to 40,000 for quick iteration.
- Material properties load from `data/materials.json` (see Appendix).

---

## Phase 2: Single-Phase CFD with Conjugate Heat Transfer

**Duration:** 5 weeks. **Goal:** heat water from 20 °C to 95 °C, with natural convection, validated against lumped capacitance.

### 2.1 Grid Layout (MAC Staggered Grid)

- Cell centers (N_x × N_y × N_z) hold: pressure p, temperature T, α
- Face centers hold: u_x on (N_x + 1) × N_y × N_z faces, u_y on N_x × (N_y + 1) × N_z, u_z on N_x × N_y × (N_z + 1)

```python
class Grid:
    def __init__(self, nx, ny, nz, dx):
        self.nx, self.ny, self.nz, self.dx = nx, ny, nz, dx
        self.T = wp.zeros((nx, ny, nz), dtype=float)
        self.p = wp.zeros((nx, ny, nz), dtype=float)
        self.alpha = wp.zeros((nx, ny, nz), dtype=float)
        self.ux = wp.zeros((nx + 1, ny, nz), dtype=float)
        self.uy = wp.zeros((nx, ny + 1, nz), dtype=float)
        self.uz = wp.zeros((nx, ny, nz + 1), dtype=float)
        # Material ID per cell: 0=fluid, 1=pot_wall, 2=air, 3=carrot
        self.mat = wp.zeros((nx, ny, nz), dtype=int)
```

Memory estimate for 400 × 400 × 240 grid at 0.5 mm resolution (production): about 3.8 × 10⁷ cells × ~40 bytes per cell across 8 fields ≈ 12 GB. Leaves 36 GB of VRAM free for bubbles, carrot FE system, and checkpoint copies. Drop to 200 × 200 × 120 at 1 mm resolution for rapid iteration during development (≈ 1.5 GB per full field set).

### 2.2 Time-Stepping Loop

```python
def step(grid, dt, cfg):
    advect_velocity(grid, dt)        # semi-Lagrangian
    advect_temperature(grid, dt)     # semi-Lagrangian, skip in solids
    apply_buoyancy(grid, dt, cfg)
    diffuse_temperature(grid, dt)    # implicit, conjugate across interfaces
    diffuse_velocity(grid, dt)       # implicit viscosity
    apply_boundary_conditions(grid, cfg)
    pressure_projection(grid, dt)    # enforces incompressibility
    apply_heat_sources(grid, dt, cfg) # stove flux at base
```

### 2.3 Semi-Lagrangian Advection Kernel

```python
@wp.kernel
def advect_scalar(
    field_new: wp.array3d(dtype=float),
    field_old: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    dt: float,
):
    i, j, k = wp.tid()
    if mat[i, j, k] != 0:  # only advect in fluid
        field_new[i, j, k] = field_old[i, j, k]
        return

    # Sample velocity at cell center
    u = 0.5 * (ux[i, j, k] + ux[i + 1, j, k])
    v = 0.5 * (uy[i, j, k] + uy[i, j + 1, k])
    w = 0.5 * (uz[i, j, k] + uz[i, j, k + 1])

    # Trace back
    x_back = float(i) - u * dt / dx
    y_back = float(j) - v * dt / dx
    z_back = float(k) - w * dt / dx

    field_new[i, j, k] = trilinear_sample(field_old, x_back, y_back, z_back)
```

### 2.4 Pressure Projection: Poisson Solve

The discrete Laplacian on the MAC grid gives a symmetric positive-definite system. Use Warp's built-in `warp.fem` conjugate gradient, or write a Jacobi iteration for a first pass:

```python
@wp.kernel
def jacobi_pressure_step(
    p_new: wp.array3d(dtype=float),
    p_old: wp.array3d(dtype=float),
    div_u: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    dx: float,
    dt: float,
    rho: float,
):
    i, j, k = wp.tid()
    if mat[i, j, k] != 0:
        p_new[i, j, k] = 0.0
        return
    s = p_old[i - 1, j, k] + p_old[i + 1, j, k] \
      + p_old[i, j - 1, k] + p_old[i, j + 1, k] \
      + p_old[i, j, k - 1] + p_old[i, j, k + 1]
    rhs = rho * dx * dx * div_u[i, j, k] / dt
    p_new[i, j, k] = (s - rhs) / 6.0
```

For production, replace this with a preconditioned conjugate gradient from `warp.fem`. Jacobi converges slowly; CG is 10x to 100x faster for this problem size.

### 2.5 Conjugate Heat Transfer at Interfaces

Use the harmonic mean of thermal conductivities at fluid-solid faces:

```python
@wp.kernel
def heat_conduction_flux(
    T: wp.array3d(dtype=float),
    flux_x: wp.array3d(dtype=float),  # on x-faces
    mat: wp.array3d(dtype=int),
    k_fluid: float,
    k_pot: float,
    k_carrot: float,
    dx: float,
):
    i, j, k = wp.tid()  # face index
    if i == 0 or i >= flux_x.shape[0]:
        flux_x[i, j, k] = 0.0
        return
    m_left = mat[i - 1, j, k]
    m_right = mat[i, j, k]
    k_left = material_k(m_left, k_fluid, k_pot, k_carrot)
    k_right = material_k(m_right, k_fluid, k_pot, k_carrot)
    k_face = 2.0 * k_left * k_right / (k_left + k_right + 1e-12)
    flux_x[i, j, k] = -k_face * (T[i, j, k] - T[i - 1, j, k]) / dx
```

This single kernel handles all three pairs (fluid-pot, fluid-carrot, pot-air) automatically through the material ID lookup.

### 2.6 Boundary Conditions

- **Stove base:** constant heat flux q"_base into the bottom pot cells. Add q"_base · A_cell · dt / (ρ c_p V_cell) to T per step.
- **Outer pot surface (air side):** Newton's law of cooling with h_conv ≈ 10 W/m²·K and T_ambient = 22 °C.
- **Water free surface:** evaporative cooling, modeled as additional heat loss 0.1 × q"_base until boiling begins.
- **Side walls and top of domain:** outflow (zero-gradient) for velocity, adiabatic for temperature outside the pot.

### 2.7 Validation: Lumped Capacitance Reference

Total energy to raise water from 20 °C to 95 °C:

$$
E = m_{\text{water}} c_{p,\text{water}} \Delta T + m_{\text{pot}} c_{p,\text{pot}} \Delta T_{\text{pot}}
$$

For 1.5 L water, 500 g steel pot, heating 75 K:

$$
E \approx 1.5 \times 4186 \times 75 + 0.5 \times 500 \times 75 \approx 489{,}750 \text{ J}
$$

At 1000 W stove input with 80% efficiency to water:

$$
t \approx \frac{489{,}750}{800} \approx 612 \text{ s} \approx 10.2 \text{ min}
$$

Your simulation should land within 5-10% of this, across stainless steel, aluminum, and copper pots. Copper heats fastest (higher k, thinner thermal BL), stainless slowest. Record time-to-95 °C for each and confirm the ordering.

### 2.8 Phase 2 Acceptance

- Time-to-95 °C within 10% of lumped capacitance across three materials.
- Natural convection cells visible in the water: warm rising plume over the base, cooler descent along walls.
- Grid convergence: halving dx changes the final temperature by less than 1%.
- Wall time per simulated second: target < 1 s on the RTX 6000 Ada at 1 mm resolution (development), < 4 s at 0.5 mm resolution (production).

---

## Phase 3: Nucleate Boiling and Vapor Generation

**Duration:** 6 weeks. **Goal:** active boiling with bubbles that nucleate, grow, detach, rise, and vent.

### 3.1 Nucleation Site Detection

At each step, scan the inner pot surface cells. Where T_wall - T_sat > ΔT_ONB and no bubble is currently seeded within a radius D_d, spawn a new bubble with probability per unit time equal to the departure frequency f (Cole correlation).

```python
@wp.kernel
def detect_nucleation_sites(
    T: wp.array3d(dtype=float),
    mat: wp.array3d(dtype=int),
    site_active: wp.array3d(dtype=int),  # 1 if bubble growing here
    candidate_sites: wp.array(dtype=wp.vec3),
    num_candidates: wp.array(dtype=int),
    T_sat: float,
    dT_onb: float,
    origin: wp.vec3,
    dx: float,
):
    i, j, k = wp.tid()
    # Only bottom-pot-to-water interface cells
    if mat[i, j, k] != 0:  # not fluid
        return
    if mat[i, j, k - 1] != 1:  # not directly above pot wall
        return
    if site_active[i, j, k] == 1:
        return
    if T[i, j, k - 1] - T_sat < dT_onb:
        return

    idx = wp.atomic_add(num_candidates, 0, 1)
    candidate_sites[idx] = origin + wp.vec3(float(i), float(j), float(k)) * dx
```

### 3.2 Bubble Data Structure (Lagrangian Particles)

```python
@wp.struct
class Bubble:
    position: wp.vec3
    velocity: wp.vec3
    radius: float
    birth_time: float
    active: int
    site_ijk: wp.vec3i  # for deactivating the site on departure

MAX_BUBBLES = 1000000  # 1M particle pool fits comfortably in 48 GB VRAM
bubbles = wp.zeros(MAX_BUBBLES, dtype=Bubble)
```

### 3.3 Bubble Growth and Transport

```python
@wp.kernel
def update_bubbles(
    bubbles: wp.array(dtype=Bubble),
    T_field: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    site_active: wp.array3d(dtype=int),
    dt: float,
    dx: float,
    origin: wp.vec3,
    # Fluid properties
    T_sat: float,
    rho_l: float, rho_v: float, c_pl: float, h_lv: float, k_l: float,
    sigma: float, theta: float, g: float,
):
    b = wp.tid()
    bubble = bubbles[b]
    if bubble.active == 0:
        return

    # Sample local liquid temperature
    ijk = grid_index(bubble.position, origin, dx)
    T_local = T_field[ijk[0], ijk[1], ijk[2]]

    # Mikic-Rohsenow thermal growth
    Ja = rho_l * c_pl * wp.max(T_local - T_sat, 0.0) / (rho_v * h_lv)
    alpha_l = k_l / (rho_l * c_pl)
    age = wp.get_current_time() - bubble.birth_time  # pseudocode
    R_target = (2.0 / wp.sqrt(3.14159)) * Ja * wp.sqrt(alpha_l * age)
    bubble.radius = wp.min(R_target, bubble.radius + 0.0005 * dt / age)  # clamp

    # Fritz departure check
    D_d = 0.0208 * theta * wp.sqrt(sigma / (g * (rho_l - rho_v)))
    if bubble.radius * 2.0 >= D_d:
        # Detach: start rising under buoyancy
        bubble.velocity = wp.vec3(
            sample_lerp(ux, bubble.position, origin, dx),
            sample_lerp(uy, bubble.position, origin, dx),
            sample_lerp(uz, bubble.position, origin, dx) + 0.2,  # buoyancy kick
        )
        site_active[bubble.site_ijk[0], bubble.site_ijk[1], bubble.site_ijk[2]] = 0

    # Advect
    bubble.position = bubble.position + bubble.velocity * dt

    # Vent at free surface (z above water line)
    if bubble.position[2] > water_surface_z:
        bubble.active = 0

    bubbles[b] = bubble
```

### 3.4 Two-Way Momentum Coupling

Each bubble exerts a buoyancy force on surrounding fluid cells, proportional to its volume. The reaction appears as a source term in the momentum equation:

$$
\mathbf{f}_{\text{bubble}}(x) = \sum_{b} V_b (\rho_l - \rho_v) \mathbf{g} \, W(x - x_b, h)
$$

Where W is a compactly-supported kernel (trilinear or SPH cubic spline) with support radius h ≈ 2Δx.

### 3.5 Two-Way Energy Coupling (Latent Heat Sink)

Each growing bubble removes energy from the surrounding liquid at rate:

$$
\dot{Q}_b = \rho_v h_{lv} \frac{dV_b}{dt} = \rho_v h_{lv} 4\pi R^2 \frac{dR}{dt}
$$

Spread this sink across the liquid cells within the kernel support:

```python
@wp.kernel
def scatter_latent_heat(
    bubbles: wp.array(dtype=Bubble),
    T_field: wp.array3d(dtype=float),
    T_prev: wp.array3d(dtype=float),
    rho_v: float, h_lv: float, rho_l: float, c_pl: float,
    dx: float, dt: float, origin: wp.vec3,
):
    b = wp.tid()
    bubble = bubbles[b]
    if bubble.active == 0:
        return
    dR = bubble.radius - bubble.radius_prev
    dV = 4.0 * 3.14159 * bubble.radius * bubble.radius * dR
    Q = rho_v * h_lv * dV
    dT = -Q / (rho_l * c_pl * dx * dx * dx)

    # Distribute with trilinear weight
    ijk = grid_index(bubble.position, origin, dx)
    # ... scatter dT to 8 surrounding cells with trilinear weights
```

### 3.6 Volume of Fluid Evolution

When a bubble exists at cell (i, j, k), reduce the liquid volume fraction α proportionally to (bubble volume within cell) / (cell volume). This keeps mass balance consistent without resolving the interface sharply.

### 3.7 Validation Targets

1. **Rohsenow wall heat flux:** at steady boiling, integrate q"_w over the pot base. Compare to the Rohsenow correlation (Section 2.5 of Mathematical Foundations). Acceptance: within 30%.

2. **Time to rolling boil:** 1 L water at 1000 W. Published stopwatch data: 3-5 minutes from 95 °C to vigorous boil. Acceptance: within 20%.

3. **Bubble statistics:** departure diameter 2-3 mm, departure frequency 20-100 Hz for water on steel. Acceptance: median within the published range.

### 3.8 Phase 3 Acceptance

- Visible bubble column when the base heat flux is applied.
- Mean bubble departure diameter within [1.5 mm, 4 mm].
- Wall heat flux in the nucleate-boiling regime matches Rohsenow within 30%.
- No numerical blow-up for 10 minutes of simulated time at Δt = 2 ms.
- Wall time per simulated second: target < 6 s at 0.5 mm production resolution with active boiling on the RTX 6000 Ada.

---

## Phase 4: Carrot Nutrient Retention Coupling

**Duration:** 4 weeks. **Goal:** coupled simulation whose integrated retention matches the 80-90% SASEL band for a 10-minute boil.

### 4.1 Carrot Finite-Element Setup

Use `warp.fem` with a tetrahedral mesh. Define two fields:

- T_c(x, t): temperature inside carrot, solved with the general energy equation (Section 2.2).
- C(x, t): beta-carotene concentration, solved with reaction-diffusion (Section 2.6).

```python
import warp.fem as fem

# Build geometry from tet mesh
geo = fem.Tetmesh3D(
    positions=wp.array(carrot_points, dtype=wp.vec3),
    tet_vertex_indices=wp.array(carrot_tets, dtype=int),
)

# Linear function space for T and C
space_T = fem.make_polynomial_space(geo, degree=1)
space_C = fem.make_polynomial_space(geo, degree=1)

T_field = space_T.make_field()
C_field = space_C.make_field()
```

### 4.2 Weak Form for Beta-Carotene

For test function v, the weak form of ∂C/∂t = D∇²C − k(T)C is:

$$
\int_{V_c} \frac{\partial C}{\partial t} v \, dV = -D \int_{V_c} \nabla C \cdot \nabla v \, dV - \int_{V_c} k(T) C v \, dV + \int_{\Gamma_c} (D \nabla C \cdot \mathbf{n}) v \, dA
$$

The surface term becomes the leaching flux (Section 2.6). In `warp.fem` syntax:

```python
@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, D: float):
    return D * wp.dot(fem.grad(u, s), fem.grad(v, s))

@fem.integrand
def reaction_form(s: fem.Sample, u: fem.Field, v: fem.Field, T: fem.Field,
                  k0: float, Ea: float, R: float):
    T_local = T(s)
    k = k0 * wp.exp(-Ea / (R * T_local))
    return k * u(s) * v(s)
```

Time step with backward Euler or Crank-Nicolson. The assembled system at each step:

$$
(M + \Delta t \, D \, K + \Delta t \, M_k) C^{n+1} = M \, C^n + \Delta t \, f_{\text{boundary}}
$$

Solve with `fem.bsr_cg` (block sparse CG) from `warp.fem`.

### 4.3 Coupling Geometry: Carrot Surface ↔ Water Grid

The carrot sits inside the Eulerian water grid. Couple the two via immersed-boundary:

1. Every timestep, scan each carrot surface triangle.
2. Interpolate local water T and local water velocity magnitude |u| to each triangle's centroid using trilinear sampling on the Eulerian grid.
3. Use these as boundary conditions for the carrot FE solve:
   - T_carrot_surface = T_water at centroid (or a heat flux condition using h_conv from a Sherwood correlation).
   - Leaching flux uses h_m and the local |u|.
4. After the carrot solves, accumulate the leached mass back into the water grid's passive scalar field.

**Sherwood correlation** for mass transfer coefficient around a cylinder:

$$
Sh = 0.683 \, Re^{0.466} \, Sc^{1/3}
$$

Then:

$$
h_m = \frac{Sh \cdot D_{\text{water}}}{L_{\text{char}}}
$$

With D_water the molecular diffusivity in water and L_char the carrot diameter.

### 4.4 Kinetic Parameters

From literature (Koca 2007, Knockaert 2012, plus the encapsulation review):

| Parameter | Value | Units |
|-----------|-------|-------|
| E_a | 70,000 (default) | J/mol |
| k_0 | 1.58 × 10⁸ | 1/min (convert to 1/s) |
| D_eff (beta-carotene in carrot) | 2 × 10⁻¹⁰ | m²/s |
| K_p (partition) | 0.3 | dimensionless |
| C(x, 0) | 83 | mg/kg (8.3 mg per 100 g) |

Store in `data/materials.json` and allow per-scenario override.

### 4.5 Integrated Retention Output

At every output step:

```python
def compute_retention(C_field, volumes, C0_integral):
    # Integrate C over carrot volume using FE mass matrix
    C_integral = wp.utils.array_inner_product(C_field, volumes)
    return 100.0 * C_integral / C0_integral
```

Write R(t) to an HDF5 file and plot at end of run.

### 4.6 Validation: The SASEL Band

Run the pipeline for 10 minutes of simulated boil time. Expected output:

- Final retention R(600 s) ∈ [80%, 90%].
- Experimental reference value: 84% (per your poster).
- Spatial gradient: surface concentration < interior concentration (surface is leached).

Run three sensitivity cases:

1. Smaller carrot (12 mm diameter): expect lower retention, because higher surface-to-volume ratio → more leaching.
2. Larger carrot (40 mm diameter): expect higher retention.
3. Gentle simmer (85 °C steady vs rolling boil): expect higher retention due to slower Arrhenius rate.

If the model gets the qualitative rankings right and the 25 mm reference within the band, Phase 4 is validated.

### 4.7 Phase 4 Acceptance

- R(600 s) for the 25 mm reference carrot ∈ [80%, 90%].
- Correct ordering across the three sensitivity cases.
- Mass balance: total beta-carotene (carrot + water) conserved within 2% over the full run.
- Full coupled step time on the RTX 6000 Ada: target < 8 s of wall time per simulated second at 0.5 mm grid and 200k-tet carrot (production), < 2 s at 1 mm grid and 40k-tet carrot (development).
- With 1,000,000-bubble pool active and 200k-tet carrot, peak VRAM usage should stay under 35 GB, leaving headroom for checkpoints.

---

## Phase 5: Rust and Custom CUDA Acceleration

**Duration:** 4 weeks. **Goal:** 2-4x speedup on the three hottest kernels by replacing Warp implementations with hand-written CUDA driven from Rust.

### 5.1 Profile First

Run the profiler inside WSL and open the report on the Windows host for graphical analysis:

```bash
# Inside WSL
nsys profile --stats=true -o profiles/phase4_baseline \
    python -m boilingsim.pipeline --config configs/scenarios/default.yaml --steps 500
```

```powershell
# On Windows, after installing Nsight Systems from the NVIDIA Developer portal
nsys-ui.exe \\wsl$\Ubuntu-24.04\home\<user>\boiling-sim\profiles\phase4_baseline.nsys-rep
```

Rank kernels by cumulative time. The expected top three (from similar projects) are:

1. Pressure Poisson solve (CG iterations).
2. Scatter from Lagrangian bubbles to Eulerian grid.
3. FE stiffness matrix assembly for the carrot.

On the RTX 6000 Ada at production resolution, the Poisson solve will dominate even more than on the 4090 because the grid is 8x larger while memory bandwidth is only ~1.5x higher. This makes it the best ROI target for the custom CUDA work.

### 5.2 Rust Orchestration Layer

Structure `crates/sim-core/src/lib.rs`:

```rust
use pyo3::prelude::*;
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

#[pyclass]
pub struct SimCore {
    device: Arc<CudaDevice>,
    ptx_module: cudarc::driver::CudaModule,
}

#[pymethods]
impl SimCore {
    #[new]
    fn new(device_id: usize) -> PyResult<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        // Load our hand-compiled PTX
        let ptx = include_bytes!(env!("CUDA_PTX_PATH"));
        let module = device.load_ptx(ptx.into(), "boiling_kernels",
            &["poisson_cg_step", "bubble_scatter", "fe_assemble"])?;
        Ok(Self { device, ptx_module: module })
    }

    /// Zero-copy kernel launch on Warp arrays via __cuda_array_interface__
    fn poisson_step(&self, p_ptr: u64, rhs_ptr: u64, n: usize) -> PyResult<()> {
        let f = self.ptx_module.get_func("poisson_cg_step")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            f.launch(cfg, (p_ptr, rhs_ptr, n as i32))?;
        }
        self.device.synchronize()?;
        Ok(())
    }
}

#[pymodule]
fn sim_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SimCore>()?;
    Ok(())
}
```

Build with `maturin develop --release`. On the Python side:

```python
from boilingsim import sim_core
core = sim_core.SimCore(device_id=0)

# warp_array.__cuda_array_interface__['data'][0] is the raw device pointer
core.poisson_step(p.__cuda_array_interface__['data'][0],
                  rhs.__cuda_array_interface__['data'][0], n)
```

### 5.3 Hand-Written Poisson CG Kernel

The hot kernel is SpMV for the 7-point stencil (or 27-point for the FE carrot case). A manually tiled kernel with shared memory beats a generic library implementation on fixed grids:

```cuda
// crates/cuda-kernels/src/poisson_spmv.cu
extern "C" __global__
void poisson_spmv_7pt(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int* __restrict__ mat,  // 0=fluid, !=0=solid
    int nx, int ny, int nz,
    float inv_dx2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = (k * ny + j) * nx + i;
    if (mat[idx] != 0) { y[idx] = 0.0f; return; }

    float xc = x[idx];
    float s = 0.0f;
    int valid = 0;

    if (i > 0)      { s += x[idx - 1];        valid++; }
    if (i < nx - 1) { s += x[idx + 1];        valid++; }
    if (j > 0)      { s += x[idx - nx];       valid++; }
    if (j < ny - 1) { s += x[idx + nx];       valid++; }
    if (k > 0)      { s += x[idx - nx*ny];    valid++; }
    if (k < nz - 1) { s += x[idx + nx*ny];    valid++; }

    y[idx] = (s - float(valid) * xc) * inv_dx2;
}
```

Launch with 8x8x4 blocks. Use the cudarc API to set this up once per simulation and reuse.

### 5.4 Numerical Regression Tests

Critical: any custom kernel must produce **bitwise-tolerant** results against the Warp reference. Write tests that:

1. Run a 50-step simulation with Warp kernels only, dump state to HDF5.
2. Run the same 50 steps with the Rust/CUDA kernels swapped in.
3. Compute max absolute and max relative error per field.

Acceptance: max relative error < 1e-4 for pressure, < 1e-5 for temperature, < 1e-5 for C.

### 5.5 Phase 5 Acceptance

- Nsight report showing 2-4x speedup on each of the three target kernels.
- End-to-end pipeline wall time reduced by at least 30%.
- All scientific regression tests pass.
- `cargo nextest run --release` green in CI.

---

## Phase 6: Live 3D Dashboard

**Duration:** 5 weeks. **Goal:** web dashboard that streams simulation at 30-60 FPS with live parameter control.

### 6.1 Architecture

```
┌──────────────┐   CUDA+Warp   ┌──────────────┐
│  Python      │ ────────────→ │ RTX 6000 Ada │
│  Pipeline    │               │   fields     │
└──────┬───────┘               └──────────────┘
       │ snapshot (pinned host mem)
       ↓
┌──────────────┐
│ Rust Axum    │← control messages (params)
│ WS server    │→ compressed snapshots (zstd)
└──────┬───────┘
       │ WebSocket @ ws://host:8080/stream
       ↓
┌──────────────┐
│ React +      │
│ R3F viewer   │
└──────────────┘
```

### 6.2 Snapshot Format

Binary, zstd-compressed, versioned:

```rust
#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,           // 1
    pub t_sim: f32,             // simulated time in seconds
    pub step: u64,
    pub grid: GridMeta,         // nx, ny, nz, dx, origin
    // Downsampled fields (for streaming; full-res on pause)
    pub temperature: Vec<f32>,  // nx/2 × ny/2 × nz/2
    pub alpha: Vec<f32>,        // same resolution
    pub bubbles: Vec<BubbleState>,
    pub carrot_retention: f32,
    pub carrot_surface_C: Vec<f32>,  // per carrot-vertex
    pub wall_temperature_mean: f32,
    pub wall_heat_flux: f32,
}
```

Downsample before sending: the 200³ full grid is 32 MB uncompressed; downsampling to 100³ and zstd-compressing gets you to 1-2 MB, streamable at 10 Hz over a home network.

### 6.3 Rust WebSocket Server

```rust
use axum::{extract::WebSocketUpgrade, response::IntoResponse, routing::get, Router};
use axum::extract::ws::{Message, WebSocket};
use tokio::sync::broadcast;

async fn ws_handler(ws: WebSocketUpgrade, state: AppState) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let mut rx = state.snapshot_tx.subscribe();
    loop {
        tokio::select! {
            Ok(snap) = rx.recv() => {
                let compressed = zstd::encode_all(&snap[..], 3).unwrap();
                if socket.send(Message::Binary(compressed)).await.is_err() { break; }
            }
            Some(msg) = socket.recv() => {
                if let Ok(Message::Text(t)) = msg {
                    let cmd: ControlMessage = serde_json::from_str(&t).unwrap();
                    state.params_tx.send(cmd).ok();
                }
            }
        }
    }
}
```

Control messages:

```typescript
type ControlMessage =
  | { type: 'set_heat_flux'; value: number }
  | { type: 'set_material'; value: 'steel_304' | 'copper' | 'aluminum' }
  | { type: 'set_carrot_size'; diameter_mm: number; length_mm: number }
  | { type: 'pause' | 'resume' | 'reset' }
  | { type: 'request_full_snapshot' };
```

### 6.4 React + R3F Front-End

```typescript
// web/src/components/BoilingScene.tsx
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import { useSnapshot } from '../hooks/useSnapshot';

function WaterVolume({ temperature, alpha, meta }: Props) {
    const ref = useRef<THREE.Mesh>(null);
    const texture = useMemo(() => {
        return new THREE.Data3DTexture(
            new Uint8Array(temperature.length * 4),
            meta.nx, meta.ny, meta.nz
        );
    }, [meta]);

    useEffect(() => {
        // Upload temperature + alpha as RGBA volume texture
        for (let i = 0; i < temperature.length; i++) {
            texture.image.data[i*4+0] = alpha[i] * 255;
            texture.image.data[i*4+1] = ((temperature[i] - 20) / 80) * 255;
        }
        texture.needsUpdate = true;
    }, [temperature, alpha]);

    return (
        <mesh ref={ref}>
            <boxGeometry args={[meta.nx * meta.dx, meta.ny * meta.dx, meta.nz * meta.dx]} />
            <shaderMaterial
                vertexShader={volumeVert}
                fragmentShader={volumeFrag}
                uniforms={{ uVolume: { value: texture } }}
                transparent
            />
        </mesh>
    );
}

function Bubbles({ bubbles }: { bubbles: BubbleState[] }) {
    return (
        <instancedMesh args={[undefined, undefined, bubbles.length]}>
            <sphereGeometry args={[1, 8, 8]} />
            <meshPhysicalMaterial roughness={0} transmission={0.9} />
            {bubbles.map((b, i) => (
                <BubbleInstance key={i} position={b.position} radius={b.radius} index={i} />
            ))}
        </instancedMesh>
    );
}

export function BoilingScene() {
    const { snapshot, sendCommand } = useSnapshot('ws://localhost:8080/stream');
    if (!snapshot) return <div>Connecting...</div>;
    return (
        <Canvas camera={{ position: [0.3, 0.3, 0.2] }}>
            <ambientLight intensity={0.3} />
            <directionalLight position={[1, 1, 1]} />
            <Pot />  {/* loaded from GLB */}
            <WaterVolume {...snapshot} />
            <Bubbles bubbles={snapshot.bubbles} />
            <CarrotMesh retention={snapshot.carrot_retention} vertexC={snapshot.carrot_surface_C} />
            <OrbitControls />
        </Canvas>
    );
}
```

### 6.5 Control Panel and Plots

Use Recharts for the time series:

- Water mean temperature vs time
- Wall heat flux vs time
- Carrot retention vs time
- Bubble count vs time

Sliders for heat flux, material dropdown, carrot size inputs, all sending ControlMessage over the websocket.

### 6.6 Share-Link Mechanism

Serialize current parameters and camera pose into URL query string. Anyone opening the link sees the same view:

```
http://host/?hf=1200&mat=copper&cd=25&cl=50&t=300&cx=0.3&cy=0.3&cz=0.2&cfx=0&cfy=0&cfz=0.06
```

### 6.7 Phase 6 Acceptance

- Dashboard displays a live simulation at 30+ FPS.
- User can change heat flux, material, and carrot size without restarting.
- A share link reconstructs the exact view on another machine.
- Deployable as a single `docker-compose up` with three services (Python solver, Rust WS server, static front end).

### 6.8 Deployment Notes for Lambda Vector (WSL2 + Windows)

The solver runs in WSL with GPU access, while the browser runs natively on Windows. WSL2 exposes any port bound to `0.0.0.0` as `localhost` on Windows, so `http://localhost:3000` in Edge or Chrome hits the Vite dev server inside WSL with no extra config. For remote access from other machines on the local network:

```powershell
# One-time, on the Windows host as Administrator
# Forward Windows port 8080 to WSL's port 8080
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$(wsl hostname -I | ForEach-Object { $_.Trim() })
New-NetFirewallRule -DisplayName "Boiling Sim WS" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
```

For production deployment to a shared URL, publish Docker images built inside WSL and push them to a cloud instance. Nothing in the dashboard is Windows-specific.

---

## Phase 7: Optional Omniverse Kit Migration

**Duration:** 3-4 weeks. **Goal:** production-grade rendering for external demos.

Unlike the earlier phases, Omniverse Kit is well supported natively on Windows and the Lambda Vector is an ideal host for it. Run the Kit application directly on Windows (not in WSL) and bridge to the solver through the WebSocket or MQTT layer. The RTX 6000 Ada's 48 GB of VRAM and pro driver are exactly what Kit expects.

### 7.1 Kit App Template

Clone the NVIDIA `kit-app-template` repo. Start from the `usd_viewer` template. Configure it to load your scene's USD file and subscribe to live updates.

### 7.2 Live Sync via MQTT

For high-frequency field updates, publish binary payloads to an MQTT topic from the Rust WS server, and have a Kit extension subscribe and update USD prims:

```python
# Omniverse Kit extension
import omni.ext
import paho.mqtt.client as mqtt

class BoilingSimBridge(omni.ext.IExt):
    def on_startup(self):
        self.client = mqtt.Client()
        self.client.on_message = self.on_update
        self.client.connect("localhost", 1883)
        self.client.subscribe("boiling/snapshot")
        self.client.loop_start()

    def on_update(self, client, userdata, msg):
        snapshot = parse_snapshot(msg.payload)
        stage = omni.usd.get_context().get_stage()
        for bubble in snapshot.bubbles:
            prim = stage.GetPrimAtPath(f"/World/Bubbles/b{bubble.id}")
            prim.GetAttribute("xformOp:translate").Set(bubble.position)
```

### 7.3 WebRTC Streaming

Follow the `kit-app-streaming` reference: package your Kit app in a container, expose the WebRTC signaling ports, and host a thin HTML viewer using `@nvidia/omniverse-webrtc-streaming-library`. Same application runs locally or on a cloud GPU instance.

### 7.4 Phase 7 Acceptance

- Path-traced water and realistic caustics in the Omniverse render.
- Live simulation updates visible in the Kit app.
- Browser viewer reaches the Kit app over WebRTC on a public URL.

---

## Appendix: Material Properties and Constants

Store in `data/materials.json`:

```json
{
  "water": {
    "rho_ref":        997.0,
    "c_p":            4186.0,
    "k":              0.606,
    "mu":             0.00089,
    "sigma":          0.0589,
    "beta":           0.000207,
    "T_sat":          373.15,
    "h_lv":           2.257e6,
    "rho_vapor":      0.598
  },
  "steel_304": {
    "rho":            8000.0,
    "c_p":            500.0,
    "k":              16.2
  },
  "cast_iron": {
    "rho":            7200.0,
    "c_p":            460.0,
    "k":              55.0
  },
  "aluminum": {
    "rho":            2700.0,
    "c_p":            900.0,
    "k":              237.0
  },
  "copper": {
    "rho":            8960.0,
    "c_p":            385.0,
    "k":              401.0
  },
  "carrot": {
    "rho":            1040.0,
    "c_p":            3900.0,
    "k":              0.605,
    "D_beta_carotene": 2.0e-10,
    "K_partition":    0.3,
    "initial_C_mg_per_kg": 83.0,
    "Ea_J_per_mol":   70000.0,
    "k0_per_s":       2.63e6
  },
  "constants": {
    "R_gas":          8.314,
    "g":              9.81
  }
}
```

### Unit Conventions

- Length: meters
- Mass: kilograms
- Time: seconds
- Temperature: Kelvin internally, Celsius for user-facing I/O
- Energy: Joules
- Concentration: kg/m³ internally, mg/kg or mg/100g for user-facing I/O

### Reference Simulation Parameters

The Lambda Vector's 48 GB of VRAM supports two resolution tiers. Use the development tier for iteration and debugging, then switch to the production tier for validation runs and published results.

| Parameter | Development | Production | Notes |
|-----------|-------------|------------|-------|
| dx | 1.0 mm | 0.5 mm | halve for convergence study |
| dt | 2 ms | 0.5 ms | CFL-limited; may need 0.2 ms with vigorous boiling |
| Total sim time | 600 s | 600 s | 10 minutes of boil |
| Output cadence | 100 ms | 50 ms | 6,000 to 12,000 frames |
| Grid cells | ~5 × 10⁶ | ~3.8 × 10⁷ | production uses 8x more cells |
| Carrot tets | ~40,000 | ~200,000 | linear elements |
| MAX_BUBBLES | 100,000 | 1,000,000 | Lagrangian particle pool |
| Peak VRAM | ~6 GB | ~35 GB | leaves 12 GB free for checkpoints |
| Wall-time/sim-second | < 2 s | < 8 s | full coupled pipeline on RTX 6000 Ada |

For overnight parameter sweeps, use the development tier and run 20 scenarios in parallel through a simple Python orchestrator (the 64 Threadripper threads handle the dispatch, one GPU process at a time).

---

## Closing Notes for the Developer

Do these three things, in this order, on day one:

1. Run Phase 0, all the way through, even the Rust CUDA smoke test. The toolchain is the foundation and every later hour you spend fighting it is an hour not spent on physics. Pay special attention to the WSL2 driver rule: Windows driver only, CUDA toolkit only inside WSL.
2. Write every kernel with a scalar CPU reference first, even a slow one. Use it as the oracle for your unit tests. Once the CUDA version matches the reference, you are free to optimize.
3. Commit the validation plots from each phase to the repo before starting the next phase. You want a flipbook of scientific evidence that the simulation works, not just a claim that it does.

A fourth habit worth forming on the Lambda Vector specifically: **develop at 1 mm, validate at 0.5 mm**. The development tier runs in seconds per step, which is fast enough for the iterative kernel debugging and physics exploration that fills most of the day. Only the final validation runs, the figures for papers, and the numbers that go in the retention table need the production tier. Conflating the two will burn VRAM, time, and patience without adding scientific value.

The project is ambitious but decomposable. Each phase is a demonstrable milestone. Build the foundation slowly, validate at every step, and the final dashboard will stand on solid ground.

---

*End of Developer Guide*
