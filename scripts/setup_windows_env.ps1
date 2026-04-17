# ============================================================================
# Phase 0: Windows-Native Environment Setup for boiling-sim
# ============================================================================
# Run from the project root in PowerShell:
#   cd C:\Users\Windows\Desktop\Dev_Projects\boiling-sim
#   .\scripts\setup_windows_env.ps1
#
# Prerequisites (install these BEFORE running this script):
#   1. NVIDIA driver 560+ (you have 595.97)           [MANUAL]
#   2. Visual Studio 2019/2022 Build Tools with MSVC  [MANUAL — you have 2019]
#   3. CUDA Toolkit 12.6                              [MANUAL — see below]
# ============================================================================

$ErrorActionPreference = "Stop"

function Write-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ---- Pre-flight: Driver check ----
Write-Info "Checking NVIDIA driver..."
$smi = nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
if (-not $smi) { Write-Err "nvidia-smi not found. Install the NVIDIA driver first." }
$driverMajor = [int]($smi.Trim().Split('.')[0])
Write-Info "Driver: $($smi.Trim())"
if ($driverMajor -lt 560) { Write-Err "Driver too old. Need >= 560. Update from nvidia.com/drivers." }

# ---- Pre-flight: MSVC check + env activation ----
Write-Info "Checking MSVC and activating dev environment..."
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
)
$vsRoot = $null
foreach ($vs in $vsPaths) {
    if (Test-Path $vs) { $vsRoot = $vs; break }
}
if (-not $vsRoot) {
    Write-Err "No Visual Studio Build Tools found. Install VS 2022 Build Tools with 'Desktop development with C++' workload."
}
Write-Info "Found Visual Studio at: $vsRoot"

# Check if cl.exe is already on PATH (dev shell already active)
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        Write-Err "vcvars64.bat not found at $vcvars. Reinstall VS Build Tools with C++ workload."
    }
    Write-Info "Importing MSVC environment from vcvars64.bat..."
    # Run vcvars64.bat in cmd and capture the resulting env vars into this PowerShell session
    $envDump = cmd /c "`"$vcvars`" && set" 2>&1
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
        }
    }
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $cl) { Write-Err "cl.exe still not on PATH after importing vcvars64.bat." }
}
Write-Info "cl.exe: $($cl.Source)"

# ---- Pre-flight: CUDA Toolkit check ----
Write-Info "Checking CUDA Toolkit..."
if (-not $env:CUDA_PATH) {
    # CUDA_PATH might not be in this shell's env yet (installer just ran).
    # Try reading it from the machine-level environment directly.
    $machineCudaPath = [System.Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
    if ($machineCudaPath) {
        $env:CUDA_PATH = $machineCudaPath
        Write-Info "Picked up CUDA_PATH from machine env: $env:CUDA_PATH"
    } else {
        # Last resort: derive from nvcc location if it's on PATH
        $nvccPath = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
        if ($nvccPath) {
            $derivedCuda = Split-Path (Split-Path $nvccPath -Parent) -Parent
            $env:CUDA_PATH = $derivedCuda
            Write-Info "Derived CUDA_PATH from nvcc: $env:CUDA_PATH"
        } else {
            Write-Warn "CUDA Toolkit not found. Install CUDA 12.6 from:"
            Write-Warn "  https://developer.nvidia.com/cuda-12-6-0-download-archive"
            Write-Err "Then close and reopen PowerShell and re-run this script."
        }
    }
}
Write-Info "CUDA_PATH: $env:CUDA_PATH"

# Ensure the CUDA bin is on PATH for this session
$cudaBin = Join-Path $env:CUDA_PATH "bin"
if ($env:PATH -notlike "*$cudaBin*") {
    $env:PATH = "$cudaBin;$env:PATH"
}

$nvcc = & nvcc --version 2>$null | Select-String "release"
if (-not $nvcc) { Write-Err "nvcc not found even after setting CUDA_PATH. Reinstall CUDA Toolkit or restart PowerShell." }
Write-Info "nvcc: $nvcc"

# ---- Pre-flight: Rust check ----
Write-Info "Checking Rust..."
$rustVer = rustc --version 2>$null
if (-not $rustVer) { Write-Err "Rust not found. Install from rustup.rs" }
Write-Info "Rust: $rustVer"

# ---- CUDA hello-world smoke test ----
Write-Info "Running CUDA hello-world smoke test..."
$tmpDir = [System.IO.Path]::GetTempPath()
$helloCu = Join-Path $tmpDir "hello_cuda.cu"
$helloExe = Join-Path $tmpDir "hello_cuda.exe"
@'
#include <cstdio>
__global__ void hi() { printf("GPU thread %d alive\n", threadIdx.x); }
int main() { hi<<<1, 4>>>(); cudaDeviceSynchronize(); return 0; }
'@ | Set-Content -Path $helloCu -Encoding ASCII

& nvcc $helloCu -o $helloExe
if ($LASTEXITCODE -ne 0) { Write-Err "nvcc failed to compile hello kernel" }
& $helloExe
if ($LASTEXITCODE -ne 0) { Write-Err "CUDA hello-world kernel failed at runtime" }
Remove-Item -Force $helloCu, $helloExe -ErrorAction SilentlyContinue
Write-Info "CUDA smoke test passed!"

# ---- Install uv ----
Write-Info "Checking uv..."
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Info "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Add to PATH for this session
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
} else {
    Write-Info "uv: $(uv --version)"
}

# ---- Python 3.11 venv ----
Push-Location $PSScriptRoot\..
if (-not (Test-Path .venv)) {
    Write-Info "Creating Python 3.11 venv with uv..."
    uv venv --python 3.11
}

Write-Info "Activating venv and installing Python dependencies..."
& .\.venv\Scripts\Activate.ps1
uv pip install `
    "warp-lang[examples]" `
    numpy scipy matplotlib h5py `
    pyvista trimesh pygmsh meshio `
    usd-core `
    fastapi "uvicorn[standard]" websockets `
    zstandard pyyaml pydantic `
    pytest pytest-benchmark

# ---- Cargo tools ----
# Note: cargo writes progress to stderr; temporarily relax error handling.
Write-Info "Installing cargo tools (cargo-watch, cargo-nextest)..."
$prevEA = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & cargo install cargo-watch cargo-nextest 2>&1 | Out-Host
} catch {
    Write-Warn "cargo-watch/cargo-nextest install had issues (non-fatal): $_"
}
$ErrorActionPreference = $prevEA

# ---- Node check (already have it) ----
$nodeVer = node --version 2>$null
Write-Info "Node: $nodeVer"
$pnpmVer = pnpm --version 2>$null
Write-Info "pnpm: $pnpmVer"

# ---- Init git ----
if (-not (Test-Path .git)) {
    Write-Info "Initializing git repository..."
    git init
}

Pop-Location

# ---- Summary ----
Write-Host ""
Write-Info "========================================"
Write-Info "  Phase 0 Environment Setup Complete!"
Write-Info "========================================"
Write-Host ""
Write-Host "  GPU:    NVIDIA RTX 6000 Ada Generation"
Write-Host "  Driver: $($smi.Trim())"
Write-Host "  CUDA:   $nvcc"
Write-Host "  Python: $(python --version 2>&1)"
Write-Host "  Rust:   $rustVer"
Write-Host "  Node:   $nodeVer"
Write-Host ""
Write-Info "Next steps:"
Write-Info "  1. .\.venv\Scripts\Activate.ps1"
Write-Info "  2. python -m warp.examples.core.example_sph"
Write-Info "  3. python -m warp.examples.fem.example_diffusion"
Write-Info "  4. cargo build --release"
Write-Info "  5. pytest python\tests\"
Write-Info "  6. Fill in benchmarks\baseline.md"
