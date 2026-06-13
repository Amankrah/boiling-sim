# ============================================================================
# Phase 5 Bootstrap (Windows): Rust toolchain + maturin editable install
# ============================================================================
# Run AFTER scripts/setup_windows_env.ps1 has finished (driver, MSVC, CUDA
# Toolkit, uv, Python 3.11 venv). Picks up where Phase 0 left off and gets
# the Rust extension (`sim_core`) building into the venv.
#
# Usage (from project root, in PowerShell):
#   .\scripts\bootstrap.ps1
#
# What this does:
#   1. Installs rustup if missing.
#   2. The rust-toolchain.toml at the repo root pins channel 1.85; rustup
#      auto-installs that channel + clippy + rustfmt on first cargo call.
#   3. Verifies CUDA_PATH is set (cuda-kernels/build.rs requires it).
#   4. Verifies nvcuda.dll is reachable on PATH so `cargo test` from
#      crates/cuda-kernels works.
#   5. Runs `uv pip install -e .` which invokes maturin to build sim_core
#      and drop the .pyd into the editable install.
# ============================================================================

$ErrorActionPreference = "Stop"

function Write-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

Push-Location $PSScriptRoot\..

# ---- 1. Rustup ----
Write-Info "Checking rustup..."
$rustup = Get-Command rustup -ErrorAction SilentlyContinue
if (-not $rustup) {
    Write-Info "Installing rustup (will auto-install Rust 1.85 from rust-toolchain.toml)..."
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "$env:TEMP\rustup-init.exe"
    & "$env:TEMP\rustup-init.exe" -y --default-toolchain none --profile minimal
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    Remove-Item "$env:TEMP\rustup-init.exe" -ErrorAction SilentlyContinue
} else {
    Write-Info "rustup: $(rustup --version)"
}

# Warm the toolchain so the first `cargo build` doesn't pause for a sync.
Write-Info "Warming Rust toolchain (rust-toolchain.toml pins 1.85)..."
cargo --version | Out-Host

# ---- 2. CUDA_PATH guard ----
Write-Info "Verifying CUDA_PATH..."
if (-not $env:CUDA_PATH) {
    $machineCudaPath = [System.Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
    if ($machineCudaPath) {
        $env:CUDA_PATH = $machineCudaPath
        Write-Info "Picked up CUDA_PATH from machine env: $env:CUDA_PATH"
    } else {
        Write-Warn "CUDA_PATH is not set. crates/cuda-kernels/build.rs will fail."
        Write-Warn "Run scripts/setup_windows_env.ps1 first, or:"
        Write-Warn "  setx CUDA_PATH `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`""
        Write-Warn "Then close and reopen PowerShell."
        Write-Err   "Aborting bootstrap."
    }
}
Write-Info "CUDA_PATH: $env:CUDA_PATH"

# ---- 3. nvcuda.dll on PATH ----
# cudarc loads nvcuda.dll dynamically at runtime; without it on PATH, the
# `cargo test` step in crates/cuda-kernels fails with a misleading
# "DLL not found" error.
Write-Info "Verifying nvcuda.dll is reachable..."
$nvcuda = (Get-Command nvcuda.dll -ErrorAction SilentlyContinue)
if (-not $nvcuda) {
    # Driver puts nvcuda.dll in System32 by default; usually fine, but flag
    # if PATH lookup misses it so the failure mode at cargo-test time is
    # diagnosed here instead of inside cudarc.
    $sys32 = Join-Path $env:SystemRoot "System32\nvcuda.dll"
    if (Test-Path $sys32) {
        Write-Info "nvcuda.dll found at $sys32 (loaded via System32 directly; ignore PATH miss)."
    } else {
        Write-Warn "nvcuda.dll not on PATH and not in System32 -- driver may be broken."
    }
}

# ---- 4. venv check ----
if (-not (Test-Path .venv)) {
    Write-Err "No .venv found. Run scripts/setup_windows_env.ps1 first."
}

# ---- 5. Editable install via maturin ----
Write-Info "Installing boilingsim (editable) -- maturin will build sim_core..."
uv pip install -e .

# ---- 6. Smoke verify ----
Write-Info "Smoke-verifying sim_core import..."
& .\.venv\Scripts\python.exe -c @"
from sim_core.props import MaterialProps
mp = MaterialProps.from_json('data/materials.json')
assert mp.beta_100c == 7.5e-4, f'beta_100c got {mp.beta_100c}'
print('sim_core.props OK:', mp)
"@
if ($LASTEXITCODE -ne 0) { Write-Err "sim_core import smoke test failed." }

Pop-Location

Write-Info "========================================"
Write-Info "  Phase 5 bootstrap complete."
Write-Info "  Run pytest python\tests\ to verify the full suite."
Write-Info "========================================"
