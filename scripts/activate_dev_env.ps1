# ============================================================================
# Dev Environment Activation (run once per PowerShell session)
# ============================================================================
# Activates MSVC + CUDA + Python venv so 'cargo build' and 'pytest' work.
#
# Usage:  . .\scripts\activate_dev_env.ps1
#         (note the leading dot + space - this DOT-SOURCES the script so
#          env changes persist in your current shell)
# ============================================================================

# ---- MSVC ----
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vsPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        "C:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
    )
    $vsRoot = $vsPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($vsRoot) {
        $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
        Write-Host "[dev-env] Importing MSVC from $vcvars" -ForegroundColor Green
        cmd /c "`"$vcvars`" && set" 2>&1 | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                Set-Item -Path "Env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
            }
        }
    }
}

# ---- CUDA ----
if (-not $env:CUDA_PATH) {
    $env:CUDA_PATH = [System.Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
}
$cudaBin = Join-Path $env:CUDA_PATH "bin"
if ($env:PATH -notlike "*$cudaBin*") { $env:PATH = "$cudaBin;$env:PATH" }

# ---- uv ----
$uvBin = "$env:USERPROFILE\.local\bin"
if ($env:PATH -notlike "*$uvBin*") { $env:PATH = "$uvBin;$env:PATH" }

# ---- Python venv ----
if (Test-Path "$PSScriptRoot\..\.venv\Scripts\Activate.ps1") {
    & "$PSScriptRoot\..\.venv\Scripts\Activate.ps1"
}

# ---- Summary ----
Write-Host "[dev-env] Ready:" -ForegroundColor Green
Write-Host "  cl.exe:  $((Get-Command cl.exe -ErrorAction SilentlyContinue).Source)"
Write-Host "  nvcc:    $((Get-Command nvcc -ErrorAction SilentlyContinue).Source)"
Write-Host "  python:  $((Get-Command python -ErrorAction SilentlyContinue).Source)"
Write-Host "  cargo:   $((Get-Command cargo -ErrorAction SilentlyContinue).Source)"
