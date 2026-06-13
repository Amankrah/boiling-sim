fn cuda_lib_dir_unix() -> String {
    if let Ok(root) = std::env::var("CUDA_PATH") {
        return format!("{root}/lib64");
    }
    for dir in ["/usr/local/cuda/lib64", "/usr/lib/cuda/lib64"] {
        if std::path::Path::new(dir).is_dir() {
            return dir.to_string();
        }
    }
    panic!(
        "Could not find CUDA libraries (tried CUDA_PATH, /usr/local/cuda/lib64, /usr/lib/cuda/lib64).\n\
         Install nvidia-cuda-toolkit or set CUDA_PATH to the toolkit root."
    );
}

/// On Windows, MSVC < VS 2019 16.11 (`_MSC_VER < 1929`) silently miscompiles
/// device code with CUDA 12.x. The `cc` crate picks `cl.exe` via its MSVC
/// detector; the version it picks is whatever `vcvars64.bat` set up. We fail
/// fast here with a clear message instead of letting an incompatible MSVC
/// limp into kernel compilation and emit subtly broken code.
#[cfg(target_os = "windows")]
fn assert_msvc_recent_enough() {
    use std::process::Command;
    let tool = cc::Build::new()
        .target(&std::env::var("TARGET").unwrap_or_else(|_| "x86_64-pc-windows-msvc".to_string()))
        .host(&std::env::var("HOST").unwrap_or_else(|_| "x86_64-pc-windows-msvc".to_string()))
        .opt_level(0)
        .get_compiler();
    let cl_path = tool.path();

    // `cl.exe` prints its banner to stderr, including a version like
    // "Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31933 for x64".
    // The first dotted number is the _MSC_VER major.minor; e.g. 19.34 ->
    // _MSC_VER 1934. We pull that out and gate >= 1929 (VS 2019 16.11).
    let output = match Command::new(cl_path).output() {
        Ok(o) => o,
        Err(e) => {
            println!(
                "cargo:warning=could not run cl.exe ({:?}); skipping MSVC version check: {}",
                cl_path, e
            );
            return;
        }
    };
    let banner = String::from_utf8_lossy(&output.stderr);
    let needle = "Version ";
    if let Some(start) = banner.find(needle) {
        let tail = &banner[start + needle.len()..];
        let mut parts = tail.split('.');
        if let (Some(maj), Some(min)) = (parts.next(), parts.next()) {
            if let (Ok(maj), Ok(min)) = (maj.parse::<u32>(), min.parse::<u32>()) {
                let ms_ver = maj * 100 + min;
                println!("cargo:warning=detected MSVC _MSC_VER ~ {ms_ver}");
                if ms_ver < 1929 {
                    panic!(
                        "MSVC _MSC_VER {ms_ver} is too old for CUDA 12.x device code \
                         (need >= 1929, i.e. VS 2019 16.11 or VS 2022 17.x). \
                         Reinstall MSVC Build Tools or open a Developer Command Prompt for VS 2022."
                    );
                }
                return;
            }
        }
    }
    println!(
        "cargo:warning=could not parse cl.exe version banner; skipping MSVC version check. \
         Banner was: {}",
        banner.trim()
    );
}

#[cfg(not(target_os = "windows"))]
fn assert_msvc_recent_enough() {}

fn main() {
    assert_msvc_recent_enough();

    // GPU arch: hardcoded to sm_89 (Ada Lovelace, RTX 6000 Ada). Override via
    // BOILINGSIM_GPU_ARCH=compute_XX,sm_XX for Hopper / Blackwell / older
    // arches. Per-build, not pinned — anyone building for a different GPU
    // can opt in without editing this file.
    let arch = std::env::var("BOILINGSIM_GPU_ARCH")
        .unwrap_or_else(|_| "compute_89,code=sm_89".to_string());
    println!("cargo:rerun-if-env-changed=BOILINGSIM_GPU_ARCH");
    println!("cargo:rustc-env=BOILINGSIM_GPU_ARCH={arch}");

    // FMA policy:
    // * Default (release/perf): FMA enabled (nvcc default). The Phase 5 M3
    //   measurements showed --fmad=false costs ~5% on the 7-point stencil,
    //   and the M2 parity gate per dev-guide §5.4 allows max_rel_diff < 1e-4,
    //   which FMA-on satisfies comfortably (single FMA per cell, ~1 ULP).
    // * Bit-exact debugging: set BOILINGSIM_FMAD=false to reproduce the
    //   pre-M3 bit-exact 1-step parity (useful for hunting arithmetic
    //   regressions where exact equality is the easiest invariant to
    //   reason about). Default off so production builds get full speed.
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag(&format!("arch={arch}"))
        .file("src/vector_add.cu")
        .file("src/scale.cu")
        .file("src/jacobi_pressure.cu")
        .file("src/scatter_latent_heat.cu")
        .file("src/scatter_momentum.cu")
        .file("src/reduce_water_alpha.cu")
        .file("src/update_bubbles.cu")
        .file("src/laplacian_spmv.cu")
        .file("src/diag_inverse_apply.cu")
        .file("src/dot_reduce.cu")
        .file("src/axpy_device.cu")
        .file("src/pressure_solve_pcg.cu");
    println!("cargo:rerun-if-env-changed=BOILINGSIM_FMAD");
    if std::env::var("BOILINGSIM_FMAD").as_deref() == Ok("false") {
        println!("cargo:warning=BOILINGSIM_FMAD=false -- compiling with --fmad=false for bit-exact parity");
        build.flag("--fmad=false");
    }
    // Phase 5.5 Lever 2: surface ptxas register / shared-mem report so we can
    // see whether __launch_bounds__ is actually hitting the 2-blocks-per-SM
    // occupancy target. Triggered by BOILINGSIM_PTXAS_VERBOSE=1.
    println!("cargo:rerun-if-env-changed=BOILINGSIM_PTXAS_VERBOSE");
    if std::env::var("BOILINGSIM_PTXAS_VERBOSE").as_deref() == Ok("1") {
        build.flag("--ptxas-options=-v");
    }
    build.compile("boilingsim_kernels");

    let cuda_lib_dir = if cfg!(target_os = "windows") {
        let cuda_path = std::env::var("CUDA_PATH")
            .expect("CUDA_PATH env var not set — install CUDA Toolkit and restart your shell");
        format!("{cuda_path}\\lib\\x64")
    } else {
        cuda_lib_dir_unix()
    };

    println!("cargo:rustc-link-search=native={cuda_lib_dir}");
    println!("cargo:rustc-link-lib=cudart");
}
