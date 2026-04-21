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

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_89,code=sm_89")
        .file("src/vector_add.cu")
        .compile("vector_add");

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
