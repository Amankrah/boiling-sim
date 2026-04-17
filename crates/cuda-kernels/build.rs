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
        "/usr/local/cuda/lib64".to_string()
    };

    println!("cargo:rustc-link-search=native={cuda_lib_dir}");
    println!("cargo:rustc-link-lib=cudart");
}
