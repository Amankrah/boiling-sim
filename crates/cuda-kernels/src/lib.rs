//! CUDA kernel bindings for the boiling simulation.
//!
//! This crate compiles hand-written .cu kernels via nvcc (see build.rs)
//! and provides safe Rust wrappers that launch them through cudarc.
//!
//! Phase 0 scope: verify the nvcc -> cc crate -> cudarc pipeline works.
//! Phase 5 scope: real kernel launches and bandwidth benchmarks.

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

extern "C" {
    #[allow(dead_code)]
    fn vector_add(a: *const f32, b: *const f32, c: *mut f32, n: i32);
}

/// Scaffold: allocate host & device buffers for vector_add and round-trip them.
///
/// Returns the elapsed time in seconds. The actual kernel launch is deferred
/// to Phase 5; this function verifies that cudarc device memory ops work.
pub fn run_vector_add_benchmark(device: &Arc<CudaDevice>, n: usize) -> Result<f64> {
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let _a_dev = device.htod_sync_copy(&a_host)?;
    let _b_dev = device.htod_sync_copy(&b_host)?;
    let c_dev: CudaSlice<f32> = device.alloc_zeros(n)?;

    device.synchronize()?;
    let start = std::time::Instant::now();
    // TODO(phase5): launch kernel via cudarc module API.
    device.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();

    let _c_host = device.dtoh_sync_copy(&c_dev)?;
    Ok(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_device_available() {
        let _device = CudaDevice::new(0).expect("No CUDA device found");
    }

    #[test]
    fn test_vector_add_compiles() {
        // Build succeeding means nvcc compiled vector_add.cu via build.rs.
    }

    #[test]
    fn test_round_trip_device_memory() {
        let device = CudaDevice::new(0).expect("No CUDA device found");
        let elapsed = run_vector_add_benchmark(&device, 1_000_000).unwrap();
        assert!(elapsed < 5.0, "Round trip took too long: {elapsed}s");
    }
}
