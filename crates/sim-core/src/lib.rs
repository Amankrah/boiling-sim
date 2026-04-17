use pyo3::prelude::*;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[pyclass]
pub struct SimCore {
    _device: Arc<CudaDevice>,
    device_id: usize,
}

#[pymethods]
impl SimCore {
    #[new]
    fn new(device_id: usize) -> PyResult<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { _device: device, device_id })
    }

    fn device_info(&self) -> PyResult<String> {
        Ok(format!("CUDA device {} ready", self.device_id))
    }
}

#[pymodule]
fn sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimCore>()?;
    Ok(())
}
