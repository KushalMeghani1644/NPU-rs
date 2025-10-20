use ndarray::ArrayD;
use crate::compute::{MatMulUnit, ConvUnit};
use crate::device::NpuDevice;
use crate::error::Result;
use crate::perf_monitor::PerformanceMonitor;
use std::sync::Arc;

/// Execution context for NPU operations.
pub struct ExecutionContext {
    device: Arc<NpuDevice>,
    matmul_unit: MatMulUnit,
    conv_unit: ConvUnit,
    perf_monitor: Arc<PerformanceMonitor>,
}

impl ExecutionContext {
    /// Create a new execution context.
    pub fn new(device: Arc<NpuDevice>) -> Self {
        let info = device.get_info();
        Self {
            device: device.clone(),
            matmul_unit: MatMulUnit::new(info.peak_throughput_tops),
            conv_unit: ConvUnit::new(info.peak_throughput_tops),
            perf_monitor: device.get_perf_monitor(),
        }
    }

    /// Execute matrix multiplication operation.
    pub fn execute_matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if !self.device.is_ready() {
            return Err(crate::error::NpuError::DeviceError(
                "Device not ready".to_string(),
            ));
        }

        let result = self.matmul_unit.gemm(a, b)?;

        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        let ops = (2 * m * k * n) as u64;

        self.perf_monitor.record_operation(ops);

        Ok(result)
    }

    /// Execute batched matrix multiplication.
    pub fn execute_batched_matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if !self.device.is_ready() {
            return Err(crate::error::NpuError::DeviceError(
                "Device not ready".to_string(),
            ));
        }

        let result = self.matmul_unit.batched_gemm(a, b)?;

        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];
        let ops = (2 * batch * m * k * n) as u64;

        self.perf_monitor.record_operation(ops);

        Ok(result)
    }

    /// Execute 1x1 convolution.
    pub fn execute_conv1x1(
        &self,
        input: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        if !self.device.is_ready() {
            return Err(crate::error::NpuError::DeviceError(
                "Device not ready".to_string(),
            ));
        }

        let result = self.conv_unit.conv1x1(input, kernel)?;

        let batch = input.shape()[0];
        let height = input.shape()[1];
        let width = input.shape()[2];
        let c_in = input.shape()[3];
        let c_out = kernel.shape()[3];
        let ops = (2 * batch * height * width * c_in * c_out) as u64;

        self.perf_monitor.record_operation(ops);

        Ok(result)
    }

    /// Get current throughput in GOPS.
    pub fn get_current_throughput_gops(&self) -> f64 {
        self.perf_monitor.get_throughput_gops()
    }

    /// Get performance metrics.
    pub fn get_metrics(&self) -> crate::perf_monitor::PerformanceMetrics {
        self.perf_monitor.get_metrics()
    }

    /// Get underlying device.
    pub fn get_device(&self) -> Arc<NpuDevice> {
        self.device.clone()
    }
}

/// Batch execution scheduler for efficient workload distribution.
pub struct BatchScheduler {
    context: ExecutionContext,
    batch_size: usize,
}

impl BatchScheduler {
    /// Create a new batch scheduler.
    pub fn new(device: Arc<NpuDevice>, batch_size: usize) -> Self {
        Self {
            context: ExecutionContext::new(device),
            batch_size,
        }
    }

    /// Submit a batch of operations.
    pub fn submit_batch(&self, operations: Vec<(&ArrayD<f32>, &ArrayD<f32>)>) -> Result<Vec<ArrayD<f32>>> {
        let mut results = Vec::new();

        for (a, b) in operations {
            let result = self.context.execute_matmul(a, b)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get the execution context.
    pub fn get_context(&self) -> &ExecutionContext {
        &self.context
    }

    /// Get batch size.
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}
