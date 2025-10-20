use ndarray::{ArrayD, IxDyn, Array2};
use crate::error::{NpuError, Result};

/// Matrix multiplication on NPU hardware.
pub struct MatMulUnit {
    peak_throughput_tops: f32,
}

impl MatMulUnit {
    /// Create a new matrix multiplication unit.
    pub fn new(peak_throughput_tops: f32) -> Self {
        Self {
            peak_throughput_tops,
        }
    }

    /// Perform matrix multiplication between two 2D tensors.
    /// Shape: (M, K) @ (K, N) -> (M, N)
    pub fn gemm(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(NpuError::InvalidShape(
                "MatMul requires 2D tensors".to_string(),
            ));
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape[1] != b_shape[0] {
            return Err(NpuError::InvalidShape(
                format!("Dimension mismatch: {} != {}", a_shape[1], b_shape[0]),
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_2d = a.view().into_shape((m, k)).map_err(|_| {
            NpuError::ComputationError("Failed to reshape A".to_string())
        })?;
        let b_2d = b.view().into_shape((k, n)).map_err(|_| {
            NpuError::ComputationError("Failed to reshape B".to_string())
        })?;

        let mut result = Array2::zeros((m, n));
        ndarray::linalg::general_mat_mul(1.0, &a_2d, &b_2d, 0.0, &mut result);

        Ok(result.into_dyn())
    }

    /// Perform batched matrix multiplication.
    /// Shape: (B, M, K) @ (B, K, N) -> (B, M, N)
    pub fn batched_gemm(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if a.ndim() != 3 || b.ndim() != 3 {
            return Err(NpuError::InvalidShape(
                "Batched MatMul requires 3D tensors".to_string(),
            ));
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1] {
            return Err(NpuError::InvalidShape(
                "Batch size or dimension mismatch".to_string(),
            ));
        }

        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        let mut results = Vec::new();

        for b_idx in 0..batch_size {
            let a_slice_2d = a.slice(ndarray::s![b_idx, .., ..]).to_owned();
            let b_slice_2d = b.slice(ndarray::s![b_idx, .., ..]).to_owned();

            let a_slice = a_slice_2d.into_shape((m, k)).map_err(|_| {
                NpuError::ComputationError("Failed to reshape A".to_string())
            })?;
            let b_slice = b_slice_2d.into_shape((k, n)).map_err(|_| {
                NpuError::ComputationError("Failed to reshape B".to_string())
            })?;

            let mut result = Array2::zeros((m, n));
            ndarray::linalg::general_mat_mul(1.0, &a_slice, &b_slice, 0.0, &mut result);
            results.push(result.into_dyn());
        }

        let mut final_result = ArrayD::zeros(IxDyn(&[batch_size, m, n]));
        for (b_idx, result) in results.iter().enumerate() {
            final_result.slice_mut(ndarray::s![b_idx, .., ..]).assign(result);
        }

        Ok(final_result)
    }

    /// Get estimated TOPS for a given matrix multiplication.
    pub fn estimate_tops(&self, m: usize, k: usize, n: usize) -> f32 {
        let ops = (2 * m * k * n) as f32 / 1e12;
        ops * 1e-3
    }
}

/// Convolution operations on NPU.
pub struct ConvUnit {
    peak_throughput_tops: f32,
}

impl ConvUnit {
    /// Create a new convolution unit.
    pub fn new(peak_throughput_tops: f32) -> Self {
        Self {
            peak_throughput_tops,
        }
    }

    /// 1x1 convolution (typically the bottleneck).
    /// Shape: (N, H, W, C_in) * (1, 1, C_in, C_out) -> (N, H, W, C_out)
    pub fn conv1x1(
        &self,
        input: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        if input.ndim() != 4 || kernel.ndim() != 4 {
            return Err(NpuError::InvalidShape(
                "Conv1x1 requires 4D tensors".to_string(),
            ));
        }

        let input_shape = input.shape();
        let kernel_shape = kernel.shape();

        if input_shape[3] != kernel_shape[2] {
            return Err(NpuError::InvalidShape(
                format!("Channel mismatch: {} != {}", input_shape[3], kernel_shape[2]),
            ));
        }

        let batch = input_shape[0];
        let height = input_shape[1];
        let width = input_shape[2];
        let c_out = kernel_shape[3];

        let mut output = ArrayD::zeros(IxDyn(&[batch, height, width, c_out]));

        for n in 0..batch {
            for h in 0..height {
                for w in 0..width {
                    let pixel = input.slice(ndarray::s![n, h, w, ..]).to_owned();
                    
                    for c_out_idx in 0..c_out {
                        let filter = kernel.slice(ndarray::s![0, 0, .., c_out_idx]).to_owned();
                        let dot_product: f32 = pixel.iter().zip(filter.iter()).map(|(a, b)| a * b).sum();
                        output[[n, h, w, c_out_idx]] = dot_product;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Estimate TOPS for conv operation.
    pub fn estimate_tops(&self, batch: usize, height: usize, width: usize, c_in: usize, c_out: usize) -> f32 {
        let ops = (2 * batch * height * width * c_in * c_out) as f32 / 1e12;
        ops * 1e-3
    }
}
