use crate::error::{NpuError, Result};
use ndarray::ArrayD;

/// Quantization statistics for calibration.
#[derive(Debug, Clone)]
pub struct QuantStats {
    pub min_val: f32,
    pub max_val: f32,
    pub mean_val: f32,
    pub std_val: f32,
}

impl QuantStats {
    /// Compute statistics from a tensor.
    pub fn from_tensor(data: &ArrayD<f32>) -> Self {
        let values: Vec<f32> = data.iter().cloned().collect();
        
        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let mean_val = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean_val).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std_val = variance.sqrt();

        Self {
            min_val,
            max_val,
            mean_val,
            std_val,
        }
    }

    /// Get scale factor for quantization.
    pub fn get_scale(&self, num_bits: u32) -> f32 {
        let levels = (1u64 << num_bits) as f32;
        self.max_val / (levels - 1.0)
    }

    /// Get zero point.
    pub fn get_zero_point(&self, num_bits: u32, signed: bool) -> i32 {
        let levels = (1u64 << num_bits) as f32;
        if signed {
            (-(levels / 2.0)) as i32
        } else {
            0
        }
    }
}

/// Quantization converter.
pub struct QuantConverter {
    scale: f32,
    zero_point: i32,
    num_bits: u32,
}

impl QuantConverter {
    /// Create a new quantization converter.
    pub fn new(stats: &QuantStats, num_bits: u32, signed: bool) -> Self {
        Self {
            scale: stats.get_scale(num_bits),
            zero_point: stats.get_zero_point(num_bits, signed),
            num_bits,
        }
    }

    /// Quantize float32 to integer.
    pub fn quantize(&self, value: f32) -> i32 {
        let quantized = (value / self.scale) as i32 + self.zero_point;
        let max_val = (1i64 << self.num_bits) as i32 - 1;
        let min_val = -(1i64 << (self.num_bits - 1)) as i32;
        quantized.max(min_val).min(max_val)
    }

    /// Dequantize integer to float32.
    pub fn dequantize(&self, value: i32) -> f32 {
        ((value - self.zero_point) as f32) * self.scale
    }

    /// Quantize entire tensor.
    pub fn quantize_tensor(&self, tensor: &ArrayD<f32>) -> Result<Vec<i32>> {
        Ok(tensor.iter().map(|&v| self.quantize(v)).collect())
    }

    /// Dequantize tensor.
    pub fn dequantize_tensor(&self, quantized: &[i32]) -> Result<ArrayD<f32>> {
        let values: Vec<f32> = quantized.iter().map(|&v| self.dequantize(v)).collect();
        Ok(ArrayD::from_shape_vec(
            ndarray::IxDyn(&[quantized.len()]),
            values,
        ).map_err(|_| NpuError::InvalidShape("Failed to reshape".to_string()))?)
    }
}

/// Post-training quantization engine.
pub struct PTQEngine {
    num_bits: u32,
    signed: bool,
}

impl PTQEngine {
    /// Create a new PTQ engine.
    pub fn new(num_bits: u32, signed: bool) -> Self {
        Self { num_bits, signed }
    }

    /// Calibrate on sample data.
    pub fn calibrate(&self, sample_data: &[ArrayD<f32>]) -> Result<QuantConverter> {
        if sample_data.is_empty() {
            return Err(NpuError::InvalidConfiguration(
                "No calibration data provided".to_string(),
            ));
        }

        let mut all_values = Vec::new();
        for tensor in sample_data {
            all_values.extend(tensor.iter().cloned());
        }

        let combined = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[all_values.len()]),
            all_values,
        ).map_err(|_| NpuError::InvalidShape("Failed to calibrate".to_string()))?;

        let stats = QuantStats::from_tensor(&combined);
        Ok(QuantConverter::new(&stats, self.num_bits, self.signed))
    }
}
