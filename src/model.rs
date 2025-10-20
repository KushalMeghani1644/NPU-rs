use serde::{Serialize, Deserialize};
use crate::error::Result;

/// Quantization format for models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantFormat {
    Float32,
    Float16,
    Int8,
    Int4,
}

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    O1,
    O2,
    O3,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub quant_format: QuantFormat,
    pub optimization_level: OptimizationLevel,
    pub use_cache: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "default_model".to_string(),
            input_shape: vec![1, 224, 224, 3],
            output_shape: vec![1, 1000],
            quant_format: QuantFormat::Float32,
            optimization_level: OptimizationLevel::O2,
            use_cache: true,
        }
    }
}

/// Model runtime for executing inference.
pub struct ModelRuntime {
    config: ModelConfig,
}

impl ModelRuntime {
    /// Create a new model runtime.
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    /// Load model from path.
    pub fn load_from_path(_path: &str) -> Result<Self> {
        let config = ModelConfig::default();
        Ok(Self::new(config))
    }

    /// Get model configuration.
    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get input shape.
    pub fn input_shape(&self) -> &[usize] {
        &self.config.input_shape
    }

    /// Get output shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.config.output_shape
    }

    /// Validate input dimensions.
    pub fn validate_input(&self, shape: &[usize]) -> Result<()> {
        if shape == self.config.input_shape {
            Ok(())
        } else {
            Err(crate::error::NpuError::InvalidShape(
                format!(
                    "Input shape mismatch: {:?} != {:?}",
                    shape, self.config.input_shape
                ),
            ))
        }
    }
}

/// Layer types supported by the NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullyConnected,
    Convolution,
    DepthwiseConvolution,
    PointwiseConvolution,
    Activation,
    BatchNorm,
    Pooling,
    Concat,
    Add,
}

/// Layer definition.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub layer_type: LayerType,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl Layer {
    /// Create a new layer.
    pub fn new(name: String, layer_type: LayerType, input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            name,
            layer_type,
            input_shape,
            output_shape,
        }
    }

    /// Estimate TOPS for this layer.
    pub fn estimate_tops(&self) -> f32 {
        match self.layer_type {
            LayerType::FullyConnected => {
                if self.input_shape.len() >= 2 && self.output_shape.len() >= 1 {
                    let m = self.input_shape[0];
                    let k = self.input_shape[1];
                    let n = self.output_shape[1];
                    (2 * m * k * n) as f32 / 1e12
                } else {
                    0.0
                }
            }
            LayerType::Convolution => {
                if self.input_shape.len() >= 3 && self.output_shape.len() >= 3 {
                    let batch = self.input_shape[0];
                    let h = self.input_shape[1];
                    let w = self.input_shape[2];
                    let c_in = self.input_shape[3];
                    let c_out = self.output_shape[3];
                    (2 * batch * h * w * c_in * c_out) as f32 / 1e12
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

/// Neural network model graph.
pub struct NeuralNetwork {
    name: String,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    /// Create a new neural network.
    pub fn new(name: String) -> Self {
        Self {
            name,
            layers: Vec::new(),
        }
    }

    /// Add a layer to the network.
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Get all layers.
    pub fn get_layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Compute total estimated TOPS.
    pub fn total_tops(&self) -> f32 {
        self.layers.iter().map(|l| l.estimate_tops()).sum()
    }

    /// Get network name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get layer count.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}
