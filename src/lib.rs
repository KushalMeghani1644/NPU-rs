//! NPU Driver for 20 TOPS RISC Board
//!
//! A complete driver implementation for neural processing units with support for:
//! - Matrix multiplication and convolution operations
//! - Dynamic voltage and frequency scaling (DVFS)
//! - Thermal management
//! - Performance monitoring
//! - Model quantization (PTQ)
//! - Graph optimization
//! - Profiling

pub mod tensor;
pub mod error;
pub mod memory;
pub mod perf_monitor;
pub mod compute;
pub mod device;
pub mod execution;
pub mod model;
pub mod power;
pub mod quantization;
pub mod optimizer;
pub mod profiler;

pub use error::{NpuError, Result};
pub use device::{NpuDevice, DeviceInfo, DeviceState, DeviceRegistry};
pub use execution::{ExecutionContext, BatchScheduler};
pub use memory::{MemoryManager, MemoryPool, MemoryStats};
pub use model::{ModelRuntime, ModelConfig, QuantFormat, OptimizationLevel, NeuralNetwork, Layer, LayerType};
pub use compute::{MatMulUnit, ConvUnit};
pub use power::{DvfsController, PowerDomain, ThermalManager, PowerState};
pub use quantization::{QuantStats, QuantConverter, PTQEngine};
pub use optimizer::{GraphOptimizer, ComputationGraph, ComputeNode, FusionPattern};
pub use profiler::{Profiler, ProfileEvent, ProfileReport};
pub use tensor::Tensor;
pub use perf_monitor::{PerformanceMonitor, PerformanceMetrics};
