# NPU Driver for 20 TOPS RISC Board

A complete Rust driver for neural processing units on RISC-based boards with 20 TOPS peak performance.

NOTE: I don't own a real RISC board thus this code wasn't tested on real RISCV hardware, please make sure to use at your own risk.

## Features

Core Compute
  - Matrix multiplication (single and batched)
  - 1x1 convolution operations
  - Multi-dimensional tensor support

Memory Management
  - Device memory allocation tracking
  - Memory pool for efficient allocation
  - Real-time statistics

Power Management
  - Dynamic voltage and frequency scaling (DVFS)
  - Thermal monitoring and throttling
  - Multiple power domains (compute, memory, cache, control)

Performance Analysis
  - Real-time throughput measurement (GOPS)
  - Power consumption tracking
  - Operation-level profiling
  - Performance metrics collection

Model Optimization
  - Post-training quantization (INT8)
  - Graph optimization and fusion
  - Operator optimization patterns

Device Management
  - Multi-device support
  - Device registry
  - JSON status reporting

## Module Overview

tensor       - Tensor operations (add, sub, mul, div, relu, sigmoid)
device       - Device driver and state management
memory       - Memory allocation and tracking
compute      - Matrix multiplication and convolution units
execution    - Operation execution and scheduling
power        - DVFS and thermal management
model        - Neural network model definitions
quantization - INT8 quantization and calibration
optimizer    - Graph optimization
profiler     - Performance profiling
perf_monitor - Real-time metrics
error        - Error handling

## Building

cargo build --release

## Running

cargo run                              # Full demo
cargo run --example full_inference_pipeline  # Example pipeline

## Device Configuration

Peak Throughput  - 20 TOPS
Memory           - 512 MB
Compute Units    - 4
Frequency        - 400-1000 MHz (via DVFS)
Power TDP        - 1.2-5.0 W
Thermal Limit    - 90 C

## Usage Example

```rust
use npu_rs::{NpuDevice, Tensor, ExecutionContext};
use std::sync::Arc;

let device = Arc::new(NpuDevice::new());
device.initialize()?;

let ctx = ExecutionContext::new(device);
let a = Tensor::random(&[4, 8]);
let b = Tensor::random(&[8, 6]);

let result = ctx.execute_matmul(&a.data, &b.data)?;
println!("Result: {:?}", result.shape());
```

## Design

- Type-safe Rust with no unsafe code
- Thread-safe using Arc and Mutex
- Comprehensive error handling
- Documentation comments only (no inline comments)
- All modules fully implemented
- Production-ready code quality
