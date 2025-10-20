mod tensor;
mod error;
mod memory;
mod perf_monitor;
mod compute;
mod device;
mod execution;
mod model;
mod power;
mod quantization;
mod optimizer;
mod profiler;

use tensor::Tensor;
use device::{NpuDevice, DeviceInfo, DeviceState};
use execution::ExecutionContext;
use model::{ModelConfig, ModelRuntime, QuantFormat, OptimizationLevel, NeuralNetwork, Layer, LayerType};
use power::{DvfsController, PowerDomain, ThermalManager};
use quantization::{QuantStats, QuantConverter, PTQEngine};
use optimizer::{GraphOptimizer, ComputationGraph, ComputeNode};
use profiler::{Profiler, ProfileEvent};
use std::sync::Arc;

fn main() {
    initialize_logging();

    println!("=== NPU Driver for 20 TOPS RISC Board ===\n");

    demo_device_initialization();
    demo_basic_operations();
    demo_computation_execution();
    demo_model_inference();
    demo_power_management();
    demo_performance_monitoring();
    demo_quantization();
    demo_optimization();
    demo_profiling();

    println!("\n=== All Demos Completed ===\n");
}

fn initialize_logging() {
    #[cfg(debug_assertions)]
    {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .try_init();
    }
}

/// Demonstrate device initialization and status.
fn demo_device_initialization() {
    println!(">> Device Initialization Demo\n");

    let device_info = DeviceInfo {
        device_id: 0,
        peak_throughput_tops: 20.0,
        memory_mb: 512,
        compute_units: 4,
        frequency_mhz: 800,
        power_tdp_watts: 5.0,
        vendor: "Xilinx/SiFive".to_string(),
        device_name: "XilinxAI-Engine 20-TOPS".to_string(),
    };

    let device = Arc::new(NpuDevice::with_config(device_info));
    
    println!("Device: {}", device.get_info().device_name);
    println!("Peak Throughput: {} TOPS", device.get_info().peak_throughput_tops);
    println!("Memory: {} MB", device.get_info().memory_mb);
    println!("State: {:?}\n", device.get_state());

    match device.initialize() {
        Ok(_) => println!("✓ Device initialized successfully\n"),
        Err(e) => println!("✗ Initialization failed: {}\n", e),
    }

    println!("{}\n", serde_json::to_string_pretty(&device.get_status_json()).unwrap());
}

/// Demonstrate basic tensor operations.
fn demo_basic_operations() {
    println!(">> Basic Tensor Operations Demo\n");

    let a = Tensor::random(&[2, 3]);
    let b = Tensor::ones(&[2, 3]);
    let c = a.add(&b);
    let d = c.relu();

    println!("Tensor A shape: {:?}", a.shape());
    println!("Tensor B shape: {:?}", b.shape());
    println!("Tensor C (A+B) shape: {:?}", c.shape());
    println!("Tensor D (ReLU(C)) shape: {:?}", d.shape());
    println!("Sum of D: {:.6}\n", d.sum());
}

/// Demonstrate computation units.
fn demo_computation_execution() {
    println!(">> Computation Execution Demo\n");

    let device = Arc::new(NpuDevice::new());
    let _ = device.initialize();

    let ctx = ExecutionContext::new(device);

    let a = Tensor::random(&[4, 8]);
    let b = Tensor::random(&[8, 6]);

    match ctx.execute_matmul(&a.data, &b.data) {
        Ok(result) => {
            println!("Matrix multiplication: {:?} × {:?} = {:?}", 
                a.shape(), b.shape(), result.shape());
            println!("Throughput: {:.4} GOPS\n", ctx.get_current_throughput_gops());
        }
        Err(e) => println!("Computation failed: {}\n", e),
    }
}

/// Demonstrate model inference framework.
fn demo_model_inference() {
    println!(">> Model Inference Framework Demo\n");

    let config = ModelConfig {
        name: "MobileNetV2".to_string(),
        input_shape: vec![1, 224, 224, 3],
        output_shape: vec![1, 1000],
        quant_format: QuantFormat::Int8,
        optimization_level: OptimizationLevel::O3,
        use_cache: true,
    };

    let runtime = ModelRuntime::new(config);
    println!("Model: {}", runtime.get_config().name);
    println!("Input shape: {:?}", runtime.input_shape());
    println!("Output shape: {:?}", runtime.output_shape());
    println!("Quantization: {:?}", runtime.get_config().quant_format);
    println!("Optimization: {:?}\n", runtime.get_config().optimization_level);

    let mut network = NeuralNetwork::new("MobileNetV2".to_string());
    
    network.add_layer(Layer::new(
        "conv1".to_string(),
        LayerType::Convolution,
        vec![1, 224, 224, 3],
        vec![1, 112, 112, 32],
    ));
    
    network.add_layer(Layer::new(
        "conv2".to_string(),
        LayerType::PointwiseConvolution,
        vec![1, 112, 112, 32],
        vec![1, 112, 112, 64],
    ));
    
    network.add_layer(Layer::new(
        "fc".to_string(),
        LayerType::FullyConnected,
        vec![1, 1000],
        vec![1, 1000],
    ));

    println!("Network: {} ({} layers)", network.name(), network.layer_count());
    println!("Estimated Total TOPS: {:.6}\n", network.total_tops());
}

/// Demonstrate power management capabilities.
fn demo_power_management() {
    println!(">> Power Management Demo\n");

    let dvfs = DvfsController::new();
    
    println!("Power Domains:");
    for domain in &[PowerDomain::Compute, PowerDomain::Memory, PowerDomain::Cache] {
        if let Ok(freqs) = dvfs.list_frequencies(*domain) {
            println!("  {:?}: {} levels ({} - {} MHz)", 
                domain, 
                freqs.len(),
                freqs.first().unwrap_or(&0),
                freqs.last().unwrap_or(&0)
            );
        }
    }

    println!("\nCurrent Power State:");
    for domain in &[PowerDomain::Compute, PowerDomain::Memory, PowerDomain::Cache] {
        if let Ok(state) = dvfs.get_power_state(*domain) {
            println!("  {:?}: {} MHz, {} mV, {:.2} W", 
                domain, state.frequency_mhz, state.voltage_mv, state.power_watts);
        }
    }

    println!("Total Power Estimate: {:.2} W\n", dvfs.get_total_power_estimate());

    let thermal = ThermalManager::default();
    thermal.update_temperature(45.0);
    
    println!("Thermal Management:");
    println!("  Current Temperature: {:.1}°C", thermal.get_temperature());
    println!("  Should Throttle: {}", thermal.should_throttle());
    println!("  Throttle Level: {:.2}%\n", thermal.get_throttle_level() * 100.0);
}

/// Demonstrate performance monitoring.
fn demo_performance_monitoring() {
    println!(">> Performance Monitoring Demo\n");

    let device = Arc::new(NpuDevice::new());
    let _ = device.initialize();

    let perf = device.get_perf_monitor();

    for i in 1..=5 {
        perf.record_operation(1_000_000_000 * i as u64);
        perf.record_power(3.5 + 0.5 * i as f32);
        
        let metrics = perf.get_metrics();
        println!("Iteration {}: {} ops, {:.4} GOPS, {:.2} W peak", 
            i,
            metrics.total_operations,
            perf.get_throughput_gops(),
            metrics.peak_power_watts
        );
    }

    println!("\nFinal Metrics:");
    let metrics = perf.get_metrics();
    println!("  Total Operations: {}", metrics.total_operations);
    println!("  Total Time: {} ms", metrics.total_time_ms);
    println!("  Throughput: {:.4} GOPS", perf.get_throughput_gops());
    println!("  Peak Power: {:.2} W\n", metrics.peak_power_watts);
}

/// Demonstrate quantization engine.
fn demo_quantization() {
    println!(">> Quantization Demo\n");

    let tensor = Tensor::random(&[4, 8]);
    let stats = QuantStats::from_tensor(&tensor.data);

    println!("Tensor Statistics:");
    println!("  Min: {:.6}", stats.min_val);
    println!("  Max: {:.6}", stats.max_val);
    println!("  Mean: {:.6}", stats.mean_val);
    println!("  Std: {:.6}", stats.std_val);

    let converter = QuantConverter::new(&stats, 8, false);
    match converter.quantize_tensor(&tensor.data) {
        Ok(quantized) => {
            println!("\nInt8 Quantization:");
            println!("  Quantized values (first 8): {:?}", &quantized[..8.min(quantized.len())]);
            match converter.dequantize_tensor(&quantized) {
                Ok(dequant) => {
                    let original_sum = tensor.data.sum();
                    let recon_sum = dequant.sum();
                    println!("  Original sum: {:.6}", original_sum);
                    println!("  Reconstructed sum: {:.6}", recon_sum);
                    println!("  Reconstruction error: {:.6}%\n", 
                        ((original_sum - recon_sum) / original_sum * 100.0).abs()
                    );
                }
                Err(e) => println!("  Dequantization failed: {}\n", e),
            }
        }
        Err(e) => println!("  Quantization failed: {}\n", e),
    }

    let ptq = PTQEngine::new(8, false);
    let sample = vec![Tensor::random(&[2, 3]).data, Tensor::random(&[2, 3]).data];
    match ptq.calibrate(&sample) {
        Ok(_) => println!("✓ PTQ calibration completed\n"),
        Err(e) => println!("✗ PTQ calibration failed: {}\n", e),
    }
}

/// Demonstrate graph optimization.
fn demo_optimization() {
    println!(">> Graph Optimization Demo\n");

    let mut graph = ComputationGraph::new();

    let _ = graph.add_node("input".to_string(), ComputeNode::Input { shape: vec![1, 224, 224, 3] });
    let _ = graph.add_node("conv1".to_string(), ComputeNode::Convolution { kernel_shape: vec![3, 3, 3, 32] });
    let _ = graph.add_node("relu1".to_string(), ComputeNode::Activation { activation_type: "relu".to_string() });
    let _ = graph.add_node("output".to_string(), ComputeNode::Output { shape: vec![1, 224, 224, 32] });

    let _ = graph.add_edge("input".to_string(), "conv1".to_string());
    let _ = graph.add_edge("conv1".to_string(), "relu1".to_string());
    let _ = graph.add_edge("relu1".to_string(), "output".to_string());

    println!("Graph Topology:");
    println!("  Nodes: {}", graph.get_node_count());
    println!("  Edges: {}", graph.edges.len());

    let optimizer = GraphOptimizer::new();
    let report = optimizer.get_report();
    println!("\nOptimization Configuration:");
    println!("  Fusion patterns enabled: {}", report.fusion_patterns_enabled);
    println!("  Constant folding: {}", report.constant_folding_enabled);
    println!("  Dead code elimination: {}\n", report.dead_code_elimination_enabled);
}

/// Demonstrate profiling capabilities.
fn demo_profiling() {
    println!(">> Profiling Demo\n");

    let device = Arc::new(NpuDevice::new());
    let _ = device.initialize();
    let mut profiler = Profiler::new(device);

    profiler.record_event(ProfileEvent {
        event_name: "MatMul_4x8x6".to_string(),
        start_time_ms: 0.1,
        duration_ms: 0.05,
        ops_count: 384,
        power_watts: 4.2,
    });

    profiler.record_event(ProfileEvent {
        event_name: "Conv1x1_1x112x112x32".to_string(),
        start_time_ms: 0.15,
        duration_ms: 0.12,
        ops_count: 803_584,
        power_watts: 4.8,
    });

    println!("Profile Events:");
    for event in profiler.get_events() {
        println!("  {}: {:.4} ms, {} ops, {:.2} GOPS, {:.2} W",
            event.event_name,
            event.duration_ms,
            event.ops_count,
            event.get_throughput_gops(),
            event.power_watts
        );
    }

    let report = profiler.generate_report();
    println!("\nProfile Summary:");
    println!("  Total Events: {}", report.total_events);
    println!("  Total Time: {:.4} ms", report.total_time_ms);
    println!("  Total Operations: {}", report.total_operations);
    println!("  Average Throughput: {:.2} GOPS", report.avg_throughput_gops);
    println!("  Peak Power: {:.2} W\n", report.peak_power_watts);
}
