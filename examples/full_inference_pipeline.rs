use npu_rs::{
    NpuDevice, DeviceInfo, Tensor, ExecutionContext, ModelConfig, ModelRuntime,
    QuantFormat, OptimizationLevel, NeuralNetwork, Layer, LayerType,
    QuantStats, QuantConverter, PTQEngine, GraphOptimizer, ComputationGraph,
    ComputeNode, Profiler, ProfileEvent,
};
use std::sync::Arc;

/// Complete inference pipeline example.
fn main() {
    println!("=== Full NPU Inference Pipeline ===\n");

    setup_device();
    build_model();
    quantize_model();
    execute_inference();
    monitor_performance();

    println!("\n=== Pipeline Completed ===\n");
}

fn setup_device() {
    println!("1. Device Setup");
    
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
    match device.initialize() {
        Ok(_) => {
            println!("   ✓ Device initialized");
            let info = device.get_info();
            println!("   Device: {} ({} MB memory, {} TOPS peak)", 
                info.device_name, info.memory_mb, info.peak_throughput_tops
            );
            
            let memory_pool = device.get_memory_pool();
            let manager = memory_pool.get_manager();
            println!("   Available memory: {} MB\n", 
                manager.get_available_bytes() / 1024 / 1024
            );
        }
        Err(e) => println!("   ✗ Init failed: {}\n", e),
    }
}

fn build_model() {
    println!("2. Model Building");

    let model_config = ModelConfig {
        name: "ResNet18-Lite".to_string(),
        input_shape: vec![1, 224, 224, 3],
        output_shape: vec![1, 1000],
        quant_format: QuantFormat::Int8,
        optimization_level: OptimizationLevel::O3,
        use_cache: true,
    };

    let runtime = ModelRuntime::new(model_config);
    println!("   Model: {}", runtime.get_config().name);
    println!("   Input: {:?}", runtime.input_shape());
    println!("   Output: {:?}", runtime.output_shape());

    let mut network = NeuralNetwork::new(runtime.get_config().name.clone());
    
    network.add_layer(Layer::new(
        "stem_conv".to_string(),
        LayerType::Convolution,
        vec![1, 224, 224, 3],
        vec![1, 112, 112, 64],
    ));
    
    network.add_layer(Layer::new(
        "residual_block_1".to_string(),
        LayerType::PointwiseConvolution,
        vec![1, 112, 112, 64],
        vec![1, 112, 112, 64],
    ));
    
    network.add_layer(Layer::new(
        "residual_block_2".to_string(),
        LayerType::PointwiseConvolution,
        vec![1, 56, 56, 128],
        vec![1, 56, 56, 128],
    ));
    
    network.add_layer(Layer::new(
        "global_avg_pool".to_string(),
        LayerType::Pooling,
        vec![1, 7, 7, 512],
        vec![1, 512],
    ));
    
    network.add_layer(Layer::new(
        "classifier".to_string(),
        LayerType::FullyConnected,
        vec![1, 512],
        vec![1, 1000],
    ));

    println!("   Layers: {}", network.layer_count());
    println!("   Estimated TOPS: {:.6}\n", network.total_tops());
}

fn quantize_model() {
    println!("3. Model Quantization");

    let calibration_data = vec![
        Tensor::random(&[1, 224, 224, 3]).data,
        Tensor::random(&[1, 224, 224, 3]).data,
        Tensor::random(&[1, 224, 224, 3]).data,
    ];

    let ptq = PTQEngine::new(8, false);
    match ptq.calibrate(&calibration_data) {
        Ok(converter) => {
            println!("   ✓ Calibration complete");
            
            let sample = &calibration_data[0];
            let stats = QuantStats::from_tensor(sample);
            println!("   Calibration Stats:");
            println!("   - Min: {:.6}", stats.min_val);
            println!("   - Max: {:.6}", stats.max_val);
            println!("   - Mean: {:.6}", stats.mean_val);
            println!("   - Std: {:.6}", stats.std_val);
            
            match converter.quantize_tensor(sample) {
                Ok(quantized) => {
                    println!("   ✓ Quantization complete: {} values", quantized.len());
                    println!("   Compression: {:.2}x\n", 
                        (sample.len() * 4) as f64 / quantized.len() as f64
                    );
                }
                Err(e) => println!("   ✗ Quantization failed: {}\n", e),
            }
        }
        Err(e) => println!("   ✗ Calibration failed: {}\n", e),
    }
}

fn execute_inference() {
    println!("4. Inference Execution");

    let device = Arc::new(NpuDevice::new());
    match device.initialize() {
        Ok(_) => {
            let ctx = ExecutionContext::new(device);
            
            let input = Tensor::random(&[1, 224, 224, 3]);
            let weights = Tensor::random(&[1, 1, 3, 64]);

            println!("   Input: {:?}", input.shape());
            println!("   Weights: {:?}", weights.shape());

            match ctx.execute_conv1x1(&input.data, &weights.data) {
                Ok(output) => {
                    println!("   ✓ Conv1x1 executed");
                    println!("   Output: {:?}", output.shape());
                    println!("   Throughput: {:.4} GOPS\n", ctx.get_current_throughput_gops());
                }
                Err(e) => println!("   ✗ Execution failed: {}\n", e),
            }
        }
        Err(e) => println!("   ✗ Device init failed: {}\n", e),
    }
}

fn monitor_performance() {
    println!("5. Performance Monitoring");

    let device = Arc::new(NpuDevice::new());
    let _ = device.initialize();

    let mut profiler = Profiler::new(device);

    let ops_profile = vec![
        ("Conv3x3_In16_Out32", 1728, 0.25, 4.5),
        ("MatMul_512x1000", 1_024_000, 0.15, 4.2),
        ("ReLU_Activation", 512_000, 0.05, 2.1),
    ];

    println!("   Recording {} operations...", ops_profile.len());
    for (name, ops, duration, power) in ops_profile {
        profiler.record_event(ProfileEvent {
            event_name: name.to_string(),
            start_time_ms: 0.0,
            duration_ms: duration,
            ops_count: ops,
            power_watts: power,
        });
    }

    println!("   \n   Operations Profile:");
    for event in profiler.get_events() {
        println!("   - {}: {:.2} GOPS, {:.2} W",
            event.event_name,
            event.get_throughput_gops(),
            event.power_watts
        );
    }

    let report = profiler.generate_report();
    println!("\n   Performance Summary:");
    println!("   - Total Ops: {}", report.total_operations);
    println!("   - Total Time: {:.4} ms", report.total_time_ms);
    println!("   - Avg Throughput: {:.2} GOPS", report.avg_throughput_gops);
    println!("   - Peak Power: {:.2} W\n", report.peak_power_watts);
}
