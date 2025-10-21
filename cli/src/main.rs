use std::io::{self, Write};
use npu_rs::{
    NpuDevice, Tensor, ExecutionContext,
};
use std::sync::Arc;

fn main() {
    println!("\n╔════════════════════════════════════════╗");
    println!("║     NPU Driver CLI - Interactive       ║");
    println!("╚════════════════════════════════════════╝\n");

    let device = Arc::new(NpuDevice::new());
    
    if let Err(e) = device.initialize() {
        eprintln!("Failed to initialize device: {}", e);
        return;
    }
    println!("✓ Device initialized\n");

    loop {
        print_menu();
        match get_user_choice() {
            1 => show_device_info(&device),
            2 => show_memory_stats(&device),
            3 => run_matmul_demo(&device),
            4 => run_tensor_ops_demo(),
            5 => show_performance_stats(&device),
            6 => {
                println!("\nGoodbye!\n");
                break;
            }
            _ => println!("Invalid choice. Please try again.\n"),
        }
    }
}

fn print_menu() {
    println!("┌─── Main Menu ───────────────────────┐");
    println!("│ 1. Device Information               │");
    println!("│ 2. Memory Statistics                │");
    println!("│ 3. Run MatMul Demo                  │");
    println!("│ 4. Run Tensor Operations Demo       │");
    println!("│ 5. Performance Statistics           │");
    println!("│ 6. Exit                             │");
    println!("└─────────────────────────────────────┘");
    print!("Select option (1-6): ");
    let _ = io::stdout().flush();
}

fn get_user_choice() -> usize {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    input.trim().parse().unwrap_or(0)
}

fn show_device_info(device: &Arc<NpuDevice>) {
    println!("\n╭─ Device Information ─╮");
    let info = device.get_info();
    println!("│ Device ID:         {}", info.device_id);
    println!("│ Device Name:       {}", info.device_name);
    println!("│ Vendor:            {}", info.vendor);
    println!("│ Peak Throughput:   {} TOPS", info.peak_throughput_tops);
    println!("│ Memory:            {} MB", info.memory_mb);
    println!("│ Compute Units:     {}", info.compute_units);
    println!("│ Frequency:         {} MHz", info.frequency_mhz);
    println!("│ Power TDP:         {} W", info.power_tdp_watts);
    println!("│ State:             {:?}", device.get_state());
    println!("╰─────────────────────╯\n");
}

fn show_memory_stats(device: &Arc<NpuDevice>) {
    println!("\n╭─ Memory Statistics ─╮");
    let memory_pool = device.get_memory_pool();
    let manager = memory_pool.get_manager();
    let stats = manager.get_stats();
    
    println!("│ Allocated:         {} MB", stats.allocated_bytes / 1024 / 1024);
    println!("│ Peak Usage:        {} MB", stats.peak_bytes / 1024 / 1024);
    println!("│ Allocations:       {}", stats.num_allocations);
    println!("│ Available:         {} MB", manager.get_available_bytes() / 1024 / 1024);
    println!("│ Total Memory:      {} MB", device.get_info().memory_mb);
    println!("╰─────────────────────╯\n");
}

fn run_matmul_demo(device: &Arc<NpuDevice>) {
    println!("\n╭─ Matrix Multiplication Demo ─╮");
    
    let ctx = ExecutionContext::new(device.clone());
    
    let a = Tensor::random(&[16, 32]);
    let b = Tensor::random(&[32, 8]);
    
    println!("│ Matrix A: {:?}", a.shape());
    println!("│ Matrix B: {:?}", b.shape());
    
    match ctx.execute_matmul(&a.data, &b.data) {
        Ok(result) => {
            println!("│ Result:   {:?}", result.shape());
            println!("│ Throughput: {:.4} GOPS", ctx.get_current_throughput_gops());
        }
        Err(e) => println!("│ Error: {}", e),
    }
    println!("╰──────────────────────────────╯\n");
}

fn run_tensor_ops_demo() {
    println!("\n╭─ Tensor Operations Demo ─╮");
    
    let a = Tensor::random(&[4, 4]);
    let b = Tensor::ones(&[4, 4]);
    
    let add_result = a.add(&b);
    let relu_result = add_result.relu();
    
    println!("│ Tensor A shape:     {:?}", a.shape());
    println!("│ Tensor B shape:     {:?}", b.shape());
    println!("│ A + B shape:        {:?}", add_result.shape());
    println!("│ ReLU(A+B) shape:    {:?}", relu_result.shape());
    println!("│ Sum of ReLU result: {:.6}", relu_result.sum());
    println!("╰──────────────────────────────╯\n");
}

fn show_performance_stats(device: &Arc<NpuDevice>) {
    println!("\n╭─ Performance Statistics ─╮");
    let perf = device.get_perf_monitor();
    let metrics = perf.get_metrics();
    
    println!("│ Total Operations:  {}", metrics.total_operations);
    println!("│ Total Time:        {} ms", metrics.total_time_ms);
    println!("│ Throughput:        {:.4} GOPS", perf.get_throughput_gops());
    println!("│ Peak Power:        {:.2} W", metrics.peak_power_watts);
    println!("│ Avg Utilization:   {:.2}%", metrics.avg_utilization_percent);
    println!("╰──────────────────────────────╯\n");
}
