use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use serde::{Serialize, Deserialize};

/// Performance metrics for NPU operations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_operations: u64,
    pub total_time_ms: u64,
    pub peak_power_watts: f32,
    pub avg_power_watts: f32,
    pub memory_used_mb: u64,
    pub memory_peak_mb: u64,
    pub avg_utilization_percent: f32,
}

/// Real-time performance monitor for NPU.
pub struct PerformanceMonitor {
    start_time: Instant,
    operation_count: Arc<AtomicU64>,
    peak_power: Arc<AtomicU64>,
    total_power: Arc<AtomicU64>,
    metrics: Arc<parking_lot::Mutex<PerformanceMetrics>>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            operation_count: Arc::new(AtomicU64::new(0)),
            peak_power: Arc::new(AtomicU64::new(0)),
            total_power: Arc::new(AtomicU64::new(0)),
            metrics: Arc::new(parking_lot::Mutex::new(PerformanceMetrics::default())),
        }
    }

    /// Record a completed operation.
    pub fn record_operation(&self, ops: u64) {
        self.operation_count.fetch_add(ops, Ordering::SeqCst);
    }

    /// Record power consumption in watts.
    pub fn record_power(&self, power_watts: f32) {
        let power_bits = power_watts.to_bits() as u64;
        self.total_power.fetch_add(power_bits as u64, Ordering::SeqCst);
        
        let current_peak = f32::from_bits(self.peak_power.load(Ordering::SeqCst) as u32);
        if power_watts > current_peak {
            self.peak_power.store(power_bits, Ordering::SeqCst);
        }
    }

    /// Get current metrics snapshot.
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let mut metrics = self.metrics.lock().clone();
        let elapsed_ms = self.start_time.elapsed().as_millis() as u64;
        
        metrics.total_operations = self.operation_count.load(Ordering::SeqCst);
        metrics.total_time_ms = elapsed_ms;
        metrics.peak_power_watts = f32::from_bits(self.peak_power.load(Ordering::SeqCst) as u32);
        
        metrics
    }

    /// Get throughput in GOPS (giga operations per second).
    pub fn get_throughput_gops(&self) -> f64 {
        let metrics = self.get_metrics();
        if metrics.total_time_ms == 0 {
            return 0.0;
        }
        (metrics.total_operations as f64) / (metrics.total_time_ms as f64 / 1000.0) / 1e9
    }

    /// Reset monitoring counters.
    pub fn reset(&self) {
        self.operation_count.store(0, Ordering::SeqCst);
        self.peak_power.store(0, Ordering::SeqCst);
        self.total_power.store(0, Ordering::SeqCst);
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
