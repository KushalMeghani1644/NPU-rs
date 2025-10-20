use std::sync::Arc;
use crate::device::NpuDevice;
use crate::execution::ExecutionContext;
use crate::error::Result;

/// Profiler for NPU operations.
pub struct Profiler {
    device: Arc<NpuDevice>,
    event_log: Vec<ProfileEvent>,
}

/// A profile event.
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub event_name: String,
    pub start_time_ms: f64,
    pub duration_ms: f64,
    pub ops_count: u64,
    pub power_watts: f32,
}

impl ProfileEvent {
    /// Calculate throughput for this event.
    pub fn get_throughput_gops(&self) -> f64 {
        if self.duration_ms > 0.0 {
            (self.ops_count as f64 / 1e9) / (self.duration_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Calculate power efficiency (GOPS/W).
    pub fn get_efficiency_gops_per_watt(&self) -> f64 {
        if self.power_watts > 0.0 {
            self.get_throughput_gops() / (self.power_watts as f64)
        } else {
            0.0
        }
    }
}

impl Profiler {
    /// Create a new profiler.
    pub fn new(device: Arc<NpuDevice>) -> Self {
        Self {
            device,
            event_log: Vec::new(),
        }
    }

    /// Record a profile event.
    pub fn record_event(&mut self, event: ProfileEvent) {
        self.event_log.push(event);
    }

    /// Get all events.
    pub fn get_events(&self) -> &[ProfileEvent] {
        &self.event_log
    }

    /// Clear event log.
    pub fn clear(&mut self) {
        self.event_log.clear();
    }

    /// Generate profiling report.
    pub fn generate_report(&self) -> ProfileReport {
        let total_time: f64 = self.event_log.iter().map(|e| e.duration_ms).sum();
        let total_ops: u64 = self.event_log.iter().map(|e| e.ops_count).sum();
        let peak_power: f32 = self.event_log.iter().map(|e| e.power_watts).fold(0.0, f32::max);
        let avg_throughput: f64 = if total_time > 0.0 {
            (total_ops as f64 / 1e9) / (total_time / 1000.0)
        } else {
            0.0
        };

        ProfileReport {
            total_events: self.event_log.len(),
            total_time_ms: total_time,
            total_operations: total_ops,
            avg_throughput_gops: avg_throughput,
            peak_power_watts: peak_power,
            event_count: self.event_log.len(),
        }
    }
}

/// Profile report summary.
#[derive(Debug)]
pub struct ProfileReport {
    pub total_events: usize,
    pub total_time_ms: f64,
    pub total_operations: u64,
    pub avg_throughput_gops: f64,
    pub peak_power_watts: f32,
    pub event_count: usize,
}
