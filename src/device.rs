use crate::error::{NpuError, Result};
use crate::memory::MemoryPool;
use crate::perf_monitor::PerformanceMonitor;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::SystemTime;

/// NPU device state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    Uninitialized,
    Initialized,
    Computing,
    Error,
}

/// NPU device information.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_id: u32,
    pub peak_throughput_tops: f32,
    pub memory_mb: usize,
    pub compute_units: usize,
    pub frequency_mhz: u32,
    pub power_tdp_watts: f32,
    pub vendor: String,
    pub device_name: String,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            device_id: 0,
            peak_throughput_tops: 20.0,
            memory_mb: 512,
            compute_units: 4,
            frequency_mhz: 800,
            power_tdp_watts: 5.0,
            vendor: "RISC NPU Vendor".to_string(),
            device_name: "20-TOPS NPU Accelerator".to_string(),
        }
    }
}

/// Main NPU device driver.
pub struct NpuDevice {
    info: DeviceInfo,
    state: Arc<Mutex<DeviceState>>,
    memory_pool: MemoryPool,
    perf_monitor: Arc<PerformanceMonitor>,
    initialized_at: Arc<Mutex<Option<SystemTime>>>,
}

impl NpuDevice {
    /// Create a new NPU device with default configuration.
    pub fn new() -> Self {
        Self::with_config(DeviceInfo::default())
    }

    /// Create a new NPU device with custom configuration.
    pub fn with_config(info: DeviceInfo) -> Self {
        Self {
            info: info.clone(),
            state: Arc::new(Mutex::new(DeviceState::Uninitialized)),
            memory_pool: MemoryPool::new(info.memory_mb),
            perf_monitor: Arc::new(PerformanceMonitor::new()),
            initialized_at: Arc::new(Mutex::new(None)),
        }
    }

    /// Initialize the NPU device.
    pub fn initialize(&self) -> Result<()> {
        let mut state = self.state.lock();

        match *state {
            DeviceState::Uninitialized => {
                *state = DeviceState::Initialized;
                *self.initialized_at.lock() = Some(SystemTime::now());
                Ok(())
            }
            DeviceState::Initialized => {
                Err(NpuError::InitializationFailed(
                    "Device already initialized".to_string(),
                ))
            }
            DeviceState::Error => {
                Err(NpuError::InitializationFailed(
                    "Device in error state".to_string(),
                ))
            }
            _ => Err(NpuError::InitializationFailed(
                "Invalid state transition".to_string(),
            )),
        }
    }

    /// Reset the device.
    pub fn reset(&self) -> Result<()> {
        let mut state = self.state.lock();
        self.perf_monitor.reset();
        *state = DeviceState::Initialized;
        Ok(())
    }

    /// Get device information.
    pub fn get_info(&self) -> DeviceInfo {
        self.info.clone()
    }

    /// Get current device state.
    pub fn get_state(&self) -> DeviceState {
        *self.state.lock()
    }

    /// Get memory pool.
    pub fn get_memory_pool(&self) -> MemoryPool {
        self.memory_pool.clone()
    }

    /// Get performance monitor.
    pub fn get_perf_monitor(&self) -> Arc<PerformanceMonitor> {
        Arc::clone(&self.perf_monitor)
    }

    /// Check if device is ready for computation.
    pub fn is_ready(&self) -> bool {
        matches!(*self.state.lock(), DeviceState::Initialized)
    }

    /// Get device status as JSON.
    pub fn get_status_json(&self) -> serde_json::Value {
        let state = self.get_state();
        let memory_stats = self.memory_pool.get_manager().get_stats();
        let perf_metrics = self.perf_monitor.get_metrics();

        serde_json::json!({
            "device_id": self.info.device_id,
            "device_name": self.info.device_name,
            "state": format!("{:?}", state),
            "peak_throughput_tops": self.info.peak_throughput_tops,
            "current_memory_mb": memory_stats.allocated_bytes / 1024 / 1024,
            "peak_memory_mb": memory_stats.peak_bytes / 1024 / 1024,
            "total_memory_mb": self.info.memory_mb,
            "performance": {
                "total_operations": perf_metrics.total_operations,
                "total_time_ms": perf_metrics.total_time_ms,
                "peak_power_watts": perf_metrics.peak_power_watts,
                "throughput_gops": self.perf_monitor.get_throughput_gops(),
            }
        })
    }

    /// Shutdown the device.
    pub fn shutdown(&self) -> Result<()> {
        let mut state = self.state.lock();
        match *state {
            DeviceState::Initialized | DeviceState::Computing => {
                *state = DeviceState::Uninitialized;
                Ok(())
            }
            _ => Err(NpuError::DeviceError(
                "Cannot shutdown device not in valid state".to_string(),
            )),
        }
    }
}

impl Default for NpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// Global device registry for multi-device support.
pub struct DeviceRegistry {
    devices: Vec<Arc<NpuDevice>>,
}

impl DeviceRegistry {
    /// Create a new device registry.
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
        }
    }

    /// Register a device.
    pub fn register(&mut self, device: Arc<NpuDevice>) -> Result<u32> {
        if self.devices.len() >= 16 {
            return Err(NpuError::DeviceError(
                "Maximum number of devices reached".to_string(),
            ));
        }
        let device_id = self.devices.len() as u32;
        self.devices.push(device);
        Ok(device_id)
    }

    /// Get device by ID.
    pub fn get_device(&self, device_id: u32) -> Result<Arc<NpuDevice>> {
        self.devices
            .get(device_id as usize)
            .cloned()
            .ok_or_else(|| NpuError::DeviceError(format!("Device {} not found", device_id)))
    }

    /// Get total devices.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }
}

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
