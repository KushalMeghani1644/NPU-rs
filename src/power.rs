use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use crate::error::{NpuError, Result};

/// Power domain for voltage and frequency scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerDomain {
    Compute,
    Memory,
    Cache,
    Control,
}

/// Power state (DVFS level).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerState {
    pub frequency_mhz: u32,
    pub voltage_mv: u32,
    pub power_watts: f32,
}

/// Dynamic Voltage and Frequency Scaling (DVFS) controller.
pub struct DvfsController {
    states: HashMap<PowerDomain, Vec<PowerState>>,
    current_state: Arc<RwLock<HashMap<PowerDomain, usize>>>,
}

impl DvfsController {
    /// Create a new DVFS controller with default states.
    pub fn new() -> Self {
        let mut states = HashMap::new();

        let compute_states = vec![
            PowerState {
                frequency_mhz: 400,
                voltage_mv: 750,
                power_watts: 1.2,
            },
            PowerState {
                frequency_mhz: 600,
                voltage_mv: 850,
                power_watts: 2.5,
            },
            PowerState {
                frequency_mhz: 800,
                voltage_mv: 950,
                power_watts: 4.0,
            },
            PowerState {
                frequency_mhz: 1000,
                voltage_mv: 1050,
                power_watts: 5.0,
            },
        ];

        let memory_states = vec![
            PowerState {
                frequency_mhz: 100,
                voltage_mv: 750,
                power_watts: 0.5,
            },
            PowerState {
                frequency_mhz: 200,
                voltage_mv: 850,
                power_watts: 1.0,
            },
            PowerState {
                frequency_mhz: 300,
                voltage_mv: 950,
                power_watts: 1.5,
            },
        ];

        states.insert(PowerDomain::Compute, compute_states);
        states.insert(PowerDomain::Memory, memory_states);
        states.insert(PowerDomain::Cache, vec![PowerState {
            frequency_mhz: 800,
            voltage_mv: 950,
            power_watts: 0.5,
        }]);
        states.insert(PowerDomain::Control, vec![PowerState {
            frequency_mhz: 200,
            voltage_mv: 850,
            power_watts: 0.3,
        }]);

        let mut current_state = HashMap::new();
        current_state.insert(PowerDomain::Compute, 3);
        current_state.insert(PowerDomain::Memory, 2);
        current_state.insert(PowerDomain::Cache, 0);
        current_state.insert(PowerDomain::Control, 0);

        Self {
            states,
            current_state: Arc::new(RwLock::new(current_state)),
        }
    }

    /// Set frequency for a power domain.
    pub fn set_frequency(&self, domain: PowerDomain, level: usize) -> Result<()> {
        let states = self
            .states
            .get(&domain)
            .ok_or_else(|| NpuError::DeviceError("Invalid power domain".to_string()))?;

        if level >= states.len() {
            return Err(NpuError::InvalidConfiguration(
                format!("Invalid frequency level: {}", level),
            ));
        }

        let mut current = self.current_state.write();
        current.insert(domain, level);
        Ok(())
    }

    /// Get current power state for domain.
    pub fn get_power_state(&self, domain: PowerDomain) -> Result<PowerState> {
        let states = self
            .states
            .get(&domain)
            .ok_or_else(|| NpuError::DeviceError("Invalid power domain".to_string()))?;

        let current = self.current_state.read();
        let level = current
            .get(&domain)
            .ok_or_else(|| NpuError::DeviceError("Power domain not initialized".to_string()))?;

        Ok(states[*level])
    }

    /// Get total power consumption estimate.
    pub fn get_total_power_estimate(&self) -> f32 {
        let current = self.current_state.read();
        let mut total = 0.0;

        for (domain, level) in current.iter() {
            if let Some(states) = self.states.get(domain) {
                if let Some(state) = states.get(*level) {
                    total += state.power_watts;
                }
            }
        }

        total
    }

    /// List available frequency levels for a domain.
    pub fn list_frequencies(&self, domain: PowerDomain) -> Result<Vec<u32>> {
        self.states
            .get(&domain)
            .ok_or_else(|| NpuError::DeviceError("Invalid power domain".to_string()))
            .map(|states| states.iter().map(|s| s.frequency_mhz).collect())
    }

    /// Enable power gating (reduce power domain to minimum state).
    pub fn enable_power_gating(&self, domain: PowerDomain) -> Result<()> {
        self.set_frequency(domain, 0)
    }

    /// Disable power gating (maximize power domain).
    pub fn disable_power_gating(&self, domain: PowerDomain) -> Result<()> {
        let states = self
            .states
            .get(&domain)
            .ok_or_else(|| NpuError::DeviceError("Invalid power domain".to_string()))?;

        self.set_frequency(domain, states.len().saturating_sub(1))
    }
}

impl Default for DvfsController {
    fn default() -> Self {
        Self::new()
    }
}

/// Thermal management monitor.
pub struct ThermalManager {
    max_temp_celsius: f32,
    current_temp_celsius: Arc<RwLock<f32>>,
    throttle_threshold: f32,
}

impl ThermalManager {
    /// Create a new thermal manager.
    pub fn new(max_temp_celsius: f32) -> Self {
        Self {
            max_temp_celsius,
            current_temp_celsius: Arc::new(RwLock::new(35.0)),
            throttle_threshold: max_temp_celsius * 0.85,
        }
    }

    /// Update temperature reading.
    pub fn update_temperature(&self, temp: f32) {
        *self.current_temp_celsius.write() = temp;
    }

    /// Get current temperature.
    pub fn get_temperature(&self) -> f32 {
        *self.current_temp_celsius.read()
    }

    /// Check if throttling is needed.
    pub fn should_throttle(&self) -> bool {
        self.get_temperature() >= self.throttle_threshold
    }

    /// Get throttle level (0.0 = no throttle, 1.0 = full throttle).
    pub fn get_throttle_level(&self) -> f32 {
        let temp = self.get_temperature();
        if temp < self.throttle_threshold {
            0.0
        } else if temp >= self.max_temp_celsius {
            1.0
        } else {
            (temp - self.throttle_threshold) / (self.max_temp_celsius - self.throttle_threshold)
        }
    }
}

impl Default for ThermalManager {
    fn default() -> Self {
        Self::new(90.0)
    }
}

impl Clone for ThermalManager {
    fn clone(&self) -> Self {
        Self {
            max_temp_celsius: self.max_temp_celsius,
            current_temp_celsius: Arc::clone(&self.current_temp_celsius),
            throttle_threshold: self.throttle_threshold,
        }
    }
}
