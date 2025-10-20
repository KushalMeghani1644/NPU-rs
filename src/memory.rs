use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::error::{NpuError, Result};

/// Memory statistics.
#[derive(Clone, Debug)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub num_allocations: usize,
}

/// NPU device memory manager.
pub struct MemoryManager {
    device_memory_mb: usize,
    allocated: Arc<AtomicUsize>,
    peak_allocated: Arc<AtomicUsize>,
    allocations: Arc<AtomicUsize>,
}

impl MemoryManager {
    /// Create a new memory manager for NPU device.
    pub fn new(device_memory_mb: usize) -> Self {
        Self {
            device_memory_mb,
            allocated: Arc::new(AtomicUsize::new(0)),
            peak_allocated: Arc::new(AtomicUsize::new(0)),
            allocations: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Allocate memory on device.
    pub fn allocate(&self, bytes: usize) -> Result<()> {
        let current = self.allocated.load(Ordering::SeqCst);
        let new_total = current + bytes;

        if new_total > self.device_memory_mb * 1024 * 1024 {
            return Err(NpuError::MemoryError(
                format!(
                    "Out of device memory: {} > {}MB",
                    new_total / 1024 / 1024,
                    self.device_memory_mb
                ),
            ));
        }

        self.allocated.store(new_total, Ordering::SeqCst);
        self.allocations.fetch_add(1, Ordering::SeqCst);

        let peak = self.peak_allocated.load(Ordering::SeqCst);
        if new_total > peak {
            self.peak_allocated.store(new_total, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Free memory on device.
    pub fn deallocate(&self, bytes: usize) {
        self.allocated.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Get current memory statistics.
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            allocated_bytes: self.allocated.load(Ordering::SeqCst),
            peak_bytes: self.peak_allocated.load(Ordering::SeqCst),
            num_allocations: self.allocations.load(Ordering::SeqCst),
        }
    }

    /// Get available memory in bytes.
    pub fn get_available_bytes(&self) -> usize {
        let total = self.device_memory_mb * 1024 * 1024;
        let used = self.allocated.load(Ordering::SeqCst);
        total.saturating_sub(used)
    }

    /// Check if enough memory is available.
    pub fn has_capacity(&self, bytes: usize) -> bool {
        self.get_available_bytes() >= bytes
    }

    /// Reset all statistics (useful for testing).
    pub fn reset(&self) {
        self.allocated.store(0, Ordering::SeqCst);
        self.peak_allocated.store(0, Ordering::SeqCst);
        self.allocations.store(0, Ordering::SeqCst);
    }
}

impl Clone for MemoryManager {
    fn clone(&self) -> Self {
        Self {
            device_memory_mb: self.device_memory_mb,
            allocated: Arc::clone(&self.allocated),
            peak_allocated: Arc::clone(&self.peak_allocated),
            allocations: Arc::clone(&self.allocations),
        }
    }
}

/// NPU device memory pool for optimal allocation patterns.
pub struct MemoryPool {
    manager: MemoryManager,
    buffers: Arc<RwLock<Vec<(usize, Vec<f32>)>>>,
}

impl MemoryPool {
    /// Create a new memory pool.
    pub fn new(device_memory_mb: usize) -> Self {
        Self {
            manager: MemoryManager::new(device_memory_mb),
            buffers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Allocate a buffer from the pool.
    pub fn allocate_buffer(&self, size: usize) -> Result<Vec<f32>> {
        let byte_size = size * std::mem::size_of::<f32>();
        self.manager.allocate(byte_size)?;
        Ok(vec![0.0; size])
    }

    /// Get memory manager.
    pub fn get_manager(&self) -> MemoryManager {
        self.manager.clone()
    }
}

impl Clone for MemoryPool {
    fn clone(&self) -> Self {
        Self {
            manager: self.manager.clone(),
            buffers: Arc::clone(&self.buffers),
        }
    }
}
