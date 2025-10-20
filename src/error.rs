use thiserror::Error;

/// NPU driver result type.
pub type Result<T> = std::result::Result<T, NpuError>;

/// Error types for the NPU driver.
#[derive(Debug, Error)]
pub enum NpuError {
    #[error("Device not available")]
    DeviceNotAvailable,

    #[error("Initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Performance monitoring error: {0}")]
    PerformanceError(String),

    #[error("Synchronization timeout")]
    SyncTimeout,

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}
