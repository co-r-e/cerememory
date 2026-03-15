//! Error types for Cerememory.

use thiserror::Error;

/// Top-level error type for Cerememory operations.
#[derive(Error, Debug)]
pub enum CerememoryError {
    #[error("Record not found: {0}")]
    RecordNotFound(String),

    #[error("Invalid store type: {0}")]
    StoreInvalid(String),

    #[error("Content too large: {size} bytes exceeds limit of {limit} bytes")]
    ContentTooLarge { size: usize, limit: usize },

    #[error("Modality not supported: {0}")]
    ModalityUnsupported(String),

    #[error("Working memory at capacity")]
    WorkingMemoryFull,

    #[error("Decay engine busy, retry after {retry_after_secs}s")]
    DecayEngineBusy { retry_after_secs: u32 },

    #[error("Consolidation in progress")]
    ConsolidationInProgress,

    #[error("Export failed: {0}")]
    ExportFailed(String),

    #[error("Import conflict: {0}")]
    ImportConflict(String),

    #[error("Forget operation requires confirmation")]
    ForgetUnconfirmed,

    #[error("Protocol version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: String, got: String },

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Rate limited, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u32 },
}
