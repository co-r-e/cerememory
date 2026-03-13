//! Core traits that define the interfaces between Cerememory components.

use crate::error::CerememoryError;
use crate::types::*;
use std::future::Future;
use uuid::Uuid;

/// Trait that all memory stores must implement.
///
/// All methods are async to support both in-memory and disk-backed implementations.
/// Disk-backed stores (e.g., redb) use `spawn_blocking` internally.
pub trait Store: Send + Sync {
    /// Store a new memory record.
    fn store(
        &self,
        record: MemoryRecord,
    ) -> impl Future<Output = Result<Uuid, CerememoryError>> + Send;

    /// Retrieve a memory record by ID.
    fn get(
        &self,
        id: &Uuid,
    ) -> impl Future<Output = Result<Option<MemoryRecord>, CerememoryError>> + Send;

    /// Delete a memory record.
    fn delete(&self, id: &Uuid) -> impl Future<Output = Result<bool, CerememoryError>> + Send;

    /// Update fidelity state for a record.
    fn update_fidelity(
        &self,
        id: &Uuid,
        fidelity: FidelityState,
    ) -> impl Future<Output = Result<(), CerememoryError>> + Send;

    /// Text-based substring search across record content.
    fn query_text(
        &self,
        query: &str,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<MemoryRecord>, CerememoryError>> + Send;

    /// List all record IDs in this store.
    fn list_ids(&self) -> impl Future<Output = Result<Vec<Uuid>, CerememoryError>> + Send;

    /// Count of records in this store.
    fn count(&self) -> impl Future<Output = Result<usize, CerememoryError>> + Send;

    /// Update an existing record (content, emotion, metadata).
    fn update_record(
        &self,
        id: &Uuid,
        content: Option<MemoryContent>,
        emotion: Option<EmotionVector>,
        metadata: Option<serde_json::Value>,
    ) -> impl Future<Output = Result<(), CerememoryError>> + Send;
}

/// Graph-based association access trait.
///
/// Used by the spreading activation engine and the hippocampal coordinator
/// to traverse cross-store associations.
pub trait AssociationGraph: Send + Sync {
    /// Get all associations for a given record.
    fn get_associations(
        &self,
        record_id: &Uuid,
    ) -> impl Future<Output = Result<Vec<Association>, CerememoryError>> + Send;

    /// Determine which store a record lives in.
    fn get_record_store_type(
        &self,
        record_id: &Uuid,
    ) -> impl Future<Output = Result<Option<StoreType>, CerememoryError>> + Send;
}

/// Trait for the decay engine (ADR-005).
///
/// Designed as a pure computation: takes record data in, returns updated fidelity states.
/// This enables rayon-based parallelism without holding store locks.
pub trait DecayEngine: Send + Sync {
    /// Compute decay for a batch of records.
    ///
    /// Returns updated `FidelityState` for each record, along with summary stats.
    fn compute_tick(&self, records: &[DecayInput], tick_duration_secs: f64) -> DecayTickResult;
}

/// Input data for a single record's decay computation.
#[derive(Debug, Clone)]
pub struct DecayInput {
    pub id: Uuid,
    pub fidelity: FidelityState,
    pub emotion: EmotionVector,
    pub last_accessed_at: chrono::DateTime<chrono::Utc>,
    pub access_count: u32,
}

/// Result of a decay tick operation.
#[derive(Debug, Clone)]
pub struct DecayTickResult {
    pub updates: Vec<DecayOutput>,
    pub records_updated: u32,
    pub records_below_threshold: u32,
    pub records_pruned: u32,
}

/// Output for a single record's decay computation.
#[derive(Debug, Clone)]
pub struct DecayOutput {
    pub id: Uuid,
    pub new_fidelity: FidelityState,
    pub should_prune: bool,
}

/// Trait for the spreading activation engine (ADR-004).
pub trait AssociationEngine: Send + Sync {
    /// Propagate activation from a source record through the association graph.
    fn activate(
        &self,
        source_id: &Uuid,
        depth: u32,
        min_weight: f64,
    ) -> impl Future<Output = Result<Vec<ActivatedRecord>, CerememoryError>> + Send;
}

/// A record activated through spreading activation.
#[derive(Debug, Clone)]
pub struct ActivatedRecord {
    pub record_id: Uuid,
    pub activation_level: f64,
    pub path: Vec<Uuid>,
}

/// Trait for LLM adapters (ADR-006).
pub trait LLMAdapter: Send + Sync {
    /// Serialize memories into LLM-consumable format.
    fn serialize_context(&self, memories: &[MemoryRecord], budget_tokens: usize) -> String;

    /// Estimate token count for memory content.
    fn estimate_tokens(&self, content: &MemoryContent) -> usize;

    /// Model metadata.
    fn model_info(&self) -> ModelInfo;
}

/// Information about the LLM model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub provider: String,
    pub model_name: String,
    pub max_context_tokens: usize,
}
