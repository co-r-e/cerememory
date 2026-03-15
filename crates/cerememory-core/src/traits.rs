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

    /// Update access metadata (access count and last accessed timestamp).
    ///
    /// Used by reconsolidation to persist retrieval statistics.
    fn update_access(
        &self,
        id: &Uuid,
        access_count: u32,
        last_accessed_at: chrono::DateTime<chrono::Utc>,
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

/// Capabilities advertised by an LLM provider.
///
/// Used by the engine to decide which multimodal operations to attempt.
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    /// Can generate text embeddings.
    pub text_embedding: bool,
    /// Can generate image embeddings (via vision → text → embed pipeline).
    pub image_embedding: bool,
    /// Can transcribe audio to text.
    pub audio_transcription: bool,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            text_embedding: true,
            image_embedding: false,
            audio_transcription: false,
        }
    }
}

/// Async trait for LLM providers with embed/summarize/extract capabilities.
///
/// This is the Phase 4+ provider trait. Designed to be optional:
/// `Option<Arc<dyn LLMProvider>>` in the engine. When absent, all operations
/// fall back to existing behavior (manual embeddings, truncation summaries).
///
/// Uses boxed futures for dyn-compatibility (needed for `Arc<dyn LLMProvider>`).
///
/// Phase 8 additions: `embed_image`, `transcribe_audio`, `capabilities` with
/// default implementations that return `ModalityUnsupported` / text-only caps.
pub trait LLMProvider: Send + Sync {
    /// Generate an embedding vector for the given text.
    fn embed(
        &self,
        text: &str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>>;

    /// Summarize multiple texts into a single concise summary.
    fn summarize(
        &self,
        texts: &[String],
        max_tokens: usize,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, CerememoryError>> + Send + '_>>;

    /// Extract semantic relations (subject-predicate-object triples) from text.
    fn extract_relations(
        &self,
        text: &str,
    ) -> std::pin::Pin<
        Box<
            dyn Future<Output = Result<Vec<crate::types::ExtractedRelation>, CerememoryError>>
                + Send
                + '_,
        >,
    >;

    /// Generate an embedding for an image by describing it, then embedding the description.
    ///
    /// Default implementation returns `ModalityUnsupported`.
    fn embed_image(
        &self,
        _data: &[u8],
        _format: &str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>>
    {
        Box::pin(async {
            Err(CerememoryError::ModalityUnsupported(
                "Image embedding not supported by this provider".to_string(),
            ))
        })
    }

    /// Transcribe audio data to text.
    ///
    /// Default implementation returns `ModalityUnsupported`.
    fn transcribe_audio(
        &self,
        _data: &[u8],
        _format: &str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, CerememoryError>> + Send + '_>> {
        Box::pin(async {
            Err(CerememoryError::ModalityUnsupported(
                "Audio transcription not supported by this provider".to_string(),
            ))
        })
    }

    /// Advertise which multimodal capabilities this provider supports.
    ///
    /// Default: text embedding only.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }
}

/// Truncate a string to at most `max_len` bytes, ensuring the cut
/// falls on a valid UTF-8 character boundary.
pub fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// No-op LLM provider that returns empty/default results.
///
/// Used as a fallback when no real LLM provider is configured.
pub struct NoOpProvider;

impl LLMProvider for NoOpProvider {
    fn embed(
        &self,
        _text: &str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>>
    {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn summarize(
        &self,
        texts: &[String],
        _max_tokens: usize,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, CerememoryError>> + Send + '_>> {
        let joined: String = texts.join(" ");
        let result = if joined.len() > 200 {
            format!("{}...", truncate_str(&joined, 200))
        } else {
            joined
        };
        Box::pin(async move { Ok(result) })
    }

    fn extract_relations(
        &self,
        _text: &str,
    ) -> std::pin::Pin<
        Box<
            dyn Future<Output = Result<Vec<crate::types::ExtractedRelation>, CerememoryError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_embedding: false,
            image_embedding: false,
            audio_transcription: false,
        }
    }
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
