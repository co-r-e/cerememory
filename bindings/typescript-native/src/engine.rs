//! napi-rs wrapper around `cerememory_engine::CerememoryEngine`.
//!
//! Each method constructs the appropriate CMP request, executes it
//! on a dedicated Tokio runtime via `block_on`, then converts the
//! response to a JSON value for transparent JS/TS consumption.

use uuid::Uuid;

use cerememory_core::protocol::*;
use cerememory_core::types::*;
use cerememory_engine::EngineConfig;

use crate::types::{
    parse_raw_source, parse_raw_speaker, parse_raw_visibility, parse_secrecy_level,
    parse_store_type, parse_uuid, recall_response_to_json, record_to_json, stats_to_json,
    to_napi_error,
};

/// Native Cerememory engine instance.
///
/// Wraps the full in-memory CerememoryEngine with a dedicated Tokio
/// runtime. All operations are synchronous from the JS caller's
/// perspective (backed by `runtime.block_on`).
#[napi]
pub struct CerememoryEngine {
    inner: cerememory_engine::CerememoryEngine,
    runtime: std::sync::Mutex<tokio::runtime::Runtime>,
}

impl CerememoryEngine {
    fn runtime(&self) -> napi::Result<std::sync::MutexGuard<'_, tokio::runtime::Runtime>> {
        self.runtime
            .lock()
            .map_err(|e| napi::Error::from_reason(format!("Tokio runtime lock poisoned: {e}")))
    }
}

#[napi]
impl CerememoryEngine {
    /// Create a new in-memory Cerememory engine.
    ///
    /// Uses all default configuration: in-memory stores, default
    /// decay parameters, human recall mode, no LLM provider.
    #[napi(constructor)]
    pub fn new() -> napi::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to create Tokio runtime: {e}"))
            })?;

        let inner = cerememory_engine::CerememoryEngine::new(EngineConfig::default())
            .map_err(to_napi_error)?;

        Ok(Self {
            inner,
            runtime: std::sync::Mutex::new(runtime),
        })
    }

    /// Store a text memory and return the new record's UUID.
    ///
    /// @param text - The text content to store.
    /// @param store - Optional store type: "episodic", "semantic",
    ///   "procedural", "emotional", or "working". Auto-routed if omitted.
    /// @param metadata - Optional JSON metadata to attach to the record.
    /// @returns The UUID of the newly created record.
    #[napi]
    pub fn store(
        &self,
        text: String,
        store: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> napi::Result<String> {
        let store_type = match store {
            Some(ref s) => Some(parse_store_type(s)?),
            None => None,
        };

        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: text.into_bytes(),
                    embedding: None,
                }],
                summary: None,
            },
            store: store_type,
            emotion: None,
            context: None,
            metadata,
            associations: None,
        };

        let runtime = self.runtime()?;
        let resp = runtime
            .block_on(self.inner.encode_store(req))
            .map_err(to_napi_error)?;

        Ok(resp.record_id.to_string())
    }

    /// Recall memories matching a text query.
    ///
    /// @param query - The text cue for memory retrieval.
    /// @param limit - Maximum number of memories to return (default: 10).
    /// @param reconsolidate - Whether to reconsolidate recalled memories.
    /// @param activation_depth - Activation depth for spreading activation.
    /// @returns JSON object with `memories`, `total_candidates`, and
    ///   optional `activation_trace`.
    #[napi]
    pub fn recall(
        &self,
        query: String,
        limit: Option<u32>,
        reconsolidate: Option<bool>,
        activation_depth: Option<u32>,
    ) -> napi::Result<serde_json::Value> {
        let req = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some(query),
                ..Default::default()
            },
            stores: None,
            limit: limit.unwrap_or(10),
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: reconsolidate.unwrap_or(DEFAULT_RECONSOLIDATE),
            activation_depth: activation_depth.unwrap_or(DEFAULT_ACTIVATION_DEPTH),
            recall_mode: RecallMode::Human,
        };

        let runtime = self.runtime()?;
        let resp = runtime
            .block_on(self.inner.recall_query(req))
            .map_err(to_napi_error)?;

        recall_response_to_json(&resp)
    }

    /// Get a single memory record by its UUID.
    ///
    /// @param id - The record UUID as a string.
    /// @returns JSON representation of the full MemoryRecord.
    #[napi]
    pub fn get_record(&self, id: String) -> napi::Result<serde_json::Value> {
        let uuid = parse_uuid(&id)?;

        let req = RecordIntrospectRequest {
            header: None,
            record_id: uuid,
            include_history: false,
            include_associations: true,
            include_versions: false,
        };

        let runtime = self.runtime()?;
        let record = runtime
            .block_on(self.inner.introspect_record(req))
            .map_err(to_napi_error)?;

        record_to_json(&record)
    }

    /// Delete memory records by their UUIDs.
    ///
    /// @param ids - Array of UUID strings to delete.
    /// @param confirm - Must be true to actually delete. Defaults to false
    ///   (dry-run that returns 0 and errors with "confirm required").
    /// @returns Number of records successfully deleted.
    #[napi]
    pub fn forget(&self, ids: Vec<String>, confirm: Option<bool>) -> napi::Result<u32> {
        let uuids: Vec<Uuid> = ids
            .iter()
            .map(|s| parse_uuid(s))
            .collect::<napi::Result<_>>()?;

        let req = ForgetRequest {
            header: None,
            record_ids: Some(uuids),
            store: None,
            temporal_range: None,
            cascade: false,
            confirm: confirm.unwrap_or(false),
        };

        let runtime = self.runtime()?;
        let deleted = runtime
            .block_on(self.inner.lifecycle_forget(req))
            .map_err(to_napi_error)?;

        Ok(deleted)
    }

    /// Get engine-wide statistics.
    ///
    /// @returns JSON object with total_records, records_by_store,
    ///   avg_fidelity, total_associations, and more.
    #[napi]
    pub fn stats(&self) -> napi::Result<serde_json::Value> {
        let runtime = self.runtime()?;
        let stats = runtime
            .block_on(self.inner.introspect_stats())
            .map_err(to_napi_error)?;

        stats_to_json(&stats)
    }

    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub fn store_raw(
        &self,
        text: String,
        session_id: String,
        topic_id: Option<String>,
        source: Option<String>,
        speaker: Option<String>,
        visibility: Option<String>,
        secrecy_level: Option<String>,
    ) -> napi::Result<String> {
        let req = EncodeStoreRawRequest {
            header: None,
            session_id,
            turn_id: None,
            topic_id,
            source: parse_raw_source(source.as_deref().unwrap_or("conversation"))?,
            speaker: parse_raw_speaker(speaker.as_deref().unwrap_or("user"))?,
            visibility: parse_raw_visibility(visibility.as_deref().unwrap_or("normal"))?,
            secrecy_level: parse_secrecy_level(secrecy_level.as_deref().unwrap_or("public"))?,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: text.into_bytes(),
                    embedding: None,
                }],
                summary: None,
            },
            metadata: None,
        };

        let runtime = self.runtime()?;
        let resp = runtime
            .block_on(self.inner.encode_store_raw(req))
            .map_err(to_napi_error)?;
        Ok(resp.record_id.to_string())
    }

    #[napi]
    pub fn recall_raw(
        &self,
        query: Option<String>,
        session_id: Option<String>,
        limit: Option<u32>,
        include_private_scratch: Option<bool>,
        include_sealed: Option<bool>,
    ) -> napi::Result<serde_json::Value> {
        let req = RecallRawQueryRequest {
            header: None,
            session_id,
            query,
            temporal: None,
            limit: limit.unwrap_or(10),
            include_private_scratch: include_private_scratch.unwrap_or(false),
            include_sealed: include_sealed.unwrap_or(false),
            secrecy_levels: None,
        };

        let runtime = self.runtime()?;
        let resp = runtime
            .block_on(self.inner.recall_raw_query(req))
            .map_err(to_napi_error)?;

        serde_json::to_value(resp)
            .map_err(|e| napi::Error::from_reason(format!("Failed to serialize raw recall: {e}")))
    }

    #[napi]
    pub fn dream_tick(
        &self,
        session_id: Option<String>,
        dry_run: Option<bool>,
        max_groups: Option<u32>,
        include_private_scratch: Option<bool>,
        include_sealed: Option<bool>,
        promote_semantic: Option<bool>,
    ) -> napi::Result<serde_json::Value> {
        let req = DreamTickRequest {
            header: None,
            session_id,
            dry_run: dry_run.unwrap_or(false),
            max_groups: max_groups.unwrap_or(10),
            include_private_scratch: include_private_scratch.unwrap_or(false),
            include_sealed: include_sealed.unwrap_or(false),
            promote_semantic: promote_semantic.unwrap_or(true),
            secrecy_levels: None,
        };

        let runtime = self.runtime()?;
        let resp = runtime
            .block_on(self.inner.lifecycle_dream_tick(req))
            .map_err(to_napi_error)?;

        serde_json::to_value(resp)
            .map_err(|e| napi::Error::from_reason(format!("Failed to serialize dream tick: {e}")))
    }
}
