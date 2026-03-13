//! CMP (Cerememory Protocol) request and response types.
//!
//! Implements CMP Spec v1.0 — all operation categories:
//! Encode, Recall, Lifecycle, Introspect, plus error handling and versioning.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::*;

// ─── Protocol Versioning (CMP Spec §8) ───────────────────────────────

/// Protocol version header included in all CMP messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CMPHeader {
    pub protocol: String,
    pub version: String,
    pub request_id: Uuid,
    pub timestamp: DateTime<Utc>,
}

impl CMPHeader {
    pub fn new() -> Self {
        Self {
            protocol: "cmp".to_string(),
            version: "1.0".to_string(),
            request_id: Uuid::now_v7(),
            timestamp: Utc::now(),
        }
    }

    pub fn is_compatible(&self) -> bool {
        self.protocol == "cmp" && self.version.starts_with("1.")
    }
}

impl Default for CMPHeader {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Encode Operations (CMP Spec §3) ─────────────────────────────────

/// Context for encoding operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EncodeContext {
    pub source: Option<String>,
    pub session_id: Option<String>,
    pub spatial: Option<serde_json::Value>,
    pub temporal: Option<serde_json::Value>,
}

/// Manual association hint provided during encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualAssociation {
    pub target_id: Uuid,
    pub association_type: AssociationType,
    pub weight: f64,
}

/// encode.store request (CMP Spec §3.1) — Store a new memory record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeStoreRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub content: MemoryContent,
    #[serde(default)]
    pub store: Option<StoreType>,
    #[serde(default)]
    pub emotion: Option<EmotionVector>,
    #[serde(default)]
    pub context: Option<EncodeContext>,
    #[serde(default)]
    pub associations: Option<Vec<ManualAssociation>>,
}

/// encode.store response (CMP Spec §3.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeStoreResponse {
    pub record_id: Uuid,
    pub store: StoreType,
    pub initial_fidelity: f64,
    pub associations_created: u32,
}

/// encode.batch request (CMP Spec §3.2) — Store multiple records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeBatchRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub records: Vec<EncodeStoreRequest>,
    #[serde(default)]
    pub infer_associations: bool,
}

/// encode.batch response (CMP Spec §3.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeBatchResponse {
    pub results: Vec<EncodeStoreResponse>,
    pub associations_inferred: u32,
}

/// encode.update request (CMP Spec §3.3) — Update an existing record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeUpdateRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub record_id: Uuid,
    #[serde(default)]
    pub content: Option<MemoryContent>,
    #[serde(default)]
    pub emotion: Option<EmotionVector>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

// ─── Recall Operations (CMP Spec §4) ─────────────────────────────────

/// Multimodal recall cue (CMP Spec §4.1).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RecallCue {
    pub text: Option<String>,
    pub image: Option<Vec<u8>>,
    pub audio: Option<Vec<u8>>,
    pub emotion: Option<EmotionVector>,
    pub temporal: Option<TemporalRange>,
    pub spatial: Option<serde_json::Value>,
    pub semantic: Option<serde_json::Value>,
    /// Optional embedding vector for semantic similarity search.
    pub embedding: Option<Vec<f32>>,
}

/// Temporal range filter for recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// recall.query request (CMP Spec §4.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallQueryRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub cue: RecallCue,
    #[serde(default)]
    pub stores: Option<Vec<StoreType>>,
    #[serde(default = "default_recall_limit")]
    pub limit: u32,
    #[serde(default)]
    pub min_fidelity: Option<f64>,
    #[serde(default)]
    pub include_decayed: bool,
    #[serde(default = "default_true")]
    pub reconsolidate: bool,
    #[serde(default = "default_activation_depth")]
    pub activation_depth: u32,
    #[serde(default = "default_recall_mode")]
    pub recall_mode: RecallMode,
}

fn default_recall_limit() -> u32 {
    10
}
fn default_true() -> bool {
    true
}
fn default_activation_depth() -> u32 {
    2
}
fn default_recall_mode() -> RecallMode {
    RecallMode::Human
}

/// A single recalled memory with relevance scoring (CMP Spec §4.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledMemory {
    pub record: MemoryRecord,
    pub relevance_score: f64,
    #[serde(default)]
    pub activation_path: Option<Vec<Uuid>>,
    pub rendered_content: MemoryContent,
}

/// Activation trace for debugging recall paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationTrace {
    pub source_id: Uuid,
    pub activations: Vec<ActivationNode>,
}

/// A single node in an activation trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationNode {
    pub record_id: Uuid,
    pub activation_level: f64,
    pub hop: u32,
    pub edge_type: AssociationType,
}

/// recall.query response (CMP Spec §4.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallQueryResponse {
    pub memories: Vec<RecalledMemory>,
    #[serde(default)]
    pub activation_trace: Option<ActivationTrace>,
    pub total_candidates: u32,
}

/// recall.associate request (CMP Spec §4.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallAssociateRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub record_id: Uuid,
    #[serde(default)]
    pub association_types: Option<Vec<AssociationType>>,
    #[serde(default = "default_activation_depth")]
    pub depth: u32,
    #[serde(default = "default_min_weight")]
    pub min_weight: f64,
    #[serde(default = "default_recall_limit")]
    pub limit: u32,
}

fn default_min_weight() -> f64 {
    0.1
}

/// recall.associate response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallAssociateResponse {
    pub memories: Vec<RecalledMemory>,
    pub total_candidates: u32,
}

/// Time granularity for timeline queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeGranularity {
    Minute,
    Hour,
    Day,
    Week,
    Month,
}

/// recall.timeline request (CMP Spec §4.3, OPTIONAL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallTimelineRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub range: TemporalRange,
    #[serde(default = "default_granularity")]
    pub granularity: TimeGranularity,
    #[serde(default)]
    pub min_fidelity: Option<f64>,
    #[serde(default)]
    pub emotion_filter: Option<EmotionVector>,
}

fn default_granularity() -> TimeGranularity {
    TimeGranularity::Hour
}

/// recall.graph request (CMP Spec §4.4, OPTIONAL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallGraphRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    #[serde(default)]
    pub center_id: Option<Uuid>,
    #[serde(default = "default_activation_depth")]
    pub depth: u32,
    #[serde(default)]
    pub edge_types: Option<Vec<String>>,
    #[serde(default = "default_recall_limit")]
    pub limit_nodes: u32,
}

// ─── Lifecycle Operations (CMP Spec §5) ──────────────────────────────

/// Consolidation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsolidationStrategy {
    Full,
    Incremental,
    Selective,
}

/// lifecycle.consolidate request (CMP Spec §5.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidateRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    #[serde(default = "default_consolidation_strategy")]
    pub strategy: ConsolidationStrategy,
    #[serde(default)]
    pub min_age_hours: u32,
    #[serde(default)]
    pub min_access_count: u32,
    #[serde(default)]
    pub dry_run: bool,
}

fn default_consolidation_strategy() -> ConsolidationStrategy {
    ConsolidationStrategy::Incremental
}

/// lifecycle.consolidate response (CMP Spec §5.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidateResponse {
    pub records_processed: u32,
    pub records_migrated: u32,
    pub records_compressed: u32,
    pub records_pruned: u32,
    pub semantic_nodes_created: u32,
}

/// lifecycle.decay_tick request (CMP Spec §5.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayTickRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    #[serde(default)]
    pub tick_duration_seconds: Option<u32>,
}

/// lifecycle.decay_tick response (CMP Spec §5.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayTickResponse {
    pub records_updated: u32,
    pub records_below_threshold: u32,
    pub records_pruned: u32,
}

/// lifecycle.set_mode request (CMP Spec §5.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetModeRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub mode: RecallMode,
    #[serde(default)]
    pub scope: Option<Vec<StoreType>>,
}

/// lifecycle.forget request (CMP Spec §5.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    #[serde(default)]
    pub record_ids: Option<Vec<Uuid>>,
    #[serde(default)]
    pub store: Option<StoreType>,
    #[serde(default)]
    pub temporal_range: Option<TemporalRange>,
    #[serde(default)]
    pub cascade: bool,
    pub confirm: bool,
}

/// lifecycle.export request (CMP Spec §5.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub stores: Option<Vec<StoreType>>,
    #[serde(default)]
    pub encrypt: bool,
    #[serde(default)]
    pub encryption_key: Option<String>,
}

fn default_format() -> String {
    "cma".to_string()
}

/// lifecycle.export response (CMP Spec §5.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResponse {
    pub archive_id: String,
    pub size_bytes: u64,
    pub record_count: u32,
    pub checksum: String,
}

/// Import conflict resolution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolution {
    KeepExisting,
    KeepImported,
    KeepNewer,
}

/// Import strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImportStrategy {
    Merge,
    Replace,
}

/// lifecycle.import request (CMP Spec §5.6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub archive_id: String,
    #[serde(default = "default_import_strategy")]
    pub strategy: ImportStrategy,
    #[serde(default = "default_conflict_resolution")]
    pub conflict_resolution: ConflictResolution,
    #[serde(default)]
    pub decryption_key: Option<String>,
}

fn default_import_strategy() -> ImportStrategy {
    ImportStrategy::Merge
}

fn default_conflict_resolution() -> ConflictResolution {
    ConflictResolution::KeepNewer
}

// ─── Introspect Operations (CMP Spec §6) ─────────────────────────────

/// introspect.stats response (CMP Spec §6.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResponse {
    pub total_records: u32,
    pub records_by_store: std::collections::HashMap<StoreType, u32>,
    pub total_associations: u32,
    pub avg_fidelity: f64,
    pub avg_fidelity_by_store: std::collections::HashMap<StoreType, f64>,
    #[serde(default)]
    pub oldest_record: Option<DateTime<Utc>>,
    #[serde(default)]
    pub newest_record: Option<DateTime<Utc>>,
    pub total_recall_count: u64,
    #[serde(default)]
    pub evolution_metrics: Option<EvolutionMetrics>,
    /// Whether background decay is enabled and running.
    #[serde(default)]
    pub background_decay_enabled: bool,
}

/// Parameter adjustment record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterAdjustment {
    pub store: StoreType,
    pub parameter: String,
    pub original_value: f64,
    pub current_value: f64,
    pub reason: String,
}

/// Evolution metrics (CMP Spec §6.4, OPTIONAL).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EvolutionMetrics {
    pub parameter_adjustments: Vec<ParameterAdjustment>,
    pub detected_patterns: Vec<String>,
    pub schema_adaptations: Vec<String>,
}

/// introspect.record request (CMP Spec §6.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordIntrospectRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub record_id: Uuid,
    #[serde(default)]
    pub include_history: bool,
    #[serde(default)]
    pub include_associations: bool,
    #[serde(default)]
    pub include_versions: bool,
}

/// introspect.decay_forecast request (CMP Spec §6.3, OPTIONAL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayForecastRequest {
    #[serde(default)]
    pub header: Option<CMPHeader>,
    pub record_ids: Vec<Uuid>,
    pub forecast_at: DateTime<Utc>,
}

/// A single record's decay forecast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayForecast {
    pub record_id: Uuid,
    pub current_fidelity: f64,
    pub forecasted_fidelity: f64,
    #[serde(default)]
    pub estimated_threshold_date: Option<DateTime<Utc>>,
}

/// introspect.decay_forecast response (CMP Spec §6.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayForecastResponse {
    pub forecasts: Vec<DecayForecast>,
}

// ─── Error Handling (CMP Spec §7) ────────────────────────────────────

/// CMP error codes (CMP Spec §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CMPErrorCode {
    RecordNotFound,
    StoreInvalid,
    ContentTooLarge,
    ValidationError,
    ModalityUnsupported,
    WorkingMemoryFull,
    DecayEngineBusy,
    ConsolidationInProgress,
    ExportFailed,
    ImportConflict,
    ForgetUnconfirmed,
    VersionMismatch,
    InternalError,
}

/// Standardized CMP error envelope (CMP Spec §7).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CMPError {
    pub code: CMPErrorCode,
    pub message: String,
    #[serde(default)]
    pub details: Option<serde_json::Value>,
    #[serde(default)]
    pub retry_after: Option<u32>,
}

impl CMPError {
    pub fn new(code: CMPErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
            retry_after: None,
        }
    }
}

impl From<&crate::error::CerememoryError> for CMPError {
    fn from(err: &crate::error::CerememoryError) -> Self {
        use crate::error::CerememoryError;
        match err {
            CerememoryError::RecordNotFound(id) => {
                CMPError::new(CMPErrorCode::RecordNotFound, format!("Record not found: {id}"))
            }
            CerememoryError::StoreInvalid(s) => {
                CMPError::new(CMPErrorCode::StoreInvalid, format!("Invalid store: {s}"))
            }
            CerememoryError::ContentTooLarge { size, limit } => {
                CMPError::new(CMPErrorCode::ContentTooLarge, format!("{size} bytes exceeds {limit}"))
            }
            CerememoryError::ModalityUnsupported(m) => {
                CMPError::new(CMPErrorCode::ModalityUnsupported, format!("Unsupported: {m}"))
            }
            CerememoryError::WorkingMemoryFull => {
                CMPError::new(CMPErrorCode::WorkingMemoryFull, "Working memory at capacity")
            }
            CerememoryError::DecayEngineBusy { retry_after_secs } => {
                let mut e = CMPError::new(CMPErrorCode::DecayEngineBusy, "Decay engine busy");
                e.retry_after = Some(*retry_after_secs);
                e
            }
            CerememoryError::ConsolidationInProgress => {
                CMPError::new(CMPErrorCode::ConsolidationInProgress, "Consolidation in progress")
            }
            CerememoryError::ExportFailed(msg) => {
                CMPError::new(CMPErrorCode::ExportFailed, msg.clone())
            }
            CerememoryError::ImportConflict(msg) => {
                CMPError::new(CMPErrorCode::ImportConflict, msg.clone())
            }
            CerememoryError::ForgetUnconfirmed => {
                CMPError::new(CMPErrorCode::ForgetUnconfirmed, "Confirm required")
            }
            CerememoryError::VersionMismatch { expected, got } => {
                CMPError::new(CMPErrorCode::VersionMismatch, format!("Expected {expected}, got {got}"))
            }
            CerememoryError::Validation(msg) => {
                CMPError::new(CMPErrorCode::ValidationError, msg.clone())
            }
            CerememoryError::Storage(msg) => {
                CMPError::new(CMPErrorCode::InternalError, format!("Storage: {msg}"))
            }
            CerememoryError::Serialization(msg) => {
                CMPError::new(CMPErrorCode::InternalError, format!("Serialization: {msg}"))
            }
            CerememoryError::Internal(msg) => {
                CMPError::new(CMPErrorCode::InternalError, msg.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmp_header_defaults_are_valid() {
        let header = CMPHeader::new();
        assert_eq!(header.protocol, "cmp");
        assert_eq!(header.version, "1.0");
        assert!(header.is_compatible());
    }

    #[test]
    fn cmp_header_version_check() {
        let mut header = CMPHeader::new();
        header.version = "1.1".to_string();
        assert!(header.is_compatible());
        header.version = "2.0".to_string();
        assert!(!header.is_compatible());
    }

    #[test]
    fn encode_store_request_json_roundtrip() {
        let req = EncodeStoreRequest {
            header: Some(CMPHeader::new()),
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Hello world".to_vec(),
                    embedding: None,
                }],
                summary: Some("Test".to_string()),
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            associations: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        let decoded: EncodeStoreRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.store, Some(StoreType::Episodic));
        assert_eq!(decoded.content.blocks.len(), 1);
    }

    #[test]
    fn encode_store_request_msgpack_roundtrip() {
        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Test data".to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            store: None,
            emotion: None,
            context: None,
            associations: None,
        };

        let packed = rmp_serde::to_vec(&req).unwrap();
        let decoded: EncodeStoreRequest = rmp_serde::from_slice(&packed).unwrap();
        assert_eq!(decoded.content.blocks[0].data, b"Test data");
    }

    #[test]
    fn recall_query_request_defaults() {
        let json = r#"{"cue":{"text":"hello"}}"#;
        let req: RecallQueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 10);
        assert!(req.reconsolidate);
        assert_eq!(req.activation_depth, 2);
        assert_eq!(req.recall_mode, RecallMode::Human);
    }

    #[test]
    fn cmp_error_from_cerememory_error() {
        let err = crate::error::CerememoryError::RecordNotFound("abc".to_string());
        let cmp_err = CMPError::from(&err);
        assert_eq!(cmp_err.code, CMPErrorCode::RecordNotFound);
    }

    #[test]
    fn decay_tick_request_defaults() {
        let json = r#"{}"#;
        let req: DecayTickRequest = serde_json::from_str(json).unwrap();
        assert!(req.tick_duration_seconds.is_none());
        assert!(req.header.is_none());
    }

    #[test]
    fn forget_request_requires_confirm() {
        let json = r#"{"confirm": true, "record_ids": ["01916e3a-1234-7000-8000-000000000001"]}"#;
        let req: ForgetRequest = serde_json::from_str(json).unwrap();
        assert!(req.confirm);
        assert_eq!(req.record_ids.unwrap().len(), 1);
    }

    #[test]
    fn stats_response_roundtrip() {
        let mut by_store = std::collections::HashMap::new();
        by_store.insert(StoreType::Episodic, 42u32);

        let stats = StatsResponse {
            total_records: 42,
            records_by_store: by_store,
            total_associations: 10,
            avg_fidelity: 0.85,
            avg_fidelity_by_store: std::collections::HashMap::new(),
            oldest_record: None,
            newest_record: None,
            total_recall_count: 100,
            evolution_metrics: None,
            background_decay_enabled: false,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let decoded: StatsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_records, 42);
    }
}
