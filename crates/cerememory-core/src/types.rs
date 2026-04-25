//! Core data types for Cerememory.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

fn default_metadata() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}
fn default_version() -> u32 {
    1
}

/// The fundamental unit of storage in Cerememory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub store: StoreType,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
    #[serde(default = "Utc::now")]
    pub last_accessed_at: DateTime<Utc>,
    #[serde(default)]
    pub access_count: u32,
    pub content: MemoryContent,
    #[serde(default)]
    pub fidelity: FidelityState,
    #[serde(default)]
    pub emotion: EmotionVector,
    #[serde(default)]
    pub associations: Vec<Association>,
    #[serde(default = "default_metadata")]
    pub metadata: serde_json::Value,
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub meta: MetaMemory,
}

/// A verbatim preserved record in the raw journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawJournalRecord {
    pub id: Uuid,
    pub session_id: String,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub topic_id: Option<String>,
    pub source: RawSource,
    pub speaker: RawSpeaker,
    pub visibility: RawVisibility,
    pub secrecy_level: SecrecyLevel,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
    pub content: MemoryContent,
    #[serde(default = "default_metadata")]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub derived_memory_ids: Vec<Uuid>,
    #[serde(default)]
    pub suppressed: bool,
    #[serde(default)]
    pub meta: MetaMemory,
}

/// Origin channel for a raw journal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawSource {
    Conversation,
    ToolIo,
    Scratchpad,
    Summary,
    Imported,
}

/// The speaker or emitter of a raw journal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawSpeaker {
    User,
    Assistant,
    System,
    Tool,
}

/// Visibility class for raw journal records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawVisibility {
    Normal,
    PrivateScratch,
    Sealed,
}

/// Secrecy tier for preserved raw material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecrecyLevel {
    Public,
    Sensitive,
    Secret,
}

/// Identifies the target memory store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoreType {
    Episodic,
    Semantic,
    Procedural,
    Emotional,
    Working,
}

/// Structured meta-memory attached to every preserved or curated memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaMemory {
    #[serde(default = "default_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub capture_status: MetaCaptureStatus,
    #[serde(default)]
    pub intent: Option<String>,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub trigger: Option<String>,
    #[serde(default)]
    pub goals: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<MetaEvidenceRef>,
    #[serde(default)]
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub alternatives: Vec<MetaAlternative>,
    #[serde(default)]
    pub decision: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub source_record_ids: Vec<Uuid>,
    #[serde(default)]
    pub parent_meta_ids: Vec<Uuid>,
    #[serde(default)]
    pub context_edges: Vec<MetaEdge>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub captured_at: Option<DateTime<Utc>>,
}

/// How a meta-memory payload was captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MetaCaptureStatus {
    Provided,
    Inferred,
    #[default]
    Legacy,
    Unavailable,
}

/// Evidence that supports a meta-memory rationale or relation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MetaEvidenceRef {
    pub record_id: Option<Uuid>,
    pub raw_record_id: Option<Uuid>,
    pub label: Option<String>,
    pub excerpt: Option<String>,
    pub uri: Option<String>,
    pub confidence: Option<f64>,
}

/// An option that was considered while deciding what to do or believe.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MetaAlternative {
    pub option: String,
    pub reason: Option<String>,
    pub chosen: bool,
}

/// Typed context-graph relation carried by the meta-memory plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetaEdge {
    pub source_id: Option<Uuid>,
    pub target_id: Uuid,
    pub relation: MetaRelation,
    pub rationale: Option<String>,
    pub evidence: Vec<MetaEvidenceRef>,
    pub confidence: Option<f64>,
    pub created_at: Option<DateTime<Utc>>,
}

/// Relation kinds for the meta-memory context graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MetaRelation {
    MotivatedBy,
    CausedBy,
    DerivedFrom,
    Supports,
    Contradicts,
    ChoseOver,
    Explains,
    #[default]
    ContextOf,
}

impl Default for MetaMemory {
    fn default() -> Self {
        Self::legacy()
    }
}

impl Default for MetaEdge {
    fn default() -> Self {
        Self {
            source_id: None,
            target_id: Uuid::nil(),
            relation: MetaRelation::ContextOf,
            rationale: None,
            evidence: Vec::new(),
            confidence: None,
            created_at: None,
        }
    }
}

impl MetaMemory {
    pub const LEGACY_RATIONALE: &'static str =
        "No structured meta-memory was captured for this legacy record.";

    /// Meta-memory marker used for records created before the typed meta plane existed.
    pub fn legacy() -> Self {
        Self {
            schema_version: 1,
            capture_status: MetaCaptureStatus::Legacy,
            intent: None,
            rationale: Some(Self::LEGACY_RATIONALE.into()),
            trigger: None,
            goals: Vec::new(),
            evidence: Vec::new(),
            assumptions: Vec::new(),
            alternatives: Vec::new(),
            decision: None,
            confidence: None,
            source_record_ids: Vec::new(),
            parent_meta_ids: Vec::new(),
            context_edges: Vec::new(),
            tags: Vec::new(),
            captured_at: None,
        }
    }

    /// Meta-memory marker for new records when no explicit rationale was supplied.
    pub fn unavailable(trigger: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            capture_status: MetaCaptureStatus::Unavailable,
            intent: None,
            rationale: Some("No explicit meta-rationale was provided at ingest time.".into()),
            trigger: Some(trigger.into()),
            goals: Vec::new(),
            evidence: Vec::new(),
            assumptions: Vec::new(),
            alternatives: Vec::new(),
            decision: None,
            confidence: None,
            source_record_ids: Vec::new(),
            parent_meta_ids: Vec::new(),
            context_edges: Vec::new(),
            tags: Vec::new(),
            captured_at: Some(Utc::now()),
        }
    }

    /// Meta-memory marker for deterministic engine-derived records.
    pub fn inferred(
        trigger: impl Into<String>,
        rationale: impl Into<String>,
        source_record_ids: Vec<Uuid>,
    ) -> Self {
        Self {
            schema_version: 1,
            capture_status: MetaCaptureStatus::Inferred,
            intent: None,
            rationale: Some(rationale.into()),
            trigger: Some(trigger.into()),
            goals: Vec::new(),
            evidence: source_record_ids
                .iter()
                .map(|raw_record_id| MetaEvidenceRef {
                    raw_record_id: Some(*raw_record_id),
                    label: Some("source_raw_record".to_string()),
                    confidence: Some(1.0),
                    ..Default::default()
                })
                .collect(),
            assumptions: Vec::new(),
            alternatives: Vec::new(),
            decision: None,
            confidence: Some(1.0),
            source_record_ids,
            parent_meta_ids: Vec::new(),
            context_edges: Vec::new(),
            tags: Vec::new(),
            captured_at: Some(Utc::now()),
        }
    }

    pub fn validate(&self) -> Result<(), crate::error::CerememoryError> {
        if self.schema_version == 0 {
            return Err(crate::error::CerememoryError::Validation(
                "MetaMemory schema_version must be at least 1".to_string(),
            ));
        }
        validate_optional_confidence(self.confidence, "MetaMemory confidence")?;
        for evidence in &self.evidence {
            validate_optional_confidence(evidence.confidence, "MetaMemory evidence confidence")?;
        }
        for edge in &self.context_edges {
            edge.validate()?;
        }
        Ok(())
    }
}

impl MetaEdge {
    pub fn validate(&self) -> Result<(), crate::error::CerememoryError> {
        if self.target_id.is_nil() {
            return Err(crate::error::CerememoryError::Validation(
                "MetaEdge target_id must not be nil".to_string(),
            ));
        }
        validate_optional_confidence(self.confidence, "MetaEdge confidence")?;
        for evidence in &self.evidence {
            validate_optional_confidence(evidence.confidence, "MetaEdge evidence confidence")?;
        }
        Ok(())
    }
}

fn validate_optional_confidence(
    value: Option<f64>,
    label: &str,
) -> Result<(), crate::error::CerememoryError> {
    if let Some(value) = value {
        if !(0.0..=1.0).contains(&value) || value.is_nan() {
            return Err(crate::error::CerememoryError::Validation(format!(
                "{label} must be in [0.0, 1.0]"
            )));
        }
    }
    Ok(())
}

/// The payload of a memory record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContent {
    pub blocks: Vec<ContentBlock>,
    pub summary: Option<String>,
}

/// A single content block within a memory record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    pub modality: Modality,
    pub format: String,
    pub data: Vec<u8>,
    pub embedding: Option<Vec<f32>>,
}

/// Content modality types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Structured,
    Spatial,
    Temporal,
    Interoceptive,
}

/// Represents the current decay state of a memory record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FidelityState {
    /// 0.0 (fully decayed) to 1.0 (pristine)
    pub score: f64,
    /// 0.0 (no noise) to 1.0 (fully noisy)
    pub noise_level: f64,
    /// Per-second decay coefficient
    pub decay_rate: f64,
    /// Emotional modulation (higher = slower decay)
    pub emotional_anchor: f64,
    /// Times reactivated
    pub reinforcement_count: u32,
    /// Stability constant (increases with each retrieval)
    pub stability: f64,
    pub last_decay_tick: DateTime<Utc>,
}

impl Default for FidelityState {
    fn default() -> Self {
        Self {
            score: 1.0,
            noise_level: 0.0,
            decay_rate: 0.3,
            emotional_anchor: 1.0,
            reinforcement_count: 0,
            stability: 1.0,
            last_decay_tick: Utc::now(),
        }
    }
}

/// Multi-dimensional affective representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmotionVector {
    pub joy: f64,
    pub trust: f64,
    pub fear: f64,
    pub surprise: f64,
    pub sadness: f64,
    pub disgust: f64,
    pub anger: f64,
    pub anticipation: f64,
    pub intensity: f64,
    /// -1.0 (negative) to 1.0 (positive)
    pub valence: f64,
}

impl Default for EmotionVector {
    fn default() -> Self {
        Self {
            joy: 0.0,
            trust: 0.0,
            fear: 0.0,
            surprise: 0.0,
            sadness: 0.0,
            disgust: 0.0,
            anger: 0.0,
            anticipation: 0.0,
            intensity: 0.0,
            valence: 0.0,
        }
    }
}

impl std::str::FromStr for EmotionVector {
    type Err = crate::error::CerememoryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim();

        // Plutchik primary emotions + common aliases (case-insensitive, no allocation)
        let (field_setter, valence): (fn(&mut Self), f64) = if t.eq_ignore_ascii_case("joy")
            || t.eq_ignore_ascii_case("happy")
            || t.eq_ignore_ascii_case("happiness")
        {
            (|e: &mut Self| e.joy = 1.0, 1.0)
        } else if t.eq_ignore_ascii_case("trust") {
            (|e| e.trust = 1.0, 0.7)
        } else if t.eq_ignore_ascii_case("fear") {
            (|e| e.fear = 1.0, -0.8)
        } else if t.eq_ignore_ascii_case("surprise") {
            (|e| e.surprise = 1.0, 0.0)
        } else if t.eq_ignore_ascii_case("sadness") || t.eq_ignore_ascii_case("sad") {
            (|e| e.sadness = 1.0, -1.0)
        } else if t.eq_ignore_ascii_case("disgust") {
            (|e| e.disgust = 1.0, -0.9)
        } else if t.eq_ignore_ascii_case("anger") || t.eq_ignore_ascii_case("angry") {
            (|e| e.anger = 1.0, -0.9)
        } else if t.eq_ignore_ascii_case("anticipation") || t.eq_ignore_ascii_case("anticipatory") {
            (|e| e.anticipation = 1.0, 0.4)
        } else {
            return Err(crate::error::CerememoryError::Validation(format!(
                    "Invalid emotion label: {t}. Use one of: joy, trust, fear, surprise, sadness, disgust, anger, anticipation"
                )));
        };

        let mut ev = Self {
            intensity: 1.0,
            valence,
            ..Default::default()
        };
        field_setter(&mut ev);
        Ok(ev)
    }
}

/// A weighted, typed, bidirectional link between two memory records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    pub target_id: Uuid,
    pub association_type: AssociationType,
    pub weight: f64,
    pub created_at: DateTime<Utc>,
    pub last_co_activation: DateTime<Utc>,
}

/// Types of associations between memory records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssociationType {
    Temporal,
    Spatial,
    Semantic,
    Emotional,
    Causal,
    Sequential,
    CrossModal,
    UserDefined,
}

/// A relation extracted by an LLM from text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// Recall mode for memory retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecallMode {
    /// Apply fidelity/noise (realistic human-like recall)
    Human,
    /// Return original data, no noise applied
    Perfect,
}

/// Maximum content size per modality.
pub const MAX_TEXT_SIZE: usize = 1_048_576; // 1 MB
pub const MAX_IMAGE_SIZE: usize = 10_485_760; // 10 MB
pub const MAX_AUDIO_SIZE: usize = 52_428_800; // 50 MB

impl MemoryRecord {
    /// Create a new record with a single content block.
    fn new_single_block(store: StoreType, block: ContentBlock) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            store,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            access_count: 0,
            content: MemoryContent {
                blocks: vec![block],
                summary: None,
            },
            fidelity: FidelityState::default(),
            emotion: EmotionVector::default(),
            associations: Vec::new(),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            meta: MetaMemory::unavailable("memory_record_constructor"),
            version: 1,
        }
    }

    /// Create a new binary (non-text) memory record.
    pub fn new_binary(
        store: StoreType,
        modality: Modality,
        format: impl Into<String>,
        data: Vec<u8>,
        embedding: Option<Vec<f32>>,
    ) -> Self {
        Self::new_single_block(
            store,
            ContentBlock {
                modality,
                format: format.into(),
                data,
                embedding,
            },
        )
    }

    /// Create a new text-only memory record with sensible defaults.
    pub fn new_text(store: StoreType, text: impl Into<String>) -> Self {
        Self::new_single_block(
            store,
            ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: text.into().into_bytes(),
                embedding: None,
            },
        )
    }

    /// Validate record invariants including modality-specific size limits.
    pub fn validate(&self) -> Result<(), crate::error::CerememoryError> {
        if self.content.blocks.is_empty() {
            return Err(crate::error::CerememoryError::Validation(
                "Record must have at least one content block".to_string(),
            ));
        }
        if self.fidelity.score < 0.0 || self.fidelity.score > 1.0 {
            return Err(crate::error::CerememoryError::Validation(format!(
                "Fidelity score {} out of range [0.0, 1.0]",
                self.fidelity.score
            )));
        }
        if self.fidelity.noise_level < 0.0 || self.fidelity.noise_level > 1.0 {
            return Err(crate::error::CerememoryError::Validation(format!(
                "Noise level {} out of range [0.0, 1.0]",
                self.fidelity.noise_level
            )));
        }
        for assoc in &self.associations {
            if assoc.weight < 0.0 || assoc.weight > 1.0 {
                return Err(crate::error::CerememoryError::Validation(format!(
                    "Association weight {} out of range [0.0, 1.0]",
                    assoc.weight
                )));
            }
        }
        self.meta.validate()?;
        // Modality-specific size limits and embedding validation
        for block in &self.content.blocks {
            let limit = match block.modality {
                Modality::Text => MAX_TEXT_SIZE,
                Modality::Image => MAX_IMAGE_SIZE,
                Modality::Audio | Modality::Video => MAX_AUDIO_SIZE,
                Modality::Structured
                | Modality::Spatial
                | Modality::Temporal
                | Modality::Interoceptive => MAX_TEXT_SIZE,
            };
            if block.data.len() > limit {
                return Err(crate::error::CerememoryError::ContentTooLarge {
                    size: block.data.len(),
                    limit,
                });
            }
            if block.modality == Modality::Text && std::str::from_utf8(&block.data).is_err() {
                return Err(crate::error::CerememoryError::Validation(
                    "Text content must be valid UTF-8".to_string(),
                ));
            }
            if block.modality == Modality::Structured {
                serde_json::from_slice::<serde_json::Value>(&block.data).map_err(|e| {
                    crate::error::CerememoryError::Validation(format!(
                        "Structured content must be valid JSON: {e}"
                    ))
                })?;
            }
            // Validate embedding: reject empty, NaN, Inf
            if let Some(ref emb) = block.embedding {
                if emb.is_empty() {
                    return Err(crate::error::CerememoryError::Validation(
                        "Embedding vector must not be empty".to_string(),
                    ));
                }
                if emb.iter().any(|v| v.is_nan() || v.is_infinite()) {
                    return Err(crate::error::CerememoryError::Validation(
                        "Embedding vector contains NaN or Inf".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Extract text content from the first text block, if any.
    pub fn text_content(&self) -> Option<&str> {
        self.content
            .blocks
            .iter()
            .find(|b| b.modality == Modality::Text)
            .and_then(|b| std::str::from_utf8(&b.data).ok())
    }

    /// Check if this record's text content contains the query (case-insensitive).
    pub fn matches_text(&self, query_lower: &str) -> bool {
        self.text_content()
            .map(|t| t.to_lowercase().contains(query_lower))
            .unwrap_or(false)
    }

    /// Apply partial updates to this record.
    pub fn apply_updates(
        &mut self,
        content: Option<MemoryContent>,
        emotion: Option<EmotionVector>,
        metadata: Option<serde_json::Value>,
        meta: Option<MetaMemory>,
    ) {
        if let Some(c) = content {
            self.content = c;
        }
        if let Some(e) = emotion {
            self.emotion = e;
        }
        if let Some(m) = metadata {
            self.metadata = m;
        }
        if let Some(meta) = meta {
            self.meta = meta;
        }
        self.updated_at = Utc::now();
    }
}

impl RawJournalRecord {
    /// Create a new text-only raw journal record with sensible defaults.
    pub fn new_text(
        session_id: impl Into<String>,
        source: RawSource,
        speaker: RawSpeaker,
        visibility: RawVisibility,
        secrecy_level: SecrecyLevel,
        text: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            session_id: session_id.into(),
            turn_id: None,
            topic_id: None,
            source,
            speaker,
            visibility,
            secrecy_level,
            created_at: now,
            updated_at: now,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: text.into().into_bytes(),
                    embedding: None,
                }],
                summary: None,
            },
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            meta: MetaMemory::unavailable("raw_journal_constructor"),
            derived_memory_ids: Vec::new(),
            suppressed: false,
        }
    }

    /// Validate record invariants using the same content rules as `MemoryRecord`.
    pub fn validate(&self) -> Result<(), crate::error::CerememoryError> {
        if self.session_id.trim().is_empty() {
            return Err(crate::error::CerememoryError::Validation(
                "session_id must not be empty".to_string(),
            ));
        }

        let surrogate = MemoryRecord {
            id: self.id,
            store: StoreType::Working,
            created_at: self.created_at,
            updated_at: self.updated_at,
            last_accessed_at: self.updated_at,
            access_count: 0,
            content: self.content.clone(),
            fidelity: FidelityState::default(),
            emotion: EmotionVector::default(),
            associations: Vec::new(),
            metadata: self.metadata.clone(),
            meta: self.meta.clone(),
            version: 1,
        };
        surrogate.validate()
    }

    /// Extract text content from the first text block, if any.
    pub fn text_content(&self) -> Option<&str> {
        self.content
            .blocks
            .iter()
            .find(|b| b.modality == Modality::Text)
            .and_then(|b| std::str::from_utf8(&b.data).ok())
    }

    /// Check if this raw record's text content contains the query (case-insensitive).
    pub fn matches_text(&self, query_lower: &str) -> bool {
        self.text_content()
            .map(|t| t.to_lowercase().contains(query_lower))
            .unwrap_or(false)
    }
}

/// Rough token estimate from byte count (1 token ≈ 4 bytes).
pub fn estimate_tokens_from_bytes(bytes: usize) -> usize {
    bytes.div_ceil(4)
}

impl std::str::FromStr for StoreType {
    type Err = crate::error::CerememoryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "episodic" => Ok(StoreType::Episodic),
            "semantic" => Ok(StoreType::Semantic),
            "procedural" => Ok(StoreType::Procedural),
            "emotional" => Ok(StoreType::Emotional),
            "working" => Ok(StoreType::Working),
            other => Err(crate::error::CerememoryError::StoreInvalid(
                other.to_string(),
            )),
        }
    }
}

impl std::fmt::Display for StoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreType::Episodic => write!(f, "episodic"),
            StoreType::Semantic => write!(f, "semantic"),
            StoreType::Procedural => write!(f, "procedural"),
            StoreType::Emotional => write!(f, "emotional"),
            StoreType::Working => write!(f, "working"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_text_record_has_valid_defaults() {
        let record = MemoryRecord::new_text(StoreType::Episodic, "Hello, world!");
        assert_eq!(record.store, StoreType::Episodic);
        assert_eq!(record.text_content(), Some("Hello, world!"));
        assert_eq!(record.fidelity.score, 1.0);
        assert_eq!(record.fidelity.noise_level, 0.0);
        assert_eq!(record.access_count, 0);
        assert_eq!(record.version, 1);
        assert!(record.validate().is_ok());
    }

    #[test]
    fn validate_catches_bad_fidelity() {
        let mut record = MemoryRecord::new_text(StoreType::Episodic, "test");
        record.fidelity.score = 1.5;
        assert!(record.validate().is_err());
    }

    #[test]
    fn validate_catches_bad_association_weight() {
        let mut record = MemoryRecord::new_text(StoreType::Episodic, "test");
        record.associations.push(Association {
            target_id: Uuid::now_v7(),
            association_type: AssociationType::Temporal,
            weight: -0.5,
            created_at: Utc::now(),
            last_co_activation: Utc::now(),
        });
        assert!(record.validate().is_err());
    }

    #[test]
    fn validate_rejects_invalid_utf8_text_blocks() {
        let mut record = MemoryRecord::new_binary(
            StoreType::Episodic,
            Modality::Text,
            "text/plain",
            vec![0xFF, 0xFE, 0xFD],
            None,
        );
        record.content.summary = Some("invalid text".to_string());

        let err = record.validate().unwrap_err();
        assert!(matches!(
            err,
            crate::error::CerememoryError::Validation(msg) if msg.contains("UTF-8")
        ));
    }

    #[test]
    fn validate_rejects_invalid_structured_json() {
        let record = MemoryRecord::new_binary(
            StoreType::Semantic,
            Modality::Structured,
            "application/json",
            b"{not valid json}".to_vec(),
            None,
        );

        let err = record.validate().unwrap_err();
        assert!(matches!(
            err,
            crate::error::CerememoryError::Validation(msg) if msg.contains("valid JSON")
        ));
    }

    #[test]
    fn memory_record_json_roundtrip() {
        let record = MemoryRecord::new_text(StoreType::Semantic, "Knowledge fact");
        let json = serde_json::to_string(&record).unwrap();
        let decoded: MemoryRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.text_content(), Some("Knowledge fact"));
    }

    #[test]
    fn memory_record_legacy_json_defaults_meta() {
        let record = MemoryRecord::new_text(StoreType::Semantic, "Legacy JSON");
        let mut json = serde_json::to_value(&record).unwrap();
        json.as_object_mut().unwrap().remove("meta");

        let decoded: MemoryRecord = serde_json::from_value(json).unwrap();

        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.text_content(), Some("Legacy JSON"));
        assert_eq!(decoded.meta.capture_status, MetaCaptureStatus::Legacy);
        assert!(decoded.meta.validate().is_ok());
    }

    #[test]
    fn memory_record_msgpack_roundtrip() {
        let record = MemoryRecord::new_text(StoreType::Working, "Active thought");
        let packed = rmp_serde::to_vec(&record).unwrap();
        let decoded: MemoryRecord = rmp_serde::from_slice(&packed).unwrap();
        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.text_content(), Some("Active thought"));
    }

    #[test]
    fn memory_record_legacy_msgpack_defaults_meta() {
        #[derive(serde::Serialize)]
        struct LegacyMemoryRecord {
            id: Uuid,
            store: StoreType,
            created_at: chrono::DateTime<chrono::Utc>,
            updated_at: chrono::DateTime<chrono::Utc>,
            last_accessed_at: chrono::DateTime<chrono::Utc>,
            access_count: u32,
            content: MemoryContent,
            fidelity: FidelityState,
            emotion: EmotionVector,
            associations: Vec<Association>,
            metadata: serde_json::Value,
            version: u32,
        }

        let record = MemoryRecord::new_text(StoreType::Working, "Legacy record");
        let legacy = LegacyMemoryRecord {
            id: record.id,
            store: record.store,
            created_at: record.created_at,
            updated_at: record.updated_at,
            last_accessed_at: record.last_accessed_at,
            access_count: record.access_count,
            content: record.content.clone(),
            fidelity: record.fidelity.clone(),
            emotion: record.emotion.clone(),
            associations: record.associations.clone(),
            metadata: record.metadata.clone(),
            version: record.version,
        };

        let packed = rmp_serde::to_vec(&legacy).unwrap();
        let decoded: MemoryRecord = rmp_serde::from_slice(&packed).unwrap();

        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.text_content(), Some("Legacy record"));
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.meta.capture_status, MetaCaptureStatus::Legacy);
        assert!(decoded.meta.validate().is_ok());
    }

    #[test]
    fn meta_memory_partial_json_defaults_to_current_schema() {
        let meta: MetaMemory =
            serde_json::from_str(r#"{"intent":"remember the rationale","confidence":0.8}"#)
                .unwrap();

        assert_eq!(meta.schema_version, 1);
        assert_eq!(meta.intent.as_deref(), Some("remember the rationale"));
        assert!(meta.rationale.is_none());
        assert_eq!(meta.confidence, Some(0.8));
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn fidelity_state_defaults_are_spec_compliant() {
        let f = FidelityState::default();
        assert_eq!(f.score, 1.0);
        assert_eq!(f.noise_level, 0.0);
        assert_eq!(f.decay_rate, 0.3);
        assert_eq!(f.emotional_anchor, 1.0);
        assert_eq!(f.stability, 1.0);
        assert_eq!(f.reinforcement_count, 0);
    }

    #[test]
    fn emotion_vector_default_is_neutral() {
        let e = EmotionVector::default();
        assert_eq!(e.intensity, 0.0);
        assert_eq!(e.valence, 0.0);
    }

    #[test]
    fn store_type_display() {
        assert_eq!(StoreType::Episodic.to_string(), "episodic");
        assert_eq!(StoreType::Working.to_string(), "working");
    }

    #[test]
    fn store_type_from_str_valid() {
        assert_eq!(
            "episodic".parse::<StoreType>().unwrap(),
            StoreType::Episodic
        );
        assert_eq!(
            "semantic".parse::<StoreType>().unwrap(),
            StoreType::Semantic
        );
        assert_eq!(
            "procedural".parse::<StoreType>().unwrap(),
            StoreType::Procedural
        );
        assert_eq!(
            "emotional".parse::<StoreType>().unwrap(),
            StoreType::Emotional
        );
        assert_eq!("working".parse::<StoreType>().unwrap(), StoreType::Working);
    }

    #[test]
    fn store_type_from_str_invalid() {
        let err = "unknown".parse::<StoreType>().unwrap_err();
        assert!(matches!(
            err,
            crate::error::CerememoryError::StoreInvalid(_)
        ));
    }

    #[test]
    fn emotion_from_str_valid_labels() {
        let joy: EmotionVector = "joy".parse().unwrap();
        assert_eq!(joy.joy, 1.0);
        assert_eq!(joy.intensity, 1.0);
        assert_eq!(joy.valence, 1.0);

        let sadness: EmotionVector = "sadness".parse().unwrap();
        assert_eq!(sadness.sadness, 1.0);
        assert_eq!(sadness.valence, -1.0);

        let anger: EmotionVector = "anger".parse().unwrap();
        assert_eq!(anger.anger, 1.0);
        assert_eq!(anger.valence, -0.9);
    }

    #[test]
    fn emotion_from_str_case_insensitive_and_aliases() {
        let happy: EmotionVector = "Happy".parse().unwrap();
        assert_eq!(happy.joy, 1.0);

        let sad: EmotionVector = "  SAD  ".parse().unwrap();
        assert_eq!(sad.sadness, 1.0);

        let angry: EmotionVector = "ANGRY".parse().unwrap();
        assert_eq!(angry.anger, 1.0);

        let anticipatory: EmotionVector = "Anticipatory".parse().unwrap();
        assert_eq!(anticipatory.anticipation, 1.0);
    }

    #[test]
    fn emotion_from_str_invalid() {
        let err = "unknown_emotion".parse::<EmotionVector>().unwrap_err();
        assert!(matches!(
            err,
            crate::error::CerememoryError::Validation(msg) if msg.contains("Invalid emotion label")
        ));
    }

    #[test]
    fn raw_journal_new_text_has_valid_defaults() {
        let record = RawJournalRecord::new_text(
            "sess-1",
            RawSource::Conversation,
            RawSpeaker::User,
            RawVisibility::Normal,
            SecrecyLevel::Public,
            "raw hello",
        );

        assert_eq!(record.session_id, "sess-1");
        assert_eq!(record.text_content(), Some("raw hello"));
        assert!(!record.suppressed);
        assert!(record.validate().is_ok());
    }

    #[test]
    fn raw_journal_legacy_json_defaults_meta() {
        let record = RawJournalRecord::new_text(
            "sess-json",
            RawSource::Conversation,
            RawSpeaker::Assistant,
            RawVisibility::Normal,
            SecrecyLevel::Public,
            "raw json",
        );
        let mut json = serde_json::to_value(&record).unwrap();
        json.as_object_mut().unwrap().remove("meta");

        let decoded: RawJournalRecord = serde_json::from_value(json).unwrap();

        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.session_id, "sess-json");
        assert_eq!(decoded.text_content(), Some("raw json"));
        assert_eq!(decoded.meta.capture_status, MetaCaptureStatus::Legacy);
        assert!(decoded.meta.validate().is_ok());
    }

    #[test]
    fn raw_journal_legacy_msgpack_defaults_meta() {
        #[derive(serde::Serialize)]
        struct LegacyRawJournalRecord {
            id: Uuid,
            session_id: String,
            turn_id: Option<String>,
            topic_id: Option<String>,
            source: RawSource,
            speaker: RawSpeaker,
            visibility: RawVisibility,
            secrecy_level: SecrecyLevel,
            created_at: chrono::DateTime<chrono::Utc>,
            updated_at: chrono::DateTime<chrono::Utc>,
            content: MemoryContent,
            metadata: serde_json::Value,
            derived_memory_ids: Vec<Uuid>,
            suppressed: bool,
        }

        let record = RawJournalRecord::new_text(
            "sess-legacy",
            RawSource::Conversation,
            RawSpeaker::Assistant,
            RawVisibility::Normal,
            SecrecyLevel::Public,
            "raw legacy",
        );
        let legacy = LegacyRawJournalRecord {
            id: record.id,
            session_id: record.session_id.clone(),
            turn_id: record.turn_id.clone(),
            topic_id: record.topic_id.clone(),
            source: record.source,
            speaker: record.speaker,
            visibility: record.visibility,
            secrecy_level: record.secrecy_level,
            created_at: record.created_at,
            updated_at: record.updated_at,
            content: record.content.clone(),
            metadata: record.metadata.clone(),
            derived_memory_ids: record.derived_memory_ids.clone(),
            suppressed: record.suppressed,
        };

        let packed = rmp_serde::to_vec(&legacy).unwrap();
        let decoded: RawJournalRecord = rmp_serde::from_slice(&packed).unwrap();

        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.session_id, "sess-legacy");
        assert_eq!(decoded.text_content(), Some("raw legacy"));
        assert_eq!(decoded.meta.capture_status, MetaCaptureStatus::Legacy);
        assert!(decoded.meta.validate().is_ok());
    }

    #[test]
    fn raw_journal_validate_rejects_blank_session() {
        let record = RawJournalRecord::new_text(
            "   ",
            RawSource::Conversation,
            RawSpeaker::Assistant,
            RawVisibility::PrivateScratch,
            SecrecyLevel::Sensitive,
            "scratch",
        );

        let err = record.validate().unwrap_err();
        assert!(matches!(
            err,
            crate::error::CerememoryError::Validation(msg) if msg.contains("session_id")
        ));
    }
}
