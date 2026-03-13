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
        Self::new_single_block(store, ContentBlock {
            modality,
            format: format.into(),
            data,
            embedding,
        })
    }

    /// Create a new text-only memory record with sensible defaults.
    pub fn new_text(store: StoreType, text: impl Into<String>) -> Self {
        Self::new_single_block(store, ContentBlock {
            modality: Modality::Text,
            format: "text/plain".to_string(),
            data: text.into().into_bytes(),
            embedding: None,
        })
    }

    /// Validate record invariants including modality-specific size limits.
    pub fn validate(&self) -> Result<(), crate::error::CerememoryError> {
        if self.content.blocks.is_empty() {
            return Err(crate::error::CerememoryError::Validation(
                "Record must have at least one content block".to_string(),
            ));
        }
        if self.fidelity.score < 0.0 || self.fidelity.score > 1.0 {
            return Err(crate::error::CerememoryError::Validation(
                format!("Fidelity score {} out of range [0.0, 1.0]", self.fidelity.score),
            ));
        }
        if self.fidelity.noise_level < 0.0 || self.fidelity.noise_level > 1.0 {
            return Err(crate::error::CerememoryError::Validation(
                format!("Noise level {} out of range [0.0, 1.0]", self.fidelity.noise_level),
            ));
        }
        for assoc in &self.associations {
            if assoc.weight < 0.0 || assoc.weight > 1.0 {
                return Err(crate::error::CerememoryError::Validation(
                    format!("Association weight {} out of range [0.0, 1.0]", assoc.weight),
                ));
            }
        }
        // Modality-specific size limits and embedding validation
        for block in &self.content.blocks {
            let limit = match block.modality {
                Modality::Text => MAX_TEXT_SIZE,
                Modality::Image => MAX_IMAGE_SIZE,
                Modality::Audio | Modality::Video => MAX_AUDIO_SIZE,
                Modality::Structured | Modality::Spatial | Modality::Temporal | Modality::Interoceptive => MAX_TEXT_SIZE,
            };
            if block.data.len() > limit {
                return Err(crate::error::CerememoryError::ContentTooLarge {
                    size: block.data.len(),
                    limit,
                });
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
        self.updated_at = Utc::now();
    }
}

/// Rough token estimate from byte count (1 token ≈ 4 bytes).
pub fn estimate_tokens_from_bytes(bytes: usize) -> usize {
    bytes.div_ceil(4)
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
    fn memory_record_json_roundtrip() {
        let record = MemoryRecord::new_text(StoreType::Semantic, "Knowledge fact");
        let json = serde_json::to_string(&record).unwrap();
        let decoded: MemoryRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.text_content(), Some("Knowledge fact"));
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
}
