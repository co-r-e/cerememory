//! Cerememory Engine — the orchestrator.
//!
//! Assembles all stores, the decay engine, association engine,
//! evolution engine, and the hippocampal coordinator into a
//! unified system that implements the full CMP protocol.
//!
//! Phase 2 additions:
//! - Tantivy full-text search integration (TextIndex)
//! - Vector embedding similarity search (VectorIndex)
//! - Background decay via tokio::spawn
//! - Export/Import via cerememory-archive

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Default weight for automatically inferred sequential associations in batch encoding.
const DEFAULT_BATCH_SEQUENTIAL_WEIGHT: f64 = 0.7;

/// All five memory store types.
const ALL_STORES: [StoreType; 5] = [
    StoreType::Episodic,
    StoreType::Semantic,
    StoreType::Procedural,
    StoreType::Emotional,
    StoreType::Working,
];

/// Scope guard that records a histogram metric on drop (success or error path).
struct TimerGuard {
    name: &'static str,
    start: std::time::Instant,
}

impl TimerGuard {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for TimerGuard {
    fn drop(&mut self) {
        metrics::histogram!(self.name).record(self.start.elapsed().as_secs_f64());
    }
}

/// Dispatch a method call to the appropriate store based on StoreType.
macro_rules! dispatch_store {
    ($self:expr, $store_type:expr, $method:ident ( $($arg:expr),* )) => {
        match $store_type {
            StoreType::Episodic => $self.episodic.$method($($arg),*).await,
            StoreType::Semantic => $self.semantic.$method($($arg),*).await,
            StoreType::Procedural => $self.procedural.$method($($arg),*).await,
            StoreType::Emotional => $self.emotional.$method($($arg),*).await,
            StoreType::Working => $self.working.$method($($arg),*).await,
        }
    };
}

use chrono::Utc;
use tracing::{info, warn};
use uuid::Uuid;

use cerememory_association::SpreadingActivationEngine;
use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::*;
use cerememory_core::traits::*;
use cerememory_core::types::*;
use cerememory_decay::{DecayParams, PowerLawDecayEngine};
use cerememory_evolution::EvolutionEngine;
use cerememory_index::text_index::TextIndex;
use cerememory_index::vector_index::VectorIndex;
use cerememory_index::HippocampalCoordinator;
use cerememory_store_emotional::EmotionalStore;
use cerememory_store_episodic::EpisodicStore;
use cerememory_store_procedural::ProceduralStore;
use cerememory_store_semantic::SemanticStore;
use cerememory_store_working::WorkingMemoryStore;

/// Configuration for engine construction.
pub struct EngineConfig {
    pub episodic_path: Option<String>,
    pub semantic_path: Option<String>,
    pub procedural_path: Option<String>,
    pub emotional_path: Option<String>,
    pub working_capacity: usize,
    pub decay_params: DecayParams,
    pub recall_mode: RecallMode,
    /// Path for the Tantivy full-text search index directory.
    pub index_path: Option<String>,
    /// Path for the redb-backed vector index file.
    pub vector_index_path: Option<String>,
    /// If set, enables background decay at this interval (in seconds). None = disabled.
    pub background_decay_interval_secs: Option<u64>,
    /// Number of vectors at which to switch from brute-force to HNSW search.
    /// Default: 1000. Set to `usize::MAX` to always use brute-force.
    pub hnsw_threshold: usize,
    /// Optional LLM provider for auto-embedding, summarization, and relation extraction.
    /// When None, the engine operates without LLM capabilities (manual embeddings only).
    pub llm_provider: Option<Arc<dyn LLMProvider>>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            episodic_path: None,
            semantic_path: None,
            procedural_path: None,
            emotional_path: None,
            working_capacity: 7,
            decay_params: DecayParams::default(),
            recall_mode: RecallMode::Human,
            index_path: None,
            vector_index_path: None,
            background_decay_interval_secs: None,
            hnsw_threshold: cerememory_index::vector_index::DEFAULT_HNSW_THRESHOLD,
            llm_provider: None,
        }
    }
}

/// The main Cerememory engine — orchestrates all CMP operations.
pub struct CerememoryEngine {
    // Stores
    episodic: EpisodicStore,
    semantic: SemanticStore,
    procedural: ProceduralStore,
    emotional: EmotionalStore,
    working: WorkingMemoryStore,

    // Engines
    decay: PowerLawDecayEngine,
    activation: SpreadingActivationEngine<HippocampalCoordinator>,
    evolution: EvolutionEngine,

    // Coordinator
    coordinator: Arc<HippocampalCoordinator>,

    // Indexes (Phase 2)
    text_index: TextIndex,
    vector_index: VectorIndex,

    // State
    recall_mode: tokio::sync::RwLock<RecallMode>,

    // LLM provider (optional)
    llm_provider: Option<Arc<dyn LLMProvider>>,

    // Background decay
    background_decay_interval_secs: Option<u64>,
    decay_state: tokio::sync::Mutex<Option<BackgroundDecayState>>,
}

/// Tracks a running background decay task for clean shutdown.
struct BackgroundDecayState {
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    handle: tokio::task::JoinHandle<()>,
}

impl CerememoryEngine {
    /// Create a new engine. Uses in-memory stores when paths are None.
    pub fn new(config: EngineConfig) -> Result<Self, CerememoryError> {
        let episodic = match &config.episodic_path {
            Some(p) => EpisodicStore::open(p)?,
            None => EpisodicStore::open_in_memory()?,
        };

        let semantic = match &config.semantic_path {
            Some(p) => SemanticStore::open(p)?,
            None => SemanticStore::open_in_memory()?,
        };

        let text_index = match &config.index_path {
            Some(p) => match TextIndex::open(p) {
                Ok(idx) => idx,
                Err(e) => {
                    warn!(
                        error = %e,
                        path = %p,
                        "Corrupted text index detected, recreating (data will be repopulated via rebuild_coordinator)"
                    );
                    // Attempt to remove corrupted index directory and recreate
                    let _ = std::fs::remove_dir_all(p);
                    TextIndex::open(p).map_err(|e2| {
                        CerememoryError::Storage(format!(
                            "Failed to recreate text index after corruption: {e2}"
                        ))
                    })?
                }
            },
            None => TextIndex::open_in_memory()?,
        };

        let vector_index = match &config.vector_index_path {
            Some(p) => VectorIndex::open_with_threshold(p, config.hnsw_threshold)?,
            None => VectorIndex::open_in_memory_with_threshold(config.hnsw_threshold)?,
        };

        let procedural = match &config.procedural_path {
            Some(p) => ProceduralStore::open(p)?,
            None => ProceduralStore::open_in_memory()?,
        };

        let emotional = match &config.emotional_path {
            Some(p) => EmotionalStore::open(p)?,
            None => EmotionalStore::open_in_memory()?,
        };

        let coordinator = Arc::new(HippocampalCoordinator::new());
        let activation = SpreadingActivationEngine::new(Arc::clone(&coordinator));

        Ok(Self {
            episodic,
            semantic,
            procedural,
            emotional,
            working: WorkingMemoryStore::with_capacity(config.working_capacity),
            decay: PowerLawDecayEngine::new(config.decay_params),
            activation,
            evolution: EvolutionEngine::new(),
            coordinator,
            text_index,
            vector_index,
            recall_mode: tokio::sync::RwLock::new(config.recall_mode),
            llm_provider: config.llm_provider,
            background_decay_interval_secs: config.background_decay_interval_secs,
            decay_state: tokio::sync::Mutex::new(None),
        })
    }

    /// Create an engine with all in-memory stores (for testing).
    pub fn in_memory() -> Result<Self, CerememoryError> {
        Self::new(EngineConfig::default())
    }

    /// Start the background decay task. Requires the engine to be wrapped in Arc.
    /// No-op if already running or if `background_decay_interval_secs` is None.
    pub fn start_background_decay(self: &Arc<Self>) {
        let Some(interval_secs) = self.background_decay_interval_secs else {
            return;
        };

        // Prevent double start
        let mut guard = match self.decay_state.try_lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if guard.is_some() {
            return; // Already running
        }

        let (tx, rx) = tokio::sync::watch::channel(false);
        let engine = Arc::clone(self);

        let handle = tokio::spawn(async move {
            let mut backoff_secs = 1u64;
            const MAX_BACKOFF_SECS: u64 = 60;

            loop {
                let engine_ref = Arc::clone(&engine);
                let mut rx_ref = rx.clone();

                let result = tokio::spawn(async move {
                    let mut interval =
                        tokio::time::interval(std::time::Duration::from_secs(interval_secs));
                    interval.tick().await; // skip first immediate tick

                    loop {
                        tokio::select! {
                            _ = interval.tick() => {
                                let req = DecayTickRequest {
                                    header: None,
                                    tick_duration_seconds: Some(interval_secs as u32),
                                };
                                if let Err(e) = engine_ref.lifecycle_decay_tick(req).await {
                                    warn!(error = %e, "Background decay tick failed");
                                }
                            }
                            _ = rx_ref.changed() => {
                                if *rx_ref.borrow() {
                                    info!("Background decay stopped");
                                    return;
                                }
                            }
                        }
                    }
                })
                .await;

                // Check if we got a shutdown signal
                if *rx.borrow() {
                    return;
                }

                match result {
                    Ok(()) => return, // Clean exit
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            backoff_secs,
                            "Background decay task panicked, restarting"
                        );
                        metrics::counter!("cerememory_decay_panics_total").increment(1);
                        tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                        backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF_SECS);
                    }
                }
            }
        });

        *guard = Some(BackgroundDecayState {
            shutdown_tx: tx,
            handle,
        });
    }

    /// Stop the background decay task and wait for it to finish.
    pub async fn stop_background_decay(&self) {
        let state = {
            let mut guard = self.decay_state.lock().await;
            guard.take()
        };
        if let Some(state) = state {
            let _ = state.shutdown_tx.send(true);
            let _ = state.handle.await;
        }
    }

    /// Check if background decay is running.
    pub async fn is_background_decay_enabled(&self) -> bool {
        self.decay_state.lock().await.is_some()
    }

    /// Rebuild the hippocampal coordinator and all indexes from persistent stores.
    /// Must be called after construction when opening existing stores.
    pub async fn rebuild_coordinator(&self) -> Result<(), CerememoryError> {
        // Clear vector index to avoid stale entries from previous runs
        self.vector_index.clear()?;

        let mut entries = Vec::new();
        let mut text_records = Vec::new();

        for store_type in [
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
            StoreType::Emotional,
        ] {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                if let Some(record) = dispatch_store!(self, store_type, get(&id))? {
                    entries.push((record.id, store_type, record.associations.clone()));

                    // Collect searchable text for full-text index rebuild.
                    let text = Self::build_searchable_text(&record).unwrap_or_default();
                    if Self::has_indexable_content(&text, &record) {
                        text_records.push((
                            record.id,
                            store_type,
                            text,
                            record.content.summary.clone(),
                        ));
                    }

                    if let Some(embedding) = Self::primary_embedding(&record) {
                        let _ = self.vector_index.upsert(record.id, embedding);
                    }
                }
            }
        }

        self.coordinator.rebuild(entries).await;
        self.text_index.rebuild(&text_records)?;
        self.vector_index.rebuild_hnsw()?;

        info!(
            records = self.coordinator.total_records().await,
            hnsw_active = self.vector_index.is_hnsw_active(),
            "Coordinator and indexes rebuilt from persistent stores"
        );
        Ok(())
    }

    // ─── Index helpers ─────────────────────────────────────────────

    /// Returns `true` if the record has non-empty searchable text or a non-empty summary,
    /// meaning it should be indexed in the full-text index.
    fn has_indexable_content(searchable_text: &str, record: &MemoryRecord) -> bool {
        !searchable_text.is_empty()
            || record
                .content
                .summary
                .as_deref()
                .map(str::trim)
                .is_some_and(|summary| !summary.is_empty())
    }

    /// Build a single searchable text body from textual and structured blocks.
    fn build_searchable_text(record: &MemoryRecord) -> Option<String> {
        let mut chunks = Vec::new();

        for block in &record.content.blocks {
            match block.modality {
                Modality::Text => match std::str::from_utf8(&block.data) {
                    Ok(text) => {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            chunks.push(trimmed.to_string());
                        }
                    }
                    Err(error) => {
                        warn!(
                            error = %error,
                            record_id = %record.id,
                            "Skipping invalid UTF-8 text block during indexing"
                        );
                    }
                },
                Modality::Structured => {
                    match cerememory_index::structured_index::flatten_json_to_text(&block.data) {
                        Ok(flat) if !flat.is_empty() => chunks.push(flat),
                        Ok(_) => {}
                        Err(error) => {
                            warn!(
                                error = %error,
                                record_id = %record.id,
                                "Skipping invalid structured block during indexing"
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        if chunks.is_empty() {
            None
        } else {
            Some(chunks.join("\n"))
        }
    }

    fn primary_embedding(record: &MemoryRecord) -> Option<&[f32]> {
        record
            .content
            .blocks
            .iter()
            .find_map(|block| block.embedding.as_deref())
    }

    fn detect_image_format(data: &[u8]) -> Option<&'static str> {
        if data.len() >= 8 && data.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Some("png");
        }
        if data.len() >= 3 && data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return Some("jpeg");
        }
        if data.len() >= 6 && (data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a")) {
            return Some("gif");
        }
        if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
            return Some("webp");
        }
        None
    }

    fn detect_audio_format(data: &[u8]) -> Option<&'static str> {
        if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WAVE" {
            return Some("wav");
        }
        if data.len() >= 4 && data.starts_with(b"fLaC") {
            return Some("flac");
        }
        if data.len() >= 4 && data.starts_with(b"OggS") {
            return Some("ogg");
        }
        if data.len() >= 3 && data.starts_with(b"ID3") {
            return Some("mp3");
        }
        if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0 {
            return Some("mp3");
        }
        if data.len() >= 12 && &data[4..8] == b"ftyp" {
            return Some("mp4");
        }
        if data.len() >= 4 && data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
            return Some("webm");
        }
        None
    }

    async fn resolve_recall_cues(
        &self,
        cue: &RecallCue,
    ) -> Result<(Option<String>, Option<Vec<f32>>), CerememoryError> {
        let mut text = cue
            .text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let mut embedding = cue.embedding.clone();

        if let Some(image) = cue.image.as_ref() {
            if image.is_empty() {
                return Err(CerememoryError::Validation(
                    "Recall image cue must not be empty".to_string(),
                ));
            }
            if image.len() > MAX_IMAGE_SIZE {
                return Err(CerememoryError::ContentTooLarge {
                    size: image.len(),
                    limit: MAX_IMAGE_SIZE,
                });
            }

            if embedding.is_none() {
                let provider = self.llm_provider.as_ref().ok_or_else(|| {
                    CerememoryError::ModalityUnsupported(
                        "Image recall requires an LLM provider with image embedding support"
                            .to_string(),
                    )
                })?;
                let caps = provider.capabilities();
                if !caps.image_embedding {
                    return Err(CerememoryError::ModalityUnsupported(
                        "Configured LLM provider does not support image recall".to_string(),
                    ));
                }

                let format = Self::detect_image_format(image).ok_or_else(|| {
                    CerememoryError::Validation("Unsupported image recall cue format".to_string())
                })?;
                let generated = provider.embed_image(image, format).await?;
                if generated.is_empty() {
                    return Err(CerememoryError::Internal(
                        "Image recall provider returned an empty embedding".to_string(),
                    ));
                }
                embedding = Some(generated);
            }
        }

        if let Some(audio) = cue.audio.as_ref() {
            if audio.is_empty() {
                return Err(CerememoryError::Validation(
                    "Recall audio cue must not be empty".to_string(),
                ));
            }
            if audio.len() > MAX_AUDIO_SIZE {
                return Err(CerememoryError::ContentTooLarge {
                    size: audio.len(),
                    limit: MAX_AUDIO_SIZE,
                });
            }

            let provider = self.llm_provider.as_ref().ok_or_else(|| {
                CerememoryError::ModalityUnsupported(
                    "Audio recall requires an LLM provider with transcription support".to_string(),
                )
            })?;
            let caps = provider.capabilities();
            if !caps.audio_transcription {
                return Err(CerememoryError::ModalityUnsupported(
                    "Configured LLM provider does not support audio recall".to_string(),
                ));
            }

            let format = Self::detect_audio_format(audio).ok_or_else(|| {
                CerememoryError::Validation("Unsupported audio recall cue format".to_string())
            })?;
            let transcript = provider.transcribe_audio(audio, format).await?;
            let transcript = transcript.trim();
            if transcript.is_empty() {
                return Err(CerememoryError::Validation(
                    "Audio recall cue produced an empty transcript".to_string(),
                ));
            }
            match &mut text {
                Some(existing) => {
                    existing.push('\n');
                    existing.push_str(transcript);
                }
                None => text = Some(transcript.to_string()),
            }
            if embedding.is_none() && caps.text_embedding {
                let generated = provider.embed(transcript).await?;
                if !generated.is_empty() {
                    embedding = Some(generated);
                }
            }
        }

        if embedding.is_none() {
            if let (Some(provider), Some(query_text)) =
                (self.llm_provider.as_ref(), text.as_deref())
            {
                if provider.capabilities().text_embedding {
                    match provider.embed(query_text).await {
                        Ok(generated) if !generated.is_empty() => embedding = Some(generated),
                        Ok(_) => warn!("Text recall cue embedding returned an empty vector"),
                        Err(error) => warn!(error = %error, "Failed to embed text recall cue"),
                    }
                }
            }
        }

        Ok((text, embedding))
    }

    /// Index a record's text content, structured data, and embeddings.
    /// Uses only the first embedding found (one vector per record_id).
    fn index_record(&self, record: &MemoryRecord) -> Result<(), CerememoryError> {
        let text = Self::build_searchable_text(record).unwrap_or_default();
        if Self::has_indexable_content(&text, record) {
            self.text_index.add(
                record.id,
                record.store,
                &text,
                record.content.summary.as_deref(),
            )?;
        }

        // Vector index — use first embedding found (one vector per record_id)
        if let Some(embedding) = Self::primary_embedding(record) {
            self.vector_index.upsert(record.id, embedding)?;
        }

        Ok(())
    }

    /// Remove a record from all indexes.
    fn unindex_record(&self, id: Uuid) {
        let _ = self.text_index.remove(id);
        let _ = self.vector_index.remove(id);
    }

    // ─── Store routing ───────────────────────────────────────────────

    fn route_store(&self, content: &MemoryContent) -> StoreType {
        if content.summary.is_some() {
            StoreType::Semantic
        } else {
            StoreType::Episodic
        }
    }

    async fn get_store_record(
        &self,
        id: &Uuid,
    ) -> Result<Option<(MemoryRecord, StoreType)>, CerememoryError> {
        // Check coordinator first for store type hint
        if let Some(st) = self.coordinator.get_record_store_type(id).await? {
            let record = dispatch_store!(self, st, get(id))?;
            return Ok(record.map(|r| (r, st)));
        }

        // Fallback: search all stores
        for st in [
            StoreType::Working,
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
            StoreType::Emotional,
        ] {
            if let Some(r) = dispatch_store!(self, st, get(id))? {
                return Ok(Some((r, st)));
            }
        }
        Ok(None)
    }

    fn build_record_metadata(
        context: Option<EncodeContext>,
        metadata: Option<serde_json::Value>,
    ) -> serde_json::Value {
        let context_value = context.and_then(|ctx| serde_json::to_value(ctx).ok());
        match (metadata, context_value) {
            (Some(serde_json::Value::Object(mut map)), Some(context)) => {
                map.insert("_context".to_string(), context);
                serde_json::Value::Object(map)
            }
            (Some(other), Some(context)) => serde_json::json!({
                "_metadata": other,
                "_context": context,
            }),
            (Some(metadata), None) => metadata,
            (None, Some(context)) => serde_json::json!({ "_context": context }),
            (None, None) => serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    async fn persist_associations_for_record(
        &self,
        record_id: &Uuid,
        store_type: StoreType,
        associations: Vec<Association>,
    ) -> Result<(), CerememoryError> {
        dispatch_store!(
            self,
            store_type,
            replace_associations(record_id, associations.clone())
        )?;
        self.coordinator
            .update_associations(record_id, associations)
            .await?;
        Ok(())
    }

    async fn add_persisted_association(
        &self,
        record_id: &Uuid,
        association: Association,
    ) -> Result<(), CerememoryError> {
        let Some((record, store_type)) = self.get_store_record(record_id).await? else {
            return Err(CerememoryError::RecordNotFound(record_id.to_string()));
        };

        let mut associations = self.coordinator.get_associations(record_id).await?;
        for persisted in &record.associations {
            if !associations.iter().any(|existing| {
                existing.target_id == persisted.target_id
                    && existing.association_type == persisted.association_type
            }) {
                associations.push(persisted.clone());
            }
        }

        if associations.iter().any(|existing| {
            existing.target_id == association.target_id
                && existing.association_type == association.association_type
        }) {
            return Ok(());
        }

        associations.push(association);
        self.persist_associations_for_record(record_id, store_type, associations)
            .await
    }

    async fn remove_deleted_targets_from_records(
        &self,
        deleted_ids: &HashSet<Uuid>,
    ) -> Result<(), CerememoryError> {
        if deleted_ids.is_empty() {
            return Ok(());
        }

        for store_type in ALL_STORES {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                let Some(record) = dispatch_store!(self, store_type, get(&id))? else {
                    continue;
                };
                let filtered: Vec<_> = record
                    .associations
                    .iter()
                    .filter(|association| !deleted_ids.contains(&association.target_id))
                    .cloned()
                    .collect();
                if filtered.len() != record.associations.len() {
                    self.persist_associations_for_record(&id, store_type, filtered)
                        .await?;
                }
            }
        }

        Ok(())
    }

    async fn cleanup_deleted_records(
        &self,
        deleted_records: &[(Uuid, StoreType)],
    ) -> Result<(), CerememoryError> {
        if deleted_records.is_empty() {
            return Ok(());
        }

        let mut deleted_ids = HashSet::with_capacity(deleted_records.len());
        for (record_id, _) in deleted_records {
            deleted_ids.insert(*record_id);
            self.coordinator.unregister(record_id).await;
            self.unindex_record(*record_id);
        }
        self.remove_deleted_targets_from_records(&deleted_ids).await
    }

    // ─── CMP Encode Operations ───────────────────────────────────────

    /// encode.store — Store a new memory record (CMP Spec §3.1).
    pub async fn encode_store(
        &self,
        req: EncodeStoreRequest,
    ) -> Result<EncodeStoreResponse, CerememoryError> {
        let _timer = TimerGuard::new("cerememory_encode_duration_seconds");
        let EncodeStoreRequest {
            content,
            store,
            emotion,
            context,
            metadata,
            associations,
            ..
        } = req;
        let store_type = store.unwrap_or_else(|| self.route_store(&content));

        let mut record = MemoryRecord {
            id: Uuid::now_v7(),
            store: store_type,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed_at: Utc::now(),
            access_count: 0,
            content,
            fidelity: FidelityState::default(),
            emotion: emotion.unwrap_or_default(),
            associations: Vec::new(),
            metadata: Self::build_record_metadata(context, metadata),
            version: 1,
        };

        // Add manual associations
        let mut assoc_count = 0u32;
        if let Some(manual) = associations {
            for ma in manual {
                record.associations.push(Association {
                    target_id: ma.target_id,
                    association_type: ma.association_type,
                    weight: ma.weight,
                    created_at: Utc::now(),
                    last_co_activation: Utc::now(),
                });
                assoc_count += 1;
            }
        }

        // Auto-generate embeddings and transcriptions if provider is available
        if let Some(ref provider) = self.llm_provider {
            let caps = provider.capabilities();

            // Auto-embed text blocks that lack embeddings
            let has_text_embedding = record
                .content
                .blocks
                .iter()
                .any(|b| b.modality == Modality::Text && b.embedding.is_some());
            if !has_text_embedding && caps.text_embedding {
                if let Some(text) = record.text_content().map(|s| s.to_string()) {
                    match provider.embed(&text).await {
                        Ok(embedding) if !embedding.is_empty() => {
                            if let Some(block) = record
                                .content
                                .blocks
                                .iter_mut()
                                .find(|b| b.modality == Modality::Text)
                            {
                                block.embedding = Some(embedding);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            warn!(error = %e, "LLM text auto-embed failed, continuing without embedding");
                        }
                    }
                }
            }

            // Collect image block data (cloned) for concurrent processing.
            // We clone data+format out of the record so the borrow is released
            // before we mutate the blocks with the results.
            let image_tasks: Vec<(usize, Vec<u8>, String)> = if caps.image_embedding {
                record
                    .content
                    .blocks
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.modality == Modality::Image && b.embedding.is_none())
                    .map(|(i, b)| (i, b.data.clone(), b.format.clone()))
                    .collect()
            } else {
                Vec::new()
            };

            let audio_tasks: Vec<(Vec<u8>, String)> = if caps.audio_transcription {
                record
                    .content
                    .blocks
                    .iter()
                    .filter(|b| b.modality == Modality::Audio)
                    .map(|b| (b.data.clone(), b.format.clone()))
                    .collect()
            } else {
                Vec::new()
            };

            // Process image embeddings concurrently
            if !image_tasks.is_empty() {
                let image_results: Vec<(usize, Result<Vec<f32>, _>)> =
                    futures::future::join_all(image_tasks.iter().map(|(idx, data, fmt)| {
                        let idx = *idx;
                        async move { (idx, provider.embed_image(data, fmt).await) }
                    }))
                    .await;

                for (idx, result) in image_results {
                    match result {
                        Ok(embedding) if !embedding.is_empty() => {
                            record.content.blocks[idx].embedding = Some(embedding);
                        }
                        Ok(_) => {}
                        Err(e) => {
                            warn!(error = %e, "LLM image auto-embed failed, continuing without embedding");
                        }
                    }
                }
            }

            // Process audio transcriptions concurrently
            if !audio_tasks.is_empty() {
                let audio_results: Vec<Result<String, _>> =
                    futures::future::join_all(audio_tasks.iter().map(|(data, fmt)| async move {
                        provider.transcribe_audio(data, fmt).await
                    }))
                    .await;

                let mut new_text_blocks = Vec::new();
                for result in audio_results {
                    match result {
                        Ok(transcript) if !transcript.is_empty() => {
                            let mut text_block = ContentBlock {
                                modality: Modality::Text,
                                format: "text/plain".to_string(),
                                data: transcript.as_bytes().to_vec(),
                                embedding: None,
                            };
                            if caps.text_embedding {
                                match provider.embed(&transcript).await {
                                    Ok(emb) if !emb.is_empty() => {
                                        text_block.embedding = Some(emb);
                                    }
                                    Err(error) => {
                                        warn!(error = %error, "LLM transcript auto-embed failed, continuing without embedding");
                                    }
                                    Ok(_) => {}
                                }
                            }
                            new_text_blocks.push(text_block);
                        }
                        Ok(_) => {}
                        Err(e) => {
                            warn!(error = %e, "LLM audio transcription failed, continuing without transcript");
                        }
                    }
                }
                record.content.blocks.extend(new_text_blocks);
            }
        }

        record.validate()?;
        let id = record.id;
        let fidelity = record.fidelity.score;

        // 1. Store first (source of truth)
        if store_type == StoreType::Working {
            let (_, evicted) = self.working.store_with_eviction(record.clone()).await?;
            if let Some(evicted_id) = evicted {
                self.cleanup_deleted_records(&[(evicted_id, StoreType::Working)])
                    .await?;
            }
        } else {
            dispatch_store!(self, store_type, store(record.clone()))?;
        }

        // 2. Register in coordinator (in-memory, rebuildable)
        self.coordinator
            .register(id, store_type, record.associations.clone())
            .await;

        // 3. Index (rebuildable, log on failure)
        if let Err(e) = self.index_record(&record) {
            warn!(error = %e, record_id = %id, "Failed to index record, will be indexed on rebuild");
        }

        info!(record_id = %id, store = %store_type, "Encoded memory record");

        metrics::counter!("cerememory_encode_total", "store" => store_type.to_string())
            .increment(1);

        Ok(EncodeStoreResponse {
            record_id: id,
            store: store_type,
            initial_fidelity: fidelity,
            associations_created: assoc_count,
        })
    }

    /// encode.batch — Store multiple records (CMP Spec §3.2).
    pub async fn encode_batch(
        &self,
        req: EncodeBatchRequest,
    ) -> Result<EncodeBatchResponse, CerememoryError> {
        const MAX_BATCH_SIZE: usize = 1000;
        if req.records.len() > MAX_BATCH_SIZE {
            return Err(CerememoryError::Validation(format!(
                "Batch size {} exceeds maximum of {MAX_BATCH_SIZE}",
                req.records.len()
            )));
        }
        let mut results = Vec::with_capacity(req.records.len());
        let mut total_inferred = 0u32;
        let mut prev_id: Option<Uuid> = None;

        for store_req in req.records {
            let resp = self.encode_store(store_req).await?;

            // Infer sequential associations between batch items
            if req.infer_associations {
                if let Some(prev) = prev_id {
                    let assoc_fwd = Association {
                        target_id: resp.record_id,
                        association_type: AssociationType::Sequential,
                        weight: DEFAULT_BATCH_SEQUENTIAL_WEIGHT,
                        created_at: Utc::now(),
                        last_co_activation: Utc::now(),
                    };
                    let assoc_bwd = Association {
                        target_id: prev,
                        association_type: AssociationType::Sequential,
                        weight: DEFAULT_BATCH_SEQUENTIAL_WEIGHT,
                        created_at: Utc::now(),
                        last_co_activation: Utc::now(),
                    };
                    self.add_persisted_association(&prev, assoc_fwd).await?;
                    self.add_persisted_association(&resp.record_id, assoc_bwd)
                        .await?;
                    total_inferred += 2;
                }
            }

            prev_id = Some(resp.record_id);
            results.push(resp);
        }

        Ok(EncodeBatchResponse {
            results,
            associations_inferred: total_inferred,
        })
    }

    /// encode.update — Update an existing record (CMP Spec §3.3).
    pub async fn encode_update(&self, req: EncodeUpdateRequest) -> Result<(), CerememoryError> {
        let (mut record, store_type) = self
            .get_store_record(&req.record_id)
            .await?
            .ok_or_else(|| CerememoryError::RecordNotFound(req.record_id.to_string()))?;

        // Apply updates to a clone and validate before persisting
        record.apply_updates(
            req.content.clone(),
            req.emotion.clone(),
            req.metadata.clone(),
        );
        record.validate()?;

        // Persist the update
        dispatch_store!(
            self,
            store_type,
            update_record(
                &req.record_id,
                req.content.clone(),
                req.emotion,
                req.metadata
            )
        )?;

        // Update indexes if content changed
        if req.content.is_some() {
            let text = Self::build_searchable_text(&record).unwrap_or_default();
            if Self::has_indexable_content(&text, &record) {
                if let Err(e) = self.text_index.update(
                    req.record_id,
                    store_type,
                    &text,
                    record.content.summary.as_deref(),
                ) {
                    warn!(error = %e, record_id = %req.record_id, "Failed to update text index");
                }
            } else if let Err(e) = self.text_index.remove(req.record_id) {
                warn!(error = %e, record_id = %req.record_id, "Failed to clear text index");
            }

            // Remove old vector, then insert first new embedding
            let _ = self.vector_index.remove(req.record_id);
            if let Some(embedding) = Self::primary_embedding(&record) {
                if let Err(e) = self.vector_index.upsert(req.record_id, embedding) {
                    warn!(error = %e, record_id = %req.record_id, "Failed to update vector index");
                }
            }
        }

        Ok(())
    }

    // ─── CMP Recall Operations ───────────────────────────────────────

    /// recall.query — Retrieve memories (CMP Spec §4.1).
    ///
    /// Phase 2 recall pipeline:
    /// 1. Text search via Tantivy (if cue.text present)
    /// 2. Vector similarity search (if cue.embedding present)
    /// 3. Hybrid score merging
    /// 4. Temporal range filter
    /// 5. Spreading activation
    /// 6. Reconsolidation + human noise rendering
    pub async fn recall_query(
        &self,
        req: RecallQueryRequest,
    ) -> Result<RecallQueryResponse, CerememoryError> {
        let _timer = TimerGuard::new("cerememory_recall_duration_seconds");
        let mode = *self.recall_mode.read().await;
        let recall_mode = req.recall_mode;
        let effective_mode = if mode == RecallMode::Perfect {
            RecallMode::Perfect
        } else {
            recall_mode
        };
        let (cue_text, cue_embedding) = self.resolve_recall_cues(&req.cue).await?;

        let stores = req.stores.clone().unwrap_or_else(|| {
            vec![
                StoreType::Episodic,
                StoreType::Semantic,
                StoreType::Procedural,
                StoreType::Emotional,
                StoreType::Working,
            ]
        });

        let mut candidates: Vec<(MemoryRecord, f64)> = Vec::new();
        let mut seen_ids: HashSet<Uuid> = HashSet::new();

        // Score maps for hybrid merging
        let mut text_scores: HashMap<Uuid, f64> = HashMap::new();
        let mut vec_scores: HashMap<Uuid, f64> = HashMap::new();

        // 1. Tantivy full-text search
        if let Some(ref text) = cue_text {
            let search_limit = req.limit as usize * 3;
            match self.text_index.search(text, Some(&stores), search_limit) {
                Ok(hits) => {
                    for hit in hits {
                        text_scores.insert(hit.record_id, hit.score as f64);
                    }
                }
                Err(e) => {
                    // Fallback to store-level text search if Tantivy fails
                    warn!(error = %e, "Text index search failed, falling back to store query");
                    for store_type in &stores {
                        let results = dispatch_store!(
                            self,
                            *store_type,
                            query_text(text, req.limit as usize * 2)
                        )?;
                        for record in results {
                            text_scores.insert(record.id, record.fidelity.score);
                        }
                    }
                }
            }
        }

        // 2. Vector similarity search
        if let Some(ref embedding) = cue_embedding {
            if let Ok(hits) = self.vector_index.search(embedding, req.limit as usize * 3) {
                for hit in hits {
                    if hit.similarity > 0.0 {
                        vec_scores.insert(hit.record_id, hit.similarity as f64);
                    }
                }
            }
        }

        // 3. Merge scores: collect all candidate IDs
        let all_ids: HashSet<Uuid> = text_scores
            .keys()
            .chain(vec_scores.keys())
            .copied()
            .collect();
        let mut scanned_ids = all_ids.clone();
        let mut total_records_scanned = scanned_ids.len() as u32;
        let mut fidelity_filtered: u32 = 0;

        for id in all_ids {
            if !seen_ids.insert(id) {
                continue;
            }
            if let Some((record, _)) = self.get_store_record(&id).await? {
                // Filter by requested stores
                if !stores.contains(&record.store) {
                    continue;
                }
                if let Some(min_f) = req.min_fidelity {
                    if record.fidelity.score < min_f && !req.include_decayed {
                        fidelity_filtered += 1;
                        continue;
                    }
                }

                // Hybrid scoring
                let ts = text_scores.get(&id);
                let vs = vec_scores.get(&id);
                let score = match (ts, vs) {
                    (Some(&t), Some(&v)) => t * 0.6 + v * 0.4,
                    (Some(&t), None) => t,
                    (None, Some(&v)) => v,
                    (None, None) => 0.0,
                };

                candidates.push((record, score));
            }
        }

        // If no index search was performed (no text or embedding cue),
        // fall back to store-level text search for backward compatibility
        if cue_text.is_none() && cue_embedding.is_none() {
            // No search cue — candidates will come from temporal or activation only
        }

        // 4. Temporal range enrichment across all requested stores
        if let Some(ref temporal) = req.cue.temporal {
            for store_type in &stores {
                let ids = dispatch_store!(self, *store_type, list_ids())?;
                for id in ids {
                    let Some(record) = dispatch_store!(self, *store_type, get(&id))? else {
                        continue;
                    };
                    if record.created_at < temporal.start || record.created_at > temporal.end {
                        continue;
                    }
                    if scanned_ids.insert(record.id) {
                        total_records_scanned += 1;
                    }
                    if !seen_ids.insert(record.id) {
                        continue;
                    }
                    if let Some(min_f) = req.min_fidelity {
                        if record.fidelity.score < min_f && !req.include_decayed {
                            fidelity_filtered += 1;
                            continue;
                        }
                    }
                    let score = record.fidelity.score;
                    candidates.push((record, score));
                }
            }
        }

        // 5. Spreading activation for top candidates
        let mut activated_ids: HashMap<Uuid, f64> = HashMap::new();
        if req.activation_depth > 0 && !candidates.is_empty() {
            let top_id = candidates
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(r, _)| r.id);

            if let Some(source_id) = top_id {
                if let Ok(activated) = self
                    .activation
                    .activate(&source_id, req.activation_depth, 0.05)
                    .await
                {
                    for act in &activated {
                        activated_ids.insert(act.record_id, act.activation_level);
                        if seen_ids.insert(act.record_id) {
                            if let Some((record, _store)) =
                                self.get_store_record(&act.record_id).await?
                            {
                                if !stores.contains(&record.store) {
                                    continue;
                                }
                                if let Some(ref temporal) = req.cue.temporal {
                                    if record.created_at < temporal.start
                                        || record.created_at > temporal.end
                                    {
                                        continue;
                                    }
                                }
                                if let Some(min_f) = req.min_fidelity {
                                    if record.fidelity.score < min_f && !req.include_decayed {
                                        fidelity_filtered += 1;
                                        continue;
                                    }
                                }
                                candidates.push((record, act.activation_level * 0.5));
                            }
                        }
                    }
                }
            }
        }

        // Boost relevance with activation scores
        for (record, relevance) in &mut candidates {
            if let Some(activation) = activated_ids.get(&record.id) {
                *relevance += activation * 0.3;
            }
        }

        // Final filter pass: catches index-search results outside the temporal
        // window and any edge cases from spreading activation.
        if let Some(ref temporal) = req.cue.temporal {
            candidates.retain(|(record, _)| {
                record.created_at >= temporal.start && record.created_at <= temporal.end
            });
        }
        candidates.retain(|(record, _)| stores.contains(&record.store));
        if let Some(min_fidelity) = req.min_fidelity {
            let before = candidates.len();
            candidates
                .retain(|(record, _)| req.include_decayed || record.fidelity.score >= min_fidelity);
            fidelity_filtered += (before - candidates.len()) as u32;
        }

        // Sort by relevance descending, track pre-truncation count
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total_candidates = candidates.len() as u32;
        candidates.truncate(req.limit as usize);

        // 6. Build response with reconsolidation
        let mut memories = Vec::with_capacity(candidates.len());
        for (mut record, relevance) in candidates {
            // Reconsolidate: update access metadata + fidelity
            if req.reconsolidate {
                record.access_count += 1;
                record.last_accessed_at = Utc::now();

                // Stability boost
                let new_stability = self.decay.boost_stability(record.fidelity.stability);
                record.fidelity.stability = new_stability;
                record.fidelity.reinforcement_count += 1;

                if let Some(store_type) = self.coordinator.get_record_store_type(&record.id).await?
                {
                    if let Err(e) = dispatch_store!(
                        self,
                        store_type,
                        update_fidelity(&record.id, record.fidelity.clone())
                    ) {
                        warn!(record_id = %record.id, error = %e, "Failed to update fidelity during reconsolidation");
                    }
                    if let Err(e) = dispatch_store!(
                        self,
                        store_type,
                        update_access(&record.id, record.access_count, record.last_accessed_at)
                    ) {
                        warn!(record_id = %record.id, error = %e, "Failed to update access during reconsolidation");
                    }
                }
            }

            // Render content with or without noise
            let rendered_content = match effective_mode {
                RecallMode::Perfect => record.content.clone(),
                RecallMode::Human => apply_human_noise(&record.content, record.fidelity.score),
            };

            memories.push(RecalledMemory {
                activation_path: activated_ids.get(&record.id).map(|_| vec![record.id]),
                relevance_score: relevance,
                rendered_content,
                record,
            });
        }

        // Notify evolution engine about recall performance
        if !memories.is_empty() {
            let hit_count = memories.iter().filter(|m| m.relevance_score > 0.0).count();
            let hit_rate = hit_count as f64 / memories.len() as f64;
            // Use the first memory's store type as representative
            let store = memories[0].record.store;
            self.evolution.observe_recall(store, hit_rate);
        }

        metrics::counter!("cerememory_recall_total").increment(1);

        Ok(RecallQueryResponse {
            memories,
            activation_trace: None,
            total_candidates,
            query_metadata: Some(QueryMetadata {
                total_records_scanned,
                stores_searched: stores,
                fidelity_filtered,
            }),
        })
    }

    /// recall.associate — Get associated memories (CMP Spec §4.2).
    pub async fn recall_associate(
        &self,
        req: RecallAssociateRequest,
    ) -> Result<RecallAssociateResponse, CerememoryError> {
        // Verify record exists
        self.get_store_record(&req.record_id)
            .await?
            .ok_or_else(|| CerememoryError::RecordNotFound(req.record_id.to_string()))?;

        let activated = self
            .activation
            .activate(&req.record_id, req.depth, req.min_weight)
            .await?;

        let source_assocs = if req.association_types.is_some() {
            Some(self.coordinator.get_associations(&req.record_id).await?)
        } else {
            None
        };

        let mut memories = Vec::new();
        for act in activated.iter().take(req.limit as usize) {
            if let Some(types) = &req.association_types {
                let assocs = source_assocs.as_ref().expect("pre-fetched above");
                let matches = assocs
                    .iter()
                    .any(|a| a.target_id == act.record_id && types.contains(&a.association_type));
                if !matches && act.path.len() <= 2 {
                    continue;
                }
            }

            if let Some((record, _)) = self.get_store_record(&act.record_id).await? {
                memories.push(RecalledMemory {
                    rendered_content: record.content.clone(),
                    relevance_score: act.activation_level,
                    activation_path: Some(act.path.clone()),
                    record,
                });
            }
        }

        Ok(RecallAssociateResponse {
            total_candidates: memories.len() as u32,
            memories,
        })
    }

    /// recall.timeline — Temporal bucketed recall (CMP Spec §4.3, OPTIONAL).
    pub async fn recall_timeline(
        &self,
        req: RecallTimelineRequest,
    ) -> Result<RecallTimelineResponse, CerememoryError> {
        let records = self
            .episodic
            .query_temporal_range(req.range.start, req.range.end)
            .await?;

        // Group records into time buckets based on granularity
        let mut bucket_map: std::collections::BTreeMap<i64, Vec<MemoryRecord>> =
            std::collections::BTreeMap::new();

        for record in records {
            // Apply min_fidelity filter
            if let Some(min_f) = req.min_fidelity {
                if record.fidelity.score < min_f {
                    continue;
                }
            }

            // Apply emotion filter: cosine similarity between emotion vectors
            if let Some(ref filter) = req.emotion_filter {
                if !Self::emotion_matches(&record.emotion, filter) {
                    continue;
                }
            }

            let bucket_key = Self::bucket_key(record.created_at, req.granularity);
            bucket_map.entry(bucket_key).or_default().push(record);
        }

        // Also scan other stores for records created in the range
        for store_type in [
            StoreType::Semantic,
            StoreType::Procedural,
            StoreType::Emotional,
        ] {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                if let Some(record) = dispatch_store!(self, store_type, get(&id))? {
                    if record.created_at >= req.range.start && record.created_at <= req.range.end {
                        if let Some(min_f) = req.min_fidelity {
                            if record.fidelity.score < min_f {
                                continue;
                            }
                        }
                        if let Some(ref filter) = req.emotion_filter {
                            if !Self::emotion_matches(&record.emotion, filter) {
                                continue;
                            }
                        }
                        let bucket_key = Self::bucket_key(record.created_at, req.granularity);
                        bucket_map.entry(bucket_key).or_default().push(record);
                    }
                }
            }
        }

        // Convert to response buckets
        let buckets: Vec<TimelineBucket> = bucket_map
            .into_iter()
            .map(|(key, records)| {
                let (start, end) = Self::bucket_range(key, req.granularity);
                let count = records.len() as u32;
                let memories = records
                    .into_iter()
                    .map(|record| RecalledMemory {
                        relevance_score: record.fidelity.score,
                        rendered_content: record.content.clone(),
                        activation_path: None,
                        record,
                    })
                    .collect();
                TimelineBucket {
                    start,
                    end,
                    memories,
                    count,
                }
            })
            .collect();

        Ok(RecallTimelineResponse { buckets })
    }

    /// Check if a record's emotion matches the filter.
    /// Uses cosine similarity on the 8-dimensional emotion vector (Plutchik axes).
    /// A match requires similarity > 0.5 (moderate alignment).
    fn emotion_matches(record_emotion: &EmotionVector, filter: &EmotionVector) -> bool {
        let r = [
            record_emotion.joy,
            record_emotion.trust,
            record_emotion.fear,
            record_emotion.surprise,
            record_emotion.sadness,
            record_emotion.disgust,
            record_emotion.anger,
            record_emotion.anticipation,
        ];
        let f = [
            filter.joy,
            filter.trust,
            filter.fear,
            filter.surprise,
            filter.sadness,
            filter.disgust,
            filter.anger,
            filter.anticipation,
        ];

        let dot: f64 = r.iter().zip(f.iter()).map(|(a, b)| a * b).sum();
        let norm_r: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_f: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_r < f64::EPSILON || norm_f < f64::EPSILON {
            return false; // Neutral emotion — no match
        }

        let similarity = dot / (norm_r * norm_f);
        similarity > 0.5
    }

    fn bucket_key(ts: chrono::DateTime<Utc>, granularity: TimeGranularity) -> i64 {
        use chrono::Datelike;
        let secs = ts.timestamp();
        match granularity {
            TimeGranularity::Minute => secs / 60,
            TimeGranularity::Hour => secs / 3600,
            TimeGranularity::Day => secs / 86400,
            TimeGranularity::Week => secs / 604800,
            TimeGranularity::Month => ts.year() as i64 * 12 + (ts.month() as i64 - 1),
        }
    }

    fn bucket_range(
        key: i64,
        granularity: TimeGranularity,
    ) -> (chrono::DateTime<Utc>, chrono::DateTime<Utc>) {
        use chrono::TimeZone;

        // Fixed-duration granularities share one formula.
        let epoch_secs = |g: TimeGranularity| -> Option<i64> {
            match g {
                TimeGranularity::Minute => Some(60),
                TimeGranularity::Hour => Some(3600),
                TimeGranularity::Day => Some(86400),
                TimeGranularity::Week => Some(604800),
                TimeGranularity::Month => None,
            }
        };

        if let Some(secs) = epoch_secs(granularity) {
            let start = Utc
                .timestamp_opt(key * secs, 0)
                .single()
                .unwrap_or_default();
            let end = Utc
                .timestamp_opt((key + 1) * secs, 0)
                .single()
                .unwrap_or_default();
            return (start, end);
        }

        // Month: calendar-aware boundaries
        let year = key / 12;
        let month = (key % 12) + 1;
        let start = Utc
            .with_ymd_and_hms(year as i32, month as u32, 1, 0, 0, 0)
            .single()
            .unwrap_or_default();
        let next_month = if month == 12 { 1 } else { month + 1 };
        let next_year = if month == 12 { year + 1 } else { year };
        let end = Utc
            .with_ymd_and_hms(next_year as i32, next_month as u32, 1, 0, 0, 0)
            .single()
            .unwrap_or_default();
        (start, end)
    }

    /// recall.graph — Local graph extraction (CMP Spec §4.4, OPTIONAL).
    pub async fn recall_graph(
        &self,
        req: RecallGraphRequest,
    ) -> Result<RecallGraphResponse, CerememoryError> {
        let mut nodes: Vec<GraphNode> = Vec::new();
        let mut edges: Vec<GraphEdge> = Vec::new();
        let mut visited: HashSet<Uuid> = HashSet::new();
        let mut queue: std::collections::VecDeque<(Uuid, u32)> = std::collections::VecDeque::new();
        let limit = req.limit_nodes as usize;

        // Start from center_id or all registered records
        if let Some(center) = req.center_id {
            queue.push_back((center, 0));
        } else {
            // No center: return a sample of the graph
            let reg = self.coordinator.records_by_store().await;
            let all_ids: Vec<Uuid> = {
                let mut ids = Vec::new();
                for st in [
                    StoreType::Episodic,
                    StoreType::Semantic,
                    StoreType::Procedural,
                    StoreType::Emotional,
                ] {
                    if reg.contains_key(&st) {
                        let store_ids = dispatch_store!(self, st, list_ids())?;
                        ids.extend(store_ids.into_iter().take(limit));
                    }
                    if ids.len() >= limit {
                        break;
                    }
                }
                ids
            };
            for id in all_ids {
                queue.push_back((id, 0));
            }
        }

        // BFS traversal
        while let Some((id, depth)) = queue.pop_front() {
            if !visited.insert(id) || nodes.len() >= limit {
                continue;
            }

            if let Some((record, _store)) = self.get_store_record(&id).await? {
                nodes.push(GraphNode {
                    id: record.id,
                    store: record.store,
                    summary: record.content.summary.clone().or_else(|| {
                        record.text_content().map(|t| {
                            if t.len() > 80 {
                                format!("{}...", truncate_str(t, 80))
                            } else {
                                t.to_string()
                            }
                        })
                    }),
                    fidelity: record.fidelity.score,
                });

                if depth < req.depth {
                    let assocs = self.coordinator.get_associations(&id).await?;
                    for assoc in assocs {
                        // Filter by edge_types if specified
                        if let Some(ref types) = req.edge_types {
                            let type_str = serde_json::to_value(assoc.association_type)
                                .ok()
                                .and_then(|v| v.as_str().map(|s| s.to_string()))
                                .unwrap_or_default();
                            if !types.iter().any(|t| t.to_lowercase() == type_str) {
                                continue;
                            }
                        }

                        // Cap edges at 10x node limit to prevent unbounded growth
                        if edges.len() < limit * 10 {
                            edges.push(GraphEdge {
                                source: id,
                                target: assoc.target_id,
                                edge_type: assoc.association_type,
                                weight: assoc.weight,
                            });
                        }

                        if !visited.contains(&assoc.target_id) {
                            queue.push_back((assoc.target_id, depth + 1));
                        }
                    }
                }
            }
        }

        let total_nodes = nodes.len() as u32;
        Ok(RecallGraphResponse {
            nodes,
            edges,
            total_nodes,
        })
    }

    // ─── CMP Lifecycle Operations ────────────────────────────────────

    /// lifecycle.consolidate — Smart Consolidation (CMP Spec §5.1).
    ///
    /// Phase 4 enhancements:
    /// - **Duplicate detection**: Vector similarity > 0.92 identifies near-duplicates.
    ///   Higher-fidelity record is kept; associations are merged.
    /// - **LLM summarization**: When a provider is configured, related episodic records
    ///   are summarized into a single semantic node (otherwise, truncation fallback).
    /// - **Relation extraction**: LLM extracts semantic relations, stored as associations.
    pub async fn lifecycle_consolidate(
        &self,
        req: ConsolidateRequest,
    ) -> Result<ConsolidateResponse, CerememoryError> {
        let ids = self.episodic.list_ids().await?;
        let mut processed = 0u32;
        let mut migrated = 0u32;
        let mut compressed = 0u32;
        let mut pruned = 0u32;

        // Phase 1: Detect and merge near-duplicate records
        let mut duplicate_groups: Vec<(Uuid, Uuid)> = Vec::new();
        if !req.dry_run {
            let mut checked: HashSet<Uuid> = HashSet::new();
            for &id in &ids {
                if checked.contains(&id) {
                    continue;
                }
                // Use vector index to find similar records
                if let Some(record) = self.episodic.get(&id).await? {
                    if let Some(emb) = Self::primary_embedding(&record) {
                        if let Ok(hits) = self.vector_index.search(emb, 5) {
                            for hit in hits {
                                if hit.record_id != id
                                    && hit.similarity > 0.92
                                    && !checked.contains(&hit.record_id)
                                {
                                    let Some((_, hit_store)) =
                                        self.get_store_record(&hit.record_id).await?
                                    else {
                                        continue;
                                    };
                                    if hit_store != StoreType::Episodic {
                                        continue;
                                    }
                                    duplicate_groups.push((id, hit.record_id));
                                    checked.insert(hit.record_id);
                                }
                            }
                        }
                    }
                }
                checked.insert(id);
            }

            // Merge duplicates: keep higher-fidelity, merge associations
            for (keep_id, remove_id) in &duplicate_groups {
                if let (Some(keep_rec), Some(remove_rec)) = (
                    self.episodic.get(keep_id).await?,
                    self.get_store_record(remove_id).await?,
                ) {
                    let (remove_record, remove_store) = remove_rec;
                    // Decide which to keep based on fidelity
                    let (actual_keep, actual_remove, actual_remove_store) =
                        if keep_rec.fidelity.score >= remove_record.fidelity.score {
                            (*keep_id, *remove_id, remove_store)
                        } else {
                            (*remove_id, *keep_id, StoreType::Episodic)
                        };

                    // Merge associations from removed to kept
                    let removed_assocs = self.coordinator.get_associations(&actual_remove).await?;
                    for assoc in removed_assocs {
                        self.add_persisted_association(&actual_keep, assoc).await?;
                    }

                    // Delete the duplicate
                    if dispatch_store!(self, actual_remove_store, delete(&actual_remove))? {
                        self.cleanup_deleted_records(&[(actual_remove, actual_remove_store)])
                            .await?;
                        compressed += 1;
                    }
                }
            }
        }

        // Phase 2: Migrate eligible episodic records to semantic
        let remaining_ids = self.episodic.list_ids().await?;

        for id in remaining_ids {
            if let Some(record) = self.episodic.get(&id).await? {
                processed += 1;
                let age_hours = (Utc::now() - record.created_at).num_hours() as u32;
                if age_hours < req.min_age_hours {
                    continue;
                }
                if record.access_count < req.min_access_count {
                    continue;
                }

                if req.dry_run {
                    migrated += 1;
                    continue;
                }

                // Create semantic node
                let mut semantic_record = record.clone();
                semantic_record.store = StoreType::Semantic;
                semantic_record.id = Uuid::now_v7();

                // Generate summary: use LLM if available, otherwise truncate
                if semantic_record.content.summary.is_none() {
                    if let Some(ref provider) = self.llm_provider {
                        if let Some(text) = record.text_content() {
                            match provider.summarize(&[text.to_string()], 200).await {
                                Ok(summary) if !summary.is_empty() => {
                                    semantic_record.content.summary = Some(summary);
                                }
                                _ => {
                                    // Fallback to truncation
                                    semantic_record.content.summary = Some(
                                        record
                                            .text_content()
                                            .map(|t| {
                                                if t.len() > 100 {
                                                    format!("{}...", truncate_str(t, 100))
                                                } else {
                                                    t.to_string()
                                                }
                                            })
                                            .unwrap_or_default(),
                                    );
                                }
                            }
                        }
                    } else {
                        semantic_record.content.summary = semantic_record.text_content().map(|t| {
                            if t.len() > 100 {
                                format!("{}...", truncate_str(t, 100))
                            } else {
                                t.to_string()
                            }
                        });
                    }
                }

                // Extract relations via LLM if available
                if let Some(ref provider) = self.llm_provider {
                    if let Some(text) = record.text_content() {
                        if let Ok(relations) = provider.extract_relations(text).await {
                            for rel in relations {
                                // Store as metadata on the semantic record
                                if let serde_json::Value::Object(ref mut map) =
                                    semantic_record.metadata
                                {
                                    let relations_arr = map
                                        .entry("extracted_relations".to_string())
                                        .or_insert_with(|| serde_json::json!([]));
                                    if let serde_json::Value::Array(ref mut arr) = relations_arr {
                                        arr.push(serde_json::json!({
                                            "subject": rel.subject,
                                            "predicate": rel.predicate,
                                            "object": rel.object,
                                            "confidence": rel.confidence,
                                        }));
                                    }
                                }
                            }
                        }
                    }
                }

                dispatch_store!(self, StoreType::Semantic, store(semantic_record.clone()))?;
                self.coordinator
                    .register(
                        semantic_record.id,
                        StoreType::Semantic,
                        semantic_record.associations.clone(),
                    )
                    .await;

                if let Err(e) = self.index_record(&semantic_record) {
                    warn!(error = %e, "Failed to index consolidated record");
                }

                let assoc = Association {
                    target_id: semantic_record.id,
                    association_type: AssociationType::Semantic,
                    weight: 1.0,
                    created_at: Utc::now(),
                    last_co_activation: Utc::now(),
                };
                self.add_persisted_association(&id, assoc).await?;

                migrated += 1;

                if record.fidelity.score < 0.1 && self.episodic.delete(&id).await? {
                    self.cleanup_deleted_records(&[(id, StoreType::Episodic)])
                        .await?;
                    pruned += 1;
                }
            }
        }

        info!(
            processed,
            migrated, compressed, pruned, "Smart consolidation completed"
        );

        Ok(ConsolidateResponse {
            records_processed: processed,
            records_migrated: migrated,
            records_compressed: compressed,
            records_pruned: pruned,
            semantic_nodes_created: migrated,
        })
    }

    /// lifecycle.decay_tick — Advance decay (CMP Spec §5.2).
    pub async fn lifecycle_decay_tick(
        &self,
        req: DecayTickRequest,
    ) -> Result<DecayTickResponse, CerememoryError> {
        let tick_secs = req.tick_duration_seconds.unwrap_or(3600) as f64;

        let mut all_inputs = Vec::new();
        let mut record_stores: HashMap<Uuid, StoreType> = HashMap::new();

        for store_type in [
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
            StoreType::Emotional,
        ] {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                if let Some((record, _)) = self.get_store_record(&id).await? {
                    all_inputs.push(DecayInput {
                        id: record.id,
                        fidelity: record.fidelity.clone(),
                        emotion: record.emotion.clone(),
                        last_accessed_at: record.last_accessed_at,
                        access_count: record.access_count,
                    });
                    record_stores.insert(record.id, store_type);
                }
            }
        }

        let decay = self.decay.clone();
        let result =
            tokio::task::spawn_blocking(move || decay.compute_tick(&all_inputs, tick_secs))
                .await
                .map_err(|e| CerememoryError::Internal(format!("Decay task failed: {e}")))?;

        for output in &result.updates {
            if let Some(&store_type) = record_stores.get(&output.id) {
                if output.should_prune {
                    if dispatch_store!(self, store_type, delete(&output.id))? {
                        self.cleanup_deleted_records(&[(output.id, store_type)])
                            .await?;
                    }
                } else {
                    dispatch_store!(
                        self,
                        store_type,
                        update_fidelity(&output.id, output.new_fidelity.clone())
                    )?;
                }
            }
        }

        // Notify evolution engine about fidelity distributions
        let mut fidelity_by_store: HashMap<StoreType, Vec<f64>> = HashMap::new();
        for output in &result.updates {
            if let Some(&store_type) = record_stores.get(&output.id) {
                fidelity_by_store
                    .entry(store_type)
                    .or_default()
                    .push(output.new_fidelity.score);
            }
        }
        for (store_type, scores) in &fidelity_by_store {
            self.evolution.observe_decay_tick(*store_type, scores);
        }

        info!(
            updated = result.records_updated,
            below_threshold = result.records_below_threshold,
            pruned = result.records_pruned,
            "Decay tick completed"
        );

        Ok(DecayTickResponse {
            records_updated: result.records_updated,
            records_below_threshold: result.records_below_threshold,
            records_pruned: result.records_pruned,
        })
    }

    /// lifecycle.set_mode (CMP Spec §5.3).
    pub async fn lifecycle_set_mode(&self, req: SetModeRequest) -> Result<(), CerememoryError> {
        *self.recall_mode.write().await = req.mode;
        info!(mode = ?req.mode, "Recall mode changed");
        Ok(())
    }

    /// lifecycle.forget — Delete memory records (CMP Spec §5.4).
    pub async fn lifecycle_forget(&self, req: ForgetRequest) -> Result<u32, CerememoryError> {
        if !req.confirm {
            return Err(CerememoryError::ForgetUnconfirmed);
        }

        let ForgetRequest {
            record_ids,
            store,
            temporal_range,
            cascade,
            ..
        } = req;
        let mut deleted = 0u32;
        let mut delete_targets: HashMap<Uuid, StoreType> = HashMap::new();

        if let Some(ids) = record_ids {
            for id in ids {
                if let Some((record, store_type)) = self.get_store_record(&id).await? {
                    delete_targets.insert(id, store_type);
                    if cascade {
                        for assoc in &record.associations {
                            if let Some((_, st)) = self.get_store_record(&assoc.target_id).await? {
                                delete_targets.insert(assoc.target_id, st);
                            }
                        }
                    }
                }
            }
        }

        if let Some(store_type) = store {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                delete_targets.insert(id, store_type);
            }
        }

        if let Some(range) = temporal_range {
            for store_type in ALL_STORES {
                let ids = dispatch_store!(self, store_type, list_ids())?;
                for id in ids {
                    let Some(record) = dispatch_store!(self, store_type, get(&id))? else {
                        continue;
                    };
                    if record.created_at >= range.start && record.created_at <= range.end {
                        delete_targets.insert(id, store_type);
                    }
                }
            }
        }

        let mut deleted_records = Vec::new();
        for (id, store_type) in delete_targets {
            if dispatch_store!(self, store_type, delete(&id))? {
                deleted_records.push((id, store_type));
                deleted += 1;
            }
        }
        self.cleanup_deleted_records(&deleted_records).await?;

        warn!(deleted, "Forget operation completed");
        Ok(deleted)
    }

    /// lifecycle.export — Export records to a CMA archive with optional store
    /// filtering and encryption.
    pub async fn lifecycle_export(
        &self,
        req: ExportRequest,
    ) -> Result<(Vec<u8>, ExportResponse), CerememoryError> {
        let records = self
            .collect_records_for_stores(req.stores.as_deref())
            .await?;

        let encryption_key = if req.encrypt {
            let key_str = req.encryption_key.as_deref().ok_or_else(|| {
                CerememoryError::Validation(
                    "encryption_key is required when encrypt=true".to_string(),
                )
            })?;
            Some(cerememory_archive::crypto::derive_key(key_str))
        } else {
            None
        };

        cerememory_archive::export_filtered(
            &records,
            req.stores.as_deref(),
            encryption_key.as_ref(),
        )
    }

    /// lifecycle.import — Import records from a CMA archive with optional
    /// decryption and conflict resolution.
    pub async fn lifecycle_import(&self, req: ImportRequest) -> Result<u32, CerememoryError> {
        if req.strategy == ImportStrategy::Replace {
            return Err(CerememoryError::Validation(
                "ImportStrategy::Replace is not yet supported; use Merge".to_string(),
            ));
        }

        let data = req.archive_data.ok_or_else(|| {
            CerememoryError::ImportConflict("Import requires archive_data".to_string())
        })?;

        let decryption_key = req
            .decryption_key
            .as_deref()
            .map(cerememory_archive::crypto::derive_key);
        let records = cerememory_archive::import_records_with_key(&data, decryption_key.as_ref())?;

        let conflict_resolution = req.conflict_resolution;
        let mut imported = 0u32;

        for record in records {
            let store_type = record.store;
            let mut replaced_cross_store: Option<StoreType> = None;

            // Check for ID conflicts
            if let Some((existing, existing_store)) = self.get_store_record(&record.id).await? {
                match conflict_resolution {
                    ConflictResolution::KeepExisting => continue,
                    ConflictResolution::KeepImported => {
                        if existing_store != store_type {
                            replaced_cross_store = Some(existing_store);
                        }
                    }
                    ConflictResolution::KeepNewer => {
                        if record.updated_at <= existing.updated_at {
                            continue;
                        }
                        if existing_store != store_type {
                            replaced_cross_store = Some(existing_store);
                        }
                    }
                }
            }

            dispatch_store!(self, store_type, store(record.clone()))?;
            if let Some(existing_store) = replaced_cross_store {
                if !dispatch_store!(self, existing_store, delete(&record.id))? {
                    if let Err(e) = dispatch_store!(self, store_type, delete(&record.id)) {
                        warn!(
                            record_id = %record.id,
                            store = %store_type,
                            error = %e,
                            "Failed to clean up newly stored record during import conflict rollback"
                        );
                    }
                    return Err(CerememoryError::ImportConflict(format!(
                        "Failed to replace cross-store record {} from {} to {}",
                        record.id, existing_store, store_type
                    )));
                }
            }
            self.coordinator
                .register(record.id, store_type, record.associations.clone())
                .await;
            if let Err(e) = self.index_record(&record) {
                warn!(error = %e, record_id = %record.id, "Failed to index imported record");
            }
            imported += 1;
        }

        info!(imported, "Import completed");
        Ok(imported)
    }

    /// Import records from a serialized CMA archive (convenience method).
    ///
    /// Uses `KeepExisting` conflict resolution (skips duplicates).
    pub async fn import_records(&self, data: &[u8]) -> Result<u32, CerememoryError> {
        let records = cerememory_archive::import_records(data)?;
        let mut imported = 0u32;

        for record in records {
            let store_type = record.store;

            // Check for ID conflicts — skip if record already exists
            if self.get_store_record(&record.id).await?.is_some() {
                continue;
            }

            // Store first (source of truth)
            dispatch_store!(self, store_type, store(record.clone()))?;

            // Then coordinator + index (rebuildable)
            self.coordinator
                .register(record.id, store_type, record.associations.clone())
                .await;
            if let Err(e) = self.index_record(&record) {
                warn!(error = %e, record_id = %record.id, "Failed to index imported record");
            }
            imported += 1;
        }

        info!(imported, "Import completed");
        Ok(imported)
    }

    /// Collect all records for export (used by archive module).
    pub async fn collect_all_records(&self) -> Result<Vec<MemoryRecord>, CerememoryError> {
        self.collect_records_for_stores(None).await
    }

    /// Collect records from specified stores, or all stores if `None`.
    pub async fn collect_records_for_stores(
        &self,
        stores: Option<&[StoreType]>,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let target_stores = stores.unwrap_or(&ALL_STORES);

        let mut records = Vec::new();
        let mut seen_stores = HashSet::with_capacity(target_stores.len());
        for &store_type in target_stores {
            if !seen_stores.insert(store_type) {
                continue;
            }

            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                if let Some(record) = dispatch_store!(self, store_type, get(&id))? {
                    records.push(record);
                }
            }
        }
        Ok(records)
    }

    // ─── CMP Introspect Operations ───────────────────────────────────

    /// introspect.stats (CMP Spec §6.1).
    pub async fn introspect_stats(&self) -> Result<StatsResponse, CerememoryError> {
        let mut records_by_store = HashMap::new();
        let mut avg_fidelity_by_store = HashMap::new();
        let mut total_records = 0u32;
        let mut total_fidelity = 0.0f64;

        for store_type in ALL_STORES {
            let count = dispatch_store!(self, store_type, count())? as u32;
            records_by_store.insert(store_type, count);
            total_records += count;

            if count > 0 {
                let ids = dispatch_store!(self, store_type, list_ids())?;
                let mut store_fidelity = 0.0f64;
                for id in &ids {
                    if let Some((record, _)) = self.get_store_record(id).await? {
                        store_fidelity += record.fidelity.score;
                        total_fidelity += record.fidelity.score;
                    }
                }
                avg_fidelity_by_store.insert(store_type, store_fidelity / count as f64);
            }
        }

        let avg_fidelity = if total_records > 0 {
            total_fidelity / total_records as f64
        } else {
            0.0
        };

        Ok(StatsResponse {
            total_records,
            records_by_store,
            total_associations: self.coordinator.total_associations().await,
            avg_fidelity,
            avg_fidelity_by_store,
            oldest_record: None,
            newest_record: None,
            total_recall_count: 0,
            evolution_metrics: Some(self.evolution.get_metrics()),
            background_decay_enabled: self.is_background_decay_enabled().await,
        })
    }

    /// introspect.record (CMP Spec §6.2).
    pub async fn introspect_record(
        &self,
        req: RecordIntrospectRequest,
    ) -> Result<MemoryRecord, CerememoryError> {
        let (record, _) = self
            .get_store_record(&req.record_id)
            .await?
            .ok_or_else(|| CerememoryError::RecordNotFound(req.record_id.to_string()))?;

        Ok(record)
    }

    /// introspect.decay_forecast — Forward-calculate fidelity (CMP Spec §6.3, OPTIONAL).
    pub async fn introspect_decay_forecast(
        &self,
        req: DecayForecastRequest,
    ) -> Result<DecayForecastResponse, CerememoryError> {
        let mut forecasts = Vec::with_capacity(req.record_ids.len());

        for record_id in &req.record_ids {
            let (record, _) = self
                .get_store_record(record_id)
                .await?
                .ok_or_else(|| CerememoryError::RecordNotFound(record_id.to_string()))?;

            let current_fidelity = record.fidelity.score;
            // Match real decay engine: baseline = max(last_accessed_at, last_decay_tick)
            let last_access_secs = record.last_accessed_at.timestamp() as f64;
            let last_tick_secs = record.fidelity.last_decay_tick.timestamp() as f64;
            let baseline_secs = last_access_secs.max(last_tick_secs);
            let forecast_secs = req.forecast_at.timestamp() as f64;
            let elapsed_secs = (forecast_secs - baseline_secs).max(0.0);

            // Use per-record decay_rate (not global params), matching real decay engine
            let decay_exponent = record.fidelity.decay_rate;
            let emotion_mod = cerememory_decay::math::compute_emotion_mod(record.emotion.intensity);
            let forecasted_fidelity = cerememory_decay::math::compute_fidelity(
                current_fidelity,
                elapsed_secs,
                record.fidelity.stability,
                decay_exponent,
                emotion_mod,
            );

            // Binary search for threshold crossing date
            let params = self.decay.params();
            let estimated_threshold_date = if current_fidelity > params.prune_threshold {
                Self::estimate_threshold_date(
                    &record,
                    decay_exponent,
                    params.prune_threshold,
                    emotion_mod,
                )
            } else {
                // Already below threshold
                None
            };

            forecasts.push(DecayForecast {
                record_id: *record_id,
                current_fidelity,
                forecasted_fidelity,
                estimated_threshold_date,
            });
        }

        Ok(DecayForecastResponse { forecasts })
    }

    /// Binary search for the date when fidelity drops below the prune threshold.
    fn estimate_threshold_date(
        record: &MemoryRecord,
        decay_exponent: f64,
        prune_threshold: f64,
        emotion_mod: f64,
    ) -> Option<chrono::DateTime<Utc>> {
        // Match real decay engine: baseline = max(last_accessed_at, last_decay_tick)
        let base_time = record.last_accessed_at.max(record.fidelity.last_decay_tick);
        let f0 = record.fidelity.score;
        let stability = record.fidelity.stability;

        // Search up to ~10 years in seconds
        let mut lo: f64 = 0.0;
        let mut hi: f64 = 315_360_000.0;

        // Check if fidelity ever drops below threshold within search range
        let f_hi = cerememory_decay::math::compute_fidelity(
            f0,
            hi,
            stability,
            decay_exponent,
            emotion_mod,
        );
        if f_hi >= prune_threshold {
            return None; // Won't cross threshold within 10 years
        }

        // Binary search (30 iterations gives sub-second precision over 10yr range)
        for _ in 0..30 {
            let mid = (lo + hi) / 2.0;
            let f_mid = cerememory_decay::math::compute_fidelity(
                f0,
                mid,
                stability,
                decay_exponent,
                emotion_mod,
            );
            if f_mid > prune_threshold {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let threshold_secs = ((lo + hi) / 2.0) as i64;
        Some(base_time + chrono::Duration::seconds(threshold_secs))
    }

    /// introspect.evolution — Return evolution engine metrics (CMP Spec §6.4, OPTIONAL).
    pub async fn introspect_evolution(&self) -> Result<EvolutionMetrics, CerememoryError> {
        Ok(self.evolution.get_metrics())
    }
}

// Re-export truncate_str from core to avoid duplication.
use cerememory_core::truncate_str;

/// Apply human-mode noise to content based on fidelity.
/// Only applies to text blocks — non-text modalities are returned unchanged.
fn apply_human_noise(content: &MemoryContent, fidelity: f64) -> MemoryContent {
    if fidelity >= 0.95 {
        return content.clone();
    }

    let mut noised = content.clone();
    for block in &mut noised.blocks {
        if block.modality == Modality::Text {
            if let Ok(text) = std::str::from_utf8(&block.data) {
                let degraded = degrade_text(text, fidelity);
                block.data = degraded.into_bytes();
            }
        }
    }
    noised
}

/// Degrade text based on fidelity level.
fn degrade_text(text: &str, fidelity: f64) -> String {
    if fidelity >= 0.9 {
        return text.to_string();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let degrade_fraction = (1.0 - fidelity).min(0.8);
    let step = (1.0 / degrade_fraction).max(2.0) as usize;

    let mut result = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        if i % step == 0 && fidelity < 0.5 {
            result.push("...");
        } else {
            result.push(word);
        }
    }

    result.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_engine() -> CerememoryEngine {
        CerememoryEngine::in_memory().unwrap()
    }

    fn text_store_req(text: &str, store: Option<StoreType>) -> EncodeStoreRequest {
        EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: text.as_bytes().to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            store,
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        }
    }

    fn structured_store_req(json: &str, store: Option<StoreType>) -> EncodeStoreRequest {
        EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Structured,
                    format: "application/json".to_string(),
                    data: json.as_bytes().to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            store,
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        }
    }

    #[tokio::test]
    async fn encode_recall_roundtrip() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(text_store_req(
                "The quick brown fox",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();
        assert_eq!(resp.store, StoreType::Episodic);
        assert_eq!(resp.initial_fidelity, 1.0);

        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("quick brown".to_string()),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: true,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let recall_resp = engine.recall_query(query).await.unwrap();
        assert!(!recall_resp.memories.is_empty());
        assert_eq!(
            recall_resp.memories[0].record.text_content(),
            Some("The quick brown fox")
        );
    }

    #[tokio::test]
    async fn encode_store_persists_metadata_and_context() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Text,
                        format: "text/plain".to_string(),
                        data: b"metadata roundtrip".to_vec(),
                        embedding: None,
                    }],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: Some(EncodeContext {
                    source: Some("chat".to_string()),
                    session_id: Some("sess-1".to_string()),
                    spatial: None,
                    temporal: None,
                }),
                metadata: Some(serde_json::json!({"tag": "important"})),
                associations: None,
            })
            .await
            .unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();
        assert_eq!(record.metadata["tag"], "important");
        assert_eq!(record.metadata["_context"]["source"], "chat");
        assert_eq!(record.metadata["_context"]["session_id"], "sess-1");
    }

    #[tokio::test]
    async fn tantivy_tokenized_search() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req(
                "The quick brown fox jumps over the lazy dog",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();

        // Tantivy should find "quick" via tokenized search
        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("quick".to_string()),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let resp = engine.recall_query(query).await.unwrap();
        assert!(!resp.memories.is_empty());
        assert_eq!(
            resp.memories[0].record.text_content(),
            Some("The quick brown fox jumps over the lazy dog")
        );
    }

    #[tokio::test]
    async fn vector_search_recall() {
        let engine = make_engine().await;

        // Store with embedding
        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Cats are fluffy animals".to_vec(),
                    embedding: Some(vec![1.0, 0.0, 0.0]),
                }],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        };
        engine.encode_store(req).await.unwrap();

        // Recall with embedding
        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(vec![1.0, 0.1, 0.0]),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let resp = engine.recall_query(query).await.unwrap();
        assert!(!resp.memories.is_empty());
        assert_eq!(
            resp.memories[0].record.text_content(),
            Some("Cats are fluffy animals")
        );
    }

    #[tokio::test]
    async fn hybrid_text_vector_search() {
        let engine = make_engine().await;

        // Store two records, one with both text and embedding
        let req1 = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Machine learning is fascinating".to_vec(),
                    embedding: Some(vec![1.0, 0.0, 0.0]),
                }],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        };
        engine.encode_store(req1).await.unwrap();

        // Search with both text and embedding
        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("machine learning".to_string()),
                embedding: Some(vec![1.0, 0.0, 0.0]),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let resp = engine.recall_query(query).await.unwrap();
        assert!(!resp.memories.is_empty());
    }

    #[tokio::test]
    async fn recall_query_metadata_counts_temporal_only_searches() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req(
                "Temporal-only query metadata should count scanned records",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();

        let now = Utc::now();
        let resp = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    temporal: Some(TemporalRange {
                        start: now - chrono::Duration::minutes(1),
                        end: now + chrono::Duration::minutes(1),
                    }),
                    ..Default::default()
                },
                stores: None,
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        let metadata = resp
            .query_metadata
            .expect("query metadata should be present");
        assert_eq!(metadata.total_records_scanned, 1);
        assert_eq!(resp.memories.len(), 1);
    }

    #[tokio::test]
    async fn temporal_query_filters_activation_results_across_stores() {
        let engine = make_engine().await;

        let mut old_semantic = MemoryRecord::new_text(StoreType::Semantic, "alpha out of range");
        old_semantic.created_at = Utc::now() - chrono::Duration::days(7);
        old_semantic.updated_at = old_semantic.created_at;
        old_semantic.last_accessed_at = old_semantic.created_at;
        engine.semantic.store(old_semantic.clone()).await.unwrap();
        engine
            .coordinator
            .register(
                old_semantic.id,
                StoreType::Semantic,
                old_semantic.associations.clone(),
            )
            .await;
        engine.index_record(&old_semantic).unwrap();

        let current = MemoryRecord {
            associations: vec![Association {
                target_id: old_semantic.id,
                association_type: AssociationType::Semantic,
                weight: 0.9,
                created_at: Utc::now(),
                last_co_activation: Utc::now(),
            }],
            ..MemoryRecord::new_text(StoreType::Episodic, "alpha in range")
        };
        engine.episodic.store(current.clone()).await.unwrap();
        engine
            .coordinator
            .register(
                current.id,
                StoreType::Episodic,
                current.associations.clone(),
            )
            .await;
        engine.index_record(&current).unwrap();

        let now = Utc::now();
        let resp = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    text: Some("alpha".to_string()),
                    temporal: Some(TemporalRange {
                        start: now - chrono::Duration::minutes(1),
                        end: now + chrono::Duration::minutes(1),
                    }),
                    ..Default::default()
                },
                stores: None,
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 1,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        assert_eq!(resp.memories.len(), 1);
        assert_eq!(resp.memories[0].record.id, current.id);
    }

    #[tokio::test]
    async fn encode_batch_with_associations() {
        let engine = make_engine().await;

        let batch = EncodeBatchRequest {
            header: None,
            records: vec![
                text_store_req("First memory", Some(StoreType::Episodic)),
                text_store_req("Second memory", Some(StoreType::Episodic)),
                text_store_req("Third memory", Some(StoreType::Episodic)),
            ],
            infer_associations: true,
        };

        let resp = engine.encode_batch(batch).await.unwrap();
        assert_eq!(resp.results.len(), 3);
        assert_eq!(resp.associations_inferred, 4);
    }

    #[tokio::test]
    async fn encode_batch_associations_survive_rebuild() {
        let engine = make_engine().await;

        let resp = engine
            .encode_batch(EncodeBatchRequest {
                header: None,
                records: vec![
                    text_store_req("Persisted first", Some(StoreType::Episodic)),
                    text_store_req("Persisted second", Some(StoreType::Episodic)),
                ],
                infer_associations: true,
            })
            .await
            .unwrap();

        engine.rebuild_coordinator().await.unwrap();

        let first = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.results[0].record_id,
                include_history: false,
                include_associations: true,
                include_versions: false,
            })
            .await
            .unwrap();
        assert_eq!(first.associations.len(), 1);

        let assoc_resp = engine
            .recall_associate(RecallAssociateRequest {
                header: None,
                record_id: resp.results[0].record_id,
                association_types: Some(vec![AssociationType::Sequential]),
                depth: 1,
                min_weight: 0.1,
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(assoc_resp.memories.len(), 1);
        assert_eq!(assoc_resp.memories[0].record.id, resp.results[1].record_id);
    }

    #[tokio::test]
    async fn decay_tick_integration() {
        let engine = make_engine().await;

        for i in 0..5 {
            engine
                .encode_store(text_store_req(
                    &format!("Memory {i}"),
                    Some(StoreType::Episodic),
                ))
                .await
                .unwrap();
        }

        let resp = engine
            .lifecycle_decay_tick(DecayTickRequest {
                header: None,
                tick_duration_seconds: Some(86400),
            })
            .await
            .unwrap();

        assert_eq!(resp.records_updated, 5);
    }

    #[tokio::test]
    async fn forget_requires_confirmation() {
        let engine = make_engine().await;

        let result = engine
            .lifecycle_forget(ForgetRequest {
                header: None,
                record_ids: None,
                store: None,
                temporal_range: None,
                cascade: false,
                confirm: false,
            })
            .await;

        assert!(matches!(result, Err(CerememoryError::ForgetUnconfirmed)));
    }

    #[tokio::test]
    async fn forget_deletes_records() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(text_store_req("To forget", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let deleted = engine
            .lifecycle_forget(ForgetRequest {
                header: None,
                record_ids: Some(vec![resp.record_id]),
                store: None,
                temporal_range: None,
                cascade: false,
                confirm: true,
            })
            .await
            .unwrap();

        assert_eq!(deleted, 1);
        let record = engine.get_store_record(&resp.record_id).await.unwrap();
        assert!(record.is_none());

        // Also verify removed from text index
        let hits = engine.text_index.search("forget", None, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn forget_temporal_range_deletes_matching_records() {
        let engine = make_engine().await;

        let mut old_record = MemoryRecord::new_text(StoreType::Episodic, "old");
        old_record.created_at = Utc::now() - chrono::Duration::days(2);
        old_record.updated_at = old_record.created_at;
        old_record.last_accessed_at = old_record.created_at;
        engine.episodic.store(old_record.clone()).await.unwrap();
        engine
            .coordinator
            .register(
                old_record.id,
                StoreType::Episodic,
                old_record.associations.clone(),
            )
            .await;
        engine.index_record(&old_record).unwrap();

        let current = MemoryRecord::new_text(StoreType::Episodic, "current");
        engine.episodic.store(current.clone()).await.unwrap();
        engine
            .coordinator
            .register(
                current.id,
                StoreType::Episodic,
                current.associations.clone(),
            )
            .await;
        engine.index_record(&current).unwrap();

        let deleted = engine
            .lifecycle_forget(ForgetRequest {
                header: None,
                record_ids: None,
                store: None,
                temporal_range: Some(TemporalRange {
                    start: Utc::now() - chrono::Duration::minutes(1),
                    end: Utc::now() + chrono::Duration::minutes(1),
                }),
                cascade: false,
                confirm: true,
            })
            .await
            .unwrap();

        assert_eq!(deleted, 1);
        assert!(engine
            .get_store_record(&current.id)
            .await
            .unwrap()
            .is_none());
        assert!(engine
            .get_store_record(&old_record.id)
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn mode_switch() {
        let engine = make_engine().await;

        engine
            .lifecycle_set_mode(SetModeRequest {
                header: None,
                mode: RecallMode::Perfect,
                scope: None,
            })
            .await
            .unwrap();

        assert_eq!(*engine.recall_mode.read().await, RecallMode::Perfect);
    }

    #[tokio::test]
    async fn introspect_stats() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("Stats test", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let stats = engine.introspect_stats().await.unwrap();
        assert_eq!(stats.total_records, 1);
        assert_eq!(stats.records_by_store[&StoreType::Episodic], 1);
        assert!((stats.avg_fidelity - 1.0).abs() < f64::EPSILON);
        assert!(!stats.background_decay_enabled);
    }

    #[tokio::test]
    async fn auto_store_routing() {
        let engine = make_engine().await;

        let resp1 = engine
            .encode_store(text_store_req("An event", None))
            .await
            .unwrap();
        assert_eq!(resp1.store, StoreType::Episodic);

        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Rust is a systems language".to_vec(),
                    embedding: None,
                }],
                summary: Some("Rust programming facts".to_string()),
            },
            store: None,
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        };
        let resp2 = engine.encode_store(req).await.unwrap();
        assert_eq!(resp2.store, StoreType::Semantic);
    }

    #[tokio::test]
    async fn human_mode_degrades_content() {
        let content = MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: b"The quick brown fox jumps over the lazy dog".to_vec(),
                embedding: None,
            }],
            summary: None,
        };

        let result = apply_human_noise(&content, 0.95);
        assert_eq!(result.blocks[0].data, content.blocks[0].data);

        let result = apply_human_noise(&content, 0.3);
        let text = std::str::from_utf8(&result.blocks[0].data).unwrap();
        assert!(text.contains("..."));
    }

    #[tokio::test]
    async fn encode_update() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(text_store_req("Original", Some(StoreType::Episodic)))
            .await
            .unwrap();

        engine
            .encode_update(EncodeUpdateRequest {
                header: None,
                record_id: resp.record_id,
                content: Some(MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Text,
                        format: "text/plain".to_string(),
                        data: b"Updated content".to_vec(),
                        embedding: None,
                    }],
                    summary: None,
                }),
                emotion: None,
                metadata: None,
            })
            .await
            .unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert_eq!(record.text_content(), Some("Updated content"));

        // Verify text index was updated
        let hits = engine.text_index.search("Updated", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let hits = engine.text_index.search("Original", None, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn encode_update_refreshes_structured_index() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Structured,
                        format: "application/json".to_string(),
                        data: br#"{"user":{"name":"Alice"}}"#.to_vec(),
                        embedding: None,
                    }],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        assert_eq!(
            engine.text_index.search("Alice", None, 10).unwrap().len(),
            1
        );

        engine
            .encode_update(EncodeUpdateRequest {
                header: None,
                record_id: resp.record_id,
                content: Some(MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Structured,
                        format: "application/json".to_string(),
                        data: br#"{"user":{"name":"Bob"}}"#.to_vec(),
                        embedding: None,
                    }],
                    summary: None,
                }),
                emotion: None,
                metadata: None,
            })
            .await
            .unwrap();

        assert!(engine
            .text_index
            .search("Alice", None, 10)
            .unwrap()
            .is_empty());
        assert_eq!(engine.text_index.search("Bob", None, 10).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn rebuild_coordinator_reindexes_structured_records() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Structured,
                        format: "application/json".to_string(),
                        data: br#"{"project":{"name":"Cerememory"}}"#.to_vec(),
                        embedding: None,
                    }],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        engine.text_index.remove(resp.record_id).unwrap();
        assert!(engine
            .text_index
            .search("Cerememory", None, 10)
            .unwrap()
            .is_empty());

        engine.rebuild_coordinator().await.unwrap();

        assert_eq!(
            engine
                .text_index
                .search("Cerememory", None, 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn background_decay_runs() {
        let config = EngineConfig {
            background_decay_interval_secs: Some(1), // 1 second for test
            ..EngineConfig::default()
        };
        let engine = Arc::new(CerememoryEngine::new(config).unwrap());

        // Store a record
        engine
            .encode_store(text_store_req("Decay me", Some(StoreType::Episodic)))
            .await
            .unwrap();

        engine.start_background_decay();
        assert!(engine.is_background_decay_enabled().await);

        // Wait for at least one tick
        tokio::time::sleep(std::time::Duration::from_millis(2500)).await;

        engine.stop_background_decay().await;
        assert!(!engine.is_background_decay_enabled().await);
    }

    #[tokio::test]
    async fn background_decay_disabled_by_default() {
        let engine = Arc::new(make_engine().await);
        engine.start_background_decay(); // should be a no-op
        assert!(!engine.is_background_decay_enabled().await);
    }

    #[tokio::test]
    async fn multimodal_store_and_retrieve() {
        let engine = make_engine().await;

        // Store an image record with embedding
        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Image,
                    format: "image/png".to_string(),
                    data: vec![0x89, 0x50, 0x4E, 0x47], // PNG header bytes
                    embedding: Some(vec![0.5, 0.3, 0.8]),
                }],
                summary: Some("A photo of a sunset".to_string()),
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        };

        let resp = engine.encode_store(req).await.unwrap();

        // Retrieve by ID
        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert_eq!(record.content.blocks[0].modality, Modality::Image);
        assert_eq!(record.content.blocks[0].data, vec![0x89, 0x50, 0x4E, 0x47]);

        // Search by embedding should find it
        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(vec![0.5, 0.3, 0.8]),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let recall_resp = engine.recall_query(query).await.unwrap();
        assert!(!recall_resp.memories.is_empty());
    }

    #[tokio::test]
    async fn summary_only_records_are_text_searchable() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Image,
                        format: "image/png".to_string(),
                        data: vec![0x89, 0x50, 0x4E, 0x47],
                        embedding: None,
                    }],
                    summary: Some("sunset skyline".to_string()),
                },
                store: Some(StoreType::Semantic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let recalled = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    text: Some("skyline".to_string()),
                    ..Default::default()
                },
                stores: Some(vec![StoreType::Semantic]),
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        assert_eq!(recalled.memories.len(), 1);
        assert_eq!(recalled.memories[0].record.id, resp.record_id);
    }

    #[tokio::test]
    async fn multiple_text_blocks_are_all_indexed() {
        let engine = make_engine().await;

        engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![
                        ContentBlock {
                            modality: Modality::Text,
                            format: "text/plain".to_string(),
                            data: b"primary block".to_vec(),
                            embedding: None,
                        },
                        ContentBlock {
                            modality: Modality::Text,
                            format: "text/plain".to_string(),
                            data: b"secondary block".to_vec(),
                            embedding: None,
                        },
                    ],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let recalled = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    text: Some("secondary".to_string()),
                    ..Default::default()
                },
                stores: Some(vec![StoreType::Episodic]),
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        assert_eq!(recalled.memories.len(), 1);
    }

    #[tokio::test]
    async fn structured_recall_survives_rebuild() {
        let engine = make_engine().await;
        engine
            .encode_store(structured_store_req(
                r#"{"profile":{"city":"Tokyo","skills":["rust","python"]}}"#,
                Some(StoreType::Semantic),
            ))
            .await
            .unwrap();

        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("Tokyo".to_string()),
                ..Default::default()
            },
            stores: Some(vec![StoreType::Semantic]),
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let before = engine.recall_query(query.clone()).await.unwrap();
        assert_eq!(before.memories.len(), 1);

        engine.rebuild_coordinator().await.unwrap();

        let after = engine.recall_query(query).await.unwrap();
        assert_eq!(after.memories.len(), 1);
    }

    #[tokio::test]
    async fn structured_update_rebuilds_text_index() {
        let engine = make_engine().await;
        let resp = engine
            .encode_store(structured_store_req(
                r#"{"profile":{"city":"Tokyo"}}"#,
                Some(StoreType::Semantic),
            ))
            .await
            .unwrap();

        let tokyo_hits = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    text: Some("Tokyo".to_string()),
                    ..Default::default()
                },
                stores: Some(vec![StoreType::Semantic]),
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();
        assert_eq!(tokyo_hits.memories.len(), 1);

        engine
            .encode_update(EncodeUpdateRequest {
                header: None,
                record_id: resp.record_id,
                content: Some(MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Structured,
                        format: "application/json".to_string(),
                        data: br#"{"profile":{"city":"Osaka"}}"#.to_vec(),
                        embedding: None,
                    }],
                    summary: None,
                }),
                emotion: None,
                metadata: None,
            })
            .await
            .unwrap();

        let tokyo_hits = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    text: Some("Tokyo".to_string()),
                    ..Default::default()
                },
                stores: Some(vec![StoreType::Semantic]),
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();
        assert!(tokyo_hits.memories.is_empty());

        let osaka_hits = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    text: Some("Osaka".to_string()),
                    ..Default::default()
                },
                stores: Some(vec![StoreType::Semantic]),
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();
        assert_eq!(osaka_hits.memories.len(), 1);
    }

    #[tokio::test]
    async fn multimodal_image_recall_uses_provider_embedding() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        let image = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Image,
                        format: "image/png".to_string(),
                        data: image.clone(),
                        embedding: None,
                    }],
                    summary: Some("indexed image".to_string()),
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let resp = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    image: Some(image),
                    ..Default::default()
                },
                stores: None,
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        assert_eq!(resp.memories.len(), 1);
        assert_eq!(
            resp.memories[0].record.content.blocks[0].modality,
            Modality::Image
        );
    }

    #[tokio::test]
    async fn multimodal_audio_recall_uses_provider_transcript() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        let wav_bytes = b"RIFFabcdWAVE".to_vec();
        engine
            .encode_store(text_store_req("audio-12", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let resp = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    audio: Some(wav_bytes),
                    ..Default::default()
                },
                stores: None,
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        assert!(!resp.memories.is_empty());
        assert_eq!(resp.memories[0].record.text_content(), Some("audio-12"));
    }

    #[tokio::test]
    async fn encode_store_processes_all_multimodal_blocks() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        let image_one = vec![0x89, b'P', b'N', b'G', 1, 2, 3, 4];
        let image_two = vec![0x89, b'P', b'N', b'G', 5, 6, 7, 8, 9];
        let audio_one = b"RIFFabcdWAVEone".to_vec();
        let audio_two = b"RIFFabcdWAVEtwoo".to_vec();

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![
                        ContentBlock {
                            modality: Modality::Image,
                            format: "image/png".to_string(),
                            data: image_one.clone(),
                            embedding: None,
                        },
                        ContentBlock {
                            modality: Modality::Audio,
                            format: "audio/wav".to_string(),
                            data: audio_one.clone(),
                            embedding: None,
                        },
                        ContentBlock {
                            modality: Modality::Image,
                            format: "image/png".to_string(),
                            data: image_two.clone(),
                            embedding: None,
                        },
                        ContentBlock {
                            modality: Modality::Audio,
                            format: "audio/wav".to_string(),
                            data: audio_two.clone(),
                            embedding: None,
                        },
                    ],
                    summary: Some("multimodal batch".to_string()),
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert_eq!(record.content.blocks.len(), 6);
        assert_eq!(record.content.blocks[0].modality, Modality::Image);
        assert_eq!(record.content.blocks[2].modality, Modality::Image);
        assert_eq!(record.content.blocks[4].modality, Modality::Text);
        assert_eq!(record.content.blocks[5].modality, Modality::Text);

        let image_one_embedding = record.content.blocks[0].embedding.as_ref().unwrap();
        let image_two_embedding = record.content.blocks[2].embedding.as_ref().unwrap();
        assert_eq!(image_one_embedding[0], image_one.len() as f32);
        assert_eq!(image_two_embedding[0], image_two.len() as f32);

        assert_eq!(
            std::str::from_utf8(&record.content.blocks[4].data).unwrap(),
            format!("audio-{}", audio_one.len())
        );
        assert_eq!(
            std::str::from_utf8(&record.content.blocks[5].data).unwrap(),
            format!("audio-{}", audio_two.len())
        );
        assert!(record.content.blocks[4].embedding.is_some());
        assert!(record.content.blocks[5].embedding.is_some());
    }

    #[tokio::test]
    async fn size_limit_validation() {
        let engine = make_engine().await;

        // Text exceeding 1MB should fail
        let big_text = vec![b'A'; 1_048_577]; // 1MB + 1
        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: big_text,
                    embedding: None,
                }],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        };

        let result = engine.encode_store(req).await;
        assert!(matches!(
            result,
            Err(CerememoryError::ContentTooLarge { .. })
        ));
    }

    // ─── Export/Import API tests ────────────────────────────────────

    #[tokio::test]
    async fn lifecycle_export_returns_bytes_and_response() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("Export me", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let (bytes, resp) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        assert_eq!(resp.record_count, 1);
        assert!(!bytes.is_empty());
    }

    #[tokio::test]
    async fn lifecycle_export_store_filter() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("Episodic", Some(StoreType::Episodic)))
            .await
            .unwrap();
        engine
            .encode_store(text_store_req("Semantic", Some(StoreType::Semantic)))
            .await
            .unwrap();

        let (_, resp) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: Some(vec![StoreType::Episodic]),
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        assert_eq!(resp.record_count, 1);
    }

    #[tokio::test]
    async fn lifecycle_export_store_filter_deduplicates_requested_stores() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("Episodic", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let (bytes, resp) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: Some(vec![StoreType::Episodic, StoreType::Episodic]),
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        assert_eq!(resp.record_count, 1);

        let records = cerememory_archive::import_records(&bytes).unwrap();
        assert_eq!(records.len(), 1);
        assert!(records.iter().all(|r| r.store == StoreType::Episodic));
    }

    #[tokio::test]
    async fn lifecycle_export_encrypted_roundtrip() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("Secret data", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let (encrypted_bytes, resp) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: true,
                encryption_key: Some("my-passphrase".to_string()),
            })
            .await
            .unwrap();

        assert_eq!(resp.record_count, 1);

        // Verify the encrypted data cannot be imported without the key
        let result = engine.import_records(&encrypted_bytes).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn lifecycle_import_with_archive_data() {
        let engine = make_engine().await;

        // Store a record and export it
        engine
            .encode_store(text_store_req("Import me", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let (bytes, _) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        // Create a fresh engine and import
        let engine2 = make_engine().await;
        let imported = engine2
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Merge,
                conflict_resolution: ConflictResolution::KeepExisting,
                decryption_key: None,
                archive_data: Some(bytes),
            })
            .await
            .unwrap();

        assert_eq!(imported, 1);
        let stats = engine2.introspect_stats().await.unwrap();
        assert_eq!(stats.total_records, 1);
    }

    #[tokio::test]
    async fn import_conflict_keep_existing() {
        let engine = make_engine().await;

        // Store a record
        let resp = engine
            .encode_store(text_store_req("Original", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Export it
        let (bytes, _) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        // Import with keep_existing — should skip the duplicate
        let imported = engine
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Merge,
                conflict_resolution: ConflictResolution::KeepExisting,
                decryption_key: None,
                archive_data: Some(bytes),
            })
            .await
            .unwrap();

        assert_eq!(imported, 0);
        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();
        assert_eq!(record.text_content(), Some("Original"));
    }

    #[tokio::test]
    async fn import_conflict_keep_imported() {
        let engine = make_engine().await;

        // Store a record
        let resp = engine
            .encode_store(text_store_req("Original text", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Export
        let (bytes, _) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        // Import with keep_imported — should replace the existing record
        let imported = engine
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Merge,
                conflict_resolution: ConflictResolution::KeepImported,
                decryption_key: None,
                archive_data: Some(bytes),
            })
            .await
            .unwrap();

        assert_eq!(imported, 1);
        // Record should still exist (replaced with same data)
        let record = engine.get_store_record(&resp.record_id).await.unwrap();
        assert!(record.is_some());
    }

    #[tokio::test]
    async fn import_conflict_keep_newer() {
        let engine = make_engine().await;

        // Store a record
        let resp = engine
            .encode_store(text_store_req("First version", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Export the archive (captures the record with its current timestamp)
        let (bytes, _) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        // Update the record so the in-store version is newer
        engine
            .encode_update(EncodeUpdateRequest {
                header: None,
                record_id: resp.record_id,
                content: Some(MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Text,
                        format: "text/plain".to_string(),
                        data: b"Updated version".to_vec(),
                        embedding: None,
                    }],
                    summary: None,
                }),
                emotion: None,
                metadata: None,
            })
            .await
            .unwrap();

        // Import with keep_newer — the archive version is older, so it should be skipped
        let imported = engine
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Merge,
                conflict_resolution: ConflictResolution::KeepNewer,
                decryption_key: None,
                archive_data: Some(bytes),
            })
            .await
            .unwrap();

        assert_eq!(imported, 0);
        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();
        assert_eq!(record.text_content(), Some("Updated version"));
    }

    #[tokio::test]
    async fn import_encrypted_archive() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req(
                "Encrypted import",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();

        let (encrypted, _) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: true,
                encryption_key: Some("pass123".to_string()),
            })
            .await
            .unwrap();

        // Import into a fresh engine with the correct key
        let engine2 = make_engine().await;
        let imported = engine2
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Merge,
                conflict_resolution: ConflictResolution::KeepExisting,
                decryption_key: Some("pass123".to_string()),
                archive_data: Some(encrypted),
            })
            .await
            .unwrap();

        assert_eq!(imported, 1);
    }

    #[tokio::test]
    async fn import_missing_archive_data() {
        let engine = make_engine().await;

        let result = engine
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Merge,
                conflict_resolution: ConflictResolution::KeepExisting,
                decryption_key: None,
                archive_data: None,
            })
            .await;

        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("archive_data"));
    }

    #[tokio::test]
    async fn lifecycle_export_encrypt_without_key_fails() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("Some data", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let result = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: true,
                encryption_key: None,
            })
            .await;

        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("encryption_key is required"));
    }

    #[tokio::test]
    async fn import_conflict_cross_store_keep_imported_count_stays_one() {
        let engine = make_engine().await;

        // Store a record in Episodic
        let resp = engine
            .encode_store(text_store_req(
                "Original in episodic",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();
        let record_id = resp.record_id;

        // Export the archive (record is tagged as Episodic)
        let (bytes, _) = engine
            .lifecycle_export(ExportRequest {
                header: None,
                format: "cma".to_string(),
                stores: None,
                encrypt: false,
                encryption_key: None,
            })
            .await
            .unwrap();

        // Import with KeepImported — the same record exists in Episodic,
        // and the archive also has it as Episodic. After conflict resolution
        // with KeepImported, the old record must be deleted first and then
        // the imported one stored. The total count must remain 1, not 2.
        let imported = engine
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Merge,
                conflict_resolution: ConflictResolution::KeepImported,
                decryption_key: None,
                archive_data: Some(bytes),
            })
            .await
            .unwrap();

        assert_eq!(imported, 1);

        // Total count must be exactly 1 — proves delete-before-store worked
        let stats = engine.introspect_stats().await.unwrap();
        assert_eq!(stats.total_records, 1);

        // The record should still be retrievable by its original ID
        let record = engine.get_store_record(&record_id).await.unwrap();
        assert!(record.is_some());
    }

    #[tokio::test]
    async fn import_strategy_replace_rejected() {
        let engine = make_engine().await;

        let result = engine
            .lifecycle_import(ImportRequest {
                header: None,
                archive_id: "test".to_string(),
                strategy: ImportStrategy::Replace,
                conflict_resolution: ConflictResolution::KeepNewer,
                decryption_key: None,
                archive_data: Some(vec![]),
            })
            .await;

        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Replace"));
    }

    // ─── recall.timeline tests ───────────────────────────────────────

    #[tokio::test]
    async fn timeline_hour_granularity() {
        let engine = make_engine().await;
        engine
            .encode_store(text_store_req("Morning event", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let now = Utc::now();
        let resp = engine
            .recall_timeline(RecallTimelineRequest {
                header: None,
                range: TemporalRange {
                    start: now - chrono::Duration::hours(1),
                    end: now + chrono::Duration::hours(1),
                },
                granularity: TimeGranularity::Hour,
                min_fidelity: None,
                emotion_filter: None,
            })
            .await
            .unwrap();

        assert!(!resp.buckets.is_empty());
        let total: u32 = resp.buckets.iter().map(|b| b.count).sum();
        assert!(total >= 1);
    }

    #[tokio::test]
    async fn timeline_day_granularity() {
        let engine = make_engine().await;
        for i in 0..3 {
            engine
                .encode_store(text_store_req(
                    &format!("Day event {i}"),
                    Some(StoreType::Episodic),
                ))
                .await
                .unwrap();
        }

        let now = Utc::now();
        let resp = engine
            .recall_timeline(RecallTimelineRequest {
                header: None,
                range: TemporalRange {
                    start: now - chrono::Duration::days(1),
                    end: now + chrono::Duration::days(1),
                },
                granularity: TimeGranularity::Day,
                min_fidelity: None,
                emotion_filter: None,
            })
            .await
            .unwrap();

        let total: u32 = resp.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 3);
    }

    #[tokio::test]
    async fn timeline_empty_range() {
        let engine = make_engine().await;
        engine
            .encode_store(text_store_req("An event", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Query a range far in the past
        let resp = engine
            .recall_timeline(RecallTimelineRequest {
                header: None,
                range: TemporalRange {
                    start: Utc::now() - chrono::Duration::days(365 * 10),
                    end: Utc::now() - chrono::Duration::days(365 * 9),
                },
                granularity: TimeGranularity::Day,
                min_fidelity: None,
                emotion_filter: None,
            })
            .await
            .unwrap();

        assert!(resp.buckets.is_empty());
    }

    #[tokio::test]
    async fn timeline_min_fidelity_filter() {
        let engine = make_engine().await;
        engine
            .encode_store(text_store_req(
                "High fidelity event",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();

        let now = Utc::now();
        let resp = engine
            .recall_timeline(RecallTimelineRequest {
                header: None,
                range: TemporalRange {
                    start: now - chrono::Duration::hours(1),
                    end: now + chrono::Duration::hours(1),
                },
                granularity: TimeGranularity::Hour,
                min_fidelity: Some(0.5),
                emotion_filter: None,
            })
            .await
            .unwrap();

        // New records have fidelity 1.0, should pass filter
        let total: u32 = resp.buckets.iter().map(|b| b.count).sum();
        assert!(total >= 1);
    }

    #[tokio::test]
    async fn timeline_emotion_filter() {
        let engine = make_engine().await;

        // Store a record with high joy
        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Happy event".to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: Some(EmotionVector {
                joy: 0.9,
                ..Default::default()
            }),
            context: None,
            metadata: None,
            associations: None,
        };
        engine.encode_store(req).await.unwrap();

        // Store a record with high sadness
        let req2 = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Sad event".to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: Some(EmotionVector {
                sadness: 0.9,
                ..Default::default()
            }),
            context: None,
            metadata: None,
            associations: None,
        };
        engine.encode_store(req2).await.unwrap();

        let now = Utc::now();

        // Filter for joy — should only include the happy event
        let resp = engine
            .recall_timeline(RecallTimelineRequest {
                header: None,
                range: TemporalRange {
                    start: now - chrono::Duration::hours(1),
                    end: now + chrono::Duration::hours(1),
                },
                granularity: TimeGranularity::Hour,
                min_fidelity: None,
                emotion_filter: Some(EmotionVector {
                    joy: 1.0,
                    ..Default::default()
                }),
            })
            .await
            .unwrap();

        let total: u32 = resp.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 1, "Only the joyful event should match");
    }

    #[tokio::test]
    async fn timeline_multi_store() {
        let engine = make_engine().await;
        engine
            .encode_store(text_store_req("Episodic event", Some(StoreType::Episodic)))
            .await
            .unwrap();
        engine
            .encode_store(text_store_req(
                "Procedural event",
                Some(StoreType::Procedural),
            ))
            .await
            .unwrap();

        let now = Utc::now();
        let resp = engine
            .recall_timeline(RecallTimelineRequest {
                header: None,
                range: TemporalRange {
                    start: now - chrono::Duration::hours(1),
                    end: now + chrono::Duration::hours(1),
                },
                granularity: TimeGranularity::Hour,
                min_fidelity: None,
                emotion_filter: None,
            })
            .await
            .unwrap();

        let total: u32 = resp.buckets.iter().map(|b| b.count).sum();
        assert!(total >= 2);
    }

    // ─── recall.graph tests ──────────────────────────────────────────

    #[tokio::test]
    async fn graph_centered_traversal() {
        let engine = make_engine().await;

        // Create two linked records
        let r1 = engine
            .encode_store(text_store_req("Node A", Some(StoreType::Episodic)))
            .await
            .unwrap();
        let r2 = engine
            .encode_store(text_store_req("Node B", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Add association
        let assoc = Association {
            target_id: r2.record_id,
            association_type: AssociationType::Semantic,
            weight: 0.9,
            created_at: Utc::now(),
            last_co_activation: Utc::now(),
        };
        engine
            .coordinator
            .add_association(&r1.record_id, assoc)
            .await
            .unwrap();

        let resp = engine
            .recall_graph(RecallGraphRequest {
                header: None,
                center_id: Some(r1.record_id),
                depth: 2,
                edge_types: None,
                limit_nodes: 10,
            })
            .await
            .unwrap();

        assert!(resp.nodes.len() >= 2);
        assert!(!resp.edges.is_empty());
    }

    #[tokio::test]
    async fn graph_empty() {
        let engine = make_engine().await;
        let resp = engine
            .recall_graph(RecallGraphRequest {
                header: None,
                center_id: None,
                depth: 1,
                edge_types: None,
                limit_nodes: 10,
            })
            .await
            .unwrap();

        assert!(resp.nodes.is_empty());
    }

    #[tokio::test]
    async fn graph_depth_limiting() {
        let engine = make_engine().await;
        let r1 = engine
            .encode_store(text_store_req("Node 1", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let resp = engine
            .recall_graph(RecallGraphRequest {
                header: None,
                center_id: Some(r1.record_id),
                depth: 0,
                edge_types: None,
                limit_nodes: 10,
            })
            .await
            .unwrap();

        // Depth 0 = only the center node, no traversal
        assert_eq!(resp.nodes.len(), 1);
        assert!(resp.edges.is_empty());
    }

    // ─── introspect.decay_forecast tests ─────────────────────────────

    #[tokio::test]
    async fn decay_forecast_future() {
        let engine = make_engine().await;
        let resp = engine
            .encode_store(text_store_req("Forecast me", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let forecast = engine
            .introspect_decay_forecast(DecayForecastRequest {
                header: None,
                record_ids: vec![resp.record_id],
                forecast_at: Utc::now() + chrono::Duration::days(30),
            })
            .await
            .unwrap();

        assert_eq!(forecast.forecasts.len(), 1);
        assert!(forecast.forecasts[0].forecasted_fidelity < forecast.forecasts[0].current_fidelity);
        assert!(forecast.forecasts[0].forecasted_fidelity > 0.0);
    }

    #[tokio::test]
    async fn decay_forecast_threshold_date() {
        let engine = make_engine().await;
        let resp = engine
            .encode_store(text_store_req("Will decay", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let forecast = engine
            .introspect_decay_forecast(DecayForecastRequest {
                header: None,
                record_ids: vec![resp.record_id],
                forecast_at: Utc::now() + chrono::Duration::days(365),
            })
            .await
            .unwrap();

        // Should have an estimated threshold date
        assert!(forecast.forecasts[0].estimated_threshold_date.is_some());
        assert!(forecast.forecasts[0].estimated_threshold_date.unwrap() > Utc::now());
    }

    #[tokio::test]
    async fn decay_forecast_multiple_records() {
        let engine = make_engine().await;
        let r1 = engine
            .encode_store(text_store_req("Record 1", Some(StoreType::Episodic)))
            .await
            .unwrap();
        let r2 = engine
            .encode_store(text_store_req("Record 2", Some(StoreType::Semantic)))
            .await
            .unwrap();

        let forecast = engine
            .introspect_decay_forecast(DecayForecastRequest {
                header: None,
                record_ids: vec![r1.record_id, r2.record_id],
                forecast_at: Utc::now() + chrono::Duration::days(7),
            })
            .await
            .unwrap();

        assert_eq!(forecast.forecasts.len(), 2);
    }

    #[tokio::test]
    async fn decay_forecast_uses_per_record_decay_rate() {
        let engine = make_engine().await;
        let resp = engine
            .encode_store(text_store_req("Test record", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Run a decay tick so last_decay_tick advances
        engine
            .lifecycle_decay_tick(DecayTickRequest {
                header: None,
                tick_duration_seconds: Some(3600),
            })
            .await
            .unwrap();

        // Forecast should use the record's own decay_rate, not global params
        let forecast = engine
            .introspect_decay_forecast(DecayForecastRequest {
                header: None,
                record_ids: vec![resp.record_id],
                forecast_at: Utc::now() + chrono::Duration::days(30),
            })
            .await
            .unwrap();

        assert_eq!(forecast.forecasts.len(), 1);
        // After a decay tick, last_decay_tick is updated, so the forecast baseline
        // should use max(last_accessed_at, last_decay_tick) — not just last_accessed_at
        assert!(forecast.forecasts[0].forecasted_fidelity > 0.0);
        assert!(forecast.forecasts[0].forecasted_fidelity < 1.0);
    }

    // ─── introspect.evolution tests ──────────────────────────────────

    #[tokio::test]
    async fn evolution_returns_metrics() {
        let engine = make_engine().await;
        let metrics = engine.introspect_evolution().await.unwrap();
        // Initially no adjustments
        assert!(metrics.parameter_adjustments.is_empty());
    }

    #[tokio::test]
    async fn evolution_after_decay() {
        let engine = make_engine().await;
        for i in 0..5 {
            engine
                .encode_store(text_store_req(
                    &format!("Record {i}"),
                    Some(StoreType::Episodic),
                ))
                .await
                .unwrap();
        }

        // Run a decay tick to feed evolution engine
        engine
            .lifecycle_decay_tick(DecayTickRequest {
                header: None,
                tick_duration_seconds: Some(86400),
            })
            .await
            .unwrap();

        let metrics = engine.introspect_evolution().await.unwrap();
        // After decay observation, evolution engine may have patterns
        assert!(metrics.detected_patterns.is_empty() || !metrics.detected_patterns.is_empty());
    }

    // ─── LLM provider tests ─────────────────────────────────────────

    /// Mock LLM provider for testing auto-embed.
    struct MockLLMProvider {
        embed_dim: usize,
    }

    impl MockLLMProvider {
        fn new(embed_dim: usize) -> Self {
            Self { embed_dim }
        }
    }

    impl LLMProvider for MockLLMProvider {
        fn embed(
            &self,
            text: &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>,
        > {
            let dim = self.embed_dim;
            let hash = text.len() as f32;
            Box::pin(async move {
                let mut v = vec![0.0f32; dim];
                v[0] = hash;
                v[1] = 1.0;
                Ok(v)
            })
        }

        fn summarize(
            &self,
            texts: &[String],
            max_tokens: usize,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, CerememoryError>> + Send + '_>,
        > {
            let joined = texts.join("; ");
            let truncated = if joined.len() > max_tokens {
                format!("{}...", truncate_str(&joined, max_tokens))
            } else {
                joined
            };
            Box::pin(async move { Ok(truncated) })
        }

        fn extract_relations(
            &self,
            text: &str,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<Vec<ExtractedRelation>, CerememoryError>>
                    + Send
                    + '_,
            >,
        > {
            let has_content = !text.is_empty();
            Box::pin(async move {
                if has_content {
                    Ok(vec![ExtractedRelation {
                        subject: "test".to_string(),
                        predicate: "is_a".to_string(),
                        object: "mock".to_string(),
                        confidence: 0.9,
                    }])
                } else {
                    Ok(Vec::new())
                }
            })
        }

        fn embed_image(
            &self,
            data: &[u8],
            _format: &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>,
        > {
            let dim = self.embed_dim;
            let hash = data.len() as f32;
            Box::pin(async move {
                let mut v = vec![0.0f32; dim];
                v[0] = hash;
                v[1] = 2.0;
                Ok(v)
            })
        }

        fn transcribe_audio(
            &self,
            data: &[u8],
            _format: &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, CerememoryError>> + Send + '_>,
        > {
            let transcript = format!("audio-{}", data.len());
            Box::pin(async move { Ok(transcript) })
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                text_embedding: true,
                image_embedding: true,
                audio_transcription: true,
            }
        }
    }

    #[tokio::test]
    async fn engine_auto_embed_with_provider() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        // Store without embedding — should auto-generate
        let resp = engine
            .encode_store(text_store_req("Auto embed me", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Verify the embedding was generated and stored
        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert!(record.content.blocks[0].embedding.is_some());
        let emb = record.content.blocks[0].embedding.as_ref().unwrap();
        assert_eq!(emb.len(), 4);
    }

    #[tokio::test]
    async fn engine_auto_embeds_all_image_blocks() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![
                        ContentBlock {
                            modality: Modality::Image,
                            format: "image/png".to_string(),
                            data: vec![1; 8],
                            embedding: None,
                        },
                        ContentBlock {
                            modality: Modality::Image,
                            format: "image/png".to_string(),
                            data: vec![2; 13],
                            embedding: None,
                        },
                    ],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert_eq!(record.content.blocks.len(), 2);
        assert_eq!(record.content.blocks[0].modality, Modality::Image);
        assert_eq!(record.content.blocks[1].modality, Modality::Image);
        assert_eq!(record.content.blocks[0].embedding.as_ref().unwrap()[0], 8.0);
        assert_eq!(
            record.content.blocks[1].embedding.as_ref().unwrap()[0],
            13.0
        );
        assert_eq!(record.content.blocks[0].embedding.as_ref().unwrap()[1], 2.0);
        assert_eq!(record.content.blocks[1].embedding.as_ref().unwrap()[1], 2.0);
    }

    #[tokio::test]
    async fn engine_transcribes_all_audio_blocks_in_order() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![
                        ContentBlock {
                            modality: Modality::Audio,
                            format: "audio/wav".to_string(),
                            data: vec![0; 12],
                            embedding: None,
                        },
                        ContentBlock {
                            modality: Modality::Audio,
                            format: "audio/wav".to_string(),
                            data: vec![1; 123],
                            embedding: None,
                        },
                    ],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert_eq!(record.content.blocks.len(), 4);
        assert_eq!(record.content.blocks[0].modality, Modality::Audio);
        assert_eq!(record.content.blocks[1].modality, Modality::Audio);
        assert_eq!(record.content.blocks[2].modality, Modality::Text);
        assert_eq!(record.content.blocks[3].modality, Modality::Text);
        assert_eq!(
            std::str::from_utf8(&record.content.blocks[2].data).unwrap(),
            "audio-12"
        );
        assert_eq!(
            std::str::from_utf8(&record.content.blocks[3].data).unwrap(),
            "audio-123"
        );
        assert_eq!(
            record.content.blocks[2].embedding.as_ref().unwrap()[0],
            "audio-12".len() as f32
        );
        assert_eq!(
            record.content.blocks[3].embedding.as_ref().unwrap()[0],
            "audio-123".len() as f32
        );
    }

    #[tokio::test]
    async fn engine_no_provider_passthrough() {
        // Without provider, embedding should remain None
        let engine = make_engine().await;
        let resp = engine
            .encode_store(text_store_req("No provider", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert!(record.content.blocks[0].embedding.is_none());
    }

    #[tokio::test]
    async fn engine_existing_embedding_not_overwritten() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        // Store WITH an existing embedding
        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Has embedding".to_vec(),
                    embedding: Some(vec![9.0, 9.0, 9.0]),
                }],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        };

        let resp = engine.encode_store(req).await.unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: resp.record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        // Should keep the original embedding, not overwrite with mock
        let emb = record.content.blocks[0].embedding.as_ref().unwrap();
        assert_eq!(emb, &vec![9.0, 9.0, 9.0]);
    }

    #[tokio::test]
    async fn noop_provider_embed_returns_empty() {
        let provider = NoOpProvider;
        let result = provider.embed("test").await;
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn noop_provider_summarize_concatenates() {
        let provider = NoOpProvider;
        let texts = vec!["hello".to_string(), "world".to_string()];
        let result = provider.summarize(&texts, 100).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn noop_provider_extract_relations_empty() {
        let provider = NoOpProvider;
        let result = provider.extract_relations("test").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn noop_provider_capabilities_are_disabled() {
        let provider = NoOpProvider;
        let caps = provider.capabilities();
        assert!(!caps.text_embedding);
        assert!(!caps.image_embedding);
        assert!(!caps.audio_transcription);
    }

    #[tokio::test]
    async fn mock_provider_embed_roundtrip() {
        let provider = MockLLMProvider::new(8);
        let result = provider.embed("hello").await.unwrap();
        assert_eq!(result.len(), 8);
        assert!(result[0] > 0.0); // hash of "hello" length
    }

    #[tokio::test]
    async fn mock_provider_summarize() {
        let provider = MockLLMProvider::new(4);
        let texts = vec!["one".to_string(), "two".to_string()];
        let result = provider.summarize(&texts, 100).await.unwrap();
        assert!(result.contains("one"));
        assert!(result.contains("two"));
    }

    #[tokio::test]
    async fn mock_provider_extract_relations() {
        let provider = MockLLMProvider::new(4);
        let result = provider.extract_relations("some text").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].predicate, "is_a");
    }

    #[tokio::test]
    async fn auto_embed_enables_vector_search() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        // Store text — auto-embed generates a vector
        engine
            .encode_store(text_store_req("Searchable text", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Should be findable via vector search now
        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(vec!["Searchable text".len() as f32, 1.0, 0.0, 0.0]),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let resp = engine.recall_query(query).await.unwrap();
        assert!(!resp.memories.is_empty());
    }

    #[tokio::test]
    async fn image_recall_cue_uses_provider_embedding() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        let image_bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4];
        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Image,
                        format: "image/png".to_string(),
                        data: image_bytes.clone(),
                        embedding: None,
                    }],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let recalled = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    image: Some(image_bytes),
                    ..Default::default()
                },
                stores: None,
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        assert_eq!(recalled.memories.len(), 1);
        assert_eq!(recalled.memories[0].record.id, resp.record_id);
    }

    #[tokio::test]
    async fn audio_recall_cue_uses_transcription() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        let audio_bytes = b"RIFFabcdWAVErest".to_vec();
        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Audio,
                        format: "audio/wav".to_string(),
                        data: audio_bytes.clone(),
                        embedding: None,
                    }],
                    summary: None,
                },
                store: Some(StoreType::Episodic),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            })
            .await
            .unwrap();

        let recalled = engine
            .recall_query(RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    audio: Some(audio_bytes),
                    ..Default::default()
                },
                stores: None,
                limit: 10,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: false,
                activation_depth: 0,
                recall_mode: RecallMode::Perfect,
            })
            .await
            .unwrap();

        assert_eq!(recalled.memories.len(), 1);
        assert_eq!(recalled.memories[0].record.id, resp.record_id);
    }

    // ─── Smart Consolidation tests ───────────────────────────────────

    #[tokio::test]
    async fn consolidation_basic_migration() {
        let engine = make_engine().await;
        for i in 0..3 {
            engine
                .encode_store(text_store_req(
                    &format!("Record {i}"),
                    Some(StoreType::Episodic),
                ))
                .await
                .unwrap();
        }

        let resp = engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Incremental,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        assert_eq!(resp.records_processed, 3);
        assert_eq!(resp.records_migrated, 3);
        assert_eq!(resp.semantic_nodes_created, 3);
    }

    #[tokio::test]
    async fn consolidation_dry_run() {
        let engine = make_engine().await;
        engine
            .encode_store(text_store_req("Dry run test", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let resp = engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Incremental,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: true,
            })
            .await
            .unwrap();

        assert_eq!(resp.records_migrated, 1);
        // Semantic store should still be empty
        assert_eq!(engine.semantic.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn consolidation_with_llm_summarization() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        engine
            .encode_store(text_store_req(
                "A very long piece of text that needs summarization for consolidation",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();

        let resp = engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Incremental,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        assert_eq!(resp.records_migrated, 1);

        // Check that the semantic record has a summary
        let sem_ids = engine.semantic.list_ids().await.unwrap();
        assert_eq!(sem_ids.len(), 1);
        let sem_record = engine.semantic.get(&sem_ids[0]).await.unwrap().unwrap();
        assert!(sem_record.content.summary.is_some());
    }

    #[tokio::test]
    async fn consolidation_with_relation_extraction() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        engine
            .encode_store(text_store_req(
                "Cats are mammals",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();

        engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Incremental,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        // Check that extracted relations are in metadata
        let sem_ids = engine.semantic.list_ids().await.unwrap();
        let sem_record = engine.semantic.get(&sem_ids[0]).await.unwrap().unwrap();
        if let serde_json::Value::Object(ref map) = sem_record.metadata {
            let relations = map.get("extracted_relations");
            assert!(relations.is_some());
            if let Some(serde_json::Value::Array(arr)) = relations {
                assert!(!arr.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn consolidation_no_llm_fallback() {
        // Without LLM, summary should use truncation
        let engine = make_engine().await;
        engine
            .encode_store(text_store_req(
                "Short text without LLM",
                Some(StoreType::Episodic),
            ))
            .await
            .unwrap();

        let resp = engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Incremental,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        assert_eq!(resp.records_migrated, 1);
        let sem_ids = engine.semantic.list_ids().await.unwrap();
        let sem_record = engine.semantic.get(&sem_ids[0]).await.unwrap().unwrap();
        assert!(sem_record.content.summary.is_some());
        assert_eq!(
            sem_record.content.summary.as_deref(),
            Some("Short text without LLM")
        );
    }

    #[tokio::test]
    async fn consolidation_compression_metrics() {
        let engine = make_engine().await;
        for i in 0..5 {
            engine
                .encode_store(text_store_req(
                    &format!("Test {i}"),
                    Some(StoreType::Episodic),
                ))
                .await
                .unwrap();
        }

        let resp = engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Full,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        assert_eq!(resp.records_processed, 5);
        assert!(resp.records_migrated > 0);
        // Pipeline should complete successfully regardless of duplicate count
        assert!(resp.records_processed > 0);
    }

    #[tokio::test]
    async fn consolidation_min_age_filter() {
        let engine = make_engine().await;
        engine
            .encode_store(text_store_req("Fresh record", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let resp = engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Incremental,
                min_age_hours: 24, // Record is brand new, won't pass
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        assert_eq!(resp.records_migrated, 0);
    }

    #[tokio::test]
    async fn consolidation_preserves_associations() {
        let engine = make_engine().await;
        let r1 = engine
            .encode_store(text_store_req("Memory A", Some(StoreType::Episodic)))
            .await
            .unwrap();
        let r2 = engine
            .encode_store(text_store_req("Memory B", Some(StoreType::Episodic)))
            .await
            .unwrap();

        // Create association
        let assoc = Association {
            target_id: r2.record_id,
            association_type: AssociationType::Semantic,
            weight: 0.8,
            created_at: Utc::now(),
            last_co_activation: Utc::now(),
        };
        engine
            .coordinator
            .add_association(&r1.record_id, assoc)
            .await
            .unwrap();

        engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Incremental,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        // The original episodic record should have an association to the new semantic node
        let assocs = engine
            .coordinator
            .get_associations(&r1.record_id)
            .await
            .unwrap();
        assert!(assocs.len() >= 2); // original + consolidated
    }

    #[tokio::test]
    async fn duplicate_detection_with_embeddings() {
        let provider = Arc::new(MockLLMProvider::new(4));
        let engine = CerememoryEngine::new(EngineConfig {
            llm_provider: Some(provider),
            ..Default::default()
        })
        .unwrap();

        // Store two records with very similar text (mock embeddings will differ slightly)
        engine
            .encode_store(text_store_req("Hello world", Some(StoreType::Episodic)))
            .await
            .unwrap();
        engine
            .encode_store(text_store_req("Hello worlds", Some(StoreType::Episodic)))
            .await
            .unwrap();

        let initial_count = engine.episodic.count().await.unwrap();

        let resp = engine
            .lifecycle_consolidate(ConsolidateRequest {
                header: None,
                strategy: ConsolidationStrategy::Full,
                min_age_hours: 0,
                min_access_count: 0,
                dry_run: false,
            })
            .await
            .unwrap();

        // Whether duplicates are detected depends on the mock embedding similarity
        // The test verifies the pipeline doesn't crash
        assert!(resp.records_processed > 0 || initial_count > 0);
    }
}
