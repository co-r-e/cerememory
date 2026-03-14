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
            Some(p) => TextIndex::open(p)?,
            None => TextIndex::open_in_memory()?,
        };

        let vector_index = match &config.vector_index_path {
            Some(p) => VectorIndex::open(p)?,
            None => VectorIndex::open_in_memory()?,
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

        let (tx, mut rx) = tokio::sync::watch::channel(false);
        let engine = Arc::clone(self);

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            interval.tick().await; // skip first immediate tick

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let req = DecayTickRequest {
                            header: None,
                            tick_duration_seconds: Some(interval_secs as u32),
                        };
                        if let Err(e) = engine.lifecycle_decay_tick(req).await {
                            warn!(error = %e, "Background decay tick failed");
                        }
                    }
                    _ = rx.changed() => {
                        if *rx.borrow() {
                            info!("Background decay stopped");
                            return;
                        }
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

                    // Collect text for full-text index rebuild
                    if let Some(text) = record.text_content() {
                        text_records.push((
                            record.id,
                            store_type,
                            text.to_string(),
                            record.content.summary.clone(),
                        ));
                    }

                    // Rebuild vector index from embeddings
                    for block in &record.content.blocks {
                        if let Some(ref emb) = block.embedding {
                            let _ = self.vector_index.upsert(record.id, emb);
                        }
                    }
                }
            }
        }

        self.coordinator.rebuild(entries).await;
        self.text_index.rebuild(&text_records)?;

        info!(
            records = self.coordinator.total_records().await,
            "Coordinator and indexes rebuilt from persistent stores"
        );
        Ok(())
    }

    // ─── Index helpers ─────────────────────────────────────────────

    /// Index a record's text content and embeddings.
    /// Uses only the first embedding found (one vector per record_id).
    fn index_record(&self, record: &MemoryRecord) -> Result<(), CerememoryError> {
        // Text index
        if let Some(text) = record.text_content() {
            self.text_index.add(
                record.id,
                record.store,
                text,
                record.content.summary.as_deref(),
            )?;
        }

        // Vector index — use first embedding found (one vector per record_id)
        if let Some(emb) = record
            .content
            .blocks
            .iter()
            .find_map(|b| b.embedding.as_ref())
        {
            self.vector_index.upsert(record.id, emb)?;
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

    async fn get_store_record(&self, id: &Uuid) -> Result<Option<(MemoryRecord, StoreType)>, CerememoryError> {
        // Check coordinator first for store type hint
        if let Some(st) = self.coordinator.get_record_store_type(id).await? {
            let record = dispatch_store!(self, st, get(id))?;
            return Ok(record.map(|r| (r, st)));
        }

        // Fallback: search all stores
        for st in [StoreType::Working, StoreType::Episodic, StoreType::Semantic, StoreType::Procedural, StoreType::Emotional] {
            if let Some(r) = dispatch_store!(self, st, get(id))? {
                return Ok(Some((r, st)));
            }
        }
        Ok(None)
    }

    // ─── CMP Encode Operations ───────────────────────────────────────

    /// encode.store — Store a new memory record (CMP Spec §3.1).
    pub async fn encode_store(&self, req: EncodeStoreRequest) -> Result<EncodeStoreResponse, CerememoryError> {
        let store_type = req.store.unwrap_or_else(|| self.route_store(&req.content));

        let mut record = MemoryRecord {
            id: Uuid::now_v7(),
            store: store_type,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed_at: Utc::now(),
            access_count: 0,
            content: req.content,
            fidelity: FidelityState::default(),
            emotion: req.emotion.unwrap_or_default(),
            associations: Vec::new(),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            version: 1,
        };

        // Add manual associations
        let mut assoc_count = 0u32;
        if let Some(manual) = req.associations {
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

        record.validate()?;
        let id = record.id;
        let fidelity = record.fidelity.score;

        // 1. Store first (source of truth)
        if store_type == StoreType::Working {
            let (_, evicted) = self.working.store_with_eviction(record.clone()).await?;
            if let Some(evicted_id) = evicted {
                self.coordinator.unregister(&evicted_id).await;
                self.unindex_record(evicted_id);
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

        Ok(EncodeStoreResponse {
            record_id: id,
            store: store_type,
            initial_fidelity: fidelity,
            associations_created: assoc_count,
        })
    }

    /// encode.batch — Store multiple records (CMP Spec §3.2).
    pub async fn encode_batch(&self, req: EncodeBatchRequest) -> Result<EncodeBatchResponse, CerememoryError> {
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
                    let _ = self.coordinator.add_association(&prev, assoc_fwd).await;
                    let _ = self.coordinator.add_association(&resp.record_id, assoc_bwd).await;
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
        record.apply_updates(req.content.clone(), req.emotion.clone(), req.metadata.clone());
        record.validate()?;

        // Persist the update
        dispatch_store!(self, store_type, update_record(&req.record_id, req.content.clone(), req.emotion, req.metadata))?;

        // Update indexes if content changed
        if let Some(ref content) = req.content {
            let text = content
                .blocks
                .iter()
                .find(|b| b.modality == Modality::Text)
                .and_then(|b| std::str::from_utf8(&b.data).ok())
                .unwrap_or("");
            if let Err(e) = self.text_index.update(
                req.record_id,
                store_type,
                text,
                content.summary.as_deref(),
            ) {
                warn!(error = %e, record_id = %req.record_id, "Failed to update text index");
            }

            // Remove old vector, then insert first new embedding
            let _ = self.vector_index.remove(req.record_id);
            if let Some(emb) = content.blocks.iter().find_map(|b| b.embedding.as_ref()) {
                if let Err(e) = self.vector_index.upsert(req.record_id, emb) {
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
    pub async fn recall_query(&self, req: RecallQueryRequest) -> Result<RecallQueryResponse, CerememoryError> {
        let mode = *self.recall_mode.read().await;
        let recall_mode = req.recall_mode;
        let effective_mode = if mode == RecallMode::Perfect { RecallMode::Perfect } else { recall_mode };

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
        if let Some(ref text) = req.cue.text {
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
                        let results = dispatch_store!(self, *store_type, query_text(text, req.limit as usize * 2))?;
                        for record in results {
                            text_scores.insert(record.id, record.fidelity.score);
                        }
                    }
                }
            }
        }

        // 2. Vector similarity search
        if let Some(ref embedding) = req.cue.embedding {
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
        if req.cue.text.is_none() && req.cue.embedding.is_none() {
            // No search cue — candidates will come from temporal or activation only
        }

        // 4. Temporal range filter
        if let Some(ref temporal) = req.cue.temporal {
            let results = self
                .episodic
                .query_temporal_range(temporal.start, temporal.end)
                .await?;
            for record in results {
                if !seen_ids.insert(record.id) {
                    continue;
                }
                if let Some(min_f) = req.min_fidelity {
                    if record.fidelity.score < min_f && !req.include_decayed {
                        continue;
                    }
                }
                let score = record.fidelity.score;
                candidates.push((record, score));
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
                            if let Some((record, _store)) = self.get_store_record(&act.record_id).await? {
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

                if let Some(store_type) = self.coordinator.get_record_store_type(&record.id).await? {
                    let _ = dispatch_store!(self, store_type, update_fidelity(&record.id, record.fidelity.clone()));
                    let _ = dispatch_store!(self, store_type, update_access(&record.id, record.access_count, record.last_accessed_at));
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

        Ok(RecallQueryResponse {
            memories,
            activation_trace: None,
            total_candidates,
        })
    }

    /// recall.associate — Get associated memories (CMP Spec §4.2).
    pub async fn recall_associate(&self, req: RecallAssociateRequest) -> Result<RecallAssociateResponse, CerememoryError> {
        // Verify record exists
        self.get_store_record(&req.record_id)
            .await?
            .ok_or_else(|| CerememoryError::RecordNotFound(req.record_id.to_string()))?;

        let activated = self
            .activation
            .activate(&req.record_id, req.depth, req.min_weight)
            .await?;

        let mut memories = Vec::new();
        for act in activated.iter().take(req.limit as usize) {
            if let Some(types) = &req.association_types {
                let assocs = self.coordinator.get_associations(&req.record_id).await?;
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

    // ─── CMP Lifecycle Operations ────────────────────────────────────

    /// lifecycle.consolidate — Migrate episodic → semantic (CMP Spec §5.1).
    pub async fn lifecycle_consolidate(&self, req: ConsolidateRequest) -> Result<ConsolidateResponse, CerememoryError> {
        let ids = self.episodic.list_ids().await?;
        let mut processed = 0u32;
        let mut migrated = 0u32;
        let mut pruned = 0u32;

        for id in ids {
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

                // Create semantic node from episodic record
                let mut semantic_record = record.clone();
                semantic_record.store = StoreType::Semantic;
                semantic_record.id = Uuid::now_v7();

                if semantic_record.content.summary.is_none() {
                    semantic_record.content.summary = semantic_record.text_content().map(|t| {
                        if t.len() > 100 {
                            format!("{}...", &t[..100])
                        } else {
                            t.to_string()
                        }
                    });
                }

                dispatch_store!(self, StoreType::Semantic, store(semantic_record.clone()))?;
                self.coordinator
                    .register(
                        semantic_record.id,
                        StoreType::Semantic,
                        semantic_record.associations.clone(),
                    )
                    .await;

                // Index the new semantic record
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
                let _ = self.coordinator.add_association(&id, assoc).await;

                migrated += 1;

                if record.fidelity.score < 0.1 {
                    self.episodic.delete(&id).await?;
                    self.coordinator.unregister(&id).await;
                    self.unindex_record(id);
                    pruned += 1;
                }
            }
        }

        Ok(ConsolidateResponse {
            records_processed: processed,
            records_migrated: migrated,
            records_compressed: 0,
            records_pruned: pruned,
            semantic_nodes_created: migrated,
        })
    }

    /// lifecycle.decay_tick — Advance decay (CMP Spec §5.2).
    pub async fn lifecycle_decay_tick(&self, req: DecayTickRequest) -> Result<DecayTickResponse, CerememoryError> {
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
        let result = tokio::task::spawn_blocking(move || {
            decay.compute_tick(&all_inputs, tick_secs)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("Decay task failed: {e}")))?;

        for output in &result.updates {
            if let Some(&store_type) = record_stores.get(&output.id) {
                if output.should_prune {
                    dispatch_store!(self, store_type, delete(&output.id))?;
                    self.coordinator.unregister(&output.id).await;
                    self.unindex_record(output.id);
                } else {
                    dispatch_store!(self, store_type, update_fidelity(&output.id, output.new_fidelity.clone()))?;
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

        let mut deleted = 0u32;

        if let Some(ids) = req.record_ids {
            for id in ids {
                if let Some(store_type) = self.coordinator.get_record_store_type(&id).await? {
                    let cascade_targets = if req.cascade {
                        self.coordinator.get_associations(&id).await?
                    } else {
                        Vec::new()
                    };

                    if dispatch_store!(self, store_type, delete(&id))? {
                        self.coordinator.unregister(&id).await;
                        self.unindex_record(id);
                        deleted += 1;
                    }

                    for assoc in cascade_targets {
                        if let Some(st) = self.coordinator.get_record_store_type(&assoc.target_id).await? {
                            if dispatch_store!(self, st, delete(&assoc.target_id))? {
                                self.coordinator.unregister(&assoc.target_id).await;
                                self.unindex_record(assoc.target_id);
                                deleted += 1;
                            }
                        }
                    }
                }
            }
        }

        if let Some(store_type) = req.store {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                if dispatch_store!(self, store_type, delete(&id))? {
                    self.coordinator.unregister(&id).await;
                    self.unindex_record(id);
                    deleted += 1;
                }
            }
        }

        warn!(deleted, "Forget operation completed");
        Ok(deleted)
    }

    /// lifecycle.export — Export records to a CMA archive with optional store
    /// filtering and encryption.
    pub async fn lifecycle_export(
        &self,
        req: ExportRequest,
    ) -> Result<(Vec<u8>, ExportResponse), CerememoryError> {
        let records = self.collect_all_records().await?;

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
        let records =
            cerememory_archive::import_records_with_key(&data, decryption_key.as_ref())?;

        let conflict_resolution = req.conflict_resolution;
        let mut imported = 0u32;

        for record in records {
            let store_type = record.store;

            // Check for ID conflicts
            if let Some((existing, existing_store)) = self.get_store_record(&record.id).await? {
                match conflict_resolution {
                    ConflictResolution::KeepExisting => continue,
                    ConflictResolution::KeepImported => {
                        // Delete existing, then store new
                        dispatch_store!(self, existing_store, delete(&record.id))?;
                        self.coordinator.unregister(&record.id).await;
                        self.unindex_record(record.id);
                    }
                    ConflictResolution::KeepNewer => {
                        if record.updated_at <= existing.updated_at {
                            continue;
                        }
                        dispatch_store!(self, existing_store, delete(&record.id))?;
                        self.coordinator.unregister(&record.id).await;
                        self.unindex_record(record.id);
                    }
                }
            }

            dispatch_store!(self, store_type, store(record.clone()))?;
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
        let mut records = Vec::new();
        for store_type in [
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
            StoreType::Emotional,
            StoreType::Working,
        ] {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                if let Some((record, _)) = self.get_store_record(&id).await? {
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

        for store_type in [
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
            StoreType::Emotional,
            StoreType::Working,
        ] {
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
    pub async fn introspect_record(&self, req: RecordIntrospectRequest) -> Result<MemoryRecord, CerememoryError> {
        let (record, _) = self
            .get_store_record(&req.record_id)
            .await?
            .ok_or_else(|| CerememoryError::RecordNotFound(req.record_id.to_string()))?;

        Ok(record)
    }
}

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
            associations: None,
        }
    }

    #[tokio::test]
    async fn encode_recall_roundtrip() {
        let engine = make_engine().await;

        let resp = engine
            .encode_store(text_store_req("The quick brown fox", Some(StoreType::Episodic)))
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
    async fn tantivy_tokenized_search() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("The quick brown fox jumps over the lazy dog", Some(StoreType::Episodic)))
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

        let record = engine.introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        }).await.unwrap();

        assert_eq!(record.text_content(), Some("Updated content"));

        // Verify text index was updated
        let hits = engine.text_index.search("Updated", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let hits = engine.text_index.search("Original", None, 10).unwrap();
        assert!(hits.is_empty());
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
            associations: None,
        };

        let resp = engine.encode_store(req).await.unwrap();

        // Retrieve by ID
        let record = engine.introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        }).await.unwrap();

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
            associations: None,
        };

        let result = engine.encode_store(req).await;
        assert!(matches!(result, Err(CerememoryError::ContentTooLarge { .. })));
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
        let record = engine.introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        }).await.unwrap();
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
        let record = engine.introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        }).await.unwrap();
        assert_eq!(record.text_content(), Some("Updated version"));
    }

    #[tokio::test]
    async fn import_encrypted_archive() {
        let engine = make_engine().await;

        engine
            .encode_store(text_store_req("Encrypted import", Some(StoreType::Episodic)))
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
            .encode_store(text_store_req("Original in episodic", Some(StoreType::Episodic)))
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
}
