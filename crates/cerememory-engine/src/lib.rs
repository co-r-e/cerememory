//! Cerememory Engine — the orchestrator.
//!
//! Assembles all stores, the decay engine, association engine,
//! evolution engine, and the hippocampal coordinator into a
//! unified system that implements the full CMP protocol.

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
    pub working_capacity: usize,
    pub decay_params: DecayParams,
    pub recall_mode: RecallMode,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            episodic_path: None,
            semantic_path: None,
            working_capacity: 7,
            decay_params: DecayParams::default(),
            recall_mode: RecallMode::Human,
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

    // State
    recall_mode: tokio::sync::RwLock<RecallMode>,
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

        let coordinator = Arc::new(HippocampalCoordinator::new());
        let activation = SpreadingActivationEngine::new(Arc::clone(&coordinator));

        Ok(Self {
            episodic,
            semantic,
            procedural: ProceduralStore::new(),
            emotional: EmotionalStore::new(),
            working: WorkingMemoryStore::with_capacity(config.working_capacity),
            decay: PowerLawDecayEngine::new(config.decay_params),
            activation,
            evolution: EvolutionEngine::new(),
            coordinator,
            recall_mode: tokio::sync::RwLock::new(config.recall_mode),
        })
    }

    /// Create an engine with all in-memory stores (for testing).
    pub fn in_memory() -> Result<Self, CerememoryError> {
        Self::new(EngineConfig::default())
    }

    /// Rebuild the hippocampal coordinator from persistent stores.
    /// Must be called after construction when opening existing stores.
    pub async fn rebuild_coordinator(&self) -> Result<(), CerememoryError> {
        let mut entries = Vec::new();

        for store_type in [
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
        ] {
            let ids = dispatch_store!(self, store_type, list_ids())?;
            for id in ids {
                if let Some(record) = dispatch_store!(self, store_type, get(&id))? {
                    entries.push((record.id, store_type, record.associations.clone()));
                }
            }
        }

        self.coordinator.rebuild(entries).await;
        info!(
            records = self.coordinator.total_records().await,
            "Coordinator rebuilt from persistent stores"
        );
        Ok(())
    }

    // ─── Store routing ───────────────────────────────────────────────

    fn route_store(&self, content: &MemoryContent) -> StoreType {
        // Phase 1: simple heuristic — default to Episodic for text content
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

        // Register in coordinator
        self.coordinator
            .register(id, store_type, record.associations.clone())
            .await;

        // Store in the appropriate store, handling Working store LRU eviction
        if store_type == StoreType::Working {
            let (_, evicted) = self.working.store_with_eviction(record).await?;
            if let Some(evicted_id) = evicted {
                self.coordinator.unregister(&evicted_id).await;
            }
        } else {
            dispatch_store!(self, store_type, store(record))?;
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
        let (_, store_type) = self
            .get_store_record(&req.record_id)
            .await?
            .ok_or_else(|| CerememoryError::RecordNotFound(req.record_id.to_string()))?;

        dispatch_store!(self, store_type, update_record(&req.record_id, req.content, req.emotion, req.metadata))
    }

    // ─── CMP Recall Operations ───────────────────────────────────────

    /// recall.query — Retrieve memories (CMP Spec §4.1).
    pub async fn recall_query(&self, req: RecallQueryRequest) -> Result<RecallQueryResponse, CerememoryError> {
        let mode = *self.recall_mode.read().await;
        let recall_mode = req.recall_mode;
        let effective_mode = if mode == RecallMode::Perfect { RecallMode::Perfect } else { recall_mode };

        let stores = req.stores.unwrap_or_else(|| {
            vec![
                StoreType::Episodic,
                StoreType::Semantic,
                StoreType::Procedural,
                StoreType::Working,
            ]
        });

        let mut candidates: Vec<(MemoryRecord, f64)> = Vec::new();
        let mut seen_ids: HashSet<Uuid> = HashSet::new();

        // Text search across requested stores
        if let Some(text) = &req.cue.text {
            for store_type in &stores {
                let results = dispatch_store!(self, *store_type, query_text(text, req.limit as usize * 2))?;
                for record in results {
                    if !seen_ids.insert(record.id) {
                        continue; // Already seen
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
        }

        // Temporal range filter (apply same fidelity/store filters)
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

        // Spreading activation for top candidates
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

        // Build response with reconsolidation
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
                    // Persist fidelity changes (stability boost + reinforcement count)
                    let _ = dispatch_store!(self, store_type, update_fidelity(&record.id, record.fidelity.clone()));
                    // Persist access metadata (access count + last accessed timestamp)
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
                // Filter by association type — check if the edge type matches
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

                // If the record has a summary, keep it; otherwise generate one from content
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

                // Create cross-reference association
                let assoc = Association {
                    target_id: semantic_record.id,
                    association_type: AssociationType::Semantic,
                    weight: 1.0,
                    created_at: Utc::now(),
                    last_co_activation: Utc::now(),
                };
                let _ = self.coordinator.add_association(&id, assoc).await;

                migrated += 1;

                // Prune original if fidelity is very low
                if record.fidelity.score < 0.1 {
                    self.episodic.delete(&id).await?;
                    self.coordinator.unregister(&id).await;
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

        // Collect records from all persistent stores
        for store_type in [
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
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

        // Run decay computation on rayon thread pool
        let decay = self.decay.clone();
        let result = tokio::task::spawn_blocking(move || {
            decay.compute_tick(&all_inputs, tick_secs)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("Decay task failed: {e}")))?;

        // Apply results back to stores
        for output in &result.updates {
            if let Some(&store_type) = record_stores.get(&output.id) {
                if output.should_prune {
                    dispatch_store!(self, store_type, delete(&output.id))?;
                    self.coordinator.unregister(&output.id).await;
                } else {
                    dispatch_store!(self, store_type, update_fidelity(&output.id, output.new_fidelity.clone()))?;
                }
            }
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
                    // Collect cascade targets BEFORE deleting the source
                    let cascade_targets = if req.cascade {
                        self.coordinator.get_associations(&id).await?
                    } else {
                        Vec::new()
                    };

                    if dispatch_store!(self, store_type, delete(&id))? {
                        self.coordinator.unregister(&id).await;
                        deleted += 1;
                    }

                    for assoc in cascade_targets {
                        if let Some(st) = self.coordinator.get_record_store_type(&assoc.target_id).await? {
                            if dispatch_store!(self, st, delete(&assoc.target_id))? {
                                self.coordinator.unregister(&assoc.target_id).await;
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
                    deleted += 1;
                }
            }
        }

        warn!(deleted, "Forget operation completed");
        Ok(deleted)
    }

    /// lifecycle.export — Not implemented in Phase 1.
    pub async fn lifecycle_export(&self, _req: ExportRequest) -> Result<ExportResponse, CerememoryError> {
        Err(CerememoryError::Internal(
            "Export not implemented in Phase 1".to_string(),
        ))
    }

    /// lifecycle.import — Not implemented in Phase 1.
    pub async fn lifecycle_import(&self, _req: ImportRequest) -> Result<(), CerememoryError> {
        Err(CerememoryError::Internal(
            "Import not implemented in Phase 1".to_string(),
        ))
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

            // Compute average fidelity per store
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
/// Low fidelity = more words replaced with "..." or removed.
fn degrade_text(text: &str, fidelity: f64) -> String {
    if fidelity >= 0.9 {
        return text.to_string();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    // Fraction of words to degrade
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
        assert_eq!(resp.associations_inferred, 4); // 2 pairs * 2 directions
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
                tick_duration_seconds: Some(86400), // 24 hours
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

        // Verify it's gone
        let record = engine.get_store_record(&resp.record_id).await.unwrap();
        assert!(record.is_none());
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
    }

    #[tokio::test]
    async fn auto_store_routing() {
        let engine = make_engine().await;

        // Without summary → routes to Episodic
        let resp1 = engine
            .encode_store(text_store_req("An event", None))
            .await
            .unwrap();
        assert_eq!(resp1.store, StoreType::Episodic);

        // With summary → routes to Semantic
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

        // High fidelity → no change
        let result = apply_human_noise(&content, 0.95);
        assert_eq!(result.blocks[0].data, content.blocks[0].data);

        // Low fidelity → degraded
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
    }
}
