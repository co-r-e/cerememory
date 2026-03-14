//! Cerememory semantic store — redb-based graph store.
//!
//! Stores semantic memory nodes and typed, weighted edges in a persistent
//! graph structure backed by [redb](https://docs.rs/redb). All redb I/O
//! is sync; we wrap every call in [`tokio::task::spawn_blocking`] so the
//! public API is fully async.
//!
//! ## Tables
//!
//! | Table | Key | Value |
//! |---|---|---|
//! | `semantic_nodes` | UUID bytes (16) | MessagePack `MemoryRecord` |
//! | `semantic_edges` | `source ++ target ++ type_byte` (33) | MessagePack `EdgeEntry` |
//! | `semantic_reverse_edges` | `target ++ source ++ type_byte` (33) | `()` (empty) |
//! | `semantic_concepts` | concept `&str` | MessagePack `Vec<Uuid>` |

use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cerememory_core::{
    Association, AssociationType, CerememoryError, EmotionVector, FidelityState, MemoryContent,
    MemoryRecord, Store,
};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

const NODES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("semantic_nodes");
const EDGES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("semantic_edges");
const REVERSE_EDGES: TableDefinition<&[u8], &[u8]> =
    TableDefinition::new("semantic_reverse_edges");
const CONCEPTS: TableDefinition<&str, &[u8]> = TableDefinition::new("semantic_concepts");

// ---------------------------------------------------------------------------
// Edge key helpers
// ---------------------------------------------------------------------------

/// Composite key for the edge tables: `source(16) ++ target(16) ++ type(1)` = 33 bytes.
fn edge_key(source: &Uuid, target: &Uuid, assoc_type: AssociationType) -> [u8; 33] {
    let mut key = [0u8; 33];
    key[..16].copy_from_slice(source.as_bytes());
    key[16..32].copy_from_slice(target.as_bytes());
    key[32] = association_type_to_byte(assoc_type);
    key
}

/// Reverse key: `target(16) ++ source(16) ++ type(1)`.
fn reverse_edge_key(source: &Uuid, target: &Uuid, assoc_type: AssociationType) -> [u8; 33] {
    let mut key = [0u8; 33];
    key[..16].copy_from_slice(target.as_bytes());
    key[16..32].copy_from_slice(source.as_bytes());
    key[32] = association_type_to_byte(assoc_type);
    key
}

fn association_type_to_byte(t: AssociationType) -> u8 {
    match t {
        AssociationType::Temporal => 0,
        AssociationType::Spatial => 1,
        AssociationType::Semantic => 2,
        AssociationType::Emotional => 3,
        AssociationType::Causal => 4,
        AssociationType::Sequential => 5,
        AssociationType::CrossModal => 6,
        AssociationType::UserDefined => 7,
    }
}

fn byte_to_association_type(b: u8) -> AssociationType {
    match b {
        0 => AssociationType::Temporal,
        1 => AssociationType::Spatial,
        2 => AssociationType::Semantic,
        3 => AssociationType::Emotional,
        4 => AssociationType::Causal,
        5 => AssociationType::Sequential,
        6 => AssociationType::CrossModal,
        _ => AssociationType::UserDefined,
    }
}

// ---------------------------------------------------------------------------
// EdgeEntry (stored value for semantic_edges)
// ---------------------------------------------------------------------------

/// Persisted edge payload (separate from the composite key which already
/// encodes source, target, and type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeEntry {
    pub target_id: Uuid,
    pub association_type: AssociationType,
    pub weight: f64,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// SemanticStore
// ---------------------------------------------------------------------------

/// A graph-oriented persistent store for semantic memories.
///
/// Nodes are full [`MemoryRecord`]s; edges are typed, weighted links with
/// both forward and reverse indexes for efficient bidirectional traversal.
#[derive(Clone)]
pub struct SemanticStore {
    db: Arc<Database>,
}

impl SemanticStore {
    /// Open an in-memory semantic store (for testing).
    pub fn open_in_memory() -> Result<Self, CerememoryError> {
        let tmp = tempfile::NamedTempFile::new()
            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
        Self::open(tmp.path())
    }

    /// Open (or create) a semantic store at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CerememoryError> {
        let db = Database::create(path).map_err(|e| CerememoryError::Storage(e.to_string()))?;

        // Ensure all tables exist.
        let txn = db
            .begin_write()
            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
        txn.open_table(NODES)
            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
        txn.open_table(EDGES)
            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
        txn.open_table(REVERSE_EDGES)
            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
        txn.open_table(CONCEPTS)
            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
        txn.commit()
            .map_err(|e| CerememoryError::Storage(e.to_string()))?;

        Ok(Self {
            db: Arc::new(db),
        })
    }

    // -----------------------------------------------------------------------
    // Graph-specific public API
    // -----------------------------------------------------------------------

    /// Add a weighted, typed edge between two nodes.
    ///
    /// Both forward (`source → target`) and reverse (`target → source`)
    /// indexes are written atomically.
    pub async fn add_edge(
        &self,
        source: Uuid,
        target: Uuid,
        assoc_type: AssociationType,
        weight: f64,
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_write()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            {
                // Verify both nodes exist
                let nodes = txn
                    .open_table(NODES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                if nodes.get(source.as_bytes().as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?.is_none() {
                    return Err(CerememoryError::RecordNotFound(source.to_string()));
                }
                if nodes.get(target.as_bytes().as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?.is_none() {
                    return Err(CerememoryError::RecordNotFound(target.to_string()));
                }
                drop(nodes);

                let fwd_key = edge_key(&source, &target, assoc_type);
                let rev_key = reverse_edge_key(&source, &target, assoc_type);
                let entry = EdgeEntry {
                    target_id: target,
                    association_type: assoc_type,
                    weight,
                    created_at: Utc::now(),
                };
                let entry_bytes = rmp_serde::to_vec(&entry)
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;

                let mut edges = txn
                    .open_table(EDGES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                edges
                    .insert(fwd_key.as_slice(), entry_bytes.as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;

                let mut rev = txn
                    .open_table(REVERSE_EDGES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                rev.insert(rev_key.as_slice(), &[] as &[u8])
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            }
            txn.commit()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    /// Remove an edge (both directions) between two nodes.
    pub async fn remove_edge(
        &self,
        source: Uuid,
        target: Uuid,
        assoc_type: AssociationType,
    ) -> Result<bool, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_write()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let removed = {
                let fwd_key = edge_key(&source, &target, assoc_type);
                let rev_key = reverse_edge_key(&source, &target, assoc_type);

                let mut edges = txn
                    .open_table(EDGES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let existed = edges
                    .remove(fwd_key.as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?
                    .is_some();

                let mut rev = txn
                    .open_table(REVERSE_EDGES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                rev.remove(rev_key.as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;

                existed
            };
            txn.commit()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(removed)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    /// Get all outgoing neighbors (forward edges) of a node.
    pub async fn get_neighbors(
        &self,
        id: Uuid,
    ) -> Result<Vec<Association>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_read()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let edges = txn
                .open_table(EDGES)
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;

            let prefix = id.as_bytes().to_vec();
            let mut result = Vec::new();

            for entry in edges.iter().map_err(|e| CerememoryError::Storage(e.to_string()))? {
                let (k, v) = entry.map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let key_bytes = k.value();
                if key_bytes.len() == 33 && key_bytes.starts_with(&prefix) {
                    let edge: EdgeEntry = rmp_serde::from_slice(v.value())
                        .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                    let source_target_type_byte = key_bytes[32];
                    let target_uuid =
                        Uuid::from_bytes(key_bytes[16..32].try_into().expect("16 bytes"));
                    result.push(Association {
                        target_id: target_uuid,
                        association_type: byte_to_association_type(source_target_type_byte),
                        weight: edge.weight,
                        created_at: edge.created_at,
                        last_co_activation: edge.created_at,
                    });
                }
            }
            Ok(result)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    /// Get all nodes that have edges *pointing to* the given node (reverse lookup).
    pub async fn get_reverse_neighbors(
        &self,
        id: Uuid,
    ) -> Result<Vec<(Uuid, AssociationType)>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_read()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let rev = txn
                .open_table(REVERSE_EDGES)
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;

            let prefix = id.as_bytes().to_vec();
            let mut result = Vec::new();

            for entry in rev.iter().map_err(|e| CerememoryError::Storage(e.to_string()))? {
                let (k, _v) = entry.map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let key_bytes = k.value();
                if key_bytes.len() == 33 && key_bytes.starts_with(&prefix) {
                    let source_uuid =
                        Uuid::from_bytes(key_bytes[16..32].try_into().expect("16 bytes"));
                    let assoc_type = byte_to_association_type(key_bytes[32]);
                    result.push((source_uuid, assoc_type));
                }
            }
            Ok(result)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    /// BFS traversal from `center_id` up to `depth` hops, returning the full
    /// subgraph of reachable records (including the center).
    pub async fn get_subgraph(
        &self,
        center_id: Uuid,
        depth: u32,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue: VecDeque<(Uuid, u32)> = VecDeque::new();
        queue.push_back((center_id, 0));
        visited.insert(center_id);

        let mut result_ids = Vec::new();

        while let Some((current, current_depth)) = queue.pop_front() {
            result_ids.push(current);
            if current_depth >= depth {
                continue;
            }
            let neighbors = self.get_neighbors(current).await?;
            for assoc in &neighbors {
                if visited.insert(assoc.target_id) {
                    queue.push_back((assoc.target_id, current_depth + 1));
                }
            }
        }

        // Fetch all records.
        let mut records = Vec::new();
        for id in result_ids {
            if let Some(rec) = self.get(&id).await? {
                records.push(rec);
            }
        }
        Ok(records)
    }

    // -----------------------------------------------------------------------
    // Concept index helpers
    // -----------------------------------------------------------------------

    /// Index a record under a concept string.
    fn index_concepts_for_record(
        concepts_table: &mut redb::Table<&str, &[u8]>,
        record: &MemoryRecord,
    ) -> Result<(), CerememoryError> {
        if let Some(ref summary) = record.content.summary {
            let words: Vec<&str> = summary.split_whitespace().collect();
            for word in words {
                let concept = word.to_lowercase();
                let mut ids: Vec<Uuid> = match concepts_table.get(concept.as_str()) {
                    Ok(Some(val)) => rmp_serde::from_slice(val.value())
                        .map_err(|e| CerememoryError::Serialization(e.to_string()))?,
                    Ok(None) => Vec::new(),
                    Err(e) => return Err(CerememoryError::Storage(e.to_string())),
                };
                if !ids.contains(&record.id) {
                    ids.push(record.id);
                    let encoded = rmp_serde::to_vec(&ids)
                        .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                    concepts_table
                        .insert(concept.as_str(), encoded.as_slice())
                        .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    /// Remove a record's ID from all concept entries (best-effort).
    fn deindex_concepts_for_record(
        concepts_table: &mut redb::Table<&str, &[u8]>,
        record: &MemoryRecord,
    ) -> Result<(), CerememoryError> {
        if let Some(ref summary) = record.content.summary {
            let words: Vec<&str> = summary.split_whitespace().collect();
            for word in words {
                let concept = word.to_lowercase();
                // Read into owned data, then drop the guard before mutating.
                let existing: Option<Vec<Uuid>> = {
                    match concepts_table.get(concept.as_str()) {
                        Ok(Some(val)) => {
                            let ids: Vec<Uuid> = rmp_serde::from_slice(val.value())
                                .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                            Some(ids)
                        }
                        Ok(None) => None,
                        Err(e) => return Err(CerememoryError::Storage(e.to_string())),
                    }
                };
                if let Some(mut ids) = existing {
                    ids.retain(|uid| uid != &record.id);
                    if ids.is_empty() {
                        concepts_table
                            .remove(concept.as_str())
                            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                    } else {
                        let encoded = rmp_serde::to_vec(&ids)
                            .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                        concepts_table
                            .insert(concept.as_str(), encoded.as_slice())
                            .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Look up record IDs indexed under a concept string.
    pub async fn get_concept_ids(&self, concept: &str) -> Result<Vec<Uuid>, CerememoryError> {
        let db = self.db.clone();
        let concept = concept.to_lowercase();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_read()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let concepts = txn
                .open_table(CONCEPTS)
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            match concepts.get(concept.as_str()) {
                Ok(Some(val)) => {
                    let ids: Vec<Uuid> = rmp_serde::from_slice(val.value())
                        .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                    Ok(ids)
                }
                Ok(None) => Ok(Vec::new()),
                Err(e) => Err(CerememoryError::Storage(e.to_string())),
            }
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }
}

// ---------------------------------------------------------------------------
// Store trait implementation
// ---------------------------------------------------------------------------

impl Store for SemanticStore {
    async fn store(&self, record: MemoryRecord) -> Result<Uuid, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let id = record.id;
            let bytes = rmp_serde::to_vec(&record)
                .map_err(|e| CerememoryError::Serialization(e.to_string()))?;

            let txn = db
                .begin_write()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            {
                let mut nodes = txn
                    .open_table(NODES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                nodes
                    .insert(id.as_bytes().as_slice(), bytes.as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;

                let mut concepts = txn
                    .open_table(CONCEPTS)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                SemanticStore::index_concepts_for_record(&mut concepts, &record)?;
            }
            txn.commit()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(id)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn get(&self, id: &Uuid) -> Result<Option<MemoryRecord>, CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_read()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let nodes = txn
                .open_table(NODES)
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            match nodes.get(id.as_bytes().as_slice()) {
                Ok(Some(val)) => {
                    let rec: MemoryRecord = rmp_serde::from_slice(val.value())
                        .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                    Ok(Some(rec))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(CerememoryError::Storage(e.to_string())),
            }
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn delete(&self, id: &Uuid) -> Result<bool, CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_write()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let existed = {
                let mut nodes = txn
                    .open_table(NODES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;

                // First read the record so we can de-index concepts.
                let record: Option<MemoryRecord> = match nodes.get(id.as_bytes().as_slice()) {
                    Ok(Some(val)) => {
                        let rec: MemoryRecord = rmp_serde::from_slice(val.value())
                            .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                        Some(rec)
                    }
                    _ => None,
                };

                let removed = nodes
                    .remove(id.as_bytes().as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?
                    .is_some();

                if let Some(ref rec) = record {
                    let mut concepts = txn
                        .open_table(CONCEPTS)
                        .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                    SemanticStore::deindex_concepts_for_record(&mut concepts, rec)?;
                }

                // Clean up edges: forward (source=id) and reverse (target=id)
                {
                    let mut edges = txn
                        .open_table(EDGES)
                        .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                    let mut rev_edges = txn
                        .open_table(REVERSE_EDGES)
                        .map_err(|e| CerememoryError::Storage(e.to_string()))?;

                    // Collect forward edge keys where source=id
                    let prefix = id.as_bytes();
                    let fwd_keys: Vec<Vec<u8>> = edges
                        .range(prefix.as_slice()..)
                        .map_err(|e| CerememoryError::Storage(e.to_string()))?
                        .take_while(|r| {
                            r.as_ref()
                                .map(|(k, _)| k.value().starts_with(prefix))
                                .unwrap_or(false)
                        })
                        .filter_map(|r| r.ok().map(|(k, _)| k.value().to_vec()))
                        .collect();

                    for key in &fwd_keys {
                        let _ = edges.remove(key.as_slice());
                        // Build corresponding reverse key
                        if key.len() == 33 {
                            let mut rev = [0u8; 33];
                            rev[..16].copy_from_slice(&key[16..32]);
                            rev[16..32].copy_from_slice(&key[..16]);
                            rev[32] = key[32];
                            let _ = rev_edges.remove(rev.as_slice());
                        }
                    }

                    // Collect reverse edge keys where target=id
                    let rev_keys: Vec<Vec<u8>> = rev_edges
                        .range(prefix.as_slice()..)
                        .map_err(|e| CerememoryError::Storage(e.to_string()))?
                        .take_while(|r| {
                            r.as_ref()
                                .map(|(k, _)| k.value().starts_with(prefix))
                                .unwrap_or(false)
                        })
                        .filter_map(|r| r.ok().map(|(k, _)| k.value().to_vec()))
                        .collect();

                    for key in &rev_keys {
                        let _ = rev_edges.remove(key.as_slice());
                        if key.len() == 33 {
                            let mut fwd = [0u8; 33];
                            fwd[..16].copy_from_slice(&key[16..32]);
                            fwd[16..32].copy_from_slice(&key[..16]);
                            fwd[32] = key[32];
                            let _ = edges.remove(fwd.as_slice());
                        }
                    }
                }

                removed
            };
            txn.commit()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(existed)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn update_fidelity(
        &self,
        id: &Uuid,
        fidelity: FidelityState,
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_write()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            {
                let mut nodes = txn
                    .open_table(NODES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let val = nodes
                    .get(id.as_bytes().as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut rec: MemoryRecord = rmp_serde::from_slice(val.value())
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                drop(val);

                rec.fidelity = fidelity;
                rec.updated_at = Utc::now();

                let bytes = rmp_serde::to_vec(&rec)
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                nodes
                    .insert(id.as_bytes().as_slice(), bytes.as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            }
            txn.commit()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn query_text(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let query = query.to_lowercase();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_read()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let nodes = txn
                .open_table(NODES)
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;

            let mut results = Vec::new();
            for entry in nodes.iter().map_err(|e| CerememoryError::Storage(e.to_string()))? {
                let (_k, v) = entry.map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let rec: MemoryRecord = rmp_serde::from_slice(v.value())
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;

                let matches = rec
                    .text_content()
                    .map(|t| t.to_lowercase().contains(&query))
                    .unwrap_or(false)
                    || rec
                        .content
                        .summary
                        .as_deref()
                        .map(|s| s.to_lowercase().contains(&query))
                        .unwrap_or(false);

                if matches {
                    results.push(rec);
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            Ok(results)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn list_ids(&self) -> Result<Vec<Uuid>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_read()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let nodes = txn
                .open_table(NODES)
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;

            let mut ids = Vec::new();
            for entry in nodes.iter().map_err(|e| CerememoryError::Storage(e.to_string()))? {
                let (k, _v) = entry.map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let bytes: [u8; 16] = k
                    .value()
                    .try_into()
                    .map_err(|_| CerememoryError::Internal("Invalid UUID key length".to_string()))?;
                ids.push(Uuid::from_bytes(bytes));
            }
            Ok(ids)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn count(&self) -> Result<usize, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_read()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let nodes = txn
                .open_table(NODES)
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            let len = nodes
                .len()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(len as usize)
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn update_record(
        &self,
        id: &Uuid,
        content: Option<MemoryContent>,
        emotion: Option<EmotionVector>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_write()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            {
                let mut nodes = txn
                    .open_table(NODES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let val = nodes
                    .get(id.as_bytes().as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut rec: MemoryRecord = rmp_serde::from_slice(val.value())
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                drop(val);

                // De-index old concepts before content change.
                let old_summary = rec.content.summary.clone();

                if let Some(c) = content {
                    rec.content = c;
                }
                if let Some(e) = emotion {
                    rec.emotion = e;
                }
                if let Some(m) = metadata {
                    rec.metadata = m;
                }
                rec.updated_at = Utc::now();
                rec.version += 1;

                // Re-index concepts if summary changed.
                let new_summary = rec.content.summary.clone();
                if old_summary != new_summary {
                    let mut concepts = txn
                        .open_table(CONCEPTS)
                        .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                    // Build a temporary record with old summary for de-indexing.
                    if old_summary.is_some() {
                        let mut old_rec = rec.clone();
                        old_rec.content.summary = old_summary;
                        SemanticStore::deindex_concepts_for_record(&mut concepts, &old_rec)?;
                    }
                    SemanticStore::index_concepts_for_record(&mut concepts, &rec)?;
                }

                let bytes = rmp_serde::to_vec(&rec)
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                nodes
                    .insert(id.as_bytes().as_slice(), bytes.as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            }
            txn.commit()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }

    async fn update_access(
        &self,
        id: &Uuid,
        access_count: u32,
        last_accessed_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db
                .begin_write()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            {
                let mut nodes = txn
                    .open_table(NODES)
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
                let val = nodes
                    .get(id.as_bytes().as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut rec: MemoryRecord = rmp_serde::from_slice(val.value())
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                drop(val);

                rec.access_count = access_count;
                rec.last_accessed_at = last_accessed_at;
                rec.updated_at = Utc::now();

                let bytes = rmp_serde::to_vec(&rec)
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                nodes
                    .insert(id.as_bytes().as_slice(), bytes.as_slice())
                    .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            }
            txn.commit()
                .map_err(|e| CerememoryError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(e.to_string()))?
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::{MemoryContent, MemoryRecord, StoreType};

    fn make_record(text: &str) -> MemoryRecord {
        MemoryRecord::new_text(StoreType::Semantic, text)
    }

    fn make_record_with_summary(text: &str, summary: &str) -> MemoryRecord {
        let mut rec = MemoryRecord::new_text(StoreType::Semantic, text);
        rec.content.summary = Some(summary.to_string());
        rec
    }

    fn temp_store() -> SemanticStore {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.redb");
        // Leak the dir so it isn't dropped (and deleted) while the store is alive.
        std::mem::forget(dir);
        SemanticStore::open(path).unwrap()
    }

    // -----------------------------------------------------------------------
    // 1. Node CRUD
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn node_store_and_get() {
        let store = temp_store();
        let rec = make_record("Rust is a systems programming language");
        let id = store.store(rec.clone()).await.unwrap();
        let fetched = store.get(&id).await.unwrap().expect("record must exist");
        assert_eq!(fetched.id, id);
        assert_eq!(
            fetched.text_content(),
            Some("Rust is a systems programming language")
        );
    }

    #[tokio::test]
    async fn node_delete() {
        let store = temp_store();
        let rec = make_record("To be deleted");
        let id = store.store(rec).await.unwrap();
        assert!(store.get(&id).await.unwrap().is_some());

        let deleted = store.delete(&id).await.unwrap();
        assert!(deleted);
        assert!(store.get(&id).await.unwrap().is_none());

        // Deleting again returns false.
        let deleted_again = store.delete(&id).await.unwrap();
        assert!(!deleted_again);
    }

    #[tokio::test]
    async fn node_count_and_list_ids() {
        let store = temp_store();
        assert_eq!(store.count().await.unwrap(), 0);

        let id1 = store.store(make_record("alpha")).await.unwrap();
        let id2 = store.store(make_record("beta")).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        let ids = store.list_ids().await.unwrap();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[tokio::test]
    async fn node_update_fidelity() {
        let store = temp_store();
        let rec = make_record("Fidelity test");
        let id = store.store(rec).await.unwrap();

        let new_fidelity = FidelityState {
            score: 0.42,
            ..Default::default()
        };
        store.update_fidelity(&id, new_fidelity).await.unwrap();

        let fetched = store.get(&id).await.unwrap().unwrap();
        assert!((fetched.fidelity.score - 0.42).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn node_update_record() {
        let store = temp_store();
        let rec = make_record("Original content");
        let id = store.store(rec).await.unwrap();

        let new_content = MemoryContent {
            blocks: vec![cerememory_core::ContentBlock {
                modality: cerememory_core::Modality::Text,
                format: "text/plain".to_string(),
                data: b"Updated content".to_vec(),
                embedding: None,
            }],
            summary: None,
        };

        store
            .update_record(&id, Some(new_content), None, None)
            .await
            .unwrap();

        let fetched = store.get(&id).await.unwrap().unwrap();
        assert_eq!(fetched.text_content(), Some("Updated content"));
        assert_eq!(fetched.version, 2);
    }

    // -----------------------------------------------------------------------
    // 2. Edge add/remove
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn edge_add_and_remove() {
        let store = temp_store();
        let a = store.store(make_record("Node A")).await.unwrap();
        let b = store.store(make_record("Node B")).await.unwrap();

        store
            .add_edge(a, b, AssociationType::Semantic, 0.9)
            .await
            .unwrap();

        let neighbors = store.get_neighbors(a).await.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].target_id, b);
        assert!((neighbors[0].weight - 0.9).abs() < f64::EPSILON);
        assert_eq!(neighbors[0].association_type, AssociationType::Semantic);

        let removed = store
            .remove_edge(a, b, AssociationType::Semantic)
            .await
            .unwrap();
        assert!(removed);

        let neighbors_after = store.get_neighbors(a).await.unwrap();
        assert!(neighbors_after.is_empty());
    }

    #[tokio::test]
    async fn remove_nonexistent_edge_returns_false() {
        let store = temp_store();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let removed = store
            .remove_edge(a, b, AssociationType::Temporal)
            .await
            .unwrap();
        assert!(!removed);
    }

    // -----------------------------------------------------------------------
    // 3. get_neighbors
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn get_neighbors_multiple_edges() {
        let store = temp_store();
        let center = store.store(make_record("Center")).await.unwrap();
        let n1 = store.store(make_record("N1")).await.unwrap();
        let n2 = store.store(make_record("N2")).await.unwrap();
        let n3 = store.store(make_record("N3")).await.unwrap();

        store
            .add_edge(center, n1, AssociationType::Semantic, 0.8)
            .await
            .unwrap();
        store
            .add_edge(center, n2, AssociationType::Causal, 0.5)
            .await
            .unwrap();
        store
            .add_edge(center, n3, AssociationType::Temporal, 0.3)
            .await
            .unwrap();

        let neighbors = store.get_neighbors(center).await.unwrap();
        assert_eq!(neighbors.len(), 3);
        let target_ids: Vec<Uuid> = neighbors.iter().map(|a| a.target_id).collect();
        assert!(target_ids.contains(&n1));
        assert!(target_ids.contains(&n2));
        assert!(target_ids.contains(&n3));
    }

    // -----------------------------------------------------------------------
    // 4. Bidirectional edge traversal
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn bidirectional_edge_traversal() {
        let store = temp_store();
        let a = store.store(make_record("A")).await.unwrap();
        let b = store.store(make_record("B")).await.unwrap();

        store
            .add_edge(a, b, AssociationType::Semantic, 0.7)
            .await
            .unwrap();

        // Forward: a → b
        let fwd = store.get_neighbors(a).await.unwrap();
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].target_id, b);

        // Reverse: who points to b? → a
        let rev = store.get_reverse_neighbors(b).await.unwrap();
        assert_eq!(rev.len(), 1);
        assert_eq!(rev[0].0, a);
        assert_eq!(rev[0].1, AssociationType::Semantic);

        // b should have no forward edges.
        let b_fwd = store.get_neighbors(b).await.unwrap();
        assert!(b_fwd.is_empty());

        // a should have no reverse edges.
        let a_rev = store.get_reverse_neighbors(a).await.unwrap();
        assert!(a_rev.is_empty());
    }

    // -----------------------------------------------------------------------
    // 5. get_subgraph with depth
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn get_subgraph_depth_0() {
        let store = temp_store();
        let a = store.store(make_record("Root")).await.unwrap();
        let _b = store.store(make_record("Child")).await.unwrap();
        store
            .add_edge(a, _b, AssociationType::Semantic, 0.5)
            .await
            .unwrap();

        // depth=0 → only the root itself.
        let sub = store.get_subgraph(a, 0).await.unwrap();
        assert_eq!(sub.len(), 1);
        assert_eq!(sub[0].id, a);
    }

    #[tokio::test]
    async fn get_subgraph_depth_1() {
        let store = temp_store();
        let a = store.store(make_record("Root")).await.unwrap();
        let b = store.store(make_record("Child1")).await.unwrap();
        let c = store.store(make_record("Child2")).await.unwrap();
        let d = store.store(make_record("GrandChild")).await.unwrap();

        store
            .add_edge(a, b, AssociationType::Semantic, 0.5)
            .await
            .unwrap();
        store
            .add_edge(a, c, AssociationType::Causal, 0.6)
            .await
            .unwrap();
        store
            .add_edge(b, d, AssociationType::Temporal, 0.4)
            .await
            .unwrap();

        // depth=1 → root + direct children (b, c). NOT grandchild d.
        let sub = store.get_subgraph(a, 1).await.unwrap();
        let ids: Vec<Uuid> = sub.iter().map(|r| r.id).collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
        assert!(ids.contains(&c));
        assert!(!ids.contains(&d));
    }

    #[tokio::test]
    async fn get_subgraph_depth_2() {
        let store = temp_store();
        let a = store.store(make_record("Root")).await.unwrap();
        let b = store.store(make_record("Child")).await.unwrap();
        let c = store.store(make_record("GrandChild")).await.unwrap();

        store
            .add_edge(a, b, AssociationType::Semantic, 0.5)
            .await
            .unwrap();
        store
            .add_edge(b, c, AssociationType::Causal, 0.6)
            .await
            .unwrap();

        // depth=2 → a, b, c
        let sub = store.get_subgraph(a, 2).await.unwrap();
        let ids: Vec<Uuid> = sub.iter().map(|r| r.id).collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
        assert!(ids.contains(&c));
    }

    // -----------------------------------------------------------------------
    // 6. query_text
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn query_text_matches_content() {
        let store = temp_store();
        store
            .store(make_record("The quick brown fox"))
            .await
            .unwrap();
        store
            .store(make_record("Lazy dog sleeping"))
            .await
            .unwrap();
        store
            .store(make_record("Fox and dog are friends"))
            .await
            .unwrap();

        let results = store.query_text("fox", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn query_text_matches_summary() {
        let store = temp_store();
        store
            .store(make_record_with_summary("body text", "Photosynthesis overview"))
            .await
            .unwrap();

        let results = store.query_text("photosynthesis", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn query_text_respects_limit() {
        let store = temp_store();
        for i in 0..10 {
            store
                .store(make_record(&format!("keyword-match item {}", i)))
                .await
                .unwrap();
        }

        let results = store.query_text("keyword-match", 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 7. Concept index
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn concept_index_stores_and_retrieves() {
        let store = temp_store();
        let rec1 = make_record_with_summary("data1", "Rust programming");
        let rec2 = make_record_with_summary("data2", "Rust compiler");
        let rec3 = make_record_with_summary("data3", "Python scripting");

        let id1 = store.store(rec1).await.unwrap();
        let id2 = store.store(rec2).await.unwrap();
        let _id3 = store.store(rec3).await.unwrap();

        let rust_ids = store.get_concept_ids("rust").await.unwrap();
        assert_eq!(rust_ids.len(), 2);
        assert!(rust_ids.contains(&id1));
        assert!(rust_ids.contains(&id2));

        let python_ids = store.get_concept_ids("python").await.unwrap();
        assert_eq!(python_ids.len(), 1);
    }

    #[tokio::test]
    async fn concept_index_case_insensitive() {
        let store = temp_store();
        let rec = make_record_with_summary("data", "Machine Learning");
        store.store(rec).await.unwrap();

        let ids = store.get_concept_ids("machine").await.unwrap();
        assert_eq!(ids.len(), 1);

        let ids_upper = store.get_concept_ids("MACHINE").await.unwrap();
        assert_eq!(ids_upper.len(), 1);
    }

    #[tokio::test]
    async fn concept_index_cleaned_on_delete() {
        let store = temp_store();
        let rec = make_record_with_summary("data", "Quantum computing");
        let id = store.store(rec).await.unwrap();

        assert_eq!(store.get_concept_ids("quantum").await.unwrap().len(), 1);

        store.delete(&id).await.unwrap();

        assert_eq!(store.get_concept_ids("quantum").await.unwrap().len(), 0);
    }
}
