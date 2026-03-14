//! HNSW-based approximate nearest neighbor index.
//!
//! Wraps `hnsw_rs` to provide O(log n) vector search when the index
//! exceeds a configurable threshold. Below the threshold, the existing
//! brute-force search is used instead.
//!
//! The HNSW graph lives entirely in memory and is rebuilt from the
//! redb-backed vector store on startup or when crossing the threshold.

use std::collections::HashMap;
use std::sync::RwLock;

use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::DistCosine;
use tracing::info;
use uuid::Uuid;

use cerememory_core::error::CerememoryError;

/// Default HNSW construction parameters.
const DEFAULT_MAX_NB_CONNECTION: usize = 16;
const DEFAULT_MAX_LAYER: usize = 16;
const DEFAULT_EF_CONSTRUCTION: usize = 200;
/// ef_search parameter for queries — higher = more accurate, slower.
const DEFAULT_EF_SEARCH: usize = 64;

/// HNSW approximate nearest neighbor index.
///
/// Manages an in-memory HNSW graph with UUID ↔ DataId mapping.
/// Thread-safe via `RwLock`.
pub struct HnswVectorIndex {
    threshold: usize,
    state: RwLock<Option<HnswState>>,
}

/// Internal state wrapping the HNSW graph.
///
/// The `Hnsw` type requires `'b: 'static` when T=f32 (Send+Sync+Clone+'static),
/// so we can store it in a struct directly.
struct HnswState {
    hnsw: Hnsw<'static, f32, DistCosine>,
    uuid_to_dataid: HashMap<Uuid, usize>,
    dataid_to_uuid: HashMap<usize, Uuid>,
    next_id: usize,
    dimension: usize,
}

/// A search result from the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswSearchHit {
    pub record_id: Uuid,
    /// Cosine similarity (1.0 - cosine_distance).
    pub similarity: f32,
}

impl HnswVectorIndex {
    /// Create a new HNSW index with the given activation threshold.
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            state: RwLock::new(None),
        }
    }

    /// Whether the HNSW graph is active (built and available for search).
    pub fn is_active(&self) -> bool {
        self.state.read().unwrap().is_some()
    }

    /// Number of vectors in the HNSW graph, or 0 if not active.
    pub fn count(&self) -> usize {
        self.state
            .read()
            .unwrap()
            .as_ref()
            .map_or(0, |s| s.uuid_to_dataid.len())
    }

    /// Get the activation threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Insert a single vector into the HNSW graph.
    /// No-op if the graph is not active.
    pub fn insert(&self, id: Uuid, embedding: &[f32]) -> Result<(), CerememoryError> {
        let mut guard = self.state.write().unwrap();
        let state = match guard.as_mut() {
            Some(s) => s,
            None => return Ok(()),
        };

        if embedding.len() != state.dimension {
            return Err(CerememoryError::Validation(format!(
                "HNSW dimension mismatch: expected {}, got {}",
                state.dimension,
                embedding.len()
            )));
        }

        // If already present, skip (HNSW doesn't support in-place update)
        if state.uuid_to_dataid.contains_key(&id) {
            return Ok(());
        }

        let data_id = state.next_id;
        state.next_id += 1;

        let owned: Vec<f32> = embedding.to_vec();
        state.hnsw.insert((&owned, data_id));
        state.uuid_to_dataid.insert(id, data_id);
        state.dataid_to_uuid.insert(data_id, id);

        Ok(())
    }

    /// Remove a vector from the UUID mapping.
    /// Note: hnsw_rs doesn't support true removal. The vector remains in the
    /// graph but will be filtered out in search results. A periodic rebuild
    /// cleans up removed entries.
    pub fn remove(&self, id: &Uuid) {
        let mut guard = self.state.write().unwrap();
        if let Some(state) = guard.as_mut() {
            if let Some(data_id) = state.uuid_to_dataid.remove(id) {
                state.dataid_to_uuid.remove(&data_id);
            }
        }
    }

    /// Search for the top-k most similar vectors.
    /// Returns results sorted by similarity (highest first).
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<HnswSearchHit>, CerememoryError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let guard = self.state.read().unwrap();
        let state = match guard.as_ref() {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };

        if query.len() != state.dimension {
            return Err(CerememoryError::Validation(format!(
                "HNSW query dimension mismatch: expected {}, got {}",
                state.dimension,
                query.len()
            )));
        }

        // Request more candidates to account for removed entries
        let ef_search = DEFAULT_EF_SEARCH.max(limit * 2);
        let owned_query: Vec<f32> = query.to_vec();
        let neighbours = state.hnsw.search(&owned_query, limit + 10, ef_search);

        let mut results: Vec<HnswSearchHit> = neighbours
            .into_iter()
            .filter_map(|n| {
                state.dataid_to_uuid.get(&n.d_id).map(|uuid| HnswSearchHit {
                    record_id: *uuid,
                    // DistCosine returns 1.0 - cos_sim, so similarity = 1.0 - distance
                    similarity: 1.0 - n.distance,
                })
            })
            .take(limit)
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Rebuild the HNSW graph from a set of (UUID, embedding) entries.
    /// Called when crossing the activation threshold or on startup.
    pub fn rebuild(&self, entries: &[(Uuid, Vec<f32>)]) -> Result<(), CerememoryError> {
        if entries.is_empty() || entries.len() < self.threshold {
            *self.state.write().unwrap() = None;
            return Ok(());
        }

        let dimension = entries[0].1.len();
        if dimension == 0 {
            return Err(CerememoryError::Validation(
                "Cannot build HNSW with zero-dimension vectors".to_string(),
            ));
        }

        // Validate all dimensions match
        for (i, (_, emb)) in entries.iter().enumerate() {
            if emb.len() != dimension {
                return Err(CerememoryError::Validation(format!(
                    "HNSW dimension mismatch at entry {i}: expected {dimension}, got {}",
                    emb.len()
                )));
            }
        }

        let max_elements = entries.len() * 2; // Allow room for growth
        let hnsw = Hnsw::<f32, DistCosine>::new(
            DEFAULT_MAX_NB_CONNECTION,
            max_elements,
            DEFAULT_MAX_LAYER,
            DEFAULT_EF_CONSTRUCTION,
            DistCosine {},
        );

        let mut uuid_to_dataid = HashMap::with_capacity(entries.len());
        let mut dataid_to_uuid = HashMap::with_capacity(entries.len());

        for (data_id, (uuid, embedding)) in entries.iter().enumerate() {
            hnsw.insert((embedding, data_id));
            uuid_to_dataid.insert(*uuid, data_id);
            dataid_to_uuid.insert(data_id, *uuid);
        }

        let next_id = entries.len();

        *self.state.write().unwrap() = Some(HnswState {
            hnsw,
            uuid_to_dataid,
            dataid_to_uuid,
            next_id,
            dimension,
        });

        info!(
            count = entries.len(),
            dimension, "HNSW index rebuilt"
        );

        Ok(())
    }

    /// Deactivate the HNSW index (free memory).
    pub fn deactivate(&self) {
        *self.state.write().unwrap() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries(n: usize, dim: usize) -> Vec<(Uuid, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                (Uuid::now_v7(), v)
            })
            .collect()
    }

    #[test]
    fn hnsw_builds_above_threshold() {
        let idx = HnswVectorIndex::new(5);
        assert!(!idx.is_active());

        let entries = make_entries(10, 8);
        idx.rebuild(&entries).unwrap();
        assert!(idx.is_active());
        assert_eq!(idx.count(), 10);
    }

    #[test]
    fn hnsw_stays_inactive_below_threshold() {
        let idx = HnswVectorIndex::new(100);
        let entries = make_entries(10, 8);
        idx.rebuild(&entries).unwrap();
        assert!(!idx.is_active());
    }

    #[test]
    fn hnsw_topk_correct() {
        let idx = HnswVectorIndex::new(3);
        let entries = vec![
            (Uuid::now_v7(), vec![1.0, 0.0, 0.0]),
            (Uuid::now_v7(), vec![0.9, 0.4, 0.0]),
            (Uuid::now_v7(), vec![0.0, 1.0, 0.0]),
            (Uuid::now_v7(), vec![0.0, 0.0, 1.0]),
        ];
        let target_id = entries[0].0;
        idx.rebuild(&entries).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].record_id, target_id);
        assert!(results[0].similarity > results[1].similarity);
    }

    #[test]
    fn hnsw_vs_brute_force_consistency() {
        let idx = HnswVectorIndex::new(3);
        let entries = vec![
            (Uuid::now_v7(), vec![1.0, 0.0, 0.0]),
            (Uuid::now_v7(), vec![0.0, 1.0, 0.0]),
            (Uuid::now_v7(), vec![0.0, 0.0, 1.0]),
        ];
        let closest_id = entries[0].0;
        idx.rebuild(&entries).unwrap();

        let results = idx.search(&[0.95, 0.05, 0.0], 1).unwrap();
        assert_eq!(results[0].record_id, closest_id);
    }

    #[test]
    fn hnsw_insert_incremental() {
        let idx = HnswVectorIndex::new(3);
        let entries = make_entries(5, 4);
        idx.rebuild(&entries).unwrap();

        let new_id = Uuid::now_v7();
        idx.insert(new_id, &[0.5, 0.5, 0.5, 0.5]).unwrap();
        assert_eq!(idx.count(), 6);
    }

    #[test]
    fn hnsw_remove() {
        let idx = HnswVectorIndex::new(3);
        let entries = make_entries(5, 4);
        let remove_id = entries[0].0;
        idx.rebuild(&entries).unwrap();

        idx.remove(&remove_id);
        assert_eq!(idx.count(), 4);

        // Removed vector should not appear in results
        let results = idx.search(&entries[0].1, 5).unwrap();
        assert!(!results.iter().any(|r| r.record_id == remove_id));
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let idx = HnswVectorIndex::new(2);
        let entries = vec![
            (Uuid::now_v7(), vec![1.0, 0.0]),
            (Uuid::now_v7(), vec![0.0, 1.0]),
        ];
        idx.rebuild(&entries).unwrap();

        let result = idx.insert(Uuid::now_v7(), &[1.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn empty_index_search() {
        let idx = HnswVectorIndex::new(10);
        let results = idx.search(&[1.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn single_vector_search() {
        let idx = HnswVectorIndex::new(1);
        let id = Uuid::now_v7();
        let entries = vec![(id, vec![1.0, 0.0, 0.0])];
        idx.rebuild(&entries).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record_id, id);
        assert!((results[0].similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn hnsw_rebuild_preserves_results() {
        let idx = HnswVectorIndex::new(3);
        let entries = vec![
            (Uuid::now_v7(), vec![1.0, 0.0, 0.0]),
            (Uuid::now_v7(), vec![0.0, 1.0, 0.0]),
            (Uuid::now_v7(), vec![0.0, 0.0, 1.0]),
        ];
        let target_id = entries[0].0;

        idx.rebuild(&entries).unwrap();
        let r1 = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();

        idx.rebuild(&entries).unwrap();
        let r2 = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();

        assert_eq!(r1[0].record_id, target_id);
        assert_eq!(r2[0].record_id, target_id);
    }

    #[test]
    fn search_after_remove() {
        let idx = HnswVectorIndex::new(3);
        let entries = vec![
            (Uuid::now_v7(), vec![1.0, 0.0, 0.0]),
            (Uuid::now_v7(), vec![0.9, 0.4, 0.0]),
            (Uuid::now_v7(), vec![0.0, 1.0, 0.0]),
        ];
        let second_id = entries[1].0;
        idx.rebuild(&entries).unwrap();

        // Remove the closest match
        idx.remove(&entries[0].0);

        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].record_id, second_id);
    }

    #[test]
    fn threshold_transition_up() {
        let idx = HnswVectorIndex::new(5);
        let small = make_entries(3, 4);
        idx.rebuild(&small).unwrap();
        assert!(!idx.is_active());

        let large = make_entries(6, 4);
        idx.rebuild(&large).unwrap();
        assert!(idx.is_active());
    }

    #[test]
    fn threshold_transition_down() {
        let idx = HnswVectorIndex::new(5);
        let large = make_entries(6, 4);
        idx.rebuild(&large).unwrap();
        assert!(idx.is_active());

        let small = make_entries(3, 4);
        idx.rebuild(&small).unwrap();
        assert!(!idx.is_active());
    }

    #[test]
    fn search_quality_at_moderate_scale() {
        let idx = HnswVectorIndex::new(10);
        let dim = 32;
        let n = 500;

        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.1).sin())
                .collect();
            entries.push((Uuid::now_v7(), v));
        }

        let target_id = entries[0].0;
        let query = entries[0].1.clone();

        idx.rebuild(&entries).unwrap();

        let results = idx.search(&query, 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record_id, target_id);
        assert!(results[0].similarity > 0.99);
    }
}
