//! Brute-force cosine similarity vector search for Cerememory.
//!
//! Stores embedding vectors in redb and performs exhaustive nearest-neighbor
//! search using a BinaryHeap for O(n log k) top-k selection.
//! Sufficient for the expected scale (thousands to tens of thousands
//! of records). ANN indexing deferred to Phase 3.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use ordered_float::OrderedFloat;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

use cerememory_core::error::CerememoryError;

const VECTORS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("embedding_vectors");

/// A hit from the vector similarity search.
#[derive(Debug, Clone)]
pub struct VectorSearchHit {
    pub record_id: Uuid,
    pub similarity: f32,
}

/// Brute-force vector index backed by redb.
pub struct VectorIndex {
    db: Arc<Database>,
}

impl VectorIndex {
    fn ensure_table(db: &Database) -> Result<(), CerememoryError> {
        let txn = db
            .begin_write()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex txn: {e}")))?;
        {
            let _table = txn
                .open_table(VECTORS)
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
        }
        txn.commit()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex commit: {e}")))?;
        Ok(())
    }

    /// Open or create a file-backed vector index.
    pub fn open(path: &str) -> Result<Self, CerememoryError> {
        let db = Database::create(path)
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex db open: {e}")))?;
        Self::ensure_table(&db)?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Create an in-memory vector index (for testing).
    pub fn open_in_memory() -> Result<Self, CerememoryError> {
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex in-memory: {e}")))?;
        Self::ensure_table(&db)?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Remove all vectors from the index.
    pub fn clear(&self) -> Result<(), CerememoryError> {
        let txn = self
            .db
            .begin_write()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex txn: {e}")))?;
        {
            let mut table = txn
                .open_table(VECTORS)
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
            // Collect all keys first to avoid borrow conflict
            let keys: Vec<Vec<u8>> = table
                .iter()
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex iter: {e}")))?
                .filter_map(|entry| entry.ok().map(|(k, _)| k.value().to_vec()))
                .collect();
            for key in keys {
                table
                    .remove(key.as_slice())
                    .map_err(|e| CerememoryError::Storage(format!("VectorIndex remove: {e}")))?;
            }
        }
        txn.commit()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex commit: {e}")))?;
        Ok(())
    }

    /// Validate an embedding vector: must not be empty and must not contain NaN/Inf.
    fn validate_embedding(embedding: &[f32]) -> Result<(), CerememoryError> {
        if embedding.is_empty() {
            return Err(CerememoryError::Validation(
                "Embedding vector must not be empty".to_string(),
            ));
        }
        if embedding.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(CerememoryError::Validation(
                "Embedding vector contains NaN or Inf".to_string(),
            ));
        }
        Ok(())
    }

    /// Insert or update an embedding vector. The vector is L2-normalized before storage.
    /// Rejects empty vectors or vectors containing NaN/Inf.
    pub fn upsert(&self, id: Uuid, embedding: &[f32]) -> Result<(), CerememoryError> {
        Self::validate_embedding(embedding)?;
        let normalized = l2_normalize(embedding);
        let bytes =
            rmp_serde::to_vec(&normalized).map_err(|e| CerememoryError::Serialization(e.to_string()))?;

        let txn = self
            .db
            .begin_write()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex txn: {e}")))?;
        {
            let mut table = txn
                .open_table(VECTORS)
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
            table
                .insert(id.as_bytes().as_slice(), bytes.as_slice())
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex insert: {e}")))?;
        }
        txn.commit()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex commit: {e}")))?;
        Ok(())
    }

    /// Insert or update multiple embedding vectors in a single write transaction.
    /// Each vector is validated and L2-normalized before storage. Serialization
    /// is performed before opening the transaction to minimize lock duration.
    pub fn upsert_batch(&self, entries: &[(Uuid, &[f32])]) -> Result<(), CerememoryError> {
        if entries.is_empty() {
            return Ok(());
        }
        // Validate and serialize all embeddings before opening the transaction
        let prepared: Vec<(Uuid, Vec<u8>)> = entries
            .iter()
            .map(|(id, embedding)| {
                Self::validate_embedding(embedding)?;
                let normalized = l2_normalize(embedding);
                let bytes = rmp_serde::to_vec(&normalized)
                    .map_err(|e| CerememoryError::Serialization(e.to_string()))?;
                Ok((*id, bytes))
            })
            .collect::<Result<Vec<_>, CerememoryError>>()?;

        let txn = self
            .db
            .begin_write()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex txn: {e}")))?;
        {
            let mut table = txn
                .open_table(VECTORS)
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
            for (id, bytes) in &prepared {
                table
                    .insert(id.as_bytes().as_slice(), bytes.as_slice())
                    .map_err(|e| CerememoryError::Storage(format!("VectorIndex insert: {e}")))?;
            }
        }
        txn.commit()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex commit: {e}")))?;
        Ok(())
    }

    /// Remove a vector by record ID.
    pub fn remove(&self, id: Uuid) -> Result<(), CerememoryError> {
        let txn = self
            .db
            .begin_write()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex txn: {e}")))?;
        {
            let mut table = txn
                .open_table(VECTORS)
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
            table
                .remove(id.as_bytes().as_slice())
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex remove: {e}")))?;
        }
        txn.commit()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex commit: {e}")))?;
        Ok(())
    }

    /// Search for the top-k most similar vectors to the query.
    ///
    /// Uses a min-heap (BinaryHeap with Reverse) for O(n log k) top-k selection,
    /// which is more efficient than the naive O(n log n) sort+truncate approach
    /// when k << n.
    ///
    /// The query embedding is L2-normalized before comparison.
    pub fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorSearchHit>, CerememoryError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        Self::validate_embedding(query_embedding)?;

        let query_norm = l2_normalize(query_embedding);

        let txn = self
            .db
            .begin_read()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex read txn: {e}")))?;
        let table = txn
            .open_table(VECTORS)
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;

        let iter = table
            .iter()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex iter: {e}")))?;

        // Min-heap of size k: keeps the k largest similarity scores.
        // Reverse turns BinaryHeap (max-heap) into a min-heap so we can
        // efficiently evict the smallest of our top-k candidates.
        let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, Uuid)>> = BinaryHeap::new();

        for entry in iter {
            let (key, value) = entry
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex entry: {e}")))?;

            let key_bytes = key.value();
            if key_bytes.len() != 16 {
                continue;
            }
            let id = Uuid::from_bytes(key_bytes.try_into().unwrap());

            let vec: Vec<f32> = rmp_serde::from_slice(value.value())
                .map_err(|e| CerememoryError::Serialization(e.to_string()))?;

            let sim = cosine_similarity(&query_norm, &vec);
            let score = OrderedFloat(sim);

            if heap.len() < limit {
                heap.push(Reverse((score, id)));
            } else if let Some(&Reverse((min_score, _))) = heap.peek() {
                if score > min_score {
                    heap.pop();
                    heap.push(Reverse((score, id)));
                }
            }
        }

        // Extract results sorted descending by similarity.
        // into_sorted_vec() returns ascending order for BinaryHeap's Ord.
        // Since we store Reverse, ascending Reverse order means descending
        // by inner score — exactly the order we want (highest similarity first).
        let results: Vec<VectorSearchHit> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse((score, id))| VectorSearchHit {
                record_id: id,
                similarity: score.into_inner(),
            })
            .collect();
        Ok(results)
    }

    /// Count stored vectors.
    pub fn count(&self) -> Result<usize, CerememoryError> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex read txn: {e}")))?;
        let table = txn
            .open_table(VECTORS)
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
        Ok(table.len().unwrap_or(0) as usize)
    }
}

/// Cosine similarity between two vectors (assumed to be L2-normalized).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalize a vector. Returns a zero vector if the input norm is zero.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_unit() {
        let v = l2_normalize(&[3.0, 4.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn upsert_and_search() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        idx.upsert(id1, &[1.0, 0.0, 0.0]).unwrap();
        idx.upsert(id2, &[0.0, 1.0, 0.0]).unwrap();

        // Search for vector similar to id1
        let hits = idx.search(&[1.0, 0.1, 0.0], 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].record_id, id1);
        assert!(hits[0].similarity > hits[1].similarity);
    }

    #[test]
    fn identical_vector_similarity_one() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();
        let v = vec![0.5, 0.3, 0.8, 0.1];
        idx.upsert(id, &v).unwrap();

        let hits = idx.search(&v, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn remove_vector() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();
        idx.upsert(id, &[1.0, 0.0]).unwrap();
        assert_eq!(idx.count().unwrap(), 1);

        idx.remove(id).unwrap();
        assert_eq!(idx.count().unwrap(), 0);

        let hits = idx.search(&[1.0, 0.0], 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn topk_ordering() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::now_v7()).collect();

        // Vectors at different angles
        idx.upsert(ids[0], &[1.0, 0.0]).unwrap();
        idx.upsert(ids[1], &[0.9, 0.4]).unwrap();
        idx.upsert(ids[2], &[0.5, 0.87]).unwrap();
        idx.upsert(ids[3], &[0.0, 1.0]).unwrap();
        idx.upsert(ids[4], &[-1.0, 0.0]).unwrap();

        let hits = idx.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(hits.len(), 3);
        // Should be sorted: ids[0], ids[1], ids[2]
        assert_eq!(hits[0].record_id, ids[0]);
        assert_eq!(hits[1].record_id, ids[1]);
        assert!(hits[0].similarity >= hits[1].similarity);
        assert!(hits[1].similarity >= hits[2].similarity);
    }

    #[test]
    fn empty_index_search() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let hits = idx.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn upsert_overwrites() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();

        idx.upsert(id, &[1.0, 0.0]).unwrap();
        idx.upsert(id, &[0.0, 1.0]).unwrap();

        assert_eq!(idx.count().unwrap(), 1);

        // Should now be similar to [0, 1], not [1, 0]
        let hits = idx.search(&[0.0, 1.0], 1).unwrap();
        assert!((hits[0].similarity - 1.0).abs() < 1e-5);
    }

    // --- BinaryHeap top-k tests ---

    #[test]
    fn topk_heap_ordering() {
        // Verify BinaryHeap returns correct top-k in descending similarity order
        let idx = VectorIndex::open_in_memory().unwrap();
        let mut ids = Vec::new();
        // Insert 10 vectors at varying angles from [1,0]
        for i in 0..10 {
            let id = Uuid::now_v7();
            ids.push(id);
            let angle = (i as f32) * std::f32::consts::PI / 10.0;
            idx.upsert(id, &[angle.cos(), angle.sin()]).unwrap();
        }

        let hits = idx.search(&[1.0, 0.0], 5).unwrap();
        assert_eq!(hits.len(), 5);

        // Verify descending order
        for w in hits.windows(2) {
            assert!(
                w[0].similarity >= w[1].similarity,
                "Results not in descending order: {} < {}",
                w[0].similarity,
                w[1].similarity
            );
        }

        // The top hit should be ids[0] (angle=0, most similar to [1,0])
        assert_eq!(hits[0].record_id, ids[0]);
    }

    #[test]
    fn topk_heap_limit() {
        // Verify limit is respected even when more results are available
        let idx = VectorIndex::open_in_memory().unwrap();
        for _ in 0..20 {
            idx.upsert(Uuid::now_v7(), &[1.0, 0.0, 0.0]).unwrap();
        }

        let hits = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(hits.len(), 5);

        let hits = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn search_limit_zero() {
        let idx = VectorIndex::open_in_memory().unwrap();
        idx.upsert(Uuid::now_v7(), &[1.0, 0.0]).unwrap();
        let hits = idx.search(&[1.0, 0.0], 0).unwrap();
        assert!(hits.is_empty());
    }

    // --- upsert_batch tests ---

    #[test]
    fn upsert_batch_inserts() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        idx.upsert_batch(&[
            (id1, &[1.0, 0.0, 0.0][..]),
            (id2, &[0.0, 1.0, 0.0][..]),
            (id3, &[0.0, 0.0, 1.0][..]),
        ])
        .unwrap();

        assert_eq!(idx.count().unwrap(), 3);
    }

    #[test]
    fn upsert_batch_empty() {
        let idx = VectorIndex::open_in_memory().unwrap();
        idx.upsert_batch(&[]).unwrap();
        assert_eq!(idx.count().unwrap(), 0);
    }

    #[test]
    fn upsert_batch_overwrites() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();

        idx.upsert(id, &[1.0, 0.0]).unwrap();

        // Batch overwrite the same ID
        idx.upsert_batch(&[(id, &[0.0, 1.0][..])]).unwrap();

        assert_eq!(idx.count().unwrap(), 1);

        // Should now be similar to [0, 1], not [1, 0]
        let hits = idx.search(&[0.0, 1.0], 1).unwrap();
        assert!((hits[0].similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn search_after_batch() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        idx.upsert_batch(&[
            (id1, &[1.0, 0.0][..]),
            (id2, &[0.0, 1.0][..]),
        ])
        .unwrap();

        let hits = idx.search(&[1.0, 0.1], 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id1);
    }

    #[test]
    fn upsert_batch_validates_all_before_write() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        // Second entry has NaN — entire batch should fail
        let result = idx.upsert_batch(&[
            (id1, &[1.0, 0.0][..]),
            (id2, &[f32::NAN, 0.0][..]),
        ]);
        assert!(result.is_err());

        // Nothing should have been written
        assert_eq!(idx.count().unwrap(), 0);
    }

    // --- Embedding validation negative tests ---

    #[test]
    fn search_rejects_empty_embedding() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let result = idx.search(&[], 10);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("empty"), "error should mention empty: {err}");
    }

    #[test]
    fn search_rejects_nan_embedding() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let result = idx.search(&[f32::NAN, 0.5, 0.5], 10);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("NaN") || err.contains("Inf"),
            "error should mention NaN or Inf: {err}"
        );
    }

    #[test]
    fn search_rejects_inf_embedding() {
        let idx = VectorIndex::open_in_memory().unwrap();
        let result = idx.search(&[f32::INFINITY, 0.5, 0.5], 10);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("NaN") || err.contains("Inf"),
            "error should mention NaN or Inf: {err}"
        );
    }
}
