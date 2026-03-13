//! Brute-force cosine similarity vector search for Cerememory.
//!
//! Stores embedding vectors in redb and performs exhaustive nearest-neighbor
//! search. Sufficient for the expected scale (thousands to tens of thousands
//! of records). ANN indexing deferred to Phase 3.

use std::sync::Arc;

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

    /// Insert or update an embedding vector. The vector is L2-normalized before storage.
    /// Rejects empty vectors or vectors containing NaN/Inf.
    pub fn upsert(&self, id: Uuid, embedding: &[f32]) -> Result<(), CerememoryError> {
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
    /// The query embedding is L2-normalized before comparison.
    pub fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorSearchHit>, CerememoryError> {
        let query_norm = l2_normalize(query_embedding);

        let txn = self
            .db
            .begin_read()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex read txn: {e}")))?;
        let table = txn
            .open_table(VECTORS)
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;

        let mut scored: Vec<(Uuid, f32)> = Vec::with_capacity(table.len().unwrap_or(0) as usize);

        let iter = table
            .iter()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex iter: {e}")))?;

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
            scored.push((id, sim));
        }

        // Sort descending by similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored
            .into_iter()
            .map(|(record_id, similarity)| VectorSearchHit {
                record_id,
                similarity,
            })
            .collect())
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
}
