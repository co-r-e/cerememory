//! Deterministic vector search for Cerememory.
//!
//! Stores normalized embedding vectors in redb, which is the only source of
//! truth. Search uses an exact brute-force cosine scan so correctness does not
//! depend on a secondary in-memory ANN graph.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use ordered_float::OrderedFloat;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use tracing::{debug, warn};
use uuid::Uuid;

use cerememory_core::error::CerememoryError;
use cerememory_store_common::{StoreRecordCodec, StoreRecordMigrationStats};

const VECTORS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("embedding_vectors");

/// Search backend currently used by the vector index.
pub const VECTOR_SEARCH_BACKEND: &str = "brute_force";

/// A hit from the vector similarity search.
#[derive(Debug, Clone)]
pub struct VectorSearchHit {
    pub record_id: Uuid,
    pub similarity: f32,
}

/// Vector index backed by redb with exact cosine search.
pub struct VectorIndex {
    db: Arc<Database>,
    codec: StoreRecordCodec,
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
        Self::open_with_codec(path, StoreRecordCodec::plaintext())
    }

    /// Open or create a file-backed vector index with a custom storage codec.
    pub fn open_with_codec(path: &str, codec: StoreRecordCodec) -> Result<Self, CerememoryError> {
        let db = Database::create(path)
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex db open: {e}")))?;
        Self::ensure_table(&db)?;
        Ok(Self {
            db: Arc::new(db),
            codec,
        })
    }

    /// Create an in-memory vector index (for testing).
    pub fn open_in_memory() -> Result<Self, CerememoryError> {
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex in-memory: {e}")))?;
        Self::ensure_table(&db)?;
        Ok(Self {
            db: Arc::new(db),
            codec: StoreRecordCodec::plaintext(),
        })
    }

    /// Rewrite legacy plaintext embedding payloads using this index's encrypted codec.
    pub fn migrate_plaintext_records_to_encrypted(
        &self,
    ) -> Result<StoreRecordMigrationStats, CerememoryError> {
        if !self.codec.encrypts_new_records() {
            return Err(CerememoryError::Validation(
                "store encryption passphrase is required before migrating vector records"
                    .to_string(),
            ));
        }

        let txn = self
            .db
            .begin_write()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex txn: {e}")))?;
        let mut stats = StoreRecordMigrationStats::empty();
        let mut rewrites: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        {
            let table = txn
                .open_table(VECTORS)
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
            for entry in table
                .iter()
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex iter: {e}")))?
            {
                let (key, value) = entry
                    .map_err(|e| CerememoryError::Storage(format!("VectorIndex entry: {e}")))?;
                let payload = value.value();
                let vector: Vec<f32> = self.codec.decode(payload)?;
                stats.records_total += 1;

                if StoreRecordCodec::is_encrypted_payload(payload) {
                    stats.records_already_encrypted += 1;
                } else {
                    rewrites.push((key.value().to_vec(), self.codec.encode(&vector)?));
                    stats.records_migrated += 1;
                }
            }
        }

        if !rewrites.is_empty() {
            let mut table = txn
                .open_table(VECTORS)
                .map_err(|e| CerememoryError::Storage(format!("VectorIndex table: {e}")))?;
            for (key, payload) in rewrites {
                table
                    .insert(key.as_slice(), payload.as_slice())
                    .map_err(|e| CerememoryError::Storage(format!("VectorIndex insert: {e}")))?;
            }
        }

        txn.commit()
            .map_err(|e| CerememoryError::Storage(format!("VectorIndex commit: {e}")))?;
        Ok(stats)
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
        let bytes = self.codec.encode(&normalized)?;

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
                let bytes = self.codec.encode(&normalized)?;
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
    /// Uses exact brute-force O(n log k) search via redb scan.
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
        debug!(
            backend = self.search_backend(),
            limit, "Searching vector index"
        );
        self.search_brute_force(&query_norm, limit)
    }

    /// Brute-force search over all vectors in redb.
    fn search_brute_force(
        &self,
        query_norm: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorSearchHit>, CerememoryError> {
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
        let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, Uuid)>> = BinaryHeap::new();

        for entry in iter {
            let (key, value) =
                entry.map_err(|e| CerememoryError::Storage(format!("VectorIndex entry: {e}")))?;

            let key_bytes = key.value();
            if key_bytes.len() != 16 {
                warn!(
                    key_len = key_bytes.len(),
                    "Skipping vector index entry with invalid key length"
                );
                continue;
            }
            let id = Uuid::from_bytes(key_bytes.try_into().expect("length validated above"));

            let vec: Vec<f32> = self.codec.decode(value.value())?;

            let sim = cosine_similarity(query_norm, &vec);
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

    /// Name of the active vector search backend.
    pub fn search_backend(&self) -> &'static str {
        VECTOR_SEARCH_BACKEND
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
    fn migrate_plaintext_records_to_encrypted_is_idempotent() {
        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = tmp_dir.path().join("vectors.redb");
        let db_path = db_path.to_string_lossy();
        let plaintext = VectorIndex::open(&db_path).unwrap();
        let id = Uuid::now_v7();
        plaintext.upsert(id, &[1.0, 0.0]).unwrap();
        drop(plaintext);

        let encrypted = VectorIndex::open_with_codec(
            &db_path,
            StoreRecordCodec::encrypted_from_passphrase("migration passphrase").unwrap(),
        )
        .unwrap();
        let stats = encrypted.migrate_plaintext_records_to_encrypted().unwrap();
        assert_eq!(stats.records_total, 1);
        assert_eq!(stats.records_migrated, 1);
        assert_eq!(stats.records_already_encrypted, 0);
        assert_eq!(encrypted.search(&[1.0, 0.0], 1).unwrap()[0].record_id, id);

        let stats = encrypted.migrate_plaintext_records_to_encrypted().unwrap();
        assert_eq!(stats.records_total, 1);
        assert_eq!(stats.records_migrated, 0);
        assert_eq!(stats.records_already_encrypted, 1);
        drop(encrypted);

        let plaintext_reopen = VectorIndex::open(&db_path).unwrap();
        let err = plaintext_reopen.search(&[1.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, CerememoryError::Unauthorized(_)));
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

        idx.upsert_batch(&[(id1, &[1.0, 0.0][..]), (id2, &[0.0, 1.0][..])])
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
        let result = idx.upsert_batch(&[(id1, &[1.0, 0.0][..]), (id2, &[f32::NAN, 0.0][..])]);
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
