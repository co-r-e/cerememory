//! Cerememory episodic store implementation backed by redb.
//!
//! This crate provides a persistent, file-based episodic memory store using
//! [redb](https://docs.rs/redb) as the storage engine. All records are serialized
//! via MessagePack (`rmp-serde`) and indexed by time and fidelity for efficient
//! temporal range queries and decay-based sweeps.
//!
//! Because redb is synchronous, every I/O operation is wrapped in
//! `tokio::task::spawn_blocking` so the async `Store` trait can be fulfilled
//! without blocking the Tokio runtime.

use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

use cerememory_core::error::CerememoryError;
use cerememory_core::traits::Store;
use cerememory_core::types::{
    Association, EmotionVector, FidelityState, MemoryContent, MemoryRecord, MetaMemory,
};
use cerememory_store_common::{
    fidelity_bucket, fidelity_key, get_all_sync, get_record_sync, record_matches_text, storage_err,
};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// Primary table: UUID (16 bytes) → MessagePack-encoded `MemoryRecord`.
const EPISODIC_RECORDS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("episodic_records");

/// Temporal index: (timestamp_millis_be ++ record_id) → ().
/// Composite key enables ordered scans by time.
const EPISODIC_TIME_INDEX: TableDefinition<&[u8], ()> = TableDefinition::new("episodic_time_index");

/// Fidelity index: (fidelity_bucket_u8 ++ record_id) → ().
/// Bucket = (fidelity.score * 100).round() clamped to [0, 100].
const EPISODIC_FIDELITY_INDEX: TableDefinition<&[u8], ()> =
    TableDefinition::new("episodic_fidelity_index");

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

/// Build the 24-byte time-index key: 8 bytes big-endian millis + 16 bytes UUID.
fn time_key(ts: &DateTime<Utc>, id: &Uuid) -> [u8; 24] {
    let mut buf = [0u8; 24];
    let millis = ts.timestamp_millis();
    buf[..8].copy_from_slice(&millis.to_be_bytes());
    buf[8..].copy_from_slice(id.as_bytes());
    buf
}

/// Extract the UUID from a 24-byte time-index key.
fn uuid_from_time_key(key: &[u8]) -> Uuid {
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&key[8..24]);
    Uuid::from_bytes(bytes)
}

// ---------------------------------------------------------------------------
// EpisodicStore
// ---------------------------------------------------------------------------

/// A redb-backed episodic memory store.
///
/// Thread-safe via `Arc<Database>`. All public methods are async and delegate
/// to `tokio::task::spawn_blocking` internally.
#[derive(Clone)]
pub struct EpisodicStore {
    db: Arc<Database>,
}

impl EpisodicStore {
    /// Open (or create) a persistent store at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CerememoryError> {
        let db = Database::create(path.as_ref())
            .map_err(|e| CerememoryError::Storage(format!("Failed to open redb database: {e}")))?;

        let store = Self { db: Arc::new(db) };
        store.ensure_tables()?;
        Ok(store)
    }

    /// Create an ephemeral in-memory store backed by a temporary file.
    ///
    /// Useful for testing. The file is automatically cleaned up on drop via `tempfile`.
    pub fn open_in_memory() -> Result<Self, CerememoryError> {
        let tmp = tempfile::NamedTempFile::new()
            .map_err(|e| CerememoryError::Storage(format!("Failed to create temp file: {e}")))?;
        // We intentionally persist the path — redb manages the file.
        let path = tmp.into_temp_path();
        // Remove the file so redb can create it fresh.
        let _ = std::fs::remove_file(&path);
        let db = Database::create(&path).map_err(|e| {
            CerememoryError::Storage(format!("Failed to open in-memory redb database: {e}"))
        })?;

        let store = Self { db: Arc::new(db) };
        store.ensure_tables()?;
        Ok(store)
    }

    /// Ensure all three tables exist by performing a write transaction.
    fn ensure_tables(&self) -> Result<(), CerememoryError> {
        let txn = self.db.begin_write().map_err(storage_err)?;
        {
            let _ = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;
            let _ = txn.open_table(EPISODIC_TIME_INDEX).map_err(storage_err)?;
            let _ = txn
                .open_table(EPISODIC_FIDELITY_INDEX)
                .map_err(storage_err)?;
        }
        txn.commit().map_err(storage_err)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Additional query methods (not part of the Store trait)
    // ------------------------------------------------------------------

    /// Return all records whose `created_at` falls within `[start, end)`.
    pub async fn query_temporal_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let time_table = txn.open_table(EPISODIC_TIME_INDEX).map_err(storage_err)?;
            let records_table = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

            let start_key = {
                let mut buf = [0u8; 24];
                buf[..8].copy_from_slice(&start.timestamp_millis().to_be_bytes());
                // all-zero UUID → minimum for this timestamp
                buf
            };
            let end_key = {
                let mut buf = [0u8; 24];
                buf[..8].copy_from_slice(&end.timestamp_millis().to_be_bytes());
                buf
            };

            let mut results = Vec::new();
            let range = time_table
                .range(start_key.as_slice()..end_key.as_slice())
                .map_err(storage_err)?;
            for entry in range {
                let (key_guard, _) = entry.map_err(storage_err)?;
                let key_bytes = key_guard.value();
                let id = uuid_from_time_key(key_bytes);
                if let Some(record) = get_record_sync(&records_table, &id)? {
                    results.push(record);
                }
            }
            Ok(results)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    /// Return all records whose fidelity score is strictly below `threshold`.
    pub async fn records_below_fidelity(
        &self,
        threshold: f64,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let fidelity_table = txn
                .open_table(EPISODIC_FIDELITY_INDEX)
                .map_err(storage_err)?;
            let records_table = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

            // Scan buckets [0, threshold_bucket + 1) and then exact-filter.
            // Because the index uses rounded buckets, values just below the
            // threshold can live in the threshold bucket.
            let threshold_bucket = fidelity_bucket(threshold);

            // Build range end key: (threshold_bucket + 1) ++ all-zero UUID.
            let end_key = {
                let mut buf = [0u8; 17];
                buf[0] = threshold_bucket.saturating_add(1);
                buf
            };

            let start_key = [0u8; 17]; // bucket 0, all-zero UUID

            let mut results = Vec::new();
            if threshold_bucket == 0 {
                // Nothing can be strictly below bucket 0
                return Ok(results);
            }

            let range = fidelity_table
                .range(start_key.as_slice()..end_key.as_slice())
                .map_err(storage_err)?;
            for entry in range {
                let (key_guard, _) = entry.map_err(storage_err)?;
                let key_bytes = key_guard.value();
                if key_bytes.len() >= 17 {
                    let mut id_bytes = [0u8; 16];
                    id_bytes.copy_from_slice(&key_bytes[1..17]);
                    let id = Uuid::from_bytes(id_bytes);
                    if let Some(record) = get_record_sync(&records_table, &id)? {
                        // Double-check against the actual fidelity score, because
                        // bucket rounding can be slightly off.
                        if record.fidelity.score < threshold {
                            results.push(record);
                        }
                    }
                }
            }

            Ok(results)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// Store trait implementation
// ---------------------------------------------------------------------------

impl Store for EpisodicStore {
    async fn store(&self, record: MemoryRecord) -> Result<Uuid, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let id = record.id;
            let packed = rmp_serde::to_vec(&record)
                .map_err(|e| CerememoryError::Serialization(format!("msgpack encode: {e}")))?;

            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;

                let mut time_idx = txn.open_table(EPISODIC_TIME_INDEX).map_err(storage_err)?;
                let tk = time_key(&record.created_at, &id);
                time_idx.insert(tk.as_slice(), ()).map_err(storage_err)?;

                let mut fidelity_idx = txn
                    .open_table(EPISODIC_FIDELITY_INDEX)
                    .map_err(storage_err)?;
                let fk = fidelity_key(record.fidelity.score, &id);
                fidelity_idx
                    .insert(fk.as_slice(), ())
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;

            Ok(id)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn get(&self, id: &Uuid) -> Result<Option<MemoryRecord>, CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;
            get_record_sync(&table, &id)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn delete(&self, id: &Uuid) -> Result<bool, CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_write().map_err(storage_err)?;
            let existed = {
                let mut records = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

                // Read within the write transaction to avoid TOCTOU
                let existing: Option<MemoryRecord> =
                    match records.get(id.as_bytes().as_slice()).map_err(storage_err)? {
                        Some(guard) => {
                            let record: MemoryRecord = rmp_serde::from_slice(guard.value())
                                .map_err(|e| {
                                    CerememoryError::Serialization(format!("msgpack decode: {e}"))
                                })?;
                            drop(guard);
                            Some(record)
                        }
                        None => None,
                    };

                if let Some(record) = existing {
                    records
                        .remove(id.as_bytes().as_slice())
                        .map_err(storage_err)?;

                    let mut time_idx = txn.open_table(EPISODIC_TIME_INDEX).map_err(storage_err)?;
                    let tk = time_key(&record.created_at, &id);
                    time_idx.remove(tk.as_slice()).map_err(storage_err)?;

                    let mut fidelity_idx = txn
                        .open_table(EPISODIC_FIDELITY_INDEX)
                        .map_err(storage_err)?;
                    let fk = fidelity_key(record.fidelity.score, &id);
                    fidelity_idx.remove(fk.as_slice()).map_err(storage_err)?;

                    true
                } else {
                    false
                }
            };
            txn.commit().map_err(storage_err)?;
            Ok(existed)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn update_fidelity(
        &self,
        id: &Uuid,
        fidelity: FidelityState,
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

                // Read within the write transaction to avoid TOCTOU
                let guard = records
                    .get(id.as_bytes().as_slice())
                    .map_err(storage_err)?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut record: MemoryRecord = rmp_serde::from_slice(guard.value())
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;
                drop(guard);

                let old_fidelity_score = record.fidelity.score;
                record.fidelity = fidelity;
                record.updated_at = Utc::now();

                let packed = rmp_serde::to_vec(&record)
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack encode: {e}")))?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;

                // Update fidelity index: remove old, insert new.
                let mut fidelity_idx = txn
                    .open_table(EPISODIC_FIDELITY_INDEX)
                    .map_err(storage_err)?;
                let old_fk = fidelity_key(old_fidelity_score, &id);
                fidelity_idx
                    .remove(old_fk.as_slice())
                    .map_err(storage_err)?;
                let new_fk = fidelity_key(record.fidelity.score, &id);
                fidelity_idx
                    .insert(new_fk.as_slice(), ())
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
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
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

            let mut results = Vec::new();
            let iter = table.iter().map_err(storage_err)?;
            for entry in iter {
                let (_, value_guard) = entry.map_err(storage_err)?;
                let record: MemoryRecord = rmp_serde::from_slice(value_guard.value())
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;

                if record_matches_text(&record, &query) {
                    results.push(record);
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            Ok(results)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn list_ids(&self) -> Result<Vec<Uuid>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

            let mut ids = Vec::new();
            let iter = table.iter().map_err(storage_err)?;
            for entry in iter {
                let (key_guard, _) = entry.map_err(storage_err)?;
                let key_bytes = key_guard.value();
                if key_bytes.len() == 16 {
                    let mut buf = [0u8; 16];
                    buf.copy_from_slice(key_bytes);
                    ids.push(Uuid::from_bytes(buf));
                }
            }
            Ok(ids)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn get_all(&self) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;
            get_all_sync(&table)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn count(&self) -> Result<usize, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;
            let len = table.len().map_err(storage_err)?;
            Ok(len as usize)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn update_record(
        &self,
        id: &Uuid,
        content: Option<MemoryContent>,
        emotion: Option<EmotionVector>,
        metadata: Option<serde_json::Value>,
        meta: Option<MetaMemory>,
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

                // Read within the write transaction to avoid TOCTOU
                let guard = records
                    .get(id.as_bytes().as_slice())
                    .map_err(storage_err)?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut record: MemoryRecord = rmp_serde::from_slice(guard.value())
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;
                drop(guard);

                if let Some(c) = content {
                    record.content = c;
                }
                if let Some(e) = emotion {
                    record.emotion = e;
                }
                if let Some(m) = metadata {
                    record.metadata = m;
                }
                if let Some(meta) = meta {
                    record.meta = meta;
                }
                record.updated_at = Utc::now();
                record.version += 1;

                let packed = rmp_serde::to_vec(&record)
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack encode: {e}")))?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    async fn replace_associations(
        &self,
        id: &Uuid,
        associations: Vec<Association>,
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;
                let guard = records
                    .get(id.as_bytes().as_slice())
                    .map_err(storage_err)?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut record: MemoryRecord = rmp_serde::from_slice(guard.value())
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;
                drop(guard);

                record.associations = associations;
                record.updated_at = Utc::now();
                record.version += 1;

                let packed = rmp_serde::to_vec(&record)
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack encode: {e}")))?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
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
            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(EPISODIC_RECORDS).map_err(storage_err)?;

                let guard = records
                    .get(id.as_bytes().as_slice())
                    .map_err(storage_err)?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut record: MemoryRecord = rmp_serde::from_slice(guard.value())
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;
                drop(guard);

                record.access_count = access_count;
                record.last_accessed_at = last_accessed_at;
                record.updated_at = Utc::now();

                let packed = rmp_serde::to_vec(&record)
                    .map_err(|e| CerememoryError::Serialization(format!("msgpack encode: {e}")))?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;
            Ok(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::types::{MemoryRecord, StoreType};
    use chrono::{Duration, Utc};
    /// Helper: create a store in a fresh temp directory.
    fn temp_store() -> EpisodicStore {
        EpisodicStore::open_in_memory().expect("failed to create in-memory store")
    }

    /// Helper: create a simple text record.
    fn make_record(text: &str) -> MemoryRecord {
        MemoryRecord::new_text(StoreType::Episodic, text)
    }

    // 1. CRUD roundtrip
    #[tokio::test]
    async fn crud_roundtrip() {
        let store = temp_store();

        // Store
        let record = make_record("Hello, episodic memory!");
        let id = store.store(record.clone()).await.unwrap();
        assert_eq!(id, record.id);

        // Get
        let retrieved = store.get(&id).await.unwrap().expect("record should exist");
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.text_content(), Some("Hello, episodic memory!"));

        // Delete
        let deleted = store.delete(&id).await.unwrap();
        assert!(deleted);

        // Verify gone
        let gone = store.get(&id).await.unwrap();
        assert!(gone.is_none());

        // Delete again returns false
        let deleted_again = store.delete(&id).await.unwrap();
        assert!(!deleted_again);
    }

    // 2. MessagePack serialize integrity
    #[tokio::test]
    async fn msgpack_serialize_integrity() {
        let store = temp_store();

        let mut record = make_record("Integrity test");
        record.fidelity.score = 0.75;
        record.fidelity.noise_level = 0.1;
        record.fidelity.decay_rate = 0.5;
        record.fidelity.emotional_anchor = 0.9;
        record.fidelity.reinforcement_count = 3;
        record.fidelity.stability = 2.0;
        record.emotion.joy = 0.8;
        record.emotion.trust = 0.5;
        record.emotion.fear = 0.1;
        record.emotion.surprise = 0.3;
        record.emotion.sadness = 0.0;
        record.emotion.disgust = 0.05;
        record.emotion.anger = 0.0;
        record.emotion.anticipation = 0.4;
        record.emotion.intensity = 0.7;
        record.emotion.valence = 0.6;
        record.metadata = serde_json::json!({"source": "test", "count": 42});
        record.access_count = 5;
        record.version = 2;

        let original_id = record.id;
        let original_created = record.created_at;
        let original_updated = record.updated_at;

        store.store(record).await.unwrap();

        let retrieved = store.get(&original_id).await.unwrap().unwrap();

        assert_eq!(retrieved.id, original_id);
        assert_eq!(retrieved.store, StoreType::Episodic);
        assert_eq!(retrieved.created_at, original_created);
        assert_eq!(retrieved.updated_at, original_updated);
        assert_eq!(retrieved.access_count, 5);
        assert_eq!(retrieved.version, 2);

        // Fidelity
        assert!((retrieved.fidelity.score - 0.75).abs() < f64::EPSILON);
        assert!((retrieved.fidelity.noise_level - 0.1).abs() < f64::EPSILON);
        assert!((retrieved.fidelity.decay_rate - 0.5).abs() < f64::EPSILON);
        assert!((retrieved.fidelity.emotional_anchor - 0.9).abs() < f64::EPSILON);
        assert_eq!(retrieved.fidelity.reinforcement_count, 3);
        assert!((retrieved.fidelity.stability - 2.0).abs() < f64::EPSILON);

        // Emotion
        assert!((retrieved.emotion.joy - 0.8).abs() < f64::EPSILON);
        assert!((retrieved.emotion.trust - 0.5).abs() < f64::EPSILON);
        assert!((retrieved.emotion.fear - 0.1).abs() < f64::EPSILON);
        assert!((retrieved.emotion.surprise - 0.3).abs() < f64::EPSILON);
        assert!((retrieved.emotion.sadness - 0.0).abs() < f64::EPSILON);
        assert!((retrieved.emotion.disgust - 0.05).abs() < f64::EPSILON);
        assert!((retrieved.emotion.anger - 0.0).abs() < f64::EPSILON);
        assert!((retrieved.emotion.anticipation - 0.4).abs() < f64::EPSILON);
        assert!((retrieved.emotion.intensity - 0.7).abs() < f64::EPSILON);
        assert!((retrieved.emotion.valence - 0.6).abs() < f64::EPSILON);

        // Content
        assert_eq!(retrieved.text_content(), Some("Integrity test"));

        // Metadata
        assert_eq!(retrieved.metadata["source"], "test");
        assert_eq!(retrieved.metadata["count"], 42);
    }

    // 3. Temporal range query
    #[tokio::test]
    async fn temporal_range_query() {
        let store = temp_store();

        let now = Utc::now();

        // Create records at different timestamps.
        let mut r1 = make_record("Event 1 - past");
        r1.created_at = now - Duration::hours(2);
        let mut r2 = make_record("Event 2 - recent");
        r2.created_at = now - Duration::minutes(30);
        let mut r3 = make_record("Event 3 - now");
        r3.created_at = now;
        let mut r4 = make_record("Event 4 - future");
        r4.created_at = now + Duration::hours(1);

        for r in [&r1, &r2, &r3, &r4] {
            store.store(r.clone()).await.unwrap();
        }

        // Query: last hour (should get r2 and r3)
        let results = store
            .query_temporal_range(now - Duration::hours(1), now + Duration::seconds(1))
            .await
            .unwrap();

        let result_ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
        assert!(result_ids.contains(&r2.id), "should contain r2");
        assert!(result_ids.contains(&r3.id), "should contain r3");
        assert!(
            !result_ids.contains(&r1.id),
            "should not contain r1 (too old)"
        );
        assert!(
            !result_ids.contains(&r4.id),
            "should not contain r4 (too new)"
        );
    }

    // 4. Records below fidelity
    #[tokio::test]
    async fn records_below_fidelity_threshold() {
        let store = temp_store();

        let mut r_high = make_record("High fidelity");
        r_high.fidelity.score = 0.9;
        let mut r_mid = make_record("Mid fidelity");
        r_mid.fidelity.score = 0.5;
        let mut r_low = make_record("Low fidelity");
        r_low.fidelity.score = 0.1;
        let mut r_zero = make_record("Zero fidelity");
        r_zero.fidelity.score = 0.0;

        for r in [&r_high, &r_mid, &r_low, &r_zero] {
            store.store(r.clone()).await.unwrap();
        }

        // Threshold 0.5: should get r_low and r_zero
        let results = store.records_below_fidelity(0.5).await.unwrap();
        let result_ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
        assert!(result_ids.contains(&r_low.id));
        assert!(result_ids.contains(&r_zero.id));
        assert!(!result_ids.contains(&r_mid.id));
        assert!(!result_ids.contains(&r_high.id));

        // Threshold 0.0: should get nothing
        let empty = store.records_below_fidelity(0.0).await.unwrap();
        assert!(empty.is_empty());

        // Threshold 1.0: should get r_high, r_mid, r_low, r_zero
        let all = store.records_below_fidelity(1.0).await.unwrap();
        assert_eq!(all.len(), 4);
    }

    // 5. Query text substring search
    #[tokio::test]
    async fn query_text_substring_search() {
        let store = temp_store();

        store
            .store(make_record("The quick brown fox jumps over the lazy dog"))
            .await
            .unwrap();
        store
            .store(make_record("A lazy cat sleeps all day"))
            .await
            .unwrap();
        store
            .store(make_record("Active running exercises"))
            .await
            .unwrap();

        // Search for "lazy" - should match 2 records
        let results = store.query_text("lazy", 10).await.unwrap();
        assert_eq!(results.len(), 2);

        // Case-insensitive
        let results = store.query_text("LAZY", 10).await.unwrap();
        assert_eq!(results.len(), 2);

        // Limit
        let results = store.query_text("lazy", 1).await.unwrap();
        assert_eq!(results.len(), 1);

        // No match
        let results = store.query_text("elephant", 10).await.unwrap();
        assert!(results.is_empty());
    }

    // 6. Persistence (open, write, close, reopen, read)
    #[tokio::test]
    async fn persistence_across_reopen() {
        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = tmp_dir.path().join("persist_test.redb");

        let record_id;
        {
            let store = EpisodicStore::open(&db_path).unwrap();
            let record = make_record("Persistent memory");
            record_id = store.store(record).await.unwrap();

            // Verify it's there
            let retrieved = store.get(&record_id).await.unwrap();
            assert!(retrieved.is_some());
        }
        // Store is dropped, database is closed.

        {
            let store = EpisodicStore::open(&db_path).unwrap();
            let retrieved = store.get(&record_id).await.unwrap().unwrap();
            assert_eq!(retrieved.text_content(), Some("Persistent memory"));
            assert_eq!(retrieved.id, record_id);
        }
    }

    // 7. Multiple records (store 20+, verify count)
    #[tokio::test]
    async fn multiple_records_count() {
        let store = temp_store();

        let n = 25;
        let mut ids = Vec::new();
        for i in 0..n {
            let record = make_record(&format!("Record number {i}"));
            let id = store.store(record).await.unwrap();
            ids.push(id);
        }

        // Count
        let count = store.count().await.unwrap();
        assert_eq!(count, n);

        // List IDs
        let listed = store.list_ids().await.unwrap();
        assert_eq!(listed.len(), n);
        for id in &ids {
            assert!(listed.contains(id), "listed IDs should contain {id}");
        }

        // Delete a few and re-check
        store.delete(&ids[0]).await.unwrap();
        store.delete(&ids[1]).await.unwrap();
        let count_after = store.count().await.unwrap();
        assert_eq!(count_after, n - 2);
    }

    // Additional: update_fidelity
    #[tokio::test]
    async fn update_fidelity_updates_record_and_index() {
        let store = temp_store();

        let mut record = make_record("Fidelity update test");
        record.fidelity.score = 0.9;
        let id = store.store(record).await.unwrap();

        // Update fidelity to 0.3
        let new_fidelity = FidelityState {
            score: 0.3,
            noise_level: 0.5,
            ..Default::default()
        };
        store.update_fidelity(&id, new_fidelity).await.unwrap();

        // Verify the record has updated fidelity
        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert!((retrieved.fidelity.score - 0.3).abs() < f64::EPSILON);
        assert!((retrieved.fidelity.noise_level - 0.5).abs() < f64::EPSILON);

        // Verify fidelity index works: record should now appear below 0.5
        let below = store.records_below_fidelity(0.5).await.unwrap();
        assert!(below.iter().any(|r| r.id == id));

        // And should NOT appear below 0.2
        let below_low = store.records_below_fidelity(0.2).await.unwrap();
        assert!(!below_low.iter().any(|r| r.id == id));
    }

    // Additional: update_record
    #[tokio::test]
    async fn update_record_fields() {
        let store = temp_store();

        let record = make_record("Original content");
        let id = store.store(record).await.unwrap();

        // Update content
        let new_content = cerememory_core::types::MemoryContent {
            blocks: vec![cerememory_core::types::ContentBlock {
                modality: cerememory_core::types::Modality::Text,
                format: "text/plain".to_string(),
                data: "Updated content".as_bytes().to_vec(),
                embedding: None,
            }],
            summary: Some("A summary".to_string()),
        };
        store
            .update_record(&id, Some(new_content), None, None, None)
            .await
            .unwrap();

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.text_content(), Some("Updated content"));
        assert_eq!(retrieved.content.summary.as_deref(), Some("A summary"));
        assert_eq!(retrieved.version, 2);
    }

    // Additional: update_fidelity on missing record returns error
    #[tokio::test]
    async fn update_fidelity_missing_record() {
        let store = temp_store();
        let fake_id = Uuid::now_v7();
        let result = store
            .update_fidelity(&fake_id, FidelityState::default())
            .await;
        assert!(result.is_err());
    }

    // Additional: update_record on missing record returns error
    #[tokio::test]
    async fn update_record_missing_record() {
        let store = temp_store();
        let fake_id = Uuid::now_v7();
        let result = store.update_record(&fake_id, None, None, None, None).await;
        assert!(result.is_err());
    }
}
