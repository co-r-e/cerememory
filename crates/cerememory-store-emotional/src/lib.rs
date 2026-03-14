//! Cerememory emotional store implementation backed by redb.
//!
//! This crate provides a persistent, file-based emotional memory store using
//! [redb](https://docs.rs/redb) as the storage engine. All records are serialized
//! via MessagePack (`rmp-serde`) and indexed by fidelity and emotion intensity
//! for efficient decay-based sweeps and emotion-filtered queries.
//!
//! Because redb is synchronous, every I/O operation is wrapped in
//! `tokio::task::spawn_blocking` so the async `Store` trait can be fulfilled
//! without blocking the Tokio runtime.

use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

use cerememory_core::error::CerememoryError;
use cerememory_core::traits::Store;
use cerememory_core::types::{EmotionVector, FidelityState, MemoryContent, MemoryRecord};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// Primary table: UUID (16 bytes) -> MessagePack-encoded `MemoryRecord`.
const EMOTIONAL_RECORDS: TableDefinition<&[u8], &[u8]> =
    TableDefinition::new("emotional_records");

/// Fidelity index: (fidelity_bucket_u8 ++ record_id) -> ().
/// Bucket = (fidelity.score * 100).round() clamped to [0, 100].
const EMOTIONAL_FIDELITY_INDEX: TableDefinition<&[u8], ()> =
    TableDefinition::new("emotional_fidelity_index");

/// Intensity index: (intensity_bucket_u8 ++ record_id) -> ().
/// Bucket = (emotion.intensity * 100).round() clamped to [0, 100].
const EMOTIONAL_INTENSITY_INDEX: TableDefinition<&[u8], ()> =
    TableDefinition::new("emotional_intensity_index");

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

/// Build the 17-byte fidelity-index key: 1 byte bucket + 16 bytes UUID.
fn fidelity_key(fidelity_score: f64, id: &Uuid) -> [u8; 17] {
    let mut buf = [0u8; 17];
    buf[0] = fidelity_bucket(fidelity_score);
    buf[1..].copy_from_slice(id.as_bytes());
    buf
}

/// Fidelity bucket byte from score.
fn fidelity_bucket(score: f64) -> u8 {
    (score * 100.0).round().clamp(0.0, 100.0) as u8
}

/// Build the 17-byte intensity-index key: 1 byte bucket + 16 bytes UUID.
fn intensity_key(intensity: f64, id: &Uuid) -> [u8; 17] {
    let mut buf = [0u8; 17];
    buf[0] = intensity_bucket(intensity);
    buf[1..].copy_from_slice(id.as_bytes());
    buf
}

/// Intensity bucket byte from intensity value.
fn intensity_bucket(intensity: f64) -> u8 {
    (intensity * 100.0).round().clamp(0.0, 100.0) as u8
}

// ---------------------------------------------------------------------------
// EmotionalStore
// ---------------------------------------------------------------------------

/// A redb-backed emotional memory store.
///
/// Thread-safe via `Arc<Database>`. All public methods are async and delegate
/// to `tokio::task::spawn_blocking` internally.
#[derive(Clone)]
pub struct EmotionalStore {
    db: Arc<Database>,
}

impl EmotionalStore {
    /// Create an ephemeral in-memory store. Alias for `open_in_memory()`.
    pub fn new() -> Self {
        Self::open_in_memory().expect("failed to create in-memory EmotionalStore")
    }

    /// Open (or create) a persistent store at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CerememoryError> {
        let db = Database::create(path.as_ref()).map_err(|e| {
            CerememoryError::Storage(format!("Failed to open redb database: {e}"))
        })?;

        let store = Self { db: Arc::new(db) };
        store.ensure_tables()?;
        Ok(store)
    }

    /// Create an ephemeral in-memory store backed by a temporary file.
    ///
    /// Useful for testing. The file is automatically cleaned up on drop via `tempfile`.
    pub fn open_in_memory() -> Result<Self, CerememoryError> {
        let tmp = tempfile::NamedTempFile::new().map_err(|e| {
            CerememoryError::Storage(format!("Failed to create temp file: {e}"))
        })?;
        // We intentionally persist the path -- redb manages the file.
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
            let _ = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;
            let _ = txn.open_table(EMOTIONAL_FIDELITY_INDEX).map_err(storage_err)?;
            let _ = txn.open_table(EMOTIONAL_INTENSITY_INDEX).map_err(storage_err)?;
        }
        txn.commit().map_err(storage_err)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Additional query methods (not part of the Store trait)
    // ------------------------------------------------------------------

    /// Return all records whose fidelity score is strictly below `threshold`.
    pub async fn records_below_fidelity(
        &self,
        threshold: f64,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let fidelity_table = txn.open_table(EMOTIONAL_FIDELITY_INDEX).map_err(storage_err)?;
            let records_table = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

            // Scan buckets [0, threshold_bucket).
            // The threshold bucket itself is *excluded* because we want < threshold.
            let threshold_bucket = fidelity_bucket(threshold);

            // Build range end key: threshold_bucket ++ all-zero UUID
            let end_key = {
                let mut buf = [0u8; 17];
                buf[0] = threshold_bucket;
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

    /// Return records whose `emotion.intensity` falls within `[min_intensity, max_intensity]`.
    ///
    /// Scans the intensity index from `intensity_bucket(min)` to `intensity_bucket(max)`
    /// inclusive, then filters by exact intensity values. Respects `limit`.
    pub async fn query_by_intensity_range(
        &self,
        min_intensity: f64,
        max_intensity: f64,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let intensity_table =
                txn.open_table(EMOTIONAL_INTENSITY_INDEX).map_err(storage_err)?;
            let records_table = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

            let start_bucket = intensity_bucket(min_intensity);
            // End bucket is inclusive, so we need bucket + 1 for the range end.
            let end_bucket = intensity_bucket(max_intensity);

            let start_key = {
                let mut buf = [0u8; 17];
                buf[0] = start_bucket;
                buf
            };

            // Build exclusive end key: (end_bucket + 1) ++ all-zero UUID,
            // or if end_bucket == 100, use an upper bound of [101, 0..0] which
            // is beyond any valid key.
            let end_key = {
                let mut buf = [0u8; 17];
                buf[0] = end_bucket.saturating_add(1);
                buf
            };

            let mut results = Vec::new();

            let range = intensity_table
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
                        // Double-check against the actual intensity value
                        if record.emotion.intensity >= min_intensity
                            && record.emotion.intensity <= max_intensity
                        {
                            results.push(record);
                            if results.len() >= limit {
                                break;
                            }
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

impl Default for EmotionalStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Store trait implementation
// ---------------------------------------------------------------------------

impl Store for EmotionalStore {
    async fn store(&self, record: MemoryRecord) -> Result<Uuid, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let id = record.id;
            let packed = rmp_serde::to_vec(&record)
                .map_err(|e| CerememoryError::Serialization(format!("msgpack encode: {e}")))?;

            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;

                let mut fidelity_idx =
                    txn.open_table(EMOTIONAL_FIDELITY_INDEX).map_err(storage_err)?;
                let fk = fidelity_key(record.fidelity.score, &id);
                fidelity_idx
                    .insert(fk.as_slice(), ())
                    .map_err(storage_err)?;

                let mut intensity_idx =
                    txn.open_table(EMOTIONAL_INTENSITY_INDEX).map_err(storage_err)?;
                let ik = intensity_key(record.emotion.intensity, &id);
                intensity_idx
                    .insert(ik.as_slice(), ())
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
            let table = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;
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
                let mut records = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

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

                    let mut fidelity_idx =
                        txn.open_table(EMOTIONAL_FIDELITY_INDEX).map_err(storage_err)?;
                    let fk = fidelity_key(record.fidelity.score, &id);
                    fidelity_idx.remove(fk.as_slice()).map_err(storage_err)?;

                    let mut intensity_idx =
                        txn.open_table(EMOTIONAL_INTENSITY_INDEX).map_err(storage_err)?;
                    let ik = intensity_key(record.emotion.intensity, &id);
                    intensity_idx.remove(ik.as_slice()).map_err(storage_err)?;

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
                let mut records = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

                // Read within the write transaction to avoid TOCTOU
                let guard = records
                    .get(id.as_bytes().as_slice())
                    .map_err(storage_err)?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut record: MemoryRecord = rmp_serde::from_slice(guard.value())
                    .map_err(|e| {
                        CerememoryError::Serialization(format!("msgpack decode: {e}"))
                    })?;
                drop(guard);

                let old_fidelity_score = record.fidelity.score;
                record.fidelity = fidelity;
                record.updated_at = Utc::now();

                let packed = rmp_serde::to_vec(&record)
                    .map_err(|e| {
                        CerememoryError::Serialization(format!("msgpack encode: {e}"))
                    })?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;

                // Update fidelity index: remove old, insert new.
                let mut fidelity_idx =
                    txn.open_table(EMOTIONAL_FIDELITY_INDEX).map_err(storage_err)?;
                let old_fk = fidelity_key(old_fidelity_score, &id);
                fidelity_idx.remove(old_fk.as_slice()).map_err(storage_err)?;
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
            let table = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

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
            let table = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

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

    async fn count(&self) -> Result<usize, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;
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
    ) -> Result<(), CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

                // Read within the write transaction to avoid TOCTOU
                let guard = records
                    .get(id.as_bytes().as_slice())
                    .map_err(storage_err)?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut record: MemoryRecord = rmp_serde::from_slice(guard.value())
                    .map_err(|e| {
                        CerememoryError::Serialization(format!("msgpack decode: {e}"))
                    })?;
                drop(guard);

                let old_intensity = record.emotion.intensity;

                if let Some(c) = content {
                    record.content = c;
                }
                if let Some(e) = &emotion {
                    record.emotion = e.clone();
                }
                if let Some(m) = metadata {
                    record.metadata = m;
                }
                record.updated_at = Utc::now();
                record.version += 1;

                let packed = rmp_serde::to_vec(&record)
                    .map_err(|e| {
                        CerememoryError::Serialization(format!("msgpack encode: {e}"))
                    })?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;

                // If emotion was updated, update intensity index.
                if emotion.is_some() {
                    let mut intensity_idx =
                        txn.open_table(EMOTIONAL_INTENSITY_INDEX).map_err(storage_err)?;
                    let old_ik = intensity_key(old_intensity, &id);
                    intensity_idx.remove(old_ik.as_slice()).map_err(storage_err)?;
                    let new_ik = intensity_key(record.emotion.intensity, &id);
                    intensity_idx
                        .insert(new_ik.as_slice(), ())
                        .map_err(storage_err)?;
                }
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
                let mut records = txn.open_table(EMOTIONAL_RECORDS).map_err(storage_err)?;

                let guard = records
                    .get(id.as_bytes().as_slice())
                    .map_err(storage_err)?
                    .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
                let mut record: MemoryRecord = rmp_serde::from_slice(guard.value())
                    .map_err(|e| {
                        CerememoryError::Serialization(format!("msgpack decode: {e}"))
                    })?;
                drop(guard);

                record.access_count = access_count;
                record.last_accessed_at = last_accessed_at;
                record.updated_at = Utc::now();

                let packed = rmp_serde::to_vec(&record)
                    .map_err(|e| {
                        CerememoryError::Serialization(format!("msgpack encode: {e}"))
                    })?;
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
// Internal helpers
// ---------------------------------------------------------------------------

/// Read a single `MemoryRecord` from the records table (synchronous).
fn get_record_sync(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    id: &Uuid,
) -> Result<Option<MemoryRecord>, CerememoryError> {
    match table.get(id.as_bytes().as_slice()).map_err(storage_err)? {
        Some(value_guard) => {
            let record: MemoryRecord = rmp_serde::from_slice(value_guard.value())
                .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;
            Ok(Some(record))
        }
        None => Ok(None),
    }
}

/// Check if a record matches a text query (case-insensitive substring match).
fn record_matches_text(record: &MemoryRecord, query_lower: &str) -> bool {
    // Check text content blocks
    for block in &record.content.blocks {
        if block.modality == cerememory_core::types::Modality::Text {
            if let Ok(text) = std::str::from_utf8(&block.data) {
                if text.to_lowercase().contains(query_lower) {
                    return true;
                }
            }
        }
    }
    // Check summary
    if let Some(ref summary) = record.content.summary {
        if summary.to_lowercase().contains(query_lower) {
            return true;
        }
    }
    false
}

/// Convert any `Display` error into `CerememoryError::Storage`.
fn storage_err(e: impl std::fmt::Display) -> CerememoryError {
    CerememoryError::Storage(e.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::types::{MemoryRecord, StoreType};

    /// Helper: create a store in a fresh temp directory.
    fn temp_store() -> EmotionalStore {
        EmotionalStore::open_in_memory().expect("failed to create in-memory store")
    }

    /// Helper: create a simple text record.
    fn make_record(text: &str) -> MemoryRecord {
        MemoryRecord::new_text(StoreType::Emotional, text)
    }

    // 1. CRUD roundtrip
    #[tokio::test]
    async fn crud_roundtrip() {
        let store = temp_store();

        // Store
        let record = make_record("Hello, emotional memory!");
        let id = store.store(record.clone()).await.unwrap();
        assert_eq!(id, record.id);

        // Get
        let retrieved = store.get(&id).await.unwrap().expect("record should exist");
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.text_content(), Some("Hello, emotional memory!"));

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
    async fn msgpack_integrity() {
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
        assert_eq!(retrieved.store, StoreType::Emotional);
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

    // 3. Persistence across reopen
    #[tokio::test]
    async fn persistence_across_reopen() {
        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = tmp_dir.path().join("persist_test.redb");

        let record_id;
        {
            let store = EmotionalStore::open(&db_path).unwrap();
            let record = make_record("Persistent memory");
            record_id = store.store(record).await.unwrap();

            // Verify it's there
            let retrieved = store.get(&record_id).await.unwrap();
            assert!(retrieved.is_some());
        }
        // Store is dropped, database is closed.

        {
            let store = EmotionalStore::open(&db_path).unwrap();
            let retrieved = store.get(&record_id).await.unwrap().unwrap();
            assert_eq!(retrieved.text_content(), Some("Persistent memory"));
            assert_eq!(retrieved.id, record_id);
        }
    }

    // 4. Query text substring search
    #[tokio::test]
    async fn query_text() {
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

    // 5. Update fidelity
    #[tokio::test]
    async fn update_fidelity() {
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

    // 6. Update record
    #[tokio::test]
    async fn update_record() {
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
            .update_record(&id, Some(new_content), None, None)
            .await
            .unwrap();

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.text_content(), Some("Updated content"));
        assert_eq!(retrieved.content.summary.as_deref(), Some("A summary"));
        assert_eq!(retrieved.version, 2);
    }

    // 7. Update access
    #[tokio::test]
    async fn update_access() {
        let store = temp_store();

        let record = make_record("Access test");
        let id = store.store(record).await.unwrap();

        let now = Utc::now();
        store.update_access(&id, 42, now).await.unwrap();

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.access_count, 42);
        assert_eq!(retrieved.last_accessed_at, now);
    }

    // 8. Delete nonexistent
    #[tokio::test]
    async fn delete_nonexistent() {
        let store = temp_store();
        let fake_id = Uuid::now_v7();
        let result = store.delete(&fake_id).await.unwrap();
        assert!(!result);
    }

    // 9. Count
    #[tokio::test]
    async fn count() {
        let store = temp_store();

        assert_eq!(store.count().await.unwrap(), 0);

        let n = 25;
        let mut ids = Vec::new();
        for i in 0..n {
            let record = make_record(&format!("Record number {i}"));
            let id = store.store(record).await.unwrap();
            ids.push(id);
        }

        assert_eq!(store.count().await.unwrap(), n);

        // Delete a few and re-check
        store.delete(&ids[0]).await.unwrap();
        store.delete(&ids[1]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), n - 2);
    }

    // 10. List IDs
    #[tokio::test]
    async fn list_ids() {
        let store = temp_store();

        let n = 10;
        let mut ids = Vec::new();
        for i in 0..n {
            let record = make_record(&format!("Record {i}"));
            let id = store.store(record).await.unwrap();
            ids.push(id);
        }

        let listed = store.list_ids().await.unwrap();
        assert_eq!(listed.len(), n);
        for id in &ids {
            assert!(listed.contains(id), "listed IDs should contain {id}");
        }
    }

    // 11. Concurrent reads
    #[tokio::test]
    async fn concurrent_reads() {
        let store = temp_store();

        let record = make_record("Concurrent test");
        let id = store.store(record).await.unwrap();

        let mut handles = Vec::new();
        for _ in 0..10 {
            let s = store.clone();
            let handle = tokio::spawn(async move { s.get(&id).await.unwrap().unwrap() });
            handles.push(handle);
        }

        for handle in handles {
            let retrieved = handle.await.unwrap();
            assert_eq!(retrieved.id, id);
            assert_eq!(retrieved.text_content(), Some("Concurrent test"));
        }
    }

    // 12. Fidelity index query
    #[tokio::test]
    async fn fidelity_index_query() {
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

        // Threshold 1.0: should get all 4
        let all = store.records_below_fidelity(1.0).await.unwrap();
        assert_eq!(all.len(), 4);
    }

    // 13. Limit zero
    #[tokio::test]
    async fn limit_zero() {
        let store = temp_store();

        store.store(make_record("Something")).await.unwrap();

        let results = store.query_text("Something", 0).await.unwrap();
        assert!(results.is_empty());

        let results = store
            .query_by_intensity_range(0.0, 1.0, 0)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    // 14. Missing record error
    #[tokio::test]
    async fn missing_record_error() {
        let store = temp_store();
        let fake_id = Uuid::now_v7();

        let result = store
            .update_fidelity(&fake_id, FidelityState::default())
            .await;
        assert!(result.is_err());

        let result = store
            .update_record(&fake_id, None, None, None)
            .await;
        assert!(result.is_err());
    }

    // 15. Intensity range query
    #[tokio::test]
    async fn intensity_range_query() {
        let store = temp_store();

        let mut r_low = make_record("Low intensity");
        r_low.emotion.intensity = 0.1;
        let mut r_mid = make_record("Mid intensity");
        r_mid.emotion.intensity = 0.5;
        let mut r_high = make_record("High intensity");
        r_high.emotion.intensity = 0.9;
        let mut r_max = make_record("Max intensity");
        r_max.emotion.intensity = 1.0;

        for r in [&r_low, &r_mid, &r_high, &r_max] {
            store.store(r.clone()).await.unwrap();
        }

        // Query [0.4, 0.6] -> should get r_mid only
        let results = store
            .query_by_intensity_range(0.4, 0.6, 100)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, r_mid.id);

        // Query [0.0, 0.2] -> should get r_low only
        let results = store
            .query_by_intensity_range(0.0, 0.2, 100)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, r_low.id);

        // Query [0.8, 1.0] -> should get r_high and r_max
        let results = store
            .query_by_intensity_range(0.8, 1.0, 100)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        let result_ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
        assert!(result_ids.contains(&r_high.id));
        assert!(result_ids.contains(&r_max.id));

        // Query full range [0.0, 1.0] -> all 4
        let results = store
            .query_by_intensity_range(0.0, 1.0, 100)
            .await
            .unwrap();
        assert_eq!(results.len(), 4);

        // Query with limit
        let results = store
            .query_by_intensity_range(0.0, 1.0, 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    // 16. Intensity index updated on update_record
    #[tokio::test]
    async fn intensity_index_updated_on_update() {
        let store = temp_store();

        let mut record = make_record("Intensity update test");
        record.emotion.intensity = 0.2;
        let id = store.store(record).await.unwrap();

        // Should appear in [0.1, 0.3]
        let results = store
            .query_by_intensity_range(0.1, 0.3, 100)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);

        // Should NOT appear in [0.7, 0.9]
        let results = store
            .query_by_intensity_range(0.7, 0.9, 100)
            .await
            .unwrap();
        assert!(results.is_empty());

        // Update emotion to high intensity
        let new_emotion = EmotionVector {
            intensity: 0.8,
            ..Default::default()
        };
        store
            .update_record(&id, None, Some(new_emotion), None)
            .await
            .unwrap();

        // Should now appear in [0.7, 0.9]
        let results = store
            .query_by_intensity_range(0.7, 0.9, 100)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);

        // Should NOT appear in old range [0.1, 0.3]
        let results = store
            .query_by_intensity_range(0.1, 0.3, 100)
            .await
            .unwrap();
        assert!(results.is_empty());
    }
}
