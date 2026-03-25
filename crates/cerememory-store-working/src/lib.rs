//! In-memory working memory store for Cerememory.
//!
//! Implements a bounded-capacity store with LRU eviction, modelling the
//! "7 +/- 2" capacity limit of human working memory (default capacity: 7).
//! Concurrent access is managed via `tokio::sync::RwLock`.

use std::collections::HashMap;

use cerememory_core::error::CerememoryError;
use cerememory_core::traits::Store;
use cerememory_core::types::*;
use chrono::Utc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Default working memory capacity, inspired by Miller's Law (7 +/- 2).
pub const DEFAULT_CAPACITY: usize = 7;

/// An in-memory working memory store with LRU eviction.
///
/// When the store reaches capacity and a new record is inserted, the record
/// with the oldest `last_accessed_at` timestamp is evicted to make room.
pub struct WorkingMemoryStore {
    capacity: usize,
    inner: RwLock<StoreInner>,
}

struct StoreInner {
    records: HashMap<Uuid, MemoryRecord>,
}

impl WorkingMemoryStore {
    /// Create a new `WorkingMemoryStore` with the default capacity (7).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new `WorkingMemoryStore` with a specified capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "WorkingMemoryStore capacity must be > 0");
        Self {
            capacity,
            inner: RwLock::new(StoreInner {
                records: HashMap::with_capacity(capacity),
            }),
        }
    }
}

impl Default for WorkingMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkingMemoryStore {
    /// Store a record and return the evicted record's ID, if any.
    /// Use this instead of `Store::store` when you need to track evictions.
    pub async fn store_with_eviction(
        &self,
        record: MemoryRecord,
    ) -> Result<(Uuid, Option<Uuid>), CerememoryError> {
        let id = record.id;
        let mut inner = self.inner.write().await;

        let evicted = if inner.records.len() >= self.capacity && !inner.records.contains_key(&id) {
            inner
                .records
                .iter()
                .min_by_key(|(_, r)| r.last_accessed_at)
                .map(|(id, _)| *id)
                .inspect(|evict_id| {
                    inner.records.remove(evict_id);
                })
        } else {
            None
        };

        inner.records.insert(id, record);
        Ok((id, evicted))
    }
}

impl Store for WorkingMemoryStore {
    /// Store a new memory record.
    ///
    /// If the store is at capacity, the least-recently-accessed record is
    /// evicted before inserting the new one.
    async fn store(&self, record: MemoryRecord) -> Result<Uuid, CerememoryError> {
        let id = record.id;
        let mut inner = self.inner.write().await;

        // Evict the LRU record if at capacity (and the new id isn't already present).
        if inner.records.len() >= self.capacity && !inner.records.contains_key(&id) {
            let lru_id = inner
                .records
                .iter()
                .min_by_key(|(_, r)| r.last_accessed_at)
                .map(|(id, _)| *id);

            if let Some(evict_id) = lru_id {
                inner.records.remove(&evict_id);
            }
        }

        inner.records.insert(id, record);
        Ok(id)
    }

    /// Retrieve a memory record by ID.
    ///
    /// Updates `last_accessed_at` and increments `access_count` on each
    /// successful retrieval (LRU touch).
    async fn get(&self, id: &Uuid) -> Result<Option<MemoryRecord>, CerememoryError> {
        let mut inner = self.inner.write().await;

        let Some(record) = inner.records.get_mut(id) else {
            return Ok(None);
        };

        // LRU touch: update access metadata.
        record.last_accessed_at = Utc::now();
        record.access_count = record.access_count.saturating_add(1);

        Ok(Some(record.clone()))
    }

    /// Delete a memory record by ID.
    ///
    /// Returns `true` if the record existed and was removed.
    async fn delete(&self, id: &Uuid) -> Result<bool, CerememoryError> {
        let mut inner = self.inner.write().await;
        Ok(inner.records.remove(id).is_some())
    }

    /// Update the fidelity state for a record.
    async fn update_fidelity(
        &self,
        id: &Uuid,
        fidelity: FidelityState,
    ) -> Result<(), CerememoryError> {
        let mut inner = self.inner.write().await;

        let record = inner
            .records
            .get_mut(id)
            .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;

        record.fidelity = fidelity;
        record.updated_at = Utc::now();
        Ok(())
    }

    /// Text-based substring search across record content.
    ///
    /// Returns up to `limit` records whose text content contains `query`
    /// (case-insensitive). Results are ordered by most-recently-accessed first.
    async fn query_text(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let inner = self.inner.read().await;
        let query_lower = query.to_lowercase();

        let mut matches: Vec<&MemoryRecord> = inner
            .records
            .values()
            .filter(|r| r.matches_text(&query_lower))
            .collect();

        // Most recently accessed first.
        matches.sort_by(|a, b| b.last_accessed_at.cmp(&a.last_accessed_at));
        matches.truncate(limit);

        Ok(matches.into_iter().cloned().collect())
    }

    /// List all record IDs in this store.
    async fn list_ids(&self) -> Result<Vec<Uuid>, CerememoryError> {
        let inner = self.inner.read().await;
        Ok(inner.records.keys().copied().collect())
    }

    async fn get_all(&self) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let inner = self.inner.read().await;
        Ok(inner.records.values().cloned().collect())
    }

    async fn count(&self) -> Result<usize, CerememoryError> {
        let inner = self.inner.read().await;
        Ok(inner.records.len())
    }

    /// Update an existing record's content, emotion, and/or metadata.
    async fn update_record(
        &self,
        id: &Uuid,
        content: Option<MemoryContent>,
        emotion: Option<EmotionVector>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), CerememoryError> {
        let mut inner = self.inner.write().await;

        let record = inner
            .records
            .get_mut(id)
            .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;

        record.apply_updates(content, emotion, metadata);
        record.version = record.version.saturating_add(1);
        Ok(())
    }

    async fn replace_associations(
        &self,
        id: &Uuid,
        associations: Vec<Association>,
    ) -> Result<(), CerememoryError> {
        let mut inner = self.inner.write().await;
        let record = inner
            .records
            .get_mut(id)
            .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
        record.associations = associations;
        record.updated_at = Utc::now();
        record.version = record.version.saturating_add(1);
        Ok(())
    }

    async fn update_access(
        &self,
        id: &Uuid,
        access_count: u32,
        last_accessed_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), CerememoryError> {
        let mut inner = self.inner.write().await;
        let record = inner
            .records
            .get_mut(id)
            .ok_or_else(|| CerememoryError::RecordNotFound(id.to_string()))?;
        record.access_count = access_count;
        record.last_accessed_at = last_accessed_at;
        record.updated_at = Utc::now();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::types::{MemoryRecord, StoreType};
    use std::sync::Arc;

    /// Helper: create a text record for the Working store.
    fn make_record(text: &str) -> MemoryRecord {
        MemoryRecord::new_text(StoreType::Working, text)
    }

    // -----------------------------------------------------------------------
    // 1. Basic CRUD
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn store_and_get() {
        let store = WorkingMemoryStore::new();
        let record = make_record("hello world");
        let id = store.store(record).await.unwrap();

        let retrieved = store.get(&id).await.unwrap().expect("record should exist");
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.text_content(), Some("hello world"));
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let store = WorkingMemoryStore::new();
        let missing = Uuid::now_v7();
        assert!(store.get(&missing).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_existing_record() {
        let store = WorkingMemoryStore::new();
        let id = store.store(make_record("to delete")).await.unwrap();

        assert!(store.delete(&id).await.unwrap());
        assert!(store.get(&id).await.unwrap().is_none());
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn delete_nonexistent_returns_false() {
        let store = WorkingMemoryStore::new();
        assert!(!store.delete(&Uuid::now_v7()).await.unwrap());
    }

    #[tokio::test]
    async fn count_and_list_ids() {
        let store = WorkingMemoryStore::new();
        let id1 = store.store(make_record("a")).await.unwrap();
        let id2 = store.store(make_record("b")).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 2);

        let ids = store.list_ids().await.unwrap();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    // -----------------------------------------------------------------------
    // 2. LRU Eviction
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn lru_eviction_at_capacity() {
        let store = WorkingMemoryStore::with_capacity(3);

        // Store 3 records. Record "first" has the oldest last_accessed_at.
        let id_first = store.store(make_record("first")).await.unwrap();
        let id_second = store.store(make_record("second")).await.unwrap();
        let id_third = store.store(make_record("third")).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 3);

        // Access "first" to move it ahead in the LRU ordering.
        let _ = store.get(&id_first).await.unwrap();

        // Now "second" has the oldest last_accessed_at. Storing a 4th record
        // should evict "second".
        let id_fourth = store.store(make_record("fourth")).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 3);
        assert!(store.get(&id_second).await.unwrap().is_none());
        // The others should still exist (get touches them, but that's fine).
        assert!(store.get(&id_first).await.unwrap().is_some());
        assert!(store.get(&id_third).await.unwrap().is_some());
        assert!(store.get(&id_fourth).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn store_same_id_twice_does_not_evict() {
        let store = WorkingMemoryStore::with_capacity(2);

        let mut record = make_record("original");
        let id = record.id;
        store.store(record.clone()).await.unwrap();

        let id2 = store.store(make_record("other")).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        // Re-store the same record (updated content) — should NOT evict.
        record.content = MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: b"updated".to_vec(),
                embedding: None,
            }],
            summary: None,
        };
        store.store(record).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 2);
        let r = store.get(&id).await.unwrap().unwrap();
        assert_eq!(r.text_content(), Some("updated"));
        assert!(store.get(&id2).await.unwrap().is_some());
    }

    // -----------------------------------------------------------------------
    // 3. query_text substring search
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn query_text_finds_matching_records() {
        let store = WorkingMemoryStore::new();
        store
            .store(make_record("The quick brown fox"))
            .await
            .unwrap();
        store.store(make_record("Lazy dog sleeps")).await.unwrap();
        store
            .store(make_record("QUICK uppercase match"))
            .await
            .unwrap();

        let results = store.query_text("quick", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn query_text_respects_limit() {
        let store = WorkingMemoryStore::new();
        for i in 0..5 {
            store
                .store(make_record(&format!("item {i}")))
                .await
                .unwrap();
        }

        let results = store.query_text("item", 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn query_text_no_match_returns_empty() {
        let store = WorkingMemoryStore::new();
        store.store(make_record("hello")).await.unwrap();

        let results = store.query_text("zzzzz", 10).await.unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // 4. Concurrent access
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn concurrent_store_and_get() {
        let store = Arc::new(WorkingMemoryStore::with_capacity(100));
        let mut handles = Vec::new();

        for i in 0..20 {
            let s = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                let record = make_record(&format!("concurrent-{i}"));
                let id = s.store(record).await.unwrap();
                let retrieved = s.get(&id).await.unwrap();
                assert!(retrieved.is_some());
                id
            }));
        }

        let mut ids = Vec::new();
        for h in handles {
            ids.push(h.await.unwrap());
        }

        assert_eq!(store.count().await.unwrap(), 20);

        // All IDs should be listable.
        let listed = store.list_ids().await.unwrap();
        for id in &ids {
            assert!(listed.contains(id));
        }
    }

    #[tokio::test]
    async fn concurrent_writes_and_reads_interleaved() {
        let store = Arc::new(WorkingMemoryStore::with_capacity(50));

        // Writers
        let mut handles = Vec::new();
        for i in 0..10 {
            let s = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                s.store(make_record(&format!("writer-{i}"))).await.unwrap()
            }));
        }

        let ids: Vec<Uuid> = {
            let mut v = Vec::new();
            for h in handles {
                v.push(h.await.unwrap());
            }
            v
        };

        // Readers + deleters in parallel
        let mut handles = Vec::new();
        for &id in &ids {
            let s = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                // Read
                let _ = s.get(&id).await.unwrap();
                // Query
                let _ = s.query_text("writer", 5).await.unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // 5. update_fidelity and update_record
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn update_fidelity_changes_score() {
        let store = WorkingMemoryStore::new();
        let id = store.store(make_record("memory")).await.unwrap();

        let new_fidelity = FidelityState {
            score: 0.5,
            noise_level: 0.2,
            ..FidelityState::default()
        };
        store.update_fidelity(&id, new_fidelity).await.unwrap();

        let r = store.get(&id).await.unwrap().unwrap();
        assert!((r.fidelity.score - 0.5).abs() < f64::EPSILON);
        assert!((r.fidelity.noise_level - 0.2).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn update_fidelity_nonexistent_record_errors() {
        let store = WorkingMemoryStore::new();
        let result = store
            .update_fidelity(&Uuid::now_v7(), FidelityState::default())
            .await;
        assert!(matches!(result, Err(CerememoryError::RecordNotFound(_))));
    }

    #[tokio::test]
    async fn update_record_partial_update() {
        let store = WorkingMemoryStore::new();
        let id = store.store(make_record("original text")).await.unwrap();

        // Update only metadata.
        let meta = serde_json::json!({"tag": "important"});
        store
            .update_record(&id, None, None, Some(meta.clone()))
            .await
            .unwrap();

        let r = store.get(&id).await.unwrap().unwrap();
        assert_eq!(r.text_content(), Some("original text"));
        assert_eq!(r.metadata, meta);
        assert_eq!(r.version, 2);
    }

    #[tokio::test]
    async fn update_record_full_update() {
        let store = WorkingMemoryStore::new();
        let id = store.store(make_record("old")).await.unwrap();

        let new_content = MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: b"new content".to_vec(),
                embedding: None,
            }],
            summary: Some("updated summary".to_string()),
        };
        let new_emotion = EmotionVector {
            joy: 0.8,
            ..EmotionVector::default()
        };
        let new_meta = serde_json::json!({"updated": true});

        store
            .update_record(
                &id,
                Some(new_content),
                Some(new_emotion),
                Some(new_meta.clone()),
            )
            .await
            .unwrap();

        let r = store.get(&id).await.unwrap().unwrap();
        assert_eq!(r.text_content(), Some("new content"));
        assert_eq!(r.content.summary, Some("updated summary".to_string()));
        assert!((r.emotion.joy - 0.8).abs() < f64::EPSILON);
        assert_eq!(r.metadata, new_meta);
        assert_eq!(r.version, 2);
    }

    #[tokio::test]
    async fn update_record_nonexistent_errors() {
        let store = WorkingMemoryStore::new();
        let result = store.update_record(&Uuid::now_v7(), None, None, None).await;
        assert!(matches!(result, Err(CerememoryError::RecordNotFound(_))));
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn get_updates_access_count() {
        let store = WorkingMemoryStore::new();
        let id = store.store(make_record("counter")).await.unwrap();

        // Initial access_count is 0 on the record.
        // Each get increments it.
        let r1 = store.get(&id).await.unwrap().unwrap();
        assert_eq!(r1.access_count, 1);

        let r2 = store.get(&id).await.unwrap().unwrap();
        assert_eq!(r2.access_count, 2);

        let r3 = store.get(&id).await.unwrap().unwrap();
        assert_eq!(r3.access_count, 3);
    }

    #[tokio::test]
    #[should_panic(expected = "capacity must be > 0")]
    async fn zero_capacity_panics() {
        let _store = WorkingMemoryStore::with_capacity(0);
    }

    #[tokio::test]
    async fn capacity_one_evicts_immediately() {
        let store = WorkingMemoryStore::with_capacity(1);

        let id1 = store.store(make_record("first")).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        let id2 = store.store(make_record("second")).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        assert!(store.get(&id1).await.unwrap().is_none());
        assert!(store.get(&id2).await.unwrap().is_some());
    }
}
