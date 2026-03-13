//! Cerememory emotional store implementation.
//!
//! Phase 1 stub: no-op store. Emotion vectors are stored inline in MemoryRecord.
//! The emotional store in Phase 2+ will provide cross-cutting emotion indices
//! and emotion-filtered queries across all stores.

use cerememory_core::error::CerememoryError;
use cerememory_core::traits::Store;
use cerememory_core::types::*;
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

/// In-memory emotional store stub.
///
/// In Phase 1, emotion data lives inline in MemoryRecord.emotion.
/// This store holds records explicitly tagged as StoreType::Emotional.
pub struct EmotionalStore {
    records: RwLock<HashMap<Uuid, MemoryRecord>>,
}

impl EmotionalStore {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for EmotionalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for EmotionalStore {
    async fn store(&self, record: MemoryRecord) -> Result<Uuid, CerememoryError> {
        let id = record.id;
        self.records.write().await.insert(id, record);
        Ok(id)
    }

    async fn get(&self, id: &Uuid) -> Result<Option<MemoryRecord>, CerememoryError> {
        Ok(self.records.read().await.get(id).cloned())
    }

    async fn delete(&self, id: &Uuid) -> Result<bool, CerememoryError> {
        Ok(self.records.write().await.remove(id).is_some())
    }

    async fn update_fidelity(
        &self,
        id: &Uuid,
        fidelity: FidelityState,
    ) -> Result<(), CerememoryError> {
        let mut records = self.records.write().await;
        match records.get_mut(id) {
            Some(r) => {
                r.fidelity = fidelity;
                Ok(())
            }
            None => Err(CerememoryError::RecordNotFound(id.to_string())),
        }
    }

    async fn query_text(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, CerememoryError> {
        let records = self.records.read().await;
        let query_lower = query.to_lowercase();
        Ok(records
            .values()
            .filter(|r| r.matches_text(&query_lower))
            .take(limit)
            .cloned()
            .collect())
    }

    async fn list_ids(&self) -> Result<Vec<Uuid>, CerememoryError> {
        Ok(self.records.read().await.keys().copied().collect())
    }

    async fn count(&self) -> Result<usize, CerememoryError> {
        Ok(self.records.read().await.len())
    }

    async fn update_record(
        &self,
        id: &Uuid,
        content: Option<MemoryContent>,
        emotion: Option<EmotionVector>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), CerememoryError> {
        let mut records = self.records.write().await;
        match records.get_mut(id) {
            Some(r) => {
                r.apply_updates(content, emotion, metadata);
                Ok(())
            }
            None => Err(CerememoryError::RecordNotFound(id.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn basic_crud() {
        let store = EmotionalStore::new();
        let record = MemoryRecord::new_text(StoreType::Emotional, "emotional memory");
        let id = store.store(record).await.unwrap();
        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.text_content(), Some("emotional memory"));
        assert!(store.delete(&id).await.unwrap());
    }
}
