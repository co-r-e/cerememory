//! Hippocampal Coordinator — cross-store record registry and global association graph.
//!
//! Maps record_id → StoreType and maintains a global association index
//! that spans all stores. Phase 1 keeps everything in-memory, rebuilt
//! from stores on startup.
//!
//! Phase 2 adds:
//! - [`text_index::TextIndex`] — Tantivy full-text search across all stores
//! - [`vector_index::VectorIndex`] — Brute-force cosine similarity search

pub mod structured_index;
pub mod text_index;
pub mod vector_index;

use cerememory_core::error::CerememoryError;
use cerememory_core::traits::AssociationGraph;
use cerememory_core::types::*;
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Entry in the coordinator's registry.
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub store_type: StoreType,
    pub associations: Vec<Association>,
}

/// Hippocampal Coordinator: knows where every record lives
/// and maintains the global association graph.
pub struct HippocampalCoordinator {
    registry: RwLock<HashMap<Uuid, RegistryEntry>>,
}

impl HippocampalCoordinator {
    pub fn new() -> Self {
        Self {
            registry: RwLock::new(HashMap::new()),
        }
    }

    /// Register a record in the coordinator.
    pub async fn register(
        &self,
        record_id: Uuid,
        store_type: StoreType,
        associations: Vec<Association>,
    ) {
        self.registry.write().await.insert(
            record_id,
            RegistryEntry {
                store_type,
                associations,
            },
        );
    }

    /// Remove a record from the coordinator and clean up inbound associations.
    pub async fn unregister(&self, record_id: &Uuid) -> bool {
        let mut reg = self.registry.write().await;
        let existed = reg.remove(record_id).is_some();
        // Remove inbound associations from all other records pointing to this one
        if existed {
            for entry in reg.values_mut() {
                entry.associations.retain(|a| a.target_id != *record_id);
            }
        }
        existed
    }

    /// Update associations for a record.
    pub async fn update_associations(
        &self,
        record_id: &Uuid,
        associations: Vec<Association>,
    ) -> Result<(), CerememoryError> {
        let mut reg = self.registry.write().await;
        match reg.get_mut(record_id) {
            Some(entry) => {
                entry.associations = associations;
                Ok(())
            }
            None => Err(CerememoryError::RecordNotFound(record_id.to_string())),
        }
    }

    /// Add a single association to a record.
    pub async fn add_association(
        &self,
        record_id: &Uuid,
        association: Association,
    ) -> Result<(), CerememoryError> {
        let mut reg = self.registry.write().await;
        match reg.get_mut(record_id) {
            Some(entry) => {
                // Prevent duplicate associations to the same target with same type
                let exists = entry.associations.iter().any(|a| {
                    a.target_id == association.target_id
                        && a.association_type == association.association_type
                });
                if !exists {
                    entry.associations.push(association);
                }
                Ok(())
            }
            None => Err(CerememoryError::RecordNotFound(record_id.to_string())),
        }
    }

    /// Get all record IDs in a given store.
    pub async fn records_in_store(&self, store_type: StoreType) -> Vec<Uuid> {
        self.registry
            .read()
            .await
            .iter()
            .filter(|(_, e)| e.store_type == store_type)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Total number of registered records.
    pub async fn total_records(&self) -> usize {
        self.registry.read().await.len()
    }

    /// Count records per store type.
    pub async fn records_by_store(&self) -> HashMap<StoreType, u32> {
        let reg = self.registry.read().await;
        let mut counts = HashMap::new();
        for entry in reg.values() {
            *counts.entry(entry.store_type).or_insert(0) += 1;
        }
        counts
    }

    /// Total number of associations across all records.
    pub async fn total_associations(&self) -> u32 {
        self.registry
            .read()
            .await
            .values()
            .map(|e| e.associations.len() as u32)
            .sum()
    }

    /// Rebuild the coordinator from a list of records.
    /// Called on startup to repopulate the in-memory index.
    pub async fn rebuild(&self, records: Vec<(Uuid, StoreType, Vec<Association>)>) {
        let mut reg = self.registry.write().await;
        reg.clear();
        for (id, store_type, associations) in records {
            reg.insert(
                id,
                RegistryEntry {
                    store_type,
                    associations,
                },
            );
        }
    }
}

impl Default for HippocampalCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl AssociationGraph for HippocampalCoordinator {
    async fn get_associations(
        &self,
        record_id: &Uuid,
    ) -> Result<Vec<Association>, CerememoryError> {
        let reg = self.registry.read().await;
        match reg.get(record_id) {
            Some(entry) => Ok(entry.associations.clone()),
            None => Ok(Vec::new()),
        }
    }

    async fn get_record_store_type(
        &self,
        record_id: &Uuid,
    ) -> Result<Option<StoreType>, CerememoryError> {
        Ok(self
            .registry
            .read()
            .await
            .get(record_id)
            .map(|e| e.store_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_association(target: Uuid) -> Association {
        Association {
            target_id: target,
            association_type: AssociationType::Temporal,
            weight: 0.8,
            created_at: Utc::now(),
            last_co_activation: Utc::now(),
        }
    }

    #[tokio::test]
    async fn register_and_lookup() {
        let coord = HippocampalCoordinator::new();
        let id = Uuid::now_v7();
        coord.register(id, StoreType::Episodic, vec![]).await;

        let store = coord.get_record_store_type(&id).await.unwrap();
        assert_eq!(store, Some(StoreType::Episodic));
    }

    #[tokio::test]
    async fn unregister_removes_record() {
        let coord = HippocampalCoordinator::new();
        let id = Uuid::now_v7();
        coord.register(id, StoreType::Semantic, vec![]).await;
        assert!(coord.unregister(&id).await);
        assert_eq!(coord.get_record_store_type(&id).await.unwrap(), None);
    }

    #[tokio::test]
    async fn get_associations_returns_registered() {
        let coord = HippocampalCoordinator::new();
        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();
        let assoc = make_association(id_b);
        coord.register(id_a, StoreType::Episodic, vec![assoc]).await;

        let associations = coord.get_associations(&id_a).await.unwrap();
        assert_eq!(associations.len(), 1);
        assert_eq!(associations[0].target_id, id_b);
    }

    #[tokio::test]
    async fn cross_store_associations() {
        let coord = HippocampalCoordinator::new();
        let ep_id = Uuid::now_v7();
        let sem_id = Uuid::now_v7();

        let ep_assoc = make_association(sem_id);
        let sem_assoc = make_association(ep_id);

        coord
            .register(ep_id, StoreType::Episodic, vec![ep_assoc])
            .await;
        coord
            .register(sem_id, StoreType::Semantic, vec![sem_assoc])
            .await;

        // Episodic record points to semantic
        let ep_assocs = coord.get_associations(&ep_id).await.unwrap();
        assert_eq!(
            coord
                .get_record_store_type(&ep_assocs[0].target_id)
                .await
                .unwrap(),
            Some(StoreType::Semantic)
        );

        // Semantic record points to episodic
        let sem_assocs = coord.get_associations(&sem_id).await.unwrap();
        assert_eq!(
            coord
                .get_record_store_type(&sem_assocs[0].target_id)
                .await
                .unwrap(),
            Some(StoreType::Episodic)
        );
    }

    #[tokio::test]
    async fn rebuild_from_records() {
        let coord = HippocampalCoordinator::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        coord
            .rebuild(vec![
                (id1, StoreType::Episodic, vec![]),
                (id2, StoreType::Semantic, vec![]),
            ])
            .await;

        assert_eq!(coord.total_records().await, 2);
        let by_store = coord.records_by_store().await;
        assert_eq!(by_store[&StoreType::Episodic], 1);
        assert_eq!(by_store[&StoreType::Semantic], 1);
    }

    #[tokio::test]
    async fn records_in_store_filter() {
        let coord = HippocampalCoordinator::new();
        let e1 = Uuid::now_v7();
        let e2 = Uuid::now_v7();
        let s1 = Uuid::now_v7();

        coord.register(e1, StoreType::Episodic, vec![]).await;
        coord.register(e2, StoreType::Episodic, vec![]).await;
        coord.register(s1, StoreType::Semantic, vec![]).await;

        let episodic = coord.records_in_store(StoreType::Episodic).await;
        assert_eq!(episodic.len(), 2);
    }

    #[tokio::test]
    async fn add_association_to_existing() {
        let coord = HippocampalCoordinator::new();
        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();
        coord.register(id_a, StoreType::Episodic, vec![]).await;

        coord
            .add_association(&id_a, make_association(id_b))
            .await
            .unwrap();

        let assocs = coord.get_associations(&id_a).await.unwrap();
        assert_eq!(assocs.len(), 1);
    }

    #[tokio::test]
    async fn total_associations_counts_all() {
        let coord = HippocampalCoordinator::new();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();

        coord
            .register(a, StoreType::Episodic, vec![make_association(b)])
            .await;
        coord
            .register(
                b,
                StoreType::Semantic,
                vec![make_association(a), make_association(c)],
            )
            .await;
        coord.register(c, StoreType::Semantic, vec![]).await;

        assert_eq!(coord.total_associations().await, 3);
    }
}
