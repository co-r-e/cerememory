//! Spreading activation engine for Cerememory.
//!
//! Implements weighted breadth-first spreading activation with configurable
//! decay factor and hop limit. See ADR-004 for the algorithm specification.
//!
//! # Algorithm
//!
//! 1. The source record is seeded with activation = 1.0.
//! 2. At each hop, activation propagates as:
//!    `new_activation = parent_activation * edge_weight * decay_factor`
//! 3. Nodes with activation below `threshold` are pruned.
//! 4. Propagation stops at `default_depth` hops.
//! 5. If a node is reached via multiple paths, the highest activation wins.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use cerememory_core::{
    ActivatedRecord, AssociationEngine, AssociationGraph, CerememoryError,
};
use uuid::Uuid;

/// Tunable parameters for the spreading activation algorithm.
#[derive(Debug, Clone)]
pub struct ActivationParams {
    /// Multiplicative decay applied at each hop. Default: 0.5.
    pub decay_factor: f64,
    /// Minimum activation level; nodes below this are pruned. Default: 0.05.
    pub threshold: f64,
    /// Maximum number of hops from the source. Default: 2.
    pub default_depth: u32,
}

impl Default for ActivationParams {
    fn default() -> Self {
        Self {
            decay_factor: 0.5,
            threshold: 0.05,
            default_depth: 2,
        }
    }
}

/// Spreading activation engine that traverses an [`AssociationGraph`] using
/// weighted breadth-first search.
pub struct SpreadingActivationEngine<G: AssociationGraph> {
    graph: Arc<G>,
    params: ActivationParams,
}

impl<G: AssociationGraph> SpreadingActivationEngine<G> {
    /// Create a new engine with the given graph and default parameters.
    pub fn new(graph: Arc<G>) -> Self {
        Self {
            graph,
            params: ActivationParams::default(),
        }
    }

    /// Create a new engine with explicit parameters.
    pub fn with_params(graph: Arc<G>, params: ActivationParams) -> Self {
        Self { graph, params }
    }
}

/// Internal state for a single node in the BFS queue.
struct QueueEntry {
    record_id: Uuid,
    activation: f64,
    path: Vec<Uuid>,
    depth: u32,
}

impl<G: AssociationGraph> AssociationEngine for SpreadingActivationEngine<G> {
    async fn activate(
        &self,
        source_id: &Uuid,
        depth: u32,
        min_weight: f64,
    ) -> Result<Vec<ActivatedRecord>, CerememoryError> {
        let max_depth = if depth == 0 {
            self.params.default_depth
        } else {
            depth
        };

        // Track the best (highest) activation seen for each node, along with
        // the path that produced it.
        let mut best: HashMap<Uuid, (f64, Vec<Uuid>)> = HashMap::new();

        let mut queue: VecDeque<QueueEntry> = VecDeque::new();
        queue.push_back(QueueEntry {
            record_id: *source_id,
            activation: 1.0,
            path: vec![*source_id],
            depth: 0,
        });

        // Mark the source so we don't include it in results, but track it
        // to prevent re-processing at lower activation.
        best.insert(*source_id, (1.0, vec![*source_id]));

        while let Some(entry) = queue.pop_front() {
            // If this entry's activation is stale (we already found a better
            // path to this node), skip it.
            if let Some(&(best_act, _)) = best.get(&entry.record_id) {
                if entry.activation < best_act && entry.record_id != *source_id {
                    continue;
                }
            }

            // Don't expand beyond max depth.
            if entry.depth >= max_depth {
                continue;
            }

            let associations = self.graph.get_associations(&entry.record_id).await?;

            for assoc in associations {
                // Respect the caller's min_weight filter.
                if assoc.weight < min_weight {
                    continue;
                }

                let new_activation =
                    entry.activation * assoc.weight * self.params.decay_factor;

                // Prune below threshold.
                if new_activation < self.params.threshold {
                    continue;
                }

                // Skip self-loops (target already in this path).
                if entry.path.contains(&assoc.target_id) {
                    continue;
                }

                let should_enqueue = match best.get(&assoc.target_id) {
                    Some(&(prev_act, _)) => new_activation > prev_act,
                    None => true,
                };

                if should_enqueue {
                    let mut new_path = entry.path.clone();
                    new_path.push(assoc.target_id);

                    best.insert(
                        assoc.target_id,
                        (new_activation, new_path.clone()),
                    );

                    queue.push_back(QueueEntry {
                        record_id: assoc.target_id,
                        activation: new_activation,
                        path: new_path,
                        depth: entry.depth + 1,
                    });
                }
            }
        }

        // Build result: exclude the source itself.
        let mut results: Vec<ActivatedRecord> = best
            .into_iter()
            .filter(|(id, _)| id != source_id)
            .map(|(record_id, (activation_level, path))| ActivatedRecord {
                record_id,
                activation_level,
                path,
            })
            .collect();

        // Sort by activation level descending.
        results.sort_by(|a, b| {
            b.activation_level
                .partial_cmp(&a.activation_level)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::{Association, AssociationType, StoreType};
    use chrono::Utc;
    use std::collections::HashMap;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Mock graph
    // -----------------------------------------------------------------------

    /// A simple in-memory graph for testing.
    struct MockGraph {
        edges: HashMap<Uuid, Vec<Association>>,
        store_types: HashMap<Uuid, StoreType>,
    }

    impl MockGraph {
        fn new() -> Self {
            Self {
                edges: HashMap::new(),
                store_types: HashMap::new(),
            }
        }

        fn add_edge(&mut self, from: Uuid, to: Uuid, weight: f64) {
            let assoc = Association {
                target_id: to,
                association_type: AssociationType::Semantic,
                weight,
                created_at: Utc::now(),
                last_co_activation: Utc::now(),
            };
            self.edges.entry(from).or_default().push(assoc);
        }

        fn set_store_type(&mut self, id: Uuid, st: StoreType) {
            self.store_types.insert(id, st);
        }
    }

    impl AssociationGraph for MockGraph {
        async fn get_associations(
            &self,
            record_id: &Uuid,
        ) -> Result<Vec<Association>, CerememoryError> {
            Ok(self.edges.get(record_id).cloned().unwrap_or_default())
        }

        async fn get_record_store_type(
            &self,
            record_id: &Uuid,
        ) -> Result<Option<StoreType>, CerememoryError> {
            Ok(self.store_types.get(record_id).copied())
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_ids(n: usize) -> Vec<Uuid> {
        (0..n).map(|_| Uuid::now_v7()).collect()
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// Linear chain: A -> B -> C.
    /// Activate from A with default params (decay=0.5, threshold=0.05, depth=2).
    /// Expected: B = 1.0 * 0.8 * 0.5 = 0.4, C = 0.4 * 0.8 * 0.5 = 0.16.
    #[tokio::test]
    async fn linear_chain() {
        let ids = make_ids(3);
        let (a, b, c) = (ids[0], ids[1], ids[2]);

        let mut graph = MockGraph::new();
        graph.add_edge(a, b, 0.8);
        graph.add_edge(b, c, 0.8);

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&a, 0, 0.0).await.unwrap();

        assert_eq!(results.len(), 2);

        // First result should be B (higher activation).
        assert_eq!(results[0].record_id, b);
        assert!((results[0].activation_level - 0.4).abs() < 1e-9);
        assert_eq!(results[0].path, vec![a, b]);

        // Second result should be C.
        assert_eq!(results[1].record_id, c);
        assert!((results[1].activation_level - 0.16).abs() < 1e-9);
        assert_eq!(results[1].path, vec![a, b, c]);
    }

    /// Diamond graph: A -> B, A -> C, B -> D, C -> D.
    /// Activate from A. D should be reached via the stronger path.
    #[tokio::test]
    async fn diamond_graph() {
        let ids = make_ids(4);
        let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);

        let mut graph = MockGraph::new();
        graph.add_edge(a, b, 0.9); // A->B: 0.9 * 0.5 = 0.45
        graph.add_edge(a, c, 0.6); // A->C: 0.6 * 0.5 = 0.30
        graph.add_edge(b, d, 0.8); // B->D: 0.45 * 0.8 * 0.5 = 0.18
        graph.add_edge(c, d, 0.8); // C->D: 0.30 * 0.8 * 0.5 = 0.12

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&a, 0, 0.0).await.unwrap();

        assert_eq!(results.len(), 3); // B, C, D

        // D should be reached via B (stronger path).
        let d_result = results.iter().find(|r| r.record_id == d).unwrap();
        assert!((d_result.activation_level - 0.18).abs() < 1e-9);
        assert_eq!(d_result.path, vec![a, b, d]);
    }

    /// Depth limit: chain of 5 nodes with depth=2 should only find first 2 hops.
    #[tokio::test]
    async fn depth_limit() {
        let ids = make_ids(5);

        let mut graph = MockGraph::new();
        for i in 0..4 {
            graph.add_edge(ids[i], ids[i + 1], 0.9);
        }

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&ids[0], 2, 0.0).await.unwrap();

        // With depth=2, we should only reach ids[1] and ids[2].
        let reached_ids: Vec<Uuid> = results.iter().map(|r| r.record_id).collect();
        assert!(reached_ids.contains(&ids[1]));
        assert!(reached_ids.contains(&ids[2]));
        assert!(!reached_ids.contains(&ids[3]));
        assert!(!reached_ids.contains(&ids[4]));
    }

    /// Threshold pruning: very low weights should multiply below threshold.
    #[tokio::test]
    async fn threshold_pruning() {
        let ids = make_ids(3);
        let (a, b, c) = (ids[0], ids[1], ids[2]);

        let mut graph = MockGraph::new();
        // activation for B = 1.0 * 0.08 * 0.5 = 0.04, which is below 0.05 threshold
        graph.add_edge(a, b, 0.08);
        graph.add_edge(b, c, 0.5);

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&a, 0, 0.0).await.unwrap();

        // B is below threshold (0.04 < 0.05), so nothing should be found.
        assert!(results.is_empty());
    }

    /// Self-loop: A -> A should not cause infinite loop.
    #[tokio::test]
    async fn self_loop() {
        let ids = make_ids(2);
        let (a, b) = (ids[0], ids[1]);

        let mut graph = MockGraph::new();
        graph.add_edge(a, a, 0.9); // self-loop
        graph.add_edge(a, b, 0.8);

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&a, 3, 0.0).await.unwrap();

        // Should still find B, should not hang.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record_id, b);
    }

    /// Empty graph: source with no associations returns empty.
    #[tokio::test]
    async fn empty_graph() {
        let ids = make_ids(1);
        let graph = MockGraph::new();

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&ids[0], 0, 0.0).await.unwrap();

        assert!(results.is_empty());
    }

    /// min_weight filter: edges below min_weight should be skipped.
    #[tokio::test]
    async fn min_weight_filter() {
        let ids = make_ids(3);
        let (a, b, c) = (ids[0], ids[1], ids[2]);

        let mut graph = MockGraph::new();
        graph.add_edge(a, b, 0.3); // below min_weight=0.5
        graph.add_edge(a, c, 0.8); // above min_weight=0.5

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&a, 0, 0.5).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record_id, c);
    }

    /// Custom params: verify that custom decay_factor and threshold work.
    #[tokio::test]
    async fn custom_params() {
        let ids = make_ids(3);
        let (a, b, c) = (ids[0], ids[1], ids[2]);

        let mut graph = MockGraph::new();
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);

        let params = ActivationParams {
            decay_factor: 0.3,
            threshold: 0.05,
            default_depth: 2,
        };
        let engine = SpreadingActivationEngine::with_params(Arc::new(graph), params);
        let results = engine.activate(&a, 0, 0.0).await.unwrap();

        // B = 1.0 * 1.0 * 0.3 = 0.3
        // C = 0.3 * 1.0 * 0.3 = 0.09
        assert_eq!(results.len(), 2);
        assert!((results[0].activation_level - 0.3).abs() < 1e-9);
        assert!((results[1].activation_level - 0.09).abs() < 1e-9);
    }

    /// Cycle: A -> B -> C -> A should not loop infinitely.
    #[tokio::test]
    async fn cycle_does_not_loop() {
        let ids = make_ids(3);
        let (a, b, c) = (ids[0], ids[1], ids[2]);

        let mut graph = MockGraph::new();
        graph.add_edge(a, b, 0.9);
        graph.add_edge(b, c, 0.9);
        graph.add_edge(c, a, 0.9); // back to A

        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let results = engine.activate(&a, 3, 0.0).await.unwrap();

        // Should find B and C, should not re-process A.
        assert_eq!(results.len(), 2);
        let reached_ids: Vec<Uuid> = results.iter().map(|r| r.record_id).collect();
        assert!(reached_ids.contains(&b));
        assert!(reached_ids.contains(&c));
    }

    /// get_record_store_type on the mock graph.
    #[tokio::test]
    async fn mock_graph_store_type() {
        let id = Uuid::now_v7();
        let mut graph = MockGraph::new();
        graph.set_store_type(id, StoreType::Episodic);

        assert_eq!(
            graph.get_record_store_type(&id).await.unwrap(),
            Some(StoreType::Episodic)
        );
        assert_eq!(
            graph.get_record_store_type(&Uuid::now_v7()).await.unwrap(),
            None
        );
    }
}
