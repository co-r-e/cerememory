use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use cerememory_association::{ActivationParams, SpreadingActivationEngine};
use cerememory_core::{
    Association, AssociationEngine, AssociationGraph, AssociationType, CerememoryError, StoreType,
};
use chrono::Utc;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use uuid::Uuid;

// ─── In-memory graph implementation for benchmarks ───────────────────

/// A simple in-memory association graph for benchmark purposes.
struct BenchGraph {
    edges: HashMap<Uuid, Vec<Association>>,
}

impl BenchGraph {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
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
}

impl AssociationGraph for BenchGraph {
    async fn get_associations(
        &self,
        record_id: &Uuid,
    ) -> Result<Vec<Association>, CerememoryError> {
        Ok(self.edges.get(record_id).cloned().unwrap_or_default())
    }

    async fn get_record_store_type(
        &self,
        _record_id: &Uuid,
    ) -> Result<Option<StoreType>, CerememoryError> {
        Ok(Some(StoreType::Episodic))
    }
}

// ─── Deterministic graph generation ──────────────────────────────────

/// Deterministic pseudo-random usize from seed, bounded by max (exclusive).
fn seeded_index(seed: u64, max: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    (hasher.finish() as usize) % max
}

/// Deterministic pseudo-random f64 in [0.5, 1.0) for edge weights.
fn seeded_weight(seed: u64) -> f64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    0.5 + (hasher.finish() as f64 / u64::MAX as f64) * 0.5
}

/// Build a graph with `num_nodes` nodes, each having `edges_per_node` outgoing edges.
/// Edges are deterministically assigned to avoid self-loops and duplicates.
fn build_graph(num_nodes: usize, edges_per_node: usize) -> (BenchGraph, Vec<Uuid>) {
    let nodes: Vec<Uuid> = (0..num_nodes).map(|_| Uuid::now_v7()).collect();
    let mut graph = BenchGraph::new();

    for (i, &from) in nodes.iter().enumerate() {
        let mut targets_added = 0;
        let mut attempt = 0u64;

        while targets_added < edges_per_node && attempt < (edges_per_node as u64 * 10) {
            let target_idx = seeded_index(i as u64 * 1000 + attempt, num_nodes);
            attempt += 1;

            // Skip self-loops
            if target_idx == i {
                continue;
            }

            let weight = seeded_weight(i as u64 * 1000 + target_idx as u64);
            graph.add_edge(from, nodes[target_idx], weight);
            targets_added += 1;
        }
    }

    (graph, nodes)
}

// ─── Benchmarks ──────────────────────────────────────────────────────

fn bench_activation_node_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_node_count");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Fixed density (5 edges per node), varying node count
    for &num_nodes in &[100, 500, 1000] {
        let (graph, nodes) = build_graph(num_nodes, 5);
        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let source = nodes[0];

        group.bench_with_input(
            BenchmarkId::new("edges_5", num_nodes),
            &num_nodes,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(engine.activate(black_box(&source), 2, 0.0).await.unwrap())
                    })
                })
            },
        );
    }

    group.finish();
}

fn bench_activation_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_density");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Fixed node count (500), varying edge density
    for &edges_per_node in &[2, 5, 10] {
        let (graph, nodes) = build_graph(500, edges_per_node);
        let engine = SpreadingActivationEngine::new(Arc::new(graph));
        let source = nodes[0];

        group.bench_with_input(
            BenchmarkId::new("nodes_500", edges_per_node),
            &edges_per_node,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(engine.activate(black_box(&source), 2, 0.0).await.unwrap())
                    })
                })
            },
        );
    }

    group.finish();
}

fn bench_activation_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_depth");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Fixed graph (500 nodes, 5 edges), varying depth
    let (graph, nodes) = build_graph(500, 5);
    let engine = SpreadingActivationEngine::new(Arc::new(graph));
    let source = nodes[0];

    for &depth in &[1, 2, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("nodes_500_edges_5", depth),
            &depth,
            |b, &d| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(engine.activate(black_box(&source), d, 0.0).await.unwrap())
                    })
                })
            },
        );
    }

    group.finish();
}

fn bench_activation_with_min_weight(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_min_weight");
    let rt = tokio::runtime::Runtime::new().unwrap();

    let (graph, nodes) = build_graph(500, 10);
    let engine = SpreadingActivationEngine::new(Arc::new(graph));
    let source = nodes[0];

    for &min_weight in &[0.0, 0.3, 0.5, 0.8] {
        group.bench_with_input(
            BenchmarkId::new("nodes_500_edges_10", format!("min_{min_weight}")),
            &min_weight,
            |b, &mw| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(engine.activate(black_box(&source), 2, mw).await.unwrap())
                    })
                })
            },
        );
    }

    group.finish();
}

fn bench_activation_custom_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_custom_params");
    let rt = tokio::runtime::Runtime::new().unwrap();

    let (_graph, nodes) = build_graph(500, 5);
    let source = nodes[0];

    let configs: Vec<(&str, ActivationParams)> = vec![
        ("default", ActivationParams::default()),
        (
            "high_decay",
            ActivationParams {
                decay_factor: 0.8,
                threshold: 0.01,
                default_depth: 3,
            },
        ),
        (
            "strict_threshold",
            ActivationParams {
                decay_factor: 0.5,
                threshold: 0.2,
                default_depth: 2,
            },
        ),
    ];

    for (name, params) in configs {
        let graph_for_config = build_graph(500, 5).0;
        let engine = SpreadingActivationEngine::with_params(Arc::new(graph_for_config), params);

        group.bench_with_input(
            BenchmarkId::new("nodes_500_edges_5", name),
            &name,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(engine.activate(black_box(&source), 0, 0.0).await.unwrap())
                    })
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_activation_node_count,
    bench_activation_density,
    bench_activation_depth,
    bench_activation_with_min_weight,
    bench_activation_custom_params
);
criterion_main!(benches);
