use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use cerememory_core::protocol::{EncodeStoreRequest, RecallCue, RecallQueryRequest};
use cerememory_core::types::{ContentBlock, MemoryContent, Modality, RecallMode, StoreType};
use cerememory_engine::CerememoryEngine;

/// Build an EncodeStoreRequest for a text record.
fn make_store_request(i: usize) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: format!(
                    "Benchmark memory record number {i}. \
                     This record contains varied text content for testing \
                     storage and retrieval performance at scale."
                )
                .into_bytes(),
                embedding: None,
            }],
            summary: None,
        },
        store: Some(StoreType::Episodic),
        emotion: None,
        context: None,
        associations: None,
    }
}

/// Build a RecallQueryRequest with a text cue.
fn make_recall_request(query: &str) -> RecallQueryRequest {
    RecallQueryRequest {
        header: None,
        cue: RecallCue {
            text: Some(query.to_string()),
            ..RecallCue::default()
        },
        stores: None,
        limit: 10,
        min_fidelity: None,
        include_decayed: false,
        reconsolidate: false, // disable reconsolidation for stable benchmarks
        activation_depth: 0,
        recall_mode: RecallMode::Perfect,
    }
}

fn bench_encode_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_encode_store");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for &batch_size in &[10, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &n| {
                b.iter_with_setup(
                    || {
                        let engine = CerememoryEngine::in_memory().unwrap();
                        let requests: Vec<EncodeStoreRequest> =
                            (0..n).map(make_store_request).collect();
                        (engine, requests)
                    },
                    |(engine, requests)| {
                        rt.block_on(async {
                            for req in requests {
                                black_box(engine.encode_store(req).await.unwrap());
                            }
                        })
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_recall_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_recall_query");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for &record_count in &[100, 1000] {
        // Pre-populate the engine with records
        let engine = CerememoryEngine::in_memory().unwrap();
        rt.block_on(async {
            for i in 0..record_count {
                engine.encode_store(make_store_request(i)).await.unwrap();
            }
        });

        let req = make_recall_request("benchmark memory record");

        group.bench_with_input(
            BenchmarkId::from_parameter(record_count),
            &record_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(engine.recall_query(req.clone()).await.unwrap());
                    })
                })
            },
        );
    }

    group.finish();
}

fn bench_encode_then_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_e2e_encode_recall");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for &n in &[100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &count| {
            b.iter_with_setup(
                || {
                    let engine = CerememoryEngine::in_memory().unwrap();
                    let requests: Vec<EncodeStoreRequest> =
                        (0..count).map(make_store_request).collect();
                    let recall = make_recall_request("varied text content");
                    (engine, requests, recall)
                },
                |(engine, requests, recall)| {
                    rt.block_on(async {
                        // Encode all
                        for req in requests {
                            black_box(engine.encode_store(req).await.unwrap());
                        }
                        // Recall
                        black_box(engine.recall_query(recall).await.unwrap());
                    })
                },
            );
        });
    }

    group.finish();
}

fn bench_get_record(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_get_record");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for &record_count in &[100, 1000] {
        let engine = CerememoryEngine::in_memory().unwrap();
        let ids: Vec<uuid::Uuid> = rt.block_on(async {
            let mut ids = Vec::with_capacity(record_count);
            for i in 0..record_count {
                let resp = engine.encode_store(make_store_request(i)).await.unwrap();
                ids.push(resp.record_id);
            }
            ids
        });

        // Bench getting the record in the middle
        let target_id = ids[record_count / 2];

        group.bench_with_input(
            BenchmarkId::from_parameter(record_count),
            &record_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(
                            engine
                                .introspect_record(
                                    cerememory_core::protocol::RecordIntrospectRequest {
                                        header: None,
                                        record_id: target_id,
                                        include_history: false,
                                        include_associations: false,
                                        include_versions: false,
                                    },
                                )
                                .await
                                .unwrap(),
                        );
                    })
                })
            },
        );
    }

    group.finish();
}

fn bench_lifecycle_forget(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_forget");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for &n in &[10, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &count| {
            b.iter_with_setup(
                || {
                    let engine = CerememoryEngine::in_memory().unwrap();
                    let ids: Vec<uuid::Uuid> = rt.block_on(async {
                        let mut ids = Vec::with_capacity(count);
                        for i in 0..count {
                            let resp = engine.encode_store(make_store_request(i)).await.unwrap();
                            ids.push(resp.record_id);
                        }
                        ids
                    });
                    (engine, ids)
                },
                |(engine, ids)| {
                    rt.block_on(async {
                        black_box(
                            engine
                                .lifecycle_forget(cerememory_core::protocol::ForgetRequest {
                                    header: None,
                                    record_ids: Some(ids),
                                    store: None,
                                    temporal_range: None,
                                    cascade: false,
                                    confirm: true,
                                })
                                .await
                                .unwrap(),
                        );
                    })
                },
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode_store,
    bench_recall_query,
    bench_encode_then_recall,
    bench_get_record,
    bench_lifecycle_forget
);
criterion_main!(benches);
