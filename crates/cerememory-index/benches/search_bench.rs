use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use cerememory_core::types::StoreType;
use cerememory_index::text_index::TextIndex;
use cerememory_index::vector_index::VectorIndex;
use uuid::Uuid;

fn generate_random_embedding(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Deterministic pseudo-random for reproducibility
    let mut v = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = DefaultHasher::new();
        (seed, i as u64).hash(&mut hasher);
        let h = hasher.finish();
        v.push((h as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }
    // L2 normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");

    for &n in &[100, 500, 1000, 5000] {
        let index = VectorIndex::open_in_memory().unwrap();
        let dim = 128;

        // Populate index
        let entries: Vec<(Uuid, Vec<f32>)> = (0..n)
            .map(|i| (Uuid::now_v7(), generate_random_embedding(dim, i as u64)))
            .collect();

        for (id, emb) in &entries {
            index.upsert(*id, emb).unwrap();
        }

        let query = generate_random_embedding(dim, u64::MAX);

        group.bench_with_input(BenchmarkId::new("top10", n), &n, |b, _| {
            b.iter(|| {
                black_box(index.search(black_box(&query), 10).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("top50", n), &n, |b, _| {
            b.iter(|| {
                black_box(index.search(black_box(&query), 50).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_vector_upsert_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_upsert_batch");
    let dim = 128;

    for &batch_size in &[10, 50, 100, 500] {
        let entries: Vec<(Uuid, Vec<f32>)> = (0..batch_size)
            .map(|i| (Uuid::now_v7(), generate_random_embedding(dim, i as u64)))
            .collect();
        let refs: Vec<(Uuid, &[f32])> =
            entries.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter_with_setup(
                    || VectorIndex::open_in_memory().unwrap(),
                    |index| {
                        black_box(index.upsert_batch(black_box(&refs)).unwrap());
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_text_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_search");

    let index = TextIndex::open_in_memory().unwrap();

    // Populate with varied text
    let texts = [
        "The quick brown fox jumps over the lazy dog",
        "Rust is a systems programming language",
        "Memory databases enable fast data access",
        "Tokyo is the capital of Japan",
        "Machine learning models process embeddings",
    ];

    for i in 0..1000 {
        let text = texts[i % texts.len()];
        let id = Uuid::now_v7();
        index
            .add(
                id,
                StoreType::Episodic,
                &format!("{text} record {i}"),
                None,
            )
            .unwrap();
    }

    group.bench_function("search_1000_records", |b| {
        b.iter(|| {
            black_box(index.search(black_box("programming"), None, 10).unwrap());
        })
    });

    group.finish();
}

fn bench_text_add_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_add_batch");

    for &batch_size in &[10, 50, 100] {
        let records: Vec<(Uuid, StoreType, String)> = (0..batch_size)
            .map(|i| {
                (
                    Uuid::now_v7(),
                    StoreType::Episodic,
                    format!("Batch record number {i} with some text content"),
                )
            })
            .collect();
        let refs: Vec<(Uuid, StoreType, &str, Option<&str>)> = records
            .iter()
            .map(|(id, st, text)| (*id, *st, text.as_str(), None))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter_with_setup(
                    || TextIndex::open_in_memory().unwrap(),
                    |index| {
                        black_box(index.add_batch(black_box(&refs)).unwrap());
                    },
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_search,
    bench_vector_upsert_batch,
    bench_text_search,
    bench_text_add_batch
);
criterion_main!(benches);
