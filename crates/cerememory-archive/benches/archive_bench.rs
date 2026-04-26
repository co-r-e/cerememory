use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cerememory_archive::{export_to_bytes, import_records};
use cerememory_core::types::{MemoryRecord, StoreType};

/// Generate `n` deterministic text MemoryRecords.
fn make_records(n: usize) -> Vec<MemoryRecord> {
    let texts = [
        "The quick brown fox jumps over the lazy dog",
        "Rust is a systems programming language focused on safety",
        "Memory databases enable fast and flexible data access patterns",
        "Tokyo is the capital city of Japan with a vibrant culture",
        "Machine learning models process high-dimensional embeddings",
        "Concurrent systems require careful synchronization primitives",
        "Graph algorithms power social network and recommendation engines",
        "Cryptographic protocols ensure data integrity and confidentiality",
    ];

    (0..n)
        .map(|i| {
            let store = match i % 4 {
                0 => StoreType::Episodic,
                1 => StoreType::Semantic,
                2 => StoreType::Procedural,
                _ => StoreType::Emotional,
            };
            let text = format!("{} - record {i}", texts[i % texts.len()]);
            MemoryRecord::new_text(store, text)
        })
        .collect()
}

fn bench_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("archive_export");

    for &n in &[10, 100, 1000] {
        let records = make_records(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &records, |b, recs| {
            b.iter(|| {
                black_box(export_to_bytes(black_box(recs)).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_import(c: &mut Criterion) {
    let mut group = c.benchmark_group("archive_import");

    for &n in &[10, 100, 1000] {
        let records = make_records(n);
        let bytes = export_to_bytes(&records).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &bytes, |b, data| {
            b.iter(|| {
                black_box(import_records(black_box(data)).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("archive_roundtrip");

    for &n in &[10, 100, 1000] {
        let records = make_records(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &records, |b, recs| {
            b.iter(|| {
                let bytes = export_to_bytes(black_box(recs)).unwrap();
                black_box(import_records(&bytes).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_export_encrypted(c: &mut Criterion) {
    let mut group = c.benchmark_group("archive_export_encrypted");
    let key = cerememory_archive::crypto::derive_key("benchmark-key");

    for &n in &[10, 100, 1000] {
        let records = make_records(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &records, |b, recs| {
            b.iter(|| {
                black_box(
                    cerememory_archive::export_filtered(black_box(recs), None, Some(&key)).unwrap(),
                );
            })
        });
    }

    group.finish();
}

fn bench_import_encrypted(c: &mut Criterion) {
    let mut group = c.benchmark_group("archive_import_encrypted");
    let key = cerememory_archive::crypto::derive_key("benchmark-key");

    for &n in &[10, 100, 1000] {
        let records = make_records(n);
        let (encrypted, _) =
            cerememory_archive::export_filtered(&records, None, Some(&key)).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &encrypted, |b, data| {
            b.iter(|| {
                black_box(
                    cerememory_archive::import_records_with_key(black_box(data), Some(&key))
                        .unwrap(),
                );
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_export,
    bench_import,
    bench_roundtrip,
    bench_export_encrypted,
    bench_import_encrypted
);
criterion_main!(benches);
