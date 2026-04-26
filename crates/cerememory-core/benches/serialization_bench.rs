use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cerememory_core::types::{
    ContentBlock, MemoryContent, MemoryRecord, MetaMemory, Modality, StoreType,
};

/// Create a text-only MemoryRecord.
fn make_text_record() -> MemoryRecord {
    MemoryRecord::new_text(
        StoreType::Episodic,
        "The quick brown fox jumps over the lazy dog. This is a representative \
         text memory record used for serialization benchmarks.",
    )
}

/// Create a binary (image) MemoryRecord with a 4 KB payload.
fn make_binary_record() -> MemoryRecord {
    let data = vec![0xABu8; 4096]; // 4 KB of synthetic image data
    MemoryRecord::new_binary(
        StoreType::Semantic,
        Modality::Image,
        "image/png",
        data,
        None,
    )
}

/// Create a multi-block MemoryRecord with 3 content blocks.
fn make_multi_block_record() -> MemoryRecord {
    let text_block = ContentBlock {
        modality: Modality::Text,
        format: "text/plain".to_string(),
        data: b"Multi-block memory: textual context for the associated media.".to_vec(),
        embedding: Some(vec![0.1f32; 128]), // 128-dim embedding
    };
    let image_block = ContentBlock {
        modality: Modality::Image,
        format: "image/jpeg".to_string(),
        data: vec![0xFFu8; 2048], // 2 KB synthetic image
        embedding: None,
    };
    let structured_block = ContentBlock {
        modality: Modality::Structured,
        format: "application/json".to_string(),
        data: br#"{"key":"value","count":42,"tags":["a","b","c"]}"#.to_vec(),
        embedding: None,
    };

    let now = chrono::Utc::now();
    MemoryRecord {
        id: uuid::Uuid::now_v7(),
        store: StoreType::Procedural,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        access_count: 0,
        content: MemoryContent {
            blocks: vec![text_block, image_block, structured_block],
            summary: Some(
                "A multi-modal memory with text, image, and structured data.".to_string(),
            ),
        },
        fidelity: cerememory_core::types::FidelityState::default(),
        emotion: cerememory_core::types::EmotionVector::default(),
        associations: Vec::new(),
        metadata: serde_json::json!({"source": "benchmark"}),
        meta: MetaMemory::unavailable("benchmark"),
        version: 1,
    }
}

fn bench_json_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_serialize");

    let cases: Vec<(&str, MemoryRecord)> = vec![
        ("text", make_text_record()),
        ("binary_4kb", make_binary_record()),
        ("multi_block_3", make_multi_block_record()),
    ];

    for (name, record) in &cases {
        group.bench_with_input(BenchmarkId::new("encode", *name), record, |b, rec| {
            b.iter(|| {
                black_box(serde_json::to_vec(black_box(rec)).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_json_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_deserialize");

    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("text", serde_json::to_vec(&make_text_record()).unwrap()),
        (
            "binary_4kb",
            serde_json::to_vec(&make_binary_record()).unwrap(),
        ),
        (
            "multi_block_3",
            serde_json::to_vec(&make_multi_block_record()).unwrap(),
        ),
    ];

    for (name, data) in &cases {
        group.bench_with_input(BenchmarkId::new("decode", *name), data, |b, bytes| {
            b.iter(|| {
                black_box(serde_json::from_slice::<MemoryRecord>(black_box(bytes)).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_msgpack_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("msgpack_serialize");

    let cases: Vec<(&str, MemoryRecord)> = vec![
        ("text", make_text_record()),
        ("binary_4kb", make_binary_record()),
        ("multi_block_3", make_multi_block_record()),
    ];

    for (name, record) in &cases {
        group.bench_with_input(BenchmarkId::new("encode", *name), record, |b, rec| {
            b.iter(|| {
                black_box(rmp_serde::to_vec(black_box(rec)).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_msgpack_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("msgpack_deserialize");

    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("text", rmp_serde::to_vec(&make_text_record()).unwrap()),
        (
            "binary_4kb",
            rmp_serde::to_vec(&make_binary_record()).unwrap(),
        ),
        (
            "multi_block_3",
            rmp_serde::to_vec(&make_multi_block_record()).unwrap(),
        ),
    ];

    for (name, data) in &cases {
        group.bench_with_input(BenchmarkId::new("decode", *name), data, |b, bytes| {
            b.iter(|| {
                black_box(rmp_serde::from_slice::<MemoryRecord>(black_box(bytes)).unwrap());
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_json_serialization,
    bench_json_deserialization,
    bench_msgpack_serialization,
    bench_msgpack_deserialization
);
criterion_main!(benches);
