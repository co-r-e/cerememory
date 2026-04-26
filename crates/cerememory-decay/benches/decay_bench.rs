use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cerememory_core::{DecayEngine, DecayInput, EmotionVector, FidelityState};
use cerememory_decay::{DecayParams, PowerLawDecayEngine};
use chrono::{Duration, Utc};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

/// Deterministic pseudo-random f64 in [0.0, 1.0) from a seed.
fn seeded_f64(seed: u64) -> f64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    (hasher.finish() as f64) / (u64::MAX as f64)
}

/// Generate a Vec of DecayInput with deterministic pseudo-random fidelity states.
fn generate_decay_inputs(n: usize) -> Vec<DecayInput> {
    let base_time = Utc::now() - Duration::hours(1);

    (0..n)
        .map(|i| {
            let seed = i as u64;
            let score = 0.3 + seeded_f64(seed * 3) * 0.7; // [0.3, 1.0)
            let stability = 1.0 + seeded_f64(seed * 3 + 1) * 999.0; // [1.0, 1000.0)
            let noise = seeded_f64(seed * 3 + 2) * 0.3; // [0.0, 0.3)
            let intensity = seeded_f64(seed * 7) * 0.8; // [0.0, 0.8)

            let offset = Duration::seconds((i % 3600) as i64);
            let last_accessed = base_time + offset;

            let fidelity = FidelityState {
                score,
                noise_level: noise,
                decay_rate: 0.3,
                emotional_anchor: 1.0,
                reinforcement_count: (i % 10) as u32,
                stability,
                last_decay_tick: last_accessed,
            };

            let emotion = EmotionVector {
                intensity,
                ..EmotionVector::default()
            };

            DecayInput {
                id: Uuid::now_v7(),
                fidelity,
                emotion,
                last_accessed_at: last_accessed,
                access_count: (i % 20) as u32,
            }
        })
        .collect()
}

fn bench_compute_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_compute_tick");
    let engine = PowerLawDecayEngine::with_defaults();

    for &n in &[100, 1_000, 10_000, 100_000] {
        let inputs = generate_decay_inputs(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &inputs, |b, records| {
            b.iter(|| {
                black_box(engine.compute_tick(black_box(records), 3600.0));
            })
        });
    }

    group.finish();
}

fn bench_compute_tick_varying_duration(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_tick_duration");
    let engine = PowerLawDecayEngine::with_defaults();
    let inputs = generate_decay_inputs(10_000);

    for &duration_secs in &[60.0, 3600.0, 86400.0, 604800.0] {
        group.bench_with_input(
            BenchmarkId::new("10k_records", format!("{duration_secs}s")),
            &inputs,
            |b, records| {
                b.iter(|| {
                    black_box(engine.compute_tick(black_box(records), duration_secs));
                })
            },
        );
    }

    group.finish();
}

fn bench_boost_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_boost_stability");
    let engine = PowerLawDecayEngine::with_defaults();

    for &stability in &[1.0, 10.0, 100.0, 1000.0] {
        group.bench_with_input(
            BenchmarkId::from_parameter(stability as u64),
            &stability,
            |b, &s| {
                b.iter(|| {
                    black_box(engine.boost_stability(black_box(s)));
                })
            },
        );
    }

    group.finish();
}

fn bench_custom_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_custom_params");
    let inputs = generate_decay_inputs(10_000);

    let configs: Vec<(&str, DecayParams)> = vec![
        ("default", DecayParams::default()),
        (
            "aggressive",
            DecayParams {
                decay_exponent: 0.8,
                prune_threshold: 0.1,
                interference_rate: 0.3,
                ..DecayParams::default()
            },
        ),
        (
            "conservative",
            DecayParams {
                decay_exponent: 0.1,
                prune_threshold: 0.001,
                interference_rate: 0.01,
                ..DecayParams::default()
            },
        ),
    ];

    for (name, params) in configs {
        let engine = PowerLawDecayEngine::new(params);
        group.bench_with_input(
            BenchmarkId::new("10k_records", name),
            &inputs,
            |b, records| {
                b.iter(|| {
                    black_box(engine.compute_tick(black_box(records), 3600.0));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compute_tick,
    bench_compute_tick_varying_duration,
    bench_boost_stability,
    bench_custom_params
);
criterion_main!(benches);
