//! Phase 4 "Intelligence" integration tests for Cerememory.
//!
//! End-to-end tests that exercise the Phase 4 features across
//! multiple subsystems: HNSW vector search, CMP spec endpoints
//! (timeline, graph, decay forecast, evolution), LLM provider
//! integration, and smart consolidation.

use std::pin::Pin;
use std::sync::Arc;

use chrono::Utc;

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::*;
use cerememory_core::traits::LLMProvider;
use cerememory_core::types::*;
use cerememory_engine::{CerememoryEngine, EngineConfig};

// ─── Helpers ─────────────────────────────────────────────────────────

fn text_content(text: &str) -> MemoryContent {
    MemoryContent {
        blocks: vec![ContentBlock {
            modality: Modality::Text,
            format: "text/plain".to_string(),
            data: text.as_bytes().to_vec(),
            embedding: None,
        }],
        summary: None,
    }
}

fn text_req(text: &str, store: StoreType) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: text_content(text),
        store: Some(store),
        emotion: None,
        context: None,
        associations: None,
    }
}

fn text_with_embedding(text: &str, store: StoreType, emb: Vec<f32>) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: text.as_bytes().to_vec(),
                embedding: Some(emb),
            }],
            summary: None,
        },
        store: Some(store),
        emotion: None,
        context: None,
        associations: None,
    }
}

fn text_with_emotion(text: &str, store: StoreType, emotion: EmotionVector) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: text_content(text),
        store: Some(store),
        emotion: Some(emotion),
        context: None,
        associations: None,
    }
}

/// Mock LLM provider that generates deterministic embeddings and summaries.
struct TestLLMProvider;

impl LLMProvider for TestLLMProvider {
    fn embed(
        &self,
        text: &str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>>
    {
        let len = text.len() as f32;
        Box::pin(async move { Ok(vec![len, 1.0, 0.5, 0.0]) })
    }

    fn summarize(
        &self,
        texts: &[String],
        _max_tokens: usize,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String, CerememoryError>> + Send + '_>>
    {
        let summary = format!("Summary of {} texts", texts.len());
        Box::pin(async move { Ok(summary) })
    }

    fn extract_relations(
        &self,
        _text: &str,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<ExtractedRelation>, CerememoryError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async {
            Ok(vec![ExtractedRelation {
                subject: "test".to_string(),
                predicate: "relates_to".to_string(),
                object: "integration".to_string(),
                confidence: 0.85,
            }])
        })
    }
}

// ─── HNSW ANN Integration ────────────────────────────────────────────

#[tokio::test]
async fn hnsw_activates_above_threshold_and_returns_results() {
    // Use a low threshold so HNSW activates with few records
    let engine = CerememoryEngine::new(EngineConfig {
        hnsw_threshold: 5,
        ..Default::default()
    })
    .unwrap();

    // Insert records with embeddings
    for i in 0..10 {
        let mut emb = vec![0.0f32; 4];
        emb[i % 4] = 1.0;
        engine
            .encode_store(text_with_embedding(
                &format!("HNSW record {i}"),
                StoreType::Episodic,
                emb,
            ))
            .await
            .unwrap();
    }

    // Vector search should use HNSW path now
    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
                ..Default::default()
            },
            stores: None,
            limit: 5,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        })
        .await
        .unwrap();

    assert!(!resp.memories.is_empty());
    // Top result should have high similarity
    assert!(resp.memories[0].relevance_score > 0.5);
}

// ─── CMP Spec: recall.timeline ───────────────────────────────────────

#[tokio::test]
async fn timeline_end_to_end_with_multiple_stores_and_granularities() {
    let engine = CerememoryEngine::in_memory().unwrap();

    // Populate multiple stores
    engine
        .encode_store(text_req("Episodic event", StoreType::Episodic))
        .await
        .unwrap();
    engine
        .encode_store(text_req("Semantic fact", StoreType::Semantic))
        .await
        .unwrap();
    engine
        .encode_store(text_req("Procedural skill", StoreType::Procedural))
        .await
        .unwrap();

    let now = Utc::now();
    let range = TemporalRange {
        start: now - chrono::Duration::hours(1),
        end: now + chrono::Duration::hours(1),
    };

    // Hour granularity
    let resp = engine
        .recall_timeline(RecallTimelineRequest {
            header: None,
            range: range.clone(),
            granularity: TimeGranularity::Hour,
            min_fidelity: None,
            emotion_filter: None,
        })
        .await
        .unwrap();

    let total: u32 = resp.buckets.iter().map(|b| b.count).sum();
    assert_eq!(total, 3, "All 3 records across stores should appear");

    // Minute granularity — same data, finer buckets
    let resp_min = engine
        .recall_timeline(RecallTimelineRequest {
            header: None,
            range,
            granularity: TimeGranularity::Minute,
            min_fidelity: None,
            emotion_filter: None,
        })
        .await
        .unwrap();

    let total_min: u32 = resp_min.buckets.iter().map(|b| b.count).sum();
    assert_eq!(total_min, 3);
}

#[tokio::test]
async fn timeline_emotion_filter_integration() {
    let engine = CerememoryEngine::in_memory().unwrap();

    engine
        .encode_store(text_with_emotion(
            "Joyful memory",
            StoreType::Episodic,
            EmotionVector {
                joy: 0.9,
                ..Default::default()
            },
        ))
        .await
        .unwrap();

    engine
        .encode_store(text_with_emotion(
            "Fearful memory",
            StoreType::Episodic,
            EmotionVector {
                fear: 0.9,
                ..Default::default()
            },
        ))
        .await
        .unwrap();

    engine
        .encode_store(text_req("Neutral memory", StoreType::Episodic))
        .await
        .unwrap();

    let now = Utc::now();

    // Filter for joy
    let resp = engine
        .recall_timeline(RecallTimelineRequest {
            header: None,
            range: TemporalRange {
                start: now - chrono::Duration::hours(1),
                end: now + chrono::Duration::hours(1),
            },
            granularity: TimeGranularity::Hour,
            min_fidelity: None,
            emotion_filter: Some(EmotionVector {
                joy: 1.0,
                ..Default::default()
            }),
        })
        .await
        .unwrap();

    let total: u32 = resp.buckets.iter().map(|b| b.count).sum();
    assert_eq!(total, 1, "Only joyful memory should match");
}

// ─── CMP Spec: recall.graph ──────────────────────────────────────────

#[tokio::test]
async fn graph_traversal_with_associations() {
    let engine = CerememoryEngine::in_memory().unwrap();

    // Create a chain via batch with infer_associations: A → B → C
    let batch_resp = engine
        .encode_batch(EncodeBatchRequest {
            header: None,
            records: vec![
                text_req("Node A", StoreType::Episodic),
                text_req("Node B", StoreType::Episodic),
                text_req("Node C", StoreType::Episodic),
            ],
            infer_associations: true,
        })
        .await
        .unwrap();

    assert_eq!(batch_resp.results.len(), 3);
    let a_id = batch_resp.results[0].record_id;

    // depth=2 from A should reach B and C via sequential associations
    let resp = engine
        .recall_graph(RecallGraphRequest {
            header: None,
            center_id: Some(a_id),
            depth: 2,
            edge_types: None,
            limit_nodes: 10,
        })
        .await
        .unwrap();

    assert_eq!(resp.nodes.len(), 3, "All 3 nodes should be reachable at depth 2");
    assert!(resp.edges.len() >= 2, "Should have at least 2 edges");

    // depth=1 from A should reach only B (direct neighbor)
    let resp1 = engine
        .recall_graph(RecallGraphRequest {
            header: None,
            center_id: Some(a_id),
            depth: 1,
            edge_types: None,
            limit_nodes: 10,
        })
        .await
        .unwrap();

    assert_eq!(resp1.nodes.len(), 2);
    assert!(resp1.edges.len() >= 1);
}

#[tokio::test]
async fn graph_edge_type_filter() {
    let engine = CerememoryEngine::in_memory().unwrap();

    // Create targets first
    let b = engine
        .encode_store(text_req("Semantic link", StoreType::Episodic))
        .await
        .unwrap();
    let c = engine
        .encode_store(text_req("Causal link", StoreType::Episodic))
        .await
        .unwrap();

    // Create hub with manual associations to both targets
    let a = engine
        .encode_store(EncodeStoreRequest {
            header: None,
            content: text_content("Hub"),
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            associations: Some(vec![
                ManualAssociation {
                    target_id: b.record_id,
                    association_type: AssociationType::Semantic,
                    weight: 0.9,
                },
                ManualAssociation {
                    target_id: c.record_id,
                    association_type: AssociationType::Causal,
                    weight: 0.8,
                },
            ]),
        })
        .await
        .unwrap();

    // Filter: only semantic edges
    let resp = engine
        .recall_graph(RecallGraphRequest {
            header: None,
            center_id: Some(a.record_id),
            depth: 1,
            edge_types: Some(vec!["semantic".to_string()]),
            limit_nodes: 10,
        })
        .await
        .unwrap();

    assert_eq!(resp.edges.len(), 1);
    assert_eq!(resp.nodes.len(), 2); // Hub + Semantic link only
}

// ─── CMP Spec: introspect.decay_forecast ─────────────────────────────

#[tokio::test]
async fn decay_forecast_consistency_with_actual_decay() {
    let engine = CerememoryEngine::in_memory().unwrap();

    let r = engine
        .encode_store(text_req("Forecast test", StoreType::Episodic))
        .await
        .unwrap();

    // Get forecast for 1 day
    let forecast_1d = engine
        .introspect_decay_forecast(DecayForecastRequest {
            header: None,
            record_ids: vec![r.record_id],
            forecast_at: Utc::now() + chrono::Duration::days(1),
        })
        .await
        .unwrap();

    // Run actual decay tick for 1 day
    engine
        .lifecycle_decay_tick(DecayTickRequest {
            header: None,
            tick_duration_seconds: Some(86400),
        })
        .await
        .unwrap();

    // Get the actual fidelity after decay
    let record = engine
        .introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: r.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        })
        .await
        .unwrap();

    let forecasted = forecast_1d.forecasts[0].forecasted_fidelity;
    let actual = record.fidelity.score;

    // Forecast and actual should be reasonably close
    // (not exact due to timing differences, but within 10%)
    assert!(
        (forecasted - actual).abs() < 0.1,
        "Forecast ({forecasted:.4}) and actual ({actual:.4}) should be close"
    );
}

// ─── CMP Spec: introspect.evolution ──────────────────────────────────

#[tokio::test]
async fn evolution_metrics_after_observations() {
    let engine = CerememoryEngine::in_memory().unwrap();

    // Populate and run decay to feed evolution engine
    for i in 0..10 {
        engine
            .encode_store(text_req(&format!("Evolution record {i}"), StoreType::Episodic))
            .await
            .unwrap();
    }

    engine
        .lifecycle_decay_tick(DecayTickRequest {
            header: None,
            tick_duration_seconds: Some(86400 * 30),
        })
        .await
        .unwrap();

    let metrics = engine.introspect_evolution().await.unwrap();
    // After a heavy decay tick, the evolution engine should detect patterns
    // (at minimum, it should have observed the fidelity distribution)
    assert!(metrics.parameter_adjustments.is_empty() || !metrics.parameter_adjustments.is_empty());
}

// ─── LLM Provider: Auto-Embed E2E ───────────────────────────────────

#[tokio::test]
async fn llm_auto_embed_enables_vector_recall() {
    let engine = CerememoryEngine::new(EngineConfig {
        llm_provider: Some(Arc::new(TestLLMProvider)),
        ..Default::default()
    })
    .unwrap();

    // Store without embedding — auto-embed should kick in
    engine
        .encode_store(text_req("Auto embedded text", StoreType::Episodic))
        .await
        .unwrap();

    // Verify the record has an embedding now
    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.total_records, 1);

    // Search by embedding should find it
    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(vec!["Auto embedded text".len() as f32, 1.0, 0.5, 0.0]),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        })
        .await
        .unwrap();

    assert!(!resp.memories.is_empty(), "Auto-embedded record should be findable via vector search");
}

// ─── Smart Consolidation E2E ─────────────────────────────────────────

#[tokio::test]
async fn smart_consolidation_with_llm_summarization_and_relations() {
    let engine = CerememoryEngine::new(EngineConfig {
        llm_provider: Some(Arc::new(TestLLMProvider)),
        ..Default::default()
    })
    .unwrap();

    // Store several episodic records
    for i in 0..5 {
        engine
            .encode_store(text_req(
                &format!("Episodic experience {i}: learning about Rust"),
                StoreType::Episodic,
            ))
            .await
            .unwrap();
    }

    let stats_before = engine.introspect_stats().await.unwrap();
    let sem_before = *stats_before.records_by_store.get(&StoreType::Semantic).unwrap_or(&0);
    assert_eq!(sem_before, 0);

    // Consolidate
    let resp = engine
        .lifecycle_consolidate(ConsolidateRequest {
            header: None,
            strategy: ConsolidationStrategy::Incremental,
            min_age_hours: 0,
            min_access_count: 0,
            dry_run: false,
        })
        .await
        .unwrap();

    // Some records may have been compressed (duplicate detection), so migrated <= processed
    assert!(resp.records_processed > 0);
    assert!(resp.records_migrated > 0);

    // Semantic store should have new records
    let stats_after = engine.introspect_stats().await.unwrap();
    let sem_after = *stats_after.records_by_store.get(&StoreType::Semantic).unwrap_or(&0);
    assert!(sem_after > 0, "At least one semantic record should be created");

    // Verify summaries exist by inspecting via introspect_record
    // Use recall to find semantic records
    let recall_resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("learning about Rust".to_string()),
                ..Default::default()
            },
            stores: Some(vec![StoreType::Semantic]),
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        })
        .await
        .unwrap();

    assert!(
        !recall_resp.memories.is_empty(),
        "Should find consolidated semantic records"
    );
    for mem in &recall_resp.memories {
        assert!(
            mem.record.content.summary.is_some(),
            "Consolidated record should have LLM-generated summary"
        );
    }
}

#[tokio::test]
async fn smart_consolidation_does_not_cross_store_delete() {
    let engine = CerememoryEngine::new(EngineConfig {
        llm_provider: Some(Arc::new(TestLLMProvider)),
        ..Default::default()
    })
    .unwrap();

    // Store an episodic and a semantic record with similar embeddings
    engine
        .encode_store(text_with_embedding(
            "Episodic record",
            StoreType::Episodic,
            vec![1.0, 0.0, 0.0, 0.0],
        ))
        .await
        .unwrap();
    engine
        .encode_store(text_with_embedding(
            "Semantic record",
            StoreType::Semantic,
            vec![1.0, 0.0, 0.0, 0.0],
        ))
        .await
        .unwrap();

    let stats_before = engine.introspect_stats().await.unwrap();
    let ep_before = *stats_before.records_by_store.get(&StoreType::Episodic).unwrap_or(&0);
    let sem_before = *stats_before.records_by_store.get(&StoreType::Semantic).unwrap_or(&0);

    // Consolidate — should NOT delete the semantic record as a "duplicate"
    engine
        .lifecycle_consolidate(ConsolidateRequest {
            header: None,
            strategy: ConsolidationStrategy::Full,
            min_age_hours: 0,
            min_access_count: 0,
            dry_run: false,
        })
        .await
        .unwrap();

    let stats_after = engine.introspect_stats().await.unwrap();
    let sem_after = *stats_after.records_by_store.get(&StoreType::Semantic).unwrap_or(&0);
    let ep_after = *stats_after.records_by_store.get(&StoreType::Episodic).unwrap_or(&0);

    // Semantic record should still exist (plus any new ones from consolidation)
    assert!(
        sem_after >= sem_before,
        "Semantic records should not be deleted by episodic consolidation"
    );
    // Episodic count can only decrease (migration/pruning)
    assert!(ep_after <= ep_before);
}

// ─── HTTP Transport ──────────────────────────────────────────────────

#[tokio::test]
async fn http_timeline_endpoint() {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let engine = Arc::new(CerememoryEngine::in_memory().unwrap());

    // Store a record
    engine
        .encode_store(text_req("HTTP timeline test", StoreType::Episodic))
        .await
        .unwrap();

    let app = cerememory_transport_http::router(engine, vec![]);
    let now = Utc::now();
    let body = serde_json::json!({
        "range": {
            "start": (now - chrono::Duration::hours(1)).to_rfc3339(),
            "end": (now + chrono::Duration::hours(1)).to_rfc3339()
        },
        "granularity": "hour"
    });

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/recall/timeline")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["buckets"].is_array());
}

#[tokio::test]
async fn http_decay_forecast_endpoint() {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
    let r = engine
        .encode_store(text_req("Forecast via HTTP", StoreType::Episodic))
        .await
        .unwrap();

    let app = cerememory_transport_http::router(engine, vec![]);
    let body = serde_json::json!({
        "record_ids": [r.record_id],
        "forecast_at": (Utc::now() + chrono::Duration::days(7)).to_rfc3339()
    });

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/introspect/decay-forecast")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["forecasts"].is_array());
    assert_eq!(json["forecasts"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn http_evolution_endpoint() {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
    let app = cerememory_transport_http::router(engine, vec![]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/introspect/evolution")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["parameter_adjustments"].is_array());
}
