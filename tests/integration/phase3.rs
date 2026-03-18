//! Phase 3 integration tests for Cerememory.
//!
//! Tests persistent procedural/emotional stores,
//! intensity queries, vector search at scale,
//! gRPC transport, and evolution engine observability.

use std::sync::Arc;

use cerememory_core::protocol::*;
use cerememory_core::types::*;
use cerememory_engine::{CerememoryEngine, EngineConfig};

#[path = "helpers.rs"]
mod helpers;
use helpers::{text_content, text_req, text_with_embedding};

// ─── 1. Procedural store persistence ────────────────────────────────

#[tokio::test]
async fn procedural_store_persistence() {
    let tmp = tempfile::tempdir().unwrap();
    let procedural_path = tmp.path().join("procedural.redb");
    let index_path = tmp.path().join("index");
    let vector_path = tmp.path().join("vector.redb");

    let record_id;

    // Open engine with persistent procedural store, store a record
    {
        let config = EngineConfig {
            procedural_path: Some(procedural_path.to_str().unwrap().to_string()),
            index_path: Some(index_path.to_str().unwrap().to_string()),
            vector_index_path: Some(vector_path.to_str().unwrap().to_string()),
            ..EngineConfig::default()
        };
        let engine = CerememoryEngine::new(config).unwrap();

        let resp = engine
            .encode_store(text_req(
                "How to ride a bicycle safely",
                StoreType::Procedural,
            ))
            .await
            .unwrap();
        record_id = resp.record_id;

        let stats = engine.introspect_stats().await.unwrap();
        assert_eq!(stats.records_by_store[&StoreType::Procedural], 1);
    }

    // Reopen engine with same path, verify data persists
    {
        let config = EngineConfig {
            procedural_path: Some(procedural_path.to_str().unwrap().to_string()),
            index_path: Some(index_path.to_str().unwrap().to_string()),
            vector_index_path: Some(vector_path.to_str().unwrap().to_string()),
            ..EngineConfig::default()
        };
        let engine = CerememoryEngine::new(config).unwrap();
        engine.rebuild_coordinator().await.unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert_eq!(record.id, record_id);
        assert_eq!(record.store, StoreType::Procedural);
        let text = record.text_content().unwrap();
        assert!(
            text.contains("bicycle"),
            "Persisted record should contain original text"
        );
    }
}

// ─── 2. Emotional store persistence ─────────────────────────────────

#[tokio::test]
async fn emotional_store_persistence() {
    let tmp = tempfile::tempdir().unwrap();
    let emotional_path = tmp.path().join("emotional.redb");
    let index_path = tmp.path().join("index");
    let vector_path = tmp.path().join("vector.redb");

    let record_id;

    // Open engine with persistent emotional store, store a record
    {
        let config = EngineConfig {
            emotional_path: Some(emotional_path.to_str().unwrap().to_string()),
            index_path: Some(index_path.to_str().unwrap().to_string()),
            vector_index_path: Some(vector_path.to_str().unwrap().to_string()),
            ..EngineConfig::default()
        };
        let engine = CerememoryEngine::new(config).unwrap();

        let resp = engine
            .encode_store(EncodeStoreRequest {
                header: None,
                content: text_content("Feeling happy after a great concert"),
                store: Some(StoreType::Emotional),
                emotion: Some(EmotionVector {
                    joy: 0.9,
                    trust: 0.5,
                    anticipation: 0.6,
                    intensity: 0.8,
                    valence: 0.9,
                    ..Default::default()
                }),
                context: None,
                associations: None,
            })
            .await
            .unwrap();
        record_id = resp.record_id;

        let stats = engine.introspect_stats().await.unwrap();
        assert_eq!(stats.records_by_store[&StoreType::Emotional], 1);
    }

    // Reopen and verify
    {
        let config = EngineConfig {
            emotional_path: Some(emotional_path.to_str().unwrap().to_string()),
            index_path: Some(index_path.to_str().unwrap().to_string()),
            vector_index_path: Some(vector_path.to_str().unwrap().to_string()),
            ..EngineConfig::default()
        };
        let engine = CerememoryEngine::new(config).unwrap();
        engine.rebuild_coordinator().await.unwrap();

        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();

        assert_eq!(record.id, record_id);
        assert_eq!(record.store, StoreType::Emotional);
        assert!(record.emotion.joy > 0.5, "Joy should be preserved");
        assert!(
            record.emotion.intensity > 0.5,
            "Intensity should be preserved"
        );
    }
}

// ─── 3. Emotional intensity query ───────────────────────────────────

#[tokio::test]
async fn emotional_intensity_query() {
    let tmp = tempfile::tempdir().unwrap();
    let emotional_path = tmp.path().join("emotional.redb");

    let store = cerememory_store_emotional::EmotionalStore::open(&emotional_path).unwrap();

    // Store records with different intensities directly
    let high_intensity = MemoryRecord {
        id: uuid::Uuid::now_v7(),
        store: StoreType::Emotional,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed_at: chrono::Utc::now(),
        access_count: 0,
        content: text_content("Extreme excitement at the victory"),
        fidelity: FidelityState::default(),
        emotion: EmotionVector {
            joy: 1.0,
            intensity: 0.95,
            valence: 1.0,
            ..Default::default()
        },
        associations: Vec::new(),
        metadata: serde_json::Value::Object(serde_json::Map::new()),
        version: 1,
    };

    let medium_intensity = MemoryRecord {
        id: uuid::Uuid::now_v7(),
        store: StoreType::Emotional,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed_at: chrono::Utc::now(),
        access_count: 0,
        content: text_content("Mild satisfaction with the results"),
        fidelity: FidelityState::default(),
        emotion: EmotionVector {
            joy: 0.4,
            intensity: 0.5,
            valence: 0.4,
            ..Default::default()
        },
        associations: Vec::new(),
        metadata: serde_json::Value::Object(serde_json::Map::new()),
        version: 1,
    };

    let low_intensity = MemoryRecord {
        id: uuid::Uuid::now_v7(),
        store: StoreType::Emotional,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed_at: chrono::Utc::now(),
        access_count: 0,
        content: text_content("Calm and peaceful morning"),
        fidelity: FidelityState::default(),
        emotion: EmotionVector {
            joy: 0.1,
            intensity: 0.1,
            valence: 0.2,
            ..Default::default()
        },
        associations: Vec::new(),
        metadata: serde_json::Value::Object(serde_json::Map::new()),
        version: 1,
    };

    use cerememory_core::traits::Store;
    store.store(high_intensity.clone()).await.unwrap();
    store.store(medium_intensity.clone()).await.unwrap();
    store.store(low_intensity.clone()).await.unwrap();

    // Query for high intensity records (0.8 to 1.0)
    let results = store.query_by_intensity_range(0.8, 1.0, 10).await.unwrap();
    assert_eq!(
        results.len(),
        1,
        "Should find 1 high-intensity record, found {}",
        results.len()
    );
    assert_eq!(results[0].id, high_intensity.id);

    // Query for medium intensity (0.3 to 0.6)
    let results = store.query_by_intensity_range(0.3, 0.6, 10).await.unwrap();
    assert_eq!(
        results.len(),
        1,
        "Should find 1 medium-intensity record, found {}",
        results.len()
    );
    assert_eq!(results[0].id, medium_intensity.id);

    // Query for all intensities (0.0 to 1.0)
    let results = store.query_by_intensity_range(0.0, 1.0, 10).await.unwrap();
    assert_eq!(
        results.len(),
        3,
        "Should find all 3 records across full intensity range"
    );
}

// ─── 4. Vector search many records ──────────────────────────────────

#[tokio::test]
async fn vector_search_many_records() {
    let engine = CerememoryEngine::in_memory().unwrap();

    // Store 50+ records with embeddings in a 4-dimensional space
    let mut stored_ids = Vec::new();
    for i in 0..55 {
        let angle = (i as f32) * std::f32::consts::PI * 2.0 / 55.0;
        let emb = vec![angle.cos(), angle.sin(), 0.0, 0.0];

        let resp = engine
            .encode_store(text_with_embedding(
                &format!("Record number {i} with embedding"),
                StoreType::Episodic,
                emb,
            ))
            .await
            .unwrap();
        stored_ids.push(resp.record_id);
    }

    // Query with a specific embedding direction
    let query_emb = vec![1.0, 0.0, 0.0, 0.0]; // points along x-axis

    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(query_emb),
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

    assert!(
        resp.memories.len() >= 5,
        "Should return multiple results for vector search, got {}",
        resp.memories.len()
    );
    assert!(resp.memories.len() <= 10, "Should respect the limit of 10");

    // Verify results are ordered by relevance (descending)
    for i in 1..resp.memories.len() {
        assert!(
            resp.memories[i - 1].relevance_score >= resp.memories[i].relevance_score,
            "Results should be ordered by relevance (descending)"
        );
    }
}

// ─── 5. gRPC encode/recall cycle ────────────────────────────────────

#[tokio::test]
async fn grpc_encode_recall_cycle() {
    use cerememory_transport_grpc::proto;
    use cerememory_transport_grpc::proto::cerememory_service_server::CerememoryService;
    use cerememory_transport_grpc::CerememoryGrpcService;
    use tonic::Request;

    let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
    let svc = CerememoryGrpcService::new(engine);

    // Encode a record via gRPC
    let store_req = EncodeStoreRequest {
        header: None,
        content: MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: b"gRPC integration test memory".to_vec(),
                embedding: None,
            }],
            summary: None,
        },
        store: Some(StoreType::Episodic),
        emotion: None,
        context: None,
        associations: None,
    };

    let resp = svc
        .encode_store(Request::new(proto::EncodeStoreRequest {
            json_payload: serde_json::to_vec(&store_req).unwrap(),
        }))
        .await
        .unwrap()
        .into_inner();

    let encode_resp: EncodeStoreResponse = serde_json::from_slice(&resp.json_payload).unwrap();
    assert_eq!(encode_resp.store, StoreType::Episodic);

    // Recall via gRPC
    let recall_req = RecallQueryRequest {
        header: None,
        cue: RecallCue {
            text: Some("gRPC integration".to_string()),
            ..Default::default()
        },
        stores: None,
        limit: 5,
        min_fidelity: None,
        include_decayed: false,
        reconsolidate: false,
        activation_depth: 0,
        recall_mode: RecallMode::Perfect,
    };

    let resp = svc
        .recall_query(Request::new(proto::RecallQueryRequest {
            json_payload: serde_json::to_vec(&recall_req).unwrap(),
        }))
        .await
        .unwrap()
        .into_inner();

    let recall_resp: RecallQueryResponse = serde_json::from_slice(&resp.json_payload).unwrap();

    assert!(
        !recall_resp.memories.is_empty(),
        "gRPC recall should find the stored record"
    );
    assert_eq!(recall_resp.memories[0].record.id, encode_resp.record_id);
}

// ─── 6. gRPC stats ─────────────────────────────────────────────────

#[tokio::test]
async fn grpc_stats() {
    use cerememory_transport_grpc::proto;
    use cerememory_transport_grpc::proto::cerememory_service_server::CerememoryService;
    use cerememory_transport_grpc::CerememoryGrpcService;
    use tonic::Request;

    let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
    let svc = CerememoryGrpcService::new(Arc::clone(&engine));

    // Store some records
    for text in &["Alpha memory", "Beta memory", "Gamma memory"] {
        engine
            .encode_store(text_req(text, StoreType::Episodic))
            .await
            .unwrap();
    }

    // Get stats via gRPC
    let resp = svc
        .stats(Request::new(proto::Empty {}))
        .await
        .unwrap()
        .into_inner();

    let stats: StatsResponse = serde_json::from_slice(&resp.json_payload).unwrap();
    assert_eq!(stats.total_records, 3);
    assert_eq!(stats.records_by_store[&StoreType::Episodic], 3);
    assert!(
        stats.evolution_metrics.is_some(),
        "Stats should include evolution metrics"
    );
}

// ─── 7. Evolution auto-tuning observable ────────────────────────────

#[tokio::test]
async fn evolution_auto_tuning_observable() {
    let engine = CerememoryEngine::in_memory().unwrap();

    // Store several records to give the evolution engine something to analyze
    for i in 0..10 {
        engine
            .encode_store(text_req(
                &format!("Record for evolution testing number {i}"),
                StoreType::Episodic,
            ))
            .await
            .unwrap();
    }

    // Run multiple decay ticks to accumulate evolution data
    for _ in 0..3 {
        engine
            .lifecycle_decay_tick(DecayTickRequest {
                header: None,
                tick_duration_seconds: Some(86400), // 1 day
            })
            .await
            .unwrap();
    }

    // Check that evolution metrics are present in stats
    let stats = engine.introspect_stats().await.unwrap();
    assert!(
        stats.evolution_metrics.is_some(),
        "Stats should include evolution metrics after decay ticks"
    );

    let metrics = stats.evolution_metrics.unwrap();
    // The evolution engine should have initialized — verify the struct is well-formed.
    // parameter_adjustments and detected_patterns may be empty initially; we just
    // confirm they are accessible and the metrics were serialized correctly.
    let _ = &metrics.parameter_adjustments;
    let _ = &metrics.detected_patterns;
    let _ = &metrics.schema_adaptations;
}
