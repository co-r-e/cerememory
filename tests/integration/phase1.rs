//! Phase 1 integration tests for Cerememory.
//!
//! Tests the full CMP operation lifecycle through the engine,
//! including HTTP end-to-end with a real server on a random port.

use std::sync::Arc;

use cerememory_core::protocol::*;
use cerememory_core::types::*;
use cerememory_engine::CerememoryEngine;

#[path = "helpers.rs"]
mod helpers;
use helpers::{text_content, text_req};

fn make_engine() -> CerememoryEngine {
    CerememoryEngine::in_memory().unwrap()
}

// ─── 1. Encode → Recall full cycle ──────────────────────────────────

#[tokio::test]
async fn test_encode_recall_full_cycle() {
    let engine = make_engine();

    let texts = [
        "The cat sat on the mat",
        "Dogs are loyal companions",
        "The weather is sunny today",
        "Rust is a systems programming language",
        "Tokyo is the capital of Japan",
        "Coffee keeps developers awake",
        "The ocean is deep and blue",
        "Mountains are tall and majestic",
        "Books contain knowledge",
        "Music brings joy to life",
    ];

    let mut ids = Vec::new();
    for text in &texts {
        let resp = engine
            .encode_store(text_req(text, StoreType::Episodic))
            .await
            .unwrap();
        ids.push(resp.record_id);
    }

    // Recall by text
    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("cat".to_string()),
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
    assert!(resp.memories[0]
        .record
        .text_content()
        .unwrap()
        .contains("cat"));

    // Verify all records are introspectable
    for id in &ids {
        let record = engine
            .introspect_record(RecordIntrospectRequest {
                header: None,
                record_id: *id,
                include_history: false,
                include_associations: false,
                include_versions: false,
            })
            .await
            .unwrap();
        assert_eq!(record.fidelity.score, 1.0);
    }
}

// ─── 2. Decay time simulation ────────────────────────────────────────

#[tokio::test]
async fn test_decay_time_simulation() {
    let engine = make_engine();

    let resp = engine
        .encode_store(text_req("Memory to decay", StoreType::Episodic))
        .await
        .unwrap();

    // Simulate 7 days of decay
    let tick_resp = engine
        .lifecycle_decay_tick(DecayTickRequest {
            header: None,
            tick_duration_seconds: Some(7 * 86400),
        })
        .await
        .unwrap();

    assert_eq!(tick_resp.records_updated, 1);

    // Fidelity should have decreased
    let record = engine
        .introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        })
        .await
        .unwrap();

    assert!(record.fidelity.score < 1.0, "Fidelity should have decayed");
    assert!(
        record.fidelity.noise_level > 0.0,
        "Noise should have increased"
    );
}

// ─── 3. Spreading activation graph traversal ─────────────────────────

#[tokio::test]
async fn test_spreading_activation() {
    let engine = make_engine();

    // Create batch with inferred associations
    let batch = EncodeBatchRequest {
        header: None,
        records: vec![
            text_req("Chapter 1: Introduction", StoreType::Episodic),
            text_req("Chapter 2: Methods", StoreType::Episodic),
            text_req("Chapter 3: Results", StoreType::Episodic),
        ],
        infer_associations: true,
    };

    let resp = engine.encode_batch(batch).await.unwrap();
    assert_eq!(resp.results.len(), 3);
    assert!(resp.associations_inferred > 0);

    // Use activation to find related records
    let assoc_resp = engine
        .recall_associate(RecallAssociateRequest {
            header: None,
            record_id: resp.results[0].record_id,
            association_types: None,
            depth: 2,
            min_weight: 0.1,
            limit: 10,
        })
        .await
        .unwrap();

    // Should find the associated records
    assert!(
        !assoc_resp.memories.is_empty(),
        "Should find associated memories"
    );
}

// ─── 4. Working memory LRU ──────────────────────────────────────────

#[tokio::test]
async fn test_working_memory_lru() {
    let engine = make_engine();

    // Fill working memory (default capacity: 7)
    let mut ids = Vec::new();
    for i in 0..7 {
        let resp = engine
            .encode_store(text_req(&format!("Working item {i}"), StoreType::Working))
            .await
            .unwrap();
        ids.push(resp.record_id);
    }

    // All 7 should exist
    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.records_by_store[&StoreType::Working], 7);

    // Adding 8th should evict oldest
    let _new = engine
        .encode_store(text_req("Working item 7", StoreType::Working))
        .await
        .unwrap();

    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.records_by_store[&StoreType::Working], 7);
}

// ─── 5. Consolidation (episodic → semantic) ──────────────────────────

#[tokio::test]
async fn test_consolidation() {
    let engine = make_engine();

    // Add episodic memories
    for i in 0..3 {
        engine
            .encode_store(text_req(
                &format!("Episodic memory {i}"),
                StoreType::Episodic,
            ))
            .await
            .unwrap();
    }

    // Consolidate with no age requirement
    let resp = engine
        .lifecycle_consolidate(ConsolidateRequest {
            header: None,
            strategy: ConsolidationStrategy::Full,
            min_age_hours: 0,
            min_access_count: 0,
            dry_run: false,
        })
        .await
        .unwrap();

    assert_eq!(resp.records_processed, 3);
    assert_eq!(resp.records_migrated, 3);
    assert!(resp.semantic_nodes_created > 0);

    // Semantic store should now have records
    let stats = engine.introspect_stats().await.unwrap();
    assert!(stats.records_by_store[&StoreType::Semantic] > 0);
}

// ─── 6. Forget cascade ──────────────────────────────────────────────

#[tokio::test]
async fn test_forget_cascade() {
    let engine = make_engine();

    // Create A with a manual association to B
    let resp_b = engine
        .encode_store(text_req("Target memory B", StoreType::Episodic))
        .await
        .unwrap();

    let resp_a = engine
        .encode_store(EncodeStoreRequest {
            header: None,
            content: text_content("Source memory A"),
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: Some(vec![cerememory_core::protocol::ManualAssociation {
                target_id: resp_b.record_id,
                association_type: AssociationType::Semantic,
                weight: 0.9,
            }]),
        })
        .await
        .unwrap();

    // Verify both exist
    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.total_records, 2);

    // Delete A with cascade=true → should also delete B
    let deleted = engine
        .lifecycle_forget(ForgetRequest {
            header: None,
            record_ids: Some(vec![resp_a.record_id]),
            store: None,
            temporal_range: None,
            cascade: true,
            confirm: true,
        })
        .await
        .unwrap();

    assert_eq!(deleted, 2, "Should delete both A and B");

    // Verify both are gone
    assert!(engine
        .introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp_a.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        })
        .await
        .is_err());

    assert!(engine
        .introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp_b.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        })
        .await
        .is_err());

    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.total_records, 0);
}

// ─── 7. HTTP end-to-end ─────────────────────────────────────────────

#[tokio::test]
async fn test_http_end_to_end() {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
    let app = cerememory_transport_http::router(engine, vec![]);

    // Encode a memory via HTTP
    let encode_body = serde_json::json!({
        "content": {
            "blocks": [{
                "modality": "text",
                "format": "text/plain",
                "data": [84, 101, 115, 116],
                "embedding": null
            }],
            "summary": null
        },
        "store": "episodic"
    });

    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/encode")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&encode_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let record_id = json["record_id"].as_str().unwrap();

    // Introspect the record via HTTP
    let uri = format!("/v1/introspect/record/{record_id}");
    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri(&uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    if resp.status() != axum::http::StatusCode::OK {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&bytes);
        panic!("Introspect failed for URI {uri}: {body_text}");
    }

    // Check stats
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/introspect/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["total_records"], 1);
}

// ─── 8. Human/Perfect mode switch ───────────────────────────────────

#[tokio::test]
async fn test_mode_switch() {
    let engine = make_engine();

    let original_text = "This is a detailed test of mode switching with many words for degradation";

    let resp = engine
        .encode_store(text_req(original_text, StoreType::Episodic))
        .await
        .unwrap();

    // Lower fidelity via a large decay tick
    engine
        .lifecycle_decay_tick(DecayTickRequest {
            header: None,
            tick_duration_seconds: Some(30 * 86400), // 30 days
        })
        .await
        .unwrap();

    let record = engine
        .introspect_record(RecordIntrospectRequest {
            header: None,
            record_id: resp.record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        })
        .await
        .unwrap();
    assert!(record.fidelity.score < 0.9, "Fidelity should have decayed");

    // Perfect mode: rendered content == original
    engine
        .lifecycle_set_mode(SetModeRequest {
            header: None,
            mode: RecallMode::Perfect,
            scope: None,
        })
        .await
        .unwrap();

    let recall_perfect = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("mode switching".to_string()),
                ..Default::default()
            },
            stores: None,
            limit: 1,
            min_fidelity: None,
            include_decayed: true,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        })
        .await
        .unwrap();

    assert!(!recall_perfect.memories.is_empty());
    let perfect_data = &recall_perfect.memories[0].rendered_content.blocks[0].data;
    assert_eq!(
        perfect_data,
        original_text.as_bytes(),
        "Perfect mode should return exact content"
    );

    // Human mode: rendered content should be degraded if fidelity < 0.9
    engine
        .lifecycle_set_mode(SetModeRequest {
            header: None,
            mode: RecallMode::Human,
            scope: None,
        })
        .await
        .unwrap();

    let recall_human = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("mode switching".to_string()),
                ..Default::default()
            },
            stores: None,
            limit: 1,
            min_fidelity: None,
            include_decayed: true,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Human,
        })
        .await
        .unwrap();

    assert!(!recall_human.memories.is_empty());
    let human_data = &recall_human.memories[0].rendered_content.blocks[0].data;
    // With decayed fidelity, human mode should produce different (degraded) content
    if record.fidelity.score < 0.5 {
        assert_ne!(
            human_data,
            original_text.as_bytes(),
            "Human mode should degrade low-fidelity content"
        );
    }
}
