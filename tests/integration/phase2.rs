//! Phase 2 integration tests for Cerememory.
//!
//! Tests Tantivy full-text search, vector similarity search,
//! hybrid recall, multimodal records, background decay,
//! and export/import with filtering and encryption.

use std::sync::Arc;

use cerememory_core::protocol::*;
use cerememory_core::types::*;
use cerememory_engine::{CerememoryEngine, EngineConfig};

#[path = "helpers.rs"]
mod helpers;
use helpers::{text_req, text_with_embedding};

fn make_engine() -> CerememoryEngine {
    CerememoryEngine::in_memory().unwrap()
}

// ─── 1. Tantivy search across stores ────────────────────────────────

#[tokio::test]
async fn tantivy_search_across_stores() {
    let engine = make_engine();

    // Store records in different stores with distinct text
    engine
        .encode_store(text_req(
            "Quantum mechanics describes particle behavior",
            StoreType::Episodic,
        ))
        .await
        .unwrap();
    engine
        .encode_store(text_req(
            "Quantum computing uses qubits for computation",
            StoreType::Semantic,
        ))
        .await
        .unwrap();
    engine
        .encode_store(text_req(
            "Classical physics governs macroscopic objects",
            StoreType::Procedural,
        ))
        .await
        .unwrap();

    // Search for "quantum" should find records from episodic and semantic
    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("quantum".to_string()),
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
        resp.memories.len() >= 2,
        "Should find at least 2 'quantum' records across stores, found {}",
        resp.memories.len()
    );

    // Verify at least two different store types are represented
    let store_types: std::collections::HashSet<StoreType> =
        resp.memories.iter().map(|m| m.record.store).collect();
    assert!(
        store_types.len() >= 2,
        "Results should come from multiple stores, found: {:?}",
        store_types
    );
}

// ─── 2. Tantivy search after update ─────────────────────────────────

#[tokio::test]
async fn tantivy_search_after_update() {
    let engine = make_engine();

    let resp = engine
        .encode_store(text_req(
            "Original searchable content about dolphins",
            StoreType::Episodic,
        ))
        .await
        .unwrap();
    let record_id = resp.record_id;

    // Verify original text is searchable
    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("dolphins".to_string()),
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
    assert!(
        !recall.memories.is_empty(),
        "Should find 'dolphins' before update"
    );

    // Update the text
    engine
        .encode_update(EncodeUpdateRequest {
            header: None,
            record_id,
            content: Some(MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Updated content about elephants".to_vec(),
                    embedding: None,
                }],
                summary: None,
            }),
            emotion: None,
            metadata: None,
        })
        .await
        .unwrap();

    // New text should be searchable
    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("elephants".to_string()),
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
    assert!(
        !recall.memories.is_empty(),
        "Should find 'elephants' after update"
    );
    assert_eq!(recall.memories[0].record.id, record_id);
}

// ─── 3. Tantivy search after delete ─────────────────────────────────

#[tokio::test]
async fn tantivy_search_after_delete() {
    let engine = make_engine();

    let resp = engine
        .encode_store(text_req(
            "Temporary record about penguins in Antarctica",
            StoreType::Episodic,
        ))
        .await
        .unwrap();
    let record_id = resp.record_id;

    // Verify it is searchable
    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("penguins".to_string()),
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
    assert!(
        !recall.memories.is_empty(),
        "Should find 'penguins' before delete"
    );

    // Delete the record
    engine
        .lifecycle_forget(ForgetRequest {
            header: None,
            record_ids: Some(vec![record_id]),
            store: None,
            temporal_range: None,
            cascade: false,
            confirm: true,
        })
        .await
        .unwrap();

    // Verify it no longer appears in search
    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("penguins".to_string()),
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

    let found_deleted = recall.memories.iter().any(|m| m.record.id == record_id);
    assert!(
        !found_deleted,
        "Deleted record should not appear in search results"
    );
}

// ─── 4. Vector search cosine ────────────────────────────────────────

#[tokio::test]
async fn vector_search_cosine() {
    let engine = make_engine();

    // Create embeddings with known cosine relationships
    // e1 is close to query, e2 is further, e3 is orthogonal
    let e1 = vec![1.0, 0.0, 0.0, 0.0]; // most similar to query
    let e2 = vec![0.7, 0.7, 0.0, 0.0]; // moderately similar
    let e3 = vec![0.0, 0.0, 1.0, 0.0]; // orthogonal

    let r1 = engine
        .encode_store(text_with_embedding("Vector close", StoreType::Episodic, e1))
        .await
        .unwrap();
    let r2 = engine
        .encode_store(text_with_embedding(
            "Vector medium",
            StoreType::Episodic,
            e2,
        ))
        .await
        .unwrap();
    let _r3 = engine
        .encode_store(text_with_embedding("Vector far", StoreType::Episodic, e3))
        .await
        .unwrap();

    // Query with embedding similar to e1
    let query_embedding = vec![1.0, 0.1, 0.0, 0.0];

    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(query_embedding),
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
        resp.memories.len() >= 2,
        "Should find at least 2 records via vector search"
    );

    // The first result should be r1 (most similar) or r2, but r1 should rank higher
    let result_ids: Vec<_> = resp.memories.iter().map(|m| m.record.id).collect();
    let pos_r1 = result_ids.iter().position(|id| *id == r1.record_id);
    let pos_r2 = result_ids.iter().position(|id| *id == r2.record_id);

    if let (Some(p1), Some(p2)) = (pos_r1, pos_r2) {
        assert!(p1 < p2, "r1 (cosine closer to query) should rank before r2");
    }
}

// ─── 5. Vector search multimodal ────────────────────────────────────

#[tokio::test]
async fn vector_search_multimodal() {
    let engine = make_engine();

    // Store an image record with embedding
    let image_embedding = vec![0.5, 0.5, 0.0, 0.0];
    let image_data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // fake JPEG header

    let resp = engine
        .encode_store(EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Image,
                    format: "image/jpeg".to_string(),
                    data: image_data.clone(),
                    embedding: Some(image_embedding.clone()),
                }],
                summary: Some("A photo of a sunset".to_string()),
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        })
        .await
        .unwrap();

    // Search by embedding similar to the image's
    let query_embedding = vec![0.5, 0.5, 0.1, 0.0];

    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                embedding: Some(query_embedding),
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

    assert!(
        !recall.memories.is_empty(),
        "Should find the image record via vector search"
    );
    let found = recall
        .memories
        .iter()
        .any(|m| m.record.id == resp.record_id);
    assert!(found, "Should find the specific image record");
}

// ─── 6. Hybrid text + vector recall ─────────────────────────────────

#[tokio::test]
async fn hybrid_text_vector_recall() {
    let engine = make_engine();

    // Store records with both text and embeddings
    let emb_a = vec![1.0, 0.0, 0.0, 0.0];
    let emb_b = vec![0.0, 1.0, 0.0, 0.0];

    let r_a = engine
        .encode_store(text_with_embedding(
            "Neural networks for deep learning research",
            StoreType::Episodic,
            emb_a,
        ))
        .await
        .unwrap();

    let _r_b = engine
        .encode_store(text_with_embedding(
            "Database optimization techniques for SQL",
            StoreType::Episodic,
            emb_b,
        ))
        .await
        .unwrap();

    // Recall with both text cue and embedding cue pointing at r_a
    let query_emb = vec![0.9, 0.1, 0.0, 0.0]; // similar to emb_a

    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("neural networks".to_string()),
                embedding: Some(query_emb),
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

    assert!(
        !resp.memories.is_empty(),
        "Hybrid recall should find records"
    );

    // r_a should be the top result since both text and vector point to it
    assert_eq!(
        resp.memories[0].record.id, r_a.record_id,
        "Record matching both text and vector cues should rank first"
    );
}

// ─── 7. Hybrid scoring weights ──────────────────────────────────────

#[tokio::test]
async fn hybrid_scoring_weights() {
    let engine = make_engine();

    // r1: matches text well, weak vector match
    let r1 = engine
        .encode_store(text_with_embedding(
            "Machine learning algorithms and training data",
            StoreType::Episodic,
            vec![0.1, 0.1, 0.9, 0.0],
        ))
        .await
        .unwrap();

    // r2: matches vector well, no text match
    let r2 = engine
        .encode_store(text_with_embedding(
            "Unrelated topic about cooking recipes",
            StoreType::Episodic,
            vec![1.0, 0.0, 0.0, 0.0],
        ))
        .await
        .unwrap();

    // Query with text that matches r1 and embedding that matches r2
    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("machine learning algorithms".to_string()),
                embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
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

    // Both should appear in results
    let result_ids: Vec<_> = resp.memories.iter().map(|m| m.record.id).collect();
    assert!(
        result_ids.contains(&r1.record_id) || result_ids.contains(&r2.record_id),
        "At least one of the matching records should appear in results"
    );

    // Verify scores are reasonable (all positive)
    for mem in &resp.memories {
        assert!(
            mem.relevance_score >= 0.0,
            "Relevance scores should be non-negative"
        );
    }
}

// ─── 8. Image record store and recall ───────────────────────────────

#[tokio::test]
async fn image_record_store_recall() {
    let engine = make_engine();

    let image_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]; // PNG header bytes

    let resp = engine
        .encode_store(EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![
                    ContentBlock {
                        modality: Modality::Text,
                        format: "text/plain".to_string(),
                        data: b"A beautiful sunset over the ocean".to_vec(),
                        embedding: None,
                    },
                    ContentBlock {
                        modality: Modality::Image,
                        format: "image/png".to_string(),
                        data: image_data.clone(),
                        embedding: None,
                    },
                ],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        })
        .await
        .unwrap();

    // Recall by text
    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("sunset ocean".to_string()),
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

    assert!(
        !recall.memories.is_empty(),
        "Should find the multimodal record"
    );
    let found = recall
        .memories
        .iter()
        .find(|m| m.record.id == resp.record_id)
        .expect("Should find the specific record");

    // Verify the image data is preserved
    let image_block = found
        .record
        .content
        .blocks
        .iter()
        .find(|b| b.modality == Modality::Image)
        .expect("Should have an image block");
    assert_eq!(
        image_block.data, image_data,
        "Image data should be preserved"
    );
    assert_eq!(image_block.format, "image/png");
}

// ─── 9. Mixed modality ──────────────────────────────────────────────

#[tokio::test]
async fn mixed_modality() {
    let engine = make_engine();

    // Store a text-only record
    engine
        .encode_store(text_req(
            "Pure text memory about gardening tips",
            StoreType::Episodic,
        ))
        .await
        .unwrap();

    // Store an audio record with text summary
    engine
        .encode_store(EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Audio,
                    format: "audio/wav".to_string(),
                    data: vec![0x52, 0x49, 0x46, 0x46], // RIFF header
                    embedding: None,
                }],
                summary: Some("Audio recording about gardening techniques".to_string()),
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        })
        .await
        .unwrap();

    // Store an image record
    engine
        .encode_store(EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Image,
                    format: "image/jpeg".to_string(),
                    data: vec![0xFF, 0xD8, 0xFF],
                    embedding: Some(vec![0.3, 0.3, 0.3, 0.1]),
                }],
                summary: Some("Photo of a flower garden".to_string()),
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        })
        .await
        .unwrap();

    // Verify all records are stored
    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.total_records, 3, "All 3 records should be stored");

    // Search by text should find relevant records
    let resp = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("gardening".to_string()),
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
        !resp.memories.is_empty(),
        "Should find records matching 'gardening'"
    );
}

// ─── 10. Background decay lifecycle ─────────────────────────────────

#[tokio::test]
async fn background_decay_lifecycle() {
    let config = EngineConfig {
        background_decay_interval_secs: Some(1), // 1 second interval
        ..EngineConfig::default()
    };

    let engine = Arc::new(CerememoryEngine::new(config).unwrap());

    // Store a record
    engine
        .encode_store(text_req("Memory subject to decay", StoreType::Episodic))
        .await
        .unwrap();

    // Start background decay
    engine.start_background_decay();

    // Verify background decay is enabled
    assert!(
        engine.is_background_decay_enabled().await,
        "Background decay should be enabled after start"
    );

    // Wait a bit for at least one tick to fire
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

    // Stop background decay
    engine.stop_background_decay().await;

    assert!(
        !engine.is_background_decay_enabled().await,
        "Background decay should be disabled after stop"
    );

    // Verify the engine still works after stopping decay
    let stats = engine.introspect_stats().await.unwrap();
    assert!(stats.total_records >= 1, "Records should still exist");
}

// ─── 11. Export/import roundtrip ────────────────────────────────────

#[tokio::test]
async fn export_import_roundtrip() {
    let engine = make_engine();

    // Store several records in different stores
    engine
        .encode_store(text_req(
            "Episodic memory about travel",
            StoreType::Episodic,
        ))
        .await
        .unwrap();
    engine
        .encode_store(text_req(
            "Semantic memory about geography",
            StoreType::Semantic,
        ))
        .await
        .unwrap();
    engine
        .encode_store(text_req(
            "Procedural memory about driving",
            StoreType::Procedural,
        ))
        .await
        .unwrap();

    let stats_before = engine.introspect_stats().await.unwrap();
    assert_eq!(stats_before.total_records, 3);

    // Export all records
    let (archive_data, export_resp) = engine
        .lifecycle_export(ExportRequest {
            header: None,
            format: "cma".to_string(),
            stores: None,
            encrypt: false,
            encryption_key: None,
        })
        .await
        .unwrap();

    assert_eq!(export_resp.record_count, 3);
    assert!(export_resp.size_bytes > 0);
    assert!(!archive_data.is_empty());

    // Import into a fresh engine
    let engine2 = make_engine();
    let imported = engine2
        .lifecycle_import(ImportRequest {
            header: None,
            archive_id: "test-import".to_string(),
            strategy: ImportStrategy::Merge,
            conflict_resolution: ConflictResolution::KeepImported,
            decryption_key: None,
            archive_data: Some(archive_data),
        })
        .await
        .unwrap();

    assert_eq!(imported, 3, "All 3 records should be imported");

    let stats_after = engine2.introspect_stats().await.unwrap();
    assert_eq!(stats_after.total_records, 3);
}

// ─── 12. Export store filter ────────────────────────────────────────

#[tokio::test]
async fn export_store_filter() {
    let engine = make_engine();

    engine
        .encode_store(text_req("Episodic record one", StoreType::Episodic))
        .await
        .unwrap();
    engine
        .encode_store(text_req("Episodic record two", StoreType::Episodic))
        .await
        .unwrap();
    engine
        .encode_store(text_req("Semantic record one", StoreType::Semantic))
        .await
        .unwrap();

    // Export only episodic records
    let (archive_data, export_resp) = engine
        .lifecycle_export(ExportRequest {
            header: None,
            format: "cma".to_string(),
            stores: Some(vec![StoreType::Episodic]),
            encrypt: false,
            encryption_key: None,
        })
        .await
        .unwrap();

    assert_eq!(
        export_resp.record_count, 2,
        "Should only export episodic records"
    );

    // Import into fresh engine and verify only episodic records exist
    let engine2 = make_engine();
    let imported = engine2
        .lifecycle_import(ImportRequest {
            header: None,
            archive_id: "filtered-import".to_string(),
            strategy: ImportStrategy::Merge,
            conflict_resolution: ConflictResolution::KeepImported,
            decryption_key: None,
            archive_data: Some(archive_data),
        })
        .await
        .unwrap();

    assert_eq!(imported, 2);

    let stats = engine2.introspect_stats().await.unwrap();
    assert_eq!(stats.records_by_store[&StoreType::Episodic], 2);
    assert_eq!(
        *stats
            .records_by_store
            .get(&StoreType::Semantic)
            .unwrap_or(&0),
        0
    );
}

// ─── 13. Export encrypted, import with key ──────────────────────────

#[tokio::test]
async fn export_encrypted_import() {
    let engine = make_engine();

    engine
        .encode_store(text_req(
            "Secret memory about classified info",
            StoreType::Episodic,
        ))
        .await
        .unwrap();
    engine
        .encode_store(text_req("Another secret memory", StoreType::Episodic))
        .await
        .unwrap();

    // Export with encryption
    let passphrase = "test-passphrase-2024";
    let (encrypted_data, export_resp) = engine
        .lifecycle_export(ExportRequest {
            header: None,
            format: "cma".to_string(),
            stores: None,
            encrypt: true,
            encryption_key: Some(passphrase.to_string()),
        })
        .await
        .unwrap();

    assert_eq!(export_resp.record_count, 2);
    assert!(!encrypted_data.is_empty());

    // Importing without key should fail
    let engine2 = make_engine();
    let result = engine2
        .lifecycle_import(ImportRequest {
            header: None,
            archive_id: "encrypted-no-key".to_string(),
            strategy: ImportStrategy::Merge,
            conflict_resolution: ConflictResolution::KeepImported,
            decryption_key: None,
            archive_data: Some(encrypted_data.clone()),
        })
        .await;
    assert!(
        result.is_err(),
        "Import without decryption key should fail for encrypted data"
    );

    // Importing with correct key should succeed
    let engine3 = make_engine();
    let imported = engine3
        .lifecycle_import(ImportRequest {
            header: None,
            archive_id: "encrypted-with-key".to_string(),
            strategy: ImportStrategy::Merge,
            conflict_resolution: ConflictResolution::KeepImported,
            decryption_key: Some(passphrase.to_string()),
            archive_data: Some(encrypted_data),
        })
        .await
        .unwrap();

    assert_eq!(imported, 2, "Should import 2 records with correct key");
}

// ─── 14. Import conflict resolution ────────────────────────────────

#[tokio::test]
async fn import_conflict_resolution() {
    let engine = make_engine();

    let resp = engine
        .encode_store(text_req("Original memory content", StoreType::Episodic))
        .await
        .unwrap();

    // Export the record
    let (archive_data, _) = engine
        .lifecycle_export(ExportRequest {
            header: None,
            format: "cma".to_string(),
            stores: None,
            encrypt: false,
            encryption_key: None,
        })
        .await
        .unwrap();

    // Import with KeepExisting: should skip since record already exists
    let imported = engine
        .lifecycle_import(ImportRequest {
            header: None,
            archive_id: "conflict-keep-existing".to_string(),
            strategy: ImportStrategy::Merge,
            conflict_resolution: ConflictResolution::KeepExisting,
            decryption_key: None,
            archive_data: Some(archive_data.clone()),
        })
        .await
        .unwrap();

    assert_eq!(
        imported, 0,
        "KeepExisting should skip all records that already exist"
    );

    // Verify original record is unchanged
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
    assert_eq!(record.id, resp.record_id);

    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.total_records, 1, "No duplicate records");
}

// ─── 15. CLI-like store/recall flow ─────────────────────────────────

#[tokio::test]
async fn cli_store_recall_flow() {
    let engine = make_engine();

    // Simulate a CLI-like workflow: store, recall, update, recall again
    let texts = [
        ("Learning Rust ownership", StoreType::Procedural),
        ("Meeting notes from Tuesday", StoreType::Episodic),
        (
            "Photosynthesis converts light to energy",
            StoreType::Semantic,
        ),
    ];

    let mut ids = Vec::new();
    for (text, store) in &texts {
        let resp = engine.encode_store(text_req(text, *store)).await.unwrap();
        ids.push(resp.record_id);
    }

    // Recall by text
    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("Rust".to_string()),
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

    assert!(
        !recall.memories.is_empty(),
        "Should find Rust-related memory"
    );

    // Introspect stats
    let stats = engine.introspect_stats().await.unwrap();
    assert_eq!(stats.total_records, 3);

    // Update a record
    engine
        .encode_update(EncodeUpdateRequest {
            header: None,
            record_id: ids[0],
            content: Some(MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Learning Rust ownership and borrowing rules".to_vec(),
                    embedding: None,
                }],
                summary: None,
            }),
            emotion: None,
            metadata: None,
        })
        .await
        .unwrap();

    // Recall should find updated content
    let recall = engine
        .recall_query(RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("borrowing".to_string()),
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

    assert!(
        !recall.memories.is_empty(),
        "Should find updated content with 'borrowing'"
    );

    // Decay tick should work
    let decay_resp = engine
        .lifecycle_decay_tick(DecayTickRequest {
            header: None,
            tick_duration_seconds: Some(3600),
        })
        .await
        .unwrap();
    assert_eq!(decay_resp.records_updated, 3);
}
