//! gRPC service implementation for the Cerememory Protocol.
//!
//! Each RPC method:
//! 1. Deserializes the JSON payload from the protobuf `bytes` field.
//! 2. Calls the corresponding engine method.
//! 3. Serializes the result back to JSON bytes.
//! 4. Maps `CerememoryError` to `tonic::Status`.

use std::sync::Arc;

use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::{
    self, EXPORT_ERROR_MESSAGE, INTERNAL_ERROR_MESSAGE, SERIALIZATION_ERROR_MESSAGE,
    STORAGE_ERROR_MESSAGE,
};
use cerememory_engine::CerememoryEngine;

use crate::proto;
use crate::proto::cerememory_service_server::CerememoryService;

/// gRPC service backed by a shared `CerememoryEngine`.
pub struct CerememoryGrpcService {
    engine: Arc<CerememoryEngine>,
}

impl CerememoryGrpcService {
    pub fn new(engine: Arc<CerememoryEngine>) -> Self {
        Self { engine }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Map `CerememoryError` to an appropriate `tonic::Status` code.
fn to_status(err: CerememoryError) -> Status {
    match &err {
        CerememoryError::RecordNotFound(_) => Status::not_found(err.to_string()),
        CerememoryError::Validation(_) | CerememoryError::StoreInvalid(_) => {
            Status::invalid_argument(err.to_string())
        }
        CerememoryError::ContentTooLarge { .. } | CerememoryError::ModalityUnsupported(_) => {
            Status::invalid_argument(err.to_string())
        }
        CerememoryError::ForgetUnconfirmed => Status::failed_precondition(err.to_string()),
        CerememoryError::VersionMismatch { .. } => Status::failed_precondition(err.to_string()),
        CerememoryError::WorkingMemoryFull => Status::resource_exhausted(err.to_string()),
        CerememoryError::DecayEngineBusy { .. } => Status::unavailable(err.to_string()),
        CerememoryError::ConsolidationInProgress => Status::unavailable(err.to_string()),
        CerememoryError::ImportConflict(_) => Status::already_exists(err.to_string()),
        CerememoryError::Storage(ref msg) => {
            tracing::warn!(error = %msg, "Storage error");
            Status::internal(STORAGE_ERROR_MESSAGE)
        }
        CerememoryError::Serialization(ref msg) => {
            tracing::warn!(error = %msg, "Serialization error");
            Status::internal(SERIALIZATION_ERROR_MESSAGE)
        }
        CerememoryError::ExportFailed(ref msg) => {
            tracing::warn!(error = %msg, "Export failed");
            Status::internal(EXPORT_ERROR_MESSAGE)
        }
        CerememoryError::Internal(ref msg) => {
            tracing::warn!(error = %msg, "Internal error");
            Status::internal(INTERNAL_ERROR_MESSAGE)
        }
        CerememoryError::Unauthorized(_) => Status::unauthenticated(err.to_string()),
        CerememoryError::RateLimited { .. } => Status::resource_exhausted(err.to_string()),
    }
}

/// Deserialize a JSON payload from bytes.
#[allow(clippy::result_large_err)]
fn from_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, Status> {
    serde_json::from_slice(bytes)
        .map_err(|e| Status::invalid_argument(format!("Invalid JSON payload: {e}")))
}

/// Serialize a value to JSON bytes.
#[allow(clippy::result_large_err)]
fn to_json<T: serde::Serialize>(val: &T) -> Result<Vec<u8>, Status> {
    serde_json::to_vec(val).map_err(|e| {
        tracing::warn!(error = %e, "Failed to serialize gRPC response");
        Status::internal(SERIALIZATION_ERROR_MESSAGE)
    })
}

// ─── Service Implementation ──────────────────────────────────────────

#[tonic::async_trait]
impl CerememoryService for CerememoryGrpcService {
    // ── Encode ──

    async fn encode_store(
        &self,
        request: Request<proto::EncodeStoreRequest>,
    ) -> Result<Response<proto::EncodeStoreResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.encode_store(req).await.map_err(to_status)?;
        Ok(Response::new(proto::EncodeStoreResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn encode_batch(
        &self,
        request: Request<proto::EncodeBatchRequest>,
    ) -> Result<Response<proto::EncodeBatchResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.encode_batch(req).await.map_err(to_status)?;
        Ok(Response::new(proto::EncodeBatchResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn encode_store_raw(
        &self,
        request: Request<proto::EncodeStoreRawRequest>,
    ) -> Result<Response<proto::EncodeStoreRawResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.encode_store_raw(req).await.map_err(to_status)?;
        Ok(Response::new(proto::EncodeStoreRawResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn encode_batch_store_raw(
        &self,
        request: Request<proto::EncodeBatchStoreRawRequest>,
    ) -> Result<Response<proto::EncodeBatchStoreRawResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self
            .engine
            .encode_batch_store_raw(req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::EncodeBatchStoreRawResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn encode_update(
        &self,
        request: Request<proto::EncodeUpdateRequest>,
    ) -> Result<Response<proto::Empty>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        self.engine.encode_update(req).await.map_err(to_status)?;
        Ok(Response::new(proto::Empty {}))
    }

    // ── Recall ──

    async fn recall_query(
        &self,
        request: Request<proto::RecallQueryRequest>,
    ) -> Result<Response<proto::RecallQueryResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.recall_query(req).await.map_err(to_status)?;
        Ok(Response::new(proto::RecallQueryResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn recall_raw_query(
        &self,
        request: Request<proto::RecallRawQueryRequest>,
    ) -> Result<Response<proto::RecallRawQueryResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.recall_raw_query(req).await.map_err(to_status)?;
        Ok(Response::new(proto::RecallRawQueryResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn recall_associate(
        &self,
        request: Request<proto::RecallAssociateRequest>,
    ) -> Result<Response<proto::RecallAssociateResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.recall_associate(req).await.map_err(to_status)?;
        Ok(Response::new(proto::RecallAssociateResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn recall_timeline(
        &self,
        request: Request<proto::RecallTimelineRequest>,
    ) -> Result<Response<proto::RecallTimelineResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.recall_timeline(req).await.map_err(to_status)?;
        Ok(Response::new(proto::RecallTimelineResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn recall_graph(
        &self,
        request: Request<proto::RecallGraphRequest>,
    ) -> Result<Response<proto::RecallGraphResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self.engine.recall_graph(req).await.map_err(to_status)?;
        Ok(Response::new(proto::RecallGraphResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    // ── Lifecycle ──

    async fn consolidate(
        &self,
        request: Request<proto::ConsolidateRequest>,
    ) -> Result<Response<proto::ConsolidateResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self
            .engine
            .lifecycle_consolidate(req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::ConsolidateResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn dream_tick(
        &self,
        request: Request<proto::DreamTickRequest>,
    ) -> Result<Response<proto::DreamTickResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self
            .engine
            .lifecycle_dream_tick(req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::DreamTickResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn decay_tick(
        &self,
        request: Request<proto::DecayTickRequest>,
    ) -> Result<Response<proto::DecayTickResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self
            .engine
            .lifecycle_decay_tick(req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::DecayTickResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn set_mode(
        &self,
        request: Request<proto::SetModeRequest>,
    ) -> Result<Response<proto::Empty>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        self.engine
            .lifecycle_set_mode(req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::Empty {}))
    }

    async fn forget(
        &self,
        request: Request<proto::ForgetRequest>,
    ) -> Result<Response<proto::ForgetResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let deleted = self.engine.lifecycle_forget(req).await.map_err(to_status)?;
        Ok(Response::new(proto::ForgetResponse {
            records_deleted: deleted,
        }))
    }

    // Export: server-streaming — send archive data in 64 KiB chunks.
    type ExportStream = ReceiverStream<Result<proto::ExportChunk, Status>>;

    async fn export(
        &self,
        request: Request<proto::ExportRequest>,
    ) -> Result<Response<Self::ExportStream>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let (bytes, export_resp) = self.engine.lifecycle_export(req).await.map_err(to_status)?;
        let metadata = to_json(&export_resp)?;

        let (tx, rx) = tokio::sync::mpsc::channel(16);
        tokio::spawn(async move {
            const CHUNK_SIZE: usize = 65_536;
            let total_chunks = bytes.len().div_ceil(CHUNK_SIZE);
            // Handle empty export (0 records) gracefully.
            if total_chunks == 0 {
                let _ = tx
                    .send(Ok(proto::ExportChunk {
                        data: Vec::new(),
                        is_last: true,
                        json_metadata: metadata,
                    }))
                    .await;
                return;
            }
            for (i, chunk) in bytes.chunks(CHUNK_SIZE).enumerate() {
                let is_last = i == total_chunks - 1;
                let chunk_metadata = if i == 0 { metadata.clone() } else { Vec::new() };
                if tx
                    .send(Ok(proto::ExportChunk {
                        data: chunk.to_vec(),
                        is_last,
                        json_metadata: chunk_metadata,
                    }))
                    .await
                    .is_err()
                {
                    break; // Client disconnected.
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // Import: client-streaming — reassemble archive data from chunks.
    async fn import(
        &self,
        request: Request<tonic::Streaming<proto::ImportChunk>>,
    ) -> Result<Response<proto::ImportResponse>, Status> {
        let mut stream = request.into_inner();
        let mut all_data = Vec::new();
        let mut metadata_json: Option<Vec<u8>> = None;
        let mut is_first = true;

        const MAX_IMPORT_SIZE: usize = 256 * 1024 * 1024; // 256 MB

        while let Some(chunk) = stream.message().await? {
            // First chunk must contain metadata
            if is_first {
                if chunk.json_metadata.is_empty() {
                    return Err(Status::invalid_argument(
                        "First import chunk must contain json_metadata with import parameters",
                    ));
                }
                metadata_json = Some(chunk.json_metadata);
            }
            is_first = false;

            all_data.extend_from_slice(&chunk.data);
            if all_data.len() > MAX_IMPORT_SIZE {
                return Err(Status::resource_exhausted(format!(
                    "Import data exceeds maximum size of {} bytes",
                    MAX_IMPORT_SIZE
                )));
            }
        }

        if is_first {
            return Err(Status::invalid_argument("No import chunks received"));
        }

        let meta = metadata_json.ok_or_else(|| {
            Status::invalid_argument("Import metadata is required in the first chunk")
        })?;
        let mut import_req: protocol::ImportRequest = from_json(&meta)?;
        import_req.archive_data = Some(all_data);

        let imported = self
            .engine
            .lifecycle_import(import_req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::ImportResponse {
            records_imported: imported,
        }))
    }

    // ── Introspect ──

    async fn stats(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let resp = self.engine.introspect_stats().await.map_err(to_status)?;
        Ok(Response::new(proto::StatsResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn decay_forecast(
        &self,
        request: Request<proto::DecayForecastRequest>,
    ) -> Result<Response<proto::DecayForecastResponse>, Status> {
        let req = from_json(&request.into_inner().json_payload)?;
        let resp = self
            .engine
            .introspect_decay_forecast(req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::DecayForecastResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn evolution(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::EvolutionResponse>, Status> {
        let resp = self
            .engine
            .introspect_evolution()
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::EvolutionResponse {
            json_payload: to_json(&resp)?,
        }))
    }

    async fn introspect_record(
        &self,
        request: Request<proto::IntrospectRecordRequest>,
    ) -> Result<Response<proto::RecordResponse>, Status> {
        let record_id: uuid::Uuid = request
            .into_inner()
            .record_id
            .parse()
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {e}")))?;

        let req = protocol::RecordIntrospectRequest {
            header: None,
            record_id,
            include_history: true,
            include_associations: true,
            include_versions: true,
        };

        let record = self
            .engine
            .introspect_record(req)
            .await
            .map_err(to_status)?;
        Ok(Response::new(proto::RecordResponse {
            json_payload: to_json(&record)?,
        }))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::protocol::*;
    use cerememory_core::types::*;
    use tonic::Request;

    fn test_service() -> CerememoryGrpcService {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        CerememoryGrpcService::new(engine)
    }

    fn make_store_payload() -> Vec<u8> {
        serde_json::to_vec(&EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Hello gRPC".to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            store: Some(StoreType::Episodic),
            emotion: None,
            context: None,
            metadata: None,
            associations: None,
        })
        .unwrap()
    }

    fn make_raw_store_payload(session_id: &str, text: &str) -> Vec<u8> {
        serde_json::to_vec(&EncodeStoreRawRequest {
            header: None,
            session_id: session_id.to_string(),
            turn_id: None,
            topic_id: None,
            source: RawSource::Conversation,
            speaker: RawSpeaker::User,
            visibility: RawVisibility::Normal,
            secrecy_level: SecrecyLevel::Public,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: text.as_bytes().to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            metadata: None,
        })
        .unwrap()
    }

    /// Helper: encode a record and return its record_id.
    async fn store_one(svc: &CerememoryGrpcService) -> uuid::Uuid {
        let resp = svc
            .encode_store(Request::new(proto::EncodeStoreRequest {
                json_payload: make_store_payload(),
            }))
            .await
            .unwrap()
            .into_inner();
        let resp: EncodeStoreResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        resp.record_id
    }

    #[tokio::test]
    async fn grpc_encode_store() {
        let svc = test_service();
        let resp = svc
            .encode_store(Request::new(proto::EncodeStoreRequest {
                json_payload: make_store_payload(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: EncodeStoreResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.store, StoreType::Episodic);
        assert!(parsed.initial_fidelity > 0.0);
    }

    #[tokio::test]
    async fn grpc_encode_batch() {
        let svc = test_service();
        let batch = EncodeBatchRequest {
            header: None,
            records: vec![
                EncodeStoreRequest {
                    header: None,
                    content: MemoryContent {
                        blocks: vec![ContentBlock {
                            modality: Modality::Text,
                            format: "text/plain".to_string(),
                            data: b"Batch item 1".to_vec(),
                            embedding: None,
                        }],
                        summary: None,
                    },
                    store: Some(StoreType::Episodic),
                    emotion: None,
                    context: None,
                    metadata: None,
                    associations: None,
                },
                EncodeStoreRequest {
                    header: None,
                    content: MemoryContent {
                        blocks: vec![ContentBlock {
                            modality: Modality::Text,
                            format: "text/plain".to_string(),
                            data: b"Batch item 2".to_vec(),
                            embedding: None,
                        }],
                        summary: None,
                    },
                    store: Some(StoreType::Semantic),
                    emotion: None,
                    context: None,
                    metadata: None,
                    associations: None,
                },
            ],
            infer_associations: false,
        };

        let resp = svc
            .encode_batch(Request::new(proto::EncodeBatchRequest {
                json_payload: serde_json::to_vec(&batch).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: EncodeBatchResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.results.len(), 2);
    }

    #[tokio::test]
    async fn grpc_encode_update() {
        let svc = test_service();
        let record_id = store_one(&svc).await;

        let update = EncodeUpdateRequest {
            header: None,
            record_id,
            content: Some(MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: b"Updated content".to_vec(),
                    embedding: None,
                }],
                summary: Some("Updated".to_string()),
            }),
            emotion: None,
            metadata: None,
        };

        let resp = svc
            .encode_update(Request::new(proto::EncodeUpdateRequest {
                json_payload: serde_json::to_vec(&update).unwrap(),
            }))
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn grpc_encode_store_raw() {
        let svc = test_service();
        let resp = svc
            .encode_store_raw(Request::new(proto::EncodeStoreRawRequest {
                json_payload: make_raw_store_payload("sess-grpc-raw", "Hello raw gRPC"),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: EncodeStoreRawResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.session_id, "sess-grpc-raw");
        assert_eq!(parsed.visibility, RawVisibility::Normal);
    }

    #[tokio::test]
    async fn grpc_recall_query() {
        let svc = test_service();
        let _ = store_one(&svc).await;

        let query = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some("Hello".to_string()),
                ..Default::default()
            },
            stores: None,
            limit: 10,
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: false,
            activation_depth: 0,
            recall_mode: RecallMode::Perfect,
        };

        let resp = svc
            .recall_query(Request::new(proto::RecallQueryRequest {
                json_payload: serde_json::to_vec(&query).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: RecallQueryResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert!(!parsed.memories.is_empty());
    }

    #[tokio::test]
    async fn grpc_recall_raw_query() {
        let svc = test_service();
        let _ = svc
            .encode_store_raw(Request::new(proto::EncodeStoreRawRequest {
                json_payload: make_raw_store_payload("sess-grpc-raw", "Hello raw gRPC"),
            }))
            .await
            .unwrap();

        let req = RecallRawQueryRequest {
            header: None,
            session_id: Some("sess-grpc-raw".to_string()),
            query: Some("raw gRPC".to_string()),
            temporal: None,
            limit: 10,
            include_private_scratch: false,
            include_sealed: false,
            secrecy_levels: None,
        };

        let resp = svc
            .recall_raw_query(Request::new(proto::RecallRawQueryRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: RecallRawQueryResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.total_candidates, 1);
        assert_eq!(parsed.records[0].session_id, "sess-grpc-raw");
    }

    #[tokio::test]
    async fn grpc_recall_associate() {
        let svc = test_service();
        let record_id = store_one(&svc).await;

        let req = RecallAssociateRequest {
            header: None,
            record_id,
            association_types: None,
            depth: 1,
            min_weight: 0.0,
            limit: 10,
        };

        let resp = svc
            .recall_associate(Request::new(proto::RecallAssociateRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: RecallAssociateResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        // With a single record there may be no associations, but the call should succeed.
        assert_eq!(parsed.total_candidates, 0);
    }

    #[tokio::test]
    async fn grpc_consolidate() {
        let svc = test_service();

        let req = ConsolidateRequest {
            header: None,
            strategy: ConsolidationStrategy::Incremental,
            min_age_hours: 0,
            min_access_count: 0,
            dry_run: true,
        };

        let resp = svc
            .consolidate(Request::new(proto::ConsolidateRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: ConsolidateResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.records_processed, 0);
    }

    #[tokio::test]
    async fn grpc_dream_tick() {
        let svc = test_service();
        let _ = svc
            .encode_store_raw(Request::new(proto::EncodeStoreRawRequest {
                json_payload: make_raw_store_payload("sess-grpc-dream", "Dream this gRPC note"),
            }))
            .await
            .unwrap();

        let req = DreamTickRequest {
            header: None,
            session_id: Some("sess-grpc-dream".to_string()),
            dry_run: false,
            max_groups: 10,
            include_private_scratch: false,
            include_sealed: false,
            promote_semantic: true,
            secrecy_levels: None,
        };

        let resp = svc
            .dream_tick(Request::new(proto::DreamTickRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: DreamTickResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.groups_processed, 1);
        assert_eq!(parsed.episodic_summaries_created, 1);
        assert_eq!(parsed.semantic_nodes_created, 0);
    }

    #[tokio::test]
    async fn grpc_decay_tick() {
        let svc = test_service();

        let req = DecayTickRequest {
            header: None,
            tick_duration_seconds: Some(3600),
        };

        let resp = svc
            .decay_tick(Request::new(proto::DecayTickRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: DecayTickResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.records_updated, 0);
    }

    #[tokio::test]
    async fn grpc_set_mode() {
        let svc = test_service();

        let req = SetModeRequest {
            header: None,
            mode: RecallMode::Perfect,
            scope: None,
        };

        let resp = svc
            .set_mode(Request::new(proto::SetModeRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn grpc_forget() {
        let svc = test_service();
        let record_id = store_one(&svc).await;

        let req = ForgetRequest {
            header: None,
            record_ids: Some(vec![record_id]),
            store: None,
            temporal_range: None,
            cascade: false,
            confirm: true,
        };

        let resp = svc
            .forget(Request::new(proto::ForgetRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.records_deleted, 1);
    }

    #[tokio::test]
    async fn grpc_stats() {
        let svc = test_service();

        let resp = svc
            .stats(Request::new(proto::Empty {}))
            .await
            .unwrap()
            .into_inner();

        let parsed: StatsResponse = serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.total_records, 0);
    }

    #[tokio::test]
    async fn grpc_introspect_record() {
        let svc = test_service();
        let record_id = store_one(&svc).await;

        let resp = svc
            .introspect_record(Request::new(proto::IntrospectRecordRequest {
                record_id: record_id.to_string(),
            }))
            .await
            .unwrap()
            .into_inner();

        let parsed: cerememory_core::types::MemoryRecord =
            serde_json::from_slice(&resp.json_payload).unwrap();
        assert_eq!(parsed.id, record_id);
    }

    #[tokio::test]
    async fn grpc_not_found_status() {
        let svc = test_service();
        let fake_id = uuid::Uuid::now_v7();

        let resp = svc
            .introspect_record(Request::new(proto::IntrospectRecordRequest {
                record_id: fake_id.to_string(),
            }))
            .await;

        let err = resp.unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn grpc_validation_error_status() {
        let svc = test_service();

        // Send invalid JSON payload.
        let resp = svc
            .encode_store(Request::new(proto::EncodeStoreRequest {
                json_payload: b"not valid json".to_vec(),
            }))
            .await;

        let err = resp.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn grpc_storage_error_status_is_sanitized() {
        let status = to_status(CerememoryError::Storage("database path leaked".to_string()));
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(status.message(), STORAGE_ERROR_MESSAGE);
    }

    #[test]
    fn grpc_serialization_error_status_is_sanitized() {
        let status = to_status(CerememoryError::Serialization(
            "serde detail leaked".to_string(),
        ));
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(status.message(), SERIALIZATION_ERROR_MESSAGE);
    }

    #[tokio::test]
    async fn grpc_forget_unconfirmed_status() {
        let svc = test_service();

        let req = ForgetRequest {
            header: None,
            record_ids: None,
            store: None,
            temporal_range: None,
            cascade: false,
            confirm: false,
        };

        let resp = svc
            .forget(Request::new(proto::ForgetRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await;

        let err = resp.unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test]
    async fn grpc_export_streaming() {
        let svc = test_service();
        // Store a record so the export has data.
        let _ = store_one(&svc).await;

        let req = ExportRequest {
            header: None,
            format: "cma".to_string(),
            stores: None,
            include_raw_journal: false,
            encrypt: false,
            encryption_key: None,
        };

        let resp = svc
            .export(Request::new(proto::ExportRequest {
                json_payload: serde_json::to_vec(&req).unwrap(),
            }))
            .await
            .unwrap();

        let mut stream = resp.into_inner();
        let mut total_bytes = 0usize;
        let mut saw_last = false;

        while let Some(chunk) = tokio_stream::StreamExt::next(&mut stream).await {
            let chunk = chunk.unwrap();
            total_bytes += chunk.data.len();
            if chunk.is_last {
                saw_last = true;
            }
        }

        assert!(total_bytes > 0, "Export should produce data");
        assert!(saw_last, "Stream should end with is_last=true");
    }
}
