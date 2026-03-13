//! HTTP/REST transport binding for the Cerememory Protocol (CMP).
//!
//! Maps CMP operations to REST endpoints as defined in CMP Spec Section 9.1.
//!
//! | Method | Path                        | Operation               |
//! |--------|-----------------------------|-------------------------|
//! | POST   | /v1/encode                  | encode.store            |
//! | POST   | /v1/encode/batch            | encode.batch            |
//! | PATCH  | /v1/encode/:record_id      | encode.update           |
//! | POST   | /v1/recall/query            | recall.query            |
//! | POST   | /v1/recall/associate/{id}   | recall.associate        |
//! | POST   | /v1/lifecycle/consolidate   | lifecycle.consolidate   |
//! | POST   | /v1/lifecycle/decay-tick    | lifecycle.decay_tick    |
//! | PUT    | /v1/lifecycle/mode          | lifecycle.set_mode      |
//! | DELETE | /v1/lifecycle/forget        | lifecycle.forget        |
//! | GET    | /v1/introspect/stats        | introspect.stats        |
//! | GET    | /v1/introspect/record/{id}  | introspect.record       |

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    extract::{
        rejection::{JsonRejection, PathRejection},
        FromRequest, FromRequestParts, Path, Request, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, patch, post, put},
    Json, Router,
};
use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::*;
use cerememory_engine::CerememoryEngine;
use uuid::Uuid;

/// Shared application state.
type AppState = Arc<CerememoryEngine>;

/// Create the Axum router with all CMP endpoints.
pub fn router(engine: Arc<CerememoryEngine>) -> Router {
    Router::new()
        // Encode
        .route("/v1/encode", post(encode_store))
        .route("/v1/encode/batch", post(encode_batch))
        .route("/v1/encode/:record_id", patch(encode_update))
        // Recall
        .route("/v1/recall/query", post(recall_query))
        .route("/v1/recall/associate/:record_id", post(recall_associate))
        // Lifecycle
        .route("/v1/lifecycle/consolidate", post(lifecycle_consolidate))
        .route("/v1/lifecycle/decay-tick", post(lifecycle_decay_tick))
        .route("/v1/lifecycle/mode", put(lifecycle_set_mode))
        .route("/v1/lifecycle/forget", delete(lifecycle_forget))
        // Introspect
        .route("/v1/introspect/stats", get(introspect_stats))
        .route("/v1/introspect/record/:record_id", get(introspect_record))
        .with_state(engine)
}

/// Start the HTTP server.
pub async fn serve(
    engine: Arc<CerememoryEngine>,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = router(engine);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Cerememory HTTP server listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

// ─── Error mapping ───────────────────────────────────────────────────

/// Map CerememoryError to HTTP status + CMPError JSON.
struct AppError(CerememoryError);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            CerememoryError::RecordNotFound(_) => StatusCode::NOT_FOUND,
            CerememoryError::StoreInvalid(_) => StatusCode::BAD_REQUEST,
            CerememoryError::ContentTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            CerememoryError::ModalityUnsupported(_) => StatusCode::BAD_REQUEST,
            CerememoryError::WorkingMemoryFull => StatusCode::TOO_MANY_REQUESTS,
            CerememoryError::DecayEngineBusy { .. } => StatusCode::SERVICE_UNAVAILABLE,
            CerememoryError::ConsolidationInProgress => StatusCode::CONFLICT,
            CerememoryError::ExportFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CerememoryError::ImportConflict(_) => StatusCode::CONFLICT,
            CerememoryError::ForgetUnconfirmed => StatusCode::BAD_REQUEST,
            CerememoryError::VersionMismatch { .. } => StatusCode::BAD_REQUEST,
            CerememoryError::Validation(_) => StatusCode::BAD_REQUEST,
            CerememoryError::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CerememoryError::Serialization(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CerememoryError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let cmp_error = CMPError::from(&self.0);
        (status, Json(cmp_error)).into_response()
    }
}

impl From<CerememoryError> for AppError {
    fn from(err: CerememoryError) -> Self {
        Self(err)
    }
}

// ─── Custom extractors for CMPError envelope on rejection ────────────

/// JSON extractor that converts deserialization errors to CMPError.
struct AppJson<T>(T);

#[async_trait]
impl<S, T> FromRequest<S> for AppJson<T>
where
    axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
    T: Send,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(AppJson(value)),
            Err(rejection) => Err(AppError(CerememoryError::Validation(
                rejection.body_text(),
            ))),
        }
    }
}

/// Path extractor that converts parse errors to CMPError.
struct AppPath<T>(T);

#[async_trait]
impl<S, T> FromRequestParts<S> for AppPath<T>
where
    axum::extract::Path<T>: FromRequestParts<S, Rejection = PathRejection>,
    T: Send,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        match axum::extract::Path::<T>::from_request_parts(parts, state).await {
            Ok(Path(value)) => Ok(AppPath(value)),
            Err(rejection) => Err(AppError(CerememoryError::Validation(
                rejection.body_text(),
            ))),
        }
    }
}

// ─── Encode handlers ─────────────────────────────────────────────────

async fn encode_store(
    State(engine): State<AppState>,
    AppJson(req): AppJson<EncodeStoreRequest>,
) -> Result<Json<EncodeStoreResponse>, AppError> {
    let resp = engine.encode_store(req).await?;
    Ok(Json(resp))
}

async fn encode_batch(
    State(engine): State<AppState>,
    AppJson(req): AppJson<EncodeBatchRequest>,
) -> Result<Json<EncodeBatchResponse>, AppError> {
    let resp = engine.encode_batch(req).await?;
    Ok(Json(resp))
}

async fn encode_update(
    State(engine): State<AppState>,
    AppPath(record_id): AppPath<Uuid>,
    AppJson(mut req): AppJson<EncodeUpdateRequest>,
) -> Result<StatusCode, AppError> {
    req.record_id = record_id;
    engine.encode_update(req).await?;
    Ok(StatusCode::NO_CONTENT)
}

// ─── Recall handlers ─────────────────────────────────────────────────

async fn recall_query(
    State(engine): State<AppState>,
    AppJson(req): AppJson<RecallQueryRequest>,
) -> Result<Json<RecallQueryResponse>, AppError> {
    let resp = engine.recall_query(req).await?;
    Ok(Json(resp))
}

async fn recall_associate(
    State(engine): State<AppState>,
    AppPath(record_id): AppPath<Uuid>,
    AppJson(mut req): AppJson<RecallAssociateRequest>,
) -> Result<Json<RecallAssociateResponse>, AppError> {
    req.record_id = record_id;
    let resp = engine.recall_associate(req).await?;
    Ok(Json(resp))
}

// ─── Lifecycle handlers ──────────────────────────────────────────────

async fn lifecycle_consolidate(
    State(engine): State<AppState>,
    AppJson(req): AppJson<ConsolidateRequest>,
) -> Result<Json<ConsolidateResponse>, AppError> {
    let resp = engine.lifecycle_consolidate(req).await?;
    Ok(Json(resp))
}

async fn lifecycle_decay_tick(
    State(engine): State<AppState>,
    AppJson(req): AppJson<DecayTickRequest>,
) -> Result<Json<DecayTickResponse>, AppError> {
    let resp = engine.lifecycle_decay_tick(req).await?;
    Ok(Json(resp))
}

async fn lifecycle_set_mode(
    State(engine): State<AppState>,
    AppJson(req): AppJson<SetModeRequest>,
) -> Result<StatusCode, AppError> {
    engine.lifecycle_set_mode(req).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn lifecycle_forget(
    State(engine): State<AppState>,
    AppJson(req): AppJson<ForgetRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = engine.lifecycle_forget(req).await?;
    Ok(Json(serde_json::json!({ "records_deleted": deleted })))
}

// ─── Introspect handlers ─────────────────────────────────────────────

async fn introspect_stats(
    State(engine): State<AppState>,
) -> Result<Json<StatsResponse>, AppError> {
    let stats = engine.introspect_stats().await?;
    Ok(Json(stats))
}

async fn introspect_record(
    State(engine): State<AppState>,
    AppPath(record_id): AppPath<Uuid>,
) -> Result<Json<cerememory_core::types::MemoryRecord>, AppError> {
    let record = engine
        .introspect_record(RecordIntrospectRequest {
            header: None,
            record_id,
            include_history: false,
            include_associations: false,
            include_versions: false,
        })
        .await?;
    Ok(Json(record))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_app() -> Router {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        router(engine)
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn encode_store_and_recall() {
        let app = test_app();

        // Encode
        let req_body = serde_json::json!({
            "content": {
                "blocks": [{
                    "modality": "text",
                    "format": "text/plain",
                    "data": [72, 101, 108, 108, 111],
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
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["store"], "episodic");
        assert!(json["record_id"].is_string());

        // Recall
        let recall_body = serde_json::json!({
            "cue": { "text": "Hello" },
            "limit": 10,
            "recall_mode": "perfect",
            "activation_depth": 0
        });

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/recall/query")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&recall_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn introspect_stats_endpoint() {
        let app = test_app();

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

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total_records"], 0);
    }

    #[tokio::test]
    async fn forget_requires_confirm() {
        let app = test_app();

        let body = serde_json::json!({
            "confirm": false,
            "record_ids": []
        });

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/v1/lifecycle/forget")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["code"], "FORGET_UNCONFIRMED");
    }

    #[tokio::test]
    async fn set_mode_endpoint() {
        let app = test_app();

        let body = serde_json::json!({
            "mode": "perfect"
        });

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("PUT")
                    .uri("/v1/lifecycle/mode")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn record_not_found_returns_404() {
        let app = test_app();
        let fake_id = Uuid::now_v7();

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(format!("/v1/introspect/record/{fake_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn malformed_json_returns_cmp_error() {
        let app = test_app();

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/encode")
                    .header("content-type", "application/json")
                    .body(Body::from(b"not valid json".to_vec()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn invalid_uuid_returns_cmp_error() {
        let app = test_app();

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/v1/introspect/record/not-a-uuid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn decay_tick_endpoint() {
        let app = test_app();

        let body = serde_json::json!({
            "tick_duration_seconds": 3600
        });

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/lifecycle/decay-tick")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["records_updated"], 0);
    }
}
