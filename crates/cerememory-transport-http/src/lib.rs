//! HTTP/REST transport binding for the Cerememory Protocol (CMP).
//!
//! Maps CMP operations to REST endpoints as defined in CMP Spec Section 9.1.
//!
//! | Method | Path                        | Operation               |
//! |--------|-----------------------------|-------------------------|
//! | POST   | /v1/encode                  | encode.store            |
//! | POST   | /v1/encode/batch            | encode.batch            |
//! | PATCH  | /v1/encode/{record_id}     | encode.update           |
//! | POST   | /v1/recall/query            | recall.query            |
//! | POST   | /v1/recall/associate/{id}   | recall.associate        |
//! | POST   | /v1/lifecycle/consolidate   | lifecycle.consolidate   |
//! | POST   | /v1/lifecycle/decay-tick    | lifecycle.decay_tick    |
//! | PUT    | /v1/lifecycle/mode          | lifecycle.set_mode      |
//! | DELETE | /v1/lifecycle/forget        | lifecycle.forget        |
//! | GET    | /v1/introspect/stats        | introspect.stats        |
//! | GET    | /v1/introspect/record/{id}  | introspect.record       |

use std::{convert::Infallible, sync::Arc};

use axum::{
    extract::{
        rejection::{JsonRejection, PathRejection},
        DefaultBodyLimit, FromRequest, FromRequestParts, Path, Request, State,
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, patch, post, put},
    Json, Router,
};
use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::*;
use cerememory_engine::CerememoryEngine;
use tower_http::cors::AllowOrigin;
use tower_http::cors::{Any, CorsLayer};
use tower_http::request_id::{
    MakeRequestUuid, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

pub mod auth;
pub mod metrics;
pub mod rate_limit;

pub use auth::ApiKeyAuthLayer;
pub use metrics::{install_prometheus_recorder, MetricsLayer};
use metrics_exporter_prometheus::PrometheusHandle;
pub use rate_limit::RateLimitLayer;

/// Options for configuring HTTP middleware.
pub struct HttpMiddlewareConfig {
    /// API keys for Bearer token authentication.
    pub api_keys: Vec<String>,

    /// Whether the API key layer should be active.
    /// When false, configured keys are ignored.
    pub auth_enabled: bool,

    /// Allowed CORS origins (empty = emit no CORS headers).
    pub cors_origins: Vec<String>,

    /// Trusted proxy CIDRs for forwarded client IP extraction.
    pub trusted_proxy_cidrs: Vec<String>,

    /// Whether to expose `/metrics`.
    pub metrics_enabled: bool,

    /// Rate limit: requests per second.
    pub rate_limit_rps: u64,
    /// Rate limit: burst size.
    pub rate_limit_burst: u32,
    /// Prometheus handle for /metrics endpoint (None = no metrics endpoint).
    pub prometheus_handle: Option<PrometheusHandle>,
}

impl Default for HttpMiddlewareConfig {
    fn default() -> Self {
        Self {
            api_keys: Vec::new(),
            auth_enabled: true,
            cors_origins: Vec::new(),
            trusted_proxy_cidrs: Vec::new(),
            metrics_enabled: false,
            rate_limit_rps: 100,
            rate_limit_burst: 50,
            prometheus_handle: None,
        }
    }
}

/// Shared application state.
type AppState = Arc<CerememoryEngine>;

pub(crate) fn request_id_from_headers(headers: &HeaderMap) -> Option<Uuid> {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
}

pub(crate) fn request_id_from_extensions(extensions: &axum::http::Extensions) -> Option<Uuid> {
    extensions
        .get::<RequestId>()
        .and_then(|request_id| request_id.header_value().to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
}

pub(crate) fn request_id_from_request<B>(req: &Request<B>) -> Option<Uuid> {
    request_id_from_extensions(req.extensions()).or_else(|| request_id_from_headers(req.headers()))
}

pub(crate) fn request_id_from_parts(parts: &axum::http::request::Parts) -> Option<Uuid> {
    request_id_from_extensions(&parts.extensions)
        .or_else(|| request_id_from_headers(&parts.headers))
}

/// Create the Axum router with all CMP endpoints.
///
/// `api_keys`: if non-empty, enables Bearer token authentication on all API routes.
/// Health endpoints (`/health`, `/readiness`) are always unauthenticated.
pub fn router(engine: Arc<CerememoryEngine>, api_keys: Vec<String>) -> Router {
    router_with_config(
        engine,
        HttpMiddlewareConfig {
            api_keys,
            auth_enabled: true,
            ..Default::default()
        },
    )
}

/// Create the Axum router with full middleware configuration.
pub fn router_with_config(engine: Arc<CerememoryEngine>, config: HttpMiddlewareConfig) -> Router {
    let effective_api_keys = if config.auth_enabled {
        config.api_keys
    } else {
        Vec::new()
    };
    let auth_layer = ApiKeyAuthLayer::new(effective_api_keys);

    // Build CORS layer
    let expose_headers: Vec<axum::http::HeaderName> = vec![
        "x-request-id".parse().unwrap(),
        "retry-after".parse().unwrap(),
    ];
    let cors = if config.cors_origins.is_empty() {
        None
    } else {
        let origins: Vec<axum::http::HeaderValue> = config
            .cors_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        if origins.is_empty() {
            None
        } else if origins
            .iter()
            .any(|origin: &axum::http::HeaderValue| origin.as_bytes() == b"*")
        {
            Some(
                CorsLayer::new()
                    .allow_origin(AllowOrigin::any())
                    .allow_methods(Any)
                    .allow_headers(Any)
                    .expose_headers(expose_headers),
            )
        } else {
            Some(
                CorsLayer::new()
                    .allow_origin(AllowOrigin::list(origins))
                    .allow_methods(Any)
                    .allow_headers(Any)
                    .expose_headers(expose_headers),
            )
        }
    };

    let rate_limit = RateLimitLayer::new(
        config.rate_limit_rps,
        config.rate_limit_burst,
        config.trusted_proxy_cidrs,
    );

    // Public routes — no auth, no rate limit
    let public_routes = Router::new()
        .route("/health", get(health))
        .route("/readiness", get(readiness))
        .with_state(Arc::clone(&engine));

    let metrics_routes = if config.metrics_enabled {
        config.prometheus_handle.map(|handle| {
            Router::new()
                .route(
                    "/metrics",
                    get(move || {
                        let h = handle.clone();
                        async move { h.render() }
                    }),
                )
                .with_state(Arc::clone(&engine))
                .layer(rate_limit.clone())
                .layer(auth_layer.clone())
        })
    } else {
        None
    };

    // API routes — behind auth + middleware stack
    let api_routes = Router::new()
        // Encode
        .route("/v1/encode", post(encode_store))
        .route("/v1/encode/batch", post(encode_batch))
        .route("/v1/encode/{record_id}", patch(encode_update))
        // Recall
        .route("/v1/recall/query", post(recall_query))
        .route("/v1/recall/associate/{record_id}", post(recall_associate))
        .route("/v1/recall/timeline", post(recall_timeline))
        .route("/v1/recall/graph", post(recall_graph))
        // Lifecycle
        .route("/v1/lifecycle/consolidate", post(lifecycle_consolidate))
        .route("/v1/lifecycle/decay-tick", post(lifecycle_decay_tick))
        .route("/v1/lifecycle/mode", put(lifecycle_set_mode))
        .route("/v1/lifecycle/forget", delete(lifecycle_forget))
        // Introspect
        .route("/v1/introspect/stats", get(introspect_stats))
        .route("/v1/introspect/record/{record_id}", get(introspect_record))
        .route(
            "/v1/introspect/decay-forecast",
            post(introspect_decay_forecast),
        )
        .route("/v1/introspect/evolution", get(introspect_evolution))
        .with_state(engine)
        .layer(auth_layer)
        .layer(rate_limit)
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024)); // 2 MB

    // Merge and apply global middleware (metrics, tracing, request-id, CORS)
    let merged = public_routes.merge(api_routes);
    let merged = if let Some(metrics_routes) = metrics_routes {
        merged.merge(metrics_routes)
    } else {
        merged
    };

    let merged = if let Some(cors) = cors {
        merged.layer(cors)
    } else {
        merged
    };

    merged
        .layer(MetricsLayer)
        .layer(TraceLayer::new_for_http())
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
}

// ─── Health handlers ────────────────────────────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn readiness(State(engine): State<AppState>) -> Response {
    match engine.introspect_stats().await {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({"status": "ready"}))).into_response(),
        Err(e) => {
            // Sanitize: only expose error category, not internal details
            let reason = match &e {
                CerememoryError::Storage(_) => "storage unavailable",
                CerememoryError::DecayEngineBusy { .. } => "decay engine busy",
                _ => "engine check failed",
            };
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"status": "not_ready", "reason": reason})),
            )
                .into_response()
        }
    }
}

// ─── Error mapping ───────────────────────────────────────────────────

/// Map CerememoryError to HTTP status + CMPError JSON.
struct AppError {
    error: CerememoryError,
    request_id: Option<Uuid>,
}

impl AppError {
    fn new(error: CerememoryError, request_id: Option<Uuid>) -> Self {
        Self { error, request_id }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self.error {
            CerememoryError::RecordNotFound(_) => StatusCode::NOT_FOUND,
            CerememoryError::StoreInvalid(_) => StatusCode::BAD_REQUEST,
            CerememoryError::ContentTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            CerememoryError::ModalityUnsupported(_) => StatusCode::BAD_REQUEST,
            CerememoryError::WorkingMemoryFull => StatusCode::SERVICE_UNAVAILABLE,
            CerememoryError::DecayEngineBusy { .. } => StatusCode::SERVICE_UNAVAILABLE,
            CerememoryError::ConsolidationInProgress => StatusCode::SERVICE_UNAVAILABLE,
            CerememoryError::ExportFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CerememoryError::ImportConflict(_) => StatusCode::CONFLICT,
            CerememoryError::ForgetUnconfirmed => StatusCode::BAD_REQUEST,
            CerememoryError::VersionMismatch { .. } => StatusCode::BAD_REQUEST,
            CerememoryError::Validation(_) => StatusCode::BAD_REQUEST,
            CerememoryError::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CerememoryError::Serialization(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CerememoryError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CerememoryError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            CerememoryError::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
        };

        // Log server-side errors that return opaque messages to the client
        match &self.error {
            CerememoryError::Storage(msg) => {
                tracing::warn!(error = %msg, "Storage error");
            }
            CerememoryError::Serialization(msg) => {
                tracing::warn!(error = %msg, "Serialization error");
            }
            CerememoryError::ExportFailed(msg) => {
                tracing::warn!(error = %msg, "Export failed");
            }
            CerememoryError::Internal(msg) => {
                tracing::warn!(error = %msg, "Internal error");
            }
            _ => {}
        }

        let mut cmp_error = CMPError::from(&self.error);
        cmp_error.request_id = self.request_id;
        let retry_after = cmp_error.retry_after;
        let mut response = (status, Json(cmp_error)).into_response();
        // Add Retry-After HTTP header for 503 responses
        if let Some(secs) = retry_after {
            if let Ok(val) = axum::http::HeaderValue::from_str(&secs.to_string()) {
                response.headers_mut().insert("retry-after", val);
            }
        }
        response
    }
}

impl From<CerememoryError> for AppError {
    fn from(err: CerememoryError) -> Self {
        Self::new(err, None)
    }
}

// ─── Custom extractors for CMPError envelope on rejection ────────────

/// JSON extractor that converts deserialization errors to CMPError.
struct AppJson<T>(T);

impl<S, T> FromRequest<S> for AppJson<T>
where
    axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
    T: Send,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let request_id = request_id_from_request(&req);
        match axum::Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(AppJson(value)),
            Err(rejection) => Err(AppError::new(
                CerememoryError::Validation(rejection.body_text()),
                request_id,
            )),
        }
    }
}

/// Path extractor that converts parse errors to CMPError.
struct AppPath<T>(T);

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
        let request_id = request_id_from_parts(parts);
        match axum::extract::Path::<T>::from_request_parts(parts, state).await {
            Ok(Path(value)) => Ok(AppPath(value)),
            Err(rejection) => Err(AppError::new(
                CerememoryError::Validation(rejection.body_text()),
                request_id,
            )),
        }
    }
}

struct MaybeRequestId(Option<Uuid>);

impl<S> FromRequestParts<S> for MaybeRequestId
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        Ok(Self(request_id_from_parts(parts)))
    }
}

// ─── Encode handlers ─────────────────────────────────────────────────

async fn encode_store(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<EncodeStoreRequest>,
) -> Result<Json<EncodeStoreResponse>, AppError> {
    let resp = engine
        .encode_store(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn encode_batch(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<EncodeBatchRequest>,
) -> Result<Json<EncodeBatchResponse>, AppError> {
    let resp = engine
        .encode_batch(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn encode_update(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppPath(record_id): AppPath<Uuid>,
    AppJson(mut req): AppJson<EncodeUpdateRequest>,
) -> Result<StatusCode, AppError> {
    req.record_id = record_id;
    engine
        .encode_update(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(StatusCode::NO_CONTENT)
}

// ─── Recall handlers ─────────────────────────────────────────────────

async fn recall_query(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<RecallQueryRequest>,
) -> Result<Json<RecallQueryResponse>, AppError> {
    let resp = engine
        .recall_query(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn recall_associate(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppPath(record_id): AppPath<Uuid>,
    AppJson(mut req): AppJson<RecallAssociateRequest>,
) -> Result<Json<RecallAssociateResponse>, AppError> {
    req.record_id = record_id;
    let resp = engine
        .recall_associate(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn recall_timeline(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<RecallTimelineRequest>,
) -> Result<Json<RecallTimelineResponse>, AppError> {
    let resp = engine
        .recall_timeline(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn recall_graph(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<RecallGraphRequest>,
) -> Result<Json<RecallGraphResponse>, AppError> {
    let resp = engine
        .recall_graph(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

// ─── Lifecycle handlers ──────────────────────────────────────────────

async fn lifecycle_consolidate(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<ConsolidateRequest>,
) -> Result<Json<ConsolidateResponse>, AppError> {
    let resp = engine
        .lifecycle_consolidate(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn lifecycle_decay_tick(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<DecayTickRequest>,
) -> Result<Json<DecayTickResponse>, AppError> {
    let resp = engine
        .lifecycle_decay_tick(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn lifecycle_set_mode(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<SetModeRequest>,
) -> Result<StatusCode, AppError> {
    engine
        .lifecycle_set_mode(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn lifecycle_forget(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<ForgetRequest>,
) -> Result<Json<ForgetResponse>, AppError> {
    let deleted = engine
        .lifecycle_forget(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(ForgetResponse {
        records_deleted: deleted,
    }))
}

// ─── Introspect handlers ─────────────────────────────────────────────

async fn introspect_stats(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
) -> Result<Json<StatsResponse>, AppError> {
    let stats = engine
        .introspect_stats()
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(stats))
}

async fn introspect_decay_forecast(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
    AppJson(req): AppJson<DecayForecastRequest>,
) -> Result<Json<DecayForecastResponse>, AppError> {
    let resp = engine
        .introspect_decay_forecast(req)
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn introspect_evolution(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
) -> Result<Json<EvolutionMetrics>, AppError> {
    let resp = engine
        .introspect_evolution()
        .await
        .map_err(|err| AppError::new(err, request_id))?;
    Ok(Json(resp))
}

async fn introspect_record(
    State(engine): State<AppState>,
    MaybeRequestId(request_id): MaybeRequestId,
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
        .await
        .map_err(|err| AppError::new(err, request_id))?;
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
        router(engine, vec![])
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
        assert!(json["request_id"].is_string());
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
        assert!(json["request_id"].is_string());
    }

    // ─── Health endpoint tests ───

    #[tokio::test]
    async fn health_returns_200() {
        let app = test_app();
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn readiness_returns_200() {
        let app = test_app();
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/readiness")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["status"], "ready");
    }

    #[tokio::test]
    async fn health_bypasses_auth() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router(engine, vec!["secret-key".to_string()]);

        // /health should be accessible without auth
        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // /v1/introspect/stats should require auth
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_with_valid_key_passes() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router(engine, vec!["test-api-key".to_string()]);

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .header("Authorization", "Bearer test-api-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unauthorized_response_includes_request_id() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router(engine, vec!["test-api-key".to_string()]);

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let json = body_json(resp).await;
        assert_eq!(json["code"], "UNAUTHORIZED");
        assert!(json["request_id"].is_string());
    }

    #[tokio::test]
    async fn auth_disabled_ignores_configured_keys() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                api_keys: vec!["secret".to_string()],
                auth_enabled: false,
                ..Default::default()
            },
        );

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cors_disabled_emits_no_cors_headers() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                cors_origins: vec![],
                ..Default::default()
            },
        );

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(!resp.headers().contains_key("access-control-allow-origin"));
    }

    // ─── Metrics tests ───

    #[tokio::test]
    async fn metrics_endpoint_returns_200() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                metrics_enabled: true,
                prometheus_handle: Some(handle),
                ..Default::default()
            },
        );

        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn metrics_endpoint_requires_auth_when_enabled() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                api_keys: vec!["secret".to_string()],
                metrics_enabled: true,
                prometheus_handle: Some(handle),
                ..Default::default()
            },
        );

        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/metrics")
                    .header("Authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn metrics_endpoint_rate_limited_when_enabled() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                api_keys: vec!["secret".to_string()],
                metrics_enabled: true,
                rate_limit_rps: 1,
                rate_limit_burst: 1,
                prometheus_handle: Some(handle),
                ..Default::default()
            },
        );

        let req = || {
            axum::http::Request::builder()
                .uri("/metrics")
                .header("Authorization", "Bearer secret")
                .body(Body::empty())
                .unwrap()
        };

        let first = app.clone().oneshot(req()).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);

        let second = app.oneshot(req()).await.unwrap();
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn metrics_not_available_when_handle_absent() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                metrics_enabled: true,
                prometheus_handle: None,
                ..Default::default()
            },
        );

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Without handle, /metrics route doesn't exist → 404
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ─── Rate limiting tests ───

    #[tokio::test]
    async fn rate_limit_allows_within_burst() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                rate_limit_rps: 10,
                rate_limit_burst: 10,
                ..Default::default()
            },
        );

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rate_limit_returns_429_when_exceeded() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        // 1 req/sec, burst of 1 → second request should be 429
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                rate_limit_rps: 1,
                rate_limit_burst: 1,
                ..Default::default()
            },
        );

        // First request — should pass
        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request — should be rate limited
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn rate_limit_includes_retry_after_header() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                rate_limit_rps: 1,
                rate_limit_burst: 1,
                ..Default::default()
            },
        );

        // Exhaust the burst
        let _ = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Second request should have Retry-After
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(
            resp.headers().contains_key("retry-after"),
            "429 response must include Retry-After header"
        );
        let json = body_json(resp).await;
        assert_eq!(json["code"], "RATE_LIMITED");
        assert!(json["retry_after"].is_number());
        assert!(json["request_id"].is_string());
    }

    #[tokio::test]
    async fn rate_limit_does_not_apply_to_health() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        // Very strict rate limit
        let app = router_with_config(
            engine,
            HttpMiddlewareConfig {
                rate_limit_rps: 1,
                rate_limit_burst: 1,
                ..Default::default()
            },
        );

        // Exhaust the burst on an API route
        let _ = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/introspect/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // /health should still work (outside rate limit layer)
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ─── Existing tests ───

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
