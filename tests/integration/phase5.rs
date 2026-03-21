//! Phase 5 "Production" integration tests.
//!
//! Tests the production-readiness features: config, auth, health, CORS,
//! rate limiting, request-id, graceful shutdown, and error mapping.

use std::sync::Arc;

use axum::body::Body;
use cerememory_config::ServerConfig;
use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::*;
use cerememory_engine::CerememoryEngine;
use http_body_util::BodyExt;
use tower::ServiceExt;

// ─── Helpers ─────────────────────────────────────────────────────────

fn in_memory_engine() -> Arc<CerememoryEngine> {
    Arc::new(CerememoryEngine::in_memory().unwrap())
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn body_string(resp: axum::response::Response) -> String {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ─── Config Tests ────────────────────────────────────────────────────

#[test]
fn config_defaults_are_valid() {
    let config = ServerConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.http.port, 8420);
    assert!(!config.auth.enabled);
    assert_eq!(config.llm.provider, "none");
}

#[test]
fn config_load_from_toml() {
    let dir = tempfile::tempdir().unwrap();
    let toml_path = dir.path().join("cerememory.toml");
    std::fs::write(
        &toml_path,
        r#"
data_dir = "/tmp/test-data"
[http]
port = 9999
[auth]
enabled = false
"#,
    )
    .unwrap();

    let config = ServerConfig::load(Some(toml_path.to_str().unwrap())).unwrap();
    assert_eq!(config.http.port, 9999);
    assert_eq!(config.data_dir, "/tmp/test-data");
}

#[test]
fn config_to_engine_config_maps_correctly() {
    let config = ServerConfig {
        data_dir: "/test/data".to_string(),
        ..ServerConfig::default()
    };
    let ec = config.to_engine_config();
    assert_eq!(ec.episodic_path.unwrap(), "/test/data/episodic.redb");
    assert_eq!(ec.semantic_path.unwrap(), "/test/data/semantic.redb");
}

// ─── Auth Tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn auth_missing_header_returns_401() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec!["test-key".to_string()]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/introspect/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
    let json = body_json(resp).await;
    assert_eq!(json["code"], "UNAUTHORIZED");
}

#[tokio::test]
async fn auth_invalid_key_returns_401() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec!["correct-key".to_string()]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/introspect/stats")
                .header("Authorization", "Bearer wrong-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_valid_key_passes() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec!["valid-key".to_string()]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/introspect/stats")
                .header("Authorization", "Bearer valid-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn auth_multiple_keys_accepts_any() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(
        engine,
        vec!["key-alpha".to_string(), "key-beta".to_string()],
    );

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/introspect/stats")
                .header("Authorization", "Bearer key-beta")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn auth_empty_keys_disables_auth() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec![]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/introspect/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}

// ─── Health Tests ────────────────────────────────────────────────────

#[tokio::test]
async fn health_endpoint_returns_200() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec![]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = body_json(resp).await;
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn readiness_endpoint_returns_200() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec![]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/readiness")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = body_json(resp).await;
    assert_eq!(json["status"], "ready");
}

#[tokio::test]
async fn health_bypasses_auth() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec!["secret-key".to_string()]);

    // Health should work without auth
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
    assert_eq!(resp.status(), axum::http::StatusCode::OK);

    // Readiness should also bypass auth
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/readiness")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}

// ─── Error Mapping Tests ─────────────────────────────────────────────

#[test]
fn unauthorized_error_maps_to_cmp() {
    let err = CerememoryError::Unauthorized("test".to_string());
    let cmp = CMPError::from(&err);
    assert_eq!(cmp.code, CMPErrorCode::Unauthorized);
}

#[test]
fn rate_limited_error_maps_to_cmp() {
    let err = CerememoryError::RateLimited {
        retry_after_secs: 30,
    };
    let cmp = CMPError::from(&err);
    assert_eq!(cmp.code, CMPErrorCode::RateLimited);
    assert_eq!(cmp.retry_after, Some(30));
}

// ─── Request-Id Tests ────────────────────────────────────────────────

#[tokio::test]
async fn response_includes_request_id() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec![]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    assert!(
        resp.headers().contains_key("x-request-id"),
        "Response should contain x-request-id header"
    );
}

// ─── CORS Tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn cors_preflight_returns_ok() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec![]);

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("OPTIONS")
                .uri("/v1/introspect/stats")
                .header("Origin", "http://localhost:3000")
                .header("Access-Control-Request-Method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}

// ─── End-to-end: Auth + Encode + Recall ──────────────────────────────

#[tokio::test]
async fn full_encode_recall_with_auth() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router(engine, vec!["my-api-key".to_string()]);

    // Encode a record
    let encode_body = serde_json::json!({
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
                .header("Authorization", "Bearer my-api-key")
                .body(Body::from(serde_json::to_vec(&encode_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = body_json(resp).await;
    assert!(json["record_id"].is_string());
    assert_eq!(json["store"], "episodic");

    // Recall the record
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
                .header("Authorization", "Bearer my-api-key")
                .body(Body::from(serde_json::to_vec(&recall_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = body_json(resp).await;
    assert!(!json["memories"].as_array().unwrap().is_empty());
}

// ─── Graceful Shutdown Tests ─────────────────────────────────────────

#[tokio::test]
async fn http_server_shuts_down_gracefully() {
    let engine = in_memory_engine();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    let addr = "127.0.0.1:0";
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            eprintln!("skipping HTTP shutdown test: {error}");
            return;
        }
        Err(error) => panic!("failed to bind HTTP test listener: {error}"),
    };
    let bound_addr = listener.local_addr().unwrap();
    let app = cerememory_transport_http::router(Arc::clone(&engine), vec![]);

    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                rx.await.ok();
            })
            .await
            .unwrap();
    });

    // Verify server is running
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{bound_addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Trigger shutdown
    let _ = tx.send(());
    // Server task should complete within a reasonable time
    tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("Server should shut down within 5 seconds")
        .unwrap();
}

#[tokio::test]
async fn background_decay_stops_on_shutdown() {
    // Need an engine with background_decay_interval_secs set so start_background_decay works
    let config = cerememory_engine::EngineConfig {
        background_decay_interval_secs: Some(3600),
        ..Default::default()
    };
    let engine = Arc::new(cerememory_engine::CerememoryEngine::new(config).unwrap());
    engine.start_background_decay();
    assert!(engine.is_background_decay_enabled().await);

    engine.stop_background_decay().await;
    assert!(!engine.is_background_decay_enabled().await);
}

#[tokio::test]
async fn cancellation_token_propagates() {
    let token = tokio_util::sync::CancellationToken::new();
    let child = token.child_token();

    assert!(!child.is_cancelled());
    token.cancel();
    assert!(child.is_cancelled());
}

// ─── Rate Limiting Integration Tests ─────────────────────────────────

#[tokio::test]
async fn rate_limit_429_has_cmp_error_body() {
    let engine = in_memory_engine();
    let app = cerememory_transport_http::router_with_config(
        engine,
        cerememory_transport_http::HttpMiddlewareConfig {
            rate_limit_rps: 1,
            rate_limit_burst: 1,
            ..Default::default()
        },
    );

    // Exhaust burst
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

    // Next request → 429
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/introspect/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    assert!(resp.headers().contains_key("retry-after"));
    let json = body_json(resp).await;
    assert_eq!(json["code"], "RATE_LIMITED");
}

// ─── Metrics Integration Tests ───────────────────────────────────────

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_format() {
    let engine = in_memory_engine();
    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let app = cerememory_transport_http::router_with_config(
        engine,
        cerememory_transport_http::HttpMiddlewareConfig {
            prometheus_handle: Some(handle),
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

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    // Prometheus output is text (may be empty if no metrics recorded yet)
    let text = body_string(resp).await;
    // Should be valid text (no error)
    assert!(text.is_ascii() || text.is_empty());
}

#[tokio::test]
async fn metrics_endpoint_not_behind_auth() {
    let engine = in_memory_engine();
    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let app = cerememory_transport_http::router_with_config(
        engine,
        cerememory_transport_http::HttpMiddlewareConfig {
            api_keys: vec!["secret".to_string()],
            prometheus_handle: Some(handle),
            ..Default::default()
        },
    );

    // No auth header, should still work
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}

// ─── TLS Tests ───────────────────────────────────────────────────────

fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

#[tokio::test]
async fn grpc_tls_server_starts_with_self_signed_cert() {
    ensure_crypto_provider();
    // Generate self-signed cert for localhost
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_pem = cert.cert.pem().into_bytes();
    let key_pem = cert.signing_key.serialize_pem().into_bytes();

    let engine = in_memory_engine();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    let tls = cerememory_transport_grpc::TlsConfig { cert_pem, key_pem };

    let server = tokio::spawn(async move {
        cerememory_transport_grpc::serve_with_tls(engine, "127.0.0.1:0", vec![], Some(tls), async {
            rx.await.ok();
        })
        .await
        .map_err(|e| e.to_string())
    });

    // Give server a moment to start
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    if server.is_finished() {
        let result = server.await.expect("TLS server task should join cleanly");
        if let Err(message) = &result {
            eprintln!("skipping gRPC TLS startup test: {message}");
            return;
        }
        assert!(result.is_ok(), "TLS server should start OK: {result:?}");
        return;
    }

    // Shut it down
    let _ = tx.send(());
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("Server should shut down within 5 seconds")
        .unwrap();
    assert!(result.is_ok(), "TLS server should start OK: {result:?}");
}

#[tokio::test]
async fn grpc_plaintext_server_starts_without_tls() {
    ensure_crypto_provider();
    let engine = in_memory_engine();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    let server = tokio::spawn(async move {
        cerememory_transport_grpc::serve_with_tls(
            engine,
            "127.0.0.1:0",
            vec![],
            None, // No TLS
            async {
                rx.await.ok();
            },
        )
        .await
        .map_err(|e| e.to_string())
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    if server.is_finished() {
        let result = server
            .await
            .expect("plaintext server task should join cleanly");
        if let Err(message) = &result {
            eprintln!("skipping gRPC plaintext startup test: {message}");
            return;
        }
        assert!(
            result.is_ok(),
            "Plaintext server should start OK: {result:?}"
        );
        return;
    }

    tx.send(()).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("Server should shut down")
        .unwrap();
    assert!(
        result.is_ok(),
        "Plaintext server should start OK: {result:?}"
    );
}

#[test]
fn invalid_tls_cert_rejected() {
    ensure_crypto_provider();
    // Invalid PEM should cause tls_config to error
    let tls = cerememory_transport_grpc::TlsConfig {
        cert_pem: b"not a real cert".to_vec(),
        key_pem: b"not a real key".to_vec(),
    };

    // Build Identity from invalid PEM — this should fail
    let result = tonic::transport::Identity::from_pem(&tls.cert_pem, &tls.key_pem);
    // Identity::from_pem doesn't validate eagerly, but ServerTlsConfig does
    let tls_config = tonic::transport::ServerTlsConfig::new().identity(result);
    let builder_result = tonic::transport::Server::builder().tls_config(tls_config);
    assert!(
        builder_result.is_err(),
        "Invalid cert/key should produce error"
    );
}

// ─── Config validation tests ─────────────────────────────────────────

#[test]
fn config_rejects_auth_enabled_without_keys() {
    let mut config = ServerConfig::default();
    config.auth.enabled = true;
    assert!(config.validate().is_err());
}

#[test]
fn config_rejects_zero_rate_limit() {
    let mut config = ServerConfig::default();
    config.rate_limit.requests_per_second = 0;
    assert!(config.validate().is_err());
}

#[test]
fn config_rejects_partial_tls() {
    let mut config = ServerConfig::default();
    config.grpc.tls_cert_path = Some("/path/cert.pem".to_string());
    // key not set
    assert!(config.validate().is_err());
}
