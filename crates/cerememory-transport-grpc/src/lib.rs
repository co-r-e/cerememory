//! gRPC transport binding for the Cerememory Protocol (CMP).
//!
//! Uses JSON-serialized payloads over gRPC for protocol type flexibility.
//! This avoids duplicating the entire CMP type system in protobuf while
//! retaining gRPC's streaming, flow-control, and HTTP/2 benefits.

use std::sync::Arc;

use cerememory_engine::CerememoryEngine;
use tonic::transport::{Identity, ServerTlsConfig};

pub mod auth;
pub mod proto {
    tonic::include_proto!("cerememory.v1");
}

mod service;

pub use auth::GrpcAuthInterceptor;
pub use proto::cerememory_service_server::CerememoryServiceServer;
pub use service::CerememoryGrpcService;

/// TLS configuration for gRPC server.
pub struct TlsConfig {
    pub cert_pem: Vec<u8>,
    pub key_pem: Vec<u8>,
}

/// Start the gRPC server on the given address (e.g. `"0.0.0.0:8421"`).
///
/// `api_keys`: if non-empty, enables Bearer token auth via interceptor.
/// `tls`: if Some, enables TLS with the given cert/key.
/// `shutdown_signal`: when this future completes, the server drains and stops.
/// Includes tonic-health service for gRPC health checking protocol.
pub async fn serve(
    engine: Arc<CerememoryEngine>,
    addr: &str,
    api_keys: Vec<String>,
    shutdown_signal: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error>> {
    serve_with_tls(engine, addr, api_keys, None, shutdown_signal).await
}

/// Start the gRPC server with optional TLS.
pub async fn serve_with_tls(
    engine: Arc<CerememoryEngine>,
    addr: &str,
    api_keys: Vec<String>,
    tls: Option<TlsConfig>,
    shutdown_signal: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.parse()?;
    let svc = CerememoryGrpcService::new(engine);
    let interceptor = GrpcAuthInterceptor::new(api_keys);

    // gRPC health service
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<CerememoryServiceServer<CerememoryGrpcService>>()
        .await;

    let mut builder = tonic::transport::Server::builder();

    if let Some(tls_config) = tls {
        let identity = Identity::from_pem(&tls_config.cert_pem, &tls_config.key_pem);
        builder = builder.tls_config(ServerTlsConfig::new().identity(identity))?;
        tracing::info!(%addr, "Cerememory gRPC server listening (TLS)");
    } else {
        tracing::warn!(%addr, "Cerememory gRPC server listening (plaintext, no TLS)");
    }

    builder
        .add_service(health_service)
        .add_service(CerememoryServiceServer::with_interceptor(svc, interceptor))
        .serve_with_shutdown(addr, shutdown_signal)
        .await?;

    Ok(())
}
