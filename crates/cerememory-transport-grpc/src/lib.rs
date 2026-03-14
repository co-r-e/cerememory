//! gRPC transport binding for the Cerememory Protocol (CMP).
//!
//! Uses JSON-serialized payloads over gRPC for protocol type flexibility.
//! This avoids duplicating the entire CMP type system in protobuf while
//! retaining gRPC's streaming, flow-control, and HTTP/2 benefits.
//!
//! | RPC               | Category    | Streaming       |
//! |--------------------|------------|-----------------|
//! | EncodeStore        | Encode     | unary           |
//! | EncodeBatch        | Encode     | unary           |
//! | EncodeUpdate       | Encode     | unary           |
//! | RecallQuery        | Recall     | unary           |
//! | RecallAssociate    | Recall     | unary           |
//! | Consolidate        | Lifecycle  | unary           |
//! | DecayTick          | Lifecycle  | unary           |
//! | SetMode            | Lifecycle  | unary           |
//! | Forget             | Lifecycle  | unary           |
//! | Export             | Lifecycle  | server-stream   |
//! | Import             | Lifecycle  | client-stream   |
//! | Stats              | Introspect | unary           |
//! | IntrospectRecord   | Introspect | unary           |

use std::sync::Arc;

use cerememory_engine::CerememoryEngine;

pub mod proto {
    tonic::include_proto!("cerememory.v1");
}

mod service;

pub use proto::cerememory_service_server::CerememoryServiceServer;
pub use service::CerememoryGrpcService;

/// Start the gRPC server on the given address (e.g. `"0.0.0.0:8421"`).
pub async fn serve(
    engine: Arc<CerememoryEngine>,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.parse()?;
    let svc = CerememoryGrpcService::new(engine);

    tracing::info!(%addr, "Cerememory gRPC server listening");
    tonic::transport::Server::builder()
        .add_service(CerememoryServiceServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
