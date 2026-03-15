//! API key authentication interceptor for gRPC transport.
//!
//! Uses shared `cerememory_core::auth::validate_api_key` for constant-time comparison.

use std::sync::Arc;

use cerememory_core::auth::validate_api_key;
use tonic::{Request, Status};

/// Shared authentication state for the gRPC interceptor.
#[derive(Clone)]
pub struct GrpcAuthInterceptor {
    api_keys: Arc<Vec<Vec<u8>>>,
}

impl GrpcAuthInterceptor {
    /// Create a new interceptor. If `api_keys` is empty, all requests pass through.
    pub fn new(api_keys: Vec<String>) -> Self {
        Self {
            api_keys: Arc::new(api_keys.into_iter().map(|k| k.into_bytes()).collect()),
        }
    }

    /// Validate the request's authorization metadata.
    #[allow(clippy::result_large_err)]
    pub fn check(&self, req: Request<()>) -> Result<Request<()>, Status> {
        if self.api_keys.is_empty() {
            return Ok(req);
        }

        let token = req
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        match token {
            Some(token_str) if validate_api_key(token_str.as_bytes(), &self.api_keys) => Ok(req),
            Some(_) => Err(Status::unauthenticated("Invalid API key")),
            None => Err(Status::unauthenticated("Missing authorization metadata")),
        }
    }
}

impl tonic::service::Interceptor for GrpcAuthInterceptor {
    fn call(&mut self, req: Request<()>) -> Result<Request<()>, Status> {
        self.check(req)
    }
}
