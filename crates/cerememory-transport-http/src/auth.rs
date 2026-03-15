//! API key authentication middleware for HTTP transport.
//!
//! Uses constant-time comparison via shared `cerememory_core::auth::validate_api_key`.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;
use tower::{Layer, Service};

use cerememory_core::auth::validate_api_key;
use cerememory_core::protocol::{CMPError, CMPErrorCode};

/// Shared state for API key validation.
#[derive(Clone)]
struct AuthState {
    api_keys: Vec<Vec<u8>>,
}

/// Tower layer that wraps services with API key authentication.
#[derive(Clone)]
pub struct ApiKeyAuthLayer {
    state: Arc<AuthState>,
}

impl ApiKeyAuthLayer {
    /// Create a new auth layer with the given API keys.
    ///
    /// If `api_keys` is empty, authentication is disabled (all requests pass through).
    pub fn new(api_keys: Vec<String>) -> Self {
        Self {
            state: Arc::new(AuthState {
                api_keys: api_keys.into_iter().map(|k| k.into_bytes()).collect(),
            }),
        }
    }

    /// Returns true if authentication is effectively disabled (no keys configured).
    pub fn is_disabled(&self) -> bool {
        self.state.api_keys.is_empty()
    }
}

impl<S> Layer<S> for ApiKeyAuthLayer {
    type Service = ApiKeyAuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ApiKeyAuthService {
            inner,
            state: Arc::clone(&self.state),
        }
    }
}

/// The middleware service that checks Bearer tokens.
#[derive(Clone)]
pub struct ApiKeyAuthService<S> {
    inner: S,
    state: Arc<AuthState>,
}

impl<S> Service<Request<Body>> for ApiKeyAuthService<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response, S::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        // If no keys configured, skip auth
        if self.state.api_keys.is_empty() {
            let mut inner = self.inner.clone();
            return Box::pin(async move { inner.call(req).await });
        }

        // Extract Bearer token — work with &[u8] directly, no allocation
        let token_bytes = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|t| t.as_bytes());

        match token_bytes {
            Some(token) if validate_api_key(token, &self.state.api_keys) => {
                let mut inner = self.inner.clone();
                Box::pin(async move { inner.call(req).await })
            }
            Some(_) => Box::pin(async { Ok(unauthorized_response("Invalid API key")) }),
            None => {
                Box::pin(async { Ok(unauthorized_response("Missing Authorization header")) })
            }
        }
    }
}

fn unauthorized_response(message: &str) -> Response {
    let cmp_error = CMPError::new(CMPErrorCode::Unauthorized, message);
    (StatusCode::UNAUTHORIZED, Json(cmp_error)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use axum::Router;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_app(keys: Vec<String>) -> Router {
        let app = Router::new().route("/test", get(|| async { "ok" }));
        app.layer(ApiKeyAuthLayer::new(keys))
    }

    async fn body_string(resp: Response) -> String {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn no_auth_header_returns_401() {
        let app = test_app(vec!["secret-key".to_string()]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = body_string(resp).await;
        assert!(body.contains("UNAUTHORIZED"));
    }

    #[tokio::test]
    async fn invalid_token_returns_401() {
        let app = test_app(vec!["secret-key".to_string()]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer wrong-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn valid_token_passes_through() {
        let app = test_app(vec!["secret-key".to_string()]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;
        assert_eq!(body, "ok");
    }

    #[tokio::test]
    async fn multiple_keys_any_valid() {
        let app = test_app(vec!["key-1".to_string(), "key-2".to_string()]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer key-2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn empty_keys_disables_auth() {
        let app = test_app(vec![]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn bearer_prefix_required() {
        let app = test_app(vec!["secret-key".to_string()]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("Authorization", "secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
