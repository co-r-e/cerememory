//! Governor-based per-IP rate limiting middleware.
//!
//! Each unique client IP gets its own token bucket. IP is extracted from
//! `X-Forwarded-For` header (first hop) or the peer socket address.
//! A background task periodically shrinks the DashMap to evict stale entries.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use governor::{clock::DefaultClock, state::keyed::DefaultKeyedStateStore, Quota, RateLimiter};
use std::net::IpAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use tower::{Layer, Service};

use cerememory_core::protocol::{CMPError, CMPErrorCode};

type KeyedLimiter = RateLimiter<IpAddr, DefaultKeyedStateStore<IpAddr>, DefaultClock>;

struct CleanupGuard {
    cancel: tokio_util::sync::CancellationToken,
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Tower layer for per-IP rate limiting.
#[derive(Clone)]
pub struct RateLimitLayer {
    limiter: Arc<KeyedLimiter>,
    cleanup: Arc<CleanupGuard>,
}

impl RateLimitLayer {
    /// Create a per-IP rate limiter with `per_second` requests/sec and `burst` capacity.
    ///
    /// Spawns a background task that shrinks the internal DashMap every 60 seconds
    /// to evict stale entries and prevent unbounded memory growth.
    pub fn new(per_second: u64, burst: u32) -> Self {
        let quota =
            Quota::per_second(NonZeroU32::new(per_second as u32).unwrap_or(NonZeroU32::MIN))
                .allow_burst(NonZeroU32::new(burst).unwrap_or(NonZeroU32::MIN));
        let limiter = Arc::new(RateLimiter::keyed(quota));

        let cancel = tokio_util::sync::CancellationToken::new();
        let cleanup = Arc::new(CleanupGuard {
            cancel: cancel.clone(),
        });

        // Periodic cleanup of stale IP entries
        let limiter_bg = Arc::clone(&limiter);
        let cancel_bg = cancel.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        limiter_bg.retain_recent();
                    }
                    _ = cancel_bg.cancelled() => {
                        break;
                    }
                }
            }
        });

        Self { limiter, cleanup }
    }
}

impl<S> Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            limiter: Arc::clone(&self.limiter),
            _cleanup: Arc::clone(&self.cleanup),
        }
    }
}

/// The per-IP rate limiting service.
#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    limiter: Arc<KeyedLimiter>,
    _cleanup: Arc<CleanupGuard>,
}

/// Extract client IP from the request.
///
/// Priority: `X-Forwarded-For` first entry → `X-Real-IP` → fallback `127.0.0.1`.
fn extract_client_ip(req: &Request<Body>) -> IpAddr {
    // Try X-Forwarded-For (first IP in the chain)
    if let Some(xff) = req.headers().get("x-forwarded-for") {
        if let Ok(value) = xff.to_str() {
            if let Some(first) = value.split(',').next() {
                if let Ok(ip) = first.trim().parse::<IpAddr>() {
                    return ip;
                }
            }
        }
    }

    // Try X-Real-IP
    if let Some(xri) = req.headers().get("x-real-ip") {
        if let Ok(value) = xri.to_str() {
            if let Ok(ip) = value.trim().parse::<IpAddr>() {
                return ip;
            }
        }
    }

    // Fallback — treat as single client
    IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
}

impl<S> Service<Request<Body>> for RateLimitService<S>
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
        let ip = extract_client_ip(&req);
        let request_id = crate::request_id_from_request(&req);

        match self.limiter.check_key(&ip) {
            Ok(_) => {
                let mut inner = self.inner.clone();
                Box::pin(async move { inner.call(req).await })
            }
            Err(not_until) => {
                let clock = governor::clock::DefaultClock::default();
                let retry_after = not_until.wait_time_from(governor::clock::Clock::now(&clock));
                let secs = retry_after.as_secs() + 1; // round up
                Box::pin(async move {
                    let mut cmp_error =
                        CMPError::new(CMPErrorCode::RateLimited, "Rate limit exceeded");
                    cmp_error.retry_after = Some(secs as u32);
                    cmp_error.request_id = request_id;
                    let mut resp = (StatusCode::TOO_MANY_REQUESTS, Json(cmp_error)).into_response();
                    if let Ok(val) = axum::http::HeaderValue::from_str(&secs.to_string()) {
                        resp.headers_mut().insert("retry-after", val);
                    }
                    Ok(resp)
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use axum::Router;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_app(rps: u64, burst: u32) -> Router {
        Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(RateLimitLayer::new(rps, burst))
    }

    #[tokio::test]
    async fn allows_within_burst() {
        let app = test_app(10, 10);
        let resp = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rejects_when_burst_exceeded() {
        let app = test_app(1, 1);

        // Exhaust burst
        let _ = app
            .clone()
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        // Should be 429
        let resp = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn different_ips_have_separate_buckets() {
        let app = test_app(1, 1);

        // IP-A uses its bucket
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Forwarded-For", "10.0.0.1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // IP-A is now limited
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Forwarded-For", "10.0.0.1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        // IP-B should still be allowed (separate bucket)
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Forwarded-For", "10.0.0.2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn x_real_ip_header_used() {
        let app = test_app(1, 1);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Real-IP", "192.168.1.1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Same IP → limited
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Real-IP", "192.168.1.1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn retry_after_header_present() {
        let app = test_app(1, 1);
        let _ = app
            .clone()
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let resp = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(resp.headers().contains_key("retry-after"));

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["code"], "RATE_LIMITED");
        assert!(json["retry_after"].is_number());
    }
}
