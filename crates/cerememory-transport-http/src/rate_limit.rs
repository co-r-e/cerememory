//! Governor-based per-IP rate limiting middleware.
//!
//! Each unique client IP gets its own token bucket. IP is extracted from
//! `X-Forwarded-For` header (first hop) or the peer socket address.
//! A background task periodically shrinks the DashMap to evict stale entries.

use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use governor::{clock::DefaultClock, state::keyed::DefaultKeyedStateStore, Quota, RateLimiter};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::num::NonZeroU32;
use std::sync::Arc;
use tower::{Layer, Service};

use cerememory_core::protocol::{CMPError, CMPErrorCode};

type KeyedLimiter = RateLimiter<IpAddr, DefaultKeyedStateStore<IpAddr>, DefaultClock>;

#[derive(Clone, Debug)]
enum TrustedProxyCidr {
    V4 { network: u32, prefix: u8 },
    V6 { network: u128, prefix: u8 },
}

impl TrustedProxyCidr {
    fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        let (addr, prefix) = value.split_once('/')?;
        let prefix: u8 = prefix.parse().ok()?;
        match addr.parse::<IpAddr>().ok()? {
            IpAddr::V4(ip) if prefix <= 32 => Some(Self::V4 {
                network: u32::from(ip) & ipv4_mask(prefix),
                prefix,
            }),
            IpAddr::V6(ip) if prefix <= 128 => Some(Self::V6 {
                network: u128::from(ip) & ipv6_mask(prefix),
                prefix,
            }),
            _ => None,
        }
    }

    fn contains(&self, ip: IpAddr) -> bool {
        match (self, ip) {
            (Self::V4 { network, prefix }, IpAddr::V4(ip)) => {
                (u32::from(ip) & ipv4_mask(*prefix)) == *network
            }
            (Self::V6 { network, prefix }, IpAddr::V6(ip)) => {
                (u128::from(ip) & ipv6_mask(*prefix)) == *network
            }
            _ => false,
        }
    }
}

fn ipv4_mask(prefix: u8) -> u32 {
    if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix))
    }
}

fn ipv6_mask(prefix: u8) -> u128 {
    if prefix == 0 {
        0
    } else {
        u128::MAX << (128 - u32::from(prefix))
    }
}

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
    trusted_proxy_cidrs: Arc<Vec<TrustedProxyCidr>>,
}

impl RateLimitLayer {
    /// Create a per-IP rate limiter with `per_second` requests/sec and `burst` capacity.
    ///
    /// Spawns a background task that shrinks the internal DashMap every 60 seconds
    /// to evict stale entries and prevent unbounded memory growth.
    pub fn new(per_second: u64, burst: u32, trusted_proxy_cidrs: Vec<String>) -> Self {
        let clamped = per_second.min(u32::MAX as u64) as u32;
        let quota = Quota::per_second(NonZeroU32::new(clamped).unwrap_or(NonZeroU32::MIN))
            .allow_burst(NonZeroU32::new(burst).unwrap_or(NonZeroU32::MIN));
        let limiter = Arc::new(RateLimiter::keyed(quota));
        let trusted_proxy_cidrs = Arc::new(
            trusted_proxy_cidrs
                .iter()
                .filter_map(|cidr| TrustedProxyCidr::parse(cidr))
                .collect(),
        );

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

        Self {
            limiter,
            cleanup,
            trusted_proxy_cidrs,
        }
    }
}

impl<S> Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            limiter: Arc::clone(&self.limiter),
            _cleanup: Arc::clone(&self.cleanup),
            trusted_proxy_cidrs: Arc::clone(&self.trusted_proxy_cidrs),
        }
    }
}

/// The per-IP rate limiting service.
#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    limiter: Arc<KeyedLimiter>,
    _cleanup: Arc<CleanupGuard>,
    trusted_proxy_cidrs: Arc<Vec<TrustedProxyCidr>>,
}

/// Extract client IP from the request.
///
/// Priority: trusted proxy `X-Forwarded-For` first entry → trusted proxy `X-Real-IP`
/// → peer socket address → fallback `127.0.0.1`.
fn extract_client_ip(req: &Request<Body>, trusted_proxy_cidrs: &[TrustedProxyCidr]) -> IpAddr {
    let peer_addr = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0);
    let peer_ip = peer_addr.map(|addr| addr.ip());
    let peer_is_trusted = peer_ip
        .map(|ip| trusted_proxy_cidrs.iter().any(|cidr| cidr.contains(ip)))
        .unwrap_or(false);

    let should_trust_forwarded = peer_addr.is_some() && peer_is_trusted;

    if should_trust_forwarded {
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
    }

    peer_ip.unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
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
        let ip = extract_client_ip(&req, &self.trusted_proxy_cidrs);
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
            .layer(RateLimitLayer::new(rps, burst, vec![]))
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
    async fn different_peer_ips_have_separate_buckets() {
        let app = test_app(1, 1);

        // Peer-A uses its bucket
        let mut req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        req.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([10, 0, 0, 1], 12345))));
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Peer-A is now limited
        let mut req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        req.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([10, 0, 0, 1], 12346))));
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        // Peer-B should still be allowed (separate bucket)
        let mut req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        req.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([10, 0, 0, 2], 12347))));
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn untrusted_forwarded_headers_are_ignored_without_connect_info() {
        let app = test_app(1, 1);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Forwarded-For", "192.168.1.1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Different forwarded IP is ignored; same fallback bucket is limited.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Forwarded-For", "192.168.1.2")
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

    #[tokio::test]
    async fn trusted_proxy_uses_forwarded_client_ip() {
        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(RateLimitLayer::new(1, 1, vec!["10.0.0.0/8".to_string()]));

        let mut req1 = Request::builder()
            .uri("/test")
            .header("X-Forwarded-For", "192.168.1.10")
            .body(Body::empty())
            .unwrap();
        req1.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([10, 1, 2, 3], 12345))));
        let resp = app.clone().oneshot(req1).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut req2 = Request::builder()
            .uri("/test")
            .header("X-Forwarded-For", "192.168.1.11")
            .body(Body::empty())
            .unwrap();
        req2.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([10, 1, 2, 3], 12346))));
        let resp = app.oneshot(req2).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn trusted_proxy_uses_x_real_ip() {
        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(RateLimitLayer::new(1, 1, vec!["10.0.0.0/8".to_string()]));

        let mut req1 = Request::builder()
            .uri("/test")
            .header("X-Real-IP", "192.168.1.10")
            .body(Body::empty())
            .unwrap();
        req1.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([10, 1, 2, 3], 12345))));
        let resp = app.clone().oneshot(req1).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut req2 = Request::builder()
            .uri("/test")
            .header("X-Real-IP", "192.168.1.11")
            .body(Body::empty())
            .unwrap();
        req2.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([10, 1, 2, 3], 12346))));
        let resp = app.oneshot(req2).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
