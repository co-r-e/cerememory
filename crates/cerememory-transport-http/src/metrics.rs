//! Prometheus metrics middleware and endpoint.
//!
//! Records HTTP request duration and count per method/path/status.
//! Path labels use the route template (e.g., `/v1/encode/:record_id`)
//! to avoid unbounded cardinality from dynamic path segments.

use axum::{body::Body, extract::MatchedPath, http::Request, response::Response};
use metrics::{counter, histogram};
use std::time::Instant;
use tower::{Layer, Service};

/// Tower layer that records HTTP request metrics.
#[derive(Clone, Default)]
pub struct MetricsLayer;

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsService { inner }
    }
}

/// Metrics recording service.
#[derive(Clone)]
pub struct MetricsService<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for MetricsService<S>
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
        let method = req.method().as_str().to_owned();
        // Use the matched route template to avoid UUID cardinality explosion.
        // Falls back to a fixed string if no route was matched.
        let path = req
            .extensions()
            .get::<MatchedPath>()
            .map(|m| m.as_str().to_owned())
            .unwrap_or_else(|| "unknown".to_owned());
        let start = Instant::now();

        let mut inner = self.inner.clone();
        Box::pin(async move {
            let resp = inner.call(req).await?;
            let status = resp.status().as_u16().to_string();
            let duration = start.elapsed().as_secs_f64();

            counter!("cerememory_http_requests_total", "method" => method.clone(), "path" => path.clone(), "status" => status.clone()).increment(1);
            histogram!("cerememory_http_request_duration_seconds", "method" => method, "path" => path, "status" => status).record(duration);

            Ok(resp)
        })
    }
}

/// Install the global Prometheus metrics recorder.
/// Returns a handle that can render metrics text.
/// Call this once at startup before creating the router.
pub fn install_prometheus_recorder() -> metrics_exporter_prometheus::PrometheusHandle {
    metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("Failed to install Prometheus recorder")
}
