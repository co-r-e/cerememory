use std::time::Duration;

use cerememory_core::protocol::{EncodeBatchStoreRawRequest, EncodeBatchStoreRawResponse};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde_json::json;
use uuid::Uuid;

use crate::redact::redact_string;
use crate::{RecorderConfig, RecorderError};

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct SendError {
    pub message: String,
    pub status: Option<u16>,
    pub retryable: bool,
}

impl SendError {
    fn transport(err: reqwest::Error) -> Self {
        Self {
            message: err.to_string(),
            status: err.status().map(|status| status.as_u16()),
            retryable: true,
        }
    }

    fn status(status: reqwest::StatusCode, body: String) -> Self {
        let retryable = status.as_u16() == 429 || status.is_server_error();
        let body = preview(&body);
        Self {
            message: format!("HTTP {status}: {body}"),
            status: Some(status.as_u16()),
            retryable,
        }
    }

    fn protocol(message: String) -> Self {
        Self {
            message,
            status: None,
            retryable: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecorderClient {
    base_url: String,
    client: reqwest::Client,
}

impl RecorderClient {
    pub fn new(config: &RecorderConfig) -> Result<Self, RecorderError> {
        Self::with_timeout(
            &config.server_url,
            config.api_key.as_deref(),
            config.timeout,
        )
    }

    pub fn with_timeout(
        server_url: &str,
        api_key: Option<&str>,
        timeout: Duration,
    ) -> Result<Self, RecorderError> {
        let parsed = reqwest::Url::parse(server_url.trim())
            .map_err(|err| RecorderError::Config(format!("invalid --server-url: {err}")))?;
        let api_key = api_key.map(str::trim).filter(|value| !value.is_empty());
        validate_server_url(&parsed, api_key.is_some())?;

        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(api_key) = api_key {
            let mut value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|err| RecorderError::Config(format!("invalid API key header: {err}")))?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(5))
            .timeout(timeout)
            .build()
            .map_err(|err| RecorderError::HttpClient(err.to_string()))?;

        Ok(Self {
            base_url: parsed.as_str().trim_end_matches('/').to_string(),
            client,
        })
    }

    pub async fn health(&self) -> Result<serde_json::Value, SendError> {
        self.get_json("/health").await
    }

    pub async fn readiness(&self) -> Result<serde_json::Value, SendError> {
        self.get_json("/readiness").await
    }

    pub async fn raw_recall_probe(&self) -> Result<serde_json::Value, SendError> {
        let response = self
            .client
            .post(self.url("/v1/recall/raw"))
            .json(&json!({
                "session_id": format!("recorder-doctor-auth-probe-{}", Uuid::now_v7()),
                "limit": 0
            }))
            .send()
            .await
            .map_err(SendError::transport)?;
        decode_json_response(response).await
    }

    pub async fn send_raw_batch(
        &self,
        request: &EncodeBatchStoreRawRequest,
    ) -> Result<EncodeBatchStoreRawResponse, SendError> {
        let response = self
            .client
            .post(self.url("/v1/encode/raw/batch"))
            .json(request)
            .send()
            .await
            .map_err(SendError::transport)?;
        let decoded = decode_json_response::<EncodeBatchStoreRawResponse>(response).await?;
        if decoded.results.len() != request.records.len() {
            return Err(SendError::protocol(format!(
                "raw batch response returned {} result(s) for {} request record(s)",
                decoded.results.len(),
                request.records.len()
            )));
        }
        Ok(decoded)
    }

    async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, SendError> {
        let response = self
            .client
            .get(self.url(path))
            .send()
            .await
            .map_err(SendError::transport)?;
        decode_json_response(response).await
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

fn validate_server_url(parsed: &reqwest::Url, uses_api_key: bool) -> Result<(), RecorderError> {
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(RecorderError::Config(
            "--server-url must use http or https".to_string(),
        ));
    }
    if parsed.host_str().is_none() {
        return Err(RecorderError::Config(
            "--server-url must include a host".to_string(),
        ));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(RecorderError::Config(
            "--server-url must not include username or password".to_string(),
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(RecorderError::Config(
            "--server-url must not include query parameters or fragments".to_string(),
        ));
    }
    if uses_api_key && parsed.scheme() == "http" && !server_host_is_loopback(parsed) {
        let host = parsed.host_str().unwrap_or("(missing host)");
        return Err(RecorderError::Config(format!(
            "refusing to send CEREMEMORY_SERVER_API_KEY over insecure HTTP to non-loopback host '{host}'; use https:// for remote servers or http://localhost, http://127.0.0.1, or http://[::1] for local-only HTTP"
        )));
    }
    Ok(())
}

fn server_host_is_loopback(parsed: &reqwest::Url) -> bool {
    let Some(host) = parsed.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

async fn decode_json_response<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, SendError> {
    if response.status().is_success() {
        return response.json::<T>().await.map_err(|err| SendError {
            message: format!("failed to decode JSON response: {err}"),
            status: None,
            retryable: false,
        });
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(SendError::status(status, body))
}

fn preview(body: &str) -> String {
    let redacted = redact_string(body);
    let trimmed = redacted.trim();
    if trimmed.is_empty() {
        return "<empty response body>".to_string();
    }
    const LIMIT: usize = 512;
    if trimmed.len() <= LIMIT {
        trimmed.to_string()
    } else {
        format!("{}...", trimmed.chars().take(LIMIT).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_server_url_with_query_or_fragment() {
        let err = RecorderClient::with_timeout(
            "http://127.0.0.1:8420?token=secret",
            None,
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(err.to_string().contains("query parameters"));

        let err = RecorderClient::with_timeout("file:///tmp/socket", None, Duration::from_secs(1))
            .unwrap_err();
        assert!(err.to_string().contains("http or https"));

        let err = RecorderClient::with_timeout(
            "http://user:pass@127.0.0.1:8420",
            None,
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(err.to_string().contains("username or password"));
    }

    #[test]
    fn rejects_api_key_on_non_loopback_http() {
        let err = RecorderClient::with_timeout(
            "http://example.com:8420",
            Some("secret"),
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(err.to_string().contains("non-loopback"));

        RecorderClient::with_timeout(
            "http://127.0.0.1:8420",
            Some(" secret "),
            Duration::from_secs(1),
        )
        .unwrap();

        RecorderClient::with_timeout(
            "https://example.com",
            Some("secret"),
            Duration::from_secs(1),
        )
        .unwrap();
    }

    #[test]
    fn redacts_secret_patterns_from_error_preview() {
        let err = SendError::status(
            reqwest::StatusCode::BAD_REQUEST,
            r#"invalid request api_key="abcdef123456""#.to_string(),
        );
        assert!(!err.to_string().contains("abcdef123456"));
        assert!(err.to_string().contains("[REDACTED]"));
    }
}
