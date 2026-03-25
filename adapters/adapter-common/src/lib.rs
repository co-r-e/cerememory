//! Shared utilities for Cerememory LLM adapters.
//!
//! Provides common HTTP client construction, exponential backoff configuration,
//! and transient error classification used by all LLM provider adapters.

use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use cerememory_core::CerememoryError;
use reqwest::Client;

/// Default HTTP client timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Default initial retry interval in milliseconds.
pub const DEFAULT_INITIAL_INTERVAL_MS: u64 = 500;

/// Default backoff multiplier.
pub const DEFAULT_MULTIPLIER: f64 = 2.0;

/// Default maximum elapsed time for retries in seconds.
pub const DEFAULT_MAX_ELAPSED_SECS: u64 = 30;

/// Default maximum number of retry attempts.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Build a shared HTTP client with the standard timeout.
pub fn build_http_client() -> Result<Client, CerememoryError> {
    Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
        .map_err(|e| CerememoryError::Internal(format!("failed to build HTTP client: {e}")))
}

/// Create the standard exponential backoff policy for transient-error retries.
pub fn create_backoff_policy() -> backoff::ExponentialBackoff {
    ExponentialBackoffBuilder::default()
        .with_initial_interval(Duration::from_millis(DEFAULT_INITIAL_INTERVAL_MS))
        .with_max_elapsed_time(Some(Duration::from_secs(DEFAULT_MAX_ELAPSED_SECS)))
        .with_multiplier(DEFAULT_MULTIPLIER)
        .build()
}

/// Check whether an HTTP status code represents a transient error worth retrying.
///
/// Returns `true` for 429 (Too Many Requests) and 5xx (server errors).
pub fn is_transient_error(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_client_succeeds() {
        assert!(build_http_client().is_ok());
    }

    #[test]
    fn backoff_policy_creates() {
        let policy = create_backoff_policy();
        assert_eq!(
            policy.initial_interval,
            Duration::from_millis(DEFAULT_INITIAL_INTERVAL_MS)
        );
    }

    #[test]
    fn transient_error_classification() {
        assert!(is_transient_error(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(is_transient_error(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(is_transient_error(reqwest::StatusCode::BAD_GATEWAY));
        assert!(!is_transient_error(reqwest::StatusCode::NOT_FOUND));
        assert!(!is_transient_error(reqwest::StatusCode::OK));
    }
}
