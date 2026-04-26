//! Shared utilities for Cerememory LLM adapters.
//!
//! Provides common HTTP client construction, exponential backoff configuration,
//! and transient error classification used by all LLM provider adapters.

use std::future::Future;
use std::time::Duration;

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

/// Shared retry policy for transient LLM provider failures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    pub initial_interval: Duration,
    pub max_interval: Duration,
    pub multiplier: f64,
    pub max_retries: u32,
}

impl RetryPolicy {
    pub fn next_interval(self, current: Duration) -> Duration {
        Duration::from_secs_f64(
            (current.as_secs_f64() * self.multiplier).min(self.max_interval.as_secs_f64()),
        )
    }
}

/// Error classification returned by retryable operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryError<E> {
    Transient(E),
    Permanent(E),
}

impl<E> RetryError<E> {
    pub fn transient(error: E) -> Self {
        Self::Transient(error)
    }

    pub fn permanent(error: E) -> Self {
        Self::Permanent(error)
    }
}

/// Build a shared HTTP client with the standard timeout.
pub fn build_http_client() -> Result<Client, CerememoryError> {
    Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
        .map_err(|e| CerememoryError::Internal(format!("failed to build HTTP client: {e}")))
}

/// Create the standard exponential backoff policy for transient-error retries.
pub fn create_retry_policy() -> RetryPolicy {
    RetryPolicy {
        initial_interval: Duration::from_millis(DEFAULT_INITIAL_INTERVAL_MS),
        max_interval: Duration::from_secs(DEFAULT_MAX_ELAPSED_SECS),
        multiplier: DEFAULT_MULTIPLIER,
        max_retries: DEFAULT_MAX_RETRIES,
    }
}

/// Execute an async operation with exponential backoff for transient errors.
///
/// `max_retries` is the number of retries after the initial attempt, so the
/// operation can run at most `max_retries + 1` times.
pub async fn retry_with_policy<T, E, F, Fut>(policy: RetryPolicy, mut operation: F) -> Result<T, E>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<T, RetryError<E>>>,
{
    let mut attempt = 1;
    let mut current_interval = policy.initial_interval;

    loop {
        match operation(attempt).await {
            Ok(value) => return Ok(value),
            Err(RetryError::Permanent(error)) => return Err(error),
            Err(RetryError::Transient(error)) => {
                if attempt > policy.max_retries {
                    return Err(error);
                }

                tokio::time::sleep(current_interval).await;
                current_interval = policy.next_interval(current_interval);
                attempt += 1;
            }
        }
    }
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
    fn retry_policy_creates() {
        let policy = create_retry_policy();
        assert_eq!(
            policy.initial_interval,
            Duration::from_millis(DEFAULT_INITIAL_INTERVAL_MS)
        );
    }

    #[tokio::test]
    async fn retry_with_policy_retries_transient_errors() {
        let policy = RetryPolicy {
            initial_interval: Duration::ZERO,
            max_interval: Duration::ZERO,
            multiplier: DEFAULT_MULTIPLIER,
            max_retries: 3,
        };
        let mut attempts = 0;

        let result = retry_with_policy(policy, |_| {
            attempts += 1;
            async move {
                if attempts < 3 {
                    Err(RetryError::transient("temporary"))
                } else {
                    Ok("ok")
                }
            }
        })
        .await;

        assert_eq!(result, Ok("ok"));
        assert_eq!(attempts, 3);
    }

    #[tokio::test]
    async fn retry_with_policy_stops_on_permanent_error() {
        let policy = RetryPolicy {
            initial_interval: Duration::ZERO,
            max_interval: Duration::ZERO,
            multiplier: DEFAULT_MULTIPLIER,
            max_retries: 3,
        };
        let mut attempts = 0;

        let result = retry_with_policy(policy, |_| {
            attempts += 1;
            async move { Err::<(), _>(RetryError::permanent("fatal")) }
        })
        .await;

        assert_eq!(result, Err("fatal"));
        assert_eq!(attempts, 1);
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
