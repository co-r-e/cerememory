//! Claude (Anthropic) LLM provider implementation.
//!
//! Implements `LLMProvider` for the Anthropic Messages API.
//! Claude does not offer an embedding endpoint, so `embed()` returns an error
//! directing callers to use a dedicated embedding provider.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use cerememory_core::{CerememoryError, ExtractedRelation, LLMProvider, ProviderCapabilities};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Default model used when none is specified.
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";

/// Default Anthropic API base URL.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Maximum number of retry attempts for transient failures.
const MAX_RETRIES: u32 = 3;

/// Initial retry interval in milliseconds.
const INITIAL_INTERVAL_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Request / response types for the Anthropic Messages API
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: usize,
    messages: Vec<Message<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice<'a>>,
}

#[derive(Debug, Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ToolDefinition<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ToolChoice<'a> {
    r#type: &'a str,
    name: &'a str,
}

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct RelationsToolInput {
    relations: Vec<RelationEntry>,
}

#[derive(Debug, Deserialize)]
struct RelationEntry {
    subject: String,
    predicate: String,
    object: String,
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    r#type: String,
    message: String,
}

// ---------------------------------------------------------------------------
// ClaudeProvider
// ---------------------------------------------------------------------------

/// LLM provider backed by the Anthropic Claude Messages API.
///
/// Supports `summarize` and `extract_relations` via the Messages API.
/// `embed` always returns an error because Claude has no embedding endpoint.
pub struct ClaudeProvider {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
}

impl ClaudeProvider {
    fn build_client() -> Result<Client, CerememoryError> {
        Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| {
                CerememoryError::Internal(format!("failed to build Anthropic HTTP client: {e}"))
            })
    }

    /// Create a new `ClaudeProvider`.
    ///
    /// # Arguments
    /// * `api_key` - Anthropic API key.
    /// * `model` - Model identifier. Defaults to `claude-sonnet-4-20250514`.
    /// * `base_url` - API base URL. Defaults to `https://api.anthropic.com`.
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, CerememoryError> {
        Ok(Self {
            api_key: SecretString::from(api_key),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            client: Self::build_client()?,
        })
    }

    /// Build the full URL for the messages endpoint.
    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// Send a request to the Anthropic Messages API with exponential backoff retry.
    async fn send_with_retry(
        &self,
        body: &impl Serialize,
    ) -> Result<MessagesResponse, CerememoryError> {
        let backoff = ExponentialBackoffBuilder::default()
            .with_initial_interval(Duration::from_millis(INITIAL_INTERVAL_MS))
            .with_max_elapsed_time(Some(Duration::from_secs(30)))
            .build();

        let url = self.messages_url();
        let attempt = AtomicU32::new(0);

        let op = || {
            let current = attempt.fetch_add(1, Ordering::Relaxed) + 1;
            let url = url.clone();
            async move {
                debug!(attempt = current, url = %url, "sending Anthropic API request");

                let resp = self
                    .client
                    .post(&url)
                    .header("x-api-key", self.api_key.expose_secret())
                    .header("anthropic-version", ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .json(body)
                    .send()
                    .await
                    .map_err(|e| {
                        if e.is_connect() || e.is_timeout() {
                            backoff::Error::transient(CerememoryError::Internal(format!(
                                "Anthropic API connection error: {e}"
                            )))
                        } else {
                            backoff::Error::permanent(CerememoryError::Internal(format!(
                                "Anthropic API request error: {e}"
                            )))
                        }
                    })?;

                let status = resp.status();

                if status.is_success() {
                    let parsed: MessagesResponse = resp.json().await.map_err(|e| {
                        backoff::Error::permanent(CerememoryError::Internal(format!(
                            "Failed to parse Anthropic API response: {e}"
                        )))
                    })?;
                    return Ok(parsed);
                }

                // Parse error body for better diagnostics.
                let error_body = resp.text().await.unwrap_or_default();

                // Rate-limited or server errors are transient.
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    if current >= MAX_RETRIES {
                        warn!(
                            status = %status,
                            attempt = current,
                            "Anthropic API retryable error, max retries reached"
                        );
                        return Err(backoff::Error::permanent(CerememoryError::Internal(
                        format!(
                            "Anthropic API error after {current} attempts (HTTP {status}): {error_body}"
                        ),
                    )));
                    }
                    warn!(
                        status = %status,
                        attempt = current,
                        "Anthropic API retryable error, will retry"
                    );
                    return Err(backoff::Error::transient(CerememoryError::Internal(
                        format!("Anthropic API error (HTTP {status}): {error_body}"),
                    )));
                }

                // All other errors are permanent.
                let detail = serde_json::from_str::<AnthropicErrorResponse>(&error_body)
                    .map(|e| format!("{}: {}", e.error.r#type, e.error.message))
                    .unwrap_or(error_body);

                Err(backoff::Error::permanent(CerememoryError::Internal(
                    format!("Anthropic API error (HTTP {status}): {detail}"),
                )))
            }
        };

        backoff::future::retry(backoff, op).await
    }
}

impl LLMProvider for ClaudeProvider {
    fn embed(
        &self,
        _text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>> {
        Box::pin(async {
            Err(CerememoryError::ModalityUnsupported(
                "Claude does not support text embeddings. Use a separate embedding provider."
                    .into(),
            ))
        })
    }

    fn summarize(
        &self,
        texts: &[String],
        max_tokens: usize,
    ) -> Pin<Box<dyn Future<Output = Result<String, CerememoryError>> + Send + '_>> {
        let joined = texts.join("\n\n---\n\n");
        let prompt = format!(
            "Summarize the following texts into a single concise summary. \
             Preserve key facts and relationships. Output only the summary, \
             no preamble.\n\n{joined}"
        );

        Box::pin(async move {
            let request = MessagesRequest {
                model: &self.model,
                max_tokens,
                messages: vec![Message {
                    role: "user",
                    content: &prompt,
                }],
                tools: None,
                tool_choice: None,
            };

            let response = self.send_with_retry(&request).await?;

            // Extract the first text block from the response.
            for block in &response.content {
                if let ContentBlock::Text { text } = block {
                    return Ok(text.clone());
                }
            }

            Err(CerememoryError::Internal(
                "Anthropic API returned no text content".into(),
            ))
        })
    }

    fn extract_relations(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ExtractedRelation>, CerememoryError>> + Send + '_>>
    {
        let prompt = format!(
            "Extract semantic relations (subject-predicate-object triples) from the following text. \
             Use the extract_relations tool to return your results.\n\n{text}"
        );

        Box::pin(async move {
            let tool_schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {
                                    "type": "string",
                                    "description": "The subject entity"
                                },
                                "predicate": {
                                    "type": "string",
                                    "description": "The relationship or action"
                                },
                                "object": {
                                    "type": "string",
                                    "description": "The object entity"
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score between 0.0 and 1.0",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                }
                            },
                            "required": ["subject", "predicate", "object", "confidence"]
                        }
                    }
                },
                "required": ["relations"]
            });

            let request = MessagesRequest {
                model: &self.model,
                max_tokens: 4096,
                messages: vec![Message {
                    role: "user",
                    content: &prompt,
                }],
                tools: Some(vec![ToolDefinition {
                    name: "extract_relations",
                    description: "Extract semantic relations as subject-predicate-object triples",
                    input_schema: tool_schema,
                }]),
                tool_choice: Some(ToolChoice {
                    r#type: "tool",
                    name: "extract_relations",
                }),
            };

            let response = self.send_with_retry(&request).await?;

            // Find the tool_use block.
            for block in &response.content {
                if let ContentBlock::ToolUse { input, .. } = block {
                    let parsed: RelationsToolInput = serde_json::from_value(input.clone())
                        .map_err(|e| {
                            CerememoryError::Internal(format!(
                                "Failed to parse extract_relations tool output: {e}"
                            ))
                        })?;

                    return Ok(parsed
                        .relations
                        .into_iter()
                        .map(|r| ExtractedRelation {
                            subject: r.subject,
                            predicate: r.predicate,
                            object: r.object,
                            confidence: r.confidence.clamp(0.0, 1.0),
                        })
                        .collect());
                }
            }

            Err(CerememoryError::Internal(
                "Anthropic API returned no tool_use content for extract_relations".into(),
            ))
        })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_embedding: false,
            image_embedding: false,
            audio_transcription: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_provider(base_url: &str) -> ClaudeProvider {
        ClaudeProvider::new("test-api-key".to_string(), None, Some(base_url.to_string()))
            .expect("Claude provider should build")
    }

    async fn start_mock_server() -> Option<MockServer> {
        match tokio::spawn(async { MockServer::start().await }).await {
            Ok(server) => Some(server),
            Err(err) => {
                eprintln!("skipping wiremock-based test: {err}");
                None
            }
        }
    }

    #[tokio::test]
    async fn test_embed_returns_error() {
        let provider = make_provider("http://unused");
        let result = provider.embed("hello").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Claude does not support text embeddings"),
            "unexpected error message: {msg}"
        );
    }

    #[tokio::test]
    async fn test_summarize_returns_text() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let response_body = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Alice is a software engineer who works at Acme Corp."
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 50, "output_tokens": 20 }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-api-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let result = provider
            .summarize(
                &[
                    "Alice is a software engineer.".to_string(),
                    "Alice works at Acme Corp.".to_string(),
                ],
                256,
            )
            .await;

        assert!(result.is_ok(), "summarize failed: {:?}", result.err());
        let summary = result.unwrap();
        assert_eq!(
            summary,
            "Alice is a software engineer who works at Acme Corp."
        );
    }

    #[tokio::test]
    async fn test_extract_relations_returns_triples() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let response_body = serde_json::json!({
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "extract_relations",
                    "input": {
                        "relations": [
                            {
                                "subject": "Alice",
                                "predicate": "works_at",
                                "object": "Acme Corp",
                                "confidence": 0.95
                            },
                            {
                                "subject": "Alice",
                                "predicate": "is_a",
                                "object": "software engineer",
                                "confidence": 0.9
                            }
                        ]
                    }
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 80, "output_tokens": 60 }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-api-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let result = provider
            .extract_relations("Alice is a software engineer who works at Acme Corp.")
            .await;

        assert!(
            result.is_ok(),
            "extract_relations failed: {:?}",
            result.err()
        );
        let relations = result.unwrap();
        assert_eq!(relations.len(), 2);

        assert_eq!(relations[0].subject, "Alice");
        assert_eq!(relations[0].predicate, "works_at");
        assert_eq!(relations[0].object, "Acme Corp");
        assert!((relations[0].confidence - 0.95).abs() < f64::EPSILON);

        assert_eq!(relations[1].subject, "Alice");
        assert_eq!(relations[1].predicate, "is_a");
        assert_eq!(relations[1].object, "software engineer");
        assert!((relations[1].confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_capabilities() {
        let provider = make_provider("http://unused");
        let caps = provider.capabilities();
        assert!(
            !caps.text_embedding,
            "Claude should not advertise text_embedding"
        );
        assert!(
            !caps.image_embedding,
            "Claude should not advertise image_embedding"
        );
        assert!(
            !caps.audio_transcription,
            "Claude should not advertise audio_transcription"
        );
    }

    #[tokio::test]
    async fn test_429_retry_succeeds() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let success_body = serde_json::json!({
            "id": "msg_789",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Summary after retry."
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 30, "output_tokens": 10 }
        });

        // First request returns 429, second succeeds.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limited"
                }
            })))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&success_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let result = provider
            .summarize(&["Some text to summarize.".to_string()], 128)
            .await;

        assert!(
            result.is_ok(),
            "summarize should succeed after 429 retry: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), "Summary after retry.");
    }
}
