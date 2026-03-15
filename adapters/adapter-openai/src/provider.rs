//! OpenAI LLM provider implementation.
//!
//! Provides embedding, summarization, and relation extraction via the OpenAI API.
//! Uses exponential backoff retry for transient errors (429, 5xx).

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use cerememory_core::{CerememoryError, ExtractedRelation, LLMProvider};
use reqwest::{Client, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Default base URL for the OpenAI API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Default embedding model.
const DEFAULT_EMBED_MODEL: &str = "text-embedding-3-small";

/// Maximum retry attempts for transient errors.
const MAX_RETRIES: u32 = 3;

/// Initial retry interval in milliseconds.
const INITIAL_INTERVAL_MS: u64 = 500;

// ── Request / Response types ────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ResponseFormat {
    r#type: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

#[derive(Deserialize)]
struct RelationExtractionOutput {
    relations: Vec<RelationEntry>,
}

#[derive(Deserialize)]
struct RelationEntry {
    subject: String,
    predicate: String,
    object: String,
    confidence: f64,
}

// ── Provider ────────────────────────────────────────────────────────────────

/// OpenAI-compatible LLM provider.
///
/// Supports any API that follows the OpenAI REST convention (embeddings and
/// chat completions). The API key is stored as a `SecretString` and only
/// exposed when constructing HTTP headers.
pub struct OpenAIProvider {
    api_key: SecretString,
    model: String,
    embed_model: String,
    base_url: String,
    client: Client,
}

impl OpenAIProvider {
    /// Create a new `OpenAIProvider`.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key (stored as `SecretString`).
    /// * `model` - Chat model name. Defaults to `"gpt-4o"` when `None`.
    /// * `base_url` - API base URL. Defaults to `https://api.openai.com` when `None`.
    pub fn new(api_key: String, model: Option<String>, base_url: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("failed to build reqwest client");

        Self {
            api_key: SecretString::from(api_key),
            model: model.unwrap_or_else(|| "gpt-4o".to_string()),
            embed_model: DEFAULT_EMBED_MODEL.to_string(),
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            client,
        }
    }

    /// Build the backoff policy used for transient-error retries.
    fn backoff_policy() -> backoff::ExponentialBackoff {
        ExponentialBackoffBuilder::default()
            .with_initial_interval(Duration::from_millis(INITIAL_INTERVAL_MS))
            .with_max_elapsed_time(Some(Duration::from_secs(30)))
            .with_multiplier(2.0)
            .build()
    }

    /// POST a JSON body to `url` with retries on transient HTTP errors.
    ///
    /// Returns the raw response body on success, or a `CerememoryError::Internal`
    /// if retries are exhausted or a non-retryable error occurs.
    async fn post_with_retry(
        &self,
        url: &str,
        body: &impl Serialize,
    ) -> Result<String, CerememoryError> {
        let body_bytes = serde_json::to_vec(body)
            .map_err(|e| CerememoryError::Internal(format!("request serialization: {e}")))?;

        let mut attempts: u32 = 0;
        let backoff = Self::backoff_policy();
        let mut current_interval = backoff.initial_interval;

        loop {
            attempts += 1;

            let resp = self
                .client
                .post(url)
                .header("Content-Type", "application/json")
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.expose_secret()),
                )
                .body(body_bytes.clone())
                .send()
                .await
                .map_err(|e| CerememoryError::Internal(format!("HTTP request failed: {e}")))?;

            let status = resp.status();

            if status.is_success() {
                let text = resp.text().await.map_err(|e| {
                    CerememoryError::Internal(format!("failed to read response body: {e}"))
                })?;
                return Ok(text);
            }

            let is_retryable =
                status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();

            if !is_retryable || attempts > MAX_RETRIES {
                let error_body = resp.text().await.unwrap_or_default();
                return Err(CerememoryError::Internal(format!(
                    "OpenAI API error (HTTP {status}): {error_body}"
                )));
            }

            warn!(
                status = %status,
                attempt = attempts,
                "transient OpenAI API error, retrying after {current_interval:?}"
            );

            tokio::time::sleep(current_interval).await;
            current_interval = Duration::from_secs_f64(
                (current_interval.as_secs_f64() * backoff.multiplier)
                    .min(backoff.max_interval.as_secs_f64()),
            );
        }
    }
}

impl LLMProvider for OpenAIProvider {
    fn embed(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>> {
        let text = text.to_owned();
        Box::pin(async move {
            let url = format!("{}/v1/embeddings", self.base_url);
            let req = EmbeddingRequest {
                model: &self.embed_model,
                input: &text,
            };

            debug!(model = %self.embed_model, "requesting embedding");

            let body = self.post_with_retry(&url, &req).await?;
            let resp: EmbeddingResponse = serde_json::from_str(&body).map_err(|e| {
                CerememoryError::Internal(format!("embedding response parse error: {e}"))
            })?;

            resp.data
                .into_iter()
                .next()
                .map(|d| d.embedding)
                .ok_or_else(|| {
                    CerememoryError::Internal("embedding response contained no data".to_string())
                })
        })
    }

    fn summarize(
        &self,
        texts: &[String],
        max_tokens: usize,
    ) -> Pin<Box<dyn Future<Output = Result<String, CerememoryError>> + Send + '_>> {
        let joined = texts.join("\n---\n");
        Box::pin(async move {
            let url = format!("{}/v1/chat/completions", self.base_url);

            let user_content = format!(
                "Summarize the following texts into a single concise summary. \
                 Preserve key facts and relationships.\n\n{joined}"
            );

            let req = ChatRequest {
                model: &self.model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: "You are a precise summarization assistant. \
                                  Output only the summary, no preamble.",
                    },
                    ChatMessage {
                        role: "user",
                        content: &user_content,
                    },
                ],
                max_tokens,
                response_format: None,
            };

            debug!(model = %self.model, "requesting summarization");

            let body = self.post_with_retry(&url, &req).await?;
            let resp: ChatResponse = serde_json::from_str(&body).map_err(|e| {
                CerememoryError::Internal(format!("chat response parse error: {e}"))
            })?;

            resp.choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .ok_or_else(|| {
                    CerememoryError::Internal(
                        "chat completion response contained no content".to_string(),
                    )
                })
        })
    }

    fn extract_relations(
        &self,
        text: &str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<ExtractedRelation>, CerememoryError>> + Send + '_,
        >,
    > {
        let text = text.to_owned();
        Box::pin(async move {
            let url = format!("{}/v1/chat/completions", self.base_url);

            let user_content = format!(
                "Extract semantic relations from the following text as subject-predicate-object \
                 triples. Return a JSON object with a \"relations\" array. Each element must have \
                 \"subject\", \"predicate\", \"object\" (strings) and \"confidence\" (float 0-1).\n\n\
                 Text: {text}"
            );

            let req = ChatRequest {
                model: &self.model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: "You are a precise relation extraction assistant. \
                                  Output valid JSON only.",
                    },
                    ChatMessage {
                        role: "user",
                        content: &user_content,
                    },
                ],
                max_tokens: 2048,
                response_format: Some(ResponseFormat {
                    r#type: "json_object".to_string(),
                }),
            };

            debug!(model = %self.model, "requesting relation extraction");

            let body = self.post_with_retry(&url, &req).await?;
            let resp: ChatResponse = serde_json::from_str(&body).map_err(|e| {
                CerememoryError::Internal(format!("chat response parse error: {e}"))
            })?;

            let content = resp
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .ok_or_else(|| {
                    CerememoryError::Internal(
                        "chat completion response contained no content".to_string(),
                    )
                })?;

            let parsed: RelationExtractionOutput =
                serde_json::from_str(&content).map_err(|e| {
                    CerememoryError::Internal(format!(
                        "relation extraction JSON parse error: {e}, raw: {content}"
                    ))
                })?;

            Ok(parsed
                .relations
                .into_iter()
                .map(|r| ExtractedRelation {
                    subject: r.subject,
                    predicate: r.predicate,
                    object: r.object,
                    confidence: r.confidence,
                })
                .collect())
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Helper: create a provider pointing at the mock server.
    fn mock_provider(base_url: &str) -> OpenAIProvider {
        OpenAIProvider::new(
            "test-key-1234".to_string(),
            Some("gpt-4o".to_string()),
            Some(base_url.to_string()),
        )
    }

    /// Canonical embedding response with a 3-dimensional vector.
    fn embedding_response_body() -> serde_json::Value {
        serde_json::json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3]
            }],
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 5, "total_tokens": 5 }
        })
    }

    /// Canonical chat completion response.
    fn chat_response_body(content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        })
    }

    // ── embed ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_embed_returns_vector() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(embedding_response_body()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let result = provider.embed("hello world").await;

        let vec = result.expect("embed should succeed");
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 0.1).abs() < f32::EPSILON);
        assert!((vec[1] - 0.2).abs() < f32::EPSILON);
        assert!((vec[2] - 0.3).abs() < f32::EPSILON);
    }

    // ── summarize ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_summarize_returns_text() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response_body("This is a concise summary.")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let result = provider
            .summarize(&["Text one.".to_string(), "Text two.".to_string()], 256)
            .await;

        let summary = result.expect("summarize should succeed");
        assert_eq!(summary, "This is a concise summary.");
    }

    // ── extract_relations ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_extract_relations_returns_triples() {
        let server = MockServer::start().await;

        let relations_json = serde_json::json!({
            "relations": [
                {
                    "subject": "Alice",
                    "predicate": "works_at",
                    "object": "Acme Corp",
                    "confidence": 0.95
                },
                {
                    "subject": "Bob",
                    "predicate": "knows",
                    "object": "Alice",
                    "confidence": 0.8
                }
            ]
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response_body(&relations_json.to_string())),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let result = provider
            .extract_relations("Alice works at Acme Corp. Bob knows Alice.")
            .await;

        let relations = result.expect("extract_relations should succeed");
        assert_eq!(relations.len(), 2);

        assert_eq!(relations[0].subject, "Alice");
        assert_eq!(relations[0].predicate, "works_at");
        assert_eq!(relations[0].object, "Acme Corp");
        assert!((relations[0].confidence - 0.95).abs() < f64::EPSILON);

        assert_eq!(relations[1].subject, "Bob");
        assert_eq!(relations[1].predicate, "knows");
        assert_eq!(relations[1].object, "Alice");
        assert!((relations[1].confidence - 0.8).abs() < f64::EPSILON);
    }

    // ── retry on 429 ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_429_retry_succeeds() {
        let server = MockServer::start().await;

        // First two requests get 429, third succeeds.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(429).set_body_string("rate limited"),
            )
            .up_to_n_times(2)
            .expect(2)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(embedding_response_body()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let result = provider.embed("retry me").await;

        let vec = result.expect("should succeed after retries");
        assert_eq!(vec.len(), 3);
    }

    // ── retry exhaustion on 500 ─────────────────────────────────────────

    #[tokio::test]
    async fn test_500_retry_eventually_fails() {
        let server = MockServer::start().await;

        // All requests return 500 -- should exhaust retries.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(500).set_body_string("internal server error"),
            )
            .expect(1..=4)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let result = provider.embed("will fail").await;

        let err = result.expect_err("should fail after retry exhaustion");
        let msg = err.to_string();
        assert!(
            msg.contains("500"),
            "error should mention HTTP 500, got: {msg}"
        );
    }
}
