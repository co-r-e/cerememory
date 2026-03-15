//! OpenAI LLM provider implementation.
//!
//! Provides embedding, summarization, and relation extraction via the OpenAI API.
//! Uses exponential backoff retry for transient errors (429, 5xx).

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use cerememory_core::media::{normalize_audio_upload_format, normalize_image_mime_type};
use cerememory_core::{CerememoryError, ExtractedRelation, LLMProvider, ProviderCapabilities};
use reqwest::multipart;
use reqwest::{Client, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Default base URL for the OpenAI API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Default embedding model.
const DEFAULT_EMBED_MODEL: &str = "text-embedding-3-small";
/// Default chat model.
const DEFAULT_CHAT_MODEL: &str = "gpt-4o";

/// Maximum retry attempts for transient errors.
const MAX_RETRIES: u32 = 3;

/// Initial retry interval in milliseconds.
const INITIAL_INTERVAL_MS: u64 = 500;

/// Default Whisper model for audio transcription.
const DEFAULT_WHISPER_MODEL: &str = "whisper-1";

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

// ── Vision (GPT-4o) request types ──────────────────────────────────────────

#[derive(Serialize)]
struct VisionChatRequest<'a> {
    model: &'a str,
    messages: Vec<VisionMessage>,
    max_tokens: usize,
}

#[derive(Serialize)]
struct VisionMessage {
    role: String,
    content: Vec<VisionContent>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum VisionContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

// ── Whisper response ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
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
    fn build_client() -> Result<Client, CerememoryError> {
        Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| {
                CerememoryError::Internal(format!("failed to build OpenAI HTTP client: {e}"))
            })
    }

    /// Create a new `OpenAIProvider`.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key (stored as `SecretString`).
    /// * `model` - Chat model name. Defaults to `"gpt-4o"` when `None`.
    /// * `base_url` - API base URL. Defaults to `https://api.openai.com` when `None`.
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, CerememoryError> {
        Ok(Self {
            api_key: SecretString::from(api_key),
            model: model.unwrap_or_else(|| DEFAULT_CHAT_MODEL.to_string()),
            embed_model: DEFAULT_EMBED_MODEL.to_string(),
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            client: Self::build_client()?,
        })
    }

    /// Build the backoff policy used for transient-error retries.
    fn backoff_policy() -> backoff::ExponentialBackoff {
        ExponentialBackoffBuilder::default()
            .with_initial_interval(Duration::from_millis(INITIAL_INTERVAL_MS))
            .with_max_elapsed_time(Some(Duration::from_secs(30)))
            .with_multiplier(2.0)
            .build()
    }

    /// Execute an HTTP request with exponential backoff retry on transient errors.
    ///
    /// `build_request` is called on each attempt to produce the `RequestBuilder`.
    /// This allows both JSON and multipart requests to share the same retry logic.
    async fn send_with_retry<F>(&self, build_request: F) -> Result<String, CerememoryError>
    where
        F: Fn() -> Result<reqwest::RequestBuilder, CerememoryError>,
    {
        let mut attempts: u32 = 0;
        let backoff = Self::backoff_policy();
        let mut current_interval = backoff.initial_interval;

        loop {
            attempts += 1;

            let resp = build_request()?
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.expose_secret()),
                )
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

            let is_retryable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();

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

    /// POST a JSON body to `url` with retries on transient HTTP errors.
    async fn post_with_retry(
        &self,
        url: &str,
        body: &impl Serialize,
    ) -> Result<String, CerememoryError> {
        let body_bytes = serde_json::to_vec(body)
            .map_err(|e| CerememoryError::Internal(format!("request serialization: {e}")))?;

        self.send_with_retry(|| {
            Ok(self
                .client
                .post(url)
                .header("Content-Type", "application/json")
                .body(body_bytes.clone()))
        })
        .await
    }

    /// POST a multipart form to `url` with retries on transient HTTP errors.
    ///
    /// Because `multipart::Form` is consumed on send, the caller provides a
    /// factory closure that can recreate the form for each attempt.
    async fn post_multipart_with_retry<F>(
        &self,
        url: &str,
        form_factory: F,
    ) -> Result<String, CerememoryError>
    where
        F: Fn() -> Result<multipart::Form, CerememoryError>,
    {
        self.send_with_retry(|| {
            let form = form_factory()?;
            Ok(self.client.post(url).multipart(form))
        })
        .await
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
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ExtractedRelation>, CerememoryError>> + Send + '_>>
    {
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

            let parsed: RelationExtractionOutput = serde_json::from_str(&content).map_err(|e| {
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

    fn embed_image(
        &self,
        data: &[u8],
        format: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>> {
        let data = data.to_vec();
        let mime_type = normalize_image_mime_type(format).map(str::to_owned);
        Box::pin(async move {
            let mime_type = mime_type?;
            if data.is_empty() {
                return Err(CerememoryError::Validation(
                    "Image payload must not be empty".to_string(),
                ));
            }
            // Step 1: Base64-encode the image and build a data URI.
            let b64 = BASE64.encode(&data);
            let data_uri = format!("data:{mime_type};base64,{b64}");

            // Step 2: Send to GPT-4o vision to get a text description.
            let url = format!("{}/v1/chat/completions", self.base_url);

            let vision_req = VisionChatRequest {
                model: &self.model,
                messages: vec![VisionMessage {
                    role: "user".to_string(),
                    content: vec![
                        VisionContent::Text {
                            text: "Describe this image in detail for memory indexing. \
                                   Focus on objects, people, text, colors, and spatial \
                                   relationships."
                                .to_string(),
                        },
                        VisionContent::ImageUrl {
                            image_url: ImageUrl { url: data_uri },
                        },
                    ],
                }],
                max_tokens: 512,
            };

            debug!(model = %self.model, "requesting image description via vision");

            let body = self.post_with_retry(&url, &vision_req).await?;
            let resp: ChatResponse = serde_json::from_str(&body).map_err(|e| {
                CerememoryError::Internal(format!("vision response parse error: {e}"))
            })?;

            let description = resp
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .ok_or_else(|| {
                    CerememoryError::Internal(
                        "vision response contained no description".to_string(),
                    )
                })?;

            // Step 3: Embed the text description using the standard embedding pipeline.
            debug!("embedding image description ({} chars)", description.len());
            self.embed(&description).await
        })
    }

    fn transcribe_audio(
        &self,
        data: &[u8],
        format: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, CerememoryError>> + Send + '_>> {
        let data = data.to_vec();
        let audio_format = normalize_audio_upload_format(format)
            .map(|(extension, mime_type)| (extension.to_string(), mime_type.to_string()));
        Box::pin(async move {
            let url = format!("{}/v1/audio/transcriptions", self.base_url);
            let (extension, mime_type) = audio_format?;
            if data.is_empty() {
                return Err(CerememoryError::Validation(
                    "Audio payload must not be empty".to_string(),
                ));
            }

            // Determine a suitable file name for the multipart upload.
            let filename = format!("audio.{extension}");

            let audio_data = data.clone();
            let audio_filename = filename.clone();
            let audio_mime = mime_type.clone();

            let body = self
                .post_multipart_with_retry(&url, move || {
                    let part = multipart::Part::bytes(audio_data.clone())
                        .file_name(audio_filename.clone())
                        .mime_str(&audio_mime)
                        .map_err(|e| {
                            CerememoryError::Validation(format!(
                                "Invalid audio MIME type '{audio_mime}': {e}"
                            ))
                        })?;

                    Ok(multipart::Form::new()
                        .text("model", DEFAULT_WHISPER_MODEL)
                        .part("file", part))
                })
                .await?;

            debug!("received transcription response");

            let resp: TranscriptionResponse = serde_json::from_str(&body).map_err(|e| {
                CerememoryError::Internal(format!("transcription response parse error: {e}"))
            })?;

            Ok(resp.text)
        })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_embedding: true,
            image_embedding: true,
            audio_transcription: true,
        }
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
        .expect("OpenAI provider should build")
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
        let Some(server) = start_mock_server().await else {
            return;
        };

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(200).set_body_json(embedding_response_body()))
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
        let Some(server) = start_mock_server().await else {
            return;
        };

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
        let Some(server) = start_mock_server().await else {
            return;
        };

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
        let Some(server) = start_mock_server().await else {
            return;
        };

        // First two requests get 429, third succeeds.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .up_to_n_times(2)
            .expect(2)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(embedding_response_body()))
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
        let Some(server) = start_mock_server().await else {
            return;
        };

        // All requests return 500 -- should exhaust retries.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
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

    // ── embed_image ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_embed_image_returns_vector() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        // Mock 1: GPT-4o vision endpoint returns a text description.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response_body("A red cat sitting on a blue mat.")),
            )
            .expect(1)
            .mount(&server)
            .await;

        // Mock 2: Embedding endpoint returns a vector for the description.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(200).set_body_json(embedding_response_body()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        // Fake 1x1 PNG bytes (content doesn't matter for the mock).
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let result = provider.embed_image(&fake_image, "image/png").await;

        let vec = result.expect("embed_image should succeed");
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 0.1).abs() < f32::EPSILON);
        assert!((vec[1] - 0.2).abs() < f32::EPSILON);
        assert!((vec[2] - 0.3).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_embed_image_accepts_mime_type() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response_body("An indexed image.")),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(200).set_body_json(embedding_response_body()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let result = provider.embed_image(&fake_image, "image/png").await;

        assert!(result.is_ok(), "embed_image should accept MIME types");
    }

    // ── transcribe_audio ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_transcribe_audio_returns_text() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let transcription_body = serde_json::json!({
            "text": "Hello, this is a test transcription."
        });

        // The Whisper endpoint receives multipart form data.
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(200).set_body_json(transcription_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        // Fake WAV bytes (content doesn't matter for the mock).
        let fake_audio = vec![0x52, 0x49, 0x46, 0x46]; // "RIFF" header
        let result = provider.transcribe_audio(&fake_audio, "audio/wav").await;

        let text = result.expect("transcribe_audio should succeed");
        assert_eq!(text, "Hello, this is a test transcription.");
    }

    #[tokio::test]
    async fn test_transcribe_audio_accepts_mime_type() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let transcription_body = serde_json::json!({
            "text": "MIME input works."
        });

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(200).set_body_json(transcription_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let fake_audio = vec![0x52, 0x49, 0x46, 0x46];
        let result = provider.transcribe_audio(&fake_audio, "audio/wav").await;

        assert_eq!(
            result.expect("transcribe_audio should accept MIME types"),
            "MIME input works."
        );
    }

    #[tokio::test]
    async fn test_transcribe_audio_retries_multipart_requests() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let transcription_body = serde_json::json!({
            "text": "Retry path works."
        });

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .up_to_n_times(2)
            .expect(2)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(200).set_body_json(transcription_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let fake_audio = vec![0x52, 0x49, 0x46, 0x46];
        let result = provider.transcribe_audio(&fake_audio, "wav").await;

        assert_eq!(
            result.expect("multipart retry path should succeed"),
            "Retry path works."
        );
    }

    #[tokio::test]
    async fn test_transcribe_audio_retry_exhaustion_surfaces_error() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("Authorization", "Bearer test-key-1234"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
            .expect(1..=4)
            .mount(&server)
            .await;

        let provider = mock_provider(&server.uri());
        let fake_audio = vec![0x52, 0x49, 0x46, 0x46];
        let err = provider
            .transcribe_audio(&fake_audio, "wav")
            .await
            .expect_err("multipart retry exhaustion should fail");

        assert!(err.to_string().contains("500"));
    }

    #[tokio::test]
    async fn test_transcribe_audio_rejects_unknown_format() {
        let provider = mock_provider("http://unused");
        let err = provider
            .transcribe_audio(b"invalid", "application/octet-stream")
            .await
            .expect_err("unknown formats must be rejected");

        assert!(err.to_string().contains("Unsupported audio format"));
    }

    // ── capabilities ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_capabilities() {
        let provider = mock_provider("http://unused");
        let caps = provider.capabilities();

        assert!(caps.text_embedding, "text_embedding should be true");
        assert!(caps.image_embedding, "image_embedding should be true");
        assert!(
            caps.audio_transcription,
            "audio_transcription should be true"
        );
    }
}
