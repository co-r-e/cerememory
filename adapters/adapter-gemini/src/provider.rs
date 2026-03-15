//! Google Gemini LLM provider implementation.
//!
//! Implements [`LLMProvider`] for the Gemini API, supporting:
//! - Text embeddings via `text-embedding-004`
//! - Summarization via `gemini-2.0-flash` (or configurable model)
//! - Relation extraction via structured JSON output

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use base64::Engine as _;
use cerememory_core::media::normalize_image_mime_type;
use cerememory_core::{CerememoryError, ExtractedRelation, LLMProvider, ProviderCapabilities};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_MODEL: &str = "gemini-2.0-flash";
const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-004";
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const GOOGLE_API_KEY_HEADER: &str = "x-goog-api-key";

const BACKOFF_INITIAL_INTERVAL_MS: u64 = 500;
const BACKOFF_MAX_RETRIES: u32 = 3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Google Gemini LLM provider.
///
/// Wraps the Gemini REST API for embedding, summarization and relation
/// extraction.  The API key is held in a [`SecretString`] so it won't
/// accidentally appear in logs.
pub struct GeminiProvider {
    api_key: SecretString,
    model: String,
    embedding_model: String,
    base_url: String,
    client: Client,
}

impl GeminiProvider {
    fn build_client() -> Result<Client, CerememoryError> {
        Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| {
                CerememoryError::Internal(format!("failed to build Gemini HTTP client: {e}"))
            })
    }

    /// Create a new `GeminiProvider`.
    ///
    /// * `api_key` - Gemini API key.
    /// * `model`   - Generation model name. Defaults to `gemini-2.0-flash`.
    /// * `base_url`- API base URL. Defaults to `https://generativelanguage.googleapis.com`.
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, CerememoryError> {
        Ok(Self {
            api_key: SecretString::from(api_key),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            client: Self::build_client()?,
        })
    }
}

// ---------------------------------------------------------------------------
// LLMProvider implementation
// ---------------------------------------------------------------------------

impl LLMProvider for GeminiProvider {
    fn embed(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>> {
        let text = text.to_string();
        Box::pin(async move {
            let url = format!(
                "{}/v1beta/models/{}:embedContent",
                self.base_url, self.embedding_model,
            );

            let body = EmbedRequest {
                model: format!("models/{}", self.embedding_model),
                content: EmbedContent {
                    parts: vec![TextPart { text }],
                },
            };

            let response: EmbedResponse =
                post_with_retry(&self.client, &url, self.api_key.expose_secret(), &body).await?;
            Ok(response.embedding.values)
        })
    }

    fn summarize(
        &self,
        texts: &[String],
        max_tokens: usize,
    ) -> Pin<Box<dyn Future<Output = Result<String, CerememoryError>> + Send + '_>> {
        let joined = texts.join("\n\n");
        Box::pin(async move {
            let url = format!(
                "{}/v1beta/models/{}:generateContent",
                self.base_url, self.model
            );

            let prompt = format!(
                "Summarize the following texts into a concise summary. \
                 Keep it under {} tokens.\n\n{}",
                max_tokens, joined,
            );

            let body = GenerateRequest {
                contents: vec![GenerateContent {
                    parts: vec![TextPart { text: prompt }],
                }],
                generation_config: Some(GenerationConfig {
                    max_output_tokens: Some(max_tokens as u32),
                    response_mime_type: None,
                }),
            };

            let response: GenerateResponse =
                post_with_retry(&self.client, &url, self.api_key.expose_secret(), &body).await?;
            extract_text_from_response(&response)
        })
    }

    fn extract_relations(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ExtractedRelation>, CerememoryError>> + Send + '_>>
    {
        let text = text.to_string();
        Box::pin(async move {
            let url = format!(
                "{}/v1beta/models/{}:generateContent",
                self.base_url, self.model
            );

            let prompt = format!(
                "Extract semantic relations from the following text. \
                 Return a JSON array where each element has the fields: \
                 \"subject\" (string), \"predicate\" (string), \"object\" (string), \
                 \"confidence\" (number between 0 and 1). \
                 Return ONLY the JSON array, no other text.\n\n{}",
                text,
            );

            let body = GenerateRequest {
                contents: vec![GenerateContent {
                    parts: vec![TextPart { text: prompt }],
                }],
                generation_config: Some(GenerationConfig {
                    max_output_tokens: None,
                    response_mime_type: Some("application/json".to_string()),
                }),
            };

            let response: GenerateResponse =
                post_with_retry(&self.client, &url, self.api_key.expose_secret(), &body).await?;
            let raw_text = extract_text_from_response(&response)?;

            let relations = serde_json::from_str::<GeminiRelationsResponse>(&raw_text)
                .map_err(|e| {
                    CerememoryError::Internal(format!(
                        "Failed to parse Gemini relation extraction response: {e}"
                    ))
                })?
                .into_relations();

            Ok(relations)
        })
    }

    fn embed_image(
        &self,
        data: &[u8],
        format: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CerememoryError>> + Send + '_>> {
        let image_data = data.to_vec();
        let mime_type = normalize_image_mime_type(format).map(str::to_owned);

        Box::pin(async move {
            let mime_type = mime_type?;
            if image_data.is_empty() {
                return Err(CerememoryError::Validation(
                    "Image payload must not be empty".to_string(),
                ));
            }
            let b64 = base64::engine::general_purpose::STANDARD.encode(&image_data);
            // Step 1: Describe the image using Gemini generateContent with vision.
            let generate_url = format!(
                "{}/v1beta/models/{}:generateContent",
                self.base_url, self.model
            );

            let body = MultimodalGenerateRequest {
                contents: vec![MultimodalContent {
                    parts: vec![
                        MultimodalPart::Text {
                            text: "Describe this image in detail for memory indexing. \
                                   Include objects, actions, colors, text, and spatial relationships."
                                .to_string(),
                        },
                        MultimodalPart::InlineData {
                            inline_data: InlineData {
                                mime_type,
                                data: b64,
                            },
                        },
                    ],
                }],
                generation_config: Some(GenerationConfig {
                    max_output_tokens: Some(512),
                    response_mime_type: None,
                }),
            };

            let response: GenerateResponse = post_with_retry(
                &self.client,
                &generate_url,
                self.api_key.expose_secret(),
                &body,
            )
            .await?;
            let description = extract_text_from_response(&response)?;

            // Step 2: Embed the description using the text embedding model.
            self.embed(&description).await
        })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_embedding: true,
            image_embedding: true,
            audio_transcription: false,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// POST a JSON body to `url` with exponential-backoff retry on transient errors
/// (HTTP 429, 500, 502, 503).
async fn post_with_retry<Req, Res>(
    client: &Client,
    url: &str,
    api_key: &str,
    body: &Req,
) -> Result<Res, CerememoryError>
where
    Req: Serialize + Sync,
    Res: for<'de> Deserialize<'de>,
{
    let backoff = ExponentialBackoffBuilder::default()
        .with_initial_interval(Duration::from_millis(BACKOFF_INITIAL_INTERVAL_MS))
        .with_max_elapsed_time(Some(Duration::from_secs(30)))
        .build();

    let attempts = std::sync::atomic::AtomicU32::new(0);

    backoff::future::retry(backoff, || async {
        let attempt = attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if attempt > BACKOFF_MAX_RETRIES {
            return Err(backoff::Error::permanent(CerememoryError::Internal(
                "Gemini max retries exceeded".to_string(),
            )));
        }

        let resp = client
            .post(url)
            .header(GOOGLE_API_KEY_HEADER, api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                backoff::Error::permanent(CerememoryError::Internal(format!(
                    "Gemini request failed: {e}"
                )))
            })?;

        let status = resp.status();

        if status.is_success() {
            let parsed = resp.json::<Res>().await.map_err(|e| {
                backoff::Error::permanent(CerememoryError::Internal(format!(
                    "Failed to parse Gemini response: {e}"
                )))
            })?;
            return Ok(parsed);
        }

        let response_body = resp.text().await.unwrap_or_default();

        // Transient: retry on 429 (rate-limit) and 5xx server errors.
        if status.as_u16() == 429 || status.is_server_error() {
            warn!(
                status = status.as_u16(),
                body = %response_body,
                "Gemini transient error, will retry"
            );
            return Err(backoff::Error::transient(CerememoryError::Internal(
                format!("Gemini transient error {status}: {response_body}"),
            )));
        }

        // Permanent error — don't retry.
        debug!(
            status = status.as_u16(),
            body = %response_body,
            "Gemini permanent error"
        );
        Err(backoff::Error::permanent(CerememoryError::Internal(
            format!("Gemini error {status}: {response_body}"),
        )))
    })
    .await
}

/// Extract the first text part from a Gemini `generateContent` response.
fn extract_text_from_response(response: &GenerateResponse) -> Result<String, CerememoryError> {
    response
        .candidates
        .first()
        .and_then(|c| c.content.parts.first())
        .map(|p| p.text.clone())
        .ok_or_else(|| {
            CerememoryError::Internal("Gemini response contained no text content".to_string())
        })
}

// ---------------------------------------------------------------------------
// Gemini API DTOs (private)
// ---------------------------------------------------------------------------

// -- embedContent --

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    content: EmbedContent,
}

#[derive(Serialize)]
struct EmbedContent {
    parts: Vec<TextPart>,
}

#[derive(Serialize, Deserialize, Clone)]
struct TextPart {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: EmbeddingValues,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

// -- generateContent --

#[derive(Serialize)]
struct GenerateRequest {
    contents: Vec<GenerateContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct GenerateContent {
    parts: Vec<TextPart>,
}

#[derive(Serialize)]
struct GenerationConfig {
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GeminiRelationsResponse {
    Array(Vec<ExtractedRelation>),
    Wrapped { relations: Vec<ExtractedRelation> },
}

impl GeminiRelationsResponse {
    fn into_relations(self) -> Vec<ExtractedRelation> {
        match self {
            Self::Array(relations) => relations,
            Self::Wrapped { relations } => relations,
        }
    }
}

#[derive(Deserialize)]
struct Candidate {
    content: CandidateContent,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Vec<TextPart>,
}

// -- multimodal generateContent (mixed text + inline_data parts) --

/// A content part that can be either text or inline binary data.
#[derive(Serialize)]
#[serde(untagged)]
enum MultimodalPart {
    Text { text: String },
    InlineData { inline_data: InlineData },
}

#[derive(Serialize)]
struct InlineData {
    mime_type: String,
    data: String, // base64 encoded
}

#[derive(Serialize)]
struct MultimodalContent {
    parts: Vec<MultimodalPart>,
}

#[derive(Serialize)]
struct MultimodalGenerateRequest {
    contents: Vec<MultimodalContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Helper: create a `GeminiProvider` pointing at the wiremock server.
    fn provider_for(server: &MockServer) -> GeminiProvider {
        GeminiProvider::new(
            "test-key".to_string(),
            Some("gemini-2.0-flash".to_string()),
            Some(server.uri()),
        )
        .expect("Gemini provider should build")
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
    async fn test_embed_returns_vector() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let body = serde_json::json!({
            "embedding": {
                "values": [0.1, 0.2, 0.3, 0.4]
            }
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/text-embedding-004:embedContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let result = provider.embed("Hello, world!").await.unwrap();

        assert_eq!(result, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[tokio::test]
    async fn test_summarize_returns_text() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is a summary."
                    }]
                }
            }]
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/gemini-2.0-flash:generateContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let texts = vec!["First text.".to_string(), "Second text.".to_string()];
        let result = provider.summarize(&texts, 100).await.unwrap();

        assert_eq!(result, "This is a summary.");
    }

    #[tokio::test]
    async fn test_extract_relations_returns_triples() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let relations = serde_json::json!([
            {
                "subject": "Alice",
                "predicate": "knows",
                "object": "Bob",
                "confidence": 0.95
            },
            {
                "subject": "Bob",
                "predicate": "works_at",
                "object": "Acme",
                "confidence": 0.87
            }
        ]);

        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": serde_json::to_string(&relations).unwrap()
                    }]
                }
            }]
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/gemini-2.0-flash:generateContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let result = provider
            .extract_relations("Alice knows Bob. Bob works at Acme.")
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].subject, "Alice");
        assert_eq!(result[0].predicate, "knows");
        assert_eq!(result[0].object, "Bob");
        assert!((result[0].confidence - 0.95).abs() < f64::EPSILON);
        assert_eq!(result[1].subject, "Bob");
        assert_eq!(result[1].predicate, "works_at");
        assert_eq!(result[1].object, "Acme");
    }

    #[tokio::test]
    async fn test_extract_relations_accepts_wrapped_payload() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let relations = serde_json::json!({
            "relations": [
                {
                    "subject": "Alice",
                    "predicate": "knows",
                    "object": "Bob",
                    "confidence": 0.95
                }
            ]
        });

        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": serde_json::to_string(&relations).unwrap()
                    }]
                }
            }]
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/gemini-2.0-flash:generateContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let result = provider
            .extract_relations("Alice knows Bob.")
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].subject, "Alice");
        assert_eq!(result[0].object, "Bob");
    }

    #[test]
    fn test_capabilities() {
        let server_uri = "http://unused";
        let provider =
            GeminiProvider::new("test-key".to_string(), None, Some(server_uri.to_string()))
                .expect("Gemini provider should build");
        let caps = provider.capabilities();
        assert!(
            caps.text_embedding,
            "Gemini should advertise text_embedding"
        );
        assert!(
            caps.image_embedding,
            "Gemini should advertise image_embedding"
        );
        assert!(
            !caps.audio_transcription,
            "Gemini should not advertise audio_transcription"
        );
    }

    #[tokio::test]
    async fn test_embed_image_describes_then_embeds() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        // Mock generateContent (vision) — returns a text description.
        let generate_body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "A red apple on a wooden table."
                    }]
                }
            }]
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/gemini-2.0-flash:generateContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&generate_body))
            .expect(1)
            .mount(&server)
            .await;

        // Mock embedContent — returns an embedding vector.
        let embed_body = serde_json::json!({
            "embedding": {
                "values": [0.5, 0.6, 0.7]
            }
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/text-embedding-004:embedContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        // A trivial 1-pixel PNG-like payload (content doesn't matter for the mock).
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47];
        let result = provider.embed_image(&fake_image, "png").await;

        assert!(result.is_ok(), "embed_image failed: {:?}", result.err());
        assert_eq!(result.unwrap(), vec![0.5, 0.6, 0.7]);
    }

    #[tokio::test]
    async fn test_embed_image_accepts_mime_type() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let generate_body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "A blue square."
                    }]
                }
            }]
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/gemini-2.0-flash:generateContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&generate_body))
            .expect(1)
            .mount(&server)
            .await;

        let embed_body = serde_json::json!({
            "embedding": {
                "values": [0.9, 0.8, 0.7]
            }
        });

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/text-embedding-004:embedContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47];
        let result = provider.embed_image(&fake_image, "image/png").await;

        assert_eq!(result.unwrap(), vec![0.9, 0.8, 0.7]);
    }

    #[tokio::test]
    async fn test_embed_image_unsupported_format() {
        let Some(server) = start_mock_server().await else {
            return;
        };
        let provider = provider_for(&server);

        let result = provider.embed_image(b"data", "bmp").await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unsupported image format"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn test_429_retry_succeeds() {
        let Some(server) = start_mock_server().await else {
            return;
        };

        let success_body = serde_json::json!({
            "embedding": {
                "values": [1.0, 2.0, 3.0]
            }
        });

        // First call returns 429, second succeeds.
        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/text-embedding-004:embedContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path_regex(
                r"/v1beta/models/text-embedding-004:embedContent",
            ))
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&success_body))
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let result = provider.embed("retry test").await.unwrap();

        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
}
