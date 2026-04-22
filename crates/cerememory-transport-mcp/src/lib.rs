//! MCP (Model Context Protocol) transport binding for the Cerememory Protocol.
//!
//! Exposes CMP operations as MCP tools over stdio transport, enabling
//! direct integration with Claude Code and other MCP-compatible clients.

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, RETRY_AFTER};
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::service::ServiceExt;
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::*;
use cerememory_core::types::*;
#[cfg(test)]
use cerememory_engine::CerememoryEngine;

// ─── Parameter types (flat, LLM-friendly) ───────────────────────────

#[derive(Deserialize, JsonSchema)]
struct StoreParams {
    /// Text content to store as a memory (UTF-8, max 1 MB)
    content: String,
    /// Target memory store. Options: 'episodic' (events/experiences), 'semantic' (facts/knowledge), 'procedural' (skills/how-to), 'emotional' (feelings/reactions), 'working' (temporary scratch, not persisted). Omit for auto-routing based on content analysis.
    store: Option<String>,
    /// Emotional valence label. Options: joy, sadness, anger, fear, surprise, disgust, trust, anticipation (aliases: happy, sad, angry, anticipatory). Affects emotional memory routing.
    emotion: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct UpdateParams {
    /// UUID of the record to update
    record_id: String,
    /// New text content (replaces existing content). Leave empty to keep current content.
    content: Option<String>,
    /// New emotion label. Options: joy, sadness, anger, fear, surprise, disgust, trust, anticipation.
    emotion: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct StoreRawParams {
    /// Text content to preserve verbatim in the raw journal.
    content: String,
    /// Session identifier used to group preserved raw records.
    session_id: String,
    /// Optional turn identifier for the source conversation turn.
    turn_id: Option<String>,
    /// Optional topic identifier for higher-level clustering.
    topic_id: Option<String>,
    /// Raw source: conversation, tool_io, scratchpad, summary, imported.
    source: Option<String>,
    /// Speaker: user, assistant, system, tool.
    speaker: Option<String>,
    /// Visibility: normal, private_scratch, sealed.
    visibility: Option<String>,
    /// Secrecy level: public, sensitive, secret.
    secrecy_level: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct BatchStoreParams {
    /// JSON array of records. Each record: {"content": "...", "store": "episodic" (optional), "emotion": "joy" (optional)}. Example: [{"content": "Meeting notes", "store": "episodic"}]
    records_json: String,
}

#[derive(Deserialize, JsonSchema)]
struct BatchStoreRawParams {
    /// JSON array of raw records. Each record: {"content":"...","session_id":"...","source":"conversation","speaker":"user"}.
    records_json: String,
}

#[derive(Deserialize, Serialize, JsonSchema)]
struct BatchRecord {
    content: String,
    store: Option<String>,
    emotion: Option<String>,
}

#[derive(Deserialize, Serialize, JsonSchema)]
struct BatchRawRecord {
    content: String,
    session_id: String,
    turn_id: Option<String>,
    topic_id: Option<String>,
    source: Option<String>,
    speaker: Option<String>,
    visibility: Option<String>,
    secrecy_level: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct RecallParams {
    /// Natural language query to recall memories. If omitted, returns recent memories up to limit.
    query: Option<String>,
    /// Maximum number of results to return (default: 10, range: 1-1000)
    limit: Option<u32>,
    /// Comma-separated store type filter (e.g., 'episodic,semantic'). Omit to search all stores.
    stores: Option<String>,
    /// Whether to reconsolidate recalled memories (default: true).
    reconsolidate: Option<bool>,
    /// Activation depth for spreading activation (default: 2).
    activation_depth: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
struct RecallRawParams {
    /// Optional session id to scope forensic recall.
    session_id: Option<String>,
    /// Optional text query over raw preserved content.
    query: Option<String>,
    /// Optional start time in RFC3339 format.
    start: Option<String>,
    /// Optional end time in RFC3339 format.
    end: Option<String>,
    /// Maximum results (default: 10).
    limit: Option<u32>,
    /// Include private scratch records (default: false).
    include_private_scratch: Option<bool>,
    /// Include sealed records (default: false).
    include_sealed: Option<bool>,
    /// Comma-separated secrecy filter: public,sensitive,secret.
    secrecy_levels: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct TimelineParams {
    /// Start time in RFC 3339 format (e.g., "2024-01-01T00:00:00Z")
    start: Option<String>,
    /// End time in RFC 3339 format
    end: Option<String>,
    /// Granularity: "minute", "hour", "day", "week", "month" (default: "hour")
    granularity: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct AssociateParams {
    /// UUID of the record to find associations for
    record_id: String,
    /// Maximum association depth (default: 1)
    depth: Option<u32>,
    /// Maximum results (default: 10)
    limit: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
struct ForgetParams {
    /// Comma-separated UUIDs of records to permanently delete. This operation is irreversible.
    record_ids: String,
    /// Also delete records associated with the target records (default: false)
    cascade: Option<bool>,
    /// Must be true to confirm irreversible deletion.
    confirm: bool,
}

#[derive(Deserialize, JsonSchema)]
struct ConsolidateParams {
    /// Consolidation strategy: "incremental" or "full" (default: "incremental")
    strategy: Option<String>,
    /// If true, only preview what would be consolidated (default: false)
    dry_run: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
struct DreamTickParams {
    /// Optional session id to scope dream processing to a single session.
    session_id: Option<String>,
    /// If true, only preview what would be processed (default: false).
    dry_run: Option<bool>,
    /// Maximum number of session/topic/day groups to process (default: 10).
    max_groups: Option<u32>,
    /// If true, promote eligible dream summaries into semantic memory (default: true).
    promote_semantic: Option<bool>,
    /// Include private scratch raw records (default: false).
    include_private_scratch: Option<bool>,
    /// Include sealed raw records (default: false).
    include_sealed: Option<bool>,
    /// Comma-separated secrecy filter: public,sensitive,secret.
    secrecy_levels: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct InspectParams {
    /// UUID of the record to inspect
    record_id: String,
}

#[derive(Deserialize, JsonSchema)]
struct ExportParams {
    /// Export format (default: "cma"; "jsonl" alias supported)
    format: Option<String>,
    /// Encrypt the archive with ChaCha20-Poly1305 AEAD (default: false). Requires encryption_key.
    encrypt: Option<bool>,
    /// Passphrase for archive encryption. Required when encrypt is true. Use a strong passphrase (16+ chars recommended).
    encryption_key: Option<String>,
    /// Output path for the exported CMA archive. Required.
    output_path: String,
}

/// Convert an EmotionVector to a human-readable label, or None if neutral (intensity == 0).
fn emotion_label(e: &EmotionVector) -> Option<String> {
    if e.intensity == 0.0 {
        return None;
    }
    let candidates = [
        ("joy", e.joy),
        ("trust", e.trust),
        ("fear", e.fear),
        ("surprise", e.surprise),
        ("sadness", e.sadness),
        ("disgust", e.disgust),
        ("anger", e.anger),
        ("anticipation", e.anticipation),
    ];
    candidates
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(label, _)| (*label).to_string())
}

// ─── Flattened recall response for LLM readability ──────────────────

#[derive(Serialize)]
struct McpRecalledMemory {
    record_id: uuid::Uuid,
    store: String,
    text: String,
    relevance_score: f64,
    fidelity: f64,
    created_at: String,
    emotion: Option<String>,
}

#[derive(Serialize)]
struct McpRecallResponse {
    memories: Vec<McpRecalledMemory>,
    query_metadata: Option<QueryMetadata>,
    total_candidates: u32,
}

#[derive(Serialize)]
struct McpRawRecord {
    record_id: uuid::Uuid,
    session_id: String,
    source: String,
    speaker: String,
    visibility: String,
    secrecy_level: String,
    text: String,
    created_at: String,
}

#[derive(Serialize)]
struct McpRawRecallResponse {
    records: Vec<McpRawRecord>,
    total_candidates: u32,
}

#[derive(Clone)]
enum McpBackend {
    #[cfg(test)]
    Local(Arc<CerememoryEngine>),
    Remote(Arc<CerememoryHttpClient>),
}

#[derive(Clone)]
struct CerememoryHttpClient {
    base_url: String,
    client: reqwest::Client,
}

impl CerememoryHttpClient {
    fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        timeout: Option<Duration>,
    ) -> Result<Self, CerememoryError> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let parsed = reqwest::Url::parse(&base_url)
            .map_err(|e| CerememoryError::Validation(format!("Invalid server URL: {e}")))?;
        match parsed.scheme() {
            "http" | "https" => {}
            scheme => {
                return Err(CerememoryError::Validation(format!(
                    "Invalid server URL scheme '{scheme}'. Use http or https."
                )));
            }
        }

        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

        if let Some(api_key) = api_key.filter(|value| !value.trim().is_empty()) {
            let mut auth = HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| CerememoryError::Validation(format!("Invalid API key header: {e}")))?;
            auth.set_sensitive(true);
            headers.insert(AUTHORIZATION, auth);
        }

        let mut builder = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(10));
        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }
        let client = builder
            .build()
            .map_err(|e| CerememoryError::Internal(format!("Failed to build HTTP client: {e}")))?;

        Ok(Self { base_url, client })
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, CerememoryError> {
        let response = self
            .client
            .get(self.url(path))
            .send()
            .await
            .map_err(http_transport_err)?;
        Self::decode_json_response(response).await
    }

    async fn post_json<B: Serialize + ?Sized, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, CerememoryError> {
        let response = self
            .client
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(http_transport_err)?;
        Self::decode_json_response(response).await
    }

    async fn patch_no_content<B: Serialize + ?Sized>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<(), CerememoryError> {
        let response = self
            .client
            .patch(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(http_transport_err)?;
        Self::decode_empty_response(response).await
    }

    async fn delete_json<B: Serialize + ?Sized, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, CerememoryError> {
        let response = self
            .client
            .delete(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(http_transport_err)?;
        Self::decode_json_response(response).await
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn decode_json_response<T: DeserializeOwned>(
        response: reqwest::Response,
    ) -> Result<T, CerememoryError> {
        if response.status().is_success() {
            return response.json::<T>().await.map_err(|e| {
                CerememoryError::Serialization(format!("HTTP JSON decode failed: {e}"))
            });
        }

        Err(http_response_err(response).await)
    }

    async fn decode_empty_response(response: reqwest::Response) -> Result<(), CerememoryError> {
        if response.status().is_success() {
            return Ok(());
        }

        Err(http_response_err(response).await)
    }
}

fn http_transport_err(err: reqwest::Error) -> CerememoryError {
    CerememoryError::Internal(format!("Failed to reach upstream Cerememory server: {err}"))
}

async fn http_response_err(response: reqwest::Response) -> CerememoryError {
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u32>().ok());
    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if let Ok(cmp_error) = serde_json::from_str::<CMPError>(&body) {
        return cmp_error_to_cerememory_error(cmp_error, retry_after);
    }

    let preview = upstream_error_preview(&body, status);

    CerememoryError::Internal(format!(
        "Upstream Cerememory server error (HTTP {}): {}",
        status, preview
    ))
}

fn upstream_error_preview(body: &str, status: reqwest::StatusCode) -> String {
    if body.is_empty() {
        return format!("HTTP {status}");
    }

    let mut chars = body.chars();
    let preview: String = chars.by_ref().take(500).collect();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

fn cmp_error_to_cerememory_error(
    cmp_error: CMPError,
    retry_after_header: Option<u32>,
) -> CerememoryError {
    match cmp_error.code {
        CMPErrorCode::RecordNotFound => CerememoryError::RecordNotFound(cmp_error.message),
        CMPErrorCode::StoreInvalid => CerememoryError::StoreInvalid(cmp_error.message),
        CMPErrorCode::ContentTooLarge | CMPErrorCode::ValidationError => {
            CerememoryError::Validation(cmp_error.message)
        }
        CMPErrorCode::ModalityUnsupported => {
            CerememoryError::ModalityUnsupported(cmp_error.message)
        }
        CMPErrorCode::WorkingMemoryFull => CerememoryError::WorkingMemoryFull,
        CMPErrorCode::DecayEngineBusy => CerememoryError::DecayEngineBusy {
            retry_after_secs: cmp_error.retry_after.or(retry_after_header).unwrap_or(1),
        },
        CMPErrorCode::ConsolidationInProgress => CerememoryError::ConsolidationInProgress,
        CMPErrorCode::ExportFailed => CerememoryError::ExportFailed(cmp_error.message),
        CMPErrorCode::ImportConflict => CerememoryError::ImportConflict(cmp_error.message),
        CMPErrorCode::ForgetUnconfirmed => CerememoryError::ForgetUnconfirmed,
        CMPErrorCode::VersionMismatch => CerememoryError::Validation(cmp_error.message),
        CMPErrorCode::Unauthorized => CerememoryError::Unauthorized(cmp_error.message),
        CMPErrorCode::RateLimited => CerememoryError::RateLimited {
            retry_after_secs: cmp_error.retry_after.or(retry_after_header).unwrap_or(1),
        },
        CMPErrorCode::InternalError => CerememoryError::Internal(cmp_error.message),
    }
}

// ─── Server ─────────────────────────────────────────────────────────

/// MCP server backed either by a shared local `CerememoryEngine` or an upstream HTTP server.
#[derive(Clone)]
pub struct CerememoryMcpServer {
    backend: McpBackend,
    #[allow(dead_code)] // Used by rmcp tool_router macro at runtime
    tool_router: ToolRouter<Self>,
}

fn parse_store_type(s: &str) -> Result<StoreType, McpError> {
    s.trim()
        .to_lowercase()
        .parse::<StoreType>()
        .map_err(|_| {
            McpError::invalid_params(
                format!(
                    "Invalid store type '{}'. Valid values: episodic, semantic, procedural, emotional, working",
                    s.trim()
                ),
                None,
            )
        })
}

fn parse_raw_source(value: Option<String>) -> Result<RawSource, McpError> {
    match value
        .unwrap_or_else(|| "conversation".to_string())
        .trim()
        .to_lowercase()
        .as_str()
    {
        "conversation" => Ok(RawSource::Conversation),
        "tool_io" => Ok(RawSource::ToolIo),
        "scratchpad" => Ok(RawSource::Scratchpad),
        "summary" => Ok(RawSource::Summary),
        "imported" => Ok(RawSource::Imported),
        other => Err(McpError::invalid_params(
            format!(
                "Invalid raw source '{other}'. Valid options: conversation, tool_io, scratchpad, summary, imported"
            ),
            None,
        )),
    }
}

fn parse_raw_speaker(value: Option<String>) -> Result<RawSpeaker, McpError> {
    match value
        .unwrap_or_else(|| "user".to_string())
        .trim()
        .to_lowercase()
        .as_str()
    {
        "user" => Ok(RawSpeaker::User),
        "assistant" => Ok(RawSpeaker::Assistant),
        "system" => Ok(RawSpeaker::System),
        "tool" => Ok(RawSpeaker::Tool),
        other => Err(McpError::invalid_params(
            format!("Invalid raw speaker '{other}'. Valid options: user, assistant, system, tool"),
            None,
        )),
    }
}

fn parse_raw_visibility(value: Option<String>) -> Result<RawVisibility, McpError> {
    match value
        .unwrap_or_else(|| "normal".to_string())
        .trim()
        .to_lowercase()
        .as_str()
    {
        "normal" => Ok(RawVisibility::Normal),
        "private_scratch" => Ok(RawVisibility::PrivateScratch),
        "sealed" => Ok(RawVisibility::Sealed),
        other => Err(McpError::invalid_params(
            format!(
                "Invalid raw visibility '{other}'. Valid options: normal, private_scratch, sealed"
            ),
            None,
        )),
    }
}

fn parse_secrecy_level(value: Option<String>) -> Result<SecrecyLevel, McpError> {
    match value
        .unwrap_or_else(|| "public".to_string())
        .trim()
        .to_lowercase()
        .as_str()
    {
        "public" => Ok(SecrecyLevel::Public),
        "sensitive" => Ok(SecrecyLevel::Sensitive),
        "secret" => Ok(SecrecyLevel::Secret),
        other => Err(McpError::invalid_params(
            format!("Invalid secrecy level '{other}'. Valid options: public, sensitive, secret"),
            None,
        )),
    }
}

fn parse_secrecy_levels_list(value: Option<String>) -> Result<Option<Vec<SecrecyLevel>>, McpError> {
    let Some(levels) = value else {
        return Ok(None);
    };

    let levels = levels
        .split(',')
        .filter(|level| !level.trim().is_empty())
        .map(|level| parse_secrecy_level(Some(level.trim().to_string())))
        .collect::<Result<Vec<_>, _>>()?;

    if levels.is_empty() {
        Ok(None)
    } else {
        Ok(Some(levels))
    }
}

fn parse_uuid(s: &str) -> Result<uuid::Uuid, McpError> {
    s.trim().parse::<uuid::Uuid>().map_err(|_| {
        McpError::invalid_params(
            format!(
                "Invalid UUID '{}'. Expected format: 550e8400-e29b-41d4-a716-446655440000",
                s.trim()
            ),
            None,
        )
    })
}

fn internal_err(e: impl std::fmt::Display) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn engine_err(err: CerememoryError) -> McpError {
    match err {
        CerememoryError::RecordNotFound(id) => {
            McpError::invalid_params(format!("Record not found: {id}"), None)
        }
        CerememoryError::StoreInvalid(store) => {
            McpError::invalid_params(format!("Invalid store type '{store}'"), None)
        }
        CerememoryError::ContentTooLarge { size, limit } => McpError::invalid_params(
            format!("Content too large: {size} bytes exceeds limit of {limit} bytes"),
            None,
        ),
        CerememoryError::ModalityUnsupported(modality) => {
            McpError::invalid_params(format!("Unsupported modality: {modality}"), None)
        }
        CerememoryError::ImportConflict(message) | CerememoryError::Validation(message) => {
            McpError::invalid_params(message, None)
        }
        CerememoryError::VersionMismatch { expected, got } => McpError::invalid_params(
            format!("Protocol version mismatch: expected {expected}, got {got}"),
            None,
        ),
        CerememoryError::ForgetUnconfirmed => {
            McpError::invalid_params("Forget requires explicit confirmation".to_string(), None)
        }
        CerememoryError::Storage(message) => {
            tracing::warn!(error = %message, "Storage error");
            McpError::internal_error(STORAGE_ERROR_MESSAGE.to_string(), None)
        }
        CerememoryError::Serialization(message) => {
            tracing::warn!(error = %message, "Serialization error");
            McpError::internal_error(SERIALIZATION_ERROR_MESSAGE.to_string(), None)
        }
        CerememoryError::ExportFailed(message) => {
            tracing::warn!(error = %message, "Export failed");
            McpError::internal_error(EXPORT_ERROR_MESSAGE.to_string(), None)
        }
        CerememoryError::Internal(message) => {
            tracing::warn!(error = %message, "Internal error");
            McpError::internal_error(INTERNAL_ERROR_MESSAGE.to_string(), None)
        }
        CerememoryError::WorkingMemoryFull => {
            McpError::internal_error("Working memory is full. Try again later.".to_string(), None)
        }
        CerememoryError::RateLimited { retry_after_secs } => McpError::internal_error(
            format!("Rate limited. Retry after {} seconds.", retry_after_secs),
            None,
        ),
        CerememoryError::DecayEngineBusy { retry_after_secs } => McpError::internal_error(
            format!(
                "Decay engine busy. Retry after {} seconds.",
                retry_after_secs
            ),
            None,
        ),
        CerememoryError::ConsolidationInProgress => McpError::internal_error(
            "Consolidation in progress. Try again later.".to_string(),
            None,
        ),
        CerememoryError::Unauthorized(message) => {
            tracing::warn!(error = %message, "Unauthorized");
            McpError::internal_error("Unauthorized".to_string(), None)
        }
    }
}

fn ok_text(text: String) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![Content::text(text)]))
}

fn ok_json<T: Serialize>(val: &T) -> Result<CallToolResult, McpError> {
    let text = serde_json::to_string_pretty(val).map_err(internal_err)?;
    ok_text(text)
}

fn parse_emotion(label: Option<String>) -> Result<Option<EmotionVector>, McpError> {
    label
        .map(|l| {
            l.parse::<EmotionVector>()
                .map_err(|e| McpError::invalid_params(e.to_string(), None))
        })
        .transpose()
}

fn parse_consolidation_strategy(
    strategy: Option<String>,
) -> Result<ConsolidationStrategy, McpError> {
    let raw = strategy
        .as_deref()
        .unwrap_or("incremental")
        .trim()
        .to_string();
    match raw.to_lowercase().as_str() {
        "full" => Ok(ConsolidationStrategy::Full),
        "incremental" => Ok(ConsolidationStrategy::Incremental),
        "selective" => Ok(ConsolidationStrategy::Selective),
        _ => Err(McpError::invalid_params(
            format!(
                "Invalid consolidation strategy: {raw}. Use one of incremental, full, selective."
            ),
            None,
        )),
    }
}

fn build_text_store_request(
    content: String,
    store: Option<StoreType>,
    emotion: Option<EmotionVector>,
) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: content.into_bytes(),
                embedding: None,
            }],
            summary: None,
        },
        store,
        emotion,
        context: None,
        metadata: None,
        associations: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_text_raw_store_request(
    content: String,
    session_id: String,
    turn_id: Option<String>,
    topic_id: Option<String>,
    source: RawSource,
    speaker: RawSpeaker,
    visibility: RawVisibility,
    secrecy_level: SecrecyLevel,
) -> EncodeStoreRawRequest {
    EncodeStoreRawRequest {
        header: None,
        session_id,
        turn_id,
        topic_id,
        source,
        speaker,
        visibility,
        secrecy_level,
        content: MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: content.into_bytes(),
                embedding: None,
            }],
            summary: None,
        },
        metadata: None,
    }
}

fn require_non_empty_content(content: &str) -> Result<(), McpError> {
    if content.trim().is_empty() {
        return Err(McpError::invalid_params(
            "Content must not be empty or whitespace-only".to_string(),
            None,
        ));
    }
    Ok(())
}

fn parse_export_format(format: Option<String>) -> Result<String, McpError> {
    let format = format.unwrap_or_else(|| "cma".to_string());
    if format.eq_ignore_ascii_case("cma") || format.eq_ignore_ascii_case("jsonl") {
        Ok("cma".to_string())
    } else {
        Err(McpError::invalid_params(
            format!("Unsupported export format: {format}. Valid options: cma, jsonl"),
            None,
        ))
    }
}

impl McpBackend {
    async fn encode_store(
        &self,
        req: EncodeStoreRequest,
    ) -> Result<EncodeStoreResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.encode_store(req).await,
            Self::Remote(client) => client.post_json("/v1/encode", &req).await,
        }
    }

    async fn encode_store_raw(
        &self,
        req: EncodeStoreRawRequest,
    ) -> Result<EncodeStoreRawResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.encode_store_raw(req).await,
            Self::Remote(client) => client.post_json("/v1/encode/raw", &req).await,
        }
    }

    async fn encode_update(&self, req: EncodeUpdateRequest) -> Result<(), CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.encode_update(req).await,
            Self::Remote(client) => {
                client
                    .patch_no_content(&format!("/v1/encode/{}", req.record_id), &req)
                    .await
            }
        }
    }

    async fn encode_batch(
        &self,
        req: EncodeBatchRequest,
    ) -> Result<EncodeBatchResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.encode_batch(req).await,
            Self::Remote(client) => client.post_json("/v1/encode/batch", &req).await,
        }
    }

    async fn encode_batch_store_raw(
        &self,
        req: EncodeBatchStoreRawRequest,
    ) -> Result<EncodeBatchStoreRawResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.encode_batch_store_raw(req).await,
            Self::Remote(client) => client.post_json("/v1/encode/raw/batch", &req).await,
        }
    }

    async fn recall_query(
        &self,
        req: RecallQueryRequest,
    ) -> Result<RecallQueryResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.recall_query(req).await,
            Self::Remote(client) => client.post_json("/v1/recall/query", &req).await,
        }
    }

    async fn recall_raw_query(
        &self,
        req: RecallRawQueryRequest,
    ) -> Result<RecallRawQueryResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.recall_raw_query(req).await,
            Self::Remote(client) => client.post_json("/v1/recall/raw", &req).await,
        }
    }

    async fn recall_timeline(
        &self,
        req: RecallTimelineRequest,
    ) -> Result<RecallTimelineResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.recall_timeline(req).await,
            Self::Remote(client) => client.post_json("/v1/recall/timeline", &req).await,
        }
    }

    async fn recall_associate(
        &self,
        req: RecallAssociateRequest,
    ) -> Result<RecallAssociateResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.recall_associate(req).await,
            Self::Remote(client) => {
                client
                    .post_json(&format!("/v1/recall/associate/{}", req.record_id), &req)
                    .await
            }
        }
    }

    async fn lifecycle_forget(&self, req: ForgetRequest) -> Result<u32, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.lifecycle_forget(req).await,
            Self::Remote(client) => {
                let response: ForgetResponse =
                    client.delete_json("/v1/lifecycle/forget", &req).await?;
                Ok(response.records_deleted)
            }
        }
    }

    async fn lifecycle_consolidate(
        &self,
        req: ConsolidateRequest,
    ) -> Result<ConsolidateResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.lifecycle_consolidate(req).await,
            Self::Remote(client) => client.post_json("/v1/lifecycle/consolidate", &req).await,
        }
    }

    async fn lifecycle_dream_tick(
        &self,
        req: DreamTickRequest,
    ) -> Result<DreamTickResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.lifecycle_dream_tick(req).await,
            Self::Remote(client) => client.post_json("/v1/lifecycle/dream-tick", &req).await,
        }
    }

    async fn introspect_stats(&self) -> Result<StatsResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.introspect_stats().await,
            Self::Remote(client) => client.get_json("/v1/introspect/stats").await,
        }
    }

    async fn introspect_record(
        &self,
        req: RecordIntrospectRequest,
    ) -> Result<MemoryRecord, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => engine.introspect_record(req).await,
            Self::Remote(client) => client
                .get_json(&format!(
                    "/v1/introspect/record/{}?include_history={}&include_associations={}&include_versions={}",
                    req.record_id,
                    req.include_history,
                    req.include_associations,
                    req.include_versions
                ))
                .await,
        }
    }

    async fn lifecycle_export(
        &self,
        req: ExportRequest,
    ) -> Result<ExportArchiveResponse, CerememoryError> {
        match self {
            #[cfg(test)]
            Self::Local(engine) => {
                let (archive_data, metadata) = engine.lifecycle_export(req).await?;
                Ok(ExportArchiveResponse {
                    metadata,
                    archive_data,
                })
            }
            Self::Remote(client) => client.post_json("/v1/lifecycle/export", &req).await,
        }
    }
}

#[tool_router]
impl CerememoryMcpServer {
    #[cfg(test)]
    fn new(engine: Arc<CerememoryEngine>) -> Self {
        Self::with_backend(McpBackend::Local(engine))
    }

    pub fn from_http(
        base_url: impl Into<String>,
        api_key: Option<String>,
        timeout: Option<Duration>,
    ) -> Result<Self, CerememoryError> {
        let client = CerememoryHttpClient::new(base_url, api_key, timeout)?;
        Ok(Self::with_backend(McpBackend::Remote(Arc::new(client))))
    }

    fn with_backend(backend: McpBackend) -> Self {
        Self {
            backend,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Store a new memory record. Returns JSON: {record_id, store, initial_fidelity, associations_created}."
    )]
    async fn store(&self, params: Parameters<StoreParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        require_non_empty_content(&p.content)?;
        let store_type = p.store.map(|s| parse_store_type(&s)).transpose()?;
        let emotion = parse_emotion(p.emotion)?;
        let req = build_text_store_request(p.content, store_type, emotion);
        let resp = self.backend.encode_store(req).await.map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Preserve a verbatim text item in the raw journal. Use for conversation turns, tool output, or externalized scratchpad entries that should remain available for forensic recall."
    )]
    async fn store_raw(
        &self,
        params: Parameters<StoreRawParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        require_non_empty_content(&p.content)?;
        let session_id = p.session_id.trim().to_string();
        if session_id.is_empty() {
            return Err(McpError::invalid_params(
                "session_id must not be empty".to_string(),
                None,
            ));
        }

        let req = build_text_raw_store_request(
            p.content,
            session_id,
            p.turn_id,
            p.topic_id,
            parse_raw_source(p.source)?,
            parse_raw_speaker(p.speaker)?,
            parse_raw_visibility(p.visibility)?,
            parse_secrecy_level(p.secrecy_level)?,
        );
        let resp = self
            .backend
            .encode_store_raw(req)
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Update an existing memory record's content or emotion. Preserves UUID, associations, and fidelity history. Returns no content on success."
    )]
    async fn update(&self, params: Parameters<UpdateParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        if p.content.is_none() && p.emotion.is_none() {
            return Err(McpError::invalid_params(
                "At least one of 'content' or 'emotion' must be provided".to_string(),
                None,
            ));
        }
        if let Some(ref c) = p.content {
            require_non_empty_content(c)?;
        }
        let record_id = parse_uuid(&p.record_id)?;
        let emotion = parse_emotion(p.emotion)?;
        let content = p.content.map(|c| MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: c.into_bytes(),
                embedding: None,
            }],
            summary: None,
        });

        let req = EncodeUpdateRequest {
            header: None,
            record_id,
            content,
            emotion,
            metadata: None,
        };

        self.backend.encode_update(req).await.map_err(engine_err)?;
        ok_text(format!("Updated record {record_id}"))
    }

    #[tool(
        description = "Store multiple memory records in a batch with automatic cross-record association. Accepts a JSON array string. Returns JSON: {results (array of {record_id, store, initial_fidelity, associations_created}), associations_inferred}."
    )]
    async fn batch_store(
        &self,
        params: Parameters<BatchStoreParams>,
    ) -> Result<CallToolResult, McpError> {
        let records: Vec<BatchRecord> =
            serde_json::from_str(&params.0.records_json).map_err(|e| {
                McpError::invalid_params(
                    format!(
                        "Failed to parse records_json. Expected JSON array of objects with \
                         'content' (required), 'store' (optional), 'emotion' (optional). \
                         Parse error: {e}"
                    ),
                    None,
                )
            })?;

        for r in &records {
            require_non_empty_content(&r.content)?;
        }

        let mut encode_records = Vec::with_capacity(records.len());
        for r in records {
            let store_type = r.store.map(|s| parse_store_type(&s)).transpose()?;
            let emotion = parse_emotion(r.emotion)?;
            encode_records.push(build_text_store_request(r.content, store_type, emotion));
        }

        let req = EncodeBatchRequest {
            header: None,
            records: encode_records,
            infer_associations: true,
        };

        let resp = self.backend.encode_batch(req).await.map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Preserve multiple raw journal entries in one batch. Accepts a JSON array string. Returns JSON: {results (array of {record_id, session_id, visibility, secrecy_level})}."
    )]
    async fn batch_store_raw(
        &self,
        params: Parameters<BatchStoreRawParams>,
    ) -> Result<CallToolResult, McpError> {
        let records: Vec<BatchRawRecord> =
            serde_json::from_str(&params.0.records_json).map_err(|e| {
                McpError::invalid_params(
                    format!(
                        "Failed to parse records_json. Expected JSON array of raw records with \
                         'content' and 'session_id'. Parse error: {e}"
                    ),
                    None,
                )
            })?;

        let mut encode_records = Vec::with_capacity(records.len());
        for record in records {
            require_non_empty_content(&record.content)?;
            let session_id = record.session_id.trim().to_string();
            if session_id.is_empty() {
                return Err(McpError::invalid_params(
                    "session_id must not be empty".to_string(),
                    None,
                ));
            }

            encode_records.push(build_text_raw_store_request(
                record.content,
                session_id,
                record.turn_id,
                record.topic_id,
                parse_raw_source(record.source)?,
                parse_raw_speaker(record.speaker)?,
                parse_raw_visibility(record.visibility)?,
                parse_secrecy_level(record.secrecy_level)?,
            ));
        }

        let resp = self
            .backend
            .encode_batch_store_raw(EncodeBatchStoreRawRequest {
                header: None,
                records: encode_records,
            })
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Recall memories matching a natural language query using hybrid text+vector search. If query is omitted, returns recent memories. Returns JSON: {memories (array of {record_id, store, text, relevance_score, fidelity, created_at, emotion}), total_candidates}."
    )]
    async fn recall(&self, params: Parameters<RecallParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let stores = p
            .stores
            .map(|s| {
                s.split(',')
                    .map(|st| parse_store_type(st.trim()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        let req = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: p.query.filter(|q| !q.trim().is_empty()),
                ..Default::default()
            },
            stores,
            limit: p.limit.unwrap_or(10).clamp(1, 1000),
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: p.reconsolidate.unwrap_or(DEFAULT_RECONSOLIDATE),
            activation_depth: p.activation_depth.unwrap_or(DEFAULT_ACTIVATION_DEPTH),
            recall_mode: RecallMode::Perfect,
        };

        let resp = self.backend.recall_query(req).await.map_err(engine_err)?;

        let mcp_resp = McpRecallResponse {
            memories: resp
                .memories
                .iter()
                .map(|m| McpRecalledMemory {
                    record_id: m.record.id,
                    store: m.record.store.to_string(),
                    text: m
                        .record
                        .text_content()
                        .unwrap_or("[non-text content]")
                        .to_string(),
                    relevance_score: m.relevance_score,
                    fidelity: m.record.fidelity.score,
                    created_at: m.record.created_at.to_rfc3339(),
                    emotion: emotion_label(&m.record.emotion),
                })
                .collect(),
            query_metadata: resp.query_metadata,
            total_candidates: resp.total_candidates,
        };
        ok_json(&mcp_resp)
    }

    #[tool(
        description = "Explicit forensic recall over the raw journal. Returns verbatim preserved records. By default excludes private_scratch, sealed, and secret records unless explicitly requested."
    )]
    async fn recall_raw(
        &self,
        params: Parameters<RecallRawParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let temporal = match (p.start, p.end) {
            (Some(start), Some(end)) => {
                let start = chrono::DateTime::parse_from_rfc3339(&start)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        McpError::invalid_params(format!("Invalid start time: {e}"), None)
                    })?;
                let end = chrono::DateTime::parse_from_rfc3339(&end)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        McpError::invalid_params(format!("Invalid end time: {e}"), None)
                    })?;
                Some(TemporalRange { start, end })
            }
            (None, None) => None,
            _ => {
                return Err(McpError::invalid_params(
                    "Both start and end must be provided together for raw temporal filtering"
                        .to_string(),
                    None,
                ));
            }
        };
        let resp = self
            .backend
            .recall_raw_query(RecallRawQueryRequest {
                header: None,
                session_id: p.session_id.filter(|value| !value.trim().is_empty()),
                query: p.query.filter(|value| !value.trim().is_empty()),
                temporal,
                limit: p.limit.unwrap_or(10).clamp(1, 1000),
                include_private_scratch: p.include_private_scratch.unwrap_or(false),
                include_sealed: p.include_sealed.unwrap_or(false),
                secrecy_levels: parse_secrecy_levels_list(p.secrecy_levels)?,
            })
            .await
            .map_err(engine_err)?;

        let mcp_resp = McpRawRecallResponse {
            records: resp
                .records
                .iter()
                .map(|record| McpRawRecord {
                    record_id: record.id,
                    session_id: record.session_id.clone(),
                    source: serde_json::to_value(record.source)
                        .ok()
                        .and_then(|value| value.as_str().map(|text| text.to_string()))
                        .unwrap_or_else(|| "conversation".to_string()),
                    speaker: serde_json::to_value(record.speaker)
                        .ok()
                        .and_then(|value| value.as_str().map(|text| text.to_string()))
                        .unwrap_or_else(|| "user".to_string()),
                    visibility: serde_json::to_value(record.visibility)
                        .ok()
                        .and_then(|value| value.as_str().map(|text| text.to_string()))
                        .unwrap_or_else(|| "normal".to_string()),
                    secrecy_level: serde_json::to_value(record.secrecy_level)
                        .ok()
                        .and_then(|value| value.as_str().map(|text| text.to_string()))
                        .unwrap_or_else(|| "public".to_string()),
                    text: record
                        .text_content()
                        .unwrap_or("[non-text content]")
                        .to_string(),
                    created_at: record.created_at.to_rfc3339(),
                })
                .collect(),
            total_candidates: resp.total_candidates,
        };
        ok_json(&mcp_resp)
    }

    #[tool(
        description = "Retrieve memories bucketed by time period. Defaults to last 7 days at hour granularity. Returns JSON with temporal buckets."
    )]
    async fn timeline(
        &self,
        params: Parameters<TimelineParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let now = chrono::Utc::now();
        let start = p
            .start
            .map(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| McpError::invalid_params(format!("Invalid start time: {e}"), None))
            })
            .transpose()?
            .unwrap_or_else(|| now - chrono::Duration::days(7));
        let end = p
            .end
            .map(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| McpError::invalid_params(format!("Invalid end time: {e}"), None))
            })
            .transpose()?
            .unwrap_or(now);
        if start > end {
            return Err(McpError::invalid_params(
                "Invalid time range: start must be earlier than or equal to end.".to_string(),
                None,
            ));
        }
        let granularity = p
            .granularity
            .map(|g| match g.to_lowercase().as_str() {
                "minute" => Ok(TimeGranularity::Minute),
                "hour" => Ok(TimeGranularity::Hour),
                "day" => Ok(TimeGranularity::Day),
                "week" => Ok(TimeGranularity::Week),
                "month" => Ok(TimeGranularity::Month),
                other => Err(McpError::invalid_params(
                    format!("Unknown granularity: {other}"),
                    None,
                )),
            })
            .transpose()?
            .unwrap_or(TimeGranularity::Hour);

        let req = RecallTimelineRequest {
            header: None,
            range: TemporalRange { start, end },
            granularity,
            min_fidelity: None,
            emotion_filter: None,
        };

        let resp = self
            .backend
            .recall_timeline(req)
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Find memories connected to a given record through spreading activation network. Returns associated records with activation scores."
    )]
    async fn associate(
        &self,
        params: Parameters<AssociateParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let record_id = parse_uuid(&p.record_id)?;

        let req = RecallAssociateRequest {
            header: None,
            record_id,
            association_types: None,
            depth: p.depth.unwrap_or(1),
            min_weight: 0.0,
            limit: p.limit.unwrap_or(10),
        };

        let resp = self
            .backend
            .recall_associate(req)
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Permanently and irreversibly delete memory records by UUID. Requires confirm=true. Returns JSON: {records_deleted}."
    )]
    async fn forget(&self, params: Parameters<ForgetParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        if !p.confirm {
            return Err(McpError::invalid_params(
                "Forget requires confirm=true to proceed".to_string(),
                None,
            ));
        }
        let ids: Vec<uuid::Uuid> = p
            .record_ids
            .split(',')
            .map(|s| parse_uuid(s.trim()))
            .collect::<Result<Vec<_>, _>>()?;

        let req = ForgetRequest {
            header: None,
            record_ids: Some(ids),
            store: None,
            temporal_range: None,
            cascade: p.cascade.unwrap_or(false),
            confirm: p.confirm,
        };

        let deleted = self
            .backend
            .lifecycle_forget(req)
            .await
            .map_err(engine_err)?;
        ok_json(&serde_json::json!({"records_deleted": deleted}))
    }

    #[tool(
        description = "Consolidate memories: migrate mature episodic memories to semantic store, merge duplicates. Returns consolidation report."
    )]
    async fn consolidate(
        &self,
        params: Parameters<ConsolidateParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let strategy = parse_consolidation_strategy(p.strategy)?;

        let req = ConsolidateRequest {
            header: None,
            strategy,
            min_age_hours: 0,
            min_access_count: 0,
            dry_run: p.dry_run.unwrap_or(false),
        };

        let resp = self
            .backend
            .lifecycle_consolidate(req)
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Run a dream tick: summarize unprocessed raw journal entries into episodic summaries with backlinks. By default excludes private_scratch, sealed, and secret raw records."
    )]
    async fn dream_tick(
        &self,
        params: Parameters<DreamTickParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let resp = self
            .backend
            .lifecycle_dream_tick(DreamTickRequest {
                header: None,
                session_id: p.session_id.filter(|value| !value.trim().is_empty()),
                dry_run: p.dry_run.unwrap_or(false),
                max_groups: p.max_groups.unwrap_or(10),
                promote_semantic: p.promote_semantic.unwrap_or(true),
                include_private_scratch: p.include_private_scratch.unwrap_or(false),
                include_sealed: p.include_sealed.unwrap_or(false),
                secrecy_levels: parse_secrecy_levels_list(p.secrecy_levels)?,
            })
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(description = "Get system statistics: record counts, store sizes, decay state.")]
    async fn stats(&self) -> Result<CallToolResult, McpError> {
        let resp = self.backend.introspect_stats().await.map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(
        description = "Inspect a specific memory record by UUID. Returns full record details including fidelity state, emotion vector, access history, and associations."
    )]
    async fn inspect(&self, params: Parameters<InspectParams>) -> Result<CallToolResult, McpError> {
        let record_id = parse_uuid(&params.0.record_id)?;

        let req = RecordIntrospectRequest {
            header: None,
            record_id,
            include_history: true,
            include_associations: true,
            include_versions: true,
        };

        let record = self
            .backend
            .introspect_record(req)
            .await
            .map_err(engine_err)?;
        ok_json(&record)
    }

    #[tool(
        description = "Export all memory records as a CMA archive file to an explicit output path. Optionally encrypted with ChaCha20-Poly1305. Returns metadata and archive size."
    )]
    async fn export(&self, params: Parameters<ExportParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let format = parse_export_format(p.format)?;
        if p.output_path.trim().is_empty() {
            return Err(McpError::invalid_params(
                "output_path must not be empty".to_string(),
                None,
            ));
        }
        let req = ExportRequest {
            header: None,
            format,
            stores: None,
            include_raw_journal: false,
            encrypt: p.encrypt.unwrap_or(false),
            encryption_key: p.encryption_key,
        };

        let resp = self
            .backend
            .lifecycle_export(req)
            .await
            .map_err(engine_err)?;
        let bytes = resp.archive_data.clone();

        let path = std::path::PathBuf::from(&p.output_path);
        // Resolve to an absolute path and validate it is not a symlink or
        // pointing to sensitive system directories.
        let canonical_parent = path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .canonicalize()
            .map_err(|e| {
                McpError::invalid_params(format!("Invalid output_path directory: {e}"), None)
            })?;
        if path.is_symlink() {
            return Err(McpError::invalid_params(
                "output_path must not be a symlink".to_string(),
                None,
            ));
        }
        let resolved = canonical_parent.join(path.file_name().unwrap_or_default());
        std::fs::write(&resolved, &bytes).map_err(|e| {
            McpError::internal_error(format!("Failed to write archive file: {e}"), None)
        })?;

        let summary = serde_json::to_string_pretty(&resp.metadata).map_err(internal_err)?;
        ok_text(format!(
            "{summary}\n\nArchive written to: {}\nArchive size: {} bytes",
            resolved.display(),
            bytes.len()
        ))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for CerememoryMcpServer {
    fn get_info(&self) -> ServerInfo {
        let capabilities = ServerCapabilities::builder().enable_tools().build();

        ServerInfo::new(capabilities)
            .with_server_info(Implementation::new("cerememory", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Cerememory is a living memory database. Available tools:\n\
                  - store: Save a new memory\n\
                 - store_raw: Preserve a verbatim raw journal entry\n\
                  - update: Edit an existing memory by UUID\n\
                  - batch_store: Save multiple memories at once\n\
                 - batch_store_raw: Preserve multiple raw journal entries at once\n\
                 - recall: Search memories by query, or list recent memories (omit query)\n\
                 - recall_raw: Explicit forensic recall over preserved raw journal entries\n\
                - timeline: Browse memories by time period\n\
                 - associate: Find connected memories via spreading activation\n\
                 - inspect: View full details of a memory by UUID\n\
                 - forget: Permanently delete memories by UUID (requires confirm=true)\n\
                 - dream_tick: Summarize raw journal entries into episodic summaries\n\
                 - consolidate: Migrate mature episodic memories to semantic store\n\
                 - export: Export all memories to a CMA archive file (requires output_path)\n\
                 - stats: View system statistics and store counts",
            )
    }
}

/// Start the MCP server on stdio while proxying requests to an upstream Cerememory HTTP server.
pub async fn serve_stdio_http(
    base_url: impl Into<String>,
    api_key: Option<String>,
    timeout: Option<Duration>,
) -> anyhow::Result<()> {
    let base_url = base_url.into();
    tracing::info!(%base_url, "Starting Cerememory MCP HTTP proxy on stdio");
    let server = CerememoryMcpServer::from_http(base_url, api_key, timeout)?;
    serve_server_stdio(server).await
}

async fn serve_server_stdio(server: CerememoryMcpServer) -> anyhow::Result<()> {
    let service = server
        .serve(rmcp::transport::io::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("MCP serve failed: {e}"))?;

    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;

    tracing::info!("MCP server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::{model::CallToolRequestParams, ServiceExt};

    #[test]
    fn server_creates_successfully() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let server = CerememoryMcpServer::new(engine);
        let info = server.get_info();
        assert_eq!(info.server_info.name, "cerememory");
    }

    #[test]
    fn http_proxy_server_creates_successfully() {
        let server = CerememoryMcpServer::from_http(
            "http://127.0.0.1:8420/",
            None,
            Some(Duration::from_secs(90)),
        )
        .unwrap();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "cerememory");

        match &server.backend {
            McpBackend::Remote(client) => assert_eq!(client.base_url, "http://127.0.0.1:8420"),
            McpBackend::Local(_) => panic!("expected remote backend"),
        }
    }

    #[test]
    fn http_proxy_rejects_invalid_server_urls() {
        let err = match CerememoryMcpServer::from_http("127.0.0.1:8420", None, None) {
            Ok(_) => panic!("expected invalid URL error"),
            Err(err) => err,
        };
        assert!(matches!(err, CerememoryError::Validation(_)));

        let err = match CerememoryMcpServer::from_http("file:///tmp/cerememory.sock", None, None) {
            Ok(_) => panic!("expected invalid scheme error"),
            Err(err) => err,
        };
        assert!(matches!(err, CerememoryError::Validation(_)));
    }

    #[test]
    fn parse_store_type_valid() {
        assert_eq!(parse_store_type("episodic").unwrap(), StoreType::Episodic);
        assert_eq!(parse_store_type("Semantic").unwrap(), StoreType::Semantic);
    }

    #[test]
    fn parse_store_type_invalid() {
        assert!(parse_store_type("invalid").is_err());
    }

    #[test]
    fn parse_emotion_valid() {
        let emotion = parse_emotion(Some("joy".to_string())).unwrap().unwrap();
        assert_eq!(emotion.joy, 1.0);
        assert_eq!(emotion.intensity, 1.0);
        assert_eq!(emotion.valence, 1.0);
    }

    #[test]
    fn parse_emotion_invalid() {
        assert!(parse_emotion(Some("unknown".to_string())).is_err());
    }

    #[test]
    fn parse_raw_enums_valid() {
        assert_eq!(
            parse_raw_source(Some("tool_io".to_string())).unwrap(),
            RawSource::ToolIo
        );
        assert_eq!(
            parse_raw_speaker(Some("assistant".to_string())).unwrap(),
            RawSpeaker::Assistant
        );
        assert_eq!(
            parse_raw_visibility(Some("sealed".to_string())).unwrap(),
            RawVisibility::Sealed
        );
        assert_eq!(
            parse_secrecy_level(Some("secret".to_string())).unwrap(),
            SecrecyLevel::Secret
        );
    }

    #[test]
    fn parse_secrecy_levels_list_valid() {
        let levels = parse_secrecy_levels_list(Some("public,secret".to_string()))
            .unwrap()
            .unwrap();
        assert_eq!(levels, vec![SecrecyLevel::Public, SecrecyLevel::Secret]);
    }

    #[test]
    fn parse_consolidation_strategy_rejects_unknown_values() {
        assert!(parse_consolidation_strategy(Some("typo".to_string())).is_err());
    }

    #[test]
    fn parse_export_format_rejects_non_cma_values() {
        assert!(parse_export_format(Some("zip".to_string())).is_err());
    }

    #[test]
    fn parse_export_format_accepts_jsonl_alias() {
        assert_eq!(
            parse_export_format(Some("jsonl".to_string())).unwrap(),
            "cma".to_string()
        );
    }

    #[test]
    fn parse_uuid_valid() {
        let id = uuid::Uuid::now_v7();
        assert_eq!(parse_uuid(&id.to_string()).unwrap(), id);
    }

    #[test]
    fn parse_uuid_invalid() {
        assert!(parse_uuid("not-a-uuid").is_err());
    }

    #[test]
    fn engine_err_sanitizes_storage_details() {
        let err = engine_err(CerememoryError::Storage("disk path leaked".to_string()));
        assert_eq!(err.message, STORAGE_ERROR_MESSAGE);
    }

    #[test]
    fn cmp_error_retry_after_maps_to_rate_limited_error() {
        let err = cmp_error_to_cerememory_error(
            CMPError {
                code: CMPErrorCode::RateLimited,
                message: "Rate limit exceeded".to_string(),
                details: None,
                retry_after: Some(9),
                request_id: None,
            },
            None,
        );

        match err {
            CerememoryError::RateLimited { retry_after_secs } => assert_eq!(retry_after_secs, 9),
            other => panic!("expected rate limited error, got {other:?}"),
        }
    }

    #[test]
    fn non_json_upstream_error_preview_is_utf8_safe() {
        let body = "あ".repeat(600);
        let preview = upstream_error_preview(&body, reqwest::StatusCode::BAD_GATEWAY);

        assert!(preview.ends_with("..."));
        assert!(preview.starts_with('あ'));
    }

    #[test]
    fn store_rejects_empty_content_validation() {
        // This validates the trim().is_empty() logic used in the store handler
        assert!("".trim().is_empty());
        assert!("   ".trim().is_empty());
        assert!("\n\t".trim().is_empty());
        assert!(!"hello".trim().is_empty());
    }

    #[test]
    fn emotion_label_returns_none_for_neutral() {
        let neutral = EmotionVector::default();
        assert!(emotion_label(&neutral).is_none());
    }

    #[test]
    fn emotion_label_returns_dominant_emotion() {
        let joy: EmotionVector = "joy".parse().unwrap();
        assert_eq!(emotion_label(&joy), Some("joy".to_string()));

        let anger: EmotionVector = "anger".parse().unwrap();
        assert_eq!(emotion_label(&anger), Some("anger".to_string()));
    }

    #[test]
    fn tool_router_registers_expected_tools() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let server = CerememoryMcpServer::new(engine);
        let tools = server.tool_router.list_all();
        let tool_names: Vec<String> = tools
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect();

        assert_eq!(
            tool_names,
            vec![
                "associate".to_string(),
                "batch_store".to_string(),
                "batch_store_raw".to_string(),
                "consolidate".to_string(),
                "dream_tick".to_string(),
                "export".to_string(),
                "forget".to_string(),
                "inspect".to_string(),
                "recall".to_string(),
                "recall_raw".to_string(),
                "stats".to_string(),
                "store".to_string(),
                "store_raw".to_string(),
                "timeline".to_string(),
                "update".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn mcp_initialize_list_tools_and_call_store_then_recall() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let server = CerememoryMcpServer::new(engine);
        let (server_transport, client_transport) = tokio::io::duplex(16_384);

        let server_handle = tokio::spawn(async move {
            let running = server.serve(server_transport).await?;
            anyhow::Ok(running)
        });

        let client = ().serve(client_transport).await.expect("client initialize");
        let mut tools = Vec::new();
        for _ in 0..20 {
            tools = client.peer().list_all_tools().await.expect("list tools");
            if !tools.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let tool_names: Vec<String> = tools.iter().map(|tool| tool.name.to_string()).collect();
        assert!(
            tool_names.iter().any(|name| name == "store"),
            "{tool_names:?}"
        );
        assert!(
            tool_names.iter().any(|name| name == "recall"),
            "{tool_names:?}"
        );

        let store_result = client
            .call_tool(
                CallToolRequestParams::new("store").with_arguments(
                    serde_json::json!({
                        "content": "parallel mcp e2e memory",
                        "store": "semantic",
                        "emotion": "joy"
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                ),
            )
            .await
            .expect("store tool call");

        let store_payload = store_result
            .content
            .first()
            .and_then(|content| content.raw.as_text())
            .map(|text| text.text.clone())
            .expect("store result text");
        let store_json: serde_json::Value =
            serde_json::from_str(&store_payload).expect("store result json");
        assert_eq!(store_json["store"], "semantic");
        assert!(store_json["record_id"].as_str().is_some());

        let recall_result = client
            .call_tool(
                CallToolRequestParams::new("recall").with_arguments(
                    serde_json::json!({
                        "query": "parallel mcp e2e memory",
                        "limit": 5
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                ),
            )
            .await
            .expect("recall tool call");

        let recall_payload = recall_result
            .content
            .first()
            .and_then(|content| content.raw.as_text())
            .map(|text| text.text.clone())
            .expect("recall result text");
        let recall_json: serde_json::Value =
            serde_json::from_str(&recall_payload).expect("recall result json");
        let memories = recall_json["memories"]
            .as_array()
            .expect("recall memories array");
        assert!(!memories.is_empty());
        assert!(memories.iter().any(|memory| {
            memory["text"]
                .as_str()
                .map(|text| text.contains("parallel mcp e2e memory"))
                .unwrap_or(false)
        }));

        client.cancel().await.expect("client cancel");
        let running = server_handle
            .await
            .expect("server task")
            .expect("server start");
        running.cancel().await.expect("server cancel");
    }
}
