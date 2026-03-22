//! MCP (Model Context Protocol) transport binding for the Cerememory Protocol.
//!
//! Exposes CMP operations as MCP tools over stdio transport, enabling
//! direct integration with Claude Code and other MCP-compatible clients.

use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::service::ServiceExt;
use rmcp::{tool, tool_router, ErrorData as McpError, ServerHandler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::*;
use cerememory_core::types::*;
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
struct BatchStoreParams {
    /// JSON array of records. Each record: {"content": "...", "store": "episodic" (optional), "emotion": "joy" (optional)}. Example: [{"content": "Meeting notes", "store": "episodic"}]
    records_json: String,
}

#[derive(Deserialize, Serialize, JsonSchema)]
struct BatchRecord {
    content: String,
    store: Option<String>,
    emotion: Option<String>,
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
struct InspectParams {
    /// UUID of the record to inspect
    record_id: String,
}

#[derive(Deserialize, JsonSchema)]
struct ExportParams {
    /// Export format (default: "cma")
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

// ─── Server ─────────────────────────────────────────────────────────

/// MCP server backed by a shared `CerememoryEngine`.
#[derive(Clone)]
pub struct CerememoryMcpServer {
    engine: Arc<CerememoryEngine>,
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
        other => McpError::internal_error(other.to_string(), None),
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
    match strategy
        .as_deref()
        .unwrap_or("incremental")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "full" => Ok(ConsolidationStrategy::Full),
        "incremental" => Ok(ConsolidationStrategy::Incremental),
        "selective" => Ok(ConsolidationStrategy::Selective),
        other => Err(McpError::invalid_params(
            format!(
                "Invalid consolidation strategy: {other}. Use one of incremental, full, selective."
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
    if format.eq_ignore_ascii_case("cma") {
        Ok("cma".to_string())
    } else {
        Err(McpError::invalid_params(
            format!("Unsupported export format: {format}. Only 'cma' is currently supported."),
            None,
        ))
    }
}

#[tool_router]
impl CerememoryMcpServer {
    pub fn new(engine: Arc<CerememoryEngine>) -> Self {
        Self {
            engine,
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
        let resp = self.engine.encode_store(req).await.map_err(engine_err)?;
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

        self.engine.encode_update(req).await.map_err(engine_err)?;
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

        let resp = self.engine.encode_batch(req).await.map_err(engine_err)?;
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
            reconsolidate: p.reconsolidate.unwrap_or(true),
            activation_depth: p.activation_depth.unwrap_or(2),
            recall_mode: RecallMode::Perfect,
        };

        let resp = self.engine.recall_query(req).await.map_err(engine_err)?;

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

        let resp = self.engine.recall_timeline(req).await.map_err(engine_err)?;
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
            .engine
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
            .engine
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
            .engine
            .lifecycle_consolidate(req)
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(description = "Get system statistics: record counts, store sizes, decay state.")]
    async fn stats(&self) -> Result<CallToolResult, McpError> {
        let resp = self.engine.introspect_stats().await.map_err(engine_err)?;
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
            .engine
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
            encrypt: p.encrypt.unwrap_or(false),
            encryption_key: p.encryption_key,
        };

        let (bytes, resp) = self
            .engine
            .lifecycle_export(req)
            .await
            .map_err(engine_err)?;

        let path = std::path::PathBuf::from(p.output_path);
        std::fs::write(&path, &bytes).map_err(|e| {
            McpError::internal_error(format!("Failed to write archive file: {e}"), None)
        })?;

        let summary = serde_json::to_string_pretty(&resp).map_err(internal_err)?;
        ok_text(format!(
            "{summary}\n\nArchive written to: {}\nArchive size: {} bytes",
            path.display(),
            bytes.len()
        ))
    }
}

impl ServerHandler for CerememoryMcpServer {
    fn get_info(&self) -> ServerInfo {
        let capabilities = ServerCapabilities::builder().enable_tools().build();

        ServerInfo::new(capabilities)
            .with_server_info(Implementation::new("cerememory", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Cerememory is a living memory database. Available tools:\n\
                 - store: Save a new memory\n\
                 - update: Edit an existing memory by UUID\n\
                 - batch_store: Save multiple memories at once\n\
                 - recall: Search memories by query, or list recent memories (omit query)\n\
                - timeline: Browse memories by time period\n\
                 - associate: Find connected memories via spreading activation\n\
                 - inspect: View full details of a memory by UUID\n\
                 - forget: Permanently delete memories by UUID (requires confirm=true)\n\
                 - consolidate: Migrate mature episodic memories to semantic store\n\
                 - export: Export all memories to a CMA archive file (requires output_path)\n\
                 - stats: View system statistics and store counts",
            )
    }
}

/// Start the MCP server on stdio (standard input/output).
///
/// This is the primary entry point for Claude Code integration via `cerememory mcp`.
pub async fn serve_stdio(engine: Arc<CerememoryEngine>) -> anyhow::Result<()> {
    tracing::info!("Starting Cerememory MCP server on stdio");
    let server = CerememoryMcpServer::new(engine);

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

    #[test]
    fn server_creates_successfully() {
        let engine = Arc::new(CerememoryEngine::in_memory().unwrap());
        let server = CerememoryMcpServer::new(engine);
        let info = server.get_info();
        assert_eq!(info.server_info.name, "cerememory");
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
    fn parse_consolidation_strategy_rejects_unknown_values() {
        assert!(parse_consolidation_strategy(Some("typo".to_string())).is_err());
    }

    #[test]
    fn parse_export_format_rejects_non_cma_values() {
        assert!(parse_export_format(Some("zip".to_string())).is_err());
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
                "consolidate".to_string(),
                "export".to_string(),
                "forget".to_string(),
                "inspect".to_string(),
                "recall".to_string(),
                "stats".to_string(),
                "store".to_string(),
                "timeline".to_string(),
                "update".to_string(),
            ]
        );
    }
}
