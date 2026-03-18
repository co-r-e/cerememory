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

use cerememory_core::protocol::*;
use cerememory_core::types::*;
use cerememory_engine::CerememoryEngine;

// ─── Parameter types (flat, LLM-friendly) ───────────────────────────

#[derive(Deserialize, JsonSchema)]
struct StoreParams {
    /// Text content to store as a memory
    content: String,
    /// Target store: episodic, semantic, procedural, emotional, working (default: auto-route)
    store: Option<String>,
    /// Emotional valence label (e.g., "joy", "sadness")
    emotion: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct BatchStoreParams {
    /// JSON array of store records, each with "content", optional "store", optional "emotion"
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
    /// Natural language query to recall memories
    query: String,
    /// Maximum number of results (default: 10)
    limit: Option<u32>,
    /// Comma-separated store filters (e.g., "episodic,semantic")
    stores: Option<String>,
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
    /// Comma-separated UUIDs of records to delete
    record_ids: String,
    /// Also delete associated records (default: false)
    cascade: Option<bool>,
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
    /// Encrypt the export (default: false)
    encrypt: Option<bool>,
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
    s.to_lowercase()
        .parse::<StoreType>()
        .map_err(|e| McpError::invalid_params(format!("Invalid store type: {e}"), None))
}

fn parse_uuid(s: &str) -> Result<uuid::Uuid, McpError> {
    s.trim()
        .parse::<uuid::Uuid>()
        .map_err(|e| McpError::invalid_params(format!("Invalid UUID: {e}"), None))
}

fn engine_err(e: impl std::fmt::Display) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn ok_text(text: String) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![Content::text(text)]))
}

fn ok_json<T: Serialize>(val: &T) -> Result<CallToolResult, McpError> {
    let text = serde_json::to_string_pretty(val).map_err(engine_err)?;
    ok_text(text)
}

fn parse_emotion(label: Option<String>) -> Option<EmotionVector> {
    label.map(|_l| EmotionVector::default())
}

#[tool_router]
impl CerememoryMcpServer {
    pub fn new(engine: Arc<CerememoryEngine>) -> Self {
        Self {
            engine,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Store a new memory record. Returns the record ID, store type, and initial fidelity.")]
    async fn store(&self, params: Parameters<StoreParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let store_type = p.store.map(|s| parse_store_type(&s)).transpose()?;

        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: p.content.into_bytes(),
                    embedding: None,
                }],
                summary: None,
            },
            store: store_type,
            emotion: parse_emotion(p.emotion),
            context: None,
            associations: None,
        };

        let resp = self.engine.encode_store(req).await.map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(description = "Store multiple memory records in a batch. Accepts a JSON array string of records.")]
    async fn batch_store(
        &self,
        params: Parameters<BatchStoreParams>,
    ) -> Result<CallToolResult, McpError> {
        let records: Vec<BatchRecord> =
            serde_json::from_str(&params.0.records_json).map_err(|e| {
                McpError::invalid_params(format!("Invalid records_json: {e}"), None)
            })?;

        let mut encode_records = Vec::with_capacity(records.len());
        for r in records {
            let store_type = r.store.map(|s| parse_store_type(&s)).transpose()?;
            encode_records.push(EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Text,
                        format: "text/plain".to_string(),
                        data: r.content.into_bytes(),
                        embedding: None,
                    }],
                    summary: None,
                },
                store: store_type,
                emotion: parse_emotion(r.emotion),
                context: None,
                associations: None,
            });
        }

        let req = EncodeBatchRequest {
            header: None,
            records: encode_records,
            infer_associations: true,
        };

        let resp = self.engine.encode_batch(req).await.map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(description = "Recall memories matching a natural language query. Returns ranked results with relevance scores.")]
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
                text: Some(p.query),
                ..Default::default()
            },
            stores,
            limit: p.limit.unwrap_or(10),
            min_fidelity: None,
            include_decayed: false,
            reconsolidate: true,
            activation_depth: 2,
            recall_mode: RecallMode::Perfect,
        };

        let resp = self.engine.recall_query(req).await.map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(description = "Retrieve memories along a timeline. Filter by start/end time and granularity.")]
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
            .engine
            .recall_timeline(req)
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(description = "Find memories associated with a given record through spreading activation.")]
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

    #[tool(description = "Permanently delete memory records. Provide comma-separated UUIDs.")]
    async fn forget(&self, params: Parameters<ForgetParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
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
            confirm: true,
        };

        let deleted = self
            .engine
            .lifecycle_forget(req)
            .await
            .map_err(engine_err)?;
        ok_text(format!("Deleted {deleted} record(s)"))
    }

    #[tool(description = "Consolidate memories (migrate episodic to semantic, merge duplicates).")]
    async fn consolidate(
        &self,
        params: Parameters<ConsolidateParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let strategy = match p
            .strategy
            .as_deref()
            .unwrap_or("incremental")
            .to_lowercase()
            .as_str()
        {
            "full" => ConsolidationStrategy::Full,
            _ => ConsolidationStrategy::Incremental,
        };

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
        let resp = self
            .engine
            .introspect_stats()
            .await
            .map_err(engine_err)?;
        ok_json(&resp)
    }

    #[tool(description = "Inspect a specific memory record by UUID. Returns full details including history.")]
    async fn inspect(
        &self,
        params: Parameters<InspectParams>,
    ) -> Result<CallToolResult, McpError> {
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

    #[tool(description = "Export all memory records as a CMA archive. Returns metadata and archive size.")]
    async fn export(&self, params: Parameters<ExportParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let req = ExportRequest {
            header: None,
            format: p.format.unwrap_or_else(|| "cma".to_string()),
            stores: None,
            encrypt: p.encrypt.unwrap_or(false),
            encryption_key: None,
        };

        let (bytes, resp) = self
            .engine
            .lifecycle_export(req)
            .await
            .map_err(engine_err)?;

        let summary = serde_json::to_string_pretty(&resp).map_err(engine_err)?;
        ok_text(format!("{summary}\n\nArchive size: {} bytes", bytes.len()))
    }
}

impl ServerHandler for CerememoryMcpServer {
    fn get_info(&self) -> ServerInfo {
        let capabilities = ServerCapabilities::builder()
            .enable_tools()
            .build();

        ServerInfo::new(capabilities)
            .with_server_info(Implementation::new(
                "cerememory",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Cerememory is a living memory database. Use 'store' to save memories, \
                 'recall' to search, 'timeline' for temporal browsing, 'associate' for \
                 spreading activation, 'forget' to delete, and 'stats' for system overview.",
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
        assert_eq!(
            parse_store_type("episodic").unwrap(),
            StoreType::Episodic
        );
        assert_eq!(
            parse_store_type("Semantic").unwrap(),
            StoreType::Semantic
        );
    }

    #[test]
    fn parse_store_type_invalid() {
        assert!(parse_store_type("invalid").is_err());
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
}
