//! Cerememory CLI — command-line interface for the living memory database.

use std::io::{IsTerminal, Write};
use std::net::ToSocketAddrs;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use cerememory_config::ServerConfig;
use cerememory_core::protocol::*;
use cerememory_core::types::*;
use cerememory_engine::CerememoryEngine;

#[derive(Parser)]
#[command(
    name = "cerememory",
    about = "A living memory database for the age of AI",
    version
)]
struct Cli {
    /// Data directory for persistent storage
    #[arg(long, global = true)]
    data_dir: Option<String>,

    /// Path to configuration TOML file
    #[arg(long, global = true)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP server (and optionally a gRPC server)
    Serve {
        /// HTTP port to listen on
        #[arg(long)]
        port: Option<u16>,

        /// gRPC port (disabled if not specified)
        #[arg(long)]
        grpc_port: Option<u16>,
    },

    /// Consolidate episodic memory into semantic memory.
    Consolidate {
        /// Consolidation strategy: incremental, full, or selective.
        #[arg(long, default_value = "incremental")]
        strategy: String,

        /// Minimum record age in hours before consolidation.
        #[arg(long, default_value_t = 0)]
        min_age_hours: u32,

        /// Minimum access count before consolidation.
        #[arg(long, default_value_t = 0)]
        min_access_count: u32,

        /// Preview the work without mutating state.
        #[arg(long)]
        dry_run: bool,
    },

    /// Change the recall mode used by stateful operations.
    SetMode {
        /// Recall mode: human or perfect.
        mode: String,

        /// Optional comma-separated store filter.
        #[arg(long)]
        scope: Option<String>,
    },

    /// Store a new memory record
    Store {
        /// Text content to store
        text: String,

        /// Target store (episodic, semantic, procedural, emotional, working)
        #[arg(long, default_value = "episodic")]
        store: String,

        /// Comma-separated embedding vector (e.g., "0.1,0.2,0.3")
        #[arg(long)]
        embedding: Option<String>,
    },

    /// Recall memories by query
    Recall {
        /// Text query
        query: String,

        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: u32,

        /// Recall mode (human or perfect)
        #[arg(long, default_value = "human")]
        mode: String,
    },

    /// Inspect a specific record
    Inspect {
        /// Record UUID
        record_id: Uuid,
    },

    /// Show system statistics
    Stats,

    /// Run a decay tick
    DecayTick {
        /// Simulated duration in seconds
        #[arg(long, default_value = "3600")]
        duration: u32,
    },

    /// Forget (delete) a record
    Forget {
        /// Record UUID to forget
        record_id: Uuid,

        /// Also delete associated records
        #[arg(long)]
        cascade: bool,

        /// Required confirmation flag
        #[arg(long)]
        confirm: bool,
    },

    /// Export all records to a CMA archive file
    Export {
        /// Output file path
        #[arg(long, default_value = "./backup.cma")]
        output: String,

        /// Archive format.
        #[arg(long, default_value = "cma")]
        format: String,

        /// Comma-separated store filter.
        #[arg(long)]
        stores: Option<String>,

        /// Encrypt the exported archive.
        #[arg(long)]
        encrypt: bool,

        /// Passphrase for archive encryption.
        #[arg(long)]
        encryption_key: Option<String>,

        /// Read the archive encryption passphrase from stdin.
        #[arg(long)]
        encryption_key_stdin: bool,
    },

    /// Health check (for Docker HEALTHCHECK / K8s liveness probe)
    Healthcheck {
        /// HTTP port to check (default: 8420)
        #[arg(long, default_value = "8420")]
        port: u16,
    },

    /// Import records from a CMA archive file
    Import {
        /// Path to the CMA archive file
        path: String,

        /// Decryption passphrase for encrypted archives.
        #[arg(long)]
        decryption_key: Option<String>,

        /// Read the archive decryption passphrase from stdin.
        #[arg(long)]
        decryption_key_stdin: bool,

        /// Conflict resolution: keep_existing, keep_imported, keep_newer.
        #[arg(long, default_value = "keep_newer")]
        conflict_resolution: String,
    },

    /// Start the MCP (Model Context Protocol) server on stdio
    Mcp,
}

fn parse_store_type(s: &str) -> Result<StoreType> {
    s.to_lowercase()
        .parse::<StoreType>()
        .map_err(|_| {
            anyhow::anyhow!(
                "Invalid store type '{}'. Valid values: episodic, semantic, procedural, emotional, working",
                s
            )
        })
}

fn parse_embedding(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|v| v.trim().parse::<f32>().context("Invalid embedding value"))
        .collect()
}

fn parse_store_list(value: &str) -> Result<Vec<StoreType>> {
    value
        .split(',')
        .map(|item| {
            item.trim().parse::<StoreType>().map_err(|_| {
                anyhow::anyhow!(
                    "Invalid store type '{}'. Valid values: episodic, semantic, procedural, emotional, working",
                    item.trim()
                )
            })
        })
        .collect()
}

fn parse_consolidation_strategy(value: &str) -> Result<ConsolidationStrategy> {
    match value.trim().to_lowercase().as_str() {
        "incremental" => Ok(ConsolidationStrategy::Incremental),
        "full" => Ok(ConsolidationStrategy::Full),
        "selective" => Ok(ConsolidationStrategy::Selective),
        other => anyhow::bail!(
            "Invalid consolidation strategy '{}'. Valid options: incremental, full, selective",
            other
        ),
    }
}

fn parse_conflict_resolution(value: &str) -> Result<ConflictResolution> {
    match value.trim().to_lowercase().as_str() {
        "keep_existing" => Ok(ConflictResolution::KeepExisting),
        "keep_imported" => Ok(ConflictResolution::KeepImported),
        "keep_newer" => Ok(ConflictResolution::KeepNewer),
        other => anyhow::bail!(
            "Invalid conflict resolution '{}'. Valid options: keep_existing, keep_imported, keep_newer",
            other
        ),
    }
}

fn read_secret_from_stdin(label: &str) -> Result<String> {
    if std::io::stdin().is_terminal() {
        eprint!("{label}: ");
        std::io::stderr().flush()?;
    }
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let secret = input.trim().to_string();
    if secret.is_empty() {
        anyhow::bail!("{label} must not be empty");
    }
    Ok(secret)
}

fn bind_address_is_loopback(bind_address: &str) -> bool {
    bind_address
        .to_socket_addrs()
        .map(|addrs| addrs.into_iter().all(|addr| addr.ip().is_loopback()))
        .unwrap_or(false)
}

fn load_config(cli: &Cli) -> Result<ServerConfig> {
    let mut config =
        ServerConfig::load(cli.config.as_deref()).context("Failed to load configuration")?;

    // CLI flags override config file
    if let Some(data_dir) = &cli.data_dir {
        config.data_dir = data_dir.clone();
    }

    if let Commands::Serve { port, grpc_port } = &cli.command {
        if let Some(port) = port {
            config.http.port = *port;
        }
        if let Some(grpc_port) = grpc_port {
            config.grpc.port = Some(*grpc_port);
        }
    }

    config
        .validate()
        .map_err(|e| anyhow::anyhow!("Invalid configuration: {e}"))?;

    Ok(config)
}

fn build_llm_provider(
    config: &ServerConfig,
) -> Result<Option<Arc<dyn cerememory_core::LLMProvider>>> {
    let provider_name = config.llm.provider.as_str();
    let model = config.llm.model.clone();
    let base_url = config.llm.base_url.clone();

    let require_api_key = || {
        config
            .llm
            .api_key_exposed()
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("LLM provider '{provider_name}' requires an API key"))
    };

    match provider_name {
        #[cfg(feature = "llm-openai")]
        "openai" => Ok(Some(Arc::new(
            cerememory_adapter_openai::OpenAIProvider::new(require_api_key()?, model, base_url)?,
        ))),
        #[cfg(feature = "llm-claude")]
        "claude" | "anthropic" => Ok(Some(Arc::new(
            cerememory_adapter_claude::ClaudeProvider::new(require_api_key()?, model, base_url)?,
        ))),
        #[cfg(feature = "llm-gemini")]
        "gemini" | "google" => Ok(Some(Arc::new(
            cerememory_adapter_gemini::GeminiProvider::new(require_api_key()?, model, base_url)?,
        ))),
        "none" | "" => Ok(None),
        other => {
            tracing::warn!(
                provider = other,
                "Unknown LLM provider, running without LLM"
            );
            Ok(None)
        }
    }
}

fn create_engine_from_config(config: &ServerConfig) -> Result<CerememoryEngine> {
    std::fs::create_dir_all(&config.data_dir).context("Failed to create data directory")?;
    let mut engine_config = config.to_engine_config();
    engine_config.llm_provider = build_llm_provider(config)?;
    Ok(CerememoryEngine::new(engine_config)?)
}

/// Initialize tracing subscriber from config.
///
/// `RUST_LOG` env var overrides config if set.
fn init_logging(config: &ServerConfig) {
    // RUST_LOG env var takes priority over config
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::try_new(&config.log.level).unwrap_or_else(|_| EnvFilter::new("info"))
    };

    match config.log.format.as_str() {
        "json" => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(filter)
                .init();
        }
        _ => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli)?;
    init_logging(&config);

    // Install global panic handler to capture panics in structured logs
    std::panic::set_hook(Box::new(|info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".to_string()
        };
        tracing::error!(location = %location, payload = %payload, "Panic occurred");
    }));

    // Serve uses its own engine lifecycle with background decay + graceful shutdown
    if let Commands::Serve { .. } = &cli.command {
        let engine = Arc::new(create_engine_from_config(&config)?);
        engine.rebuild_coordinator().await?;
        engine.start_background_decay();

        // Shutdown coordination: CancellationToken propagates to all components
        let cancel = tokio_util::sync::CancellationToken::new();

        // Register signal handlers
        let cancel_signal = cancel.clone();
        tokio::spawn(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(mut sigterm) => {
                        tokio::select! {
                            _ = ctrl_c => {},
                            _ = sigterm.recv() => {},
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to register SIGTERM handler, using ctrl-c only");
                        let _ = ctrl_c.await;
                    }
                }
            }
            #[cfg(not(unix))]
            {
                let _ = ctrl_c.await;
            }
            tracing::info!("Shutdown signal received, draining connections...");
            cancel_signal.cancel();
        });

        println!("Cerememory v{}", env!("CARGO_PKG_VERSION"));

        // Print startup summary
        let abs_data_dir = std::fs::canonicalize(&config.data_dir)
            .unwrap_or_else(|_| std::path::PathBuf::from(&config.data_dir));
        println!("Data  {}", abs_data_dir.display());
        println!("Config {}", cli.config.as_deref().unwrap_or("(defaults)"));
        let llm_status = match config.llm.provider.as_str() {
            "none" | "" => "disabled".to_string(),
            provider => format!(
                "{provider} ({})",
                config.llm.model.as_deref().unwrap_or("default")
            ),
        };
        println!("LLM   {llm_status}");

        // Optionally start gRPC server on a separate port.
        let mut grpc_handle: Option<tokio::task::JoinHandle<()>> = None;
        if let Some(grpc_port) = config.grpc.port {
            let engine_grpc = Arc::clone(&engine);
            let grpc_addr = format!("{}:{grpc_port}", config.http.bind_address);
            let grpc_keys = config.auth.api_key_strings();
            let grpc_auth_enabled = config.auth.enabled;
            let grpc_cancel = cancel.clone();

            // Load TLS config if paths are provided
            let tls = match (&config.grpc.tls_cert_path, &config.grpc.tls_key_path) {
                (Some(cert_path), Some(key_path)) => {
                    let cert_pem =
                        std::fs::read(cert_path).context("Failed to read TLS cert file")?;
                    let key_pem = std::fs::read(key_path).context("Failed to read TLS key file")?;
                    Some(cerememory_transport_grpc::TlsConfig { cert_pem, key_pem })
                }
                _ => None,
            };

            let grpc_tls_required =
                grpc_auth_enabled || !bind_address_is_loopback(&config.http.bind_address);
            if grpc_tls_required && tls.is_none() {
                anyhow::bail!(
                    "gRPC TLS is required when auth is enabled or the bind address is not loopback"
                );
            }

            // Verify the gRPC port is bindable before starting
            let listener = tokio::net::TcpListener::bind(&grpc_addr)
                .await
                .with_context(|| format!("Failed to bind gRPC port {grpc_port}"))?;
            drop(listener);
            let tls_label = if tls.is_some() { " (TLS)" } else { "" };
            println!("gRPC listening on {grpc_addr}{tls_label}");
            grpc_handle = Some(tokio::spawn(async move {
                if let Err(e) = cerememory_transport_grpc::serve_with_tls(
                    engine_grpc,
                    &grpc_addr,
                    grpc_auth_enabled,
                    grpc_keys,
                    tls,
                    grpc_cancel.cancelled_owned(),
                )
                .await
                {
                    tracing::error!(error = %e, "gRPC server failed");
                }
            }));
        }

        let api_keys = config.auth.api_key_strings();
        if config.auth.enabled {
            println!("Auth  enabled ({} key(s))", api_keys.len());
        }

        // Initialize Prometheus metrics and build router with full middleware config
        let prom_handle = if config.http.metrics_enabled {
            Some(cerememory_transport_http::install_prometheus_recorder())
        } else {
            None
        };
        let http_config = cerememory_transport_http::HttpMiddlewareConfig {
            api_keys,
            auth_enabled: config.auth.enabled,
            cors_origins: config.http.cors_origins.clone(),
            trusted_proxy_cidrs: config.http.trusted_proxy_cidrs.clone(),
            metrics_enabled: config.http.metrics_enabled,
            rate_limit_rps: config.rate_limit.requests_per_second,
            rate_limit_burst: config.rate_limit.burst,
            prometheus_handle: prom_handle,
        };
        let app = cerememory_transport_http::router_with_config(Arc::clone(&engine), http_config);

        let addr = format!("{}:{}", config.http.bind_address, config.http.port);
        println!("HTTP  listening on {addr}");
        if config.http.bind_address == "0.0.0.0" && !config.auth.enabled {
            tracing::warn!("Server bound to 0.0.0.0 with authentication disabled — accessible from entire network");
        }
        let http_cancel = cancel.clone();
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("Failed to bind HTTP port {}", config.http.port))?;
        axum::serve(listener, app)
            .with_graceful_shutdown(http_cancel.cancelled_owned())
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Await gRPC server shutdown
        if let Some(handle) = grpc_handle {
            match tokio::time::timeout(std::time::Duration::from_secs(30), handle).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, "gRPC server task panicked"),
                Err(_) => tracing::warn!("gRPC server shutdown timed out after 30 seconds"),
            }
        }

        // After HTTP server stops, clean up background tasks
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            engine.stop_background_decay(),
        )
        .await
        {
            Ok(()) => {}
            Err(_) => {
                tracing::warn!("Background decay shutdown timed out after 30 seconds");
            }
        }
        tracing::info!("Shutdown complete");
        return Ok(());
    }

    // MCP: stdio server, needs engine but no HTTP/gRPC
    if matches!(cli.command, Commands::Mcp) {
        let engine = Arc::new(create_engine_from_config(&config)?);
        engine.rebuild_coordinator().await?;
        cerememory_transport_mcp::serve_stdio(engine).await?;
        return Ok(());
    }

    // Healthcheck: lightweight HTTP probe, no engine needed
    if let Commands::Healthcheck { port } = &cli.command {
        let url = format!("http://127.0.0.1:{port}/health");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .context("Failed to build HTTP client")?;
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                println!("ok");
                return Ok(());
            }
            Ok(resp) => {
                anyhow::bail!("Health check failed: HTTP {}", resp.status());
            }
            Err(e) => {
                anyhow::bail!("Health check failed: {e}");
            }
        }
    }

    let engine = create_engine_from_config(&config)?;

    // Only rebuild coordinator/indexes for commands that need up-to-date index state.
    let needs_rebuild = matches!(
        &cli.command,
        Commands::Store { .. }
            | Commands::Recall { .. }
            | Commands::Consolidate { .. }
            | Commands::SetMode { .. }
            | Commands::Import { .. }
            | Commands::DecayTick { .. }
            | Commands::Forget { .. }
            | Commands::Export { .. }
    );
    if needs_rebuild {
        engine.rebuild_coordinator().await?;
    }

    match cli.command {
        Commands::Serve { .. } => unreachable!(),

        Commands::Consolidate {
            strategy,
            min_age_hours,
            min_access_count,
            dry_run,
        } => {
            let req = ConsolidateRequest {
                header: None,
                strategy: parse_consolidation_strategy(&strategy)?,
                min_age_hours,
                min_access_count,
                dry_run,
            };
            let resp = engine.lifecycle_consolidate(req).await?;
            println!(
                "Consolidated: processed={}, migrated={}, compressed={}, pruned={}, semantic_nodes_created={}",
                resp.records_processed,
                resp.records_migrated,
                resp.records_compressed,
                resp.records_pruned,
                resp.semantic_nodes_created
            );
        }

        Commands::SetMode { mode, scope } => {
            let mode = match mode.to_lowercase().as_str() {
                "human" => RecallMode::Human,
                "perfect" => RecallMode::Perfect,
                other => anyhow::bail!(
                    "Invalid recall mode '{}'. Valid options: human, perfect",
                    other
                ),
            };
            let scope = scope.as_deref().map(parse_store_list).transpose()?;
            engine
                .lifecycle_set_mode(SetModeRequest {
                    header: None,
                    mode,
                    scope,
                })
                .await?;
            println!("Recall mode updated to {mode:?}");
        }

        Commands::Store {
            text,
            store,
            embedding,
        } => {
            let store_type = parse_store_type(&store)?;
            let emb = embedding.map(|s| parse_embedding(&s)).transpose()?;

            let req = EncodeStoreRequest {
                header: None,
                content: MemoryContent {
                    blocks: vec![ContentBlock {
                        modality: Modality::Text,
                        format: "text/plain".to_string(),
                        data: text.into_bytes(),
                        embedding: emb,
                    }],
                    summary: None,
                },
                store: Some(store_type),
                emotion: None,
                context: None,
                metadata: None,
                associations: None,
            };

            if store_type == StoreType::Working {
                eprintln!(
                    "Warning: working store is in-memory only and will not persist across restarts"
                );
            }

            let resp = engine.encode_store(req).await?;
            println!(
                "Stored: {} (store: {}, fidelity: {})",
                resp.record_id, resp.store, resp.initial_fidelity
            );
        }

        Commands::Recall { query, limit, mode } => {
            let recall_mode = match mode.to_lowercase().as_str() {
                "human" => RecallMode::Human,
                "perfect" => RecallMode::Perfect,
                other => anyhow::bail!(
                    "Invalid recall mode '{}'. Valid options: human, perfect",
                    other
                ),
            };

            let req = RecallQueryRequest {
                header: None,
                cue: RecallCue {
                    text: Some(query),
                    ..Default::default()
                },
                stores: None,
                limit,
                min_fidelity: None,
                include_decayed: false,
                reconsolidate: true,
                activation_depth: 2,
                recall_mode,
            };

            let resp = engine.recall_query(req).await?;
            println!(
                "Found {} memories (total candidates: {})\n",
                resp.memories.len(),
                resp.total_candidates
            );

            for (i, mem) in resp.memories.iter().enumerate() {
                let text = mem.record.text_content().unwrap_or("[non-text]");
                println!(
                    "{}. [{}] score={:.4} fidelity={:.4}",
                    i + 1,
                    mem.record.store,
                    mem.relevance_score,
                    mem.record.fidelity.score
                );
                println!("   ID: {}", mem.record.id);
                println!("   {}", text);
                println!();
            }
        }

        Commands::Inspect { record_id } => {
            let record = engine
                .introspect_record(RecordIntrospectRequest {
                    header: None,
                    record_id,
                    include_history: true,
                    include_associations: true,
                    include_versions: true,
                })
                .await?;

            println!("{}", serde_json::to_string_pretty(&record)?);
        }

        Commands::Stats => {
            let stats = engine.introspect_stats().await?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }

        Commands::DecayTick { duration } => {
            let resp = engine
                .lifecycle_decay_tick(DecayTickRequest {
                    header: None,
                    tick_duration_seconds: Some(duration),
                })
                .await?;

            println!("Decay tick completed:");
            println!("  Records updated: {}", resp.records_updated);
            println!("  Below threshold: {}", resp.records_below_threshold);
            println!("  Pruned: {}", resp.records_pruned);
        }

        Commands::Forget {
            record_id,
            cascade,
            confirm,
        } => {
            let confirmed = if confirm {
                true
            } else if std::io::stdin().is_terminal() {
                eprintln!(
                    "Warning: This will permanently delete record {record_id}. This cannot be undone."
                );
                eprint!("Are you sure? [y/N] ");
                std::io::stderr().flush()?;
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                input.trim().eq_ignore_ascii_case("y")
            } else {
                anyhow::bail!(
                    "Forget requires confirmation. Use --confirm flag in non-interactive mode."
                );
            };

            if !confirmed {
                println!("Aborted.");
                return Ok(());
            }

            let deleted = engine
                .lifecycle_forget(ForgetRequest {
                    header: None,
                    record_ids: Some(vec![record_id]),
                    store: None,
                    temporal_range: None,
                    cascade,
                    confirm: true,
                })
                .await?;

            println!("Deleted {deleted} record(s)");
        }

        Commands::Export {
            output,
            format,
            stores,
            encrypt,
            encryption_key,
            encryption_key_stdin,
        } => {
            if encryption_key.is_some() && encryption_key_stdin {
                anyhow::bail!(
                    "Use either --encryption-key or --encryption-key-stdin, not both."
                );
            }
            let encryption_key = if encryption_key_stdin {
                Some(read_secret_from_stdin("Export encryption passphrase")?)
            } else {
                encryption_key
            };
            let stores = stores.as_deref().map(parse_store_list).transpose()?;
            let req = ExportRequest {
                header: None,
                format,
                stores,
                encrypt,
                encryption_key,
            };
            let (bytes, resp) = engine.lifecycle_export(req).await?;

            std::fs::write(&output, &bytes).context("Failed to write archive file")?;
            println!(
                "Exported {} records to {output} ({} bytes, archive_id={}, checksum={})",
                resp.record_count,
                bytes.len(),
                resp.archive_id,
                resp.checksum
            );
        }

        Commands::Import {
            path,
            decryption_key,
            decryption_key_stdin,
            conflict_resolution,
        } => {
            if decryption_key.is_some() && decryption_key_stdin {
                anyhow::bail!(
                    "Use either --decryption-key or --decryption-key-stdin, not both."
                );
            }
            let decryption_key = if decryption_key_stdin {
                Some(read_secret_from_stdin("Import decryption passphrase")?)
            } else {
                decryption_key
            };
            let bytes = std::fs::read(&path).context("Failed to read archive file")?;
            let archive_id = std::path::Path::new(&path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("cli-import")
                .to_string();
            let imported = engine
                .lifecycle_import(ImportRequest {
                    header: None,
                    archive_id,
                    strategy: ImportStrategy::Merge,
                    conflict_resolution: parse_conflict_resolution(&conflict_resolution)?,
                    decryption_key,
                    archive_data: Some(bytes),
                })
                .await?;
            println!("Imported {imported} records from {path}");
        }

        Commands::Healthcheck { .. } => unreachable!("Handled above"),
        Commands::Mcp => unreachable!("Handled above"),
    }

    Ok(())
}
