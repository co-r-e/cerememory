//! Cerememory CLI — command-line interface for the living memory database.

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
    },
}

fn parse_store_type(s: &str) -> Result<StoreType> {
    match s.to_lowercase().as_str() {
        "episodic" => Ok(StoreType::Episodic),
        "semantic" => Ok(StoreType::Semantic),
        "procedural" => Ok(StoreType::Procedural),
        "emotional" => Ok(StoreType::Emotional),
        "working" => Ok(StoreType::Working),
        _ => anyhow::bail!("Invalid store type: {s}. Use: episodic, semantic, procedural, emotional, working"),
    }
}

fn parse_embedding(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|v| v.trim().parse::<f32>().context("Invalid embedding value"))
        .collect()
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
) -> Option<Arc<dyn cerememory_core::LLMProvider>> {
    let provider_name = config.llm.provider.as_str();
    let api_key = config.llm.api_key_exposed()?.to_string();
    let model = config.llm.model.clone();
    let base_url = config.llm.base_url.clone();

    match provider_name {
        #[cfg(feature = "llm-openai")]
        "openai" => Some(Arc::new(cerememory_adapter_openai::OpenAIProvider::new(
            api_key, model, base_url,
        ))),
        #[cfg(feature = "llm-claude")]
        "claude" | "anthropic" => Some(Arc::new(
            cerememory_adapter_claude::ClaudeProvider::new(api_key, model, base_url),
        )),
        #[cfg(feature = "llm-gemini")]
        "gemini" | "google" => Some(Arc::new(
            cerememory_adapter_gemini::GeminiProvider::new(api_key, model, base_url),
        )),
        "none" | "" => None,
        other => {
            tracing::warn!(provider = other, "Unknown LLM provider, running without LLM");
            None
        }
    }
}

fn create_engine_from_config(config: &ServerConfig) -> Result<CerememoryEngine> {
    std::fs::create_dir_all(&config.data_dir).context("Failed to create data directory")?;
    let mut engine_config = config.to_engine_config();
    engine_config.llm_provider = build_llm_provider(config);
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
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .init();
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli)?;
    init_logging(&config);

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
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("Failed to register SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
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

        // Optionally start gRPC server on a separate port.
        if let Some(grpc_port) = config.grpc.port {
            let engine_grpc = Arc::clone(&engine);
            let grpc_addr = format!("0.0.0.0:{grpc_port}");
            let grpc_keys = config.auth.api_key_strings();
            let grpc_cancel = cancel.clone();

            // Load TLS config if paths are provided
            let tls = match (&config.grpc.tls_cert_path, &config.grpc.tls_key_path) {
                (Some(cert_path), Some(key_path)) => {
                    let cert_pem =
                        std::fs::read(cert_path).context("Failed to read TLS cert file")?;
                    let key_pem =
                        std::fs::read(key_path).context("Failed to read TLS key file")?;
                    Some(cerememory_transport_grpc::TlsConfig { cert_pem, key_pem })
                }
                _ => None,
            };

            // Verify the gRPC port is bindable before starting
            let listener = tokio::net::TcpListener::bind(&grpc_addr)
                .await
                .with_context(|| format!("Failed to bind gRPC port {grpc_port}"))?;
            drop(listener);
            let tls_label = if tls.is_some() { " (TLS)" } else { "" };
            println!("gRPC listening on {grpc_addr}{tls_label}");
            tokio::spawn(async move {
                if let Err(e) = cerememory_transport_grpc::serve_with_tls(
                    engine_grpc,
                    &grpc_addr,
                    grpc_keys,
                    tls,
                    grpc_cancel.cancelled_owned(),
                )
                .await
                {
                    tracing::error!(error = %e, "gRPC server failed");
                }
            });
        }

        let api_keys = config.auth.api_key_strings();
        if config.auth.enabled {
            println!("Auth  enabled ({} key(s))", api_keys.len());
        }

        // Initialize Prometheus metrics and build router with full middleware config
        let prom_handle = cerememory_transport_http::install_prometheus_recorder();
        let http_config = cerememory_transport_http::HttpMiddlewareConfig {
            api_keys,
            cors_origins: config.http.cors_origins.clone(),
            rate_limit_rps: config.rate_limit.requests_per_second,
            rate_limit_burst: config.rate_limit.burst,
            prometheus_handle: Some(prom_handle),
        };
        let app = cerememory_transport_http::router_with_config(Arc::clone(&engine), http_config);

        let addr = format!("0.0.0.0:{}", config.http.port);
        println!("HTTP  listening on {addr}");
        let http_cancel = cancel.clone();
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("Failed to bind HTTP port {}", config.http.port))?;
        axum::serve(listener, app)
            .with_graceful_shutdown(http_cancel.cancelled_owned())
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // After HTTP server stops, clean up background tasks
        engine.stop_background_decay().await;
        tracing::info!("Shutdown complete");
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

    // Only rebuild coordinator/indexes for commands that need up-to-date index state
    let needs_rebuild = matches!(
        cli.command,
        Commands::Store { .. }
            | Commands::Recall { .. }
            | Commands::Import { .. }
            | Commands::DecayTick { .. }
    );
    if needs_rebuild {
        engine.rebuild_coordinator().await?;
    }

    match cli.command {
        Commands::Serve { .. } => unreachable!(),

        Commands::Store { text, store, embedding } => {
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
                associations: None,
            };

            if store_type == StoreType::Working {
                eprintln!("Warning: working store is in-memory only and will not persist across restarts");
            }

            let resp = engine.encode_store(req).await?;
            println!("Stored: {} (store: {}, fidelity: {})", resp.record_id, resp.store, resp.initial_fidelity);
        }

        Commands::Recall { query, limit, mode } => {
            let recall_mode = match mode.to_lowercase().as_str() {
                "perfect" => RecallMode::Perfect,
                _ => RecallMode::Human,
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
            println!("Found {} memories (total candidates: {})\n", resp.memories.len(), resp.total_candidates);

            for (i, mem) in resp.memories.iter().enumerate() {
                let text = mem.record.text_content().unwrap_or("[non-text]");
                println!("{}. [{}] score={:.4} fidelity={:.4}", i + 1, mem.record.store, mem.relevance_score, mem.record.fidelity.score);
                println!("   ID: {}", mem.record.id);
                println!("   {}", text);
                println!();
            }
        }

        Commands::Inspect { record_id } => {
            let record = engine.introspect_record(RecordIntrospectRequest {
                header: None,
                record_id,
                include_history: true,
                include_associations: true,
                include_versions: true,
            }).await?;

            println!("{}", serde_json::to_string_pretty(&record)?);
        }

        Commands::Stats => {
            let stats = engine.introspect_stats().await?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }

        Commands::DecayTick { duration } => {
            let resp = engine.lifecycle_decay_tick(DecayTickRequest {
                header: None,
                tick_duration_seconds: Some(duration),
            }).await?;

            println!("Decay tick completed:");
            println!("  Records updated: {}", resp.records_updated);
            println!("  Below threshold: {}", resp.records_below_threshold);
            println!("  Pruned: {}", resp.records_pruned);
        }

        Commands::Forget { record_id, cascade, confirm } => {
            let deleted = engine.lifecycle_forget(ForgetRequest {
                header: None,
                record_ids: Some(vec![record_id]),
                store: None,
                temporal_range: None,
                cascade,
                confirm,
            }).await?;

            println!("Deleted {deleted} record(s)");
        }

        Commands::Export { output } => {
            let records = engine.collect_all_records().await?;
            let bytes = cerememory_archive::export_to_bytes(&records)?;

            std::fs::write(&output, &bytes).context("Failed to write archive file")?;
            println!("Exported {} records to {output} ({} bytes)", records.len(), bytes.len());
        }

        Commands::Import { path } => {
            let bytes = std::fs::read(&path).context("Failed to read archive file")?;
            let imported = engine.import_records(&bytes).await?;
            println!("Imported {imported} records from {path}");
        }

        Commands::Healthcheck { .. } => unreachable!("Handled above"),
    }

    Ok(())
}
