//! Cerememory CLI — command-line interface for the living memory database.

use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use cerememory_core::protocol::*;
use cerememory_core::types::*;
use cerememory_engine::{CerememoryEngine, EngineConfig};

#[derive(Parser)]
#[command(
    name = "cerememory",
    about = "A living memory database for the age of AI",
    version
)]
struct Cli {
    /// Data directory for persistent storage
    #[arg(long, default_value = "./data", global = true)]
    data_dir: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP server (and optionally a gRPC server)
    Serve {
        /// HTTP port to listen on
        #[arg(long, default_value = "8420")]
        port: u16,

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

fn build_config(data_dir: &str, background_decay_interval_secs: Option<u64>) -> EngineConfig {
    EngineConfig {
        episodic_path: Some(format!("{data_dir}/episodic.redb")),
        semantic_path: Some(format!("{data_dir}/semantic.redb")),
        procedural_path: Some(format!("{data_dir}/procedural.redb")),
        emotional_path: Some(format!("{data_dir}/emotional.redb")),
        index_path: Some(format!("{data_dir}/text_index")),
        vector_index_path: Some(format!("{data_dir}/vectors.redb")),
        background_decay_interval_secs,
        ..EngineConfig::default()
    }
}

fn create_engine(data_dir: &str) -> Result<CerememoryEngine> {
    std::fs::create_dir_all(data_dir).context("Failed to create data directory")?;
    let config = build_config(data_dir, None);
    Ok(CerememoryEngine::new(config)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // Serve uses its own engine lifecycle with background decay
    if let Commands::Serve { port, grpc_port } = &cli.command {
        std::fs::create_dir_all(&cli.data_dir).context("Failed to create data directory")?;
        let config = build_config(&cli.data_dir, Some(3600));
        let engine = Arc::new(CerememoryEngine::new(config)?);
        engine.rebuild_coordinator().await?;
        engine.start_background_decay();

        println!("Cerememory v{}", env!("CARGO_PKG_VERSION"));

        // Optionally start gRPC server on a separate port.
        if let Some(grpc_port) = grpc_port {
            let engine_grpc = Arc::clone(&engine);
            let grpc_addr = format!("0.0.0.0:{grpc_port}");
            // Verify the gRPC port is bindable before starting
            let listener = tokio::net::TcpListener::bind(&grpc_addr)
                .await
                .with_context(|| format!("Failed to bind gRPC port {grpc_port}"))?;
            drop(listener);
            println!("gRPC listening on {grpc_addr}");
            tokio::spawn(async move {
                if let Err(e) = cerememory_transport_grpc::serve(engine_grpc, &grpc_addr).await {
                    tracing::error!(error = %e, "gRPC server failed");
                }
            });
        }

        let addr = format!("0.0.0.0:{port}");
        println!("HTTP  listening on {addr}");
        cerememory_transport_http::serve(engine, &addr)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        return Ok(());
    }

    let engine = create_engine(&cli.data_dir)?;

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
    }

    Ok(())
}
