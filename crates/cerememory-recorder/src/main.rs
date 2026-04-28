//! `cerememory-recorder` command-line interface.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use cerememory_recorder::config::DEFAULT_SERVER_URL;
use cerememory_recorder::hook::install_codex_hook;
use cerememory_recorder::{run_doctor_with_options, run_ingest, DoctorOptions, RecorderConfig};

#[derive(Parser)]
#[command(
    name = "cerememory-recorder",
    about = "Normalize observed events into Cerememory raw journal batches",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Read Capture Event JSONL from stdin and send raw journal batches.
    Ingest {
        /// Cerememory HTTP server URL.
        #[arg(long, default_value = DEFAULT_SERVER_URL)]
        server_url: String,

        /// Local JSONL spool directory for failed batches.
        #[arg(long)]
        spool_dir: Option<PathBuf>,

        /// Maximum accepted JSONL event size in bytes.
        #[arg(
            long,
            default_value_t = cerememory_recorder::config::DEFAULT_MAX_EVENT_BYTES,
            value_parser = parse_nonzero_usize
        )]
        max_event_bytes: usize,

        /// Maximum records per HTTP batch.
        #[arg(
            long,
            default_value_t = cerememory_recorder::config::DEFAULT_BATCH_MAX_RECORDS,
            value_parser = parse_nonzero_usize
        )]
        batch_max_records: usize,

        /// Maximum delay before flushing a partial batch.
        #[arg(
            long,
            default_value_t = cerememory_recorder::config::DEFAULT_FLUSH_INTERVAL_MS,
            value_parser = parse_nonzero_u64
        )]
        flush_interval_ms: u64,
    },

    /// Install companion hook helpers without overwriting existing settings.
    InstallHook {
        #[command(subcommand)]
        adapter: HookAdapter,
    },

    /// Check server health, auth, raw ingest, and spool writability.
    Doctor {
        /// Cerememory HTTP server URL.
        #[arg(long, default_value = DEFAULT_SERVER_URL)]
        server_url: String,

        /// Local JSONL spool directory to verify.
        #[arg(long)]
        spool_dir: Option<PathBuf>,

        /// Skip the raw ingest probe so doctor does not write a private_scratch record.
        #[arg(long)]
        skip_raw_ingest_probe: bool,
    },
}

#[derive(Subcommand)]
enum HookAdapter {
    /// Generate a Codex hook adapter script in .codex/hooks.
    Codex {
        /// Cerememory HTTP server URL.
        #[arg(long, default_value = DEFAULT_SERVER_URL)]
        server_url: String,

        /// Directory where .codex/hooks should be created.
        #[arg(long)]
        target_dir: Option<PathBuf>,

        /// Overwrite previously generated recorder hook files.
        #[arg(long)]
        force: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Ingest {
            server_url,
            spool_dir,
            max_event_bytes,
            batch_max_records,
            flush_interval_ms,
        } => {
            let config = build_config(
                server_url,
                spool_dir,
                Some(max_event_bytes),
                Some(batch_max_records),
                Some(flush_interval_ms),
            )?;
            let stats = run_ingest(config).await?;
            eprintln!(
                "cerememory-recorder: accepted={}, duplicate={}, sent={}, spooled={}",
                stats.accepted, stats.duplicate, stats.sent, stats.spooled
            );
        }
        Commands::InstallHook { adapter } => match adapter {
            HookAdapter::Codex {
                server_url,
                target_dir,
                force,
            } => {
                let base_dir = match target_dir {
                    Some(path) => path,
                    None => std::env::current_dir().context("failed to read current directory")?,
                };
                let result = install_codex_hook(&base_dir, &server_url, force)?;
                println!("script: {}", result.script_path.display());
                println!("example: {}", result.example_path.display());
                println!("API key: set CEREMEMORY_SERVER_API_KEY in the hook environment");
            }
        },
        Commands::Doctor {
            server_url,
            spool_dir,
            skip_raw_ingest_probe,
        } => {
            let config = build_config(server_url, spool_dir, None, None, None)?;
            let report = run_doctor_with_options(
                config,
                DoctorOptions {
                    raw_ingest_probe: !skip_raw_ingest_probe,
                },
            )
            .await?;
            println!("{}", report.render_text());
            if !report.ok() {
                bail!("cerememory-recorder doctor found failing checks");
            }
        }
    }

    Ok(())
}

fn build_config(
    server_url: String,
    spool_dir: Option<PathBuf>,
    max_event_bytes: Option<usize>,
    batch_max_records: Option<usize>,
    flush_interval_ms: Option<u64>,
) -> Result<RecorderConfig> {
    let mut config = RecorderConfig::new(server_url)?;
    if let Some(spool_dir) = spool_dir {
        config = config.with_spool_dir(spool_dir);
    }
    if let Some(max_event_bytes) = max_event_bytes {
        config = config.with_max_event_bytes(max_event_bytes);
    }
    if let Some(batch_max_records) = batch_max_records {
        config = config.with_batch_max_records(batch_max_records);
    }
    if let Some(flush_interval_ms) = flush_interval_ms {
        config = config.with_flush_interval(Duration::from_millis(flush_interval_ms.max(1)));
    }
    Ok(config)
}

fn parse_nonzero_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("expected a positive integer: {err}"))?;
    if parsed == 0 {
        return Err("value must be at least 1".to_string());
    }
    Ok(parsed)
}

fn parse_nonzero_u64(value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|err| format!("expected a positive integer: {err}"))?;
    if parsed == 0 {
        return Err("value must be at least 1".to_string());
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonzero_cli_parsers_reject_zero() {
        assert_eq!(parse_nonzero_usize("1").unwrap(), 1);
        assert_eq!(parse_nonzero_u64("1").unwrap(), 1);
        assert!(parse_nonzero_usize("0").is_err());
        assert!(parse_nonzero_u64("0").is_err());
    }
}
