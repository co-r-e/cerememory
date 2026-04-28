use std::path::PathBuf;
use std::time::Duration;

use crate::RecorderError;

pub const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8420";
pub const DEFAULT_MAX_EVENT_BYTES: usize = 256 * 1024;
pub const DEFAULT_BATCH_MAX_RECORDS: usize = 50;
pub const DEFAULT_FLUSH_INTERVAL_MS: u64 = 1_000;
pub const DEFAULT_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub server_url: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
    pub max_event_bytes: usize,
    pub batch_max_records: usize,
    pub flush_interval: Duration,
    pub spool_dir: PathBuf,
}

impl RecorderConfig {
    pub fn new(server_url: impl Into<String>) -> Result<Self, RecorderError> {
        let server_url = server_url.into();
        let api_key = std::env::var("CEREMEMORY_SERVER_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty());
        Ok(Self {
            server_url,
            api_key,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
            batch_max_records: DEFAULT_BATCH_MAX_RECORDS,
            flush_interval: Duration::from_millis(DEFAULT_FLUSH_INTERVAL_MS),
            spool_dir: default_spool_dir()?,
        })
    }

    pub fn with_spool_dir(mut self, spool_dir: PathBuf) -> Self {
        self.spool_dir = spool_dir;
        self
    }

    pub fn with_max_event_bytes(mut self, max_event_bytes: usize) -> Self {
        self.max_event_bytes = max_event_bytes;
        self
    }

    pub fn with_batch_max_records(mut self, batch_max_records: usize) -> Self {
        self.batch_max_records = batch_max_records.max(1);
        self
    }

    pub fn with_flush_interval(mut self, flush_interval: Duration) -> Self {
        self.flush_interval = flush_interval;
        self
    }
}

fn default_spool_dir() -> Result<PathBuf, RecorderError> {
    if let Ok(path) = std::env::var("CEREMEMORY_RECORDER_SPOOL_DIR") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .ok_or_else(|| {
            RecorderError::Config(
                "HOME is not set; pass --spool-dir or set CEREMEMORY_RECORDER_SPOOL_DIR"
                    .to_string(),
            )
        })?;
    Ok(home.join(".cerememory").join("recorder").join("spool"))
}
