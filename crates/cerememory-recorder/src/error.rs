use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum RecorderError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("invalid capture event on line {line}: {message}")]
    InvalidEvent { line: usize, message: String },
    #[error("event is too large: {size} bytes exceeds {limit} byte limit")]
    EventTooLarge { size: usize, limit: usize },
    #[error("HTTP client error: {0}")]
    HttpClient(String),
    #[error("send failed: {0}")]
    Send(#[from] crate::client::SendError),
    #[error("spool error at {path}: {source}")]
    Spool {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{operation} failed at {path}: {source}")]
    File {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
