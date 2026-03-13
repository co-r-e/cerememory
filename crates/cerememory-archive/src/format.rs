//! CMA (Cerememory Archive) format types.
//!
//! JSON Lines format:
//! - Line 1: ArchiveHeader
//! - Lines 2..N: One MemoryRecord per line (JSON)
//! - Last line: ArchiveFooter with SHA-256 checksum

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Header line of a CMA archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveHeader {
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub record_count: u32,
}

/// Footer line of a CMA archive (integrity verification).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveFooter {
    /// SHA-256 hex digest of all record lines (excluding header and footer).
    pub checksum: String,
}

/// Current archive format version.
pub const ARCHIVE_VERSION: &str = "1.0";
