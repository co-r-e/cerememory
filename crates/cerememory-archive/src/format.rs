//! CMA (Cerememory Archive) format types.
//!
//! JSON Lines format:
//! - Line 1: ArchiveHeader
//! - Lines 2..N: One MemoryRecord per line (JSON)
//! - Last line: ArchiveFooter with SHA-256 checksum

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use cerememory_core::types::{MemoryRecord, RawJournalRecord};

/// Header line of a CMA archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveHeader {
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub record_count: u32,
    #[serde(default)]
    pub curated_record_count: Option<u32>,
    #[serde(default)]
    pub raw_record_count: Option<u32>,
}

/// Footer line of a CMA archive (integrity verification).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveFooter {
    /// SHA-256 hex digest of all record lines (excluding header and footer).
    pub checksum: String,
}

/// Current archive format version.
pub const ARCHIVE_VERSION: &str = "1.0";
/// Bundle archive format version that can carry both curated and raw records.
pub const BUNDLE_ARCHIVE_VERSION: &str = "2.0";

/// Archive entry variants for bundle archives.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum ArchiveEntry {
    MemoryRecord(MemoryRecord),
    RawJournalRecord(RawJournalRecord),
}

/// A fully decoded archive bundle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchiveBundle {
    #[serde(default)]
    pub records: Vec<MemoryRecord>,
    #[serde(default)]
    pub raw_records: Vec<RawJournalRecord>,
}
