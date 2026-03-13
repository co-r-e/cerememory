//! CMA archive importer with checksum verification.

use sha2::{Digest, Sha256};

use cerememory_core::error::CerememoryError;
use cerememory_core::types::MemoryRecord;

use crate::format::{ArchiveFooter, ArchiveHeader, ARCHIVE_VERSION};

/// Maximum archive size: 512 MB.
const MAX_ARCHIVE_SIZE: usize = 512 * 1024 * 1024;

/// Import memory records from CMA JSON Lines archive bytes.
///
/// Validates header version/record_count, verifies SHA-256 checksum,
/// and runs `MemoryRecord::validate()` on each imported record.
pub fn import(data: &[u8]) -> Result<Vec<MemoryRecord>, CerememoryError> {
    if data.len() > MAX_ARCHIVE_SIZE {
        return Err(CerememoryError::ImportConflict(format!(
            "Archive too large: {} bytes (max {})",
            data.len(),
            MAX_ARCHIVE_SIZE
        )));
    }

    let text = std::str::from_utf8(data)
        .map_err(|e| CerememoryError::ImportConflict(format!("Invalid UTF-8: {e}")))?;

    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return Err(CerememoryError::ImportConflict(
            "Archive too short: need at least header and footer".to_string(),
        ));
    }

    // Parse and validate header
    let header: ArchiveHeader = serde_json::from_str(lines[0])
        .map_err(|e| CerememoryError::ImportConflict(format!("Invalid header: {e}")))?;

    if header.version != ARCHIVE_VERSION {
        return Err(CerememoryError::ImportConflict(format!(
            "Unsupported archive version: {} (expected {})",
            header.version, ARCHIVE_VERSION
        )));
    }

    // Parse footer (last line)
    let footer: ArchiveFooter = serde_json::from_str(lines[lines.len() - 1])
        .map_err(|e| CerememoryError::ImportConflict(format!("Invalid footer: {e}")))?;

    // Parse records (lines between header and footer)
    let record_lines = &lines[1..lines.len() - 1];

    // Validate record_count matches actual lines
    if header.record_count as usize != record_lines.len() {
        return Err(CerememoryError::ImportConflict(format!(
            "record_count mismatch: header says {}, found {} record lines",
            header.record_count,
            record_lines.len()
        )));
    }

    let mut hasher = Sha256::new();
    let mut records = Vec::with_capacity(record_lines.len());

    for line in record_lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");

        let record: MemoryRecord = serde_json::from_str(line)
            .map_err(|e| CerememoryError::ImportConflict(format!("Invalid record: {e}")))?;

        // Validate each imported record
        record
            .validate()
            .map_err(|e| CerememoryError::ImportConflict(format!("Record validation failed: {e}")))?;

        records.push(record);
    }

    // Verify checksum
    let computed = format!("{:x}", hasher.finalize());
    if computed != footer.checksum {
        return Err(CerememoryError::ImportConflict(format!(
            "Checksum mismatch: expected {}, got {}",
            footer.checksum, computed
        )));
    }

    Ok(records)
}
