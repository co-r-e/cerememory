//! CMA archive importer with checksum verification.

use std::io::BufRead;

use sha2::{Digest, Sha256};

use cerememory_core::error::CerememoryError;
use cerememory_core::types::MemoryRecord;

use crate::format::{ArchiveFooter, ArchiveHeader, ARCHIVE_VERSION};

/// Maximum archive size: 512 MB.
const MAX_ARCHIVE_SIZE: usize = 512 * 1024 * 1024;

/// Import memory records from a reader in CMA JSON Lines format (streaming).
///
/// Reads one line at a time, computing the SHA-256 checksum incrementally.
/// Peak memory usage is O(records) for the result vector, but avoids loading
/// the entire raw archive into a single buffer.
pub fn import_from_reader<R: BufRead>(reader: R) -> Result<Vec<MemoryRecord>, CerememoryError> {
    let mut lines = reader.lines();

    // Parse header (first line)
    let header_line = lines
        .next()
        .ok_or_else(|| {
            CerememoryError::ImportConflict("Archive is empty: missing header".to_string())
        })?
        .map_err(|e| CerememoryError::ImportConflict(format!("Read error: {e}")))?;

    let header: ArchiveHeader = serde_json::from_str(&header_line)
        .map_err(|e| CerememoryError::ImportConflict(format!("Invalid header: {e}")))?;

    if header.version != ARCHIVE_VERSION {
        return Err(CerememoryError::ImportConflict(format!(
            "Unsupported archive version: {} (expected {})",
            header.version, ARCHIVE_VERSION
        )));
    }

    // Read record lines and trailing footer
    let mut hasher = Sha256::new();
    let mut records = Vec::with_capacity(header.record_count as usize);
    let mut last_line: Option<String> = None;

    for line_result in lines {
        let line =
            line_result.map_err(|e| CerememoryError::ImportConflict(format!("Read error: {e}")))?;

        if line.is_empty() {
            continue;
        }

        // The previous "last_line" was a record; process it now.
        if let Some(prev) = last_line.take() {
            hasher.update(prev.as_bytes());
            hasher.update(b"\n");

            let record: MemoryRecord = serde_json::from_str(&prev)
                .map_err(|e| CerememoryError::ImportConflict(format!("Invalid record: {e}")))?;
            record.validate().map_err(|e| {
                CerememoryError::ImportConflict(format!("Record validation failed: {e}"))
            })?;
            records.push(record);
        }

        last_line = Some(line);
    }

    // The final line should be the footer
    let footer_line = last_line.ok_or_else(|| {
        CerememoryError::ImportConflict("Archive too short: missing footer".to_string())
    })?;

    let footer: ArchiveFooter = serde_json::from_str(&footer_line)
        .map_err(|e| CerememoryError::ImportConflict(format!("Invalid footer: {e}")))?;

    // Validate record count
    if header.record_count as usize != records.len() {
        return Err(CerememoryError::ImportConflict(format!(
            "record_count mismatch: header says {}, found {} record lines",
            header.record_count,
            records.len()
        )));
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

/// Import memory records from CMA JSON Lines archive bytes (buffered).
///
/// Convenience wrapper around [`import_from_reader`] with size validation.
pub fn import(data: &[u8]) -> Result<Vec<MemoryRecord>, CerememoryError> {
    if data.len() > MAX_ARCHIVE_SIZE {
        return Err(CerememoryError::ImportConflict(format!(
            "Archive too large: {} bytes (max {})",
            data.len(),
            MAX_ARCHIVE_SIZE
        )));
    }
    import_from_reader(data)
}
