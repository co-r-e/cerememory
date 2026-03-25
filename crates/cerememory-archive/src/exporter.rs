//! CMA archive exporter.

use std::io::Write;

use chrono::Utc;
use sha2::{Digest, Sha256};

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::ExportResponse;
use cerememory_core::types::MemoryRecord;

use crate::format::{ArchiveFooter, ArchiveHeader, ARCHIVE_VERSION};

fn write_err(e: std::io::Error) -> CerememoryError {
    CerememoryError::ExportFailed(format!("Write error: {e}"))
}

/// Export records to a writer in CMA JSON Lines format (streaming).
///
/// Writes one record at a time, computing the SHA-256 checksum incrementally.
/// Peak memory usage is O(largest_single_record) rather than O(total_archive).
pub fn export_to_writer<W: Write>(
    records: &[MemoryRecord],
    writer: &mut W,
) -> Result<ExportResponse, CerememoryError> {
    let header = ArchiveHeader {
        version: ARCHIVE_VERSION.to_string(),
        timestamp: Utc::now(),
        record_count: records.len() as u32,
    };
    let header_line = serde_json::to_string(&header)
        .map_err(|e| CerememoryError::ExportFailed(format!("Header serialization: {e}")))?;
    writeln!(writer, "{header_line}").map_err(write_err)?;
    let mut bytes_written = header_line.len() as u64 + 1;

    let mut hasher = Sha256::new();
    for record in records {
        let line = serde_json::to_string(record)
            .map_err(|e| CerememoryError::ExportFailed(format!("Record serialization: {e}")))?;
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
        writeln!(writer, "{line}").map_err(write_err)?;
        bytes_written += line.len() as u64 + 1;
    }

    let checksum = format!("{:x}", hasher.finalize());
    let footer = ArchiveFooter {
        checksum: checksum.clone(),
    };
    let footer_line = serde_json::to_string(&footer)
        .map_err(|e| CerememoryError::ExportFailed(format!("Footer serialization: {e}")))?;
    writeln!(writer, "{footer_line}").map_err(write_err)?;
    bytes_written += footer_line.len() as u64 + 1;

    Ok(ExportResponse {
        archive_id: uuid::Uuid::now_v7().to_string(),
        size_bytes: bytes_written,
        record_count: records.len() as u32,
        checksum,
    })
}

/// Export a slice of memory records to CMA JSON Lines format (buffered).
///
/// Convenience wrapper around [`export_to_writer`] that returns the archive as bytes.
pub fn export(records: &[MemoryRecord]) -> Result<(Vec<u8>, ExportResponse), CerememoryError> {
    let mut output = Vec::new();
    let resp = export_to_writer(records, &mut output)?;
    Ok((output, resp))
}
