//! CMA archive exporter.

use chrono::Utc;
use sha2::{Digest, Sha256};

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::ExportResponse;
use cerememory_core::types::MemoryRecord;

use crate::format::{ArchiveFooter, ArchiveHeader, ARCHIVE_VERSION};

/// Export a slice of memory records to CMA JSON Lines format.
///
/// Returns the serialized archive bytes and an ExportResponse with metadata.
pub fn export(records: &[MemoryRecord]) -> Result<(Vec<u8>, ExportResponse), CerememoryError> {
    let mut output = Vec::new();

    // Write header
    let header = ArchiveHeader {
        version: ARCHIVE_VERSION.to_string(),
        timestamp: Utc::now(),
        record_count: records.len() as u32,
    };
    let header_line = serde_json::to_string(&header)
        .map_err(|e| CerememoryError::ExportFailed(format!("Header serialization: {e}")))?;
    output.extend_from_slice(header_line.as_bytes());
    output.push(b'\n');

    // Write records and compute checksum
    let mut hasher = Sha256::new();
    for record in records {
        let line = serde_json::to_string(record)
            .map_err(|e| CerememoryError::ExportFailed(format!("Record serialization: {e}")))?;
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
        output.extend_from_slice(line.as_bytes());
        output.push(b'\n');
    }

    // Write footer
    let checksum = format!("{:x}", hasher.finalize());
    let footer = ArchiveFooter {
        checksum: checksum.clone(),
    };
    let footer_line = serde_json::to_string(&footer)
        .map_err(|e| CerememoryError::ExportFailed(format!("Footer serialization: {e}")))?;
    output.extend_from_slice(footer_line.as_bytes());
    output.push(b'\n');

    let resp = ExportResponse {
        archive_id: uuid::Uuid::now_v7().to_string(),
        size_bytes: output.len() as u64,
        record_count: records.len() as u32,
        checksum,
    };

    Ok((output, resp))
}
