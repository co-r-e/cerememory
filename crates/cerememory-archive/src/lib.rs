//! CMA (Cerememory Archive) format implementation.
//!
//! JSON Lines-based portable archive for memory export, import, and backup.
//!
//! Format:
//! - Line 1: Header (version, timestamp, record_count)
//! - Lines 2..N: One MemoryRecord per line (JSON)
//! - Last line: Footer (SHA-256 checksum of record lines)

pub mod exporter;
pub mod format;
pub mod importer;

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::ExportResponse;
use cerememory_core::types::MemoryRecord;

/// Export records to CMA archive format. Returns both the archive bytes and metadata.
pub fn export(records: &[MemoryRecord]) -> Result<(Vec<u8>, ExportResponse), CerememoryError> {
    exporter::export(records)
}

/// Export records to CMA archive bytes only.
pub fn export_to_bytes(records: &[MemoryRecord]) -> Result<Vec<u8>, CerememoryError> {
    let (bytes, _) = exporter::export(records)?;
    Ok(bytes)
}

/// Import records from CMA archive bytes with checksum verification.
pub fn import_records(data: &[u8]) -> Result<Vec<MemoryRecord>, CerememoryError> {
    importer::import(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::types::{MemoryRecord, StoreType};

    fn make_records(n: usize) -> Vec<MemoryRecord> {
        (0..n)
            .map(|i| MemoryRecord::new_text(StoreType::Episodic, format!("Memory {i}")))
            .collect()
    }

    #[test]
    fn roundtrip_export_import() {
        let records = make_records(3);
        let bytes = export_to_bytes(&records).unwrap();
        let imported = import_records(&bytes).unwrap();

        assert_eq!(imported.len(), 3);
        for (orig, imp) in records.iter().zip(imported.iter()) {
            assert_eq!(orig.id, imp.id);
            assert_eq!(orig.text_content(), imp.text_content());
        }
    }

    #[test]
    fn empty_archive() {
        let records: Vec<MemoryRecord> = Vec::new();
        let bytes = export_to_bytes(&records).unwrap();
        let imported = import_records(&bytes).unwrap();
        assert!(imported.is_empty());
    }

    #[test]
    fn large_archive() {
        let records = make_records(1000);
        let bytes = export_to_bytes(&records).unwrap();
        let imported = import_records(&bytes).unwrap();
        assert_eq!(imported.len(), 1000);
    }

    #[test]
    fn corrupted_archive_detected() {
        let records = make_records(3);
        let mut bytes = export_to_bytes(&records).unwrap();

        // Corrupt a byte in the middle
        let mid = bytes.len() / 2;
        bytes[mid] = b'X';

        let result = import_records(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn checksum_mismatch_detected() {
        let records = make_records(2);
        let bytes = export_to_bytes(&records).unwrap();
        let text = String::from_utf8(bytes).unwrap();

        // Replace the checksum in the footer
        let lines: Vec<&str> = text.lines().collect();
        let last = lines.len() - 1;
        let mut footer: serde_json::Value = serde_json::from_str(lines[last]).unwrap();
        footer["checksum"] = serde_json::Value::String("0000000000".to_string());
        let bad_footer = serde_json::to_string(&footer).unwrap();

        let mut new_lines: Vec<String> = lines[..last].iter().map(|s| s.to_string()).collect();
        new_lines.push(bad_footer);
        let bad_archive = new_lines.join("\n");

        let result = import_records(bad_archive.as_bytes());
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Checksum mismatch"));
    }

    #[test]
    fn version_mismatch_rejected() {
        let records = make_records(1);
        let bytes = export_to_bytes(&records).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        // Tamper with version
        let mut header: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        header["version"] = serde_json::Value::String("99.0".to_string());
        lines[0] = serde_json::to_string(&header).unwrap();
        let bad_archive = lines.join("\n");
        let result = import_records(bad_archive.as_bytes());
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Unsupported archive version"));
    }

    #[test]
    fn record_count_mismatch_rejected() {
        let records = make_records(2);
        let bytes = export_to_bytes(&records).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        // Tamper with record_count
        let mut header: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        header["record_count"] = serde_json::Value::Number(serde_json::Number::from(999));
        lines[0] = serde_json::to_string(&header).unwrap();
        let bad_archive = lines.join("\n");
        let result = import_records(bad_archive.as_bytes());
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("record_count mismatch"));
    }

    #[test]
    fn export_response_metadata() {
        let records = make_records(5);
        let (bytes, resp) = export(&records).unwrap();
        assert_eq!(resp.record_count, 5);
        assert_eq!(resp.size_bytes, bytes.len() as u64);
        assert!(!resp.checksum.is_empty());
    }
}
