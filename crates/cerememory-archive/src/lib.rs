//! CMA (Cerememory Archive) format implementation.
//!
//! JSON Lines-based portable archive for memory export, import, and backup.
//!
//! Format:
//! - Line 1: Header (version, timestamp, record_count)
//! - Lines 2..N: One MemoryRecord per line (JSON)
//! - Last line: Footer (SHA-256 checksum of record lines)

pub mod crypto;
pub mod exporter;
pub mod format;
pub mod importer;

use cerememory_core::error::CerememoryError;
use cerememory_core::protocol::ExportResponse;
use cerememory_core::types::{MemoryRecord, StoreType};

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

/// Export records with optional store filtering and encryption.
///
/// When `stores` is `Some`, only records whose `store` field matches one of
/// the given store types are included. When `encryption_key` is `Some`, the
/// serialized archive is encrypted with ChaCha20-Poly1305 AEAD.
pub fn export_filtered(
    records: &[MemoryRecord],
    stores: Option<&[StoreType]>,
    encryption_key: Option<&[u8; 32]>,
) -> Result<(Vec<u8>, ExportResponse), CerememoryError> {
    // Filter by stores if specified
    let filtered: Vec<&MemoryRecord> = if let Some(store_filter) = stores {
        records
            .iter()
            .filter(|r| store_filter.contains(&r.store))
            .collect()
    } else {
        records.iter().collect()
    };

    // Convert to owned for serialization
    let owned: Vec<MemoryRecord> = filtered.into_iter().cloned().collect();
    let (bytes, resp) = export(&owned)?;

    // Encrypt if key provided
    if let Some(key) = encryption_key {
        let encrypted = crypto::encrypt(&bytes, key)?;
        let mut resp = resp;
        resp.size_bytes = encrypted.len() as u64;
        Ok((encrypted, resp))
    } else {
        Ok((bytes, resp))
    }
}

/// Import records with optional decryption.
///
/// If `decryption_key` is `Some`, the data is decrypted before parsing.
/// Otherwise, the data is treated as plaintext CMA archive bytes.
pub fn import_records_with_key(
    data: &[u8],
    decryption_key: Option<&[u8; 32]>,
) -> Result<Vec<MemoryRecord>, CerememoryError> {
    let plaintext = if let Some(key) = decryption_key {
        crypto::decrypt(data, key)?
    } else {
        data.to_vec()
    };
    import_records(&plaintext)
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

    // ─── Store filter tests ─────────────────────────────────────────

    fn make_mixed_records() -> Vec<MemoryRecord> {
        vec![
            MemoryRecord::new_text(StoreType::Episodic, "Episodic memory 1".to_string()),
            MemoryRecord::new_text(StoreType::Semantic, "Semantic memory 1".to_string()),
            MemoryRecord::new_text(StoreType::Procedural, "Procedural memory 1".to_string()),
            MemoryRecord::new_text(StoreType::Episodic, "Episodic memory 2".to_string()),
            MemoryRecord::new_text(StoreType::Emotional, "Emotional memory 1".to_string()),
        ]
    }

    #[test]
    fn export_single_store_filter() {
        let records = make_mixed_records();
        let (bytes, resp) =
            export_filtered(&records, Some(&[StoreType::Episodic]), None).unwrap();
        assert_eq!(resp.record_count, 2);
        let imported = import_records(&bytes).unwrap();
        assert_eq!(imported.len(), 2);
        assert!(imported.iter().all(|r| r.store == StoreType::Episodic));
    }

    #[test]
    fn export_multi_store_filter() {
        let records = make_mixed_records();
        let (bytes, resp) = export_filtered(
            &records,
            Some(&[StoreType::Semantic, StoreType::Procedural]),
            None,
        )
        .unwrap();
        assert_eq!(resp.record_count, 2);
        let imported = import_records(&bytes).unwrap();
        assert_eq!(imported.len(), 2);
        assert!(imported
            .iter()
            .all(|r| r.store == StoreType::Semantic || r.store == StoreType::Procedural));
    }

    #[test]
    fn export_no_filter() {
        let records = make_mixed_records();
        let (_, resp) = export_filtered(&records, None, None).unwrap();
        assert_eq!(resp.record_count, 5);
    }

    #[test]
    fn export_encrypted() {
        let records = make_records(3);
        let key = crypto::derive_key("test-key");
        let (encrypted_bytes, resp) = export_filtered(&records, None, Some(&key)).unwrap();
        assert_eq!(resp.record_count, 3);

        // Encrypted data should not be valid CMA archive
        let result = import_records(&encrypted_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn encrypted_roundtrip() {
        let records = make_records(5);
        let key = crypto::derive_key("roundtrip-key");

        let (encrypted, _) = export_filtered(&records, None, Some(&key)).unwrap();
        let imported = import_records_with_key(&encrypted, Some(&key)).unwrap();
        assert_eq!(imported.len(), 5);
        for (orig, imp) in records.iter().zip(imported.iter()) {
            assert_eq!(orig.id, imp.id);
            assert_eq!(orig.text_content(), imp.text_content());
        }
    }

    #[test]
    fn encrypted_wrong_key_fails() {
        let records = make_records(2);
        let key = crypto::derive_key("correct-key");
        let wrong = crypto::derive_key("wrong-key");

        let (encrypted, _) = export_filtered(&records, None, Some(&key)).unwrap();
        let result = import_records_with_key(&encrypted, Some(&wrong));
        assert!(result.is_err());
    }

    #[test]
    fn unencrypted_import_with_no_key() {
        let records = make_records(3);
        let bytes = export_to_bytes(&records).unwrap();
        let imported = import_records_with_key(&bytes, None).unwrap();
        assert_eq!(imported.len(), 3);
    }

    #[test]
    fn export_filter_and_encrypt_combined() {
        let records = make_mixed_records();
        let key = crypto::derive_key("combined-test");

        let (encrypted, resp) =
            export_filtered(&records, Some(&[StoreType::Emotional]), Some(&key)).unwrap();
        assert_eq!(resp.record_count, 1);

        let imported = import_records_with_key(&encrypted, Some(&key)).unwrap();
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].store, StoreType::Emotional);
    }
}
