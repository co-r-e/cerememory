//! Shared utilities for Cerememory redb-backed store implementations.
//!
//! Provides common helper functions used by the emotional, episodic, and
//! procedural stores to avoid code duplication.

use cerememory_core::error::CerememoryError;
use cerememory_core::types::{MemoryRecord, Modality};
use redb::ReadableTable;
use uuid::Uuid;

/// Convert any `Display` error into `CerememoryError::Storage`.
pub fn storage_err(e: impl std::fmt::Display) -> CerememoryError {
    CerememoryError::Storage(e.to_string())
}

/// Read and deserialize a `MemoryRecord` from a redb read-only table.
pub fn get_record_sync(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    id: &Uuid,
) -> Result<Option<MemoryRecord>, CerememoryError> {
    match table.get(id.as_bytes().as_slice()).map_err(storage_err)? {
        Some(value_guard) => {
            let record: MemoryRecord = rmp_serde::from_slice(value_guard.value())
                .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;
            Ok(Some(record))
        }
        None => Ok(None),
    }
}

/// Check if a record matches a text query (case-insensitive substring match).
///
/// Searches both text content blocks and the summary field.
pub fn record_matches_text(record: &MemoryRecord, query_lower: &str) -> bool {
    for block in &record.content.blocks {
        if block.modality == Modality::Text {
            if let Ok(text) = std::str::from_utf8(&block.data) {
                if text.to_lowercase().contains(query_lower) {
                    return true;
                }
            }
        }
    }
    if let Some(ref summary) = record.content.summary {
        if summary.to_lowercase().contains(query_lower) {
            return true;
        }
    }
    false
}

/// Build a 17-byte fidelity index key: `[bucket(1)] ++ [uuid(16)]`.
pub fn fidelity_key(fidelity_score: f64, id: &Uuid) -> [u8; 17] {
    let mut buf = [0u8; 17];
    buf[0] = fidelity_bucket(fidelity_score);
    buf[1..].copy_from_slice(id.as_bytes());
    buf
}

/// Map a fidelity score (0.0–1.0) to a single-byte bucket (0–100).
pub fn fidelity_bucket(score: f64) -> u8 {
    (score * 100.0).round().clamp(0.0, 100.0) as u8
}

/// Read all records from a redb table in a single transaction.
///
/// Iterates the entire table and deserializes each value as a `MemoryRecord`.
pub fn get_all_sync(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
) -> Result<Vec<MemoryRecord>, CerememoryError> {
    let mut records = Vec::new();
    for entry in table.iter().map_err(storage_err)? {
        let (_, value) = entry.map_err(storage_err)?;
        let record: MemoryRecord = rmp_serde::from_slice(value.value())
            .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fidelity_bucket_clamps() {
        assert_eq!(fidelity_bucket(0.0), 0);
        assert_eq!(fidelity_bucket(0.5), 50);
        assert_eq!(fidelity_bucket(1.0), 100);
        assert_eq!(fidelity_bucket(1.5), 100);
        assert_eq!(fidelity_bucket(-0.1), 0);
    }

    #[test]
    fn fidelity_key_layout() {
        let id = Uuid::nil();
        let key = fidelity_key(0.75, &id);
        assert_eq!(key[0], 75);
        assert_eq!(&key[1..], id.as_bytes());
    }
}
