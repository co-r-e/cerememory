//! Tamper-evident audit log support.
//!
//! The audit log records operation metadata, not memory contents. Each JSONL
//! entry commits to the previous entry hash, forming a SHA-256 hash chain.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use cerememory_core::error::CerememoryError;
use cerememory_core::types::StoreType;

const AUDIT_VERSION: u8 = 1;
const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditLogEntry {
    pub version: u8,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub record_ids: Vec<Uuid>,
    pub store_types: Vec<StoreType>,
    pub summary: Value,
    pub prev_hash: String,
    pub entry_hash: String,
}

#[derive(Debug, Clone, Serialize)]
struct AuditHashPayload<'a> {
    version: u8,
    sequence: u64,
    timestamp: DateTime<Utc>,
    operation: &'a str,
    record_ids: &'a [Uuid],
    store_types: &'a [StoreType],
    summary: &'a Value,
    prev_hash: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditLogVerification {
    pub entries: u64,
    pub last_sequence: u64,
    pub head_hash: [u8; 32],
}

impl AuditLogVerification {
    pub fn head_hash_hex(&self) -> String {
        hex_hash(&self.head_hash)
    }
}

#[derive(Debug)]
struct AuditLogState {
    sequence: u64,
    head_hash: String,
    file: File,
}

#[derive(Debug)]
pub struct AuditLog {
    path: PathBuf,
    state: Mutex<AuditLogState>,
}

impl AuditLog {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CerememoryError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    CerememoryError::Storage(format!(
                        "Failed to create audit log directory '{}': {e}",
                        parent.display()
                    ))
                })?;
            }
        }

        let verification = Self::verify_path(&path)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| {
                CerememoryError::Storage(format!(
                    "Failed to open audit log '{}': {e}",
                    path.display()
                ))
            })?;

        Ok(Self {
            path,
            state: Mutex::new(AuditLogState {
                sequence: verification.last_sequence,
                head_hash: verification.head_hash_hex(),
                file,
            }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn verify_path(path: impl AsRef<Path>) -> Result<AuditLogVerification, CerememoryError> {
        let path = path.as_ref();
        let file = match File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(AuditLogVerification {
                    entries: 0,
                    last_sequence: 0,
                    head_hash: parse_hash(GENESIS_HASH)?,
                });
            }
            Err(e) => {
                return Err(CerememoryError::Storage(format!(
                    "Failed to read audit log '{}': {e}",
                    path.display()
                )));
            }
        };

        let reader = BufReader::new(file);
        let mut expected_sequence = 1u64;
        let mut prev_hash = GENESIS_HASH.to_string();

        for (line_index, line) in reader.lines().enumerate() {
            let line_number = line_index + 1;
            let line = line.map_err(|e| {
                CerememoryError::Storage(format!(
                    "Failed to read audit log '{}' line {}: {e}",
                    path.display(),
                    line_number
                ))
            })?;
            if line.trim().is_empty() {
                return Err(CerememoryError::Storage(format!(
                    "Audit log '{}' line {} is blank",
                    path.display(),
                    line_number
                )));
            }

            let entry: AuditLogEntry = serde_json::from_str(&line).map_err(|e| {
                CerememoryError::Storage(format!(
                    "Audit log '{}' line {} is not valid JSON: {e}",
                    path.display(),
                    line_number
                ))
            })?;

            if entry.version != AUDIT_VERSION {
                return Err(CerememoryError::Storage(format!(
                    "Audit log '{}' line {} has unsupported version {}",
                    path.display(),
                    line_number,
                    entry.version
                )));
            }
            if entry.sequence != expected_sequence {
                return Err(CerememoryError::Storage(format!(
                    "Audit log '{}' line {} has sequence {}, expected {}",
                    path.display(),
                    line_number,
                    entry.sequence,
                    expected_sequence
                )));
            }
            if entry.prev_hash != prev_hash {
                return Err(CerememoryError::Storage(format!(
                    "Audit log '{}' line {} has previous hash {}, expected {}",
                    path.display(),
                    line_number,
                    entry.prev_hash,
                    prev_hash
                )));
            }

            let computed = compute_entry_hash(
                entry.sequence,
                entry.timestamp,
                &entry.operation,
                &entry.record_ids,
                &entry.store_types,
                &entry.summary,
                &entry.prev_hash,
            )?;
            let computed_hex = hex_hash(&computed);
            if entry.entry_hash != computed_hex {
                return Err(CerememoryError::Storage(format!(
                    "Audit log '{}' line {} hash mismatch: stored {}, computed {}",
                    path.display(),
                    line_number,
                    entry.entry_hash,
                    computed_hex
                )));
            }

            prev_hash = entry.entry_hash;
            expected_sequence += 1;
        }

        Ok(AuditLogVerification {
            entries: expected_sequence - 1,
            last_sequence: expected_sequence - 1,
            head_hash: parse_hash(&prev_hash)?,
        })
    }

    pub fn append(
        &self,
        operation: &str,
        record_ids: Vec<Uuid>,
        store_types: Vec<StoreType>,
        summary: Value,
    ) -> Result<AuditLogEntry, CerememoryError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CerememoryError::Internal("Audit log mutex poisoned".to_string()))?;
        let sequence = state.sequence + 1;
        let timestamp = Utc::now();
        let prev_hash = state.head_hash.clone();
        let hash = compute_entry_hash(
            sequence,
            timestamp,
            operation,
            &record_ids,
            &store_types,
            &summary,
            &prev_hash,
        )?;
        let entry_hash = hex_hash(&hash);
        let entry = AuditLogEntry {
            version: AUDIT_VERSION,
            sequence,
            timestamp,
            operation: operation.to_string(),
            record_ids,
            store_types,
            summary,
            prev_hash,
            entry_hash: entry_hash.clone(),
        };

        serde_json::to_writer(&mut state.file, &entry).map_err(|e| {
            CerememoryError::Storage(format!(
                "Failed to serialize audit log entry '{}': {e}",
                self.path.display()
            ))
        })?;
        state.file.write_all(b"\n").map_err(|e| {
            CerememoryError::Storage(format!(
                "Failed to write audit log '{}': {e}",
                self.path.display()
            ))
        })?;
        state.file.flush().map_err(|e| {
            CerememoryError::Storage(format!(
                "Failed to flush audit log '{}': {e}",
                self.path.display()
            ))
        })?;
        state.file.sync_data().map_err(|e| {
            CerememoryError::Storage(format!(
                "Failed to sync audit log '{}': {e}",
                self.path.display()
            ))
        })?;

        state.sequence = sequence;
        state.head_hash = entry_hash;
        Ok(entry)
    }
}

fn compute_entry_hash(
    sequence: u64,
    timestamp: DateTime<Utc>,
    operation: &str,
    record_ids: &[Uuid],
    store_types: &[StoreType],
    summary: &Value,
    prev_hash: &str,
) -> Result<[u8; 32], CerememoryError> {
    let payload = AuditHashPayload {
        version: AUDIT_VERSION,
        sequence,
        timestamp,
        operation,
        record_ids,
        store_types,
        summary,
        prev_hash,
    };
    let bytes = serde_json::to_vec(&payload).map_err(|e| {
        CerememoryError::Serialization(format!("Audit hash payload serialization failed: {e}"))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.into())
}

fn hex_hash(hash: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn parse_hash(value: &str) -> Result<[u8; 32], CerememoryError> {
    if value.len() != 64 {
        return Err(CerememoryError::Storage(format!(
            "Invalid audit hash length: expected 64 hex chars, got {}",
            value.len()
        )));
    }
    let mut hash = [0u8; 32];
    for (idx, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let hex = std::str::from_utf8(chunk).map_err(|e| {
            CerememoryError::Storage(format!("Invalid audit hash UTF-8 at byte {}: {e}", idx * 2))
        })?;
        hash[idx] = u8::from_str_radix(hex, 16).map_err(|e| {
            CerememoryError::Storage(format!("Invalid audit hash hex at byte {}: {e}", idx * 2))
        })?;
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_verify_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("audit.jsonl");
        let audit = AuditLog::open(&path).unwrap();

        audit
            .append(
                "encode.store",
                vec![Uuid::now_v7()],
                vec![StoreType::Episodic],
                serde_json::json!({"records": 1}),
            )
            .unwrap();

        let verification = AuditLog::verify_path(&path).unwrap();
        assert_eq!(verification.entries, 1);
        assert_eq!(verification.last_sequence, 1);
        assert_ne!(verification.head_hash_hex(), GENESIS_HASH);
    }

    #[test]
    fn verify_rejects_tampered_line() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("audit.jsonl");
        let audit = AuditLog::open(&path).unwrap();

        audit
            .append(
                "encode.store",
                vec![Uuid::now_v7()],
                vec![StoreType::Semantic],
                serde_json::json!({"records": 1}),
            )
            .unwrap();

        let mut contents = std::fs::read_to_string(&path).unwrap();
        contents = contents.replace("encode.store", "encode.tampered");
        std::fs::write(&path, contents).unwrap();

        let err = AuditLog::verify_path(&path).unwrap_err();
        assert!(err.to_string().contains("hash mismatch"));
    }
}
