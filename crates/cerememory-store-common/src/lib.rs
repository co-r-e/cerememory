//! Shared utilities for Cerememory redb-backed store implementations.
//!
//! Provides common helper functions used by the emotional, episodic, and
//! procedural stores to avoid code duplication.

use cerememory_core::error::CerememoryError;
use cerememory_core::types::{MemoryRecord, Modality};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use rand::Rng;
use redb::ReadableTable;
use serde::{de::DeserializeOwned, Serialize};
use uuid::Uuid;
use zeroize::Zeroize;

const ENCRYPTED_RECORD_MAGIC: &[u8; 4] = b"CMR1";
const STORE_KEY_SALT: &[u8] = b"cerememory-store-record-v1";
const STORE_KEY_LEN: usize = 32;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;

/// Convert any `Display` error into `CerememoryError::Storage`.
pub fn storage_err(e: impl std::fmt::Display) -> CerememoryError {
    CerememoryError::Storage(e.to_string())
}

/// Codec for redb record payloads.
///
/// Unencrypted payloads remain plain MessagePack for backward compatibility.
/// Encrypted payloads are framed as `CMR1 || nonce(12) || ciphertext+tag`.
#[derive(Clone)]
pub struct StoreRecordCodec {
    key: Option<[u8; STORE_KEY_LEN]>,
}

impl StoreRecordCodec {
    /// Plain MessagePack codec.
    pub fn plaintext() -> Self {
        Self { key: None }
    }

    /// Encrypted codec derived from a user-held passphrase.
    pub fn encrypted_from_passphrase(passphrase: &str) -> Result<Self, CerememoryError> {
        let trimmed = passphrase.trim();
        if trimmed.is_empty() {
            return Err(CerememoryError::Validation(
                "store encryption passphrase must not be empty".to_string(),
            ));
        }

        let params = argon2::Params::new(65_536, 3, 1, Some(STORE_KEY_LEN)).map_err(|e| {
            CerememoryError::Internal(format!("Failed to configure store key derivation: {e}"))
        })?;
        let argon2 =
            argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
        let mut key = [0u8; STORE_KEY_LEN];
        argon2
            .hash_password_into(trimmed.as_bytes(), STORE_KEY_SALT, &mut key)
            .map_err(|e| CerememoryError::Internal(format!("Store key derivation failed: {e}")))?;

        Ok(Self { key: Some(key) })
    }

    /// Return whether new payloads are encrypted.
    pub fn encrypts_new_records(&self) -> bool {
        self.key.is_some()
    }

    /// Return whether a payload uses Cerememory's encrypted record framing.
    pub fn is_encrypted_payload(payload: &[u8]) -> bool {
        payload.starts_with(ENCRYPTED_RECORD_MAGIC)
    }

    /// Encode a serializable value for redb storage.
    pub fn encode<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, CerememoryError> {
        let mut packed = rmp_serde::to_vec(value)
            .map_err(|e| CerememoryError::Serialization(format!("msgpack encode: {e}")))?;

        let Some(key) = &self.key else {
            return Ok(packed);
        };

        let cipher = ChaCha20Poly1305::new(key.into());
        let mut nonce_bytes = [0u8; NONCE_LEN];
        rand::rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher.encrypt(nonce, packed.as_slice()).map_err(|e| {
            CerememoryError::Storage(format!("Failed to encrypt store record: {e}"))
        })?;
        packed.zeroize();

        let mut framed =
            Vec::with_capacity(ENCRYPTED_RECORD_MAGIC.len() + NONCE_LEN + ciphertext.len());
        framed.extend_from_slice(ENCRYPTED_RECORD_MAGIC);
        framed.extend_from_slice(&nonce_bytes);
        framed.extend_from_slice(&ciphertext);
        Ok(framed)
    }

    /// Decode either encrypted framed data or legacy plain MessagePack.
    pub fn decode<T: DeserializeOwned>(&self, payload: &[u8]) -> Result<T, CerememoryError> {
        let decoded = if payload.starts_with(ENCRYPTED_RECORD_MAGIC) {
            self.decrypt_payload(payload)?
        } else {
            payload.to_vec()
        };

        rmp_serde::from_slice(&decoded)
            .map_err(|e| CerememoryError::Serialization(format!("msgpack decode: {e}")))
    }

    fn decrypt_payload(&self, payload: &[u8]) -> Result<Vec<u8>, CerememoryError> {
        let Some(key) = &self.key else {
            return Err(CerememoryError::Unauthorized(
                "store encryption key is required to read encrypted records".to_string(),
            ));
        };

        let min_len = ENCRYPTED_RECORD_MAGIC.len() + NONCE_LEN + TAG_LEN;
        if payload.len() < min_len {
            return Err(CerememoryError::Storage(
                "encrypted store record is truncated".to_string(),
            ));
        }

        let nonce_start = ENCRYPTED_RECORD_MAGIC.len();
        let nonce_end = nonce_start + NONCE_LEN;
        let nonce = Nonce::from_slice(&payload[nonce_start..nonce_end]);
        let ciphertext = &payload[nonce_end..];
        let cipher = ChaCha20Poly1305::new(key.into());
        cipher.decrypt(nonce, ciphertext).map_err(|_| {
            CerememoryError::Unauthorized(
                "failed to decrypt store record; wrong key or corrupted data".to_string(),
            )
        })
    }
}

/// Result of rewriting legacy plaintext store payloads to the active codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct StoreRecordMigrationStats {
    pub records_total: usize,
    pub records_migrated: usize,
    pub records_already_encrypted: usize,
}

impl StoreRecordMigrationStats {
    pub fn empty() -> Self {
        Self {
            records_total: 0,
            records_migrated: 0,
            records_already_encrypted: 0,
        }
    }
}

impl Default for StoreRecordCodec {
    fn default() -> Self {
        Self::plaintext()
    }
}

impl Drop for StoreRecordCodec {
    fn drop(&mut self) {
        if let Some(key) = &mut self.key {
            key.zeroize();
        }
    }
}

/// Read and deserialize a `MemoryRecord` from a redb read-only table.
pub fn get_record_sync(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    id: &Uuid,
) -> Result<Option<MemoryRecord>, CerememoryError> {
    get_record_sync_with_codec(table, id, &StoreRecordCodec::plaintext())
}

/// Read and deserialize a `MemoryRecord` using the provided codec.
pub fn get_record_sync_with_codec(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    id: &Uuid,
    codec: &StoreRecordCodec,
) -> Result<Option<MemoryRecord>, CerememoryError> {
    match table.get(id.as_bytes().as_slice()).map_err(storage_err)? {
        Some(value_guard) => Ok(Some(codec.decode(value_guard.value())?)),
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
    get_all_sync_with_codec(table, &StoreRecordCodec::plaintext())
}

/// Read all records from a redb table with the provided codec.
pub fn get_all_sync_with_codec(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    codec: &StoreRecordCodec,
) -> Result<Vec<MemoryRecord>, CerememoryError> {
    let mut records = Vec::new();
    for entry in table.iter().map_err(storage_err)? {
        let (_, value) = entry.map_err(storage_err)?;
        let record: MemoryRecord = codec.decode(value.value())?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::types::StoreType;

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

    #[test]
    fn encrypted_codec_roundtrips_and_differs_from_msgpack() {
        let codec = StoreRecordCodec::encrypted_from_passphrase("test passphrase").unwrap();
        let record = MemoryRecord::new_text(StoreType::Episodic, "secret memory");
        let plain = rmp_serde::to_vec(&record).unwrap();
        let encoded = codec.encode(&record).unwrap();

        assert_ne!(encoded, plain);
        assert!(encoded.starts_with(ENCRYPTED_RECORD_MAGIC));

        let decoded: MemoryRecord = codec.decode(&encoded).unwrap();
        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.text_content(), Some("secret memory"));
    }

    #[test]
    fn plaintext_codec_reads_legacy_msgpack() {
        let codec = StoreRecordCodec::plaintext();
        let record = MemoryRecord::new_text(StoreType::Episodic, "legacy memory");
        let plain = rmp_serde::to_vec(&record).unwrap();

        let decoded: MemoryRecord = codec.decode(&plain).unwrap();
        assert_eq!(decoded.id, record.id);
    }

    #[test]
    fn encrypted_payload_requires_key() {
        let encrypted = StoreRecordCodec::encrypted_from_passphrase("test passphrase").unwrap();
        let record = MemoryRecord::new_text(StoreType::Episodic, "secret memory");
        let encoded = encrypted.encode(&record).unwrap();
        let err = StoreRecordCodec::plaintext()
            .decode::<MemoryRecord>(&encoded)
            .unwrap_err();

        assert!(matches!(err, CerememoryError::Unauthorized(_)));
    }

    #[test]
    fn encrypted_payload_rejects_wrong_key() {
        let encrypted = StoreRecordCodec::encrypted_from_passphrase("right passphrase").unwrap();
        let wrong = StoreRecordCodec::encrypted_from_passphrase("wrong passphrase").unwrap();
        let record = MemoryRecord::new_text(StoreType::Episodic, "secret memory");
        let encoded = encrypted.encode(&record).unwrap();
        let err = wrong.decode::<MemoryRecord>(&encoded).unwrap_err();

        assert!(matches!(err, CerememoryError::Unauthorized(_)));
    }
}
