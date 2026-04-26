//! Low-level ChaCha20-Poly1305 helpers for encrypted payload experiments.
//!
//! These helpers are not currently wired into the redb-backed memory stores.
//! Production archive encryption lives in `cerememory-archive::crypto`, which
//! adds passphrase derivation and archive-specific framing.
//!
//! The encrypted format is: `nonce (12 bytes) || ciphertext + auth tag`.
//! Each encryption generates a fresh random 96-bit nonce.

use crate::error::CerememoryError;
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use rand::Rng;
use std::io::Write;
use std::path::Path;

/// Nonce size in bytes (96 bits, as required by ChaCha20-Poly1305).
const NONCE_SIZE: usize = 12;

/// Key size in bytes (256 bits).
const KEY_SIZE: usize = 32;
/// Authentication tag size in bytes.
const TAG_SIZE: usize = 16;

/// 256-bit encryption key for ChaCha20-Poly1305.
///
/// Wraps a raw 32-byte key and provides construction helpers for
/// generation, file I/O, and raw byte access.
pub struct EncryptionKey {
    key: [u8; KEY_SIZE],
}

impl EncryptionKey {
    /// Create from raw 32-byte key material.
    pub fn from_bytes(bytes: &[u8; KEY_SIZE]) -> Self {
        Self { key: *bytes }
    }

    /// Generate a new random key using a cryptographically secure RNG.
    pub fn generate() -> Self {
        let mut key = [0u8; KEY_SIZE];
        rand::rng().fill_bytes(&mut key);
        Self { key }
    }

    /// Load key from a file path.
    ///
    /// The file must contain exactly 32 bytes of raw key material.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, CerememoryError> {
        let path = path.as_ref();
        let mut data = std::fs::read(path).map_err(|e| {
            CerememoryError::Internal(format!("Failed to read key file '{}': {e}", path.display()))
        })?;
        if data.len() != KEY_SIZE {
            return Err(CerememoryError::Internal(format!(
                "Key file '{}' has invalid size: expected {KEY_SIZE} bytes, got {}",
                path.display(),
                data.len()
            )));
        }
        let mut key = [0u8; KEY_SIZE];
        key.copy_from_slice(&data);
        data.fill(0);
        Ok(Self { key })
    }

    /// Save key to a file path.
    ///
    /// On Unix, the file is created with permissions 0600 (owner read/write only).
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), CerememoryError> {
        let path = path.as_ref();

        #[cfg(unix)]
        {
            use std::fs::Permissions;
            use std::os::unix::fs::PermissionsExt;

            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let mut temp = tempfile::Builder::new()
                .prefix(".cerememory-key")
                .tempfile_in(parent)
                .map_err(|e| {
                    CerememoryError::Internal(format!(
                        "Failed to create temporary key file in '{}': {e}",
                        parent.display()
                    ))
                })?;

            temp.as_file_mut().write_all(&self.key).map_err(|e| {
                CerememoryError::Internal(format!(
                    "Failed to write key file '{}': {e}",
                    path.display()
                ))
            })?;
            temp.as_file_mut().sync_all().map_err(|e| {
                CerememoryError::Internal(format!(
                    "Failed to flush key file '{}': {e}",
                    path.display()
                ))
            })?;

            temp.persist(path).map_err(|e| {
                CerememoryError::Internal(format!(
                    "Failed to persist key file '{}': {}",
                    path.display(),
                    e.error
                ))
            })?;

            std::fs::set_permissions(path, Permissions::from_mode(0o600)).map_err(|e| {
                CerememoryError::Internal(format!(
                    "Failed to set permissions on key file '{}': {e}",
                    path.display()
                ))
            })?;
            #[allow(clippy::needless_return)]
            return Ok(());
        }

        #[cfg(not(unix))]
        {
            std::fs::write(path, self.key).map_err(|e| {
                CerememoryError::Internal(format!(
                    "Failed to write key file '{}': {e}",
                    path.display()
                ))
            })?;
            Ok(())
        }
    }

    /// Export raw key bytes.
    pub fn as_bytes(&self) -> &[u8; KEY_SIZE] {
        &self.key
    }
}

impl Drop for EncryptionKey {
    /// Zeroize key material on drop to reduce exposure in memory.
    fn drop(&mut self) {
        // Use volatile write to prevent the compiler from optimizing this away.
        // This is a best-effort defense-in-depth measure.
        for byte in self.key.iter_mut() {
            // SAFETY: `write_volatile` prevents the compiler from optimizing
            // away the zeroing of sensitive key material. We hold an exclusive
            // `&mut self` reference, so there are no concurrent reads.
            unsafe {
                std::ptr::write_volatile(byte, 0);
            }
        }
    }
}

/// Encrypt data using ChaCha20-Poly1305 AEAD.
///
/// Returns `nonce (12 bytes) || ciphertext + authentication tag`.
/// A fresh random nonce is generated for each invocation.
pub fn encrypt(key: &EncryptionKey, plaintext: &[u8]) -> Result<Vec<u8>, CerememoryError> {
    let cipher = ChaCha20Poly1305::new((&key.key).into());

    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| CerememoryError::Internal(format!("Encryption failed: {e}")))?;

    let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypt data that was encrypted with [`encrypt`].
///
/// Expects the format `nonce (12 bytes) || ciphertext + authentication tag`.
pub fn decrypt(key: &EncryptionKey, data: &[u8]) -> Result<Vec<u8>, CerememoryError> {
    if data.len() < NONCE_SIZE {
        return Err(CerememoryError::Validation(
            "Encrypted data too short: missing nonce".into(),
        ));
    }
    if data.len() < NONCE_SIZE + TAG_SIZE {
        return Err(CerememoryError::Validation(
            "Encrypted data too short: missing authentication tag".into(),
        ));
    }

    let (nonce_bytes, ciphertext) = data.split_at(NONCE_SIZE);
    let nonce = Nonce::from_slice(nonce_bytes);
    let cipher = ChaCha20Poly1305::new((&key.key).into());

    cipher.decrypt(nonce, ciphertext).map_err(|_| {
        CerememoryError::Internal("Decryption failed: wrong key or corrupted data".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = EncryptionKey::generate();
        let plaintext = b"Hello, Cerememory encryption at rest!";

        let encrypted = encrypt(&key, plaintext).unwrap();
        // Encrypted data must differ from plaintext
        assert_ne!(&encrypted[..], &plaintext[..]);
        // Size: nonce (12) + plaintext + auth tag (16)
        assert_eq!(encrypted.len(), NONCE_SIZE + plaintext.len() + 16);

        let decrypted = decrypt(&key, &encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn different_plaintexts_produce_different_ciphertexts() {
        let key = EncryptionKey::generate();
        let ct1 = encrypt(&key, b"message one").unwrap();
        let ct2 = encrypt(&key, b"message two").unwrap();

        // Different plaintexts must produce different ciphertexts
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn same_plaintext_produces_different_ciphertexts_due_to_random_nonce() {
        let key = EncryptionKey::generate();
        let plaintext = b"same message";
        let ct1 = encrypt(&key, plaintext).unwrap();
        let ct2 = encrypt(&key, plaintext).unwrap();

        // Random nonces ensure different ciphertexts even for identical plaintext
        assert_ne!(ct1, ct2);

        // Both must decrypt to the same plaintext
        assert_eq!(decrypt(&key, &ct1).unwrap(), plaintext);
        assert_eq!(decrypt(&key, &ct2).unwrap(), plaintext);
    }

    #[test]
    fn decrypt_with_wrong_key_fails() {
        let key1 = EncryptionKey::generate();
        let key2 = EncryptionKey::generate();
        let plaintext = b"secret data";

        let encrypted = encrypt(&key1, plaintext).unwrap();
        let result = decrypt(&key2, &encrypted);

        let err = result.expect_err("should fail with wrong key");
        let msg = format!("{err}");
        assert!(msg.contains("Decryption failed"), "unexpected error: {msg}");
    }

    #[test]
    fn decrypt_truncated_data_fails() {
        let key = EncryptionKey::generate();

        // Too short to contain even a nonce
        let result = decrypt(&key, &[0u8; 5]);
        let err = result.expect_err("should fail on truncated data");
        let msg = format!("{err}");
        assert!(msg.contains("too short"), "unexpected error: {msg}");
    }

    #[test]
    fn decrypt_missing_auth_tag_fails() {
        let key = EncryptionKey::generate();
        let result = decrypt(&key, &[0u8; NONCE_SIZE + TAG_SIZE - 1]);
        let err = result.expect_err("should fail without full authentication tag");
        let msg = format!("{err}");
        assert!(
            msg.contains("authentication tag"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn decrypt_corrupted_ciphertext_fails() {
        let key = EncryptionKey::generate();
        let plaintext = b"important data";

        let mut encrypted = encrypt(&key, plaintext).unwrap();
        // Flip a byte in the ciphertext portion (after the nonce)
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0xFF;

        let result = decrypt(&key, &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn key_generation_produces_valid_keys() {
        let key1 = EncryptionKey::generate();
        let key2 = EncryptionKey::generate();

        // Two generated keys should differ (probability of collision is negligible)
        assert_ne!(key1.as_bytes(), key2.as_bytes());

        // Both should be usable for encryption
        let plaintext = b"test";
        let ct1 = encrypt(&key1, plaintext).unwrap();
        let ct2 = encrypt(&key2, plaintext).unwrap();
        assert_eq!(decrypt(&key1, &ct1).unwrap(), plaintext);
        assert_eq!(decrypt(&key2, &ct2).unwrap(), plaintext);
    }

    #[test]
    fn key_from_bytes_roundtrip() {
        let original = EncryptionKey::generate();
        let bytes = *original.as_bytes();
        let restored = EncryptionKey::from_bytes(&bytes);

        assert_eq!(original.as_bytes(), restored.as_bytes());

        // Restored key must decrypt data encrypted with the original
        let plaintext = b"roundtrip test";
        let encrypted = encrypt(&original, plaintext).unwrap();
        let decrypted = decrypt(&restored, &encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn empty_plaintext_roundtrip() {
        let key = EncryptionKey::generate();
        let plaintext = b"";

        let encrypted = encrypt(&key, plaintext).unwrap();
        // Even empty plaintext produces nonce + auth tag
        assert_eq!(encrypted.len(), NONCE_SIZE + 16);

        let decrypted = decrypt(&key, &encrypted).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn large_plaintext_roundtrip() {
        let key = EncryptionKey::generate();
        let plaintext = vec![0xABu8; 1024 * 1024]; // 1 MB

        let encrypted = encrypt(&key, &plaintext).unwrap();
        let decrypted = decrypt(&key, &encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn key_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.key");

        let key = EncryptionKey::generate();
        key.save_to_file(&path).unwrap();

        // Verify file permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = std::fs::metadata(&path).unwrap();
            let mode = meta.permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "Key file must have 0600 permissions");
        }

        let loaded = EncryptionKey::from_file(&path).unwrap();
        assert_eq!(key.as_bytes(), loaded.as_bytes());
    }

    #[test]
    fn key_from_file_wrong_size_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.key");

        // Write a file with wrong size
        std::fs::write(&path, [0u8; 16]).unwrap();

        let result = EncryptionKey::from_file(&path);
        match result {
            Ok(_) => panic!("should fail on wrong-size key file"),
            Err(err) => {
                let msg = format!("{err}");
                assert!(msg.contains("invalid size"), "unexpected error: {msg}");
            }
        }
    }

    #[test]
    fn key_from_nonexistent_file_fails() {
        let result = EncryptionKey::from_file("/nonexistent/path/to/key.file");
        match result {
            Ok(_) => panic!("should fail on nonexistent file"),
            Err(err) => {
                let msg = format!("{err}");
                assert!(
                    msg.contains("Failed to read key file"),
                    "unexpected error: {msg}"
                );
            }
        }
    }
}
