//! Archive encryption via ChaCha20-Poly1305 (AEAD) with per-encryption salt.
//!
//! Provides authenticated encryption for CMA archives. Each encryption
//! generates a random 16-byte salt for key diversification, producing the
//! format: `salt (16 bytes) || nonce (12 bytes) || ciphertext+tag`.
//!
//! Key derivation uses Argon2id (memory-hard KDF) for passphrase-based keys,
//! and the salt-based key diversification uses an HKDF-like two-round SHA-256
//! construction for per-encryption key uniqueness.

use argon2::{Algorithm, Argon2, Params, Version};
use cerememory_core::error::CerememoryError;
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use sha2::{Digest, Sha256};
use zeroize::Zeroize;

/// Derive a 32-byte encryption key from a passphrase using Argon2id.
///
/// Uses Argon2id with 64 MiB memory, 3 iterations, and 1 lane for
/// memory-hard key derivation. A fixed domain-separated salt ensures
/// deterministic output for a given passphrase. The resulting key is
/// further diversified with a per-encryption random salt inside [`encrypt()`].
pub fn derive_key(passphrase: &str) -> [u8; 32] {
    let params = Params::new(65536, 3, 1, Some(32)).expect("valid Argon2 params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let salt = b"cerememory-archive-v2";
    let mut key = [0u8; 32];
    argon2
        .hash_password_into(passphrase.as_bytes(), salt, &mut key)
        .expect("Argon2 hash failed");
    key
}

/// Diversify a key with a random salt using HKDF-like two-round SHA-256.
///
/// This provides per-encryption key uniqueness, preventing key reuse attacks
/// even when the same passphrase-derived key is used across multiple encryptions.
fn diversify_key(key: &[u8; 32], salt: &[u8; 16]) -> [u8; 32] {
    // Extract: mix key with salt
    let mut hasher = Sha256::new();
    hasher.update(salt);
    hasher.update(key);
    let prk = hasher.finalize();

    // Expand: one more round with domain separation
    let mut hasher = Sha256::new();
    hasher.update(prk);
    hasher.update(b"cerememory-archive-key-v2");
    hasher.update([1u8]); // counter byte
    let result = hasher.finalize();

    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    // prk is dropped here; GenericArray doesn't implement Zeroize by default,
    // but the stack will be overwritten. The important zeroization happens in
    // encrypt/decrypt where the derived key is explicitly zeroized after use.
    out
}

/// Encrypt plaintext using ChaCha20-Poly1305 AEAD with per-encryption salt.
///
/// Returns `salt (16 bytes) || nonce (12 bytes) || ciphertext+tag`.
/// A random salt diversifies the key for each encryption, providing
/// unique keys even with a shared passphrase-derived base key.
pub fn encrypt(plaintext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, CerememoryError> {
    use rand::Rng;

    // Generate random salt for per-encryption key diversification
    let mut salt = [0u8; 16];
    rand::rng().fill_bytes(&mut salt);

    let mut diversified = diversify_key(key, &salt);
    let cipher = ChaCha20Poly1305::new((&diversified).into());
    diversified.zeroize(); // Zeroize derived key after use

    // Generate random 12-byte nonce
    let mut nonce_bytes = [0u8; 12];
    rand::rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| CerememoryError::ExportFailed(format!("Encryption failed: {e}")))?;

    // Format: salt (16) || nonce (12) || ciphertext+tag
    let mut result = Vec::with_capacity(16 + 12 + ciphertext.len());
    result.extend_from_slice(&salt);
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypt data that was encrypted with [`encrypt()`].
///
/// Expects the format `salt (16 bytes) || nonce (12 bytes) || ciphertext+tag`.
pub fn decrypt(data: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, CerememoryError> {
    if data.len() < 28 {
        // 16 salt + 12 nonce minimum
        return Err(CerememoryError::ImportConflict(
            "Encrypted data too short (missing salt/nonce)".to_string(),
        ));
    }

    let (salt, rest) = data.split_at(16);
    let (nonce_bytes, ciphertext) = rest.split_at(12);

    let mut salt_arr = [0u8; 16];
    salt_arr.copy_from_slice(salt);
    let mut diversified = diversify_key(key, &salt_arr);

    let nonce = Nonce::from_slice(nonce_bytes);
    let cipher = ChaCha20Poly1305::new((&diversified).into());
    diversified.zeroize(); // Zeroize derived key after use

    cipher.decrypt(nonce, ciphertext).map_err(|_| {
        CerememoryError::ImportConflict(
            "Decryption failed: wrong key or corrupted data".to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = derive_key("test-passphrase");
        let plaintext = b"Hello, Cerememory!";

        let encrypted = encrypt(plaintext, &key).unwrap();
        assert_ne!(&encrypted[..], plaintext);
        // Must be at least salt (16) + nonce (12) + tag (16) + plaintext
        assert!(encrypted.len() >= 16 + 12 + 16 + plaintext.len());

        let decrypted = decrypt(&encrypted, &key).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_key_fails() {
        let key = derive_key("correct-key");
        let wrong_key = derive_key("wrong-key");
        let plaintext = b"secret data";

        let encrypted = encrypt(plaintext, &key).unwrap();
        let result = decrypt(&encrypted, &wrong_key);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Decryption failed"));
    }

    #[test]
    fn empty_data_encrypts() {
        let key = derive_key("key");
        let plaintext = b"";

        let encrypted = encrypt(plaintext, &key).unwrap();
        let decrypted = decrypt(&encrypted, &key).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn too_short_data_fails() {
        let key = derive_key("key");
        let short = vec![0u8; 20]; // Less than 28 bytes (salt 16 + nonce 12)

        let result = decrypt(&short, &key);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("too short"));
    }

    #[test]
    fn key_derivation_deterministic() {
        let k1 = derive_key("same-passphrase");
        let k2 = derive_key("same-passphrase");
        assert_eq!(k1, k2);

        let k3 = derive_key("different-passphrase");
        assert_ne!(k1, k3);
    }

    #[test]
    fn large_payload() {
        let key = derive_key("key");
        let plaintext = vec![0xABu8; 1024 * 1024]; // 1 MB

        let encrypted = encrypt(&plaintext, &key).unwrap();
        let decrypted = decrypt(&encrypted, &key).unwrap();
        assert_eq!(decrypted, plaintext);
    }
}
