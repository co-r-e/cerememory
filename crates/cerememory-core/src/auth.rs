//! Shared authentication utilities.
//!
//! Constant-time API key validation used by both HTTP and gRPC transports.

use subtle::ConstantTimeEq;

/// Validate a token against a list of pre-stored API keys using constant-time comparison.
///
/// Uses HMAC-style fixed-length comparison: both inputs are compared byte-by-byte
/// up to the length of the longer input to avoid length-based timing leaks.
///
/// Returns `true` if the token matches any of the keys.
pub fn validate_api_key(token: &[u8], keys: &[Vec<u8>]) -> bool {
    keys.iter().any(|key| constant_time_eq(key, token))
}

/// Constant-time equality comparison that avoids length-based timing leaks.
///
/// When lengths differ, pads the shorter input with zeros and compares the
/// full length, then rejects — so all comparisons take time proportional
/// to `max(a.len(), b.len())` regardless of whether lengths match.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        // Compare against zero-padded version to keep timing uniform.
        // Result is always false since lengths differ.
        let max_len = a.len().max(b.len());
        let mut padded_a = vec![0u8; max_len];
        let mut padded_b = vec![0u8; max_len];
        padded_a[..a.len()].copy_from_slice(a);
        padded_b[..b.len()].copy_from_slice(b);
        let _ = padded_a.ct_eq(&padded_b);
        false
    } else {
        a.ct_eq(b).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_key_returns_true() {
        let keys = vec![b"secret-key".to_vec()];
        assert!(validate_api_key(b"secret-key", &keys));
    }

    #[test]
    fn wrong_key_returns_false() {
        let keys = vec![b"secret-key".to_vec()];
        assert!(!validate_api_key(b"wrong-key", &keys));
    }

    #[test]
    fn different_length_returns_false() {
        let keys = vec![b"short".to_vec()];
        assert!(!validate_api_key(b"much-longer-key", &keys));
    }

    #[test]
    fn multiple_keys_any_match() {
        let keys = vec![b"key-1".to_vec(), b"key-2".to_vec()];
        assert!(validate_api_key(b"key-2", &keys));
        assert!(!validate_api_key(b"key-3", &keys));
    }

    #[test]
    fn empty_keys_returns_false() {
        let keys: Vec<Vec<u8>> = vec![];
        assert!(!validate_api_key(b"anything", &keys));
    }
}
