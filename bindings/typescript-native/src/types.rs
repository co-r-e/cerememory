//! Conversion helpers between Cerememory Rust types and serde_json::Value.
//!
//! These functions produce clean JSON representations suitable for
//! consumption by JavaScript/TypeScript code via napi-rs's serde-json bridge.

use cerememory_core::protocol::StatsResponse;
use cerememory_core::types::MemoryRecord;

/// Convert a `MemoryRecord` to a JSON value.
///
/// Leverages the record's `Serialize` derive for a faithful representation.
pub fn record_to_json(record: &MemoryRecord) -> napi::Result<serde_json::Value> {
    serde_json::to_value(record)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize record: {e}")))
}

/// Convert a `StatsResponse` to a JSON value.
///
/// Leverages the response's `Serialize` derive for full fidelity.
pub fn stats_to_json(stats: &StatsResponse) -> napi::Result<serde_json::Value> {
    serde_json::to_value(stats)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize stats: {e}")))
}

/// Map a `CerememoryError` to a `napi::Error` with an appropriate message.
///
/// Preserves the full error chain via `Display` for debuggability.
pub fn to_napi_error(err: cerememory_core::error::CerememoryError) -> napi::Error {
    napi::Error::from_reason(format!("{err}"))
}

/// Parse a store type string (e.g. "episodic") into `StoreType`.
///
/// Returns a napi error if the string doesn't match any known variant.
pub fn parse_store_type(s: &str) -> napi::Result<cerememory_core::types::StoreType> {
    s.to_lowercase()
        .parse::<cerememory_core::types::StoreType>()
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Parse a UUID string, returning a napi error on invalid input.
pub fn parse_uuid(s: &str) -> napi::Result<uuid::Uuid> {
    s.parse::<uuid::Uuid>()
        .map_err(|e| napi::Error::from_reason(format!("Invalid UUID '{s}': {e}")))
}

/// Convert a `RecallQueryResponse` to a JSON value.
pub fn recall_response_to_json(
    resp: &cerememory_core::protocol::RecallQueryResponse,
) -> napi::Result<serde_json::Value> {
    serde_json::to_value(resp)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize recall response: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::types::StoreType;

    #[test]
    fn parse_store_type_is_case_insensitive() {
        assert_eq!(parse_store_type("EPISODIC").unwrap(), StoreType::Episodic);
        assert_eq!(parse_store_type("semantic").unwrap(), StoreType::Semantic);
    }

    #[test]
    fn parse_uuid_rejects_invalid_values() {
        let err = parse_uuid("not-a-uuid").unwrap_err();
        assert!(err.reason.contains("Invalid UUID"));
    }

    #[test]
    fn record_to_json_serializes_core_fields() {
        let record = MemoryRecord::new_text(StoreType::Semantic, "binding test");
        let json = record_to_json(&record).unwrap();

        assert_eq!(json["store"], "semantic");
        assert_eq!(json["content"]["blocks"][0]["modality"], "text");
    }
}
