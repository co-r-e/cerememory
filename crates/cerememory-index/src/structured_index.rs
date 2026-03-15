//! Structured data flattening for Cerememory's Structured modality.
//!
//! Converts JSON data into searchable key-value text that can be fed into
//! the existing [`super::text_index::TextIndex`]. This allows structured records
//! (e.g. user profiles, configuration, metadata) to participate in full-text search.

use cerememory_core::error::CerememoryError;
use serde_json::Value;

/// Maximum nesting depth allowed when flattening JSON.
/// Prevents abuse from deeply nested payloads that would produce
/// excessively long key prefixes or blow the stack.
const MAX_DEPTH: usize = 10;

/// Flatten a JSON value into searchable text.
///
/// # Examples
///
/// ```text
/// {"name": "Alice", "age": 30, "tags": ["rust", "python"]}
/// → "name: Alice\nage: 30\ntags: rust\ntags: python"
/// ```
///
/// Nested objects use dot-separated keys:
/// ```text
/// {"a": {"b": 1}} → "a.b: 1"
/// ```
///
/// # Errors
///
/// Returns [`CerememoryError::Validation`] if the input is not valid JSON
/// or if nesting exceeds [`MAX_DEPTH`] levels.
pub fn flatten_json_to_text(data: &[u8]) -> Result<String, CerememoryError> {
    let value: Value = serde_json::from_slice(data)
        .map_err(|e| CerememoryError::Validation(format!("Invalid JSON: {e}")))?;

    let mut lines: Vec<String> = Vec::new();
    flatten_value(&value, "", 0, &mut lines)?;
    Ok(lines.join("\n"))
}

fn push_leaf(lines: &mut Vec<String>, prefix: &str, value: impl std::fmt::Display) {
    if prefix.is_empty() {
        lines.push(value.to_string());
    } else {
        lines.push(format!("{prefix}: {value}"));
    }
}

/// Recursively flatten a [`serde_json::Value`] into `key: value` lines.
fn flatten_value(
    value: &Value,
    prefix: &str,
    depth: usize,
    lines: &mut Vec<String>,
) -> Result<(), CerememoryError> {
    if depth > MAX_DEPTH {
        return Err(CerememoryError::Validation(format!(
            "JSON nesting exceeds maximum depth of {MAX_DEPTH}"
        )));
    }

    match value {
        Value::Object(map) => {
            for (key, val) in map {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_value(val, &new_prefix, depth + 1, lines)?;
            }
        }
        Value::Array(arr) => {
            for item in arr {
                flatten_value(item, prefix, depth + 1, lines)?;
            }
        }
        Value::String(s) => {
            push_leaf(lines, prefix, s);
        }
        Value::Number(n) => {
            push_leaf(lines, prefix, n);
        }
        Value::Bool(b) => {
            push_leaf(lines, prefix, b);
        }
        Value::Null => {
            push_leaf(lines, prefix, "null");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_flat_object() {
        let json = br#"{"name": "Alice", "city": "Tokyo"}"#;
        let text = flatten_json_to_text(json).unwrap();
        assert!(text.contains("name: Alice"));
        assert!(text.contains("city: Tokyo"));
    }

    #[test]
    fn nested_object_dot_separated_keys() {
        let json = br#"{"a": {"b": {"c": 42}}}"#;
        let text = flatten_json_to_text(json).unwrap();
        assert_eq!(text, "a.b.c: 42");
    }

    #[test]
    fn arrays_repeat_key() {
        let json = br#"{"tags": ["rust", "python", "go"]}"#;
        let text = flatten_json_to_text(json).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "tags: rust");
        assert_eq!(lines[1], "tags: python");
        assert_eq!(lines[2], "tags: go");
    }

    #[test]
    fn mixed_types() {
        let json = br#"{"s": "hello", "n": 3.14, "b": true, "null_val": null}"#;
        let text = flatten_json_to_text(json).unwrap();
        assert!(text.contains("s: hello"));
        assert!(text.contains("n: 3.14"));
        assert!(text.contains("b: true"));
        assert!(text.contains("null_val: null"));
    }

    #[test]
    fn deep_nesting_limit() {
        // Build JSON nested 12 levels deep: {"a":{"a":{"a":...}}}
        let mut json = String::from("1");
        for _ in 0..12 {
            json = format!(r#"{{"a": {json}}}"#);
        }
        let result = flatten_json_to_text(json.as_bytes());
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CerememoryError::Validation(msg) => {
                assert!(
                    msg.contains("nesting"),
                    "Expected nesting error, got: {msg}"
                );
            }
            other => panic!("Expected Validation error, got: {other:?}"),
        }
    }

    #[test]
    fn invalid_json_error() {
        let bad = b"not json at all";
        let result = flatten_json_to_text(bad);
        assert!(result.is_err());
        match result.unwrap_err() {
            CerememoryError::Validation(msg) => {
                assert!(
                    msg.contains("Invalid JSON"),
                    "Expected 'Invalid JSON', got: {msg}"
                );
            }
            other => panic!("Expected Validation error, got: {other:?}"),
        }
    }

    #[test]
    fn empty_object() {
        let json = b"{}";
        let text = flatten_json_to_text(json).unwrap();
        assert!(
            text.is_empty(),
            "Expected empty string for empty object, got: {text:?}"
        );
    }

    #[test]
    fn empty_array() {
        let json = b"[]";
        let text = flatten_json_to_text(json).unwrap();
        assert!(
            text.is_empty(),
            "Expected empty string for empty array, got: {text:?}"
        );
    }

    #[test]
    fn complex_user_profile() {
        let json = br#"{
            "user": {
                "name": "Masato Okuwaki",
                "age": 35,
                "active": true,
                "address": {
                    "city": "Tokyo",
                    "country": "Japan"
                },
                "skills": ["rust", "typescript", "go"],
                "metadata": null
            }
        }"#;
        let text = flatten_json_to_text(json).unwrap();

        assert!(text.contains("user.name: Masato Okuwaki"));
        assert!(text.contains("user.age: 35"));
        assert!(text.contains("user.active: true"));
        assert!(text.contains("user.address.city: Tokyo"));
        assert!(text.contains("user.address.country: Japan"));
        assert!(text.contains("user.skills: rust"));
        assert!(text.contains("user.skills: typescript"));
        assert!(text.contains("user.skills: go"));
        assert!(text.contains("user.metadata: null"));
    }

    #[test]
    fn nested_arrays_with_objects() {
        let json = br#"{"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}"#;
        let text = flatten_json_to_text(json).unwrap();
        assert!(text.contains("items.id: 1"));
        assert!(text.contains("items.name: a"));
        assert!(text.contains("items.id: 2"));
        assert!(text.contains("items.name: b"));
    }

    #[test]
    fn exactly_at_depth_limit() {
        // Build JSON nested exactly MAX_DEPTH (10) levels deep — should succeed.
        // The root object is depth 0, each nested object adds 1.
        // We need 10 nested objects so the innermost value is at depth 10 (leaf).
        let mut json = String::from("1");
        for _ in 0..MAX_DEPTH {
            json = format!(r#"{{"a": {json}}}"#);
        }
        let result = flatten_json_to_text(json.as_bytes());
        assert!(
            result.is_ok(),
            "Expected success at exactly MAX_DEPTH, got: {result:?}"
        );

        let expected_key = (0..MAX_DEPTH).map(|_| "a").collect::<Vec<_>>().join(".");
        let text = result.unwrap();
        assert!(text.contains(&format!("{expected_key}: 1")));
    }

    #[test]
    fn top_level_primitive() {
        // JSON allows top-level primitives
        let text = flatten_json_to_text(b"42").unwrap();
        assert_eq!(text, "42");
    }

    #[test]
    fn top_level_array_of_primitives() {
        let text = flatten_json_to_text(br#"["rust", "python"]"#).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines, vec!["rust", "python"]);
    }

    #[test]
    fn unicode_content() {
        let json = br#"{"greeting": "\u3053\u3093\u306b\u3061\u306f"}"#;
        let text = flatten_json_to_text(json).unwrap();
        assert!(text.contains("greeting:"));
    }
}
