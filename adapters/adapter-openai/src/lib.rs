//! Cerememory LLM adapter for OpenAI.
//!
//! Implements the `LLMAdapter` trait for OpenAI models.
//! Uses JSON system message format for memory serialization.

use cerememory_core::{estimate_tokens_from_bytes, LLMAdapter, MemoryContent, MemoryRecord, ModelInfo};
use serde::Serialize;

/// LLM adapter for OpenAI models.
///
/// Serializes memories as a JSON object with a `memories` array,
/// suitable for injection into OpenAI system messages.
/// Phase 1: text-only stub with byte-based token estimation.
#[derive(Debug, Clone, Default)]
pub struct OpenAIAdapter;

impl OpenAIAdapter {
    pub fn new() -> Self {
        Self
    }
}

/// Internal struct for JSON serialization of a single memory entry.
#[derive(Serialize)]
struct MemoryEntry {
    id: String,
    store: String,
    fidelity: f64,
    content: String,
}

/// Wrapper for the memories array.
#[derive(Serialize)]
struct MemoriesPayload {
    memories: Vec<MemoryEntry>,
}

impl LLMAdapter for OpenAIAdapter {
    fn serialize_context(&self, memories: &[MemoryRecord], budget_tokens: usize) -> String {
        let mut entries = Vec::new();
        let mut accumulated_tokens = 0;

        // Account for the JSON envelope overhead: {"memories":[]}
        let envelope_overhead = r#"{"memories":[]}"#.len();
        let envelope_tokens = estimate_tokens_from_bytes(envelope_overhead);
        if envelope_tokens > budget_tokens {
            return String::new();
        }
        accumulated_tokens += envelope_tokens;

        for record in memories {
            let text = match record.text_content() {
                Some(t) => t,
                None => continue,
            };

            let entry = MemoryEntry {
                id: record.id.to_string(),
                store: record.store.to_string(),
                fidelity: record.fidelity.score,
                content: text.to_string(),
            };

            // Estimate the token cost of this entry (including JSON overhead like commas, braces)
            let entry_json = serde_json::to_string(&entry).unwrap_or_default();
            // Add 1 for the comma separator between entries
            let entry_tokens = estimate_tokens_from_bytes(entry_json.len() + 1);

            if accumulated_tokens + entry_tokens > budget_tokens {
                break;
            }

            accumulated_tokens += entry_tokens;
            entries.push(entry);
        }

        let payload = MemoriesPayload { memories: entries };
        serde_json::to_string(&payload).unwrap_or_default()
    }

    fn estimate_tokens(&self, content: &MemoryContent) -> usize {
        let total_bytes: usize = content.blocks.iter().map(|b| b.data.len()).sum();
        let summary_bytes = content
            .summary
            .as_ref()
            .map(|s| s.len())
            .unwrap_or(0);
        estimate_tokens_from_bytes(total_bytes + summary_bytes)
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            provider: "openai".to_string(),
            model_name: "gpt-4o".to_string(),
            max_context_tokens: 128_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::{MemoryRecord, StoreType};

    fn make_adapter() -> OpenAIAdapter {
        OpenAIAdapter::new()
    }

    fn make_record(text: &str) -> MemoryRecord {
        MemoryRecord::new_text(StoreType::Semantic, text)
    }

    #[test]
    fn serialize_context_produces_non_empty_output() {
        let adapter = make_adapter();
        let records = vec![make_record("Hello, world!")];
        let output = adapter.serialize_context(&records, 1000);
        assert!(!output.is_empty());

        // Validate it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let memories = parsed["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0]["content"], "Hello, world!");
    }

    #[test]
    fn serialize_context_respects_token_budget() {
        let adapter = make_adapter();
        let records: Vec<MemoryRecord> = (0..100)
            .map(|i| {
                make_record(&format!(
                    "This is memory number {} with some longer content to consume tokens",
                    i
                ))
            })
            .collect();

        let small_output = adapter.serialize_context(&records, 50);
        let large_output = adapter.serialize_context(&records, 100_000);

        // Both should be valid JSON
        let small_parsed: serde_json::Value = serde_json::from_str(&small_output).unwrap();
        let large_parsed: serde_json::Value = serde_json::from_str(&large_output).unwrap();

        let small_count = small_parsed["memories"].as_array().unwrap().len();
        let large_count = large_parsed["memories"].as_array().unwrap().len();

        assert!(
            small_count < large_count,
            "Small budget should include fewer memories ({} vs {})",
            small_count,
            large_count
        );
    }

    #[test]
    fn serialize_context_zero_budget_produces_empty() {
        let adapter = make_adapter();
        let records = vec![make_record("Hello")];
        let output = adapter.serialize_context(&records, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn estimate_tokens_returns_reasonable_value() {
        let adapter = make_adapter();
        let record = make_record("Hello, world!");
        let tokens = adapter.estimate_tokens(&record.content);
        assert!(tokens > 0);
        assert!(tokens <= 13);
    }

    #[test]
    fn model_info_returns_correct_provider() {
        let adapter = make_adapter();
        let info = adapter.model_info();
        assert_eq!(info.provider, "openai");
        assert_eq!(info.model_name, "gpt-4o");
        assert_eq!(info.max_context_tokens, 128_000);
    }
}
