//! Cerememory LLM adapter for Claude.
//!
//! Implements the `LLMAdapter` trait for Anthropic Claude models.
//! Uses XML tag format for memory serialization, which aligns with
//! Claude's strong XML parsing capabilities.
//!
//! Also provides [`ClaudeProvider`], an `LLMProvider` implementation that
//! uses the Anthropic Messages API for summarization and relation extraction.

pub mod provider;
pub use provider::ClaudeProvider;

use cerememory_core::{estimate_tokens_from_bytes, LLMAdapter, MemoryContent, MemoryRecord, ModelInfo};

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// LLM adapter for Anthropic Claude models.
///
/// Serializes memories using XML tags, which Claude handles natively well.
/// Phase 1: text-only stub with byte-based token estimation.
#[derive(Debug, Clone, Default)]
pub struct ClaudeAdapter;

impl ClaudeAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl LLMAdapter for ClaudeAdapter {
    fn serialize_context(&self, memories: &[MemoryRecord], budget_tokens: usize) -> String {
        let mut output = String::new();
        let mut remaining_tokens = budget_tokens;

        for record in memories {
            let text = match record.text_content() {
                Some(t) => t,
                None => continue,
            };

            let escaped = xml_escape(text);
            let entry = format!(
                "<memory id=\"{}\" store=\"{}\" fidelity=\"{:.2}\">\n{}\n</memory>\n",
                record.id, record.store, record.fidelity.score, escaped,
            );

            let entry_tokens = estimate_tokens_from_bytes(entry.len());
            if entry_tokens > remaining_tokens {
                break;
            }

            remaining_tokens -= entry_tokens;
            output.push_str(&entry);
        }

        output
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
            provider: "anthropic".to_string(),
            model_name: "claude-sonnet-4-20250514".to_string(),
            max_context_tokens: 200_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::{MemoryRecord, StoreType};

    fn make_adapter() -> ClaudeAdapter {
        ClaudeAdapter::new()
    }

    fn make_record(text: &str) -> MemoryRecord {
        MemoryRecord::new_text(StoreType::Episodic, text)
    }

    #[test]
    fn serialize_context_produces_non_empty_output() {
        let adapter = make_adapter();
        let records = vec![make_record("Hello, world!")];
        let output = adapter.serialize_context(&records, 1000);
        assert!(!output.is_empty());
        assert!(output.contains("<memory"));
        assert!(output.contains("Hello, world!"));
        assert!(output.contains("</memory>"));
    }

    #[test]
    fn serialize_context_respects_token_budget() {
        let adapter = make_adapter();
        // Create many records with substantial text
        let records: Vec<MemoryRecord> = (0..100)
            .map(|i| make_record(&format!("This is memory number {} with some longer content to consume tokens", i)))
            .collect();

        // Very small budget: should truncate significantly
        let small_output = adapter.serialize_context(&records, 50);
        let large_output = adapter.serialize_context(&records, 100_000);

        assert!(
            small_output.len() < large_output.len(),
            "Small budget output ({} bytes) should be shorter than large budget output ({} bytes)",
            small_output.len(),
            large_output.len()
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
        // "Hello, world!" is 13 bytes -> ceil(13/4) = 4 tokens
        assert!(tokens > 0);
        assert!(tokens <= 13); // at most 1 token per byte
    }

    #[test]
    fn model_info_returns_correct_provider() {
        let adapter = make_adapter();
        let info = adapter.model_info();
        assert_eq!(info.provider, "anthropic");
        assert_eq!(info.model_name, "claude-sonnet-4-20250514");
        assert_eq!(info.max_context_tokens, 200_000);
    }
}
