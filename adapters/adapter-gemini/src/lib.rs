//! Cerememory LLM adapter for Google Gemini.
//!
//! Implements the `LLMAdapter` trait for Gemini models.
//! Uses Markdown structured text for memory serialization.
//!
//! Also provides [`GeminiProvider`], an [`LLMProvider`](cerememory_core::LLMProvider)
//! implementation that calls the Gemini REST API for embedding,
//! summarization, and relation extraction.

pub mod provider;

pub use provider::GeminiProvider;

use cerememory_core::{
    estimate_tokens_from_bytes, LLMAdapter, MemoryContent, MemoryRecord, ModelInfo,
};

/// LLM adapter for Google Gemini models.
///
/// Serializes memories as Markdown-formatted text with headers and metadata,
/// which works well with Gemini's text processing.
/// Phase 1: text-only stub with byte-based token estimation.
#[derive(Debug, Clone, Default)]
pub struct GeminiAdapter;

impl GeminiAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl LLMAdapter for GeminiAdapter {
    fn serialize_context(&self, memories: &[MemoryRecord], budget_tokens: usize) -> String {
        let mut output = String::new();
        let mut remaining_tokens = budget_tokens;

        for record in memories {
            let text = match record.text_content() {
                Some(t) => t,
                None => continue,
            };

            let entry = format!(
                "## Memory [{}] (store: {}, fidelity: {:.2})\n{}\n\n",
                record.id, record.store, record.fidelity.score, text,
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
        let summary_bytes = content.summary.as_ref().map(|s| s.len()).unwrap_or(0);
        estimate_tokens_from_bytes(total_bytes + summary_bytes)
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            provider: "google".to_string(),
            model_name: "gemini-2.0-flash".to_string(),
            max_context_tokens: 1_000_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::{MemoryRecord, StoreType};

    fn make_adapter() -> GeminiAdapter {
        GeminiAdapter::new()
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
        assert!(output.contains("## Memory"));
        assert!(output.contains("Hello, world!"));
        assert!(output.contains("episodic"));
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
        assert!(tokens > 0);
        assert!(tokens <= 13);
    }

    #[test]
    fn model_info_returns_correct_provider() {
        let adapter = make_adapter();
        let info = adapter.model_info();
        assert_eq!(info.provider, "google");
        assert_eq!(info.model_name, "gemini-2.0-flash");
        assert_eq!(info.max_context_tokens, 1_000_000);
    }
}
