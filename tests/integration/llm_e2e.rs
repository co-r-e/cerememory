//! LLM Adapter E2E tests.
//!
//! These tests call real LLM APIs and are gated behind environment variables:
//! - `CEREMEMORY_LLM_E2E=1` — master switch (required)
//! - `OPENAI_API_KEY` — enables OpenAI tests
//! - `ANTHROPIC_API_KEY` — enables Claude tests
//! - `GEMINI_API_KEY` — enables Gemini tests
//!
//! All tests use `#[ignore]` so they never run in normal CI.
//! Run manually: `CEREMEMORY_LLM_E2E=1 OPENAI_API_KEY=sk-... cargo test --test llm_e2e -- --ignored`

use cerememory_core::LLMProvider;

fn llm_e2e_enabled() -> bool {
    std::env::var("CEREMEMORY_LLM_E2E").is_ok_and(|v| v == "1")
}

fn provider_key(env_var: &str) -> Option<String> {
    if !llm_e2e_enabled() {
        return None;
    }
    std::env::var(env_var).ok()
}

// ─── OpenAI ─────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn openai_embed() {
    let Some(key) = provider_key("OPENAI_API_KEY") else {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_openai::OpenAIProvider::new(key, None, None).unwrap();
    let vec = provider.embed("The quick brown fox").await.unwrap();
    assert!(vec.len() > 100, "Embedding should have many dimensions");
    assert!(
        vec.iter().any(|&v| v != 0.0),
        "Embedding should contain non-zero values"
    );
}

#[tokio::test]
#[ignore]
async fn openai_summarize() {
    let Some(key) = provider_key("OPENAI_API_KEY") else {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_openai::OpenAIProvider::new(key, None, None).unwrap();
    let texts = vec![
        "Alice went to the park and met Bob.".to_string(),
        "They discussed the upcoming project deadline.".to_string(),
        "Bob suggested extending the timeline by two weeks.".to_string(),
    ];
    let summary = provider.summarize(&texts, 100).await.unwrap();
    assert!(!summary.is_empty(), "Summary should not be empty");
    assert!(
        summary.len() < 2000,
        "Summary should be concise, got {} chars",
        summary.len()
    );
}

#[tokio::test]
#[ignore]
async fn openai_extract_relations() {
    let Some(key) = provider_key("OPENAI_API_KEY") else {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_openai::OpenAIProvider::new(key, None, None).unwrap();
    let relations = provider
        .extract_relations("Alice works at Acme Corp as a senior engineer. She reports to Bob.")
        .await
        .unwrap();
    assert!(
        !relations.is_empty(),
        "Should extract at least one relation"
    );
}

// ─── Claude ─────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn claude_summarize() {
    let Some(key) = provider_key("ANTHROPIC_API_KEY") else {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_claude::ClaudeProvider::new(key, None, None).unwrap();
    let texts = vec![
        "Alice went to the park and met Bob.".to_string(),
        "They discussed the upcoming project deadline.".to_string(),
        "Bob suggested extending the timeline by two weeks.".to_string(),
    ];
    let summary = provider.summarize(&texts, 100).await.unwrap();
    assert!(!summary.is_empty(), "Summary should not be empty");
    assert!(
        summary.len() < 2000,
        "Summary should be concise, got {} chars",
        summary.len()
    );
}

#[tokio::test]
#[ignore]
async fn claude_extract_relations() {
    let Some(key) = provider_key("ANTHROPIC_API_KEY") else {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_claude::ClaudeProvider::new(key, None, None).unwrap();
    let relations = provider
        .extract_relations("Alice works at Acme Corp as a senior engineer. She reports to Bob.")
        .await
        .unwrap();
    assert!(
        !relations.is_empty(),
        "Should extract at least one relation"
    );
}

// ─── Gemini ─────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn gemini_embed() {
    let Some(key) = provider_key("GEMINI_API_KEY") else {
        eprintln!("Skipping: GEMINI_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_gemini::GeminiProvider::new(key, None, None).unwrap();
    let vec = provider.embed("The quick brown fox").await.unwrap();
    assert!(!vec.is_empty(), "Embedding should not be empty");
    assert!(
        vec.iter().any(|&v| v != 0.0),
        "Embedding should contain non-zero values"
    );
}

#[tokio::test]
#[ignore]
async fn gemini_summarize() {
    let Some(key) = provider_key("GEMINI_API_KEY") else {
        eprintln!("Skipping: GEMINI_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_gemini::GeminiProvider::new(key, None, None).unwrap();
    let texts = vec![
        "Alice went to the park and met Bob.".to_string(),
        "They discussed the upcoming project deadline.".to_string(),
        "Bob suggested extending the timeline by two weeks.".to_string(),
    ];
    let summary = provider.summarize(&texts, 100).await.unwrap();
    assert!(!summary.is_empty(), "Summary should not be empty");
    assert!(
        summary.len() < 2000,
        "Summary should be concise, got {} chars",
        summary.len()
    );
}

#[tokio::test]
#[ignore]
async fn gemini_extract_relations() {
    let Some(key) = provider_key("GEMINI_API_KEY") else {
        eprintln!("Skipping: GEMINI_API_KEY not set");
        return;
    };
    let provider = cerememory_adapter_gemini::GeminiProvider::new(key, None, None).unwrap();
    let relations = provider
        .extract_relations("Alice works at Acme Corp as a senior engineer. She reports to Bob.")
        .await
        .unwrap();
    assert!(
        !relations.is_empty(),
        "Should extract at least one relation"
    );
}
