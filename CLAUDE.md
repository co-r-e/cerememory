# Cerememory

## Language

All code, comments, documentation, commit messages, and PR descriptions **must be in English**.

## Build & Test

```bash
scripts/test-rust-cool.sh -p cerememory-engine --lib  # Preferred during iteration
scripts/test-rust-cool.sh -p cerememory-integration-tests --test phase3
scripts/test-rust-workspace-cool.sh                   # Run the full suite before handoff
cargo clippy --workspace --tests -- -D warnings  # Lint
cargo fmt --all -- --check      # Format check
```

The Rust helper scripts intentionally default to a single Cargo build job and single-threaded test execution, and the TypeScript SDK Vitest config runs with a single worker by default to keep local test runs quiet.

## Architecture

Cerememory is a living memory database implementing the CMP (Cerememory Protocol). It is organized as a Cargo workspace:

- `crates/cerememory-core` — Protocol types, traits, error types
- `crates/cerememory-engine` — Orchestrator assembling all stores + decay + associations
- `crates/cerememory-store-*` — Per-store implementations (episodic, semantic, procedural, emotional, working)
- `crates/cerememory-decay` — Power-law fidelity decay engine
- `crates/cerememory-association` — Spreading activation network
- `crates/cerememory-index` — Tantivy full-text + redb-backed exact vector search
- `crates/cerememory-archive` — CMA export/import with optional encryption
- `crates/cerememory-evolution` — Self-tuning parameter evolution
- `crates/cerememory-transport-http` — Axum REST API
- `crates/cerememory-transport-grpc` — tonic gRPC transport
- `crates/cerememory-transport-mcp` — MCP (Model Context Protocol) server for Claude Code
- `crates/cerememory-config` — TOML configuration loader
- `crates/cerememory-cli` — CLI binary
- `adapters/adapter-*` — LLM provider adapters (OpenAI, Claude, Gemini)
- `bindings/` — Python and TypeScript native bindings
- `tests/integration/` — Cross-crate integration tests (phase1–phase5, llm_e2e)

## Related Repositories

- `~/Projects/dev_cerememory/cerememory-docs` — Documentation site and landing page

## Conventions

- Integration tests share helpers via `tests/integration/helpers.rs` using `#[path = "helpers.rs"] mod helpers;`
- Store iteration uses the `ALL_STORES` constant defined in `cerememory-engine`
- `EmotionVector::FromStr` in `cerememory-core` is the canonical emotion label parser; transports delegate to it
- The `dispatch_store!` macro routes operations to the correct store implementation by `StoreType`
