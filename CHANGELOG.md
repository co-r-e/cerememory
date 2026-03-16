# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-16

### Added

#### Core Engine
- Five memory stores: episodic, semantic, procedural, emotional, working
- Cerememory Protocol (CMP) with encode, recall, lifecycle, and introspect operations
- Hippocampal coordinator for cross-store indexing and vector search (hnsw_rs)
- Full-text search via Tantivy integration
- CMA archive format (SQLite-based) for export/import

#### Living Memory Dynamics
- Decay engine with power-law fidelity curves and noise accumulation
- Association engine with spreading activation across stores
- Evolution engine for self-tuning decay parameters
- Reconsolidation on recall (memories subtly modified by current context)

#### Transports
- HTTP/REST transport via axum
- gRPC transport via tonic with protobuf definitions

#### LLM Adapters
- Anthropic Claude adapter
- OpenAI GPT adapter
- Google Gemini adapter

#### SDKs
- Python HTTP SDK (`cerememory` on PyPI) with sync and async clients
- TypeScript HTTP SDK (`@cerememory/sdk` on npm)
- Python native binding via PyO3 (`cerememory-native`)
- TypeScript native binding via napi-rs (`@cerememory/native`)

#### Infrastructure
- CLI crate (`cerememory-cli`) providing the `cerememory` binary with serve, store, recall, and healthcheck commands
- Configuration management (`cerememory-config`) with TOML, env var, and CLI arg support
- Docker multi-stage build with distroless runtime
- CI/CD pipeline: formatting, clippy, tests, Docker build, SDK tests, native bindings
- Encrypted CMA export/import via ChaCha20-Poly1305

#### Benchmarks
- Store operations (write, read, query) benchmarks
- Decay engine tick benchmarks
- Association spreading activation benchmarks
- Index search benchmarks

[0.1.0]: https://github.com/co-r-e/cerememory/releases/tag/v0.1.0
