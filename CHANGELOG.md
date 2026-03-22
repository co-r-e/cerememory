# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-03-22

### Fixed

#### Security and Operations
- CLI passphrase prompts for `--encryption-key-stdin` and `--decryption-key-stdin` now disable terminal echo in interactive sessions
- HTTP rate limiting now treats forwarded IP headers as untrusted unless the peer socket address is present and matches `http.trusted_proxy_cidrs`
- HTTP serving now propagates peer socket addresses into middleware so per-client rate limiting works correctly without trusting forwarded headers

### Changed

- Release metadata is now aligned at `0.2.1` across the Rust workspace, Python SDKs, and TypeScript SDKs

## [0.2.0] - 2026-03-22

### Added

#### Security and Transport
- `http.trusted_proxy_cidrs` configuration for trusted forwarded client IP extraction
- `http.metrics_enabled` configuration to opt into `/metrics`
- Protected `/metrics` endpoint with authentication and rate limiting when enabled
- gRPC startup validation requiring TLS when auth is enabled or the bind address is non-loopback

#### Engine and Protocol
- `EncodeStoreRequest.metadata` support across engine, CLI, MCP, and SDKs
- Persistent storage for inferred associations so they survive rebuild/export/import
- `ForgetRequest.temporal_range` support in the engine
- Query metadata surfaced in TypeScript and Python SDK response models
- Request correlation ID surfaced in SDK error models

#### CLI and SDK UX
- CLI `export` flags for `--format`, `--stores`, `--encrypt`, and `--encryption-key`
- CLI `import` flags for `--decryption-key` and `--conflict-resolution`
- CLI stdin secret input flags: `--encryption-key-stdin`, `--decryption-key-stdin`
- SDK options to control recall reconsolidation and activation depth
- SDK opt-in support for retrying mutating requests

#### Documentation
- Migration and operational guidance for the 0.2.0 security and SDK behavior changes

### Changed

#### Security Defaults
- Empty `http.cors_origins` now emits no CORS headers instead of allowing all origins
- `auth.enabled` is now authoritative; configured keys are ignored when auth is disabled
- Blank or whitespace-only auth API keys are rejected at configuration validation time
- `/metrics` is disabled by default

#### Engine Behavior
- `recall.query` now applies temporal filtering across all stores and re-applies filters after activation expansion
- Decay tick computation now avoids double-counting simulated duration
- Semantic edge persistence is aligned with `MemoryRecord.associations`

#### SDK Behavior
- TypeScript and Python SDKs now default to `0` retries
- Retries now apply only to safe requests unless mutating retries are explicitly enabled

### Fixed

- CLI stateful commands now rebuild the coordinator before operating on persisted data
- Association inference and consolidation-generated associations now survive rebuilds
- Import replacement logic no longer deletes existing cross-store records before successful replacement
- `/metrics` no longer bypasses the configured auth policy when enabled

### Migration Notes

- Review browser clients that depended on implicit wildcard CORS. You now need explicit `http.cors_origins`.
- If you scrape `/metrics`, set `http.metrics_enabled = true` and provide auth when `auth.enabled = true`.
- If you expose gRPC outside loopback or enable auth, configure `grpc.tls_cert_path` and `grpc.tls_key_path`.
- If you relied on SDK retries for writes, set `retryMutatingRequests` / `retry_mutating_requests` explicitly.
- See [docs/migration-0.2.0.md](/Users/okuwakimasato/Projects/dev_cerememory/cerememory/docs/migration-0.2.0.md).

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
[0.2.0]: https://github.com/co-r-e/cerememory/releases/tag/v0.2.0
[0.2.1]: https://github.com/co-r-e/cerememory/releases/tag/v0.2.1
