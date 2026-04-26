# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Security and Storage
- Added optional live-store encryption for persistent redb payloads covering raw journal, episodic, semantic, procedural, emotional, and vector data
- Added `cerememory migrate-store-encryption --confirm` to rewrite existing plaintext store payloads with the configured store encryption passphrase
- Added a tamper-evident plaintext JSONL audit log with startup verification and `cerememory audit-verify`
- Added `security.store_encryption_passphrase`, `security.persist_search_indexes`, `security.audit_log_enabled`, and `security.audit_log_path` configuration
- Added ADRs for secure-at-rest scope and tamper-evident audit logging

### Changed

#### Security and Indexes
- Persistent full-text search indexes now default to in-memory rebuilds when store encryption is enabled, unless `security.persist_search_indexes = true` is set
- Security documentation now distinguishes live-store encryption, encrypted CMA archives, plaintext derived indexes, and audit-log integrity guarantees
- Vector search now uses a deterministic redb-backed exact cosine scan and exposes `vector_search_backend` / `vector_index_records` in stats

#### Dependencies
- Updated Rust dependencies to their latest compatible releases, including major updates for `redb`, `tantivy`, `criterion`, `ordered-float`, `rand`, and `sha2`
- Updated GitHub Actions usage to Node 24-capable `actions/checkout@v6` and `actions/cache@v5`
- Removed the HNSW vector backend and its `hnsw_rs` / `bincode` advisory surface before release
- Removed stale security advisory ignores that are no longer present after the dependency refresh
- CI now runs Clippy across all targets and checks a dedicated CLI feature matrix
- Removed the unmaintained `backoff` dependency from LLM adapters

### Fixed

#### Reliability and Security
- Hardened API key validation so configured keys are scanned without early-exit match behavior
- CLI builds without LLM adapter features now report configured LLM providers as unsupported instead of silently disabling them
- Removed stale SQLite/archive wording from docs and crate metadata

## [0.2.5] - 2026-04-26

### Added

#### Meta-Memory and Context Graph
- Added structured meta-memory metadata to memory records and raw journals, including rationale, intent, triggers, alternatives, evidence, tags, and typed context edges
- Added meta-memory normalization for legacy, partial, inferred, provided, and unavailable capture states across JSON, MessagePack, and CMP payloads
- Added meta-memory text indexing so recall can search why a memory exists, not only what happened
- Exposed typed meta-memory payloads through engine store/update flows, CLI input, MCP tools, gRPC conversion, export/import, rebuild, graph, and timeline paths

### Fixed

#### Security and CI
- Updated `rustls-webpki` to `0.103.13` to address `RUSTSEC-2026-0104`
- Derived default enum implementations for meta-memory status and relation types to satisfy workspace Clippy checks

#### Documentation
- Corrected the CMA archive format description in the changelog

## [0.2.4] - 2026-04-22

### Removed

#### Distribution and MCP Operations
- Removed direct embedded-store MCP mode; `cerememory mcp` now requires `--server-url` and operates only as a proxy to a long-lived `cerememory serve` process
- Removed package and artifact publishing workflows for GitHub Releases, Docker images, crates.io, PyPI, and npm
- Removed Dockerfile and Docker-based CI/E2E distribution checks
- Simplified README installation guidance to source build plus shared server MCP operation
- Removed Python, TypeScript, and native binding directories from the active codebase

## [0.2.3] - 2026-04-22

### Added

#### MCP and Operations
- Added `cerememory mcp --server-url` so multiple MCP clients can share one long-lived Cerememory HTTP server without competing for embedded `redb` and Tantivy locks
- Added upstream MCP proxy support for store, recall, raw journal, timeline, associate, forget, consolidate, dream tick, stats, inspect, and export tools
- Added `CEREMEMORY_SERVER_API_KEY` and `--server-timeout-secs` support for remote MCP proxy operation
- Added MCP protocol E2E coverage for initialize, tools/list, store, and recall tool calls

### Fixed

#### MCP and Reliability
- Fixed MCP tool exposure by wiring the generated tool router into the `ServerHandler`
- Improved startup guidance when a local embedded store is already locked by another Cerememory process
- Validated remote MCP server URLs before startup and rejected zero-second upstream timeouts
- Made upstream non-JSON error previews UTF-8 safe

## [0.2.2] - 2026-04-12

### Changed

#### SDK
- TypeScript SDK import example now uses single-import style (`CerememoryClient` and `CerememoryProvider` from `@cerememory/sdk`)
- Removed redundant `integrations/index.ts` barrel file; `CerememoryProvider` is re-exported from the package root

#### Documentation
- Added reference to the separate documentation site repository in README

## [0.2.1] - 2026-03-22

### Added

#### Human-Plus Memory
- Raw journal preservation plane with dedicated redb store, session index, and raw full-text search
- Raw ingest and forensic recall APIs across HTTP, MCP, and gRPC
- `lifecycle.dream_tick` for raw-to-episodic summarization with backlinks
- Topic inference for dream grouping based on time gaps and lexical shift
- Secrecy-aware dream summarization with redaction accounting
- Conditional semantic promotion from dream summaries
- Background dream processing with configurable interval and graceful shutdown
- Bundle CMA archive support for exporting/importing curated memory together with raw journal records

#### SDK and Native Bindings
- Python HTTP SDK support for `store_raw`, `recall_raw`, `dream_tick`, `lifecycle_export`, and `lifecycle_import`
- TypeScript HTTP SDK support for `storeRaw`, `recallRaw`, `dreamTick`, `lifecycleExport`, and `lifecycleImport`
- Python native binding support for raw journal and dream tick operations
- TypeScript native binding support for raw journal and dream tick operations

### Changed

#### Introspection and Operations
- `StatsResponse` now exposes raw journal counts, pending dream work, dream-derived record counts, last dream tick time, and background dream status
- CLI now exposes `store-raw`, `recall-raw`, and `dream-tick`, plus raw-inclusive archive export

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
- Hippocampal coordinator for cross-store indexing and vector search
- Full-text search via Tantivy integration
- CMA archive format (JSON Lines-based) for export/import

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
[0.2.5]: https://github.com/co-r-e/cerememory/releases/tag/v0.2.5
[0.2.4]: https://github.com/co-r-e/cerememory/releases/tag/v0.2.4
[0.2.3]: https://github.com/co-r-e/cerememory/releases/tag/v0.2.3
[0.2.2]: https://github.com/co-r-e/cerememory/releases/tag/v0.2.2
