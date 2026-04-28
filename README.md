# CEREMEMORY

<p align="center">
  <strong>A Living Memory Database for the Age of AI</strong>
</p>

<p align="center">
  Open Source | LLM-Agnostic | Brain-Inspired | User-Sovereign
</p>

<p align="center">
  <a href="https://github.com/co-r-e/cerememory/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/co-r-e/cerememory">
    <img src="https://img.shields.io/badge/status-alpha-green.svg" alt="Status">
  </a>
  <a href="https://github.com/co-r-e/cerememory">
    <img src="https://img.shields.io/badge/rust-1.95+-orange.svg" alt="Rust">
  </a>
</p>

---

## What is Cerememory?

Every Large Language Model today suffers from the same fundamental limitation: **amnesia**.

Conversations reset. Context windows flush. Users repeat themselves. The few memory solutions that exist are shallow, text-only, model-specific, and controlled by the LLM provider rather than the user.

Cerememory is an open-source memory architecture inspired by the human brain's memory systems. It is not a single database. It is a constellation of specialized, interconnected stores that mirror the distinct subsystems neuroscience has identified in human memory.

### What makes Cerememory different

**It is alive.** Stored memories decay over time, accumulate noise, and can be reactivated when related memories fire. Data is not frozen at write-time; it breathes.

**It is LLM-agnostic.** A standardized protocol (the Cerememory Protocol, or CMP) allows any LLM to read from and write to the memory layer. Switch from one model to another; your memory persists.

**It belongs to the user.** Memory data is local-first, fully exportable, and designed for user-controlled deployment. No vendor lock-in. No corporate surveillance of your cognitive history.

---

## Quick Start

Cerememory's standard operating mode uses no external LLM API keys. It can run as
a local memory server for Claude Code and other MCP clients with
`[llm].provider = "none"`, which is the default.

Cerememory is distributed as source. Build the `cerememory` binary locally:

```bash
cargo build -p cerememory-cli --release
```

External LLM adapters are not part of the default binary. Experimental provider
support must be built explicitly, for example with `--features llm-openai`.

Run exactly one long-lived server for the data directory:

```bash
target/release/cerememory serve --data-dir ~/.cerememory/data
```

Then point every MCP client at that shared server. This is the only supported MCP operating mode:

```toml
[mcp_servers.cerememory]
command = "/absolute/path/to/target/release/cerememory"
args = ["mcp", "--server-url", "http://127.0.0.1:8420"]
```

This keeps the embedded `redb` and Tantivy stores owned by one process while allowing multiple terminal sessions and MCP clients to operate concurrently.

If the shared HTTP server requires auth, set `CEREMEMORY_SERVER_API_KEY` in the MCP client environment instead of passing secrets on the command line. For long-running upstream operations, `--server-timeout-secs` can be used to set an explicit request timeout; if omitted, per-request timeout is disabled.

Optional observed-event recording lives in a separate companion binary:

```bash
cargo build -p cerememory-recorder --release
printf '%s\n' '{"session_id":"demo","event_type":"user_message","content":"hello recorder"}' \
  | target/release/cerememory-recorder ingest --server-url http://127.0.0.1:8420
```

The recorder accepts Capture Event JSONL, redacts obvious secrets, and writes
safe raw batches to `/v1/encode/raw/batch`. It defaults automatic records to
`visibility=normal` and `secrecy_level=sensitive`; curated memory promotion still
belongs to `dream_tick`.
If `CEREMEMORY_SERVER_API_KEY` is set, recorder refuses non-loopback plaintext
HTTP upstreams; use HTTPS for remote servers.

Once connected, the client can use the core memory tools plus raw/dream workflows:

| Tool | Description |
|------|-------------|
| `store` | Save a new memory |
| `store_raw` | Preserve a verbatim raw journal entry |
| `update` | Edit an existing memory by UUID |
| `batch_store` | Save multiple memories at once |
| `batch_store_raw` | Preserve multiple raw journal entries at once |
| `recall` | Search memories by query (or list recent if omitted) |
| `recall_raw` | Explicit forensic recall over preserved raw journal entries |
| `timeline` | Browse memories by time period |
| `associate` | Find connected memories via spreading activation |
| `inspect` | View full details of a memory by UUID |
| `forget` | Permanently delete memories by UUID |
| `dream_tick` | Summarize raw journal entries into episodic/semantic memory |
| `consolidate` | Migrate mature episodic memories to semantic store |
| `export` | Export curated memories to a CMA archive file |
| `stats` | View system statistics and store counts |

Every stored record also carries typed **MetaMemory**: intent, rationale,
evidence, assumptions, alternatives, decisions, confidence, and context-graph
edges that explain why the memory exists and how it relates to other memories.
When no explicit meta-memory is supplied, Cerememory records that absence as a
structured `unavailable` capture state instead of pretending to know the reason.

MCP callers are encouraged to pass `meta_json` when they already know why a
memory is worth storing. This keeps Cerememory API-free by default: the calling
agent supplies intent, tags, rationale, evidence, or relation hints, and
Cerememory persists and indexes that structured context without calling OpenAI,
Anthropic, Gemini, or any other external LLM provider.

---

## The Architecture

Cerememory's design draws directly from neuroscience. The human brain does not store memories in a single location; it distributes them across specialized subsystems. Cerememory mirrors this architecture:

```
                        ┌─────────────────────────────────┐
                        │  Agent / Application Layer      │
                        │  (Claude Code, MCP clients,     │
                        │   SDKs, local apps)             │
                        └──────────────┬──────────────────┘
                                       │ CMP (Cerememory Protocol)
                        ┌──────────────┴──────────────────┐
                        │     Hippocampal Coordinator      │
                        │     (Cross-Store Index Layer)    │
                        └──┬───┬───┬───┬───┬──────────────┘
                           │   │   │   │   │
              ┌────────────┘   │   │   │   └────────────┐
              │                │   │   │                │
     ┌────────┴──────┐ ┌──────┴───┴───┴──────┐ ┌───────┴───────┐
     │   Episodic    │ │  Semantic  │ Proced. │ │   Working     │
     │   Store       │ │  Store     │ Store   │ │   Memory      │
     │  (Events)     │ │  (Graph)   │(Patterns│ │   (Cache)     │
     └───────────────┘ └───────────┴─────────┘ └───────────────┘
              │                │         │              │
              └────────┬───────┴─────────┴──────────────┘
                       │
              ┌────────┴────────┐    ┌──────────────────┐
              │  Emotional      │    │  Association      │
              │  Metadata Layer │    │  Engine           │
              │  (Cross-cutting)│    │  (Spreading       │
              └─────────────────┘    │   Activation)     │
                                     └──────────────────┘
                       │
              ┌────────┴────────┐    ┌──────────────────┐
              │  Decay Engine   │    │  Evolution Engine │
              │  (Forgetting +  │    │  (Self-Tuning)    │
              │   Noise)        │    │                   │
              └─────────────────┘    └──────────────────┘
```

External LLM adapters exist only as optional, experimental extensions for
auto-embedding, summarization, relation extraction, and transcription. They are
not required for the default MCP memory-server path.

### Five Memory Stores

| Store | Brain Analog | Function |
|-------|-------------|----------|
| **Episodic** | Hippocampus | Temporal event logs with spatial context. "What happened, when, and where." |
| **Semantic** | Distributed Neocortex | Graph of facts, concepts, and relationships. "What things mean." |
| **Procedural** | Basal Ganglia | Behavioral patterns, preferences, and skills. "How things are done." |
| **Emotional** | Amygdala | Cross-cutting affective metadata that modulates all other stores. |
| **Working** | Prefrontal Cortex | Volatile, limited-capacity, high-speed active context cache. |

### Meta-Memory Plane

Meta-memory is a cross-cutting plane attached to both curated `MemoryRecord`s
and raw journal records. It does not behave like a sixth recall store. Instead,
it preserves the "why" around every memory:

- **Intent**: why this memory was captured or generated
- **Rationale**: why the agent or user thought this was true or useful
- **Evidence**: source records, raw journal entries, excerpts, or external references
- **Decision context**: assumptions, alternatives, chosen decision, and confidence
- **Context graph**: typed `MetaEdge`s such as `derived_from`, `motivated_by`, `supports`, and `chose_over`

`recall.query` indexes provided and inferred meta-memory text for "why" queries,
and `recall.graph` can include the full plane with `include_meta: true`.

### Living Memory Dynamics

Unlike traditional databases where stored data is immutable:

- **Decay**: Memory fidelity decreases over time following a modified power-law curve
- **Noise**: Interference from similar memories accumulates, blurring details
- **Reactivation**: Related memories firing can temporarily restore decayed memories
- **Reconsolidation**: Recalling a memory subtly modifies it with current context
- **Consolidation**: Periodic migration from episodic to semantic store (like sleep)

---

## The Cerememory Protocol (CMP)

CMP is the standard interface between any LLM (or application) and a Cerememory instance. It is transport-agnostic and defines four operation categories:

| Category | Purpose | Key Operations |
|----------|---------|----------------|
| **Encode** | Write memories | `encode.store`, `encode.batch`, `encode.update` |
| **Recall** | Retrieve memories | `recall.query`, `recall.associate`, `recall.timeline` |
| **Lifecycle** | Manage dynamics | `lifecycle.consolidate`, `lifecycle.decay_tick`, `lifecycle.forget`, `lifecycle.export` |
| **Introspect** | Observe state | `introspect.stats`, `introspect.record`, `introspect.decay_forecast` |

Recall supports two modes:
- **Human mode**: Returns memories with fidelity-weighted noise applied (realistic)
- **Perfect mode**: Returns original data with no degradation

Three transport bindings are available:

| Transport | Protocol | Use Case |
|-----------|----------|----------|
| **MCP** (stdio) | Model Context Protocol | Claude Code, MCP-compatible LLM tools |
| **HTTP/REST** | JSON over HTTP | SDKs, web apps, microservices |
| **gRPC** | Protobuf over HTTP/2 | High-throughput, low-latency integrations |

See the [CMP Specification](docs/cmp-spec-v1.pdf) for the complete protocol definition.

---

## Configuration

Copy the example and adjust:

```bash
cp cerememory.example.toml cerememory.toml
```

Key settings:

```toml
data_dir = "./data"

[http]
port = 8420
bind_address = "127.0.0.1"  # localhost only. "0.0.0.0" for network access.
cors_origins = []            # Empty = emit no CORS headers
trusted_proxy_cidrs = []     # Optional trusted reverse proxies
metrics_enabled = false      # Expose /metrics when true

[grpc]
# port = 8421
# tls_cert_path = "/path/to/cert.pem"
# tls_key_path = "/path/to/key.pem"

[auth]
enabled = false
api_keys = []  # ["sk-key1", "sk-key2"]

[security]
# Prefer CEREMEMORY_SECURITY__STORE_ENCRYPTION_PASSPHRASE instead of TOML.
# Then run: cerememory migrate-store-encryption --confirm
# store_encryption_passphrase = "change-me"
# When a passphrase is set, persistent full-text indexes default to disabled.
# Set true only if searchable derived text on disk is acceptable.
# persist_search_indexes = false
# Tamper-evident audit logging is enabled by default at data_dir/audit.jsonl.
# Run: cerememory audit-verify
# audit_log_enabled = true
# audit_log_path = "/path/to/audit.jsonl"

[llm]
provider = "none"  # standard default; no external LLM API key is used
# Optional experimental providers require an explicit CLI build feature:
#   cargo build -p cerememory-cli --release --features llm-openai
# Provider values: "openai", "claude", "gemini"
# api_key = "sk-..."  # only needed when provider is not "none"

[decay]
background_interval_secs = 3600  # once per hour

[dream]
background_interval_secs = 86400  # once per day

[rate_limit]
requests_per_second = 100
burst = 50

[log]
level = "info"    # trace, debug, info, warn, error
format = "pretty" # pretty, json
```

All settings can be overridden via environment variables with the `CEREMEMORY_` prefix (e.g., `CEREMEMORY_HTTP__PORT=9000`).

---

## Security

Cerememory is designed with security as a default:

- **Localhost-only by default**: HTTP and gRPC bind to `127.0.0.1`. Network access requires explicit `bind_address = "0.0.0.0"` configuration, which triggers a warning if auth is disabled.
- **Bearer token authentication**: Optional API key auth with constant-time comparison. Keys are wrapped in `SecretString` at runtime to prevent accidental logging.
- **Optional store encryption**: When `security.store_encryption_passphrase` is set, newly written persistent redb payloads are encrypted with ChaCha20-Poly1305 using an Argon2id-derived key. This covers raw journal, episodic, semantic, procedural, emotional, and vector payloads. Run `cerememory migrate-store-encryption --confirm` to rewrite existing plaintext payloads with the configured key. Persistent full-text indexes are disabled by default when store encryption is enabled; set `security.persist_search_indexes = true` only if searchable derived text on disk is acceptable.
- **Tamper-evident audit log**: Server-backed engines write a plaintext JSONL hash chain at `data_dir/audit.jsonl` by default. `cerememory audit-verify` checks sequence numbers, previous hashes, and entry hashes; persist the reported head hash externally if truncation detection is required.
- **Encrypted exports**: CMA archives support ChaCha20-Poly1305 AEAD encryption with Argon2id key derivation. Derived keys are zeroized after use.
- **Request size limits**: HTTP API routes accept request bodies up to 64 MB to support archive import/export workflows. Batch operations are capped at 1000 records. Image/audio recall cues are additionally validated against modality size limits.
- **Sanitized error responses**: Internal storage paths and details are never exposed to clients. 503 responses include `Retry-After` headers.

For HTTP TLS, use a reverse proxy (nginx, caddy) in front of the HTTP server. gRPC supports native TLS via cert/key configuration.

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Core Engine | Rust | Memory safety, zero-cost concurrency, embeddable native binary |
| Async Runtime | Tokio | I/O-bound CMP request handling |
| Compute Pool | Rayon | CPU-bound decay engine and spreading activation |
| Episodic Store | redb | ACID transactions, zero-copy reads |
| Vector Search | redb + exact cosine scan | Deterministic embedded vector search with no secondary ANN graph |
| Full-Text Search | Tantivy | Rust-native Lucene equivalent |
| Synchronization | parking_lot | Poison-free RwLock/Mutex (no panics on lock contention) |
| Internal Serialization | MessagePack | Compact binary, schema-less |
| Archive Format | JSON Lines CMA bundle | Portable, inspectable, single-file archive with checksum support |
| Archive Encryption | ChaCha20-Poly1305 + Argon2id | AEAD with memory-hard KDF, key zeroization via `zeroize` |

See the [Architecture Decision Records](docs/adr/) for detailed rationale behind each decision.

---

## Project Structure

```
cerememory/
  crates/
    cerememory-core/              # Core types, traits, CMP protocol types
    cerememory-store-episodic/    # Episodic store implementation
    cerememory-store-semantic/    # Semantic (graph) store implementation
    cerememory-store-procedural/  # Procedural store implementation
    cerememory-store-emotional/   # Emotional metadata layer
    cerememory-store-working/     # Working memory cache
    cerememory-decay/             # Decay engine
    cerememory-association/       # Spreading activation engine
    cerememory-evolution/         # Evolution engine
    cerememory-index/             # Hippocampal coordinator
    cerememory-engine/            # Orchestrator
    cerememory-transport-http/    # HTTP/REST binding
    cerememory-transport-grpc/    # gRPC binding
    cerememory-transport-mcp/     # MCP (Model Context Protocol) binding
    cerememory-archive/           # CMA export/import
    cerememory-cli/               # CLI tool + `cerememory` binary
    cerememory-config/            # Configuration management
  adapters/
    adapter-claude/               # Optional experimental Anthropic Claude adapter
    adapter-openai/               # Optional experimental OpenAI adapter
    adapter-gemini/               # Optional experimental Google Gemini adapter
  docs/                           # Whitepaper, CMP spec, ADRs
  tests/integration/              # Cross-crate integration tests
  benches/                        # Performance benchmarks
```

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1: Foundation** | Done | Core stores, decay engine, CMP protocol, optional adapter boundary |
| **Phase 2: Dynamics** | Done | Cross-modal associations, emotional metadata, reconsolidation |
| **Phase 3: Hardening** | Done | Error handling, observability, performance optimization |
| **Phase 4: Intelligence** | Done | Evolution engine, self-tuning parameters |
| **Phase 5: Production** | Done | CLI, config management, CI/CD |
| **Phase 7: Benchmarks** | Done | Performance benchmarks (store, decay, association, index) |
| **Phase 8: Multimodal** | Done | Image, audio, and structured data memory |
| **Phase 10: Integrations** | Done | Optional adapter integration tests |
| **Phase 11: Encryption** | Done | Encrypted CMA export/import |
| **Phase 12: Best Practices** | Done | MCP UX overhaul, security hardening, reliability improvements |

---

## Documentation

- [Whitepaper](docs/cerememory-whitepaper.pdf): Philosophy, neuroscience foundations, and system design
- [CMP Specification v1.0](docs/cmp-spec-v1.pdf): Complete protocol definition
- [MCP Agent Metadata](docs/mcp-agent-metadata.md): How MCP clients should pass MetaMemory without external LLM API keys
- [Architecture Decision Records](docs/adr/): Technology stack decisions with full rationale
- [Human-Plus Memory Phase 1](docs/human-plus-memory-phase1.md): Design for raw journal, dream tick, suppression, and forensic recall
- [Recorder Companion](docs/recorder.md): Capture Event JSONL, Codex hook helper, diagnostics, and spool behavior

The landing page and documentation site live in a separate repository at `~/Projects/dev_cerememory/cerememory-docs`.

---

## Contributing

We welcome contributions from developers, neuroscientists, AI researchers, and anyone who believes memory should be an open standard.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we especially need help:
- Rust core engine development
- Local/API-free retrieval, metadata, and optional adapter implementations
- Neuroscience review of decay and association models
- Documentation and translations

---

## Why Open Source?

Memory is the foundation of identity. A system that stores, evolves, and retrieves the accumulated context of a person's interaction with AI is too important to be controlled by any single entity.

If the age of AI means that humans will increasingly co-think with AI systems, then the memory layer that bridges human and machine cognition must be a public good.

**Your memories should be yours.**

---

## License

Cerememory is licensed under the [MIT License](LICENSE).

---

## Author

Created by [Masato Okuwaki](https://x.com/okuwaki_m) at [CORe Inc.](https://co-r-e.com)

CORe Inc. initiates and stewards this project, but Cerememory belongs to its community.
