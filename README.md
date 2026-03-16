# CEREMEMORY

<p align="center">
  <strong>A Living Memory Database for the Age of AI</strong>
</p>

<p align="center">
  Open Source | LLM-Agnostic | Brain-Inspired | User-Sovereign
</p>

<p align="center">
  <a href="https://github.com/co-r-e/cerememory/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/co-r-e/cerememory">
    <img src="https://img.shields.io/badge/status-alpha-green.svg" alt="Status">
  </a>
  <a href="https://github.com/co-r-e/cerememory">
    <img src="https://img.shields.io/badge/rust-1.77+-orange.svg" alt="Rust">
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

## The Architecture

Cerememory's design draws directly from neuroscience. The human brain does not store memories in a single location; it distributes them across specialized subsystems. Cerememory mirrors this architecture:

```
                        ┌─────────────────────────────────┐
                        │     LLM Adapter Layer           │
                        │  (Claude, GPT, Gemini, ...)     │
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

### Five Memory Stores

| Store | Brain Analog | Function |
|-------|-------------|----------|
| **Episodic** | Hippocampus | Temporal event logs with spatial context. "What happened, when, and where." |
| **Semantic** | Distributed Neocortex | Graph of facts, concepts, and relationships. "What things mean." |
| **Procedural** | Basal Ganglia | Behavioral patterns, preferences, and skills. "How things are done." |
| **Emotional** | Amygdala | Cross-cutting affective metadata that modulates all other stores. |
| **Working** | Prefrontal Cortex | Volatile, limited-capacity, high-speed active context cache. |

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

See the [CMP Specification](docs/cmp-spec-v1.pdf) for the complete protocol definition.

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Core Engine | Rust | Memory safety, zero-cost concurrency, embeddable native binary |
| Async Runtime | Tokio | I/O-bound CMP request handling |
| Compute Pool | Rayon | CPU-bound decay engine and spreading activation |
| Episodic Store | redb | ACID transactions, zero-copy reads |
| Vector Search | hnsw_rs | Lightweight embedded HNSW index |
| Full-Text Search | Tantivy | Rust-native Lucene equivalent |
| Internal Serialization | MessagePack | Compact binary, schema-less |
| Archive Format | SQLite (CMA) | Universal, inspectable, single-file |
| Python SDK | httpx + Pydantic | Typed HTTP client for Python applications |
| TypeScript SDK | TypeScript + fetch | Zero-dependency HTTP client for Node.js and browser runtimes |
| Python Native Binding | PyO3 | Direct native integration without HTTP |
| TypeScript Native Binding | napi-rs | Direct Node.js native integration without HTTP |

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
    cerememory-archive/           # CMA export/import
    cerememory-cli/               # CLI tool + `cerememory` binary
    cerememory-config/            # Configuration management
  bindings/
    python/                       # HTTP SDK (pure Python)
    python-native/                # Native SDK (PyO3)
    typescript/                   # HTTP SDK (pure TypeScript)
    typescript-native/            # Native SDK (napi-rs)
  adapters/
    adapter-claude/               # Anthropic Claude adapter
    adapter-openai/               # OpenAI GPT adapter
    adapter-gemini/               # Google Gemini adapter
  docs/                           # Whitepaper, CMP spec, ADRs
  tests/integration/              # Cross-crate integration tests
  benches/                        # Performance benchmarks
```

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1: Foundation** | Done | Core stores, decay engine, CMP protocol, LLM adapters |
| **Phase 2: Dynamics** | Done | Cross-modal associations, emotional metadata, reconsolidation |
| **Phase 3: Hardening** | Done | Error handling, observability, performance optimization |
| **Phase 4: Intelligence** | Done | Evolution engine, self-tuning parameters |
| **Phase 5: Production** | Done | CLI, config management, Docker, CI/CD |
| **Phase 6: SDK** | Done | Python & TypeScript HTTP SDKs |
| **Phase 7: Benchmarks** | Done | Performance benchmarks (store, decay, association, index) |
| **Phase 8: Multimodal** | Done | Image, audio, and structured data memory |
| **Phase 9: Native Bindings** | Done | PyO3 (Python) and napi-rs (TypeScript) native bindings |
| **Phase 10: Integrations** | Done | LLM adapter integration tests |
| **Phase 11: Encryption** | Done | Encrypted CMA export/import |

---

## Quick Start

The Python and TypeScript SDKs talk to the Cerememory HTTP server. Start a local server first:

```bash
cargo run -p cerememory-cli -- serve --port 8420
```

Or run the published container:

```bash
docker run --rm -p 8420:8420 ghcr.io/co-r-e/cerememory:latest
```

### Python

```bash
pip install cerememory
```

```python
from cerememory import Client

client = Client("http://localhost:8420")
record_id = client.store("Had coffee with Alice at the park", store="episodic")
results = client.recall("Alice", limit=5)
client.forget(record_id, confirm=True)
```

### TypeScript

```bash
npm install @cerememory/sdk
```

```typescript
import { CerememoryClient } from "@cerememory/sdk";

const client = new CerememoryClient("http://localhost:8420");
const recordId = await client.store("Had coffee with Alice at the park", {
  store: "episodic",
});
const results = await client.recall("Alice", { limit: 5 });
await client.forget(recordId, { confirm: true });
```

---

## Documentation

- [Whitepaper](docs/cerememory-whitepaper.pdf): Philosophy, neuroscience foundations, and system design
- [CMP Specification v1.0](docs/cmp-spec-v1.pdf): Complete protocol definition
- [Architecture Decision Records](docs/adr/): Technology stack decisions with full rationale

---

## Contributing

We welcome contributions from developers, neuroscientists, AI researchers, and anyone who believes memory should be an open standard.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we especially need help:
- Rust core engine development
- LLM adapter implementations
- Python and TypeScript SDK development
- Neuroscience review of decay and association models
- Documentation and translations

---

## Why Open Source?

Memory is the foundation of identity. A system that stores, evolves, and retrieves the accumulated context of a person's interaction with AI is too important to be controlled by any single entity.

If the age of AI means that humans will increasingly co-think with AI systems, then the memory layer that bridges human and machine cognition must be a public good.

**Your memories should be yours.**

---

## License

Cerememory is licensed under the [Apache License 2.0](LICENSE).

---

## Author

Created by [Masato Okuwaki](https://x.com/okuwaki_m) at [CORe Inc.](https://co-r-e.com)

CORe Inc. initiates and stewards this project, but Cerememory belongs to its community.
