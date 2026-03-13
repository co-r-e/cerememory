<p align="center">
  <h1 align="center">CEREMEMORY</h1>
  <p align="center">
    <strong>A Living Memory Database for the Age of AI</strong>
  </p>
  <p align="center">
    Open-Source | LLM-Agnostic | Brain-Inspired | User-Sovereign
  </p>
  <p align="center">
    <a href="https://github.com/co-r-e/cerememory/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
    </a>
    <a href="https://github.com/co-r-e/cerememory">
      <img src="https://img.shields.io/badge/status-pre--alpha-orange.svg" alt="Status">
    </a>
    <a href="https://github.com/co-r-e/cerememory">
      <img src="https://img.shields.io/badge/rust-1.77+-orange.svg" alt="Rust">
    </a>
  </p>
</p>

---

## What is Cerememory?

Every Large Language Model today suffers from the same fundamental limitation: **amnesia**.

Conversations reset. Context windows flush. Users repeat themselves. The few memory solutions that exist are shallow, text-only, model-specific, and controlled by the LLM provider rather than the user.

Cerememory is an open-source memory architecture inspired by the human brain's memory systems. It is not a single database. It is a constellation of specialized, interconnected stores that mirror the distinct subsystems neuroscience has identified in human memory.

### What makes Cerememory different

**It is alive.** Stored memories decay over time, accumulate noise, and can be reactivated when related memories fire. Data is not frozen at write-time; it breathes.

**It is LLM-agnostic.** A standardized protocol (the Cerememory Protocol, or CMP) allows any LLM to read from and write to the memory layer. Switch from one model to another; your memory persists.

**It belongs to the user.** Memory data is local-first, encrypted by default, and fully exportable. No vendor lock-in. No corporate surveillance of your cognitive history.

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
| Vector Search | usearch | Lightweight embedded HNSW index |
| Full-Text Search | Tantivy | Rust-native Lucene equivalent |
| Internal Serialization | MessagePack | Compact binary, schema-less |
| Archive Format | SQLite (CMA) | Universal, inspectable, single-file |
| Python SDK | PyO3 | AI/ML ecosystem integration |
| TypeScript SDK | napi-rs | Web/LLM toolchain integration |

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
    cerememory-cli/               # CLI tool
  bindings/
    python/                       # PyO3 SDK
    typescript/                   # napi-rs SDK
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

| Phase | Timeline | Focus |
|-------|----------|-------|
| **Phase 1: Foundation** | Q2 2026 | Core episodic + semantic stores, decay engine, basic LLM adapters, text-only |
| **Phase 2: Dynamics** | Q3 2026 | Cross-modal associations, emotional metadata, reconsolidation, consolidation cycles |
| **Phase 3: Multimodal** | Q4 2026 | Image, audio, and structured data memory stores |
| **Phase 4: Evolution** | Q1 2027 | Self-tuning parameters, dynamic schema generation |
| **Phase 5: Ecosystem** | Q2 2027+ | Multi-agent shared memory, plugins, mobile support |

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
