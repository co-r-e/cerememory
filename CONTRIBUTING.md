# Contributing to Cerememory

Thank you for your interest in Cerememory. This project aims to build an open standard for AI memory, and community contributions are essential to that mission.

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bug reports, feature requests, and questions
- Search existing issues before creating a new one
- For bug reports, include: Rust version, OS, steps to reproduce, expected behavior, actual behavior
- For feature requests, explain the use case and how it relates to the architecture

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main` (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure all tests pass (`cargo test --workspace`)
5. Ensure code is formatted (`cargo fmt --all`)
6. Ensure clippy passes (`cargo clippy --workspace -- -D warnings`)
7. Write or update tests for your changes
8. Submit a pull request against `main`

### Commit Messages

Use conventional commits:

```
feat(store-episodic): add temporal range query support
fix(decay): correct fidelity score underflow at very low values
docs(cmp): clarify reconsolidation behavior in recall.query
test(association): add spreading activation depth-3 test
refactor(core): extract MemoryRecord validation into trait
```

Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`

## Areas of Contribution

### Rust Core Engine

The core engine is where the most technically demanding work happens. If you are comfortable with Rust, these areas need attention:

- Store implementations (episodic, semantic, procedural, emotional, working)
- Decay engine optimization
- Spreading activation algorithm
- Cross-store index coordination
- Transport bindings (HTTP via axum, gRPC via tonic)
- Archive format (CMA) import/export

### LLM Adapters

Adapters translate between Cerememory's internal representation and LLM-specific formats. You can write adapters in Rust directly or use the Python/TypeScript SDKs:

- Claude (Anthropic)
- GPT (OpenAI)
- Gemini (Google)
- Llama (Meta)
- Mistral
- Any other LLM you use

### Python and TypeScript SDKs

The HTTP SDKs make Cerememory accessible to the broader AI ecosystem from Python and TypeScript applications. Native bindings live separately under `bindings/python-native` (PyO3) and `bindings/typescript-native` (napi-rs).

### Neuroscience Review

If you have expertise in cognitive neuroscience or memory research, we especially value:

- Review of the decay mathematical model (ADR-005)
- Review of the spreading activation algorithm (ADR-004)
- Suggestions for more accurate modeling of reconsolidation dynamics
- Literature references for cross-modal association patterns
- Validation of the multi-store architecture against current neuroscience consensus

### Documentation and Translation

- Improve documentation clarity
- Add examples and tutorials
- Translate documentation (Japanese, Chinese, Korean, Spanish, etc.)

## Development Setup

### Prerequisites

- Rust 1.77 or later (install via [rustup](https://rustup.rs/))
- Cargo (included with Rust)
- Python 3.9 or later (for Python SDK work)
- Node.js 18 or later (for TypeScript SDK work)

### Build

```bash
git clone https://github.com/co-r-e/cerememory.git
cd cerememory
cargo build --workspace
```

### Test

```bash
cargo test --workspace
```

### Format and Lint

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

### Python SDK Development

```bash
cd bindings/python
python -m pip install -e '.[dev]'
python -m pytest -v
python -m ruff check .
python -m mypy src
```

### TypeScript SDK Development

```bash
cd bindings/typescript
npm ci
npm run typecheck
npm test
```

## Architecture Overview

Before contributing, please read:

1. [Whitepaper](docs/cerememory-whitepaper.pdf): Understand the philosophy and neuroscience foundations
2. [CMP Specification](docs/cmp-spec-v1.pdf): Understand the protocol
3. [ADRs](docs/adr/): Understand why specific technology choices were made

The crate structure follows the architecture closely:

- `cerememory-core`: Shared types, traits, and CMP protocol definitions
- `cerememory-store-*`: Individual store implementations
- `cerememory-decay`: The decay engine (background fidelity/noise processing)
- `cerememory-association`: Spreading activation engine
- `cerememory-index`: The hippocampal coordinator that binds cross-store references
- `cerememory-engine`: The orchestrator that assembles everything

## Code Style

- Follow standard Rust conventions and idioms
- Use `rustfmt` defaults (no custom configuration)
- Document all public APIs with doc comments
- Write unit tests for new functionality
- Prefer explicit error types over `unwrap()` / `expect()` in library code
- Use `thiserror` for error type definitions
- Use `tracing` for logging (not `println!` or `log`)

## Communication

- GitHub Issues: Bug reports, feature requests, technical discussions
- GitHub Discussions: General questions, ideas, show-and-tell
- Pull Requests: Code contributions

## License

By contributing to Cerememory, you agree that your contributions will be licensed under the Apache License 2.0.

## Code of Conduct

Be respectful, constructive, and welcoming. We are building infrastructure for human memory; let us treat each other with the care that mission deserves.

---

Thank you for helping build the memory layer that AI has been missing.
