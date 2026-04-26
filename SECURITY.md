# Security Policy

## Reporting a Vulnerability

Cerememory handles deeply personal memory data. We take security vulnerabilities seriously.

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email: security@co-r-e.com
3. Include: description, reproduction steps, potential impact
4. We will acknowledge within 48 hours and provide a timeline for a fix

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x     | Pre-alpha, best-effort security fixes |

## Security Principles

Cerememory's security architecture is guided by these principles:

- **Local-first storage**: Memory stores are local redb/Tantivy files owned by the user. Store-level encryption at rest is not enabled by default yet; use encrypted disks or filesystem-level encryption for local store protection.
- **Optional live-store encryption**: Setting `security.store_encryption_passphrase` encrypts newly written persistent redb payloads for raw journal, episodic, semantic, procedural, emotional, and vector data. Run `cerememory migrate-store-encryption --confirm` after enabling it to rewrite existing plaintext payloads with the configured key. Persistent full-text indexes are disabled by default when store encryption is enabled; set `security.persist_search_indexes = true` only if searchable derived text on disk is acceptable.
- **Tamper-evident audit logging**: Server-backed engines write a plaintext JSONL hash chain to `data_dir/audit.jsonl` by default. Run `cerememory audit-verify` to validate sequence numbers, previous hashes, and entry hashes. Store the reported head hash outside the data directory if you need truncation detection.
- **Encrypted archives**: CMA exports can be encrypted with user-supplied passphrases using ChaCha20-Poly1305 AEAD and Argon2id key derivation.
- **User-held export keys**: Archive passphrases belong to the user, not any service.
- **Local-first**: No data leaves the user's machine unless explicitly configured
- **Explicit deletion**: `lifecycle.forget` requires confirmation and removes records from the active stores and search indexes. This is not a substitute for cryptographic erasure of previously written disk blocks or external backups.
- **Operational logging**: server operations emit tracing logs for lifecycle events and failures; the audit log is separate and records operation metadata, not memory contents.

See the [CMP Specification](docs/cmp-spec-v1.pdf) Section 10 for detailed security requirements.
