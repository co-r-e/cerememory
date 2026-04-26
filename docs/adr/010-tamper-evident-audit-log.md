# ADR-010: Tamper-Evident Audit Log

## Status

Accepted

## Context

Cerememory already emits tracing logs, but tracing is operational telemetry, not
an integrity-checkable audit trail. Security-sensitive workflows need a way to
detect edits to recorded operation metadata after the fact.

This audit feature must not leak memory contents or credentials. It also must
fit the local-first storage model and work without a remote service.

## Decision

Server-backed engines write a plaintext JSON Lines audit log at
`data_dir/audit.jsonl` by default. Each entry includes operation metadata,
target record IDs, store types, a small content-free summary, the previous
entry hash, and the current entry hash.

The current entry hash is SHA-256 over a deterministic payload containing the
entry metadata and previous hash. On startup, an engine with an audit log path
verifies the entire hash chain before opening for normal use. The CLI also
provides:

```sh
cerememory audit-verify
```

`security.audit_log_enabled = false` disables the audit log. Operators can set
`security.audit_log_path` when the default path is not appropriate.

## Scope

The audit log records successful mutating or security-sensitive operations:

- `encode.store`, `encode.batch`, `encode.update`
- `encode.store_raw`, `encode.batch_raw`, direct raw journal append
- `recall.query` when `reconsolidate = true`
- `lifecycle.dream_tick`, `lifecycle.consolidate`, `lifecycle.decay_tick`
- `lifecycle.set_mode`, `lifecycle.forget`
- `lifecycle.export`, `lifecycle.import`
- `import_records` convenience imports
- `security.migrate_store_encryption`

Audit summaries intentionally exclude memory body text, raw archive bytes,
encryption keys, API keys, and passphrases.

## Consequences

- Edited, reordered, or internally truncated audit entries are detected by
  `cerememory audit-verify` and by engine startup verification.
- A cleanly truncated suffix cannot be distinguished from an older valid log
  unless the latest reported head hash is stored outside the data directory.
- The audit log is not encrypted. Use filesystem encryption or a protected path
  if operation metadata itself is sensitive.
- Audit write failures fail the initiating operation after the primary mutation
  has been attempted, so operators see the integrity failure instead of silently
  continuing without audit evidence.
