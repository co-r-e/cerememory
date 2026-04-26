# ADR-009: Secure-at-Rest Scope

## Status

Accepted

## Context

Cerememory stores personal memory data in local redb files and maintains search
indexes for recall. The project also supports encrypted CMA archive export, but
archive encryption does not protect the live store files.

The safest implementation path is to separate three guarantees:

- live redb record payload encryption
- search index exposure
- deletion and audit guarantees

Treating them as one feature would create misleading security claims, especially
because full-text indexes intentionally store searchable text.

## Decision

Cerememory supports optional live-store payload encryption for persistent redb
data: raw journal, episodic, semantic, procedural, emotional, and vector
payloads. The value is encrypted after MessagePack serialization and framed
with a versioned envelope. Existing plaintext payloads remain readable for
migration compatibility until an operator rewrites them.

The store encryption passphrase is supplied through configuration, with
environment variables preferred over TOML:

```bash
export CEREMEMORY_SECURITY__STORE_ENCRYPTION_PASSPHRASE="..."
```

This passphrase is independent from CMA archive export passphrases.

After enabling the passphrase, operators can rewrite existing plaintext
persistent redb payloads in place:

```bash
cerememory migrate-store-encryption --confirm
```

The migration is idempotent. It also verifies that already encrypted payloads
can be decrypted with the active passphrase, so the wrong key cannot be treated
as a successful migration.

Full-text indexes are not encrypted because they must contain searchable terms.
When store encryption is enabled, Cerememory disables persistent full-text
indexes by default and rebuilds them in memory from encrypted stores at startup.
Operators may set `security.persist_search_indexes = true` only if searchable
derived text on disk is acceptable.

## Consequences

- New and migrated persistent redb payloads can be encrypted without breaking
  existing plaintext datasets during rollout.
- A missing or wrong passphrase prevents reading encrypted records.
- Existing plaintext payloads require `migrate-store-encryption` before they
  gain live-store encryption.
- Full-text search remains available through in-memory rebuilt indexes by
  default when store encryption is enabled.
- Tamper-evident audit logging is implemented separately from store encryption:
  the audit log is a plaintext operation-metadata hash chain, while store
  encryption protects selected redb payload confidentiality.
