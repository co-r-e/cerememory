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

- **Encryption at rest**: All memory stores encrypted by default (AES-256)
- **User-held keys**: Encryption keys belong to the user, not any service
- **Local-first**: No data leaves the user's machine unless explicitly configured
- **Cryptographic erasure**: `lifecycle.forget` operations are irreversible
- **Audit trail**: All encode, forget, export, and import operations are logged

See the [CMP Specification](docs/cmp-spec-v1.pdf) Section 10 for detailed security requirements.
