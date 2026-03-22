# Migration Guide: 0.2.0

This release tightens security defaults, changes SDK retry behavior, and aligns the engine's persisted graph model with runtime behavior.

## Breaking Changes

### CORS is no longer wildcard-by-default

Previous behavior:

- Empty `http.cors_origins` effectively allowed browser access from any origin.

Current behavior:

- Empty `http.cors_origins` emits no CORS headers.

Action:

- Set `http.cors_origins` explicitly if you have a browser client.
- Use `["*"]` only if you intentionally want broad browser access.

### `/metrics` is opt-in and protected

Previous behavior:

- `/metrics` was available by default.

Current behavior:

- `/metrics` exists only when `http.metrics_enabled = true`.
- When auth is enabled, `/metrics` requires the same Bearer token policy as the API.
- `/metrics` is rate-limited.

Action:

- Enable metrics explicitly in config.
- Update Prometheus or other scrapers to send auth headers when needed.

### gRPC now requires TLS in more cases

Current rule:

- If `auth.enabled = true`, gRPC requires TLS.
- If `http.bind_address` is non-loopback, gRPC requires TLS.

Action:

- Configure both `grpc.tls_cert_path` and `grpc.tls_key_path`.
- If you only need local development, keep the bind address on loopback and auth disabled.

### SDK retries are now conservative

Previous behavior:

- SDK retries could apply to mutating requests.

Current behavior:

- Default retries are `0`.
- Even when retries are enabled, only safe requests are retried unless mutating retries are explicitly enabled.

Action:

- TypeScript: set `retryMutatingRequests: true` only if your workflow is idempotent.
- Python: set `retry_mutating_requests=True` only if your workflow is idempotent.

## New Config and Flags

### Config

- `http.trusted_proxy_cidrs`
- `http.metrics_enabled`

### CLI

Export:

- `--format`
- `--stores`
- `--encrypt`
- `--encryption-key`
- `--encryption-key-stdin`

Import:

- `--decryption-key`
- `--decryption-key-stdin`
- `--conflict-resolution`

## Behavioral Fixes You Should Know About

- Inferred associations are now persisted and survive rebuild/export/import.
- `ForgetRequest.temporal_range` is now implemented.
- `recall.query` temporal filtering now applies across all stores.
- Decay ticks no longer over-apply simulated time.

## Recommended Rollout Checklist

1. Update server configuration for CORS, metrics, and gRPC TLS.
2. Update any browser client origin allowlists.
3. Update metrics scrapers and dashboards.
4. Review SDK client retry settings in production code.
5. Validate backup and restore workflows using the new CLI flags.
