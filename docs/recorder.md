# Cerememory Recorder Companion

`cerememory-recorder` is a companion binary for preserving observed agent and
tool events in Cerememory's raw journal. It keeps the core database focused on
memory storage: the recorder only normalizes Capture Events, applies small
safety guards, and sends batches to the Cerememory HTTP API.

It does not execute tools, plan work, operate clients, or promote raw material
into semantic memory. Normal memory curation remains the job of `dream_tick`.

## Capture Event JSONL

`ingest` reads one JSON object per line from stdin.

Required fields:

| Field | Meaning |
|-------|---------|
| `session_id` | Stable session or conversation identifier |
| `event_type` | One of the supported recorder event types |
| `content` | Text or JSON payload to preserve verbatim |

Optional fields:

| Field | Meaning |
|-------|---------|
| `turn_id` | Conversation turn identifier |
| `topic_id` | Optional topic/thread grouping identifier forwarded to raw journal |
| `action_id` | Tool/action identifier |
| `source` | Client or adapter name, stored as metadata |
| `speaker` | `user`, `assistant`, `tool`, or `system`; otherwise inferred |
| `timestamp` | RFC3339 observation timestamp, stored as metadata |
| `visibility` | `normal`, `private_scratch`, or `sealed`; default `normal` |
| `secrecy_level` | `public`, `sensitive`, or `secret`; default `sensitive` |
| `metadata` | Caller-supplied metadata, recursively redacted |
| `meta` | Caller-supplied MetaMemory, forwarded only when present and redacted |

Supported `event_type` values:

- `user_message`
- `assistant_message`
- `tool_call`
- `tool_result`
- `command`
- `file_change`
- `session_summary`
- `error`

Example:

```json
{"session_id":"codex-2026-04-28","event_type":"user_message","content":"Investigate the failing recorder test","source":"codex","metadata":{"cwd":"/repo"}}
```

## Ingest

Start a Cerememory HTTP server first:

```bash
cerememory serve --port 8420
```

Then pipe Capture Event JSONL into the recorder:

```bash
printf '%s\n' '{"session_id":"demo","event_type":"user_message","content":"hello recorder"}' \
  | cerememory-recorder ingest --server-url http://127.0.0.1:8420
```

If the server requires HTTP auth, set the API key in the environment. Do not put
API keys on the command line:

```bash
export CEREMEMORY_SERVER_API_KEY='...'
```

When `CEREMEMORY_SERVER_API_KEY` is set, the recorder refuses to send it to a
non-loopback `http://` URL. Use `https://` for remote servers, or local-only
`http://localhost`, `http://127.0.0.1`, or `http://[::1]`.

`ingest` sends `/v1/encode/raw/batch` requests. Failed batches are written as
recorder-owned `batch-*.jsonl` files under `~/.cerememory/recorder/spool` by
default, after redaction has been applied. Retryable failures such as network
errors, HTTP 429, and HTTP 5xx are spooled and ingestion continues.
Non-retryable failures such as HTTP 401 are also spooled, but the command exits
with an error so configuration problems are not silently hidden.

On Unix, newly created spool directories are set to `0700` and spool files are
created with `0600`. Existing spool directories must already be private; the
recorder refuses to chmod an arbitrary existing directory. A later `ingest` run
attempts to flush pending recorder-owned spool files before reading new stdin.
If a recorder-owned `batch-*.jsonl` spool file cannot be parsed, it is moved to
`*.bad-*` in the same private directory and the recorder continues with other
pending files. The quarantined file is retained for manual inspection.

## Codex Hook Helper

Generate a non-destructive helper script:

```bash
cerememory-recorder install-hook codex --server-url http://127.0.0.1:8420
```

The command writes:

- `.codex/hooks/cerememory-recorder-codex-hook.py`
- `.codex/hooks/cerememory-recorder.example.json`

Existing files are not overwritten unless `--force` is passed. The installer
checks both generated paths before writing either file and refuses symlink
targets, including with `--force`. It does not edit existing Codex hook
settings; copy the generated command into the hook configuration you choose to
use and provide `CEREMEMORY_SERVER_API_KEY` through the shell or hook
environment if auth is enabled.

The generated hook is best-effort: if the recorder binary is missing, fails, or
does not return within 15 seconds, the hook prints a diagnostic to stderr and
lets the Codex session continue.
It preserves common tool payload fields such as `tool_name`, `tool_call_id`,
`tool_input`, `tool_result`, `exit_code`, `cwd`, and `transcript_path` in
Capture Event metadata when those fields are present.

## Doctor

Run diagnostics against the HTTP server:

```bash
cerememory-recorder doctor --server-url http://127.0.0.1:8420
```

`doctor` checks:

- `/health`
- `/readiness`
- spool directory writability
- authenticated raw batch ingest

The raw ingest check writes a small `private_scratch` probe record so it can
distinguish connectivity, auth, rate-limit, and server failures.
Use `--skip-raw-ingest-probe` when you need a non-mutating check; in that mode
the recorder verifies auth with a `limit=0` raw recall probe instead of writing
a raw journal record:

```bash
cerememory-recorder doctor --server-url http://127.0.0.1:8420 --skip-raw-ingest-probe
```

## Safety Defaults

- `visibility` defaults to `normal`.
- `secrecy_level` defaults to `sensitive`.
- JSONL lines larger than 256 KiB are rejected before parsing, including
  surrounding whitespace.
- recent duplicate events are dropped only when they carry an explicit
  `turn_id`, `action_id`, or `timestamp` identity and the same content.
- obvious bearer/basic auth headers, API keys, tokens, cookies, client secrets,
  private-key blocks, passwords, and common token shapes are redacted from text,
  metadata, and caller-supplied `meta` before sending or spooling.
- caller-supplied `meta` is forwarded only when present; the recorder does not
  invent MetaMemory intent or rationale.
