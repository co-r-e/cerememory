# Human-Plus Memory Phase 1

## Goal

Build a memory architecture that preserves the full externalized conversation history while still behaving like a dynamic, human-like memory system during normal recall.

Phase 1 establishes the split between:

- `Preservation Plane`: append-only raw journal of everything worth preserving verbatim
- `Cognition Plane`: curated episodic / semantic / procedural / emotional memory used for normal recall

The system should support both:

- human-like day-to-day recall with compression, suppression, decay, and consolidation
- forensic recall of the original preserved material when explicitly requested

## Non-Goals

Phase 1 does not attempt to:

- capture hidden model chain-of-thought
- expose raw journal records in normal `recall.query`
- solve full access-control or multi-tenant policy design
- replace existing five-store recall behavior
- implement a fully autonomous dream pipeline

## Existing Constraints

The current engine already provides the core dynamics needed for the cognition plane:

- five stores via `StoreType`: episodic, semantic, procedural, emotional, working
- per-record metadata and context injection
- human vs perfect recall rendering
- consolidation from episodic to semantic
- decay and evolution feedback loops

These are already sufficient to support a human-like curated memory layer. What is missing is a separate preserved raw substrate.

## Architectural Decision

Phase 1 introduces a new **sidecar raw journal**, not a new `StoreType`.

Rationale:

- Adding `Raw` to `StoreType` would immediately contaminate default recall paths because current recall walks all stores by default.
- Raw preservation and normal memory cognition serve different purposes and should be separated at the API and index level.
- A sidecar layer can evolve independently without destabilizing the current engine and SDK surface.

## Phase 1 Components

### 1. Raw Journal Store

Add a new crate:

- `crates/cerememory-store-raw`

Responsibilities:

- append-only persistence of raw preserved events
- lookup by journal id
- filtering by session id, turn id, topic id, speaker, visibility, secrecy level, and time range
- optional full-text search over preserved text

Recommended backend:

- `redb` for persistence symmetry with existing stores
- append-first storage model
- secondary in-memory or Tantivy-backed index can be added later

### 2. Raw Journal Record

Add a new record type separate from `MemoryRecord`.

```rust
pub struct RawJournalRecord {
    pub id: Uuid,
    pub session_id: String,
    pub turn_id: Option<String>,
    pub topic_id: Option<String>,
    pub source: RawSource,
    pub speaker: RawSpeaker,
    pub visibility: RawVisibility,
    pub secrecy_level: SecrecyLevel,
    pub created_at: DateTime<Utc>,
    pub content: MemoryContent,
    pub metadata: serde_json::Value,
    pub derived_memory_ids: Vec<Uuid>,
    pub suppressed: bool,
}
```

New enums:

- `RawSource`: `conversation`, `tool_io`, `scratchpad`, `summary`, `imported`
- `RawSpeaker`: `user`, `assistant`, `system`, `tool`
- `RawVisibility`: `normal`, `private_scratch`, `sealed`
- `SecrecyLevel`: `public`, `sensitive`, `secret`

### 3. Journal Ingest API

Add new protocol operations instead of overloading `encode.store`.

Recommended API additions:

- `encode.store_raw`
- `encode.batch_store_raw`

Why:

- current MCP `store` only accepts `content`, `store`, and `emotion`
- raw preservation needs session and visibility metadata
- curated memory and preserved raw input should remain separate concepts

Minimum request shape:

```rust
pub struct EncodeStoreRawRequest {
    pub header: Option<CMPHeader>,
    pub session_id: String,
    pub turn_id: Option<String>,
    pub topic_id: Option<String>,
    pub source: RawSource,
    pub speaker: RawSpeaker,
    pub visibility: RawVisibility,
    pub secrecy_level: SecrecyLevel,
    pub content: MemoryContent,
    pub metadata: Option<serde_json::Value>,
}
```

### 4. Forensic Recall API

Do not change normal `recall.query` in Phase 1.

Instead add explicit raw retrieval APIs:

- `recall.raw_query`
- `recall.session`

Capabilities:

- query raw journal by keyword
- retrieve all preserved records for a session
- filter by visibility and secrecy level
- return reverse links to derived curated memories when present

### 5. Dream Tick Stub

Add a new lifecycle operation:

- `lifecycle.dream_tick`

Phase 1 behavior:

- scan raw journal records not yet processed
- group by session/topic/day
- emit compact episodic summaries into the existing episodic store
- attach backlinks from summary records to the source raw journal ids via metadata
- mark source raw records as dream-processed

Phase 1 should not yet:

- mutate semantic memory aggressively
- rewrite or delete raw records
- implement advanced suppression learning

## Recall Modes

Keep current `RecallMode::{Human, Perfect}` unchanged for curated memory.

Introduce **recall profiles** at the API layer rather than overloading `RecallMode`.

Recommended profiles:

- `human`: current curated recall behavior
- `deep`: curated recall with larger activation depth and less suppression
- `forensic`: raw journal aware, explicit, opt-in only
- `private_scratch`: includes `RawVisibility::PrivateScratch`

Phase 1 only needs:

- preserve current behavior for `human`
- add explicit forensic endpoints for raw access

## Suppression Model

Phase 1 suppression is simple and rule-based.

Rules:

- `private_scratch` is excluded from normal recall
- `sealed` is excluded from all recall except explicit forensic access
- `secret` records are excluded from normal MCP recall
- dream summaries inherit backlinks but do not expose sealed raw content

Suppression in Phase 1 is **reversible filtering**, not deletion.

## Security Requirements

Raw journal introduces much higher sensitivity than the current curated stores.

Minimum requirements for Phase 1:

- raw journal must support encrypted export from day one
- `sealed` and `secret` records must never appear in normal `recall.query`
- forensic retrieval must be explicit and auditable
- secret-bearing records should be taggable during ingest

## Metadata Conventions

Curated records generated by `dream_tick` should write backlinks into metadata:

```json
{
  "_origin": {
    "raw_session_id": "sess-123",
    "raw_record_ids": ["...", "..."],
    "dream_tick_at": "2026-04-06T00:00:00Z"
  }
}
```

Raw records should track derived outputs:

```json
{
  "_derived": {
    "episodic_summary_ids": ["..."],
    "semantic_ids": []
  }
}
```

## MCP Strategy

MCP should not auto-monitor conversations by itself.

Phase 1 MCP changes:

- keep current curated tools as-is
- add explicit raw-preservation tools
- add a `dream_tick` tool for scheduled or manual invocation

Recommended MCP additions:

- `store_raw`
- `batch_store_raw`
- `recall_raw`
- `session_raw`
- `dream_tick`

## Implementation Order

### Milestone 1

- add `cerememory-store-raw`
- add `RawJournalRecord` and supporting enums to core types
- add engine ownership and persistence plumbing for raw journal
- add unit tests for append, retrieve, and session filtering

### Milestone 2

- add protocol types for raw ingest and raw recall
- add engine methods for raw ingest / raw retrieval
- add HTTP and MCP bindings for those methods
- add tests proving raw records do not leak into normal recall

### Milestone 3

- add `lifecycle.dream_tick`
- generate episodic summaries from raw session clusters
- store backlinks between raw and curated records
- add tests proving dream outputs are traceable back to raw inputs

## Acceptance Criteria

Phase 1 is complete when:

- verbatim preserved session material can be written and retrieved by explicit raw APIs
- normal `recall.query` behavior remains backward compatible
- secret/private raw records do not leak into normal recall
- `dream_tick` can produce episodic summary records from raw sessions
- curated summary records retain raw backlinks

## Open Questions For Phase 2

- Should raw journal text use Tantivy or a dedicated append-log index?
- Should suppression be per-user policy, per-session policy, or record-local metadata?
- Should nightly dream run as a background task or as explicit scheduled lifecycle work?
- Should semantic promotion from raw-derived episodic summaries be automatic or threshold-based?
