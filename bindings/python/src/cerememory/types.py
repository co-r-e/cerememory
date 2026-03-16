"""Pydantic v2 models for the Cerememory Protocol (CMP).

These types mirror the Rust structs in ``cerememory-core/src/types.rs`` and
``cerememory-core/src/protocol.rs``, serialised as JSON over the REST API.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StoreType(str, Enum):
    """Target memory store."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EMOTIONAL = "emotional"
    WORKING = "working"


class Modality(str, Enum):
    """Content modality for a content block."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    INTEROCEPTIVE = "interoceptive"


class AssociationType(str, Enum):
    """Type of association between memory records."""

    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    CAUSAL = "causal"
    SEQUENTIAL = "sequential"
    CROSS_MODAL = "cross_modal"
    USER_DEFINED = "user_defined"


class RecallMode(str, Enum):
    """Recall fidelity mode."""

    HUMAN = "human"
    PERFECT = "perfect"


class TimeGranularity(str, Enum):
    """Granularity for timeline recall queries."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class ConsolidationStrategy(str, Enum):
    """Strategy for memory consolidation."""

    FULL = "full"
    INCREMENTAL = "incremental"
    SELECTIVE = "selective"


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------


class ContentBlock(BaseModel):
    """A single content block within a memory record.

    The ``data`` field is transmitted as a base64-encoded string over JSON
    and decoded to ``bytes`` in Python.
    """

    modality: Modality
    format: str
    data: bytes
    embedding: list[float] | None = None

    @field_serializer("data")
    @classmethod
    def _serialize_data(cls, v: bytes, _info: Any) -> list[int]:
        """Serialize bytes as a JSON array of integers (matching Rust serde default)."""
        return list(v)

    @field_validator("data", mode="before")
    @classmethod
    def _deserialize_data(cls, v: Any) -> bytes:
        """Accept bytes, list of ints, or base64 string."""
        if isinstance(v, bytes):
            return v
        if isinstance(v, list):
            return bytes(v)
        if isinstance(v, str):
            return base64.b64decode(v)
        raise ValueError(f"Cannot convert {type(v).__name__} to bytes")


class MemoryContent(BaseModel):
    """The payload of a memory record."""

    blocks: list[ContentBlock]
    summary: str | None = None


class FidelityState(BaseModel):
    """Decay state of a memory record."""

    score: float = 1.0
    noise_level: float = 0.0
    decay_rate: float = 0.3
    emotional_anchor: float = 1.0
    reinforcement_count: int = 0
    stability: float = 1.0
    last_decay_tick: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


class EmotionVector(BaseModel):
    """Multi-dimensional affective representation (Plutchik model)."""

    joy: float = 0.0
    trust: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    sadness: float = 0.0
    disgust: float = 0.0
    anger: float = 0.0
    anticipation: float = 0.0
    intensity: float = 0.0
    valence: float = 0.0


class Association(BaseModel):
    """A weighted, typed link between two memory records."""

    target_id: UUID
    association_type: AssociationType
    weight: float
    created_at: datetime
    last_co_activation: datetime


class MemoryRecord(BaseModel):
    """The fundamental unit of storage in Cerememory."""

    id: UUID
    store: StoreType
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime
    access_count: int = 0
    content: MemoryContent
    fidelity: FidelityState = Field(default_factory=FidelityState)
    emotion: EmotionVector = Field(default_factory=EmotionVector)
    associations: list[Association] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1


# ---------------------------------------------------------------------------
# Protocol Types — CMP Header
# ---------------------------------------------------------------------------


class CMPHeader(BaseModel):
    """CMP protocol header."""

    protocol: str = "cmp"
    version: str = "1.0"
    request_id: UUID
    timestamp: datetime


# ---------------------------------------------------------------------------
# Encode Operations
# ---------------------------------------------------------------------------


class EncodeContext(BaseModel):
    """Contextual metadata for encode operations."""

    source: str | None = None
    session_id: str | None = None
    spatial: Any | None = None
    temporal: Any | None = None


class ManualAssociation(BaseModel):
    """Manual association hint provided during encoding."""

    target_id: UUID
    association_type: AssociationType
    weight: float


class EncodeStoreRequest(BaseModel):
    """Request body for ``POST /v1/encode``."""

    header: CMPHeader | None = None
    content: MemoryContent
    store: StoreType | None = None
    emotion: EmotionVector | None = None
    context: EncodeContext | None = None
    associations: list[ManualAssociation] | None = None


class EncodeStoreResponse(BaseModel):
    """Response from ``POST /v1/encode``."""

    record_id: UUID
    store: StoreType
    initial_fidelity: float
    associations_created: int


class EncodeBatchRequest(BaseModel):
    """Request body for ``POST /v1/encode/batch``."""

    header: CMPHeader | None = None
    records: list[EncodeStoreRequest]
    infer_associations: bool = False


class EncodeBatchResponse(BaseModel):
    """Response from ``POST /v1/encode/batch``."""

    results: list[EncodeStoreResponse]
    associations_inferred: int


class EncodeUpdateRequest(BaseModel):
    """Request body for ``PATCH /v1/encode/:record_id``."""

    header: CMPHeader | None = None
    record_id: UUID
    content: MemoryContent | None = None
    emotion: EmotionVector | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Recall Operations
# ---------------------------------------------------------------------------


class TemporalRange(BaseModel):
    """Time range filter."""

    start: datetime
    end: datetime


class RecallCue(BaseModel):
    """Multimodal recall cue."""

    text: str | None = None
    image: bytes | None = None
    audio: bytes | None = None
    emotion: EmotionVector | None = None
    temporal: TemporalRange | None = None
    spatial: Any | None = None
    semantic: Any | None = None
    embedding: list[float] | None = None

    @field_serializer("image", "audio")
    @classmethod
    def _serialize_bytes(cls, v: bytes | None, _info: Any) -> list[int] | None:
        if v is None:
            return None
        return list(v)

    @field_validator("image", "audio", mode="before")
    @classmethod
    def _deserialize_bytes(cls, v: Any) -> bytes | None:
        if v is None:
            return None
        if isinstance(v, bytes):
            return v
        if isinstance(v, list):
            return bytes(v)
        if isinstance(v, str):
            return base64.b64decode(v)
        raise ValueError(f"Cannot convert {type(v).__name__} to bytes")


class RecallQueryRequest(BaseModel):
    """Request body for ``POST /v1/recall/query``."""

    header: CMPHeader | None = None
    cue: RecallCue
    stores: list[StoreType] | None = None
    limit: int = 10
    min_fidelity: float | None = None
    include_decayed: bool = False
    reconsolidate: bool = True
    activation_depth: int = 2
    recall_mode: RecallMode = RecallMode.HUMAN


class RecalledMemory(BaseModel):
    """A recalled memory with relevance scoring."""

    record: MemoryRecord
    relevance_score: float
    activation_path: list[UUID] | None = None
    rendered_content: MemoryContent


class ActivationNode(BaseModel):
    """A node in an activation trace."""

    record_id: UUID
    activation_level: float
    hop: int
    edge_type: AssociationType


class ActivationTrace(BaseModel):
    """Activation trace for debugging recall paths."""

    source_id: UUID
    activations: list[ActivationNode]


class RecallQueryResponse(BaseModel):
    """Response from ``POST /v1/recall/query``."""

    memories: list[RecalledMemory]
    activation_trace: ActivationTrace | None = None
    total_candidates: int


class RecallAssociateRequest(BaseModel):
    """Request body for ``POST /v1/recall/associate/:id``."""

    header: CMPHeader | None = None
    record_id: UUID
    association_types: list[AssociationType] | None = None
    depth: int = 2
    min_weight: float = 0.1
    limit: int = 10


class RecallAssociateResponse(BaseModel):
    """Response from ``POST /v1/recall/associate/:id``."""

    memories: list[RecalledMemory]
    total_candidates: int


class RecallTimelineRequest(BaseModel):
    """Request body for ``POST /v1/recall/timeline``."""

    header: CMPHeader | None = None
    range: TemporalRange
    granularity: TimeGranularity = TimeGranularity.HOUR
    min_fidelity: float | None = None
    emotion_filter: EmotionVector | None = None


class TimelineBucket(BaseModel):
    """A time bucket in a timeline response."""

    start: datetime
    end: datetime
    memories: list[RecalledMemory]
    count: int


class RecallTimelineResponse(BaseModel):
    """Response from ``POST /v1/recall/timeline``."""

    buckets: list[TimelineBucket]


class RecallGraphRequest(BaseModel):
    """Request body for ``POST /v1/recall/graph``."""

    header: CMPHeader | None = None
    center_id: UUID | None = None
    depth: int = 2
    edge_types: list[str] | None = None
    limit_nodes: int = 10


class GraphNode(BaseModel):
    """A node in the memory graph."""

    id: UUID
    store: StoreType
    summary: str | None = None
    fidelity: float


class GraphEdge(BaseModel):
    """An edge in the memory graph."""

    source: UUID
    target: UUID
    edge_type: AssociationType
    weight: float


class RecallGraphResponse(BaseModel):
    """Response from ``POST /v1/recall/graph``."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_nodes: int


# ---------------------------------------------------------------------------
# Lifecycle Operations
# ---------------------------------------------------------------------------


class ConsolidateRequest(BaseModel):
    """Request body for ``POST /v1/lifecycle/consolidate``."""

    header: CMPHeader | None = None
    strategy: ConsolidationStrategy = ConsolidationStrategy.INCREMENTAL
    min_age_hours: int = 0
    min_access_count: int = 0
    dry_run: bool = False


class ConsolidateResponse(BaseModel):
    """Response from ``POST /v1/lifecycle/consolidate``."""

    records_processed: int
    records_migrated: int
    records_compressed: int
    records_pruned: int
    semantic_nodes_created: int


class DecayTickRequest(BaseModel):
    """Request body for ``POST /v1/lifecycle/decay-tick``."""

    header: CMPHeader | None = None
    tick_duration_seconds: int | None = None


class DecayTickResponse(BaseModel):
    """Response from ``POST /v1/lifecycle/decay-tick``."""

    records_updated: int
    records_below_threshold: int
    records_pruned: int


class SetModeRequest(BaseModel):
    """Request body for ``PUT /v1/lifecycle/mode``."""

    header: CMPHeader | None = None
    mode: RecallMode
    scope: list[StoreType] | None = None


class ForgetRequest(BaseModel):
    """Request body for ``DELETE /v1/lifecycle/forget``."""

    header: CMPHeader | None = None
    record_ids: list[UUID] | None = None
    store: StoreType | None = None
    temporal_range: TemporalRange | None = None
    cascade: bool = False
    confirm: bool


class ForgetResponse(BaseModel):
    """Response from ``DELETE /v1/lifecycle/forget``."""

    records_deleted: int


# ---------------------------------------------------------------------------
# Introspect Operations
# ---------------------------------------------------------------------------


class ParameterAdjustment(BaseModel):
    """A single parameter adjustment record."""

    store: StoreType
    parameter: str
    original_value: float
    current_value: float
    reason: str


class EvolutionMetrics(BaseModel):
    """Self-evolution metrics."""

    parameter_adjustments: list[ParameterAdjustment] = Field(default_factory=list)
    detected_patterns: list[str] = Field(default_factory=list)
    schema_adaptations: list[str] = Field(default_factory=list)


class StatsResponse(BaseModel):
    """Response from ``GET /v1/introspect/stats``."""

    total_records: int
    records_by_store: dict[str, int]
    total_associations: int
    avg_fidelity: float
    avg_fidelity_by_store: dict[str, float] = Field(default_factory=dict)
    oldest_record: datetime | None = None
    newest_record: datetime | None = None
    total_recall_count: int
    evolution_metrics: EvolutionMetrics | None = None
    background_decay_enabled: bool = False


class DecayForecastRequest(BaseModel):
    """Request body for ``POST /v1/introspect/decay-forecast``."""

    header: CMPHeader | None = None
    record_ids: list[UUID]
    forecast_at: datetime


class DecayForecast(BaseModel):
    """A single record's decay forecast."""

    record_id: UUID
    current_fidelity: float
    forecasted_fidelity: float
    estimated_threshold_date: datetime | None = None


class DecayForecastResponse(BaseModel):
    """Response from ``POST /v1/introspect/decay-forecast``."""

    forecasts: list[DecayForecast]


# ---------------------------------------------------------------------------
# Health / Readiness
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response from ``GET /health``."""

    status: str


class ReadinessResponse(BaseModel):
    """Response from ``GET /readiness``."""

    status: str
