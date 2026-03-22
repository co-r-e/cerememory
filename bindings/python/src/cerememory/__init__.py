"""Cerememory Python SDK -- HTTP client for the Cerememory Protocol (CMP).

Usage::

    import cerememory

    # Sync
    client = cerememory.Client("http://localhost:8420", api_key="sk-...")
    record_id = client.store("Coffee chat with Alice", store="episodic")
    memories = client.recall("coffee", limit=5)
    client.forget(record_id, confirm=True)

    # Async
    async with cerememory.AsyncClient("http://localhost:8420", api_key="sk-...") as client:
        record_id = await client.store("hello", store="episodic")
        memories = await client.recall("hello")
"""

from cerememory.client import AsyncClient, Client
from cerememory.errors import (
    CerememoryError,
    ConsolidationInProgressError,
    ContentTooLargeError,
    DecayEngineBusyError,
    ExportFailedError,
    ForgetUnconfirmedError,
    ImportConflictError,
    InternalError,
    ModalityUnsupportedError,
    RateLimitedError,
    RecordNotFoundError,
    StoreInvalidError,
    UnauthorizedError,
    ValidationError,
    VersionMismatchError,
)
from cerememory.types import (
    ActivationNode,
    ActivationTrace,
    Association,
    AssociationType,
    CMPHeader,
    ConsolidateRequest,
    ConsolidateResponse,
    ConsolidationStrategy,
    ContentBlock,
    DecayForecast,
    DecayForecastRequest,
    DecayForecastResponse,
    DecayTickRequest,
    DecayTickResponse,
    EmotionVector,
    EncodeBatchRequest,
    EncodeBatchResponse,
    EncodeContext,
    EncodeStoreRequest,
    EncodeStoreResponse,
    EncodeUpdateRequest,
    EvolutionMetrics,
    FidelityState,
    ForgetRequest,
    ForgetResponse,
    GraphEdge,
    GraphNode,
    HealthResponse,
    ManualAssociation,
    MemoryContent,
    MemoryRecord,
    Modality,
    ParameterAdjustment,
    QueryMetadata,
    ReadinessResponse,
    RecallAssociateRequest,
    RecallAssociateResponse,
    RecallCue,
    RecalledMemory,
    RecallGraphRequest,
    RecallGraphResponse,
    RecallMode,
    RecallQueryRequest,
    RecallQueryResponse,
    RecallTimelineRequest,
    RecallTimelineResponse,
    SetModeRequest,
    StatsResponse,
    StoreType,
    TemporalRange,
    TimeGranularity,
    TimelineBucket,
)

__version__ = "0.1.0"

__all__ = [
    # Clients
    "Client",
    "AsyncClient",
    # Version
    "__version__",
    # Enums
    "StoreType",
    "Modality",
    "AssociationType",
    "RecallMode",
    "TimeGranularity",
    "ConsolidationStrategy",
    # Core types
    "ContentBlock",
    "MemoryContent",
    "FidelityState",
    "EmotionVector",
    "Association",
    "MemoryRecord",
    "CMPHeader",
    # Encode
    "EncodeContext",
    "ManualAssociation",
    "EncodeStoreRequest",
    "EncodeStoreResponse",
    "EncodeBatchRequest",
    "EncodeBatchResponse",
    "EncodeUpdateRequest",
    # Recall
    "TemporalRange",
    "RecallCue",
    "RecallQueryRequest",
    "RecalledMemory",
    "ActivationNode",
    "ActivationTrace",
    "QueryMetadata",
    "RecallQueryResponse",
    "RecallAssociateRequest",
    "RecallAssociateResponse",
    "RecallTimelineRequest",
    "TimelineBucket",
    "RecallTimelineResponse",
    "RecallGraphRequest",
    "GraphNode",
    "GraphEdge",
    "RecallGraphResponse",
    # Lifecycle
    "ConsolidateRequest",
    "ConsolidateResponse",
    "DecayTickRequest",
    "DecayTickResponse",
    "SetModeRequest",
    "ForgetRequest",
    "ForgetResponse",
    # Introspect
    "ParameterAdjustment",
    "EvolutionMetrics",
    "StatsResponse",
    "DecayForecastRequest",
    "DecayForecast",
    "DecayForecastResponse",
    # Health
    "HealthResponse",
    "ReadinessResponse",
    # Errors
    "CerememoryError",
    "ValidationError",
    "StoreInvalidError",
    "ModalityUnsupportedError",
    "ForgetUnconfirmedError",
    "VersionMismatchError",
    "UnauthorizedError",
    "RecordNotFoundError",
    "ConsolidationInProgressError",
    "ImportConflictError",
    "ContentTooLargeError",
    "WorkingMemoryFullError",
    "RateLimitedError",
    "InternalError",
    "ExportFailedError",
    "DecayEngineBusyError",
]

# Ensure WorkingMemoryFullError is importable from the top-level even though
# it's not imported above (add it to avoid __all__ mismatch).
from cerememory.errors import WorkingMemoryFullError  # noqa: E402
