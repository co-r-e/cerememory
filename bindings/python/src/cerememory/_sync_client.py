"""Synchronous Cerememory HTTP client.

Provides full CMP protocol access over HTTP using ``httpx``.
"""

from __future__ import annotations

from typing import Any, TypeVar
from uuid import UUID

import httpx

from cerememory._endpoints import dump, dump_without_id
from cerememory._transport import SyncTransport
from cerememory.types import (
    ConsolidateRequest,
    ConsolidateResponse,
    DecayForecastRequest,
    DecayForecastResponse,
    DecayTickRequest,
    DecayTickResponse,
    EncodeBatchRequest,
    EncodeBatchResponse,
    EncodeBatchStoreRawRequest,
    EncodeBatchStoreRawResponse,
    EncodeStoreRequest,
    EncodeStoreResponse,
    EncodeStoreRawRequest,
    EncodeStoreRawResponse,
    EncodeUpdateRequest,
    DreamTickRequest,
    DreamTickResponse,
    EvolutionMetrics,
    ExportArchiveResponse,
    ExportRequest,
    ForgetRequest,
    ForgetResponse,
    HealthResponse,
    MemoryRecord,
    ReadinessResponse,
    RecallAssociateRequest,
    RecallAssociateResponse,
    RecallGraphRequest,
    RecallGraphResponse,
    RecallRawQueryRequest,
    RecallRawQueryResponse,
    RecallQueryRequest,
    RecallQueryResponse,
    RecallTimelineRequest,
    RecallTimelineResponse,
    SetModeRequest,
    ImportRequest,
    ImportResponse,
    StatsResponse,
)

M = TypeVar("M")


class SyncCerememoryClient:
    """Synchronous HTTP client for the Cerememory REST API.

    Args:
        base_url: The base URL of the Cerememory server (e.g. ``"http://localhost:8420"``).
        api_key: Optional Bearer token for authentication.
        timeout: Request timeout in seconds (default 30).
        max_retries: Number of retries on transient errors (default 0).
        retry_mutating_requests: Whether retries may also apply to mutating requests.
        headers: Additional headers to include in every request.
        http_client: Optional pre-configured ``httpx.Client`` instance.

    Example::

        client = SyncCerememoryClient("http://localhost:8420", api_key="sk-...")
        resp = client.encode_store(EncodeStoreRequest(...))
        print(resp.record_id)
        client.close()
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 0,
        retry_mutating_requests: bool = False,
        headers: dict[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._transport = SyncTransport(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_mutating_requests=retry_mutating_requests,
            headers=headers,
            http_client=http_client,
        )

    def close(self) -> None:
        """Close the HTTP transport."""
        self._transport.close()

    def __enter__(self) -> SyncCerememoryClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, body: Any, response_model: type[M]) -> M:
        resp = self._transport.request("POST", path, json=body)
        return response_model.model_validate(resp.json())

    def _get(self, path: str, response_model: type[M], params: dict[str, Any] | None = None) -> M:
        resp = self._transport.request("GET", path, params=params)
        return response_model.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> HealthResponse:
        """Check server health (``GET /health``)."""
        return self._get("/health", HealthResponse)

    def readiness(self) -> ReadinessResponse:
        """Check server readiness (``GET /readiness``)."""
        return self._get("/readiness", ReadinessResponse)

    # ------------------------------------------------------------------
    # Encode Operations
    # ------------------------------------------------------------------

    def encode_store(self, request: EncodeStoreRequest) -> EncodeStoreResponse:
        """Store a new memory record (``POST /v1/encode``)."""
        return self._post("/v1/encode", dump(request), EncodeStoreResponse)

    def encode_batch(self, request: EncodeBatchRequest) -> EncodeBatchResponse:
        """Store multiple memory records (``POST /v1/encode/batch``)."""
        return self._post("/v1/encode/batch", dump(request), EncodeBatchResponse)

    def encode_store_raw(
        self, request: EncodeStoreRawRequest
    ) -> EncodeStoreRawResponse:
        """Store a raw journal record (``POST /v1/encode/raw``)."""
        return self._post("/v1/encode/raw", dump(request), EncodeStoreRawResponse)

    def encode_batch_store_raw(
        self, request: EncodeBatchStoreRawRequest
    ) -> EncodeBatchStoreRawResponse:
        """Store multiple raw journal records (``POST /v1/encode/raw/batch``)."""
        return self._post(
            "/v1/encode/raw/batch", dump(request), EncodeBatchStoreRawResponse
        )

    def encode_update(self, record_id: UUID, request: EncodeUpdateRequest) -> None:
        """Update an existing memory record (``PATCH /v1/encode/:record_id``)."""
        self._transport.request("PATCH", f"/v1/encode/{record_id}", json=dump_without_id(request))

    # ------------------------------------------------------------------
    # Recall Operations
    # ------------------------------------------------------------------

    def recall_query(self, request: RecallQueryRequest) -> RecallQueryResponse:
        """Query memories by cue (``POST /v1/recall/query``)."""
        return self._post("/v1/recall/query", dump(request), RecallQueryResponse)

    def recall_raw_query(
        self, request: RecallRawQueryRequest
    ) -> RecallRawQueryResponse:
        """Query raw journal records (``POST /v1/recall/raw``)."""
        return self._post("/v1/recall/raw", dump(request), RecallRawQueryResponse)

    def recall_associate(
        self, record_id: UUID, request: RecallAssociateRequest
    ) -> RecallAssociateResponse:
        """Find associated memories (``POST /v1/recall/associate/:id``)."""
        resp = self._transport.request(
            "POST", f"/v1/recall/associate/{record_id}", json=dump_without_id(request)
        )
        return RecallAssociateResponse.model_validate(resp.json())

    def recall_timeline(self, request: RecallTimelineRequest) -> RecallTimelineResponse:
        """Query memories by timeline (``POST /v1/recall/timeline``)."""
        return self._post("/v1/recall/timeline", dump(request), RecallTimelineResponse)

    def recall_graph(self, request: RecallGraphRequest) -> RecallGraphResponse:
        """Query the memory association graph (``POST /v1/recall/graph``)."""
        return self._post("/v1/recall/graph", dump(request), RecallGraphResponse)

    # ------------------------------------------------------------------
    # Lifecycle Operations
    # ------------------------------------------------------------------

    def lifecycle_consolidate(self, request: ConsolidateRequest) -> ConsolidateResponse:
        """Trigger memory consolidation (``POST /v1/lifecycle/consolidate``)."""
        return self._post("/v1/lifecycle/consolidate", dump(request), ConsolidateResponse)

    def lifecycle_decay_tick(self, request: DecayTickRequest) -> DecayTickResponse:
        """Trigger a decay tick (``POST /v1/lifecycle/decay-tick``)."""
        return self._post("/v1/lifecycle/decay-tick", dump(request), DecayTickResponse)

    def lifecycle_dream_tick(self, request: DreamTickRequest) -> DreamTickResponse:
        """Trigger a dream tick (``POST /v1/lifecycle/dream-tick``)."""
        return self._post("/v1/lifecycle/dream-tick", dump(request), DreamTickResponse)

    def lifecycle_set_mode(self, request: SetModeRequest) -> None:
        """Set the recall mode (``PUT /v1/lifecycle/mode``)."""
        self._transport.request("PUT", "/v1/lifecycle/mode", json=dump(request))

    def lifecycle_forget(self, request: ForgetRequest) -> ForgetResponse:
        """Forget (delete) memory records (``DELETE /v1/lifecycle/forget``)."""
        resp = self._transport.request("DELETE", "/v1/lifecycle/forget", json=dump(request))
        return ForgetResponse.model_validate(resp.json())

    def lifecycle_export(self, request: ExportRequest) -> ExportArchiveResponse:
        """Export an archive bundle (``POST /v1/lifecycle/export``)."""
        return self._post("/v1/lifecycle/export", dump(request), ExportArchiveResponse)

    def lifecycle_import(self, request: ImportRequest) -> ImportResponse:
        """Import an archive bundle (``POST /v1/lifecycle/import``)."""
        return self._post("/v1/lifecycle/import", dump(request), ImportResponse)

    # ------------------------------------------------------------------
    # Introspect Operations
    # ------------------------------------------------------------------

    def introspect_stats(self) -> StatsResponse:
        """Get memory store statistics (``GET /v1/introspect/stats``)."""
        return self._get("/v1/introspect/stats", StatsResponse)

    def introspect_record(self, record_id: UUID) -> MemoryRecord:
        """Get a single memory record (``GET /v1/introspect/record/:id``)."""
        return self._get(f"/v1/introspect/record/{record_id}", MemoryRecord)

    def introspect_decay_forecast(
        self, request: DecayForecastRequest
    ) -> DecayForecastResponse:
        """Forecast future decay of records (``POST /v1/introspect/decay-forecast``)."""
        return self._post("/v1/introspect/decay-forecast", dump(request), DecayForecastResponse)

    def introspect_evolution(self) -> EvolutionMetrics:
        """Get evolution metrics (``GET /v1/introspect/evolution``)."""
        return self._get("/v1/introspect/evolution", EvolutionMetrics)
