"""Synchronous Cerememory HTTP client.

Provides full CMP protocol access over HTTP using ``httpx``.
"""

from __future__ import annotations

from typing import Any, TypeVar
from uuid import UUID

import httpx

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
    EncodeStoreRequest,
    EncodeStoreResponse,
    EncodeUpdateRequest,
    EvolutionMetrics,
    ForgetRequest,
    ForgetResponse,
    HealthResponse,
    MemoryRecord,
    ReadinessResponse,
    RecallAssociateRequest,
    RecallAssociateResponse,
    RecallGraphRequest,
    RecallGraphResponse,
    RecallQueryRequest,
    RecallQueryResponse,
    RecallTimelineRequest,
    RecallTimelineResponse,
    SetModeRequest,
    StatsResponse,
)

M = TypeVar("M")


class SyncCerememoryClient:
    """Synchronous HTTP client for the Cerememory REST API.

    Args:
        base_url: The base URL of the Cerememory server (e.g. ``"http://localhost:8420"``).
        api_key: Optional Bearer token for authentication.
        timeout: Request timeout in seconds (default 30).
        max_retries: Number of retries on transient errors (default 3).
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
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._transport = SyncTransport(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
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
        """Store a new memory record (``POST /v1/encode``).

        Args:
            request: The encode store request.

        Returns:
            The encode store response with record_id.
        """
        return self._post(
            "/v1/encode",
            request.model_dump(mode="json", exclude_none=True),
            EncodeStoreResponse,
        )

    def encode_batch(self, request: EncodeBatchRequest) -> EncodeBatchResponse:
        """Store multiple memory records (``POST /v1/encode/batch``).

        Args:
            request: The batch encode request.

        Returns:
            Batch encode response with results per record.
        """
        return self._post(
            "/v1/encode/batch",
            request.model_dump(mode="json", exclude_none=True),
            EncodeBatchResponse,
        )

    def encode_update(self, record_id: UUID, request: EncodeUpdateRequest) -> None:
        """Update an existing memory record (``PATCH /v1/encode/:record_id``).

        Args:
            record_id: ID of the record to update.
            request: Fields to update.
        """
        body = request.model_dump(mode="json", exclude_none=True)
        body.pop("record_id", None)
        self._transport.request("PATCH", f"/v1/encode/{record_id}", json=body)

    # ------------------------------------------------------------------
    # Recall Operations
    # ------------------------------------------------------------------

    def recall_query(self, request: RecallQueryRequest) -> RecallQueryResponse:
        """Query memories by cue (``POST /v1/recall/query``).

        Args:
            request: The recall query request.

        Returns:
            Matching memories with relevance scores.
        """
        return self._post(
            "/v1/recall/query",
            request.model_dump(mode="json", exclude_none=True),
            RecallQueryResponse,
        )

    def recall_associate(
        self, record_id: UUID, request: RecallAssociateRequest
    ) -> RecallAssociateResponse:
        """Find associated memories (``POST /v1/recall/associate/:id``).

        Args:
            record_id: ID of the source record.
            request: Association query parameters.

        Returns:
            Associated memories.
        """
        body = request.model_dump(mode="json", exclude_none=True)
        body.pop("record_id", None)
        resp = self._transport.request(
            "POST", f"/v1/recall/associate/{record_id}", json=body
        )
        return RecallAssociateResponse.model_validate(resp.json())

    def recall_timeline(self, request: RecallTimelineRequest) -> RecallTimelineResponse:
        """Query memories by timeline (``POST /v1/recall/timeline``).

        Args:
            request: Timeline query parameters.

        Returns:
            Timeline buckets with memories.
        """
        return self._post(
            "/v1/recall/timeline",
            request.model_dump(mode="json", exclude_none=True),
            RecallTimelineResponse,
        )

    def recall_graph(self, request: RecallGraphRequest) -> RecallGraphResponse:
        """Query the memory association graph (``POST /v1/recall/graph``).

        Args:
            request: Graph query parameters.

        Returns:
            Graph nodes and edges.
        """
        return self._post(
            "/v1/recall/graph",
            request.model_dump(mode="json", exclude_none=True),
            RecallGraphResponse,
        )

    # ------------------------------------------------------------------
    # Lifecycle Operations
    # ------------------------------------------------------------------

    def lifecycle_consolidate(self, request: ConsolidateRequest) -> ConsolidateResponse:
        """Trigger memory consolidation (``POST /v1/lifecycle/consolidate``).

        Args:
            request: Consolidation parameters.

        Returns:
            Consolidation results.
        """
        return self._post(
            "/v1/lifecycle/consolidate",
            request.model_dump(mode="json", exclude_none=True),
            ConsolidateResponse,
        )

    def lifecycle_decay_tick(self, request: DecayTickRequest) -> DecayTickResponse:
        """Trigger a decay tick (``POST /v1/lifecycle/decay-tick``).

        Args:
            request: Decay tick parameters.

        Returns:
            Decay tick results.
        """
        return self._post(
            "/v1/lifecycle/decay-tick",
            request.model_dump(mode="json", exclude_none=True),
            DecayTickResponse,
        )

    def lifecycle_set_mode(self, request: SetModeRequest) -> None:
        """Set the recall mode (``PUT /v1/lifecycle/mode``).

        Args:
            request: Mode and optional scope.
        """
        self._transport.request(
            "PUT",
            "/v1/lifecycle/mode",
            json=request.model_dump(mode="json", exclude_none=True),
        )

    def lifecycle_forget(self, request: ForgetRequest) -> ForgetResponse:
        """Forget (delete) memory records (``DELETE /v1/lifecycle/forget``).

        Args:
            request: Forget parameters. ``confirm`` must be ``True``.

        Returns:
            Number of records deleted.
        """
        resp = self._transport.request(
            "DELETE",
            "/v1/lifecycle/forget",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        return ForgetResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Introspect Operations
    # ------------------------------------------------------------------

    def introspect_stats(self) -> StatsResponse:
        """Get memory store statistics (``GET /v1/introspect/stats``).

        Returns:
            Aggregate statistics.
        """
        return self._get("/v1/introspect/stats", StatsResponse)

    def introspect_record(self, record_id: UUID) -> MemoryRecord:
        """Get a single memory record (``GET /v1/introspect/record/:id``).

        Args:
            record_id: The record UUID.

        Returns:
            The full memory record.
        """
        return self._get(f"/v1/introspect/record/{record_id}", MemoryRecord)

    def introspect_decay_forecast(
        self, request: DecayForecastRequest
    ) -> DecayForecastResponse:
        """Forecast future decay of records (``POST /v1/introspect/decay-forecast``).

        Args:
            request: Record IDs and forecast time.

        Returns:
            Per-record decay forecasts.
        """
        return self._post(
            "/v1/introspect/decay-forecast",
            request.model_dump(mode="json", exclude_none=True),
            DecayForecastResponse,
        )

    def introspect_evolution(self) -> EvolutionMetrics:
        """Get evolution metrics (``GET /v1/introspect/evolution``).

        Returns:
            Parameter adjustments, detected patterns, schema adaptations.
        """
        return self._get("/v1/introspect/evolution", EvolutionMetrics)
