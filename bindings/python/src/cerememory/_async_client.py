"""Asynchronous Cerememory HTTP client.

Provides full CMP protocol access over HTTP using ``httpx.AsyncClient``.
"""

from __future__ import annotations

from typing import Any, TypeVar
from uuid import UUID

import httpx

from cerememory._endpoints import dump, dump_without_id
from cerememory._transport import AsyncTransport
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


class AsyncCerememoryClient:
    """Asynchronous HTTP client for the Cerememory REST API.

    Best used as an async context manager::

        async with AsyncCerememoryClient("http://localhost:8420", api_key="sk-...") as client:
            resp = await client.encode_store(EncodeStoreRequest(...))
            print(resp.record_id)

    Args:
        base_url: The base URL of the Cerememory server.
        api_key: Optional Bearer token for authentication.
        timeout: Request timeout in seconds (default 30).
        max_retries: Number of retries on transient errors (default 0).
        retry_mutating_requests: Whether retries may also apply to mutating requests.
        headers: Additional headers to include in every request.
        http_client: Optional pre-configured ``httpx.AsyncClient`` instance.
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
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._transport = AsyncTransport(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_mutating_requests=retry_mutating_requests,
            headers=headers,
            http_client=http_client,
        )

    async def close(self) -> None:
        """Close the HTTP transport."""
        await self._transport.close()

    async def __aenter__(self) -> AsyncCerememoryClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, body: Any, response_model: type[M]) -> M:
        resp = await self._transport.request("POST", path, json=body)
        return response_model.model_validate(resp.json())

    async def _get(
        self, path: str, response_model: type[M], params: dict[str, Any] | None = None
    ) -> M:
        resp = await self._transport.request("GET", path, params=params)
        return response_model.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> HealthResponse:
        """Check server health (``GET /health``)."""
        return await self._get("/health", HealthResponse)

    async def readiness(self) -> ReadinessResponse:
        """Check server readiness (``GET /readiness``)."""
        return await self._get("/readiness", ReadinessResponse)

    # ------------------------------------------------------------------
    # Encode Operations
    # ------------------------------------------------------------------

    async def encode_store(self, request: EncodeStoreRequest) -> EncodeStoreResponse:
        """Store a new memory record (``POST /v1/encode``)."""
        return await self._post("/v1/encode", dump(request), EncodeStoreResponse)

    async def encode_batch(self, request: EncodeBatchRequest) -> EncodeBatchResponse:
        """Store multiple memory records (``POST /v1/encode/batch``)."""
        return await self._post("/v1/encode/batch", dump(request), EncodeBatchResponse)

    async def encode_update(self, record_id: UUID, request: EncodeUpdateRequest) -> None:
        """Update an existing memory record (``PATCH /v1/encode/:record_id``)."""
        await self._transport.request(
            "PATCH", f"/v1/encode/{record_id}", json=dump_without_id(request)
        )

    # ------------------------------------------------------------------
    # Recall Operations
    # ------------------------------------------------------------------

    async def recall_query(self, request: RecallQueryRequest) -> RecallQueryResponse:
        """Query memories by cue (``POST /v1/recall/query``)."""
        return await self._post("/v1/recall/query", dump(request), RecallQueryResponse)

    async def recall_associate(
        self, record_id: UUID, request: RecallAssociateRequest
    ) -> RecallAssociateResponse:
        """Find associated memories (``POST /v1/recall/associate/:id``)."""
        resp = await self._transport.request(
            "POST", f"/v1/recall/associate/{record_id}", json=dump_without_id(request)
        )
        return RecallAssociateResponse.model_validate(resp.json())

    async def recall_timeline(
        self, request: RecallTimelineRequest
    ) -> RecallTimelineResponse:
        """Query memories by timeline (``POST /v1/recall/timeline``)."""
        return await self._post("/v1/recall/timeline", dump(request), RecallTimelineResponse)

    async def recall_graph(self, request: RecallGraphRequest) -> RecallGraphResponse:
        """Query the memory association graph (``POST /v1/recall/graph``)."""
        return await self._post("/v1/recall/graph", dump(request), RecallGraphResponse)

    # ------------------------------------------------------------------
    # Lifecycle Operations
    # ------------------------------------------------------------------

    async def lifecycle_consolidate(
        self, request: ConsolidateRequest
    ) -> ConsolidateResponse:
        """Trigger memory consolidation (``POST /v1/lifecycle/consolidate``)."""
        return await self._post("/v1/lifecycle/consolidate", dump(request), ConsolidateResponse)

    async def lifecycle_decay_tick(self, request: DecayTickRequest) -> DecayTickResponse:
        """Trigger a decay tick (``POST /v1/lifecycle/decay-tick``)."""
        return await self._post("/v1/lifecycle/decay-tick", dump(request), DecayTickResponse)

    async def lifecycle_set_mode(self, request: SetModeRequest) -> None:
        """Set the recall mode (``PUT /v1/lifecycle/mode``)."""
        await self._transport.request("PUT", "/v1/lifecycle/mode", json=dump(request))

    async def lifecycle_forget(self, request: ForgetRequest) -> ForgetResponse:
        """Forget (delete) memory records (``DELETE /v1/lifecycle/forget``)."""
        resp = await self._transport.request(
            "DELETE", "/v1/lifecycle/forget", json=dump(request)
        )
        return ForgetResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Introspect Operations
    # ------------------------------------------------------------------

    async def introspect_stats(self) -> StatsResponse:
        """Get memory store statistics (``GET /v1/introspect/stats``)."""
        return await self._get("/v1/introspect/stats", StatsResponse)

    async def introspect_record(self, record_id: UUID) -> MemoryRecord:
        """Get a single memory record (``GET /v1/introspect/record/:id``)."""
        return await self._get(f"/v1/introspect/record/{record_id}", MemoryRecord)

    async def introspect_decay_forecast(
        self, request: DecayForecastRequest
    ) -> DecayForecastResponse:
        """Forecast future decay of records (``POST /v1/introspect/decay-forecast``)."""
        return await self._post(
            "/v1/introspect/decay-forecast", dump(request), DecayForecastResponse
        )

    async def introspect_evolution(self) -> EvolutionMetrics:
        """Get evolution metrics (``GET /v1/introspect/evolution``)."""
        return await self._get("/v1/introspect/evolution", EvolutionMetrics)
