"""High-level Cerememory client with convenience methods.

Provides both ``Client`` (sync) and ``AsyncClient`` (async) with simplified
store/recall/forget/stats methods, plus full CMP protocol access via the
underlying transport clients.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx

from cerememory._async_client import AsyncCerememoryClient
from cerememory._sync_client import SyncCerememoryClient
from cerememory.types import (
    ConsolidateRequest,
    ConsolidateResponse,
    ContentBlock,
    DecayForecastRequest,
    DecayForecastResponse,
    DecayTickRequest,
    DecayTickResponse,
    EmotionVector,
    EncodeBatchRequest,
    EncodeBatchResponse,
    EncodeStoreRequest,
    EncodeStoreResponse,
    EncodeUpdateRequest,
    EvolutionMetrics,
    ForgetRequest,
    ForgetResponse,
    HealthResponse,
    MemoryContent,
    MemoryRecord,
    Modality,
    ReadinessResponse,
    RecallAssociateRequest,
    RecallAssociateResponse,
    RecallCue,
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
)


def _text_content(text: str) -> MemoryContent:
    """Build a simple text-only MemoryContent."""
    return MemoryContent(
        blocks=[
            ContentBlock(
                modality=Modality.TEXT,
                format="text/plain",
                data=text.encode("utf-8"),
            )
        ],
    )


class Client:
    """Synchronous Cerememory client with high-level convenience methods.

    Example::

        client = cerememory.Client("http://localhost:8420", api_key="sk-...")
        record_id = client.store("Coffee chat with Alice", store="episodic")
        memories = client.recall("coffee", limit=5)
        client.forget(record_id, confirm=True)
        stats = client.stats()

    For full CMP protocol access, use the protocol methods directly::

        response = client.encode_store(EncodeStoreRequest(...))
        response = client.recall_query(RecallQueryRequest(...))

    Args:
        base_url: Base URL of the Cerememory server.
        api_key: Optional Bearer token for authentication.
        timeout: Request timeout in seconds (default 30).
        max_retries: Number of retries on transient errors (default 3).
        headers: Additional headers to include in every request.
        http_client: Optional pre-configured ``httpx.Client`` instance.
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
        self._client = SyncCerememoryClient(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            http_client=http_client,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # High-level convenience methods
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        *,
        store: str | StoreType = StoreType.EPISODIC,
        emotion: EmotionVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Store a text memory and return its record ID.

        Args:
            text: The text content to store.
            store: Target store (default ``"episodic"``).
            emotion: Optional emotion vector.
            metadata: Optional metadata dict (sent as context source).

        Returns:
            The UUID of the newly created record.
        """
        if isinstance(store, str):
            store = StoreType(store)
        request = EncodeStoreRequest(
            content=_text_content(text),
            store=store,
            emotion=emotion,
        )
        response = self._client.encode_store(request)
        return response.record_id

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        stores: list[str | StoreType] | None = None,
        mode: str | RecallMode = RecallMode.HUMAN,
        min_fidelity: float | None = None,
    ) -> list[RecalledMemory]:
        """Recall memories matching a text query.

        Args:
            query: The text cue to search for.
            limit: Maximum number of results (default 10).
            stores: Optional list of stores to search.
            mode: Recall mode (default ``"human"``).
            min_fidelity: Optional minimum fidelity threshold.

        Returns:
            List of recalled memories with relevance scores.
        """

        if isinstance(mode, str):
            mode = RecallMode(mode)
        resolved_stores: list[StoreType] | None = None
        if stores is not None:
            resolved_stores = [
                StoreType(s) if isinstance(s, str) else s for s in stores
            ]
        request = RecallQueryRequest(
            cue=RecallCue(text=query),
            limit=limit,
            stores=resolved_stores,
            recall_mode=mode,
            min_fidelity=min_fidelity,
        )
        response = self._client.recall_query(request)
        return response.memories

    def forget(
        self,
        *record_ids: UUID,
        store: str | StoreType | None = None,
        confirm: bool = True,
        cascade: bool = False,
    ) -> int:
        """Forget (delete) memory records.

        Args:
            *record_ids: UUIDs of records to delete.
            store: Delete all records in this store.
            confirm: Must be ``True`` to proceed (safety guard).
            cascade: Also delete associated records.

        Returns:
            Number of records deleted.
        """
        resolved_store: StoreType | None = None
        if store is not None:
            resolved_store = StoreType(store) if isinstance(store, str) else store
        ids = list(record_ids) if record_ids else None
        request = ForgetRequest(
            record_ids=ids,
            store=resolved_store,
            confirm=confirm,
            cascade=cascade,
        )
        response = self._client.lifecycle_forget(request)
        return response.records_deleted

    def stats(self) -> StatsResponse:
        """Get memory store statistics.

        Returns:
            Aggregate statistics about the memory store.
        """
        return self._client.introspect_stats()

    def get_record(self, record_id: UUID) -> MemoryRecord:
        """Get a single memory record by ID.

        Args:
            record_id: The record UUID.

        Returns:
            The full memory record.
        """
        return self._client.introspect_record(record_id)

    # ------------------------------------------------------------------
    # Full CMP protocol access (delegation)
    # ------------------------------------------------------------------

    def health(self) -> HealthResponse:
        """Check server health."""
        return self._client.health()

    def readiness(self) -> ReadinessResponse:
        """Check server readiness."""
        return self._client.readiness()

    def encode_store(self, request: EncodeStoreRequest) -> EncodeStoreResponse:
        """Full CMP encode.store operation."""
        return self._client.encode_store(request)

    def encode_batch(self, request: EncodeBatchRequest) -> EncodeBatchResponse:
        """Full CMP encode.batch operation."""
        return self._client.encode_batch(request)

    def encode_update(self, record_id: UUID, request: EncodeUpdateRequest) -> None:
        """Full CMP encode.update operation."""
        self._client.encode_update(record_id, request)

    def recall_query(self, request: RecallQueryRequest) -> RecallQueryResponse:
        """Full CMP recall.query operation."""
        return self._client.recall_query(request)

    def recall_associate(
        self, record_id: UUID, request: RecallAssociateRequest
    ) -> RecallAssociateResponse:
        """Full CMP recall.associate operation."""
        return self._client.recall_associate(record_id, request)

    def recall_timeline(self, request: RecallTimelineRequest) -> RecallTimelineResponse:
        """Full CMP recall.timeline operation."""
        return self._client.recall_timeline(request)

    def recall_graph(self, request: RecallGraphRequest) -> RecallGraphResponse:
        """Full CMP recall.graph operation."""
        return self._client.recall_graph(request)

    def lifecycle_consolidate(self, request: ConsolidateRequest) -> ConsolidateResponse:
        """Full CMP lifecycle.consolidate operation."""
        return self._client.lifecycle_consolidate(request)

    def lifecycle_decay_tick(self, request: DecayTickRequest) -> DecayTickResponse:
        """Full CMP lifecycle.decay_tick operation."""
        return self._client.lifecycle_decay_tick(request)

    def lifecycle_set_mode(self, request: SetModeRequest) -> None:
        """Full CMP lifecycle.set_mode operation."""
        self._client.lifecycle_set_mode(request)

    def lifecycle_forget(self, request: ForgetRequest) -> ForgetResponse:
        """Full CMP lifecycle.forget operation."""
        return self._client.lifecycle_forget(request)

    def introspect_stats(self) -> StatsResponse:
        """Full CMP introspect.stats operation."""
        return self._client.introspect_stats()

    def introspect_record(self, record_id: UUID) -> MemoryRecord:
        """Full CMP introspect.record operation."""
        return self._client.introspect_record(record_id)

    def introspect_decay_forecast(
        self, request: DecayForecastRequest
    ) -> DecayForecastResponse:
        """Full CMP introspect.decay_forecast operation."""
        return self._client.introspect_decay_forecast(request)

    def introspect_evolution(self) -> EvolutionMetrics:
        """Full CMP introspect.evolution operation."""
        return self._client.introspect_evolution()


class AsyncClient:
    """Asynchronous Cerememory client with high-level convenience methods.

    Example::

        async with cerememory.AsyncClient("http://localhost:8420", api_key="sk-...") as client:
            record_id = await client.store("Coffee chat with Alice", store="episodic")
            memories = await client.recall("coffee", limit=5)
            await client.forget(record_id, confirm=True)
            stats = await client.stats()

    Args:
        base_url: Base URL of the Cerememory server.
        api_key: Optional Bearer token for authentication.
        timeout: Request timeout in seconds (default 30).
        max_retries: Number of retries on transient errors (default 3).
        headers: Additional headers to include in every request.
        http_client: Optional pre-configured ``httpx.AsyncClient`` instance.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = AsyncCerememoryClient(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            http_client=http_client,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # High-level convenience methods
    # ------------------------------------------------------------------

    async def store(
        self,
        text: str,
        *,
        store: str | StoreType = StoreType.EPISODIC,
        emotion: EmotionVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Store a text memory and return its record ID."""
        if isinstance(store, str):
            store = StoreType(store)
        request = EncodeStoreRequest(
            content=_text_content(text),
            store=store,
            emotion=emotion,
        )
        response = await self._client.encode_store(request)
        return response.record_id

    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        stores: list[str | StoreType] | None = None,
        mode: str | RecallMode = RecallMode.HUMAN,
        min_fidelity: float | None = None,
    ) -> list[RecalledMemory]:
        """Recall memories matching a text query."""

        if isinstance(mode, str):
            mode = RecallMode(mode)
        resolved_stores: list[StoreType] | None = None
        if stores is not None:
            resolved_stores = [
                StoreType(s) if isinstance(s, str) else s for s in stores
            ]
        request = RecallQueryRequest(
            cue=RecallCue(text=query),
            limit=limit,
            stores=resolved_stores,
            recall_mode=mode,
            min_fidelity=min_fidelity,
        )
        response = await self._client.recall_query(request)
        return response.memories

    async def forget(
        self,
        *record_ids: UUID,
        store: str | StoreType | None = None,
        confirm: bool = True,
        cascade: bool = False,
    ) -> int:
        """Forget (delete) memory records."""
        resolved_store: StoreType | None = None
        if store is not None:
            resolved_store = StoreType(store) if isinstance(store, str) else store
        ids = list(record_ids) if record_ids else None
        request = ForgetRequest(
            record_ids=ids,
            store=resolved_store,
            confirm=confirm,
            cascade=cascade,
        )
        response = await self._client.lifecycle_forget(request)
        return response.records_deleted

    async def stats(self) -> StatsResponse:
        """Get memory store statistics."""
        return await self._client.introspect_stats()

    async def get_record(self, record_id: UUID) -> MemoryRecord:
        """Get a single memory record by ID."""
        return await self._client.introspect_record(record_id)

    # ------------------------------------------------------------------
    # Full CMP protocol access (delegation)
    # ------------------------------------------------------------------

    async def health(self) -> HealthResponse:
        """Check server health."""
        return await self._client.health()

    async def readiness(self) -> ReadinessResponse:
        """Check server readiness."""
        return await self._client.readiness()

    async def encode_store(self, request: EncodeStoreRequest) -> EncodeStoreResponse:
        """Full CMP encode.store operation."""
        return await self._client.encode_store(request)

    async def encode_batch(self, request: EncodeBatchRequest) -> EncodeBatchResponse:
        """Full CMP encode.batch operation."""
        return await self._client.encode_batch(request)

    async def encode_update(self, record_id: UUID, request: EncodeUpdateRequest) -> None:
        """Full CMP encode.update operation."""
        await self._client.encode_update(record_id, request)

    async def recall_query(self, request: RecallQueryRequest) -> RecallQueryResponse:
        """Full CMP recall.query operation."""
        return await self._client.recall_query(request)

    async def recall_associate(
        self, record_id: UUID, request: RecallAssociateRequest
    ) -> RecallAssociateResponse:
        """Full CMP recall.associate operation."""
        return await self._client.recall_associate(record_id, request)

    async def recall_timeline(
        self, request: RecallTimelineRequest
    ) -> RecallTimelineResponse:
        """Full CMP recall.timeline operation."""
        return await self._client.recall_timeline(request)

    async def recall_graph(self, request: RecallGraphRequest) -> RecallGraphResponse:
        """Full CMP recall.graph operation."""
        return await self._client.recall_graph(request)

    async def lifecycle_consolidate(
        self, request: ConsolidateRequest
    ) -> ConsolidateResponse:
        """Full CMP lifecycle.consolidate operation."""
        return await self._client.lifecycle_consolidate(request)

    async def lifecycle_decay_tick(
        self, request: DecayTickRequest
    ) -> DecayTickResponse:
        """Full CMP lifecycle.decay_tick operation."""
        return await self._client.lifecycle_decay_tick(request)

    async def lifecycle_set_mode(self, request: SetModeRequest) -> None:
        """Full CMP lifecycle.set_mode operation."""
        await self._client.lifecycle_set_mode(request)

    async def lifecycle_forget(self, request: ForgetRequest) -> ForgetResponse:
        """Full CMP lifecycle.forget operation."""
        return await self._client.lifecycle_forget(request)

    async def introspect_stats(self) -> StatsResponse:
        """Full CMP introspect.stats operation."""
        return await self._client.introspect_stats()

    async def introspect_record(self, record_id: UUID) -> MemoryRecord:
        """Full CMP introspect.record operation."""
        return await self._client.introspect_record(record_id)

    async def introspect_decay_forecast(
        self, request: DecayForecastRequest
    ) -> DecayForecastResponse:
        """Full CMP introspect.decay_forecast operation."""
        return await self._client.introspect_decay_forecast(request)

    async def introspect_evolution(self) -> EvolutionMetrics:
        """Full CMP introspect.evolution operation."""
        return await self._client.introspect_evolution()


# Re-export the recalled memory type for use in return type annotations.
from cerememory.types import RecalledMemory as RecalledMemory  # noqa: E402, F811
