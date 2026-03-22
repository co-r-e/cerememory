"""Tests for the asynchronous AsyncClient."""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest
import respx

from cerememory import (
    AsyncClient,
    CerememoryError,
    ConsolidateRequest,
    ContentBlock,
    DecayForecastRequest,
    DecayTickRequest,
    EmotionVector,
    EncodeBatchRequest,
    EncodeStoreRequest,
    EncodeUpdateRequest,
    ForgetRequest,
    ForgetUnconfirmedError,
    MemoryContent,
    Modality,
    RecallAssociateRequest,
    RecallCue,
    RecallGraphRequest,
    RecallMode,
    RecallQueryRequest,
    RecallTimelineRequest,
    RecordNotFoundError,
    SetModeRequest,
    StoreType,
    TemporalRange,
    UnauthorizedError,
)
from tests.conftest import (
    API_KEY,
    BASE_URL,
    SAMPLE_RECORD_ID,
    make_cmp_error,
    make_encode_store_response,
    make_memory_record_json,
    make_recall_query_response,
    make_stats_response,
)


@pytest.fixture
def async_high_level_client():
    """Async high-level client fixture."""
    return AsyncClient(BASE_URL, api_key=API_KEY, max_retries=0)


class TestAsyncHighLevelStore:
    """Test AsyncClient.store() convenience method."""

    @pytest.mark.asyncio
    async def test_store_text(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/encode").mock(
                return_value=httpx.Response(200, json=make_encode_store_response())
            )
            record_id = await async_high_level_client.store(
                "Hello, world!", store="episodic"
            )
            assert record_id == SAMPLE_RECORD_ID
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_store_with_emotion(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/encode").mock(
                return_value=httpx.Response(200, json=make_encode_store_response())
            )
            record_id = await async_high_level_client.store(
                "happy memory", emotion=EmotionVector(joy=0.9)
            )
            assert isinstance(record_id, UUID)
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/encode").mock(
                return_value=httpx.Response(200, json=make_encode_store_response())
            )
            await async_high_level_client.store(
                "Session note", metadata={"source": "chat"}
            )

            import json

            body = json.loads(mock_api.calls.last.request.content)
            assert body["metadata"]["source"] == "chat"
        await async_high_level_client.close()


class TestAsyncHighLevelRecall:
    """Test AsyncClient.recall() convenience method."""

    @pytest.mark.asyncio
    async def test_recall_text(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/recall/query").mock(
                return_value=httpx.Response(200, json=make_recall_query_response())
            )
            memories = await async_high_level_client.recall("hello")
            assert len(memories) == 1
            assert memories[0].relevance_score == 0.95
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_recall_with_stores(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/recall/query").mock(
                return_value=httpx.Response(200, json=make_recall_query_response())
            )
            memories = await async_high_level_client.recall(
                "hello", stores=["episodic", "semantic"]
            )
            assert len(memories) == 1
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_recall_with_activation_options(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/recall/query").mock(
                return_value=httpx.Response(200, json=make_recall_query_response())
            )
            await async_high_level_client.recall(
                "hello", reconsolidate=False, activation_depth=5
            )

            import json

            body = json.loads(mock_api.calls.last.request.content)
            assert body["reconsolidate"] is False
            assert body["activation_depth"] == 5
        await async_high_level_client.close()


class TestAsyncHighLevelForget:
    """Test AsyncClient.forget() convenience method."""

    @pytest.mark.asyncio
    async def test_forget_by_id(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.delete("/v1/lifecycle/forget").mock(
                return_value=httpx.Response(200, json={"records_deleted": 1})
            )
            count = await async_high_level_client.forget(
                SAMPLE_RECORD_ID, confirm=True
            )
            assert count == 1
        await async_high_level_client.close()


class TestAsyncHighLevelStats:
    """Test AsyncClient.stats() convenience method."""

    @pytest.mark.asyncio
    async def test_stats(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get("/v1/introspect/stats").mock(
                return_value=httpx.Response(200, json=make_stats_response())
            )
            stats = await async_high_level_client.stats()
            assert stats.total_records == 42
        await async_high_level_client.close()


class TestAsyncProtocolMethods:
    """Test full CMP protocol method delegation via async client."""

    @pytest.mark.asyncio
    async def test_encode_store(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/encode").mock(
                return_value=httpx.Response(200, json=make_encode_store_response())
            )
            req = EncodeStoreRequest(
                content=MemoryContent(
                    blocks=[
                        ContentBlock(
                            modality=Modality.TEXT, format="text/plain", data=b"hi"
                        )
                    ]
                ),
                store=StoreType.EPISODIC,
            )
            resp = await async_high_level_client.encode_store(req)
            assert resp.record_id == SAMPLE_RECORD_ID
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_encode_batch(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/encode/batch").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "results": [make_encode_store_response()],
                        "associations_inferred": 0,
                    },
                )
            )
            req = EncodeBatchRequest(
                records=[
                    EncodeStoreRequest(
                        content=MemoryContent(
                            blocks=[
                                ContentBlock(
                                    modality=Modality.TEXT,
                                    format="text/plain",
                                    data=b"batch",
                                )
                            ]
                        )
                    )
                ]
            )
            resp = await async_high_level_client.encode_batch(req)
            assert len(resp.results) == 1
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_encode_update(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.patch(f"/v1/encode/{SAMPLE_RECORD_ID}").mock(
                return_value=httpx.Response(204)
            )
            req = EncodeUpdateRequest(
                record_id=SAMPLE_RECORD_ID,
                emotion=EmotionVector(joy=0.5),
            )
            await async_high_level_client.encode_update(SAMPLE_RECORD_ID, req)
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_recall_query(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/recall/query").mock(
                return_value=httpx.Response(200, json=make_recall_query_response())
            )
            req = RecallQueryRequest(cue=RecallCue(text="test"))
            resp = await async_high_level_client.recall_query(req)
            assert resp.total_candidates == 1
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_recall_associate(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post(f"/v1/recall/associate/{SAMPLE_RECORD_ID}").mock(
                return_value=httpx.Response(
                    200,
                    json={"memories": [], "total_candidates": 0},
                )
            )
            req = RecallAssociateRequest(record_id=SAMPLE_RECORD_ID)
            resp = await async_high_level_client.recall_associate(
                SAMPLE_RECORD_ID, req
            )
            assert resp.total_candidates == 0
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_recall_timeline(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/recall/timeline").mock(
                return_value=httpx.Response(200, json={"buckets": []})
            )
            req = RecallTimelineRequest(
                range=TemporalRange(
                    start="2026-03-01T00:00:00Z", end="2026-03-15T00:00:00Z"
                )
            )
            resp = await async_high_level_client.recall_timeline(req)
            assert resp.buckets == []
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_recall_graph(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/recall/graph").mock(
                return_value=httpx.Response(
                    200, json={"nodes": [], "edges": [], "total_nodes": 0}
                )
            )
            req = RecallGraphRequest()
            resp = await async_high_level_client.recall_graph(req)
            assert resp.total_nodes == 0
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_lifecycle_consolidate(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/lifecycle/consolidate").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "records_processed": 10,
                        "records_migrated": 2,
                        "records_compressed": 3,
                        "records_pruned": 1,
                        "semantic_nodes_created": 4,
                    },
                )
            )
            resp = await async_high_level_client.lifecycle_consolidate(
                ConsolidateRequest()
            )
            assert resp.records_processed == 10
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_lifecycle_decay_tick(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/lifecycle/decay-tick").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "records_updated": 5,
                        "records_below_threshold": 1,
                        "records_pruned": 0,
                    },
                )
            )
            resp = await async_high_level_client.lifecycle_decay_tick(
                DecayTickRequest(tick_duration_seconds=3600)
            )
            assert resp.records_updated == 5
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_lifecycle_set_mode(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.put("/v1/lifecycle/mode").mock(
                return_value=httpx.Response(204)
            )
            await async_high_level_client.lifecycle_set_mode(
                SetModeRequest(mode=RecallMode.PERFECT)
            )
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_lifecycle_forget(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.delete("/v1/lifecycle/forget").mock(
                return_value=httpx.Response(200, json={"records_deleted": 2})
            )
            resp = await async_high_level_client.lifecycle_forget(
                ForgetRequest(
                    record_ids=[SAMPLE_RECORD_ID], confirm=True
                )
            )
            assert resp.records_deleted == 2
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_introspect_record(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get(f"/v1/introspect/record/{SAMPLE_RECORD_ID}").mock(
                return_value=httpx.Response(200, json=make_memory_record_json())
            )
            record = await async_high_level_client.introspect_record(SAMPLE_RECORD_ID)
            assert record.id == SAMPLE_RECORD_ID
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_introspect_evolution(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get("/v1/introspect/evolution").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "parameter_adjustments": [],
                        "detected_patterns": [],
                        "schema_adaptations": [],
                    },
                )
            )
            metrics = await async_high_level_client.introspect_evolution()
            assert metrics.parameter_adjustments == []
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_introspect_decay_forecast(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.post("/v1/introspect/decay-forecast").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "forecasts": [
                            {
                                "record_id": str(SAMPLE_RECORD_ID),
                                "current_fidelity": 0.9,
                                "forecasted_fidelity": 0.6,
                            }
                        ]
                    },
                )
            )
            resp = await async_high_level_client.introspect_decay_forecast(
                DecayForecastRequest(
                    record_ids=[SAMPLE_RECORD_ID],
                    forecast_at="2026-04-01T00:00:00Z",
                )
            )
            assert len(resp.forecasts) == 1
        await async_high_level_client.close()


class TestAsyncHealthEndpoints:
    """Test health/readiness via async client."""

    @pytest.mark.asyncio
    async def test_health(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get("/health").mock(
                return_value=httpx.Response(200, json={"status": "ok"})
            )
            resp = await async_high_level_client.health()
            assert resp.status == "ok"
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_readiness(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get("/readiness").mock(
                return_value=httpx.Response(200, json={"status": "ready"})
            )
            resp = await async_high_level_client.readiness()
            assert resp.status == "ready"
        await async_high_level_client.close()


class TestAsyncErrorHandling:
    """Test error handling in async client."""

    @pytest.mark.asyncio
    async def test_record_not_found(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get(f"/v1/introspect/record/{SAMPLE_RECORD_ID}").mock(
                return_value=httpx.Response(
                    404, json=make_cmp_error("RECORD_NOT_FOUND", "Not found")
                )
            )
            with pytest.raises(RecordNotFoundError) as exc_info:
                await async_high_level_client.get_record(SAMPLE_RECORD_ID)
            assert exc_info.value.status_code == 404
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_unauthorized(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get("/v1/introspect/stats").mock(
                return_value=httpx.Response(
                    401, json=make_cmp_error("UNAUTHORIZED", "Bad token")
                )
            )
            with pytest.raises(UnauthorizedError):
                await async_high_level_client.stats()
        await async_high_level_client.close()

    @pytest.mark.asyncio
    async def test_forget_unconfirmed(self, async_high_level_client):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.delete("/v1/lifecycle/forget").mock(
                return_value=httpx.Response(
                    400,
                    json=make_cmp_error("FORGET_UNCONFIRMED", "Confirm required"),
                )
            )
            with pytest.raises(ForgetUnconfirmedError):
                await async_high_level_client.forget(
                    SAMPLE_RECORD_ID, confirm=False
                )
        await async_high_level_client.close()


class TestAsyncContextManager:
    """Test that AsyncClient can be used as an async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        with respx.mock(base_url=BASE_URL) as mock_api:
            mock_api.get("/health").mock(
                return_value=httpx.Response(200, json={"status": "ok"})
            )
            async with AsyncClient(
                BASE_URL, api_key="key", max_retries=0
            ) as client:
                resp = await client.health()
                assert resp.status == "ok"


class TestAsyncRetryBehavior:
    """Test retry logic in async transport."""

    @pytest.mark.asyncio
    async def test_retry_disabled_by_default(self):
        with respx.mock(base_url=BASE_URL) as mock_api:
            route = mock_api.get("/v1/introspect/stats")
            route.mock(
                return_value=httpx.Response(
                    503, json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy")
                )
            )

            async with AsyncClient(BASE_URL, api_key="key") as client:
                with pytest.raises(CerememoryError):
                    await client.stats()
            assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_503(self):
        with respx.mock(base_url=BASE_URL) as mock_api:
            route = mock_api.get("/v1/introspect/stats")
            route.side_effect = [
                httpx.Response(
                    503,
                    json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy", retry_after=0),
                ),
                httpx.Response(200, json=make_stats_response()),
            ]

            async with AsyncClient(
                BASE_URL, api_key="key", max_retries=1
            ) as client:
                stats = await client.stats()
                assert stats.total_records == 42
            assert route.call_count == 2


    @pytest.mark.asyncio
    async def test_no_retry_on_mutating_post_by_default(self):
        with respx.mock(base_url=BASE_URL) as mock_api:
            route = mock_api.post("/v1/encode")
            route.side_effect = [
                httpx.Response(
                    503,
                    json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy", retry_after=0),
                ),
                httpx.Response(200, json=make_encode_store_response()),
            ]

            async with AsyncClient(BASE_URL, api_key="key", max_retries=1) as client:
                req = EncodeStoreRequest(
                    content=MemoryContent(
                        blocks=[
                            ContentBlock(
                                modality=Modality.TEXT, format="text/plain", data=b"x"
                            )
                        ]
                    )
                )
                with pytest.raises(CerememoryError):
                    await client.encode_store(req)
            assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_mutating_post_when_opted_in(self):
        with respx.mock(base_url=BASE_URL) as mock_api:
            route = mock_api.post("/v1/encode")
            route.side_effect = [
                httpx.Response(
                    503,
                    json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy", retry_after=0),
                ),
                httpx.Response(200, json=make_encode_store_response()),
            ]

            async with AsyncClient(
                BASE_URL,
                api_key="key",
                max_retries=1,
                retry_mutating_requests=True,
            ) as client:
                req = EncodeStoreRequest(
                    content=MemoryContent(
                        blocks=[
                            ContentBlock(
                                modality=Modality.TEXT, format="text/plain", data=b"x"
                            )
                        ]
                    )
                )
                resp = await client.encode_store(req)
                assert resp.record_id == SAMPLE_RECORD_ID
            assert route.call_count == 2
