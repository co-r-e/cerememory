"""Tests for the synchronous Client (high-level and protocol-level)."""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest

from cerememory import (
    CerememoryError,
    Client,
    ConsolidateRequest,
    ContentBlock,
    DecayForecastRequest,
    DecayTickRequest,
    EmotionVector,
    EncodeBatchRequest,
    EncodeStoreRequest,
    EncodeUpdateRequest,
    ForgetUnconfirmedError,
    MemoryContent,
    Modality,
    RateLimitedError,
    RecallCue,
    RecallMode,
    RecallQueryRequest,
    RecordNotFoundError,
    SetModeRequest,
    StoreType,
    UnauthorizedError,
    ValidationError,
)
from tests.conftest import (
    BASE_URL,
    NOW_ISO,
    SAMPLE_RECORD_ID,
    make_cmp_error,
    make_encode_store_response,
    make_memory_record_json,
    make_recall_query_response,
    make_stats_response,
)


class TestHighLevelStore:
    """Test Client.store() convenience method."""

    def test_store_text(self, high_level_client, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(
                200, json=make_encode_store_response()
            )
        )
        record_id = high_level_client.store("Hello, world!", store="episodic")
        assert record_id == SAMPLE_RECORD_ID

    def test_store_with_store_enum(self, high_level_client, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(
                200, json=make_encode_store_response(store="semantic")
            )
        )
        record_id = high_level_client.store("fact", store=StoreType.SEMANTIC)
        assert isinstance(record_id, UUID)

    def test_store_with_emotion(self, high_level_client, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )
        record_id = high_level_client.store(
            "happy memory", emotion=EmotionVector(joy=0.9, valence=0.8)
        )
        assert isinstance(record_id, UUID)

        # Verify the request body included emotion
        request = mock_api.calls.last.request
        import json

        body = json.loads(request.content)
        assert body["emotion"]["joy"] == 0.9

    def test_store_with_metadata(self, high_level_client, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )
        high_level_client.store("Session note", metadata={"source": "chat"})

        import json

        request = mock_api.calls.last.request
        body = json.loads(request.content)
        assert body["metadata"]["source"] == "chat"


class TestHighLevelRecall:
    """Test Client.recall() convenience method."""

    def test_recall_text(self, high_level_client, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(200, json=make_recall_query_response())
        )
        memories = high_level_client.recall("hello")
        assert len(memories) == 1
        assert memories[0].relevance_score == 0.95

    def test_recall_with_limit(self, high_level_client, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(200, json=make_recall_query_response())
        )
        high_level_client.recall("hello", limit=5)

        import json

        request = mock_api.calls.last.request
        body = json.loads(request.content)
        assert body["limit"] == 5

    def test_recall_with_mode(self, high_level_client, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(200, json=make_recall_query_response())
        )
        high_level_client.recall("hello", mode="perfect")

        import json

        request = mock_api.calls.last.request
        body = json.loads(request.content)
        assert body["recall_mode"] == "perfect"

    def test_recall_with_activation_options(self, high_level_client, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(200, json=make_recall_query_response())
        )
        high_level_client.recall(
            "hello", reconsolidate=False, activation_depth=5
        )

        import json

        request = mock_api.calls.last.request
        body = json.loads(request.content)
        assert body["reconsolidate"] is False
        assert body["activation_depth"] == 5


class TestHighLevelForget:
    """Test Client.forget() convenience method."""

    def test_forget_by_id(self, high_level_client, mock_api):
        mock_api.delete("/v1/lifecycle/forget").mock(
            return_value=httpx.Response(200, json={"records_deleted": 1})
        )
        count = high_level_client.forget(SAMPLE_RECORD_ID, confirm=True)
        assert count == 1

    def test_forget_requires_confirm(self, high_level_client, mock_api):
        mock_api.delete("/v1/lifecycle/forget").mock(
            return_value=httpx.Response(
                400, json=make_cmp_error("FORGET_UNCONFIRMED", "Confirm required")
            )
        )
        with pytest.raises(ForgetUnconfirmedError):
            high_level_client.forget(SAMPLE_RECORD_ID, confirm=False)


class TestHighLevelStats:
    """Test Client.stats() convenience method."""

    def test_stats(self, high_level_client, mock_api):
        mock_api.get("/v1/introspect/stats").mock(
            return_value=httpx.Response(200, json=make_stats_response())
        )
        stats = high_level_client.stats()
        assert stats.total_records == 42
        assert stats.avg_fidelity == 0.85


class TestHighLevelRawAndDream:
    def test_store_raw(self, high_level_client, mock_api):
        mock_api.post("/v1/encode/raw").mock(
            return_value=httpx.Response(
                200,
                json={
                    "record_id": str(SAMPLE_RECORD_ID),
                    "session_id": "sess-1",
                    "visibility": "normal",
                    "secrecy_level": "public",
                },
            )
        )
        record_id = high_level_client.store_raw("Raw note", session_id="sess-1")
        assert record_id == SAMPLE_RECORD_ID

    def test_recall_raw(self, high_level_client, mock_api):
        mock_api.post("/v1/recall/raw").mock(
            return_value=httpx.Response(
                200,
                json={
                    "records": [
                        {
                            "id": str(SAMPLE_RECORD_ID),
                            "session_id": "sess-1",
                            "turn_id": None,
                            "topic_id": None,
                            "source": "conversation",
                            "speaker": "user",
                            "visibility": "normal",
                            "secrecy_level": "public",
                            "created_at": NOW_ISO,
                            "updated_at": NOW_ISO,
                            "content": {
                                "blocks": [
                                    {
                                        "modality": "text",
                                        "format": "text/plain",
                                        "data": list(b"Raw note"),
                                        "embedding": None,
                                    }
                                ],
                                "summary": None,
                            },
                            "metadata": {},
                            "derived_memory_ids": [],
                            "suppressed": False,
                        }
                    ],
                    "total_candidates": 1,
                },
            )
        )
        records = high_level_client.recall_raw(session_id="sess-1")
        assert len(records) == 1
        assert records[0].session_id == "sess-1"

    def test_dream_tick(self, high_level_client, mock_api):
        mock_api.post("/v1/lifecycle/dream-tick").mock(
            return_value=httpx.Response(
                200,
                json={
                    "groups_processed": 1,
                    "raw_records_processed": 2,
                    "episodic_summaries_created": 1,
                    "semantic_nodes_created": 1,
                },
            )
        )
        resp = high_level_client.dream_tick(session_id="sess-1")
        assert resp.groups_processed == 1
        assert resp.semantic_nodes_created == 1


class TestHighLevelGetRecord:
    """Test Client.get_record() convenience method."""

    def test_get_record(self, high_level_client, mock_api):
        mock_api.get(f"/v1/introspect/record/{SAMPLE_RECORD_ID}").mock(
            return_value=httpx.Response(200, json=make_memory_record_json())
        )
        record = high_level_client.get_record(SAMPLE_RECORD_ID)
        assert record.id == SAMPLE_RECORD_ID
        assert record.store == StoreType.EPISODIC


class TestProtocolMethods:
    """Test full CMP protocol method delegation."""

    def test_encode_store(self, high_level_client, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )
        req = EncodeStoreRequest(
            content=MemoryContent(
                blocks=[
                    ContentBlock(modality=Modality.TEXT, format="text/plain", data=b"hi")
                ]
            ),
            store=StoreType.EPISODIC,
        )
        resp = high_level_client.encode_store(req)
        assert resp.record_id == SAMPLE_RECORD_ID

    def test_encode_batch(self, high_level_client, mock_api):
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
                                data=b"batch item",
                            )
                        ]
                    )
                )
            ]
        )
        resp = high_level_client.encode_batch(req)
        assert len(resp.results) == 1

    def test_encode_update(self, high_level_client, mock_api):
        mock_api.patch(f"/v1/encode/{SAMPLE_RECORD_ID}").mock(
            return_value=httpx.Response(204)
        )
        req = EncodeUpdateRequest(
            record_id=SAMPLE_RECORD_ID,
            emotion=EmotionVector(joy=0.5),
        )
        # Should not raise
        high_level_client.encode_update(SAMPLE_RECORD_ID, req)

    def test_recall_query(self, high_level_client, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(200, json=make_recall_query_response())
        )
        req = RecallQueryRequest(cue=RecallCue(text="test"))
        resp = high_level_client.recall_query(req)
        assert resp.total_candidates == 1

    def test_lifecycle_consolidate(self, high_level_client, mock_api):
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
        resp = high_level_client.lifecycle_consolidate(ConsolidateRequest())
        assert resp.records_processed == 10

    def test_lifecycle_decay_tick(self, high_level_client, mock_api):
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
        resp = high_level_client.lifecycle_decay_tick(
            DecayTickRequest(tick_duration_seconds=3600)
        )
        assert resp.records_updated == 5

    def test_lifecycle_set_mode(self, high_level_client, mock_api):
        mock_api.put("/v1/lifecycle/mode").mock(
            return_value=httpx.Response(204)
        )
        high_level_client.lifecycle_set_mode(
            SetModeRequest(mode=RecallMode.PERFECT)
        )

    def test_introspect_evolution(self, high_level_client, mock_api):
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
        metrics = high_level_client.introspect_evolution()
        assert metrics.parameter_adjustments == []

    def test_introspect_decay_forecast(self, high_level_client, mock_api):
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
        resp = high_level_client.introspect_decay_forecast(
            DecayForecastRequest(
                record_ids=[SAMPLE_RECORD_ID],
                forecast_at="2026-04-01T00:00:00Z",
            )
        )
        assert len(resp.forecasts) == 1


class TestHealthEndpoints:
    """Test health/readiness via the high-level client."""

    def test_health(self, high_level_client, mock_api):
        mock_api.get("/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        resp = high_level_client.health()
        assert resp.status == "ok"

    def test_readiness(self, high_level_client, mock_api):
        mock_api.get("/readiness").mock(
            return_value=httpx.Response(200, json={"status": "ready"})
        )
        resp = high_level_client.readiness()
        assert resp.status == "ready"


class TestErrorHandling:
    """Test that CMP errors are properly mapped to Python exceptions."""

    def test_record_not_found(self, high_level_client, mock_api):
        mock_api.get(f"/v1/introspect/record/{SAMPLE_RECORD_ID}").mock(
            return_value=httpx.Response(
                404, json=make_cmp_error("RECORD_NOT_FOUND", "Not found")
            )
        )
        with pytest.raises(RecordNotFoundError) as exc_info:
            high_level_client.get_record(SAMPLE_RECORD_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.code == "RECORD_NOT_FOUND"

    def test_validation_error(self, high_level_client, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(
                400, json=make_cmp_error("VALIDATION_ERROR", "Bad input")
            )
        )
        req = EncodeStoreRequest(
            content=MemoryContent(
                blocks=[
                    ContentBlock(modality=Modality.TEXT, format="text/plain", data=b"x")
                ]
            )
        )
        with pytest.raises(ValidationError):
            high_level_client.encode_store(req)

    def test_unauthorized(self, high_level_client, mock_api):
        mock_api.get("/v1/introspect/stats").mock(
            return_value=httpx.Response(
                401, json=make_cmp_error("UNAUTHORIZED", "Bad token")
            )
        )
        with pytest.raises(UnauthorizedError) as exc_info:
            high_level_client.stats()
        assert exc_info.value.status_code == 401

    def test_rate_limited_with_retry_after(self, high_level_client, mock_api):
        mock_api.get("/v1/introspect/stats").mock(
            return_value=httpx.Response(
                429,
                json=make_cmp_error("RATE_LIMITED", "Too fast", retry_after=5),
            )
        )
        with pytest.raises(RateLimitedError) as exc_info:
            high_level_client.stats()
        assert exc_info.value.retry_after == 5

    def test_error_preserves_request_id(self, high_level_client, mock_api):
        mock_api.get("/v1/introspect/stats").mock(
            return_value=httpx.Response(
                401,
                json=make_cmp_error(
                    "UNAUTHORIZED", "Missing key", request_id="req-123"
                ),
            )
        )
        with pytest.raises(UnauthorizedError) as exc_info:
            high_level_client.stats()
        assert exc_info.value.request_id == "req-123"

    def test_unknown_error_code(self, high_level_client, mock_api):
        mock_api.get("/v1/introspect/stats").mock(
            return_value=httpx.Response(
                500,
                json={"code": "TOTALLY_NEW_ERROR", "message": "Something new"},
            )
        )
        with pytest.raises(CerememoryError) as exc_info:
            high_level_client.stats()
        assert exc_info.value.code == "TOTALLY_NEW_ERROR"

    def test_non_json_error_body(self, high_level_client, mock_api):
        mock_api.get("/v1/introspect/stats").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        with pytest.raises(CerememoryError) as exc_info:
            high_level_client.stats()
        assert exc_info.value.status_code == 500


class TestContextManager:
    """Test that the Client can be used as a context manager."""

    def test_sync_context_manager(self, mock_api):
        mock_api.get("/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        with Client(BASE_URL, api_key="key", max_retries=0) as client:
            resp = client.health()
            assert resp.status == "ok"


class TestRetryBehavior:
    """Test retry logic in the transport layer."""

    def test_retry_disabled_by_default(self, mock_api):
        route = mock_api.get("/v1/introspect/stats")
        route.mock(
            return_value=httpx.Response(
                503, json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy")
            )
        )

        client = Client(BASE_URL, api_key="key")
        try:
            with pytest.raises(CerememoryError):
                client.stats()
            assert route.call_count == 1
        finally:
            client.close()

    def test_retry_on_503(self, mock_api):
        route = mock_api.get("/v1/introspect/stats")
        route.side_effect = [
            httpx.Response(
                503,
                json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy", retry_after=0),
            ),
            httpx.Response(200, json=make_stats_response()),
        ]

        # max_retries=1 means it tries once, then retries once
        client = Client(BASE_URL, api_key="key", max_retries=1)
        try:
            stats = client.stats()
            assert stats.total_records == 42
        finally:
            client.close()

    def test_no_retry_on_400(self, mock_api):
        route = mock_api.post("/v1/encode")
        route.mock(
            return_value=httpx.Response(
                400, json=make_cmp_error("VALIDATION_ERROR", "Bad")
            )
        )

        client = Client(BASE_URL, api_key="key", max_retries=3)
        try:
            req = EncodeStoreRequest(
                content=MemoryContent(
                    blocks=[
                        ContentBlock(
                            modality=Modality.TEXT, format="text/plain", data=b"x"
                        )
                    ]
                )
            )
            with pytest.raises(ValidationError):
                client.encode_store(req)
            # Should only have been called once (no retries for 400)
            assert route.call_count == 1
        finally:
            client.close()

    def test_no_retry_on_mutating_post_by_default(self, mock_api):
        route = mock_api.post("/v1/encode")
        route.side_effect = [
            httpx.Response(
                503,
                json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy", retry_after=0),
            ),
            httpx.Response(200, json=make_encode_store_response()),
        ]

        client = Client(BASE_URL, api_key="key", max_retries=1)
        try:
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
                client.encode_store(req)
            assert route.call_count == 1
        finally:
            client.close()

    def test_retry_on_mutating_post_when_opted_in(self, mock_api):
        route = mock_api.post("/v1/encode")
        route.side_effect = [
            httpx.Response(
                503,
                json=make_cmp_error("DECAY_ENGINE_BUSY", "Busy", retry_after=0),
            ),
            httpx.Response(200, json=make_encode_store_response()),
        ]

        client = Client(
            BASE_URL,
            api_key="key",
            max_retries=1,
            retry_mutating_requests=True,
        )
        try:
            req = EncodeStoreRequest(
                content=MemoryContent(
                    blocks=[
                        ContentBlock(
                            modality=Modality.TEXT, format="text/plain", data=b"x"
                        )
                    ]
                )
            )
            resp = client.encode_store(req)
            assert resp.record_id == SAMPLE_RECORD_ID
            assert route.call_count == 2
        finally:
            client.close()
