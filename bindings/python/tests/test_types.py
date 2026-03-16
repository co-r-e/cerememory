"""Tests for Pydantic models in cerememory.types."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from cerememory.types import (
    Association,
    AssociationType,
    ConsolidateRequest,
    ConsolidationStrategy,
    ContentBlock,
    DecayForecastResponse,
    DecayTickRequest,
    EmotionVector,
    EncodeStoreRequest,
    EncodeStoreResponse,
    EvolutionMetrics,
    FidelityState,
    ForgetRequest,
    ForgetResponse,
    HealthResponse,
    MemoryContent,
    MemoryRecord,
    Modality,
    ReadinessResponse,
    RecallCue,
    RecallGraphResponse,
    RecallMode,
    RecallQueryRequest,
    SetModeRequest,
    StatsResponse,
    StoreType,
    TimeGranularity,
)


class TestEnums:
    """Test that enum values match the Rust serde serialization."""

    def test_store_type_values(self):
        assert StoreType.EPISODIC.value == "episodic"
        assert StoreType.SEMANTIC.value == "semantic"
        assert StoreType.PROCEDURAL.value == "procedural"
        assert StoreType.EMOTIONAL.value == "emotional"
        assert StoreType.WORKING.value == "working"

    def test_modality_values(self):
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.INTEROCEPTIVE.value == "interoceptive"

    def test_association_type_values(self):
        assert AssociationType.TEMPORAL.value == "temporal"
        assert AssociationType.CROSS_MODAL.value == "cross_modal"
        assert AssociationType.USER_DEFINED.value == "user_defined"

    def test_recall_mode_values(self):
        assert RecallMode.HUMAN.value == "human"
        assert RecallMode.PERFECT.value == "perfect"

    def test_time_granularity_values(self):
        assert TimeGranularity.MINUTE.value == "minute"
        assert TimeGranularity.MONTH.value == "month"

    def test_consolidation_strategy_values(self):
        assert ConsolidationStrategy.FULL.value == "full"
        assert ConsolidationStrategy.INCREMENTAL.value == "incremental"
        assert ConsolidationStrategy.SELECTIVE.value == "selective"

    def test_store_type_from_string(self):
        assert StoreType("episodic") == StoreType.EPISODIC

    def test_store_type_invalid_raises(self):
        with pytest.raises(ValueError):
            StoreType("nonexistent")


class TestContentBlock:
    """Test ContentBlock data serialization/deserialization."""

    def test_data_from_bytes(self):
        block = ContentBlock(modality=Modality.TEXT, format="text/plain", data=b"hello")
        assert block.data == b"hello"

    def test_data_from_int_list(self):
        block = ContentBlock(
            modality=Modality.TEXT, format="text/plain", data=[72, 101, 108, 108, 111]
        )
        assert block.data == b"Hello"

    def test_data_from_base64_string(self):
        encoded = base64.b64encode(b"Hello").decode()
        block = ContentBlock(modality=Modality.TEXT, format="text/plain", data=encoded)
        assert block.data == b"Hello"

    def test_data_serializes_to_int_list(self):
        block = ContentBlock(modality=Modality.TEXT, format="text/plain", data=b"Hi")
        dumped = block.model_dump(mode="json")
        assert dumped["data"] == [72, 105]

    def test_embedding_optional(self):
        block = ContentBlock(modality=Modality.TEXT, format="text/plain", data=b"x")
        assert block.embedding is None

    def test_embedding_present(self):
        block = ContentBlock(
            modality=Modality.TEXT,
            format="text/plain",
            data=b"x",
            embedding=[0.1, 0.2, 0.3],
        )
        assert block.embedding == [0.1, 0.2, 0.3]

    def test_json_roundtrip(self):
        block = ContentBlock(
            modality=Modality.IMAGE, format="image/png", data=b"\x89PNG"
        )
        json_str = block.model_dump_json()
        restored = ContentBlock.model_validate_json(json_str)
        assert restored.data == b"\x89PNG"
        assert restored.modality == Modality.IMAGE


class TestFidelityState:
    """Test FidelityState defaults match Rust defaults."""

    def test_defaults(self):
        f = FidelityState()
        assert f.score == 1.0
        assert f.noise_level == 0.0
        assert f.decay_rate == 0.3
        assert f.emotional_anchor == 1.0
        assert f.reinforcement_count == 0
        assert f.stability == 1.0

    def test_json_roundtrip(self):
        f = FidelityState(score=0.5, noise_level=0.1)
        json_str = f.model_dump_json()
        restored = FidelityState.model_validate_json(json_str)
        assert restored.score == 0.5
        assert restored.noise_level == 0.1


class TestEmotionVector:
    """Test EmotionVector defaults."""

    def test_defaults_are_neutral(self):
        e = EmotionVector()
        assert e.joy == 0.0
        assert e.intensity == 0.0
        assert e.valence == 0.0

    def test_custom_values(self):
        e = EmotionVector(joy=0.8, valence=0.5, intensity=0.7)
        assert e.joy == 0.8
        assert e.valence == 0.5


class TestMemoryRecord:
    """Test MemoryRecord serialization."""

    def test_from_json(self):
        now = datetime.now(timezone.utc).isoformat()
        rid = str(uuid4())
        data = {
            "id": rid,
            "store": "episodic",
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": now,
            "access_count": 5,
            "content": {
                "blocks": [
                    {
                        "modality": "text",
                        "format": "text/plain",
                        "data": list(b"Hello world"),
                    }
                ],
                "summary": "A greeting",
            },
            "fidelity": {
                "score": 0.9,
                "noise_level": 0.05,
                "decay_rate": 0.3,
                "emotional_anchor": 1.0,
                "reinforcement_count": 2,
                "stability": 1.5,
                "last_decay_tick": now,
            },
            "emotion": {"joy": 0.5, "valence": 0.3},
            "associations": [],
            "metadata": {"source": "test"},
            "version": 1,
        }
        record = MemoryRecord.model_validate(data)
        assert record.id == UUID(rid)
        assert record.store == StoreType.EPISODIC
        assert record.content.blocks[0].data == b"Hello world"
        assert record.fidelity.score == 0.9
        assert record.emotion.joy == 0.5

    def test_json_roundtrip(self):
        now = datetime.now(timezone.utc).isoformat()
        rid = str(uuid4())
        data = {
            "id": rid,
            "store": "semantic",
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": now,
            "content": {
                "blocks": [
                    {"modality": "text", "format": "text/plain", "data": [65, 66]}
                ],
            },
            "version": 2,
        }
        record = MemoryRecord.model_validate(data)
        json_str = record.model_dump_json()
        restored = MemoryRecord.model_validate_json(json_str)
        assert restored.id == record.id
        assert restored.store == StoreType.SEMANTIC
        assert restored.version == 2


class TestAssociation:
    """Test Association model."""

    def test_from_json(self):
        now = datetime.now(timezone.utc).isoformat()
        tid = str(uuid4())
        data = {
            "target_id": tid,
            "association_type": "cross_modal",
            "weight": 0.75,
            "created_at": now,
            "last_co_activation": now,
        }
        a = Association.model_validate(data)
        assert a.target_id == UUID(tid)
        assert a.association_type == AssociationType.CROSS_MODAL
        assert a.weight == 0.75


class TestEncodeRequests:
    """Test encode request/response models."""

    def test_encode_store_request_minimal(self):
        req = EncodeStoreRequest(
            content=MemoryContent(
                blocks=[
                    ContentBlock(modality=Modality.TEXT, format="text/plain", data=b"Hi")
                ]
            )
        )
        dumped = req.model_dump(mode="json", exclude_none=True)
        assert "store" not in dumped
        assert dumped["content"]["blocks"][0]["data"] == [72, 105]

    def test_encode_store_request_full(self):
        req = EncodeStoreRequest(
            content=MemoryContent(
                blocks=[
                    ContentBlock(modality=Modality.TEXT, format="text/plain", data=b"Hi")
                ]
            ),
            store=StoreType.SEMANTIC,
            emotion=EmotionVector(joy=0.5),
        )
        dumped = req.model_dump(mode="json", exclude_none=True)
        assert dumped["store"] == "semantic"
        assert dumped["emotion"]["joy"] == 0.5

    def test_encode_store_response(self):
        data = {
            "record_id": str(uuid4()),
            "store": "episodic",
            "initial_fidelity": 1.0,
            "associations_created": 2,
        }
        resp = EncodeStoreResponse.model_validate(data)
        assert resp.store == StoreType.EPISODIC
        assert resp.associations_created == 2


class TestRecallRequests:
    """Test recall request/response models."""

    def test_recall_query_request_defaults(self):
        req = RecallQueryRequest(cue=RecallCue(text="coffee"))
        assert req.limit == 10
        assert req.recall_mode == RecallMode.HUMAN
        assert req.reconsolidate is True
        assert req.activation_depth == 2

    def test_recall_query_request_custom(self):
        req = RecallQueryRequest(
            cue=RecallCue(text="coffee"),
            limit=5,
            recall_mode=RecallMode.PERFECT,
            stores=[StoreType.EPISODIC, StoreType.SEMANTIC],
            min_fidelity=0.5,
        )
        dumped = req.model_dump(mode="json", exclude_none=True)
        assert dumped["limit"] == 5
        assert dumped["recall_mode"] == "perfect"
        assert dumped["stores"] == ["episodic", "semantic"]

    def test_recall_cue_with_embedding(self):
        cue = RecallCue(embedding=[0.1, 0.2, 0.3])
        dumped = cue.model_dump(mode="json", exclude_none=True)
        assert dumped["embedding"] == [0.1, 0.2, 0.3]


class TestLifecycleRequests:
    """Test lifecycle request/response models."""

    def test_consolidate_request_defaults(self):
        req = ConsolidateRequest()
        assert req.strategy == ConsolidationStrategy.INCREMENTAL
        assert req.dry_run is False

    def test_decay_tick_request(self):
        req = DecayTickRequest(tick_duration_seconds=3600)
        dumped = req.model_dump(mode="json", exclude_none=True)
        assert dumped["tick_duration_seconds"] == 3600

    def test_set_mode_request(self):
        req = SetModeRequest(mode=RecallMode.PERFECT)
        dumped = req.model_dump(mode="json", exclude_none=True)
        assert dumped["mode"] == "perfect"

    def test_forget_request(self):
        rid = uuid4()
        req = ForgetRequest(record_ids=[rid], confirm=True, cascade=False)
        dumped = req.model_dump(mode="json", exclude_none=True)
        assert len(dumped["record_ids"]) == 1
        assert dumped["confirm"] is True

    def test_forget_response(self):
        resp = ForgetResponse.model_validate({"records_deleted": 3})
        assert resp.records_deleted == 3


class TestIntrospectResponses:
    """Test introspect response models."""

    def test_stats_response(self):
        data = {
            "total_records": 42,
            "records_by_store": {"episodic": 30, "semantic": 12},
            "total_associations": 10,
            "avg_fidelity": 0.85,
            "avg_fidelity_by_store": {},
            "oldest_record": None,
            "newest_record": None,
            "total_recall_count": 100,
            "background_decay_enabled": True,
        }
        stats = StatsResponse.model_validate(data)
        assert stats.total_records == 42
        assert stats.records_by_store["episodic"] == 30
        assert stats.background_decay_enabled is True

    def test_evolution_metrics(self):
        data = {
            "parameter_adjustments": [
                {
                    "store": "episodic",
                    "parameter": "decay_rate",
                    "original_value": 0.3,
                    "current_value": 0.25,
                    "reason": "frequent recall",
                }
            ],
            "detected_patterns": ["daily_review"],
            "schema_adaptations": [],
        }
        metrics = EvolutionMetrics.model_validate(data)
        assert len(metrics.parameter_adjustments) == 1
        assert metrics.parameter_adjustments[0].store == StoreType.EPISODIC

    def test_decay_forecast_response(self):
        rid = str(uuid4())
        data = {
            "forecasts": [
                {
                    "record_id": rid,
                    "current_fidelity": 0.9,
                    "forecasted_fidelity": 0.6,
                    "estimated_threshold_date": "2026-04-01T00:00:00Z",
                }
            ]
        }
        resp = DecayForecastResponse.model_validate(data)
        assert len(resp.forecasts) == 1
        assert resp.forecasts[0].forecasted_fidelity == 0.6

    def test_graph_response(self):
        id1 = str(uuid4())
        id2 = str(uuid4())
        data = {
            "nodes": [
                {"id": id1, "store": "episodic", "summary": "test", "fidelity": 0.9},
                {"id": id2, "store": "semantic", "fidelity": 0.8},
            ],
            "edges": [
                {
                    "source": id1,
                    "target": id2,
                    "edge_type": "semantic",
                    "weight": 0.7,
                }
            ],
            "total_nodes": 2,
        }
        resp = RecallGraphResponse.model_validate(data)
        assert resp.total_nodes == 2
        assert resp.edges[0].weight == 0.7


class TestHealthModels:
    """Test health/readiness response models."""

    def test_health_response(self):
        resp = HealthResponse.model_validate({"status": "ok"})
        assert resp.status == "ok"

    def test_readiness_response(self):
        resp = ReadinessResponse.model_validate({"status": "ready"})
        assert resp.status == "ready"
