"""Shared fixtures for Cerememory SDK tests."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
import respx

from cerememory import Client
from cerememory._async_client import AsyncCerememoryClient
from cerememory._sync_client import SyncCerememoryClient

BASE_URL = "http://testserver:8420"
API_KEY = "test-api-key-123"


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

SAMPLE_RECORD_ID = UUID("01916e3a-1234-7000-8000-000000000001")
SAMPLE_RECORD_ID_2 = UUID("01916e3a-5678-7000-8000-000000000002")
NOW_ISO = "2026-03-15T00:00:00Z"


def make_encode_store_response(
    record_id: UUID = SAMPLE_RECORD_ID,
    store: str = "episodic",
    initial_fidelity: float = 1.0,
    associations_created: int = 0,
) -> dict[str, Any]:
    return {
        "record_id": str(record_id),
        "store": store,
        "initial_fidelity": initial_fidelity,
        "associations_created": associations_created,
    }


def make_memory_record_json(
    record_id: UUID = SAMPLE_RECORD_ID,
    store: str = "episodic",
    text: str = "Hello, world!",
) -> dict[str, Any]:
    return {
        "id": str(record_id),
        "store": store,
        "created_at": NOW_ISO,
        "updated_at": NOW_ISO,
        "last_accessed_at": NOW_ISO,
        "access_count": 0,
        "content": {
            "blocks": [
                {
                    "modality": "text",
                    "format": "text/plain",
                    "data": list(text.encode("utf-8")),
                    "embedding": None,
                }
            ],
            "summary": None,
        },
        "fidelity": {
            "score": 1.0,
            "noise_level": 0.0,
            "decay_rate": 0.3,
            "emotional_anchor": 1.0,
            "reinforcement_count": 0,
            "stability": 1.0,
            "last_decay_tick": NOW_ISO,
        },
        "emotion": {
            "joy": 0.0,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.0,
            "intensity": 0.0,
            "valence": 0.0,
        },
        "associations": [],
        "metadata": {},
        "version": 1,
    }


def make_recall_query_response(
    record_id: UUID = SAMPLE_RECORD_ID,
    text: str = "Hello, world!",
    total_candidates: int = 1,
) -> dict[str, Any]:
    record = make_memory_record_json(record_id=record_id, text=text)
    return {
        "memories": [
            {
                "record": record,
                "relevance_score": 0.95,
                "activation_path": None,
                "rendered_content": record["content"],
            }
        ],
        "activation_trace": None,
        "total_candidates": total_candidates,
    }


def make_stats_response() -> dict[str, Any]:
    return {
        "total_records": 42,
        "records_by_store": {"episodic": 30, "semantic": 12},
        "total_associations": 10,
        "avg_fidelity": 0.85,
        "avg_fidelity_by_store": {"episodic": 0.9, "semantic": 0.75},
        "oldest_record": NOW_ISO,
        "newest_record": NOW_ISO,
        "total_recall_count": 100,
        "evolution_metrics": None,
        "background_decay_enabled": False,
    }


def make_cmp_error(
    code: str = "RECORD_NOT_FOUND",
    message: str = "Record not found",
    details: Any = None,
    retry_after: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        result["details"] = details
    if retry_after is not None:
        result["retry_after"] = retry_after
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api():
    """respx mock router for synchronous httpx transport."""
    with respx.mock(base_url=BASE_URL, assert_all_mocked=False) as router:
        yield router


@pytest.fixture
def sync_client(mock_api):
    """SyncCerememoryClient with mocked transport."""
    client = SyncCerememoryClient(BASE_URL, api_key=API_KEY, max_retries=0)
    yield client
    client.close()


@pytest.fixture
def high_level_client(mock_api):
    """High-level Client with mocked transport."""
    client = Client(BASE_URL, api_key=API_KEY, max_retries=0)
    yield client
    client.close()


@pytest.fixture
def async_client():
    """AsyncCerememoryClient with mocked transport (use inside respx context)."""
    client = AsyncCerememoryClient(BASE_URL, api_key=API_KEY, max_retries=0)
    return client
