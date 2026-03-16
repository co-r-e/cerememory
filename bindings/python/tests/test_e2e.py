"""End-to-end tests against a running Cerememory server.

These tests require a live server at CEREMEMORY_URL (default: http://localhost:8420).
They are skipped by default in unit test runs. Run with:

    CEREMEMORY_E2E=1 pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pytest

from cerememory import AsyncClient, Client, RecordNotFoundError

pytestmark = pytest.mark.skipif(
    os.environ.get("CEREMEMORY_E2E") != "1",
    reason="Set CEREMEMORY_E2E=1 to run E2E tests",
)

T = TypeVar("T")


def wait_for(
    callback: Callable[[], T | None],
    *,
    timeout: float = 10.0,
    interval: float = 0.25,
) -> T:
    """Poll until the callback returns a truthy value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = callback()
        if result:
            return result
        time.sleep(interval)
    raise AssertionError("Timed out waiting for E2E condition")


async def wait_for_async(
    callback: Callable[[], Awaitable[T | None]],
    *,
    timeout: float = 10.0,
    interval: float = 0.25,
) -> T:
    """Async variant of wait_for()."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await callback()
        if result:
            return result
        await asyncio.sleep(interval)
    raise AssertionError("Timed out waiting for async E2E condition")


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("CEREMEMORY_URL", "http://localhost:8420")


@pytest.fixture
def client(base_url: str):
    with Client(base_url) as sdk_client:
        yield sdk_client


class TestEpisodicCRUD:
    """Store → recall → forget round-trip for the high-level Python client."""

    def test_store_and_recall(self, client: Client) -> None:
        tag = str(uuid.uuid4())
        record_id = client.store(f"E2E episodic memory {tag}", store="episodic")

        record = client.get_record(record_id)
        assert record.id == record_id

        recalled = wait_for(
            lambda: next(
                (
                    memory
                    for memory in client.recall(tag, stores=["episodic"])
                    if memory.record.id == record_id
                ),
                None,
            )
        )
        assert recalled.record.id == record_id

        deleted = client.forget(record_id, confirm=True)
        assert deleted >= 1

        with pytest.raises(RecordNotFoundError):
            client.get_record(record_id)

    def test_store_and_recall_semantic(self, client: Client) -> None:
        tag = str(uuid.uuid4())
        record_id = client.store(f"E2E semantic memory {tag}", store="semantic")

        recalled = wait_for(
            lambda: next(
                (
                    memory
                    for memory in client.recall(tag, stores=["semantic"])
                    if memory.record.id == record_id
                ),
                None,
            )
        )
        assert recalled.record.store.value == "semantic"

        deleted = client.forget(record_id, confirm=True)
        assert deleted >= 1


@pytest.mark.asyncio
class TestAsyncCRUD:
    """Async variant of the CRUD round-trip."""

    async def test_async_store_and_recall(self, base_url: str) -> None:
        tag = str(uuid.uuid4())

        async with AsyncClient(base_url) as client:
            record_id = await client.store(f"Async E2E memory {tag}", store="episodic")

            record = await client.get_record(record_id)
            assert record.id == record_id

            async def recall_once():
                return next(
                    (
                        memory
                        for memory in await client.recall(tag, stores=["episodic"])
                        if memory.record.id == record_id
                    ),
                    None,
                )

            recalled = await wait_for_async(recall_once)
            assert recalled.record.id == record_id

            deleted = await client.forget(record_id, confirm=True)
            assert deleted >= 1

            with pytest.raises(RecordNotFoundError):
                await client.get_record(record_id)


class TestHealthcheck:
    """Verify the server is reachable."""

    def test_health(self, client: Client) -> None:
        health = client.health()
        assert health.status == "ok"
