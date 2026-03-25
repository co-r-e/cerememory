"""Shared endpoint utilities for sync and async Cerememory clients.

Provides common request serialization helpers and endpoint path constants
to avoid duplication between _sync_client.py and _async_client.py.
"""

from __future__ import annotations

from typing import Any


def dump(request: Any) -> dict[str, Any]:
    """Serialize a Pydantic model for JSON transport."""
    return request.model_dump(mode="json", exclude_none=True)


def dump_without_id(request: Any) -> dict[str, Any]:
    """Serialize a Pydantic model, stripping ``record_id`` for path-param endpoints."""
    body = dump(request)
    body.pop("record_id", None)
    return body
