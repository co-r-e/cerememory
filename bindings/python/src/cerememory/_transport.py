"""HTTP transport layer with retry logic and CMP error mapping.

This module is internal. External callers should use :class:`cerememory.Client`
or :class:`cerememory.AsyncClient` instead.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Type, TypeVar

import httpx

from cerememory.errors import (
    CerememoryError,
    ConnectionError as CerememoryConnectionError,
    TimeoutError as CerememoryTimeoutError,
    error_from_code,
)

logger = logging.getLogger("cerememory")

T = TypeVar("T")

# HTTP status codes that are eligible for automatic retry.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Default configuration.
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_MAX = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0


def _build_headers(api_key: Optional[str], extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build the default request headers."""
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "cerememory-python/0.1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def _parse_cmp_error(response: httpx.Response) -> CerememoryError:
    """Parse a CMP error envelope from an HTTP error response.

    Falls back to a generic ``CerememoryError`` if the body is not valid
    CMP JSON.
    """
    status = response.status_code
    try:
        body = response.json()
        code = body.get("code", "INTERNAL_ERROR")
        message = body.get("message", response.text)
        details = body.get("details")
        retry_after = body.get("retry_after")
        return error_from_code(
            code,
            message,
            details=details,
            retry_after=retry_after,
            status_code=status,
        )
    except Exception:
        return CerememoryError(
            f"HTTP {status}: {response.text}",
            status_code=status,
        )


def _compute_backoff(attempt: int, retry_after: Optional[int] = None) -> float:
    """Compute wait time in seconds using exponential backoff with jitter.

    If the server sent a ``retry_after`` hint, that takes precedence.
    """
    if retry_after is not None and retry_after > 0:
        return float(retry_after)
    delay = min(
        DEFAULT_BACKOFF_BASE * (DEFAULT_BACKOFF_FACTOR ** attempt),
        DEFAULT_BACKOFF_MAX,
    )
    return delay


# ---------------------------------------------------------------------------
# Synchronous transport
# ---------------------------------------------------------------------------


class SyncTransport:
    """Synchronous HTTP transport with retry and error mapping.

    This is an internal class. Use :class:`cerememory.SyncCerememoryClient`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers = _build_headers(api_key, headers)

        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers=self._default_headers,
                timeout=httpx.Timeout(timeout),
            )
            self._owns_client = True

    def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry logic.

        Raises:
            CerememoryError: On any non-2xx final response or transport error.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                )
            except httpx.ConnectError as exc:
                last_error = CerememoryConnectionError(
                    f"Failed to connect to {self._base_url}: {exc}"
                )
                if attempt < self._max_retries:
                    time.sleep(_compute_backoff(attempt))
                    continue
                raise last_error from exc
            except httpx.TimeoutException as exc:
                last_error = CerememoryTimeoutError(
                    f"Request timed out after {self._timeout}s: {exc}"
                )
                if attempt < self._max_retries:
                    time.sleep(_compute_backoff(attempt))
                    continue
                raise last_error from exc

            if response.status_code < 400:
                return response

            # Parse the error for retry-after hints.
            error = _parse_cmp_error(response)

            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < self._max_retries
            ):
                wait = _compute_backoff(attempt, error.retry_after)
                logger.debug(
                    "Retryable error %d on %s %s (attempt %d/%d), waiting %.1fs",
                    response.status_code,
                    method,
                    path,
                    attempt + 1,
                    self._max_retries + 1,
                    wait,
                )
                time.sleep(wait)
                last_error = error
                continue

            raise error

        # Should not be reached, but satisfy type checker.
        assert last_error is not None
        raise last_error


# ---------------------------------------------------------------------------
# Asynchronous transport
# ---------------------------------------------------------------------------


class AsyncTransport:
    """Asynchronous HTTP transport with retry and error mapping.

    This is an internal class. Use :class:`cerememory.AsyncCerememoryClient`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers = _build_headers(api_key, headers)

        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._default_headers,
                timeout=httpx.Timeout(timeout),
            )
            self._owns_client = True

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry logic.

        Raises:
            CerememoryError: On any non-2xx final response or transport error.
        """
        import asyncio

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                )
            except httpx.ConnectError as exc:
                last_error = CerememoryConnectionError(
                    f"Failed to connect to {self._base_url}: {exc}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                    continue
                raise last_error from exc
            except httpx.TimeoutException as exc:
                last_error = CerememoryTimeoutError(
                    f"Request timed out after {self._timeout}s: {exc}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                    continue
                raise last_error from exc

            if response.status_code < 400:
                return response

            error = _parse_cmp_error(response)

            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < self._max_retries
            ):
                wait = _compute_backoff(attempt, error.retry_after)
                logger.debug(
                    "Retryable error %d on %s %s (attempt %d/%d), waiting %.1fs",
                    response.status_code,
                    method,
                    path,
                    attempt + 1,
                    self._max_retries + 1,
                    wait,
                )
                await asyncio.sleep(wait)
                last_error = error
                continue

            raise error

        assert last_error is not None
        raise last_error
