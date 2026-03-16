"""CerememoryError hierarchy mapping CMP error codes to Python exceptions.

Each CMP error code has a corresponding exception class. The transport layer
maps HTTP error responses to these exceptions, preserving the original error
code, message, details, and retry_after hint from the server.
"""

from __future__ import annotations

from typing import Any


class CerememoryError(Exception):
    """Base exception for all Cerememory SDK errors.

    Attributes:
        code: The CMP error code string (e.g. ``"RECORD_NOT_FOUND"``).
        message: Human-readable error description from the server.
        details: Optional structured details from the server.
        retry_after: Optional seconds to wait before retrying.
        status_code: The HTTP status code that was returned.
    """

    code: str = "UNKNOWN"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: Any | None = None,
        retry_after: int | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.details = details
        self.retry_after = retry_after
        self.status_code = status_code


# --- 400 Bad Request ---


class ValidationError(CerememoryError):
    """Request validation failed (400)."""

    code = "VALIDATION_ERROR"


class StoreInvalidError(CerememoryError):
    """Invalid store type specified (400)."""

    code = "STORE_INVALID"


class ModalityUnsupportedError(CerememoryError):
    """Unsupported content modality (400)."""

    code = "MODALITY_UNSUPPORTED"


class ForgetUnconfirmedError(CerememoryError):
    """Forget operation requires explicit confirmation (400)."""

    code = "FORGET_UNCONFIRMED"


class VersionMismatchError(CerememoryError):
    """Protocol version mismatch (400)."""

    code = "VERSION_MISMATCH"


# --- 401 Unauthorized ---


class UnauthorizedError(CerememoryError):
    """Authentication failed or missing (401)."""

    code = "UNAUTHORIZED"


# --- 404 Not Found ---


class RecordNotFoundError(CerememoryError):
    """The requested record does not exist (404)."""

    code = "RECORD_NOT_FOUND"


# --- 409 Conflict ---


class ConsolidationInProgressError(CerememoryError):
    """A consolidation operation is already running (409)."""

    code = "CONSOLIDATION_IN_PROGRESS"


class ImportConflictError(CerememoryError):
    """Import conflict with existing data (409)."""

    code = "IMPORT_CONFLICT"


# --- 413 Payload Too Large ---


class ContentTooLargeError(CerememoryError):
    """Content exceeds the maximum allowed size (413)."""

    code = "CONTENT_TOO_LARGE"


# --- 429 Too Many Requests ---


class WorkingMemoryFullError(CerememoryError):
    """Working memory store is at capacity (429)."""

    code = "WORKING_MEMORY_FULL"


class RateLimitedError(CerememoryError):
    """Request rate limit exceeded (429)."""

    code = "RATE_LIMITED"


# --- 500 Internal Server Error ---


class InternalError(CerememoryError):
    """Server-side internal error (500)."""

    code = "INTERNAL_ERROR"


class ExportFailedError(CerememoryError):
    """Export operation failed on the server (500)."""

    code = "EXPORT_FAILED"


# --- 503 Service Unavailable ---


class DecayEngineBusyError(CerememoryError):
    """The decay engine is busy; retry later (503)."""

    code = "DECAY_ENGINE_BUSY"


# --- Connection / Transport Errors ---


class ConnectionError(CerememoryError):
    """Failed to connect to the Cerememory server."""

    code = "CONNECTION_ERROR"


class TimeoutError(CerememoryError):
    """Request timed out."""

    code = "TIMEOUT"


# Mapping from CMP error code strings to exception classes.
_CODE_TO_EXCEPTION: dict[str, type[CerememoryError]] = {
    "RECORD_NOT_FOUND": RecordNotFoundError,
    "STORE_INVALID": StoreInvalidError,
    "CONTENT_TOO_LARGE": ContentTooLargeError,
    "VALIDATION_ERROR": ValidationError,
    "MODALITY_UNSUPPORTED": ModalityUnsupportedError,
    "WORKING_MEMORY_FULL": WorkingMemoryFullError,
    "DECAY_ENGINE_BUSY": DecayEngineBusyError,
    "CONSOLIDATION_IN_PROGRESS": ConsolidationInProgressError,
    "EXPORT_FAILED": ExportFailedError,
    "IMPORT_CONFLICT": ImportConflictError,
    "FORGET_UNCONFIRMED": ForgetUnconfirmedError,
    "VERSION_MISMATCH": VersionMismatchError,
    "UNAUTHORIZED": UnauthorizedError,
    "RATE_LIMITED": RateLimitedError,
    "INTERNAL_ERROR": InternalError,
}


def error_from_code(
    code: str,
    message: str,
    *,
    details: Any | None = None,
    retry_after: int | None = None,
    status_code: int | None = None,
) -> CerememoryError:
    """Create the appropriate exception subclass from a CMP error code."""
    cls = _CODE_TO_EXCEPTION.get(code, CerememoryError)
    return cls(
        message,
        code=code,
        details=details,
        retry_after=retry_after,
        status_code=status_code,
    )
