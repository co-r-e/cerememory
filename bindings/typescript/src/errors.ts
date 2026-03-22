/**
 * Error classes for the Cerememory SDK.
 *
 * Maps server CMP error codes to specific error subclasses for
 * structured error handling in client code.
 *
 * @module
 */

import type { CMPErrorCode, CMPErrorEnvelope } from "./types.js";

/**
 * Base error class for all Cerememory SDK errors.
 *
 * Contains the CMP error code, human-readable message, optional details,
 * and an optional retry-after hint (in seconds).
 */
export class CerememoryError extends Error {
  /** CMP error code from the server. */
  public readonly code: CMPErrorCode;

  /** Optional structured details from the server. */
  public readonly details: unknown | null;

  /** Seconds to wait before retrying, if applicable. */
  public readonly retryAfter: number | null;

  /** HTTP status code from the response, if available. */
  public readonly statusCode: number | null;

  /** Request correlation ID from the server, if available. */
  public requestId: string | null;

  constructor(
    code: CMPErrorCode,
    message: string,
    options?: {
      details?: unknown | null;
      retryAfter?: number | null;
      statusCode?: number | null;
      requestId?: string | null;
    },
  ) {
    super(message);
    this.name = "CerememoryError";
    this.code = code;
    this.details = options?.details ?? null;
    this.retryAfter = options?.retryAfter ?? null;
    this.statusCode = options?.statusCode ?? null;
    this.requestId = options?.requestId ?? null;
  }

  /** Whether the client should retry the request. */
  get isRetryable(): boolean {
    return (
      this.code === "RATE_LIMITED" ||
      this.code === "DECAY_ENGINE_BUSY" ||
      this.code === "INTERNAL_ERROR"
    );
  }
}

/** The requested record does not exist. */
export class RecordNotFoundError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("RECORD_NOT_FOUND", message, { statusCode });
    this.name = "RecordNotFoundError";
  }
}

/** The specified store type is invalid. */
export class StoreInvalidError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("STORE_INVALID", message, { statusCode });
    this.name = "StoreInvalidError";
  }
}

/** The content exceeds the maximum allowed size. */
export class ContentTooLargeError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("CONTENT_TOO_LARGE", message, { statusCode });
    this.name = "ContentTooLargeError";
  }
}

/** Request validation failed. */
export class ValidationError extends CerememoryError {
  constructor(message: string, details?: unknown, statusCode?: number) {
    super("VALIDATION_ERROR", message, { details, statusCode });
    this.name = "ValidationError";
  }
}

/** The requested modality is not supported. */
export class ModalityUnsupportedError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("MODALITY_UNSUPPORTED", message, { statusCode });
    this.name = "ModalityUnsupportedError";
  }
}

/** Working memory has reached capacity. */
export class WorkingMemoryFullError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("WORKING_MEMORY_FULL", message, { statusCode });
    this.name = "WorkingMemoryFullError";
  }
}

/** The decay engine is busy processing. */
export class DecayEngineBusyError extends CerememoryError {
  constructor(message: string, retryAfter?: number, statusCode?: number) {
    super("DECAY_ENGINE_BUSY", message, { retryAfter, statusCode });
    this.name = "DecayEngineBusyError";
  }
}

/** A consolidation operation is already in progress. */
export class ConsolidationInProgressError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("CONSOLIDATION_IN_PROGRESS", message, { statusCode });
    this.name = "ConsolidationInProgressError";
  }
}

/** An export operation failed. */
export class ExportFailedError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("EXPORT_FAILED", message, { statusCode });
    this.name = "ExportFailedError";
  }
}

/** An import conflict was detected. */
export class ImportConflictError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("IMPORT_CONFLICT", message, { statusCode });
    this.name = "ImportConflictError";
  }
}

/** A forget operation was not confirmed. */
export class ForgetUnconfirmedError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("FORGET_UNCONFIRMED", message, { statusCode });
    this.name = "ForgetUnconfirmedError";
  }
}

/** The protocol version does not match. */
export class VersionMismatchError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("VERSION_MISMATCH", message, { statusCode });
    this.name = "VersionMismatchError";
  }
}

/** Authentication failed. */
export class UnauthorizedError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("UNAUTHORIZED", message, { statusCode });
    this.name = "UnauthorizedError";
  }
}

/** The client has been rate limited. */
export class RateLimitedError extends CerememoryError {
  constructor(message: string, retryAfter?: number, statusCode?: number) {
    super("RATE_LIMITED", message, { retryAfter, statusCode });
    this.name = "RateLimitedError";
  }
}

/** An internal server error occurred. */
export class InternalError extends CerememoryError {
  constructor(message: string, statusCode?: number) {
    super("INTERNAL_ERROR", message, { statusCode });
    this.name = "InternalError";
  }
}

/** Network or connection-level error (not a CMP error code). */
export class NetworkError extends CerememoryError {
  constructor(message: string, cause?: Error) {
    super("INTERNAL_ERROR", message);
    this.name = "NetworkError";
    if (cause) {
      this.cause = cause;
    }
  }
}

/** Request timed out. */
export class TimeoutError extends CerememoryError {
  constructor(message: string) {
    super("INTERNAL_ERROR", message);
    this.name = "TimeoutError";
  }
}

/**
 * Parse a CMP error envelope from the server into a typed error instance.
 *
 * @param envelope - The parsed error JSON from the server.
 * @param statusCode - The HTTP status code from the response.
 * @returns A specific CerememoryError subclass.
 */
export function fromEnvelope(
  envelope: CMPErrorEnvelope,
  statusCode: number,
): CerememoryError {
  const { code, message, details, retry_after, request_id } = envelope;

  const attachRequestId = <T extends CerememoryError>(err: T): T => {
    err.requestId = request_id ?? null;
    return err;
  };

  switch (code) {
    case "RECORD_NOT_FOUND":
      return attachRequestId(new RecordNotFoundError(message, statusCode));
    case "STORE_INVALID":
      return attachRequestId(new StoreInvalidError(message, statusCode));
    case "CONTENT_TOO_LARGE":
      return attachRequestId(new ContentTooLargeError(message, statusCode));
    case "VALIDATION_ERROR":
      return attachRequestId(new ValidationError(message, details, statusCode));
    case "MODALITY_UNSUPPORTED":
      return attachRequestId(new ModalityUnsupportedError(message, statusCode));
    case "WORKING_MEMORY_FULL":
      return attachRequestId(new WorkingMemoryFullError(message, statusCode));
    case "DECAY_ENGINE_BUSY":
      return attachRequestId(new DecayEngineBusyError(
        message,
        retry_after ?? undefined,
        statusCode,
      ));
    case "CONSOLIDATION_IN_PROGRESS":
      return attachRequestId(
        new ConsolidationInProgressError(message, statusCode),
      );
    case "EXPORT_FAILED":
      return attachRequestId(new ExportFailedError(message, statusCode));
    case "IMPORT_CONFLICT":
      return attachRequestId(new ImportConflictError(message, statusCode));
    case "FORGET_UNCONFIRMED":
      return attachRequestId(new ForgetUnconfirmedError(message, statusCode));
    case "VERSION_MISMATCH":
      return attachRequestId(new VersionMismatchError(message, statusCode));
    case "UNAUTHORIZED":
      return attachRequestId(new UnauthorizedError(message, statusCode));
    case "RATE_LIMITED":
      return attachRequestId(new RateLimitedError(
        message,
        retry_after ?? undefined,
        statusCode,
      ));
    case "INTERNAL_ERROR":
      return attachRequestId(new InternalError(message, statusCode));
    default: {
      // Fallback for unknown codes — future-proof
      const err = new CerememoryError(code, message, {
        details,
        retryAfter: retry_after,
        statusCode,
        requestId: request_id,
      });
      return err;
    }
  }
}
