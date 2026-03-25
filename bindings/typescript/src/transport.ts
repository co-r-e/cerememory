/**
 * Fetch-based HTTP transport with retry logic for the Cerememory SDK.
 *
 * Uses native `fetch` (Node.js 18+ / browser). Zero external dependencies.
 *
 * @module
 */

import type { CMPErrorEnvelope } from "./types.js";
import {
  CerememoryError,
  NetworkError,
  TimeoutError,
  fromEnvelope,
} from "./errors.js";

/** Configuration for the HTTP transport layer. */
export interface TransportConfig {
  /** Base URL of the Cerememory server (e.g., "http://localhost:8420"). */
  baseUrl: string;

  /** Bearer token for Authorization header. */
  apiKey?: string;

  /** Request timeout in milliseconds. Default: 30000 (30s). */
  timeoutMs?: number;

  /** Maximum number of retries for retryable errors. Default: 0 (opt-in). */
  maxRetries?: number;

  /** When true, retries may also apply to mutating requests. Default: false. */
  retryMutatingRequests?: boolean;

  /** Base delay for exponential backoff in milliseconds. Default: 500. */
  retryBaseDelayMs?: number;

  /**
   * Custom fetch implementation. Defaults to the global `fetch`.
   * Useful for testing or custom HTTP handling.
   */
  fetch?: typeof globalThis.fetch;

  /** Custom headers to include in every request. */
  headers?: Record<string, string>;
}

/** HTTP method type. */
type HttpMethod = "GET" | "HEAD" | "POST" | "PUT" | "PATCH" | "DELETE";

/** Result of a transport request. */
export interface TransportResponse<T> {
  status: number;
  data: T;
  headers: Headers;
}

/** Resolved transport configuration with defaults applied. */
interface ResolvedConfig {
  baseUrl: string;
  apiKey: string | undefined;
  timeoutMs: number;
  maxRetries: number;
  retryMutatingRequests: boolean;
  retryBaseDelayMs: number;
  fetchFn: typeof globalThis.fetch;
  headers: Record<string, string>;
}

function resolveConfig(config: TransportConfig): ResolvedConfig {
  // Normalize base URL: strip trailing slash
  let baseUrl = config.baseUrl;
  while (baseUrl.endsWith("/")) {
    baseUrl = baseUrl.slice(0, -1);
  }

  return {
    baseUrl,
    apiKey: config.apiKey,
    timeoutMs: config.timeoutMs ?? 30_000,
    maxRetries: config.maxRetries ?? 0,
    retryMutatingRequests: config.retryMutatingRequests ?? false,
    retryBaseDelayMs: config.retryBaseDelayMs ?? 500,
    fetchFn: config.fetch ?? globalThis.fetch,
    headers: config.headers ?? {},
  };
}

/**
 * Determine if an HTTP status code is retryable.
 * Retries on 429 (rate limited) and 5xx (server errors).
 */
function isRetryableStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

function canRetryMethod(
  method: HttpMethod,
  retryMutatingRequests: boolean,
): boolean {
  return method === "GET" || method === "HEAD" || retryMutatingRequests;
}

/**
 * Calculate the delay for a given retry attempt using exponential backoff
 * with jitter. Delay = baseDelay * 2^attempt + random jitter [0, baseDelay).
 */
function retryDelay(attempt: number, baseDelayMs: number): number {
  const exponential = baseDelayMs * Math.pow(2, attempt);
  const jitter = Math.random() * baseDelayMs;
  return exponential + jitter;
}

/**
 * Sleep for the specified number of milliseconds.
 * Uses a simple Promise wrapper around setTimeout.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * HTTP transport that wraps native fetch with retry logic, timeouts,
 * and CMP error mapping.
 */
export class Transport {
  private readonly config: ResolvedConfig;
  private readonly baseHeaders: Record<string, string>;

  constructor(config: TransportConfig) {
    this.config = resolveConfig(config);
    const headers: Record<string, string> = {
      Accept: "application/json",
      ...this.config.headers,
    };
    if (this.config.apiKey) {
      headers["Authorization"] = `Bearer ${this.config.apiKey}`;
    }
    this.baseHeaders = headers;
  }

  /**
   * Perform a GET request.
   *
   * @param path - The URL path (e.g., "/v1/introspect/stats").
   * @returns The parsed response data.
   */
  async get<T>(path: string): Promise<T> {
    const response = await this.request<T>("GET", path);
    return response.data;
  }

  /**
   * Perform a POST request.
   *
   * @param path - The URL path.
   * @param body - The JSON request body.
   * @returns The parsed response data.
   */
  async post<T>(path: string, body?: unknown): Promise<T> {
    const response = await this.request<T>("POST", path, body);
    return response.data;
  }

  /**
   * Perform a PUT request.
   *
   * @param path - The URL path.
   * @param body - The JSON request body.
   */
  async put(path: string, body?: unknown): Promise<void> {
    await this.request<void>("PUT", path, body);
  }

  /**
   * Perform a PATCH request.
   *
   * @param path - The URL path.
   * @param body - The JSON request body.
   */
  async patch(path: string, body?: unknown): Promise<void> {
    await this.request<void>("PATCH", path, body);
  }

  /**
   * Perform a DELETE request.
   *
   * @param path - The URL path.
   * @param body - The JSON request body.
   * @returns The parsed response data.
   */
  async delete<T>(path: string, body?: unknown): Promise<T> {
    const response = await this.request<T>("DELETE", path, body);
    return response.data;
  }

  /**
   * Core request method with retry logic, timeout, and error mapping.
   */
  private async request<T>(
    method: HttpMethod,
    path: string,
    body?: unknown,
  ): Promise<TransportResponse<T>> {
    const url = `${this.config.baseUrl}${path}`;
    const maxAttempts = this.config.maxRetries + 1;

    let lastError: Error | undefined;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await this.executeRequest(method, url, body);

        // Success: 2xx
        if (response.status >= 200 && response.status < 300) {
          return await this.parseSuccessResponse<T>(response);
        }

        // Parse error envelope
        const error = await this.parseErrorResponse(response);

        // Check if retryable and we have attempts remaining
        if (
          isRetryableStatus(response.status) &&
          canRetryMethod(method, this.config.retryMutatingRequests) &&
          attempt < maxAttempts - 1
        ) {
          lastError = error;
          const delay = this.getRetryDelay(error, attempt);
          await sleep(delay);
          continue;
        }

        // Non-retryable or no more attempts
        throw error;
      } catch (err) {
        // If it's already a CerememoryError, check retryability
        if (err instanceof CerememoryError) {
          if (
            err.isRetryable &&
            canRetryMethod(method, this.config.retryMutatingRequests) &&
            attempt < maxAttempts - 1
          ) {
            lastError = err;
            const delay = this.getRetryDelay(err, attempt);
            await sleep(delay);
            continue;
          }
          throw err;
        }

        // Network/timeout errors
        const wrappedError = this.wrapFetchError(err);

        if (
          canRetryMethod(method, this.config.retryMutatingRequests) &&
          attempt < maxAttempts - 1
        ) {
          lastError = wrappedError;
          await sleep(retryDelay(attempt, this.config.retryBaseDelayMs));
          continue;
        }

        throw wrappedError;
      }
    }

    // Should not reach here, but satisfy TypeScript
    throw lastError ?? new NetworkError("Request failed after all retries");
  }

  /**
   * Execute a single fetch request with timeout via AbortController.
   */
  private async executeRequest(
    method: HttpMethod,
    url: string,
    body?: unknown,
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.config.timeoutMs,
    );

    try {
      const init: RequestInit = {
        method,
        headers: { ...this.baseHeaders },
        signal: controller.signal,
      };

      if (body !== undefined) {
        headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
      }

      return await this.config.fetchFn(url, init);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Parse a successful (2xx) response. Handles 204 No Content gracefully.
   */
  private async parseSuccessResponse<T>(
    response: Response,
  ): Promise<TransportResponse<T>> {
    if (
      response.status === 204 ||
      response.headers.get("content-length") === "0"
    ) {
      return {
        status: response.status,
        data: undefined as T,
        headers: response.headers,
      };
    }

    const text = await response.text();
    if (!text) {
      return {
        status: response.status,
        data: undefined as T,
        headers: response.headers,
      };
    }

    const data = JSON.parse(text) as T;
    return {
      status: response.status,
      data,
      headers: response.headers,
    };
  }

  /**
   * Parse an error response into a CerememoryError.
   * Attempts to parse the CMP error envelope; falls back to a generic error.
   */
  private async parseErrorResponse(response: Response): Promise<CerememoryError> {
    let text: string;
    try {
      text = await response.text();
    } catch {
      return new NetworkError(
        `HTTP ${response.status}: Failed to read error response body`,
      );
    }

    if (!text) {
      return new CerememoryError("INTERNAL_ERROR", `HTTP ${response.status}`, {
        statusCode: response.status,
      });
    }

    try {
      const envelope = JSON.parse(text) as CMPErrorEnvelope;
      if (envelope.code && envelope.message) {
        return fromEnvelope(envelope, response.status);
      }
    } catch {
      // Not valid JSON or not a CMP envelope
    }

    return new CerememoryError(
      "INTERNAL_ERROR",
      `HTTP ${response.status}: ${text.slice(0, 500)}`,
      { statusCode: response.status },
    );
  }

  /**
   * Get the delay before a retry, respecting server-provided retry_after.
   */
  private getRetryDelay(error: Error, attempt: number): number {
    if (error instanceof CerememoryError && error.retryAfter != null) {
      // Server told us how long to wait (in seconds)
      return error.retryAfter * 1000;
    }
    return retryDelay(attempt, this.config.retryBaseDelayMs);
  }

  /**
   * Wrap native fetch errors into SDK error types.
   */
  private wrapFetchError(err: unknown): CerememoryError {
    if (err instanceof DOMException && err.name === "AbortError") {
      return new TimeoutError(
        `Request timed out after ${this.config.timeoutMs}ms`,
      );
    }

    if (err instanceof TypeError) {
      // TypeError is thrown by fetch for network errors
      return new NetworkError(`Network error: ${err.message}`, err);
    }

    if (err instanceof Error) {
      return new NetworkError(err.message, err);
    }

    return new NetworkError(String(err));
  }
}
