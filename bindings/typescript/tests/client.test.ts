/**
 * Tests for @cerememory/sdk — CerememoryClient, Transport, errors.
 *
 * Uses vitest with mocked fetch to test without a running server.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  CerememoryClient,
  CerememoryError,
  RecordNotFoundError,
  UnauthorizedError,
  RateLimitedError,
  ValidationError,
  TimeoutError,
  NetworkError,
  InternalError,
  ForgetUnconfirmedError,
  fromEnvelope,
  Transport,
} from "../src/index.js";
import type {
  StoreType,
  CMPErrorCode,
  CMPErrorEnvelope,
  EncodeStoreRawResponse,
  EncodeStoreResponse,
  RecallQueryResponse,
  RecallRawQueryResponse,
  StatsResponse,
  MemoryRecord,
  HealthResponse,
  DreamTickResponse,
  ForgetResponse,
  EvolutionMetrics,
  ConsolidateResponse,
  DecayTickResponse,
  DecayForecastResponse,
  RecallAssociateResponse,
  RecallTimelineResponse,
  RecallGraphResponse,
  EncodeBatchResponse,
} from "../src/index.js";

// ─── Test helpers ────────────────────────────────────────────────────

/** Create a mock fetch that returns a single response. */
function mockFetch(
  status: number,
  body?: unknown,
  headers?: Record<string, string>,
): typeof globalThis.fetch {
  return vi.fn().mockResolvedValue({
    status,
    headers: new Headers(headers ?? {}),
    text: () => Promise.resolve(body != null ? JSON.stringify(body) : ""),
  } satisfies Partial<Response> as unknown as Response);
}

/** Create a mock fetch that returns sequential responses. */
function mockFetchSequence(
  responses: Array<{
    status: number;
    body?: unknown;
    headers?: Record<string, string>;
  }>,
): typeof globalThis.fetch {
  let callIndex = 0;
  return vi.fn().mockImplementation(() => {
    const idx = callIndex++;
    const resp = responses[idx] ?? responses[responses.length - 1]!;
    return Promise.resolve({
      status: resp.status,
      headers: new Headers(resp.headers ?? {}),
      text: () =>
        Promise.resolve(resp.body != null ? JSON.stringify(resp.body) : ""),
    } satisfies Partial<Response> as unknown as Response);
  });
}

/** Create a client with a custom fetch mock. */
function createClient(
  fetchFn: typeof globalThis.fetch,
  options?: { apiKey?: string },
): CerememoryClient {
  return new CerememoryClient("http://localhost:8420", {
    fetch: fetchFn,
    apiKey: options?.apiKey,
    maxRetries: 0, // Disable retries by default for deterministic tests
    timeoutMs: 5000,
  });
}

/** A minimal valid EncodeStoreResponse. */
const ENCODE_RESPONSE: EncodeStoreResponse = {
  record_id: "01916e3a-1234-7000-8000-000000000001",
  store: "episodic",
  initial_fidelity: 1.0,
  associations_created: 0,
};

const ENCODE_RAW_RESPONSE: EncodeStoreRawResponse = {
  record_id: "01916e3a-1234-7000-8000-000000000099",
  session_id: "sess-1",
  visibility: "normal",
  secrecy_level: "public",
};

const RECALL_RAW_RESPONSE: RecallRawQueryResponse = {
  records: [
    {
      id: "01916e3a-1234-7000-8000-000000000099",
      session_id: "sess-1",
      turn_id: null,
      topic_id: null,
      source: "conversation",
      speaker: "user",
      visibility: "normal",
      secrecy_level: "public",
      created_at: "2025-06-15T12:00:00Z",
      updated_at: "2025-06-15T12:00:00Z",
      content: {
        blocks: [
          {
            modality: "text",
            format: "text/plain",
            data: [82, 97, 119],
            embedding: null,
          },
        ],
        summary: null,
      },
      metadata: {},
      derived_memory_ids: [],
      suppressed: false,
    },
  ],
  total_candidates: 1,
};

const DREAM_TICK_RESPONSE: DreamTickResponse = {
  groups_processed: 1,
  raw_records_processed: 2,
  episodic_summaries_created: 1,
  semantic_nodes_created: 1,
};

/** A minimal valid StatsResponse. */
const STATS_RESPONSE: StatsResponse = {
  total_records: 42,
  records_by_store: { episodic: 30, semantic: 12 },
  total_associations: 10,
  avg_fidelity: 0.85,
  avg_fidelity_by_store: { episodic: 0.9, semantic: 0.75 },
  oldest_record: "2025-01-01T00:00:00Z",
  newest_record: "2025-06-15T12:00:00Z",
  total_recall_count: 100,
  raw_journal_records: 5,
  raw_journal_pending_dream: 2,
  dream_episodic_summaries: 3,
  dream_semantic_nodes: 1,
  last_dream_tick_at: "2025-06-15T12:00:00Z",
  evolution_metrics: null,
  background_decay_enabled: true,
  background_dream_enabled: true,
};

/** A minimal MemoryRecord. */
const MEMORY_RECORD: MemoryRecord = {
  id: "01916e3a-1234-7000-8000-000000000001",
  store: "episodic",
  created_at: "2025-01-01T00:00:00Z",
  updated_at: "2025-01-01T00:00:00Z",
  last_accessed_at: "2025-01-01T00:00:00Z",
  access_count: 5,
  content: {
    blocks: [
      {
        modality: "text",
        format: "text/plain",
        data: [72, 101, 108, 108, 111],
        embedding: null,
      },
    ],
    summary: null,
  },
  fidelity: {
    score: 0.95,
    noise_level: 0.05,
    decay_rate: 0.3,
    emotional_anchor: 1.0,
    reinforcement_count: 3,
    stability: 1.5,
    last_decay_tick: "2025-06-15T12:00:00Z",
  },
  emotion: {
    joy: 0.8,
    trust: 0.5,
    fear: 0.0,
    surprise: 0.1,
    sadness: 0.0,
    disgust: 0.0,
    anger: 0.0,
    anticipation: 0.3,
    intensity: 0.5,
    valence: 0.7,
  },
  associations: [],
  metadata: {},
  version: 1,
};

// ─── Type tests ──────────────────────────────────────────────────────

describe("types", () => {
  it("StoreType union includes all stores", () => {
    const stores: StoreType[] = [
      "episodic",
      "semantic",
      "procedural",
      "emotional",
      "working",
    ];
    expect(stores).toHaveLength(5);
  });

  it("CMPErrorCode union includes all error codes", () => {
    const codes: CMPErrorCode[] = [
      "RECORD_NOT_FOUND",
      "STORE_INVALID",
      "CONTENT_TOO_LARGE",
      "VALIDATION_ERROR",
      "MODALITY_UNSUPPORTED",
      "WORKING_MEMORY_FULL",
      "DECAY_ENGINE_BUSY",
      "CONSOLIDATION_IN_PROGRESS",
      "EXPORT_FAILED",
      "IMPORT_CONFLICT",
      "FORGET_UNCONFIRMED",
      "VERSION_MISMATCH",
      "UNAUTHORIZED",
      "RATE_LIMITED",
      "INTERNAL_ERROR",
    ];
    expect(codes).toHaveLength(15);
  });
});

// ─── Error tests ─────────────────────────────────────────────────────

describe("errors", () => {
  it("CerememoryError has correct properties", () => {
    const err = new CerememoryError("RECORD_NOT_FOUND", "Not found", {
      statusCode: 404,
      details: { id: "abc" },
      requestId: "req-123",
    });
    expect(err.code).toBe("RECORD_NOT_FOUND");
    expect(err.message).toBe("Not found");
    expect(err.statusCode).toBe(404);
    expect(err.details).toEqual({ id: "abc" });
    expect(err.retryAfter).toBeNull();
    expect(err.requestId).toBe("req-123");
    expect(err.name).toBe("CerememoryError");
    expect(err).toBeInstanceOf(Error);
  });

  it("isRetryable is true for RATE_LIMITED, DECAY_ENGINE_BUSY, INTERNAL_ERROR", () => {
    expect(
      new CerememoryError("RATE_LIMITED", "rate limited").isRetryable,
    ).toBe(true);
    expect(
      new CerememoryError("DECAY_ENGINE_BUSY", "busy").isRetryable,
    ).toBe(true);
    expect(
      new CerememoryError("INTERNAL_ERROR", "error").isRetryable,
    ).toBe(true);
  });

  it("isRetryable is false for non-retryable codes", () => {
    expect(
      new CerememoryError("RECORD_NOT_FOUND", "nf").isRetryable,
    ).toBe(false);
    expect(
      new CerememoryError("VALIDATION_ERROR", "bad").isRetryable,
    ).toBe(false);
    expect(
      new CerememoryError("UNAUTHORIZED", "unauth").isRetryable,
    ).toBe(false);
  });

  it("specific error subclasses have correct names and codes", () => {
    const errors = [
      { cls: RecordNotFoundError, code: "RECORD_NOT_FOUND", name: "RecordNotFoundError" },
      { cls: UnauthorizedError, code: "UNAUTHORIZED", name: "UnauthorizedError" },
      { cls: ForgetUnconfirmedError, code: "FORGET_UNCONFIRMED", name: "ForgetUnconfirmedError" },
    ] as const;

    for (const { cls, code, name } of errors) {
      const err = new cls("test");
      expect(err.code).toBe(code);
      expect(err.name).toBe(name);
      expect(err).toBeInstanceOf(CerememoryError);
      expect(err).toBeInstanceOf(Error);
    }
  });

  it("RateLimitedError carries retryAfter", () => {
    const err = new RateLimitedError("slow down", 30, 429);
    expect(err.retryAfter).toBe(30);
    expect(err.statusCode).toBe(429);
    expect(err.isRetryable).toBe(true);
  });

  it("NetworkError wraps cause", () => {
    const cause = new TypeError("Failed to fetch");
    const err = new NetworkError("Network error", cause);
    expect(err.name).toBe("NetworkError");
    expect(err.cause).toBe(cause);
  });

  it("TimeoutError has correct name", () => {
    const err = new TimeoutError("timed out after 5000ms");
    expect(err.name).toBe("TimeoutError");
    expect(err.code).toBe("INTERNAL_ERROR");
  });

  describe("fromEnvelope", () => {
    it("maps RECORD_NOT_FOUND envelope", () => {
      const envelope: CMPErrorEnvelope = {
        code: "RECORD_NOT_FOUND",
        message: "Record not found: abc",
      };
      const err = fromEnvelope(envelope, 404);
      expect(err).toBeInstanceOf(RecordNotFoundError);
      expect(err.statusCode).toBe(404);
    });

    it("maps RATE_LIMITED envelope with retry_after", () => {
      const envelope: CMPErrorEnvelope = {
        code: "RATE_LIMITED",
        message: "Rate limit exceeded",
        retry_after: 60,
        request_id: "req-rate",
      };
      const err = fromEnvelope(envelope, 429);
      expect(err).toBeInstanceOf(RateLimitedError);
      expect(err.retryAfter).toBe(60);
      expect(err.requestId).toBe("req-rate");
    });

    it("maps VALIDATION_ERROR envelope with details", () => {
      const envelope: CMPErrorEnvelope = {
        code: "VALIDATION_ERROR",
        message: "Bad input",
        details: { field: "content" },
      };
      const err = fromEnvelope(envelope, 400);
      expect(err).toBeInstanceOf(ValidationError);
      expect(err.details).toEqual({ field: "content" });
    });

    it("maps all known error codes to specific subclasses", () => {
      const codeToClass: Array<[CMPErrorCode, new (...args: never[]) => CerememoryError]> = [
        ["RECORD_NOT_FOUND", RecordNotFoundError],
        ["UNAUTHORIZED", UnauthorizedError],
        ["RATE_LIMITED", RateLimitedError],
        ["VALIDATION_ERROR", ValidationError],
        ["INTERNAL_ERROR", InternalError],
        ["FORGET_UNCONFIRMED", ForgetUnconfirmedError],
      ];

      for (const [code, ErrorClass] of codeToClass) {
        const envelope: CMPErrorEnvelope = { code, message: "test" };
        const err = fromEnvelope(envelope, 400);
        expect(err).toBeInstanceOf(ErrorClass);
      }
    });

    it("returns generic CerememoryError for unknown codes", () => {
      const envelope = {
        code: "FUTURE_ERROR" as CMPErrorCode,
        message: "Something new",
      };
      const err = fromEnvelope(envelope, 500);
      expect(err).toBeInstanceOf(CerememoryError);
      expect(err.code).toBe("FUTURE_ERROR");
    });
  });
});

// ─── Transport tests ─────────────────────────────────────────────────

describe("Transport", () => {
  it("sends Authorization header when apiKey is set", async () => {
    const fetchFn = mockFetch(200, { status: "ok" });
    const transport = new Transport({
      baseUrl: "http://localhost:8420",
      apiKey: "sk-test-key",
      fetch: fetchFn,
      maxRetries: 0,
    });

    await transport.get("/health");

    expect(fetchFn).toHaveBeenCalledOnce();
    const [, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
    expect(init.headers["Authorization"]).toBe("Bearer sk-test-key");
  });

  it("does not send Authorization header when apiKey is not set", async () => {
    const fetchFn = mockFetch(200, { status: "ok" });
    const transport = new Transport({
      baseUrl: "http://localhost:8420",
      fetch: fetchFn,
      maxRetries: 0,
    });

    await transport.get("/health");

    const [, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
    expect(init.headers["Authorization"]).toBeUndefined();
  });

  it("strips trailing slashes from base URL", async () => {
    const fetchFn = mockFetch(200, { status: "ok" });
    const transport = new Transport({
      baseUrl: "http://localhost:8420///",
      fetch: fetchFn,
      maxRetries: 0,
    });

    await transport.get("/health");

    const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
    expect(url).toBe("http://localhost:8420/health");
  });

  it("sets Content-Type for POST requests", async () => {
    const fetchFn = mockFetch(200, ENCODE_RESPONSE);
    const transport = new Transport({
      baseUrl: "http://localhost:8420",
      fetch: fetchFn,
      maxRetries: 0,
    });

    await transport.post("/v1/encode", { content: {} });

    const [, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(init.method).toBe("POST");
    expect(init.body).toBeTruthy();
  });

  it("handles 204 No Content responses", async () => {
    const fetchFn = mockFetch(204, null, { "content-length": "0" });
    const transport = new Transport({
      baseUrl: "http://localhost:8420",
      fetch: fetchFn,
      maxRetries: 0,
    });

    const result = await transport.put("/v1/lifecycle/mode", { mode: "perfect" });
    expect(result).toBeUndefined();
  });

  it("includes custom headers in requests", async () => {
    const fetchFn = mockFetch(200, { status: "ok" });
    const transport = new Transport({
      baseUrl: "http://localhost:8420",
      fetch: fetchFn,
      maxRetries: 0,
      headers: { "X-Custom": "value" },
    });

    await transport.get("/health");

    const [, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
    expect(init.headers["X-Custom"]).toBe("value");
  });

  it("uses AbortController for timeout", async () => {
    const fetchFn = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      // Verify signal is passed
      expect(init.signal).toBeInstanceOf(AbortSignal);
      return Promise.resolve({
        status: 200,
        headers: new Headers(),
        text: () => Promise.resolve(JSON.stringify({ status: "ok" })),
      });
    });

    const transport = new Transport({
      baseUrl: "http://localhost:8420",
      fetch: fetchFn as typeof globalThis.fetch,
      maxRetries: 0,
      timeoutMs: 1000,
    });

    await transport.get("/health");
    expect(fetchFn).toHaveBeenCalledOnce();
  });

  describe("retry logic", () => {
    it("does not retry by default", async () => {
      const fetchFn = mockFetch(500, {
        code: "INTERNAL_ERROR",
        message: "Transient failure",
      });

      const client = new CerememoryClient("http://localhost:8420", {
        fetch: fetchFn,
      });

      await expect(client.stats()).rejects.toThrow(InternalError);
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });

    it("retries on 429 and succeeds on second attempt", async () => {
      const fetchFn = mockFetchSequence([
        {
          status: 429,
          body: { code: "RATE_LIMITED", message: "Too many requests", retry_after: 1 },
        },
        {
          status: 200,
          body: STATS_RESPONSE,
        },
      ]);

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn,
        maxRetries: 1,
        retryBaseDelayMs: 10, // fast for tests
      });

      const result = await transport.get<StatsResponse>("/v1/introspect/stats");
      expect(result.total_records).toBe(42);
      expect(fetchFn).toHaveBeenCalledTimes(2);
    });

    it("does not retry mutating requests unless explicitly enabled", async () => {
      const fetchFn = mockFetchSequence([
        {
          status: 503,
          body: { code: "DECAY_ENGINE_BUSY", message: "Busy" },
        },
        {
          status: 200,
          body: ENCODE_RESPONSE,
        },
      ]);

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn,
        maxRetries: 1,
        retryBaseDelayMs: 10,
      });

      await expect(transport.post("/v1/encode", {})).rejects.toThrow(
        CerememoryError,
      );
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });

    it("can retry mutating requests when explicitly enabled", async () => {
      const fetchFn = mockFetchSequence([
        {
          status: 503,
          body: { code: "DECAY_ENGINE_BUSY", message: "Busy" },
        },
        {
          status: 200,
          body: ENCODE_RESPONSE,
        },
      ]);

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn,
        maxRetries: 1,
        retryMutatingRequests: true,
        retryBaseDelayMs: 10,
      });

      const result = await transport.post<EncodeStoreResponse>("/v1/encode", {});
      expect(result.record_id).toBe(ENCODE_RESPONSE.record_id);
      expect(fetchFn).toHaveBeenCalledTimes(2);
    });

    it("retries on 500 and succeeds on second attempt", async () => {
      const fetchFn = mockFetchSequence([
        {
          status: 500,
          body: { code: "INTERNAL_ERROR", message: "Transient failure" },
        },
        {
          status: 200,
          body: STATS_RESPONSE,
        },
      ]);

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn,
        maxRetries: 1,
        retryBaseDelayMs: 10,
      });

      const result = await transport.get<StatsResponse>("/v1/introspect/stats");
      expect(result.total_records).toBe(42);
      expect(fetchFn).toHaveBeenCalledTimes(2);
    });

    it("throws after exhausting retries", async () => {
      const fetchFn = mockFetchSequence([
        {
          status: 500,
          body: { code: "INTERNAL_ERROR", message: "Failure 1" },
        },
        {
          status: 500,
          body: { code: "INTERNAL_ERROR", message: "Failure 2" },
        },
        {
          status: 500,
          body: { code: "INTERNAL_ERROR", message: "Failure 3" },
        },
      ]);

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn,
        maxRetries: 2,
        retryBaseDelayMs: 10,
      });

      await expect(
        transport.get("/v1/introspect/stats"),
      ).rejects.toThrow(InternalError);
      expect(fetchFn).toHaveBeenCalledTimes(3);
    });

    it("does not retry on 400 (non-retryable)", async () => {
      const fetchFn = mockFetch(400, {
        code: "VALIDATION_ERROR",
        message: "Bad request",
      });

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn,
        maxRetries: 3,
        retryBaseDelayMs: 10,
      });

      await expect(transport.post("/v1/encode", {})).rejects.toThrow(
        ValidationError,
      );
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });

    it("does not retry on 404 (non-retryable)", async () => {
      const fetchFn = mockFetch(404, {
        code: "RECORD_NOT_FOUND",
        message: "Record not found",
      });

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn,
        maxRetries: 3,
        retryBaseDelayMs: 10,
      });

      await expect(
        transport.get("/v1/introspect/record/abc"),
      ).rejects.toThrow(RecordNotFoundError);
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });

    it("retries on network errors", async () => {
      let callCount = 0;
      const fetchFn = vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.reject(new TypeError("Failed to fetch"));
        }
        return Promise.resolve({
          status: 200,
          headers: new Headers(),
          text: () => Promise.resolve(JSON.stringify(STATS_RESPONSE)),
        });
      });

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn as typeof globalThis.fetch,
        maxRetries: 1,
        retryBaseDelayMs: 10,
      });

      const result = await transport.get<StatsResponse>("/v1/introspect/stats");
      expect(result.total_records).toBe(42);
      expect(fetchFn).toHaveBeenCalledTimes(2);
    });
  });

  describe("error mapping", () => {
    it("maps non-JSON error bodies", async () => {
      const fetchFn = vi.fn().mockResolvedValue({
        status: 502,
        headers: new Headers(),
        text: () => Promise.resolve("Bad Gateway"),
      });

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn as typeof globalThis.fetch,
        maxRetries: 0,
      });

      await expect(transport.get("/health")).rejects.toThrow(CerememoryError);
    });

    it("handles empty error response body", async () => {
      const fetchFn = vi.fn().mockResolvedValue({
        status: 503,
        headers: new Headers(),
        text: () => Promise.resolve(""),
      });

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn as typeof globalThis.fetch,
        maxRetries: 0,
      });

      await expect(transport.get("/health")).rejects.toThrow(CerememoryError);
    });

    it("wraps AbortError as TimeoutError", async () => {
      const fetchFn = vi.fn().mockRejectedValue(
        new DOMException("The operation was aborted.", "AbortError"),
      );

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn as typeof globalThis.fetch,
        maxRetries: 0,
        timeoutMs: 100,
      });

      await expect(transport.get("/health")).rejects.toThrow(TimeoutError);
    });

    it("wraps TypeError as NetworkError", async () => {
      const fetchFn = vi.fn().mockRejectedValue(
        new TypeError("Failed to fetch"),
      );

      const transport = new Transport({
        baseUrl: "http://localhost:8420",
        fetch: fetchFn as typeof globalThis.fetch,
        maxRetries: 0,
      });

      await expect(transport.get("/health")).rejects.toThrow(NetworkError);
    });
  });
});

// ─── Client high-level API tests ─────────────────────────────────────

describe("CerememoryClient", () => {
  describe("health", () => {
    it("checks server health", async () => {
      const fetchFn = mockFetch(200, { status: "ok" });
      const client = createClient(fetchFn);

      const result = await client.health();
      expect(result.status).toBe("ok");

      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/health");
    });
  });

  describe("store", () => {
    it("stores text with default options", async () => {
      const fetchFn = mockFetch(200, ENCODE_RESPONSE);
      const client = createClient(fetchFn);

      const id = await client.store("Coffee chat with Alice");

      expect(id).toBe(ENCODE_RESPONSE.record_id);
      const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/encode");
      expect(init.method).toBe("POST");

      const body = JSON.parse(init.body);
      expect(body.store).toBe("episodic");
      expect(body.content.blocks).toHaveLength(1);
      expect(body.content.blocks[0].modality).toBe("text");
      expect(body.content.blocks[0].format).toBe("text/plain");
      // Check that the data is UTF-8 encoded bytes of the text
      const decoder = new TextDecoder();
      const text = decoder.decode(new Uint8Array(body.content.blocks[0].data));
      expect(text).toBe("Coffee chat with Alice");
    });

    it("stores text with custom store type", async () => {
      const fetchFn = mockFetch(200, {
        ...ENCODE_RESPONSE,
        store: "semantic",
      });
      const client = createClient(fetchFn);

      await client.store("Knowledge fact", { store: "semantic" });

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.store).toBe("semantic");
    });

    it("stores text with partial emotion", async () => {
      const fetchFn = mockFetch(200, ENCODE_RESPONSE);
      const client = createClient(fetchFn);

      await client.store("Happy memory", {
        emotion: { joy: 0.9, valence: 0.8 },
      });

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.emotion.joy).toBe(0.9);
      expect(body.emotion.valence).toBe(0.8);
      // Defaults should be filled in
      expect(body.emotion.fear).toBe(0);
      expect(body.emotion.anger).toBe(0);
    });

    it("stores text with context", async () => {
      const fetchFn = mockFetch(200, ENCODE_RESPONSE);
      const client = createClient(fetchFn);

      await client.store("Session memory", {
        context: { source: "chat", session_id: "sess-123" },
      });

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.context.source).toBe("chat");
      expect(body.context.session_id).toBe("sess-123");
    });

    it("stores text with metadata", async () => {
      const fetchFn = mockFetch(200, ENCODE_RESPONSE);
      const client = createClient(fetchFn);

      await client.store("Session memory", {
        metadata: { source: "chat", session_id: "sess-123" },
      });

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.metadata.source).toBe("chat");
      expect(body.metadata.session_id).toBe("sess-123");
    });
  });

  describe("recall", () => {
    it("recalls memories with default options", async () => {
      const response: RecallQueryResponse = {
        memories: [
          {
            record: MEMORY_RECORD,
            relevance_score: 0.95,
            activation_path: null,
            rendered_content: MEMORY_RECORD.content,
          },
        ],
        activation_trace: null,
        query_metadata: null,
        total_candidates: 1,
      };

      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const memories = await client.recall("coffee");

      expect(memories).toHaveLength(1);
      expect(memories[0]!.relevance_score).toBe(0.95);

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.cue.text).toBe("coffee");
      expect(body.limit).toBe(10);
    });

    it("recalls with custom options", async () => {
      const response: RecallQueryResponse = {
        memories: [],
        activation_trace: null,
        query_metadata: null,
        total_candidates: 0,
      };

      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      await client.recall("test", {
        limit: 5,
        stores: ["episodic", "semantic"],
        recall_mode: "perfect",
        min_fidelity: 0.5,
        include_decayed: true,
        reconsolidate: false,
        activation_depth: 4,
      });

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.limit).toBe(5);
      expect(body.stores).toEqual(["episodic", "semantic"]);
      expect(body.recall_mode).toBe("perfect");
      expect(body.min_fidelity).toBe(0.5);
      expect(body.include_decayed).toBe(true);
      expect(body.reconsolidate).toBe(false);
      expect(body.activation_depth).toBe(4);
    });
  });

  describe("forget", () => {
    it("forgets a record with confirm=true", async () => {
      const fetchFn = mockFetch(200, { records_deleted: 1 });
      const client = createClient(fetchFn);

      const deleted = await client.forget(MEMORY_RECORD.id, { confirm: true });

      expect(deleted).toBe(1);
      const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/lifecycle/forget");
      expect(init.method).toBe("DELETE");

      const body = JSON.parse(init.body);
      expect(body.record_ids).toEqual([MEMORY_RECORD.id]);
      expect(body.confirm).toBe(true);
      expect(body.cascade).toBe(false);
    });

    it("forgets a record with cascade=true", async () => {
      const fetchFn = mockFetch(200, { records_deleted: 3 });
      const client = createClient(fetchFn);

      const deleted = await client.forget(MEMORY_RECORD.id, {
        confirm: true,
        cascade: true,
      });

      expect(deleted).toBe(3);
      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.cascade).toBe(true);
    });

    it("throws ForgetUnconfirmedError when confirm is false", async () => {
      const fetchFn = mockFetch(400, {
        code: "FORGET_UNCONFIRMED",
        message: "Confirm required",
      });
      const client = createClient(fetchFn);

      await expect(
        client.forget(MEMORY_RECORD.id, { confirm: false }),
      ).rejects.toThrow(ForgetUnconfirmedError);
    });
  });

  describe("stats", () => {
    it("retrieves system stats", async () => {
      const fetchFn = mockFetch(200, STATS_RESPONSE);
      const client = createClient(fetchFn);

      const stats = await client.stats();

      expect(stats.total_records).toBe(42);
      expect(stats.records_by_store.episodic).toBe(30);
      expect(stats.avg_fidelity).toBe(0.85);
      expect(stats.background_decay_enabled).toBe(true);

      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/introspect/stats");
    });
  });

  // ─── Full CMP protocol tests ──────────────────────────────────────

  describe("encodeStore", () => {
    it("sends full encode store request", async () => {
      const fetchFn = mockFetch(200, ENCODE_RESPONSE);
      const client = createClient(fetchFn);

      const result = await client.encodeStore({
        content: MEMORY_RECORD.content,
        store: "episodic",
        emotion: MEMORY_RECORD.emotion,
      });

      expect(result.record_id).toBe(ENCODE_RESPONSE.record_id);
      expect(result.store).toBe("episodic");
      expect(result.initial_fidelity).toBe(1.0);
    });
  });

  describe("encodeBatch", () => {
    it("sends batch encode request", async () => {
      const batchResponse: EncodeBatchResponse = {
        results: [ENCODE_RESPONSE],
        associations_inferred: 0,
      };
      const fetchFn = mockFetch(200, batchResponse);
      const client = createClient(fetchFn);

      const result = await client.encodeBatch({
        records: [{ content: MEMORY_RECORD.content, store: "episodic" }],
        infer_associations: true,
      });

      expect(result.results).toHaveLength(1);
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/encode/batch");
    });
  });

  describe("encodeUpdate", () => {
    it("sends PATCH request to update record", async () => {
      const fetchFn = mockFetch(204, null, { "content-length": "0" });
      const client = createClient(fetchFn);

      await client.encodeUpdate(MEMORY_RECORD.id, {
        content: MEMORY_RECORD.content,
        metadata: { updated: true },
      });

      const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe(
        `http://localhost:8420/v1/encode/${MEMORY_RECORD.id}`,
      );
      expect(init.method).toBe("PATCH");
    });
  });

  describe("recallQuery", () => {
    it("sends full recall query request", async () => {
      const response: RecallQueryResponse = {
        memories: [],
        activation_trace: null,
        query_metadata: {
          total_records_scanned: 7,
          stores_searched: ["episodic"],
          fidelity_filtered: 0,
        },
        total_candidates: 0,
      };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.recallQuery({
        cue: { text: "hello", embedding: [0.1, 0.2, 0.3] },
        limit: 20,
        recall_mode: "perfect",
        activation_depth: 3,
      });

      expect(result.memories).toEqual([]);
      expect(result.total_candidates).toBe(0);
      expect(result.query_metadata?.total_records_scanned).toBe(7);
    });
  });

  describe("recallAssociate", () => {
    it("sends association query for a record", async () => {
      const response: RecallAssociateResponse = {
        memories: [],
        total_candidates: 0,
      };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.recallAssociate(MEMORY_RECORD.id, {
        depth: 3,
        min_weight: 0.5,
      });

      expect(result.memories).toEqual([]);
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe(
        `http://localhost:8420/v1/recall/associate/${MEMORY_RECORD.id}`,
      );
    });
  });

  describe("recallTimeline", () => {
    it("sends timeline query", async () => {
      const response: RecallTimelineResponse = { buckets: [] };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.recallTimeline({
        range: {
          start: "2025-01-01T00:00:00Z",
          end: "2025-12-31T23:59:59Z",
        },
        granularity: "day",
      });

      expect(result.buckets).toEqual([]);
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/recall/timeline");
    });
  });

  describe("recallGraph", () => {
    it("sends graph query", async () => {
      const response: RecallGraphResponse = {
        nodes: [],
        edges: [],
        total_nodes: 0,
      };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.recallGraph({
        center_id: MEMORY_RECORD.id,
        depth: 2,
      });

      expect(result.total_nodes).toBe(0);
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/recall/graph");
    });
  });

  describe("lifecycleConsolidate", () => {
    it("triggers consolidation", async () => {
      const response: ConsolidateResponse = {
        records_processed: 10,
        records_migrated: 2,
        records_compressed: 5,
        records_pruned: 1,
        semantic_nodes_created: 3,
      };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.lifecycleConsolidate({
        strategy: "incremental",
        dry_run: false,
      });

      expect(result.records_processed).toBe(10);
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/lifecycle/consolidate");
    });
  });

  describe("lifecycleDecayTick", () => {
    it("triggers decay tick", async () => {
      const response: DecayTickResponse = {
        records_updated: 5,
        records_below_threshold: 2,
        records_pruned: 0,
      };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.lifecycleDecayTick({
        tick_duration_seconds: 3600,
      });

      expect(result.records_updated).toBe(5);
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/lifecycle/decay-tick");
    });
  });

  describe("lifecycleSetMode", () => {
    it("sets recall mode via PUT", async () => {
      const fetchFn = mockFetch(204, null, { "content-length": "0" });
      const client = createClient(fetchFn);

      await client.lifecycleSetMode({ mode: "perfect" });

      const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/lifecycle/mode");
      expect(init.method).toBe("PUT");
    });
  });

  describe("lifecycleForget", () => {
    it("sends full forget request", async () => {
      const response: ForgetResponse = { records_deleted: 2 };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.lifecycleForget({
        store: "working",
        confirm: true,
        cascade: true,
      });

      expect(result.records_deleted).toBe(2);
    });
  });

  describe("raw and dream APIs", () => {
    it("storeRaw stores a raw journal record", async () => {
      const fetchFn = mockFetch(200, ENCODE_RAW_RESPONSE);
      const client = createClient(fetchFn);

      const id = await client.storeRaw("Raw note", { sessionId: "sess-1" });
      expect(id).toBe(ENCODE_RAW_RESPONSE.record_id);
    });

    it("recallRaw recalls raw journal records", async () => {
      const fetchFn = mockFetch(200, RECALL_RAW_RESPONSE);
      const client = createClient(fetchFn);

      const records = await client.recallRaw({ sessionId: "sess-1" });
      expect(records).toHaveLength(1);
      expect(records[0].session_id).toBe("sess-1");
    });

    it("dreamTick triggers dream processing", async () => {
      const fetchFn = mockFetch(200, DREAM_TICK_RESPONSE);
      const client = createClient(fetchFn);

      const result = await client.dreamTick({ session_id: "sess-1" });
      expect(result.groups_processed).toBe(1);
      expect(result.semantic_nodes_created).toBe(1);
    });
  });

  describe("introspectRecord", () => {
    it("retrieves a single record by ID", async () => {
      const fetchFn = mockFetch(200, MEMORY_RECORD);
      const client = createClient(fetchFn);

      const record = await client.introspectRecord(MEMORY_RECORD.id);

      expect(record.id).toBe(MEMORY_RECORD.id);
      expect(record.store).toBe("episodic");
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe(
        `http://localhost:8420/v1/introspect/record/${MEMORY_RECORD.id}`,
      );
    });

    it("throws RecordNotFoundError for missing record", async () => {
      const fetchFn = mockFetch(404, {
        code: "RECORD_NOT_FOUND",
        message: "Record not found: fake-id",
      });
      const client = createClient(fetchFn);

      await expect(client.introspectRecord("fake-id")).rejects.toThrow(
        RecordNotFoundError,
      );
    });
  });

  describe("introspectDecayForecast", () => {
    it("sends decay forecast request", async () => {
      const response: DecayForecastResponse = {
        forecasts: [
          {
            record_id: MEMORY_RECORD.id,
            current_fidelity: 0.95,
            forecasted_fidelity: 0.7,
            estimated_threshold_date: "2025-12-01T00:00:00Z",
          },
        ],
      };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.introspectDecayForecast({
        record_ids: [MEMORY_RECORD.id],
        forecast_at: "2025-12-01T00:00:00Z",
      });

      expect(result.forecasts).toHaveLength(1);
      expect(result.forecasts[0]!.forecasted_fidelity).toBe(0.7);
      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/introspect/decay-forecast");
    });
  });

  describe("introspectEvolution", () => {
    it("retrieves evolution metrics", async () => {
      const response: EvolutionMetrics = {
        parameter_adjustments: [],
        detected_patterns: ["daily-peak"],
        schema_adaptations: [],
      };
      const fetchFn = mockFetch(200, response);
      const client = createClient(fetchFn);

      const result = await client.introspectEvolution();

      expect(result.detected_patterns).toEqual(["daily-peak"]);
      const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/introspect/evolution");
      expect(init.method).toBe("GET");
    });
  });

  // ─── Authentication tests ─────────────────────────────────────────

  describe("authentication", () => {
    it("sends Bearer token when apiKey is configured", async () => {
      const fetchFn = mockFetch(200, { status: "ok" });
      const client = createClient(fetchFn, { apiKey: "sk-test-123" });

      await client.health();

      const [, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(init.headers["Authorization"]).toBe("Bearer sk-test-123");
    });

    it("throws UnauthorizedError on 401", async () => {
      const fetchFn = mockFetch(401, {
        code: "UNAUTHORIZED",
        message: "Invalid API key",
      });
      const client = createClient(fetchFn);

      await expect(client.stats()).rejects.toThrow(UnauthorizedError);
    });
  });

  // ─── Error handling tests ─────────────────────────────────────────

  describe("error handling", () => {
    it("throws specific error for each known CMP error code", async () => {
      const testCases: Array<{
        code: CMPErrorCode;
        status: number;
        errorName: string;
      }> = [
        { code: "RECORD_NOT_FOUND", status: 404, errorName: "RecordNotFoundError" },
        { code: "VALIDATION_ERROR", status: 400, errorName: "ValidationError" },
        { code: "UNAUTHORIZED", status: 401, errorName: "UnauthorizedError" },
        { code: "RATE_LIMITED", status: 429, errorName: "RateLimitedError" },
        { code: "INTERNAL_ERROR", status: 500, errorName: "InternalError" },
      ];

      for (const { code, status, errorName } of testCases) {
        const fetchFn = mockFetch(status, {
          code,
          message: `Test ${code}`,
        });
        const client = createClient(fetchFn);

        try {
          await client.stats();
          expect.fail(`Expected ${errorName} to be thrown`);
        } catch (err) {
          expect(err).toBeInstanceOf(CerememoryError);
          expect((err as CerememoryError).name).toBe(errorName);
          expect((err as CerememoryError).code).toBe(code);
          expect((err as CerememoryError).statusCode).toBe(status);
        }
      }
    });

    it("handles malformed error response gracefully", async () => {
      const fetchFn = vi.fn().mockResolvedValue({
        status: 500,
        headers: new Headers(),
        text: () => Promise.resolve("Internal Server Error"),
      });

      const client = createClient(fetchFn as typeof globalThis.fetch);

      await expect(client.stats()).rejects.toThrow(CerememoryError);
    });

    it("handles empty error response body gracefully", async () => {
      const fetchFn = vi.fn().mockResolvedValue({
        status: 500,
        headers: new Headers(),
        text: () => Promise.resolve(""),
      });

      const client = createClient(fetchFn as typeof globalThis.fetch);

      await expect(client.stats()).rejects.toThrow(CerememoryError);
    });
  });
});
