/**
 * CerememoryClient — the primary entry point for the Cerememory TypeScript SDK.
 *
 * Provides both a high-level convenience API (`store`, `recall`, `forget`, `stats`)
 * and full CMP protocol methods (`encodeStore`, `recallQuery`, etc.).
 *
 * @example
 * ```ts
 * const client = new CerememoryClient("http://localhost:8420", { apiKey: "sk-..." });
 *
 * // Simple API
 * const id = await client.store("Coffee chat with Alice", { store: "episodic" });
 * const memories = await client.recall("coffee", { limit: 5 });
 * await client.forget(id, { confirm: true });
 *
 * // Full CMP protocol
 * const response = await client.encodeStore({
 *   content: { blocks: [...], summary: null },
 *   store: "episodic",
 * });
 * ```
 *
 * @module
 */

import type {
  ConsolidateRequest,
  ConsolidateResponse,
  DecayForecastRequest,
  DecayForecastResponse,
  DecayTickRequest,
  DecayTickResponse,
  EncodeBatchRequest,
  EncodeBatchResponse,
  EncodeStoreRequest,
  EncodeStoreResponse,
  EncodeUpdateRequest,
  EvolutionMetrics,
  ForgetOptions,
  ForgetRequest,
  ForgetResponse,
  HealthResponse,
  MemoryContent,
  MemoryRecord,
  RecallAssociateRequest,
  RecallAssociateResponse,
  RecallGraphRequest,
  RecallGraphResponse,
  RecallOptions,
  RecallQueryRequest,
  RecallQueryResponse,
  RecallTimelineRequest,
  RecallTimelineResponse,
  RecalledMemory,
  SetModeRequest,
  StatsResponse,
  StoreOptions,
} from "./types.js";
import { Transport } from "./transport.js";
import type { TransportConfig } from "./transport.js";

/** Options for constructing a CerememoryClient. */
export interface CerememoryClientOptions {
  /** Bearer token for authentication. */
  apiKey?: string;

  /** Request timeout in milliseconds. Default: 30000 (30s). */
  timeoutMs?: number;

  /** Maximum number of retries for retryable errors. Default: 3. */
  maxRetries?: number;

  /** Base delay for exponential backoff in ms. Default: 500. */
  retryBaseDelayMs?: number;

  /**
   * Custom fetch implementation. Defaults to the global `fetch`.
   * Useful for testing or custom HTTP handling.
   */
  fetch?: typeof globalThis.fetch;

  /** Custom headers to include in every request. */
  headers?: Record<string, string>;
}

/**
 * Create a text-only MemoryContent payload from a string.
 *
 * @param text - The text content to encode.
 * @returns A MemoryContent object with a single text block.
 */
function textContent(text: string): MemoryContent {
  const encoder = new TextEncoder();
  const bytes = Array.from<number>(encoder.encode(text));
  return {
    blocks: [
      {
        modality: "text",
        format: "text/plain",
        data: bytes,
        embedding: null,
      },
    ],
    summary: null,
  };
}

/**
 * The main Cerememory SDK client.
 *
 * Wraps the Cerememory REST API with a type-safe interface. Uses native
 * `fetch` for HTTP transport, with automatic retry on 429/5xx errors
 * and configurable timeouts.
 */
export class CerememoryClient {
  private readonly transport: Transport;

  /**
   * Create a new CerememoryClient.
   *
   * @param baseUrl - The base URL of the Cerememory server (e.g., "http://localhost:8420").
   * @param options - Client configuration options.
   */
  constructor(baseUrl: string, options: CerememoryClientOptions = {}) {
    const transportConfig: TransportConfig = {
      baseUrl,
      apiKey: options.apiKey,
      timeoutMs: options.timeoutMs,
      maxRetries: options.maxRetries,
      retryBaseDelayMs: options.retryBaseDelayMs,
      fetch: options.fetch,
      headers: options.headers,
    };
    this.transport = new Transport(transportConfig);
  }

  // ─── High-level convenience API ──────────────────────────────────────

  /**
   * Store a text memory with sensible defaults.
   *
   * @param text - The text content to store.
   * @param options - Optional store configuration.
   * @returns The UUID of the created record.
   *
   * @example
   * ```ts
   * const id = await client.store("Had coffee with Alice", {
   *   store: "episodic",
   *   emotion: { joy: 0.8, intensity: 0.5, valence: 0.7 },
   * });
   * ```
   */
  async store(text: string, options: StoreOptions = {}): Promise<string> {
    const content = textContent(text);

    const request: EncodeStoreRequest = {
      content,
      store: options.store ?? "episodic",
    };

    if (options.emotion) {
      request.emotion = {
        joy: 0,
        trust: 0,
        fear: 0,
        surprise: 0,
        sadness: 0,
        disgust: 0,
        anger: 0,
        anticipation: 0,
        intensity: 0,
        valence: 0,
        ...options.emotion,
      };
    }

    if (options.context) {
      request.context = options.context;
    }

    const response = await this.encodeStore(request);
    return response.record_id;
  }

  /**
   * Recall memories matching a text query.
   *
   * @param query - The text query to search for.
   * @param options - Optional recall configuration.
   * @returns Array of recalled memories.
   *
   * @example
   * ```ts
   * const memories = await client.recall("coffee", { limit: 5 });
   * ```
   */
  async recall(
    query: string,
    options: RecallOptions = {},
  ): Promise<RecalledMemory[]> {
    const request: RecallQueryRequest = {
      cue: { text: query },
      limit: options.limit ?? 10,
    };

    if (options.stores) {
      request.stores = options.stores;
    }
    if (options.min_fidelity !== undefined) {
      request.min_fidelity = options.min_fidelity;
    }
    if (options.recall_mode) {
      request.recall_mode = options.recall_mode;
    }
    if (options.include_decayed !== undefined) {
      request.include_decayed = options.include_decayed;
    }

    const response = await this.recallQuery(request);
    return response.memories;
  }

  /**
   * Delete a specific memory record.
   *
   * @param recordId - The UUID of the record to delete.
   * @param options - Forget options (confirm must be true).
   * @returns The number of records deleted.
   *
   * @example
   * ```ts
   * await client.forget(recordId, { confirm: true });
   * ```
   */
  async forget(recordId: string, options: ForgetOptions): Promise<number> {
    const request: ForgetRequest = {
      record_ids: [recordId],
      confirm: options.confirm,
      cascade: options.cascade ?? false,
    };

    const response = await this.lifecycleForget(request);
    return response.records_deleted;
  }

  /**
   * Get memory system statistics.
   *
   * @returns Current stats including record counts, avg fidelity, etc.
   */
  async stats(): Promise<StatsResponse> {
    return this.introspectStats();
  }

  /**
   * Check if the server is healthy.
   *
   * @returns The health status.
   */
  async health(): Promise<HealthResponse> {
    return this.transport.get<HealthResponse>("/health");
  }

  // ─── Full CMP protocol: Encode ───────────────────────────────────────

  /**
   * Store a new memory record (CMP encode.store).
   *
   * @param request - The encode store request.
   * @returns The encode store response with record_id and metadata.
   */
  async encodeStore(request: EncodeStoreRequest): Promise<EncodeStoreResponse> {
    return this.transport.post<EncodeStoreResponse>("/v1/encode", request);
  }

  /**
   * Store multiple memory records in a single batch (CMP encode.batch).
   *
   * @param request - The batch encode request.
   * @returns Results for each record and count of inferred associations.
   */
  async encodeBatch(
    request: EncodeBatchRequest,
  ): Promise<EncodeBatchResponse> {
    return this.transport.post<EncodeBatchResponse>(
      "/v1/encode/batch",
      request,
    );
  }

  /**
   * Update an existing memory record (CMP encode.update).
   *
   * @param recordId - The UUID of the record to update.
   * @param request - The fields to update.
   */
  async encodeUpdate(
    recordId: string,
    request: Omit<EncodeUpdateRequest, "record_id">,
  ): Promise<void> {
    await this.transport.patch(`/v1/encode/${recordId}`, {
      ...request,
      record_id: recordId,
    });
  }

  // ─── Full CMP protocol: Recall ──────────────────────────────────────

  /**
   * Query memories by multimodal cue (CMP recall.query).
   *
   * @param request - The recall query request.
   * @returns Matched memories with relevance scores.
   */
  async recallQuery(
    request: RecallQueryRequest,
  ): Promise<RecallQueryResponse> {
    return this.transport.post<RecallQueryResponse>(
      "/v1/recall/query",
      request,
    );
  }

  /**
   * Retrieve associated memories for a given record (CMP recall.associate).
   *
   * @param recordId - The UUID of the source record.
   * @param request - Association query options.
   * @returns Associated memories.
   */
  async recallAssociate(
    recordId: string,
    request: Omit<RecallAssociateRequest, "record_id"> = {},
  ): Promise<RecallAssociateResponse> {
    return this.transport.post<RecallAssociateResponse>(
      `/v1/recall/associate/${recordId}`,
      { ...request, record_id: recordId },
    );
  }

  /**
   * Query memories organized by time buckets (CMP recall.timeline).
   *
   * @param request - The timeline query request.
   * @returns Time-bucketed memories.
   */
  async recallTimeline(
    request: RecallTimelineRequest,
  ): Promise<RecallTimelineResponse> {
    return this.transport.post<RecallTimelineResponse>(
      "/v1/recall/timeline",
      request,
    );
  }

  /**
   * Query the memory association graph (CMP recall.graph).
   *
   * @param request - The graph query request.
   * @returns Graph nodes and edges.
   */
  async recallGraph(
    request: RecallGraphRequest,
  ): Promise<RecallGraphResponse> {
    return this.transport.post<RecallGraphResponse>(
      "/v1/recall/graph",
      request,
    );
  }

  // ─── Full CMP protocol: Lifecycle ───────────────────────────────────

  /**
   * Trigger memory consolidation (CMP lifecycle.consolidate).
   *
   * @param request - Consolidation options.
   * @returns Results of the consolidation process.
   */
  async lifecycleConsolidate(
    request: ConsolidateRequest = {},
  ): Promise<ConsolidateResponse> {
    return this.transport.post<ConsolidateResponse>(
      "/v1/lifecycle/consolidate",
      request,
    );
  }

  /**
   * Trigger a decay tick (CMP lifecycle.decay_tick).
   *
   * @param request - Decay tick options.
   * @returns Results of the decay tick.
   */
  async lifecycleDecayTick(
    request: DecayTickRequest = {},
  ): Promise<DecayTickResponse> {
    return this.transport.post<DecayTickResponse>(
      "/v1/lifecycle/decay-tick",
      request,
    );
  }

  /**
   * Set the recall mode (CMP lifecycle.set_mode).
   *
   * @param request - The mode to set.
   */
  async lifecycleSetMode(request: SetModeRequest): Promise<void> {
    await this.transport.put("/v1/lifecycle/mode", request);
  }

  /**
   * Delete memory records (CMP lifecycle.forget).
   *
   * @param request - The forget request (confirm must be true).
   * @returns The number of records deleted.
   */
  async lifecycleForget(request: ForgetRequest): Promise<ForgetResponse> {
    return this.transport.delete<ForgetResponse>(
      "/v1/lifecycle/forget",
      request,
    );
  }

  // ─── Full CMP protocol: Introspect ─────────────────────────────────

  /**
   * Get memory system statistics (CMP introspect.stats).
   *
   * @returns Current system statistics.
   */
  async introspectStats(): Promise<StatsResponse> {
    return this.transport.get<StatsResponse>("/v1/introspect/stats");
  }

  /**
   * Get a single memory record by ID (CMP introspect.record).
   *
   * @param recordId - The UUID of the record.
   * @returns The full memory record.
   */
  async introspectRecord(recordId: string): Promise<MemoryRecord> {
    return this.transport.get<MemoryRecord>(
      `/v1/introspect/record/${recordId}`,
    );
  }

  /**
   * Forecast fidelity decay for records (CMP introspect.decay_forecast).
   *
   * @param request - The decay forecast request.
   * @returns Decay forecasts for each requested record.
   */
  async introspectDecayForecast(
    request: DecayForecastRequest,
  ): Promise<DecayForecastResponse> {
    return this.transport.post<DecayForecastResponse>(
      "/v1/introspect/decay-forecast",
      request,
    );
  }

  /**
   * Get evolution metrics (CMP introspect.evolution).
   *
   * @returns Current evolution metrics.
   */
  async introspectEvolution(): Promise<EvolutionMetrics> {
    return this.transport.get<EvolutionMetrics>("/v1/introspect/evolution");
  }
}
