/**
 * @cerememory/sdk — TypeScript HTTP client for the Cerememory REST API.
 *
 * Zero external dependencies. Uses native `fetch` (Node.js 18+ / browser).
 *
 * @example
 * ```ts
 * import { CerememoryClient } from "@cerememory/sdk";
 *
 * const client = new CerememoryClient("http://localhost:8420", {
 *   apiKey: "sk-...",
 * });
 *
 * // Store a memory
 * const id = await client.store("Coffee chat with Alice", {
 *   store: "episodic",
 * });
 *
 * // Recall memories
 * const memories = await client.recall("coffee", { limit: 5 });
 *
 * // Check stats
 * const stats = await client.stats();
 * ```
 *
 * @module
 */

// ─── Client ──────────────────────────────────────────────────────────

export { CerememoryClient } from "./client.js";
export type { CerememoryClientOptions } from "./client.js";

// ─── Types ───────────────────────────────────────────────────────────

export type {
  // Enums (string unions)
  StoreType,
  Modality,
  AssociationType,
  RecallMode,
  ConsolidationStrategy,
  TimeGranularity,
  CMPErrorCode,

  // Core data types
  ContentBlock,
  MemoryContent,
  FidelityState,
  EmotionVector,
  Association,
  MemoryRecord,

  // Protocol header
  CMPHeader,

  // Encode operations
  EncodeContext,
  ManualAssociation,
  EncodeStoreRequest,
  EncodeStoreResponse,
  EncodeBatchRequest,
  EncodeBatchResponse,
  EncodeUpdateRequest,

  // Recall operations
  TemporalRange,
  RecallCue,
  RecallQueryRequest,
  RecalledMemory,
  ActivationNode,
  ActivationTrace,
  RecallQueryResponse,
  RecallAssociateRequest,
  RecallAssociateResponse,
  RecallTimelineRequest,
  TimelineBucket,
  RecallTimelineResponse,
  RecallGraphRequest,
  GraphNode,
  GraphEdge,
  RecallGraphResponse,

  // Lifecycle operations
  ConsolidateRequest,
  ConsolidateResponse,
  DecayTickRequest,
  DecayTickResponse,
  SetModeRequest,
  ForgetRequest,
  ForgetResponse,

  // Introspect operations
  ParameterAdjustment,
  EvolutionMetrics,
  StatsResponse,
  DecayForecast,
  DecayForecastRequest,
  DecayForecastResponse,

  // Health
  HealthResponse,

  // Server error envelope
  CMPErrorEnvelope,

  // Convenience types
  StoreOptions,
  RecallOptions,
  ForgetOptions,
} from "./types.js";

// ─── Errors ──────────────────────────────────────────────────────────

export {
  CerememoryError,
  RecordNotFoundError,
  StoreInvalidError,
  ContentTooLargeError,
  ValidationError,
  ModalityUnsupportedError,
  WorkingMemoryFullError,
  DecayEngineBusyError,
  ConsolidationInProgressError,
  ExportFailedError,
  ImportConflictError,
  ForgetUnconfirmedError,
  VersionMismatchError,
  UnauthorizedError,
  RateLimitedError,
  InternalError,
  NetworkError,
  TimeoutError,
  fromEnvelope,
} from "./errors.js";

// ─── Transport (advanced usage) ──────────────────────────────────────

export { Transport } from "./transport.js";
export type { TransportConfig, TransportResponse } from "./transport.js";

// ─── Framework integrations ─────────────────────────────────────────

export { CerememoryProvider } from "./integrations/ai-sdk.js";
export type {
  CerememoryTools,
  StoreMemoryResult,
  StoreMemoryArgs,
  RecallMemoriesArgs,
  ToolDefinition,
} from "./integrations/ai-sdk.js";
