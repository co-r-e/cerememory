/**
 * TypeScript type definitions for the Cerememory Protocol (CMP).
 *
 * These types mirror the Rust types in `cerememory-core/src/types.rs`
 * and `cerememory-core/src/protocol.rs`.
 *
 * @module
 */

// ─── Enums as string unions ──────────────────────────────────────────

/** Identifies the target memory store. */
export type StoreType =
  | "episodic"
  | "semantic"
  | "procedural"
  | "emotional"
  | "working";

/** Content modality types. */
export type Modality =
  | "text"
  | "image"
  | "audio"
  | "video"
  | "structured"
  | "spatial"
  | "temporal"
  | "interoceptive";

/** Types of associations between memory records. */
export type AssociationType =
  | "temporal"
  | "spatial"
  | "semantic"
  | "emotional"
  | "causal"
  | "sequential"
  | "cross_modal"
  | "user_defined";

/** Recall mode for memory retrieval. */
export type RecallMode = "human" | "perfect";

/** Consolidation strategy. */
export type ConsolidationStrategy = "full" | "incremental" | "selective";

/** Time granularity for timeline queries. */
export type TimeGranularity = "minute" | "hour" | "day" | "week" | "month";

/** CMP error codes returned by the server. */
export type CMPErrorCode =
  | "RECORD_NOT_FOUND"
  | "STORE_INVALID"
  | "CONTENT_TOO_LARGE"
  | "VALIDATION_ERROR"
  | "MODALITY_UNSUPPORTED"
  | "WORKING_MEMORY_FULL"
  | "DECAY_ENGINE_BUSY"
  | "CONSOLIDATION_IN_PROGRESS"
  | "EXPORT_FAILED"
  | "IMPORT_CONFLICT"
  | "FORGET_UNCONFIRMED"
  | "VERSION_MISMATCH"
  | "UNAUTHORIZED"
  | "RATE_LIMITED"
  | "INTERNAL_ERROR";

// ─── Core data types ─────────────────────────────────────────────────

/** A single content block within a memory record. */
export interface ContentBlock {
  modality: Modality;
  format: string;
  /** Raw byte data, serialized as a number array (matching Rust Vec<u8>). */
  data: number[];
  embedding?: number[] | null;
}

/** The payload of a memory record. */
export interface MemoryContent {
  blocks: ContentBlock[];
  summary?: string | null;
}

/** Represents the current decay state of a memory record. */
export interface FidelityState {
  /** 0.0 (fully decayed) to 1.0 (pristine). */
  score: number;
  /** 0.0 (no noise) to 1.0 (fully noisy). */
  noise_level: number;
  /** Per-second decay coefficient. */
  decay_rate: number;
  /** Emotional modulation (higher = slower decay). */
  emotional_anchor: number;
  /** Times reactivated. */
  reinforcement_count: number;
  /** Stability constant (increases with each retrieval). */
  stability: number;
  /** ISO 8601 timestamp. */
  last_decay_tick: string;
}

/** Multi-dimensional affective representation (Plutchik's wheel). */
export interface EmotionVector {
  joy: number;
  trust: number;
  fear: number;
  surprise: number;
  sadness: number;
  disgust: number;
  anger: number;
  anticipation: number;
  intensity: number;
  /** -1.0 (negative) to 1.0 (positive). */
  valence: number;
}

/** A weighted, typed link between two memory records. */
export interface Association {
  target_id: string;
  association_type: AssociationType;
  weight: number;
  /** ISO 8601 timestamp. */
  created_at: string;
  /** ISO 8601 timestamp. */
  last_co_activation: string;
}

/** The fundamental unit of storage in Cerememory. */
export interface MemoryRecord {
  /** UUID (v7). */
  id: string;
  store: StoreType;
  /** ISO 8601 timestamp. */
  created_at: string;
  /** ISO 8601 timestamp. */
  updated_at: string;
  /** ISO 8601 timestamp. */
  last_accessed_at: string;
  access_count: number;
  content: MemoryContent;
  fidelity: FidelityState;
  emotion: EmotionVector;
  associations: Association[];
  metadata: Record<string, unknown>;
  version: number;
}

// ─── Protocol header ─────────────────────────────────────────────────

/** Protocol version header included in all CMP messages. */
export interface CMPHeader {
  protocol: string;
  version: string;
  request_id: string;
  /** ISO 8601 timestamp. */
  timestamp: string;
}

// ─── Encode operations (CMP Spec section 3) ──────────────────────────

/** Context for encoding operations. */
export interface EncodeContext {
  source?: string | null;
  session_id?: string | null;
  spatial?: unknown | null;
  temporal?: unknown | null;
}

/** Manual association hint provided during encoding. */
export interface ManualAssociation {
  target_id: string;
  association_type: AssociationType;
  weight: number;
}

/** encode.store request (CMP Spec section 3.1). */
export interface EncodeStoreRequest {
  header?: CMPHeader | null;
  content: MemoryContent;
  store?: StoreType | null;
  emotion?: EmotionVector | null;
  context?: EncodeContext | null;
  metadata?: Record<string, unknown> | null;
  associations?: ManualAssociation[] | null;
}

/** encode.store response (CMP Spec section 3.1). */
export interface EncodeStoreResponse {
  record_id: string;
  store: StoreType;
  initial_fidelity: number;
  associations_created: number;
}

/** encode.batch request (CMP Spec section 3.2). */
export interface EncodeBatchRequest {
  header?: CMPHeader | null;
  records: EncodeStoreRequest[];
  infer_associations?: boolean;
}

/** encode.batch response (CMP Spec section 3.2). */
export interface EncodeBatchResponse {
  results: EncodeStoreResponse[];
  associations_inferred: number;
}

/** encode.update request (CMP Spec section 3.3). */
export interface EncodeUpdateRequest {
  header?: CMPHeader | null;
  record_id: string;
  content?: MemoryContent | null;
  emotion?: EmotionVector | null;
  metadata?: Record<string, unknown> | null;
}

// ─── Recall operations (CMP Spec section 4) ──────────────────────────

/** Temporal range filter for recall. */
export interface TemporalRange {
  /** ISO 8601 timestamp. */
  start: string;
  /** ISO 8601 timestamp. */
  end: string;
}

/** Multimodal recall cue (CMP Spec section 4.1). */
export interface RecallCue {
  text?: string | null;
  image?: number[] | null;
  audio?: number[] | null;
  emotion?: EmotionVector | null;
  temporal?: TemporalRange | null;
  spatial?: unknown | null;
  semantic?: unknown | null;
  embedding?: number[] | null;
}

/** recall.query request (CMP Spec section 4.1). */
export interface RecallQueryRequest {
  header?: CMPHeader | null;
  cue: RecallCue;
  stores?: StoreType[] | null;
  limit?: number;
  min_fidelity?: number | null;
  include_decayed?: boolean;
  reconsolidate?: boolean;
  activation_depth?: number;
  recall_mode?: RecallMode;
}

/** A single recalled memory with relevance scoring. */
export interface RecalledMemory {
  record: MemoryRecord;
  relevance_score: number;
  activation_path?: string[] | null;
  rendered_content: MemoryContent;
}

/** A single node in an activation trace. */
export interface ActivationNode {
  record_id: string;
  activation_level: number;
  hop: number;
  edge_type: AssociationType;
}

/** Activation trace for debugging recall paths. */
export interface ActivationTrace {
  source_id: string;
  activations: ActivationNode[];
}

/** recall.query response (CMP Spec section 4.1). */
export interface RecallQueryResponse {
  memories: RecalledMemory[];
  activation_trace?: ActivationTrace | null;
  query_metadata?: QueryMetadata | null;
  total_candidates: number;
}

/** Metadata about a recall query execution. */
export interface QueryMetadata {
  total_records_scanned: number;
  stores_searched: StoreType[];
  fidelity_filtered: number;
}

/** recall.associate request (CMP Spec section 4.2). */
export interface RecallAssociateRequest {
  header?: CMPHeader | null;
  record_id: string;
  association_types?: AssociationType[] | null;
  depth?: number;
  min_weight?: number;
  limit?: number;
}

/** recall.associate response. */
export interface RecallAssociateResponse {
  memories: RecalledMemory[];
  total_candidates: number;
}

/** recall.timeline request (CMP Spec section 4.3). */
export interface RecallTimelineRequest {
  header?: CMPHeader | null;
  range: TemporalRange;
  granularity?: TimeGranularity;
  min_fidelity?: number | null;
  emotion_filter?: EmotionVector | null;
}

/** A single time bucket in a timeline query. */
export interface TimelineBucket {
  /** ISO 8601 timestamp. */
  start: string;
  /** ISO 8601 timestamp. */
  end: string;
  memories: RecalledMemory[];
  count: number;
}

/** recall.timeline response (CMP Spec section 4.3). */
export interface RecallTimelineResponse {
  buckets: TimelineBucket[];
}

/** recall.graph request (CMP Spec section 4.4). */
export interface RecallGraphRequest {
  header?: CMPHeader | null;
  center_id?: string | null;
  depth?: number;
  edge_types?: string[] | null;
  limit_nodes?: number;
}

/** A node in the memory graph. */
export interface GraphNode {
  id: string;
  store: StoreType;
  summary?: string | null;
  fidelity: number;
}

/** An edge in the memory graph. */
export interface GraphEdge {
  source: string;
  target: string;
  edge_type: AssociationType;
  weight: number;
}

/** recall.graph response (CMP Spec section 4.4). */
export interface RecallGraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
}

// ─── Lifecycle operations (CMP Spec section 5) ──────────────────────

/** lifecycle.consolidate request (CMP Spec section 5.1). */
export interface ConsolidateRequest {
  header?: CMPHeader | null;
  strategy?: ConsolidationStrategy;
  min_age_hours?: number;
  min_access_count?: number;
  dry_run?: boolean;
}

/** lifecycle.consolidate response (CMP Spec section 5.1). */
export interface ConsolidateResponse {
  records_processed: number;
  records_migrated: number;
  records_compressed: number;
  records_pruned: number;
  semantic_nodes_created: number;
}

/** lifecycle.decay_tick request (CMP Spec section 5.2). */
export interface DecayTickRequest {
  header?: CMPHeader | null;
  tick_duration_seconds?: number | null;
}

/** lifecycle.decay_tick response (CMP Spec section 5.2). */
export interface DecayTickResponse {
  records_updated: number;
  records_below_threshold: number;
  records_pruned: number;
}

/** lifecycle.set_mode request (CMP Spec section 5.3). */
export interface SetModeRequest {
  header?: CMPHeader | null;
  mode: RecallMode;
  scope?: StoreType[] | null;
}

/** lifecycle.forget request (CMP Spec section 5.4). */
export interface ForgetRequest {
  header?: CMPHeader | null;
  record_ids?: string[] | null;
  store?: StoreType | null;
  temporal_range?: TemporalRange | null;
  cascade?: boolean;
  confirm: boolean;
}

/** lifecycle.forget response. */
export interface ForgetResponse {
  records_deleted: number;
}

// ─── Introspect operations (CMP Spec section 6) ─────────────────────

/** Parameter adjustment record. */
export interface ParameterAdjustment {
  store: StoreType;
  parameter: string;
  original_value: number;
  current_value: number;
  reason: string;
}

/** Evolution metrics (CMP Spec section 6.4). */
export interface EvolutionMetrics {
  parameter_adjustments: ParameterAdjustment[];
  detected_patterns: string[];
  schema_adaptations: string[];
}

/** introspect.stats response (CMP Spec section 6.1). */
export interface StatsResponse {
  total_records: number;
  records_by_store: Partial<Record<StoreType, number>>;
  total_associations: number;
  avg_fidelity: number;
  avg_fidelity_by_store: Partial<Record<StoreType, number>>;
  oldest_record?: string | null;
  newest_record?: string | null;
  total_recall_count: number;
  evolution_metrics?: EvolutionMetrics | null;
  background_decay_enabled: boolean;
}

/** A single record's decay forecast. */
export interface DecayForecast {
  record_id: string;
  current_fidelity: number;
  forecasted_fidelity: number;
  estimated_threshold_date?: string | null;
}

/** introspect.decay_forecast request (CMP Spec section 6.3). */
export interface DecayForecastRequest {
  header?: CMPHeader | null;
  record_ids: string[];
  /** ISO 8601 timestamp. */
  forecast_at: string;
}

/** introspect.decay_forecast response (CMP Spec section 6.3). */
export interface DecayForecastResponse {
  forecasts: DecayForecast[];
}

// ─── Server error envelope (CMP Spec section 7) ─────────────────────

/** Standardized CMP error envelope returned by the server. */
export interface CMPErrorEnvelope {
  code: CMPErrorCode;
  message: string;
  details?: unknown | null;
  retry_after?: number | null;
  request_id?: string | null;
}

// ─── Health ──────────────────────────────────────────────────────────

/** Health check response. */
export interface HealthResponse {
  status: string;
}

// ─── Convenience types for simple API ────────────────────────────────

/** Options for the simplified `store()` method. */
export interface StoreOptions {
  store?: StoreType;
  emotion?: Partial<EmotionVector>;
  context?: EncodeContext;
  /** Stored alongside the record as request metadata. */
  metadata?: Record<string, unknown>;
}

/** Options for the simplified `recall()` method. */
export interface RecallOptions {
  limit?: number;
  stores?: StoreType[];
  min_fidelity?: number;
  recall_mode?: RecallMode;
  include_decayed?: boolean;
  /** Whether recall should reconsolidate the result set. Defaults to the protocol default. */
  reconsolidate?: boolean;
  /** Activation depth for spreading activation. Defaults to the protocol default. */
  activation_depth?: number;
}

/** Options for the simplified `forget()` method. */
export interface ForgetOptions {
  confirm: boolean;
  cascade?: boolean;
}
