/**
 * Tests for the CerememoryProvider (Vercel AI SDK integration).
 *
 * Uses vitest with mocked fetch -- no running server required.
 */

import { describe, it, expect, vi } from "vitest";
import { CerememoryClient, CerememoryProvider } from "../src/index.js";
import type {
  RecallQueryResponse,
  StatsResponse,
  EncodeStoreResponse,
  RecalledMemory,
  MemoryRecord,
} from "../src/index.js";

// ─── Test helpers ────────────────────────────────────────────────────

/** Create a mock fetch that returns a single response. */
function mockFetch(
  status: number,
  body?: unknown,
): typeof globalThis.fetch {
  return vi.fn().mockResolvedValue({
    status,
    headers: new Headers(),
    text: () => Promise.resolve(body != null ? JSON.stringify(body) : ""),
  } satisfies Partial<Response> as unknown as Response);
}

/** Create a client backed by a mock fetch function. */
function createClient(
  fetchFn: typeof globalThis.fetch,
): CerememoryClient {
  return new CerememoryClient("http://localhost:8420", {
    fetch: fetchFn,
    maxRetries: 0,
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

/** A minimal valid MemoryRecord. */
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
        data: Array.from(new TextEncoder().encode("Coffee with Alice")),
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

/** A recall response with one memory. */
const RECALL_RESPONSE: RecallQueryResponse = {
  memories: [
    {
      record: MEMORY_RECORD,
      relevance_score: 0.95,
      activation_path: null,
      rendered_content: MEMORY_RECORD.content,
    },
  ],
  activation_trace: null,
  total_candidates: 1,
};

/** An empty recall response. */
const EMPTY_RECALL_RESPONSE: RecallQueryResponse = {
  memories: [],
  activation_trace: null,
  total_candidates: 0,
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
  evolution_metrics: null,
  background_decay_enabled: true,
};

// ─── CerememoryProvider.tools() ──────────────────────────────────────

describe("CerememoryProvider", () => {
  describe("tools()", () => {
    it("returns three tool definitions", () => {
      const fetchFn = mockFetch(200, {});
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const tools = provider.tools();

      expect(Object.keys(tools)).toEqual([
        "store_memory",
        "recall_memories",
        "memory_stats",
      ]);
    });

    it("store_memory has correct schema", () => {
      const fetchFn = mockFetch(200, {});
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { store_memory } = provider.tools();

      expect(store_memory.description).toContain("Store");
      expect(store_memory.parameters.type).toBe("object");
      expect(store_memory.parameters.properties).toHaveProperty("text");
      expect(store_memory.parameters.properties).toHaveProperty("store");
      expect(store_memory.parameters.required).toEqual(["text"]);
      expect(typeof store_memory.execute).toBe("function");
    });

    it("recall_memories has correct schema", () => {
      const fetchFn = mockFetch(200, {});
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { recall_memories } = provider.tools();

      expect(recall_memories.description).toContain("Recall");
      expect(recall_memories.parameters.type).toBe("object");
      expect(recall_memories.parameters.properties).toHaveProperty("query");
      expect(recall_memories.parameters.properties).toHaveProperty("limit");
      expect(recall_memories.parameters.required).toEqual(["query"]);
      expect(typeof recall_memories.execute).toBe("function");
    });

    it("memory_stats has correct schema", () => {
      const fetchFn = mockFetch(200, {});
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { memory_stats } = provider.tools();

      expect(memory_stats.description).toContain("statistics");
      expect(memory_stats.parameters.type).toBe("object");
      expect(typeof memory_stats.execute).toBe("function");
    });
  });

  describe("tool execution", () => {
    it("store_memory.execute calls client.store", async () => {
      const fetchFn = mockFetch(200, ENCODE_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { store_memory } = provider.tools();
      const result = await store_memory.execute({ text: "Hello world" });

      expect(result.record_id).toBe(ENCODE_RESPONSE.record_id);

      // Verify the fetch was called with the right URL
      const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock
        .calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/encode");
      expect(init.method).toBe("POST");

      // Verify the request body
      const body = JSON.parse(init.body);
      expect(body.store).toBe("episodic"); // default
    });

    it("store_memory.execute uses custom store type", async () => {
      const fetchFn = mockFetch(200, {
        ...ENCODE_RESPONSE,
        store: "semantic",
      });
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { store_memory } = provider.tools();
      await store_memory.execute({
        text: "A fact",
        store: "semantic",
      });

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.store).toBe("semantic");
    });

    it("recall_memories.execute calls client.recall", async () => {
      const fetchFn = mockFetch(200, RECALL_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { recall_memories } = provider.tools();
      const result = await recall_memories.execute({ query: "coffee" });

      expect(result).toHaveLength(1);
      expect(result[0]!.relevance_score).toBe(0.95);

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.cue.text).toBe("coffee");
      expect(body.limit).toBe(5); // default
    });

    it("recall_memories.execute respects custom limit", async () => {
      const fetchFn = mockFetch(200, RECALL_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { recall_memories } = provider.tools();
      await recall_memories.execute({ query: "test", limit: 20 });

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.limit).toBe(20);
    });

    it("memory_stats.execute calls client.stats", async () => {
      const fetchFn = mockFetch(200, STATS_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const { memory_stats } = provider.tools();
      const result = await memory_stats.execute({} as Record<string, never>);

      expect(result.total_records).toBe(42);
      expect(result.avg_fidelity).toBe(0.85);

      const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(url).toBe("http://localhost:8420/v1/introspect/stats");
    });
  });

  describe("augmentPrompt", () => {
    it("appends memories to the base prompt", async () => {
      const fetchFn = mockFetch(200, RECALL_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const result = await provider.augmentPrompt(
        "You are a helpful assistant.",
        "coffee",
        5,
      );

      expect(result).toContain("You are a helpful assistant.");
      expect(result).toContain("Relevant memories:");
      expect(result).toContain("[Memory 1]:");
      expect(result).toContain("Coffee with Alice");
    });

    it("returns base prompt when no memories found", async () => {
      const fetchFn = mockFetch(200, EMPTY_RECALL_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      const result = await provider.augmentPrompt(
        "You are a helpful assistant.",
        "nonexistent topic",
      );

      expect(result).toBe("You are a helpful assistant.");
    });

    it("uses default limit of 5", async () => {
      const fetchFn = mockFetch(200, EMPTY_RECALL_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      await provider.augmentPrompt("prompt", "query");

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.limit).toBe(5);
    });

    it("passes custom limit to recall", async () => {
      const fetchFn = mockFetch(200, EMPTY_RECALL_RESPONSE);
      const client = createClient(fetchFn);
      const provider = new CerememoryProvider(client);

      await provider.augmentPrompt("prompt", "query", 15);

      const body = JSON.parse(
        (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]![1].body,
      );
      expect(body.limit).toBe(15);
    });
  });
});

// ─── Re-export sanity check ─────────────────────────────────────────

describe("re-exports", () => {
  it("CerememoryProvider is exported from the main index", async () => {
    const mod = await import("../src/index.js");
    expect(mod.CerememoryProvider).toBeDefined();
    expect(typeof mod.CerememoryProvider).toBe("function");
  });
});
