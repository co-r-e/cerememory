/**
 * Vercel AI SDK integration for Cerememory.
 *
 * Provides tool definitions that can be used with AI SDK's
 * `generateText` / `streamText`, and a helper to augment prompts with
 * relevant memories.
 *
 * @example
 * ```typescript
 * import { CerememoryClient, CerememoryProvider } from "@cerememory/sdk";
 * import { generateText } from "ai";
 *
 * const client = new CerememoryClient("http://localhost:8420", {
 *   apiKey: "sk-...",
 * });
 * const provider = new CerememoryProvider(client);
 *
 * const result = await generateText({
 *   model: anthropic("claude-sonnet-4-20250514"),
 *   tools: provider.tools(),
 *   prompt: "What do you remember about coffee?",
 * });
 * ```
 *
 * @module
 */

import type { CerememoryClient } from "../client.js";
import type { StoreType, RecalledMemory, StatsResponse } from "../types.js";

/** Result of a store_memory tool invocation. */
export interface StoreMemoryResult {
  record_id: string;
}

/** Arguments accepted by the store_memory tool. */
export interface StoreMemoryArgs {
  text: string;
  store?: StoreType;
}

/** Arguments accepted by the recall_memories tool. */
export interface RecallMemoriesArgs {
  query: string;
  limit?: number;
}

/** A single tool definition compatible with the AI SDK tool interface. */
export interface ToolDefinition<TArgs, TResult> {
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
  execute: (args: TArgs) => Promise<TResult>;
}

/** The set of tools exposed by {@link CerememoryProvider}. */
export interface CerememoryTools {
  store_memory: ToolDefinition<StoreMemoryArgs, StoreMemoryResult>;
  recall_memories: ToolDefinition<RecallMemoriesArgs, RecalledMemory[]>;
  memory_stats: ToolDefinition<Record<string, never>, StatsResponse>;
}

/**
 * Cerememory tool provider for the Vercel AI SDK.
 *
 * Wraps a {@link CerememoryClient} and exposes tool definitions that an
 * LLM can invoke to store, recall, and inspect memories.
 */
export class CerememoryProvider {
  private readonly client: CerememoryClient;

  constructor(client: CerememoryClient) {
    this.client = client;
  }

  /**
   * Return tool definitions for use with AI SDK's tool system.
   *
   * Each tool has a JSON Schema `parameters` object and an `execute`
   * function that calls the underlying Cerememory client.
   */
  tools(): CerememoryTools {
    return {
      store_memory: {
        description: "Store a new memory in Cerememory",
        parameters: {
          type: "object" as const,
          properties: {
            text: {
              type: "string",
              description: "The text content to store",
            },
            store: {
              type: "string",
              enum: [
                "episodic",
                "semantic",
                "procedural",
                "emotional",
                "working",
              ],
              description: "Memory store type (default: episodic)",
            },
          },
          required: ["text"],
        },
        execute: async (args: StoreMemoryArgs): Promise<StoreMemoryResult> => {
          const recordId = await this.client.store(args.text, {
            store: args.store ?? "episodic",
          });
          return { record_id: recordId };
        },
      },

      recall_memories: {
        description: "Recall memories from Cerememory matching a query",
        parameters: {
          type: "object" as const,
          properties: {
            query: {
              type: "string",
              description: "Search query",
            },
            limit: {
              type: "number",
              description: "Maximum number of results (default: 5)",
            },
          },
          required: ["query"],
        },
        execute: async (
          args: RecallMemoriesArgs,
        ): Promise<RecalledMemory[]> => {
          return this.client.recall(args.query, {
            limit: args.limit ?? 5,
          });
        },
      },

      memory_stats: {
        description: "Get Cerememory system statistics",
        parameters: {
          type: "object" as const,
          properties: {},
        },
        execute: async (): Promise<StatsResponse> => {
          return this.client.stats();
        },
      },
    };
  }

  /**
   * Augment a base system prompt with relevant memories.
   *
   * Recalls memories matching *query* and appends them to the prompt.
   * If no memories are found the original prompt is returned unchanged.
   *
   * @param basePrompt - The base system/instruction prompt.
   * @param query - A cue to search for relevant memories.
   * @param limit - Maximum number of memories to include (default: 5).
   * @returns The augmented prompt string.
   */
  async augmentPrompt(
    basePrompt: string,
    query: string,
    limit: number = 5,
  ): Promise<string> {
    const memories = await this.client.recall(query, { limit });

    if (!memories || memories.length === 0) {
      return basePrompt;
    }

    const lines = memories.map((m, i) => {
      const text = extractText(m);
      return `[Memory ${i + 1}]: ${text}`;
    });

    return `${basePrompt}\n\nRelevant memories:\n${lines.join("\n")}`;
  }
}

/**
 * Extract plain-text from a recalled memory's rendered content.
 *
 * @internal
 */
function extractText(memory: RecalledMemory): string {
  try {
    const block = memory.rendered_content?.blocks?.[0];
    if (block?.data) {
      const decoder = new TextDecoder();
      return decoder.decode(new Uint8Array(block.data));
    }
  } catch {
    // fall through
  }
  return "";
}
