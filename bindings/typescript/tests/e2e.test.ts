/**
 * End-to-end tests against a running Cerememory server.
 *
 * These tests require a live server at CEREMEMORY_URL (default: http://localhost:8420).
 * Run with:
 *
 *   CEREMEMORY_E2E=1 npx vitest run tests/e2e.test.ts
 */

import { randomUUID } from "crypto";
import { beforeAll, describe, expect, it } from "vitest";

import { CerememoryClient, RecordNotFoundError } from "../src/index.js";

const E2E_ENABLED = process.env.CEREMEMORY_E2E === "1";
const BASE_URL = process.env.CEREMEMORY_URL ?? "http://localhost:8420";

const describeE2E = E2E_ENABLED ? describe : describe.skip;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor<T>(
  callback: () => Promise<T | null>,
  timeoutMs = 10_000,
  intervalMs = 250,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await callback();
    if (result) {
      return result;
    }
    await sleep(intervalMs);
  }
  throw new Error("Timed out waiting for E2E condition");
}

describeE2E("E2E: High-level client CRUD", () => {
  let client: CerememoryClient;

  beforeAll(() => {
    client = new CerememoryClient(BASE_URL);
  });

  it("stores, recalls, and forgets an episodic memory", async () => {
    const tag = randomUUID();
    const recordId = await client.store(`E2E episodic memory ${tag}`, {
      store: "episodic",
    });
    expect(recordId).toBeTruthy();

    const record = await client.introspectRecord(recordId);
    expect(record.id).toBe(recordId);

    const recalled = await waitFor(async () => {
      const results = await client.recall(tag, { stores: ["episodic"] });
      return results.find((memory) => memory.record.id === recordId) ?? null;
    });
    expect(recalled.record.id).toBe(recordId);

    const deleted = await client.forget(recordId, { confirm: true });
    expect(deleted).toBeGreaterThan(0);

    await expect(client.introspectRecord(recordId)).rejects.toBeInstanceOf(
      RecordNotFoundError,
    );
  });

  it("stores and recalls a semantic memory", async () => {
    const tag = randomUUID();
    const recordId = await client.store(`E2E semantic memory ${tag}`, {
      store: "semantic",
    });
    expect(recordId).toBeTruthy();

    const recalled = await waitFor(async () => {
      const results = await client.recall(tag, { stores: ["semantic"] });
      return results.find((memory) => memory.record.id === recordId) ?? null;
    });
    expect(recalled.record.store).toBe("semantic");

    const deleted = await client.forget(recordId, { confirm: true });
    expect(deleted).toBeGreaterThan(0);
  });
});

describeE2E("E2E: Healthcheck", () => {
  let client: CerememoryClient;

  beforeAll(() => {
    client = new CerememoryClient(BASE_URL);
  });

  it("returns healthy status", async () => {
    const health = await client.health();
    expect(health.status).toBe("ok");
  });
});
