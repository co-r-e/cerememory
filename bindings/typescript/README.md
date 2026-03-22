# Cerememory TypeScript SDK

TypeScript/Node.js bindings for Cerememory.

## Features

- High-level `store()`, `recall()`, `forget()`, and `stats()` helpers
- Full CMP protocol access, including `encodeStore`, `recallQuery`, and lifecycle APIs
- `store()` supports `metadata` and `emotion` wiring
- `recall()` supports `reconsolidate` and `activation_depth`
- Retries are opt-in; the default is `maxRetries: 0`
- Native package support via `@cerememory/native`

## Example

```ts
import { CerememoryClient } from "@cerememory/sdk";

const client = new CerememoryClient("http://localhost:8420");

const id = await client.store("Coffee chat with Alice", {
  store: "episodic",
  metadata: { source: "chat" },
});

const memories = await client.recall("coffee", {
  limit: 5,
  reconsolidate: false,
  activation_depth: 4,
});
```
