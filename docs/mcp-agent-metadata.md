# MCP Agent Metadata

Cerememory's standard MCP mode does not call external LLM APIs. Claude Code and
other MCP-compatible agents already know the local task context, so they should
send useful metadata at write time instead of asking Cerememory to infer it with
OpenAI, Anthropic, Gemini, or another provider.

## Default Contract

- `[llm].provider = "none"` is the standard configuration.
- `store`, `store_raw`, `update`, `batch_store`, and `batch_store_raw` accept
  optional `meta_json`.
- If `meta_json` is omitted, Cerememory stores a structured `unavailable`
  capture state. It does not pretend to know the rationale.
- Provided MetaMemory is indexed for recall, so later queries can search for
  intent, rationale, tags, evidence, decisions, and context edges.
- External LLM providers are optional experimental extensions only and require
  explicit build features such as `llm-openai`, `llm-claude`, or `llm-gemini`.

## Recommended Agent Behavior

When an agent stores a memory, it should include `meta_json` if it can explain
why the memory matters. Good metadata is concise and operational:

- `intent`: why the memory is being stored
- `rationale`: why it is believed to be useful or true
- `trigger`: the workflow or tool action that caused capture
- `goals`: expected future use
- `tags`: short lookup handles
- `evidence`: source excerpts, URIs, raw record references, or labels
- `decision`: selected decision when the memory records a choice
- `confidence`: agent confidence from 0.0 to 1.0
- `context_edges`: typed links to related records when IDs are known

## Example

Pass `meta_json` as a JSON string in the MCP tool arguments:

```json
{
  "content": "Cerememory should run with provider = none by default.",
  "store": "semantic",
  "meta_json": "{\"capture_status\":\"provided\",\"intent\":\"Preserve the product default for future repo work\",\"rationale\":\"The project is intended to serve Claude Code and other MCP clients without requiring external LLM API keys.\",\"trigger\":\"user_request\",\"goals\":[\"avoid accidental external API dependency\",\"keep release checks focused on the no-API path\"],\"tags\":[\"no-external-api\",\"mcp\",\"defaults\"],\"confidence\":0.95}"
}
```

For batch tools, each record can carry its own `meta_json` field.

## What Cerememory Still Does Locally

Without an external LLM provider, Cerememory still stores, indexes, and recalls:

- text content with full-text search
- provided embeddings when a caller supplies them
- structured JSON fields
- typed MetaMemory fields
- raw journal entries and deterministic dream summaries
- associations explicitly provided by the caller or derived by engine logic

Image recall, audio transcription, automatic text/image embeddings, and
LLM-quality abstractive summaries require either caller-supplied data, a future
local provider, or an explicitly enabled external provider.
