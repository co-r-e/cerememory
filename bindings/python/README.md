# Cerememory Python SDK

Python bindings for Cerememory.

## Features

- Synchronous `Client` and asynchronous `AsyncClient`
- High-level `store()`, `recall()`, `forget()`, and `stats()` helpers
- Full CMP protocol access via the underlying transport clients
- `store()` supports `metadata`
- `recall()` supports `reconsolidate` and `activation_depth`
- Retries are opt-in; the default is `max_retries=0`
- Native package support via `cerememory_native`

## Example

```python
from cerememory import Client

client = Client("http://localhost:8420")

record_id = client.store(
    "Coffee chat with Alice",
    store="episodic",
    metadata={"source": "chat"},
)

memories = client.recall("coffee", reconsolidate=False, activation_depth=4)
```
