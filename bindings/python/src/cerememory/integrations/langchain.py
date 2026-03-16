"""LangChain integrations for the Cerememory SDK.

Provides ``CerememoryMemory`` (BaseMemory) and ``CerememoryVectorStore``
(VectorStore) adapters that delegate to a Cerememory server.

LangChain is an *optional* dependency.  When it is not installed the module
still imports cleanly, but the classes will not be usable with LangChain
chains (they won't pass ``isinstance`` checks against the real base classes).
"""

from __future__ import annotations

import contextlib
from typing import Any
from uuid import UUID

from cerememory.client import Client

# ---------------------------------------------------------------------------
# Graceful optional imports
# ---------------------------------------------------------------------------

_LANGCHAIN_AVAILABLE = False

try:
    from langchain_core.documents import Document  # type: ignore[import-untyped]
    from langchain_core.embeddings import Embeddings  # type: ignore[import-untyped]
    from langchain_core.memory import BaseMemory  # type: ignore[import-untyped]
    from langchain_core.vectorstores import VectorStore  # type: ignore[import-untyped]

    _LANGCHAIN_AVAILABLE = True
except ImportError:

    class BaseMemory:  # type: ignore[no-redef]
        """Stub used when ``langchain_core`` is not installed."""

    class VectorStore:  # type: ignore[no-redef]
        """Stub used when ``langchain_core`` is not installed."""

    class Document:  # type: ignore[no-redef]
        """Stub used when ``langchain_core`` is not installed."""

        def __init__(
            self,
            *,
            page_content: str = "",
            metadata: dict[str, Any] | None = None,
        ) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    class Embeddings:  # type: ignore[no-redef]
        """Stub used when ``langchain_core`` is not installed."""


# ---------------------------------------------------------------------------
# CerememoryMemory
# ---------------------------------------------------------------------------


class CerememoryMemory(BaseMemory):  # type: ignore[misc]
    """LangChain ``BaseMemory`` backed by a Cerememory server.

    This adapter stores conversation turns via ``save_context`` and recalls
    relevant memories via ``load_memory_variables``.  It can be dropped into
    any LangChain chain that accepts a ``memory`` parameter.

    Example::

        from cerememory.integrations.langchain import CerememoryMemory

        memory = CerememoryMemory(
            base_url="http://localhost:8420",
            api_key="sk-...",
        )
        memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
        variables = memory.load_memory_variables({"input": "What did I say?"})
    """

    # Instance attributes (not Pydantic fields -- works with both the real
    # BaseMemory and the stub).
    _client: Client
    _memory_key: str
    _input_key: str
    _output_key: str
    _recall_limit: int
    _store: str

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        recall_limit: int = 10,
        store: str = "episodic",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        # BaseMemory.__init__ may or may not accept kwargs depending on
        # whether the real class or the stub is being used.  We guard
        # against both cases.
        with contextlib.suppress(TypeError):
            super().__init__()

        self._client = Client(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._memory_key = memory_key
        self._input_key = input_key
        self._output_key = output_key
        self._recall_limit = recall_limit
        self._store = store

    # -- BaseMemory interface ------------------------------------------------

    @property
    def memory_variables(self) -> list[str]:
        """Return the list of memory variable names provided by this memory."""
        return [self._memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Recall relevant memories using the latest user input as a cue.

        If no query text is available, an empty list is returned.
        """
        query = str(inputs.get(self._input_key, ""))
        if not query:
            return {self._memory_key: []}

        recalled = self._client.recall(query, limit=self._recall_limit)

        # Convert RecalledMemory objects to plain dicts for downstream
        # chain consumption.
        memories: list[dict[str, Any]] = []
        for mem in recalled:
            text = _extract_text(mem)
            memories.append(
                {
                    "text": text,
                    "record_id": str(mem.record.id),
                    "relevance_score": mem.relevance_score,
                }
            )

        return {self._memory_key: memories}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Store the input/output pair as an episodic memory."""
        input_str = str(inputs.get(self._input_key, ""))
        output_str = str(outputs.get(self._output_key, ""))
        text = f"Input: {input_str}\nOutput: {output_str}"
        self._client.store(text, store=self._store)

    def clear(self) -> None:
        """No-op -- Cerememory does not expose a bulk-clear API."""


# ---------------------------------------------------------------------------
# CerememoryVectorStore
# ---------------------------------------------------------------------------


class CerememoryVectorStore(VectorStore):  # type: ignore[misc]
    """LangChain ``VectorStore`` backed by a Cerememory server.

    Cerememory manages its own embeddings, so the ``embedding`` parameter
    required by LangChain's interface is accepted but **ignored**.

    Example::

        from cerememory.integrations.langchain import CerememoryVectorStore

        vs = CerememoryVectorStore(
            base_url="http://localhost:8420",
            api_key="sk-...",
        )
        vs.add_texts(["Hello world", "Foo bar"])
        docs = vs.similarity_search("hello", k=5)
    """

    _client: Client
    _store: str

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        store: str = "semantic",
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        with contextlib.suppress(TypeError):
            super().__init__(**kwargs)

        self._client = Client(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._store = store

    # -- VectorStore interface -----------------------------------------------

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store each text as a separate memory record.

        Returns a list of record-ID strings.
        """
        ids: list[str] = []
        for text in texts:
            record_id: UUID = self._client.store(text, store=self._store)
            ids.append(str(record_id))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for memories similar to *query*.

        Returns up to *k* ``Document`` instances.
        """
        recalled = self._client.recall(query, limit=k)
        documents: list[Document] = []
        for mem in recalled:
            text = _extract_text(mem)
            doc = Document(
                page_content=text,
                metadata={
                    "record_id": str(mem.record.id),
                    "relevance_score": mem.relevance_score,
                    "store": (
                        mem.record.store.value
                        if hasattr(mem.record.store, "value")
                        else str(mem.record.store)
                    ),
                },
            )
            documents.append(doc)
        return documents

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CerememoryVectorStore:
        """Create a new store and populate it with *texts*."""
        store = cls(**kwargs)
        store.add_texts(texts, metadatas)
        return store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(mem: Any) -> str:
    """Extract plain-text content from a ``RecalledMemory``."""
    try:
        blocks = mem.rendered_content.blocks
        if blocks:
            return blocks[0].data.decode("utf-8")
    except (AttributeError, IndexError, UnicodeDecodeError):
        pass
    return ""
