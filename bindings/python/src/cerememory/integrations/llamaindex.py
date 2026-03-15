"""LlamaIndex VectorStore integration for the Cerememory SDK.

Provides ``CerememoryVectorStore`` -- a ``BasePydanticVectorStore`` adapter
that delegates storage and retrieval to a Cerememory server.

LlamaIndex is an *optional* dependency.  When it is not installed the module
still imports cleanly, but the class will not interoperate with LlamaIndex
pipelines.
"""

from __future__ import annotations

from typing import Any, List, Optional

from cerememory.client import Client

# ---------------------------------------------------------------------------
# Graceful optional imports
# ---------------------------------------------------------------------------

_LLAMAINDEX_AVAILABLE = False

try:
    from llama_index.core.vector_stores.types import (  # type: ignore[import-untyped]
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.schema import TextNode  # type: ignore[import-untyped]

    _LLAMAINDEX_AVAILABLE = True
except ImportError:

    class BasePydanticVectorStore:  # type: ignore[no-redef]
        """Stub used when ``llama_index`` is not installed."""

    class VectorStoreQuery:  # type: ignore[no-redef]
        """Stub used when ``llama_index`` is not installed."""

        def __init__(
            self,
            query_str: Optional[str] = None,
            similarity_top_k: int = 10,
            **kwargs: Any,
        ) -> None:
            self.query_str = query_str
            self.similarity_top_k = similarity_top_k

    class VectorStoreQueryResult:  # type: ignore[no-redef]
        """Stub used when ``llama_index`` is not installed."""

        def __init__(
            self,
            nodes: Optional[List[Any]] = None,
            similarities: Optional[List[float]] = None,
            ids: Optional[List[str]] = None,
        ) -> None:
            self.nodes = nodes or []
            self.similarities = similarities or []
            self.ids = ids or []

    class TextNode:  # type: ignore[no-redef]
        """Stub used when ``llama_index`` is not installed."""

        def __init__(self, text: str = "", id_: str = "", **kwargs: Any) -> None:
            self.text = text
            self.id_ = id_

        def get_content(self) -> str:
            return self.text


# ---------------------------------------------------------------------------
# CerememoryVectorStore
# ---------------------------------------------------------------------------


class CerememoryVectorStore(BasePydanticVectorStore):  # type: ignore[misc]
    """LlamaIndex ``BasePydanticVectorStore`` backed by a Cerememory server.

    Cerememory manages its own embeddings, so no external embedding model is
    needed.

    Example::

        from cerememory.integrations.llamaindex import CerememoryVectorStore

        vs = CerememoryVectorStore(
            base_url="http://localhost:8420",
            api_key="sk-...",
        )
        # Use with a LlamaIndex VectorStoreIndex:
        # index = VectorStoreIndex.from_vector_store(vs)
    """

    _client: Client
    _store: str

    # LlamaIndex BasePydanticVectorStore class-level flags.
    stores_text: bool = True
    is_embedding_query: bool = False

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        store: str = "semantic",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        try:
            super().__init__()
        except TypeError:
            pass

        self._client = Client(
            base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._store = store

    # -- BasePydanticVectorStore interface -----------------------------------

    def add(self, nodes: List[Any], **kwargs: Any) -> List[str]:
        """Add LlamaIndex ``TextNode`` instances to Cerememory.

        Returns a list of record-ID strings.
        """
        ids: List[str] = []
        for node in nodes:
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            record_id = self._client.store(text, store=self._store)
            ids.append(str(record_id))
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete a memory record by its ID.

        Automatically confirms the deletion.
        """
        from uuid import UUID

        self._client.forget(UUID(ref_doc_id), confirm=True)

    def query(self, query: "VectorStoreQuery", **kwargs: Any) -> "VectorStoreQueryResult":
        """Execute a similarity search against Cerememory.

        Returns a ``VectorStoreQueryResult`` with ``TextNode`` objects,
        similarity scores, and record IDs.
        """
        limit = getattr(query, "similarity_top_k", None) or 10
        query_str = getattr(query, "query_str", None) or ""

        recalled = self._client.recall(query_str, limit=limit)

        nodes: List[Any] = []
        similarities: List[float] = []
        ids: List[str] = []

        for mem in recalled:
            text = _extract_text(mem)
            record_id = str(mem.record.id)
            node = TextNode(text=text, id_=record_id)
            nodes.append(node)
            similarities.append(mem.relevance_score)
            ids.append(record_id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )


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
