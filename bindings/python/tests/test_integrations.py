"""Tests for framework integrations (LangChain, LlamaIndex).

All tests use mocked HTTP responses so no running server is needed.
The test suite verifies that the integration classes correctly delegate
to the underlying Cerememory ``Client``.
"""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

import httpx
import pytest
import respx

from cerememory.integrations.langchain import (
    CerememoryMemory,
    CerememoryVectorStore,
    _LANGCHAIN_AVAILABLE,
)
from cerememory.integrations.llamaindex import (
    CerememoryVectorStore as LlamaIndexVectorStore,
    _LLAMAINDEX_AVAILABLE,
)
from tests.conftest import (
    BASE_URL,
    SAMPLE_RECORD_ID,
    make_encode_store_response,
    make_recall_query_response,
)


# ---------------------------------------------------------------------------
# Module-level sanity checks
# ---------------------------------------------------------------------------


class TestGracefulImport:
    """Verify that integrations import cleanly regardless of framework availability."""

    def test_langchain_module_imports(self):
        """The langchain module should always be importable."""
        import cerememory.integrations.langchain  # noqa: F401

    def test_llamaindex_module_imports(self):
        """The llamaindex module should always be importable."""
        import cerememory.integrations.llamaindex  # noqa: F401

    def test_integrations_package_imports(self):
        """The integrations __init__ should always be importable."""
        import cerememory.integrations  # noqa: F401


# ---------------------------------------------------------------------------
# LangChain: CerememoryMemory
# ---------------------------------------------------------------------------


class TestCerememoryMemory:
    """Test the LangChain BaseMemory adapter."""

    @pytest.fixture
    def mock_api(self):
        with respx.mock(base_url=BASE_URL, assert_all_mocked=False) as router:
            yield router

    @pytest.fixture
    def memory(self, mock_api):
        return CerememoryMemory(
            base_url=BASE_URL,
            api_key="test-key",
            recall_limit=5,
            store="episodic",
        )

    def test_memory_variables(self, memory):
        assert memory.memory_variables == ["history"]

    def test_custom_memory_key(self, mock_api):
        mem = CerememoryMemory(
            base_url=BASE_URL,
            memory_key="context",
        )
        assert mem.memory_variables == ["context"]

    def test_load_memory_variables_calls_recall(self, memory, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(200, json=make_recall_query_response())
        )

        result = memory.load_memory_variables({"input": "coffee"})

        assert "history" in result
        memories = result["history"]
        assert len(memories) == 1
        assert memories[0]["relevance_score"] == 0.95
        assert memories[0]["record_id"] == str(SAMPLE_RECORD_ID)

        # Verify the request had the right query
        import json

        request = mock_api.calls.last.request
        body = json.loads(request.content)
        assert body["cue"]["text"] == "coffee"
        assert body["limit"] == 5

    def test_load_memory_variables_empty_input(self, memory, mock_api):
        result = memory.load_memory_variables({})
        assert result == {"history": []}

    def test_load_memory_variables_blank_input(self, memory, mock_api):
        result = memory.load_memory_variables({"input": ""})
        assert result == {"history": []}

    def test_save_context_calls_store(self, memory, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )

        memory.save_context(
            {"input": "Hello there"},
            {"output": "Hi! How can I help?"},
        )

        # Verify the request body
        import json

        request = mock_api.calls.last.request
        body = json.loads(request.content)
        # The stored text should contain both input and output
        data_bytes = bytes(body["content"]["blocks"][0]["data"])
        text = data_bytes.decode("utf-8")
        assert "Input: Hello there" in text
        assert "Output: Hi! How can I help?" in text
        assert body["store"] == "episodic"

    def test_clear_is_noop(self, memory, mock_api):
        """clear() should not raise or make any API calls."""
        memory.clear()
        assert len(mock_api.calls) == 0


# ---------------------------------------------------------------------------
# LangChain: CerememoryVectorStore
# ---------------------------------------------------------------------------


class TestCerememoryVectorStore:
    """Test the LangChain VectorStore adapter."""

    @pytest.fixture
    def mock_api(self):
        with respx.mock(base_url=BASE_URL, assert_all_mocked=False) as router:
            yield router

    @pytest.fixture
    def vector_store(self, mock_api):
        return CerememoryVectorStore(
            base_url=BASE_URL,
            api_key="test-key",
            store="semantic",
        )

    def test_add_texts(self, vector_store, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )

        ids = vector_store.add_texts(["Hello world", "Foo bar"])

        assert len(ids) == 2
        assert ids[0] == str(SAMPLE_RECORD_ID)
        assert mock_api.calls.call_count == 2

    def test_add_texts_uses_correct_store(self, vector_store, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )

        vector_store.add_texts(["test"])

        import json

        body = json.loads(mock_api.calls.last.request.content)
        assert body["store"] == "semantic"

    def test_similarity_search(self, vector_store, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(
                200, json=make_recall_query_response(text="Hello, world!")
            )
        )

        docs = vector_store.similarity_search("hello", k=3)

        assert len(docs) == 1
        assert docs[0].page_content == "Hello, world!"
        assert docs[0].metadata["record_id"] == str(SAMPLE_RECORD_ID)
        assert docs[0].metadata["relevance_score"] == 0.95

        # Verify limit was passed
        import json

        body = json.loads(mock_api.calls.last.request.content)
        assert body["limit"] == 3

    def test_from_texts(self, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )

        vs = CerememoryVectorStore.from_texts(
            texts=["one", "two"],
            base_url=BASE_URL,
            api_key="test-key",
        )

        assert isinstance(vs, CerememoryVectorStore)
        assert mock_api.calls.call_count == 2


# ---------------------------------------------------------------------------
# LlamaIndex: CerememoryVectorStore
# ---------------------------------------------------------------------------


class TestLlamaIndexVectorStore:
    """Test the LlamaIndex BasePydanticVectorStore adapter."""

    @pytest.fixture
    def mock_api(self):
        with respx.mock(base_url=BASE_URL, assert_all_mocked=False) as router:
            yield router

    @pytest.fixture
    def vector_store(self, mock_api):
        return LlamaIndexVectorStore(
            base_url=BASE_URL,
            api_key="test-key",
            store="semantic",
        )

    def test_class_flags(self, vector_store):
        assert vector_store.stores_text is True
        assert vector_store.is_embedding_query is False

    def test_add_nodes(self, vector_store, mock_api):
        mock_api.post("/v1/encode").mock(
            return_value=httpx.Response(200, json=make_encode_store_response())
        )

        # Use the stub TextNode (or real one if llama_index is installed)
        from cerememory.integrations.llamaindex import TextNode

        nodes = [
            TextNode(text="Hello world", id_="node-1"),
            TextNode(text="Foo bar", id_="node-2"),
        ]

        ids = vector_store.add(nodes)

        assert len(ids) == 2
        assert ids[0] == str(SAMPLE_RECORD_ID)
        assert mock_api.calls.call_count == 2

    def test_query(self, vector_store, mock_api):
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(
                200, json=make_recall_query_response(text="Hello, world!")
            )
        )

        from cerememory.integrations.llamaindex import VectorStoreQuery

        query = VectorStoreQuery(query_str="hello", similarity_top_k=3)
        result = vector_store.query(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].text == "Hello, world!"
        assert result.similarities[0] == 0.95
        assert result.ids[0] == str(SAMPLE_RECORD_ID)

        # Verify limit was passed
        import json

        body = json.loads(mock_api.calls.last.request.content)
        assert body["limit"] == 3

    def test_delete(self, vector_store, mock_api):
        mock_api.delete("/v1/lifecycle/forget").mock(
            return_value=httpx.Response(200, json={"records_deleted": 1})
        )

        vector_store.delete(str(SAMPLE_RECORD_ID))

        import json

        body = json.loads(mock_api.calls.last.request.content)
        assert body["confirm"] is True
        assert str(SAMPLE_RECORD_ID) in str(body["record_ids"])

    def test_query_with_empty_results(self, vector_store, mock_api):
        empty_response = {
            "memories": [],
            "activation_trace": None,
            "total_candidates": 0,
        }
        mock_api.post("/v1/recall/query").mock(
            return_value=httpx.Response(200, json=empty_response)
        )

        from cerememory.integrations.llamaindex import VectorStoreQuery

        query = VectorStoreQuery(query_str="nonexistent")
        result = vector_store.query(query)

        assert len(result.nodes) == 0
        assert len(result.similarities) == 0
        assert len(result.ids) == 0
