"""Tests for RAG retrieval + indexing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from anycode import AgentRunResult, RAGConfig, TokenUsage, ToolCallRecord, VectorSearchResult
from anycode.memory.indexer import RAGIndexer
from anycode.memory.rag import RAGRetriever


class _StubStore:
    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self._results = results or []
        self.added: list[tuple[list[str], list[dict[str, Any]] | None]] = []

    async def add(self, texts: list[str], metadata: list[dict[str, Any]] | None = None) -> list[str]:
        self.added.append((texts, metadata))
        return [f"id-{i}" for i in range(len(texts))]

    async def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        return self._results[:top_k]

    async def delete(self, ids: list[str]) -> None:
        return None

    async def clear(self) -> None:
        return None


def _result(id_: str, text: str, score: float, *, namespace: str = "default", source: str = "agent:a") -> VectorSearchResult:
    return VectorSearchResult(
        id=id_, text=text, score=score, metadata={"namespace": namespace, "source": source, "timestamp": datetime.now(UTC).isoformat()}
    )


@pytest.mark.asyncio
async def test_retriever_filters_below_min_relevance() -> None:
    store = _StubStore([_result("a", "high", 0.9), _result("b", "low", 0.1)])
    retriever = RAGRetriever(store, RAGConfig(enabled=True, min_relevance=0.5))
    ctx = await retriever.retrieve("q")
    assert len(ctx.entries) == 1
    assert ctx.entries[0].text == "high"


@pytest.mark.asyncio
async def test_retriever_dedupes_seen_ids() -> None:
    store = _StubStore([_result("a", "x", 0.9)])
    retriever = RAGRetriever(store, RAGConfig(enabled=True, min_relevance=0.0))
    first = await retriever.retrieve("q")
    second = await retriever.retrieve("q")
    assert len(first.entries) == 1
    assert len(second.entries) == 0


@pytest.mark.asyncio
async def test_retriever_filters_by_namespace() -> None:
    store = _StubStore([_result("a", "x", 0.9, namespace="other"), _result("b", "y", 0.9, namespace="default")])
    retriever = RAGRetriever(store, RAGConfig(enabled=True, namespace="default", min_relevance=0.0))
    ctx = await retriever.retrieve("q")
    assert {e.text for e in ctx.entries} == {"y"}


@pytest.mark.asyncio
async def test_retriever_disabled_returns_empty() -> None:
    store = _StubStore([_result("a", "x", 0.9)])
    retriever = RAGRetriever(store, RAGConfig(enabled=False))
    ctx = await retriever.retrieve("q")
    assert ctx.entries == []


@pytest.mark.asyncio
async def test_retriever_accepts_dynamic_token_budget() -> None:
    store = _StubStore([_result("a", "x" * 80, 0.9), _result("b", "y" * 80, 0.9)])
    retriever = RAGRetriever(store, RAGConfig(enabled=True, min_relevance=0.0, max_context_tokens=1000))
    ctx = await retriever.retrieve("q", token_budget=25)
    assert [entry.text for entry in ctx.entries] == ["x" * 80]


@pytest.mark.asyncio
async def test_retriever_allows_unlimited_context_budget() -> None:
    store = _StubStore([_result("a", "x" * 80, 0.9), _result("b", "y" * 80, 0.9)])
    retriever = RAGRetriever(store, RAGConfig(enabled=True, min_relevance=0.0, max_context_tokens=None))
    ctx = await retriever.retrieve("q")
    assert len(ctx.entries) == 2


def test_format_context_renders_block() -> None:
    store = _StubStore([_result("a", "hello", 0.9)])
    retriever = RAGRetriever(store, RAGConfig(enabled=True))
    # Synchronously build a context for formatting
    import asyncio

    ctx = asyncio.run(retriever.retrieve("q"))
    text = retriever.format_context(ctx)
    assert "Relevant Context" in text
    assert "hello" in text


@pytest.mark.asyncio
async def test_indexer_chunks_long_output() -> None:
    store = _StubStore()
    indexer = RAGIndexer(store, RAGConfig(enabled=True, auto_index=True, index_tool_results=False))
    long = ("paragraph block " * 100 + "\n\n") * 8
    result = AgentRunResult(success=True, output=long, messages=[], token_usage=TokenUsage(input_tokens=1, output_tokens=1), tool_calls=[])
    ids = await indexer.index_agent_result("a", "p", result)
    assert len(ids) > 1


@pytest.mark.asyncio
async def test_indexer_includes_tool_results() -> None:
    store = _StubStore()
    indexer = RAGIndexer(store, RAGConfig(enabled=True, auto_index=True, index_tool_results=True))
    record = ToolCallRecord(tool_name="bash", input={}, output="some shell output", duration=0.01)
    result = AgentRunResult(success=True, output="x", messages=[], token_usage=TokenUsage(input_tokens=1, output_tokens=1), tool_calls=[record])
    await indexer.index_agent_result("a", "p", result)
    assert len(store.added) == 2  # one for output, one for tool


@pytest.mark.asyncio
async def test_indexer_skip_when_disabled() -> None:
    store = _StubStore()
    indexer = RAGIndexer(store, RAGConfig(enabled=False))
    result = AgentRunResult(success=True, output="x", messages=[], token_usage=TokenUsage(input_tokens=1, output_tokens=1), tool_calls=[])
    ids = await indexer.index_agent_result("a", "p", result)
    assert ids == []
    assert store.added == []
