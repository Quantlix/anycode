"""Integration tests for ChromaDBVectorStore — requires running ChromaDB container."""

from __future__ import annotations

import uuid

import pytest

from anycode.memory.chromadb_store import ChromaDBVectorStore


@pytest.mark.integration
@pytest.mark.chromadb
class TestChromaDBVectorStoreIntegration:
    @pytest.mark.asyncio
    async def test_add_and_search(self, require_chromadb: None, chromadb_url: str) -> None:

        collection = f"test_{uuid.uuid4().hex[:8]}"
        store = ChromaDBVectorStore(url=chromadb_url, collection_name=collection)
        await store.setup()
        try:
            await store.add(
                ["Python is a programming language", "JavaScript runs in the browser", "Python data science is popular"],
                [{"topic": "python"}, {"topic": "js"}, {"topic": "python"}],
            )
            results = await store.search("Python programming", top_k=2)
            assert len(results) == 2
            # At least one result should mention Python
            assert any("Python" in r.text for r in results)
        finally:
            await store.clear()
            await store.teardown()

    @pytest.mark.asyncio
    async def test_delete(self, require_chromadb: None, chromadb_url: str) -> None:

        collection = f"test_{uuid.uuid4().hex[:8]}"
        store = ChromaDBVectorStore(url=chromadb_url, collection_name=collection)
        await store.setup()
        try:
            ids = await store.add(["hello world"])
            await store.delete(ids)
            results = await store.search("hello world", top_k=5)
            assert len(results) == 0
        finally:
            await store.clear()
            await store.teardown()
