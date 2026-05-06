"""Integration tests for CompositeMemory with real backends."""

from __future__ import annotations

import pytest

from anycode.memory.composite import CompositeMemory
from anycode.memory.redis_store import RedisStore
from anycode.memory.sqlite_store import SQLiteStore
from anycode.memory.vector_store import InMemoryVectorStore


@pytest.mark.integration
class TestCompositeMemoryFileBackedIntegration:
    @pytest.mark.asyncio
    async def test_sqlite_kv_with_vector(self, tmp_path: object) -> None:
        kv = SQLiteStore(f"{tmp_path}/composite.db")
        await kv.setup()
        vector = InMemoryVectorStore()
        composite = CompositeMemory(kv_store=kv, vector_store=vector, auto_index=True)
        try:
            await composite.set("doc1", "Python type hints improve code quality")
            await composite.set("doc2", "Go has built-in concurrency with goroutines")

            got = await composite.get("doc1")
            assert got is not None
            assert "Python" in got.value

            results = await composite.search("type safety in Python", top_k=1)
            assert len(results) == 1
        finally:
            await kv.teardown()


@pytest.mark.integration
@pytest.mark.redis
class TestCompositeMemoryRedisIntegration:
    @pytest.mark.asyncio
    async def test_redis_kv_with_vector(self, require_redis: None, redis_url: str) -> None:
        kv = RedisStore(redis_url)
        await kv.setup()
        vector = InMemoryVectorStore()
        composite = CompositeMemory(kv_store=kv, vector_store=vector, auto_index=True)
        try:
            await composite.set("r1", "Redis caching strategy")
            await composite.set("r2", "Database connection pooling")

            results = await composite.search("caching", top_k=1)
            assert len(results) == 1
        finally:
            await composite.clear()
            await kv.teardown()
