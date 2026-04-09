"""Integration tests for RedisStore — requires running Redis container."""

from __future__ import annotations

import pytest

from anycode.memory.redis_store import RedisStore


@pytest.mark.integration
@pytest.mark.redis
class TestRedisStoreIntegration:
    @pytest.mark.asyncio
    async def test_set_and_get(self, require_redis: None, redis_url: str) -> None:

        store = RedisStore(redis_url)
        await store.setup()
        try:
            await store.set("hello", "world")
            got = await store.get("hello")
            assert got is not None
            assert got.value == "world"
        finally:
            await store.clear()
            await store.teardown()

    @pytest.mark.asyncio
    async def test_list(self, require_redis: None, redis_url: str) -> None:

        store = RedisStore(redis_url)
        await store.setup()
        try:
            for i in range(3):
                await store.set(f"k{i}", str(i))
            entries = await store.list()
            assert len(entries) == 3
        finally:
            await store.clear()
            await store.teardown()

    @pytest.mark.asyncio
    async def test_delete(self, require_redis: None, redis_url: str) -> None:

        store = RedisStore(redis_url)
        await store.setup()
        try:
            await store.set("x", "v")
            await store.delete("x")
            assert await store.get("x") is None
        finally:
            await store.clear()
            await store.teardown()
