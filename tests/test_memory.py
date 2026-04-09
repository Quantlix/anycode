"""Unit tests for pluggable memory — SQLite, vector, composite, factory, SharedMemory DI."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from anycode.collaboration.kv_store import InMemoryStore
from anycode.collaboration.shared_mem import SharedMemory
from anycode.memory.composite import CompositeMemory
from anycode.memory.factory import create_memory_store
from anycode.memory.sqlite_store import SQLiteStore
from anycode.memory.vector_store import InMemoryVectorStore
from anycode.types import MemoryConfig


class TestInMemoryStoreUpdatedAt:
    @pytest.mark.asyncio
    async def test_set_populates_updated_at(self) -> None:
        store = InMemoryStore()
        await store.set("k1", "v1")
        entry = await store.get("k1")
        assert entry is not None
        assert entry.updated_at is not None

    @pytest.mark.asyncio
    async def test_update_preserves_created_at(self) -> None:
        store = InMemoryStore()
        await store.set("k1", "v1")
        entry1 = await store.get("k1")
        assert entry1 is not None
        await store.set("k1", "v2")
        entry2 = await store.get("k1")
        assert entry2 is not None
        assert entry2.created_at == entry1.created_at
        assert entry2.value == "v2"
        assert entry2.updated_at is not None
        assert entry2.updated_at >= entry1.updated_at  # type: ignore[operator]


class TestSQLiteStore:
    @pytest.mark.asyncio
    async def test_crud_round_trip(self) -> None:

        store = SQLiteStore(":memory:")
        await store.setup()
        try:
            await store.set("key1", "value1", {"tag": "test"})
            entry = await store.get("key1")
            assert entry is not None
            assert entry.key == "key1"
            assert entry.value == "value1"
            assert entry.metadata == {"tag": "test"}
            assert entry.created_at is not None
            assert entry.updated_at is not None
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_list_and_delete(self) -> None:

        store = SQLiteStore(":memory:")
        await store.setup()
        try:
            await store.set("a", "1")
            await store.set("b", "2")
            entries = await store.list()
            assert len(entries) == 2
            await store.delete("a")
            entries = await store.list()
            assert len(entries) == 1
            assert entries[0].key == "b"
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_clear(self) -> None:

        store = SQLiteStore(":memory:")
        await store.setup()
        try:
            await store.set("x", "y")
            await store.clear()
            entries = await store.list()
            assert entries == []
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_update_preserves_created_at(self) -> None:

        store = SQLiteStore(":memory:")
        await store.setup()
        try:
            await store.set("k", "v1")
            e1 = await store.get("k")
            assert e1 is not None
            await store.set("k", "v2")
            e2 = await store.get("k")
            assert e2 is not None
            assert e2.created_at == e1.created_at
            assert e2.value == "v2"
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self) -> None:

        store = SQLiteStore(":memory:")
        await store.setup()
        try:

            async def _write(i: int) -> None:
                await store.set(f"key-{i}", f"val-{i}")

            await asyncio.gather(*[_write(i) for i in range(10)])
            entries = await store.list()
            assert len(entries) == 10
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self) -> None:

        store = SQLiteStore(":memory:")
        await store.setup()
        try:
            assert await store.get("missing") is None
        finally:
            await store.teardown()


class TestInMemoryVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self) -> None:
        vs = InMemoryVectorStore()
        ids = await vs.add(["Python is a programming language", "Dogs are loyal animals", "Machine learning uses data"])
        assert len(ids) == 3
        results = await vs.search("programming code")
        assert len(results) > 0
        assert results[0].text == "Python is a programming language"

    @pytest.mark.asyncio
    async def test_search_scores_ordered(self) -> None:
        vs = InMemoryVectorStore()
        await vs.add(["apple pie recipe", "banana smoothie recipe", "apple cider vinegar"])
        results = await vs.search("apple", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_search(self) -> None:
        vs = InMemoryVectorStore()
        results = await vs.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        vs = InMemoryVectorStore()
        ids = await vs.add(["hello world"])
        await vs.delete(ids)
        results = await vs.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        vs = InMemoryVectorStore()
        await vs.add(["a", "b", "c"])
        await vs.clear()
        results = await vs.search("a")
        assert results == []

    @pytest.mark.asyncio
    async def test_metadata_preserved(self) -> None:
        vs = InMemoryVectorStore()
        await vs.add(["test document"], [{"source": "unit_test"}])
        results = await vs.search("test")
        assert len(results) == 1
        assert results[0].metadata == {"source": "unit_test"}


class TestCompositeMemory:
    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        kv = InMemoryStore()
        vs = InMemoryVectorStore()
        mem = CompositeMemory(kv, vs)
        await mem.set("key1", "Searching for relevant data", {"topic": "search"})
        entry = await mem.get("key1")
        assert entry is not None
        assert entry.value == "Searching for relevant data"

    @pytest.mark.asyncio
    async def test_vector_search_after_set(self) -> None:
        kv = InMemoryStore()
        vs = InMemoryVectorStore()
        mem = CompositeMemory(kv, vs)
        await mem.set("k1", "Python programming tutorial")
        await mem.set("k2", "Dog training guide")
        results = await mem.search("coding tutorial")
        assert len(results) > 0
        assert any("Python" in r.text for r in results)

    @pytest.mark.asyncio
    async def test_search_without_vector_store(self) -> None:
        kv = InMemoryStore()
        mem = CompositeMemory(kv, vector_store=None)
        await mem.set("k1", "hello")
        results = await mem.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_clear_clears_both(self) -> None:
        kv = InMemoryStore()
        vs = InMemoryVectorStore()
        mem = CompositeMemory(kv, vs)
        await mem.set("k", "v")
        await mem.clear()
        assert await mem.get("k") is None
        assert await vs.search("v") == []


class TestMemoryFactory:
    def test_default_returns_in_memory(self) -> None:
        store = create_memory_store()
        assert isinstance(store, InMemoryStore)

    def test_memory_backend(self) -> None:
        store = create_memory_store(MemoryConfig(backend="memory"))
        assert isinstance(store, InMemoryStore)

    def test_sqlite_backend(self) -> None:

        store = create_memory_store(MemoryConfig(backend="sqlite", path=":memory:"))
        assert isinstance(store, SQLiteStore)

    def test_dict_config(self) -> None:
        store = create_memory_store({"backend": "memory"})
        assert isinstance(store, InMemoryStore)

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValidationError):
            create_memory_store(MemoryConfig(backend="postgres"))  # type: ignore[arg-type]


class TestSharedMemoryDI:
    @pytest.mark.asyncio
    async def test_default_uses_in_memory(self) -> None:
        mem = SharedMemory()
        await mem.write("agent1", "key", "value")
        entry = await mem.read("agent1/key")
        assert entry is not None
        assert entry.value == "value"

    @pytest.mark.asyncio
    async def test_injected_sqlite_store(self) -> None:

        store = SQLiteStore(":memory:")
        await store.setup()
        try:
            mem = SharedMemory(store=store)
            await mem.write("agent1", "key", "value")
            entry = await mem.read("agent1/key")
            assert entry is not None
            assert entry.value == "value"
            raw = await store.get("agent1/key")
            assert raw is not None
            assert raw.value == "value"
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_get_store_returns_injected(self) -> None:

        store = SQLiteStore(":memory:")
        mem = SharedMemory(store=store)
        assert mem.get_store() is store
