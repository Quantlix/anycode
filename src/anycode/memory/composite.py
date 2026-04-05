"""Composite memory — unified interface querying both KV and vector stores."""

from __future__ import annotations

from typing import Any

from anycode.types import MemoryEntry, MemoryStore, VectorSearchResult, VectorStore


class CompositeMemory:
    """Queries a KV store and an optional vector store as a single memory layer."""

    def __init__(self, kv_store: MemoryStore, vector_store: VectorStore | None = None, *, auto_index: bool = True) -> None:
        self._kv = kv_store
        self._vector = vector_store
        self._auto_index = auto_index

    @property
    def kv_store(self) -> MemoryStore:
        return self._kv

    @property
    def vector_store(self) -> VectorStore | None:
        return self._vector

    async def get(self, key: str) -> MemoryEntry | None:
        return await self._kv.get(key)

    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        await self._kv.set(key, value, metadata)
        if self._auto_index and self._vector:
            await self._vector.add([value], [{"key": key, **(metadata or {})}])

    async def list(self) -> list[MemoryEntry]:
        return await self._kv.list()

    async def delete(self, key: str) -> None:
        await self._kv.delete(key)

    async def clear(self) -> None:
        await self._kv.clear()
        if self._vector:
            await self._vector.clear()

    async def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        if self._vector is None:
            return []
        return await self._vector.search(query, top_k)
