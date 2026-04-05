"""Async key-value store backed by a dict."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from anycode.types import MemoryEntry


class InMemoryStore:
    """Async in-memory key-value store implementing the MemoryStore protocol."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}

    async def get(self, key: str) -> MemoryEntry | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        existing = self._data.get(key)
        now = datetime.now(UTC)
        self._data[key] = MemoryEntry(
            key=key,
            value=value,
            metadata=dict(metadata) if metadata else None,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

    async def list(self) -> list[MemoryEntry]:
        return list(self._data.values())

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def clear(self) -> None:
        self._data.clear()

    async def search(self, query: str) -> list[MemoryEntry]:
        if not query:
            return await self.list()
        lower = query.lower()
        return [e for e in self._data.values() if lower in e.key.lower() or lower in e.value.lower()]

    @property
    def size(self) -> int:
        return len(self._data)

    def has(self, key: str) -> bool:
        return key in self._data
