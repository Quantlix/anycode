"""Async Redis-backed MemoryStore with connection retry."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

try:
    import redis.asyncio as aioredis
except ImportError as exc:
    raise ImportError("RedisStore requires the 'redis' package. Install it with: pip install anycode-py[redis]") from exc

from anycode.types import MemoryEntry

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 0.5
_KEY_PREFIX = "anycode:mem:"


class RedisStore:
    """Async Redis MemoryStore implementation with connection retry."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url = url
        self._client: aioredis.Redis | None = None  # type: ignore[type-arg]

    async def setup(self) -> None:
        self._client = aioredis.from_url(self._url, decode_responses=True)
        await self._retry(self._client.ping)

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _conn(self) -> aioredis.Redis:  # type: ignore[type-arg]
        if self._client is None:
            raise RuntimeError("RedisStore not initialized. Call setup() first.")
        return self._client

    async def get(self, key: str) -> MemoryEntry | None:
        raw = await self._conn().get(_KEY_PREFIX + key)
        if raw is None:
            return None
        return _deserialize(raw)

    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        existing = await self.get(key)
        created = existing.created_at.isoformat() if existing else now
        payload = json.dumps({"key": key, "value": value, "metadata": metadata, "created_at": created, "updated_at": now})
        await self._conn().set(_KEY_PREFIX + key, payload)

    async def list(self) -> list[MemoryEntry]:
        keys: list[str] = []
        async for k in self._conn().scan_iter(match=f"{_KEY_PREFIX}*"):
            keys.append(k)
        if not keys:
            return []
        values = await self._conn().mget(keys)
        return [_deserialize(v) for v in values if v is not None]

    async def delete(self, key: str) -> None:
        await self._conn().delete(_KEY_PREFIX + key)

    async def clear(self) -> None:
        keys: list[str] = []
        async for k in self._conn().scan_iter(match=f"{_KEY_PREFIX}*"):
            keys.append(k)
        if keys:
            await self._conn().delete(*keys)

    async def _retry(self, fn: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return await fn()
            except (ConnectionError, OSError) as exc:
                last_exc = exc
                await asyncio.sleep(RETRY_BASE_SECONDS * (2**attempt))
        raise ConnectionError(f"RedisStore: failed after {MAX_RETRIES} retries") from last_exc


def _deserialize(raw: str) -> MemoryEntry:
    data = json.loads(raw)
    return MemoryEntry(
        key=data["key"],
        value=data["value"],
        metadata=data.get("metadata"),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
    )
