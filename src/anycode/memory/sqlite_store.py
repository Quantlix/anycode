"""Async SQLite-backed MemoryStore using aiosqlite with WAL mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

try:
    import aiosqlite
except ImportError as exc:
    raise ImportError("SQLiteStore requires the 'aiosqlite' package. Install it with: pip install anycode-py[persistence]") from exc

from anycode.types import MemoryEntry

_ROW_TYPE = tuple[str, str, str | None, str, str]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_entries (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_at)"


class SQLiteStore:
    """Async SQLite MemoryStore implementation with WAL mode for concurrent reads."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def setup(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()

    async def teardown(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteStore not initialized. Call setup() first.")
        return self._db

    async def get(self, key: str) -> MemoryEntry | None:
        cursor = await self._conn().execute("SELECT key, value, metadata, created_at, updated_at FROM memory_entries WHERE key = ?", (key,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_entry(row)  # type: ignore[arg-type]

    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        existing = await self.get(key)
        created = existing.created_at.isoformat() if existing else now
        await self._conn().execute(
            "INSERT OR REPLACE INTO memory_entries (key, value, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (key, value, meta_json, created, now),
        )
        await self._conn().commit()

    async def list(self) -> list[MemoryEntry]:
        cursor = await self._conn().execute("SELECT key, value, metadata, created_at, updated_at FROM memory_entries ORDER BY created_at")
        rows = await cursor.fetchall()
        return [_row_to_entry(r) for r in rows]  # type: ignore[arg-type]

    async def delete(self, key: str) -> None:
        await self._conn().execute("DELETE FROM memory_entries WHERE key = ?", (key,))
        await self._conn().commit()

    async def clear(self) -> None:
        await self._conn().execute("DELETE FROM memory_entries")
        await self._conn().commit()


def _row_to_entry(row: tuple[Any, ...]) -> MemoryEntry:
    key, value, meta_json, created_at, updated_at = row
    return MemoryEntry(
        key=key,
        value=value,
        metadata=json.loads(meta_json) if meta_json else None,
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
    )
