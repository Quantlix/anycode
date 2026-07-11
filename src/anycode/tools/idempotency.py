"""Atomic idempotency claims for side-effecting tool calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from anycode.security.redaction import redact_sensitive
from anycode.types import ToolIdempotencyConfig, ToolResult

try:
    import aiosqlite
except ImportError:
    aiosqlite: Any = None

IdempotencyClaimOutcome = Literal["execute", "replay", "in_progress", "conflict"]


class IdempotencyClaim(BaseModel):
    """Decision returned by an atomic idempotency claim."""

    model_config = ConfigDict(frozen=True)
    outcome: IdempotencyClaimOutcome
    result: ToolResult | None = None


@runtime_checkable
class ToolIdempotencyStore(Protocol):
    """Atomic claims that remain unresolved until explicitly completed or deleted."""

    async def claim(self, tool_name: str, key: str, input_fingerprint: str) -> IdempotencyClaim: ...

    async def complete(self, tool_name: str, key: str, result: ToolResult) -> None: ...

    async def delete(self, tool_name: str, key: str) -> None: ...

    async def prune_completed(self, before: datetime) -> int: ...


class _MemoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_fingerprint: str
    result: ToolResult | None = None
    updated_at: datetime


class InMemoryToolIdempotencyStore:
    """Process-local atomic claim store for one executor or shared agent group."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], _MemoryRecord] = {}
        self._lock = asyncio.Lock()

    async def claim(self, tool_name: str, key: str, input_fingerprint: str) -> IdempotencyClaim:
        key = _storage_key(key)
        async with self._lock:
            record = self._records.get((tool_name, key))
            if record is None:
                self._records[(tool_name, key)] = _MemoryRecord(
                    input_fingerprint=input_fingerprint,
                    updated_at=datetime.now(UTC),
                )
                return IdempotencyClaim(outcome="execute")
            if record.input_fingerprint != input_fingerprint:
                return IdempotencyClaim(outcome="conflict")
            if record.result is None:
                return IdempotencyClaim(outcome="in_progress")
            return IdempotencyClaim(outcome="replay", result=record.result)

    async def complete(self, tool_name: str, key: str, result: ToolResult) -> None:
        key = _storage_key(key)
        async with self._lock:
            record = self._records.get((tool_name, key))
            if record is None:
                raise RuntimeError("Cannot complete an idempotency key that was not claimed")
            self._records[(tool_name, key)] = record.model_copy(update={"result": result, "updated_at": datetime.now(UTC)})

    async def delete(self, tool_name: str, key: str) -> None:
        key = _storage_key(key)
        async with self._lock:
            self._records.pop((tool_name, key), None)

    async def prune_completed(self, before: datetime) -> int:
        """Prune only completed outcomes explicitly marked retry-safe."""
        async with self._lock:
            stale = [
                record_key
                for record_key, record in self._records.items()
                if record.result is not None and record.result.retry_safe is True and record.updated_at < before
            ]
            for record_key in stale:
                del self._records[record_key]
            return len(stale)


class SQLiteToolIdempotencyStore:
    """Cross-process SQLite claim store for restart-safe tool deduplication."""

    def __init__(self, path: str | Path = ".anycode/tool-idempotency.db", *, redact_sensitive_data: bool = True) -> None:
        self._path = str(path)
        self._redact_sensitive_data = redact_sensitive_data
        self._db: Any = None
        self._setup_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()

    async def setup(self) -> None:
        if aiosqlite is None:
            raise ImportError('SQLiteToolIdempotencyStore requires: pip install "anycode-py[persistence]"')
        async with self._setup_lock:
            if self._db is not None:
                return
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            database = await aiosqlite.connect(self._path)
            try:
                await database.execute("PRAGMA journal_mode=WAL")
                await database.execute(
                    "CREATE TABLE IF NOT EXISTS tool_idempotency ("
                    "tool_name TEXT NOT NULL, key TEXT NOT NULL, input_fingerprint TEXT NOT NULL, "
                    "result_json TEXT, prunable INTEGER NOT NULL DEFAULT 0, "
                    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                    "PRIMARY KEY (tool_name, key))"
                )
                await database.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tool_idempotency_completed "
                    "ON tool_idempotency(updated_at) WHERE result_json IS NOT NULL AND prunable = 1"
                )
                await database.commit()
            except BaseException:
                await database.close()
                raise
            self._db = database

    async def teardown(self) -> None:
        async with self._setup_lock:
            if self._db is not None:
                await self._db.close()
                self._db = None

    async def claim(self, tool_name: str, key: str, input_fingerprint: str) -> IdempotencyClaim:
        """Prune only completed outcomes explicitly marked retry-safe."""
        await self.setup()
        key = _storage_key(key)
        async with self._operation_lock:
            database = self._connection()
            await database.execute("BEGIN IMMEDIATE")
            try:
                cursor = await database.execute(
                    "SELECT input_fingerprint, result_json FROM tool_idempotency WHERE tool_name = ? AND key = ?",
                    (tool_name, key),
                )
                row = await cursor.fetchone()
                if row is None:
                    now = datetime.now(UTC).isoformat()
                    await database.execute(
                        "INSERT INTO tool_idempotency "
                        "(tool_name, key, input_fingerprint, result_json, created_at, updated_at) "
                        "VALUES (?, ?, ?, NULL, ?, ?)",
                        (tool_name, key, input_fingerprint, now, now),
                    )
                    await database.commit()
                    return IdempotencyClaim(outcome="execute")
                stored_fingerprint, result_json = row
                await database.commit()
            except BaseException:
                await database.rollback()
                raise

        if stored_fingerprint != input_fingerprint:
            return IdempotencyClaim(outcome="conflict")
        if result_json is None:
            return IdempotencyClaim(outcome="in_progress")
        return IdempotencyClaim(outcome="replay", result=ToolResult.model_validate_json(result_json))

    async def complete(self, tool_name: str, key: str, result: ToolResult) -> None:
        await self.setup()
        key = _storage_key(key)
        payload: Mapping[str, object] = result.model_dump(mode="json")
        if self._redact_sensitive_data:
            payload = redact_sensitive(payload)
        async with self._operation_lock:
            database = self._connection()
            await database.execute("BEGIN IMMEDIATE")
            try:
                cursor = await database.execute(
                    "UPDATE tool_idempotency SET result_json = ?, prunable = ?, updated_at = ? WHERE tool_name = ? AND key = ?",
                    (json.dumps(payload), int(result.retry_safe is True), datetime.now(UTC).isoformat(), tool_name, key),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Cannot complete an idempotency key that was not claimed")
                await database.commit()
            except BaseException:
                await database.rollback()
                raise

    async def delete(self, tool_name: str, key: str) -> None:
        await self.setup()
        key = _storage_key(key)
        async with self._operation_lock:
            database = self._connection()
            await database.execute("BEGIN IMMEDIATE")
            try:
                await database.execute(
                    "DELETE FROM tool_idempotency WHERE tool_name = ? AND key = ?",
                    (tool_name, key),
                )
                await database.commit()
            except BaseException:
                await database.rollback()
                raise

    async def prune_completed(self, before: datetime) -> int:
        await self.setup()
        async with self._operation_lock:
            database = self._connection()
            await database.execute("BEGIN IMMEDIATE")
            try:
                cursor = await database.execute(
                    "DELETE FROM tool_idempotency WHERE result_json IS NOT NULL AND prunable = 1 AND updated_at < ?",
                    (before.isoformat(),),
                )
                await database.commit()
                return max(0, cursor.rowcount)
            except BaseException:
                await database.rollback()
                raise

    def _connection(self) -> Any:
        if self._db is None:
            raise RuntimeError("SQLite tool idempotency store is not initialized")
        return self._db


def create_tool_idempotency_store(config: ToolIdempotencyConfig | None = None) -> ToolIdempotencyStore:
    """Build a shared tool idempotency store from runtime configuration."""
    resolved = config or ToolIdempotencyConfig()
    if resolved.backend == "sqlite":
        if aiosqlite is None:
            raise ImportError('SQLite tool idempotency requires: pip install "anycode-py[persistence]"')
        return SQLiteToolIdempotencyStore(
            resolved.path,
            redact_sensitive_data=resolved.redact_sensitive_data,
        )
    return InMemoryToolIdempotencyStore()


def _storage_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
