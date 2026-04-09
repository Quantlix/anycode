"""Checkpoint storage backends — filesystem and SQLite."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from anycode.checkpoint.serializer import deserialize_checkpoint, serialize_checkpoint
from anycode.constants import DEFAULT_ENCODING
from anycode.types import CheckpointData

try:
    import aiosqlite
except ImportError:
    aiosqlite: Any = None


class FilesystemCheckpointStore:
    """JSON file–based checkpoint store with atomic writes."""

    def __init__(self, path: str = ".anycode/checkpoints") -> None:
        self._root = Path(path)

    async def save(self, checkpoint: CheckpointData) -> None:
        directory = self._root / checkpoint.workflow_id
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        target = directory / f"{checkpoint.id}.json"
        tmp = target.with_suffix(".tmp")
        content = serialize_checkpoint(checkpoint)
        await asyncio.to_thread(_atomic_write, tmp, target, content)

    async def load(self, checkpoint_id: str) -> CheckpointData | None:
        for path in await asyncio.to_thread(lambda: list(self._root.rglob(f"{checkpoint_id}.json"))):
            raw = await asyncio.to_thread(path.read_text, DEFAULT_ENCODING)
            return deserialize_checkpoint(raw)
        return None

    async def latest(self, workflow_id: str) -> CheckpointData | None:
        directory = self._root / workflow_id
        exists = await asyncio.to_thread(directory.is_dir)
        if not exists:
            return None
        files = await asyncio.to_thread(lambda: sorted(directory.glob("*.json")))
        if not files:
            return None
        raw = await asyncio.to_thread(files[-1].read_text, DEFAULT_ENCODING)
        return deserialize_checkpoint(raw)

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        directory = self._root / workflow_id
        exists = await asyncio.to_thread(directory.is_dir)
        if not exists:
            return []
        files = await asyncio.to_thread(lambda: sorted(directory.glob("*.json")))
        return [f.stem for f in files]

    async def delete(self, checkpoint_id: str) -> None:
        for path in await asyncio.to_thread(lambda: list(self._root.rglob(f"{checkpoint_id}.json"))):
            await asyncio.to_thread(path.unlink, True)

    async def prune(self, workflow_id: str, keep_last: int) -> None:
        directory = self._root / workflow_id
        exists = await asyncio.to_thread(directory.is_dir)
        if not exists:
            return
        files = await asyncio.to_thread(lambda: sorted(directory.glob("*.json")))
        to_delete = files[: max(0, len(files) - keep_last)]
        for f in to_delete:
            await asyncio.to_thread(f.unlink, True)


class SQLiteCheckpointStore:
    """SQLite-backed checkpoint store using aiosqlite."""

    def __init__(self, path: str = ".anycode/checkpoints.db") -> None:
        self._path = path
        self._db: Any = None

    async def setup(self) -> None:
        if aiosqlite is None:
            raise ImportError('SQLiteCheckpointStore requires: pip install "anycode-py[persistence]"')
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            "  id TEXT PRIMARY KEY,"
            "  workflow_id TEXT NOT NULL,"
            "  wave_index INTEGER NOT NULL,"
            "  data TEXT NOT NULL,"
            "  created_at TEXT NOT NULL"
            ")"
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_cp_workflow ON checkpoints(workflow_id, created_at)")
        await self._db.commit()

    async def teardown(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _conn(self) -> Any:
        if self._db is None:
            raise RuntimeError("SQLiteCheckpointStore not initialized. Call setup() first.")
        return self._db

    async def save(self, checkpoint: CheckpointData) -> None:
        content = serialize_checkpoint(checkpoint)
        await self._conn().execute(
            "INSERT OR REPLACE INTO checkpoints (id, workflow_id, wave_index, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (checkpoint.id, checkpoint.workflow_id, checkpoint.wave_index, content, checkpoint.created_at.isoformat()),
        )
        await self._conn().commit()

    async def load(self, checkpoint_id: str) -> CheckpointData | None:
        cursor = await self._conn().execute("SELECT data FROM checkpoints WHERE id = ?", (checkpoint_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return deserialize_checkpoint(row[0])

    async def latest(self, workflow_id: str) -> CheckpointData | None:
        cursor = await self._conn().execute("SELECT data FROM checkpoints WHERE workflow_id = ? ORDER BY created_at DESC LIMIT 1", (workflow_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return deserialize_checkpoint(row[0])

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        cursor = await self._conn().execute("SELECT id FROM checkpoints WHERE workflow_id = ? ORDER BY created_at", (workflow_id,))
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def delete(self, checkpoint_id: str) -> None:
        await self._conn().execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
        await self._conn().commit()

    async def prune(self, workflow_id: str, keep_last: int) -> None:
        cursor = await self._conn().execute("SELECT id FROM checkpoints WHERE workflow_id = ? ORDER BY created_at DESC", (workflow_id,))
        rows = list(await cursor.fetchall())
        to_delete = [r[0] for r in rows[keep_last:]]
        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            await self._conn().execute(f"DELETE FROM checkpoints WHERE id IN ({placeholders})", to_delete)
            await self._conn().commit()


def _atomic_write(tmp: Path, target: Path, content: str) -> None:
    tmp.write_text(content, encoding=DEFAULT_ENCODING)
    os.replace(str(tmp), str(target))
