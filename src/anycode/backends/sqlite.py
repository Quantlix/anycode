"""SQLite durability backend using the deterministic semantic state machine."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import JsonValue

from anycode.backends.memory import AmbiguousBackendResultError, InMemoryDurabilityBackend
from anycode.backends.models import (
    Admission,
    AdmissionResult,
    AppendResult,
    ArtifactReferenceRecord,
    BackendCapabilities,
    BackendHealth,
    BackendSnapshot,
    BackendVersion,
    ClaimResult,
    CommitResult,
    ExternalSignal,
    WakeRegistration,
    WorkClaim,
    WorkItem,
)
from anycode.contracts.models import Checkpoint, Event, Run, Task

_FORMAT_VERSION = 1
_IMPLEMENTATION_VERSION = "1.0"
_STATE_ROW_ID = 1
_BUSY_TIMEOUT_MS = 5_000
T = TypeVar("T")


class UnsupportedBackendStateVersionError(RuntimeError):
    """Raised when a SQLite state record requires an unknown migration."""


class SQLiteDurabilityBackend(InMemoryDurabilityBackend):
    """Persistent local backend with transactional, cross-process serialization.

    A single versioned state row intentionally favors correctness and migration
    simplicity over write throughput. The capability report makes that tradeoff
    visible so high-volume deployments can select an external backend.
    """

    def __init__(self, path: str | Path = ".anycode/backend.db") -> None:
        super().__init__()
        self._path = Path(path)
        self._database_lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _load_driver() -> Any:
        try:
            import aiosqlite
        except ImportError as error:
            raise ImportError("SQLiteDurabilityBackend requires the 'persistence' extra: uv add 'anycode-py[persistence]'.") from error
        return aiosqlite

    async def _ensure_schema(self, database: Any) -> None:
        await database.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        await database.execute("PRAGMA journal_mode = WAL")
        await database.execute(
            """
            CREATE TABLE IF NOT EXISTS anycode_backend_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                format_version INTEGER NOT NULL,
                revision INTEGER NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        await database.execute(
            "INSERT OR IGNORE INTO anycode_backend_state (id, format_version, revision, payload) VALUES (?, ?, 0, '{}')",
            (_STATE_ROW_ID, _FORMAT_VERSION),
        )
        await database.commit()

    async def _read_row(self, database: Any) -> tuple[int, dict[str, JsonValue]]:
        cursor = await database.execute("SELECT format_version, revision, payload FROM anycode_backend_state WHERE id = ?", (_STATE_ROW_ID,))
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise RuntimeError("SQLite backend state row is missing")
        format_version, revision, raw_payload = row
        if format_version != _FORMAT_VERSION:
            raise UnsupportedBackendStateVersionError(f"Unsupported backend state version {format_version}; supported version is {_FORMAT_VERSION}.")
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict):
            raise ValueError("SQLite backend state payload must be a JSON object")
        return int(revision), payload

    async def _write_row(self, database: Any, revision: int) -> None:
        payload = json.dumps(self._dump_state(), sort_keys=True, separators=(",", ":"))
        cursor = await database.execute(
            "UPDATE anycode_backend_state SET revision = ?, payload = ? WHERE id = ? AND revision = ?",
            (revision + 1, payload, _STATE_ROW_ID, revision),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("SQLite backend state changed during a serialized transaction")

    async def _with_state(self, operation: Callable[[], Awaitable[T]], *, write: bool) -> T:
        driver = self._load_driver()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with self._database_lock, driver.connect(self._path) as database:
            await self._ensure_schema(database)
            await database.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            revision, payload = await self._read_row(database)
            self._restore_state(payload)
            try:
                result = await operation()
            except AmbiguousBackendResultError:
                if write:
                    await self._write_row(database, revision)
                    await database.commit()
                else:
                    await database.rollback()
                raise
            except Exception:
                await database.rollback()
                raise
            if write:
                await self._write_row(database, revision)
            await database.commit()
            return result

    async def admit(self, admission: Admission) -> AdmissionResult:
        operation = super().admit
        return await self._with_state(lambda: operation(admission), write=True)

    async def enqueue(self, work: WorkItem) -> None:
        operation = super().enqueue
        await self._with_state(lambda: operation(work), write=True)

    async def claim(self, owner_id: str, *, lease_seconds: float = 30.0) -> ClaimResult:
        operation = super().claim
        return await self._with_state(lambda: operation(owner_id, lease_seconds=lease_seconds), write=True)

    async def heartbeat(self, claim: WorkClaim, *, lease_seconds: float = 30.0) -> ClaimResult:
        operation = super().heartbeat
        return await self._with_state(lambda: operation(claim, lease_seconds=lease_seconds), write=True)

    async def append_event(
        self,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        tasks: tuple[Task, ...] = (),
    ) -> AppendResult:
        operation = super().append_event
        return await self._with_state(lambda: operation(event, expected_sequence=expected_sequence, run=run, tasks=tasks), write=True)

    async def commit(
        self,
        claim: WorkClaim,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        task: Task | None = None,
    ) -> CommitResult:
        operation = super().commit
        return await self._with_state(lambda: operation(claim, event, expected_sequence=expected_sequence, run=run, task=task), write=True)

    async def request_cancellation(self, run: Run, event: Event, *, expected_sequence: int) -> AppendResult:
        operation = super().request_cancellation
        return await self._with_state(lambda: operation(run, event, expected_sequence=expected_sequence), write=True)

    async def save_checkpoint(self, checkpoint: Checkpoint) -> AppendResult:
        operation = super().save_checkpoint
        return await self._with_state(lambda: operation(checkpoint), write=True)

    async def load_checkpoint(self, run_id: str) -> Checkpoint | None:
        operation = super().load_checkpoint
        return await self._with_state(lambda: operation(run_id), write=False)

    async def register_wake(self, wake: WakeRegistration) -> None:
        operation = super().register_wake
        await self._with_state(lambda: operation(wake), write=True)

    async def due_wakes(self, *, before: datetime | None = None) -> tuple[WakeRegistration, ...]:
        operation = super().due_wakes
        return await self._with_state(lambda: operation(before=before), write=False)

    async def deliver_signal(self, signal: ExternalSignal) -> bool:
        operation = super().deliver_signal
        return await self._with_state(lambda: operation(signal), write=True)

    async def read_signals(self, run_id: str) -> tuple[ExternalSignal, ...]:
        operation = super().read_signals
        return await self._with_state(lambda: operation(run_id), write=False)

    async def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[Event, ...]:
        operation = super().read_events
        return await self._with_state(lambda: operation(run_id, after_sequence=after_sequence), write=False)

    async def record_artifact_reference(self, record: ArtifactReferenceRecord) -> None:
        operation = super().record_artifact_reference
        await self._with_state(lambda: operation(record), write=True)

    async def read_artifact_references(self, run_id: str) -> tuple[ArtifactReferenceRecord, ...]:
        operation = super().read_artifact_references
        return await self._with_state(lambda: operation(run_id), write=False)

    async def export_run(self, run_id: str) -> BackendSnapshot | None:
        operation = super().export_run
        return await self._with_state(lambda: operation(run_id), write=False)

    async def health(self) -> BackendHealth:
        try:
            operation = super().health
            health = await self._with_state(operation, write=False)
        except Exception as error:
            return BackendHealth(status="unavailable", message=str(error))
        return health.model_copy(update={"details": {**health.details, "path": str(self._path)}})

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend="sqlite",
            persistent=True,
            external=False,
            limitations=("A database uses one coarse-grained writer transaction; select Dapr for distributed deployment.",),
        )

    def version(self) -> BackendVersion:
        return BackendVersion(
            backend="sqlite",
            implementation_version=_IMPLEMENTATION_VERSION,
            store_name=self._path.name,
            store_version=str(_FORMAT_VERSION),
        )
