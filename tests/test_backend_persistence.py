"""Persistence and migration tests for SQLite and Dapr backends."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from anycode.backends import (
    Admission,
    DaprDurabilityBackend,
    DaprStateRecord,
    InMemoryDurabilityBackend,
    SQLiteDurabilityBackend,
    WorkItem,
    import_backend_snapshot,
)
from anycode.contracts import Event, Run


def _admission(run_id: str = "run-1") -> Admission:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    run = Run(id=run_id, correlation_id=f"corr-{run_id}", created_at=now, updated_at=now)
    event = Event(
        id=f"event-{run_id}-1",
        run_id=run_id,
        sequence=1,
        type="run.accepted",
        correlation_id=run.correlation_id,
        emitted_at=now,
    )
    return Admission(admission_key=f"admit-{run_id}", run=run, initial_event=event)


class FakeDaprTransport:
    store_name = "test-store"

    def __init__(self) -> None:
        self.value: dict[str, object] | None = None
        self.revision = 0
        self.fail_compare_count = 0
        self.available = True

    async def get(self, key: str) -> DaprStateRecord:
        del key
        value = deepcopy(self.value)
        return DaprStateRecord(value=value, etag=str(self.revision) if self.value is not None else None)  # type: ignore[arg-type]

    async def compare_and_set(self, key: str, value: dict[str, object], etag: str | None) -> bool:
        del key
        if self.fail_compare_count:
            self.fail_compare_count -= 1
            return False
        expected = str(self.revision) if self.value is not None else None
        if etag != expected:
            return False
        self.value = json.loads(json.dumps(value))
        self.revision += 1
        return True

    async def health(self) -> bool:
        return self.available


async def test_sqlite_backend_survives_new_instance(tmp_path: Path) -> None:
    path = tmp_path / "backend.db"
    first = SQLiteDurabilityBackend(path)
    admission = _admission()
    admitted = await first.admit(admission)
    await first.enqueue(
        WorkItem(id="work-1", run_id=admission.run.id, task_id="task-1", available_at=datetime(2000, 1, 1, tzinfo=UTC))
    )

    second = SQLiteDurabilityBackend(path)
    events = await second.read_events(admission.run.id)
    claimed = await second.claim("worker")
    duplicate = await second.admit(admission)

    assert admitted.admitted
    assert [event.id for event in events] == [admission.initial_event.id]
    assert claimed.claimed and claimed.claim is not None and claimed.claim.work.id == "work-1"
    assert duplicate.admitted and duplicate.duplicate
    assert second.capabilities().persistent and not second.capabilities().external


async def test_dapr_backend_retries_etag_conflict_and_shares_state() -> None:
    transport = FakeDaprTransport()
    transport.fail_compare_count = 1
    first = DaprDurabilityBackend(transport)
    admission = _admission()

    admitted = await first.admit(admission)
    second = DaprDurabilityBackend(transport)
    duplicate = await second.admit(admission)

    assert admitted.admitted
    assert duplicate.admitted and duplicate.duplicate
    assert second.capabilities().external
    assert second.version().store_name == "test-store"
    assert (await second.health()).status == "healthy"


async def test_portable_snapshot_import_is_idempotent() -> None:
    source = InMemoryDurabilityBackend()
    admission = _admission()
    assert (await source.admit(admission)).admitted
    snapshot = await source.export_run(admission.run.id)
    assert snapshot is not None
    target = InMemoryDurabilityBackend()

    imported = await import_backend_snapshot(target, snapshot, admission_key="migration-1")
    replayed = await import_backend_snapshot(target, snapshot, admission_key="migration-1")

    assert imported.admitted and not imported.duplicate
    assert replayed.admitted and replayed.duplicate
    assert await target.read_events(admission.run.id) == snapshot.events
