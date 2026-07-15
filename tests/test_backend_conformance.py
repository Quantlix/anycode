"""Common failure and restart semantics across durability backends."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import JsonValue

from anycode.backends import (
    Admission,
    AmbiguousBackendResultError,
    DaprDurabilityBackend,
    DaprStateRecord,
    ExternalSignal,
    InMemoryDurabilityBackend,
    SQLiteDurabilityBackend,
    WakeRegistration,
    WorkItem,
)
from anycode.contracts import Event, Run, acknowledge_cancellation, request_cancellation, transition_run

_PAST = datetime(2000, 1, 1, tzinfo=UTC)


class SharedDaprTransport:
    store_name = "conformance-store"

    def __init__(self) -> None:
        self.value: dict[str, JsonValue] | None = None
        self.revision = 0

    async def get(self, key: str) -> DaprStateRecord:
        del key
        return DaprStateRecord(value=deepcopy(self.value), etag=str(self.revision) if self.value is not None else None)

    async def compare_and_set(self, key: str, value: dict[str, JsonValue], etag: str | None) -> bool:
        del key
        expected = str(self.revision) if self.value is not None else None
        if etag != expected:
            return False
        self.value = json.loads(json.dumps(value))
        self.revision += 1
        return True

    async def health(self) -> bool:
        return True


def _admission(run_id: str) -> Admission:
    now = datetime.now(UTC)
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


def _backends(tmp_path: Path) -> tuple[InMemoryDurabilityBackend, ...]:
    return (
        InMemoryDurabilityBackend(),
        SQLiteDurabilityBackend(tmp_path / "contract.db"),
        DaprDurabilityBackend(SharedDaprTransport()),
    )


@pytest.mark.parametrize("backend_index", (0, 1, 2))
async def test_cancellation_invalidates_inflight_claim_on_every_backend(tmp_path: Path, backend_index: int) -> None:
    backend = _backends(tmp_path)[backend_index]
    admission = _admission(f"cancel-{backend_index}")
    admitted = await backend.admit(admission)
    assert admitted.run is not None
    await backend.enqueue(WorkItem(id="work-1", run_id=admission.run.id, task_id="task-1", available_at=_PAST))
    claimed = await backend.claim("worker")
    assert claimed.claim is not None
    requested = request_cancellation(admitted.run, reason="operator")
    assert requested.event is not None
    canceled = await backend.request_cancellation(requested.run, requested.event, expected_sequence=1)
    acknowledged = acknowledge_cancellation(requested.run)
    assert acknowledged.event is not None and acknowledged.run is not None

    late = await backend.commit(claimed.claim, acknowledged.event, expected_sequence=2, run=acknowledged.run)

    assert canceled.accepted
    assert not late.accepted and late.stale_owner
    assert [event.type for event in await backend.read_events(admission.run.id)] == ["run.accepted", "cancellation.requested"]


@pytest.mark.parametrize("backend_index", (0, 1, 2))
async def test_ambiguous_commit_replay_never_duplicates_event(tmp_path: Path, backend_index: int) -> None:
    backend = _backends(tmp_path)[backend_index]
    admission = _admission(f"ambiguous-{backend_index}")
    admitted = await backend.admit(admission)
    assert admitted.run is not None
    await backend.enqueue(WorkItem(id="work-1", run_id=admission.run.id, task_id="task-1", available_at=_PAST))
    claimed = await backend.claim("worker")
    assert claimed.claim is not None
    queued = transition_run(admitted.run, "queued")
    assert queued.event is not None and queued.run is not None
    backend.inject_failure("commit", after_commit=True)

    with pytest.raises(AmbiguousBackendResultError):
        await backend.commit(claimed.claim, queued.event, expected_sequence=1, run=queued.run)
    replay = await backend.commit(claimed.claim, queued.event, expected_sequence=1, run=queued.run)

    assert not replay.accepted and replay.stale_owner
    assert [event.sequence for event in await backend.read_events(admission.run.id)] == [1, 2]


async def test_wakes_and_signals_survive_local_and_external_backend_restarts(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "restart.db"
    transport = SharedDaprTransport()
    pairs = (
        (SQLiteDurabilityBackend(sqlite_path), SQLiteDurabilityBackend(sqlite_path)),
        (DaprDurabilityBackend(transport), DaprDurabilityBackend(transport)),
    )
    for index, (writer, reader) in enumerate(pairs):
        admission = _admission(f"restart-{index}")
        assert (await writer.admit(admission)).admitted
        wake = WakeRegistration(
            id=f"wake-{index}", run_id=admission.run.id, wake_at=datetime.now(UTC) + timedelta(seconds=1), reason="timer"
        )
        signal = ExternalSignal(id=f"signal-{index}", run_id=admission.run.id, name="continue", payload={"approved": True})
        await writer.register_wake(wake)
        assert await writer.deliver_signal(signal)

        assert await reader.due_wakes(before=wake.wake_at) == (wake,)
        assert await reader.read_signals(admission.run.id) == (signal,)

