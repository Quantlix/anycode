"""Shared semantic tests for the pluggable durability contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from anycode.backends import (
    Admission,
    AmbiguousBackendResultError,
    ArtifactReferenceRecord,
    BackendUnavailableError,
    DurabilityBackend,
    ExternalSignal,
    InMemoryDurabilityBackend,
    WakeRegistration,
    WorkItem,
)
from anycode.contracts import (
    ArtifactReference,
    Checkpoint,
    Event,
    Run,
    request_cancellation,
    transition_run,
)
from anycode.contracts import (
    Task as SemanticTask,
)


class ManualClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 16, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _run(clock: ManualClock, run_id: str = "run-1") -> Run:
    return Run(id=run_id, correlation_id=f"corr-{run_id}", created_at=clock(), updated_at=clock())


def _event(run: Run, sequence: int, event_type: str, clock: ManualClock) -> Event:
    return Event(
        id=f"event-{run.id}-{sequence}",
        run_id=run.id,
        sequence=sequence,
        type=event_type,
        correlation_id=run.correlation_id,
        emitted_at=clock(),
    )


def _task(run: Run) -> SemanticTask:
    return SemanticTask(id="task-1", run_id=run.id, title="work", correlation_id=run.correlation_id)


async def _admit(backend: InMemoryDurabilityBackend, clock: ManualClock, run_id: str = "run-1") -> Run:
    run = _run(clock, run_id)
    admission = Admission(
        admission_key=f"admit-{run_id}", run=run, initial_event=_event(run, 1, "run.accepted", clock), tasks=(_task(run),)
    )
    result = await backend.admit(admission)
    assert result.admitted and result.run is not None
    return result.run


async def test_memory_backend_implements_contract_and_idempotent_admission() -> None:
    clock = ManualClock()
    backend = InMemoryDurabilityBackend(clock=clock)
    run = _run(clock)
    admission = Admission(admission_key="same-key", run=run, initial_event=_event(run, 1, "run.accepted", clock), tasks=(_task(run),))

    first = await backend.admit(admission)
    duplicate = await backend.admit(admission)
    conflict_run = _run(clock, "run-2")
    conflict = await backend.admit(
        Admission(admission_key="same-key", run=conflict_run, initial_event=_event(conflict_run, 1, "run.accepted", clock))
    )

    assert isinstance(backend, DurabilityBackend)
    assert first.admitted and not first.duplicate and first.run is not None and first.run.last_event_sequence == 1
    assert duplicate.admitted and duplicate.duplicate and duplicate.run == first.run
    assert not conflict.admitted and conflict.error is not None and conflict.error.code == "admission_key_conflict"
    assert backend.capabilities().fencing
    assert backend.version().contract_version == "1.0"
    assert (await backend.health()).status == "healthy"


async def test_lost_lease_reassigns_work_and_rejects_stale_commit() -> None:
    clock = ManualClock()
    backend = InMemoryDurabilityBackend(clock=clock)
    run = await _admit(backend, clock)
    work = WorkItem(id="work-1", run_id=run.id, task_id="task-1", available_at=clock())
    await backend.enqueue(work)

    first = await backend.claim("worker-a", lease_seconds=5)
    assert first.claimed and first.claim is not None
    clock.advance(6)
    second = await backend.claim("worker-b", lease_seconds=5)
    assert second.claimed and second.claim is not None
    assert second.claim.fencing_token > first.claim.fencing_token
    assert second.claim.generation > first.claim.generation

    transition = transition_run(run, "queued", now=clock())
    assert transition.ok and transition.run is not None and transition.event is not None
    stale = await backend.commit(first.claim, transition.event, expected_sequence=1, run=transition.run)
    accepted = await backend.commit(second.claim, transition.event, expected_sequence=1, run=transition.run)

    assert not stale.accepted and stale.stale_owner
    assert accepted.accepted and not accepted.stale_owner
    assert [event.sequence for event in await backend.read_events(run.id)] == [1, 2]


async def test_heartbeat_and_optimistic_event_append() -> None:
    clock = ManualClock()
    backend = InMemoryDurabilityBackend(clock=clock)
    run = await _admit(backend, clock)
    await backend.enqueue(WorkItem(id="work-1", run_id=run.id, task_id="task-1", available_at=clock()))
    claimed = await backend.claim("worker", lease_seconds=2)
    assert claimed.claim is not None
    clock.advance(1)
    renewed = await backend.heartbeat(claimed.claim, lease_seconds=10)
    assert renewed.claimed and renewed.claim is not None
    assert renewed.claim.lease_expires_at == clock() + timedelta(seconds=10)

    transition = transition_run(run, "queued", now=clock())
    assert transition.run is not None and transition.event is not None
    conflict = await backend.append_event(transition.event, expected_sequence=0, run=transition.run)
    appended = await backend.append_event(transition.event, expected_sequence=1, run=transition.run)

    assert not conflict.accepted and conflict.error is not None and conflict.error.code == "event_conflict"
    assert appended.accepted and appended.current_sequence == 2


async def test_checkpoint_wake_signal_artifact_and_export_round_trip() -> None:
    clock = ManualClock()
    backend = InMemoryDurabilityBackend(clock=clock)
    run = await _admit(backend, clock)
    snapshot_run = run.model_copy(update={"last_event_sequence": 1})
    checkpoint = Checkpoint(
        id="checkpoint-1",
        run_id=run.id,
        event_cursor=1,
        generation=run.generation,
        attempt=run.attempt,
        correlation_id=run.correlation_id,
        run=snapshot_run,
        tasks=(_task(run),),
    )
    saved = await backend.save_checkpoint(checkpoint)
    incompatible = checkpoint.model_copy(update={"event_cursor": 0, "run": snapshot_run.model_copy(update={"last_event_sequence": 0})})
    rejected = await backend.save_checkpoint(incompatible)

    wake = WakeRegistration(id="wake-1", run_id=run.id, wake_at=clock() + timedelta(seconds=5), reason="retry")
    await backend.register_wake(wake)
    signal = ExternalSignal(id="signal-1", run_id=run.id, name="approved", payload={"by": "operator"}, delivered_at=clock())
    assert await backend.deliver_signal(signal)
    assert not await backend.deliver_signal(signal)
    reference = ArtifactReferenceRecord(
        artifact_id="artifact-1",
        run_id=run.id,
        reference=ArtifactReference(uri="s3://bucket/object", provider="s3"),
        recorded_at=clock(),
    )
    await backend.record_artifact_reference(reference)
    clock.advance(5)
    exported = await backend.export_run(run.id)

    assert saved.accepted
    assert not rejected.accepted and rejected.error is not None and rejected.error.code == "checkpoint_incompatible"
    assert await backend.load_checkpoint(run.id) == checkpoint
    assert await backend.due_wakes() == (wake,)
    assert await backend.read_signals(run.id) == (signal,)
    assert await backend.read_artifact_references(run.id) == (reference,)
    assert exported is not None
    assert exported.checkpoint == checkpoint and exported.wakes == (wake,) and exported.signals == (signal,)


async def test_cancellation_removes_ready_work() -> None:
    clock = ManualClock()
    backend = InMemoryDurabilityBackend(clock=clock)
    run = await _admit(backend, clock)
    await backend.enqueue(WorkItem(id="work-1", run_id=run.id, task_id="task-1", available_at=clock()))
    requested = request_cancellation(run, reason="operator", now=clock())
    assert requested.event is not None

    result = await backend.request_cancellation(requested.run, requested.event, expected_sequence=1)

    assert result.accepted
    assert not (await backend.claim("worker")).claimed
    assert (await backend.export_run(run.id)).run.cancellation.status == "requested"  # type: ignore[union-attr]


async def test_fault_injection_models_partition_and_ambiguous_commit() -> None:
    clock = ManualClock()
    backend = InMemoryDurabilityBackend(clock=clock)
    backend.inject_failure("admit")
    run = _run(clock)
    admission = Admission(admission_key="admit", run=run, initial_event=_event(run, 1, "run.accepted", clock))
    with pytest.raises(BackendUnavailableError):
        await backend.admit(admission)

    backend.inject_failure("admit", after_commit=True)
    with pytest.raises(AmbiguousBackendResultError):
        await backend.admit(admission)
    replay = await backend.admit(admission)

    assert replay.admitted and replay.duplicate
