"""Portable export and import helpers for backend migrations."""

from __future__ import annotations

import json

from pydantic import JsonValue

from anycode.backends.models import Admission, AdmissionResult, BackendSnapshot
from anycode.backends.protocol import DurabilityBackend
from anycode.contracts.models import Cancellation, Checkpoint, ContractError, Event, Run, RunState
from anycode.runstore.protocol import RunStore
from anycode.types import RunStatus

_LEGACY_STATE_MAP: dict[RunStatus, RunState] = {
    "running": "running",
    "paused": "waiting",
    "interrupted": "waiting",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "canceled",
}


def _json_payload(payload: object) -> dict[str, JsonValue]:
    serialized = json.dumps(payload, default=str)
    value = json.loads(serialized)
    return value if isinstance(value, dict) else {"value": value}


def export_filesystem_run(store: RunStore, run_id: str) -> BackendSnapshot | None:
    """Translate a legacy ``RunStore`` record into the semantic backend format."""
    record = store.read_record(run_id)
    if record is None:
        return None
    legacy_events = store.read_events(run_id)
    correlation_id = record.metadata.get("correlation_id", run_id)
    events = tuple(
        Event(
            id=f"legacy:{run_id}:{sequence}",
            run_id=run_id,
            sequence=sequence,
            type=f"legacy.{legacy.kind}",
            payload={"legacy_sequence": legacy.seq, **_json_payload(legacy.payload)},
            correlation_id=correlation_id,
            emitted_at=legacy.ts,
        )
        for sequence, legacy in enumerate(legacy_events, start=1)
    )
    if not events:
        events = (
            Event(
                id=f"legacy:{run_id}:1",
                run_id=run_id,
                sequence=1,
                type="legacy.run_record",
                payload={"status": record.status},
                correlation_id=correlation_id,
                emitted_at=record.created_at,
            ),
        )
    state = _LEGACY_STATE_MAP[record.status]
    waiting_reason = "external_signal" if record.status == "paused" else "retry_backoff" if record.status == "interrupted" else None
    error = ContractError(code="legacy_run_failed", message="The legacy run record was marked failed.") if state == "failed" else None
    cancellation = (
        Cancellation(status="acknowledged", requested_at=record.updated_at, acknowledged_at=record.updated_at)
        if state == "canceled"
        else Cancellation()
    )
    run = Run(
        id=run_id,
        state=state,
        correlation_id=correlation_id,
        waiting_reason=waiting_reason,
        cancellation=cancellation,
        error=error,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_event_sequence=len(events),
        metadata={"legacy_agent_name": record.agent_name, "legacy_model": record.model, **record.metadata},
    )
    legacy_checkpoint = store.load_latest_checkpoint(run_id)
    checkpoint = None
    if legacy_checkpoint is not None:
        checkpoint = Checkpoint(
            id=f"legacy:{run_id}:turn:{legacy_checkpoint.turn}",
            run_id=run_id,
            event_cursor=len(events),
            generation=run.generation,
            attempt=run.attempt,
            correlation_id=correlation_id,
            run=run,
            metadata={"legacy_turn": legacy_checkpoint.turn, "legacy_checkpoint_created_at": legacy_checkpoint.created_at.isoformat()},
        )
    return BackendSnapshot(run=run, events=events, checkpoint=checkpoint)


async def import_backend_snapshot(
    backend: DurabilityBackend,
    snapshot: BackendSnapshot,
    *,
    admission_key: str,
) -> AdmissionResult:
    """Import a portable snapshot without duplicating an already admitted run."""
    if not snapshot.events or snapshot.events[0].sequence != 1:
        return AdmissionResult(
            admitted=False,
            error=ContractError(code="invalid_snapshot", message="A backend snapshot must begin with event sequence 1."),
        )
    admission_run = snapshot.run.model_copy(update={"last_event_sequence": 0})
    admitted = await backend.admit(
        Admission(
            admission_key=admission_key,
            run=admission_run,
            initial_event=snapshot.events[0],
            tasks=snapshot.tasks,
        )
    )
    if not admitted.admitted or admitted.duplicate:
        return admitted
    for event in snapshot.events[1:]:
        run = snapshot.run if event.sequence == snapshot.run.last_event_sequence else None
        appended = await backend.append_event(event, expected_sequence=event.sequence - 1, run=run)
        if not appended.accepted:
            return AdmissionResult(admitted=False, run=admitted.run, error=appended.error)
    if snapshot.checkpoint is not None:
        saved = await backend.save_checkpoint(snapshot.checkpoint)
        if not saved.accepted:
            return AdmissionResult(admitted=False, run=admitted.run, error=saved.error)
    for wake in snapshot.wakes:
        await backend.register_wake(wake)
    for signal in snapshot.signals:
        await backend.deliver_signal(signal)
    for reference in snapshot.artifact_references:
        await backend.record_artifact_reference(reference)
    return AdmissionResult(admitted=True, run=snapshot.run)
