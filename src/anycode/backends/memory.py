"""Deterministic in-memory durability backend with fault injection."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import JsonValue

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
from anycode.contracts.models import Checkpoint, ContractError, Event, Run, Task

_IMPLEMENTATION_VERSION = "1.0"


class BackendUnavailableError(RuntimeError):
    """Raised when an injected or external backend partition prevents a call."""


class AmbiguousBackendResultError(BackendUnavailableError):
    """Raised after a write commits but its acknowledgement is lost."""


@dataclass
class _FaultCounter:
    remaining: int
    after_commit: bool


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _admission_digest(admission: Admission) -> str:
    payload = admission.model_dump(mode="json", exclude={"admission_key"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class InMemoryDurabilityBackend:
    """Reference backend whose clock and failures can be controlled by tests."""

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self._runs: dict[str, Run] = {}
        self._tasks: dict[tuple[str, str], Task] = {}
        self._events: dict[str, list[Event]] = {}
        self._admissions: dict[str, tuple[str, str]] = {}
        self._ready: dict[str, WorkItem] = {}
        self._claims: dict[str, WorkClaim] = {}
        self._fencing_tokens: dict[str, int] = {}
        self._checkpoints: dict[str, Checkpoint] = {}
        self._wakes: dict[str, WakeRegistration] = {}
        self._signals: dict[str, list[ExternalSignal]] = {}
        self._signal_ids: set[str] = set()
        self._artifacts: dict[str, list[ArtifactReferenceRecord]] = {}
        self._faults: dict[str, _FaultCounter] = {}

    def inject_failure(self, operation: str, *, times: int = 1, after_commit: bool = False) -> None:
        """Fail an operation deterministically before or after its state change."""
        if times < 1:
            raise ValueError("times must be at least 1")
        self._faults[operation] = _FaultCounter(remaining=times, after_commit=after_commit)

    def clear_failures(self) -> None:
        self._faults.clear()

    def _dump_state(self) -> dict[str, JsonValue]:
        """Return the complete semantic state in a persistence-neutral shape."""
        return {
            "runs": [run.model_dump(mode="json") for run in self._runs.values()],
            "tasks": [task.model_dump(mode="json") for task in self._tasks.values()],
            "events": [event.model_dump(mode="json") for events in self._events.values() for event in events],
            "admissions": {key: [digest, run_id] for key, (digest, run_id) in self._admissions.items()},
            "ready": [work.model_dump(mode="json") for work in self._ready.values()],
            "claims": [claim.model_dump(mode="json") for claim in self._claims.values()],
            "fencing_tokens": dict(self._fencing_tokens),
            "checkpoints": [checkpoint.model_dump(mode="json") for checkpoint in self._checkpoints.values()],
            "wakes": [wake.model_dump(mode="json") for wake in self._wakes.values()],
            "signals": [signal.model_dump(mode="json") for signals in self._signals.values() for signal in signals],
            "artifact_references": [record.model_dump(mode="json") for records in self._artifacts.values() for record in records],
        }

    def _restore_state(self, payload: dict[str, JsonValue] | None) -> None:
        """Replace process-local state from a validated persistence payload."""
        data = payload or {}
        runs = [Run.model_validate(item) for item in data.get("runs", [])]  # type: ignore[arg-type]
        tasks = [Task.model_validate(item) for item in data.get("tasks", [])]  # type: ignore[arg-type]
        events = [Event.model_validate(item) for item in data.get("events", [])]  # type: ignore[arg-type]
        ready = [WorkItem.model_validate(item) for item in data.get("ready", [])]  # type: ignore[arg-type]
        claims = [WorkClaim.model_validate(item) for item in data.get("claims", [])]  # type: ignore[arg-type]
        checkpoints = [Checkpoint.model_validate(item) for item in data.get("checkpoints", [])]  # type: ignore[arg-type]
        wakes = [WakeRegistration.model_validate(item) for item in data.get("wakes", [])]  # type: ignore[arg-type]
        signals = [ExternalSignal.model_validate(item) for item in data.get("signals", [])]  # type: ignore[arg-type]
        artifacts = [ArtifactReferenceRecord.model_validate(item) for item in data.get("artifact_references", [])]  # type: ignore[arg-type]
        raw_admissions = data.get("admissions", {})
        raw_tokens = data.get("fencing_tokens", {})

        self._runs = {run.id: run for run in runs}
        self._tasks = {(task.run_id, task.id): task for task in tasks}
        self._events = {run_id: [] for run_id in self._runs}
        for event in events:
            self._events.setdefault(event.run_id, []).append(event)
        self._admissions = {
            str(key): (str(value[0]), str(value[1]))
            for key, value in (raw_admissions.items() if isinstance(raw_admissions, dict) else ())
            if isinstance(value, list) and len(value) == 2
        }
        self._ready = {work.id: work for work in ready}
        self._claims = {claim.work.id: claim for claim in claims}
        self._fencing_tokens = {
            str(key): int(value) for key, value in (raw_tokens.items() if isinstance(raw_tokens, dict) else ()) if isinstance(value, int)
        }
        self._checkpoints = {checkpoint.run_id: checkpoint for checkpoint in checkpoints}
        self._wakes = {wake.id: wake for wake in wakes}
        self._signals = {}
        self._signal_ids = set()
        for signal in signals:
            self._signals.setdefault(signal.run_id, []).append(signal)
            self._signal_ids.add(signal.id)
        self._artifacts = {}
        for record in artifacts:
            self._artifacts.setdefault(record.run_id, []).append(record)

    def _maybe_fail(self, operation: str, *, after_commit: bool) -> None:
        fault = self._faults.get(operation)
        if fault is None or fault.after_commit != after_commit:
            return
        fault.remaining -= 1
        if fault.remaining == 0:
            del self._faults[operation]
        error_type = AmbiguousBackendResultError if after_commit else BackendUnavailableError
        raise error_type(f"Injected {'post-commit ' if after_commit else ''}failure for {operation}")

    async def admit(self, admission: Admission) -> AdmissionResult:
        self._maybe_fail("admit", after_commit=False)
        digest = _admission_digest(admission)
        async with self._lock:
            prior = self._admissions.get(admission.admission_key)
            if prior is not None:
                prior_digest, run_id = prior
                if prior_digest != digest:
                    return AdmissionResult(
                        admitted=False,
                        error=ContractError(code="admission_key_conflict", message="Admission key was reused with a different payload."),
                    )
                return AdmissionResult(admitted=True, duplicate=True, run=self._runs[run_id])
            if admission.run.id in self._runs:
                return AdmissionResult(
                    admitted=False,
                    error=ContractError(code="run_conflict", message="Run id is already present in the backend."),
                )
            stored_run = admission.run.model_copy(update={"last_event_sequence": 1, "updated_at": admission.initial_event.emitted_at})
            self._runs[stored_run.id] = stored_run
            self._events[stored_run.id] = [admission.initial_event]
            self._tasks.update({(stored_run.id, task.id): task for task in admission.tasks})
            self._admissions[admission.admission_key] = (digest, stored_run.id)
        self._maybe_fail("admit", after_commit=True)
        return AdmissionResult(admitted=True, run=stored_run)

    async def enqueue(self, work: WorkItem) -> None:
        self._maybe_fail("enqueue", after_commit=False)
        async with self._lock:
            if work.run_id not in self._runs:
                raise KeyError(f"Unknown run: {work.run_id}")
            self._ready[work.id] = work
        self._maybe_fail("enqueue", after_commit=True)

    async def claim(self, owner_id: str, *, lease_seconds: float = 30.0) -> ClaimResult:
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        self._maybe_fail("claim", after_commit=False)
        async with self._lock:
            now = self._clock()
            eligible = [
                work
                for work in self._ready.values()
                if work.available_at <= now and (work.id not in self._claims or self._claims[work.id].lease_expires_at <= now)
            ]
            if not eligible:
                return ClaimResult(claimed=False)
            work = min(eligible, key=lambda item: (-item.priority, item.available_at, item.id))
            prior = self._claims.get(work.id)
            generation = max(work.generation, (prior.generation + 1) if prior else work.generation)
            token = self._fencing_tokens.get(work.id, 0) + 1
            self._fencing_tokens[work.id] = token
            claim = WorkClaim(
                work=work,
                owner_id=owner_id,
                generation=generation,
                fencing_token=token,
                claimed_at=now,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            self._claims[work.id] = claim
        self._maybe_fail("claim", after_commit=True)
        return ClaimResult(claimed=True, claim=claim)

    async def heartbeat(self, claim: WorkClaim, *, lease_seconds: float = 30.0) -> ClaimResult:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        self._maybe_fail("heartbeat", after_commit=False)
        async with self._lock:
            current = self._claims.get(claim.work.id)
            now = self._clock()
            if current is None or not self._claim_matches(current, claim) or current.lease_expires_at <= now:
                return ClaimResult(
                    claimed=False,
                    error=ContractError(code="stale_owner", message="The work lease is no longer owned by this claimant."),
                )
            renewed = current.model_copy(update={"lease_expires_at": now + timedelta(seconds=lease_seconds)})
            self._claims[claim.work.id] = renewed
        self._maybe_fail("heartbeat", after_commit=True)
        return ClaimResult(claimed=True, claim=renewed)

    async def append_event(
        self,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        tasks: tuple[Task, ...] = (),
    ) -> AppendResult:
        self._maybe_fail("append_event", after_commit=False)
        async with self._lock:
            result = self._append_locked(event, expected_sequence=expected_sequence, run=run, tasks=tasks)
        if result.accepted:
            self._maybe_fail("append_event", after_commit=True)
        return result

    def _append_locked(
        self,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None,
        tasks: tuple[Task, ...],
    ) -> AppendResult:
        events = self._events.get(event.run_id)
        if events is None:
            return AppendResult(
                accepted=False,
                current_sequence=0,
                error=ContractError(code="run_not_found", message="Run is not present in the backend."),
            )
        current_sequence = events[-1].sequence if events else 0
        if expected_sequence != current_sequence or event.sequence != current_sequence + 1:
            return AppendResult(
                accepted=False,
                current_sequence=current_sequence,
                error=ContractError(code="event_conflict", message="Event sequence did not match the materialized run cursor.", retryable=True),
            )
        if run is not None and (run.id != event.run_id or run.last_event_sequence != event.sequence):
            return AppendResult(
                accepted=False,
                current_sequence=current_sequence,
                error=ContractError(code="view_conflict", message="Materialized run view does not match the appended event."),
            )
        if any(task.run_id != event.run_id for task in tasks):
            return AppendResult(
                accepted=False,
                current_sequence=current_sequence,
                error=ContractError(code="view_conflict", message="Materialized task belongs to a different run."),
            )
        events.append(event)
        if run is not None:
            self._runs[event.run_id] = run
        else:
            current_run = self._runs[event.run_id]
            self._runs[event.run_id] = current_run.model_copy(update={"last_event_sequence": event.sequence, "updated_at": event.emitted_at})
        self._tasks.update({(event.run_id, task.id): task for task in tasks})
        return AppendResult(accepted=True, current_sequence=event.sequence, event=event)

    async def commit(
        self,
        claim: WorkClaim,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        task: Task | None = None,
    ) -> CommitResult:
        self._maybe_fail("commit", after_commit=False)
        async with self._lock:
            current = self._claims.get(claim.work.id)
            if current is None or not self._claim_matches(current, claim) or current.lease_expires_at <= self._clock():
                current_sequence = len(self._events.get(claim.work.run_id, ()))
                return CommitResult(
                    accepted=False,
                    stale_owner=True,
                    current_sequence=current_sequence,
                    error=ContractError(code="stale_owner", message="A stale or expired worker cannot commit results."),
                )
            tasks = (task,) if task is not None else ()
            appended = self._append_locked(event, expected_sequence=expected_sequence, run=run, tasks=tasks)
            if not appended.accepted:
                return CommitResult(**appended.model_dump(), stale_owner=False)
            self._ready.pop(claim.work.id, None)
            self._claims.pop(claim.work.id, None)
            result = CommitResult(**appended.model_dump(), stale_owner=False)
        self._maybe_fail("commit", after_commit=True)
        return result

    @staticmethod
    def _claim_matches(current: WorkClaim | None, submitted: WorkClaim) -> bool:
        return bool(
            current
            and current.owner_id == submitted.owner_id
            and current.generation == submitted.generation
            and current.fencing_token == submitted.fencing_token
        )

    async def request_cancellation(self, run: Run, event: Event, *, expected_sequence: int) -> AppendResult:
        self._maybe_fail("request_cancellation", after_commit=False)
        async with self._lock:
            result = self._append_locked(event, expected_sequence=expected_sequence, run=run, tasks=())
            if result.accepted:
                work_ids = [work_id for work_id, work in self._ready.items() if work.run_id == run.id]
                for work_id in work_ids:
                    self._ready.pop(work_id, None)
                    self._claims.pop(work_id, None)
        if result.accepted:
            self._maybe_fail("request_cancellation", after_commit=True)
        return result

    async def save_checkpoint(self, checkpoint: Checkpoint) -> AppendResult:
        self._maybe_fail("save_checkpoint", after_commit=False)
        async with self._lock:
            events = self._events.get(checkpoint.run_id)
            current_sequence = events[-1].sequence if events else 0
            if events is None:
                return AppendResult(
                    accepted=False,
                    current_sequence=0,
                    error=ContractError(code="run_not_found", message="Run is not present in the backend."),
                )
            if checkpoint.event_cursor != current_sequence:
                return AppendResult(
                    accepted=False,
                    current_sequence=current_sequence,
                    error=ContractError(code="checkpoint_incompatible", message="Checkpoint cursor does not match event history."),
                )
            self._checkpoints[checkpoint.run_id] = checkpoint
            result = AppendResult(accepted=True, current_sequence=current_sequence)
        self._maybe_fail("save_checkpoint", after_commit=True)
        return result

    async def load_checkpoint(self, run_id: str) -> Checkpoint | None:
        self._maybe_fail("load_checkpoint", after_commit=False)
        async with self._lock:
            return self._checkpoints.get(run_id)

    async def register_wake(self, wake: WakeRegistration) -> None:
        self._maybe_fail("register_wake", after_commit=False)
        async with self._lock:
            if wake.run_id not in self._runs:
                raise KeyError(f"Unknown run: {wake.run_id}")
            self._wakes[wake.id] = wake
        self._maybe_fail("register_wake", after_commit=True)

    async def due_wakes(self, *, before: datetime | None = None) -> tuple[WakeRegistration, ...]:
        self._maybe_fail("due_wakes", after_commit=False)
        horizon = before or self._clock()
        async with self._lock:
            return tuple(sorted((wake for wake in self._wakes.values() if wake.wake_at <= horizon), key=lambda wake: (wake.wake_at, wake.id)))

    async def deliver_signal(self, signal: ExternalSignal) -> bool:
        self._maybe_fail("deliver_signal", after_commit=False)
        async with self._lock:
            if signal.run_id not in self._runs:
                raise KeyError(f"Unknown run: {signal.run_id}")
            if signal.id in self._signal_ids:
                return False
            self._signal_ids.add(signal.id)
            self._signals.setdefault(signal.run_id, []).append(signal)
        self._maybe_fail("deliver_signal", after_commit=True)
        return True

    async def read_signals(self, run_id: str) -> tuple[ExternalSignal, ...]:
        self._maybe_fail("read_signals", after_commit=False)
        async with self._lock:
            return tuple(self._signals.get(run_id, ()))

    async def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[Event, ...]:
        self._maybe_fail("read_events", after_commit=False)
        async with self._lock:
            return tuple(event for event in self._events.get(run_id, ()) if event.sequence > after_sequence)

    async def record_artifact_reference(self, record: ArtifactReferenceRecord) -> None:
        self._maybe_fail("record_artifact_reference", after_commit=False)
        async with self._lock:
            if record.run_id not in self._runs:
                raise KeyError(f"Unknown run: {record.run_id}")
            records = self._artifacts.setdefault(record.run_id, [])
            if not any(existing.artifact_id == record.artifact_id for existing in records):
                records.append(record)
        self._maybe_fail("record_artifact_reference", after_commit=True)

    async def read_artifact_references(self, run_id: str) -> tuple[ArtifactReferenceRecord, ...]:
        self._maybe_fail("read_artifact_references", after_commit=False)
        async with self._lock:
            return tuple(self._artifacts.get(run_id, ()))

    async def export_run(self, run_id: str) -> BackendSnapshot | None:
        self._maybe_fail("export_run", after_commit=False)
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return BackendSnapshot(
                run=run,
                tasks=tuple(task for (stored_run_id, _), task in self._tasks.items() if stored_run_id == run_id),
                events=tuple(self._events.get(run_id, ())),
                checkpoint=self._checkpoints.get(run_id),
                wakes=tuple(wake for wake in self._wakes.values() if wake.run_id == run_id),
                signals=tuple(self._signals.get(run_id, ())),
                artifact_references=tuple(self._artifacts.get(run_id, ())),
            )

    async def health(self) -> BackendHealth:
        try:
            self._maybe_fail("health", after_commit=False)
        except BackendUnavailableError as error:
            return BackendHealth(status="unavailable", message=str(error))
        return BackendHealth(status="healthy", details={"runs": len(self._runs), "ready_work": len(self._ready)})

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(backend="memory", persistent=False, external=False, limitations=("State is lost when the process exits.",))

    def version(self) -> BackendVersion:
        return BackendVersion(backend="memory", implementation_version=_IMPLEMENTATION_VERSION)
