"""Transport-neutral contract for local and external durability backends."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class DurabilityBackend(Protocol):
    """Async persistence operations required by distributed workers."""

    async def admit(self, admission: Admission) -> AdmissionResult: ...

    async def enqueue(self, work: WorkItem) -> None: ...

    async def claim(self, owner_id: str, *, lease_seconds: float = 30.0) -> ClaimResult: ...

    async def heartbeat(self, claim: WorkClaim, *, lease_seconds: float = 30.0) -> ClaimResult: ...

    async def append_event(
        self,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        tasks: tuple[Task, ...] = (),
    ) -> AppendResult: ...

    async def commit(
        self,
        claim: WorkClaim,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        task: Task | None = None,
    ) -> CommitResult: ...

    async def request_cancellation(self, run: Run, event: Event, *, expected_sequence: int) -> AppendResult: ...

    async def save_checkpoint(self, checkpoint: Checkpoint) -> AppendResult: ...

    async def load_checkpoint(self, run_id: str) -> Checkpoint | None: ...

    async def register_wake(self, wake: WakeRegistration) -> None: ...

    async def due_wakes(self, *, before: datetime | None = None) -> tuple[WakeRegistration, ...]: ...

    async def deliver_signal(self, signal: ExternalSignal) -> bool: ...

    async def read_signals(self, run_id: str) -> tuple[ExternalSignal, ...]: ...

    async def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[Event, ...]: ...

    async def record_artifact_reference(self, record: ArtifactReferenceRecord) -> None: ...

    async def read_artifact_references(self, run_id: str) -> tuple[ArtifactReferenceRecord, ...]: ...

    async def export_run(self, run_id: str) -> BackendSnapshot | None: ...

    async def health(self) -> BackendHealth: ...

    def capabilities(self) -> BackendCapabilities: ...

    def version(self) -> BackendVersion: ...
