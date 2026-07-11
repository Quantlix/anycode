"""Storage contract for durable agent runs."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from anycode.types import RunRecord, RunRetentionPolicy, RunStatus, TranscriptEvent, TranscriptEventKind, TurnCheckpoint, WakeCondition


@runtime_checkable
class RunPayloadProtector(Protocol):
    """Protects serialized run-store bytes, typically using envelope encryption."""

    def protect(self, payload: bytes) -> bytes: ...

    def unprotect(self, payload: bytes) -> bytes: ...


@runtime_checkable
class RunStore(Protocol):
    """Persistence operations required by durable runners and schedulers."""

    def create_run(
        self,
        run_id: str,
        *,
        agent_name: str,
        model: str,
        metadata: dict[str, str] | None = None,
    ) -> RunRecord: ...

    def read_record(self, run_id: str) -> RunRecord | None: ...

    def update_status(self, run_id: str, status: RunStatus) -> RunRecord: ...

    def pause_run(self, run_id: str, wake: WakeCondition) -> RunRecord: ...

    def due_wakes(self, *, tolerance_seconds: float = 30.0) -> list[RunRecord]: ...

    def try_acquire_sweep_lock(self, run_id: str, *, stale_after_seconds: float = 60.0) -> bool: ...

    def release_sweep_lock(self, run_id: str) -> None: ...

    def touch_heartbeat(self, run_id: str) -> None: ...

    def list_runs(self) -> list[RunRecord]: ...

    def mark_interrupted_runs(self, stale_after_seconds: float) -> list[str]: ...

    def prune_runs(self, policy: RunRetentionPolicy, *, now: datetime | None = None) -> list[str]: ...

    def append_event(
        self,
        run_id: str,
        kind: TranscriptEventKind,
        payload: dict[str, object] | None = None,
    ) -> TranscriptEvent: ...

    def read_events(self, run_id: str, after_seq: int = 0) -> list[TranscriptEvent]: ...

    def save_checkpoint(self, checkpoint: TurnCheckpoint, *, keep_last: int = 3) -> object: ...

    def load_latest_checkpoint(self, run_id: str) -> TurnCheckpoint | None: ...
