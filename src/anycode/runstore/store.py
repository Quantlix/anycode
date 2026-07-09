"""Filesystem-backed run store.

Layout (one directory per run):

    <root>/<run_id>/
        meta.json           RunRecord — status, heartbeat, timestamps
        transcript.jsonl    append-only TranscriptEvent log (source of truth)
        checkpoints/
            turn-000042.json    TurnCheckpoint — fast-resume state

Write discipline: `meta.json` and checkpoints use temp-file + `os.replace`
(atomic on POSIX and Windows); the transcript is append-only and never
rewritten. A torn tail line (crash mid-append) is skipped with a warning on
read, never a fatal error. There is no separate index — the directory scan is
the ground truth, so nothing can desync.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from anycode.checkpoint.serializer import _deserialize_message, _serialize_message
from anycode.types import (
    BudgetSnapshot,
    ContextManifest,
    LifecycleEvent,
    QualityGateDecision,
    RunRecord,
    RunStatus,
    TokenUsage,
    TranscriptEvent,
    TranscriptEventKind,
    TurnCheckpoint,
    VerificationResult,
    WakeCondition,
)

logger = logging.getLogger(__name__)

_META = "meta.json"
_TRANSCRIPT = "transcript.jsonl"
_CHECKPOINT_DIR = "checkpoints"


def _now() -> datetime:
    return datetime.now(UTC)


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


class FilesystemRunStore:
    """Durable per-run persistence on the local filesystem (stdlib only)."""

    def __init__(self, root: str | Path = ".anycode/runs") -> None:
        self._root = Path(root)
        self._seq: dict[str, int] = {}

    @property
    def root(self) -> Path:
        return self._root

    def _run_dir(self, run_id: str) -> Path:
        return self._root / run_id

    # -- run records --

    def create_run(
        self,
        run_id: str,
        *,
        agent_name: str,
        model: str,
        metadata: dict[str, str] | None = None,
    ) -> RunRecord:
        run_dir = self._run_dir(run_id)
        (run_dir / _CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        now = _now()
        record = RunRecord(
            run_id=run_id,
            agent_name=agent_name,
            model=model,
            status="running",
            created_at=now,
            updated_at=now,
            last_heartbeat=now,
            metadata=dict(metadata or {}),
        )
        self._write_record(record)
        (run_dir / _TRANSCRIPT).touch()
        return record

    def read_record(self, run_id: str) -> RunRecord | None:
        path = self._run_dir(run_id) / _META
        if not path.exists():
            return None
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_record(self, record: RunRecord) -> None:
        _atomic_write(self._run_dir(record.run_id) / _META, record.model_dump_json(indent=2))

    def update_status(self, run_id: str, status: RunStatus) -> RunRecord:
        record = self.read_record(run_id)
        if record is None:
            raise FileNotFoundError(f"No run record for '{run_id}'")
        now = _now()
        update: dict[str, object] = {"status": status, "updated_at": now, "last_heartbeat": now}
        if status != "paused":
            update["wake"] = None
        updated = record.model_copy(update=update)
        self._write_record(updated)
        return updated

    def pause_run(self, run_id: str, wake: WakeCondition) -> RunRecord:
        """Pause a run with a persisted wake condition; the process may exit."""
        record = self.read_record(run_id)
        if record is None:
            raise FileNotFoundError(f"No run record for '{run_id}'")
        now = _now()
        updated = record.model_copy(update={"status": "paused", "wake": wake, "updated_at": now, "last_heartbeat": now})
        self._write_record(updated)
        self.append_event(run_id, "pause", {"kind": wake.kind, "wake_at": str(wake.wake_at), "note": wake.note})
        return updated

    def due_wakes(self, *, tolerance_seconds: float = 30.0) -> list[RunRecord]:
        """Paused runs whose timed wake condition is due (clock-skew tolerant).

        `on_approval` and `manual` wakes have no timestamp and never auto-fire;
        they resume through their own signal (approval response, operator).
        """
        horizon = _now() + timedelta(seconds=tolerance_seconds)
        due: list[RunRecord] = []
        for record in self.list_runs():
            if record.status != "paused" or record.wake is None:
                continue
            if record.wake.wake_at is not None and record.wake.wake_at <= horizon:
                due.append(record)
        return due

    def try_acquire_sweep_lock(self, run_id: str, *, stale_after_seconds: float = 60.0) -> bool:
        """Per-run lock so two concurrent sweeps cannot double-resume a run.

        Lock files older than `stale_after_seconds` are treated as leftovers
        from a dead sweep and taken over.
        """
        path = self._run_dir(run_id) / "sweep.lock"
        try:
            with path.open("x", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            return True
        except FileExistsError:
            try:
                age = _now().timestamp() - path.stat().st_mtime
            except OSError:
                return False
            if age > stale_after_seconds:
                try:
                    path.unlink()
                    with path.open("x", encoding="utf-8") as fh:
                        fh.write(str(os.getpid()))
                    return True
                except (OSError, FileExistsError):
                    return False
            return False

    def release_sweep_lock(self, run_id: str) -> None:
        (self._run_dir(run_id) / "sweep.lock").unlink(missing_ok=True)

    def touch_heartbeat(self, run_id: str) -> None:
        record = self.read_record(run_id)
        if record is None:
            return
        self._write_record(record.model_copy(update={"last_heartbeat": _now()}))

    def list_runs(self) -> list[RunRecord]:
        if not self._root.exists():
            return []
        records: list[RunRecord] = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            record = self.read_record(entry.name)
            if record is not None:
                records.append(record)
        return records

    def mark_interrupted_runs(self, stale_after_seconds: float) -> list[str]:
        """Mark `running` runs with a stale heartbeat as `interrupted`.

        Call on startup so a crashed process never leaves phantom `running`
        states behind. Returns the run ids that were transitioned.
        """
        cutoff = _now() - timedelta(seconds=stale_after_seconds)
        interrupted: list[str] = []
        for record in self.list_runs():
            if record.status == "running" and record.last_heartbeat < cutoff:
                self.update_status(record.run_id, "interrupted")
                interrupted.append(record.run_id)
        return interrupted

    # -- transcript --

    def append_event(self, run_id: str, kind: TranscriptEventKind, payload: dict[str, object] | None = None) -> TranscriptEvent:
        seq = self._next_seq(run_id)
        event = TranscriptEvent(seq=seq, ts=_now(), kind=kind, payload=dict(payload or {}))
        path = self._run_dir(run_id) / _TRANSCRIPT
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.model_dump(mode="json"), default=str) + "\n")
            fh.flush()
        return event

    def read_events(self, run_id: str, after_seq: int = 0) -> list[TranscriptEvent]:
        """Read transcript events after `after_seq`. A torn tail line is skipped."""
        path = self._run_dir(run_id) / _TRANSCRIPT
        if not path.exists():
            return []
        events: list[TranscriptEvent] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = TranscriptEvent.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Skipping corrupt transcript line %d in run %s", line_no, run_id)
                    continue
                if event.seq > after_seq:
                    events.append(event)
        return events

    def _next_seq(self, run_id: str) -> int:
        if run_id not in self._seq:
            events = self.read_events(run_id)
            self._seq[run_id] = events[-1].seq if events else 0
        self._seq[run_id] += 1
        return self._seq[run_id]

    # -- turn checkpoints --

    def save_checkpoint(self, checkpoint: TurnCheckpoint, *, keep_last: int = 3) -> Path:
        cp_dir = self._run_dir(checkpoint.run_id) / _CHECKPOINT_DIR
        cp_dir.mkdir(parents=True, exist_ok=True)
        path = cp_dir / f"turn-{checkpoint.turn:06d}.json"
        _atomic_write(path, _serialize_turn_checkpoint(checkpoint))
        for stale in sorted(cp_dir.glob("turn-*.json"))[:-keep_last]:
            stale.unlink(missing_ok=True)
        return path

    def load_latest_checkpoint(self, run_id: str) -> TurnCheckpoint | None:
        cp_dir = self._run_dir(run_id) / _CHECKPOINT_DIR
        if not cp_dir.exists():
            return None
        for path in sorted(cp_dir.glob("turn-*.json"), reverse=True):
            try:
                return _deserialize_turn_checkpoint(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.warning("Skipping corrupt checkpoint %s", path)
                continue
        return None


def _serialize_turn_checkpoint(cp: TurnCheckpoint) -> str:
    payload = cp.model_dump(mode="json")
    # Message content blocks are a union; serialize through the shared
    # checkpoint helpers so deserialization is deterministic.
    payload["messages"] = [_serialize_message(m) for m in cp.messages]
    return json.dumps(payload, indent=2, default=str)


def _deserialize_turn_checkpoint(raw: str) -> TurnCheckpoint:
    data = json.loads(raw)
    return TurnCheckpoint(
        run_id=data["run_id"],
        turn=data["turn"],
        messages=[_deserialize_message(m) for m in data["messages"]],
        token_usage=TokenUsage(**data["token_usage"]),
        budget=BudgetSnapshot(**data.get("budget", {})),
        loop_window=tuple(data.get("loop_window", ())),
        last_output=data.get("last_output", ""),
        retries=data.get("retries", 0),
        lifecycle_events=[LifecycleEvent.model_validate(e) for e in data.get("lifecycle_events", [])],
        context_manifests=[ContextManifest.model_validate(m) for m in data.get("context_manifests", [])],
        verification_results=[VerificationResult.model_validate(v) for v in data.get("verification_results", [])],
        gate_decisions=[QualityGateDecision.model_validate(g) for g in data.get("gate_decisions", [])],
        created_at=data["created_at"],
    )
