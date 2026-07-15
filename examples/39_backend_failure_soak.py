"""Exercise ambiguous worker commits; set ANYCODE_SOAK_SECONDS=86400 for 24 hours."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime

from anycode.backends import Admission, AmbiguousBackendResultError, InMemoryDurabilityBackend, WorkItem
from anycode.contracts import Event, Run, transition_run
from anycode.helpers.uuid7 import uuid7

DEFAULT_SOAK_SECONDS = 2.0
LEASE_SECONDS = 30.0


async def main() -> None:
    duration = float(os.getenv("ANYCODE_SOAK_SECONDS", str(DEFAULT_SOAK_SECONDS)))
    deadline = time.monotonic() + duration
    completed = 0
    while time.monotonic() < deadline:
        backend = InMemoryDurabilityBackend()
        run_id = str(uuid7())
        now = datetime.now(UTC)
        run = Run(id=run_id, correlation_id=run_id, created_at=now, updated_at=now)
        initial = Event(id=str(uuid7()), run_id=run_id, sequence=1, type="run.accepted", correlation_id=run_id, emitted_at=now)
        admitted = await backend.admit(Admission(admission_key=run_id, run=run, initial_event=initial))
        if admitted.run is None:
            raise RuntimeError("Admission failed")
        work = WorkItem(id=str(uuid7()), run_id=run_id, task_id="soak-task", available_at=now)
        await backend.enqueue(work)
        claimed = await backend.claim("soak-worker", lease_seconds=LEASE_SECONDS)
        queued = transition_run(admitted.run, "queued")
        if claimed.claim is None or queued.run is None or queued.event is None:
            raise RuntimeError("Work setup failed")
        backend.inject_failure("commit", after_commit=True)
        try:
            await backend.commit(claimed.claim, queued.event, expected_sequence=1, run=queued.run)
        except AmbiguousBackendResultError:
            pass
        replay = await backend.commit(claimed.claim, queued.event, expected_sequence=1, run=queued.run)
        events = await backend.read_events(run_id)
        if replay.accepted or not replay.stale_owner or len(events) != 2:
            raise RuntimeError("State divergence or duplicate committed event detected")
        completed += 1
        await asyncio.sleep(0)
    print(f"completed={completed} duration_seconds={duration:.1f} duplicate_events=0")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())

