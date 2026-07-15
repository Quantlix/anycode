"""Run one semantic workload on SQLite or a Dapr state store."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

from anycode.backends import (
    Admission,
    DaprDurabilityBackend,
    DaprHTTPTransport,
    ExternalSignal,
    SQLiteDurabilityBackend,
    WakeRegistration,
    WorkItem,
)
from anycode.contracts import Checkpoint, Event, Run, transition_run
from anycode.helpers.uuid7 import uuid7

DEFAULT_DATABASE = ".anycode/examples/durability.db"
DEFAULT_DAPR_STORE = "statestore"
DEFAULT_DAPR_URL = "http://127.0.0.1:3500"
LEASE_SECONDS = 30.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "dapr"), default="sqlite")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--dapr-store", default=os.getenv("DAPR_STATE_STORE", DEFAULT_DAPR_STORE))
    parser.add_argument("--dapr-url", default=os.getenv("DAPR_HTTP_ENDPOINT", DEFAULT_DAPR_URL))
    return parser.parse_args()


async def main() -> None:
    args = _arguments()
    if args.backend == "dapr":
        transport = DaprHTTPTransport(
            args.dapr_store,
            base_url=args.dapr_url,
            api_token=os.getenv("DAPR_API_TOKEN"),
        )
        backend = DaprDurabilityBackend(transport)
    else:
        backend = SQLiteDurabilityBackend(args.database)

    run_id = str(uuid7())
    now = datetime.now(UTC)
    run = Run(id=run_id, correlation_id=run_id, created_at=now, updated_at=now)
    initial = Event(id=str(uuid7()), run_id=run_id, sequence=1, type="run.accepted", correlation_id=run_id, emitted_at=now)
    admitted = await backend.admit(Admission(admission_key=f"example:{run_id}", run=run, initial_event=initial))
    if not admitted.admitted or admitted.run is None:
        raise RuntimeError(admitted.error.message if admitted.error else "Run admission failed")

    await backend.enqueue(WorkItem(id=str(uuid7()), run_id=run_id, task_id="example-task", available_at=now))
    claimed = await backend.claim("example-worker", lease_seconds=LEASE_SECONDS)
    if claimed.claim is None:
        raise RuntimeError("Ready work could not be claimed")
    queued = transition_run(admitted.run, "queued", now=now)
    if queued.run is None or queued.event is None:
        raise RuntimeError("Run transition failed")
    committed = await backend.commit(claimed.claim, queued.event, expected_sequence=1, run=queued.run)
    if not committed.accepted:
        raise RuntimeError(committed.error.message if committed.error else "Commit failed")

    checkpoint = Checkpoint(
        id=str(uuid7()),
        run_id=run_id,
        event_cursor=2,
        generation=queued.run.generation,
        attempt=queued.run.attempt,
        correlation_id=run_id,
        run=queued.run,
    )
    await backend.save_checkpoint(checkpoint)
    await backend.register_wake(
        WakeRegistration(id=str(uuid7()), run_id=run_id, wake_at=now + timedelta(minutes=5), reason="scheduled follow-up")
    )
    await backend.deliver_signal(ExternalSignal(id=str(uuid7()), run_id=run_id, name="operator-note", payload="continue"))
    snapshot = await backend.export_run(run_id)
    print(json.dumps(snapshot.model_dump(mode="json") if snapshot else None, indent=2))  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())

