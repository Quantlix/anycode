"""Exercise the preview semantic contract without provider credentials.

Run with::

    uv run python examples/37_semantic_contract.py

The example admits and completes a run, fences one side effect, stores its
artifact, and proves that independent event projections produce the same view.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from anycode import (
    CONTRACT_SCHEMA_VERSION,
    ArtifactAccessContext,
    ArtifactWriteRequest,
    Event,
    IncrementalRunProjector,
    InMemoryOperationStore,
    LocalArtifactStore,
    Run,
    canonical_input_digest,
    project_run,
    transition_run,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
LEASE_SECONDS = 30


async def main() -> None:
    admitted = Event(
        id="event-admitted",
        run_id="run-example",
        sequence=1,
        type="run.admitted",
        correlation_id="corr-example",
        emitted_at=NOW,
    )
    run = Run(
        id="run-example",
        root_task_id="task-example",
        correlation_id="corr-example",
        created_at=NOW,
        updated_at=NOW,
        last_event_sequence=1,
    )
    queued = transition_run(run, "queued", causation_id=admitted.id, now=NOW)
    assert queued.run is not None and queued.event is not None
    running = transition_run(queued.run, "running", causation_id=queued.event.id, now=NOW)
    assert running.run is not None and running.event is not None

    payload = b'{"status":"approved"}'
    operation_key = "approval:task-example"
    operation_store = InMemoryOperationStore()
    claimed = await operation_store.claim(
        operation_key=operation_key,
        run_id=run.id,
        task_id=run.root_task_id,
        input_digest=canonical_input_digest({"task_id": run.root_task_id}),
        owner_id="worker-example",
        lease_seconds=LEASE_SECONDS,
        now=NOW,
    )
    assert claimed.claim is not None

    with tempfile.TemporaryDirectory(prefix="anycode-contract-") as directory:
        artifact_store = LocalArtifactStore(Path(directory), max_inline_bytes=0)
        written = await artifact_store.put(
            ArtifactWriteRequest(
                artifact_id="artifact-example",
                run_id=run.id,
                task_id=run.root_task_id,
                name="approval.json",
                media_type="application/json",
                data=base64.b64encode(payload).decode("ascii"),
                producer="worker-example",
                operation_key=operation_key,
                correlation_id=run.correlation_id,
            ),
            ArtifactAccessContext(principal="example"),
        )
        assert written.artifact is not None
        read_back = await artifact_store.get(written.artifact.id, ArtifactAccessContext(principal="example"))
        assert read_back.read_bytes() == payload

        committed = await operation_store.commit(
            operation_key=operation_key,
            owner_id=claimed.claim.owner_id,
            fencing_token=claimed.claim.fencing_token,
            result_artifact_id=written.artifact.id,
            now=NOW,
        )
        artifact_event = Event(
            id="event-artifact",
            run_id=run.id,
            task_id=run.root_task_id,
            sequence=4,
            type="artifact.committed",
            payload={"artifact_id": written.artifact.id},
            correlation_id=run.correlation_id,
            causation_id=running.event.id,
            emitted_at=NOW,
        )
        run_with_artifact = running.run.model_copy(update={"last_event_sequence": artifact_event.sequence})
        succeeded = transition_run(run_with_artifact, "succeeded", causation_id=artifact_event.id, now=NOW)
        assert succeeded.run is not None and succeeded.event is not None

        events = [admitted, queued.event, running.event, artifact_event, succeeded.event]
        batch = project_run(events)
        incremental = IncrementalRunProjector(run.id)
        for event in events:
            assert incremental.apply(event).ok
        assert batch.ok and batch.projection == incremental.projection

        print(
            json.dumps(
                {
                    "artifact": {"digest": written.artifact.digest, "form": written.artifact.content.form},
                    "contract_version": CONTRACT_SCHEMA_VERSION,
                    "event_cursor": incremental.projection.cursor,
                    "operation": committed.outcome,
                    "projection_match": True,
                    "run_state": succeeded.run.state,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
