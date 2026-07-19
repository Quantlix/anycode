---
title: "Configure AnyCode Durability Backends"
description: Choose AnyCode in-memory, SQLite, or Dapr durability backends and configure guarantees for runs, events, leases, checkpoints, wakes, migrations, and signals.
keywords: AnyCode durability backend, SQLite agent state, Dapr agent persistence, AI agent leases, durable agent runs
---

# Configure AnyCode durability backends

An AnyCode `DurabilityBackend` stores the semantic state needed to admit work, append ordered events, claim tasks with leases and fencing, save checkpoints, register wakes, deliver signals, and retain artifact references. Use the in-memory backend for deterministic tests, SQLite for one-host persistence, and Dapr for an external state store in a distributed deployment.

## Which durability backend should you choose?

| Backend | Persistence | Deployment shape | Best fit | Main limit |
| --- | --- | --- | --- | --- |
| `InMemoryDurabilityBackend` | Process lifetime | One Python process | Tests, examples, conformance, and fault injection | State disappears when the process exits |
| `SQLiteDurabilityBackend` | Local database | One host or one mounted-volume replica | Local services and restart-safe development | One coarse writer transaction limits write throughput |
| `DaprDurabilityBackend` | External state store | Multiple service replicas | Distributed workers with shared durable state | The configured store must provide strong reads, ETags, and transactions |

Every backend reports its guarantees through `capabilities()` and its implementation and store versions through `version()`. Check those reports during startup instead of assuming that a configured provider supports a required operation.

## Start with the in-memory backend

The in-memory backend implements the complete contract and accepts a controllable clock. It is the reference implementation used by the conformance and failure-soak tests.

```python
from anycode import InMemoryDurabilityBackend

backend = InMemoryDurabilityBackend()

capabilities = backend.capabilities()
health = await backend.health()

assert capabilities.fencing
assert capabilities.external_signals
assert health.status == "healthy"
```

`InMemoryDurabilityBackend.inject_failure()` can fail an operation before or after its state change. This is useful for testing ambiguous acknowledgements, retries, and reconciliation without taking down an external service.

## Persist local state with SQLite

Install the persistence extra, then point the backend at a local database file:

```bash
uv add "anycode-py[persistence]"
```

```python
from anycode import SQLiteDurabilityBackend

backend = SQLiteDurabilityBackend(".anycode/backend.db")

print(backend.capabilities().model_dump())
print(backend.version().model_dump())
```

SQLite serializes backend operations through a versioned state row and a writer transaction. That favors correctness and migration simplicity. Choose an external backend when multiple hosts need to claim work concurrently or when the state volume makes a single aggregate impractical.

## Connect an external Dapr state store

`DaprHTTPTransport` uses the Dapr v1.0 state and health endpoints. The runtime backend applies compare-and-set writes against the returned ETag.

```python
import os

from anycode import DaprDurabilityBackend, DaprHTTPTransport

transport = DaprHTTPTransport(
    "statestore",
    base_url="http://127.0.0.1:3500",
    api_token=os.environ.get("DAPR_API_TOKEN"),
)
backend = DaprDurabilityBackend(transport)

health = await backend.health()
if health.status != "healthy":
    raise RuntimeError(health.message)
```

The selected Dapr component must support transactions, strong reads, and ETags. The preview stores one coarse runtime aggregate, so the state store's item-size and contention limits still apply. Review the capability limitations at startup and include the Dapr component in backup, encryption, and regional-placement controls.

## What does the backend contract guarantee?

The `DurabilityBackend` protocol groups operations by the state they protect:

| Concern | Operations | Consistency rule |
| --- | --- | --- |
| Admission | `admit`, `enqueue` | An admission key is idempotent and cannot silently represent different run input |
| Worker ownership | `claim`, `heartbeat`, `commit` | Leases expire, fencing tokens increase, and stale owners cannot commit |
| Event history | `append_event`, `read_events` | Callers provide the expected sequence and receive a typed conflict on mismatch |
| Recovery | `save_checkpoint`, `load_checkpoint`, `register_wake`, `due_wakes` | Checkpoints keep an event cursor; wakes remain explicit runtime records |
| External input | `deliver_signal`, `read_signals` | Signal identifiers are deduplicated and keep their execution context |
| Artifacts | `record_artifact_reference`, `read_artifact_references` | The backend stores references and integrity metadata, not an implicit blob store |
| Operations | `health`, `capabilities`, `version`, `export_run` | Operators can inspect readiness, guarantees, versions, and portable run state |

An ambiguous write acknowledgement is different from a rejected write. Re-read the event cursor or exported run before retrying an operation that may already have committed. Business actions connected to that operation still need their own idempotency key.

## Migrate a filesystem run

The migration helpers translate the existing `RunStore` format into a portable backend snapshot, then import it without duplicating an admitted run.

```python
from anycode import (
    FilesystemRunStore,
    SQLiteDurabilityBackend,
    export_filesystem_run,
    import_backend_snapshot,
)

source = FilesystemRunStore(".anycode/runs")
target = SQLiteDurabilityBackend(".anycode/backend.db")

snapshot = export_filesystem_run(source, "run-42")
if snapshot is None:
    raise LookupError("run-42 was not found")

result = await import_backend_snapshot(
    target,
    snapshot,
    admission_key="migration:run-42",
)
if not result.admitted:
    raise RuntimeError(result.error.message if result.error else "migration failed")
```

Keep the source data until the target run, event cursor, checkpoint, wakes, signals, and artifact references have been verified. A migration should add history, not rewrite or delete the source history.

## Verify backend behavior before deployment

Run the shared contract tests against any custom implementation. The built-in suite covers duplicate admission, ordered events, lease takeover, stale fencing, cancellation races, signals, checkpoint recovery, persistence, version rejection, injected partitions, and lost acknowledgements.

The credential-free examples provide a smaller starting point:

- `examples/38_pluggable_durability.py` exercises admission, claims, checkpoints, migration, and backend reports.
- `examples/39_backend_failure_soak.py` exercises partitions and ambiguous post-commit failures.
- `tests/test_backend_conformance.py` is the reusable behavioral contract.

## The complete, runnable program

This one file runs the full backend workload end to end: admit a run, enqueue and claim its work under a lease, commit a state transition against the expected event sequence, save a checkpoint, register a timed wake, deliver an external signal, then export the portable snapshot. It uses `InMemoryDurabilityBackend`, so it needs no external services or extra dependencies. Swap the one backend line for `SQLiteDurabilityBackend(".anycode/backend.db")` (needs the `[persistence]` extra) or a configured `DaprDurabilityBackend` — the workload below does not change, which is the point of the contract.

```python title="durability_backend.py"
import asyncio
from datetime import UTC, datetime, timedelta

from anycode import (
    Admission,
    Checkpoint,
    Event,
    InMemoryDurabilityBackend,
    Run,
    WorkItem,
    transition_run,
    uuid7,
)
from anycode.backends import ExternalSignal, WakeRegistration

LEASE_SECONDS = 30.0


async def main() -> None:
    backend = InMemoryDurabilityBackend()

    caps = backend.capabilities()
    print(f"backend={caps.backend} persistent={caps.persistent} external={caps.external}")
    health = await backend.health()
    print(f"health: {health.status}")

    run_id = str(uuid7())
    now = datetime.now(UTC)

    # 1. Admit the run with its first event. The admission key is idempotent.
    run = Run(id=run_id, correlation_id=run_id, created_at=now, updated_at=now)
    initial = Event(id=str(uuid7()), run_id=run_id, sequence=1, type="run.accepted", correlation_id=run_id, emitted_at=now)
    admitted = await backend.admit(Admission(admission_key=f"example:{run_id}", run=run, initial_event=initial))
    if not admitted.admitted or admitted.run is None:
        raise RuntimeError(admitted.error.message if admitted.error else "admission failed")

    # 2. Enqueue ready work and claim it under a lease.
    await backend.enqueue(WorkItem(id=str(uuid7()), run_id=run_id, task_id="export-task", available_at=now))
    claimed = await backend.claim("worker-1", lease_seconds=LEASE_SECONDS)
    if claimed.claim is None:
        raise RuntimeError("ready work could not be claimed")

    # 3. Commit a state transition against the expected event sequence.
    queued = transition_run(admitted.run, "queued", now=now)
    if queued.run is None or queued.event is None:
        raise RuntimeError(queued.error.message if queued.error else "run transition failed")
    committed = await backend.commit(claimed.claim, queued.event, expected_sequence=1, run=queued.run)
    if not committed.accepted:
        raise RuntimeError(committed.error.message if committed.error else "commit failed")

    # 4. Save a checkpoint, register a timed wake, and deliver an external signal.
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

    # 5. Export the portable snapshot — the same shape every backend produces.
    snapshot = await backend.export_run(run_id)
    assert snapshot is not None
    print(f"exported run={snapshot.run.id} state={snapshot.run.state}")
    print(f"events={len(snapshot.events)} wakes={len(snapshot.wakes)} signals={len(snapshot.signals)} checkpoint={snapshot.checkpoint is not None}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python durability_backend.py
```

!!! tip "Tested copy"
    See [`examples/38_pluggable_durability.py`](https://github.com/Quantlix/anycode/blob/main/examples/38_pluggable_durability.py) for the CI-tested version, which runs the same workload against SQLite or a Dapr state store, and [`examples/39_backend_failure_soak.py`](https://github.com/Quantlix/anycode/blob/main/examples/39_backend_failure_soak.py) for the injected-failure soak that proves an ambiguous commit never duplicates an event.

## Next steps

- [Propagate execution identity and policy](execution-identity.md)
- [Host AnyCode services](hosting-services.md)
- [Review runtime contracts](../reference/runtime-contracts.md)
- [Deploy portable infrastructure](portable-infrastructure.md)
