---
title: "Configure AnyCode Durability Backends"
description: Choose and configure AnyCode in-memory, SQLite, or Dapr durability backends for runs, events, leases, checkpoints, wakes, and signals.
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

## Next steps

- [Propagate execution identity and policy](execution-identity.md)
- [Host AnyCode services](hosting-services.md)
- [Review runtime contracts](../reference/runtime-contracts.md)
- [Deploy portable infrastructure](portable-infrastructure.md)
