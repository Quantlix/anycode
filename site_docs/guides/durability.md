---
title: "Checkpoint, Resume, and Schedule Durable AnyCode Runs"
description: "Persist AnyCode work with workflow checkpoints and durable runs, resume after a crash, chain calendar-scale goals with SessionChain, and wake paused runs on a schedule."
keywords: anycode durability, checkpointing, durable runs, resume run, FilesystemRunStore, DurabilityConfig, CheckpointManager, SessionChain, scheduled wakeups, anycode runs cli
---

# Durable and Resumable Runs

Long jobs fail — a provider blips, a process is killed, a machine reboots. AnyCode has two independent durability systems so a run can pick up where it left off instead of paying to redo finished work, plus tools for goals that span days or weeks. This guide covers all four: workflow checkpoints, durable single-agent runs, session chains, and scheduled wakeups.

!!! note "Two 'checkpoint' systems — don't mix them"
    **Workflow checkpoints** (`CheckpointManager`, async, keyed by workflow + wave) snapshot a multi-agent team run. **Durable runs** (`FilesystemRunStore`, synchronous, keyed by run + turn) snapshot a single `AgentRunner` loop. They are separate subsystems with separate config.

## Checkpoint a team workflow

`CheckpointManager` saves the state of a wave-based team run — tasks, per-agent results, and token usage — so a fresh process can reload the latest snapshot and continue from the next wave.

```python title="checkpoints.py"
from anycode import CheckpointConfig, CheckpointManager

config = CheckpointConfig(enabled=True, path=".anycode/checkpoints", keep_last=5)
manager = CheckpointManager(config)

# After finishing a wave:
await manager.auto_save(
    workflow_id="nightly-report",
    tasks=tasks,
    agent_results=results,
    wave_index=wave,
    total_usage=usage,
)

# In a later process, reload and continue:
restored = await CheckpointManager(config).load_latest(workflow_id="nightly-report")
if restored is not None:
    start_wave = restored.wave_index + 1
```

| `CheckpointConfig` field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `False` | Turn checkpointing on |
| `path` | `.anycode/checkpoints` | Where snapshots are written |
| `keep_last` | `5` | Snapshots to retain per workflow before pruning |
| `redact_sensitive_data` | `True` | Scrub recognized credentials before persistence |

!!! warning "Resume is manual, and `backend` is not honored by the manager"
    There is no auto-resume — you reload with `load_latest` and drive the loop yourself. `CheckpointManager` always writes to the filesystem; to use the SQLite store you must construct `SQLiteCheckpointStore(...)` (from `anycode.checkpoint`) and pass it in explicitly.

## Make a single agent run durable

Durable runs persist every turn of an `AgentRunner` to a `FilesystemRunStore`. Enable it with a `DurabilityConfig`; on a crash, reload the last turn checkpoint and hand it to a new runner via `resume_from`.

```python title="durable.py"
from anycode import AgentRunner, DurabilityConfig, FilesystemRunStore, RunnerOptions, ToolExecutor

store = FilesystemRunStore(".anycode/runs")
runner = AgentRunner(
    adapter, registry, ToolExecutor(registry),
    RunnerOptions(model="claude-haiku-4-5", agent_name="worker", max_turns=10),
    durability=DurabilityConfig(enabled=True, run_root=str(store.root), checkpoint_every_turns=1),
    run_store=store,
)
async for _event in runner.stream(messages):
    ...

# In a fresh process — reload the checkpoint and resume:
record = store.list_runs()[0]
checkpoint = store.load_latest_checkpoint(record.run_id)
resumed = AgentRunner(adapter, registry, ToolExecutor(registry), options, run_store=store, resume_from=checkpoint)
result = await resumed.run([])   # empty seed is fine when resuming
```

| `DurabilityConfig` field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `False` | Turn durability on |
| `run_root` | `.anycode/runs` | Root directory for run records |
| `checkpoint_every_turns` | `5` | Save a turn checkpoint every N turns |
| `keep_last_checkpoints` | `3` | Turn checkpoints retained per run |
| `heartbeat_seconds` | `30.0` | Heartbeat interval for stale-run detection |
| `redact_sensitive_data` | `True` | Scrub recognized credentials from records, events, and turn checkpoints |

A durable run that hits a provider outage doesn't fail — it **pauses** with a wake condition to retry later. `resume_from` takes a loaded `TurnCheckpoint`, not a run ID.

!!! warning "Redaction changes persisted replay data"
    Checkpoint and run-store redaction is enabled by default. Recognized credentials are replaced with `<redacted-secret>` in serialized messages, tool inputs and outputs, metadata, errors, and transcript payloads. A resumed run therefore receives the placeholder rather than the original credential. Pass `redact_sensitive_data=False` only when exact replay is required and the store is protected with encryption, access control, and an appropriate retention policy.

Low-level constructors expose the same explicit opt-out. For durable runs, pair it with a payload protector rather than relying on filesystem permissions alone:

```python title="protected-store.py"
class EnvelopeProtector:
    def protect(self, payload: bytes) -> bytes:
        return envelope_encryption_service.encrypt(payload)

    def unprotect(self, payload: bytes) -> bytes:
        return envelope_encryption_service.decrypt(payload)


store = FilesystemRunStore(
    "/protected/runs",
    redact_sensitive_data=False,
    payload_protector=EnvelopeProtector(),
)
checkpoint_store = FilesystemCheckpointStore("/protected/checkpoints", redact_sensitive_data=False)
```

`RunPayloadProtector` is a synchronous byte-oriented protocol, so the implementation can use a KMS-backed envelope-encryption library, an HSM, or a platform secret service without coupling AnyCode to one key provider. The filesystem backend wraps protected bytes in a versioned text envelope. Legacy plaintext runs remain readable when a protector is configured; a protected run fails closed when its protector is missing, its key cannot open the payload, or its envelope version is unsupported.

AnyCode does not generate, store, rotate, or escrow encryption keys. The run ID, directory layout, filenames, and file sizes remain visible. Use restricted filesystem permissions as well as encryption, and test key rotation and disaster recovery against copies of real-size checkpoints before deployment. Workflow `FilesystemCheckpointStore` remains plaintext and needs storage-layer encryption when exact, unredacted replay is enabled.

Run records, transcript events, and turn checkpoints include `format_version=1`. Artifacts written before this field existed are treated as v1. Workflow checkpoints use format v2 and still read explicit v1 snapshots. `UnsupportedRunStoreVersionError` and `UnsupportedCheckpointVersionError` are raised for future formats so an older runtime cannot silently misinterpret newer state. When the newest plain turn checkpoint is corrupt, the filesystem store scans backward to the latest valid snapshot.

## Use a production run-store backend

`AgentRunner`, `sweep_once`, and `RunScheduler` depend on the public `RunStore` protocol rather than the filesystem class. A custom database or object-store backend can therefore be injected through `run_store=` without changing the runner. It must preserve atomic run-record updates, ordered event sequence numbers, checkpoint recovery, and mutually exclusive sweep locks; those semantics are part of the contract, not optional optimizations.

## Bound durable-run retention

No run is deleted unless a retention policy is supplied. `RunRetentionPolicy` can bound terminal runs by age, count, or both:

```python title="retention.py"
from anycode import FilesystemRunStore, RunRetentionPolicy, sweep_once

store = FilesystemRunStore(".anycode/runs")
policy = RunRetentionPolicy(max_age_days=30, max_runs=1_000)
report = await sweep_once(store, retention_policy=policy)
print(report.pruned)
```

Age pruning runs first; the count bound then keeps the newest remaining terminal runs. Only `completed`, `failed`, and `cancelled` runs are eligible. `running`, `paused`, and recoverable `interrupted` runs are retained. Apply the same policy on every scheduler tick with `RunScheduler(..., retention_policy=policy)` or from an external scheduler:

```bash title="Prune during a watchdog sweep"
anycode runs sweep --retention-days 30 --max-runs 1000
```

## Inspect runs from the CLI

The `anycode runs` commands read a run store directly:

```bash title="Inspect durable runs"
anycode runs list                 # table of every run and its status
anycode runs show <run_id>        # status, wake condition, recent events
anycode runs tail <run_id>        # transcript events after a sequence number
anycode runs audit <run_id>       # event-kind digest over a time window
anycode runs sweep                # watchdog pass; optional explicit retention bounds
```

!!! warning "There is no `anycode runs resume`"
    `sweep` reports which runs are due to wake but does not resume them. Resuming is programmatic — via `AgentRunner(resume_from=...)` or a `RunScheduler`.

## Chain calendar-scale goals

For work too large for one context window, `SessionChain` runs a series of **fresh** sessions against a durable `GoalContract`. Each session gets a clean context, works the next unmet criterion, and an external `verifier` — never the agent's own say-so — decides whether the criterion passed.

```python title="session_chain.py"
from anycode import GoalContract, GoalCriterion, SessionChain

contract = GoalContract(
    goal="Ship the data-export feature",
    criteria=(
        GoalCriterion(id="schema", description="Design the export schema"),
        GoalCriterion(id="endpoint", description="Implement the export endpoint"),
        GoalCriterion(id="tests", description="Cover the endpoint with tests"),
    ),
)

chain = SessionChain(
    runner_factory=runner_factory,   # returns a NEW AgentRunner (fresh context) each session
    contract=contract,
    work_dir="export-feature",
    verifier=verifier,               # async (criterion, result) -> evidence str | None
    max_sessions=6,
)
final = await chain.run()            # loops until final.complete or max_sessions
```

Only the contract (`contract.json`) and an append-only `progress.md` carry between sessions — the conversation itself does not. This is the mechanism for goals measured in days rather than turns.

Goal contracts and progress-log output are redacted before they are written by default. Pass `redact_sensitive_data=False` to `SessionChain` only for a protected work directory that requires exact persisted text.

## Wake paused runs on a schedule

A paused run carries a `WakeCondition`. A **sweep** is the wake mechanism: it finds runs whose wake time has arrived and calls your resume function.

```python title="scheduler.py"
from anycode import FilesystemRunStore, RunScheduler, sweep_once

store = FilesystemRunStore(".anycode/runs")

async def resume(run_id: str) -> None:
    ...   # load the checkpoint and continue the run

# One pass:
report = await sweep_once(store, resume=resume)
print(report.woken)   # run IDs resumed this pass

# Or run a background tick loop:
scheduler = RunScheduler(store, resume=resume, interval_seconds=5.0)
```

Only timed wakes (`at_time`, `on_provider_recovery`) fire automatically; `on_approval` and `manual` wakes resume through their own signal. For lightweight periodic jobs that aren't full agent runs, `run_scheduled_task` handles `notification`, `script`, `agent`, and `hybrid` modes.

## Next steps

- [Build a resumable pipeline](../tutorials/resumable-pipeline.md) — durability applied to a real crash-and-resume project.
- [Track and cap cost](cost-tracking.md) — pair durable runs with a spend ceiling.
- [Production controls](production-controls.md) — the broader hardening picture.
- [CLI reference](../reference/cli.md) — the full `anycode runs` command set.
