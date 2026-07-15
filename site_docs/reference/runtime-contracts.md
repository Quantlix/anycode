---
title: "Runtime Contracts and Baseline"
description: "The executable AnyCode lifecycle, verification boundaries, persisted local formats, supported resume scenarios, and reproducible M0 runtime baseline."
keywords: AnyCode lifecycle state machine, verification phases, checkpoint format, durable resume, runtime benchmark
---

# Runtime Contracts and Baseline

This page describes the behavior implemented by the embedded Python runtime. It is a current alpha contract, not a remote-worker protocol: execution remains in one Python process, persisted recovery is local, and general external side effects are not exactly-once.

## Public model inventory

The [complete API inventory](api-inventory.md) is generated from `anycode.__all__` and is the authoritative list of supported package-root imports. Runtime models fall into these contract groups:

| Group | Principal public models |
| --- | --- |
| Agent execution | `AgentConfig`, `RunnerOptions`, `AgentState`, `AgentRunResult`, `RunResult`, `TeamRunResult` |
| Work scheduling | `Task`, `TaskStatus`, `TeamConfig`, `RouteDecision`, `Handoff` |
| Lifecycle and verification | `ExecutionPhase`, `LifecycleEvent`, `StopReason`, `VerificationResult`, `QualityGateDecision`, `SensorContext` |
| Durable local state | `CheckpointData`, `TurnCheckpoint`, `RunRecord`, `TranscriptEvent`, `WakeCondition` |
| Tool execution | `ToolDefinition`, `ToolUseContext`, `ToolResult`, `ToolCallRecord`, `ToolSecurityPolicy` |
| Context and accounting | `ContextPolicy`, `ContextManifest`, `ContextUsageReport`, `TokenUsage`, `CostReport` |

All Pydantic models in these groups are frozen. Runtime code creates updated copies instead of mutating an existing model.

## Current capability matrix

| Capability | Status | Enforced boundary |
| --- | --- | --- |
| In-process agents, teams, DAG tasks, and bounded concurrency | Shipped | One Python process; no remote placement or worker lease. |
| Lifecycle events and four verification attachment points | Shipped | Embedded runner and orchestrator results; not a versioned wire event stream. |
| Workflow checkpoints and durable turn resume | Shipped | Local filesystem and SQLite checkpoint stores; supported fixtures and process-exit tests. |
| Side-effect idempotency claims | Shipped locally | In-memory or SQLite claim store; no distributed fencing or general exactly-once external effect. |
| Tool allowlists, path policy, approvals, and shell controls | Shipped as application policy | Host isolation, IAM, network policy, and tenant boundaries remain operator-owned. |
| MCP tool client and bridge | Partial interoperability | Implemented feature set; current-revision conformance and asynchronous MCP Tasks are not claimed. |
| Transport-neutral run service and A2A | Absent | No public submit/get/list/stream/cancel service, Agent Card, or A2A binding. |
| Distributed durable workers | Absent | No claim lease, heartbeat reassignment, generation, or stale-owner fencing contract. |
| TypeScript client/runtime | Absent | Python package only. |

## Execution lifecycle

`LifecycleEmitter` records `initialized` at construction. Registered listeners receive subsequent transitions; they are not called retroactively for the initial event. Strict emitters accept only these transitions:

| Current phase | Allowed next phases |
| --- | --- |
| `initialized` | `planning`, `executing`, `cancelled`, `failed` |
| `planning` | `executing`, `cancelled`, `failed` |
| `executing` | `observing`, `verifying`, `recovering`, `completed`, `failed`, `cancelled` |
| `observing` | `executing`, `verifying`, `recovering`, `completed`, `failed`, `cancelled` |
| `verifying` | `executing`, `recovering`, `completed`, `failed`, `cancelled` |
| `recovering` | `executing`, `observing`, `completed`, `failed`, `cancelled` |
| `completed` | none |
| `failed` | none |
| `cancelled` | none |

The built-in `AgentRunner` emits `initialized`, `executing`, `observing`, `verifying`, `recovering`, and one terminal phase. `planning` is available to explicit lifecycle emitters but is not currently emitted by `AgentRunner` or `AnyCode`; orchestrator progress uses the separate `OrchestratorEvent` surface.

Terminal phase and stop reason are separate. A result is `completed` only for the `success` stop code. Limits, provider exhaustion, verification failure, uncertain side effects, and unknown errors settle as `failed`; caller cancellation settles as `cancelled` and re-raises `asyncio.CancelledError`.

## Verification boundaries

The four sensor phases have precise attachment points:

| Sensor phase | Emission and enforcement point |
| --- | --- |
| `before_tool` | Immediately before each non-empty tool-call batch. A passing, warning, or retry decision returns the lifecycle to `executing`; block and escalation stop before invocation. |
| `after_tool` | After the complete tool batch returns and before results are added to the next model turn. Block and escalation retain the completed call records and stop the run. |
| `after_task` | When the model produces a terminal response with no tool calls, after output validation and before success is committed. Retry feedback can start another model turn. |
| `after_team` | Once after a successful explicit task run, or once after coordinator and task outputs have been combined by `run_team()`. Decisions and sensor evidence remain on `TeamRunResult`. |

An upstream agent or task failure skips `after_team`, because there is no successful candidate result to verify. At the team boundary, `retry` means a recoverable `verification_failed` result; AnyCode does not silently rerun the whole team. `warn` and `pass` preserve success, while `block` and `escalate` fail the result.

Every phase is covered by runtime integration tests in `tests/test_harness_runtime.py`. The complete team path is also runnable without credentials in `examples/35_lifecycle_contract.py`.

## Task and durable-run states

The embedded task queue uses `pending`, `in_progress`, `completed`, `failed`, and `blocked`. Dependency failure is terminal for affected descendants: they become `failed` with an explicit prerequisite message rather than remaining indefinitely blocked.

The local durable run store uses `running`, `paused`, `interrupted`, `completed`, `failed`, and `cancelled`. A startup watchdog can mark a stale `running` record as `interrupted`; it does not infer success or automatically replay work.

## Persisted formats

| Artifact | Current writer | Supported reader range | Storage behavior |
| --- | --- | --- | --- |
| Declarative YAML/TOML | v1 | v1 | Missing version is legacy v1; unknown fields and future versions fail closed. |
| Workflow checkpoint | v2 | v1-v2 | Atomic JSON file or SQLite row; v1 observability fields receive safe defaults. |
| Durable run record | v1 | v1 | Atomic `meta.json`; format checked before model validation. |
| Durable transcript | v1 | v1 | Append-only JSONL; a torn final line is ignored. |
| Durable turn checkpoint | v1 | v1 | Atomic, pruned JSON snapshots beneath the run directory. |
| Protected payload envelope | protector-defined | protector-defined | Independent from the run schema; the configured protector owns keys and migration. |

The [compatibility policy](compatibility.md) defines upgrade, rollback, and breaking-change rules. Future versions are rejected rather than guessed. Built-in redaction is enabled by default, but confidentiality and integrity still depend on operator-owned storage permissions, encryption, backups, and key management.

## Supported local resume scenarios

| Scenario | Supported behavior | Boundary |
| --- | --- | --- |
| Team workflow checkpoint | `run_tasks(..., resume_from=checkpoint_id)` restores task status, completed agent results, usage, and the next wave. | The task specification must remain compatible; this is not event-history replay. |
| Durable agent turn | A fresh `AgentRunner` accepts the latest `TurnCheckpoint` and continues history, budgets, usage, retries, loop detection, lifecycle evidence, and run identity. | Recovery begins at a saved turn boundary. |
| Ungraceful process exit | A stale run is marked `interrupted`, its latest atomic checkpoint is loaded in a fresh process, and completed turns are not asked of the provider again. | An external effect completed before an unrecorded crash still requires idempotency and reconciliation. |
| Caller cancellation | The runner persists a `user_cancelled` stop and terminal checkpoint before cancellation remains caller-visible. | Custom tools must propagate cancellation and clean up their own resources. |
| Provider outage | A durable run can pause with an `on_provider_recovery` wake condition. | A scheduler or operator must perform the wake; telemetry is not authoritative state. |

`tests/test_runstore.py` includes an actual child-process exit fixture. The process terminates without Python cleanup, then a parent process detects and resumes the run from disk.

## Side-effect contract

Side-effecting tools claim an idempotency key before execution. Completed claims replay their stored result; conflicting inputs, concurrent in-progress claims, unavailable claim storage, and uncertain post-execution outcomes fail closed. An unrecorded external effect is reported as `side_effect_unknown` and must be reconciled before retry.

This prevents two AnyCode-committed results for one local claim. It does not make an arbitrary external API transactional, fence stale remote workers, or provide a multi-host exactly-once guarantee.

## Reproduce the baseline

Run the credential-free fixture from the repository root:

```bash
uv run python examples/36_runtime_baseline.py
```

The JSON contains five measurements under `metrics`: `task_admission`, `execution`, `checkpoint_size`, `event_volume`, and `context_growth`. Workload sizes and schema version are fixed in the example; elapsed timings are observations for the current host, while counts, shape, and monotonic context growth are contract-tested. CI runs the same command so every build retains comparable log evidence.

To retain release evidence outside the source tree:

```bash
uv run python examples/36_runtime_baseline.py --output artifacts/runtime-baseline.json
```

Do not compare elapsed values across unlike operating systems, Python versions, power profiles, or shared CI runners as if they were regressions. Establish thresholds only from a controlled runner and multiple samples.

## Contract evidence

| Contract | Primary evidence |
| --- | --- |
| Legal and illegal lifecycle transitions | `tests/test_lifecycle.py` |
| All verification attachment points | `tests/test_harness_runtime.py` |
| Checkpoint formats and local stores | `tests/test_checkpoint.py`, `tests/integration/test_checkpoint_stores.py` |
| Process interruption and durable resume | `tests/test_runstore.py`, `tests/fixtures/process_crash_runner.py` |
| Dependency failure propagation and cancellation | `tests/test_harness_runtime.py`, `tests/test_tool_engine.py` |
| Side-effect claims and uncertain outcomes | `tests/test_tool_engine.py`, `tests/test_runner_streaming.py` |
| Baseline structure and examples | `tests/test_runtime_baseline.py`, `examples/35_lifecycle_contract.py`, `examples/36_runtime_baseline.py` |
