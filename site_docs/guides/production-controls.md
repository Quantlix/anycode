---
title: "AnyCode Production Controls: Budgets, Gates, and Checkpoints"
description: Add cost and token budgets, human approval gates, output validators, checkpoints, durable runs, and verification quality gates to AnyCode agent workflows.
keywords: AnyCode production controls, cost budgets, token budgets, HITL approval gates, verification gates, checkpoints, durable runs, context policies, telemetry, output validators
---

# Production Controls

Add cost budgets, human approval gates, verification gates, and durable checkpoints to AnyCode workflows that need limits, audit trails, and resumable state.

AnyCode remains alpha-stage software. These controls are building blocks for bounded production deployments, but they do not supply operating-system isolation, network policy, identity, key management, or incident response. Production eligibility depends on the complete workload and operating environment.

!!! warning "Necessary, not sufficient"
    These controls harden a run, but the package alone is not a production boundary. Review the [security and threat model](../reference/security.md), then require every applicable item in the [production readiness checklist](production-readiness.md) before release.

## What production controls does AnyCode ship?

Each control below is available today and can be layered onto a single agent run or a whole team. Start with limits, then add gates and durability as a workflow grows.

| Control | What it gives you |
| --- | --- |
| Cost and token budgets | A hard ceiling on spend and turns before a run touches live providers. |
| Provider capacity limits | Shared concurrency bulkheads, request pacing, and bounded queue waits around provider calls. |
| Cancellation and shutdown | Direct cancellation propagation plus owned-task and process-tree cleanup. |
| Side-effect idempotency | Atomic claims, replay, conflict detection, and fail-closed restart behavior for mutating tools. |
| HITL approval gates | A human checkpoint before sensitive or irreversible tasks execute. |
| Output validators | Programmatic checks on agent output before it is accepted. |
| Turn hooks | Callbacks around each turn for logging, mutation, or early stops. |
| Structured output | Schema-shaped responses instead of free text. |
| Verification quality gates | `ruff`, `pyright`, `pytest`, `schema`, and `regex` sensors that can block a run. |
| Checkpoints and resume | Recorded task state and agent results so a run can continue after a failure. |
| Durable runs | Run metadata, transcript events, heartbeats, wake conditions, and turn checkpoints. |
| Protected storage and retention | Pluggable run-store backends, payload-protection hooks, and explicit terminal-run pruning. |
| Context policies | Rules for trimming, masking, offloading, or compacting context as history grows. |
| Telemetry | OpenTelemetry tracing through the `telemetry` extra. |
| Data redaction | Default-on credential scrubbing for telemetry, checkpoints, transcripts, context artifacts, eval reports, and evidence bundles. |

## Start with explicit limits

The first control to add is a budget. Set a spend ceiling plus per-agent turn and token caps so a misbehaving loop stops itself before it costs money. You can declare the cost budget in YAML or in Python — both configure the same control.

=== "YAML"

    ```yaml title="team.yaml"
    cost:
      budget_usd: 1.00
      on_budget_exceeded: warn
    ```

=== "Python"

    ```python title="limits.py"
    from anycode import AnyCode, CostConfig

    engine = AnyCode(config={"cost": CostConfig(budget_usd=1.00, on_budget_exceeded="warn")})
    ```

`on_budget_exceeded` controls what happens when the budget is reached — the example uses `warn`. Pair the budget with per-agent `max_turns` and `max_tokens` so each agent has its own ceiling:

```python title="limits.py"
from anycode import AgentConfig, AnyCode, CostConfig

engine = AnyCode(
    config={
        "cost": CostConfig(budget_usd=1.00, on_budget_exceeded="warn"),
    }
)

agent = AgentConfig(
    name="assistant",
    provider="openai",
    model="gpt-4o-mini",
    tools=[],
    max_turns=4,
    max_tokens=1200,
)
```

!!! tip "Set limits before live calls"
    Budgets and turn limits are cheapest to add first. Configure them before pointing an agent at a paid provider, then test the same workflow against a `FakeAdapter` to confirm behavior without spending tokens.

## Bound provider traffic

`ProviderResilienceConfig` caps simultaneous SDK attempts across agents that share a provider scope. Set `requests_per_minute` when a vendor publishes a request quota, and keep `capacity_wait_timeout_seconds` bounded so overload is shed as `ProviderCapacityError` rather than accumulating an unbounded queue.

```yaml title="team.yaml"
provider_resilience:
    max_concurrency: 4
    requests_per_minute: 120
    capacity_scope: shared-production-key
    capacity_wait_timeout_seconds: 30
```

Retries consume capacity and request-rate reservations just like first attempts. The limiter is local to one event loop; use a provider gateway or distributed limiter for aggregate quotas across worker processes or hosts. Request pacing does not enforce tokens-per-minute limits.

## Make side effects idempotent

Mark a custom tool with `side_effecting=True` to claim an idempotency key before it executes. Completed calls replay their stored `ToolResult`; a reused key with different validated input is rejected. Concurrent or crash-interrupted claims are treated as indeterminate, and the runner stops with `side_effect_unknown` instead of asking the model to try again.

```python title="idempotent_tool.py"
from pydantic import BaseModel

from anycode import ToolResult, ToolUseContext, define_tool


class ChargeInput(BaseModel):
    amount_cents: int
    idempotency_key: str


async def charge(input: ChargeInput, context: ToolUseContext) -> ToolResult:
    await payment_api.charge(  # application-owned client
        amount_cents=input.amount_cents,
        idempotency_key=context.idempotency_key,
    )
    return ToolResult(data="charged")


charge_tool = define_tool(
    name="charge",
    description="Charge a payment method once.",
    input_model=ChargeInput,
    execute=charge,
    side_effecting=True,
)
```

An explicit validated `idempotency_key` field takes precedence. Otherwise, the runner supplies a deterministic key derived from the run, turn, and tool-call position so a durable restart claims the same operation. Storage keys and input fingerprints are SHA-256 hashes; persisted results are redacted by default. After invocation starts, an exception or error result is non-retryable unless the tool explicitly returns `ToolResult(..., retry_safe=True)` because it knows no external effect occurred.

The default in-memory store protects one engine process only. Configure `ToolIdempotencyConfig(backend="sqlite")` or inject a shared `ToolIdempotencyStore` for restart or multi-worker coordination. `prune_completed()` removes only outcomes explicitly marked retry-safe; in-progress and uncertain terminal records require operator reconciliation. Call `complete()` with the verified result when the external effect is confirmed, or `delete()` only after confirming no effect occurred. Keep the same key at the downstream API boundary whenever that API supports native idempotency.

AnyCode classifies `bash`, file writes, file edits, knowledge saves, and all MCP tools as side-effecting. MCP `readOnlyHint` annotations are advisory server metadata and are not trusted to bypass idempotency controls.

## Gate sensitive work with human approval

Human-in-the-loop (HITL) approval gates intercept a task before it executes and wait for a person to allow or deny it. Reach for them whenever an action is hard to undo or crosses a trust boundary:

- Irreversible actions such as deleting data or publishing changes.
- External side effects such as sending messages, opening pull requests, or calling paid APIs.
- Workflows that touch private or regulated data.

An approval gate turns "the agent did something surprising" into "a human approved every sensitive step," which is the difference between a demo and an internal tool you can trust.

## Add verification gates

Verification sensors run computational checks at defined phases — before tools, after tools, after a task, or after a whole team finishes — and can block a run when a check fails. This lets output pass a real quality bar rather than a model's self-assessment.

Built-in sensor factories:

| Sensor | Checks |
| --- | --- |
| `ruff` | Lint rules on source files. |
| `pyright` | Static type correctness. |
| `pytest` | Test suites. |
| `schema` | Output shape against a schema. |
| `regex` | Lightweight pattern checks on output. |

Declare sensors in a `verification` block. Here, a `regex` sensor runs after each task and blocks the run unless the output matches `DONE`:

```yaml title="team.yaml"
verification:
  - name: regex
    kind: computational
    phases: [after_task]
    block_on_failure: true
    options:
      pattern: DONE
      expect: match
```

Set `block_on_failure: true` for gates that must pass, and use `phases` to place the check where it matters — for example, running `pytest` `after_team` before you trust a full workflow. See [Use YAML Config](yaml-config.md) for the surrounding config structure.

## Preserve state with checkpoints and durable runs

Checkpointing records task state and agent results so a team run can resume after a failure instead of restarting from zero. Durable runs extend this with run metadata, transcript events, heartbeats, wake conditions, and turn checkpoints — the foundation for long or resumable workflows.

Use checkpoints for any workflow where rerunning completed tasks would waste time, money, or external rate limits. For durable runs operated from the terminal, the CLI exposes the full lifecycle:

```bash title="Inspect durable runs"
anycode runs list
anycode runs show
anycode runs tail
anycode runs audit
anycode runs sweep
```

## Cancel and shut down cleanly

Cancelling an agent run raises `asyncio.CancelledError` to the caller rather than returning a normal failure result. The agent state becomes `cancelled`, lifecycle listeners receive a terminal `cancelled` phase with a `user_cancelled` stop reason, and durable runs persist that state with a final checkpoint. Concurrent tool calls and wave siblings are cancelled and awaited before the parent exits; shell-tool cancellation terminates and reaps the spawned process tree.

Call `await engine.close()` during application shutdown. The orchestrator stops and drains tracked standalone, coordinator, team, reflection, and handoff operations before disconnecting MCP clients or closing persistent stores. Cancellation remains cooperative for custom tools and integrations: do not swallow `asyncio.CancelledError`, and put resource cleanup in `finally` blocks or async context managers.

## Manage context pressure

Context policies define how AnyCode responds as conversation history grows. A policy can trim, mask, offload, compact, or hand off context while preserving what matters — task state and verification failures survive even as lower-priority content is reduced. This keeps long runs inside a model's context window without silently dropping the information a downstream agent needs.

## Protect exported and persisted data

AnyCode redacts recognized credentials at built-in telemetry and persistence boundaries by default. Structured keys such as `api_key`, `authorization`, `password`, and provider-specific token fields are replaced, as are common token formats found inside free-form output and exception messages.

The policy covers console, JSONL, and OTLP span exports, telemetry-event serialization, workflow and turn checkpoints, run transcripts and metadata, context and session-chain artifacts, persistent SQLite/Redis/Chroma/knowledge memory, eval reports, and harness artifacts. It does not mutate the active model conversation or tool result in memory.

Use `redact_sensitive`, `redact_text`, and `safe_exception_message` for custom exporters or persistence adapters. Redaction is not encryption or data-loss prevention: classify data before a run, minimize what agents can access, and protect storage and telemetry sinks.

For durable runs, `FilesystemRunStore(payload_protector=...)` accepts a `RunPayloadProtector` implementation for KMS- or HSM-backed envelope encryption. AnyCode defines the byte contract and versioned storage envelope but does not manage keys. `RunRetentionPolicy` bounds completed, failed, and cancelled runs by age and count; pass it to `sweep_once` or `RunScheduler`, or use `anycode runs sweep --retention-days ... --max-runs ...`. No retention policy means no automatic deletion.

## Inspect run results

Every run returns a record you can log, assert on, or serialize. `AgentRunResult` and `TeamRunResult` expose what you need for debugging and audit:

| Field | What it reports |
| --- | --- |
| Success state and output | Whether the run succeeded and the produced result. |
| Token usage | Input and output tokens consumed. |
| Tool calls | Tools invoked during the run. |
| Lifecycle events | Ordered state transitions. |
| Context manifests | How context was assembled and managed. |
| Verification results and gate decisions | Which sensors ran and what they decided. |
| Handoff requests and handoff chains | Routing and handoff between agents. |
| Cost reports | Spend, when cost tracking is enabled. |

Because results are structured, you can turn any run into a test assertion or a telemetry event rather than eyeballing logs.

## Next steps

- [Production readiness checklist](production-readiness.md) — make a workload-specific go or no-go decision.
- [Security and threat model](../reference/security.md) — review trust boundaries, enforced invariants, and residual risks.
- [Use YAML Config](yaml-config.md) — declare budgets, routing, and verification gates in one file.
- [Run a Multi-Agent Team](multi-agent-team.md) — the team runtime these controls wrap around.
- [CLI reference](../reference/cli.md) — the `anycode runs` commands for durable, resumable runs.
- [Public API reference](../reference/public-api.md) — `CostConfig`, `AgentConfig`, and the run-result types.
