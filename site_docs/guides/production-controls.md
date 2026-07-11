---
title: "AnyCode Production Controls: Budgets, Gates, and Checkpoints"
description: Add cost and token budgets, human approval gates, output validators, checkpoints, durable runs, and verification quality gates to AnyCode agent workflows.
keywords: AnyCode production controls, cost budgets, token budgets, HITL approval gates, verification gates, checkpoints, durable runs, context policies, telemetry, output validators
---

# Production Controls

Add cost budgets, human approval gates, verification gates, and durable checkpoints to your AnyCode workflows so prototypes behave more like production systems — with limits, audit trails, and resumable state.

AnyCode is alpha-stage software, so treat these as production-*oriented* controls, not a production guarantee. They exist so you can test production-like behavior in prototypes and internal tools: cap spend, require a human on sensitive steps, gate output on real quality checks, and resume long runs after a failure instead of paying to redo finished work.

!!! warning "Production-oriented, not production-ready"
    These controls harden a run, but AnyCode is not yet meant for production systems — especially those handling sensitive data, irreversible actions, or customer workloads. Keep agents in disposable workspaces and scope tool access tightly.

## What production controls does AnyCode ship?

Each control below is available today and can be layered onto a single agent run or a whole team. Start with limits, then add gates and durability as a workflow grows.

| Control | What it gives you |
| --- | --- |
| Cost and token budgets | A hard ceiling on spend and turns before a run touches live providers. |
| HITL approval gates | A human checkpoint before sensitive or irreversible tasks execute. |
| Output validators | Programmatic checks on agent output before it is accepted. |
| Turn hooks | Callbacks around each turn for logging, mutation, or early stops. |
| Structured output | Schema-shaped responses instead of free text. |
| Verification quality gates | `ruff`, `pyright`, `pytest`, `schema`, and `regex` sensors that can block a run. |
| Checkpoints and resume | Recorded task state and agent results so a run can continue after a failure. |
| Durable runs | Run metadata, transcript events, heartbeats, wake conditions, and turn checkpoints. |
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

## Manage context pressure

Context policies define how AnyCode responds as conversation history grows. A policy can trim, mask, offload, compact, or hand off context while preserving what matters — task state and verification failures survive even as lower-priority content is reduced. This keeps long runs inside a model's context window without silently dropping the information a downstream agent needs.

## Protect exported and persisted data

AnyCode redacts recognized credentials at built-in telemetry and persistence boundaries by default. Structured keys such as `api_key`, `authorization`, `password`, and provider-specific token fields are replaced, as are common token formats found inside free-form output and exception messages.

The policy covers console and OTLP span exports, telemetry-event serialization, workflow and turn checkpoints, run transcripts and metadata, context and session-chain artifacts, persistent SQLite/Redis/Chroma/knowledge memory, eval reports, and harness artifacts. It does not mutate the active model conversation or tool result in memory.

Use `redact_sensitive`, `redact_text`, and `safe_exception_message` for custom exporters or persistence adapters. Redaction is not encryption or data-loss prevention: classify data before a run, minimize what agents can access, protect storage and telemetry sinks, and define retention outside the process.

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

- [Use YAML Config](yaml-config.md) — declare budgets, routing, and verification gates in one file.
- [Run a Multi-Agent Team](multi-agent-team.md) — the team runtime these controls wrap around.
- [CLI reference](../reference/cli.md) — the `anycode runs` commands for durable, resumable runs.
- [Public API reference](../reference/public-api.md) — `CostConfig`, `AgentConfig`, and the run-result types.
