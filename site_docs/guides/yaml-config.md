---
title: "AnyCode YAML Config: Define Agent Teams in One File"
description: Configure AnyCode agents, tasks, dependencies, cost, routing, verification, and context engineering in one YAML file, then run it via the CLI or Python.
keywords: AnyCode YAML config, AnyCode from_config, multi-agent YAML, agent config file, task dependencies, cost budget YAML, verification gates, context engineering, anycode run CLI
---

# Configure AnyCode With YAML

Define a whole AnyCode team — its agents, tasks, dependencies, and runtime controls — in a single YAML file, then run it with one CLI command or load it in Python.

A YAML config is the most reviewable, repeatable way to describe an AnyCode workflow. Instead of wiring agents and task graphs in code, you declare them once, commit the file, and let the framework build the run. The CLI executes a config directly with `uv run anycode run team.yaml`, and Python loads the same file with `AnyCode.from_config()` — so the file you review is exactly the file that runs.

!!! tip "Install the CLI extra first"
    YAML and TOML config loading ships in the optional `cli` extra. Add it with `uv add "anycode-py[cli]"` before running or scaffolding a config. See the [installation guide](../getting-started/installation.md) for the full extras matrix.

## How an AnyCode YAML config is structured

A config is a single YAML document with a small set of top-level keys. You add only the blocks you need — a two-agent config can be a dozen lines, and the same file scales up to routing, verification, and context policies without changing shape.

| Top-level key | Purpose |
| --- | --- |
| `format_version` | Declarative config contract. Use `1`; omitted files are treated as legacy v1. |
| `name` | Human-readable team name. |
| `agents` | The agents on the team — each has a `name`, `provider`, `model`, `system_prompt`, and `tools`. |
| `tasks` | Work items and their wiring — each has a `title`, `description`, `assignee`, and optional `depends_on`. |
| `shared_memory` | Share memory across agents on the team. |
| `max_concurrency` | Cap how many agents run at once. |
| `provider_resilience` | Provider concurrency, request pacing, retry, deadline, and circuit settings. |
| `tool_idempotency` | Process-local or restart-safe claims for side-effecting tools. |
| `cost` | Spend budget and overspend behavior. |
| `routing` | Model routing across providers and models. |
| `verification` | Quality-gate sensors that run at defined phases. |
| `context_engineering` | Context-window budgeting and per-section overflow policies. |

!!! note "More blocks are available"
    AnyCode configs can also declare RAG memory and reflection blocks. Browse the [examples cookbook](https://github.com/Quantlix/anycode) and the [public API reference](../reference/public-api.md) for the full set of fields.

## Write a minimal team config

Start with one agent and one task. This is a complete, runnable config.

```yaml title="team.yaml"
format_version: 1
name: docs-crew
agents:
  - name: writer
    provider: anthropic
    model: claude-haiku-4-5
    system_prompt: Write concise developer documentation.
    tools: []

tasks:
  - title: Draft overview
    description: Explain the project in one short page.
    assignee: writer
```

Each agent needs a `provider` and `model`; the `assignee` on a task must match an agent `name`. Give an agent an empty `tools: []` list when it only needs to reason and write.

Unknown root, agent, task, and nested configuration fields fail validation. This catches misspellings and prevents an older runtime from silently ignoring a setting written for a newer release. See the [compatibility policy](../reference/compatibility.md) before changing a persisted config format.

## Run the config

Run the file directly from the terminal, or load it in Python when you want to inspect the result programmatically. Both paths execute the same team.

=== "CLI"

    ```bash title="Run team.yaml"
    uv run anycode run team.yaml
    ```

=== "Python"

    ```python title="run_team.py"
    import asyncio

    from anycode import AnyCode


    async def main() -> None:
        async with AnyCode.from_config("team.yaml") as engine:
          result = await engine.run_team_from_config()
        print(result.success)


    asyncio.run(main())
    ```

`AnyCode.from_config()` returns a ready-to-run engine; `run_team_from_config()` executes the tasks declared in the file and returns a `TeamRunResult` you can inspect, log, or serialize.

## Add task dependencies

Chain tasks with `depends_on` to build a dependency-aware pipeline. AnyCode topologically sorts the graph and runs independent tasks concurrently, so you describe *what depends on what* instead of manually sequencing prompts.

```yaml title="team.yaml"
tasks:
  - title: Plan
    description: Create the implementation plan.
    assignee: planner
  - title: Build
    description: Implement the plan.
    assignee: builder
    depends_on:
      - Plan
  - title: Review
    description: Review the implementation.
    assignee: reviewer
    depends_on:
      - Build
```

Each value in `depends_on` references another task's `title`, so keep titles unique within a workflow. `Build` waits for `Plan`, and `Review` waits for `Build`.

## Add runtime controls

Cost budgets, routing, and verification gates all live in the same file. Add them once and every run of the config inherits them.

```yaml title="team.yaml"
cost:
  budget_usd: 2.00
  on_budget_exceeded: warn

routing:
  enabled: true

verification:
  - name: pytest
    kind: computational
    phases: [after_team]
    block_on_failure: true
    options:
      command: uv run python -m pytest

provider_resilience:
  max_concurrency: 4
  requests_per_minute: 120
  capacity_scope: shared-production-key
  capacity_wait_timeout_seconds: 30

tool_idempotency:
  backend: sqlite
  path: .anycode/tool-idempotency.db
  redact_sensitive_data: true
```

The keys above map to the controls you tune most often:

| Block | Key | What it sets |
| --- | --- | --- |
| `cost` | `budget_usd` | Spend ceiling for the run, in USD. |
| `cost` | `on_budget_exceeded` | Action when the budget is reached (for example, `warn`). |
| `routing` | `enabled` | Turn provider and model routing on. |
| `verification` | `name` | Sensor to run, such as `pytest` or `regex`. |
| `verification` | `kind` | Sensor category, such as `computational`. |
| `verification` | `phases` | When the sensor runs, such as `after_team` or `after_task`. |
| `verification` | `block_on_failure` | Fail the run when the sensor fails. |
| `verification` | `options` | Sensor-specific settings, such as `command`, `pattern`, and `expect`. |
| `provider_resilience` | `max_concurrency` | Simultaneous provider attempts shared by agents in one scope. |
| `provider_resilience` | `requests_per_minute` | Evenly paced request starts; every retry counts. |
| `provider_resilience` | `capacity_scope` | Quota identity shared by adapters; use separate values for separate API-key quotas. |
| `provider_resilience` | `capacity_wait_timeout_seconds` | Queue wait before the call is load-shed. |
| `tool_idempotency` | `backend` | `memory` for one process or `sqlite` for restart-safe claims. |
| `tool_idempotency` | `path` | SQLite database path when the persistent backend is selected. |

For the full catalog of budgets, gates, and durable-run controls, see [Production Controls](production-controls.md).

An agent can override the top-level policy with its own `provider_resilience` mapping. Give that override a distinct `capacity_scope` when it represents an independent quota; conflicting settings under one scope are rejected.

## Add context engineering

When a run may carry large tool outputs, long histories, or model-specific context windows, declare a `context_engineering` block. It budgets the context window and sets how each section behaves under pressure.

```yaml title="team.yaml"
context_engineering:
  enabled: auto
  window:
    reserved_response_tokens: 2048
  sections:
    system_instructions:
      priority: required
      overflow: error
    tool_results:
      priority: medium
      overflow: summarize
```

Here, system instructions are `required` and error on overflow so they are never silently dropped, while tool results are `medium` priority and are summarized when the window fills. Use context policies to keep critical task state intact while lower-priority content is trimmed or compacted.

## Scaffold a project

To start from a working layout instead of a blank file, let the CLI generate one:

```bash title="Create a starter project"
uv run anycode init my-anycode-project
```

The scaffold includes `team.yaml`, `main.py`, `.env.example`, a `tools/` package, and `.gitignore` — everything you need to edit the config and run it.

!!! warning "Alpha framing"
    AnyCode is alpha-stage software. Config keys, defaults, and runtime behavior can change between releases, so pin your version and re-check this guide when you upgrade.

## Next steps

- [Production Controls](production-controls.md) — add budgets, approval gates, verification, and durable runs to a config.
- [Run a Multi-Agent Team](multi-agent-team.md) — build the same team in Python when you need programmatic control.
- [CLI reference](../reference/cli.md) — every `anycode` command, including `run`, `init`, and `inspect`.
- [Public API reference](../reference/public-api.md) — `AnyCode.from_config`, run methods, and result types.
