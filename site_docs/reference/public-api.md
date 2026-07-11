---
title: Public API Reference — AnyCode Python Framework
description: API reference for the AnyCode Python framework: orchestrator, TaskSpec, agent and team config models, result types, tool registry, and provider adapters.
keywords: AnyCode, AnyCode Python, public API, orchestrator, TaskSpec, AgentConfig, TeamConfig, ToolRegistry, provider adapter, mkdocstrings
---

# Public API

This page is the stable, importable surface of the AnyCode Python framework, generated directly from source docstrings so signatures and fields stay in sync with the code. Use it to look up exact arguments, methods, and model attributes.

Every symbol below is exported from the package root, so `from anycode import AnyCode, TaskSpec, AgentConfig` is the supported import style — treat internal submodule paths as private.

!!! tip "New here? Read the concepts first"
    This reference lists signatures, not tutorials. For how these pieces fit together, start with the [Concepts overview](../concepts/overview.md), then the [YAML config guide](../guides/yaml-config.md) and [Tools guide](../guides/tools.md).

## Core classes

The orchestrator is the primary entry point: configure it once, then build agents, assemble teams, and run tasks. `TaskSpec` describes a single unit of work handed to an agent or team.

::: anycode.core.orchestrator.AnyCode
    options:
      members:
        - __init__
        - configure
        - build_agent
        - create_team
        - run_agent
        - run_team
        - run_tasks
        - register_plugin
        - load_installed_plugins
        - list_plugins

::: anycode.core.orchestrator.TaskSpec
    options:
      members:
        - __init__

## Configuration models

These typed models define agents, teams, and orchestrator-wide settings. They accept the same fields you set declaratively in a [YAML or TOML config](../guides/yaml-config.md).

::: anycode.types.AgentConfig

::: anycode.types.TeamConfig

::: anycode.types.OrchestratorConfig

## Result models

Run methods return structured results — output, token usage, and per-task detail — rather than raw strings, so you can inspect and audit every run programmatically.

::: anycode.types.AgentRunResult

::: anycode.types.TeamRunResult

::: anycode.types.ToolResult

## Tool APIs

Define custom tools with `define_tool`, manage them through the `ToolRegistry`, and run them via the `ToolExecutor`. See the [Tools guide](../guides/tools.md) for end-to-end examples.

::: anycode.tools.registry.define_tool

::: anycode.tools.registry.ToolRegistry
    options:
      members:
        - register
        - get
        - list
        - has
        - deregister
        - to_tool_defs

::: anycode.tools.executor.ToolExecutor

Side-effecting tools use atomic claims to prevent duplicate execution. The in-memory store coordinates one process; the SQLite store persists claims and results across restarts.

::: anycode.tools.idempotency.ToolIdempotencyStore

::: anycode.tools.idempotency.InMemoryToolIdempotencyStore

::: anycode.tools.idempotency.SQLiteToolIdempotencyStore

::: anycode.tools.idempotency.IdempotencyClaim

::: anycode.types.ToolIdempotencyConfig

## Provider API

`create_adapter` resolves the right provider adapter for a configured model — Anthropic, OpenAI, Google, Bedrock, Azure OpenAI, Ollama, or a plugin-registered backend — giving the runtime a uniform interface across providers.

::: anycode.providers.adapter.create_adapter

::: anycode.providers.resilience.ProviderCapacityLimiter

::: anycode.types.ProviderResilienceConfig

## Durable storage APIs

`AgentRunner` accepts any backend satisfying `RunStore`. The built-in filesystem backend adds local atomic writes, protected-payload envelopes, checkpoint recovery, and terminal-run retention.

::: anycode.runstore.protocol.RunStore

::: anycode.runstore.protocol.RunPayloadProtector

::: anycode.runstore.store.FilesystemRunStore

::: anycode.types.RunRetentionPolicy

## Next steps

- [Concepts overview](../concepts/overview.md) — how the orchestrator, agents, and tools fit together.
- [Agents and teams](../concepts/agents-and-teams.md) — model the roles these APIs configure.
- [Tools guide](../guides/tools.md) — build and register custom tools.
- [CLI reference](cli.md) — drive the same runtime from the command line.
