---
title: Public API Reference — AnyCode Python Framework
description: "Reference AnyCode's public Python API for orchestration, teams, tools, providers, routing, memory, durability, verification, observability, and extensions."
keywords: AnyCode, AnyCode Python, public API, orchestrator, TaskSpec, AgentConfig, TeamConfig, ToolRegistry, provider adapter, durability, verification, MCP, plugins, telemetry, mkdocstrings
---

# Public API

This page is the **curated, workflow-organized reference** for every public subsystem of AnyCode. Each section explains what a group of symbols is for, then generates signatures directly from the source docstrings. If you want a flat, exhaustive index instead, use the [complete API inventory](api-inventory.md), which lists every name in `anycode.__all__`.

Everything documented here is exported from the package root, so `from anycode import AnyCode, TaskSpec, AgentConfig` is always the supported import style. Treat internal submodule paths as private implementation detail. The [compatibility policy](compatibility.md) defines additive patch releases, pre-1.0 breaking-change rules, and persisted-format guarantees.

!!! tip "New here? Read the concepts first"
    This reference lists signatures, not tutorials. For how the pieces fit together, start with the [Concepts overview](../concepts/overview.md), then the [How-to guides](../guides/index.md).

## The core surface

Fifteen symbols cover the overwhelming majority of real use. Everything else on this page
stays public and supported; this is the front door, not a fence.

| Symbol | Use it to |
| --- | --- |
| `Agent` | Build one agent from keyword arguments, run it, stream it, converse with it |
| `tool` | Turn a Python function into an agent tool |
| `Crew` | Run several agents over a list of tasks with dependencies |
| `TaskSpec` | Declare one unit of a crew's work |
| `Workflow` | Compose agents into a state graph with branching and loops |
| `START` / `END` | The virtual entry and exit nodes of a workflow |
| `AnyCode` | The engine underneath, for durability, MCP, plugins, routing, and gates |
| `AgentConfig` | The frozen agent configuration `Agent` builds internally |
| `TeamConfig` | The frozen team configuration `Crew` builds internally |
| `ToolResult` | What a tool returns |
| `ToolRegistry` / `ToolExecutor` | Manual tool wiring, when you need it |
| `SubAgentSpec` | Declare a sub-agent for `Agent(subagents=[...])` |
| `create_adapter` | Construct a provider adapter directly |

Print the same list, with live signatures, at any time:

```bash
anycode api --core
```

See [Recipes](recipes.md) for runnable snippets and
[Using AnyCode from an AI coding agent](llm-guide.md) for a token-efficient orientation.

## How this page is organized

The API is grouped the way you actually use it — from a first agent run out to production controls and the extension surface.

<div class="grid cards" markdown>

-   :material-cog: **[Core orchestration](#core-orchestration)** — the engine, agents, runners, and task specs.
-   :material-tune: **[Configuration models](#configuration-models)** — typed settings for agents, teams, and the engine.
-   :material-package-variant: **[Result models](#result-models)** — structured, auditable run output.
-   :material-account-group: **[Teams and collaboration](#teams-and-collaboration)** — messaging and shared memory.
-   :material-wrench: **[Tools](#tools)** — define, register, secure, and execute typed tools.
-   :material-code-json: **[Structured output](#structured-output)** — force a validated object.
-   :material-server-network: **[Providers](#providers-and-model-adapters)** — model adapters and resilience.
-   :material-sitemap: **[Routing](#routing)** — send each task to the right model.
-   :material-swap-horizontal: **[Handoff](#agent-handoff)** — delegate between agents mid-run.
-   :material-refresh: **[Reflection](#reflection)** — critique-and-revise loops.
-   :material-database: **[Memory and RAG](#memory-and-rag)** — persistence, retrieval, knowledge.
-   :material-window-restore: **[Context engineering](#context-engineering)** — manage the context window.
-   :material-cash: **[Cost](#cost)** — price, track, and cap spend.
-   :material-shield-check: **[Guardrails](#guardrails)** — budgets, hooks, validators.
-   :material-check-decagram: **[Verification and quality gates](#verification-and-quality-gates)** — block bad output.
-   :material-account-check: **[Human-in-the-loop](#human-in-the-loop)** — approval gates.
-   :material-content-save: **[Durable runs](#durable-runs)** — the run store and payload protection.
-   :material-history: **[Checkpointing](#checkpointing)** — save and resume in-flight work.
-   :material-link-variant: **[Session chaining](#session-chaining)** — long-running goal contracts.
-   :material-clock-outline: **[Scheduling and watchdogs](#scheduling-and-watchdogs)** — wake and sweep paused runs.
-   :material-swap-vertical-bold: **[Durability backends](#durability-backends)** — in-memory, SQLite, Dapr.
-   :material-file-document-outline: **[Semantic contracts](#semantic-contracts-preview)** — language-neutral run models.
-   :material-account-key: **[Execution identity](#execution-identity-and-policy)** — tenancy, delegation, policy.
-   :material-cube-outline: **[Sandboxes](#sandboxes)** — isolated command execution.
-   :material-cloud-outline: **[Managed hosting](#managed-hosting)** — admission, drain, Agent Cards.
-   :material-connection: **[MCP](#model-context-protocol-mcp)** — external tool servers.
-   :material-puzzle: **[Plugins](#plugins)** — package tools, providers, sensors.
-   :material-chart-line: **[Observability](#observability)** — tracing, metrics, events, GenAI telemetry.
-   :material-test-tube: **[Evaluation](#evaluation)** — scenario suites and scoring.
-   :material-graph: **[Visualization](#visualization)** — render DAGs and timelines.
-   :material-heart-pulse: **[Lifecycle](#execution-lifecycle)** — phases, loop detection, fingerprints.
-   :material-atom: **[Adaptive harness](#adaptive-harness-advanced)** — evidence and evolution (advanced).
-   :material-shield-lock: **[Security helpers](#security-helpers)** — redaction and safe errors.
-   :material-format-list-checks: **[Tasks](#tasks)** — the low-level task queue.
-   :material-tools: **[Helpers and constants](#helpers-and-constants)** — small utilities and event names.

</div>

---

## Core orchestration

The `AnyCode` engine is the primary entry point: configure it once, then build agents, assemble teams, and run tasks. `TaskSpec` describes a single unit of work handed to an agent or team. `Agent` and `AgentRunner` are the lower-level primitives the engine wires together, available directly when you want fine-grained control.

::: anycode.core.orchestrator.AnyCode
    options:
      members:
        - __init__
        - from_config
        - configure
        - build_agent
        - create_team
        - run_agent
        - run_team
        - run_tasks
        - run_team_from_config
        - connect_mcp_servers
        - disconnect_mcp_servers
        - register_plugin
        - load_installed_plugins
        - list_plugins
        - close

::: anycode.core.orchestrator.TaskSpec
    options:
      members:
        - __init__

::: anycode.core.agent.Agent
    options:
      members:
        - __init__
        - run
        - run_structured
        - prompt
        - stream
        - get_state
        - get_history
        - reset
        - add_tool
        - remove_tool
        - get_tools

::: anycode.core.runner.AgentRunner
    options:
      members:
        - __init__
        - run
        - stream

::: anycode.core.pool.AgentPool

::: anycode.core.scheduler.Scheduler

---

## Configuration models

These typed models define agents, teams, and orchestrator-wide settings. They accept the same fields you set declaratively in a [YAML or TOML config](../guides/yaml-config.md), and every `AnyCode` method that takes a plain `dict` validates it into one of these first.

::: anycode.types.AgentConfig

::: anycode.types.TeamConfig

::: anycode.types.OrchestratorConfig

::: anycode.types.RunnerOptions

::: anycode.types.RunnerStreamingConfig

---

## Result models

Run methods return structured results — output, token usage, and per-task detail — rather than raw strings, so you can inspect and audit every run programmatically.

::: anycode.types.AgentRunResult

::: anycode.types.TeamRunResult

::: anycode.types.RunResult

::: anycode.types.StructuredAgentResult

::: anycode.types.StructuredRunResult

::: anycode.types.ToolResult

::: anycode.types.TokenUsage

::: anycode.types.ToolCallRecord

---

## Teams and collaboration

A `Team` coordinates several agents: it holds their configs, routes messages between them, tracks tasks, and exposes shared memory. `MessageBus` and `SharedMemory` are the collaboration primitives underneath — useful directly when you build custom coordination on top of the engine.

::: anycode.collaboration.team.Team
    options:
      members:
        - __init__
        - get_agents
        - get_agent
        - send_message
        - get_messages
        - broadcast
        - add_task
        - get_tasks
        - get_tasks_by_assignee
        - update_task
        - get_next_task
        - get_shared_memory
        - "on"
        - emit

::: anycode.collaboration.message_bus.MessageBus
    options:
      members:
        - __init__
        - send
        - broadcast
        - get_unread
        - get_all
        - mark_read
        - get_conversation
        - subscribe

::: anycode.collaboration.shared_mem.SharedMemory
    options:
      members:
        - __init__
        - write
        - read
        - list_all
        - list_by_agent
        - get_summary
        - get_store

::: anycode.collaboration.kv_store.InMemoryStore

---

## Tools

Define custom tools with `define_tool`, manage them through the `ToolRegistry`, and run them via the `ToolExecutor`. Every tool is a typed async function backed by a Pydantic input model. See the [Tools guide](../guides/tools.md) for end-to-end examples.

::: anycode.tools.registry.define_tool

::: anycode.tools.registry.ToolRegistry
    options:
      members:
        - __init__
        - register
        - get
        - list
        - has
        - deregister
        - register_from_mcp
        - deregister_prefix
        - to_tool_defs

::: anycode.tools.executor.ToolExecutor
    options:
      members:
        - __init__
        - execute
        - execute_batch

::: anycode.types.ToolDefinition

::: anycode.types.ToolUseContext

::: anycode.types.ToolSecurityPolicy

### Built-in tools

AnyCode ships six built-in tools — `bash`, `file_read`, `file_write`, `file_edit`, `grep`, and `list_files`. Register them all at once, then allow each agent only the names it needs. The [built-in tools reference](built-in-tools.md) documents every parameter and safety limit.

::: anycode.tools.built_in.register_built_in_tools

::: anycode.tools.built_in.BUILT_IN_TOOLS

### Tool idempotency

Side-effecting tools use atomic claims to prevent duplicate execution when a run is retried or resumed. The in-memory store coordinates one process; the SQLite store persists claims and results across restarts.

::: anycode.tools.idempotency.ToolIdempotencyStore

::: anycode.tools.idempotency.InMemoryToolIdempotencyStore

::: anycode.tools.idempotency.SQLiteToolIdempotencyStore

::: anycode.tools.idempotency.create_tool_idempotency_store

::: anycode.tools.idempotency.IdempotencyClaim

::: anycode.types.ToolIdempotencyConfig

---

## Structured output

Force an agent to return a validated Pydantic object instead of free text. The high-level path is `Agent.run_structured`; these helpers implement it and are public so you can drive an adapter directly. See the [structured output guide](../guides/structured-output.md).

::: anycode.structured.output.schema_to_tool_def

::: anycode.structured.output.schema_to_openai_response_format

::: anycode.structured.output.parse_structured_output

::: anycode.structured.output.build_retry_prompt

::: anycode.types.StructuredOutputConfig

---

## Providers and model adapters

`create_adapter` resolves the right provider adapter for a configured model — Anthropic, OpenAI, Google, Bedrock, Azure OpenAI, Ollama, or a plugin-registered backend — giving the runtime one uniform interface across providers. `FakeAdapter` replays a scripted sequence for deterministic tests. `ResilientAdapter` and `ProviderCapacityLimiter` add retries, capacity limits, and failover. See the [providers guide](../guides/providers.md).

::: anycode.providers.adapter.create_adapter

::: anycode.providers.fake.FakeAdapter

::: anycode.providers.fake.FakeResponse

::: anycode.providers.resilience.ResilientAdapter

::: anycode.providers.resilience.ProviderCapacityLimiter

::: anycode.types.ProviderResilienceConfig

::: anycode.types.RetryPolicy

### Adapter protocol and message types

Implement `LLMAdapter` to add a custom backend. These are the request and response types every adapter speaks.

::: anycode.types.LLMAdapter

::: anycode.types.LLMMessage

::: anycode.types.LLMResponse

::: anycode.types.LLMToolDef

::: anycode.types.LLMChatOptions

::: anycode.types.LLMStreamOptions

::: anycode.types.StreamEvent

---

## Routing

Classify a task by complexity and send it to the right model, or apply a full policy that filters providers by region, capability, budget, and latency. See the [routing](../guides/routing.md) and [policy routing](../guides/policy-routing.md) guides.

::: anycode.routing.classifier.classify_task

::: anycode.routing.router.DefaultRouter

::: anycode.routing.policy.PolicyRouter
    options:
      members:
        - __init__
        - route

::: anycode.routing.policy.ModelRoutingRequest

::: anycode.routing.policy.ProviderCapabilityDescriptor

::: anycode.routing.rules.evaluate_rules

::: anycode.routing.rules.match_rule

::: anycode.types.RoutingConfig

::: anycode.types.RoutingRule

::: anycode.types.RouteDecision

::: anycode.types.ComplexityLevel

---

## Agent handoff

Let an agent delegate mid-run to a teammate better suited to the next step. `HandoffExecutor` performs the transfer; the protocol helpers build the prompts and trim carried context. See the [handoff guide](../guides/handoff.md).

::: anycode.handoff.executor.HandoffExecutor
    options:
      members:
        - __init__
        - execute

::: anycode.handoff.protocol.build_handoff_system_prompt

::: anycode.handoff.protocol.build_handoff_user_message

::: anycode.handoff.protocol.trim_context

::: anycode.types.HandoffRequest

::: anycode.types.Handoff

::: anycode.types.HandoffPolicy

---

## Reflection

Run an agent inside a critique-and-revise loop that keeps improving output until it clears a quality threshold or a retry budget. See the [reflection guide](../guides/reflection.md).

::: anycode.reflection.loop.ReflectionLoop
    options:
      members:
        - __init__
        - run

::: anycode.reflection.critic.LLMCritic

::: anycode.reflection.evaluator.parse_critic_json

::: anycode.types.ReflectionConfig

::: anycode.types.Critic

::: anycode.types.CriticResult

---

## Memory and RAG

Persist and retrieve context across runs: `CompositeMemory` for working memory, `RAGRetriever`/`RAGIndexer` for semantic retrieval, and `KnowledgeStore` for durable, verifiable facts. The factories build the configured store from an `OrchestratorConfig`. See the [memory and RAG guide](../guides/memory-and-rag.md).

::: anycode.memory.rag.RAGRetriever
    options:
      members:
        - __init__
        - retrieve
        - format_context

::: anycode.memory.indexer.RAGIndexer

::: anycode.memory.knowledge.KnowledgeStore
    options:
      members:
        - __init__
        - save
        - get
        - list_entries
        - search
        - verify

::: anycode.memory.knowledge.build_knowledge_tools

::: anycode.memory.knowledge.apply_retention

::: anycode.memory.composite.CompositeMemory

::: anycode.memory.vector_store.InMemoryVectorStore

::: anycode.memory.factory.create_memory_store

::: anycode.memory.factory.create_vector_store

::: anycode.types.MemoryConfig

::: anycode.types.RAGConfig

---

## Context engineering

Assemble the context window under an explicit token policy — trim, mask, offload, and compact by pressure — and offload large blocks to artifacts to keep the live window small. See the [context engineering guide](../guides/context-engineering.md).

::: anycode.core.context_manager.ContextManager
    options:
      members:
        - __init__
        - assemble
        - note_actual
        - reconcile

::: anycode.core.context_manager.estimate_messages_tokens

::: anycode.core.context_manager.rebuild_from_handoff

::: anycode.core.context_artifacts.offload_text

::: anycode.core.context_artifacts.restore_text

::: anycode.types.ContextPolicy

::: anycode.types.ContextUsageReport

::: anycode.types.ContextManifest

---

## Cost

Price a model call, track spend in real time, enforce a budget, and render a per-agent or per-model report. `DEFAULT_PRICING` seeds current provider rates. See the [cost tracking guide](../guides/cost-tracking.md).

::: anycode.cost.tracker.CostTracker
    options:
      members:
        - record
        - is_budget_exhausted
        - is_budget_alert_due
        - by_agent
        - by_model
        - get_status

::: anycode.cost.pricing.calculate_cost

::: anycode.cost.pricing.find_pricing

::: anycode.cost.report.build_cost_report

::: anycode.cost.pricing.DEFAULT_PRICING

::: anycode.types.CostConfig

::: anycode.types.CostReport

::: anycode.types.ModelPricing

---

## Guardrails

Cap a run by tokens, turns, and tool calls with `BudgetTracker`; run hooks around each turn; and validate output against blocklists, length limits, and required substrings. See [production controls](../guides/production-controls.md).

::: anycode.guardrails.budget.BudgetTracker
    options:
      members:
        - __init__
        - record_usage
        - record_turn
        - record_tool_call
        - is_exhausted
        - get_exhaustion_reason
        - is_tool_blocked
        - snapshot
        - restore
        - get_status

::: anycode.guardrails.budget.estimate_cost

::: anycode.guardrails.hooks.HookRunner

::: anycode.guardrails.hooks.LoggingHook

::: anycode.guardrails.validators.run_validators

::: anycode.guardrails.validators.MaxLengthValidator

::: anycode.guardrails.validators.ContainsValidator

::: anycode.guardrails.validators.BlocklistValidator

::: anycode.types.GuardrailConfig

---

## Verification and quality gates

Block an agent's output on real evidence: run `ruff`, `pyright`, `pytest`, or a JSON-schema check as sensors, then let a `QualityGate` decide pass or fail. See the [verification gates guide](../guides/verification-gates.md).

::: anycode.verification.gate.QualityGate
    options:
      members:
        - __init__
        - evaluate

::: anycode.verification.gate.decide_gate

::: anycode.verification.sensor.Sensor

::: anycode.verification.sensor.SensorContext

::: anycode.verification.builtins.ruff_sensor

::: anycode.verification.builtins.pyright_sensor

::: anycode.verification.builtins.pytest_sensor

::: anycode.verification.builtins.schema_sensor

::: anycode.types.VerificationResult

::: anycode.types.VerificationSensorConfig

::: anycode.types.QualityGateDecision

---

## Human-in-the-loop

Pause the workflow and wait for a person before a sensitive action runs. `ApprovalManager` pairs an `ApprovalConfig` with a gate — a callback, an interactive terminal, or a webhook. See the [human-in-the-loop guide](../guides/human-in-the-loop.md).

::: anycode.hitl.approval.ApprovalManager
    options:
      members:
        - __init__
        - check_and_request

::: anycode.hitl.channels.CallbackApprovalGate

::: anycode.hitl.channels.StdinApprovalGate

::: anycode.hitl.channels.WebhookApprovalGate

::: anycode.types.ApprovalConfig

::: anycode.types.ApprovalRequest

::: anycode.types.ApprovalResponse

---

## Durable runs

`AgentRunner` accepts any backend satisfying `RunStore`. The built-in filesystem backend adds local atomic writes, protected-payload envelopes, checkpoint recovery, and terminal-run retention. See the [durability guide](../guides/durability.md).

::: anycode.runstore.protocol.RunStore

::: anycode.runstore.protocol.RunPayloadProtector

::: anycode.runstore.store.FilesystemRunStore
    options:
      members:
        - __init__
        - create_run
        - read_record
        - update_status
        - pause_run
        - due_wakes
        - list_runs
        - prune_runs
        - append_event
        - read_events
        - save_checkpoint
        - load_latest_checkpoint

::: anycode.types.DurabilityConfig

::: anycode.types.RunRecord

::: anycode.types.RunRetentionPolicy

::: anycode.types.RunStatus

---

## Checkpointing

Snapshot in-flight team state and restore it after a crash or a deliberate pause. `CheckpointManager` prunes old snapshots and detects when the task specification changed since the last save. See the [durability guide](../guides/durability.md).

::: anycode.checkpoint.manager.CheckpointManager
    options:
      members:
        - __init__
        - auto_save
        - load_latest
        - compute_spec_hash
        - detect_spec_change

::: anycode.checkpoint.store.FilesystemCheckpointStore

::: anycode.types.CheckpointConfig

::: anycode.types.CheckpointData

---

## Session chaining

Keep a long-running goal alive across many bounded sessions, checking each against explicit acceptance criteria until the contract is satisfied. See [durable and resumable runs](../guides/durability.md).

::: anycode.core.session_chain.SessionChain
    options:
      members:
        - __init__
        - run
        - run_session

::: anycode.core.session_chain.load_contract

::: anycode.core.session_chain.save_contract

::: anycode.core.session_chain.contract_status_summary

::: anycode.types.GoalContract

::: anycode.types.GoalCriterion

---

## Scheduling and watchdogs

Resume paused runs when their wake condition fires, and sweep for interrupted or overdue runs on a tick. See [durable and resumable runs](../guides/durability.md).

::: anycode.schedule.scheduler.RunScheduler
    options:
      members:
        - __init__
        - run
        - stop

::: anycode.schedule.scheduler.sweep_once

::: anycode.schedule.scheduler.SweepReport

::: anycode.schedule.tasks.run_scheduled_task

::: anycode.schedule.tasks.ScheduledTask

::: anycode.schedule.tasks.ScheduledTaskResult

::: anycode.types.WakeCondition

---

## Durability backends

Swap the durability substrate without touching runtime code. All three backends satisfy `DurabilityBackend`: in-memory for tests, SQLite for a single node, and Dapr for a distributed state store. Migration helpers move a run between substrates. See [configure durability backends](../guides/durability-backends.md).

::: anycode.backends.protocol.DurabilityBackend

::: anycode.backends.memory.InMemoryDurabilityBackend

::: anycode.backends.sqlite.SQLiteDurabilityBackend

::: anycode.backends.dapr.DaprDurabilityBackend

::: anycode.backends.dapr.DaprHTTPTransport

::: anycode.backends.models.BackendCapabilities

::: anycode.backends.models.BackendSnapshot

::: anycode.backends.models.WorkItem

::: anycode.backends.models.Admission

::: anycode.backends.migration.export_filesystem_run

::: anycode.backends.migration.import_backend_snapshot

---

## Semantic contracts (preview)

Language-neutral, versioned models for runs, tasks, events, and artifacts, plus the pure functions that transition and project them. This surface is a **preview**: it underpins cross-language interoperability and is documented in full on the [semantic contracts](semantic-contracts.md) reference page. The models are re-exported from the root under semantic-prefixed names (`SemanticTask`, `SemanticMessage`, `SemanticRetryPolicy`, `SemanticVerificationResult`) to avoid clashing with the runtime models above.

::: anycode.contracts.models.Run

::: anycode.contracts.models.Task

::: anycode.contracts.models.Event

::: anycode.contracts.models.Artifact

::: anycode.contracts.state.transition_run

::: anycode.contracts.state.transition_task

::: anycode.contracts.state.decide_retry

::: anycode.contracts.state.evaluate_dependencies

::: anycode.contracts.projection.project_run

::: anycode.contracts.projection.validate_event_stream

::: anycode.contracts.effects.canonical_input_digest

::: anycode.contracts.schema.contract_schema_bundle

---

## Execution identity and policy

Carry tenant, principal, and delegation context through a run, and enforce an external authorization decision at each boundary. See [propagate identity and policy](../guides/execution-identity.md).

::: anycode.identity.context.ExecutionContext

::: anycode.identity.context.DelegationGrant

::: anycode.identity.policy.PolicyEnforcer
    options:
      members:
        - __init__
        - evaluate
        - enforce

::: anycode.identity.policy.PolicyRequest

---

## Sandboxes

Run commands and file operations inside an isolated provider, gated by a capability and network policy. `PolicySandboxProvider` wraps any provider with policy checks; Daytona, E2B, Modal, Runloop, Vercel, LangSmith, and companion providers are concrete backends, and `create_sandbox_provider` builds any of them by name. See [run work in sandboxes](../guides/sandbox-providers.md).

::: anycode.sandbox.protocol.SandboxProvider

::: anycode.sandbox.factory.create_sandbox_provider

::: anycode.sandbox.policy.PolicySandboxProvider

::: anycode.sandbox.daytona.DaytonaSandboxProvider

::: anycode.sandbox.e2b.E2BSandboxProvider

::: anycode.sandbox.modal.ModalSandboxProvider

::: anycode.sandbox.runloop.RunloopSandboxProvider

::: anycode.sandbox.vercel.VercelSandboxProvider

::: anycode.sandbox.langsmith.LangSmithSandboxProvider

::: anycode.sandbox.companion.CompanionSandboxAdapter

::: anycode.sandbox.models.SandboxSpec

::: anycode.sandbox.models.SandboxCommand

::: anycode.sandbox.models.SandboxCapabilities

---

## Managed hosting

Integrate AnyCode with a managed host: control admission, signal readiness, drain gracefully, and publish a deployment Agent Card for A2A discovery. See [host AnyCode services](../guides/hosting-services.md).

::: anycode.hosting.lifecycle.HostLifecycle

::: anycode.hosting.agent_card.A2AAgentCard

::: anycode.hosting.agent_card.build_deployment_agent_card

::: anycode.hosting.agent_card.A2A_AGENT_CARD_PATH

---

## Model Context Protocol (MCP)

Connect to MCP servers over stdio or HTTP, discover their tools, and register them in a `ToolRegistry` so agents call them exactly like built-in tools. See the [MCP guide](../guides/mcp.md).

::: anycode.mcp.client.MCPClient
    options:
      members:
        - __init__
        - connect
        - disconnect
        - discover_tools
        - call_tool

::: anycode.mcp.bridge.discover_and_register

::: anycode.mcp.bridge.mcp_tool_to_definition

::: anycode.mcp.bridge.schema_to_pydantic_model

::: anycode.mcp.config.validate_server_config

::: anycode.types.MCPServerConfig

::: anycode.types.MCPTrustPolicy

::: anycode.types.MCPToolInfo

---

## Plugins

Package tools, providers, sensors, and hooks as installable extensions, discovered via entry points and governed by a trust policy. See the [plugins guide](../guides/plugins.md).

::: anycode.plugins.plugin.PluginBase

::: anycode.plugins.registry.PluginRegistry

::: anycode.plugins.discovery.discover_entry_point_plugins

::: anycode.plugins.registry.register_provider_factory

::: anycode.plugins.registry.get_provider_factory

::: anycode.plugins.registry.list_registered_providers

::: anycode.types.PluginManifest

::: anycode.types.PluginTrustPolicy

---

## Observability

Emit OpenTelemetry-shaped spans, collect counters and histograms, publish structured events, and map runs to the GenAI semantic conventions for safe export. See the [observability](../guides/observability.md) and [GenAI telemetry](../guides/genai-telemetry.md) guides.

::: anycode.telemetry.tracer.Tracer
    options:
      members:
        - __init__
        - start_span
        - end_span
        - span
        - async_span
        - force_flush
        - shutdown

::: anycode.telemetry.tracer.Span

::: anycode.telemetry.tracer.ConsoleExporter

::: anycode.telemetry.tracer.JSONLExporter

::: anycode.telemetry.tracer.OTLPExporter

::: anycode.telemetry.metrics.MetricsCollector
    options:
      members:
        - increment
        - record
        - record_token_usage
        - record_cost
        - record_latency
        - record_error
        - record_run
        - get_summary
        - reset

::: anycode.telemetry.metrics.Timer

::: anycode.telemetry.events.EventEmitter

::: anycode.telemetry.events.TelemetryEvent

::: anycode.telemetry.genai.GenAITelemetryMapper

::: anycode.telemetry.genai.GenAITelemetryConfig

---

## Evaluation

Run scenario suites against real or fake providers, score the results, and diff two reports to catch regressions. See the [evaluation guide](../guides/evaluation.md).

::: anycode.eval.suite.run_scenario

::: anycode.eval.suite.run_suite

::: anycode.eval.suite.detect_provider

::: anycode.eval.scenario.load_scenario

::: anycode.eval.scenario.load_scenarios

::: anycode.eval.report.write_report

::: anycode.eval.report.read_report

::: anycode.eval.report.render_markdown

::: anycode.eval.report.compare_reports

::: anycode.types.EvalScenario

::: anycode.types.EvalReport

---

## Visualization

Render a task DAG or a per-agent timeline from a completed run, for docs, dashboards, or a quick terminal view. See the [visualization guide](../guides/visualization.md).

::: anycode.viz.dag.render_dag

::: anycode.viz.timeline.render_timeline

---

## Execution lifecycle

Observe and constrain the agent turn loop: subscribe to phase transitions, detect repeated tool calls that signal a loop, and fingerprint a call for idempotency. See [the execution lifecycle example](https://github.com/Quantlix/anycode/blob/main/examples/18_execution_lifecycle.py).

::: anycode.core.lifecycle.LifecycleEmitter

::: anycode.core.lifecycle.LifecycleListener

::: anycode.core.lifecycle.LoopDetector

::: anycode.core.lifecycle.fingerprint_call

::: anycode.core.lifecycle.is_valid_transition

::: anycode.types.ExecutionPhase

::: anycode.types.LifecycleEvent

::: anycode.types.StopReason

---

## Adaptive harness (advanced)

The harness turns run trajectories into structured evidence and, optionally, drives a governed loop that proposes and evaluates changes to the harness itself. Production runs need only the registry and evidence modules; evolution and meta-optimization are opt-in. This is an advanced surface — most applications never touch it.

::: anycode.harness.evidence.EvidenceCollector
    options:
      members:
        - __init__
        - record
        - finalize

::: anycode.harness.registry.HarnessRegistry

::: anycode.harness.registry.build_default_registry

::: anycode.harness.distill.distill_evidence

::: anycode.harness.failure_taxonomy.categorize_run

::: anycode.types.EvidencePacket

::: anycode.types.HarnessManifest

---

## Security helpers

Redact secrets from text and structures before they reach logs, telemetry, or persisted payloads, and turn an exception into a message safe to surface. See the [security reference](security.md).

::: anycode.security.redaction.redact_sensitive

::: anycode.security.redaction.redact_text

::: anycode.security.redaction.safe_exception_message

::: anycode.security.redaction.REDACTED_SECRET

---

## Tasks

The low-level task primitives the engine builds on: a dependency-aware `TaskQueue` and pure functions to create, order, and validate tasks. Most applications use `TaskSpec` and `run_tasks` instead; reach for these when you build custom scheduling.

::: anycode.tasks.queue.TaskQueue

::: anycode.tasks.task.create_task

::: anycode.tasks.task.get_task_dependency_order

::: anycode.tasks.task.is_task_ready

::: anycode.tasks.task.validate_task_dependencies

::: anycode.types.Task

::: anycode.types.TaskStatus

---

## Helpers and constants

Small utilities used across the framework, and the canonical event-name constants for subscribing to orchestrator and queue events.

::: anycode.helpers.concurrency_gate.Semaphore

::: anycode.helpers.usage_tracker.merge_usage

::: anycode.helpers.uuid7.uuid7

The `stop_reasons` module namespace exposes canonical stop-reason constants for comparisons and integrations. Event-name constants (`ORCH_EVENT_*`, `QUEUE_EVENT_*`) are exported from the root for subscribing to lifecycle and queue events; they are listed in full in the [complete API inventory](api-inventory.md).

---

## Next steps

- [Complete API inventory](api-inventory.md) — every supported package-root export, as a flat index.
- [Concepts overview](../concepts/overview.md) — how the orchestrator, agents, and tools fit together.
- [How-to guides](../guides/index.md) — task-oriented recipes with complete, runnable examples.
- [Configuration reference](configuration.md) — every field on the core config models.
- [CLI reference](cli.md) — drive the same runtime from the command line.
