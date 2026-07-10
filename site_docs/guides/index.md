---
title: "AnyCode How-to Guides"
description: "Task-oriented recipes for AnyCode: teams, tools, streaming, reasoning, memory and RAG, durability, routing, handoff, cost, verification, eval, telemetry, and more."
keywords: AnyCode how to, guides, streaming, reasoning models, memory rag, durable runs, routing, handoff, cost tracking, verification gates, evaluation, observability, plugins
---

# How-to Guides

How-to guides are **task-oriented**: each one solves a specific problem and assumes you've done the [tutorials](../getting-started/index.md). Jump straight to the one you need.

## Build and run

<div class="grid cards" markdown>

-   **Run a multi-agent team** — task graphs, dependencies, shared memory, concurrent waves.

    [:octicons-arrow-right-24: Multi-agent team](multi-agent-team.md)

-   **Work with tools** — built-in tools and custom Pydantic tools.

    [:octicons-arrow-right-24: Tools](tools.md)

-   **Structured output** — force a validated Pydantic object instead of free text.

    [:octicons-arrow-right-24: Structured output](structured-output.md)

-   **Configure providers** — Anthropic, OpenAI, Google, Ollama, Bedrock, Azure, and resilience.

    [:octicons-arrow-right-24: Providers](providers.md)

</div>

## Streaming and reasoning

<div class="grid cards" markdown>

-   **Stream agent output** — incremental text, thinking, and tool events; configure fallback.

    [:octicons-arrow-right-24: Streaming](streaming.md)

-   **Use reasoning models** — extended thinking and reasoning effort per provider.

    [:octicons-arrow-right-24: Reasoning models](reasoning-models.md)

</div>

## Memory and context

<div class="grid cards" markdown>

-   **Memory and RAG** — persistent backends, shared memory, and automatic retrieval.

    [:octicons-arrow-right-24: Memory and RAG](memory-and-rag.md)

-   **Engineer the context window** — trim, mask, offload, and compact by pressure.

    [:octicons-arrow-right-24: Context engineering](context-engineering.md)

</div>

## Coordination

<div class="grid cards" markdown>

-   **Route tasks by complexity** — classify tasks and send each to the right model.

    [:octicons-arrow-right-24: Routing](routing.md)

-   **Agent handoff** — let an agent delegate to a teammate mid-run.

    [:octicons-arrow-right-24: Handoff](handoff.md)

-   **Self-reflection** — critique and revise output against a quality threshold.

    [:octicons-arrow-right-24: Reflection](reflection.md)

</div>

## Reliability and ops

<div class="grid cards" markdown>

-   **Durable and resumable runs** — checkpoint, resume, chain, and schedule.

    [:octicons-arrow-right-24: Durability](durability.md)

-   **Human-in-the-loop approval** — gate sensitive actions on a person.

    [:octicons-arrow-right-24: Approval](human-in-the-loop.md)

-   **Track and cap cost** — measure spend per agent and enforce a budget.

    [:octicons-arrow-right-24: Cost tracking](cost-tracking.md)

-   **Verification gates** — block output on lint, types, tests, and schema.

    [:octicons-arrow-right-24: Verification](verification-gates.md)

-   **Evaluate agents** — scenario suites, scoring, and regression checks.

    [:octicons-arrow-right-24: Evaluation](evaluation.md)

-   **Observability** — tracing, metrics, and events.

    [:octicons-arrow-right-24: Observability](observability.md)

-   **Production controls** — budgets, gates, and durability together.

    [:octicons-arrow-right-24: Production controls](production-controls.md)

</div>

## Integrate and extend

<div class="grid cards" markdown>

-   **Connect MCP servers** — stdio and HTTP transports with bearer-token auth.

    [:octicons-arrow-right-24: MCP servers](mcp.md)

-   **Extend with plugins** — package tools, providers, sensors, and hooks.

    [:octicons-arrow-right-24: Plugins](plugins.md)

-   **Visualize runs** — render task DAGs and per-agent timelines.

    [:octicons-arrow-right-24: Visualization](visualization.md)

-   **Use YAML config** — declare teams, agents, and tasks in a file.

    [:octicons-arrow-right-24: YAML config](yaml-config.md)

</div>
