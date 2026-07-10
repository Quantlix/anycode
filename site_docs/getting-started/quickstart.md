---
title: "AnyCode Quickstart — Run Your First Agent and Team"
description: Run your first AnyCode agent, then a dependency-aware team of planner, writer, and reviewer agents, using provider auto-detection for Anthropic or OpenAI.
keywords: anycode quickstart, run ai agent python, multi-agent team, task dependencies, anthropic openai agent, run_agent, run_tasks, create_team
---

# AnyCode Quickstart

Run your first AnyCode agent in a few lines of Python, then coordinate a three-agent team with task dependencies. This quickstart uses provider auto-detection, so it works whether you have an Anthropic or an OpenAI key.

You will build two scripts: a single agent that answers one prompt, and a planner → writer → reviewer team where each task feeds the next. Both use the same core API — `AnyCode`, `run_agent`, `run_tasks`, and `create_team` — so the jump from one agent to a coordinated team is small.

!!! note "Before you start"
    You need AnyCode installed on Python 3.12+ and at least one provider key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in your environment. See [Installation](installation.md) if you have not done that yet.

## Run your first agent

Create `quickstart_agent.py`. The script picks a provider and model from whichever key it finds, spins up an `AnyCode` engine, and runs one agent to completion:

```python title="quickstart_agent.py"
import asyncio
import os

from dotenv import load_dotenv

from anycode import AnyCode

load_dotenv()


def resolve_model() -> tuple[str, str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY first.")


async def main() -> None:
    provider, model = resolve_model()
    engine = AnyCode(config={"default_provider": provider, "default_model": model})

    result = await engine.run_agent(
        config={
            "name": "explainer",
            "provider": provider,
            "model": model,
            "system_prompt": "You explain Python clearly and briefly.",
            "tools": [],
            "max_turns": 2,
        },
        prompt="Explain what an async generator is in two sentences.",
    )

    print(result.output)
    print(f"tokens: in={result.token_usage.input_tokens} out={result.token_usage.output_tokens}")


asyncio.run(main())
```

Run it:

```bash
uv run python quickstart_agent.py
```

The `run_agent` call returns a result object with two things worth noting:

- `result.output` — the agent's final text response.
- `result.token_usage` — input and output token counts for the run.

!!! tip "Give the agent tools"
    This agent runs with `tools: []`, so it only reasons and replies. To let an agent read files, run commands, or call your own functions, pass built-in or custom tools — see [Work with tools](../guides/tools.md).

## Run a team with dependencies

A single agent answers one prompt. A team coordinates several agents through a task graph. Create `quickstart_team.py` — it defines a `planner`, a `writer`, and a `reviewer`, then wires three tasks so each one depends on the previous:

```python title="quickstart_team.py"
import asyncio
import os

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, TaskSpec, TeamConfig

load_dotenv()

PROVIDER = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
MODEL = "claude-haiku-4-5" if PROVIDER == "anthropic" else "gpt-4o-mini"


async def main() -> None:
    engine = AnyCode(config={"max_concurrency": 3})

    team = engine.create_team(
        "guide-crew",
        TeamConfig(
            name="guide-crew",
            shared_memory=True,
            agents=[
                AgentConfig(
                    name="planner",
                    provider=PROVIDER,
                    model=MODEL,
                    system_prompt="Create concise technical plans.",
                    tools=[],
                ),
                AgentConfig(
                    name="writer",
                    provider=PROVIDER,
                    model=MODEL,
                    system_prompt="Turn plans into clear developer documentation.",
                    tools=[],
                ),
                AgentConfig(
                    name="reviewer",
                    provider=PROVIDER,
                    model=MODEL,
                    system_prompt="Review documentation for clarity and missing steps.",
                    tools=[],
                ),
            ],
        ),
    )

    result = await engine.run_tasks(
        team,
        [
            TaskSpec(
                title="Plan guide",
                description="Outline a getting started guide for a Python agent framework.",
                assignee="planner",
            ),
            TaskSpec(
                title="Draft guide",
                description="Write the guide using the planner output.",
                assignee="writer",
                depends_on=["Plan guide"],
            ),
            TaskSpec(
                title="Review guide",
                description="Review the draft and list concrete improvements.",
                assignee="reviewer",
                depends_on=["Draft guide"],
            ),
        ],
    )

    print(f"success={result.success}")
    for agent_name, agent_result in result.agent_results.items():
        print(f"\n[{agent_name}]\n{agent_result.output[:600]}")


asyncio.run(main())
```

Run it:

```bash
uv run python quickstart_team.py
```

The key pieces of the team API are:

- `TeamConfig` and `AgentConfig` — declare the team and the role, provider, model, and tools of each agent.
- `shared_memory=True` — lets agents share context across the run.
- `TaskSpec(..., depends_on=[...])` — declares which tasks must finish before a task can start.
- `create_team` and `run_tasks` — register the team and execute the task graph.

## What happened

The team example creates three agents and three tasks. When you call `run_tasks`, AnyCode:

1. Resolves the `depends_on` relationships into a dependency graph.
2. Runs each ready task in a concurrent wave, bounded by `max_concurrency`.
3. Passes each dependency's result into the prompts of the tasks that follow it.
4. Returns a `TeamRunResult` with per-agent outputs, token usage, lifecycle data, and optional verification data.

Because `Draft guide` depends on `Plan guide` and `Review guide` depends on `Draft guide`, the three tasks run in order — the writer sees the planner's output, and the reviewer sees the writer's draft.

## Next steps

- [Agents and teams](../concepts/agents-and-teams.md) — the mental model behind agents, tasks, and team runs.
- [Run a multi-agent team](../guides/multi-agent-team.md) — a fuller walkthrough of team coordination.
- [Work with tools](../guides/tools.md) — let agents read files, call APIs, or run local functions.
- [Use YAML config](../guides/yaml-config.md) — define and run the same workflows outside Python code.
