---
title: "Run a Crew — Multi-Agent Teams With Dependent Tasks"
description: Build an AnyCode Crew from agents and tasks, choose sequential or dependency execution, set expected output, and read results without touching the orchestrator.
keywords: anycode crew, multi agent python, dependent tasks, sequential process, expected output, crew result, agent team, task dependencies, crew kickoff
---

# Run a crew

A `Crew` is a team of agents working through a list of tasks. It owns an `AnyCode` engine
and a `Team` underneath and adds no scheduling of its own — the same wavefront scheduler
runs the work.

```python
from anycode import Agent, Crew, TaskSpec

analyst = Agent(name="analyst", role="a market analyst", goal="state the plain facts", tools=[])
writer = Agent(name="writer", role="a newsletter writer", goal="turn figures into a note", tools=[])

crew = Crew(
    agents=[analyst, writer],
    tasks=[
        TaskSpec("Collect", "Look up the closing prices for ACME and GLOBEX.", agent=analyst),
        TaskSpec(
            "Write",
            "Write a three-sentence market note.",
            agent=writer,
            depends_on=["Collect"],
            expected_output="Exactly three sentences, no bullet points.",
        ),
    ],
    verbose=True,
)

result = crew.run_sync()
print(result.output)
```

## Declaring tasks

`tasks=` accepts three shapes:

```python
tasks=["summarize the changelog"]                            # a bare title
tasks=[{"title": "T", "description": "D", "expected_output": "E"}]
tasks=[TaskSpec("T", "D", agent=writer, depends_on=["Other"])]
```

A task with no assignee goes to the first agent in the crew. Assigning a task to someone
outside the crew raises `CrewError` listing the members.

`expected_output` is appended to the prompt as an explicit `Expected output:` line. It is
the cheapest way to make a result usable by the next task.

## Sequential or dependency-driven

```python
Crew(agents=[...], tasks=[...], process="dependency")   # default: declared edges only
Crew(agents=[...], tasks=[...], process="sequential")   # each task waits for the previous one
```

Under `sequential`, a task that already declares `depends_on` keeps its own edge — the
chaining only fills in the blanks. Under `dependency`, independent tasks run concurrently.

## Letting the crew plan

Leave `tasks` out and pass a goal to `run()`. The first agent decomposes the goal, assigns
work to the roster, and the crew executes the resulting plan:

```python
import asyncio
from anycode import Agent, Crew

async def main() -> None:
    async with Crew(agents=[analyst, writer], name="research-desk") as crew:
        result = await crew.run("Report which ticker closed highest and why it matters.")
        print(result.output)

asyncio.run(main())
```

## Reading the result

`CrewResult` gives you the ergonomic accessors and keeps the full engine result attached:

| Field | Meaning |
|---|---|
| `success` | every task succeeded |
| `output` | the final task's output — `str(result)` returns this |
| `outputs` | agent name → output |
| `usage` | combined `TokenUsage` |
| `cost` | `CostReport` when cost tracking is on |
| `team_result` | the complete `TeamRunResult` — handoffs, route decisions, lifecycle events, gate decisions |

## Engine features

Any `OrchestratorConfig` field can be passed straight through:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    max_concurrency=4,
    cost={"enabled": True, "budget_usd": 1.0},
    checkpoint={"enabled": True, "path": ".anycode/checkpoints"},
    routing={"enabled": True},
)
```

An unknown option raises `CrewError` listing the valid ones rather than a Pydantic
traceback. To reuse an engine you already configured — with MCP servers connected or
plugins installed — pass `engine=`. A crew never closes an engine it did not create.

## Lifecycle

```python
async with Crew(agents=[...], tasks=[...]) as crew:
    result = await crew.run()
```

The context manager connects MCP servers on entry and releases engine resources on exit.
`run_sync()` does both around a single blocking call.

## Crew or workflow?

Use a **crew** when the shape of the work is a dependency graph known up front. Use a
[**workflow**](workflows.md) when you need branching, looping, or retry — control flow the
dependency graph cannot express. A crew can be a node inside a workflow.

## See also

- [Workflows](workflows.md)
- [Multi-agent teams](multi-agent-team.md) — the underlying `AnyCode` + `Team` API
- [Recipes](../reference/recipes.md)
