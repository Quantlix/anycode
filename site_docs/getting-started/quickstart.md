---
title: "AnyCode Quickstart — Your First Agent, Crew, and Workflow"
description: Build an AnyCode agent with a custom tool in ten lines, then a crew with dependent tasks, then a workflow graph with a review loop, using automatic provider detection.
keywords: anycode quickstart, python ai agent, tool decorator, crew tasks, workflow graph, run_sync, provider auto detection, first agent, multi agent python
---

# Quickstart

Three levels, each building on the last. Every snippet is complete — copy it into a file
and run it.

!!! note "Before you start"
    Python 3.12+, AnyCode installed, and one provider key in your environment or `.env`
    (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `OLLAMA_BASE_URL`).
    See [Installation](installation.md).

---

## Level 1 — one agent, one tool

```python
# quickstart_agent.py
from dotenv import load_dotenv

from anycode import Agent, tool

load_dotenv()


@tool
def word_count(text: str) -> int:
    """Count the words in a block of text."""
    return len(text.split())


editor = Agent(
    name="editor",
    instructions="You are a concise copy editor. Use your tools rather than guessing.",
    tools=[word_count],
)

result = editor.run_sync("How many words are in: 'the quick brown fox jumps over it'?")
print(result.output)
print(f"{result.token_usage.input_tokens} in / {result.token_usage.output_tokens} out")
```

```bash
uv run python quickstart_agent.py
```

Three things happened without you asking:

- **The provider was detected** from whichever API key is set, along with a sensible default
  model. Pass `provider=` and `model=` to override.
- **The tool schema was derived** from the signature, and its description from the
  docstring. No input model, no registry, no executor.
- **`run_sync` handled the event loop.** Use `await agent.run(...)` inside async code.

Streaming and conversation are one method away:

```python
for event in editor.stream_sync("Explain closures in one sentence."):
    if event.type == "text":
        print(event.data, end="", flush=True)

editor.prompt_sync("What is a generator?")
editor.prompt_sync("Show one example.")  # remembers the first turn
```

---

## Level 2 — a crew with dependent tasks

When one agent is not enough, give each a role and let the work flow between them.

```python
# quickstart_crew.py
from dotenv import load_dotenv

from anycode import Agent, Crew, TaskSpec

load_dotenv()

researcher = Agent(
    name="researcher",
    role="a research analyst",
    goal="gather concrete, verifiable facts",
    tools=[],
)
writer = Agent(
    name="writer",
    role="a technical writer",
    goal="turn findings into prose a busy reader finishes",
    backstory="You write for engineers who want the point in the first sentence.",
    tools=[],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[
        TaskSpec("Research", "List three concrete tradeoffs of vector databases.", agent=researcher),
        TaskSpec(
            "Write",
            "Turn those tradeoffs into a short briefing.",
            agent=writer,
            depends_on=["Research"],
            expected_output="Under 120 words, no bullet points.",
        ),
    ],
    verbose=True,
)

result = crew.run_sync()
print(result)  # the final task's output
print(result.usage.output_tokens)
```

`depends_on` names the earlier task by title; its output is fed into the later prompt.
Independent tasks run concurrently. Set `process="sequential"` to chain everything in
declaration order, or drop `tasks` entirely and call `crew.run("some goal")` to have the
first agent plan the work itself.

---

## Level 3 — a workflow with a loop

A crew fans out over dependencies. When you need branching, looping, or retry, use a
workflow graph.

```python
# quickstart_workflow.py
import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from anycode import END, START, Agent, Workflow

load_dotenv()


class ReviewState(BaseModel):
    model_config = ConfigDict(frozen=True)

    topic: str = ""
    draft: str = ""
    critique: str = ""
    rounds: int = 0


writer = Agent(name="writer", instructions="Write a two-sentence technical blurb.", tools=[])
critic = Agent(
    name="critic",
    instructions="Reply with exactly APPROVED, or one sentence naming the biggest problem.",
    tools=[],
)

workflow = Workflow(ReviewState)


@workflow.node
async def write(state: ReviewState) -> dict:
    prior = f"\n\nFix this: {state.critique}" if state.critique else ""
    result = await writer.run(f"Write about {state.topic}.{prior}")
    return {"draft": result.output, "rounds": state.rounds + 1}


@workflow.node
async def review(state: ReviewState) -> dict:
    result = await critic.run(state.draft)
    return {"critique": result.output}


def gate(state: ReviewState) -> str:
    approved = "APPROVED" in state.critique.upper()
    return END if approved or state.rounds >= 3 else "write"


workflow.add_edge(START, "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edge("review", gate)


async def main() -> None:
    app = workflow.compile()
    print(app.to_mermaid())
    result = await app.run(ReviewState(topic="hybrid search"), max_steps=8)
    print(result.state.draft)
    print("path:", " -> ".join(result.path))


asyncio.run(main())
```

Nodes return a **patch** — the fields they changed — and never mutate state. `compile()`
validates the graph before anything runs and reports every structural problem at once.
`max_steps` turns a runaway loop into an inspectable result instead of a hang.

---

## Where to go next

| You want | Read |
|---|---|
| More on tools | [Function tools](../guides/function-tools.md) |
| More on teams | [Crews](../guides/crews.md) |
| More on graphs | [Workflows](../guides/workflows.md) |
| Long-running research agents | [Long-horizon agents](../guides/long-horizon-agents.md) |
| Copy-paste snippets | [Recipes](../reference/recipes.md) |
| Durability, gates, sandboxes | [How-to guides](../guides/index.md) |

Everything above is a facade over the `AnyCode` engine, which stays public and unchanged.
When you need durable runs, MCP servers, plugins, routing, or verification gates, either
pass those options through `Crew(...)` or use
[`AnyCode`](../guides/multi-agent-team.md) directly.
