---
title: "Using AnyCode From an AI Coding Agent"
description: A compact orientation for AI coding agents writing AnyCode: the core surface, which abstraction to pick, the commands that answer questions, common mistakes.
keywords: anycode for ai agents, llm coding agent guide, anycode api command, core surface, agent crew workflow choice, token efficient docs, common mistakes
---

# Using AnyCode from an AI coding agent

This page is written for an AI coding agent that has been asked to build something with
AnyCode. It is deliberately short.

## Read these two things first

```bash
anycode api --core        # 15 symbols, ~4.5 KB — the front door
```

plus [Recipes](recipes.md) — complete runnable snippets for the tasks you are usually asked
for. Those two cover most work. Do not read the source tree to learn the API.

For one symbol in full, including the real signature:

```bash
anycode api Agent
anycode api --json --compact       # every symbol, no signatures
```

From Python: `anycode.describe("Crew")` returns the same data as an `ApiEntry`.

## Which abstraction

| The task | Use |
|---|---|
| One agent answers, possibly with tools | `Agent` |
| Several agents, work shaped as a dependency graph | `Crew` |
| Branching, looping, retry, or human checkpoints | `Workflow` |
| Long-running research or authoring | `Agent(planning=True, subagents=[...], workspace=...)` |
| Durability, MCP, plugins, verification gates, routing | `AnyCode` directly, or pass the options through `Crew` |

Start at the top of that table and move down only when the task requires it.

## The shortest correct program

```python
from anycode import Agent, tool

@tool
def word_count(text: str) -> int:
    """Count the words in a block of text."""
    return len(text.split())

agent = Agent(name="editor", instructions="You are a copy editor.", tools=[word_count])
print(agent.run_sync("Count the words in 'the quick brown fox'.").output)
```

Provider and model come from whichever API key is in the environment. Everything else is
optional.

## Rules that will save you a wrong turn

- **Do not build a `ToolRegistry` or `ToolExecutor` by hand.** `Agent(tools=[...])` does it.
  They stay public for advanced use, not for getting started.
- **Do not write a Pydantic input model for a tool.** `@tool` derives it from the signature.
  Use `define_tool` only for a schema a signature cannot express.
- **Do not copy a provider-detection block into a script.** `Agent(name=...)` detects it.
- **`tools=None` means every built-in tool; `tools=[]` means none.** They are different.
- **Every model is frozen.** Use `model_copy(update={...})`; never assign to a field.
- **Everything is async underneath.** `run_sync`, `prompt_sync`, `stream_sync`, and
  `run_sync` on `Crew`/`CompiledWorkflow` exist for scripts. They raise a clear
  `RuntimeError` if an event loop is already running — use the async form there.
- **Workflow nodes return a patch, not a new state.** Returning an undeclared field raises.
- **A `Crew` with declared tasks takes no goal**, and one without tasks requires one.

## Writing tests

Use `FakeAdapter` rather than real API calls:

```python
from anycode.providers.fake import FakeAdapter, FakeResponse

async def _create(*args, **kwargs):
    return FakeAdapter(responses=[FakeResponse(text="canned")])

monkeypatch.setattr("anycode.core.agent.create_adapter", _create)
```

To exercise a tool with no LLM at all: `await agent.call_tool("name", **arguments)`.

## Errors are meant to be read

AnyCode errors name the offending value and state the fix. If one does not, that is a bug
worth reporting. Common ones:

| Error | Cause |
|---|---|
| `AgentConfigError: ... needs a name` | keyword construction without `name=` |
| `AgentConfigError: no LLM provider could be detected` | no API key in the environment |
| `ToolDefinitionError: ... has no description` | a tool function with no docstring |
| `ToolDefinitionError: Unknown built-in tool "x"` | typo in a `tools=[...]` string |
| `CrewError: ... unknown option(s)` | a misspelled orchestrator setting |
| `WorkflowError: ... cannot be compiled` | a structural graph problem; every issue is listed |
| `AttributeError: ... Did you mean: Agent?` | a misspelled import from `anycode` |

## Working inside this repository

Read `AGENTS.md` at the repo root for the build, test, and style rules. The short version:
`uv` never `pip`, Python 3.12+, frozen Pydantic models, and the landing gate is
`uv run ruff check src/ && uv run ruff format --check src/ && uv run pyright && uv run pytest`.
