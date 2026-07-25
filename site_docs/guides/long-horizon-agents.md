---
title: "Long-Horizon Agents — Planning, Sub-Agents, and a Workspace"
description: Switch on planning, sub-agent delegation, and a confined workspace with three keywords on the same AnyCode Agent, with no separate deep-agent class to learn.
keywords: anycode long horizon agent, deep agent python, write_todos planning tool, sub agent delegation, agent workspace, tool security policy, subagentspec
---

# Long-horizon agents

Long-running work needs three things a single-turn agent does not: an explicit plan, a way
to hand isolated sub-tasks to focused helpers, and somewhere to put intermediate artifacts
so they stop consuming context.

In AnyCode these are keywords on the `Agent` you already use. **There is no separate
deep-agent class.** Each is inert unless you switch it on, so a plain agent's prompt and
tool set are untouched.

```python
from anycode import Agent, SubAgentSpec, tool

@tool
def search(query: str) -> str:
    """Search the corpus."""
    ...

agent = Agent(
    name="researcher",
    instructions="Research the topic thoroughly and produce a written report.",
    tools=[search, "file_write", "file_read", "list_files"],
    planning=True,
    subagents=[SubAgentSpec(name="critic", instructions="Critique a draft. Be specific.")],
    workspace="./.anycode/workspace",
    max_turns=40,
)

result = agent.run_sync("Write a report on hybrid search tradeoffs.")
print(agent.todos)
```

## Planning

`planning=True` registers the `write_todos` tool and adds one clause to the system prompt.
The tool returns the rendered checklist as its result, so the plan re-enters the
conversation every time it changes — that feedback loop is what keeps a long run oriented.

```text
[x] Look up the cache spec
[>] Draft the comparison
[ ] Have the reviewer critique it

1/3 complete.
```

Read the plan from your own code with `agent.todos`, a tuple of `TodoItem`. Exactly one
step may be `in_progress`; sending more returns an error result telling the model to fix
it, because a model mistake is not a program fault.

## Sub-agents

`subagents=` registers a `delegate` tool over the specs you provide:

```python
SubAgentSpec(
    name="critic",
    instructions="Critique a draft. Be specific.",
    tools=[],             # defaults to none
    model=None,           # inherits the parent's model
    provider=None,        # inherits the parent's provider
    max_turns=None,       # inherits the parent's limit
)
```

A sub-agent is a full `Agent`, built lazily on first use, inheriting the parent's model,
provider, security policy, and execution context. It runs on a **fresh conversation** —
only the task text and the context the caller chooses to pass cross the boundary. That
isolation is the point: the parent's context stays small while a focused agent does narrow
work.

Sub-agents never receive the `delegate` tool themselves, so delegation depth is one by
construction and a runaway fan-out is impossible. Their token usage is merged into the
parent's `AgentRunResult`, so `result.token_usage` is the true total.

Delegating to a name that does not exist returns an error result listing the valid ones.

## Workspace

`workspace=` creates the directory and confines the file tools to it using the existing
`ToolSecurityPolicy`:

```python
ToolSecurityPolicy(
    workspace_root="/abs/path",
    allowed_path_roots=("/abs/path",),
    allow_shell=False,      # True when "bash" is in tools=
)
```

Artifacts are real files, so a run is inspectable after the fact. A write outside the root
is rejected before it reaches the filesystem. `agent.workspace` holds the resolved path.

If you pass `tool_security=` explicitly, your policy wins and `workspace=` only creates the
directory — AnyCode never silently overrides a security decision you made.

## Prompt clauses

Each capability appends one short clause after your `instructions`, in a fixed order:
planning, then delegation, then workspace. They are module constants
(`PLANNING_CLAUSE`, `DELEGATION_CLAUSE`, `WORKSPACE_CLAUSE` in `anycode.core.agent`) and
are snapshot-tested, so prompt drift shows up as a reviewable diff.

With no capabilities switched on, your instructions reach the model verbatim.

## Combining with the rest

The capabilities compose with everything else on `Agent`:

```python
agent = Agent(
    name="researcher",
    instructions="...",
    tools=[search],
    planning=True,
    subagents=[critic],
    workspace="./work",
    context_policy=ContextPolicy(...),      # context compaction
    verification=(...,),                     # quality gates
    provider_resilience=ProviderResilienceConfig(...),
)
```

A long-horizon agent can also be a node in a [workflow](workflows.md) or a member of a
[crew](crews.md).

## See also

- [Function tools](function-tools.md)
- [Engineer the context window](context-engineering.md)
- `examples/48_long_horizon_agent.py`
