---
title: "Build a Workflow Graph — Branching, Loops, and Explicit State"
description: Compose AnyCode agents into a state graph with nodes, conditional edges, reducers, and a step cap, then stream, render, and run it with full control flow.
keywords: anycode workflow, state graph python, conditional edge, agent loop, reducer, langgraph alternative, workflow streaming, mermaid graph, max steps
---

# Build a workflow

A `Workflow` is an explicit state graph: you register nodes, connect them, and compile.
Where a [crew](crews.md) fans out over task dependencies, a workflow gives you branching,
looping, retry, and human checkpoints — control flow a dependency graph cannot express.

```python
import asyncio
from typing import Annotated
from pydantic import BaseModel, ConfigDict
from anycode import END, START, Agent, Workflow
from anycode.workflow import add

class ReviewState(BaseModel):
    model_config = ConfigDict(frozen=True)
    topic: str = ""
    draft: str = ""
    critique: str = ""
    rounds: int = 0
    log: Annotated[list[str], add] = []

writer = Agent(name="writer", instructions="Write a two-sentence blurb.", tools=[])
critic = Agent(name="critic", instructions="Reply APPROVED, or name the biggest problem.", tools=[])

workflow = Workflow(ReviewState)

@workflow.node
async def write(state: ReviewState) -> dict:
    result = await writer.run(f"Write about {state.topic}. Prior critique: {state.critique}")
    return {"draft": result.output, "rounds": state.rounds + 1, "log": [f"round {state.rounds + 1}"]}

@workflow.node
async def review(state: ReviewState) -> dict:
    result = await critic.run(state.draft)
    return {"critique": result.output}

def gate(state: ReviewState) -> str:
    return END if "APPROVED" in state.critique.upper() or state.rounds >= 3 else "write"

workflow.add_edge(START, "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edge("review", gate)

result = asyncio.run(workflow.compile().run(ReviewState(topic="vector databases")))
print(result.state.draft, result.path)
```

## State

State is a frozen Pydantic model you supply, or a plain `dict` when you call `Workflow()`
with no schema. A node never mutates it — it returns a **patch** describing what changed:

| A node returns | Effect |
|---|---|
| `dict` | those fields are merged |
| a state instance | its fields are merged |
| `None` | no change |
| `Command` | routing override, with an optional patch |

Returning a field the schema does not declare raises `WorkflowError` naming the field and
listing the valid ones — a typo fails loudly instead of vanishing.

### Reducers

By default a patch replaces a field. Annotate a field with a reducer to accumulate instead:

```python
log: Annotated[list[str], add] = []      # appends
counters: Annotated[dict, merge] = {}    # shallow-merges
tokens: Annotated[int, add] = 0          # sums
first_seen: Annotated[str, keep_first] = ""
```

`add`, `merge`, `keep_first`, and `keep_last` ship in `anycode.workflow`. Any
`Callable[[current, incoming], value]` works.

## Nodes

A node may be:

- an `async def` taking state
- a plain `def` taking state — run on a worker thread so it never blocks the loop
- an `Agent` — `add_node("ask", agent, input_key="question", output_key="answer")`
- a `Crew`
- another `CompiledWorkflow`, as a sub-graph

`Agent`, `Crew`, and sub-graph nodes have their token usage accumulated into
`result.usage` automatically. A plain function that calls an agent itself does not — track
those tokens in state with an `add`-annotated field.

Register with `add_node(name, target)`, or use the decorator form `@workflow.node` /
`@workflow.node(name="custom")`.

## Edges

```python
workflow.add_edge(START, "first")                       # entry point
workflow.add_edge("first", "second")                    # unconditional
workflow.add_edge("second", END)                        # terminate
workflow.add_edge("split", "left")                      # repeat a source
workflow.add_edge("split", "right")                     # to fan out concurrently
workflow.add_conditional_edge("check", router)          # router(state) -> node name or END
workflow.add_conditional_edge("check", router, {"ok": END, "retry": "first"})   # via a path map
```

A node with no outgoing edge routes to `END`. A node routes one way or the other — mixing a
static and a conditional edge on the same node raises immediately.

Fan-out targets run concurrently and their patches merge. Two concurrent nodes writing the
same unreduced field raise `WorkflowError` naming both — annotate the field with a reducer,
or have one node own it.

### Dynamic routing from inside a node

```python
from anycode import Command

workflow.add_node("triage", triage_fn, goto=["escalate", "resolve"])

def triage_fn(state):
    return Command(goto="escalate" if state.severity > 3 else "resolve", update={"seen": True})
```

`goto=` on `add_node` declares which nodes a `Command` may jump to. Declaring them keeps
reachability analysis accurate — a node reachable only by `Command` would otherwise look
like an orphan.

## Compiling

`compile()` validates the graph and returns an immutable `CompiledWorkflow`. It reports
every structural problem at once:

- no entry point, or no nodes
- an edge or path-map entry pointing at an unregistered node
- a node unreachable from the entry point
- a cycle that can never reach `END`

Editing the builder afterwards does not affect an already-compiled graph.

## Running

```python
result = await app.run(state, max_steps=25)
result = app.run_sync(state)
```

`WorkflowResult` carries `success`, `state`, `steps`, `path`, `usage`, `stop_reason`, and
`error`. Exceeding `max_steps` returns `success=False` with
`stop_reason.code == "max_steps"` and `recoverable=True` rather than raising — a runaway
loop is a result you can inspect, not a crash.

## Streaming and rendering

```python
async for event in app.stream(state):
    if event.type == "node_start":
        print("->", event.node)
    elif event.type == "route":
        print("  ", event.node, "routes to", ", ".join(event.targets))
    elif event.type == "done":
        final = event.result
```

Event types are `node_start`, `node_end`, `route`, `error`, and `done`. `run()` is built on
`stream()`, so there is exactly one execution path.

`app.to_mermaid()` renders a flowchart and `app.to_dict()` returns a JSON-friendly
description. A conditional edge without a path map is drawn as `?` to every candidate,
since its targets are only known at runtime — supply `path_map=` for a precise diagram.

## See also

- [Crews](crews.md) — declarative dependency fan-out
- [Recipes](../reference/recipes.md)
- `examples/47_workflow_graph.py`
