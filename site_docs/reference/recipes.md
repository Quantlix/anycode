---
title: "AnyCode Recipes — Copy-Paste Snippets for Common Agent Tasks"
description: Short runnable AnyCode snippets for tools, agents, crews, workflows, streaming, structured output, memory, MCP, cost caps, durability, and verification gates.
keywords: anycode recipes, python agent examples, tool decorator, crew example, workflow graph, streaming agent, structured output, mcp server, cost cap, checkpoint resume
---

# Recipes

Every snippet is complete and runnable. Copy one, change the strings, run it.

If you are an AI coding agent working with AnyCode, this page plus `anycode api --core`
is usually all the context you need.

---

## One agent, one tool

```python
from anycode import Agent, tool

@tool
def word_count(text: str) -> int:
    """Count the words in a block of text."""
    return len(text.split())

agent = Agent(
    name="editor",
    instructions="You are a concise copy editor.",
    tools=[word_count],
)

result = agent.run_sync("How many words are in: 'the quick brown fox jumps'?")
print(result.output)
```

Provider and model are detected from whichever API key is present. Pass
`provider="openai", model="gpt-4o-mini"` to be explicit.

---

## Async agent

```python
import asyncio
from anycode import Agent

async def main() -> None:
    agent = Agent(name="helper", instructions="Answer in one sentence.", tools=[])
    result = await agent.run("What is a closure?")
    print(result.output)

asyncio.run(main())
```

---

## Streaming

```python
import asyncio
from anycode import Agent

async def main() -> None:
    agent = Agent(name="narrator", instructions="Explain briefly.", tools=[])
    async for event in agent.stream("Explain database indexes."):
        if event.type == "text":
            print(event.data, end="", flush=True)

asyncio.run(main())
```

Blocking equivalent: `for event in agent.stream_sync("..."): ...`

---

## Multi-turn conversation

```python
from anycode import Agent

tutor = Agent(name="tutor", instructions="You teach Python. Be brief.", tools=[])
print(tutor.prompt_sync("What is a generator?").output)
print(tutor.prompt_sync("Show me one example.").output)
print(len(tutor.get_history()), "messages so far")
```

---

## Structured output

```python
import asyncio
from pydantic import BaseModel
from anycode import Agent

class Review(BaseModel):
    sentiment: str
    score: int
    summary: str

async def main() -> None:
    agent = Agent(name="analyst", instructions="Analyze product reviews.", tools=[])
    result = await agent.run_structured("Review: 'Fast shipping, poor packaging.'", Review)
    print(result.parsed.sentiment, result.parsed.score)

asyncio.run(main())
```

---

## Call a tool without the LLM

```python
from anycode import Agent, tool

@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right

agent = Agent(name="calc", tools=[add])
print(agent.call_tool_sync("add", left=2, right=3).data)   # "5"
```

Useful for testing a tool under the same validation and security policy the agent applies.

---

## A crew with dependent tasks

```python
from anycode import Agent, Crew, TaskSpec

researcher = Agent(name="researcher", role="a research analyst", goal="gather the facts", tools=[])
writer = Agent(name="writer", role="a technical writer", goal="turn facts into prose", tools=[])

crew = Crew(
    agents=[researcher, writer],
    tasks=[
        TaskSpec("Research", "Find three facts about vector databases.", agent=researcher),
        TaskSpec(
            "Write",
            "Write a short paragraph from those facts.",
            agent=writer,
            depends_on=["Research"],
            expected_output="One paragraph, no bullet points.",
        ),
    ],
)

print(crew.run_sync().output)
```

---

## A crew that plans its own work

```python
import asyncio
from anycode import Agent, Crew

async def main() -> None:
    async with Crew(agents=[
        Agent(name="lead", role="a project lead", tools=[]),
        Agent(name="helper", role="a research assistant", tools=[]),
    ]) as crew:
        result = await crew.run("Produce a competitive brief on managed vector databases.")
        print(result.output)

asyncio.run(main())
```

---

## A workflow with a loop

```python
import asyncio
from pydantic import BaseModel, ConfigDict
from anycode import END, START, Workflow

class State(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: int = 0

workflow = Workflow(State)
workflow.add_node("increment", lambda state: {"value": state.value + 1})
workflow.add_edge(START, "increment")
workflow.add_conditional_edge("increment", lambda state: END if state.value >= 3 else "increment")

async def main() -> None:
    result = await workflow.compile().run(State())
    print(result.state.value, result.path)

asyncio.run(main())
```

---

## An agent as a workflow node

```python
from pydantic import BaseModel, ConfigDict
from anycode import START, Agent, Workflow

class State(BaseModel):
    model_config = ConfigDict(frozen=True)
    question: str = ""
    answer: str = ""

workflow = Workflow(State)
workflow.add_node("ask", Agent(name="qa", tools=[]), input_key="question", output_key="answer")
workflow.add_edge(START, "ask")

result = workflow.compile().run_sync(State(question="What is a B-tree?"))
print(result.state.answer)
```

---

## Draw the graph

```python
print(workflow.compile().to_mermaid())
```

---

## A long-horizon agent

```python
from anycode import Agent, SubAgentSpec

agent = Agent(
    name="researcher",
    instructions="Research the topic and write a report.",
    tools=["file_write", "file_read"],
    planning=True,
    subagents=[SubAgentSpec(name="critic", instructions="Critique a draft. Be specific.")],
    workspace="./.anycode/workspace",
    max_turns=40,
)

result = agent.run_sync("Write a report on hybrid search tradeoffs.")
print(agent.todos)
```

There is no separate deep-agent class — these are keywords on the same `Agent`.

---

## Cap spending

```python
from anycode import Agent, Crew

crew = Crew(
    agents=[Agent(name="worker", tools=[])],
    tasks=["summarize the changelog"],
    cost={"enabled": True, "budget_usd": 0.50, "on_budget_exceeded": "stop"},
)
result = crew.run_sync()
print(result.cost)
```

---

## Connect an MCP server

```python
import asyncio
from anycode import AnyCode

async def main() -> None:
    async with AnyCode({"mcp_servers": [{"name": "fs", "command": "npx",
                                         "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}]}) as engine:
        result = await engine.run_agent(
            {"name": "worker", "model": "gpt-4o-mini", "provider": "openai", "mcp_servers": ["fs"]},
            "List the files in the current directory.",
        )
        print(result.output)

asyncio.run(main())
```

---

## Durable, resumable runs

```python
from anycode import Agent, Crew, TaskSpec

crew = Crew(
    agents=[Agent(name="worker", tools=[])],
    tasks=[TaskSpec("Step one", "Do the first thing."), TaskSpec("Step two", "Do the second.")],
    checkpoint={"enabled": True, "path": ".anycode/checkpoints"},
)
result = crew.run_sync()
```

Re-running the same task list resumes from the last completed wave.

---

## Verification gates

```python
from anycode import Agent

agent = Agent(
    name="coder",
    instructions="Write Python and keep it lint-clean.",
    tools=["file_write", "bash"],
    verification=({"kind": "ruff", "phase": "after_turn", "severity": "error"},),
)
```

A failing sensor sets `result.stop_reason.code == "verification_failed"`.

---

## Persistent memory

```python
from anycode import create_memory_store

store = create_memory_store({"backend": "sqlite", "path": ".anycode/memory.db"})
```

Backends: `memory`, `sqlite` (`persistence` extra), `redis` (`redis` extra).

---

## Swap the provider

```python
from anycode import Agent

anthropic = Agent(name="a", provider="anthropic", model="claude-haiku-4-5", tools=[])
openai_ = Agent(name="b", provider="openai", model="gpt-4o-mini", tools=[])
local = Agent(name="c", provider="ollama", model="qwen3:8b", tools=[])
```

Or set `ANYCODE_DEFAULT_PROVIDER` and `ANYCODE_DEFAULT_MODEL` and pass neither.

---

## Test without an API call

```python
from anycode import Agent
from anycode.providers.fake import FakeAdapter, FakeResponse

async def fake_adapter(*args, **kwargs):
    return FakeAdapter(responses=[FakeResponse(text="canned answer")])

# in a test: monkeypatch.setattr("anycode.core.agent.create_adapter", fake_adapter)
```

---

## Discover the API

```bash
anycode api --core       # the 15 symbols that cover most use
anycode api Agent        # one symbol, full signature
anycode api --json       # the whole surface, machine-readable
```

```python
import anycode
anycode.describe("Crew").signature
```
