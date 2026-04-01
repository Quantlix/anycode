<p align="center">
  <img src="https://img.shields.io/pypi/v/anycode?style=flat-square&color=0078D4" alt="PyPI version" />
  <img src="https://img.shields.io/pypi/l/anycode?style=flat-square" alt="license" />
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Built%20by-Quantlix-blueviolet?style=flat-square" alt="Built by Quantlix" />
</p>

# AnyCode

### Scalable Multi-Agent AI Orchestration Framework for Python

> Developed and maintained by **[Quantlix](https://github.com/Quantlix)**

AnyCode is a lightweight yet powerful orchestration engine written entirely in Python. It enables you to compose autonomous AI agents into collaborative teams that communicate, share context, resolve task dependencies, and operate concurrently — all from a single runtime. Whether you're deploying on bare metal, inside containers, across serverless functions, or within CI/CD pipelines, AnyCode adapts to your infrastructure without friction.

Instead of managing individual agents in silos, AnyCode introduces a team-oriented paradigm: agents exchange messages through a built-in event bus, persist shared knowledge in memory stores, and execute work items according to a topologically sorted task graph. The result is a cohesive system where every agent understands its role and collaborates toward a unified objective.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Quick Start](#quick-start)
- [Building Agent Teams](#building-agent-teams)
- [Defining Task Pipelines](#defining-task-pipelines)
- [Creating Custom Tools](#creating-custom-tools)
- [Cross-Provider Model Mixing](#cross-provider-model-mixing)
- [Live Streaming Output](#live-streaming-output)
- [Architecture Overview](#architecture-overview)
- [Built-In Tool Reference](#built-in-tool-reference)
- [Core Concepts at a Glance](#core-concepts-at-a-glance)
- [Contributing](#contributing)
- [License](#license)

---

## Key Capabilities

Traditional agent libraries focus on running a single LLM in a loop. AnyCode takes a fundamentally different approach — it gives you **an entire coordinated team**:

| Feature | Description |
|---------|-------------|
| **Inter-agent communication** | Agents relay information through `MessageBus`, share persistent state via `SharedMemory`, and synchronize through managed task queues |
| **Dependency-driven execution** | Express task relationships with `depends_on` and let `TaskQueue` resolve ordering through topological sorting — no manual sequencing needed |
| **Automatic goal decomposition** | Provide a high-level objective and the orchestrator intelligently partitions it into targeted subtasks assigned to the right agents |
| **Provider-agnostic design** | Seamlessly use Anthropic Claude, OpenAI GPT, or integrate any custom backend through the `LLMAdapter` protocol |
| **Schema-validated tooling** | Every tool is declared with a Pydantic model for input validation, plus five practical tools are included out of the box |
| **Bounded parallelism** | Independent work items execute simultaneously, governed by a configurable concurrency semaphore |
| **Flexible scheduling strategies** | Choose between round-robin, least-busy, capability-match, or dependency-first assignment policies |
| **Incremental streaming** | Receive real-time text deltas from any agent as an `AsyncGenerator[StreamEvent, None]` |
| **Full type safety** | Strict Pydantic models enforced at every layer, with validation at all external boundaries |

---

## Quick Start

Install the package from PyPI:

```bash
pip install anycode
# or with uv
uv add anycode
```

### Single Agent Execution

The simplest way to get started — spin up one agent and hand it a task:

```python
import asyncio
from anycode import AnyCode

async def main():
    engine = AnyCode()

    result = await engine.run_agent(
        config={
            "name": "engineer",
            "model": "claude-sonnet-4-6",
            "tools": ["bash", "file_write"],
        },
        prompt="Create a Python utility that checks whether a given string is a palindrome, save it to /tmp/palindrome.py, and execute it.",
    )

    print(result.output)

asyncio.run(main())
```

> **Note:** Export `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` as environment variables before running any example.

---

## Building Agent Teams

Real-world workflows benefit from specialization. AnyCode lets you define distinct agents — each with its own system prompt, model, and tool access — and unify them into a collaborative team:

```python
import asyncio
from anycode import AnyCode, AgentConfig, TeamConfig

planner = AgentConfig(
    name="planner",
    model="claude-sonnet-4-6",
    system_prompt="You draft module interfaces, folder layouts, and endpoint schemas.",
    tools=["file_write"],
)

builder = AgentConfig(
    name="builder",
    model="claude-sonnet-4-6",
    system_prompt="You translate specifications into production-ready code.",
    tools=["bash", "file_read", "file_write", "file_edit"],
)

auditor = AgentConfig(
    name="auditor",
    model="claude-sonnet-4-6",
    system_prompt="You inspect code for bugs, edge cases, and readability concerns.",
    tools=["file_read", "grep"],
)

async def main():
    engine = AnyCode(config={
        "default_model": "claude-sonnet-4-6",
        "on_progress": lambda ev: print(ev.type, ev.agent or ev.task or ""),
    })

    team = engine.create_team("backend-crew", TeamConfig(
        name="backend-crew",
        agents=[planner, builder, auditor],
        shared_memory=True,
    ))

    result = await engine.run_team(team, "Scaffold a CRUD API for a notes app in /tmp/notes-api/")

    print(f"Completed: {result.success}")
    print(f"Tokens used: {result.total_token_usage.output_tokens}")

asyncio.run(main())
```

---

## Defining Task Pipelines

For workflows that demand precise control over the execution graph, you can manually specify tasks along with their dependencies:

```python
from anycode import TaskSpec

result = await engine.run_tasks(team, [
    TaskSpec(
        title="Draft schema definitions",
        description="Produce Python type declarations and save them to /tmp/types.md",
        assignee="planner",
    ),
    TaskSpec(
        title="Implement core logic",
        description="Read /tmp/types.md and build the service layer in /tmp/lib/",
        assignee="builder",
        depends_on=["Draft schema definitions"],
    ),
    TaskSpec(
        title="Write unit tests",
        description="Author pytest test suites covering all service methods.",
        assignee="builder",
        depends_on=["Implement core logic"],
    ),
    TaskSpec(
        title="Audit implementation",
        description="Examine /tmp/lib/ and generate a detailed review report.",
        assignee="auditor",
        depends_on=["Implement core logic"],
    ),
])
```

The `TaskQueue` resolves the dependency graph using topological sorting. Tasks with no unmet dependencies are dispatched in parallel, while dependent tasks wait until their predecessors complete successfully.

---

## Creating Custom Tools

Extend agent capabilities by registering your own tools. Each tool is defined with a Pydantic model for automatic validation:

```python
from pydantic import BaseModel, Field
from anycode import define_tool, Agent, ToolRegistry, ToolExecutor, register_built_in_tools, ToolResult, ToolUseContext

class ArticleSearchInput(BaseModel):
    topic: str = Field(description="Subject to search for.")
    limit: int = Field(default=5, description="Maximum articles to return.")

async def fetch_articles(params: ArticleSearchInput, ctx: ToolUseContext) -> ToolResult:
    articles = await my_knowledge_base(params.topic, params.limit)
    return ToolResult(data=json.dumps(articles), is_error=False)

fetch_articles_tool = define_tool(
    name="fetch_articles",
    description="Retrieves relevant articles from the knowledge base.",
    input_model=ArticleSearchInput,
    execute=fetch_articles,
)

registry = ToolRegistry()
register_built_in_tools(registry)
registry.register(fetch_articles_tool)

executor = ToolExecutor(registry)
agent = Agent(
    config={"name": "analyst", "model": "claude-sonnet-4-6", "tools": ["fetch_articles"]},
    tool_registry=registry,
    tool_executor=executor,
)

result = await agent.run("Summarize the latest changes in the Python typing module.")
```

---

## Cross-Provider Model Mixing

Combine different LLM providers within a single team. Assign a reasoning-heavy model to your strategist and a fast coding model to your implementer:

```python
thinker = AgentConfig(
    name="thinker",
    model="claude-opus-4-6",
    provider="anthropic",
    system_prompt="You devise architectural blueprints and technical strategies.",
    tools=["file_write"],
)

implementer = AgentConfig(
    name="implementer",
    model="gpt-4o",
    provider="openai",
    system_prompt="You transform plans into functional, tested code.",
    tools=["bash", "file_read", "file_write"],
)

team = engine.create_team("cross-provider", TeamConfig(
    name="cross-provider",
    agents=[thinker, implementer],
    shared_memory=True,
))

await engine.run_team(team, "Create a CLI utility that transforms YAML files into JSON format.")
```

---

## Live Streaming Output

For interactive applications or real-time feedback, stream agent output token-by-token:

```python
import asyncio
import sys
from anycode import Agent, ToolRegistry, ToolExecutor, register_built_in_tools

async def main():
    registry = ToolRegistry()
    register_built_in_tools(registry)
    executor = ToolExecutor(registry)

    narrator = Agent(
        config={"name": "narrator", "model": "claude-sonnet-4-6", "max_turns": 3},
        tool_registry=registry,
        tool_executor=executor,
    )

    async for ev in narrator.stream("Describe the observer pattern in three sentences."):
        if ev.type == "text" and isinstance(ev.data, str):
            sys.stdout.write(ev.data)

asyncio.run(main())
```

---

## Architecture Overview

```
+--------------------------------------------------------------+
|  AnyCode  (orchestrator)                                     |
|                                                              |
|  create_team()  ·  run_team()  ·  run_tasks()  ·  run_agent()|
+-----------------------------+--------------------------------+
                              |
                   +----------v----------+
                   |  Team               |
                   |  AgentConfig[]      |
                   |  MessageBus         |
                   |  TaskQueue          |
                   |  SharedMemory       |
                   +----------+----------+
                              |
                +-------------+-------------+
                |                           |
       +--------v---------+    +------------v-----------+
       |  AgentPool       |    |  TaskQueue             |
       |  Semaphore       |    |  dependency graph      |
       |  run_parallel()  |    |  cascade failure       |
       +--------+---------+    +------------------------+
                |
       +--------v---------+
       |  Agent           |    +------------------------+
       |  run / prompt /  |--->|  LLMAdapter            |
       |  stream          |    |  Anthropic · OpenAI    |
       +--------+---------+    +------------------------+
                |
       +--------v---------+
       |  AgentRunner     |    +------------------------+
       |  conversation    |--->|  ToolRegistry          |
       |  loop + dispatch |    |  define_tool + 5       |
       +------------------+    |  built-in tools        |
                               +------------------------+
```

**Data flow summary:**

1. The **orchestrator** receives a goal or an explicit task list
2. A **Team** manages the agent roster, message bus, and shared memory
3. The **AgentPool** dispatches work using a bounded concurrency semaphore
4. The **TaskQueue** resolves dependencies via topological sort and cascades failures
5. Each **Agent** runs a conversation loop through **AgentRunner**, invoking tools from the **ToolRegistry** as needed
6. LLM calls are routed through the **LLMAdapter** abstraction, supporting any provider

---

## Built-In Tool Reference

AnyCode ships with five practical tools that cover the most common agent operations:

| Tool | What It Does |
|------|-------------|
| `bash` | Executes shell commands with stdout/stderr capture, configurable timeout, and working-directory support |
| `file_read` | Reads file contents from an absolute path, with optional offset and line-limit for handling large files |
| `file_write` | Creates or overwrites a file at the specified path — parent directories are generated automatically |
| `file_edit` | Performs targeted substring replacement within a file, with an option to replace all occurrences |
| `grep` | Runs regex-based searches across files, leveraging ripgrep when available or falling back to a pure Python implementation |

All tools follow the same `define_tool()` pattern, so extending or replacing them works identically to registering custom tools.

---

## Core Concepts at a Glance

| Concept | Component | Responsibility |
|---------|-----------|----------------|
| Conversation loop | `AgentRunner` | Manages the model <-> tool turn cycle until the task completes |
| Typed tool declaration | `define_tool()` | Defines tools with Pydantic-validated input models |
| Orchestration | `AnyCode` | Decomposes goals, assigns work, and manages concurrency |
| Team coordination | `Team` + `MessageBus` | Enables inter-agent messaging and shared knowledge state |
| Task scheduling | `TaskQueue` | Resolves execution order through topological dependency sorting |

---

## Contributing

Contributions, suggestions, and issue reports are welcome. Please open an issue or submit a pull request on the [GitHub repository](https://github.com/Quantlix/anycode).

---

## License

Released under the MIT License — see [LICENSE](./LICENSE) for details.

---

<p align="center">
  Built with purpose by <strong><a href="https://github.com/Quantlix">Quantlix</a></strong>
</p>
