---
title: "Migrating to AnyCode 0.10"
description: Nothing is required to move from AnyCode 0.9 to 0.10. Here is what is new, the one behavioral note worth knowing, and old-to-new snippets for each addition.
keywords: anycode 0.10 migration, upgrade anycode, backward compatible, tool decorator migration, agent keywords, crew workflow, lazy imports, breaking changes
---

# Migrating to 0.10

**Nothing is required.** Every 0.9 public symbol keeps its name, its import path, and its
semantics. Code written against 0.9 runs unchanged on 0.10.

0.10 adds a layer on top: a shorter way to say the same things, plus two new composition
primitives. This page shows the old and new forms side by side so you can adopt what helps
and ignore what does not.

## Defining a tool

=== "0.9 (still supported)"

    ```python
    class WeatherInput(BaseModel):
        city: str = Field(description="City name.")

    async def lookup_weather_fn(params: WeatherInput, _ctx: ToolUseContext) -> ToolResult:
        return ToolResult(data=json.dumps({...}))

    weather_tool = define_tool(
        name="lookup_weather",
        description="Retrieve current weather for a city.",
        input_model=WeatherInput,
        execute=lookup_weather_fn,
    )
    ```

=== "0.10"

    ```python
    @tool
    def lookup_weather(city: str) -> dict:
        """Retrieve current weather for a city.

        Args:
            city: City name.
        """
        return {...}
    ```

Keep `define_tool` for schemas a signature cannot express. See
[Function tools](../guides/function-tools.md).

## Building an agent

=== "0.9 (still supported)"

    ```python
    registry = ToolRegistry()
    register_built_in_tools(registry)
    registry.register(weather_tool)
    executor = ToolExecutor(registry)

    agent = Agent(
        AgentConfig(name="collector", model=MODEL, provider=PROVIDER, tools=["lookup_weather"]),
        registry,
        executor,
    )
    result = await agent.run("...")
    ```

=== "0.10"

    ```python
    agent = Agent(
        name="collector",
        instructions="You collect data.",
        tools=[lookup_weather],
    )
    result = agent.run_sync("...")
    ```

The three-positional-argument constructor is unchanged and is still what
`AnyCode.build_agent` uses. Passing a config object **and** a conflicting field keyword
raises `AgentConfigError` naming the conflict.

## Running a team

=== "0.9 (still supported)"

    ```python
    engine = AnyCode({"max_concurrency": 3})
    team = engine.create_team("crew", TeamConfig(name="crew", agents=[cfg_a, cfg_b]))
    result = await engine.run_tasks(team, [TaskSpec("A", "..."), TaskSpec("B", "...", depends_on=["A"])])
    ```

=== "0.10"

    ```python
    crew = Crew(agents=[agent_a, agent_b], tasks=[
        TaskSpec("A", "..."),
        TaskSpec("B", "...", depends_on=["A"]),
    ])
    result = crew.run_sync()
    ```

`Crew` owns an `AnyCode` engine and a `Team`; the scheduler is the same. Pass an engine you
configured yourself with `engine=`.

## The one behavioral note

`Agent(tools=[])` now means **no tools**. In 0.9 an empty list was indistinguishable from
`None`, which means "every built-in tool". `tools=None` is still the default and still
means every built-in.

If you were passing `tools=[]` to an `AgentConfig` expecting all built-ins, pass
`tools=None` instead.

## Additions worth knowing about

| Addition | Where |
|---|---|
| `@tool` decorator | [Function tools](../guides/function-tools.md) |
| Keyword `Agent(...)`, `role`/`goal`/`backstory`, provider auto-detection | [Quickstart](../getting-started/quickstart.md) |
| `run_sync`, `prompt_sync`, `stream_sync`, `call_tool` | [Recipes](../reference/recipes.md) |
| `Crew` and `CrewResult` | [Crews](../guides/crews.md) |
| `Workflow`, `START`, `END`, `Command`, reducers | [Workflows](../guides/workflows.md) |
| `planning=`, `subagents=`, `workspace=` on `Agent` | [Long-horizon agents](../guides/long-horizon-agents.md) |
| `anycode api` and `anycode.describe()` | [LLM guide](../reference/llm-guide.md) |
| `TaskSpec(agent=..., expected_output=...)` | [Crews](../guides/crews.md) |
| `AnyCode.register_agent(agent)` | adopt a pre-built agent into an engine |

## Import cost

`import anycode` no longer imports every subsystem. It dropped from roughly 1.9 seconds and
1300 modules to about 30 milliseconds and 70 modules; symbols resolve on first access.

Two consequences worth knowing:

- Optional dependencies (chromadb, redis, OpenTelemetry, the MCP SDK, provider clients) are
  imported only when the code path that needs them runs.
- A misspelled import now raises `AttributeError` with a suggestion rather than
  `ImportError`.

If you monkeypatched a symbol on `anycode.core.orchestrator` that it imported from
elsewhere — `MCPClient` and `discover_and_register` are the two — patch the source module
(`anycode.mcp.client`, `anycode.mcp.bridge`) instead.

## Also fixed

- `anycode.types`, `anycode.helpers`, `anycode.contracts`, `anycode.identity`,
  `anycode.core`, and `anycode.memory` are each importable on their own. A latent circular
  import previously made `import anycode.types` fail unless `anycode` was imported first.
- `Agent.call_tool` supplies an idempotency key, so side-effecting tools can be invoked
  directly.
- `FilesystemRunStore.mark_interrupted_runs` treats the staleness cutoff as inclusive, so
  `stale_after_seconds=0` means "every running run is stale" on every platform.
