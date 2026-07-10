---
title: "Build Custom Tools in AnyCode with Pydantic Validation"
description: "Give AnyCode agents built-in tools like bash and grep, or build custom tools with define_tool and Pydantic input models, then register them in a ToolRegistry."
keywords: AnyCode tools, custom tools, define_tool, ToolRegistry, ToolExecutor, Pydantic tool, built-in tools, bash grep file_read, tool schema
---

# Work With Tools

Give an AnyCode agent tools by listing built-in tool names in its `AgentConfig`, or build a custom tool with `define_tool()` and a Pydantic input model and register it in a `ToolRegistry`. Every AnyCode tool is a typed async function with four parts: a name, a description, a Pydantic input model, and an execute function that returns a `ToolResult`.

The Pydantic model does double duty. It validates arguments before your code runs, and it becomes the JSON schema the provider sees when it decides how to call the tool. Clear field descriptions therefore improve both safety and the quality of the model's tool calls.

## Built-in AnyCode tools

AnyCode ships six built-in tools that cover reading, writing, editing, searching, listing, and running commands. Register them once, then allow them per agent by name.

| Tool | What it does |
| --- | --- |
| `bash` | Execute shell commands with a timeout and captured output |
| `file_read` | Read file contents with line and size controls |
| `file_write` | Create or overwrite files and parent directories |
| `file_edit` | Replace targeted text in an existing file |
| `grep` | Search files with regex, using ripgrep when it is available |
| `list_files` | List project files fast, preferring git/ripgrep/fd and respecting ignore rules |

Add the tool names to an agent's `tools` list to permit usage. An agent can only call tools that appear in this allowlist, which is your first line of scoping.

=== "OpenAI"

    ```python title="agent.py"
    from anycode import AgentConfig

    agent = AgentConfig(
        name="repo-reader",
        provider="openai",
        model="gpt-4o-mini",
        tools=["file_read", "grep"],
    )
    ```

=== "Anthropic"

    ```python title="agent.py"
    from anycode import AgentConfig

    agent = AgentConfig(
        name="repo-reader",
        provider="anthropic",
        model="claude-haiku-4-5",
        tools=["file_read", "grep"],
    )
    ```

!!! note "External tools follow the same path"
    Tools discovered from Model Context Protocol (MCP) servers run through the same validation and execution path as built-in tools, so the guidance below applies to them too.

## Anatomy of a tool

Before writing one, it helps to see the four moving parts and where each is used:

- **Name** identifies the tool in an agent's allowlist and in the provider's tool call.
- **Description** tells the model when to reach for the tool. Write it for the model, not for a human reader.
- **Input model** is a Pydantic `BaseModel`. AnyCode turns it into the JSON schema sent to the provider and validates arguments against it before your code runs.
- **Execute function** is an async function that receives the validated params and a `ToolUseContext`, then returns a `ToolResult`.

## Define a custom tool

Use a Pydantic model for the tool input, then pass an async execute function to `define_tool()`. The `Field(description=...)` text flows straight into the schema the provider receives, so make it specific.

```python title="tools/word_count.py"
from pydantic import BaseModel, Field

from anycode import ToolResult, ToolUseContext, define_tool


class WordCountInput(BaseModel):
    text: str = Field(description="Text to count.")


async def count_words(params: WordCountInput, ctx: ToolUseContext) -> ToolResult:
    count = len(params.text.split())
    return ToolResult(data=str(count), is_error=False)


word_count_tool = define_tool(
    name="count_words",
    description="Count words in a text string.",
    input_model=WordCountInput,
    execute=count_words,
)
```

The execute function returns `ToolResult(data=..., is_error=False)` on success. Set `is_error=True` when the tool fails in a way the agent can recover from, so the model sees the error and can adjust instead of crashing the run.

## Register a custom tool

For manual agent assembly, put your tools in a `ToolRegistry`, wrap it in a `ToolExecutor`, and hand both to an `Agent`. Call `register_built_in_tools()` first if the agent should also reach the built-in tools, then register your custom tool.

```python title="assemble.py"
from anycode import Agent, AgentConfig, ToolExecutor, ToolRegistry, register_built_in_tools

registry = ToolRegistry()
register_built_in_tools(registry)
registry.register(word_count_tool)

executor = ToolExecutor(registry)
agent = Agent(
    AgentConfig(
        name="counter",
        provider="openai",
        model="gpt-4o-mini",
        tools=["count_words"],
    ),
    registry,
    executor,
)
```

The `tools` allowlist still governs access: this agent registered the built-in tools but can only call `count_words`, because that is the single name in its config.

## Enforce a workspace security policy

`ToolSecurityPolicy` adds a second, runtime-enforced boundary around every tool call. The executor applies tool allowlists and denylists to built-in, custom, and MCP tools. Filesystem tools resolve paths and symlinks before checking that the target remains inside an allowed root.

```python title="secure_agent.py"
from anycode import AgentConfig, ToolSecurityPolicy

policy = ToolSecurityPolicy(
    allowed_tools=("file_read", "grep"),
    workspace_root="/srv/anycode/workspaces/review-42",
    allow_shell=False,
    inherit_environment=False,
)

agent = AgentConfig(
    name="reviewer",
    provider="anthropic",
    model="claude-haiku-4-5",
    tools=["file_read", "grep"],
    tool_security=policy,
)
```

Important controls:

- `workspace_root` and `allowed_path_roots` constrain `file_read`, `file_write`, `file_edit`, `grep`, `list_files`, and the shell working directory.
- `allow_shell=False` blocks `bash` even when the tool was registered accidentally.
- `allowed_shell_commands` permits only one executable per call and rejects shell control operators such as `&&`, `|`, and `;`.
- `inherit_environment=False` starts child commands without the parent environment. Add only required names to `allowed_environment_variables`.
- `allowed_tools` and `denied_tools` are enforced centrally by `ToolExecutor`, including for custom tools.

!!! warning "Policy enforcement is not an operating-system sandbox"
    The policy constrains AnyCode's built-in execution paths. A custom Python tool still runs inside the host process and can use normal Python APIs. Run untrusted workloads in a container, VM, or another operating-system isolation boundary, and keep credentials outside that boundary.

## Tool design checklist

Well-shaped tools make agents more reliable and easier to audit. Keep each of these in view when you add one:

- Give each tool one clear job. Small, single-purpose tools are easier for the model to choose correctly.
- Write descriptive Pydantic fields so the provider receives useful schema text.
- Return structured JSON strings when a downstream agent needs to parse the result.
- Keep errors explicit with `ToolResult(is_error=True)` when the agent can recover.
- Grant file or shell tools only to agents that genuinely need them.

## Scope tools to the least privilege they need

!!! danger "A tool-enabled agent is a privileged process"
    Tools that read files, write files, or run shell commands turn an agent into an automation process with real reach. Run tool-enabled agents in an isolated workspace, apply `ToolSecurityPolicy`, grant the minimum set of tools, and add human approval gates before irreversible or sensitive actions. See [Production Controls](production-controls.md) for gates, budgets, and checkpoints.

See `examples/04_hybrid_tooling.py` for a complete agent that combines built-in and custom tools.

## Next steps

- [Run a Multi-Agent Team](multi-agent-team.md) shows how to give a builder scoped file tools inside a crew.
- [Use YAML Config](yaml-config.md) declares each agent's tools in a reviewable config file.
- [Production Controls](production-controls.md) wraps tool use in approval and verification gates.
- [Public API](../reference/public-api.md) lists the full signatures for `define_tool`, `ToolRegistry`, and `ToolExecutor`.
