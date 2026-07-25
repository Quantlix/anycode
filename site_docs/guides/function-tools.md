---
title: "Turn Python Functions Into Agent Tools With @tool"
description: Use the AnyCode @tool decorator to expose any Python function as an agent tool, with the signature becoming the schema and the docstring the description.
keywords: anycode tool decorator, python function as llm tool, tool schema from type hints, google docstring tool description, tool context injection, async tool
---

# Work with function tools

The `@tool` decorator turns any Python function into a tool an agent can call. The
signature becomes the input schema, the docstring becomes the description the model reads,
and the return value is converted into a `ToolResult`.

```python
from anycode import Agent, tool

@tool
def convert_currency(amount: float, rate: float, target: str = "EUR") -> dict:
    """Convert an amount using a fixed exchange rate.

    Args:
        amount: The source amount to convert.
        rate: Units of the target currency per unit of the source currency.
        target: ISO code of the target currency.
    """
    return {"converted": round(amount * rate, 2), "currency": target}

agent = Agent(name="analyst", instructions="Use tools for every calculation.", tools=[convert_currency])
print(agent.run_sync("Convert 250 at a rate of 0.92.").output)
```

That is the whole setup. There is no registry to create, no input model to declare, and no
executor to wire.

## What the decorator derives

| From the function | Becomes |
|---|---|
| `__name__` | the tool name |
| Docstring summary | the tool description |
| `Args:` entries | per-parameter descriptions in the schema |
| Type hints | the JSON Schema types |
| Defaults | optional parameters |
| Parameters with no default | required parameters |

Google-style docstrings are parsed for the `Args:` block. An unparseable docstring degrades
to summary-only rather than raising — but a function with **no** description at all raises
`ToolDefinitionError` at decoration time, because a tool the model cannot understand is
worse than no tool.

## Configuring the tool

```python
@tool(name="business_days", description="Count business days since a date.", side_effecting=False)
async def days_since(iso_date: str) -> int:
    ...
```

`side_effecting=True` opts the tool into the idempotency claim store, so a retried call is
replayed rather than repeated.

## Async and sync

Both work. An async function is awaited directly; a synchronous function runs on a worker
thread, so a blocking call never stalls the event loop.

## Constraints and rich types

`Annotated` metadata is preserved, so Pydantic constraints reach the schema and reject bad
input before your function runs:

```python
from typing import Annotated
from pydantic import Field

@tool
def scaled(value: Annotated[int, Field(ge=0, le=10, description="Clamped value.")]) -> int:
    """Double a bounded value."""
    return value * 2
```

## Reading the calling context

Declare a parameter annotated `ToolUseContext` — or conventionally named `ctx`, `context`,
or `tool_context` — and it is injected at call time and excluded from the schema:

```python
from anycode import ToolUseContext, tool

@tool
def whoami(ctx: ToolUseContext) -> str:
    """Report the calling agent."""
    return f"{ctx.agent.name} running {ctx.agent.model}"
```

## Return values

| You return | The agent sees |
|---|---|
| `str` | the string |
| `dict`, `list`, `int`, `float`, `bool` | JSON |
| a Pydantic model | its JSON |
| `None` | an empty result |
| `ToolResult` | passed through, flags intact |

Raising an exception produces an error result with a redacted message — you do not need to
catch and wrap.

## The function stays a function

The decorator returns your function unchanged, so it is still directly callable:

```python
assert convert_currency(100.0, 0.92) == {"converted": 92.0, "currency": "EUR"}
```

Unit-test tools as ordinary functions. To exercise one through the full validation and
security path without an LLM, use `agent.call_tool_sync("convert_currency", amount=100, rate=0.92)`.

## Mixing tool sources

`tools=` accepts decorated functions, plain functions, `ToolDefinition` objects, and the
names of bundled tools:

```python
agent = Agent(name="mixed", tools=[convert_currency, "bash", "file_read", some_definition])
```

- `tools=None` (the default) allows every built-in tool.
- `tools=[]` means no tools at all.
- A duplicate name raises `ToolDefinitionError` naming the collision.

Run `anycode inspect tools` to list the bundled names.

## When to use `define_tool` instead

`@tool` covers most cases. Reach for [`define_tool`](tools.md) when you need an input model
that cannot be expressed as a signature — a deeply nested schema, a discriminated union, or
a model shared with the rest of your application.

## See also

- [Recipes](../reference/recipes.md) — copy-paste snippets
- [Built-in tools](../reference/built-in-tools.md)
- [Tools reference](tools.md) — the typed `define_tool` path
