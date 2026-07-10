---
title: "Stream Agent Output in AnyCode"
description: "Consume incremental text, thinking, and tool events from an AnyCode agent run with AgentRunner.stream or Agent.stream, with automatic fallback to non-streaming chat."
keywords: AnyCode streaming, StreamEvent, AgentRunner stream, RunnerStreamingConfig, incremental output, token streaming, fallback_to_chat, parallel tool execution
---

# Stream Agent Output

AnyCode streams by default. Every agent run is an async iterator of `StreamEvent` objects — incremental `text` and `thinking` chunks, `tool_use` and `tool_result` events as tools execute, and a terminal `done` event carrying the final `RunResult`. You consume it with `async for`; there is no callback API for tokens.

Streaming is transparent to correctness: the final `RunResult` (output, token usage, stop reason) is identical whether streaming is on or off, and tests assert this parity.

## Consume a stream

The high-level entry point is `Agent.stream(prompt)`. The lower-level runner equivalent is `AgentRunner.stream(seed_messages)`:

```python title="stream_events.py"
from anycode import (
    AgentRunner, LLMMessage, RunnerOptions,
    RunnerStreamingConfig, TextBlock, ToolExecutor, ToolRegistry,
)
from anycode.providers.fake import FakeAdapter, FakeResponse

adapter = FakeAdapter(
    responses=[FakeResponse(text="Streaming lets the runner surface tokens.", text_chunks=6)],
    model_name="fake-stream",
)
registry = ToolRegistry()
options = RunnerOptions(
    model="fake-stream",
    agent_name="streamer",
    max_turns=2,
    streaming=RunnerStreamingConfig(enabled=True),  # this is already the default
)
runner = AgentRunner(adapter, registry, ToolExecutor(registry), options)

seed = [LLMMessage(role="user", content=[TextBlock(text="Explain streaming.")])]

final = None
async for event in runner.stream(seed):
    if event.type == "text" and isinstance(event.data, str):
        print(event.data, end="", flush=True)   # incremental chunk
    elif event.type == "done":
        final = event.data                       # RunResult
```

`AgentRunner.run(...)` is a convenience wrapper that drains the same stream internally and returns the `RunResult` from the `done` event.

## Event types

`StreamEvent` has two fields: `type` and `data`. The payload depends on the type:

| `event.type` | `event.data` | When |
| --- | --- | --- |
| `text` | `str` | Incremental chunk while streaming; one consolidated string on the chat fallback path |
| `thinking` | `str` | Incremental reasoning text from [reasoning models](reasoning-models.md) |
| `tool_use` | `ToolUseBlock` | One per tool call, emitted before execution |
| `tool_result` | `ToolCallRecord` | One per completed tool call |
| `handoff` | `HandoffRequest` | Just before a handoff-triggered `done` |
| `done` | `RunResult` | Terminal; always the last event on success or cancellation |
| `error` | `Exception` | Terminal failure path (the raw exception object) |

Ordering guarantees, asserted in the test suite:

- Incremental `text` events always precede `done`; a chunked reply produces multiple `text` events.
- Tools execute exactly once under streaming — no double dispatch.
- A cancelled run (`asyncio.CancelledError`) still yields one final `done` event with `stop_reason.code == "user_cancelled"` before the cancellation propagates.

## Configure streaming

Streaming is controlled by `RunnerStreamingConfig` on `RunnerOptions.streaming`:

| Field | Default | Effect |
| --- | --- | --- |
| `enabled` | `True` | Stream tokens from the provider; `False` uses one `chat()` call and emits a single consolidated `text` event |
| `fallback_to_chat` | `True` | If the stream fails **before any output was emitted**, retry once via non-streaming `chat()` |

When `RunnerOptions.streaming` is `None`, the runner substitutes `RunnerStreamingConfig()` — so streaming is on unless you turn it off.

!!! warning "Fallback stops once output has been emitted"
    The chat fallback only fires when the stream fails before producing anything. Once any text or tool event has been emitted, a mid-stream failure propagates instead — this guards against duplicated output and repeated tool side effects. `ProviderUnavailableError` is never retried via chat either; by the time it surfaces, the resilience layer has already exhausted its retries.

## Parallel tool execution

When a model requests several tool calls in one turn, AnyCode executes them concurrently with `asyncio.gather`, bounded by a semaphore of `DEFAULT_TOOL_CONCURRENCY = 4`. Two properties are worth relying on:

- **Results keep their order.** Execution is parallel, but `tool_result` events are emitted in the original block order, aligned with their `tool_use` ids.
- **One failure doesn't abort the batch.** A tool that raises is converted to an error `ToolResult`; the other tools in the turn complete normally.

Tools blocked by a budget or guardrail policy short-circuit to an error result without executing.

## See also

- [Use reasoning models](reasoning-models.md) — `thinking` events and extended-thinking budgets
- [Production controls](production-controls.md) — resilience, budgets, and durability around the stream
