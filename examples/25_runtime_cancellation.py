"""Phase 6 — runtime cancellation and terminal lifecycle telemetry.

Cancels an in-flight ``AgentRunner.stream`` and inspects the final
``RunResult`` to confirm the runner emits a ``cancelled`` terminal phase with
the ``user_cancelled`` stop reason — both new behaviours wired in Phase 6.

Run::

    uv run python examples/25_runtime_cancellation.py
"""

from __future__ import annotations

import asyncio

from anycode import FakeAdapter, FakeResponse
from anycode.core.runner import AgentRunner
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import LLMMessage, RunnerOptions, TextBlock


async def main() -> None:
    adapter = FakeAdapter(responses=[FakeResponse(text="hi")])
    original_chat = adapter.chat

    async def slow_chat(messages, options):  # type: ignore[no-untyped-def]
        await asyncio.sleep(2.0)
        return await original_chat(messages, options)

    adapter.chat = slow_chat  # type: ignore[method-assign]

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    runner = AgentRunner(
        adapter,
        registry,
        executor,
        RunnerOptions(model="fake-model", agent_name="cancellable", max_turns=2),
    )

    captured = []

    async def consume() -> None:
        try:
            async for ev in runner.stream(
                [LLMMessage(role="user", content=[TextBlock(text="please respond")])],
            ):
                captured.append(ev)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    done = [ev for ev in captured if ev.type == "done"]
    if not done:
        print("No done event emitted (runner raised before yielding).")
        return

    result = done[-1].data
    print(f"terminal_phase: {result.terminal_phase}")
    if result.stop_reason:
        print(f"stop_reason:    code={result.stop_reason.code} recoverable={result.stop_reason.recoverable}")


if __name__ == "__main__":
    asyncio.run(main())
