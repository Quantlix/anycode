"""Child process used to prove durable resume after an ungraceful exit."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel

from anycode import (
    AgentRunner,
    DurabilityConfig,
    FakeAdapter,
    FakeResponse,
    FilesystemRunStore,
    LLMMessage,
    RunnerOptions,
    TextBlock,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)

CRASH_EXIT_CODE = 73


class _EmptyInput(BaseModel):
    pass


async def _echo(**_kwargs: object) -> ToolResult:
    return ToolResult(data="echoed")


class _CrashAfterDurableTurn(FakeAdapter):
    async def chat(self, messages, options):  # type: ignore[no-untyped-def]
        if self._cursor >= 1:
            os._exit(CRASH_EXIT_CODE)
        return await super().chat(messages, options)


async def _run(root: Path) -> None:
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="echo", description="echo", input_model=_EmptyInput, execute=_echo))
    runner = AgentRunner(
        _CrashAfterDurableTurn(responses=[FakeResponse(text="durable turn", tool_calls=(("echo", {}),))]),
        registry,
        ToolExecutor(registry),
        RunnerOptions(model="fake-model", max_turns=5, agent_name="crash-fixture"),
        durability=DurabilityConfig(enabled=True, run_root=str(root), checkpoint_every_turns=1),
        run_store=FilesystemRunStore(root),
    )
    await runner.run([LLMMessage(role="user", content=[TextBlock(text="start")])])


if __name__ == "__main__":
    asyncio.run(_run(Path(sys.argv[1])))
