"""Concurrent tool dispatcher with Pydantic validation."""

from __future__ import annotations

import asyncio
from typing import Any

from anycode.helpers.concurrency_gate import Semaphore
from anycode.tools.registry import ToolRegistry
from anycode.types import BatchToolCall, ToolDefinition, ToolResult, ToolUseContext


class ToolExecutor:
    """Validates inputs via Pydantic then invokes tools, with semaphore-bounded batch execution."""

    def __init__(self, registry: ToolRegistry, max_concurrency: int = 4) -> None:
        self._registry = registry
        self._semaphore = Semaphore(max_concurrency)

    async def execute(self, tool_name: str, raw_input: dict[str, Any], context: ToolUseContext) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return _failure(f'Tool "{tool_name}" is not registered in the current registry.')
        return await self._invoke(tool, raw_input, context)

    async def execute_batch(
        self, calls: list[BatchToolCall], context: ToolUseContext
    ) -> dict[str, ToolResult]:
        results: dict[str, ToolResult] = {}

        async def _run(call: BatchToolCall) -> None:
            result = await self._semaphore.run(lambda: self.execute(call.name, call.input, context))
            results[call.id] = result

        await asyncio.gather(*[_run(c) for c in calls])
        return results

    async def _invoke(self, tool: ToolDefinition, raw_input: dict[str, Any], context: ToolUseContext) -> ToolResult:
        try:
            validated = tool.input_model.model_validate(raw_input)
        except Exception as e:
            return _failure(f'Invalid input for tool "{tool.name}": {e}')

        try:
            return await tool.execute(validated, context)
        except Exception as e:
            return _failure(f'Tool "{tool.name}" raised an error: {e}')


def _failure(message: str) -> ToolResult:
    return ToolResult(data=message, is_error=True)
