"""Concurrent tool dispatcher with Pydantic validation."""

from __future__ import annotations

import asyncio
from typing import Any

from anycode.helpers.concurrency_gate import Semaphore
from anycode.telemetry.tracer import Tracer
from anycode.tools.registry import ToolRegistry
from anycode.types import BatchToolCall, SpanAttributes, ToolDefinition, ToolResult, ToolUseContext

DEFAULT_TOOL_CONCURRENCY = 4


class ToolExecutor:
    """Validates inputs via Pydantic then invokes tools, with semaphore-bounded batch execution."""

    def __init__(self, registry: ToolRegistry, max_concurrency: int = DEFAULT_TOOL_CONCURRENCY, tracer: Tracer | None = None) -> None:
        self._registry = registry
        self._semaphore = Semaphore(max_concurrency)
        self._tracer = tracer or Tracer()

    async def execute(self, tool_name: str, raw_input: dict[str, Any], context: ToolUseContext) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return _failure(f'Tool "{tool_name}" is not registered in the current registry.')
        return await self._invoke(tool, raw_input, context)

    async def execute_batch(self, calls: list[BatchToolCall], context: ToolUseContext) -> dict[str, ToolResult]:
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

        async with self._tracer.async_span(f"anycode.tool.{tool.name}") as span:
            span.set_attributes(SpanAttributes(tool_name=tool.name, agent_name=context.agent.name))
            try:
                result = await tool.execute(validated, context)
                span.set_attribute("is_error", bool(result.is_error))
                return result
            except Exception as e:
                span.set_error(str(e))
                return _failure(f'Tool "{tool.name}" raised an error: {e}')


def _failure(message: str) -> ToolResult:
    return ToolResult(data=message, is_error=True)
