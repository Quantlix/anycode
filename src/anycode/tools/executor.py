"""Concurrent tool dispatcher with Pydantic validation."""

from __future__ import annotations

import asyncio
from typing import Any

from anycode.constants import DEFAULT_TOOL_CONCURRENCY
from anycode.helpers.concurrency_gate import Semaphore
from anycode.hitl.approval import ApprovalManager
from anycode.telemetry.tracer import Tracer
from anycode.tools.registry import ToolRegistry
from anycode.types import BatchToolCall, SpanAttributes, ToolDefinition, ToolResult, ToolUseContext


class ToolExecutor:
    """Validates inputs via Pydantic then invokes tools, with semaphore-bounded batch execution."""

    def __init__(
        self,
        registry: ToolRegistry,
        max_concurrency: int = DEFAULT_TOOL_CONCURRENCY,
        tracer: Tracer | None = None,
        approval_manager: object | None = None,
    ) -> None:
        self._registry = registry
        self._semaphore = Semaphore(max_concurrency)
        self._tracer = tracer or Tracer()
        self._approval_manager = approval_manager

    async def execute(self, tool_name: str, raw_input: dict[str, Any], context: ToolUseContext) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return _failure(f'Tool "{tool_name}" is not registered in the current registry.')

        # HITL: tool-level approval
        if self._approval_manager is not None:
            if isinstance(self._approval_manager, ApprovalManager):
                response = await self._approval_manager.check_and_request(
                    request_type="tool_call",
                    agent=context.agent.name,
                    description=f"Execute tool: {tool_name}",
                    context={"tool_name": tool_name, "input": raw_input},
                )
                if response and not response.approved:
                    return _failure(f'Approval denied for tool "{tool_name}": {response.reason or "rejected"}')
                if response and response.modified_input:
                    raw_input = response.modified_input

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
