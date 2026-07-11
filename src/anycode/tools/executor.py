"""Concurrent tool dispatcher with Pydantic validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from anycode.constants import DEFAULT_TOOL_CONCURRENCY
from anycode.helpers.concurrency_gate import Semaphore
from anycode.hitl.approval import ApprovalManager
from anycode.security.policy import ToolSecurityError, check_tool_access
from anycode.security.redaction import safe_exception_message
from anycode.telemetry.tracer import Tracer
from anycode.tools.idempotency import InMemoryToolIdempotencyStore, ToolIdempotencyStore
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
        idempotency_store: ToolIdempotencyStore | None = None,
    ) -> None:
        self._registry = registry
        self._semaphore = Semaphore(max_concurrency)
        self._tracer = tracer or Tracer()
        self._approval_manager = approval_manager
        self._idempotency_store = idempotency_store or InMemoryToolIdempotencyStore()

    async def execute(
        self,
        tool_name: str,
        raw_input: dict[str, Any],
        context: ToolUseContext,
        *,
        idempotency_key: str | None = None,
    ) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return _failure(f'Tool "{tool_name}" is not registered in the current registry.')

        try:
            check_tool_access(tool_name, context.security_policy)
        except ToolSecurityError as error:
            return _failure(safe_exception_message(error))

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

        return await self._invoke(tool, raw_input, context, idempotency_key=idempotency_key)

    async def execute_batch(self, calls: list[BatchToolCall], context: ToolUseContext) -> dict[str, ToolResult]:
        results: dict[str, ToolResult] = {}

        async def _run(call: BatchToolCall) -> None:
            result = await self._semaphore.run(lambda: self.execute(call.name, call.input, context, idempotency_key=call.id))
            results[call.id] = result

        await asyncio.gather(*[_run(c) for c in calls])
        return results

    async def _invoke(
        self,
        tool: ToolDefinition,
        raw_input: dict[str, Any],
        context: ToolUseContext,
        *,
        idempotency_key: str | None,
    ) -> ToolResult:
        try:
            validated = tool.input_model.model_validate(raw_input)
        except Exception as e:
            return _failure(f'Invalid input for tool "{tool.name}": {safe_exception_message(e)}')

        claim_key: str | None = None
        if tool.side_effecting:
            resolved_key = _resolve_idempotency_key(tool, validated, idempotency_key)
            if resolved_key is None:
                return _failure(f'Side-effecting tool "{tool.name}" requires a non-empty idempotency key.')
            claim_key = resolved_key
            fingerprint = _input_fingerprint(validated)
            try:
                claim = await self._idempotency_store.claim(tool.name, claim_key, fingerprint)
            except Exception as error:
                return _failure(
                    f'Idempotency store unavailable for side-effecting tool "{tool.name}"; the tool was not executed: {safe_exception_message(error)}'
                )
            if claim.outcome == "replay" and claim.result is not None:
                return _normalize_result(tool, claim.result)
            if claim.outcome == "in_progress":
                return _failure(
                    f'Side-effecting tool "{tool.name}" with this idempotency key is already in progress; its outcome is not yet known.',
                    retry_safe=False,
                )
            if claim.outcome == "conflict":
                return _failure(
                    f'Idempotency key conflict for tool "{tool.name}": the key was already used with different input.',
                    retry_safe=False,
                )
            context = context.model_copy(update={"idempotency_key": resolved_key})

        async with self._tracer.async_span(f"anycode.tool.{tool.name}") as span:
            span.set_attributes(SpanAttributes(tool_name=tool.name, agent_name=context.agent.name))
            try:
                result = _normalize_result(tool, await tool.execute(validated, context))
                span.set_attribute("is_error", bool(result.is_error))
            except Exception as e:
                message = safe_exception_message(e)
                span.set_error(message)
                result = _failure(
                    f'Tool "{tool.name}" raised an error: {message}',
                    retry_safe=not tool.side_effecting,
                )
            if claim_key is not None:
                try:
                    await self._idempotency_store.complete(tool.name, claim_key, result)
                except Exception as error:
                    message = safe_exception_message(error)
                    span.set_error(message)
                    return _failure(
                        f'Side-effecting tool "{tool.name}" finished, but its idempotency outcome could not be recorded; '
                        f"the outcome is unknown and must not be retried automatically: {message}",
                        retry_safe=False,
                    )
            return result


def _failure(message: str, *, retry_safe: bool = True) -> ToolResult:
    return ToolResult(data=message, is_error=True, retry_safe=retry_safe)


def _normalize_result(tool: ToolDefinition, result: ToolResult) -> ToolResult:
    if result.retry_safe is not None:
        return result
    retry_safe = not (tool.side_effecting and bool(result.is_error))
    return result.model_copy(update={"retry_safe": retry_safe})


def _resolve_idempotency_key(tool: ToolDefinition, validated: Any, fallback: str | None) -> str | None:
    key: object | None = None
    if tool.idempotency_key_field:
        key = getattr(validated, tool.idempotency_key_field, None)
    if key is None:
        key = fallback
    if not isinstance(key, str) or not key.strip():
        return None
    return key.strip()


def _input_fingerprint(validated: Any) -> str:
    payload = validated.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
