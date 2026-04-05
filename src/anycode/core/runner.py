"""Agentic dialogue driver — manages LLM interactions, tool dispatch, and turn looping."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Callable

from pydantic import BaseModel

from anycode.constants import DEFAULT_TURN_LIMIT, MAX_VALIDATION_RETRIES, MS_PER_SECOND
from anycode.guardrails.budget import BudgetTracker
from anycode.guardrails.hooks import HookRunner
from anycode.guardrails.validators import run_validators
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.structured.output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    build_retry_prompt,
    parse_structured_output,
    schema_to_tool_def,
)
from anycode.telemetry.tracer import Tracer
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentInfo,
    ContentBlock,
    GuardrailConfig,
    LLMAdapter,
    LLMChatOptions,
    LLMMessage,
    OutputValidator,
    RunnerOptions,
    RunResult,
    SpanAttributes,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
    TurnHook,
)


def _pull_text(blocks: list[ContentBlock]) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def _filter_tool_calls(blocks: list[ContentBlock]) -> list[ToolUseBlock]:
    return [b for b in blocks if isinstance(b, ToolUseBlock)]


class AgentRunner:
    """Orchestrates the full model-tool-model turn loop until completion or limit."""

    def __init__(
        self,
        adapter: LLMAdapter,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        options: RunnerOptions,
        *,
        tracer: Tracer | None = None,
        guardrail_config: GuardrailConfig | None = None,
        hooks: list[TurnHook] | None = None,
        output_validators: list[OutputValidator] | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        self._adapter = adapter
        self._registry = tool_registry
        self._executor = tool_executor
        self._options = options
        self._turn_limit = options.max_turns or DEFAULT_TURN_LIMIT
        self._tracer = tracer or Tracer()
        self._budget = BudgetTracker(guardrail_config, model=options.model)
        self._hook_runner = HookRunner(hooks)
        self._validators = list(output_validators) if output_validators else []
        self._output_schema = output_schema

    @property
    def budget_tracker(self) -> BudgetTracker:
        return self._budget

    async def run(
        self,
        messages: list[LLMMessage],
        on_message: Callable[[LLMMessage], None] | None = None,
    ) -> RunResult:
        fallback = RunResult(messages=[], output="", tool_calls=[], token_usage=EMPTY_USAGE, turns=0)
        async for event in self.stream(messages, on_message=on_message):
            if event.type == "done":
                return event.data  # type: ignore[return-value]
        return fallback

    async def stream(
        self,
        seed_messages: list[LLMMessage],
        on_message: Callable[[LLMMessage], None] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        conversation = list(seed_messages)
        cumulative_usage: TokenUsage = EMPTY_USAGE
        tool_calls: list[ToolCallRecord] = []
        last_output = ""
        turn_count = 0
        validation_retries = 0
        structured_retries = 0
        agent_name = self._options.agent_name or "runner"

        all_defs = self._registry.to_tool_defs()
        active_defs = [d for d in all_defs if d.name in self._options.allowed_tools] if self._options.allowed_tools else all_defs

        if self._output_schema:
            structured_tool = schema_to_tool_def(self._output_schema)
            active_defs = list(active_defs) + [structured_tool] if active_defs else [structured_tool]

        chat_params = LLMChatOptions(
            model=self._options.model,
            tools=active_defs if active_defs else None,
            max_tokens=self._options.max_tokens,
            temperature=self._options.temperature,
            system_prompt=self._options.system_prompt,
        )

        try:
            while turn_count < self._turn_limit:
                if self._budget.is_exhausted():
                    reason = self._budget.get_exhaustion_reason() or "Budget exhausted."
                    last_output = reason
                    yield StreamEvent(type="text", data=reason)
                    break

                turn_count += 1
                self._budget.record_turn()

                ctx_info = self._build_agent_info()
                conversation = await self._hook_runner.run_before_turn(conversation, ctx_info)

                async with self._tracer.async_span(f"anycode.agent.{agent_name}.turn.{turn_count}") as turn_span:
                    turn_span.set_attributes(
                        SpanAttributes(
                            agent_name=agent_name,
                            model=self._options.model,
                            turn_number=turn_count,
                        )
                    )

                    llm_start = time.monotonic()
                    async with self._tracer.async_span("anycode.llm.chat", parent=turn_span) as llm_span:
                        response = await self._adapter.chat(conversation, chat_params)
                        llm_span.set_attributes(
                            SpanAttributes(
                                model=self._options.model,
                                provider=self._adapter.name,
                                token_input=response.usage.input_tokens,
                                token_output=response.usage.output_tokens,
                            )
                        )

                    turn_span.set_attribute("llm_duration_ms", (time.monotonic() - llm_start) * MS_PER_SECOND)
                    cumulative_usage = merge_usage(cumulative_usage, response.usage)
                    self._budget.record_usage(response.usage)

                    response = await self._hook_runner.run_after_turn(response, ctx_info)

                    assistant_msg = LLMMessage(role="assistant", content=response.content)
                    conversation.append(assistant_msg)
                    if on_message:
                        on_message(assistant_msg)

                    turn_text = _pull_text(response.content)
                    if turn_text:
                        yield StreamEvent(type="text", data=turn_text)

                    tool_blocks = _filter_tool_calls(response.content)

                    if self._output_schema:
                        structured_block = next((b for b in tool_blocks if b.name == STRUCTURED_OUTPUT_TOOL_NAME), None)
                        if structured_block:
                            raw_json = json.dumps(structured_block.input)
                            parsed = parse_structured_output(raw_json, self._output_schema)
                            if parsed is not None:
                                last_output = raw_json
                                turn_span.set_attribute("structured_output.valid", True)
                                tool_blocks = [b for b in tool_blocks if b.name != STRUCTURED_OUTPUT_TOOL_NAME]
                                if not tool_blocks:
                                    break
                            else:
                                turn_span.set_attribute("structured_output.valid", False)
                                tool_blocks = [b for b in tool_blocks if b.name != STRUCTURED_OUTPUT_TOOL_NAME]
                                structured_retries += 1
                                if structured_retries <= MAX_VALIDATION_RETRIES and turn_count < self._turn_limit:
                                    retry_msg = LLMMessage(
                                        role="user",
                                        content=[
                                            TextBlock(
                                                text=build_retry_prompt(
                                                    "",
                                                    "Response did not match the required schema. Return valid JSON matching the schema exactly.",
                                                )
                                            )
                                        ],
                                    )
                                    conversation.append(retry_msg)
                                    if on_message:
                                        on_message(retry_msg)
                                    if not tool_blocks:
                                        continue

                    for block in tool_blocks:
                        yield StreamEvent(type="tool_use", data=block)

                    if not tool_blocks:
                        last_output = turn_text or last_output

                        if self._validators and last_output:
                            validation = await run_validators(last_output, self._validators, ctx_info)
                            if not validation.valid and validation.retry:
                                validation_retries += 1
                                if validation_retries <= MAX_VALIDATION_RETRIES and turn_count < self._turn_limit:
                                    retry_msg = LLMMessage(
                                        role="user",
                                        content=[TextBlock(text=build_retry_prompt("", validation.reason or "Output validation failed."))],
                                    )
                                    conversation.append(retry_msg)
                                    if on_message:
                                        on_message(retry_msg)
                                    continue
                        break

                    ctx = self._build_context()
                    results: list[tuple[ToolResultBlock, ToolCallRecord]] = []

                    for block in tool_blocks:
                        if self._budget.is_tool_blocked(block.name):
                            result = ToolResult(data=f'Tool "{block.name}" is blocked by guardrail policy.', is_error=True)
                            result_block = ToolResultBlock(tool_use_id=block.id, content=result.data, is_error=True)
                            record = ToolCallRecord(tool_name=block.name, input=block.input, output=result.data, duration=0.0)
                            results.append((result_block, record))
                            continue

                        began = time.monotonic()
                        async with self._tracer.async_span(f"anycode.tool.{block.name}", parent=turn_span) as tool_span:
                            try:
                                result = await self._executor.execute(block.name, block.input, ctx)
                            except Exception as e:
                                result = ToolResult(data=str(e), is_error=True)
                                tool_span.set_error(str(e))

                            duration = time.monotonic() - began
                            tool_span.set_attributes(SpanAttributes(tool_name=block.name))
                            tool_span.set_attribute("duration_ms", duration * MS_PER_SECOND)
                            tool_span.set_attribute("is_error", bool(result.is_error))

                        result_block = ToolResultBlock(
                            tool_use_id=block.id,
                            content=result.data,
                            is_error=result.is_error,
                        )
                        record = ToolCallRecord(
                            tool_name=block.name,
                            input=block.input,
                            output=result.data,
                            duration=duration,
                        )
                        results.append((result_block, record))

                    self._budget.record_tool_call(len(results))

                    result_blocks: list[ContentBlock] = [r[0] for r in results]
                    for _, record in results:
                        tool_calls.append(record)
                        yield StreamEvent(type="tool_result", data=record)

                    tool_msg = LLMMessage(role="user", content=result_blocks)
                    conversation.append(tool_msg)
                    if on_message:
                        on_message(tool_msg)

        except Exception as e:
            yield StreamEvent(type="error", data=e)
            return

        if not last_output and conversation:
            for msg in reversed(conversation):
                if msg.role == "assistant":
                    last_output = _pull_text(msg.content)
                    break

        yield StreamEvent(
            type="done",
            data=RunResult(
                messages=conversation[len(seed_messages) :],
                output=last_output,
                tool_calls=tool_calls,
                token_usage=cumulative_usage,
                turns=turn_count,
            ),
        )

    def _build_context(self) -> ToolUseContext:
        return ToolUseContext(agent=self._build_agent_info())

    def _build_agent_info(self) -> AgentInfo:
        return AgentInfo(
            name=self._options.agent_name or "runner",
            role=self._options.agent_role or "assistant",
            model=self._options.model,
        )
