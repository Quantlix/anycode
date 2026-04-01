"""Agentic dialogue driver — manages LLM interactions, tool dispatch, and turn looping."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable

from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentInfo,
    ContentBlock,
    LLMAdapter,
    LLMChatOptions,
    LLMMessage,
    RunnerOptions,
    RunResult,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
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
    ) -> None:
        self._adapter = adapter
        self._registry = tool_registry
        self._executor = tool_executor
        self._options = options
        self._turn_limit = options.max_turns or 10

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

        # Prepare tool defs
        all_defs = self._registry.to_tool_defs()
        active_defs = (
            [d for d in all_defs if d.name in self._options.allowed_tools]
            if self._options.allowed_tools
            else all_defs
        )

        chat_params = LLMChatOptions(
            model=self._options.model,
            tools=active_defs if active_defs else None,
            max_tokens=self._options.max_tokens,
            temperature=self._options.temperature,
            system_prompt=self._options.system_prompt,
        )

        try:
            while turn_count < self._turn_limit:
                turn_count += 1
                response = await self._adapter.chat(conversation, chat_params)
                cumulative_usage = merge_usage(cumulative_usage, response.usage)

                assistant_msg = LLMMessage(role="assistant", content=response.content)
                conversation.append(assistant_msg)
                if on_message:
                    on_message(assistant_msg)

                turn_text = _pull_text(response.content)
                if turn_text:
                    yield StreamEvent(type="text", data=turn_text)

                tool_blocks = _filter_tool_calls(response.content)
                for block in tool_blocks:
                    yield StreamEvent(type="tool_use", data=block)

                if not tool_blocks:
                    last_output = turn_text
                    break

                ctx = self._build_context()
                results: list[tuple[ToolResultBlock, ToolCallRecord]] = []

                for block in tool_blocks:
                    began = time.monotonic()
                    try:
                        result = await self._executor.execute(block.name, block.input, ctx)
                    except Exception as e:
                        result = ToolResult(data=str(e), is_error=True)

                    result_block = ToolResultBlock(
                        tool_use_id=block.id,
                        content=result.data,
                        is_error=result.is_error,
                    )
                    record = ToolCallRecord(
                        tool_name=block.name,
                        input=block.input,
                        output=result.data,
                        duration=time.monotonic() - began,
                    )
                    results.append((result_block, record))

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
                messages=conversation[len(seed_messages):],
                output=last_output,
                tool_calls=tool_calls,
                token_usage=cumulative_usage,
                turns=turn_count,
            ),
        )

    def _build_context(self) -> ToolUseContext:
        return ToolUseContext(
            agent=AgentInfo(
                name=self._options.agent_name or "runner",
                role=self._options.agent_role or "assistant",
                model=self._options.model,
            )
        )
