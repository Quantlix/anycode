"""Anthropic SDK adapter."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

from anycode.constants import (
    BLOCK_TYPE_BASE64,
    BLOCK_TYPE_IMAGE,
    BLOCK_TYPE_REDACTED_THINKING,
    BLOCK_TYPE_TEXT,
    BLOCK_TYPE_THINKING,
    BLOCK_TYPE_TOOL_RESULT,
    BLOCK_TYPE_TOOL_USE,
    DEFAULT_MAX_TOKENS,
    STOP_REASON_END_TURN,
)
from anycode.security.redaction import safe_exception_message
from anycode.types import (
    ContentBlock,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    LLMToolDef,
    RedactedThinkingBlock,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolUseBlock,
)

# Extended-thinking budget floor (Anthropic requires >= 1024) and the token
# budget each reasoning-effort tier maps to when no explicit budget is given.
_MIN_THINKING_BUDGET = 1024
_EFFORT_BUDGET_TOKENS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}


def _resolve_thinking_budget(options: LLMChatOptions) -> int | None:
    if options.thinking_budget_tokens is not None:
        return max(options.thinking_budget_tokens, _MIN_THINKING_BUDGET)
    if options.reasoning_effort is not None:
        return _EFFORT_BUDGET_TOKENS.get(options.reasoning_effort)
    return None


def _apply_thinking(kwargs: dict[str, Any], options: LLMChatOptions) -> None:
    """Enable extended thinking when requested.

    With thinking on, the sampling temperature must be unset and ``max_tokens``
    must exceed the thinking budget, so both are reconciled here.
    """
    budget = _resolve_thinking_budget(options)
    if budget is None:
        return
    kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    kwargs.pop("temperature", None)
    if kwargs.get("max_tokens", 0) <= budget:
        kwargs["max_tokens"] = budget + DEFAULT_MAX_TOKENS


def _map_content_block(block: ContentBlock) -> dict[str, Any]:
    if block.type == BLOCK_TYPE_TEXT:
        return {"type": BLOCK_TYPE_TEXT, "text": block.text}
    elif block.type == BLOCK_TYPE_TOOL_USE:
        return {"type": BLOCK_TYPE_TOOL_USE, "id": block.id, "name": block.name, "input": block.input}
    elif block.type == BLOCK_TYPE_TOOL_RESULT:
        result: dict[str, Any] = {"type": BLOCK_TYPE_TOOL_RESULT, "tool_use_id": block.tool_use_id, "content": block.content}
        if block.is_error is not None:
            result["is_error"] = block.is_error
        return result
    elif block.type == BLOCK_TYPE_IMAGE:
        return {"type": BLOCK_TYPE_IMAGE, "source": {"type": BLOCK_TYPE_BASE64, "media_type": block.source.media_type, "data": block.source.data}}
    elif block.type == BLOCK_TYPE_THINKING:
        return {"type": BLOCK_TYPE_THINKING, "thinking": block.thinking, "signature": block.signature}
    elif block.type == BLOCK_TYPE_REDACTED_THINKING:
        return {"type": BLOCK_TYPE_REDACTED_THINKING, "data": block.data}
    raise ValueError(f"Unexpected block type: {block.type}")


def _map_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    return [{"role": msg.role, "content": [_map_content_block(b) for b in msg.content]} for msg in messages]


def _map_tool_defs(tools: list[LLMToolDef]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": {"type": "object", **t.input_schema}} for t in tools]


def _apply_cache_control(kwargs: dict[str, Any]) -> None:
    """Place prompt-cache breakpoints on the stable prefix (tools -> system).

    Render order is tools -> system -> messages, so a single breakpoint on the
    last system block caches tools and system together. When there is no system
    prompt, the breakpoint goes on the last tool definition instead.
    """
    if kwargs.get("system"):
        kwargs["system"] = [{"type": "text", "text": kwargs["system"], "cache_control": {"type": "ephemeral"}}]
    elif kwargs.get("tools"):
        kwargs["tools"][-1]["cache_control"] = {"type": "ephemeral"}


def _parse_block(block: Any) -> ContentBlock:
    if block.type == BLOCK_TYPE_TEXT:
        return TextBlock(text=block.text)
    elif block.type == BLOCK_TYPE_TOOL_USE:
        return ToolUseBlock(id=block.id, name=block.name, input=block.input if isinstance(block.input, dict) else {})
    elif block.type == BLOCK_TYPE_THINKING:
        return ThinkingBlock(thinking=getattr(block, "thinking", ""), signature=getattr(block, "signature", "") or "")
    elif block.type == BLOCK_TYPE_REDACTED_THINKING:
        return RedactedThinkingBlock(data=getattr(block, "data", ""))
    return TextBlock(text=f"[unrecognized block: {block.type}]")


class AnthropicAdapter:
    """Wraps the Anthropic Python SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        if _anthropic is None:
            raise ImportError('Anthropic support requires: pip install "anycode-py[anthropic]"')
        self._client = _anthropic.AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def name(self) -> str:
        return "anthropic"

    async def chat(
        self,
        messages: list[LLMMessage],
        options: LLMChatOptions,
        *,
        structured_tool: LLMToolDef | None = None,
    ) -> LLMResponse:
        mapped = _map_messages(messages)
        kwargs: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": mapped,
        }
        if options.system_prompt:
            kwargs["system"] = options.system_prompt
        if options.tools:
            tool_list = list(options.tools)
            if structured_tool:
                tool_list.append(structured_tool)
            kwargs["tools"] = _map_tool_defs(tool_list)
        elif structured_tool:
            kwargs["tools"] = _map_tool_defs([structured_tool])
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        _apply_thinking(kwargs, options)
        if options.enable_prompt_cache:
            _apply_cache_control(kwargs)

        # Force structured output tool when provided
        if structured_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": structured_tool.name}

        response = await self._client.messages.create(**kwargs)

        return LLMResponse(
            id=response.id,
            content=[_parse_block(b) for b in response.content],
            model=response.model,
            stop_reason=response.stop_reason or STOP_REASON_END_TURN,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            ),
        )

    async def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterator[StreamEvent]:
        mapped = _map_messages(messages)
        kwargs: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": mapped,
        }
        if options.system_prompt:
            kwargs["system"] = options.system_prompt
        if options.tools:
            kwargs["tools"] = _map_tool_defs(options.tools)
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        _apply_thinking(kwargs, options)
        if options.enable_prompt_cache:
            _apply_cache_control(kwargs)

        json_buffers: dict[int, dict[str, str]] = {}

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            json_buffers[event.index] = {"id": block.id, "name": block.name, "json": ""}
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamEvent(type="text", data=delta.text)
                        elif delta.type == "thinking_delta":
                            yield StreamEvent(type="thinking", data=delta.thinking)
                        elif delta.type == "input_json_delta":
                            buf = json_buffers.get(event.index)
                            if buf is not None:
                                buf["json"] += delta.partial_json
                    elif event.type == "content_block_stop":
                        buf = json_buffers.pop(event.index, None)
                        if buf is not None:
                            parsed_input: dict[str, Any] = {}
                            try:
                                parsed = json.loads(buf["json"])
                                if isinstance(parsed, dict):
                                    parsed_input = parsed
                            except (json.JSONDecodeError, TypeError):
                                pass
                            yield StreamEvent(
                                type="tool_use",
                                data=ToolUseBlock(id=buf["id"], name=buf["name"], input=parsed_input),
                            )

                final = await stream.get_final_message()
                yield StreamEvent(
                    type="done",
                    data=LLMResponse(
                        id=final.id,
                        content=[_parse_block(b) for b in final.content],
                        model=final.model,
                        stop_reason=final.stop_reason or "end_turn",
                        usage=TokenUsage(
                            input_tokens=final.usage.input_tokens,
                            output_tokens=final.usage.output_tokens,
                            cache_creation_input_tokens=getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
                            cache_read_input_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                        ),
                    ),
                )
        except Exception as e:
            yield StreamEvent(type="error", data=safe_exception_message(e))
