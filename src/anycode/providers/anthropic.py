"""Anthropic SDK adapter."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from anycode.constants import (
    BLOCK_TYPE_BASE64,
    BLOCK_TYPE_IMAGE,
    BLOCK_TYPE_TEXT,
    BLOCK_TYPE_TOOL_RESULT,
    BLOCK_TYPE_TOOL_USE,
    DEFAULT_MAX_TOKENS,
    STOP_REASON_END_TURN,
)
from anycode.types import (
    ContentBlock,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    LLMToolDef,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)


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
    raise ValueError(f"Unexpected block type: {block.type}")


def _map_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    return [{"role": msg.role, "content": [_map_content_block(b) for b in msg.content]} for msg in messages]


def _map_tool_defs(tools: list[LLMToolDef]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": {"type": "object", **t.input_schema}} for t in tools]


def _parse_block(block: Any) -> ContentBlock:
    if block.type == BLOCK_TYPE_TEXT:
        return TextBlock(text=block.text)
    elif block.type == BLOCK_TYPE_TOOL_USE:
        return ToolUseBlock(id=block.id, name=block.name, input=block.input if isinstance(block.input, dict) else {})
    return TextBlock(text=f"[unrecognized block: {block.type}]")


class AnthropicAdapter:
    """Wraps the Anthropic Python SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

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

        # Force structured output tool when provided
        if structured_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": structured_tool.name}

        response = await self._client.messages.create(**kwargs)

        return LLMResponse(
            id=response.id,
            content=[_parse_block(b) for b in response.content],
            model=response.model,
            stop_reason=response.stop_reason or STOP_REASON_END_TURN,
            usage=TokenUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens),
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
                        usage=TokenUsage(input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens),
                    ),
                )
        except Exception as e:
            yield StreamEvent(type="error", data=str(e))
