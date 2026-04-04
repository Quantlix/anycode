"""OpenAI SDK adapter."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import openai

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


def _map_tool_def(tool: LLMToolDef) -> dict[str, Any]:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}}


def _map_messages(messages: list[LLMMessage], system_prompt: str | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "assistant":
            result.append(_map_assistant(msg))
        else:
            has_tool_results = any(b.type == "tool_result" for b in msg.content)
            if not has_tool_results:
                result.append(_map_user(msg))
            else:
                non_tool = [b for b in msg.content if b.type != "tool_result"]
                if non_tool:
                    result.append(_map_user(LLMMessage(role="user", content=non_tool)))
                for b in msg.content:
                    if b.type == "tool_result":
                        result.append({"role": "tool", "tool_call_id": b.tool_use_id, "content": b.content})
    return result


def _map_user(msg: LLMMessage) -> dict[str, Any]:
    if len(msg.content) == 1 and msg.content[0].type == "text":
        return {"role": "user", "content": msg.content[0].text}
    parts: list[dict[str, Any]] = []
    for b in msg.content:
        if b.type == "text":
            parts.append({"type": "text", "text": b.text})
        elif b.type == "image":
            parts.append({"type": "image_url", "image_url": {"url": f"data:{b.source.media_type};base64,{b.source.data}"}})
    return {"role": "user", "content": parts}


def _map_assistant(msg: LLMMessage) -> dict[str, Any]:
    tool_calls = []
    text_parts: list[str] = []
    for b in msg.content:
        if b.type == "tool_use":
            tool_calls.append({"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}})
        elif b.type == "text":
            text_parts.append(b.text)
    result: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) if text_parts else None}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _map_stop_reason(reason: str | None) -> str:
    mapping = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "content_filter"}
    return mapping.get(reason or "stop", reason or "end_turn")


def _parse_json_safe(s: str) -> dict[str, Any]:
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


class OpenAIAdapter:
    """Wraps the OpenAI Python SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def name(self) -> str:
        return "openai"

    async def chat(
        self,
        messages: list[LLMMessage],
        options: LLMChatOptions,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        oai_msgs = _map_messages(messages, options.system_prompt)
        kwargs: dict[str, Any] = {"model": options.model, "messages": oai_msgs}
        if options.max_tokens:
            kwargs["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.tools:
            kwargs["tools"] = [_map_tool_def(t) for t in options.tools]

        # Structured output via response_format
        if response_format:
            kwargs["response_format"] = response_format

        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        content: list[ContentBlock] = []

        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))
        for tc in choice.message.tool_calls or []:
            content.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=_parse_json_safe(tc.function.arguments)))

        return LLMResponse(
            id=completion.id,
            content=content,
            model=completion.model,
            stop_reason=_map_stop_reason(choice.finish_reason),
            usage=TokenUsage(
                input_tokens=completion.usage.prompt_tokens if completion.usage else 0,
                output_tokens=completion.usage.completion_tokens if completion.usage else 0,
            ),
        )

    async def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterator[StreamEvent]:
        oai_msgs = _map_messages(messages, options.system_prompt)
        kwargs: dict[str, Any] = {"model": options.model, "messages": oai_msgs, "stream": True, "stream_options": {"include_usage": True}}
        if options.max_tokens:
            kwargs["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.tools:
            kwargs["tools"] = [_map_tool_def(t) for t in options.tools]

        completion_id = ""
        completion_model = ""
        last_stop = "stop"
        prompt_tokens = 0
        gen_tokens = 0
        json_buffers: dict[int, dict[str, str]] = {}
        full_text = ""

        try:
            stream_resp = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream_resp:
                completion_id = chunk.id
                completion_model = chunk.model

                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    gen_tokens = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    full_text += delta.content
                    yield StreamEvent(type="text", data=delta.content)

                for tc_delta in delta.tool_calls or []:
                    idx = tc_delta.index
                    if idx not in json_buffers:
                        json_buffers[idx] = {"id": tc_delta.id or "", "name": (tc_delta.function and tc_delta.function.name) or "", "args": ""}
                    buf = json_buffers[idx]
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        buf["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        buf["args"] += tc_delta.function.arguments

                if choice.finish_reason:
                    last_stop = choice.finish_reason

            # Flush tool call buffers
            tool_blocks: list[ToolUseBlock] = []
            for buf in json_buffers.values():
                block = ToolUseBlock(id=buf["id"], name=buf["name"], input=_parse_json_safe(buf["args"]))
                tool_blocks.append(block)
                yield StreamEvent(type="tool_use", data=block)

            done_content: list[ContentBlock] = []
            if full_text:
                done_content.append(TextBlock(text=full_text))
            done_content.extend(tool_blocks)

            yield StreamEvent(
                type="done",
                data=LLMResponse(
                    id=completion_id,
                    content=done_content,
                    model=completion_model,
                    stop_reason=_map_stop_reason(last_stop),
                    usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=gen_tokens),
                ),
            )
        except Exception as e:
            yield StreamEvent(type="error", data=str(e))
