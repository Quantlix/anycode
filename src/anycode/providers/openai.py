"""OpenAI SDK adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

try:
    import openai as _openai
except ImportError:
    _openai = None

from anycode.providers._openai_compat import (
    apply_model_params,
    map_messages,
    map_stop_reason,
    map_tool_def,
    parse_chat_response,
    parse_json_safe,
    parse_token_usage,
)
from anycode.types import (
    ContentBlock,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)


class OpenAIAdapter:
    """Wraps the OpenAI Python SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        if _openai is None:
            raise ImportError('OpenAI support requires: pip install "anycode-py[openai]"')
        self._client = _openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

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
        oai_msgs = map_messages(messages, options.system_prompt)
        kwargs: dict[str, Any] = {"model": options.model, "messages": oai_msgs}
        apply_model_params(
            kwargs,
            options.model,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            reasoning_effort=options.reasoning_effort,
        )
        if options.tools:
            kwargs["tools"] = [map_tool_def(t) for t in options.tools]

        if response_format:
            kwargs["response_format"] = response_format

        completion = await self._client.chat.completions.create(**kwargs)
        content, stop, usage = parse_chat_response(completion)

        return LLMResponse(
            id=completion.id,
            content=content,
            model=completion.model,
            stop_reason=stop,
            usage=usage,
        )

    async def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterator[StreamEvent]:
        oai_msgs = map_messages(messages, options.system_prompt)
        kwargs: dict[str, Any] = {"model": options.model, "messages": oai_msgs, "stream": True, "stream_options": {"include_usage": True}}
        apply_model_params(
            kwargs,
            options.model,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            reasoning_effort=options.reasoning_effort,
        )
        if options.tools:
            kwargs["tools"] = [map_tool_def(t) for t in options.tools]

        completion_id = ""
        completion_model = ""
        last_stop = "stop"
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        json_buffers: dict[int, dict[str, str]] = {}
        full_text = ""

        try:
            stream_resp = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream_resp:
                completion_id = chunk.id
                completion_model = chunk.model

                if chunk.usage:
                    usage = parse_token_usage(chunk.usage)

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

            tool_blocks: list[ToolUseBlock] = []
            for buf in json_buffers.values():
                block = ToolUseBlock(id=buf["id"], name=buf["name"], input=parse_json_safe(buf["args"]))
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
                    stop_reason=map_stop_reason(last_stop),
                    usage=usage,
                ),
            )
        except Exception as e:
            yield StreamEvent(type="error", data=str(e))
