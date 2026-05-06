"""Azure OpenAI adapter — reuses shared OpenAI-compatible mapping logic."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from anycode.constants import AZURE_DEFAULT_API_VERSION
from anycode.providers._openai_compat import (
    map_messages,
    map_stop_reason,
    map_tool_def,
    parse_chat_response,
    parse_json_safe,
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

try:
    import openai
except ImportError:
    openai: Any = None


class AzureOpenAIAdapter:
    """Wraps the Azure OpenAI Python SDK — shares mapping logic with the OpenAI adapter."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
    ) -> None:
        if openai is None:
            raise ImportError('openai is required for the Azure OpenAI provider. Install it with: pip install "anycode-py[azure]"')

        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not resolved_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required for the Azure OpenAI provider.")
        if not resolved_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required for the Azure OpenAI provider.")

        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=resolved_endpoint,
            api_key=resolved_key,
            api_version=api_version or AZURE_DEFAULT_API_VERSION,
        )

    @property
    def name(self) -> str:
        return "azure"

    async def chat(
        self,
        messages: list[LLMMessage],
        options: LLMChatOptions,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        oai_msgs = map_messages(messages, options.system_prompt)
        kwargs: dict[str, Any] = {"model": options.model, "messages": oai_msgs}
        if options.max_tokens:
            kwargs["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.tools:
            kwargs["tools"] = [map_tool_def(t) for t in options.tools]

        if response_format:
            kwargs["response_format"] = response_format

        completion = await self._client.chat.completions.create(**kwargs)
        content, stop, input_tokens, output_tokens = parse_chat_response(completion)

        return LLMResponse(
            id=completion.id,
            content=content,
            model=completion.model,
            stop_reason=stop,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterator[StreamEvent]:
        oai_msgs = map_messages(messages, options.system_prompt)
        kwargs: dict[str, Any] = {"model": options.model, "messages": oai_msgs, "stream": True, "stream_options": {"include_usage": True}}
        if options.max_tokens:
            kwargs["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.tools:
            kwargs["tools"] = [map_tool_def(t) for t in options.tools]

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
                    usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=gen_tokens),
                ),
            )
        except Exception as e:
            yield StreamEvent(type="error", data=str(e))
