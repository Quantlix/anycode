"""Google Gemini adapter — uses the ``google-genai`` SDK (>= 1.0).

The deprecated ``google-generativeai`` package is **not** used.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from anycode.constants import (
    BLOCK_TYPE_TEXT,
    BLOCK_TYPE_TOOL_RESULT,
    BLOCK_TYPE_TOOL_USE,
    DEFAULT_MAX_TOKENS,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_TOOL_USE,
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

try:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types as genai_types  # type: ignore[import-untyped]
except ImportError:
    genai: Any = None
    genai_types: Any = None


def _map_tool_defs(tools: list[LLMToolDef]) -> list[Any]:
    """Convert AnyCode tool defs to ``types.Tool(function_declarations=[...])``."""
    declarations: list[Any] = []
    for t in tools:
        schema = dict(t.input_schema)
        schema.setdefault("type", "object")
        declarations.append(
            genai_types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters_json_schema=schema,
            )
        )
    return [genai_types.Tool(function_declarations=declarations)]


def _map_messages(messages: list[LLMMessage]) -> list[Any]:
    """Convert AnyCode LLMMessages to ``types.Content`` list."""
    contents: list[Any] = []
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        parts: list[Any] = []

        for block in msg.content:
            if block.type == BLOCK_TYPE_TEXT:
                parts.append(genai_types.Part.from_text(text=block.text))
            elif block.type == BLOCK_TYPE_TOOL_USE:
                parts.append(
                    genai_types.Part.from_function_call(
                        name=block.name,
                        args=block.input,
                    )
                )
            elif block.type == BLOCK_TYPE_TOOL_RESULT:
                parts.append(
                    genai_types.Part.from_function_response(
                        name=block.tool_use_id,
                        response={"result": block.content},
                    )
                )

        if parts:
            contents.append(genai_types.Content(role=role, parts=parts))

    return contents


def _map_stop_reason(reason: Any) -> str:
    """Map Gemini finish reason to AnyCode stop reason."""
    reason_str = str(reason).upper() if reason else "STOP"
    if "MAX_TOKENS" in reason_str:
        return STOP_REASON_MAX_TOKENS
    if "TOOL" in reason_str or "FUNCTION" in reason_str:
        return STOP_REASON_TOOL_USE
    return STOP_REASON_END_TURN


def _parse_response_parts(parts: Any) -> list[ContentBlock]:
    """Parse Gemini response parts into AnyCode ContentBlocks."""
    blocks: list[ContentBlock] = []
    if parts is None:
        return blocks

    for part in parts:
        if hasattr(part, "text") and part.text:
            blocks.append(TextBlock(text=part.text))

        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            args: dict[str, Any] = {}
            raw = getattr(fc, "args", None)
            if raw:
                try:
                    if isinstance(raw, str):
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            args = parsed
                    elif isinstance(raw, dict):
                        args = dict(raw)
                    else:
                        args = dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            tool_id = f"call_{uuid.uuid4().hex[:24]}"
            blocks.append(ToolUseBlock(id=tool_id, name=fc.name, input=args))

    return blocks


class GeminiAdapter:
    """Wraps the ``google-genai`` SDK (``from google import genai``)."""

    def __init__(self, api_key: str | None = None) -> None:
        if genai is None:
            raise ImportError('google-genai is required for the Gemini provider. Install it with: pip install "anycode-py[google]"')

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable is required for the Gemini provider.")

        self._client = genai.Client(api_key=resolved_key)

    @property
    def name(self) -> str:
        return "google"

    async def chat(
        self,
        messages: list[LLMMessage],
        options: LLMChatOptions,
    ) -> LLMResponse:
        config_kwargs: dict[str, Any] = {}
        if options.system_prompt:
            config_kwargs["system_instruction"] = options.system_prompt
        config_kwargs["max_output_tokens"] = options.max_tokens or DEFAULT_MAX_TOKENS
        if options.temperature is not None:
            config_kwargs["temperature"] = options.temperature
        if options.tools:
            config_kwargs["tools"] = _map_tool_defs(options.tools)
            config_kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(disable=True)

        contents = _map_messages(messages)
        response = await self._client.aio.models.generate_content(
            model=options.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )

        candidate_parts = response.candidates[0].content.parts if response.candidates else None
        content_blocks = _parse_response_parts(candidate_parts)
        stop_reason = _map_stop_reason(response.candidates[0].finish_reason if response.candidates else None)

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            input_tokens = getattr(meta, "prompt_token_count", 0) or 0
            output_tokens = getattr(meta, "candidates_token_count", 0) or 0

        return LLMResponse(
            id=f"gemini-{uuid.uuid4().hex[:12]}",
            content=content_blocks,
            model=options.model,
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterator[StreamEvent]:
        config_kwargs: dict[str, Any] = {}
        if options.system_prompt:
            config_kwargs["system_instruction"] = options.system_prompt
        config_kwargs["max_output_tokens"] = options.max_tokens or DEFAULT_MAX_TOKENS
        if options.temperature is not None:
            config_kwargs["temperature"] = options.temperature
        if options.tools:
            config_kwargs["tools"] = _map_tool_defs(options.tools)
            config_kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(disable=True)

        contents = _map_messages(messages)

        try:
            response_stream = await self._client.aio.models.generate_content_stream(
                model=options.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )

            full_text = ""
            all_blocks: list[ContentBlock] = []
            input_tokens = 0
            output_tokens = 0

            async for chunk in response_stream:
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    meta = chunk.usage_metadata
                    input_tokens = getattr(meta, "prompt_token_count", 0) or 0
                    output_tokens = getattr(meta, "candidates_token_count", 0) or 0

                if not chunk.candidates:
                    continue

                parts = chunk.candidates[0].content.parts if chunk.candidates[0].content else None
                blocks = _parse_response_parts(parts)

                for block in blocks:
                    if isinstance(block, TextBlock):
                        full_text += block.text
                        yield StreamEvent(type="text", data=block.text)
                    elif isinstance(block, ToolUseBlock):
                        yield StreamEvent(type="tool_use", data=block)
                    all_blocks.append(block)

            done_content: list[ContentBlock] = []
            if full_text:
                done_content.append(TextBlock(text=full_text))
            done_content.extend(b for b in all_blocks if isinstance(b, ToolUseBlock))

            yield StreamEvent(
                type="done",
                data=LLMResponse(
                    id=f"gemini-{uuid.uuid4().hex[:12]}",
                    content=done_content,
                    model=options.model,
                    stop_reason=STOP_REASON_END_TURN,
                    usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                ),
            )
        except Exception as e:
            yield StreamEvent(type="error", data=str(e))
