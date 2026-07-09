"""Shared mapping helpers for OpenAI-compatible API formats.

Reused by the OpenAI, Azure OpenAI, and Ollama adapters.
"""

from __future__ import annotations

import json
from typing import Any

from anycode.constants import (
    BLOCK_TYPE_IMAGE,
    BLOCK_TYPE_TEXT,
    BLOCK_TYPE_TOOL_RESULT,
    BLOCK_TYPE_TOOL_USE,
    STOP_REASON_CONTENT_FILTER,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_TOOL_USE,
)
from anycode.types import (
    ContentBlock,
    LLMMessage,
    LLMToolDef,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)


def map_tool_def(tool: LLMToolDef) -> dict[str, Any]:
    """Convert a unified LLMToolDef to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def map_messages(messages: list[LLMMessage], system_prompt: str | None) -> list[dict[str, Any]]:
    """Convert AnyCode LLMMessages to OpenAI-compatible message dicts."""
    result: list[dict[str, Any]] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "assistant":
            result.append(map_assistant(msg))
        else:
            has_tool_results = any(b.type == BLOCK_TYPE_TOOL_RESULT for b in msg.content)
            if not has_tool_results:
                result.append(map_user(msg))
            else:
                non_tool: list[ContentBlock] = [b for b in msg.content if b.type != BLOCK_TYPE_TOOL_RESULT]
                if non_tool:
                    result.append(map_user(LLMMessage(role="user", content=non_tool)))
                for b in msg.content:
                    if b.type == BLOCK_TYPE_TOOL_RESULT:
                        result.append({"role": "tool", "tool_call_id": b.tool_use_id, "content": b.content})
    return result


def map_user(msg: LLMMessage) -> dict[str, Any]:
    """Convert a user LLMMessage to OpenAI format."""
    if len(msg.content) == 1 and msg.content[0].type == BLOCK_TYPE_TEXT:
        return {"role": "user", "content": msg.content[0].text}
    parts: list[dict[str, Any]] = []
    for b in msg.content:
        if b.type == BLOCK_TYPE_TEXT:
            parts.append({"type": BLOCK_TYPE_TEXT, "text": b.text})
        elif b.type == BLOCK_TYPE_IMAGE:
            parts.append({"type": "image_url", "image_url": {"url": f"data:{b.source.media_type};base64,{b.source.data}"}})
    return {"role": "user", "content": parts}


def map_assistant(msg: LLMMessage) -> dict[str, Any]:
    """Convert an assistant LLMMessage to OpenAI format."""
    tool_calls = []
    text_parts: list[str] = []
    for b in msg.content:
        if b.type == BLOCK_TYPE_TOOL_USE:
            tool_calls.append({"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}})
        elif b.type == BLOCK_TYPE_TEXT:
            text_parts.append(b.text)
    result: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) if text_parts else None}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def map_stop_reason(reason: str | None) -> str:
    """Map OpenAI stop reason to AnyCode stop reason."""
    mapping = {
        "stop": STOP_REASON_END_TURN,
        "tool_calls": STOP_REASON_TOOL_USE,
        "length": STOP_REASON_MAX_TOKENS,
        "content_filter": STOP_REASON_CONTENT_FILTER,
    }
    return mapping.get(reason or "stop", reason or STOP_REASON_END_TURN)


def parse_json_safe(s: str) -> dict[str, Any]:
    """Parse JSON string safely, returning empty dict on failure."""
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _get_usage_int(value: Any, name: str) -> int:
    if value is None:
        return 0
    raw = value.get(name, 0) if isinstance(value, dict) else getattr(value, name, 0)
    if isinstance(raw, int | float):
        return int(raw)
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return 0


def parse_token_usage(usage: Any) -> TokenUsage:
    """Parse OpenAI-compatible usage details into AnyCode's split cache model."""
    if usage is None:
        return TokenUsage(input_tokens=0, output_tokens=0)

    prompt_tokens = _get_usage_int(usage, "prompt_tokens")
    output_tokens = _get_usage_int(usage, "completion_tokens")
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    cached_tokens = _get_usage_int(prompt_details, "cached_tokens")
    cache_creation_tokens = _get_usage_int(prompt_details, "cache_creation_tokens") or _get_usage_int(prompt_details, "cache_creation_input_tokens")
    fresh_input_tokens = max(prompt_tokens - cached_tokens - cache_creation_tokens, 0)
    return TokenUsage(
        input_tokens=fresh_input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cached_tokens,
    )


def parse_chat_response(completion: Any) -> tuple[list[ContentBlock], str, TokenUsage]:
    """Parse an OpenAI chat completion into content blocks, stop reason, and usage."""
    choice = completion.choices[0]
    content: list[ContentBlock] = []

    if choice.message.content:
        content.append(TextBlock(text=choice.message.content))
    for tc in choice.message.tool_calls or []:
        content.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=parse_json_safe(tc.function.arguments)))

    stop = map_stop_reason(choice.finish_reason)
    usage = parse_token_usage(completion.usage)
    return content, stop, usage
