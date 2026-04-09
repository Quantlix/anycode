"""AWS Bedrock adapter — supports Claude models via Anthropic message format."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from anycode.constants import (
    BLOCK_TYPE_TEXT,
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
    import boto3
except ImportError:
    boto3: Any = None


def _ensure_boto3() -> None:
    if boto3 is None:
        raise ImportError('boto3 is required for the Bedrock provider. Install it with: pip install "anycode-py[bedrock]"')


def _map_content_block(block: ContentBlock) -> dict[str, Any]:
    """Map AnyCode ContentBlock to Bedrock Anthropic message format."""
    if block.type == BLOCK_TYPE_TEXT:
        return {"type": "text", "text": block.text}
    elif block.type == BLOCK_TYPE_TOOL_USE:
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    elif block.type == "tool_result":
        result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.tool_use_id, "content": block.content}
        if block.is_error is not None:
            result["is_error"] = block.is_error
        return result
    elif block.type == "image":
        return {"type": "image", "source": {"type": "base64", "media_type": block.source.media_type, "data": block.source.data}}
    raise ValueError(f"Unexpected block type: {block.type}")


def _map_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Convert AnyCode LLMMessages to Bedrock Anthropic format."""
    return [{"role": msg.role, "content": [_map_content_block(b) for b in msg.content]} for msg in messages]


def _map_tool_defs(tools: list[LLMToolDef]) -> list[dict[str, Any]]:
    """Convert tool definitions to Bedrock Anthropic tool format."""
    return [{"name": t.name, "description": t.description, "input_schema": {"type": "object", **t.input_schema}} for t in tools]


def _parse_block(block: dict[str, Any]) -> ContentBlock:
    """Parse a Bedrock response content block."""
    block_type = block.get("type", "")
    if block_type == "text":
        return TextBlock(text=block.get("text", ""))
    elif block_type == "tool_use":
        return ToolUseBlock(
            id=block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
            name=block.get("name", ""),
            input=block.get("input", {}),
        )
    return TextBlock(text=f"[unrecognized block: {block_type}]")


def _map_stop_reason(reason: str | None) -> str:
    """Map Bedrock stop reason to AnyCode stop reason."""
    mapping = {
        "end_turn": STOP_REASON_END_TURN,
        "tool_use": STOP_REASON_TOOL_USE,
        "max_tokens": STOP_REASON_MAX_TOKENS,
    }
    return mapping.get(reason or "end_turn", reason or STOP_REASON_END_TURN)


class BedrockAdapter:
    """Wraps the AWS Bedrock Runtime API for Anthropic Claude models."""

    def __init__(
        self,
        region: str | None = None,
        profile: str | None = None,
    ) -> None:
        _ensure_boto3()

        session_kwargs: dict[str, str] = {}
        resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.Session(region_name=resolved_region, **session_kwargs)
        self._client = session.client("bedrock-runtime")
        self._region = resolved_region

    @property
    def name(self) -> str:
        return "bedrock"

    async def chat(
        self,
        messages: list[LLMMessage],
        options: LLMChatOptions,
    ) -> LLMResponse:
        body = self._build_request_body(messages, options)
        model_id = options.model

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            ),
        )

        response_body = json.loads(response["body"].read())
        return self._parse_response(response_body, model_id)

    async def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterator[StreamEvent]:
        body = self._build_request_body(messages, options)
        model_id = options.model

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.invoke_model_with_response_stream(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                ),
            )

            full_text = ""
            tool_blocks: list[ToolUseBlock] = []
            current_tool: dict[str, Any] = {}
            input_tokens = 0
            output_tokens = 0

            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                event_type = chunk.get("type", "")

                if event_type == "message_start":
                    usage = chunk.get("message", {}).get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)

                elif event_type == "content_block_start":
                    block = chunk.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {"id": block.get("id", ""), "name": block.get("name", ""), "json": ""}

                elif event_type == "content_block_delta":
                    delta = chunk.get("delta", {})
                    delta_type = delta.get("type", "")
                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        full_text += text
                        yield StreamEvent(type="text", data=text)
                    elif delta_type == "input_json_delta":
                        if current_tool:
                            current_tool["json"] += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    if current_tool:
                        args: dict[str, Any] = {}
                        try:
                            parsed = json.loads(current_tool["json"])
                            if isinstance(parsed, dict):
                                args = parsed
                        except (json.JSONDecodeError, TypeError):
                            pass
                        block = ToolUseBlock(id=current_tool["id"], name=current_tool["name"], input=args)
                        tool_blocks.append(block)
                        yield StreamEvent(type="tool_use", data=block)
                        current_tool = {}

                elif event_type == "message_delta":
                    usage = chunk.get("usage", {})
                    output_tokens = usage.get("output_tokens", output_tokens)

            done_content: list[ContentBlock] = []
            if full_text:
                done_content.append(TextBlock(text=full_text))
            done_content.extend(tool_blocks)

            yield StreamEvent(
                type="done",
                data=LLMResponse(
                    id=f"bedrock-{uuid.uuid4().hex[:12]}",
                    content=done_content,
                    model=model_id,
                    stop_reason=STOP_REASON_TOOL_USE if tool_blocks else STOP_REASON_END_TURN,
                    usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                ),
            )
        except Exception as e:
            yield StreamEvent(type="error", data=str(e))

    def _build_request_body(self, messages: list[LLMMessage], options: LLMChatOptions | LLMStreamOptions) -> dict[str, Any]:
        """Build the Bedrock Anthropic request body."""
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-10-16",
            "max_tokens": options.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": _map_messages(messages),
        }
        if options.system_prompt:
            body["system"] = options.system_prompt
        if options.tools:
            body["tools"] = _map_tool_defs(options.tools)
        if options.temperature is not None:
            body["temperature"] = options.temperature
        return body

    def _parse_response(self, data: dict[str, Any], model_id: str) -> LLMResponse:
        """Parse the raw Bedrock response JSON into an LLMResponse."""
        content_blocks = [_parse_block(b) for b in data.get("content", [])]
        stop = _map_stop_reason(data.get("stop_reason"))
        usage = data.get("usage", {})

        return LLMResponse(
            id=data.get("id", f"bedrock-{uuid.uuid4().hex[:12]}"),
            content=content_blocks,
            model=model_id,
            stop_reason=stop,
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
        )
