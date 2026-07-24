"""Ollama adapter — HTTP-based, zero external dependencies beyond httpx."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from anycode.constants import OLLAMA_DEFAULT_BASE_URL, OLLAMA_REQUEST_TIMEOUT_S, STOP_REASON_END_TURN, STOP_REASON_TOOL_USE
from anycode.providers._openai_compat import map_messages, map_tool_def, parse_json_safe
from anycode.security.redaction import safe_exception_message
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
    import httpx
except ImportError:
    httpx: Any = None

OLLAMA_REQUEST_TIMEOUT = OLLAMA_REQUEST_TIMEOUT_S


def _ensure_httpx() -> None:
    if httpx is None:
        raise ImportError('httpx is required for the Ollama provider. Install it with: pip install "anycode-py[ollama]"')


def _to_native_messages(oai_msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten OpenAI-style content-part lists into Ollama's native message shape.

    The native ``/api/chat`` endpoint takes ``content`` as a plain string plus an
    optional per-message ``images`` array of base64 payloads; it does not accept
    ``image_url`` parts.
    """
    native: list[dict[str, Any]] = []
    for msg in oai_msgs:
        content = msg.get("content")
        if not isinstance(content, list):
            native.append(msg)
            continue
        texts: list[str] = []
        images: list[str] = []
        for part in content:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                images.append(url.split(",", 1)[1] if "," in url else url)
        flattened = {**msg, "content": "".join(texts)}
        if images:
            flattened["images"] = images
        native.append(flattened)
    return native


class OllamaAdapter:
    """HTTP-based adapter for Ollama's native ``/api/chat`` endpoint."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        _ensure_httpx()
        self._base_url = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self._default_model = model

    @property
    def name(self) -> str:
        return "ollama"

    def _build_payload(self, messages: list[LLMMessage], options: LLMChatOptions, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._default_model or options.model,
            "messages": _to_native_messages(map_messages(messages, options.system_prompt)),
            "stream": stream,
        }
        if options.tools:
            payload["tools"] = [map_tool_def(t) for t in options.tools]
        if options.temperature is not None:
            payload.setdefault("options", {})["temperature"] = options.temperature
        return payload

    async def chat(
        self,
        messages: list[LLMMessage],
        options: LLMChatOptions,
    ) -> LLMResponse:
        payload = self._build_payload(messages, options, stream=False)
        model = payload["model"]

        async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content: list[ContentBlock] = []
        msg_data = data.get("message", {})

        if msg_data.get("content"):
            content.append(TextBlock(text=msg_data["content"]))

        for tc in msg_data.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = parse_json_safe(args)
            tool_id = f"call_{uuid.uuid4().hex[:24]}"
            content.append(ToolUseBlock(id=tool_id, name=func.get("name", ""), input=args))

        stop_reason = STOP_REASON_TOOL_USE if msg_data.get("tool_calls") else STOP_REASON_END_TURN

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMResponse(
            id=f"ollama-{uuid.uuid4().hex[:12]}",
            content=content,
            model=model,
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(messages, options, stream=True)
        model = payload["model"]

        full_text = ""
        tool_blocks: list[ToolUseBlock] = []
        input_tokens = 0
        output_tokens = 0

        try:
            async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT) as client:
                async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("prompt_eval_count"):
                            input_tokens = chunk["prompt_eval_count"]
                        if chunk.get("eval_count"):
                            output_tokens = chunk["eval_count"]

                        msg_data = chunk.get("message", {})
                        if msg_data.get("content"):
                            text = msg_data["content"]
                            full_text += text
                            yield StreamEvent(type="text", data=text)

                        for tc in msg_data.get("tool_calls", []):
                            func = tc.get("function", {})
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                args = parse_json_safe(args)
                            tool_id = f"call_{uuid.uuid4().hex[:24]}"
                            block = ToolUseBlock(id=tool_id, name=func.get("name", ""), input=args)
                            tool_blocks.append(block)
                            yield StreamEvent(type="tool_use", data=block)

            done_content: list[ContentBlock] = []
            if full_text:
                done_content.append(TextBlock(text=full_text))
            done_content.extend(tool_blocks)

            stop = STOP_REASON_TOOL_USE if tool_blocks else STOP_REASON_END_TURN

            yield StreamEvent(
                type="done",
                data=LLMResponse(
                    id=f"ollama-{uuid.uuid4().hex[:12]}",
                    content=done_content,
                    model=model,
                    stop_reason=stop,
                    usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                ),
            )
        except Exception as e:
            yield StreamEvent(type="error", data=safe_exception_message(e))
