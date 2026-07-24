"""Ollama adapter — HTTP-based, zero external dependencies beyond httpx."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from anycode.constants import (
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT_S,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_TOOL_USE,
)
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
    ThinkingBlock,
    TokenUsage,
    ToolUseBlock,
)

try:
    import httpx
except ImportError:
    httpx: Any = None

OLLAMA_REQUEST_TIMEOUT = OLLAMA_REQUEST_TIMEOUT_S

# Ollama's `think` levels; `minimal` has no native equivalent and maps to "low".
_EFFORT_TO_THINK = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}

_DONE_REASON_MAP = {"stop": STOP_REASON_END_TURN, "length": STOP_REASON_MAX_TOKENS}


def _map_done_reason(done_reason: str | None, *, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return STOP_REASON_TOOL_USE
    return _DONE_REASON_MAP.get(done_reason or "stop", STOP_REASON_END_TURN)


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

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        keep_alive: str | float | None = None,
        think: bool | str | None = None,
        default_options: dict[str, Any] | None = None,
    ) -> None:
        _ensure_httpx()
        self._base_url = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self._default_model = model
        self._api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self._keep_alive = keep_alive
        self._think = think
        self._default_options = dict(default_options) if default_options else {}

    def _headers(self) -> dict[str, str]:
        # ollama.com cloud requires a Bearer key; local servers ignore the header.
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def _raise_for_status(self, resp: Any, model: str | None) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise httpx.HTTPStatusError(
                    f"Model '{model}' was not found on the Ollama server at {self._base_url}. Pull it first: ollama pull {model}",
                    request=e.request,
                    response=e.response,
                ) from e
            raise

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
        model_options: dict[str, Any] = dict(self._default_options)
        if options.max_tokens is not None:
            model_options["num_predict"] = options.max_tokens
        if options.temperature is not None:
            model_options["temperature"] = options.temperature
        if model_options:
            payload["options"] = model_options
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        think = self._resolve_think(options)
        if think is not None:
            payload["think"] = think
        return payload

    def _resolve_think(self, options: LLMChatOptions) -> bool | str | None:
        if options.reasoning_effort is not None:
            return _EFFORT_TO_THINK[options.reasoning_effort]
        if options.thinking_budget_tokens is not None:
            # Ollama has no budget concept; any budget request enables thinking.
            return True
        return self._think

    @staticmethod
    def _to_native_format(response_format: dict[str, Any]) -> str | dict[str, Any]:
        """Translate an OpenAI-style response_format into Ollama's ``format`` value."""
        kind = response_format.get("type")
        if kind == "json_object":
            return "json"
        if kind == "json_schema":
            return response_format.get("json_schema", {}).get("schema", {})
        return response_format

    async def chat(
        self,
        messages: list[LLMMessage],
        options: LLMChatOptions,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(messages, options, stream=False)
        if response_format:
            payload["format"] = self._to_native_format(response_format)
        model = payload["model"]

        async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT, headers=self._headers()) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            self._raise_for_status(resp, model)
            data = resp.json()

        content: list[ContentBlock] = []
        msg_data = data.get("message", {})

        if msg_data.get("thinking"):
            content.append(ThinkingBlock(thinking=msg_data["thinking"]))
        if msg_data.get("content"):
            content.append(TextBlock(text=msg_data["content"]))

        for tc in msg_data.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = parse_json_safe(args)
            tool_id = f"call_{uuid.uuid4().hex[:24]}"
            content.append(ToolUseBlock(id=tool_id, name=func.get("name", ""), input=args))

        stop_reason = _map_done_reason(data.get("done_reason"), has_tool_calls=bool(msg_data.get("tool_calls")))

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
        full_thinking = ""
        tool_blocks: list[ToolUseBlock] = []
        input_tokens = 0
        output_tokens = 0
        done_reason: str | None = None

        try:
            async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT, headers=self._headers()) as client:
                async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                    self._raise_for_status(resp, model)
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Ollama reports mid-stream failures as NDJSON error objects
                        # on an already-established 200 response.
                        if chunk.get("error"):
                            yield StreamEvent(type="error", data=str(chunk["error"]))
                            return

                        if chunk.get("done_reason"):
                            done_reason = chunk["done_reason"]
                        if chunk.get("prompt_eval_count"):
                            input_tokens = chunk["prompt_eval_count"]
                        if chunk.get("eval_count"):
                            output_tokens = chunk["eval_count"]

                        msg_data = chunk.get("message", {})
                        if msg_data.get("thinking"):
                            thinking_text = msg_data["thinking"]
                            full_thinking += thinking_text
                            yield StreamEvent(type="thinking", data=thinking_text)
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
            if full_thinking:
                done_content.append(ThinkingBlock(thinking=full_thinking))
            if full_text:
                done_content.append(TextBlock(text=full_text))
            done_content.extend(tool_blocks)

            stop = _map_done_reason(done_reason, has_tool_calls=bool(tool_blocks))

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
