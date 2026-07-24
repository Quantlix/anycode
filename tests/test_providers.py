"""Tests for additional LLM provider adapters."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anycode.providers._openai_compat import map_messages, map_stop_reason, map_tool_def, parse_json_safe, parse_token_usage
from anycode.providers.adapter import create_adapter
from anycode.providers.anthropic import AnthropicAdapter
from anycode.providers.azure import AzureOpenAIAdapter
from anycode.providers.bedrock import BedrockAdapter
from anycode.providers.google import GeminiAdapter
from anycode.providers.ollama import OllamaAdapter
from anycode.providers.openai import OpenAIAdapter
from anycode.types import (
    ImageBlock,
    ImageSource,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    LLMToolDef,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# ---------------------------------------------------------------------------
# Shared helper tests (_openai_compat)
# ---------------------------------------------------------------------------


def test_anthropic_adapter_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anycode.providers.anthropic._anthropic", None)
    with pytest.raises(ImportError, match=r"anycode-py\[anthropic\]"):
        AnthropicAdapter(api_key="test")


def test_openai_adapter_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anycode.providers.openai._openai", None)
    with pytest.raises(ImportError, match=r"anycode-py\[openai\]"):
        OpenAIAdapter(api_key="test")


class TestOpenAICompat:
    def test_map_tool_def(self) -> None:
        tool = LLMToolDef(name="test", description="A test tool", input_schema={"properties": {"x": {"type": "string"}}})
        result = map_tool_def(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "test"
        assert result["function"]["description"] == "A test tool"

    def test_map_messages_with_system(self) -> None:
        msgs = [LLMMessage(role="user", content=[TextBlock(text="hello")])]
        result = map_messages(msgs, "You are helpful")
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_map_messages_without_system(self) -> None:
        msgs = [LLMMessage(role="user", content=[TextBlock(text="hello")])]
        result = map_messages(msgs, None)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_map_messages_with_tool_results(self) -> None:
        msgs = [
            LLMMessage(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="call_1", content="result"),
                    TextBlock(text="here is the result"),
                ],
            )
        ]
        result = map_messages(msgs, None)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "here is the result"}
        assert result[1] == {"role": "tool", "tool_call_id": "call_1", "content": "result"}

    def test_map_stop_reason(self) -> None:
        assert map_stop_reason("stop") == "end_turn"
        assert map_stop_reason("tool_calls") == "tool_use"
        assert map_stop_reason("length") == "max_tokens"
        assert map_stop_reason(None) == "end_turn"

    def test_parse_json_safe_valid(self) -> None:
        assert parse_json_safe('{"a": 1}') == {"a": 1}

    def test_parse_json_safe_invalid(self) -> None:
        assert parse_json_safe("not json") == {}
        assert parse_json_safe("") == {}

    def test_parse_token_usage_splits_cached_prompt_tokens(self) -> None:
        usage = parse_token_usage(
            {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 50, "cache_creation_tokens": 10},
            }
        )
        assert usage.input_tokens == 60
        assert usage.output_tokens == 30
        assert usage.cache_creation_input_tokens == 10
        assert usage.cache_read_input_tokens == 50


# ---------------------------------------------------------------------------
# Provider factory tests
# ---------------------------------------------------------------------------


class TestProviderFactory:
    async def test_factory_anthropic(self) -> None:
        adapter = await create_adapter("anthropic", api_key="test-key")
        assert adapter.name == "anthropic"

    async def test_factory_openai(self) -> None:
        adapter = await create_adapter("openai", api_key="test-key")
        assert adapter.name == "openai"

    async def test_factory_google(self) -> None:
        with patch("google.genai.Client"):
            adapter = await create_adapter("google", api_key="test-key")
            assert adapter.name == "google"

    async def test_factory_ollama(self) -> None:
        adapter = await create_adapter("ollama", base_url="http://localhost:11434")
        assert adapter.name == "ollama"

    async def test_factory_bedrock(self) -> None:
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.client.return_value = MagicMock()
            adapter = await create_adapter("bedrock", region="us-east-1")
            assert adapter.name == "bedrock"

    async def test_factory_azure(self) -> None:
        adapter = await create_adapter("azure", api_key="test-key", endpoint="https://test.openai.azure.com")
        assert adapter.name == "azure"

    async def test_factory_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            await create_adapter("unknown")


# ---------------------------------------------------------------------------
# Google Gemini Adapter
# ---------------------------------------------------------------------------


class TestGeminiAdapter:
    def _make_adapter(self) -> object:
        with patch("google.genai.Client"):
            return GeminiAdapter(api_key="test-key")

    async def test_chat_returns_llmresponse(self) -> None:
        adapter = self._make_adapter()

        mock_part = MagicMock()
        mock_part.text = "Hello world"
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.cached_content_token_count = 0

        adapter._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        options = LLMChatOptions(model="gemini-2.5-pro")
        messages = [LLMMessage(role="user", content=[TextBlock(text="Hi")])]
        result = await adapter.chat(messages, options)

        assert isinstance(result, LLMResponse)
        assert len(result.content) >= 1
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    async def test_chat_with_tool_call(self) -> None:
        adapter = self._make_adapter()

        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        mock_part = MagicMock()
        mock_part.text = ""
        mock_part.function_call = mock_fc
        type(mock_part).text = ""

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "FUNCTION_CALL"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 15

        adapter._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        options = LLMChatOptions(
            model="gemini-2.5-pro",
            tools=[LLMToolDef(name="search", description="Search", input_schema={"properties": {"query": {"type": "string"}}})],
        )
        messages = [LLMMessage(role="user", content=[TextBlock(text="Search for test")])]
        result = await adapter.chat(messages, options)

        assert isinstance(result, LLMResponse)
        tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1

    async def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiAdapter()

    async def test_system_prompt_passed(self) -> None:
        adapter = self._make_adapter()

        mock_part = MagicMock()
        mock_part.text = "response"
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.cached_content_token_count = 0

        generate_mock = AsyncMock(return_value=mock_response)
        adapter._client.aio.models.generate_content = generate_mock

        options = LLMChatOptions(model="gemini-2.5-pro", system_prompt="Be helpful")
        messages = [LLMMessage(role="user", content=[TextBlock(text="Hi")])]
        await adapter.chat(messages, options)

        generate_mock.assert_called_once()
        config_arg = generate_mock.call_args[1].get("config") or generate_mock.call_args.kwargs.get("config")
        assert config_arg is not None
        assert config_arg.system_instruction == "Be helpful"


# ---------------------------------------------------------------------------
# Ollama Adapter
# ---------------------------------------------------------------------------


def _ollama_chat_client(response_data: dict) -> AsyncMock:
    """Mock httpx.AsyncClient returning a single non-streaming chat response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


def _ollama_stream_client(lines: list[str]) -> AsyncMock:
    """Mock httpx.AsyncClient streaming NDJSON lines from /api/chat."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    async def aiter_lines():
        for line in lines:
            yield line

    mock_resp.aiter_lines = aiter_lines
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=stream_cm)
    return mock_client


class TestOllamaAdapter:
    async def test_chat_sends_native_base64_images(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        mock_client = _ollama_chat_client({"message": {"role": "assistant", "content": "A cat."}})

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMChatOptions(model="llava")
            messages = [
                LLMMessage(
                    role="user",
                    content=[
                        TextBlock(text="What is in this image?"),
                        ImageBlock(source=ImageSource(media_type="image/png", data="aGVsbG8=")),
                    ],
                )
            ]
            await adapter.chat(messages, options)

        payload = mock_client.post.call_args.kwargs["json"]
        sent = payload["messages"][0]
        assert sent["content"] == "What is in this image?"
        assert sent["images"] == ["aGVsbG8="]
        assert "image_url" not in json.dumps(payload)

    async def test_chat_forwards_sampling_options_and_keep_alive(self) -> None:
        adapter = OllamaAdapter(
            base_url="http://localhost:11434",
            keep_alive="5m",
            default_options={"seed": 7, "top_p": 0.9, "num_predict": 1},
        )
        mock_client = _ollama_chat_client({"message": {"role": "assistant", "content": "ok"}})

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMChatOptions(model="llama3.3:70b", max_tokens=256, temperature=0.2)
            await adapter.chat([LLMMessage(role="user", content=[TextBlock(text="Hi")])], options)

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["keep_alive"] == "5m"
        # Per-call options override constructor defaults; untouched defaults pass through.
        assert payload["options"] == {"seed": 7, "top_p": 0.9, "num_predict": 256, "temperature": 0.2}

    async def test_chat_omits_options_and_keep_alive_by_default(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        mock_client = _ollama_chat_client({"message": {"role": "assistant", "content": "ok"}})

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMChatOptions(model="llama3.3:70b")
            await adapter.chat([LLMMessage(role="user", content=[TextBlock(text="Hi")])], options)

        payload = mock_client.post.call_args.kwargs["json"]
        assert "options" not in payload
        assert "keep_alive" not in payload

    async def test_chat_maps_reasoning_effort_to_think_and_parses_thinking(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        mock_client = _ollama_chat_client({"message": {"role": "assistant", "content": "4", "thinking": "2+2 is 4."}, "eval_count": 3})

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMChatOptions(model="qwen3", reasoning_effort="high")
            result = await adapter.chat([LLMMessage(role="user", content=[TextBlock(text="2+2?")])], options)

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["think"] == "high"
        assert isinstance(result.content[0], ThinkingBlock)
        assert result.content[0].thinking == "2+2 is 4."
        assert isinstance(result.content[1], TextBlock)
        assert result.content[1].text == "4"

    async def test_think_resolution_order(self) -> None:
        adapter = OllamaAdapter(think="max")
        assert adapter._resolve_think(LLMChatOptions(model="m")) == "max"
        assert adapter._resolve_think(LLMChatOptions(model="m", thinking_budget_tokens=1024)) is True
        assert adapter._resolve_think(LLMChatOptions(model="m", reasoning_effort="minimal")) == "low"
        assert OllamaAdapter()._resolve_think(LLMChatOptions(model="m")) is None

    async def test_stream_yields_thinking_events(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        lines = [
            json.dumps({"message": {"role": "assistant", "thinking": "Consider"}}),
            json.dumps({"message": {"role": "assistant", "thinking": " carefully."}}),
            json.dumps({"message": {"role": "assistant", "content": "Answer"}}),
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 5, "eval_count": 9}),
        ]
        mock_client = _ollama_stream_client(lines)

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMStreamOptions(model="qwen3")
            events = [e async for e in adapter.stream([LLMMessage(role="user", content=[TextBlock(text="Q")])], options)]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert [e.data for e in thinking_events] == ["Consider", " carefully."]
        done = events[-1]
        assert done.type == "done"
        blocks = done.data.content
        assert isinstance(blocks[0], ThinkingBlock)
        assert blocks[0].thinking == "Consider carefully."
        assert isinstance(blocks[1], TextBlock)
        assert blocks[1].text == "Answer"
        assert done.data.usage.input_tokens == 5
        assert done.data.usage.output_tokens == 9

    @pytest.mark.parametrize(
        ("response_format", "expected"),
        [
            ({"type": "json_object"}, "json"),
            (
                {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}},
                {"type": "object", "properties": {"x": {"type": "integer"}}},
            ),
            ({"type": "object", "properties": {"y": {"type": "string"}}}, {"type": "object", "properties": {"y": {"type": "string"}}}),
        ],
    )
    async def test_chat_translates_response_format(self, response_format: dict, expected: object) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        mock_client = _ollama_chat_client({"message": {"role": "assistant", "content": "{}"}})

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMChatOptions(model="llama3.3:70b")
            await adapter.chat([LLMMessage(role="user", content=[TextBlock(text="Hi")])], options, response_format=response_format)

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["format"] == expected

    async def test_chat_returns_llmresponse(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        response_data = {
            "message": {"role": "assistant", "content": "Hello!"},
            "prompt_eval_count": 15,
            "eval_count": 25,
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMChatOptions(model="llama3.3:70b")
            messages = [LLMMessage(role="user", content=[TextBlock(text="Hello")])]
            result = await adapter.chat(messages, options)

        assert isinstance(result, LLMResponse)
        assert result.content[0].text == "Hello!"
        assert result.usage.input_tokens == 15
        assert result.usage.output_tokens == 25

    async def test_chat_with_tool_call(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        response_data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "NYC"},
                        }
                    }
                ],
            },
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = LLMChatOptions(model="llama3.3:70b")
            messages = [LLMMessage(role="user", content=[TextBlock(text="Weather?")])]
            result = await adapter.chat(messages, options)

        assert result.stop_reason == "tool_use"
        tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "get_weather"

    async def test_default_model_override(self) -> None:
        adapter = OllamaAdapter(model="custom-model")
        assert adapter._default_model == "custom-model"


# ---------------------------------------------------------------------------
# AWS Bedrock Adapter
# ---------------------------------------------------------------------------


class TestBedrockAdapter:
    def _make_adapter(self) -> object:
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.client.return_value = MagicMock()
            return BedrockAdapter(region="us-east-1")

    async def test_chat_returns_llmresponse(self) -> None:
        adapter = self._make_adapter()

        response_body = {
            "id": "msg_bedrock_123",
            "content": [{"type": "text", "text": "Hello from Bedrock"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_body).encode()

        adapter._client.invoke_model = MagicMock(return_value={"body": mock_body})

        options = LLMChatOptions(model="anthropic.claude-3-sonnet-20240229-v1:0")
        messages = [LLMMessage(role="user", content=[TextBlock(text="Hello")])]
        result = await adapter.chat(messages, options)

        assert isinstance(result, LLMResponse)
        assert result.content[0].text == "Hello from Bedrock"
        assert result.usage.input_tokens == 10

    async def test_chat_with_system_prompt(self) -> None:
        adapter = self._make_adapter()

        response_body = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_body).encode()

        call_args: dict = {}

        def capture_invoke(**kwargs: object) -> dict:
            call_args.update(kwargs)
            return {"body": mock_body}

        adapter._client.invoke_model = capture_invoke

        options = LLMChatOptions(model="anthropic.claude-3-sonnet-20240229-v1:0", system_prompt="Be brief")
        messages = [LLMMessage(role="user", content=[TextBlock(text="Hello")])]
        await adapter.chat(messages, options)

        body = json.loads(call_args["body"])
        assert body["system"] == "Be brief"

    async def test_chat_with_tools(self) -> None:
        adapter = self._make_adapter()

        response_body = {
            "content": [{"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "test"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_body).encode()
        adapter._client.invoke_model = MagicMock(return_value={"body": mock_body})

        options = LLMChatOptions(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            tools=[LLMToolDef(name="search", description="Search", input_schema={"properties": {"q": {"type": "string"}}})],
        )
        messages = [LLMMessage(role="user", content=[TextBlock(text="Search")])]
        result = await adapter.chat(messages, options)

        assert result.stop_reason == "tool_use"
        tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1


# ---------------------------------------------------------------------------
# Azure OpenAI Adapter
# ---------------------------------------------------------------------------


class TestAzureOpenAIAdapter:
    async def test_chat_returns_llmresponse(self) -> None:
        adapter = AzureOpenAIAdapter(endpoint="https://test.openai.azure.com", api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = "Hello from Azure"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 12
        mock_usage.completion_tokens = 8

        mock_completion = MagicMock()
        mock_completion.id = "chatcmpl-azure-123"
        mock_completion.model = "gpt-4"
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage

        adapter._client.chat.completions.create = AsyncMock(return_value=mock_completion)

        options = LLMChatOptions(model="gpt-4")
        messages = [LLMMessage(role="user", content=[TextBlock(text="Hello")])]
        result = await adapter.chat(messages, options)

        assert isinstance(result, LLMResponse)
        assert result.content[0].text == "Hello from Azure"
        assert result.usage.input_tokens == 12

    async def test_missing_endpoint_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
                AzureOpenAIAdapter(api_key="test-key")

    async def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
                AzureOpenAIAdapter(endpoint="https://test.openai.azure.com")
