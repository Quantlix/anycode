"""Tests for additional LLM provider adapters."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anycode.providers._openai_compat import map_messages, map_stop_reason, map_tool_def, parse_json_safe
from anycode.providers.adapter import create_adapter
from anycode.providers.azure import AzureOpenAIAdapter
from anycode.providers.bedrock import BedrockAdapter
from anycode.providers.google import GeminiAdapter
from anycode.providers.ollama import OllamaAdapter
from anycode.types import (
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMToolDef,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# ---------------------------------------------------------------------------
# Shared helper tests (_openai_compat)
# ---------------------------------------------------------------------------


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


class TestOllamaAdapter:
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
