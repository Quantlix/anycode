"""Reasoning-model conformance: OpenAI param shaping and Anthropic extended thinking."""

from __future__ import annotations

from types import SimpleNamespace

from anycode.checkpoint.serializer import _deserialize_content_block, _serialize_content_block
from anycode.providers._openai_compat import apply_model_params, is_reasoning_model
from anycode.providers.anthropic import (
    _apply_thinking,
    _map_content_block,
    _parse_block,
    _resolve_thinking_budget,
)
from anycode.types import LLMChatOptions, LLMMessage, RedactedThinkingBlock, TextBlock, ThinkingBlock

# -- OpenAI reasoning-model params ------------------------------------------


def test_is_reasoning_model() -> None:
    assert is_reasoning_model("o1")
    assert is_reasoning_model("o3-mini")
    assert is_reasoning_model("gpt-5")
    assert is_reasoning_model("openai/o3")  # OpenRouter-style prefix
    assert not is_reasoning_model("gpt-4o")
    assert not is_reasoning_model("gpt-4.1")


def test_apply_model_params_reasoning_model() -> None:
    kwargs: dict[str, object] = {}
    apply_model_params(kwargs, "o3", max_tokens=1000, temperature=0.7, reasoning_effort="high")
    assert kwargs["max_completion_tokens"] == 1000
    assert kwargs["reasoning_effort"] == "high"
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs  # reasoning models reject non-default temperature


def test_apply_model_params_standard_model() -> None:
    kwargs: dict[str, object] = {}
    apply_model_params(kwargs, "gpt-4o", max_tokens=1000, temperature=0.7, reasoning_effort=None)
    assert kwargs["max_tokens"] == 1000
    assert kwargs["temperature"] == 0.7
    assert "max_completion_tokens" not in kwargs


# -- Anthropic extended thinking --------------------------------------------


def test_resolve_thinking_budget_from_effort() -> None:
    assert _resolve_thinking_budget(LLMChatOptions(model="claude", reasoning_effort="medium")) == 8192
    assert _resolve_thinking_budget(LLMChatOptions(model="claude", thinking_budget_tokens=500)) == 1024  # clamped to floor
    assert _resolve_thinking_budget(LLMChatOptions(model="claude")) is None


def test_apply_thinking_shapes_request() -> None:
    kwargs: dict[str, object] = {"max_tokens": 2000, "temperature": 0.5}
    _apply_thinking(kwargs, LLMChatOptions(model="claude", reasoning_effort="high"))
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "temperature" not in kwargs  # unset when thinking is enabled
    assert kwargs["max_tokens"] > 16384  # bumped above the thinking budget


def test_apply_thinking_noop_without_reasoning() -> None:
    kwargs: dict[str, object] = {"max_tokens": 2000, "temperature": 0.5}
    _apply_thinking(kwargs, LLMChatOptions(model="claude"))
    assert "thinking" not in kwargs
    assert kwargs["temperature"] == 0.5


def test_thinking_block_maps_back_with_signature() -> None:
    block = ThinkingBlock(thinking="let me reason", signature="sig-abc")
    mapped = _map_content_block(block)
    assert mapped == {"type": "thinking", "thinking": "let me reason", "signature": "sig-abc"}

    redacted = _map_content_block(RedactedThinkingBlock(data="opaque"))
    assert redacted == {"type": "redacted_thinking", "data": "opaque"}


def test_parse_thinking_block_from_sdk() -> None:
    sdk_block = SimpleNamespace(type="thinking", thinking="reasoning", signature="sig-xyz")
    parsed = _parse_block(sdk_block)
    assert isinstance(parsed, ThinkingBlock)
    assert parsed.thinking == "reasoning"
    assert parsed.signature == "sig-xyz"

    sdk_redacted = SimpleNamespace(type="redacted_thinking", data="opaque")
    parsed_r = _parse_block(sdk_redacted)
    assert isinstance(parsed_r, RedactedThinkingBlock)
    assert parsed_r.data == "opaque"


def test_thinking_block_survives_checkpoint_roundtrip() -> None:
    for block in (ThinkingBlock(thinking="deep", signature="s1"), RedactedThinkingBlock(data="enc")):
        restored = _deserialize_content_block(_serialize_content_block(block))
        assert restored == block


def test_message_with_thinking_and_text_roundtrips() -> None:
    msg = LLMMessage(
        role="assistant",
        content=[ThinkingBlock(thinking="hmm", signature="s"), TextBlock(text="answer")],
    )
    restored = [_deserialize_content_block(_serialize_content_block(b)) for b in msg.content]
    assert restored == list(msg.content)
