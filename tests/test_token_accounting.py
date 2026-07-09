"""Tests for token counting strategies and confidence labelling."""

from __future__ import annotations

from anycode.context.tokenizer import (
    DEFAULT_TOKENIZER,
    HeuristicTokenizer,
    count_messages,
    count_text,
    select_tokenizer,
)
from anycode.types import LLMMessage, TextBlock, ToolResultBlock, ToolUseBlock


def test_heuristic_text_counts_proportional_to_length() -> None:
    short = count_text("hello")
    longer = count_text("hello " * 100)
    assert short >= 1
    assert longer > short


def test_heuristic_messages_includes_role_overhead() -> None:
    msg = LLMMessage(role="user", content=[TextBlock(text="hi")])
    n = count_messages([msg])
    assert n >= 1


def test_heuristic_block_counts_for_tool_use_and_result() -> None:
    t = HeuristicTokenizer()
    use = ToolUseBlock(id="t1", name="echo", input={"x": "hello"})
    result = ToolResultBlock(tool_use_id="t1", content="ok")
    assert t.count_block(use) > 0
    assert t.count_block(result) > 0


def test_select_tokenizer_returns_heuristic_for_unknown_strategy() -> None:
    tok = select_tokenizer("heuristic")
    assert tok.confidence == "heuristic"


def test_select_tokenizer_falls_back_when_tiktoken_missing() -> None:
    # Even if tiktoken isn't installed, the call must not raise.
    tok = select_tokenizer("tiktoken", model="gpt-4o")
    assert tok.confidence in ("tokenizer", "heuristic")


def test_default_tokenizer_is_heuristic() -> None:
    assert DEFAULT_TOKENIZER.confidence == "heuristic"
