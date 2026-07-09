"""Pluggable token counting with explicit confidence labels.

Strategy resolution mirrors `ModelContextProfile.tokenizer_strategy`:

* `provider` — provider-native counter (e.g. Anthropic counter API). Not
  embedded here to avoid heavy SDK calls during prompt assembly. When a
  provider counter is unavailable the engine downgrades to `tiktoken`
  if installed, then heuristic, and labels the manifest accordingly.
* `tiktoken` — uses the optional `tiktoken` dependency when present.
* `heuristic` — a deterministic ~4 chars per token approximation.

All counters are pure functions over text/messages; they never mutate state
and never call the network.
"""

from __future__ import annotations

import json
from typing import Final, Protocol

from anycode.types import (
    ContentBlock,
    CountingConfidence,
    LLMMessage,
    TextBlock,
    TokenizerStrategy,
    ToolResultBlock,
    ToolUseBlock,
)

# Heuristic constants ------------------------------------------------------
_CHARS_PER_TOKEN: Final[float] = 4.0
_TOOL_USE_OVERHEAD: Final[int] = 16
_TOOL_RESULT_OVERHEAD: Final[int] = 8
_OTHER_BLOCK_OVERHEAD: Final[int] = 16
_MESSAGE_ROLE_OVERHEAD: Final[int] = 4


class Tokenizer(Protocol):
    """Protocol for token counters used by the context engine."""

    @property
    def confidence(self) -> CountingConfidence: ...
    def count_text(self, text: str) -> int: ...
    def count_block(self, block: ContentBlock) -> int: ...
    def count_messages(self, messages: list[LLMMessage]) -> int: ...


def _heuristic_text(text: str) -> int:
    if not text:
        return 0
    # Round up so very small strings still cost at least one token.
    return max(1, int(round(len(text) / _CHARS_PER_TOKEN)))


def _heuristic_block(block: ContentBlock) -> int:
    if isinstance(block, TextBlock):
        return _heuristic_text(block.text)
    if isinstance(block, ToolUseBlock):
        payload = json.dumps(block.input, default=str, sort_keys=True)
        return _heuristic_text(payload) + _TOOL_USE_OVERHEAD
    if isinstance(block, ToolResultBlock):
        return _heuristic_text(block.content) + _TOOL_RESULT_OVERHEAD
    return _OTHER_BLOCK_OVERHEAD


class HeuristicTokenizer:
    """Deterministic chars/4 token estimator. Always available."""

    @property
    def confidence(self) -> CountingConfidence:
        return "heuristic"

    def count_text(self, text: str) -> int:
        return _heuristic_text(text)

    def count_block(self, block: ContentBlock) -> int:
        return _heuristic_block(block)

    def count_messages(self, messages: list[LLMMessage]) -> int:
        total = 0
        for msg in messages:
            total += _MESSAGE_ROLE_OVERHEAD
            for block in msg.content:
                total += _heuristic_block(block)
        return total


class _TiktokenTokenizer:
    """Optional tiktoken-backed counter. Construction may raise ImportError."""

    def __init__(self, model: str | None = None) -> None:
        import tiktoken  # type: ignore[import-not-found]

        try:
            self._enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except (KeyError, ValueError):
            self._enc = tiktoken.get_encoding("cl100k_base")

    @property
    def confidence(self) -> CountingConfidence:
        return "tokenizer"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_block(self, block: ContentBlock) -> int:
        if isinstance(block, TextBlock):
            return self.count_text(block.text)
        if isinstance(block, ToolUseBlock):
            payload = json.dumps(block.input, default=str, sort_keys=True)
            return self.count_text(payload) + _TOOL_USE_OVERHEAD
        if isinstance(block, ToolResultBlock):
            return self.count_text(block.content) + _TOOL_RESULT_OVERHEAD
        return _OTHER_BLOCK_OVERHEAD

    def count_messages(self, messages: list[LLMMessage]) -> int:
        total = 0
        for msg in messages:
            total += _MESSAGE_ROLE_OVERHEAD
            for block in msg.content:
                total += self.count_block(block)
        return total


DEFAULT_TOKENIZER: Tokenizer = HeuristicTokenizer()


def select_tokenizer(strategy: TokenizerStrategy, *, model: str | None = None) -> Tokenizer:
    """Return the best available tokenizer for `strategy`, falling back safely.

    Resolution order:
        provider -> tiktoken -> heuristic
        tiktoken -> heuristic
        heuristic -> heuristic
    """
    if strategy in ("provider", "tiktoken"):
        try:
            return _TiktokenTokenizer(model=model)
        except ImportError:
            return DEFAULT_TOKENIZER
    return DEFAULT_TOKENIZER


# Convenience module-level helpers (heuristic confidence) ------------------


def count_text(text: str) -> int:
    return DEFAULT_TOKENIZER.count_text(text)


def count_messages(messages: list[LLMMessage]) -> int:
    return DEFAULT_TOKENIZER.count_messages(messages)
