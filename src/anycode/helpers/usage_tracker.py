"""Shared token-usage helpers."""

from __future__ import annotations

from anycode.types import TokenUsage

EMPTY_USAGE = TokenUsage(input_tokens=0, output_tokens=0)


def merge_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
    )
