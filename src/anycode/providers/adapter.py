"""Provider factory — resolves the appropriate LLM backend on demand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anycode.constants import PROVIDER_ANTHROPIC, PROVIDER_OPENAI

if TYPE_CHECKING:
    from anycode.types import LLMAdapter

SupportedProvider = str  # "anthropic" | "openai"


async def create_adapter(provider: str, api_key: str | None = None) -> LLMAdapter:
    """Lazy-load the provider SDK and return an adapter instance."""
    if provider == PROVIDER_ANTHROPIC:
        from anycode.providers.anthropic import AnthropicAdapter

        return AnthropicAdapter(api_key=api_key)
    elif provider == PROVIDER_OPENAI:
        from anycode.providers.openai import OpenAIAdapter

        return OpenAIAdapter(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider requested: {provider}")
