"""Provider factory — resolves the appropriate LLM backend on demand."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anycode.types import LLMAdapter

SupportedProvider = str  # "anthropic" | "openai"


async def create_adapter(provider: str, api_key: str | None = None) -> LLMAdapter:
    """Lazy-load the provider SDK and return an adapter instance."""
    if provider == "anthropic":
        from anycode.providers.anthropic import AnthropicAdapter

        return AnthropicAdapter(api_key=api_key)
    elif provider == "openai":
        from anycode.providers.openai import OpenAIAdapter

        return OpenAIAdapter(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider requested: {provider}")
