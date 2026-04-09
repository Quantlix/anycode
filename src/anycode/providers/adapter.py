"""Provider factory — resolves the appropriate LLM backend on demand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anycode.constants import (
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE,
    PROVIDER_BEDROCK,
    PROVIDER_GOOGLE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)

if TYPE_CHECKING:
    from anycode.types import LLMAdapter

SupportedProvider = str  # "anthropic" | "openai" | "google" | "ollama" | "bedrock" | "azure"


async def create_adapter(
    provider: str,
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
    region: str | None = None,
    profile: str | None = None,
    model: str | None = None,
) -> LLMAdapter:
    """Lazy-load the provider SDK and return an adapter instance."""
    if provider == PROVIDER_ANTHROPIC:
        from anycode.providers.anthropic import AnthropicAdapter

        return AnthropicAdapter(api_key=api_key)

    elif provider == PROVIDER_OPENAI:
        from anycode.providers.openai import OpenAIAdapter

        return OpenAIAdapter(api_key=api_key)

    elif provider == PROVIDER_GOOGLE:
        from anycode.providers.google import GeminiAdapter

        return GeminiAdapter(api_key=api_key)

    elif provider == PROVIDER_OLLAMA:
        from anycode.providers.ollama import OllamaAdapter

        return OllamaAdapter(base_url=base_url, model=model)

    elif provider == PROVIDER_BEDROCK:
        from anycode.providers.bedrock import BedrockAdapter

        return BedrockAdapter(region=region, profile=profile)

    elif provider == PROVIDER_AZURE:
        from anycode.providers.azure import AzureOpenAIAdapter

        return AzureOpenAIAdapter(endpoint=endpoint, api_key=api_key, api_version=api_version)

    else:
        raise ValueError(f"Unknown provider requested: {provider}")
