"""Provider factory — resolves the appropriate LLM backend on demand.

Built-in providers are dispatched inline; plugin-registered providers go through the
`anycode.plugins.registry` provider-factory registry so external packages can extend
`create_adapter` without modifying core code.
"""

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
    from anycode.types import LLMAdapter, ProviderResilienceConfig

SupportedProvider = str  # "anthropic" | "openai" | "google" | "ollama" | "bedrock" | "azure" | plugin


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
    resilience: ProviderResilienceConfig | None = None,
) -> LLMAdapter:
    """Lazy-load the provider SDK (or plugin factory) and return an adapter instance.

    Adapters are wrapped in `ResilientAdapter` (retry/backoff, deadlines, circuit
    breaker) by default; pass `resilience=ProviderResilienceConfig(enabled=False)`
    for the raw adapter.
    """
    adapter = await _create_raw_adapter(
        provider,
        api_key,
        base_url=base_url,
        endpoint=endpoint,
        api_version=api_version,
        region=region,
        profile=profile,
        model=model,
    )
    from anycode.providers.resilience import ResilientAdapter
    from anycode.types import ProviderResilienceConfig as _Config

    config = resilience or _Config()
    if config.enabled:
        return ResilientAdapter(adapter, config)
    return adapter


async def _create_raw_adapter(
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

    from anycode.plugins.registry import get_provider_factory

    factory = get_provider_factory(provider)
    if factory is not None:
        return await factory(
            api_key=api_key,
            base_url=base_url,
            endpoint=endpoint,
            api_version=api_version,
            region=region,
            profile=profile,
            model=model,
        )

    raise ValueError(f"Unknown provider requested: {provider}")
