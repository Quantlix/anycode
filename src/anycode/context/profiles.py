"""Model context-window profiles and resolution rules.

Profiles describe the effective context window and tokenizer behaviour for
a given model. Resolution order (highest priority first):

1. Explicit per-call `ModelContextProfile` (e.g. `ContextPolicy.model_profile`).
2. Custom developer-defined profile in `ContextPolicy.custom_profiles`.
3. Built-in exact match (provider+model).
4. Built-in provider default.
5. Unbounded fallback (no AnyCode-imposed ceiling, with manifest warning).

`max_context_tokens=None` in the resolved profile means "no AnyCode-imposed
ceiling" — the engine still honours known provider failure modes by emitting
warnings and never invents a synthetic ceiling.
"""

from __future__ import annotations

from collections.abc import Iterable

from anycode.types import ModelContextProfile

# Conservative published context windows. These are static for determinism;
# providers ship newer models faster than this registry — `custom_profiles`
# in `ContextPolicy` lets users override or add entries without touching code.
BUILT_IN_PROFILES: tuple[ModelContextProfile, ...] = (
    # --- Anthropic ---------------------------------------------------------
    ModelContextProfile(
        provider="anthropic",
        model="claude-opus-4-5",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        supports_prompt_cache=True,
        tokenizer_strategy="provider",
    ),
    ModelContextProfile(
        provider="anthropic",
        model="claude-opus-4-6",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        supports_prompt_cache=True,
        tokenizer_strategy="provider",
    ),
    ModelContextProfile(
        provider="anthropic",
        model="claude-sonnet-4-5",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        supports_prompt_cache=True,
        tokenizer_strategy="provider",
    ),
    ModelContextProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_context_tokens=1_000_000,
        max_output_tokens=8_192,
        supports_prompt_cache=True,
        tokenizer_strategy="provider",
    ),
    ModelContextProfile(
        provider="anthropic",
        model="claude-haiku-4-5",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        supports_prompt_cache=True,
        tokenizer_strategy="provider",
    ),
    # --- OpenAI ------------------------------------------------------------
    ModelContextProfile(
        provider="openai",
        model="gpt-4o",
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        tokenizer_strategy="tiktoken",
    ),
    ModelContextProfile(
        provider="openai",
        model="gpt-4o-mini",
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        tokenizer_strategy="tiktoken",
    ),
    ModelContextProfile(
        provider="openai",
        model="gpt-4.1",
        max_context_tokens=1_000_000,
        max_output_tokens=32_768,
        tokenizer_strategy="tiktoken",
    ),
    ModelContextProfile(
        provider="openai",
        model="gpt-4.1-mini",
        max_context_tokens=1_000_000,
        max_output_tokens=32_768,
        tokenizer_strategy="tiktoken",
    ),
    ModelContextProfile(
        provider="openai",
        model="gpt-4.1-nano",
        max_context_tokens=1_000_000,
        max_output_tokens=32_768,
        tokenizer_strategy="tiktoken",
    ),
    ModelContextProfile(
        provider="openai",
        model="o3",
        max_context_tokens=200_000,
        max_output_tokens=100_000,
        tokenizer_strategy="tiktoken",
    ),
    ModelContextProfile(
        provider="openai",
        model="o3-mini",
        max_context_tokens=200_000,
        max_output_tokens=100_000,
        tokenizer_strategy="tiktoken",
    ),
    # --- Google ------------------------------------------------------------
    ModelContextProfile(
        provider="google",
        model="gemini-2.5-pro",
        max_context_tokens=2_000_000,
        max_output_tokens=8_192,
        tokenizer_strategy="provider",
    ),
    ModelContextProfile(
        provider="google",
        model="gemini-2.5-flash",
        max_context_tokens=1_000_000,
        max_output_tokens=8_192,
        tokenizer_strategy="provider",
    ),
)

# Provider-level fallback profiles when an exact model match is not found.
PROVIDER_DEFAULT_PROFILES: dict[str, ModelContextProfile] = {
    "anthropic": ModelContextProfile(
        provider="anthropic",
        model="*",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        supports_prompt_cache=True,
        tokenizer_strategy="provider",
    ),
    "openai": ModelContextProfile(
        provider="openai",
        model="*",
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        tokenizer_strategy="tiktoken",
    ),
    "google": ModelContextProfile(
        provider="google",
        model="*",
        max_context_tokens=1_000_000,
        max_output_tokens=8_192,
        tokenizer_strategy="provider",
    ),
    "azure": ModelContextProfile(
        provider="azure",
        model="*",
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        tokenizer_strategy="tiktoken",
    ),
    "bedrock": ModelContextProfile(
        provider="bedrock",
        model="*",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        tokenizer_strategy="heuristic",
    ),
    "ollama": ModelContextProfile(
        provider="ollama",
        model="*",
        max_context_tokens=None,
        max_output_tokens=None,
        tokenizer_strategy="heuristic",
    ),
    "fake": ModelContextProfile(
        provider="fake",
        model="*",
        max_context_tokens=None,
        max_output_tokens=None,
        tokenizer_strategy="heuristic",
    ),
}

# Final fallback — no AnyCode-imposed ceiling. The manifest must surface a
# warning so developers know the resolved profile is not an authoritative
# provider limit.
UNBOUNDED_PROFILE: ModelContextProfile = ModelContextProfile(
    provider="unknown",
    model="*",
    max_context_tokens=None,
    max_output_tokens=None,
    tokenizer_strategy="heuristic",
)


def _find(
    profiles: Iterable[ModelContextProfile],
    *,
    provider: str | None,
    model: str,
) -> ModelContextProfile | None:
    for profile in profiles:
        if profile.model != model:
            continue
        if provider and profile.provider not in (provider, "unknown", "*"):
            continue
        return profile
    return None


def resolve_profile(
    *,
    provider: str | None,
    model: str,
    override: ModelContextProfile | None = None,
    custom_profiles: Iterable[ModelContextProfile] = (),
) -> tuple[ModelContextProfile, str | None]:
    """Resolve the effective context profile for a (provider, model) pair.

    Returns the resolved profile and an optional warning describing how the
    resolution fell back. The warning is suitable for surfacing on the
    `ContextManifest.warnings` tuple.
    """
    if override is not None:
        return override, None

    custom_match = _find(custom_profiles, provider=provider, model=model)
    if custom_match is not None:
        return custom_match, None

    exact = _find(BUILT_IN_PROFILES, provider=provider, model=model)
    if exact is not None:
        return exact, None

    if provider and provider in PROVIDER_DEFAULT_PROFILES:
        default = PROVIDER_DEFAULT_PROFILES[provider]
        return (
            default.model_copy(update={"model": model}),
            f"No exact profile for {provider}:{model}; using provider default.",
        )

    warn = (
        f"No model profile found for provider={provider!r} model={model!r}. "
        "Using unbounded fallback — provider failures may occur if the prompt exceeds the real model window."
    )
    return UNBOUNDED_PROFILE.model_copy(update={"provider": provider or "unknown", "model": model}), warn
