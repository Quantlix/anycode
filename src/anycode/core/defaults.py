"""Provider and model defaults resolved from the environment.

Keyword-constructed agents infer their provider from whichever credentials are present,
so a script does not have to repeat the same key-detection block every time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from anycode.constants import (
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE,
    PROVIDER_BEDROCK,
    PROVIDER_GOOGLE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)

DEFAULT_MODEL_ENV_VAR = "ANYCODE_DEFAULT_MODEL"
DEFAULT_PROVIDER_ENV_VAR = "ANYCODE_DEFAULT_PROVIDER"
OLLAMA_MODEL_ENV_VAR = "OLLAMA_MODEL"

# Detection order — first provider with a populated credential wins.
PROVIDER_CREDENTIAL_ENV_VARS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (PROVIDER_ANTHROPIC, ("ANTHROPIC_API_KEY",)),
    (PROVIDER_OPENAI, ("OPENAI_API_KEY",)),
    (PROVIDER_GOOGLE, ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
    (PROVIDER_AZURE, ("AZURE_OPENAI_API_KEY",)),
    (PROVIDER_BEDROCK, ("AWS_DEFAULT_REGION",)),
    (PROVIDER_OLLAMA, ("OLLAMA_BASE_URL", "OLLAMA_API_KEY")),
)

DEFAULT_MODELS: dict[str, str] = {
    PROVIDER_ANTHROPIC: "claude-haiku-4-5",
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_GOOGLE: "gemini-2.0-flash",
    PROVIDER_AZURE: "gpt-4o-mini",
    PROVIDER_BEDROCK: "anthropic.claude-3-5-haiku-20241022-v1:0",
}


def detect_provider(env: Mapping[str, str] | None = None) -> str | None:
    """Return the provider implied by the environment, or ``None`` when nothing is set."""
    source = env if env is not None else os.environ
    explicit = source.get(DEFAULT_PROVIDER_ENV_VAR, "").strip()
    if explicit:
        return explicit
    for provider, variables in PROVIDER_CREDENTIAL_ENV_VARS:
        if any(source.get(variable, "").strip() for variable in variables):
            return provider
    return None


def default_model(provider: str, env: Mapping[str, str] | None = None) -> str | None:
    """Return the default model for *provider*, honoring environment overrides."""
    source = env if env is not None else os.environ
    override = source.get(DEFAULT_MODEL_ENV_VAR, "").strip()
    if override:
        return override
    if provider == PROVIDER_OLLAMA:
        return source.get(OLLAMA_MODEL_ENV_VAR, "").strip() or None
    return DEFAULT_MODELS.get(provider)


def missing_provider_message() -> str:
    """Explain which environment variables enable provider auto-detection."""
    variables = ", ".join(variables[0] for _, variables in PROVIDER_CREDENTIAL_ENV_VARS)
    return (
        "no LLM provider could be detected from the environment. "
        f"Set one of {variables} (a .env file is loaded by python-dotenv), "
        'or pass provider="..." explicitly.'
    )


def missing_model_message(provider: str) -> str:
    """Explain how to supply a model for a provider that has no built-in default."""
    return f'no default model is known for provider "{provider}". Pass model="..." or set {DEFAULT_MODEL_ENV_VAR} in the environment.'
