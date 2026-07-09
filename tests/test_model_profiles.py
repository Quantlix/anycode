"""Tests for the model context profile registry and resolution rules."""

from __future__ import annotations

from anycode.context.profiles import (
    BUILT_IN_PROFILES,
    PROVIDER_DEFAULT_PROFILES,
    UNBOUNDED_PROFILE,
    resolve_profile,
)
from anycode.types import ModelContextProfile


def test_exact_built_in_match() -> None:
    profile, warning = resolve_profile(provider="anthropic", model="claude-sonnet-4-6")
    assert warning is None
    assert profile.provider == "anthropic"
    assert profile.model == "claude-sonnet-4-6"
    assert profile.max_context_tokens == 1_000_000


def test_provider_default_when_model_unknown() -> None:
    profile, warning = resolve_profile(provider="anthropic", model="claude-x-future")
    assert warning is not None and "default" in warning
    assert profile.provider == "anthropic"
    assert profile.model == "claude-x-future"
    assert profile.max_context_tokens == PROVIDER_DEFAULT_PROFILES["anthropic"].max_context_tokens


def test_custom_profile_beats_built_in() -> None:
    custom = ModelContextProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_context_tokens=5_000_000,
        max_output_tokens=32_768,
        supports_prompt_cache=True,
        tokenizer_strategy="provider",
    )
    profile, warning = resolve_profile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        custom_profiles=(custom,),
    )
    assert warning is None
    assert profile.max_context_tokens == 5_000_000


def test_explicit_override_wins() -> None:
    override = ModelContextProfile(provider="x", model="y", max_context_tokens=42)
    profile, warning = resolve_profile(provider="anthropic", model="claude-sonnet-4-6", override=override)
    assert warning is None
    assert profile.max_context_tokens == 42


def test_unknown_provider_falls_back_to_unbounded() -> None:
    profile, warning = resolve_profile(provider="acme", model="acme-omega")
    assert warning is not None and "unbounded" in warning
    assert profile.max_context_tokens is None
    assert profile.tokenizer_strategy == UNBOUNDED_PROFILE.tokenizer_strategy


def test_built_in_table_non_empty() -> None:
    assert len(BUILT_IN_PROFILES) >= 5
    assert all(p.provider for p in BUILT_IN_PROFILES)
