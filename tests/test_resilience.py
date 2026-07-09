"""Tests for the provider resilience layer (retry, deadline, circuit breaker, caching).

Uses FakeAdapter and scripted failure adapters so no LLM keys are required.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from anycode.providers.fake import FakeAdapter
from anycode.providers.resilience import (
    CircuitBreaker,
    ProviderUnavailableError,
    ResilientAdapter,
    is_retryable,
    retry_after_seconds,
)
from anycode.types import (
    LLMChatOptions,
    LLMMessage,
    LLMStreamOptions,
    ProviderResilienceConfig,
    RetryPolicy,
    StreamEvent,
    TextBlock,
)

FAST_RETRY = RetryPolicy(max_attempts=3, base_delay_seconds=0.001, max_delay_seconds=0.01, call_timeout_seconds=5.0)
OPTIONS = LLMChatOptions(model="fake-model")
MESSAGES = [LLMMessage(role="user", content=[TextBlock(text="hi")])]


class _FakeStatusError(Exception):
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        if headers is not None:
            self.response = SimpleNamespace(headers=headers)


class _FlakyAdapter:
    """Fails `fail_times` chat calls with `error`, then delegates to FakeAdapter."""

    def __init__(self, fail_times: int, error: Exception) -> None:
        self._fail_times = fail_times
        self._error = error
        self.calls = 0
        self._inner = FakeAdapter.from_texts(["recovered"])

    @property
    def name(self) -> str:
        return "flaky"

    async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error
        return await self._inner.chat(messages, options)

    def stream(self, messages, options):  # type: ignore[no-untyped-def]
        return self._inner.stream(messages, options)


def _wrap(inner, retry=FAST_RETRY, threshold=5):  # type: ignore[no-untyped-def]
    return ResilientAdapter(
        inner,
        ProviderResilienceConfig(retry=retry, circuit_failure_threshold=threshold, circuit_reset_seconds=60.0),
    )


def test_is_retryable_classification() -> None:
    assert is_retryable(_FakeStatusError(429))
    assert is_retryable(_FakeStatusError(529))
    assert is_retryable(TimeoutError())
    assert is_retryable(ConnectionError("reset"))
    assert not is_retryable(_FakeStatusError(401))
    assert not is_retryable(ValueError("bad request"))


def test_retry_after_extraction() -> None:
    assert retry_after_seconds(_FakeStatusError(429, headers={"retry-after": "2"})) == 2.0
    assert retry_after_seconds(_FakeStatusError(429)) is None
    assert retry_after_seconds(_FakeStatusError(429, headers={"retry-after": "nope"})) is None


async def test_retry_429_then_success() -> None:
    inner = _FlakyAdapter(fail_times=1, error=_FakeStatusError(429, headers={"retry-after": "0"}))
    adapter = _wrap(inner)
    response = await adapter.chat(MESSAGES, OPTIONS)
    assert inner.calls == 2
    assert response.content[0].text == "recovered"  # type: ignore[union-attr]


async def test_terminal_error_not_retried() -> None:
    inner = _FlakyAdapter(fail_times=99, error=_FakeStatusError(401))
    adapter = _wrap(inner)
    with pytest.raises(_FakeStatusError):
        await adapter.chat(MESSAGES, OPTIONS)
    assert inner.calls == 1


async def test_retries_exhausted_raises_provider_unavailable() -> None:
    inner = _FlakyAdapter(fail_times=99, error=_FakeStatusError(429))
    adapter = _wrap(inner)
    with pytest.raises(ProviderUnavailableError):
        await adapter.chat(MESSAGES, OPTIONS)
    assert inner.calls == FAST_RETRY.max_attempts


async def test_deadline_cuts_hung_call() -> None:
    class _HungAdapter:
        calls = 0

        @property
        def name(self) -> str:
            return "hung"

        async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
            type(self).calls += 1
            await asyncio.sleep(5)

        def stream(self, messages, options):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    retry = RetryPolicy(max_attempts=2, base_delay_seconds=0.001, call_timeout_seconds=0.05)
    adapter = _wrap(_HungAdapter())
    adapter._config = ProviderResilienceConfig(retry=retry)
    with pytest.raises(ProviderUnavailableError):
        await adapter.chat(MESSAGES, OPTIONS)
    assert _HungAdapter.calls == 2


async def test_circuit_opens_and_fails_fast() -> None:
    inner = _FlakyAdapter(fail_times=99, error=_FakeStatusError(429))
    retry = RetryPolicy(max_attempts=1, base_delay_seconds=0.001)
    adapter = _wrap(inner, retry=retry, threshold=2)

    for _ in range(2):
        with pytest.raises(ProviderUnavailableError):
            await adapter.chat(MESSAGES, OPTIONS)
    assert inner.calls == 2

    # Circuit is now open: fail fast without touching the inner adapter.
    with pytest.raises(ProviderUnavailableError, match="circuit open"):
        await adapter.chat(MESSAGES, OPTIONS)
    assert inner.calls == 2


def test_circuit_half_open_probe_after_reset() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_seconds=0.0)
    breaker.record_failure()
    assert breaker.allow()  # reset window elapsed -> half-open probe permitted
    breaker.record_success()
    assert breaker.allow()


async def test_stream_error_before_output_retries() -> None:
    class _FlakyStream:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def name(self) -> str:
            return "flaky-stream"

        async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def stream(self, messages, options):  # type: ignore[no-untyped-def]
            self.calls += 1
            first = self.calls == 1

            async def _gen():  # type: ignore[no-untyped-def]
                if first:
                    yield StreamEvent(type="error", data="rate limited")
                else:
                    yield StreamEvent(type="text", data="hello")

            return _gen()

    inner = _FlakyStream()
    adapter = _wrap(inner)
    events = [e async for e in adapter.stream(MESSAGES, LLMStreamOptions(model="fake-model"))]
    assert inner.calls == 2
    assert [e.type for e in events] == ["text"]
    assert events[0].data == "hello"


async def test_stream_retries_exhausted_yields_error_event() -> None:
    class _AlwaysErrorStream:
        calls = 0

        @property
        def name(self) -> str:
            return "dead-stream"

        async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def stream(self, messages, options):  # type: ignore[no-untyped-def]
            type(self).calls += 1

            async def _gen():  # type: ignore[no-untyped-def]
                yield StreamEvent(type="error", data="down")

            return _gen()

    retry = RetryPolicy(max_attempts=2, base_delay_seconds=0.001)
    adapter = _wrap(_AlwaysErrorStream(), retry=retry)
    events = [e async for e in adapter.stream(MESSAGES, LLMStreamOptions(model="fake-model"))]
    assert len(events) == 1
    assert events[0].type == "error"
    assert "unavailable" in str(events[0].data)


async def test_anthropic_cache_control_on_system_and_tools() -> None:
    from anycode.providers.anthropic import AnthropicAdapter
    from anycode.types import LLMToolDef

    adapter = AnthropicAdapter(api_key="test-key")
    captured: dict[str, object] = {}

    async def _fake_create(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(
            id="msg_test",
            content=[],
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
            ),
        )

    adapter._client.messages.create = _fake_create  # type: ignore[method-assign]

    tool = LLMToolDef(name="echo", description="echo", input_schema={"properties": {}})
    options = LLMChatOptions(
        model="claude-sonnet-4-6",
        system_prompt="You are helpful.",
        tools=[tool],
        enable_prompt_cache=True,
    )
    await adapter.chat(MESSAGES, options)

    # Breakpoint on the last system block caches tools + system together.
    system = captured["system"]
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == {"type": "ephemeral"}

    # Without a system prompt the breakpoint lands on the last tool.
    captured.clear()
    options_no_system = LLMChatOptions(model="claude-sonnet-4-6", tools=[tool], enable_prompt_cache=True)
    await adapter.chat(MESSAGES, options_no_system)
    assert captured["tools"][-1]["cache_control"] == {"type": "ephemeral"}  # type: ignore[index]

    # Caching disabled: request shape unchanged.
    captured.clear()
    options_off = LLMChatOptions(model="claude-sonnet-4-6", system_prompt="s", tools=[tool])
    await adapter.chat(MESSAGES, options_off)
    assert captured["system"] == "s"
    assert "cache_control" not in captured["tools"][-1]  # type: ignore[operator]


async def test_runner_surfaces_provider_unavailable_stop_reason() -> None:
    from anycode.core.runner import AgentRunner
    from anycode.tools.executor import ToolExecutor
    from anycode.tools.registry import ToolRegistry
    from anycode.types import RunnerOptions

    inner = _FlakyAdapter(fail_times=99, error=_FakeStatusError(429))
    retry = RetryPolicy(max_attempts=1, base_delay_seconds=0.001)
    adapter = _wrap(inner, retry=retry, threshold=1)

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    options = RunnerOptions(model="fake-model", max_turns=2, agent_name="t")
    runner = AgentRunner(adapter, registry, executor, options)

    result = await runner.run(MESSAGES)
    assert result.stop_reason is not None
    assert result.stop_reason.code == "provider_unavailable"
    assert result.stop_reason.recoverable is True
    assert result.terminal_phase == "failed"


async def test_create_adapter_wraps_by_default() -> None:
    from anycode.providers.adapter import create_adapter

    adapter = await create_adapter("anthropic", api_key="test-key")
    assert isinstance(adapter, ResilientAdapter)

    raw = await create_adapter(
        "anthropic", api_key="test-key", resilience=ProviderResilienceConfig(enabled=False)
    )
    assert not isinstance(raw, ResilientAdapter)
