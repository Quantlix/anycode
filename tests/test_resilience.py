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
    ProviderCapacityConfigurationError,
    ProviderCapacityError,
    ProviderCapacityLimiter,
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
        error = self._error
        inner_stream = self._inner.stream

        async def _gen():  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls <= self._fail_times:
                raise error
            async for event in inner_stream(messages, options):
                yield event

        return _gen()


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


async def test_capacity_limiter_paces_request_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    from anycode.providers import resilience

    clock = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(resilience, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(resilience, "_sleep", fake_sleep)
    limiter = ProviderCapacityLimiter(max_concurrency=None, requests_per_minute=2)

    for _ in range(3):
        await limiter.acquire()
        limiter.release()

    assert sleeps == [30.0, 30.0]
    assert limiter.total_acquisitions == 3


@pytest.mark.parametrize(
    ("max_concurrency", "requests_per_minute"),
    ((0, None), (None, 0)),
)
def test_capacity_limiter_rejects_invalid_limits(max_concurrency: int | None, requests_per_minute: int | None) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        ProviderCapacityLimiter(max_concurrency, requests_per_minute)


async def test_capacity_is_shared_across_provider_adapters() -> None:
    class _Tracker:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

    class _BlockingAdapter:
        def __init__(self, tracker: _Tracker) -> None:
            self._tracker = tracker
            self._inner = FakeAdapter.from_texts(["ok"])

        @property
        def name(self) -> str:
            return "shared-provider"

        async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
            self._tracker.calls += 1
            self._tracker.active += 1
            self._tracker.max_active = max(self._tracker.max_active, self._tracker.active)
            self._tracker.started.set()
            try:
                await self._tracker.release.wait()
                return await self._inner.chat(messages, options)
            finally:
                self._tracker.active -= 1

        def stream(self, messages, options):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    tracker = _Tracker()
    config = ProviderResilienceConfig(max_concurrency=1, capacity_scope="shared-capacity-test")
    first = ResilientAdapter(_BlockingAdapter(tracker), config)
    second = ResilientAdapter(_BlockingAdapter(tracker), config)

    first_task = asyncio.create_task(first.chat(MESSAGES, OPTIONS))
    await tracker.started.wait()
    second_task = asyncio.create_task(second.chat(MESSAGES, OPTIONS))
    await asyncio.sleep(0)

    assert tracker.calls == 1
    assert tracker.max_active == 1

    tracker.release.set()
    await asyncio.gather(first_task, second_task)
    assert tracker.calls == 2
    assert tracker.max_active == 1


async def test_each_retry_reacquires_provider_capacity() -> None:
    inner = _FlakyAdapter(fail_times=1, error=_FakeStatusError(429, headers={"retry-after": "0"}))
    limiter = ProviderCapacityLimiter(max_concurrency=1, requests_per_minute=None)
    adapter = ResilientAdapter(
        inner,
        ProviderResilienceConfig(retry=FAST_RETRY),
        capacity_limiter=limiter,
    )

    await adapter.chat(MESSAGES, OPTIONS)

    assert limiter.total_acquisitions == 2
    assert limiter.active == 0
    assert limiter.pending == 0


async def test_cancellation_releases_provider_capacity() -> None:
    class _BlockedAdapter:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self._inner = FakeAdapter.from_texts(["unreachable"])

        @property
        def name(self) -> str:
            return "cancel-capacity"

        async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
            self.entered.set()
            await asyncio.Event().wait()
            return await self._inner.chat(messages, options)

        def stream(self, messages, options):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    inner = _BlockedAdapter()
    limiter = ProviderCapacityLimiter(max_concurrency=1, requests_per_minute=None)
    adapter = ResilientAdapter(inner, capacity_limiter=limiter)
    task = asyncio.create_task(adapter.chat(MESSAGES, OPTIONS))
    await inner.entered.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert limiter.active == 0
    assert limiter.pending == 0


async def test_closing_stream_releases_provider_capacity() -> None:
    class _StreamingAdapter:
        @property
        def name(self) -> str:
            return "stream-capacity"

        async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def stream(self, messages, options):  # type: ignore[no-untyped-def]
            async def _gen():  # type: ignore[no-untyped-def]
                yield StreamEvent(type="text", data="started")
                await asyncio.Event().wait()

            return _gen()

    limiter = ProviderCapacityLimiter(max_concurrency=1, requests_per_minute=None)
    adapter = ResilientAdapter(_StreamingAdapter(), capacity_limiter=limiter)
    stream = adapter.stream(MESSAGES, LLMStreamOptions(model="fake-model")).__aiter__()

    assert (await anext(stream)).data == "started"
    assert limiter.active == 1
    await stream.aclose()  # type: ignore[attr-defined]
    assert limiter.active == 0


async def test_capacity_wait_timeout_load_sheds_without_opening_circuit() -> None:
    limiter = ProviderCapacityLimiter(max_concurrency=1, requests_per_minute=None)
    await limiter.acquire()
    adapter = ResilientAdapter(
        FakeAdapter.from_texts(["ok"]),
        ProviderResilienceConfig(capacity_wait_timeout_seconds=0.01, circuit_failure_threshold=1),
        capacity_limiter=limiter,
    )

    with pytest.raises(ProviderCapacityError, match="capacity wait timed out"):
        await adapter.chat(MESSAGES, OPTIONS)

    limiter.release()
    response = await adapter.chat(MESSAGES, OPTIONS)
    assert response.content[0].text == "ok"  # type: ignore[union-attr]


async def test_shared_capacity_scope_rejects_conflicting_limits() -> None:
    first = ResilientAdapter(
        FakeAdapter.from_texts(["first"]),
        ProviderResilienceConfig(max_concurrency=1, capacity_scope="conflicting-scope"),
    )
    second = ResilientAdapter(
        FakeAdapter.from_texts(["second"]),
        ProviderResilienceConfig(max_concurrency=2, capacity_scope="conflicting-scope"),
    )

    await first.chat(MESSAGES, OPTIONS)
    with pytest.raises(ProviderCapacityConfigurationError, match="already uses different limits"):
        await second.chat(MESSAGES, OPTIONS)


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
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0),
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

    raw = await create_adapter("anthropic", api_key="test-key", resilience=ProviderResilienceConfig(enabled=False))
    assert not isinstance(raw, ResilientAdapter)


async def test_agent_passes_provider_resilience_to_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    from anycode.core import agent as agent_module
    from anycode.core.agent import Agent
    from anycode.tools.executor import ToolExecutor
    from anycode.tools.registry import ToolRegistry
    from anycode.types import AgentConfig

    captured: dict[str, object] = {}

    async def fake_create_adapter(provider: str, **kwargs: object):  # type: ignore[no-untyped-def]
        captured["provider"] = provider
        captured.update(kwargs)
        return FakeAdapter.from_texts(["ok"])

    monkeypatch.setattr(agent_module, "create_adapter", fake_create_adapter)
    resilience = ProviderResilienceConfig(max_concurrency=2, requests_per_minute=60)
    registry = ToolRegistry()
    agent = Agent(
        AgentConfig(name="capacity", model="fake-model", provider="openai", provider_resilience=resilience),
        registry,
        ToolExecutor(registry),
    )

    await agent._get_runner()

    assert captured == {"provider": "openai", "resilience": resilience}


def test_orchestrator_applies_global_capacity_with_agent_override() -> None:
    from anycode.core.orchestrator import AnyCode
    from anycode.types import AgentConfig, OrchestratorConfig

    global_policy = ProviderResilienceConfig(max_concurrency=3, capacity_scope="global-openai")
    override = ProviderResilienceConfig(max_concurrency=1, capacity_scope="isolated-openai")
    engine = AnyCode(OrchestratorConfig(provider_resilience=global_policy))

    inherited = engine.build_agent(AgentConfig(name="inherited", model="fake", provider="openai"))
    isolated = engine.build_agent(AgentConfig(name="isolated", model="fake", provider="openai", provider_resilience=override))

    assert inherited.config.provider_resilience == global_policy
    assert isolated.config.provider_resilience == override
