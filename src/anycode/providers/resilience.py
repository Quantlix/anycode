"""Provider resilience layer — retry, per-call deadlines, and circuit breaking.

`ResilientAdapter` wraps any `LLMAdapter` (same Protocol) so every provider gains
transient-failure handling without per-adapter changes:

* Transient failures (429, 5xx, timeouts, connection errors) retry with
  exponential backoff and full jitter, honoring `Retry-After` when present.
* Every call runs under a wall-clock deadline (`asyncio.wait_for`).
* Repeated failures open a per-adapter circuit; while open, calls fail fast
  with `ProviderUnavailableError` instead of hammering a downed provider.
  After `circuit_reset_seconds` a single half-open probe is allowed.

Terminal errors (auth, invalid request) are never retried and propagate as-is.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterable, AsyncIterator

from anycode.types import (
    LLMAdapter,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    ProviderResilienceConfig,
    RetryPolicy,
    StreamEvent,
)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_RETRYABLE_NAME_HINTS = ("timeout", "connection", "ratelimit", "rate_limit", "overloaded")


class ProviderUnavailableError(Exception):
    """Raised when the circuit is open or transient-failure retries are exhausted."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"Provider '{provider}' unavailable: {message}")
        self.provider = provider


def is_retryable(error: BaseException) -> bool:
    """Classify an adapter/SDK error as transient (retry) or terminal (raise)."""
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        status = getattr(error, "status", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    name = type(error).__name__.lower()
    return any(hint in name for hint in _RETRYABLE_NAME_HINTS)


def retry_after_seconds(error: BaseException) -> float | None:
    """Extract a Retry-After hint (seconds) from an SDK error, if present."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _backoff_delay(attempt: int, retry: RetryPolicy) -> float:
    delay = min(retry.max_delay_seconds, retry.base_delay_seconds * (2 ** (attempt - 1)))
    if retry.jitter:
        delay *= random.uniform(0.5, 1.0)
    return delay


class CircuitBreaker:
    """Consecutive-failure circuit with a half-open probe after the reset window."""

    def __init__(self, failure_threshold: int, reset_seconds: float) -> None:
        self._threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None and (time.monotonic() - self._opened_at) < self._reset_seconds

    def allow(self) -> bool:
        return not self.is_open

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()


class ResilientAdapter:
    """Wraps any LLMAdapter with retry/backoff, deadlines, and a circuit breaker."""

    def __init__(self, inner: LLMAdapter, config: ProviderResilienceConfig | None = None) -> None:
        self._inner = inner
        self._config = config or ProviderResilienceConfig()
        self._breaker = CircuitBreaker(
            self._config.circuit_failure_threshold,
            self._config.circuit_reset_seconds,
        )

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def inner(self) -> LLMAdapter:
        return self._inner

    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions, **kwargs: object) -> LLMResponse:
        retry = self._config.retry
        attempt = 0
        while True:
            if not self._breaker.allow():
                raise ProviderUnavailableError(self.name, "circuit open after repeated failures")
            attempt += 1
            try:
                response = await asyncio.wait_for(
                    self._inner.chat(messages, options, **kwargs),
                    timeout=retry.call_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._breaker.record_failure()
                if not is_retryable(e):
                    raise
                if attempt >= retry.max_attempts:
                    raise ProviderUnavailableError(self.name, f"retries exhausted after {attempt} attempts: {e}") from e
                delay = retry_after_seconds(e) if retry.respect_retry_after else None
                if delay is None:
                    delay = _backoff_delay(attempt, retry)
                await asyncio.sleep(delay)
                continue
            self._breaker.record_success()
            return response

    def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterable[StreamEvent]:
        async def _gen() -> AsyncIterator[StreamEvent]:
            retry = self._config.retry
            attempt = 0
            while True:
                if not self._breaker.allow():
                    raise ProviderUnavailableError(self.name, "circuit open after repeated failures")
                attempt += 1
                emitted = False
                pre_output_error: str | None = None
                try:
                    async for event in self._inner.stream(messages, options):
                        # Adapters swallow exceptions into `error` events. An error
                        # before any output is safely retryable; after output has
                        # been forwarded the turn cannot be re-issued transparently.
                        if event.type == "error" and not emitted:
                            pre_output_error = str(event.data)
                            break
                        emitted = True
                        yield event
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if emitted or not is_retryable(e):
                        self._breaker.record_failure()
                        raise
                    pre_output_error = str(e)
                if pre_output_error is None:
                    self._breaker.record_success()
                    return
                self._breaker.record_failure()
                if attempt >= retry.max_attempts:
                    yield StreamEvent(
                        type="error",
                        data=f"Provider '{self.name}' unavailable after {attempt} attempts: {pre_output_error}",
                    )
                    return
                await asyncio.sleep(_backoff_delay(attempt, retry))

        return _gen()
