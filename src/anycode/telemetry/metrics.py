"""Counters, histograms, and gauges for token usage, cost, and latency."""

from __future__ import annotations

import time
from typing import Any

METRIC_TOKENS_INPUT = "anycode.tokens.input"
METRIC_TOKENS_OUTPUT = "anycode.tokens.output"
METRIC_TOKENS_TOTAL = "anycode.tokens.total"
METRIC_COST_USD = "anycode.cost.usd"
METRIC_LATENCY_MS = "anycode.latency.ms"
METRIC_ERRORS = "anycode.errors"

_MS_PER_SECOND = 1000


class MetricsCollector:
    """Collects and aggregates telemetry metrics (counters, histograms)."""

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if not self._enabled:
            return
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + value

    def record(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        if not self._enabled:
            return
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_histogram(self, name: str, labels: dict[str, str] | None = None) -> list[float]:
        key = self._make_key(name, labels)
        return list(self._histograms.get(key, []))

    def record_token_usage(self, agent_name: str, model: str, input_tokens: int, output_tokens: int) -> None:
        if not self._enabled:
            return
        labels = {"agent": agent_name, "model": model}
        self.increment(METRIC_TOKENS_INPUT, float(input_tokens), labels)
        self.increment(METRIC_TOKENS_OUTPUT, float(output_tokens), labels)
        self.increment(METRIC_TOKENS_TOTAL, float(input_tokens + output_tokens), labels)

    def record_cost(self, agent_name: str, model: str, cost_usd: float) -> None:
        if not self._enabled:
            return
        labels = {"agent": agent_name, "model": model}
        self.increment(METRIC_COST_USD, cost_usd, labels)

    def record_latency(self, operation: str, duration_ms: float, labels: dict[str, str] | None = None) -> None:
        if not self._enabled:
            return
        merged = {"operation": operation, **(labels or {})}
        self.record(METRIC_LATENCY_MS, duration_ms, merged)

    def record_error(self, operation: str, error_type: str, labels: dict[str, str] | None = None) -> None:
        if not self._enabled:
            return
        merged = {"operation": operation, "error_type": error_type, **(labels or {})}
        self.increment(METRIC_ERRORS, 1.0, merged)

    def get_summary(self) -> dict[str, Any]:
        return {"counters": dict(self._counters), "histograms": {k: len(v) for k, v in self._histograms.items()}}

    def reset(self) -> None:
        self._counters.clear()
        self._histograms.clear()

    @staticmethod
    def _make_key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        parts = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{parts}}}"


class Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, labels: dict[str, str] | None = None) -> None:
        self._collector = collector
        self._name = name
        self._labels = labels
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        duration_ms = (time.monotonic() - self._start) * _MS_PER_SECOND
        self._collector.record_latency(self._name, duration_ms, self._labels)
