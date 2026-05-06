"""Tests for cost tracking (Phase 5.2)."""

from __future__ import annotations

import pytest

from anycode import (
    DEFAULT_PRICING,
    CostConfig,
    CostTracker,
    ModelPricing,
    TokenUsage,
    build_cost_report,
    calculate_cost,
    find_pricing,
)


def test_find_pricing_exact_match() -> None:
    pricing = find_pricing("claude-haiku-4-5", DEFAULT_PRICING)
    assert pricing is not None
    assert pricing.model == "claude-haiku-4-5"


def test_find_pricing_wildcard_fallback() -> None:
    custom = [ModelPricing(model="*ollama*", provider="ollama", input_cost_per_1k=0.0, output_cost_per_1k=0.0)]
    assert find_pricing("any-ollama-model", custom) is not None


def test_find_pricing_unknown_returns_none() -> None:
    assert find_pricing("nonexistent-model-xyz", DEFAULT_PRICING) is None


def test_calculate_cost_basic() -> None:
    pricing = [ModelPricing(model="m", provider="x", input_cost_per_1k=0.001, output_cost_per_1k=0.002)]
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    cost = calculate_cost(usage, "m", pricing)
    assert cost == pytest.approx(0.001 + 0.001)


def test_calculate_cost_unknown_model_returns_zero() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    assert calculate_cost(usage, "missing", []) == 0.0


def test_cost_tracker_records_and_aggregates() -> None:
    config = CostConfig(budget_usd=1.0, on_budget_exceeded="warn")
    tracker = CostTracker(config=config)
    usage = TokenUsage(input_tokens=1000, output_tokens=1000)
    cost = tracker.record("agent-a", "claude-haiku-4-5", usage)
    assert cost > 0
    assert tracker.total_cost_usd == pytest.approx(cost)
    assert tracker.total_input_tokens == 1000
    assert tracker.total_output_tokens == 1000
    by_agent = tracker.by_agent()
    assert any(b.agent == "agent-a" for b in by_agent)
    by_model = tracker.by_model()
    assert any(b.model == "claude-haiku-4-5" for b in by_model)


def test_cost_tracker_budget_exhaustion() -> None:
    tracker = CostTracker(config=CostConfig(budget_usd=0.0001))
    big = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    tracker.record("a", "claude-haiku-4-5", big)
    assert tracker.is_budget_exhausted() is True


def test_cost_tracker_alert_one_shot() -> None:
    tracker = CostTracker(config=CostConfig(budget_usd=0.0001, alert_threshold=0.5))
    tracker.record("a", "claude-haiku-4-5", TokenUsage(input_tokens=100_000, output_tokens=0))
    assert tracker.is_budget_alert_due() is True
    assert tracker.is_budget_alert_due() is False


def test_build_cost_report() -> None:
    tracker = CostTracker(config=CostConfig())
    tracker.record("alpha", "claude-haiku-4-5", TokenUsage(input_tokens=100, output_tokens=100))
    tracker.record("beta", "claude-haiku-4-5", TokenUsage(input_tokens=50, output_tokens=50))
    report = build_cost_report(tracker)
    assert report.total_cost_usd > 0
    agents = {b.agent for b in report.by_agent if b.agent}
    assert agents == {"alpha", "beta"}
