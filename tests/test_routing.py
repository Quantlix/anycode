"""Tests for intelligent routing (classifier, rules, router)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from anycode.routing.classifier import classify_task
from anycode.routing.router import DefaultRouter
from anycode.routing.rules import evaluate_rules, match_rule
from anycode.types import AgentConfig, RoutingConfig, RoutingRule, Task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    description: str = "short desc",
    title: str = "task",
    depends_on: list[str] | None = None,
) -> Task:
    return Task(
        id="t1",
        title=title,
        description=description,
        depends_on=depends_on,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_agent(
    name: str = "agent1",
    model: str = "claude-sonnet-4-20250514",
) -> AgentConfig:
    return AgentConfig(name=name, model=model)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifyTask:
    def test_trivial(self) -> None:
        task = _make_task(description="x" * 50, depends_on=[])
        assert classify_task(task) == "trivial"

    def test_simple(self) -> None:
        task = _make_task(description="x" * 150, depends_on=["a"])
        assert classify_task(task) == "simple"

    def test_moderate(self) -> None:
        task = _make_task(description="x" * 350, depends_on=["a", "b", "c"])
        assert classify_task(task) == "moderate"

    def test_complex(self) -> None:
        task = _make_task(description="x" * 800, depends_on=[])
        assert classify_task(task) == "complex"

    def test_expert(self) -> None:
        task = _make_task(description="x" * 1500, depends_on=[])
        assert classify_task(task) == "expert"

    def test_none_depends_treated_as_empty(self) -> None:
        task = _make_task(description="x" * 50, depends_on=None)
        assert classify_task(task) == "trivial"

    def test_boundary_trivial_to_simple(self) -> None:
        # ROUTING_TRIVIAL_MAX_LEN = 100
        task = _make_task(description="x" * 100, depends_on=[])
        result = classify_task(task)
        assert result in ("trivial", "simple")


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------


class TestMatchRule:
    def test_complexity_equality_match(self) -> None:
        task = _make_task(description="x" * 50)
        rule = RoutingRule(condition="complexity == 'trivial'", target_model="haiku")
        assert match_rule(task, "trivial", rule) is True

    def test_complexity_equality_no_match(self) -> None:
        task = _make_task()
        rule = RoutingRule(condition="complexity == 'expert'", target_model="opus")
        assert match_rule(task, "trivial", rule) is False

    def test_keyword_in_title(self) -> None:
        task = _make_task(title="Run unit tests")
        rule = RoutingRule(condition="'test' in task.title.lower()", target_model="haiku")
        assert match_rule(task, "simple", rule) is True

    def test_keyword_in_description(self) -> None:
        task = _make_task(title="Task", description="implement comprehensive testing strategy")
        rule = RoutingRule(condition="'testing' in task.description.lower()", target_model="haiku")
        assert match_rule(task, "simple", rule) is True

    def test_keyword_no_match(self) -> None:
        task = _make_task(title="Write code")
        rule = RoutingRule(condition="'test' in task.title.lower()", target_model="haiku")
        assert match_rule(task, "simple", rule) is False

    def test_regex_match(self) -> None:
        task = _make_task(title="Deploy to production", description="release version 2.0")
        rule = RoutingRule(condition="re:deploy|release", target_model="opus")
        assert match_rule(task, "complex", rule) is True

    def test_regex_no_match(self) -> None:
        task = _make_task(title="Write docs")
        rule = RoutingRule(condition="re:deploy|release", target_model="opus")
        assert match_rule(task, "complex", rule) is False

    def test_unknown_condition_returns_false(self) -> None:
        task = _make_task()
        rule = RoutingRule(condition="garbage condition", target_model="haiku")
        assert match_rule(task, "trivial", rule) is False


# ---------------------------------------------------------------------------
# Rule evaluation (priority-ordered)
# ---------------------------------------------------------------------------


class TestEvaluateRules:
    def test_highest_priority_wins(self) -> None:
        task = _make_task(description="x" * 50)
        rules = [
            RoutingRule(condition="complexity == 'trivial'", target_model="haiku", priority=1),
            RoutingRule(condition="complexity == 'trivial'", target_model="sonnet", priority=10),
        ]
        result = evaluate_rules(task, "trivial", rules)
        assert result is not None
        assert result.target_model == "sonnet"

    def test_returns_none_when_no_match(self) -> None:
        task = _make_task()
        rules = [
            RoutingRule(condition="complexity == 'expert'", target_model="opus", priority=1),
        ]
        result = evaluate_rules(task, "trivial", rules)
        assert result is None

    def test_empty_rules_returns_none(self) -> None:
        task = _make_task()
        result = evaluate_rules(task, "trivial", [])
        assert result is None

    def test_stops_at_first_match(self) -> None:
        task = _make_task(title="Run tests")
        rules = [
            RoutingRule(condition="complexity == 'trivial'", target_model="haiku", priority=5),
            RoutingRule(condition="'test' in task.title.lower()", target_model="sonnet", priority=10),
        ]
        result = evaluate_rules(task, "trivial", rules)
        assert result is not None
        # Priority 10 wins, matches keyword
        assert result.target_model == "sonnet"


# ---------------------------------------------------------------------------
# DefaultRouter
# ---------------------------------------------------------------------------


class TestDefaultRouter:
    async def test_disabled_returns_none(self) -> None:
        config = RoutingConfig(enabled=False)
        router = DefaultRouter(config)
        task = _make_task()
        result = await router.route(task, [_make_agent()])
        assert result is None

    async def test_routes_by_rule(self) -> None:
        config = RoutingConfig(
            enabled=True,
            rules=[RoutingRule(condition="complexity == 'trivial'", target_model="haiku", target_provider="anthropic")],
        )
        router = DefaultRouter(config)
        task = _make_task(description="x" * 50)
        result = await router.route(task, [_make_agent()])

        assert result is not None
        assert result.routed_model == "haiku"
        assert result.routed_provider == "anthropic"
        assert result.complexity == "trivial"
        assert "Matched rule" in result.reason

    async def test_falls_back_to_default_model(self) -> None:
        config = RoutingConfig(
            enabled=True,
            rules=[RoutingRule(condition="complexity == 'expert'", target_model="opus")],
            default_model="sonnet",
            default_provider="anthropic",
        )
        router = DefaultRouter(config)
        task = _make_task(description="x" * 50)  # trivial, won't match expert rule
        result = await router.route(task, [_make_agent()])

        assert result is not None
        assert result.routed_model == "sonnet"
        assert "Default route" in result.reason

    async def test_no_rules_no_default_returns_none(self) -> None:
        config = RoutingConfig(enabled=True)
        router = DefaultRouter(config)
        task = _make_task()
        result = await router.route(task, [_make_agent()])
        assert result is None

    async def test_route_decision_has_task_id(self) -> None:
        config = RoutingConfig(enabled=True, default_model="haiku")
        router = DefaultRouter(config)
        task = _make_task()
        result = await router.route(task, [_make_agent()])
        assert result is not None
        assert result.task_id == "t1"

    async def test_route_decision_captures_original_model(self) -> None:
        config = RoutingConfig(enabled=True, default_model="haiku")
        router = DefaultRouter(config)
        task = _make_task()
        agent = _make_agent(model="claude-opus-4-20250514")
        result = await router.route(task, [agent])
        assert result is not None
        assert result.original_model == "claude-opus-4-20250514"

    async def test_route_decision_is_frozen(self) -> None:
        config = RoutingConfig(enabled=True, default_model="haiku")
        router = DefaultRouter(config)
        task = _make_task()
        result = await router.route(task, [_make_agent()])
        assert result is not None
        with pytest.raises(Exception):
            result.task_id = "other"
