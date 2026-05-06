# Demo 12 — Intelligent Task Routing
# Execute: uv run python examples/12_intelligent_routing.py
#
# Demonstrates:
#   1. Heuristic task classification (5 complexity levels)
#   2. Declarative routing rules (complexity, keyword, regex)
#   3. DefaultRouter with fallback to default model
#   4. Route decisions with full audit trail
#
# No API keys required — routing is a zero-cost heuristic.

import asyncio
from datetime import UTC, datetime

from anycode.routing.classifier import classify_task
from anycode.routing.router import DefaultRouter
from anycode.routing.rules import evaluate_rules
from anycode.types import AgentConfig, RoutingConfig, RoutingRule, Task


def _task(title: str, description: str, depends_on: list[str] | None = None) -> Task:
    return Task(
        id=f"task-{title.lower().replace(' ', '-')}",
        title=title,
        description=description,
        depends_on=depends_on,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


async def main() -> None:
    print("=== Intelligent Routing Demo ===\n")

    # --- Section A: Task classification ---
    print("Section A: Heuristic Task Classification")
    tasks = [
        _task("Lint check", "Run linter"),
        _task("Unit tests", "Run the test suite for auth module", ["t1"]),
        _task("API design", "x" * 400, ["t1", "t2", "t3"]),
        _task("Full refactor", "x" * 900),
        _task("System redesign", "x" * 1200),
    ]

    for t in tasks:
        level = classify_task(t)
        dep_count = len(t.depends_on or [])
        print(f"  {t.title:20s} desc_len={len(t.description):4d} deps={dep_count} -> {level}")

    # --- Section B: Declarative rule matching ---
    print("\nSection B: Rule Matching")
    rules = [
        RoutingRule(condition="complexity == 'trivial'", target_model="haiku", priority=1),
        RoutingRule(condition="complexity == 'expert'", target_model="opus", priority=2),
        RoutingRule(condition="'test' in task.title.lower()", target_model="sonnet", priority=10),
        RoutingRule(condition="re:deploy|release|production", target_model="opus", target_provider="anthropic", priority=20),
    ]

    test_cases = [
        (_task("Lint check", "short"), "trivial"),
        (_task("Unit tests", "Run tests"), "simple"),
        (_task("Deploy API", "Deploy to production"), "moderate"),
        (_task("Big task", "x" * 1500), "expert"),
    ]

    for task, complexity in test_cases:
        matched = evaluate_rules(task, complexity, rules)
        if matched:
            print(f"  {task.title:20s} complexity={complexity:10s} -> model={matched.target_model} (rule: {matched.condition})")
        else:
            print(f"  {task.title:20s} complexity={complexity:10s} -> no rule matched")

    # --- Section C: DefaultRouter ---
    print("\nSection C: DefaultRouter (end-to-end)")
    config = RoutingConfig(
        enabled=True,
        rules=[
            RoutingRule(condition="complexity == 'trivial'", target_model="claude-haiku", target_provider="anthropic", priority=1),
            RoutingRule(condition="complexity == 'expert'", target_model="claude-opus", target_provider="anthropic", priority=2),
            RoutingRule(condition="'test' in task.title.lower()", target_model="gpt-4o-mini", target_provider="openai", priority=10),
        ],
        default_model="claude-sonnet",
        default_provider="anthropic",
    )
    router = DefaultRouter(config)
    agents = [AgentConfig(name="worker", model="claude-sonnet-4-20250514")]

    for t in tasks:
        decision = await router.route(t, agents)
        if decision:
            print(f"  {t.title:20s} -> {decision.routed_model:15s} provider={decision.routed_provider or 'default':10s} ({decision.reason})")
        else:
            print(f"  {t.title:20s} -> no routing decision")

    # --- Section D: Disabled routing ---
    print("\nSection D: Disabled Routing (pass-through)")
    disabled = RoutingConfig(enabled=False)
    disabled_router = DefaultRouter(disabled)
    result = await disabled_router.route(tasks[0], agents)
    print(f"  result: {result}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
