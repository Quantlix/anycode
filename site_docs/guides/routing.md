---
title: "Route AnyCode Tasks to the Right Model by Complexity"
description: "Use AnyCode's DefaultRouter and RoutingConfig to classify task complexity and route each task to a cheaper or stronger model with priority-ordered rules."
keywords: anycode routing, DefaultRouter, RoutingConfig, RoutingRule, classify_task, task complexity, model routing, cost optimization, route decision
---

# Route Tasks by Complexity

Not every task needs your most expensive model. AnyCode can classify each task's complexity and route it to a cheaper or stronger model automatically, using rules you control. This guide covers the heuristic classifier, the three rule syntaxes, and how to wire routing into an engine.

## How classification works

`classify_task` scores a task into one of five levels — `trivial`, `simple`, `moderate`, `complex`, `expert` — from its description length and dependency count. It's a fast, deterministic heuristic with no LLM call.

```python title="classify.py"
from anycode import classify_task

level = classify_task(task)   # "trivial" .. "expert"
```

| Level | Roughly when |
| --- | --- |
| `trivial` | Very short description, no dependencies |
| `simple` | Short description, at most one dependency |
| `moderate` | Medium description, a few dependencies |
| `complex` | Long description |
| `expert` | Very long description |

## Write routing rules

A `RoutingRule` maps a **condition** to a target model (and optional provider). Rules are evaluated highest-`priority` first; the first match wins. Conditions come in three flavors:

| Condition syntax | Matches when |
| --- | --- |
| `complexity == 'expert'` | The classified level equals the value |
| `'deploy' in task.title.lower()` | A keyword appears in the title or description |
| `re:deploy\|release\|production` | A regex matches the title + description |

```python title="routing.py"
from anycode import DefaultRouter
from anycode.types import RoutingConfig, RoutingRule

config = RoutingConfig(
    enabled=True,
    rules=[
        RoutingRule(condition="complexity == 'trivial'", target_model="claude-haiku-4-5",
                    target_provider="anthropic", priority=1),
        RoutingRule(condition="complexity == 'expert'", target_model="claude-sonnet-5",
                    target_provider="anthropic", priority=2),
        RoutingRule(condition="'test' in task.title.lower()", target_model="gpt-4o-mini",
                    target_provider="openai", priority=10),
    ],
    default_model="claude-haiku-4-5",
    default_provider="anthropic",
)

router = DefaultRouter(config)
decision = await router.route(task, agents)   # RouteDecision | None
```

A `RouteDecision` reports the `routed_model`, `routed_provider`, the classified `complexity`, and a human-readable `reason`.

| `RoutingConfig` field | Default | Effect |
| --- | --- | --- |
| `enabled` | `False` | Routing is off until you set this |
| `rules` | `None` | Priority-ordered match list |
| `default_model` | `None` | Fallback when no rule matches |
| `default_provider` | `None` | Provider for the fallback |

!!! warning "A router can decline to route"
    `route` returns `None` when routing is disabled, or when no rule matches **and** no `default_model` is set — a valid "no decision," not an error. Always provide a `default_model` if you want every task routed. (The `classify_with_llm` field exists but is not currently consumed; classification is always heuristic.)

## Turn on routing for an engine

```python title="engine.py"
from anycode import AnyCode
from anycode.types import RoutingConfig

engine = AnyCode(config={"routing": RoutingConfig(enabled=True, default_model="claude-haiku-4-5")})
```

With routing enabled, the orchestrator classifies each task and applies a matching rule before the agent runs — so a wave of trivial tasks lands on a cheap model while the one expert task escalates to a stronger one.

## Enforce provider policy before fallback

`PolicyRouter` is the hard-constraint routing surface for multi-provider deployments. Each `ProviderCapabilityDescriptor` declares region, modalities, context size, structured output, tool use, classifications, health, cost, latency, and a fallback compatibility class. A request is filtered before selection, and every candidate retains inspectable rejection reasons.

```python
from anycode import ModelRoutingRequest, PolicyRouter, ProviderCapabilityDescriptor

router = PolicyRouter((
    ProviderCapabilityDescriptor(
        provider="private-provider",
        model="review-model",
        context_window=128_000,
        structured_output=True,
        tool_use=True,
        regions=("eu-west",),
        allowed_classifications=("public", "internal", "confidential"),
        compatibility_class="review-v1",
    ),
))

decision = router.route(ModelRoutingRequest(
    task_id="review-1",
    classification="confidential",
    required_region="eu-west",
    structured_output=True,
))
```

Fallback is not a relaxation path. Setting `fallback_compatibility_class` adds another hard filter; provider allow/deny lists, region, classification, budget, latency, health, modality, context, and capability restrictions continue to apply. A request with no eligible candidate returns a typed `no_eligible_model` error and the complete assessment set.

For descriptor design, cost estimation, every rejection reason, and compatible fallback handling, see [Route models with hard policy constraints](policy-routing.md).

## The complete, runnable program

The snippets above are fragments of one file. Here is the whole thing, ready to copy into `routing.py` and run. Routing is a zero-cost heuristic, so this program needs no API key. It classifies a wave of tasks, matches a rule directly, then drives every task through a `DefaultRouter` with a default fallback.

```python title="routing.py"
import asyncio
from datetime import UTC, datetime

from anycode import AgentConfig, DefaultRouter, Task, classify_task, evaluate_rules
from anycode.types import RoutingConfig, RoutingRule


def make_task(title: str, description: str, depends_on: list[str] | None = None) -> Task:
    now = datetime.now(UTC)
    return Task(
        id=f"task-{title.lower().replace(' ', '-')}",
        title=title,
        description=description,
        depends_on=depends_on,
        created_at=now,
        updated_at=now,
    )


async def main() -> None:
    tasks = [
        make_task("Lint check", "Run linter"),
        make_task("Unit tests", "Run the test suite for the auth module", ["t1"]),
        make_task("API design", "x" * 400, ["t1", "t2", "t3"]),
        make_task("System redesign", "x" * 1200),
    ]

    # 1. Heuristic classification — deterministic, no LLM call.
    print("Classification:")
    for task in tasks:
        print(f"  {task.title:16s} -> {classify_task(task)}")

    # 2. Match rules directly against a known complexity level.
    print("\nRule matching:")
    rules = [
        RoutingRule(condition="complexity == 'trivial'", target_model="claude-haiku-4-5", priority=1),
        RoutingRule(condition="complexity == 'expert'", target_model="claude-sonnet-5", priority=2),
        RoutingRule(condition="'test' in task.title.lower()", target_model="gpt-4o-mini", priority=10),
    ]
    matched = evaluate_rules(make_task("Unit tests", "Run tests"), "simple", rules)
    print(f"  matched: {matched.condition if matched else None} -> {matched.target_model if matched else None}")

    # 3. End-to-end routing through DefaultRouter, with a default fallback.
    config = RoutingConfig(
        enabled=True,
        rules=[
            RoutingRule(condition="complexity == 'trivial'", target_model="claude-haiku-4-5",
                        target_provider="anthropic", priority=1),
            RoutingRule(condition="complexity == 'expert'", target_model="claude-sonnet-5",
                        target_provider="anthropic", priority=2),
            RoutingRule(condition="'test' in task.title.lower()", target_model="gpt-4o-mini",
                        target_provider="openai", priority=10),
        ],
        default_model="claude-haiku-4-5",
        default_provider="anthropic",
    )
    router = DefaultRouter(config)
    agents = [AgentConfig(name="worker", model="claude-haiku-4-5")]

    print("\nDefaultRouter decisions:")
    for task in tasks:
        decision = await router.route(task, agents)
        if decision:
            print(f"  {task.title:16s} -> {decision.routed_model:16s} ({decision.reason})")
        else:
            print(f"  {task.title:16s} -> no routing decision")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python routing.py
```

!!! tip "Tested copy"
    See [`examples/12_intelligent_routing.py`](https://github.com/Quantlix/anycode/blob/main/examples/12_intelligent_routing.py) for the CI-tested version, which adds a disabled-routing pass-through and prints a full audit trail.

## Next steps

- [Route models with hard policy constraints](policy-routing.md) - enforce region, classification, capability, budget, and latency requirements.

- [Build a support-triage system](../tutorials/support-triage.md) — routing incoming tickets to specialist agents.
- [Hand off between agents](handoff.md) — dynamic delegation as an alternative to static routing.
- [Track and cap cost](cost-tracking.md) — measure the savings routing produces.
- [Configuration reference](../reference/configuration.md) — every `RoutingConfig` and `RoutingRule` field.
