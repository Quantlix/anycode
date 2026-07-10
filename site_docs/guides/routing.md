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

## Next steps

- [Build a support-triage system](../tutorials/support-triage.md) — routing incoming tickets to specialist agents.
- [Hand off between agents](handoff.md) — dynamic delegation as an alternative to static routing.
- [Track and cap cost](cost-tracking.md) — measure the savings routing produces.
- [Configuration reference](../reference/configuration.md) — every `RoutingConfig` and `RoutingRule` field.
