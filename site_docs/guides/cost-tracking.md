---
title: "Track and Cap LLM Spend in AnyCode"
description: "Measure token cost per agent and model with CostTracker and CostReport, set a budget_usd ceiling, and safely stop or warn when an AnyCode run exceeds it."
keywords: anycode cost tracking, CostConfig, CostTracker, CostReport, budget_usd, token cost, LLM spend, calculate_cost, DEFAULT_PRICING, cost budget
---

# Track and Cap Cost

Agent runs cost money, and a loop that misbehaves can cost a lot of it quickly. AnyCode tracks token spend per agent and per model, and can stop a run the moment it crosses a budget. This guide shows how to enable cost tracking, read a report, and enforce a ceiling.

## Enable tracking on the engine

Cost tracking is automatic at the orchestrator level: attach a `CostConfig` and every team run records what each agent spent, attaching a `CostReport` to the result.

```python title="cost.py"
from anycode import AnyCode, CostConfig

engine = AnyCode(config={
    "cost": CostConfig(budget_usd=0.50, alert_threshold=0.8, on_budget_exceeded="warn"),
})

result = await engine.run_tasks(team, tasks)

if result.cost_report:
    report = result.cost_report
    print(f"total: ${report.total_cost_usd:.4f}")
    print(f"tokens in/out: {report.total_input_tokens}/{report.total_output_tokens}")
    for row in report.by_agent:
        print(f"  {row.agent} [{row.model}] ${row.total_cost_usd:.4f} over {row.calls} calls")
```

| `CostConfig` field | Default | Effect |
| --- | --- | --- |
| `enabled` | `True` | Master switch |
| `budget_usd` | `None` | Spend ceiling; `None` means no limit |
| `alert_threshold` | `0.8` | Fraction of budget that triggers a one-time alert |
| `on_budget_exceeded` | `"stop"` | `stop`, `warn`, or `continue` when the budget is hit |
| `custom_pricing` | `None` | Override the built-in price table |

## Enforce a budget

Set `budget_usd` and choose the enforcement behavior. With `on_budget_exceeded="stop"`, the run halts the task that pushes spend over the ceiling; `"warn"` emits an event and continues; `"continue"` just records.

!!! tip "Test the ceiling for free"
    Point the same team at a `FakeAdapter` to exercise the budget-stop path deterministically before you spend a cent against a live provider. Unknown models (including the fake one) have no price entry, so `calculate_cost` returns `0.0` for them.

## Where prices come from

AnyCode ships a `DEFAULT_PRICING` table (USD per 1,000 tokens) covering current Anthropic, OpenAI, and Google models. Override or extend it with `custom_pricing` when you run a model the table doesn't know, or negotiated rates:

```python title="custom_pricing.py"
from anycode import CostConfig
from anycode.types import ModelPricing

config = CostConfig(
    budget_usd=5.0,
    custom_pricing=[
        ModelPricing(model="my-model", provider="myvendor",
                     input_cost_per_1k=0.001, output_cost_per_1k=0.004),
    ],
)
```

## Track cost outside the orchestrator

For a standalone script, drive `CostTracker` yourself — the orchestrator's automatic tracking does **not** apply to a bare `Agent.run()`.

```python title="manual_cost.py"
from anycode import CostTracker, build_cost_report, CostConfig

tracker = CostTracker(config=CostConfig(budget_usd=1.0))
tracker.record("worker", "claude-sonnet-5", usage)   # returns the USD cost of this call
if tracker.is_budget_exhausted():
    ...
report = build_cost_report(tracker)
```

!!! warning "Two pricing tables exist"
    The cost engine (`CostConfig` / `CostTracker`) and the guardrail `BudgetTracker` use *separate* price tables that can disagree, and `calculate_cost` silently returns `0.0` for a model it doesn't recognize. Treat cost numbers as close estimates, and add `custom_pricing` for any model you rely on.

## The complete, runnable program

The cost engine is pure arithmetic over token counts, so you can exercise the whole thing without spending a cent or setting an API key. This one file records a few calls against a budget, stops the moment the ceiling is crossed, renders a `CostReport`, and shows `calculate_cost` both estimating ahead of a run and returning `0.0` for an unknown model until you supply `custom_pricing`.

```python title="cost_math.py"
from anycode import CostConfig, CostTracker, build_cost_report, calculate_cost
from anycode.types import ModelPricing, TokenUsage


def main() -> None:
    # A budget-aware tracker. record() returns each call's USD cost and
    # accumulates spend per agent and per model.
    tracker = CostTracker(config=CostConfig(budget_usd=0.10))

    calls = [
        ("planner", "claude-haiku-4-5", TokenUsage(input_tokens=1_200, output_tokens=300)),
        ("builder", "claude-haiku-4-5", TokenUsage(input_tokens=8_000, output_tokens=2_500)),
        ("reviewer", "claude-sonnet-4-5", TokenUsage(input_tokens=40_000, output_tokens=9_000)),
    ]
    for agent, model, usage in calls:
        cost = tracker.record(agent, model, usage)
        print(f"{agent:9s} {model:20s} ${cost:.6f}")
        if tracker.is_budget_exhausted():
            print(f"  budget of ${tracker.config.budget_usd:.2f} exhausted — stop the run here")
            break

    report = build_cost_report(tracker)
    print("\n=== Cost report ===")
    print(f"total: ${report.total_cost_usd:.6f}")
    print(f"tokens in/out: {report.total_input_tokens}/{report.total_output_tokens}")
    for row in report.by_agent:
        print(f"  {row.agent} [{row.model}] ${row.total_cost_usd:.6f} over {row.calls} call(s)")

    # calculate_cost is a pure function — handy for estimating before you run.
    estimate = calculate_cost(TokenUsage(input_tokens=100_000, output_tokens=20_000), "claude-haiku-4-5")
    print(f"\nestimate for 100k in / 20k out on claude-haiku-4-5: ${estimate:.4f}")

    # An unknown model bills as $0.00 until you supply custom_pricing.
    unknown = TokenUsage(input_tokens=1_000, output_tokens=1_000)
    print(f"unknown model, default table: ${calculate_cost(unknown, 'my-model'):.4f}")
    priced = calculate_cost(
        unknown,
        "my-model",
        [ModelPricing(model="my-model", provider="myvendor", input_cost_per_1k=0.001, output_cost_per_1k=0.004)],
    )
    print(f"unknown model, custom pricing: ${priced:.4f}")


if __name__ == "__main__":
    main()
```

Run it from the project root:

```bash
uv run python cost_math.py
```

!!! tip "Tested copy"
    See [`examples/13_cost_tracking.py`](https://github.com/Quantlix/anycode/blob/main/examples/13_cost_tracking.py) for the CI-tested version, which attaches the same `CostConfig` to a live two-agent team and reads the `CostReport` off the `TeamRunResult`.

## Next steps

- [Route tasks by complexity](routing.md) — send cheap tasks to cheap models and measure the savings here.
- [Build a resumable pipeline](../tutorials/resumable-pipeline.md) — cost tracking across a long, durable run.
- [Production controls](production-controls.md) — budgets alongside approval and verification gates.
- [Configuration reference](../reference/configuration.md) — every `CostConfig` and `ModelPricing` field.
