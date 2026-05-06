# Demo 13 — Cost Tracking & Budget Enforcement (Phase 5.2)
# Execute: uv run python examples/13_cost_tracking.py
#
# Demonstrates:
#   1. Configuring a USD budget with alert threshold
#   2. Per-agent and per-model cost breakdown
#   3. CostReport returned on TeamRunResult
#
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.

import asyncio
import os
import sys

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, CostConfig, OrchestratorConfig, TaskSpec, TeamConfig

load_dotenv()


def _resolve_provider() -> tuple[str, str] | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None


async def main() -> None:
    resolved = _resolve_provider()
    if resolved is None:
        print("ERROR: set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.")
        sys.exit(1)
    provider, model = resolved

    config = OrchestratorConfig(cost=CostConfig(budget_usd=0.50, alert_threshold=0.5, on_budget_exceeded="warn"))
    engine = AnyCode(config)

    team = engine.create_team(
        "writers",
        TeamConfig(
            name="writers",
            agents=[
                AgentConfig(
                    name="haiku",
                    model=model,
                    provider=provider,
                    system_prompt="Write a single-line haiku in response to the prompt.",
                    tools=[],
                ),
                AgentConfig(
                    name="critic",
                    model=model,
                    provider=provider,
                    system_prompt="Give a short two-sentence critique.",
                    tools=[],
                ),
            ],
        ),
    )

    tasks = [
        TaskSpec(title="haiku", description="Write a haiku about distributed systems.", assignee="haiku"),
        TaskSpec(title="critique", description="Critique the previous haiku.", assignee="critic", depends_on=["haiku"]),
    ]

    result = await engine.run_tasks(team, tasks)

    print(f"\nsuccess={result.success}")
    if result.cost_report:
        print(f"\n=== Cost Report ===")
        print(f"Total: ${result.cost_report.total_cost_usd:.6f}")
        print(f"Tokens: in={result.cost_report.total_input_tokens} out={result.cost_report.total_output_tokens}")
        for b in result.cost_report.by_agent:
            print(f"  {b.agent} ({b.model}): ${b.input_cost_usd + b.output_cost_usd:.6f} over {b.calls} call(s)")


if __name__ == "__main__":
    asyncio.run(main())
