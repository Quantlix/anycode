"""Render execution timelines from a TeamRunResult."""

from __future__ import annotations

from anycode.types import TeamRunResult


def render_timeline(result: TeamRunResult, *, width: int = 40) -> str:
    """Render a simple ASCII timeline of agent durations.

    Note: AgentRunResult does not currently carry timestamps, so this renders
    relative call counts and token usage as a rough activity proxy.
    """
    if not result.agent_results:
        return "<empty timeline>"

    rows: list[str] = []
    max_tokens = (
        max(
            (r.token_usage.input_tokens + r.token_usage.output_tokens for r in result.agent_results.values()),
            default=1,
        )
        or 1
    )

    for agent, run in result.agent_results.items():
        total = run.token_usage.input_tokens + run.token_usage.output_tokens
        bar_width = max(1, int((total / max_tokens) * width))
        bar = "\u2588" * bar_width + "\u2591" * (width - bar_width)
        status = "OK " if run.success else "FAIL"
        rows.append(f"{agent:20s} [{status}] {bar} tokens={total}")

    return "\n".join(rows)
