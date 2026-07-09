"""Build CostReport instances from a CostTracker."""

from __future__ import annotations

from anycode.cost.tracker import CostTracker
from anycode.types import CostBreakdown, CostReport


def build_cost_report(tracker: CostTracker) -> CostReport:
    """Render a CostReport from the current tracker state."""
    by_agent = [
        CostBreakdown(
            agent=a.agent,
            model=a.model,
            input_tokens=a.input_tokens,
            output_tokens=a.output_tokens,
            cache_creation_input_tokens=a.cache_creation_input_tokens,
            cache_read_input_tokens=a.cache_read_input_tokens,
            input_cost_usd=a.input_cost_usd,
            output_cost_usd=a.output_cost_usd,
            cache_read_cost_usd=a.cache_read_cost_usd,
            total_cost_usd=a.input_cost_usd + a.output_cost_usd,
            calls=a.calls,
        )
        for a in tracker.by_agent()
    ]
    by_model = [
        CostBreakdown(
            agent="",
            model=m.model,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
            cache_creation_input_tokens=m.cache_creation_input_tokens,
            cache_read_input_tokens=m.cache_read_input_tokens,
            input_cost_usd=m.input_cost_usd,
            output_cost_usd=m.output_cost_usd,
            cache_read_cost_usd=m.cache_read_cost_usd,
            total_cost_usd=m.input_cost_usd + m.output_cost_usd,
            calls=m.calls,
        )
        for m in tracker.by_model()
    ]

    budget = tracker.config.budget_usd if tracker.config else None
    remaining = (budget - tracker.total_cost_usd) if budget is not None else None

    return CostReport(
        total_cost_usd=tracker.total_cost_usd,
        total_input_tokens=tracker.total_input_tokens,
        total_output_tokens=tracker.total_output_tokens,
        total_cache_creation_input_tokens=tracker.total_cache_creation_input_tokens,
        total_cache_read_input_tokens=tracker.total_cache_read_input_tokens,
        by_agent=by_agent,
        by_model=by_model,
        budget_usd=budget,
        budget_remaining_usd=remaining,
    )
