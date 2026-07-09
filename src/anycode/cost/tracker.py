"""Real-time cost tracking with budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

from anycode.cost.pricing import calculate_cost
from anycode.types import CostConfig, ModelPricing, TokenUsage


@dataclass
class _AgentCost:
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0
    calls: int = 0


@dataclass
class CostTracker:
    """Tracks cumulative cost per agent and per model."""

    config: CostConfig | None = None
    pricing: list[ModelPricing] | None = None
    _by_agent: dict[str, _AgentCost] = field(default_factory=dict)
    _by_model: dict[str, _AgentCost] = field(default_factory=dict)
    _alerted: bool = False

    def record(self, agent: str, model: str, usage: TokenUsage) -> float:
        """Record an LLM call and return its USD cost."""
        custom = self.config.custom_pricing if (self.config and self.config.custom_pricing) else None
        pricing = custom or self.pricing
        cost = calculate_cost(usage, model, pricing)

        agent_entry = self._by_agent.setdefault(agent, _AgentCost(agent=agent, model=model))
        agent_entry.model = model
        agent_entry.input_tokens += usage.input_tokens
        agent_entry.output_tokens += usage.output_tokens
        agent_entry.cache_creation_input_tokens += usage.cache_creation_input_tokens
        agent_entry.cache_read_input_tokens += usage.cache_read_input_tokens
        agent_entry.calls += 1

        model_entry = self._by_model.setdefault(model, _AgentCost(agent="", model=model))
        model_entry.input_tokens += usage.input_tokens
        model_entry.output_tokens += usage.output_tokens
        model_entry.cache_creation_input_tokens += usage.cache_creation_input_tokens
        model_entry.cache_read_input_tokens += usage.cache_read_input_tokens
        model_entry.calls += 1

        from anycode.cost.pricing import find_pricing

        price = find_pricing(model, pricing)
        if price is not None:
            cache_read_rate = price.cached_input_cost_per_1k if price.cached_input_cost_per_1k is not None else price.input_cost_per_1k
            fresh_input = usage.input_tokens + usage.cache_creation_input_tokens
            agent_input = (fresh_input / 1000) * price.input_cost_per_1k
            cache_read_cost = (usage.cache_read_input_tokens / 1000) * cache_read_rate
            agent_input += cache_read_cost
            agent_output = (usage.output_tokens / 1000) * price.output_cost_per_1k
            agent_entry.input_cost_usd += agent_input
            agent_entry.output_cost_usd += agent_output
            agent_entry.cache_read_cost_usd += cache_read_cost
            model_entry.input_cost_usd += agent_input
            model_entry.output_cost_usd += agent_output
            model_entry.cache_read_cost_usd += cache_read_cost

        return cost

    @property
    def total_cost_usd(self) -> float:
        return sum(c.input_cost_usd + c.output_cost_usd for c in self._by_agent.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self._by_agent.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self._by_agent.values())

    @property
    def total_cache_creation_input_tokens(self) -> int:
        return sum(c.cache_creation_input_tokens for c in self._by_agent.values())

    @property
    def total_cache_read_input_tokens(self) -> int:
        return sum(c.cache_read_input_tokens for c in self._by_agent.values())

    def is_budget_exhausted(self) -> bool:
        if self.config is None or self.config.budget_usd is None:
            return False
        return self.total_cost_usd >= self.config.budget_usd

    def is_budget_alert_due(self) -> bool:
        if self._alerted or self.config is None or self.config.budget_usd is None:
            return False
        if self.total_cost_usd >= self.config.budget_usd * self.config.alert_threshold:
            self._alerted = True
            return True
        return False

    def by_agent(self) -> list[_AgentCost]:
        return list(self._by_agent.values())

    def by_model(self) -> list[_AgentCost]:
        return list(self._by_model.values())
