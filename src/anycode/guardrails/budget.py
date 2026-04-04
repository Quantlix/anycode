"""Token + cost budget tracking and enforcement."""

from __future__ import annotations

from anycode.types import BudgetStatus, GuardrailConfig, TokenUsage

DEFAULT_COST_PER_1M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

_FALLBACK_COST = {"input": 5.0, "output": 15.0}
_TOKENS_PER_PRICING_UNIT = 1_000_000


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given model and token count."""
    rates = DEFAULT_COST_PER_1M.get(model, _FALLBACK_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / _TOKENS_PER_PRICING_UNIT


class BudgetTracker:
    """Tracks cumulative token usage, cost, turns, and tool calls against configured limits."""

    def __init__(self, config: GuardrailConfig | None = None, model: str = "") -> None:
        self._config = config
        self._model = model
        self._tokens_used: int = 0
        self._cost_used: float = 0.0
        self._turns_used: int = 0
        self._tool_calls_used: int = 0

    @property
    def config(self) -> GuardrailConfig | None:
        return self._config

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def cost_used(self) -> float:
        return self._cost_used

    @property
    def turns_used(self) -> int:
        return self._turns_used

    @property
    def tool_calls_used(self) -> int:
        return self._tool_calls_used

    def record_usage(self, usage: TokenUsage) -> None:
        """Record token usage from a single LLM call."""
        total_tokens = usage.input_tokens + usage.output_tokens
        self._tokens_used += total_tokens
        self._cost_used += estimate_cost(self._model, usage.input_tokens, usage.output_tokens)

    def record_turn(self) -> None:
        self._turns_used += 1

    def record_tool_call(self, count: int = 1) -> None:
        self._tool_calls_used += count

    def is_exhausted(self) -> bool:
        if self._config is None:
            return False
        if self._config.max_tokens_per_agent is not None and self._tokens_used >= self._config.max_tokens_per_agent:
            return True
        if self._config.max_cost_usd is not None and self._cost_used >= self._config.max_cost_usd:
            return True
        if self._config.max_turns is not None and self._turns_used >= self._config.max_turns:
            return True
        if self._config.max_tool_calls is not None and self._tool_calls_used >= self._config.max_tool_calls:
            return True
        return False

    def get_exhaustion_reason(self) -> str | None:
        """Return the reason for budget exhaustion, or None."""
        if self._config is None:
            return None
        if self._config.max_tokens_per_agent is not None and self._tokens_used >= self._config.max_tokens_per_agent:
            return f"Token budget exhausted: {self._tokens_used}/{self._config.max_tokens_per_agent} tokens used."
        if self._config.max_cost_usd is not None and self._cost_used >= self._config.max_cost_usd:
            return f"Cost budget exhausted: ${self._cost_used:.4f}/${self._config.max_cost_usd:.4f} USD used."
        if self._config.max_turns is not None and self._turns_used >= self._config.max_turns:
            return f"Turn limit reached: {self._turns_used}/{self._config.max_turns} turns used."
        if self._config.max_tool_calls is not None and self._tool_calls_used >= self._config.max_tool_calls:
            return f"Tool call limit reached: {self._tool_calls_used}/{self._config.max_tool_calls} tool calls used."
        return None

    def is_tool_blocked(self, tool_name: str) -> bool:
        if self._config is None:
            return False
        if self._config.blocked_tools and tool_name in self._config.blocked_tools:
            return True
        return False

    def get_status(self) -> BudgetStatus:
        return BudgetStatus(
            tokens_used=self._tokens_used,
            tokens_limit=self._config.max_tokens_per_agent if self._config else None,
            cost_used=self._cost_used,
            cost_limit=self._config.max_cost_usd if self._config else None,
            turns_used=self._turns_used,
            turns_limit=self._config.max_turns if self._config else None,
            tool_calls_used=self._tool_calls_used,
            tool_calls_limit=self._config.max_tool_calls if self._config else None,
            exhausted=self.is_exhausted(),
        )
