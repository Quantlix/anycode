"""Model pricing registry and cost calculation."""

from __future__ import annotations

from anycode.types import ModelPricing, TokenUsage

TOKENS_PER_PRICING_UNIT = 1000

# Default pricing table (USD per 1K tokens). Update as provider prices change.
DEFAULT_PRICING: list[ModelPricing] = [
    # Anthropic
    ModelPricing(model="claude-opus-4-5", provider="anthropic", input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    ModelPricing(model="claude-opus-4-6", provider="anthropic", input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    ModelPricing(model="claude-sonnet-4-5", provider="anthropic", input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    ModelPricing(model="claude-sonnet-4-6", provider="anthropic", input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    ModelPricing(model="claude-haiku-3-5", provider="anthropic", input_cost_per_1k=0.0008, output_cost_per_1k=0.004),
    ModelPricing(model="claude-haiku-4-5", provider="anthropic", input_cost_per_1k=0.0008, output_cost_per_1k=0.004),
    # OpenAI
    ModelPricing(model="gpt-4o", provider="openai", input_cost_per_1k=0.0025, output_cost_per_1k=0.01),
    ModelPricing(model="gpt-4o-mini", provider="openai", input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    ModelPricing(model="gpt-4.1", provider="openai", input_cost_per_1k=0.002, output_cost_per_1k=0.008),
    ModelPricing(model="gpt-4.1-mini", provider="openai", input_cost_per_1k=0.0004, output_cost_per_1k=0.0016),
    ModelPricing(model="gpt-4.1-nano", provider="openai", input_cost_per_1k=0.0001, output_cost_per_1k=0.0004),
    ModelPricing(model="o3", provider="openai", input_cost_per_1k=0.01, output_cost_per_1k=0.04),
    ModelPricing(model="o3-mini", provider="openai", input_cost_per_1k=0.0011, output_cost_per_1k=0.0044),
    # Google
    ModelPricing(model="gemini-2.5-pro", provider="google", input_cost_per_1k=0.00125, output_cost_per_1k=0.01),
    ModelPricing(model="gemini-2.5-flash", provider="google", input_cost_per_1k=0.000075, output_cost_per_1k=0.0003),
    # Local — wildcard match
    ModelPricing(model="*ollama*", provider="ollama", input_cost_per_1k=0.0, output_cost_per_1k=0.0),
]


def find_pricing(model: str, pricing: list[ModelPricing] | None = None) -> ModelPricing | None:
    """Resolve pricing for a model name. Supports `*pattern*` wildcards.

    Exact matches take precedence over wildcard matches.
    """
    table = pricing or DEFAULT_PRICING

    for entry in table:
        if entry.model == model:
            return entry

    for entry in table:
        if "*" in entry.model:
            pattern = entry.model.strip("*")
            if pattern and pattern in model:
                return entry
    return None


def calculate_cost(usage: TokenUsage, model: str, pricing: list[ModelPricing] | None = None) -> float:
    """Return total USD cost for a given token usage and model."""
    price = find_pricing(model, pricing)
    if price is None:
        return 0.0
    input_cost = (usage.input_tokens / TOKENS_PER_PRICING_UNIT) * price.input_cost_per_1k
    output_cost = (usage.output_tokens / TOKENS_PER_PRICING_UNIT) * price.output_cost_per_1k
    return input_cost + output_cost
