---
title: "Use Reasoning Models and Extended Thinking in AnyCode"
description: "Enable extended thinking on Anthropic models and reasoning effort on OpenAI o-series/GPT-5 through AnyCode's reasoning_effort and thinking_budget_tokens options, and consume thinking output as stream events."
keywords: AnyCode reasoning models, extended thinking, reasoning_effort, thinking_budget_tokens, o3, gpt-5, claude thinking, ThinkingBlock, thinking stream events
---

# Use Reasoning Models

AnyCode exposes two portable knobs for reasoning models: `reasoning_effort` (a tier: `"minimal"`, `"low"`, `"medium"`, or `"high"`) and `thinking_budget_tokens` (a raw Anthropic-style token budget). Both live on `LLMChatOptions`, `LLMStreamOptions`, and `RunnerOptions` — not on `AgentConfig` — and each provider adapter maps them to whatever its API supports.

```python title="two ways to ask for reasoning"
from anycode import RunnerOptions

# Tier-based (portable) — Anthropic derives a token budget from the tier:
options = RunnerOptions(
    model="claude-sonnet-4-5", agent_name="thinker", reasoning_effort="medium",
)

# Explicit budget — takes precedence over the tier when both are set:
options = RunnerOptions(
    model="claude-sonnet-4-5", agent_name="thinker", thinking_budget_tokens=6000,
)
```

`reasoning_effort` is validated as a `Literal` — any other string raises a Pydantic `ValidationError`.

## What each provider does

| Provider | Uses | Mapping | If unsupported |
| --- | --- | --- | --- |
| `anthropic` | both knobs | `thinking={"type": "enabled", "budget_tokens": N}` | No-op when neither knob is set |
| `openai` | `reasoning_effort` only | `reasoning_effort` + `max_completion_tokens` on reasoning models | Non-reasoning models drop the effort and use `max_tokens` + `temperature` |
| `azure` | `reasoning_effort` only | Same as `openai` | Same as `openai` |
| `bedrock` | — | Ignored (thinking blocks in responses are still parsed) | Silently ignored |
| `ollama` | — | Ignored | Silently ignored |
| `google` | — | Ignored | Silently ignored |

### Anthropic: budget resolution

The adapter resolves a thinking budget in this order: explicit `thinking_budget_tokens` wins; otherwise the effort tier maps to a budget; otherwise thinking stays off.

| `reasoning_effort` | Budget tokens |
| --- | --- |
| `minimal` | 1,024 |
| `low` | 2,048 |
| `medium` | 8,192 |
| `high` | 16,384 |

Three side effects to know about, all covered by tests:

- Budgets below the API floor clamp **up** to 1,024 tokens — they are not rejected.
- `temperature` is removed from the request; the Anthropic API requires it unset when thinking is enabled.
- If `max_tokens` is not comfortably above the budget, it is bumped to `budget + 4096` so the answer isn't squeezed out by the thinking allowance.

### OpenAI and Azure: reasoning-model detection

Models are classified by prefix: `o1`, `o3`, `o4`, and `gpt-5` count as reasoning models (provider prefixes like `openai/o3` are tolerated). For those, the adapter sends `max_completion_tokens` and `reasoning_effort`, and omits `max_tokens` and `temperature` — reasoning models reject a non-default temperature. `gpt-4o` and `gpt-4.1` are not reasoning models and keep the standard parameters.

## Read the thinking output

Reasoning text comes back in two ways:

**As content blocks** on `LLMResponse.content`:

- `ThinkingBlock` — `thinking: str` plus a `signature` that must be echoed back verbatim on the next turn, or a thinking-enabled tool-use turn is rejected. AnyCode handles the echo for you, and thinking blocks survive checkpoint serialization round-trips.
- `RedactedThinkingBlock` — opaque `data`, passed back as-is.

**As stream events** with `type="thinking"`, distinct from `"text"`:

```python title="stream_thinking.py"
from anycode import LLMStreamOptions
from anycode.providers import create_adapter

adapter = await create_adapter("anthropic")
options = LLMStreamOptions(
    model="claude-sonnet-4-5", max_tokens=4096, thinking_budget_tokens=2048,
)
async for event in adapter.stream(messages, options):
    if event.type == "thinking":
        ...  # reasoning text delta
    elif event.type == "text":
        ...  # answer text delta
```

!!! note "Streaming thinking is Anthropic-only today"
    Bedrock parses thinking blocks in non-streaming responses but does not emit `thinking` stream events, and the OpenAI-family adapters emit none at all. Don't build UI that requires live thinking deltas on those providers.

See [`examples/32_reasoning_models.py`](https://github.com/Quantlix/anycode/blob/main/examples/32_reasoning_models.py) for a runnable walkthrough, including model classification and parameter shaping.

## See also

- [Stream agent output](streaming.md) — the full `StreamEvent` contract
- [Configuration reference](../reference/configuration.md) — every option on `RunnerOptions`
