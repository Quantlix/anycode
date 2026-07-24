---
title: "Use Reasoning Models and Extended Thinking in AnyCode"
description: "Configure Anthropic extended thinking and OpenAI reasoning effort in AnyCode with portable tiers, token budgets, and streamed thinking events across providers."
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
| `ollama` | both knobs | `reasoning_effort` → `think` level (`minimal`→`low`); a token budget enables `think: true` | Non-thinking models return no `thinking` field |
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

!!! note "Streaming thinking: Anthropic and Ollama"
    Anthropic and Ollama emit live `thinking` stream events. Bedrock parses thinking blocks in non-streaming responses but does not emit `thinking` stream events, and the OpenAI-family adapters emit none at all. Don't build UI that requires live thinking deltas on those providers.

### Ollama: think levels

The Ollama adapter maps `reasoning_effort` to the native `think` parameter (`minimal` and `low` → `"low"`, `medium` → `"medium"`, `high` → `"high"`); a `thinking_budget_tokens` request enables `think: true` since Ollama has no budget concept. You can also set a default on the adapter itself with `OllamaAdapter(think=...)`, including the `"max"` level that has no portable tier. Thinking text returns as a `ThinkingBlock` (no signature) and as `thinking` stream events on models such as `qwen3`, `gpt-oss`, and `deepseek-r1`.

See [`examples/32_reasoning_models.py`](https://github.com/Quantlix/anycode/blob/main/examples/32_reasoning_models.py) for a runnable walkthrough, including model classification and parameter shaping.

## The complete, runnable program

The snippets above are fragments of one file. Here is the whole thing, ready to copy into `reasoning_models.py`. The first three sections need no API key — they show the two knobs, OpenAI's reasoning-model detection, and the thinking content types. The final section streams real `thinking` deltas, which is Anthropic-only today, so it runs live only when `ANTHROPIC_API_KEY` is set and otherwise degrades gracefully.

```python title="reasoning_models.py"
import asyncio
import os

from dotenv import load_dotenv

from anycode import (
    LLMMessage,
    LLMStreamOptions,
    RedactedThinkingBlock,
    RunnerOptions,
    TextBlock,
    ThinkingBlock,
    create_adapter,
)
from anycode.providers._openai_compat import apply_model_params, is_reasoning_model

load_dotenv()

# Anthropic maps each reasoning_effort tier to an extended-thinking token budget.
EFFORT_BUDGET = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}


async def main() -> None:
    # 1. The two portable reasoning knobs on RunnerOptions.
    tiered = RunnerOptions(model="claude-sonnet-4-5", agent_name="thinker", reasoning_effort="medium")
    print(f"reasoning_effort={tiered.reasoning_effort!r} -> budget {EFFORT_BUDGET[tiered.reasoning_effort]} tokens")

    explicit = RunnerOptions(model="claude-sonnet-4-5", agent_name="thinker", thinking_budget_tokens=6000)
    print(f"thinking_budget_tokens={explicit.thinking_budget_tokens} (an explicit budget wins over the tier)")

    # 2. OpenAI/Azure reasoning-model detection changes which params are sent.
    print("\nOpenAI model classification:")
    for model in ("gpt-5", "o3-mini", "gpt-4o"):
        params: dict = {}
        apply_model_params(params, model, max_tokens=1024, temperature=0.7, reasoning_effort="low")
        flavor = "reasoning" if is_reasoning_model(model) else "chat"
        print(f"  {model:<10} {flavor:<10} -> {params}")

    # 3. Reasoning text also arrives as content blocks.
    thinking = ThinkingBlock(thinking="Work through it step by step...", signature="sig-abc123")
    redacted = RedactedThinkingBlock(data="<encrypted>")
    print(f"\nThinkingBlock.signature={thinking.signature!r} (echoed back verbatim next turn)")
    print(f"RedactedThinkingBlock.type={redacted.type} (opaque, passed back as-is)")

    # 4. Stream live thinking deltas — Anthropic extended thinking only.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nSet ANTHROPIC_API_KEY to stream real thinking events.")
        return

    model = os.environ.get("ANYCODE_THINKING_MODEL", "claude-sonnet-4-5")
    adapter = await create_adapter("anthropic")
    options = LLMStreamOptions(model=model, max_tokens=4096, thinking_budget_tokens=2048)
    messages = [LLMMessage(role="user", content=[TextBlock(text="What is 17 * 24? Reason briefly, then answer.")])]

    thinking_chars = 0
    text_chars = 0
    try:
        async for event in adapter.stream(messages, options):
            if event.type == "thinking":
                thinking_chars += len(str(event.data))
            elif event.type == "text":
                text_chars += len(str(event.data))
        print(f"\nmodel={model}  thinking={thinking_chars} chars  answer={text_chars} chars")
    except Exception as exc:  # noqa: BLE001 - stay runnable if the model lacks thinking
        print(f"\nlive run failed ({type(exc).__name__}: {exc})")
        print("tip: set ANYCODE_THINKING_MODEL to a thinking-capable Claude model")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python reasoning_models.py
```

!!! tip "Tested copy"
    See [`examples/32_reasoning_models.py`](https://github.com/Quantlix/anycode/blob/main/examples/32_reasoning_models.py).

## See also

- [Stream agent output](streaming.md) — the full `StreamEvent` contract
- [Configuration reference](../reference/configuration.md) — every option on `RunnerOptions`
