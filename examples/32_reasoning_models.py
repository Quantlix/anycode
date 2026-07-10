# Demo 32 — Reasoning Models
# Execute: uv run python examples/32_reasoning_models.py
#
# Demonstrates the phase-11 reasoning-model controls:
#   1. ReasoningEffort levels and the two knobs on RunnerOptions/LLMChatOptions:
#      reasoning_effort (OpenAI-style tiers) and thinking_budget_tokens (a raw
#      Anthropic extended-thinking budget)
#   2. How OpenAI classifies reasoning models (o1/o3/o4/gpt-5) vs chat models,
#      which changes whether reasoning_effort / max_completion_tokens apply
#   3. ThinkingBlock / RedactedThinkingBlock content types, and how the runner
#      surfaces incremental `thinking` events distinct from `text`
#
# Sections A-C need no API key. Section D runs live against Anthropic extended
# thinking only when ANTHROPIC_API_KEY is set, and degrades gracefully otherwise.

import asyncio
import os

from dotenv import load_dotenv

from anycode.providers._openai_compat import apply_model_params, is_reasoning_model
from anycode.types import (
    LLMMessage,
    LLMStreamOptions,
    RedactedThinkingBlock,
    RunnerOptions,
    TextBlock,
    ThinkingBlock,
)

load_dotenv()

SEPARATOR = "-" * 60

# Anthropic maps each reasoning_effort tier to an extended-thinking token budget
# (floor 1024). Shown here for reference; the provider applies it internally.
EFFORT_BUDGET = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}


async def main() -> None:
    print("=== Reasoning Models Demo ===\n")

    # --- Section A: the two reasoning knobs ---
    print(SEPARATOR)
    print("Section A: reasoning_effort & thinking_budget_tokens\n")

    # Tier-based: portable across providers; Anthropic derives a token budget.
    tiered = RunnerOptions(model="claude-sonnet-4-5", agent_name="thinker", reasoning_effort="medium")
    print(f"  RunnerOptions.reasoning_effort = {tiered.reasoning_effort!r}")
    print(f"    -> Anthropic thinking budget  = {EFFORT_BUDGET[tiered.reasoning_effort]} tokens")

    # Explicit budget: overrides the tier mapping when you want precise control.
    explicit = RunnerOptions(model="claude-sonnet-4-5", agent_name="thinker", thinking_budget_tokens=6000)
    print(f"  RunnerOptions.thinking_budget_tokens = {explicit.thinking_budget_tokens}")
    print("  (an explicit budget takes precedence over the effort tier)")

    # --- Section B: OpenAI reasoning-model classification ---
    print(f"\n{SEPARATOR}")
    print("Section B: OpenAI reasoning vs chat model classification\n")

    for model in ("gpt-5", "o3-mini", "o4-mini", "gpt-4o-mini", "gpt-4o"):
        reasoning = is_reasoning_model(model)
        kwargs: dict = {}
        apply_model_params(kwargs, model, max_tokens=1024, temperature=0.7, reasoning_effort="low")
        flavor = "reasoning" if reasoning else "chat"
        print(f"  {model:<12} {flavor:<10} -> {kwargs}")
    print("\n  note: reasoning models take max_completion_tokens + reasoning_effort")
    print("  and reject a custom temperature; chat models take max_tokens + temperature.")

    # --- Section C (types): thinking content blocks ---
    print(f"\n{SEPARATOR}")
    print("Section C: thinking content blocks\n")

    thinking = ThinkingBlock(thinking="Let me work through this step by step...", signature="sig-abc123")
    redacted = RedactedThinkingBlock(data="<encrypted>")
    print(f"  ThinkingBlock.type          = {thinking.type}")
    print(f"  ThinkingBlock.signature     = {thinking.signature!r} (must be echoed back verbatim)")
    print(f"  RedactedThinkingBlock.type  = {redacted.type} (opaque, passed back as-is)")

    # --- Section D (live): stream real thinking events ---
    print(f"\n{SEPARATOR}")
    print("Section D: live extended thinking (Anthropic)\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  skipped: set ANTHROPIC_API_KEY to stream real thinking events")
        print(f"\n{SEPARATOR}\nDone.")
        return

    from anycode.providers.adapter import create_adapter

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
        print(f"  model: {model}")
        print(f"  thinking stream: {thinking_chars} chars")
        print(f"  answer stream:   {text_chars} chars")
    except Exception as e:  # noqa: BLE001 - example stays runnable if the model lacks thinking
        print(f"  live run failed (model may not support extended thinking): {type(e).__name__}: {e}")
        print("  tip: set ANYCODE_THINKING_MODEL to a thinking-capable Claude model")

    print(f"\n{SEPARATOR}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
