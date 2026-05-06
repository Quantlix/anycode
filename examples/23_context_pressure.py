"""Phase 6 — context lifecycle with provider overrides and preserved state.

Demonstrates ``ContextPolicy.provider_overrides``, ``preserved_task_state``,
and ``preserved_verification_failures``. Builds two ContextManagers — one for
each provider — and shows that compaction emits the preserved sections.

Run::

    uv run python examples/23_context_pressure.py
"""

from __future__ import annotations

from anycode.core.context_manager import ContextManager
from anycode.types import ContextPolicy, LLMMessage, TextBlock


def make_messages(n: int) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="user" if i % 2 == 0 else "assistant",
            content=[TextBlock(text=f"turn {i}: " + ("lorem ipsum " * 80))],
        )
        for i in range(n)
    ]


def main() -> None:
    base = ContextPolicy(
        enabled=True,
        max_context_tokens=4_000,
        compact_ratio=0.5,
        keep_recent_messages=2,
        preserved_task_state={"current_step": "compacting", "objective": "summarise"},
        preserved_verification_failures=("regex sensor failed: missing DONE marker",),
        provider_overrides={
            "anthropic": ContextPolicy(
                enabled=True,
                max_context_tokens=2_000,
                compact_ratio=0.5,
                keep_recent_messages=2,
                preserved_task_state={"current_step": "compacting", "objective": "summarise"},
                preserved_verification_failures=("regex sensor failed: missing DONE marker",),
            ),
        },
    )

    for provider in ("openai", "anthropic"):
        cm = ContextManager(base, provider=provider)
        prepared, manifest = cm.assemble(make_messages(20))
        print(f"\n=== provider={provider} ===")
        print(f"  pressure:        {manifest.pressure}")
        print(f"  estimated_tokens:{manifest.estimated_tokens} / {manifest.max_tokens}")
        print(f"  preserved_state: {manifest.preserved_task_state}")
        print(f"  open_failures:   {manifest.preserved_verification_failures}")
        print(f"  prepared msgs:   {len(prepared)}")


if __name__ == "__main__":
    main()
