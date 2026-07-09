"""Context engineering for huge-context models.

Demonstrates:
- Auto-mode that resolves the model's real window (no AnyCode-imposed cap)
- Custom `ModelContextProfile` registration for hypothetical 5M-token models
- Per-section budgets with explicit overflow strategies
- First-class files, memory, task-state, verification, and artifact context sections
- `render_usage_report_table` for human-readable run summaries
- `reconcile` swap to provider-actual token counting

Run with the bundled FakeAdapter — no API key needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from anycode import (
    AgentRunner,
    ContextSectionInput,
    FakeAdapter,
    FakeResponse,
    LLMMessage,
    RunnerOptions,
    TextBlock,
    ToolExecutor,
    ToolRegistry,
)
from anycode.context.reporting import render_usage_report_table
from anycode.core.context_manager import ContextManager
from anycode.types import (
    ContextPolicy,
    ContextSectionBudget,
    ModelContextProfile,
)


async def main() -> None:
    # 1. Define a hypothetical huge-context model profile (5M tokens).
    giga = ModelContextProfile(
        provider="fake",
        model="giga-5m",
        max_context_tokens=5_000_000,
        max_output_tokens=64_000,
        supports_prompt_cache=True,
        tokenizer_strategy="heuristic",
    )

    # 2. Build a policy that uses the profile in auto mode and constrains
    #    `tool_results` to 40k tokens with summarize-on-overflow.
    policy = ContextPolicy(
        enabled=True,
        mode="auto",
        reserved_response_tokens=8_192,
        custom_profiles=(giga,),
        sections={
            "tool_results": ContextSectionBudget(kind="tool_results", max_tokens=40_000, overflow="summarize"),
        },
    )

    # 3. Run a single turn through the runner.
    adapter = FakeAdapter(responses=[FakeResponse(text="Acknowledged.")])
    runner = AgentRunner(
        adapter,
        ToolRegistry(),
        ToolExecutor(ToolRegistry()),
        RunnerOptions(model="giga-5m", agent_name="huge", max_turns=1),
        context_policy=policy,
    )
    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="Summarize the manual.")])])

    manifest = result.context_manifests[0]
    usage_report = manifest.usage_report
    assert usage_report is not None, "ContextManager.assemble always emits a usage_report"
    print(f"Effective window: {usage_report.max_context_tokens:,} tokens")
    print(f"Reserved response: {usage_report.reserved_response_tokens:,}")
    print(f"Counting confidence: {usage_report.counting_confidence}")
    print(f"Actual input tokens (from provider): {manifest.actual_input_tokens}")
    print()
    print(render_usage_report_table(usage_report))

    # 4. Demonstrate first-class section inputs for non-chat context.
    sample_path = Path(__file__).parent / "config" / "huge_context_sample.txt"
    file_excerpt = sample_path.read_text(encoding="utf-8") if sample_path.exists() else ("def important_path():\n    return 'stable'\n" * 80)
    manager = ContextManager(policy, provider="fake", model="giga-5m")
    section_messages, section_manifest = manager.assemble(
        [LLMMessage(role="user", content=[TextBlock(text="Use the supplied sections to prepare the next action.")])],
        context_sections=(
            ContextSectionInput(kind="files", label=str(sample_path), content=file_excerpt),
            ContextSectionInput(
                kind="memory_rag", label="release_notes", content="Section-aware context budgets are required for huge-context runs."
            ),
            ContextSectionInput(kind="verification", label="pytest", content="All focused context tests should pass before release."),
        ),
        task_state={"phase": "context-engineering", "next": "run focused tests"},
    )
    section_usage = section_manifest.usage_report
    assert section_usage is not None, "ContextManager.assemble always emits a usage_report"
    section_kinds = ", ".join(section.kind for section in section_usage.sections)
    print()
    print(f"Section-aware assemble produced {len(section_messages)} provider messages.")
    print(f"Reported sections: {section_kinds}")


if __name__ == "__main__":
    asyncio.run(main())
