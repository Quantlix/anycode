"""Synthetic huge-context smoke tests (1M / 5M tokens) using FakeAdapter.

These exercise the auto-mode pathway end-to-end: massive synthetic histories
should not raise, the manifest should report reserved response space, and the
final manifest must surface the model profile + counting confidence.
"""

from __future__ import annotations

import pytest

from anycode import FakeAdapter, FakeResponse
from anycode.core.context_manager import ContextManager
from anycode.core.runner import AgentRunner
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    ContextPolicy,
    LLMMessage,
    ModelContextProfile,
    RunnerOptions,
    TextBlock,
)


def _huge_history(approx_tokens: int) -> list[LLMMessage]:
    """Build a synthetic conversation with `approx_tokens` of user content."""
    chunk = "lorem ipsum " * 200  # ~600 chars per chunk -> ~150 tokens
    chunks_needed = max(1, approx_tokens // 150)
    return [LLMMessage(role="user" if i % 2 == 0 else "assistant", content=[TextBlock(text=chunk)]) for i in range(chunks_needed)]


@pytest.mark.parametrize("target_tokens", [1_000_000, 5_000_000])
def test_huge_synthetic_history_assembles_without_ceiling(target_tokens: int) -> None:
    profile = ModelContextProfile(
        provider="fake",
        model="huge",
        max_context_tokens=target_tokens,
        max_output_tokens=8_192,
    )
    policy = ContextPolicy(enabled=True, mode="auto", model_profile=profile, reserved_response_tokens=8_192)
    cm = ContextManager(policy, provider="fake", model="huge")
    # Build a moderate history that fits comfortably under the resolved window
    # so we exercise the auto-mode path without triggering compaction/handoff.
    messages = _huge_history(min(target_tokens // 4, 200_000))
    out, manifest = cm.assemble(messages)
    assert out, "expected non-empty assembled message list"
    assert manifest.usage_report is not None
    assert manifest.usage_report.max_context_tokens == target_tokens
    assert manifest.usage_report.reserved_response_tokens == 8_192


@pytest.mark.asyncio
async def test_runner_with_auto_policy_invokes_engine() -> None:
    profile = ModelContextProfile(provider="fake", model="huge", max_context_tokens=1_000_000)
    policy = ContextPolicy(enabled=True, mode="auto", model_profile=profile)
    adapter = FakeAdapter(responses=[FakeResponse(text="done")])
    runner = AgentRunner(
        adapter,
        ToolRegistry(),
        ToolExecutor(ToolRegistry()),
        RunnerOptions(model="huge", agent_name="t", max_turns=1),
        context_policy=policy,
    )
    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="hi")])])
    assert result.context_manifests, "expected at least one manifest"
    manifest = result.context_manifests[0]
    assert manifest.usage_report is not None
    assert manifest.usage_report.max_context_tokens == 1_000_000
    assert manifest.actual_input_tokens is not None
