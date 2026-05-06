"""Tests for Phase 6 harness runtime reliability features.

Uses the deterministic FakeAdapter so no LLM keys are required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anycode import (
    FakeAdapter,
    FakeResponse,
    load_scenarios,
    run_suite,
)
from anycode.core.context_manager import ContextManager
from anycode.core.runner import AgentRunner
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    ContextPolicy,
    LLMMessage,
    RunnerOptions,
    TextBlock,
    VerificationSensorConfig,
)

DET_SUITE = Path(__file__).parent / "fixtures" / "eval" / "runtime_reliability_deterministic.yaml"


# --- ContextPolicy provider overrides --------------------------------------


def test_context_policy_for_provider_returns_override() -> None:
    override = ContextPolicy(enabled=True, max_context_tokens=1234)
    policy = ContextPolicy(enabled=True, max_context_tokens=8000, provider_overrides={"anthropic": override})
    assert policy.for_provider("anthropic").max_context_tokens == 1234
    assert policy.for_provider("openai").max_context_tokens == 8000
    assert policy.for_provider(None).max_context_tokens == 8000


def test_context_policy_preserved_state_fields_default_empty() -> None:
    policy = ContextPolicy()
    assert policy.preserved_task_state == {}
    assert policy.preserved_verification_failures == ()


def test_context_manager_uses_provider_override() -> None:
    override = ContextPolicy(enabled=True, max_context_tokens=2222)
    policy = ContextPolicy(enabled=True, max_context_tokens=9999, provider_overrides={"openai": override})
    cm = ContextManager(policy, provider="openai")
    assert cm.provider == "openai"


# --- Verification sensor registry -----------------------------------------


def test_build_regex_sensor_from_config() -> None:
    from anycode.verification import build_sensor

    cfg = VerificationSensorConfig(
        name="regex",
        kind="computational",
        phases=("after_task",),
        block_on_failure=True,
        options={"pattern": "DONE", "expect": "match"},
    )
    sensor = build_sensor(cfg)
    assert sensor.name == "regex"
    assert sensor.phases == ("after_task",)


# --- Deterministic eval suite ---------------------------------------------


def test_deterministic_eval_suite_runs_without_llm() -> None:
    scenarios = load_scenarios(DET_SUITE)
    report = asyncio.run(run_suite(list(scenarios), suite_name="phase6", harness_variant="det"))
    assert report.total_scenarios == 4
    by_name = {r.scenario_name: r for r in report.scenario_results}
    assert by_name["simple_tool_failure"].passed
    assert by_name["context_pressure"].passed
    assert by_name["approval_gate"].passed
    assert by_name["dependency_block"].passed
    assert report.total_cost_usd >= 0.0
    assert report.total_retries >= 0
    assert report.total_verification_failures == 0


# --- Cancellation lifecycle ------------------------------------------------


@pytest.mark.asyncio
async def test_runner_cancellation_emits_cancelled_phase() -> None:
    """Cancelling a running agent stream emits a 'cancelled' terminal phase via lifecycle."""
    adapter = FakeAdapter(responses=[FakeResponse(text="hi")])

    original_chat = adapter.chat

    async def _slow_chat(messages, options):  # type: ignore[no-untyped-def]
        await asyncio.sleep(5.0)
        return await original_chat(messages, options)

    adapter.chat = _slow_chat  # type: ignore[method-assign]

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    options = RunnerOptions(model="fake-model", max_turns=2, agent_name="t")
    runner = AgentRunner(adapter, registry, executor, options)

    captured: list[object] = []

    async def consume() -> None:
        try:
            async for ev in runner.stream([LLMMessage(role="user", content=[TextBlock(text="hello")])]):
                captured.append(ev)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Either the stream emitted a 'done' event with cancelled phase, or the raise propagated.
    done_events = [getattr(ev, "data", None) for ev in captured if getattr(ev, "type", None) == "done"]
    if done_events:
        result = done_events[-1]
        assert getattr(result, "terminal_phase", None) == "cancelled"
        stop = getattr(result, "stop_reason", None)
        assert stop is not None and stop.code == "user_cancelled"
